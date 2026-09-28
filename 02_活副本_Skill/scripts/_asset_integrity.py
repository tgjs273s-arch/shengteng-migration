"""Read-only local asset inventories. Hashes identify observed bytes, not official bytes."""

import hashlib
import json
from pathlib import Path
import sys

from _qwen35_weights import WeightContractError, metadata_contract, sha256, weight_headers_contract

SK04_SCRIPTS = Path(__file__).resolve().parents[1] / "sk04_judge" / "scripts"
sys.path.insert(0, str(SK04_SCRIPTS))
from data_id import identity_of_file  # noqa: E402

REQUIRED_HF_FILES = ("config.json", "model.safetensors.index.json",
                     "preprocessor_config.json", "tokenizer_config.json", "tokenizer.json")
OPTIONAL_HF_FILES = ("processor_config.json", "special_tokens_map.json",
                     "generation_config.json", "chat_template.jinja")
CONVERTER_COMMIT = "5b5505331924634da64e3d9a1925d02b10babe9f"
CONVERTER_SHA256 = "2901488ed98ab162487200ff28f7bc8367d72304a9c57fc878ce0b2c3cdc2e95"
CONVERTER_RELATIVE = ("mindspeed_mm/fsdp/tools/data_tool/"
                      "llava_instruct_2_mllm_demo_format.py")
CONVERTER_ARGS = ("--llava_json_path", "--coco_path", "--output_json_path")


def _digest_rows(rows):
    raw = json.dumps(rows, ensure_ascii=False, sort_keys=True,
                     separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def inspect_hf(root):
    """Pinned metadata + every indexed header + local full-file identities."""
    root = Path(root).resolve()
    missing = [name for name in REQUIRED_HF_FILES if not (root / name).is_file()]
    if missing:
        return {"status": "incomplete", "missing_files": missing,
                "validation_level": "missing_required_files"}
    try:
        tokenizer = json.loads((root / "tokenizer.json").read_text(encoding="utf-8"))
        if not isinstance(tokenizer, dict) or not tokenizer:
            raise WeightContractError("invalid tokenizer.json")
        metadata, _, _ = metadata_contract(root)
        missing_shards = [name for name in metadata["shards"] if not (root / name).is_file()]
        if missing_shards:
            return {"status": "incomplete", "missing_files": missing_shards,
                    "validation_level": "indexed_shards_missing"}
        headers = weight_headers_contract(root)
        files = sorted(set(REQUIRED_HF_FILES) | set(headers["shards"]) |
                       {name for name in OPTIONAL_HF_FILES if (root / name).is_file()})
        inventory = [{"path": name, "bytes": (root / name).stat().st_size,
                      "sha256": sha256(root / name)} for name in files]
    except (OSError, UnicodeError, json.JSONDecodeError, WeightContractError) as exc:
        return {"status": "incomplete", "missing_files": [], "error": str(exc),
                "validation_level": "headers_or_files_failed"}
    return {"status": "local_complete", "missing_files": [],
            "validation_level": "headers_and_local_file_hashes",
            "metadata_revision": headers["model_revision"],
            "model_revision": headers["model_revision"],
            "config_sha256": headers["config_sha256"],
            "index_sha256": headers["index_sha256"],
            "upstream_payload_revision": None, "upstream_payload_identity": "unverified",
            "header_keys": headers["header_keys"], "indexed_keys": headers["indexed_keys"],
            "mtp_source_keys": headers["mtp_source_keys"],
            "file_inventory": inventory, "file_inventory_sha256": _digest_rows(inventory),
            "payload_hashes_checked": True, "model_state_dict_checked": False,
            "reload_verified": False}


def inspect_dcp(root):
    """Require tracker, real PyTorch DCP metadata references and payload hashes."""
    root = Path(root).resolve()
    tracker = root / "latest_checkpointed_iteration.txt"
    metadata = root / "release" / ".metadata"
    missing = []
    if not tracker.is_file():
        missing.append("latest_checkpointed_iteration.txt")
    try:
        if not metadata.is_file() or metadata.stat().st_size == 0:
            missing.append("release/.metadata(nonempty)")
    except OSError:
        missing.append("release/.metadata(readable)")
    if missing:
        return {"status": "incomplete", "missing_files": missing}
    try:
        if tracker.read_text(encoding="utf-8").strip() != "release":
            return {"status": "incomplete", "missing_files": [], "error": "tracker is not release"}
        try:
            storage_data = _read_dcp_storage_data(root / "release")
        except DCPReaderUnavailable as exc:
            return {"status": "metadata_unverified", "missing_files": [],
                    "error": "PyTorch DCP metadata reader unavailable: %s" % exc,
                    "metadata_parsed": False, "reload_verified": False}
        try:
            referenced = validate_dcp_storage_ranges(root / "release", storage_data)
        except (ValueError, OSError) as exc:
            return {"status": "incomplete", "missing_files": [],
                    "error": "invalid DCP storage metadata: %s" % exc,
                    "metadata_parsed": False, "reload_verified": False}
        payloads = sorted(path for path in (root / "release").rglob("*.distcp")
                          if path.is_file())
        if not payloads or not set(referenced).issubset(set(payloads)):
            return {"status": "incomplete", "missing_files": ["referenced release/*.distcp"]}
        files = [tracker, metadata, *payloads]
        inventory = [{"path": path.relative_to(root).as_posix(), "bytes": path.stat().st_size,
                      "sha256": sha256(path)} for path in files]
    except Exception as exc:
        return {"status": "incomplete", "missing_files": [], "error": str(exc)}
    if any(item["bytes"] == 0 for item in inventory):
        return {"status": "incomplete", "missing_files": [], "error": "empty DCP payload"}
    return {"status": "local_structure_and_hashes", "missing_files": [],
            "validation_level": "release_payload_file_hashes_no_reload",
            "file_inventory": inventory, "file_inventory_sha256": _digest_rows(inventory),
            "payload_count": len(payloads), "metadata_parsed": True,
            "storage_entry_count": len(storage_data),
            "reload_verified": False}


class DCPReaderUnavailable(Exception):
    """The optional PyTorch DCP reader could not be imported."""


def _read_dcp_storage_data(release_dir):
    """Use PyTorch's own metadata reader; never parse its pickle by hand."""
    try:
        from torch.distributed.checkpoint import FileSystemReader
    except (ImportError, OSError) as exc:
        raise DCPReaderUnavailable(str(exc)) from exc
    metadata = FileSystemReader(str(release_dir)).read_metadata()
    return metadata.storage_data


def validate_dcp_storage_ranges(release_dir, storage_data):
    """Validate all storage_data references against actual payload byte spans."""
    release_dir = Path(release_dir).resolve()
    if not isinstance(storage_data, dict) or not storage_data:
        raise ValueError("storage_data missing or empty")
    referenced = set()
    for info in storage_data.values():
        name = getattr(info, "relative_path", None)
        offset = getattr(info, "offset", None)
        length = getattr(info, "length", None)
        if not isinstance(name, str) or not name or Path(name).is_absolute() or \
                ".." in Path(name).parts or Path(name).suffix != ".distcp":
            raise ValueError("unsafe DCP storage path: %s" % name)
        if type(offset) is not int or type(length) is not int or offset < 0 or length < 1:
            raise ValueError("invalid DCP storage offset/length: %s" % name)
        path = (release_dir / name).resolve()
        if not path.is_relative_to(release_dir) or not path.is_file():
            raise ValueError("missing or escaping DCP payload: %s" % name)
        if offset + length > path.stat().st_size:
            raise ValueError("truncated DCP payload: %s" % name)
        referenced.add(path)
    return referenced


def _decode_image(path):
    """Use the actual image decoder when available; absence stays unverified."""
    try:
        from PIL import Image
    except ImportError:
        return {"status": "unverified", "reason": "Pillow unavailable"}
    try:
        with Image.open(path) as image:
            image.load()
            return {"status": "verified", "format": image.format,
                    "size": list(image.size)}
    except Exception as exc:
        return {"status": "invalid", "reason": "%s: %s" % (type(exc).__name__, exc)}


def inspect_data(path, coco_dir):
    """Validate converter's list/images/messages schema and every referenced image."""
    path, coco_dir = Path(path), Path(coco_dir)
    identity = identity_of_file(str(path))
    if identity.get("status") != "ok":
        return {"status": "incomplete", "error": identity.get("note", "JSON unavailable")}
    try:
        rows = json.loads(path.read_text(encoding="utf-8-sig"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        return {"status": "incomplete", "error": str(exc)}
    if not isinstance(rows, list):
        return {"status": "incomplete", "error": "converted JSON must be a list"}
    if not rows:
        return {"status": "incomplete", "error": "converted JSON has no samples"}
    image_paths = set()
    for index, row in enumerate(rows):
        if not isinstance(row, dict) or not isinstance(row.get("images"), list) or \
                not isinstance(row.get("messages"), list) or not row["messages"]:
            return {"status": "incomplete", "error": "invalid row schema at %d" % index}
        for msg in row["messages"]:
            if (not isinstance(msg, dict) or msg.get("role") not in ("user", "assistant")
                    or not isinstance(msg.get("content"), str)):
                return {"status": "incomplete", "error": "invalid messages at %d" % index}
        for image in row["images"]:
            if not isinstance(image, str) or not image:
                return {"status": "incomplete", "error": "invalid image reference at %d" % index}
            relative = Path(image)
            if (relative.is_absolute() or len(relative.parts) != 2 or
                    relative.parts[0] != "train2017" or relative.parts[1] in (".", "..") or
                    relative.suffix.lower() not in (".jpg", ".jpeg", ".png")):
                return {"status": "incomplete", "error": "unsafe image reference at %d: %s" % (index, image)}
            image_paths.add(relative.as_posix())
    images = []
    decode_unverified = []
    for image in sorted(image_paths):
        file = coco_dir / image
        try:
            if not file.resolve().is_relative_to(coco_dir.resolve()):
                return {"status": "incomplete", "error": "image escapes COCO root: %s" % image}
            if not file.is_file() or file.stat().st_size == 0:
                return {"status": "incomplete", "error": "missing/empty image: %s" % image}
            image_row = {"path": image, "bytes": file.stat().st_size, "sha256": sha256(file)}
            decoded = _decode_image(file)
            image_row["decode"] = decoded
            if decoded["status"] == "invalid":
                return {"status": "incomplete", "error": "invalid image %s: %s" %
                        (image, decoded["reason"]), "image_inventory": images + [image_row]}
            if decoded["status"] != "verified":
                decode_unverified.append(image)
            images.append(image_row)
        except OSError as exc:
            return {"status": "incomplete", "error": "unreadable image %s: %s" % (image, exc)}
    return {"status": "decode_unverified" if decode_unverified else "local_content_verified",
            "json_sha256": identity["json_sha256"],
            "order_sha256": identity["order_sha256"], "row_count": identity["row_count"],
            "referenced_image_count": len(images), "image_inventory": images,
            "referenced_images_sha256": _digest_rows(images),
            "image_bytes_readable": True, "image_decode_verified": not decode_unverified,
            "image_decode_unverified": decode_unverified,
            "official_json_sha256": None, "official_same_bytes": "unverified",
            "conversion_source_mapping_verified": False}


def inspect_llava(path):
    """Identify the local source JSON and validate fields consumed by converter."""
    identity = identity_of_file(str(path))
    if identity.get("status") != "ok":
        return {"status": "incomplete", "error": identity.get("note", "source JSON unavailable")}
    try:
        rows = json.loads(Path(path).read_text(encoding="utf-8-sig"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        return {"status": "incomplete", "error": str(exc)}
    if not isinstance(rows, list):
        return {"status": "incomplete", "error": "LLaVA source must be a list"}
    if not rows:
        return {"status": "incomplete", "error": "LLaVA source has no samples"}
    for index, row in enumerate(rows):
        if not isinstance(row, dict) or not isinstance(row.get("conversations"), list) or \
                not row["conversations"]:
            return {"status": "incomplete", "error": "invalid LLaVA row at %d" % index}
        if row.get("image") is not None and not isinstance(row["image"], str):
            return {"status": "incomplete", "error": "invalid LLaVA image at %d" % index}
        for turn in row["conversations"]:
            if (not isinstance(turn, dict) or turn.get("from") not in ("human", "gpt")
                    or not isinstance(turn.get("value"), str)):
                return {"status": "incomplete", "error": "invalid LLaVA conversation at %d" % index}
    return {"status": "local_content_verified", "json_sha256": identity["json_sha256"],
            "order_sha256": identity["order_sha256"], "row_count": identity["row_count"],
            "official_sha256": None, "official_same_bytes": "unverified"}


def converter_identity(msmm_dir):
    path = Path(msmm_dir) / CONVERTER_RELATIVE
    if not path.is_file():
        return {"status": "unverified", "path": str(path), "error": "converter script missing"}
    observed = sha256(path)
    return {"status": "pinned" if observed == CONVERTER_SHA256 else "mismatch",
            "path": str(path), "sha256": observed, "expected_sha256": CONVERTER_SHA256,
            "target_commit": CONVERTER_COMMIT, "parameters": list(CONVERTER_ARGS)}
