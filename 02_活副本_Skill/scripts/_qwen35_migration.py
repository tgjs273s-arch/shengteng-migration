"""Pinned target identity and the single auditable 0.8B converter patch."""

import hashlib
from pathlib import Path
import subprocess


GPU_COMMIT = "fc9137225880a9d03f130634c20f9dbe36a7b8bf"
TARGET_COMMIT = "6c45b4869f9938892b982a203cc121803c345db2"
CONVERTER = Path("checkpoint/vlm_model/converters/qwen3_5.py")
BASE_CONVERTER_SHA256 = "d10f742ba9a4281dbaa40ef22ab4085a3ea0cacb4e52ec7f83de476999ca91cb"
MODEL_SHA256 = "40049ef6476d1e2e178e3633f8e46eacd5e2ca621d851bc1dae3339712c4ef89"
MODELHUB_SHA256 = "8a2a17190c81ca200ad17c4759c878fd0c272a5f930636f5f141e87203bcc676"

# The upstream conversion callback sees one shard at a time. Inspect all shard
# headers before any DCP output is opened so a missing tied source fails early.
OLD = """        def state_dict_convert_func(state_dict):
            if tie_weight_mapping:
"""
NEW = """        if tie_weight_mapping:
            import json
            shard_paths = sorted(Path(hf_dir).glob("*.safetensors"))
            if not shard_paths:
                raise ValueError("no HF safetensors shards for tied weight conversion")
            available_keys = set()
            for shard_path in shard_paths:
                with shard_path.open("rb") as handle:
                    size = int.from_bytes(handle.read(8), "little")
                    if not 0 < size <= 16 * 1024 * 1024:
                        raise ValueError(f"invalid safetensors header: {shard_path}")
                    header = json.loads(handle.read(size))
                available_keys.update(key for key in header if key != "__metadata__")
            for target_key, source_key in tie_weight_mapping.items():
                if source_key not in available_keys:
                    raise ValueError(f"missing tied source weight {source_key} for {target_key}")

        def state_dict_convert_func(state_dict):
            if tie_weight_mapping:
"""


class TargetIdentityError(ValueError):
    pass


def digest(data):
    return hashlib.sha256(data).hexdigest()


def patched_converter_bytes(original):
    if digest(original) != BASE_CONVERTER_SHA256:
        raise TargetIdentityError("target converter base hash mismatch")
    decoded = original.decode("utf-8")
    if decoded.count(OLD) != 1:
        raise TargetIdentityError("converter patch context is not unique")
    return decoded.replace(OLD, NEW, 1).encode("utf-8")


def _git(checkout, *args):
    result = subprocess.run(["git", "-C", str(checkout), *args],
                            capture_output=True, check=False)
    if result.returncode:
        raise TargetIdentityError("cannot inspect target Git checkout")
    return result.stdout


def verify_target_checkout(checkout, require_patched=True):
    """Require fixed HEAD, untouched model/registry, and only exact converter delta.

    All expectations derive from this code and the immutable Git base blob,
    never from an overlay's self-reported manifest hashes.
    """
    root = Path(checkout)
    if _git(root, "rev-parse", "HEAD").decode("ascii").strip() != TARGET_COMMIT:
        raise TargetIdentityError("wrong MindSpeed-MM HEAD")
    base = _git(root, "show", "HEAD:" + CONVERTER.as_posix())
    expected = patched_converter_bytes(base)
    converter = root / CONVERTER
    if not converter.is_file():
        raise TargetIdentityError("target converter missing")
    current = converter.read_bytes()
    if current != (expected if require_patched else base):
        raise TargetIdentityError("target converter is not the exact expected version")
    for relative, wanted in (("mindspeed_mm/fsdp/models/qwen3_5/modeling_qwen3_5.py", MODEL_SHA256),
                             ("mindspeed_mm/fsdp/models/modelhub.py", MODELHUB_SHA256)):
        path = root / relative
        if not path.is_file() or digest(path.read_bytes()) != wanted:
            raise TargetIdentityError("target model/registry drift: %s" % relative)
    dirty = _git(root, "status", "--porcelain", "--untracked-files=no").decode("utf-8").splitlines()
    allowed = CONVERTER.as_posix() if require_patched else None
    if any(line[3:].strip('"') != allowed or line[:2] != " M" for line in dirty):
        raise TargetIdentityError("target checkout has other tracked changes")
    if require_patched and len(dirty) != 1:
        raise TargetIdentityError("expected converter patch is not the sole tracked delta")
    return {"target_commit": TARGET_COMMIT, "converter_base_sha256": digest(base),
            "converter_current_sha256": digest(current), "model_sha256": MODEL_SHA256,
            "modelhub_sha256": MODELHUB_SHA256,
            "patch_identity": digest(expected), "tracked_delta": CONVERTER.as_posix() if require_patched else None}
