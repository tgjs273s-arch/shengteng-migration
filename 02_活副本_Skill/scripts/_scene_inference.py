"""Local artifact and scene-output contracts for one inference attempt."""

import hashlib
import json
import math
from pathlib import Path


class SceneError(ValueError):
    pass


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_hash(value):
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True,
        separators=(",", ":")).encode("utf-8")).hexdigest()


def file_inventory(root):
    root = Path(root).resolve(strict=True)
    if not root.is_dir():
        raise SceneError("export path is not a directory")
    rows = []
    for path in sorted(root.rglob("*")):
        if path.is_symlink():
            raise SceneError("symlink in exported model: %s" % path)
        if path.is_file():
            if path.stat().st_size == 0:
                raise SceneError("empty exported file: %s" % path)
            rows.append({"path": path.relative_to(root).as_posix(),
                         "bytes": path.stat().st_size, "sha256": sha256(path)})
    if not rows:
        raise SceneError("exported model directory is empty")
    return {"files": rows, "sha256": canonical_hash(rows)}


def _check_evidence(root, entry):
    if not isinstance(entry, dict) or not entry.get("path") or not entry.get("sha256"):
        raise SceneError("round-trip evidence reference missing")
    path = Path(entry["path"]).resolve(strict=True)
    if not path.is_file() or not path.is_relative_to(root) or sha256(path) != entry["sha256"]:
        raise SceneError("round-trip evidence changed: %s" % path)


def verify_artifact(manifest_path, model_dir, processor_dir):
    """Recheck the *whole* export and T08's identity, not just a claimed subset."""
    path = Path(manifest_path).resolve(strict=True)
    manifest = json.loads(path.read_text(encoding="utf-8"))
    if (manifest.get("schema") != "model_artifact.v1" or
            manifest.get("status") != "RELOAD_VERIFIED" or
            manifest.get("reload_verified") is not True or
            manifest.get("post_export_identity_rechecked") is not True or
            manifest.get("problems") != []):
        raise SceneError("model artifact is not verified for inference")
    exported = manifest.get("export_hf") or {}
    root = Path(exported.get("path") or "").resolve(strict=True)
    if not root.is_relative_to(path.parent):
        raise SceneError("export path is outside its T08 attempt")
    if Path(model_dir).resolve() != root or Path(processor_dir).resolve() != root:
        raise SceneError("model and processor must use the verified local export")
    observed = file_inventory(root)
    if observed != exported.get("inventory"):
        raise SceneError("actual exported files differ from model artifact inventory")
    if not {"config.json", "model.safetensors.index.json", "preprocessor_config.json",
            "tokenizer_config.json", "tokenizer.json"}.issubset(
                {row["path"] for row in observed["files"]}):
        raise SceneError("model or processor files missing")
    train = manifest.get("train_run") or {}
    checkpoint = manifest.get("checkpoint") or {}
    migration = manifest.get("migration_id")
    if (not migration or not train.get("run_id") or
            not train.get("run_json_sha256") or
            not train.get("train_integrity_sha256") or
            checkpoint.get("format") != "dcp" or
            not (checkpoint.get("inventory") or {}).get("sha256") or
            not manifest.get("model_artifact_id")):
        raise SceneError("migration, training or checkpoint identity missing")
    identity = {"migration_id": migration, "train_run_id": train["run_id"],
                "checkpoint_inventory_sha256": checkpoint["inventory"]["sha256"],
                "export_inventory_sha256": observed["sha256"]}
    artifact_id = "qwen35-0p8b-" + canonical_hash(identity)[:20]
    if artifact_id != manifest["model_artifact_id"]:
        raise SceneError("model artifact ID does not bind observed export")
    for name in ("assets_json", "migration_manifest"):
        ref = manifest.get(name) or {}
        if not ref.get("path") or not ref.get("sha256") or \
                sha256(ref["path"]) != ref["sha256"]:
            raise SceneError("source reference changed: %s" % name)
    checkpoint_path = Path(checkpoint.get("path") or "").resolve(strict=True)
    run_dir = checkpoint_path.parent.parent
    if run_dir.name != train["run_id"] or checkpoint_path.parent.name != "checkpoints":
        raise SceneError("checkpoint path differs from bound training run")
    for name, digest in (("run.json", train["run_json_sha256"]),
                         ("train_integrity.json", train["train_integrity_sha256"])):
        if sha256(run_dir / name) != digest:
            raise SceneError("training receipt changed: %s" % name)
    claimed_checkpoint = checkpoint["inventory"]
    rows = claimed_checkpoint.get("files")
    if not isinstance(rows, list) or not rows:
        raise SceneError("checkpoint file inventory missing")
    checkpoint_root = checkpoint_path.parent
    present = file_inventory(checkpoint_root)
    selected_rows = [row for row in present["files"] if
                     row["path"] == "latest_checkpointed_iteration.txt" or
                     row["path"].startswith(checkpoint_path.name + "/")]
    if selected_rows != rows or canonical_hash(selected_rows) != claimed_checkpoint["sha256"]:
        raise SceneError("selected checkpoint files changed")
    comparison = manifest.get("round_trip") or {}
    if (comparison.get("status") != "verified" or
            comparison.get("checkpoint_inventory_sha256") != checkpoint["inventory"]["sha256"] or
            comparison.get("device") != "cpu"):
        raise SceneError("real round-trip comparison evidence missing")
    evidence_root = (path.parent / "round_trip").resolve(strict=True)
    stored = json.loads((evidence_root / "comparison.json").read_text(encoding="utf-8"))
    if stored != comparison:
        raise SceneError("round-trip comparison changed")
    for name in ("dcp_logits", "export_logits"):
        _check_evidence(evidence_root, comparison.get(name))
    input_path = evidence_root / "input.json"
    if sha256(input_path) != comparison.get("input_sha256"):
        raise SceneError("round-trip input changed")
    return manifest, observed, sha256(path)


def parse_scene_json(raw, width, height):
    try:
        value = json.loads(raw)
    except (ValueError, TypeError) as exc:
        raise SceneError("generation is not JSON: %s" % exc) from exc
    if not isinstance(value, dict):
        raise SceneError("generation JSON must be an object")
    required = {"compliant", "step", "issue", "evidence", "bbox", "confidence"}
    if set(value) != required:
        raise SceneError("generation JSON fields differ from scene contract")
    if value["compliant"] not in ("是", "否", "无法判断") or not isinstance(value["compliant"], str):
        raise SceneError("compliant must be 是, 否 or 无法判断")
    for key in ("step", "issue", "evidence"):
        if not isinstance(value[key], str) or not value[key].strip():
            raise SceneError("%s must be nonempty text" % key)
    bbox = value["bbox"]
    if value["compliant"] == "无法判断":
        if bbox is not None:
            raise SceneError("bbox must be null when evidence is insufficient")
    else:
        _check_bbox(bbox, width, height)
    confidence = value["confidence"]
    if (type(confidence) not in (int, float) or not math.isfinite(confidence) or
            not 0 <= confidence <= 1):
        raise SceneError("confidence must be finite and within [0,1]")
    return value


def _check_bbox(bbox, width, height):
    if (not isinstance(bbox, list) or len(bbox) != 4 or
            any(type(v) not in (int, float) or not math.isfinite(v) for v in bbox)):
        raise SceneError("bbox must contain four finite coordinates")
    x1, y1, x2, y2 = bbox
    if not (0 <= x1 < x2 <= width and 0 <= y1 < y2 <= height):
        raise SceneError("bbox outside image or inverted")
