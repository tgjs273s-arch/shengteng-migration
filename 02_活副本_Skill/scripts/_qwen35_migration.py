"""Pinned target identity and the single auditable 0.8B converter patch."""

import hashlib
from pathlib import Path
import subprocess


GPU_COMMIT = "fc9137225880a9d03f130634c20f9dbe36a7b8bf"
TARGET_COMMIT = "5b5505331924634da64e3d9a1925d02b10babe9f"
CONVERTER = Path("checkpoint/vlm_model/converters/qwen3_5.py")
BASE_CONVERTER_SHA256 = "846406988585f09bb5c0e06e72cd34bf778c17a4b938c4d55652fda6669cd0f1"
TARGET_FILES_SHA256 = {
    "mindspeed_mm/fsdp/models/qwen3_5/modeling_qwen3_5.py": "180712ccf3815f7a171a9bddf184117c59a4e56573d2368f527f656bb67381d5",
    "mindspeed_mm/fsdp/models/modelhub.py": "f316fca993ae0c924a1629e3b2ade96ebd4688c6626bc9112d4e7098a807d7fe",
    "mindspeed_mm/fsdp/utils/register.py": "92e5656f32cdfcbab9764b5f538f5e5dfccac845ffa4d2c778479468778cfdfd",
    "mindspeed_mm/fsdp/models/mtp.py": "b02fffc547f82681687cda887ae958914b582f6739e2973f8f5b82d64dcbb6f4",
    "mindspeed_mm/fsdp/models/qwen3_5/causal_conv1d.py": "11c87733af26644d12d7680787ad8e288e9a06c83f649577280706cc83c492d1",
    "mindspeed_mm/fsdp/params/model_args.py": "443fcb4271cf74c9675a637b798b762119a75a0b4d1e9cbddb55e7c5abbcb998",
    "mindspeed_mm/fsdp/params/feature_args.py": "3dcde395caaf7a3ffa4f2f02787d753130d450d0d55f89a5bdf18dffbf73832a",
    "examples/qwen3_5/qwen3_5_4B_config.yaml": "dbe58e4f8d37b40ebedf20dad4b3211735f65a920303e82e9daab95a292d5831",
}
MODEL_SHA256 = TARGET_FILES_SHA256["mindspeed_mm/fsdp/models/qwen3_5/modeling_qwen3_5.py"]
MODELHUB_SHA256 = TARGET_FILES_SHA256["mindspeed_mm/fsdp/models/modelhub.py"]

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
    for relative, wanted in TARGET_FILES_SHA256.items():
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
            "modelhub_sha256": MODELHUB_SHA256, "target_files_sha256": TARGET_FILES_SHA256.copy(),
            "patch_identity": digest(expected), "tracked_delta": CONVERTER.as_posix() if require_patched else None}
