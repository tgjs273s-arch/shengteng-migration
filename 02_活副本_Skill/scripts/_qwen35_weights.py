"""Read-only Qwen3.5-0.8B HF checkpoint checks shared by migration and P4.

Only safetensors headers are read. This module never loads tensor payloads.
"""

import hashlib
import json
import math
from pathlib import Path


MODEL_REVISION = "2fc06364715b967f1860aea9cf38778875588b17"
CONFIG_SHA256 = "b90b86f35c8e6925ef74ee04d0e758f0a845c83a42089ad82bbaa948de9b4204"
INDEX_SHA256 = "d8a08838a613b025eb7952ed9db11696213e57e76a375661ef5c12f9dd5dcf4e"
PREPROCESSOR_SHA256 = "27225450ac9c6529872ee1924fcb0962ff5634834f817040f444118116f4e516"
TOKENIZER_CONFIG_SHA256 = "49e2b6e395f959f077f1e992b338919c0d4a9732fc6e613995e06557f843500c"
EMBED_KEY = "model.language_model.embed_tokens.weight"
HEAD_KEY = "lm_head.weight"
TIE_MAPPING = {HEAD_KEY: EMBED_KEY}
HEADER_LIMIT = 16 * 1024 * 1024
DTYPE_BYTES = {"BF16": 2, "F16": 2, "F32": 4}


class WeightContractError(ValueError):
    pass


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _json_file(path, expected_hash):
    path = Path(path)
    if not path.is_file() or sha256(path) != expected_hash:
        raise WeightContractError("missing or wrong revision: %s" % path)
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise WeightContractError("invalid JSON: %s" % path) from exc


def metadata_contract(hf_dir):
    """Validate pinned metadata and key names, without claiming tensor presence."""
    root = Path(hf_dir)
    config = _json_file(root / "config.json", CONFIG_SHA256)
    index = _json_file(root / "model.safetensors.index.json", INDEX_SHA256)
    _json_file(root / "preprocessor_config.json", PREPROCESSOR_SHA256)
    _json_file(root / "tokenizer_config.json", TOKENIZER_CONFIG_SHA256)
    text = config.get("text_config") or {}
    if (config.get("model_type") != "qwen3_5" or
            config.get("architectures") != ["Qwen3_5ForConditionalGeneration"] or
            config.get("tie_word_embeddings") is not True or
            text.get("num_hidden_layers") != 24 or
            text.get("hidden_size") != 1024 or
            text.get("mtp_num_hidden_layers") != 1 or
            text.get("vocab_size") != 248320 or
            "linear_attention" not in text.get("layer_types", []) or
            "full_attention" not in text.get("layer_types", [])):
        raise WeightContractError("0.8B config architecture differs from pinned revision")
    weight_map = index.get("weight_map")
    if not isinstance(weight_map, dict) or len(weight_map) != 488:
        raise WeightContractError("invalid 0.8B weight index")
    if EMBED_KEY not in weight_map or HEAD_KEY in weight_map:
        raise WeightContractError("unexpected tied embedding/head keys")
    mtp_keys = sorted(key for key in weight_map if key.startswith("mtp."))
    if len(mtp_keys) != 15 or any(".mlp.experts." in key for key in mtp_keys):
        raise WeightContractError("unexpected 0.8B MTP index keys")
    shards = set(weight_map.values())
    if not shards or any(not isinstance(name, str) or Path(name).name != name or
                         not name.endswith(".safetensors") for name in shards):
        raise WeightContractError("invalid shard names in weight index")
    return {"model_revision": MODEL_REVISION, "config_sha256": CONFIG_SHA256,
            "index_sha256": INDEX_SHA256, "indexed_keys": len(weight_map),
            "shards": sorted(shards), "tie_mapping": TIE_MAPPING.copy(),
            "mtp_keys": len(mtp_keys), "mtp_source_keys": mtp_keys,
            "mtp_configured_source_layers": 1,
            "validation_level": "metadata_only"}, config, weight_map


def _header(path):
    with Path(path).open("rb") as handle:
        size_bytes = handle.read(8)
        if len(size_bytes) != 8:
            raise WeightContractError("short safetensors header: %s" % path)
        size = int.from_bytes(size_bytes, "little")
        if not 0 < size <= HEADER_LIMIT:
            raise WeightContractError("invalid safetensors header size: %s" % path)
        raw = handle.read(size)
        if len(raw) != size:
            raise WeightContractError("truncated safetensors header: %s" % path)
    try:
        header = json.loads(raw)
    except (UnicodeError, json.JSONDecodeError) as exc:
        raise WeightContractError("invalid safetensors header JSON: %s" % path) from exc
    if not isinstance(header, dict):
        raise WeightContractError("invalid safetensors tensor list: %s" % path)
    header.pop("__metadata__", None)
    payload_bytes = Path(path).stat().st_size - 8 - size
    intervals = []
    for key, tensor in header.items():
        offsets = tensor.get("data_offsets") if isinstance(tensor, dict) else None
        shape = tensor.get("shape") if isinstance(tensor, dict) else None
        dtype = tensor.get("dtype") if isinstance(tensor, dict) else None
        if (not isinstance(offsets, list) or len(offsets) != 2 or
                not all(isinstance(n, int) and n >= 0 for n in offsets) or
                not isinstance(shape, list) or
                not all(isinstance(n, int) and n > 0 for n in shape) or
                dtype not in DTYPE_BYTES or
                offsets[1] > payload_bytes or
                offsets[1] - offsets[0] != math.prod(shape) * DTYPE_BYTES[dtype]):
            raise WeightContractError("tensor payload truncated or offsets invalid: %s" % key)
        intervals.append((offsets[0], offsets[1], key))
    intervals.sort()
    for left, right in zip(intervals, intervals[1:]):
        if left[1] > right[0]:
            raise WeightContractError("overlapping tensor offsets: %s, %s" % (left[2], right[2]))
    return header


def weight_headers_contract(hf_dir, include_inventory=False):
    """Check every indexed tensor header, critical shapes/dtypes, and tied head.

    This proves header coverage, not tensor payload integrity or model loading.
    """
    report, config, weight_map = metadata_contract(hf_dir)
    root = Path(hf_dir)
    actual = {}
    for shard in report["shards"]:
        path = root / shard
        if not path.is_file():
            raise WeightContractError("missing weight shard: %s" % path)
        header = _header(path)
        for key, tensor in header.items():
            if key in actual or weight_map.get(key) != shard:
                raise WeightContractError("unindexed or duplicate tensor: %s" % key)
            if not isinstance(tensor, dict):
                raise WeightContractError("invalid tensor metadata: %s" % key)
            shape, dtype, offsets = tensor.get("shape"), tensor.get("dtype"), tensor.get("data_offsets")
            if (not isinstance(shape, list) or not all(isinstance(n, int) and n > 0 for n in shape) or
                    dtype not in {"BF16", "F16", "F32"} or
                    not isinstance(offsets, list) or len(offsets) != 2 or
                    not all(isinstance(n, int) and n >= 0 for n in offsets) or offsets[1] <= offsets[0]):
                raise WeightContractError("invalid shape/dtype/offsets: %s" % key)
            actual[key] = {"shape": shape, "dtype": dtype}
    missing = set(weight_map) - set(actual)
    if missing:
        raise WeightContractError("missing indexed tensors: %s" % ", ".join(sorted(missing)[:8]))
    text, vision = config["text_config"], config["vision_config"]
    critical = {
        EMBED_KEY: [text["vocab_size"], text["hidden_size"]],
        "model.visual.patch_embed.proj.weight": [vision["hidden_size"], vision["in_channels"],
                                                 vision["temporal_patch_size"], vision["patch_size"],
                                                 vision["patch_size"]],
        "model.language_model.layers.0.linear_attn.conv1d.weight": [
            2 * text["linear_num_key_heads"] * text["linear_key_head_dim"] +
            text["linear_num_value_heads"] * text["linear_value_head_dim"], 1,
            text["linear_conv_kernel_dim"]],
    }
    for key, shape in critical.items():
        if key not in actual or actual[key]["shape"] != shape:
            raise WeightContractError("critical tensor shape mismatch: %s" % key)
        if actual[key]["dtype"] != "BF16":
            raise WeightContractError("critical tensor dtype mismatch: %s" % key)
    report.update({"validation_level": "all_weight_headers_checked",
                   "header_keys": len(actual), "critical_shapes": critical,
                   "payload_hashes_checked": False, "model_state_dict_checked": False})
    if include_inventory:
        report["tensor_headers"] = actual
    return report
