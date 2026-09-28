"""P5 local input binding. VERIFIED means matching local bytes, not runtime import or NPU proof."""

import hashlib
import importlib.util
import json
from pathlib import Path

from _asset_integrity import inspect_data, inspect_dcp, inspect_hf, inspect_llava
from _qwen35_migration import verify_target_checkout

SCHEMA = "p5_asset_binding.v1"


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _path(value, base):
    if not isinstance(value, str) or not value:
        raise ValueError("missing configured asset path")
    p = Path(value)
    return str((p if p.is_absolute() else Path(base) / p).resolve())


def _recorded_path(value):
    if not isinstance(value, str) or not Path(value).is_absolute():
        raise ValueError("P4 asset path must be absolute for runtime binding")
    return str(Path(value).resolve())


def _one_dataset(value):
    """P2 writes one JSON path in a list; accept legacy scalar, reject ambiguous sets."""
    if isinstance(value, list):
        if len(value) != 1 or not isinstance(value[0], str):
            raise ValueError("dataset must contain exactly one path")
        return value[0]
    if not isinstance(value, str):
        raise ValueError("dataset must be one path or a one-item list")
    return value


def _p4_module():
    source = Path(__file__).with_name("40_prepare_assets.py")
    spec = importlib.util.spec_from_file_location("p4_for_p5_binding", source)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def capture(assets_json, bundle, overlay, target_checkout, effective_config):
    """Reinspect all P4 input bytes and T03 target; return a bounded identity record."""
    import yaml
    p4 = _p4_module()
    assets_path = Path(assets_json).resolve()
    assets = json.loads(assets_path.read_text(encoding="utf-8"))
    if assets.get("schema") != "migrator_assets.v2":
        raise ValueError("unsupported P4 assets schema")
    migration = p4.verified_migration(bundle, overlay, target_checkout)
    target = verify_target_checkout(target_checkout, require_patched=True)
    if (not assets.get("attempt_id") or
            Path(assets.get("migration_manifest_path", "")).resolve() !=
            Path(migration["manifest_path"]).resolve() or
            assets.get("migration_id") != migration["migration_id"] or
            assets.get("migration_manifest_sha256") != migration["manifest_sha256"] or
            assets.get("migration_identity_state") != "validated_overlay_and_target" or
            assets.get("readiness") != "local_complete_official_unverified" or
            assets.get("missing_required") or
            any(not item.get("ok") for item in assets.get("checks", []))):
        raise ValueError("P4 assets/migration identity or readiness mismatch")
    paths = {key: _recorded_path(assets["assets"][key]["path"])
             for key in ("weight_hf", "weight_dcp", "llava_json", "converted_json", "coco_images")}
    config = yaml.safe_load(effective_config)
    if not isinstance(config, dict):
        raise ValueError("effective configuration is not a mapping")
    params = config["data"]["dataset_param"]
    basic = params["basic_parameters"]
    configured = {
        "weight_hf": params["preprocess_parameters"]["model_name_or_path"],
        "weight_dcp": config["training"]["load"],
        "converted_json": _one_dataset(basic["dataset"]),
        "coco_images": str(Path(basic["dataset_dir"]) / "train2017"),
    }
    if _path(config["model"]["model_name_or_path"], target_checkout) != paths["weight_hf"]:
        raise ValueError("effective model path differs from P4 HF")
    for key, value in configured.items():
        if _path(value, target_checkout) != paths[key]:
            raise ValueError("effective configuration uses different %s" % key)
    hf = inspect_hf(paths["weight_hf"])
    dcp = inspect_dcp(paths["weight_dcp"])
    llava = inspect_llava(paths["llava_json"])
    data = inspect_data(paths["converted_json"], str(Path(paths["coco_images"]).parent))
    expected = assets
    checks = (
        (hf, expected.get("hf_identity"), "local_complete",
         ("file_inventory_sha256", "config_sha256", "index_sha256", "model_revision")),
        (dcp, expected.get("dcp_identity"), "local_structure_and_hashes",
         ("file_inventory_sha256",)),
        (llava, expected.get("llava_identity"), "local_content_verified",
         ("json_sha256", "order_sha256", "row_count")),
        (data, expected.get("data_identity"), "local_content_verified",
         ("json_sha256", "order_sha256", "row_count", "referenced_images_sha256")),
    )
    for observed, declared, required, fields in checks:
        if observed.get("status") != required or not isinstance(declared, dict):
            raise ValueError("local asset completeness changed: %s" % required)
        if any(not observed.get(field) or observed.get(field) != declared.get(field)
               for field in fields):
            raise ValueError("local asset content differs from P4 receipt")
    receipt_path = Path(paths["weight_dcp"]) / "migration_conversion_receipt.json"
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    if receipt != p4._conversion_receipt(paths["weight_dcp"], hf, target, migration):
        raise ValueError("DCP conversion receipt differs from current inputs")
    receipt_sha = sha256(receipt_path)
    if expected.get("conversion", {}).get("receipt_sha256") != receipt_sha:
        raise ValueError("DCP conversion receipt hash differs from P4")
    return {
        "assets_json_path": str(assets_path), "assets_json_sha256": sha256(assets_path),
        "p4_attempt_id": assets.get("attempt_id"),
        "migration_id": migration["migration_id"],
        "migration_bundle": str(Path(bundle).resolve()),
        "migration_overlay": str(Path(overlay).resolve()),
        "migration_manifest_path": migration["manifest_path"],
        "migration_manifest_sha256": migration["manifest_sha256"],
        "target_checkout": str(Path(target_checkout).resolve()),
        "target_commit": target["target_commit"],
        "target_source_files_sha256": target["target_files_sha256"],
        "conversion_receipt_path": str(receipt_path.resolve()),
        "conversion_receipt_sha256": receipt_sha,
        "paths": paths,
        "hf_file_inventory_sha256": hf["file_inventory_sha256"],
        "dcp_file_inventory_sha256": dcp["file_inventory_sha256"],
        "llava_json_sha256": llava["json_sha256"],
        "llava_order_sha256": llava["order_sha256"],
        "data_json_sha256": data["json_sha256"],
        "data_order_sha256": data["order_sha256"],
        "referenced_images_sha256": data["referenced_images_sha256"],
    }


def new_binding():
    return {"schema": SCHEMA, "state": "UNBOUND", "prelaunch": None,
            "postrun": None, "problems": [],
            "scope": "local_disk_identity_only; runtime_import_and_npu_unverified"}
