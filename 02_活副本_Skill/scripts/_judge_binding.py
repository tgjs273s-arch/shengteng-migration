"""T07 read-only binding of P2, P4 and P5 local evidence.

This checks current bytes and receipt pairing. It cannot prove that assets had
the same bytes during training or that they match an official reference.
"""
import hashlib
import json
from pathlib import Path
import sys
import copy

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))
SK04_SCRIPTS = Path(__file__).resolve().parents[1] / "sk04_judge" / "scripts"
if str(SK04_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SK04_SCRIPTS))
from _asset_integrity import inspect_hf, inspect_dcp, inspect_data, inspect_llava


def digest(path):
    h = hashlib.sha256()
    with open(path, "rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _json(path):
    doc = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(doc, dict):
        raise ValueError("JSON root is not an object")
    return doc


def _map(value):
    return value if isinstance(value, dict) else {}


def _same_path(a, b):
    try:
        return bool(a and b) and Path(a).resolve() == Path(b).resolve()
    except (OSError, TypeError, ValueError):
        return False


def _file_matches(path, expected):
    try:
        return bool(path and expected and Path(path).is_file() and digest(path) == expected)
    except (OSError, TypeError, ValueError):
        return False


def verify_binding(config_manifest, assets_json, train_integrity, log, baseline_id,
                   reference_log_sha, config_arg=None):
    """Return explicit mismatches; never infer official or in-run asset equality."""
    supplied = [config_manifest, assets_json, train_integrity]
    result = {"state": "NOT_PROVIDED", "issues": [], "inputs": {
        "config_manifest": config_manifest, "assets_json": assets_json,
        "train_integrity": train_integrity},
        "runtime_asset_binding": "UNVERIFIED", "official_asset_identity": "UNVERIFIED"}
    if not any(supplied):
        return result
    if not all(supplied):
        result.update(state="REJECTED", issues=["binding_inputs_incomplete"])
        return result
    issues = result["issues"]
    try:
        plan = _json(config_manifest)
        assets = _json(assets_json)
        train = _json(train_integrity)
        run_path = Path(train_integrity).resolve().parent / "run.json"
        run = _json(run_path)
    except (OSError, UnicodeError, ValueError, json.JSONDecodeError) as exc:
        result.update(state="REJECTED", issues=["binding_artifact_unreadable"], error=str(exc))
        return result
    if plan.get("schema") != "migrator_config.v1":
        issues.append("config_manifest_schema")
    if assets.get("schema") != "migrator_assets.v2":
        issues.append("assets_schema")
    if train.get("schema") != "train_integrity.v1":
        issues.append("train_integrity_schema")
    source = _map(plan.get("effective_config"))
    source_path, source_hash = source.get("path"), source.get("sha256")
    if not _file_matches(source_path, source_hash):
        issues.append("p2_source_config_changed")
    if (not _same_path(source_path, train.get("source_config")) or
            source_hash != train.get("source_config_sha256") or
            source_hash != run.get("source_config_sha256") or
            source_hash != train.get("source_config_sha256_after_run")):
        issues.append("p2_p5_source_config_mismatch")
    snapshot = train.get("config")
    snapshot_hash = train.get("config_sha256")
    if (not _file_matches(snapshot, snapshot_hash) or
            not _same_path(snapshot, run.get("config_snapshot")) or
            snapshot_hash != train.get("effective_config_sha256") or
            snapshot_hash != run.get("effective_config_sha256") or
            snapshot_hash != train.get("config_snapshot_sha256_after_run")):
        issues.append("p5_effective_config_mismatch")
    if config_arg and not _same_path(config_arg, snapshot):
        issues.append("judge_config_is_not_p5_snapshot")
    if (train.get("state") != "COMPLETE" or train.get("train_rc") != 0 or
            train.get("failure_markers") or train.get("problems")):
        issues.append("train_not_complete")
    if (not _same_path(train.get("log"), log) or not _same_path(run.get("log"), log) or
            not _file_matches(log, train.get("log_sha256"))):
        issues.append("train_log_mismatch")
    run_id = train.get("run_id")
    if (not run_id or run_id != run.get("run_id") or
            run_id != Path(train_integrity).resolve().parent.name):
        issues.append("train_run_id_mismatch")
    if train.get("scope") != "FULL_CONFIGURED_TRAINING":
        issues.append("train_scope_not_full")
    if (plan.get("baseline_id") != baseline_id or
            _map(plan.get("baseline_source")).get("sha256") != reference_log_sha):
        issues.append("plan_baseline_mismatch")
    reference = _map(plan.get("reference_config"))
    if not _file_matches(reference.get("path"), reference.get("sha256")):
        issues.append("reference_config_changed")
    if (assets.get("readiness") != "local_complete_official_unverified" or
            assets.get("missing_required") or
            assets.get("migration_identity_state") != "validated_overlay_and_target"):
        issues.append("assets_not_locally_ready")
    migration_path = assets.get("migration_manifest_path")
    migration_hash = assets.get("migration_manifest_sha256")
    if not _file_matches(migration_path, migration_hash):
        issues.append("migration_manifest_changed")
    else:
        try:
            migration = _json(migration_path)
            if (migration.get("migration_id") != assets.get("migration_id") or
                    _map(migration.get("target")).get("commit") != plan.get("target_framework_commit")):
                issues.append("migration_identity_mismatch")
        except (OSError, UnicodeError, ValueError, json.JSONDecodeError):
            issues.append("migration_manifest_invalid")
    asset_paths = _map(assets.get("assets"))
    hf = _map(assets.get("hf_identity"))
    dcp = _map(assets.get("dcp_identity"))
    hf_root = _map(asset_paths.get("weight_hf")).get("path")
    dcp_root = _map(asset_paths.get("weight_dcp")).get("path")
    coco_images = _map(asset_paths.get("coco_images")).get("path")
    try:
        current_hf = inspect_hf(hf_root) if isinstance(hf_root, str) else {}
    except Exception:
        current_hf = {}
    if (hf.get("status") != "local_complete" or
            current_hf.get("status") != "local_complete" or
            current_hf.get("file_inventory") != hf.get("file_inventory") or
            current_hf.get("file_inventory_sha256") != hf.get("file_inventory_sha256")):
        issues.append("hf_content_mismatch")
    try:
        current_dcp = inspect_dcp(dcp_root) if isinstance(dcp_root, str) else {}
    except Exception:
        current_dcp = {}
    if (dcp.get("status") != "local_structure_and_hashes" or
            current_dcp.get("status") != "local_structure_and_hashes" or
            current_dcp.get("file_inventory") != dcp.get("file_inventory") or
            current_dcp.get("file_inventory_sha256") != dcp.get("file_inventory_sha256")):
        issues.append("dcp_content_mismatch")
    llava = _map(assets.get("llava_identity"))
    llava_path = _map(asset_paths.get("llava_json")).get("path")
    try:
        current_llava = inspect_llava(llava_path) if isinstance(llava_path, str) else {}
    except Exception:
        current_llava = {}
    if (llava.get("status") != "local_content_verified" or
            current_llava.get("status") != "local_content_verified" or
            current_llava.get("json_sha256") != llava.get("json_sha256") or
            current_llava.get("order_sha256") != llava.get("order_sha256") or
            _map(assets.get("data_conversion")).get("source_json_sha256") != llava.get("json_sha256")):
        issues.append("llava_source_mismatch")
    data = _map(assets.get("data_identity"))
    converted = _map(asset_paths.get("converted_json"))
    try:
        current_data = inspect_data(converted.get("path"), Path(coco_images).parent) if (
            isinstance(converted.get("path"), str) and isinstance(coco_images, str)) else {}
    except Exception:
        current_data = {}
    if (data.get("status") != "local_content_verified" or
            current_data.get("status") != "local_content_verified" or
            any(current_data.get(key) != data.get(key) for key in (
                "json_sha256", "order_sha256", "row_count", "image_inventory",
                "referenced_images_sha256", "image_decode_verified")) or
            converted.get("sha256") != data.get("json_sha256") or
            converted.get("order_sha256") != data.get("order_sha256")):
        issues.append("data_content_or_order_mismatch")
    receipt_path = Path(dcp_root if isinstance(dcp_root, str) else "__missing_dcp__") / "migration_conversion_receipt.json"
    conversion = _map(assets.get("conversion"))
    if not _file_matches(receipt_path, conversion.get("receipt_sha256")):
        issues.append("conversion_receipt_changed")
    else:
        try:
            receipt = _json(receipt_path)
            if (receipt.get("schema") != "qwen35_0p8b_conversion.v3" or
                    receipt.get("migration_id") != assets.get("migration_id") or
                    receipt.get("migration_manifest_sha256") != migration_hash or
                    receipt.get("hf_file_inventory_sha256") != hf.get("file_inventory_sha256") or
                    receipt.get("dcp_file_inventory_sha256") != dcp.get("file_inventory_sha256") or
                    receipt.get("target_commit") != plan.get("target_framework_commit")):
                issues.append("conversion_receipt_identity_mismatch")
        except (OSError, UnicodeError, ValueError, json.JSONDecodeError):
            issues.append("conversion_receipt_invalid")
    try:
        import yaml
        config = yaml.safe_load(Path(snapshot).read_text(encoding="utf-8"))
        source_config = yaml.safe_load(Path(source_path).read_text(encoding="utf-8"))
        training = config["training"]
        source_training = source_config["training"]
        source_present = "save" in source_training
        source_value = source_training.get("save")
        enabled = bool(source_value)
        expected_save = str(Path(train_integrity).resolve().parent / "checkpoints") if enabled else None
        save_intent = _map(run.get("checkpoint_save"))
        source_without_save = copy.deepcopy(source_config)
        effective_without_save = copy.deepcopy(config)
        source_without_save["training"].pop("save", None)
        effective_without_save["training"].pop("save", None)
        if (source_without_save != effective_without_save or
                save_intent.get("source_present") != source_present or
                save_intent.get("source_value") != source_value or
                save_intent.get("enabled") != enabled or
                save_intent.get("effective_path") != expected_save or
                save_intent.get("effective_present") != enabled or
                save_intent.get("actual_format") != ("dcp" if enabled else None) or
                _map(train.get("checkpoint_save")) != save_intent or
                (enabled and training.get("save") != expected_save) or
                (not enabled and "save" in training) or
                (not source_present and digest(source_path) != digest(snapshot))):
            issues.append("p5_snapshot_source_delta_mismatch")
        data_config = config["data"]["dataset_param"]["basic_parameters"]
        preprocess = config["data"]["dataset_param"]["preprocess_parameters"]
        model = config["model"]
        load_path = training.get("load")
        dataset = data_config.get("dataset")
        dataset_dir = data_config.get("dataset_dir")
        if isinstance(dataset, list):
            dataset = dataset[0] if len(dataset) == 1 else None
        if (not Path(str(load_path)).is_absolute() or not _same_path(
                load_path, _map(asset_paths.get("weight_dcp")).get("path"))):
            issues.append("train_weight_path_mismatch")
        if (not isinstance(dataset, str) or not Path(dataset).is_absolute() or
                not _same_path(dataset, _map(asset_paths.get("converted_json")).get("path"))):
            issues.append("train_data_path_mismatch")
        if (not isinstance(dataset_dir, str) or not Path(dataset_dir).is_absolute() or
                not isinstance(coco_images, str) or
                not _same_path(dataset_dir, Path(coco_images).parent)):
            issues.append("train_image_root_mismatch")
        if (not isinstance(hf_root, str) or
                not _same_path(preprocess.get("model_name_or_path"), hf_root) or
                not _same_path(model.get("model_name_or_path"), hf_root)):
            issues.append("train_hf_path_mismatch")
    except Exception:  # Untrusted YAML, missing optional parser or malformed nested fields.
        issues.append("p5_snapshot_unreadable")
    result["state"] = "LOCAL_BINDING_VERIFIED" if not issues else "REJECTED"
    result["run_id"] = run_id
    result["config_role"] = plan.get("role")
    result["effective_config_path"] = snapshot
    result["effective_config_sha256"] = snapshot_hash
    result["asset_attempt_id"] = assets.get("attempt_id")
    result["migration_id"] = assets.get("migration_id")
    return result
