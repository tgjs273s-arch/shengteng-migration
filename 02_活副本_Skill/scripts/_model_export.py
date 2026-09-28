"""Identity and tensor checks for one Qwen3.5-0.8B training export."""

import hashlib
import json
from pathlib import Path
import shutil

from _asset_integrity import _digest_rows, inspect_hf, validate_dcp_storage_ranges
from _qwen35_weights import _header, weight_headers_contract


class ExportError(ValueError):
    pass


def file_hash(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def read_json(path):
    with Path(path).open(encoding="utf-8") as stream:
        return json.load(stream)


def inventory(root, paths):
    root = Path(root).resolve()
    rows = []
    for name in sorted(set(paths)):
        path = (root / name).resolve()
        if not path.is_relative_to(root) or not path.is_file() or path.stat().st_size == 0:
            raise ExportError("missing, empty or escaping file: %s" % name)
        rows.append({"path": name, "bytes": path.stat().st_size, "sha256": file_hash(path)})
    return {"files": rows, "sha256": _digest_rows(rows)}


def training_receipt(run_dir):
    run_dir = Path(run_dir).resolve()
    run = read_json(run_dir / "run.json")
    check = read_json(run_dir / "train_integrity.json")
    rid = run_dir.name
    save = run.get("checkpoint_save")
    if (not rid.startswith("p5-") or run.get("run_id") != rid or
            check.get("run_id") != rid or check.get("state") != "COMPLETE" or
            check.get("train_rc") != 0 or check.get("failure_markers") or
            check.get("problems") or
            not isinstance(save, dict) or not save.get("enabled") or
            save.get("actual_format") != "dcp" or
            check.get("checkpoint_save") != save):
        raise ExportError("training result, run identity or DCP save intent invalid")
    save_path = Path(save.get("effective_path") or "").resolve()
    if save_path != run_dir / "checkpoints":
        raise ExportError("checkpoint save path not isolated to this run")
    snapshot = run_dir / "effective_config.yaml"
    if (Path(run.get("config_snapshot") or "").resolve() != snapshot or
            Path(check.get("config") or "").resolve() != snapshot or
            Path(run.get("log") or "").resolve() !=
            Path(check.get("log") or "").resolve()):
        raise ExportError("run/check paths do not bind this snapshot and log")
    for key, path in (("source_config_sha256", check.get("source_config")),
                      ("effective_config_sha256", snapshot)):
        expected = run.get(key)
        if not path or not expected or file_hash(path) != expected:
            raise ExportError("training config changed: %s" % key)
    if (check.get("source_config_sha256") != run["source_config_sha256"] or
            check.get("config_sha256") != run["effective_config_sha256"] or
            check.get("config_snapshot_sha256_after_run") != run["effective_config_sha256"] or
            check.get("source_config_sha256_after_run") != run["source_config_sha256"] or
            check.get("config_train_iters") != check.get("expected_end")):
        raise ExportError("run and train_integrity config hashes disagree")
    binding = run.get("asset_binding")
    check_binding = check.get("asset_binding")
    run_sha = file_hash(run_dir / "run.json")
    if (not isinstance(binding, dict) or
            binding.get("schema") != "p5_asset_binding.v1" or
            binding.get("state") != "VERIFIED" or binding.get("problems") != [] or
            not isinstance(binding.get("prelaunch"), dict) or
            not binding["prelaunch"] or binding.get("postrun") != binding["prelaunch"] or
            not isinstance(check_binding, dict) or
            check_binding.get("schema") != "p5_asset_binding.v1" or
            check_binding.get("state") != "VERIFIED" or
            check_binding.get("problems") != [] or
            Path(check_binding.get("run_record") or "").resolve() != run_dir / "run.json" or
            check_binding.get("run_record_sha256") != run_sha or
            check_binding.get("assets_json_sha256") !=
                binding["prelaunch"].get("assets_json_sha256") or
            check_binding.get("migration_id") != binding["prelaunch"].get("migration_id")):
        raise ExportError("P5 asset binding missing, unverified or from another run")
    log_path = check.get("log")
    if not log_path or file_hash(log_path) != check.get("log_sha256"):
        raise ExportError("training log missing or changed")
    return {"run_id": rid, "run_json_sha256": run_sha,
            "train_integrity_sha256": file_hash(run_dir / "train_integrity.json"),
            "config_snapshot_mtime_ns": snapshot.stat().st_mtime_ns,
            "integrity_mtime_ns": (run_dir / "train_integrity.json").stat().st_mtime_ns,
            "source_config_sha256": run["source_config_sha256"],
            "effective_config_sha256": run["effective_config_sha256"],
            "log_sha256": check["log_sha256"], "save_path": str(save_path),
            "expected_end": check.get("expected_end"), "start": check.get("expected_start"),
            "unique_steps": check.get("unique_steps"),
            "world_size": check.get("world_size"),
            "asset_binding": binding["prelaunch"]}


def selected_checkpoint(save_path, iteration, train):
    if type(iteration) is not int or iteration < 1:
        raise ExportError("positive iteration required")
    if (type(train["expected_end"]) is not int or iteration > train["expected_end"] or
            type(train["start"]) is not int or iteration < train["start"]):
        raise ExportError("checkpoint iteration outside this completed training range")
    if not isinstance(train.get("unique_steps"), list) or iteration not in train["unique_steps"]:
        raise ExportError("checkpoint iteration absent from this run's training log")
    root = Path(save_path).resolve()
    tracker = root / "latest_checkpointed_iteration.txt"
    if not tracker.is_file() or tracker.read_text(encoding="utf-8").strip() != str(iteration):
        raise ExportError("numeric tracker does not select requested iteration")
    selected = root / ("iter_%07d" % iteration)
    if not selected.is_dir() or not (selected / ".metadata").is_file():
        raise ExportError("selected DCP checkpoint missing metadata")
    try:
        from torch.distributed.checkpoint import FileSystemReader
        metadata = FileSystemReader(str(selected)).read_metadata()
        referenced = validate_dcp_storage_ranges(selected, metadata.storage_data)
    except (ImportError, OSError, ValueError, AttributeError) as exc:
        raise ExportError("selected DCP metadata unreadable or payload invalid: %s" % exc) from exc
    payloads = sorted(p for p in selected.rglob("*.distcp") if p.is_file())
    if not payloads or not referenced.issubset(set(payloads)):
        raise ExportError("selected DCP payload inventory incomplete")
    world = train.get("world_size")
    if type(world) is not int or world < 1:
        raise ExportError("training world size not established")
    for rank in range(world):
        if not (selected / "extra_state" / ("extra_state_rank_%d.pt" % rank)).is_file():
            raise ExportError("selected DCP rank extra_state missing: %d" % rank)
    all_files = sorted(p for p in selected.rglob("*") if p.is_file())
    for file in [tracker, *all_files]:
        if (file.stat().st_mtime_ns < train["config_snapshot_mtime_ns"] or
                file.stat().st_mtime_ns > train["integrity_mtime_ns"]):
            raise ExportError("checkpoint file timestamp outside this P5 run: %s" % file)
    paths = ["latest_checkpointed_iteration.txt"]
    paths += [p.relative_to(root).as_posix() for p in all_files]
    return selected, metadata, inventory(root, paths)


def source_assets(assets_path):
    assets = read_json(assets_path)
    if (assets.get("schema") != "migrator_assets.v2" or
            assets.get("readiness") != "local_complete_official_unverified" or
            assets.get("migration_identity_state") != "validated_overlay_and_target" or
            not assets.get("migration_id")):
        raise ExportError("P4 assets not locally complete with validated migration")
    source = assets.get("hf_identity") or {}
    origin = Path((assets.get("assets") or {}).get("weight_hf", {}).get("path") or "").resolve()
    if source.get("status") != "local_complete" or not source.get("file_inventory"):
        raise ExportError("P4 HF identity missing")
    observed = inspect_hf(origin)
    if observed.get("status") != "local_complete" or \
            observed.get("file_inventory") != source["file_inventory"] or \
            observed.get("file_inventory_sha256") != source["file_inventory_sha256"]:
        raise ExportError("P4 HF files changed since asset receipt")
    current = {"files": observed["file_inventory"],
               "sha256": observed["file_inventory_sha256"]}
    report = weight_headers_contract(origin, include_inventory=True)
    if (report["config_sha256"] != source["config_sha256"] or
            report["index_sha256"] != source["index_sha256"]):
        raise ExportError("origin HF config/index identity differs")
    return assets, origin, report, current


def expected_model_keys(report, metadata):
    source = report["tensor_headers"]
    mtp = sorted(k for k in source if k.startswith("mtp."))
    if len(source) != 488 or len(mtp) != 15:
        raise ExportError("unexpected original 0.8B key/MTP structure")
    kept = {k: v for k, v in source.items() if not k.startswith("mtp.")}
    states = metadata.state_dict_metadata
    if not isinstance(states, dict):
        raise ExportError("DCP state_dict_metadata missing")
    actual = {k for k in states if k.startswith("model.")}
    expected = {"model." + k for k in kept}
    # The HF index omits the tied lm_head, while DCP may store its alias.
    extra = actual - expected - {"model.lm_head.weight"}
    missing = expected - actual
    if missing or extra or any("mtp." in k for k in actual):
        raise ExportError("DCP model key mismatch: missing=%s extra=%s" %
                          (sorted(missing)[:8], sorted(extra)[:8]))
    dtypes = {}
    head = states.get("model.lm_head.weight")
    if head is not None:
        embedding = source["model.language_model.embed_tokens.weight"]
        if list(getattr(head, "size", [])) != embedding["shape"]:
            raise ExportError("tied lm_head DCP shape differs from embedding")
        head_dtype = str(getattr(getattr(head, "properties", None), "dtype", ""))
        embedding_dtype = str(getattr(getattr(states["model.model.language_model.embed_tokens.weight"],
                                            "properties", None), "dtype", ""))
        if head_dtype != embedding_dtype:
            raise ExportError("tied lm_head DCP dtype differs from embedding")
    for key, origin_tensor in kept.items():
        tensor = states["model." + key]
        shape = list(getattr(tensor, "size", []))
        prop = getattr(tensor, "properties", None)
        dtype = str(getattr(prop, "dtype", "")).replace("torch.", "")
        if shape != origin_tensor["shape"] or dtype not in ("bfloat16", "float16", "float32"):
            raise ExportError("DCP shape/dtype mismatch: %s" % key)
        dtypes[key] = {"shape": shape, "dtype": {"bfloat16": "BF16", "float16": "F16",
                                                   "float32": "F32"}[dtype]}
    return kept, dtypes


def derived_origin(origin, index, attempt, retained, verified_files):
    """The fixed converter selects by origin index; remove source-only MTP0 keys."""
    derived = attempt / "derived_origin"
    derived.mkdir()
    permitted = {row["path"] for row in verified_files}
    for name in sorted(permitted):
        if name.endswith(".safetensors") or name == "model.safetensors.index.json":
            continue
        source = origin / name
        if (Path(name).name != name or not source.is_file() or source.is_symlink()):
            raise ExportError("unverified or unsafe processor asset: %s" % name)
        shutil.copy2(source, derived / name)
    config = read_json(derived / "config.json")
    config["text_config"]["mtp_num_hidden_layers"] = 0
    config["text_config"]["mtp_num_layers"] = 0
    (derived / "config.json").write_text(json.dumps(config, ensure_ascii=False, indent=2) + "\n",
                                         encoding="utf-8")
    filtered = dict(index)
    filtered["weight_map"] = {k: v for k, v in index["weight_map"].items()
                              if not k.startswith("mtp.")}
    # The original total_size includes 15 source-only MTP tensors.
    if isinstance(filtered.get("metadata"), dict):
        size_per_dtype = {"BF16": 2, "F16": 2, "F32": 4}
        total = 0
        for tensor in retained.values():
            count = 1
            for extent in tensor["shape"]:
                count *= extent
            total += count * size_per_dtype[tensor["dtype"]]
        filtered["metadata"] = dict(filtered["metadata"], total_size=total)
    (derived / "model.safetensors.index.json").write_text(
        json.dumps(filtered, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return derived, inventory(derived, [p.name for p in derived.iterdir() if p.is_file()])


def inspect_export(export_dir, expected, dtype_expected):
    export_dir = Path(export_dir).resolve()
    required = ("config.json", "model.safetensors.index.json", "preprocessor_config.json",
                "tokenizer_config.json", "tokenizer.json")
    for name in required:
        if not (export_dir / name).is_file():
            raise ExportError("export missing processor/config file: %s" % name)
        if name.endswith(".json"):
            try:
                value = read_json(export_dir / name)
            except (OSError, UnicodeError, json.JSONDecodeError) as exc:
                raise ExportError("export processor/config JSON invalid: %s" % name) from exc
            if not isinstance(value, dict) or not value:
                raise ExportError("export processor/config JSON empty: %s" % name)
    config = read_json(export_dir / "config.json")
    if (config.get("text_config", {}).get("mtp_num_hidden_layers") != 0 or
            config.get("text_config", {}).get("mtp_num_layers") != 0):
        raise ExportError("export config still declares original MTP layer")
    index = read_json(export_dir / "model.safetensors.index.json")
    weights = index.get("weight_map")
    if not isinstance(weights, dict) or set(weights) != set(expected):
        raise ExportError("export index key mismatch")
    listed_shards = set(weights.values())
    present_shards = {p.name for p in export_dir.glob("*.safetensors") if p.is_file()}
    if present_shards != listed_shards:
        raise ExportError("export shard set differs from index")
    actual = {}
    for name in sorted(set(weights.values())):
        if Path(name).name != name or not name.endswith(".safetensors"):
            raise ExportError("unsafe export shard name")
        path = export_dir / name
        if not path.is_file():
            raise ExportError("export shard missing: %s" % name)
        for key, tensor in _header(path).items():
            if key in actual or weights.get(key) != name:
                raise ExportError("export extra, duplicated or misrouted tensor: %s" % key)
            actual[key] = {"shape": tensor["shape"], "dtype": tensor["dtype"]}
    if set(actual) != set(weights):
        raise ExportError("export indexed tensor missing")
    for key, value in actual.items():
        if value["shape"] != expected[key]["shape"] or value["dtype"] != dtype_expected[key]:
            raise ExportError("export tensor shape/dtype mismatch: %s" % key)
    bytes_per_dtype = {"BF16": 2, "F16": 2, "F32": 4}
    total_size = 0
    for value in actual.values():
        count = 1
        for extent in value["shape"]:
            count *= extent
        total_size += count * bytes_per_dtype[value["dtype"]]
    if index.get("metadata", {}).get("total_size") != total_size:
        raise ExportError("export index total_size differs from actual tensor dtype/shape")
    files = [p.relative_to(export_dir).as_posix() for p in export_dir.iterdir() if p.is_file()]
    return {"inventory": inventory(export_dir, files), "key_count": len(actual),
            "storage_dtypes": {dtype: sum(v["dtype"] == dtype for v in actual.values())
                               for dtype in ("BF16", "F16", "F32")},
            "config_sha256": file_hash(export_dir / "config.json"),
            "index_sha256": file_hash(export_dir / "model.safetensors.index.json"),
            "processor_files": [name for name in required if name not in
                                ("config.json", "model.safetensors.index.json")],
            "tensor_headers_sha256": hashlib.sha256(json.dumps(actual, sort_keys=True,
                separators=(",", ":")).encode()).hexdigest()}
