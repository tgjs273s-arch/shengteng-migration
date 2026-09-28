#!/usr/bin/env python3
"""Export one isolated P5 DCP checkpoint to a checked HF model artifact.

The command never downloads assets or runs training. A successful conversion
is deliberately distinct from a verified real-model reload.
"""

import argparse
import gc
import hashlib
import importlib
import inspect
import json
import math
from pathlib import Path
import subprocess
import sys
import uuid

from _model_export import (ExportError, derived_origin, expected_model_keys, file_hash,
                           inspect_export, inventory, read_json, selected_checkpoint, source_assets,
                           training_receipt)
from _qwen35_migration import verify_target_checkout


def unwrapped_source_path(function):
    """Attribute a decorated target API to its original implementation."""
    return Path(inspect.getfile(inspect.unwrap(function))).resolve()


def verify_round_trip(dcp_dir, exported, derived, target, dcp_inventory, tensors,
                      attempt, prompt, atol, rtol):
    """Load selected DCP state through the fixed target, then compare real logits."""
    import torch
    from transformers import AutoConfig, AutoModelForImageTextToText, AutoProcessor

    target = Path(target).resolve()
    sys.path.insert(0, str(target))
    try:
        module = importlib.import_module("checkpoint.common.merge_dcp_to_hf")
        loader_path = unwrapped_source_path(module.load_dcp_state_dict)
        if (Path(module.__file__).resolve() != target / "checkpoint/common/merge_dcp_to_hf.py" or
                loader_path != target / "checkpoint/common/merge_dcp_to_hf.py" or
                inspect.unwrap(module.load_dcp_state_dict).__module__ != module.__name__):
            raise ExportError("DCP loader imported outside fixed target checkout")
        model_module = importlib.import_module(
            "mindspeed_mm.fsdp.models.qwen3_5.modeling_qwen3_5")
        modelhub_module = importlib.import_module("mindspeed_mm.fsdp.models.modelhub")
        from mindspeed_mm.fsdp.utils.register import model_register
        from mindspeed_mm.fsdp.params.model_args import ModelArguments
        from mindspeed_mm.fsdp.params.feature_args import FeatureArguments
        paths = {
            "modeling": Path(model_module.__file__).resolve(),
            "modelhub": Path(modelhub_module.__file__).resolve(),
            "register": Path(sys.modules[model_register.__class__.__module__].__file__).resolve(),
            "model_args": Path(sys.modules[ModelArguments.__module__].__file__).resolve(),
            "feature_args": Path(sys.modules[FeatureArguments.__module__].__file__).resolve()}
        required_paths = {
            "modeling": "mindspeed_mm/fsdp/models/qwen3_5/modeling_qwen3_5.py",
            "modelhub": "mindspeed_mm/fsdp/models/modelhub.py",
            "register": "mindspeed_mm/fsdp/utils/register.py",
            "model_args": "mindspeed_mm/fsdp/params/model_args.py",
            "feature_args": "mindspeed_mm/fsdp/params/feature_args.py"}
        for name, path in paths.items():
            if path != target / required_paths[name]:
                raise ExportError("target %s module imported outside checkout" % name)
        model_cls = model_register.get("qwen3_5")
        if (not isinstance(model_cls, type) or
                model_cls.__name__ != "Qwen3_5ForConditionalGeneration" or
                model_cls.__module__ != model_module.__name__):
            raise ExportError("registered target Qwen3.5 class mismatch")
        state = module.load_dcp_state_dict(dcp_dir)
    finally:
        sys.path.remove(str(target))
    if not isinstance(state, dict):
        raise ExportError("target DCP loader did not return a model state dict")
    expected = set(tensors)
    allowed_head = "lm_head.weight"
    if set(state) - expected - {allowed_head} or expected - set(state):
        raise ExportError("loaded DCP tensor keys differ from metadata contract")
    for key, expected_tensor in tensors.items():
        value = state[key]
        dtype = str(value.dtype).replace("torch.", "")
        if (list(value.shape) != expected_tensor["shape"] or
                dtype != {"BF16": "bfloat16", "F16": "float16",
                          "F32": "float32"}[expected_tensor["dtype"]]):
            raise ExportError("loaded DCP tensor shape/dtype differs: %s" % key)
    if allowed_head in state and not torch.equal(
            state[allowed_head], state["model.language_model.embed_tokens.weight"]):
        raise ExportError("DCP tied lm_head values differ from embedding")
    if allowed_head not in state:
        state[allowed_head] = state["model.language_model.embed_tokens.weight"]
    processor = AutoProcessor.from_pretrained(str(exported), trust_remote_code=True,
                                              local_files_only=True)
    inputs = processor(text=prompt, return_tensors="pt")
    evidence_dir = attempt / "round_trip"
    evidence_dir.mkdir()
    (evidence_dir / "input.json").write_text(json.dumps(
        {"text": prompt, "processor_tensors": {name: {"dtype": str(value.dtype),
            "shape": list(value.shape), "values": value.tolist()}
            for name, value in inputs.items()}}, ensure_ascii=False,
        indent=2) + "\n", encoding="utf-8")
    config = AutoConfig.from_pretrained(str(derived), trust_remote_code=True,
                                        local_files_only=True)
    model_args = ModelArguments(model_id="qwen3_5", mtp_num_layers=0,
                                mtp_loss_scaling_factor=0.1, gdn_implementation="eager",
                                causal_conv1d_implementation="eager", skip_gdn_recompute=False,
                                skip_flash_attn_recompute=False)
    feature_args = FeatureArguments(enable_chunk_loss=False,
                                    enable_dynamic_chunk_loss=False)
    config = model_cls.overwrite_transformer_config(config, model_args, feature_args)
    if config.text_config.mtp_num_layers != 0:
        raise ExportError("target DCP model unexpectedly enables MTP")
    config._attn_implementation = "eager"
    config.text_config._attn_implementation = "eager"
    config.vision_config._attn_implementation = "eager"
    dcp_model = model_cls._from_config(config)
    dcp_model_class = type(dcp_model).__module__ + "." + type(dcp_model).__qualname__
    dcp_model_file = Path(inspect.getfile(type(dcp_model))).resolve()
    if dcp_model_file != paths["modeling"]:
        raise ExportError("constructed DCP model class differs from fixed target")
    load_result = dcp_model.load_state_dict(state, strict=True)
    if load_result.missing_keys or load_result.unexpected_keys:
        raise ExportError("DCP model initialization has missing/unexpected keys: %s" %
                          (load_result.missing_keys, load_result.unexpected_keys))
    dcp_model.tie_weights()
    if any(parameter.is_meta for parameter in dcp_model.parameters()):
        raise ExportError("DCP model retained uninitialized meta parameters")
    if (dcp_model.get_input_embeddings().weight.data_ptr() !=
            dcp_model.get_output_embeddings().weight.data_ptr()):
        raise ExportError("target model tied embeddings do not share storage")
    dcp_model.cpu().to(dtype=torch.float32).eval()
    dcp_execution_dtype = str(next(dcp_model.parameters()).dtype)
    with torch.inference_mode():
        dcp_logits = dcp_model(**inputs).logits.float().cpu().contiguous()
    dcp_raw = evidence_dir / "dcp_logits.f32le"
    dcp_raw.write_bytes(dcp_logits.numpy().astype("<f4", copy=False).tobytes())
    del dcp_model, state
    gc.collect()
    hf_model, loading_info = AutoModelForImageTextToText.from_pretrained(
        str(exported), trust_remote_code=True, local_files_only=True,
        output_loading_info=True, torch_dtype=torch.float32,
        attn_implementation="eager")
    if any(loading_info.get(k) for k in ("missing_keys", "unexpected_keys", "mismatched_keys",
                                          "error_msgs")):
        raise ExportError("HF reload has missing/unexpected/mismatched model keys: %s" %
                          {k: loading_info.get(k) for k in
                           ("missing_keys", "unexpected_keys", "mismatched_keys", "error_msgs")})
    hf_model.cpu().to(dtype=torch.float32).eval()
    hf_execution_dtype = str(next(hf_model.parameters()).dtype)
    with torch.inference_mode():
        hf_logits = hf_model(**inputs).logits.float().cpu().contiguous()
    hf_raw = evidence_dir / "export_logits.f32le"
    hf_raw.write_bytes(hf_logits.numpy().astype("<f4", copy=False).tobytes())
    if dcp_logits.shape != hf_logits.shape or not torch.isfinite(dcp_logits).all() or \
            not torch.isfinite(hf_logits).all():
        raise ExportError("round-trip logits shape or finite-value mismatch")
    delta = (dcp_logits - hf_logits).abs()
    relative = delta / hf_logits.abs().clamp_min(1e-12)
    passed = bool(torch.allclose(dcp_logits, hf_logits, atol=atol, rtol=rtol))
    result = {"status": "verified" if passed else "mismatch", "input_text": prompt,
              "input_sha256": file_hash(evidence_dir / "input.json"),
              "logits_shape": list(dcp_logits.shape),
              "dcp_logits": {"path": str(dcp_raw), "sha256": file_hash(dcp_raw)},
              "export_logits": {"path": str(hf_raw), "sha256": file_hash(hf_raw)},
              "bitwise_equal": bool(torch.equal(dcp_logits, hf_logits)),
              "max_abs_error": float(delta.max()),
              "max_relative_error": float(relative.max()),
              "mismatched_elements": int((delta > atol + rtol * hf_logits.abs()).sum()),
              "atol": atol, "rtol": rtol,
              "target_loader_file": str(loader_path),
              "target_loader_sha256": file_hash(loader_path),
              "target_imports": {name: {"path": str(path), "sha256": file_hash(path)}
                                 for name, path in paths.items()},
              "checkpoint_inventory_sha256": dcp_inventory["sha256"],
              "dcp_execution_dtype": dcp_execution_dtype,
              "export_execution_dtype": hf_execution_dtype,
              "attention_implementation": "eager",
              "dcp_model_class": dcp_model_class,
              "dcp_model_file": str(dcp_model_file),
              "dcp_model_source_sha256": file_hash(dcp_model_file),
              "hf_model_class": type(hf_model).__module__ + "." +
                                type(hf_model).__qualname__,
              "framework": "torch+transformers", "device": "cpu",
              "dcp_load_missing_keys": load_result.missing_keys,
              "hf_loading_info": loading_info}
    (evidence_dir / "comparison.json").write_text(json.dumps(result, ensure_ascii=False,
        indent=2) + "\n", encoding="utf-8")
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-run-dir", required=True)
    parser.add_argument("--assets-json", required=True)
    parser.add_argument("--target-checkout", required=True)
    parser.add_argument("--iteration", required=True, type=int)
    parser.add_argument("--out", required=True, help="parent of new attempt directory")
    parser.add_argument("--python", default=sys.executable, help="Python with fixed target dependencies")
    parser.add_argument("--inspect-only", action="store_true")
    parser.add_argument("--verify-reload", action="store_true")
    parser.add_argument("--prompt", default="请简要描述工业作业中的安全风险。")
    parser.add_argument("--atol", type=float, default=0.05)
    parser.add_argument("--rtol", type=float, default=0.05)
    args = parser.parse_args(argv)
    if args.verify_reload and args.inspect_only:
        parser.error("--verify-reload requires actual export")
    if not math.isfinite(args.atol) or not math.isfinite(args.rtol) or \
            args.atol < 0 or args.rtol < 0:
        parser.error("tolerances must be finite and nonnegative")
    out_parent = Path(args.out).resolve()
    out_parent.mkdir(parents=True, exist_ok=True)
    attempt = out_parent / ("attempt-" + uuid.uuid4().hex)
    attempt.mkdir()
    receipt = {"schema": "model_artifact.v1", "attempt_id": attempt.name,
               "status": "INCOMPLETE", "reload_verified": False,
               "round_trip": {"status": "not_run"}, "problems": []}
    result_path = attempt / "model_artifact.json"
    rc = 3
    try:
        train = training_receipt(args.train_run_dir)
        receipt["train_run"] = train
        assets, origin, report, origin_inventory = source_assets(args.assets_json)
        bound = train["asset_binding"]
        if (Path(bound.get("assets_json_path") or "").resolve() !=
                Path(args.assets_json).resolve() or
                bound.get("assets_json_sha256") != file_hash(args.assets_json) or
                bound.get("p4_attempt_id") != assets.get("attempt_id") or
                bound.get("migration_id") != assets.get("migration_id") or
                bound.get("hf_file_inventory_sha256") != origin_inventory["sha256"] or
                Path((bound.get("paths") or {}).get("weight_hf") or "").resolve() != origin or
                Path(bound.get("target_checkout") or "").resolve() !=
                Path(args.target_checkout).resolve()):
            raise ExportError("P5 bound P4 attempt/HF/target differs from export inputs")
        from _runtime_asset_binding import capture
        present_binding = capture(bound["assets_json_path"], bound["migration_bundle"],
                                  bound["migration_overlay"], args.target_checkout,
                                  (Path(args.train_run_dir) / "effective_config.yaml").read_bytes())
        if present_binding != bound:
            raise ExportError("P5 bound asset or target bytes drifted after training")
        receipt["current_asset_binding"] = present_binding
        receipt["assets_json"] = {"path": str(Path(args.assets_json).resolve()),
                                  "sha256": file_hash(args.assets_json),
                                  "attempt_id": assets.get("attempt_id")}
        receipt["migration_id"] = assets["migration_id"]
        receipt["migration_manifest"] = {
            "path": assets.get("migration_manifest_path"),
            "sha256": assets.get("migration_manifest_sha256")}
        manifest_path = receipt["migration_manifest"]["path"]
        if not manifest_path or file_hash(manifest_path) != receipt["migration_manifest"]["sha256"]:
            raise ExportError("migration manifest changed since P4")
        target = Path(args.target_checkout).resolve()
        target_id = verify_target_checkout(target, require_patched=True)
        conversion = assets.get("conversion") or {}
        if (conversion.get("target_commit") != target_id["target_commit"] or
                conversion.get("converter_sha256") != target_id["converter_current_sha256"]):
            raise ExportError("P4 target converter identity differs from requested checkout")
        receipt["target"] = target_id
        receipt["origin_hf"] = {"path": str(origin), "model_revision": report["model_revision"],
                                 "config_sha256": report["config_sha256"],
                                 "index_sha256": report["index_sha256"],
                                 "file_inventory_sha256": origin_inventory["sha256"],
                                 "source_index_keys": report["indexed_keys"],
                                 "source_mtp_keys": report["mtp_source_keys"],
                                 "upstream_payload_identity": "unverified"}
        dcp_dir, metadata, dcp_files = selected_checkpoint(train["save_path"],
                                                             args.iteration, train)
        retained, dcp_tensors = expected_model_keys(report, metadata)
        receipt["checkpoint"] = {"path": str(dcp_dir), "format": "dcp",
                                  "iteration": args.iteration, "inventory": dcp_files,
                                  "model_key_count": len(retained), "mtp_key_count": 0,
                                  "storage_dtypes": {dtype: sum(v["dtype"] == dtype
                                      for v in dcp_tensors.values())
                                      for dtype in ("BF16", "F16", "F32")},
                                  "tensor_headers_sha256": hashlib.sha256(json.dumps(
                                      dcp_tensors, sort_keys=True,
                                      separators=(",", ":")).encode()).hexdigest()}
        index = read_json(origin / "model.safetensors.index.json")
        derived, derived_files = derived_origin(origin, index, attempt, dcp_tensors,
                                                 origin_inventory["files"])
        receipt["derived_origin"] = {"path": str(derived), "inventory": derived_files,
                                      "config_sha256": file_hash(derived / "config.json"),
                                      "index_sha256": file_hash(derived / "model.safetensors.index.json"),
                                      "removed_mtp_keys": report["mtp_source_keys"],
                                      "retained_key_count": len(retained),
                                      "reason": "MTP0 training excludes 15 original MTP keys"}
        receipt["converter"] = {"entry": "checkpoint.convert_cli Qwen35Converter dcp_to_hf",
                                "target_commit": target_id["target_commit"],
                                "converter_sha256": target_id["converter_current_sha256"],
                                "to_bf16": False, "keep_origin_mtp_weights": False,
                                "dcp_prefix": "", "hf_prefix": ""}
        if args.inspect_only:
            receipt["status"] = "PREFLIGHT_ONLY"
            rc = 0
        else:
            exported = attempt / "hf"
            exported.mkdir()
            cmd = [args.python, "-m", "checkpoint.convert_cli", "Qwen35Converter",
                   "dcp_to_hf", "--dcp_dir", str(dcp_dir), "--save_hf_dir", str(exported),
                   "--origin_hf_dir", str(derived), "--to_bf16", "false",
                   "--keep_origin_mtp_weights", "false"]
            receipt["converter"]["argv"] = cmd
            log = attempt / "converter.log"
            with log.open("wb") as stream:
                try:
                    done = subprocess.run(cmd, cwd=target, stdout=stream,
                                          stderr=subprocess.STDOUT, timeout=7200, check=False)
                    receipt["converter"]["returncode"] = done.returncode
                except (OSError, subprocess.TimeoutExpired) as exc:
                    receipt["converter"]["error"] = "%s: %s" % (type(exc).__name__, exc)
                    raise ExportError("converter launch or timeout failure") from exc
            receipt["converter"]["log"] = str(log)
            receipt["converter"]["log_sha256"] = file_hash(log)
            if done.returncode != 0:
                raise ExportError("fixed converter failed with rc=%d" % done.returncode)
            dtype_expected = {k: v["dtype"] for k, v in dcp_tensors.items()}
            export = inspect_export(exported, retained, dtype_expected)
            receipt["export_hf"] = {"path": str(exported), **export}
            identity = {"migration_id": receipt["migration_id"],
                        "train_run_id": train["run_id"],
                        "checkpoint_inventory_sha256": dcp_files["sha256"],
                        "export_inventory_sha256": export["inventory"]["sha256"]}
            receipt["model_artifact_id"] = "qwen35-0p8b-" + hashlib.sha256(
                json.dumps(identity, sort_keys=True, separators=(",", ":")).encode()).hexdigest()[:20]
            receipt["status"] = "EXPORTED_RELOAD_UNVERIFIED"
            comparison_passed = False
            if args.verify_reload:
                comparison = verify_round_trip(dcp_dir, exported, derived, target, dcp_files,
                                               dcp_tensors, attempt, args.prompt,
                                               args.atol, args.rtol)
                receipt["round_trip"] = comparison
                if comparison["status"] != "verified":
                    raise ExportError("real-model round-trip exceeds tolerance")
                comparison_passed = True
            if capture(bound["assets_json_path"], bound["migration_bundle"],
                       bound["migration_overlay"], args.target_checkout,
                       (Path(args.train_run_dir) / "effective_config.yaml").read_bytes()) != bound:
                raise ExportError("source assets or target changed during export")
            _, _, final_dcp_files = selected_checkpoint(train["save_path"],
                                                         args.iteration, train)
            derived_now = inventory(derived, [row["path"] for row in derived_files["files"]])
            exported_now = inspect_export(exported, retained, dtype_expected)["inventory"]
            final_train = training_receipt(args.train_run_dir)
            if (final_train != train or final_dcp_files != dcp_files or
                    derived_now != derived_files or
                    exported_now != export["inventory"]):
                raise ExportError("training receipt, checkpoint or exported HF changed during export/reload")
            receipt["post_export_identity_rechecked"] = True
            if comparison_passed:
                receipt["reload_verified"] = True
                receipt["status"] = "RELOAD_VERIFIED"
            rc = 0
    except Exception as exc:
        receipt["problems"].append("%s: %s" % (type(exc).__name__, exc))
        receipt["status"] = "FAILED"
    finally:
        conversion_log = attempt / "converter.log"
        if conversion_log.is_file():
            receipt.setdefault("converter", {})["log"] = str(conversion_log)
            receipt["converter"]["log_sha256"] = file_hash(conversion_log)
        partial = attempt / "hf"
        if partial.is_dir():
            receipt["partial_export_files"] = [
                {"path": path.relative_to(partial).as_posix(), "bytes": path.stat().st_size,
                 "sha256": file_hash(path)} for path in sorted(partial.rglob("*")) if path.is_file()]
        comparison = attempt / "round_trip"
        if comparison.is_dir():
            receipt["round_trip_files"] = [
                {"path": str(path), "bytes": path.stat().st_size,
                 "sha256": file_hash(path)} for path in sorted(comparison.iterdir()) if path.is_file()]
        result_path.write_text(json.dumps(receipt, ensure_ascii=False, indent=2) + "\n",
                               encoding="utf-8")
    print(result_path)
    return rc


if __name__ == "__main__":
    sys.exit(main())
