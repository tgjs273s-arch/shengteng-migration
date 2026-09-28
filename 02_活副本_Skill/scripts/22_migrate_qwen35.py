#!/usr/bin/env python3
"""Replay the pinned Qwen3.5 GPU -> MindSpeed-MM 0.8B converter delta.

The official MindSpeed-MM model, registration and forward are reused verbatim.
This project's patch rejects missing tied embedding input before DCP conversion.
"""

import argparse
import difflib
import hashlib
import importlib
import json
from pathlib import Path
import shutil
import sys
from urllib.request import urlopen
from urllib.error import URLError

from _qwen35_weights import (HEAD_KEY, metadata_contract, sha256,
                             weight_headers_contract, WeightContractError)
from _qwen35_migration import (CONVERTER, GPU_COMMIT, TARGET_COMMIT, BASE_CONVERTER_SHA256,
                               TARGET_FILES_SHA256,
                               TargetIdentityError, digest, patched_converter_bytes,
                               verify_target_checkout)


MODEL_REVISION = "2fc06364715b967f1860aea9cf38778875588b17"
REQUIREMENTS_DOC_SHA256 = "59dcb02f76b7f8cd068d8d188204e2bfbf4c2a44b8846efeb6e4bfaf674b554f"
REFERENCE_TRAINING_POLICY = {
    "mtp_num_layers": 0, "mtp_loss_scaling_factor": 0.1,
    "source_log_sha256": "f64ec7c38213d9a2e3bc7041f0423fff867a71920c8fd9c32ae6919ccd3e2e26",
    "source_log_lines": [90, 91], "scope": "reference_log_not_this_run"}
GPU_BASE = "https://raw.githubusercontent.com/huggingface/transformers/" + GPU_COMMIT + "/"
TARGET_BASE = "https://raw.githubusercontent.com/Ascend/MindSpeed-MM/" + TARGET_COMMIT + "/"
MODEL_BASE = "https://huggingface.co/Qwen/Qwen3.5-0.8B/resolve/" + MODEL_REVISION + "/"
FILES = {
    "gpu/src/transformers/models/qwen3_5/modeling_qwen3_5.py":
        (GPU_BASE + "src/transformers/models/qwen3_5/modeling_qwen3_5.py", "b6f02dcd1b66610df293084e00bf9bea4fc6a7e5336ffc6ff446edc7ddcd8601"),
    "gpu/src/transformers/models/qwen3_5/configuration_qwen3_5.py":
        (GPU_BASE + "src/transformers/models/qwen3_5/configuration_qwen3_5.py", "2280c6e6bd9d66d7281155243f67cf5fed2756a828af41316566afe611ff16c0"),
    "gpu/src/transformers/models/qwen3_5/modular_qwen3_5.py":
        (GPU_BASE + "src/transformers/models/qwen3_5/modular_qwen3_5.py", "3ef5bf5c0c7606e638f56aaeeb6cdc7bb6b8e4860a59c68f330da0a7f9601022"),
    "target/mindspeed_mm/fsdp/models/qwen3_5/modeling_qwen3_5.py":
        (TARGET_BASE + "mindspeed_mm/fsdp/models/qwen3_5/modeling_qwen3_5.py", TARGET_FILES_SHA256["mindspeed_mm/fsdp/models/qwen3_5/modeling_qwen3_5.py"]),
    "target/mindspeed_mm/fsdp/models/modelhub.py":
        (TARGET_BASE + "mindspeed_mm/fsdp/models/modelhub.py", TARGET_FILES_SHA256["mindspeed_mm/fsdp/models/modelhub.py"]),
    "target/checkpoint/vlm_model/converters/qwen3_5.py":
        (TARGET_BASE + "checkpoint/vlm_model/converters/qwen3_5.py", BASE_CONVERTER_SHA256),
    "model/config.json": (MODEL_BASE + "config.json", "b90b86f35c8e6925ef74ee04d0e758f0a845c83a42089ad82bbaa948de9b4204"),
    "model/model.safetensors.index.json":
        (MODEL_BASE + "model.safetensors.index.json", "d8a08838a613b025eb7952ed9db11696213e57e76a375661ef5c12f9dd5dcf4e"),
    "model/preprocessor_config.json":
        (MODEL_BASE + "preprocessor_config.json", "27225450ac9c6529872ee1924fcb0962ff5634834f817040f444118116f4e516"),
    "model/tokenizer_config.json":
        (MODEL_BASE + "tokenizer_config.json", "49e2b6e395f959f077f1e992b338919c0d4a9732fc6e613995e06557f843500c"),
}
FILES.update({"target/" + relative: (TARGET_BASE + relative, expected)
              for relative, expected in TARGET_FILES_SHA256.items()
              if "target/" + relative not in FILES})

class MigrationError(ValueError):
    pass


class DependencyUnavailable(Exception):
    """A required import is absent before target model construction starts."""


def _sha_bytes(data):
    return hashlib.sha256(data).hexdigest()


def _write_new(path, data):
    path = Path(path)
    if path.exists():
        if path.is_file() and sha256(path) == _sha_bytes(data):
            return
        raise MigrationError("existing output differs: %s" % path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as handle:
        handle.write(data)


def fetch(bundle):
    """Fetch only pinned public source/metadata files, never model weights."""
    bundle = Path(bundle)
    for relative, (url, expected) in FILES.items():
        path = bundle / relative
        if path.exists():
            if sha256(path) != expected:
                raise MigrationError("wrong existing bundle file: %s" % path)
            continue
        with urlopen(url, timeout=30) as response:
            data = response.read(1024 * 1024 + 1)
        if len(data) > 1024 * 1024 or _sha_bytes(data) != expected:
            raise MigrationError("remote file hash/size mismatch: %s" % relative)
        _write_new(path, data)
    print("PINNED_CODE_FETCHED %s" % bundle)


def validate_bundle(bundle):
    bundle = Path(bundle)
    for relative, (_, expected) in FILES.items():
        path = bundle / relative
        if not path.is_file() or sha256(path) != expected:
            raise MigrationError("source or target identity mismatch: %s" % path)
    metadata, _, _ = metadata_contract(bundle / "model")
    return metadata


def apply(bundle, out):
    bundle, out = Path(bundle), Path(out)
    metadata = validate_bundle(bundle)
    original = (bundle / "target" / CONVERTER).read_bytes()
    patched = patched_converter_bytes(original)
    relative = Path("overlay") / CONVERTER
    diff = "".join(difflib.unified_diff(original.decode("utf-8").splitlines(True),
                                        patched.decode("utf-8").splitlines(True),
                                        fromfile="a/" + str(CONVERTER).replace("\\", "/"),
                                        tofile="b/" + str(CONVERTER).replace("\\", "/")))
    gpu_model = (bundle / "gpu/src/transformers/models/qwen3_5/modeling_qwen3_5.py").read_text(encoding="utf-8")
    target_model = (bundle / "target/mindspeed_mm/fsdp/models/qwen3_5/modeling_qwen3_5.py").read_text(encoding="utf-8")
    upstream_diff = "".join(difflib.unified_diff(gpu_model.splitlines(True), target_model.splitlines(True),
                                                  fromfile="gpu/fc91372/modeling_qwen3_5.py",
                                                  tofile="mindspeed-mm/5b55053/modeling_qwen3_5.py"))
    patch_sha = _sha_bytes(diff.encode("utf-8"))
    migration_id = "qwen35-0p8b-" + _sha_bytes((GPU_COMMIT + TARGET_COMMIT + MODEL_REVISION + patch_sha).encode())[:16]
    manifest = {
        "schema": "qwen35_migration.v1", "migration_id": migration_id,
        "input_files_sha256": {name: expected for name, (_, expected) in sorted(FILES.items())},
        "source": {"repository": "huggingface/transformers", "commit": GPU_COMMIT,
                   "modeling_sha256": FILES["gpu/src/transformers/models/qwen3_5/modeling_qwen3_5.py"][1]},
        "target": {"repository": "Ascend/MindSpeed-MM", "commit": TARGET_COMMIT,
                   "modeling_sha256": FILES["target/mindspeed_mm/fsdp/models/qwen3_5/modeling_qwen3_5.py"][1],
                   "modelhub_sha256": FILES["target/mindspeed_mm/fsdp/models/modelhub.py"][1],
                   "source_files_sha256": TARGET_FILES_SHA256.copy(),
                   "converter_before_sha256": _sha_bytes(original),
                   "converter_after_sha256": _sha_bytes(patched),
                   "converter_path": str(CONVERTER).replace("\\", "/")},
        "model_metadata": metadata,
        "reference_training_policy": REFERENCE_TRAINING_POLICY.copy(),
        "requirements_reference": {"pdf_sha256": REQUIREMENTS_DOC_SHA256,
                                   "requested_ref": "26.1.0 branch",
                                   "engineering_locked_commit": TARGET_COMMIT,
                                   "original_material_commit_known": False},
        "project_delta": {"patch_file": "qwen35_converter.patch", "patch_sha256": patch_sha,
                          "reason": "reject missing tied embedding before per-shard HF-to-DCP conversion"},
        "upstream_reuse": ["Qwen3_5ForConditionalGeneration model/forward/registration",
                           "Qwen35Converter shard conversion and existing GDN implementation"],
        "upstream_model_diff": {"path": "upstream_model_diff.patch",
                                "sha256": _sha_bytes(upstream_diff.encode("utf-8")),
                                "ownership": "Ascend/MindSpeed-MM existing implementation"},
        "execution_state": "STATIC_APPLIED_RUNTIME_PENDING", "runtime_import_verified": False,
        "complete_weights_verified": False, "npu_numeric_verified": False,
        "open_gates": ["all local safetensors headers and model state_dict",
                       "fixed target framework import/instantiation", "GDN numeric equivalence", "NPU training"]}
    manifest_bytes = (json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n").encode("utf-8")
    _write_new(out / relative, patched)
    _write_new(out / "qwen35_converter.patch", diff.encode("utf-8"))
    _write_new(out / "upstream_model_diff.patch", upstream_diff.encode("utf-8"))
    _write_new(out / "migration_manifest.json", manifest_bytes)
    print("MIGRATION_STATIC_APPLIED_RUNTIME_PENDING %s" % migration_id)


def validate_overlay(bundle, overlay):
    """Recompute the exact output from pinned inputs and this patch code."""
    metadata = validate_bundle(bundle)
    overlay = Path(overlay)
    manifest = json.loads((overlay / "migration_manifest.json").read_text(encoding="utf-8"))
    if (manifest.get("source", {}).get("commit") != GPU_COMMIT or
            manifest.get("target", {}).get("commit") != TARGET_COMMIT or
            manifest.get("model_metadata", {}).get("model_revision") != MODEL_REVISION or
            manifest.get("source", {}).get("modeling_sha256") !=
                FILES["gpu/src/transformers/models/qwen3_5/modeling_qwen3_5.py"][1] or
            manifest.get("target", {}).get("modeling_sha256") !=
                FILES["target/mindspeed_mm/fsdp/models/qwen3_5/modeling_qwen3_5.py"][1] or
            manifest.get("target", {}).get("modelhub_sha256") !=
                FILES["target/mindspeed_mm/fsdp/models/modelhub.py"][1] or
            manifest.get("target", {}).get("source_files_sha256") != TARGET_FILES_SHA256 or
            manifest.get("model_metadata") != metadata or
            manifest.get("reference_training_policy") != REFERENCE_TRAINING_POLICY or
            manifest.get("requirements_reference") != {
                "pdf_sha256": REQUIREMENTS_DOC_SHA256,
                "requested_ref": "26.1.0 branch",
                "engineering_locked_commit": TARGET_COMMIT,
                "original_material_commit_known": False} or
            manifest.get("input_files_sha256") !=
                {name: expected for name, (_, expected) in sorted(FILES.items())} or
            manifest.get("execution_state") != "STATIC_APPLIED_RUNTIME_PENDING" or
            manifest.get("runtime_import_verified") is not False or
            manifest.get("complete_weights_verified") is not False or
            manifest.get("npu_numeric_verified") is not False):
        raise MigrationError("overlay identity mismatch")
    base = (Path(bundle) / "target" / CONVERTER).read_bytes()
    expected = patched_converter_bytes(base)
    if (manifest["target"].get("converter_before_sha256") != BASE_CONVERTER_SHA256 or
            manifest["target"].get("converter_after_sha256") != digest(expected)):
        raise MigrationError("overlay converter claims differ from code-derived patch")
    diff = "".join(difflib.unified_diff(base.decode("utf-8").splitlines(True),
                                        expected.decode("utf-8").splitlines(True),
                                        fromfile="a/" + CONVERTER.as_posix(),
                                        tofile="b/" + CONVERTER.as_posix()))
    if ((overlay / "qwen35_converter.patch").read_bytes() != diff.encode("utf-8") or
            manifest.get("project_delta", {}).get("patch_sha256") != digest(diff.encode("utf-8"))):
        raise MigrationError("overlay patch differs from code-derived diff")
    expected_id = "qwen35-0p8b-" + digest((GPU_COMMIT + TARGET_COMMIT + MODEL_REVISION +
                                            digest(diff.encode("utf-8"))).encode())[:16]
    if manifest.get("migration_id") != expected_id:
        raise MigrationError("overlay migration_id mismatch")
    gpu_model = (Path(bundle) / "gpu/src/transformers/models/qwen3_5/modeling_qwen3_5.py").read_text(encoding="utf-8")
    target_model = (Path(bundle) / "target/mindspeed_mm/fsdp/models/qwen3_5/modeling_qwen3_5.py").read_text(encoding="utf-8")
    upstream_diff = "".join(difflib.unified_diff(gpu_model.splitlines(True), target_model.splitlines(True),
                                                  fromfile="gpu/fc91372/modeling_qwen3_5.py",
                                                  tofile="mindspeed-mm/5b55053/modeling_qwen3_5.py"))
    if ((overlay / "upstream_model_diff.patch").read_bytes() != upstream_diff.encode("utf-8") or
            manifest.get("upstream_model_diff", {}).get("sha256") != digest(upstream_diff.encode("utf-8"))):
        raise MigrationError("upstream comparison diff identity mismatch")
    patched = overlay / "overlay" / CONVERTER
    if not patched.is_file() or patched.read_bytes() != expected:
        raise MigrationError("overlay converter identity mismatch")
    return manifest, expected


def install(bundle, overlay, checkout):
    """Apply the reviewed overlay to an isolated checkout at the exact commit."""
    manifest, expected = validate_overlay(bundle, overlay)
    overlay, checkout = Path(overlay), Path(checkout)
    patched = overlay / "overlay" / CONVERTER
    target = checkout / CONVERTER
    if target.is_file() and target.read_bytes() == expected:
        verify_target_checkout(checkout, require_patched=True)
        print("MIGRATION_ALREADY_INSTALLED")
        return
    verify_target_checkout(checkout, require_patched=False)
    # No shell, no checkout rewrite. Git retains the immutable original blob.
    shutil.copyfile(patched, target)
    verify_target_checkout(checkout, require_patched=True)
    print("MIGRATION_INSTALLED %s" % manifest["migration_id"])


def _checked_module_path(module, expected):
    actual = Path(getattr(module, "__file__", "")).resolve()
    if actual != Path(expected).resolve():
        raise MigrationError("imported module is outside pinned target checkout: %s" % actual)
    return str(actual)


def _registered_model_class(model_module, model_register):
    # ModelHub resolves the registered class with get(), regardless of decorator return.
    try:
        model_cls = model_register.get("qwen3_5")
    except KeyError as exc:
        raise MigrationError("qwen3_5 model registration missing") from exc
    if (not isinstance(model_cls, type)
            or model_cls.__name__ != "Qwen3_5ForConditionalGeneration"
            or model_cls.__module__ != model_module.__name__
            or sys.modules.get(model_cls.__module__) is not model_module):
        raise MigrationError("qwen3_5 model registration mismatch")
    return model_cls


def runtime(bundle, overlay, checkout, out, hf_dir=None, processor_dir=None):
    """Optional dependency-backed model construction and parameter shape check."""
    manifest, _ = validate_overlay(bundle, overlay)
    identity = verify_target_checkout(checkout, require_patched=True)
    if manifest.get("target", {}).get("converter_after_sha256") != identity["patch_identity"]:
        raise MigrationError("runtime manifest patch differs from code-derived target identity")
    try:
        from accelerate import init_empty_weights
        from transformers import AutoConfig, AutoProcessor
        sys.path.insert(0, str(Path(checkout).resolve()))
        model_module = importlib.import_module("mindspeed_mm.fsdp.models.qwen3_5.modeling_qwen3_5")
        modelhub_module = importlib.import_module("mindspeed_mm.fsdp.models.modelhub")
        from mindspeed_mm.fsdp.utils.register import model_register
        from mindspeed_mm.fsdp.params.model_args import ModelArguments
        from mindspeed_mm.fsdp.params.feature_args import FeatureArguments
    except (ImportError, ModuleNotFoundError) as exc:
        raise DependencyUnavailable("target model dependencies unavailable: %s" % exc) from exc
    register_module = sys.modules[model_register.__class__.__module__]
    target_root = Path(checkout).resolve()
    import_paths = {
        "modeling": _checked_module_path(
            model_module, target_root / "mindspeed_mm/fsdp/models/qwen3_5/modeling_qwen3_5.py"),
        "modelhub": _checked_module_path(
            modelhub_module, target_root / "mindspeed_mm/fsdp/models/modelhub.py"),
        "register": _checked_module_path(
            register_module, target_root / "mindspeed_mm/fsdp/utils/register.py"),
        "model_args": _checked_module_path(
            sys.modules[ModelArguments.__module__], target_root / "mindspeed_mm/fsdp/params/model_args.py"),
        "feature_args": _checked_module_path(
            sys.modules[FeatureArguments.__module__], target_root / "mindspeed_mm/fsdp/params/feature_args.py")}
    model_cls = _registered_model_class(model_module, model_register)
    config = AutoConfig.from_pretrained(str(Path(bundle) / "model"), local_files_only=True)
    # Match ModelHub's config override API; the reference loss log uses mtp_num_layers=0.
    model_args = ModelArguments(model_id="qwen3_5", mtp_num_layers=0, mtp_loss_scaling_factor=0.1,
                                gdn_implementation="eager", causal_conv1d_implementation="eager",
                                skip_gdn_recompute=False, skip_flash_attn_recompute=False)
    feature_args = FeatureArguments(enable_chunk_loss=False, enable_dynamic_chunk_loss=False)
    config = model_cls.overwrite_transformer_config(config, model_args, feature_args)
    if (config.text_config.mtp_num_layers != 0 or
            config.text_config.gdn_implementation != "eager" or
            config.text_config.causal_conv1d_implementation != "eager"):
        raise MigrationError("meta model config overrides differ from reference policy")
    with init_empty_weights():
        model = model_cls._from_config(config)
    state = model.state_dict()
    if any(key.startswith("mtp.") for key in state):
        raise MigrationError("meta model unexpectedly enabled MTP parameters")
    required_inputs = {"input_ids", "attention_mask", "position_ids", "labels",
                       "pixel_values", "image_grid_thw"}
    import inspect
    if not required_inputs.issubset(inspect.signature(model.forward).parameters):
        raise MigrationError("target forward interface mismatch")
    result = {"schema": "qwen35_runtime_validation.v1", "migration_id": manifest["migration_id"],
              "import_paths": import_paths,
              "meta_overrides": {"gdn_implementation": "eager", "causal_conv1d_implementation": "eager",
                                 "mtp_num_layers": 0, "mtp_loss_scaling_factor": 0.1,
                                 "scope": "meta_instantiation_only",
                                 "target_triton_executed": False},
              "mtp_policy": {"reference_log_mtp_num_layers": 0,
                             "indexed_source_keys": manifest["model_metadata"]["mtp_source_keys"],
                             "source_weight_headers_checked": False,
                             "target_meta_parameters": "disabled",
                             "dcp_mtp_reload_verified": False},
              "model_registration": "verified", "meta_instantiation": "verified",
              "forward_signature": "verified", "state_dict_shapes": "pending",
              "processor": "pending", "npu_numeric_verified": False,
              "ignored_checkpoint_keys": {"rule": "^mtp.*", "keys": None,
                                          "status": "pending_no_weight_headers"},
              "tied_weight_exclusion": {"key": HEAD_KEY,
                                          "source": "model.language_model.embed_tokens.weight"}}
    if hf_dir:
        headers = weight_headers_contract(hf_dir, include_inventory=True)["tensor_headers"]
        missing, wrong_shape = [], []
        for key, tensor in state.items():
            if key == HEAD_KEY:
                continue  # Explicit tied mapping, checked by weight_headers_contract.
            if key not in headers:
                missing.append(key)
            elif list(tensor.shape) != headers[key]["shape"]:
                wrong_shape.append(key)
        ignored_mtp = sorted(key for key in headers if key.startswith("mtp.") and key not in state)
        extra = sorted(set(headers) - set(state) - set(ignored_mtp))
        if missing or wrong_shape or extra:
            raise MigrationError("state_dict mismatch missing=%s shape=%s extra=%s" %
                                 (missing[:8], wrong_shape[:8], extra[:8]))
        result["state_dict_shapes"] = "verified_excluding_listed_mtp_and_tied_head"
        result["ignored_checkpoint_keys"] = {"rule": "^mtp.*", "keys": ignored_mtp,
                                              "status": "listed_from_local_weight_headers"}
        result["mtp_policy"]["source_weight_headers_checked"] = True
        result["tensor_payload_hashes_verified"] = False
    if processor_dir:
        metadata_contract(processor_dir)
        AutoProcessor.from_pretrained(processor_dir, local_files_only=True)
        result["processor"] = "loaded_from_local_files"
    _write_new(Path(out) / "runtime_validation.json",
               (json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True) + "\n").encode("utf-8"))
    print("MIGRATION_RUNTIME_CHECKED %s" % manifest["migration_id"])


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("fetch", help="download small pinned public source and metadata only")
    p.add_argument("--bundle", required=True)
    p = sub.add_parser("apply", help="validate bundle and produce converter overlay + manifest")
    p.add_argument("--bundle", required=True)
    p.add_argument("--out", required=True)
    p = sub.add_parser("install", help="install overlay into an isolated exact-commit checkout")
    p.add_argument("--bundle", required=True)
    p.add_argument("--overlay", required=True)
    p.add_argument("--target-checkout", required=True)
    p = sub.add_parser("runtime", help="optional model registration/meta/state_dict check")
    p.add_argument("--bundle", required=True)
    p.add_argument("--overlay", required=True)
    p.add_argument("--target-checkout", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--hf-dir", help="local complete safetensors checkout; never downloaded here")
    p.add_argument("--processor-dir", help="local processor files; never downloaded here")
    args = parser.parse_args(argv)
    try:
        if args.command == "fetch":
            fetch(args.bundle)
        elif args.command == "apply":
            apply(args.bundle, args.out)
        elif args.command == "install":
            install(args.bundle, args.overlay, args.target_checkout)
        else:
            runtime(args.bundle, args.overlay, args.target_checkout, args.out,
                    args.hf_dir, args.processor_dir)
    except DependencyUnavailable as exc:
        print("MIGRATION_RUNTIME_UNAVAILABLE: %s" % exc, file=sys.stderr)
        return 4
    except (MigrationError, TargetIdentityError, WeightContractError, OSError, UnicodeError,
            json.JSONDecodeError, ValueError, URLError, RuntimeError) as exc:
        print("MIGRATION_FAILED: %s" % exc, file=sys.stderr)
        return 3
    return 0


if __name__ == "__main__":
    sys.exit(main())
