"""Target-bound Qwen3.5 operator checks. No model weights or training are loaded.

These are engineering comparisons, not an official loss or accuracy gate.
"""

import hashlib
import importlib
import importlib.util
import json
import subprocess
import sys
from array import array
from pathlib import Path

TARGET_COMMIT = "5b5505331924634da64e3d9a1925d02b10babe9f"
MODEL = "mindspeed_mm/fsdp/models/qwen3_5/modeling_qwen3_5.py"
# Fixed before observing any NPU error. BF16 is the target GDN kernel dtype.
TOLERANCE = {"atol": 0.05, "rtol": 0.05, "origin": "T04 engineering BF16 smoke threshold; not official precision rule"}


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def within(path, root):
    try:
        Path(path).resolve().relative_to(Path(root).resolve())
        return True
    except ValueError:
        return False


def _load_t03():
    """Use T03's fixed producer checks, including converter and tracked delta."""
    scripts = Path(__file__).resolve().parent
    if str(scripts) not in sys.path:
        sys.path.insert(0, str(scripts))
    spec = importlib.util.spec_from_file_location("t03_migrator_for_numeric", scripts / "22_migrate_qwen35.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    target = importlib.import_module("_qwen35_migration")
    return module.validate_overlay, target.verify_target_checkout


def verify_identity(manifest_path, bundle, checkout):
    manifest_path = Path(manifest_path).resolve()
    if manifest_path.name != "migration_manifest.json" or not manifest_path.is_file():
        raise ValueError("explicit migration manifest file required")
    bundle, checkout = Path(bundle).resolve(), Path(checkout).resolve()
    validate_overlay, verify_checkout = _load_t03()
    doc, _ = validate_overlay(bundle, manifest_path.parent)
    target_receipt = verify_checkout(checkout, require_patched=True)
    if doc["target"]["commit"] != target_receipt["target_commit"]:
        raise ValueError("validated overlay and target checkout differ")
    return doc, target_receipt


def row(op, status, **detail):
    return {"op": op, "status": status, **detail}


class ArtifactWriter:
    """Write self-describing float32 little-endian tensor bytes, without pickle."""

    def __init__(self, root):
        self.root = Path(root).resolve()
        self.records = []
        self.root.mkdir(parents=True, exist_ok=True)

    def dump(self, name, tensor):
        target = self.root / (name.replace(".", "/") + ".f32le")
        target.parent.mkdir(parents=True, exist_ok=True)
        values = array("f", tensor.detach().float().cpu().reshape(-1).tolist())
        if sys.byteorder != "little":
            values.byteswap()
        payload = values.tobytes()
        target.write_bytes(payload)
        self.records.append({"name": name, "path": str(target), "sha256": hashlib.sha256(payload).hexdigest(),
                             "shape": list(tensor.shape), "source_dtype": str(tensor.dtype),
                             "encoding": "float32_le"})


def compare(a, b, label, torch, artifacts=None):
    if tuple(a.shape) != tuple(b.shape):
        raise ValueError(f"{label}: shape {tuple(a.shape)} != {tuple(b.shape)}")
    aa, bb = a.detach().float().cpu(), b.detach().float().cpu()
    if artifacts is not None:
        artifacts.dump(label + ".candidate", a)
        artifacts.dump(label + ".reference", b)
    if not torch.isfinite(aa).all() or not torch.isfinite(bb).all():
        raise ValueError(f"{label}: nonfinite output")
    delta = (aa - bb).abs()
    limit = TOLERANCE["atol"] + TOLERANCE["rtol"] * bb.abs()
    flat_index = int(delta.reshape(-1).argmax()) if delta.numel() else None
    result = {"shape": list(aa.shape), "max_abs": float(delta.max()) if delta.numel() else 0.0,
              "max_rel_nonzero": float((delta / bb.abs().clamp_min(1e-6)).max()) if delta.numel() else 0.0,
              "worst_flat_index": flat_index,
              "worst_candidate": float(aa.reshape(-1)[flat_index]) if flat_index is not None else None,
              "worst_reference": float(bb.reshape(-1)[flat_index]) if flat_index is not None else None,
              "within_tolerance": bool(torch.all(delta <= limit))}
    return result


def measured_row(op, metrics, implementation, reference, level="numeric_gradient", **detail):
    passed = bool(metrics) and all(value["within_tolerance"] for value in metrics.values())
    return row(op, "forward_ok" if passed else "error", validation_level=level if passed else "numeric_failed",
               implementation=implementation, reference=reference, metrics=metrics,
               tolerance=TOLERANCE, **detail)


def imported_path(obj):
    module = importlib.import_module(obj.__module__)
    return str(Path(module.__file__).resolve())


def ensure_target_impl(path, checkout):
    if not within(path, checkout):
        raise ValueError("backend imported outside target checkout: " + path)
    rel = str(Path(path).resolve().relative_to(Path(checkout).resolve())).replace("\\", "/")
    tracked = subprocess.run(["git", "-C", str(checkout), "ls-files", "--error-unmatch", "--", rel],
                             capture_output=True, text=True, encoding="utf-8", timeout=10)
    clean = subprocess.run(["git", "-C", str(checkout), "diff", "--exit-code", "HEAD", "--", rel],
                           capture_output=True, text=True, encoding="utf-8", timeout=10)
    if tracked.returncode or clean.returncode:
        raise ValueError("backend is not an unmodified file from target commit: " + rel)
    return {"path": str(Path(path).resolve()), "sha256": sha(path), "git_path": rel}


def check_gdn(modeling, torch, device, checkout, config, progress):
    op = "target_gdn_triton_prefill"
    fn = importlib.import_module("mindspeed_mm.fsdp.ops.gdn.chunk_gated_delta_rule").chunk_gated_delta_rule
    implementation = imported_path(fn)
    backend_identity = ensure_target_impl(implementation, checkout)
    progress["actual_backends"] = {"chunk_gated_delta_rule": backend_identity}
    reference = modeling.torch_chunk_gated_delta_rule
    H, K, V = config.linear_num_value_heads, config.linear_key_head_dim, config.linear_value_head_dim
    metrics = progress["metrics"]
    artifacts = progress["artifacts"]
    for length in (63, 64, 65):
        progress["stage"] = f"GDN T={length}"
        torch.manual_seed(7400 + length)
        values = [torch.randn((1, length, H, dim), dtype=torch.float32).mul_(0.1).to(torch.bfloat16)
                  for dim in (K, K, V)]
        values += [torch.randn(1, length, H, dtype=torch.float32).mul_(0.1).sub_(1),
                   torch.rand(1, length, H, dtype=torch.float32).to(torch.bfloat16)]
        for name, value in zip(("q", "k", "v", "g", "beta"), values):
            artifacts.dump(f"gdn.T{length}.input.{name}", value)
        left = [x.to(device).detach().requires_grad_(True) for x in values]
        right = [x.detach().clone().requires_grad_(True) for x in values]
        out, state = fn(*left, output_final_state=True, use_qk_l2norm_in_kernel=True, skip_recompute=True)
        ref_out, ref_state = reference(*right, output_final_state=True, use_qk_l2norm_in_kernel=True)
        prefix = f"T{length}"
        metrics[prefix + ".output"] = compare(out, ref_out, prefix + ".output", torch, artifacts)
        metrics[prefix + ".final_state"] = compare(state, ref_state, prefix + ".final_state", torch, artifacts)
        out.float().sum().backward()
        ref_out.float().sum().backward()
        for name, a, b in zip(("q", "k", "v", "g", "beta"), left, right):
            metrics[prefix + ".grad_" + name] = compare(a.grad, b.grad, prefix + ".grad_" + name, torch, artifacts)
    return measured_row(op, metrics, implementation, imported_path(reference), backend_identity=backend_identity,
                        device=str(device), dtype="bfloat16",
                        shapes={"heads": H, "key_dim": K, "value_dim": V, "lengths": [63, 64, 65]},
                        skip_gdn_recompute=True, seed_formula="7400 + length")


def check_conv(torch, device, checkout, config, progress):
    import torch.nn.functional as F
    op = "target_causal_conv_triton_prefill"
    fn = importlib.import_module("mindspeed_mm.fsdp.models.qwen3_5.causal_conv1d").causal_conv1d
    implementation = imported_path(fn)
    backend_identity = ensure_target_impl(implementation, checkout)
    progress["actual_backends"] = {"causal_conv1d_fn": backend_identity}
    channels = 2 * config.linear_num_key_heads * config.linear_key_head_dim + config.linear_num_value_heads * config.linear_value_head_dim
    kernel = config.linear_conv_kernel_dim
    metrics = progress["metrics"]
    artifacts = progress["artifacts"]
    for length in (1, 4, 65):
        progress["stage"] = f"causal_conv T={length}"
        torch.manual_seed(8200 + length)
        x = torch.randn(1, length, channels, dtype=torch.float32).mul_(0.1).to(torch.bfloat16)
        w = torch.randn(kernel, channels, dtype=torch.float32).mul_(0.1).to(torch.bfloat16)
        artifacts.dump(f"conv.T{length}.input.x", x)
        artifacts.dump(f"conv.T{length}.input.weight", w)
        xx, ww = x.to(device).detach().requires_grad_(True), w.to(device).detach().requires_grad_(True)
        rx, rw = x.detach().requires_grad_(True), w.detach().requires_grad_(True)
        out, _ = fn(x=xx, weight=ww, activation="silu")
        ref = F.silu(F.conv1d(rx.transpose(1, 2), rw.transpose(0, 1).unsqueeze(1),
                             padding=kernel - 1, groups=channels)[:, :, :length].transpose(1, 2))
        prefix = f"T{length}"
        metrics[prefix + ".output"] = compare(out, ref, prefix + ".output", torch, artifacts)
        out.float().sum().backward()
        ref.float().sum().backward()
        metrics[prefix + ".grad_x"] = compare(xx.grad, rx.grad, prefix + ".grad_x", torch, artifacts)
        metrics[prefix + ".grad_weight"] = compare(ww.grad, rw.grad, prefix + ".grad_weight", torch, artifacts)
    return measured_row(op, metrics, implementation, "torch.nn.functional.conv1d+SiLU", backend_identity=backend_identity,
                        device=str(device),
                        dtype="bfloat16", shapes={"channels": channels, "kernel": kernel, "lengths": [1, 4, 65]},
                        seed_formula="8200 + length")


def small_config():
    return dict(vocab_size=64, hidden_size=128, intermediate_size=256, num_hidden_layers=2,
                  num_attention_heads=2, num_key_value_heads=1, head_dim=64,
                  linear_key_head_dim=32, linear_value_head_dim=32, linear_num_key_heads=2,
                  linear_num_value_heads=2, linear_conv_kernel_dim=4,
                  layer_types=["linear_attention", "full_attention"], mtp_num_layers=0,
                  skip_gdn_recompute=False, causal_conv1d_implementation="eager", gdn_implementation="eager")


def dump_small_model_inputs(model, artifacts, torch):
    for name, tensor in model.state_dict().items():
        artifacts.dump("model.weights." + name, tensor)
    artifacts.dump("model.input.prefill_tokens", torch.tensor([[1, 2, 3, 4]], dtype=torch.long))
    artifacts.dump("model.input.decode_token", torch.tensor([[5]], dtype=torch.long))
    artifacts.dump("model.input.decode_position", torch.tensor([4], dtype=torch.long))


def check_cpu_eager(modeling, torch, progress):
    """Run the actual fixed target text class on CPU, including one decode step."""
    op = "target_text_model_cpu_eager_cache"
    torch.manual_seed(9137)
    model = modeling.Qwen3_5TextModel(modeling.Qwen3_5TextConfig(**small_config())).cpu().eval()
    artifacts = progress["artifacts"]
    dump_small_model_inputs(model, artifacts, torch)
    gdn = model.layers[0].linear_attn
    actual = {name: imported_path(getattr(gdn, name)) if getattr(gdn, name) is not None else None
              for name in ("chunk_gated_delta_rule", "causal_conv1d_fn", "recurrent_gated_delta_rule", "causal_conv1d_update")}
    progress["actual_backends"] = actual
    tokens = torch.tensor([[1, 2, 3, 4]], dtype=torch.long)
    with torch.no_grad():
        progress["stage"] = "CPU eager prefill"
        prefill = model(input_ids=tokens, use_cache=True)
        for name, tensor in (("hidden", prefill.last_hidden_state),
                             ("conv_cache", prefill.past_key_values.conv_states[0]),
                             ("recurrent_cache", prefill.past_key_values.recurrent_states[0])):
            if tensor is None or not bool(torch.isfinite(tensor).all()):
                raise ValueError("CPU eager prefill " + name + " is missing/nonfinite")
            artifacts.dump("cpu.prefill." + name, tensor)
            progress["metrics"]["prefill." + name] = {"shape": list(tensor.shape), "finite": True}
        progress["stage"] = "CPU eager decode"
        decode = model(input_ids=torch.tensor([[5]], dtype=torch.long), past_key_values=prefill.past_key_values,
                       use_cache=True, cache_position=torch.tensor([4]))
        for name, tensor in (("hidden", decode.last_hidden_state),
                             ("conv_cache", decode.past_key_values.conv_states[0]),
                             ("recurrent_cache", decode.past_key_values.recurrent_states[0])):
            if tensor is None or not bool(torch.isfinite(tensor).all()):
                raise ValueError("CPU eager decode " + name + " is missing/nonfinite")
            artifacts.dump("cpu.decode." + name, tensor)
            progress["metrics"]["decode." + name] = {"shape": list(tensor.shape), "finite": True}
    return row(op, "forward_ok", validation_level="forward_cache", implementation=str(Path(modeling.__file__).resolve()),
               actual_backends=actual, metrics=progress["metrics"], device="cpu", dtype="float32", seed=9137,
               tokens=[1, 2, 3, 4, 5], scope="text hidden/cache only; no visual inputs or logits")


def check_text_model(modeling, torch, device, checkout, progress):
    """Same initialized small target text model: eager versus target Triton prefill/decode."""
    op = "target_text_model_prefill_decode_cache"
    cfg_type = modeling.Qwen3_5TextConfig
    common = small_config()
    torch.manual_seed(9137)
    reference = modeling.Qwen3_5TextModel(cfg_type(**common)).cpu().eval()
    artifacts = progress["artifacts"]
    dump_small_model_inputs(reference, artifacts, torch)
    candidate_config = cfg_type(**{**common, "causal_conv1d_implementation": "triton",
                                   "gdn_implementation": "triton", "skip_gdn_recompute": True})
    candidate = modeling.Qwen3_5TextModel(candidate_config).to(device=device, dtype=torch.bfloat16).eval()
    candidate.load_state_dict(reference.state_dict(), strict=True)
    gdn = candidate.layers[0].linear_attn
    actual = {name: imported_path(getattr(gdn, name)) for name in
              ("chunk_gated_delta_rule", "causal_conv1d_fn", "recurrent_gated_delta_rule", "causal_conv1d_update")}
    progress["actual_backends"] = actual
    identities = {name: ensure_target_impl(actual[name], checkout) for name in
                  ("chunk_gated_delta_rule", "causal_conv1d_fn")}
    tokens = torch.tensor([[1, 2, 3, 4]], dtype=torch.long)
    metrics = progress["metrics"]
    with torch.no_grad():
        progress["stage"] = "text model prefill"
        ref_prefill = reference(input_ids=tokens, use_cache=True)
        got_prefill = candidate(input_ids=tokens.to(device), use_cache=True)
        metrics["prefill.hidden"] = compare(got_prefill.last_hidden_state, ref_prefill.last_hidden_state, "prefill.hidden", torch, artifacts)
        metrics["prefill.conv_cache"] = compare(got_prefill.past_key_values.conv_states[0],
                                                  ref_prefill.past_key_values.conv_states[0], "prefill.conv_cache", torch, artifacts)
        metrics["prefill.recurrent_cache"] = compare(got_prefill.past_key_values.recurrent_states[0],
                                                       ref_prefill.past_key_values.recurrent_states[0], "prefill.recurrent_cache", torch, artifacts)
        progress["stage"] = "text model decode"
        next_token = torch.tensor([[5]], dtype=torch.long)
        ref_decode = reference(input_ids=next_token, past_key_values=ref_prefill.past_key_values,
                               use_cache=True, cache_position=torch.tensor([4]))
        got_decode = candidate(input_ids=next_token.to(device), past_key_values=got_prefill.past_key_values,
                               use_cache=True, cache_position=torch.tensor([4], device=device))
        metrics["decode.hidden"] = compare(got_decode.last_hidden_state, ref_decode.last_hidden_state, "decode.hidden", torch, artifacts)
        metrics["decode.conv_cache"] = compare(got_decode.past_key_values.conv_states[0],
                                                 ref_decode.past_key_values.conv_states[0], "decode.conv_cache", torch, artifacts)
        metrics["decode.recurrent_cache"] = compare(got_decode.past_key_values.recurrent_states[0],
                                                      ref_decode.past_key_values.recurrent_states[0], "decode.recurrent_cache", torch, artifacts)
    return measured_row(op, metrics, str(Path(modeling.__file__).resolve()), "same target class with eager config", level="numeric_cache",
                        actual_backends=actual, backend_identities=identities, device=str(device),
                        dtype="float32 CPU reference; bfloat16 NPU candidate",
                        shape={"batch": 1, "prefill_tokens": 4, "decode_tokens": 1, "hidden": 128}, seed=9137,
                         tokens=[1, 2, 3, 4, 5], skip_gdn_recompute={"candidate": True, "eager_reference": False},
                         scope="text hidden/cache only; no visual inputs or logits")


def execute_check(name, fn, *, fixture, identity, runtime, artifact_dir):
    progress = {"stage": "initialization", "metrics": {}}
    try:
        progress["artifacts"] = ArtifactWriter(Path(artifact_dir) / name)
        result = fn(progress)
        return {**result, "identity": identity, "runtime": runtime, "fixture": fixture,
                "artifacts": progress["artifacts"].records}
    except Exception as exc:
        return row(name, "error", validation_level="execution_failed", failed_stage=progress["stage"],
                   err=f"{type(exc).__name__}: {exc}", completed_metrics=progress["metrics"],
                   actual_backends=progress.get("actual_backends"), identity=identity,
                   runtime=runtime, fixture=fixture,
                   artifacts=progress["artifacts"].records if "artifacts" in progress else [])


def run(manifest_path, bundle, checkout, artifact_dir):
    doc, target_receipt = verify_identity(manifest_path, bundle, checkout)
    identity = {"migration_id": doc["migration_id"], "source_commit": doc["source"]["commit"],
                "target_commit": TARGET_COMMIT, "manifest_sha256": sha(manifest_path),
                "target_checkout": str(Path(checkout).resolve()), "bundle": str(Path(bundle).resolve()),
                "target_receipt": target_receipt, "artifact_dir": str(Path(artifact_dir).resolve())}
    cpu_name = "target_text_model_cpu_eager_cache"
    pending = ("target_gdn_triton_prefill", "target_causal_conv_triton_prefill",
               "target_text_model_prefill_decode_cache")
    all_names = (cpu_name,) + pending
    try:
        import torch
    except (ImportError, ModuleNotFoundError) as exc:
        return [row(op, "contract-only", validation_level="dependency_unavailable",
                    reason=f"{type(exc).__name__}: {exc}", identity=identity) for op in all_names]

    runtime = {"torch_version": str(torch.__version__), "torch_file": str(Path(torch.__file__).resolve())}
    npu_probe_error = None
    try:
        import torch_npu  # noqa: F401 -- attach torch.npu before target module import
        npu_available = bool(torch.npu.is_available())
    except (ImportError, ModuleNotFoundError):
        npu_available = False
    except Exception as exc:
        npu_available = False
        npu_probe_error = f"{type(exc).__name__}: {exc}"
    runtime["npu_available"] = npu_available
    checkout = Path(checkout).resolve()
    if str(checkout) not in sys.path:
        sys.path.insert(0, str(checkout))
    try:
        modeling = importlib.import_module("mindspeed_mm.fsdp.models.qwen3_5.modeling_qwen3_5")
    except (ImportError, ModuleNotFoundError) as exc:
        return [row(op, "contract-only", validation_level="dependency_unavailable",
                    reason=f"{type(exc).__name__}: {exc}", identity=identity, runtime=runtime) for op in all_names]
    except Exception as exc:
        return [row("target_runtime_import", "error", validation_level="execution_failed",
                    err=f"{type(exc).__name__}: {exc}", identity=identity, runtime=runtime)]
    if Path(modeling.__file__).resolve() != checkout / MODEL or sha(modeling.__file__) != doc["target"]["modeling_sha256"]:
        return [row("target_runtime_import", "error", validation_level="identity_failed",
                    err="actual imported Qwen3.5 model differs from migration target", identity=identity, runtime=runtime)]
    runtime["target_model_file"] = str(Path(modeling.__file__).resolve())

    def execute(name, fn, fixture):
        return execute_check(name, fn, fixture=fixture, identity=identity, runtime=runtime,
                             artifact_dir=artifact_dir)

    results = [execute(cpu_name, lambda progress: check_cpu_eager(modeling, torch, progress),
                       fixture={"seed": 9137, "tokens": [1, 2, 3, 4, 5], "dtype": "float32",
                                "shape": [1, 4], "mtp_num_layers": 0})]
    if not npu_available:
        if npu_probe_error:
            return results + [row("target_npu_probe", "error", validation_level="execution_failed",
                                  err=npu_probe_error, identity=identity, runtime=runtime)]
        reason = "current interpreter has no available Ascend NPU; target Triton not executed"
        return results + [row(op, "contract-only", validation_level="not_executed", reason=reason,
                              identity=identity, runtime=runtime) for op in pending]

    config_path = Path(bundle).resolve() / "model" / "config.json"
    if sha(config_path) != doc["model_metadata"]["config_sha256"]:
        return results + [row("target_model_config", "error", validation_level="identity_failed",
                              err="0.8B config differs from validated migration bundle", identity=identity,
                              runtime=runtime)]
    config = modeling.Qwen3_5Config.from_dict(json.loads(config_path.read_text(encoding="utf-8"))).text_config
    device = torch.device("npu:0")
    results.append(execute(pending[0], lambda p: check_gdn(modeling, torch, device, checkout, config, p),
                           fixture={"seed_formula": "7400 + length", "lengths": [63, 64, 65],
                                    "dtype": "q/k/v/beta=bfloat16,g=float32", "skip_gdn_recompute": True,
                                    "shape": [1, "T", config.linear_num_value_heads, config.linear_key_head_dim]}))
    results.append(execute(pending[1], lambda p: check_conv(torch, device, checkout, config, p),
                           fixture={"seed_formula": "8200 + length", "lengths": [1, 4, 65],
                                    "dtype": "bfloat16", "channels": 2 * config.linear_num_key_heads * config.linear_key_head_dim +
                                    config.linear_num_value_heads * config.linear_value_head_dim}))
    results.append(execute(pending[2], lambda p: check_text_model(modeling, torch, device, checkout, p),
                           fixture={"seed": 9137, "tokens": [1, 2, 3, 4, 5], "dtype": "float32/bfloat16",
                                    "shape": [1, 4, 128], "mtp_num_layers": 0,
                                    "scope": "small text hidden/cache, no visual inputs or logits"}))
    return results
