"""One-attempt local scene inference, with persistent failure evidence."""

import argparse
import importlib
import inspect
import json
from pathlib import Path
import re
import sys
import time
import uuid

from _scene_inference import SceneError, file_inventory, parse_scene_json, sha256, verify_artifact


def save_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def load_runtime():
    import torch
    import transformers
    from PIL import Image
    from transformers import AutoModelForImageTextToText, AutoProcessor
    return torch, transformers, Image, AutoModelForImageTextToText, AutoProcessor


def checked_device(torch, requested, cpu_diagnostic):
    """Resolve an NPU index and require every tensor to use exactly that card."""
    if requested == "cpu":
        if not cpu_diagnostic:
            raise SceneError("CPU requires --cpu-diagnostic")
        return torch.device("cpu"), None
    if not re.fullmatch(r"npu(?::\d+)?", requested):
        raise SceneError("device must be npu[:index] or cpu")
    try:
        torch_npu = importlib.import_module("torch_npu")
    except Exception as exc:
        raise SceneError("torch_npu is required for NPU inference: %s" % exc) from exc
    if not hasattr(torch, "npu") or not torch.npu.is_available():
        raise SceneError("requested NPU is unavailable")
    index = int(requested.split(":", 1)[1]) if ":" in requested else int(torch.npu.current_device())
    device = torch.device("npu:%d" % index)
    torch.npu.set_device(device)
    if int(torch.npu.current_device()) != index:
        raise SceneError("requested NPU index was not selected")
    return device, getattr(torch_npu, "__version__", None)


def checked_tensor_devices(tensors, device):
    actual = sorted({str(t.device) for t in tensors})
    if not tensors or any(t.is_meta or t.device.type != device.type or
                          t.device.index != device.index for t in tensors):
        raise SceneError("model/input tensors are not fully on requested device %s; actual=%s" %
                         (device, actual))
    return actual


def decode_generated(processor, generated, prompt_tokens):
    """Return raw generation while excluding every prompt token from decode."""
    if (generated.ndim != 2 or generated.shape[0] != 1 or
            generated.shape[-1] < prompt_tokens):
        raise SceneError("generated token shape shorter than prompt")
    all_ids = generated[0].detach().cpu().tolist()
    new_ids = all_ids[prompt_tokens:]
    if not new_ids:
        raise SceneError("model generated no new tokens")
    raw = processor.batch_decode([new_ids], skip_special_tokens=True,
                                 clean_up_tokenization_spaces=False)[0]
    return all_ids, new_ids, raw


def infer(args, receipt, image_path, sop_text):
    torch, transformers, Image, Model, Processor = load_runtime()
    try:
        device, torch_npu_version = checked_device(torch, args.device, args.cpu_diagnostic)
    except SceneError as exc:
        receipt["runtime"] = {"torch_npu_import": "failed" if "torch_npu" in str(exc)
                              else "not_verified"}
        raise
    with Image.open(image_path) as image:
        image.verify()
    with Image.open(image_path) as image:
        width, height = image.size
    if width < 1 or height < 1:
        raise SceneError("empty image")
    receipt["image_size"] = [width, height]
    started = time.perf_counter_ns()
    processor = Processor.from_pretrained(args.processor_dir, local_files_only=True,
                                          trust_remote_code=True)
    model, loading = Model.from_pretrained(args.model_dir, local_files_only=True,
        trust_remote_code=True, output_loading_info=True,
        torch_dtype=torch.bfloat16 if device.type == "npu" else torch.float32,
        attn_implementation=args.attention)
    if any(loading.get(k) for k in ("missing_keys", "unexpected_keys",
                                    "mismatched_keys", "error_msgs")):
        raise SceneError("model reload reports missing or mismatched parameters")
    if type(model).__name__ != "Qwen3_5ForConditionalGeneration":
        raise SceneError("loaded model class differs from Qwen3.5 artifact")
    model.to(device).eval()
    tensors = list(model.parameters()) + list(model.buffers())
    model_devices = checked_tensor_devices(tensors, device)
    source = Path(inspect.getfile(type(model))).resolve()
    receipt["runtime"] = {
        "torch_version": torch.__version__, "torch_npu_version": torch_npu_version,
        "torch_npu_import": "success" if device.type == "npu" else "not_requested",
        "transformers_version": transformers.__version__,
        "model_class": type(model).__module__ + "." + type(model).__qualname__,
        "model_source": str(source), "model_source_sha256": sha256(source),
        "processor_class": type(processor).__module__ + "." + type(processor).__qualname__,
        "device": str(device), "model_tensor_devices": model_devices,
        "requested_attention": args.attention,
        "actual_attention": getattr(model.config, "_attn_implementation", None),
        "text_attention": getattr(model.config.text_config, "_attn_implementation", None),
        "vision_attention": getattr(model.config.vision_config, "_attn_implementation", None),
        "gdn_implementation": getattr(model.config.text_config,
                                       "gdn_implementation", None),
        "causal_conv1d_implementation": getattr(model.config.text_config,
                                                   "causal_conv1d_implementation", None),
        "boundary": "local HF export in Transformers; training kernels not implied"}
    expected_class = receipt["round_trip_model_class"]
    if receipt["runtime"]["model_class"] != expected_class:
        raise SceneError("inference class differs from T08 verified HF reload")
    receipt["timing_ms"]["load_model_processor_host_elapsed"] = (time.perf_counter_ns() - started) / 1e6
    if any(receipt["runtime"][key] != args.attention for key in
           ("actual_attention", "text_attention", "vision_attention")):
        raise SceneError("actual attention backend differs from requested backend")
    prompt = ("这是工业作业画面。依据以下标准作业程序判断操作是否合规。\n"
              "SOP:\n" + sop_text + "\n"
              "仅输出 JSON 对象，字段为 compliant(是、否或无法判断)、step、issue、evidence、"
              "bbox([x1,y1,x2,y2]，图像像素坐标)、confidence(0到1)。"
              "证据不足时 compliant=无法判断、bbox=null，issue 和 evidence 说明不可判断原因；不要猜测。")
    messages = [{"role": "user", "content": [
        {"type": "image", "image": str(image_path)},
        {"type": "text", "text": prompt}]}]
    prompt_path = Path(receipt["attempt_dir"]) / "prompt.txt"
    prompt_path.write_text(prompt, encoding="utf-8")
    receipt["prompt"] = {"path": str(prompt_path), "sha256": sha256(prompt_path)}
    started = time.perf_counter_ns()
    inputs = processor.apply_chat_template(messages, tokenize=True,
        add_generation_prompt=True, return_dict=True, return_tensors="pt")
    if "input_ids" not in inputs or inputs["input_ids"].shape[0] != 1:
        raise SceneError("processor did not return one tokenized prompt")
    prompt_tokens = int(inputs["input_ids"].shape[-1])
    inputs = {key: value.to(device) if hasattr(value, "to") else value
              for key, value in inputs.items()}
    receipt["timing_ms"]["preprocess_and_transfer_host_elapsed"] = (time.perf_counter_ns() - started) / 1e6
    receipt["runtime"]["input_tensor_devices"] = checked_tensor_devices(
        [value for value in inputs.values() if hasattr(value, "device")], device)
    if device.type == "npu":
        torch.npu.synchronize(device)
    started = time.perf_counter_ns()
    with torch.inference_mode():
        generated = model.generate(**inputs, max_new_tokens=args.max_new_tokens,
                                   do_sample=False)
    if device.type == "npu":
        torch.npu.synchronize(device)
    receipt["timing_ms"]["generate_synchronized"] = (time.perf_counter_ns() - started) / 1e6
    all_ids, new_ids, raw = decode_generated(processor, generated, prompt_tokens)
    ids_path = Path(receipt["attempt_dir"]) / "generated_ids.json"
    save_json(ids_path, {"prompt_token_count": prompt_tokens,
                         "all_token_ids": all_ids, "new_token_ids": new_ids})
    receipt["generated_ids"] = {"path": str(ids_path), "sha256": sha256(ids_path),
                                "new_token_count": len(new_ids)}
    raw_path = Path(receipt["attempt_dir"]) / "raw_generation.txt"
    raw_path.write_text(raw, encoding="utf-8")
    receipt["raw_generation"] = {"path": str(raw_path), "sha256": sha256(raw_path)}
    if sha256(source) != receipt["runtime"]["model_source_sha256"]:
        raise SceneError("model implementation changed during inference")
    return raw, width, height


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("legacy_image", nargs="?", help="legacy positional image path")
    parser.add_argument("--out", default="out/scene", help="parent of new attempt directory")
    parser.add_argument("--model-artifact")
    parser.add_argument("--model-dir")
    parser.add_argument("--processor-dir")
    parser.add_argument("--image")
    parser.add_argument("--sop", help="UTF-8 SOP text file")
    parser.add_argument("--device", default="npu:0")
    parser.add_argument("--cpu-diagnostic", action="store_true")
    parser.add_argument("--attention", default="sdpa", choices=("sdpa", "eager"))
    parser.add_argument("--max-new-tokens", type=int, default=160)
    args = parser.parse_args(argv)
    if args.legacy_image and args.image:
        parser.error("use either positional image or --image, not both")
    if args.legacy_image:
        args.image = args.legacy_image
    attempt = Path(args.out).resolve() / ("attempt-" + uuid.uuid4().hex)
    try:
        attempt.mkdir(parents=True)
    except OSError as exc:
        print("unable to create inference attempt: %s" % exc, file=sys.stderr)
        return 3
    receipt = {"schema": "inference_result.v1", "inference_run_id": attempt.name,
               "attempt_dir": str(attempt), "status": "FAILED",
               "real_inference_verified": False, "device_requested": args.device,
               "timing_ms": {}, "problems": []}
    rc = 3
    try:
        if not 1 <= args.max_new_tokens <= 4096:
            raise SceneError("max-new-tokens must be within 1..4096")
        if args.device == "cpu" and not args.cpu_diagnostic:
            raise SceneError("CPU requires --cpu-diagnostic")
        if args.device != "cpu" and not re.fullmatch(r"npu(?::\d+)?", args.device):
            raise SceneError("device must be npu[:index] or cpu")
        for name in ("model_artifact", "model_dir", "processor_dir", "image", "sop"):
            if not getattr(args, name):
                raise SceneError("missing --%s" % name.replace("_", "-"))
        image = Path(args.image).resolve(strict=True)
        sop = Path(args.sop).resolve(strict=True)
        if not image.is_file() or not sop.is_file() or image.stat().st_size == 0:
            raise SceneError("image or SOP file missing/empty")
        sop_text = sop.read_text(encoding="utf-8").strip()
        if not sop_text:
            raise SceneError("SOP text is empty")
        receipt["inputs"] = {"image": {"path": str(image), "sha256": sha256(image)},
                             "sop": {"path": str(sop), "sha256": sha256(sop)}}
        artifact, before, artifact_sha = verify_artifact(
            args.model_artifact, args.model_dir, args.processor_dir)
        receipt["model_artifact"] = {"path": str(Path(args.model_artifact).resolve()),
            "sha256": artifact_sha, "model_artifact_id": artifact["model_artifact_id"],
            "migration_id": artifact["migration_id"],
            "train_run_id": artifact["train_run"]["run_id"],
            "checkpoint_iteration": artifact["checkpoint"]["iteration"],
            "checkpoint_inventory_sha256": artifact["checkpoint"]["inventory"]["sha256"],
            "export_inventory_sha256": before["sha256"],
            "target_commit": (artifact.get("target") or {}).get("target_commit"),
            "origin_model_revision": (artifact.get("origin_hf") or {}).get("model_revision")}
        receipt["round_trip_model_class"] = artifact["round_trip"]["hf_model_class"]
        receipt["generation"] = {"max_new_tokens": args.max_new_tokens,
                                 "do_sample": False, "attention": args.attention}
        try:
            raw, width, height = infer(args, receipt, image, sop_text)
        finally:
            if (sha256(image) != receipt["inputs"]["image"]["sha256"] or
                    sha256(sop) != receipt["inputs"]["sop"]["sha256"] or
                    sha256(args.model_artifact) != artifact_sha or
                    file_inventory(args.model_dir) != before):
                raise SceneError("model artifact or scene inputs changed during inference")
        output = receipt.get("raw_generation") or {}
        ids = receipt.get("generated_ids") or {}
        if (not output.get("path") or not ids.get("path") or
                sha256(output["path"]) != output.get("sha256") or
                sha256(ids["path"]) != ids.get("sha256") or
                Path(output["path"]).read_text(encoding="utf-8") != raw):
            raise SceneError("raw generation or token IDs not saved intact")
        npu = bool(re.fullmatch(r"npu(?::\d+)?", args.device))
        receipt["execution_state"] = "NPU_EXECUTED" if npu else "CPU_DIAGNOSTIC"
        try:
            parsed = parse_scene_json(raw, width, height)
        except SceneError as exc:
            receipt["status"] = "PARSE_FAILED"
            receipt["problems"].append(str(exc))
        else:
            parsed_path = attempt / "parsed_result.json"
            save_json(parsed_path, parsed)
            receipt["parsed_result"] = {"path": str(parsed_path), "sha256": sha256(parsed_path)}
            receipt["status"] = ("EVIDENCE_INSUFFICIENT" if
                                 parsed["compliant"] == "无法判断" else "EXECUTED_PARSED")
            rc = 0
        receipt["real_inference_verified"] = npu
        receipt["business_validity"] = "NOT_EVALUATED"
    except Exception as exc:
        receipt["problems"].append("%s: %s" % (type(exc).__name__, exc))
        receipt["status"] = "FAILED"
    try:
        save_json(attempt / "inference_result.json", receipt)
    except OSError as exc:
        print("unable to save inference result: %s" % exc, file=sys.stderr)
        return 3
    print(attempt / "inference_result.json")
    return rc
