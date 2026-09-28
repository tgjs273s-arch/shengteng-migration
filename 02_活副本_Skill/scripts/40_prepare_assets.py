#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
40_prepare_assets.py — P4 权重与数据准备（含字节级校验锚点）

职责：
  1. 权重：hf 权重下载（modelscope 优先）→ DCP 转换（官方 convert_cli）
  2. 数据：LLaVA 提示词 + COCO2017 图片 → 官方转换脚本生成训练 json
  3. 校验：对关键资产做**字节级/sample 数校验**（锚点来自 config/env_matrix.yaml → assets）
  4. 产出 out/assets/assets.json（路径 + 大小 + 校验结果 + 可比性等级）

设计（冗余）：每类资产有主源与备源；下载失败自动提示备源；已存在且校验通过则跳过（幂等）。
退出码：0 成功；2 参数/IO；3 校验失败；4 缺关键工具（modelscope 等）
"""

import argparse
import importlib.util
import json
import os
from pathlib import Path
import shutil
import shlex
import subprocess
import sys
import time
import uuid

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _qwen35_weights import (MODEL_REVISION, TIE_MAPPING, WeightContractError,
                             sha256, weight_headers_contract)
from _qwen35_migration import TargetIdentityError, verify_target_checkout
from _asset_integrity import (REQUIRED_HF_FILES, CONVERTER_COMMIT, converter_identity,
                              inspect_data, inspect_dcp, inspect_hf, inspect_llava)

# 校验锚点（与官方一致；修改需有依据）
ANCHORS = {
    "llava_json": {"name": "llava_instruct_150k.json", "bytes": 228941895},
    "converted_json": {"name": "output_llava_coco_data.json", "samples": 157712},
    "coco_images": {"images": 118287},
}


# ---------------------------------------------------------------------
# ★ 坑 70：**"已存在"的判断必须与"就绪"的判断用同一套标准**。
#   真机实测：`/root/Qwen3.5-0.8B-hf` 是个**空目录**（上次下载失败留下的），
#   而主流程用 `os.path.isdir(model_hf)` 判"已存在" → 直接跳过下载分支 →
#   **权重永远下载不了**（连续两轮 `du -sh` 都是 0）。
#   教训：跳过条件（幂等）与验收条件是同一个问题的两面，不能用两套宽严不同的标准。
# ---------------------------------------------------------------------
def _dir_nonempty(p):
    try:
        return os.path.isdir(p) and bool(os.listdir(p))
    except Exception:
        return False


def _dcp_release_ready(path):
    """A release tracker alone is written before conversion finishes."""
    root = os.path.abspath(path)
    tracker = os.path.join(root, "latest_checkpointed_iteration.txt")
    metadata = os.path.join(root, "release", ".metadata")
    try:
        with open(tracker, encoding="utf-8") as handle:
            return handle.read().strip() == "release" and os.path.isfile(metadata)
    except OSError:
        return False


def _conversion_receipt(path, source, target, migration=None):
    dcp = inspect_dcp(path)
    if dcp.get("status") != "local_structure_and_hashes":
        raise ValueError("DCP payload incomplete; cannot issue conversion receipt")
    return {"schema": "qwen35_0p8b_conversion.v3", "model_revision": source["model_revision"],
            "config_sha256": source["config_sha256"], "index_sha256": source["index_sha256"],
            "hf_file_inventory_sha256": source["file_inventory_sha256"],
            "dcp_file_inventory_sha256": dcp["file_inventory_sha256"],
            "migration_id": (migration or {}).get("migration_id"),
            "migration_manifest_sha256": (migration or {}).get("manifest_sha256"),
            "target_commit": target["target_commit"], "converter_sha256": target["converter_current_sha256"],
            "patch_identity": target["patch_identity"],
            "target_source_files_sha256": target["target_files_sha256"],
            "tie_weight_mapping": TIE_MAPPING.copy(),
            "mtp_source_keys": source["mtp_source_keys"],
            "reference_training_mtp_num_layers": 0, "dcp_mtp_reload_verified": False,
            "dcp_metadata_sha256": sha256(os.path.join(path, "release", ".metadata"))}


def _existing_conversion_verified(path, source, target, migration=None):
    receipt_path = os.path.join(path, "migration_conversion_receipt.json")
    if not _dcp_release_ready(path) or not os.path.isfile(receipt_path):
        return False
    try:
        with open(receipt_path, encoding="utf-8") as handle:
            saved = json.load(handle)
        return saved == _conversion_receipt(path, source, target, migration)
    except (OSError, ValueError, KeyError, json.JSONDecodeError):
        return False


def _weights_ready(p):
    """hf 权重就绪 = 目录非空 且 **必需文件齐备**。

    ★ 坑 94：原判据是 `any(f.endswith(('.safetensors','.bin','.json')))` —— **只要有一个 json 就算成功**，
      完全发现不了 processor/tokenizer 类文件缺失。真机因此报
      `ValueError: Processor was not found, please check and update your model file.`，
      而资产检查一路绿灯。**改为必需文件清单。**
    """
    if weights_missing_files(p):
        return False
    try:
        weight_headers_contract(p)
    except (WeightContractError, OSError):
        return False
    return True


def weights_missing_files(p):
    """返回缺失的必需文件项（供诊断输出）。"""
    if not os.path.isdir(p):
        return ["目录不可读"]
    return [name for name in REQUIRED_HF_FILES if not os.path.isfile(os.path.join(p, name))]


def _file_bytes_ok(p, expect):
    try:
        return os.path.isfile(p) and os.path.getsize(p) == expect
    except Exception:
        return False


def _json_samples(p):
    try:
        import json as _j
        return len(_j.load(open(p, encoding="utf-8")))
    except Exception:
        return None


def sh(cmd, timeout=120, background=False):
    if background:
        return subprocess.Popen(cmd, shell=True, stdout=subprocess.DEVNULL,
                                stderr=subprocess.DEVNULL, start_new_session=True)
    try:
        r = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=timeout)
        return (r.stdout or "") + (r.stderr or "")
    except Exception as e:
        return str(e)


def human(n):
    for u in ("B", "KB", "MB", "GB"):
        if n < 1024:
            return "%.1f %s" % (n, u)
        n /= 1024.0
    return "%.1f TB" % n


def file_size(p):
    try:
        return os.path.getsize(p)
    except Exception:
        return None


def count_images(d):
    try:
        return len([f for f in os.listdir(d) if f.lower().endswith((".jpg", ".jpeg", ".png"))])
    except Exception:
        return None


def count_samples(p):
    try:
        import json as _j
        return len(_j.load(open(p, encoding="utf-8")))
    except Exception:
        return None


def ensure(p, dry=False):
    if dry:
        return
    os.makedirs(p, exist_ok=True)


# =====================================================================
# ★ 鲁棒下载：重试 + 退避 + 多源 + 字节校验 + 降级留痕（坑 10/29）
# =====================================================================
try:
    # ★ 坑 153（pyflakes 静态扫描发现，2026-09-21）：`pip_install` 此前**从未被导入**，
    #   而第 362 行用到了它 —— 更要命的是那句调用被 `except Exception: pass` 包着，
    #   于是 `NameError` 被**静默吞掉**：依赖（jsonargparse / docstring-parser）
    #   根本没装，直到后面的 `convert_cli` 才以"缺模块"的面目失败。
    #   **静默 except 把"没装成"伪装成了"装过了"**。
    #   这里把 `pip_install` 一并纳入导入清单（`_envcompat` 里本就有这个函数）。
    from _envcompat import SOURCES, degrade, download, pip_install, python_exe  # noqa
    _HAVE_COMPAT = True
except Exception:                                     # 兼容单文件运行
    _HAVE_COMPAT = False
    SOURCES = {"modelscope": ["https://www.modelscope.cn"], "hf": ["https://hf-mirror.com"]}

    def pip_install(packages, exe=None, index_urls=None, **kw):
        """退化实现：`_envcompat` 不可用时用当前解释器直接装（保持同一语义）。"""
        cmd = [exe or sys.executable, "-m", "pip", "install", "--quiet"] + list(packages)
        return subprocess.run(cmd, capture_output=True, text=True)

DEG_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                        "..", "out", "degradations.json")


def _degrade(name, reason, severity="degraded"):
    if _HAVE_COMPAT:
        try:
            degrade(os.path.abspath(DEG_FILE), name, reason, severity=severity)
            return
        except Exception:
            pass
    print("[DEGRADED] %s — %s" % (name, reason))


def _retry(fn, tries=3, backoff=10, label=""):
    """带退避的重试。fn(i) 返回 True 表示成功。★ 应对 HTTP 429 限流与网络抖动。"""
    last = ""
    for i in range(1, tries + 1):
        try:
            if fn(i):
                return True, i
        except Exception as e:
            last = "%s: %s" % (type(e).__name__, e)
        if i < tries:
            wait = min(90, backoff * i)
            print("    (第 %d/%d 次尝试失败%s，退避 %ds 后重试)" % (i, tries, (" " + last) if last else "", wait))
            time.sleep(wait)
    return False, tries


def _ms_cli():
    """★ 坑 79：modelscope 下载走的是 **CLI**（`modelscope download ...`）。
    若该 CLI 不存在，原实现会用**退避重试 3 次**去跑一个不存在的命令 ——
    白等 3×15 秒，日志里只留一句含糊失败，然后静默转备源，
    让人误判成"网络问题"（真机上权重因此一直是 0）。
    现：先探测 CLI，缺失即**立刻**返回明确原因。"""
    try:
        r = subprocess.run("command -v modelscope", shell=True, capture_output=True,
                           text=True, timeout=20)
        return (r.stdout or "").strip() or None
    except Exception:
        return None


def _ms_download(args, label):
    """modelscope CLI 下载，带重试。返回 (ok, log_tail)。"""
    cli = _ms_cli()
    if not cli:
        return False, ("modelscope CLI 不存在（P4 下载依赖它）—— "
                       "安装: python3 -m pip install modelscope（bringup.sh phase2 亦会安装）")
    log = {"t": ""}

    def once(_i):
        try:
            r = subprocess.run(cli + " " + args, shell=True, capture_output=True,
                               text=True, timeout=14400)
            log["t"] = ((r.stdout or "") + (r.stderr or ""))[-500:]
            return r.returncode == 0
        except Exception as e:
            log["t"] = "%s: %s" % (type(e).__name__, e)
            return False

    ok, n = _retry(once, tries=3, backoff=15, label=label)
    return ok, log["t"]


def conversion_command(workdir, argv):
    """Quote Linux shell boundaries without changing converter arguments."""
    return "cd %s && %s" % (shlex.quote(workdir), shlex.join(argv))


def verified_migration(bundle, overlay, target_checkout):
    """Consume T03's validator; never derive a migration ID independently."""
    script = os.path.join(os.path.dirname(__file__), "22_migrate_qwen35.py")
    spec = importlib.util.spec_from_file_location("qwen35_migrate_for_assets", script)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    manifest, _ = module.validate_overlay(Path(bundle), Path(overlay))
    target = verify_target_checkout(target_checkout, require_patched=True)
    if (manifest.get("target", {}).get("commit") != target["target_commit"] or
            manifest.get("target", {}).get("converter_after_sha256") !=
            target["converter_current_sha256"]):
        raise ValueError("migration overlay differs from installed target checkout")
    manifest_path = Path(overlay) / "migration_manifest.json"
    return {"migration_id": manifest["migration_id"],
            "manifest_path": str(manifest_path), "manifest_sha256": sha256(manifest_path),
            "target_commit": target["target_commit"]}


def fetch_llava(path, dry=False):
    """llava_instruct_150k.json：modelscope 主源 → hf 直链备源，**字节级锚点校验**。"""
    anchor = ANCHORS["llava_json"]["bytes"]
    if dry:
        return False
    ensure(os.path.dirname(path))
    d = os.path.dirname(path)
    ok, tail = _ms_download(
        "download --dataset AI-ModelScope/LLaVA-Instruct-150K llava_instruct_150k.json "
        "--local_dir %s" % shlex.quote(d), "llava-modelscope")
    if os.path.isfile(path) and os.path.getsize(path) == anchor:
        return True
    # 备源：hf 直链（走 _envcompat.download → 支持续传 + 大小校验）
    print("  modelscope 未得到正确字节数，尝试 hf 备源 ...")
    for host in SOURCES.get("hf", ["https://hf-mirror.com"]):
        url = ("%s/datasets/liuhaotian/LLaVA-Instruct-150K/resolve/main/"
               "llava_instruct_150k.json" % host.rstrip("/"))
        if _HAVE_COMPAT:
            res = download(url, path, expect_bytes=anchor, retries=3)
            if res.get("ok") and res.get("bytes") == anchor:
                return True
            print("    %s → %s" % (host, res.get("error") or "size %s" % res.get("bytes")))
    _degrade("llava_download", "两个源均未取得锚点字节数 %d" % anchor)
    return False


def _hf_snapshot_download(**kwargs):
    """Load the optional official Hub client only for an authorized download."""
    from huggingface_hub import snapshot_download
    return snapshot_download(**kwargs)


def fetch_model(hf_dir, dry=False):
    """Request the fixed Hugging Face repository commit, without mirror revision guessing."""
    if dry:
        return False
    ensure(hf_dir)
    try:
        _hf_snapshot_download(repo_id="Qwen/Qwen3.5-0.8B", revision=MODEL_REVISION,
                              local_dir=hf_dir)
    except Exception as exc:
        _degrade("model_download", "HF 固定提交下载失败或 huggingface_hub 不可用: %s" % exc)
        return False
    if _weights_ready(hf_dir):
        return True
    _degrade("model_download", "HF 固定提交 %s 未取得完整权重；不回退浮动版本" % MODEL_REVISION)
    return False


def fetch_coco(coco_dir, dry=False):
    """COCO train2017（约 19GB）：modelscope 下载 + **大小/图片数双重校验** + 断点续传。"""
    if dry:
        return None
    ensure(coco_dir)
    zip_path = os.path.join(coco_dir, "train2017.zip")
    img_dir = os.path.join(coco_dir, "train2017")

    def is_complete(_i=None):
        return os.path.isdir(img_dir) and \
            len([f for f in os.listdir(img_dir) if f.lower().endswith((".jpg", ".png"))]) == \
            ANCHORS["coco_images"]["images"]

    if is_complete():
        return len(os.listdir(img_dir))
    ok, tail = _ms_download("download --dataset PAI/COCO2017 train2017.zip --local_dir %s"
                            % shlex.quote(coco_dir), "coco-modelscope")
    if not os.path.isfile(zip_path):
        _degrade("coco_download", "COCO 压缩包未取得（19GB，可能需要人工介入）", severity="blocked")
        return None
    # 解压（幂等；-o 覆盖）
    try:
        subprocess.run(["unzip", "-q", "-o", zip_path, "-d", coco_dir], timeout=7200)
    except Exception as e:
        print("  unzip 异常: %s" % e)
    n = len([f for f in os.listdir(img_dir) if f.lower().endswith((".jpg", ".png"))]) \
        if os.path.isdir(img_dir) else 0
    if n != ANCHORS["coco_images"]["images"]:
        _degrade("coco_images_incomplete", "解压后图片数 %d != 锚点 %d" % (n, ANCHORS["coco_images"]["images"]))
    return n


def main():
    ap = argparse.ArgumentParser(description="P4 权重与数据准备（含字节校验）")
    ap.add_argument("--env", default="out/probe/env.json")
    ap.add_argument("--out", default="out/assets")
    ap.add_argument("--model-dir", default="/root", help="权重存放目录")
    ap.add_argument("--data-dir", default="/root/data", help="数据存放目录")
    ap.add_argument("--msmm-dir", default="/root/MindSpeed-MM")
    ap.add_argument("--migration-bundle", default=None, help="T03 固定输入 bundle（与 --migration-overlay 成对）")
    ap.add_argument("--migration-overlay", default=None, help="T03 apply 输出目录（与 --migration-bundle 成对）")
    ap.add_argument("--skip-coco", action="store_true", help="跳过 COCO 图片下载（约 18GB）")
    ap.add_argument("--skip-convert", action="store_true", help="跳过数据转换")
    ap.add_argument("--no-download", action="store_true", help="只检查不下载（校验模式）")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    outdir = os.path.abspath(args.out)
    ensure(outdir, args.dry_run)
    result = {"schema": "migrator_assets.v2", "assets": {}, "checks": [], "warnings": [],
              "attempt_id": uuid.uuid4().hex,
              "model_revision_requested": MODEL_REVISION,
              "model_revision_observed": None,
              "migration_id": None,
              "migration_identity_state": "awaiting_T03_manifest_reference",
              "reload_verified": False}
    if bool(args.migration_bundle) != bool(args.migration_overlay):
        print("FATAL --migration-bundle 与 --migration-overlay 必须成对提供", file=sys.stderr)
        return 2
    migration = None
    if args.migration_bundle:
        try:
            migration = verified_migration(args.migration_bundle, args.migration_overlay,
                                           args.msmm_dir)
        except Exception as exc:  # T03 validator exposes its own MigrationError type.
            result["migration_identity_state"] = "invalid"
            result["migration_error"] = str(exc)
            if not args.dry_run:
                with open(os.path.join(outdir, "assets.json"), "w", encoding="utf-8") as handle:
                    json.dump(result, handle, ensure_ascii=False, indent=1)
            print("FATAL T03 migration/target 身份无效：%s" % exc, file=sys.stderr)
            return 3
        result.update(migration_id=migration["migration_id"],
                      migration_manifest_sha256=migration["manifest_sha256"],
                      migration_manifest_path=migration["manifest_path"],
                      migration_identity_state="validated_overlay_and_target")

    def record(key, path, extra=None, status="unknown"):
        rec = {"path": path, "exists": os.path.exists(path) if path else False, "status": status}
        sz = file_size(path) if (path and os.path.isfile(path)) else None
        if sz is not None:
            rec["bytes"] = sz
            rec["human"] = human(sz)
        if extra:
            rec.update(extra)
        result["assets"][key] = rec
        return rec

    def check(desc, ok, detail=""):
        result["checks"].append({"check": desc, "ok": bool(ok), "detail": detail})
        print("  %s  %s%s" % ("PASS" if ok else "FAIL", desc, (" — " + detail) if detail else ""))
        return ok

    print("== P4 权重与数据准备 ==")
    model_hf = os.path.join(args.model_dir, "Qwen3.5-0.8B-hf")
    model_dcp = os.path.join(args.model_dir, "Qwen3.5-0.8B-dcp")
    data_llava = os.path.join(args.data_dir, "llava", ANCHORS["llava_json"]["name"])
    data_coco_dir = os.path.join(args.data_dir, "coco")
    data_coco_img = os.path.join(data_coco_dir, "train2017")
    converted = os.path.join(args.data_dir, ANCHORS["converted_json"]["name"])

    # ---------------- 权重 ----------------
    print("\n[1/3] 模型权重")
    record("weight_hf", model_hf)
    record("weight_dcp", model_dcp)
    hf_ready_before = _weights_ready(model_hf)
    if not hf_ready_before and not args.no_download:
        print("  从 Hugging Face 固定仓库提交下载 hf 权重...")
        if not args.dry_run:
            if fetch_model(model_hf):
                print("  权重就绪: %s" % model_hf)
    elif hf_ready_before:
        print("  hf 权重已存在且内容就绪，跳过下载")
    elif os.path.isdir(model_hf):
        # ★ 坑 70：目录存在但为空/无权重文件 —— 曾经因此**永久跳过下载**（权重一直是 0）
        print("  hf 权重目录存在但**内容不就绪**（空目录/无权重文件）→ 将重新下载")
        result["warnings"].append("weight_hf 目录非空但无权重文件，已触发重新下载")
    else:
        result["warnings"].append("hf 权重缺失且 --no-download：请先下载 Qwen/Qwen3.5-0.8B（modelscope）")

    hf_inventory = inspect_hf(model_hf)
    result["hf_identity"] = hf_inventory
    result["metadata_revision_observed"] = hf_inventory.get("metadata_revision")
    result["conversion"] = {"converter": "Qwen35Converter", "model_size": "0.8B",
                            "tie_weight_mapping": TIE_MAPPING.copy(), "status": "not_run"}
    if not _dcp_release_ready(model_dcp) and not args.no_download:
        if _dir_nonempty(model_dcp):
            result["conversion"]["status"] = "blocked_partial_dcp"
            check("DCP 输出目录可安全写入", False, "已有不完整内容，使用新输出路径人工核查")
        elif hf_inventory["status"] == "local_complete":
            try:
                target_identity = verify_target_checkout(args.msmm_dir, require_patched=True)
                header_report = weight_headers_contract(model_hf)
            except (TargetIdentityError, WeightContractError, OSError) as exc:
                result["conversion"].update(status="blocked", error=str(exc),
                                            validation_level="source_target_or_headers_failed")
                check("固定目标源码、0.8B 权重索引及全部分片头", False, str(exc))
            else:
                result["conversion"].update(status="headers_checked",
                                            validation_level=header_report["validation_level"],
                                            source_revision=header_report["model_revision"],
                                            indexed_keys=header_report["indexed_keys"],
                                            header_keys=header_report["header_keys"],
                                            mtp_source_keys=header_report["mtp_source_keys"],
                                            reference_training_mtp_num_layers=0,
                                            dcp_mtp_reload_verified=False,
                                            target_commit=target_identity["target_commit"],
                                            converter_sha256=target_identity["converter_current_sha256"],
                                            patch_identity=target_identity["patch_identity"],
                                            hf_file_inventory_sha256=hf_inventory["file_inventory_sha256"],
                                            payload_hashes_checked=True,
                                            upstream_payload_same_bytes="unverified",
                                            model_state_dict_checked=False)
                check("固定目标源码、0.8B 权重索引及全部分片头", True,
                      "仅校验头部，未验证 tensor payload 或模型数值")
                print("  转换 hf → DCP（Qwen35Converter，带 tied lm_head 映射）...")
            if result["conversion"]["status"] == "headers_checked" and _HAVE_COMPAT \
                    and not args.dry_run:
                # ★ 坑 100：`convert_cli` 需要**两个**依赖，缺一不可 ——
                #   `jsonargparse`（第 8 行 import）与 `docstring-parser`（jsonargparse 的
                #   可选依赖，但在此路径上被强制要求）。
                #   我上一轮"用 jsonargparse 替换 docstring_parser"是**错的**，两者都要。
                #   pip 名与模块名不同：pip `docstring-parser` ↔ module `docstring_parser`。
                for extra in ("jsonargparse", "docstring-parser"):
                    try:
                        pip_install([extra])
                    except Exception:
                        pass
            # ★ 坑 98/102：这条命令的参数名我错了 4 次，最终以**实际入口的 `--help`** 定案：
            #     ① `--load_path/--save_path`（凭记忆猜）              → 错
            #     ② `--hf_dir/--dcp_dir`（旧机器 a3_conv2.sh，**正确**）→ 被我自己改掉
            #     ③ `--hf-dir/--dcp-dir`（读 `checkpoint/common/hf_to_dcp.py` 源码）→ 错
            #     ④ `--hf_dir/--dcp_dir`（`convert_cli.py ... hf_to_dcp --help` 打印）→ **正确**
            #   根因：`hf_to_dcp.py` 是**独立脚本**（自己的 argparse，用连字符），
            #   而我们调用的是 **`convert_cli.py` 子命令包装**（用下划线）。
            #   ——**同名功能的不同入口可以有不同参数约定**；只有"实际入口的 --help"是权威。
            #   纪律：`<实际调用方式> --help` 的输出优先于任何源码阅读与旧脚本记忆。
            if result["conversion"]["status"] == "headers_checked":
                argv = [
                    python_exe() if _HAVE_COMPAT else "python3", "-m", "checkpoint.convert_cli",
                    "Qwen35Converter", "hf_to_dcp", "--hf_dir", model_hf, "--dcp_dir", model_dcp,
                    "--tie_weight_mapping", json.dumps(TIE_MAPPING, separators=(",", ":"))]
                cmd = conversion_command(args.msmm_dir, argv)
                result["conversion"]["argv"] = argv
                print("  命令: %s" % cmd)
                if args.dry_run:
                    result["conversion"]["status"] = "planned_not_executed"
                else:
                    ok, attempts = _retry(lambda _i: subprocess.run(cmd, shell=True, timeout=3600).returncode == 0,
                                          tries=2, backoff=15, label="hf2dcp")
                    dcp_after = inspect_dcp(model_dcp)
                    ready = dcp_after["status"] == "local_structure_and_hashes"
                    result["conversion"].update(status="converted_structure_only" if ok and ready else "failed",
                                                attempts=attempts, dcp_release_structure_ready=ready)
                    if not ok or not ready:
                        check("hf→DCP 转换返回码与 release 结构", False,
                              "转换失败或缺少 tracker/release/.metadata")
                        _degrade("dcp_convert", "hf→dcp 转换未有效产出（P5 依赖）", severity="blocked")
                    else:
                        receipt = _conversion_receipt(model_dcp, hf_inventory, target_identity, migration)
                        receipt_path = os.path.join(model_dcp, "migration_conversion_receipt.json")
                        try:
                            with open(receipt_path, "x", encoding="utf-8") as handle:
                                json.dump(receipt, handle, ensure_ascii=False, indent=2)
                        except FileExistsError:
                            if not _existing_conversion_verified(model_dcp, hf_inventory, target_identity, migration):
                                result["conversion"]["status"] = "failed"
                                check("转换身份收据", False, "已有收据与本次输入/目标不一致")
                        except OSError as exc:
                            result["conversion"]["status"] = "failed"
                            check("转换身份收据", False, str(exc))
                        if result["conversion"]["status"] != "failed":
                            result["conversion"]["receipt_sha256"] = sha256(receipt_path)
                            check("hf→DCP 转换返回码、release 结构与身份收据", True,
                                  "完整权重重载待 T08")
    elif _dcp_release_ready(model_dcp):
        try:
            target_identity = verify_target_checkout(args.msmm_dir, require_patched=True)
            verified = hf_inventory["status"] == "local_complete" and \
                _existing_conversion_verified(model_dcp, hf_inventory, target_identity, migration)
        except (TargetIdentityError, WeightContractError, OSError) as exc:
            verified = False
            result["conversion"]["error"] = str(exc)
        result["conversion"]["status"] = "existing_verified_receipt" if verified else "existing_unverified"
        if verified:
            result["conversion"].update(source_revision=hf_inventory["model_revision"],
                                        mtp_source_keys=hf_inventory["mtp_source_keys"],
                                        reference_training_mtp_num_layers=0,
                                        dcp_mtp_reload_verified=False,
                                        target_commit=target_identity["target_commit"],
                                        converter_sha256=target_identity["converter_current_sha256"],
                                        receipt_sha256=sha256(os.path.join(model_dcp, "migration_conversion_receipt.json")))
        check("既有 DCP 转换身份收据", verified,
              "结构存在但无同源收据不得作为本轮可用 DCP" if not verified else "仅确认结构与转换身份")
    result["weight_validation_level"] = (
        "local_hf_and_dcp_hashes_receipt_no_reload"
        if result["conversion"]["status"] in ("converted_structure_only", "existing_verified_receipt")
        else "insufficient")
    dcp_inventory = inspect_dcp(model_dcp)
    result["dcp_identity"] = dcp_inventory
    record("weight_hf", model_hf, status="local_complete" if hf_inventory["status"] == "local_complete" else "missing")
    record("weight_dcp", model_dcp, status="structure_only" if dcp_inventory["status"] == "local_structure_and_hashes" else "missing")

    # ---------------- 数据下载 ----------------
    print("\n[2/3] 数据集")
    # llava 提示词（主源 modelscope → 备源 hf 直链；**字节级锚点强制校验**）
    if not _file_bytes_ok(data_llava, ANCHORS["llava_json"]["bytes"]) and not args.no_download:
        print("  下载 LLaVA-Instruct-150K（锚点 %d 字节）..." % ANCHORS["llava_json"]["bytes"])
        if not args.dry_run:
            if fetch_llava(data_llava):
                print("  llava json 就绪")
    llava_identity = inspect_llava(data_llava)
    result["llava_identity"] = llava_identity
    rec = record("llava_json", data_llava,
                 extra={"sha256": llava_identity.get("json_sha256"),
                        "order_sha256": llava_identity.get("order_sha256"),
                        "samples": llava_identity.get("row_count"),
                        "official_sha256": None, "official_same_bytes": "unverified"})
    if rec.get("bytes"):
        anchor = ANCHORS["llava_json"]["bytes"]
        check("llava_instruct_150k.json 预期字节数 = %d（仅大小校验）" % anchor,
              rec["bytes"] == anchor, "实际 %d" % rec["bytes"])
    else:
        result["warnings"].append("llava json 缺失：数据可比性无法确认")

    # COCO 图片（19GB：重试 + 幂等解压 + 图片数校验）
    n_img = count_images(data_coco_img) if os.path.isdir(data_coco_img) else None
    if (n_img is None or n_img != ANCHORS["coco_images"]["images"]) and \
            not args.skip_coco and not args.no_download:
        print("  下载 COCO2017 train2017.zip（约 19GB，带重试；建议后台执行）...")
        if not args.dry_run:
            n_img = fetch_coco(data_coco_dir)
    rec = record("coco_images", data_coco_img, extra={"images": n_img})
    if n_img is not None:
        check("COCO train2017 图片数 = %d" % ANCHORS["coco_images"]["images"],
              n_img == ANCHORS["coco_images"]["images"], "实际 %d" % n_img)
    elif args.skip_coco:
        result["warnings"].append("已按 --skip-coco 跳过图片下载：数据可比性降级为「子集/不可比」")

    # ---------------- 转换 ----------------
    print("\n[3/3] 数据格式转换")
    script_identity = converter_identity(args.msmm_dir)
    result["data_conversion"] = {"script": script_identity,
                                  "parameters": {"llava_json_path": data_llava,
                                                 "coco_path": data_coco_dir,
                                                 "output_json_path": converted},
                                  "source_json_sha256": llava_identity.get("json_sha256"),
                                  "status": "not_run"}
    if _json_samples(converted) != ANCHORS["converted_json"]["samples"] and not args.skip_convert and not args.no_download:
        script = script_identity["path"]
        if script_identity["status"] != "pinned":
            check("固定目标数据转换脚本", False, script_identity.get("error") or "脚本内容与固定提交不一致")
            result["data_conversion"]["status"] = "blocked_script_identity"
        elif llava_identity["status"] == "local_content_verified" and os.path.isdir(data_coco_img):
            cmd = conversion_command(args.msmm_dir, [
                python_exe() if _HAVE_COMPAT else "python3", script,
                "--llava_json_path", data_llava, "--coco_path", data_coco_dir,
                "--output_json_path", converted])
            result["data_conversion"]["command_argv"] = [
                python_exe() if _HAVE_COMPAT else "python3", script,
                "--llava_json_path", data_llava, "--coco_path", data_coco_dir,
                "--output_json_path", converted]
            print("  命令: %s" % cmd)
            if not args.dry_run:
                logs = []
                def run_conversion(_i):
                    log_path = os.path.join(outdir, "data-conversion-%s-attempt-%d.log" %
                                            (result["attempt_id"], _i))
                    with open(log_path, "wb") as stream:
                        try:
                            proc = subprocess.run(cmd, shell=True, timeout=7200,
                                                  stdout=stream, stderr=subprocess.STDOUT)
                            rc = proc.returncode
                        except subprocess.TimeoutExpired:
                            rc = None
                    with open(log_path, "rb") as stream:
                        stream.seek(max(0, os.path.getsize(log_path) - 2000))
                        tail = stream.read().decode("utf-8", errors="replace")
                    logs.append({"path": log_path, "sha256": sha256(log_path),
                                 "returncode": rc, "tail": tail})
                    return rc == 0
                ok, attempts = _retry(run_conversion, tries=2, backoff=20, label="convert")
                result["data_conversion"].update(status="generated" if ok else "failed",
                                                 attempts=logs, attempted_count=attempts)
            else:
                result["data_conversion"]["status"] = "planned_not_executed"
        else:
            result["warnings"].append("缺少 llava json 或 COCO 图片，无法转换")
    data_identity = inspect_data(converted, data_coco_dir)
    result["data_identity"] = data_identity
    if result["data_conversion"]["status"] == "generated":
        if (data_identity["status"] != "local_content_verified" or
                data_identity.get("row_count") != ANCHORS["converted_json"]["samples"]):
            result["data_conversion"]["status"] = "failed_output_validation"
            check("本次数据转换返回码与新产物身份", False,
                  data_identity.get("error") or "样本数/图片解码未达标")
        else:
            result["data_conversion"]["output_json_sha256"] = data_identity["json_sha256"]
            result["data_conversion"]["output_order_sha256"] = data_identity["order_sha256"]
    elif result["data_conversion"]["status"] == "failed":
        check("本次数据转换返回码", False, "转换失败；完整输出见本次日志清单")
    elif result["data_conversion"]["status"] == "not_run" and os.path.isfile(converted):
        result["data_conversion"]["status"] = "existing_local_identity_only"
    samples = data_identity.get("row_count")
    rec = record("converted_json", converted, extra={"samples": samples,
                  "sha256": data_identity.get("json_sha256"),
                  "order_sha256": data_identity.get("order_sha256"),
                  "status": data_identity.get("status")})
    if samples is not None:
        check("output_llava_coco_data.json 样本数 = %d" % ANCHORS["converted_json"]["samples"],
              samples == ANCHORS["converted_json"]["samples"], "实际 %d" % samples)

    # Local content hashes are useful for later comparison, but no official
    # dataset byte manifest is available to establish official same-bytes.
    if data_identity["status"] == "local_content_verified":
        level = "本地内容身份已记录；官方同字节未核对"
    else:
        level = "不可比（数据内容或引用图片未验证）"
    result["data_comparability"] = level
    result["official_data_identity"] = {"status": "unverified", "json_sha256": None,
                                        "image_manifest_sha256": None}

    print("\n=== 资产清单 ===")
    for k, v in result["assets"].items():
        print("  %-16s %-8s %s" % (k, v.get("status", "?"),
                                   v.get("human") or v.get("samples") or ""))
    print("数据可比性等级: %s" % level)
    if result["warnings"]:
        print("\n警告：")
        for w in result["warnings"]:
            print("  - %s" % w)
    fails = [c for c in result["checks"] if not c["ok"]]
    # ★ 坑 47：原实现只按 `checks` 判成败 —— `--no-download` 时不产生任何 check，
    #   于是"资产一个都没有"也会打印 `ASSETS_OK checks=0 fails=0` 并返回 0（**假阳性**）。
    #   现改为：**先按必需资产的实质完整性判定**，再叠加 check 失败。
    # ★ 坑 61：**不能只看"路径存在"** —— 真机上 `/root/Qwen3.5-0.8B-hf` 是个**空目录**，
    #   却被判为 `weight_hf ok` → 完整性门形同虚设。必须校验**内容**：
    #   目录非空且含权重文件 / 字节数与锚点一致 / 样本数与锚点一致 / 图片数与锚点一致。

    def substantive(key):
        """返回 (ok, detail)：对资产做**内容级**判定，而非存在性判定。"""
        rec = result["assets"].get(key, {})
        p = rec.get("path")
        if not p or not os.path.exists(p):
            return False, "不存在"
        if key in ("weight_hf", "weight_dcp"):
            if key == "weight_hf":
                if hf_inventory["status"] != "local_complete":
                    return False, "HF 文件、索引/分片或内容身份不足: %s" % (
                        hf_inventory.get("error") or ",".join(hf_inventory.get("missing_files", [])))
                return True, "完整本地文件哈希，非上游同字节证明"
            if dcp_inventory["status"] != "local_structure_and_hashes":
                return False, "DCP tracker/metadata/payload 不完整: %s" % (
                    dcp_inventory.get("error") or ",".join(dcp_inventory.get("missing_files", [])))
            if key == "weight_dcp" and result["conversion"]["status"] not in (
                    "converted_structure_only", "existing_verified_receipt"):
                return False, "DCP 缺少本次完整源/DCP payload 哈希绑定收据"
            return True, "DCP 本地 payload 哈希与本次收据一致；未重载"
        if key == "llava_json":
            if llava_identity["status"] != "local_content_verified":
                return False, llava_identity.get("error", "来源 JSON 内容未验证")
            n = rec.get("bytes")
            if not n:
                return False, "空文件"
            if n != ANCHORS["llava_json"]["bytes"]:
                return False, "字节数 %d != 锚点 %d" % (n, ANCHORS["llava_json"]["bytes"])
            if not rec.get("sha256"):
                return False, "来源 JSON 无法读取并计算内容哈希"
            return True, "本地字节哈希已记录；官方同字节未核对"
        if key == "converted_json":
            if data_identity["status"] != "local_content_verified":
                return False, data_identity.get("error", "JSON/图片内容未验证")
            s = data_identity["row_count"]
            if s != ANCHORS["converted_json"]["samples"]:
                return False, "样本数 %d != 锚点 %d" % (s, ANCHORS["converted_json"]["samples"])
            return True, "%d 样本、顺序及引用图片本地哈希已记录" % s
        if key == "coco_images":
            n = rec.get("images")
            if not n:
                return False, "0 张图片"
            if n != ANCHORS["coco_images"]["images"]:
                return False, "%d != 锚点 %d 张" % (n, ANCHORS["coco_images"]["images"])
            return True, "%d 张" % n
        return bool(rec.get("exists")), ""

    # ★ 坑 101：`weight_dcp` 曾被我归为"可选" —— 但 **mock 冒烟与 P5 训练都依赖它**
    #   （`TrainEngine.__init__` 会 `self.load()` 从 DCP 恢复）→ DCP 缺失时
    #   `ASSETS_OK` 依然成立，phase3 报 `[PASS] 资产齐备（含 DCP 权重）`，**又是一个假 PASS**。
    #   "可选"必须按**下游依赖**定义，而不是按"我们觉得它次要"。
    REQUIRED_ASSETS = ("weight_hf", "weight_dcp", "llava_json", "converted_json")
    OPTIONAL_ASSETS = ("coco_images",)
    missing = []
    for k in REQUIRED_ASSETS:
        ok_k, det = substantive(k)
        if not ok_k:
            missing.append("%s(%s)" % (k, det))
    opt_missing = []
    for k in OPTIONAL_ASSETS:
        ok_k, det = substantive(k)
        if not ok_k:
            opt_missing.append("%s(%s)" % (k, det))

    result["missing_required"] = missing
    result["missing_optional"] = opt_missing
    result["readiness"] = (("local_complete_official_unverified" if migration else
                            "local_complete_migration_unbound") if not missing and not fails
                           else "incomplete")
    if not args.dry_run:
        os.makedirs(outdir, exist_ok=True)
        with open(os.path.join(outdir, "assets.json"), "w", encoding="utf-8") as f:
            json.dump(result, f, ensure_ascii=False, indent=1)

    print("\n=== 完整性判定（内容级，非存在性）===")
    for k in REQUIRED_ASSETS + OPTIONAL_ASSETS:
        ok_k, det = substantive(k)
        print("  %-16s %-4s %s" % (k, "OK" if ok_k else "MISS", det))

    if missing:
        print("\nASSETS_INCOMPLETE missing_required=%s%s checks=%d fails=%d out=%s"
              % (";".join(missing),
                 (" optional_missing=%s" % ";".join(opt_missing)) if opt_missing else "",
                 len(result["checks"]), len(fails), os.path.join(outdir, "assets.json")))
        print("  → 补齐: python3 %s --data-dir <数据目录> --msmm-dir <MSMM目录>%s"
              % (os.path.basename(__file__), "" if args.no_download else ""))
        if args.no_download:
            print("  → 注意：本次为 --no-download（只检查不下载），因此资产缺失属预期；"
                  "但**不得**据此认为资产已就绪。")
        return 3
    if fails:
        print("\nASSETS_WARN checks=%d fails=%d out=%s"
              % (len(result["checks"]), len(fails), os.path.join(outdir, "assets.json")))
        return 3
    print("\nASSETS_OK checks=%d fails=0 out=%s"
          % (len(result["checks"]), os.path.join(outdir, "assets.json")))
    return 0


if __name__ == "__main__":
    sys.exit(main())
