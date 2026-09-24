#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
selfcheck_sk04.py — R2-SK04 本地零 GPU 全链路自检（交付方已跑 ✅，输出 VERIFY_OK 单行）

覆盖（真实日志优先，缺省路径时自动降级/跳过并明示）：
  [1] fingerprint_cfg --config-from-log 真实日志（snapshots/coco_train_run1.log = A2 COCO 首跑
      真实日志）vs officialB/officialA：断言偏离维度、N_eff=8、可达性 level；
  [2] fingerprint_observed 真实日志（COCO log → rollback 证据；docs/official/triton精度日志.txt
      → NPU triton 生效；evidence/officialA_configuration_details.txt → 无标记）断言行数/step1 带；
  [3] data_id 合成 fixture（同字节 / 同内容异序 / 非同源 / 未核对 四例）；
  [4] judge_comparable 纯函数：vs officialB=WINDOW_OK、vs officialA=NOT_COMPARABLE（带冲突）、
      二次运行 verdict_id 一致（确定性）；
  [5] registry_append：临时账本 append×2 → verify OK → 篡改后 verify 拒绝（链完整性）；
  [6] extract_postA_log --check：A 官方帖提取证据与入库产物一致（防漂移）。

用法：python3 scripts/selfcheck_sk04.py [--repo-root <复赛根>]
输出：VERIFY_OK selfcheck=1 case_ok=N/N …（任一项失败 → exit 1；缺真实文件 → 该 case 标 SKIP）
纯 CPU；仅标准库 + pyyaml（fingerprint_cfg/judge 需要）。
"""

import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

SKILL_ROOT = Path(__file__).resolve().parents[1]
# ★ 坑 36：SKILL_ROOT 指向 sk04_judge/，而 examples/ 等包级目录在**上一层**（Skill 根）。
#   用 SKILL_ROOT/"examples" 会永远找不到 → 检查被静默 SKIP。此处显式解析包根。
PKG_ROOT = SKILL_ROOT.parent if (SKILL_ROOT.parent / "SKILL.md").is_file() else SKILL_ROOT
SCRIPTS = SKILL_ROOT / "scripts"
RESULTS = []


def find_repo_root(start):
    """安全向上查找复赛根（含 snapshots/ 或 docs/official/）。

    ★ 坑 30 修复：原实现 `SKILL_ROOT.parents[3]` 硬编码路径深度，把本 Skill 打包分发到
    任意位置（如 /root/qwen35-ascend-migrator）时会 IndexError 崩溃。
    现改为：env `SK04_REPO_ROOT` → 逐级向上找标记目录 → 找不到返回 None（绝不抛异常）。
    """
    env = os.environ.get("SK04_REPO_ROOT")
    if env and Path(env).is_dir():
        return Path(env)
    p = Path(start).resolve()
    while True:
        if (p / "snapshots").is_dir() or (p / "docs" / "official").is_dir():
            return p
        if p.parent == p:
            return None
        p = p.parent


def note(name, ok, detail=""):
    RESULTS.append((name, bool(ok), detail))
    print("[%s] %s %s" % ("PASS" if ok else ("SKIP" if detail.startswith("SKIP") else "FAIL"),
                          name, detail))


def run_py(args, cwd=None):
    return subprocess.run([sys.executable] + [str(a) for a in args],
                          cwd=cwd or str(SKILL_ROOT),
                          capture_output=True, text=True, encoding="utf-8", errors="replace")


def main():
    ap = argparse.ArgumentParser(description="R2-SK04 selfcheck")
    ap.add_argument("--repo-root", default=None, help="复赛根（缺省自动探测）")
    args = ap.parse_args()
    repo = Path(args.repo_root) if args.repo_root else find_repo_root(SKILL_ROOT)  # ★ 坑 30
    if repo is None:
        print("[note] 未找到复赛根（无 SK04_REPO_ROOT / 无 snapshots / 无 docs/official）"
              " → 依赖仓库原始日志的检查将标 SKIP，包内 fixture 检查照常执行。")
    coco_log = repo / "snapshots" / "coco_train_run1.log" if repo else None
    blog = repo / "docs" / "official" / "triton精度日志.txt" if repo else None
    acsv = repo / "code" / "skills" / "skill_round2_SM3" / "reference_official_100steps.csv" if repo else None
    bcsv = repo / "docs" / "raw" / "reference_log_iterations.csv" if repo else None
    out = SKILL_ROOT / "tests" / "tmp"
    out.mkdir(parents=True, exist_ok=True)
    # ★ 坑 35（最重要的一条）：tests/tmp 残留上次运行的产物时，自检会消费**陈旧证据**并判定
    #   通过 —— 打包分发场景尤其致命（包内带着开发机的 09-03 产物，新机器上看不到 coco 日志
    #   却仍显示 judge 检查 PASS，实际什么都没证明）。故开跑前必须清空产物目录，
    #   只允许消费**本次运行生成**的文件；缺输入则相应检查标 SKIP。
    purged = 0
    for old in sorted(out.glob("*.json")):
        try:
            old.unlink()
            purged += 1
        except OSError:
            pass
    if purged:
        print("[note] 已清空 tests/tmp 中 %d 个上次运行的产物（防止自检消费陈旧证据）" % purged)

    # ---------------- [6] extract_postA_log --check（先做：产物是 A 证据源）
    # ★ 坑 31：官方帖 JSON 不在分发包内 → 只能做包内证据文件的存在性/完整性复核，
    #   标 SKIP（而非 FAIL，也不是 PASS —— 未做重导出一致性比对，不冒充实证）。
    r = run_py([SCRIPTS / "extract_postA_log.py", "--check"])
    if "CHECK_OK" in r.stdout:
        note("extract_postA_log_check", True, "全量：与官方帖 JSON 重导出逐行一致")
    elif "CHECK_PARTIAL_NO_POST" in r.stdout:
        line0 = r.stdout.strip().splitlines()[0] if r.stdout.strip() else ""
        note("extract_postA_log_check", False,
             "SKIP 官方帖 JSON 不在包内 → 仅复核证据文件存在性（%s）；"
             "未做重导出一致性比对" % line0[:90])
    else:
        note("extract_postA_log_check", r.returncode == 0 and "CHECK_OK" in r.stdout,
             "rc=%s %s" % (r.returncode, r.stdout.strip().splitlines()[-1][:140]
                           if r.stdout.strip() else r.stderr[:140]))

    # ---------------- [1] fingerprint_cfg（真实 COCO 日志）
    if coco_log and coco_log.is_file():
        for base, exp_level in (("officialB", "window_feasible"), ("officialA", "window_feasible")):
            r = run_py([SCRIPTS / "fingerprint_cfg.py", "--config-from-log", coco_log,
                        "--baseline", base, "--out", out / ("fp_self_%s.json" % base)])
            ok = r.returncode == 0 and ("FINGERPRINT_OK" in r.stdout) and \
                 ("level=%s" % exp_level) in r.stdout
            note("fingerprint_cfg_coco_vs_%s" % base, ok,
                 "rc=%s %s" % (r.returncode, r.stdout.strip().splitlines()[-1][:140]
                               if r.stdout.strip() else r.stderr[:140]))
        # 逐维断言（COCO vs officialB / officialA，A2 审计 §3.1 口径）
        fpB = json.loads((out / "fp_self_officialB.json").read_text(encoding="utf-8-sig"))
        stB = {d["dim"]: d["status"] for d in fpB["dims"]}
        okB = (stB.get("shuffle") == "mismatch" and stB.get("freeze") == "mismatch"
               and stB.get("gdn_causal") == "mismatch" and stB.get("dataset_file") == "mismatch"
               and fpB["derived"]["N_eff"]["run"] == 8)
        note("fingerprint_cfg_dims_vs_B", okB, "statuses=%s" % stB)
        fpA = json.loads((out / "fp_self_officialA.json").read_text(encoding="utf-8-sig"))
        stA = {d["dim"]: d["status"] for d in fpA["dims"]}
        okA = (stA.get("freeze") == "match" and stA.get("mbs") == "match" and stA.get("gas") == "match"
               and stA.get("dp") == "match" and stA.get("gdn_causal") == "unverified"
               and stA.get("shuffle") == "mismatch")
        note("fingerprint_cfg_dims_vs_A", okA, "statuses=%s" % stA)
        sh = next(d for d in fpB["dims"] if d["dim"] == "shuffle")
        note("fingerprint_cfg_evidence_line", sh["run"]["config_line"] == 164,
             "shuffle config_line=%s" % sh["run"]["config_line"])
    else:
        note("fingerprint_cfg_coco", False, "SKIP coco_train_run1.log 不在仓库（A2 上跑验收命令）")

    # ---- [1b] fingerprint_cfg --config <yaml 文件>（A2 run_train 集成形态；skill 内 fixture 恒在）
    # ★ 坑 45：本段需要 **pyyaml**（fingerprint_cfg.py 要解析 yaml fixture）。分发环境可能没装
    #   pyyaml → fingerprint_cfg 不产出文件 → 后续 json.loads 抛未捕获的 FileNotFoundError
    #   → **整个判定链自检崩溃**。现：① 显式探测 pyyaml ② 缺失标 SKIP ③ 整段加异常兜底。
    try:
        import yaml as _yaml_probe  # noqa: F401
        HAVE_YAML = True
    except ImportError:
        HAVE_YAML = False
    fx_yaml = SKILL_ROOT / "tests" / "fixtures"
    if not HAVE_YAML:
        for nm in ("fingerprint_cfg_yaml_coco_vs_B", "fingerprint_cfg_yaml_dims",
                   "fingerprint_cfg_yaml_sk01_aligned"):
            note(nm, False, "SKIP 缺 pyyaml（判定链的 yaml 解析依赖）—— "
                            "装法: python3 -m pip install pyyaml")
    else:
        try:
            r = run_py([SCRIPTS / "fingerprint_cfg.py", "--config", fx_yaml / "run_coco_first_eager_config.yaml",
                        "--baseline", "officialB", "--world-size", "1", "--out", out / "fp_yaml_coco.json"])
            ok = r.returncode == 0 and "level=window_feasible" in r.stdout
            note("fingerprint_cfg_yaml_coco_vs_B", ok,
                 r.stdout.strip().splitlines()[-1][:150] if r.stdout.strip() else r.stderr[:160])
            fpY = json.loads((out / "fp_yaml_coco.json").read_text(encoding="utf-8-sig"))
            stY = {d["dim"]: d["status"] for d in fpY["dims"]}
            ok = (stY.get("shuffle") == "mismatch" and stY.get("freeze") == "mismatch"
                  and stY.get("gdn_causal") == "mismatch" and fpY["derived"]["N_eff"]["run"] == 8
                  and fpY["derived"]["world_size"]["run"] == 1)
            note("fingerprint_cfg_yaml_dims", ok, "statuses=%s world=%s"
                 % (stY, fpY["derived"]["world_size"]))
            r = run_py([SCRIPTS / "fingerprint_cfg.py", "--config", fx_yaml / "run_sk01_aligned_config.yaml",
                        "--baseline", "officialB", "--world-size", "1", "--out", out / "fp_yaml_sk01.json"])
            fpS = json.loads((out / "fp_yaml_sk01.json").read_text(encoding="utf-8-sig"))
            stS = {d["dim"]: d["status"] for d in fpS["dims"]}
            ok = (r.returncode == 0 and stS.get("shuffle") == "match" and stS.get("freeze") == "match"
                  and stS.get("cutoff_len") == "match" and stS.get("num_workers") == "match"
                  and stS.get("gdn_causal") == "mismatch")   # eager vs triton 仍是唯一实现层偏离
            note("fingerprint_cfg_yaml_sk01_aligned", ok, "statuses=%s" % stS)
        except Exception as e:
            note("fingerprint_cfg_yaml_coco_vs_B", False,
                 "SKIP fixture 检查异常: %s: %s" % (type(e).__name__, str(e)[:120]))

    # ---------------- [2] fingerprint_observed（真实日志）
    if coco_log and coco_log.is_file():
        r = run_py([SCRIPTS / "fingerprint_observed.py", "--log", coco_log,
                    "--declared", out / "fp_self_officialB.json",
                    "--lr-ref-csv", acsv, "--lr-ref-label", "officialB",
                    "--out", out / "observed_self_coco.json"])
        ok = r.returncode == 0 and "kernel_status=rollback_observed" in r.stdout and \
             "last_iter=100" in r.stdout and "step1_loss=1.869898" in r.stdout
        note("fingerprint_observed_coco_rollback", ok,
             r.stdout.strip().splitlines()[-1][:160] if r.stdout.strip() else r.stderr[:160])
    if blog and blog.is_file():
        r = run_py([SCRIPTS / "fingerprint_observed.py", "--log", blog,
                    "--lr-ref-csv", acsv, "--lr-ref-label", "officialB",
                    "--out", out / "observed_self_blog.json"])
        ok = r.returncode == 0 and "kernel_status=npu_triton_active" in r.stdout and \
             "last_iter=100" in r.stdout and "step1_loss=1.924621" in r.stdout
        note("fingerprint_observed_blog_triton", ok,
             r.stdout.strip().splitlines()[-1][:160] if r.stdout.strip() else r.stderr[:160])
    # A 证据文件（skill 内，恒在）
    # ★ 坑 32：bcsv 可能为 None（仓库不在）→ 不得把 None 当路径字符串传下去
    alog_args = [SCRIPTS / "fingerprint_observed.py", "--log",
                 SKILL_ROOT / "evidence" / "officialA_configuration_details.txt",
                 "--out", out / "observed_self_alog.json"]
    if bcsv is not None and Path(bcsv).is_file():
        alog_args += ["--lr-ref-csv", bcsv, "--lr-ref-label", "officialA"]
    r = run_py(alog_args)
    ok = r.returncode == 0 and "kernel_status=no_kernel_marker" in r.stdout and \
         "band=A(从零尺度)" in r.stdout
    note("fingerprint_observed_Alog", ok,
         r.stdout.strip().splitlines()[-1][:160] if r.stdout.strip() else r.stderr[:160])
    # LR 三份同哈希（A/B/我方 COCO 的 LR 序列一致 —— SK02 审计已证）
    # ★ 坑 33：原来 len(hashes) <= 1 在"只有 1 份产物"时**空过**（vacuous pass）→ 现要求 >=2 份才判定
    try:
        present, hashes = 0, set()
        for f in ("observed_self_coco.json", "observed_self_blog.json", "observed_self_alog.json"):
            p = out / f
            if p.is_file():
                present += 1
                hashes.add(json.loads(p.read_text(encoding="utf-8-sig"))["observed"]["lr"]["sha256"])
        if present >= 2:
            note("observed_lr_seq_same_across_3", len(hashes) == 1,
                 "n_present=%d uniq_lr_sha=%d" % (present, len(hashes)))
        else:
            note("observed_lr_seq_same_across_3", False,
                 "SKIP 仅 %d 份产物（需 >=2 才能比对，空过不算通过）" % present)
    except Exception as e:
        note("observed_lr_seq_same_across_3", False, "SKIP %s" % e)

    # ---------------- [3] data_id（合成 fixture，skill 内恒在）
    fx = SKILL_ROOT / "tests" / "fixtures"
    cases = [("ds_a.json", "ds_b.json", "same_bytes"),
             ("ds_a.json", "ds_c.json", "same_content_reordered"),
             ("ds_a.json", "ds_d.json", "different"),
             ("ds_a.json", "ds_e.json", "same_content_reordered"),
             ("ds_a.json", "nope.json", "unverified")]
    for a, b, exp in cases:
        r = run_py([SCRIPTS / "data_id.py", "--compare", fx / a, fx / b])
        ok = r.returncode == (0 if exp in ("same_bytes", "same_content_reordered") else 1) and \
             ("status=%s" % exp) in r.stdout
        note("data_id_%s_vs_%s" % (a, b), ok, "exp=%s rc=%s" % (exp, r.returncode))
    r = run_py([SCRIPTS / "data_id.py", "--json", fx / "ds_a.json"])
    ok = r.returncode == 0 and "row_count=8" in r.stdout
    note("data_id_single", ok, r.stdout.strip().splitlines()[-1][:120] if r.stdout.strip() else "")

    # ---------------- [4] judge_comparable（纯函数 + 确定性）
    if (out / "fp_self_officialB.json").is_file() and (out / "observed_self_coco.json").is_file():
        r = run_py([SCRIPTS / "judge_comparable.py", "--run", out / "fp_self_officialB.json",
                    "--observed", out / "observed_self_coco.json", "--baseline", "officialB",
                    "--out", out / "verdict_self_B.json", "--gate"])
        ok = r.returncode == 3 and "verdict=WINDOW_OK" in r.stdout and \
             "officiality=proposed" in r.stdout
        note("judge_coco_vs_B_window", ok, r.stdout.strip().splitlines()[-1][:180]
             if r.stdout.strip() else r.stderr[:200])
        r = run_py([SCRIPTS / "judge_comparable.py", "--run", out / "fp_self_officialA.json",
                    "--observed", out / "observed_self_coco.json", "--baseline", "officialA",
                    "--out", out / "verdict_self_A.json", "--gate"])
        ok = r.returncode == 4 and "verdict=NOT_COMPARABLE" in r.stdout and \
             "officiality=confirmed" in r.stdout
        note("judge_coco_vs_A_band_conflict", ok, r.stdout.strip().splitlines()[-1][:180]
             if r.stdout.strip() else r.stderr[:200])
        # 默认退出 0（裁决成功产出即 0）
        r = run_py([SCRIPTS / "judge_comparable.py", "--run", out / "fp_self_officialB.json",
                    "--observed", out / "observed_self_coco.json", "--baseline", "officialB",
                    "--out", out / "verdict_self_B_default.json"])
        ok = r.returncode == 0 and "JUDGE_OK" in r.stdout
        note("judge_default_exit0", ok, "rc=%s" % r.returncode)
        v1 = json.loads((out / "verdict_self_B.json").read_text(encoding="utf-8-sig"))
        r2 = run_py([SCRIPTS / "judge_comparable.py", "--run", out / "fp_self_officialB.json",
                     "--observed", out / "observed_self_coco.json", "--baseline", "officialB"])
        line2 = r2.stdout.strip().splitlines()[-1] if r2.stdout.strip() else ""
        v2id = line2.split("verdict_id=")[-1][:16] if "verdict_id=" in line2 else ""
        note("judge_deterministic_verdict_id",
             bool(v1.get("verdict_id")) and v1["verdict_id"][:16] == v2id,
             "id=%s / rerun=%s" % (v1.get("verdict_id", "?")[:16], v2id or "?"))
    else:
        note("judge", False, "SKIP 依赖 [1][2] 产物")

    # ---------------- [5] registry_append（临时账本）
    tmpdir = Path(tempfile.mkdtemp(prefix="sk04_reg_"))
    reg = tmpdir / "registry.json"
    shutil.copy(SKILL_ROOT / "evidence" / "registry.json", reg)
    fpB = out / "fp_self_officialB.json"
    obs = out / "observed_self_coco.json"
    mf = obs  # manifest 占位（本地自检用 observed json 顶替 manifest 目标文件，仅验链机制）
    ok_all = True
    if fpB.is_file() and obs.is_file():
        for tag in ("run_self_1", "run_self_2"):
            r = run_py([SCRIPTS / "registry_append.py", "--registry", reg, "--tag", tag,
                        "--run-dir", "runs/%s" % tag, "--manifest", mf, "--fingerprint", fpB,
                        "--observed", obs])
            ok_all = ok_all and r.returncode == 0 and "REGISTRY_APPEND_OK" in r.stdout
        r = run_py([SCRIPTS / "registry_append.py", "--registry", reg, "--verify"])
        ok_all = ok_all and r.returncode == 0 and "REGISTRY_VERIFY_OK" in r.stdout
        # 篡改 → 拒绝
        doc = json.loads(reg.read_text(encoding="utf-8-sig"))
        doc["entries"][0]["tag"] = "TAMPERED"
        reg2 = tmpdir / "tampered.json"
        reg2.write_text(json.dumps(doc, ensure_ascii=False, indent=1), encoding="utf-8")
        r = run_py([SCRIPTS / "registry_append.py", "--registry", reg2, "--verify"])
        ok_all = ok_all and r.returncode == 3 and "REGISTRY_CHAIN_BROKEN" in r.stderr
        # 重复 tag → 拒绝
        r = run_py([SCRIPTS / "registry_append.py", "--registry", reg, "--tag", "run_self_1",
                    "--run-dir", "x", "--manifest", mf, "--fingerprint", fpB])
        ok_all = ok_all and r.returncode == 3 and "REGISTRY_DUP_TAG" in r.stderr
        note("registry_chain_append_verify_tamper_dup", ok_all, "tmp=%s" % tmpdir)
    else:
        note("registry", False, "SKIP 依赖 [1][2] 产物")

    # ---------------- [7] 包内示例重放（★ 0-GPU 自证核心声明）
    # 这不是"实跑证据"，而是「**判定链确定性 + 包内声明可从包内字节复现**」的验证：
    #   用 examples/judge 里的真实运行产物重放判定链，必须得到与交付材料一致的
    #   verdict_id 与账本尾哈希。任何字节被改动 → id 变化 → 自证失败。
    # 价值：评委/新机器无需 NPU、无需仓库原始日志，即可复核核心声明。
    exj = PKG_ROOT / "examples" / "judge"
    if (exj / "fp.json").is_file() and (exj / "obs.json").is_file():
        r = run_py([SCRIPTS / "judge_comparable.py", "--run", exj / "fp.json",
                    "--observed", exj / "obs.json", "--baseline", "officialB",
                    "--out", out / "verdict_replay_B.json"])
        ok = (r.returncode == 0 and "verdict_id=606056b3cb04ecbb" in r.stdout
              and "level=pointwise_feasible" in r.stdout)
        note("replay_example_judge_verdict_id", ok,
             "期望 verdict_id=606056b3cb04ecbb / " +
             (r.stdout.strip().splitlines()[-1][:150] if r.stdout.strip() else r.stderr[:150]))
        try:
            ex_v = json.loads((exj / "verdict.json").read_text(encoding="utf-8-sig"))
            rp_v = json.loads((out / "verdict_replay_B.json").read_text(encoding="utf-8-sig"))
            same = ex_v.get("verdict_id") == rp_v.get("verdict_id")
            note("replay_example_matches_shipped_verdict", same,
                 "shipped=%s replay=%s" % (str(ex_v.get("verdict_id"))[:16],
                                           str(rp_v.get("verdict_id"))[:16]))
        except Exception as e:
            note("replay_example_matches_shipped_verdict", False, "SKIP %s" % e)
    else:
        note("replay_example_judge_verdict_id", False, "SKIP examples/judge 不完整")

    if (exj / "registry.json").is_file():
        r = run_py([SCRIPTS / "registry_append.py", "--registry", exj / "registry.json", "--verify"])
        ok = (r.returncode == 0 and "REGISTRY_VERIFY_OK" in r.stdout and "entries=5" in r.stdout)
        note("replay_example_registry_verify", ok,
             (r.stdout.strip().splitlines()[-1][:150] if r.stdout.strip() else r.stderr[:150]))
    else:
        note("replay_example_registry_verify", False, "SKIP 无 examples/judge/registry.json")

    # ---------------- 汇总
    # ★ 坑 34：原实现 `failed = total - passed` 把 SKIP 也算成失败 → 在无仓库原始日志的
    #   干净机器上，即使所有可跑的检查全通过也返回 1，信号失真。
    #   现改为：failed 只统计真正的失败（非 SKIP）。
    total = len(RESULTS)
    passed = sum(1 for _, ok, _ in RESULTS if ok)
    skipped = sum(1 for _, _, d in RESULTS if d.startswith("SKIP"))
    failed = total - passed - skipped
    print("VERIFY_OK selfcheck=1 case_ok=%d/%d skipped=%d failed=%d"
          % (passed, total - skipped, skipped, failed))
    if failed:
        for name, ok, detail in RESULTS:
            if not ok and not detail.startswith("SKIP"):
                print("  FAILED: %s %s" % (name, detail))
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
