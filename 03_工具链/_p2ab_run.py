#!/usr/bin/env python3
# -*- coding: utf-8 -*-
r"""P2 候选端到端确认实验（P3 规矩）。

设计（**看结果之前就已固定**）
--------------------------
· 臂：`N` = 基线（chunk_o.py **原文件**）；`C1` = 候选（打 `_p2_patch.py` 的补丁）
· 6 轮，**区组/顺序固定**（拉丁方，抵消单调漂移）：N, C1, C1, N, N, C1
· 每轮 100 步、GBS=8、dp2、同一配置、同一批次、串行（一次只跑一个训练任务）
· 每轮**按臂设置文件状态**（N ⇒ 从备份还原；C1 ⇒ 打补丁），并在每轮前后记 sha256
· 收工**强制还原**并核对 sha256 == 备份
· 失败传播：任一轮 rc≠0 或步数不全 ⇒ status.json 记 FAIL、退出码非零
· 产出 `runs_index.json`（与既有分析器同形状）⇒ 本地用**交付副本**的 59/62 判
"""
import hashlib
import json
import os
import subprocess
import sys
import time

import yaml

SKILL = "/root/qwen35-ascend-migrator"
MIND = "/root/MindSpeed-MM"
BASE_CFG = "/root/ops/qwen3_5_0_8B_mock_dp2_v3.yaml"
ENV_JSON = SKILL + "/out/probe/env.json"
TARGET = MIND + "/mindspeed_mm/fsdp/ops/gdn/triton/chunk_o.py"
BK = "/root/ops/fwbackup_p2/chunk_o.py.orig"
STEPS = 100
WORLD = 2
SEQ = ["N", "C1", "C1", "N", "N", "C1"]     # ★ 预先固定


def sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as fh:
        for c in iter(lambda: fh.read(1 << 20), b""):
            h.update(c)
    return h.hexdigest()


def env_for():
    e = os.environ.copy()
    e["LD_LIBRARY_PATH"] = ("/usr/local/Ascend/driver/lib64:/usr/local/Ascend/driver/lib64/common:"
                            "/usr/local/Ascend/driver/lib64/driver:"
                            "/usr/local/Ascend/ascend-toolkit/latest/lib64:" + e.get("LD_LIBRARY_PATH", ""))
    e["NON_MEGATRON"] = "true"
    e["TASK_QUEUE_ENABLE"] = "2"
    e["ASCEND_LAUNCH_BLOCKING"] = "0"
    e["PYTORCH_NPU_ALLOC_CONF"] = "expandable_segments:True"
    e["TRITON_CACHE_DIR"] = "/root/triton_cache"
    e["PYTHONPATH"] = MIND + ":" + e.get("PYTHONPATH", "")
    return e


def set_arm(arm, orig_sha):
    """把 chunk_o.py 置成该臂需要的状态，返回该状态的 sha256。

    ★★ **必须断言臂状态真的生效**（2026-09-22 事故）：
      上一版只**记录**了 `arm_file_sha256` 却没有**断言**，而 `_p2_patch.py` 当时
      因为"备份已存在"直接 SKIP ⇒ 候选臂跑的是**原文件** ⇒ "候选 vs 基线"退化成
      "基线 vs 基线"，整批实验作废。记录 ≠ 断言：**记录要靠人眼看，断言会自己拦下来**。
    """
    import shutil
    shutil.copy2(BK, TARGET)                      # 两个臂都先从**原始备份**起步
    if arm != "N":
        rc = subprocess.run([sys.executable, "/root/ops/_p2_patch.py", "apply", "--force"],
                            capture_output=True, text=True)
        if rc.returncode != 0:
            raise RuntimeError("打补丁失败: %s" % ((rc.stdout or "") + (rc.stderr or ""))[-200:])
    rc = subprocess.run([sys.executable, "-m", "py_compile", TARGET], capture_output=True)
    if rc.returncode != 0:
        raise RuntimeError("py_compile 失败: %s" % TARGET)
    now = sha(TARGET)
    if arm == "N" and now != orig_sha:
        raise RuntimeError("基线臂的文件 sha 与备份不一致（%s）" % now[:12])
    if arm != "N" and now == orig_sha:
        raise RuntimeError("★ 候选臂打完补丁后 sha 仍等于原文件 ⇒ 补丁没生效，拒绝用这一轮")
    return now


def main():
    ts = time.strftime("%Y%m%d_%H%M%S")
    root = "/root/ops/p2ab_%s" % ts
    os.makedirs(root, exist_ok=True)
    status = {"schema": "dsh.p2ab.status.v1", "root": root, "seq": SEQ, "steps": STEPS,
              "roles": {"N": "baseline(chunk_o.py 原文件)", "C1": "candidate(常量搬运设备化)"},
              "flow_finished": False, "verification_passed": False, "failures": [], "runs": []}
    print("P2AB_ROOT=%s" % root, flush=True)
    if not os.path.isfile(BK):
        print("P2AB_ABORT 缺备份 %s" % BK)
        status["failures"].append("NO_BACKUP")
        return finish(root, status, 1)
    orig_sha = sha(BK)
    doc0 = yaml.safe_load(open(BASE_CFG, encoding="utf-8"))
    runs = []
    try:
        for i, arm in enumerate(SEQ):
            tag = "%02d_%s" % (i, arm)
            case = os.path.join(root, tag)
            os.makedirs(case, exist_ok=True)
            try:
                arm_sha = set_arm(arm, orig_sha)
            except RuntimeError as exc:
                status["failures"].append("SET_ARM_%s_%s" % (tag, exc))
                print("### SET_ARM_FAIL %s: %s" % (tag, exc), flush=True)
                break
            cfg = dict(doc0)
            cfg["training"] = dict(doc0["training"])
            cfg["training"]["train_iters"] = STEPS
            cfg["training"]["save"] = os.path.join(case, "checkpoint")
            config = os.path.join(case, "config.yaml")
            with open(config, "w", encoding="utf-8") as fh:
                yaml.safe_dump(cfg, fh, sort_keys=False)
            log = os.path.join(case, "train.log")
            drv = os.path.join(case, "driver.log")
            cmd = [sys.executable, os.path.join(SKILL, "scripts", "50_train.py"),
                   "--config", config, "--env", ENV_JSON, "--log", log,
                   "--workdir", MIND, "--steps", str(STEPS), "--world-size", str(WORLD),
                   "--timeout", "1800", "--foreground"]
            t0 = time.time()
            with open(drv, "w", encoding="utf-8") as fh:
                p = subprocess.run(cmd, cwd=SKILL, env=env_for(), stdout=fh,
                                   stderr=subprocess.STDOUT, timeout=2400)
            wall = round(time.time() - t0, 1)
            txt = open(log, encoding="utf-8", errors="replace").read() if os.path.isfile(log) else ""
            n_iter = txt.count("elapsed time per iteration")
            rec = {"tag": tag, "arm": arm, "MM_CAND": ("P2CHUNKO" if arm != "N" else None),
                   "rc": p.returncode, "wall_seconds": wall,
                   "train_log": log, "driver_log": drv, "config": config,
                   "arm_file_sha256": arm_sha, "iteration_lines": n_iter,
                   "identity": {"boot": open("/proc/sys/kernel/random/boot_id").read().strip()}}
            runs.append(rec)
            print("RUN %s arm=%s rc=%s wall=%.0fs iter=%d sha=%s"
                  % (tag, arm, p.returncode, wall, n_iter, arm_sha[:12]), flush=True)
            if p.returncode != 0 or n_iter < STEPS:
                status["failures"].append("RUN_%s_RC_%d_ITER_%d" % (tag, p.returncode, n_iter))
                break
    finally:
        import shutil
        shutil.copy2(BK, TARGET)
        back = sha(TARGET)
        status["restored_sha256"] = back
        status["orig_sha256"] = orig_sha
        status["restored_ok"] = (back == orig_sha)
        print("RESTORE sha_match=%s sha=%s" % (back == orig_sha, back[:12]), flush=True)
        if not status["restored_ok"]:
            status["failures"].append("RESTORE_MISMATCH")
    status["runs"] = runs
    with open(os.path.join(root, "runs_index.json"), "w", encoding="utf-8") as fh:
        json.dump({"out": root, "steps": STEPS, "runs": runs,
                   "identity": {"boot": open("/proc/sys/kernel/random/boot_id").read().strip()}},
                  fh, ensure_ascii=False, indent=1)
    return finish(root, status, 0 if not status["failures"] else 1)


def finish(root, status, code):
    status["flow_finished"] = True
    status["verification_passed"] = bool(code == 0 and not status["failures"])
    status["exit_code"] = code
    with open(root + "/status.json", "w", encoding="utf-8") as fh:
        json.dump(status, fh, ensure_ascii=False, indent=1)
    word = ("P2AB_VERIFIED" if status["verification_passed"] else "P2AB_FLOW_DONE_FAILED")
    with open(root + "/all.done", "w") as fh:
        fh.write(word + "\n")
    print("### %s exit=%d failures=%s" % (word, code, status["failures"]), flush=True)
    return code


if __name__ == "__main__":
    sys.exit(main())
