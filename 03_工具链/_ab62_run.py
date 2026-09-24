#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""§4-2 端到端 A/B —— 步骤 2/3：同一批次跑 N / C1 / C2 各 3 轮（拉丁方平衡顺序）。

★ 分工（重要，且是被现场打出来的）
--------------------------------
远端 `/root/qwen35-ascend-migrator/scripts/59_config_ab.py` 是**旧版**
（md5 `218e635b…`，**没有 `win_ms_of`**），而 `paired_stats` 依赖它
⇒ **不能拿远端 59 当判据来源**（否则版本偏斜会悄悄改变判据语义）。

因此本脚本的分工是：**远端只负责「跑 + 记录原始事实」**（rc / wall / 日志路径 / 行数），
**任何判据与统计都不在这里实现**；解析与统计回到本地，用**交付副本**的
`PAT` / `compute_valid` / `paired_stats` 做（单一来源 = 交付物）。
"""
import copy
import json
import os
import subprocess
import sys
import time

import yaml

SKILL = "/root/qwen35-ascend-migrator"
MIND = "/root/MindSpeed-MM"
BASE_CFG = "/root/ops/qwen3_5_0_8B_mock_dp2_v3.yaml"
STEPS = 100
WORLD = 2
# ★ 拉丁方：每臂在每个位置各出现一次 ⇒ 位置/漂移效应被抵消
SEQ = [("N", None), ("C1", "C1"), ("C2", "C2"),
       ("C1", "C1"), ("C2", "C2"), ("N", None),
       ("C2", "C2"), ("N", None), ("C1", "C1")]


def main():
    ts = time.strftime("%Y%m%d_%H%M%S")
    out = "/root/ops/ab62_%s" % ts
    os.makedirs(out, exist_ok=True)
    boot = open("/proc/sys/kernel/random/boot_id").read().strip()
    print("AB62_OUT=%s boot=%s steps=%d world=%d" % (out, boot, STEPS, WORLD), flush=True)
    doc0 = yaml.safe_load(open(BASE_CFG))
    env_json = os.path.join(SKILL, "out", "probe", "env.json")
    print("AB62_ENV_JSON exists=%s" % os.path.exists(env_json), flush=True)

    index = {"out": out, "boot": boot, "steps": STEPS, "world": WORLD,
             "seq": [{"arm": a, "MM_CAND": e} for a, e in SEQ],
             "hostname": open("/proc/sys/kernel/hostname").read().strip()}
    runs = []
    for i, (arm, envv) in enumerate(SEQ):
        tag = "%02d_%s" % (i, arm)
        case = os.path.join(out, tag)
        os.makedirs(case, exist_ok=True)
        cfg = copy.deepcopy(doc0)
        cfg["training"]["train_iters"] = STEPS
        cfg["training"]["save"] = os.path.join(case, "checkpoint")
        config = os.path.join(case, "config.yaml")
        with open(config, "w", encoding="utf-8") as fh:
            yaml.safe_dump(cfg, fh, sort_keys=False)
        log = os.path.join(case, "train.log")
        drv = os.path.join(case, "driver.log")
        env = os.environ.copy()
        if envv:
            env["MM_CAND"] = envv
        else:
            env.pop("MM_CAND", None)
        cmd = [sys.executable, os.path.join(SKILL, "scripts", "50_train.py"),
               "--config", config, "--env", env_json, "--log", log,
               "--workdir", MIND, "--steps", str(STEPS), "--world-size", str(WORLD),
               "--timeout", "900", "--foreground"]
        t0 = time.time()
        rc = None
        try:
            with open(drv, "w", encoding="utf-8") as fh:
                p = subprocess.run(cmd, cwd=SKILL, env=env, stdout=fh,
                                   stderr=subprocess.STDOUT, timeout=1500)
            rc = p.returncode
        except subprocess.TimeoutExpired:
            rc = -9
        wall = time.time() - t0
        ttext = open(log, encoding="utf-8", errors="replace").read() if os.path.exists(log) else ""
        # 只做「活性计数」，不做判据（判据在本地）
        n_iter = ttext.count("elapsed time per iteration")
        rec = {"tag": tag, "arm": arm, "MM_CAND": envv, "rc": rc,
               "wall_seconds": round(wall, 1),
               "train_log": log, "driver_log": drv, "config": config,
               "train_log_bytes": os.path.getsize(log) if os.path.exists(log) else 0,
               "iteration_lines": n_iter,
               "identity": {"boot": boot, "hostname": index["hostname"]}}
        runs.append(rec)
        with open(os.path.join(case, "run_record.json"), "w", encoding="utf-8") as fh:
            json.dump(rec, fh, ensure_ascii=False, indent=1)
        print("RUN %s MM_CAND=%s rc=%s wall=%.0fs iter_lines=%d logB=%d"
              % (tag, envv, rc, wall, n_iter, rec["train_log_bytes"]), flush=True)
        if n_iter == 0:
            print("AB62_STOP 该臂一行 iteration 都没有 ⇒ 停止（不拿残缺数据继续）", flush=True)
            index["runs"] = runs
            with open(os.path.join(out, "runs_index.json"), "w", encoding="utf-8") as fh:
                json.dump(index, fh, ensure_ascii=False, indent=1)
            return 1

    markers = {"c1_active": os.path.exists("/tmp/mm_cand_c1_active"),
               "c2_active": os.path.exists("/tmp/mm_cand_c2_active")}
    index["runs"] = runs
    index["markers"] = markers
    with open(os.path.join(out, "runs_index.json"), "w", encoding="utf-8") as fh:
        json.dump(index, fh, ensure_ascii=False, indent=1)
    print("MARKERS %s" % json.dumps(markers), flush=True)
    print("AB62_DONE out=%s" % out, flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
