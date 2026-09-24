#!/usr/bin/env python3
# -*- coding: utf-8 -*-
r"""③ 非零学习率单步回放 —— 编排（P0 修复版，在远端跑）。

P0 修复点（对应复核意见）
------------------------
1. ★ **调度语义一致**：cfg1 与 cfg2 **都保持 `train_iters=100`**（与基线配置一致），
   cfg1 只把 `save_interval` 设为 30 ⇒ 在**中间步**落完整状态；**不再**把 train_iters 改成 30
   来"冒充提前结束"（那会让 cfg1/cfg2 的 cosine 计划长度不同 ⇒ lr 轨迹不可比）。
   cfg1 **完整跑完 100 步**，取 `iter_0000030` 作为回放检查点。
2. ★ **不再打印写死的 warmup**：warmup 步数从**实际配置**算
   （`lr_warmup_steps = int(train_iters * lr_warmup_ratio)`，与 lr_scheduler.py 同一算式），
   并把 cfg1/cfg2 的实际值一起落盘。
3. ★ **`load_rank0_and_broadcast=false`**：本机实测该广播路径会
   `KeyError: 'optimizer'`（`broadcast_utils.py:150` 无条件取 `shard_state_dict['optimizer']`），
   而标准 `dcp.load()` 分支才同时恢复模型与优化器。
4. ★ **失败传播**：任一阶段失败 / 任一 rank 失败 / 回放前提 INVALID / 比较失败
   ⇒ 最终 `status.json` 记 FAIL 且**退出码非零**；`all.done` 也写明确状态词。
   **"流程结束"与"验证通过"分开表达**（`flow_finished` vs `verification_passed`）。
"""
import json
import os
import subprocess
import sys
import time

import yaml

BASE = "/root/ops/qwen3_5_0_8B_mock_dp2_v3.yaml"
SKILL = "/root/qwen35-ascend-migrator"
MIND = "/root/MindSpeed-MM"
ENV_JSON = SKILL + "/out/probe/env.json"
TOTAL_STEPS = 100          # ★ cfg1/cfg2 一致
SAVE_AT = 30               # 中间检查点
REPLAY_FROM = 30
DETSC_DIR = "/root/ops/detsc62"   # 确定性注入件所在目录（--detsc on 时挂到 PYTHONPATH 最前）
DETSC = False                     # ★ 由 --detsc 设定；**两次回放共用同一个值**（不允许一边开一边关）


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
    pp = MIND + ":" + e.get("PYTHONPATH", "")
    if DETSC:
        # ★ 放在最前 ⇒ sitecustomize 一定先于其它包被导入
        pp = DETSC_DIR + ":" + pp
        e["DETSC_MODE"] = "on"
    e["PYTHONPATH"] = pp
    return e


def run(cmd, log, cwd=None, timeout=3600, extra_env=None):
    env = env_for()
    if extra_env:
        env.update(extra_env)
    t0 = time.time()
    with open(log, "w", encoding="utf-8") as fh:
        p = subprocess.run(cmd, cwd=cwd, env=env, stdout=fh,
                           stderr=subprocess.STDOUT, timeout=timeout)
    return p.returncode, round(time.time() - t0, 1)


def make_resume_root(root, ckpt_iter):
    """构造"只含指定迭代"的恢复根目录。

    ★ 为什么必须这么做：`dcp_checkpointer.load()` **先读 tracker**
      （`latest_checkpointed_iteration.txt`）解析 iteration，再拼 `iter_%07d`。
      而训练根目录里的 tracker 指向**最新**检查点（本批跑到 100）⇒ 若直接拿根目录当
      `load`，回放的是 **iter_100**，不是我们要的**中间步 30**。
      （本机实测：第一版就是这样被自己的守卫拦下的——
       `TRACKER_ITERATION_MISMATCH_100_expected_30`。）
    """
    rr = os.path.join(root, "resume_from_%d" % ckpt_iter)
    os.makedirs(rr, exist_ok=True)
    src = os.path.join(root, "ckpt", "iter_%07d" % ckpt_iter)
    dst = os.path.join(rr, "iter_%07d" % ckpt_iter)
    if not os.path.exists(dst):
        try:
            os.symlink(src, dst)
        except OSError:
            import shutil
            shutil.copytree(src, dst)
    with open(os.path.join(rr, "latest_checkpointed_iteration.txt"), "w") as fh:
        fh.write(str(ckpt_iter))
    return rr


def main(reuse_root=None):
    ts = time.strftime("%Y%m%d_%H%M%S")
    root = reuse_root or ("/root/ops/nzreplay_%s" % ts)
    os.makedirs(root, exist_ok=True)
    status = {"schema": "dsh.nzreplay.status.v1", "root": root, "reuse_root": bool(reuse_root),
              "flow_finished": False, "verification_passed": False,
              "phases": {}, "failures": []}
    print("NZREPLAY_ROOT=%s reuse=%s" % (root, bool(reuse_root)), flush=True)

    base = yaml.safe_load(open(BASE, encoding="utf-8"))
    tr0 = base["training"]
    it0 = int(tr0.get("train_iters") or 0)
    ratio = float(tr0.get("lr_warmup_ratio") or 0.0)
    status["config_actual"] = {
        "base_train_iters": it0, "lr": tr0.get("lr"), "lr_decay_style": tr0.get("lr_decay_style"),
        "lr_warmup_ratio": ratio,
        # ★ 与 lr_scheduler.py 同一算式（不写死）
        "lr_warmup_steps_of_base": int(it0 * ratio),
        "lr_warmup_steps_of_cfg1": int(TOTAL_STEPS * ratio),
        "save_interval_cfg1": SAVE_AT, "replay_from_iteration": REPLAY_FROM,
    }
    print("### CONFIG_ACTUAL %s" % json.dumps(status["config_actual"], ensure_ascii=False), flush=True)

    # ---- cfg1：完整 100 步，中途保存完整状态（模型+优化器+RNG） ----
    c1 = yaml.safe_load(open(BASE, encoding="utf-8"))
    c1["training"]["train_iters"] = TOTAL_STEPS          # ★ 与 cfg2 一致
    c1["training"]["save_interval"] = SAVE_AT
    c1["training"]["save"] = root + "/ckpt"
    c1["training"]["no_save_optim"] = False              # ⇒ 框架自动切 save_format=dcp
    c1["training"]["no_save_rng"] = False
    c1["training"]["no_load_optim"] = True               # 载入的是 hf 权重，必须保持 True
    c1["training"]["no_load_rng"] = True
    cfg1 = root + "/cfg1_make_ckpt.yaml"
    with open(cfg1, "w", encoding="utf-8") as fh:
        yaml.safe_dump(c1, fh, sort_keys=False)

    # ---- cfg2：从检查点根恢复（tracker 解析 iteration），载入优化器/RNG ----
    c2 = yaml.safe_load(open(BASE, encoding="utf-8"))
    c2["training"]["train_iters"] = TOTAL_STEPS          # ★ 调度长度一致
    c2["training"]["load"] = root + "/ckpt"              # ★ 根目录（靠 tracker 解析）
    c2["training"]["load_format"] = "dcp"
    c2["training"]["no_load_optim"] = False
    c2["training"]["no_load_rng"] = False
    c2["training"]["no_save_optim"] = True
    c2["training"]["no_save_rng"] = True
    # ★ 本机实测：该广播路径对含优化器的 dcp 会 KeyError:'optimizer'
    c2["training"]["load_rank0_and_broadcast"] = False
    cfg2 = root + "/cfg2_replay_from_ckpt.yaml"
    with open(cfg2, "w", encoding="utf-8") as fh:
        yaml.safe_dump(c2, fh, sort_keys=False)
    status["cfg1"] = cfg1
    status["cfg2"] = cfg2

    # ---- PHASE 1（可跳过：`--reuse-root` 复用已产出的检查点，**不重跑训练**）----
    if not reuse_root:
        print("### PHASE1 完整 %d 步，save_interval=%d" % (TOTAL_STEPS, SAVE_AT), flush=True)
        rc1, w1 = run([sys.executable, os.path.join(SKILL, "scripts", "50_train.py"),
                       "--config", cfg1, "--env", ENV_JSON,
                       "--log", root + "/phase1_train.log", "--workdir", MIND,
                       "--steps", str(TOTAL_STEPS), "--world-size", "2", "--timeout", "1800",
                       "--foreground"],
                      log=root + "/phase1.driver.log", cwd=SKILL)
        status["phases"]["phase1"] = {"rc": rc1, "wall": w1}
        print("### PHASE1 rc=%s wall=%ss" % (rc1, w1), flush=True)
        if rc1 != 0:
            status["failures"].append("PHASE1_RC_%d" % rc1)
            return finish(root, status, 1)
    else:
        status["phases"]["phase1"] = {"skipped": True, "reason": "reuse-root（不重跑训练）"}
        print("### PHASE1 SKIPPED（复用已有检查点）", flush=True)

    ck_root = root + "/ckpt"
    ckdir = os.path.join(ck_root, "iter_%07d" % SAVE_AT)
    if not os.path.isdir(ckdir):
        status["failures"].append("CKPT_DIR_MISSING_%s" % ckdir)
        return finish(root, status, 1)
    # ★ 构造 tracker=SAVE_AT 的恢复根（见 make_resume_root 的说明）
    rr = make_resume_root(root, SAVE_AT)
    tracker = os.path.join(rr, "latest_checkpointed_iteration.txt")
    tval = open(tracker).read().strip()
    status["resume_root"] = rr
    status["tracker"] = {"path": tracker, "exists": True, "value": tval, "expected": SAVE_AT}
    print("### RESUME_ROOT=%s tracker=%s" % (rr, tval), flush=True)
    if str(tval) != str(SAVE_AT):
        status["failures"].append("TRACKER_ITERATION_MISMATCH_%s_expected_%d" % (tval, SAVE_AT))
        return finish(root, status, 1)
    c2["training"]["load"] = rr
    with open(cfg2, "w", encoding="utf-8") as fh:
        yaml.safe_dump(c2, fh, sort_keys=False)
    status["ckpt_dir"] = ckdir
    status["ckpt_files"] = sorted(os.listdir(ckdir))[:6] + ["..."]
    status["ckpt_has_metadata"] = os.path.isfile(os.path.join(ckdir, ".metadata"))
    status["ckpt_has_extra_state"] = os.path.isdir(os.path.join(ckdir, "extra_state"))
    print("### CK=%s metadata=%s extra_state=%s"
          % (ckdir, status["ckpt_has_metadata"], status["ckpt_has_extra_state"]), flush=True)

    # ---- PHASE 2：两次回放（同一检查点、**同一模式**、串行，一次只跑一个任务） ----
    tags = (("a_detsc", 6140), ("b_detsc", 6141)) if DETSC else (("a", 6130), ("b", 6131))
    status["detsc_mode"] = "on" if DETSC else "off"
    status["detsc_same_mode_both_runs"] = True   # 由 tags 单一来源决定，结构上不可能一边开一边关
    print("### PHASE2 DETSC=%s tags=%s" % (status["detsc_mode"], [t for t, _ in tags]), flush=True)
    for tag, port in tags:
        out = os.path.join(root, tag)
        os.makedirs(out, exist_ok=True)
        cmd = ["torchrun", "--nproc_per_node", "2", "--nnodes", "1", "--node_rank", "0",
               "--master_addr", "localhost", "--master_port", str(port),
               os.path.join(SKILL, "scripts", "52_replay.py"), cfg2]
        rc, wall = run(cmd, os.path.join(out, "run.log"), cwd=MIND,
                       timeout=1800, extra_env={"REPLAY_OUT": out})
        with open(os.path.join(out, "rc.txt"), "w") as fh:
            fh.write("rc=%d\n" % rc)
        rlog = ""
        p = os.path.join(out, "run.log")
        if os.path.isfile(p):
            rlog = open(p, encoding="utf-8", errors="replace").read()
        precond_invalid = [ln for ln in rlog.splitlines() if "REPLAY_PRECOND_INVALID" in ln]
        detsc_active = rlog.count("DETSC active")
        detsc_failed = rlog.count("DETSC FAILED")
        dumps = [f for f in os.listdir(out) if f.startswith("replay_dump.")]
        status["phases"]["phase2_%s" % tag] = {
            "rc": rc, "wall": wall, "dumps": sorted(dumps),
            "precond_invalid_lines": precond_invalid[:4],
            "precond_ok": bool(rc == 0 and not precond_invalid),
            "detsc_active_lines": detsc_active, "detsc_failed_lines": detsc_failed,
        }
        print("### PHASE2-%s rc=%s wall=%ss dumps=%s precond_invalid=%d detsc_active=%d detsc_failed=%d"
              % (tag, rc, wall, sorted(dumps), len(precond_invalid), detsc_active, detsc_failed),
              flush=True)
        if rc != 0:
            status["failures"].append("PHASE2_%s_RC_%d" % (tag, rc))
        if precond_invalid:
            status["failures"].append("PHASE2_%s_PRECOND_INVALID" % tag)
        if not dumps:
            status["failures"].append("PHASE2_%s_NO_DUMP" % tag)
        if DETSC:
            # ★ 注入自证：torchrun 主进程 + 2 个 rank worker ⇒ 至少 3 行；且不得有 FAILED
            if detsc_active < 3:
                status["failures"].append("PHASE2_%s_DETSC_NOT_ACTIVE_%d" % (tag, detsc_active))
            if detsc_failed:
                status["failures"].append("PHASE2_%s_DETSC_FAILED_%d" % (tag, detsc_failed))

    if status["failures"]:
        return finish(root, status, 1)

    # ---- 恢复步号核对（从 dump 里读实际 iteration） ----
    iters = {}
    for tag, _p in tags:
        f = os.path.join(root, tag, "replay_dump.rank0.json")
        try:
            d = json.load(open(f, encoding="utf-8"))
            iters[tag] = (d.get("fixed") or {}).get("iteration")
        except Exception as exc:
            iters[tag] = "READ_FAIL:%s" % type(exc).__name__
    status["restored_iteration"] = iters
    print("### RESTORED_ITERATION %s (expected %d)" % (json.dumps(iters), REPLAY_FROM), flush=True)
    for tag, v in iters.items():
        if v != REPLAY_FROM:
            status["failures"].append("RESTORED_ITERATION_MISMATCH_%s_%s" % (tag, v))
    if status["failures"]:
        return finish(root, status, 1)

    # ---- PHASE 3：比较 ----
    rc3, w3 = run([sys.executable, os.path.join(SKILL, "scripts", "_replay_diff.py"),
                   "--a", os.path.join(root, tags[0][0]),
                   "--b", os.path.join(root, tags[1][0])],
                  log=root + "/diff_%s.txt" % status["detsc_mode"],
                  cwd=os.path.join(SKILL, "scripts"))
    status["phases"]["phase3"] = {"rc": rc3, "wall": w3}
    print("### PHASE3 rc=%s wall=%ss" % (rc3, w3), flush=True)
    if rc3 != 0:
        status["failures"].append("PHASE3_RC_%d" % rc3)
    status["diff_tail"] = open(root + "/diff_%s.txt" % status["detsc_mode"], encoding="utf-8",
                               errors="replace").read()[-1200:]
    return finish(root, status, 0 if rc3 == 0 else 1)


def finish(root, status, code):
    status["flow_finished"] = True
    status["verification_passed"] = bool(code == 0 and not status["failures"])
    status["exit_code"] = code
    with open(root + "/status.json", "w", encoding="utf-8") as fh:
        json.dump(status, fh, ensure_ascii=False, indent=1)
    word = ("NZREPLAY_VERIFIED" if status["verification_passed"]
            else ("NZREPLAY_FLOW_DONE_FAILED" if status["flow_finished"] else "NZREPLAY_ABORT"))
    with open(root + "/all.done", "w") as fh:
        fh.write(word + "\n")
    print("### %s exit=%d failures=%s" % (word, code, status["failures"]), flush=True)
    print("NZREPLAY_STATUS %s/status.json" % root, flush=True)
    return code


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--reuse-root", default=None, dest="reuse_root",
                    help="复用已有批次目录里的检查点（**跳过 PHASE1，不重跑训练**）")
    ap.add_argument("--detsc", default="off", choices=["on", "off"],
                    help="确定性诊断对照：on = 两次回放**都用** sitecustomize 注入 "
                         "torch.use_deterministic_algorithms(True)（两侧同一模式）")
    a = ap.parse_args()
    DETSC = (a.detsc == "on")
    sys.exit(main(reuse_root=a.reuse_root))
