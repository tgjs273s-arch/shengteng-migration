# -*- coding: utf-8 -*-
r"""_replay_diff.py —— 比较两次"一次更新回放"的产物（协议 §criteria.decision_rule）

判据（与 `protocols/one_step_replay_20260922.json` 一致）：
  1. **先看固定状态**：两次的 `fixed`（input/params_before/opt_state/rng/lr/iteration）必须一致，
     否则整轮**不可比** ⇒ 直接 `REPLAY_FAIL fixed_state_mismatch`（fail-closed：状态没固定住，
     谈"哪一段先不同"毫无意义）。
  2. 再按 C1 →（C2 恒为 UNVERIFIED，不参与）→ C3 → C4 找**第一处不同**：
     · `sha_all` 相同 ⇒ 该点**逐字节相同**；
     · 不同 ⇒ 报出不同张量个数，并对**抽样元素**给"不同元素占比 + 最大相对差"
       （抽样口径 head64+rand64，**不是**全量；口径写在 dumps 里，本工具也打印出来）。
  3. NaN/Inf 计数 > 0 ⇒ 该点 `INVALID`（不进入"相同"）。
  4. 全同 ⇒ 输出 `REPLAY_IDENTICAL`，并**明确**：只证明"这一步"；不推出全程可复现。
"""
import argparse
import io
import json
import os
import sys


def load(d, rank):
    p = os.path.join(d, "replay_dump.rank%d.json" % rank)
    if not os.path.isfile(p):
        return None, p
    return json.load(io.open(p, encoding="utf-8")), p


def sample_stats(a, b):
    """抽样口径下的（不同元素占比, 最大相对差%）。分母为 0 记 undecidable 并计数。"""
    sa, sb = a.get("sample") or [], b.get("sample") or []
    if len(sa) != len(sb):
        return None, None, len(sa), len(sb), 0
    diff, maxpct, undec = 0, 0.0, 0
    for x, y in zip(sa, sb):
        if x != y:
            diff += 1
        if x == 0:
            undec += 1
            continue
        pct = abs(y - x) / abs(x) * 100.0
        maxpct = max(maxpct, pct)
    n = len(sa) or 1
    return diff / n, maxpct, len(sa), len(sb), undec


def cmp_group(ga, gb, label):
    """返回 (same, detail_lines)。same=True 表示组级与逐张量均逐字节相同。"""
    lines, same = [], True
    if (ga is None) != (gb is None):
        return False, ["  %s: 一侧缺该对比点 ⇒ UNVERIFIED（不得用其它点替代）" % label]
    if ga is None:
        return False, ["  %s: 两侧都缺 ⇒ UNVERIFIED" % label]
    sa, sb = ga.get("sha_all"), gb.get("sha_all")
    if ga.get("nonfinite") or gb.get("nonfinite"):
        same = False
        lines.append("  %s: **INVALID**（非有限值 a=%d b=%d）" % (label, ga.get("nonfinite"), gb.get("nonfinite")))
    if sa == sb:
        lines.append("  %s: 逐字节相同（n_tensors=%d sha_all=%s）" % (label, ga.get("n_tensors"), str(sa)[:16]))
        return same, lines
    same = False
    ta, tb = ga.get("tensors") or {}, gb.get("tensors") or {}
    keys = sorted(set(ta) | set(tb))
    bad = [k for k in keys if (ta.get(k) or {}).get("sha256") != (tb.get(k) or {}).get("sha256")]
    lines.append("  %s: **不同**（sha_all %s ≠ %s；逐张量不同 %d/%d）"
                 % (label, str(sa)[:16], str(sb)[:16], len(bad), len(keys)))
    for k in bad[:4]:
        ra, rb = ta.get(k) or {}, tb.get(k) or {}
        r, mp, na, nb, un = sample_stats(ra, rb)
        lines.append("     · %-58s 不同元素占比=%s 最大相对差=%s%% 零分母=%d（抽样 %d/%d）"
                     % (k[:58], ("%.4f" % r) if r is not None else "长度不等",
                        ("%.4f" % mp) if mp is not None else "-", un, na, nb))
    return same, lines


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--a", required=True, help="run A 的产物目录")
    ap.add_argument("--b", required=True, help="run B 的产物目录")
    ap.add_argument("--ranks", default="0,1")
    a_ = ap.parse_args()
    rc = 0
    for rank in [int(x) for x in a_.ranks.split(",") if x.strip()]:
        da, pa = load(a_.a, rank)
        db, pb = load(a_.b, rank)
        print("== rank %d ==" % rank)
        if da is None or db is None:
            print("  REPLAY_FAIL 缺产物：%s / %s" % (pa, pb))
            return 3
        fa, fb = da.get("fixed") or {}, db.get("fixed") or {}
        # ★ 实测暴露的语义错误（首次跨运行比对即踩到）：`fixed` 里我混进了**结果**字段
        #   （grads_sha_all / params_after_sha_all / opt_state_sha_all），而它们其实是 C3/C4 的产物。
        #   把它们当"固定状态前提"⇒ 两轮一旦梯度不同就直接报"不可比"，**恰好抹掉了协议要找的
        #   『第一处不同』**（C1 相同而 C3 不同这个最关键的信息）。
        #   ⇒ 前提 = 输入/初始参数/RNG/学习率/步号；结果 = 梯度/更新后参数/优化器状态。
        PRE = ("input_sha_all", "params_before_sha_all", "rng_cpu_sha", "iteration", "lr")
        OUT = ("grads_sha_all", "params_after_sha_all", "opt_state_sha_all")
        keys = sorted(set(fa) | set(fb))
        hard = [k for k in keys if k in PRE and k != "lr" and fa.get(k) != fb.get(k)]
        soft = [k for k in keys if k == "lr" and fa.get(k) != fb.get(k)]
        for k in keys:
            tag = "前提" if k in PRE else ("结果" if k in OUT else "其它")
            mark = "同" if fa.get(k) == fb.get(k) else "**不同**"
            print("  [%s] fixed.%-22s %s  A=%s B=%s" % (tag, k, mark, str(fa.get(k))[:24], str(fb.get(k))[:24]))
        if hard:
            print("  ★ 前提不一致（%s）⇒ 本轮**不可比**" % hard)
            print("REPLAY_FAIL fixed_state_mismatch rank=%d" % rank)
            return 2
        if soft:
            print("  ★ 学习率不同（%s）⇒ 这不是『同状态』比较，必须修正后再判" % soft)
            print("REPLAY_FAIL lr_mismatch rank=%d" % rank)
            return 2
        if fa.get("params_after_sha_all") == fa.get("params_before_sha_all"):
            print("  ⚠ **C4 无区分力**：params_after == params_before（本配置 lr=%s ⇒ 第一次更新是空更新）"
                  "⇒ C4 的『相同』不能作为『优化器路径可复现』的证据" % fa.get("lr"))

        first, lines = None, []
        # C1
        c1a, c1b = da.get("C1_forward") or {}, db.get("C1_forward") or {}
        loss_same = (c1a.get("loss") or {}).get("sha256") == (c1b.get("loss") or {}).get("sha256")
        logit_same = (c1a.get("logits") or {}).get("sha256") == (c1b.get("logits") or {}).get("sha256")
        lines.append("  C1 前向输出：loss %s（%s vs %s）；logits %s"
                     % ("逐字节相同" if loss_same else "**不同**",
                        str((c1a.get("loss") or {}).get("sha256"))[:12],
                        str((c1b.get("loss") or {}).get("sha256"))[:12],
                        "逐字节相同" if logit_same else "**不同**"))
        if not (loss_same and logit_same):
            first = "C1"
            ra, rb = (c1a.get("loss") or {}), (c1b.get("loss") or {})
            r, mp, na, nb, un = sample_stats(ra, rb)
            lines.append("     · loss 抽样：不同元素占比=%s 最大相对差=%s%%（零分母 %d）"
                         % (("%.4f" % r) if r is not None else "-",
                            ("%.4f" % mp) if mp is not None else "-", un))
        lines.append("  C2 通信前局部梯度：UNVERIFIED（FSDP2 不可观测；**不参与**判定，不用 C3 顶替）")
        # C3
        same3, l3 = cmp_group(da.get("C3_post_comm_grad"), db.get("C3_post_comm_grad"), "C3 通信后梯度")
        lines += l3
        if not same3 and first is None:
            first = "C3"
        # C4
        same4, l4 = cmp_group(da.get("C4_after_one_update"), db.get("C4_after_one_update"), "C4 一次更新后参数")
        lines += l4
        if not same4 and first is None:
            first = "C4"
        for l in lines:
            print(l)
        if first is None:
            print("REPLAY_IDENTICAL rank=%d（**只**证明这一步；不推出全程可复现）" % rank)
        else:
            print("REPLAY_VERDICT rank=%d first_diff=%s（判定树：C1⇒前向 / C3⇒通信 / C4⇒优化器更新）"
                  % (rank, first))
            rc = 1
    return rc


if __name__ == "__main__":
    sys.exit(main())
