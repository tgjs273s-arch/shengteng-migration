#!/usr/bin/env python3
# -*- coding: utf-8 -*-
r"""_aa_absdiff.py —— A/A 分歧的**量纲感知**刻画（本地，不占卡）。

为什么要它
----------
`_replay_diff.py` 报的是**相对差**（如 `layers.0.input_layernorm` 梯度最大相对差 37.7%）。
但梯度的相对差会被**小分母**放大：两个都接近 0 的元素，相对差可以很大而绝对差微不足道。
所以"37.7%"既不能证明结构性分歧，也不能证明只是噪声。

方法（**先声明，后计算**；本脚本只做**刻画**，不设通过/失败阈值）
------------------------------------------------------------
A/A 的判据是**逐字节相同**（已不成立，见 diff.txt）。本脚本回答的是"差多少、什么量纲"：
  · 用 `_dump_group` 里**全张量**统计量 `sum` 与 `absmax`（不是抽样）算相对差；
  · 用 `sample`（head+rand，固定种子）算**逐元素**绝对差与尺度比 `|Δ| / max|a|`；
  · 给出分布（中位/最大），并按"有多少张量逐字节相同"计数。
★ 不设阈值 ⇒ 不会出现"事后挑一个能通过的容差"。
"""
import json
import os
import statistics
import sys


def load(p):
    with open(p, encoding="utf-8") as fh:
        return json.load(fh)


def stats_for_group(name, ga, gb):
    ta, tb = (ga or {}).get("tensors") or {}, (gb or {}).get("tensors") or {}
    names = sorted(set(ta) & set(tb))
    same = 0
    sum_rel, absmax_rel, elem_abs, elem_scale = [], [], [], []
    for n in names:
        A, B = ta[n], tb[n]
        if A.get("sha256") and A.get("sha256") == B.get("sha256"):
            same += 1
            continue
        sa, sb = A.get("sum"), B.get("sum")
        if isinstance(sa, (int, float)) and isinstance(sb, (int, float)):
            sum_rel.append(abs(sa - sb) / max(abs(sa), 1e-30))
        aa, ab = A.get("absmax"), B.get("absmax")
        if isinstance(aa, (int, float)) and isinstance(ab, (int, float)):
            absmax_rel.append(abs(aa - ab) / max(abs(aa), 1e-30))
            for x, y in zip(A.get("sample") or [], B.get("sample") or []):
                elem_abs.append(abs(x - y))
                elem_scale.append(abs(x - y) / max(abs(x), abs(aa), 1e-30))
    def q(v, f, d=None):
        return round(f(v), 12) if v else d
    return {
        "group": name, "n_tensors": len(names), "n_byte_identical": same,
        "sum_rel_diff": {"median": q(sum_rel, statistics.median), "max": q(sum_rel, max)},
        "absmax_rel_diff": {"median": q(absmax_rel, statistics.median), "max": q(absmax_rel, max)},
        "sample_elem_abs_diff": {"median": q(elem_abs, statistics.median), "max": q(elem_abs, max)},
        "sample_elem_scale_ratio": {"median": q(elem_scale, statistics.median),
                                    "max": q(elem_scale, max)},
        "n_sample_pairs": len(elem_abs),
    }


def main():
    if len(sys.argv) < 4:
        print("USAGE: _aa_absdiff.py <a_dump> <b_dump> <out.json>")
        return 2
    path_a, path_b, outp = sys.argv[1], sys.argv[2], sys.argv[3]
    da, db = load(path_a), load(path_b)
    res = {"schema": "dsh.aa_absdiff.v1",
           "purpose": "A/A 分歧的**量纲感知**刻画（只刻画，不设通过阈值）",
           "a": os.path.basename(path_a), "b": os.path.basename(path_b),
           "note": "sum/absmax 为**全张量**统计量；sample 为固定种子抽样（head+rand）",
           "groups": []}
    for gname, key in (("C3_post_comm_grad", "C3_post_comm_grad"),
                       ("C4_after_one_update", "C4_after_one_update"),
                       ("OPT_before", "__opt_before"),
                       ("OPT_after", "__opt_after")):
        if key == "__opt_before":
            ga = (da.get("C4_optimizer_state") or {}).get("before")
            gb = (db.get("C4_optimizer_state") or {}).get("before")
        elif key == "__opt_after":
            ga = (da.get("C4_optimizer_state") or {}).get("after")
            gb = (db.get("C4_optimizer_state") or {}).get("after")
        else:
            ga, gb = da.get(key), db.get(key)
        if not ga or not gb:
            continue
        res["groups"].append(stats_for_group(gname, ga, gb))
    os.makedirs(os.path.dirname(os.path.abspath(outp)), exist_ok=True)
    with open(outp, "w", encoding="utf-8") as fh:
        json.dump(res, fh, ensure_ascii=False, indent=1)
    for g in res["groups"]:
        print("== %s" % g["group"])
        print("   张量 %d 个，逐字节相同 %d 个（不同 %d）"
              % (g["n_tensors"], g["n_byte_identical"], g["n_tensors"] - g["n_byte_identical"]))
        print("   全张量 sum    相对差：中位 %s 最大 %s"
              % (g["sum_rel_diff"]["median"], g["sum_rel_diff"]["max"]))
        print("   全张量 absmax 相对差：中位 %s 最大 %s"
              % (g["absmax_rel_diff"]["median"], g["absmax_rel_diff"]["max"]))
        print("   抽样逐元素 |Δ|：中位 %s 最大 %s（对数 %d）"
              % (g["sample_elem_abs_diff"]["median"], g["sample_elem_abs_diff"]["max"],
                 g["n_sample_pairs"]))
        print("   抽样 |Δ|/尺度：中位 %s 最大 %s"
              % (g["sample_elem_scale_ratio"]["median"], g["sample_elem_scale_ratio"]["max"]))
    print("AA_ABSDIFF_OUT %s" % outp)
    return 0


if __name__ == "__main__":
    sys.exit(main())
