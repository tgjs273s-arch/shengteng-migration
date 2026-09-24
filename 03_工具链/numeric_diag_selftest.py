#!/usr/bin/env python3
# -*- coding: utf-8 -*-
r"""numeric_diag_selftest.py —— v2 分析器的坏例集（复核指出的四条 + 本轮四条 + 正对照）

★ 本轮（2026-09-22 第二轮）为什么重写断言：
  旧版坏例**只断言"输出里出现过 INVALID"**，于是出现了最坏的一种"自检通过"——
  分析器**打印**了 INVALID，却**汇总**成"满足"，而自检照样 PASS（外部复核实测确认）。
  ⇒ 现在**每条坏例同时断言三件事**：① 原因码 ② **最终判定措辞**（不得出现"满足"）
     ③ **退出码**（3=步号未认证 / 4=身份未认证 / 5=无法裁决）。缺任何一项就是自检不及格。

★ 身份字段：从本轮起分析器**默认要求**每次运行声明 运行状态/配置身份/数据身份；
  所以下面的 `mkrun()` 正常运行时都会带上（否则所有用例都会被判 rc=4）。
  身份缺失本身也是一条坏例（见 A5/A6），事后重析用 `--allow-missing-identity`（见 A5b）。
"""
import io
import json
import os
import subprocess
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
PY = sys.executable
TOOL = os.path.join(HERE, "numeric_diag_v2.py")
PROTO = os.path.join(HERE, "protocols", "numeric_diag_phase1b.json")
PROTO_V2 = os.path.join(HERE, "protocols", "numeric_diag_phase1b_v2.json")


def run_tool(results, protocol=PROTO, extra=()):
    td = tempfile.mkdtemp(prefix="nd2_")
    rp = os.path.join(td, "results.json")
    io.open(rp, "w", encoding="utf-8").write(json.dumps(results, ensure_ascii=False))
    cmd = [PY, TOOL, "--protocol", protocol, "--results", rp] + list(extra)
    r = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8", errors="replace")
    return r.returncode, (r.stdout or "") + (r.stderr or "")


def mkrun(tag, variant, loss, gn, steps=None, identity=True, **extra):
    """默认带三层身份（运行状态/配置身份/数据身份）；identity=False 用于身份坏例。"""
    d = {"tag": tag, "variant": variant, "valid": True, "loss": loss, "grad_norm": gn}
    if steps is not None:
        d["steps"] = steps
    if identity:
        d.update({"rc": 0, "config_sha256": "cfg_" + variant.lower(), "data_identity": "mock_dp2"})
    d.update(extra)
    return d


def main():
    fails = []

    def ck(name, cond, extra=""):
        print("  [%s] %-64s %s" % ("PASS" if cond else "FAIL", name, extra))
        if not cond:
            fails.append(name)

    S8 = list(range(1, 9))
    base_loss = [0.1, 0.1, 0.09, 0.08, 0.07, 0.06, 0.05, 0.04]
    base_gn = [1.0, 1.0, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4]

    # ---------- 正对照 ----------
    rc, out = run_tool([mkrun("r1", "A", base_loss, base_gn, S8),
                        mkrun("r2", "A", list(base_loss), list(base_gn), S8)])
    ck("正对照①：逐位相同 ⇒ 『满足』且 rc=0",
       ("重复性要求（完整窗口 1..8，阈值 2.0%）：满足" in out) and rc == 0, "rc=%s" % rc)

    l2 = list(base_loss)
    l2[2] = 0.2
    rc, out = run_tool([mkrun("r1", "A", base_loss, base_gn, S8),
                        mkrun("r2", "A", l2, list(base_gn), S8)])
    ck("正对照②：第 3 步起分叉 ⇒ 首次超阈步=3、超阈 1/8",
       ("loss 超阈 1/8" in out) and ("首次超阈步 3" in out), "rc=%s" % rc)
    ck("正对照②：不得出现『缩窗/判据不可用/截到』",
       ("缩窗" not in out) and ("判据不可用" not in out) and ("截到" not in out))
    ck("正对照②：判定为『未满足』且仍是有效裁决（rc=0，不是 5）",
       ("）：未满足" in out) and rc == 0, "rc=%s" % rc)

    # ---------- 坏例 A1：长度不等（两臂都自报 8 步，序列 8 vs 7）----------
    rc, out = run_tool([mkrun("r1", "A", base_loss, base_gn, S8),
                        mkrun("r2", "A", base_loss[:7], base_gn[:7], S8)])
    ck("坏例A1：长度不等 ⇒ INVALID(length_mismatch) 且**不得满足**、rc=5",
       ("INVALID(length_mismatch)" in out) and ("）：无法裁决" in out) and rc == 5, "rc=%s" % rc)

    # ---------- 坏例 A2：NaN ----------
    ln = list(base_loss)
    ln[4] = float("nan")
    rc, out = run_tool([mkrun("r1", "A", base_loss, base_gn, S8),
                        mkrun("r2", "A", ln, list(base_gn), S8)])
    ck("坏例A2：NaN ⇒ INVALID(non_finite) 且**不得满足**、rc=5",
       ("INVALID(non_finite)" in out) and ("）：无法裁决" in out) and rc == 5, "rc=%s" % rc)

    # ---------- 坏例 A3（外部复核 D2）：自报 8 步但两条序列都只有 2 步 ----------
    rc, out = run_tool([mkrun("r1", "A", base_loss[:2], base_gn[:2], S8),
                        mkrun("r2", "A", base_loss[:2], base_gn[:2], S8)])
    ck("坏例A3：自报 8 步但序列仅 2 步 ⇒ INVALID(coverage_shortfall)、不得满足、rc=5",
       ("INVALID(coverage_shortfall)" in out) and ("）：无法裁决" in out) and rc == 5, "rc=%s" % rc)

    # ---------- 坏例 A4（外部复核 D3）：零分母被计数却仍判满足 ----------
    rc, out = run_tool([mkrun("r1", "A", [0.0] + base_loss[1:], base_gn, S8),
                        mkrun("r2", "A", [0.0] + base_loss[1:], list(base_gn), S8)])
    ck("坏例A4：零分母 ⇒ 计数上报 且**不得满足**、rc=5（旧版这里会说『满足』）",
       ("零分母 1" in out) and ("）：不可判定" in out) and rc == 5, "rc=%s" % rc)

    # ---------- 坏例 A5：身份缺失 ----------
    rc, out = run_tool([mkrun("r1", "A", base_loss, base_gn, S8, identity=False),
                        mkrun("r2", "A", base_loss, base_gn, S8, identity=False)])
    ck("坏例A5：缺三层身份 ⇒ UNVERIFIED_IDENTITY、rc=4、不得满足",
       ("NUMERIC_V2_UNVERIFIED_IDENTITY" in out) and ("）：满足" not in out) and rc == 4, "rc=%s" % rc)
    rc, out = run_tool([mkrun("r1", "A", base_loss, base_gn, S8, identity=False),
                        mkrun("r2", "A", base_loss, base_gn, S8, identity=False)],
                       extra=("--allow-missing-identity",))
    ck("坏例A5b：事后重析开关 ⇒ 显式 POSTHOC 标记（结论降级，不当身份证据）",
       ("NUMERIC_V2_POSTHOC_NO_IDENTITY" in out) and rc == 0, "rc=%s" % rc)

    # ---------- 坏例 A6：身份字段存在但为空串（负向对照：空串不算声明）----------
    r1 = mkrun("r1", "A", base_loss, base_gn, S8, identity=False)
    r2 = mkrun("r2", "A", base_loss, base_gn, S8, identity=False)
    for r in (r1, r2):
        r.update({"rc": "", "config_sha256": "   ", "data_identity": None})
    rc, out = run_tool([r1, r2])
    ck("坏例A6：身份字段存在但为空/空白 ⇒ 仍 UNVERIFIED_IDENTITY、rc=4",
       ("NUMERIC_V2_UNVERIFIED_IDENTITY" in out) and rc == 4, "rc=%s" % rc)

    # ---------- 坏例 A7：步号未认证（缺 steps / 非 1..N）----------
    rc1, out1 = run_tool([mkrun("r1", "A", base_loss, base_gn, None),
                          mkrun("r2", "A", base_loss, base_gn, None)])
    ck("坏例A7a：缺 steps ⇒ rc=3 + UNVERIFIED_STEPS",
       rc1 == 3 and "NUMERIC_V2_UNVERIFIED_STEPS" in out1, "rc=%s" % rc1)
    rc2, out2 = run_tool([mkrun("r1", "A", base_loss, base_gn, [2, 3, 4, 5, 6, 7, 8, 9]),
                          mkrun("r2", "A", base_loss, base_gn, [2, 3, 4, 5, 6, 7, 8, 9])])
    ck("坏例A7b：步号 2..9（非 1..N）⇒ rc=3 + UNVERIFIED_STEPS",
       rc2 == 3 and "NUMERIC_V2_UNVERIFIED_STEPS" in out2, "rc=%s" % rc2)

    # ---------- A8：跨臂无效**不得**拖垮同臂重复性结论（判别"修复是否过宽"）----------
    rc, out = run_tool([mkrun("r1", "A", base_loss, base_gn, S8),
                        mkrun("r2", "A", base_loss, base_gn, S8),
                        mkrun("b1", "B", base_loss[:3], base_gn[:3], S8)])
    ck("A8：同臂完全重复 + 跨臂序列残缺 ⇒ 重复性仍『满足』(rc=0)，跨臂单独提示无法裁决",
       ("）：满足" in out) and ("跨臂比较本轮无法裁决" in out) and rc == 0, "rc=%s" % rc)

    # ---------- A9：当前冻结的 v2 协议（inherited_criteria）能被直接读取 ----------
    rc, out = run_tool([mkrun("P2r1", "P2", base_loss, base_gn, None),
                        mkrun("P2r2", "P2", base_loss, base_gn, None)], protocol=PROTO_V2,
                       extra=("--allow-missing-identity",))
    ck("A9：协议 v2(inherited_criteria) 可读 ⇒ 窗口=1..100、phase1b-v2-inherited_criteria",
       ("phase1b-v2-inherited_criteria" in out) and ("窗口=1..100" in out), "rc=%s" % rc)

    print("NUMERIC_V2_SELFTEST_%s cases=%d failed=%d"
          % ("OK" if not fails else "FAIL", 14, len(fails)))
    return 0 if not fails else 1


if __name__ == "__main__":
    sys.exit(main())
