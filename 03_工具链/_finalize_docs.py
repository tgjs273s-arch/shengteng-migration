# -*- coding: utf-8 -*-
r"""_finalize_docs.py —— ①把"正确性未闭合"的证据链写进 v2 决策记录；②协议登记诊断工具（非闸门）；
③重打包并核对

口径全部来自已落盘证据（`远端证据_20260922/ckpt_compare_00A_vs_03A.md` 与 v2 协议的
`evidence_chain_2026_09_22`），**不新增任何未测过的说法**。
"""
import io
import json
import os
import subprocess
import sys

PKG = r"C:\Users\HUAWEI\Desktop\转交给codex的内容"
TOOLS = os.path.join(PKG, "04_核心资产", "工具_重建")
SKILL = os.path.join(PKG, "04_核心资产", "qwen35-ascend-migrator_整合版")
DOC = os.path.join(SKILL, "docs", "推荐配置与保底配置_决策记录_v2_20260922.md")
PY = sys.executable

SECTION = '''

---

## 8. 正确性：**未闭合** —— 证据链与边界（2026-09-22 补）

本节把"同配置重复不满足一致性"从**现象**收窄为**有位置、有边界**的结论。所有数字来自落盘证据
（`远端证据_20260922/ckpt_compare_00A_vs_03A.md`、各 run 的 `train.log`）。

### 8.1 证据链（四步，各自独立）

| # | 观测 | 判读 |
|---|---|---|
| ① | 启动输入 dump：四份日志（`00A/03A/P2r1/P2r2`）哈希**完全相同**（`ec701fb9d5ebd668`，19 行） | 数据侧在**打印精度内**一致 |
| ② | 同配置（`deter=false`）内：第 1 步 `loss` 打印相同，但**第 1 步 `grad_norm` 即不同**（`47.326 / 47.34 / 47.337`） | 分歧**首次出现在梯度**上，不是前向输入 |
| ③ | 100 步后落盘权重：**320/473 张量不同**（另 153 逐位相同），量级 `1e-4~1e-3`、`mean_rel 0.02%–0.26%`、**无非有限值**，且**遍布** embedding / attention(q,k) / GDN(in_proj_qkv,in_proj_z,out_proj) / MLP(gate_proj) | 形态与「**全局归约/累加顺序类**非确定性」**一致** |
| ④ | 开 `use_deter_comp=true`：100 步 `max=0.00%`、连第 1 步 `grad_norm` 都一致 | 该非确定性**可被门控消除** |

### 8.2 可以写、不可以写

**可以写**：

> 在当前已测试环境、mock 数据与该配置下，**同配置重复未满足完整窗口（1..100）的逐点 2% 一致性要求**；
> 证据指向**反向/梯度路径的归约类非确定性**（形态一致，**未做算子级归因**）；该现象可由
> `use_deter_comp=true` 门控消除。原因仍在定位，**尚未归因于平台，也不据此推断官方基线不可复现**。

**不可以写**（逐条与本节的证据冲突或超出证据）：

1. ~~"判据不可用，所以可以只看前 k 步"~~ —— 窗口不缩；未满足就是未满足。
2. ~~"昇腾平台无法复现，任何人都做不到"~~ —— 无同输入同状态的最小复现，不得泛化。
3. ~~"A 配置可放行"~~ —— 正确性未闭合；**A 目前只是时间口径上的候选**。
4. ~~"已定位到某个算子/kernel"~~ —— 只到"梯度/归约类形态一致"这一层。
5. ~~"`use_deter_comp=true` 可以当出货配置"~~ —— 它**改变前向计算**：step1 loss `0.1154926 → 0.1169518`，
   连 `grad_norm` 也从 `47.326 → 47.787` ⇒ **仅作诊断**。

### 8.3 边界（本方法的可见范围）

- `grad_norm` 只打印 **3 位小数**、`loss` 8 位有效 ⇒ **精度以下的差异本方法看不见**；
  "第 1 步 loss 相同"**不等于**"第 1 步计算逐位一致"。
- 落盘权重为 **bf16** ⇒ "逐位相同（153 个张量）"只在该存储精度下成立。
- 权重差异是 **100 步累积结果**，不能单独把责任落到第 1 步；它与②是**两条独立证据**，方向一致。
- ★ **缺正对照**：`deter=true` 那一对的 checkpoint 被早前清理（坑 187）⇒
  "确定性路径下权重应逐位相同"这一条**尚无实测正对照**，需重跑。

### 8.4 待补（已列入待办）

1. 重跑 2 次 `use_deter_comp=true`（100 步）并**保留 checkpoint** ⇒ 补上 8.3 的正对照；
2. 每步 batch 指纹（现仅有启动 dump）⇒ 回答"每一步的输入是否相同"。
'''

if os.path.isfile(DOC):
    d = io.open(DOC, encoding="utf-8").read()
    if "## 8. 正确性" in d:
        print("DOC_SKIP 已含第 8 节")
    else:
        io.open(DOC, "w", encoding="utf-8", newline="").write(d.rstrip("\n") + SECTION)
        print("DOC_APPENDED v2 决策记录已补第 8 节（证据链/可写不可写/边界/待补）")
else:
    print("DOC_MISSING %s" % DOC)
    sys.exit(2)

# ---- ② 协议登记诊断工具（明确"不是闸门"）----
P = os.path.join(TOOLS, "protocols", "numeric_diag_phase1b_v2.json")
j = json.load(io.open(P, encoding="utf-8"))
j["diagnostic_tools"] = {
    "note": ("★ 这些是**诊断**工具，**不进**推送前闸门（闸门只跑本地可复现的检查；它们需要真机产物）。"
             "登记在此是为了让结论可追溯到具体工具与命令。"),
    "tools": [
        {"name": "numeric_diag_v2.py", "purpose": "按协议做逐步数值裁决（三层状态、窗口不缩、归因纪律）",
         "gate": False},
        {"name": "logs_to_results.py", "purpose": "train.log → results.json（含完整步号集合，供认证）",
         "gate": False},
        {"name": "localize_first_diff.py", "purpose": "分开看 loss/grad_norm，判断分歧先进前向还是反向",
         "gate": False},
        {"name": "compare_ckpt.py", "purpose": "逐张量比对两次运行的落盘权重（不需框架接口）",
         "gate": False},
        {"name": "check_source_consistency.py", "purpose": "交付源码包 vs Skill 同名文件内容一致",
         "gate": True},
        {"name": "safe_pack_sync.py --verify-only", "purpose": "zip/两处解包副本/活副本三方哈希一致",
         "gate": True}
    ]
}
json.dump(j, io.open(P, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
print("PROTO_UPDATED 已登记 diagnostic_tools（%d 项）" % len(j["diagnostic_tools"]["tools"]))

# ---- ③ 重打包 + 核对 ----
for args, label in ((["--apply"], "打包同步"), (["--verify-only"], "只读核对")):
    r = subprocess.run([PY, os.path.join(TOOLS, "safe_pack_sync.py")] + args,
                       capture_output=True, text=True, encoding="utf-8", errors="replace")
    keep = [l for l in ((r.stdout or "") + (r.stderr or "")).splitlines() if "SAFE_PACK" in l]
    print("%s rc=%d | %s" % (label, r.returncode, keep[-1] if keep else ""))
print("FINALIZE_DOCS_DONE")
