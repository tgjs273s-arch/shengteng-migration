# -*- coding: utf-8 -*-
r"""_finalize_poscontrol.py —— 回写正对照结果：证据文件 / 协议 / 决策记录（含改掉已过时的一句）

背景：先前 `compare_ckpt.py` 只能给出"同配置两次 320/473 张量不同"，**缺正对照**（坑 187）。
现补齐：重跑 2 次 `use_deter_comp=true`（100 步，**保留 checkpoint**）⇒ **473/473 逐位相同**。
⇒ 方法够灵敏，且该非确定性被该开关**完整门控**。
★ 决策记录 §8.3 里"缺正对照"那句**已过时**，必须改（否则交付文档自相矛盾）。
"""
import io
import json
import os
import subprocess
import sys

PKG = r"C:\Users\HUAWEI\Desktop\转交给codex的内容"
TOOLS = os.path.join(PKG, "04_核心资产", "工具_重建")
SKILL = os.path.join(PKG, "04_核心资产", "qwen35-ascend-migrator_整合版")
EV = os.path.join(PKG, "04_核心资产", "远端证据_20260922")
PY = sys.executable

ADD = '''

---

## 正对照（2026-09-22 补，坑 187 的处置落地）

- 方法：同一 `compare_ckpt.py`，对象换成 **`use_deter_comp=true`** 的两次独立运行
  （`/root/ops/p2_keepckpt_20260921_173205/run{1,2}/save/iter_0000100`，各 100 步，**保留 checkpoint**）
- 派生断言：`training.use_deter_comp False→True` 回读通过、`GBS=8`、`train_iters=100`（**未截断**）
- 生效取证：两轮 `train.log` 的配置 dump 均显示 `use_deter_comp: True`

```
CKPT_SUM 共同张量=473  **逐位相同=473**  不同=0  含非有限=0
CKPT_IDENTICAL 全部 473 个张量逐位相同
```

## 对照表（同一方法、同一负载、同一 100 步协议）

| 配置 | 两次运行落盘权重 | 逐步 loss/grad_norm |
|---|---|---|
| `use_deter_comp=false`（出厂） | **320/473 张量不同**，量级 `1e-4~1e-3`，无非有限值 | 90–95/100 步超 2%；第 1 步 `grad_norm` 即不同 |
| `use_deter_comp=true`（诊断） | **473/473 逐位相同** | 0/100 步超阈，`max=0.00%`；连第 1 步 `grad_norm` 一致 |

**两条推论（各有边界，不得越读）**：

1. **方法灵敏度已被证明**：该报"相同"时确实报相同（473/473），因此 `false` 侧的 320/473 不同
   不是工具噪声 ⇒ 该对比有判别力。
2. **该非确定性被 `use_deter_comp` 完整门控**（两个可观测量——逐步 loss/grad_norm 与落盘权重——同时成立）；
   但**仍不做算子级归因**（不知道具体是哪个 kernel/归约），且**该开关改变前向计算**
   （step1 loss `0.1154926→0.1169518`、`grad_norm 47.326→47.787`）⇒ **只能作诊断**。
'''

# ① 证据文件追加正对照
p = os.path.join(EV, "ckpt_compare_00A_vs_03A.md")
d = io.open(p, encoding="utf-8").read()
if "正对照（2026-09-22 补" in d:
    print("EV_SKIP 证据文件已含正对照")
else:
    io.open(p, "w", encoding="utf-8", newline="").write(d.rstrip("\n") + ADD)
    print("EV_APPENDED 证据文件已补正对照")

# ② 协议更新
P = os.path.join(TOOLS, "protocols", "numeric_diag_phase1b_v2.json")
j = json.load(io.open(P, encoding="utf-8"))
j["positive_control_2026_09_22"] = {
    "object": "use_deter_comp=true 两次独立 100 步运行（保留 checkpoint）",
    "ckpt_result": "473/473 张量逐位相同（不同=0）",
    "contrast": "use_deter_comp=false 两次运行：320/473 张量不同",
    "implication": ["方法灵敏度已证明（该相同时确实报相同）",
                    "该非确定性被 use_deter_comp 完整门控（逐步量 + 落盘权重两个可观测量同时成立）"],
    "limits": ["不做算子级归因", "该开关改变前向计算 ⇒ 只能作诊断，不能当出货配置"]
}
j["next_actions"] = [x for x in j.get("next_actions", []) if not x.startswith("①")]
j["next_actions"].insert(0, "① ~~建 e：固定完整状态的一步回放~~ **已由『落盘权重对照 + 正对照』取代**"
                            "（无需框架内部接口，且已给出正对照）；如需算子级归因，再考虑分段回放")
json.dump(j, io.open(P, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
print("PROTO_UPDATED 已加 positive_control 并更新 next_actions")

# ③ 决策记录：改掉已过时的"缺正对照"
DOC = os.path.join(SKILL, "docs", "推荐配置与保底配置_决策记录_v2_20260922.md")
t = io.open(DOC, encoding="utf-8").read()
lines = t.splitlines(True)
out, changed = [], 0
for ln in lines:
    if "缺正对照" in ln:
        out.append("- ✅ **正对照已补（2026-09-22）**：重跑 2 次 `use_deter_comp=true`（100 步、保留 checkpoint）⇒"
                   " 落盘权重 **473/473 逐位相同**（`CKPT_IDENTICAL`），而 `false` 侧为 320/473 不同 ⇒"
                   " **方法灵敏度已证明**，且该非确定性被该开关完整门控。\n")
        changed += 1
    elif ln.startswith("1. 重跑 2 次 `use_deter_comp=true`"):
        out.append("1. ✅ **已完成**（正对照已补，见 8.3）。\n")
        changed += 1
    else:
        out.append(ln)
if changed:
    io.open(DOC, "w", encoding="utf-8", newline="").write("".join(out))
    print("DOC_PATCHED 决策记录改掉 %d 处已过时表述" % changed)
else:
    print("DOC_SKIP 未找到待改表述（可能已改）")

# ④ 重打包 + 清单 + 核对
subprocess.run([PY, os.path.join(TOOLS, "safe_pack_sync.py"), "--apply"],
               capture_output=True, text=True, encoding="utf-8")
r = subprocess.run([PY, os.path.join(TOOLS, "safe_pack_sync.py"), "--verify-only"],
                   capture_output=True, text=True, encoding="utf-8", errors="replace")
print("VERIFY %s" % [l for l in (r.stdout or "").splitlines() if "SAFE_PACK" in l][-1:])
m = subprocess.run([PY, os.path.join(TOOLS, "evidence_manifest.py"), "--write", "--dir", EV],
                   capture_output=True, text=True, encoding="utf-8", errors="replace")
print("MANIFEST %s" % (m.stdout or "").strip().splitlines()[-1])
print("FINALIZE_POSCONTROL_DONE")
