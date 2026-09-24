# -*- coding: utf-8 -*-
r"""_finalize_chain.py —— ①落盘 ckpt 比对证据；②把证据链写进 v2 协议；③记坑 187；④更新证据清单

③ 的由来（**我自己的失误，值得记**）：跑阶段 1b 时我在脚本里写了"清 checkpoint（不是本实验的证据）"
并真的 `rm -rf` 掉了 P1/P2 的 checkpoint。**后来我建立的方法（比对落盘权重）恰好需要那些 checkpoint**
——`deter=true` 那一对是这条对照的**正对照**，现在没了，只能重跑。教训：**"这是不是证据"取决于你后来
用什么方法判读，过早清掉等于把未来的对照删了。**
"""
import io
import json
import os
import subprocess
import sys

TOOLS = r"C:\Users\HUAWEI\Desktop\转交给codex的内容\04_核心资产\工具_重建"
EV = r"C:\Users\HUAWEI\Desktop\转交给codex的内容\04_核心资产\远端证据_20260922"
PY = sys.executable

# ---- ① ckpt 比对结果落盘（数值取自远端实跑输出） ----
report = """# 落盘权重逐张量比对（同配置两次运行）

- 方法：`compare_ckpt.py`（**不需要框架内部接口**；权重是"前向→反向→更新"整条路径的累积结果）
- 对象：`/root/ops/AB_AVSB_20260921/00_A/checkpoint/iter_0000100` vs `03_A/.../iter_0000100`
- 配置：两份**同配置**（出厂 A，`use_deter_comp=false`），各 100 步
- 权重文件：`model.safetensors-00001-of-00001.safetensors`（3,412,002,208 B / 每份）

## 结果

```
共同张量 = 473    逐位相同 = 153    不同 = 320    含非有限 = 0

差异最大的张量（max_abs / mean_rel）：
  model.language_model.layers.23.self_attn.q_proj.weight      max_abs=0.000884051  mean_rel=0.00153
  model.language_model.embed_tokens.weight                    max_abs=0.000615392  mean_rel=0.000172
  model.language_model.layers.19.self_attn.k_proj.weight      max_abs=0.0005625    mean_rel=0.0026
  model.language_model.layers.19.self_attn.q_proj.weight      max_abs=0.000418812  mean_rel=0.00165
  model.language_model.layers.13.linear_attn.out_proj.weight   max_abs=0.000371039  mean_rel=0.00244
  model.language_model.layers.20.linear_attn.in_proj_qkv.wei  max_abs=0.000370421  mean_rel=0.000818
  model.language_model.layers.11.mlp.gate_proj.weight         max_abs=0.000365908  mean_rel=0.00222
  model.language_model.layers.17.linear_attn.in_proj_z.weight max_abs=0.000358642  mean_rel=0.00111
```

## 形态（按**事前写死**的判读规则读）

差异**遍布各类模块**（embedding / attention 的 q,k / GDN 的 in_proj_qkv,in_proj_z,out_proj / MLP 的
gate_proj），量级集中 `1e-4 ~ 1e-3`（mean_rel 0.02%–0.26%），**无非有限值**。
⇒ 属"与全局归约/累加顺序类非确定性**形态一致**"这一类，**不是**"某个算子炸了"也不是"只有某层不同"。
另有 153 个张量**逐位相同** ⇒ 并非"整条训练都在乱跑"。

## 边界（不得越读）

1. 权重差异是 **100 步累积结果**，**不能**单独把责任落到"第 1 步反向"上；第 1 步梯度的不可复现
   （`grad_norm` 47.326 / 47.34 / 47.337）是**另一条独立证据**，两者方向一致。
2. 存储为 **bf16** ⇒ "逐位相同"只在该存储精度下成立，更细差异被存盘精度掩盖。
3. 阈值/精度以下的差异本方法看不见。
4. **不做归因结论**：只能说形态一致，不能说"就是 HCCL 归约顺序"。
5. ★ **缺正对照**：`use_deter_comp=true` 那一对（P2）的 checkpoint 被我早前 `rm -rf` 清掉了
   ⇒ 无法在此处给出"确定性路径下权重应逐位相同"的正对照，需要时须重跑（见坑 187）。
"""
io.open(os.path.join(EV, "ckpt_compare_00A_vs_03A.md"), "w", encoding="utf-8", newline="").write(report)
print("① 已落盘 %s" % os.path.join(EV, "ckpt_compare_00A_vs_03A.md"))

# ---- ② 把证据链写进 v2 协议 ----
P = os.path.join(TOOLS, "protocols", "numeric_diag_phase1b_v2.json")
d = json.load(io.open(P, encoding="utf-8"))
d["evidence_chain_2026_09_22"] = [
    "① 启动输入 dump：四份日志（00A/03A/P2r1/P2r2）哈希完全相同 ⇒ 数据侧在打印精度内一致",
    "② 同配置内（deter=false）：第 1 步 loss 打印相同、**第 1 步 grad_norm 即不同**（47.326/47.34/47.337）⇒ 分歧首次出现在**梯度**上",
    "③ 落盘权重（100 步累积）：320/473 张量不同、量级均匀（1e-4~1e-3）、无非有限值 ⇒ 与**全局归约/累加顺序类**非确定性形态一致",
    "④ 开 use_deter_comp=true：100 步 max=0.00%、连第 1 步 grad_norm 都一致 ⇒ 该非确定性**可被门控消除**；但它改变前向（step1 loss 0.1154926→0.1169518）⇒ **只能作诊断，不能当出货配置**",
    "⑤ 由此收窄：e（固定状态一步回放）只需在**第 1 步**比较前向/局部梯度/一次更新，实验量降为 1–2 次 1 步",
    "★ 交付可写结论：同配置重复未满足完整窗口逐点一致性；证据指向反向/梯度路径的**归约类**非确定性（形态一致，未做算子级归因）；该现象可由 use_deter_comp 门控消除，但开启会改变前向计算，故仅作诊断。"
]
json.dump(d, io.open(P, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
print("② v2 协议已补 evidence_chain（%d 条）" % len(d["evidence_chain_2026_09_22"]))

# ---- ③ 坑 187 ----
ROW = '''| **187** | ★★ **我把"当时的非证据"清掉了，而它正是我后来方法的唯一正对照**。跑阶段 1b 时，我在脚本里写下"清 checkpoint（不是本实验的证据）"并真的 `rm -rf` 了 P1/P2 的 checkpoint（每份 3.2 GB）。**几十分钟后**我建立了比对落盘权重的方法（不需要框架内部接口，是回答"第一处不同"最便宜的客观量），而这条方法**恰好需要那些 checkpoint**：`deter=true` 那两轮是"确定性路径下权重应逐位相同"的**正对照**。结果只能做到"同配置两次不同（320/473 张量）"，**给不出正对照** ⇒ 缺一个"该方法是够灵敏、且这个量本来可以相同"的基准 | ① **"这是不是证据"取决于后来用什么方法判读**，而我在**还不知道要问什么**的时候就定了性——"不是本实验的证据"这句话当时就**没有依据**（我只是想省磁盘：两个 checkpoint 6.4 GB）；② 更一般的形态：**清理动作是不可逆的，而清理判据是当下的、可变的**（与坑 180"跑没读过的脚本"同族：都是"不可逆操作 + 判据不足"）；③ 而且我**在协议里正是用这批 checkpoint 做过的另一件事**（P2 的生效取证是读 `train.log`，但复盘需要权重）——同一批产物有两个用途，我只按当时想到的那个判了 | ① 记下这条并**明确标注"缺正对照"**（已写进 ckpt 比对证据文件的"边界"栏，不假装该方法已经完整）；② 定规矩：**清理任何运行产物前先问"未来判读可能需要它吗"**，答不上来就保留（磁盘 300 GB，当前仅用 31% ⇒ 没有清理的紧迫性）；③ 需要正对照时**重跑 2 次确定性配置并保留 checkpoint**（成本约 3 分钟机时 + 6.4 GB，已列入待办） | 现场：阶段 1b 脚本内 `rm -rf "$ROOT/cfg_save"` 每轮执行；P2 脚本内同样 `rm -rf "$ROOT/cfg_save"`（`P2_DONE` 后无 checkpoint 残留）；随后 `compare_ckpt.py` 只能比对 `AB_AVSB_20260921/{00_A,03_A}`（该目录的 checkpoint 未被清理 ✓ 每份 3.2 GB），得到 `逐位相同=153 不同=320 含非有限=0`，**无法给出 deter=true 的正对照** |
'''
rf = os.path.join(TOOLS, "_rows", "_row187.md")
os.makedirs(os.path.dirname(rf), exist_ok=True)
io.open(rf, "w", encoding="utf-8", newline="").write(ROW)
r = subprocess.run([PY, os.path.join(TOOLS, "_append_pitfall.py"), "--row", rf,
                    "--expect-max", "187"], capture_output=True, text=True,
                   encoding="utf-8", errors="replace")
print("③ 追加 187：rc=%d | %s" % (r.returncode,
      ((r.stdout or "") + (r.stderr or "")).strip().splitlines()[-1]))

# ---- ④ 证据清单更新 ----
r2 = subprocess.run([PY, os.path.join(TOOLS, "evidence_manifest.py"), "--write", "--dir", EV],
                    capture_output=True, text=True, encoding="utf-8", errors="replace")
print("④ 清单：%s" % ((r2.stdout or "").strip().splitlines()[-1]))
sys.exit(r.returncode | r2.returncode)
