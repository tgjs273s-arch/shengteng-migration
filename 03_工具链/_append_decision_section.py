# -*- coding: utf-8 -*-
r"""_append_decision_section.py —— 把 2026-09-22 当日实测证据追加进决策记录 v2（评审要看的活文档）

为什么现在做：开关类优化候选（预取深度、log_interval）已全部被排除，而性能主结论仍只有
A vs B 的 19.6%（同批次同机）。此时对交付物最有价值的不是再薅 1–2%，而是**把今天的实测证据与
边界写进评审会读的那份文档**：正确性定位、数据侧身份、profile 归因、噪声底噪规则、已排查清单。
"""
import io
import os
import shutil
import time

DOC = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                   "qwen35-ascend-migrator_整合版", "docs",
                   "推荐配置与保底配置_决策记录_v2_20260922.md")
MARK = "## 9. 2026-09-22 当日实测证据（追加，不改上文结论）"

SECTION = """
---

## 9. 2026-09-22 当日实测证据（追加，不改上文结论）

> 本节只**追加**当日新证据与边界，不修改第 0–8 节的任何结论。所有运行均为**同机同批次**；
> 机器为 `199.98.58.213`（host `4ac9ca3712ba`，npu-smi 26.1.1、torch_npu 2.7.1.post10），
> **与 09-21 那批（`199.103.55.150`，CANN 9.1.0-beta.3）不是同一台** ⇒ **绝对步时长不可跨批对比**。

### 9.1 数据侧：从"启动 dump 相同"升级为"100 步逐字节相同"
- 新工具 `scripts/51_train_fp.py` + `scripts/_batchfp.py`（只读代理，覆写 `get_dataloader` 后在最外层套一层；
  记录每微批每字段的 dtype/shape/nbytes/sha256）。
- 结果（`protocols/batch_fingerprint_20260922.json`）：
  - 两轮同配置的指纹文件**逐字节相同**（`sha256=9b8d5edb62a9`）；
  - 全窗口 100 步内 **不同 micro sha 的个数 = 1** ⇒ 本 mock 数据集**每步喂的是同一批**；
  - **`RANKS_SAME_BATCH=True`**：dp2 两个 rank 拿到的也是同一批；
  - **负向对照**：把副本数据集里一个真实文本字段改一个字符（派生配置**只差 1 个键**）⇒
    指纹改变 262 行，且变化精确落在 `input_ids/labels/attention_mask/position_ids/rope_deltas`、
    `pixel_values/image_grid_thw` **不变** ⇒ 判据灵敏且归因正确。
- **推论（重要）**：P0 的同臂分歧**不能**归因于"数据随步变化"或"rank 间数据不同"；
  同时**收紧**一条旧假设：`num_workers=1`（P1）那一臂在这份数据上**无法检验它声称要检验的东西**。

### 9.2 正确性：分岔定位到**反向传播段**（段落级定位）
- 新工具 `scripts/52_replay.py` + `scripts/_replay_diff.py`（固定完整状态做**一次**前向+反向+一次更新；
  只在最外层包一层只读观测，训练循环仍走框架原代码）。
- 结果（`protocols/one_step_replay_20260922.json`，两轮两 rank 一致）：
  - 前提（输入 / 初始参数 / RNG / 学习率 / 步号）两轮**完全相同**；
  - **C1 前向输出（loss 与 logits）逐字节相同**；
  - **C3 通信后梯度不同**（逐张量 200/200；抽样元素最大相对差 106.75% / 379.65%）；
  - C2（通信**前**局部梯度）在 FSDP2 下用户侧**不可观测** ⇒ 记 `UNVERIFIED`，**未用 C3 顶替**；
  - C4（一次更新后参数）**无区分力**：该配置步 0 学习率为 0 ⇒ 空更新。
- **结论**：**分歧起源于反向传播段**（C1 相同、C3 不同）。这与机制证据互相印证：
  `use_deter_comp=true` 会设 `HCCL_DETERMINISTIC=True` 与 `CLOSE_MATMUL_K_SHIFT=1`
  （关掉 matmul K 轴 shift），两者都作用在反向的 matmul/集合通信数值路径上。
- **仍未闭合**：无法进一步区分"反向计算本身"与"反向中的集合通信"（受 C2 不可观测所限）。

### 9.3 性能：profile 三层归因 ⇒ 方向是"等待/同步"，而**开关类候选已被排除**
- 桶层（`protocols/profile_short_20260922.json`）：`Computing 70.9–78.0%`、`Free 19.3–26.6%`、
  `Communication 5.2–5.6%`、**未重叠通信仅 2.5–2.7%** ⇒ **通信不是瓶颈**（两轮一致）。
- 算子层：**GDN/linear-attn 族 ≈28.2%**（> matmul 族 20.5%）；小算子/拷贝/cast 簇 ≈16.3%、**7150 发射/步**。
- 栈层（with-stack，`Device Self Duration`）：**`wait_event` 合计 46.60%**，其中
  **`reduce_scatter_tensor` 17.45%**、叶子帧 `wait` 16.60%；**重计算仅 0.78%**。
  ⇒ 方向 = **重叠/同步不足**；同时**排除**"按模块重计算"为省时间的手段
  （若做 chunk loss，动机只能是**显存**：logits/loss 链占峰值 ≈5.86 GB / 8.5 GB）。
- **已排查且均未达标（同批次四臂 × 2 轮，8/8 rc=0）**：
  `预取深度 4` = −0.76%（方向不一致）、`log_interval=10` = −2.33%、两者组合 = −2.62%
  ⇒ **没有任何一臂优于基线**；此前一轮报的"预取 4 省 3.91%"**未复现**。
- ★ **噪声底噪 ≈2%**：同机同会话、同配置的两批基线中位步时长为 731.6 ms 与 716.5 ms。
  ⇒ 任何小于**噪声 2 倍**的效应一律先按 UNCERTAIN 处理；要报小效应必须增加重复数并给区间。
  **这也正是"≥5%"门槛的必要性来源。**

### 9.4 能宣称 / 不能宣称（提交前必读）
| 可以宣称 | 不可以宣称 |
|---|---|
| 在本机本 mock 数据下，配置 A 相对 B **省时 19.606%**（同批次同机，3 对，区间 [18.323%, 20.889%]）| 不可宣称"优于官方基线"或给出与官方窗口中位数的比值（数据不同、机器也不同）|
| 100 步窗口内**数据侧逐字节一致**（跨运行、两 rank 同一批）| 不可宣称"所有输入都相同"而不加限定（限于本 mock 数据；换真实 COCO 后每步输入会变）|
| 分歧**起源于反向段**（前向逐字节相同、通信后梯度不同）| 不可宣称"已定位到具体算子/通信"，也不可宣称"数值正确性已闭合"|
| `use_deter_comp=true` 能改善重复性（0/100 超阈、473/473 张量逐位相同）| 不可宣称它可交付：它改变算子数值（第 1 步 loss 就不同），只能作诊断|
| 通信**不是**本配置的瓶颈（未重叠 2.5–2.7%）| 不可宣称"通信无需优化"这种普适结论（仅限本配置/本数据/本机）|

### 9.5 新增的流程规则（已落进工具与协议）
1. **任何性能实验的第一步是"同批次内重复基线，先量噪声"**（本次事后才补做，代价是报了一个未复现的 3.91%）。
2. **跨机/跨批次的绝对时间一律作废**；比较前先核对 `hostname + npu-smi 版本 + CANN 版本 + boot id`。
3. **协议 JSON 可解析性**由闸门 G31 保证；**文档里的闸门数字**由 G30 保证与驱动器一致。
4. 交付物内**禁止凭证与备份文件**（内容闸门 + 写入端自动脱敏）。
"""


def main():
    txt = io.open(DOC, encoding="utf-8").read()
    if MARK in txt:
        print("APPEND_SKIP 该节已存在")
        return 0
    bak_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "_backup", "skill_baks")
    os.makedirs(bak_dir, exist_ok=True)
    shutil.copy2(DOC, os.path.join(bak_dir, "decision_v2.bak_%s" % time.strftime("%Y%m%d_%H%M%S")))
    io.open(DOC, "w", encoding="utf-8").write(txt.rstrip() + "\n" + SECTION)
    back = io.open(DOC, encoding="utf-8").read()
    assert MARK in back and "9.4 能宣称 / 不能宣称" in back
    print("APPEND_OK 已追加第 9 节（%d → %d 字符）" % (len(txt), len(back)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
