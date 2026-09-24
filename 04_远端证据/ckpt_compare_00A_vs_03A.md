# 落盘权重逐张量比对（同配置两次运行）

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
