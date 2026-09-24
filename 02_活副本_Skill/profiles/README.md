# profiles/ —— v2 的"模型档案"扩展点（当前为**空占位**，v1 不用它）

## 这是什么

v2 泛化路线（见 `SKILL.md` §8）要把**迁移点识别规则外置**成模型档案：

```
profiles/
  qwen3_5.json      # 规则：linear_attn_GatedDeltaNet / full_attn / vision_3d_conv_patch / custom_act_or_norm
  <其他模型>.json
```

P1（`scripts/10_analyze_points.py`）读入档案 → 用同一套 AST 扫描器适配任意 HF 多模态模型。

## 当前状态（诚实说明）

* **v1 未使用本目录**：4 类规则**内置在** `scripts/10_analyze_points.py`，只服务 Qwen3.5 系列；
* 本目录**故意留空**作为 v2 扩展点，**不是**未完成品 —— 但也**不要**把它读成"已支持任意模型"；
* 通用性至今**只在 1 个模型（Qwen3.5-0.8B）上验证过**。

## 为什么不放一个空目录了事

空目录在交付包里容易被误读为"功能缺失"。所以这里放一份说明：
**要么**等 v2 真正把规则外置后往这里放 `*.json`；**要么**在 v1 交付包里删除本目录（二选一，不要两者都不是）。
