# T3 · 标准算子迁移（复制到 dsh 会话）

先阅读 workflow/rules/ascend-facts.md 和 workflow/skills/02-op-migrate/SKILL.md。

任务：执行 T3——迁移标准算子（Conv3d/RMSNorm/全注意力等）。

要求：
1. 按 02 技能三级决策流逐个处理：先探测（官方样例/CANN 有无现成），再决策，再实现；
2. 每个算子产出 docs/op_probe_<名称>.md，结论必须标注来源；
3. 每完成一个算子立即调用 03 技能验证（compare.py，余弦>0.999），通过才做下一个；
4. 维护 docs/op_status.md 状态总表；
5. 全程日志留档。

一次只做一个算子。第一个从 `<指定算子，建议 RMSNorm>` 开始，先给探测计划。
