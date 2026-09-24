# T4 · GatedDeltaNet 算子攻坚（复制到 dsh 会话）

先阅读 workflow/rules/ascend-facts.md 和 workflow/skills/02-op-migrate/SKILL.md。

任务：执行 T4——迁移 GatedDeltaNet（核心难点，无现成实现）。

严格按三级决策流，**第一步只做探测，不写实现**：
1. 探测：在官方样例与参考实现中搜索 `chunk_gated_delta_rule`、`causal_conv1d`、delta rule 状态更新有无现成或近似实现；
2. 产出 docs/op_probe_gdn.md：数学含义（delta rule 递推公式）、输入输出形状 [B,T,H,K]→[B,T,H,V]、依赖原语、每条结论标来源，推测项标【待核实】；
3. 决策：选择第 2 级（纯 PyTorch 组合实现，先求对）作为保底，给出等价性论证；
4. 实现计划经我确认后写码；第 3 级（Triton）仅在保底版精度通过且有性能需求时启动，启动前必须留有基线数据；
5. 验证调用 03 技能，GDN 状态更新层坚持余弦 >0.999。

这是答辩创新点的来源，探测报告要写扎实。先给探测与实现计划。
