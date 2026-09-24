# T2 · 权重转换与加载（复制到 dsh 会话）

先阅读 workflow/rules/ascend-facts.md 和 workflow/skills/00-main/SKILL.md。

任务：执行 T2——把 Qwen3.5 的 HuggingFace 权重转换并加载到迁移框架。

要求：
1. 先产出权重格式分析：config.json 关键项、safetensors 层名结构、与目标框架的层名映射表；
2. 转换方案必须引用官方工具（如 mm-convert）或官方文档章节，标注来源；无官方途径时说明替代方案并标【待核实-昇腾】；
3. 实现转换脚本，加载后做首层输出对比（与基线一致）；
4. 运行带 `2>&1 | tee logs/T2_weight_<时间戳>.log`；
5. 产出 docs/weight_mapping.md（层名映射表）+ 加载成功证据。

先给转换计划（映射策略、风险点、验证方法），确认后实施。
