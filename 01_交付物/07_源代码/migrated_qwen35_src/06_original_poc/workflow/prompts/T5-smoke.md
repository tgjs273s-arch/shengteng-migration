# T5 · 端到端冒烟（复制到 dsh 会话）

先阅读 workflow/rules/ascend-facts.md 和 workflow/skills/00-main/SKILL.md。

任务：执行 T5——端到端冒烟测试（图+文输入 → 模型输出）。

要求：
1. 设计冒烟用例：至少 1 张真实图片 + 1 条文本指令，输出应为结构化判定；
2. 运行前核对 backend，确认推理在 NPU（非 CPU 回退）；
3. 判定标准：输出非空、格式正确、与输入语义相关（人工抽查）；
4. 运行带 `2>&1 | tee logs/T5_smoke_<时间戳>.log`；
5. 产出 reports/T5_smoke_report.md：输入、输出、backend 证据、结论。

失败则定位断点（权重/算子/前处理），回流对应阶段。先给冒烟方案。
