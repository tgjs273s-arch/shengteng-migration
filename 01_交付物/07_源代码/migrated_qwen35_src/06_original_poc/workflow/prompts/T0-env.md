# T0 · 环境自检（复制到 dsh 会话）

先阅读 workflow/rules/ascend-facts.md 和 workflow/skills/01-env-gate/SKILL.md。

任务：执行 T0 环境自检。昇腾环境信息：`<在此填写服务器地址/登录方式>`。

要求：
1. 按 01 技能的五步逐项执行（卡可见→CANN激活→torch_npu可用→版本配套核对→最小算子冒烟）；
2. 版本配套必须与官方配套表逐行核对，不一致立即停止并报告；
3. 产出 reports/env_report.md（含结果表与结论）；
4. 全程日志存 logs/T0_env_<时间戳>.log；
5. 任何一步失败：给出诊断建议，不要猜测性重试超过 2 次。

执行前先给我五步的执行计划。
