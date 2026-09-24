# T8 · Skill 封装与交付（复制到 dsh 会话）

先阅读 workflow/rules/ascend-facts.md 和 workflow/skills/05-package/SKILL.md。

任务：执行 T8——把已跑通的迁移流程封装成复赛要求的 Skill 文件夹。

要求：
1. 按 05 技能的目标结构组装 skill_qwen35/（analyze/migrate/verify/report + SKILL.md + README.md）；
2. 所有绝对路径改为配置/参数传入，确保可移植；
3. 封装后在干净环境完整重跑一次全链路（防止封装引入回归），日志留档；
4. 逐项核对 05 技能的"8 项复赛交付物自查清单"，报告每项就绪状态；
5. 打包 skill_qwen35.zip。

注意：封装不改语义，只做工程化整理。先给封装计划（文件映射关系 + 回归验证方案），确认后执行。
