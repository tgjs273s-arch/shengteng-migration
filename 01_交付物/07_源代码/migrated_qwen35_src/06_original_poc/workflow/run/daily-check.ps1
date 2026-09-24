# ============================================================
# daily-check.ps1 —— DSH headless 每日体检
# 用法（本地执行）：右键"使用 PowerShell 运行"，或在终端执行
#   powershell -ExecutionPolicy Bypass -File .\daily-check.ps1
# 作用：让 dsh 汇总当日运行日志，产出检查报告
# ============================================================

$dsh = "D:\develop\bin\dsh.cmd"
$project = "C:\Users\HUAWEI\Desktop\qwen3_5-migrate-poc"

$task = @"
当前工作区为 $project。请阅读 workflow/skills/03-verify/SKILL.md 中的日报支持要求，
检查 logs/ 目录下今日新增的所有日志文件，按以下结构汇总到 reports/daily_check.md：
1. 通过项（阶段/任务/证据文件）
2. 失败项（阶段/任务/错误摘要/建议下一步）
3. 待人工决策项（需用户判断的问题清单）
4. 明日建议任务（按总控技能 00-main 的阶段门禁推断当前应处的阶段）
只基于日志事实汇总，不要臆测未发生的运行。
"@

# 注意：若你的 dsh 版本需要额外参数（如指定 workspace），请按 --help 补充
& $dsh --profile headless $task
