# 本地后台作业：等预取实验跑完 → 拉 6 份日志 → 跑分析器（一次启动，最后收结果）
$ErrorActionPreference = "Continue"
$T = "C:\Users\HUAWEI\Desktop\转交给codex的内容\04_核心资产\工具_重建"
$PY = "D:\MInconda\python.exe"
$base = "/root/ops/pref_20260922_120450"
$evid = "C:\Users\HUAWEI\Desktop\转交给codex的内容\04_核心资产\远端证据_20260922\prefetch"
New-Item -ItemType Directory -Force $evid | Out-Null

$done = $false
for ($i = 1; $i -le 70; $i++) {
    $out = & $PY "$T\_rsh.py" --file "$T\_rcmd_pref_check.sh" --timeout 60 2>&1 | Out-String
    if ($out -match 'ALL_DONE_YES') { $done = $true; Write-Output "WAIT_DONE after $i polls"; Write-Output $out; break }
    if ($i % 5 -eq 0) { Write-Output ("poll $i : " + ($out -replace "\s+", " ").Trim()) }
    Start-Sleep -Seconds 25
}
if (-not $done) { Write-Output "WAIT_TIMEOUT 实验未在预算内结束（继续拉已有日志）" }

$files = @()
foreach ($v in 1,2,4) { foreach ($r in 1,2) {
    $files += "$base/v${v}_r${r}/train.log"
    $files += "$base/v${v}_r${r}/rc.txt"
} }
Write-Output "=== 拉日志 ==="
& $PY "$T\pull_evidence.py" --out $evid --files @files 2>&1 | Select-Object -Last 14 | Out-String | Write-Output

Write-Output "=== 分析 ==="
$args2 = @("$T\_prefetch_analyze.py")
foreach ($v in 1,2,4) { foreach ($r in 1,2) {
    $local = Join-Path $evid ("pref_20260922_120450__v${v}_r${r}__train.log")
    if (Test-Path $local) { $args2 += "v${v}_r${r}=$local" }
} }
& $PY @args2 2>&1 | Out-String | Write-Output
Write-Output "PREF_JOB_DONE"
