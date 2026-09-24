# 本地后台作业：等四臂批次跑完 → 拉日志 → 分析（一次启动，期间我可以不在）
$ErrorActionPreference = "Continue"
$T = "C:\Users\HUAWEI\Desktop\转交给codex的内容\04_核心资产\工具_重建"
$PY = "D:\MInconda\python.exe"
$evid = "C:\Users\HUAWEI\Desktop\转交给codex的内容\04_核心资产\远端证据_20260922\batch4"
New-Item -ItemType Directory -Force $evid | Out-Null

# 找到本次批次的目录
$base = $null
for ($i = 1; $i -le 10; $i++) {
    $out = & $PY "$T\_rsh.py" --file "$T\_rcmd_batch4_check.sh" --timeout 60 2>&1 | Out-String
    if ($out -match 'BATCH4_BASE=(\S+)') { $base = $Matches[1]; break }
    Start-Sleep -Seconds 10
}
if (-not $base) { Write-Output "BATCH4_JOB_FAIL 找不到批次目录"; exit 1 }
Write-Output "BATCH4_JOB base=$base"

$done = $false
for ($i = 1; $i -le 120; $i++) {
    $out = & $PY "$T\_rsh.py" --file "$T\_rcmd_batch4_check.sh" --timeout 60 2>&1 | Out-String
    if ($out -match 'BATCH4_ALL_DONE') { $done = $true; Write-Output "WAIT_DONE after $i polls"; break }
    if ($i % 6 -eq 0) { Write-Output ("poll $i : " + ($out -replace "\s+", " ").Trim()) }
    Start-Sleep -Seconds 25
}
Write-Output ("done=" + $done)

$files = @()
foreach ($arm in "a","b","c","d") { foreach ($r in 1,2) {
    $files += "$base/${arm}_r${r}/train.log"
    $files += "$base/${arm}_r${r}/rc.txt"
} }
Write-Output "=== 拉日志 ==="
& $PY "$T\pull_evidence.py" --out $evid --files @files 2>&1 | Select-Object -Last 6 | Out-String | Write-Output

Write-Output "=== 分析 ==="
$aa = @("$T\_batch4_analyze.py")
foreach ($arm in "a","b","c","d") { foreach ($r in 1,2) {
    $bn = ($base -replace '.*/', '')
    $local = Join-Path $evid ("${bn}__${arm}_r${r}__train.log")
    if (Test-Path $local) { $aa += "${arm}_r${r}=$local" }
} }
& $PY @aa 2>&1 | Out-String | Write-Output
Write-Output "BATCH4_JOB_DONE"
