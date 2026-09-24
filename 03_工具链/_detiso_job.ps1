# 本地后台作业：等分离实验跑完 → 拉日志 → 用 numeric_diag_v2 判"同臂重复性"
$ErrorActionPreference = "Continue"
$T = "C:\Users\HUAWEI\Desktop\转交给codex的内容\04_核心资产\工具_重建"
$PY = "D:\MInconda\python.exe"
$S = "C:\Users\HUAWEI\Desktop\转交给codex的内容\04_核心资产\qwen35-ascend-migrator_整合版"
$evid = "C:\Users\HUAWEI\Desktop\转交给codex的内容\04_核心资产\远端证据_20260922\detiso"
New-Item -ItemType Directory -Force $evid | Out-Null

$base = $null
for ($i = 1; $i -le 12; $i++) {
    $out = & $PY "$T\_rsh.py" --file "$T\_rcmd_detiso_check.sh" --timeout 60 2>&1 | Out-String
    if ($out -match 'DETISO_BASE=(\S+)') { $base = $Matches[1]; break }
    Start-Sleep -Seconds 8
}
if (-not $base) { Write-Output "DETISO_JOB_FAIL 找不到目录"; exit 1 }
Write-Output "DETISO_JOB base=$base"

$done = $false
for ($i = 1; $i -le 90; $i++) {
    $out = & $PY "$T\_rsh.py" --file "$T\_rcmd_detiso_check.sh" --timeout 60 2>&1 | Out-String
    if ($out -match 'DETISO_ALL_DONE') { $done = $true; Write-Output "WAIT_DONE after $i polls"; break }
    if ($i % 8 -eq 0) { Write-Output ("poll $i : " + ($out -replace "\s+", " ").Trim()) }
    Start-Sleep -Seconds 25
}
Write-Output ("done=" + $done)

$files = @()
foreach ($arm in "v1","v2") { foreach ($r in 1,2) {
    $files += "$base/${arm}_r${r}/train.log"
    $files += "$base/${arm}_r${r}/rc.txt"
    $files += "$base/${arm}_r${r}/launch.out"
} }
Write-Output "=== 拉日志 ==="
& $PY "$T\pull_evidence.py" --out $evid --files @files 2>&1 | Select-Object -Last 4 | Out-String | Write-Output

$bn = ($base -replace '.*/', '')
function Analyze($armName, $tag, $l1, $l2, $outJson) {
    Write-Output "=== $armName（$tag）==="
    & $PY "$T\logs_to_results.py" --out $outJson --data-identity "mock_1img_cutoff1024_dp2_nonofficial" "$tag`_r1#$tag=$l1" "$tag`_r2#$tag=$l2" 2>&1 | Select-Object -First 1 | Out-String | Write-Output
    & $PY "$T\numeric_diag_v2.py" --protocol "$T\protocols\numeric_diag_phase1b_v2.json" --results $outJson 2>&1 |
        Select-String -Pattern '重复性要求|同臂 |NUMERIC_V2' | ForEach-Object { $_.Line } | Out-String | Write-Output
}

$v1a = Join-Path $evid "${bn}__v1_r1__train.log"; $v1b = Join-Path $evid "${bn}__v1_r2__train.log"
$v2a = Join-Path $evid "${bn}__v2_r1__train.log"; $v2b = Join-Path $evid "${bn}__v2_r2__train.log"
if ((Test-Path $v1a) -and (Test-Path $v1b)) { Analyze "v1 只开 HCCL_DETERMINISTIC" "V1" $v1a $v1b "$evid\results_v1.json" }
if ((Test-Path $v2a) -and (Test-Path $v2b)) { Analyze "v2 只关 CLOSE_MATMUL_K_SHIFT" "V2" $v2a $v2b "$evid\results_v2.json" }

# 对照：今日批次4 的基线两轮（两个变量都不设）
$ca = "C:\Users\HUAWEI\Desktop\转交给codex的内容\04_核心资产\远端证据_20260922\batch4\batch4_20260922_122428__a_r1__train.log"
$cb = "C:\Users\HUAWEI\Desktop\转交给codex的内容\04_核心资产\远端证据_20260922\batch4\batch4_20260922_122428__a_r2__train.log"
if ((Test-Path $ca) -and (Test-Path $cb)) { Analyze "对照：都不设（deter=false）" "CTL" $ca $cb "$evid\results_ctl.json" }
Write-Output "DETISO_JOB_DONE"
