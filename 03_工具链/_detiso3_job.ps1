# 本地后台作业：v3（仅 torch 层开关）/ v4（三者全设，正对照）→ 拉日志 → 核对注入 → 判重复性
$ErrorActionPreference = "Continue"
$T = "C:\Users\HUAWEI\Desktop\转交给codex的内容\04_核心资产\工具_重建"
$PY = "D:\MInconda\python.exe"
$evid = "C:\Users\HUAWEI\Desktop\转交给codex的内容\04_核心资产\远端证据_20260922\detiso3"
New-Item -ItemType Directory -Force $evid | Out-Null

$base = $null
for ($i = 1; $i -le 12; $i++) {
    $out = & $PY "$T\_rsh.py" --file "$T\_rcmd_detiso3_check.sh" --timeout 60 2>&1 | Out-String
    if ($out -match 'DETISO3_BASE=(\S+)') { $base = $Matches[1]; break }
    Start-Sleep -Seconds 8
}
if (-not $base) { Write-Output "DETISO3_JOB_FAIL 找不到目录"; exit 1 }
Write-Output "DETISO3_JOB base=$base"

$done = $false
for ($i = 1; $i -le 90; $i++) {
    $out = & $PY "$T\_rsh.py" --file "$T\_rcmd_detiso3_check.sh" --timeout 60 2>&1 | Out-String
    if ($out -match 'DETISO3_ALL_DONE') { $done = $true; Write-Output "WAIT_DONE after $i polls"; break }
    if ($i % 8 -eq 0) { Write-Output ("poll $i : " + ($out -replace "\s+", " ").Trim()) }
    Start-Sleep -Seconds 25
}
Write-Output ("done=" + $done)

$files = @()
foreach ($arm in "v3","v4") { foreach ($r in 1,2) {
    $files += "$base/${arm}_r${r}/train.log"
    $files += "$base/${arm}_r${r}/rc.txt"
} }
Write-Output "=== 拉日志 ==="
& $PY "$T\pull_evidence.py" --out $evid --files @files 2>&1 | Select-Object -Last 3 | Out-String | Write-Output

$bn = ($base -replace '.*/', '')
Write-Output "=== 注入生效核对（DETSC 标记；每轮应有多个进程各打一行）==="
foreach ($arm in "v3","v4") { foreach ($r in 1,2) {
    $p = Join-Path $evid "${bn}__${arm}_r${r}__train.log"
    if (Test-Path $p) {
        $n = (Select-String -Path $p -Pattern 'DETSC active' -SimpleMatch:$false | Measure-Object).Count
        $f = (Select-String -Path $p -Pattern 'DETSC FAILED' | Measure-Object).Count
        Write-Output ("  ${arm}_r${r}: DETSC active = $n，DETSC FAILED = $f")
    }
} }

function Analyze($label, $tag, $l1, $l2, $outJson) {
    Write-Output "=== $label ==="
    & $PY "$T\logs_to_results.py" --out $outJson --data-identity "mock_1img_cutoff1024_dp2_nonofficial" "$tag`_r1#$tag=$l1" "$tag`_r2#$tag=$l2" 2>&1 | Select-Object -First 1 | Out-String | Write-Output
    & $PY "$T\numeric_diag_v2.py" --protocol "$T\protocols\numeric_diag_phase1b_v2.json" --results $outJson 2>&1 |
        Select-String -Pattern '重复性要求|同臂 |NUMERIC_V2' | ForEach-Object { $_.Line } | Out-String | Write-Output
}
$v3a = Join-Path $evid "${bn}__v3_r1__train.log"; $v3b = Join-Path $evid "${bn}__v3_r2__train.log"
$v4a = Join-Path $evid "${bn}__v4_r1__train.log"; $v4b = Join-Path $evid "${bn}__v4_r2__train.log"
if ((Test-Path $v3a) -and (Test-Path $v3b)) { Analyze "v3 仅 torch.use_deterministic_algorithms(True)" "V3" $v3a $v3b "$evid\results_v3.json" }
if ((Test-Path $v4a) -and (Test-Path $v4b)) { Analyze "v4 三者全设（正对照）" "V4" $v4a $v4b "$evid\results_v4.json" }
Write-Output "DETISO3_JOB_DONE"
