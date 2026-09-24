#!/bin/bash
# 把作废批次标注为 VOID（**改名 + 写原因，不删除**）
set -u
R=/root/ops/p2ab_20260922_162640
if [ ! -d "$R" ]; then echo "SKIP 目录不存在: $R"; exit 0; fi
cat > "$R/VOID_REASON.txt" <<'EOS'
★ 本批次作废（VOID）—— 不是失败，也不是成功，是"没有候选信息"。

原因（详见 docs/PITFALLS_坑表.md 坑 217）：
  6 轮（N,C1,C1,N,N,C1）的 arm_file_sha256 **全部等于原文件** fefdb197e98a，
  连候选臂 C1 也是 ⇒ "候选 vs 基线"实际退化成"基线 vs 基线"。
  根因：_p2_patch.py 的 apply 把"备份已存在"误当成"已打补丁"而 SKIP 返回 0；
  更深一层是编排器**只记录** sha 却**没有断言**它，于是 status.json 仍报
  verification_passed=true（判据只挂在 rc/步数/还原上，这三项在"基线 vs 基线"下全过）。

可继续使用的部分：这 6 轮本身是**有效的基线重复**（6 轮同批次基线墙钟
  128.8/129.7/131.7/132.7/130.0/131.7 s ⇒ 同批次底噪 ≈3.0%），
  可作噪声参考；但**不得**用于任何候选比较。

对照：修正后的有效批次为 /root/ops/p2ab_20260922_164204
  （00_N/03_N/04_N sha=fefdb197e98a；01_C1/02_C1/05_C1 sha=591a08714cc0）。

⛔ 不得删除本目录，也不得把它改写为成功。
EOS
cd /root/ops
if [ ! -d "${R}_VOID" ]; then mv "$R" "${R}_VOID"; echo "RENAMED -> ${R}_VOID"; else echo "已改过名"; fi
ls -d /root/ops/p2ab_* 2>/dev/null
echo "VOID_MARK_DONE"
