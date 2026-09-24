#!/bin/bash
CG=/root/MindSpeed-MM/mindspeed_mm/fsdp/optimizer/clip_grad_norm.py
echo "########## clip_grad_norm.py L40-70 (非 EP 路径：归约与是否裁剪) ##########"
sed -n '40,70p' $CG
echo "########## clip_grad_norm.py L95-175 (EP 路径 + _fsdp2_reduce_group) ##########"
sed -n '95,175p' $CG
echo "########## 指纹/日志是否含 grad_norm ##########"
echo "--- train.log 的 loss / grad norm 行样本 ---"
grep -m3 -n "grad norm" /root/qwen35-ascend-migrator/out/train/train.log 2>/dev/null
echo "--- 51_train_fp.py 取哪些字段 ---"
grep -n "loss\|grad_norm\|iter_ms\|PAT" /root/qwen35-ascend-migrator/scripts/51_train_fp.py 2>/dev/null | head -20
echo "--- _batchfp.py 取哪些字段 ---"
grep -n "loss\|grad_norm\|iter_ms\|re.compile" /root/qwen35-ascend-migrator/scripts/_batchfp.py 2>/dev/null | head -20
echo "--- 59_config_ab.py 的 PAT（计时/判据正则，单一来源） ---"
grep -n "PAT = \|PAT=" /root/qwen35-ascend-migrator/scripts/59_config_ab.py 2>/dev/null | head
echo "########## DONE ##########"
