#!/bin/bash
# §4-2 只读侦察 2：把注入点的源码原文取回来（拒绝凭记忆）
TE=/root/MindSpeed-MM/mindspeed_mm/fsdp/train/train_engine.py
CG=/root/MindSpeed-MM/mindspeed_mm/fsdp/optimizer/clip_grad_norm.py
echo "########## train_engine.py L85-100  average_losses_across_data_parallel_group ##########"
sed -n '85,100p' $TE
echo "########## train_engine.py L125-195  train_step ##########"
sed -n '125,195p' $TE
echo "########## train_engine.py L255-280  train() 优化器序列 ##########"
sed -n '255,280p' $TE
echo "########## clip_grad_norm.py L1-70 ##########"
sed -n '1,70p' $CG
echo "########## clip_grad_norm.py L95-175 ##########"
sed -n '95,175p' $CG
echo "########## 指纹工具是否用到 grad_norm ##########"
grep -n "grad_norm\|grad norm\|loss" /root/qwen35-ascend-migrator/scripts/51_train_fp.py 2>/dev/null | head -25
echo "########## _batchfp.py 指纹字段 ##########"
grep -n "grad_norm\|def .*fingerprint\|loss" /root/qwen35-ascend-migrator/scripts/_batchfp.py 2>/dev/null | head -25
echo "########## DONE ##########"
