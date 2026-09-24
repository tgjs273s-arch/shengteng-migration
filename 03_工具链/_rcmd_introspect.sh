#!/bin/bash
# 远端工作⑭：一次回放实现所需的**精确内省**（只读，输出刻意压到最小）
set -u
cd /root/MindSpeed-MM
F=mindspeed_mm/fsdp/train/train_engine.py
echo "=== TrainEngine.__init__ 里 self.* 赋值 ==="
sed -n '31,90p' $F | grep -nE 'self\.[a-z_]+ *=' | head -25
echo "=== train() 里 231-280 的关键行 ==="
sed -n '231,280p' $F | grep -nE 'self\.|for |next\(|clip_grad|step\(\)|zero_grad|loss' | head -30
echo "=== train_step 130-196 的 self.* 使用（去重）==="
sed -n '130,196p' $F | grep -oE 'self\.[a-z_]+' | sort -u | head -20
echo "=== args 里与 clip_grad 有关的字段 ==="
grep -nE 'clip_grad' mindspeed_mm/fsdp/params/training_args.py | head -6
echo "=== TrainEngine 是否持有 args ==="
grep -nE 'args' $F | head -6
echo "INTROSPECT_DONE"
