#!/bin/bash
echo "########## 归档 A/B 的每轮耗时（用于规划批次） ##########"
python3 - <<'PY'
import json
p="/root/qwen35-ascend-migrator/out/ab/summary.json"
try:
    d=json.load(open(p))
except Exception as e:
    print("no out/ab/summary.json:", e); raise SystemExit
print("schema",d.get("schema"),"steps",d.get("steps"),"order",d.get("order_used"))
for r in d.get("runs",[]):
    print("  %-8s %-4s valid=%s win_med=%s wall=%ss" % (
        r.get("tag"), r.get("variant"), r.get("valid"),
        r.get("window_median_ms") or r.get("window_11_end_median_ms"), r.get("wall_seconds")))
PY
echo "########## grad_norm 的打印行（日志格式） ##########"
grep -m2 -n "grad norm" /root/qwen35-ascend-migrator/out/train/train.log
echo "########## 70_judge 是否消费 grad_norm ##########"
grep -rn "grad_norm\|grad norm" /root/qwen35-ascend-migrator/sk04_judge/scripts/*.py 2>/dev/null | head -12
echo "########## 59/63 的 PAT 正则原文（单一来源） ##########"
sed -n '38,50p' /root/qwen35-ascend-migrator/scripts/59_config_ab.py
echo "########## 框架是否有 mirror 副本需要同步（坑：修一条路径不问对称的那条） ##########"
find /root/MindSpeed-MM -name "clip_grad_norm.py" 2>/dev/null
find /root/MindSpeed-MM -path "*fsdp/train/train_engine.py" 2>/dev/null
md5sum /root/MindSpeed-MM/mindspeed_mm/fsdp/optimizer/clip_grad_norm.py \
       /root/MindSpeed-MM/mindspeed_mm/fsdp/train/train_engine.py 2>/dev/null
echo "########## 当前 A 臂配置的关键项 ##########"
grep -n "train_iters\|global_batch_size\|micro_batch_size\|data_parallel_size\|clip_grad" \
  /root/qwen35-ascend-migrator/out/plan/train_config.yaml
echo "########## 磁盘余量与 GPU 空闲 ##########"
df -h /root | tail -2
echo "########## DONE ##########"
