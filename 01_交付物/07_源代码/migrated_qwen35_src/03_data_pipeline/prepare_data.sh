#!/bin/bash
# ============================================================
# 数据准备：LLaVA-Instruct-150K 提示词 + COCO2017 全量图片 → 训练格式 json
# 依据：官方《数据及训练配置、脚本》PDF；转换脚本为 MindSpeed-MM 自带
# 校验锚点：llava json = 228,941,895 字节；转换产物 = 157,712 样本
# ============================================================
set -e
DATA_DIR=${1:-/root/data}
MSMM_DIR=${MSMM_DIR:-/root/MindSpeed-MM}

mkdir -p "$DATA_DIR/llava" "$DATA_DIR/coco"

echo "=== [1/3] 下载 LLaVA 提示词（约 228MB）==="
modelscope download --dataset AI-ModelScope/LLaVA-Instruct-150K \
  llava_instruct_150k.json --local_dir "$DATA_DIR/llava"

echo "=== [2/3] 下载 COCO2017 训练图片（约 19GB，建议后台执行）==="
cd "$DATA_DIR/coco"
modelscope download --dataset PAI/COCO2017 train2017.zip --local_dir .
unzip -q -o train2017.zip
cd - >/dev/null

echo "=== [3/3] 转换为训练格式 ==="
python3 "$MSMM_DIR/mindspeed_mm/fsdp/tools/data_tool/llava_instruct_2_mllm_demo_format.py" \
  --llava_json_path "$DATA_DIR/llava/llava_instruct_150k.json" \
  --coco_path "$DATA_DIR/coco" \
  --output_json_path "$DATA_DIR/output_llava_coco_data.json"

echo "=== 校验 ==="
LLAVA_BYTES=$(stat -c%s "$DATA_DIR/llava/llava_instruct_150k.json")
IMAGES=$(ls "$DATA_DIR/coco/train2017" | wc -l)
python3 - <<PY
import json
n = len(json.load(open("$DATA_DIR/output_llava_coco_data.json")))
print("LLaVA 字节数 : %d  （期望 228941895）" % $LLAVA_BYTES)
print("COCO 图片数  : %d  （期望 118287）" % $IMAGES)
print("转换后样本数 : %d  （期望 157712）" % n)
PY
echo "数据准备完成: $DATA_DIR/output_llava_coco_data.json"
