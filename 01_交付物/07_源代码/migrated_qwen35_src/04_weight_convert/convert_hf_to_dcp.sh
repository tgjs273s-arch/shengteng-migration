#!/bin/bash
# ============================================================
# 权重准备：HF 格式 → DCP 格式（MindSpeed-MM 训练加载格式）
# 用法: bash convert_hf_to_dcp.sh <hf_dir> <dcp_dir>
# ============================================================
set -e
HF_DIR=${1:-/root/Qwen3.5-0.8B-hf}
DCP_DIR=${2:-/root/Qwen3.5-0.8B-dcp}
MSMM_DIR=${MSMM_DIR:-/root/MindSpeed-MM}

if [ ! -d "$HF_DIR" ]; then
  echo "HF 权重缺失，开始下载（ModelScope，官方确认与基线同款）..."
  python3 -c "from modelscope import snapshot_download; snapshot_download('Qwen/Qwen3.5-0.8B', local_dir='$HF_DIR')"
fi

if [ -d "$DCP_DIR" ]; then
  echo "DCP 权重已存在: $DCP_DIR（跳过转换）"
else
  echo "转换 HF → DCP ..."
  cd "$MSMM_DIR"
  python3 -m checkpoint.convert_cli GenericDCPConverter hf_to_dcp \
    --load_path "$HF_DIR" --save_path "$DCP_DIR"
fi

echo "完成: $DCP_DIR"
ls -la "$DCP_DIR" | head -5
