#!/bin/bash
# 远端工作⑨：① 全窗口指纹基线（BATCHFP_MAX=100，覆盖 1..100 步）
#            ② 造"负向对照"数据：复制 mock 数据集并**改动一个真实文本/像素相关字段**（改的是数据本身）
#            ③ 派生配置（只改 dataset 路径，其余一字不动）
set -eu
CFG=/root/ops/qwen3_5_0_8B_mock_dp2_v3.yaml
TS=$(date +%Y%m%d_%H%M%S)
ROOT=/root/ops/fp51full_$TS
NEG=/root/ops/fp_neg
mkdir -p "$ROOT" "$NEG"

echo "FP51_FULL_ROOT=$ROOT"

# ---- ② 负向对照数据：改**真实数据**（对照 Codex 的要求：不要只改 seed/cutoff）----
python3 - <<'PY'
import json, io, os, hashlib
src = "/root/MindSpeed-MM/data/mocked_vl_data/mock_data_pic_num_4_textlen_512.json"
dst = "/root/ops/fp_neg/mock_data_perturbed.json"
d = json.load(io.open(src, encoding="utf-8"))
n = len(d) if isinstance(d, list) else len(d.get("data", []))
recs = d if isinstance(d, list) else d["data"]
# 找第一个含文本的样本，**只改一个字符**（这会让 input_ids/labels 变化 —— 是"改真实 token"）
target = recs[0]
before = json.dumps(target, ensure_ascii=False)[:160]
changed = False
def perturb(o):
    """深度优先找到第一个 str 字段，在末尾追加一个字符。"""
    global changed
    if isinstance(o, dict):
        for k, v in list(o.items()):
            if isinstance(v, str) and v.strip() and not v.startswith("/") and len(v) > 8:
                o[k] = v + "Z"          # ★ 追加一个字符 ⇒ 分词后 token 序列变化
                return True
            if perturb(v):
                return True
    elif isinstance(o, list):
        for x in o:
            if perturb(x):
                return True
    return False
changed = perturb(target)
after = json.dumps(target, ensure_ascii=False)[:160]
io.open(dst, "w", encoding="utf-8").write(json.dumps(d, ensure_ascii=False))
print("NEG_DATA src=%s" % src)
print("NEG_DATA dst=%s records=%d changed=%s" % (dst, n, changed))
print("NEG_DATA before=%r" % before[:110])
print("NEG_DATA after =%r" % after[:110])
for p in (src, dst):
    print("  sha256[%s]=%s" % (os.path.basename(p),
                               hashlib.sha256(io.open(p, "rb").read()).hexdigest()[:16]))
PY

# ---- ③ 派生配置：只改 dataset 路径 ----
python3 - <<'PY'
import yaml, io
src = "/root/ops/qwen3_5_0_8B_mock_dp2_v3.yaml"
dst = "/root/ops/fp_neg/cfg_perturbed.yaml"
d = yaml.safe_load(io.open(src, encoding="utf-8"))
d["data"]["dataset_param"]["basic_parameters"]["dataset"] = "/root/ops/fp_neg/mock_data_perturbed.json"
io.open(dst, "w", encoding="utf-8").write(yaml.safe_dump(d, allow_unicode=True, sort_keys=False))
print("NEG_CFG dst=%s dataset=%s" % (dst, d["data"]["dataset_param"]["basic_parameters"]["dataset"]))
print("NEG_CFG 其余键是否与基线一致=待比对（下一步 diff yaml）")
PY

# ---- ① 全窗口指纹基线（原数据，MAX=100）----
cat > "$ROOT/launch.sh" <<EOF
source /usr/local/Ascend/ascend-toolkit/set_env.sh 2>/dev/null
export LD_LIBRARY_PATH=/usr/local/Ascend/driver/lib64:/usr/local/Ascend/driver/lib64/common:/usr/local/Ascend/driver/lib64/driver:/usr/local/Ascend/ascend-toolkit/latest/lib64:\$LD_LIBRARY_PATH
export NON_MEGATRON=true
export TASK_QUEUE_ENABLE=2
export ASCEND_LAUNCH_BLOCKING=0
export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
export TRITON_CACHE_DIR=/root/triton_cache
export BATCHFP=1
export BATCHFP_MAX=100
export BATCHFP_LOG=$ROOT/fp.log
cd /root/MindSpeed-MM && PYTHONPATH=/root/MindSpeed-MM:\${PYTHONPATH:-} timeout 1200 torchrun \\
  --nproc_per_node 2 --nnodes 1 --node_rank 0 --master_addr localhost --master_port 6115 \\
  /root/qwen35-ascend-migrator/scripts/51_train_fp.py $CFG > $ROOT/train.log 2>&1
echo "train_rc=\$?" > $ROOT/rc.txt
EOF
nohup bash "$ROOT/launch.sh" > "$ROOT/launch.out" 2>&1 &
echo "LAUNCHED_FULL pid=$!"
echo "FP51_FULL_START_DONE"
