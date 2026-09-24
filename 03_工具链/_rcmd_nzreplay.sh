#!/bin/bash
# ③ 非零学习率回放：① 造含优化器状态的 dcp 检查点 → ② 从它起跑两次回放(A/A) → ③ 比对
# 全程后台，产物落 /root/ops/nzreplay_<ts>/
set -u
TS=$(date +%Y%m%d_%H%M%S)
ROOT=/root/ops/nzreplay_$TS
mkdir -p "$ROOT"
BASE=/root/ops/qwen3_5_0_8B_mock_dp2_v3.yaml
echo "NZREPLAY_ROOT=$ROOT"

echo "=== ① 构造两个配置 ==="
python3 - "$BASE" "$ROOT" <<'PY'
import sys, yaml
base, root = sys.argv[1], sys.argv[2]
d = yaml.safe_load(open(base, encoding="utf-8"))
tr = d["training"]
print("BASE: lr=%s warmup_ratio=%s iters=%s save_interval=%s no_save_optim=%s load=%s save_format=%s"
      % (tr.get("lr"), tr.get("lr_warmup_ratio"), tr.get("train_iters"),
         tr.get("save_interval"), tr.get("no_save_optim"), tr.get("load"), tr.get("save_format")))

# --- cfg1：跑到 step 30 并在那里落一个**含优化器+RNG**的 dcp 检查点 ---
c1 = yaml.safe_load(open(base, encoding="utf-8"))
c1["training"]["train_iters"] = 30
c1["training"]["save_interval"] = 30
c1["training"]["save"] = root + "/ckpt"
c1["training"]["no_save_optim"] = False     # ⇒ 框架会自动把 save_format 切到 dcp
c1["training"]["no_save_rng"] = False
c1["training"]["no_load_optim"] = True      # 载入的是 hf 权重 ⇒ 必须保持 True
c1["training"]["no_load_rng"] = True
yaml.safe_dump(c1, open(root + "/cfg1_make_ckpt.yaml", "w", encoding="utf-8"), sort_keys=False)

# --- cfg2：从 cfg1 落下的 dcp 检查点**恢复**，且载入优化器/RNG ---
c2 = yaml.safe_load(open(base, encoding="utf-8"))
c2["training"]["load"] = root + "/ckpt/iter_0000030"
c2["training"]["load_format"] = "dcp"       # ★ hf 不允许载入优化器状态（会 raise）
c2["training"]["no_load_optim"] = False
c2["training"]["no_load_rng"] = False
c2["training"]["no_save_optim"] = True
c2["training"]["no_save_rng"] = True
yaml.safe_dump(c2, open(root + "/cfg2_replay_from_ckpt.yaml", "w", encoding="utf-8"), sort_keys=False)
print("CFG1=%s/cfg1_make_ckpt.yaml" % root)
print("CFG2=%s/cfg2_replay_from_ckpt.yaml" % root)
PY
echo "NZREPLAY_CFG_DONE"

cat > "$ROOT/all.sh" <<EOF
set -u
ROOT="$ROOT"
source /usr/local/Ascend/ascend-toolkit/set_env.sh 2>/dev/null
export LD_LIBRARY_PATH=/usr/local/Ascend/driver/lib64:/usr/local/Ascend/driver/lib64/common:/usr/local/Ascend/driver/lib64/driver:/usr/local/Ascend/ascend-toolkit/latest/lib64:\$LD_LIBRARY_PATH
export NON_MEGATRON=true
export TASK_QUEUE_ENABLE=2
export ASCEND_LAUNCH_BLOCKING=0
export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
export TRITON_CACHE_DIR=/root/triton_cache

echo "### PHASE1 start \$(date +%H:%M:%S)"
cd /root/qwen35-ascend-migrator
python3 scripts/50_train.py --config "\$ROOT/cfg1_make_ckpt.yaml" \\
  --env /root/qwen35-ascend-migrator/out/probe/env.json \\
  --log "\$ROOT/phase1_train.log" --workdir /root/MindSpeed-MM \\
  --steps 30 --world-size 2 --timeout 900 --foreground > "\$ROOT/phase1.driver.log" 2>&1
echo "### PHASE1 rc=\$?"
ls -la "\$ROOT/ckpt" 2>/dev/null | head -5
find "\$ROOT/ckpt" -maxdepth 2 2>/dev/null | head -15

# 发现 dcp 检查点目录（若 iter_0000030 不存在就取最新一个）
CK=\$(ls -1d "\$ROOT"/ckpt/iter_* 2>/dev/null | tail -1)
echo "### CKDIR=\$CK"
if [ -z "\$CK" ]; then echo "### ABORT 没有检查点"; echo "NZREPLAY_ABORT" > "\$ROOT/all.done"; exit 1; fi
# 把 cfg2 的 load 指到真实目录
python3 - "\$ROOT/cfg2_replay_from_ckpt.yaml" "\$CK" <<'PY'
import sys, yaml
p, ck = sys.argv[1], sys.argv[2]
d = yaml.safe_load(open(p, encoding="utf-8"))
d["training"]["load"] = ck
yaml.safe_dump(d, open(p, "w", encoding="utf-8"), sort_keys=False)
print("cfg2.load ->", ck)
PY

mkdir -p "\$ROOT/a" "\$ROOT/b"
mk () {  # \$1=tag \$2=port \$3=out
cat > "\$ROOT/launch_\$1.sh" <<EOS
source /usr/local/Ascend/ascend-toolkit/set_env.sh 2>/dev/null
export LD_LIBRARY_PATH=/usr/local/Ascend/driver/lib64:/usr/local/Ascend/driver/lib64/common:/usr/local/Ascend/driver/lib64/driver:/usr/local/Ascend/ascend-toolkit/latest/lib64:\\\$LD_LIBRARY_PATH
export NON_MEGATRON=true
export TASK_QUEUE_ENABLE=2
export ASCEND_LAUNCH_BLOCKING=0
export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
export TRITON_CACHE_DIR=/root/triton_cache
export REPLAY_OUT=$3
cd /root/MindSpeed-MM && PYTHONPATH=/root/MindSpeed-MM:\\\${PYTHONPATH:-} timeout 900 torchrun \\
  --nproc_per_node 2 --nnodes 1 --node_rank 0 --master_addr localhost --master_port $2 \\
  /root/qwen35-ascend-migrator/scripts/52_replay.py $ROOT/cfg2_replay_from_ckpt.yaml > $3/run.log 2>&1
echo "rc=\\\$?" > $3/rc.txt
EOS
}
mk a 6130 "\$ROOT/a"
mk b 6131 "\$ROOT/b"
echo "### PHASE2 start \$(date +%H:%M:%S)"
bash "\$ROOT/launch_a.sh"
echo "### PHASE2a rc=\$?"
bash "\$ROOT/launch_b.sh"
echo "### PHASE2b rc=\$?"
echo "### PHASE3 diff"
cd /root/qwen35-ascend-migrator/scripts
python3 _replay_diff.py --a "\$ROOT/a" --b "\$ROOT/b" > "\$ROOT/diff.txt" 2>&1
echo "### PHASE3 rc=\$?"
echo "NZREPLAY_DONE" > "\$ROOT/all.done"
EOF

nohup bash "$ROOT/all.sh" > "$ROOT/all.out" 2>&1 &
echo "LAUNCHED_NZREPLAY pid=$!"
sleep 25
echo "=== 25 秒后 ==="
tail -n 8 "$ROOT/all.out" 2>/dev/null || echo "(还没有输出)"
echo "NZREPLAY_LAUNCH_DONE"
