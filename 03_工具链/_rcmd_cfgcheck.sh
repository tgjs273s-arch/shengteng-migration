set -u
CFG=/root/ops/qwen3_5_0_8B_mock_dp2_v3.yaml
python3 - "$CFG" <<'PY'
import yaml, io, sys
d = yaml.safe_load(io.open(sys.argv[1], encoding="utf-8"))
t = d.get("training", {})
for k in ("log_interval", "save_interval", "val_interval", "train_iters", "seed", "clip_grad",
          "use_deter_comp", "gradient_accumulation_steps", "micro_batch_size"):
    print("  %-32s %s" % (k, t.get(k, "(缺)")))
print("  data.val_dataset_param 存在=%s" % ("val_dataset_param" in d.get("data", {})))
PY
echo "CFGCHECK_DONE"
