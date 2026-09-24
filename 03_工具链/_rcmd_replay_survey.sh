#!/bin/bash
# 远端工作⑬：一次更新回放的**接口可行性勘查**（只读）
# 目标：找到"能固定状态跑一步"的最靠近的合法入口，以及"前向输出/梯度/更新"各自能否被观测
set -u
cd /root/MindSpeed-MM
echo "=== train_engine.py 顶层符号 ==="
grep -nE '^class |^    def |^        def ' mindspeed_mm/fsdp/train/train_engine.py | head -40
echo "=== train_engine.py 训练步关键语句 ==="
grep -nE 'backward\(\)|optimizer\.step|zero_grad|loss = |def train_step|micro_batch|all_reduce|reduce_scatter|clip_grad' mindspeed_mm/fsdp/train/train_engine.py | head -35
echo "=== trainer.py 里与训练/引擎相关的行 ==="
grep -nE 'def train|TrainEngine|self\.engine|self\.train_dataloader|def get_model' mindspeed_mm/fsdp/train/trainer.py | head -25
echo "=== optimizer 包暴露了什么（梯度裁剪/更新） ==="
grep -nE '^class |^def ' mindspeed_mm/fsdp/optimizer/optimizer.py mindspeed_mm/fsdp/optimizer/clip_grad_norm.py | head -25
echo "=== 模型 forward 签名（能否拿到 logits） ==="
grep -rnE 'def forward' mindspeed_mm/fsdp/models/base_model.py | head -6
echo "PROBE5_DONE"
