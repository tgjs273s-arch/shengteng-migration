# -*- coding: utf-8 -*-
r"""_patch_replay_protocol.py —— 把"一次回放"的**接口勘查结果**并入已冻结的协议

为什么：协议 §prerequisite 要求"先查 `use_deter_comp` 到底门控了什么"，同时实现要落到**真实函数**上。
本轮已只读查证（不猜接口）：
  · `mindspeed_mm/fsdp/train/train_engine.py`：`class TrainEngine`；
    `train_step(train_dataloader_iter)` L130（内含 L153 `loss = output.loss / gas`、L168 `loss.backward()`、
    L183 `average_losses_across_data_parallel_group`）；`train()` L231（内含 L262-270 的
    `clip_grad_norm(...)` → `optimizer.step()` → `optimizer.zero_grad()`）；
    `get_batch` L99、`set_loss_func` L112、`evaluate` L197、`save` L390。
  · `trainer.py`：L94 `self.trainer = TrainEngine(...)`、L468 `def train()`。
  · `optimizer/clip_grad_norm.py`：L21 `clip_grad_norm`、L163 `_fsdp2_reduce_group`（本地梯度在此被合并，
    但**不对外暴露原始局部梯度**）。
⇒ 结论：C1/C3/C4 有**明确入口**；**C2（通信前局部梯度）在 FSDP2 下用户侧不可观测** ⇒ 按协议必须记 UNVERIFIED，
   **不得**用"通信后梯度相同"代替（协议 §boundaries 已写明）。
"""
import io
import json
import os
import shutil
import time

P = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                 "qwen35-ascend-migrator_整合版", "protocols", "one_step_replay_20260922.json")
KEY = "framework_survey_2026_09_22"

d = json.load(io.open(P, encoding="utf-8"))
if KEY in d:
    print("PATCH_SKIP 已并入过")
    raise SystemExit(0)
d[KEY] = {
    "note": "只读查证（未连远端执行训练）：把四个对比点落到**真实函数与行号**上，避免实现时猜接口。",
    "entry_points": {
        "engine_class": "mindspeed_mm/fsdp/train/train_engine.py:28 class TrainEngine",
        "single_step": "train_engine.py:130 def train_step(self, train_dataloader_iter) —— 内含 "
                       "L153 loss = output.loss / gradient_accumulation_steps、L168 loss.backward()、"
                       "L183 total_loss = average_losses_across_data_parallel_group([total_loss])",
        "optimizer_sequence": "train_engine.py:262-270 —— clip_grad_norm(model, max_norm=clip_grad, foreach=...) "
                              "→ optimizer.step() → optimizer.zero_grad()",
        "batch_and_loss": "train_engine.py:99 get_batch(data_iterator)、L112 set_loss_func(batch_data)",
        "trainer_wiring": "trainer.py:94 self.trainer = TrainEngine(...)；trainer.py:468 def train()",
        "grad_norm_internals": "optimizer/clip_grad_norm.py:21 clip_grad_norm、L163 _fsdp2_reduce_group "
                               "（本地梯度在此被合并但不暴露）"
    },
    "observability": {
        "C1": "可观测：train_step 内的 output.loss（以及 loss.backward 前的 loss 张量本体）",
        "C2": "★ **不可观测**（FSDP2 在 backward 内部 reduce-scatter）⇒ 记 UNVERIFIED，不作替代",
        "C3": "可观测：train_step 返回后各 param.grad（已是本地分片的归约结果）",
        "C4": "可观测：按 train() 的同一序列执行 clip_grad_norm → optimizer.step() 一次后的参数"
    },
    "implementation_plan": [
        "scripts/52_replay.py：与 50_train.py 同源的环境组装 + `_ReplayTrainer(Trainer)`，"
        "在构造完成后（trainer/model/optimizer 均已就绪）**只跑一步**：取一个 iter、调 train_step、"
        "落 C1/C3、再按 train() 的序列做一次更新落 C4，然后退出（不进入 100 步循环）。",
        "scripts/_replay_diff.py：两个 run 的摘要做逐张量比对（不同元素占比 + 最大相对差 + 非有限计数），"
        "并按协议 §criteria.decision_rule 输出 C1..C4 中**第一处不同**。",
        "必须打印自证行：`REPLAY_FIXED input_sha=… params_sha=… opt_state=… rng=… lr=… step=…`"
        "（固定状态六项的**实际值摘要**），否则『固定完整状态』这句话没有依据。",
        "C2 位置必须显式打印 `REPLAY_C2 UNVERIFIED（FSDP2 不可观测）`，不得留空、不得用 C3 顶替。"
    ],
    "recorded_at": time.strftime("%Y-%m-%d %H:%M:%S")
}
bak = P + ".bak_survey_%s" % time.strftime("%Y%m%d_%H%M%S")
shutil.copy2(P, bak)
json.dump(d, io.open(P, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
back = json.load(io.open(P, encoding="utf-8"))
assert KEY in back and back["status"] == d["status"] and "fixed_state" in back
print("PATCH_OK 已并入 %s（未改判据：status=%s）；备份=%s" % (KEY, back["status"], bak))
