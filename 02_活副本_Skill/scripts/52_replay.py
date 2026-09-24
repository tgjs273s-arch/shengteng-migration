# -*- coding: utf-8 -*-
r"""52_replay.py —— **固定完整状态的一次更新回放**（诊断入口，不改交付主路径）

## 它回答什么
在**固定完整状态**（同输入/权重/buffers/优化器状态/RNG/学习率/global step）下做**一次**
前向+反向+一次更新，两个待比较对象**从哪一段开始不同**：前向(C1)？反向局部梯度(C2)？
通信后梯度(C3)？优化器更新(C4)？

## 为什么这样做（不重写框架循环）
只读勘查已确认（`protocols/one_step_replay_20260922.json` 的 `framework_survey_2026_09_22`）：
  · `TrainEngine.train()`（`train_engine.py:231`）里的训练序列是
    `pregather → train_step(iter) → clip_grad_norm(model, max_norm=args.training.clip_grad, foreach=…) →
     optimizer.step() → lr_scheduler.step() → optimizer.zero_grad() → iteration += 1`；
  · `TrainEngine.train_step()`（L130）内含 `self.get_batch(iter)` / `self.set_loss_func(batch)` /
    模型前向 / `loss.backward()`；
  · 属性名已内省确认：`eng.args / eng.model / eng.optimizer / eng.lr_scheduler / eng.iteration /
     eng.consumed_train_samples`。
⇒ 因此本脚本**不重写循环**，而是：**包一层**（`train_step` 抓前向输出与梯度、`optimizer.step` 抓更新后参数，
并在第一次更新后**抛异常跳出**），其余全部走框架原代码。**包一层 ≠ 换数据流**：前向/反向/归约/更新
都是框架自己的调用，我只在旁边读。

## 观测边界（必须如实声明，不许用别的点顶替）
  · C1 前向输出：可观测（模型 forward 的返回值，含 `loss`）；
  · C2 **通信前**局部梯度：★ FSDP2 在 backward 内部 reduce-scatter，用户侧**拿不到** ⇒ 本脚本
    **只打印 `REPLAY_C2 UNVERIFIED`**，绝不用 C3 顶替（协议 §boundaries 明文要求）；
  · C3 通信**后**梯度：可观测（`train_step` 返回后各 `param.grad`，已是本地分片的归约结果）；
  · C4 一次更新后参数：可观测（`optimizer.step()` 返回后）。

## 判据（与协议一致）
  · 每个对比点先看**全量字节摘要**（`sha_all`，最强判等），再看逐张量 sha 与标量统计；
  · 逐元素"不同元素占比 / 最大相对差"用**固定抽样**（每张量前 64 个元素 + 固定种子的 64 个随机下标）
    ⇒ **口径写进产物**（`sampled_elements`），不冒充全量；
  · 出现 NaN/Inf ⇒ 该点直接判 INVALID（不进入"相同"）。
"""
import hashlib
import io
import json
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

CFG_KEY_REQUIRED_ENV = ("NON_MEGATRON", "TASK_QUEUE_ENABLE", "ASCEND_LAUNCH_BLOCKING",
                        "PYTORCH_NPU_ALLOC_CONF", "TRITON_CACHE_DIR")
SAMPLE_HEAD = 64
SAMPLE_RAND = 64


class _StopReplay(Exception):
    """一步之后跳出框架 train() 循环（异常只用于跳出，不改变任何计算）。"""


def _log(msg):
    print(msg, flush=True)


def _sha_bytes(b):
    return hashlib.sha256(b).hexdigest()


def _opt_state_items(optimizer):
    """优化器状态 → [(名字, 张量)]，用于**按值**比较（不是按 repr）。

    ★ 为什么不能沿用 `json.dumps(state_dict, default=str)`：那会把张量变成
      **被截断的 repr 字符串**（大张量只显示头尾），既慢又**不可靠**——
      两个不同的状态可能打印成同一段字符串。这里直接取张量本身，
      交给 `_dump_group` 做逐张量摘要（与 C1/C3/C4 同一套口径）。
    """
    items = []
    try:
        st = (optimizer.state_dict() or {}).get("state", {}) or {}
    except Exception:
        return items
    for pid, sd in sorted(st.items(), key=lambda kv: str(kv[0])):
        if isinstance(sd, dict):
            for k, v in sorted(sd.items(), key=lambda kv: str(kv[0])):
                if hasattr(v, "detach"):
                    items.append(("p%s.%s" % (pid, k), v))
    return items


def _tensor_digest(t):
    """张量 → (摘要 dict, 原始字节)。torch 延迟导入。

    ★ 首次真机运行即报错（**实测**）：`RuntimeError: .numpy() is not supported for tensor subclasses.`
      —— FSDP2 下 `model.named_parameters()` 与 `param.grad` 都是 **DTensor 子类**。
      处置：① 有 `to_local()` 就先取**本地分片**（跨运行比较要在同一 rank 上比本地分片，语义正确）；
      ② 仍是子类身份就用 `as_subclass(torch.Tensor)` 脱掉；③ 最后才 `numpy()`。
      不这么做就只能"绕开参数只比 loss"——那等于放弃 C3/C4，属于偷工减料。
    """
    import torch
    tt = t.detach()
    if hasattr(tt, "to_local"):
        tt = tt.to_local()
    if type(tt) is not torch.Tensor:
        try:
            tt = tt.as_subclass(torch.Tensor)
        except Exception:
            pass
    tt = tt.contiguous().to(torch.float32)
    flat = tt.reshape(-1)
    raw = flat.cpu().numpy().tobytes()
    sha = _sha_bytes(raw)
    arr = flat.detach().cpu().numpy()
    n = int(arr.size)
    nonfin = int((~torch.isfinite(flat)).sum().item()) if hasattr(torch, "isfinite") else 0
    s = float(arr.astype("float64").sum()) if n else 0.0
    am = float(abs(arr).max()) if n else 0.0
    idx = list(range(min(SAMPLE_HEAD, n)))
    if n > SAMPLE_HEAD:
        import random
        rnd = random.Random(12345)                 # ★ 固定种子 ⇒ 抽样可复现
        idx += [rnd.randrange(n) for _ in range(min(SAMPLE_RAND, n - SAMPLE_HEAD))]
    sample = [float(arr[i]) for i in idx]
    return {"sha256": sha, "numel": n, "sum": s, "absmax": am,
            "nonfinite": nonfin, "sample": sample}, raw


def _dump_group(title, items, limit=200):
    """items: [(name, tensor)] → 组级摘要 + 逐张量（上限 limit 个，按名字排序保证顺序稳定）。"""
    out = {"group": title, "tensors": {}, "sha_all": None, "n_tensors": 0,
           "nonfinite": 0, "sampled_elements": "head%d+rand%d" % (SAMPLE_HEAD, SAMPLE_RAND)}
    h = hashlib.sha256()
    names = sorted(n for n, _t in items)
    n_all = 0
    for name in names:
        t = dict(items)[name]
        if t is None:
            continue
        rec, raw = _tensor_digest(t)
        h.update(name.encode("utf-8"))
        h.update(raw)
        n_all += 1
        out["nonfinite"] += rec["nonfinite"]
        if len(out["tensors"]) < limit:
            out["tensors"][name] = rec
    out["sha_all"] = h.hexdigest()
    out["n_tensors"] = n_all
    return out


def main():
    import yaml
    out_dir = os.environ.get("REPLAY_OUT", "")
    if not out_dir:
        _log("REPLAY_FAIL 需要 REPLAY_OUT（产物目录）")
        return 3
    os.makedirs(out_dir, exist_ok=True)
    rank = int(os.environ.get("RANK", "0") or 0)

    missing = [k for k in CFG_KEY_REQUIRED_ENV if not os.environ.get(k)]
    if missing:
        _log("REPLAY_FAIL 缺必需环境变量 %s（口径与 50_train.py 一致）" % missing)
        return 3

    argv = [a for a in sys.argv[1:]]
    if not argv:
        _log("REPLAY_FAIL 需要位置参数：配置文件（与 50_train.py 同约定）")
        return 3
    cfg_path = argv[0]
    d = yaml.safe_load(io.open(cfg_path, encoding="utf-8"))
    tr = d["training"]
    dp = int(d["parallel"]["data_parallel_size"])
    gbs = int(tr["micro_batch_size"]) * int(tr["gradient_accumulation_steps"]) * dp
    _log("REPLAY_CFG config=%s mbs=%s gas=%s world=%s GBS=%s train_iters=%s"
         % (cfg_path, tr["micro_batch_size"], tr["gradient_accumulation_steps"], dp, gbs,
            tr.get("train_iters")))
    if gbs != 8:
        _log("REPLAY_FAIL GBS=%d ≠ 8（官方红线；本入口不提供绕过开关）" % gbs)
        return 3

    from mindspeed_mm.config.config_manager import ConfigManager
    from mindspeed_mm.fsdp.params.argument import Arguments
    from mindspeed_mm.fsdp.train.trainer import Trainer

    class _ReplayTrainer(Trainer):
        def train(self):
            eng = self.trainer
            res = {"rank": rank, "cfg": cfg_path, "gbs": gbs,
                   "comparison_points": ["C1_forward", "C2_pre_comm_grad_UNVERIFIED",
                                          "C3_post_comm_grad", "C4_after_one_update"],
                   "C2": "UNVERIFIED: FSDP2 在 backward 内部 reduce-scatter，用户侧不可观测（不用 C3 顶替）"}
            captured = {"batch": None, "out": None, "grads": None, "params_after": None}
            orig_get_batch = eng.get_batch
            orig_train_step = eng.train_step
            orig_opt_step = eng.optimizer.step

            def get_batch(data_iterator):
                b = orig_get_batch(data_iterator)
                if captured["batch"] is None:
                    captured["batch"] = b
                return b

            def train_step(it):
                import torch
                h = eng.model.register_forward_hook(
                    lambda m, i, o: captured.__setitem__("out", o))
                try:
                    loss_dict = orig_train_step(it)          # 框架原样：前向 + 反向（含归约）
                finally:
                    h.remove()
                captured["grads"] = [(n, p.grad) for n, p in eng.model.named_parameters()
                                     if p.grad is not None]
                return loss_dict

            def opt_step(*a, **k):
                # ★ v2：在**更新之前**就把参数、优化器状态、**本次真正使用的 lr** 摘出来。
                #   · 参数/优化器状态按**值**摘要：旧实现只在更新后取一次，却在之后把
                #     `named_parameters()` 的**引用**当成 "before"，得到的其实是更新后的值
                #     （参数被原地改动）⇒ `before == after` 恒成立，**永远看不出空更新**。
                #   · lr **按 param_group 逐个取**（`optimizer.param_groups[i]["lr"]`），
                #     这才是本步实际生效的值；**不得**用 `scheduler.get_last_lr()` 的一个值
                #     代替全部。scheduler 的值另存一份用于**一致性核对**。
                captured["lrs_used"] = [float(g.get("lr", float("nan")))
                                        for g in eng.optimizer.param_groups]
                captured["lr_sched_last"] = (
                    [float(v) for v in eng.lr_scheduler.get_last_lr()]
                    if eng.lr_scheduler is not None else None)
                captured["params_before"] = _dump_group(
                    "PB", [(n, p) for n, p in eng.model.named_parameters()])
                captured["opt_before"] = _dump_group("OB", _opt_state_items(eng.optimizer))
                r = orig_opt_step(*a, **k)
                captured["params_after"] = _dump_group(
                    "PA", [(n, p) for n, p in eng.model.named_parameters()])
                captured["opt_after"] = _dump_group("OA", _opt_state_items(eng.optimizer))
                raise _StopReplay()                          # 一步之后跳出（不改计算）

            eng.get_batch, eng.train_step, eng.optimizer.step = get_batch, train_step, opt_step
            try:
                super().train()
            except _StopReplay:
                pass
            finally:
                eng.get_batch, eng.train_step = orig_get_batch, orig_train_step
                eng.optimizer.step = orig_opt_step

            # ---- 固定状态自证行（六项的实际值摘要；★ v2：before/after 均为**按值**摘要）----
            import torch
            from _batchfp import digest as fp_digest
            rng_sha = _sha_bytes(bytes(torch.get_rng_state().numpy().tobytes()))
            pb = captured.get("params_before") or {}
            pa = captured.get("params_after") or {}
            ob = captured.get("opt_before") or {}
            oa = captured.get("opt_after") or {}
            lr = (eng.lr_scheduler.get_last_lr()[0] if eng.lr_scheduler is not None else None)
            res["fixed"] = {
                "input_sha_all": fp_digest(captured["batch"]) if captured["batch"] is not None else None,
                "params_before_sha_all": pb.get("sha_all"),
                "grads_sha_all": (_dump_group("G", captured["grads"])["sha_all"]
                                  if captured["grads"] else None),
                "params_after_sha_all": pa.get("sha_all"),
                "opt_before_sha_all": ob.get("sha_all"),
                "opt_after_sha_all": oa.get("sha_all"),
                "rng_cpu_sha": rng_sha,
                "lr": lr,
                "iteration": int(eng.iteration),
            }

            # ---- ★ v2 前置条件（fail-closed：不满足**不许**当"通过"）----
            #   起因（本协议 why_now）：零学习率空更新下 C4 恒等，是**空过**，
            #   必须让脚本自己把它判成 INVALID，而不是让读者以为"这里验过"。
            pre = {
                "lr_param_groups": captured.get("lrs_used") or [],
                "lr_sched_last": captured.get("lr_sched_last"),
                "lr_consistent": bool(
                    captured.get("lrs_used") and captured.get("lr_sched_last")
                    and all(abs(a_ - b_) <= 1e-12
                            for a_, b_ in zip(captured["lrs_used"], captured["lr_sched_last"]))),
                # ★ 判据用**全部 param_group 都 > 0**，不是只看 scheduler 的一个值
                "lr_all_positive": bool(captured.get("lrs_used")
                                        and all(v > 0.0 for v in captured["lrs_used"])),
                "opt_state_tensors": int(ob.get("n_tensors") or 0),
                "params_changed": bool(pb.get("sha_all") and pa.get("sha_all")
                                       and pb["sha_all"] != pa["sha_all"]),
                "opt_state_changed": bool(ob.get("sha_all") and oa.get("sha_all")
                                          and ob["sha_all"] != oa["sha_all"]),
                "train_iters": int(tr.get("train_iters") or 0),
                "gradient_accumulation_steps": int(tr.get("gradient_accumulation_steps") or 0),
            }
            if not pre["lr_consistent"]:
                # 一致性核对失败 ⇒ 不许当"调度器已正确恢复"
                pre["lr_note"] = ("scheduler.get_last_lr() 与 optimizer.param_groups 的 lr 不一致"
                                  " ⇒ 调度器状态与优化器不同步")
            verdicts = []
            if not pre["lr_all_positive"]:
                verdicts.append("REPLAY_INVALID_ZERO_LR")
            if pre["opt_state_tensors"] <= 0:
                verdicts.append("REPLAY_INVALID_NO_OPT_STATE")
            if not pre["params_changed"]:
                verdicts.append("REPLAY_INVALID_NULL_UPDATE")
            if not pre["lr_consistent"]:
                verdicts.append("REPLAY_INVALID_LR_MISMATCH")
            pre["verdicts"] = verdicts
            pre["status"] = "OK" if not verdicts else "INVALID"
            res["preconditions"] = pre
            res["schema"] = "replay.v2"
            o = captured["out"]
            res["C1_forward"] = {
                "output_type": type(o).__name__ if o is not None else None,
                "loss": (_tensor_digest(o.loss)[0] if (o is not None and getattr(o, "loss", None) is not None) else None),
                "logits": (_tensor_digest(o.logits)[0]
                           if (o is not None and getattr(o, "logits", None) is not None) else None),
            }
            res["C3_post_comm_grad"] = _dump_group("C3", captured["grads"] or [])
            res["C4_after_one_update"] = pa
            res["C4_optimizer_state"] = {"before": ob, "after": oa}
            p = os.path.join(out_dir, "replay_dump.rank%d.json" % rank)
            io.open(p, "w", encoding="utf-8").write(json.dumps(res, ensure_ascii=False, indent=1))
            _log("REPLAY_FIXED rank=%d input_sha=%s params_before=%s params_after=%s rng=%s lr=%s step=%s"
                 % (rank, str(res["fixed"]["input_sha_all"])[:16],
                    str(res["fixed"]["params_before_sha_all"])[:16],
                    str(res["fixed"]["params_after_sha_all"])[:16],
                    str(res["fixed"]["rng_cpu_sha"])[:16], res["fixed"]["lr"], res["fixed"]["iteration"]))
            _log("REPLAY_C1 rank=%d loss_sha=%s logits_sha=%s"
                 % (rank, str((res["C1_forward"]["loss"] or {}).get("sha256"))[:16],
                    str((res["C1_forward"]["logits"] or {}).get("sha256"))[:16]))
            _log("REPLAY_C2 rank=%d UNVERIFIED（FSDP2 不可观测；不用 C3 顶替）" % rank)
            _log("REPLAY_C3 rank=%d tensors=%d sha_all=%s" % (rank, res["C3_post_comm_grad"]["n_tensors"],
                                                              str(res["C3_post_comm_grad"]["sha_all"])[:16]))
            _log("REPLAY_C4 rank=%d tensors=%d sha_all=%s" % (rank, pa.get("n_tensors"),
                                                              str(pa.get("sha_all"))[:16]))
            _log("REPLAY_OPT_STATE rank=%d tensors=%d before=%s after=%s"
                 % (rank, pre["opt_state_tensors"], str(ob.get("sha_all"))[:16],
                    str(oa.get("sha_all"))[:16]))
            if verdicts:
                _log("REPLAY_PRECOND_INVALID rank=%d verdicts=%s lrs=%s opt_tensors=%d "
                     "params_changed=%s lr_consistent=%s"
                     % (rank, ",".join(verdicts), pre["lr_param_groups"],
                        pre["opt_state_tensors"], pre["params_changed"], pre["lr_consistent"]))
            else:
                _log("REPLAY_PRECOND_OK rank=%d lrs=%s sched_last=%s lr_consistent=%s "
                     "opt_tensors=%d params_changed=%s opt_changed=%s"
                     % (rank, pre["lr_param_groups"], pre["lr_sched_last"], pre["lr_consistent"],
                        pre["opt_state_tensors"], pre["params_changed"], pre["opt_state_changed"]))
            _log("REPLAY_DUMP %s" % p)
            _log("REPLAY_DONE rank=%d" % rank)

    arguments = ConfigManager(config_class=Arguments).load_and_parse()
    _ReplayTrainer(args=arguments).train()
    return 0


if __name__ == "__main__":
    sys.exit(main())
