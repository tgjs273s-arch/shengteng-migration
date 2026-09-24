"""A/A 确定性诊断对照的注入件（**两侧一致开启**）。

来历与边界
----------
· 机制沿用坑 213 的做法：`sitecustomize.py` 在**解释器启动时**注入，
  这样 torchrun 主进程与每个 rank worker 都会带上补丁。
· ★ **只注入 `torch.use_deterministic_algorithms(True)`**，
  **不设** `HCCL_DETERMINISTIC` / `CLOSE_MATMUL_K_SHIFT` —— 坑 213 已把它们**单独证伪**
  （187/186 vs 对照 186，单位 /200 超阈，三者不可区分）。
· ★ **自证注入真的生效**（纪律 #3：否则"无效"可能是注入失败）：
  启动时打印 `DETSC active`，并写 per-rank marker 文件。
· ⛔ 本件**仅作诊断对照**（两侧都开），**不得**据此宣称"可交付"：
  PyTorch 文档明示该开关可能选择不同实现、且单独开启**不保证**整个应用可复现，NPU 上须实测。
"""
import os

_ok = False
try:
    import torch
    torch.use_deterministic_algorithms(True)
    _ok = True
except Exception as exc:            # noqa: BLE001
    print("DETSC FAILED %s: %s" % (type(exc).__name__, exc), flush=True)

if _ok:
    _rank = os.environ.get("RANK", "x")
    print("DETSC active pid=%s rank=%s" % (os.getpid(), _rank), flush=True)
    try:
        with open("/tmp/detsc62_active_rank_%s" % _rank, "w") as fh:
            fh.write("1")
    except OSError:
        pass
