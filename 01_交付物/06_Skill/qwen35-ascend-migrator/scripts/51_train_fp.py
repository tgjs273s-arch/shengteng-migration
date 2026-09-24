#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
51_train_fp.py — 逐步 batch 指纹的**诊断专用入口**（不改交付主路径）

=== 这是什么 ===
它是 `mindspeed_mm/fsdp/train/trainer.py` 的 `__main__`（L473-477）的**诊断替身**：
同一套 env 组装、同一个位置参数约定、同一套 fail-closed 检查，唯一差别是
把 `Trainer` 换成只覆写了 `get_dataloader` 的 `_FPTrainer`（设计文档 §3bis 首选方案），
从而在最外层多套一个**只读**指纹代理（`scripts/_batchfp.py`）。

=== ★ 纪律（必须连同结果一起声明）===
  1. 本入口是**诊断入口**：它的产物**不得**用作性能收益证据。
     记录本身（逐字段遍历 + sha256）会改变时序，而"最外层多套一个对象"也改变了数据流对象身份。
     正式计时 / 性能收益那次运行必须走 `50_train.py` 且 **BATCHFP 全关**（默认即关）。
  2. 用本入口跑出来的**任何**运行，`plan_consistent` **必须**声明为 **false** ——
     本文件会主动打印那一行 `FP_PLAN_CONSISTENT=false`（不许由使用者"记得"去写）。
  3. 指纹只覆盖"进入模型之前"的 batch；模型内部算子/通信不在覆盖内（见 `_batchfp.py` 文件头）。

=== 启动方式（与 50_train.py **完全同一套**环境/参数组装，只换入口文件）===
  50_train.py 拼出的训练命令（L560-565）是：
      cd <MSMM> && PYTHONPATH=<MSMM>:${PYTHONPATH} timeout <T> torchrun \
          --nproc_per_node <world> --nnodes 1 --node_rank 0 \
          --master_addr localhost --master_port <port> \
          mindspeed_mm/fsdp/train/trainer.py <config.yaml> > <log> 2>&1
  本入口**沿用同一个位置参数约定**（yaml 作为位置参数；`load_and_parse()` 自己去解析 argv），
  只是把入口文件换成 `python3 <SKILL>/scripts/51_train_fp.py`，并在前面加上指纹开关：
      BATCHFP=1 BATCHFP_MAX=3 BATCHFP_LOG=out/train/fp.log
  （环境变量前缀仍由 50_train.py 的 `build_env_string()` / 等价 shell 前缀提供；
    本脚本**不**发明新的启动约定，也不改 50_train.py。）

=== 本文件的 `# 待查证：` 条目 ===
  见文件末尾 `_provider_deadend()` —— provider（`dataloader_provider`）路径需要**逐行复刻**
  `get_dataloader()`（trainer.py:339-400，100+ 行），而该正文本机读不到、且不允许连远端，
  故该分支 **fail-closed**（明确报错 + 说明），绝不编一个"看起来能跑"的 provider。
"""

import os
import re
import sys

# ★ 与 50_train.py L49-50 同一做法：把本脚本目录放进 sys.path，才能 import 同目录的 _batchfp。
_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

import _batchfp  # noqa: E402  （同目录模块，必须在 sys.path 注入之后 import）

# ---------------------------------------------------------------- 环境变量自检
# ★ 照抄 50_train.py 的 REQUIRED_ENV（同名 / 同值 / 同诊断文案）：不新增约定、不改口径。
#   为什么在这里**再查一遍**：50_train.py 是在**启动侧**注入这些变量；若有人手工起了一次
#   torchrun（或改了启动脚本），注入就可能整批丢失，而症状会在几分钟后才以别的面目出现
#   （最典型：缺 NON_MEGATRON → ModuleNotFoundError: No module named 'megatron'）。
#   本入口选择 **fail-closed**：启动前就大声拒绝，而不是等到框架深处。
REQUIRED_ENV = [
    ("NON_MEGATRON", "true",
     "缺 → 框架误走 Megatron 路径报 ModuleNotFoundError: No module named 'megatron'"),
    ("TASK_QUEUE_ENABLE", "2",
     "实测 =1 会导致慢步回归"),
    ("ASCEND_LAUNCH_BLOCKING", "0", ""),
    ("PYTORCH_NPU_ALLOC_CONF", "expandable_segments:True", "显存碎片整理"),
    ("TRITON_CACHE_DIR", "/root/triton_cache",
     "官方脚本每次清空；我方持久化以规避重编译（报告需双口径披露）"),
]

# 本入口**自己**的参数。它们必须在调用框架的 load_and_parse() **之前**从 argv 里摘掉。
# 为什么要摘：load_and_parse() 由框架实现（trainer.py:474），我们不掌握它对未知参数的容忍度 ——
# 把自家参数混进它的 argv 属于"猜接口"。摘掉自家参数、其余**一字不动**留给框架，才是安全的。
OWN_FLAGS = ("--use-provider", "--fp-off", "--fp-max", "--fp-log", "--allow-gbs-mismatch")


def log(msg):
    print(msg, flush=True)


# ---------------------------------------------------------------- argv / 开关
def _pop_own_flags(argv):
    """只摘掉 OWN_FLAGS 里的参数，其余**原样**返回给框架。

    返回 (opts, cleaned)。opts 键：use_provider / fp_off / fp_max / fp_log / allow_gbs_mismatch。
    ★ 任何不属于 OWN_FLAGS 的参数一律不碰 —— 框架的报错由框架自己发（fail loudly）。
    """
    opts = {"use_provider": False, "fp_off": False, "fp_max": None,
            "fp_log": None, "allow_gbs_mismatch": False}
    cleaned = []
    i = 0
    n = len(argv)
    while i < n:
        a = argv[i]
        key, val = a, None
        if a.startswith("--") and "=" in a:
            key, val = a.split("=", 1)
        if key == "--use-provider":
            opts["use_provider"] = True
        elif key == "--fp-off":
            opts["fp_off"] = True
        elif key == "--allow-gbs-mismatch":
            opts["allow_gbs_mismatch"] = True
        elif key in ("--fp-max", "--fp-log"):
            if val is None:
                if i + 1 >= n:
                    log("FATAL %s 需要一个值" % key)
                    sys.exit(2)
                i += 1
                val = argv[i]
            if key == "--fp-max":
                try:
                    opts["fp_max"] = max(0, int(val))
                except Exception:
                    log("FATAL --fp-max 需要整数，收到 %r" % val)
                    sys.exit(2)
            else:
                opts["fp_log"] = val
        else:
            cleaned.append(a)
            i += 1
            continue
        i += 1
    return opts, cleaned


def _apply_fp_env(opts):
    """把开关落到环境变量上（`_batchfp` 从环境变量解析），并把**最终生效值**打印出来。

    为什么默认打开：本入口的**唯一用途**就是取证，不打开等于白跑。
    这与 `_batchfp` 的"库缺省关闭"不矛盾 —— 库负责能被全关，入口负责别忘开。
    """
    defaulted = []
    if opts["fp_off"]:
        os.environ["BATCHFP"] = "0"
        log("FP_SWITCH BATCHFP=0（--fp-off：本次运行**不记**指纹；此时它就不是取证运行）")
    elif not os.environ.get("BATCHFP"):
        os.environ["BATCHFP"] = "1"
        defaulted.append("BATCHFP=1")
    if opts["fp_max"] is not None:
        os.environ["BATCHFP_MAX"] = str(opts["fp_max"])
    elif not os.environ.get("BATCHFP_MAX"):
        _d = _batchfp.default_max_micro_batches()
        os.environ["BATCHFP_MAX"] = str(_d)
        defaulted.append("BATCHFP_MAX=%d" % _d)
    if opts["fp_log"]:
        os.environ["BATCHFP_LOG"] = opts["fp_log"]
    if defaulted:
        log("FP_SWITCH 未显式设置，已默认打开：%s（可用 --fp-off / BATCHFP=0 关闭）"
            % " ".join(defaulted))
    log("FP_SWITCH 生效值 BATCHFP=%s BATCHFP_MAX=%s BATCHFP_LOG=%s"
        % (os.environ.get("BATCHFP"), os.environ.get("BATCHFP_MAX"),
           os.environ.get("BATCHFP_LOG") or "none"))


# ---------------------------------------------------------------- 环境 fail-closed
def check_required_env():
    """必需环境变量自检。返回 0 = 全部就位；3 = 拒绝执行（与 50_train.py 的退出码 3 同义）。"""
    bad = []
    for name, want, why in REQUIRED_ENV:
        got = os.environ.get(name)
        if got != want:
            bad.append((name, want, got, why))
    if not bad:
        log("FP_ENV_CHECK OK（%d 项必需环境变量全部就位，口径与 50_train.py 一致）"
            % len(REQUIRED_ENV))
        return 0
    log("")
    log("!! 必需环境变量缺失/不一致 —— **在构建任何一个 Trainer 之前拒绝执行**（fail-closed）")
    for name, want, got, why in bad:
        log("   · %s: 期望 %r，实际 %r%s" % (name, want, got, ("　← " + why) if why else ""))
    log("   原因：这些变量由 50_train.py 的 REQUIRED_ENV / build_env_string() 注入；")
    log("         手工起 torchrun 时最容易整批丢掉，症状却会在几分钟后以别的面目出现。")
    log("   处置：用 50_train.py 同一套环境组装来启动本入口（见本文件头部的启动方式），")
    log("         即 `source set_env.sh; export NON_MEGATRON=true TASK_QUEUE_ENABLE=2 ...`")
    return 3


# ---------------------------------------------------------------- 几何自检（复用 50_train.py 做法）
def _as_int(v):
    if v is None or isinstance(v, bool):
        return None
    try:
        return int(v)
    except Exception:
        return None


def _resolve_config_path(argv):
    """**只读地**从命令行里找配置文件路径（不摘、不改，仍然留给框架去解析）。

    依据：50_train.py L560-565 就是把 yaml 作为**位置参数**传给 trainer.py 的
    （`... trainer.py %s`）。本入口沿用同一位置参数约定 ⇒ 找"第一个非选项、且像 yaml / 存在的文件"。
    找不到就返回 None —— 调用方必须标注"未检查"，**不得**当作通过。
    """
    for a in argv:
        if a.startswith("-"):
            continue
        if a.lower().endswith((".yaml", ".yml")) or os.path.isfile(a):
            return a
    return None


def geometry_selfcheck(cfg_path, allow_mismatch):
    """★ 复用 50_train.py（L419-500）的做法与口径：先解析 YAML（解析不了就拒启动），
    再算 GBS = mbs × gas × world，与之不等则拒绝执行（除非显式 --allow-gbs-mismatch）。

    world 取自 torchrun 注入的 `WORLD_SIZE`（环境事实，不是猜框架接口）。
    返回 (rc, gas)。rc==3 表示拒绝执行；(rc, gas) 里 gas 供 step_hint 推导使用。
    """
    gas_for_hint = None
    if cfg_path is None:
        # ★ 明确标注"未检查"，而不是打印一个假的"通过"。
        log("FP_GBS_CHECK skipped reason=config_path_unresolved —— 未检查，**不得**当作通过")
        return 0, None

    log("FP_GBS_CHECK config=%s" % os.path.abspath(cfg_path))
    try:
        txt = open(cfg_path, encoding="utf-8", errors="replace").read()
    except Exception as e:
        log("!! 配置读不了：%s（%s）—— 拒绝执行" % (cfg_path, e))
        return 3, None

    try:
        import yaml as _yaml
    except ImportError:
        _yaml = None

    doc, parse_err = None, None
    if _yaml is not None:
        try:
            doc = _yaml.safe_load(txt)
            if not isinstance(doc, dict):
                parse_err = "顶层不是映射（got %s）" % type(doc).__name__
        except Exception as e:
            parse_err = str(e)
    if parse_err:
        # ★ 坑 111 同款纪律：运行条件读原始文本、验收条件读解析结果，两者不同源；
        #   这里坚持"解析不了就不许开跑"，绝不用正则扫出来的数字假装检查通过。
        log("!! 配置 YAML 无法解析 —— 在构建 Trainer 之前拒绝执行（同 50_train.py 的做法）")
        log("   原因: %s" % parse_err)
        log("   处置: 用 P2 重新生成配置：python3 scripts/20_plan_migration.py ...")
        return 3, None

    if doc is not None:
        tr = doc.get("training") or {}
        mbs = _as_int(tr.get("micro_batch_size"))
        gas = _as_int(tr.get("gradient_accumulation_steps"))
        src = "yaml"
    else:
        # 降级路径：无 pyyaml 时只能行扫描，**明确标注为降级**，不假装与解析等价。
        def _g(k):
            m = re.search(r"(?m)^\s*%s\s*:\s*(\S+)" % k, txt)
            if not m:
                return None
            return _as_int(str(m.group(1)).strip("\"'"))
        mbs, gas, src = _g("micro_batch_size"), _g("gradient_accumulation_steps"), "regex_fallback(no_pyyaml)"
        log("⚠ 未安装 pyyaml → GBS 自检降级为正则行扫描（不等价于解析校验）")

    world = _as_int(os.environ.get("WORLD_SIZE"))
    if not mbs or not gas or not world or world < 1:
        # ★ 三个量任何一个拿不到就不算 GBS，如实标注"未检查"。
        log("FP_GBS_CHECK skipped reason=missing_geometry mbs=%s gas=%s WORLD_SIZE=%s world=%s "
            "src=%s —— 未检查，**不得**当作通过"
            % (mbs, gas, os.environ.get("WORLD_SIZE"), world, src))
        return 0, gas

    gbs = mbs * gas * world
    log("FP_GEOMETRY mbs=%s × gas=%s × world=%s = GBS %s（官方红线 = 8；取值来源 %s）"
        % (mbs, gas, world, gbs, src))
    if gbs != 8:
        if allow_mismatch:
            log("⚠ 已显式允许 GBS≠8（--allow-gbs-mismatch，与 50_train.py 同名同义）："
                "结果**不可用于官方精度对标**")
        else:
            log("")
            log("!! GBS ≠ 8 —— 违反项目硬红线，**拒绝执行**（与 50_train.py 同一判据）")
            log("   官方验收要求 GBS = mbs × gas × dp = 8；当前为 %s。" % gbs)
            log("   处置：修配置（P2 重新生成）或显式加 --allow-gbs-mismatch（仅流程连通性验证）。")
            return 3, gas
    return 0, gas


# ---------------------------------------------------------------- 框架导入（照抄设计文档 §4 的钉死路径）
# ★ 这三条 import 的模块位置**已由只读查证钉死**，照抄，勿改：
#     mindspeed_mm/config/config_manager.py:13  → ConfigManager
#     mindspeed_mm/fsdp/params/argument.py:20   → Arguments
#     mindspeed_mm/fsdp/train/trainer.py        → Trainer
from mindspeed_mm.config.config_manager import ConfigManager          # noqa: E402
from mindspeed_mm.fsdp.params.argument import Arguments               # noqa: E402
from mindspeed_mm.fsdp.train.trainer import Trainer                   # noqa: E402


def _peek_gas_from_args(args):
    """读 `args.training.gradient_accumulation_steps` —— **键名来自设计文档 §3 表（已查证）**。

    取不到就返回 (None, "unreadable")，**不猜别的键名**（例如不试
    `args.training.micro_batch_size`：那个键名不在已查证清单里，见下面的待查证标注）。
    返回值只用于**推导** step_hint（日志标注），不参与数据通路 —— 所以取不到不影响正确性，
    只影响 step_hint 的精度，且会在日志里如实写成 gas1 假定。
    """
    # 待查证：args.training.micro_batch_size 是否存在于 Arguments（键名未在设计文档里查证过）。
    #          正因未查证，本文件**不用它**：几何自检改为读配置 yaml 的同名键
    #          （50_train.py 读的就是 yaml 的 training.micro_batch_size，同源、无需猜 API）。
    try:
        return int(args.training.gradient_accumulation_steps), "args.training.gradient_accumulation_steps"
    except Exception:
        return None, "unreadable"


# ---------------------------------------------------------------- 首选方案：子类覆写 get_dataloader
class _FPTrainer(Trainer):
    """只覆写 `get_dataloader`（设计文档 §3bis 的**首选方案**）。

    为什么它优于 `dataloader_provider`：
      · `super().get_dataloader()` 调的是框架**同一份代码** ⇒ `PrefetchGradAccDataLoader` 的
        条件包装（仅当 `args.features.loss_cfg.loss_type == "per_token_loss"`）、train/val 拆分、
        val 的 shuffle/drop_last 覆盖**全部自动继承**，一处都不会漏；
      · provider 则要求我们**逐行复刻**那 100+ 行，漏一处就是"静默换数据流"（最难发现的失效模式）。
    本覆写只做一件事：在**最外层**套一个只读代理（转发一切，只在产出时记指纹）。
    """

    def get_dataloader(self):
        res = super().get_dataloader()   # ← 框架原封不动地构建（含上面全部后处理）
        # ★ res 可能是 (train_loader, val_loader) 二元组，也可能是单个 loader —— 两种都正确处理；
        #   val 一字不动（只包装 train，减小对验证语义的影响面）。
        wrapped = _batchfp.wrap_dataloader_result(res, tag="train")
        try:
            kind = "tuple(len=%d)" % len(res) if isinstance(res, (tuple, list)) else type(res).__name__
        except Exception:
            kind = "?"
        log("FP_WRAP get_dataloader_res=%s wrapped_outermost=%s plan_consistent=false"
            % (kind, type(wrapped).__name__))
        return wrapped


# ---------------------------------------------------------------- 降级实验路径：provider（fail-closed）
def _provider_deadend():
    """`--use-provider` 分支：**保留入口但 fail-closed**，不实现复刻体。

    §3 的陷阱（必须写在注释里，因为它会**静默**失真）：
      `dataloader_provider` 的返回值被**直接**当作 train_dataloader（trainer.py:79），
      **不经过** `get_dataloader()` 的后处理：
        · `PrefetchGradAccDataLoader` 的条件包装（仅当 loss_type == "per_token_loss"）被丢掉
          ⇒ 梯度累积的预取/切分语义变了 ⇒ **静默换了一条数据流**，A/B 不可比；
        · train/val 拆分被丢掉 ⇒ 可能把 val 当 train；
        · val 的 shuffle=False / drop_last=False / val_micro_batch_size 覆盖被丢掉。
      因此 provider 路径的运行 `plan_consistent` **必须**为 false，
      并且 provider 里**必须**打印一行自证（`FP_PROVIDER_MIRRORS ...`），否则"我只是加了个 logger"
      这句话没有依据。

    为什么这里**不写**它的实现：
      §3 要求 provider "逐行复刻 `get_dataloader()`"，而该正文（trainer.py:339-400，100+ 行）
      **本机读不到**（本机没有 MindSpeed-MM 源码，且本轮明确不允许连远端）。
      # 待查证：mindspeed_mm/fsdp/train/trainer.py:339-400 的完整正文 —— 具体要查：
      #         ① build_mm_dataset(...) / build_mm_dataloader(...) 的确切调用参数；
      #         ② train/val 拆分分支（返回值是 tuple 时取 [0]/[1]）的确切写法与边界；
      #         ③ val 的 shuffle=False / drop_last=False / val_micro_batch_size 覆盖的确切位置；
      #         ④ PrefetchGradAccDataLoader 包装的确切条件与参数名。
      # 在没有读到之前写"复刻体"= 编一个看起来能跑的实现 —— 本项目的纪律明令禁止。
      所以此分支明确报错并指向首选方案，而不是给出一个会静默失真的 provider。
    """
    log("")
    log("!! --use-provider 分支**不可用**（fail-closed，本机未读到 get_dataloader() 正文）")
    log("   本分支的陷阱（设计文档 §3）：provider 的返回值会被**直接**当作 train_dataloader，")
    log("   绕过 get_dataloader() 的**全部**后处理 ⇒ PrefetchGradAccDataLoader 条件包装被丢、")
    log("   train/val 拆分被丢、val 的 shuffle/drop_last/val_micro_batch_size 覆盖被丢")
    log("   ⇒ **静默换了一条数据流** ⇒ 该运行的 plan_consistent 必须为 false。")
    # ★ 自证行照 §3 要求打印，但**绝不**把未验证的项写成 True：
    #   trainer.py:339-400 的正文没有读到，所以这里只能是 unknown。
    log("FP_PROVIDER_MIRRORS get_dataloader=unknown loss_type=unknown prefetch_wrapped=unknown "
        "train_val_split=unknown val_overrides=unknown (未复刻：正文未读到 ⇒ 不得声明 True)")
    log("   待查证：trainer.py:339-400 的完整正文（build_mm_dataset/build_mm_dataloader 的调用参数、")
    log("           train/val 拆分、val 参数覆盖、PrefetchGradAccDataLoader 的确切包装条件）。")
    log("   处置：去掉 --use-provider，走**首选方案**（本文件默认路径：子类覆写 get_dataloader）。")
    return 2


def _make_provider(*_a, **_k):
    """第二道护栏：万一有人把 provider 接回 `Trainer(..., dataloader_provider=...)`，
    这里立刻**大声失败**，而不是返回一个会静默换数据流的对象。"""
    raise RuntimeError(
        "51_train_fp.py 未实现 dataloader_provider：需逐行复刻 get_dataloader()"
        "（trainer.py:339-400），该正文未读到/未查证 ⇒ fail-closed。请走 _FPTrainer 覆写路径。")


# ---------------------------------------------------------------- main
def main():
    argv = sys.argv[1:]
    opts, cleaned = _pop_own_flags(argv)
    log("== 51_train_fp.py 逐步 batch 指纹：**诊断入口** ==")
    log("FP_ENTRY entry=51_train_fp.py based_on=trainer.py:473-477 "
        "mode=%s" % ("use-provider(fail-closed)" if opts["use_provider"] else "subclass_override"))
    # ★ 纪律：本入口的产物不得用作性能收益证据；plan_consistent 必须为 false。
    #   刻意做成**主动打印**而不是"文档里写一句" —— 让它在日志里可 grep、不可能被漏掉。
    log("FP_PLAN_CONSISTENT=false reason=diagnostic_entry_outer_proxy "
        "note=本入口的产物不得用作性能收益证据；正式计时请走 50_train.py 且 BATCHFP 全关")
    if cleaned != argv:
        log("FP_ARGV 已从框架 argv 中摘掉本入口自有参数：%s（其余原样留给 load_and_parse()）"
            % [a for a in argv if a not in cleaned])
    log("FP_ARGV 交给框架的 argv = %s" % cleaned)

    # ① 环境 fail-closed（同 50_train.py 的口径）
    rc = check_required_env()
    if rc:
        return rc

    # ② 指纹开关落到环境变量，并打印生效值
    _apply_fp_env(opts)

    # ③ 几何自检（复用 50_train.py 做法；GBS 红线同样拒绝执行）
    cfg_path = _resolve_config_path(cleaned)
    rc, gas_yaml = geometry_selfcheck(cfg_path, opts["allow_gbs_mismatch"])
    if rc:
        return rc

    # ④ provider 分支：fail-closed，绝不返回一个会静默换数据流的 provider
    if opts["use_provider"]:
        return _provider_deadend()

    # ⑤ 首选方案：ConfigManager → ConfigManager.load_and_parse() → _FPTrainer(...).train()
    #    （骨架照抄 trainer.py:473-477 的框架入口，只把 Trainer 换成 _FPTrainer）
    arguments = ConfigManager(config_class=Arguments).load_and_parse()

    gas_args, gas_src = _peek_gas_from_args(arguments)
    gas, gas_src_used = (gas_args, gas_src) if gas_args else (gas_yaml, "config_yaml.training.gradient_accumulation_steps")
    if not gas:
        gas, gas_src_used = None, "assumed_1"
    _batchfp.set_grad_accum_steps(gas, gas_src_used)
    log("FP_GAS grad_accum_steps=%s src=%s（仅用于**推导** step_hint；取不到即写成 gas1 假定，"
        "不影响数据通路）" % (gas if gas else "none", gas_src_used))

    log("FP_TRAIN_START trainer=_FPTrainer（只覆写 get_dataloader：super() 之后在最外层套只读代理）")
    _FPTrainer(args=arguments).train()
    log("FP_TRAIN_END（返回码/失败判定请沿用 50_train.py 的日志判据；本入口不产出性能数字）")
    return 0


if __name__ == "__main__":
    sys.exit(main())
