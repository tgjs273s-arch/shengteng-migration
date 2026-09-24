#!/bin/bash
# ============================================================
# bringup.sh — 昇腾机器一键拉起（v2 · 跨发行版 / 跨 Python 版本冗余）
#
# 用法: bash bringup.sh [--stage=N] [--pybin=/path/to/python3]
#       BRINGUP_LOG=<path> bash bringup.sh ...   # 覆盖日志路径（缺省 /root/ascend_bringup_run.log）
# 幂等: 已满足的阶段自动跳过; 每阶段输出 [PASS]/[FAIL]+修复提示
# 依赖: root
#
# v2 相对 v1 的修复（均由 2026-09-16 新机器实测暴露，见 docs/PITFALLS 坑 30-38）:
#   · 坑 37: 原实现硬编码 `apt install python3-dev build-essential` —— openEuler 无 apt，
#            且包名应为 python3-devel/gcc-c++。改为**按包管理器分派**。
#   · 坑 38: 原实现用裸 `python3 -m pip` —— 新机器上 `python3`=/usr/bin/python3.11（**无 pip**）
#            而 `pip3`=/usr/local/python3.12.13/bin/pip3，两者是**不同解释器** → 装完 import 不到。
#            改为**显式挑选并锁定单一解释器**（PYBIN），全程只用它。
#   · 坑 22 前置闸门: 装 triton 前必须确认 Python.h 存在，否则 kernel 编译失败后**静默回退 CPU**。
#   · 坑 10 补充: 源站可能返回 HTTP 429（限流）→ 装包改为带退避重试，而非立即失败。
# ============================================================
set -u
# ★ 坑 77：`MSMM_DIR` 在 phase4 被使用，但本脚本**从未定义它** ——
#   在 `set -u` 下直接 `unbound variable` 中止 phase4。凡跨 phase 使用的路径变量，
#   必须在文件顶部集中定义（并允许用环境变量覆盖）。
MSMM_DIR="${MSMM_DIR:-/root/MindSpeed-MM}"
DATA_DIR="${DATA_DIR:-/root/data}"
# ★ 坑 75：日志路径**必须允许调用方覆盖**。原实现写死 `LOG=/root/ascend_bringup_run.log`
#   并在非 TTY 时 `exec >> "$LOG"` —— 于是调用方写的 `... > /root/bringup2.log` 被**静默覆盖**，
#   `tail /root/bringup2.log` 永远是空的，看起来像"脚本没跑"。排查成本极高。
#   现支持 `BRINGUP_LOG=<path> bash scripts/bringup.sh ...`。
LOG="${BRINGUP_LOG:-/root/ascend_bringup_run.log}"
# ★ 坑 44：原实现无条件用 `exec > >(tee -a "$LOG") 2>&1` —— 进程替换会产生一个 tee 子进程，
#   它持有**原始的 stdout 写端**（即 SSH 通道）。于是在 SSH 里 `setsid nohup bash bringup.sh &`
#   之后，SSH 通道永远不关闭 → paramiko recv_exit_status 阻塞 → 启动命令挂死超时。
#   现改为：**仅当 stdout 是 TTY（交互式）才用 tee**；非交互（SSH 后台/重定向）直接追加到日志。
if [ -t 1 ]; then
  exec > >(tee -a "$LOG") 2>&1
else
  exec >> "$LOG" 2>&1
  echo "===== bringup start $(date '+%F %T') (non-tty: 输出仅写入 $LOG) ====="
fi

START_STAGE=0
PYBIN_OVERRIDE=""
for a in "$@"; do
  case "$a" in
    --stage=*)  START_STAGE="${a#--stage=}" ;;
    --pybin=*)  PYBIN_OVERRIDE="${a#--pybin=}" ;;
    *) [ -z "${1:-}" ] || true ;;
  esac
done

# CANN 环境（devel 镜像一般已设，但显式补齐更稳）
if [ -d /usr/local/Ascend/ascend-toolkit/latest ]; then
  export ASCEND_HOME_PATH=/usr/local/Ascend/ascend-toolkit/latest
  export LD_LIBRARY_PATH=/usr/local/Ascend/driver/lib64:/usr/local/Ascend/driver/lib64/common:/usr/local/Ascend/driver/lib64/driver:${ASCEND_HOME_PATH}/lib64:${ASCEND_HOME_PATH}/lib64/plugin/opskernel:${ASCEND_HOME_PATH}/lib64/plugin/nnengine:${LD_LIBRARY_PATH:-}
elif [ -d /usr/local/Ascend/cann ]; then
  export ASCEND_HOME_PATH=/usr/local/Ascend/cann
fi

say()  { echo; echo "===== $1 ====="; }
pass() { echo "[PASS] $1"; }
fail() { echo "[FAIL] $1"; echo "  修复: $2"; }
warn() { echo "[WARN] $1"; }

PYBIN=""          # ★ 全局锁定的解释器，后续所有阶段只用它
PKG=""            # 包管理器: dnf | yum | apt

# ---------- phase 0: 硬件 / CANN / OS / 包管理器 / Python 候选 ----------
phase0() {
  say "phase0 硬件/CANN/OS/包管理器/Python 探测"

  # 芯片
  if command -v npu-smi >/dev/null 2>&1; then
    NDIE=$(npu-smi info -l 2>/dev/null | awk -F: '/Chip Count/{gsub(/ /,"",$2); print $2}')
    NDEV=$(ls /dev/davinci[0-9]* 2>/dev/null | wc -l)
    CHIP=$(npu-smi info 2>/dev/null | awk '/Ascend/{print $3; exit}')
    pass "npu-smi 可用: die=${NDIE:-?} davinci=${NDEV} chip=${CHIP:-?}"
    [ "${NDIE:-0}" -ge 2 ] && pass "die>=2 → 可复刻官方 dp2 几何" \
      || warn "die<2 → 几何与官方不同，仅窗口口径可比（Skill 自动降级并标注）"
  else
    fail "npu-smi 不可用" "LD_LIBRARY_PATH 需含 /usr/local/Ascend/driver/lib64*"
  fi
  [ -d "${ASCEND_HOME_PATH:-/nonexistent}" ] && pass "CANN: $ASCEND_HOME_PATH" \
    || fail "CANN toolkit 缺失" "devel 镜像应自带；检查 /usr/local/Ascend/"

  # OS / 架构
  OS=$(grep -E '^PRETTY_NAME=' /etc/os-release 2>/dev/null | cut -d'"' -f2)
  echo "  OS=${OS:-unknown}  ARCH=$(uname -m)  KERNEL=$(uname -r)"

  # 包管理器分派（坑 37）
  if command -v dnf >/dev/null 2>&1; then PKG=dnf
  elif command -v yum >/dev/null 2>&1; then PKG=yum
  elif command -v apt-get >/dev/null 2>&1; then PKG=apt
  else PKG=""; fi
  [ -n "$PKG" ] && pass "包管理器 = $PKG" || fail "无可用包管理器" "需手动装编译器与 Python 头文件"

  # Python 候选枚举（坑 38）
  echo "  Python 候选："
  for cand in "$PYBIN_OVERRIDE" /usr/bin/python3 /usr/local/python3.12.13/bin/python3 \
              /usr/local/python3.11.6/bin/python3 "$(command -v python3 2>/dev/null)"; do
    [ -n "$cand" ] && [ -x "$cand" ] || continue
    v=$("$cand" -c 'import sys;print("%d.%d.%d"%sys.version_info[:3])' 2>/dev/null)
    haspip=$("$cand" -m pip -V >/dev/null 2>&1 && echo yes || echo no)
    echo "    $cand  version=${v:-?}  pip=$haspip"
  done
}

# ---------- 挑选并锁定解释器（含 pip 与 Python.h 保障） ----------
choose_python() {
  say "phase0b 锁定解释器 PYBIN（坑 38：python3 与 pip3 可能是不同解释器）"
  local cands="$PYBIN_OVERRIDE /usr/bin/python3 /usr/local/python3.12.13/bin/python3 $(command -v python3 2>/dev/null)"
  local seen=" "
  for cand in $cands; do
    [ -n "$cand" ] && [ -x "$cand" ] || continue
    case "$seen" in *" $cand "*) continue ;; esac
    seen="$seen$cand "
    local v
    v=$("$cand" -c 'import sys;print("%d.%d"%sys.version_info[:2])' 2>/dev/null) || continue
    echo "  尝试: $cand (py$v)"

    # 1) 保障 pip
    if ! "$cand" -m pip -V >/dev/null 2>&1; then
      if [ "$PKG" = "dnf" ] || [ "$PKG" = "yum" ]; then
        $PKG install -y python3-pip >/dev/null 2>&1
      elif [ "$PKG" = "apt" ]; then
        apt-get install -y -qq python3-pip >/dev/null 2>&1
      fi
      "$cand" -m pip -V >/dev/null 2>&1 || {
        timeout 90 curl -sSL --max-time 80 https://bootstrap.pypa.io/get-pip.py -o /tmp/gp.py 2>/dev/null \
          && "$cand" /tmp/gp.py >/dev/null 2>&1
      }
    fi
    "$cand" -m pip -V >/dev/null 2>&1 || { echo "    ✗ 无 pip，跳过"; continue; }

    # 2) 保障 Python.h（坑 22 前置闸门）
    local inc
    inc=$("$cand" -c 'import sysconfig;print(sysconfig.get_paths()["include"])' 2>/dev/null)
    if [ ! -f "$inc/Python.h" ]; then
      echo "    Python.h 缺失（$inc）→ 安装 dev 包"
      if [ "$PKG" = "dnf" ] || [ "$PKG" = "yum" ]; then
        $PKG install -y python3-devel gcc gcc-c++ make >/dev/null 2>&1
      elif [ "$PKG" = "apt" ]; then
        apt-get install -y -qq python3-dev build-essential >/dev/null 2>&1
      fi
      [ -f "$inc/Python.h" ] || { echo "    ⚠ Python.h 仍缺失（后续 triton 有静默回退 CPU 风险）"; }
    fi
    [ -f "$inc/Python.h" ] && echo "    ✓ Python.h ok: $inc/Python.h" || echo "    ✗ Python.h 缺失"

    PYBIN="$cand"
    pass "锁定 PYBIN=$PYBIN (py$v)  pip=$("$PYBIN" -m pip -V 2>/dev/null | cut -d' ' -f1-2)"
    return 0
  done
  fail "找不到可用解释器" "手工指定: bash bringup.sh --pybin=/path/to/python3"
  return 1
}

# ---------- pip 安装（带 429 退避重试，坑 10） ----------
pip_install() {
  local pkg="$1"; shift
  local i=1
  while [ $i -le 3 ]; do
    if "$PYBIN" -m pip install -q "$@" $pkg 2>&1 | tail -2; then
      return 0
    fi
    echo "    (第 $i 次安装 $pkg 未成功，退避 $((i*10))s 后重试)"
    sleep $((i*10)); i=$((i+1))
  done
  return 1
}

# ---------- 依赖闭环求解（★ 不枚举包名，靠实测逐个暴露并补齐） ----------
# ★ 坑 86：前三轮我一路上"猜依赖"（einops → pydantic → protobuf → accelerate），
#   每轮真机只能推进一个 —— 因为**枚举永远不全**（与坑 84 同因）。
#   现改为**闭环**：反复尝试导入训练入口模块，从 `No module named 'X'` 里取出真实缺失项，
#   装完再试，直到通过或达到迭代上限。这把"猜测"变成"求解"。
map_pkg() {
  case "$1" in
    yaml)            echo "pyyaml" ;;
    PIL)             echo "pillow" ;;
    cv2)             echo "opencv-python-headless" ;;
    sklearn)         echo "scikit-learn" ;;
    google|google.protobuf) echo "protobuf" ;;
    dateutil)        echo "python-dateutil" ;;
    *)               echo "$1" ;;
  esac
}

resolve_deps() {
  local max_iter=15 i=1 mod pkg out
  while [ "$i" -le "$max_iter" ]; do
    out=$(cd "$MSMM_DIR" 2>/dev/null && env NON_MEGATRON=true PYTHONPATH="$MSMM_DIR" \
          "$PYBIN" -c "import mindspeed_mm.fsdp.train.trainer" 2>&1)
    if ! echo "$out" | grep -qE 'Traceback|Error'; then
      pass "依赖闭环：训练入口可导入（补齐迭代 $((i - 1)) 次）"
      return 0
    fi
    mod=$(echo "$out" | grep -oE "No module named '[^']+'" | head -1 \
          | sed "s/No module named '//; s/'$//")
    if [ -z "$mod" ]; then
      warn "依赖闭环：出现非 ModuleNotFound 的导入错误，停止自动补齐（需人工判读）"
      echo "$out" | grep -E "^[A-Za-z_][A-Za-z0-9_.]*(Error|Exception)" | head -3
      return 1
    fi
    pkg=$(map_pkg "$mod")
    echo "  [闭环 $i/$max_iter] 缺 $mod → 安装 $pkg"
    pip_install "$pkg" || warn "安装 $pkg 失败，继续尝试后续"
    i=$((i + 1))
  done
  warn "依赖闭环达到上限 $max_iter 次仍有缺失（把上面的模块名发我，或手工安装）"
  return 1
}

# ---------- phase 1: torch / torch_npu ----------
phase1() {
  say "phase1 torch / torch_npu（解释器: ${PYBIN:-未锁定}）"
  [ -n "$PYBIN" ] || { fail "PYBIN 未锁定" "先修 phase0b"; return 1; }

  if "$PYBIN" -c "import torch, torch_npu" >/dev/null 2>&1; then
    pass "torch/torch_npu 已装: $("$PYBIN" -c 'import torch,torch_npu;print(torch.__version__, torch_npu.__version__)' 2>/dev/null)"
  else
    echo "  安装 torch==2.7.1 torch_npu==2.7.1.post10 ..."
    pip_install "torch==2.7.1 torch_npu==2.7.1.post10 pyyaml numpy" \
      --index-url https://mirrors.huaweicloud.com/ascend/repos/pypi \
      --extra-index-url https://repo.huaweicloud.com/repository/pypi/simple \
      || echo "  ⚠ 首选源失败，回退纯 pypi"
    "$PYBIN" -c "import torch_npu" >/dev/null 2>&1 \
      || pip_install "torch_npu==2.7.1.post10" \
           --extra-index-url https://mirrors.huaweicloud.com/ascend/repos/pypi
    "$PYBIN" -c "import torch, torch_npu" >/dev/null 2>&1 \
      && pass "torch/torch_npu 安装成功" \
      || { fail "torch_npu 安装失败" "确认 py$("$PYBIN" -c 'import sys;print("%d.%d"%sys.version_info[:2])' 2>/dev/null) 是否有 wheel；必要时 ---pybin 换解释器"; return 1; }
  fi

  # 可见性 + 实算
  source /usr/local/Ascend/ascend-toolkit/set_env.sh 2>/dev/null
  N=$("$PYBIN" -c "import torch,torch_npu;print(torch.npu.device_count())" 2>/dev/null)
  [ "${N:-0}" -ge 1 ] && pass "torch_npu 可见 ${N} 个 device" \
    || fail "torch_npu 看不到卡" "检查驱动/CANN 与 torch_npu 版本匹配"
  "$PYBIN" -c "import torch,torch_npu; a=torch.randn(64,64,device='npu'); assert bool((a@a.T).sum().isfinite()); print('matmul ok')" >/dev/null 2>&1 \
    && pass "NPU matmul ok" || fail "NPU 计算失败" "查驱动/卡健康"
}

# ---------- phase 2: MindSpeed-MM + transformers + triton-ascend ----------
phase2() {
  say "phase2 MindSpeed-MM v26.1.0 + transformers + triton-ascend"
  [ -n "$PYBIN" ] || { fail "PYBIN 未锁定" "先修 phase0b"; return 1; }

  if [ -d /root/MindSpeed-MM ]; then
    T=$(git -C /root/MindSpeed-MM describe --tags 2>/dev/null)
    echo "  当前 tag: ${T:-未知}"
    [ "$T" = "v26.1.0" ] && pass "MSMM v26.1.0" || warn "tag 非 v26.1.0，按需切换"
  else
    git clone --depth 1 --branch v26.1.0 https://github.com/Ascend/MindSpeed-MM.git /root/MindSpeed-MM >/dev/null 2>&1 \
      && pass "MSMM cloned v26.1.0" \
      || fail "clone 失败" "网络受限时改用 gitcode 镜像: https://gitcode.com/Ascend/MindSpeed-MM.git"
  fi

  "$PYBIN" -c "import transformers; assert transformers.__version__=='5.2.0'" >/dev/null 2>&1 \
    && pass "transformers 5.2.0" || pip_install "transformers==5.2.0 pyyaml"

  # ---- triton-ascend：**先实测，再决定装不装**（冗余优先，不按版本号盲判）----
  #   ★ 坑 46：旧实现用 `triton.program_id(0)` 判据 → 该属性在 JIT 命名空间不存在 →
  #     假阴性 → Skill 误判 triton 不可用 → 降级 ascendc/eager → **白扔性能最优路径**。
  #     现：判据唯一权威来源 = scripts/00b_triton_min_kernel.py（规范写法 tl.program_id）。
  #   ★ 版本策略：不再假设"必须 3.2.2"。**能用就用**（记录实际版本）；不通才按配对装 3.2.2 复测。
  #     理由：实测 CANN 9.1.0-beta.3 镜像自带 triton-ascend 3.2.0 且 kernel 通过，
  #     而 A3(CANN 9.1.0 GA) 上 3.2.0/3.2.1 会回退 CPU → 配对关系**取决于 CANN 版本**，须实测。
  MIN_KERNEL="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/00b_triton_min_kernel.py"
  check_triton() {
    [ -f "$MIN_KERNEL" ] || { echo "TRITON_NPU_FAIL reason=probe_script_missing"; return 1; }
    "$PYBIN" "$MIN_KERNEL" 2>&1 | tail -3
  }

  T_OUT=$(check_triton)
  if echo "$T_OUT" | grep -q TRITON_NPU_OK; then
    TV=$("$PYBIN" -c 'import triton;print(triton.__version__)' 2>/dev/null)
    pass "triton 可用（实测 kernel 通过）: $T_OUT"
    echo "      → triton-ascend $TV 在本机（CANN 版本 + torch 组合）下**实测可用**，无需按 lock 强改"
  else
    warn "现有 triton 不可用，按配对尝试安装 3.2.2 后复测"
    echo "      首测结果: $T_OUT"
    pip_install "triton-ascend==3.2.2 attrs==24.2.0" \
      --index-url https://mirrors.huaweicloud.com/ascend/repos/pypi \
      --extra-index-url https://repo.huaweicloud.com/repository/pypi/simple
    T_OUT=$(check_triton)
    if echo "$T_OUT" | grep -q TRITON_NPU_OK; then
      pass "装 3.2.2 后 triton 可用: $T_OUT"
    else
      warn "triton 两轮均不通 → Skill 将自动降级后端 triton→ascendc→eager，并标注 degraded"
      echo "      末次结果: $T_OUT"
      echo "      排查: ① Python.h 是否就位（坑22）② triton-ascend 版本配对（坑20/46）③ 清场重装"
    fi
  fi

  # ★ 坑 79：P4 的资产下载依赖 **modelscope CLI**（`modelscope download ...`）。
  #   未安装时下载会以含糊方式失败（真机上权重一直是 0）。此处确保装上。
  if command -v modelscope >/dev/null 2>&1; then
    pass "modelscope CLI 已装"
  else
    echo "  安装 modelscope CLI（P4 下载依赖）..."
    pip_install "modelscope" 2>&1 | tail -2
    command -v modelscope >/dev/null 2>&1 \
      && pass "modelscope CLI 安装成功" \
      || warn "modelscope 未装上 → P4 将走备源或失败（权重/llava/COCO 需人工介入）"
  fi

  # ★ 坑 85：MSMM v26.1.0 **没有顶层的运行期 requirements.txt** —— 仓库里唯一的
  #   `requirements.txt` 在 `UserGuide/` 下，内容是 **sphinx 文档依赖**，与运行无关。
  #   （第一版"消费 requirements.txt"的前提是错的；真机上正是**核心兜底清单**救了场：
  #    补装了 einops/pydantic/PIL。）现改为：
  #     ① 只认**运行期**依赖声明：顶层 requirements*.txt / pyproject.toml / setup.py 的 install_requires
  #     ② 明确跳过 UserGuide / docs 下的文档依赖
  #     ③ 无论如何都用一份**核心清单**兜底，并以 **mock 冒烟**作为最终权威判据
  MSMM_ROOT=/root/MindSpeed-MM
  RT_REQ=""
  for cand in "$MSMM_ROOT/requirements.txt" "$MSMM_ROOT/requirements-dev.txt"; do
    [ -f "$cand" ] && RT_REQ="$cand" && break
  done
  if [ -n "$RT_REQ" ]; then
    echo "  安装 MSMM 运行期依赖: $RT_REQ"
    "$PYBIN" -m pip install -q -r "$RT_REQ" \
      --index-url https://mirrors.huaweicloud.com/ascend/repos/pypi \
      --extra-index-url https://repo.huaweicloud.com/repository/pypi/simple 2>&1 | tail -3 \
      && pass "运行期 requirements.txt 安装完成" \
      || warn "requirements.txt 安装未完全成功，继续用核心清单兜底"
  else
    echo "  MSMM 无顶层运行期 requirements.txt（UserGuide/ 下的是 sphinx 文档依赖，已跳过）"
    if [ -f "$MSMM_ROOT/pyproject.toml" ] || [ -f "$MSMM_ROOT/setup.py" ]; then
      echo "  尝试按项目元数据安装（install_requires）..."
      "$PYBIN" -m pip install -q -e "$MSMM_ROOT" --no-deps 2>&1 | tail -2
    fi
  fi

  # 核心依赖兜底（幂等；含 protobuf —— 真机上 torch/CANN 工具链会读它的元数据，见坑 83）
  # 核心清单先装一轮（快速命中常见项），随后由闭环求解兜底剩余未知项
  MSMM_CORE="einops pydantic pyyaml numpy pillow safetensors tokenizers sentencepiece protobuf accelerate datasets jsonargparse docstring-parser"
  MISSING=""
  for m in einops pydantic yaml numpy PIL safetensors; do
    "$PYBIN" -c "import $m" >/dev/null 2>&1 || MISSING="$MISSING $m"
  done
  if [ -n "$MISSING" ]; then
    echo "  补装核心依赖:$MISSING"
    pip_install "$MSMM_CORE" || true
  fi
  CORE_OK=1
  for m in einops pydantic yaml numpy; do
    "$PYBIN" -c "import $m" >/dev/null 2>&1 || { CORE_OK=0; echo "  ✗ 仍缺: $m"; }
  done
  [ "$CORE_OK" = "1" ] && pass "MSMM 核心依赖（einops/pydantic/pyyaml/numpy）齐备" \
    || fail "MSMM 核心依赖缺失" \
         "手工: $PYBIN -m pip install $MSMM_CORE"

  # ★ 坑 83：`protobuf` 的**元数据**必须可查 —— 真机报
  #   `PackageNotFoundError: No package metadata was found for protobuf`，
  #   即"包可能能 import，但 importlib.metadata 查不到版本"。这类缺失只看 import 是发现不了的。
  if "$PYBIN" -c "import importlib.metadata as m; m.version('protobuf')" >/dev/null 2>&1; then
    pass "protobuf 元数据可查（$("$PYBIN" -c "import importlib.metadata as m;print(m.version('protobuf'))" 2>/dev/null)）"
  else
    echo "  protobuf 元数据缺失 → 重装以补齐 dist-info ..."
    "$PYBIN" -m pip install -q --force-reinstall --no-deps protobuf 2>&1 | tail -2
    "$PYBIN" -c "import importlib.metadata as m; m.version('protobuf')" >/dev/null 2>&1 \
      && pass "protobuf 元数据已补齐" \
      || warn "protobuf 元数据仍缺失（若 mock 冒烟通过可暂时忽略，否则手工: pip install protobuf）"
  fi

  # ★ 坑 95：**版本锁的"关键运行依赖"清单本身就是一份可执行的契约**，必须被消费。
  #   实测：缺 `torchvision` → `AutoProcessor` 回退 → 抛 `TypeError: argument of type
  #   'NoneType' is not iterable` → MSMM 报 `Processor was not found, please check and
  #   update your model file.` —— **错误信息把矛头指向"模型文件"，而真因是缺一个 Python 包**，
  #   排查被严重误导（而 torchvision 恰好就写在 versions.lock 里，是我没装）。
  #
  #   安全注意：torchvision 必须与 torch 2.7.1 **配对**（0.22.x），且用 `--no-deps`
  #   安装，避免 pip 顺手升级 torch 而破坏 NPU 栈。
  LOCK_FILE="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)/config/versions.lock"
  if [ -f "$LOCK_FILE" ]; then
    LOCK_DEPS="$(awk '/关键运行依赖/{f=1;next} f&&NF&&$0!~/^#/{print;exit}' "$LOCK_FILE" \
                 | tr '|' ' ' | tr -s ' ')"
    echo "  versions.lock 关键运行依赖: $LOCK_DEPS"
    for d in $LOCK_DEPS; do
      pkg_only="${d%%==*}"
      mod="$pkg_only"
      case "$pkg_only" in
        pyyaml) mod=yaml ;;
        pillow) mod=PIL ;;
        qwen_vl_utils) mod=qwen_vl_utils ;;
        attrs)   mod=attr ;;
      esac
      if "$PYBIN" -c "import $mod" >/dev/null 2>&1; then
        continue
      fi
      echo "  [lock 依赖] 缺 $pkg_only → 安装"
      # torchvision 与 torch 强配对，且绝不能让它改动 torch
      case "$pkg_only" in
        torchvision) pip_install "torchvision==0.22.1" --no-deps || warn "torchvision 安装失败" ;;
        *)           pip_install "$d" --no-deps || pip_install "$d" || warn "$pkg_only 安装失败" ;;
      esac
    done
    # 专项确认 torchvision（processor 依赖它；缺失会给出极具误导性的错误）
    if "$PYBIN" -c "import torchvision" >/dev/null 2>&1; then
      pass "torchvision 就位（$("$PYBIN" -c 'import torchvision;print(torchvision.__version__)' 2>/dev/null)）"
    else
      fail "torchvision 缺失" \
           "AutoProcessor 会回退并抛出误导性的 'Processor was not found'；" \
           "手工: $PYBIN -m pip install torchvision==0.22.1 --no-deps"
    fi
  else
    warn "未找到 versions.lock（$LOCK_FILE），跳过 lock 依赖安装"
  fi

  # ★ 坑 71：MSMM 依赖 **MindSpeed 核心包**（`versions.lock` 早已写明：
  #   `mindspeed | 0.12.1 | gitcode.com/Ascend/MindSpeed editable --no-deps`），
  #   但原实现只检查 `mindspeed_mm` 且**仅 warn** → 训练时必然
  #   `ModuleNotFoundError: No module named 'mindspeed'` → **P5 永远失败**。
  #   现：真正安装，并给出可执行修复。
  if "$PYBIN" -c "import mindspeed" >/dev/null 2>&1; then
    pass "mindspeed 已装 $("$PYBIN" -c 'import mindspeed;print(getattr(mindspeed,"__version__","?"))' 2>/dev/null)"
  else
    echo "  安装 MindSpeed 核心包（MSMM 必需；editable --no-deps）..."
    MS_OK=0
    for u in https://gitcode.com/Ascend/MindSpeed.git \
             https://github.com/Ascend/MindSpeed.git \
             https://gitee.com/ascend/MindSpeed.git; do
      rm -rf /root/MindSpeed
      if git clone --depth 1 "$u" /root/MindSpeed >/dev/null 2>&1; then
        echo "  已克隆: $u"
        "$PYBIN" -m pip install -q -e /root/MindSpeed --no-deps 2>&1 | tail -2
        MS_OK=1
        break
      fi
    done
    if [ "$MS_OK" = "1" ] && "$PYBIN" -c "import mindspeed" >/dev/null 2>&1; then
      pass "mindspeed 安装成功（/root/MindSpeed, editable --no-deps）"
    else
      fail "mindspeed 未安装" \
        "MSMM 必需核心依赖（versions.lock: gitcode.com/Ascend/MindSpeed editable --no-deps）；" \
        "手工: git clone https://gitcode.com/Ascend/MindSpeed.git /root/MindSpeed && $PYBIN -m pip install -e /root/MindSpeed --no-deps"
    fi
  fi

  # mindspeed_mm 可导入性：优先 pip install -e（失败则由 PYTHONPATH 兜底，见 50_train.py）
  # ★ 坑 80：`mindspeed_mm/__init__.py` 有守卫 —— **不设 `NON_MEGATRON=true` 时会去 import
  #   megatron 并抛 `ModuleNotFoundError: No module named 'megatron'`**（本项目不用 Megatron）。
  #   于是任何"MSMM 装好了吗"的裸导入检查都会得到**假阴性**：明明装好了却报缺 megatron。
  #   → 判定 MSMM 可导入性时**必须带上 NON_MEGATRON=true**（训练本身也需要它）。
  MSMM_ENV="NON_MEGATRON=true"
  if env $MSMM_ENV "$PYBIN" -c "import mindspeed_mm" >/dev/null 2>&1; then
    pass "mindspeed_mm import ok（NON_MEGATRON=true）"
  else
    echo "  安装 mindspeed_mm（editable, --no-deps）..."
    "$PYBIN" -m pip install -q -e /root/MindSpeed-MM --no-deps 2>&1 | tail -2
    # 兜底：仅在 MSMM 目录下可导入也接受（torchrun 从该目录启动）
    (cd /root/MindSpeed-MM 2>/dev/null && env $MSMM_ENV "$PYBIN" -c "import mindspeed_mm" >/dev/null 2>&1) \
      && pass "mindspeed_mm 可在 MSMM 目录内导入（50_train.py 会注入 PYTHONPATH + NON_MEGATRON）" \
      || fail "mindspeed_mm 不可导入" \
           "已确保 NON_MEGATRON=true；检查 /root/MindSpeed-MM 是否完整、mindspeed 是否已装"
  fi

  # ★ 坑 86：依赖闭环求解 —— 以"训练入口能否导入"为判据，逐个暴露并补齐真实缺失项
  resolve_deps
}

# ---------- phase 3: 权重 / 数据（**做实际补齐**，不只是检查） ----------
phase3() {
  say "phase3 权重与数据（补齐缺失项；不触发 COCO 19GB 下载）"
  [ -n "$PYBIN" ] || PYBIN=$(command -v python3)
  SCRIPT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/40_prepare_assets.py"
  if [ ! -f "$SCRIPT" ]; then
    warn "未找到 40_prepare_assets.py，跳过"
    return 0
  fi
  # ★ 坑 99：原实现用 `--no-download` 调本脚本 —— 那是**纯检查模式**，
  #   于是 **hf→dcp 转换永远不会被执行**；而 phase4 的 mock 冒烟，
  #   训练第 1 步后就要从 DCP 检查点恢复（`self.load()`）→ 必然
  #   `FileNotFoundError: /root/Qwen3.5-0.8B-dcp/iter_-000001/.metadata`。
  #   **流程顺序与模式必须匹配：mock（phase4）依赖 DCP，DCP 必须在 phase3 产出。**
  #   现改为：`--skip-coco`（允许补齐权重/DCP/转换，但不拉 19GB 图片；图片已存在时也不会重下）。
  "$PYBIN" "$SCRIPT" --data-dir "$DATA_DIR" --model-dir /root \
    --msmm-dir "$MSMM_DIR" --skip-coco --out /tmp/_assets_provision 2>&1 | tail -22
  rc=${PIPESTATUS[0]}
  if [ "$rc" = "0" ]; then
    pass "资产齐备（含 DCP 权重与转换产物）"
  else
    warn "资产未完全齐备（rc=$rc）—— 见上方明细；mock/训练可能失败"
    echo "  → 全量补齐（含 COCO 19GB）: $PYBIN $SCRIPT --data-dir $DATA_DIR --msmm-dir $MSMM_DIR"
  fi
}

# ---------- phase 4: mock 迁移验证 ----------
phase4() {
  say "phase4 mock 迁移验证（冒烟）"
  [ -n "$PYBIN" ] || PYBIN=$(command -v python3)
  # ★ 坑 48：原实现假定 `/root/MindSpeed-MM/examples/qwen3_5/qwen3_5_0_8B_mock_config.yaml` 存在
  #   —— 该文件其实是开发期自己放的，**从未随 Skill 打包**。全新 clone 的 MSMM 里没有它，
  #   于是 phase4 永远被跳过（"缺 mock 配置"）。现改为：从 Skill 自带的 config/templates/
  #   provision 到 MSMM 的 examples 目录（幂等；已存在则不覆盖）。
  SKILL_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
  MOCK_SRC="$SKILL_DIR/config/templates/qwen3_5_0_8B_mock.yaml"
  MSMM_EX="$MSMM_DIR/examples/qwen3_5"
  MOCK_CFG="$MSMM_EX/qwen3_5_0_8B_mock_config.yaml"

  if [ ! -d "$MSMM_DIR" ]; then
    warn "MSMM 目录不存在（$MSMM_DIR），跳过 phase4"
    return 0
  fi
  mkdir -p "$MSMM_EX"
  if [ ! -f "$MOCK_CFG" ]; then
    if [ -f "$MOCK_SRC" ]; then
      cp "$MOCK_SRC" "$MOCK_CFG" && echo "  已 provision mock 配置: $MOCK_SRC → $MOCK_CFG"
    else
      warn "Skill 内也没有 mock 模板（$MOCK_SRC），跳过 phase4"
      return 0
    fi
  else
    echo "  mock 配置已存在: $MOCK_CFG（不覆盖）"
  fi

  cd "$MSMM_DIR" || return 1
  source /usr/local/Ascend/ascend-toolkit/set_env.sh 2>/dev/null
  export NON_MEGATRON=true TASK_QUEUE_ENABLE=2 ASCEND_LAUNCH_BLOCKING=0
  export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True TRITON_CACHE_DIR=/root/triton_cache

  # ★ 坑 89：mock **配置**随包发布了，但它依赖的 **mock 数据集**没有 ——
  #   `data/mocked_vl_data/mock_data_pic_num_4_textlen_512.json` 是开发期在旧机器上生成的，
  #   与坑 48（mock 配置本身）是同族问题：**配置与其数据必须一起被"发现"。**
  #   正解不是"把数据也塞进包里"，而是**调用 MSMM 自带的生成器**（项目自己的工具最可靠）。
  # ★ 坑 91：但生成器的**输出文件名与其参数有关**（实测只给了 jpg，没给配置期望的 json）。
  #   因此不能假设"跑一次就有正确文件名" —— 改为**自愈式适配**：
  #     ① 按配置文件名的语义给出参数（mock_data_pic_num_4_textlen_512 → pic_num=4, text_len=512）
  #     ② 生成后若配置期望的文件仍缺失，就把配置里的 dataset 路径**改指向实际产出的 json**
  MOCK_DATA_DIR="$MSMM_DIR/data/mocked_vl_data"
  MOCK_GEN="mindspeed_mm/fsdp/tools/data_tool/generate_mock_data_for_vlmodel.py"
  MOCK_JSON_EXPECTED="$(grep -oE '/[^ ]*mock_data[^ ]*\.json' "$MOCK_CFG" 2>/dev/null | head -1)"
  if [ -d "$MOCK_DATA_DIR" ] && [ -n "$(ls -A "$MOCK_DATA_DIR" 2>/dev/null)" ]; then
    echo "  mock 数据目录已存在: $MOCK_DATA_DIR"
  fi
  if [ ! -f "$MOCK_JSON_EXPECTED" ] && [ -f "$MOCK_GEN" ]; then
    echo "  生成 mock 数据集（调用 MSMM 自带生成器）..."
    mkdir -p "$MOCK_DATA_DIR"
    # ★ 坑 92：生成器的**真实参数名**是 `--num_pics` / `--text_length`（不是 pic_num/text_len），
    #   且 `--tokenizer_path` 的默认值是 `/home/weights/Qwen3.5-35B-A3B/` —— **本机不存在**，
    #   于是它在"画完图之后、写 json 之前"就死了（所以只留下一张 jpg）。
    #   正解：**从配置反推参数**，而不是照抄文件名去猜：
    #     · 期望文件名 mock_data_pic_num_4_textlen_512.json → --num_pics 4 --text_length 512
    #     · tokenizer 路径取自配置里的 model_name_or_path
    MOCK_NUM_PICS="$(echo "$MOCK_JSON_EXPECTED" | grep -oE 'pic_num_[0-9]+' | grep -oE '[0-9]+' | head -1)"
    MOCK_TEXT_LEN="$(echo "$MOCK_JSON_EXPECTED" | grep -oE 'textlen_[0-9]+' | grep -oE '[0-9]+' | head -1)"
    MOCK_TOKENIZER="$(grep -oE 'model_name_or_path:[[:space:]]*[^ #]+' "$MOCK_CFG" 2>/dev/null | head -1 | sed 's/.*:[[:space:]]*//')"
    [ -n "$MOCK_NUM_PICS" ] || MOCK_NUM_PICS=4
    [ -n "$MOCK_TEXT_LEN" ] || MOCK_TEXT_LEN=512
    [ -n "$MOCK_TOKENIZER" ] || MOCK_TOKENIZER=/root/Qwen3.5-0.8B-hf
    echo "    参数: --num_pics $MOCK_NUM_PICS --text_length $MOCK_TEXT_LEN --tokenizer_path $MOCK_TOKENIZER"
    "$PYBIN" "$MOCK_GEN" \
      --num_pics "$MOCK_NUM_PICS" --text_length "$MOCK_TEXT_LEN" \
      --tokenizer_path "$MOCK_TOKENIZER" \
      --save_dir "$MOCK_DATA_DIR/" 2>&1 | tail -4
  fi
  # 自愈：若期望的 json 仍不存在，找出目录里实际存在的 json 并改配置指向它
  if [ ! -f "$MOCK_JSON_EXPECTED" ]; then
    ACTUAL_JSON="$(ls -1 "$MOCK_DATA_DIR"/*.json 2>/dev/null | head -1)"
    if [ -n "$ACTUAL_JSON" ]; then
      echo "  自愈：把 mock 配置的 dataset 指向实际产出 → $ACTUAL_JSON"
      "$PYBIN" - "$MOCK_CFG" "$ACTUAL_JSON" <<'PYFIX' 2>&1 | tail -3
import re, sys
cfg, actual = sys.argv[1], sys.argv[2]
t = open(cfg, encoding="utf-8").read()
t2 = re.sub(r'(?m)^(\s*dataset:\s*).*$', r'\g<1>' + actual, t, count=1)
if t2 != t:
    open(cfg, "w", encoding="utf-8").write(t2)
    print("PATCHED dataset ->", actual)
else:
    print("NO_PATCH (未找到 dataset: 行)")
PYFIX
    else
      warn "mock 目录里没有任何 json（生成器可能未按预期工作）"
      echo "      生成器帮助：$PYBIN $MOCK_GEN --help"
    fi
  fi
  if [ -f "$MOCK_JSON_EXPECTED" ] || ls -1 "$MOCK_DATA_DIR"/*.json >/dev/null 2>&1; then
    pass "mock 数据集就绪: $(ls -1 "$MOCK_DATA_DIR" | tr '\n' ' ')"
  else
    fail "mock 数据集未就绪" "手工: $PYBIN $MOCK_GEN --help 查看参数"
  fi

  # ★ 坑 87：**依赖闭环的探针必须就是最终要跑通的那件事本身**。
  #   上一版 resolve_deps() 用 `import mindspeed_mm.fsdp.train.trainer` 作探针 ——
  #   它只覆盖"trainer 模块的 import 图"，**不覆盖运行期图**（数据加载需要 `datasets`）。
  #   于是闭环在"训练入口可导入"处**错误收敛**，mock 依旧失败。
  #   现改为：**以 mock 冒烟本身为判据**，每次失败提取真实缺失模块并补齐，再跑一遍（上限 10 轮）。
  MOCK_ITER=0
  MOCK_MAX=10
  while [ "$MOCK_ITER" -le "$MOCK_MAX" ]; do
    timeout 900 "$PYBIN" -m torch.distributed.run --nproc_per_node 1 --nnodes 1 --node_rank 0 \
      --master_addr localhost --master_port 6050 \
      mindspeed_mm/fsdp/train/trainer.py "$MOCK_CFG" > /root/mock_run.log 2>&1
    # ★ 坑 96：**判据不能只是"日志里出现过 iteration"** —— 实测 mock 跑完 1 步后崩溃，
    #   而 `grep -q iteration` 依然满足 → **假 PASS**（"判据弱于问题"家族第 7 次）。
    #   现要求：① 至少一次 iteration ② 且**日志中不含致命错误标记**。
    if grep -q 'iteration' /root/mock_run.log \
       && ! grep -qE 'ChildFailedError|ERR99999|Traceback \(most recent call last\)' /root/mock_run.log; then
      pass "mock 训练跑通（/root/mock_run.log；iteration=$(grep -c iteration /root/mock_run.log) 次；依赖闭环补齐 $MOCK_ITER 次）"
      return 0
    fi
    if grep -q 'iteration' /root/mock_run.log; then
      warn "mock 有 iteration 但**日志以致命错误结尾** → 判定为未跑通（坑 96：假 PASS）"
    fi
    MOD=$(grep -oE "No module named '[^']+'" /root/mock_run.log | head -1 \
          | sed "s/No module named '//; s/'$//")
    if [ -z "$MOD" ] || [ "$MOCK_ITER" -ge "$MOCK_MAX" ]; then
      break
    fi
    PKG=$(map_pkg "$MOD")
    echo "  [mock闭环 $((MOCK_ITER + 1))/$MOCK_MAX] 缺 $MOD → 安装 $PKG"
    pip_install "$PKG" || warn "安装 $PKG 失败，继续尝试"
    MOCK_ITER=$((MOCK_ITER + 1))
  done

  # 到这里说明闭环没能自愈
  fail "mock 训练失败" "tail -50 /root/mock_run.log"
  echo "  ── 真实异常（★ 坑 88：必须容忍 torchrun 的 '[rankN]: ' 行前缀）──"
  # ★ 坑 88：torchrun 给**每一行**加 `[rank0]: ` 前缀 —— 于是所有 `^Xxx` 行首锚定模式**全部失效**，
  #   提取器在真机上一个字都打不出来（上一版就是这样，白排查一轮）。
  #   凡面向训练日志的匹配，都要容忍该前缀。
  grep -nE '^(\[[^]]+\]:[[:space:]]*)?[A-Za-z_][A-Za-z0-9_.]*(Error|Exception)(:|$)' \
    /root/mock_run.log 2>/dev/null | tail -6
  echo "  ── 最深的调用帧（通常直接指出真凶）─────────────────"
  grep -nE 'File ".*", line [0-9]+' /root/mock_run.log 2>/dev/null | tail -6
  echo "  ── 第一处 Traceback 起 25 行 ─────────────────────"
  awk '/Traceback \(most recent call last\)/{c++} c==1{print; n++} n>=25{exit}' \
    /root/mock_run.log 2>/dev/null
  if [ -n "${MOD:-}" ]; then
    echo "  → 闭环已尝试 $MOCK_MAX 轮仍缺模块: $MOD（手工: $PYBIN -m pip install $(map_pkg "$MOD")）"
  else
    echo "  → 闭环判定：**本次失败不是缺 Python 模块**（未出现 No module named），故未继续补装。"
    echo "     请按上面「最深的调用帧」定位（配置/数据路径/权限/显存等）。"
    echo "     常见检查 —— mock 数据是否随 MSMM 仓库存在："
    ls -la /root/MindSpeed-MM/data/mocked_vl_data/ 2>&1 | head -5
  fi
  echo "  ────────────────────────────────────────────"
}

# ============================ 主流程 ============================
echo "=================================================================="
echo " 昇腾 bringup v2   起始阶段=$START_STAGE   $(date '+%F %T')"
echo " 日志: $LOG"
echo "=================================================================="
phase0
# ★ 坑 78：原实现把 `choose_python` 放在 `if [ START_STAGE -le 1 ]` 里 ——
#   于是 `--stage=2`（最常见的"续跑安装依赖"用法）**跳过了解释器锁定** →
#   到 phase2 直接 `[FAIL] PYBIN 未锁定` → **`--stage≥2` 从设计上就不可能工作**。
#   phase0 / phase0b 都是**廉价且幂等**的，应当无条件先跑（它们不安装任何东西，
#   只做探测与系统依赖补齐），阶段号只用来控制"要不要执行某个 phase"。
choose_python || echo ">>> 解释器未锁定（phase1+ 将无法执行）"
for s in $(seq "$START_STAGE" 4); do
  [ "$s" -eq 0 ] && continue
  case "$s" in
    1) phase1 || echo ">>> 阶段 1 未通过, 修复后重跑: bash bringup.sh --stage=1" ;;
    2) phase2 ;;
    3) phase3 ;;
    4) phase4 ;;
  esac
done
echo
echo ">>> 全流程结束 $(date '+%F %T')   PYBIN=${PYBIN:-未锁定}  PKG=${PKG:-无}"
