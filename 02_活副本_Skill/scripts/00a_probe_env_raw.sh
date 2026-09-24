#!/bin/bash
# ============================================================================
# probe_env_readonly.sh —— 昇腾机器只读环境探测（通用版，任意型号/发行版）
#
# 纪律：不安装、不写配置、不启动训练。纯读取。
#       唯一副作用：/tmp/_probe_triton_cache（Triton 编译缓存）
#
# 用法：bash probe_env_readonly.sh 2>&1 | tee /root/probe_env_$(date +%m%d_%H%M).log
#
# v2 修复（2026-09-16 新机器实测暴露，见 docs/PITFALLS 坑 37-40）：
#   坑 39: 原脚本用 `hostname` —— 极简镜像可能没有该命令（openEuler 容器实测 MISSING）
#          → 改用 /proc/sys/kernel/hostname，并内建"基础命令缺失清单"探测
#   坑 40: 原脚本按 version.cfg / ascend_toolkit_install.info 找 CANN 版本 —— 实际
#          CANN 9.1.0-beta.3 为 compiler/version.info + opp/version.info，
#          且 ascend-toolkit/latest 是指向 cann-<ver> 的**符号链接**
#   坑 38: 新增 python3 与 pip3 **是否同一解释器**的判定（不同则是静默陷阱）
#   坑 37: 新增包管理器探测（openEuler 无 apt，需 dnf）
# ============================================================================

echo "########## 0. 主机身份 ##########"
cat /proc/sys/kernel/hostname 2>/dev/null || echo "(无 hostname)"
uname -a 2>&1
grep -E '^(PRETTY_NAME|VERSION)=' /etc/os-release 2>/dev/null | head -2
echo "nproc=$(nproc 2>/dev/null)  mem_gb=$(awk '/MemTotal/{printf "%.0f",$2/1048576}' /proc/meminfo 2>/dev/null)"
echo

echo "########## 0b. 基础命令可用性（极简镜像可能缺） ##########"
for c in hostname unzip tar git curl wget gcc g++ make cmake python3 pip3 basename nproc awk sed grep find; do
  p=$(command -v $c 2>/dev/null); printf '  %-10s %s\n' "$c" "${p:-MISSING}"
done
echo "  包管理器: $(for m in dnf yum microdnf apt apt-get; do command -v $m 2>/dev/null; done | tr '\n' ' ')"
echo

echo "########## 1. 芯片身份（★ 决定拓扑与能否凑 GBS=8） ##########"
echo "--- npu-smi info -l ---"
npu-smi info -l 2>&1 | head -50
echo "--- npu-smi info ---"
npu-smi info 2>&1 | head -40
echo "--- /dev/davinci* ---"
ls -1 /dev/davinci* 2>&1 | head -20
echo

echo "########## 2. 驱动 / CANN 版本（★ 坑 40：路径名不固定） ##########"
echo "--- driver ---"
cat /usr/local/Ascend/driver/version.info 2>&1 | head -8
echo "--- ascend-toolkit/latest 指向 ---"
readlink -f /usr/local/Ascend/ascend-toolkit/latest 2>&1
echo "--- CANN 版本文件（多路径尝试） ---"
for f in /usr/local/Ascend/ascend-toolkit/latest/version.cfg \
         /usr/local/Ascend/ascend-toolkit/latest/ascend_toolkit_install.info \
         /usr/local/Ascend/ascend-toolkit/latest/compiler/version.info \
         /usr/local/Ascend/ascend-toolkit/latest/opp/version.info \
         /usr/local/Ascend/cann/compiler/version.info \
         /usr/local/Ascend/cann/opp/version.info; do
  [ -f "$f" ] && { echo "[$f]"; head -6 "$f"; }
done
echo "--- /usr/local/Ascend 下所有条目 ---"
ls -1d /usr/local/Ascend/* 2>&1 | head -15
echo "--- set_env.sh ---"
ls -la /usr/local/Ascend/ascend-toolkit/set_env.sh /usr/local/Ascend/set_env.sh 2>&1
echo

echo "########## 3. Python 软件栈（★ 坑 38：python3 与 pip3 可能不是同一解释器） ##########"
echo "--- python3 与 pip3 解析 ---"
echo "  python3 -> $(readlink -f "$(command -v python3 2>/dev/null)" 2>/dev/null)"
echo "  pip3    -> $(readlink -f "$(command -v pip3 2>/dev/null)" 2>/dev/null)"
python3 -V 2>&1
python3 -m pip -V 2>&1 | head -1
echo "--- 系统内所有 python 解释器 ---"
ls -1d /usr/local/python* /usr/bin/python3.* 2>/dev/null | head -10
for p in /usr/bin/python3 /usr/local/python3.12.13/bin/python3; do
  [ -x "$p" ] && echo "  $p -> $("$p" -V 2>&1)  pip=$("$p" -m pip -V >/dev/null 2>&1 && echo yes || echo no)"
done
echo "--- 关键包 ---"
python3 -m pip list 2>/dev/null | grep -iE 'torch|triton|transformers|mindspeed|numpy|pyyaml|pillow|safetensors|accelerate|deepspeed' || echo "  (pip 不可用或无包)"
echo "--- torch / torch_npu ---"
source /usr/local/Ascend/ascend-toolkit/set_env.sh 2>/dev/null
python3 - <<'PYEOF' 2>&1
try:
    import torch, torch_npu
    print("torch       :", torch.__version__)
    print("torch_npu   :", torch_npu.__version__)
    print("npu available:", torch.npu.is_available())
    n = torch.npu.device_count()
    print("device_count:", n)
    for i in range(n):
        print("  dev[%d] name: %s" % (i, torch.npu.get_device_name(i)))
    for i in range(n):
        try:
            free, total = torch.npu.mem_get_info(i)
            print("  dev[%d] mem : %.1f / %.1f GB free" % (i, free / 1024**3, total / 1024**3))
        except Exception as e:
            print("  dev[%d] mem_get_info failed: %s" % (i, e))
except Exception as e:
    print("!! import torch/torch_npu FAILED:", type(e).__name__, e)
PYEOF
echo "--- triton 家族 ---"
python3 -c "import triton; print('triton      :', triton.__version__)" 2>&1 | tail -1
# ★ 坑 49：`import triton_ascend` 是错的 —— triton-ascend 装完模块名就是 `triton`，
#   只是额外提供 ascend 后端。正确判据 = triton.backends 里是否含 'ascend'。
python3 -c "from triton.backends import backends; print('triton backends:', list(backends.keys()))" 2>&1 | tail -1
echo "--- 权威可用性判据（规范写法 tl.program_id） ---"
if [ -f "$(dirname "$0")/00b_triton_min_kernel.py" ]; then
  python3 "$(dirname "$0")/00b_triton_min_kernel.py" 2>&1 | tail -2
else
  python3 - <<'PYEOF' 2>&1 | tail -2
import torch, torch_npu, triton
import triton.language as tl
@triton.jit
def _k(X, Y, N, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    m = offs < N
    tl.store(Y + offs, tl.load(X + offs, mask=m, other=0.0) * 2.0, mask=m)
N = 1024
x = torch.randn(N).npu(); y = torch.empty_like(x)
_k[(triton.cdiv(N, 256),)](x, y, N, BLOCK=256)
torch.npu.synchronize()
print("TRITON_NPU_OK" if bool(torch.allclose(y.cpu(), x.cpu() * 2)) else "TRITON_WRONG_RESULT")
PYEOF
fi
echo "--- transformers / pyyaml / mindspeed ---"
python3 -c "import transformers; print('transformers:', transformers.__version__)" 2>&1 | tail -1
python3 -c "import yaml; print('pyyaml:', yaml.__version__)" 2>&1 | tail -1
python3 -c "import mindspeed; print('mindspeed:', getattr(mindspeed,'__version__','?'))" 2>&1 | tail -1
echo

echo "########## 4. 关键前置依赖（坑 22 判据：缺则 triton 静默回退 CPU） ##########"
INC=$(python3 -c 'import sysconfig;print(sysconfig.get_paths()["include"])' 2>/dev/null)
echo "  include dir: ${INC:-?}"
if [ -n "$INC" ]; then
  if [ -f "$INC/Python.h" ]; then echo "  Python.h: 存在 ✓"; else echo "  Python.h: **缺失** ✗（需装 python3-devel / python3-dev）"; fi
fi
echo "  编译器: $(gcc --version 2>/dev/null | head -1)"
echo "--- 源站可达性（注意 429 限流） ---"
for u in https://mirrors.huaweicloud.com/ascend/repos/pypi https://repo.huaweicloud.com/repository/pypi/simple https://pypi.org/simple/; do
  code=$(timeout 10 curl -s -o /dev/null -w '%{http_code}' --max-time 8 "$u" 2>/dev/null)
  echo "  $code  $u"
done
echo

echo "########## 5. 框架与数据是否已就位 ##########"
echo "--- MindSpeed-MM ---"
ls -d /root/MindSpeed-MM 2>&1
cat /root/MindSpeed-MM/version.txt 2>/dev/null | head -2
git -C /root/MindSpeed-MM describe --tags 2>/dev/null
ls /root/MindSpeed-MM/examples/ 2>/dev/null | head -10
echo "--- 权重 ---"
ls -d /root/Qwen3.5-0.8B-hf /root/Qwen3.5-0.8B-dcp 2>&1
echo "--- 数据 ---"
ls -la /root/data/ 2>&1 | head -10
echo "--- 已有产物 / 日志 ---"
ls -1 /root/*.log 2>/dev/null | head -10
echo

echo "########## 6. 资源 ##########"
free -g 2>&1 | head -3
df -h / /root 2>&1 | head -5
echo

echo "########## 7. 昇腾环境变量快照 ##########"
env | grep -iE 'ascend|npu|triton|task_queue|pytorch_npu|hccl' | sort 2>&1 | cut -c1-200
echo

echo "########## 8. triton 可用性最小判据（只编译一个 kernel，不训练） ##########"
python3 - <<'PYEOF' 2>&1
import os
os.environ.setdefault("TRITON_CACHE_DIR", "/tmp/_probe_triton_cache")
try:
    import torch, torch_npu, triton
    import triton.language as tl

    @triton.jit
    def _probe_kernel(X, Y, N, BLOCK: tl.constexpr):
        pid = tl.program_id(0)
        offs = pid * BLOCK + tl.arange(0, BLOCK)
        m = offs < N
        x = tl.load(X + offs, mask=m, other=0.0)
        tl.store(Y + offs, x * 2.0, mask=m)

    N = 1024
    x = torch.randn(N, dtype=torch.float32).npu()
    y = torch.empty_like(x)
    _probe_kernel[(triton.cdiv(N, 256),)](x, y, N, BLOCK=256)
    torch.npu.synchronize()
    ok = bool(torch.allclose(y.cpu(), x.cpu() * 2.0))
    print("TRITON_MINIMAL_KERNEL: %s" % ("PASS" if ok else "WRONG_RESULT"))
except Exception as e:
    print("TRITON_MINIMAL_KERNEL: FAIL ->", type(e).__name__, e)
PYEOF
echo

echo "########## 9. 判定提示 ##########"
echo "  A. chip_count >= 2 ?               -> 能否凑官方几何 nproc_per_node=2（GBS=8）"
echo "  B. torch_npu 可用且 device_count>=2 ?"
echo "  C. TRITON_MINIMAL_KERNEL: PASS ?   -> FAIL 则降级 triton->ascendc->eager（数值等价已验证）"
echo "  D. Python.h 存在 ?                 -> 缺则先装 python3-devel/python3-dev（坑22）"
echo "  E. python3 与 pip3 是否同一解释器 ? -> 不同则必须先锁定单一 PYBIN（坑38）"
echo "  F. 包管理器是 dnf 还是 apt ?       -> 决定装依赖的命令（坑37）"
echo "  G. MindSpeed-MM + 权重 + 数据是否就位 ? -> 否则走 Skill P4 资产准备"
echo "########## DONE ##########"
