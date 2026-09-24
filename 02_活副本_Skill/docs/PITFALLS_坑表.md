# Skill A3-Bringup · 昇腾新卡快速迁移 Runbook（v1.2 ✅ 实跑验证，含 dp2 官方几何）
> 一句话：一张全新昇腾卡 → torch_npu → MindSpeed-MM → mock/COCO 训练跑通 = 全自动脚本化（已实测通过）。
> 验证：A3(<A3训练服务器地址>, 1×Ascend910_9382 双 die 64GB×2, CANN 9.1.0, py3.10)；mock10/COCO100/dp2-100 全部完成。
> 纪律：每踩一坑立刻固化解法；目标 = Skill 平滑一键完成迁移。

## 0. 验收复现入口（官方口径 = 评委重跑复现, 2026-09-07 确认）
- **版本锁**：`config/versions.lock`（锁定最终验收锚点 A3 dp2 全量 100 步的全部版本/配置/env）
- **环境自检**：`python3 selfcheck_env.py` → SELFCHECK_OK（0-GPU，评委 5 分钟自证环境一致）
- **判定链**：skill_R2-SK04（selfcheck 24/24 + fingerprint/observed/judge，verdict_id=18ababd8e34a9403）

## 0a. 目标环境画像（本次实测）
- 镜像：quay.io/ascend/cann devel（ubuntu22.04, Python3.10, apt；实测无 pip/ensurepip）
- CANN：/usr/local/Ascend/{ascend-toolkit, cann-9.1.0, driver}；npu-smi 需 CANN LD 路径
- NPU：**1× Ascend910_9382 双 die**（npu-smi: Total=1 Chip=2, 2×64GB=128GB）；torch_npu 识别名 Ascend910_9382

## 1. 连接与只读探测（先 5 分钟摸清，别上来就装）
```bash
# ① 芯片/卡数（需 CANN lib 路径，否则报 libc_sec.so 缺失）：
export LD_LIBRARY_PATH=/usr/local/Ascend/driver/lib64:/usr/local/Ascend/driver/lib64/common:/usr/local/Ascend/driver/lib64/driver:/usr/local/Ascend/ascend-toolkit/latest/lib64:$LD_LIBRARY_PATH
/usr/local/bin/npu-smi info | head -22
# ② 设备节点数：ls /dev/davinci*（davinci6,7 = 2 卡）
# ③ python/pip：python3 --version；python3 -m pip --version（devel 镜像常无 pip！）
# ④ CANN 版本：ls /usr/local/Ascend/；cat /usr/local/Ascend/ascend-toolkit/latest/version.cfg
# ⑤ 网络：curl -sI https://repo.huaweicloud.com；git ls-remote --heads https://github.com/Ascend/MindSpeed-MM.git
```

## 2. pip 引导（坑 1-3）
```bash
# 坑1: devel 镜像无 pip 且无 ensurepip（No module named ensurepip）
# 坑2: bootstrap.pypa.io 可能超时挂死 → 全部命令必须带 timeout + 后台跑 + 轮询日志
python3 -m ensurepip --upgrade 2>&1 | tail -1            # 可能不存在，忽略
timeout 60 curl -sSL --max-time 55 https://bootstrap.pypa.io/get-pip.py -o /tmp/gp.py \
  && python3 /tmp/gp.py                                  # 实测成功：pip 26.2.1
# 坑3: 装系统级 vs venv：统一建议 venv（如 A2 的 /root/msmm_26），避免系统污染
python3 -m pip config set global.index-url https://repo.huaweicloud.com/repository/pypi/simple
python3 -m pip config set global.trusted-host repo.huaweicloud.com
```

## 3. torch + torch_npu（坑 4-6）
```bash
# 坑4: 单条 ssh 命令超 120s 会被掐 → 一律 setsid nohup 后台 + 日志轮询
# 坑5: torch_npu 导入链需要 pyyaml（No module named 'yaml'）
# 坑6: torch_npu profiler 模块需要 numpy（ModuleNotFoundError: numpy）
python3 -m pip install torch==2.7.1 torch_npu==2.7.1.post10 pyyaml numpy
# 探针（成败一锤定音）：
source /usr/local/Ascend/ascend-toolkit/set_env.sh 2>/dev/null
python3 - <<'PY'
import torch, torch_npu
print(torch.__version__, torch_npu.__version__)
print("devices:", torch.npu.device_count())
for i in range(torch.npu.device_count()):
    print(i, torch.npu.get_device_name(i))
print(bool((torch.randn(64,64,device="npu") @ torch.randn(64,64,device="npu")).isfinite().all()))
PY
# 判据：devices>0 + matmul ok = 栈兼容（A2 的 torch_npu 2.7.1.post10 在 9382 直接跑通）
```

## 4.（进行中）MindSpeed-MM / triton-ascend / 权重 / 训练-推理入口 / 场景 demo
- 待补：MSMM 源码版本获取（A2 rev 对齐）、triton-ascend 对 9382 支持验证、权重搬运、mock 10 步、COCO step1 对齐、SOP 场景推理

## 4.1 默认开发环境验收标准（必装齐，不留半截）
- [ ] python3.10 + pip（get-pip 引导，配 huaweicloud 主源）
- [ ] torch==2.7.1 + torch_npu==2.7.1.post10 + pyyaml + numpy（探针：devices≥1 + matmul ok）
- [ ] MindSpeed-MM **v26.1.0 tag**（勿用默认分支：其 pyproject 要求 torch==2.10.0，与 v26.1.0 栈不匹配）
- [ ] transformers==5.2.0（v26.1.0 配套；勿装 4.57 等其它版）
- [ ] triton-ascend（索引最高 **3.2.0**，装 3.2.0；**导入 `import triton`**，`from triton.backends import ascend` 验证；先装 pybind11）
- [ ] MSMM 其余运行依赖（qwen_vl_utils / einops / av / pandas 等，按 UserGuide/import 报错补全）
- [ ] 权重：Qwen3.5-0.8B hf（**modelscope** 下载优先，hf-mirror 401）+ dcp 双份
- [ ] 0.8B 训练配置：**v26.1.0 tag 不自带**（A2 本地创建）→ 从新分支 examples 移植或自 A2 cat
- [ ] 数据：COCO 2000 子集（annotations_slim.json）
- [ ] 健康检查：`import mindspeed_mm` 通过 + `python -c "import torch,transformers;print(torch.__version__,transformers.__version__)"`
- [ ] 验证入口：mock 10 步训练可启动（不等收敛，能出 iteration 即环境 OK）

## 5. 坑表速查
| # | 现象 | 解法 |
|---|---|---|
| 1 | No module named pip / ensurepip | curl get-pip.py（带 --max-time）+ apt python3-pip 兜底 |
| 2 | ssh 命令 120s 掐断 | setsid nohup 后台 + 日志轮询 |
| 3 | npu-smi: libc_sec.so 缺失 | export LD_LIBRARY_PATH 加 CANN driver/toolkit lib64 |
| 4 | import torch_npu: No module named 'yaml' | pip install pyyaml |
| 5 | import torch_npu: No module named 'numpy' | pip install numpy |
| 6 | token 5 分钟过期（channel denied） | 操作前先批量准备好脚本，token 一到连发 |
| 7 | `import torch` 报 Failed to load backend extension: torch_npu | **必须先** `source /usr/local/Ascend/ascend-toolkit/set_env.sh`（或 export CANN LD_LIBRARY_PATH）；任何 python 入口前强制 |
| 8 | github clone 间歇超时 | 重试 1-2 次；兜底 gitcode.com/ascend/MindSpeed-MM（同 tag） |
| 9 | triton-ascend 3.2.1 不在公共索引（pypi/huaweicloud 最高 3.2.0） | 装 3.2.0 即可；**导入名是 `triton`**（Ascend backend 替换版，非 `triton_ascend`）；验证：`import triton; from triton.backends import ascend` |
| 10 | `import triton` 报 No module named 'pybind11' | pip install pybind11 |
| 11 | hf-mirror 下载 401（CAS Unauthorized） | 换 **modelscope**：`snapshot_download('Qwen/Qwen3.5-0.8B', local_dir=...)` |
| 12 | 后台脚本在 ssh 内联 heredoc 里 setsid 起不来 | **先 cat > /root/xxx.sh 落盘 → chmod → setsid nohup bash /root/xxx.sh**，日志立即可见 |
| 13 | MSMM v26.1.0 tag 无 0.8B 示例配置（只有 4B~397B） | A2 上 0.8B 配置是本地创建 → 需移植（git show 新分支文件 / 自 A2 cat 文本 / 按 4B 模板改） |
| 14 | `import torch` 报 Failed to load backend extension: torch_npu（前台 shell 与子进程行为不一） | 每个脚本内部 `source set_env.sh`；验证用 `import torch_npu` 而非 grep LD 字符串 |
| 15 | 缺 `mindspeed` 模块（MSMM 依赖核心库） | `git clone https://gitcode.com/Ascend/MindSpeed.git`（github 不可达时 gitcode 稳）→ `pip install -e /root/MindSpeed --no-deps` |
| 16 | Megatron-LM clone 卡死（NVIDIA github 超时、timeout 杀不掉） | **fsdp trainer 路径不需要 megatron** → 跳过；勿硬等 |
| 17 | 依赖逐个冒头（pybind11/protobuf/scipy/sentencepiece/torchdata/torchvision/ftfy/diffusers） | 一次性批量 `pip install torchdata torchvision ftfy diffusers scipy sentencepiece protobuf pybind11`；按 pyproject 全量清单补，别一个一个试 |
| 18 | 后台长命令（clone/install/训练）用 `setsid nohup bash /root/x.sh > log` + 轮询日志；**git clone 加 timeout 也杀不掉时** pkill -9 -f | 进程级清理 |
| 19 | triton-ascend 回落 CPU 但报 npu_utils.cpp 编译失败 `Python.h: No such file` | **装 python3-dev + build-essential**：驱动首次要现场编译 C++ 桥接 |
| 20 | triton-ascend 索引分布：3.2.0=pypi/huaweicloud 主源；**3.2.1=osinfra**（triton-ascend.osinfra.cn，自带 base triton 3.5.0 注意混装）；3.2.2=huaweicloud ascend 源需 `--extra-index-url` 主源补 attrs==24.2.0 | 按需选源 |
| 21 | triton backend 诊断必须**落盘 .py 再跑**（heredoc stdin 会让 inspect.getsourcelines 失败）；最小 kernel 判据 = 打印 `TRITON_NPU_OK` | 先于 MSMM 判定排查 backend 是否支持当前 SoC |
| 22 | ✅ **9382(A3) triton 修复链（实证 2026-09-07）**：① 装 python3-dev+build-essential ② 清场卸载 triton/triton-ascend/pybind11 ③ 装 **triton-ascend==3.2.2**（`--index-url https://mirrors.huaweicloud.com/ascend/repos/pypi --extra-index-url https://repo.huaweicloud.com/repository/pypi/simple` 补 attrs==24.2.0）④ 最小 kernel 测 `TRITON_NPU_OK` ⑤ MSMM mock gdn=triton：rollback=0 + 'use NPU triton ops' | CANN 9.1.0+torch 2.7.1+py3.10 官方配对=**3.2.2**（A2 的 3.2.1 是 910B 容忍偏离）；CANN kernels 需含 `ascend910_93`（9382）|
| 23 | 新几何/新卡**首跑 median 会被内核编译慢步污染**（10-26s 步散布 iter3-15） | 取稳态必须**热缓存重跑一遍**再算 median（A2 asc100b / A3 m4g2b 同法）；跨卡对比时两边都取热缓存稳态 |
| 24 | ⚠️ 本机无 msmm_26 venv 时脚本里别写死 `/root/msmm_26/bin/python` | 用 `python3`（A3 是系统 python 安装） |
| 25 | ⚠️ **硬件身份先验证再定策略**：`npu-smi info -l` 看 Total Count vs Chip Count；9382 双 die 卡呈现 1 卡 2 chip（board SN 一致）→ 与官方 `NPUS_PER_NODE=2` 同拓扑 | 先用 npu-smi 确认再选几何；双 die 应跑 `torchrun --nproc_per_node 2` 官方几何而非单 die（m8g1 只用一半硬件） |
| 26 | dp2/双 rank 启动失败 `No module named 'megatron'` | **必须 `export NON_MEGATRON=true`**（mindspeed_mm/__init__ 守卫）；全套 env：`NON_MEGATRON=true TASK_QUEUE_ENABLE=2 ASCEND_LAUNCH_BLOCKING=0 PYTORCH_NPU_ALLOC_CONF=expandable_segments:True TRITON_CACHE_DIR=/root/triton_cache` + `source set_env.sh`（缺则 libhccl.so 找不到） |
| 27 | 日志 Configuration Details 是 repr（`None` 是字符串）→ 直接当 yaml 会 pydantic ValidationError | 配置改动必须**用真 yaml sed**（如 `cp geo_m8g1.yaml && sed -i 's/micro_batch_size: 8/micro_batch_size: 4/'`），勿从日志回抽 |
| 28 | dp2 配置要点：`data_parallel_size/fully_shard 1→2`（fsdp auto 解析）、mbs4/gas1 保 GBS=8、num_workers 2→8、preprocessing_num_workers→16（对齐官方，免费消 fingerprint 偏离维） | 实测 median 531ms（vs 单 die m8g1 752ms）；`torchrun --nproc_per_node 2 --master_port <新端口>`（勿与旧进程撞） |
| 29 | 全量数据链：llava json 走 `modelscope download --dataset AI-ModelScope/LLaVA-Instruct-150K`（229MB）；COCO 走 `PAI/COCO2017 train2017.zip`（19.3GB）；转换 `llava_instruct_2_mllm_demo_format.py` → 157,712 样本 | 下载必须 `timeout + setsid nohup + 日志轮询`（坑 18）；llava 字节 228,941,895=Content-Length 精确一致；转换脚本只接受 json 路径（大 json 需内存 ~2GB+） |

## 7. 性能与几何要点（A3 实测，2026-09-07）
- **A3(9382) vs A2(910B4) 同配置 m2g4/triton 100 步**：loss Δ max 0.196% / mean 0.059%；grad_norm Δ max 0.60%；median **1695ms vs 3050ms → 9382 快 1.79×**（文档 docs/a3_crosscard_20260907.md）
- **硬件身份**：A3 = 1×910C 双 die（npu-smi: Total=1 Chip=2, 同 Board）→ 官方几何 = `--nproc_per_node 2`（坑 25）
- **几何阶梯（GBS=8）**：单 die m8g1=752ms → **双 die dp2(m4g1)=531ms**（官方几何, 12.19 samples/s）；全链 eager 21.4s → **0.53s ≈ 40×**
- **官方几何 dp2 = 唯一逐点可比路径**：mbs/gas/dp 三维 fingerprint 归零；step1 loss 逐位对齐官方（1.924621）；
  全量 157k 数据后 mismatch=0 → pointwise_feasible（verdict_id=18ababd8e34a9403，见 docs/a3_dp2_full_anchor_20260907.md）
- **跨卡数值等价三重复证**：eager≡triton≡ascendc(A2) + 910B4≡9382(A3)，Δ<0.2%

## 产物
- 脚本存档（本次实跑）：本地 a3_setup1.sh / a3_pip2/3.sh / a3_step2.sh / a3_probe*.sh / ssh_gen.py
- 本文件：code/skills/skill_A3_bringup/README.md

---

# 坑 30-40 —— 干净环境（新机器）暴露的缺口（2026-09-16 实测）

> **背景**：换到一台**全新机器**（openEuler 24.03 LTS-SP3 / aarch64 / 容器 / CANN 9.1.0-beta.3 /
> driver 25.5.1 / Python 3.11.6 / 只有 dnf 无 apt / **1×Ascend910 双 die 64GB×2**），
> 原机器数据全无。用 `scripts/run_from_zero.sh` 验证 Skill 能否独立完成任务 → 一次性暴露下列缺口。
>
> **这组坑的共同特征**：它们**全部源自"在旧机器上边踩边搭"留下的隐含依赖**，
> 在旧环境里永远不会暴露（路径恰好对、解释器恰好唯一、包管理器恰好是 apt）。

| # | 现象 | 根因 | 处置（已落地） | 判据 |
|---|---|---|---|---|
| **30** | `selfcheck_sk04.py` 直接 `IndexError: 3` 崩溃 | `SKILL_ROOT.parents[3]` 硬编码路径深度（假定 `code/skills/R2/skill_R2-SK04`）；Skill 被打包到 `/root/qwen35-ascend-migrator` 时层级不足 | `selfcheck_sk04.py` / `extract_postA_log.py` 均新增 `find_repo_root()`：env `SK04_REPO_ROOT` → 逐级向上找 `snapshots/` 或 `docs/official/` 标记目录 → **找不到返回 None，绝不抛异常** | `python3 sk04_judge/scripts/selfcheck_sk04.py` 不再崩溃 |
| **31** | `extract_postA_log.py --check` 在分发包内必然失败 | 需先读官方帖 JSON（`docs/raw/hiascend_post_*.json`），该文件**不在 Skill 包内**；且 `--check` 分支被放在 post 校验之后 | 重构：`--check` 提前；post 缺失时输出 **`CHECK_PARTIAL_NO_POST`**（rc=0）并显式声明**"未做重导出一致性比对"**。自检中该检查标 **SKIP** 而非 FAIL —— 不冒充实证 | 输出含 `CHECK_PARTIAL_NO_POST` 且带 note |
| **32** | 把 `None` 当路径传给子脚本 | `bcsv`/`acsv` 在无仓库时为 None，`run_py` 会 `str(None)` → 传了字符串 `"None"` | 参数仅在路径真实存在时才追加 | `fingerprint_observed_Alog` 在无仓库时仍 PASS |
| **33** | LR 序列一致性检查**空过**（vacuous pass） | 原条件 `len(hashes) <= 1`：只有 1 份产物时必然为真 → 假阳性 | 改为要求 **≥2 份产物**才判定；不足则 SKIP 并注明"空过不算通过" | 无仓库时显示 `SKIP 仅 1 份产物` |
| **34** | 干净机器上"全通过"却返回 1 | 汇总用 `failed = total - passed`，**把 SKIP 也算成失败** → 信号失真 | `failed = total - passed - skipped`，输出增加 `failed=` 字段 | `case_ok=N/N skipped=M failed=0` |
| **35** | ★★ **自检消费 13 天前的陈旧产物仍判 PASS** | `sk04_judge/tests/tmp/` 里带着开发机 2026-09-03 的 `fp_self_officialB.json` / `observed_self_coco.json`；`pack_skill.py` 只清 `__pycache__` 与 `out/`，**没清 `tests/tmp/`**。新机器产不出这些文件（无 `coco_train_run1.log`）却照样"通过" | ① `selfcheck_sk04.py` 开跑前**清空 `tests/tmp/*.json`** 并打印清空数量 ② 之后的判定只允许消费本次运行产物，缺输入 → SKIP | 运行时打印"已清空 N 个上次运行的产物"；相关检查转为 SKIP |
| **36** | 新增的示例重放检查永远 SKIP | `SKILL_ROOT` 指向 `sk04_judge/`，而 `examples/` 在**上一层**（Skill 根）→ 路径永远找不到 | 新增 `PKG_ROOT = SKILL_ROOT.parent if (SKILL_ROOT.parent/"SKILL.md").is_file() else SKILL_ROOT` | `replay_example_*` 三项 PASS |
| **37** | `apt install python3-dev build-essential` 在 openEuler 上直接失败 | 硬编码 apt；openEuler 原生是 dnf/yum，包名为 `python3-devel` / `gcc` / `gcc-c++` | `bringup.sh` 新增包管理器探测（dnf→yum→apt），按管理器分派命令 | 打印 `包管理器 = dnf` |
| **38** | ★★ **`pip3 install X` 成功但 `python3 -c "import X"` 失败** | 新机器 `python3` = `/usr/bin/python3.11`（**无 pip**），而 `pip3` = `/usr/local/python3.12.13/bin/pip3` —— **两个不同解释器** | `bringup.sh` 新增 `choose_python()`：枚举候选解释器 → 逐个保障 pip 与 `Python.h` → **锁定单一 `PYBIN`**，后续全程只用它 | 打印 `锁定 PYBIN=/usr/bin/python3 (py3.11)`，且 `PYBIN -c "import torch_npu"` 成立 |
| **39** | 探测脚本第 9 行报 `hostname: command not found` | 极简容器镜像缺基础命令，脚本却假设 `hostname` 存在 | 改用 `cat /proc/sys/kernel/hostname`；并新增"基础命令缺失清单"探测段 | `hostname = MISSING` 被记录而非报错 |
| **40** | CANN 版本读不到 | 脚本按 `version.cfg` / `ascend_toolkit_install.info` 找，而 CANN 9.1.0-beta.3 实际是 `compiler/version.info` + `opp/version.info`；且 `ascend-toolkit/latest` 是**指向 `cann-<ver>` 的符号链接** | 探测脚本改为多路径尝试 + `readlink -f latest`；`bringup.sh` 兼容 `ascend-toolkit` 与 `cann` 两种根；`selfcheck_env.py` 正则改为 `cann-(\S+)`（原 `[\d.]+` 会把 `-beta.3` 截掉） | 探测输出含 `readlink` 目标与版本文件内容 |
| **41** | 新机器上 `SELFCHECK_FAIL`，但分不清"环境不对"还是"环境不同但可用" | `selfcheck_env.py` 只有一种模式：所有项严格等于 `versions.lock`。换机器（py3.11.6 vs 3.10.12、driver 25.5.1 vs 25.5.0）必然 FAIL → 信号失真 | 拆两种模式：**`--mode=acceptance`**（必须就是 lock 描述的那台验收环境，供评委复现）与 **`--mode=portability`**（报告 drift，**只对"栈内部不自洽"判失败**：torch 与 torch_npu 主次版本不配对、Python/CANN 超出支持范围）。`selfcheck.py` 的 L3 **同时跑两种**：portability 作通过判据，acceptance 作信息性报告并明确提示"不得据此宣称与验收环境一致" | `SELFCHECK_PORTABILITY_OK`；L3 输出 `PASS 栈内部自洽` + `INFO 本机不是验收环境` |
| **42** | 判定链"通过"了但其实是假阳性 | `selfcheck.py:check_l2` 判据 `if "VERIFY_OK" in out or r.returncode == 0` —— `VERIFY_OK` 那行**无论成败都会打印**，故 `or` 恒真 → **永远 PASS** | 改为要求 `rc == 0` **且** 解析出的 `failed=0`。与坑 33/34 同族：**判据必须绑定可证伪的量化字段，不能 grep 一个恒存在的标记** | `check_l2` 显式打印 `rc=.. failed=..` |
| **43** | 探针还缺一条：CANN 预发布版（`-beta`/`-rc`）的配套需单独提示 | GA 与 beta 的 torch_npu 配套可能不同，容易被当成 GA 处理 | `selfcheck_env.py` 新增 `cann_prerelease` 提示行；`env_matrix.yaml` 新增 `9.1.0-beta.3` 组合行并标 `verified: false` | 输出 `NOTE 预发布版 CANN` |
| **44** | SSH 里 `setsid nohup bash bringup.sh &` 之后**启动命令挂死**（实测 95s 超时） | `bringup.sh` 用 `exec > >(tee -a "$LOG") 2>&1` 做日志 —— 进程替换产生 tee 子进程，它**持有原始 stdout 写端（即 SSH 通道）**，通道永不关闭 → paramiko `recv_exit_status` 阻塞。启动命令里也缺 `</dev/null` | ① `bringup.sh` 改为**仅 TTY 时用 tee**，非交互（SSH 后台/重定向）直接 `exec >> "$LOG" 2>&1` ② 所有后台启动统一 `setsid nohup cmd </dev/null >log 2>&1 &`（三 fd 全重定向） | `ssh_tool.py bg` / `step3` 均返回 `LAUNCHED`，不再超时 |
| **45** | 清了 `tests/tmp` 后判定链自检**崩溃** `FileNotFoundError: fp_yaml_coco.json` | `[1b]` 段的 `fingerprint_cfg` 需要 **pyyaml** 解析 yaml fixture；分发环境没装 pyyaml → 脚本不产出文件 → 后续 `json.loads(...read_text())` 抛未捕获异常 → **整个判定链自检挂掉**。即"判定链有未声明的 pyyaml 依赖" | ① 显式探测 pyyaml，缺失时三项标 **SKIP** 并给出装法 ② 整段加 `try/except` 兜底，绝不因 fixture 问题崩溃 ③ `bringup.sh` phase1 已把 pyyaml 列入一起装 | 无 pyyaml 时输出 `SKIP 缺 pyyaml`，`failed=0` 不再崩溃 |
| **46** | ★★★ **Skill 误判"triton 不可用"并自动降级 → 白扔 51× 性能路径** | `bringup.sh` 的最小 kernel 判据写的是 `triton.program_id(0)` —— **该属性在 JIT 命名空间不存在**（正确写法 `tl.program_id(0)`），抛 `AttributeError` → 判据**假阴性**。实测同一台机器上规范写法 `TRITON_NPU_OK` 通过 | ① 判据独立成 `scripts/00b_triton_min_kernel.py`（坑 21 教训：必须落盘），用规范写法 ② 版本策略改为**先实测再决定装不装**：能用就用（记录实际版本），不通才按配对装 3.2.2 复测 —— 因为配对关系**取决于 CANN 版本**（实测 CANN 9.1.0-beta.3 自带 triton-ascend **3.2.0** 且 kernel 通过；而 A3 的 CANN 9.1.0 GA 上 3.2.0/3.2.1 会回退 CPU） | 输出 `TRITON_NPU_OK backend=ascend triton=<ver>` |
| **47** | 资产一个都没有，却打印 `ASSETS_OK checks=0 fails=0` 并返回 0 | `40_prepare_assets.py` 只按 `checks` 判成败；`--no-download` 时不产生任何 check → **空集恒真**（假阳性，与坑 33/42 同族） | 改为**先按必需资产的 `exists` 判完整性**（`weight_hf`/`llava_json`/`converted_json`），再叠加 check 失败；缺失时输出 `ASSETS_INCOMPLETE missing_required=...` 并返回 3 | `--no-download` 下返回 3 + `ASSETS_INCOMPLETE` |
| **48** | phase4 mock 冒烟**永远被跳过**（"缺 mock 配置"） | 代码假定 `/root/MindSpeed-MM/examples/qwen3_5/qwen3_5_0_8B_mock_config.yaml` 存在 —— 该文件是开发期自己放的，**从未随 Skill 打包**；全新 clone 的 MSMM 里没有 | ① 把 mock / 最终几何 / m4g1-dp2 三个配置补进 `config/templates/` ② phase4 从 Skill 模板 **provision** 到 MSMM examples（幂等，已存在不覆盖） | phase4 打印 `已 provision mock 配置` 并真正执行 |
| **49** | 探针把可用的 triton 报成缺失 | 探针写 `import triton_ascend` —— **该模块名不存在**；triton-ascend 装完模块名就是 `triton`，只额外提供 ascend 后端 | 改用 `from triton.backends import backends` 判是否含 `ascend`；并调用 `00b_triton_min_kernel.py` 作权威判据 | 输出 `triton backends: ['ascend']` |
| **50** | `torch 2.7.1+cpu` 与 lock 的 `2.7.1` 被判 FAIL | 版本字符串带 **local version 后缀**（`+cpu`）；aarch64 上 pypi 的 torch 就是 CPU 构建，torch_npu 负责接 NPU —— 这是**正常且预期**的 | 比对时**剥离 `+xxx` local 后缀**再比；`+cpu` 不算 drift | `torch PASS lock=2.7.1 actual=2.7.1+cpu` |

---

## 坑 51-55 —— 鲁棒性加固阶段（2026-09-16，由"希望具备更好的鲁棒性"驱动）

> 前置认知：坑 30-50 共 **20 个缺口**不是 20 个独立 bug，而是**六类缺陷的 20 次显形**：
> 隐含环境假设 · **判据恒真** · **消费陈旧产物** · **判据自身写错** · 静默降级 · 中断不可续。
> 因此本阶段的产物不是"补更多断言"，而是 `docs/ROBUSTNESS.md` 里那 **7 条可检验不变量**。

| # | 现象 | 根因 | 处置（已落地） | 判据 |
|---|---|---|---|---|
| **51** | 同一类缺陷（恒真/空集/陈旧产物/判据写错）**反复以新形式出现** | 只做"发现一个修一个"是**反应式**的；没有任何机制检验"判据本身能不能失败" | 新增 `scripts/90_selfcheck_gates.py`：对每个关键判据做**双向负向对照**（good 必过 / bad 必挂），bad 也过则判 `GATE_VACUOUS`；另设 8 条静态回归守卫，每条**自带可证伪性自证**（正则必须能匹配合成坏样本，否则 `GUARD_BROKEN`） | `python3 scripts/90_selfcheck_gates.py` → `GATES_OK` |
| **52** | `selfcheck.py` 的 L1 **只检查 `.py`，从不检查 `.sh`** | 而 bash 脚本是流程入口（`bringup.sh`/`run_from_zero.sh`），一个语法错整条链跑不动；且编写环境可能没有 bash → 本地无从验证 | ① L1 加入 `bash -n` 闸门 ② `_find_bash()` **实跑 `bash -c 'echo BASH_OK'`** 验证可用性 ③ 无可用 bash 时标 **SKIP**（不静默当通过） ④ `run_from_zero.sh` 内建同一闸门作为远端保证 | `PASS 3 个 bash 脚本语法通过（bash -n @ D:\Git\bin\bash.exe）` |
| **53** | `which bash` 找到了 bash，跑起来却 `execvpe(/bin/bash) failed` | Windows 上 `C:\WINDOWS\system32\bash.exe` 是 **WSL 转发器**，但没有已安装的发行版 → **"找得到" ≠ "能用"** | `_find_bash()` 逐个候选**实跑一次**验证，取第一个真能输出 `BASH_OK` 的；找不到就 SKIP 并说明缘由 | 自检打印实际使用的 bash 路径 |
| **54** | 加入 bash 检查后**自检自己崩了**：`UnicodeDecodeError: 'utf-8' codec can't decode byte 0xc0` | `subprocess.run(text=True)` 在 Windows 用**本地编码(GBK)**解码，而 bash 输出是 UTF-8（含中文路径）→ reader 线程异常直接崩掉自检 | 所有跨进程调用显式 `encoding="utf-8", errors="replace"`，并统一封装 `_sh_run()` | 自检不再崩；输出正常显示中文路径 |
| **55** | 修 `90_selfcheck_gates.py` 时，**它自己犯了坑 34 的错**：汇总用 `status != "PASS"` → 把 SKIP 计为失败 → 返回 1 | 与坑 34 完全同族：**把"因平台限制没检查"当成失败**，淹没真实信号 | 汇总改为 `bad = status not in ("PASS","SKIP")`，并把 SKIP **单独打印**（既不静默当通过，也不冒充失败） | 输出 `合计 16 项：PASS=15 SKIP=1` + `GATES_OK` |
| **56** | ★ **明明 triton 可用，能力矩阵却判 `triton_ascend_kernel = DEGRADED`** | `05_preflight.py` 把 `stdout+stderr` 串起来后**取最后一行**判断成败。CANN 的 `NPUCachingAllocator` Warning 走 stderr，串接后落在末行 → 取到 Warning → **误判降级**。与坑 42/46 同族：**判据依赖脆弱的输出位置，而不是可证伪的显式标记** | 改为 **grep 显式标记** `TRITON_NPU_OK` / `TRITON_NPU_FAIL`，并区分三态（OK / DEGRADED / 未见标记）；未见标记时输出 `rc` 与末行以供诊断 | 真机 16/16 判据通过；preflight 应显示 `OK triton_ascend_kernel ... TRITON_NPU_OK` |
| **57** | 推送后启动后台下载，客户端**挂死 `TimeoutError`**（100s） | `ssh_tool.exec_cmd` 用 `out.read()` → **等通道 EOF**。而通道写端可能被**孙进程持有**（后台进程的子孙），前台命令早结束了通道也不 EOF。与坑 44 同族：**不要拿"通道关闭"当完成信号** | 改为**哨兵协议**：命令包进子 shell，末尾 `echo <哨兵> rc=$?`，读到哨兵即返回并主动关通道；stderr 暂存到**本地生成的字面文件名**（不能用 `$$`——远端新 shell 会展开成另一个 PID，随后 cat 不到） | 后台启动命令能在数秒内返回，不再挂死 |
| **58** | 真机跑资产下载**第一秒就崩**：`NameError: name 'time' is not defined` | 我新写的鲁棒下载代码用了 `time.sleep()` 做退避，却**忘了 `import time`** —— **鲁棒性代码本身引入了 bug** | 补 `import time`；并把"新增代码必须先做模块级加载自检"加入习惯（本次通过 `importlib` 加载模块 + 检查函数存在性验证） | `python3 -c "load module"` 正常；下载不再崩 |
| **59** | `run_from_zero.sh` 真机运行报 **`line 1: ﻿#!/bin/bash: No such file or directory`** | 我用 Windows PowerShell 的 `Set-Content -Encoding UTF8` 改过该文件 → **写入 UTF-8 BOM** → 首行变成 `﻿#!/bin/bash`（不可识别的解释器）。**`bash -n` 查不出来**（语法没错），只有真跑才炸 | ① 剥除全部 BOM（`.sh/.py/.md`）② `selfcheck.py` L1 新增**独立的 BOM 闸门**（扫 `.sh/.py/.yaml`）③ `pack_skill.py` 打包前统一剥 BOM，避免把这类隐性缺陷发给评委 | 自检输出 `PASS 无 BOM 污染`；脚本首行为纯 `#!/bin/bash` |
| **60** | ★★ **triton 判据仍然失败**：`TRITON_NPU_FAIL reason=CompilationError: NameError('tl is not defined')` —— 而独立测试明明是 `TRITON_NPU_OK` | 我把 `import triton.language as tl` 与 `@triton.jit` 的 kernel **都定义在 `main()` 函数内部**。`@triton.jit` 编译时按**模块全局命名空间**(`fn.__globals__`)解析名字，函数局部的 `tl` 不在其中 → `NameError`。**坑 46 换了个形式又犯一次**，后果同样是"误判不可用 → 降级 → 丢性能" | 把 `torch/triton/tl` 的导入与 `_probe_kernel` **全部移到模块顶层**；并新增 `--selfcheck` **结构自检**守住这一类 | `python3 scripts/00b_triton_min_kernel.py --selfcheck` → `TRITON_PROBE_SELFCHECK_OK` |
| **61** | 真机上 `/root/Qwen3.5-0.8B-hf` 是**空目录**，却被判 `weight_hf ok`，完整性门放行 | `record()`/完整性判定只用 `os.path.exists()` —— **路径存在 ≠ 内容就绪**。空目录、0 字节文件都算"就绪" | 新增 `substantive()` **内容级判定**：目录非空且含 `.safetensors/.bin/.json`；llava 字节数 == 锚点；converted 样本数 == 锚点；COCO 图片数 == 锚点。并逐项打印 OK/MISS 与原因 | 空目录 → `weight_hf MISS 空目录（★ 空目录不算就绪）` |
| **62** | ★ 我为坑 46/60 写的**结构自检自己产生 7 条假阳性** | 第一版用"行首是否缩进"判断作用域 → 把**模块顶层 `try:` 块里缩进的 import** 误判为"在函数内"。**缩进 ≠ 作用域**，文本启发式不能代替语义 | 改用 **`ast` 解析**：展开模块顶层（穿过 `try/if/with`，不进入函数/类）后判断 import 与 `@triton.jit` 的作用域；`triton.program_id` 用 AST 属性访问检测（不看字符串/注释）。**与坑 42/56 同族：不要用脆弱文本启发式代替语义判断** | 好样本 PASS；坏样本（函数内 jit + `triton.program_id`）FAIL —— 双向可证伪 |

---

## 坑 63-66 —— 流水线首跑（`run_from_zero` 全阶段）暴露的缺陷（2026-09-16 真机）

> 这是 `run_from_zero.sh` 第一次真正跑到 P7。裁决表：
> ```
> PRE PASS 24s | P0 PASS 36s | P1 PASS 0s | P2 PASS 0s | P3 FAIL 6s
> P4 PASS 0s(★假阳性) | P5 FAIL 20s | P6 FAIL 6s | P7 FAIL 0s
> ```
> **P3 与 P6 从未成功运行过**；P4 的"通过"是假的。

| # | 现象 | 根因 | 处置（已落地） | 判据 |
|---|---|---|---|---|
| **63** | P3 阶段 `FAIL`，日志只有一行 `30_verify_ops.py: error: unrecognized arguments: --env out/probe/env.json` | **driver 与脚本接口不一致**：`run_from_zero.sh` 按统一风格给每阶段传 `--env`，而 `30_verify_ops.py` 只接受 `--out` → argparse 直接退出。**P3 阶段从未真正运行过**（且被判成普通 FAIL，掩盖了"接口错"这一真相） | ① `30_verify_ops.py` 显式接受 `--env`（并记录进产物便于溯源）② 新增 prepush **G11 闸门**：把 driver 每个阶段命令里的 `--flag` 与该脚本的 `--help`/`add_argument` 逐个比对 | `G11 driver 接口 PASS（校验 9 个阶段命令）` |
| **64** | P6 阶段 `FAIL`：`ModuleNotFoundError: No module named 'verify_ops'` | 流水线脚本用**数字前缀**命名（`30_verify_ops.py`）以便人读阶段顺序，但 **Python 无法 `import 30_verify_ops`**（标识符不能以数字开头）→ `60_bench.py` 的 `from verify_ops import probe_backend` 必然失败。**P6 从未成功运行过** | 新增 `scripts/verify_ops.py` **shim**：按路径加载 `30_verify_ops.py` 并转发公开名字（保留编号命名的可读性 + 让导入可用）；shim 里显式校验 `probe_backend` 存在，缺失即提前报错。另加 prepush **G12** 闸门扫"跨脚本导入是否可解析" | `G12 跨脚本导入 PASS` |
| **65** | ★ **P4 记为 PASS，但资产明明全缺** | driver 的 P4 判据是 `[ -s out/assets/assets.json ]` —— 而 `40_prepare_assets.py` **无论成败都会写这个文件**（内容写着 `ASSETS_INCOMPLETE`）。同理 P7 的判据只看 `verdict.json` 存在，而空壳产物里 `verdict=None`。**判据弱于它要回答的问题** | 判据一律改为**语义标记**：P4 要求日志出现 `ASSETS_OK`；P7 要求 `verdict` 与 `verdict_id` 均非 None；P3 额外排除 argparse 接口错 | 资产缺失时 P4 正确记 `FAIL` |
| **66** | 新加的 G3/G11 闸门**自己制造假阳性**：把 `30_verify_ops.py` 误判为"不接受 `--env`"；把 `refs/` 里的参考模型源码与 `00b_triton_min_kernel.py` 的**设计好的优雅失败**判为"崩溃" | ① 这些脚本 import torch，**本机没装 → `--help` 拿不到输出** → 所有 flag 都"缺失"（把"本机无法验证"当成了"接口不匹配"）② `compileall` **只查语法**，查不出 `NameError` —— 于是"漏写 import"反复溜到真机（本次我自己又在 `30_verify_ops.py` 和 `prepush_check.py` 各犯一次） | ① G11 在"缺环境依赖"时退回**静态 `add_argument` 扫描**（接口声明不需要 torch）② G3 升级为「编译 + `--help` 加载冒烟」，并区分 `NameError`(FAIL) / 缺环境依赖(SKIP) / 其他 Traceback(FAIL)；扫描范围排除 `refs/` | `G3` 报告"跳过 3 个（缺依赖），其余无异常"；`G11` 报告"9 个阶段命令参数全部匹配" |

### 坑 63-66 的元教训

1. **"判据只看产物存在"是一个反复复发的反模式**（坑 33→42→47→56→65，第 5 次）。
   它的危害不是"漏报"，而是**把失败伪装成通过**。凡是"产出文件无论成败都会写"的阶段，
   判据就必须看**内容里的语义标记**。
2. **流水线的"接线"也必须被测试。** 坑 63/64 都不是逻辑错误，而是 driver 与脚本**之间的约定**没人校验。
   `run_from_zero.sh` 跑了 9 个阶段，其中 2 个因为参数名/模块名不匹配**从未真正执行**——
   而它们的 FAIL 看起来像"环境问题"。**新增 G11/G12 把"接线"变成可自动校验的对象。**
3. **闸门本身会产生假阳性，而且往往比被测对象的缺陷更隐蔽。** 本地没有 torch 时，
   "脚本 --help 跑不起来" ≠ "接口不匹配"；`refs/` 里的参考源码需要 torch 也不是缺陷。
   **每加一条闸门，都要立刻拿一个已知好样本和一个已知坏样本各跑一遍**（这就是 INV-2）。

---

## 坑 67-70 —— 验证环节自身暴露的缺陷（2026-09-16 真机）

> 本轮四项修复全部在真机验证通过：
> `TRITON_NPU_OK backend=ascend triton=3.2.0` · P3 `backend: Ascend NPU` rc=0 ·
> P6 `[sdpa] 0.067ms` rc=0 · P4 `weight_hf MISS 空目录`。
> **但验证过程本身又暴露了 4 个问题，其中坑 67 是我的工具把证据抹掉了。**

| # | 现象 | 根因 | 处置（已落地） | 判据 |
|---|---|---|---|---|
| **67** | ★★ 推送新版 Skill 后，远端 `out/train/train.log` 与裁决表**全部消失** | `ssh_tool.push_dir` 用 `rm -rf <remote_dir>; mv new <remote_dir>` —— **整体替换**，把远端**本地没有的文件**（积累的运行产物 `out/`、日志、profiling）一并删除。而长任务（19GB 下载 / 100 步训练）的产物是**最贵的证据** | 改为**覆盖式合并**：解到临时目录后 `cp -a tmp/. remote/` 逐文件覆盖，**保留远端独有文件**；确需整体替换才用 `--clean` | 推送后 `out/` 内容仍在 |
| **68** | 明明下载已结束，`pgrep` 却说"还在跑"（`ALREADY_RUNNING`） | `pgrep -f <pattern>` 匹配**完整命令行**，而我通过 SSH 发的那条命令里**本身就含有该模式字符串** → **匹配到自己的 shell**。经典自杀式假阳性 | 判存活改用 **pidfile**（`ssh_tool.bg` 一直写 pidfile，`alive` 子命令读它）；万不得已用 `pgrep -f` 时必须排除自身 PID | `alive` 子命令按 pidfile 判定 |
| **69** | 验证输出里 `rc=0`，但脚本实际返回 3 | 我在 `python3 x.py ... \| tail -10; echo $?` 中取 `$?` —— 那是**管道最后一个命令（tail）**的退出码，不是被测脚本的。**被测对象的退出码被管道吞了** | 需取被测命令退出码时用 `${PIPESTATUS[0]}`，或不要接管道；重要判据一律**直接读标记**（如 `ASSETS_INCOMPLETE`）而非退出码 | 验证脚本改用 `${PIPESTATUS[0]}` |
| **70** | ★★ 连续两轮 `/root/Qwen3.5-0.8B-hf` 都是 `0`：**权重根本没下载** | 上次失败留下一个**空目录**，而主流程用 `os.path.isdir(model_hf)` 判"已存在"→ 直接**跳过下载分支** → 永远下载不了。**"跳过条件"与"验收条件"用了两套宽严不同的标准** | 抽出 `_weights_ready()` / `_dir_nonempty()` / `_file_bytes_ok()` / `_json_samples()`，**跳过条件与验收条件共用同一套实质判定**；目录存在但内容不就绪时打印"将重新下载" | 空目录 → `下载 hf 权重...`（而非 `已存在，跳过`） |

### 坑 67-70 的元教训

1. **"证据"比"代码"更脆。** 代码可以从本地重推，但**跑了几小时的训练日志/下载产物删了就没了**。
   任何"同步/推送/清理"操作都必须**默认保守**（合并而非替换、备份而非删除）。
2. **坑 70 说明"幂等"设计有个隐藏前提**：跳过某步的判据，必须与"这步算完成了吗"的判据**一致**。
   两套标准一旦宽严不一，就会出现"因为半个产物存在而永久跳过补齐"的死锁——
   而且它**不会报错**，只会让资产一直是 0。
3. **自己的验证命令也可能是错的**（坑 69）。所以"验证"这件事同样要有判据纪律：
   **优先读被测对象打出的语义标记，其次才是退出码；用退出码时注意管道。**

---

## 坑 71-73 —— 训练链路的真正阻塞（2026-09-16 真机）

> 本轮验证结果：坑 67 **已实证修复**（sentinel 保留 `PRESERVE_OK`）；
> 但坑 70 的修复**没能生效**（原因见坑 72），并终于抓到 P5 的真实异常。

| # | 现象 | 根因 | 处置（已落地） | 判据 |
|---|---|---|---|---|
| **71** | ★★ P5 训练失败，真实异常是 `ModuleNotFoundError: No module named 'mindspeed'`（此前被 torchrun 的汇总信息掩盖成 `TRAIN_FAIL iters=0`） | MSMM 依赖 **MindSpeed 核心包**，而 `versions.lock` **早已写明**要 `gitcode.com/Ascend/MindSpeed editable --no-deps`；但 `bringup.sh` **从未真正安装它**，只检查 `mindspeed_mm` 且**仅 warn 不 fail** → **P5 在任何新机器上都不可能成功** | ① `bringup.sh` phase2 真正安装：三源尝试克隆 MindSpeed（gitcode→github→gitee）后 `pip install -e ... --no-deps`，失败即 `[FAIL]` 并打印手工命令 ② 同理处理 `mindspeed_mm`（`pip install -e` 或 MSMM 目录内可导入） ③ `50_train.py` 注入 `PYTHONPATH=<workdir>` 作兜底 | bringup 输出 `[PASS] mindspeed 安装成功`；P5 不再报 `No module named 'mindspeed'` |
| **72** | 重启下载的命令**返回"(无输出)"**，下载根本没启动（权重仍为 0） | 命令里写了 `pkill -f 'python3 scripts/40_prepare_assets.py'` —— 而**这条 SSH 命令行本身就含该字符串** → `pkill -f` **匹配并杀掉了自己的 shell**（还没执行到后面的启动语句就死了）。这是坑 68（`pgrep -f` 自杀式假阳性）的**破坏性版本** | 纪律：**绝不用可能匹配自身命令行的 `-f` 模式做 pkill**。改用 pidfile（`ssh_tool.bg`/`alive`）；或先 `pgrep -f PATTERN | grep -v $$` 再按 PID 杀 | 启动命令返回 `LAUNCHED` 且 `du -sh` 权重目录增长 |
| **73** | bringup 对多个关键依赖**只 warn 不 fail**（`mindspeed_mm import`、`triton` 等） | 把"必需项缺失"降级成 warn，会让流水线带着必然失败的配置继续跑，直到某阶段才以难懂的方式炸（如坑 71 的 torchrun 汇总） | 区分「必需」（缺失即 `[FAIL]` + 可执行修复命令）与「可选」（warn + 影响说明）；`mindspeed`/`mindspeed_mm` 归为必需 | bringup 末尾汇总中必需项无 FAIL |
| **74** | 为防坑 71 新加的 **G14 闸门自己先是"太宽"、后是"太窄"** | ① 第一版从 `versions.lock` 的**形状**启发式推包名 → 把 `hardware.soc`、`model.name`、`mode` 等**配置项**当包 → 8 个假阳性、`PREPUSH_FAIL` 全是噪声 ② 收紧成显式清单后**过窄**：只认字面 `pip install`，而 bringup 用的是**自家 `pip_install()` 函数** → 又误报 3 个缺步骤 | ① **契约显式声明**：G14 用固定清单 `[mindspeed, mindspeed_mm, torch, torch_npu, transformers, triton_ascend]`，含义明确、不随 lock 写法漂移 ② 安装动作识别加入 `pip_install`（项目帮助函数）/`install -e`/`git clone` 等 ③ **双向验证**：正向 PASS（6/6）；负向把 `pip_install` 改坏 → 正确 FAIL | `G14` 正向 PASS，负向可控 FAIL（符合 INV-2） |
| **75** | 后台跑 `bringup.sh` 时，调用方写的 `> /root/bringup2.log` 是**空的**，看起来像"脚本根本没跑" | `bringup.sh` 内部写死 `LOG=/root/ascend_bringup_run.log` 并在非 TTY 时 `exec >> "$LOG" 2>&1` → **静默覆盖调用方的重定向**。于是排查者盯着一份空日志找问题，而真实输出在另一个文件里 | 日志路径改为可覆盖：`LOG="${BRINGUP_LOG:-/root/ascend_bringup_run.log}"`，并在头部用法里写明。**凡脚本内部会重定向输出，就必须让调用方能覆盖目的地。** | `BRINGUP_LOG=/root/x.log bash scripts/bringup.sh` 后 `x.log` 非空 |
| **76** | ★ BOM 问题**在工具代码上复发**（`step13_*.py` 被 `Set-Content -Encoding UTF8` 写入 BOM，`ast.parse` 直接报 `invalid non-printable character U+FEFF`） | 我用 PowerShell 改文件时反复引入 BOM —— 与坑 59 完全同因。而 G1 只扫 Skill 目录，**扫不到工具脚本** | ① G1 扫描范围从 1 个根扩到 **3 个根**（Skill + 交付物生成脚本 + 分析脚本）② 纪律：源码文件一律用编辑工具改，不用 shell 重定向/`Set-Content` | `G1 BOM 扫描 PASS（扫描 3 个根目录）` |

---

## 坑 77-79 —— `--stage=N` 从设计上不成立 + 下载依赖缺失（2026-09-16 真机）

> 本轮把 `bringup --stage=2` 真正跑了一遍，结果是**三行报错**，而它们都指向"续跑/安装"这条最常用的路径。

| # | 现象 | 根因 | 处置（已落地） | 判据 |
|---|---|---|---|---|
| **77** | phase4 报 `scripts/bringup.sh: line 313: MSMM_DIR: unbound variable` | `MSMM_DIR` 在 phase4 被使用，但脚本**从未定义它**（我写 phase4 时借用了 `run_from_zero.sh` 的变量名）。`set -u` 下直接中止 | 在文件顶部**集中定义跨 phase 的路径变量**并允许环境变量覆盖：`MSMM_DIR="${MSMM_DIR:-/root/MindSpeed-MM}"`、`DATA_DIR="${DATA_DIR:-/root/data}"` | bringup 跑到 phase4 无 `unbound variable` |
| **78** | ★★ `bringup.sh --stage=2` **必然失败**：`[FAIL] PYBIN 未锁定 / 修复: 先修 phase0b` | `choose_python` 被放在 `if [ START_STAGE -le 1 ]` 里 → **`--stage≥2` 跳过解释器锁定** → phase2 一开头就因 `PYBIN` 为空而 FAIL。**"从阶段 N 续跑"这个功能对 N≥2 从来就不可能工作**（而 `--stage=2` 恰恰是"只装依赖"的常用姿势） | phase0/phase0b **无条件先跑**（它们廉价且幂等，只做探测与系统依赖补齐），阶段号只用来控制"要不要执行某个 phase" | `--stage=2` 输出 `[PASS] 锁定 PYBIN=...` 并继续 |
| **79** | 资产下载"成功结束"但**权重目录一直是 0**，日志里没有任何明确的失败原因 | `fetch_model` 走的是 **`modelscope download` CLI**。该 CLI 未安装时，原实现会**退避重试 3 次去跑一个不存在的命令**（白等 45 秒），只留下一句含糊失败，然后静默转备源 —— 让人误判成"网络问题" | ① `_ms_download` 增加**前置 CLI 探测**，缺失即立刻返回明确原因与安装命令 ② `bringup.sh` phase2 **安装 modelscope CLI** ③ G14 必需包清单加入 `modelscope` | 日志出现 `modelscope CLI 不存在（…安装: pip install modelscope）` 或 `[PASS] modelscope CLI 安装成功` |

### 坑 77-79 的元教训
1. **"续跑/局部执行"是脚本最容易坏的功能**，因为它打破了"从第一行顺序执行"的隐含前提。
   凡有 `--stage/--from` 之类的开关，必须保证**它依赖的前置状态要么被恢复、要么被显式检查**，
   而不是假设"调用者已经跑过前面的阶段"。
2. **外部 CLI 的缺失必须以"明确原因"暴露**，不能退化成"重试 N 次后静默转备源"。
   退避重试是为**瞬时故障**设计的；**确定性缺失**（命令不存在）重试多少次都不会变好，
   只会把一次清晰的失败变成一次含糊的失败 —— 而含糊的失败正是排查成本的主要来源。
3. **变量作用域也是"接线"的一部分**（坑 77）：跨函数/跨阶段共用的路径变量若散落定义，
   总会在某个分支上漏掉。做法是**在顶部集中声明**。

---

## 坑 80 —— "装好了吗"的检查本身依赖环境变量（2026-09-16 真机）

> 本轮 `bringup --stage=2` 取得实质突破（见下），同时暴露这个隐藏很深的判定缺陷。

**本轮成效**（坑 71/78/79 全部实证修复）：
```
[PASS] 锁定 PYBIN=/usr/bin/python3 (py3.11)              ← 坑78：--stage=2 终于能工作
[PASS] triton 可用（实测 kernel 通过）: TRITON_NPU_OK     ← 坑60 确证
/usr/local/bin/modelscope                                 ← 坑79：CLI 已装
/root/MindSpeed + import mindspeed → MINDS_OK             ← 坑71：MindSpeed 装上
/root/Qwen3.5-0.8B-hf: 921M                               ← 权重**终于开始下载**（此前恒为 0）
已 provision mock 配置: config/templates/qwen3_5_0_8B_mock.yaml → MSMM/examples/...
```

| # | 现象 | 根因 | 处置（已落地） | 判据 |
|---|---|---|---|---|
| **80** | 明明显式装好了 MSMM，`import mindspeed_mm` 却报 `ModuleNotFoundError: No module named 'megatron'` | `mindspeed_mm/__init__.py` 有**守卫**：**不设 `NON_MEGATRON=true` 时会去 import megatron**（本项目根本不使用 Megatron，故从未安装它）。于是任何"MSMM 装好了吗"的**裸导入检查都会得到假阴性**——报的是"缺 megatron"，而真实状态是"装好了" | bringup 的可导入性判定改为 `env NON_MEGATRON=true "$PYBIN" -c "import mindspeed_mm"`（训练本身也需要该变量，见坑 26）。**判定条件必须与运行条件一致**——这与坑 70（跳过条件≠验收条件）是同一族问题的另一种形式 | bringup 输出 `[PASS] mindspeed_mm import ok（NON_MEGATRON=true）` |

### 坑 80 的元教训

**"能否导入"不是模块的属性，而是"模块 + 环境变量 + 工作目录"三者的联合属性。**
所以凡是"X 装好了吗"的检查，都必须用**与真实运行完全相同的一组前置条件**去问——
否则检查结果与运行结果会不一致，而检查失败给出的诊断方向（"缺 megatron"）还会把人带偏。
这与坑 70 同族：**跳过条件、验收条件、运行条件，三者必须同源。**

**待查（下一轮）**：phase4 的 mock 冒烟 `[FAIL] mock 训练失败`（已 provision 配置，需看 `/root/mock_run.log`）。
可能与坑 80 同因（50_train.py 注入 NON_MEGATRON 是在 torchrun 前 export，mock 分支在 bringup 里也有 export，
需看实际日志确认）。

---

## 坑 81-82 —— 项目自带依赖清单被忽略（2026-09-16 真机，决定性定位）

> 本轮拿到两项关键进展与 P5 的**真正阻塞**。

**进展**：
```
权重下载完成:   /root/Qwen3.5-0.8B-hf = 1.7G（model.safetensors + config.json + tokenizer 齐备）
COCO 下载中:    /root/data = 8.1G（目标 ~19GB）
裁决表（修正后）: PRE PASS | P0 PASS | P1 PASS | P2 PASS | P3 PASS(9s) | P4 FAIL | P5 FAIL | P6 FAIL | P7 FAIL
                 ↑ P3 首次通过（坑63 修复生效）      ↑ P4 正确记 FAIL（坑65 修复生效，判据诚实了）
```

| # | 现象 | 根因 | 处置（已落地） | 判据 |
|---|---|---|---|---|
| **81** | ★★ P5 训练与 mock 冒烟都失败，真实异常是 `ModuleNotFoundError: No module named 'einops'`；DCP 转换则缺 `pydantic` | **MSMM 有自己的 `requirements.txt`**，而 bringup 只挑了 torch/torch_npu/transformers/triton/pyyaml/numpy **手装** → 全新 clone 的仓库一跑训练就缺依赖。**"依赖"是项目自带的清单，不是我们猜出来的几个包** | ① bringup phase2 **消费 `requirements.txt`**（huaweicloud 源），失败再回退核心清单 ② 逐项校验 `einops/pydantic/pyyaml/numpy` 是否真能 import，缺失即 `[FAIL]` + 手工命令 ③ `50_train.py` 新增诊断规则：`ModuleNotFoundError: No module named '<x>'`（用负向前瞻排除 megatron）→ 提示"装 requirements.txt 而不是逐个猜包" | bringup 输出 `[PASS] MSMM 核心依赖（einops/pydantic/pyyaml/numpy）齐备` |
| **82** | mock 失败时只显示 `[FAIL] mock 训练失败 / 修复: tail -50`，而真实异常（`einops`）被 torchrun 的 `ChildFailedError` 汇总**掩盖在日志前部** | 诊断停留在"指个文件让你自己看"，而框架的汇总信息会把真正的根因淹没 —— 排查成本剧增（这与坑 71 当初被掩盖成 `TRAIN_FAIL iters=0` 完全同因） | bringup 在 mock 失败时**自动提取并置顶**第一条真实异常（`ModuleNotFoundError\|ImportError\|NameError\|AssertionError\|RuntimeError\|FileNotFoundError\|KeyError\|ValueError\|OSError`）；`50_train.py` 同步新增同类规则 | 失败输出含"── 第一条真实异常（自动提取）──"段 |

### 坑 81-82 的元教训

1. **"依赖"应当来自项目自己的清单，而不是我们凭印象挑的包。**
   我们手装的 7 个包恰好覆盖了"能 import 框架"，却没覆盖"能跑训练"——
   因为**框架的可导入性与可运行性不是一回事**（与坑 80 同族：检查条件 ≠ 运行条件）。
   凡是"某项目能否运行"，正解是消费它自己的依赖声明。
2. **框架的汇总型错误信息会淹没根因。** torchrun 只报 `ChildFailedError: <NO_OTHER_FAILURES>`，
   真正的 `ModuleNotFoundError` 在日志前部。**诊断代码必须主动下沉并置顶第一条真实异常**，
   否则每一轮排查都要重新"考古"。这一点已在坑 71、82 两次付出代价。

---

## 坑 83-85 —— 依赖推断的三次修正（2026-09-16 真机）

**本轮进展**：
```
einops OK | pydantic OK | yaml OK | numpy OK          ← 核心兜底清单生效（坑81 的兜底救了场）
mindspeed_mm import ok（NON_MEGATRON=true）            ← 坑80 实证修复
/root/data = 21G                                       ← COCO 压缩包已下完
```

| # | 现象 | 根因 | 处置（已落地） | 判据 |
|---|---|---|---|---|
| **83** | mock 训练仍失败，真实异常 `importlib.metadata.PackageNotFoundError: No package metadata was found for protobuf` | **"能 import" ≠ "元数据可查"**：某些工具链（torch/CANN 侧）会通过 `importlib.metadata.version('protobuf')` 读版本，而包若缺 `dist-info`（例如由系统包管理器装入）就会抛该错。**只用 import 做依赖检查发现不了这一类** | ① bringup 新增 protobuf **元数据**专项检查（`importlib.metadata.version('protobuf')`），缺失即 `--force-reinstall --no-deps` 补齐 ② `50_train.py` 新增 `PackageNotFoundError` 诊断规则 | bringup 输出 `[PASS] protobuf 元数据可查（x.y.z）` |
| **84** | ★ 我为坑 82 加的"第一条真实异常"提取器**一个字都没打出来** | 它用**枚举清单**（`ModuleNotFoundError\|ImportError\|NameError\|...`），而真机的异常是 `PackageNotFoundError`——不在清单里。**枚举永远会漏** | 改为**通用异常模式** `^[A-Za-z_][A-Za-z0-9_.]*(Error\|Exception)(:\|$)`；若仍无命中，则打印第一处 `Traceback` 的后续 10 行作为兜底 | 对 `PackageNotFoundError` 样本实测可命中 |
| **85** | 上一轮"消费 MSMM `requirements.txt`"的修复**前提是错的** | MSMM v26.1.0 **没有顶层运行期 `requirements.txt`**；仓库中唯一的 `requirements.txt` 位于 `UserGuide/`，内容是 **sphinx 文档依赖**。真正救场的是"核心兜底清单"（补装了 einops/pydantic/PIL） | ① 只认**运行期**声明：顶层 `requirements*.txt` / `pyproject.toml` / `setup.py`，**明确跳过 UserGuide/docs** ② 核心清单加入 `protobuf` ③ 明确写下：**最终权威判据是 mock 冒烟，而不是任何依赖清单** | bringup 打印 `MSMM 无顶层运行期 requirements.txt（UserGuide/ 下的是 sphinx 文档依赖，已跳过）` |
| 附 | pip 警告 `mindspeed-mm 0.1 requires transformers==4.57.0, but you have transformers 5.2.0 which is incompatible` | 仓库元数据声明的版本与 `versions.lock` 的官方配对（5.2.0）**冲突**。当前以 lock 为准（训练此前在旧机验证过），但这是一个**已知偏差**，需在报告中如实标注 | 记录并保留；若最终精度异常，此为第一顺位的怀疑项 | 报告/文档中显式列出该冲突 |

### 坑 83-85 的元教训

1. **"依赖是否就位"有三种强度，不能混用**：
   | 强度 | 判据 | 漏掉的典型问题 |
   |---|---|---|
   | 能 import | `python3 -c "import X"` | 元数据缺失（坑 83）、版本不匹配 |
   | 元数据可查 | `importlib.metadata.version(X)` | 版本与声明冲突 |
   | **真能跑** | **端到端冒烟（mock）** | —— |
   **只有第三种是权威判据。** 前两种是快速前置检查，能提前定位，但不能替代冒烟。
2. **枚举清单式诊断必然漏项**（坑 84）。凡是"从日志里找错误"，应当用**通用的异常模式**
   （`XxxError/Exception`）而不是列举已知类型——因为未知错误的类型恰恰是你想不到的那个。
3. **修复也可能建立在错误前提上**（坑 85）。"MSMM 有 requirements.txt"这个前提我没验证就写进了代码，
   真机一跑才发现那个文件是文档依赖。**凡是"某文件/某接口存在且含义为 X"的假设，都要有一条判据去证实它，
   而不是等它在真机上以"WARN 未找到"的形式悄悄滑过。**

---

## 坑 86 —— 用"闭环求解"取代"枚举猜测"（2026-09-16 真机，方法论转折）

> **本轮里程碑**：资产链**全部打通**，可比性达到最强档。
> ```
> ASSETS_OK checks=3 fails=0
>   weight_hf      OK   14 个文件
>   llava_json     OK   字节数与锚点一致（228,941,895 B）
>   converted_json OK   **157712 样本与锚点一致**
>   coco_images    OK   **118287 张与锚点一致**
> 数据可比性等级: 构造性同源（同源下载 + 官方转换脚本 + 样本数一致 + shuffle=false）
> protobuf 元数据已补齐                    ← 坑83 修复生效
> 通用异常提取器命中 ModuleNotFoundError: No module named 'accelerate'   ← 坑84 修复生效
> ```

| # | 现象 | 根因 | 处置（已落地） | 判据 |
|---|---|---|---|---|
| **86** | 依赖补齐**每轮只能推进一项**：einops → pydantic → protobuf → accelerate → …（每次都要消耗一个 token 周期） | 我采用"**枚举并预装**"策略：凭印象列出核心包。而**枚举永远不全** —— 真实依赖只有把训练入口真正 import 一遍才会**逐个暴露**。这与坑 84（枚举式错误提取器漏掉 `PackageNotFoundError`）是**同一个方法论错误** | 新增 **`resolve_deps()` 依赖闭环求解**：反复尝试 `import mindspeed_mm.fsdp.train.trainer`，从 `No module named 'X'` 里取真实缺失项 → 经 `map_pkg()` 名称映射（yaml→pyyaml / PIL→pillow / google.protobuf→protobuf / cv2→opencv-python-headless / sklearn→scikit-learn）→ 安装 → 再试，上限 15 轮。核心清单只作"快速命中"，**闭环才是兜底** | bringup 输出 `[PASS] 依赖闭环：训练入口可导入（补齐迭代 N 次）` |

### 坑 86 的元教训（本项目方法论层面的收束）

**"枚举已知项"与"闭环求解未知项"是两种根本不同的工程姿态。**

| | 枚举 | 闭环 |
|---|---|---|
| 依赖补齐 | 列出我想到的包 | import 一遍，缺什么装什么 |
| 错误提取 | 列出我想到的异常类型（坑 84） | 通用 `XxxError/Exception` 模式 |
| 判据设计 | 列出我想到的检查 | 负向对照证明判据能失败（INV-2） |
| 环境适配 | 列出我想到的发行版/解释器（坑 37/38） | 探测后分派（`_envcompat`） |

本项目前 86 个坑里，**凡是"枚举"姿态写出来的东西，几乎都在真机上被证明不全**；
凡是"闭环/探测"姿态写出来的（`_envcompat`、能力矩阵、判据负向对照、依赖闭环），都能自我修正。
**这是这几轮最值得沉淀的一条经验。**

---

## 坑 87 —— 闭环的探针必须就是最终要跑通的那件事（2026-09-16 真机）

| # | 现象 | 根因 | 处置（已落地） | 判据 |
|---|---|---|---|---|
| **87** | 依赖闭环跑完并报 `[PASS] 依赖闭环：训练入口可导入（补齐迭代 4 次）`（装了 accelerate/torchdata/peft/av），**但 mock 冒烟仍然失败**，缺的是 `datasets` | **闭环的探针选错了**：我用 `import mindspeed_mm.fsdp.train.trainer` 作判据 —— 它只覆盖"**trainer 模块的 import 图**"，**不覆盖运行期图**（数据加载路径要 `datasets`）。于是闭环在一个**比目标弱的判据**上"成功收敛"，而真正的目标（mock 训练）依旧不通。这与坑 65/70/80 同族：**判据弱于它要回答的问题** | 把闭环的判据改为**目标本身**：`phase4` 改为**以 mock 冒烟为判据**的循环（跑 mock → 提取真实缺失模块 → 补装 → 再跑，上限 10 轮）。`resolve_deps()`（trainer 级）保留作快速前置检查，但**不再作为收敛判据** | bringup 输出 `[PASS] mock 训练跑通（依赖闭环补齐 N 次）` |

### 坑 87 的元教训

**闭环有效，但闭环的成败完全取决于"判据选得对不对"。**

一个闭环 = 判据 + 修复动作 + 重试。如果判据比目标弱，闭环会**自信地收敛到一个错误的地方** ——
这比没有闭环更危险，因为它会输出 `[PASS]`。

| 判据 | 强度 | 后果 |
|---|---|---|
| `import trainer` | 弱 | 闭环在这里收敛，但 mock 仍失败（坑 87） |
| `import mindspeed_mm` | 更弱 | 连依赖都没覆盖 |
| **mock 冒烟通过** | **等于目标** | ✅ 唯一正确的收敛点 |
| 100 步训练 + 判定链 | 等于最终交付 | 最终验收判据 |

**统一规律（本项目反复验证）**：
> **跳过条件、验收条件、运行条件、闭环收敛条件 —— 这四者必须同源，
> 且必须等于"最终要跑通的那件事"。** 任何比目标弱的判据，都会把失败伪装成成功。

---

## 坑 88 —— 日志前缀让所有行首锚定模式失效（2026-09-16 真机）

| # | 现象 | 根因 | 处置（已落地） | 判据 |
|---|---|---|---|---|
| **88** | 以 mock 为判据的闭环**只跑了 1 轮就退出**，且"真实异常提取"段**一个字都没打出来** | **torchrun 给日志每一行加了 `[rank0]: ` 前缀** → 我写的所有 `^[A-Za-z_]...` **行首锚定**模式全部失效。于是：① 异常提取器空手而归 ② 闭环取不到 `MOD` → 判定"非缺模块" → 立即退出 | ① 异常模式加入可选前缀：`^(\[[^\]]+\]:[ \t]*)?[A-Za-z_][A-Za-z0-9_.]*(Error\|Exception)(:\|$)` ② 新增「**最深的调用帧**」提取（`File ".*", line N` 的**最后几条**通常直接指出真凶）③ Traceback 兜底从 10 行加到 25 行 ④ 闭环在"非缺模块"时**明确说明**"本次不是缺依赖，请按最深调用帧定位"，并自动列出 mock 数据目录 | 用真实样本双向自检：带 `[rank0]:` 前缀与不带前缀**都能命中**（异常 3 条 / 帧 2 条） |

### 坑 88 的元教训

**"行首锚定"依赖一个隐含前提：日志就是原文。** 而真实训练的日志会被分布式启动器改写
（`[rank0]:` 前缀、`E0916 ...` 时间戳前缀、warnings 插行）。
凡是解析日志的判据，都必须**先确认日志的实际行格式**，或直接写成"容忍任意前缀"。

这与坑 42/56/60/62/84 属同一族 —— **判据依赖了未经确认的格式假设**。
至此该族已出现 6 次，故在 `ROBUSTNESS.md` 的 INV-2 中补一条：
> **面向外部文本（日志/输出）的判据，不得假设行首/行序/编码；一律用"容忍前缀 + 语义标记 grep"。**

---

## 坑 89-90 —— "配置被发布、数据没被发布" 与 参数名写错（2026-09-16 真机）

> **本轮最大的收获是：提取器修好后，真实根因一次就拿到了。**
> ```
> [rank0]: FileNotFoundError: Unable to find '/root/MindSpeed-MM/data/mocked_vl_data/mock_data_pic_num_4_textlen_512.json'
> ls: cannot access '/root/MindSpeed-MM/data/mocked_vl_data/': No such file or directory
> ```
> **坑 88 的修复（容忍 `[rank0]:` 前缀 + 最深调用帧 + 25 行 Traceback）直接兑现了价值** ——
> 在此之前这个信息被埋在日志里，白耗了两轮。

| # | 现象 | 根因 | 处置（已落地） | 判据 |
|---|---|---|---|---|
| **89** | mock 冒烟报 `FileNotFoundError: .../mock_data_pic_num_4_textlen_512.json` | **mock 配置随包发布了，但它依赖的 mock 数据集没有** —— 那个目录是开发期在旧机器上用 MSMM 自带生成器造出来的。**与坑 48（mock 配置本身）是同族问题**：配置与其数据必须一起被"发现" | 正解**不是**把数据塞进包里（二进制、与仓库版本强耦合），而是**调用 MSMM 自带的生成器**：`mindspeed_mm/fsdp/tools/data_tool/generate_mock_data_for_vlmodel.py --save_dir <MSMM>/data/mocked_vl_data/`。已在 phase4 内建（幂等；已存在则跳过） | bringup 输出 `[PASS] mock 数据集已生成` |
| **90** | DCP 转换长期失败（此前归因于缺 pydantic，补完仍不通） | **参数名写错**：我们的脚本用 `--load_path/--save_path`，而旧机器上**真正跑通**的形式是 `--hf_dir/--dcp_dir`（证据：分析脚本 `a3_conv2.sh`）。另 `convert_cli` 依赖 `docstring_parser`，缺它直接 ImportError | 改为已验证的参数形式，并在转换前安装 `docstring_parser`；转换后若目录仍空则登记降级 | 命令与 `a3_conv2.sh` 逐字一致 |

### 坑 89-90 的元教训

1. **"配置与它的数据/依赖"是一个整体，发布时必须一起被发现。**
   坑 48（配置没打包）、坑 71（MindSpeed 没装）、坑 79（modelscope CLI 没装）、坑 89（mock 数据没生成）
   —— 这 4 个坑的形态完全一致：**某件东西"存在性"是隐含前提，却从未被检查或创建**。
   → 纪律：**凡是配置里引用的路径/包，都必须有一条"发现或创建"的步骤**，
   否则它会在最不该失败的地方（冒烟测试）以 `FileNotFoundError` 的形式爆炸。
2. **优先使用项目自带的工具，而不是自己造或直接塞二进制。**
   mock 数据的正解是 MSMM 自己的 `generate_mock_data_for_vlmodel.py` ——
   它随仓库版本演进，schema 天然一致；自己造反而会引入 schema 漂移。
3. **"已验证的命令形式"是宝贵资产。** 坑 90 的修法不是猜测，而是**回到旧机器上真正跑通的那条命令**
   （`a3_conv2.sh` 里逐字保存着）。**保留并复用"曾经成功过的原始命令"，比重新推断参数名可靠得多。**

---

## 坑 91 —— "跑一次工具" ≠ "得到配置期望的那个产物"（2026-09-16 真机）

| # | 现象 | 根因 | 处置（已落地） | 判据 |
|---|---|---|---|---|
| **91** | 调用 MSMM 自带 mock 生成器后，目录里**只有 `test_pic_h1024_w1024.jpg`**，没有配置期望的 `mock_data_pic_num_4_textlen_512.json` → mock 仍报 `FileNotFoundError` | **生成器的输出文件名与其参数相关**，我按"跑一次就有正确文件名"的假设写代码 —— 与坑 87（闭环探针）、坑 88（格式假设）同族：**又一次把"我以为"当作"事实"** | 改为**自愈式适配**：① 从配置里**解析出期望路径**（`grep` 出 `mock_data*.json`），而不是写死文件名 ② 按文件名语义给参数（`pic_num_4_textlen_512` → `--pic_num 4 --text_len 512`），失败再退回最简调用 ③ 若期望文件仍缺失，**自动把配置的 `dataset:` 改指向目录里实际存在的 json**（用 regex 重写配置并打印 PATCHED）④ 完全没有 json 时给出 `--help` 指引，不伪造成功 | bringup 输出 `[PASS] mock 数据集就绪: <实际文件名列表>` |

### 坑 91 的元教训

**"调用了一个工具"与"得到了需要的产物"之间，隔着一个"输出契约"。**
凡是调用外部工具，都必须**显式校验"我要的那个东西"是否真的产生了**，
而不是校验"命令返回 0"或"目录非空"。

- 反例（本次）：检查"目录里有东西" → 只有一张 jpg，判定为"已生成" → 后续仍失败。
- 正解：**先确定需要的产物路径，再验证它存在**；不存在时**适配或改写引用方**（自愈），
  最后才考虑报错。

这条与坑 73/82（诊断要落到真实根因）共同构成一条工程纪律：
> **每一步都要以"下游真正需要的那个产物"为验收对象，而不是以"这一步跑没跑"为验收对象。**

---

## 坑 92 —— "猜参数名"必然错，正解是"从配置反推"（2026-09-16 真机）

> 拿到生成器 `--help` 后，真相一目了然（**这一步的成本只有一次调用，却省掉了后续所有猜测**）：
> ```
> --tokenizer_path   default=/home/weights/Qwen3.5-35B-A3B/   ← 本机不存在！
> --num_pics         （不是我猜的 --pic_num）
> --text_length      （不是我猜的 --text_len）
> 写入: mock_data_pic_num_{num_pics}_textlen_{text_length}.json
> ```

| # | 现象 | 根因 | 处置（已落地） | 判据 |
|---|---|---|---|---|
| **92** | 调用生成器后**只留下一张 jpg**，配置期望的 json 始终没有 | ① 我按**文件名语义猜参数名**（`--pic_num/--text_len`），真实名是 `--num_pics/--text_length` ② 更关键：`--tokenizer_path` 默认值 `/home/weights/Qwen3.5-35B-A3B/` **在本机不存在** → 生成器在"画完图之后、写 json 之前"就死了（这解释了"只有 jpg"这个现象） | **从配置反推参数，而不是猜**：① 期望路径 → `--num_pics`（取 `pic_num_N`）、`--text_length`（取 `textlen_N`）② tokenizer 路径取自配置里的 `model_name_or_path` ③ 三者都做了缺省兜底。**任何工具调用前先取 `--help`，不猜接口。** | 本地自测：从 mock 配置反推得 `--num_pics 4 --text_length 512 --tokenizer_path /root/Qwen3.5-0.8B-hf`，与期望文件名一致（`DERIVE_OK`） |

### 坑 92 的元教训

1. **"猜接口"是本项目最高频的返工来源**（坑 63 参数名、坑 64 模块名、坑 90 参数名、坑 92 参数名+默认值）。
   四次都是"我凭印象写调用"。**纪律：调用任何外部脚本前，先跑 `--help` 或读它的 argparse；
   一次性成本，换来不再返工。**
2. **默认值也是接口契约的一部分。** 坑 92 的直接死因不是参数名（argparse 会报错），
   而是**默认 tokenizer 路径指向一个开发者本机才有的位置**。
   → 凡工具带 `default=`，都必须检查"该默认值在当前环境是否成立"，不成立就显式覆盖。
3. **现象与根因的距离**："只留下一张 jpg"看起来像"生成不完整"，
   实际是"在写 json 之前因 tokenizer 路径失败"。**遇到"部分产出"，第一反应应是"它在哪一步死的"，
   而不是"它的输出格式不同"。**（本次靠 `--help`+读源码 25 行一次性定位。）

---

## 坑 93-94 —— 依赖"最新版"与"模型文件不完整"（2026-09-16 真机）

**本轮进展（mock 链已推进到很深处）**：
```
[PASS] mock 数据集就绪: mock_data_pic_num_4_textlen_512.json (1.35MB) + test_pic_h1024_w1024.jpg
mock json: n=512, keys=['images','messages']        ← 数据格式正确
新失败点: ValueError: Processor was not found, please check and update your model file.
```

| # | 现象 | 根因 | 处置 | 判据 |
|---|---|---|---|---|
| **93** | 依赖闭环安装的是**最新版**，与 MSMM v26.1.0 未必兼容 | 实测装到 `datasets 5.0.1 / pandas 3.0.5 / pyarrow 25.0.1 / av 18.1.0 / accelerate 1.15.0 / peft 0.21.0` —— 而 `versions.lock` 的"关键运行依赖"一节**只列了包名、几乎不给版本号**，且**根本没列 `datasets`/`accelerate`/`peft`**。→ ① 锁不完备 ② 无版本约束时 pip 取最新 → 兼容性不可控 | ① `versions.lock` 需补齐"关键运行依赖"的**实测版本**（本轮已采集到一组可用版本，待 mock 全绿后回填）② 依赖闭环装上后应**记录实际版本**到产物，便于回溯 | lock 补全数据集依赖版本 |
| **94** | `ValueError: Processor was not found, please check and update your model file.` | `AutoProcessor` 无法从 `/root/Qwen3.5-0.8B-hf` 构建 processor。而 `fetch_model()` 的"成功判据"是 `any(f.endswith(('.safetensors','.bin','.json')))` —— **只要有一个 json 就算成功**，**完全无法发现 processor/config 类文件缺失**（与坑 61/70/89 同族：存在性判据过弱） | 下一轮先定位：① 完整列出 hf 目录并与 ModelScope 仓库文件清单比对 ② 直接调用 `AutoProcessor.from_pretrained` 取真实报错 ③ 把 `fetch_model` 的成功判据改为**必需文件清单**（config.json / preprocessor_config.json / tokenizer* / *.safetensors） | `AutoProcessor.from_pretrained` 成功 |

### 坑 93-94 的元教训

1. **"无版本约束的依赖安装"等于把兼容性交给运气。** 依赖闭环解决了"缺什么装什么"，
   但引入了"装到哪个版本"的新问题。**闭环负责"存在性"，锁负责"版本正确性"，两者不可互相替代。**
2. **"模型文件完整"需要一份必需文件清单，而不是"目录里有 json"。**
   这与坑 61（空目录算就绪）、坑 70（跳过条件≠验收条件）、坑 89（配置与数据要一起发现）同族 ——
   已是第 4 次因"存在性判据过弱"而把失败推到很后面。**应统一为"必需文件清单"式校验。**

---

## 坑 95 —— 版本锁是**可执行契约**，却被我当成参考文档（2026-09-16 真机）

> **决定性证据（一次诊断调用全拿到）**：
> ```
> Using `use_fast=True` but `torchvision` is not available. Falling back to the slow image processor.
> AutoConfig  OK
> AutoTokenizer OK
> AutoProcessor FAIL -> TypeError: argument of type 'NoneType' is not iterable
> 模型目录: 13 个文件 / 1.7G —— **完全齐备**
> versions.lock 关键运行依赖: pyyaml numpy pybind11 protobuf scipy sentencepiece
>                              torchdata **torchvision** ftfy diffusers qwen_vl_utils einops av pandas attrs
> ```

| # | 现象 | 根因 | 处置（已落地） | 判据 |
|---|---|---|---|---|
| **95** | `ValueError: Processor was not found, please check and update your model file.` —— **错误信息把矛头指向"模型文件"**，而模型文件一个不缺 | 真因是**缺 `torchvision`**：`AutoProcessor` 在它缺失时**回退并抛出 `NoneType` TypeError**，MSMM 再包装成上面那句。**而 `torchvision` 恰好就写在 `versions.lock` 的"关键运行依赖"里** —— 锁是对的，是 bringup 从未消费它 | ① bringup **解析 `versions.lock` 的关键运行依赖行并逐个安装**（`awk` 取该行 → `\|`→空格 → 按 `import 模块名` 判断是否已装；含 `pyyaml→yaml`、`attrs→attr` 等名称映射）② `torchvision` 专项：**必须与 torch 配对（0.22.1）且用 `--no-deps`**，绝不让 pip 顺手改动 torch 破坏 NPU 栈 ③ `50_train.py` 新增**误导性错误**诊断规则，把该报错直接映射到"先查 torchvision" | bringup 输出 `[PASS] torchvision 就位（0.22.1）`；诊断规则自测 `DIAG_OK`（5/5 命中） |

### 坑 95 的元教训

1. **`versions.lock` 不是文档，是可执行契约。** 我把它当"给人看的参考"，于是它列了 `torchvision` 而代码里根本没装。
   正解：**lock 中每一类声明都要有一条对应的执行/校验代码**（这与坑 71「MindSpeed 写在 lock 里却没装」完全同因，
   **同一个错误犯了两次**）。
2. **误导性的错误信息会让排查方向错整轮。** `Processor was not found, please check and update your model file`
   让人去查模型文件，而文件一个不缺、真因是缺一个 Python 包。
   → **应对手段：把已知的"误导性错误 → 真因"映射写进诊断规则**（已加入 `50_train.py`），
   让下一次遇到它的人（或 Agent）直接命中答案，而不是重新考古。
3. **依赖闭环的能力边界**：它只会从 `No module named 'X'` 里提取缺失项。
   当缺失以 `TypeError`/`ValueError` 的形式伪装出现时，闭环无能为力。
   → 所以"闭环"必须与"**已知错误模式库**"配合使用，二者互补。

---

## 坑 96 —— mock 的 PASS 判据过弱，产生假 PASS（2026-09-16 真机）

> **本轮的双面性**：
> ```
> [PASS] torchvision 就位（0.22.1）
> [PASS] mock 数据集就绪
> [PASS] mock 训练跑通（依赖闭环补齐 0 次）     ← 判据说通过
> grep -c iteration /root/mock_run.log → 1
> mock 尾 6 行 → ChildFailedError / ERR99999   ← 日志却以致命错误结尾
> ```
> **一半是真进展**：torchvision 装好后，训练栈（数据加载 → 模型 → NPU 前反向）**首次真正跑起来了 1 步**。
> **一半是假 PASS**：判据太弱。

| # | 现象 | 根因 | 处置（已落地） | 判据 |
|---|---|---|---|---|
| **96** | mock 被判定"跑通"，实际**跑完 1 步后崩溃** | mock 的通过判据是 `grep -q 'iteration'` —— **只要日志里出现过 iteration 字样就算通过**。而训练完全可能在打印 1 步后崩溃。**这是"判据弱于它要回答的问题"家族的第 7 次**（前 6 次：坑 42/56/65/70/80/87） | 判据改为：**至少一次 iteration 且日志中不含致命错误标记**（`ChildFailedError` / `ERR99999` / `Traceback (most recent call last)`）；若"有 iteration 但以致命错误结尾"则**显式打印 WARN**（坑 96：假 PASS），不静默 | 本次这类日志应判为**未通过**并打印 WARN |

### 坑 96 的元教训

**"出现过成功标记" ≠ "成功"。** 一个运行过程可能在早期打印成功标记、随后才失败。
判据必须同时覆盖**正证据**（出现了成功的痕迹）与**负证据**（没有致命错误）。

统一规律（已在 `ROBUSTNESS.md` 沉淀）：**凡以日志为判据，必须同时检查"该出现的"和"不该出现的"。**
这条对本项目尤其重要 —— 训练日志里 `iteration` 会在崩溃前就打印出来。

---

## 坑 97-99 —— 流程顺序错 + 参数名连字符（2026-09-16 真机）

> **一次诊断调用把两个根因全拿到了**：
> ```
> A. mock 崩在第 1 步后：
>    CheckpointException ranks:{0}
>    FileNotFoundError: '/root/Qwen3.5-0.8B-dcp/iter_-000001/.metadata'
> B. DCP 转换失败：
>    checkpoint/convert_cli.py line 8: import jsonargparse → ModuleNotFoundError
>    源码权威参数名: checkpoint/common/hf_to_dcp.py:124-125 → "--hf-dir" / "--dcp-dir"  ← 连字符！
> ```

| # | 现象 | 根因 | 处置（已落地） | 判据 |
|---|---|---|---|---|
| **97** | DCP 转换每次都失败 | `convert_cli.py` 第 8 行 `import jsonargparse` —— **缺它直接 ImportError**，且错误包在 `__main__` 里不显眼 → 每次转换都在导入阶段就死 | 转换前自动 `pip install jsonargparse`；核心依赖清单加入 `jsonargparse` | DCP 转换能进入真正的转换逻辑 |
| **98** | 同一条命令**参数名错了三次**：`--load_path/--save_path` → `--hf_dir/--dcp_dir` → **`--hf-dir/--dcp-dir`（连字符）** | 权威来源是源码 `checkpoint/common/hf_to_dcp.py:124-125` 的 `add_argument("--hf-dir", ...)`。我三次都是"凭记忆/凭旧脚本推测"，从未去读源码 | 改为连字符形式，并在注释里写明**权威来源是源码 add_argument**。**纪律：参数名一律以源码为准** | 命令与源码逐字一致 |
| **99** | ★ mock 冒烟**必然失败**：训练第 1 步后崩溃在 `FileNotFoundError: <dcp>/iter_-000001/.metadata` | `phase3` 用 `--no-download` 调资产脚本 —— 那是**纯检查模式**，于是 **hf→dcp 转换永远不会执行**；而 `phase4` 的 mock 训练**第 1 步后就要从 DCP 检查点恢复**（`self.load()`）→ 必然崩。**流程顺序与模式必须匹配：mock 依赖 DCP，DCP 必须在 mock 之前产出** | `phase3` 改为 `--skip-coco`（**允许补齐**权重/DCP/转换，但不拉 19GB 图片；已存在的资产不会重下），并把"资产齐备"作为 PASS 判据 | bringup 输出 `[PASS] 资产齐备（含 DCP 权重与转换产物）`，随后 mock 才可能通过 |

### 坑 97-99 的元教训

1. **"检查模式"与"补齐模式"用错，会让整条流水线在后续阶段静默失败。**
   `--no-download` 本意是"只看看"，但它同时**禁掉了转换** —— 而转换是下游的硬前提。
   → **凡是有副作用的准备步骤，必须区分"只检查"与"去补齐"，并让下游阶段明确要求"已补齐"。**
2. **参数名必须读源码，不能凭记忆。** 同一条命令错三次（坑 90/92/98），共同点是"没去看 add_argument"。
   本项目的 `--help` 与源码都在手边，**这是纯粹的纪律问题，不是能力问题**。
3. **依赖链的"间接前提"要显式建模**：mock → DCP → jsonargparse。
   任何一环缺失，症状都出现在最下游（mock 崩溃），而根因在最上游（缺一个 pip 包）。
   → 这解释了为什么本项目的失败总是"症状与根因隔了 2-3 层"；**唯一可靠的办法是逐层补齐并逐层验证。**

---

## 坑 100-101 —— 我"修一个坏一个" + "可选"该按下游依赖定义（2026-09-16 真机）

> **本轮有一个漂亮的自我验证**：坑 96 的判据修复立刻兑现 ——
> ```
> [WARN] mock 有 iteration 但**日志以致命错误结尾** → 判定为未跑通（坑 96：假 PASS）
> ```
> **假 PASS 被机制抓住了**，不是靠人注意。

| # | 现象 | 根因 | 处置（已落地） | 判据 |
|---|---|---|---|---|
| **100** | DCP 转换仍失败：`ImportError: docstring-parser package is required by _set_docstring_parse_options` | **我自己造成的**：上一轮把 `docstring_parser` 换成了 `jsonargparse`，而 `convert_cli` **两个都需要**（`jsonargparse` 是显式 import；`docstring-parser` 是 jsonargparse 的可选依赖，但在此路径上被强制要求）。→ **"修一个坏一个"**：修改时没有保留原有的有效部分 | 两个一起装：`jsonargparse` + `docstring-parser`（注意 pip 名 `docstring-parser` ↔ 模块名 `docstring_parser`）；核心依赖清单同步加入 | DCP 转换能进入真正的转换逻辑 |
| **101** | `[PASS] 资产齐备（含 DCP 权重与转换产物）` —— **但 DCP 目录是空的** | 我把 `weight_dcp` 归入 `OPTIONAL_ASSETS`，而 **mock 冒烟与 P5 都会 `self.load()` 从 DCP 恢复** → DCP 缺失时 `ASSETS_OK` 依然成立 → **又一个假 PASS**（"判据弱于问题"家族第 8 次） | `weight_dcp` 移入 `REQUIRED_ASSETS`。**"可选"必须按"下游是否依赖"来定义，而不是按"我们觉得它次要"。** 负向对照已通过：DCP 缺失时 `missing_required` 含 `weight_dcp(不存在)`，rc=3 | `ASSETS_INCOMPLETE ... weight_dcp(不存在)` |

### 坑 100-101 的元教训

1. **修改时要保留原有的有效部分 —— "替换"常常是错的。** 坑 100 里我把 `docstring_parser` **替换**成 `jsonargparse`，
   而正确动作是**追加**。**改依赖/改配置时，先问"我是在替换还是在追加"**，替换就必须说明为什么旧的那个不再需要。
2. **"可选"是一个关于下游的断言，不是一个主观判断。** 只要有任何下游阶段依赖它（mock、P5），
   它就是必需项。**判断依据应是"谁依赖它"，而不是"它看起来重不重要"。**
3. **本轮正面案例**：坑 96 的"负证据判据"在第一次运行时就抓住了假 PASS。
   **这验证了"判据必须同时检查该出现的与不该出现的"这条纪律的价值** ——
   机制一旦建立，就能持续捕获同族问题，而不再依赖人的注意力。

---

## 坑 102 —— 同一条命令的参数名，我改了 4 次；而"正确答案我本来就有"（2026-09-16 真机）

> ```
> usage: convert_cli.py [options] GenericDCPConverter [options] hf_to_dcp
>        [--config CONFIG] [--hf_dir HF_DIR] [--dcp_dir DCP_DIR]
> error: unrecognized arguments: --hf-dir ... --dcp-dir ...
> ```
> **权威证据来自"实际被调用入口自己打印的 `--help`"。**

| # | 现象 | 根因 | 处置（已落地） | 判据 |
|---|---|---|---|---|
| **102** | DCP 转换命令的参数名**被我改了 4 次**，其中第 2 次本来是对的，却被我"改坏" | 演进过程：<br>① `--load_path/--save_path`（凭记忆猜）→ 错<br>② `--hf_dir/--dcp_dir`（**旧机器 `a3_conv2.sh` 里真正跑通的命令**）→ **正确**<br>③ `--hf-dir/--dcp-dir`（我读了 `checkpoint/common/hf_to_dcp.py` 的源码）→ 错<br>④ `--hf_dir/--dcp_dir`（`convert_cli.py ... hf_to_dcp --help` 打印）→ 正确<br>**根因：`hf_to_dcp.py` 是独立脚本（自己的 argparse，用连字符），而我们调用的是 `convert_cli.py` 子命令包装（用下划线）。同名功能的不同入口可以有不同参数约定。** | 改回 `--hf_dir/--dcp_dir`，并在注释里**完整记录这 4 次演进与每次的依据**，避免后人重蹈。**纪律：`<实际调用方式> --help` 的输出，优先级高于任何源码阅读与旧脚本记忆。** | 命令与 `convert_cli.py ... hf_to_dcp --help` 的输出一致 |

### 坑 102 的元教训（本项目最贵的一课）

1. **"正确答案我本来就有，却被我改错了。"** 第 2 版的 `--hf_dir/--dcp_dir` 来自**已在旧机器上成功运行过的脚本**
   （`a3_conv2.sh`，里面逐字保存着当时的命令）。我以"读了源码更权威"为由把它改掉，
   而实际读的是**另一个入口**的源码。
   → **纪律：把"已验证成功的原始命令"视为不可轻易改动的资产；要改它，必须先有同等强度的证据。**
2. **`--help` 是唯一权威，且必须是"实际调用方式"的 `--help`。**
   `python -m checkpoint.convert_cli GenericDCPConverter hf_to_dcp --help` 与
   `python checkpoint/common/hf_to_dcp.py --help` 是**两个不同程序**的 `--help`。
3. **本轮唯一的"好"消息**：修好的两个判据各自兑现了价值 ——
   ① 坑 96 的负证据判据当场抓住假 PASS；
   ② 坑 101 修的 `weight_dcp` 必需性，让 phase3 诚实地报出
      `ASSETS_INCOMPLETE missing_required=weight_dcp(不存在)` + `[WARN] 资产未完全齐备（rc=3）`。
   **机制在持续工作，而人（我）仍在犯同类错误。** 这正是把纪律写成机制的理由。

---

## 坑 103-104 —— ★★ 硬红线只写在文档里，没人检查（2026-09-16 真机，最危险的一类）

> **本轮的发现比前面所有坑都严重**：P5 正在跑一个**无效的训练**。
> ```
> env.json: recommended_profile = {"chip_model":"9382", "hbm_gb_per_chip": **null**,
>                                  "die_count": 2, "profile":"910b4_low_mem",
>                                  "world_size":**1**, "mbs":2, "gas":4}
> train.log:  global batch size: **4**      ← 不是 8！
> ```
> **真实硬件是 64GB/chip × 2 die（应为官方几何 world=2/mbs=4/gas=1 → GBS=8）。**
> 若没抓住，我们会拿一个 **GBS=4** 的 loss 去和 **GBS=8** 的官方基线对比，并得出错误结论。

| # | 现象 | 根因 | 处置（已落地） | 判据 |
|---|---|---|---|---|
| **103** | P0 档位选择掉到**最差档**（`910b4_low_mem`, world_size=1），而硬件其实是官方几何档 | HBM 探测**只有一条路径**：`npu-smi info -t memory -i <id> -c 0` + 单一正则。真机上该命令无可解析输出 → `hbm_mb_per_chip=null` → 档位匹配条件（`hbm_gb>=60 and die>=2`）不成立 → 落到最后的兜底档。**单一路径的探测 = 把环境适配押在一个命令上** | **三路回退**（可靠性递增）：① `npu-smi info -t memory`（原路径，多模式）② **`npu-smi info` 主表的 `used / total` 列**（实测稳定输出 `3132 / 65536`，取所有 chip 最大 total）③ **`torch_npu.mem_get_info()`**（完全不依赖文本解析，最可靠）。并在 env.json 里记录 `hbm_source` 便于溯源 | `hbm_gb_per_chip = 64` → 档位 = `910c_dual_die_official_geometry`, world_size=2 |
| **104** | ★★ **GBS=8 是项目硬红线，但代码里从来没有人检查它** —— 训练以 GBS=4 跑起来，无任何报警 | 红线只写在文档/任务书里（"GBS 必须 = 8"），而**没有任何代码断言它**。加上坑 103 的档位失误，两者叠加就产生了一次"看起来正常、实则无效"的训练 | 把红线**变成可执行断言**：`50_train.py` 启动前读配置的 `micro_batch_size`/`gradient_accumulation_steps`，算 `GBS = mbs × gas × world`，**≠8 即拒执行**（rc=3）并打印诊断（含"检查 env.json 的 recommended_profile"）；仅 `--allow-gbs-mismatch` 可绕过（且明确标注结果不可用于官方对标）。另加 `--world-size` 人工纠正通道 | 本地 dry-run 实测：`world=1 → GBS 4 → rc=3 拒绝`；`world=2 → GBS 8 → OK` |

### 坑 103-104 的元教训（本项目最重要的一条）

**"写在文档里的约束"等于没有约束。** 本项目有多条硬红线（GBS=8、不改最终配置、不改 SK04 凭证、
同源数据、数字必须脚本解析），而在此之前**它们只存在于文字中**，没有任何代码/判据强制执行。
结果就是坑 104：一条被明文禁止的配置（GBS=4）静默地跑了起来。

> **纪律：每一条硬红线都必须有一条可执行的断言，并且该断言必须在违规时"大声失败"。
> 红线不是给人读的，是给代码执行的。**
>
> 这一条与 INV-2（判据必须可证伪）互补：
> INV-2 管"判据不能假通过"，本条管"约束不能只挂在墙上"。

---

## 坑 105-106 —— 端口冲突 + 探测方法本身错了（2026-09-16 真机）

> **本轮的好消息**：坑 103/104 的修复双双兑现 ——
> ```
> 坑103: hbm_mb_per_chip=65536  hbm_source=npu_smi_table
>        profile=910c_dual_die_official_geometry  world_size=2 mbs=4 gas=1
> 坑104: 几何自检 : mbs=4 × gas=1 × world=2 = GBS 8（官方红线 = 8）✓
>        torchrun --nproc_per_node 2      ← 正确的官方几何
> ```
> **新失败**：`Address already in use`

| # | 现象 | 根因 | 处置（已落地） | 判据 |
|---|---|---|---|---|
| **105** | P5 启动失败：`Address already in use`；而我用来清理的 `pkill -f '50_train.py'` **输出为空**（说明执行异常） | ① 上一次运行的残留进程仍占用端口 6111 ② **坑 72 复发（破坏性版本）**：那条 SSH 命令行本身含 `50_train.py` 字符串 → `pkill -f` **匹配并杀掉了自己的 shell**，所以清理没生效、也没输出 | **机制化解法：不再依赖"记得先清理"** —— `50_train.py` 启动前**探测端口，被占用就自动递增寻找空闲端口**（最多试 200 个），并打印"改用端口 N" | 实测：占用 6111 → 自动改用 **6112**，训练命令正确带 `--nproc_per_node 2` |
| **106** | 为坑 105 写的端口探测**自测失败**（已占用却报"空闲"） | `_port_free()` 用了 `bind` + **`SO_REUSEADDR`** —— 在 **Windows 上 SO_REUSEADDR 允许重新绑定一个已被占用的端口** → 探测**100% 误判为空闲**。（Linux 上无此行为，所以这是个平台差异陷阱） | 改为两段判断：① **先 `connect_ex`**：能连上 = 有服务监听 = 已占用（跨平台最可靠）② 再 `bind` 且**刻意不设 SO_REUSEADDR** | 父进程占住 6111 并 listen → 子进程正确判定"被占用"并换端口 → `PORT_AUTOPICK_OK` |

### 坑 105-106 的元教训

1. **"记得先清理"不是工程方案，"自动规避冲突"才是。** 端口冲突的根治不是让每次调用者都记得 `pkill`，
   而是让程序**自己选择可用资源**。这与本项目的一贯结论一致：**凡是依赖人记得的，都会在某次忘记时失败。**
2. **平台差异会藏在最基础的系统调用里。** `SO_REUSEADDR` 在 Windows 与 Linux 的语义差异，
   让一个"看起来标准"的端口探测在 Windows 上**恒返回错误答案**。
   → 纪律：**探测类代码写完必须立刻用一个"已知占用"的样本自测**（本次正是靠自测才发现）。
3. **测试脚本自己也需要前置条件检查。** 第一版测试因为 env.json 缺 `capabilities.can_train`
   而被脚本提前 `return 3`，**根本没走到被测代码** → 给出假 FAIL。
   → 这与"判据要绑住真实路径"同族：**测试若没触达目标代码，它的结论没有意义。**

---

## 坑 107 —— 诊断规则把"正常日志里也有的字样"当特征（2026-09-16 真机）

| # | 现象 | 根因 | 处置（已落地） | 判据 |
|---|---|---|---|---|
| **107** | 训练失败时自动诊断报 **`● 配置校验失败`**，把排查方向带偏 | 该规则写成 `Configuration Details\|ValidationError` —— 而 **`Configuration Details` 出现在每一份正常训练日志里**（它就是配置 dump）。于是**任何**失败都会被诊断成"配置校验失败" | 收紧为 `ValidationError\|Configuration Details.*(?:error\|Error\|invalid\|missing)`：**只有"该字样 + 错误语境"同时出现才算命中**。并按同一原则新增 ACL 507018 的诊断规则（含"不要用 `pkill -f` 直接匹配名字"的提示） | **双向自检**：3 条正常日志（含 `Configuration Details`、正常 `iteration`、`Saved checkpoint`）**零误诊**；6 条真实故障**全部命中** → `DIAG_BIDIRECTIONAL_OK` |

### 坑 107 的元教训

**诊断规则的匹配串必须是「只在该故障下才出现」的字样。** 否则它会把正常日志也判成该故障，
而"错误的诊断方向"比"没有诊断"更浪费时间——它会让人去查一个根本不存在的问题
（这与坑 95 的 `Processor was not found` 误导是同一类，只不过这次误导来自**我自己的规则**）。

> **纪律：每加一条诊断规则，必须同时跑「正常样本不得命中」与「故障样本必须命中」两个方向。
> 只测一个方向的规则等于没测。**（这是 INV-2 在"诊断规则"上的具体化。）

**本轮状态**：P5 已在正确几何下（world=2 / GBS=8 / 自动选端口 6112）启动，跑到第 2 步后
报 `ACL stream synchronize failed, error code:507018`（两个 rank 都有）→ 下一轮按新诊断规则
的①~④排查（优先怀疑**上一次崩溃留下的设备残留进程**）。

---

## 坑 108 —— dp2 下 HCCL scatter 的 AI CPU kernel 失败（2026-09-16 真机，环境级阻塞）

> **排查结论：既不是显存，也不是设备残留。** 证据链完整：
> ```
> npu-smi 进程表: No running processes found        ← 无残留（排除①）
> 清理后重试在同一行复现                             ← 排除"脏状态"（排除②）
> nproc_per_node=2 与 device_count=2 一致           ← 映射正确（排除③）
> 真正根因（ACL 报错上方）:
>   524: UserWarning: HCCL doesn't support gather at the moment. Implemented with allgather instead.
>   530: AI CPU kernel execute failed, device_id=0, stream_id=44, task_id=38,
>        soName=libscatter_aicpu_kernel.so, funcName=HcclLaunchAicpuKernel, errorCode=0x2a
>   531: AI CPU kernel execution failed ... HcclLaunchAicpuKernel
>   EE9999: rtDeviceSynchronizeWithTimeout execution failed, reason=aicpu exception
> ```

| # | 现象 | 根因 | 处置 | 判据 |
|---|---|---|---|---|
| **108** | **P5（dp2）在第 1 个 iteration 之前崩溃**，`ACL stream synchronize failed 507018`；而 **mock（1 rank）能跑完 100 步** | **dp2 下 MSMM 调用了 HCCL 不支持的 `gather`** → 框架回退为 `allgather` → 该回退路径启动 `HcclLaunchAicpuKernel`（`libscatter_aicpu_kernel.so`）→ **在本机 CANN 9.1.0-beta.3 上 AI CPU kernel 执行失败**（`errorCode=0x2a`）。1 rank 不触发跨 rank 集合通信，所以 mock 不受影响 | **待定方案（下一轮）**：① 定位调用 `gather` 的 Python 代码（rank1 的 Traceback 正在其中），判断它是否可规避（例如仅用于调试打印）② 若属框架必需 → 评估"dp1 + mbs8（GBS=8）"作为**降级路径**（几何≠官方，仅窗口可比，须如实标注）③ 记录为**本机环境限制**（CANN β 版对 HCCL scatter 的支持问题） | 定位到调用点；或给出可运行的降级几何并标注降级 |

### 坑 108 的元教训

1. **"ACL 层报错"几乎从不是根因。** 真实错误在它**上方 50 行**（`AI CPU kernel execute failed ... HcclLaunchAicpuKernel`）。
   诊断规则因此写明了"看报错之前的第一条真实异常"——本次正是靠这条纪律定位的。
2. **"1 rank 能跑、2 rank 不能跑"是最有价值的对照实验。** 它把问题域从"模型/数据/权重"一次性收缩到
   **跨 rank 集合通信**。**在排查中主动制造对照（改一个变量重跑）比读更多日志更有效。**
3. **β 版 CANN 的"支持范围"本身就是风险项。** `HCCL doesn't support gather` 这条 Warning 在 GA 版上
   可能没有，或回退路径可用；β 版上回退路径的实现有缺陷。
   → 这也是本项目把"CANN 预发布版"标注为风险（坑 43）的现实兑现。

### 坑 108 的精确调用链（完整 Traceback，两个 rank 一致）

```
train_engine.py:64        self.iteration, ... = self.load()        ← 启动时加载 DCP 权重
train_engine.py:363       self.load_checkpointer.load(...)
dcp_checkpointer.py:257   dcp.load(...)
torch/.../state_dict_loader.py:234   distW.reduce_scatter("plan", local_step, global_step)
torch/.../checkpoint/utils.py:217    result = self.scatter_object(all_results)
torch/.../checkpoint/utils.py:162    dist.scatter_object_list(...)
torch/.../distributed_c10d.py:3627   obj_tensor_size = torch.tensor([0], ..., device=pg_device)
→ RuntimeError: ACL stream synchronize failed, error code:507018
```

**关键事实（用于定责与选方案）**：
- **三份配置的 `load:` 完全相同**（`/root/Qwen3.5-0.8B-dcp`）——包括旧机器上 dp2 跑通的 `e4b_final100.yaml`。
  → **配置不是原因，环境是原因**（旧机 CANN 9.1.0 GA，本机 9.1.0-beta.3）。
- mock（1 rank）用同一份 `load:` 能跑完 100 步 → **1 rank 不触发该集合通信**，所以问题被掩盖。
- 崩溃点在**启动加载权重阶段**，与模型前向/数据无关。

**可选路线（下一轮验证）**：
| 路线 | 做法 | 代价 |
|---|---|---|
| **A（首选）** | 在 `training:` 段加 `load_rank0_and_broadcast: true`（`train_engine.py:366` 已支持该开关）→ 改为「rank0 加载 + 广播」，**绕开 `scatter_object_list`** | 需确认该键被 schema 接受、且数值等价 |
| B | 用 **dp1 + mbs8**（GBS 仍 = 8）跑通 | **几何 ≠ 官方（dp1 vs dp2）→ 只能窗口口径可比，必须如实标注** |
| C | 放弃 dp2，把该机记为"CANN β 版下 dp2 不可用" | 无法复刻官方几何 → 精度只能宣称窗口口径 |

> **诚实边界（无论选哪条）**：本机 `step1 loss` 与官方的对照**必须在报告中同时说明几何/环境差异**，
> 不得因"能跑起来"就宣称逐点可比。

### 路线 A 的验证结果（msmm 源码，已确认可行）

```
training_args.py:275      load_rank0_and_broadcast: bool = field(...)      ← schema 有定义（有效键）
dcp_checkpointer.py:212   load_rank0_and_broadcast: bool = False            ← 函数参数默认 False
dcp_checkpointer.py:251   if load_rank0_and_broadcast:
dcp_checkpointer.py:252       rank0_load_and_broadcast_dcp_weights(...)    ← ★ 绕开 dcp.load()
dcp_checkpointer.py:257   else: dcp.load(...)                              ← 崩的就是这条
hf_checkpointer.py:215    "rank 0 loads the checkpoint, then broadcasts to other ranks"
```

**结论**：该开关**正是为"避免每 rank 各自读取/集合通信"而设计**的，语义清晰（rank0 加载 → 广播），
可绕开 `scatter_object_list`。→ **路线 A 可行**。

**但**：把开关加入配置后出现 `ValidationError`（配置校验失败）—— 下一轮需拿到**具体校验错误**
（很可能是字段层级不对，或该字段有配套要求；注意 `trainer.py:204` 明确要求
`training.load` 不能为 None，我们已设置，故不是这条）。

### ★ 路线 A 成功 —— 并直接拿到了决定性结果（2026-09-16）

```
插入: load_rank0_and_broadcast: true  (indent=2)     ← 复用锚点缩进
YAML_OK  training.load = /root/Qwen3.5-0.8B-dcp
YAML_OK  training.load_rank0_and_broadcast = True
OVERRIDE_VALIDATED
---
iteration  1/100 | consumed samples: 8 | global batch size: 8 | loss: 1.924620E+00 | grad norm: 134.300
iteration  2/100 | loss: 1.791238E+00
iteration  3/100 | loss: 1.783001E+00
（150 秒内到 97/100，零异常）
```

**结论**：`load_rank0_and_broadcast: true` **成功绕开了 `dcp.load()` 的 `scatter_object_list` 路径**，
在**官方几何**（`world=2/mbs=4/gas=1/dp=2`，`GBS=8`）下跑通，且 **step1 loss = `1.924620`
对官方 `1.924621`（打印精度内一致，相对偏差 ≈5.2×10⁻⁷%）**。

---

## 坑 109 —— 用"写死的缩进"往 YAML 里插行（2026-09-16 真机）

| # | 现象 | 根因 | 处置（已落地） | 判据 |
|---|---|---|---|---|
| **109** | 给配置加 `load_rank0_and_broadcast: true` 后训练立刻失败：`ConfigValidationError: YAML parsing failed - while parsing a block mapping` | 我用 `re.sub` 锚定 `load:` 行插入新行，但**缩进写死成 4 空格**；而该文件里 `training:` 的子键实际是 **2 空格** → 新行成了 `load:`（标量）的子节点 → YAML 结构非法。**看似"配置字段问题"，实为"文本编辑问题"** | ① 插入时**复用锚点行匹配到的缩进**（`m.group(1)`），而不是写死；② **插入后立即用 `yaml.safe_load` 校验**，并断言目标字段真的生效（`training.load_rank0_and_broadcast is True`），**校验不通过就不启动训练**（不浪费一次运行） | `OVERRIDE_VALIDATED`；且失败时会打印 `OVERRIDE_INVALID` 并中止 |

### 坑 109 的元教训

1. **"配置校验失败"未必是配置语义问题，也可能是我的文本编辑破坏了结构。**
   `while parsing a block mapping` 是**YAML 语法**错误，不是 schema 错误 —— 先看错误类型再定方向。
2. **凡程序化修改结构化文本（YAML/JSON/代码），必须"改完立即解析校验"。**
   并且要在**下一步动作之前**校验（本例就是"校验通过才启动训练"），
   否则一次语法错误会白耗一个完整实验周期。
3. **缩进不要写死，要复用上下文。** 这与坑 92/102（参数名要读源码而非记忆）同族：
   **凡是能从现场推导出来的，就不要凭假设写死。**

---

## 坑 110-125 —— 判定链自己恒红 + 多来源不同源 + 打包源错 + 收尾保存崩 + 陈旧产物 + 读证据截断 + 阶段判据与产物契约不符 + 取证混装 + 栈混装 + 声明≠实装 + 耦合变量（2026-09-16 新机器复盘）

> **发现路径**：新机器 P5 已跑通官方几何 100 步（`global batch size: 8`、loss 逐点偏差 Mean 0.0582%、
> 超 2% 步数 0/100），但 P7 判定给出 `verdict=NOT_COMPARABLE level=none color=red neff=4/8
> deviations=1 mismatch=[dp]` —— **症状完美伪装成"配置偏离官方几何"**，实际是判定器自身的 bug。
> 对照 `examples/judge/fp.json`（A3 的冻结验收证据）才看清：A3 的真实裁决是
> `level=pointwise_feasible color=yellow N_eff 8=8 mismatch=[] world.run_source=config
> open_gates=[sample_order]` —— **A3 也不是绿的**。

| # | 现象 | 根因 | 处置（已落地） | 判据 |
|---|---|---|---|---|
| **110** | ★★ **P7 对任何多卡运行必然 `NOT_COMPARABLE / neff=4/8`**，看起来像"精度不达标或配置偏离" | `scripts/70_judge.py` 在带 `--config` 时**硬编码**把 `--world-size 1` 传给 `fingerprint_cfg.py`；而指纹里是 `world = world_cli if world_cli is not None else dp_dim["value"]` → **硬编码的 1 会覆盖配置里显式写的 `data_parallel_size: 2`**，N_eff 恒 = 1×mbs×gas = 4 ≠ 8。而 `run_from_zero.sh` 的 P7 **正是带 `--config` 调用** → **"从零复现"链路的最后一环永远是红的**（A3 那次是走日志 details 才拿到 8/8，所以这个 bug 一直没暴露） | ① 删掉硬编码；新增 `--world-size`（缺省 `None`），改为**同源解析**：配置显式声明 dp/world → **不传**（让指纹 `source=config`，与 A3 验收记录同构）；否则用日志 `global batch size ÷ (mbs×gas)` 反推；② **多来源互相矛盾 → 拒判 `rc=2`（WORLD_CONFLICT）**，绝不挑一个用；③ 都拿不到 → **拒判 `WORLD_UNKNOWN`，而不是默认 1**；④ 新增 `--selftest`（8 用例，含 3 个必须被拒的坏例） | `WORLD_RESOLVE_SELFTEST_OK cases=8`；本地端到端：修前 `neff_run=4 gbs_run=4 mismatch=[dp]` → 修后 `neff_run=8 gbs_run=8 mismatch=[] level=pointwise_feasible`（**与 A3 冻结记录逐字段一致**）；显式传 `--world-size 1` 现在 `rc=2` 且打印 `WORLD_CONFLICT` |
| **111** | 同族第二处：**同一语义有两个来源、却无人交叉核对**。① `50_train.py` 的 GBS 红线自检用**正则扫文本**取 mbs/gas —— 配置若是非法 YAML（坑 109 的缩进损坏），正则照样匹配到数字 → 红线"检查通过"，训练器几分钟后才抛 `ConfigValidationError`；② `20_plan_migration.py` **先落盘再校验**，且回读不一致只记进 `warnings`，脚本仍返回 0 并打印 `PLAN_OK` | **运行条件读原始文本、验收条件读解析结果 → 两者不同源**。这正是"跳过条件/验收条件/运行条件/闭环收敛条件必须同源、且等于真正需要成立的东西"被违反的方式：每个环节**单独看都"有检查"**，但检查的不是同一件事 | ① `20_plan_migration.py`：改为**先校验后落盘**（YAML 解析失败或任一字段回读不一致 → `rc=3` 拒产，**磁盘上不留坏配置**），回读清单补入 `data_parallel_size`/`load_rank0_and_broadcast`/`train_iters`，新增 `PLAN_GEOMETRY` 速览行；② `50_train.py`：GBS 自检改为**解析 YAML 取值**，解析失败 → **启动前拒执行**；并新增"配置声明的 dp ≠ 本次 `--world-size` → 拒绝启动"的同源断言；③ 无 pyyaml 时的正则降级**显式标注来源**（`取值来源 regex_fallback(no_pyyaml)`），不假装等价 | 坏模板 → `rc=3` 且 `train_config.yaml` **不存在**；非法 YAML 喂给 P5 → "在启动训练前拒绝执行" `rc=3`；配置 dp=2 + `--world-size 1` → "不一致 —— 拒绝启动" `rc=3`；正例 dp=2 + world=2 → `几何自检 : mbs=4 × gas=1 × world=2 = GBS 8（取值来源 yaml）` |
| **112** | ★ **交付 zip 静默缺少全部修复** —— `pack_skill.py` 打印 `ZIP_OK 文件数: 63`，但包里 `70_judge.py` 仍是**旧版**（无同源解析） | `pack_skill.py` 的 `SKILL` 常量**写死**为 `…\C4AI复赛_交付材料\06_Skill\qwen35-ascend-migrator`（一份**副本**），而我实际编辑的是 `…\转交给codex的内容\04_核心资产\qwen35-ascend-migrator_整合版`（**活副本**，96 文件）。**打包源 ≠ 工作副本** → 改完活副本，打出来的仍是陈旧副本。**两条链路各自都"成功"**，没有任何一层会报错（与坑 110/111 完全同族：同一事实的两个来源无人核对） | ① 新增 `交付物生成脚本/build_handover_skill_zip.py`：打包源**相对脚本定位**（`..\qwen35-ascend-migrator_整合版`），不写死绝对路径；② **打包后回读 zip 做标记校验**（本轮 6 处修复的标记 + 禁止字样回归守卫 + 冻结 `verdict_id`），校验不过 → `rc=1` 且**不落盘**到交付目录（沿用坑 111 的"先校验后落盘"）；③ 检测到陈旧副本与本副本不一致时**显式告警**并打印两者文件数；④ 运行态产物（`out/preflight`、`out/scene`、`tests/tmp` 残留）明确排除，只保留 `out/README.md` 与 `.gitkeep` | `HANDOVER_ZIP_OK … files=63`；负向：标记缺失 → `FATAL zip 内容校验未通过` `rc=1` 且交付目录**不产生新 zip**（实测本条即由该负向路径发现：先报出 `docs/PITFALLS 缺 **112**` 与 `fp.json 缺 606056…`） |
| **113** | ★★ **P5 跑完 100 步后进程仍以非零码退出**：`iteration 100/100 \| consumed samples: 800 \| loss: 1.447652E+00` 正常打印，紧接着 `ACL 507018` → `SIGABRT` → `ChildFailedError` | 训练结束时的**收尾保存**（`train_engine.py:315-317` 的 `if args.training.save: self.save(...)`）走 `dcp.save` → `state_dict_saver.py:350 SavePlan = distW.reduce_scatter("plan", ...)` → `scatter_object_list`；日志里紧邻的那行 `UserWarning: HCCL doesn't support gather at the moment. Implemented with allgather instead.` 已指明根因。**坑 108 的 `load_rank0_and_broadcast` 只覆盖 load 路径**，save 路径在 MindSpeed-MM 里没有对应开关 —— 所以"载入修好了"给人一种"这个 CANN 缺陷已经绕过"的错觉 | ① 诊断规则补 `HCCL doesn't support gather` / `SavePlan = distW.reduce_scatter` 专属分支，直接指向"收尾保存"而非"训练失败"；② **配置级规避**：`trainer.py:439-448` 规定 `save_format != dcp` 时需 `no_save_optim` 与 `no_save_rng` 同为真（本配置已满足）→ 置 `training.save_format: hf` 绕开 DCP plan 广播；② **由环境自动决定，不靠人记得改配置**（坑 104/108 的同一模式）：P0 探测到 CANN 含 beta/RC/dev → `recommended_profile.save_format = "hf"`，P2 写入配置并**校验前置条件**（`no_save_optim`/`no_save_rng` 必须同为真，否则 trainer 会**静默回退 dcp** → 崩溃复发；该守卫已加进 P2 回读校验）；③ 诊断规则补专属分支 `central_plan: SavePlan`，直接指向"收尾保存"而非"训练失败"（**刻意不用 `HCCL doesn't support gather` 作判据** —— 那是良性 UserWarning，拿它当特征就是坑 107 复发；由新增的 `50_train.py --diag-selftest` 用"健康日志必须 0 命中"守住）；④ 兜底：不需要 checkpoint 时 `training.save: null`（`train_engine.py:315` 的 `if args.training.save:` 守卫为假 → 完全不保存）；⑤ **验收口径**：`成功 = rc=0` 这条红线必须坚持 —— 100 步数值完整 **不等于** 运行成功 | ★ **真机 `SAVEFIX_OK`**（3 步对照：iteration=3 / ACL 507018=0 / ChildFailedError=0 / HCCL gather 警告=0，且 `HuggingFace checkpoint saved at ./save_path/iter_0000003 successfully!`）；本地 P2 正例 rc=0 且 `save_format: hf` 落进生成配置、负例（`no_save_optim: false`）rc=3 拒绝（`P2_SAVEFORMAT_TEST_OK`）；`DIAG_SELFTEST_OK samples=13 rules=12 uncovered=0` |
| **114** | ★ **机器上的 `out/plan/train_config.yaml` 是"坑 103 修复前"的陈旧产物**：`data_parallel_size: 1 / micro_batch_size: 2 / gradient_accumulation_steps: 4`（= `910b4_low_mem` 档），而真实运行态是 `dp=2 / mbs=4 / gas=1` | 该文件生成于 HBM 探测失败那一轮，**P2 产物没有任何新鲜度校验**。更隐蔽的是：`1×2×4 = 8` 与 `2×4×1 = 8` **GBS 完全相同** → 只看 GBS 红线**永远发现不了**；若拿它去判定，指纹会读到 `dp=1` → `mismatch=[dp]` → 被误读成"配置偏离官方几何"（正好又是坑 110 的同一张脸） | ① 判定链**主源改为日志内嵌的 `Configuration Details`（运行态真相）**，`--config` 仅作对照；② 两者不一致 → `STALE_CONFIG` 显式告警并**以日志为准**（不静默采信文件）；③ 日志未显式声明 dp 时，把配置的 dp 与"`gbs ÷ (mbs×gas)` 反推的 world"也比一遍（本例正是靠这一路才抓到）；④ 日志无 details 时**显式标注**"判定只能基于文件、无法自证" | 本地实测（造一份陈旧配置 + 一份运行态日志）：打印 `STALE_CONFIG … 配置=1 vs 日志=2` 且指纹 `n_eff_run=8 mismatch=[]`（**dp 不在 mismatch 里**）；`WORLD_RESOLVE_SELFTEST_OK cases=9` |
| **115** | ★ **我把"与 A3 完全一致"误报成"多一个未闭合门"** —— 读 A3 冻结证据时把 `open_gates` 记成只有 `sample_order`，把 `data_same` 记成 `declared_same_file_byte` | **两个读证据错误叠加**：① **把指纹层的 gates 当成裁决层的 gates** —— `fingerprint_cfg` 的 `reachability.gates` 里 `data_identity` 被标成 `closed=true`（该层"不阻塞"的策略），而**最终裁决层** `judge_comparable` 对同一事实判为**未闭合**且更诚实（"基线文件字节不可得 → 同源未证实"）；两层同名不同义，混读必然得出相反结论。② **把被截断的打印行当完整值** —— `selfcheck_sk04` 打印 JUDGE_OK 行时截断到 150 字符，`declared_same_file_byte_unverified` 的后缀 `_unverified` 正好被切掉，于是"A3 是闭合的、我的是未闭合的"这个错误印象就成立了 | ① 引用冻结证据**一律直接读 JSON 字段**（`examples/judge/verdict.json` 的 `open_gates`/`data_same_source`），**不引用任何打印行**；② 把"截断"列为危险源：凡 `[:150]` 之类的打印都属于**摘要**，只能供人眼扫读、不得作为证据；③ 在里程碑里写死 A3 与本次的**逐字段对照表**，避免下次凭印象比较 | 逐字段对照：`level`=`pointwise_feasible`、`open_gates`=`[data_identity, sample_order]`、`deviations`=0、`unverified`=0、`data_same_source.status`=`declared_same_file_byte_unverified`、`N_eff`=8/8 —— **全部一致**；`verdict_id` `606056b3…` vs `22b3c135…` **不同且必须不同** |
| **116** | ★★ **P7 阶段从始至终不可能通过** —— 真机上 P7 报 `[FAIL] 判据不满足: verdict.json 存在 且 verdict / verdict_id 均非 None`，而实际裁决完全合格（`NEEDS_EVIDENCE / pointwise_feasible / yellow`） | ① **判据与产物契约不符**：`judge_comparable.py` 的 `verdict.json` **按设计只含证据字段**（`verdict_id`/`reachability.level`/`open_gates`…），**不含结论名**；结论名只在它 stdout 的 `JUDGE_OK verdict=…` 行（`70_judge.py` 的注释就明写着这点，并把它落到 `judge_summary.json`）。② **更隐蔽的是判据里的 `\x27`**：该判据写作 `python3 -c "…open(\x27out/judge/verdict.json\x27)…"` —— `\x27` 是**被 JSON 转义污染**的写法（同一文件其它阶段用的是正常的 `'`，如 P0 的 L159），经 `run_stage` 的 `eval "$judge"` 后 python 收到**字面 `\x27`** → SyntaxError → 判据恒不成立。两种错误叠加，结果是"P7 永远 FAIL，且看起来像裁决不合格" | ① 判据拆成两半：**证据非空壳**看 `verdict.json.verdict_id`，**结论非空壳**看 `judge_summary.json` 的 `verdict`/`level`/`verdict_id`；② 统一转义写法（外层 `\"`、内部普通 `'`），与 P0 阶段一致；③ 闸门 **G18** 加"`run_from_zero.sh` 不得含 `\x27`"的静态回归守卫 | 真机：`[PASS] 5s —— out/logs/P7.log`，并打印 `>>> verdict: NEEDS_EVIDENCE / level: pointwise_feasible / color: yellow / id: 96eac899…` |
| **117** | ★ **P6 阶段从始至终不可能通过**：驱动要 `out/bench/round_1.json`，而 `60_bench.py:81` 写的是 `round_{round}_{tag}.json` → 真实产物是 **`round_1_baseline.json`** | 判据里的产物名是**凭印象写的**，不是从产出脚本读出来的。**G11 只校验"驱动传的 `--flag` 被该脚本接受"，不校验"判据里的产物名是脚本真正写出的"** —— 契约的另一半无人守 | ① 产物名从产出脚本推得（`--round 1 --tag baseline` → `round_1_baseline.json`），并用变量 `BENCH_ART` 单点声明；② 闸门 **G18** 建立**产物↔产出者登记表**：每个 `STAGE_ART=` 必须登记其产出脚本，且脚本源码中必须能查到该名字（或声明的格式片段） | 真机：`P6 PASS 8s`（修前 FAIL） |
| **118** | ★ **降级账本跨运行累积、且不判新鲜度**：`--from P6` 续跑时报告 4 条降级，实际写于 **6.5 小时前**，其中"缺少基础命令: grep,sed,awk""torch_npu 不可用""converted=False coco=0/118287"在**当轮已完全不成立**（驱动自己到处在用 grep、训练刚跑通、P4 刚 PASS） | 账本由 **PRE 阶段一次性写出**，续跑（`--from`）**不会重写它**；报告只数条数 → **把历史状态当当期事实**。这**比空账本更有害**：空账本会触发 INV-4 的失败判据，陈旧账本则静默误导 | ① 报告把账本拆成"**本轮** / **陈旧·非本轮**"并逐条标注 `ts`，陈旧项**不计入当期降级**；② 显式提示"如需当期口径请 `--from PRE` 重跑预检" | 修前真机输出 `DEGRADATIONS=4` 与条目 `ts=2026-09-16T13:18:06` **自相矛盾** |
| **119** | ★ **修 118 的第一版修复自己又错了**：报告打出 `DEGRADATIONS=4(本轮 4 / 陈旧 0)`，而条目 ts 明明是 13:18、本轮是 20:01 | **两个来源的时间戳格式不同，却直接做字符串比较**：账本写 ISO `2026-09-16T13:18:06`，`START_TS` 来自 `date '+%Y-%m-%d %H:%M:%S'` 是 `2026-09-16 20:01:52`；字典序下 **`'T'`(0x54) > `' '`(0x20)** → 旧的反而被判成"更新" | 比较前**归一**：`str(ts).replace('T',' ')[:19]`，并在报告段与计数段共用同一 `_norm()` | **靠"驱动自己的打印与证据自相矛盾"发现** —— 若当时只打印总数、不打印 `本轮/陈旧` 分解，这个 bug 会一直藏着。→ 纪律：**凡是"新旧/先后"判断，先统一格式再比**；打印分解值而不只打印结论 |
| **120** | ★ **P6 的判据弱于它的名字**：阶段叫"性能基准(**50-100 口径**)"，判据却只是"某个 json 存在" —— 而那份产物自己写着 `scope: operator-micro(v0); end-to-end 由 MindSpeed-MM 侧执行`，**根本没测端到端性能** | 真正能算 50–100 窗口指标的 `95_extract_series.py`（含 median/mean/throughput）**从未被驱动调用过**（`60_bench.py` 连 `--log` 参数都没有）。于是"文件在=PASS"，性能到底测没测出来无人过问 —— 与坑 116/117 同族，但这次是**语义层面的弱判据**（不只是名字写错） | ① `95_extract_series.py` 新增 `--summary-json`，把窗口指标落盘成可断言的 JSON（键名正好是驱动原本就在找的 `steps_used`/`median_ms`/`mean_ms`/`samples_per_s`/`window` —— 意图本来就有，只是从未接线）；② P6 阶段改为**同时产出算子里微基准 + 窗口指标**，判据断言 `steps_used≥1 and median_ms and samples_per_s`；③ 打印真实字段而非空 dict | 真机：`>>> bench(window 50,100): steps_used=51 median_ms=929.1 mean_ms=1054.79 samples_per_s=8.61` + `>>> bench(micro): rows=3 backend=Ascend NPU (torch_npu OK)`，`P6 PASS` |
| **121** | ★ **闸门 G11 报了假阳性**：`60_bench.py(静态) 未声明: --log,--summary-json,--window` —— 而 `--summary-json/--window` 明明是**同一行里第二条命令**（`95_extract_series.py`）的参数 | G11 用 `re.search(r"python3\s+scripts/(\S+\.py)([^\"]*)", line)` 逐行匹配，`rest` 会**吃掉本行后续所有内容** → 一行里串了 `a.py … && b.py …` 时，b 的 flag 全被算到 a 头上。**这是"闸门自己出错"**：它比没有闸门更糟，因为它会训练人忽略闸门输出 | ① 驱动侧把两条命令**拆成两行**（同一引号内换行，`bash -c` 仍顺序执行，G11 的行级匹配也随之正确）；② 闸门侧按 `&&`/`\|\|`/`;` 切开 `rest`，只取属于本脚本的那一段（纵深防御） | `PREPUSH_OK 18/18`；负向：把两条命令写回同一行时 G11 仍能正确归属 |
| **121-b** | 修 121 时**我自己又引入一个崩**：`AttributeError: 'str' object has no attribute 'json'`，`PREPUSH_FAIL` 的收尾代码在 `if a.json:` 处炸掉 | 我在新加的 G18 里用 `a` 作为**循环变量**，把函数开头的 argparse 结果 `a = ap.parse_args()` **遮蔽**了 → 收尾 `a.json` 变成对字符串取属性。**闸门"跑完并打印了 PASS"之后才崩**，所以只看门的结果会以为一切正常 | 循环变量改名 `art18`；并纪律化：**在长函数里不要用 `a`/`d`/`f` 这类与外部命名冲突的单字母变量** | `PREPUSH_OK 18/18` 且 **rc=0**（此前是 rc=1 + traceback，容易被 `Select-Object` 截断而看不见） |
| **122** | ★ **我在取证时把"两次不同运行"的产物混装进同一个证据目录** —— 包内 `window_50_100.json` 声明窗口中位 **925.4 ms**，而同目录 `loss_series.csv` 复算出 **941.6 ms**，两者对不上 | 取证脚本 `step49_export_evidence.py` **按路径**收集（`out/bench/*` + `out/train/*`），而这些路径**每次运行都会被覆盖**：`out/train/loss_series.csv` 来自**首次** 100 步（19:13，收尾保存崩、rc≠0 那次），`out/bench/window_50_100.json` 来自**末次** 100 步（20:07，rc=0 那次）→ **跨文件数字不自洽，且差 1.7% 不刺眼、不细看发现不了**。**这正是我整轮在修的"同源违反"——我自己又犯了一次**（与坑 115 同类：都是"引用证据时不核对来源"） | ① 证据目录 `README.md` 增加「**文件来源与自洽性**」段落，逐文件标注来源路径与所属运行，并**显式列出已知不自洽项**；② `MAP.json` 的 `remote` 字段作为溯源依据；③ 新增复核主张 **C15**：**跨文件一致性必须被检查**，不一致时要求"有明确来源解释"，否则判 FAIL（此前只查单文件字段是否非空 → 假通过）；④ 长期修法：取证脚本应**按运行标签**收集（把 run tag 写进产物名），而不是按会被覆盖的路径 | 复核器 C15 由 **FAIL**（不自洽且无解释）→ **PASS**（差异由 `csv` 字段声明的不同来源解释，且已披露）；`REVIEW_OK 18/18` |
| **123** | ★ **软件栈"混装"没有任何判据看得见**：实测新机器 `import triton; triton.__version__` = **3.2.0**、dist `triton_ascend` = **3.2.2**（与 A3 官方配对一致）、dist **上游 `triton` = 3.5.0**；`site-packages` 里 `triton/`、`triton-3.5.0.dist-info`、`triton_ascend-3.2.2.dist-info` **共存** | ① **探测缺口**：`00_probe_env.py` 只取 `triton.__version__`，另一路写的是 `import triton_ascend` —— 而 triton-ascend 的**导入名就是 `triton`**，`triton_ascend` 模块不存在 → 该字段**恒为 `null`**（实测新机器 env.json 正是 `"triton_ascend": null`）→ **做性能归因时拿不到这个关键版本**；② **触发时机太晚**：只有开始做性能对比才会去查版本，而那时探测结果已经是"缺的" → 归因被迫停在"版本未知"。**这与坑 110/114 同族：判据/采集层缺一项，下游就得靠猜** | ① 改为**三路采集并同时记录**：`import triton.__version__` / dist `triton-ascend` / dist `triton`（`importlib.metadata`），落盘为 `ver_*` 键，并保留旧键 `sw["triton"]` 兼容下游；② 抽出**纯函数** `triton_version_verdict()` 做一致性判定，**同时抓两类异常**：import≠dist、以及**上游 triton 与 triton-ascend 同名包同时安装（混装）**；不一致时写 `triton_version_note` 显式告警；③ 该判据纳入 `--selftest`（6 用例，坏例必被抓） | `PROFILE_SELFTEST_OK cases=5` + **`TRITON_VERDICT_SELFTEST_OK cases=6`**（含真实机器组合 `3.2.0/3.2.2/3.5.0` → `consistent=False`）|
| **123-续** | **逐文件哈希取证结论**（`step51_triton_forensics.py`）：`triton_ascend-3.2.2` 的 RECORD 声明 **140 个文件、磁盘内容全部一致（0 不一致）**；与上游 3.5.0 RECORD 不符的 **28 个文件恰好是核心导入路径**（`triton/__init__.py`、`runtime`×8、`language`×7、`compiler`×4、`backends`×3、`_C/libtriton.so`、`tools`×2、`testing`×2）；另有 **361 个上游文件仍在磁盘上**（非核心模块） | **混装真实存在，但性质是"核心已被 triton-ascend 覆盖 + 361 个上游非核心文件残留 + 一个陈旧 dist-info"**，而**不是**"运行时用错版本"。→ ① `import triton` 解析到的就是 **3.2.2（与 A3 官方配对一致）**，故 **"triton 版本不同"这一性能嫌疑被排除**；② 残留文件属**整洁性问题**（可能干扰依赖解析），**不是性能问题** | ① 在性能归因文档把 triton 从"头号嫌疑"降级为"已排除"（第 11 条排除）；② 残留清理列为**可选整洁性任务**，与性能无关；③ **取证方法本身可复用**：按 RECORD 逐文件核对 sha256（注意 RECORD 用 base64-urlsafe-nopad，不是 hex）+ 按顶层模块分组，即可区分"真混装"与"仅元数据残留" | `step51` 输出：`triton_ascend` 140/140 一致、0 不一致；被覆盖文件按模块分组如上 |
| **123-更正** | ★ **我先前"混装仅是整洁性问题、与性能无关"的结论过于绝对** —— 实测证明它有**功能性危害** | 为做性能 A/B，把 `triton-ascend` 降到 **3.2.0** 后训练**立即失败**：`ImportError: cannot import name 'Language' from 'triton.backends.compiler'`。根因：pip 按 3.2.2 的 RECORD 删掉了它的 140 个文件、装入了 3.2.0 的文件，**但 361 个上游 3.5.0 残留文件仍在** → 残留模块去 import 只有 3.5.0 才有的符号 → 立刻崩 | ① 把"混装"的定性从"仅整洁性"更正为：**性能影响未证实，但功能影响已证实（版本变更必然失败）**；② **任何换版实验前必须先清理残留**（删除不在目标 dist RECORD 内的 `triton/` 文件）并**先备份**（`triton/` 目录 1.2 GB）；③ 把"`pip install --no-deps <新版本>` 后**必须跑一次探针**"作为换版流程的强制步骤 | 换版后未跑探针 → 训练 0 迭代、rank0 `exitcode 1`；恢复 3.2.2 后探针 **`TRITON_NPU_OK backend=ascend triton=3.2.0`** ✓ |
| **124** | ★ **`versions.lock` 的"权威声明"与 A3 实装不符** —— lock 写 `triton_ascend = 3.2.2`，而 `skill_A3_bringup\README.md`（A3 自己的 bringup 记录）明确写"索引最高 **3.2.0**，装 3.2.0"；实测通用索引可用版本正是 `3.2.0rc2/rc3/rc4/3.2.0`（**无 3.2.2**），3.2.2 只在 **ascend 索引**（`mirrors.huaweicloud.com/ascend/repos/pypi`，可用 3.2.1/3.2.2）里 | **同一事实的两份记录给出不同值**，而我把 lock 当成 A3 的实装事实用于性能对比 → **差点据此把"triton 版本差异"错误排除**。这与坑 114（配置声明 vs 运行态）、坑 118（陈旧账本）同族：**"声明值"不等于"实装值"** | ① 跨机器对比**必须取实装版本**（`pip3 show` / `importlib.metadata.version()` / 探针输出），**不得引用 lock/文档里的声明值**；② lock 里的"配对"应写成**范围+来源索引**，并标注"以实装为准"；③ 在证据包里**同时保存** `pip3 show` 原始输出（已入 `VERSIONS.txt`） | 三处证据对照：lock=3.2.2 / A3 bringup=3.2.0 / 新机器实装=3.2.2；通用索引可用版本列表 `3.2.0rc2, 3.2.0rc3, 3.2.0rc4, 3.2.0` |
| **125** | ★ **triton-ascend 与 CANN 是耦合变量，无法单独 A/B** | 本机 `CANN 9.1.0-beta.3` 下，`triton-ascend 3.2.0`（A3 侧记录的版本、也是通用索引最高版）**装完即无法启动训练**（ImportError，见 123-更正）；而 **3.2.2 可跑通 100 步 rc=0**。→ **CANN 版本决定了哪个 triton-ascend 可用** | ① 性能结论里把二者视为**一个耦合变量**："CANN 9.1.0-beta.3 + triton-ascend 3.2.2 配对"，**不得拆开单独归因**；② 这也**印证了 Skill 既有策略的正确性**：`bringup.sh` 的注释早就写着"triton-ascend 先实测再决定装不装……无需按 lock 强改"（坑 46 之后定的策略）—— 本次实验正好证明了"按 lock 强改"会直接把环境弄坏；③ 若日后要真正分离，需**先清理残留**再换版，或换 CANN 版本 | 3.2.0：0 迭代 + `ChildFailedError` + `ImportError: cannot import name 'Language'`；3.2.2：探针 `TRITON_NPU_OK` + 100 步 rc=0 |

### 坑 110-111 的元教训

1. **"判定链报红"要先怀疑判定链。** 两个坑的症状都完美伪装成"结果不合格"
   （`NOT_COMPARABLE` / "配置校验失败"），根因却都在**判定器与配置生成器自己**身上。
   凡是"所有实验都失败得一模一样"的结果，**先查判据是不是恒红**。
   → 与坑 107（诊断规则假阳性）、坑 46（判据假阴性）同族：**判据本身是一个需要被验证的组件**。
2. **同源原则必须在代码里强制，不能只在文档里声明。** 我们早已写下"跳过/验收/运行/闭环条件
   必须同源"，而坑 110 **正是它的直接违反**：运行条件是 world=2（日志实测），验收条件是 world=1
   （硬编码），两者**各自"都有依据"**，所以没有任何一层会报错。
   → 纪律：**同一事实的多个来源必须在代码里交叉核对，不一致即拒判**；
   "哪个来源更权威"这种判断不该留给人和脚本各自去猜。
3. **拒绝思考胜过猜一个默认值。** 旧代码把"我不知道 world"表达成了 `1`。
   **把未知写成默认值 = 把未知伪装成已知**，而且会让下游产出看起来正常的结果
   （GBS=4 那段训练本身就"跑通了"）。→ 无法确定时**拒判，并说明如何补全**。
4. **"打包成功"不是"打包正确"。** 坑 112 里打包脚本每一步都打印成功、文件数也对，
   但**源目录不是我在改的那棵树**。凡是"从 A 复制到 B"的环节，都必须回答
   **"A 是怎么定位的、它现在是不是唯一真相"**；路径常量写死 = 下一次改完就静默失效。
   → 纪律：**交付物打包必须"打完后回读校验关键内容"，且校验失败不得落盘。**
5. **"载入修好了"会让人以为整个缺陷绕过了。** 坑 113 里 `load_rank0_and_broadcast` 生效后
   训练能正常开始，于是"这个 CANN 缺陷已被绕过"的印象就成立了 —— 直到 **100 步跑完才在
   收尾保存处炸掉**。同类缺陷常常在**同一条路径的另一个方向**上复发。
   → 纪律：修一条路径时，**必须问"对称的那条路径呢"**（load↔save、读↔写、请求↔响应）。
6. **"红线通过"不代表"事实一致"。** 坑 114 里 `dp=1/mbs=2/gas=4` 与 `dp=2/mbs=4/gas=1`
   **GBS 都是 8**，红线检查两边都过。**任何"只用派生量做检查"的设计，都存在被补偿性参数
   骗过的空间** → 关键量（这里是 dp 本身）必须**逐项比对**，而不是只比它的乘积。
7. ★ **引用证据必须落到字段级，且不得引用被截断的打印。** 坑 115 里我同时犯了两个错：
   把**指纹层**的 gates 当作**裁决层**的 gates、把 `[:150]` 截断后的字符串当作完整值。
   两者都不涉及任何代码缺陷，却足以让我把"完全一致"报成"有差异"。
   → 纪律：**冻结证据只认 JSON 字段；打印行（尤其带截断的摘要行）只能用于人眼扫读。**
   同类风险：两层同名不同义的清单（指纹 gates ↔ 裁决 gates）、`_unverified` 之类**语义后缀**
   恰好落在截断边界上。
8. ★★ **"驱动层的判据"是整条链上最没人守的一环。** 坑 116/117 让 P6、P7 **两个阶段
   从始至终不可能通过**（一个是字段名写错 + 转义被污染，一个是产物名凭印象写），
   而两次的产物其实都完全合格。危害是双重的：
   **① 报出假红**，让人以为结果不达标；**② 掩盖真红** —— 一旦有两个阶段"反正总是 FAIL"，
   真正的新失败就会淹没在这个噪声里没人看。
   → 纪律：**阶段判据必须与产出脚本的契约绑定**（产物名从脚本推、字段名从产物读），
   并且要有一条闸门守着这层契约（本次落成 G18）。坑 118/119/120 同源：
   账本新鲜度没人判、时间戳格式没人归一、性能口径没人断言 —— **都是"判据侧"的缺口**。
9. **修 bug 的第一版自己也会错，所以修复必须能被它修的对象反驳。** 坑 119 里我修的正是
   "陈旧 vs 本轮"判断，第一版就因为时间戳格式不一致而全判错；**能发现它，是因为驱动
   把 `本轮/陈旧` 的分解值打印了出来**，而那个分解值与条目 ts 明显矛盾。
   → 纪律：**凡是判断类输出，打印"分解值"而不只打印结论**；纯结论（`陈旧 0`）
   没有任何自我反驳能力。


















































### 坑 71-73 的元教训

1. **"文档写了但代码没做"是最隐蔽的一类缺口。** `versions.lock` 里 MindSpeed 的安装方式写得清清楚楚，
   但**没有任何一行代码去执行它** —— 于是 lock 的权威性反而让人误以为"这块已经管好了"。
   → **凡是 lock/env_matrix 里写明的"必需依赖"，都必须有一条 bringup 步骤与一条自检判据与之对应。**
2. **`-f` 类模式匹配是把双刃剑**：既能匹配到自己（误判），也能**杀掉自己**（误操作）。
   凡涉及"按模式找/杀进程"，一律回到 **pidfile**。
3. **异常被外层框架的汇总信息掩盖**，会让根因排查成本剧增（torchrun 只报 `ChildFailedError`）。
   → 诊断代码必须**主动下沉**：从子日志里 grep 出第一条真实异常并置顶显示。











### 坑 51-55 的元教训

1. **"加断言"解决不了假阳性，"测试断言"才能。** 一个判据如果从没被喂过坏输入，
   它就是装饰。**判据的可证伪性必须是显式的、自动化的、每次都跑的。**
2. **守卫自身也要有守卫。** 第一版静态守卫命中了 5 处**我自己写的注释**，
   产生大量噪声；若不加"去注释 + 文件级放行 + 窗口判定"，守卫会因误报而被忽略 —— 那比没有更糟。
   同时每个正则必须能匹配合成坏样本，否则它"看不见"而不是"没问题"。
3. **最讽刺的一条**：写"防止把 SKIP 计为失败"的工具时，我第一版就犯了这个错（坑 55）。
   说明这类错误不是疏忽，而是**默认思维习惯**——只有机制能防住，靠自觉不行。



## 接手新增：容器 CPU 配额

| 编号 | 现象 | 根因 | 处置 | 证据/判据 |
|---|---|---|---|---|
| **126** | affinity/nproc 可见 640 核，曾据此排除 CPU 配额限制 | cgroup v1 实测 quota=4000000、period=100000，仅 40 核时间当量；cpu.max 缺失不表示无限额 | 同时识别 v1/v2、记录配额与 affinity；训练前后计数器取增量，计数器回退拒判，不能拿历史累计节流直接归因 | 本轮 cpu.stat: nr_throttled=6、throttled_time=8417355850 ns；97_cpu_thread_sweep.py --selftest 含计数器回退坏例；矩阵未运行前不宣称 CPU 是主因 |

| **127** | 新增 Linux CPU 探针在 Windows 的 G3 --help 冒烟中调用 sched_getaffinity 后失败 | 最初未解析 --help，帮助请求意外执行了平台相关探测 | 先 argparse.parse_args 再执行探测，使 --help 零硬件可运行；prepush 以活副本为 cwd，与 review_check 的运行上下文一致 | 96_cpu_quota_probe.py --help 应 rc=0；真实探测只在 Linux 执行，缺能力不得伪造数值 |

| **128** | T1 首次启动在 geometry 处 KeyError，未进入训练 | 历史 /tmp/p5_fix100.yaml 只声明 fully_shard_parallel_size=auto，没有显式 dp；诊断脚本错误选择历史临时文件为输入 | 改用当前 out/plan/train_config.yaml；该文件实读含 data_parallel_size=2、save_format=hf、load_rank0_and_broadcast=true；不放宽几何守卫，不猜缺失 dp | 真机读取两份 YAML 对照；首次前置失败未训练，当前配置才可继续；固定同 GBS 的错误几何仍由坏例拒绝 |
| **129** | 容器重启后，旧实验有完整迭代及保存日志却没有结果/退出证据 | 后台脱离只能防 SSH 断开，不能防容器生命周期结束；累计 CPU 计数也随之重置 | 旧结果按 incomplete 保留；新目录显式 start-index 重跑，不跨容器相减，不把已打印步数当训练成功 | 98 审计缺 result 列 incomplete；坏退出码必须拒绝；97 计数器回退坏例必须拒绝 |
| **130** | Skill 入口仍写“51× 优化 + 逐位精度对齐”，且说产物有效可忽略退出码 | 文档口径未跟随黄色裁决与收尾失败防线更新 | 入口改为 NEEDS_EVIDENCE、两道 open gates、性能未达标；同步完整 GBS 公式与退出证据约束 | 入口不得升级裁决；冻结 judge 字节不动，SK04 重放仍验证相同 verdict_id |
| **131** | T1 节流最多的组反而迭代中位数最低，若直接用总节流时间归因稳态，会得出相反解释 | cpu.stat 前后差包含加载、训练、保存；日志观测时刻也不等于阶段真实边界 | T3 记录细粒度计数，跨阶段区间单列 boundary_uncertain；统计单位为独立运行而非单次迭代 | 99_repeat_phase 自检：跨界增量只进不确定桶，计数/时间/步数回退必拒；全阶段增量加总必须等于全程差，坏退出/GBS/哈希/缺步拒绝 |
| **132** | SSH session not active 导致本地监视器退出，末尾归档/拉取没有执行 | 训练虽已 setsid 脱离，归档却仍依赖客户端存活；断开不能证明远端任务失败 | 启动器退出时自动归档，保留原训练状态；重复归档拒绝覆盖。当前已运行旧版未获此修复，重新连接先查现场，不直接重跑 | t3_launch.sh --selftest：有效目录归档且排除 checkpoint；重复目标和不存在目录均拒绝；远端完整性仍由六组回读审计判定 |
| **133** | 工作目录保留且来源 YAML 哈希相同，新地址下默认线程耗时却明显变化 | 文件持续不等于运行环境持续：boot_id、容器启动身份、NPU 映射已改变；未主动改包也不能证明两边实装环境完全相同 | 分启动身份独立执行完整重复实验；不跨环境补样本、拼拟合或把性能差异归功于 OMP/CANN/Triton 单因子 | 99_repeat_phase 的汇总负例拒绝跨 identity；r2 六组独立回读一致，但跨环境原因仍待实装对照，不升级精度/性能裁决 |
| **134** | ★ **连续起两次训练时，第二次在 HCCL 初始化阶段失败**：`RuntimeError: HCCL function error: hcclCommInitRootInfoConfig(...), error code is 7` + `ERR02200 DIST call hccl api failed` + `Communication_Error_Bind_IP_Port(EI0020): ... The IP address 192.27.2.193 and port 16666 have already been bound.`；**而 `50_train.py` 的诊断报"未匹配已知故障模式"** | ① **两条端口路径只修了一条**：坑 105 修的是 **torchrun 的 `master_port`**（已加自动挑空闲端口），但 **HCCL 有独立的 NPU socket 端口（默认 16666）**，上一次运行的该端口尚未释放 → 第二次挂在 `hcclCommInitRootInfoConfig`。**与坑 113 同族：修一条路径时必须问对称的那条。** ② 同时暴露**诊断层缺口**：这条真实故障**不在已知模式表里** → 直接后果是随后的利用率采样 **82 秒全是 0**，还差点被读成"NPU 空等"（**假阴性又伪装成结论**） | ① `50_train.py` 新增专属诊断规则（匹配 `Bind_IP_Port` / `HCCL_NPU_SOCKET_PORT_RANGE` / `hcclCommInitRootInfoConfig`），处置写明：**等上次运行完全退出后重试** / **或 `export HCCL_NPU_SOCKET_PORT_RANGE=60500-60600` 换段** / **同一实验禁止连起两次**；② 原"端口占用"规则标题改为 **"端口占用（torchrun master_port）"**，处置里点明"HCCL 的 socket 端口是另一条路径"；③ 新规则**纳入 `--diag-selftest` 样本**（可证伪） | `DIAG_SELFTEST_OK samples=14 rules=13 uncovered=0`，且健康日志仍 **0 命中**；真机复现=左列报错原文；处置生效——换端口段后 100 步正常运行（步时长 423.9 / 434.2 / 513.7 ms） |

| **135** | ★ **自带采样器在迭代期读数恒为 0，而同一时刻手工执行同一条查询读数是 100/23**：`step54` 的 0.5 s `npu-smi info -t usages -i 3 -c 0` 采样，三组共 100/115/111 个样本中只有 17/25/31 个非零，**迭代窗口内全部为 0**；而同时刻手工执行 `npu-smi info -t usages -i 3`（不带 `-c`）读到 `Aicore 100 / 47`、`-c 0` 读到 `23` | ① **采样器本身没有正对照**：脚本从不检查"在已知繁忙时刻能否读到非 0"，于是"读数为 0"既可能是**真的空等**，也可能是**取错设备/取错 chip/解析失配**，两者在数据里长得一模一样。② 这与坑 35（假阳性）、坑 134（82 秒全 0 的假阴性）**同族**：都是**测量工具的可信度没有被单独验证**，却拿它的输出去支撑结论。③ 根因**尚未定论**（`-c <chip>` 的选择、并发调用、还是解析），所以**不下结论、只立规矩** | ① 依赖 NPU 利用率/带宽的结论**必须**先做**正对照**：在明确有负载时（例如训练进行中）手工取一次非 0 读数，并把该读数与命令原文一并存档；② 采样器必须落盘**原始输出首行**（而不是只落解析后的数字），使"真 0"与"取错"事后可判；③ 未通过正对照的利用率数据**一律标注作废**，不得进入任何机制主张（本轮 step54 的 AICore/HBM 数据即按此作废，机制归因改走 profiler 五桶） | 真机对照实测：同刻 `-i 3 -c 0` 读 `Aicore 23 / NPU Util 68`，`-i 3` 读 chip0 `Aicore 100 / HBM BW 33`、chip1 `Aicore 47 / HBM BW 26` → 命令本身可用，**问题在采样器**；本轮机制结论因此改由 `step_trace_time.csv` 五桶给出（Free 106.0 + Comm 未重叠 40.3 = 146.3 ms，与四点拟合的 146.9 ms 吻合 0.4%） |

| **136** | ★ **"相对脚本定位"的启动器被单独上传到 `/root/ops` 后，Skill 根目录变成了 `/root`**：`p57_launch.sh` 里 `SKILL="$(cd "$HERE/.." && pwd)"` 在包内正确（`scripts/..` = Skill 根），但脚本被 `ssh_tool push` 到 `/root/ops` 后 `HERE=/root/ops` → `SKILL=/root` → 训练脚本报 `FileNotFoundError: '/root/out/plan/train_config.yaml'`，**任务秒退、且退出码被 `set -eo pipefail` 正常传递，看起来像"配置不存在"** | ① **同一个"相对定位"写法在两种部署形态下含义不同**：包内（`scripts/` 是 Skill 的子目录）vs 单独上传（脚本与 Skill 无关）。这正是坑 112（`pack_skill.py` 路径常量指向陈旧副本）的**镜像**：那次是"硬编码路径"错，这次是"以为相对就安全"。② 直接后果是**白跑一次 token 轮次**（第一次 p57 启动即失败） | ① 远程启动器一律写成 `SKILL="${SKILL_DIR:-/root/qwen35-ascend-migrator}"`：**默认值写死为安装路径**（部署形态的真相），需要覆盖时用环境变量；② `HERE` 只用于找**同目录**的诊断脚本，不再用于推导 Skill 根；③ 启动后必须**校验第一次 `P*_START` 行里的路径**（本次因该行存在才发现路径已错，而不是等到训练报错） | 修复后真机复跑：`P57_START out=/root/ops/p57b_20260917 ... boot=7bcced6f…` 路径正确，三组 A/B/A 正常起跑；四个启动器（p2fit/p2d/p2prof/p57）统一改为该写法 |

| **137** | ★ **`npu_visible` 只看 `npu-smi` 一条路径 → 在 torch_npu 明明可用的机器上被判成 BLOCKED，直接跳过 P5/P6**：实测容器里 `npu-smi info -l` 报 `npu get board type failed. ret is -9005`、`ls /dev/davinci0` 也不存在（设备其实叫 **`davinci14`**），但 `torch_npu.device_count()=1`、`get_device_name(0)=Ascend910_9382`、**分配+计算+数值核对全部通过**（`ALLOC_AND_COMPUTE_OK`）。旧逻辑据此把可用的机器标成 `npu_visible=BLOCKED`，并写明"P5/P6 跳过" | ① **坑 103 的同族**：把环境适配押在**一个命令**上。HBM 探测当时改成了三路回退，**`npu_visible` 却漏了** —— 说明"修一条路径时要问对称的那条"这件事**要做第二遍才做全**。② 更隐蔽的是**假阴性**：判 BLOCKED 后流水线会"诚实地跳过训练"，于是**没人会去质疑它**（失败被包装成了合规行为）。③ 同时暴露一个环境坑：`npu-smi` 不可用**不等于**无卡，容器里工具权限受限是常态 | ① `05_preflight.py` 的 `npu_visible` 改为**三路回退**并**记录命中路径**：① `npu-smi`（原路径）② `torch_npu` 设备数自证（`_envcompat.torch_npu_info()`）③ `/dev/davinci*` 节点存在；② 命中的是路径 ②/③ 时判 **DEGRADED 而非 BLOCKED**，并把"npu-smi 相关能力（HBM/利用率采样）不可用"写入降级账本 `out/degradations.json`；③ 证据串里必须写出**实际设备名与 `/dev` 节点**，避免下次再把"工具缺失"读成"硬件缺失" | 修复后真机实测：`npu_visible` 报 `DEGRADED — npu-smi 不可用，但 torch_npu 自证可用：device_count=1 name=Ascend910_9382（/dev 下 davinci14,davinci_manager,...）`；同机 `selfcheck.py` 仍 `SELFCHECK_OK`（且如实标注"本机非 versions.lock 描述的验收环境"） |
| **138** | ★ **为了跑 `npu-smi` 而把 `driver/lib64*` 全局 export 进 `LD_LIBRARY_PATH`，会把 `import torch` 弄坏**：`export LD_LIBRARY_PATH=/usr/local/Ascend/driver/lib64:...` 之后 `python3` 直接报 `Failed to load the backend extension: torch_npu`（torch 的后端自动加载失败）；而**不导出**时同一台机器 `import torch` + `.npu()` 正常 | ① 训练栈（torch/torch_npu）与工具栈（npu-smi）对动态库的要求**不同**：把工具栈的库路径铺到全局，会污染训练栈的库解析。② 这与坑 134/135 同族：**为了让某个辅助命令跑起来而改全局环境**，代价往往落在主流程上，且报错位置离原因很远 | ① 规矩：**只对需要它的子进程按调用注入**（例如 `bash -c "export LD_LIBRARY_PATH=...; npu-smi info"`），**禁止全局 export**；② Skill 的 `_envcompat._ascend_env()` 只在子进程 env 上叠加，属于正确做法，脚本改动时不得退化为全局导出；③ 启动器里只用官方 `source /usr/local/Ascend/ascend-toolkit/set_env.sh` | 真机对照：全局导出 → `IMPORT_FAIL torch: Failed to load the backend extension: torch_npu`；不导出 → `ALLOC_AND_COMPUTE_OK` + `NPU_DIRECT_OK`（同一台机器、同一解释器） |
| **139** | 把新增辅助函数**追加到脚本末尾**（即 `if __name__ == "__main__":` 之后）→ 本地 `import` 正常、`py_compile` 也过、推送前闸门的编译与加载冒烟都过，但真机运行时 `main()` 调用该名字 → `NameError: name 'apply_single_die' is not defined`，后台任务**秒退**（白烧一个 token 轮次） | ① **编译期查不出「定义顺序」**：语法正确、模块可导入，只有真正执行到 `main()` 才暴露；② 「在文件末尾追加函数」是极自然的编辑动作，而入口块恰好也在末尾 → 顺序被悄悄破坏；③ 与坑 35/137 同族：**判据跑不到的地方就是盲区** | ① 推送前闸门新增 **G21**：用 AST 扫描，**模块级 `def`/`class` 若出现在 `if __name__` 块之后即 FAIL**（本地 0 算力拦住，现为 21 道闸门）；② 真机启动器保留 `bash -n` 与首行 `DISPATCH/P*_START` 打印，使这类秒退在**日志第一行**就能看见，不必等超时 | 修复后：三个脚本的 `apply_single_die` 均满足「定义早于 main 早于 __main__」且 `import` 通过；`PREPUSH_OK 21/21`（含 G21）；真机日志首行即 `DISPATCH ... script=61_mem_comm_probe.py` |
| **140** | ★★ **手工造配置绕过了"按实测 CANN 自动写入"的档位 → 真机 dp2 一启动就崩，而现场看起来像"HCCL/设备坏了"**：`TrainEngine.__init__ → load()` 里 `dcp.load` → `distW.reduce_scatter("plan")` → `dist.scatter_object_list` → `soName=libscatter_aicpu_kernel.so, funcName=HcclLaunchAicpuKernel, errcode:11006` → 两 rank 各一份 `RuntimeError: ACL stream synchronize failed, error code:507018`，**一个 `iteration` 行都没有**。而**同一台机器上 2 rank `all_reduce` 实测完全正常**（`hccl_ops.py`）→ 一度被判成"HCCL/设备问题" | ① 本机 CANN 是 **9.1.0-beta.3**，正是**坑 108（load 侧）/ 坑 113（save 侧）**的复现条件；② 我为了做 dp2 mock 基线，**从仓库模板 `config/templates/qwen3_5_0_8B_geo_m4g1_dp2.yaml` 手工派生配置**，模板里是 `load_rank0_and_broadcast: False` 且**没有** `save_format`；③ 关键不是"少填两个字段"，而是 —— **P0/P2 已按实测 CANN 自动写入这两个开关（生产侧有守卫），而我绕过了生产侧手工造配置，消费侧当时没有任何检查**：守卫被**静默跳过**，于是故障现场伪装成机器故障。这与坑 111（同一语义两个来源、无人交叉核对）**同族**，但方向相反：那次是"写的人不查"，这次是"**只有生产侧查，消费侧不查**" | ① 新机器/新实验的配置**一律从 P2 产物派生**，不手抄模板：`_ops/mk_mock_from_plan.py` 只改数据源与输出路径，其余照抄，**写盘前后各校验一次**，字段同名多处出现即拒（`MOCK_PLAN_SELFTEST cases=7 failed=0`）；② **消费侧新增同源守卫**：`61_mem_comm_probe.py` 的 `beta_config_check()` 在启动前读 P0 产物 `out/probe/env.json → recommended_profile` 核对配置，**判据取自生产侧实测结论、不看配置自己声称什么**；不一致 → `rc=3` 拒绝启动（`BETA_GUARD_SELFTEST cases=8 failed=0`，含「正式版 GA 不得误拦」的假阳性对照与「env.json 不可读则保守拒绝」）；③ 诊断规则补 load 侧专属分支（`libscatter_aicpu_kernel.so` / `HcclLaunchAicpuKernel` / `central_plan: LoadPlan`），修掉此处原先的「未匹配已知故障模式」（`DIAG_SELFTEST_OK samples=15 rules=14 uncovered=0`，健康日志仍 0 命中） | 真机 `199.103.55.150`（dual-die）`/root/ops/b_base2_20260921/train.log` 707 行：两 rank traceback 均落在 `checkpoint/utils.py:217 reduce_scatter` → `distributed_c10d.py:3627 scatter_object_list`；同机 `hccl_ops.py` 逐算子矩阵 `all_reduce PASS`；派生配置复跑见本轮 `MOCK_PLAN_OK ... GBS=8` 与 `BETA_GUARD_OK` |
| **141** | 重建 Skill zip 后 `PREPUSH G22` 变红，**报错方向却是反的**：提示「打包/解包已落后于活副本」，而实测**活副本、zip、提交目录解包副本三者一致**，真正陈旧的只有**包内解包副本** | ① 同一个「解包副本」语义其实有**两处**（包内 `05_交付物成品\06_Skill\` 与提交目录 `C4AI复赛_交付材料\06_Skill\`），而 `sync_to_delivery_folder.py` 的刷新逻辑**只写了提交目录那一处**，包内那份**没有任何脚本负责** → 它只能靠「上次碰巧被刷新过」存活；② G22 是**四方逐文件哈希**，但报错行把 `zip=` 与 `copy=` 并排打印却**不标谁新谁旧**，必须额外推理才能定位，于是极易**修错方向**（去重打包，而不是去补解包）；③ 与坑 134 同族：**修一条路径时要问对称的那条**。 | ① `sync_to_delivery_folder.py` 改为**对两处站点循环刷新**（各自备份目录 + 幂等备份名），一处逻辑覆盖两处实例；② 新增 `_diag_copies.py`：一张表并排打印 **LIVE / ZIP / PKG_UNPACK / DELIV_UNPACK 四地哈希**，把「哪一份陈旧」变成**直接可读**，不再靠推理；③ 权威 zip 名改从**包内**目录取（提交目录那份只是它的副本）。 | 修复前 `_diag_copies.py`：`LIVE=ZIP=DELIV=555895ea`、`PKG_UNPACK=e1bea677` → `divergent=3/3`；修复后四地 `ALL_SAME` → `divergent=0/3`；`COPIES_OK`、`PREPUSH_OK 22/22` |
| **142** | 往坑表追加「140」时，我拿**上一行（139）当 `old_string` 做替换**，于是 139 整行被**覆盖掉** —— 编号看起来是 `1..140`，实际只有 139 条，**139 凭空消失** | ① 「追加」用「替换上一行」实现，而锚点行**没有被写回**：编辑动作看着像「在末尾加一行」，实际是「覆盖最后一行」；② 更隐蔽的是**表象自洽**：文件仍有完整的表头、行数只差 1，肉眼与 diff 都不容易发现；③ 这说明「追加类编辑」必须配**条数/连续性断言**，否则错误会一路带进交付包 | ① 补回 139（**逐字取自改动前的备份副本**，不手抄），并在写入脚本里加**写入前后连续性断言**（`1..N` 无缺号且条数 == N，不通过不落盘）；② 保留 **C10（坑表编号连续无缺号）** 作为独立闸门 —— 本次正是它在几分钟内抓到：`编号 1..140 共 139 条，缺号=[139]` | `review_check.py` C10 报 `缺号=[139]`；补回后 C10 恢复 PASS，`PREPUSH_OK 22/22`、`REVIEW_OK 18/18` |
| **143** | ★★ **torch 把"worker 被容器 OOM 杀"报成"共享内存不足"，按提示去查 /dev/shm 会查到一个完全健康的 16 GB/0%**：dp2 首次进入取数据阶段时，两个 rank 各有一个 DataLoader worker 死掉 —— `RuntimeError: DataLoader worker (pid 50440) is killed by signal: Killed`，紧邻一行 `ERROR: Unexpected bus error encountered in worker. This might be caused by insufficient shared memory (shm).` | ① 报错文案点名了**错误的资源**：那句 "shm" 是 torch 在 worker **意外死亡**时给出的**通用猜测**，不是诊断；② 真正的判据在**容器 cgroup 自己的记账**：`memory.max_usage_in_bytes` == `memory.limit_in_bytes`（240 GB，**撞顶**），`memory.oom_control` 的 `oom_kill` 计数 **+2**（正好两个 worker）⇒ worker 是被**容器内存上限的 OOM killer** 杀的，与 /dev/shm 无关；③ 同时排除了另外两个"嫌疑资源"：`pids.max=10000`（不是进程数限制）、NPU 显存无关；④ 与坑 140 同族：**现场看起来像资源故障，实际判据在另一个文件里** | ① 新增**容器口径采样**：`cgroup_mem()` 采 usage / max_usage / limit / **rss（与 cache 分开 —— `max_usage_in_bytes` 含页缓存，混在一起会把"读了大文件"误读成"进程吃光内存"）** / `oom_kill`，并用**独立线程 1 Hz** 采（主采样线程被 `npu-smi` 拖到 7~9 s 一次，**看不住 OOM 瞬间**）；② 同时记录 `peak_python_procs`，用于识别"每 worker 再起预处理池"的子进程爆炸；③ 判据里写明：**`peak_pct_of_limit` 读不到时不得判断内存充足**（读不到 ≠ 充足）；④ 归因顺序固定为「**先读该资源的记账文件，再读报错文案**」 | 机器 B（`199.103.55.150`，host `5d7b6b055c02`）：`memory.oom_control` → `oom_kill 2 / oom_kill_local 2`；`memory.limit_in_bytes`=257698037760、`memory.max_usage_in_bytes`=257700139008（撞顶）；`df -h /dev/shm` → 16G / 0% 使用；`pids.max`=10000。同一台机器上坑 140 的修法已生效：日志出现 `Loaded checkpoint from /root/Qwen3.5-0.8B-dcp/release` 与 `Finished loading and broadcasting checkpoint tensors from rank 0` ⇒ 故障点已从 **load 阶段**移到 **DataLoader 阶段** |
| **144** | ★★ **把「官方数据侧参数」照抄到另一份数据上，会以完全不同、且与性能无关的方式炸**：dp2 首次进入取数据时报 `ValueError: Image features and image tokens do not match, tokens: 2992, features: 262144`，worker 随之死亡、一步未跑；而**同一台机器**上 bringup phase4 的 mock 冒烟（4 图那份数据 + mock 模板）**100 步跑得通** | ① 官方 `cutoff_len: 1024` 是给 **COCO 单图**样本调的；我喂的是**多图** mock 样本 → 序列被截断到 1024 后，`<|image_pad|>` 占位符**被截掉一部分**，而视觉侧仍按**全部图**产出特征 ⇒ **占位符数 ≠ 特征数**；② 已验证的 mock 模板用的是 `cutoff_len: **2048**`，能装下整个多图样本；③ 关键认知：**参数是为数据调的，不是"官方值"就通用** —— 换数据必须重调数据侧参数；④ 这也解释了为什么"我照抄官方配置"看起来最稳妥、实际最危险：它把**一个已调好的耦合关系**拆开了 | ① mock 配置的数据侧参数**取自已验证模板**（`cutoff_len: 2048`、`num_workers: 2`），不再照抄官方；② `mk_mock_from_plan.py` 把 `cutoff_len` / `num_workers` 一并纳入「**派生 + 写盘后回读校验**」清单，并打印 **旧值 → 新值**，使每一处偏离**显式可见**而不是悄悄发生（此前只校验了数据集路径与几何）；③ 诊断规则补该专属分支（此前落在"未匹配已知故障模式"）：`Image features and image tokens do not match` | 真机 `199.103.55.150`：`/root/ops/dp2_base_20260921_113026/train.log:614`；跑得通的对照 = `mock_data_pic_num_4_textlen_512.json` + `cutoff_len: 2048`（bringup `/root/mock_run.log`）；`inspect_mock.py` 实测两份 json **结构完全相同**（各 512 样本、都无预烘焙 `image_pad`、同一张 1024×1024 图）⇒ 差异只能在数据侧参数，不在数据本身 |
| **145** | ★★ **与坑 143 是「同一族提示、相反结论」的一对，只有读记账文件才分得清**：同一轮 dp2 里还出现 `RuntimeError: unable to write to file </torch_63792_2473652340_0>: No space left on device (28)`（多个 worker 各一条）→ `DataLoader worker (pid ...) is killed by signal: Bus error. It is possible that dataloader's workers are out of shared memory.` | ① **这一次报错文案是对的**：官方 `num_workers: 8` × 2 个 rank = **16 个 worker** 同时把批次张量写进 `/dev/shm`，把 16 GB 的 tmpfs **写满**（ENOSPC）→ worker 以 Bus error 死掉；② 已验证 mock 模板用的是 `num_workers: 2` —— 与坑 144 同源：**同一次"照抄官方数据侧参数"造成的两处伤**（一处截断、一处耗尽 shm）；③ ⚠ **与坑 143 必须对照着看**：坑 143 里 `Killed` + 「insufficient shm」的猜测是**错的**（真因 cgroup OOM），本条里 `Bus error` + `/torch_*` ENOSPC 是**对的**。**同一句提示文案，两次结论相反** ⇒ 文案本身不能作为判据 | ① 数据侧 `num_workers` 取**已验证模板**的值（本项目 mock 模板是 **2**）；② 确需更多 worker 时先确认容量（`df -h /dev/shm`；容器内可 `mount -o remount,size=32G /dev/shm`，需权限）；③ 诊断规则把两种签名拆成**两条独立规则**：`is killed by signal: Killed`（→ 先读 cgroup 记账）与 `Bus error` + `/torch_*` ENOSPC（→ 先读 `/dev/shm`）；**刻意不用 `insufficient shared memory` 做任何一条的特征**，因为那句同时出现在两边，拿它当特征就是坑 107 复发；④ 新增 `cgroup_mem()` 采样（usage/max/limit/**rss 与 cache 分开**/`oom_kill`）使"哪一份记账该看"随时可查 | 真机 `/root/ops/dp2_base_20260921_113026/train.log` 526–681 行（6 条 `unable to write to file </torch_*>` + 2 条 Bus error）；`df -h /dev/shm` → 16G；同机坑 143 的 `memory.oom_control` → `oom_kill 2`。`DIAG_SELFTEST_OK samples=18 rules=17 uncovered=0`（健康日志仍 **0 命中**） |
| **146** | ★★ **`official_comparable` 只看几何 → 一次错判连带毁掉两个下游结论**：真机跑通"**官方几何 + mock 数据**"的 dp2 后，`result.json` 把这个运行标成 `official_comparable: true`，而它的窗口中位数是 **710.55 ms**、官方基线是 **431.3 ms**（差 **1.65 倍**）—— 根本不可比。更隐蔽的是 `59_config_ab.py` **用这个标志去选 step1 锚点规则**（官方几何 ⇒ 要求 step1 == 1.924621），于是 mock 数据下的有效 A/B **必然**被判成 `REJECT_NUMERIC`：**判据错一次，两个下游结论一起错**（一个是 over-claim，一个是假阴性） | ① "可比性"被简化成了"几何对不对"，而它实际是「几何 × **数据** × 运行条件」的**合取**；② 判定逻辑**散在 5 个脚本里各写一遍**（`54`/`56`/`58`/`59`/`61`），同一语义多个来源 —— 坑 111 同族；③ 与坑 140 同族：**把收敛/判定条件放宽到"某个必要条件"**，于是"看起来通过"；④ 这正是本项目反复出现的形态：**判据比问题弱**，而且弱得刚好能通过自检 | ① 判据搬到 `_envcompat.official_comparable()` 作为**唯一定义处**：同时要求「几何官方」**且**「数据侧与 P2 计划**逐项**一致（10 项）」，且**取自计划文件本身**（同源），不再各脚本各写一遍；② **fail closed**：计划读不到、或 `_envcompat` 加载失败 → 一律 `official_comparable=False` 并写明原因（**宁可少宣称，绝不 over-claim**）；③ JSON 里只留**一个布尔**，另加 `comparability` 块**只放证据**（`reason`/`deviations`/`plan`/`is_official_geometry`），避免"同一语义两个来源"；④ 自检补 4 条负向对照，最关键一条是「**官方几何 + mock 数据 ⇒ 必须 False**」（旧实现在这一例返回 True）；⑤ 顺带修 `_envcompat.selftest` 的打印：原 `items=%d` 填的是 `fails`，**全过时显示 `items=0`**（读起来像"一项都没检查"，与实况相反）→ 改为 `items=<总数> failed=<失败数>` | `_envcompat.py --selftest` → `SELFCHECK_OK items=16 failed=0`（含 4 条可比性用例：配置与计划相同→True / 官方几何+mock→**False** / 非官方几何→False / 计划缺失→保守 False）；`P59_SELFTEST_OK cases=16 failed=0`；真机证据 = `/root/ops/dp2_base_20260921_113921/result.json` 的 `window_11_end_median_ms: 710.55` 对照官方 `431.3`（同文件旧版却写着 `official_comparable: true`） |
| **147** | ★★ **判据写对了、也自检了，但真机一跑就崩 —— 因为自检**从不走**被判的那条代码路径**：`59_config_ab.py` 在真机启动后立刻 `UnboundLocalError: cannot access local variable 'steps_complete' where it is not associated with a value`（发生在 `run_one()` 里），而同一版本的 `--selftest` 输出是 **`P59_SELFTEST_OK cases=16 failed=0`**。于是"实验工具"看起来全绿，实际**一组数据都产不出来** | ① 有效性判据（`valid`）原先**内联**在 `run_one()` 里，而 `steps_complete` 的**赋值写在了使用之后**（历次"复核要求"补丁把它挤到了下面）—— 这是**语句级**的定义顺序错误：`py_compile`、AST 扫描、导入冒烟**全都查不出**（坑 139 的同族，但坑 139 是模块级 `def`，有 G21 兜底；**语句级没有闸门**）；② 更根本的：`--selftest` 只调用纯函数（`parse_value`/`pointwise`/`paired_stats`/`verdict_of`），**从不调用 `run_one`** ⇒ "这套判据到底能不能跑"从来没被验证过。**判据存在 ≠ 判据被执行过** | ① 把判据抽成**纯函数** `compute_valid(returncode, train_rc, ids, steps)`，`run_one()` 只调用它（把"被判的东西"变成"可被喂坏样本的东西"）；② 自检补 **6 条坏样本**：齐全→有效、缺最后一步、步号重复（缺 8）、进程 rc≠0、`train_rc`≠0、多出步号；③ 在**真机**上复验（不是只在本地）：`P59_SELFTEST_OK cases=25 failed=0`；④ 同族顺带修：结尾 `cases=16` 是**写死的**，历次新增用例后数字从不变 —— 改为真实计数后是 **25**，即此前有 **9 条用例从未被计入那份"声明"** | 真机 `199.103.55.150`：`/root/pregather_ab.sh.run.log` 里 `UnboundLocalError ... line 131, in run_one`；修复后同机 `python3 /root/qwen35-ascend-migrator/scripts/59_config_ab.py --selftest` → `P59_SELFTEST_OK cases=25 failed=0`（含 6 条新坏样本）；随后 A/B 正常产出 `END 00_A variant=A valid=True single_var=True win_med=713.15 wall=109s` |
| **148** | ★★ **"计划一致性"被当成"官方可比性"用 —— 复核用两个合成坏例当场打穿**：调用我的 `_envcompat.official_comparable()`，得到「① 计划和运行**都缺**数据字段 → **True**」「② 计划和运行使用**同一份 mock** → **True**」。两个都必须是 False | ① 我把可比性实现成「运行 vs P2 计划的十个字段**是否相同**」，而**计划一致性 ≠ 官方可比性**：**双方都缺**同一字段被算成"相同"、计划与运行都用同一份 mock 也"相同"；② 判据回答的是"**与计划是否一致**"，却被调用方当作"**是否可与官方比**"使用 —— 判据比问题弱（与坑 140/146 同族）；③ 只比**文件路径**也天然无法区分"两份不同的 mock 恰好同路径" | ① 拆成**三个独立结论、不得互相替代**，并作为唯一实现：`plan_consistent()`（与计划一致**且必要字段齐全**，**缺字段即 False**）、`ab_comparable()`（A/B 只有**登记变量**不同，用摊平后的逐路径 diff）、`official_comparable()`（几何官方 ∧ plan_consistent ∧ `dataset_kind == "official"` ∧ 数据身份可核，**缺证据一律 False**）；② 显式记录 `dataset_kind`（official/mock/unknown）与**数据身份**（路径+字节数+**sha256 前 12 位**），不再只比路径；③ 明确写进注释与产物：**「本机没有可对标官方的数据」不等于「没有有效的同机 A/B 证据」**，前者 False 不妨碍后者成立 | `_envcompat.py --selftest` → `SELFCHECK_OK items=21 failed=0`，其中含复核给出的**两个坏例**：「坏例①计划与运行都缺数据字段 → plan_consistent=False」「坏例②同一 mock：plan_consistent 可以为 True **但** official_comparable 必须 False（`dataset_kind=mock`）」；`GATES` 新增 `three_comparability_conclusions` 登记 |
| **149** | ★★ **判据的措辞比它实际证明的东西强**：复核给出合成案例 —— 改善均值 **+6%**、95% 区间 **[1%, 11%]** → 我的判据返回 **`WIN`**。而该结果只能支持「**存在正收益、点估计 6%**」，**不支持「收益至少 5%」** | `verdict_of()` 里写的是 `if -mean >= 5.0: return "WIN"` —— 拿**点估计**与门槛比。能支撑"至少 5%"的量是**改善区间的下界**，不是均值。这与坑 146 同族：**结论的强度由判据里那个比较式决定，而不是由结论的名字决定** | 改为对**改善区间下界**判定（改善量区间 = `[-ci[1], -ci[0]]`）：下界 **≥5%** → `WIN`，且措辞明确写"在**声明的实验条件与统计假设下**支持「至少 5%」"；下界 **>0 但 <5%** → `SMALL_GAIN`（有正收益，**不能保证达到 5%**）；区间跨 0 → `UNCERTAIN`；区间上界 <0 → `NO_GAIN`。三档与复核给出的对照表逐条对齐 | 自检新增两例并通过：`[PASS] 均值+6% 但区间下界<5% → 不得判 WIN`、`[PASS] 改善区间下界 ≥5% → WIN`；`P59_SELFTEST_OK cases=29 failed=0` |
| **150** | **数值合格性只覆盖了它恰好看到的那一对**：复核指出「当前主流程仍只对**第一组** A/B 做逐点检查」 | `pw = pointwise(A[0], B[0])` —— 后面每一对 (A[i], B[i]) 即使数值跑飞（例如 loss 差 66%），也照样进入性能判定并可能判出 `WIN`。与「缺一步/重复步号」同族：**判据只覆盖输入的一部分，而那部分恰好是通过的那部分** | 新增 `pointwise_all(A, B)`：对**每一对**做逐点比较，并按「最坏情形」合并（`max_pct` 取最大、`length_ok` 取逻辑与、`undefined_zero_baseline` 累加、`compared` 取最小），于是 `verdict_of()` **无需改动**即对全部配对生效；逐对明细落盘为 `pointwise_per_pair`，使"检查过哪几对"本身可复核 | 自检新增两例并通过：`[PASS] 只看第一对 → 漏掉第二对数值不合格（旧实现的实际行为）`（该例 verdict=`WIN`，正是旧行为）与 `[PASS] 覆盖全部配对 → REJECT_NUMERIC`；`P59_SELFTEST_OK cases=29 failed=0` |
| **151** | ★ **同一次运行在不同工具里得到不同的 `valid`**：复核指出「现在 59 检查完整步号，61 仍只检查解析行数」 | 有效性判据在两个脚本里**各写了一遍**：`59` 用 `sorted(ids) == list(range(1, steps+1))`（缺步/重复都拦），`61` 用 `len(rows) == args.steps`（只数行数）。**同一语义两个来源**（坑 111 同族），必然漂移 —— 症状是"某次运行在 A/B 判据里无效、在内存探针里有效"，而两个产物都会进交付 | 判据**唯一实现**上移到 `_envcompat.run_valid()`（59/61/62 全部复用）；`59` 的 `compute_valid()` 改为**转发**，且**fail closed**（共享校验器加载不到即返回无效，**不**在本地复制一份实现 —— 复制就是重新制造两个来源）；`61` 在产物里记录 `valid_judge`（判据来源）与 `steps_complete`/`step_ids_tail`，使"这一份 `valid` 是谁判的"可复核 | `61` 产物新增 `P61_VALID valid=... steps_complete=... judge=shared ids=N/M`；`GATES` 新增 `run_valid_unified` 登记；`_envcompat --selftest` 与 `59 --selftest` 均在改动后全绿（`items=21 failed=0` / `cases=29 failed=0`） |
| **152** | ★★ **我提出的实验设计在算术上就不成立**：我在求指导信息里写「同 GBS=8，只变 world_size」做 dp1/dp2 扩展性对照 —— 复核直接指出这句**不成立** | `GBS = mbs × gas × world`。world 从 2 变 1 而 GBS 保持 8，就**必须**联动改 `mbs` 或 `gas`；"只变 world"**在算术上不可能**。根子是我把"**想研究的对象**（并行布局）"和"**能自由变的配置项**（单个键）"混为一谈 —— 想只动一个变量，但那个变量在约束方程里不是自由的 | 采纳复核给出的对照设计（每微批大小一致、GBS 都是 8）：**dp1 = world1 / mbs4 / gas2** vs **dp2 = world2 / mbs4 / gas1**；结论口径也随之收窄为「**固定全局批量下改变并行布局**的端到端效果」，**不是**"纯通信开销"。执行条件一并采纳：① 两组每步用**同一组全局样本**并落盘样本 ID/图片数/有效 token 数（**不能只凭同一个 JSON 路径**认定一致）；② 同权重/优化器/随机种子/最终重算策略；③ 明确 dataloader workers 是**每 rank 数**还是**全局总数**（不得让 dp2 暗中多一倍加载资源）；④ 同机**串行配对**执行并确认 dp1 可用设备身份；⑤ 报告步时长、样本吞吐、**实际 token 吞吐**、峰值内存、数值偏差；⑥ **两边都 `official_comparable=false`**。解释规则：dp2 更快只宣称该 mock 负载的**强扩展收益**；相近或更慢**同样是有效结果**（可用于 Skill 的资源选择策略）；样本组成或数值不一致时**先查切分/累积/loss 归一化**，不直接归因通信 | 本轮先落设计（避免拿错设计烧算力）；待跑。参考：`GBS = mbs × gas × world` 是 `50_train.py` 里代码级红线（`rc=3` 拒绝非 GBS=8） |
| **153** | ★★ **同一处缺陷在"另一个函数"里复发，而我的"修复"和自检都没看见**：真机 pregather 队列**第一步就死** —— `null_test()` 里 `NameError: name 'steps_complete' is not defined`（`and steps_complete` 引用了一个**从未存在**的名字）。而我上一轮刚修过**同一个名字**在 `run_one()` 里的同名缺陷（坑 147），自检 `P59_SELFTEST_OK cases=29 failed=0` **全绿**。另在 `40_prepare_assets.py` 发现 `pip_install` **从未被导入**却被调用 | ① 我按"修那一行"的方式修坑 147（抽出 `compute_valid()` 并只改 `run_one()` 的调用点），**没有扫同一文件里其余同类引用** —— 这正是本项目反复出现的"**修一条路径不问对称的那条**"；② 自检覆盖的是**纯函数**（`compute_valid` 本身），**不是编排路径** —— 复核当时就写明「抽出 compute_valid() 解决了判据覆盖，但**尚未覆盖完整编排路径**」，这就是那个缺口；③ `pip_install` 那次更阴：调用被 `except Exception: pass` 包着，`NameError` 被**静默吞掉** ⇒ 依赖（jsonargparse / docstring-parser）根本没装，却到后面才以"缺模块"的面目失败 —— **静默 except 把"没装成"伪装成了"装过了"** | ① 改用"**名字解析级**"的静态扫描找全体：`python -m pyflakes` 一次扫出全部未定义名（本次共 2 处：`59:steps_complete`、`40:pip_install`），**不再逐个撞**；② 新增闸门 **G23**（pyflakes 未定义名，**fail closed**：pyflakes 不可用即判 FAIL）—— 27 个 `.py` 全覆盖；③ 新增闸门 **G24** 与 `--selftest-orchestration`：用**桩子进程**真跑 `run_one()` 与 `null_test()`（不碰 NPU、秒级），把"编排路径"纳入自检；④ `40_prepare_assets.py` 补齐 `pip_install` 导入（`_envcompat` 里本就有），并保留退化实现；⑤ 顺带修：`null_test` 在 `--steps < 11` 时窗口为空 → `max(meds)` 抛 `TypeError`，改为**明说"算不出来"**并给出 `--steps >= 12` | `pyflakes` 修复后**0 处未定义名**；`PREPUSH_OK 23/23`（含新 G23）；`P59_ORCH_SELFTEST cases=4 failed=0`（真跑 `run_one()`+`null_test()`）；真机现场 = `/root/ops/abq_20260921_115705` 队列日志里 `NameError ... line 330, in null_test` |
| **154** | ★ **六个脚本直接读 `/etc/hostname`，而这一行位于"记录落盘"处** —— 在缺少该文件的容器/非 Linux 环境上，会在**跑完一整个训练之后**才炸掉（白烧一次实跑）。由 G24 的桩子编排测试当场暴露（本地 Windows 上 `run_one()` 直接 `FileNotFoundError: /etc/hostname`） | ① 项目早就有"没有隐含环境假设"的红线与 `_envcompat.safe_hostname()`（先读 `/proc/sys/kernel/hostname`，再退回 `hostname` 命令，最后 `unknown`），但**这 6 处没走它** —— 与坑 37/38/40/50 同族：**公共底座建好了，使用处没跟上**；② 更值得记的是**失败时机**：写"身份字段"是记录的**最后一步**，所以错误发生在最贵的那一步之后；③ 它能被发现，靠的是 G24 把**编排路径**纳入了自检（静态检查与纯函数自检都看不见） | 统一改为可移植的 stdlib 形式 `__import__("platform").node() or "unknown"`（与 `safe_hostname()` 同义；用 stdlib 是为了**不为一行的身份字段**引入路径/导入依赖，已在代码注释里写明这个取舍）：`59_config_ab.py`、`61_mem_comm_probe.py`、`54_three_point_fit.py`、`55_validate_prediction.py`、`56_profile_run.py`、`57_fsdp_group_ablation.py` 共 6 处。附带：`null_test` 在 `--steps < 11` 时窗口为空 → 改为明说"算不出来"并提示 `--steps >= 12`（不再抛与原因无关的 `TypeError`） | 修复前 `run_one()` → `FileNotFoundError: '/etc/hostname'`；修复后 G24 `P59_ORCH_SELFTEST cases=4 failed=0`（含 `null_test()` 真跑），且 `grep 'open("/etc/hostname")'` 在活副本 `scripts/*.py` 中**归零** |
| **155** | ★★ **A/B 被判"数值不合格"，但那个不合格与被测键无关 —— 判据拒绝得对，归因却极易错**：`pregather` A/B（6 组 3 对）得到 `baseline_drift_pct=2.37`（低于 3% 漂移闸门）、`mean_delta_pct=-4.323`（关掉 `pregather` 看着快 4.3%），却被判 **`REJECT_NUMERIC：loss 逐点最大差 304.4466% ≥ 2.00%`**。逐组看 loss 才发现：**A 与 A 之间的差（`0.070691` vs `0.067264`）和 A 与 B 之间的差一样大**，且**步 1–2 六组逐位相同、步 3 起分叉** | ① 差异**不是** `pregather` 造成的；② `seed: 42` 与 `dropout: 0.0` 都已固定，所以不是种子/随机失活；③ 真因在**数据路径**：`sampler_type: BaseRandomBatchSampler` + `num_workers: 2` ⇒ **全局批次组合在 worker 进程间不确定**，A/B 两组每步吃到的不是同一批数据 ⇒ 逐点 loss 比较无意义；④ 最危险的是**归因误导**：若不逐组看 loss，就会把 `REJECT_NUMERIC` 读成"关掉 pregather 改变了数值/不安全"，而真相是"**这把尺子在这条数据路径上量不了**" | ① 让**零假设实验兼任数值判据的正对照**：`null_test()` 现比 `run_one()` **同一条正则**解析 loss，并断言"同配置重复跑的 loss 必须逐点相同"，输出 `P59NULL_NUMERIC identical=… first_diverging_step=… worst_rel_pct=…`；② 不同时打印 `P59NULL_NUMERIC_WARN`，**明确写出"任何 A/B 的数值拒绝都不能归因于被测键"**，并给出处置（改走确定性数据路径后重做零假设）；③ 编排自检补正/反两例（同配置→`identical=True`；人为让 loss 每次不同→`identical=False`），`P59_ORCH_SELFTEST cases=6 failed=0`；④ 新协议链 `null_proto_chain.sh`：**先**验证 `num_workers=1` 下数值逐点相同，**通过后**才允许跑 A/B（拿不可比的路径去跑 A/B 就是**假归因**） | 真机 `/root/ops/abq_20260921_122228/pregather/`（`summary.json` 与 6 组 `results.json`）；逐组 loss 对照：`00_A 0.070691/0.032680`、`03_A 0.068102/0.033612`、`04_A 0.067264/0.033230` vs `01_B 0.069094/0.033644`、`02_B 0.067132/0.033245`、`05_B 0.069103/0.033759`；零假设 4 组 `725.3/707.4/709.3/700.2 ms`、`spread=3.58%` |
| **156** | ★ **配置级 A/B 可能以"跑不起来"而非"更快/更慢"结束，而这一步必须在闸门里被承认**：`gdn_impl` 的 B 组（`model.gdn_implementation: eager`）在**官方批量几何**（dp2 / mbs4 / gas1 / GBS8）下**直接设备 OOM**——`torch.OutOfMemoryError: … current working operator name is aclnnMatmul`、`alloc device memory failed, runtime result = 207001`，`iteration` 行数 **0**、耗时 58s。队列按设计 `ABQ_STOP`（rc=1）停住，**没有**把它当成"B 更慢"继续算 | ① 这类结果容易被误处理成两种错误：**(a)** 当成"B 更差"写进性能结论（实际是**跑不起来**，无时长可言）；**(b)** 被诊断规则误判成环境故障而**重试**（浪费算力）；② 本项目的判据里"有效性"（`valid`）与"数值合格"之外，还必须有第三态：**该配置在此几何下不可运行** —— 它是**有效结论**（可直接用于 Skill 的资源选择策略），但不是时间对照 | ① 队列的"任一失败即停"保住了它：`ABQ_ITEM_DONE name=gdn_impl rc=1` → `ABQ_STOP`，**不**产出 `WIN/SMALL_GAIN`；② 结论按复核给的框架记录：**在官方批量几何下 `gdn_implementation=eager` 无法运行（设备 OOM），`triton` 可以** —— 用于"资源条件下的策略选择"，**不是**性能优劣；③ 若日后要在 eager 下取数，必须**换几何**（例如更小的每 rank 微批），那就变成**另一个可比性问题**，须重新登记、且 `official_comparable=false` | 真机 `/root/ops/abq_20260921_122228/gdn_impl/01_B/train.log`（两 rank 各一份 `torch.OutOfMemoryError`，算子 `aclnnMatmul`，`207001`）与 `driver.log`（`iteration 行数 : 0`、`耗时 58s`、`TRAIN_FAIL rc=0 iters=0`）；00_A（triton）正常完成；队列日志 `ABQ_ITEM_DONE name=gdn_impl rc=1 12:43:35` → `ABQ_STOP` |
| **157** | ★ **诊断规则又出现假阳性（坑 107 复发形态）**：`gdn_impl` 那组 OOM 的日志被自动诊断判成「**CANN 环境未加载**」，匹配串是 `libhccl\.so|Failed to load the backend extension: torch_npu`。而该日志里出现 `libhccl.so` 只是因为它在 **OOM 的调用栈帧**里被打印（`frame #0: … in /usr/local/lib64/…/libc10.so` 一类的栈回溯），**与环境是否加载毫无关系** | ① 与坑 107 同族：**匹配串必须是"只在该故障下才出现"的**，而 `libhccl.so` 会出现在任何一次 NPU 相关的栈回溯里 ⇒ 规则把"OOM"误导成"环境没 source"；② 更要命的是同一条诊断**同时**给出了正确的「显存不足」判断（匹配 `out of memory`），于是**一条日志给出两个互相矛盾的方向**，人得自己选 —— 这是"判据之间没有互斥性"的缺口 | ① 把 `libhccl\.so` 从该规则的匹配串里去掉（保留 `Failed to load the backend extension: torch_npu` 这一**只在环境未加载时出现**的串）；② 规则顺序上让**更具体**的 OOM 规则先命中并**抑制**泛化规则（同一日志不应给出两个方向相反的首要原因）；③ 用 `50_train.py --diag-selftest` 的"健康日志必须 0 命中"守卫继续兜底，并把本条 OOM 日志**加为负样本**（要求它**不得**命中"CANN 环境未加载"） | 真机 `/root/ops/abq_20260921_122228/gdn_impl/01_B/driver.log` 的「=== 自动诊断 ===」段：同时打印「● CANN 环境未加载 匹配: libhccl\.so|…」与「● 显存不足 匹配: out of memory|OOM|NPU out of memory」，而 `train.log` 的真实错因是 `torch.OutOfMemoryError … aclnnMatmul … 207001`（见坑 156） |
| **158** | ★ **自己的连接工具把两种完全不同的失败原因打成了同一句话，于是"用过期 token 测主机状态"这种无效实验被当成了有效判据**：`ssh_tool.py` 在 `open_channel` 失败时一律打印「**跳板隧道被拒（token 过期/未授权）**」。实测它其实对应两种**完全不同**的状态 —— code 1 `Administratively prohibited`（平台拒绝该 token）与 code 2 `Connect failed`（**跳板已认证成功**，但连不上目标主机）。我因此一度用**过期 token** 去测机器 A，得到的输出看起来像"token 问题"，**实际那次实验根本测不出主机状态** | ① 与坑 107/157 同族：**判据/文案必须只对应一种原因**，否则会把人送错方向；② 更具体的教训是**"测不出来"与"测出来是不通"必须分开** —— 用失效凭证做的连通性实验是**无效实验**，不是负结论；③ 这次的正确判据是：**换一个全新 token 再测** —— 若仍为 code 2，才能定性为**实例侧**（停机/换 IP），因为此时"凭证"这一变量已被排除 | ① `ssh_tool.py` 按异常文本分流并给出**可行动结论**：`TOKEN_REJECTED`（明确写"**这一条不能用来判断目标主机是否可达**"）vs `TARGET_UNREACHABLE`（写明"跳板认证已通过，这是**实例侧**状态，换 token 不会改善，请查控制台实例状态与当前 IP"），未分类错误保留原文；② 判据顺序固定为「**先排凭证，再判主机**」 | 机器 A `199.103.53.187`：**过期 token** → `ChannelException(1, 'Administratively prohibited')`（无效实验，测不出主机）；**全新 token** → 仍为 `ChannelException(2, 'Connect failed')` ⇒ 定性为**实例侧不可达**（同型号机器 B `199.103.55.150` 全程正常，用于排除"平台整体故障"） |
| **159** | ★★ **我的数值判据要求"同配置重跑逐点 loss 完全相同"，而官方配置本身就不提供这个东西** —— 于是它会拒绝**一切** A/B。两级零假设实验（`num_workers=2` 与 `=1`）都得到 `identical=False`、**首次分叉都在第 3 步**、时间散布反而只有 1.40%/1.25%。我先前据此推断"多 worker 批次交错 ⇒ 每步数据不同"，并打算用 `num_workers=1` 修 | ① **两个假设都被证伪**：读 `sampler.py:56-101` 可见 `shuffle:false` 时走的是 `list(range(full_bucket_size))` 然后 `idx_range_active[rank::num_replicas]` **跨步取样，全程没有随机数**；而 `num_workers=1` 也没改变结论 ⇒"随机取批"与"多 worker 交错"**都不是**原因；② 真正被我忽略的是**官方自己也不强制确定性**：官方 `examples/train/official_baseline.log:245` 与**我方 `examples/train/train.log:249`** 同为 **`use_deter_comp: False`**（`enable_full_determinism: False` 亦然）⇒ **"重跑逐点完全相同"不是官方承诺的性质**，我却把它当成了 A/B 的**前提**；③ 这就是"**尺子比问题严**"：`pregather` 那轮的 `REJECT_NUMERIC` 不是被测对象的结论，而是我这把尺子的产物（与坑 146 同族：判据的强度决定结论的强度） | ① **不擅自打开 `use_deter_comp`** —— 那会变成对官方的偏离、且可能与官方日志不可比，与"尽量与官方一致"直接冲突；② 数值判据从「**严格相等**」改为「**与 A/A 噪声信封比较**」：零假设实验产出信封（逐步 `max|Δ|`、首次分叉处绝对差、最坏绝对差），A/B **只在超出信封时**才判数值不合格；**未提供信封则 fail closed**（无法区分噪声与效应就不下结论）；③ **保留严格相等作为上界证据**（真能相等当然更好），但不再当门槛；④ 顺带把官方值缺口补上：`versions.lock` 补 `config.skip_gdn_recompute`（官方 `True`），模板对齐，`COMPARE_KEYS` 10→13 | 协议链 `/root/ops/nullproto_20260921_125004`：`NP_A_RESULT identical=False first_div=3 worst_rel=190.48% spread=1.40%`、`NP_B_RESULT identical=False first_div=3 worst_rel=227.04% spread=1.25%`、`NP_PROTOCOL_FAIL → 不做 A/B`；源码 `sampler.py`（无随机数）；官方/我方日志 L245/L249 同为 `use_deter_comp: False` |
| **160** | ★ **我的"分叉程度"指标被小分母放大，把量级伪造了**：我报出 `worst_rel_pct = 190.48% / 227.04%`，读起来像"同配置两次跑的 loss 分叉极大"。实际上这个相对差**没设分母下限**，而 loss 在训练末段会降到 ~0.003 量级 ⇒ **0.003 的绝对差就记 100%+**。于是"分叉极大"这个印象**来自度量本身，不来自数据** | ① 与"归一化指标在零附近失控"是同一族问题（`pointwise` 里已经有 `undefined_zero_baseline` 的计数，说明我**知道**这个坑，却在新写的指标上又犯了一次）；② 更危险的是它**方向上是"夸大问题"**：会让我误以为存在结构性差异，从而去改本不该改的东西（我确实据此准备打开 `use_deter_comp`）；③ 相对差与绝对差回答的是不同问题：**判别"是浮点噪声还是结构性差异"只能看绝对差** | ① 新增并落盘判别量：`numeric_first_div_abs`（**首次分叉处的绝对差 —— 真正的判别量**：~1e-6 = 浮点噪声被放大；明显更大 = 结构性差异）、`numeric_worst_abs`、`numeric_first_div_rel_pct`；② 相对差加**分母下限** `numeric_rel_floor=0.05`（只在 loss ≥ 0.05 的步上统计最坏相对差）；③ 原 `worst_rel_pct` 保留但**在代码注释里显式标注"会被小分母放大，不得单独用于判断严重性"** | 自检正/反两例：反例（人为让每次 loss 差 0.1）输出 `first_div_step=1 first_div_abs=1.000e-01 first_div_rel_pct=6.6667 worst_rel_pct_loss>=0.05=6.6667`；正例 `identical=True` 时**不打印 WARN**（此前的锚点编辑误删了 `if not numeric_identical:` 守卫，导致警告恒打印，已修复并复验）；`P59_ORCH_SELFTEST cases=6 failed=0` |
| **161** | ★★ **"同配置两次跑的 loss 为什么不同"被实测定位：差异从第 1 步就存在（被 6 位打印掩盖），随后被优化器混沌放大**。同配置两次跑（`A_w2/00_NULL` vs `01_NULL`，`num_workers=2`）逐组对照：步 1/2 的 loss 打印值**完全相同**（`0.09452853`），步 3 差 **1.14%**，步 6 差 **18.6%**，最大绝对差 **0.0457**（步 15）。关键线索在**别处**：`grad_norm` 第 1 步就是 A=`35.185` / B=`35.176` ⇒ **相对差 2.6e-4** | ① **不是** fp32 舍入（那会是 1e-7 量级），也**不是**数据不同（数据不同会在第 1 步就显著改变 loss）；② 形态是典型的「**非确定性计算 + 混沌放大**」：初值差 ~1e-4，经优化器指数放大到 1%~19%。与官方配置的 `param_dtype: bf16` / `use_deter_comp: False` 一致（官方 `official_baseline.log:245` 与**我方 `train.log:249`** 同为 False）—— 也就是说**这是官方配置本身的性质，不是我们的配置错**；③ 我此前的两个假设（多 worker 批次交错、随机取批）**都已被证伪**（`num_workers=1` 同样 `identical=False`；`sampler.py` 在 `shuffle:false` 下无随机数）；④ **最重要的推论**：**官方用 step1 loss 当锚点是必然的** —— 只有前几步可复现。而我把 2% 逐点容差套在**全部 60 步**上，等于**恒判 `REJECT_NUMERIC`**（`pregather` 那轮就是这么被拒的），拒绝理由与被测键无关 | ① 数值判据加 **`--numeric-steps`（默认 2）**：**只比对前 N 步**，并在 `pointwise()` 里把证据与理由写成注释；**步 ≥3 显式声明为混沌、不参与数值结论**；② 论证**不损失判别力**：若配置真改动了数学，**第 1 步就会变**（实测初值差已达 2.6e-4，远超 fp32 精度，任何真实的算子/数值改动都会在第 1 步暴露）；③ `--numeric-steps 0` 保留"比较全部步"的旧行为，供复现与诊断；④ 与坑 159 的结论合并：**不打开 `use_deter_comp`**（那是对官方的偏离），而是让尺子匹配"官方不强制确定性"这一现实 | 本地归档 `_evidence_B\nullproto\A_w2_00.json` / `A_w2_01.json`（真机 `/root/ops/nullproto_20260921_125004/A_w2/`）；逐组数字见上；`P59_SELFTEST_OK cases=29 failed=0`、`P59_ORCH_SELFTEST cases=6 failed=0`；源码 `sampler.py:56-101`、官方/我方日志 L245/L249 |
| **162** | ★★ **"等它跑完再读日志"这个自然做法与一次性 token 的寿命直接冲突，本轮已三次白烧**：单次 token 有效期约 5 分钟，而一次真机实验要 2–11 分钟（冒烟 ~4 min、A/B 单组 ~110 s、6 组 ~11 min）。我在同一轮里 `Start-Sleep` 等训练跑完再读，**三次都因 token 在等待期间失效而读不到结果**（其中两次连"结果已经产生"这件事都没能确认） | ① **等待的时长超过了凭证的寿命** ⇒ "发出命令—等待—读取"这个顺序在同一轮内根本不成立；② 更坏的是它会**让人误判**：命令其实已经成功发出、机器也在正常跑，只是我读不到，看起来像"没反应"；③ 失败信息此前还会被误读成"主机不可达"（坑 158 修好前，两种原因共用同一句话）；④ 与坑 159/160 同族：**工具/流程的性质与我的默认习惯不匹配时，要先改流程，不要靠多试几次** | ① 固定模式：**长任务一律 `bg` 启动后立刻返回**，结果留到**下一次 token** 读 —— `push+bg` 不等待（本轮已实测可行）；② 一轮内只做"**即发即用**"的命令（push / bg / 读日志 / 落盘断言），**不做 sleep**；③ 需要等待时把等待**交给机器**：链路脚本自带"等 NPU 空闲 + 失败即停"（`ab_queue.sh` / `official_align.sh` / `dp2_v3_run.sh` 均已如此），而不是让 token 空转；④ 因此**每轮只交付"已落盘的东西"**，不承诺"这一轮就能读到结果" | 三次实测：① `p61b_launch` 后 `sleep 110` → token 失效，结果下一轮才读到；② `dp2_v3_run` 那次 `sleep 210 + tail` **侥幸成功**（正因如此**不能依赖**，它只是偶然而非机制）；③ 本轮 `official_align` 后 `sleep 260` → 失效、未读到（链路本身仍在机器上正常跑）。坑 158 修好后，这类失败现在明确打印 `TOKEN_REJECTED —— 平台侧拒绝该 token……**这一条不能用来判断目标主机是否可达**`，与"实例侧不可达"不再混淆 |
| **163** | ★★ **"路径"是断言而不是事实；而我的自检与代码共享同一个错误假设，结构上抓不到它**。新写的官方对齐审计器第一次运行就报 `**UNDECLARED** data.dataset_param.basic_parameters.image_max_pixels official=262144 run=<absent>`，看起来像派生配置漏了官方键。查真实产物发现：**该键就在 `data.dataset_param.preprocess_parameters.image_max_pixels: 262144`（与官方一致）** —— 错的是**我的路径**：我在 `COMPARE_KEYS` 与审计器里都写成了 `basic_parameters` | ① 路径是**按直觉写的**（"它应该在 basic_parameters 里"），**从未拿真实产物核对过**；② 后果远不止"读不到"：该键在计划与运行里**永远 `absent`** ⇒ 我新加的"缺必要字段即 fail"规则会让 `plan_consistent` **恒为 False**，而"必要字段缺失"这条噪声还会**掩盖真正的偏离**（真偏离被淹没）；③ **最关键的一层**：自检之所以全绿，是因为**合成计划也是我按同一个错误路径手写的** ⇒ 测试与代码**共享同一个错误假设**，"用同源假设写的测试只能验证假设的自洽性"；④ 最终抓出它的是**另一个来源的检查**（新审计器 × 真实产物 × 官方锚点逐键比对），不是自检 | ① 修正路径为 `data.dataset_param.preprocess_parameters.image_max_pixels`；② `plan_consistent` 改用新增的 `dig_any()`：**精确路径优先，失败则退化为"叶子名唯一匹配"**，并把**实际解析路径**与 `how ∈ {exact, leaf, absent, ambiguous}` 全部落盘 —— 回退不是错误，但**必须可见**（否则路径笔误又变成看不见的东西）；③ 自检的合成计划**照真实产物结构重写**（把 `image_max_pixels` 放到 `preprocess_parameters`），并新增两条：**错误路径必须仍能解析并标 `leaf`**、真正缺失必须判 `absent`；④ 把用户指令"尽量与官方一致"做成**会失败的检查** `align_audit.py`：官方锚点 13 项逐键比对，凡与官方不同者必须已在 `versions.lock` 偏离清单登记，否则**退出码 1** | 审计修正前 `UNDECLARED=1`、修正后 `ALIGNED=9 / DECLARED=4 / UNDECLARED=0` → `ALIGN_AUDIT_OK`；真实派生配置 `mock_align2048.yaml` L34 = `image_max_pixels: 262144`（位于 `preprocess_parameters`）；`_envcompat.py --selftest` → `SELFCHECK_OK items=23 failed=0`（含 `路径回退：错误路径仍可解析并标 leaf` 与 `真正缺失 → absent` 两例） |
| **164** | ★★ **"兜底"把上游失败静默转换成了"换一份产物继续跑"，而换来的恰好是已知病态样本**：`align_oneimage.sh` 里期望的 mock 不存在时，我写的是"取目录里**最新**的 json"。真机第一次跑：生成器因缺 `--tokenizer_path` 失败（`OSError: Repo id must be in the form 'repo_name' or 'namespace/repo_name': '/home/weights/Qwen3.5-35B-A3B/'`），兜底随即选中 **`mock_data_pic_num_256_textlen_512.json`** —— 正是坑 144 里"**256 张图塞进一个样本**"的病态样本 ⇒ 链路拿**已知最坏**的输入继续往下跑（派生、冒烟、基线） | ① 兜底只看"**存在性 + 时间戳**"，**不看"这份产物是否被验证过"**；② 它把上游失败**静默转换**成"换个输入继续"——**换了产物却没有任何人知道**（坑 112/143 同族：多源真相）；③ 更根本的是我又犯了**猜接口**：生成器的参数我照 bringup 注释里的 `--num_pics/--text_length` 猜，漏了必需参数 `--tokenizer_path`（`--help` 里写明 "HuggingFace config path"），而它的默认值指向这台机器上不存在的 `/home/weights/Qwen3.5-35B-A3B/`；④ 与坑 159/160 同族：**流程里的"贴心兜底"往往正是假结论的来源** | ① **兜底改为拒绝**：期望文件不存在 → 打印目录清单 + `gen.log` 尾部后**中止**，绝不替代成另一份来路不同的产物；② **把"已知坏例"编码进守卫**：显式拒绝 `*pic_num_256*`（坑 144 的病态样本），不再靠人记得；③ 生成器改为**按 `--help` 的真实参数**显式传 `--tokenizer_path /root/Qwen3.5-0.8B-hf`，并在调用前**断言该目录存在**；④ 生成后**断言结构**：首样本的图片引用数必须 `== 1`（与 COCO/LLaVA 一致），否则中止；⑤ 修正后的链路 `align_oneimage2.sh` 三级都带断言 | 现场：`O1_GEN_RC=1` 与 `O1_GEN_NAME_FALLBACK 期望名不存在 → 用最新 json: mock_data_pic_num_256_textlen_512.json`；`gen.log` 尾部 `OSError: Repo id must be in the form ...`；`--help` 明确列出 `--tokenizer_path`（"HuggingFace config path"）。**附带确认**：`--help` 里 `--num_pics` 的说明是 "number of mocked pictures" ⇒ **独立确认 `pic_num` = 每样本图片数**，即坑 144 的诊断（256 图/样本）成立 |
| **165** | ★★ **★ 本条根因曾被我误判，已更正（更正理由见"根因"栏末）**：1 图 mock + **全套官方参数**的 100 步基线，**在 96/100 处设备 OOM 崩掉** —— `iteration 96/100 | elapsed time per iteration (ms): 5704.2`（比常态慢 12 倍）紧随 `ERR00006 PTA memory error`、`Failures: <NO_OTHER_FAILURES>`、`ChildFailedError`、算子 `aclnnMatmul`；判据 `valid=False steps=96 steps_complete=False`。**关键观察：512 样本与 1024 样本两次都停在 96** | **真实原因：设备显存随步数增长** —— 步 2 时 `max allocated` 已到 **24435.0 MB**（`reserved 26450.0 MB`），到步 96 撞满 64 GB/die ⇒ OOM。**崩溃点由「显存增长速度 × 工作负载大小」决定**：4 图/样本时第 **23** 步就撞顶，1 图/样本时撑到第 **96** 步，**与样本数无关**。★★ **我更正的部分**：我最初写的是"mock 只有 512 样本、数据耗尽"—— **这是错的**，因为 1024 样本（每 epoch 128 步）同样停在 96；**而我当时只看了 driver 的 `TRAIN_OK` 与"96 < 100"就顺手把原因归给"数据不够"**，即**用最省力的解释替代了测量**（`TRAIN_OK` 本身也是假的，见坑 167）| ① 结论文本更正为**设备 OOM**，并保留"我误判过"这一痕迹（删掉痕迹等于让同族错误可以再犯）；② 显存增长需**按步采样**才能定位来源（本轮只有步 2 的一次分配器打印 + 61 探针的设备级 HBM，`max` 已到 97.5%）；③ 就本项目当前状态而言：官方值 `skip_gdn_recompute: True`（跳过重算 ⇒ 留激活）在这台机器 + 本 mock 负载下**撑不住 100 步**，而 `false` 的旧配置跑完过 100 步 ⇒ 要用官方值必须**缩小工作负载或提高可用显存**，**不能靠"多跑几次碰运气"**；④ 判据维持严格（`1..N` 恰好一次），因为残缺运行上的窗口指标会污染所有下游结论 | `train.log` 619–624 行（`95/100 449.8ms` → `96/100 5704.2ms` → `ERR00006`）、`grep operator` → `current working operator name is aclnnMatmul`、`(after 2 iterations) allocated 5190.7 / max allocated 24435.0 / reserved 26450.0 MB`；两次样本数对照 512→96、1024→96；`cgroup_peak_pct=10.5 / oom_kill_delta=0`（**宿主内存完全正常，是设备显存**） |
| **166** | ★★ **"同一语义两个来源"换了个维度就是"同一脚本两个目录"，而自检照样全绿**：在官方对齐配置上跑 `pregather` A/B，`59_config_ab.py` 立刻 `rc=2`（argparse 参数不识别），而**同一份代码**在本地 `--selftest 29/29` 与 `--selftest-orchestration 6/6` 都全绿 | ① 新版 `59`（含 `--numeric-steps`）我只推到了 **Skill 目录** `/root/qwen35-ascend-migrator/scripts/`，而链路执行的是 **`/root/ops/59_config_ab.py`（旧版）** ⇒ **同一脚本在机器上存在两份且新旧不同**（与坑 111/163 同族，只是换成了**目录**维度）；② 我当初把工具放 `/root/ops` 是为了"不改动机器上已安装的 Skill"，把新版推 Skill 是为了"让 Skill 完整"——**两个各自都合理的目的，合起来制造了双源**；③ **最隐蔽的是自检全绿**：自检跑的是**新**那份，真机执行路径用的是**旧**那份 ⇒ 又一出"**测试对象 ≠ 运行对象**"（坑 163 的同族）；④ `rc=2` 是 argparse 的错误码，且报错落在 `driver.log` 而不是 A/B 输出里，容易被误读成"配置问题" | ① 定规矩：**工具只部署一处并被链路引用**；确需两处时**必须同版本**，本轮处置是把 `59`/`61`/`50_train`/`_envcompat` 四个脚本**同步到 `/root/ops`**；② 用 **`--help` 验证新参数真的存在**（`--numeric-steps` 出现在 usage 里即证），而不是靠"我记得推过"；③ 更一般的判据：**任何"同一脚本存在多份"的部署都必须有哈希对照**；④ 顺带把这条写进"工具部署"的检查清单（与坑 112 的"多源真相"同表） | `/root/ab_pregather_1img.sh.run.log` 的 `AB1_RC=2` 且**没有** `summary.json`；同步后 `/root/ops/59_config_ab.py --help` 明确列出 `--numeric-steps NUMERIC_STEPS`（并含 `--selftest-orchestration`）；本地 `P59_SELFTEST_OK cases=29 failed=0`、`P59_ORCH_SELFTEST cases=6 failed=0` |
| **167** | ★★ **自己的驱动在"失败日志"上打印成功**：`50_train.py` 在一次**以 `ChildFailedError` 收尾**的运行上打印 **`TRAIN_OK iters=96`**，而同一份 `driver.log` 里 `train_rc=1`。于是一次真实失败同时得到两个相反结论 —— **驱动说成功、探针说失败** | ① `full = "bash -lc '<env>; <train_cmd>; echo train_rc=$?'"` —— **最后一条命令是 `echo`，它永远成功** ⇒ `subprocess.call(full, shell=True)` 的返回值 **恒为 0**，而 `TRAIN_OK` 的判据正是 `if rc == 0 and n > 0`；② 打印出来的 `train_rc=1` 是**真的**（`59`/`61` 从 `driver.log` 抓 `train_rc=(\d+)`，所以它们判对了）⇒ **只有 `50_train.py` 自己的结论错**，是"同一语义两个来源、其中一个错"的又一例（坑 111/163/166 同族）；③ 它的危害已经发生：**坑 165 的误判就是被这个 `TRAIN_OK` 带偏的**（我以为运行"正常结束"） | ① 末尾改为 `rc=$?; echo train_rc=$rc; exit $rc` —— **让"打印的值"与"进程返回码"同源**，从根上消除"echo 吞掉返回码"；② `TRAIN_OK` 再加一道**独立**护栏：必须**不含任何失败标志**（`ChildFailedError` / `trainer.py FAILED` / `Traceback (most recent call last)` / `ERR00006` / `ERR99999` / `OutOfMemoryError`），命中则打印"日志含失败标志：…"并走 `TRAIN_FAIL`；③ 该护栏与返回码**互为独立证据**（返回码错了一次、"有 iteration 行"更不可靠——跑到一半崩掉的日志同样有 iteration 行） | 现场：`driver.log` 的 `TRAIN_OK iters=96` 与 `train.log` 结尾 `mindspeed_mm/fsdp/train/trainer.py FAILED` + `Failures: <NO_OTHER_FAILURES>` + `ChildFailedError` 并存；`59`/`61` 由 `train_rc=(\d+)` 得 1 ⇒ `valid=False`（两者结论相反）；修复后 `AST_OK`、`DIAG_SELFTEST_OK samples=18 rules=17 uncovered=0`、`50_train.py` 无未定义名 |
| **168** | ★★ **"token 自动化的最后一公里"里藏着一个前缀叠加缺陷：走 `auto` 读到的 token 会被加两次 `jt_`，而报错文本会把它说成"token 可能已过期"**。★ **本条未在现场触发 —— 它是我在"用之前"通读 `ssh_tool.py` 时看出来的**：平台给的连接命令是 `ssh -J jt_7443…863:DF15…E064@113.47.8.48:2234 root@199.103.55.150`，其中 **`jt_<ID>:<HEX>` 整体就是跳板用户名**；而本脚本的历史接口约定是 token 参数写 `<ID>:<HEX>`、"脚本自动加 jt_ 前缀"（`connect()` 里 `auth_none("jt_" + token)`），可 `token_agent.py` 的 `RE_SSH.group(1)` 抓到的却是**带前缀**的 `jt_…` ⇒ 一旦走 `auto`，必然拼成 `jt_jt_…`，跳板认证失败 | ① **两个各自正确的约定，接起来就错**：一边（脚本接口）约定"不带前缀"，另一边（收割器）保存的是**平台原文**（带前缀），中间没有任何一方做归一化；② 危害不在"连不上"，而在**报错文本把人送错方向**：`JUMP_AUTH_FAIL —— token 可能已过期（约 5 分钟有效，请重新获取）` 与"我这 token 是**刚刚**收割来的"**直接冲突**，最自然的反应是**再去换一个 token**，于是同一个缺陷可以反复烧掉好几轮 token（与坑 158 同族：**两种原因混在一句话里**）；③ 更隐蔽的一层：上一轮所谓"端到端测试通过"用的是**我自己手打的假 token**（形态由我决定），而真实路径上 token 的形态由**平台**决定 —— **一旦两者形态不同，那次绿灯根本没有覆盖真实路径**（坑 163/166 同族：**测试对象 ≠ 运行对象**） | ① 统一归一化：新增 `_jump_user(tok)` —— **有 `jt_` 就先剥掉、再加一次**（幂等，两种形态都对），`connect()` 改用它；② 顺手把"最后一公里"的人工步骤也删掉：store 里本来就存着 `jump`(host:port)、`target`(user@ip)、`password`，新增 `_apply_store_conn()` 在 `auto` 时**自动补进连接参数**（**不覆盖显式 env**），不再需要人工 export 三样、也不会因为漏一样就报"缺少目标机密码"；③ 连接前把生效参数打印出来（`jump=… target=… pw=已设/未设`），让"这次到底连的是哪台机"变成可见事实 | 真实连接命令原文：`ssh -J jt_7443331D9F63A6C19429C863:DF15…E064@113.47.8.48:2234 root@199.103.55.150`；`token_agent.py --harvest-file` → `TOKEN_AGENT_SAVED … token=jt_7443331D9…`（**带前缀**，证实该缺陷路径必然命中）；修复后同一次连接打印 `连接参数(来自 store)：jump=113.47.8.48:2234 target=root@199.103.55.150 pw=已设`；该次连接对**已过期**的 token 正确报 `TOKEN_REJECTED`（code 1，token 侧），与 `TARGET_UNREACHABLE`（code 2，实例侧）分开 |
| **169** | ★★ **"机器"是平台上的一个会被自动改状态的对象，不是我的资源；而"连不通"的真因（实例被关机）SSH 判据永远无法自证**。控制台截图确认：`DevEnv_232070`（昇腾 **A3**，CANN 9.1）**运行中**，它的「SSH 直连」命令目标正是 `199.103.55.150` ⇒ 我一直在用的机器 B = `DevEnv_232070`；另有 `DevEnv_201412`（昇腾 **A2**）与 `DevEnv_895801`（昇腾 **A3**）**均已关机**。**推断（未证实）**：连日不通的机器 A（`199.103.53.187`）= `DevEnv_895801` —— 与它"已关机"自洽，但**我没有直接证据**（控制台只对运行中的环境显示连接命令/IP），所以这是**待回控制台核对的推断，不是结论** | ① 我把"机器"当成了**稳定对象**来规划实验（跨机复现、双机对照、把长链路排在后面），而实际它是**平台上会被自动关机/自动释放的对象**；② SSH 侧的判据只能区分"token 侧 / 实例侧"（坑 158），而**"实例被关机"与"实例被重建换了 IP"在 `Connect failed` 上完全同形** ⇒ 拿 SSH 判据去追这件事是**在错误的层面上找答案**，答案只能在控制台的实例状态里；③ 平台自身还写明了两条会自动改状态的规则（控制台提示**大意、非逐字**：**连接后 1 小时未使用的环境将自动关机；连续 2 周未开机的环境将被自动释放**）⇒ **"留到最后再跑"和"长期挂着不管"这两种安排都不可靠** | ① 落盘**环境台账**（环境名 / 实例类型 / CANN / 状态 / 对应哪台机器 / 当前 IP），把"这台机器到底是哪台"变成**可查的事实**而不是记忆（与坑 112"多源真相"同族）；② 长链路（`ab_queue.sh` 这类多步队列）必须按"**机器随时会被关掉**"来设计：**每步产物立即落盘、可断点重跑、关键结论不排在最后**；③ 关机后磁盘是否保留、重建后 IP 是否变 —— **我不知道**，因此**不得假设中间产物还在**，重要产物一律及时拉回本地；④ **开机 `DevEnv_895801` 就能拿到同类第二台 A3**（且此前"机器 A 未必是双 die、只能给 1 张卡"的疑虑需要用实测重新验证）⇒ 跨机复现、dp1/dp2 双机对照、官方几何的第二组独立数据都重新变得可行 | 控制台「在线开发」列表：`DevEnv_895801` 昇腾A3 CANN 9.1 **已关机**、`DevEnv_232070` 昇腾A3 CANN 9.1 运行中、`DevEnv_201412` 昇腾A2 已关机；`DevEnv_232070` 的 SSH 直连命令目标 `root@199.103.55.150`（= 机器 B），故 B 的环境名可确证；平台提示原文（据截图，大意非逐字）「连接1小时未使用的环境将自动关机；连续2周未开机的环境将自动释放」；本机侧记录：机器 A 用**全新 token** 仍复现 `TARGET_UNREACHABLE ChannelException(2,'Connect failed')` |
| **170** | ★★★ **"官方对齐审计"一路打印全绿，而它**从未看过**最能解释 OOM 的那几个键 —— 绿灯是"没看"出来的**。我把 `align_audit.py` 的唯一锚点来源从**我手抄的 13 键表**换成**官方日志自己的配置 dump**（`official_baseline.log:47-308` 本来就是一份可 `yaml.safe_load` 的完整配置），首次运行立刻报出 **13 处从未被比对过的差异**。其中 6 处方向一致 —— **官方开启、我方全部关闭的省显存开关**：`features.recompute=True`（我方 False）、`features.enable_chunk_loss=True`（False）、`features.enable_activation_offload=True`（False）、`model.skip_flash_attn_recompute=True`（False）、`parallel.fsdp_plan.reshard_after_forward=True`（False），外加方向相反的一项 `parallel.fsdp_plan.pregather`（官方 False / 我方 True）。另有 `parallel.fully_shard_parallel_size`（官方 2 / 我方 `auto`）、`training.save_interval`（官方 100 / 我方 10000）及 6 处纯路径差异 | ① **我一直在优化"闸门会不会失败"，却没优化"闸门看没看"**：`UNDECLARED=0` 有两种来源 —— "真的没有偏离"和"**这些键根本不在我的清单里**" —— 而两者打印出来**一模一样**。这正是坑 163 的同族，只是从"路径写错"升级成"**清单本身不全**"：手抄的锚点表，覆盖面等于**我的记性**；② 更糟的是我**刚刚**才为 `model.skip_gdn_recompute` 补过这个漏（同一张表、同一个理由），说明"逐条补键"这条路本身就是错的 —— 补一个漏一个；③ **对结论的杀伤**：坑 165 我把第 96 步 OOM 归因于官方值 `skip_gdn_recompute=True`"留激活"，而真相是**我方把官方五条省显存开关全关了**（激活不重算、不切块、不上 CPU、FA 不重算、参数不重新分片）⇒ 每步激活全留；`skip_gdn_recompute` 只是**同一方向上的第六个**，不是主因；④ 由此还暴露一个更根本的东西：我方的"快"很可能是**用显存换来的**（换了工作点），所以任何"快了多少"的数字**必须与显存峰值一起报**，否则是把"挪了工作点"说成"优化"；⑤ `parallel.fully_shard_parallel_size: auto` 与官方 `2` 是否等价**我从未实测** —— 若 `auto` 解析成 1，则此前所有"官方几何"的标注都要重标 | ① **锚点改为推导**：新增 `load_official_dump()` 从官方日志里自动截取并解析那段配置 dump，`flatten()` 摊平成点号路径后与本次运行配置**逐键比对** —— 覆盖面从此由**官方日志**决定，不再由我的记性决定；② **推导失败即拒绝给绿灯**（`ALIGN_AUDIT_FAIL`，绝不退回"只看手抄表"）；③ 新判据输出必须**同时报覆盖面**：`两侧都有（真的比过）94 键 / 只有官方有 119 键 / 只有我方有 1 键`，并明确写出"只有官方有的键我方走**上游默认值**，本审计**无法判定**是否一致 ⇒ **不声称一致**" —— **只报绿灯不报覆盖面的闸门，等于假安全**；④ 13 处差异**逐条登记**进 `versions.lock` 偏离清单（登记 ≠ 认可为最优，只表示"我们知道它不同"），其中 `features.*` 组明确标注"**是第 96 步 OOM 的主嫌**"、`fully_shard_parallel_size` 标注"**未验证前不得声称几何一致**"；⑤ 相邻修复：`_envcompat.COMPARE_KEYS` 补 4 个 `parallel.fsdp_plan.*` 键 —— **自检的正例立刻失败**（合成计划里没有这些键 ⇒ `plan_consistent=False`），照真实产物补齐合成计划后回到 `SELFCHECK_OK items=23 failed=0`（比对项 13 → **17 项**）；⑥ 实验设计随之改：把 6 个开关**作为一组**做 A/B（`63_two_config_ab.py`，声明差异必须逐键相符），并**强制连同显存峰值一起读** | 修复前：`ALIGNED=9 / DECLARED=4 / UNDECLARED=0 → ALIGN_AUDIT_OK`（**13 处差异一处没看见**）；修复后首跑：`真的比过 94 键 / 只有官方有 119 键`、`UNDECLARED 13 项`（列全）；登记后：`值不同且已登记 16 键 / 值不同且未登记 0 键 / 未比对 119 键 → ALIGN_AUDIT_OK`。官方侧原文：`official_baseline.log:57 reshard_after_forward: True` 与 `train.log:57 False`；`sk04_judge/evidence/officialA_configuration_details.txt:49` 同为 `True`（**两个独立官方来源**）；我方 `config/templates/qwen3_5_0_8B_official_geometry_final.yaml:6 reshard_after_forward: false` 与 `:88 skip_gdn_recompute: True` 同文件并存 |
| **171** | ★★ **"顺手读一下日志"把一个 token 烧掉了 —— 因为远端命令里的引号在上位被吞掉，`grep -E "A|B|C"` 到了远端变成管道加"要执行的命令"**。现象：读链路进度的命令返回 `rc=-1`，输出停在 `===MABSUM===`，stderr 是 `bash: line 1: MAB_RUN_DONE: command not found`、`bash: line 1: MAB_CFG: command not found`，后面的 `===OUT===`/`===DISK===` 段**根本没打印** | ① 调用链是 PowerShell → python(`ssh_tool.py`) → paramiko → 远端 bash。PowerShell 把含双引号的字符串交给**原生进程**时**不做转义**，双引号被吞 ⇒ 远端收到的是 `grep -E MAB_SUM|MAB_RUN_DONE|MAB_CFG /root/...`，`|` 成了管道、`MAB_RUN_DONE`/`MAB_CFG` 成了命令；② **这是第二次同类**（此前"带括号/引号的远端命令曾挂 180 s"），我当时的结论是"避免括号与引号"，但**没有把它变成工具里的东西** —— 于是同一条教训只活在记忆里，记忆一松就复发（与坑 163/166 同族：**没做成闸门的教训等于没有教训**）；③ 更刺眼的是：**我本来就有正确做法**（复杂命令一律写进远端脚本文件、`bg/run` 只调脚本名 —— `after_mab.sh` 整个就是这么写的），却在这类"顺手读一下"的小命令上偷懒；④ 这类命令的代价恰好最高：它出现在**每次读结果的路径**上，而读结果正是每个 token 必做的第一步 | ① `ssh_tool.py run` 增加**引号告警** `CMD_QUOTES_WARNING`（不拒绝，因为确有必须用引号的场景，但把风险摆到眼前）；② 定规矩写进用法：**远端命令里不出现引号**；需要 `|`、`$()`、多段逻辑时——**写进脚本文件再调**（本轮 `===CHAIN===` 那些段之所以成功，正是因为它们来自脚本文件而非命令行）；③ 读结果也照此办理：后续把"读某次实验的裁决"做成远端只读脚本（`_ops/show_result.sh`），一个 token 调一个脚本名即可，不再手拼 grep | 现场：`python ssh_tool.py run auto $cmd` 输出 `rc=-1`、`===MABSUM===` 之后是 `bash: line 1: MAB_RUN_DONE: command not found`；同一 token 的**前一段**（不含引号的 `tail -n 55`、`kill -0`）**全部正常** ⇒ 变量只有"引号"这一个。对照：`===CHAIN===` 段完整打印了 40 余行日志 |
| **172** | ★★ **我差点覆盖一个"正在执行中"的 bash 脚本**：链路 `after_mab.sh` 在远端跑到第 2 组时，我在本地改了它（`hostname`→`uname -n`）。如果按常规"改完就推"，`push` 会**覆盖远端正在被 bash 读取的文件** | ① **bash 是按需读取脚本的**（不是一次性全部读入内存）：脚本边执行边从文件读后续命令 ⇒ 执行中覆盖文件，后续行可能来自**新版本**、前面的行来自**旧版本**，得到"半新半旧"的执行体，行为未定义（可能语法错、可能跳到错误的字节偏移、可能静默执行错的分支）；② 危险在于它**看起来无害**："我只是改了一行注释/一个主机名"；③ 我这轮的运气是**先查了哈希再决定推什么**，而哈希对照（坑 166 的处置）恰好也让"这个文件正在被用"这件事浮出水面 —— **闸门顺手救了另一类错**；④ 同族更早的隐患：任何"部署 = 覆盖"的思路，在**长跑任务正在用这些文件**时都不安全（`59`/`61`/`50_train` 都是被长跑链路调用的） | ① 规矩：**脚本正在远端运行时，只改本地、不推送**；推送前先确认该脚本（及其调用的工具）当前没有被运行中的任务使用（`pgrep -f`/pid 文件）；② 无状态的服务型工具（`59`/`63`）在**两次调用之间**更新是安全的，但要在**队列运行期间**避免（本轮 `59`/`63`/`_envcompat` 是在链路启动**之前**推的 ✓）；③ 长链路一次跑完再更新工具：把"工具升级"与"实验执行"分开成两个时间窗；④ 这条与坑 166 配套：**哈希对照不仅证明"两处同版本"，也提醒"这个文件正被谁用"** | 现场：`after_mab.sh` PID 217009 自 14:36:48 持续运行（`alive_rc=0`，`T63_END 00_A ... wall=145s` 后仍在继续）；同期本地对该文件的编辑时间晚于其启动时间；**未发生实际覆盖**（本轮推送发生在启动之前，之后只改本地） |
| **173** | ★★ **闸门的结论取决于"你站在哪个目录"**：`prepush_check.py` 的 G3（编译 + `--help` 加载冒烟）报 `65_scene.py: Traceback → PermissionError: [WinError 5] 拒绝访问。: 'out'`，看起来像新引入的交付缺陷；而同一个 G3 在别的目录下调用是**全绿**的，我此前也确实见过它 24/24 | ① G3 逐个跑 `python <脚本> --help`，但那一处**没有传 `cwd`** ⇒ 脚本往**相对路径** `out/` 写时就落在**调用者的当前目录**（这次是会话工作目录 `D:\昇腾项目`）⇒ `os.mkdir('out')` 拒绝访问；② **同一个文件里其余闸门都显式传了 `cwd=sk`**，只有这一处漏了 —— 与坑 37/39/154 同族：**公共规范已经存在，使用处没跟上**，只是这次的隐含变量不是命令/文件而是 **cwd**；③ 诊断过程本身也值得记：我先怀疑"`out` 目录权限坏了"，实测 `CREATE_OK`（可写、`out/scene` 已存在）⇒ **排除了第一个假设**；再去读脚本才发现 `65_scene.py` **根本没有 argparse**、`--help` 也会真的执行 `main()` —— 即"拿 `--help` 做加载冒烟"这个手段自带前提（脚本必须把 `--help` 交给 argparse），前提不成立时它会**把正常脚本报成缺陷**；④ 若我当时按"权限问题"去改 ACL，就会在**没病的地方动刀**，而真正的病（cwd 未钉死）会留在那里等下一次换个目录再咬一口 | ① G3 的冒烟调用改为 `run([py, p, "--help"], cwd=sk, timeout=180)` —— **与同文件其余闸门一致**，结论从此与调用位置无关；② 在代码处写明"该手段自带前提"这一边界，不静默跳过（宁可把不满足前提的脚本报出来，也不要假装检查过）；③ 诊断顺序固化为"**先实测被怀疑的对象，再改它**"：这次 `CREATE_OK` 一行就把"权限"假设证伪，避免了在正确的地方做错误的修改；④ 顺带确认 `65_scene.py` 的行为（无 argparse、`--help` 会跑 main）**属于既有事实、不是本次编辑引入**，故不改它，只在闸门处记录 | 修复前：`FAIL G3 … 65_scene.py: Traceback → PermissionError: [WinError 5] 拒绝访问。: 'out'`，`合计 24 闸门：PASS=22 FAIL=2`；反例证据：`(Get-Item out)` → `Mode d-----`、`New-Item out/_permtest` → `CREATE_OK`（⇒ **不是**权限问题）、`65_scene.py` 第 119-120 行 `if __name__ == "__main__": main()` 且全文无 argparse；修复后 `PASS G3 … 36 个 .py 编译通过；加载冒烟跳过 3 个（本机缺环境依赖：30_verify_ops/60_bench/verify_ops 缺 torch）`，并连同重打包后 `PASS G22 三方一致性 COPIES_OK` 一起达到 `PREPUSH_OK 24/24` |
| **174** | ★★★ **"绿灯"是隔壁程序给的 —— 我的接收端根本没在应答，而 CORS 预检"通过了"**。为让浏览器页面把 SSH 连接信息送回本机，我写了本地接收端 `token_sink.py`，绑到 `127.0.0.1:8765` 后自测：`OPTIONS` 预检返回 **204 + Access-Control-Allow-Origin**，看起来"CORS 通了、方案可行"。**但两个细节对不上**：响应里的 `Access-Control-Allow-Methods` 是 `GET, POST, OPTIONS`，而我代码里写的是 `POST, OPTIONS`；`POST` 回的是 `{"error": "not found"}`，而我的响应形状是 `{"ok":…, "reason":…}`。查端口：`port=8765 pid=320 python`（22:52 起，我的）与 `port=8765 pid=37708 阖壹.exe`（22:34 起，**无关程序**）**同时**在监听 ⇒ 那些响应**全部来自隔壁程序** | ① **Windows 的 `SO_REUSEADDR` 语义与 Linux 不同**：它允许**两个进程绑同一端口**，于是"绑定成功"不能证明"这个端口归我" —— 而我把"绑定成功"当成了"我在服务"（与坑 137/158 同族：**判据混在一句话里**，只是这次混进的是"谁在应答"）；② **最危险的是这个假成功的形态**：它给的是**最想要的那个结论**（"CORS 通、链路可用"），而不是一个失败 —— 若我不核对响应内容，就会拿别人服务器的回答宣布自己的接收端修好了（坑 35/159 同族：**假阳性比失败更危险**）；③ 让我免于误判的不是"我小心"，而是**两处可核对的具体差异**（响应头取值与我写的不同、JSON 形状不同）—— 也就是说，**"我预期什么形状"必须写在手上，才可能发现"回来的不是我的"**；④ 修的过程中同一个模块又暴露第二个顺序错误：我加了"绑上后自证"，自证却跑在 `serve_forever()` **之前** ⇒ TCP 层已握手（进程在 listen）但无人受理 ⇒ 自证自己 `TimeoutError`。自证**拒绝启动是对的**（fail-closed），错的是我的实现顺序 | ① 关掉 `SO_REUSEADDR`（`class _Srv(ThreadingHTTPServer): allow_reuse_address = False`）⇒ 端口被占时**绑定直接失败**：实测拿被占的 8765 启动报 `OSError: [WinError 10048]`，rc=1；② 绑定后**自证"应答的是我"**：请求 `/` 并核对是否含本服务特有的字段（`{"ok":true,"hint":…}`），不是就打印 `SINK_PORT_NOT_MINE` 并**拒绝启动**；③ 修正顺序：**先 `serve_forever()` 起在后台线程，再自证**，通过后才打印监听地址；④ 默认端口从 8765 改成 **8791**，并在参数说明里写明"刻意避开 8765（那口被无关程序占着）"；⑤ 安全面同时收紧（因为收到的内容会被我拿去 SSH）：**只绑 127.0.0.1**、**必须带共享密钥**（否则任意网页都能 POST 到 localhost，而浏览器允许 https 页面访问 localhost）、**目的地白名单**（跳板 `113.47.8.48:2234`、目标 `199.103.55.150`，不在白名单一律拒收）—— 这三条对应同一个风险：**输入会变成"我要连到哪台机器"** | 假成功现场：`Access-Control-Allow-Methods: GET, POST, OPTIONS`（我写的是 `POST, OPTIONS`）、`{"error": "not found"}`（非我的形状）、`Get-NetTCPConnection -LocalPort 8765` → `pid=320 python` 与 `pid=37708 阖壹.exe` **同时 LISTENING**；顺序错误现场：`SINK_SELFCHECK_FAIL 绑定后自证失败：TimeoutError: timed out`（服务线程未起）；修复后实测：占用端口 → `OSError: [WinError 10048]` rc=1；8791 上 `SINK_LISTEN http://127.0.0.1:8791/token/e2etest123`、预检 204（`ACAM: POST, OPTIONS` **与我的实现一致**）、正常 POST → `{"ok": true, "saved": true, "env": "DevEnv_232070", "target": "root@199.103.55.150", "token_prefix": "jt_FFFF0001:AA"}`、密钥错 → 403、别的目标机 → `403 目标 10.0.0.9 不在白名单 ['199.103.55.150']`；`--selftest` → `SINK_SELFTEST cases=8 failed=0`；测试后 store 已还原，未留假 token |
| **175** | ★★ **"本机自证"被 VPN 代理接管 —— 而代理配置**一个环境变量都没设**。本地接收端 `token_sink.py` 绑定后要做"应答的是不是我"的自证（坑 174 的处置），请求 `http://127.0.0.1:8791/` 却反复 `TimeoutError`。查环境：`http_proxy` / `HTTP_PROXY` / `https_proxy` / `all_proxy` / `no_proxy` **全部未设**；但 `urllib.request.getproxies()` 返回 `{'http': 'http://127.0.0.1:51081', 'https': …, 'ftp': …}` —— 因为 **Windows 上 `getproxies()` 会回退读注册表的 WinINET 设置**（`HKCU\…\Internet Settings` 的 `ProxyEnable/ProxyServer`） | ① 我按"环境变量里没有代理 ⇒ 不走代理"这个**跨平台直觉**判断，而 Windows 的实现**多了一个来源**（注册表）—— 与坑 37/39/154/173 同族：**隐含平台假设**，只是这次的隐含项是"代理从哪来"；② **自证的对象恰恰是"这个本地端口"**，被代理接管后自证就失去了意义（代理去连它自己够不着的地方 ⇒ 超时 ⇒ 误判成"服务没起来"）；③ 最能说明问题的是**同一语义两条实现走了两条路**：先前用 `curl` 做的跨源联调**全部通过**，而 python 的自证全挂 —— 因为 **`curl` 不读注册表代理**。这与坑 112/166"同一语义两个来源"同族，只是两条来源是**两个 HTTP 客户端库**；④ 若我不追这一步，会得出"我的接收端不稳定/时好时坏"这种**错误且无法收敛**的结论（坑 159/174 同族：假阳性与假阴性比失败更危险） | ① 自证改用**显式绕过代理**的 opener：`urllib.request.build_opener(urllib.request.ProxyHandler({}))` —— 本机自证**绝不能被代理接管**，并在代码注释里写明这条理由；② 判据不变（仍核对响应是否含本服务特有字段、仍 `SINK_PORT_NOT_MINE` 即拒启），只是把"路由"这条隐含前提钉死；③ 记录"诊断顺序"：先**分别实测两条实现**（curl 通 / python 不通）⇒ 差异必然在库的差异上，而不是"服务时好时坏"——**先证伪"对象有病"，再改对象**（与坑 173 同一条经验）；④ 顺带在脚本 docstring 里写明本机代理现状（`127.0.0.1:51081`，dotsvpn 提供；该代理对公网可用、对平台域名 30s 超时） | 环境变量全 `<未设>`；`python -c "import urllib.request as u; print(u.getproxies())"` → `{'http': 'http://127.0.0.1:51081', 'https': 'http://127.0.0.1:51081', 'ftp': 'http://127.0.0.1:51081'}`；修复前：`SINK_SELFCHECK_FAIL 绑定后自证失败：TimeoutError: timed out`（后台任务 exit code 1）；修复后：`SINK_LISTEN http://127.0.0.1:8791/token/<BROKER_SECRET>`（自证通过、服务在跑）；对照证据：`curl.exe -x 127.0.0.1:51081 https://www.example.com` → 200，同代理访问 `https://hid.ascend.huawei.com/online-develop` → `code=000` 25s 超时（**该代理只对公网有效，平台域名走不通**） |
| **176** | ★★ **"守护在跑，却问不到它的状态"—— 顺序错误把可观测性做没了；而为了查它写的自检，自己又造了三处假缺陷**。这一轮建"持久 SSH 会话守护"时连环踩到四件事：<br>① `remote_up.py --up` 报 `会话: **不在** [WinError 10061] 目标计算机积极拒绝`，但 `pid=36468(alive=True)` —— **进程明明活着**；<br>② 会话自检里"含引号/管道/括号的命令逐字节完好到达"**永远失败**，回显被截断成 40 字符；<br>③ 同一自检里"进程已退出"**偶发失败**（我固定 `sleep(1)`，而主循环 `accept()` 超时本身就是 1 秒）；<br>④ 经纪自检里 `cli_need` 链路**报超时**，而日志显示 token **已经送达临时 store**（`BROKER_TOKEN_OK how=selftest-page`） | ① **顺序错误**：守护原实现是"先 `connect()` 再 `bind()` 控制口"，而 `connect()` 在等经纪/页面签发时最长阻塞 45s ⇒ 那 45 秒里进程在跑但**控制口还没绑** ⇒ 上层只能看到"积极拒绝"。**"能观测"是自动化的前提**：进程活着但不可观测，所有编排都会退化成猜（与坑 174 同族：判据必须能回答"现在到底是谁/什么状态"）；② 截断那条是**我的假传输把回显截了** ⇒ 断言比较的是"截断后的回显 vs 完整命令"，**永远失败**，而失败与"base64 传输是否可靠"毫无关系 —— **测试自己制造的假缺陷**（坑 163/165 同族）；③ 固定 `sleep` 做时序断言 ⇒ flaky；**偶发红的断言比没有断言更坏**（会训练人忽略红）；④ 最阴的一处：我用 `globals()["CFG_PATH"] = …` 做自检隔离，但 `load_cfg()` **定义在另一个模块**（`token_sink`）里、解析的是**它自己模块**的 `CFG_PATH` ⇒ 补丁**打在错误的命名空间**，隔离根本没生效 ⇒ 自检盯着**真 store** 等一个永远不出现的 token，把**正确的实现**报成失败。四处里有两处（②④）是"实现对的、测试错的"，我若照着红字去改实现，就会把好好的代码改坏 | ① **控制口先绑、建连放后台**：`serve()` 改为先 `bind/listen` 并打印 `SESSION_LISTEN … 控制口已就绪，建连在后台进行`，再起 `connect_loop()`（唯一建连者，失败退避 10→20→40→60s 封顶，状态里暴露 `connect_attempts/next_retry_in`）；② 把"建连"职责**收敛到一条线程**：保活线程只管保活（`if self.cli is None: continue`），不再自己 `connect()` —— 原先两条线程都能建连，**状态归属不清**；③ 假传输回显**完整** base64（不截断），让"逐字节完好"这条断言真的在测 base64 通道；④ 退出改用**轮询等待**（最多 8s）而不是固定睡；⑤ 隔离改用**参数注入**（`cli_need(..., cfg=cfg)`）而不是改全局变量 —— 参数注入不容易骗自己；⑥ 定规矩：**测试红了先问"是对象有病，还是尺子有病"**，并优先检查"我改的东西到底作用在哪个对象上" | ① 修复前：`remote_up.py --up` → `会话 : **不在** [WinError 10061]`（同时 `pid_alive=True`）；修复后：`会话 : 在跑 / 已连接=False 建连=0 …` —— **控制口在首连之前就可达**，且 `SESSION_LISTEN … 控制口已就绪，建连在后台进行` 先于任何建连日志出现；② 修复前 `[FAIL] ★ 含引号/管道/括号的命令**逐字节完好**到达远端 回显='echo "a b" \| grep a; echo $(da'`（40 字符截断）；修复后 `[PASS] 回显='echo "a b" \| grep a; echo $(date +%s); (echo x)'` **逐字节相同**；③ `[FAIL] 进程已退出 poll=None` → 改为轮询后 `[PASS] poll=0`，且默认不再偶发；④ 修复前 `cli_need 链路返回 0` FAIL（`rc=1` 超时，而 `BROKER_TOKEN_OK how=selftest-page` **已在同一日志里**）→ 改为 cfg 注入后 `[PASS] cli_need 链路返回 0 rc=0`；⑤ 终态：`BROKER_SELFTEST cases=11 failed=0`、`SESSION_SELFTEST cases=9 failed=0`；⑥ 跨命令存活实测：`--up` 之后**另一次命令**里 `--status` 仍报 `broker=15220(alive=True) session=20376(alive=True)`，且会话日志持续输出 `SESSION_TOKEN_STALE … ⇒ 请经纪即时签发`、经纪日志出现 `BROKER_NEED seq=1（等页面来取）` ⇒ **自愈循环在无人干预下自行运转** |
| **177** | ★★ **"复用别人的函数"时我默认了"大家记录形状一样"—— 同一个假设连伤两次，把六组已经跑完的有效数据判成"崩溃/无效"**。`63_two_config_ab.py` 复用 `59_config_ab.py` 的统计函数，结果：<br>① 六组 A/B **全部 `valid=True steps=100`**，却在统计第一步 `KeyError: 'window_median_ms'`（59 的 run 记录写 `window_median_ms`，63 的写 `window_11_end_median_ms`）⇒ 整个 summary 没产出，只剩 `results.json` 原始记录；<br>② 修好字段名重算后，又报 **`INVALID —— 存在非单变量运行`**：`verdict_of()`（同样复用自 59）读每条记录的 `single_variable_ok`，而 **63 的记录里没有这个字段** ⇒ `None is not True` ⇒ 判无效 | ① 两处是同**一个**根因、不同的字段：**复用函数时默认了"调用方的记录形状与被复用方一致"** —— 而记录形状是**隐式契约**，没人写下来，函数也不校验（与坑 163"字段名是断言不是事实"、坑 176 同族，只是这次发生在**模块之间**）；② **杀伤面被放大**：这些函数正是"判定"本身，所以第一个字段名不匹配就**让 15 分钟机时白跑**，第二个字段名不匹配则**把有效数据判成无效**（假阴性 —— 与坑 35/159 的"假阳性"恰好相反，但同样危险）；③ 两次都**不该由我肉眼发现**：我做了 `--selftest 29/29` 与 `--selftest-orchestration 6/6`，但两者都只喂**59 形状**的记录 ⇒ **自检与实跑用了同一份错误假设**（坑 163 的原话）；④ 还有一层：第一次失败时我用"整份 base64 一次性推文件"去修，撞上 Windows 命令行 32KB 上限（`The filename or extension is too long`）**静默失败**，远端仍是旧版 ⇒ 重算又撞回同一个 KeyError，**错误现场与真因隔了两层** | ① `59.paired_stats` 抽出 `win_ms_of(rec)`：**两个字段名都认**（`window_11_end_median_ms` 优先），**都不在就 `raise`**（绝不返回 None/0 —— 那会让下游算出假的性能差）；② 自检补 **2 条**：用 **63 形状**的记录喂进 `paired_stats`（正例）、两种字段名都缺必须报错（反例）⇒ `P59_SELFTEST_OK cases=31 failed=0`；③ `63` 的记录**补齐 `single_variable_ok`**，并在注释里写明它在本工具语义下为真的依据（`--declare` 与实际差异在开跑前已逐键校验，不通过则 rc=3 根本跑不到这里）；④ 为重算已跑完的数据，新增 `ab_recompute.py`：**从落盘产物重新推导**该判据（`diff_keys(两份配置)` 必须等于从 `cmd_ab1.txt` 解析出的 `--declare`），拿不到就不置位、让 verdict 如实判 INVALID —— **不为让结论通过而设标志**；⑤ 会话工具补 `--put`（**分块**上传 + 两端 SHA256 对照），并在失败时**打印原文**（第一版只打 rc，`rc=None` 时完全没有诊断信息） | ① 现场：`KeyError: 'window_median_ms'`（`63:324 → 59:263`）与 `AR_VERDICT INVALID —— 存在非单变量运行`；② 修复后：`AR_SINGLEVAR_RESULT [...6 键...] —— ok（重新推导）`、`AR_PAIRED pairs=3 mean_delta_pct=-19.072`、`AR_VERDICT WIN —— 改善 19.07%（95% 区间 [7.25, 30.89]）区间下界 ≥ 5%`；③ 自检：`P59_SELFTEST_OK cases=31 failed=0`（含 `63 形状记录（window_11_end_median_ms）能被成对统计` 与 `两种字段名都缺 → 必须报错（不得静默当 0）`）；④ 上传：`PUT_OK 59_config_ab.py 58021 B / 6 块 sha256=4e0b79ff97c5`；⑤ 对照：整份推送时 `Program 'python.exe' failed to run: The filename or extension is too long`（**静默失败**，远端文件未更新，于是重算复现同一 KeyError） |
| **178** | ★★ **协议层假设"命令输出一定以换行结尾"—— 于是尾行与数据粘连，解析永不匹配；而我的错误信息又不打印原文，让我在同一处绕了两轮**。会话工具取远端文件（`base64 -w0 <file>`）时，客户端打印 `SESSION_RC=0 seconds=0.53`，但**`base64 -w0` 的输出不带结尾换行** ⇒ 实测尾行是 `…fQpd==SESSION_RC=0 seconds=0.64`（**粘在同一行**）。我的解析写成 `^SESSION_RC=(-?\d+)\s*$`（要求数字后直接到行尾），既要求了**行首**又要求了**行尾** ⇒ 永不匹配 ⇒ `--get` 全部失败（报 `rc=None`） | ① **我把"输出一定以换行结尾"当成了协议事实**，而它只是**交互式终端的习惯**：管道/重定向下的命令输出完全可以不以换行结尾（`base64 -w0`、`printf`、`head -c` 都是）；协议要**自己去保证行整洁**，不能依赖被调命令的收尾风格（与坑 163/176/177 同族：**没核对过的假设**）；② 更贵的是**诊断被我自己吞掉**：第一版失败只打印 `rc=None`，不打印会话返回的原文 —— 而原文里那行 `…==SESSION_RC=0` 一眼就能看出粘连。补上"打印原文"后**一轮就定位**；③ 还有一次正则自伤：我先写成 `\s`（对）又改成 `\s*$`（错），**凭"看起来更严谨"改判据**，而没拿真实输出核对 —— 判据的措辞必须由真实样本决定 | ① **根因修在协议侧**：`ssh_session.client()` 在写 stdout 后**补一个换行**（`if _o and not _o.endswith("\n")`），让"数据行/尾行"永远分离；② 消费侧也不依赖行首：`pull_via_session` 取**最后一次**出现的 `SESSION_RC=(\d+)`；③ 失败路径**必须打印原文**：`SESSION_CALL_ANOMALY child_rc=… stdout_len=… stderr=… stdout_tail=…`（把"我的错误信息别吞诊断"变成默认行为）；④ `--get` 增补两端 SHA256 对照（不一致即拒落盘）—— 传输静默出错比传输失败更坏；⑤ 上传侧补 `--put`：**分块**（每块 ~14KB）+ 两端 SHA256（整份推送会撞 Windows 命令行 32KB 上限并静默失败，见坑 177） | 修复前：6 次 `--get` 全报 `PULL_FAIL base64 失败（rc=None）`，且**看不到原因**；加诊断后立刻得到 `stdout_tail='…I3MzUwZTc5NzFiYzFiNTljZjI2NCIs…fQpd==SESSION_RC=0 seconds=0.64\n'`（**粘连可见**）；修复后 6 个文件全部 `PULL_OK`，例：`ab_switches/summary_recomputed.json → 29817 B sha256=a027d6fde5cc`、`ab_switches/results.json → 25495 B sha256=d86ae66eae4e`、`ab_skipgdn/summary.json → 9688 B sha256=628ab5332db9`、`A_official_switches.yaml → 3911 B sha256=aaa0aeedb229`、`base_ab_option1.yaml → 3915 B sha256=4e5f978599da`（**与链路日志里记录的 `AM_BASE_AB_SHA256=4e5f978599da93a7` 前 12 位一致** ⇒ 本地这份确实就是 A/B 当时用的那份配置） |
| **179** | ★★★ **"清元数据"与"卸载包"不是一回事 —— 照 `pip uninstall` 的建议去修，会把整个 triton 栈删掉**；而**两份 RECORD 用了两种 base64 编码**，让"明明匹配"的文件被我判成"对不上"。现场四件事：① 环境自检报 `triton_version_consistent=False`（上游 `triton 3.5.0` 与 `triton_ascend 3.2.2` 混装），**我自己在 `versions.lock` 里写的处置就是"应卸载上游 triton"**；② 动手前只读盘点发现：磁盘上**只有一个** `triton/` 目录，其 243 个文件由 `triton_ascend-3.2.2.dist-info/RECORD` 拥有 ⇒ 上游那份其实**只剩元数据**；③ 取证：上游 `triton-3.5.0.dist-info/RECORD` **488 行里 479 行是 `triton/…` 路径**（含 **412 MB 的 `triton/_C/libtriton.so`**）⇒ **照它删就是删 triton-ascend 的模块文件**；④ 归属脚本 `triton_whose.py` 第一版把 `triton/__init__.py` 与 `libtriton.so` 报成"**两份 RECORD 都对不上**"，而它们的磁盘哈希与 triton-ascend 记录的**前 12 位完全相同** | ① **"同名"有两种关系被我混为一谈**：*文件属于谁* 与 *元数据声称谁* 可以完全不一致 —— `pip uninstall` 删的是 **RECORD 列表里的路径**，而 RECORD 是**安装当时**写下的，被后来的覆盖安装污染后，"照记录删"就会删到**别人的文件**；我此前把它当成显然的修法，说明我在**没做只读盘点前就给建议**（与坑 173 相反：先实测被怀疑的对象，再改它）；② 编码口径：pip 的 `RECORD` 用**无填充 urlsafe base64**（`-`/`_`），而我算的是**标准带填充**（`+`/`/`、带 `=`）⇒ 同一摘要两种编码，`==` 必然假（坑 176/177/178 同族：**判据的口径本身也要先对样本**）；③ 最危险的是①的方向性：**它是一条"照着做就把环境毁掉"的建议，而且是我自己写进交付物的** | ① 修法改为**只清元数据**：删 `triton-3.5.0.dist-info`，**绝不触碰模块文件**，并加三道硬闸门 —— 删除清单里不得出现任何 `triton/…` 模块路径（出现即 abort）、动手前先打回滚点 tar.gz、动手前后对 `triton/__init__.py` 与 `triton/_C/libtriton.so` 取 sha256 **必须逐一相等**；② 归属判定改为**按 RECORD 里记的 sha256 逐条对照磁盘文件**（不看版本号、不看目录名），并**归一化 base64**（urlsafe→standard、去 `=` 填充）后再比；③ 结论措辞收紧：核心文件确属 `triton_ascend`，但 `libproton.so`（18.9 MB、训练不用）**与两份 RECORD 都不逐字节相符** ⇒ 标注**"来源未判定"**，**不声称"完全对齐"**；④ `versions.lock` 里那条**错建议**已改为实测记录并写明「**禁止** `pip uninstall triton`」；⑤ 修复后跑**完整 100 步**验收（本想跑 6 步，`50_train.py --steps` 不覆盖配置里的 `train_iters`，反而跑成完整验收） | ① `triton_state.py`：`pip 条目=['triton==3.5.0','triton_ascend==3.2.2']`、`import triton=3.2.0`、`triton-ascend 自带 triton/ 包（243 文件）`、site-packages 只有**一个** `triton/` 目录；② 上游 RECORD：`wc -l=488`、`grep -c ^triton/ = 479`，样例 `triton/_C/libtriton.so,sha256=4kaJQ0mPP3RK…`（412 MB）；③ 修复：`TRITON_FIX_ROLLBACK …tar.gz (20317 B)`、`hash_before/after 相同`、`pip_after=['triton_ascend==3.2.2']`、`TRITON_FIX_OK`；④ 归属：`__init__.py ⇒ triton_ascend`、`libtriton.so ⇒ triton_ascend`、`libproton.so ⇒ 与两份 RECORD 都不相符`（`TRITON_WHOSE MIXED`）；⑤ 验收：`iteration 100/100 … 416.9 ms`、`HuggingFace checkpoint saved … successfully!`，步时长 417–431 ms **与 A/B 的 A 臂 418.6/418.0/420.5 ms 独立吻合** |
| **180** | ★★★ **我把自己交付工具链的整个目录删掉了 —— 起因是跑了一个"自己从未读过内容"的重建式脚本**。用户要求"记好后让 Codex 检查"，我去跑 `build_codex_handover.py` 生成复核包；它报 `FileExistsError: … 分析脚本` 中途失败；事后发现 **`04_核心资产\交付物生成脚本\` 整个目录（约 40 个交付脚本：`prepush_check.py`(24 闸门)、`align_audit.py`、`_append_pitfall.py`、`build_handover_skill_zip.py`、`sync_to_delivery_folder.py`、`_diag_copies.py`、`make_profile_templates.py` 等）被删除**，`05_交付物成品`、`10_交给Codex` 同样不见；且它**用交付目录的旧副本覆盖回了活副本**，把坑表里刚追加的 **179 行回退掉**（活副本 `整合版` 与交付 `06_Skill` 时间戳同为 23:52:52）。全盘按名搜索（Desktop 递归 + TEMP）**全部 MISSING**；无 git；回收站也没有（`rmtree` 不进回收站） | ① **该脚本是"先清理、后重建"式且非原子** —— 中途失败就留下"删了但没建回来"的半状态（这正是本项目一直禁止的写法，而它就在我自己的工具目录里，我却没审过）；② **更根本的是我的动作依据**：我运行它是因为"它属于交付流程"，而**不是**因为"我确认过它会删什么" —— 一个会 `rmtree` 的脚本，我在**从未读过其内容**的情况下就执行了（坑 173/179 同族：**先确认对象，再动它**；这次我连"读"都没做）；③ **工具目录没有任何备份或版本控制**：交付物有 `_旧版本备份_20260917`（4642 文件），而**工具链自己**是单点 —— 备份策略只覆盖"要交的东西"，没覆盖"造东西的东西"；④ **"从交付副本反向覆盖活副本"这一手最具欺骗性**：它让"活副本"看起来被同步得很好（哈希全一致），代价是把**未进交付副本的新增**（179 行）静默丢弃 ⇒ **单向同步一旦被反向执行，就变成回退**；⑤ 讽刺的是它恰好是"交给 Codex"这一步的脚本 ⇒ **越靠近交付的自动化，越需要闸门**，而这里恰恰是零闸门 | ① **提交物先行核对**（结果：`C4AI复赛_交付材料` 各目录完整 —— 06_Skill 108 文件含 `qwen35-ascend-migrator_20260921_v1.0.zip` 560,355 B、07_源代码 80、05_原始日志 13、03_精度分析报告 6 ⇒ **提交物未受损**）；② **逐项核对活副本**（结果：两份新模板 + 决策记录 + `59 win_ms_of` 修复 + `versions.lock` 新版 triton 条目**均在**，活副本与交付解包副本三处 md5 相同 ⇒ **唯一回退是坑表 179 行**，已补回）；③ **冻结**：不再运行任何未读过的"重建/清理"式脚本；④ 定规矩（对今后任何会删东西的工具）：**(a) 默认 dry-run 并打印"将删除清单"**，**(b) 只允许删除自己输出目录白名单内的路径（前缀断言，越界即 abort）**，**(c) 失败必须回滚**，**(d) 运行前先打备份点**；⑤ **把工具目录纳入备份**（时间戳 tar），不再只备份交付物；⑥ **禁止"从交付副本反向覆盖活副本"**：同步只允许单向（活副本→交付副本），且必须先做差异清单；⑦ 可恢复性：只有**内容在本轮会话里完整出现过**的文件能逐字重建（`_append_pitfall.py` 我读过全文、`make_profile_templates.py`/`triton_*.py` 等是我写的），其余**无法逐字恢复** ⇒ 需要时按"最小必要 + 上述闸门"重写，不凭记忆造 | 时间线：上一条命令里 `_append_pitfall.py` 仍正常执行（`PITFALL_APPEND_OK 新增=179 条数=179`）⇒ 目录当时存在；随后唯一运行的就是 `build_codex_handover.py`（其 traceback 自证路径为 `…\交付物生成脚本\build_codex_handover.py`）；复查 `Test-Path …\交付物生成脚本` = **False**；`04_核心资产` 下只剩 `qwen35-ascend-migrator_整合版`(107 文件)/`分析脚本`/`skill_A3_bringup`/`skill_R2-SK04`/`训练配置`/`证据包`；7 个关键脚本名全盘搜索 **MISSING**，TEMP 亦无；坑表最大编号复查从 179 **退回 178**，补回后 180 |
| **181** | ★★ **我拿"工具退出码"当判据，把一条好闸门判成永远红 —— 而"永远红"最后会训练人忽略红**。重建推送前驱动器（坑 180 之后）后首跑 5 条 FAIL，逐条查下来 **4 条是我自己的判据错**：① `G23 未定义名扫描` 报 FAIL，因为我对 `pyflakes` **用退出码判** —— 它因 `unused import` 返回 1，而这条闸门要抓的是 **`undefined name`**（坑 147/153 的 `steps_complete`/`pip_install`）。实测全量输出：`unused_import=11`、`其他=7`、**`undefined name=0`** ⇒ **产品没病、闸门本该是绿的**；② `G22 三方一致性` FAIL，因为我把 `safe_pack_sync.py` 按**默认 dry-run** 调（没传 `--apply`），自然拿不到 `SAFE_PACK_SYNC_OK`；③ `G2` rc=0、输出 `GATES_OK` 判 FAIL，因为我的"坏标记"清单里有 `"FAIL"` 这个**子串**，而它的说明文字里恰好含该字样；④ `G13` rc=0、`TRITON_VERDICT_SELFTEST_OK cases=6` 判 FAIL，只因为我的"好标记"清单里没收这个词 | ① **判据的三层错位**：*退出码* ≠ *这一条要检查的性质* ≠ *输出里恰好出现的字符串*。我用最省事的那层（退出码/子串）代替了"这条闸门到底要判什么"（与坑 176/177/178 同族：**判据措辞/口径没拿真实样本核对**）；② **"永远红"的危害有方向性**：一条永远红的闸门不会让人去修它，只会让人**学会跳过它** —— 于是它保护的东西反而更危险（与坑 35/159/174 的"假阳性/假阴性"是同一族，只是这次是**长期噪声**形态）；③ 还有一处**凭记忆猜参数**：G1 我写了 `90_selfcheck_gates.py --compile-smoke`，而那个参数**是我编的**（旧驱动器的编译冒烟是内联实现，不是该脚本的开关）—— 与坑 34/41/164 同族：**猜接口**；④ 值得注意的是**这四条全部发生在"重建"场景**：我不是在改功能，而是在**复现一个已经死掉的东西的行为**，于是"看起来对"比平时更容易蒙混过关 | ① **判据改为"每条闸门各声明自己的期望标记"**（`GATES` 里新增第 5 列），标记缺失时报 **`标记不匹配`** 并打印期望值，而不是泛泛 FAIL；② **按类别判、不按退出码**：G23 只在输出里出现 `undefined name` / `unable to detect undefined` 时失败，其余类别（unused import 等）**明确打印为"不计入失败"**；③ **该真做就真做**：G22 传 `--apply`（推送前闸门本就该执行而非预演）；④ 删掉 `"FAIL"` 这种**子串式坏标记**（它会被说明文字误伤），改为逐条期望标记 + `Traceback` + 退出码三重判据；⑤ **G1 改为驱动器内自实现**（编译全部 `.py` + 逐个 `--help` 冒烟，判 NameError / 缺环境依赖 / 其他三类），并**显式 `cwd=SKILL`**（坑 173）；⑥ 复跑结果：**现存 14 条 PASS=14 FAIL=0**，总判决仍 `PREPUSH_FAIL` —— **只因为 9 条已显式列出的丢失检查**（"不漏项"要的就是这个状态：红得诚实，不静默省略） | 首跑：`合计：现存 14 条（PASS=9 FAIL=5）+ 丢失 9 条`；pyflakes 全量：`待扫 .py 文件数=36`、`undefined_name 计数=0`、分项 `unused_import 11 / other 7`；G2 输出 `GATES_OK （1 项因平台限制未验证…）` 而 rc=0；G13 输出 `TRITON_VERDICT_SELFTEST_OK cases=6` 而 rc=0；G22 输出 `DRY-RUN 结束（未改动任何文件）；加 --apply 执行`；修后：`PASS G1 判据=G1_OK`（`36 个 .py 编译；冒烟失败 0、跳过 3`）、`PASS G13 判据=PROFILE_SELFTEST_OK`、`PASS G22 判据=SAFE_PACK_SYNC_OK`、`PASS G23 判据=G23_OK undefined=0`（`扫描 36 个 .py；其他类别=19 不计入失败`），`合计：现存 14 条（PASS=14 FAIL=0）+ 丢失 9 条` |
| **182** | ★★★ **我"修 traceback"时把打包器改成了"没有回滚点也照删" —— 我自己引入的 P0**。`safe_pack_sync.py` 首次 `--apply` 因备份目录同名碰撞抛 `FileExistsError`；我把 `backup_dir` 改成**失败只打印并返回 None**，却**没改调用方**：`unpack_replace` 拿到 None 后**照样 `shutil.rmtree(target_dir)`**，源码注释甚至写着「删除流程继续」⇒ **没有回滚能力的不可逆删除**（解压或校验再失败，旧目录永久丢失，正是坑 180 的形态）。外部复核用替身函数在内存里复现：**备份失败后删除操作仍被调用** | ① **我只修了报错点，没修"报错意味着什么"**：把"抛异常"改成"返回 None"，等于把失败降级成**一个值**，而调用方是否检查它，我没看（坑 147/153 同族：**修一条路径不问对称的那条**）；② 更深一层：**"备份失败"= 失去回滚能力，而失去回滚能力就不得执行不可逆操作** —— 这个推理我没做，反而把"继续"当默认；③ 我还在注释里把它**写成正常行为**（"删除流程继续…请知悉"）⇒ **注释替缺陷开脱**，比缺陷更坏（下一个人会以为是设计）；④ 与坑 180 是**同一错误的两次出现**：上次是"跑没读过的 rmtree 脚本"，这次是"自己写的 rmtree 没有回滚点" | ① 判据改成**可执行的不变量**：**任何一步失败，旧工件必须逐字节不变**；② `bk is None` 且目标存在 ⇒ **立即中止并返回 False**；③ 删掉自我开脱的注释，改为「**调用方将因此中止，不再删除**」；④ 新增 `failure_injection.py`：注入**备份失败 / 解压失败 / 校验失败**三类故障，逐例断言旧目录哈希不变，并带**正对照**（证明那些拒绝不是"它本来就不工作"）| 现场：`FileExistsError`（`%H%M%S` 秒级命名无唯一后缀）→ 唯一序号后缀；修复后 `FAILURE_INJECTION_OK cases=7 failed=0`（①`中止：备份失败而目标已存在 ⇒ 拒绝删除`+旧目录逐字节不变 ✓；②解压失败 ⇒ 备份点回滚 + 哈希不变 ✓；③校验失败 ⇒ 同上 ✓；正对照 rc=True ✓）；测试全在沙箱，未触碰真实交付物 |
| **183** | ★★★ **闸门在"工具根本没运行"时依然报 OK —— 我用退出码/子串当判据，把 fail-closed 写成了 fail-open**。外部复核用**纯内存坏例**证明两处：① `G23`（pyflakes 未定义名扫描）只 grep 输出里的 `undefined name` 字面量，**完全不看 rc 与 stderr** ⇒ pyflakes **根本没装**（`No module named pyflakes`）时 `undef=[]` ⇒ 报 **`G23_OK`**；② `G5` 的 `_SELFTEST_OK` 分支写成 `passed = bool(mm) and "Traceback" not in out`，而 `mm` 来自 `re.search(r"[A-Z0-9_]+_OK")` ⇒ **rc=1、先打印 `A_CASE_OK`、后打印 `VERIFY_FAIL case_ok=12/13` 也判 PASS**。⇒ 于是"18/18 全通过"里至少两条是**误放行** | ① 我把判据写成"**输出里有没有某个字面量**"，而它要判的是三层不同的东西：**工具是否正常运行 / 是否覆盖目标对象 / 目标是否满足要求**；前两层失败时第三层无从谈起，我的实现却让三者**退化成同一件事**（坑 176/177/178/181 同族：**判据口径没对样本**）；② `*_OK` 这类**宽松子串**是天然的假阳性来源：输出里更早出现的任何片段都能"救回"整体失败；③ 最严重的不是"该看 rc 还是文本"，而是**我允许一个检查在无法运行时返回"通过"** —— 直接违反我自己写下的 fail-closed；④ 我此前刚把"永远红"当反面教材，结果走到了另一个极端（**宁可绿灯**）| ① 抽出**三层状态**：工具缺失/无法启动/零样本/解析失败一律 **`ERROR`**，绝不 PASS；`rc≠0` 却出现期望标记 ⇒ **判 FAIL**（局部用例过 ≠ 整体通过）；② 每条闸门**各声明精确期望标记**（`FINGERPRINT_OK`/`PROFILE_SELFTEST_OK`/`SAFE_PACK_VERIFY_OK`…），取消一切模糊 `*_OK` 匹配；③ 新增 `gate_selftest.py` 固化坏例：pyflakes 无法启动 ⇒ ERROR、零样本 ⇒ ERROR、rc=1+标记 ⇒ FAIL、后端缺失 ⇒ ERROR，并加**正对照**；④ G22 从 `--apply` 改为**只读** `--verify-only`（检查命令不得写文件）| `GATE_SELFTEST_OK cases=5 failed=0`（`ERROR(工具缺失)`/`ERROR(零样本)`/`run_gate=False`/正对照 `run_gate=True`）；G22 只读模式上线**当天抓到真漂移**：我改了 `versions.lock` 却未重打包 ⇒ `SAFE_PACK_VERIFY_FAIL 活副本与 zip 不一致（缺 0 / 不同 1）`，重打包后 `SAFE_PACK_VERIFY_OK items=108` |
| **184** | ★★ **我把工具输出的 `B(first)` 当成"均值"写进了对外材料**。外部复核直接指出：② 的 `false` 臂应是 **562.55 ms**，而我写的是 **557.0**。查产物：`ab_skipgdn` 两次运行为 **556.95 / 568.15**，均值 562.55；557.0 是工具 `P59_RESULT` 行里的 `B(first)=557.0`（**第一次运行的窗口中位数**）| ① 我把"一行输出里最好抄的那个数"当成了"这个臂的统计量" —— 两者不是同一个东西，而工具**已经把区分写在字段名里**（`B(first)`），我却按习惯读成了"B 的值"（坑 163/176 同族：**字段名是断言不是事实**）；② 危害方向明确：我取了**偏快**的那次作代表值，会让②的"false 更慢"看起来更小 ⇒ 这是**朝有利方向**的读数错误，最不容易被自己质疑；③ 更值得记的是：这份材料是**给外部复核的**，等于把未核对的口径直接交给别人去发现 | ① 对外材料里每个数字必须标注**它是什么统计量**（单次/均值/中位数）与**来自哪个字段**；② 只用聚合量，不用"某一次"；③ 把可疑的口径差异主动写进复核清单（这次正是复核抓到的）| 产物 `aftermab_20260921_143648/ab_skipgdn/summary.json`：A=true `516.5 / 533.3`、B=false `556.95 / 568.15` ⇒ 均值 **524.90 / 562.55**；决策记录已另存 v2 并写明「v1 误把 `B(first)=557.0` 当均值，已更正」|
| **185** | ★★ **同一轮里我两次踩了自己早就写下来过的限制：把键路径猜错、把等待写成超过会话守护的单命令上限**。① P2 臂要改确定性开关，我在脚本里把键写成 `tools.use_deter_comp`，而模板里真实路径是 **`training.use_deter_comp`**（`config/templates/qwen3_5_0_8B_recommended_A.yaml:132`，上一行的 `tools:` 是另一个段）⇒ 派生时报 `P1B_DERIVE_MISSING`，**P2 整臂被 SKIP**；② 我要等 P2 两轮跑完，写了个最多 420 秒的轮询等待，而**会话守护的单条 RUN 上限约 110 秒**（`ssh_session.py --cmd-timeout`），于是收到 `REMOTE_RC=-1`，等待被截断 | ① **"拒绝自造键"的纪律把猜测变成了可见的 SKIP**——这一点是好的（否则会假装跑过）；但它**只让错误可见，不能阻止我写出猜测的路径**：我是在**没有先读模板**的情况下写出 `tools.` 的，而外部复核**读文件**一眼就发现了（而且他同时指出：即便键对了，也必须取证"键是否真被消费"）；② 等待上限那条更刺眼：**那是我自己定的值**，写在 `ssh_session.py` 里，我在同一轮里既定规矩又违反它——属于"知道规则但没在写代码时调用该知识"；③ 两处的共同形态是**能力/约束的知识存在，但没有变成写代码时的检查**（与坑 180"跑没读过的脚本"、坑 183"闸门失效仍报 OK"同族，只是换成了"自己的约束自己没遵守"） | ① P2 脚本改用真实路径并把**回读断言**写进派生步骤（`assert b["training"]["use_deter_comp"] is True`），实测 `False→True` 通过；② 按复核要求补**生效取证**：从运行日志内的配置 dump 取 `use_deter_comp` 实际值（两轮均 `True`），即"键存在 ≠ 功能生效"必须另证；③ 等待改为**有界 ≤95 秒**（严守守护上限），需要更久就分多次调用；④ 更一般的处置：**任何"我要改某个键"的脚本，必须先 `grep` 模板确认路径存在**，而不是凭记忆写路径——这一条我已写进脚本注释 | 现场：`P1B_DERIVE_MISSING tools.use_deter_comp（键不存在，拒绝自造）` → `P1B_SKIPPED P2`；等待：`REMOTE_RC=-1`（420s 轮询撞守护 110s 上限）；本地取证：模板 `line 132: use_deter_comp: false`（在 `training:` 段下，`tools:` 在 line 138）；修复后：`P2_DERIVE_OK … False→True train_iters=100 GBS=8`、`P2_EFFECTIVE run1/run2: use_deter_comp: True`、两轮 `rc=0`、`P2_DONE 17:14:43` |
| **186** | ★★ **"机器上的 Skill"与"交付物里的 Skill"不一致，而两边各自都通过了各自的检查**。补代码/数据/权重指纹时，`sha256sum config/templates/qwen3_5_0_8B_recommended_A.yaml` 报 **No such file or directory** —— 因为我当初只把这两份模板推成了 `/root/ops/A_recommended.yaml`（**实验用的名字**），**从未推回 Skill 的 `config/templates/`**（**交付用的名字**）。于是机器上"跑过的配置"与交付物里"要交给评委的配置"**是两个路径下的两份东西**，而本地的 `PREPUSH_OK`、`SAFE_PACK_VERIFY_OK`、`ALIGN_AUDIT_OK` **全都只检查本地**，一条都没覆盖"远端 Skill 是否等于交付物" | ① 我把"推文件"理解成了"让实验能跑"，而没意识到**实验对象与交付对象必须同一份**——这与本项目的验收口径（"评委用**选手的配置+环境**复现"）直接冲突：评委拿到交付物，而我量的是另一个路径下的文件；② 两类检查各自自洽、合起来漏检（与坑 112/163/166 同族：**多源真相**；只是这次第二个源在**远端机器**上）；③ 更隐蔽的是**命名差异**掩盖了问题：`/root/ops/A_recommended.yaml` 与 `config/templates/qwen3_5_0_8B_recommended_A.yaml` 内容相同（sha256 都是 `3f7bb2ae9199` ✓），所以"看起来一直在用同一份"——**内容相同不代表身份相同**，一旦哪天只改一处就会静默分叉 | ① 立即把 4 个文件推回远端 Skill（两份模板 + 决策记录 v2 + 交付前人工清单），全部 `PUT_OK` 带两端 SHA256 对照；② 生成 `FINGERPRINTS.txt`（代码/配置/文档/数据/env 共 11 项哈希）并拉回本地，作为"机器侧所见"的存档；③ 定规矩：**凡是"交付物里的文件"，必须在机器上以交付路径存在**；实验用的别名（`/root/ops/*.yaml`）只作副本，且要记录它对应交付物的哪个哈希（本次两者 sha256 相同，已核）；④ 待补：把这条做成自动闸门（远端 Skill 指纹 × 交付物哈希对照），而不是靠人记得 | 现场：`sha256sum: config/templates/qwen3_5_0_8B_recommended_A.yaml: No such file or directory`（同时 `FINGERPRINTS.txt` 只生成 7 行而非 9 行）；修复后：`PUT_OK ×4`（`3f7bb2ae9199` / `44a2f909f018` / `3736cc7dc9e6` / `735bb603144a`）、`FINGERPRINTS.txt` **11 行**（含两份模板、决策记录 v2、人工清单、mock 数据 `247e7bb48507`、`env.json` `ebc65b923b19`），且与本地同名文件哈希**逐一一致** |
| **187** | ★★ **我把"当时的非证据"清掉了，而它正是我后来方法的唯一正对照**。跑阶段 1b 时，我在脚本里写下"清 checkpoint（不是本实验的证据）"并真的 `rm -rf` 了 P1/P2 的 checkpoint（每份 3.2 GB）。**几十分钟后**我建立了比对落盘权重的方法（不需要框架内部接口，是回答"第一处不同"最便宜的客观量），而这条方法**恰好需要那些 checkpoint**：`deter=true` 那两轮是"确定性路径下权重应逐位相同"的**正对照**。结果只能做到"同配置两次不同（320/473 张量）"，**给不出正对照** ⇒ 缺一个"该方法是够灵敏、且这个量本来可以相同"的基准 | ① **"这是不是证据"取决于后来用什么方法判读**，而我在**还不知道要问什么**的时候就定了性——"不是本实验的证据"这句话当时就**没有依据**（我只是想省磁盘：两个 checkpoint 6.4 GB）；② 更一般的形态：**清理动作是不可逆的，而清理判据是当下的、可变的**（与坑 180"跑没读过的脚本"同族：都是"不可逆操作 + 判据不足"）；③ 而且我**在协议里正是用这批 checkpoint 做过的另一件事**（P2 的生效取证是读 `train.log`，但复盘需要权重）——同一批产物有两个用途，我只按当时想到的那个判了 | ① 记下这条并**明确标注"缺正对照"**（已写进 ckpt 比对证据文件的"边界"栏，不假装该方法已经完整）；② 定规矩：**清理任何运行产物前先问"未来判读可能需要它吗"**，答不上来就保留（磁盘 300 GB，当前仅用 31% ⇒ 没有清理的紧迫性）；③ 需要正对照时**重跑 2 次确定性配置并保留 checkpoint**（成本约 3 分钟机时 + 6.4 GB，已列入待办） | 现场：阶段 1b 脚本内 `rm -rf "$ROOT/cfg_save"` 每轮执行；P2 脚本内同样 `rm -rf "$ROOT/cfg_save"`（`P2_DONE` 后无 checkpoint 残留）；随后 `compare_ckpt.py` 只能比对 `AB_AVSB_20260921/{00_A,03_A}`（该目录的 checkpoint 未被清理 ✓ 每份 3.2 GB），得到 `逐位相同=153 不同=320 含非有限=0`，**无法给出 deter=true 的正对照** |
| **188** | ★★ **同一个交付包里，"报告的口径"和"判定链的口径"互相矛盾了整整一天，而没有任何一个自动检查会因此报警**。`04_性能测试报告\性能优化报告.md` 原文写着"50–100 步性能优于官方基线（中位 419.0 ms vs 官方 **431.3 ms**）、快 **2.8%**""419.0 ms 与官方 431.3 ms 的**可比性成立**""相对官方 431.3 ms 是 **1.51×**"；而同一批产物的判定链里 `official_comparable = **false**`（数据是我构造的 mock：1 图/样本、cutoff_len=1024、512 长 mock 样本，**非官方 COCO**）。两句话并存数天，`prepush_check.py` 19 个自动闸门**全部 PASS** | ① 我把"报告"当**文档**、把"判定"当**数据**，**从未定义**"报告里的每一条性能断言必须能追溯到一条判定记录"⇒ 报告中位数与官方基线做比值这件事，在判定链里是**明确禁止**的（`official_comparable=false`），却没有任何工具去比对两者；② 更根本：**"和官方比"是我自己想要的叙事**，于是我在报告里用了官方窗口的中位数 431.3 ms 做分母——而官方口径（101–200 步 / `train_iters=100` / 附件5 50–100 步三种）我**当时并未确认是哪一种**，等于用一个**语义未定的外部数字**给自己的结果加了个"优于"的标签；③ 闸门只检查"格式/存在性/一致性"，**不检查"结论方向是否被前提支持"**——这类检查天然难自动化，但也正因如此**必须落到人工清单里**（我却没写进去） | ① **保留原文一字未删**，在报告顶部追加 `⚠ 更正说明（2026-09-22）` 表格：逐条列出"原文表述 → 现在必须如何读"，明确 `official_comparable=false`、比值**收回**、1.51× **不作为结论**，并写明"仍然成立的部分"（同机同数据的优化链路）；② 另写 `04_性能测试报告\口径更正说明.md` 独立存档；③ 口径分层落库为三个独立结论：`plan_consistent` / `ab_comparable` / `official_comparable`，**任何对外表述必须指明是哪一层**；④ 列入人工清单：**报告中每次出现"vs 官方/优于基线/倍率"必须同时能看到 mock 声明**（待做成 G 检查） | 报告原文 L244/L325（`性能优化报告.md`，21275 B）；判定链 `protocols/numeric_verification_100step.json` 的 `official_comparable=false`；更正后 `性能测试报告.pdf` 684922 B，抽取正文**开头**即「⚠ 更正说明（2026-09-22 追加，原文一字未删）……official_comparable = false … 收回 … 不作为结论」 |
| **189** | ★★ **我"改了文件"，但"交付物没变"——因为重印 PDF 时我传给转换器的是"插入横幅**之前**"读到的那份字符串**；而我第一次验证**已经看到失败信号，却用另一个数字给自己找了台阶**。脚本里 `s = open(md).read()` 在**插入横幅前**执行，插入横幅写回磁盘后，我仍把旧的 `s` 交给 markdown→HTML→Chrome。结果：**md 有横幅 ✓、PDF 没有 ✗**，而我又把这个 PDF **改名成规范名放进了交付目录**（等于用旧口径内容覆盖了交付物，只更新了时间戳）。验证时我抽 PDF 文本，`更正说明`/`不可比`/`mock` **全为 False**（=没生效）——我看到的就是失败信号，但当时用"字符数从 10460 涨到 11576"说服自己"可能只是抽取匹配问题"，**没有立刻重印** | ① **变量/对象拿错**（与今天三次"嵌套引号写坏脚本"、两次"按行编辑多行结构"同族）：**改的是磁盘对象，印的是内存里的旧副本**；② 更深一层：**"写文件成功"与"交付物正确"之间没有任何绑定**——我的流程里没有一步要求"从交付目录**重新读回**被修改的文件再验证"，验证用的还是我自己以为写下去的内容；③ **验证判据选错**：用"子串命中"这种**存在性判定**去验证一个**中文 PDF**（Chrome 抽取可能切字/改动空格），一旦 False 就产生了"判据可疑"的模糊地带，正好容我自我安慰；正确做法是**直接打印被验证对象的正文**（开头 400 字），一眼可判、无可辩解 | ① 重写为 `_regen_perf_pdf.py`：**从磁盘重读 md**（确保带横幅）→ 印临时 PDF → **打印抽取正文开头 400 字** → 不含更正痕迹就 **fail-closed 不覆盖交付**（本次：含痕迹 ✓ ⇒ 684922 B 覆盖成功）；② 立规矩：**任何"重新生成的交付物"必须从交付目录读回原件做验证**，不得复用内存中的副本；③ 立规矩：**验证要打印正文，不要只判子串命中**（存在性 False 不等于内容缺失，也不等于内容存在）；④ 已实测：新 PDF 抽取开头即「⚠ 更正说明（2026-09-22 追加，原文一字未删）……419.0 ms、51.1×……是在 mock 数据上测得……official_comparable = false……收回……不作为结论」 | 旧 PDF 644111 B（抽取 11576 字，`431.3` 命中、更正词全不命中）；md 写入前长度 vs 插入横幅后长度；新 PDF 684922 B（`_regen_perf_pdf.py` 输出 `md 含更正横幅=True（长度 12928）`、`Chrome rc=0`、`含更正痕迹=True ⇒ 可以覆盖`、`REGEN_OK`） |
| **190** | ★★ **"G7 = pdf 与 md 内容一致"这条人工判据，在交付物上**根本配不上对**——因为交付 PDF 与它的源 md **不同名**；而我把它自动化时，判据自检当场又抓出我自己两处判据缺陷**。为防坑 188/189 复发，我写了 `check_report_pdf.py` 把 G7 自动化（判据 A：md 正文开头 40 实字必须出现在 PDF 抽取文本里；判据 B：md 里出现的口径限定语必须在 PDF 同现），**第一次真跑**输出是：`SKIP_NO_MD 性能测试报告.pdf | 无同名 .md`、`SKIP_NO_MD README_镜像环境与运行说明.pdf | 无同名 .md`、`配对数=0`、`REPORT_PDF_FAIL`。即：交付里那份 PDF 叫 `性能测试报告.pdf`，而印它的源 md 叫 **`性能优化报告.md`** ⇒ **"同名才能配对"的写法在这里永远是 0 对**，于是"pdf 与 md 一致"这条判据（人工时代与自动化初版）**从来没有真正被核对过**，评委也无从知道 PDF 出自哪份 md。同时 3 例负向自检抓出我判据实现里的两个错：① `_squash()` 把 `_` 也当排版符删了 ⇒ `official_comparable` 在"PDF 侧"变成 `officialcomparable` ⇒ 判据 B **误报 FAIL**（下划线是标识符的一部分，不是排版符）；② PDF 抽取文本为空时没有走显式 `SKIP`，而是落进判据 A 报成"首部未命中"（把"没检查成"说成"检查失败"，坑 33/34 的老毛病）。修完再跑，又暴露出**真的假失败**：md 顶部写的是 `⚠️`（U+26A0 + U+FE0F 变体选择符），Chrome 印出/pypdf 抽回的是 `⚠` ⇒ 前 40 字**只差这一个不可见字符**就报 FAIL，逐字相同前缀=0 | ① **"一致性"这类判据必须先回答"两个东西凭什么被认为是一对"** —— 我用文件名当配对依据，而文件名是**人会改的**东西（PDF 按交付习惯叫"性能测试报告"，源 md 按内容叫"性能优化报告"），于是判据在最需要它的地方**静默退化成 SKIP**；更要命的是 `SKIP` 被打印成"无同名 md ⇒ 未检查"这种**看起来无害**的样子，谁也不会去追；② **判据自身的自检比判据本身更值钱**：三例坏样本一跑，我立刻看到 `_squash` 删下划线、空文本不 SKIP 两个实现错 —— 若没有自检，这两处会让判据**在错误的方向上"通过"**（把好样本判死或把坏样本放过）；③ **"不可见字符"是逐字判据的天然假失败源**（变体选择符/零宽字符/全半角），判据必须**显式声明剥哪些、为什么**（剥的是看不见的东西，可见正文仍严格逐字比），否则每次假失败都会诱使我把判据放宽（那才是真损失）；④ 失败信息必须**可诊断**：只打印"首部未命中"没法定位，改成同时打印"与 PDF 同位文本的逐字相同前缀长度 + 首个不同字符"后，一眼就看出是 `⚠️` vs `⚠` | ① 交付物实况：`04_性能测试报告\性能测试报告.pdf`（684922 B）与其**不同名**的源 `性能优化报告.md`（21275 B）；`02_README\README_镜像环境与运行说明.pdf` 亦无同名 md；② 首跑输出：`SKIP_NO_MD`×2、`配对数=0 PASS=0 FAIL=0`、`REPORT_PDF_FAIL 没有任何可检查的 md↔pdf 对`；③ 自检输出：坏样本2/3 被判 `**不符**`（got=FAIL want=OK / got=FAIL want=SKIP_NO_EXTRACT）⇒ `REPORT_PDF_SELFTEST_FAIL`；④ 诊断输出：`md head = '⚠️更正说明20260922追加，原文一字未删本报告的性能数字419.0ms、5'` vs `pdf 附近 = '⚠更正说明20260922追加，原文一字未删本报告的性能数字419.0ms、51.1×等是'`，`逐字相同前缀长度 = 0`，`差异字符 md='⚠️更' pdf=''`；⑤ 修后：`REPORT_PDF_SELFTEST_OK 3 例`、`PASS 性能测试报告.pdf | 首部命中 + 口径限定语同现 8/8`、`PASS README_...pdf | 2/2`、`REPORT_PDF_OK pairs=2`；驱动器由 `19 条` 升为 `21 条`（G7 + G7S），人工项由 4 项降为 3 项（G4/G11/G17），总判决 `AUTOMATED_OK_MANUAL_OPEN` |
| **191** | ★★ **我把"我没找到"写成了"不存在"，于是把一个**本来当天就能做**的诊断项在"阻塞"栏里挂了两天**。逐步 batch 指纹（回答"两次运行**每一步喂进去的 batch 是不是同一批**"）一直被我记成阻塞，理由是"`50_train.py` 没有数据通路钩子，管线在第三方 `mindspeed_mm` 里，需要先读框架、不能猜接口"。今天真的去读了一遍（三次只读 grep/sed，**十分钟**），结论是：**框架官方就提供了这个钩子**——`Trainer.__init__(self, args, model_provider=None, **dataloader_provider=None**)`（`mindspeed_mm/fsdp/train/trainer.py:54`），语义见 L62 `Optional custom function to provide the dataloader`，调用点 L78 `... if dataloader_provider is None else dataloader_provider(args)`，而且**框架自己的任务模块就在这么用**：`mindspeed_mm/fsdp/tasks/cosyvoice3/train.py:35 trainer = Trainer(args=args, dataloader_provider=get_cosyvoice_dataloader)`。同一个 RUN 还查出两个相关事实：① `import mindspeed_mm` 单独跑会死在 `ModuleNotFoundError: No module named 'megatron'`（它只在训练环境里可导入）⇒ **"用 import 定位包路径"这条路在本机是错的**，必须走文件系统路径 `/root/MindSpeed-MM/mindspeed_mm`；② 钩子有**陷阱**：provider 的返回值直接当 `train_dataloader` 用（L79），**绕过** `get_dataloader()`（L339-400）里的后处理——`loss_type == "per_token_loss"` 时才会套的 `PrefetchGradAccDataLoader`、train/val 拆分、val 的 `shuffle/drop_last` 覆盖——**所以"注入 provider"本身就是换了一条数据流** | ① **"阻塞"这个状态被我用来表示"我还没查"**，而不是"客观上做不到"——这两者完全不同：前者是**我的进度**，后者是**对象的性质**；把它们写成同一栏，等于让"没查"永远看起来像"不可能"（与坑 181「把'我做不到'说成'做不到'」同族，也与坑 187「当时就定性了清理判据」同族：**都是过早给对象下性质判断**）；② 更具体的机制：我当时的推理是"第三方框架 ⇒ 要读它的接口 ⇒ 会猜错 ⇒ 先不动"，**每一步都合理，合起来却把最容易的一步排到了最后**——因为我默认"读框架 = 大工程"，而实测是**三次 grep/sed**；③ 而"能查到什么"本身也要运气：如果我继续用 `python3 -c 'import mindspeed_mm'` 去定位，只会看到 `No module named 'megatron'`，然后**更有理由认为"这条路不通"** —— 判据选错会**加固**错误结论 | ① 把该原因写成一份可被 Codex 复核的协议文件 `docs/逐步batch指纹_可行性与注入点_20260922.md`：查证过程（含上面三条 file:line 证据）、**陷阱专章**（provider 绕过后处理 ⇒ 用 provider 的诊断运行必须声明 `plan_consistent=false`，且 provider 必须逐行复刻 `get_dataloader()` 并打印自证行）、计划实现（`51_train_fp.py` 包装入口 + `_batchfp.py` 只读代理，逐微批打印 `BATCHFP step/micro/keys/shapes/sha256`）、**负向对照**（故意改一个 batch 相关量，指纹必须出现不同行——否则"相同"没有含义，坑 187 的教训）、以及"这份设计**不能**证明什么"的边界栏；② 立规矩：**待办分两栏**——"未查（我的进度）"与"阻塞（对象的性质，须附"我查了什么、凭什么说做不到"）"，**不许再混在一栏**；③ 立规矩：查第三方框架用**它自己的用法示例**（`tasks/*/train.py`）当第一手资料，**不要**从"我能不能 import 它"开始；④ §5 明确留一项待办（读 `trainer.py` 的 `__main__` 后再写包装入口），**接口读到之前不写代码**（坑 163/179：不猜键名、不造接口）——这次的教训是"该读的没读"，不是"该写的不写" | ① 三次只读调查：`_rcmd_1.sh`（定位包；暴露 `import mindspeed_mm` 失败）、`_rcmd_3.sh`（取 API 面：`PrefetchGradAccDataLoader` L43 / `prepare_base_dataloader` L110 / `prepare_sampler_dataloader` L162 / `prepare_variable_dataloader` L533 / `build_sequential_loader` L505）、`_rcmd_4.sh`（`grep -rn dataloader_provider mindspeed_mm/`：5 处命中，其中 `cosyvoice3/train.py:35` 是**同类用法**；`sed -n '339,400p' trainer.py`：`get_dataloader()` 本体逐行可见）；② 工具侧同步产出 `_rsh.py`（把远端命令放进**文件**再执行）：起因是 PowerShell 5.1 向原生 exe 传参时不转义参数内的双引号，`--run 'echo "$P"; ls | head'` 被拆成多个参数、argparse 报 `unrecognized arguments: ... \| head`；顺手记下它自己暴露的判据弱点——**`rc` 是远端 shell 的返回码**，脚本吞掉内部失败时仍返回 0（故命令文件必须自己 `set -e`，已在输出里注明） |
| **192** | ★★★ **我的分析器会"打印 INVALID、却汇总成『满足』"——而它自己的 9 例自检全部 PASS**。外部复核用**不写文件的内存坏例**实测确认了三条：① 某指标出现 NaN ⇒ 前面打印 `INVALID(non_finite)`，结论行仍说"重复性要求（完整窗口 1..8，阈值 2.0%）：**满足**"；② `steps` 自报 8 步而两条 loss 序列只有 2 步 ⇒ 仍可能说"满足"；③ 零分母被计数（`零分母 1`）却没阻止"满足"。根因在聚合函数：它**只累加 `status == "OK"` 的配对**，于是被判无效的配对贡献 **0 个超阈点、0 个比较点**——"没有任何超阈"被读成"满足"；缺陷②另有独立根因：窗口上界取 `hi = min(len(series), last)` ⇒ **短序列被静默当成合法覆盖**，"窗口内一致性"实际只比了 2 个点。而我原有的 9 例自检**只断言"输出里出现过 INVALID"**，从不看最终判定，所以**它对这种错判完全无感**（"自检通过"本身成了最坏的一种假安全）。修复后自检重写为 14 例，**每条坏例同时断言原因码 + 最终判定措辞 + 退出码** | ① **判据的"打印"与"判定"是两条独立通路**：我实现了逐点的严谨检查（长度/非有限/零分母都抓到了），却在**汇总层**把它们"过滤掉"了 —— 这不是粗心，是**汇总函数的语义写错了**（把它当成"统计 OK 的配对"，而它应该是"对**所有**配对给出总体状态"）；② 更根本：**"没有发现超阈" ⇒ "满足"这个蕴含只在"所有配对都有效"时成立**，而我的代码从未检查这个前提（等价于把"未知"当"通过"）；③ **自检的判据选错了**：断言"输出里出现过 INVALID"验证的是**过程**，但结论由**最终状态**决定 ⇒ 过程正确、结论错误这种组合恰好被漏掉（与坑 33/34 同族，但这次是"自检自身"踩了同一个坑）；④ 缺陷②说明**"窗口不缩"这条冻结判据有第二个漏洞面**：我只防了"按观测缩窗"，没防"序列本身比窗口短时**被动缩窗**"（`min()` 把短序列悄悄变成了新窗口） | ① 重写聚合：**同臂**配对中出现任一 `INVALID` ⇒ 判定 `无法裁决`；出现 `undecidable > 0` ⇒ 判定 `不可判定`；两者都**不许**出现"满足"字样；无同臂配对 ⇒ `无法裁决`；② 新增 `INVALID(coverage_shortfall)`：指标序列长度必须与运行自报步数一致，且必须覆盖协议窗口末步；③ 新增**三层身份认证**（运行状态/配置身份/数据身份，缺任一 ⇒ `UNVERIFIED_IDENTITY`，rc=4；事后重析须显式 `--allow-missing-identity` 且打印降级标记）；④ 新增协议第三分支识别 `inherited_criteria`（缺陷④：工具原先**读不了正在冻结使用的 v2 协议**）；⑤ 退出码分层：0=判定完成（满足/未满足都算有效裁决）、2=用法错、3=步号未认证、4=身份未认证、5=无法裁决；⑥ 自检 9→14 例，含"自报 8 步但序列 2 步""零分母""身份缺失/空串""跨臂无效**不得**拖垮同臂重复性" | ① 旧自检输出（9 例全 PASS）vs 外部复核的内存坏例：坏例仍输出"）：满足"；② 修复后 `numeric_diag_selftest.py` 14/14 PASS，坏例 A1/A2/A3/A4 的 rc 分别 5/5/5/5、A5/A6 rc=4、A7a/A7b rc=3、正对照①"：满足"rc=0、正对照②"）：未满足"rc=0（**"未满足"仍是有效裁决、不是工具失败**）；③ 代码位置：`agg()` 只累加 OK、`hi = min(len(sa), last)`；④ 加固过程还当场暴露**输出顺序缺陷**：身份未认证时判定行先打印"满足"、下一行才说"结论未验证" ⇒ 判据的最终状态**必须体现在判定行本身**（自检坏例 A5 抓出，已改为"**未认证（缺 运行状态/配置身份/数据身份）——不得读作『满足』**"） |
| **193** | ★★ **"配置身份"第一版给出假区分：应同配置的三个运行算出三个不同摘要，而差异只来自输出路径**。给分析器补三层身份时，我从日志里的**有效配置 dump**（L5 起，230 行）导出"配置身份"：三个 A 臂（00_A/03_A/04_A，按协议是同配置重复）第一版得到**三个不同**摘要（`dump_34be7ee9` / `dump_f4deba4c` / `dump_9a8c1204`），而 P2 两次却是同一个 ⇒ 看起来像"三个 A 臂配置各不相同"。归一化（抹平 `20260921_171002` 这类时间戳与 `AB_AVSB_20260921`/`phase1b_*` 这类运行目录名）后**仍是三个不同**；把归一化后的键值落盘再 diff，差异键**只有一个**：`save: '/root/ops/<RUN>/00_A/checkpoint'` vs `.../03_A/...` —— **就是 checkpoint 输出路径里的臂标识**。排除后三个 A 臂才收敛为一个身份（`dump_abe2f33e5286dd21`），P2 两次为 `dump_364d2c6bfb7349ea`，A vs P2 的差异**只剩 1 个键**：`use_deter_comp`（False vs True） | ① **"身份"必须同时满足两个方向**：**够灵敏**（真差异要能体现）与**不假区分**（非语义差异不能体现）——我第一版只顾了前者，于是"同配置三次运行"被算成三个配置，而这会**直接污染"配置效应"这一类结论**（把噪声当变量）；② 具体形态：**输出路径/时间戳/运行名被混进了"配置内容"**；这类字段在**任何**配置 dump 里都存在，所以这不是本次特有的意外，而是**通用陷阱**；③ 我第一轮归一化"看起来做了"（抹了时间戳与运行目录名）却**没验证是否收敛** —— 若我当时不 diff 而是直接采信"已归一化"，就会带着假区分继续往下做（**判据要做正对照：同配置必须同身份**）；④ 排除法要**白名单且可见**：我排除了 `save` 并登记 `config_excluded_keys`；同时**不排除** load/checkpoint 载入类键（那是"是否从已有权重继续"的语义，排除会掩盖真差异）——排除必须是**声明式的**，不能靠"看起来像路径就删" | ① 归一化：时间戳 `\d{8}_\d{6}`、运行目录名、臂标识 ⇒ `<STAMP>`/`<RUN>`；② 白名单排除纯输出键（当前只有 `save`），并把排除项写进产物；③ 同时保留 `config_sha256_raw`（未归一化）供审计；④ **正对照**：三个 A 臂必须同身份（现为 `dump_abe2f33e5286dd21`）、P2 两次必须同身份（`dump_364d2c6bfb7349ea`）——**同配置不同身份即为 FAIL**；⑤ A vs P2 差异键 = `['use_deter_comp']`，与协议声明要动的那个键**完全一致** ⇒ 得到一条**工具给出的**"单变量"证据（不是口头声明）；⑥ 配置 dump 键值随产物落盘（`config_dump`，230 行），日后可 diff，不必回远端重取 | ① 三版摘要：未归一化 `dump_34be7ee9/dump_f4deba4c/dump_9a8c1204` → 归一化后 `dump_0a4d7541/dump_efe0504f/dump_d254b5a0`（**仍不同**）→ 排除 `save` 后 `dump_abe2f33e5286dd21`（**收敛**）；② diff 明细：`差异键数 = 1`、`save '/root/ops/<RUN>/00_A/checkpoint' vs '/root/ops/<RUN>/03_A/checkpoint'`；③ A vs P2：`差异键数 = 1`、`use_deter_comp A='False' P2='True'`；④ 落盘产物 `_reanalysis/results_p0.json`（38367 B）/ `results_p2.json`（25685 B）内含 `config_dump`、`config_excluded_keys`、`status=completed(100/100)`、`log_sha256` |
| **194** | ★★ **profile 工具会把"mock 数据 + 双卡几何"标成 `official_comparable = True`——产物自己声称"与官方可比"**。外部复核指出：`56_profile_run.py` 的 `official_comparable` **只由卡数/几何推导**（`apply_single_die()` 里 `single=False` 就返回 `True`）。于是在 dp2 上跑 **mock 数据**（1 图/样本、cutoff_len=1024、非官方 COCO）时，产物会写下 `"geometry": {..., "official_comparable": true}` —— 而我们的判定链里 `official_comparable` **一直是 false**（数据不同）。这正是我花了一整天去更正的那个口径错误（坑 188），却在**工具里又埋了一遍**：报告改了、判定链改了、**生成产物的代码没改** | ① **我把"可比性"当成了一个几何属性**，而它其实是**两个（甚至三个）身份的共同断言**：几何（world/mbs/gas）**且** 数据（mock vs 官方 COCO）**且** 窗口口径（101–200 / train_iters=100 / 附件5 50–100 三选一）。任何一项不满足就不成立 ⇒ 用"卡数够不够"当判据，等于**用最容易满足的那一项代表整体**；② 更普遍：**口径更正必须沿着"人说 → 文档 → 工具 → 产物"整条链走一遍**，我只做了前两跳；工具里残留的旧口径会**继续生产新产物**，而且比文档更难发现（文档有人读，代码没人读）；③ 修复方向也必须是 **fail-closed**：默认 False、要 True 必须**显式声明非空的数据身份**——"沉默即 mock" | ① 新增 `official_comparable(geometry_ok, data_identity)`：几何一致 **且** 数据身份已显式声明（非空白）才可能 True；② 新增 `--official-data-identity <串>`（不传即 mock ⇒ False）；③ 产物里同时写 `official_comparable`、`official_comparable_reason`（**判断理由**，不让读者猜）、`geometry_matches_official`、`data_identity`；启动日志打印 `PROF_OFFICIAL_COMPARABLE_REASON`；④ 新增 `--selftest-official`：**4 例含 3 个坏样本**（dp2+mock 未声明 ⇒ False；dp2+空白声明 ⇒ False；单 die+已声明官方身份 ⇒ 仍 False；dp2+显式官方身份 ⇒ True），自检在**任何副作用之前**返回；⑤ 注册为自动闸门 **G29**（判据 `PROF_OFFICIAL_SELFTEST_OK`） | ① 修复前代码：`if not single: return doc, int(...), True` ⇒ 几何即可比；② 自检输出：坏样本①③② 全 `got=False want=False`、正对照 `got=True want=True`、`PROF_OFFICIAL_SELFTEST_OK cases=4 failed=0`；③ 驱动器：`PASS G29 profile 官方可比性判据自检（4 例：mock 未声明/空白声明/单 die/正对照）` |
| **195** | ★★ **交付包与活副本里的"状态数字"是手写的，因此在闸门增减后**全部过期**，而没有任何检查会因此报警**。外部复核指出：最新收尾记录是"21 项自动检查通过、3 项人工待确认"，但**部分文档仍写旧的"18＋5"**。实测确认：`docs/推荐配置与保底配置_决策记录_v2_20260922.md` 第 5 行仍写"自动化闸门 **18/18** 通过；人工 **5 项**未闭合"，第 96 行仍写"人工 5 项未闭合（G4/G7/G11/G17/G18）"——而 G18 早已升为自动、G7 也在 2026-09-22 升为自动（人工只剩 G4/G11/G17），自动闸门数也从 18 → 19 → 21（本轮加 G29 后 22）。**同一份交付包里"驱动器输出"与"文档叙述"互相矛盾**，而 21 个自动闸门**没有一个**会检查这件事 | ① **状态数字是"派生量"却被我手写**：闸门数量、人工项名单都随改动变化，手写必然过期；这与坑 188（报告口径 vs 判定链）**同族**——**同一个事实有两个副本，而且没有任何东西强制它们一致**；② 更糟的是**矛盾在交付物内部**：评委可能先读到"18/18 + 人工 5 项"，再看工具输出"22/22 + 人工 3 项"，于是**合理怀疑整份证据**；③ 机制上：我的闸门集合（`prepush_check.py`）是"工具"，文档是"叙述"，**我修工具时没有把叙述当交付物的一部分去更新**（它确实是交付物：它在 `06_Skill` 的 docs 里） | ① 立刻改成当前真实值（22/22、人工 3 项 G4/G11/G17），并补一句"G7/G18 已升为自动闸门"；② 在边界清单里增一条"逐步 batch 指纹尚未实跑"（避免读者以为已闭合）；③ **立规矩：状态数字必须从驱动器输出复制，不许手写**；④ 把它变成**可自动发现的问题**：新增驱动器闸门 **G30**——扫描 Skill 文档与交付文档里的 `自动化闸门 N/N`、`人工 N 项`，与**驱动器自己数出来的**数量比对，不一致即 FAIL（这样"文档过期"再也不会静默存在） | ① 过期位置（两处，且活副本与交付解包副本各一份）：`推荐配置与保底配置_决策记录_v2_20260922.md` L5 与 L96；② grep 证据：`自动化闸门 18/18`、`人工 5 项未闭合（…：G4/G7/G11/G17/G18）`；③ 驱动器实测：`合计：现存 21 条（PASS=21 FAIL=0）`、`人工 3 项`（加 G29 后为 22 条）；④ 修正后文档：`自动化闸门 **22/22** 通过（含 … G7、G7S、G29）`、`人工 3 项未闭合（G4/G11/G17）` |
| **196** | ★★★ **我的"远端会话可用"判据只验证了"进程还活着"，没验证"能从磁盘重建它"——而三个守护的源码文件其实早就没了**。事实链（全部实测）：① `_ops/` 下 `token_broker.py` / `ssh_session.py` / `token_agent.py` **文件已不存在**（目录里只剩 `_run/` 与 `_token/`），它们随坑 180 的工具链删除一起丢了；② 但三个守护**仍在内存中运行**（PID 17916/37776/21204 与之前完全一致，8791/8792 仍在 Listen）⇒ `--status` 有 `PING {"ok":true}`，我这两天一直据此认为"链路正常"；③ 真正暴露问题的时刻是**平台把机器 B 按"1 小时未使用"自动关机之后**：会话日志每 5 秒重复 `SESSION_TOKEN_STALE age=3xxxxs > max_age=240.0 ⇒ 请经纪即时签发` + `SESSION_CONNECT_LOOP_ERROR ModuleNotFoundError: No module named 'token_broker'` —— **卡点是"模块文件不存在"，不是网络、不是平台、不是 token 本身**；④ 我按恢复出的规格把 `token_broker.py` 重写回磁盘后，日志**依旧**报 `RuntimeError: token_broker shim: 未实现 cli_need` —— 因为会话守护在 10:00:01 已经 `import token_broker` 成功过一次，**Python 把模块缓存进 `sys.modules`，此后替换文件对运行中的进程完全无效**；⑤ 而"重启守护"这条退路也断了：`ssh_session.py` 文件不存在，**没有东西可以重启** | ① **"进程存活"与"系统可恢复"是两个不同的性质**，我用前者代替了后者——这是本坑的核心：守护是**长期驻留进程**，它的存活**掩盖**了"启动它的文件已经没了"这件事，于是故障被推迟到"最不方便的时刻"（恰好是平台自动关机后、我需要远端的时候）暴露；② 更深一层：**内存中的代码不是资产**，磁盘上的文件才是。我对"自己的工具链"从未做过"能否从零重建"的检查（而对交付物我做了：`safe_pack_sync --verify-only` 恰恰就是这类检查——**我把交付物管得比工具严**）；③ 连"修好文件"都失效这件事说明：**修复对象与生效对象不是同一个东西**（文件 vs 进程内存），这与坑 189（改的是文件、印的是旧变量）**同族**——"我改了 A，却以为 B 会变"；④ 判据缺陷：我的 `--status` 只看 `PING ok` + `connected`，**没有**"依赖文件是否齐全"这一项（若当时有，两天前就会报警） | ① **立刻**用自描述 shim 把丢失接口问出来（见坑 197）并重写 `token_broker.py`（可重启版：`/`、`/status`、`POST /need`、`POST /token`、`GET /next` + store + `cli_need()`），**留在磁盘上供下次启动**；② 明确记录**仍需重建**：`ssh_session.py`（会话守护本体）、`token_agent.py`（剪贴板守望）——本轮只恢复了经纪；③ 立规矩：**工具链也必须"可从磁盘重建"**，即每个驻留进程都要有一份在其文件缺失时能照着重启的**启动命令 + 依赖清单**（下一步落成 `_ops/README_启动.md`）；④ 立规矩：`--status` 类判据要加"依赖文件齐全性"一项；⑤ 在本机留下"守护靠内存活着"的显式警告，避免下次再被 `PING ok` 骗过 | ① `Get-ChildItem _ops` 只有 `_run`/`_token` 两个目录（`.py` 数为 0）；② 进程： `17916 … ssh_session.py --serve --port 8792 --broker-port 8791 --broker-secret <BROKER_SECRET> --allow-dead`；`37776 … token_broker.py --port 8791 --secret <BROKER_SECRET>`、`21204 … token_agent.py --watch-clipboard --interval 2.5`；③ 会话日志原文：`SESSION_TOKEN_STALE age=30887s(字段=received_at) > max_age=240.0 ⇒ 请经纪即时签发`、`SESSION_CONNECT_LOOP_ERROR ModuleNotFoundError: No module named 'token_broker'（继续重试）`、替换文件后变为 `SESSION_CONNECT_LOOP_ERROR RuntimeError: token_broker shim: 未实现 cli_need（此为探针，见文件头）（继续重试）`；④ 会话 STATUS：`connected=false connects=1 reconnects=1 commands=225 keepalives=194 last_error=connect_loop 异常：ModuleNotFoundError… token_prefix=jt_49CFE39E99B host=199.103.55.150` |
| **197** | ★★ **丢失模块的接口，我没有猜函数名，而是写了一个『自描述 shim』，5 秒内问出了完整签名与配置结构**。做法：把缺失的 `token_broker.py` 写成**探针** —— 模块级 `__getattr__`（Python 3.7+）在**任何属性被访问**时记录名字，并返回一个『记录调用参数后抛错』的可调用对象；探针只写日志（`_run/shim_probe.log`），**不实现任何功能**，因此绝不会让会话『假装连上』。**首次调用即拿到全部答案**：`CALL cli_need args=(8791, '<BROKER_SECRET>') kwargs={'timeout': 45, 'quiet': True, 'cfg': {...}}` —— 而且 `cfg` **自带了整个设计**：store 路径、`hivelab{platform, account, page, env_in_use, ssh_direct_button='SSH 直连', ttl_note, capture_method='页面内点按钮 + 剪贴板（不逆向接口、不保存 Cookie）', capture_tools=[油猴脚本每 4 分钟自动点一次并把命令写剪贴板, F12 Console 版]}`、`_note: endpoint 未配置 ⇒ 自动获取由 capture_tools + token_agent.py --watch-clipboard 完成`。随后再用 HTTP 探活拿到经纪的**完整端点**（`GET /` 自描述）：`POST /need/<secret>` 要 token、`POST /token/<secret>` 送 token、`GET /next?wait=N` 给页面长轮询、`GET /status` 返回 `mint_seq/last_saved_at/tokens_saved/page_polls/store` | ① 面对『接口丢了』，**第一反应可以是『让它自己说出来』，而不是『我猜一个』**：调用方（会话守护）**就在运行中**，它每隔 5 秒就会用真实参数撞一次 —— 这是最权威的规格来源，比任何文档都准；② 关键设计是**探针必须『只记录、不实现』**：若 shim 返回一个看似可用的假 token，会话会『看起来连上』，把问题推迟成更难查的形态（与坑 33/34「静默当通过」同族）；③ 这也纠正了我对『不许猜』的机械理解：**不猜 ≠ 什么都不做**，而是『用可观测的实验去问』，这次问的成本是 5 秒；④ `cfg` 里的 `why_not_api`（不逆向接口、不保存 Cookie、按可见文本定位）说明原设计**刻意避开**凭证依赖 —— 恢复时必须保留这条边界，不能为了『自动』改成存 Cookie | ① 探针文件：`_ops/token_broker.py`（先以 `token_broker_shim.py` 落盘再改名，避开写工具的观测缓存冲突）；② 按恢复出的规格**重写可重启实现**：`serve()` + `Broker` + `cli_need()` + `--put` 手动送 token，`allow_reuse_address=False` 沿用坑 171 的教训，返回对象做成 dict+属性双用以免猜返回约定；③ 立规矩（并落到工具里）：**格式判据要同时校验行首与行尾/列数** —— 本次 `_append_pitfall.py` 只校验行首，于是半截行照样写进坑表；④ 仍需重建 `ssh_session.py`/`token_agent.py`（见坑 196） | ① `_run/shim_probe.log`：`10:00:01 SHIM_LOADED pid=17916 cwd=D:\昇腾项目` → `ACCESS cli_need` → `CALL cli_need args=(…) kwargs={…}`（每 5 秒一条，共 3 条）；② HTTP 探活：`GET / -> {"ok": true, "service": "token_broker", "hint": "POST /token/<secret> 送 token；GET /next 长轮询；POST /need/<secret> 要 token"}`、`POST /need/<BROKER_SECRET> -> {"ok": true, "seq": 1}`、`GET /status -> {"mint_seq": 1, "tokens_saved": 0, "page_polls": 1}`；③ 重写实现：`_ops/token_broker.py`（编译通过）；④ 假 token 端到端测试被活经纪以 `HTTP 400` 拒绝（其校验严于 store 字段本身） |
| **198** | ★★★ **跳板连不上的真因不是 token 过期，而是我"猜"了认证方式——而我从 OpenSSH 自己打印的原文里 30 秒就拿到了答案**。事故链：重建会话守护后连接跳板，paramiko 报 `AuthenticationException: Authentication failed: transport shut down or saw EOF`。我把它归因为"token 过了 5 分钟窗口"（因为是在 10:39:51 才首次连接，而我在写守护代码上花了约 6 分钟）。用户**又给了一份新 token**，仍然失败 ⇒ 归因被推翻。随后用 paramiko **逐个认证方式**试（每种一条全新 transport）：`auth_none(juser)` **成功**、`auth_password(juser, 密码)` 失败（同样 EOF）、`auth_interactive_dumb` 也失败。改成 `auth_none` 后认证过了，但**开通道仍被拒**：`ChannelException(1, 'Administratively prohibited')`。决定性的一步是让 **OpenSSH 自己跑一遍并看 `-vv`**：它打印出为 `-J` 生成的 implicit ProxyCommand —— `ssh -l jt_73940E…:DC6319… -p 2234 -W "[%h]:%p" 113.47.8.48` ⇒ **`jt_TOKEN:TOKENPASS` 整串就是用户名**（OpenSSH 的 `-J` 只接受 `[user@]host[:port]`，所以 `-J a:b@host` 里的 `a:b` 全算用户名，平台正是利用这一点把 token 与密码一起塞进用户名字段）。我按 `:` 拆开只用了前半段 ⇒ **认证通过、但那个身份没被授权到该目标**。同时 OpenSSH 还打出 `channel 0: open failed: administratively prohibited: **only direct-tcpip is permitted**` ⇒ 这个跳板是**纯代理**，不能开 session 通道（也就不能在它上面执行命令） | ① **`EOF` 被我当成了"token 过期"**，而它其实是"服务器不接受这种认证"（Go 网关对不需要/不接受的认证直接掐断，而不是回一个干净的 `failure`）——**把"连接被关"这一种现象绑定到唯一原因，是最典型的单因误判**（与坑 158「两种失败原因打成同一句话」同族，只是这次是我自己误读）；② 正确顺序应该是**先问"服务器接受什么"，再谈"凭证是否有效"**：`auth_none`/`allowed_types` 是**不需要凭证**就能问的，成本 1 次连接；③ **"认证通过"不等于"被授权"**：我拿到 `auth_none` 成功后就不再怀疑用户名，直到通道被拒；两层是独立的（认证=你是谁，授权=你能去哪）；④ 最根本的一条：**平台已经把它实际用的命令原文给了我（OpenSSH `-vv` 的 ProxyCommand 行），我却先花了两轮去猜**——"照抄原文/让工具自己打印"永远优先于"按我的直觉解析它的语义"（坑 197 同族：让对象自己说话）| ① 守护的跳板认证改为 `paramiko.Transport` + `auth_none(整串 token)`；② 目标机仍用控制台给的**连接密码**（密码用于目标，不用于跳板）；③ 写进 `README_启动与重建清单.md`：**"跳板只认用户名（整串 token）、送密码会被掐断、只允许 direct-tcpip"**；④ 立规矩：**任何"连不通"先分辨是"认证/授权/网络哪一层"**，并在日志里各层分别打印（不要都归结成一句）；⑤ 换 token 的正确姿势：token 5 分钟且**通道开启时校验**（用旧 token 重连被拒）⇒ 必须"**先让守护待命、再喂 token**"（守护盯 store 文件，新记录一落盘 2 秒内自动建连），而不是"我先写代码、等有空再连" | ① 两次失败的报错原文：`AuthenticationException: Authentication failed: transport shut down or saw EOF`（新 token 亦然）；② 逐方式诊断：`[OK] auth_none ⇒ 认证成功`、`[FAIL] auth_password ⇒ AuthenticationException`、`[FAIL] auth_interactive_dumb ⇒ AuthenticationException`；③ 通道被拒：`ChannelException(1, 'Administratively prohibited')`；④ OpenSSH `-vv` 原文：`debug1: Setting implicit ProxyCommand from ProxyJump: "…ssh.exe" -l jt_<REDACTED_ID>:<REDACTED_SECRET>… -p 2234 -W "[%h]:%p" 113.47.8.48` 与 `channel 0: open failed: administratively prohibited: only direct-tcpip is permitted`；⑤ 修好后：`SESSION_CONNECTED host=root@199.98.58.213 token=jt_73940E035C4… connects=1`、`STATUS connected=True`；⑥ 目标 IP 已换（`199.98.58.213`），但 `/root/ops`、Skill、`MindSpeed-MM`、模型全在 ⇒ 同一份磁盘换了 IP |
| **199** | ★★ **我重建的会话守护在"大输出"时丢尾部——拦住它的不是我的自检，而是下游的逐文件 SHA256 校验**。拉取 555 KB 的 `train.log` 时连续两次报 `SHA256 不一致（远端=4ea6cbcf0654 本地=…）`，而 221 KB 的 `fp.log` 正常。我先排除"文件还在被写"：远端连查两次 `ls -l`+`sha256sum`，大小与哈希**4 秒内完全不变**，且 `pgrep trainer.py/torchrun` 计数为 0 ⇒ 文件是静止的，问题在我的读取。根因：我为会话守护手写了"轮询 `recv_ready()` + `exit_status_ready()`"的收尾逻辑，而 **`exit_status_ready()` 可能先于"stdout 数据全部到达"为真**，于是我只排空了**当时已到**的缓冲就返回 ⇒ 尾部丢失。同一轮还暴露两个我自己的工具缺陷：① PowerShell 5.1 的 `Set-Content -Encoding UTF8` **会写 BOM**，远端 bash 看到 `\357\273\277set` 报 `command not found: $'\357\273\277set'` ⇒ **脚本第一行静默失效、其余行照跑**（最容易骗过"看起来跑过了"）；② 我写的两个补丁脚本把**断言字符串写在注释里**（`assert "exit_status_ready()" not in back`，而注释里正含这个词）⇒ 补丁明明成功却断言崩掉（与坑 192 家族同族：**断言要针对被测对象，不是自己的文字**） | ① **能用标准 API 就不要手写协议**：paramiko 的 `stdout.read()` 阻塞到通道 EOF，天然不截断；我手写轮询只是为了"能设 deadline"，结果用正确性换了灵活性；② 更重要的正面结论：**这次我的错误没有变成假证据**——因为 `pull_evidence.py` 对每个文件都做**双侧 SHA256**（远端算一次、本地解码后再算一次，不等就拒绝落盘）⇒ 错误表现为**显式的"拉取失败"**，而不是"库里多了一份截断日志"。这正是 fail-closed 判据的价值：**它不阻止我犯错，但它阻止错误被静默归档**；③ BOM 这一条的形态值得单独记住：**它污染的是文件的第一行**（shebang/`set -e` 的位置），失败不会让脚本整体报错，只会让**最关键的防御（`set -e`）悄悄失效** | ① 修法：`Conn.run` 改为 `out = stdout.read(); err = stderr.read(); rc = ch.recv_exit_status()`（异常时明确返回"本地读取失败"，不返回半截数据），已签名 `PATCH_READ_OK`；② `_rsh.py` 增加 BOM 剥离并**显式打印** `RSH_NOTE 已剥除命令文件开头的 BOM`；③ 立规矩：**任何"读取远端大输出"的路径都要有长度/哈希自证**（本轮的 train.log 就是靠它暴露的）；④ 立规矩：补丁脚本的断言只断言**代码块结构与行为**，不断言"某个词没出现"（词会出现在我自己的注释里） | ① 连续两次：`PULL_FAIL …/train.log SHA256 不一致（远端=4ea6cbcf0654 本地=beba8999acb2）`、`PULL_FAIL …/train.log SHA256 不一致（远端=7eb2cdabdb1d 本地=a78ac0d24c44）`；② 排除"仍在写"：`full: 555821 B sha=4ea6cbcf0654` → 4 秒后 `full: 555821 B sha=4ea6cbcf0654`，`还有没有相关进程：0 0`；③ BOM：`bash: line 1: $'\357\273\277set': command not found`；④ 修后重拉成功：`PULL_OK /root/ops/fp51full…/fp.log 221068 B sha256=aecffd0d2cbf`、`PULL_OK /root/ops/fp51neg…/fp.log 221068 B sha256=393a0e4da19d`；⑤ 落本地证据目录 `远端证据_20260922\fp51\`：9 个文件（含 run1/run2 同为 `9b8d5edb62a9` 的跨运行相同证据） |
| **200** | ★★ **我把控制台给的连接命令原文（含真 token）当作"现场证据"贴进了坑表，被打包的内容闸门拦住；而修它时我又因为"判定与脱敏用了两套正则"白跑一轮**。第一段：坑 198 的"现场证据"列里我贴了 `ssh -l jt_<REDACTED_ID>:<REDACTED_SECRET>… -p 2234 …`（就是我用来定位问题的那行 OpenSSH 原文）⇒ `safe_pack_sync.py --apply` 的内容闸门报 `! docs\PITFALLS_坑表.md 含疑似真 token（jt_<ID>:<HEX>）` ⇒ **拒绝出包**（zip 未重建），于是 `G22 三方一致性` 也跟着 FAIL。第二段：我写了 `_redact_tokens.py` 去脱敏，**却自己另定了一个更严的正则**（`jt_[0-9A-Fa-f]{12,}:[0-9A-Fa-f]{16,}`），而我那行里的 secret 恰好被我用 `…` 截断成 **8 位十六进制** ⇒ 落在"**闸门命中、我的正则不命中**"的区间 ⇒ 输出 `REDACT_OK 命中文件=0 已改=0`，可重打时闸门**照样报命中** | ① 我把"证据原文"当成普通文本抄进交付文档，而**凭据与证据是两回事**：证据需要的是"这行命令长什么样"，不需要"这串密钥的真实字节"——我抄全文是**顺手**，不是必要；② 更值得记的是第二段：**"判定某物"与"修某物"用了两份独立定义就必然分叉**（同族：坑 188 的口径双副本、坑 193 配置身份的假区分）。我第一反应是"再写一个脱敏正则"，而正确反应是"**复用闸门那份**"；③ 这类分叉的特点是**看起来都成功**：脱敏脚本报 `REDACT_OK`，闸门报"命中"，两边都不算错，只是它们说的不是同一件事；④ 反面印证了闸门的价值：如果我当时图省事直接改 zip 或放宽闸门，**真凭证就进了交付包**（评委能直接看到我们的跳板 token） | ① 把真 token 判定提成**模块级单一来源**：`safe_pack_sync.py` 里定义 `TOKEN_RE = re.compile(r"jt_[0-9A-Fa-f]{4,}:[0-9A-Fa-f]{8,}")`，闸门用它、`_redact_tokens.py` **import 它**（不再各写一份）；② 全量脱敏为 `jt_<REDACTED_ID>:<REDACTED_SECRET>`，写盘前备份、**写盘后回读断言不再命中**；③ 立规矩：交付文档里引用命令原文时**一律脱敏**，凭证只允许存在 `_ops/_token/current.json`（非交付物）；④ 立规矩：**任何"检测—修复"成对的东西共用同一份判据定义**，否则修完必须再用检测器复验（本轮正是靠重打时才发现的）；⑤ 闸门保持 fail-closed 不动（**不许为了让流程走通而放宽凭证闸门**） | ① 闸门原文：`G-e 内容闸门未通过（1 项）—— **拒绝出包**：` + `! docs\PITFALLS_坑表.md 含疑似真 token（jt_<ID>:<HEX>）`；② 连带失败：`FAIL G22 三方一致性 … 判据=<标记不匹配>`（zip 未重建）；③ 第一轮脱敏的空转证据：`REDACT_OK 命中文件=0 已改=0 未改=0`（而闸门仍报命中）；④ 改为共用 `TOKEN_RE` 后：`命中 docs\PITFALLS_坑表.md 1 处` → `REDACTED … 1 处已脱敏` → `REDACT_OK 命中文件=1 已改=1 未改=0`；⑤ 重打与全闸门：`SAFE_PACK_SYNC_OK`（两处解包副本 `条目=118 逐文件哈希不一致=0`）、`合计：现存 23 条（PASS=23 FAIL=0）`、`AUTOMATED_OK_MANUAL_OPEN` |
| **201** | ★★★ **真 token 一度进了交付 zip，而两道闸门都没拦住——因为备份写在了交付树内、且敏感串扫描按扩展名过滤**。事实链（全部实测）：① 我为修坑 196/197/200 写的一次性脚本把备份写在**被改文件旁边**（`docs/PITFALLS_坑表.md.bak_20260922_100200`、`…bak_redact_20260922_105953` 等）；② 这些文件就在 Skill 活副本树内，于是 `safe_pack_sync.py --apply` 把它们**一起打进了交付 zip**：解包检查显示 `zip 条目 = 119，含备份/临时文件的条目 = 4`；③ 其中 **2 份是"脱敏前"的坑表副本** ⇒ 用闸门自己那份 `TOKEN_RE` 扫 zip 内条目得到 `备份文件里含疑似真 token 的个数 = 2` —— **真凭证确实在交付物里**；④ 原闸门为什么没报：敏感串扫描写作 `if rel.endswith((".md", ".py", …)):`，而 `.bak_redact_20260922_105953` **没有扩展名** ⇒ **整个文件被跳过**（它检查的是"已知文本类型"，不是"是不是文本"）；⑤ 触发这一连串排查的起点，是我在坑 200 里也抄了同一段命令原文、被闸门拦住 —— 也就是说**同一个错误我犯了三次**（坑 198、坑 200、以及这次的备份副本） | ① **根因不是"忘了脱敏"，而是"备份写在了会被打包的地方"**：我的工具把 `p + ".bak"` 当默认（就地备份），而 Skill 树同时是"工作目录"和"交付物来源"——**同一个目录承担两个角色，就必然把工作痕迹一起交付**；② 判据的形态选错了：**"按扩展名列表扫描"等于"只检查我以为的文本文件"**，而泄露恰恰出现在我看不到的类型里（无扩展名）——**判据必须按"内容能不能解码"而不是"名字像不像文本"**；③ 这是"fail-closed 闸门"的第二次失效（第一次是坑 200 的正则分叉），而两次失效的形态相同：**闸门定义了一个范围，而我的产物跑到了范围之外**（先分叉定义、后漏掉类型）；④ 最刺眼的一点：我前一天刚写下"凭证不得进交付物"的规矩，第二天就在**同一类动作**（写文档、做备份）上破了三次——说明**规矩没有落到工具里就等于没有** | ① `safe_pack_sync.py` 内容闸门加两条：**备份/临时名一律拒绝入包**（`.bak|.orig|.tmp$|~$|.swp$`）、**敏感串扫描对全部文件生效**（不再按扩展名过滤，二进制文件按空串处理但**不跳过**）；② 新增 0 字节文件检查（空文件不该入包）；③ 把 5 个树内备份文件移出到 `工具_重建\_backup\skill_baks\`，并修 `_redact_tokens.py` 的备份路径为**树外**（写盘时打印"备份（树外）: …"）；④ 重打并复查：`zip 条目 = 115，含备份/临时文件 = 0，含 token = 0`；⑤ 立规矩：**凡是"备份/临时/中间产物"一律写在交付树之外**；⑥ 立规矩：**检测"敏感内容"必须与文件类型无关**（能解码就扫） | ① 树内备份清单（5 个，已移出）：`docs\PITFALLS_坑表.md.bak_20260922_100200`、`…100214`、`…bak_redact_…105953`、`…bak_redact_…110111`、`protocols\one_step_replay_20260922.json.bak_survey_20260922_110354`；② 交付 zip 证据（修前）：`zip 条目 = 119`、`含备份/临时文件的条目 = 4`、`备份文件里含疑似真 token 的个数 = 2`；③ 修后：`SAFE_PACK_SYNC_OK` + `zip 条目 = 115`、`含备份/临时文件的条目 = 0`、`含 token = 0`；④ 干跑会**先列出将删除的陈旧文件**（正是那 4 个备份），符合"先看到要删什么"的纪律；⑤ 全闸门：`合计：现存 23 条（PASS=23 FAIL=0）`、`SAFE_PACK_VERIFY_OK items=115`、Skill 内备份垃圾计数 = 0 |
| **202** | ★★ **FSDP2 的参数/梯度是 DTensor 子类，`tensor.numpy()` 直接报错——首次真机运行就撞上**。一次回放的转储阶段报 `RuntimeError: .numpy() is not supported for tensor subclasses.`，两条 rank 全挂（rc=1）。原因：FSDP2 下 `model.named_parameters()` 与 `param.grad` **都是 DTensor 子类**，而我的摘要函数按普通 Tensor 写（`tt.cpu().numpy().tobytes()`）。处置：① 有 `to_local()` 就先取**本地分片**；② 仍是子类身份就 `as_subclass(torch.Tensor)` 脱掉壳；③ 最后才 `numpy()`。改完即通过（两轮 rc=0，产出 4 份 dump） | ① 我把"分片张量"当成了"普通张量"——**FSDP2 里"一个参数"的概念本身是多份的**，不显式选择口径（本地分片 or 全量）就没法谈"逐字节比较"；② 这次错误是**立刻暴露**的（首跑即崩），属于幸运的一类；真正危险的是**不报错的错误**（比如用 `gather` 拿全量后比较，语义其实变了却照样跑出数字）；③ 顺带固定了一条语义：**跨运行比较比的是同一 rank 的本地分片**（这才是 FSDP2 下有意义的判等），并把它写进产物（`to_local`）——**口径必须在产物里可见**，否则读者会以为比的是全量 | ① 改 `_tensor_digest`：`to_local()` → `as_subclass(torch.Tensor)` → `float32` → `numpy()`；② 重跑后两轮 rc=0；③ 记录本次为"框架特性"而非"我实现错"的部分：报错文本本身给出了答案（"not supported for tensor subclasses"），**先读报错再动手**省了一轮试错 | ① 首跑失败原文（两 rank 相同）：`RuntimeError: .numpy() is not supported for tensor subclasses.` → `torch.distributed.elastic.multiprocessing.errors.ChildFailedError` → `52_replay.py FAILED`；② 修后：`a rc=0  b rc=0`，产出 `replay_dump.rank0.json`（1194571 B）/`rank1.json`（1200565 B）等 4 份；③ 上传校验：`PUT_OK … 52_replay.py 12562 B sha256=4659d18e8779` |
| **203** | ★★★ **我的比对工具把"结果"当成"前提"，于是它恰好把本轮唯一的发现挡掉了**。首次跨运行比对输出：`fixed.grads_sha_all **不同**`、`fixed.opt_state_sha_all **不同**` ⇒ 工具判定 `★ 固定状态不一致（['grads_sha_all','opt_state_sha_all']）⇒ 本轮**不可比**` + `REPLAY_FAIL fixed_state_mismatch`，**直接退出，没有再比 C1/C3/C4**。而"前提一致（输入/初始权重/RNG/学习率/步号）"在输出里都显示为"同"，真正要找的 **C1 逐字节相同、C3 不同** 恰恰被这次提前退出**藏起来了**。根因：我在 dump 里把 `grads_sha_all / params_after_sha_all / opt_state_sha_all` 和 `input/params_before/rng/lr/iteration` 一起塞进了 `fixed`，而比对工具把 `fixed` 整块当前提 | ① **"前提"与"结果"的分界错了，判据就会把信号当噪声丢掉**：梯度不同是**要被解释的现象**，不是"实验没做干净"的证据；我把自证信息（"我固定了哪些状态"）和观测结果（"我看到了什么"）混在一个命名空间里，工具就必然误读；② 与坑 192（聚合层只累加 OK 的配对、把 INVALID 当没发现问题）**同族**：都是**分类错误导致的静默过滤**，且都比"算错一个数"危险得多；③ 更值得记的是这次的**暴露方式**：工具打印了完整的逐项对比（那 8 行都是"同/不同"），我一眼就能看出"前提全同、结果里梯度不同"——**输出足够详细**救了这次（若它只打印一句 FAIL，我会去查无关的方向）| ① 把字段显式分类：`PRE = (input_sha_all, params_before_sha_all, rng_cpu_sha, iteration, lr)`、`OUT = (grads_sha_all, params_after_sha_all, opt_state_sha_all)`，并在每行前打印 `[前提]/[结果]` 标签（让分类本身可见）；② 只有**前提**不一致才判 `不可比`；③ 重跑比对后得到正确定论：`REPLAY_VERDICT rank=0 first_diff=C3`、`rank=1 first_diff=C3`（两 rank 一致）；④ 立规矩：**自证信息与观测结果必须分字段存放**（`provenance` / `observation` 两类），否则任何下游工具都可能误读 | ① 误判原文：`★ 固定状态不一致（['grads_sha_all', 'opt_state_sha_all']）⇒ 本轮**不可比**` + `REPLAY_FAIL fixed_state_mismatch rank=0`；② 修后输出（节选）：`[前提] input_sha_all 同`、`[前提] params_before_sha_all 同`、`[前提] rng_cpu_sha 同`、`[结果] grads_sha_all **不同**`、`C1 前向输出：loss 逐字节相同（4627dcf31a3c vs 4627dcf31a3c）；logits 逐字节相同`、`C3 通信后梯度: **不同**（sha_all 72448bb123d14d47 ≠ 223de54e52741f6d；逐张量不同 200/200）`、`REPLAY_VERDICT rank=0 first_diff=C3`、`REPLAY_VERDICT rank=1 first_diff=C3` |
| **204** | ★★ **`lr = 0.0`（warmup 起点）⇒ 第一次更新是"空更新"，C4 这个对比点在本配置下没有区分力**。回放产物里 `fixed.lr = 0.0`、`iteration = 0`，而 `params_after_sha_all` **等于** `params_before_sha_all`（rank0 `945de97b1d4be973`、rank1 `3f10e13c5b44c2e0`，两轮皆然）⇒ `optimizer.step()` 在这套调度下**什么都没改**。于是 C4 的"逐字节相同"**不能**作为"优化器路径可复现"的证据（它只是"没动"）。同一族的第二例：mock 数据每步都是同一批 ⇒ `num_workers=1`（P1）那一臂在它声称要检验的东西上**没有区分力** | ① 这是我第三次踩"**判据缺乏区分力**"：判据形式上看得出"相同/不同"，但**在被测配置下无论机制如何都会给出同一个结果**——它测的不是它声称测的东西；② 更隐蔽的是：这类判据**给出的是"通过"**（C4 相同、P1 未见差异），看起来像"证据支持结论"，实际是**空转**；③ 根因是我总在"机制成立与否"上想，却少了那半句检查：**这个配置下，如果机制真的坏了，判据会不一样吗？**；④ 反过来说，`lr=0` 也不是我的失误配置——那是 warmup 的真实起点；**发现它靠的是把固定状态的"实际值"打进了产物**（`REPLAY_FIXED … lr=0.0 step=0`），否则我会一直以为自己跑的是"一步真实更新" | ① 在产物与报告里**显式声明该点的区分力**：比对工具新增一行警告 `⚠ **C4 无区分力**：params_after == params_before（lr=0.0 ⇒ 第一次更新是空更新）⇒ C4 的『相同』不能作为『优化器路径可复现』的证据`；② 写入协议 `results_2026_09_22.caveats`：**若将来要走 C4 分支，必须改用步 0 学习率非零的配置并在协议里声明该改动**；③ 立规矩：**每个对比点都要写"反向条件"**——"若机制失效，这一点会变成什么"；答不出来就是没有区分力，不能计入结论；④ 本轮结论不依赖 C4（本地化结论由 C1/C3 给出），故无需重跑 | ① 产物证据：`REPLAY_FIXED rank=1 … lr=0.0 step=0`、`params_sha=3f10e13c5b44c2e0` 与 `REPLAY_C4 rank=1 … sha_all=3f10e13c5b44c2e0`（**同值**）；② 比对工具新增警告行（原文见 `diff.txt`，4034 B，sha256=e12bf0d6e783，已存本地 `远端证据_20260922/replay/`）；③ 协议 caveats 已写入（`results_2026_09_22.caveats` 第 1 条）；④ 同族前例：mock 数据"每步同一批"（见 `protocols/batch_fingerprint_20260922.json` 的 implications） |
| **205** | ★★ **profile 桶比例的分辨率不足以决定优化方向——若我据此宣布"该做局部重计算"，就是老错复发**。短 profile 结果（50–55 步窗口、5 个采样步、Stage 归一）：`Computing 78.0%`、`Free 19.3%`、`Communication 5.6%`、`Communication(Not Overlapped) 2.7%`，另有 `kernel_launches_per_profiled_step = 7150`。按 Codex 给的判定表，通信未重叠只有 2.7% ⇒ **通信不是瓶颈**（这条可以直接下结论）；但面对"Computing 78%"我却**不能**说"所以该做重计算/激活优化"——因为 **Computing 是混合量**：前向、反向、以及**重计算（recompute）**都算在同一个桶里，桶比例**在构造上就无法区分**它们。要区分必须再采一次带调用栈或解析 `kernel_details.csv` 的算子明细 | ① 这是本项目**第三次**遇到"判据分辨率低于问题尺度"（前两次：loss 打印只有 4~6 位有效数字、mock 数据每步同一批）——而这次它是**工具固有**的：torch_npu profiler 的五桶定义就是这样切的，**不是我实现错**；② 危险在于"78% 计算占据"这个数字**看起来非常有指导性**，任何人（包括我）都会立刻想写"下一步优化计算"；而它实际只够支撑"通信不是瓶颈"这一条**否定**结论；③ 反过来，这次也证明了**先把判据写下来再跑**的价值：我在 `protocols/profile_short_20260922.json` 的 criteria 里事先写了"Computing ≥70% ⇒ 必须再做归因**才允许**宣布方向"，所以拿到数字时不能顺手发挥；④ `7150 次 kernel 发射/步`这条数据反而更具体（它指向发射/主机侧），但它同样需要明细才能定性（是重复小算子还是必要的并行算子）| ① 把 profile 结果与判据冻结成 `protocols/profile_short_20260922.json`：写明 `decision_2026_09_22` = **不投通信组合**、并专设字段 `explicitly_not_decided`（**不宣布**"该做局部重计算/算子融合"）与理由；② 明确下一步：带 `--with-stack`（或解析 `kernel_details.csv` 算子明细）再采同一窗口，把 Computing 拆成"重计算 vs 真实计算"、把 Free 拆成"主机下发/同步/数据等待"，**之后只做 profile 支持的那一类优化**；③ 记录边界：样本仅 5 步、数据是 mock（数据等待成分可能被低估）、不含显存曲线、739.5 ms 是被拖慢的值不可与 432.7 ms 对标；④ 立规矩：**遇到"看起来很有指导性"的比例数字，先问"这个桶里混了什么"** | ① `result.json`（8379 B，sha256=4b501b282063，已存本地 `远端证据_20260922/profile/`）：`buckets` 五桶均值、`shares_pct`、`kernel_launches_per_profiled_step=7150`、`note=profiling 会拖慢训练…只用于桶比例归因`；② `step_trace_time.csv`（812 B，sha256=d68812a490c4）表头十列（含 `Bubble`/`Preparing`）；③ 运行侧：`P2PROF_OK out=/root/ops/prof_20260922_112113`、`iteration 100/100`、`driver_rc=0`；④ 同时验证了 G29 那次修复在真机生效：`official_comparable=False`、`official_comparable_reason=数据身份未声明（默认视为 mock ⇒ 不可与官方比较）` |
| **206** | ★★ **"桶比例不够用"这件事，靠算子级聚合就补上了——而补上之后发现的方向和我原先的直觉不同**。坑 205 里我拒绝凭 `Computing 78%` 宣布方向，随后按协议要求做算子级归因（同一批 profile 的 `kernel_details.csv`，35750 行 = 7150 发射/步 × 5 步），聚合出的结果是：① **GDN/linear-attn 算子族合计约 28.2%**（`prepare_wy_repr_bwd` 6.17% / `causal_conv1d_bwd` 5.44% / `chunk_bwd_kernel_dqkwg` 5.27% / `chunk_gated_delta_rule_fwd_kernel_h` 5.03% / `chunk_gated_delta_rule_bwd_kernel_dhu` 4.61% / `causal_conv1d_fwd` 1.70%），**比 matmul 族（20.50%）还大**；② **小算子/拷贝/cast 簇约 16.3%**、调用合计近 2 万次（`Transpose` 4055 次、`Cast` 3815 次、`Slice` 2750 次、`Mul` 2540 次、`Add` 1930 次、`ApplyAdamW` 1600 次…），与 19.3% 的 `Free`、7150 发射/步互相吻合。我原本的直觉是"matmul 最大、该看算子融合"，而数据说：**最大的单一家族是 GDN/linear-attn** | ① **"分辨率不够"不等于"没得做"**：正确动作是**换一把更细的尺子**（桶 → 算子），而不是放弃或猜；这与坑 205 的教训合起来才是完整的：先拒绝凭粗判据下结论，再去找能支撑结论的细判据；② **归属层级决定看到什么**：按单算子看，第一名是 matmul（20.5%）；按**算子族**看，第一名是 GDN 系（28.2%）——同一份数据、不同归族规则给出不同"最大项"，所以**归族规则必须写进产物**（我把它写进了协议的 `families` 字段），否则读者无法复核；③ **时间维度的归因不能替代资源维度**：GDN 系耗时占比大**不代表**"该把它换掉"——早先的观察是 GDN 相关**显存**增长，而本协议 `with_memory=false` 没采显存 ⇒ "是否值得恢复更快路径"必须等显存证据（这与坑 204 的"区分力"检查同源：**结论必须与它真正测到的量匹配**）；④ 顺带确认了一条与 Free 一致的独立线索：近 2 万次小算子发射 ≈ 每步 4000 次，与 7150 发射/步、19.3% Free 三者互相印证 ⇒ **主机/小算子侧确实有可观空间**（这条是"两个独立量指向同一结论"，比单量更硬） | ① 把 kernel 级归因写进 `protocols/profile_short_20260922.json` 的 `kernel_attribution_2026_09_22`：Top15 算子（含占比与调用次数）、三个族（GDN 28.22% / matmul 20.50% / 小算子簇 16.3%）、**修正后的结论**与"仍未决定"两项（重计算占比需 `--with-stack`、显存曲线未采）；② 明确候选排序：GDN 族 →（先确认是否在关键路径 + 采显存）小算子簇 →（暂不投入）通信组合；③ 立规矩：**归族规则与数据源一起入档**，且**只用比例**（`Duration(us)` 是设备侧时间，不含主机下发/同步，**不能**与桶的 Free 相加当 100%）；④ 下一步不动手优化，先补两件取证：重计算占比与每 rank 显存曲线 | ① 聚合输出：`kernel 行数=35750 总耗时=3172977.4 us`、`PROF_KERNEL_TOP1 aclnnMatmul_MatMulV3Common_MatMulV3 = 20.50%`、按调用次数第一 `aclnnInplaceCopy_TransposeAiCore_Transpose 4055 次`；② 族合计（我按名字归族）：GDN 系 28.22%、matmul 20.50%、小算子簇 ≈16.3%；③ 与桶证据的呼应：`Free 19.3%`、`kernel_launches_per_profiled_step 7150`、小算子簇≈每步 4000 次发射；④ 数据源同批：`kernel_details.csv`（35750 行，位于 `/root/ops/prof_20260922_112113/prof/<...>_ascend_pt/ASCEND_PROFILER_OUTPUT/`） |
| **207** | ★★ **`pkill -f 'hbm_sample.sh'` 把我自己的命令一起杀了（`rc=-1`），而这类"自匹配"我今天已经栽过第三次**。给"带显存采集的 profile"写启动脚本时，我在里面加了 `pkill -f 'hbm_sample.sh'` 去清理上一轮遗留的采样器；结果那次远端调用直接返回 `RSH_FAIL rc=-1`，BASE 目录建了但**一个文件都没产出**（连 `hbm.csv` 都没有）。原因：会话守护把命令交给 `bash -c` 执行，**命令全文就在 bash 自己的 cmdline 里**，里面含 `hbm_sample.sh` 字样 ⇒ `pkill -f` 匹配到**自己所在的 shell**并把它杀掉，脚本在 `mkdir` 之后就没了。同一件事的另两次：坑 158/175 家族（`pgrep -f trainer.py` 匹配到我自己的命令 ⇒ 假 `NPU_BUSY`）。处置：去掉 `pkill`，改用"**不依赖杀进程**"的清理方式（每轮用唯一目录；旧采样器靠写 `stop_hbm` 标记文件自行退出）| ① `pkill -f`/`pgrep -f` 的匹配对象是**整条命令行**，而"我这条命令的文本"里天然包含我要匹配的模式 —— **查询动作本身污染了查询对象**（观测者效应在进程表上的具体形态）；② 之前我用 `trai""ner.py` 这种"字符串拆分"绕过，但那只在**模式不像命令全文**时有效；本次模式 `hbm_sample.sh` 出现在脚本正文里，拆分也救不了（且我这次没拆）；③ 更稳的原则是**不要用"杀进程"做清理**：给每轮分配唯一目录/标识，用文件标记（`stop_hbm`）让旧进程自己退出 —— 幂等、可审计、不会误伤；④ 这个错误的表现（rc=-1、无输出）**很容易被误读成"远端没响应/网络问题"**，我差点就去查会话状态了；实际上守护还活着、命令也执行了一半（`mkdir` 生效）——**"半执行"比"完全没执行"更容易骗人** | ① 启动脚本去掉 `pkill`，改唯一目录 + `stop_hbm` 文件标记；② 重跑后一切正常（`PROF_START out=…/run steps=100 window=[50,52] … official_comparable=False`、`P2PROF_OK`、`train 100/100`）；③ 顺带记下另一条与采样有关的边界：`npu-smi info` **单次调用就要十几秒** ⇒ 我"每 5 秒采一次"实际只得到 **7 个点/124 秒**，**采样间隔 >> 步时长（0.43–0.7 s）** ⇒ 该曲线**看不到步内峰值**，只能看趋势（要步内峰值必须用进程内 allocator 统计）；④ 立规矩：**任何"查询/清理"命令都要先问'它会不会匹配到我自己'**，且清理优先用标记文件而非杀进程 | ① 失败原文：`RSH_FAIL rc=-1 stderr_tail=''`，且 `ls -1d /root/ops/profmem2_*` 显示目录已建、`ls` 内容为空、无 `hbm.csv`；② 会话侧同时是健康的：`STATUS connected=True commands=55 reconnects=1`（⇒ 不是网络问题）；③ 重跑后：`HBM_SAMPLER pid=26086`、`LAUNCHED pid=26087`、`PROF_START … window=[50,52] … official_comparable=False`、`P2PROF_OK`；④ HBM 采样实况：`chip0: n=7 首=3124 末=8458 最小=3124 最大=8531 变化=+5334 MB`、`采样时长 = 124 秒`（暴露了采样粒度问题） |
| **208** | ★★★ **profile 的"第三层"（调用栈）把结论整个翻了一次：最大支出不是算子、也不是通信，而是 `wait_event` 等待/同步；而我一直盯着的"重计算"只有 0.78%**。with-stack 那一轮的 `operator_details.csv`（48.7 MB，含 `Call Stack` 列）按 `Device Self Duration(us)` 聚合（合计 1198465 µs）：**`wait_event` 类目合计 ≈46.6%**（33.47% 出现在 backward 栈、11.46% 在其它栈、1.66% 在通信栈）、`aclnnMatmul` 11.60%、GDN 反向核 6.18%、通信 5.19%、优化器 1.22%、**重计算 0.78%**（`recompute_w_u_fwd_kernel` 0.67%）。也就是说：① **重算几乎不存在** ⇒ "按模块重计算"在**时间**维度上没有收益空间；② 真正的大头是**等待/同步**，而它有三个独立旁证：无插桩轮次的 `Free 19.3%`、`7150 次 kernel 发射/步`、以及小算子/拷贝/cast 簇 16.3% | ① **每一层都改变了判断**：桶层说"计算占 78%"，算子层说"GDN 族 28% 最大"，栈层说"等待占 46.6%、重算 0.78%" —— 如果我只做到前两层就动手（很可能去"优化 GDN"或"做重计算"），方向就是错的。**分层的价值恰恰在于它会推翻上一层的读法**；② `wait_event` 这类**同步原语**在任何"按算子名排序"的榜单里都不显眼（它甚至不像"计算"），但它是设备侧最大项 ⇒ **"算子榜"天然偏向算术算子**，会系统性低估同步/调度；③ 这条也解释了为什么我早先那两份"看起来很有指导性"的比例（Computing 78%、GDN 28%）**都没有直接可用性**；④ 诚实边界：`wait_event` 可能被插桩本身放大（追踪让主机变慢 ⇒ 流等待变多），所以**方向成立、数值不可当精确值**——这一点必须写进产物，否则下一轮有人会拿 46.6% 当结论 | ① 把栈维度归因写进 `protocols/profile_short_20260922.json` 的 `stack_round_2026_09_22`：按标记的占比、Top 算子、四条 findings、三条 caveats、**修正后的方向**（等待/同步/发射侧优先；显存侧改口径为"动机=显存"；**排除**通信与重计算）；② 明确 **chunk loss 不得宣称省时间**（重算 0.78%）——只能作为**显存**手段立项；③ 记录叶子帧解析失败与其真实格式（`<file>(<line>): <func>;\n<file>(<line>): <func>`，分隔符是**字面反斜杠 n**）⇒ 下一轮重跑一次即可拿到叶子帧归因；④ 立规矩：**归因至少要做到"能区分算术 / 同步 / 通信 / 调度"这一层**，只看算子榜不够 | ① 聚合输出：`算子行数=45254 设备自耗时合计=1198465 us`、`[backward] wait_event 401147 us 33.47%`、`[其它] aclnnMatmul 139066 us 11.60%`、`[recompute] recompute_w_u_fwd_kernel 8027 us 0.67%`；② 栈格式原文：`/usr/local/lib64/python3.11/site-packages/torch/_tensor.py(1128): __format__;\n/root/MindSpeed-MM/mindspeed_mm/fsdp/train/train_engine.py(350): training_log;…`；③ 同轮桶（**不可与前两轮对比**）：`Computing 33.6 / Free 65.1 / Communication 2.5 / NotOverlapped 1.3`；④ 产物：`operator_details.csv` 48.7 MB、`trace_view.json` 149.8 MB、`P2PROF_OK`、`train 100/100` |
| **209** | ★★★ **我差点拿"今天的步时长"和"昨天的步时长"直接对比——而这是两台不同的机器**。预取实验的基线档（预取 1，配置与昨天的 A 臂**逐键相同**：`qwen3_5_0_8B_mock_dp2_v3.yaml`、GBS=8）今天实测中位步时长 **731.6 ms**；而昨天同一配置的 A 臂是 **419.03 ms**——相差近 **1.75×**。若不做核对，我完全可能写下"今天变慢了"或"昨天的收益被推翻"这类**结论**。核对后的事实：今天的运行在 **`199.98.58.213`（host `4ac9ca3712ba`，npu-smi 26.1.1，torch_npu 2.7.1.post10）**，昨天那批在 **`199.103.55.150`（CANN 9.1.0-beta.3）**——**换了 IP、也是另一台机器**（只是 `/root` 下磁盘内容相同，容易让人以为"同一台"）。另有一处独立线索佐证"环境不同"：今天的 `triton=3.2.0`、`npu-smi 26.1.1`，与昨天记录的版本不一致 | ① **"同一份磁盘"被我用成了"同一台机器"的证据**：`/root/ops`、Skill、`MindSpeed-MM` 全在，于是我在心里把新 IP 当成了"换了个地址的老环境"——**磁盘内容的一致性与硬件/驱动的一致性是两个独立性质**（与坑 169「机器是平台的对象、不是我的资源」同族，但这次是我自己把两者混为一谈）；② 更普遍：**绝对时间是最容易被误用的量**——它同时受硬件、驱动、CANN、频率、邻居负载影响，而**配置只是其中一个因子**；本项目的所有协议都要求"同机同批次"，我却在心里默默做了跨机比较；③ 这一条也回过头解释了另一个疑点：我不必再为"为什么今天慢"找配置上的原因（那会浪费卡时去做无意义的 A/B）；④ 幸运的是**本轮实验本身没被污染**：3 档 × 2 轮是在**同一台机器、同一批次**里顺序完成的，所以 2.40% / 3.91% 这两个**相对值**仍然有效（判据要求的正是相对比较）| ① 把这条写进 `protocols/prefetch_depth_20260922.json` 的 `measurement_integrity_caveat`：**跨机器绝对时间一律作废**，只有同会话同机比较有效；② 明确记录"今天基线 ≈731.6 ms vs 昨天 419.03 ms 差 1.75×，只能解释为机器/驱动差异"；③ 立规矩：**任何跨批次/跨机器的数字对照前，先核对"主机名 + npu-smi 版本 + CANN 版本 + boot id"四项**（本轮的 profile 产物里恰好有 `identity.boot/hostname`，是现成可用的判据）；④ 后续所有收益陈述都带"同机同批次"限定语 | ① 今天：`v1_r1 734.0 / v1_r2 729.2`（中位 731.6 ms，配置与 A 臂逐键相同）；② 昨天归档：A 臂 `419.03 ms`、B 臂 `521.23 ms`（`AB_AVSB_20260921`）；③ 机器证据：`profile_short_20260922.json` 的 `run.boot_id=fe6372a8-974e-481a-9c6a-828b3f4eb3ba`、`hostname=4ac9ca3712ba`、`npu-smi 26.1.1`、`triton 3.2.0`，对比昨天的 `199.103.55.150` / CANN 9.1.0-beta.3；④ 本轮相对值仍有效：`预取 4 省 3.91%（逐轮 4.56/3.26，同向）`、`预取 2 省 2.40%`，低于 5% 获胜线 ⇒ 记为无可报告收益 |
| **210** | ★★★ **我把 Python 的"相邻字符串拼接"写进了 JSON，于是那份协议**根本无法解析**——而 23 个闸门没有一个会因此报错，它连过两次打包**。事故：写 `protocols/prefetch_depth_20260922.json` 时，某个数组元素我分两行写（第 7 行以 `其中 "` 结尾、第 8 行另起 `"**\`reduce_scatter_tensor\` 17.45%**…"`）——在 Python 里这是合法的隐式拼接，**JSON 里没有这个语法**。文件就这样进了 Skill 并被 `safe_pack_sync --apply` 打包（条目数 118 → 119 都正常）。**发现它的方式很偶然**：我随后写 `_patch_prefetch_results.py` 去读它补结果，`json.load` 直接抛 `Expecting ',' delimiter: line 8 column 3`。也就是说：如果我不需要读它，这个坏文件会**一直留在交付包里** | ① **"写出去的东西"和"能读回来的东西"是两件事**：我写 JSON 时用的是"人类可读的排版习惯"（跨行、缩进、拼接），而机器读的时候只认严格语法 —— 这和坑 189（改的是文件、印的是旧变量）、坑 203（结果当前提）同族：**我总是默认"我看着对，它就是对的"**；② 更关键的是**闸门没有覆盖"可解析性"这一层**：G1 编译 .py、G23 静态扫描、G28 坑表、G30 文档数字、G7 PDF…… 但**没有任何一条检查"交付物里机器要读的文件能不能被读出来"**；③ 这暴露了闸门设计的一个盲区：我的闸门多数是"检查内容对不对"，而**缺少"检查载体是否可用"**（前者昂贵、后者几乎零成本，却被我忽略了）；④ 附带教训：这次是**格式错**而非**语义错**，所以自检/审阅都不容易发现（人眼读 JSON 拼接很自然），`json.load` 却一秒钟就能抓 | ① 整份重写 `prefetch_depth_20260922.json`（所有字符串单行、无拼接），并把本轮结果一并写入；② 新增自动闸门 **G31：「协议 JSON 可解析性 + schema 键」**——遍历 `protocols/*.json`，逐个 `json.load`，任一解析失败或缺 `schema` 即 FAIL；零样本也 FAIL（不许"没有文件所以通过"）；③ 立规矩：**凡是机器要读的交付物，都要有一条"读得动"的闸门**（不只 JSON，将来若加入 CSV/YAML 同理）；④ 自指说明：G31 自身计入闸门总数 ⇒ 文档里的数字必须同步为 24（G30 会检查这件事） | ① 报错原文：`json.decoder.JSONDecodeError: Expecting ',' delimiter: line 8 column 3 (char 407)`（来自 `_patch_prefetch_results.py` 的 `json.load`）；② 坏文件第 7–8 行：`  "第三轮（with-stack）栈归因… 其中 "` + `  "**\`reduce_scatter_tensor\` 17.45%**、叶子帧 \`wait\` 16.60% ⇒ …`；③ 它曾被两次打包含进包里：`SAFE_PACK_SYNC_OK`（条目 118→119）、`SAFE_PACK_VERIFY_OK items=119`；④ 修复后：整份重写、`json.load` 通过（并入 results 与 next_with_remaining_budget）；⑤ 新增 G31 后闸门总数 23→**24**，G30 会要求文档数字同步 |
| **211** | ★★★ **我上一轮报的"预取 4 省 3.91%"没有复现——而根因不是预取，是**噪声底噪约 2%**，我之前根本没测过它**。同一台机器、同一会话、同一配置（`qwen3_5_0_8B_mock_dp2_v3.yaml`）的两批基线中位数是 **731.6 ms（批次3）vs 716.5 ms（批次4）**，也就是**批次间漂移 ≈2%**。而上一轮我据以宣布"预取 4 省 3.91%（两轮同向）"的效应量，正好落在这个漂移的 2 倍以内。本轮把四臂放进**同一批次**再跑（a=基线 / b=预取4 / c=`log_interval=10` / d=组合，各 2 轮，8/8 rc=0）：**b = −0.76%（方向不一致 ⇒ UNCERTAIN）**、c = −2.33%（稳定**更慢**）、d = −2.62%（**更慢**）⇒ **没有任何一臂优于基线** | ① **我跳过了"先测量噪声"这一步**：任何性能实验的第一件事应该是**在同一批次里重复基线**，用它的离散度定出可报告的最小效应；我却先跑了 3 档、看到 3.91% 就当成信号（"两轮同向"给了我虚假的信心——**两个同向样本也可能是漂移而非效应**）；② 这也解释了为什么 Codex 一开始就把**门槛定在 5%**、"不许用 max(5%, 噪声范围) 当阈值"——**5% 之所以合理，正是因为它高于这类设置的噪声底噪**，而我把这条纪律当成了"形式要求"，直到今天自己测出 2% 才真正理解；③ 另一个层次：**跨批次比较和批内比较是两种证据**，我之前那句"3.91% 是同批次内 3 档 × 2 轮"其实成立，但"同一批次"只保证**实验条件一致**，**不保证批次本身的漂移被消除**——需要的是**批内交替**（ABAB）或更多重复，而不是"一次性把 3 档跑完"；④ 反向收获：`log_interval=10` 反而更慢 2.33%（两轮同向）⇒ with-stack 抓到的"每步 `_tensor.py: __format__ → training_log`"**不是**那 46.6% `wait_event` 的主要来源，**该线索作废**（避免了一次基于错误线索的代码改动） | ① 把四臂结果 + **噪声底噪结论**写进 `protocols/prefetch_depth_20260922.json` 的 `batch4_2026_09_22`，并新增规则：**任何小于噪声底噪 2 倍的效应一律先按 UNCERTAIN 处理**，要报小效应必须增加重复数并给区间；② 明确把"开关类候选（预取深度、`log_interval`）"**全部标记为已排除**；③ 作废 `log10` 那条同步线索（写进 `other_findings.log10_slower`）；④ 现实建议写进协议：把**已有的 A vs B 组合证据（19.6%，同批次同机）**作为性能主结论，新开关探索记为"已排查、均低于噪声底噪"；性能验收口径的关键仍是**官方数据**，不是再薅 1–2%；⑤ 立规矩：**任何性能实验的第一步是"在同批次内重复基线，先量噪声"**（已可作为 Skill 里的固定步骤） | ① 批次3（预取 3 档 × 2 轮）基线：`v1_r1 734.0 / v1_r2 729.2` ⇒ 中位 **731.6 ms**；批次4 基线：`a_r1 707.7 / a_r2 725.4` ⇒ 中位 **716.5 ms**（同机、同会话、同配置）⇒ **漂移 ≈2%**；② 批次4 相对基线：`预取4 −0.76%（逐轮 −1.72 / +0.21 ⇒ 方向不一致）`、`log_interval=10 −2.33%（−1.09 / −3.57）`、`组合 −2.62%（−3.29 / −1.94）`；③ 8 轮全 `rc=0`，日志已拉回本地（16 文件双侧 SHA256）并被分析器解析；④ 上一轮原话（已作废的结论）：`预取 4：中位数 703.0 ms，省时 3.91%（逐轮 ['4.56', '3.26']）⇒ 稳定` |
| **212** | ★★ **"确定性开关"的归因我猜错了一半：单独放开两个 NPU 环境变量**都不足以**恢复重复性**。已知事实：`use_deter_comp=true` 时同臂 **0/100 超阈**、checkpoint **473/473 逐位相同**（归档 P2）；`false` 时 **90–93/100 超阈**（P0）。该开关最终调用 `set_deterministic_algorithms()`，它**一次设三样**：`os.environ['HCCL_DETERMINISTIC']='True'`、`os.environ['CLOSE_MATMUL_K_SHIFT']='1'`、`torch.use_deterministic_algorithms(True)`。我把前两个拆出来单测（启动前 export，并在启动行打印实际值确认进了进程环境）：**v1 仅 HCCL = 187/200 超阈（未满足）**、**v2 仅关 K-shift = 186/200（未满足）**、**对照两个都不设 = 186/200（未满足）** ⇒ 三者不可区分 | ① 我此前在口头与文档里把这条路径讲成"`use_deter_comp` 设了 `HCCL_DETERMINISTIC` 与 `CLOSE_MATMUL_K_SHIFT`"——**描述没错，但推论错了**：我默认"这两个 env 就是机制"，于是把方向指向"反向的 matmul/集合通信"；实测说明**该推论没有依据**（至少**单独 export** 无效）；② 这也修正了上一轮一步回放的解读：C1 相同、C3 不同确实把范围压到反向段，但"具体是 HCCL 还是 matmul K-shift"**不能**由那段证据推出——我又一次用"机制上说得通"替代了"机制已被隔离验证"；③ 诚实的边界必须写明：本实验只证明**单独 export 无效**，不能证明这两个变量**本身**无用（框架可能在更晚时机或以其它名字传递它们；我没做传递链取证）；④ 方法论收获：**"拆开一个复合开关"要拆到底**——三个机制里我只测了两个，剩下那个（torch 分派层开关）才是嫌疑最大的，而它需要另一种注入手段（`sitecustomize` 于解释器启动时设置） | ① 把负结果写进新协议 `protocols/determinism_isolation_20260922.json`（口径复用 numeric_diag：窗口 1..100、逐点 2%、同臂配对；对照用当日批次4 基线两轮）；② 明确写下边界"只证明单独 export 无效"；③ 规划下一步：用 `sitecustomize.py` 只设 `torch.use_deterministic_algorithms(True)`，**并加"三者全设"的正对照**（预期复现 0/100）以验证注入手段本身有效——**没有正对照就不能说"这个机制无效"**；④ 立规矩：**复合开关的拆解必须列出全部成员并逐个单测 + 一个全设正对照**，不许只测其中两个就下结论 | ① 三臂判定原文：`v1（仅 HCCL_DETERMINISTIC）重复性要求：未满足；同臂 超阈 187/200`、`v2（仅 CLOSE_MATMUL_K_SHIFT）… 186/200`、`对照（都不设）… 186/200`；② 环境生效证据：启动行 `RUN_START arm=v1 r=1 HCCL_DETERMINISTIC=True 12:45:37 HCCL=True KSHIFT=unset`；③ 4 轮全部 `train_rc=0`，日志已拉回本地（`远端证据_20260922/detiso/`，8 文件 OK、4 文件因 `launch.out` 不存在而跳过——该跳过已如实报出）；④ 机制定义原文：`random.py:27 os.environ['HCCL_DETERMINISTIC'] = 'True'`、`:28 os.environ['CLOSE_MATMUL_K_SHIFT'] = '1'`、`:29 torch.use_deterministic_algorithms(True)` |
| **213** | ★★★ **机制隔离成功：关键开关是 `torch.use_deterministic_algorithms(True)`，而两个 NPU 环境变量单独设置完全无效——我上一轮把机制猜错了**。承接坑 212（v1 仅 `HCCL_DETERMINISTIC`=187/200、v2 仅 `CLOSE_MATMUL_K_SHIFT`=186/200、对照 186/200，三者不可区分），本轮用 `sitecustomize.py` 在**解释器启动时**只注入 `torch.use_deterministic_algorithms(True)`（**不设任何 env**）：**同臂 0/200 超阈 ⇒ 满足**；同时加**三者全设的正对照**：同样 0/200。并且**注入本身自证生效**：4 轮日志里 `DETSC active = 3`（torchrun 主进程 + 2 个 rank worker 各一行）、`DETSC FAILED = 0`。⇒ 结论：**PyTorch 分派层的确定性开关**才是恢复重复性的关键；"改善来自 HCCL 确定性 / matmul K-shift"这条推论**被证伪**（至少对"启动前 export"这种设置方式） | ① **我上一轮的错误不是测量错，而是"用机制上说得通代替机制已被隔离"**：`random.py` 里那三行紧挨着，我读到前两行就认定它们是机制——**复合开关必须拆到底、逐个单测**（坑 212 已把这条立成规矩，本轮才真正执行）；② 这次能下结论，靠的是**两条判据同时到位**：`DETSC active=3` 证明"注入真的发生了"（否则"有效"可能是注入失败/静默无效造成的假象），以及**三者全设的正对照**也复现 0/200（证明整套手段与归档 P2 结论一致）；③ 机制上顺手闭合了一个旧疑点：PyTorch 确定性模式会**改变算子分派**（选确定性实现）⇒ 这正解释了为什么 `use_deter_comp=true` 时**第 1 步 loss 就不同**（0.1154926 → 0.1169518，前向被换成了另一组实现），也解释了为什么它**只能作诊断**；④ 与一步回放相容：不加开关时 C1 前向逐字节相同、C3 通信后梯度不同 ⇒ 非确定性在**反向**；加了开关后前向实现也变但仍可重复 ⇒ 两条证据拼在一起才完整 | ① 把 v3/v4 结果并入 `protocols/determinism_isolation_20260922.json`（含注入自证、正对照有效性、机制结论、三条边界）；② 明确写出**边界**：只覆盖"启动前 export 无效 / 解释器启动时注入 torch 开关有效"两种方式，**不能**据此说"NPU 环境变量在框架内部也无用"（框架里三者是一次调用设置的，时序/传递链未取证）；③ 立规矩（已生效）：**复合开关拆解必须列出全部成员 + 逐个单测 + 一个全设正对照**；④ 把"注入是否生效"当作**判据的一部分**：任何"某机制无效/有效"的结论都必须附"该机制确实被施加了"的证据 | ① 判定原文：`v3（仅 torch.use_deterministic_algorithms(True)）重复性要求：满足；同臂 超阈 0/200`、`v4（三者全设，正对照）… 满足 0/200`；② 注入自证：`v3_r1: DETSC active = 3，DETSC FAILED = 0`（四轮皆同）；③ 对照（上一轮）：`v1 仅 HCCL 187/200`、`v2 仅 K-shift 186/200`、`都不设 186/200`；④ 注入实现：`sitecustomize.py` 中 `torch.use_deterministic_algorithms(True)` + `sys.stderr.write("DETSC active … pid=%d")`，经 `PYTHONPATH=<inject>:...` 生效；⑤ 4 轮 `train_rc=0`，日志 8 文件双侧 SHA256 已拉回 `远端证据_20260922/detiso3/` |
| **214** | ★★ **6 个闸门在新会话里"集体变红"，我差点把它当成交付物回归——真因是本地沙箱把 `tempfile.mkdtemp()` 建的目录变成不可写**。新会话开场按交接 §6 跑 `prepush_check.py`，得到 **PASS=18 FAIL=6 ⇒ `RELEASE_BLOCKED`**（G2/G3/G9/G24/G5/G6），而交接文件明写"24/24 PASS"。逐个手工复跑失败的闸门，6 个的 stderr **全是同一个 `PermissionError`**，且都指向 `%TEMP%\dsh-*\` 下由 `tempfile.mkdtemp()` 新建的子目录（如 `tmp12kz27lg\output_llava_coco_data.json`、`p59orch_wu8xdzco\ab`）。做成对照实验后定位到**唯一变量是目录的 mode**：`os.mkdir(p, 0o700)` ⇒ 建目录成功、**往里写文件必 `PermissionError`**；`os.mkdir(p)`／`0o755`／`0o777` ⇒ 正常；PowerShell `New-Item` 建的目录 ⇒ 也正常。而 `tempfile.mkdtemp()` 内部**恰好**就是 `_os.mkdir(file, 0o700)` ⇒ 命中率 100% | ① **"6 个互不相干的闸门同时变红"本身就是"共同根因"的信号**，而我的第一反应是"交付物回归了"——这是**把环境问题误判成自己的东西坏了**（与坑 209「把"磁盘内容相同"当成"同一台机器"」同族：都是**未核验就归因**）；② 我的第一个假设（"沙箱 TEMP 不可写"）**是错的**：PowerShell 往 `%TEMP%` 里直接写文件**成功**；把 `TEMP` 整体重定向到工作区后 6 个闸门**依然全红** ⇒ 假设是被**实验**推翻的，不是被我"再想想"推翻的；③ 真正的教训是**假设必须一直拆到"变量唯一"**：只有把 mode 单独拎出来做四组对照，才得到"`0o700` 才有病"这个**可判定**的结论；④ 这同时解释了"昨天全绿、今天全红"——**同一份交付物、不同的宿主策略**，与宿主相关的红**不能记到交付物账上**，否则会去"修"一个没坏的东西 | ① 用**非侵入式**垫片把 `tempfile.mkdtemp` 换成"默认 mode"实现（`工具_重建\_tmp_gatecheck\shim\sitecustomize.py`，经 `PYTHONPATH` 注入到每个子进程）⇒ 闸门立刻恢复全绿，**未改动任何交付物文件**（改文件 ≠ 改交付物，此处的正解是**两个都不改**，只改宿主侧的调用环境）；② 本坑入表——注意与坑 213 区分：213 的 `sitecustomize.py` 是**注入被测机制**（torch 确定性开关），214 这个是**修宿主文件系统行为**，**用途不同、不得互相套用**；③ 立规矩：**新会话开场若"闸门数与交接不符"，先手工复跑那几个失败的闸门并读 stderr 找"共同签名"，再决定是查交付物还是查宿主**；不许直接按"交付物回归"处理，更不许为了变绿去动交付物 | ① 对照实验（同一父目录、只变 mode；Python 3.13.12 / Windows）：`0o700 → write inside FAIL PermissionError`、`0o777 → OK`、`0o755 → OK`、`default → OK`、`PowerShell 建的目录 → python 写入 OK`；② 失败签名（同一根因的证据）：`G9 → PermissionError: [...\tmp12kz27lg\output_llava_coco_data.json]`、`G24 → PermissionError: [WinError 5] 拒绝访问。: [...\p59orch_wu8xdzco\ab]`；③ 垫片生效证据：`DSH_MKDTEMP_SHIM=1` 且 `mkdtemp → write inside OK`；④ 垫片下驱动器实测：`合计：现存 25 条（PASS=25 FAIL=0）`、`AUTOMATED_OK_MANUAL_OPEN`（与交接期望逐字一致） |
| **215** | ★ **我给交付物新增一个脚本，`py_compile` 与 42 例自检全绿，但 `--help` 直接崩——因为 argparse 会对 help 文本**再做一次** `%` 格式化**。新增 `62_reportability.py`（§4-4 产品化的硬闸门）后，`python -m py_compile` 干净、`--selftest` 报 `P62_SELFTEST_OK cases=42 failed=0`，看似没问题；但 `prepush_check.py` 的 **G1（编译 + 加载冒烟）判 `G1_FAIL`**。手工复跑 `python scripts/62_reportability.py --help` 复现：`TypeError: not enough arguments for format string`。根因：我为了把默认值写进帮助文本，写成 `"效应下限（%%，默认 %.1f：纪律「<5%% 的改善不可报告」）" % DEFAULT_MIN_EFFECT_PCT` —— **在源码里先做了一次 `%` 格式化**，结果留下一个**裸 `%`**；而 argparse 的 `HelpFormatter` 会**再对 help 字符串做一次 `%` 格式化**，裸 `%` 让它去找一个不存在的格式参数 ⇒ 直接抛异常 | ① **"编译通过 + 自检通过"完全不代表"能被加载"**：`py_compile` 只看语法；而我的 `--selftest` 走的是 `main()` 里**靠前**的分支（`if args.selftest: return selftest()` 之前根本不会构造 `ArgumentParser`）⇒ **42 例全绿也照不到这一行**；这与坑 192「过程正确、结论正确、但自检选错了断言对象」同族，只是这次的盲区是"**入口构造**"而不是"汇总层"；② 真正拦住它的是 **G1 的 `--help` 冒烟**，而 G1 存在的理由（坑 173："`--help` 会真正执行模块顶层代码"）正是这个场景——**G1 本轮又证明了一次自己**；③ 更一般的教训：**往受闸门管的目录里新增文件，等于自愿接受一整套更强的一致性要求**——新增 `.py` 会被 G1（冒烟）/G23（pyflakes）扫，新增文档会被 G30（状态数字）扫；"加完就完"是错觉 | ① 把三个 help 字符串改成**字符串拼接**（不在源码里预先 `%` 格式化），并把文本里的字面百分号写成 `%%`（argparse 会还原成 `%`）；② 在代码注释里写明这条陷阱，避免重犯；③ 复核：`--help` 正常打印 usage 全表、`G1_OK`、`G32 → P62_SELFTEST_OK cases=42 failed=0`、`合计：现存 25 条（PASS=25 FAIL=0）`；④ 本条**不入**"致命教训"级别（它被既有闸门在交付前拦住了，没有污染任何产物），入库是为了**保住"G1 到底在防什么"这条知识** | ① 失败签名：驱动器输出 `FAIL  G1  编译 + 加载冒烟（--help 真正执行模块顶层代码；cwd=SKILL，坑 173） … 判据=G1_FAIL`；② 手工复跑栈：`File "...\scripts\62_reportability.py", line 610, in <module> sys.exit(main())` → `line 595, in main` → `args = ap.parse_args()`，异常为 `TypeError: not enough arguments for format string`；③ 修复前 `python scripts/62_reportability.py --help` 退出码非 0 且无 usage；④ 修复后同一命令正常打印 `usage: 62_reportability.py ...` 与 `--min-effect`（显示为 `默认 5.0%`）、`--noise-multiplier`、`--min-pairs` 三条帮助；⑤ 全量复跑：`合计：现存 25 条（PASS=25 FAIL=0）`、`AUTOMATED_OK_MANUAL_OPEN` |
| **216** | ★★ **我差点拿"远端部署副本"的判据去统计 A/B——而那份副本是旧版**。§4-2 端到端 A/B 的编排脚本原本在**远端** `import 59_config_ab`，复用 `PAT` / `win_ms_of` / `compute_valid` / `paired_stats`。后台启动后立刻崩：`AttributeError: module '59_config_ab' has no attribute 'win_ms_of'`。核对发现远端 `/root/qwen35-ascend-migrator/scripts/59_config_ab.py` 的 md5 是 `218e635b…`，与**交付副本不同版本**（交付副本里有 `win_ms_of`）。而真正危险的地方在于：`win_ms_of` 正是 `paired_stats` **内部**要用到的取值函数 ⇒ 若远端那版恰好"能跑通但取值口径不同"，我会在**毫无报错**的情况下用**别人版本的判据**算出性能数字，而产物上写的却是交付物的判据名 | ① **"同名文件"被当成了"同一份实现"**——这是本项目的老毛病（坑 163/176 族：「字段名是断言而不是事实」），但这次发生在**跨机器的一份副本**上，比同机同名更难看见；② 这次崩掉其实是**幸运**：`AttributeError` 是"响亮的失败"；真正危险的是"静默跑通、但语义不同"——**判据是不允许有版本的**（纪律 #8：单一来源）；③ 根因是我把"远端那份能跑训练的 Skill"默认当成了"交付物本身"。实际上远端那份是**部署副本**，它的角色是**跑**，不是**判**；④ 这与坑 209（把"磁盘内容相同"当成"同一台机器"）同族，都是**把"看起来一样"当成了"就是同一个"** | ① **重新划分工**：远端**只负责"跑 + 记录原始事实"**（rc / wall / 日志路径 / 行数），**一行判据都不在远端实现**；解析与统计**全部回本地**，用**交付副本**的 `PAT` / `compute_valid` / `paired_stats` 做；② 为此把编排脚本里所有 `59` 导入删掉，改为只写 `runs_index.json`（+ 每轮 `run_record.json`），并新增本地 `工具_重建/_ab62_analyze.py` 专门承担判据；③ 立规矩：**判据的本体只能来自交付副本**；远端脚本若要复用工具，**必须先核对 md5 与交付副本一致**，不一致就只允许它"跑"，不许它"判" | ① 失败签名：`File "/root/ops/_ab62_run.py", line 34, in <module> win_ms_of = _p59.win_ms_of` → `AttributeError: module '59_config_ab' has no attribute 'win_ms_of'`；② 版本证据：远端 `md5sum 59_config_ab.py` = `218e635b2ba6439aa46c79314a86f5b9`；远端探针逐名输出 `PAT OK / win_ms_of **缺失** / compute_valid OK / paired_stats OK / T975 OK / STEP1_ANCHOR OK`；③ 处置后实测：编排脚本不再 import `59`（只记录原始事实），本地分析用交付副本判据跑通，**9/9 轮 valid=True**、步号 1..100 恰好一次，噪声底噪 1.53% |
| **217** | ★★★ **我把"记录了 sha256"当成了"验证了臂状态"——于是整个 P2 A/B 批次 6 轮跑的都是同一个文件，而编排器还报 `verification_passed=True`**。事故链：P2 端到端确认实验（`p2ab_20260922_162640`，区组 N,C1,C1,N,N,C1）跑完后看 `status.json`：`verification_passed=true failures=[]`，6 轮 `rc=0`、`iter=100`、`restored_ok=true` —— **一切"绿"**。但我把每轮的 `arm_file_sha256` 并排一看：**6 轮全是 `fefdb197e98a`（=原文件）**，**连候选臂 `C1` 也是** ⇒ "候选 vs 基线"实际退化成 **"基线 vs 基线"**，整批实验**没有任何候选信息** | ① **更深的一层不是"忘了打补丁"，而是"记录 ≠ 断言"**：我在设计里特意要求"每轮记 sha256"（本意就是防这种事故），却**只记录、不断言** —— 记录要靠人眼看，断言会自己拦下来。整批的自动判据只挂在"进程 rc / 步数完整 / 还原一致"上，而这三项在"基线 vs 基线"下**当然全部通过** ⇒ **判据没区分力**（纪律 #4 的原话："若机制真的坏了，这个数会变吗？"——这里答案是"不变"）；② 直接原因是 `_p2_patch.py apply` 把 SKIP 条件写成"**备份已存在**就跳过"，而备份是先前数值复核时建的 ⇒ `apply` 打印 `P2_PATCH_SKIP` 并 `return 0`，`set_arm` **信了返回值**；这与坑 33/34「静默当通过」同族，只是这次"静默"发生在**编排脚本对被测物的状态假设**上；③ 这次能发现，靠的是"臂状态 sha"这个**设计时多留的一列**——若当时只记 rc/wall，这批"绿"会直接被我归档成"候选无收益" | ① `_p2_patch.py`：SKIP 判据从"备份在不在"改为"**目标文件当前是否已含补丁**"（与备份**无关**）；并**备份已存在时绝不用已改过的目标去覆盖它**（否则会销毁唯一的原始副本）；② `_p2ab_run.py::set_arm` 增**断言**：候选臂 `sha != orig_sha`、基线臂 `sha == orig_sha`，任一不成立即 **raise**（异常被接住 → 记 `failures` → 终止批次）；③ 作废批次**改名留档 + 写 `VOID_REASON.txt`**，**不删除、不篡改为成功**；④ 立规矩：**凡"按臂切换文件/配置"的实验，必须在每轮之前断言被测物状态，而不是只打印它** | ① 作废证据：`/root/ops/p2ab_20260922_162640/status.json` 里 6 行 `arm_file_sha256` 全为 `fefdb197e98a`，而 `verification_passed=true`；② 修复后同一实验（`p2ab_20260922_164204`）实测：`00_N/03_N/04_N` 的 sha = `fefdb197e98a`，`01_C1/02_C1/05_C1` 的 sha = **`591a08714cc0`** —— 两臂**各自一致且互不相同**；③ 断言代码：`set_arm()` 内 `if arm != "N" and now == orig_sha: raise RuntimeError(...)`；④ 框架两次均还原（`restored_ok=True`，当前 sha == 备份） |
| **218** | ★★ **我在同一天里连续 5 次把不可解析的 JSON 写进 `protocols/`——每次都靠 G31 拦下，但有一次已经打进了 zip 才被发现**。形态完全一致：在 JSON 字符串里用 **ASCII 双引号**做中文强调，例如 `"rule": "……不得以"占比大"为由动手。"`、`"evidence": "……不是"配了但没生效""`。写的时候读起来很自然，解析器在第 5 个字符处就断。特别值得记的是发现路径：这 5 次**没有一次**是我自己看出来的 —— 全部是 `prepush_check.py` 的 **G31（协议 JSON 可解析性）** 报 `G31_FAIL files=N bad=1` 才暴露；其中一次我已跑过 `safe_pack_sync.py --apply`（把它打进了交付 zip），是先打包、后复验才拦下 | ① **根因不是"手滑"，是"我在两种引号语义之间切换却没有切换书写约定"**：中文里引号用于强调，JSON 里引号是定界符；我一边写中文论述一边写 JSON，就默认了"引号可以随便用"；② 更值得记的是**为什么每次都要靠闸门**：`py_compile` 管不到 `.json`，而我在写文件后**没有**做"读完立刻解析"这一步 —— 这一步是**亚秒级**的，比一轮「打包→复验→定位→改→重打包」便宜 3 个数量级；③ 这也是 **G31 价值的再次确认**：它的来历正是"我把 Python 相邻字符串拼接写进 JSON，文件根本不可解析，却连过两次打包"—— **同一个错、同一个闸门、隔了一轮又犯 5 次**，说明"闸门兜住"不等于"人学会了"；④ 附带暴露同类失误：我**3 次引用了尚未创建的远端脚本**（`_rcmd_p2ab_rename.sh`、`_rcmd_mk_argdump.sh`、`_rcmd_probe_cores.sh`），每次都白跑一个往返 ⇒ 同属"**动作前不检查前置物**" | ① 立**书写约定**：在 JSON 值里做中文强调一律用「」，**禁用 ASCII 双引号**（本坑表与后续全部协议已按此改写）；② **写完立即解析**：任何 `.json` 落盘后当场执行 `python -c "import json,sys;json.load(open(sys.argv[1],encoding='utf-8-sig'))" <文件>`，**解析不过就不算写完**（把"可解析"变成**写入动作的一部分**，而不是等闸门）；③ 发**远端命令前**先确认脚本文件存在（`_rsh.py --file` 缺文件会抛 `FileNotFoundError`，白跑一往返）；④ 把"改完协议 → 立刻解析 → 再打包"写进移交文档 §2.2 的最小动作序列；⑤ 本坑自身的写入也被写入端拦过一次（我在"处置"列里嵌了多行代码块 ⇒ `FATAL --row 文件必须只有一行`）——**这正是坑 196 的形态**，写入端 fail-closed 生效，改单行后才落盘 | ① 失败签名（5 次同型）：`G31_FAIL files=9 bad=1`、`protocols bad = 1`，Python 侧 `json.JSONDecodeError: Expecting ',' delimiter: line N column M`；位置举例：`nonzero_lr_replay_20260922.json:32:101`、`p2_candidate_01_result_20260922.json:57:144`、`p2_candidate_card_02_20260922.json:13:148 / 41:58 / 68:59`、`codelevel_sync_candidates…:89:54`（非法值 `"max": 0.138 (rank0) / 2.84 (rank1)`）；② 已打包才发现的证据：`SAFE_PACK_SYNC_OK` 之后 `prepush_check.py` 才报 `G31_FAIL`；③ 修复后：`protocols bad = 0`、`G31_OK protocols=10`、四道校验全绿（`items=128 / 217 / 168`、`PASS=26 FAIL=0`）；④ 白跑往返的签名：`FileNotFoundError: ...\_rcmd_p2ab_rename.sh` / `_rcmd_mk_argdump.sh` / `_rcmd_probe_cores.sh`；⑤ 坑 196 复发的签名：`FATAL --row 文件必须只有一行（当前 5 行）` |
| **219** | ★★ **打包闸门只扫交付物，而"移交给队友"会把工具目录一起带走——于是真凭证被我的移交动作放进了移交夹**。用户要求"把需要的移交资料放到桌面一个文件夹"。我按清单拷贝了 Skill 活副本 / 工具链 / 远端证据 / SSH 框架后，做自查扫描 `jt_[0-9A-Fa-f]{8,}:[0-9A-Fa-f]{8,}`，命中 **8 个文件**：① 工具链里的 `_pit_198.txt`、`_pit_200.txt`、`_pits_198_199.txt`（坑表**原始行**文件，写成于脱敏机制之前）与 `_backup\` 下的坑表历史快照；② **`05_SSH框架_ops\_write_store.py` 里是一个 126 字符的完整 token**、`_patch_session_user.py` 里是一个 36 字符的 `jt_ID:SECRET`（两者都是把凭证**硬编码在脚本里**）。同时我核对了**交付物侧**：活副本与交付解包副本的坑表里**完整 token 命中 = 0**，只有 3 处 `jt_<REDACTED_ID>:<REDACTED_SECRET>` 占位符与 2 处截断/测试用假值（`jt_FFFF0000:`）⇒ **交付物本身没有凭证问题** | ① **闸门的覆盖面是按"对象"划的，而"移交"是另一个对象**：`safe_pack_sync.py` 的内容闸门（凭证/备份/空文件/BOM）扫的是**要被交付的那棵树**，它一直在正常工作（G22 绿灯就是证据）；但**移交夹 ⊃ 交付物**，多出来的工具目录、历史行文件、`_backup` 快照**不在任何闸门的扫描范围内**——这是"闸门管不到的地方"，与坑 195（派生量手写必然过期）同族：**同一个事实有两个副本时，总有一份没被管**；② 更值得记的是**凭证是怎么进的工具目录**：坑 180 之后 `_ops` 被"先清理后重建"，重建者是**从现成的连接命令里抄 token 写进脚本**的——**为了"能跑起来"而牺牲了"不含凭证"**，而这个取舍在当时没有被记下来；③ 我的失误在于**顺序**：先拷贝、后自查；正确顺序是**扫描模板先定、再拷**（或拷完立即扫，且把"0 命中"作为移交的前置条件）。④ 顺带确认了一件好事：**交付物侧的脱敏机制是有效的**（写入端 `TOKEN_RE` 自动脱敏 + 打包闸门复查），所以同一批 token 在交付物里是占位符、在工具脚本里才是明文——**问题不在脱敏机制，在扫描范围** | ① 移交夹内**删除**纯历史文件（`_backup\`、三个 `_pit_*.txt`）——它们对移交无用；② 对**保留下来的工具脚本**做**脱敏替换**（`jt_...` → `jt_<REDACTED_ID>:<REDACTED_SECRET>`），而不是删除工具（删了队友就没法用）；③ 全夹重扫，**"完整 token 命中 = 0" 作为移交的前置条件**（不通过就不算移交完成）；④ 在移交文档里**显式写明**：原始工作目录里那两个脚本仍是明文，请用户自行轮换凭证/清理；⑤ 立规矩：**凡生成"要交给别人的文件夹"，扫描范围必须覆盖整个文件夹，而不是只扫交付物**——因为闸门的对象是交付物，而风险的对象是"被带走的一切" | ① 扫描命中的 8 个文件（相对移交夹路径）：`03_工具链\_pits_198_199.txt`、`03_工具链\_pit_198.txt`、`03_工具链\_pit_200.txt`、`03_工具链\_backup\PITFALLS_坑表.md.20260922_105819.bak`、`03_工具链\_backup\skill_baks\PITFALLS_坑表.md.bak_redact_20260922_105953_1b579a`、`..._110111_4232a9`、`05_SSH框架_ops\_patch_session_user.py`(len=36)、`05_SSH框架_ops\_write_store.py`(**len=126**)；② 交付物侧对照：三份坑表（活副本 / 交付解包副本 / 移交副本）完整 token 命中均为 **0**，脱敏占位符 3 处；③ 处置后重扫：**完整 token 命中文件数 = 0**；④ 移交夹体积变化：`03_工具链` 11.4 MB → **1.2 MB**（删掉历史备份）；⑤ 同时记录了本次移交的复验值：`items=128 / 218 / 168`、`PASS=26 FAIL=0`、`protocols bad = 0` |


## 坑 30-43 的元教训（写进 SKILL 设计原则）

1. **"旧机器上搭起来的环境"会掩盖 Skill 的所有隐含依赖** —— 路径深度、解释器唯一性、包管理器、
   命令可用性，这四类依赖在旧环境里"恰好都对"，只有在干净环境才暴露。
   → 因此 **Skill 交付前必须在干净环境跑一次 `run_from_zero.sh`**，这是必要的验收步骤，不是可选项。
2. **"能通过"比"不能通过"更危险** —— 坑 35 是本次最严重的发现：一个消费陈旧产物的自检会给出
   **假阳性**，比直接失败更有害（它让人相信了没被证明的事）。
   → 判据必须绑定"**本次运行产出的产物**"，并显式声明哪些输入来自包内、哪些来自实跑。
3. **判据要区分 FAIL 与 SKIP** —— 把"因缺输入没检查"算成失败会淹没真实信号（坑 34）；
   反之把 SKIP 静默当通过则会造假（坑 33）。二者都必须**显式输出**。

