# 环境说明（ENV）

> 先给结论：本 POC 的三环都支持"环境够就真跑、不够就设计内降级"，**降级是预期行为，不是 bug**。
> 下面**先讲没卡的跑法**（初赛绝大多数人走这条），再讲有卡怎么升级。

| 环 | 真跑要啥 | 没就绪时预期输出（设计内降级） | 证明啥 |
|---|---|---|---|
| poc1 | 啥都不要，纯静态扫，任何机器都能跑 | 不降级；源码缺节点会如实报"未找到" | 静态定位待迁移节点 |
| poc2 | NPU 卡 + torch_npu | `backend: CPU-回退`，forward/shape 仍通 | 算子链路 shape 正确 |
| poc3 | 迁移落地 + torch≥2.4 + 权重/算力 | `[降级] …` + 预期 JSON，干净退出无 traceback | 场景推理链路成立 |

## 一、没 NPU 卡（初赛默认，先走这条）

只要一个装了 PyTorch 的环境就行，环境叫啥不重要（conda / venv 都可以）。poc1 只用标准库 `ast`，poc2 只用 torch 的 CPU 张量，poc3 走降级不依赖外部包——所以**无卡复现的最小依赖基本就是一个 torch**。

```bash
# 0. 先 cd 进仓库根（别站在家目录，否则报 No module / No such file）
cd <你的仓库路径>/QWEN3_5-MIGRATE-POC

# 1. 装 torch（版本用你已有的就行，我本地 2.2.2 跑通过无卡路径；环境里已有就跳过）
pip install torch

# 2. 验证包结构没坏
python -c "import mindspeed_mm.fsdp.models.qwen3_5 as m; print('PKG_OK', m.__file__)"
# 应打印 PKG_OK，且路径指向仓库内的 mindspeed_mm/...，不是 site-packages

# 3. 跑三环
python run_all.py

```

预期：环2 的 backend 行显示 `CPU-回退`，环3 显示 `[降级]`，最后 `[ALL DONE]`。这就够初赛交差了。

## 二、有 NPU 卡（复赛 / 想真跑 poc2 的轨A）

软件栈安装顺序（三者版本要配套，对照昇腾社区版本配套表）：NPU 驱动+固件 → CANN（Toolkit + 对应芯片 ops 包）→ PyTorch + torch_npu。

```bash
# 1. 确认卡可见
npu-smi info

# 2. 加载 CANN 环境变量
source /usr/local/Ascend/ascend-toolkit/set_env.sh

# 3. 装 torch_npu（版本对齐你的 torch）
pip install torch_npu

# 4. 验证
python -c "import torch, torch_npu; print(torch.npu.is_available())"  # 应输出 True

# 5. 跑 poc2
python poc2_npu_forward.py
# 输出中 backend 行应显示: Ascend NPU (torch_npu OK)

```

> 环3 真跑还要 torch≥2.4 + 权重 + qwen_vl_utils，详见 requirements.txt 的"复赛/真跑"块。