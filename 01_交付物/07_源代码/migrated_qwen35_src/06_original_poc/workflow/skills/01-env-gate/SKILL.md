---
name: qwen35-env-gate
description: 昇腾环境自检与版本配套门禁技能（T0阶段）。在昇腾910B环境执行，产出环境报告。
---

# 技能 01：环境自检与版本门禁（T0）

## 触发时机

算力环境首次就绪时执行一次；环境有任何变更（升级/重装）后重跑。

## 执行步骤

1. **卡可见检查**
   ```bash
   npu-smi info
   ```
   通过标准：列出 910B，无 ECC/温度告警。

2. **CANN 环境激活**
   ```bash
   source /usr/local/Ascend/ascend-toolkit/set_env.sh
   ```
   通过标准：无报错，`echo $ASCEND_TOOLKIT_HOME` 有值。

3. **torch_npu 可用性**
   ```bash
   python -c "import torch, torch_npu; print('npu_available:', torch.npu.is_available()); print('device_count:', torch.npu.device_count())"
   ```
   通过标准：`npu_available: True`。

4. **版本配套核对**
   ```bash
   pip show torch torch_npu | grep -E "^(Name|Version)"
   cat /usr/local/Ascend/ascend-toolkit/latest/version.cfg 2>/dev/null || npu-smi info -t product
   ```
   与官方版本配套表逐行核对（配套表位置见 rules/ascend-facts.md），**不一致即停止，先修版本**。

5. **最小算子冒烟**
   ```bash
   python -c "
   import torch, torch_npu
   x = torch.randn(4, 8).npu()
   y = (x @ x.T).relu()
   print('matmul+relu ok, device:', y.device)"
   ```
   通过标准：输出 `device: npu:0`（不是 CPU 回退）。

## 产物

- `reports/env_report.md`：五项检查结果表 + 版本配套核对截图引用 + 结论（PASS/FAIL）
- `logs/T0_env_<时间戳>.log`：全程日志

## 门禁

五项全过 → 进入 T1；任何一项失败 → 输出诊断建议（参考官方安装指南章节），不进入下一阶段。
