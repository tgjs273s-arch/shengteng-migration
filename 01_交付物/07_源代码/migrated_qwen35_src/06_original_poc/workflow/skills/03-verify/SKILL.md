---
name: qwen35-verify
description: 数值精度验证技能（贯穿T2-T7）。运行逐层数值对比，产出验证报告，判定通过/失败。
---

# 技能 03：数值精度验证

## 验证框架

项目内 `verify/compare.py`（由 dsh 按《DSH实战指南》§3.2 模板生成）。

## 双层验证

| 层 | 验证对象 | 工具 | 判据 |
|---|---|---|---|
| 数值层 | 迁移后每层算得对不对 | compare.py（逐层） | 余弦相似度 > 0.999 |
| 任务层 | 模型整体能力有无退化 | COCO 官方数据（T6） | 精度差异在官方允许范围 |

## 标准执行

```bash
# CPU 形状验证（本地可跑，只验形状与逻辑，不算迁移证据）
python verify/compare.py --backend cpu --stage <阶段名>

# NPU 真验证（昇腾环境，唯一有效证据）
source /usr/local/Ascend/ascend-toolkit/set_env.sh
python verify/compare.py --backend npu --stage <阶段名> 2>&1 | tee logs/<时间戳>_verify_<阶段>.log
```

## 结果判定

1. 逐层检查输出表：最大绝对误差 / 相对误差 / 余弦相似度；
2. **先核对 backend 行**——出现 `CPU-回退` 则本次结果作废，重查环境；
3. 全部层余弦 > 0.999 → 该阶段精度验证通过，结果 JSON 存入 `verify/` 并登记；
4. 有层不达标 → 输出"失败层清单 + 误差排序 + 最可能原因"，回流技能 02 修复。

## 混合精度注意

bf16/fp16 下阈值可放宽至 0.995，但必须在报告中注明精度模式与放宽依据；关键层（GDN 状态更新）坚持 0.999。

## 日报支持

headless 每日体检时，汇总当日 `logs/` 中所有验证日志：通过项 / 失败项 / 待人工决策项 → `reports/daily_check.md`。
