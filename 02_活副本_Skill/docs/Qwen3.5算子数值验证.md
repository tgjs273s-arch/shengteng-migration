# Qwen3.5 目标算子数值验证（T04）

本入口是工程诊断，不是官方训练 loss 或业务精度验收。固定目标为 MindSpeed-MM `5b5505331924634da64e3d9a1925d02b10babe9f`，输入 GPU 源身份、迁移 ID 和目标源码摘要从 T03 `migration_manifest.json` 取得。0.8B 参考配置选择 Triton GDN、Triton causal conv、MTP=0；4B 示例中的 AscendC 不是这里的已测后端。

从 Skill 根目录运行（使用已安装完整依赖的固定目标 checkout；以下路径按实际 QA 产物替换）：

```bash
python3 scripts/30_verify_ops.py --env out/probe/env.json \
  --migration-manifest /path/to/migration_manifest.json \
  --migration-bundle /path/to/fixed/bundle \
  --target-checkout /path/to/MindSpeed-MM \
  --out /path/to/isolated/verify
```

三个新参数必须成组提供。未提供时保留旧 P3 四行独立 forward/shape 检查和 GDN `contract-only`。提供时直接复用 T03 `validate_overlay(bundle, overlay)` 与 `verify_target_checkout(checkout, require_patched=True)`，重新计算固定输入、converter 补丁、manifest ID、目标 HEAD 和唯一跟踪差异；不信任 manifest 自报的哈希。随后从该 checkout 实际导入目标模型与 Triton 后端；后端必须是该提交中的未修改跟踪文件。原有 `matrix`、`summary`、`forward_ok`、`contract-only`、`error`、退出码及 `--env/--out` 保持兼容。新行给出具体范围和身份。身份错误记 `error`；缺 PyTorch/完整目标依赖的项目记 `contract-only`，汇总为 `PARTIAL` 或 `CONTRACT_ONLY`，不得视为数值通过。

CPU 依赖就绪时，额外通过真实固定目标 `Qwen3_5TextModel` 运行 eager prefill 和单 token decode/cache，记录每段形状、有限值及实际函数来源。这只验证目标类的 CPU 合同，不证明 NPU 数值对照。CPU 类导入或执行失败会保留失败子阶段和先前已经完成的指标。

NPU 定向检查使用固定 0.8B 元数据的头数、维度、小序列和确定性随机输入，不下载权重或启动训练：

- `target_gdn_triton_prefill`：真实 `ops/gdn/chunk_gated_delta_rule.py` 对比目标模型自带 Torch 参考，BF16 q/k/v、模型相同的 q/k L2 norm，长度 63/64/65；比较输出、末状态和 q/k/v/g/beta 梯度。真实候选按参考配置 `skip_gdn_recompute=True`；eager 参考实现无此开关。
- `target_causal_conv_triton_prefill`：真实 `models/qwen3_5/causal_conv1d.py` 对比 PyTorch depthwise Conv1d+SiLU，长度 1/4/65；比较输出、输入与权重梯度。目标 wrapper 在 arch35 会明确拒绝，原始错误保留。
- `target_text_model_prefill_decode_cache`：同权重的真实目标小文本模型，eager 与 Triton 配置分别运行 prefill 和单 token decode；分别比较隐藏状态、卷积缓存与 recurrent 缓存，并记录实际绑定的 chunk、conv、recurrent、update 函数。该项覆盖目标类调用路径，但小模型不代表完整 0.8B 权重重载或生成质量。

BF16 比较预先使用 `atol=0.05, rtol=0.05` 的逐元素组合阈值，记录最大绝对误差、非零参考相对误差、最大差异位置及该处两侧原值。每项真实执行先将输入、目标小模型初始化权重写入隔离 `--out/target_numeric/`，随后保存已经产生的候选/参考输出和梯度。张量以无 pickle 的 float32 小端原始字节保存；`ops_matrix.json` 的 `artifacts` 列表记录原始 dtype、形状、文件路径和 SHA-256。失败行保留已完成指标、已写文件、失败子阶段、seed、dtype、形状、输入 token/生成公式、Torch 版本与实际导入路径。写文件失败记 `error`，不能成为通过。此为诊断阈值，不是官方精度规则；如测试失败，调查来源后另定版本化规则，不在失败时放宽。NPU 未就绪、完整框架不可导入或固定输入不匹配时不得填写 `numeric_pass`。

目标源码的 decode 分支目前有待核实的布局问题：prefill 缓存 `[B,C,K]`，但单 token 输入 `mixed_qkv` 为 `[B,1,C]`，直接传给按 `[B,C,T]` 处理的 `torch_causal_conv1d_update`。真实类用例会将该错误留作 `error`。修改目标源码会改变 T03 迁移身份，需先由主管协调补丁与重放；不可在 P3 中转置输入以掩盖该路径。

当前只完成实现；共享 QA 尚未执行新测试，NPU 对照与真实权重模型运行均未验证。完整依赖、实机版本和真实误差留 T12 留证。

离线测试中的 mock 仅验证身份拒绝、异常留存、输出失败不可显示通过等控制流，不代替固定目标类或 NPU 数值运行。
