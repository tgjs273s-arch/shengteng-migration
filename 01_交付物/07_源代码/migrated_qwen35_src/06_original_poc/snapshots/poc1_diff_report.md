# poc1 识别 -> mindspeed 落点映射 / 待核对清单
> 本报告由 poc1_parse.py 据 AST 识别结果自动生成; 命中行号为真实解析所得。
> '逐层 diff' 需在 mindspeed 官方参考实现源码就位后由本脚本扩展生成; 当前产出为映射+待核对清单, 不臆造 diff 数字。

## linear_attn_GatedDeltaNet
- 命中次数: 26　去重行号数: 17　去重符号: ['Qwen3_5GatedDeltaNet', 'causal_conv1d_fn', 'causal_conv1d_update', 'chunk_gated_delta_rule', 'torch_causal_conv1d_update', 'torch_chunk_gated_delta_rule']
- 行号样例: [75, 77, 81, 83, 226, 230, 254, 380, 427, 428, 429, 481]
- mindspeed 落点/适配建议: mindspeed_mm/fsdp/models/qwen3_5 下复核 GatedDeltaNet; chunk 前向+反向优先昇腾 Triton, 极个别 kernel 下沉 Ascend C; 标[需算子落地+逐层余弦核对]

## full_attn
- 命中次数: 3　去重行号数: 3　去重符号: ['Qwen3_5Attention']
- 行号样例: [655, 774, 833]
- mindspeed 落点/适配建议: 复用 CANN/MindSpeed 全注意力实现; 标[逐层核对]

## vision_3d_conv_patch
- 命中次数: 7　去重行号数: 6　去重符号: ['Conv3d', 'patch_embed', 'temporal_patch_size']
- 行号样例: [869, 873, 874, 879, 1044, 1109]
- mindspeed 落点/适配建议: 复用 CANN 标准 Conv3d + 图算融合; 标[shape 核对, 环2 已 forward 验证]

## custom_act_or_norm
- 命中次数: 13　去重行号数: 13　去重符号: ['FusedRMSNormGated', 'Qwen3_5RMSNorm', 'Qwen3_5RMSNormGated']
- 行号样例: [80, 84, 195, 416, 417, 418, 679, 680, 746, 776, 777, 845]
- mindspeed 落点/适配建议: RMSNorm/quick_gelu/silu 映射 CANN 等价算子; 标[逐层核对]

--- 合计命中 49 处(去重行号见各类) ---