# Qwen3.5 在 MindSpeed-MM FSDP 下的结构落地占位 (对照 mindspeed_mm/fsdp/models/qwen3vl/ 实现)
# 状态: 骨架占位; 完整结构由 poc1 识别清单驱动补全, 逐层 diff 见 snapshots/poc1_diff_report.md
# TODO: 依 poc1_nodes.json 的 linear_attn_GatedDeltaNet / vision_3d_conv_patch / custom_act_or_norm
#       逐类落地 nn.Module; GatedDeltaNet chunk 算子优先昇腾 Triton, 极个别下沉 Ascend C (PerfTune)。
# from . import modeling_qwen3_5  # 待补: 由 refs/ 基线迁移而来的训练侧模型类(文件未建前先注释, 避免 ImportError)