# -*- coding: utf-8 -*-
r"""_patch_profile_kernels.py —— 把 kernel 级归因并入 profile 协议（判据不动，补结果与修正结论）"""
import io
import json
import os
import shutil
import time

P = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                 "qwen35-ascend-migrator_整合版", "protocols", "profile_short_20260922.json")
KEY = "kernel_attribution_2026_09_22"
d = json.load(io.open(P, encoding="utf-8"))
if KEY in d:
    print("PATCH_SKIP 已并入")
    raise SystemExit(0)
d[KEY] = {
    "note": "桶比例（Computing 78%）无法区分『真实算子 vs 重计算』⇒ 按 criteria 的要求做**算子级归因**。"
            "数据源：同一批 profile 的 kernel_details.csv（35750 行 = 7150 发射/步 × 5 步；总耗时 3172977 us）。"
            "★ 纪律：**只用比例**；被 profiling 拖慢的绝对时间不与稳态对标。",
    "top_kernels_by_time": {
        "aclnnMatmul_MatMulV3Common_MatMulV3": {"pct": 20.50, "calls": 2085},
        "prepare_wy_repr_bwd_kernel": {"pct": 6.17, "calls": 90},
        "causal_conv1d_bwd_kernel": {"pct": 5.44, "calls": 90},
        "chunk_bwd_kernel_dqkwg": {"pct": 5.27, "calls": 90},
        "chunk_gated_delta_rule_fwd_kernel_h_blockdim64": {"pct": 5.03, "calls": 180},
        "chunk_gated_delta_rule_bwd_kernel_dhu_blockdim64": {"pct": 4.61, "calls": 90},
        "aclnnInplaceCopy_TransposeAiCore_Transpose": {"pct": 3.96, "calls": 4055},
        "aclnnMul_MulAiCore_Mul": {"pct": 3.51, "calls": 2540},
        "aclnnInplaceCopy_CastAiCore_Cast": {"pct": 3.30, "calls": 3815},
        "aclnnLogSoftmaxBackward_LogSoftmaxGrad": {"pct": 2.53, "calls": 5},
        "aclnnLogSoftmax_LogSoftmaxAiCore_LogSoftmaxV2": {"pct": 2.08, "calls": 5},
        "aclnnAdd_AddAiCore_Add": {"pct": 1.73, "calls": 1930},
        "causal_conv1d_fwd_kernel": {"pct": 1.70, "calls": 180},
        "RmsNormGrad": {"pct": 1.68, "calls": 395},
        "aclnnApplyAdamWV2": {"pct": 1.65, "calls": 1600}
    },
    "families": {
        "GDN / linear-attn 系列": {"pct": 28.22,
                                  "members": ["prepare_wy_repr_bwd", "causal_conv1d_bwd", "chunk_bwd_kernel_dqkwg",
                                              "chunk_gated_delta_rule_fwd_kernel_h", "chunk_gated_delta_rule_bwd_dhu",
                                              "causal_conv1d_fwd"]},
        "matmul 族（MatMulV3）": {"pct": 20.50},
        "小算子/拷贝/类型转换簇": {"pct": 16.3,
                                 "members": ["InplaceCopy_Transpose", "InplaceCopy_Cast", "InplaceCopy_Slice",
                                             "Mul", "Add", "ApplyAdamWV2", "LpNormV2", "Subs"],
                                 "calls_total": "≈20000 次/5 步（每步约 4000 次）"}
    },
    "revised_decision": {
        "verdict": "profile 支持两个候选方向（按证据强度排序）：① **GDN/linear-attn 算子族（28.2%，最大算子族）**；"
                   "② **小算子/拷贝/cast 簇（16.3%、近 2 万次发射，与 19.3% Free 吻合）**。通信仍不在列（未重叠 2.7%）。",
        "why_this_supersedes_bucket_only_reading": "桶比例只能说『计算占 78%』；算子级归因才能说清"
                                                   "『哪一族占大头』——这一步正是 criteria 里要求的『再做归因才允许宣布方向』。",
        "still_not_decided": "① **重计算占比**仍未量化（需要 `--with-stack` 或识别 recompute 重复 kernel 序列）；"
                             "② **显存曲线**未采（`with_memory=false`），GDN 相关显存增长仍未取证 ⇒ "
                             "『是否值得为 GDN 恢复更快路径』必须等显存证据，不能只看耗时占比。",
        "next_candidates_ranked": [
            "GDN/linear-attn 族：先确认它是否落在关键路径（而非可与 matmul 重叠），再考虑局部优化；"
            "同时采显存曲线以判断『恢复更快路径』的代价",
            "小算子簇：核对是否可由融合/减少 cast-copy 消掉（与 Free 19.3%、7150 发射/步一致）",
            "通信组合（reshard_after_forward × pregather）：**暂不投入**（未重叠仅 2.7%）"
        ]
    },
    "boundaries": [
        "kernel_details.csv 的 `Duration(us)` 是 kernel 级设备时间，**不含**主机下发与同步等待 ⇒ "
        "它与桶的 Free 19.3% 是**两类信息**，不能相加当成 100%。",
        "族占比是**按算子名手工归族**的结果（名字里含 GDN/chunk/conv1d 者归入 GDN 族）；"
        "归族规则写在这里，若换版本算子命名变化，需重新归族。",
        "样本仍为 5 个步（50–55）、mock 数据、单机单次。"
    ],
    "recorded_at": time.strftime("%Y-%m-%d %H:%M:%S")
}
out_bak_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "_backup", "skill_baks")
os.makedirs(out_bak_dir, exist_ok=True)
shutil.copy2(P, os.path.join(out_bak_dir, "profile_short_20260922.json.bak_kernel_%s"
                             % time.strftime("%Y%m%d_%H%M%S")))
json.dump(d, io.open(P, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
back = json.load(io.open(P, encoding="utf-8"))
assert KEY in back and "criteria" in back and "results" in back
print("PATCH_OK 已并入 %s（判据未动）" % KEY)
