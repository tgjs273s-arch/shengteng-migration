# -*- coding: utf-8 -*-
r"""_patch_v1_banner.py —— 给 v1 决策记录加「已被 v2 取代」的头部指引，然后重打包并核对

为什么必须做：v1 与 v2 同时存在于包里，而 **v1 的 ② 数字（557.0）、③ 口径（24.391% 未标分母语言）
与「前两步一致即正确性保障」都已被证伪或收紧**。包交给任何人之前，v1 必须自带署名指引，
否则"引用到旧数"是迟早的事（这正是坑 112「多源真相」的形态）。

闸门（本脚本自带）：
  · 幂等：已含指引则跳过（不重复插入）
  · 身份校验：**必须**在文件里找到 v1 的特征旧数（`24.391` 或 `557.0`），否则判定"改错文件"并中止
  · 改前备份：写入 `工具_重建\_backup/`
"""
import io
import os
import shutil
import subprocess
import sys
import time

PKG = r"C:\Users\HUAWEI\Desktop\转交给codex的内容"
TOOLS = os.path.join(PKG, "04_核心资产", "工具_重建")
SKILL = os.path.join(PKG, "04_核心资产", "qwen35-ascend-migrator_整合版")
PY = sys.executable
V1 = os.path.join(SKILL, "docs", "推荐配置与保底配置_决策记录.md")
V2 = "docs/推荐配置与保底配置_决策记录_v2_20260922.md"

BANNER = """> ## ⚠️ 本文已被 **v2** 取代（2026-09-22，外部复核后更正）
>
> **请以 `{v2}` 为准。** 本文保留作历史与更正痕迹，其中以下内容**已被证伪或收紧**，不得再引用：
>
> | 本文（v1）的旧说法 | 现状 |
> |---|---|
> | ② `skip_gdn=false` 臂 = **557.0 ms** | 错：557.0 是工具输出里的 `B(first)`（第一次运行值）；两次运行为 556.95 / 568.15 ⇒ 均值 **562.55 ms** |
> | ③ "A 比 B 快 **24.391%**" | 分母未声明；**主口径改为省时比例 `(B−A)/B` = 19.606%（95% 区间 [18.323%, 20.889%]，按三对比例重算）**，`(B−A)/A` 的 24.389% 降为次口径 |
> | "前两步逐点一致 ⇒ 数值正确性有保障" | **证伪**：全步复算（1..100）下**同臂** loss 524/600（87.3%）、grad_norm 541/600（90.2%）超 2%，**A/A 第 3–4 步即分叉**；完整数值一致性**未建立** |
> | "triton 只清元数据 ⇒ 此前 A/B 仍有效、无需重跑" | **收回**：历史 A/B 只保留在**当时环境**口径；缺完整哈希归档与修复后运行验证 |
> | "①②是机制分解，说明各部分贡献多少" | 更正为：①是**六键组合的探索性效果**（不可分配到单键），②是**另一基线上的单键探索且不确定**，③是最终直接比较 |
>
> 复算工具与冻结协议：`工具_重建/numeric_diag.py` + `工具_重建/protocols/numeric_verification_100step.json`

""".replace("{v2}", V2)


def main():
    if not os.path.isfile(V1):
        print("BANNER_FAIL 缺 %s" % V1)
        return 2
    s = io.open(V1, encoding="utf-8").read()
    if "本文已被 **v2** 取代" in s:
        print("BANNER_SKIP 已含指引（幂等）")
    else:
        # 身份校验：必须能找到 v1 的特征旧数，否则拒绝改
        marks = [m for m in ("24.391", "557.0", "24.39") if m in s]
        if not marks:
            print("BANNER_ABORT 未在目标文件里找到 v1 特征旧数 ⇒ 可能改错文件，拒绝写入")
            return 3
        os.makedirs(os.path.join(TOOLS, "_backup"), exist_ok=True)
        bk = os.path.join(TOOLS, "_backup", "决策记录_v1.%s.bak" % time.strftime("%Y%m%d_%H%M%S"))
        shutil.copy2(V1, bk)
        io.open(V1, "w", encoding="utf-8", newline="").write(BANNER + s)
        print("BANNER_OK 已插入指引（特征旧数命中：%s）；备份点 %s" % (marks, os.path.basename(bk)))

    # 重打包 + 只读核对（同一脚本内完成，避免"改了却没进包"）
    for args, label in ((["--apply"], "打包同步"), (["--verify-only"], "只读核对")):
        r = subprocess.run([PY, os.path.join(TOOLS, "safe_pack_sync.py")] + args,
                           capture_output=True, text=True, encoding="utf-8", errors="replace")
        out = ((r.stdout or "") + (r.stderr or "")).strip().splitlines()
        keep = [l for l in out if "SAFE_PACK" in l or "不一致" in l]
        print("%s：rc=%d | %s" % (label, r.returncode, " / ".join(keep[-2:])))
        if r.returncode != 0:
            print("BANNER_FAIL 打包/核对未通过")
            return 1
    print("BANNER_DONE_OK")
    return 0


if __name__ == "__main__":
    sys.exit(main())
