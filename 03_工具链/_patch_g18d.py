# -*- coding: utf-8 -*-
r"""_patch_g18d.py —— 按整块重建 MANUAL_GATES（停止用"行的形状"做判据）

前两版的错都在同一处：**用"行的形状"判断结构**。
  第 1 版：按行过滤 ⇒ 只删了 G18 的首行，留下悬空续行；
  第 2 版：按"引号开头 + `),` 结尾"判孤儿 ⇒ **误伤其余 4 条的首行**（形状完全相同），
          于是 4 条元组只剩开括号、没了收尾。
教训：**结构必须按结构改**。本版直接从 `MANUAL_GATES = [` 到匹配的 `]` 整块替换为正确内容。
"""
import io
import py_compile

P = r"C:\Users\HUAWEI\Desktop\转交给codex的内容\04_核心资产\工具_重建\prepush_check.py"
s = io.open(P, encoding="utf-8").read()

BLOCK = '''MANUAL_GATES = [
    ("G4", "配置模板字段一致性", "旧驱动器内联；两套模板已由 G26/G27 逐键审计覆盖",
     "人工确认：config/templates/*.yaml 与 config/versions.lock 的 geometry/dataloader 键一致"),
    ("G7", "报告 / PDF 生成一致性", "PDF 生成脚本被删（无法逐字恢复）",
     "人工确认：04_性能测试报告/、02_README/ 下的 pdf 与 md 内容一致（时间戳与正文）"),
    ("G11", "目标机依赖覆盖（torch/torch_npu 真机）", "旧驱动器内联",
     "人工确认：真机 05_preflight 输出（本轮已由 env.json + triton 修复记录覆盖）"),
    ("G17", "创意书字段完整性", "check_proposal.py / fill_proposal_fields.py 被删",
     "人工确认：01_项目创意书 是否仍有【待填/待确认】字样（用户负责）"),
]
# 注：G18（源代码包一致性）**已升为自动闸门**（check_source_consistency.py）⇒ 从人工清单移除，
#     否则同一次输出里会既 PASS 又 MANUAL（自相矛盾的状态比缺项更坏）。
'''

start = s.find("MANUAL_GATES = [")
if start < 0:
    print("FAIL 找不到 MANUAL_GATES")
    raise SystemExit(1)
# 从 start 起做括号配对，找到与之匹配的 ']'
depth, end = 0, None
for k in range(start, len(s)):
    if s[k] == "[":
        depth += 1
    elif s[k] == "]":
        depth -= 1
        if depth == 0:
            end = k + 1
            break
if end is None:
    print("FAIL 括号不配对")
    raise SystemExit(1)
s2 = s[:start] + BLOCK.rstrip("\n") + s[end:]
io.open(P, "w", encoding="utf-8", newline="").write(s2)
print("MANUAL_GATES 已按整块重建（区间 %d..%d 字符被替换）" % (start, end))

try:
    py_compile.compile(P, doraise=True)
    print("COMPILE_OK")
except Exception as exc:
    print("COMPILE_FAIL %s" % exc)
    raise SystemExit(1)
