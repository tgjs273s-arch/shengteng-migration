# -*- coding: utf-8 -*-
r"""_patch_prepush3.py —— 把 G5/G6/G12 接进驱动器，并把剩余丢失项分流

- 新增 G5（sk04_judge 自带自检，**实测存在的接口** `selfcheck_sk04.py --repo-root`）
- 新增 G6（negative_controls.py：四项坏样本必须被拒）
- 新增 G12（evidence_manifest.py --verify）
- 新增 G8（驱动器内自检交付目录结构：必需目录存在且非空）
- 其余丢失项（G4/G7/G11/G17/G18）移入 MANUAL_GATES：**逐条打印 + 写进 docs 的人工清单**，
  并且总判决用**专有措辞** `PREPUSH_OK_WITH_MANUAL`，避免被读成"全绿"
"""
import io

P = r"C:\Users\HUAWEI\Desktop\转交给codex的内容\04_核心资产\工具_重建\prepush_check.py"
s = io.open(P, encoding="utf-8").read()

# 1) 追加三条闸门 + 一条驱动器内闸门
anchor = '''    ("G28", "坑表连续性（1..N 无缺号无重复）",
     [PY, os.path.join(TOOLS, "check_pitfalls.py")], TOOLS, "PITFALL_CHECK_OK"),
]'''
add = '''    ("G28", "坑表连续性（1..N 无缺号无重复）",
     [PY, os.path.join(TOOLS, "check_pitfalls.py")], TOOLS, "PITFALL_CHECK_OK"),
    # ---- 2026-09-22 补：三条原本"丢失"的技术闸门，按现存入口重建 ----
    ("G5", "SK04 判定链自检（sk04_judge **本体健在**，此处只补调用封装）",
     [PY, os.path.join(SKILL, "sk04_judge", "scripts", "selfcheck_sk04.py"),
      "--repo-root", SKILL], SKILL, "_SELFTEST_OK"),
    ("G6", "判据负向对照（四项坏样本必须被拒绝）",
     [PY, os.path.join(TOOLS, "negative_controls.py")], TOOLS, "NEGATIVE_CONTROLS_OK"),
    ("G12", "证据逐文件哈希表（远端证据 15 文件）",
     [PY, os.path.join(TOOLS, "evidence_manifest.py"), "--verify", "--dir",
      os.path.join(PKG, "04_核心资产", "远端证据_20260922")], TOOLS, "EVIDENCE_MANIFEST_OK"),
    ("G8", "交付目录结构（必需目录存在且非空）", [], "", "_DRIVER_"),
]'''
assert anchor in s, "GATES 锚点未找到"
s = s.replace(anchor, add, 1)

# 2) 丢失项分流：把已补齐的四条从 LOST 移除，其余移到 MANUAL
old_lost = s[s.index("LOST_GATES = ["):s.index("]\n\nENV_DEPS") + 2]
new_lost = '''# 只能**人工确认**的项（逐条打印 + 见 docs/交付前人工清单.md）；不再计入自动判决的 FAIL，
# 但总判决会输出专有措辞 PREPUSH_OK_WITH_MANUAL —— 防止被读成"全绿"。
MANUAL_GATES = [
    ("G4", "配置模板字段一致性", "旧驱动器内联；两套模板已由 G26/G27 逐键审计覆盖",
     "人工确认：config/templates/*.yaml 与 config/versions.lock 的 geometry/dataloader 键一致"),
    ("G7", "报告 / PDF 生成一致性", "PDF 生成脚本被删（无法逐字恢复）",
     "人工确认：04_性能测试报告/、02_README/ 下的 pdf 与 md 内容一致（时间戳与正文）"),
    ("G11", "目标机依赖覆盖（torch/torch_npu 真机）", "旧驱动器内联",
     "人工确认：真机 05_preflight 输出（本轮已由 env.json + triton 修复记录覆盖）"),
    ("G17", "创意书字段完整性", "check_proposal.py / fill_proposal_fields.py 被删",
     "人工确认：01_项目创意书 是否仍有【待填/待确认】字样（用户负责）"),
    ("G18", "源代码包一致性", "build_source_package.py 被删",
     "人工确认：交付 07_源代码 与活副本 scripts/ 的差异是否只含预期项"),
]

LOST_GATES = []   # 已无"既没重建、也没写进人工清单"的项（2026-09-22 清空）
'''
s = s.replace(old_lost, new_lost, 1)

# 3) 驱动器内 G8 实现 + 分派
s = s.replace('''        if gid == "G1":
            passed, got = gate_compile_smoke(cwd, timeout)''',
'''        if gid == "G1":
            passed, got = gate_compile_smoke(cwd, timeout)
        elif gid == "G8":
            passed, got = gate_delivery_tree()''', 1)

s = s.replace('''def check_backend(argv, cwd):''',
'''def gate_delivery_tree():
    """G8：交付目录结构体检（驱动器内实现；不碰文件，只数数与判存在）。"""
    root = r"C:\\Users\\HUAWEI\\Desktop\\C4AI复赛_交付材料"
    need = ["00_数据与结论", "01_项目创意书", "02_README", "03_精度分析报告", "04_性能测试报告",
            "05_原始日志", "06_Skill", "07_源代码", "08_复现视频", "99_官方模板与要求"]
    bad = []
    for d in need:
        p = os.path.join(root, d)
        n = sum(len(fs) for _r, _dd, fs in os.walk(p)) if os.path.isdir(p) else -1
        if n <= 0:
            bad.append("%s(文件=%s)" % (d, n))
    zips = [f for f in os.listdir(os.path.join(root, "06_Skill"))
            if f.endswith(".zip")] if os.path.isdir(os.path.join(root, "06_Skill")) else []
    if len(zips) != 1:
        bad.append("06_Skill 下 zip 数=%d（应为 1 ⇒ 多版本会让评委困惑）" % len(zips))
    print("        交付目录 %d 个必需目录；zip 数=%d" % (len(need), len(zips)))
    for b in bad[:5]:
        print("        ! %s" % b)
    return (not bad), ("G8_OK dirs=%d zip=%s" % (len(need), zips[0] if zips else "-")) if not bad \\
        else "G8_FAIL"


def check_backend(argv, cwd):''', 1)

# 4) 汇总与判决
s = s.replace('''    print("\\n-- 已知丢失（计入 FAIL）--")
    for gid, name, why, alt in LOST_GATES:
        print("  MISSING %-4s %-44s %s" % (gid, name[:44], why))''',
'''    if LOST_GATES:
        print("\\n-- 已知丢失（计入 FAIL）--")
        for gid, name, why, alt in LOST_GATES:
            print("  MISSING %-4s %-44s %s" % (gid, name[:44], why))
    print("\\n-- 需人工确认（见 docs/交付前人工清单.md；不计入自动判决）--")
    for gid, name, why, how in MANUAL_GATES:
        print("  MANUAL %-4s %-40s %s" % (gid, name[:40], how))''', 1)

s = s.replace('''    if npass != len(results) or LOST_GATES:''', '''    if npass != len(results) or LOST_GATES:''', 1)
s = s.replace('''    print("PREPUSH_OK —— 现存闸门全通过且无缺失检查")
    return 0''',
'''    print("PREPUSH_OK_WITH_MANUAL —— 自动化闸门 %d/%d 全通过、无丢失项；"
          "另有 %d 项**需人工确认**（清单见 docs/交付前人工清单.md）"
          % (npass, len(results), len(MANUAL_GATES)))
    return 0''', 1)

io.open(P, "w", encoding="utf-8", newline="").write(s)
print("PATCH_PREPUSH3_OK 闸门数=%d" % s.count('("G'))
