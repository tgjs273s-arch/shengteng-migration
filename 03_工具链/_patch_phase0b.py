# -*- coding: utf-8 -*-
r"""_patch_phase0b.py —— 修两个自伤缺陷（都是我自己刚引入的）

① `prepush_check.py`：上次补丁用"函数边界切片"替换 run_gate，**把夹在中间的 `gate_delivery_tree`
  一并删掉了** ⇒ G8 抛 `NameError: gate_delivery_tree`。教训：**切片替换要确认被切掉的区间里没有
   别的定义**（这次没有）。处置：把该函数原样插回 `def run_gate(` 之前。
② `gate_selftest.py`：正对照用例在**桩函数仍生效**的状态下运行（我直到最后才 `m._run = real_run`）
   ⇒ 把"rc=0 且标记命中"的用例判成 FAIL。这属于"测试自己制造的假缺陷"（今天第 N 次），
   处置：正对照前**显式还原真实实现**。
"""
import io

TOOLS = r"C:\Users\HUAWEI\Desktop\转交给codex的内容\04_核心资产\工具_重建"

# ---- ① 把 gate_delivery_tree 插回 ----
P1 = TOOLS + r"\prepush_check.py"
s = io.open(P1, encoding="utf-8").read()
fn = '''def gate_delivery_tree():
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
    if bad:
        return False, "G8_FAIL"
    return True, "G8_OK dirs=%d zip=%s" % (len(need), zips[0])


'''
if "def gate_delivery_tree(" not in s:
    assert "def run_gate(" in s
    s = s.replace("def run_gate(", fn + "def run_gate(", 1)
    io.open(P1, "w", encoding="utf-8", newline="").write(s)
    print("已插回 gate_delivery_tree")
else:
    print("gate_delivery_tree 已存在，跳过")

# ---- ② 正对照前还原真实 _run ----
P2 = TOOLS + r"\gate_selftest.py"
t = io.open(P2, encoding="utf-8").read()
old = """    # ---- 正对照：正常工具 + 正确标记 ⇒ PASS ----
    ok = m.run_gate("""
new = """    # ---- 正对照：正常工具 + 正确标记 ⇒ PASS ----
    m._run = real_run          # ★ 必须先还原真实实现：否则正对照仍在桩下运行（测试自伤）
    ok = m.run_gate("""
assert old in t, "正对照锚点未找到"
t = t.replace(old, new, 1)
io.open(P2, "w", encoding="utf-8", newline="").write(t)
print("正对照已恢复使用真实 _run")
print("PATCH_PHASE0B_OK")
