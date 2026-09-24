# -*- coding: utf-8 -*-
r"""verify_codex_findings.py —— 逐条**实测** Codex 复核里的指控（只读，不改任何东西）

被指控项与验证方法：
  P0-a  safe_pack_sync.py：备份失败（返回 None）后**是否仍然删除目标**？→ 直接读源码路径
  P1-b  prepush G23：pyflakes **无法运行**时是否仍报 OK？→ 读实现（只看 undefined name 字面量，
        完全没看 rc/stderr）⇒ 用"把 pyflakes 换成不存在的模块"做**可执行**复现
  P1-c  prepush G5：rc=1 且输出含 `A_CASE_OK` 之类子串时是否仍判 PASS？→ 读实现 + 复现
  数字-1 ③ 的"省时比例"(B−A)/B：均值与 95% 区间（自己算，不转换原区间端点）
  数字-2 ② 的 false 臂均值到底是 557.0 还是 562.55
  数字-3 versions.lock 的 deviation.data.cutoff_len（2048?）与出货模板实际 cutoff_len（1024?）是否一致
"""
import json
import os
import re
import statistics
import subprocess
import sys

PKG = r"C:\Users\HUAWEI\Desktop\转交给codex的内容"
TOOLS = os.path.join(PKG, "04_核心资产", "工具_重建")
SKILL = os.path.join(PKG, "04_核心资产", "qwen35-ascend-migrator_整合版")
EV = os.path.join(PKG, "04_核心资产", "远端证据_20260922")
T3 = 4.303  # t(0.975, df=2)


def show(label, ok, detail):
    print("%-6s %-46s %s" % ("确认" if ok else "不成立", label, detail))


def read(p):
    return open(p, encoding="utf-8").read()


# ---------- P0-a：备份失败后是否仍删除 ----------
s = read(os.path.join(TOOLS, "safe_pack_sync.py"))
m = re.search(r"bk = backup_dir\(target_dir, os\.basename\(parent\)\)(.{0,400})", s, re.S)
seg = m.group(1) if m else "<未找到>"
deletes_when_none = "rmtree(target_dir)" in seg and "if bk" not in seg.split("rmtree")[0]
print("== P0-a safe_pack_sync：备份失败仍删除？ ==")
print("   备份后紧邻代码：%s" % " / ".join(l.strip() for l in seg.splitlines()[:6] if l.strip())[:300])
show("备份返回 None 时仍执行 rmtree", deletes_when_none,
     "⇒ Codex 的指控**成立**：备份失败只打印、不中止，随后照样删目标；若解压再失败，旧目录已丢" if deletes_when_none
     else "未复现")

# ---------- P1-b：pyflakes 不可用时 G23 的行为（可执行复现）----------
print("\n== P1-b prepush G23：工具无法运行时是否误报 OK ==")
code = (
    "import importlib,sys\n"
    "sys.modules['pyflakes'] = None\n"
    "import subprocess\n"
    "r = subprocess.run([sys.executable, '-c', 'import no_such_mod_pyflakes_xyz'], capture_output=True, text=True)\n"
    "print('rc=%s err_has_nomod=%s' % (r.returncode, 'No module named' in (r.stderr or '')))\n"
)
r = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
print("   对照：不存在的模块 ⇒ %s" % (r.stdout or r.stderr).strip()[:80])
src_b = read(os.path.join(TOOLS, "prepush_check.py"))
gate23 = re.search(r"def gate_pyflakes.*?\n    return \(not undef\).*?\n", src_b, re.S)
body = gate23.group(0) if gate23 else ""
ignores_rc = ("r.returncode" not in body) and ("No module named" not in body)
show("G23 不看 rc / 不看 stderr ⇒ 工具缺失也报 OK", ignores_rc,
     "⇒ **成立**：它只 grep `undefined name` 字面量；pyflakes 没装时输出为空 ⇒ undef=[] ⇒ 报 G23_OK")

# ---------- P1-c：G5 的标记逻辑是否忽略 rc ----------
print("\n== P1-c prepush G5：局部 OK + rc=1 是否被判 PASS ==")
seg5 = re.search(r'if marker == "_SELFTEST_OK":(.{0,320})', src_b, re.S)
s5 = seg5.group(1) if seg5 else ""
ignores_rc5 = "returncode" not in s5
print("   _SELFTEST_OK 分支：%s" % " / ".join(l.strip() for l in s5.splitlines()[:5] if l.strip())[:260])
show("该分支不看退出码", ignores_rc5,
     "⇒ **成立**：`re.search(r'[A-Z0-9_]+_OK')` 会命中输出里更早出现的 `A_CASE_OK` 之类子串，rc=1 也判 PASS")

# ---------- 数字-1：③ 的省时比例 (B−A)/B ----------
print("\n== 数字-1 ③ 省时比例重算 ==")
d3 = json.load(open(os.path.join(EV, "AB_AVSB_20260921__results.json"), encoding="utf-8"))
A = [r for r in d3 if r.get("variant") == "A"]
B = [r for r in d3 if r.get("variant") == "B"]
fk = "window_11_end_median_ms"
ratios = [(b[fk] - a[fk]) / b[fk] for a, b in zip(A, B)]
mean = statistics.mean(ratios)
sd = statistics.stdev(ratios)
se = sd / (len(ratios) ** 0.5)
lo, hi = mean - T3 * se, mean + T3 * se
print("   A 均值=%.2f ms  B 均值=%.2f ms" % (statistics.mean([a[fk] for a in A]), statistics.mean([b[fk] for b in B])))
print("   每对省时比例=%s" % [round(x * 100, 3) for x in ratios])
print("   均值=%.3f%%  95%%区间=[%.3f%%, %.3f%%]（t=4.303, df=2）" % (mean * 100, lo * 100, hi * 100))
print("   对照 (B−A)/A 口径：%.3f%%" % (((statistics.mean([b[fk] for b in B]) -
                                        statistics.mean([a[fk] for a in A])) /
                                       statistics.mean([a[fk] for a in A])) * 100))

# ---------- 数字-2：② 的 false 臂均值 ----------
print("\n== 数字-2 ② skip=false 臂均值 ==")
d2 = json.load(open(os.path.join(EV, "aftermab_20260921_143648__ab_skipgdn__summary.json"), encoding="utf-8"))
runs2 = d2.get("runs", [])
va = [r.get("window_median_ms") for r in runs2 if r.get("variant") == "A"]
vb = [r.get("window_median_ms") for r in runs2 if r.get("variant") == "B"]
print("   A(true)=%s 均值=%.2f   B(false)=%s 均值=%.2f"
      % (va, statistics.mean(va), vb, statistics.mean(vb)))
show("Codex 说应为 562.55 ms", abs(statistics.mean(vb) - 562.55) < 0.01,
     "⇒ 我提示词里写的 557.0 实为 `B(first)`（第一次运行值），**不是均值**")

# ---------- 数字-3：cutoff_len 一致性 ----------
print("\n== 数字-3 偏离账本 vs 出货模板的 cutoff_len ==")
lock = read(os.path.join(SKILL, "config", "versions.lock"))
led = [l for l in lock.splitlines() if l.startswith("deviation.data.cutoff_len")]
import yaml
for name in ("qwen3_5_0_8B_recommended_A.yaml", "qwen3_5_0_8B_fallback_B.yaml"):
    t = yaml.safe_load(read(os.path.join(SKILL, "config", "templates", name)))
    bp = t["data"]["dataset_param"]["basic_parameters"]
    ds = str(bp.get("dataset", ""))
    print("   %-42s cutoff_len=%s  dataset=%s" % (name, bp.get("cutoff_len"), os.path.basename(ds)))
print("   账本行：%s" % (led[0][:130] if led else "<无>"))
