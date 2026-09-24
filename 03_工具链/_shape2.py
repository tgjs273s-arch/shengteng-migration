import json, io, glob
p = json.load(io.open("protocols/numeric_diag_phase1b_v2.json", encoding="utf-8"))
print("--- inherited_criteria.fail_closed ---")
for x in p["inherited_criteria"]["fail_closed"]: print("  *", x)
print("--- revision_notes ---")
for x in p["revision_notes"]: print("  *", x)
print("--- diagnostic_tools.tools（name/gate）---")
for t in p["diagnostic_tools"]["tools"]: print("  *", t.get("name"), "gate=", t.get("gate"))
print("--- 结果文件 ---")
for f in sorted(glob.glob("**/*results*.json", recursive=True)) + sorted(glob.glob("**/*.results.json", recursive=True)):
    print("  ", f)
