import json, io, os
P = "protocols/numeric_diag_phase1b_v2.json"
d = json.load(io.open(P, encoding="utf-8"))
k = "reanalysis_2026_09_22_r2"
print("键存在 =", k in d, "| 文件 %d B" % os.path.getsize(P))
r = d[k]
print("P0 判定 =", r["p0_verdict"]["verdict"], "| 与归档一致 =", r["p0_verdict"]["matches_archived_record"])
print("P2 判定 =", r["p2_verdict"]["verdict"], "| 与归档一致 =", r["p2_verdict"]["matches_archived_record"])
print("判据未动：window=%s thr=%s fail_closed=%d 条" % (
    d["inherited_criteria"]["window"], d["inherited_criteria"]["per_step_rel_pct"],
    len(d["inherited_criteria"]["fail_closed"])))
print("顶层键数 =", len(d))
