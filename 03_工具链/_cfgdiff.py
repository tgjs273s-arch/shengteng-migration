import json, io
p0 = json.load(io.open("_reanalysis/results_p0.json", encoding="utf-8"))
p2 = json.load(io.open("_reanalysis/results_p2.json", encoding="utf-8"))
print("A 臂三次身份:", sorted(set(r["config_sha256"] for r in p0)))
print("P2 两次身份 :", sorted(set(r["config_sha256"] for r in p2)))
a, b = p0[0]["config_dump"], p2[0]["config_dump"]
diff = [(k, a.get(k), b.get(k)) for k in sorted(set(a) | set(b)) if a.get(k) != b.get(k)]
print("A vs P2 配置差异键数 =", len(diff))
for k, x, y in diff[:12]: print("   %-46s A=%r  P2=%r" % (k, x, y))
