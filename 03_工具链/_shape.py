import json, glob, io, os
def shape(o, d=0, maxd=3):
    pad = "  " * d
    if isinstance(o, dict):
        for k, v in o.items():
            if isinstance(v, (dict, list)) and d < maxd:
                print("%s%s:" % (pad, k)); shape(v, d+1, maxd)
            else:
                print("%s%s = %r" % (pad, k, (v if not isinstance(v, str) or len(v) < 70 else v[:70]+"…")))
    elif isinstance(o, list):
        print("%s[list len=%d] 首元素:" % (pad, len(o)))
        if o: shape(o[0], d+1, maxd)
for p in sorted(glob.glob("protocols/*.json")):
    print("="*20, p)
    shape(json.load(io.open(p, encoding="utf-8")), 0, 2)
print("="*20, "results 样例（若存在）")
for p in sorted(glob.glob("**/results*.json", recursive=True))[:3]:
    print("--", p); shape(json.load(io.open(p, encoding="utf-8")), 0, 2)
