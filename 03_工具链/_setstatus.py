import io, json, os, shutil, time
P = r"C:\Users\HUAWEI\Desktop\转交给codex的内容\04_核心资产\qwen35-ascend-migrator_整合版\protocols\one_step_replay_20260922.json"
d = json.load(io.open(P, encoding="utf-8"))
old = d.get("status")
d["status"] = "criteria_frozen_and_executed（两轮同配置已实测；两 rank 一致：C1 逐字节相同、C3 不同 ⇒ first_diff=C3）"
out = os.path.join(os.path.dirname(os.path.abspath(__file__)) if "__file__" in dir() else ".", "_backup", "skill_baks")
json.dump(d, io.open(P, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
back = json.load(io.open(P, encoding="utf-8"))
assert back["status"].startswith("criteria_frozen_and_executed")
print("STATUS_OK %s → %s" % (old, back["status"]))
