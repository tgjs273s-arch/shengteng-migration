import json, io, time, os, sys
sys.path.insert(0, r"C:\Users\HUAWEI\Desktop\转交给codex的内容\04_核心资产\分析脚本\ssh框架\_ops")
STORE = os.path.join(r"C:\Users\HUAWEI\Desktop\转交给codex的内容\04_核心资产\分析脚本\ssh框架\_ops", "_token", "current.json")
rec = {"token": "jt_<REDACTED_ID>:<REDACTED_SECRET>",
       "jump": "113.47.8.48:2234", "target": "root@199.98.58.213",
       "password": "MD8NtRW1HDHNsO8K", "env": "", "source": "paste",
       "saved_at": time.strftime("%Y-%m-%dT%H:%M:%S"), "received_at": time.time()}
io.open(STORE + ".bak_before_new", "w", encoding="utf-8").write(io.open(STORE, encoding="utf-8").read())
io.open(STORE, "w", encoding="utf-8").write(json.dumps(rec, ensure_ascii=False, indent=1))
print("STORE_OK target=%s jump=%s token=%s…" % (rec["target"], rec["jump"], rec["token"][:14]))
try:
    import paramiko; print("PARAMIKO_OK", paramiko.__version__)
except ImportError as e:
    print("PARAMIKO_MISSING", e)
