import os
from urllib.parse import quote
BROKER_SECRET = os.environ.get("ZERO_RISK_BROKER_SECRET", "")
if not BROKER_SECRET.strip():
    raise SystemExit("请设置 ZERO_RISK_BROKER_SECRET")
AUTH_PATH = quote(BROKER_SECRET, safe="")
import json, urllib.request, time
rec = {"token": "jt_FAKEPROBE0000000000000000000000000000000000000000000000000000000000:FAKE",
       "jump": "113.47.8.48:2234", "target": "root@199.103.55.150", "password": "FAKE",
       "env": "", "source": "probe"}
req = urllib.request.Request("http://127.0.0.1:8791/token/" + AUTH_PATH,
                             data=json.dumps(rec).encode(), method="POST",
                             headers={"Content-Type": "application/json"})
op = urllib.request.build_opener(urllib.request.ProxyHandler({}))
with op.open(req, timeout=10) as r:
    print("POST /token ->", r.status, r.read().decode("utf-8", "replace")[:160])
with op.open("http://127.0.0.1:8791/status", timeout=10) as r:
    print("GET /status ->", r.read().decode("utf-8", "replace")[:260])
