import os
from urllib.parse import quote
BROKER_SECRET = os.environ.get("ZERO_RISK_BROKER_SECRET", "")
if not BROKER_SECRET.strip():
    raise SystemExit("请设置 ZERO_RISK_BROKER_SECRET")
AUTH_PATH = quote(BROKER_SECRET, safe="")
import urllib.request, urllib.error, json
def call(method, path, t=8):
    url = "http://127.0.0.1:8791" + path
    req = urllib.request.Request(url, method=method, data=(b"{}" if method == "POST" else None),
                                 headers={"Content-Type": "application/json"})
    op = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    try:
        with op.open(req, timeout=t) as r:
            return r.status, (r.read() or b"").decode("utf-8", "replace")[:300]
    except urllib.error.HTTPError as e:
        return e.code, (e.read() or b"").decode("utf-8", "replace")[:200]
    except Exception as e:
        return "EXC", "%s: %s" % (type(e).__name__, e)
for m, p in [("GET", "/"), ("GET", "/token/" + AUTH_PATH), ("GET", "/next?wait=1"),
             ("POST", "/need/" + AUTH_PATH), ("GET", "/status"), ("GET", "/token/bad")]:
    print(("%-5s %-28s -> %s" % (m, p, call(m, p))).replace(AUTH_PATH, "<BROKER_SECRET>").replace(BROKER_SECRET, "<BROKER_SECRET>"))
