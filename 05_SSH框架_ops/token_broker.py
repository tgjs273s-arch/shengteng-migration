# -*- coding: utf-8 -*-
r"""token_broker.py —— **按恢复出的规格重写**（原文件随坑 180 的工具链删除一起丢失）

## 为什么会有这个文件（事件经过，全部实测）
1. `_ops/` 下的 `token_broker.py` / `ssh_session.py` / `token_agent.py` **文件已不存在**（目录里只剩 `_run`/`_token`）；
2. 但三个守护**仍在内存中运行**（PID 与之前相同，8791/8792 仍在监听）⇒ 表面"一切正常"；
3. 会话日志每 5 秒重复 `SESSION_TOKEN_STALE age=3xxxxs > max_age=240.0 ⇒ 请经纪即时签发`
   与 `SESSION_CONNECT_LOOP_ERROR ModuleNotFoundError: No module named 'token_broker'`
   ⇒ **真实卡点只是"这个模块不存在"**（不是网络、不是平台、不是 token 本身）。
4. 我先放了一个**自描述 shim**（模块级 `__getattr__` + 日志），5 秒内就问出了丢失的接口签名：
   `cli_need(port, secret, timeout=45, quiet=True, cfg={store, alert, refreshEverySec, hivelab{...}})`。
   ⇒ **不猜函数名**（坑 163/179 纪律）。
5. 再用 HTTP 探活拿到经纪的完整端点（`GET /` 自描述）：
   `POST /need/<secret>` 要 token；`POST /token/<secret>` 送 token；`GET /next?wait=N` 给页面长轮询；
   `GET /status` 返回 `mint_seq/last_saved_at/tokens_saved/page_polls/store` 等。
   实测响应：`POST /need/… -> {"ok": true, "seq": 1, "note": "已声明需要 token；页面若在长轮询，会立刻去取"}`。

## 本实现的范围与边界
- **实现**：经纪 HTTP 服务（`/`、`/status`、`/need/<secret>`、`/token/<secret>`、`/next`）+ 落盘 store +
  `cli_need()`（会话守护 import 它来"即时签发"）。
- **不实现**（原设计的边界，见 cfg 原文）：**不逆向平台接口、不保存 Cookie**。token 来自
  「你已登录的 HiDevLab 页面里点『SSH 直连』」→ 剪贴板 → `token_agent.py --watch-clipboard` 落 store，
  或页面上跑的 `hivelab_token.user.js` 每 4 分钟自动点一次并 POST 回来。
  ⇒ **页面不在 = 永远取不到新 token**，此时 `cli_need` 必须**明确超时报错**，绝不假装拿到。
- **丢功能**：原文件还有哪些入口（CLI 子命令等）**无法恢复**（文件没了）。本文件只保证：
  服务端端点 + store 格式 + `cli_need` 被会话守护调用这一条链路可用，**其余入口按未知处理**。

## store 格式（与既有 `_token/current.json` 一致，不改字段名）
{"token": "jt_…:…", "jump": "ip:port", "target": "root@ip", "password": "…",
 "env": "", "source": "paste|page", "saved_at": ISO8601, "received_at": epoch}
"""
import argparse
import http.server
import io
import json
import os
import socketserver
import sys
import threading
import time
import urllib.error
import urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_STORE = os.path.join(HERE, "_token", "current.json")
# Authentication is configured at startup; no repository default credential.


def _now():
    return time.time()


def read_store(path):
    try:
        with io.open(path, encoding="utf-8") as fh:
            return json.load(fh)
    except (OSError, ValueError):
        return None


def write_store(path, rec):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path + ".tmp"
    with io.open(tmp, "w", encoding="utf-8") as fh:
        json.dump(rec, fh, ensure_ascii=False, indent=1)
    os.replace(tmp, path)               # 原子替换：守望/会话可能同时读


class Broker:
    """经纪状态（内存）+ 落盘。页面与会话通过 HTTP 端点交互。"""

    def __init__(self, store, secret):
        self.store = store
        self.secret = secret
        self.lock = threading.Lock()
        self.mint_seq = 0
        self.mint_requested_at = 0.0
        self.tokens_saved = 0
        self.tokens_rejected = 0
        self.last_saved_at = 0.0
        self.last_how = ""
        self.last_reject = ""
        self.page_polls = 0
        self.page_last_poll = 0.0

    def need(self):
        with self.lock:
            self.mint_seq += 1
            self.mint_requested_at = _now()
            return self.mint_seq

    def put_token(self, rec, how="page"):
        with self.lock:
            if not rec.get("token") or not rec.get("jump") or not rec.get("target"):
                self.tokens_rejected += 1
                self.last_reject = "缺 token/jump/target"
                return False, "缺 token/jump/target"
            rec["received_at"] = _now()
            rec.setdefault("source", how)
            write_store(self.store, rec)
            self.tokens_saved += 1
            self.last_saved_at = rec["received_at"]
            self.last_how = how
            return True, "saved"

    def snapshot(self):
        with self.lock:
            return {"mint_requested_at": self.mint_requested_at, "mint_seq": self.mint_seq,
                    "tokens_saved": self.tokens_saved, "tokens_rejected": self.tokens_rejected,
                    "last_saved_at": self.last_saved_at, "last_how": self.last_how,
                    "last_reject": self.last_reject, "page_polls": self.page_polls,
                    "page_last_poll": self.page_last_poll, "store": self.store}


def make_handler(broker):
    class H(http.server.BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def log_message(self, fmt, *args):        # 静音默认 stderr 访问日志
            pass

        def _json(self, obj, code=200):
            body = json.dumps(obj, ensure_ascii=False).encode("utf-8")
            self.send_response(code)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def _body(self):
            n = int(self.headers.get("Content-Length") or 0)
            raw = self.rfile.read(n) if n else b""
            if not raw:
                return {}
            try:
                return json.loads(raw.decode("utf-8"))
            except ValueError:
                return {}

        def do_GET(self):
            path = self.path.split("?")[0]
            if path == "/":
                return self._json({"ok": True, "service": "token_broker",
                                   "hint": "POST /token/<secret> 送 token；GET /next 长轮询；"
                                           "POST /need/<secret> 要 token"})
            if path == "/status":
                return self._json(broker.snapshot())
            if path == "/next":                   # 页面长轮询：mint=true 时立刻去平台取一个
                wait = 0.0
                q = self.path.split("?", 1)[1] if "?" in self.path else ""
                for kv in q.split("&"):
                    if kv.startswith("wait="):
                        try:
                            wait = min(float(kv[5:]), 30.0)
                        except ValueError:
                            wait = 0.0
                with broker.lock:
                    broker.page_polls += 1
                    broker.page_last_poll = _now()
                    seq = broker.mint_seq
                mint = seq > 0
                if not mint and wait > 0:
                    time.sleep(wait)              # 长轮询：等到有人要 token（或超时）
                    with broker.lock:
                        mint = broker.mint_seq > 0
                return self._json({"mint": mint, "seq": seq, "secret": broker.secret,
                                   "hint": "mint=true 时请立刻去平台取一个连接命令并 "
                                           "POST /token/<secret>"})
            return self._json({"ok": True, "service": "token_broker",
                               "hint": "POST /token/<secret> 送 token；GET /next 长轮询；"
                                       "POST /need/<secret> 要 token"})

        def do_POST(self):
            path = self.path.split("?")[0]
            parts = [p for p in path.split("/") if p]
            body = self._body()
            if len(parts) == 2 and parts[0] == "need":
                if parts[1] != broker.secret:
                    return self._json({"ok": False, "error": "secret 不匹配"}, 403)
                seq = broker.need()
                print("BROKER_NEED seq=%d（等页面来取）" % seq, flush=True)
                return self._json({"ok": True, "seq": seq,
                                   "note": "已声明需要 token；页面若在长轮询，会立刻去取"})
            if len(parts) == 2 and parts[0] == "token":
                if parts[1] != broker.secret:
                    return self._json({"ok": False, "error": "secret 不匹配"}, 403)
                ok, why = broker.put_token(body, how="page")
                print("BROKER_TOKEN ok=%s why=%s" % (ok, why), flush=True)
                return self._json({"ok": ok, "why": why}, 200 if ok else 400)
            return self._json({"ok": False, "error": "未知路径 %s" % path}, 404)

    return H


class _Server(socketserver.ThreadingTCPServer):
    allow_reuse_address = False               # ★ 坑 171 同族：Windows 上 True 会让**另一个进程**抢答


def serve(port, secret, store):
    broker = Broker(store, secret)
    srv = _Server(("127.0.0.1", port), make_handler(broker))
    print("BROKER_LISTEN http://127.0.0.1:%d/token/<secret>" % port, flush=True)
    print("  长轮询: GET /next?wait=25   要 token: POST /need/<secret>", flush=True)
    print("  store=%s" % store, flush=True)
    srv.serve_forever()


class _Tok(dict):
    """返回给会话守护的对象：**既是 dict 又支持属性访问**。

    为什么这么做：丢失模块的返回约定已无法从文件里读到，而会话守护此刻**正在运行**、
    会立刻拿真实调用去撞。做成"两用"可以一次覆盖最常见的两种期待（dict 取值 / .token 属性），
    再用日志把真实用法暴露出来 —— 比猜一个形态然后反复重启更快，也不会假通过。
    """

    def __getattr__(self, k):
        try:
            return self[k]
        except KeyError:
            raise AttributeError(k)


def cli_need(port, secret, timeout=45, quiet=True, cfg=None):
    """会话守护调用的入口：声明需要 token → 等页面/守望把**新** token 落 store → 返回它。

    ★ 从实测签名恢复而来（探针日志：`CALL cli_need args=(8791, '<BROKER_SECRET>')
      kwargs={'timeout': 45, 'quiet': True, 'cfg': {...}}`）。

    判据（fail-closed）：只有 store 里出现 **received_at 晚于本次请求时刻** 的记录才算"新"，
    否则超时报错 —— **绝不**把上一轮的旧 token 当新 token 返回（那会让会话反复用过期 token 撞墙，
    而日志看起来像"网络问题"，正是坑 158 的形态）。
    """
    cfg = cfg or {}
    store = cfg.get("store") or DEFAULT_STORE
    base = "http://127.0.0.1:%d" % int(port)
    op = urllib.request.build_opener(urllib.request.ProxyHandler({}))   # ★ 绕开 Windows 代理（实测坑）

    def _post(path, payload=None):
        data = json.dumps(payload or {}).encode("utf-8")
        req = urllib.request.Request(base + path, data=data, method="POST",
                                     headers={"Content-Type": "application/json"})
        with op.open(req, timeout=10) as r:
            return json.loads((r.read() or b"{}").decode("utf-8"))

    t0 = _now()
    try:
        res = _post("/need/%s" % secret)
    except (urllib.error.URLError, OSError) as exc:
        raise RuntimeError("token_broker 不可达（%s）：%s —— 请先启动经纪" % (base, exc))
    seq = res.get("seq")
    if not quiet:
        print("CLI_NEED seq=%s（已请求签发，等待页面/守望落 store）" % seq, flush=True)

    deadline = t0 + float(timeout)
    old = read_store(store)
    old_at = (old or {}).get("received_at") or 0.0
    while _now() < deadline:
        rec = read_store(store)
        if rec and (rec.get("received_at") or 0.0) > max(t0 - 0.5, old_at):
            if not quiet:
                print("CLI_NEED got token=%s… jump=%s target=%s source=%s"
                      % (str(rec.get("token"))[:14], rec.get("jump"), rec.get("target"),
                         rec.get("source")), flush=True)
            return _Tok(rec)
        time.sleep(0.5)
    raise TimeoutError(
        "cli_need 超时（%.0fs）：store 未出现新 token。**这不是网络故障** —— "
        "需要在已登录的 HiDevLab 页面点『SSH 直连』（或让 hivelab_token.user.js 在页面上自动点），"
        "token_agent.py 会把剪贴板里的命令写进 store。store=%s" % (timeout, store))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8791)
    ap.add_argument("--secret", default=os.environ.get("ZERO_RISK_BROKER_SECRET"),
                    help="认证密钥（默认读取 ZERO_RISK_BROKER_SECRET 环境变量）")
    ap.add_argument("--store", default=DEFAULT_STORE)
    ap.add_argument("--put", default="", help="手动送一份 token JSON 进 store（POST /token 的等价物）")
    a = ap.parse_args()
    if not a.secret or not a.secret.strip():
        ap.error("请设置 ZERO_RISK_BROKER_SECRET 或显式传入 --secret；禁止使用空密钥")
    if a.put:
        rec = json.loads(a.put)
        b = Broker(a.store, a.secret)
        ok, why = b.put_token(rec, how="cli")
        print("BROKER_PUT ok=%s why=%s store=%s" % (ok, why, a.store))
        return 0 if ok else 1
    serve(a.port, a.secret, a.store)
    return 0


if __name__ == "__main__":
    sys.exit(main())
