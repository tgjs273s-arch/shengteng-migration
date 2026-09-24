# -*- coding: utf-8 -*-
r"""ssh_session.py —— **按恢复出的规格重建**（原文件随坑 180 的工具链删除一起丢失）

## 它是什么
一条**长期存活**的 SSH 连接（跳板 → 目标机）+ 本地单行协议服务，供本地工具反复发命令。
判据与做法（从会话日志、客户端协议与 cfg 反推，2026-09-22）：
  · 本地 `127.0.0.1:8792`，一行请求一行 JSON：`RUN <base64(命令)>` / `STATUS` / `PING`
    （未知动词返回 `{"ok": false, "error": "未知动词 'X'"}` —— 与原版实测行为一致）；
  · token 来自 store（`_token/current.json`，由页面点「SSH 直连」+ 剪贴板守望落盘），**只用于建连**；
    建连之后 token 不再被使用（实测曾有连接连续服务 3 小时以上）；
  · 单条命令默认超时 110 s（`--cmd-timeout`）。

## 为什么必须"一次连上"，以及为什么必须"盯着 store"
平台 token 有效期 **5 分钟**且很可能一次性 ⇒
  ① **绝不用连接命令做试探**（一次没用上就废了）；
  ② 而"取新 token"的动作在**用户侧**（点按钮 → 剪贴板 → 守望落 store），时机不可预测；
  ③ ⇒ 守护必须**盯着 store 文件**：谁写进新记录，就立刻用它建连（本版新增 `watch_store` 与
     `RECONNECT` 动词；原版有断线自愈，但入口不可知）。
实测教训（本轮第一次连接就踩到）：`AuthenticationException: Authentication failed: transport shut down
or saw EOF` —— 在**跳板**认证阶段失败，最常见原因就是 **token 已过期**（而不是机制不对）。

## 与原版的差异（诚实声明）
1. 原版用的是什么客户端无法确认（文件已丢）⇒ 本版用 **paramiko**（跳板 `direct-tcpip` 通道 + 密码认证）。
2. 原版其余 CLI 子命令/开关**不可知**；本版只保证"服务端协议 + store 取 token + 断线/换 token 自愈"这条链路。
"""
import argparse
import base64
import io
import json
import os
import socketserver
import sys
import threading
import time

HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_STORE = os.path.join(HERE, "_token", "current.json")
LOG = os.path.join(HERE, "_run", "session_rebuilt.log")


def log(msg):
    try:
        os.makedirs(os.path.dirname(LOG), exist_ok=True)
        with io.open(LOG, "a", encoding="utf-8") as fh:
            fh.write("%s %s\n" % (time.strftime("%H:%M:%S"), msg))
    except OSError:
        pass


class Conn:
    """一条通过跳板的 SSH 连接（跳板认证用 token，目标认证用 store 里的密码）。"""

    def __init__(self, rec, connect_timeout=25):
        import paramiko
        self.rec = rec
        self.jump_host, self.jump_port = rec["jump"].rsplit(":", 1)
        self.jump_port = int(self.jump_port)
        # ★ 2026-09-22 实测（OpenSSH -vv 的 implicit ProxyCommand 原文）：
        #   `ssh -l jt_TOKEN:TOKENPASS -p 2234 -W [%h]:%p 113.47.8.48` ⇒ **整串 `jt_…:…` 就是用户名**。
        #   OpenSSH 的 `-J` 只接受 `[user@]host[:port]`，故 `-J a:b@host` 里的 `a:b` 全算用户名。
        #   我第一版按 ':' 拆开只取前半段 ⇒ 网关 auth_none **通过**但**拒绝开通道**
        #   （`Administratively prohibited`）：**认证通过 ≠ 被授权**。
        self.jump_user = rec["token"]
        self.target = rec["target"]
        self.t_user, self.t_host = self.target.split("@", 1)
        self.t_pass = rec["password"]
        # ★ 2026-09-22 实测（_authdiag.py + OpenSSH -vv）——**跳板认证方式**：
        #   ① 跳板 banner = `SSH-2.0-Go`，`auth_none(username)` **直接成功** ⇒ 它只认**用户名**；
        #   ② 而 `auth_password(username, 密码)` ⇒ `Authentication failed: transport shut down or saw EOF`
        #      —— **送密码反而被掐断**，表现与"token 过期"完全一样（我因此误判了两轮；
        #      真正的教训：**EOF ≠ 过期**，先把"服务器接受哪种认证"问清楚，别靠猜）；
        #   ③ OpenSSH 原文：`channel 0: open failed: administratively prohibited: only direct-tcpip
        #      is permitted` ⇒ 跳板是**纯代理**，只允许 direct-tcpip，禁止 session 通道。
        self.jump = paramiko.Transport((self.jump_host, self.jump_port))
        self.jump.banner_timeout = connect_timeout
        self.jump.auth_timeout = connect_timeout
        self.jump.start_client(timeout=connect_timeout)
        self.jump.auth_none(self.jump_user)          # 只认用户名（token 的冒号前半段）
        self.jump.set_keepalive(30)
        chan = self.jump.open_channel("direct-tcpip", (self.t_host, 22), ("127.0.0.1", 0))
        self.ssh = paramiko.SSHClient()
        self.ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
        self.ssh.connect(self.t_host, port=22, username=self.t_user, password=self.t_pass,
                         sock=chan, timeout=connect_timeout, allow_agent=False,
                         look_for_keys=False)
        self.ssh.get_transport().set_keepalive(30)
        self.lock = threading.Lock()

    def run(self, cmd, timeout):
        with self.lock:
            stdin, stdout, stderr = self.ssh.exec_command(cmd, timeout=timeout)
            ch = stdout.channel
            ch.settimeout(timeout)
            # ★★ 2026-09-22 修 bug：第一版手写"轮询 recv_ready + exit_status_ready"的收尾，
            #   在**大输出**时丢尾部（`exit_status_ready()` 可能先于"stdout 全部到达"为真，
            #   于是只排空了当时已到的缓冲就返回）。221 KB 侥幸没事，555 KB 必踩。
            #   拦住它的不是我的自检，而是**下游的逐文件 SHA256**（pull_evidence）⇒ 报拉取失败。
            #   正确做法是 paramiko 标准 `read()`：阻塞到通道 EOF，不截断。
            try:
                out = stdout.read()
                err = stderr.read()
            except Exception as exc:
                try:
                    ch.close()
                except Exception:
                    pass
                return {"rc": None, "stdout": "",
                        "stderr": "[本地读取失败 %s: %s]" % (type(exc).__name__, exc),
                        "timeout": True}
            rc = ch.recv_exit_status()
            return {"rc": rc, "stdout": out.decode("utf-8", "replace"),
                    "stderr": err.decode("utf-8", "replace")}

    def close(self):
        for c in (self.ssh, self.jump):
            try:
                c.close()
            except Exception:
                pass


def read_store(path):
    try:
        with io.open(path, encoding="utf-8") as fh:
            return json.load(fh)
    except (OSError, ValueError):
        return None


class Session:
    def __init__(self, store, cmd_timeout):
        self.store = store
        self.cmd_timeout = cmd_timeout
        self.conn = None
        self.lock = threading.Lock()
        self.connects = 0
        self.reconnects = 0
        self.commands = 0
        self.last_error = ""
        self.token_prefix = ""
        self.host = ""

    def store_mtime(self):
        try:
            return os.path.getmtime(self.store)
        except OSError:
            return 0.0

    def ensure(self, force=False):
        with self.lock:
            if self.conn is not None and not force:
                return self.conn
            if force and self.conn is not None:
                self.conn.close()
                self.conn = None
                self.reconnects += 1
            rec = read_store(self.store)
            if not rec:
                self.last_error = "store 无记录：%s" % self.store
                log("SESSION_NO_STORE %s" % self.last_error)
                return None
            self.token_prefix = str(rec.get("token", ""))[:14]
            self.host = rec.get("target", "")
            try:
                self.conn = Conn(rec)
                self.connects += 1
                self.last_error = ""
                log("SESSION_CONNECTED host=%s token=%s… connects=%d"
                    % (self.host, self.token_prefix, self.connects))
            except Exception as exc:
                self.conn = None
                self.last_error = "%s: %s" % (type(exc).__name__, exc)
                log("SESSION_CONNECT_ERROR %s" % self.last_error)
            return self.conn

    def watch_store(self):
        """★ 自愈：**store 一变就重连**（原版有断线自愈，重建版把这条件补上）。

        token 只有 5 分钟、且刷新动作在用户侧（点「SSH 直连」→ 剪贴板 → 守望落 store），
        时机不可预测 ⇒ 让守护盯文件：谁写进新记录，立刻用新 token 建连。
        """
        last = self.store_mtime()
        while True:
            time.sleep(2.0)
            now = self.store_mtime()
            if now != last:
                last = now
                log("SESSION_STORE_CHANGED 检测到新 token ⇒ 重连")
                self.ensure(force=True)


def make_handler(sess):
    class H(socketserver.StreamRequestHandler):
        def handle(self):
            line = self.rfile.readline().decode("ascii", "replace").strip()
            if not line:
                return
            verb, _sp, arg = line.partition(" ")
            verb = verb.upper()
            if verb == "PING":
                reply = {"ok": True, "pong": True}
            elif verb == "STATUS":
                reply = {"connected": sess.conn is not None, "connects": sess.connects,
                         "reconnects": sess.reconnects, "commands": sess.commands,
                         "last_error": sess.last_error, "token_prefix": sess.token_prefix,
                         "host": sess.host, "store": sess.store}
            elif verb == "RECONNECT":
                # 原版不支持（实测返回"未知动词"）；本版新增，用于"我确认 store 已更新"时立刻重连
                ok = sess.ensure(force=True) is not None
                reply = {"ok": ok, "last_error": sess.last_error}
            elif verb == "RUN":
                conn = sess.ensure()
                if conn is None:
                    reply = {"ok": False, "rc": None, "stdout": "",
                             "stderr": "未连接：%s" % sess.last_error}
                else:
                    try:
                        cmd = base64.b64decode(arg, validate=True).decode("utf-8")
                    except Exception as exc:
                        reply = {"ok": False, "rc": None, "stdout": "",
                                 "stderr": "base64 解码失败：%s" % exc}
                    else:
                        try:
                            res = conn.run(cmd, sess.cmd_timeout)
                            sess.commands += 1
                            reply = {"ok": res.get("rc") == 0, **res}
                        except Exception as exc:
                            sess.last_error = "%s: %s" % (type(exc).__name__, exc)
                            log("SESSION_RUN_ERROR %s" % sess.last_error)
                            sess.ensure(force=True)          # 断线自愈
                            reply = {"ok": False, "rc": None, "stdout": "",
                                     "stderr": "执行异常（已尝试重连）：%s" % sess.last_error}
            else:
                reply = {"ok": False, "error": "未知动词 '%s'" % verb}
            self.wfile.write((json.dumps(reply, ensure_ascii=False) + "\n").encode("utf-8"))

    return H


class Srv(socketserver.ThreadingTCPServer):
    allow_reuse_address = False            # 坑 171：Windows 上 True 会让别的进程抢答
    daemon_threads = True


def serve(port, store, cmd_timeout):
    sess = Session(store, cmd_timeout)
    srv = Srv(("127.0.0.1", port), make_handler(sess))
    log("SESSION_LISTEN 127.0.0.1:%d store=%s cmd_timeout=%ds" % (port, store, cmd_timeout))
    print("SESSION_LISTEN 127.0.0.1:%d store=%s cmd_timeout=%ds" % (port, store, cmd_timeout),
          flush=True)
    threading.Thread(target=sess.watch_store, daemon=True).start()
    conn = sess.ensure()
    print("SESSION_FIRST_CONNECT connected=%s err=%s" % (conn is not None, sess.last_error),
          flush=True)
    srv.serve_forever()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--serve", action="store_true")
    ap.add_argument("--port", type=int, default=8792)
    ap.add_argument("--store", default=DEFAULT_STORE)
    ap.add_argument("--cmd-timeout", type=int, default=110)
    ap.add_argument("--once", default="", help="一次性执行一条命令后退出（仍只用一次连接）")
    a = ap.parse_args()
    if a.once:
        sess = Session(a.store, a.cmd_timeout)
        if sess.ensure() is None:
            print("ONCE_FAIL %s" % sess.last_error)
            return 3
        res = sess.conn.run(a.once, a.cmd_timeout)
        print(json.dumps(res, ensure_ascii=False)[:4000])
        sess.conn.close()
        return 0 if res.get("rc") == 0 else 1
    serve(a.port, a.store, a.cmd_timeout)
    return 0


if __name__ == "__main__":
    sys.exit(main())
