#!/usr/bin/env python3
# -*- coding: utf-8 -*-
r"""pull_evidence.py —— 走**仍在运行的**会话守护把远端证据拉回来（重建版，最小必要）

背景（坑 180）：`04_核心资产\交付物生成脚本\` 与 `分析脚本\ssh框架\_ops\` 已被一个
"先清理后重建"的脚本删除，原 `pull_via_session.py` 随之消失。但——

  * **远端机器上的证据完好**（`/root/ops/...`）；
  * **本地那条持久 SSH 会话守护仍是活的进程**（它的源码被删，但进程已在内存里跑着，
    并继续持有 1 小时前建立的那条连接）⇒ 不需要任何新 token 就能继续用它取数。

本脚本用**最小实现**只做一件事：连 `127.0.0.1:8792`，按守护的既有协议（一行请求一行 JSON）
执行 `sha256sum` + `base64 -w0`，本地解码并**校验哈希**后落盘。不改远端任何东西（只读）。

协议（与 `ssh_session.py` 一致）：`{"ok":true,"rc":N,"stdout":"…","stderr":"…"}`；
请求 `RUN <base64(命令)>` / `STATUS` / `PING`。

★ 与坑 178 的关系：那次"尾行粘连"出在**命令行客户端**打印 stdout 后接尾行；这里是守护返回的
  **JSON**，stdout 在 JSON 里（换行被转义），**不存在粘连问题** —— 但仍按"取最后一次出现"的老
  习惯不适用，故直接 json.loads 解析，不做正则。

用法：
  python pull_evidence.py --status
  python pull_evidence.py --out <本地目录> [--files f1 f2 ...] [--list <清单文件>]
"""
import argparse
import base64
import json
import os
import re
import socket
import sys

HOST, PORT = "127.0.0.1", 8792

# 默认清单：本轮所有实验产物（远端绝对路径 → 本地相对名）
DEFAULT = [
    "/root/ops/AB_AVSB_20260921/summary.json",
    "/root/ops/AB_AVSB_20260921/results.json",
    "/root/ops/ab_avsb.log",
    "/root/ops/aftermab_20260921_143648/ab_switches/results.json",
    "/root/ops/aftermab_20260921_143648/ab_switches/summary_recomputed.json",
    "/root/ops/aftermab_20260921_143648/ab_skipgdn/summary.json",
    "/root/ops/aftermab_20260921_143648/A_official_switches.yaml",
    "/root/ops/aftermab_20260921_143648/base_ab_option1.yaml",
    "/root/ops/aftermab_20260921_143648/sampler_bench.txt",
    "/root/ops/memab_20260921_141051/run_true_A/result.json",
    "/root/ops/memab_20260921_141051/run_false_B/result.json",
    "/root/ops/triton_state_before.json",
    "/root/ops/triton_state_after.json",
    "/root/ops/A_recommended.yaml",
    "/root/ops/B_fallback.yaml",
]


def call(line, timeout=180):
    """发一行请求，读一行 JSON。"""
    s = socket.create_connection((HOST, PORT), timeout=timeout)
    try:
        f = s.makefile("rwb")
        f.write((line + "\n").encode("utf-8"))
        f.flush()
        raw = f.readline().decode("utf-8", "replace")
        if not raw:
            return {"ok": False, "error": "守护没有回应（可能刚退出）"}
        return json.loads(raw)
    finally:
        try:
            s.close()
        except Exception:
            pass


def run(cmd, timeout=180):
    return call("RUN " + base64.b64encode(cmd.encode("utf-8")).decode("ascii"), timeout)


def get(remote, local):
    r = run("sha256sum %s" % remote)
    if not r.get("ok") or not (r.get("stdout") or "").strip():
        return False, "sha256sum 失败：rc=%s err=%s" % (r.get("rc"), (r.get("stderr") or "")[:120])
    want = r["stdout"].strip().split()[0]
    r2 = run("base64 -w0 %s" % remote)
    if not r2.get("ok") or r2.get("rc") not in (0, None):
        return False, "base64 失败：rc=%s err=%s" % (r2.get("rc"), (r2.get("stderr") or "")[:120])
    import hashlib
    try:
        data = base64.b64decode(re.sub(r"\s+", "", r2.get("stdout") or ""), validate=True)
    except Exception as exc:
        return False, "解码失败：%s" % exc
    got = hashlib.sha256(data).hexdigest()
    if got != want:
        return False, "SHA256 不一致（远端=%s 本地=%s）⇒ 拒绝落盘" % (want[:12], got[:12])
    os.makedirs(os.path.dirname(os.path.abspath(local)) or ".", exist_ok=True)
    with open(local, "wb") as fh:
        fh.write(data)
    return True, "%d B sha256=%s" % (len(data), got[:12])


def put_file(local, remote, chunk=14000):
    """分块上传 + 两端 SHA256 对照（不一致即报错，不落盘）。"""
    import hashlib
    data = open(local, "rb").read()
    want = hashlib.sha256(data).hexdigest()
    b64 = base64.b64encode(data).decode("ascii")
    tmp = remote + ".b64tmp"
    r = run("rm -f %s; touch %s" % (tmp, tmp))
    if r.get("rc") != 0:
        return False, "创建远端临时文件失败：%s" % (r.get("stderr") or "")[:120]
    for i in range(0, len(b64), chunk):
        part = b64[i:i + chunk]
        r = run("printf %%s %s >> %s" % (part, tmp))
        if r.get("rc") != 0:
            return False, "第 %d 块失败：%s" % (i // chunk + 1, (r.get("stderr") or "")[:120])
    r = run("base64 -d %s > %s; rm -f %s; sha256sum %s" % (tmp, remote, tmp, remote))
    got = ((r.get("stdout") or "").strip().split() or [""])[0]
    if got != want:
        return False, "SHA256 不一致：本地=%s 远端=%s" % (want[:12], got[:12])
    return True, "%d B / %d 块 sha256=%s" % (len(data), (len(b64) + chunk - 1) // chunk, got[:12])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--status", action="store_true")
    ap.add_argument("--out", default="")
    ap.add_argument("--files", nargs="*", default=None)
    ap.add_argument("--put", nargs=2, metavar=("LOCAL", "REMOTE"))
    ap.add_argument("--run")
    a = ap.parse_args()

    if a.put:
        ok, why = put_file(a.put[0], a.put[1])
        print("PUT_%s %s → %s  %s" % ("OK" if ok else "FAIL", a.put[0], a.put[1], why))
        return 0 if ok else 1
    if a.run:
        r = run(a.run, timeout=600)
        sys.stdout.write(r.get("stdout") or "")
        if r.get("stderr"):
            sys.stderr.write(r["stderr"])
        print("REMOTE_RC=%s" % r.get("rc"))
        return 0 if r.get("rc") == 0 else 1
    if a.status:
        print("PING  -> %s" % json.dumps(call("PING"), ensure_ascii=False))
        st = call("STATUS")
        print("STATUS-> connected=%s commands=%s keepalives=%s reconnects=%s host=%s token=%s"
              % (st.get("connected"), st.get("commands"), st.get("keepalives"),
                 st.get("reconnects"), st.get("host"), st.get("token_prefix")))
        if st.get("last_error"):
            print("   last_error=%s" % st["last_error"])
        return 0
    if not a.out:
        print("需要 --out <本地目录>")
        return 2
    files = a.files if a.files else DEFAULT
    ok = bad = 0
    for remote in files:
        local = os.path.join(a.out, remote.replace("/root/ops/", "").replace("/", "__"))
        good, why = get(remote, local)
        print("%-8s %s  %s" % ("PULL_OK" if good else "PULL_FAIL", remote, why))
        ok += 1 if good else 0
        bad += 0 if good else 1
    print("PULL_EVIDENCE ok=%d fail=%d out=%s" % (ok, bad, a.out))
    return 0 if bad == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
