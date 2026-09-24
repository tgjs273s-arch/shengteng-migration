#!/usr/bin/env python3
# -*- coding: utf-8 -*-
r"""token_store.py —— 最小可用版"连接信息落盘"（重建；原 `token_agent.py` 已在坑 180 中丢失）

为什么需要：会话守护（`ssh_session.py`）在**连接断开**时会调用 `token_broker.cli_need()` 或直接读
store 里的 token 重连。如果 store 里没有**较新**的 token，自愈就会失败（只能人工重连）。
所以每拿到一次连接命令，就应当立刻落盘 —— 这是"会话断了也能自己接上"的保险。

与旧版的取舍：旧 `token_agent.py` 支持剪贴板守望 / endpoint 自动获取（那套依赖 DSH 配置与页面脚本，
当前都用不上）。本版只做**一件事**：从句柄文本或文件里解析出 token/jump/target/password 并落盘，
**不打印 token 全文**（只打印前 14 位），并在写盘后**回读校验**（避免"写了个坏 JSON 还报成功"）。

用法：
  python token_store.py --text-file <把连接命令保存成的文件>
  python token_store.py --status
"""
import argparse
import json
import os
import re
import sys
from datetime import datetime, timezone

RE_SSH = re.compile(r"ssh\s+-J\s+(jt_[^\s@]+)@([^\s:]+):(\d+)\s+(\S+@\S+)")
RE_PW = re.compile(r"(?:连接密码|password)\s*[:：]?\s*(\S+)", re.I)
RE_ENV = re.compile(r"环境名称\s*[:：]?\s*(\S+)")
HOME = os.environ.get("DSH_HOME", "")
CFG = os.environ.get("DSH_REMOTE_TOKEN_CFG") or (os.path.join(HOME, "remote-token.json") if HOME else "")


def store_path():
    if CFG and os.path.isfile(CFG):
        try:
            with open(CFG, encoding="utf-8") as fh:
                p = json.load(fh).get("store")
            if p:
                return p
        except Exception as exc:
            print("TOKEN_STORE_CFG_WARN %s: %s" % (type(exc).__name__, exc))
    return os.environ.get("DSH_REMOTE_TOKEN_STORE") or (
        os.path.join(HOME, "remote-token-current.json") if HOME else "")


def parse(text):
    m = RE_SSH.search(text)
    if not m:
        return None, "找不到 `ssh -J jt_…@跳板:端口 用户@目标` 形式的连接命令"
    pw = RE_PW.search(text)
    if not pw:
        return None, "找不到连接密码（没有它连不上目标机）"
    env = RE_ENV.search(text)
    return {"token": m.group(1), "jump": "%s:%s" % (m.group(2), m.group(3)),
            "target": m.group(4), "password": pw.group(1),
            "env": env.group(1) if env else ""}, "ok"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--text-file")
    ap.add_argument("--status", action="store_true")
    a = ap.parse_args()
    path = store_path()
    if not path:
        print("TOKEN_STORE_FAIL 没有可用 store 路径（设 DSH_HOME 或在 remote-token.json 里给 store）")
        return 2

    if a.status:
        if not os.path.isfile(path):
            print("TOKEN_STORE_STATUS 空（%s）" % path)
            return 0
        rec = json.load(open(path, encoding="utf-8"))
        print("TOKEN_STORE_STATUS token=%s… source=%s saved_at=%s jump=%s target=%s"
              % (str(rec.get("token"))[:14], rec.get("source"), rec.get("saved_at"),
                 rec.get("jump"), rec.get("target")))
        return 0

    if not a.text_file:
        print("需要一个输入：--text-file <文件>（把控制台给的连接命令+密码存成文件）")
        return 2
    with open(a.text_file, encoding="utf-8-sig") as fh:
        info, why = parse(fh.read())
    if info is None:
        print("TOKEN_STORE_FAIL %s" % why)
        return 3
    rec = dict(info)
    rec.update({"source": "paste", "saved_at": datetime.now(timezone.utc).isoformat(),
                "received_at": __import__("time").time()})
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(rec, fh, ensure_ascii=False, indent=1)
    try:
        os.chmod(path, 0o600)
    except OSError:
        pass
    back = json.load(open(path, encoding="utf-8"))          # ★ 回读校验
    ok = back.get("token") == info["token"] and back.get("password") == info["password"]
    print("TOKEN_STORE_%s token=%s… jump=%s target=%s env=%s → %s"
          % ("OK" if ok else "FAIL", info["token"][:14], info["jump"], info["target"],
             info["env"] or "?", path))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
