# -*- coding: utf-8 -*-
r"""_feed_command.py —— 把控制台粘贴的「连接命令 + 密码」直接落进 store，触发守护自愈建连

用法（**一次调用，几秒内建连**）：
    python _feed_command.py --file 粘贴块.txt
    python _feed_command.py --line "ssh -J jt_…:…@113.47.8.48:2234 root@1.2.3.4" --password "…"

为什么要有它：token 只有 **5 分钟**有效期，而"取 token"必须人工点「SSH 直连」。
上一轮失败的原因就是这个窗口被我花在了"先写守护代码"上（第一次连接在 10:39:51，token 已过期）。
⇒ 现在守护在**盯 store 文件**，所以正确顺序是：**先让守护待命，再喂 token**。

解析规则（照控制台的可见文本，不猜接口）：
  · 从 `ssh -J <jumpuser>:<jumppass>@<jumphost>:<jumpport> <targetuser>@<targethost>` 里取全部字段；
  · 密码取「连接密码」下一行的内容；若命令行里已含 jump 密码，则密码用于**目标机**。
  · **解析不出来就报错退出**，不写半个记录（宁可失败，不要假记录）。
"""
import argparse
import io
import json
import os
import re
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
STORE = os.path.join(HERE, "_token", "current.json")
SSH_RE = re.compile(
    r"ssh\s+-J\s+(?P<juser>[^@:\s]+):(?P<jpass>[^@\s]+)@(?P<jhost>[^:\s]+):(?P<jport>\d+)\s+"
    r"(?P<tuser>[^@\s]+)@(?P<thost>[\w.\-]+)")


def parse(text, password=None):
    m = SSH_RE.search(text)
    if not m:
        raise SystemExit("FEED_FAIL 未能从文本里解析出 `ssh -J …` 命令（照原文粘贴即可）")
    g = m.groupdict()
    if not password:
        m2 = re.search(r"连接密码[:：]?\s*\n?\s*(\S+)", text)
        if m2:
            password = m2.group(1)
    if not password:
        raise SystemExit("FEED_FAIL 未找到连接密码（可用 --password 显式给出）")
    return {"token": "%s:%s" % (g["juser"], g["jpass"]),
            "jump": "%s:%s" % (g["jhost"], g["jport"]),
            "target": "%s@%s" % (g["tuser"], g["thost"]),
            "password": password, "env": "", "source": "paste",
            "saved_at": time.strftime("%Y-%m-%dT%H:%M:%S"), "received_at": time.time()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--file", default="")
    ap.add_argument("--line", default="")
    ap.add_argument("--password", default="")
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()
    text = io.open(a.file, encoding="utf-8-sig").read() if a.file else a.line
    if not text.strip():
        raise SystemExit("FEED_FAIL 没有输入（--file 或 --line）")
    rec = parse(text, a.password or None)
    print("FEED_PARSED target=%s jump=%s token=%s… 密码长度=%d"
          % (rec["target"], rec["jump"], rec["token"][:18], len(rec["password"])))
    if a.dry_run:
        print("FEED_DRY_RUN 未写盘")
        return 0
    if os.path.exists(STORE):
        bak = STORE + ".bak_feed"
        io.open(bak, "w", encoding="utf-8").write(io.open(STORE, encoding="utf-8").read())
    os.makedirs(os.path.dirname(STORE), exist_ok=True)
    io.open(STORE, "w", encoding="utf-8").write(json.dumps(rec, ensure_ascii=False, indent=1))
    print("FEED_OK 已写 store（守护应在 2 秒内自动建连）：%s" % STORE)
    return 0


if __name__ == "__main__":
    sys.exit(main())
