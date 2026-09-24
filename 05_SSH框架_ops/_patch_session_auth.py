# -*- coding: utf-8 -*-
r"""_patch_session_auth.py —— 把会话守护的**跳板认证**改成 Go 网关实际接受的方式

实测依据（2026-09-22，两条独立证据）：
1. `_authdiag.py`：对跳板 `SSH-2.0-Go`，
   · `auth_none(juser)` ⇒ **成功**（网关只认用户名，不需要密码！）
   · `auth_password(juser, jpass)` ⇒ `AuthenticationException: Authentication failed: transport shut down
     or saw EOF`（**送密码反而被掐断** —— 这与"token 过期"的表现一模一样，我因此误判了两轮）
   · `auth_interactive_dumb` ⇒ 同样 EOF
2. OpenSSH `-vv`（NumberOfPasswordPrompts=0，即**没发密码**）⇒ `Entering interactive session` 成功，
   随后 `channel 0: open failed: administratively prohibited: **only direct-tcpip is permitted**`
   ⇒ 跳板是**纯代理**：只允许 direct-tcpip 通道，不允许开 session 通道（所以"在跳板上执行命令"根本不可行）。

因此：跳板用 `paramiko.Transport` + `auth_none(username)`；目标机照旧用密码认证（密码是控制台给的连接密码）。
本脚本只改 `Conn.__init__` 里跳板那一段，改前备份、改后编译+回读断言。
"""
import io
import os
import re
import shutil
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
TARGET = os.path.join(HERE, "ssh_session.py")

OLD = '''        self.jump = paramiko.SSHClient()
        self.jump.set_missing_host_key_policy(paramiko.AutoAddPolicy())
        self.jump.connect(self.jump_host, port=self.jump_port, username=self.jump_user,
                          password=self.jump_pass, timeout=connect_timeout,
                          allow_agent=False, look_for_keys=False)
        self.jump.get_transport().set_keepalive(30)
        chan = self.jump.get_transport().open_channel(
            "direct-tcpip", (self.t_host, 22), ("127.0.0.1", 0))'''

NEW = '''        # ★ 2026-09-22 实测（_authdiag.py + OpenSSH -vv）——**跳板认证方式**：
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
        chan = self.jump.open_channel("direct-tcpip", (self.t_host, 22), ("127.0.0.1", 0))'''

src = io.open(TARGET, encoding="utf-8").read()
if NEW in src:
    print("PATCH_SKIP 已经打过补丁")
    sys.exit(0)
if OLD not in src:
    print("PATCH_FAIL 找不到待替换的跳板连接代码块（文件被改过？）")
    sys.exit(2)
bak = TARGET + ".bak_%s" % time.strftime("%Y%m%d_%H%M%S")
shutil.copy2(TARGET, bak)
out = src.replace(OLD, NEW, 1)
io.open(TARGET, "w", encoding="utf-8").write(out)

back = io.open(TARGET, encoding="utf-8").read()
assert "auth_none(self.jump_user)" in back and "auth_password" not in back.replace("auth_password 认证", "")
assert back.count("open_channel(\"direct-tcpip\"") == 1
print("PATCH_OK 跳板改为 transport+auth_none；备份=%s" % bak)
