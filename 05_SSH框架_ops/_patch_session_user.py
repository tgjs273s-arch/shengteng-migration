# -*- coding: utf-8 -*-
r"""_patch_session_user.py —— 跳板用户名必须是**整串** `jt_TOKEN:TOKENPASS`

实测依据（决定性）：OpenSSH `-vv` 打印出它为 `-J` 生成的 implicit ProxyCommand：

    ssh -l jt_<REDACTED_ID>:<REDACTED_SECRET>…  -p 2234 -W "[%h]:%p" 113.47.8.48
        ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
        整串（含冒号后半段）都是 **-l 的用户名**

原因：OpenSSH 的 `-J` 只接受 `[user@]host[:port]`，所以 `-J a:b@host` 里的 `a:b` **全部**算用户名 ——
平台正是利用这一点把 token 与密码一起塞进用户名字段，请网关自己解析。

我第一版按 ':' 拆开、只把 `jt_…` 当用户名 ⇒ 网关 **auth_none 通过**、但 **拒绝开 direct-tcpip 通道**
（`ChannelException(1, 'Administratively prohibited')`）—— 因为那个身份没有被授权到目标机。
（教训：**"认证通过"不等于"被授权"**；两轮误判都源于只看了失败的最外一层。）
"""
import io
import os
import shutil
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
TARGET = os.path.join(HERE, "ssh_session.py")

OLD = '''        self.jump_user, self.jump_pass = rec["token"].split(":", 1)   # jt_…:密码'''

NEW = '''        # ★ 2026-09-22 实测（OpenSSH -vv 的 implicit ProxyCommand 原文）：
        #   `ssh -l jt_TOKEN:TOKENPASS -p 2234 -W [%h]:%p 113.47.8.48` ⇒ **整串 `jt_…:…` 就是用户名**。
        #   OpenSSH 的 `-J` 只接受 `[user@]host[:port]`，故 `-J a:b@host` 里的 `a:b` 全算用户名。
        #   我第一版按 ':' 拆开只取前半段 ⇒ 网关 auth_none **通过**但**拒绝开通道**
        #   （`Administratively prohibited`）：**认证通过 ≠ 被授权**。
        self.jump_user = rec["token"]''' 

src = io.open(TARGET, encoding="utf-8").read()
if 'self.jump_user = rec["token"]' in src:
    print("PATCH_SKIP 已经打过补丁")
    sys.exit(0)
if OLD not in src:
    print("PATCH_FAIL 找不到待替换的行")
    sys.exit(2)
bak = TARGET + ".bak_user_%s" % time.strftime("%Y%m%d_%H%M%S")
shutil.copy2(TARGET, bak)
io.open(TARGET, "w", encoding="utf-8").write(src.replace(OLD, NEW, 1))
back = io.open(TARGET, encoding="utf-8").read()
assert 'self.jump_user = rec["token"]' in back
print("PATCH_USER_OK 跳板用户名改为整串；备份=%s" % bak)
