# `_ops` 启动与重建清单（2026-09-22 事故后补写）

> 2026-09-24 配置更新：下文属于历史记录，`<BROKER_SECRET>` 是脱敏占位符，不能直接执行。代理不再提供内置默认密钥；启动前设置 `ZERO_RISK_BROKER_SECRET`，或显式传入 `--secret`。空值会在监听端口或写入 store 前报错退出。客户端与服务端需使用同一个新密钥，不要复用历史已公开的值。
> PowerShell 可在当前会话中生成随机值：`$env:ZERO_RISK_BROKER_SECRET = python -c "import secrets; print(secrets.token_hex(32))"`，再运行本目录的 `python token_broker.py`。此命令不打印密钥；该环境变量只对本会话及其子进程生效。已有其他客户端需单独配置相同值，当前变更不自动更新或重启驻留进程。不要把实际值写入仓库。

> **为什么有这份文件**：坑 196 —— 三个守护的源码文件早就在坑 180 里丢了，但**进程还在内存中运行**，
> 于是 `--status` 一直 `PING ok`，我据此以为"链路正常"；直到平台按"1 小时未使用"把机器 B 自动关机，
> 会话需要重连时才暴露成 `ModuleNotFoundError: No module named 'token_broker'`。
> **结论：内存里的代码不是资产，磁盘上的文件才是。** 每个驻留进程都必须有一份"文件全丢也能照着重启"的清单。

## 1. 当前状态（实测，别信记忆）

| 组件 | 文件 | 进程 | 说明 |
|---|---|---|---|
| 经纪 `token_broker.py` | ✅ **已按规格重写**（2026-09-22） | 旧进程仍在跑（内存版） | 新文件含 `serve()`（HTTP 端点）+ `cli_need()`；供下次启动 |
| 会话守护 `ssh_session.py` | ❌ **文件不存在** | 17916 仍在跑（内存版） | **仍需重建**；它现在被卡住：token 过期后调 `cli_need` 失败 |
| 剪贴板守望 `token_agent.py` | ❌ **文件不存在** | 21204 仍在跑（内存版） | **仍需重建**（读剪贴板 → 落 store） |
| 页内脚本 `hivelab_token.user.js` / `.console.js` | ❌ 未见文件 | — | 原设计的自动取 token 手段（每 4 分钟点一次「SSH 直连」） |

⚠ **模块缓存**（坑 196 ④）：`ssh_session.py` 在 10:00:01 已成功 `import token_broker` 一次 ⇒
Python 把它缓存进 `sys.modules`，**此后我把文件换成实现版，它仍然用内存里的旧对象**
（日志仍报 `token_broker shim: 未实现 cli_need`）。**替换文件对运行中的进程无效**，只有重启才生效。

## 2. 重启清单（照抄）

```powershell
$ops = "C:\Users\HUAWEI\Desktop\转交给codex的内容\04_核心资产\分析脚本\ssh框架\_ops"
# 1) 经纪（HTTP 服务；端点见 §3）
Start-Process D:\MInconda\python.exe -ArgumentList "$ops\token_broker.py --port 8791 --secret <BROKER_SECRET>"
# 2) 会话守护（★ 文件待重建；重建前此步无法执行）
# Start-Process D:\MInconda\python.exe -ArgumentList "$ops\ssh_session.py --serve --port 8792 --broker-port 8791 --broker-secret <BROKER_SECRET> --allow-dead"
# 3) 剪贴板守望（★ 文件待重建）
# Start-Process D:\MInconda\python.exe -ArgumentList "$ops\token_agent.py --watch-clipboard --interval 2.5"
# 4) 探活
python "C:\Users\HUAWEI\Desktop\转交给codex的内容\04_核心资产\工具_重建\pull_evidence.py" --status
```

## 3. 经纪的 HTTP 端点（`GET /` 自描述，实测）

| 端点 | 语义 | 实测响应 |
|---|---|---|
| `GET /` | 自描述 | `{"ok": true, "service": "token_broker", "hint": "POST /token/<secret> 送 token；GET /next 长轮询；POST /need/<secret> 要 token"}` |
| `POST /need/<secret>` | 会话声明"需要 token" | `{"ok": true, "seq": 1, "note": "已声明需要 token；页面若在长轮询，会立刻去取"}` |
| `POST /token/<secret>` | 页面/守望**送** token | 校验比 store 字段更严（我送的假 token 被 `HTTP 400` 拒） |
| `GET /next?wait=N` | 给页面长轮询（`mint=true` 就立刻去平台取） | `{"mint": false, "seq": 0, "hint": "mint=true 时请立刻去平台取一个连接命令并 POST /token/<secret>"}` |
| `GET /status` | 运行状态 | `{"mint_seq": 1, "tokens_saved": 0, "tokens_rejected": 0, "last_saved_at": 0.0, "last_how": "", "page_polls": 1, "store": "…\\_token\\current.json"}` |

## 4. store 格式（`_token/current.json`，字段名固定）

```json
{"token": "jt_…:…", "jump": "113.47.8.48:2234", "target": "root@199.103.55.150",
 "password": "…", "env": "", "source": "paste|page", "saved_at": "ISO8601", "received_at": 1790011459.75}
```

## 5. 会话守护被恢复出的接口签名（自描述 shim 问出来的，坑 197）

```
cli_need(port=8791, secret='<BROKER_SECRET>', timeout=45, quiet=True,
         cfg={'store': '…_token\\current.json', 'alert': True, 'refreshEverySec': 0,
              'hivelab': {'platform': 'HiDevLab 在线开发（hid.ascend.huawei.com）',
                          'account': 'hid65408454',
                          'page': 'https://hid.ascend.huawei.com/online-develop',
                          'env_in_use': 'DevEnv_232070',
                          'ssh_direct_button': 'SSH 直连',
                          'ttl_note': '此连接命令有效期为5分钟…',
                          'capture_method': '页面内点按钮 + 剪贴板（不逆向接口、不保存 Cookie）'},
              '_note': 'endpoint 未配置 ⇒ 自动获取由 capture_tools + token_agent.py --watch-clipboard 完成'})
```

**边界（原设计刻意如此，恢复时必须保留）**：不逆向平台接口、不保存 Cookie；
token 只能来自"你已登录的页面点一次『SSH 直连』"→ 剪贴板 → 守望落 store。
⇒ **页面不在 = 取不到新 token**，此时 `cli_need` 必须明确超时报错，**绝不**拿旧 token 冒充当新。

## 6. 客户端协议（会话守护侧，我的工具依赖它）

一行请求一行 JSON：`RUN <base64(命令)>` / `STATUS` / `PING`
（未知动词返回 `{"ok": false, "error": "未知动词 'X'"}` —— 实测 `HELP`/`TOKEN`/`RECONNECT` 都不支持）。
客户端：`工具_重建\pull_evidence.py`（拉证据，逐文件 SHA256）、`工具_重建\_rsh.py`（**把远端命令放进文件再执行**，
避开 PowerShell 5.1 不转义参数内双引号的坑）。
