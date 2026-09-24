# -*- coding: utf-8 -*-
r"""_fix_pit_197.py —— 把 197 行重写为**规范 5 列单行**（表格列数不足的修复）

背景：197 我写成了 4 列（`| 编号 | 坑 | 为什么 | 怎么办(含证据) |`），而表格是 5 列
（`编号/坑/为什么/怎么办/现场证据`）⇒ 校验报"列数不足（5 个 |）"。
教训同上一条（写入端只校验行首）：**格式判据必须把列数也管住**。
本脚本按 5 列重建 197，替换原行，并回读校验"行首/行尾/列数/编号连续"。
"""
import io
import os
import re
import shutil
import time

TABLE = r"C:\Users\HUAWEI\Desktop\转交给codex的内容\04_核心资产\qwen35-ascend-migrator_整合版\docs\PITFALLS_坑表.md"

PIT = ("★★ **丢失模块的接口，我没有猜函数名，而是写了一个『自描述 shim』，5 秒内问出了完整签名与配置结构**。"
       "做法：把缺失的 `token_broker.py` 写成**探针** —— 模块级 `__getattr__`（Python 3.7+）在**任何属性被访问**时记录名字，"
       "并返回一个『记录调用参数后抛错』的可调用对象；探针只写日志（`_run/shim_probe.log`），**不实现任何功能**，"
       "因此绝不会让会话『假装连上』。**首次调用即拿到全部答案**："
       "`CALL cli_need args=(8791, '<BROKER_SECRET>') kwargs={'timeout': 45, 'quiet': True, 'cfg': {...}}` —— "
       "而且 `cfg` **自带了整个设计**：store 路径、`hivelab{platform, account, page, env_in_use, ssh_direct_button='SSH 直连', "
       "ttl_note, capture_method='页面内点按钮 + 剪贴板（不逆向接口、不保存 Cookie）', capture_tools=[油猴脚本每 4 分钟自动点一次并把命令写剪贴板, F12 Console 版]}`、"
       "`_note: endpoint 未配置 ⇒ 自动获取由 capture_tools + token_agent.py --watch-clipboard 完成`。"
       "随后再用 HTTP 探活拿到经纪的**完整端点**（`GET /` 自描述）：`POST /need/<secret>` 要 token、"
       "`POST /token/<secret>` 送 token、`GET /next?wait=N` 给页面长轮询、`GET /status` 返回 "
       "`mint_seq/last_saved_at/tokens_saved/page_polls/store`")
WHY = ("① 面对『接口丢了』，**第一反应可以是『让它自己说出来』，而不是『我猜一个』**：调用方（会话守护）**就在运行中**，"
       "它每隔 5 秒就会用真实参数撞一次 —— 这是最权威的规格来源，比任何文档都准；"
       "② 关键设计是**探针必须『只记录、不实现』**：若 shim 返回一个看似可用的假 token，会话会『看起来连上』，"
       "把问题推迟成更难查的形态（与坑 33/34「静默当通过」同族）；"
       "③ 这也纠正了我对『不许猜』的机械理解：**不猜 ≠ 什么都不做**，而是『用可观测的实验去问』，这次问的成本是 5 秒；"
       "④ `cfg` 里的 `why_not_api`（不逆向接口、不保存 Cookie、按可见文本定位）说明原设计**刻意避开**凭证依赖 —— "
       "恢复时必须保留这条边界，不能为了『自动』改成存 Cookie")
HOW = ("① 探针文件：`_ops/token_broker.py`（先以 `token_broker_shim.py` 落盘再改名，避开写工具的观测缓存冲突）；"
       "② 按恢复出的规格**重写可重启实现**：`serve()` + `Broker` + `cli_need()` + `--put` 手动送 token，"
       "`allow_reuse_address=False` 沿用坑 171 的教训，返回对象做成 dict+属性双用以免猜返回约定；"
       "③ 立规矩（并落到工具里）：**格式判据要同时校验行首与行尾/列数** —— 本次 `_append_pitfall.py` 只校验行首，"
       "于是半截行照样写进坑表；④ 仍需重建 `ssh_session.py`/`token_agent.py`（见坑 196）")
EV = ("① `_run/shim_probe.log`：`10:00:01 SHIM_LOADED pid=17916 cwd=D:\\昇腾项目` → `ACCESS cli_need` → "
      "`CALL cli_need args=(…) kwargs={…}`（每 5 秒一条，共 3 条）；"
      "② HTTP 探活：`GET / -> {\"ok\": true, \"service\": \"token_broker\", \"hint\": \"POST /token/<secret> 送 token；GET /next 长轮询；POST /need/<secret> 要 token\"}`、"
      "`POST /need/<BROKER_SECRET> -> {\"ok\": true, \"seq\": 1}`、`GET /status -> {\"mint_seq\": 1, \"tokens_saved\": 0, \"page_polls\": 1}`；"
      "③ 重写实现：`_ops/token_broker.py`（编译通过）；④ 假 token 端到端测试被活经纪以 `HTTP 400` 拒绝（其校验严于 store 字段本身）")

row = "| **197** | %s | %s | %s | %s |" % (PIT, WHY, HOW, EV)
assert row.count("|") == 6 and "\n" not in row, "列数不对：%d" % row.count("|")

lines = io.open(TABLE, encoding="utf-8").read().splitlines()
idx = [i for i, l in enumerate(lines) if l.startswith("| **197** |")]
if len(idx) != 1:
    raise SystemExit("FIX_FAIL 期望恰好 1 行 197，实际 %d" % len(idx))
bak = TABLE + ".bak_%s" % time.strftime("%Y%m%d_%H%M%S")
shutil.copy2(TABLE, bak)
lines[idx[0]] = row
io.open(TABLE, "w", encoding="utf-8").write("\n".join(lines) + "\n")

back = io.open(TABLE, encoding="utf-8").read().splitlines()
ids, bad = [], []
for l in back:
    m = re.match(r"^\|\s*\*\*(\d+)\*\*\s*\|", l)
    if m:
        ids.append(int(m.group(1)))
        if not l.rstrip().endswith("|") or l.count("|") < 6:
            bad.append((m.group(1), l.count("|"), l.rstrip().endswith("|")))
if bad:
    raise SystemExit("FIX_FAIL 仍有列数不足的行：%s" % bad[:5])
if ids != list(range(1, len(ids) + 1)):
    raise SystemExit("FIX_FAIL 编号不连续：%s" % ids[-5:])
print("FIX197_OK 条数=%d 编号=1..%d 备份=%s" % (len(ids), ids[-1], bak))
