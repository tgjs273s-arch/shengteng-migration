# 本机沙箱垫片 —— 为什么在本机跑闸门要加 `PYTHONPATH`（坑 214）

> **一句话**：`python prepush_check.py` 在本机**必须**带垫片才是真实结果；
> 不带垫片看到的 6 个 FAIL **是宿主造成的，不是交付物坏了**。

---

## 1. 现象

新会话按交接 §6 跑闸门，得到：

```
合计：现存 25 条（PASS=19 FAIL=6）
RELEASE_BLOCKED
```

而交付物自检是好的：

```
SAFE_PACK_VERIFY_OK items=122      （zip / 两处解包副本 / 活副本，逐文件哈希 0 不一致）
PITFALL_CHECK_OK items=214
EVIDENCE_MANIFEST_OK items=97
```

**6 个 FAIL 全部集中在"自检脚本用到 `tempfile`"的那几个闸门**（G2/G3/G9/G24/G5/G6）。

## 2. 根因（已用对照实验拆到"变量唯一"）

在本机这个 DSH Windows 沙箱里，**目录的 mode 决定它能不能被写**：

| 建目录方式 | 建目录 | 往里写文件 |
|---|---|---|
| `os.mkdir(p, 0o700)` | OK | ❌ `PermissionError` |
| `os.mkdir(p)`（默认） | OK | ✅ |
| `os.mkdir(p, 0o755)` / `0o777` | OK | ✅ |
| PowerShell `New-Item -ItemType Directory` | OK | ✅ |

而 `tempfile.mkdtemp()` 内部**恰好**是 `_os.mkdir(file, 0o700)`
⇒ **命中率 100%**，凡是 `mkdtemp()` 之后再写文件的代码，在本机必然炸。

典型签名（6 个闸门里都是同一个）：

```
PermissionError: [Errno 13] Permission denied:
  'C:\Users\...\AppData\Local\Temp\dsh-*\tmp12kz27lg\output_llava_coco_data.json'
PermissionError: [WinError 5] 拒绝访问。:
  'C:\Users\...\AppData\Local\Temp\dsh-*\p59orch_wu8xdzco\ab'
```

**两个被实验推翻的错误假设**（留在这里，免得下次再猜一遍）：
1. ❌「沙箱 TEMP 不可写」—— PowerShell 往 `%TEMP%` 直接写文件**成功**；
2. ❌「换个 TEMP 目录就好」—— 把 `TEMP`/`TMP` 重定向到工作区后，6 个闸门**依然全红**。

所以**唯一**的变量是 mode，不是路径。

## 3. 用法

```powershell
# ① 正确跑法（本机）——带垫片
$env:PYTHONPATH = "C:\Users\HUAWEI\Desktop\转交给codex的内容\04_核心资产\工具_重建\_local_shim"
python prepush_check.py
# 期望：合计：现存 25 条（PASS=25 FAIL=0） / AUTOMATED_OK_MANUAL_OPEN

# ② 对照跑法（证明差异来自宿主，而不是交付物）
Remove-Item Env:PYTHONPATH
python prepush_check.py
# 期望：PASS=19 FAIL=6，且 6 个 FAIL 的 stderr 都是同一个 PermissionError
```

`_local_shim\sitecustomize.py` 由 `PYTHONPATH` 注入，**解释器启动时**生效，
所以闸门派生的**每个子进程**都带上补丁（在父进程里打补丁是没用的）。

## 4. 规矩（坑 214 落地的处置）

1. ⛔ **不许**为了让本机变绿去改交付物 —— 这是宿主行为，交付物没有缺陷。
2. ✅ 新会话开场若"闸门数与交接不符"：**先手工复跑那几个失败的闸门，读 stderr 找"共同签名"**，
   再决定是查交付物还是查宿主；**不许**直接按"交付物回归"处理。
3. ✅ 与坑 213 的 `sitecustomize.py` **区分**：213 那个是**注入被测机制**（torch 确定性开关），
   本目录这个是**修宿主文件系统行为**；用途不同，**不得互相套用**。
4. ✅ 在**别的宿主**（例如正常的 Linux 开发机、或交付评审环境）上跑，**不需要**这个垫片；
   那里若也出现这 6 个 FAIL，才说明真的是代码问题。

## 4. ★ 附带后果：`0o700` 目录是**永久锁死**的（连删都删不掉）

这不是"读不了"，是**连 ACL 都查不了、连删都删不掉**。实测（对 `mkdtemp` 建出来的目录）：

```
Remove-Item -Recurse -Force  -> Access to the path ... is denied
cmd /c rmdir /s /q           -> Access is denied
icacls <dir>                 -> Access is denied      （连"读权限"都没有）
Get-Acl  <dir>               -> UnauthorizedAccessException
```

⇒ **每跑一次不带垫片的闸门，就在磁盘上留下若干个永久删不掉的空目录**（它们由 `tempfile` 在
TEMP/工作区里创建）。本次排查在 `C:\Users\HUAWEI\Desktop\_tmp_gatecheck\` 留下 **11 个**
这样的目录，用普通手段**无法回收**；要清掉需要管理员先夺回所有权：

```powershell
# 需要**管理员** PowerShell（普通会话做不了）
takeown /f "C:\Users\HUAWEI\Desktop\_tmp_gatecheck" /r /d y
icacls  "C:\Users\HUAWEI\Desktop\_tmp_gatecheck" /grant "$env:USERNAME:(F)" /t
Remove-Item "C:\Users\HUAWEI\Desktop\_tmp_gatecheck" -Recurse -Force
```

**所以"带垫片跑"不只是让闸门变绿，它还顺带避免了制造这些垃圾目录**（垫片建的是默认 mode 目录，
可正常读写删）。这条也是"宿主问题不要记到交付物账上"的另一个理由。

## 5. 附：本目录里的复算输入

`null_summary_重建件_仅本机复算.json` —— ★ **不是远端原件**，是依据
`protocols/prefetch_depth_20260922.json` 的 `noise_floor` 记录
（批次3 = 731.6 ms、批次4 = 716.5 ms ⇒ `spread_pct = 2.11`）**重建**出来的复算输入，
用于本地跑通 `62_reportability.py` 的端到端验证。**引用时必须注明是重建件**；
原始 `null_summary.json` 未随包归档（见 `docs/优化候选评估固定流程_20260922.md` §5 第 2 条）。

端到端复算命令（**cwd = Skill 目录** `04_核心资产\qwen35-ascend-migrator_整合版`；
路径已实测跑通 —— 注意是 `..\` 不是 `..\..\`）：

```powershell
python scripts\62_reportability.py `
  --noise "..\工具_重建\_local_shim\null_summary_重建件_仅本机复算.json" `
  --ab "..\远端证据_20260922\AB_AVSB_20260921__summary.json"
# 实测输出：
#   P62_INPUT  floor_pct=2.110 (noise_runs=2) effect_pct=19.6062 pairs=3 official_comparable=False
#   P62_CI95   [18.3235, 20.8889]
#   P62_VERDICT REPORTABLE reason=OK
#   P62_FASTER_ARM A 臂更快（口径 savings_b）      <- 与"配置 A 相对 B 省时 19.606%"一致
#   退出码 0
```
