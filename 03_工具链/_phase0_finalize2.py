# -*- coding: utf-8 -*-
r"""_phase0_finalize2.py —— 重跑阶段 0 收尾（上一版因"双引号里套双引号"语法错未执行）

★ 这次把三条坑的行文本一律放进**三引号字符串**（`'''`），彻底避开引号自伤——
  今天已经因此栽了三次（`failure_injection.py`、`_phase0_finalize.py`、以及更早的 PowerShell 嵌套引号）。
"""
import io
import os
import subprocess
import sys

TOOLS = r"C:\Users\HUAWEI\Desktop\转交给codex的内容\04_核心资产\工具_重建"
SKILL = r"C:\Users\HUAWEI\Desktop\转交给codex的内容\04_核心资产\qwen35-ascend-migrator_整合版"
PY = sys.executable

# ---------------- ① 账本降级 ----------------
P = os.path.join(SKILL, "config", "versions.lock")
s = io.open(P, encoding="utf-8").read()
anchor = "来源未判定，已如实留档。"
add = ("来源未判定，已如实留档。★★ **2026-09-22 按外部复核降级**："
       "『只清元数据 ⇒ 运行时字节未变 ⇒ 此前 A/B 仍有效、无需重跑』这句话**收回** —— "
       "历史 A/B 只保留在**当时实测环境**的口径下；**尚缺完整哈希归档与修复后的运行验证**，"
       "不能据此保证新环境行为等价（删元数据可能影响版本查询 / 插件入口发现 / 依赖解析）。"
       "`libproton.so` 另注：**是否进入本训练路径亦未验证**（原『训练不用』为无证据表述，已删）。")
if anchor in s and "2026-09-22 按外部复核降级" not in s:
    io.open(P, "w", encoding="utf-8", newline="").write(s.replace(anchor, add, 1))
    print("① 账本：triton 条目已追加降级语句")
else:
    print("① 账本：锚点缺失或已降级 ⇒ 跳过（需人工核对）")

# ---------------- ② 三条坑（三引号，避免引号自伤） ----------------
ROWS = {}

ROWS[182] = '''| **182** | ★★★ **我"修 traceback"时把打包器改成了"没有回滚点也照删" —— 我自己引入的 P0**。`safe_pack_sync.py` 首次 `--apply` 因备份目录同名碰撞抛 `FileExistsError`；我把 `backup_dir` 改成**失败只打印并返回 None**，却**没改调用方**：`unpack_replace` 拿到 None 后**照样 `shutil.rmtree(target_dir)`**，源码注释甚至写着「删除流程继续」⇒ **没有回滚能力的不可逆删除**（解压或校验再失败，旧目录永久丢失，正是坑 180 的形态）。外部复核用替身函数在内存里复现：**备份失败后删除操作仍被调用** | ① **我只修了报错点，没修"报错意味着什么"**：把"抛异常"改成"返回 None"，等于把失败降级成**一个值**，而调用方是否检查它，我没看（坑 147/153 同族：**修一条路径不问对称的那条**）；② 更深一层：**"备份失败"= 失去回滚能力，而失去回滚能力就不得执行不可逆操作** —— 这个推理我没做，反而把"继续"当默认；③ 我还在注释里把它**写成正常行为**（"删除流程继续…请知悉"）⇒ **注释替缺陷开脱**，比缺陷更坏（下一个人会以为是设计）；④ 与坑 180 是**同一错误的两次出现**：上次是"跑没读过的 rmtree 脚本"，这次是"自己写的 rmtree 没有回滚点" | ① 判据改成**可执行的不变量**：**任何一步失败，旧工件必须逐字节不变**；② `bk is None` 且目标存在 ⇒ **立即中止并返回 False**；③ 删掉自我开脱的注释，改为「**调用方将因此中止，不再删除**」；④ 新增 `failure_injection.py`：注入**备份失败 / 解压失败 / 校验失败**三类故障，逐例断言旧目录哈希不变，并带**正对照**（证明那些拒绝不是"它本来就不工作"）| 现场：`FileExistsError`（`%H%M%S` 秒级命名无唯一后缀）→ 唯一序号后缀；修复后 `FAILURE_INJECTION_OK cases=7 failed=0`（①`中止：备份失败而目标已存在 ⇒ 拒绝删除`+旧目录逐字节不变 ✓；②解压失败 ⇒ 备份点回滚 + 哈希不变 ✓；③校验失败 ⇒ 同上 ✓；正对照 rc=True ✓）；测试全在沙箱，未触碰真实交付物 |'''

ROWS[183] = '''| **183** | ★★★ **闸门在"工具根本没运行"时依然报 OK —— 我用退出码/子串当判据，把 fail-closed 写成了 fail-open**。外部复核用**纯内存坏例**证明两处：① `G23`（pyflakes 未定义名扫描）只 grep 输出里的 `undefined name` 字面量，**完全不看 rc 与 stderr** ⇒ pyflakes **根本没装**（`No module named pyflakes`）时 `undef=[]` ⇒ 报 **`G23_OK`**；② `G5` 的 `_SELFTEST_OK` 分支写成 `passed = bool(mm) and "Traceback" not in out`，而 `mm` 来自 `re.search(r"[A-Z0-9_]+_OK")` ⇒ **rc=1、先打印 `A_CASE_OK`、后打印 `VERIFY_FAIL case_ok=12/13` 也判 PASS**。⇒ 于是"18/18 全通过"里至少两条是**误放行** | ① 我把判据写成"**输出里有没有某个字面量**"，而它要判的是三层不同的东西：**工具是否正常运行 / 是否覆盖目标对象 / 目标是否满足要求**；前两层失败时第三层无从谈起，我的实现却让三者**退化成同一件事**（坑 176/177/178/181 同族：**判据口径没对样本**）；② `*_OK` 这类**宽松子串**是天然的假阳性来源：输出里更早出现的任何片段都能"救回"整体失败；③ 最严重的不是"该看 rc 还是文本"，而是**我允许一个检查在无法运行时返回"通过"** —— 直接违反我自己写下的 fail-closed；④ 我此前刚把"永远红"当反面教材，结果走到了另一个极端（**宁可绿灯**）| ① 抽出**三层状态**：工具缺失/无法启动/零样本/解析失败一律 **`ERROR`**，绝不 PASS；`rc≠0` 却出现期望标记 ⇒ **判 FAIL**（局部用例过 ≠ 整体通过）；② 每条闸门**各声明精确期望标记**（`FINGERPRINT_OK`/`PROFILE_SELFTEST_OK`/`SAFE_PACK_VERIFY_OK`…），取消一切模糊 `*_OK` 匹配；③ 新增 `gate_selftest.py` 固化坏例：pyflakes 无法启动 ⇒ ERROR、零样本 ⇒ ERROR、rc=1+标记 ⇒ FAIL、后端缺失 ⇒ ERROR，并加**正对照**；④ G22 从 `--apply` 改为**只读** `--verify-only`（检查命令不得写文件）| `GATE_SELFTEST_OK cases=5 failed=0`（`ERROR(工具缺失)`/`ERROR(零样本)`/`run_gate=False`/正对照 `run_gate=True`）；G22 只读模式上线**当天抓到真漂移**：我改了 `versions.lock` 却未重打包 ⇒ `SAFE_PACK_VERIFY_FAIL 活副本与 zip 不一致（缺 0 / 不同 1）`，重打包后 `SAFE_PACK_VERIFY_OK items=108` |'''

ROWS[184] = '''| **184** | ★★ **我把工具输出的 `B(first)` 当成"均值"写进了对外材料**。外部复核直接指出：② 的 `false` 臂应是 **562.55 ms**，而我写的是 **557.0**。查产物：`ab_skipgdn` 两次运行为 **556.95 / 568.15**，均值 562.55；557.0 是工具 `P59_RESULT` 行里的 `B(first)=557.0`（**第一次运行的窗口中位数**）| ① 我把"一行输出里最好抄的那个数"当成了"这个臂的统计量" —— 两者不是同一个东西，而工具**已经把区分写在字段名里**（`B(first)`），我却按习惯读成了"B 的值"（坑 163/176 同族：**字段名是断言不是事实**）；② 危害方向明确：我取了**偏快**的那次作代表值，会让②的"false 更慢"看起来更小 ⇒ 这是**朝有利方向**的读数错误，最不容易被自己质疑；③ 更值得记的是：这份材料是**给外部复核的**，等于把未核对的口径直接交给别人去发现 | ① 对外材料里每个数字必须标注**它是什么统计量**（单次/均值/中位数）与**来自哪个字段**；② 只用聚合量，不用"某一次"；③ 把可疑的口径差异主动写进复核清单（这次正是复核抓到的）| 产物 `aftermab_20260921_143648/ab_skipgdn/summary.json`：A=true `516.5 / 533.3`、B=false `556.95 / 568.15` ⇒ 均值 **524.90 / 562.55**；决策记录已另存 v2 并写明「v1 误把 `B(first)=557.0` 当均值，已更正」|'''

rowdir = os.path.join(TOOLS, "_rows")
os.makedirs(rowdir, exist_ok=True)
rc_all = 0
for num in sorted(ROWS):
    rf = os.path.join(rowdir, "_row%d.md" % num)
    io.open(rf, "w", encoding="utf-8", newline="").write(ROWS[num].strip() + "\n")
    r = subprocess.run([PY, os.path.join(TOOLS, "_append_pitfall.py"), "--row", rf,
                        "--expect-max", str(num)], capture_output=True, text=True,
                       encoding="utf-8", errors="replace")
    lines = ((r.stdout or "") + (r.stderr or "")).strip().splitlines()
    print("② 追加 %d：rc=%d | %s" % (num, r.returncode, lines[-1] if lines else ""))
    rc_all |= r.returncode
print("PHASE0_FINALIZE2_%s" % ("OK" if rc_all == 0 else "FAIL"))
sys.exit(rc_all)
