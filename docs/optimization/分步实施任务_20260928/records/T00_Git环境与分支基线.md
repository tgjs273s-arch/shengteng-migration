# T00 Git 环境与分支基线（执行准备记录）

记录日期：2026-09-28（推送与发布闸门复核为同日补记）。范围：git 环境整理、分支建立与首次推送。**未改动任何脚本、配置或证据在仓库中的内容**，未改写历史。

## 结论

- 实施分支：`feat/qwen35-mindspeed-migration`，自 `main` 的 `ed1be28` 创建并已切换；后续 T01–T15 在此分支提交，交接记录引用本分支名与提交哈希。
- 工作区干净：`git status --short`、`git diff --stat`、`git diff --cached --stat` 均无输出。
- 已推送远端（2026-09-28，用户明确要求）：`origin/main` = `ed1be28`、`origin/feat/qwen35-mindspeed-migration` = `2022158`，服务端 `git ls-remote` 与本地哈希逐一核对一致，领先/落后均为 0/0，无 force push。
- 规划文档已入库：`AGENTS.md`、`docs/optimization/方向复核与整改清单_20260928.md`、`docs/optimization/分步实施任务_20260928/`，以及两个既有文档（能力清单、项目现状评估与优化计划）的本轮改动。
- 废弃的 `AGENT.md` 仅原样保留，未读取、未修改。

## 本次提交（main）

| 提交 | 内容 |
| --- | --- |
| `613b410` | 新增 `AGENTS.md` 项目协作指南 |
| `88f68e5` | 落盘方向复核整改清单与 T01–T15 分步实施任务（含能力清单、优化计划第十一节更新） |
| `ed1be28` | 新增 `.gitattributes`，统一文本换行符为 LF |

## 换行符策略与依据

问题：仓库此前没有 `.gitattributes` 且 `core.autocrlf=true`，检出时把全部文本写成 CRLF。147 个跟踪的 `.sh` 在磁盘上带 `\r`，在 WSL/Linux 直接执行会报 `\r: command not found`；工作区字节也与仓库中已存的 LF 内容不一致。

处理：

- `.gitattributes`：`* text=auto eol=lf`；`*.bat`/`*.cmd`/`*.ps1` 保持 `eol=crlf`；常见二进制与模型权重类型显式声明为 `binary`。
- 仓库本地配置（只写 `.git/config`，未改全局）：`core.autocrlf=false`、`core.quotepath=false`。属性成为换行符的唯一事实来源。
- 工作区按索引内容重写为 LF（先删待重写文件再 `git checkout-index`；该版本 git 的 `checkout-index -f` 对已存在文件会静默跳过）。

结果：`git ls-files --eol` 为 769 个 `i/lf w/lf`、5 个 `i/lf w/crlf`（仅 `.ps1`）、59 个二进制、4 个 `i/none`。索引与 HEAD 的 blob 哈希未变。

操作提示：重写工作区后，索引里记录的仍是旧 CRLF 尺寸，`git status` 会短暂把约 746 个文件报成 modified（`porcelain=v2` 显示 `.M` 且 HEAD 与 index 哈希相同）。`git add -u .` 重录 stat 后即恢复干净，blob 无变化；这不是内容改动。

## 验证证据（Windows 本地离线，不代表 NPU/实机）

| 命令 | 结果 |
| --- | --- |
| `git ls-files --eol` | 见上，`.sh`/`.py`/`.md`/`.yaml` 全部 `w/lf` |
| `git hash-object --path=<路径> -- <文件>` 对比 `git rev-parse HEAD:<文件>` | 一致（示例 `run_from_zero.sh` → `ad8705e5…`） |
| `git status --short` / `git diff --stat` / `git diff --cached --stat` | 均无输出 |
| `python 03_工具链/prepush_check.py --show-paths` | exit 0，路径解析正常 |
| `python 02_活副本_Skill/scripts/70_judge.py --selftest` | exit 0，`WORLD_RESOLVE_SELFTEST_OK cases=9` |
| `python -m unittest discover -s 02_活副本_Skill/tests -p test_stage_contracts.py` | Ran 14 tests，OK |
| `python -m unittest discover -s 02_活副本_Skill/tests -p test_judge_wrapper.py` | Ran 5 tests，OK |
| `python -m unittest discover -s 03_工具链/tests -p "test_*.py"` | Ran 9 tests，OK |

Linux/WSL 的完整离线回归（`PIPELINE_TEST_BASH` 全量）本次未运行，留给后续任务按范围执行。

## 已知问题：证据清单与换行符（非本次引入）

`04_远端证据/_manifest.json` 共 168 条。整理前（磁盘 CRLF）只匹配 66 条、**102 条哈希不符**；整理为 LF 后匹配 147 条、不符降为 **21 条**。这 21 条经逐条验证**全部只与 CRLF 形式匹配**（两种形式都不匹配的为 0 条），证明清单里它们的哈希是在 CRLF 工作区上算出来的，例：`A_recommended.yaml` 清单 size 4585（CRLF）而仓库中 LF 为 4421；`B_fallback.yaml` 同理。

结论：这是清单内部对换行符的口径不一致，任何 LF 检出都会不匹配这 21 条，属既有证据侧问题。本次整理把不符数量从 102 条降到 21 条，没有新建问题。

处理口径：不改写历史清单、不重建哈希掩盖差异。T01/T11 复核时按此登记说明；如需恢复这些文件的可核验性，应作为一次显式更正记录原因与新证据。

## 发布闸门状态（本轮实测，非本次引入）

仓库自带 `python 03_工具链/prepush_check.py` 全量结果为 **PASS=23 / FAIL=3**，总体 `RELEASE_BLOCKED`（阻断发布；不影响只读分析与分支推送）：

| 闸门 | 结果 | 实测原因 |
| --- | --- | --- |
| G12 证据逐文件哈希表 | FAIL | `evidence_manifest.py --verify`：清单=168 现有=168 缺失=0 多出=0 **哈希不符=21**，即上节 CRLF 口径问题（整理前为 102 条不符） |
| G22 三方一致性（zip/两处解包副本/活副本） | FAIL | `safe_pack_sync.py --verify-only`：`zip 数=0（应为 1）`，本机无交付 zip（`01_交付物/**/*.zip` 被 gitignore），无从比对 |
| G8 交付目录结构 | FAIL | `06_Skill 下 zip 数=0（应为 1）`，同上，交付包未在本机生成 |

G22/G8 与本轮换行符整理无关，属交付资产缺失；三项均应在 T11/T15 的交付与复核环节处理，不能以"闸门没跑"或"历史全绿"代替。

## 未做与回退

- 未 rebase、未使用 filter-branch、未 force push；推送为纯快进（`main` 09075d0→ed1be28）与新增分支，历史提交未被改写。
- 未修改 `01_交付物/` 与 `04_远端证据/` 在仓库中的内容（index/HEAD blob 未变），`_manifest.json` 本身未被改写；工作区文本统一为 LF 属本次策略。
- 回退：`git switch main` 即回到整理后基线；若需回到整理前的 CRLF 检出，可临时设 `core.autocrlf=true` 后重新检出，但会重新引入 `.sh` 带 `\r` 的问题。
