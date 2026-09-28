# T00 Git 环境与分支基线（执行准备记录）

记录日期：2026-09-28。范围：仅 git 环境整理与分支建立。**未改动任何脚本、配置或证据的内容字节**，未 push、未改写历史。

## 结论

- 实施分支：`feat/qwen35-mindspeed-migration`，自 `main` 的 `ed1be28` 创建并已切换；后续 T01–T15 在此分支提交，交接记录引用本分支名与提交哈希。
- 工作区干净：`git status --short`、`git diff --stat`、`git diff --cached --stat` 均无输出。
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

`04_远端证据/_manifest.json` 共 168 条。整理前（磁盘 CRLF）只匹配 66 条；整理为 LF 工作区后匹配 147 条，仍有 21 条不匹配。

原因：这 21 条的清单哈希是按 CRLF 内容计算的，其中 `A_recommended.yaml` 清单 size 4585（CRLF）而仓库中 LF 为 4421；`B_fallback.yaml` 同理。也就是说清单内部对换行符的口径不一致，任何 LF 检出都会不匹配这 21 条，属既有证据侧问题。

处理口径：不改写历史清单、不重建哈希掩盖差异。T01/T11 复核时按此登记说明；如需恢复这些文件的可核验性，应作为一次显式更正记录原因与新证据。

## 未做与回退

- 未 push、未 rebase、未使用 filter-branch，历史提交未被改写。
- 未修改 `01_交付物/` 与 `04_远端证据/` 的内容字节，未做交付同步。
- 回退：`git switch main` 即回到整理后基线；若需回到整理前的 CRLF 检出，可临时设 `core.autocrlf=true` 后重新检出，但会重新引入 `.sh` 带 `\r` 的问题。
