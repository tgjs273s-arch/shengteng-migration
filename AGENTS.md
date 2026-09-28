# AGENTS.md — 零秒识危项目协作指南

适用于本仓库。依据当前源码、配置、测试及项目文档整理，核对日期为 2026-09-25。

**旧 `AGENT.md`（含大小写变体 `agent.md`）已废弃，不作为指令、事实来源或变更日志，也不要继续追加内容。** 其他文档中“先读/更新 AGENT.md”的要求同样失效。本文维护稳定的项目说明和协作约定；具体改动及验证记录写入对应功能文档或 `docs/optimization/capability_inventory.md`。

## 项目定位与现状

“零秒识危”面向工业作业的 SOP 合规判断与风险识别。当前主要实现是 **Qwen3.5-0.8B 在昇腾 NPU 上的迁移、训练编排、性能诊断和证据判定工具链**，服务于 C4-AI 昇腾赛道项目。

- **AutoMigrate**：环境探测、Qwen3.5 的 AST 迁移点识别、MindSpeed-MM 适配建议和训练配置生成。当前不是任意模型的自动源码转换器。
- **PerfTune**：算子检查、训练启动、微基准、日志统计、A/B 实验、可比性及可报告性判定。真实训练调用外部 MindSpeed-MM，完整训练框架和模型权重不在本仓库。
- **DataForge / 工业场景**：主要是 `65_scene.py` 的单图推理冒烟入口和预期输出契约；数据构建、训练产物接入、真实结果留存和业务评测尚未闭环。

当前重点是消除明确缺陷、接通真实输入输出、改善可复现性，再在数据和 NPU 环境就绪后验证效果。区分“已实现”“离线验证通过”“实机运行通过”“业务/性能评测通过”；历史运行数字不能代表当前代码的效果。

## 工作位置与事实来源

| 路径 | 职责与修改边界 |
| --- | --- |
| `02_活副本_Skill/` | **开发主副本**，功能修改优先落在这里。 |
| `02_活副本_Skill/scripts/` | 主流水线、阶段入口、性能实验和诊断脚本。 |
| `02_活副本_Skill/sk04_judge/` | 配置指纹、数据身份、实测指纹、裁决及哈希链账本。 |
| `02_活副本_Skill/config/` | 环境矩阵、模板、基线声明、历史版本记录。 |
| `02_活副本_Skill/tests/` | 阶段契约、主入口编排、训练隔离及判定包装器回归。 |
| `02_活副本_Skill/refs/` | 固定版本建模参考源码，用于 AST 分析，不是训练实现。 |
| `02_活副本_Skill/examples/` | 已保存的示例日志和产物，不是当前运行输出。 |
| `02_活副本_Skill/protocols/` | 实验判据、结果和更正；引用时检查对应更正版。 |
| `03_工具链/` | 路径解析、证据校验、提交前检查及历史实验/修补脚本。 |
| `04_远端证据/` | 已归档远端实验，`_manifest.json` 记录逐文件哈希。 |
| `05_SSH框架_ops/` | SSH 会话守护与令牌代理。 |
| `01_交付物/` | 历史报告、Skill 快照、源代码包、日志和模板；**未整体同步当前代码**。 |
| `docs/optimization/` | 能力、问题清单和工程进展。 |

开始工作先看 `git status --short`，保留用户已有修改。建议阅读顺序：

1. 根目录 `README.md`，忽略其中对废弃 `AGENT.md` 的引用。
2. 涉及功能的实际脚本和测试，确认参数、输入输出、退出码和副作用。
3. `02_活副本_Skill/docs/流水线执行与续跑.md`、`docs/optimization/capability_inventory.md`，后续修复条目会更新前面的历史状态。
4. `项目现状评估与优化计划_20260923.md` **第十一节**；前面的执行方案已被替代。
5. 按问题查 `02_活副本_Skill/docs/PITFALLS_坑表.md`、判定说明和实验协议。

`00_先读我.md`、移交文档、Skill README/SKILL.md 和交付材料仍有旧路径、过强的成果声明及未接线的设计。发生矛盾时，用当前源码和可复现证据核实行为，不照搬历史 PASS、机器在线状态或作者桌面路径。`profiles/` 目前只是扩展说明，P1 规则仍内置在代码中。

## 实际执行链路

入口为 `02_活副本_Skill/scripts/run_from_zero.sh`，会切换到 Skill 根目录。单独运行阶段脚本也应从该目录出发，或显式传入路径。

| 阶段 | 脚本（相对 Skill 的 `scripts/`） | 产物与能力边界 |
| --- | --- | --- |
| PRE | `05_preflight.py` | `out/preflight/capability.json`、降级记录；检查运行前置能力。 |
| P0 | `00_probe_env.py` | `out/probe/env.json`；硬件、软件、能力与 `recommended_profile`。 |
| P1 | `10_analyze_points.py` | `out/analyze/migrate_points.json` 和报告；识别 GatedDeltaNet、全注意力、视觉 Conv3d、激活/归一化四类。 |
| P2 | `20_plan_migration.py` | `out/plan/train_config.yaml`、`migrate_plan.md`；消费 P0/P1，修改模板后回读校验再落盘。 |
| P3 | `30_verify_ops.py` | `out/verify/ops_matrix.json`；forward/shape 检查，GatedDeltaNet 仍为 `contract-only`，没有完成数值等价验证。 |
| P4 | `40_prepare_assets.py` | `out/assets/assets.json`；检查或下载 HF 权重、转换 DCP、准备 LLaVA/COCO 数据。 |
| P5 | `50_train.py` | `torchrun` 调用外部 `mindspeed_mm/fsdp/train/trainer.py`；保存训练日志及隔离运行目录。 |
| P6 | `60_bench.py` + `95_extract_series.py` | 算子微基准和训练日志窗口统计分别产出 `round_1_baseline.json`、`loss_series.csv`、`window_50_100.json`。 |
| P7 | `70_judge.py` | 调用 `sk04_judge/scripts/` 的指纹、裁决及可选账本工具，生成本次判定目录与汇总。 |

实际依赖：P2 ← P0/P1；P3、P4 ← P0；P5 ← PRE/P2/P3/P4；P6 ← P5；P7 ← P2/P4/P5。独立诊断可继续，依赖失败的阶段必须阻断。

独立工具不能因“存在”就算作主入口已经调用：

- `51_train_fp.py`、`52_replay.py`、`_batchfp.py`：逐步 batch 指纹和单步重放。
- `54`–`58`、`61`、`96`–`99` 号脚本：性能拟合、profiler、FSDP/优化器、内存通信、CPU 配额与线程诊断。
- `59_config_ab.py`：单键 A/B 及不改配置的重复实验；`63_two_config_ab.py`：带差异声明的两配置对比。
- `62_reportability.py`：可报告性和采纳判据，**当前未接入主入口**。
- `65_scene.py`：场景推理，**当前未接入主入口**；固定加载远程模型标识，真实输出只打印，`EXPECTED` 仅为示例契约。

## 必须保留的运行契约

1. **成功必须有本次证据。** 同时核对返回码、业务状态和必需产物。失败保留日志，旧文件或一句成功标记不能替代本次结果。
2. **续跑必须可溯源。** `_stage_state.py` 绑定代码/配置/参考文件摘要、入口参数、上游 attempt、输出和日志哈希。修改阶段契约时同步核对 `OUTPUTS`、`DEPS` 和主入口。
3. **部分验证不能升级为全通过。** P3 的 `contract-only` 使主入口保持 PARTIAL；允许后续训练诊断不代表算子数值正确。P1 回退自检时必须继续传递 `source_tag`。
4. **配置与运行同源。** P1 使用 `points: {类别: [{line, sym}, ...]}`；P2 兼容旧计数格式，但显式无效输入应失败。YAML 用 PyYAML 回读，路径中的中文、空格、引号、冒号、`#` 不得改变值。
5. **步数读实际配置。** 主入口 `--steps` 经 P2 写入 `training.train_iters`；直接运行 P5 时，其 `--steps` 当前仅提示。现有官方几何检查为 `GBS = mbs × gas × dp = 8`，并核对配置 dp 与运行 world；不得静默绕过，也不能推广到任意并行拓扑。
6. **P5 按次隔离。** `--log` 父目录的 `runs/p5-*` 保存 `runner.sh`、`runner.log`、`run.json`。启动会清空指定训练日志，保留历史需传新路径。同日志由 Linux `fcntl` 锁排他，不通过删除 `.lock` 文件解锁。
7. **P7 执行状态与裁决分开。** 每次写 `out/judge/attempt-*`，读取当前 `judge_summary.json` 的 `execution_state`、`attempt_dir`。子工具错误、坏 JSON、缺产物或账本追加失败不得伪装成功。`--registry` 与 `--tag` 成对提供；不得手改历史哈希链或用旧顶层文件替代本次结果。
8. **公共输出不支持并发。** 主入口的公共 `out/`、同一 P7 输出目录、同一本账本不能并发写。P5 隔离不代表解决了 NPU 调度或端口预留。

| 入口 | 退出码含义 |
| --- | --- |
| `run_from_zero.sh` | `0` 所执行/复用阶段通过；`1` 失败、超时或阻断；`2` 参数/入口错误；`3` 部分验证。 |
| `70_judge.py` | 默认 `0` 仅表示裁决产出完成，红/黄也可正常产出；`2` 输入/IO 错误；`3` 工具故障。`--gate` 时绿/黄/红为 `0/3/4`，因此 `3` 需结合汇总判断。 |
| `62_reportability.py` | `0` 达标且可采纳；`3` 达标但采纳受阻；`4` 未达标；`5` 无法判定；`2` 用法错误。 |

## 本地检查与测试

本仓库使用 Python/Bash 和 `unittest`，没有统一的活副本 `requirements.txt`/`pyproject.toml` 或通用一键安装环境。基础离线用例需要 PyYAML；真实算子、训练、场景推理另需相应框架依赖。Windows 用于编辑、静态检查及部分离线回归；完整编排与训练隔离测试在 Linux/WSL 运行。

以下从**仓库根目录**执行；Windows 编码异常时先设置 `$env:PYTHONIOENCODING='utf-8'`：

```powershell
# 确认检查目标与可选检查项
python "03_工具链/prepush_check.py" --show-paths
python "03_工具链/prepush_check.py" --list

# 无需 NPU：阶段契约、判定包装器、工具路径与认证配置
python -m unittest discover -s "02_活副本_Skill/tests" -p "test_stage_contracts.py" -v
python -m unittest discover -s "02_活副本_Skill/tests" -p "test_judge_wrapper.py" -v
python -m unittest discover -s "03_工具链/tests" -p "test_*.py" -v
python "02_活副本_Skill/scripts/70_judge.py" --selftest
```

完整 Skill 离线回归（Linux/WSL，仓库根目录）：

```bash
PIPELINE_TEST_BASH=/usr/bin/bash python3 -m unittest discover -s 02_活副本_Skill/tests -p 'test_*.py' -v
```

Windows 的 `bash.exe` 可能只是 WSL 启动入口，不能据其存在判断能直接驱动 Windows Python 测试。WSL 使用自身 Python 与依赖。测试产物保留在被忽略的 `tmp/` 下；权限错误属于验证环境问题，不能修改断言、伪造产物或照搬历史全绿记录。

按变更范围选检查，不要求每次全跑：

| 修改范围 | 检查入口 |
| --- | --- |
| P1/P2/P3 | `test_stage_contracts.py`；输出契约变更同时验证主入口。 |
| 主入口/续跑/P4 路径 | `test_pipeline_driver.py`，Linux/WSL。 |
| P5 runner/锁/退出码 | `test_execution_isolation.py`，Linux；诊断规则另有 `50_train.py --diag-selftest`。 |
| P7 | `test_judge_wrapper.py`、`70_judge.py --selftest`；必要时核查 SK04 自检夹具与路径。 |
| P0 档位 | `00_probe_env.py --selftest`，不代表硬件验证。 |
| A/B 与可报告性 | `59_config_ab.py --selftest`、`62_reportability.py --selftest`、`63_two_config_ab.py --selftest`，选择涉及项。 |
| 工具路径/认证 | `03_工具链/tests/test_project_paths.py`、`test_broker_secret.py`。 |
| 纯文档 | 检查路径、命令、事实与差异，不新增测试。 |

Skill 独立自检从 `02_活副本_Skill/` 执行。`prepush_check.py --only <ID列表>` 支持选项检查；完整检查还涉及交付快照、PDF、额外依赖及历史资料。真实版本差异导致的一致性失败应解释，不能削弱判据。`03_工具链/_local_shim/` 是历史本机兼容垫片，使用前核对其改写行为和当前必要性，不默认启用。

## 实机运行与路径

- 真实训练依赖 Linux 昇腾栈：CANN、PyTorch、torch_npu、triton-ascend、MindSpeed/MindSpeed-MM，以及模型和数据。`config/versions.lock`、`config/env_matrix.yaml` 是基线/候选配对记录，不代表当前机器或通用安装锁。
- 先核对 NPU、驱动、Python、框架版本、权重、数据及磁盘，再生成配置、启动训练。`bringup.sh` 会安装/调整环境；P4 会下载大文件和转换资产，阅读行为后在任务授权范围内执行。
- 主入口 `--quick` 主要跳过全量 COCO 下载，**仍会尝试训练**；`--no-download` 仅约束 P4，不把主入口变为只读检查，也不补齐资产。P0 探测可能执行 Triton kernel。
- `--resume` 仍重跑 PRE/P0/P4，新身份可能使下游缓存失效；`--from` 要求有效的前置记录；`--force` 不绕过前置校验。更换硬件、外部框架、模型或数据后，从 PRE 重新验证。
- 主入口 `--data-dir` 贯穿 P0/P2/P4/P7；权重另有 P4 `--model-dir`、P2 `--weight-hf/--weight-dcp`，不能假定改数据目录就迁移了全部资产。
- `03_工具链/_project_paths.py` 支持 `--root/--skill/--tools/--deliverables/--evidence` 及对应 `ZERO_RISK_*` 环境变量。显式 `--root` 重置组件默认位置，组件相对路径以 root 为基准。
- 部分实验、SSH、`_patch_*`、`_rcmd_*` 脚本仍硬编码旧路径或历史状态。先查实际入口与目标，不批量执行历史脚本，不用全仓字符串替换修复路径。

## 证据与性能结论

- 保存本次代码/配置身份、环境、输入身份、运行 ID、原始日志和失败信息。合成日志、mock 数据、重建复算输入明确标记来源，不能替代真实实验。
- SK04 主要判断可比性：`pointwise_feasible` 不等于数值达标，`NEEDS_EVIDENCE` 不等于精度通过。核对 `officiality`、数据身份、样本顺序和 `open_gates`；首步 loss 相等不证明全程正确。
- 训练窗口常用 50–100 步；短诊断、自定义窗口、算子微基准、推理延迟分别报告。`95_extract_series.py` 当前吞吐按 GBS=8 计算，不直接用于其他 batch 几何。
- 优化比较明确 baseline/candidate、分母、配对顺序、有效步数及数据范围。A/B 是配置标签，不默认等同基线/候选。
- `62_reportability.py` 默认至少 3 对，要求省时比例的 95% 区间下界达到 `max(2 × 底噪, 5%)`，不能只看点估计。保留 `run_validity`、`accuracy_validity`、`performance_pass`、`adoption_ready` 的区别，不填无证据的精度 OK。
- 原始日志、协议、哈希清单和账本用于追溯；更正结论记录原因及新证据，不改写原始实验，不靠重建哈希清单掩盖内容漂移。
- 历史“51×”“419 ms”“19.6062%”各有不同基线、数据与环境范围，不能拼成当前或通用官方对比结论。未实测候选保留基线、开关及回退方式。

## 开发与交付约定

- 保持现有脚本式结构、CLI 和产物契约；复用 `_envcompat.py`、`_project_paths.py` 及现有判据，不重复实现同一统计逻辑。修复覆盖实际失败路径，不以大规模重构替代定位。
- 中文文档和文件读写明确使用 UTF-8，Shell 脚本保留 LF。跨进程优先参数列表；拼接 Bash 使用正确引用，覆盖中文、空格、特殊字符路径。
- 不批量删除目录、日志、证据或测试产物。`safe_pack_sync.py --apply` 的覆盖式同步已禁用并返回 2，`--verify-only` 可只读核验。交付同步单独设计可审查的增量方案，不自动覆盖 `01_交付物/`。
- 密码、私钥、token、Cookie、代理密钥不得写进代码、文档、命令输出或提交记录。`token_broker.py` 使用 `ZERO_RISK_BROKER_SECRET` 或显式 `--secret`，无可复用默认值；客户端和服务端配置需一致。
- 远端操作先确认部署版本和会话；多行命令落为脚本后通过已核查入口执行。不用宽泛的 `pkill -f` 清理任务；未授权的训练、下载、环境升级或共享任务终止不能作为本地检查的附带动作。
- 模型、checkpoint、临时输出、凭证存储和大型本地数据按 `.gitignore` 排除；已有示例和历史证据是保留资产，不因通用忽略规则删除。
- 重大改动更新对应使用说明/能力清单，记录原因、实际验证结果、未验证项和回退方式；**不更新废弃的 `AGENT.md`**。向用户交付时说明改动、验证及限制。

截至本次源码核对，主要后续缺口是 F06 场景真实输入输出闭环、F07 可报告性检查接入、F10 依赖环境整理。F01–F05、F08/F09 已有代码修复与离线回归覆盖，不代表当前版本完成 NPU 或业务验收。状态变化时更新对应段落，不把本指南变成持续追加的历史日志。
