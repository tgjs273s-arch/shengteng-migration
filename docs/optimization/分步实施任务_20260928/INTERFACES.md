# 分步实施接口约定

状态：**待实施契约，当前代码并未全部实现这些字段。** 本文约束 Agent 之间的连接，不要求建设新工作流框架。字段名可在对应实施任务中以兼容方式定案；修改时更新本文、下游任务假设和测试，不能静默漂移。

## 通用原则

- 同一来源只由一处产生身份，下游引用，不自行重建或猜测。最短追溯关系为：GPU 源码身份 → migration_id → train_run_id → checkpoint/model_artifact_id → inference_run_id。
- 同时保存 schema 版本、代码/配置身份、路径与内容摘要、实际运行状态、错误与测量范围。路径是定位信息，不能单独充当内容身份。
- 合成/夹具/真实测量的来源必须显式区分。未知值为 null/未验证，不能填 0、OK 或示例答案。
- 兼容已有 CLI/产物及返回码。优先扩展现有 JSON，不新造重复状态层；必要破坏性 schema 变化提供明确版本及迁移策略。
- 原始日志和既有哈希链不重写。新运行有独立身份；receipt 只证明纳入范围的产物，不能自动证明外部环境未变。

## 产物与字段责任

| 产物 | 最少内容 | 主要生产者 | 消费者 |
| --- | --- | --- | --- |
| 要求/基线记录 | 规则来源、版本/适用性、baseline ID、旧名映射、原件与配置/日志 hash、待确认项 | T01；T07 落实代码 | T05/T06/T07/T12/T15 |
| migration_manifest | 输入 GPU 版本及 hash、目标框架 commit、目标文件/diff 摘要、复用/自研说明、执行记录、适配状态、实际模块位置 | T03 | T04/T08/T10/T12 |
| assets.json 扩展 | 模型/processor/tokenizer revision 和完整性，数据内容/顺序身份，转换工具参数、图片可用性、缺证据 | T05 | T07/T08/T10/T12 |
| 有效配置与差异 | 参考/候选角色、baseline ID、允许差异及全部变更、实际 mbs/gas/dp/world/GBS | T05 | T06/T07/T10/T12/T13 |
| train run/完整性结果 | migration 引用、实际导入实现、有效配置/资产引用、返回码、恢复起点、预期/实际唯一步数、缺/重复/坏记录 | T06 定义；T10 连接源码身份 | T07/T08/T10/T12/T13 |
| performance result | 来源日志/步骤、rank、GBS、selection/aggregation、mean/median、吞吐定义、完整性、冷启动/缓存范围 | T06 | T10/T12/T13/T15 |
| judge_summary 扩展 | execution、comparability、numeric_metrics、规则版本/状态、numeric_validity、基线与本次运行身份、open_gates | T07 | T10/T12/T13/T15 |
| model_artifact | 来源 train_run 与 migration、保存步/格式、权重与配置/processor 文件清单、转换记录、参数检查、真实重载验证 | T08 | T09/T10/T12/T14 |
| inference_result | model_artifact 引用、加载框架/代码/设备、媒体/提示/生成参数身份、原始生成、解析、耗时边界、错误 | T09 | T10/T12/T14/T15 |
| adoption/reportability | baseline/candidate ID、同口径指标、配对/底噪/区间、数值有效性来源、工程采纳状态及代价 | T10 接线；T13 实测 | T15 |
| scene evaluation | 数据划分/标签来源、样本数量、预测/错误实例、任务指标、测量范围和不确定性 | T14 | T15 |

上述名称是逻辑产物，实际路径由生产任务在 handoff 中冻结。代码样例可以使用临时夹具，不得把建议路径或尚不存在的 schema 当成已运行产物。

## 本轮工程源码版本（2026-09-28）

**2026-09-29 状态更新：下表为此前临时工程候选。用户新提供 PDF 第 1 页指定 `git checkout 26.1.0`，不与下表 `v26.0.0` 等同。主管已采纳下一节的固定新工程目标，代码适配与验证尚在进行；不得将旧候选原型、保存格式假设或测试结论移用于新目标。**

T03 已对不可变上游提交进行只读源码核查，主管采纳下列版本作为工程输入。此决定不代替比赛指定来源确认，也不证明本机已经安装或运行对应环境。

| 身份 | 固定值与出处 |
| --- | --- |
| GPU 输入 | [Transformers fc9137225880a9d03f130634c20f9dbe36a7b8bf](https://github.com/huggingface/transformers/tree/fc9137225880a9d03f130634c20f9dbe36a7b8bf)，目标 Qwen3.5 示例指定的源码版本 |
| 目标框架 | [MindSpeed-MM 6c45b4869f9938892b982a203cc121803c345db2](https://github.com/Ascend/MindSpeed-MM/tree/6c45b4869f9938892b982a203cc121803c345db2)，tag v26.0.0；不可用同名移动分支代替 |
| 模型元数据 | [Qwen/Qwen3.5-0.8B 2fc06364715b967f1860aea9cf38778875588b17](https://huggingface.co/Qwen/Qwen3.5-0.8B/tree/2fc06364715b967f1860aea9cf38778875588b17)；当前仅核实配置及索引，未下载或验证完整权重 |

T03 应将本次实际消费的文件摘要、源码差异、上游复用与项目增量写入 migration_manifest；仓库旧 `refs/` 仍为 AST 参考，不冒充上述 GPU 输入。

目标 [train_engine.py](https://github.com/Ascend/MindSpeed-MM/blob/6c45b4869f9938892b982a203cc121803c345db2/mindspeed_mm/fsdp/train/train_engine.py) 使用 `training.load` 加载 checkpoint；[tracker 解析](https://github.com/Ascend/MindSpeed-MM/blob/6c45b4869f9938892b982a203cc121803c345db2/mindspeed_mm/fsdp/checkpoint/utils.py) 区分 `release` 初始权重和整数续训步数。恢复真实起点还需对应 checkpoint 的元数据，不能从目录存在或旧 `load_checkpoint_path` 字段猜测。

配置、注册架构和权重索引的静态吻合不证明模型可实例化、权重数值完整或 NPU 正确；相关验证分别由 T03/T04/T08/T12 留存。当前运行环境与正式精度规则仍待补齐。

## 2026-09-29 采纳的新工程目标

- 依用户提供 PDF 第 1 页的 `26.1.0` 分支要求，固定 [MindSpeed-MM `5b5505331924634da64e3d9a1925d02b10babe9f`](https://github.com/Ascend/MindSpeed-MM/tree/5b5505331924634da64e3d9a1925d02b10babe9f)。GitHub/GitCode 两仓库当前分支相同；`v26.1.0` tag 为 `d0b964b55fb33b4eccec72513d8741d391ec13c7`，与所选分支分叉，不混用。材料未给不可变 commit；这里固定的是本次核对时的分支版本，不宣称它就是材料生成时的历史提交。
- 新目标 Qwen3.5 示例 README 仍指向 GPU Transformers `fc91372`，沿用已核完整 commit `fc9137225880a9d03f130634c20f9dbe36a7b8bf`。0.8B 元数据仍锁 `2fc06364715b967f1860aea9cf38778875588b17`，作为项目明确选择而非 PDF 指定 revision；PDF 的 4B 配置文件名不能改变本项目 0.8B 目标。
- 新版注册器返回类，ModelHub 配置覆盖增加 feature_args，并有 MTP 和 causal-conv 实现变化。T03 需另存新 bundle、manifest、补丁和摘要，重放后才交接；不得覆盖旧版本证据。Triton 参考日志第 90–91 行明确 `mtp_num_layers: 0`、`mtp_loss_scaling_factor: 0.1`，参考与 runtime 检查按此显式记录 MTP 策略，不能静默启用 1 层。
- T08 已核固定新目标仍用 DCP checkpointer 保存；`load_format` 只控制加载，未知 `save_format` 可被配置解析容纳但没有保存实现消费。P2/P5 不能凭该字段宣称 HF 保存或规避 DCP；所需 HF 产物走版本匹配的显式导出。`no_save_optim`、`no_save_rng`、`load_rank0_and_broadcast` 是真实受支持字段，不能混同为无效参数。
- 原件目录 `D:/JS/官方资料/` 的七件材料摘要见 T01 记录。Triton CRLF 原件与仓库 LF 副本已实际核对为仅行尾不同；后续来源可同时绑定两种字节摘要。PDF 第 2 页已核实前 200 条取后 100 条的均值口径。精度模板示例未定义正式通过条件，RULE_PENDING 不因原件到位自动解除。

## T06 本地接口冻结与后续集成

- 共享解析器为 `scripts/_train_log.py`：`read_log(path)` 返回保留文件顺序的 rows、坏记录、日志 SHA-256/字节数；`integrity(parsed, expected_end, start_step, expected_gbs, rank)` 返回 `train_integrity.v1`，状态为 COMPLETE、INCOMPLETE 或 UNKNOWN。重复、乱序、缺/多步、rank/total/GBS 冲突和非有限数不允许静默丢弃。
- P5 每次 `<log parent>/runs/p5-*/` 增加 `effective_config.yaml` 与 `train_integrity.json`，绑定解析/执行配置快照、实际返回码、日志身份、预期/实际范围、起点与诊断范围。只有 COMPLETE 允许独立 P5 成功；不信任未知的数字续训起点。
- P6 CSV 为 `train_series.v2`，保留旧前五列；JSON 为 `train_performance.v2`。`official_selection` 保留历史字段名但显式标注当前官方适用性未验证，按前 200 条中的末 100 条取算术均值；`window_selection` 保存含端点区间，50–100 须为 51 点。均值吞吐与中位步时折算速度分列，旧 `samples_per_s` 仍表示后者。
- P6 的 `--expected-gbs` 仅核对调用者提供的数值与日志一致；T10 负责把它绑定到 P5 已核验配置，不能仅凭该整数宣称外部配置身份已验证。
- 已有 `_stage_state.code_digest()` 覆盖新脚本，但 P5 OUTPUTS 尚未绑定新增运行产物，主入口 P5/P6 仍有旧断言。T10 必须纳入当前运行 receipt/快照/完整性结果并消费新版状态，不能把本步独立入口验证说成主入口已集成。

## T03 旧候选静态原型与版本过渡

- `22_migrate_qwen35.py` 当前支持固定 10 文件 fetch、apply、隔离 checkout 安装和可选 runtime；`qwen35_migration.v1` 只表示 `STATIC_APPLIED_RUNTIME_PENDING`。源/目标/补丁按字节核对，项目 tied-source 前置检查与上游模型复用分别记载。
- 当前已验证 migration_id 为 `qwen35-0p8b-83e582c816045ba3`，对应上述旧候选。P4 converter/收据接线不等于完整权重 payload 或真实模型可加载；T05 资产子步尚需补全。
- runtime 通过固定框架注册表取类，核对实际导入路径；仅确认依赖缺失返回 4，运行/身份错误返回 3。meta 构造关闭 Triton，结果必须注明未执行目标 kernel。
- 13 项离线回归及旧候选补丁重放已通过；新官方材料的 `26.1.0` 尚未接线，因此不以此放行正式下游验收。新版本更新必须另有源码身份与重放记录，不覆盖旧证据。

## T03 新版静态接口复核结果（2026-09-29）

固定新目标的实际 migration_id 为 `qwen35-0p8b-a69b4e66be0ab549`，converter patch SHA-256 为 `2bda81ad2c3f7ccdaba6dec8d5de4e7a0898c73f37e2f6d9bbcc4027633773cb`。真实 16 文件 bundle、安装幂等/漂移拒绝与 15 项离线回归已验证，输出仍为 `STATIC_APPLIED_RUNTIME_PENDING`。P4 转换收据升级 `qwen35_0p8b_conversion.v2`，包含新目标源码与补丁身份、15 个 MTP 源键及 `dcp_mtp_reload_verified=false`；此收据仍只覆盖结构与元数据，完整 payload/资产由 T05 补齐。T04 可消费已冻结源/目标及实际后端定义；runtime 实例化、decode/cache 疑点和 NPU 数值仍未解决。

## T05 配置子步接口冻结

- P2 增加 `--config-role reference|candidate`，默认 `reference`。参考只使用独立参考模板；P0 的推荐优化必须显式选择 `candidate`。已知参考环境不匹配时保留诊断产物并返回 3，探测缺项记 `unknown`，不声明可运行。
- P2 新增 `reference_config.yaml`、`config_manifest.json`（`migrator_config.v1`），保存实际/参考配置 SHA-256、canonical baseline ID、来源模板和 P0 身份、GBS 几何、逐叶差异以及字段来源。差异有双方 `*_present`，缺失与 null/空映射不混同。
- `verified_reference_fields` 仅描述已经核对的仓库存量日志字段，其他字段列入 `template_default_fields_unverified`；整份参考模板不等于完整官方配置。新提供的外部材料由 T01 补核后再更新来源状态。
- P2 新产物、角色传参及配置到本次训练的绑定仍由 T10 接入 `_stage_state.OUTPUTS` 与主入口。T05 资产子步尚未冻结，消费方不得推测其字段。
- 新目标更正已复核：manifest 增加 `target_framework_commit`、实际 `checkpoint_format=dcp`、`checkpoint_enabled`、`checkpoint_controls` 与阻断原因；自定义 `training.save_format` 被拒绝，P0 的 HF 保存建议不应用且返回 3。参考与候选显式写日志中的 `model.mtp_num_layers=0` 和 `mtp_loss_scaling_factor=0.1`。这些是配置证据，真实 DCP 保存由 T08/T12 验证。

## T08 保存隔离子步接口

- P5 的 `effective_config.yaml` 是本次实际输入快照。启用保存时，`training.save` 改为本次 `runs/p5-*/checkpoints` 绝对路径；源配置不修改。显式 null/false/空串的关闭意图保留在收据，有效快照省略 `save` 键以兼容固定目标参数类。未提供该键时保留原字节。
- `run.json` 记录 `run_id`、`source_config_sha256`、`effective_config_sha256` 和 `checkpoint_save`（源值、启用状态、实际路径、实际 DCP 格式）。`train_integrity.json` 的 `source_config`/`source_config_sha256` 绑定源文件，`config`/`config_sha256` 绑定有效快照；旧 `config_input` 仍指源路径，不可把它与快照摘要配对。运行后分别检查两份配置，启用保存时还检查收据一致性。
- 固定目标没有 HF 训练保存开关。P5 拒绝非 DCP 的 `save_format` 声明；显式 `dcp` 仅作兼容声明，不负责选择 checkpointer。P2 仍拒绝该无效字段。
- 这些字段只证明启动配置与本次保存意图，不证明 checkpoint 已写完整或能加载。完整 `model_artifact`、导出与重载等待 T05 资产交接；T10 需纳入上述实际快照与运行收据，T07 消费配置时必须区分源配置和按次变化的保存路径。

## T07 独立数值核心接口

- `judge_summary.json` 增加 `baseline_identity`、`candidate_log_sha256`、`candidate_integrity`、`comparability`、`numeric_metrics`、`numeric_validity`、`rule_status`、`numeric_acceptance`、`alignment_evidence` 和 `numeric_open_gates`。`VALID_MEASUREMENT` 只表示所选日志可计算，当前规则仍 RULE_PENDING、数值验收 NOT_DETERMINED。
- canonical Triton ID 和旧 `officialB` 兼容，但以登记 YAML/日志 hash 配对；旧 `officialA` 继续保留独立身份。原件补核、配置/资产身份与当前训练绑定尚须后继消费，不能以别名替代证据。
- CSV 与 JSON 共用逐步误差数据，零/近零参考的相对误差为 null。SK04 观测增加 `iteration_integrity`；不完整序列不提供有效 step1 证据或支持绿色结果。默认/`--gate` 返回码定义不变。

## 验收结果语义

| 字段层次 | 要回答的问题 | 不能替代 |
| --- | --- | --- |
| 执行状态 | 工具是否成功执行并写出本次有效结果 | 数值精度或业务正确性 |
| 训练完整性 | 是否完成本次要求的完整训练/统计步骤 | 数据可比性或精度达标 |
| 可比性 | 基线与候选是否满足对应对比前提 | 实际误差是否过线 |
| 数值精度 | 在已声明规则下实际误差是否达标 | 工业识危正确率 |
| 性能测量 | 同口径速度、吞吐和统计区间是多少 | 优化一定可采纳 |
| 内部采纳 | 运行、数值、收益/代价是否满足工程规则 | 未确认的官方竞赛标准 |
| 推理执行 | 对应模型产物是否在 NPU 上产生真实结果 | 场景判断一定正确 |
| 场景质量 | 对照独立标签的判断效果如何 | 其他场景或全行业有效 |

保持 70_judge.py 默认退出码表示产出完成、--gate 颜色与执行失败现有区别。若增加正式精度/总验收退出码，应使用明确的新模式或版本并测试，不能隐式修改旧含义。

## 共享文件责任与集成顺序

| 文件/领域 | 顺序 | 约束 |
| --- | --- | --- |
| P0/P2 与环境配置 | T02 环境 → T05 参考/候选 → T08 保存配置 → T10 入口 | 顺序更新，依最新代码复核，不覆盖前项 |
| P5 | T06 完整性 → T08 模型保存 → T10 来源和总状态 | runner/锁/返回码保护保留 |
| P7/SK04 | T07 数值/基线 → T10 总验收消费 | 不把可比性结果改造成未经计算的数值 PASS |
| run_from_zero/_stage_state | T10 为主要集成责任者 | 上游先定义产物；如必须提前改，写入 handoff 并通知协调者 |
| README/SKILL/能力清单 | 每步只改对应状态，T10 整合，T15 最终复核 | 不覆盖其他任务的实际记录，不更新 AGENT.md |

本编排默认串行，以上不是并行编辑许可。若协调者另行安排独立工作，必须保证不同输出目录及写入所有权；公共 out/NPU/账本仍不得并发占用。

## 接口冻结与后继放行

生产任务 handoff 必须给出字段、单位、状态、示例来源、路径与代码位置；消费任务先检查实际产物。发现不兼容应修正生产/消费双方并补真实失败路径验证，不能填默认 OK、抓旧顶层 JSON 或反向修改历史结果来兼容。
