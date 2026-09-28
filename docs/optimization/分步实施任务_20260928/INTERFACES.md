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

T03 已对不可变上游提交进行只读源码核查，主管采纳下列版本作为工程输入。此决定不代替比赛指定来源确认，也不证明本机已经安装或运行对应环境。

| 身份 | 固定值与出处 |
| --- | --- |
| GPU 输入 | [Transformers fc9137225880a9d03f130634c20f9dbe36a7b8bf](https://github.com/huggingface/transformers/tree/fc9137225880a9d03f130634c20f9dbe36a7b8bf)，目标 Qwen3.5 示例指定的源码版本 |
| 目标框架 | [MindSpeed-MM 6c45b4869f9938892b982a203cc121803c345db2](https://github.com/Ascend/MindSpeed-MM/tree/6c45b4869f9938892b982a203cc121803c345db2)，tag v26.0.0；不可用同名移动分支代替 |
| 模型元数据 | [Qwen/Qwen3.5-0.8B 2fc06364715b967f1860aea9cf38778875588b17](https://huggingface.co/Qwen/Qwen3.5-0.8B/tree/2fc06364715b967f1860aea9cf38778875588b17)；当前仅核实配置及索引，未下载或验证完整权重 |

T03 应将本次实际消费的文件摘要、源码差异、上游复用与项目增量写入 migration_manifest；仓库旧 `refs/` 仍为 AST 参考，不冒充上述 GPU 输入。

目标 [train_engine.py](https://github.com/Ascend/MindSpeed-MM/blob/6c45b4869f9938892b982a203cc121803c345db2/mindspeed_mm/fsdp/train/train_engine.py) 使用 `training.load` 加载 checkpoint；[tracker 解析](https://github.com/Ascend/MindSpeed-MM/blob/6c45b4869f9938892b982a203cc121803c345db2/mindspeed_mm/fsdp/checkpoint/utils.py) 区分 `release` 初始权重和整数续训步数。恢复真实起点还需对应 checkpoint 的元数据，不能从目录存在或旧 `load_checkpoint_path` 字段猜测。

配置、注册架构和权重索引的静态吻合不证明模型可实例化、权重数值完整或 NPU 正确；相关验证分别由 T03/T04/T08/T12 留存。当前运行环境与正式精度规则仍待补齐。

## T06 本地接口冻结与后续集成

- 共享解析器为 `scripts/_train_log.py`：`read_log(path)` 返回保留文件顺序的 rows、坏记录、日志 SHA-256/字节数；`integrity(parsed, expected_end, start_step, expected_gbs, rank)` 返回 `train_integrity.v1`，状态为 COMPLETE、INCOMPLETE 或 UNKNOWN。重复、乱序、缺/多步、rank/total/GBS 冲突和非有限数不允许静默丢弃。
- P5 每次 `<log parent>/runs/p5-*/` 增加 `effective_config.yaml` 与 `train_integrity.json`，绑定解析/执行配置快照、实际返回码、日志身份、预期/实际范围、起点与诊断范围。只有 COMPLETE 允许独立 P5 成功；不信任未知的数字续训起点。
- P6 CSV 为 `train_series.v2`，保留旧前五列；JSON 为 `train_performance.v2`。`official_selection` 保留历史字段名但显式标注当前官方适用性未验证，按前 200 条中的末 100 条取算术均值；`window_selection` 保存含端点区间，50–100 须为 51 点。均值吞吐与中位步时折算速度分列，旧 `samples_per_s` 仍表示后者。
- P6 的 `--expected-gbs` 仅核对调用者提供的数值与日志一致；T10 负责把它绑定到 P5 已核验配置，不能仅凭该整数宣称外部配置身份已验证。
- 已有 `_stage_state.code_digest()` 覆盖新脚本，但 P5 OUTPUTS 尚未绑定新增运行产物，主入口 P5/P6 仍有旧断言。T10 必须纳入当前运行 receipt/快照/完整性结果并消费新版状态，不能把本步独立入口验证说成主入口已集成。

## T05 配置子步接口冻结

- P2 增加 `--config-role reference|candidate`，默认 `reference`。参考只使用独立参考模板；P0 的推荐优化必须显式选择 `candidate`。已知参考环境不匹配时保留诊断产物并返回 3，探测缺项记 `unknown`，不声明可运行。
- P2 新增 `reference_config.yaml`、`config_manifest.json`（`migrator_config.v1`），保存实际/参考配置 SHA-256、canonical baseline ID、来源模板和 P0 身份、GBS 几何、逐叶差异以及字段来源。差异有双方 `*_present`，缺失与 null/空映射不混同。
- `verified_reference_fields` 仅描述已经核对的仓库存量日志字段，其他字段列入 `template_default_fields_unverified`；整份参考模板不等于完整官方配置。新提供的外部材料由 T01 补核后再更新来源状态。
- P2 新产物、角色传参及配置到本次训练的绑定仍由 T10 接入 `_stage_state.OUTPUTS` 与主入口。T05 资产子步尚未冻结，消费方不得推测其字段。

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
