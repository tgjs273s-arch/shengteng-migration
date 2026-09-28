# T08 配置 API 独立复核（2026-09-29）

## 范围与来源

本次只读确认 MindSpeed-MM `26.1.0` 分支固定提交 `5b5505331924634da64e3d9a1925d02b10babe9f` 中真实 `TrainingArguments` 的 `save` 输入行为，以及 `ConfigManager` 在构造配置类前是否过滤 `None`。没有运行 P5/P3/P4、完整 Trainer、训练、权重加载或 NPU；没有安装依赖；没有修改 T08 handoff、生产文件或既有测试。

源码从隔离 checkout 的 Git HEAD blob 精确提取到忽略目录 `tmp/t08_config_api_20260929/src/`，记录 Git blob 与文件 SHA-256：

| 固定提交相对路径 | Git blob | SHA-256 | 字节数 |
| --- | --- | --- | ---: |
| `mindspeed_mm/config/arguments/base_args.py` | `968e1e9fcc02db7529bbf561bab086787cfc3033` | `bb6464eb10525240deecfc99872719e3546e8d2be63294c45f8a3defc3b6b424` | 14,285 |
| `mindspeed_mm/fsdp/params/training_args.py` | `6ba5c56399fa088b2db02e776c84c54aee83bb15` | `df7a4ccb4b3d2b63df6a3121ec21699f67e01c597dc91a7e1421ef86a7a09450` | 12,088 |
| `mindspeed_mm/fsdp/params/lora_args.py` | `f8db72fbe1c0f40c391a94e3b142f8543a93bfcc` | `94f4303b76d3ad58fd0f520795eb04380c75717354ef577f294086b4b2c000e0` | 4,674 |
| `mindspeed_mm/config/config_manager.py` | `d146fe979afd58d255dad5d7076e5cc7146bed18` | `56327377443d8823a7b2ccdc6e4298eeb3c229bb6ae28ff0a533762725055281` | 32,131 |

源身份清单：`tmp/t08_config_api_20260929/source_identity.json`。隔离目标 checkout 仍为固定 HEAD；仅保留此前 T03 安装验证所需的 converter 改动，本次没有更改该 checkout。

## 实际类验证

- 解释器：`C:\Users\王\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe`，CPython 3.12.14；Pydantic 2.13.5。
- 探针 cwd：`D:\JS\shengteng-migration`。执行命令：`$env:PYTHONIOENCODING='utf-8'; & 'C:\Users\王\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe' 'tmp/t08_config_api_20260929/probe_training_args.py' | Tee-Object -FilePath 'tmp/t08_config_api_20260929/logs/training_args_validation.json'; exit $LASTEXITCODE`；Python 进程退出码 **0**（取自本轮执行结果；没有重跑）。JSON 保存实际类来源、字段信息、解释器/Pydantic 版本和五种输入结果。
- 从隔离目录导入固定提交的原始 `BaseArguments`、`LoraArguments` 和 `TrainingArguments`。提取目录没有 `__init__.py`，Python 以 namespace package 加载，没有执行 MindSpeed-MM 包初始化代码。运行时确认 `TrainingArguments` 继承实际 `BaseArguments`，其定义来自提取的 `training_args.py`。没有 stub 类、字段或校验器；这是真实配置类的隔离加载，不是完整 Trainer。
- 为避开 `Profiler.model_post_init` 对分布式环境变量的读取，探针设置 `LOCAL_RANK=0`、`RANK=0`、`WORLD_SIZE=1`。只用于构造默认子配置，没有导入训练框架。
- 固定源码的 `TrainingArguments.save` 声明为 `save: str`，默认值为 `None`，不是必填字段（`training_args.py:260–262`）。使用实际类分别构造以下输入：

| 输入 | 结果 | 实际 `save` / 错误 |
| --- | --- | --- |
| 省略 `save` | 接受 | `None`，类型 `NoneType`（来自字段默认值） |
| `save=None` | 拒绝 | `ValidationError`：`Input should be a valid string [type=string_type, input_value=None, input_type=NoneType]` |
| `save=False` | 拒绝 | `ValidationError`：`Input should be a valid string [type=string_type, input_value=False, input_type=bool]` |
| `save=""` | 接受 | 空字符串，类型 `str` |
| `save="path"` | 接受 | `"path"`，类型 `str` |

可复核输出：`tmp/t08_config_api_20260929/logs/training_args_validation.json`；执行脚本：`tmp/t08_config_api_20260929/probe_training_args.py`。

## ConfigManager 合并行为

只读检查固定提交的 `mindspeed_mm/config/config_manager.py`：

- `load_and_parse()` 在第 104 行调用 `_safe_merge_configs(yaml_config, cli_config, self.additional_args)`，之后在第 110 行直接调用 `self.config_class(**merged_config)`。
- `_safe_merge_configs()` 的 `_merge_dict` 从各配置源迭代 `source.items()`（第 614–616 行）；对非递归合并分支将值直接写入目标（第 623–627 行）。没有针对 `value is None` 的过滤或删除。合并优先级为 YAML、`additional_config`、CLI；后续源仍可能覆盖同名键。

因此，显式配置中的 `save: null` 会留在合并字典里并传给真实 `TrainingArguments`，当前不会被转换成“省略此字段”，随后的 Pydantic 校验会拒绝它。省略键时配置类使用字段默认值 `None`，这是本次验证通过的表示。空字符串虽然通过字段类型校验，但本次没有据此断言训练端语义上等同关闭保存。

## 结论与边界

当前配置类可验证的“未设置”表示是**省略 `save` 键**；显式 `null`、`false` 均不能作为等价配置值直接进入该配置类。若配置快照要保留用户的关闭意图，键缺席是当前已验证可被配置类接受的表示；实际训练端是否据 `None` 关闭保存尚未验证。若上游必须输出 `null`，现有 `ConfigManager` 合并路径不会替它归一化，需在别处明确处理后再构造配置对象。

本次没有确认空字符串在 P5/Trainer 中是否关闭保存，也没有执行 Trainer 或检查 checkpoint 副作用。证据仅覆盖固定提交的实际配置类校验与 `ConfigManager` 源码合并逻辑。
