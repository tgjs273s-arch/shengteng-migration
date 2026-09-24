# 优化器路径筛选：已完成，筛选未通过

## warmed-v2 结果：已执行，仍未通过，停止微基准重测

boot `7bcced6f-26df-4169-ae9c-eea0227981dd`。
原始证据：`out/p58_warmed_metadata.json`、`out/p58_warmed_runs.json`、
`out/p58_warmed_summary.json`、`out/p58_warmed_remote.launch.log`。
远端目录 `/root/ops/p58_warmed_v2_20260917`；日志包含
`OPTIMIZER_SCREEN_COMPLETE`，本地读取与复算后 candidate=false。

| 块 | A1 / B / A2 中位（ms） | 基线漂移（%） | 有效性 |
|---|---|---|---|
| 1 | 14.70077969133854 / 9.451515041291714 / 14.518589712679386 | 1.2548738015514185 | 有效 |
| 2 | 14.45507537573576 / 10.819525457918644 / 15.84556419402361 | 9.619381304797063 | 超限 |
| 3 | 15.350029803812504 / 9.724765084683895 / 15.115030109882355 | 1.5547418180563461 | 有效 |

九次最终合成参数最大绝对差均为 0。独立复核输出
`WARMED_RAW_AUDIT_OK runs=9 measured=900 warmup=900 candidate=false`；
核对了协议/窗口/每步原始时长/中位数/三块漂移/降幅/t 区间/boot/源码 SHA。
不得用包含无效块的描述性平均降幅或 t 区间宣称有效收益。
按事前停止条件，不继续修改预热步数或重复该微基准；未切换正式训练优化器。
结论仅为固定预热未解决本批次的基线漂移，不等于证明 foreach 在整模型无效。
模型 loss/grad_norm、kernel 发射数、弹性仍未测；图捕获未授权、未执行。

## 已执行协议的事前登记：warmed-v2

保留 legacy-v1 失败结果及 `out/p58_original_source.py` 原源码（用于核对旧 metadata SHA）。
新批次只验证固定预热后是否稳定：每次运行先更新 100 步，再测量 100 步，
统计窗口为更新 101–200；预热原始耗时另存 `warmup_times_ms`，不得混入正式计时。
顺序仍为三组 A/B/A；不改变漂移 <3%、平均降幅 >=5%、参数最大绝对差 <=1e-6 的门槛。
只运行一批；若再次漂移超限，停止这条微基准重测路线，不试不同预热数直到通过。
新旧协议的更新次数与窗口不同，不跨批合并或直接比较绝对时长。

```bash
python3 scripts/58_optimizer_screen.py --selftest
bash scripts/p58_launch.sh /root/ops/p58_warmed_v2_20260917 warmed-v2
```

SSH 长任务需由后台方式启动并保留 launcher 退出码；上面命令不是前台长任务建议。
启动器现在显式接收新目录/协议，脚本从启动器所在目录定位，上传到 /root/ops 也有效。
原固定目录启动器的历史提交不可按新接口重放；先查是否已有任务，禁止重复启动。
新增协议自检输出 `WARMUP_PROTOCOL_SELFTEST_OK good=2 bad_rejected=1`。
本批次已上传新版两个脚本，并核对探针 SHA 与 metadata 一致。

## 结果更新（2026-09-17）

重新连接后取回原始结果；boot 仍为
`7bcced6f-26df-4169-ae9c-eea0227981dd`。
`OPTIMIZER_SCREEN_COMPLETE` 已出现，summary 的 `candidate=false`。
原始 JSON/日志存于活副本 `out/p58_{metadata,runs,summary}.json`、
`out/p58_remote.launch.log`。历史提交状态保留在后文。

| A/B/A 块 | A1 / B / A2 中位（ms） | A 漂移 | 判据 |
|---|---|---|---|
| 1 | 10.502929799258709 / 8.18286556750536 / 11.063184589147568 | 5.3342714899265654% | 超限 |
| 2 | 11.420595459640026 / 8.288259617984295 / 11.507159098982811 | 0.7579608230472542% | 块内有效 |
| 3 | 11.478990316390991 / 8.372205309569836 / 11.490053497254848 | 0.09637764784991601% | 块内有效 |

九次合成参数最终更新与第一次 A 的最大绝对差均为 0。
独立读取 runs.json 的每步时长，核对九次各 100 步、窗口中位、
三块漂移、降幅、boot 与源脚本 SHA256，一致。
原始 summary 中的平均降幅和 t 区间包含漂移超限块，
仅为描述统计，不能用于有效收益推断。不得只保留后两块宣称筛选通过。
当前未升级为正式训练候选，也未测 kernel 发射数或弹性。
未来若验证预热后的稳定性，须另立新批次、事前登记预热协议，
不改本批次门槛；本次没有重跑，也没有认定超限原因是冷启动。

## 已核实的调用路径

2026-09-17 通过 SSH 只读检查，目标 199.93.55.205，
`identity.boot=7bcced6f-26df-4169-ae9c-eea0227981dd`。

复现命令：

```bash
cat /proc/sys/kernel/random/boot_id
sed -n 300,325p /root/MindSpeed-MM/mindspeed_mm/fsdp/train/trainer.py
cat /root/MindSpeed-MM/mindspeed_mm/fsdp/optimizer/optimizer.py
cat /usr/local/lib64/python3.11/site-packages/torch_npu/optim/npu_fused_optim_base.py
```

配置 `training.adam_fused=true` 经 `trainer.get_optimizer()` 传给
`build_optimizer()`；adamw 分支构造 `torch.optim.AdamW`，其中
`foreach = not fused`。因此现有 fused 配置不等于使用 `NpuFusedAdamW`，
也不能仅凭开关证明底层发射数减少。

已安装的 `NpuFusedOptimizerBase` 仅接受 FP32/FP16，并在初始化时合并
参数与梯度存储；`zero_grad(set_to_none=True)` 明确拒绝。
它与本项目 FSDP 参数、梯度生命周期的兼容性尚未验证，不直接替换。
这属于替换前风险，不是已经证明现有训练不兼容。

## 本次新增探针与事前判据

`scripts/58_optimizer_screen.py`：只做合成 FP32 张量的 AdamW 筛选。
所有张量初值和梯度相同，A=fused，B=foreach；三组 A/B/A，
每次 100 次更新，窗口 11–100。每步同步只用于本探针计时。
判据在脚本头部预登记：每块 A 漂移严格小于 3%，平均降幅至少 5%，
最大参数绝对差不超过 1e-6、无非有限值，才可作为后续训练候选。
输出三块降幅的 t95% 区间，n=3，独立性/正态性未证实。

```bash
python3 scripts/58_optimizer_screen.py --selftest
python3 scripts/58_optimizer_screen.py --execute --out /root/ops/NEW_DIRECTORY
```

本地自检输出 `OPTIMIZER_SCREEN_SELFTEST_OK good=1 bad_rejected=4`：
缺运行、混 boot、NaN 时长必须拒绝；基线漂移过大不得成为候选。

这是独立诊断，不是 PRE/P0–P7 新阶段，不产生新的判定链产物。
不测模型 loss/grad_norm、真实 kernel 发射数或弹性；不改变 GBS，
不运行模型训练；`official_comparable=false`，不允许外推整模型收益。

## 历史真机提交状态与取回路径

部署到 `/root/ops/58_optimizer_screen.py` 和 `/root/ops/p58_launch.sh`。
后者当时为固定目录启动器，重复运行会因目录已存在而拒绝覆盖。
通过 `setsid nohup bash /root/ops/p58_launch.sh` 提交后台任务，
启动命令 rc=0，但当时日志为空；随后查询遭跳板拒绝：
`platform authorization denied: Administratively prohibited`。
不能据此断言 token 过期，也不能把启动 rc=0 当探针成功。

已只读取回（原远端路径）：

- `/root/ops/p58_optimizer_screen_20260917.launch.log`
- `/root/ops/p58_optimizer_screen_20260917/metadata.json`
- `/root/ops/p58_optimizer_screen_20260917/runs.json`
- `/root/ops/p58_optimizer_screen_20260917/summary.json`（若完成）

必须先检查现有任务状态，不重复启动。若筛选无收益或数值失败，
记录负面结果；若筛选值得继续，才设计官方几何单变量训练与独立
profiling。训练必须核验 loss/grad_norm 逐点，并实测发射数 X 与
非 profiling 步时长 Y 后才可计算弹性 Y/X；未取得这些数据时保持 null。

图捕获未授权、未执行；Triton 未清理、未换版。
两组冻结证据及 sk04_judge 未编辑；裁决仍为 NEEDS_EVIDENCE（黄），
data_identity/sample_order 未闭合。本次没有性能提升或精度验收结论。
