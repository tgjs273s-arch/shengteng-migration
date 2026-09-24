# Skill R2-SK04 · 可比性机器化三件套（配置指纹门 + 数据身份指纹 + 判定账本）

- 来源 idea：R2-C6-01/02/03（创造者 C6 系统架构师；经金牌讲师初筛并入 R2-SK04）
- 一句话：把"哪次运行比哪份基准、数据是否同源、偏离哪几维"从**人工审计**变成
  **机器裁决**——报告/验收只准引用 judge verdict id，禁止人工粘贴数字宣称 PASS。
- 目标：7 项配置偏离（A2 审计 §3.1）自动检测 + N_eff 派生 + 可达性谓词
  pointwise|window|none + 数据序/字节身份三态 + append-only 判定账本；
  **纯 CPU 可开发验证（0 GPU），已在本地用真实日志（snapshots/coco_train_run1.log =
  A2 COCO 首跑日志、docs/official/triton精度日志.txt = 官方 B 原始日志、hiascend 官方帖
  A 提取件）实跑自检 24/24 PASS**。

---

## 目录结构

```
skill_R2-SK04/
├── README.md
├── configs/baselines/
│   ├── officialB.yaml        ← 目标基准 B 声明指纹（officiality=proposed：官方身份未证实）
│   └── officialA.yaml        ← 目标基准 A 声明指纹（officiality=confirmed：hiascend 官方账号帖）
├── scripts/
│   ├── fingerprint_cfg.py      ← ① 配置指纹门（7+2 维 + 偏离表 + 可达性谓词 + 证据门）
│   ├── data_id.py              ← ② 数据身份指纹（json_sha256/order_sha256/row_count/schema；双文件三态）
│   ├── fingerprint_observed.py ← 日志实测指纹（实际 GBS/LR 序列哈希/kernel 生效/回落/墙钟）+ 声明对账
│   ├── judge_comparable.py     ← ③ 纯函数判定机（verdict_id 确定性；判定逻辑与 fingerprint_cfg 同源）
│   ├── registry_append.py      ← append-only 运行账本（哈希链防证据漂移）
│   ├── extract_postA_log.py    ← 从 hiascend 官方帖 JSON 还原基准 A 日志原文（证据可复现）
│   └── selfcheck_sk04.py       ← 本地零 GPU 全链路自检（输出 VERIFY_OK 单行）
├── evidence/
│   ├── registry.json           ← 运行账本 genesis（空链，A2 实跑后逐条追加）
│   └── officialA_configuration_details.txt  ← 官方 A 帖日志行级提取（officialA.yaml 行号出处依据）
└── tests/
    ├── fixtures/               ← 合成数据/配置夹具（selfcheck 用，可再生成）
    └── tmp/                    ← selfcheck 运行产物（可删，跑一次即再生成）
```

## 与 R1-SK05 / S-M3 的关系（职责分工，互不覆盖）

| 层 | 工具 | 职责 | 边界 |
|---|---|---|---|
| **SK04（本）** | fingerprint_cfg/data_id/observed/judge/registry | **比得可比**：配置指纹门（绿/黄/红）、数据同源、运行对账、判定账本 | 只裁决"可比性"，不裁决"数值是否达标" |
| R1-SK05 | prepare_run/manifest/GBS_eff 硬校验 | 运行前的数据平面白名单 + manifest | 管运行产物结构；本 skill 的 `--world-size`/`--data-json` 建议从 manifest/日志取 |
| S-M3 SK02 | compare.py（provenance 守卫 + 冻结基准） | **数值比**：loss/grad_norm 相对绝对误差 <2%、窗口均值 | 只在 SK04 判"绿(逐点)"或"黄(窗口)"时才有数值语义；红牌（none/带冲突）不喂 S-M3 |

- 集成建议（run_train 前置门，R2-C6-01 语义）：`fingerprint_cfg --gate` 退出码
  0=绿(可逐点)/3=黄(仅窗口)/4=红(**门失败 exit 4 不烧 GPU**)；显式跳过需留痕
  （`--skip-fingerprint` 属 run_train 层约定，本包只提供工具与退出码契约）。
- 验收口径衔接：judge 的 `loss_band` 门与 S-M3 compare.py 的 provenance 守卫
  （`LOSS_BAND_SPLIT=5.0`）同一分带逻辑——**带冲突时 judge=NOT_COMPARABLE(红)**，
  compare.py 侧同样拒绝无语义比对，两层互不打架。

---

## 1. 安装/准备（改动前先备份：命令）

- 本 skill **全部为新增文件**，不修改任何既有文件（官方配置/compare.py/框架零触碰）；
  回滚 = 删除本目录（见 §5）。
- 依赖：Python 3.8+ **标准库**；`fingerprint_cfg.py`/`judge_comparable.py` 读基线
  声明 yaml 需要 **PyYAML**（`python -m pip install pyyaml`；data_id/observed/registry/
  extract 纯标准库）。**不依赖 torch/torch_npu/MindSpeed-MM/A2 任何环境**。
- 本包无 bash 脚本（全部 python，规避本机无 bash 的 bash -n 限制；A2 上同样
  `python3 scripts/…` 直跑）；若未来加 `.sh`，先 `bash -n` 自查。
- 上 A2 核对（只读，路径随交接快照）：
  ```bash
  ls -l runs/run_<tag>/config.yaml                       # 生效 config（run_train 产物）
  ls -l runs/run_<tag>/train.log                         # 训练日志（含 Configuration Details）
  ls -l /root/coco_dl/extracted/dataset/annotations_slim.json   # COCO 数据集 json（2000 样本）
  npu-smi -L                                             # 实卡数 → fingerprint_cfg --world-size
  ```

## 2. 运行（三件套用法）

**① 配置指纹门（runA vs 目标基准）**
```bash
cd /root/skills_R2/skill_R2-SK04     # 或 A2 上任意拷贝目录
# A2 推荐形态：直接吃训练日志内嵌的"生效 Configuration Details"（自带 world_size）
python3 scripts/fingerprint_cfg.py --config-from-log runs/run_<tag>/train.log \
    --baseline officialB \
    --data-json /root/coco_dl/extracted/dataset/annotations_slim.json \
    --out evidence/fp_run_<tag>_vs_B.json
# 或 config.yaml 形态（A2 run_train 产物）：需显式 world-size
python3 scripts/fingerprint_cfg.py --config runs/run_<tag>/config.yaml --baseline officialB \
    --world-size 1 --data-json /root/coco_dl/extracted/dataset/annotations_slim.json \
    --out evidence/fp_run_<tag>_vs_B.json
# 做 run_train 前置门（红牌 exit 4 不烧 GPU）：
python3 scripts/fingerprint_cfg.py --config runs/run_<tag>/config.yaml --baseline officialB \
    --world-size 1 --gate && echo GREEN || echo "gate=$?"   # 3=黄(仅窗口) 4=红(不可比)
```
输出 JSON 含：偏离表（每维 run 值 + **config 行号证据指针** + baseline 出处 file:line）、
派生 N_eff/GBS_eff/loss 打印语义、可达性 `level=pointwise_feasible|window_feasible|none`、
证据门（data_identity/sample_order/…）。一行摘要 `FINGERPRINT_OK …`。

**② 数据身份指纹（实际文件核对；官方文件不可得 → unresolved）**
```bash
python3 scripts/data_id.py --json /root/coco_dl/extracted/dataset/annotations_slim.json
python3 scripts/data_id.py --compare <A.json> <B.json>      # 同字节/同内容(异序)/未核对
python3 scripts/data_id.py --compare /root/.../annotations_slim.json /data/.../output_llava_coco_data.json
# ↑ 官方 B 的 json 不在 A2 → 输出 unverified（诚实，不许编官方哈希）
```

**③ 日志实测指纹 + 与声明对账**
```bash
python3 scripts/fingerprint_observed.py --log runs/run_<tag>/train.log \
    --declared evidence/fp_run_<tag>_vs_B.json \
    --lr-ref-csv ../skill_round2_SM3/reference_official_100steps.csv --lr-ref-label officialB \
    --out evidence/observed_run_<tag>.json
```
产出：实测 GBS（每步 samples 增量）、LR 序列哈希（可与官方 CSV LR 列逐 iter 比对）、
**kernel 证据行**（`roll back to CPU` / `not supported on current platform` 回落命中行号、
`use NPU triton ops`/`fused ops` 生效行号）、最后 iter、墙钟、step1(lr=0) loss 带、
discrepancies（如"声明 gdn=triton 但日志回落"）。

**④ 判定机（纯函数，报告只准引用 verdict_id）**
```bash
python3 scripts/judge_comparable.py --run evidence/fp_run_<tag>_vs_B.json \
    --observed evidence/observed_run_<tag>.json --baseline officialB \
    --out evidence/verdict_run_<tag>_vs_B.json
# 一行：JUDGE_OK verdict=WINDOW_OK level=window_feasible color=yellow officiality=proposed …
```
裁决五态：`POINTWISE_OK`(绿) / `WINDOW_OK`(黄，只许窗口均值口径) /
`NOT_COMPARABLE`(红，不喂 S-M3) / `NEEDS_EVIDENCE`(黄) / 输入错误(2)。
**默认退出码 0 = 裁决成功产出**（红/黄也属成功产出，exit_hint 在 JSON/摘要里）；
`--gate` 时按色退出 0/3/4（绿/黄/红，可挂 run_train 前置门）。
同输入必同输出（无随机/墙钟）；`verdict_id` = 基线 yaml+run 指纹+observed+data_cmp 内容摘要。

**⑤ 运行账本（append-only）**
```bash
python3 scripts/registry_append.py --tag run_<tag> --run-dir runs/run_<tag> \
    --manifest runs/run_<tag>/manifest.json \
    --fingerprint evidence/fp_run_<tag>_vs_B.json \
    --observed evidence/observed_run_<tag>.json
python3 scripts/registry_append.py --verify      # 整链校验（篡改即报 REGISTRY_CHAIN_BROKEN）
python3 scripts/registry_append.py --tail 5
```
写失败标 `REGISTRY_PENDING` 不阻塞主流程（账本不可用时以 run 目录证据为准，C6-03）。

## 3. 验收

**本地（零 GPU，交付方已跑 ✅ 2026-09-02）**
```bash
python scripts/selfcheck_sk04.py
# → VERIFY_OK selfcheck=1 case_ok=24/24 skipped=0
```
覆盖：真实 COCO 首跑日志 vs officialB/A 的偏离维断言（与 A2 审计 §3.1 表一致）、
N_eff=8=8、日志实测（rollback L334-335 / triton 36 行生效 / A 无标记）、LR 三份同哈希、
data_id 四态、judge（vs B=WINDOW_OK / vs A=NOT_COMPARABLE 带冲突 / verdict_id 确定性）、
registry 链 append/verify/篡改拒绝/重复 tag 拒绝、A 帖提取防漂移 check。

**在 A2 上的验收命令（建议单条，真实 COCO config + 真实日志跑指纹门）**
```bash
cd /root/skills_R2/skill_R2-SK04 && \
python3 scripts/fingerprint_cfg.py --config-from-log runs/run_<tag>/train.log --baseline officialB \
  --data-json /root/coco_dl/extracted/dataset/annotations_slim.json --out evidence/fp.json \
&& python3 scripts/fingerprint_observed.py --log runs/run_<tag>/train.log \
  --declared evidence/fp.json --lr-ref-csv ../skill_round2_SM3/reference_official_100steps.csv \
  --out evidence/obs.json \
&& python3 scripts/judge_comparable.py --run evidence/fp.json --observed evidence/obs.json \
  --baseline officialB --out evidence/verdict.json \
&& python3 scripts/registry_append.py --tag run_<tag> --run-dir runs/run_<tag> \
  --manifest runs/run_<tag>/manifest.json --fingerprint evidence/fp.json \
  --observed evidence/obs.json \
&& python3 scripts/judge_comparable.py --run evidence/fp.json --observed evidence/obs.json \
  --baseline officialA --out evidence/verdict_vs_A.json   # 双基准标注（A2 审计纪律）
# run_train 前置门形态（--gate：0=绿 / 3=黄(仅窗口，可跑) / 4=红(exit 4 不烧 GPU)）：
python3 scripts/fingerprint_cfg.py --config runs/run_<tag>/config.yaml --baseline officialB \
  --world-size 1 --gate || echo "gate_rc=$?"   # 4 即红牌：不喂 S-M3、别烧 GPU
```
期望机器行：`FINGERPRINT_OK … level=window_feasible …` +
`OBSERVED_OK … kernel_status=rollback_observed …`（若仍 eager/3.2.0）+
`JUDGE_OK verdict=WINDOW_OK … verdict_id=<16hex>`；**若官方最终确认 B=验收基准，报告引用
verdict_id；逐点 <2% 的宣称只允许出现在 POINTWISE_OK（绿）之后（当前诚实结论 = 窗口口径）。**

## 4. 失败排查

| 现象 | 原因/解法 |
|---|---|
| `FATAL log 不存在/配置文件不存在` | 路径错或未上 A2；先跑 §1 核对命令 |
| `FATAL 缺 PyYAML…` | `pip install pyyaml`（仅 fingerprint_cfg/judge 需要；基线 yaml 必须 pyyaml） |
| `config 非合法 YAML…若目标是训练日志，请用 --config-from-log` | 把训练日志当 yaml 传了；或 run config 含官方 `{*}` glob → 官方文本风格 yaml 用 pyyaml 直接读（run 产物）没问题；日志则用 `--config-from-log` |
| 偏离表出现 `unverified` | 单侧未设置/缺省语义未钉死（如 gdn 缺省 vs 显式 eager、A 未设 gdn）——**设计如此**：不作等值断言；补齐证据或换基线 |
| `--data-json` 报 unresolved | 官方/对侧文件不可得（annotations_slim vs output_llava_coco_data）→ 诚实标 unresolved；拿到官方文件后 data_id.py 输出回填基线 yaml 的 byte_sha256 |
| `judge … NOT_COMPARABLE(红)` | 带冲突（run step1 ≈1.x B 带 vs A 带 12.x）或 N_eff/GBS 不可比 → 按审计纪律：与 A 双基准标注、不喂 S-M3；同尺度对照挂 officialB |
| `REGISTRY_CHAIN_BROKEN` | 账本被改/手工编辑 → 用 --verify 定位断链条；无法修复则该账本弃用（append-only 语义），另起 genesis |
| `REGISTRY_DUP_TAG` | tag 重复 → 换 tag；账本只允许追加 |
| `REGISTRY_PENDING` | 写失败 → 不阻塞：run 目录证据仍在，重试或人工补录 |
| `kernel_status=rollback_observed` | triton 未上 NPU（3.2.0 回落，A2 §3.4）：先升 triton-ascend 3.2.1（R2-SK03），再宣称 triton 路径 |

## 5. 回滚

- 本 skill 不修改任何既有文件 → 回滚 = 删除本目录：
  ```bash
  rm -rf /root/skills_R2/skill_R2-SK04      # 本地删 code/skills/R2/skill_R2-SK04
  ```
- 账本/证据如需保留先另存：`evidence/registry.json` 与各 `evidence/*.json` 为纯新增产物，
  可整体带走（哈希链自校验）。

---

## 6. 设计要点与诚实标注（必读）

**7+2 维指纹**（维度= A2 审计 §3.1 偏离清单；mbs/gas/dp 一项拆成三行便于机器比较）：
`shuffle / freeze / cutoff_len / mbs / gas / dp / gdn_causal / 数据集文件 / num_workers`
（共 9 行；数据集文件维度比较 basename=声明层，字节/顺序身份另由 data_id 判）；
派生：`N_eff=world×mbs×gas`（每步全局样本数）、`loss 打印语义`（dp=1 → 全 batch 均值；
dp>1 → rank0 本地均值，A2 §3.2）。

**可达性谓词语义（spec 口径，必要不充分）**：
`pointwise_feasible` = 9 维全等（含 match_declared）且 N_eff 同；`window_feasible` =
GBS_eff=8（与基线同为 8）；否则 `none`。**证据门**（data_identity/sample_order/
kernel_evidence/loss_band/lr_seq）把"声明的逐点可比"与"可宣称逐点 PASS"分开——门未闭
一律黄/红，不许宣称逐点 <2%（本地实测：COCO 首跑 vs B 得 WINDOW_OK；vs A 因带冲突
NOT_COMPARABLE，与 A2 审计结论一致）。

**基准声明出处（本 skill 的指纹值全部可回溯）**：
- officialA.yaml：hiascend 官方账号"赛事小助手"帖《精度测试日志》2026-06-27 内嵌日志
  （官方性 confirmed）；逐维出处 = `evidence/officialA_configuration_details.txt` 行号
  （extract_postA_log.py 生成，--check 防漂移）。注意 A 帖原文 **freeze=[]、gdn 缺省、
  loss_type=raw、mbs1/gas8/单卡/annotations_slim** —— 与 2026-07 的 B/PDF（triton 版）
  不同阶段，勿混写。
- officialB.yaml：docs/official/triton精度日志.txt（2026-07-24）行号 + 官方 PDF §3.3
  旁证（official_templates_parsed.md）；**officiality=proposed**（A2 审计：B 的官方身份
  未证实——chaspark 帖登录墙未核、hiascend 官方日志无 Triton 字样）；确认动作单见
  officialB.yaml 文件头。报告在官方答复前一律双基准标注。

**本地验证 vs 待 A2 实跑（诚实清单）**：
1. ✅ 本地已用**真实日志**验证机制：A2 COCO 首跑日志（rollback 证据 L334-335、step1=1.869898、
   墙钟 38.8min、LR 与官方同哈希）、官方 B triton 日志（36 行 NPU triton 生效）、官方 A 提取件
   （12.42 带、无 kernel 标记）——数字均来自日志原文解析，非编造；
2. ⏳ 待执行者上 A2：用**真实 COCO config + 真实日志**按 §3 验收命令跑一遍并 registry 落账
   （本包 evidence/registry.json 目前为空链 genesis）；
3. ⏳ 数据集字节身份：A2 的 annotations_slim.json 与官方 A 同名文件是否同字节 → 需官方文件
   或官方哈希；B 的 output_llava_coco_data.json 不可得 → unresolved（诚实）；
4. ⏳ judge 的 POINTWISE_OK 目前无真实触发样本（构造夹具可触发 yellow NEEDS_EVIDENCE；
   真绿需 7 维全等+门全闭——官方确认 + 数据字节 + triton 3.2.1 生效后才会出现）；
5. ⏳ `sampler_probe.py`（逐步 batch 组成枚举）不在本轮交付清单（R2-C6-02 提到）——逐点
   batch 相等目前用"shuffle=False + 数据字节一致"推定，README 与 JSON 均标注"建议 sampler_probe 复核"。

**退出码汇总**：fingerprint_cfg 0=产出（--gate: 0 绿/3 黄/4 红）；data_id 0=同源类/1=不同源
或未核对；observed 0=产出/2=参数 IO/3=无可解析迭代行；judge 默认 0=裁决成功产出
（--gate: 0 绿/3 黄/4 红），2=输入错；registry 0=OK/2=IO(REGISTRY_PENDING)/3=链损坏或
重复 tag/4=被引用文件缺失；selfcheck 0=全过。

**编码纪律**：无随机、无墙钟（registry 的 ts 是账本记录语义，除外）、判定逻辑单一来源
（judge import fingerprint_cfg 的 cmp/可达性函数，防口径漂移）；迭代行正则与
S-M3 compare.py / R1-SK01 三处同源，改动必须三处同步。
