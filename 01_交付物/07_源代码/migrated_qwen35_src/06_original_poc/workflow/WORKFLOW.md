# DSH 迁移工作流 · 使用指南

> 搭建日期：2026-08-30
> 依据：昇腾官方迁移技能集规范（五层架构+门禁机制）× 本项目特征 × DSH headless 能力
> 定位：把《DSH实战指南》的方法论变成 DSH 可直接执行的结构化工作流

---

## 一、目录结构

```
workflow/
├── WORKFLOW.md              ← 本文件（从这里开始读）
├── skills/                  ← 6 个技能（DSH 的 SKILL.md 规范）
│   ├── 00-main/             ← 总控：编排 T0-T8 阶段门禁
│   ├── 01-env-gate/         ← 环境自检与版本配套门禁
│   ├── 02-op-migrate/       ← 算子迁移三级决策
│   ├── 03-verify/           ← 数值精度验证
│   ├── 04-perf/             ← 性能优化循环
│   └── 05-package/          ← 复赛 Skill 封装与交付物清单
├── rules/
│   └── ascend-facts.md      ← 昇腾事实对账清单与纪律（每个会话必读）
├── prompts/                 ← T0-T8 九个阶段提示词模板（复制即用）
└── run/
    ├── daily-check.ps1      ← headless 每日体检脚本（本地执行）
    └── npu-run-template.sh  ← 昇腾环境运行模板（远端执行）
```

## 二、两种使用方式

### 方式 A：Web 界面交互式（主力）

1. `dsh web` → 打开 http://127.0.0.1:3080/
2. Choose workspace → 选择 `C:\Users\HUAWEI\Desktop\qwen3_5-migrate-poc`
3. 每个新会话的**第一条消息**固定为：
   > "先阅读 workflow/rules/ascend-facts.md 和 workflow/skills/00-main/SKILL.md，然后按总控技能执行当前阶段任务。"
4. 当前处于哪个阶段，就把 `workflow/prompts/Tx-xxx.md` 的内容复制进去执行。

### 方式 B：headless 自动化（例行任务）

```powershell
# 每日体检：汇总今日日志，产出检查报告
dsh --profile headless "阅读 workflow/skills/03-verify/SKILL.md，检查 logs/ 目录今日新增日志，按技能中的日报格式汇总到 reports/daily_check.md"
```

## 三、工作流总流程（阶段门禁制）

```
T0 环境门禁 ──通过──> T1 官方样例基线 ──跑通──> T2 权重转换 ──加载成功──>
T3 标准算子迁移 ──精度过──> T4 GDN算子 ──精度过──> T5 端到端冒烟 ──输出正常──>
T6 COCO验证 ──达标──> T7 性能循环 ──出数据──> T8 Skill封装 ──一键跑通──> 交付
```

**门禁规则（总控技能强制执行）**：
- 每个阶段必须产出"证据产物"（日志/报告/数据表），无证据不得进入下一阶段；
- 任何阶段的验证失败 → 回到上一阶段修复，禁止跳过；
- 昇腾事实一律先查 rules/ascend-facts.md 对账，对不上的标"存疑"不进代码。

## 四、与双环境开发的配合

| 环节 | 在哪做 | 用什么 |
|---|---|---|
| 读代码、改代码、审查 | 本地 Windows | dsh web + 通义灵码 |
| 跑迁移、跑验证、采性能 | 昇腾 910B（Linux） | run/npu-run-template.sh |
| 分析远端日志、形成下轮假设 | 本地 | dsh（读拉回的日志） |
| 每日汇总 | 本地 | run/daily-check.ps1（headless） |

**铁律**：远端运行前必须 `git commit`；所有运行加 `2>&1 | tee logs/<时间戳>_<任务>.log`；日志当日拉回。

## 五、工作流 → 复赛交付物的映射

| 工作流产物 | 对应复赛交付物 |
|---|---|
| 03-verify 的对比日志 | 精度分析报告 + 原始日志 |
| 04-perf 的 perf_table.md | 性能测试报告 |
| 05-package 封装的 zip | Skill 文件夹（核心交付物） |
| 全程 logs/ | 原始日志压缩包 |
| T7 复现录屏 | 全程复现视频 |

## 六、注意事项

1. **技能自动加载**：DSH 若未自动发现 skills/ 目录，就在会话开头手动让它读对应 SKILL.md（效果相同）；
2. **一个会话一个任务**：不要在同一会话里跨阶段干活，上下文污染是幻觉的温床；
3. **官方样例优先**：执行 T1 后，官方样例代码就是黄金参照系，技能中的"复用优先"原则即来源于此；
4. **这套工作流本身就是交付物雏形**：05-package 阶段会把它整理成复赛要求的 Skill zip——您现在怎么用它，将来就怎么交付它。
