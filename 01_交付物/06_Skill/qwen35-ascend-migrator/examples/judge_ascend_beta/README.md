# 新机器（昇腾 · CANN 9.1.0-beta.3）判定证据

本目录是 **同一份 Skill 在另一套软硬件栈上**跑出的判定证据，与 `examples/judge/`
（A3 · 旧机器 · CANN GA）**并列存放、互不覆盖**。

| 项 | A3（旧机器） | 本目录（新机器） |
|---|---|---|
| CANN | 9.1.0（GA） | **9.1.0-beta.3** |
| driver | 25.5.0 | 25.5.1 |
| Python | 3.10.12 | **3.11.6** |
| 依赖 | 部分本地补丁 | **纯 pip 安装** |
| `level` | `pointwise_feasible` | `pointwise_feasible` |
| `open_gates` | `[data_identity, sample_order]` | `[data_identity, sample_order]` |
| `deviations` / `unverified` | 0 / 0 | 0 / 0 |
| `data_same_source.status` | `declared_same_file_byte_unverified` | 同 |
| `N_eff` | 8/8 | 8/8 |
| `verdict_id` | `606056b3cb04ecbb…` | 见下（**不同且必须不同**） |

> **`verdict_id` 说明**：它是"本批不可变证据字节"的摘要，两组证据字节不同 →
> 两个 id **必然不同**，且**不得互相引用**（C6-03 纪律）。
>
> **口径**：这里的 `level=pointwise_feasible` 指"逐点比对在口径上是可达的"
> （9 维配置全等、N_eff 同为 8），**不等于"逐点数值已复现"**；
> 两道证据门（官方数据字节不可得 → 仅**构造性同源**；样本序）仍未闭合，
> 故裁决名为 `NEEDS_EVIDENCE`（黄）。**两个环境在这一点上完全一致。**

---

## ★ 文件来源与自洽性（**取证也要同源** —— 坑 122）

本目录文件由远端脚本按**路径**收集，而 `out/**` 下的路径**每次运行都会被覆盖**
→ 必须先核对跨文件是否自洽，否则会把两次运行的产物当成一次的证据引用。

| 文件 | 远端来源 | 所属运行 |
|---|---|---|
| `verdict.json` / `fp.json` / `obs.json` / `judge_summary.json` / `loss_compare.csv` | `out/judge/*` | 末次 P7 判定 |
| `window_50_100.json` / `round_1_baseline.json` | `out/bench/*` | 末次 P6（`--from PRE`，rc=0） |
| **`loss_series.csv`** | **`out/train/loss_series.csv`** | ⚠ **首次** 100 步（收尾保存崩、rc≠0 那次） |
| `env.json` / `train_config.yaml` / `_verdicts.tsv` / `from_zero_report.md` | `out/probe` `out/plan` `out/logs` | 末次运行 |
| `registry.json` | `sk04_judge/evidence/registry.json` | 累加账本 |

**已知不自洽项（必须披露，勿当成一次运行的证据）**

* `window_50_100.json` 声明 `median_ms=925.4`，其 `csv` 字段指向 `out/bench/loss_series.csv`（**未收录**）；
* 本目录 `loss_series.csv` 复算 `median_ms=941.6`（来自**另一条** `out/train/` 路径）；
* 两者 `step1_grad_norm` = `134.288` vs `134.3` → **确认是两次不同运行**。

> **副产品（有用）**：同配置两次 100 步窗口中位 `941.6` vs `925.4` → **run-to-run 差约 1.7%**，
> 即性能对比的**噪声底**。新机器与 A3 的 **2.25×** 差距**远大于**该噪声 → **不是抖动造成的**。
>
> 复核器主张 **C15** 会检查"跨文件一致性，或差异有明确来源解释"。
> 长期修法：取证脚本应**按运行标签**收集，而不是按会被覆盖的路径。