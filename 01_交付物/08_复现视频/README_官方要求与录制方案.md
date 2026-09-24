# 08_复现视频（官方可选项）

> 官方交付清单第 8 项：全程复现的视频录制（从开始模型迁移到跑出 log 的全过程，
> 交付件格式 .MP4）——**标注为可选**，本项目尚未录制。

## 建议录制内容（3–5 分钟，终端录屏为主）

| 段落 | 时长 | 内容 |
|---|---|---|
| 1 | 30s | 问题背景：Qwen3.5-0.8B 的 GatedDeltaNet 在昇腾上无现成实现 → 迁移点识别（49 处命中） |
| 2 | 60s | **Skill 一键迁移**：`selfcheck.py` → `00_probe_env.py` → `20_plan_migration.py`（屏幕录制，字幕标注阶段与判据） |
| 3 | 45s | 数值正确性：第 1 步 loss 与官方逐位一致（1.924621）+ 跨卡等价证据 |
| 4 | 45s | 性能：八级优化 51.1×，50–100 步中位 419.0ms 优于官方 431.3ms |
| 5 | 45s | 判定链：`70_judge.py` 输出 verdict + `registry_append.py --verify` 校验哈希链 |
| 6 | 30s | 场景：工业 SOP 判定 JSON 输出（**待补**，见总 README §四） |

## 录制命令序列（照此执行即可）

```bash
cd qwen35-ascend-migrator
python3 selfcheck.py                                    # SELFCHECK_OK（结构/判定链/环境）
python3 scripts/00_probe_env.py                         # P0 探测：路径 + 配置档
python3 scripts/20_plan_migration.py --env out/probe/env.json   # P2 生成配置
python3 scripts/70_judge.py --log examples/train/train.log \
    --baseline officialB --baseline-log examples/train/official_baseline.log --out out/judge
cd sk04_judge && python3 scripts/registry_append.py --verify     # 哈希链校验
```

> 录制完成后，将 .MP4 放入本目录，并把上表"状态"更新为已完成。
