# SMoPE 遗忘来源验证实验运行手册

## 1. 固定基线

本实验组以 `master` 的 `a93fa54` 为唯一代码基线，使用相同的：

- CIFAR-100 十任务划分；
- `prompt_param 50 5 1e-5 1e-5 0.4`；
- seeds `0 1 2`；
- `CRCT_EPOCHS=50`；
- `MAX_TASK=10`；
- 每个实验独立 `OUTDIR`。

任何一项不一致时，不做横向归因比较。

## 2. 先做两任务冒烟验证

先确认数据、预训练权重、Router 参考轨迹和历史回放路径可以完整执行：

```bash
MAX_TASK=2 REPEAT=1 CRCT_EPOCHS=1 AUDIT_MAX_SAMPLES=256 GPUID=0 \
OUTDIR=outputs/cifar-100/smoke/forgetting-audit-router \
bash experiments/cifar-100_forgetting_audit_router.sh
```

成功标志：

- `output.log` 完成两个任务；
- `forgetting_audit/router_audit.jsonl` 出现 `historical_router_replay`；
- 日志包含 `route_change=` 和 `replay_gain=`；
- `forgetting_audit/repeat-1/router_reference/` 有 Task 1、Task 2 的 `.pt` 文件。

冒烟试验只验证实现和数据链路，不用于论文结论。

## 3. 正式实验顺序

### A. Router 基线与恢复实验（最高优先级）

```bash
GPUID=0 bash experiments/cifar-100_forgetting_audit_router.sh
```

这一次训练同时产生：

1. 原始 FAA、CAA、FR 和 Accuracy Matrix；
2. 每个任务刚学完时的历史 top-k Router 决策；
3. Task 5、Task 10 的 Router Change Rate；
4. 固定最终 Key/Value/Classifier、只回放历史 Router 的恢复准确率；
5. Key、Value、Classifier 相对各历史任务检查点的参数漂移。

默认保存 top-k expert 与对应 score，不保存全部 logits。若需要对少量样本检查完整 logits：

```bash
SAVE_ROUTER_LOGITS=1 AUDIT_MAX_SAMPLES=500 REPEAT=1 GPUID=0 \
OUTDIR=outputs/cifar-100/10-task/forgetting-audit-router-logits \
bash experiments/cifar-100_forgetting_audit_router.sh
```

不要对全量三种子默认开启完整 logits，文件会明显增大。

### B. 冻结干预实验

有多张 GPU 时分别启动：

```bash
GPUID=1 bash experiments/cifar-100_forgetting_audit_freeze_prompt.sh
GPUID=2 bash experiments/cifar-100_forgetting_audit_freeze_key.sh
GPUID=3 bash experiments/cifar-100_forgetting_audit_freeze_value.sh
GPUID=4 bash experiments/cifar-100_forgetting_audit_freeze_classifier.sh
```

只有一张 GPU 时按上述顺序串行运行，并都设 `GPUID=0`。

- `prompt`：Task 2 起不再更新任何 Prompt expert；
- `key`：Task 2 起冻结 `e_pk_*`，Value 仍可学习；
- `value`：Task 2 起冻结 `e_pv_*`，Key 仍可学习；
- `classifier`：Task 2 起冻结旧类别行，但允许新类别行学习，避免“整个分类头冻结后新类完全学不会”的混淆。

冻结改变了优化路径，只能证明模块参与遗忘，不能单独证明它是唯一来源。

## 4. 汇总结果

五个实验完成后运行：

```bash
python utils/summarize_forgetting_audit.py \
  outputs/cifar-100/10-task/forgetting-audit-router \
  outputs/cifar-100/10-task/forgetting-audit-freeze-prompt \
  outputs/cifar-100/10-task/forgetting-audit-freeze-key \
  outputs/cifar-100/10-task/forgetting-audit-freeze-value \
  outputs/cifar-100/10-task/forgetting-audit-freeze-classifier \
  --output outputs/cifar-100/10-task/forgetting_audit_summary.csv
```

本仓库的统计口径：

- `FAA`：最后一个训练阶段的平均任务准确率；
- `CAA`：各训练阶段平均准确率的均值；
- `FR`：最后阶段的平均遗忘率；
- Accuracy Matrix：`results-acc/pt.yaml`；
- Router/Replay：`forgetting_audit/router_audit.jsonl`；
- 参数漂移：`forgetting_audit/component_drift.jsonl`。

`results-acc/pt.yaml` 中：

- `history` 保留每个 repeat/seed；
- `mean` 是已完成 repeats 的均值，不是某个单独 seed。

## 5. 判读逻辑

### Router 是主要来源

需要同时看到：

- 旧任务 Router Change Rate 较高；
- 历史 Router 回放在多个旧任务、多个 seed 上稳定恢复准确率；
- 回放只替换 top-k 访问路径，不恢复 Key、Value 或 Classifier 参数。

重点报告 Task 10 的旧任务平均 `replay_gain`，同时列出逐任务结果，不能只报总均值。

### Key 漂移参与遗忘

需要结合：

- Key 的 `relative_l2` 随任务增加；
- 冻结 Key 相对基线降低 FR；
- FAA/CAA 和新任务对角线没有出现不可接受的塑性损失；
- Router 回放不能解释全部丢失性能。

### Value 覆盖参与遗忘

需要结合：

- Value 的 `relative_l2` 明显增加；
- 冻结 Value 降低 FR；
- 历史 Router 回放后仍有较大未恢复差距。

### Classifier 冲突参与遗忘

需要结合：

- Classifier 漂移明显；
- 冻结旧类别行降低 FR；
- 新任务类别行仍能学习；
- 同时检查 Accuracy Matrix 的新任务对角线，排除仅靠牺牲新任务学习换取低 FR。

## 6. 最低结论标准

不要用单 seed、单任务或约 `0.1%` 的差异下结论。至少要求三种子方向一致，并同时报告：

- FAA / CAA / FR 的 mean 与 std；
- Task 10 的逐旧任务 Router Change Rate；
- Task 10 的逐旧任务 Historical Replay Gain；
- Key / Value / Classifier 漂移；
- 冻结实验的新任务塑性代价。

每次重跑建议使用新的 `OUTDIR`，避免追加式 `output.log` 混入旧文本。
