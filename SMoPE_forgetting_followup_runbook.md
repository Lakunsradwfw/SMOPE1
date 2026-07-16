# SMoPE 遗忘审计后续实验运行说明

## 1. 层×头内 Top-5 重复选择结论

统计范围为 baseline 最终 checkpoint 10、任务 1–10、seed 0–2、每任务 1000 个审计样本。

- 共有 216 个 `seed × layer × head` 专家池。
- 208/216 个池在全部十个任务中只使用 5 个专家。
- Top-5 对全部选择次数的平均覆盖率为 99.974%；208/216 个池为严格 100%。
- 210/216 个池在十个任务上的活跃专家集合完全相同。
- 2160 个 `seed × layer × head × task` 组合中，2095 个组合都是同 5 个专家分别被全部 1000 个样本选择。
- 同一 seed 内跨任务的专家集合平均 Jaccard 为 0.9975；不同 seed 间的 Top-5 集合平均 Jaccard 只有 0.1205。

因此，当前 Router 更像在每个 seed 的早期随机确定一组专家后长期锁定，而不是根据任务或样本进行明显的专家分工。不同 seed 选择的固定集合又明显不同，说明存在较强的初始化/对称性破缺效应。

## 2. Key+Value 联合冻结：先跑已有 seed 0–2

该脚本默认读取现有 baseline 的 manifest，并在找不到时直接停止：

```bash
GPUID=0 bash experiments/cifar-100_forgetting_audit_freeze_key_value.sh
```

输出：

```text
outputs/cifar-100/10-task/forgetting-audit-freeze-key-value
```

## 3. 增加 seed 3、4

必须先跑新的 seeds34 baseline，因为旧 manifest 只对应原来的 seed 集合：

```bash
GPUID=0 bash experiments/cifar-100_forgetting_audit_router_seeds34.sh
```

baseline 完成后，其余三个条件可以放到不同 GPU：

```bash
GPUID=1 bash experiments/cifar-100_forgetting_audit_freeze_key_seeds34.sh
GPUID=2 bash experiments/cifar-100_forgetting_audit_freeze_value_seeds34.sh
GPUID=3 bash experiments/cifar-100_forgetting_audit_freeze_key_value_seeds34.sh
```

优先级：baseline、Key、Key+Value 为必跑；Value 的前三个 seed 已经 3/3 变差，因此 seed 3、4 主要用于加强“排除 Value 单独为遗忘来源”的证据，可以在算力不足时最后运行。

## 4. 合并为五 seed 配对结果

以上目录全部完成后：

```bash
bash experiments/cifar-100_forgetting_audit_merge_seeds01234.sh
```

默认生成 summary-only 合并目录：

```text
outputs/cifar-100/10-task/forgetting-audit-five-seed/
```

该目录只合并指标、manifest 和顶层审计记录，不复制几十 GB 的模型 checkpoint。五 seed 汇总会启用配对显著性检验。每次正式重跑应使用新的 `MERGED_ROOT`，不要复用旧合并目录。

如果不运行 Value 的 seed 3、4：

```bash
INCLUDE_VALUE=0 bash experiments/cifar-100_forgetting_audit_merge_seeds01234.sh
```

## 5. Matched-plasticity 对照

对照不冻结任何组件，只把 OnePrompt 主训练轮数从原来的 20 降低到候选值。默认先尝试 17、18 epoch，并沿用 baseline manifest、seed 0–2、任务顺序和其余设置：

```bash
GPUID=0 bash experiments/cifar-100_forgetting_audit_matched_plasticity.sh
```

输出：

```text
outputs/cifar-100/10-task/matched-plasticity-controls/
```

自动选择器按 seed 比较任务 2–10 的 Accuracy Matrix 对角线损失：

- 平均 seed 级匹配误差默认不得超过 0.35 个百分点；
- 最大单 seed 误差不得超过 0.70 点；
- 未达到阈值时脚本以非零状态退出，并明确写出最近候选，不能把它称为 matched control。

若 17、18 均未匹配，可在新的输出目录补跑相邻 epoch：

```bash
CONTROL_ROOT=outputs/cifar-100/10-task/matched-plasticity-controls-round2 \
MAIN_EPOCHS_LIST="16 19" GPUID=0 \
EXTRA_CANDIDATES="outputs/cifar-100/10-task/matched-plasticity-controls/baseline-main-epochs-17 outputs/cifar-100/10-task/matched-plasticity-controls/baseline-main-epochs-18" \
bash experiments/cifar-100_forgetting_audit_matched_plasticity.sh
```

正式判读看 `matched_plasticity_report.md` 的 `matched retention gain`：

- 大于 0：冻结条件在相同可塑性损失下保留旧任务更好，支持该组件存在保护效应；
- 接近 0：FR 下降基本可由学习不足解释；
- 小于 0：即便匹配可塑性，冻结仍更差，不支持该组件是应被阻止更新的遗忘源。

matched-plasticity 当前主要用于 Prompt/Value 的三 seed 机制判定，不应单独写成五 seed 统计显著结论。
