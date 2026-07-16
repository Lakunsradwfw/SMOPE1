# SMoPE 遗忘来源验证实验运行手册（审查基础设施 v2）

## 1. 实验口径

代码基线为 `16e1f55`，算法对照基线为 `a93fa54`。所有横向比较必须保持：相同 seed、任务顺序、数据划分、审查样本、`MAX_TASK`、`CRCT_EPOCHS`、`prompt_param` 和 checkpoint。

本实现把证据分为四类，不能混写：

1. 自然训练中的参数/路由变化；
2. 更新方向对旧任务损失的局部影响；
3. 最终模型的组件恢复反事实；
4. Freeze 训练干预及其塑性代价。

未传任何 `--audit_*` 参数时，默认 SMoPE 前向、训练参数集合和输出保持原路径。梯度方向诊断默认关闭；它使用固定、无增强的训练集 audit 子集，不使用测试集梯度做训练或调参决策。

## 2. 先跑两任务冒烟

先按 `README.md` 准备 CIFAR-100 和
`pretrained/vit_base_patch16_224_augreg2_in21k_ft_in1k.bin`。当前审查脚本不会自动下载预训练权重。

```bash
GPUID=0 bash experiments/cifar-100_forgetting_audit_smoke.sh
```

验收：

- 两个任务训练和评估完整结束；
- `router_audit.jsonl` 同时含 `identity_replay_accuracy` 和 `identity_prompt_logits_replay_accuracy`；
- `accuracy_long.csv`、`router_long.csv`、`drift_long.csv` 可在汇总后生成；
- Task 1 在 Task 2 checkpoint 的路由回放可执行；
- `models/repeat-1/task-*/checkpoint.pt` 存在；
- 同 checkpoint 自回放时，路由 change rate 为 0、Jaccard 为 1、两级回放与自动前向一致（允许浮点误差）。

冒烟只验证实现链路，不用于论文结论。

## 3. 再跑三任务配对交叉检查

单卡串行：

```bash
GPUID=0 bash experiments/cifar-100_forgetting_audit_crosscheck.sh
```

默认运行 baseline、freeze-key、freeze-value、freeze-classifier，配置为 `MAX_TASK=3 REPEAT=2 CRCT_EPOCHS=2 AUDIT_MAX_SAMPLES=128`。检查：

- Freeze 条件复用 baseline 的 sample manifest；
- 两个 seed 均能一一配对；
- Freeze Key/Value 后仍有 Router Change；
- Task 1 `plasticity_loss` 接近 0；
- `requires_grad=False` 的冻结参数在反向后 `.grad is None`；
- 低 FR 若没有最终旧任务保留收益，会产生 `LOWER_FR_WITHOUT_RETENTION_GAIN`。

## 4. 正式 baseline 与 Freeze 实验

脚本默认三 seed，适合机制筛查：

```bash
GPUID=0 bash experiments/cifar-100_forgetting_audit_router.sh
GPUID=1 bash experiments/cifar-100_forgetting_audit_freeze_prompt.sh
GPUID=2 bash experiments/cifar-100_forgetting_audit_freeze_key.sh
GPUID=3 bash experiments/cifar-100_forgetting_audit_freeze_value.sh
GPUID=4 bash experiments/cifar-100_forgetting_audit_freeze_classifier.sh
```

正式统计建议至少五个配对 seed：

```bash
REPEAT=5 SEEDS="0 1 2 3 4" GPUID=0 bash experiments/cifar-100_forgetting_audit_router.sh
```

其余 Freeze 脚本使用相同的 `REPEAT`、`SEEDS`、`MAX_TASK`、`CRCT_EPOCHS` 和 `AUDIT_MAX_SAMPLES`。如果 baseline 已先跑完，可让 Freeze 显式校验同一 manifest：

```bash
AUDIT_SAMPLE_MANIFEST=outputs/cifar-100/10-task/forgetting-audit-router/forgetting_audit/audit_sample_manifest.json \
GPUID=2 bash experiments/cifar-100_forgetting_audit_freeze_key.sh
```

若多卡并行、baseline manifest 尚不存在，各条件会按确定性顺序生成自己的 manifest；最终汇总会逐 seed、逐任务核对样本 ID，不一致即报错。

审查脚本默认在每个 repeat 的 Accuracy Matrix 已评估、全部结果 YAML 已写入后，删除该 repeat 的
`models/repeat-*/task-*/class.pth` 临时完整模型副本。若需要保留它们用于中断续跑，可显式传入
`AUDIT_CLEANUP_CLASS_CHECKPOINTS=0`；这不会影响已完成 repeat 的汇总结果。

清理服务器已有输出时，先预览再执行：

```bash
bash experiments/cleanup_forgetting_audit_outputs.sh --root outputs/cifar-100/10-task
bash experiments/cleanup_forgetting_audit_outputs.sh --root outputs/cifar-100/10-task --apply
```

该脚本只清理三份结果 YAML 均有效且 repeat 数一致的已完成 repeat；始终保留
`forgetting-audit-router` 的 `checkpoint.pt`。如需清理旧梯度等非 baseline 条件曾保存的完整
checkpoint，再额外加入 `--delete-nonbaseline-checkpoints`。

## 5. 梯度方向诊断

该诊断明显增加任务边界耗时和显存，只在自然训练 baseline 上单独执行：

```bash
GPUID=0 AUDIT_MAX_SAMPLES=256 bash experiments/cifar-100_forgetting_audit_gradient.sh
```

梯度脚本默认不保存 `checkpoint.pt`，因为组件恢复只在 router baseline 上执行；如确有单独恢复梯度
实验模型的需要，再显式设置 `AUDIT_SAVE_FULL_CHECKPOINTS=1`。

输出 `gradient_direction.jsonl`，区分：

- `main_prompt_training`；
- `classifier_correction`。

核心字段为真实 AdamW 更新 `delta_theta` 对旧任务梯度的 `first_order_old_loss_change`、`update_old_gradient_cosine`、新旧梯度余弦，以及 Key 的 router margin/route flip。诊断通过 `torch.autograd.grad` 完成，不写入训练 `.grad`。

## 6. 最终模型组件恢复

baseline 必须用 `AUDIT_SAVE_FULL_CHECKPOINTS=1` 训练（router 脚本默认开启）。训练完成后：

```bash
GPUID=0 RUN_DIR=outputs/cifar-100/10-task/forgetting-audit-router \
bash experiments/cifar-100_forgetting_audit_restore.sh
```

恢复条件包括：

- final；
- router identity；
- router identity + historical Prompt logits；
- Key / Value / Classifier；
- Key+Value、Router+Value、Router+Classifier、Key+Value+Classifier；
- full historical checkpoint。

Key/Value 同时评估 `full_pool`、`used_experts` 和覆盖 90% 历史访问量的 `high_frequency_experts`。每个条件退出后检查状态回滚；负恢复 gain 和大于 1 的恢复比例均保留。

组件参考张量默认仅在训练进程内保留，用于计算 `component_drift.jsonl`，不会再写入
`component_reference/*.pt`。只有排查实现问题时才传入 `--audit_save_component_references` 持久化它们。

## 7. 统一汇总

五个 matched 条件完成后：

```bash
bash experiments/cifar-100_forgetting_audit_summarize.sh
```

输出目录默认为 `outputs/cifar-100/10-task/forgetting-audit-summary/`，包含：

- `accuracy_long.csv`：完整 Accuracy Matrix、对角线、最终准确率、塑性损失、最终保留收益；
- `router_long.csv`：逐 seed、逐任务、逐层两级回放和 baseline 路由分歧；
- `drift_long.csv`：全池、旧任务使用专家、高频专家漂移；
- `expert_usage_long.csv` / `expert_ownership_long.csv`：软专家所有权；
- `paired_condition_summary.csv`：FAA/CAA/FR 与 matched delta；
- `forgetting_audit_report.md`：配置检查、自动警告和允许/禁止结论。

默认阈值单位为准确率百分点：

```bash
PLASTICITY_WARN_THRESHOLD=2.0 RETENTION_GAIN_THRESHOLD=1.0 \
bash experiments/cifar-100_forgetting_audit_summarize.sh
```

少于五个 seed 时只报告描述性 paired effect，不使用“统计显著优于”措辞。

## 8. 判读顺序

先看 `paired_condition_summary.csv`：Freeze 是否真的提高最终旧任务准确率，而不是仅通过降低新任务对角线让 FR 变小。再看 `router_long.csv` 的两级回放，区分专家身份改变和 Prompt 注意力强度改变。随后用组件恢复和方向诊断检查功能因果链，最后才结合 Freeze 干预。

只有以下证据方向一致时，才可说某组件是“重要贡献来源之一”：

- 参数或路由确实变化；
- 更新方向局部增加旧任务损失；
- 最终模型恢复历史状态能恢复旧任务准确率；
- Freeze 提高最终旧任务保留；
- 新任务塑性代价较小，或有 matched-plasticity 对照排除。

每次正式重跑使用新的 `OUTDIR`；仓库日志和 JSONL 采用追加式写入，复用旧目录会污染比较。
