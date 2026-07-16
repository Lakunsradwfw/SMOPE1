# SMoPE 共享 Expert 内跨任务冲突实验指导

## 1. 研究问题与主假设

当前 SMoPE 的不少 layer/head 会长期锁定同一组 Top-5。此时活跃 expert 的 hard usage 可能完全相同，因此不能再把 `usage → conflict` 相关性作为主证据。

本实验验证的新主链条是：

\[
Persistent\ shared\ routing
\rightarrow
Within\text{-}expert\ cross\text{-}task\ conflict
\rightarrow
Harmful\ expert\ update
\rightarrow
Incremental\ task\text{-}conditioned\ drift
\rightarrow
Forgetting
\]

分析单元是：

```text
(seed, old_task, new_task, prompt_layer, attention_head, expert)
```

usage 只用于确认两个任务是否共享同一 expert，不再作为必须具有连续方差的因变量。

### 1.1 Expert 身份和频率统计层级

SMoPE 中 `expert=0` 在不同 layer 或 head 下对应不同参数。因此 expert 的唯一身份必须写成：

```text
(layer, head, expert)
```

频率报告分成两级：

1. **主层级：layer/head 池内。** 分母是该 `(layer,head)` 的全部 Top-k 选择槽位，计算每个 expert 的池内选择占比、活跃任务数、Top-k 集合稳定性和池内熵。
2. **整体描述：对 layer/head 池级指标做汇总。** 报告 72 个池中 Top-k 锁定池的比例、Top-k 质量占比、每池活跃 expert 数和连续任务 Top-k Jaccard。

允许额外给出 1800 个 `(layer,head,expert)` 参数坐标在全局选择次数中的占比，但不能把不同 layer/head 中相同编号的 expert 合并。整体统计不能替代 layer/head 池内结论。

以上两级都会分别报告两个时间口径：

- `learning_boundary`：每个任务刚学习完成时，用该时刻模型记录本任务路由；
- `final_checkpoint`：最终模型对全部任务重新路由，与 `SMoPE_expert_frequency_layer_head.xlsx` 的口径一致。

这两个口径必须分开，不能混合后计算 Top-k 稳定性。

## 2. 每条记录如何定义

### 2.1 同一 expert 内跨任务梯度冲突

在训练 Task `t` 前的同一个参数点，对旧任务 `a<t` 计算：

\[
g_{e,a}=\nabla_{\theta_e}L_a
\]

对新任务计算真实 SMoPE 主训练目标：

\[
g_{e,t}=\nabla_{\theta_e}
\left(L^{masked\ CE}_t+L^{router}_t\right)
\]

其中 `e=(layer, head, expert)`，`θ_e` 同时包含该 expert 的 Key 和 Value。

严格冲突强度为：

\[
C_{e,a,t}=\max(0,-\cos(g_{e,a},g_{e,t}))
\]

零范数梯度 pair 被标记为无效，不得填成 cosine=0。

### 2.2 Same-task split-half 对照

每个任务的固定、分层、无增强 probe 被确定性拆成两个不重叠子集 A/B，分别计算：

\[
C^{same}_{e,a}
=
\max(0,-\cos(g^A_{e,a},g^B_{e,a}))
\]

主端点：

\[
ExcessConflict_{e,a,t}
=
C_{e,a,t}
-
\frac{C^{same}_{e,a}+C^{same}_{e,t}}{2}
\]

这样能够排除有限样本、梯度估计噪声造成的假冲突。

### 2.3 真实参数更新的旧任务损害

主训练完成后记录 expert 的真实更新：

\[
\Delta\theta_{e,t}
=
\theta^{after}_{e,t}-\theta^{before}_{e,t}
\]

expert 级一阶旧损失变化：

\[
H^{pred}_{e,a,t}=g_{e,a}^{\top}\Delta\theta_{e,t}
\]

大于 0 表示该 expert 的真实更新预计增加旧任务损失。

实验还会在完整主训练前后，用相同 probe 测量：

\[
H^{obs}_{a,t}=L_a^{after}-L_a^{before}
\]

`H_obs` 是完整模型层面的 task-pair 结果，不能冒充单 expert 因果效应；它用于检查各 expert 一阶贡献的总和是否对应真实旧任务损害。

### 2.4 Task-conditioned functional drift

SMoPE expert 是共享 Key–Value prompt，不存在显式 task slot。因此任务知识用固定旧任务输入上的条件响应定义。为排除上游 prompt/feature 同时变化的混杂，在旧任务 `a` 的学习边界保存 query `q_a^ref(x)`；以后始终固定这组 query，只替换当前检查点的 expert Key/Value：

\[
z^{s\mid ref(a)}_{e,a}(x)
=softmax(q_a^{ref}(x)(K^s)^\top)_eV_e^s
\]

这里的 `softmax` 在同一 `(layer, head)` 的完整 expert 池上计算。它是连续的 task-conditioned expert functional signature，不是 hard usage 的替代统计，也不会把当前检查点重新计算的 query 混入 expert drift。

主漂移使用当前 Task `t` 单步造成的 incremental drift：

\[
D^{inc}_{e,a,t}
=
d(z^{after\ t}_{e,a},z^{after\ (t-1)}_{e,a})
\]

同时保存从旧任务学习边界到当前的 cumulative drift，但累计漂移不能直接作为 Task `t` 冲突的主结果。

## 3. 输出体积

脚本不会启用：

- `--audit_router`；
- `--audit_save_logits`；
- `--audit_save_full_checkpoints`；
- 逐样本路由或 representation tensor。

固定 query 只在单个 repeat 的内存中保存，不写入输出目录。task-pair 记录使用 `jsonl.gz`。原始训练要求的 `class.pth` 只在当前 repeat 内临时存在；完成逐任务重新加载评估和结果 YAML 写入后自动删除。因此不会再产生旧 router/freeze audit 那种约 10 GB 的长期输出。

## 4. 三任务 smoke

```bash
GPUID=0 bash experiments/cifar-100_expert_interference_smoke.sh
```

默认输出：

```text
outputs/cifar-100/3-task/within-expert-interference-smoke
```

smoke 只验证实现，不是机制证据。必须检查：

1. 三任务完整结束；
2. analyzer 没有报告 usage conservation 或 task-pair join 失败；
3. `old_task=1,new_task=2/3` 和 `old_task=2,new_task=3` 均有记录；
4. split-half 两侧样本数均大于 0；
5. 输出分析文件全部生成；
6. repeat 结束后没有残留 `class.pth`。

## 5. 三 seed 方向筛查

```bash
GPUID=0 \
SEEDS="0 1 2" \
REPEAT=3 \
OUTDIR=outputs/cifar-100/10-task/within-expert-interference-3seed \
bash experiments/cifar-100_expert_interference_mechanism.sh
```

固定设置：

```text
MAX_TASK=10
CRCT_EPOCHS=50
AUDIT_MECHANISM_MAX_SAMPLES=256
prompt_param=50 5 1e-5 1e-5 0.4
```

256 必须保持为偶数。32 只允许 smoke；128 可做临时方向检查，但不能和 256-sample 正式结果合并。

## 6. 五 seed 正式候选

```bash
GPUID=0 \
SEEDS="0 1 2 3 4" \
REPEAT=5 \
OUTDIR=outputs/cifar-100/10-task/within-expert-interference-5seed \
bash experiments/cifar-100_expert_interference_mechanism.sh
```

正式比较必须固定 commit、配置、任务顺序、seed、probe 数量、GPU/软件环境。实验内的 expert 与 task pair 是嵌套观测，不是独立重复；最终推断以 seed 为重复单位。

## 7. 验证探针不改变训练轨迹

运行匹配 no-probe control：

```bash
GPUID=0 \
SEEDS="0 1 2" \
REPEAT=3 \
OUTDIR=outputs/cifar-100/10-task/within-expert-interference-control-3seed \
bash experiments/cifar-100_expert_interference_control.sh
```

严格检查：

```bash
python utils/compare_expert_interference_control.py \
  outputs/cifar-100/10-task/within-expert-interference-control-3seed \
  outputs/cifar-100/10-task/within-expert-interference-3seed
```

默认要求每个 repeat 的 FAA、CAA、FR 最大绝对差不超过 `1e-5`。若失败，必须先判断 instrumentation 是否改变训练轨迹，不能继续解释机制结果。

## 8. 输出文件

```text
expert_interference/
  probe_manifest.jsonl
  expert_usage.jsonl.gz
  expert_task_pair_conflict.jsonl.gz
  expert_functional_drift.jsonl.gz
  task_forgetting.jsonl
  analysis/
    expert_usage_layer_head.csv
    layer_head_usage_summary.csv
    overall_usage_summary.csv
    expert_task_pair_long.csv
    task_pair_summary.csv
    layer_head_primary.csv
    seed_endpoint_summary.csv
    across_seed_summary.csv
    within_expert_interference_report.md
```

重新分析：

```bash
RUNDIR=outputs/cifar-100/10-task/within-expert-interference-5seed \
bash experiments/cifar-100_expert_interference_analyze.sh
```

## 9. 判定顺序

### 9.1 数据完整性

1. 每个 seed 应有 10 个 probe manifest；
2. 每个 task 的分层 class count 差不超过 1；
3. 每个 `(seed,task,layer,head)` 满足 `sum(usage_count)=samples×topk`；
4. 十任务应有 45 个 `(old_task,new_task)` 组合；
5. 每个 task pair 同时存在 conflict、actual update、incremental drift 和 forgetting 字段；
6. 无效梯度 pair 保留为 NaN/invalid，不得替换成 0。

### 9.2 路由复用必须先按 layer/head 判断

读取：

```text
expert_usage_layer_head.csv
layer_head_usage_summary.csv
overall_usage_summary.csv
```

其中：

- `within_layer_head_selection_share` 是 expert 频率的主定义；
- `routing_scope=final_checkpoint` 可与现有 layer/head Excel 直接核对；
- `routing_scope=learning_boundary` 用于判断锁定是否从各任务学习边界起已经形成；
- `topk_selection_share` 越接近 1，说明该池的选择质量越集中在固定 Top-k；
- `exact_same_topk_all_tasks=true` 表示该 seed 下所有任务的 Top-k 集合完全相同；
- `mean_consecutive_topk_jaccard` 衡量相邻任务 Top-k 集合稳定性；
- `overall_usage_summary.csv` 只汇总池级结果，不能把它解释成单个全局 expert 池。

如果整体坐标频率看似均匀，但大量 layer/head 的 `topk_selection_share≈1` 且 Top-k Jaccard≈1，应报告“池内严重锁定、跨池汇总掩盖锁定”，不能报告“专家使用均匀”。

### 9.3 第一主端点：内部跨任务冲突是否超过噪声

读取 `seed_endpoint_summary.csv`：

- `primary_median_excess_conflict > 0`：跨任务冲突高于同任务 split-half 噪声；
- `primary_positive_task_pair_fraction`：45 个 task pair 中方向为正的比例；
- 同时检查 `layer_head_primary.csv`，排除结果只来自少数 layer/head。

三 seed 只用于决定是否扩展。若多个 seed 的 excess conflict 非正或 layer/head 方向大面积反转，应停止并重新检查假设。

五 seed 后同时读取 `across_seed_summary.csv`，报告有效 seed 数、median、IQR 和正方向 seed 比例。五 seed 是最小正式候选，不足以依赖渐近检验；若论文要求保守的跨 seed 显著性结论，应预注册后扩展到至少 10 个独立 seed，而不是把 expert/task-pair 伪装成独立样本来压低 p 值。

### 9.4 第二端点：冲突是否对应真实有害更新

检查：

- `sum_shared_expert_first_order_old_loss_change`；
- `observed_full_model_old_loss_change`；
- `harm_to_observed_loss_spearman`。

如果跨任务负余弦存在，但真实 expert 更新的一阶旧损失变化不为正，说明“冲突存在”但未形成有害更新，不能称为遗忘机制。

### 9.5 第三端点：有害更新是否对应单步功能漂移

主字段：

```text
incremental_response_cosine_distance
harm_to_incremental_drift_spearman
conflict_to_incremental_drift_spearman
```

累计漂移只能作为补充。若只累计漂移相关、incremental drift 不相关，不能把当前 task-pair 冲突解释为漂移原因。

### 9.6 第四端点：漂移是否对应遗忘

在 task-pair 层面比较：

```text
mean_incremental_response_cosine_distance
learning_boundary_forgetting
incremental_drift_to_forgetting_spearman
```

遗忘在同一个 task pair 内对所有 expert 相同，因此不能在单个 task pair 内把重复的 forgetting 值当成 360/1800 个独立样本；必须先聚合到 task-pair，再以 seed 做最终重复。

## 10. 可以写到什么程度

仅第一端点成立：

> Different tasks produce excess gradient conflict within persistently shared SMoPE experts.

第一、二、三端点成立：

> Cross-task conflict is aligned with harmful expert updates and incremental task-conditioned response drift.

四条链均跨 seed 稳定：

> The evidence supports persistent expert sharing as a mechanism associated with forgetting.

即使全部成立，本轮仍没有完成 expert-specific counterfactual intervention。要写强因果结论，还需要下一阶段的“只投影高冲突 expert”与 layer/head、梯度范数匹配的随机 expert 对照。

## 11. 明确禁止的分析

- 不再以 hard usage–conflict Spearman 作为第一主端点；
- 不把不同 layer/head 中相同编号的 expert 合并为一个 expert；
- 不用整体频率分布否定 layer/head 池内的 Top-k 锁定；
- 不用 soft usage 偷换已退化的 hard usage 假设；
- 不把所有旧任务先平均后再声称 task-specific 机制；
- 不把累计漂移直接归因给最后一个新任务；
- 不把 expert-level p-value 当作论文最终显著性；
- 不把代码测试通过写成 CIFAR-100 机制已经成立。
