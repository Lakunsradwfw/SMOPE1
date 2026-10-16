# Qwen3.5-9B + CoIN 实验说明

smoke 性能、超长样本过滤、DDP 诊断的唯一当前结论与服务器执行顺序见
[`SMOKE_OPTIMIZATION_HANDOFF.md`](SMOKE_OPTIMIZATION_HANDOFF.md)。P0/P1/P3 已通过，Gloo
修复也已验证；P2 当前只被 OCRVQA 镜像缺少7个官方图像阻断。服务器只执行交接文档
**§12.10 的同源图像恢复与 P2 resume**，通过前不启动正式 seed-0。

本入口在 CoIN 八任务上运行三种可比较配置，全部从 post-trained 多模态
`Qwen/Qwen3.5-9B` 开始，不再支持旧的 Base+CIFAR/CUB/ImageNet-R 分类实验。

| method | 训练 | 保存内容 |
|---|---|---|
| `zero_shot` | 不训练，直接生成 | 指标、逐样本预测、环境信息 |
| `lora` | 顺序 LoRA | 最新轻量 adapter、指标、预测 |
| `smope` | 顺序生成式 SMoPE | 最新轻量 adapter、指标、预测、路由统计 |

基础模型不会复制到输出目录，也不保存逐任务完整 checkpoint。连续训练只覆盖写入一个
`latest_adapter.pt`，其中包含轻量参数、任务进度和准确率矩阵；任务中断会从当前任务重做。

## 环境与数据

目标服务器为 Linux、Python 3.11、2×H100 80GB；当前普通训练默认只使用一张卡：

```bash
conda create -n smope-coin python=3.11 -y
conda activate smope-coin
bash scripts/qwen/setup.sh
python -c "import peft; assert peft.__version__ == '0.18.1'; print(peft.__version__)"
```

本地模型默认位于 `pretrained/Qwen3.5-9B/`，必须是
`Qwen3_5ForConditionalGeneration` 的完整 safetensors 目录，并包含 post-trained 模型的
`chat_template.jinja`；目录名含 `Base` 会被直接拒绝。CoIN 根目录结构见
`SMOPEDATAS_CoIN_README.md`，每个任务必须同时存在 `train.json` 和 `test.json`；若评估文件名不同，
使用 `--eval-split` 指定。

先检查全部数据和 Processor，不加载 9B 权重：

```bash
python scripts/qwen/check_coin.py \
  --model-path /absolute/Qwen3.5-9B \
  --coin-root /absolute/CoIN
```

训练前还必须用真实 processor 和图片扫描完整训练 split，生成所有方法、seed 共用的确定性
过滤清单；过滤发生在 `--max-train-samples` 之前：

```bash
python -m qwen_smope.prepare_data \
  --model-path /absolute/Qwen3.5-9B \
  --coin-root /absolute/CoIN \
  --image-transport http \
  --filter-cache outputs/coin-qwen3.5-9b/_filter_cache \
  --scan-workers 8
```

逐任务剔除率达到 10% 时，full 会拒绝启动；只有审阅 manifest 后才能显式使用
`--allow-high-filter-rate`。评测 split 不做该过滤。HTTP 扫描并发硬限制为 14；分片保存于
`_filter_cache/.parts/` 并可断点复用。永久 404/410 记为 `image_unavailable`，瞬时超时和 5xx
只重试、最终失败则中止扫描，不能悄悄算成数据剔除。

## 运行

```bash
export MODEL_PATH=/absolute/Qwen3.5-9B
export COIN_ROOT=/absolute/CoIN

# 每种方法只跑 1 条样本，验证真实 9B 前向/反向与 SMoPE 注入
bash scripts/qwen/run.sh zero_shot preflight
bash scripts/qwen/run.sh lora preflight
bash scripts/qwen/run.sh smope preflight

# 每任务少量样本的端到端验证；分数不能作为正式结果
bash scripts/qwen/run_all.sh smoke

# 三组正式实验
bash scripts/qwen/run.sh zero_shot full
bash scripts/qwen/run.sh lora full
bash scripts/qwen/run.sh smope full

# 相同目录从最近一个已完成任务恢复
bash scripts/qwen/run.sh smope full --resume

# 多种子；zero-shot 通常只需 seed 0
SEEDS="0 1 2" bash scripts/qwen/run.sh lora full
SEEDS="0 1 2" bash scripts/qwen/run.sh smope full
```

`scripts/qwen/run.sh` 默认 `NPROC_PER_NODE=1`；这个安全默认不变。P0/P1/P3 已授权交接文档
§12.4 的正式命令显式设置 `NPROC_PER_NODE=2`，但必须先完成 §12.10 的数据恢复与 P2 resume 门禁。

默认任务顺序为论文随机顺序：

```text
ScienceQA, TextVQA, ImageNet, GQA, VizWiz, Grounding, VQAv2, OCRVQA
```

可用 `--tasks` 显式修改。正式对比必须让 LoRA 与 SMoPE 使用相同任务顺序、样本、图像预算、
最大序列长度和生成长度。

主要训练设置：BF16、每卡 batch 4、有效全局 batch 16、每任务 1 epoch、最多 1024 token、
最多 256 个合并视觉 token。SMoPE 默认 25 专家、Top-5，只注入 8 个 full-attention 层；
路由由最后 prefix token 与 prefix 均值的可学习混合决定，答案生成期间固定。

未显式指定 `--accumulation` 时，单卡自动取 4、双卡自动取 2；不整除或显式值与
`--global-batch-size` 冲突会拒绝启动。`--workers` 默认 preflight/smoke=0、full=4；继续使用
HTTP 图片通道和 `--length-bucket=128`。SMoPE 路由统计默认使用
`--routing-scan-batch-size 4`；它只做前向，不沿用生成评测的 batch 限制。评测支持显式
`--eval-batch-size 4`，通过左填充进行批量生成，但正式采用前必须按交接文档 §9.6 与 batch=1
核对 sample ID、解析答案和得分。`--routing-scan-samples N --routing-scan-seed 42` 使用固定随机
子集而非前缀，并把采样摘要写入 routing JSON。

不要在未指定评测预算时直接启动 full：`--max-eval-samples 0` 是全量评测，一个训练方法的
持续学习矩阵要生成 800,935 条答案。正式对比必须先固定同一个非零 N，并在所有方法、seed 和
轮次使用相同的 `--max-eval-samples N --eval-sample-seed 42`。当前建议先按交接文档 §9.5 做
100 条/任务的短标定，再冻结 N；不要用短标定分数作为正式结果。

增大 global batch 不是主要提速手段：保持 `--global-batch-size 16`，依次测试
`--batch-size 8`（自动 accumulation=2）和 `--batch-size 16`（自动 accumulation=1）。只有 H100
峰值显存、非有限值、样本计数和吞吐判据全部通过后，才冻结更大的物理 batch。可用
`--no-gradient-checkpointing` 做独立对照，但它不是默认值，也不能与 batch 调整同时首测。

## 2026-10-16：服务器唯一下一步

不要从下面的历史标定章节继续挑实验。先拉取 `v2` 最新提交，然后严格执行
[`SMOKE_OPTIMIZATION_HANDOFF.md` §12.10](SMOKE_OPTIMIZATION_HANDOFF.md)：从 OCR-VQA 官方发布
或其 `dataset.json.imageURL` 的同源文件恢复全部7张图，生成 SHA256 审计 manifest，确认 HTTP
和正式 N=1000 子集均为0缺图，再用原 P2 命令追加 `--resume`。禁止用相似封面、重新编码图或
运行时跳过样本。P2 四项判据全过后直接启动 §12.4 的 SMoPE exact-full-route seed-0。LoRA
正式配置仍为双卡、batch4、global batch16、默认 gradient checkpointing。

## 2026-10-15：24 小时目标的历史执行入口

本轮代码实现提交是 `83ef103`；服务器应拉取 GitHub `v2` 的最新 HEAD（其后还有回填 SHA 的文档
提交），并用 `git log -2 --oneline` 确认历史中包含 `83ef103` 后再开始。

服务器 DeepSeek 必须从 [`SMOKE_OPTIMIZATION_HANDOFF.md` §9.7](SMOKE_OPTIMIZATION_HANDOFF.md)
开始，按 **T4 → T8 →（通过后 T16）→ E1 → E4 → route scan → 可选关闭 checkpointing**
逐项执行。每次只改一个变量、GPU0 独占、回传原始 events/predictions/routing 与显存快照后再决定
下一步。不要重跑旧 A–D，不要直接启动 full，不要自行把预计 16–24h 写成实测结论。

特别注意三类变量：

- 增大物理 batch、保持 global batch=16 是吞吐候选；增大 global batch 不是替代方案。
- eval batch 只有在 batch=1/4 的解析答案和 score 完全一致后才能冻结。
- route scan=5000 会改变 SMoPE 的频率估计口径，必须通过 1000/5000/10000 保护集合稳定性判据，
  并在论文中披露 N 与 seed；它不是可以静默采用的工程优化。

普通功能 smoke 若只想确认代码路径，可临时缩短序列：

```bash
bash scripts/qwen/run.sh smope smoke --max-length 512 --max-visual-tokens 128
```

但这条命令**不得**用于 §9.7 的吞吐对照或正式实验。标定 OOM 时只降低物理
`--batch-size`，保持 `max-length=1024`、`max-visual-tokens=256`、global batch 与样本不变；否则
速度结果不具可比性。

### 历史 DDP 门禁（已完成；不要重跑）

P0 E/F 与 P1-DDP 已通过，结果和原始路径见交接文档 §11。当时的下一步是交接文档 §12.2
八任务各8000条 ETA 标定；该轮现已执行并在最后一个评测 cell 暴露缺图，后续改由 §12.10 接管。
下面命令仅保留作历史审计，不再是当前执行单。

旧 DDP 汇总的 `D=1.24×T1` 是单 rank 墙钟，不是有用吞吐。A 单卡处理100条/58.49s，D 双卡
处理200条/72.76s，未经协议对齐的有用吞吐实际为 **1.608×**。先拉取包含
`qwen_smope.compare_ddp` 与 `input_wait_seconds`/`sequence_widths` 的新提交，按交接文档 §9.4
依次跑 E、F（global batch 均为16），然后执行：

```bash
python -m qwen_smope.compare_ddp \
  --single outputs/coin-qwen3.5-9b/verify-20261015b/ddp/E-single-gbs16 \
  --dual outputs/coin-qwen3.5-9b/verify-20261015b/ddp/F-ddp-gbs16 \
  --min-speedup 1.5 \
  --output outputs/coin-qwen3.5-9b/verify-20261015b/ddp/ef-comparison.json
```

只有 `protocol_comparable=true` 且 `passes_speedup_gate=true` 才进入生产 P1 单/双卡8000条对照。
先跑单卡：

```bash
CUDA_VISIBLE_DEVICES=0 \
NPROC_PER_NODE=1 \
OUTPUT_ROOT=outputs/coin-qwen3.5-9b/verify-20261015b/prod-single-gbs16 \
bash scripts/qwen/run.sh smope smoke \
  --tasks GQA --max-train-samples 8000 \
  --batch-size 4 --global-batch-size 16 \
  --no-gradient-checkpointing --workers 4 \
  --eval-batch-size 1 --max-eval-samples 4 \
  --routing-scan-samples 1 --routing-scan-batch-size 1
```

单卡完成并回传后，保持参数不变，仅切换 world size（CLI 会把 accumulation 从4自动改为2）：

```bash
CUDA_VISIBLE_DEVICES=0,1 \
NPROC_PER_NODE=2 \
OUTPUT_ROOT=outputs/coin-qwen3.5-9b/verify-20261015b/prod-ddp-gbs16 \
bash scripts/qwen/run.sh smope smoke \
  --tasks GQA --max-train-samples 8000 \
  --batch-size 4 --global-batch-size 16 \
  --no-gradient-checkpointing --workers 4 \
  --eval-batch-size 1 --max-eval-samples 4 \
  --routing-scan-samples 1 --routing-scan-batch-size 1
```

回传两个完整目录与 `train_sparse` 的 seconds，以及 epoch 事件中的
samples_with_ddp_padding、input_wait_seconds、sequence_widths、optimizer_updates，再附峰值显存和
checkpoint 加载结果。双卡至少快 **1.5×** 才恢复为单 run 正式
方案；现有1.608×若能保持，N=10000 每 seed 保守约22.7h、理想约20.0h。若未通过，先依据
rank 间宽度顺序决定是否做同序预热；不要自行开 static graph/DDP bucket 或升级依赖。

论文主表按交接文档 §10.7 使用 frozen zero-shot、sequential LoRA、SMoPE 三组。正式 SMoPE
主结果暂按 **full route** 设计；N=10000 只能作为 sampled 效率消融，因为固定每任务 N 会改变
跨任务累计频率的样本量权重。先通过 P1 测 workers=4 的输入重叠，再单独标定 exact full route
在 workers=4、batch=1 下的时间；不要沿用 workers=0 的15.3h估算直接放弃 full。

## 输出

默认输出为 `outputs/coin-qwen3.5-9b/<method>/<mode>/seed-<seed>/`：

- `environment.json`：参数、版本、GPU、模型配置和权重索引摘要。
- `dataset_manifest.json`：实际读取的数据源；训练任务还包含原始数、有效数、实际采样数、
  逐任务剔除率、确认状态，以及每条剔除样本的 task/id/prompt/answer/full token 长度与原因。
- `events.jsonl`：训练损失、学习率、耗时与峰值显存。
- `predictions/<stage>/<task>.jsonl`：原始答案、参考答案、解析结果和样本得分。
- `accuracy_matrix.csv`：训练到每个阶段后，所有已见任务的 CoIN task accuracy。
- `summary.json`：最终平均准确率、MAA、New.ACC、BWT 和 forgetting；zero-shot 只报告各任务和平均值。
- `routing/*.json`：SMoPE 每层/每头频次、覆盖率、熵、保护状态、路由混合系数和温度。
- `latest_adapter.pt`：唯一保留的轻量恢复文件，不含基础模型。

汇总多次运行：

```bash
python scripts/qwen/summarize.py outputs/coin-qwen3.5-9b
```

## 适配边界

- 语言损失只监督最后一个 assistant 答案，路由只能读取图像与用户 prefix，杜绝答案泄漏。
- LoRA 与 SMoPE 均冻结视觉塔、Qwen 主干及 LM Head。
- 取消原分类版线性头、类别高斯原型和分类器校正；这些机制不适用于开放生成。
- Grounding 使用坐标 IoU>0.5；ScienceQA 解析选项字母；ImageNet 采用官方类别字符串包含判断；
  GQA、VizWiz、VQAv2、OCRVQA 使用忽略大小写的精确答案匹配。
- TextVQA 会从 `cl_dataset/TextVQA/TextVQA_0.5.1_{split}.json` 读取 10 个标注答案并使用
  官方软准确率；当 `test` 指令实际对应官方 `val` 时会自动回退到 val 标注。若找不到多答案标注，
  会退化为单答案精确匹配，并在逐样本预测的 `references` 中清楚体现。
- evaluator 是对 CoIN 官方任务脚本的仓库内可审计实现。原始生成文本、全部 reference 和解析结果
  都会保存，因此可另跑官方 evaluator 复核而不覆盖本实验结果。
