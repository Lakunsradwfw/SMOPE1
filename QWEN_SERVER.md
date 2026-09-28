# Qwen3.5-9B + CoIN 实验说明

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

推荐 Linux、Python 3.11、两张 A100 40GB：

```bash
conda create -n smope-coin python=3.11 -y
conda activate smope-coin
bash scripts/qwen/setup.sh
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

默认任务顺序为论文随机顺序：

```text
ScienceQA, TextVQA, ImageNet, GQA, VizWiz, Grounding, VQAv2, OCRVQA
```

可用 `--tasks` 显式修改。正式对比必须让 LoRA 与 SMoPE 使用相同任务顺序、样本、图像预算、
最大序列长度和生成长度。

主要训练设置：BF16、每卡 batch 1、梯度累积 8、每任务 1 epoch、最多 1024 token、
最多 256 个合并视觉 token。SMoPE 默认 25 专家、Top-5，只注入 8 个 full-attention 层；
路由由最后 prefix token 与 prefix 均值的可学习混合决定，答案生成期间固定。

OOM 时依次降低：

```bash
bash scripts/qwen/run.sh smope smoke --max-length 512 --max-visual-tokens 128
```

## 输出

默认输出为 `outputs/coin-qwen3.5-9b/<method>/<mode>/seed-<seed>/`：

- `environment.json`：参数、版本、GPU、模型配置和权重索引摘要。
- `dataset_manifest.json`：实际读取的指令文件、大小、修改时间和样本数。
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
