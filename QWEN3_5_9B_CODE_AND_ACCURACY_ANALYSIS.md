# Qwen3.5-9B + SMoPE 代码与精度下降分析

## 结论先行

当前 Qwen3.5-9B 版本在 CIFAR-100、10-task、seed 0 上的最终平均准确率（FAA）是 **79.23%**。论文/ViT 版本报告的 CIFAR-100 FAA 是 **89.23 ± 0.12%（5 seeds）**，因此这里是**下降 10.00 个百分点**，而不是“相对下降 10%”；换算成相对降幅是 **11.21%**。

最重要的判断是：**主要问题不是 Qwen 学不会当前任务，而是旧任务遗忘和路由退化。**

- 每个任务刚学完时的准确率（准确率矩阵对角线）平均为 **89.76%**，最终 FAA 为 **79.23%**，两者相差 **10.53 个百分点**。
- 最终 forgetting 为 **13.32 个百分点**，BWT 为 **-11.70 个百分点**；早期任务损失明显，例如 task 1 从 98.7% 降至 62.4%。
- 25 个专家中，每层每头始终只使用固定 5 个：覆盖率恒为 **5/25 = 20%**，熵恒为 **ln(5) = 1.6094**。task 1 的每个被选专家频次为 5000，task 10 时变成 50000，说明所有样本都走同一组 Top-5，动态路由事实上已经退化。

因此，现阶段不能简单得出“Qwen3.5-9B 的视觉能力比 ViT 差 10%”。更准确的说法是：**这套 Qwen 迁移实现未复现 ViT-SMoPE 的动态专家分化与抗遗忘效果，而且两版训练和评测协议并不完全等价。**

---

## 1. 实验结果与比较口径

### 1.1 Qwen 实际结果

数据来自：

- `output/qwen-global128-b16/cifar100/smope/full/seed-0/summary.json`
- `output/qwen-global128-b16/cifar100/smope/full/seed-0/accuracy_matrix.csv`
- `output/qwen-global128-b16/cifar100/smope/full/seed-0/events.jsonl`

| 指标 | Qwen3.5-9B | ViT-SMoPE（论文） | 差值 |
|---|---:|---:|---:|
| FAA / final average accuracy | 79.230 | 89.23 ± 0.12 | **-10.000 pp** |
| CAA / average incremental accuracy | 84.817 | 93.67 ± 0.82 | **-8.853 pp** |
| Forgetting | 13.322 | 当前仓库未保存对应单次 ViT 日志 | 不可直接比较 |
| BWT | -11.700 | 当前仓库未保存对应单次 ViT 日志 | 不可直接比较 |

ViT 数字来自 SMoPE 论文 Table 1：[arXiv 2509.24483](https://arxiv.org/abs/2509.24483)。论文数值是 5 次运行的均值，而当前 Qwen 只有 seed 0，一对多比较不是严格的配对实验。不过 ViT 的标准差只有 0.12，随机种子不太可能单独解释 10 个百分点。

> 注意：结果文件实际位于 `output/`，但 `environment.json` 内记录的运行参数是 `outputs/qwen-global128-b16/...`。这表明结果可能被复制或移动过；不影响文件内指标计算，但会削弱“目录即原始运行位置”的可追溯性。

### 1.2 10 个任务的准确率轨迹

| 训练完成至 task | 平均准确率 |
|---:|---:|
| 1 | 98.700 |
| 2 | 92.650 |
| 3 | 90.433 |
| 4 | 85.225 |
| 5 | 83.400 |
| 6 | 80.367 |
| 7 | 80.486 |
| 8 | 78.013 |
| 9 | 79.667 |
| 10 | 79.230 |

各任务刚训练完的对角准确率为：

```text
98.7, 90.0, 90.4, 87.5, 89.1, 86.3, 93.1, 90.1, 81.8, 90.6
平均 = 89.76
```

最终各任务准确率为：

```text
62.4, 56.2, 72.4, 84.6, 87.2, 83.4, 89.7, 89.4, 76.4, 90.6
平均 = 79.23
```

这说明 Qwen 对新任务的即时拟合能力并不弱；主要损失发生在后续任务训练之后。

---

## 2. 当前 Qwen3.5-9B 代码是怎样组织的

### 2.1 总体流程

```text
CIFAR-100 PIL 图片
  -> Qwen Processor：图片 + 固定文本 "Classify this image."
  -> 冻结的 Qwen3.5-9B 多模态主干（BF16）
  -> 在 8 个 language full-attention 层插入每头 25 个 K/V prompt experts
  -> 每层每头根据整段输入的平均 Q 选择 Top-5
  -> 取最后一个有效 token 的 hidden state
  -> 4096 -> 100 的线性分类头
  -> 当前任务交叉熵 + router loss + old-prompt loss
  -> 每任务结束后扫描专家频次、保存特征均值/对角方差
  -> 用合成特征校正分类头
  -> 在所有已见类别上评测
```

主干完全冻结，只训练分类头与 K/V prompt。日志记录：

- 总参数：8,394,743,124
- 可训练参数：2,048,100
- 注入的 full-attention 层：`[3, 7, 11, 15, 19, 23, 27, 31]`
- 两张 A100 40GB，BF16，Transformers 5.3.0
- batch size 16/GPU，梯度累积 4，world size 2，有效 batch 128

### 2.2 输入处理核心代码

位置：`qwen_smope/data.py:88-111`

```python
text = self.processor.apply_chat_template(
    [{"role": "user", "content": [{"type": "image"},
     {"type": "text", "text": "Classify this image."}]}],
    tokenize=False, add_generation_prompt=True, enable_thinking=False)

inputs = self.processor(
    text=[text] * len(images),
    images=list(images),
    padding=True,
    return_tensors="pt",
    images_kwargs={"size": {
        "shortest_edge": self.min_pixels,
        "longest_edge": self.max_pixels,
    }},
)
```

要点：输入是完整图文序列，而不是只喂视觉 token；Qwen 版也没有复用 ViT 的 `RandomResizedCrop(224)` 和 `RandomHorizontalFlip()` 数据增强。

### 2.3 专家选择核心代码

位置：`qwen_smope/model.py:37-51`

```python
weights = valid[:, None, :, None].to(q.dtype)
mean_q = (q * weights).sum(2, keepdim=True) / weights.sum(2, keepdim=True).clamp_min(1)
scores = mean_q.float() @ self.pk.float().transpose(-1, -2)

span = scores.detach().amax(-1, keepdim=True) - scores.detach().amin(-1, keepdim=True)
labels = (
    scores - span * self.used.float()[None, :, None, :] * self.epsilon
    if training else scores
)
indices = labels.topk(self.topk, dim=-1).indices.squeeze(2)
```

含义：

1. 对所有有效 token 的 Q 做平均，得到每个样本、每个注意力头的统一路由查询 `mean_q`。
2. `mean_q` 与 25 个 prefix key 点积打分。
3. 训练时惩罚过去高频专家，鼓励使用新专家。
4. 推理/扫描时不施加频次惩罚，只按学习到的分数选择专家。
5. 每个头选择 5 个专家。

### 2.4 把 prompt K/V 接入 Qwen 注意力

位置：`qwen_smope/model.py:77-107`

```python
ordinary = (q.float() @ k.float().transpose(-1, -2)) * self.scaling
ordinary = ordinary.masked_fill(~self.valid_tokens[:, None, None, :], -torch.inf)

prompt_scores, pv = self.expert.select(
    q, self.valid_tokens, self.dense, self.training
)
scores = torch.cat(
    ((prompt_scores * self.scaling).expand(-1, -1, n, -1), ordinary),
    -1,
)
weights = scores.softmax(-1).to(v.dtype)
values = torch.cat((pv.to(v.dtype), v), 2)
out = (weights @ values).transpose(1, 2).reshape(b, n, -1)
```

这是 prefix tuning：prompt expert 提供额外的 K/V，和原序列 K/V 一起参与 softmax。它不是修改 Qwen FFN 的参数 MoE，也不支持生成缓存；这是一个分类专用实现。

### 2.5 冻结主干、抽特征、分类

位置：`qwen_smope/model.py:110-149`

```python
self.backbone = backbone.requires_grad_(False)
self.classifier = nn.Linear(
    backbone.config.text_config.hidden_size, classes
)

outputs = self.backbone(**inputs, use_cache=False, return_dict=True)
last = positions.masked_fill(~valid, -1).max(-1).values
features = outputs.last_hidden_state[
    torch.arange(valid.shape[0], device=valid.device), last
]
logits = self.classifier(features.float())
```

和 ViT 使用 `[CLS]` 视觉 token 不同，Qwen 版取聊天模板中最后一个有效 token 作为整幅图的分类表示。这个 token 能通过因果注意力看到前面的图片与文本，但它不是专门为 CIFAR 分类训练的视觉 CLS token。

### 2.6 两阶段训练与损失

位置：`qwen_smope/train.py:141-186, 387-420`

```python
# 只在当前任务的类别区间计算 CE
logits, router, old, _ = network(inputs, dense=dense)
ce = F.cross_entropy(logits[:, start:end], labels - start)
loss = ce + router + old

# task 1: 10 epochs dense + 20 epochs sparse
# later tasks: 20 epochs sparse
phases = [
    ("dense", args.dense_epochs if task == 0 else 0),
    ("train", args.epochs),
]
```

第一任务先让全部 25 个专家参与 10 轮 dense 初始化，再做 20 轮 Top-5 稀疏训练；后续任务只做稀疏训练。交叉熵仅覆盖当前任务的 10 类，这和原 ViT 版屏蔽旧类 logits 的思路一致。

### 2.7 统计回放与分类器校正

位置：`qwen_smope/train.py:189-261`

```python
# 为每类保存均值与逐维方差
mean = sums[c] / counts[c]
variance = (
    (squares[c] - sums[c].square() / counts[c])
    / (counts[c] - 1).clamp_min(1)
).clamp_min(1e-4)
prototypes[c] = (mean.float().cpu(), variance.float().cpu())

# 从对角高斯采样伪特征，只更新分类头
features = means[y] + torch.randn(
    len(y), means.shape[1], device=device
) * stds[y]
logits = model.classifier(features)
loss = F.cross_entropy(logits[:, :end], y)
```

这是无图像回放：旧类只保留特征均值和逐维方差，再采样伪特征校正分类头。

---

## 3. 为什么比 ViT 低 10 个百分点

### 原因一：旧任务遗忘是直接主因（高置信度）

最直接的证据不是模型大小，而是准确率矩阵：

- 当前任务刚学完的对角平均：89.76%
- 最终所有任务平均：79.23%
- 差值：10.53 个百分点
- BWT：-11.70 个百分点

换言之，Qwen 版能把每个新任务学到约 90%，但在继续训练后无法维持早期任务。task 1 和 task 2 最终分别只剩 62.4% 和 56.2%。这与“表示能力不足”不同，是典型的持续学习稳定性问题。

### 原因二：动态路由已经塌缩为固定 Top-5（高置信度）

`experts_task_1.json` 到 `experts_task_10.json` 的统计完全一致：

| 统计 | 实际值 |
|---|---:|
| 每头专家总数 | 25 |
| 每头有非零频次的专家数 | 5 |
| coverage | 0.20 |
| entropy | 1.6094379 = ln(5) |
| task 1 被选专家单个频次 | 5000 |
| task 10 被选专家单个累计频次 | 50000 |

例如第一层第一头在 task 1 的频次是：

```text
5000, 0, 0, ..., 5000, 5000, 5000, ..., 5000, 0
```

每个非零值都恰好等于样本数，证明 5000 张不同图片选择的是同一组 5 个专家。SMoPE 依赖“不同输入激活不同专家”来减少干扰；现在这个机制没有发生。

可能的代码机制是：`mean_q` 对**整个图文序列**求平均，固定聊天模板和文本 token 会稀释图像间差异。原实现一旦在首任务形成固定 Top-5，推理/频次扫描还会对 `frequency == 0` 的专家施加 `2 × score span` 惩罚，使未记录专家很难重新进入 Top-5，形成自锁。当前代码已取消这项推理硬惩罚，但必须重新训练并检查 coverage 才能判断路由塌缩是否解除。

“路由塌缩”是日志事实；“固定文本稀释图像差异”是根据实现作出的高概率解释，仍需要用 image-token-only 路由消融来确认。

### 原因三：分类器校正强度远低于 ViT 版（高置信度）

两版校正预算并不匹配：

| 设置 | ViT 原脚本 | 当前 Qwen |
|---|---:|---:|
| correction epochs | 50 | 5 |
| 每类每轮伪特征数 | 128 | 16 |
| 每类总伪特征预算 | 6400 | 80 |
| 比例 | 80× | 1× |

此外，Qwen 特征维度是 4096，ViT 是 768。用每类 500 个真实训练样本估计 4096 维的独立方差，再只用很少的伪样本校正线性头，更容易出现统计估计偏差。校正训练损失接近 0 只能说明拟合了合成分布，不代表真实测试分布校准正确。

### 原因四：迁移并非同协议替换主干（高置信度，属于混杂变量）

| 项目 | ViT-SMoPE | Qwen3.5-9B-SMoPE |
|---|---|---|
| 输入 | 纯图像 token | 图片 + 固定聊天文本 |
| 分类特征 | 专用 `[CLS]` token | 最后一个文本 token |
| prompt 注入 | 前 6 个 ViT dense attention block | 8 个 language full-attention 层 |
| 未注入部分 | 后 6 个 ViT block | 视觉塔及其余线性注意力层 |
| 训练增强 | RandomResizedCrop + HorizontalFlip | Qwen processor 的确定性缩放 |
| prompt LR | 0.025 | 0.001 |
| head LR | 0.005 | 0.001 |
| 结果 seeds | 5 | 1 |

特别是 Qwen 只在 8 个 language full-attention 层插入 prompt，没有改视觉编码器，也没有覆盖其余线性注意力层。因此它不是 ViT SMoPE 的逐层等价迁移。

### 原因五：缺少数据增强会放大过拟合（中等置信度）

训练日志中后期各任务训练准确率几乎都是 100%，但测试对角准确率多在 82%～93%。ViT 使用随机裁剪和水平翻转，Qwen 版使用确定性图像处理。对只有 32×32 的 CIFAR-100，放大后不做增强会降低泛化。不过它更可能解释一部分单任务泛化差距，不能单独解释 11.70 的负 BWT。

### 暂无证据支持的说法

- “9B 模型太小”：9B 并不小，而且当前任务拟合很好。
- “有效 batch 不一致”：两版有效 batch 都是 128。
- “BF16 数值错误”：训练没有非有限 loss/gradient，且代码将注意力打分与 softmax 提升到 FP32。
- “10 个百分点全是随机种子”：论文 5 seeds 的 FAA 标准差是 0.12，不支持这一解释。

---

## 4. 建议的最小验证顺序

### 4.1 先验证分类器校正预算

这是成本最低、最容易形成因果证据的实验。保持训练不变，把 Qwen 校正预算先对齐到 ViT：

```bash
OUTPUT_ROOT=outputs/qwen-correction-match \
bash scripts/qwen/cifar100.sh full smope \
  --batch-size 16 --accumulation 4 \
  --correction-epochs 50 --prototype-samples 128 \
  --correction-lr 1e-4
```

用途：检验 80 倍校正预算差异能解释多少 FAA/BWT。注意该命令必须使用新输出目录，避免覆盖已有结果。

### 4.2 增加路由诊断并修复路由塌缩

优先记录每批次 `last_indices` 的样本间唯一组合数量，以及 image token 与 text token 分别计算的 `mean_q` 方差。预期正常路由的 coverage 应明显高于 0.20，且不同样本不应全部选择同一组专家。

建议消融：

1. 只用视觉 token 计算路由查询，不平均固定聊天模板 token。
2. 用最后 token 的 Q 或视觉 token attention-pooling 替代全序列均值。
3. 对比取消 `frequency == 0` 硬惩罚前后的结果，判断是否存在“首任务 Top-5 自锁”。
4. 比较修复前后的 coverage、路由组合数、FAA 和 BWT，而不是只看训练 loss。

### 4.3 跑 head-only 对照

```bash
OUTPUT_ROOT=outputs/qwen-head-only-b16 \
bash scripts/qwen/cifar100.sh full head_only \
  --batch-size 16 --accumulation 4
```

用途：如果 head-only 与当前 SMoPE 的 FAA 接近，说明 prompt 路由没有提供有效增益；如果 head-only 更好，则当前 prompt 还在制造干扰。

### 4.4 最后再做严格的 5-seed 配对

```bash
SEEDS="0 1 2 3 4" OUTPUT_ROOT=outputs/qwen-matched-5seed \
bash scripts/qwen/cifar100.sh full smope \
  --batch-size 16 --accumulation 4
```

用途：在协议确定后报告均值和标准差。不要先花 5 倍算力重复一个已知发生路由塌缩的配置。

---

## 5. 一句话概要

**Qwen 版低 10 个百分点的直接原因是抗遗忘失败，而最明显的实现异常是 25 选 5 的动态路由在所有图片上退化成同一组固定专家；同时分类器统计回放预算仅为 ViT 的约 1/80，并伴随输入、注入层、特征 token、增强和学习率等协议差异，所以不能把降幅简单归因于 Qwen 主干本身。**
