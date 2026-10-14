# 诊断报告（二）：两个新问题 —— DDP 使训练慢 45 倍 / 答案截断会崩 run

日期 2026-09-28 · 与 `DIAGNOSIS.md`（autotune 根因）**相互独立**

> `DIAGNOSIS.md` 记录的是 **autotune 按序列宽度触发、每宽度首次 6~20 秒**，那条已确认且修法
> （`--length-bucket` 分桶）正确。本文件记录的是**在做干净对照时暴露出来的另外两个问题**，
> 它们**不是** autotune，也不由分桶解决。

---

## 问题 1：DDP 让训练慢 45 倍（**最严重，未解释**）

### 证据（全部为独占 GPU、无并发的干净测量）

同一份数据、同一批参数，**唯一差别是 `NPROC_PER_NODE`**：

| 配置 | 样本/批次 | `train_dense` 耗时 |
|---|---|---|
| **单进程**（`NPROC_PER_NODE=1`，不初始化 dist、不包 DDP） | 200 / 25 | **25.55 s** ✅ |
| **2 卡 DDP**（`NPROC_PER_NODE=2`，默认） | 200 / 25 | **> 1140 s**（跑满 19 分钟被 timeout 杀掉）❌ |

**45 倍以上。** 而且**非线性**：

| 配置 | 批次 | `train_dense` |
|---|---|---|
| 2 卡 DDP | 5 | 21.25 s |
| 2 卡 DDP | 25 | **> 1140 s** |

批次数 ×5，耗时 ×54。

### 复现

```bash
cd /mnt/zq/SMOPE
# 快（约 1 分钟）
NPROC_PER_NODE=1 bash scripts/qwen/run.sh smope smoke \
  --tasks VQAv2 --max-train-samples 200 --max-eval-samples 8 \
  --workers 0 --output /tmp/single

# 慢（>19 分钟被 timeout 杀掉，train_dense 从未完成）
bash scripts/qwen/run.sh smope smoke \
  --tasks VQAv2 --max-train-samples 200 --max-eval-samples 8 \
  --workers 0 --output /tmp/ddp
```

### 已排除

- **不是 autotune**：单进程同样要 autotune（25.55 s 里已含首次各宽度 6~20 s），但很快就过去了
- **不是 `--workers` 的 DataLoader 启动开销**：两次都用 `--workers 0`
- **不是数据/形状**：同一份数据、同一 `--length-bucket 128`
- **不是 dist 初始化本身**：`preflight` 阶段单进程 20.07 s vs 2 进程 20.30 s，几乎相同
  —— 差别出现在 **`train_epoch` 里模型被 `DDP(...)` 包住之后**
- **不是 NCCL 通信量**：SMoPE 可训练参数仅 **1,638,656**（1.6M），all-reduce 量极小

### 最可疑的方向（**未验证**，供实现方参考）

1. **两个 rank 各自 autotune 不同的宽度集**
   `DistributedSampler` 让两个 rank 看到**不同的样本**，于是宽度分布不同 →
   两个 rank 的 autotune 集合不同；而每个 optimizer step 结束时的 all-reduce 是**集合点**，
   快的一方必须等慢的一方 → 可能把两边的 autotune 串行化并放大
2. **DDP 在 forward 里的 autograd 图遍历**
   `DDP.forward` 每次会重建 autograd 钩子（`prepare_for_backward`）；
   本项目对 attention 做了 monkey-patch（`qwen_smope/model.py:expert_attention`），
   图结构可能与 DDP 的假设不友好
3. **`no_sync()` 与 accumulation=2 的交互**（`train_epoch` 里 `network.no_sync()`）

### 建议的下一步（都很便宜）

- 在 `train_epoch` 里给「forward / backward / optimizer.step」**分别计时**，定位是哪个环节在 DDP 下变慢
- 试 `torch.nn.parallel.DistributedDataParallel(..., gradient_as_bucket_view=True)`、
  或调 `bucket_cap_mb`
- 试**两个 rank 喂完全相同的样本**（关掉 DistributedSampler 的 shuffle/分片），
  看是否消失 —— 这能直接检验上面的方向 1
- 试 `world=2` 但**不包 DDP**（仅各自独立训练），看是否回到 ~25 s

---

## 问题 2：答案被截断会让整个 run 崩溃（**确定性、必然发生**）

### 证据

```bash
NPROC_PER_NODE=1 bash scripts/qwen/run.sh smope smoke \
  --tasks VQAv2 --max-train-samples 200 --max-eval-samples 8 \
  --workers 0 --output /tmp/single
# -> ValueError: Answer was truncated or empty for sample 524377
```

**同一个样本 `524377` 在两次独立运行中稳定复现**（`run.sh` 那次与我的逐 batch 探针那次），
所以是确定性问题，不是偶发。

抛出位置：`qwen_smope/data.py`，`CoINCollator.__call__`：

```python
if length >= int(full["attention_mask"][row].sum()):
    raise ValueError(f"Answer was truncated or empty for sample {batch[row].sample_id}")
```

### 为什么必然发生

`train.py` 的 `--max-length` 默认 **1024**，而 CoIN 指令集里存在**提示词本身就很长**的样本
（VQAv2/OCRVQA 尤甚，Multitask 的 train.json 里更长）。当
「提示词 token 数 + 答案 token 数」超过 1024 时，答案被截断 → 监督 token 为 0 →
`CoINCollator` 正确地拒绝它。

**但拒绝等于抛异常，而 `train_epoch` 不捕获它** → 整个任务、整条 run 直接终止。

规模：VQAv2 有 **82,783** 条训练样本，只要有一条超长就会中断。

### 影响

**这是正式 run 的阻塞点。** 它与我先前诊断的性能问题无关 —— 即使性能修好，run 仍会在
跑到某个超长样本时崩溃。

### 可选修法（供实现方权衡）

| 方案 | 说明 |
|---|---|
| 跳过该样本 | `CoINCollator` 里改为「丢弃并记录」而非抛异常（需要让 collator 能返回空 batch 或让 dataset 层过滤） |
| 在 dataset 层预过滤 | 构造数据集时预先计算 token 长度，剔除超长样本，并记录被剔除的数量（**会改变训练集大小，需在报告里说明**） |
| 提高 `--max-length` | 治标；且增加显存与计算 |
| 截断提示词而非答案 | 保留答案 token，从提示词头部截断（**会改变语义**，需谨慎） |

**注意**：无论选哪种，都应在 `dataset_manifest.json` / 日志里**记录被剔除的样本数与 id**，
否则无法在论文里说明训练集规模。

---

## 附：今晚的完整结论分层

| 层 | 结论 | 状态 |
|---|---|---|
| ① | autotune 按序列宽度触发，每宽度首次 6~20 s，不落盘 | **已确认**（`DIAGNOSIS.md`） |
| ② | `--length-bucket` 把宽度收敛到 **6 种**（实测 `[256,384,512,640,768,896]`） | **已确认** |
| ③ | 换图片**不**触发重新调优（6 张不同图，只有第 1 次慢） | **已确认** |
| ④ | **DDP 使训练慢 45 倍** | **已测量，未解释** ← 本文件问题 1 |
| ⑤ | **答案截断会崩 run** | **已确认，确定性复现** ← 本文件问题 2 |

**方法论提醒**：③④⑤ 都是在做「干净对照实验」时才暴露的 —— 单进程 vs 2 卡这一对照
是整个晚上最有价值的一次测量。先前的"每步 17 秒 / 25 步跑不完"的观测，
**混了 autotune 与 DDP 两个因素**，我一度只用 autotune 解释，是错的。
