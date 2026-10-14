# 诊断报告（一）：autotune 按序列宽度触发，每宽度首次 6~20 秒

日期 2026-09-28 · 状态：**本文件的结论已确认**，但它**只解释了"训练慢"的一部分**

---

## ⚠️ 范围更正（必读）

本文件确认的是 **autotune** 这一个成因。但当晚稍后做的**干净对照实验**（单进程 vs 2 卡 DDP）
暴露出**另外两个独立问题**，见 **`DIAGNOSIS_DDP.md`**：

| # | 问题 | 量级 |
|---|---|---|
| ① | **本文件**：autotune 按序列宽度触发，不落盘 | 每宽度首次 6~20 s |
| ④ | **DDP 使训练慢 45 倍** | 同样 200 样本/25 batch：**单进程 25.55 s vs 2 卡 >1140 s** |
| ⑤ | **答案截断会崩 run**（确定性，样本 524377） | 直接终止整个 run |

**所以"每步 >17 秒"是 ①+④ 叠加的结果，本文件的 autotune 单独解释不了它。**
④⑤ 详见 `DIAGNOSIS_DDP.md`。

⚠️ 另外：我自己先前测到的"25 步跑不完"里面，**混入了 autotune 与 DDP 两个因素**，
我当时只用 autotune 解释，是错的。以单进程/2 卡的对照为准。

---

## 0. 根因（已确认到机制层）

> **fla 的 Triton 核启用了 `@triton.autotune`。首次遇到某个形状时，它在进程内逐配置基准测试，
> 成本约 20 秒；Triton 3.1.0 的结果不落盘，所以每个新进程、每个新形状都要重付。
> 于是"每个 batch 序列长度都不同"的训练会每步重付一次 —— 这就是 >17 s/step。**

### 0.0 证据链

**① 现象：同进程内首次调用极慢，之后极快**（`probe_warmup.py`，VQAv2 单样本，宽度 384，batch=1）

| 调用次序 | 前向 (s) | 反向 (s) | 合计 |
|---|---|---|---|
| **#1** | **12.010** | **7.942** | **19.95** |
| #2 | 0.112 | 0.153 | 0.265 |
| #3 / #4 | 0.109 | 0.153 | ~0.262 |

**② profiler 直接点名**（`probe_cold_first_forward.py`，全新进程、`eval()`+`no_grad`，实测 15.03 s）

```
                                     Self CUDA     # of Calls
aten::fill_                           8.787 s       70,039     ★
aten::zero_                           8.787 s       69,731     ★
vectorized_elementwise_kernel<...>    8.787 s       69,707     ★
cuLaunchKernel                        2.655 s       87,284
cudaDeviceSynchronize                 3.759 s          305
```

一次前向里 **约 7 万次 `fill_`/`zero_`** —— 这是 `do_bench` 在每次试跑前清 L2 缓存的典型特征。

**③ 每个核被调用约 1000 次**（24 层的模型）：

```
chunk_scaled_dot_kkt_fwd_kernel        24,086 次
l2norm_fwd_kernel                      22,319 次
merge_16x16_to_64x64_inverse_kernel     9,910 次
chunk_gated_delta_rule_fwd_kernel_h     9,201 次
```

**④ autotune 的确切位置**（grep fla 0.3.1）：

| fla 文件 | `autotune` 出现 | 对应核 |
|---|---|---|
| `modules/l2norm.py` | ✅ | `l2norm_fwd_kernel` |
| `ops/gated_delta_rule/wy_fast.py` | ✅ ×2 | `chunk_scaled_dot_kkt_fwd_kernel` |
| `ops/common/chunk_o.py` | ✅ | — |
| `ops/common/chunk_h_parallel.py` | ✅ | `chunk_gated_delta_rule_fwd_kernel_h_blockdim64` |
| `ops/gated_delta_rule/chunk.py` | ❌ 0 | （主 kernel 本身不 autotune） |

**⑤ 不落盘**：`~/.triton/` 下**只有 `cache/`（730MB，20598 文件），没有 `autotune/` 目录**
→ autotune 结果只存在进程内存 → 每个新进程从头再来。
（先前我数 `cache/*.json` 条目不变，据此误判"没有编译"——那是**编译**缓存，autotune 结果根本不在那儿。）

**⑥ 与 autograd 无关**：`torch.no_grad()` 下同样触发（首次 11.99 s → 第二次 0.090 s），
所以不是"首次建 autograd 图"。

### 0.1 已证伪的假说

| 假说 | 证据 |
|---|---|
| A. 缺 Triton 快核 | 已装 fla+causal-conv1d，`is_fast_path_available=True`、回退警告 0 次，**现象不变** |
| B. Triton 每形状**编译**成本 | 编译缓存 5424 条目在两次 preflight 之间**未变**；真正原因是同目录下的 **autotune**，不是编译 |
| C. batch=1 使 fla 变慢 | 见 0.3，batch=1 时 fla 比 torch 回退快 10 倍 |
| D. dist/DDP 交互 | `NPROC_PER_NODE=1`（不初始化 dist）仍 **20.26 s**，与 2 进程的 20.31 s 相同 |
| E. collate / 梯度检查 / transfer | 逐段计时：collate 0.030 s、transfer 0.001 s、梯度检查 0.006 s |
| F. autograd 首次建图 | `no_grad` 下同样触发 |

### 0.2 对可行性的重新评估

- **评测不受影响** ✅：`generate` 用 `use_cache=True`，解码每步序列长度为 1、形状恒定，
  走 recurrent 路径。实测 `evaluate_ScienceQA` 4 条样本仅 **0.26 s**。
  （注意：**评测的 prefill 仍是新形状**，仍要付一次 autotune）
- **训练受影响**：每个新宽度一次 autotune。
- **分桶后**约 8 种宽度 → 每进程约 8 × 20 = **160 s**，之后每步 ~0.4 s
- 训练本身：71,750 步 × ~0.4 s ≈ **8 小时/方法** → **总量约 10~13 小时/方法，可行**

### 0.3 建议方向（供实现方参考）

1. **显式预热（最稳，推荐）**：训练前对**所有需要的宽度**跑一次 dummy 前向（+反向，
   两者独立计费：12 s / 8 s），把 autotune 一次性做完。配合 `--length-bucket`（已实现）
   使宽度有界 → 约 8 种 × 20 s = **~3 分钟**，之后整个 run 都是 0.4 s/步
2. **关掉 autotune**：`fla` 的 autotune 装饰器可用单一配置替换（monkey-patch
   `triton.autotune` 成固定 `num_warps`/`num_stages`，或直接改 fla 源码）。
   风险：可能牺牲一点稳态吞吐，但省掉全部 20 s
3. **让 autotune 结果落盘**：Triton 3.1.0 的 autotuner 源码里**提到** cache/disk，
   但实际没有生成 `~/.triton/autotune/`；可查是否需设置环境变量（如 `TRITON_CACHE_DIR`）
   或升级 Triton（风险：与 torch 2.5.1 的兼容性）
4. **不要据此优化 fla 的 delta-rule 核或 causal-conv1d** —— 见 0.4/0.5，
   它们在**稳态下都很快**，且新装的 conv 快核反而更慢

---

## 0.3 内核层面：稳态下 fla 全面胜出

直接对拍两个 delta-rule 核（不经模型，seq=384 heads=16，**均为稳态**）：

| batch | torch 回退 前向+反向 | fla Triton 前向+反向 |
|---|---|---|
| 1 | 0.0321 s | **0.0032 s** |
| 2 | 0.0308 s | **0.0041 s** |
| 4 | 0.0369 s | **0.0049 s** |
| 8 | 0.0374 s | **0.0046 s** |

fla 在所有 batch 下快约 10 倍（含 batch=1）。24 层合计约 **77 ms** → 与 20 s 无关。

## 0.4 depthwise causal conv：所谓"快核"其实更慢

| batch | `F.silu(nn.Conv1d(...))` | `causal_conv1d_fn`（新编译 1.7.0） |
|---|---|---|
| 1 | **0.045 ms** | 0.136 ms（慢 3 倍） |
| 4 | **0.054 ms** | 0.116 ms（慢 2 倍） |

量级太小，不是 20 s 的来源；但说明**装 causal-conv1d 无收益**。

## 0.5 已证伪的假说

| 假说 | 证据 |
|---|---|
| A. 缺 Triton 快核 | 已装 fla+causal-conv1d，`is_fast_path_available=True`、回退警告 0 次，**现象不变** |
| B. Triton 每形状编译/autotune | 两次连做同一 `preflight`：#A 20.17 s、#B 20.21 s，**且缓存条目 5424 不变** → 非编译 |
| C. batch=1 使 fla 变慢 | 见 0.3，batch=1 时 fla 快 10 倍 |
| D. dist/DDP 交互 | `NPROC_PER_NODE=1`（不初始化 dist）仍 **20.26 s**，与 2 进程的 20.31 s 相同 |
| E. collate / 梯度检查 / transfer | 逐段计时：collate 0.030 s、transfer 0.001 s、梯度检查 0.006 s |

## 0.6 其余已确认事实

- 非死锁（CPU 时间持续增长）
- 两个 worker 各 100% CPU；GPU utilization 100% 但**功耗仅 ~170W** → 瓶颈在 CPU 侧
- 同 DDP 配置下换任务可差 38 倍（VQAv2 >17 s vs ScienceQA 0.446 s）—— 因为 ScienceQA
  在 smoke 中只跑 1 步，恰好只付一次预热

---

## 1. 环境

| 项 | 值 |
|---|---|
| GPU | 2 × H100 80GB（DDP，`nproc_per_node=2`） |
| torch | 2.5.1+cu121 |
| transformers | 5.3.0 |
| triton | 3.1.0 |
| CPU | 48 核 |
| 模型 | `Qwen3_5ForConditionalGeneration`，9.41B 参数，其中可训练 1.64M（SMoPE adapter） |
| 架构 | 混合注意力：8 个 full-attention 层 `[3,7,11,15,19,23,27,31]`（SMoPE 注入）+ 其余约 24 层为线性注意力 `Qwen3_5GatedDeltaNet` |

**已装的性能相关依赖**：`flash-linear-attention==0.3.1`（`--no-deps` 安装）、`einops`。
`environment.json` 的 `linear_attention_kernels` 字段确认已生效：

```json
{"causal_conv1d_fn": false, "causal_conv1d_update": false,
 "chunk_gated_delta_rule": true, "fused_recurrent_gated_delta_rule": true}
```

未装 `causal-conv1d`（CUDA C++ 扩展，本机无 nvcc，预编译 wheel 在被墙的 GitHub）。
**不影响**：depthwise conv 会退化为普通 `nn.Conv1d`（`modeling_qwen3_5.py:559-565`），代价可忽略。

数据与模型均不在 davfs 挂载上（指令集在本地盘、模型在 `/dev/shm`、图像走 HTTP 直连）。

---

## 2. 症状

命令（`scripts/qwen/run.sh smope smoke --tasks VQAv2 --max-train-samples 200 --workers 0`，
batch_size=4，accumulation=2，2 卡 DDP → 每卡 25 个 batch）：

- `train_dense` 阶段 **>7 分钟未完成 → >17 秒/batch**
- 两个 worker 进程 **各 100% CPU**
- GPU utilization **100%**，但功耗只有 **~170W**（H100 满载应 400~700W）
- **不是死锁**：进程 CPU 时间持续增长（12 秒内 +12 秒 CPU）

CPU 饱和 + GPU 空转低功耗，指向**瓶颈在 CPU 侧**。

---

## 3. 实测数据

| 测量 | 值 | 条件 |
|---|---|---|
| collate 单独（4 真实样本：HTTP 取图 + processor 分词 + 图像预处理） | **1.258 s** | 单进程，Triton 缓存热 |
| 模型前向+反向单独（同一 batch，宽度 384） | **0.413 s** | 单卡无 DDP，缓存热 |
| **两项相加** | **1.670 s** | — |
| **真实训练实测** | **>17 s/batch** | 2 卡 DDP，`--workers 0` |
| ScienceQA 真实训练 `train_sparse` | **0.446 s/batch** | **同一 DDP 配置，仅任务不同** |
| 装 fla 前的 `preflight`（单样本一次前向反向） | 1.5 s | torch 回退 |
| 装 fla 后的 `preflight` | **20.0 ~ 37.2 s** | 同一形状，两次运行 |

**两个关键事实**：
1. **92% 的耗时无法由 collate + 模型计算解释**（17 s vs 1.67 s）
2. **同配置下换任务可差 38 倍**（VQAv2 >17 s vs ScienceQA 0.446 s）

---

## 4. 已排除的假设

| # | 假设 | 排除依据 |
|---|---|---|
| 1 | davfs 挂载慢 | 数据/模型/图像均已迁出挂载（本地盘 + `/dev/shm` + HTTP 直连 95MB/s） |
| 2 | 缺 fla 快核 | 已安装并确认 `chunk_gated_delta_rule: true`；装后微基准快 3.6 倍 |
| 3 | collate 阻塞主循环 | 实测仅 1.258 s |
| 4 | 模型前向/反向本身 | 实测仅 0.413 s |
| 5 | 进程死锁 | CPU 时间持续增长 |
| 6 | `--workers` 子进程启动开销 | 那是**每个 DataLoader** 约 24~38 s 的固定成本（且几乎不随 worker 数变化），影响 smoke 的**每个阶段**，但不解释**单步 >17 s** |

---

## 5. 待验证假说：**冷 Triton 缓存**

第 3 节里 0.413 s 的模型耗时是**缓存热**时测的 —— 该宽度已被此前多次 probe 编译过。若每个新宽度要编译**一整族**内核（约 24 个线性注意力层 × chunked/backward × 各 head 配置），一族耗时数分钟，那么 25 步里若出现 8 种不同宽度就是十几分钟 —— **与实测吻合**。

**支持证据**：
- Triton 磁盘缓存 **583MB / 4894 个条目**，且仍在增长
- 同一探针第二次运行：`preflight` 37.2 s → 20.0 s（缓存部分命中，仍偏慢）
- 装了 fla 后 `preflight`（**单样本**一次前向反向）从 1.5 s 涨到 20~37 s —— 多出的时间只可能是编译
- 固定 shape 的微基准 0.326 s/步 vs 变长真实训练 >17 s/步，差 50 倍

**验证方法（正在跑）**：用 `pad_to_multiple_of=128` 把宽度收敛到 128 的倍数后，**连跑两遍同一探针**。代码、数据、形状完全相同，唯一变量是磁盘缓存。
- 第 1 遍慢、第 2 遍快 → 假说成立，是**一次性冷缓存成本**
- 两遍都慢 → 假说不成立，需重新定位

**注意**：此前一次"连跑两遍"的对比**无效**，因为两遍都用**变长**输入，每个长度都是新的，缓存永远命中不了。必须两遍都用分桶输入。

---

## 6. 已做的改动（供参考，非最终方案）

| 文件 | 改动 |
|---|---|
| `qwen_smope/data.py` | `CoINCollator._encode()`：加 `pad_to_multiple_of=self.bucket`，把 batch 宽度收敛到有限几个 |
| `qwen_smope/data.py` | 新增 `bucket` 参数（默认 128）、`CoINCollator.__init__` 接收 |
| `qwen_smope/train.py` | 新增 `--length-bucket`（默认 128，0/1 关闭）并传入两个 collator |
| `qwen_smope/train.py` | `--workers` 默认 8 → **4**（见排除项 6） |
| `qwen_smope/train.py` | **修复上游 bug**：`save_checkpoint()` 原本是空壳（只 `sync()`），其函数体被误粘进 `record_dataset()` → `record_dataset` 引用未定义名而崩溃、且 `latest_adapter.pt` 从不写入、`--resume` 完全不可用 |

### 若假说成立，可能的方向（供 GPT 参考）
- **预热**：训练前对每种宽度跑一次 dummy 前向，把编译成本前置到可控阶段
- **固定宽度**：统一 pad 到 `max_length`（1024），形状只有一种；代价是多算 padding token
- **查 fla 的 autotune**：若开销来自 `@triton.autotune` 对每个新 key 重新 benchmark，固定成单一配置可消除
- **Triton 环境变量**：如 `TRITON_ALWAYS_COMPILE`、autotune 相关开关、`~/.triton/cache` 的管理
- **确认宽度种类数**：`pad_to_multiple_of=128` + `max_length=1024` 下最多 8 种；若实测步时仍高，说明编译不是主因

---

## 7. 我在本次勘测中犯过的两个错误（请勿重蹈）

1. **微基准用固定 shape，且把 collate 排除在计时循环之外**（只构造一次 batch 后反复计时模型）
   → 报出「0.326 秒/步」，并据此估算正式运行 24 小时。
   这两个偏差都**系统性低估**真实成本。正确做法是让基准走**真实的变长输入**、并把 collate 计入。

2. **误读 `stage()` 的计时**
   我一度看到 `preflight = 0.305 s` 就宣称「100 倍提速」。但 `stage()` 是 contextmanager，
   在 `finally` 里写事件 —— **异常路径也会写**。那次是形状不匹配（`IndexError: mask [384] vs tensor [326]`）
   在 forward 早期崩溃，0.305 s 是**崩溃耗时**，不是提速。

**方法论教训**：不要用与真实路径不同的方式去测真实路径；报告数字前先确认该数字来自**成功完成**的路径。

---

## 8. 复现命令

```bash
cd /mnt/zq/SMOPE
# 前提检查（指令集/模型/HTTP 端点）
bash /root/coin-stage/run_guarded.sh true

# 症状复现：VQAv2 200 样本，观察 train_dense 耗时
bash /root/coin-stage/run_guarded.sh bash scripts/qwen/run.sh smope smoke \
  --tasks VQAv2 --max-train-samples 200 --max-eval-samples 8 \
  --workers 0 --output /tmp/repro

# 分开量 collate 与模型
/mnt/zq/SMOPE/.venv2/bin/python -u /root/coin-stage/bench_collate_vs_model.py

# 监控真实进度（注意：不要用管道接 tail，会缓冲；且 events.jsonl 只在阶段结束时写）
watch -n5 'cat /tmp/repro/events.jsonl | tail -3'
```

**观测要点**：`nvidia-smi` 看功耗（170W = GPU 空转），`ps -o time=` 看 CPU 时间是否增长（增长 = 未死锁）。
