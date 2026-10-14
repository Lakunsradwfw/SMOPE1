# Qwen3.5-9B CoIN smoke 优化与服务器交接

> 唯一权威诊断文档。本文取代原 `DIAGNOSIS.md`、`DIAGNOSIS_DDP.md`、
> `QWEN_PERFORMANCE.md` 和 `QWEN3_5_9B_CODE_AND_ACCURACY_ANALYSIS.md`。
> 最后更新：2026-10-14（Asia/Shanghai）。代码基线：GitHub `v2` 的
> `c9bf76c98d445fd61214b11856bf45067a65479a`；本文所述新增改动尚未提交。

## 1. 当前结论

### 1.1 已确认事实

1. `flash-linear-attention==0.3.1` 的 Triton autotune 会按新的序列宽度在进程内付费。
   历史探针中，同进程同一宽度第一次前向+反向约 19.95 秒，之后约 0.265 秒；冷前向
   profiler 出现约 7 万次 `fill_`/`zero_`，符合 autotune benchmark 清缓存的特征。
   当时的 Triton 3.1.0 环境没有观察到 autotune 结果落盘。
2. `--length-bucket=128` 已把历史 smoke 中的宽度收敛到
   `[256, 384, 512, 640, 768, 896]`。更换图片而宽度不变不会再次触发该轮调优。
3. FLA 稳态核并不慢：历史微基准中，batch 1/2/4/8 的前向+反向约为
   0.0032/0.0041/0.0049/0.0046 秒，约比 torch 回退快 10 倍。causal-conv1d 在本次
   模型路径上不是主要瓶颈。
4. smoke 中 `workers>0` 会为每次 DataLoader 构造增加约 24–38 秒固定启动成本；因此
   preflight/smoke 应使用 0，full 才使用 4。
5. davfs 不是可用的数据热路径：历史实测读取 233 MB 指令文件约 582 秒（~0.4 MB/s），
   HTTP 直连约 2.4 秒（~97 MB/s）。当前图像通道继续使用 HTTP。
6. VQAv2 样本 `524377` 曾稳定触发“答案被截断或为空”并中止训练。训练侧现已改为先扫描
   完整 split、确定性过滤，再应用 `--max-train-samples`；评测侧只构造 prompt，不生成
   labels，不应用该过滤。
7. 历史报告中的单/双卡比较存在口径错误：200 样本、batch=4 时，单卡是 50 个
   microbatch、accumulation=4 时 25 次更新；双卡每 rank 是 25 个 microbatch、
   accumulation=2 时约 13 次更新。原“25.55 秒 vs >1140 秒、慢 45 倍”不是严格控制变量
   的加速比，只能作为 DDP 异常的线索，不能写成论文结论。

### 1.2 推断

- 多宽度 autotune 可解释一部分冷启动时间，但不能单独解释旧 DDP run 的全部非线性变慢。
- DDP 异常可能来自多进程资源竞争、DDP 包装/同步路径，或分片后 rank 间宽度与 autotune
  不一致；三者尚未被控制实验区分。
- 若完整 split 的逐任务剔除率都低于 1%，训练分布影响大概率可忽略，但仍需在论文数据统计中
  披露；该判断只有服务器生成真实 manifest 后才成立。

### 1.3 已证伪假说

- “缺少 Triton 快核”不是旧 20 秒冷调用的原因；当时已确认 FLA 快路径生效。
- “batch=1 导致 FLA 稳态很慢”与微基准不符。
- collate、CPU→GPU transfer、梯度检查或 causal-conv1d 不是旧单步 17 秒的主要来源。
- 首次 autograd 建图不是唯一原因；`no_grad()` 冷调用仍慢。
- Triton 编译缓存条目没有随同形状重复探针变化；历史慢点更符合 autotune，而非每次重新编译。
- 进程并非简单死锁：旧观测中 CPU 时间持续增长。

### 1.4 待验证问题

- 真实八任务完整训练 split 的剔除数、逐任务剔除率，以及 `524377` 是否出现在新 manifest。
- 独占 H100 上 A/B/C/D 四阶段的逐 rank 时序；在获得原始 JSONL 前不得宣称 DDP 已修复。
- 单卡 VQAv2 200 个有效样本 smoke 是否完整结束，答案截断是否不再中止、checkpoint 是否能保存
  和恢复。
- 新过滤扫描在 HTTP 图像通道上的一次性耗时与缓存复用效果。
- 旧 profiler 与探针的原始日志未纳入仓库；论文引用数字前应在当前环境重新产出可归档日志。

## 2. 本轮实现

| 日期 | 基线/状态 | 文件 | 行为变化 | 本地验证 | 剩余风险 |
|---|---|---|---|---|---|
| 2026-10-14 18:41:09 +08:00 | `c9bf76c` + 未提交修改 | `scripts/qwen/run.sh`, `qwen_smope/train.py` | 默认单卡；全局 batch 默认 16，按 world 自动推导 accumulation；smoke/preflight workers=0，full=4 | CLI 单测 | GPU 路径待 H100 |
| 2026-10-14 18:41:09 +08:00 | 同上 | `qwen_smope/data.py` | HTTP URL 统一 POSIX `/`；加入不截断长度测量与过滤后采样 | Windows URL、长度与顺序单测 | 真实 processor 待服务器 |
| 2026-10-14 18:41:09 +08:00 | 同上 | `qwen_smope/filtering.py`, `qwen_smope/prepare_data.py` | 完整 split 过滤、配置签名缓存、原子写入、剔除率分级与高比例门禁 | 缓存命中/失效、分级单测 | 全量扫描成本待测 |
| 2026-10-14 18:41:09 +08:00 | 同上 | `qwen_smope/diagnose_ddp.py` | 新增 A/B/C/D 诊断；每 rank 独立 JSONL，分段 CUDA 同步计时 | 参数与 world-size 单测、语法编译 | 诊断同步本身会扰动绝对吞吐，仅用于定位 |
| 2026-10-14 18:41:09 +08:00 | 同上 | `tests_qwen/test_qwen.py` | 覆盖过滤先于采样、`524377`、缓存、全局 batch、URL、trace 合约 | 28 项单测通过 | 无 CUDA 集成测试 |

本地验证环境没有可容纳 9B 模型的 GPU，也没有服务器 CoIN 数据，所以没有伪造 smoke/DDP
完成结果。已执行：

```bash
python -m compileall -q qwen_smope tests_qwen scripts/qwen
python -m unittest discover -s tests_qwen -p 'test_*.py' -v
```

结果：语法编译通过，28/28 单测通过。

## 3. 当前 CLI 与产物

### 3.1 默认训练语义

- `scripts/qwen/run.sh` 默认 `NPROC_PER_NODE=1`；只有显式设置为 2 才启用 DDP。
- `--global-batch-size` 默认 16，`--batch-size` 默认 4。未写 `--accumulation` 时：单卡推导为
  4，双卡推导为 2。若不能整除或显式值不一致，启动前报错。
- `--workers` 未指定时，preflight/smoke 为 0，full 为 4。
- 图像默认 HTTP，长度桶默认 128；未升级 torch/Triton/FLA，也未启用 DDP bucket、
  static graph 或 autotune monkey-patch。

训练输出中的 `dataset_manifest.json` 按 `train:<task>:<split>` 记录：数据源、原始数、过滤后
有效数、实际采样数、剔除数、剔除率、影响等级、确认状态，以及每条剔除样本的 `task`、
`sample_id`、`prompt_len`、`answer_len`、`full_len`、原因。实现缓存还保存完整有效 source index，
缓存键不包含 `max_train_samples`，因此所有方法和 seed 共享同一清单。

影响等级与门禁：

| 逐任务剔除率 | 标记 | 行为 |
|---|---|---|
| `<1%` | `low` | 可运行，论文简要披露 |
| `1%–<10%` | `material` | 可运行，实验报告必须逐任务披露 |
| `>=10%` | `high` | full 默认拒绝；复核后显式加 `--allow-high-filter-rate`，确认状态写入 manifest |

### 3.2 独立准备入口

先生成八任务的完整过滤缓存；此命令不加载 9B 权重，但会使用真实 processor、chat template、
图片和视觉 token 设置：

```bash
cd /mnt/zq/SMOPE
source .venv2/bin/activate
python -m qwen_smope.prepare_data \
  --model-path /dev/shm/qwen35-9b \
  --coin-root /mnt/zq/SMOPE/coin_local \
  --image-transport http \
  --filter-cache outputs/coin-qwen3.5-9b/_filter_cache \
  --max-length 1024 --max-visual-tokens 256 --length-bucket 128
```

必须在继续训练前检查每个任务打印的 `excluded_rate`，并直接查看缓存 JSON 中
`excluded_samples`。VQAv2 的 `524377` 若未出现，应停止并核对数据版本、processor 与长度配置，
而不是绕过检查。

## 4. 服务器前提与执行纪律

目标环境为 2×H100 80GB。历史环境记录为 torch 2.5.1+cu121、transformers 5.3.0、
triton 3.1.0、FLA 0.3.1、48 CPU 核；执行前要把实际版本写入产物，不能假设未变化。

预期布局：

- 仓库与本地指令集：`/mnt/zq/SMOPE`、`/mnt/zq/SMOPE/coin_local`
- 模型及 processor：`/dev/shm/qwen35-9b`（tmpfs，重启后需重新 staging）
- 图像：HTTP 直连，禁止回退到 davfs 热路径
- 过滤缓存：`outputs/coin-qwen3.5-9b/_filter_cache`

执行纪律：GPU 必须独占；一次只改变一个变量；服务器 agent 每次先提出一条命令，执行后回传
完整 stdout/stderr、`nvidia-smi` 快照和新增原始产物，再决定下一步。不得并发跑性能探针，不得把
失败路径的 `stage` 耗时当成成功耗时。`TORCH_DISTRIBUTED_DEBUG=DETAIL` 只在 collective 不一致
或挂起时启用，不用于性能比较。

## 5. 服务器验证顺序

### 5.1 数据过滤与单卡 smoke

1. 运行第 3.2 节准备入口并归档所有过滤 JSON。
2. 检查八任务逐任务剔除率、原因分布与 `524377`。
3. 单卡运行 VQAv2 200 个**过滤后有效样本**的 smoke；确认不因答案截断中止。
4. 检查 `dataset_manifest.json`、`events.jsonl`、`latest_adapter.pt`，再用同一输出目录验证
   `--resume`。只有这些通过后才进入 DDP 决策树。

### 5.2 DDP 决策树

诊断入口为 `python -m qwen_smope.diagnose_ddp --variant ... --output ...`。A 使用一个进程；
B/C/D 使用 `torchrun --standalone --nproc_per_node=2`。固定同一模型、过滤缓存、task、batch=4、
25 microbatch 和 GPU 独占状态。每个输出目录必须为空且唯一。

| 阶段 | variant | 唯一改变 | 预先固定的判据 | 下一步 |
|---|---|---|---|---|
| A | `single` | 基线，单卡固定 25 microbatch，得到 `T1` | 两次重复应无异常且样本/宽度一致 | 通过后 B |
| B | `multiprocess-independent` | 两进程、两卡，不包 DDP；两 rank 使用相同 batch | 最慢 rank `<=2×T1` | 超过则停止 DDP 调参，排查 CPU/HTTP/autotune/进程竞争 |
| C | `ddp-same-batch` | 在 B 上只增加 DDP 包装/同步 | 最慢 rank `<=2×T1`，两 rank loss/样本一致 | 超过则按 forward/backward/optimizer 分段定位 |
| D | `ddp-sharded` | 在 C 上只改为分片数据 | 最慢 rank `<=2×T1` | 只有 D 超过阈值才支持 rank 间宽度/autotune 不一致假说 |

若且仅若 D 暴露宽度不一致问题，才实施“包 DDP 前各 rank 按相同顺序预热全部桶宽”，随后按同一
判据复测。不要提前尝试 bucket 参数、static graph、monkey-patch 或依赖升级。

每个 `rank-<rank>.jsonl` 包含样本 ID、序列宽度、数据等待、传输、forward、backward、
optimizer、同步状态、UTC/单调时间、loss 和异常；`configuration.json` 记录参数、world size、
过滤 manifest 与调试环境变量。双卡只有在相同数据清单和有效全局 batch 下严格快于单卡，并保持
loss、样本计数和 checkpoint 行为一致，才可恢复为默认。

## 6. 原始产物与结果状态

历史探针脚本位于服务器 `/root/coin-stage/`，已知名称包括 `probe_warmup.py`、
`probe_cold_first_forward.py`、`bench_collate_vs_model.py`、`run_guarded.sh`；旧报告只保存了摘要，
仓库中没有对应原始日志。新实验必须把以下内容复制到持久输出目录，不能只在聊天中报数字：

- 数据准备 stdout/stderr 与 `_filter_cache/*.json`
- smoke 的 `environment.json`、`dataset_manifest.json`、`events.jsonl`、checkpoint 恢复日志
- DDP 每阶段 `configuration.json`、全部 `rank-*.jsonl`、stdout/stderr、运行前后 `nvidia-smi`
- 失败时的 traceback 和最后一条已完成 microbatch；不得覆盖失败目录

| 项目 | 状态 | 原始产物位置 | 结论 |
|---|---|---|---|
| 八任务完整过滤 | 待验证 | 待服务器填写 | 不得推断剔除率 |
| VQAv2 `524377` 清单 | 待验证 | 待服务器填写 | 必须出现在剔除明细 |
| 单卡 VQAv2 200 smoke | 待验证 | 待服务器填写 | 不得宣称截断问题已在真实数据修复 |
| checkpoint 保存/恢复 | 待验证 | 待服务器填写 | — |
| A：single | 待验证 | 待服务器填写 | — |
| B：multiprocess-independent | 待验证 | 待服务器填写 | — |
| C：ddp-same-batch | 待验证 | 待服务器填写 | — |
| D：ddp-sharded | 待验证 | 待服务器填写 | — |

## 7. 测量教训

- 固定 shape 且把 collate 排除在循环外的微基准不能代表真实变长训练。
- `stage()` 在 `finally` 记录耗时；异常路径也会写事件。必须先确认阶段成功完成，再引用耗时。
- 不得并发运行 GPU 探针后把资源竞争误判为模型性能。
- 论文只引用当前代码、固定配置和可追溯原始产物产生的数字；历史摘要用于制定假说，不是最终证据。
