# Qwen3.5-9B CoIN smoke 优化与服务器交接

> 唯一权威诊断文档。本文取代原 `DIAGNOSIS.md`、`DIAGNOSIS_DDP.md`、
> `QWEN_PERFORMANCE.md` 和 `QWEN3_5_9B_CODE_AND_ACCURACY_ANALYSIS.md`。
> 最后更新：2026-10-14（Asia/Shanghai）。代码基线：GitHub `v2` 的
> `c9bf76c98d445fd61214b11856bf45067a65479a`；本文所述新增改动已提交为
> `4b0f436312bbd87e8bd44bb39ce09240c0fc3665`，服务器验证即在该提交上进行（§8）。

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
8. （2026-10-14 实测，§8）FLA 快路径在当前环境确认生效：四个核
   （causal_conv1d_fn/causal_conv1d_update/chunk_gated_delta_rule/
   fused_recurrent_gated_delta_rule）全为 true，且本机已装 causal-conv1d。
9. （2026-10-14 实测，§8.7）一次性 Triton autotune 的尺度：每进程首个宽度
   （384，batch=4）的 fwd+bwd 约 21.6s；此后每个新宽度约 6s；稳态每 microbatch
   约 0.46–0.50s（H100、bucket=128、长度 384–896）。
10. （2026-10-14 实测，§8.4）WebDAV 图像服务器在约 16–20 个并发客户端以上严重劣化
    （延迟 10–30ms→230ms+，个别请求 >115s 不返回）；扫描/训练并发应 ≤14。
11. （2026-10-14 实测，§8.4）OCR-VQA 训练集镜像缺图 54/165,348（0.033%），官方
    `prepare_data` 因此永远无法完成该任务；其余 7 任务无缺图。

### 1.2 推断

- ~~多宽度 autotune 可解释一部分冷启动时间，但不能单独解释旧 DDP run 的全部非线性变慢。~~
  **2026-10-14 更新（§8.7）**：在独占 H100 上，A/B/C/D 控制实验显示 DDP 本身没有复现
  历史非线性变慢（B 0.91×T1、C 0.99×T1、D 1.24×T1）；宽度错位预热的锁步等待确实存在，
  但在 4 个桶宽、每宽度 ~6s 预热的规模下只值 0.24×T1。旧“慢 45 倍”更可能与当时
  环境/并发干扰（如多探针并发、服务器抖动）有关，具体成因已无法复现验证。
- 完整 split 的剔除率并非全部 <1%：GQA 3.61%、Grounding 1.20%（material），论文实验
  报告必须逐任务披露；全部任务 <10%，不影响 full 的可运行性（除 OCR-VQA 缺图策略外）。

### 1.3 已证伪假说

- “缺少 Triton 快核”不是旧 20 秒冷调用的原因；当时已确认 FLA 快路径生效。
- “batch=1 导致 FLA 稳态很慢”与微基准不符。
- collate、CPU→GPU transfer、梯度检查或 causal-conv1d 不是旧单步 17 秒的主要来源。
- 首次 autograd 建图不是唯一原因；`no_grad()` 冷调用仍慢。
- Triton 编译缓存条目没有随同形状重复探针变化；历史慢点更符合 autotune，而非每次重新编译。
- 进程并非简单死锁：旧观测中 CPU 时间持续增长。
- **（2026-10-14，§8.7）**“DDP 在当前代码/独占 H100 上必然慢一个数量级”——不成立；
  B/C/D 相对单卡的实测比值 0.91/0.99/1.24。

### 1.4 待验证问题

- ~~真实八任务完整训练 split 的剔除数、逐任务剔除率，以及 `524377` 是否出现在新 manifest。~~
  **已解决（§8.3）**：全部 8 任务已扫描并归档；`524377` 在剔除明细中。
- ~~独占 H100 上 A/B/C/D 四阶段的逐 rank 时序；在获得原始 JSONL 前不得宣称 DDP 已修复。~~
  **已解决（§8.7）**：四阶段均通过 2×T1 判据，原始 JSONL 在
  `verify-20261014/ddp/*/rank-*.jsonl`。注意：这是 25 microbatch 的诊断，不等于
  八任务完整训练已验证。
- ~~单卡 VQAv2 200 个有效样本 smoke 是否完整结束，答案截断是否不再中止、checkpoint 是否能保存
  和恢复。~~ **已解决（§8.5、§8.6）**：smoke 正常结束；checkpoint 保存/加载验证通过，
  但“中途继续训练”未被覆盖（单任务序列的固有局限）。
- ~~新过滤扫描在 HTTP 图像通道上的一次性耗时与缓存复用效果。~~
  **已解决（§8.3）**：14 并发分片约 2 小时（详见 §8.3.3）；缓存复用验证为命中即免扫（
  `cache_built: false`，秒级）。
- 旧 profiler 与探针的原始日志未纳入仓库；论文引用数字前应在当前环境重新产出可归档日志。
  （本轮新产出的原始日志已归档在 `verify-20261014/` 下，但历史日志仍缺失。）
- **新**：OCR-VQA 缺图（54 条）与扫描健壮性的处理策略待维护者决定（§8.4）。
- **新**：D 阶段暴露的“rank 间宽度顺序不一致导致的锁步等待”在正式训练（shuffle=True、
  workers=4、8 任务更多宽度）下的量级未测（§8.7）。
- **新**：评测生成的 `</think>` 前缀（§8.5 观察）。

## 2. 本轮实现

| 日期 | 基线/状态 | 文件 | 行为变化 | 本地验证 | 剩余风险 |
|---|---|---|---|---|---|
| 2026-10-14 18:41:09 +08:00 | `c9bf76c` + 未提交修改 | `scripts/qwen/run.sh`, `qwen_smope/train.py` | 默认单卡；全局 batch 默认 16，按 world 自动推导 accumulation；smoke/preflight workers=0，full=4 | CLI 单测 | GPU 路径待 H100 |
| 2026-10-14 18:41:09 +08:00 | 同上 | `qwen_smope/data.py` | HTTP URL 统一 POSIX `/`；加入不截断长度测量与过滤后采样 | Windows URL、长度与顺序单测 | 真实 processor 待服务器 |
| 2026-10-14 18:41:09 +08:00 | 同上 | `qwen_smope/filtering.py`, `qwen_smope/prepare_data.py` | 完整 split 过滤、配置签名缓存、原子写入、剔除率分级与高比例门禁 | 缓存命中/失效、分级单测 | 全量扫描成本待测 |
| 2026-10-14 18:41:09 +08:00 | 同上 | `qwen_smope/diagnose_ddp.py` | 新增 A/B/C/D 诊断；每 rank 独立 JSONL，分段 CUDA 同步计时 | 参数与 world-size 单测、语法编译 | 诊断同步本身会扰动绝对吞吐，仅用于定位 |
| 2026-10-14 18:41:09 +08:00 | 同上 | `tests_qwen/test_qwen.py` | 覆盖过滤先于采样、`524377`、缓存、全局 batch、URL、trace 合约 | 28 项单测通过 | 无 CUDA 集成测试 |
| 2026-10-14 | `5ed8a62` + 未提交修改 | `qwen_smope/train.py`, `tests_qwen/test_qwen.py`, 本文与 `QWEN_SERVER.md` | route scan 从生成评测 batch=1 解耦，默认使用训练 batch=4；正式实验改为双卡独立 run，增加评测预算门禁与 ETA | 32/32 单测、compileall、diff check | batch 1/4 路由频次等价性与实际加速待 H100 短标定 |
| 2026-10-15 | 基于 `5ed8a62` 的本次 smoke-speed 提交 | `qwen_smope/{data,filtering,prepare_data,train,model,diagnose_ddp}.py`, `tests_qwen/test_qwen.py`, 本文与 `QWEN_SERVER.md` | 缺图分级与重试、分片过滤缓存、批量评测、确定性有限 route scan、checkpointing 对照开关及服务器逐步判据 | 34/34 单测、compileall、diff check | H100 上 T/E/R/C 各阶段均待验证；提交后在下一条记录补充精确 SHA |
| 2026-10-15 | 实现提交 `83ef103` | 上述 9 个文件 | 将本轮实现、测试与 DeepSeek 执行单固化为可追溯提交 | 34/34 单测、compileall、diff check | 本条仅回填 SHA；H100 结论仍全部待验证 |

本地验证环境没有可容纳 9B 模型的 GPU，也没有服务器 CoIN 数据，所以没有伪造 smoke/DDP
完成结果。已执行：

```bash
python -m compileall -q qwen_smope tests_qwen scripts/qwen
python -m unittest discover -s tests_qwen -p 'test_*.py' -v
```

历史首轮结果为 28/28；随着本轮新增覆盖，最新提交前结果为语法编译通过、34/34 单测通过。

## 3. 当前 CLI 与产物

### 3.1 默认训练语义

- `scripts/qwen/run.sh` 默认 `NPROC_PER_NODE=1`；只有显式设置为 2 才启用 DDP。
- `--global-batch-size` 默认 16，`--batch-size` 默认 4。未写 `--accumulation` 时：单卡推导为
  4，双卡推导为 2。若不能整除或显式值不一致，启动前报错。
- `--workers` 未指定时，preflight/smoke 为 0，full 为 4。
- `--routing-scan-batch-size` 默认复用训练 `--batch-size`（当前为 4）。路由扫描只做带 mask
  的前向，不受生成侧 `--eval-batch-size=1` 的限制；可显式设为 1 做服务器等价性复核。
- `--eval-batch-size` 现在允许大于 1，评测 collator 会临时使用 decoder-only 生成所需的左填充，
  随后恢复训练侧右填充。默认仍为 1，批量生成在通过 H100 预测等价性验证前不写成已确认加速。
- `--routing-scan-seed` 默认 42；当 `--routing-scan-samples` 非零时，从完整过滤后任务集做固定
  随机排列的前 N 条再恢复 source 顺序；因此同 seed 的 N=1000 是 N=5000 的真子集，不再取
  原数据的有偏前缀。routing JSON 记录数量、seed 和
  source-index SHA256。
- gradient checkpointing 默认开启；`--no-gradient-checkpointing` 仅用于 H100 显存/吞吐对照。
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
  --max-length 1024 --max-visual-tokens 256 --length-bucket 128 \
  --scan-workers 8
```

`--scan-workers` 默认 8；HTTP 通道硬限制为不超过 14，避免重现 §8.4 的 WebDAV 拥塞。
每个连续 source-index 分片会原子写入 `_filter_cache/.parts/`，中断后复用已完成分片；最终合并
前检查签名、覆盖范围、顺序、重复和遗漏，合并产物与单进程入口字节等价。

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

下表原始产物路径未写全前缀的，均相对 `outputs/coin-qwen3.5-9b/`：

| 项目 | 状态 | 原始产物位置 | 结论 |
|---|---|---|---|
| 八任务完整过滤 | **已完成**（2026-10-14） | `outputs/coin-qwen3.5-9b/_filter_cache/`、扫描日志与分片 `outputs/coin-qwen3.5-9b/verify-20261014/{partials,prepare}/` | 8 任务合计剔除 3,580/573,840（0.62%）；GQA 3.61%、Grounding 1.20% 为 material，其余 low；OCRVQA 另有 54 条缺图（见 §8.4） |
| VQAv2 `524377` 清单 | **已完成** | `_filter_cache/VQAv2-train-b1f6660412d30399.json` | 已在剔除明细：source_index 62，full_len 1297 |
| 单卡 VQAv2 200 smoke | **已完成** | `verify-20261014/runs/smope/smoke/seed-0/`、`verify-20261014/smoke/` | 167s 正常结束，无截断中止；过滤缓存命中 |
| checkpoint 保存/恢复 | **部分验证** | 同上 `latest_adapter.pt`、`resume_environment.json` | 保存→加载→签名校验→收尾通过；单任务序列无法覆盖中途继续（见 §8.6） |
| A：single | **已完成** | `verify-20261014/ddp/A-single/` | T1 = 58.5s（25 microbatch，含首次 21.6s 预热） |
| B：multiprocess-independent | **已完成** | `verify-20261014/ddp/B-multiprocess-independent/` | 最慢 rank 53.5s = 0.91×T1，通过 |
| C：ddp-same-batch | **已完成** | `verify-20261014/ddp/C-ddp-same-batch/` | 最慢 rank 57.9s = 0.99×T1，通过；两 rank loss/样本一致 |
| D：ddp-sharded | **已完成** | `verify-20261014/ddp/D-ddp-sharded/` | 最慢 rank 72.8s = 1.24×T1，通过（未超 2×）；但可见宽度错位预热代价（§8.7） |

汇总报告：`verify-20261014/ddp/summary.json`（analyze_ddp.py 生成）、
`verify-20261014/merge-validation.json`（分片=官方的字节等价证明）。

## 7. 测量教训

- 固定 shape 且把 collate 排除在循环外的微基准不能代表真实变长训练。
- `stage()` 在 `finally` 记录耗时；异常路径也会写事件。必须先确认阶段成功完成，再引用耗时。
- 不得并发运行 GPU 探针后把资源竞争误判为模型性能。
- 论文只引用当前代码、固定配置和可追溯原始产物产生的数字；历史摘要用于制定假说，不是最终证据。

## 8. 服务器验证记录（2026-10-14，`4b0f436`）

> 本节是服务器 agent 按 §5.1→§5.2 顺序执行后的实测记录。所有数字对应的原始产物都在
> `outputs/coin-qwen3.5-9b/verify-20261014/`（以及 `_filter_cache/`）下，仓库中不含产物。

### 8.1 环境与模型 staging

- `verify-20261014/env/environment.json`：python 3.11.16、torch 2.5.1+cu121、
  transformers 5.3.0、triton 3.1.0、fla 0.3.1、einops 0.8.2、2×H100（81.5GB）、48 核，
  git `4b0f436`、工作区干净。与 §4 预期一致。
- `/dev/shm` 重启后为空，重跑 `bash /root/coin-stage/stage_model_shm.sh`：19.33GB、
  4 分片、~96MB/s、约 3 分钟，`check_model` 通过。
- **FLA 快路径确认生效**：smoke `events.jsonl` 的 `linear_attention_kernels` 四项
  （`causal_conv1d_fn`/`causal_conv1d_update`/`chunk_gated_delta_rule`/
  `fused_recurrent_gated_delta_rule`）均为 true；本机**已装 causal-conv1d** 且被使用。

### 8.2 smoke 的过滤缓存复用

smoke 的 `training_filter` 事件显示 `cache_built: false`——`prepare_data` 产物被直接
命中，训练前不再重扫。缓存复用有效（秒级）。

### 8.3 八任务过滤扫描

#### 8.3.1 方法：分片扫描 + 合并（含等价性证明）

官方入口单进程扫 57.4 万样本在 HTTP 图像通道上耗时不可接受（VizWiz 20,523 条单进程实测
约 3.3 样本/s，即与 14 个分片并发时的表现；空闲时按 103ms/样本约 10 样本/s。全量外推
约 16–48 小时），因此把扫描按记录区间分片并行，合并成与官方入口
**逐字节一致**的 manifest。工具在 `/root/coin-stage/filter_shard.py`、`filter_merge.py`
（仓库外，未改仓库代码）。

等价性用两次独立实验证明（`verify-20261014/merge-validation.json` 与 `official-check/`）：

| 任务 | 官方单进程产物 | 分片+合并产物 | 结果 |
|---|---|---|---|
| ScienceQA（12,726，含图像样本） | `_filter_cache/ScienceQA-train-e389dce5d7e4b651.json` | `verify-20261014/merge-check/` 同名文件 | **逐字节一致**（sha256 `733b0773…`） |
| VizWiz（20,523，全图像） | `verify-20261014/official-check/VizWiz-train-6f215792382ee163.json` | `_filter_cache/VizWiz-train-6f215792382ee163.json` | **逐字节一致**（md5 `be0ac5ce…`） |

即：对纯文本、图文混合、全图像三类任务，分片合并的结果与官方入口不可区分。

#### 8.3.2 逐任务结果（`_filter_cache/`，共 8 份）

| 任务 | 原始 | 过滤后有效 | 剔除 | 剔除率 | 影响等级 | 剔除原因 |
|---|---|---|---|---|---|---|
| ScienceQA | 12,726 | 12,726 | 0 | 0.000% | low | — |
| TextVQA | 34,602 | 34,602 | 0 | 0.000% | low | — |
| ImageNet | 129,833 | 129,833 | 0 | 0.000% | low | — |
| GQA | 72,140 | 69,535 | 2,605 | **3.611%** | **material** | 全部超长（full_len 1025–2916，中位数 1178） |
| VizWiz | 20,523 | 20,523 | 0 | 0.000% | low | — |
| Grounding | 55,885 | 55,213 | 672 | **1.202%** | **material** | 全部超长 |
| VQAv2 | 82,783 | 82,534 | 249 | 0.301% | low | 全部超长（1025–5410，中位数 1217） |
| OCRVQA | 165,348 | 165,294 | 54 | 0.033% | low | **全部为缺图**（`image_unavailable`，见 §8.4） |
| **合计** | **573,840** | **570,260** | **3,580** | **0.624%** | — | 超长 3,526 + 缺图 54 |

GQA 与 Grounding 落在 `material`（1%–10%）：论文实验报告必须逐任务披露这两个剔除率；
两者都低于 10%，不触发 full 的默认拒绝。

#### 8.3.3 一次性扫描成本（回答问题：HTTP 图像通道耗时）

- 分片并发下（≤14 并发进程，每进程 ~5–9 样本/s）：全量 573,840 条实测 **约 2.5–3 小时**
  墙钟（含早期并发过高（31 分片）导致的重启损耗；干净复跑预计 ~2 小时）；
  单任务最重的 OCRVQA（165,348）单独约 2 小时。
- 成本构成（单样本，健康服务器）：HTTP 取图 ~54ms + chat template ~3ms + processor 编码
  （prompt+full 各一次、含视觉）~46ms ≈ **103ms**，理论上限 ~10 样本/s/进程。
- 缓存复用：命中后训练/评测启动不再扫（`cache_built: false`），只有换 processor、
  max-length、bucket 或源文件变更才会失效（缓存键含 processor 哈希与源文件
  size/mtime）。

#### 8.3.4 VQAv2 `524377`

出现在剔除明细：`source_index 62, prompt_len 1293, answer_len 4, full_len 1297,
reason full_length_exceeds_max_length`。注意其 prompt 本身（1293）已超过 max-length，
训练侧不再有机会触发“答案截断/为空”的抛出。

### 8.4 新发现：官方扫描对图像故障不健壮（需决策）

1. **HTTP 404 会中止整个任务的官方扫描**。`ImageFetcher` 对非 200 只重试一次就抛
   `FileNotFoundError`，`prepare_data` 无捕获。OCR-VQA 镜像**确实缺图**：
   - 独立并发探测：85,000 张中 33 张 404（0.039%，约 1/2,600）；
   - 全量分片扫描后合并结果：**54 条缺图**（165,348 中的 0.033%），例如
     `076030727X.jpg`、`038732464X.jpg`（服务器返回 404，curl 复核一致）。
   - 结论：**官方入口在 OCR-VQA 上永远无法完成**（第一条缺图即崩溃）。本轮用
     `--unmeasured-policy exclude` 把这 54 条记为剔除（reason `image_unavailable`，
     长度字段为 null），manifest 的键集合与官方一致；原始错误清单另存
     `OCRVQA-train-*.json.unmeasured.json`。**这是策略性偏离，必须由维护者确认**：
     是接受“缺图样本按剔除处理”，还是补齐镜像/更换数据源。
2. **瞬时 socket 超时会杀掉分片**（timeout=120s，重试一次仍失败即中止；ScienceQA 实测
   被杀过一次）。
3. **WebDAV 服务器在 ~16–20 并发客户端以上严重劣化**（实测量化）：
   - 单客户端延迟 10–30ms → 230ms+，个别请求 >115s 不返回（curl 实测挂起）；
   - 31 个并发分片时：聚合吞吐 ~93 样本/s，CPU 利用率却只有 ~5/48 核（全部在等服务器）；
   - 降到 8 个并发后：延迟立刻回到 8–74ms，单进程 8–16 样本/s。
   - 历史“8 线程 447 文件/s”只在轻并发下成立。**建议扫描/训练并发 ≤14**，并在
     长时间批量读图时监控服务器延迟。

### 8.5 单卡 VQAv2 200 smoke（§5.1.3/§5.1.4）

命令：`OUTPUT_ROOT=…/verify-20261014/runs bash scripts/qwen/run.sh smope smoke
--tasks VQAv2 --max-train-samples 200`；输出
`verify-20261014/runs/smope/smoke/seed-0/`，日志 `verify-20261014/smoke/smoke-run.log`。

- **167s 正常结束**（wall_seconds），无“答案截断”中止；200/200 样本完成
  （`samples_with_ddp_padding: 200`）。
- 阶段耗时：model_loading 6.2s / preflight 39.1s / train_dense 67.1s（50 microbatch）/
  train_sparse 35.0s（50 microbatch）/ route_scan 1.35s / evaluate_VQAv2 3.57s。
- 显存：训练峰值 allocated 28.6GiB / reserved 30.6GiB；模型 9,411,452,400 参数、
  可训练 1,638,656；full-attention 层 [3,7,11,15,19,23,27,31]。
- `dataset_manifest.json`：`original 82783 → valid 82534 → sampled 200`，
  `excluded 249 (0.30%)，impact low`，剔除明细含 `524377`；
  `train:VQAv2:train` 与 `eval:VQAv2:test`（4 条）两条记录齐全。
- 产物齐全：`environment.json / events.jsonl / latest_adapter.pt (9.9MB) /
  routing/after_01_VQAv2.json / predictions/after_01_VQAv2/VQAv2.jsonl /
  accuracy_matrix.csv / summary.json`。
- 观察（非阻断）：评测生成的 `prediction` 带 `</think>` 前缀（如
  `"</think>\n\nYes"`，归一化后判分正常）。4 个样本不足以定性；疑似评测 prompt 与
  `enable_thinking=False` 的生成路径交互，建议在正式评测前确认。

### 8.6 checkpoint 与 `--resume`

同一输出目录、同参数加 `--resume` 重跑：41s 完成，写出 `resume_environment.json`，
加载 `latest_adapter.pt`、签名校验通过、`next_task=1` 后无待跑任务直接收尾
（`run_complete`）。**覆盖范围**：保存→加载→签名一致→正常退出。**未覆盖**：单任务
中途被杀后的继续训练——需要 ≥2 任务的任务序列才能验证，建议正式长跑前用
`--tasks VQAv2,OCRVQA` 的小样本序列补一次“杀进程→resume”。

### 8.7 DDP 决策树 A/B/C/D（§5.2）

固定条件：模型 `/dev/shm/qwen35-9b`、VQAv2、过滤缓存、batch=4、accumulation=2、
25 microbatch、bucket=128、GPU 独占（前置 nvidia-smi 均为 0MiB；运行前后快照见
`verify-20261014/ddp/*-nvidia-smi-*.txt`）。原始 trace：各阶段 `rank-*.jsonl` 与
`configuration.json`；汇总 `verify-20261014/ddp/summary.json`。

| 阶段 | 变体 | 最慢 rank 墙钟 | 相对 T1 | 判据 ≤2×T1 | 备注 |
|---|---|---|---|---|---|
| A | single | **58.5s（T1）** | 1.00 | — | 首次 microbatch fwd+bwd 21.6s（预热），稳态中位 0.462s |
| B | multiprocess-independent | 53.5s | 0.91 | 通过 | 两进程不包 DDP，无异常 |
| C | ddp-same-batch | 57.9s | 0.99 | 通过 | 两 rank loss 与样本 ID **完全一致** |
| D | ddp-sharded | 72.8s | 1.24 | 通过 | 两 rank 样本不相交；但可见明确机制性变慢 |

**结论**：在独占 H100 上，历史“DDP 慢 45 倍/1174 秒”**没有复现**；B、C 与单卡几乎同速，
D 的 1.24×T1 也远低于 2× 判据。

**D 变慢的机制（逐 microbatch 分析，值得记录）**：

- 四个桶宽 [384, 512, 640, 896] 在两个 rank 上**集合相同、顺序不同**（DistributedSampler
  分片后各自遇见的顺序不同）。
- 每个新宽度在每进程内都是一次性 Triton autotune 代价：首个 384 约 20–21s，其余每宽度
  约 6s（C 与 D 的 `sequence_width` 首次出现时刻实测）。
- C 中两 rank 同一步遇到同一宽度，预热时刻对齐，无等待；D 中预热时刻错位，DDP 的锁步
  梯度同步让每个 rank 在同步点等待对方——实测锁步和 `Σ max(r0,r1)` 为 79.3s，而单 rank
  自身计算和只有 ~66s；多出来的时间全部是**等待**（data_wait 反而更小：D 5.8–6.7s vs
  C 8.4–9.1s，排除数据加载因素）。
- 这与 §5.2 的假说一致（“分片后 rank 间宽度与 autotune 不一致”），但量级温和
  （4 个宽度、预热仅 6s/宽度）。**正式训练（shuffle=True、workers=4、8 任务更多宽度）
  的量级未测**；若届时 D 型开销明显，§5.2 的预案（包 DDP 前各 rank 按相同顺序预热
  全部桶宽）可以对症下药。

### 8.8 给维护者的疑问清单

1. **OCR-VQA 缺图 54 条的策略**：接受“剔除”还是补图？若要入库代码化，建议在
   `filtering` 增加 `reason: image_unavailable` 的正式分支（含门禁等级），并把
   `ImageFetcher` 的 404 与超时区分为“永久/瞬时”。当前偏离只发生在扫描工具层。
2. **官方扫描的健壮性**：一次瞬时超时/一张缺图即杀掉整任务扫描、且不留部分进度，
   57 万样本的成本无法承受第二次。是否接受“重试加固 + 分片合并”作为标准流程
   （本轮 Scripts 在 `/root/coin-stage/`，可随时入库）？
3. **WebDAV 并发上限**：~16–20 客户端即劣化。是服务器侧配置（worker/连接数）所致吗？
   能否为批量扫描提高上限？
4. **D 的错位预热**：是否需要按 §5.2 预案实现“包 DDP 前的同序预热”？取决于正式训练
   中 D 型开销的实际量级（本轮 1.24×T1 未触发阈值）。
5. **评测 `</think>` 前缀**：评测生成是否与训练 prompt 对齐？
6. **`--resume` 的中途继续**：建议补一次“两任务小样本、杀进程、resume”的端到端验证。

### 8.9 本轮复用的工具（仓库外，可入库）

`/root/coin-stage/`：`filter_shard.py`（分片扫描，30s 超时 + 重试 + 记录 unmeasured）、
`filter_merge.py`（合并，`--unmeasured-policy fail|exclude`，默认 fail）、
`probe_missing_images.py`（缺图探测）、`run_ddp_tree.sh`（A/B/C/D）、
`analyze_ddp.py`（判据汇总）。合并逻辑已对照官方入口做字节等价验证（§8.3.1）。

## 9. 维护者分析与代码化决策（2026-10-14 23:37:07 +08:00）

本节是读取 §8 的服务器结论后，在 GitHub `v2` 提交 `5ed8a62` 基础上做出的维护决策；改动
尚需在服务器复核，不把本地单测等同于 H100 实测。

### 9.1 已采纳并实现

1. **永久缺图进入正式过滤语义**：HTTP 404/410 或 mount 上确定不存在的图片记录为
   `reason: image_unavailable`，保留 task、sample_id、source_index、HTTP 状态和位置。
   因真实视觉 token 无法在缺图时计算，三个长度字段明确为 `null`，不得伪造文本侧长度。
2. **瞬时故障不得污染数据集**：socket/HTTP 异常及 408/425/429/5xx 最多尝试 4 次，单次
   timeout 30 秒并指数退避；全部失败后让扫描失败。只有永久 404/410 才能成为剔除记录。
3. **分片扫描正式入库**：`prepare_data` 默认 8 个 spawn 进程，HTTP 最大 14；连续分片原子落盘、
   可断点复用，并在确定性合并前验证分片签名及完整覆盖。训练仍可直接读取现有 8 份缓存，
   不会因为实现入库而强制重扫。
4. **缓存验证加固**：命中 manifest 时验证有效/剔除 index 有序、唯一、不相交且恰好覆盖完整
   source split，避免损坏或错误合并的缓存静默进入训练。

### 9.2 明确不采纳或延期

- **暂不实现 DDP 同序桶宽预热**：D 比 A 多 14.3 秒但仍通过预设 2× 判据；这是一轮进程内
  autotune 的一次性成本。诊断没有保持单/双卡相同有效全局 batch，也没有正式八任务吞吐，
  当前不足以证明预热能改善总训练时间。先保留单卡默认。
- **不升级 torch/Triton/FLA，不启用 static graph、bucket 调参或 autotune monkey-patch**：
  当前快路径正常，DDP 数量级变慢未复现，修改依赖栈会引入更大的数值与兼容性变量。
- **不删除原始 prediction 中的 `</think>`**：评分器的 `final_answer()` 已在所有任务解析前剥离
  该前缀；保留原始文本便于审计。需要用更大评测样本确认准确率，而不是改变生成路径掩盖现象。
- **不实现 microbatch 级恢复**：协议本来是任务边界 checkpoint，任务内中断重跑。先补两任务
  小样本的任务边界恢复测试，避免把 checkpoint 粒度扩展成新的研究变量。

### 9.3 本地验证与服务器复核项

本轮新增单测覆盖：永久缺图审计、瞬时错误不被吞掉、503 后成功重试、404 不重试、分片合并与
单进程 manifest 字节等价、HTTP 并发上限。代码健壮性的服务器复核应只改扫描实现：用现有缓存
确认训练不重扫；另选一个小任务在独立临时缓存目录以 `--scan-workers 8` 重建，验证分片复用和
最终 SHA。不要删除现有正式缓存。

若要独立审计 §8.7 并决定未来是否预热，需要下载
`outputs/coin-qwen3.5-9b/verify-20261014/ddp/`（尤其 `summary.json` 与所有 `rank-*.jsonl`）和
`outputs/coin-qwen3.5-9b/_filter_cache/`。仅继续常规单卡正式实验时，这些原始产物不阻塞。

### 9.4 恢复双卡默认前缺少的严格吞吐对照

§8.7 的 A 固定 accumulation=2，因此单卡有效全局 batch=8；D 的双卡有效全局 batch=16。
它们用于定位机制是有效的，但不能证明双卡在相同训练协议下更快。下一轮应使用同一批前 1600 个
有效样本、全局 batch=16、100 次 optimizer update：单卡 400 microbatch、accumulation=4；
双卡每 rank 200 microbatch、accumulation=2。新 CLI 的 `--global-batch-size 16` 是断言，配置
不一致会直接失败；summary 记录每 rank 样本数、更新数和有效全局 batch。
同时记录可训练参数的 sum、L2 norm 与非有限值计数，用于检查两 rank 一致性及 E/F 训练结果漂移。

按“一次一条命令”纪律，先运行 E 并回传产物：

```bash
CUDA_VISIBLE_DEVICES=0 python -m qwen_smope.diagnose_ddp \
  --variant single --output outputs/coin-qwen3.5-9b/verify-20261014/ddp/E-single-gbs16 \
  --model-path /dev/shm/qwen35-9b --coin-root /mnt/zq/SMOPE/coin_local \
  --image-transport http --filter-cache outputs/coin-qwen3.5-9b/_filter_cache \
  --steps 400 --batch-size 4 --accumulation 4 --global-batch-size 16
```

确认 E 完成、`samples_seen=1600`、`optimizer_updates=100` 后才运行 F：

```bash
CUDA_VISIBLE_DEVICES=0,1 torchrun --standalone --nproc_per_node=2 \
  -m qwen_smope.diagnose_ddp \
  --variant ddp-sharded --output outputs/coin-qwen3.5-9b/verify-20261014/ddp/F-ddp-gbs16 \
  --model-path /dev/shm/qwen35-9b --coin-root /mnt/zq/SMOPE/coin_local \
  --image-transport http --filter-cache outputs/coin-qwen3.5-9b/_filter_cache \
  --steps 200 --batch-size 4 --accumulation 2 --global-batch-size 16
```

判据预先固定：F 两 rank 的样本 ID 并集必须与 E 完全一致且无重复，均为 100 次 update；F 最慢
rank 墙钟必须严格小于 E，loss 和可训练参数统计无异常，才说明双卡有吞吐收益。若只快几秒，考虑到正式训练的
复杂度和额外故障面，仍保持单卡默认；只有优势稳定且明显时才继续正式多任务 smoke。

### 9.5 正式实验启动决策与时长风险（2026-10-14）

读取下载到本地的 `verify-20261014/`、8 份正式 manifest 和逐 rank JSONL 后，决定**不把 E/F
作为正式实验的前置门槛**。原因是正式矩阵包含 LoRA、SMoPE 和多个 seed；B 阶段已经证明两张
H100 各跑一个独立单卡进程没有资源竞争（最慢 rank 53.5s，小于 A 的 58.5s）。因此正式 sweep
优先把两个独立 run 分配到两张卡，而不是让两张卡同步一个 DDP run。E/F 只用于将来考虑恢复
DDP 默认值，不阻塞单卡正式协议。

不过仍不能直接用当前默认参数启动 full，存在两个已量化的时长风险：

1. `--max-eval-samples=0` 表示全量留出集。一个训练方法的 36 个矩阵单元共生成 **800,935**
   条；zero-shot 为 **346,437** 条。单 seed 的 zero-shot + LoRA + SMoPE 合计 **1,948,307**
   次生成，按 smoke 的 4 条 VQAv2/3.57s 粗略外推已经是按周计，不能把它写成可接受的默认时长。
2. 旧实现把 SMoPE 路由扫描错误地绑在 `eval_batch_size=1`，每个 seed 额外做 570,260 个单样本
   前向。现改为独立 `--routing-scan-batch-size`，默认 4。该改动在数学上保持每条样本及 attention
   mask 不变，但 H100 上的频次 JSON 等价性和加速比仍标为**待验证**，不能把理论加速写成实测。

正式启动前只做两项短标定，不再重跑 A/B/C/D：

- route scan：同一 seed、同 400 条 ScienceQA，分别 batch 1 与 4；要求 routing JSON 中的整数
  频次、protected 状态完全一致、无非有限值，并预先规定 `T_batch4 <= 0.8 * T_batch1`；否则
  保留可配置入口但把正式默认值退回 1。
- evaluation：zero-shot、8 任务、每任务 100 条，测各任务真实 `samples/s`。据此再冻结论文使用的
  `--max-eval-samples`；这个值必须在所有方法、seed、轮次一致，不能跑到一半修改。

在服务器标定前只能给区间，不给伪精确 ETA。若冻结 `max_eval_samples=1000`、3 个训练 seed、
zero-shot 只跑 seed 0，并用两张 H100 并行独立单卡作业，当前证据支持的保守墙钟预算为
**4–7 天**；只跑一个 seed 约 **1.5–2.5 天**。若保持全量评测（0），三 seed 很可能升到
**数周**。标定完成后应用实测 train/route/eval 吞吐重算并记录最终 ETA。

### 9.6 单 seed 24 小时目标：优化顺序与不可混淆的变量（2026-10-15）

读取 A–D trace 后，**不建议仅增大 global batch**。A 的 optimizer 总耗时仅 0.100s/58.486s
（0.17%）；在每卡 batch 仍为 4 时，把 global batch 16 改为 32 只是把极小的 optimizer 次数
再减半，所有样本的 forward/backward 仍会执行，并会改变优化协议和学习率语义。

正确的第一优先级是增大每卡物理 batch、同步降低 accumulation，保持 global batch=16：

| 每卡物理 batch | accumulation | global batch | 协议判断 |
|---:|---:|---:|---|
| 4 | 4 | 16 | 当前基线 |
| 8 | 2 | 16 | 第一候选 |
| 16 | 1 | 16 | H100 显存允许时的目标 |

虽然三者每次 optimizer update 都含 16 条样本，但当前 loss 是“microbatch 内 token 均值再按
microbatch 等权累积”，所以不同物理 batch 不保证逐 bit 相同；必须在正式实验前冻结一个设置，
LoRA/SMoPE/所有 seed 共用，不能中途切换。

为了把单 seed 压到 24 小时，按下面顺序只改一个变量并预先固定判据：

1. **训练物理 batch**：同一 GQA 1600 样本依次测 4/8/16，global batch 固定 16。要求无 OOM、
   非有限值或样本遗漏，峰值 allocated <72GiB；采用满足条件且训练阶段最快的设置，候选相对
   batch=4 至少快 20%。不要用增大 global batch 冒充吞吐提升。
2. **批量生成**：同一 zero-shot 8×100 固定评测集对照 eval batch 1/4，batch 4 必须拥有完全
   相同的 sample ID、解析后 final answer 和 score，且总评测时间至少缩短 30%；通过后再试 8。
   原始字符串若不同但解析答案相同仍需保留 diff，不能静默忽略。
3. **有限 route scan**：建议候选为每任务固定随机 5000 条、seed=42，而不是全量 570,260 条。
   这是 SMoPE 算法口径变化，不是纯工程等价优化；必须比较 1000/5000/全量（或尽可能大的参考）
   推导出的 `frequency >= positive-frequency mean` 保护集合。5000 与参考至少 99.5% 一致才采用，
   否则提高到 10000 或保留全量，并在论文披露采样数和 seed。
4. **关闭 gradient checkpointing**：仅在上述物理 batch 确定后单独测。要求峰值 <72GiB、无 OOM，
   且训练阶段至少再快 15% 才采用；否则保持默认开启。

若物理 batch=8/16、eval batch=4、route scan=5000 均通过，`max_eval_samples=1000` 的单 seed
目标预计可降到 **约 16–24 小时**（两张 H100 分别并行 LoRA 与 SMoPE）；这是待 H100 标定的
预算，不是完成结果。若任一项未通过，优先保证协议和预测一致性，而不是为了命中 24 小时改大
global batch 或减少训练样本。

### 9.7 给服务器 DeepSeek 的逐步执行单（2026-10-15，当前均待验证）

本节是服务器 agent 的直接入口。**不要从“正式命令”开始，也不要一次跑完所有候选。**每阶段只
运行当前要求的一条命令，保存原始产物并回传判据结果；维护者确认后才进入下一阶段。A–D 的旧
DDP 诊断已经回答根因问题，本轮不重跑，也不启用 DDP、同序预热、static graph 或依赖升级。

#### 9.7.1 前置检查与统一纪律

```bash
cd /mnt/zq/SMOPE
source .venv2/bin/activate
git status --short
git log -1 --oneline
nvidia-smi
```

要求：工作区干净；HEAD 是 GitHub `v2` 最新的 smoke-speed 提交；两张 H100 无其他进程；模型
`/dev/shm/qwen35-9b` 完整；8 份缓存仍在 `outputs/coin-qwen3.5-9b/_filter_cache/`。本轮只能复用
缓存，禁止删除或重扫。所有标定固定 GPU0，GPU1 保持空闲，避免把并发调度收益混入单变量比较。

每条命令都应额外保存：stdout/stderr、运行前后 `nvidia-smi`、对应 `environment.json`、
`events.jsonl`、`dataset_manifest.json`、routing/predictions，以及 `git rev-parse HEAD`。若出现 OOM，
记录后回退物理 batch；**不要降低 max-length、视觉 token 或训练样本来制造“通过”**。

#### 9.7.2 阶段 T：训练物理 batch（先 T4，再 T8；T8 通过后才跑 T16）

T4 控制组：

```bash
CUDA_VISIBLE_DEVICES=0 \
OUTPUT_ROOT=outputs/coin-qwen3.5-9b/verify-20261015/train-b4 \
bash scripts/qwen/run.sh smope smoke \
  --tasks GQA --max-train-samples 1600 \
  --batch-size 4 --global-batch-size 16 \
  --eval-batch-size 1 --max-eval-samples 4 \
  --routing-scan-samples 1 --routing-scan-batch-size 1 \
  --workers 0
```

T8 只改变物理 batch（accumulation 自动从 4 变为 2）：

```bash
CUDA_VISIBLE_DEVICES=0 \
OUTPUT_ROOT=outputs/coin-qwen3.5-9b/verify-20261015/train-b8 \
bash scripts/qwen/run.sh smope smoke \
  --tasks GQA --max-train-samples 1600 \
  --batch-size 8 --global-batch-size 16 \
  --eval-batch-size 1 --max-eval-samples 4 \
  --routing-scan-samples 1 --routing-scan-batch-size 1 \
  --workers 0
```

先回传 T4/T8，确认 T8 无 OOM/非有限值/样本遗漏、两个训练阶段 update 语义一致、峰值
allocated <72GiB，且 `(dense+sparse)_T8 <= 0.8 * (dense+sparse)_T4`。只有全部通过，才把 T8
命令的输出目录和 `--batch-size` 改为 `train-b16`/`16`；global batch 仍为 16，accumulation
自动为 1。采用满足显存余量且最快的候选，不因数字更大而自动选 T16。

#### 9.7.3 阶段 E：批量生成（训练 batch 冻结后执行）

E1 基线：

```bash
CUDA_VISIBLE_DEVICES=0 \
OUTPUT_ROOT=outputs/coin-qwen3.5-9b/verify-20261015/eval-b1 \
bash scripts/qwen/run.sh zero_shot full \
  --eval-batch-size 1 --max-eval-samples 100 \
  --eval-sample-seed 42 --workers 4
```

E4 只改变 eval batch：

```bash
CUDA_VISIBLE_DEVICES=0 \
OUTPUT_ROOT=outputs/coin-qwen3.5-9b/verify-20261015/eval-b4 \
bash scripts/qwen/run.sh zero_shot full \
  --eval-batch-size 4 --max-eval-samples 100 \
  --eval-sample-seed 42 --workers 4
```

回传两套 `predictions/**.jsonl` 和 events。要求每个任务样本 ID 集合、`score`，以及任务对应的
`parsed_prediction` 或 `normalized_prediction` 完全一致，总评测时间 `T_E4 <= 0.7*T_E1`，无
padding warning、异常或 OOM。原始 `prediction` 若有任何差异也要报告并保留 diff。E4 通过后
才可按相同方法尝试 E8；若 E4 不通过，正式设置退回 1。

#### 9.7.4 阶段 R：route scan 工程等价性与采样稳定性

先完成 §9.5 的 ScienceQA 400 条 batch 1/4 对照：routing JSON 必须逐字节一致，且
`T_batch4 <= 0.8*T_batch1`。然后在已冻结的训练物理 batch 下，对同一任务、同一训练样本和
seed 分别运行 `--routing-scan-samples 1000`、`5000`、`10000`；`--routing-scan-seed 42` 不变。
固定随机排列保证 1000 是 5000 的子集、5000 是 10000 的子集。回传三套 routing JSON，由
维护者根据 `frequency >= positive-frequency mean` 重新计算保护集合；5000 与 10000 一致率
达到 99.5% 才能把 5000 写入正式协议，否则提高上限。这里不允许 DeepSeek 自行宣布“差不多”。

#### 9.7.5 阶段 C：gradient checkpointing（最后、可选）

复制已经获胜的训练物理 batch 命令，只增加 `--no-gradient-checkpointing` 并换新输出目录。
峰值 allocated 必须 <72GiB、无 OOM/非有限值，且 dense+sparse 至少再快 15% 才采用；否则保持
默认开启。不要把关闭 checkpointing 与改变 batch 放在同一次对照中。

#### 9.7.6 状态表与停止条件

| 阶段 | 当前状态 | 通过后冻结 | 未通过行为 |
|---|---|---|---|
| T4/T8 | **已完成（2026-10-15，§10.1）**：T8/T4=0.979，未达 0.8 判据 | 最快且 <72GiB 的物理 batch | **按预设规则保持 batch=4** |
| T16 | **不运行**（T8 未通过，前置条件不满足） | batch=16 或退回 8 | 退回 8，不改 global batch |
| E1/E4 | **已完成（2026-10-15，§10.2）**：时序 0.532× 达标，但 70 个 score/311 条原始输出不一致 | eval batch=4；可再试 8 | **语义判据不通过 → 退回 eval batch=1；E8 不运行** |
| route batch 1/4 | **已完成（2026-10-15，§10.3）**：速度 0.443× 通过，但整数频次 1,614/3,200 不一致 | routing-scan batch 跟随已冻结物理 batch | **等价性不通过 → 退回 1** |
| route 1000/5000/10000 | **已完成（2026-10-15，§10.3）**：5000-vs-10000 99.625% 合格（§9.7.4 口径）；5000-vs-全量 99.438% 不合格（§9.6 口径），10000-vs-全量 99.750% | 稳定的最小 N 与 seed=42 | 提高 N 或全量 |
| no-checkpointing | **已完成（2026-10-15，§10.4）**：0.722×（≥15% 判据通过），峰值 51.0 GiB < 72 GiB | 仅在显存和 15% 判据通过时关闭 | 保持开启 |
| 单 seed 正式实验 | **禁止启动** | 所有必需项冻结且维护者确认 | 继续标定 |

最终候选（不是授权开跑）为：global batch 16、物理 batch 8/16、eval batch 4、固定评测
N=1000、route scan N=5000/seed=42、两张卡各跑一个独立单卡方法。所有结果必须先写回本节状态表
并附原始产物路径；只有维护者确认 ETA <=24h 且语义判据通过，才启动 GPU0 LoRA / GPU1 SMoPE。

## 10. 服务器标定记录（2026-10-15，`8123181`）

> 按 §9.7 执行单逐步实施。所有标定固定 GPU0，GPU1 全程空闲；命令与环境与 §9.7 逐字一致，
> 每条命令的输出目录唯一。本节目的是回填 §9.7.6 状态，不替代维护者确认。

### 10.1 阶段 T：训练物理 batch T4/T8

前置：工作区干净、HEAD=`8123181`=origin/v2；两卡 0MiB；模型 staging 完整；8 份过滤缓存
在位（GQA 命中原缓存，`cache_built: false`，剔除 2,605/72,140=3.61% material 与 §8.3.2 一致）。
T4/T8 均 `--tasks GQA --max-train-samples 1600 --global-batch-size 16 --workers 0`，
唯一变量为物理 batch（accumulation 自动 4→2），每阶段各 1,600 样本、100 次 optimizer update。

| 指标 | T4（batch 4） | T8（batch 8） | 判据 | 结果 |
|---|---:|---:|---|---|
| train_dense | 440.53s | 424.27s | — | — |
| train_sparse | 345.73s | 345.42s | — | — |
| dense+sparse | **786.26s** | **769.70s** | T8 ≤ 0.8×T4 | **0.979×，未通过** |
| 峰值 allocated | 30.11 GiB | 42.60 GiB | <72 GiB | 通过 |
| 样本数 | 1600+1600 | 1600+1600 | 无遗漏 | 通过 |
| 非有限值/OOM | 无 | 无 | 无 | 通过 |
| loss（dense/sparse） | 0.1486 / 0.1654 | 0.1484 / 0.1647 | 语义一致 | 通过（非逐 bit 相同，符合 §9.6 预期） |

**决策（按 §9.7.2 预设规则）**：T8 未达 0.8 判据 → **冻结物理 batch=4**；T16 前置条件
不满足，未运行。原始产物：`verify-20261015/{train-b4,train-b8}/`（含 `events.jsonl`、
`environment.json`、`dataset_manifest.json`、`routing/`、`predictions/`、`latest_adapter.pt`）、
运行日志 `verify-20261015/train-b{4,8}-run.log`、前后 `nvidia-smi` 快照、
`verify-20261015/train-comparison.json`、`git-head.txt`。

**机制观察（供维护者，不改变上述预设结论）**：T4 与 T8 的**每样本**训练耗时几乎不变
（T4 dense 0.275 s/样本 vs T8 dense 0.265 s/样本；sparse 0.216 vs 0.216）。即在该配置下
（workers=0、HTTP 直连、gradient checkpointing 开启），单步耗时随物理 batch 近线性增长，
扩大物理 batch 不产生吞吐收益；成本对本实验更接近“按样本线性”，而不是按 microbatch 计。
这与 §9.6 的预期（扩大物理 batch 是首要提速手段）不符，但本轮严格只改物理 batch 一个变量，
没有在同一对照中混入 workers/评测变量；不追加“换配置再跑一次凑判据”的尝试。

### 10.2 阶段 E：批量生成 E1/E4（§9.7.3）

命令与 §9.7.3 逐字一致（zero_shot full，8 任务 ×100 样本，`--eval-sample-seed 42`，
`--workers 4`），唯一变量 `--eval-batch-size` 1→4。GPU0、独占，两卡运行前后 0MiB。

**时序**（逐任务 `zero_shot_evaluation` stage，秒）：

| 任务 | E1（batch 1） | E4（batch 4） | 速度比 |
|---|---:|---:|---:|
| ScienceQA | 31.19 | 28.00 | 0.90 |
| TextVQA | 45.38 | 25.13 | 0.55 |
| ImageNet | 24.84 | 16.78 | 0.68 |
| GQA | 23.72 | 16.81 | 0.71 |
| VizWiz | 28.92 | 18.79 | 0.65 |
| Grounding | 174.52 | 59.25 | 0.34 |
| VQAv2 | 24.18 | 17.04 | 0.70 |
| OCRVQA | 31.65 | 22.53 | 0.71 |
| **合计** | **384.40** | **204.34** | **0.532** |

wall：E1 406.79s，E4 226.72s（0.557×）。**时序判据 T_E4 ≤ 0.7×T_E1 通过**；
两套日志均无 padding warning、异常或 OOM（`eval-b{1,4}-run.log` 零告警）。

**语义判据不通过**（逐样本对照 `eval-b1-b4-comparison.json`，脚本 `compare_eval.py`）：
样本 ID 集合逐任务一致，但 8/8 任务出现差异——score 不一致合计 70/800（OCRVQA 29、
VizWiz 12、ImageNet 10、TextVQA 9、VQAv2 4、ScienceQA 3、GQA 3、Grounding 0），
解析答案不一致合计 215/800（Grounding 91、OCRVQA 45、VizWiz 25、ImageNet 19、
TextVQA 14、VQAv2 10、GQA 6、ScienceQA 5），原始输出不同 311/800。总体平均准确率
55.0%→59.7%（OCRVQA 46→71、ImageNet 29→37、VizWiz 60→64 为大头）。原始 diff 全量
存档于 `eval-b1-b4-comparison.json`。

**机制（供维护者）**：生成是贪心解码（`do_sample=False, num_beams=1`），评测 collator 的
左填充与 attention_mask 传递已核对无误（`CoINEvalCollator`），processor 逐图编码、无批内
耦合。差异形态符合“bf16 内核在不同 batch 形状下的归约次序差异 → 近持平 argmax 翻转 →
贪心解码级联”这一数值机制：Grounding 99/100 原始输出在坐标末位抖动（IoU 判分不变、
score 0 个差异），长答案任务则首个 token 翻转即整句改变（如 OCRVQA
`'Unanswerable'`→`'Science & Math'`、`'No'`→`'no'`）。该差异不是 padding 正确性缺陷，
但在当前内核/精度下**无法保证 batch>1 与 batch=1 语义等价**。

**决策（按 §9.7.3 预设规则）**：语义判据不通过 → **正式设置退回 `--eval-batch-size 1`；
E8 不运行**。原始产物：`verify-20261015/{eval-b1,eval-b4}/zero_shot/full/seed-0/`
（events、predictions、summary）、`eval-b{1,4}-run.log`、前后 nvidia-smi、
`eval-b1-times.json`、`eval-b1-b4-comparison.json`。

**附：实测评测吞吐（供冻结 `--max-eval-samples`）**——batch 1 各任务 samples/s：
ScienceQA 3.21、TextVQA 2.20、ImageNet 4.03、GQA 4.22、VizWiz 3.46、Grounding 0.57、
VQAv2 4.14、OCRVQA 3.16；8 任务合计 384.4s/800 条 ≈ **2.08 条/s**（workers=4，含
worker 启动）。冻结 N 与 ETA 由维护者按 §9.5 决定；本记录只提供实测吞吐，不给结论。

### 10.3 阶段 R：route scan（batch 1/4 已完成；N 稳定性进行中）

**方法修正（留给维护者的偏差记录）**：标准入口把 route scan 的数据集绑定为**训练集对象**
（`scan_routes(model, train_set, ...)`，`source_count = len(train_set)`）。smoke 下
`--max-train-samples` 会把扫描源截断（实测 `--max-train-samples 1` 时
`sample_count=1, source_count=1`，即第一次尝试只扫了 1 条，产物保留在 `route-b1/`），
而 full 下扫全任务集前后要付完整任务训练。为在不支付任务训练的前提下复现生产语义
（§3.1 的“从完整过滤后任务集做固定随机排列的前 N 条”），本轮用驱动脚本
`verify-20261015/route_scan_calib.py`：复用仓库的 `scan_routes`/`prepare_train_dataset`/
`load_model`，扫描源为完整过滤集（断言 12,726），每次扫描前用 `lightweight_state`/
`load_lightweight_state` 恢复同一模型状态。**偏差**：模型状态为刚加载（未训练任务）的
SMoPE 状态，而非任务训练后状态；b1/b4 等价性与 N 稳定性是批处理/采样机制问题，
未训练的近似均匀路由器是近持平（最保守）的对照情形。

**batch 1/4 对照**（同 seed=42、同 400 条，来自 12,726 的固定随机子集）：

| 配置 | 耗时 | 判据 | 结果 |
|---|---:|---|---|
| batch 1, N=400 | 58.49s（6.84 条/s） | — | 基线 |
| batch 4, N=400 | 25.92s（15.43 条/s） | T_b4 ≤ 0.8×T_b1 | **0.443×，通过** |
| batch 4, N=400 重复 | 16.70s（23.96 条/s，首次含形状预热） | 可复现 | 与首次 **逐字节一致** |

**等价性判据不通过**：batch 4 与 batch 1 的 routing JSON **非逐字节一致**。
逐项：`task_frequency` 3,200 个整数单元中 1,614 个不同（总选择数两边均 256,000=400×8×16×5）；
最大单元差 136（单头总量 2,000）；两套扫描的 `protected` 字段相同（0/8 层差异，但该字段是
扫描前的既有状态，恢复快照后必然一致，不构成证据）；按 §9.7.4 规则（cumulative frequency
>= positive-frequency mean，与 `expert.next_task()` 逐式一致）从两套 JSON 推导的 protected
集合一致率 97.750%（3,128/3,200；true 单元 773 vs 763）。

**机制（供维护者）**：未训练路由器的 logits 接近均匀，近持平 argmax 多；bf16 下 batch 形状
改变内核归约次序 → 选择翻转 → 整数频次改变（与 §10.2 E 阶段同一类数值机制）。b4 重复运行
逐字节可复现，说明差异不是随机性而是确定性的批形状数值差。

**决策（按 §9.7.6 预设规则）**：整数频次非完全一致 → **route scan batch 退回 1**；
N 稳定性序列（1000/5000/10000/full，seed 42）随即在 **batch 1** 下运行
（产物 `route-calib/routing/ScienceQA-b1-N*.json`）。
成本含义（batch 1 实测稳态 ~10.4–11.1 条/s，N≤任务集时为线性）：
全量扫描 570,260 条约 15.3 小时/方法/seed；N=10000×8 任务约 2.1 小时；N=5000 约 1.1 小时。

**N 稳定性（§9.7.4/§9.6 判据；从 cumulative frequency 按 next_task 规则逐式推导，脚本
`analyze_route_stability.py`，原始数值 `route-calib-b1-stability.txt`）**：

| 扫描 | 耗时 | 推得 protected=true 单元 |
|---|---:|---:|
| N=1000 | 93.0s | 790 |
| N=5000 | 489.8s | 816 |
| N=10000 | 963.5s | 824 |
| N=12726（全量） | 1147.2s | 830 |

一致率矩阵：

| | N=1000 | N=5000 | N=10000 | 全量 |
|---|---:|---:|---:|---:|
| N=1000 | — | 99.125% | 98.875% | 98.688% |
| N=5000 | | — | **99.625%** | 99.438% |
| N=10000 | | | — | 99.750% |

**留给维护者的两个口径（本节不做采用决定）**：§9.7.4 的字面判据“5000 与 10000 一致率
≥99.5%”**满足**（99.625%）；§9.6 的参考口径“5000 与参考（full/尽可能大）≥99.5%”
**不满足**（5000 vs 全量 99.438%），而 10000 vs 全量 99.750% 满足。即：若以全量为参考，
证据支持 10000 或保留全量；若采用 §9.7.4 口径则 5000 合规。请维护者按意向口径裁定并
写明论文披露的 N 与 seed。**重要 caveat**：本轮扫描状态为未训练路由器（近似均匀频率
分布，均值规则最敏感），且为单次扫描的 cumulative frequency；正式训练中 protected 由
跨任务累计频率在任务边界冻结，分布的集中度不同，N 稳定性的绝对值可能变化。

### 10.4 阶段 C：关闭 gradient checkpointing（§9.7.5）

命令 = 已冻结的 T4 命令 + `--no-gradient-checkpointing`（唯一变量），输出 `train-nogc/`。

| 指标 | T4（GC 开启） | noGC | 判据 | 结果 |
|---|---:|---:|---|---|
| train_dense | 440.53s | 314.29s | — | — |
| train_sparse | 345.73s | 253.24s | — | — |
| dense+sparse | 786.26s | **567.53s** | ≤0.85×T4 | **0.722×，通过（快 27.8%）** |
| 峰值 allocated | 30.11 GiB | 51.00 GiB | <72 GiB | 通过（reserved 53.36 GiB） |
| 样本数 / 非有限值 / OOM | 1600+1600 | 1600+1600，无 | 无 | 通过 |
| loss（dense/sparse） | 0.1486/0.1654 | 0.1486/0.1654 | 语义一致 | 通过（与 T4 同批逐位一致，checkpointing 数值等价） |

**决策（按 §9.7.5 预设规则）**：三项判据全部通过 → **正式训练采用
`--no-gradient-checkpointing`**，显存代价 +20.9 GiB（51.0 vs 30.1，仍低于 72 GiB 门禁）。
原始产物：`verify-20261015/train-nogc/`、`train-nogc-run.log`、前后 nvidia-smi。

**未测交叉项（仅提示，不自行加测）**：T8（物理 batch 8）在 noGC 下未测。本轮 T8 的结论
是“每样本耗时近恒定”的串行成本主导，noGC 省下的是每样本的计算时间，不改变批扩展行为；
预计 T8+noGC 约为 ~0.97×T4+noGC，仍不满足 T 判据，但这只是推断，未实测。

### 10.5 本轮标定汇总与 ETA 输入（供维护者按 §9.5 重算；以下为实测吞吐的算术，不是结论）

实测吞吐（GPU0 独占、HTTP 图像通道、GQA/ScienceQA 样本）：

| 阶段 | 冻结设置 | 实测 | 备注 |
|---|---|---|---|
| 训练 dense | batch 4, noGC, workers=0 | 0.196 s/样本（314.29s/1600） | 仅 SMoPE 首任务需要 |
| 训练 sparse | batch 4, noGC, workers=0 | 0.158 s/样本（253.24s/1600） | 每任务 1 epoch |
| route scan | batch 1, workers=0 | 稳态 ~10.4–11.1 条/s（N≥1000） | N=400 首次 7.0/s（含预热） |
| 评测 batch 1 | workers=4 | 8 任务合计 2.08 条/s；Grounding 0.57、其余 2.2–4.2 | 100 条/任务实测 |

按冻结设置（noGC、scan batch 1、eval batch 1）的算术示例——**单方法单 seed**：
训练 sparse 570,260×0.158≈25.0h；SMoPE dense 12,726×0.196≈0.7h（LoRA 无）；
route scan N=5000×8 任务≈1.1h（N=10000≈2.1h）；评测（N=1000/任务、36 个矩阵单元按逐任务
实测速率）≈4.8h；即 SMoPE ≈31.5h、LoRA ≈29.8h、zero-shot ≈1.1h。**这些是 workers=0
标定口径下的算术**；full 模式 workers=4 的流水线重叠未参与标定，实际可能偏移，维护者
应按正式设置复核后再冻结 ETA。若 N=5000 按 §9.7.4 口径采用，单 seed 双卡并行（LoRA/SMoPE）
的墙钟估计仍在 30h 量级，超出 §9.6 的 16–24h 目标，主要瓶颈是训练吞吐而非评测或扫描。
