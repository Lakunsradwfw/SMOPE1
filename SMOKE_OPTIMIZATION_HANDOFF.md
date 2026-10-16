# Qwen3.5-9B CoIN smoke 优化与服务器交接

> 唯一权威诊断文档。本文取代原 `DIAGNOSIS.md`、`DIAGNOSIS_DDP.md`、
> `QWEN_PERFORMANCE.md` 和 `QWEN3_5_9B_CODE_AND_ACCURACY_ANALYSIS.md`。
> 最后更新：2026-10-16（Asia/Shanghai）。本轮维护者裁定与实现基于 GitHub `v2`
> `1371f14ebc04f0d4c4671da90c7a1ad7386f58c7`；服务器下一步只执行 §12.8，不再从早期章节
> 自行挑选实验。历史提交与各轮服务器验证 SHA 保留在对应章节。

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
12. （2026-10-15 重新核算，§8.7、§10.8）A 单卡在 58.49s 内处理 100 条，D 双卡分片在
    72.76s 内处理 200 条。D 的单 rank 墙钟虽为 `1.24×T1`，但有用样本吞吐是
    **2.749 vs 1.710 sample/s，即 1.608× 单卡**。因此“D 比 A 慢”是口径错误；不过 A 的
    有效全局 batch=8、D=16，尚不能把 1.608× 当成同训练协议下的正式加速比。
13. （2026-10-15，§12.6）P2 的 8 个任务训练与 full-route 标定均完成，训练输入等待最多
    0.53%，加权 route 吞吐 21.29 sample/s；但第 36 个评测 cell 的对象汇总挂死，故 P2 **尚未
    完成**，17.1–17.7h 仍是临时 ETA。
14. （2026-10-15，§12.7）LoRA 在 GC 开启、batch=4、global batch=16 时双卡比单卡快
    1.782×，P3 通过；同配置关闭 GC 会 OOM，因此正式 LoRA 必须保留默认 GC。

### 1.2 推断

- ~~多宽度 autotune 可解释一部分冷启动时间，但不能单独解释旧 DDP run 的全部非线性变慢。~~
  **2026-10-14 更新（§8.7）**：在独占 H100 上，A/B/C/D 控制实验显示 DDP 本身没有复现
  历史非线性变慢（B 0.91×T1、C 0.99×T1、D 单 rank 墙钟 1.24×T1）；D 实际同时处理两倍
  样本，有用吞吐为单卡 1.608×。宽度错位预热的锁步等待确实存在，
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
  B/C/D 的最慢 rank 墙钟相对单卡为 0.91/0.99/1.24；其中只有 D 做有用分片，其未经协议
  对齐的样本吞吐为单卡 1.608×。

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
- **新**：独立 Gloo 对象通信是否能让 P2 从 task7 checkpoint 完成 OCRVQA、8 个末轮评测、
  checkpoint 与 `run_complete`；这是 §12.8 的唯一下一项服务器验证。

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
| 2026-10-15 | 实现提交 `8df9249` | `qwen_smope/compare_ddp.py`, `qwen_smope/train.py`, `tests_qwen/test_qwen.py`, 本文与 `QWEN_SERVER.md` | 修正 DDP 墙钟/吞吐口径；增加协议匹配 E/F 自动审计；epoch 增加 input wait 与宽度分布 | 36/36 单测、compileall、diff check | E/F 与生产 8000 条单/双卡对照待独占 H100 |
| 2026-10-16 | 基于 `1371f14` 的对象通信修复 | `qwen_smope/train.py`, `scripts/qwen/setup.sh`, `tests_qwen/test_qwen.py`, 本文与 `QWEN_SERVER.md` | NCCL 仅承载张量；Python 对象集合改走10分钟超时的独立 Gloo 组并校验语义标签；LoRA 正式命令保留 GC；setup/运行入口校验 peft | 38/38 单测、compileall、diff check 通过；本机 Windows 沙箱无可用 Bash，未重复执行 `bash -n` | 必须用 §12.8 的 P2 resume 在双 H100 验证，未回传前不声称死锁已修复 |

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
- 双卡时 loss/计数等张量 collective 继续使用 NCCL；预测行、宽度字典和显存记录等 Python
  对象使用独立 Gloo 进程组，超时10分钟，并携带调用点标签检查 rank 间顺序。该变更不改变样本、
  更新、loss、路由或评测语义。
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
triton 3.1.0、FLA 0.3.1、48 CPU 核；执行前要把实际版本写入产物，不能假设未变化。LoRA
还必须确认 `peft==0.18.1`；`scripts/qwen/setup.sh` 现在会导入并打印该版本，缺包必须按完整
`requirements-qwen.txt` 修复，不能把失败环境当成方法失败。

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

| 阶段 | 变体 | 最慢 rank 墙钟 | 全局有用样本 | 有用吞吐 | 备注 |
|---|---|---:|---:|---:|---|
| A | single | **58.5s（T1）** | 100 | 1.710 sample/s | 首次 microbatch fwd+bwd 21.6s（预热），稳态中位 0.462s |
| B | multiprocess-independent | 53.5s | 两份相同的 100 | 不合并（两个独立作业） | 两进程不包 DDP，无异常 |
| C | ddp-same-batch | 57.9s | 100（两 rank 重复） | 1.726 sample/s | 只隔离 DDP 包装成本，不是数据并行提速实验 |
| D | ddp-sharded | 72.8s | 200（两 rank 不相交） | **2.749 sample/s（A 的 1.608×）** | 有错位预热等待，但有效吞吐仍更高 |

**结论**：在独占 H100 上，历史“DDP 慢 45 倍/1174 秒”**没有复现**；B、C 与单卡几乎同速。
旧汇总只给出 D 的单 rank 墙钟 `1.24×T1`，容易误读为双卡更慢；按全局不重复样本计算，D
实际得到 **1.608× 有用吞吐**。但 A/D 的有效全局 batch 分别为 8/16，严格训练协议结论仍以
§9.4 的 E/F 为准。

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
  --variant single --output outputs/coin-qwen3.5-9b/verify-20261015b/ddp/E-single-gbs16 \
  --model-path /dev/shm/qwen35-9b --coin-root /mnt/zq/SMOPE/coin_local \
  --image-transport http --filter-cache outputs/coin-qwen3.5-9b/_filter_cache \
  --steps 400 --batch-size 4 --accumulation 4 --global-batch-size 16
```

确认 E 完成、`samples_seen=1600`、`optimizer_updates=100` 后才运行 F：

```bash
CUDA_VISIBLE_DEVICES=0,1 torchrun --standalone --nproc_per_node=2 \
  -m qwen_smope.diagnose_ddp \
  --variant ddp-sharded --output outputs/coin-qwen3.5-9b/verify-20261015b/ddp/F-ddp-gbs16 \
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
| 单 seed 正式实验 | **暂不启动（见 §10.6）**：当前约 32.2h 的中心值来自 workers=0/GQA 外推，LoRA 未实测 | 所有必需项冻结且维护者确认 | 先完成真实 full 输入流水线与 LoRA 吞吐标定 |

§10 标定后的**当前临时基线**（仍不是授权开跑）为：global batch 16、物理 batch 4、noGC、
eval batch 1、固定评测 N=1000、route batch 1、route N=10000/seed=42（N 仍须训练后状态复核），
两张卡各跑一个独立单卡方法。batch 8、eval batch 4、route batch 4 和 route N=5000 均不得沿用
旧“候选”措辞静默启用。只有 §10.6 的输入流水线与 LoRA 标定完成、最终 ETA 和路由 N 获维护者
确认后，才启动 GPU0 LoRA / GPU1 SMoPE。

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
实测速率）原记录粗算≈4.8h；即 SMoPE ≈31.5h、LoRA ≈29.8h、zero-shot ≈1.1h。**这些是 workers=0
标定口径下的算术**；full 模式 workers=4 的流水线重叠未参与标定，实际可能偏移，维护者
应按正式设置复核后再冻结 ETA。若 N=5000 按 §9.7.4 口径采用，单 seed 双卡并行（LoRA/SMoPE）
的墙钟估计仍在 30h 量级，超出 §9.6 的 16–24h 目标，主要瓶颈是训练吞吐而非评测或扫描。

### 10.6 维护者复核：25 小时的含义、修正 ETA 与下一步（2026-10-15）

本地已取得 `verify-20261015/` 的 93 个文件（约 43.2 MB），核对了 events、environment、
predictions/routing、三份对照脚本与日志。以下结论取代 §10.5 的粗算，但不把未跑的实验写成
结果。

#### 10.6.1 先纠正两个口径

1. **25 小时不是 davfs 挂载传输时间，也不是纯 GPU 计算时间。**标定的
   `environment.json` 明确是 `image_transport=http`；davfs 没进入训练热路径。但 T/C 标定按
   §9.7 使用 `workers=0`，因此每个 batch 的 HTTP 取图、processor/chat template 与 GPU
   前后向串行计入 `train_sparse`。full 默认 `workers=4` 会预取并重叠这些工作，所以把
   0.158 s/样本直接乘 570,260 会系统性偏向保守。现有 events 没有逐步 input-wait，尚不能把
   25 小时拆成“网络多少、GPU 多少”。
2. **标准 full 入口本身没有 route 数据源偏差。**full 且 `max_train_samples=0` 时，
   `train_set` 就是完整过滤集。偏差只发生在本轮缩短训练的 calibration：smoke 会先截断
   `train_set`，外置脚本虽恢复了完整扫描源，却使用未训练路由器。因此它足以否决 batch=4 的
   逐字节等价性，但 N 稳定性不能冒充“训练后路由器”的最终证据。

#### 10.6.2 修正后的单 seed 算术

持续评测不是“八任务平均耗时 ×36”。任务顺序固定时，ScienceQA 到 OCRVQA 在三角矩阵中分别
出现 8、7、6、5、4、3、2、1 次。用 E1 原始逐任务时间加权，`N=1000` 的正式矩阵为
**4.32h**，不是 §10.5 的 4.8h。

| 方案 | 当前可算部分（每 seed） | 中心值 | 证据边界 |
|---|---|---:|---|
| Qwen3.5 zero-shot | 8 任务各评测 1000 条 | **1.07h** | 由 E1 800 条线性外推；无训练 |
| LoRA | sparse 训练代理 25.07h + 矩阵评测 4.32h | **29.39h（仅代理）** | 没有任何 LoRA 训练标定，不能冻结 |
| SMoPE，route N=5000 | sparse 25.07h + dense 0.69h + route 1.09h + eval 4.32h | **31.17h** | N=5000 未通过 full-reference 口径 |
| SMoPE，route N=10000 | sparse 25.07h + dense 0.69h + route 2.14h + eval 4.32h | **32.22h** | 10000-vs-full 通过，但路由器状态有偏差 |
| SMoPE，全量 route | sparse 25.07h + dense 0.69h + route 约15.3h + eval 4.32h | **约45.4h** | 最保守，不满足一天目标 |

两张 H100 分别跑 LoRA 与 SMoPE 时，zero-shot 可排在较短的 LoRA 卡上，故当前一 seed 最优
调度的中心墙钟约 **32.2h**，不是三者相加；单卡串行则约 **62.7h**。三 seed、两卡的理论
下界约 94h（3.9 天），实际应留 4–5 天。SMoPE 的 32.2h 已经是上述“一 seed 中心估算”的
主导项；它不会在这个数字之外再额外加一次 32h，但若最终必须全量 route，则会升到约45h。

上述中心值仍不是正式 ETA：它把 GQA/ScienceQA 吞吐外推到八任务，且 LoRA 用 SMoPE sparse
冒充。当前应给 **30–40h/seed 的工作区间**，不能承诺 24h。

#### 10.6.3 瓶颈与能否压到 24 小时

按 N=10000 的 32.22h 代理分解：sparse 训练约 25.07h（77.8%）、评测 4.32h（13.4%）、
route 2.14h（6.6%）、dense 0.69h（2.2%）。评测/route 的 batch>1 都已因语义不等价被否决，
增大 global batch 又不会减少样本前后向。在本节原先假定的**单卡执行口径**下，能覆盖缺口的
首要方向是训练输入流水线与 GPU 计算重叠；§10.8 进一步纠正了 DDP 吞吐口径。若其他项不变，
要达到 24h，sparse 必须从 0.158 降到不高于
**0.106 s/样本（再快约32.8%）**。

现有证据说明这个目标“有可能，但未证明”：过滤侧测得 HTTP+模板+processor 约 0.103s/样本，
而 noGC sparse 总计 0.158s/样本；T4→T8 几乎不提速也符合按样本串行输入占比较大的形态。
full 的 4 个 worker 若能把输入隐藏在 GPU 计算后面，可能一次覆盖 33% 缺口。反过来，如果
workers=4 后仍高于 0.106s/样本，就不能靠 total/global batch 或 optimizer 调参解决。

代码现增加无同步扰动的 `input_wait_seconds` 与 `sequence_widths` epoch 字段；前者记录阻塞在
DataLoader 的墙钟，后者揭示 padding/桶宽构成。下一轮只跑以下三个门禁，不启动正式 full：

1. **P1（最关键）**：SMoPE、noGC、batch=4、GQA 8000，先 workers=4；以已有 workers=0
   0.158s/样本作保守参照，并读取 `input_wait_seconds/train_sparse.seconds`。若总吞吐达到
   ≤0.106s/样本，24h 目标在算术上成立；否则进入 P2。
2. **P2（条件执行）**：在 P1 同配置下只把物理 batch 改为 8（global batch 仍16，accumulation
   自动为2）。此前 T8 是 GC+workers=0 口径，不能代表 noGC+流水线重叠后的扩展性。要求峰值
   <72GiB、样本/有限值正常且比 P1 再快至少15%。
3. **P3（必须）**：用 P1/P2 获胜设置跑 LoRA GQA 8000，得到 LoRA 自己的 s/样本与显存；
   禁止继续用 SMoPE 代理。随后按 §12.2 的八任务各8000条标定逐任务重算最终 ETA；早期2000条
   建议因固定启动成本在线性外推中被过度放大而废止。

若 P2 后仍达不到 0.106s/样本，再做代码级 A/B，而不是启动 full：当前 SMoPE
`expert_attention` 把普通 Q/K 显式转成 FP32 后矩阵乘；Transformers 官方 eager 路径是原生
dtype matmul、仅 softmax 用 FP32。这是最具体的 GPU 优化候选，但会改变数值路径，必须以固定
样本比较 loss、梯度有限性、路由选择和吞吐后才能采用。另一个候选是基于实际
`sequence_widths` 做确定性的长度分组；它会改变训练顺序，属于研究协议变更，优先级低于前者。

### 10.7 论文主实验口径：基线与 route 版本决策

导师提出的“与原始 Qwen3.5-9B 比较是否掉点”和“SMoPE 是否有效”是两个不同问题，不能只用
一个对照回答：

| 运行 | 回答的问题 | 论文角色 |
|---|---|---|
| frozen Qwen3.5-9B zero-shot | 加入持续学习后，相对基础模型的同一固定评测集性能升降 | 必需基础线；无训练，只跑一次 |
| sequential LoRA | 在同样顺序、训练样本和评测预算下，SMoPE 是否优于普通参数高效持续微调 | 必需主要基线；与 SMoPE 使用相同训练 seed |
| SMoPE | 是否提高 New.ACC/MAA/BWT、降低 forgetting，并说明额外时间/显存/参数代价 | 论文主方法 |

因此论文至少报告三者；只比较 zero-shot 不能证明 SMoPE 优于持续微调，只比较 LoRA 又不能回答
基础模型能力是否下降。zero-shot 是确定性基线，固定评测 ID 后无需机械重复三个 seed；LoRA 与
SMoPE 建议三个 seed 报 mean±std。若只跑一个训练 seed，只能称为 pilot，不应包装成稳定结论。

主表必须统一：任务顺序、过滤后的训练清单、eval N=1000 与 seed=42、batch=1 生成、最大长度、
视觉预算和 scorer。报告每任务 zero-shot、最终 accuracy、MAA、New.ACC、BWT、forgetting，另列
训练 GPU-hours、峰值显存和可训练参数。SMoPE 第一任务有 dense+sparse 两次遍历，而 LoRA 只有
一次，必须在 compute/sample exposure 栏明说；若增益很小，应补匹配计算量的 LoRA 控制，不能
把多一次遍历带来的收益全部归因于框架。

**route 的论文决策：主结果用 full，N=10000 只作效率消融。**原因不是 99.75% 是否看起来够高，
而是当前 `frequency` 跨任务累加原始计数：full 下 OCRVQA 165,294 条对累计频率的权重约为
ScienceQA 12,726 条的 13 倍；固定每任务 N=10000 会把任务贡献近似等权，属于算法口径变化。
单任务、未训练路由器上的 protected 一致率无法验证这个跨任务差异。

若最终受算力限制必须把 N=10000 作为主方法，应明确命名为 sampled/均衡版 SMoPE，披露 N 与
seed，并至少在一个**训练后、跨任务累计状态**上与 full 比较 protected 集及最终指标；否则不能
声称它就是 full SMoPE 的无损提速。§10.5 的 full route 约15.3h同样来自 workers=0 驱动脚本，
不是最终成本；生产 full 的 workers=4、batch=1 route 必须先实测，优先用输入预取加速 exact
full，而不是先改算法权重。

### 10.8 双卡吞吐口径纠正与恢复 DDP 的执行门禁（2026-10-15）

#### 10.8.1 原结论错在哪里

§8.7 的 `ratio_to_T1` 比较的是“每个 rank 跑完 25 个 microbatch 的墙钟”，不是全局有用
样本吞吐。A 在 58.486s 处理 `25×4=100` 条，D 的两个 rank 各在约72.76s处理100条且样本
不相交，所以：

```text
A throughput = 100 / 58.486 = 1.710 sample/s
D throughput = 200 / 72.757 = 2.749 sample/s
D/A          = 1.608×
```

因此现有证据不是“双卡反而更慢”，而是“双卡有用吞吐快约61%但冷启动/锁步使扩展效率只有
80.4%”。真正未回答的是：A 用 `batch4×acc2×world1=8`，D 用
`batch4×acc2×world2=16`；它们没有保持相同有效全局 batch 和 optimizer update，不能直接作为
论文训练协议的加速结论。

#### 10.8.2 一天目标的意义

若 1.608× 能在协议匹配的长标定中保持：

- 只把 25.07h sparse 训练并行化，N=10000 的保守估计为
  `25.07/1.608 + 0.69 + 2.14 + 4.32 = 22.74h`；已低于一天。
- 若 dense、route 和 eval 的现有分片也取得相同扩展，N=10000 的理想估计约
  `32.22/1.608 = 20.04h`。
- full-route 即使所有阶段同样 1.608×，仍约 `45.4/1.608 = 28.2h`；所以双卡本身不足以保证
  full-route 一天内完成，仍需 workers=4 的实测收益或接近2×的扩展。

这些是外推，不是新实测。DDP 会缩短单 run 墙钟，但通常不会降低 GPU-hours；论文需同时报告
墙钟与 GPU-hours。

#### 10.8.3 P0：先完成严格 E/F（约十分钟级，不启动 full）

沿用 §9.4 命令：E 单卡为 batch=4、accumulation=4，F 双卡每 rank batch=4、
accumulation=2；两者 global batch=16、全局样本1600、optimizer update=100。完成后运行新增的
自动审计：

```bash
python -m qwen_smope.compare_ddp \
  --single outputs/coin-qwen3.5-9b/verify-20261015b/ddp/E-single-gbs16 \
  --dual outputs/coin-qwen3.5-9b/verify-20261015b/ddp/F-ddp-gbs16 \
  --min-speedup 1.5 \
  --output outputs/coin-qwen3.5-9b/verify-20261015b/ddp/ef-comparison.json
```

分析器同时检查：global batch、optimizer update、两边全局 sample ID 集合、DDP rank 间重复、
非有限参数和错误记录；只有协议完全可比且有效吞吐至少 **1.5×** 才通过。1.5×不是为了数字
好看，而是给 32.2h→24h 留出必要余量。旧 A/D 因 trace schema 较旧且 global batch 不同，
只作机制证据，不应伪装成通过本门禁。

#### 10.8.4 P1-DDP：再用真实训练入口验证生产收益

E/F 通过后，在独占两卡上对同一 GQA 前8000个过滤后样本跑生产单/双卡；除 world size 和由
CLI 自动推导的 accumulation 外不改任何变量。先单卡，回传完成后再双卡：

```bash
CUDA_VISIBLE_DEVICES=0 NPROC_PER_NODE=1 \
OUTPUT_ROOT=outputs/coin-qwen3.5-9b/verify-20261015b/prod-single-gbs16 \
bash scripts/qwen/run.sh smope smoke \
  --tasks GQA --max-train-samples 8000 \
  --batch-size 4 --global-batch-size 16 \
  --workers 4 \
  --eval-batch-size 1 --max-eval-samples 4 \
  --routing-scan-samples 1 --routing-scan-batch-size 1
```

```bash
CUDA_VISIBLE_DEVICES=0,1 NPROC_PER_NODE=2 \
OUTPUT_ROOT=outputs/coin-qwen3.5-9b/verify-20261015b/prod-ddp-gbs16 \
bash scripts/qwen/run.sh smope smoke \
  --tasks GQA --max-train-samples 8000 \
  --batch-size 4 --global-batch-size 16 \
  --workers 4 \
  --eval-batch-size 1 --max-eval-samples 4 \
  --routing-scan-samples 1 --routing-scan-batch-size 1
```

生产门禁：两边 epoch 事件的 `samples_with_ddp_padding=8000`、有效 global batch=16、
`optimizer_updates=500`，loss有限、checkpoint 可加载；双卡 `train_sparse.seconds` 至少快1.5×。
同时比较新记录的
`input_wait_seconds` 和 `sequence_widths`。若 E/F 通过但生产未达1.5×，先用逐 rank 宽度顺序
判断是否为错位 autotune；只有该证据成立才实现同序桶宽预热，不能先开 static graph、调 DDP
bucket 或升级依赖碰运气。

#### 10.8.5 正式调度决策

- 若 P0 与 P1-DDP 均通过：单个 SMoPE seed 改用双卡 DDP，目标区间先按 N=10000
  **约20–23h**；实际 ETA 按 §12.2 的八任务各8000条获胜配置复算。`run.sh` 暂不静默修改默认值，正式
  命令必须显式 `NPROC_PER_NODE=2`，避免历史脚本无意改变硬件预算。
- 若加速 <1.5×：双卡仍可能缩短单 run，但为整个论文矩阵，两张卡各跑 LoRA/SMoPE 或不同 seed
  通常有更高作业吞吐。不要为了“用了 DDP”牺牲总 makespan。
- 论文主方法仍按 §10.7 用 full route；DDP 只改变执行并行度，不解决 N=10000 与 full 的算法
  口径差异。若 full route 一天目标不成立，应诚实报告更长时间，或把 sampled 版作为明确命名的
  效率变体，不能偷偷替换主方法。

## 11. 服务器执行记录（2026-10-15，`b37bb09`）：E/F 门禁与 P1-DDP

> 服务器按 `git pull origin v2` 后执行（快进 `e907170→b37bb09`，无冲突；36/36 单测、
> compileall 通过；拉开前两卡 0MiB）。命令与 §9.4/§10.8 逐字一致，产物在
> `outputs/coin-qwen3.5-9b/verify-20261015b/`（GPU0 与 GPU0+1 前后 nvidia-smi、日志、
> git HEAD 快照齐全）。

### 11.1 P0 严格 E/F（§10.8.3）

| 运行 | 变体 | 每 rank 样本 | optimizer updates | 有效 global batch | 墙钟 | 非有限可训练参数 |
|---|---|---:|---:|---:|---:|---:|
| E | single（batch4×acc4×1） | 1600 | 100 | 16 | **388.39s** | 0 |
| F | ddp-sharded（batch4×acc2×2） | 800（rank0/rank1） | 100 | 16 | **207.72s**（最慢 rank） | 0 |

E 各 rank 宽度 `[384,512,640,768,896,1024]`；F 两 rank 分别 `[384…1024]` 与 `[384…896]`
（分片顺序差异，量级小）。两 rank 样本 ID 并集=1600 且与 E 完全一致、无重复。

`python -m qwen_smope.compare_ddp`（输出 `ddp/ef-comparison.json`）：

```text
protocol_comparable = true          issues = []
useful_sample_throughput_speedup = 1.870  (single 4.120 条/s -> dual 7.703 条/s)
passes_speedup_gate = true          (min 1.5)
```

**P0 通过** → 按 §10.8.4 进入 P1-DDP 生产对照（完成，见 §11.2）。

### 11.2 P1-DDP 生产对照（GQA 8000，§10.8.4）

命令与 §10.8.4/`QWEN_SERVER.md` 逐字一致：唯一变量 world size（accumulation 由 CLI 推导
4→2），`--no-gradient-checkpointing --workers 4`，两卡独占。先单卡，完成后按原参数切双卡。

| 指标 | 单卡（gbs16） | 双卡 DDP（gbs16） | 判据 | 结果 |
|---|---:|---:|---|---|
| train_dense | 1095.85s | 620.23s | — | 1.767× |
| train_sparse | 1054.68s | **572.66s** | ≥1.5× | **1.842×，通过** |
| 整体墙钟 | 2224.77s | 1266.42s | — | 1.757× |
| epoch 样本数 | 8000+8000 | 8000+8000 | =8000 | 通过 |
| optimizer updates | 500+500 | 500+500 | =500 | 通过 |
| 有效 global batch | 16 | 16 | 16 | 通过 |
| loss（dense/sparse） | 0.1288 / 0.1424 | 0.1289 / 0.1427 | 有限、语义一致 | 通过（物理 batch 排布不同，非逐位相同） |
| input_wait（dense/sparse） | 12.22s / 0.72s | 0.44s / 0.44s | — | 双卡输入等待被 workers=4 完全隐藏 |
| 宽度分布 | 384–1024（1024 上限触及） | 384–1024，两 rank 同集合 | — | 未观察到量级性错位 |
| 峰值 allocated | 51.00 GiB | 51.00 GiB | <72 GiB | 通过（每 rank 物理 batch 相同） |
| checkpoint 可加载 | — | `--resume` 42.2s：加载→签名校验→`next_task=1`→收尾 | 通过 | 通过 |

**P1 通过**。按 §10.8.5：**单个 SMoPE seed 的正式运行采用双卡 DDP，命令中显式
`NPROC_PER_NODE=2`**（`run.sh` 默认值保持 1，避免历史脚本无意改变硬件预算）。原始产物：
`verify-20261015b/prod-{single,ddp}-gbs16/`（events、predictions、latest_adapter.pt、
summary）、`prod-{single,ddp}-run.log`、`prod-ddp-resume-check.log`、前后 nvidia-smi。

**遗留（交由维护者）**：§10.8.5 的“八任务各 2000 条获胜配置”ETA 复算未启动——它属于正式
实验排程决定，需维护者确认任务/样本预算后再执行；本轮只完成 E/F 与 GQA 8000 两级门禁。
route 主方法口径（§10.7：full vs sampled）与 LoRA 侧是否也启用 DDP 同样待维护者裁定
（本轮门禁只覆盖 SMoPE 的 DDP 恢复）。

## 12. 维护者冻结的下一步：全量实验提速与开跑门禁（2026-10-15）

本节取代 §11.2 末尾“交由维护者”的未决状态，是服务器 agent 当前唯一执行单。严格按顺序执行，
一次只运行一条命令并先回传上一项产物；在 §12.1 和 §12.2 完成前不得启动三 seed full。

### 12.1 已经不需要再验证的事项

- 不重跑 P0 E/F 或 P1-DDP：§11 已证明相同 global batch、相同样本和相同 update 下，SMoPE
  sparse 双卡加速 1.842×，checkpoint/resume 正常。
- 不再单独验证“卡是否装得下”：P1 已实际覆盖宽度 1024，每 rank 峰值 allocated 51.00 GiB，
  距 72 GiB 门禁仍有约21 GiB；route/eval 没有 backward，显存不会高于训练主路径。
- 不实现同序预热、static graph、DDP bucket 调参或依赖升级：P1 未出现量级性宽度错位。
- 不增大 global batch。正式训练保持16；增大它会改变 optimizer update 数和优化轨迹，不是无语义
  代价的提速。

`full` 模式虽会默认选择 `workers=4`，正式命令仍必须显式写出以便审计。双卡不会自动启用，必须
设置 `NPROC_PER_NODE=2`；`--no-gradient-checkpointing` 也不是默认值，必须显式设置。双卡时每
rank 4 个 DataLoader worker，共8个 HTTP 客户端，低于已测得的 WebDAV 并发上限14；运行期间
禁止另开过滤扫描或其他大量 HTTP 取图作业。

### 12.2 P2：八任务各 8000 条的正式路径 ETA 标定（下一条命令）

目的仅是测量八任务异质性、DDP route 和持续评测的真实墙钟，**不是论文准确率结果**。训练清单
仍由完整 split 先过滤后再取固定前缀；`max_train_samples=8000` 只用于标定，不能替代 full。

这里明确推翻 §10.8.5/§11.2 中早期的“每任务2000条”建议：每次 DataLoader/FLA 宽度有固定启动
成本，若用2000条 stage 总时间线性外推 OCRVQA，会把一次固定成本错误放大约83倍。8000条把该
偏差压到约3–5%，且复用 P1 的样本尺度；多付约数小时的一次性标定，低于错误启动24小时 full
的代价。

先确认两卡空闲、`/dev/shm/qwen35-9b` 完整、8份 `_filter_cache` 未改变，然后执行：

```bash
CUDA_VISIBLE_DEVICES=0,1 \
NPROC_PER_NODE=2 \
SEEDS=0 \
OUTPUT_ROOT=outputs/coin-qwen3.5-9b/verify-20261015c/eta-8x8000 \
bash scripts/qwen/run.sh smope full \
  --tasks ScienceQA,TextVQA,ImageNet,GQA,VizWiz,Grounding,VQAv2,OCRVQA \
  --max-train-samples 8000 \
  --batch-size 4 --global-batch-size 16 \
  --no-gradient-checkpointing --workers 4 \
  --routing-scan-samples 8000 --routing-scan-batch-size 1 \
  --eval-batch-size 1 --max-eval-samples 200 --eval-sample-seed 42
```

预期是约2–4小时，但预期时间不是通过条件。必须从
`eta-8x8000/smope/full/seed-0/events.jsonl` 和 manifest 逐任务填写下表：

| task | valid full N | sparse s/8000 | sparse s/sample | route s/8000 | route sample/s | max input-wait/train | max width | peak GiB |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| ScienceQA | 12,726 | 待填 | 待填 | 待填 | 待填 | 待填 | 待填 | 待填 |
| TextVQA | 34,602 | 待填 | 待填 | 待填 | 待填 | 待填 | 待填 | 待填 |
| ImageNet | 129,833 | 待填 | 待填 | 待填 | 待填 | 待填 | 待填 | 待填 |
| GQA | 69,535 | 待填 | 待填 | 待填 | 待填 | 待填 | 待填 | 待填 |
| VizWiz | 20,523 | 待填 | 待填 | 待填 | 待填 | 待填 | 待填 | 待填 |
| Grounding | 55,213 | 待填 | 待填 | 待填 | 待填 | 待填 | 待填 | 待填 |
| VQAv2 | 82,534 | 待填 | 待填 | 待填 | 待填 | 待填 | 待填 | 待填 |
| OCRVQA | 165,294 | 待填 | 待填 | 待填 | 待填 | 待填 | 待填 | 待填 |

ETA 必须按任务加权，禁止再用 GQA 或八任务简单平均冒充 full：

```text
T_sparse = Σ_task (sparse_seconds_task / 8000) × valid_full_N_task
T_dense  = (ScienceQA_dense_seconds / 8000) × 12,726
T_route  = Σ_task (route_seconds_task / 8000) × valid_full_N_task
T_eval_upper = 5 × Σ_36_cells evaluate_seconds_at_N200
T_fixed  = calibration_wall - Σ(dense/sparse/route/evaluate stage seconds)
T_full_upper = T_sparse + T_dense + T_route + T_eval_upper + T_fixed
```

`T_eval_upper` 把 N=200 的整个 stage（包括 worker 启动）直接乘5，故是偏保守上界；不要反过来
删掉启动成本来制造低 ETA。route 的加权全局吞吐同时报告为 `570260 / T_route_seconds`。

P2 必须同时满足以下硬判据：

1. 8个任务全部完成；每个 sparse epoch 为8000条、500次 update、global batch=16；ScienceQA
   dense 同样为8000条/500次 update；所有 loss/gradient 均有限，checkpoint 可加载。
2. 每 rank 峰值 allocated `<72 GiB`，全部观测宽度 `<=1024`；任何 OOM 都判失败，不准临时缩短
   max length 或视觉预算。
3. 每任务 `input_wait_seconds / train_sparse.seconds <=10%`。超过10%先排查 HTTP/worker/并发，
   不把数据等待误报为 GPU 算力瓶颈。
4. 加权 route 全局吞吐 `>=18 sample/s`。这是让 full route 控制在约8.8小时内、给24小时总预算
   留余量的预注册门槛；route batch 始终为1，不得改回已证伪的 batch=4。
5. `T_full_upper <=23h`：直接授权正式 seed-0（至少保留1小时运行波动）；
   `23h < T_full_upper <=24h`：标为边界，只补一次
   `max_eval_samples=1000` 的双卡 zero-shot 评测以替换保守 `T_eval_upper` 后重算；`>24h`：不得
   宣称一天内完成，也不得静默把论文主方法改成 N=10000。

若某一任务是明显离群点，先核对其宽度分布与 input wait。只有 input wait 正常而 GPU 计算仍慢，
才进入 `expert_attention` BF16 matmul 的固定样本数值/吞吐 A/B；P2 前禁止动该数值路径。

### 12.3 P3：LoRA 自身的双卡门禁（P2 回传并判定后执行）

SMoPE 的1.842×不能代替 LoRA 实测。使用同一 GQA 8000、global batch16，先单卡、后双卡：

```bash
CUDA_VISIBLE_DEVICES=0 NPROC_PER_NODE=1 \
OUTPUT_ROOT=outputs/coin-qwen3.5-9b/verify-20261015c/lora-single-gbs16 \
bash scripts/qwen/run.sh lora smoke \
  --tasks GQA --max-train-samples 8000 \
  --batch-size 4 --global-batch-size 16 \
  --workers 4 \
  --eval-batch-size 1 --max-eval-samples 4
```

```bash
CUDA_VISIBLE_DEVICES=0,1 NPROC_PER_NODE=2 \
OUTPUT_ROOT=outputs/coin-qwen3.5-9b/verify-20261015c/lora-ddp-gbs16 \
bash scripts/qwen/run.sh lora smoke \
  --tasks GQA --max-train-samples 8000 \
  --batch-size 4 --global-batch-size 16 \
  --workers 4 \
  --eval-batch-size 1 --max-eval-samples 4
```

两边必须为8000条、500次 update、global batch16、loss有限且 checkpoint 可加载。双卡
`train_lora.seconds` 加速 `>=1.5×` 才给 LoRA 正式 run 使用双卡；否则 LoRA 保持单卡，并与其他
单卡作业并行排程。禁止继续用 SMoPE sparse 时间代理 LoRA ETA。上面的命令以 §12.7 实测为准，
已经删除会导致约75.6 GiB 占用并 OOM 的 `--no-gradient-checkpointing`。

### 12.4 正式命令（只有对应门禁通过后才能执行）

论文主方法已经冻结为 **exact full route**；`routing_scan_samples=0` 表示扫描每个任务全部过滤后
训练样本。`routing_scan_batch_size=1` 是语义要求。先只跑 seed-0；其结果完整、指标和恢复文件
正常后，再分别启动 seed-1/2，不使用一条长 shell 循环掩盖单 seed 失败。

```bash
CUDA_VISIBLE_DEVICES=0,1 \
NPROC_PER_NODE=2 \
SEEDS=0 \
OUTPUT_ROOT=outputs/coin-qwen3.5-9b/formal-full-route \
bash scripts/qwen/run.sh smope full \
  --tasks ScienceQA,TextVQA,ImageNet,GQA,VizWiz,Grounding,VQAv2,OCRVQA \
  --max-train-samples 0 \
  --batch-size 4 --global-batch-size 16 \
  --no-gradient-checkpointing --workers 4 \
  --routing-scan-samples 0 --routing-scan-batch-size 1 \
  --eval-batch-size 1 --max-eval-samples 1000 --eval-sample-seed 42
```

中断后只允许在**完全相同命令**末尾追加 `--resume`。不要修改 world size、任务顺序、seed、训练
预算、route N、评测 N、长度或视觉预算后复用 checkpoint。

LoRA 仅在 §12.3 双卡门禁通过后使用对应的双卡 full 命令；zero-shot 无训练，只运行一次，但可
使用 `NPROC_PER_NODE=2` 通过 `ExactShard` 并行固定 N=1000 评测。论文仍报告 zero-shot、LoRA、
SMoPE 三组；zero-shot 不能替代 LoRA。

zero-shot（只运行 seed-0 一次）：

```bash
CUDA_VISIBLE_DEVICES=0,1 \
NPROC_PER_NODE=2 \
SEEDS=0 \
OUTPUT_ROOT=outputs/coin-qwen3.5-9b/formal-zero-shot \
bash scripts/qwen/run.sh zero_shot full \
  --tasks ScienceQA,TextVQA,ImageNet,GQA,VizWiz,Grounding,VQAv2,OCRVQA \
  --workers 4 \
  --eval-batch-size 1 --max-eval-samples 1000 --eval-sample-seed 42
```

LoRA（仅当 §12.3 双卡门禁通过；先 seed-0，完成后分别改为1/2）：

```bash
python -c "import peft; assert peft.__version__ == '0.18.1'; print('peft', peft.__version__)"

CUDA_VISIBLE_DEVICES=0,1 \
NPROC_PER_NODE=2 \
SEEDS=0 \
OUTPUT_ROOT=outputs/coin-qwen3.5-9b/formal-lora \
bash scripts/qwen/run.sh lora full \
  --tasks ScienceQA,TextVQA,ImageNet,GQA,VizWiz,Grounding,VQAv2,OCRVQA \
  --max-train-samples 0 \
  --batch-size 4 --global-batch-size 16 \
  --workers 4 \
  --eval-batch-size 1 --max-eval-samples 1000 --eval-sample-seed 42
```

若 §12.3 双卡未达到1.5×，不得使用上面的 LoRA 双卡命令；改回 `NPROC_PER_NODE=1`，其余参数
保持不变，并将 LoRA 与另一个单卡作业并行调度。不要用 `run_all.sh`：三种方法的 seed 数、route
行为和双卡门禁不同，一条统一循环会掩盖错误配置。

### 12.5 每步必须回传和写回本文的产物

- 完整输出目录：`environment.json`、`events.jsonl`、`dataset_manifest.json`、`summary.json`、
  `accuracy_matrix.csv`、`routing/*.json`、`predictions/**`、`latest_adapter.pt`。
- stdout/stderr 日志、运行前后 `nvidia-smi`、准确 Git SHA、开始/结束时间；不得只贴汇总表。
- P2 的逐任务原始表、上述五项 ETA 分量、加权 route sample/s、最终门禁状态与计算脚本/命令。
- P3 的单/双卡样本、update、loss、显存、阶段墙钟、加速比及 resume 结果。
- 所有计划项先标“待验证”；只有原始产物齐全后才改为“已完成”。失败时保留目录，不覆盖、不删
  cache，并在本文写明失败点和下一分支。

## 12.6 服务器执行记录：P2 首次执行（2026-10-15，`0a5643a`）——数据与死锁失败分支

> 按 §12.2 逐字执行；两卡独占；产物在 `outputs/coin-qwen3.5-9b/verify-20261015c/`。
> **P2 状态：未完成（8/8 任务训练+route 完成，35/36 评测 cell 完成；最后一个 OCRVQA
> 评测 cell 期间 DDP 死锁，任务 8 的 checkpoint 与 run_complete 未产生）。**
> 正式 seed-0 **不启动**：见 12.6.3 的悬置风险。

### 12.6.1 逐任务实测（8×8000 标定，命令见 §12.2）

| task | sparse_s | sparse s/样本 | route_s | route 条/s | input_wait | 最大宽度 | 峰值 GiB |
|---|---:|---:|---:|---:|---:|---:|---:|
| ScienceQA | 269.28 | 0.0337 | 331.32 | 24.15 | 0.13% | 640 | 37.78 |
| TextVQA | 362.18 | 0.0453 | 384.37 | 20.81 | 0.41% | 768 | 42.06 |
| ImageNet | 272.14 | 0.0340 | 372.96 | 21.45 | 0.53% | 384 | 29.44 |
| GQA | 587.99 | 0.0735 | 389.28 | 20.55 | 0.20% | 1024 | 50.95 |
| VizWiz | 313.53 | 0.0392 | 379.08 | 21.10 | 0.44% | 384 | 29.44 |
| Grounding | 522.33 | 0.0653 | 377.74 | 21.18 | 0.17% | 1024 | 50.95 |
| VQAv2 | 385.28 | 0.0482 | 374.83 | 21.34 | 0.27% | 1024 | 50.95 |
| OCRVQA | 329.79 | 0.0412 | 373.12 | 21.44 | 0.34% | 512 | 33.55 |

ScienceQA dense：330.40s（0.0413 s/样本）。9 个 epoch（1 dense + 8 sparse）全部满足
`samples_with_ddp_padding=8000`、`optimizer_updates=500`、global batch 16，loss/grad 全部有限。

**已完成部分的五判据**：

1. 8 任务完成/样本/update/有限性：**全部通过**（除任务 8 的 checkpoint 未写出，见 12.6.2）。
2. 峰值 allocated **50.95 GiB < 72**，全部宽度 **≤1024**：通过。
3. `input_wait/train_sparse` 最大 **0.53% ≤ 10%**：通过。
4. **加权 route 全局吞吐 = 570,260 / T_route = 21.29 条/s ≥ 18：通过**
   （与驱动脚本单进程 ~10.5 条/s × 2 rank 分片一致，互验成立）。
5. T_full_upper：**仅能给出临时值**（缺 1 个 eval cell 与完整 wall）：
   T_sparse=7.422h、T_dense=0.146h、T_route=7.439h（加权公式）、
   Σ35/36 eval cells=1310.6s；实测 T_fixed ≈ 74s（wall 2.15h − 各 stage 合计 2.13h）。
   若缺失的 OCRVQA cell 估计为 200–600s，T_eval_upper=5×Σ≈2.10–2.65h，
   **临时 T_full_upper ≈ 17.1–17.7h**（< 23h）。**这是临时值：cell 缺失 + wall 含死锁
   悬挂时间被剔除，不能替代完整 P2。**

### 12.6.2 失败点：DDP 评测汇总阶段的死锁（非慢任务）

时间线（UTC）：15:33 拉取/前置检查 → 15:35 启动 → 17:42:41 写出第 7 个 eval cell
（VQAv2.jsonl）→ 之后的 OCRVQA cell 期间死锁 → 18:14 首次 py-spy 取证 → 18:25 终止进程组
（torchrun 父进程 SIGTERM）。**死锁前的 35/36 评测 cell 与全部训练/route 阶段均为同一代码
路径，全部正常。**

现场（证据存档 `verify-20261015c/hang-diagnostics/`：两份 `*-native.txt`、`nvidia-smi-at-hang.txt`、
`events-at-hang.jsonl`、`p2-log-at-hang.txt`）：

- 两个 rank 的 GPU 均 100% 空转、进程 CPU ~96%、超过 40 分钟无任何新事件；RSS 稳定
  （3.66/3.70 GB）不增长。
- **rank 1（pid 1557580）**：卡在 `evaluate_task` 的 `all_gather_object` **前置序列化**内：
  `_object_to_tensor` → `torch.ByteTensor(...).to(device)` → `cudaStreamSynchronize` 永不返回。
- **rank 0（pid 1557579）**：卡在 stage 退出处 `all_gather_object(memories, local)` 的**尺寸收集**：
  `int(object_size_list[i])` → `_local_scalar_dense_cuda` → `cudaStreamSynchronize` 永不返回。
- **修正与触发事件（2026-10-16 经 §12.9 的标签校验错误证实）**：本节初稿曾按
  `OCRVQA.jsonl` 缺失把两个 pid 的 rank 归属写反。实际是 rank 0 的 OCRVQA 评测分片
  命中缺图 `OCR-VQA/images/1421539896.jpg`（HTTP 404）而**先崩溃**，随后进入 stage 退出
  的内存集合；rank 1 正常完成循环、进入评测汇总集合。旧 NCCL 对象路径下两个不同集合
  静默互等——这就是本次挂死的真正触发链，§12.9.2 在同一触发场景下立即得到
  `Object collective order mismatch` 报错，两处证据一致。
- 分类为**集合通信错配/挂死**（NCCL 内核层自旋、两个 rank 的调用序列不再对齐），
  不是数据/IO/算力问题。

**已排除**：对象过大（用同一数据集复现 rank 分片的 rows，pickle 仅 8 KB）；内存泄漏
（RSS 稳定）；图像服务器劣化（实测 5–21ms，且死锁发生在生成之后的汇总阶段）；显存 OOM
（无当日 OOM/异常）；GPU 硬件错误（dmesg 无当日 Xid）；协议参数（与成功的 35 个 cell
完全同构）。`init_process_group("nccl", timeout=90min)`：默认 async error handling 下，
不手动终止也会在 ~90 分钟处由看门狗中止，但不会自愈。

### 12.6.3 对正式 run 的影响与下一分支（已由 §12.8 裁定）

- 观测事实仅为 37 个已尝试 DDP 评测 cell 中挂死 1 次；`1/37` 不是可靠的真实挂死概率，
  把单次观测按独立同分布外推成“36 cell 至少一次约60%”没有统计依据，不能写进论文。
  但一次可复现路径上的无限挂起已经足以构成正式 run 的工程阻断。**因此 P2 虽未通过，
  但不能把它当作偶然；在挂死被修复/缓解前不启动正式 seed-0。**
- 可选缓解（维护者裁定，服务器不擅自改协议/代码）：
  1. **对象汇总改用 gloo 进程组**（`all_gather_object(..., group=gloo_pg)`，gloo 原生支持
     对象，无 CUDA 张量中转；语义不变，改动小、属于工程修复）；
  2. **fail-fast**：缩短 NCCL 看门狗超时 + 保留 async error handling，让挂死在 N 分钟内
     报错并用完全相同命令 `--resume`（checkpoint 粒度为任务边界）；
  3. **评测 cell 改单 rank 汇总**（`ExactShard` 拆数据、rank0 收集；语义不变、墙钟变慢）；
  4. 不改代码直接重跑（消耗 3h，且风险未消除）。
- 若决定先复核根因：下一次尝试建议加 `NCCL_DEBUG=WARN`（或 INFO）并让其自然触发看门狗，
  保留错误消息后再杀——本轮人工终止早于 90 分钟超时，未取得看门狗文本。
- 任务 8 的 checkpoint 未写出（`latest_adapter.pt` 停留在 17:25 的任务 7 边界）；
  P1 已验证过 checkpoint/resume 机制本身。**目录保留，未覆盖，未删 cache。**

## 12.7 服务器执行记录：P3 LoRA 双卡门禁（2026-10-15，`0a5643a`）——通过 + 两处环境修正

> 按 §12.3 执行。**结论：LoRA 双卡 `train_lora` 加速 1.782× ≥ 1.5，P3 通过**；但执行中发现
> 两处与文档不符的环境事实，均按纪律保留失败目录并记录偏差。

### 12.7.1 两处修正（影响 §12.3/§12.4 的命令文本）

1. **venv 缺 `peft`**：§12.3 首次执行在 `load_model` 处 `ModuleNotFoundError: No module
   named 'peft'`（此前从未跑过 LoRA）。已按 `requirements-qwen.txt` 的 pin 安装
   `peft==0.18.1`（`pip install --no-deps`，其余依赖已在位且版本符合 pin）；导入与
   LoRA 模型构建验证通过。失败目录保留为 `lora-single-gbs16.failed-peft-missing/`。
2. **LoRA + `--no-gradient-checkpointing` 在 batch 4 必然 OOM**：
   `torch.OutOfMemoryError ... Tried to allocate 3.79 GiB. GPU 0 ... 75.63 GiB in use,
   71.91 GiB allocated by PyTorch`（容量 79.18 GiB）。原因是机制性差异：SMoPE 的可训练
   参数只在 8 个 full-attention 层的专家/路由器，反向不需要冻结层激活；LoRA 在全部
   transformer 块的线性层上都有适配器，反向需要保存每一处适配层的输入激活。**SMoPE 的
   no-GC 结论不能类推到 LoRA**——§12.3 与 §12.4 的 LoRA 命令按当时文本无法运行。
   失败目录保留为 `lora-single-gbs16.failed-noGC-oom/`。
   **本轮处置**：LoRA 门禁改用唯一可运行配置（**gradient checkpointing 保持默认开启**，
   其余参数与 §12.3 逐字一致），单/双卡同配置对比；维护者已在 §12.8 确认正式 LoRA 删除
   `--no-gradient-checkpointing`，不通过降低物理 batch 保留 no-GC。

### 12.7.2 P3 结果（GQA 8000、gbs 16、GC 开启、workers 4）

| 指标 | 单卡 | 双卡 DDP | 判据 | 结果 |
|---|---:|---:|---|---|
| train_lora | 2294.22s | **1287.32s** | ≥1.5× | **1.782×，通过** |
| epoch 样本 / updates | 8000 / 500 | 8000 / 500 | 一致 | 通过 |
| loss | 0.1026 | 0.1028 | 有限、一致 | 通过 |
| input_wait | 0.62s | 0.52s | — | 通过 |
| 峰值 allocated | 30.73 GiB | 30.89 GiB | <72 GiB | 通过 |
| evaluate_GQA（N=4） | 100.0% | 100.0% | — | 一致 |
| checkpoint `--resume` | — | 通过（签名校验→`next_task=1`→收尾） | 可加载 | 通过 |

**决策（按 §12.3 预设规则）**：LoRA 正式 run 可使用双卡（以 GC 开启为准）；
LoRA 的 ETA 不得再用 SMoPE 时间代理（本轮有了 LoRA 实测：0.287 s/样本单卡、
0.161 s/样本双卡全局）。原始产物：`verify-20261015c/lora-{single,ddp}-gbs16/`、对应
`lora-{single,ddp}-run.log`、`lora-ddp-resume-check.log`、前后 nvidia-smi、两个失败目录。

**注意**：该门禁只回答“LoRA 双卡是否 ≥1.5×”，样本为单任务 GQA 8000；LoRA 的八任务
异质性未测（§12.6.1 的分布是 SMoPE 的）。正式 LoRA full 的 ETA 若需要按任务加权，
应先用 LoRA 版的小规模校准，不能用 SMoPE 分布外推——但 LoRA 无 dense 阶段，且本条的
0.287/0.161 s/样本已可用于粗估。

## 12.8 维护者裁定与唯一下一步（2026-10-16）

### 12.8.1 裁定

1. **死锁方案采用“独立 Gloo 对象组 + 标签校验”，不原样重跑 P2。**默认 NCCL 组继续处理
   loss、计数、route frequency 等 GPU 张量；所有 `all_gather_object`（评测 rows、stage 显存、
   epoch 宽度分布）统一改走 CPU/Gloo，避免 `_object_to_tensor(...).to(cuda)`。每次对象集合带
   语义标签，rank 顺序不一致会显式报错；Gloo 组超时10分钟，避免继续等待90分钟。此项属于通信
   工程修复，不改变数据、训练或指标协议。现场只能证明旧 NCCL/CUDA 对象路径挂住，不能声称
   已唯一定位 PyTorch/NCCL 的内部根因；H100 resume 通过前统一表述为“待验证修复”。
2. **P2 不从头重跑。**保留原目录，从 task7 边界 checkpoint 用原参数加 `--resume`。这会按正确
   协议重做 OCRVQA 的训练/route/末轮8个评测 cell，然后写 task8 checkpoint 与 `run_complete`；
   既验证修复，也避免浪费已完成的前7任务。ETA 汇总只选首次成功的任务1–7 stage 与 resume
   成功的任务8 stage，不能把失败尝试和 resume 的 OCRVQA 重复相加。
3. **LoRA 正式配置确定为双卡、batch=4、global batch=16、GC 开启。**P3 已在这一配置上得到
   1.782×，而 no-GC 已机制性 OOM；降低 batch 来保留 no-GC 没有实测收益，还会引入新变量，
   因而不采用。§12.3/§12.4 已删除 LoRA 的 `--no-gradient-checkpointing`。SMoPE 仍使用已经通过
   显存与吞吐门禁的 no-GC，二者的 GC 设置不同属于实现内存策略，不改变有效 global batch。

### 12.8.2 服务器执行单：只做 P2 resume

先更新并核验环境；必须看到当前 Git SHA、`peft 0.18.1`、两张空闲 H100，以及 checkpoint 的
`next_task=7`。若 checkpoint 不是7或 matrix 不是7行，停止并回传，不猜测继续点。

```bash
cd /mnt/zq/SMOPE
git pull --ff-only origin v2
git rev-parse HEAD
source .venv2/bin/activate
python -c "import peft, torch; print('peft', peft.__version__); p=torch.load('outputs/coin-qwen3.5-9b/verify-20261015c/eta-8x8000/smope/full/seed-0/latest_adapter.pt', map_location='cpu', weights_only=False); print('next_task', p['next_task'], 'matrix_rows', len(p['matrix']))"
nvidia-smi
```

确认输出为 `peft 0.18.1`、`next_task 7`、`matrix_rows 7` 后，执行与 §12.2 相同的命令，仅追加
`--resume`；不要新建输出目录，不要删除旧 events/predictions：

```bash
CUDA_VISIBLE_DEVICES=0,1 \
NPROC_PER_NODE=2 \
SEEDS=0 \
OUTPUT_ROOT=outputs/coin-qwen3.5-9b/verify-20261015c/eta-8x8000 \
bash scripts/qwen/run.sh smope full \
  --tasks ScienceQA,TextVQA,ImageNet,GQA,VizWiz,Grounding,VQAv2,OCRVQA \
  --max-train-samples 8000 \
  --batch-size 4 --global-batch-size 16 \
  --no-gradient-checkpointing --workers 4 \
  --routing-scan-samples 8000 --routing-scan-batch-size 1 \
  --eval-batch-size 1 --max-eval-samples 200 --eval-sample-seed 42 \
  --resume 2>&1 | tee outputs/coin-qwen3.5-9b/verify-20261015c/p2-gloo-resume.log
```

### 12.8.3 通过判据与后续分支

P2 resume 必须同时满足：

1. `resume_environment.json` 记录 `python_object=gloo`、10分钟超时和本次 Git SHA；日志无
   collective order mismatch、Gloo/NCCL timeout、OOM、非有限 loss/gradient。
2. resume 的 OCRVQA epoch 仍为8000样本、500 updates、global batch16，训练/route/8个评测
   cell 全部结束；末尾新增且仅以最后一条 `run_complete` 作为完成证据。
3. `latest_adapter.pt` 为 `next_task=8`、matrix 8行；`after_08_OCRVQA/` 有8个非空 prediction
   JSONL，整个持续评测矩阵仍为36个 cell，checkpoint 能用相同命令再次 `--resume` 秒级收尾。
4. 重新按 §12.2 公式计算五个 ETA 分量：任务1–7使用首次成功 stage，任务8只使用本次 resume
   成功 stage；完整36-cell 的 `T_full_upper <=23h`，加权 route仍 `>=18 sample/s`。回传计算脚本、
   所选 event 行及结果，不能只报最终数字。

四项全过：将 P2 标为完成，**直接启动 §12.4 的 SMoPE exact-full-route seed-0**；不再插入新的
SMoPE 吞吐代理或 LoRA 标定。seed-0 完成且 summary、36-cell predictions、routing、checkpoint
齐全后，再跑 seed-1/2。LoRA 的 P3 已通过，轮到 LoRA 时直接用 §12.4 的 GC-on 双卡正式命令。

若失败则按证据分支：标签不一致说明 rank 控制流确实分叉，回传两 rank 完整 traceback 后再定位；
Gloo 超时但无标签结果说明某 rank 未进入对象集合，保留日志并停止正式 run；若仅 ETA >23h，才按
§12.2 的评测边界分支处理。任何失败都不得原样无限重试，也不得用 `TORCH_DISTRIBUTED_DEBUG=DETAIL`
所得时间做性能比较。

## 12.9 服务器执行记录：P2 Gloo resume（2026-10-16，`75e1c3c`）——修复生效，暴露评测集缺图

> 按 §12.8.2 逐字执行。**结论：Gloo+标签校验修复按设计工作（旧静默挂死 → 20 分钟内明确
> 报错）；但 resume 被一个新确认的数据问题阻断：OCRVQA 评测集缺图，且正式 N=1000 子集
> 恰好命中。P2 仍未完成，正式 seed-0 保持停止。**

### 12.9.1 前置核验（§12.8.2 全项通过）

HEAD=`75e1c3cfc85464a3451e0e4276561b33b7ad8c23`；`peft 0.18.1`；checkpoint
`next_task=7`、`matrix_rows=7`；两卡 0MiB 空闲。

### 12.9.2 resume 执行与结果

与 §12.2 相同命令仅追加 `--resume`（原目录）。时间线（UTC）：04:09 启动 → OCRVQA
`train_sparse:8=346.7s` → `route_scan` → 末轮 8 个评测 cell 中 7 个完成
（ScienceQA 27.1 / TextVQA 32.8 / ImageNet 27.0 / GQA 25.6 / VizWiz 31.0 /
Grounding 141.4 / VQAv2 26.2s，全部经 Gloo 对象组正常汇总并写出）→ OCRVQA cell 的
DataLoader worker 3 取图得到 **HTTP 404（`OCR-VQA/images/1421539896.jpg`）**，eval 循环
抛出 `ImageUnavailableError` → rank 0 进入 stage 退出内存集合，rank 1 已在评测汇总集合 →
**`RuntimeError: Object collective order mismatch at 'stage:7:evaluate_OCRVQA:memory'`**
（rank labels 列出两种不同标签）→ torchrun 以 exit 1 终止。总墙钟约 20 分钟。

- **判据 1 通过**：`resume_environment.json` 记录 `git_commit=75e1c3c`、
  `distributed_collectives={tensor: nccl, python_object: gloo, object_timeout_seconds: 600}`；
  日志无 Gloo/NCCL timeout、OOM、非有限 loss/gradient。
- **修复验证成功**：同一个曾静默挂死 40+ 分钟的触发场景，现在立即给出明确错误与两个
  rank 的完整 traceback（`p2-gloo-resume.log`）；§12.6.2 的 rank 归属已按此证据修正。
- 判据 2/3 不通过：OCRVQA cell 未完成 → 无 task8 checkpoint、无 run_complete；
  `after_08_OCRVQA/` 仍 7 个文件（均为本次 resume 重写）。checkpoint 保持
  `next_task=7`/7 行未动。
- 失败分支（§12.8.3）：“标签不一致 → 回传两 rank 完整 traceback 后再定位”——已定位
  （12.9.3），未做任何原样重试。

### 12.9.3 定位：OCRVQA 评测集缺图（新阻断项）

- curl 复核 `1421539896.jpg` → **404**；OCRVQA test 中 5 条记录（source_index 2467–2471，
  `question_id` `1421539896_0.._4`）引用该图。
- 八任务**评测 split** 缺图普查（12 并发 ≤14 上限，纯 HTTP；产物 `eval-probes/*.json`）：
  ScienceQA test 2,017 张、TextVQA val 5,000、ImageNet test 5,050、GQA test 12,578、
  VizWiz val 4,319、Grounding test 30,969、VQAv2 val 214,354 全部 **0 缺图**；
  **OCRVQA test 99,926 张中 35 条记录缺图（0.035%）**——注意这 35 条只对应
  **7 个不同的图片文件**（OCRVQA 每本书图 5 个问题：7×5=35），清单：
  `1285442962 / 1421539896 / 1558210571 / 1609619919 / 1931271070 / 615754090 /
  857194321`（.jpg）。若选数据修复，实际只需补回 7 个文件。
- **正式子集交集**（`check_eval_subset_missing.py` → `eval-subset-missing.json`）：
  正式 N=1000/seed=42 的 8 个评测子集中，**OCRVQA 含 1 张缺图**（即 1421539896.jpg
  所在记录），其余 7 任务 0。→ **正式 seed-0 会在末轮 OCRVQA cell 确定性崩溃**
  （新代码=快速报错，旧代码=沉默挂死）——这不是概率风险，是已知必现事件。
- 既有的训练侧 54 条缺图剔除（§8.3.2）与本次评测侧 35 张缺图是同一镜像问题的两面；
  评测侧此前从不需要缺图清单，故未建立。

### 12.9.4 待维护者决策（正式 run 保持停止）

1. **OCRVQA 评测缺图处理**（服务器不擅自选择）：
   (a) 补齐镜像中 35 张缺图（数据修复最干净；正式子集强制需要 `1421539896.jpg`，
   其他 N/seed 组合会命中其余）；
   (b) **评测侧确定性剔除**：复用本目录 `eval-probes/OCRVQA-test.json` 的缺图清单，
   对 eval split 做一次确定性缺图剔除并在 `dataset_manifest`/论文披露（需要改代码，
   属维护者权限；注意评测集在所有方法/seed 间必须一致）；
   (c) 其它方案。
2. 处理完成后，用与 §12.8.2 相同的命令 resume 即可继续（checkpoint 完好停在 task 7
   边界）；P2 四项判据全部通过后按 §12.8.3 启动正式 seed-0。
3. 本轮未做（按纪律）：原样重试、改协议参数、动其他任务（其余 7 任务评测集无缺图）。
