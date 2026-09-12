# Qwen 性能排查（2026-09-11）

已核实服务器为 Python 3.11.4、PyTorch 2.5.1+cu121、Triton 3.1.0、
Transformers 5.3.0、双 A100-PCIE-40GB。四项线性注意力内核均缺失。
这与模型加载慢是两个问题，不能用安装内核来保证缩短读权重时间。

## 1. 补充内核，保留已有 PyTorch

在服务器仓库 `/mnt/zq/SMOPE` 中、原 `.venv2` 环境下运行；不需要本次代码修改即可执行。
下面是按官方依赖条件选择的候选版本，尚未在这台 A100 上完成运行验证。
FLA core 提供本项目需要的 ops/modules，不需要额外安装整套 FLA 模型。

```bash
cd /mnt/zq/SMOPE
source .venv2/bin/activate
python -m pip freeze > "env-before-kernels-$(date +%Y%m%d-%H%M%S).txt"
python -m pip install 'einops==0.8.1' ninja packaging setuptools wheel
python -m pip install --no-deps 'fla-core==0.3.2'
# 核查本机 CUDA 编译工具链。torch.version.cuda 不等于已安装 nvcc。
nvcc --version
MAX_JOBS=4 python -m pip install --no-build-isolation --no-deps 'causal-conv1d==1.5.0.post8'
python -m pip check
python -c "from transformers.models.qwen3_5 import modeling_qwen3_5 as m; print('fast_path_available =', m.is_fast_path_available)"
```

若 causal-conv1d 安装失败，保留完整错误；可能涉及预编译 wheel、CUDA Toolkit、
编译器或 ABI 匹配。不要通过随意升级 torch/triton 来碰运气。
缺少 nvcc 时，需要匹配当前 Python/PyTorch/CUDA/ABI 的官方 wheel 或正确的编译工具链。
`--no-deps` 用于防止这两个包隐式升级 torch；`pip check` 用于发现未满足依赖。

先保留 batch=2 以便区分内核安装的影响，再测试 batch=4；有效 batch 都为16。
每次使用新的输出目录，首次 Triton 编译可能使预检耗时增加。

```bash
OUTPUT_ROOT=outputs/qwen-kernels-b2 \
bash scripts/qwen/cifar100.sh smoke smope --batch-size 2 --accumulation 4

OUTPUT_ROOT=outputs/qwen-kernels-b4 \
bash scripts/qwen/cifar100.sh smoke smope --batch-size 4 --accumulation 2
```

安装后必须通过实际图文前向/反向和两任务 smoke，而不只是导入成功。
当前小样本测试不能代表所有图片尺寸都不会 OOM，也不能代表长期稳定吞吐。

## 2. 定位加载瓶颈

原 batch=2 的 model_loading 为223.39秒，其中不含后续预检。
当前入口每个 DDP 进程各自加载一份主干，再从 CPU 搬到自身 GPU。
不能仅据路径 `/mnt/zq` 断言它是网络盘，也不能据总耗时断言一定是磁盘问题。

本次代码会在 events.jsonl 新增 rank 0 的 loading_profile：

- pretrained_cpu_seconds：from_pretrained，包含读取、参数转换及初始化；不是纯磁盘时间。
- adapter_init_seconds：冻结主干和创建专家/分类头。
- move_to_device_seconds：搬到设备并等待完成。
- linear_attention_kernels：四项内核是否可用。

它不包括后面的 DDP 包装/参数广播；原 model_loading 阶段日志仍记录两卡最大总时长。
加载时同时查看以下只读诊断：

```bash
findmnt -T pretrained/Qwen3.5-9B-Base
df -h pretrained/Qwen3.5-9B-Base
free -h
# 若已安装 sysstat，在另一终端观察加载期间磁盘读速率/等待
iostat -xz 1
```

若确定模型位于慢速共享盘、且服务器有足够空间的本地 NVMe，可将完整权重复制到
一个新的本地目录，然后用 MODEL_PATH 指向它；先核实挂载类型、空间和目标路径。
不建议在尚未确认 I/O 瓶颈时盲目增加加载线程；两进程可能进一步争抢同一磁盘。

## 3. 本次代码优化和验证边界

- 旧专家损失从逐头动态布尔索引改为固定尺寸批量矩阵运算，减少 GPU→CPU 同步；
  保留每头按已用 anchor 数量取均值，再对头求和的定义。
- eval/scan 不再计算未使用的路由与旧专家损失；仍正常计算路由和专家选择频次。
- 保留原显式专家注意力和训练数值检查，未直接将专家注意力替换成普通 FlashAttention。

服务器安装依赖和本地代码同步是独立步骤：本地改动没有自动出现在 GitHub 或服务器。
本次没有完整 A100 训练实测，不承诺某个利用率或加速倍数。比较稳定训练阶段的
samples_per_second、峰值显存和正确性；smoke 阶段仅几秒，CPU worker 启动等成本占比高。

本地验证：隔离 Transformers 5.3.0 + PyTorch 2.12.1 CPU 环境下，13项测试全部通过，
包含双进程 Gloo、旧损失数值/梯度等价、eval预测、图文反向、梯度检查点和适配器恢复。
这不是服务器 PyTorch 2.5.1/CUDA 加速内核的兼容性验证。

参考：
- https://pypi.org/project/fla-core/0.3.2/
- https://github.com/fla-org/flash-linear-attention/blob/v0.3.2/README.md
- https://github.com/Dao-AILab/causal-conv1d/tree/v1.5.0.post8
- https://github.com/huggingface/transformers/blob/v5.3.0/src/transformers/models/qwen3_5/modeling_qwen3_5.py
