# Qwen3.5-9B + SMoPE：双 A100 40GB 运行说明

本入口把原 SMoPE 的按头 K/V Prompt 专家迁移到 Qwen3.5 的完整注意力层。
保留原 ViT 代码；Qwen 训练使用独立入口 `qwen_smope.train`。
这是整段图文输入分类，**不是自回归生成，也不是把 Qwen FFN 改成参数 MoE**。

## 1. 环境与文件

推荐 Linux、Python 3.11、两张 A100 40GB。PyTorch CUDA 12.4 wheel 需要兼容的 NVIDIA 驱动；安装前运行 `nvidia-smi`。
请使用独立环境，不在原 ViT 环境安装这些依赖。

```bash
git clone -b v2 https://github.com/Lakunsradwfw/SMOPE1.git
cd SMOPE1
conda create -n smope-qwen python=3.11 -y
conda activate smope-qwen
bash scripts/qwen/setup.sh
```

模型目录默认：

```text
pretrained/Qwen3.5-9B/
  config.json
  model.safetensors.index.json
  model-00001-of-....safetensors  # 所有分片
  tokenizer.json
  tokenizer_config.json
  preprocessor_config.json
  chat_template.jinja            # 或模板存于 tokenizer 配置
  ...                           # 官方仓库其他 processor 配置也一并下载
```

建议下载官方 Hugging Face 完整模型目录；不要只放 GGUF 或纯文本模型。
脚本只读取本地权重，不自动下载、不通过推理 API 训练。
加载时若提示 `lm_head.weight UNEXPECTED`，这是分类入口不使用语言输出头的正常情况；主干缺失参数会报错。
若文件直接位于 `pretrained/`，先执行 `export MODEL_PATH="$PWD/pretrained"`。

数据沿用旧项目格式：

```text
data/cifar-100-python/{train,test,meta}
data/CUB_200_2011/{images/,images.txt,image_class_labels.txt,train_test_split.txt}
data/imagenet-r/<类别目录>/<图片>
```

ImageNet-R 使用仓库 `dataloaders/splits` 的原训练／测试列表。
可通过 `DATA_ROOT=/absolute/data/root` 指定共同数据根目录。
启动时检查缺失图片；不会静默下载或重划分测试集。

## 2. 先验证，再正式运行

以下命令均在仓库根目录执行，每次默认占用两张卡；不要同时启动多个实验争抢显存。

```bash
# 仅检查官方 Processor 的图片/token 预算，不加载主干权重
python scripts/qwen/check_processor.py pretrained/Qwen3.5-9B

# 真实模型加载 + 一个图文批次前向/反向，无优化器更新
bash scripts/qwen/cifar100.sh preflight smope

# 两任务小样本：训练、专家统计、校正、评价、轻量保存
bash scripts/qwen/cifar100.sh smoke smope
bash scripts/qwen/cub200.sh smoke smope
bash scripts/qwen/imagenet-r.sh smoke smope

# 10 个完整增量任务，默认 seed=0
bash scripts/qwen/cifar100.sh full smope
bash scripts/qwen/cub200.sh full smope
bash scripts/qwen/imagenet-r.sh full smope

# 相同类别顺序、相同评价协议的冻结特征对照
bash scripts/qwen/cifar100.sh full head_only
bash scripts/qwen/cub200.sh full head_only
bash scripts/qwen/imagenet-r.sh full head_only
```

每个 full 实验默认每任务 20 个训练 epoch；第一个任务另有 10 个初始化 epoch。
SMoPE 在初始化时使用全部专家，head_only 同期只训练分类头，以保持分类头优化预算一致。
每个任务做 5 个特征校正 epoch；smoke 分别缩至 1，并限制每类 2 张训练／测试图片。
**smoke 分数仅用于功能验证，不能当作正式实验结果。**

```bash
# 多种子依次运行；训练参数直接追加在脚本后
SEEDS="0 1 2" bash scripts/qwen/cifar100.sh full smope --epochs 20

# Top-k 消融使用不同目录，避免覆盖
OUTPUT_ROOT=outputs/qwen-top3 bash scripts/qwen/cifar100.sh full smope --topk 3

# 恢复原实验：同一输出路径、世界大小、种子和训练参数
bash scripts/qwen/cifar100.sh full smope --resume

# 单卡调试。DDP 每卡一份主干，两张 40GB 不等于一张 80GB
NPROC_PER_NODE=1 CUDA_VISIBLE_DEVICES=0 OUTPUT_ROOT=outputs/qwen-single \
  bash scripts/qwen/cifar100.sh smoke smope
```

默认每卡 batch=1、梯度累积 8 次、有效 batch=16；BF16 主干、FP32 专家和分类头，非重入梯度检查点。
图像保持纵横比，最多 256 个合并后的视觉 token；原图片不会被提前按 ViT 方式归一化。
OOM 时减小 `--max-visual-tokens 128`，用新的输出目录运行，勿混合不同分辨率结果。
预检仅代表该输入批次通过，不能保证所有后续输入／阶段绝不 OOM。

默认采用兼容性优先的 PyTorch 注意力实现；没有 `flash-linear-attention` / `causal-conv1d` 时，
Qwen 线性注意力会提示使用较慢的 PyTorch fallback。先完成 smoke 验证，再在独立环境验证加速内核；
不要在一组对比实验中途切换内核。完整训练耗时需以服务器实测为准。

## 3. 实验定义与方法适配

| 配置 | 主干 | 可训练部分 | 专家 |
|---|---|---|---|
| head_only | 冻结完整视觉—语言 Qwen | 分类头 | 无 |
| smope | 冻结完整视觉—语言 Qwen | 分类头 + K/V Prompt | 每 Q 头 25，Top-5 |

- 两者输入都是图片和固定指令 `Classify this image.`，不注入标签、候选类别或 task id。
- 取最后一个有效输入 token 的最终隐藏特征分类，不调用 `generate` 或语言输出头。
- 从模型配置识别完整注意力层；标准 9B 是零起始编号 3/7/11/15/19/23/27/31。
- 保留 Q/K norm、RoPE、GQA 原 K/V 共享、输出 gate 和投影；GQA 扩展后每 Q 头有独立专家。
- Prompt 是 RoPE 后的位置无关记忆。有效 token 的平均 Q 决定 Top-k，沿用 SMoPE 的共享 prompt 分数。
  普通 token 分数仍有因果掩码，但全输入路由不是严格自回归，禁止 KV cache / 生成。
- 首任务先 dense 初始化，然后 Top-k 训练；旧专家约束、使用频次、当前任务训练类别屏蔽保留。
- 所有评价只在**已见类别全集**中分类，不利用测试 task id 限定候选类别。
- 每个任务结束扫描当前任务训练图片；精确分片后汇总专家频次和每类特征统计，不使用旧原始图片。
- **明确适配差异：** 4096 维特征的全协方差开销较大，因此采用对角方差高斯特征采样校正分类头。
  老类保留其当时特征统计；两种 Qwen 配置使用相同校正规则。不是声称逐项复现 ViT 超参数。
- 默认 prompt/head LR 均为 1e-3，路由／旧专家损失权重各 1e-5、epsilon=0.4，AdamW、余弦衰减。
  这是启动配置而非已调优结果。`--correction-epochs 0` 可禁用校正，需为两种方法使用相同设置。
- 类别顺序遵循旧代码的 Python seeded shuffle；完整任务分别为 10×10、10×20、10×20。
  Qwen 使用确定性图片预处理，不沿用 ViT 随机裁剪；日志记录该差异，跨主干对比需说明预处理和预训练数据不同。

## 4. 复盘产物：不保存完整主干

默认目录 `outputs/qwen/<数据集>/<方法>/<模式>/seed-<种子>/`：

| 文件 | 内容 |
|---|---|
| environment.json | 配置、类别顺序、模型 config/index 摘要、Git 提交、软件/GPU信息 |
| events.jsonl | 分步骤／epoch 损失、LR、梯度范数、吞吐量、各阶段耗时、每卡峰值显存 |
| accuracy_matrix.csv | 行：训练到哪个任务；列：各测试任务准确率；未见任务留空 |
| summary.json / report.md | 最终平均准确率、过程平均准确率、遗忘率、后向迁移 |
| per_class_task_N.json | 原始／映射类别 ID、样本数、正确数、分类准确率 |
| confusion_task_N.npy | 各任务结束时的混淆矩阵 |
| experts_task_N.json | 每层每头累计选择频次、覆盖率、熵和已保护专家标记 |
| latest_adapter.pt | 仅最新轻量专家、分类头、任务状态、对角原型及恢复所需优化器/RNG |

不复制基础模型，不保留逐任务完整权重；恢复时重新读取 `MODEL_PATH` 的同一套基础权重。
检查点为 **epoch 边界恢复**：中断后重做未完成 epoch；任务后处理若中断会从最后训练 epoch 重做。
日志保留历史尝试，复盘时间包含重试开销。只加载自己生成的可信 `.pt` 文件。
基础权重标识记录配置、分片索引摘要，未逐字节散列 9B 权重；恢复时须保证权重目录没有被替换。

训练样本在 DDP 尾部可能补齐少量重复项，训练统计明确包含补齐项；评价、专家扫描和原型统计使用不重复分片。
准确率单位为百分数；遗忘率和 BWT 单位为百分点；首任务遗忘率和 BWT 定义为 0。

```bash
python scripts/qwen/summarize.py outputs/qwen
```

生成 `comparison.csv` 和 `comparison.md`。比较时保持所有训练参数、图像预算和任务划分一致；
不同自定义超参数用不同 `OUTPUT_ROOT`，不要把不一致配置的结果放在同一多种子汇总组。

## 5. 验证边界

本仓库提供无需下载权重的真实 Qwen 小配置测试，包括图文前向／反向、梯度检查点、padding、
专家状态恢复，以及两进程 CPU/Gloo 的同步、统计、评价和断点测试。
这不等于已在完整 Qwen3.5-9B / A100 / NCCL 上验证；服务器 `preflight` 和 `smoke` 是必需的交付验证步骤。
`setup.sh` 锁定 Linux CUDA PyTorch 与 Transformers 版本；本地 CPU 的 PyTorch 版本可能不同。

官方结构依据：
- https://huggingface.co/Qwen/Qwen3.5-9B
- https://github.com/huggingface/transformers/blob/v5.3.0/src/transformers/models/qwen3_5/modeling_qwen3_5.py
