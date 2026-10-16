# CoIN 数据集目录结构与文件格式说明

路径：`datas/CoIN/`

用于 CoIN（Continual Instruction tuNing）基准实验。整体分两层：

- `cl_dataset/` —— 原始图像与标注（各原始数据集原样落地）
- `Instructions_Qwen/` —— 已转成 Qwen-VL 对话格式的指令数据（**训练直接读这一层**）

---

## 1. 总体结构

```
datas/CoIN/
├── cl_dataset/                     # 原始数据集（图像 + 原始标注）
│   ├── COCO2014/
│   ├── GQA/
│   ├── ImageNet_withlabel/
│   ├── OCR-VQA/
│   ├── ScienceQA/
│   ├── TextVQA/
│   ├── VizWiz/
│   ├── VQAv2/
│   ├── refcoco/  refcoco_plus/  refcocog/     # Grounding 任务
└── Instructions_Qwen/              # Qwen 格式指令数据（任务增量顺序的 8 个任务）
    ├── VQAv2/  ImageNet/  OCRVQA/  ScienceQA/
    ├── GQA/  Grounding/  VizWiz/  TextVQA/
    ├── Multitask/                  # 8 任务合并版
    ├── copy_to_qwen.py             # 转换脚本（CoIN 原始格式 → Qwen 格式）
    └── copy_to_qwen_test.py
```

---

## 2. Instructions_Qwen —— 训练数据（重点）

每个任务目录下一个 `train.json`：

| 任务目录 | 文件 | 大小 |
|---|---|---|
| VQAv2 | `train.json` | 115 MB |
| ImageNet | `train.json` / `train_new.json` | 67 MB / 69 MB |
| OCRVQA | `train.json` / `train_new.json` | 219 MB / 223 MB |
| ScienceQA | `train.json` / `train_test.json` | 7.8 MB / 3.3 KB |
| GQA | `train.json` / `train_new.json` | 222 MB / 223 MB |
| Grounding | `train.json` | 107 MB |
| VizWiz | `train.json` / `train_new.json` | 9.7 MB / 9.8 MB |
| TextVQA | `train.json` / `train_new.json` | 20 MB / 21 MB |
| Multitask | `train.json` / `train_new.json` | 731 MB / 740 MB |

> `train_new.json` = 用 `copy_to_qwen.py` 重新生成的版本（脚本第 3 行硬编码输入路径 `playground/Instructions_Type1/<TASK>/test.json`，产物写到 `playground/Instructions_Type1_Qwen/`）。当前 `train.json` 已是 Qwen 格式，二者内容基本等价，取其一即可。

### 文件格式

一个 JSON **数组**，每个元素是一条多轮样本：

```json
[
  {
    "id": "1",
    "image": "ScienceQA/images/train/1/image.png",
    "conversations": [
      {
        "from": "user",
        "value": "Picture 1: <img>./cl_dataset/ScienceQA/images/train/1/image.png</img>\nWhich of these states is farthest north?\nA. West Virginia\nB. Louisiana\nC. Arizona\nD. Oklahoma\nAnswer with the option's letter from the given choices directly."
      },
      { "from": "assistant", "value": "A" }
    ]
  }
]
```

字段说明：

- `id` — 样本 id（字符串）
- `image` — 相对 `cl_dataset/` 的原始图像路径（不含 `./cl_dataset/` 前缀）
- `conversations` — 交替的 `user` / `assistant` 轮次
  - 图像以 `Picture 1: <img>./cl_dataset/...</img>` 内联在 user 文本里
  - `<image>` 占位符在转换时被替换成上述 `<img>` 标签

**关键坑：`./cl_dataset/...` 是相对路径，依赖运行时 cwd。** 必须在 `datas/CoIN/` 这一层启动训练，否则图像全部找不到。

---

## 3. cl_dataset —— 各子目录结构与格式

### 3.1 COCO2014/ （VQAv2 的图像源）
```
COCO2014/
├── annotations/
│   ├── captions_train2014.json          64 MB
│   ├── captions_val2014.json            31 MB
│   ├── instances_train2014.json        317 MB
│   ├── instances_val2014.json          153 MB
│   ├── person_keypoints_train2014.json 163 MB
│   ├── person_keypoints_val2014.json    78 MB
│   └── image_info_test2014.json          9 MB
├── train2014/          # ⚠ 空目录（0 个文件）
├── val2014/            # 40504 张 COCO_val2014_*.jpg
└── test2014/           # 40775 张 COCO_test2014_*.jpg
```
COCO 标注为标准格式：`{info, licenses, images[{id,file_name,height,width}], annotations[...]}`。

**⚠ `train2014` 为空，而 `Instructions_Qwen/VQAv2/train.json` 引用的正是 `./cl_dataset/COCO2014/train2014/...`** —— 跑 VQAv2 任务前必须补齐 train2014（约 8.3 万张，`COCO_train2014_*.jpg`）。

### 3.2 GQA/
```
GQA/
├── images/             # ⚠ 空目录（0 个文件）
├── questions1.2/
│   ├── train_balanced_questions.json   775 MB
│   ├── train_all_questions/            # 分片：train_all_questions_{0..9}.json
│   ├── train_all_questions.json
│   ├── val_balanced_questions.json     109 MB
│   ├── val_all_questions.json          1.62 GB
│   ├── testdev_balanced_questions.json  10 MB
│   ├── test_all_questions.json         150 MB
│   ├── challenge_*.json, submission_all_questions.json
│   └── readme.txt
└── sceneGraphs/
    ├── train_sceneGraphs.json          303 MB
    └── val_sceneGraphs.json             43 MB
```
questions 为标准 GQA 格式：`{questionId: {imageId, question, answer, fullAnswer, isBalanced, types{structural,semantic,detailed}, annotations{...}}}`；sceneGraphs 为 `{imageId: {width,height,objects{id→{name,attributes,relations}}, ...}}`。

**⚠ `images/` 为空** —— 需从 gqadataset.org 下载 GQA 图像，文件名形如 `<imageId>.jpg`。

### 3.3 ImageNet_withlabel/
```
ImageNet_withlabel/
├── train/     # 101 个类别目录，每类约 1300 张
└── val/       # 1002 个类别目录
```
`train/<wnid>/<wnid>_<id>.JPEG`，例如 `train/n01514668/n01514668_10004.JPEG`。
> train 只有 **101** 个类（CoIN 用其中 100 类）；val 为完整 1000 类 + 额外目录。

### 3.4 OCR-VQA/
```
OCR-VQA/
├── dataset.json          108 MB
├── images/               按 image id 命名，如 000195850X.jpg
├── _failed_urls.json     下载失败清单
├── LICENCE.txt
└── loadDataset.py
```
`dataset.json` 为**字典**，key = 图像 id（即图像文件名去扩展名）：
```json
{
  "<image_id>": {
    "imageURL": "...",
    "questions": ["...", "..."],
    "answers": ["...", "..."],
    "split": 1,
    "genre": "...",
    "authorName": "...",
    "title": "..."
  }
}
```
`split`：`1`=train，`2`=val，`3`=test。

### 3.5 ScienceQA/
```
ScienceQA/
├── problems.json        30 MB   # 全量题目
├── _problems.json       30 MB   # 同上（副本）
├── pid_splits.json     491 KB
└── images/{train,test,val}/<pid>/image.png
```
- `problems.json`：`{pid: {question, choices[], answer, image, hint, task, grade, subject, topic, category, skill, lecture, solution, ...}}`（标准 ScienceQA）
- `pid_splits.json`：`{train:[12726], val:[4241], test:[4241], trainval, minitrain, minival, minitest}` → **与 `Instructions_Qwen/ScienceQA/train.json` 的 12726 条一致**
- 图像每样本一目录：`images/train/<pid>/image.png`

### 3.6 TextVQA/
```
TextVQA/
├── TextVQA_0.5.1_train.json   21 MB
├── TextVQA_0.5.1_val.json      3 MB
├── train/     # <image_id>.jpg
└── val/
```
标准格式：`{data:[{question_id, question, answers[], image_id, image_classes, ...}], ...}`。

### 3.7 VizWiz/
```
VizWiz/
├── VizWiz_train_annotations.json  25 MB
├── VizWiz_val_annotations.json     5 MB
├── VizWiz_test_annotations.json    2 MB
├── train/   # <image>.jpg
├── val/
└── test/
```
标准格式：`{data:[{image, question, answers[], answer_type, answerable, ...}]}`。

### 3.8 VQAv2/
```
VQAv2/
├── v2_OpenEnded_mscoco_train2014_questions.json   40 MB
├── v2_OpenEnded_mscoco_val2014_questions.json     19 MB
├── v2_mscoco_train2014_annotations.json          339 MB
├── v2_mscoco_val2014_annotations.json            164 MB
```
标准 VQA v2 格式；图像复用 `COCO2014/{train,val}2014/`。

### 3.9 refcoco / refcoco_plus / refcocog  （Grounding 任务）
```
refcoco/refcoco/          refcoco_plus/refcoco+/     refcocog/refcocog/
├── instances.json        ├── instances.json         ├── instances.json
├── refs(unc).p           ├── refs(unc).p            ├── refs(google).p
└── refs(google).p                                   └── refs(umd).p
```
- `instances.json` — COCO 风格实例标注
- `refs(*).p` — **Python pickle**（不是 JSON），需 `pickle.load`。内容为 list，每项含：
  `{split ('train'/'val'/'test'/'trainval'), sentences[{sent_id, sent, raw, tokens, ...}], file_name, image_id, ref_id, ann_id, bbox [x,y,w,h], category_id, ...}`
- Grounding 任务图像同样复用 `COCO2014/train2014`（当前为空，见 3.1）

---

## 4. 任务 → 数据 映射（CoIN 8 任务）

| 任务 | 指令文件 | 依赖的图像目录 |
|---|---|---|
| VQAv2 | `Instructions_Qwen/VQAv2/train.json` | `cl_dataset/COCO2014/{train2014,val2014}` |
| ImageNet | `Instructions_Qwen/ImageNet/train.json` | `cl_dataset/ImageNet_withlabel/` |
| OCRVQA | `Instructions_Qwen/OCRVQA/train.json` | `cl_dataset/OCR-VQA/images/` |
| ScienceQA | `Instructions_Qwen/ScienceQA/train.json` | `cl_dataset/ScienceQA/images/` |
| GQA | `Instructions_Qwen/GQA/train.json` | `cl_dataset/GQA/images/` |
| Grounding | `Instructions_Qwen/Grounding/train.json` | `cl_dataset/COCO2014/{train2014,val2014}` + refcoco* |
| VizWiz | `Instructions_Qwen/VizWiz/train.json` | `cl_dataset/VizWiz/{train,val,test}/` |
| TextVQA | `Instructions_Qwen/TextVQA/train.json` | `cl_dataset/TextVQA/{train,val}/` |

（上表为 CoIN 论文的 8 任务增量顺序，按实际 `run` 配置可调整顺序。）

---

## 5. 跑基准前必须处理的问题

> 2026-09-28 复核：下面前四项**已解决**（数据已补齐并逐类校验通过），保留记录以免重复排查。
> 图像存在性一律用 **HTTP PROPFIND/HEAD 直连服务器**判定 —— davfs 挂载对
> `COCO2014/train2014`、`GQA/images` 这类大目录会返回过期的空列表，连单文件 `stat` 都会误报缺失。

| 问题 | 状态 | 说明 |
|---|---|---|
| `cl_dataset/COCO2014/train2014/` 为空 | ✅ 已补齐 | 82783 张、13492.5MB，上传后子树校验 ALL MATCH |
| `cl_dataset/GQA/images/` 为空 | ✅ 已补齐 | 148854 张、21979.3MB，校验 ALL MATCH |
| `ImageNet_withlabel/{train,val}` | ✅ 完整 | train 101 类（每类约 1300 张，最少 936）、val 1002 类 |
| 指令中图像为 `./cl_dataset/...` 相对路径 | ⚠️ 仍需注意 | 路径解析已改为显式基于 coin-root，不再依赖 cwd |
| `refs(*).p` 是 pickle | ⚠️ 仍未用到 | Grounding 走 `Instructions_*` 的 JSON，未直接读 pickle |
| 部分任务缺 `val.json`/`test.json` | ✅ 已处理 | 评估改用 `Instructions_Original`，见 §6 |
| `OCR-VQA/_failed_urls.json` 非空 | ⚠️ 已确认镜像缺损 | 训练侧按 manifest 确定性过滤；评测侧不得运行时跳过。2026-10-16 普查确认 test 缺 7 个文件/35 条记录，按交接手册 §12.10 从官方同源数据补齐并记录校验和 |

---

## 6. 评测集：为什么不能用 `Instructions_Qwen`，以及每任务该评哪个文件

`Instructions_Qwen/<task>/` **只有 `train.json`** —— 训练层的全部内容。留出指令只在
`Instructions_Original/<task>/`，且是**扁平格式**（`question_id` / 可选 `image` / `text` / `answer`），
没有 `<img>` 标记、没有 `conversations`。因此评测侧必须在加载时把扁平记录包成 Qwen 对话格式，
源文件保持不动（由 `qwen_smope/data.py` 的 `_convert_flat` 完成）。

**官方 test 文件对三个任务不可用**，必须改评别的 split：

| 任务 | 评测文件 | n | 为什么 |
|---|---|---|---|
| ScienceQA | `test.json` | 4241 | 2224 条**本来就无图**（官方设计，见下） |
| TextVQA | `val.json` | 5000 | `test.json` 无 answer，且 `test_images/` 从未下载 |
| ImageNet | `test.json` | 5050 | 已限定为 101 类 × 50 张 val 图 |
| GQA | `test.json` | 12578 | — |
| VizWiz | `val.json` | 4319 | `test.json` 无 answer（官方 test 不公开标注） |
| Grounding | `test.json` | 30969 | 用 `answer_bbox`（归一化坐标，与训练目标一致） |
| VQAv2 | `val.json` | 214354 | `test.json` 无 answer，且引用不存在的 `COCO2014/test2015/` |
| OCRVQA | `test.json` | 99926 | — |

映射表在 `qwen_smope/data.py` 的 `EVAL_SPLIT_FILES`；`--eval-split` 可改动，但**任务声明的
留出文件优先**于请求的名字，否则会命中不可用的 `test.json`。

### 评测规模：每任务固定采样 N 条

全量评测一轮 346437 条（VQAv2 214354 + OCRVQA 99926 + Grounding 30969 占绝大部分），
持续学习要评 36 轮，按周计。因此用 `--max-eval-samples N` 对**留出集**做采样。

关键约束：**所有方法、所有种子、所有轮次必须评同一批样本**，否则准确率矩阵在比不同的数据。
所以采样种子**不由 `--seed` 派生**，而是由任务与 split 派生
（`eval_dataset()` 里的 `f"coin-eval-{--eval-sample-seed}-{task}-{split}"`），
`--eval-sample-seed` 默认 42。采样是**随机子集且保持源文件顺序**，
不是"取前 N 条"——后者对 214k 条的 VQAv2 val 只会评到其中一段连续切片。

训练 split 的当前语义是：先用真实 processor、chat template、图片尺寸、视觉 token 上限和
`max_length` 扫描完整 split，确定性剔除无法保留完整答案的样本，再从有效样本序列取前 N 条。
过滤缓存不以 N 为键，所以 smoke/full、所有方法和 seed 使用同一有效前缀。详细字段、剔除率门禁
和服务器验证状态统一见 [`SMOKE_OPTIMIZATION_HANDOFF.md`](SMOKE_OPTIMIZATION_HANDOFF.md)。
评测采样不应用训练过滤，仍保持上文固定种子的随机子集语义。若镜像缺少官方评测图像，应补回
同一 image id 的官方原始文件，而不是在 DataLoader 中动态跳过；动态跳过会让 DDP rank 样本数
不一致，也会改变方法/seed 间的评测 ID。OCRVQA 的7文件修复流程与审计字段见
[`SMOKE_OPTIMIZATION_HANDOFF.md` §12.10](SMOKE_OPTIMIZATION_HANDOFF.md)。

> 注：曾评估过改用 `vantuan5644/CoIN-CL` 提供的固定种子子集
> （`*_subset*_seed42.json`），但那是**第三方**为另一套训练代码制作的产物，其取舍规则
> 无法从数据反推（VQAv2/OCRVQA 约 21%，ImageNet 91%、Grounding 70%），
> 且 ImageNet 子集的 `image` 路径丢了类别目录（`val/<file>.JPEG` 而非 `val/<wnid>/<file>.JPEG`）。
> 经确认**不采用**，避免与 CoIN 论文口径混淆。

### ScienceQA 的无图样本不是下载问题

官方 `problems.json` 21208 条里 **10876 条（51.3%）`image` 为 null** —— 这些是纯文本题
（靠 passage / lecture / hint 作答）。评测集 4241 条中 2224 条（52.4%）无 `image` 键，
与原始标注**逐条交叉核对 0 处不一致**，且有图的样本**图像缺失 0 个**。所以
`content` 列表里对这类样本**不能放** `{"type": "image"}`，否则 processor 报错；
训练集 `ScienceQA/train.json` 同样有 6508/12726 条无图，同样必须支持。

### 指令集来源核对（2026-09-28 定论）

用 HF tree API 把本地 11 个 `Instructions_Original/*.json` 与两个上游逐字节比对：

| 来源 | 说明 |
|---|---|
| `Zacks-Chen/CoIN` | CoIN 官方发布的指令集 |
| `vantuan5644/CoIN-CL` | 第三方镜像，README 自述指令部分即取自 Zacks-Chen，另加 4 个 seed42 子集与 `Joint/train_all8.json` |

**9/11 与两个上游逐字节一致**（GQA/Grounding/OCRVQA/ScienceQA/TextVQA×2/VQAv2×2/VizWiz-test）。
另 2 个只差 `image` 路径字段，且**本地是修正版**：

| 文件 | 本地 | 上游 | 判定 |
|---|---|---|---|
| `ImageNet/test.json` | `val/<wnid>/<file>.JPEG` | `val/<file>.JPEG`（两上游一致） | 上游路径在本数据根**找不到图**（实际在 `val/<wnid>/`），本地是修复 |
| `VizWiz/val.json` | `VizWiz_val_*.jpg` | Zacks 为 `*.jpg.jpg` **双扩展名 bug**；CoIN-CL 已修 | 本地与 CoIN-CL **逐条相同** |

两个文件的**记录数、`question_id` 集合、`answer`、`text` 均与上游完全一致，没有子集化**。
差异只是"让路径真的能指到图"。文件字节数不同处仅为 JSON 缩进与键序。

### 运行期数据布局（2026-09-28 起，davfs 不在实验路径上）

davfs 挂载**不能**承担这个负载，实测数据：

| 通道 | 233MB 的 GQA train.json | 速率 |
|---|---|---|
| `/webvad` davfs 挂载 | 582 s | ~0.4 MB/s |
| HTTP 直连 WebDAV 服务器 | 2.4 s | **97 MB/s** |

差 244 倍。原因是 davfs 把元数据串行化在**单个客户端进程**里、`dir_refresh` 默认 **60 秒**
就重新 PROPFIND 整个目录（`GQA/images` 有 **148854** 条）、按 **16KiB** 分片读、并且对
"由外部 HTTP 客户端写入"的目录树会返回**过期的空列表**（连单文件 `stat` 都误报缺失）。
所以现在：

| 数据 | 位置 | 读取方式 |
|---|---|---|
| 指令集（`Instructions_*`，约 3.1GB） | 本地 `/mnt/zq/SMOPE/coin_local/` | 本地文件系统 |
| TextVQA 官方标注（多答案软评分依赖） | 同上 `cl_dataset/TextVQA/` | 本地文件系统 |
| 图像（约 70GB，放不下本地） | webdav | `--image-transport http` 直连 |
| 模型权重（19.33GB，放不下本地） | **`/dev/shm/qwen35-9b`**（tmpfs，内存） | 本地文件系统 |
| processor / tokenizer | 随模型副本 | 本地文件系统 |

**为什么指令集要本地化**：持续学习里评测集会被**重复构造 36 次**（每个任务完成后重评所有
已见任务），走 HTTP 也白跑 36 遍。

**为什么模型放 tmpfs**：本地盘只剩 24G 放不下 19.33GB；放在 webdav 则每次启动都要按
0.4MB/s 读十几小时。tmpfs 的代价是**易失** —— 重启后重跑
`bash /root/coin-stage/stage_model_shm.sh`（HTTP，约 3~4 分钟，幂等）。

准备脚本（都在 `/root/coin-stage/`，均带字节数校验、幂等、可重跑）：

```bash
bash stage_coin_local.sh     # 指令集 -> /mnt/zq/SMOPE/coin_local
bash stage_model_shm.sh      # 模型   -> /dev/shm/qwen35-9b
```

`scripts/qwen/run.sh` 的默认值已经指向这套布局，并会检查两者是否就绪：

```bash
bash scripts/qwen/run.sh smope preflight
```

### 关于图像存在性校验

`--image-transport http` 时 **不做** `Path.is_file()` 探测（`verify_images=False`）。
两个原因：挂载的 `stat` 会撒谎；而且每个任务构造数据集时要对**每一条**记录做一次
（GQA 72140 条、VQAv2 82783 条），这些 FUSE 往返本身就是可观开销。路径仍然照常推导
（含 TextVQA 的 `train_images`→`val_images` 别名修正），只是不探测——HTTP 404 会直接报错。

### 运行期性能：两个必须知道的坑

**坑 1：必须装 `flash-linear-attention`，否则慢 20 倍。**

Qwen3.5 是混合注意力架构（SMoPE 注入的 8 个 full-attention 层，其余约 24 层是线性注意力
`Qwen3_5GatedDeltaNet`）。缺 fla 时 `modeling_qwen3_5.py` 会回退到纯 torch 实现，其
**backward 的成本与 batch 大小无关**（实测 batch 1/2/4 的「前向+反向」为
1.045 / 1.125 / 1.169 秒，而前向本身正常缩放），即每步约 0.9 秒的固定开销。

**注意**：`modeling_qwen3_5.py` 里 `is_fast_path_available` 只被用来决定**是否打印那条
"fast path is not available" 警告**，它**不控制代码路径**。四个核各自独立回退，所以
**只装 fla 就够了**（depthwise conv 会退化成普通 `nn.Conv1d`，代价可忽略）；
`causal-conv1d` 需要 nvcc 编译且 wheel 在 GitHub 上，可以不管。
装完之后**那条警告仍会打印，是误导性的**。

版本约束：`fla 0.5.x` 要求 `torch>=2.7` + `triton>=3.3`，与本环境（torch 2.5.1 /
triton 3.1.0）**不兼容**；**用 `fla 0.3.1`**（要求 `torch>=2.5` + `einops`，不钉 triton）：

```bash
pip install --no-deps flash-linear-attention==0.3.1   # 必须 --no-deps, 否则会想动 torch
pip install einops
```

**数值提示**：fla 的核是 **bf16-only**（不接受 fp32），与 torch 回退（内部升 fp32）在单次
调用上相差约 **0.5%**（实测相对差 5.25e-03），经 24 层累积后同样数据同一种子的 loss
相差 **12~17%**。两条路径各自逐位可复现。**没有 fp32 参考标准可以裁决**，但 torch 回退
的耗时不可接受，所以实际必须用 fla。三个方法必须共用同一内核（环境级，天然满足），
方法之间的对比仍然有效；`environment.json` 里的 `linear_attention_kernels` 字段可追溯。

**坑 2：`--workers` 的地板开销（主要在 smoke 模式困扰人）。**

只要 `workers>0`，每个 DataLoader 就要付一次 forkserver 启动 + 各 worker import
torch/transformers 并反序列化 processor 的代价，**实测约 24~38 秒且几乎不随 worker 数变化**。
而 `make_loader` 在训练、`route_scan` 以及**每次 `evaluate_task`** 都会被调用，所以
smoke 模式下这笔开销完全淹没真实计算：

| 阶段 | workers=8 | **workers=0** |
|---|---|---|
| `train_sparse` | 23.46 s | **0.45 s** |
| `evaluate_ScienceQA` | 46.32 s | **4.46 s** |
| `route_scan` | 49.31 s | **8.75 s** |

正式 run 总共只创建约 52 个 DataLoader，且 `workers>0` 能让 HTTP 取图与计算重叠。因此当前
默认按模式选择：preflight/smoke 为 `--workers 0`，full 为 `--workers 4`。如需覆盖必须把值
写入实验配置，避免把 worker 启动成本误认成模型步时。

### 已知的数据瑕疵（不影响可用性，但读数时要知道）

- `Instructions_Original/TextVQA/val.json` 的 `image` 写成 `TextVQA/train_images/...`，
  实际文件在 `TextVQA/val_images/`。已在 `IMAGE_DIRECTORY_ALIASES` 里做别名修正。
- ImageNet 训练标签结尾多一个句号（`"Cock."`），评测标签没有（`"Cock"`）——
  129833/129833 条如此，是转换产物，其余任务为 0%。`scoring.py` 的双向子串匹配可兼容。
- ImageNet 101 类只有 100 个唯一标签：`"Maillot"` 被 `n03710637` 与 `n03710721` 两类共用，
  最多 50 条（约 1%）本质歧义，属 ImageNet 本身的问题。

---

## 6. 转换脚本工作方式

`Instructions_Qwen/copy_to_qwen.py`：

1. 读 `playground/Instructions_Type{1,2}/<TASK>/<split>.json`（CoIN 原始指令格式：`image` + `conversations`，user 文本里用 `<image>` 占位）
2. `from` 字段：`human`→`user`，`gpt`→`assistant`
3. `<image>` → `Picture 1: <img>./cl_dataset/<image_path></img>`
4. 写到平行的 `playground/Instructions_Type{1,2}_Qwen/<TASK>/<split>.json`（`indent=4`）

`copy_to_qwen_test.py` 为另一种原始格式（单条 `text` 字段 + 可选 `image`），同样是 `<image>` → `<img>` 替换。

> 若需新增/重建某任务的指令文件，直接复用这两个脚本并修正第 3 行的输入路径即可。

---

*生成时间：2026-09-28*
