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

| 问题 | 影响 | 处理 |
|---|---|---|
| `cl_dataset/COCO2014/train2014/` 为空 | VQAv2、Grounding 训练图像缺失 | 补齐 `COCO_train2014_*.jpg`（约 83k 张） |
| `cl_dataset/GQA/images/` 为空 | GQA 任务完全无图 | 下载 GQA images，命名 `<imageId>.jpg` |
| 指令中图像为 `./cl_dataset/...` 相对路径 | cwd 不对则全部找不到图 | 一律在 `datas/CoIN/` 下启动训练 |
| `refs(*).p` 是 pickle | 用 json 库会报错 | `pickle.load`，并按 `split` 字段切分 train/val |
| 部分任务缺 `val.json`/`test.json` | 只有 train 指令 | 评估集需用 `copy_to_qwen.py` 从 CoIN 原始指令自行生成 |
| `OCR-VQA/_failed_urls.json` 非空 | 小部分图像缺失 | 先核对，缺图样本在 dataloader 里跳过 |

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