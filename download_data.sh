#!/usr/bin/env bash
# 按顺序逐个下载并解压到项目根目录的 data/（与 README 一致）。
# 用法：在项目根目录执行  sh smope/download_data.sh

set -u
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
DATA="$ROOT/data"
mkdir -p "$DATA"
cd "$DATA" || exit 1

echo "=== [1/3] CIFAR-100 ==="
if [ -d cifar-100-python ]; then
  echo "已存在 cifar-100-python，跳过。"
else
  wget -c "https://www.cs.toronto.edu/~kriz/cifar-100-python.tar.gz" -O cifar-100-python.tar.gz
  tar xzf cifar-100-python.tar.gz
  rm -f cifar-100-python.tar.gz
  echo "CIFAR-100 完成。"
fi

echo "=== [2/3] ImageNet-R ==="
if [ -d imagenet-r ]; then
  echo "已存在 imagenet-r，跳过。"
else
  wget -c "https://people.eecs.berkeley.edu/~hendrycks/imagenet-r.tar" -O imagenet-r.tar
  tar xf imagenet-r.tar
  rm -f imagenet-r.tar
  echo "ImageNet-R 完成。"
fi

echo "=== [3/3] CUB-200 ==="
if [ -d CUB_200_2011 ]; then
  echo "已存在 CUB_200_2011，跳过。"
else
  # 使用整包下载，避免对 Caltech→S3 跳转使用断点续传导致签名过期卡住
  rm -f CUB_200_2011.tgz
  curl -fL --retry 10 --retry-delay 30 -o CUB_200_2011.tgz \
    "https://data.caltech.edu/records/65de6-vp158/files/CUB_200_2011.tgz?download=1"
  tar xzf CUB_200_2011.tgz
  rm -f CUB_200_2011.tgz
  echo "CUB-200 完成。"
fi

echo "全部数据集步骤结束。数据目录: $DATA"
