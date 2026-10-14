#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../.."
if [[ $# -lt 1 ]]; then
  echo 'Usage: bash scripts/qwen/run.sh {zero_shot|lora|smope} [preflight|smoke|full] [extra arguments]'
  exit 2
fi
METHOD="$1"; shift
MODE="${1:-smoke}"; if [[ $# -gt 0 ]]; then shift; fi
case "$METHOD" in zero_shot|lora|smope) ;; *) echo "Unknown method: $METHOD"; exit 2;; esac
# 不要依赖调用者先 source activate：torchrun 在仓库内的 .venv2 里。
# 已经激活过的环境保持不变（只在该 PATH 上仍找不到 torchrun 时才补）。
if ! command -v torchrun >/dev/null 2>&1 && [[ -x "$PWD/.venv2/bin/torchrun" ]]; then
  export PATH="$PWD/.venv2/bin:$PATH"
fi
if ! command -v torchrun >/dev/null 2>&1; then
  echo "找不到 torchrun。请先激活虚拟环境:" >&2
  echo "  source $PWD/.venv2/bin/activate" >&2
  exit 2
fi
NPROC_PER_NODE="${NPROC_PER_NODE:-1}"
case "$NPROC_PER_NODE" in
  1) export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}" ;;
  2) export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1}" ;;
  *) echo "NPROC_PER_NODE 只支持 1（默认单卡）或 2（受控 DDP）" >&2; exit 2 ;;
esac
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

# 默认值针对本机布局：指令集与 processor 放本地盘，图像直连 WebDAV 服务器。
# davfs 挂载不能承担这个负载——它把元数据串行化在单进程里、每 dir_refresh 秒重新
# 拉取整个目录（GQA/images 有 148854 条）、按 16KiB 分片读，实测 233MB 要 582s，
# 而同样字节走 HTTP 只要 2.4s（97MB/s）。所以挂载完全不在实验路径上。
COIN_ROOT="${COIN_ROOT:-/mnt/zq/SMOPE/coin_local}"
# 模型放 /dev/shm(tmpfs): 约 18GB, 本地盘只剩 24G 放不下, 而走 davfs 读它
# 实测速率下可能要十几小时。副本里也含 processor/tokenizer 文件, 所以不需要
# 再单独传 --processor-path(PROCESSOR_PATH 留空即回退到 MODEL_PATH)。
MODEL_PATH="${MODEL_PATH:-/dev/shm/qwen35-9b}"
PROCESSOR_PATH="${PROCESSOR_PATH:-}"
IMAGE_TRANSPORT="${IMAGE_TRANSPORT:-http}"
FILTER_CACHE="${FILTER_CACHE:-outputs/coin-qwen3.5-9b/_filter_cache}"

if [[ ! -d "$COIN_ROOT" ]]; then
  echo "COIN_ROOT 不存在: $COIN_ROOT" >&2
  echo "  先运行: bash /root/coin-stage/stage_coin_local.sh" >&2
  exit 2
fi
if [[ ! -f "$MODEL_PATH/model.safetensors.index.json" ]]; then
  echo "MODEL_PATH 不是一份完整的模型目录: $MODEL_PATH" >&2
  echo "  /dev/shm 是 tmpfs, 重启后会清空; 先运行:" >&2
  echo "    bash /root/coin-stage/stage_model_shm.sh" >&2
  echo "  或者指定别处已有的副本: MODEL_PATH=/path/to/Qwen3.5-9B bash scripts/qwen/run.sh ..." >&2
  exit 2
fi

PROCESSOR_ARGS=()
[[ -n "$PROCESSOR_PATH" ]] && PROCESSOR_ARGS=(--processor-path "$PROCESSOR_PATH")

for SEED in ${SEEDS:-0}; do
  OUT="${OUTPUT_ROOT:-outputs/coin-qwen3.5-9b}/${METHOD}/${MODE}/seed-${SEED}"
  # 单卡是当前保守默认值；双卡 DDP 仍处于受控诊断阶段。
  torchrun --standalone --nproc_per_node="$NPROC_PER_NODE" -m qwen_smope.train \
    --method "$METHOD" --mode "$MODE" --seed "$SEED" \
    --model-path "$MODEL_PATH" --coin-root "$COIN_ROOT" \
    --image-transport "$IMAGE_TRANSPORT" --filter-cache "$FILTER_CACHE" \
    "${PROCESSOR_ARGS[@]}" --output "$OUT" "$@"
done
