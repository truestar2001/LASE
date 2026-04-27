#!/usr/bin/env bash
set -euo pipefail

# Edit this block directly when switching experiments.
# Examples:
# CONFIG="configs/train_bicodec_exp1.yaml"  # decoder stack only
# CONFIG="configs/train_bicodec_exp2.yaml"  # exp1 + speaker classification
# CONFIG="configs/train_bicodec_exp3.yaml"  # same-wav speaker time adapter
# GPU_IDS="0,1"                              # use GPU 0 and 1
# NPROC_PER_NODE=2                            # usually match the number of GPUs
# MASTER_PORT=29500
CONFIG="configs/train_bicodec_exp3.yaml"
GPU_IDS="0,1"
NPROC_PER_NODE=2
MASTER_PORT=29500

export PYTHONPATH="${PYTHONPATH:-}:$(pwd)/src"

if [[ -n "${GPU_IDS}" ]]; then
  export CUDA_VISIBLE_DEVICES="${GPU_IDS}"
fi

echo "[ddp] config=${CONFIG} gpu_ids=${GPU_IDS:-all} nproc_per_node=${NPROC_PER_NODE} master_port=${MASTER_PORT}"

torchrun \
  --standalone \
  --nproc_per_node="${NPROC_PER_NODE}" \
  --master_port="${MASTER_PORT}" \
  -m bicodec.cli.train \
  --config "${CONFIG}"
