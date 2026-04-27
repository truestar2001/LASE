#!/usr/bin/env bash
set -euo pipefail

# Edit this block directly when switching experiments.
# Examples:
# CONFIG="configs/train_bicodec_exp1.yaml"
# CONFIG="configs/train_bicodec_exp2.yaml"
# GPU_IDS="0"
CONFIG="configs/train_bicodec_exp3.yaml"
GPU_IDS="0"

export PYTHONPATH="${PYTHONPATH:-}:$(pwd)/src"

if [[ -n "${GPU_IDS}" ]]; then
  export CUDA_VISIBLE_DEVICES="${GPU_IDS}"
fi

echo "[train] config=${CONFIG} gpu_ids=${GPU_IDS:-all}"

python3 -m bicodec.cli.train --config "${CONFIG}"
