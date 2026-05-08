#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="${ROOT_DIR:-${SCRIPT_DIR}}"
REPO_ROOT="$(cd "${ROOT_DIR}/../.." && pwd)"
cd "${ROOT_DIR}"

mkdir -p logs
LOG_FILE="logs/run_advwave_defense_$(date +%Y%m%d_%H%M%S).log"
exec >>"${LOG_FILE}" 2>&1
echo "[START] AdvWave defense run at $(date)"

PYTHON_BIN="${PYTHON_BIN:-python}"
MODEL_NAME="${MODEL_NAME:-}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export PYTHONUNBUFFERED=1

"${PYTHON_BIN}" -u run_advwave_defense.py \
  --generate_tts \
  --model_name "${MODEL_NAME}" \
  --multipliers -2 -1 0 \
  --layer 15 \
  --attacks audio_ours \
  --vectors "${REPO_ROOT}/vectors/qwen2_audio/vec_ep100_layer15.pt"
