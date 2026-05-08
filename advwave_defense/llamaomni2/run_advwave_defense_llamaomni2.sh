#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="${ROOT_DIR:-${SCRIPT_DIR}}"
REPO_ROOT="$(cd "${ROOT_DIR}/../.." && pwd)"
cd "${ROOT_DIR}"

mkdir -p logs
LOG_FILE="${LOG_FILE:-logs/run_advwave_defense_llamaomni2_$(date +%Y%m%d_%H%M%S).log}"
exec >>"${LOG_FILE}" 2>&1
echo "[START] LLaMA-Omni2 AdvWave defense run at $(date)"

PYTHON_BIN="${PYTHON_BIN:-python}"
GPU_ID="${GPU_ID:-4}"
GPU_UUID="$(nvidia-smi --query-gpu=index,uuid --format=csv,noheader,nounits | awk -F',' -v idx="${GPU_ID}" '$1+0==idx {gsub(/ /, "", $2); print $2}')"
if [[ -z "${GPU_UUID}" ]]; then
  echo "ERROR: cannot resolve GPU UUID for index ${GPU_ID}" >&2
  exit 1
fi
export CUDA_VISIBLE_DEVICES="${GPU_UUID}"
export PYTHONUNBUFFERED=1
if [[ -n "${LLAMA_OMNI2_REPO:-}" ]]; then
  export PYTHONPATH="${LLAMA_OMNI2_REPO}:${PYTHONPATH:-}"
fi
echo "[GPU] GPU_ID=${GPU_ID}, CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"

CMD=(
  "${PYTHON_BIN}" -u run_advwave_defense_llamaomni2.py
  --generate_tts
  --model_name "${MODEL_NAME:-}"
  --vector_root "${VECTOR_ROOT:-${REPO_ROOT}/vectors/llamaomni2}"
  --vector_epoch "${VECTOR_EPOCH:-300}"
  --end "${END:-200}"
  --layers ${LAYERS:-12}
  --multipliers -5 -4 -3 -2 -1 0
  --attacks ${ATTACKS:-audio_ours}
  --device cuda:0
)
if [[ -n "${ATTACK_CACHE_DIR:-}" ]]; then
  CMD+=(--attack_cache_dir "${ATTACK_CACHE_DIR}")
fi
"${CMD[@]}"

echo "[DONE] LLaMA-Omni2 AdvWave defense run at $(date)"
echo "[LOG] ${LOG_FILE}"
