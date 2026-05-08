#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="${ROOT_DIR:-${SCRIPT_DIR}}"
REPO_ROOT="$(cd "${ROOT_DIR}/../.." && pwd)"
cd "${ROOT_DIR}"

RUN_TS="${RUN_TS:-$(date +%Y%m%d_%H%M%S)}"
GPU_ID="${GPU_ID:-4}"
MIN_GPU_FREE_MB="${MIN_GPU_FREE_MB:-50000}"
GPU_CHECK_INTERVAL_SEC="${GPU_CHECK_INTERVAL_SEC:-60}"
PYTHON_BIN="${PYTHON_BIN:-python}"

DATASET_CSV="${DATASET_CSV:-${REPO_ROOT}/data/advbench_infer_200.csv}"
ATTACK_CACHE_DIR="${ATTACK_CACHE_DIR:-${ROOT_DIR}/output/llamaomni2/attack_cache/audio_ours_A_ep3000_suf48000_lr001_n200_dedup_${RUN_TS}}"
LOG_FILE="${LOG_FILE:-${ROOT_DIR}/logs/attack_cache_A_200_dedup_${RUN_TS}.log}"
PID_FILE="${LOG_FILE}.pid"

mkdir -p logs "${ATTACK_CACHE_DIR}"
echo "$$" > "${PID_FILE}"

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

export SILICONFLOW_API_KEY="${SILICONFLOW_API_KEY:-}"
export SILICONFLOW_BASE_URL="${SILICONFLOW_BASE_URL:-https://api.siliconflow.cn/v1}"
export SILICONFLOW_MODEL="${SILICONFLOW_MODEL:-deepseek-ai/DeepSeek-V3.2}"

{
  echo "[START] $(date '+%F %T') LLaMA-Omni2 200-prompt dedup attack cache config A"
  echo "GPU_ID=${GPU_ID}, GPU_UUID=${GPU_UUID}, min_free_mb=${MIN_GPU_FREE_MB}"
  echo "DATASET_CSV=${DATASET_CSV}"
  echo "ATTACK_CACHE_DIR=${ATTACK_CACHE_DIR}"
  echo "CONFIG suffix_length=48000 num_epochs=3000 lr=0.01 samples=200"
} | tee -a "${LOG_FILE}"

while true; do
  free_mb="$(nvidia-smi --query-gpu=index,memory.free --format=csv,noheader,nounits | awk -F, -v idx="${GPU_ID}" '$1 + 0 == idx {gsub(/ /, "", $2); print $2}')"
  echo "[$(date '+%F %T')] GPU${GPU_ID} free=${free_mb:-unknown} MiB" | tee -a "${LOG_FILE}"
  if [[ "${free_mb:-0}" -ge "${MIN_GPU_FREE_MB}" ]]; then
    break
  fi
  sleep "${GPU_CHECK_INTERVAL_SEC}"
done

"${PYTHON_BIN}" -u run_advwave_defense_llamaomni2.py \
  --generate_tts \
  --dataset_csv "${DATASET_CSV}" \
  --dedup_prompts \
  --reuse_existing_tts_by_prompt \
  --attack_only \
  --attack_cache_dir "${ATTACK_CACHE_DIR}" \
  --model_name "${MODEL_NAME:-}" \
  --attacks audio_ours \
  --start 0 \
  --end 200 \
  --num_epochs 3000 \
  --suffix_length 48000 \
  --universal_size 10 \
  --lr 0.01 \
  --max_new_tokens 200 \
  --seed 0 \
  --device cuda:0 \
  2>&1 | tee -a "${LOG_FILE}"

status="${PIPESTATUS[0]}"
echo "[END] $(date '+%F %T') status=${status}" | tee -a "${LOG_FILE}"
exit "${status}"
