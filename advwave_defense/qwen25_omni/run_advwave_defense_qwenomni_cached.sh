#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="${ROOT_DIR:-${SCRIPT_DIR}}"
REPO_ROOT="$(cd "${ROOT_DIR}/../.." && pwd)"
cd "${ROOT_DIR}"

PYTHON_BIN="${PYTHON_BIN:-python}"
MODEL_NAME="${MODEL_NAME:-}"
VECTOR_ROOT="${VECTOR_ROOT:-${REPO_ROOT}/vectors/qwen25_omni}"
VECTOR_EPOCH="${VECTOR_EPOCH:-400}"
LAYERS="${LAYERS:-13}"
OUTPUT_DIR="${OUTPUT_DIR:-${ROOT_DIR}/output/qwenomni_defense_layer13_ep400}"

"${PYTHON_BIN}" -u run_advwave_defense_qwenomni_cached.py \
  --model_name "${MODEL_NAME}" \
  --vector_root "${VECTOR_ROOT}" \
  --vector_epoch "${VECTOR_EPOCH}" \
  --layers ${LAYERS} \
  --multipliers -5 -4 -3 -2 -1 0 \
  --test_csv "${REPO_ROOT}/data/advbench_infer_200.csv" \
  --output_dir "${OUTPUT_DIR}" \
  --device cuda:0
