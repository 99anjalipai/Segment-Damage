#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LOG_DIR="${ROOT_DIR}/logs"
mkdir -p "${LOG_DIR}"

MODE="${1:-full}"
OUTPUT_ROOT="outputs/baseline_optimizations"
EPOCHS_OVERRIDE_ARGS=()

if [[ "${MODE}" == "test" ]]; then
  OUTPUT_ROOT="outputs/baseline_optimizations_test"
  EPOCHS_OVERRIDE_ARGS=(--epochs-override 1)
fi

TIMESTAMP="$(date +"%Y%m%d_%H%M%S")"
LOG_FILE="${LOG_DIR}/baseline_optimizations_${TIMESTAMP}.log"

cd "${ROOT_DIR}"

echo "[Start] Baseline optimization sweep"
echo "[Info] Mode: ${MODE}"
echo "[Info] Log file: ${LOG_FILE}"
echo "[Info] Output root: ${OUTPUT_ROOT}"

if [[ "${MODE}" == "test" ]]; then
  python tools/run_baseline_optimizations.py \
    --base-config configs/week1_unet.yaml \
    --experiments-config configs/baseline_optimizations.yaml \
    --output-root "${OUTPUT_ROOT}" \
    --evaluate-splits train val test \
    --comparison-split val \
    --comparison-visualize-samples 3 \
    --continue-on-error \
    --epochs-override 1 \
    2>&1 | tee "${LOG_FILE}"
else
  python tools/run_baseline_optimizations.py \
    --base-config configs/week1_unet.yaml \
    --experiments-config configs/baseline_optimizations.yaml \
    --output-root "${OUTPUT_ROOT}" \
    --evaluate-splits train val test \
    --comparison-split val \
    --comparison-visualize-samples 3 \
    --continue-on-error \
    2>&1 | tee "${LOG_FILE}"
fi

echo "[Done] Finished. Summary: ${OUTPUT_ROOT}/summary.json"
