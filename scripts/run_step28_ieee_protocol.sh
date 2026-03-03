#!/usr/bin/env bash
set -euo pipefail

OUT_ROOT="${1:-outputs/progress_step28_ieee_v1}"
SEEDS="${SEEDS:-101,202,303,404,505}"
MOTORS="${MOTORS:-air56,al31,ao2}"
SCENARIOS="${SCENARIOS:-speed_step,ramp,load_step,start_stop}"
SEED_PERTURB_LEVEL="${SEED_PERTURB_LEVEL:-0.2}"
MIC_MODE="${MIC_MODE:-rule}"
SKIP_AIR56_TUNE="${SKIP_AIR56_TUNE:-1}"
CHECKPOINT_REGISTRY="${CHECKPOINT_REGISTRY:-config/checkpoint_registry.json}"
PYTHON_BIN="${PYTHON_BIN:-python3}"
if ! command -v "${PYTHON_BIN}" >/dev/null 2>&1; then
  PYTHON_BIN="python"
fi

MODE1_DIR="${OUT_ROOT}/mode1_foc_encoder_vs_mic_sensorless"
MODE2_DIR="${OUT_ROOT}/mode2_foc_sensorless_vs_mic_sensorless"

EXTRA_ARGS=()
if [[ "${SKIP_AIR56_TUNE}" == "1" ]]; then
  EXTRA_ARGS+=(--skip-air56-tune)
fi

echo "[step28] mode1 run"
"${PYTHON_BIN}" tools/step27_pipeline.py \
  --motors "${MOTORS}" \
  --seeds "${SEEDS}" \
  --scenarios "${SCENARIOS}" \
  --out-dir "${MODE1_DIR}" \
  --foc-feedback-mode encoder \
  --mic-feedback-mode sensorless \
  --mic-mode "${MIC_MODE}" \
  --checkpoint-registry "${CHECKPOINT_REGISTRY}" \
  --seed-perturbation \
  --seed-perturb-level "${SEED_PERTURB_LEVEL}" \
  "${EXTRA_ARGS[@]}"

echo "[step28] mode2 run"
"${PYTHON_BIN}" tools/step27_pipeline.py \
  --motors "${MOTORS}" \
  --seeds "${SEEDS}" \
  --scenarios "${SCENARIOS}" \
  --out-dir "${MODE2_DIR}" \
  --foc-feedback-mode sensorless \
  --mic-feedback-mode sensorless \
  --mic-mode "${MIC_MODE}" \
  --checkpoint-registry "${CHECKPOINT_REGISTRY}" \
  --seed-perturbation \
  --seed-perturb-level "${SEED_PERTURB_LEVEL}" \
  "${EXTRA_ARGS[@]}"

echo "[step28] building ieee summary"
"${PYTHON_BIN}" tools/build_step28_ieee_summary.py \
  --mode1-dir "${MODE1_DIR}" \
  --mode2-dir "${MODE2_DIR}" \
  --out-dir "${OUT_ROOT}"

echo "[step28] done: ${OUT_ROOT}/step28_ieee_summary.md"
