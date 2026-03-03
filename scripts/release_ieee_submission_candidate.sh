#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

PY_BIN="${PYTHON_BIN:-python3}"
if ! command -v "${PY_BIN}" >/dev/null 2>&1; then
  PY_BIN="python"
fi

out_root="${1:-outputs/release_ieee_submission_candidate}"
tag="${2:-}"
mic_mode="${3:-rule}"

if [[ $# -ge 1 ]]; then shift; fi
if [[ $# -ge 1 ]]; then shift; fi
if [[ $# -ge 1 ]]; then shift; fi

if [[ -z "${tag}" ]]; then
  tag="$(date -u +%Y%m%d_%H%M%S)"
fi

"${PY_BIN}" tools/reproduce_ieee_step28.py \
  --out-root "${out_root}" \
  --mic-mode "${mic_mode}" \
  --promote-release \
  --strict-verify \
  --freeze-require-publication-assets \
  --freeze-require-release-assets \
  --guardrails-policy "paper/ieee_2026/guardrails_policy.json" \
  --package-tag "${tag}" \
  "$@"
