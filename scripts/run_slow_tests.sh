#!/usr/bin/env bash
set -euo pipefail

python -m pytest -q -m "slow and not hardware" "$@"
