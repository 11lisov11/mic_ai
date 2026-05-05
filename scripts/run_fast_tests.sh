#!/usr/bin/env bash
set -euo pipefail

python -m pytest -q -m "not slow and not hardware" "$@"
