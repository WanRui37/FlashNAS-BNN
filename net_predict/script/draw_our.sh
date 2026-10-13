#!/usr/bin/env bash
set -euo pipefail
repo_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$repo_dir"
exec "${PYTHON:-python3}" main.py run --config configs/examples/predictor-imagenet100.json "$@"
