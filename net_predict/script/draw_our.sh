#!/usr/bin/env bash
set -euo pipefail
repo_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$repo_dir"
exec "${PYTHON:-python3}" -m flashnas run --config configs/predictor-imagenet100.json "$@"
