#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname -- "${BASH_SOURCE[0]}")"
# A single explicitly selected experiment; nothing starts implicitly.
exec "${PYTHON:-python3}" -m flashnas "$@"
