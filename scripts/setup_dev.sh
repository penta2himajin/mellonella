#!/usr/bin/env bash
#
# Bootstrap a local development environment for the Phase 1 PoC.
# Idempotent: skips work that's already done so it is safe on a warm cache.
#
# Usage:
#   scripts/setup_dev.sh                 # install lightweight dev extras only
#   WITH_MODELS=1 scripts/setup_dev.sh   # also install torch + speechbrain stack

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

if [[ -z "${VIRTUAL_ENV:-}" ]]; then
    if [[ ! -d .venv ]]; then
        echo "[setup_dev] creating virtualenv at .venv" >&2
        python3 -m venv .venv
    fi
    # shellcheck source=/dev/null
    source .venv/bin/activate
fi

python -m pip install --upgrade pip wheel >/dev/null

if [[ "${WITH_MODELS:-0}" == "1" ]]; then
    echo "[setup_dev] installing poc[models,dev]" >&2
    pip install -e "poc[models,dev]"
else
    echo "[setup_dev] installing poc[dev] (no model deps)" >&2
    pip install -e "poc[dev]"
fi

echo "[setup_dev] done. Activate with: source .venv/bin/activate" >&2
