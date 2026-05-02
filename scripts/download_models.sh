#!/usr/bin/env bash
#
# Pre-download the model artefacts used by the Phase 1 PoC pipeline.
# Idempotent: skips models that are already cached. Safe on a warm CI cache.
#
# Targets (paths under $MELLONELLA_MODEL_DIR, default ./models):
#   - speechbrain/spkrec-ecapa-voxceleb  (~22 MB, ECAPA-TDNN)
#   - silero-vad ONNX bundle             (~2 MB)
#   - DeepFilterNet 3 weights are fetched on first import by `df.enhance`.
#
# Usage:
#   scripts/download_models.sh
#   MELLONELLA_MODEL_DIR=/tmp/models scripts/download_models.sh

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
MODEL_DIR="${MELLONELLA_MODEL_DIR:-$REPO_ROOT/models}"
mkdir -p "$MODEL_DIR"

ECAPA_DIR="$MODEL_DIR/spkrec-ecapa-voxceleb"
if [[ -f "$ECAPA_DIR/embedding_model.ckpt" ]]; then
    echo "[download_models] ECAPA-TDNN already present at $ECAPA_DIR" >&2
else
    echo "[download_models] fetching ECAPA-TDNN (speechbrain/spkrec-ecapa-voxceleb)" >&2
    python3 - "$ECAPA_DIR" <<'PY'
import sys
from speechbrain.inference.speaker import EncoderClassifier
EncoderClassifier.from_hparams(
    source="speechbrain/spkrec-ecapa-voxceleb",
    savedir=sys.argv[1],
    run_opts={"device": "cpu"},
)
PY
fi

VAD_DIR="$MODEL_DIR/silero-vad"
if [[ -f "$VAD_DIR/silero_vad.onnx" ]]; then
    echo "[download_models] silero-vad already present at $VAD_DIR" >&2
else
    echo "[download_models] fetching silero-vad (ONNX)" >&2
    mkdir -p "$VAD_DIR"
    python3 - "$VAD_DIR/silero_vad.onnx" <<'PY'
import shutil, sys
from silero_vad.utils_vad import init_jit_model  # noqa: F401  -- ensures package is importable
import importlib.resources as ir

with ir.as_file(ir.files("silero_vad").joinpath("data/silero_vad.onnx")) as src:
    shutil.copyfile(src, sys.argv[1])
PY
fi

echo "[download_models] done." >&2
echo "MELLONELLA_MODEL_DIR=$MODEL_DIR"
