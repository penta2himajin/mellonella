# mellonella-poc (Phase 1 PoC)

Python + PyTorch implementation of the mellonella hard-gating speaker filter.
Scope is the Phase 1 / Phase 2 work described in
[`../docs/implementation.md`](../docs/implementation.md): algorithm validation
and threshold tuning before the Rust port.

## Layout

```
poc/
├── pyproject.toml
├── mellonella_poc/
│   ├── config.py        # default thresholds + Config dataclass
│   ├── gating.py        # target_score, hangover, attack/release envelope
│   ├── enrollment.py    # EmbeddingPool (anchors + auto-learn) + drift checks
│   ├── f0.py            # YIN pitch estimator (NumPy)
│   ├── dfn3.py          # DeepFilterNet 3 wrapper (heavy dep)
│   ├── vad.py           # silero-vad wrapper (heavy dep)
│   ├── embedding.py     # ECAPA-TDNN wrapper (heavy dep)
│   ├── pipeline.py      # offline orchestrator
│   └── cli.py           # `mellonella-poc` CLI entry
└── tests/
    ├── test_gating.py
    ├── test_enrollment.py
    └── test_f0.py
```

The "heavy dep" modules are guarded so the pure-logic core
(`config`, `gating`, `enrollment`, `f0`) imports without torch/speechbrain
installed. CI installs only the lightweight subset.

## Install

```bash
# Lightweight (lint/test only)
pip install -e poc[dev]

# Full stack with models
pip install -e poc[models,dev]
bash scripts/download_models.sh
```

## Run

```bash
# Enroll a target speaker from a 30s recording
mellonella-poc enroll \
    --input target_30s.wav \
    --output enrollment.json

# Process an utterance through the gating pipeline
mellonella-poc process \
    --enrollment enrollment.json \
    --input mixture.wav \
    --output filtered.wav
```

Both commands require the `[models]` extra and downloaded model artefacts.

## Tests

```bash
cd poc
pytest                     # lightweight tests only
pytest -m models           # also run model-backed integration tests
```
