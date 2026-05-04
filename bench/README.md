# mellonella-bench

Benchmark + evaluation harness for the mellonella speaker filter. Implements
the dataset / scenario / metric layout defined in
[`../docs/benchmarks.md`](../docs/benchmarks.md) and the protocol described
in [`../docs/evaluation.md`](../docs/evaluation.md).

## Layout

```
bench/
├── pyproject.toml
├── mellonella_bench/
│   ├── datasets/         # downloaders + subset preparation (run locally; not in CI)
│   ├── metrics/          # gating accuracy, NS quality (SI-SDR/PESQ/STOI), attack/release timing
│   ├── scenarios/        # one runner per scenario (Scenario 1 first; 2-6 to follow)
│   └── runners/          # CLI orchestrators (`run_all.py`)
└── tests/                # pytest, lightweight (no datasets, no models)
```

The package depends on `mellonella-poc` for the actual gating pipeline, so
install both together:

```bash
pip install -e poc[dev] -e bench[dev]
```

## Running

```bash
# all wired-up scenarios (currently scenario_1, scenario_3, scenario_6)
mellonella-bench --output benchmark_results/$(date +%Y%m%d_%H%M%S)

# specific scenarios
mellonella-bench --scenarios scenario_1,scenario_6 --output benchmark_results/...

# real pipeline (requires `pip install -e poc[models]`)
mellonella-bench --real-pipeline --output benchmark_results/...
```

`mellonella-bench --help` lists the supported scenarios and flags.

## Scenarios

| ID | What it measures | Status |
|---|---|---|
| scenario_1 | Solo target + noise; SNR sweep with PESQ/STOI/SI-SDR/TPR | wired |
| scenario_2 | Solo other speaker + noise; TNR / FPR | wired |
| scenario_3 | Alternating target ⇄ other; frame accuracy + onset/offset latency | wired |
| scenario_4 | Simultaneous target + other; FP-tolerant behaviour (target-to-other ratio sweep) | wired |
| scenario_5 | Multilingual robustness; per-language TPR/FPR + cross-language stddev | wired |
| scenario_6 | Long-running drift verification (auto-learn) | wired |

## Datasets

The downloaders live in `mellonella_bench/datasets/`. They write under
`$MELLONELLA_DATA_DIR` (default `./data/`). Each downloader is idempotent;
warm caches are reused. See the per-script docstring for licence terms.

### CommonVoice (multi-lingual)

Mozilla CommonVoice (`CC0-1.0`) is the canonical multi-language source for
Phase 2 / Scenario 5 work. URLs are signed and the corpus is multi-GB, so
the script does not auto-download — fetch the per-language tarball
manually from <https://commonvoice.mozilla.org/> and pass its path:

```bash
python -m mellonella_bench.datasets.commonvoice prepare \
    --language ja \
    --archive ~/Downloads/cv-corpus-19.0-2024-09-13-ja.tar.gz
```

This extracts under `$MELLONELLA_DATA_DIR/commonvoice/<lang>/`, picks the
top-K most-clipped speakers (default 10) with N clips each (default 20),
and writes a flat `subset/manifest.csv` for downstream calibration. The
calibrate scripts will be extended to read this manifest in a follow-up.

#### Running scenario_5 against the manifests

`scripts/scenario_5_from_manifest.py` chains the manifest output into the
multilingual robustness scenario. It picks the top-N speakers per
language, materialises target / other / noise wavs to a working dir,
runs `scenario_5`, and emits per-row stats plus a `failures.json` that
lists every (item, SNR, mode) below the configured TPR/FPR thresholds —
non-zero exit when any threshold is violated, so the script is suitable
for hard-fail CI gates.

```bash
python scripts/scenario_5_from_manifest.py \
    --manifest ja=$MELLONELLA_DATA_DIR/commonvoice/ja/subset/manifest.csv \
    --manifest en=$MELLONELLA_DATA_DIR/commonvoice/en/subset/manifest.csv \
    --output benchmark_results/scenario_5/$(date +%Y%m%d_%H%M%S) \
    --tpr-min 0.5 --fpr-max 0.5 \
    --real-pipeline
```

Drop `--real-pipeline` to exercise the wiring with the deterministic stub
(useful for CI smoke tests when CommonVoice data is not available).

## Metrics

| Module | Indicator | Implementation |
|---|---|---|
| `metrics.gating` | TPR / TNR / FPR / FNR / frame accuracy | NumPy only |
| `metrics.ns_quality.si_sdr` | SI-SDR (dB) | NumPy only |
| `metrics.ns_quality.pesq_score` | PESQ | `pesq` (extra) |
| `metrics.ns_quality.stoi_score` | STOI | `pystoi` (extra) |
| `metrics.attack_release` | attack/release time constants | NumPy only |

Tests requiring `pesq` / `pystoi` are tagged `@pytest.mark.metrics` and skipped
unless those extras are installed.

## CSV / JSON output

Per `docs/evaluation.md`, every scenario writes one CSV. `run_all.py` also
writes a `summary.json` with system info, model versions, threshold settings,
and Phase-gate status.
