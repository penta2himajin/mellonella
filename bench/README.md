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
# fast subset (Scenario 1 only)
mellonella-bench --quick --output benchmark_results/$(date +%Y%m%d_%H%M%S)

# specific scenarios
mellonella-bench --scenarios scenario_1 --output benchmark_results/...
```

`mellonella-bench --help` lists the supported scenarios and flags.

## Datasets

The downloaders live in `mellonella_bench/datasets/`. They write under
`$MELLONELLA_DATA_DIR` (default `./data/`). Each downloader is idempotent;
warm caches are reused. See the per-script docstring for licence terms.

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
