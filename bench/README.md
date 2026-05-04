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

### MLS — non-English European languages (CI-default)

`facebook/multilingual_librispeech` (`CC-BY 4.0`, ungated) covers seven
European languages — de, fr, es, it, nl, pl, pt — with real per-clip
`speaker_id`. **English was dropped from the HF repo at the parquet
migration**, so `en` is routed through Emilia-YODAS instead. The
`bench[hf]` extra adds `datasets` so the streaming-prep step works
without a local download:

```bash
pip install -e bench[hf]
python -m mellonella_bench.datasets.mls prepare --language de
python -m mellonella_bench.datasets.mls prepare --language fr
# also: es, it, nl, pl, pt
```

Output lands at `$MELLONELLA_DATA_DIR/mls/<lang>/manifest.csv` plus
`speakerNN/*.wav` directories. The CI workflow caches this materially.

### Emilia-YODAS — Asian languages + English (gated, requires `HF_TOKEN`)

`amphion/Emilia-Dataset` (`CC-BY 4.0` for the `Emilia-YODAS` shards;
the parent dataset has a CC-BY-NC half — we explicitly load YODAS
only) covers en / ja / ko / zh-CN / de / fr with real `speaker` labels.
This is also the source CI uses for **English** since MLS dropped its
`english` config.
The repo is HF-gated: agree to the dataset ToS once at
<https://huggingface.co/datasets/amphion/Emilia-Dataset>, generate a
read token, and export it:

```bash
export HF_TOKEN=hf_xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
python -m mellonella_bench.datasets.emilia prepare --language ja
python -m mellonella_bench.datasets.emilia prepare --language ko
python -m mellonella_bench.datasets.emilia prepare --language zh-CN
```

In CI the token is wired via the `HF_TOKEN` repository secret; the
`scenario-5` workflow auto-skips Emilia prep on PRs from forks (where
secrets aren't exposed) but still runs MLS for European coverage.

### CommonVoice — local-only (Mozilla migrated to a paid platform Oct-2025)

Mozilla CommonVoice was the original Scenario 5 source but is no longer
freely re-distributable: as of October 2025 the corpus moved to the
[Mozilla Data Collective](https://datacollective.mozillafoundation.org)
behind a paid access model, and the HuggingFace mirrors were removed.
The `mellonella_bench.datasets.commonvoice` module still works against a
locally-downloaded archive (the user must obtain it themselves) and
remains the recommended path for finer per-speaker calibration:

```bash
python -m mellonella_bench.datasets.commonvoice prepare \
    --language ja \
    --archive ~/Downloads/cv-corpus-19.0-2024-09-13-ja.tar.gz
```

Output is written under `$MELLONELLA_DATA_DIR/commonvoice/<lang>/` with
the same per-speaker manifest schema as MLS / Emilia-YODAS.

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
    --manifest en=$MELLONELLA_DATA_DIR/mls/en/manifest.csv \
    --manifest de=$MELLONELLA_DATA_DIR/mls/de/manifest.csv \
    --manifest ja=$MELLONELLA_DATA_DIR/emilia_yodas/ja/manifest.csv \
    --output benchmark_results/scenario_5/$(date +%Y%m%d_%H%M%S) \
    --tpr-min 0.5 --fpr-max 0.5 \
    --real-pipeline
```

Drop `--real-pipeline` to exercise the wiring with the deterministic stub
(useful for CI smoke tests when no real data is available).

The `.github/workflows/scenario_5.yml` GitHub Action runs this end-to-end
on every push / PR (uses MLS for European coverage; auto-includes
Emilia-YODAS ja/ko/zh-CN when `HF_TOKEN` is configured).

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
