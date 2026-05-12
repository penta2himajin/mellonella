# Benchmarks

## Evaluation policy

For PoC-stage performance verification, this combines **minimal and commercially clean** benchmark datasets. This document defines:

- The metrics to evaluate.
- The datasets to adopt and their licences.
- Evaluation scenarios.
- The composition of the minimal eval set.

## Metrics to evaluate

### A. NS (DFN3) quality

| Metric | Description | Range |
|---|---|---|
| PESQ | ITU-T P.862, perceptual speech quality | -0.5..4.5 |
| STOI | Short-Time Objective Intelligibility | 0..1 |
| SI-SDR | Scale-Invariant Signal-to-Distortion Ratio | dB |
| DNSMOS P.835 | Microsoft's non-intrusive perceptual quality metric (SIG / BAK / OVRL) | 1..5 |
| UTMOS | Non-intrusive UTokyo MOS predictor | 1..5 |

### B. VAD accuracy

| Metric | Description |
|---|---|
| Frame-level F1 | Per-frame F1 over speech / non-speech |
| Frame accuracy | Correct frames / total frames |
| Onset / Offset error | Onset / offset time error (ms) |

### C. SV decision accuracy

| Metric | Description |
|---|---|
| EER | Equal Error Rate (target vs non-target) |
| Gating accuracy | Per-frame correct decision rate |
| False Positive rate | Rate at which other speakers get passed |
| False Negative rate | Rate at which the target gets muted |

### D. Integrated pipeline

| Metric | Description |
|---|---|
| All-metric combination | A through C combined |
| Measured total latency | Wall-clock from input to output |
| CPU usage | Single-thread usage |
| Memory footprint | Peak usage at inference |

### E. Subjective evaluation (auxiliary)

PoC-stage listening by self / family / colleagues:

- 5-point MOS rating (1: bad → 5: excellent).
- A/B test: raw audio vs processed; hard-gating vs DFN3 standalone.

## Adopted datasets

### NS-quality evaluation: VoiceBank+DEMAND

Industry-standard SE evaluation benchmark.

- **VoiceBank (VCTK)**: CC BY 4.0 / ODC-By 1.0.
- **DEMAND**: CC BY-SA 3.0.
- **Composition**:
  - Test set: 824 paired utterances, 2 unseen speakers, 5–10 noise types.
  - SNR: 2.5 / 7.5 / 12.5 / 17.5 dB.
  - Usable at either 16 kHz or 48 kHz.
- **Role**: DFN3-standalone NS performance; standard PESQ / STOI / CSIG / CBAK / COVL baseline.

### Multilingual robustness: Mozilla Common Voice

CommonVoice is CC0-licensed and covers 250+ languages — the primary multilingual evaluation dataset for this project.

- **License**: CC0-1.0 (public domain).
- **Constraints**:
  - No re-hosting / no redistribution (in-project use is OK).
  - No attempts to identify speakers.
- **Scale**: 31,841 hours as of v18 (validated 20,789 hours); further expanded in v23 / v24.
- **PoC subset**: extract ~50 utterances each from the main 5–10 languages.
- **Target language candidates**:

| Language | Code | Purpose |
|---|---|---|
| English | en | Standard / baseline |
| Japanese | ja | Primary use language |
| German | de | Indo-European, consonant-heavy |
| French | fr | Indo-European, vowel-strong |
| Chinese | zh-CN | Tonal language |
| Spanish | es | Indo-European, Romance |
| Korean | ko | Agglutinative |
| Arabic | ar | Non-Indo-European |

ECAPA-TDNN is trained on VoxCeleb (multilingual), so it is in principle language-independent; per-language EER verifies real-world performance.

### Auxiliary multilingual: Multilingual LibriSpeech (MLS)

High-quality studio recordings from audiobooks. Utterance quality is more uniform than CommonVoice, useful for reducing evaluation variance.

- **License**: CC0 (public domain; derived from LibriVox + Project Gutenberg).
- **Languages**: 8 (English, German, Dutch, Spanish, French, Italian, Portuguese, Polish).
- **Scale**: English 44.5K hours; the rest combined 6K hours.
- **Distribution**: HuggingFace `facebook/multilingual_librispeech`.
- **PoC use**: extract 30–50 utterances per language from the test split.

### Noise datasets

#### MUSAN (Apache 2.0)

- **License**: Creative Commons (flexible).
- **Scale**: ~60 GB.
- **Composition**: speech / music / noise (3 categories).
- **Role**: generate custom eval sets across diverse noise conditions.
- **Distribution**: OpenSLR (http://www.openslr.org/17/).

#### DEMAND (CC BY-SA 3.0)

- **License**: CC BY-SA 3.0.
- **Composition**: 18 types of real-environment recordings (kitchen, office, park, traffic, etc.).
- **Role**: standard use as VoiceBank+DEMAND, providing realistic environment noise.
- **Derivative works require ShareAlike** (when publishing final outputs).

#### DNS Challenge dataset (CC BY 4.0 / MIT)

- **License**: code MIT, data CC BY 4.0.
- **Scale**: fullband (48 kHz), large-scale.
- **Role**:
  - Part of the dataset DFN3 was trained on.
  - The DNS5 (ICASSP 2023) test set includes personalised tasks.
- **Distribution**: https://github.com/microsoft/DNS-Challenge.

### Multi-speaker scenarios: LibriMix

- **License**: CC BY 4.0 (derived from LibriSpeech).
- **Generation**: script published, can be generated locally.
- **Composition**: 2-speaker / 3-speaker mixtures.
- **WHAM!-free variant**: can be generated without WHAM! using `mix_clean` mode → commercially clean.
- **Role**: quantitative evaluation of simultaneous-speech / alternating-speech scenarios.

### Japanese options

CommonVoice ja is the primary, but the following are usable as reference data:

| Dataset | License | Commercial use |
|---|---|---|
| **CommonVoice (ja)** | CC0 | ✅ recommended |
| JSUT corpus | CC BY-SA 4.0 (text/labels); audio requires individual negotiation | △ via TLO |
| JVS corpus | Same as above | △ via TLO |
| ReazonSpeech | CDLA-Sharing-1.0 | ⚠️ research-focused |

JSUT / JVS require an individual contract via the University of Tokyo TLO. With commercial deployment in mind, this project adopts **only CommonVoice ja**.

### Datasets not adopted

Excluded by prior investigation:

- **WHAM!**: CC BY-NC 4.0, non-commercial only.
- **WSJ0**: LDC proprietary, paid.
- **VoxCeleb1/2**: training data for ECAPA-TDNN, but BBC / YouTube-derived → grey zone. Use restricted to evaluation only.

## Evaluation scenarios

### Scenario 1: solo target speaker + noise

```
Input:    target-speaker audio + environment noise (MUSAN / DEMAND)
Expected: gate passes; noise-suppressed target audio
```

Metrics:
- PESQ, STOI, SI-SDR (NS quality).
- True Positive rate (gating).
- Onset / Offset error (VAD accuracy).

### Scenario 2: solo other speaker + noise

```
Input:    other-speaker audio + environment noise
Expected: gate mutes; silent output
```

Metrics:
- True Negative rate (gating).
- False Positive rate (rate of incorrect passes).

### Scenario 3: alternating speech (target → other → target)

```
Input:    target speaks → silence → other speaks → silence → target speaks
Expected: only target segments pass; other segments mute
```

Metrics:
- Frame-level accuracy.
- Attack time (transition time to pass).
- Release time (transition time to mute).

### Scenario 4: simultaneous speech (target + other)

```
Input:    target and other speaking simultaneously
Expected: pass (FP-tolerant policy); preserve target intelligibility
```

Metrics:
- Subjective rating (target intelligibility).
- Objective SI-SDR (distortion of the target audio).

### Scenario 5: multilingual robustness

```
Input:    per-language target audio + noise (cross-language)
Expected: stable SV decisions regardless of language
```

Metrics:
- Per-language EER.
- Per-language gating accuracy.
- Cross-language variation statistics (standard deviation).

#### Initial baseline (PR #17, real pipeline, MLS + Emilia-YODAS, 6 languages)

| Lang | TPR mean | FPR mean | Notes |
|---|---|---|---|
| de | 0.77 | 0.02 | ◎ |
| en | 0.78 | 0.00 | ◎ |
| fr | 0.80 | 0.00 | ◎ |
| ko | 0.80 | 0.00 | ◎ |
| ja | 0.67 | 0.07 | △ TPR drops at low SNR (FN bias) |
| zh-CN | 0.86 | 0.23 | △ FPR rises at low SNR (FP bias) |

Aggregate: TPR mean 0.78, cross-language stddev 0.058; FPR mean 0.05, stddev 0.084.

The ja / zh-CN symmetric failures show that **no joint optimisation is possible with a single global `θ_pass = 0.30`**. In response, `docs/decisions.md` D-010 decided to adopt **AS-Norm**. See also `docs/gating.md`'s "Score normalisation with AS-Norm (Adaptive S-Norm)" section.

#### Phase 2: deltas after AS-Norm (PR #19, default `theta_pass_as_norm = 1.5`, cohort 30 embeddings)

| Lang | Δ TPR | Δ FPR | Notes |
|---|---|---|---|
| de | **−0.08** | +0.02 | New regression; TPR drops to 0.48 at SNR = 0 |
| en | +0.02 | 0 | Slight improvement |
| fr | +0.03 | +0.03 | TPR slightly up, FPR roughly unchanged |
| ja | **+0.18** | +0.05 | ✅ Resolves the primary drop problem |
| ko | +0.06 | 0 | Improved |
| zh-CN | +0.01 | **+0.19** | ❌ FPR worsens significantly, 0.54 at SNR = 0 |

Aggregate: TPR mean **+0.04** / FPR mean **+0.05** / FPR cross-language stddev **+0.06**.

ja improves dramatically; zh-CN worsens. This shows the default `theta_pass_as_norm = 1.5` is too heuristic. See `docs/decisions.md` D-010 "Phase 2 observation" and the Phase 3 plan (data-driven calibration via the calibrate.py extension).

### Scenario 6: adaptation to time-varying changes (auto-learning effect)

```
Input:    same speaker with different voice-quality changes
          (cold / fatigue emulation, different-emotion utterances)
Expected: auto-learn pool updates; decision accuracy maintained
```

Metrics:
- Gating-accuracy trend over time.
- Anchor distance (drift detection).

## Minimal eval set (PoC stage)

Minimum configuration targeting < 1 hour runtime in PoC:

### Sample counts

| Dataset | Samples | Purpose |
|---|---|---|
| VoiceBank+DEMAND test | 100 utterances (random) | NS quality baseline (part of Scenario 1) |
| CommonVoice 5 languages | 50 utterances each = 250 | Multilingual robustness (Scenario 5) |
| MLS 5 languages | 30 utterances each = 150 | High-quality multilingual auxiliary |
| MUSAN | 10 noises per category = 30 | Noise for custom mixtures |
| LibriMix mix_clean test | 100 mixtures | Multi-speaker scenarios (Scenarios 2 / 3 / 4) |

Total: ~630 evaluation samples.

### Evaluation run composition

In PoC stage:

1. **Scenario 1 (solo + noise)**: VoiceBank+DEMAND primary, MUSAN auxiliary.
2. **Scenarios 2 / 3 / 4 (multi-speaker)**: LibriMix primary.
3. **Scenario 5 (multilingual)**: cross-use CommonVoice + MLS.
4. **Scenario 6 (auto-learning)**: long-duration test with self-recorded audio.

Evaluation runs in batch; results are written as CSV:

```
benchmark_results/
├── scenario_1_solo_noise.csv
├── scenario_2_other_speaker.csv
├── scenario_3_alternating.csv
├── scenario_4_simultaneous.csv
├── scenario_5_multilingual.csv
├── scenario_6_drift.csv
└── summary.json
```

## Comparison targets

To put the hard-gating performance into perspective, compare against the following.

### Baselines

1. **Unprocessed raw audio**: lower-bound baseline.
2. **DFN3 standalone**: NS only, no SV — pure measurement of the NS effect.
3. **Oracle VAD (ground truth)**: upper bound when given perfect VAD info.

### Comparison with existing methods

| Method | Use | Expected result |
|---|---|---|
| ConVoiFilter (offline) | Reference for true TSE | Achieves higher separation accuracy but with 5 s latency |
| ESPnet TD-SpeakerBeam | Near-causal TSE | Performance / latency trade-off evaluation |

These serve as reference points for "is the hard-gating type achieving accuracy comparable to a real-time-capable TSE?"

## Benchmark tools

Evaluation libraries used in the implementation:

| Use | Library | License |
|---|---|---|
| PESQ | `pesq` (PyPI) | MIT |
| STOI | `pystoi` (PyPI) | MIT |
| SI-SDR | `torchmetrics` or hand-rolled | Apache 2.0 |
| DNSMOS | Microsoft's P.835 ONNX model | MIT |
| UTMOS | UTokyo's MOS predictor | BSD-3 |
| SV (EER computation) | `speechbrain.utils.metric_stats` | Apache 2.0 |

## Benchmark-execution automation

Place the following under the `bench/` directory (built at the implementation stage):

```
bench/
├── datasets/                # dataset-download scripts
│   ├── download_vbd.sh
│   ├── download_commonvoice.py
│   ├── download_mls.py
│   ├── download_musan.sh
│   └── generate_librimix_clean.py
├── scenarios/               # per-scenario evaluation scripts
│   ├── scenario_1_solo_noise.py
│   ├── scenario_2_other_speaker.py
│   ├── scenario_3_alternating.py
│   ├── scenario_4_simultaneous.py
│   ├── scenario_5_multilingual.py
│   └── scenario_6_drift.py
├── metrics/                 # metric computation
│   ├── ns_quality.py        # PESQ, STOI, SI-SDR, DNSMOS
│   ├── vad_accuracy.py
│   ├── sv_eer.py
│   └── gating_accuracy.py
├── runners/
│   └── run_all.py           # run all scenarios in one go
└── results/                 # output target
    └── (CSV, JSON, plots)
```

## Recommended execution order

1. **Immediately after Phase 1 PoC completion**: Scenarios 1 + 5 (NS quality and multilingual robustness).
2. **After Phase 2 auto-learning implementation**: Scenario 6 (drift validation).
3. **After Phase 3 Rust port**: all scenarios + measured latency / CPU.
4. **After Phase 4 mobile deployment**: on-device latency / battery measurement on mobile.

For each phase, the gate condition before progression is to clear the minimum criterion of the corresponding scenarios. Concrete thresholds are fixed after the Phase 1 initial measurement.

## Rust pipeline micro-benchmarks (Phase 3 closeout)

Run with `cargo bench --bench pipeline_stages` from `rust/`. ONNX-dependent benches need `MELLONELLA_ECAPA_ONNX`, `MELLONELLA_VAD_ONNX`, `MELLONELLA_DFN3_ONNX`, and `ORT_DYLIB_PATH`.

Measured on Linux x86_64 (dev VM, 2-vCPU class), ONNX Runtime 1.25.1, `dev` profile = `release` (criterion default).

| Stage | Input | Wall time (median) | RTF (real-time factor) |
|---|---|---|---|
| `Fbank::compute` | 1 s @ 16 kHz | **504 µs** | ~1985× |
| `SileroVad::score` | 32 ms chunk @ 16 kHz | **168 µs** | ~190× |
| `EcapaTdnn::embed_features` | 100 frames × 80 mels (≈ 1 s speech window) | **41.2 ms** | ~24× |
| `Dfn3Pipeline::process` | 1 chunk (1.02 s @ 48 kHz) | **28.4 ms** | ~36× |
| `process_offline` (no DFN3) | 2 s @ 16 kHz, AS-Norm off, auto-learn off | **389 ms** | ~5.1× |

Interpretation:

- **Per-stage cost is dominated by ECAPA.** A single embedding refresh costs ~41 ms; the pipeline refresh cadence (250 ms = 4 000 samples) keeps that comfortably below real-time.
- **DFN3 fits in budget too** at ~36× real-time per 1-s chunk. Wall-clock cost grows linearly with chunk count for longer audio.
- **End-to-end pipeline at ~5× real-time** is the slowest path because YIN F0 (`estimate_f0_track`) runs on a 1-s window each refresh and has an O(N²)-flavoured inner loop. That's the obvious place to optimise if a real-time variant is needed — Phase 3.5.

These numbers establish the **Phase 3 perf baseline** for future regression tracking. They do not yet claim parity with the Python reference (Python timings haven't been measured under matching conditions). Add a Python-side bench harness in Phase 3.5 to close that gap.

### Profile hints for Phase 3.5

- F0 estimation: cache the FFT plan, prune `tau` search space against the previous frame's estimate, or replace YIN with a lighter-weight pitch heuristic when only `f0_match` (not the absolute pitch) is needed.
- ECAPA: optimisation levels and provider selection in `ort` (CoreML / DirectML / TensorRT-EP) likely move the needle on production hardware.
- DFN3: single-chunk-only is the current cap; streaming-overlap would amortise the per-chunk model load + state warmup.

## Rust ↔ Python cross-implementation comparison (Phase 3 sign-off)

Beyond per-component parity (cosine 2 × 10⁻⁷, Fbank 1 mdB, VAD 1 × 10⁻³, gate state byte-equal, DFN3 audio 1.5 × 10⁻²), the Rust deliverable is also exercised end-to-end against the Python reference via `scripts/rust_scenario_1.py`. The harness:

1. Builds a synthetic target + noise mixture at fixed SNR
2. Runs both Python (`mellonella_poc.pipeline.process_offline`) and Rust (`mellonella process` via subprocess, with the new `--gate-decisions` JSON output)
3. Compares per-VAD-frame gate state, gate duty cycle, audio RMS / peak, and `gate_agreement` (fraction of frames where the two pipelines agree on the binary gate decision)

On the bundled synthetic mixture (3 s @ 16 kHz, 200 Hz harmonic stack target, white noise at SNR = 10 dB), both pipelines emit `gate_agreement = 1.00` over 93 VAD frames. The synth doesn't read as speech to silero-vad at the default `vad_threshold = 0.5`, so both pipelines correctly mute the entire output — a trivial-but-correct agreement.

For non-trivial sign-off, run the harness with real recordings:

```
ORT_DYLIB_PATH=… MELLONELLA_ECAPA_ONNX=… MELLONELLA_VAD_ONNX=… \
  python scripts/rust_scenario_1.py  # synth (default)
# or pass --target / --noise / --enroll arguments once those flags
# land in a follow-up; for now edit the script's synth_* calls.
```

Real-data sign-off (scenario 1 LibriSpeech + MUSAN; scenarios 4-6 multilingual / drift) is the obvious Phase 3.5 expansion — the harness pattern transfers directly.
