# Evaluation

This document defines **how to run, how to judge, and how to record** the datasets, scenarios, and metrics specified in [benchmarks.md](benchmarks.md).

## Scope

| Document | Role |
|---|---|
| benchmarks.md | What to measure with what (selection of datasets / scenarios / metrics) |
| **evaluation.md (this doc)** | How to run / judge / record (protocol / pass-fail criteria / result management) |

## Evaluation protocol

### Overall workflow

```
[1] Evaluation environment setup
    │
    ▼
[2] Dataset preparation (download / preprocessing / mixing)
    │
    ▼
[3] Model enrollment (generate embeddings from explicit-enrollment audio)
    │
    ▼
[4] Run each scenario (full batch)
    │
    ▼
[5] Metric computation (PESQ, STOI, EER, etc.)
    │
    ▼
[6] CSV / JSON output
    │
    ▼
[7] Report generation (markdown + plots)
    │
    ▼
[8] Pass/fail check (against phase-gate criteria)
```

### Per-scenario execution steps

#### Scenario 1: Solo Target + Noise

```
INPUT:
  target_speaker_audio (clean) + noise (MUSAN / DEMAND, SNR ∈ {-5, 0, 5, 10, 15, 20} dB)
  enrollment_audio (target speaker, 30 s)

EXECUTE:
  1. Generate the embedding pool from enrollment.
  2. Generate mixed audio at each SNR.
  3. Run through the pipeline (DFN3 → decision → output).
  4. Save the output audio.

MEASURE:
  - PESQ, STOI, SI-SDR (output vs ground-truth target)
  - DNSMOS P.835 (output alone)
  - True Positive Rate (frame-level, pass rate on speech frames)
  - RMS of the output (confirm it is not muted)
```

#### Scenario 2: Solo Other Speaker + Noise

```
INPUT:
  other_speaker_audio (a non-target speaker) + noise
  enrollment_audio (target speaker, audio of a different person)

EXECUTE:
  1. Run through the pipeline.
  2. Save the output audio.

MEASURE:
  - True Negative Rate (mute rate on speech frames)
  - False Positive Rate (rate of incorrect passes)
  - Output RMS (sufficiently attenuated?)
  - SI-SDR (output vs zero) → larger is better (well muted)
```

#### Scenario 3: Alternating Speech

```
INPUT:
  Concatenated audio: target → silence → other → silence → target → silence → other ...
  Each segment 3-5 s, total 30-60 s
  Silence segments 0.5-2 s
  enrollment_audio (target)

EXECUTE:
  1. Run through the pipeline.
  2. Record frame-level gate decisions.
  3. Compare against ground-truth labels (each frame: target / other / silence).

MEASURE:
  - Frame-level accuracy
  - Confusion matrix (target / other / silence × pass / mute)
  - Onset latency (ms from target onset to pass established)
  - Offset latency (ms from other onset to mute established)
  - Measured attack / release time constants (theoretical: attack = 15 ms, release = 100 ms)
```

#### Scenario 4: Simultaneous Speech

```
INPUT:
  target_speaker_audio + other_speaker_audio (simultaneous)
  enrollment_audio (target)
  Mixing ratios: target:other ∈ {0:1, 1:3, 1:1, 3:1, 1:0}

EXECUTE:
  1. Mix at each ratio.
  2. Run through the pipeline.
  3. Save the output.

MEASURE:
  - SI-SDR (output vs target-only ground truth)
  - Subjective rating (target intelligibility, 5-point scale)
  - Pass threshold by target:other ratio (above how many dB does the target-speaker component pass?)
  - Does the FP-tolerant policy work (does it pass when the target-speaker component is present)?
```

#### Scenario 5: Multilingual Robustness

```
INPUT:
  Per-language target-speaker audio (en, ja, de, fr, zh, es) + noise
  Per language: 50 utterances × different speakers
  enrollment: 30 s enrollment per speaker per language

EXECUTE:
  1. Run enrollment per language.
  2. Same pipeline; evaluation equivalent to Scenario 1.

MEASURE:
  - Per-language EER
  - Per-language gating accuracy
  - Cross-language variation (standard deviation)
  - Cross-language variation of DNSMOS
```

#### Scenario 6: Drift Verification

```
INPUT:
  Long-duration audio of the target speaker (30-60 min)
  Audio emulates time-varying changes (cold / fatigue / different-emotion utterance segments)
  enrollment: only the first 30 s
  auto-learning: ON

EXECUTE:
  1. Explicit enrollment from the first 30 s.
  2. Process subsequent audio in real-time simulation.
  3. Record the auto-learn-pool admission history.
  4. Record `anchor_distance` over time.

MEASURE:
  - Gating-accuracy trend over time
  - Fluctuation of auto_learn_pool size
  - Trend of the anchor_distance median
  - Number of drift-detection triggers
  - Whether resets are triggered
```

### I/O format

#### Input data naming convention

```
data/
├── target_speakers/
│   └── {speaker_id}/
│       ├── enrollment.wav           # 30 s+ clean recording
│       └── test/
│           ├── utt_001.wav
│           ├── utt_002.wav
│           └── ...
├── other_speakers/                  # same structure
├── noise/
│   ├── musan/
│   │   ├── speech/
│   │   ├── music/
│   │   └── noise/
│   └── demand/
│       ├── kitchen/
│       ├── office/
│       └── ...
└── mixtures/                        # generated per scenario
    ├── scenario_1/
    ├── scenario_2/
    └── ...
```

#### Output CSV format (shared)

```csv
sample_id,scenario,language,snr_db,target_speaker,other_speaker,
gate_tpr,gate_tnr,gate_fpr,gate_fnr,
pesq,stoi,si_sdr,dnsmos_sig,dnsmos_bak,dnsmos_ovrl,
attack_ms,release_ms,
processing_time_ms,
notes
```

#### Output JSON summary

```json
{
  "evaluation_id": "eval_20260501_153045",
  "git_commit": "abc1234",
  "model_versions": {
    "dfn3": "0.5.6",
    "silero_vad": "5.1",
    "ecapa_tdnn": "speechbrain-1.0.0"
  },
  "system_info": {
    "platform": "Linux x86_64",
    "cpu": "AMD Ryzen 9 5950X",
    "ram_gb": 64,
    "python_version": "3.11.5"
  },
  "thresholds": {
    "theta_pass": 0.50,
    "theta_learn": 0.80,
    "alpha": 0.8,
    "beta": 0.2
  },
  "scenarios": {
    "scenario_1": { "n_samples": 100, "tpr_mean": 0.94, ... },
    "scenario_2": { "n_samples": 100, "tnr_mean": 0.92, ... },
    ...
  },
  "phase_gate_status": {
    "phase_1_eligible": true,
    "phase_2_eligible": false,
    "blocking_criteria": ["scenario_3_frame_accuracy"]
  }
}
```

## Hard-gating-specific evaluation perspective

There are aspects general SE / TSE metrics alone cannot capture; the following are hard-gating-specific concerns.

### TPR / FPR balance

Because the FP-tolerant policy is in effect:

- **Maximising TPR is top priority**: do not cut the target speaker.
- **FPR is secondary**: some other-speaker leakage is acceptable.

Always retain the confusion matrix during evaluation and monitor the TPR / FPR ratio. Detect situations like "TPR is high but FPR ≈ 1.0" (effectively no gating).

### Verifying attack / release behaviour

Measure deviation from theoretical values (attack = 15 ms, release = 100 ms):

- Steep transitions produce click sounds → `attack < 5 ms` is a warning.
- An overly slow release leaks other speakers → `release > 300 ms` is a warning.
- Frequency-response evaluation: feed a rectangular pulse and inspect the rise / fall waveform of the response.

### SV stability on short utterances

Measure the SV-instability phenomenon on 200-500 ms utterances such as "uh-huh" or "yes":

- Record EER separately for each utterance length (200 ms, 500 ms, 1 s, 2 s, 5 s).
- Expected: EER worsens as utterance shortens; verify the acceptable range.

### Simultaneous-speech behaviour

The target intelligibility evaluated in Scenario 4 is hard to measure with objective metrics. SI-SDR looks at **distortion against the target-only ground truth**, so it can still be computed when other-speaker leakage is present, but include the following supplementary information:

- RMS ratio of the target-speaker audio (target-derived vs other-derived audio in the output).
- Spectral overlap (frequency-band overlap between target and other).
- Subjective rating (intelligibility).

### Long-term drift behaviour

In Scenario 6, track the following long-term metrics:

- **Time evolution of `anchor_distance`**: monotonic increase = drift warning.
- **`auto_learn_pool` turnover rate**: frequency at which old embeddings are dropped from the FIFO.
- **Number of reset triggers**: frequent triggers mean the drift-mitigation thresholds need review.

## Anticipated failure modes

Before running evaluations, enumerate "what could break" and intentionally test each failure mode.

| Failure Mode | Expected scenario | Verification |
|---|---|---|
| SV-decision degradation in high-noise environments | SNR < 0 dB | Confirm via Scenario 1 SNR sweep |
| SV degradation under reverberation | Large halls, bathrooms | Test via RIR convolution |
| Passes other speakers with similar voiceprints | Same-gender family / siblings | Scenario 2 with known similar-speaker pairs |
| Misjudgment on short backchannels | "uh-huh", "yes" | Short-utterance EER evaluation |
| Cross-language performance differences | Non-English | Confirm via Scenario 5 |
| Drift of the auto-learn pool | Long-running operation | Confirm via Scenario 6 |
| Dependence on enrollment audio quality | Low SNR / poor recording quality | Enrollment SNR sweep |
| Mic characteristic differences | Different mics | Mic-change test at inference |
| Speaker health changes | Cold, hoarseness | Time-varying simulation data |

For each failure mode:

1. Predict the expected impact range (which metric degrades by how much) ahead of time.
2. Verify the deviation between prediction and measurement.
3. Significant deviations trigger failure-mode re-evaluation.

## Baseline comparison and interpretation

### Comparison targets

| Comparison target | Role | Expected result |
|---|---|---|
| Unprocessed raw audio | Lower bound | Worst case; if anything is worse there's a fundamental problem |
| DFN3 standalone | Pure measurement of the NS effect | NS portion equivalent; TPR / FPR differ depending on whether SV is present |
| Oracle VAD (ground truth) | Upper bound | TPR = 1, TNR = 1; implementation upper bound |
| ConVoiFilter (offline) | Reference for true TSE | Wins on simultaneous-speech scenes, but with 5 s latency |
| ESPnet TD-SpeakerBeam | Near-causal TSE | Reference for the performance / latency trade-off |

### Interpretation guidelines

| Observation | Interpretation |
|---|---|
| hard-gating ≈ DFN3 standalone + perfect VAD | As expected; SV portion is functioning well |
| hard-gating < DFN3 standalone | SV is misjudging and cutting the target; consider loosening the threshold |
| hard-gating ≈ ConVoiFilter (alternating-speech scenes) | Optimal for the hard-gating type |
| hard-gating << ConVoiFilter (simultaneous-speech scenes) | As expected; the hard-gating limit |

### Pitfall in "true-TSE comparison"

ConVoiFilter and similar systems win on simultaneous-speech scenes, but distinguish:

- **Simultaneous-speech-only datasets**: ConVoiFilter is significantly ahead.
- **Real-call scenarios (mostly alternating speech)**: hard-gating is equal or better.

Without comparing on scenarios that reflect real-call behaviour, hard-gating is unfairly underrated.

## Subjective-evaluation protocol

### Listening environment standardisation

- Headphones: closed-back; no quality requirement, but identical across evaluators.
- Listening volume: normalised to -23 LUFS.
- Ambient noise: quiet environment (< 30 dB SPL).

### Evaluators

PoC stage:
- The developer themself.
- 1-2 family members.
- 1-2 colleagues (if possible).

Production stage:
- A/B test with 5-10 third-party listeners.

### Items

A/B test format:

```
Subjects: original audio vs processed audio; hard-gating vs DFN3 standalone

Questions:
  Q1. Is the target speaker's voice clear?            (1: not at all / 5: very)
  Q2. Are other speakers' voices distracting?         (1: very       / 5: not at all)
  Q3. Does the target voice sound unnatural?          (1: strongly   / 5: not at all)
  Q4. How much noise remains?                         (1: a lot      / 5: none)
  Q5. Is it easy to listen to as a call partner?      (1: hard       / 5: easy)
```

5-10 pairs per sample, averaged as MOS.

### Automating MOS collection

A simple web UI or Jupyter Notebook:

- Play A/B samples in random order.
- Evaluators enter scores numerically.
- Results are appended to CSV.
- Blinded evaluation (which is A vs B is hidden).

## Pass/fail criteria (phase-gate conditions)

Gate conditions to progress to the next phase. **Initial values are tentative; tune based on Phase 1 measurements.**

### Phase 1 → Phase 2 gate

Confirms the minimum pipeline is working:

| Metric | Initial target | Notes |
|---|---|---|
| Scenario 1 TPR | > 0.85 | Pass ≥ 85 % of target frames |
| Scenario 2 TNR | > 0.80 | Mute ≥ 80 % of other-speaker frames |
| Scenario 1 PESQ improvement | > 0.3 | PESQ improvement ≥ 0.3 vs raw audio |
| Scenario 5 cross-language EER stddev | < 0.10 | Cross-language variation within 10 % |
| End-to-end latency (PoC measurement) | < 300 ms | Python implementation; final target 100 ms |

### Phase 2 → Phase 3 gate

Confirms functional completeness:

| Metric | Initial target |
|---|---|
| Scenario 3 frame accuracy | > 0.90 |
| Scenario 6 drift-detection function | Reset triggers < 5 / 60 min |
| Anchor protection | Guaranteed that auto-learning never removes anchors |
| Effect of F0 auxiliary | EER with F0 ≥ +0 % (no regression) |

### Phase 3 → Phase 4 gate

Meets performance requirements:

| Metric | Initial target |
|---|---|
| End-to-end latency (Rust measurement) | < 100 ms |
| CPU usage (single thread) | < 30 % (M1 / Ryzen 5 class) |
| Scenario 1 TPR | > 0.92 |
| Scenario 2 TNR | > 0.90 |
| Memory footprint | < 100 MB |

### Phase 4 (Mobile) gate

Confirms mobile-deployment viability:

| Metric | Initial target |
|---|---|
| iOS / Android continuous operation | ≥ 30 min without crash |
| Battery consumption | Fits within a call-equivalent power profile |
| Binary size | < 30 MB |
| Startup latency | < 2 s |

At the end of each phase, compare against the gate conditions; for any unmet item, either improve the feature or revisit the threshold. Threshold revisits are decided by considering the overall achievement and trade-offs of the corresponding phase.

## Continuous result management

### Versioning

Record the following on each evaluation run:

- Git commit hash (implementation side).
- Model versions (DFN3, silero-vad, ECAPA-TDNN).
- Evaluation dataset version (e.g. Common Voice v18).
- Evaluation timestamp.
- System info (CPU, RAM, OS).
- Threshold / parameter settings.

### History directory layout

```
benchmark_results/
├── 20260501_153045_phase1_initial/
│   ├── summary.json
│   ├── scenario_1.csv
│   ├── ...
│   └── plots/
├── 20260508_104530_phase1_threshold_tuning/
│   ├── summary.json
│   └── ...
└── latest -> 20260508_104530_phase1_threshold_tuning  (symlink)
```

### Regression detection

After the PoC implementation, re-run evaluations on each feature add / fix and compare with past results:

- **regression**: primary metric regresses by ≥ 3 % vs the previous run → fix required.
- **neutral**: within ±3 % → acceptable.
- **improvement**: ≥ 3 % gain → recorded.

Diffs are auto-reported (next/prev comparison table).

### Re-run triggers

Re-run evaluations on any of the following changes:

- Version update of any model.
- Threshold / parameter change.
- Pipeline structure change.
- Major update to a dependency library (e.g. ONNX Runtime).
- Evaluation dataset version update.

## Reporting

### End-of-phase report structure

At the end of each phase, generate `benchmark_results/<eval_id>/REPORT.md`:

```markdown
# Phase N Evaluation Report

## Summary
- Eval ID: <eval_id>
- Date: YYYY-MM-DD HH:MM
- Git commit: <hash>
- Phase gate status: PASS / FAIL

## Highlights
- <3-5 line summary of key results>

## Scenario Results
### Scenario 1: Solo Target + Noise
| SNR | TPR | TNR | PESQ improvement | DNSMOS OVRL |
|...|...|...|...|...|

### Scenario 2: ...
...

## Phase Gate Check
| Criterion | Target | Actual | Status |
|...|...|...|...|

## Comparison vs Baselines
...

## Failure Mode Analysis
...

## Next Steps
- <improvement items for the next phase>
```

### Sharing with stakeholders

PoC stage: developer only. Reports are added to Git and managed in-repo.

Future sharing (if scaled to a team):
- Share markdown reports via GitHub Issues / Discussions.
- Graph the evolution of key metrics over time.
- Immediate alerts on serious regressions.

## Benchmark-automation script layout

Evaluation scripts corresponding to the `bench/` structure shown in [benchmarks.md](benchmarks.md):

```python
# bench/runners/run_all.py
def run_evaluation(
    config_path: str,
    output_dir: str,
    scenarios: list[str] = None,  # None means all scenarios
    quick: bool = False,           # minimal eval set only
) -> EvaluationResult:
    """
    Run all evaluations and save results to output_dir.
    Returns: EvaluationResult (same structure as summary.json).
    """
    ...
```

Expected CLI:

```bash
# Quick evaluation (< 1 hour)
python bench/runners/run_all.py --quick --output benchmark_results/eval_$(date +%Y%m%d_%H%M%S)

# Full evaluation
python bench/runners/run_all.py --output benchmark_results/...

# Specific scenarios only
python bench/runners/run_all.py --scenarios scenario_1,scenario_5
```

## CI integration (future)

Once the PoC stabilises, automate part of the evaluation in CI:

- On pull request: quick evaluation (within 10 min, minimal Scenario 1 set).
- On merge to main: standard evaluation (within 1 hour, full minimal eval set).
- Weekly: full evaluation (including device-level Phase 4).

Auto-file an Issue when CI detects a regression. Finalise details at the implementation stage.
