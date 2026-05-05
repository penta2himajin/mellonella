# Gating Logic & Enrollment

## Design principles

### FP (false-positive) tolerance

In call use cases there are moments where the system must choose between leaking the other speaker's voice (FP) and cutting the user's own (FN). This system chooses **FP tolerance**:

- A frame passes whenever the target voiceprint component is present.
- Some leakage of other speakers under simultaneous speech is acceptable; passing the user's own voice reliably is non-negotiable.
- Cutting the user's own speech, even briefly, hurts UX more than leakage does.

### Single-target-speaker assumption

The system is specialised for letting a single specified speaker pass. Multi-target extensions are deferred future work and are not considered in the initial implementation.

## Combined decision formula

```
target_score(t) = α × cos_sim_max(t)  +  β × f0_match(t)

where:
    cos_sim_max(t) = max over emb_i ∈ enrollment_pool ∪ auto_learn_pool {
                        cos_similarity(current_embedding(t), emb_i)
                     }
    f0_match(t)    = exp(-((f0_mean(t) - μ_enroll) / σ_enroll)^2 / 2)
                     # goodness-of-fit to the enrolled F0 Gaussian
    α + β = 1.0
    recommended initial values: α = 0.8, β = 0.2
```

## Two-tier thresholds

Because the FP-tolerant policy is combined with auto-learning, two thresholds with separated roles are used:

| Threshold | Role | Recommended initial |
|---|---|---|
| `θ_pass` | output gate decision | 0.30 |
| `θ_learn` | admission to the auto-learn pool | 0.80 |

The relation `θ_pass < θ_learn` is strictly enforced. Reasoning:

- The output gate is intentionally a bit loose (FP-tolerant — avoid drops).
- Auto-learning is strict (drift prevention — admit only high-confidence solo speech).

> **`θ_pass` calibration history**: originally a placeholder `0.50` based on clean-vs-clean cosine intuition, but a sweep with `scripts/calibrate.py` over librosa libri1/2/3 × white/pink noise × SNR -5..20 dB (108 cells) showed `0.50` causes the gate to close completely under noise. `0.30` was selected as the smallest `θ_pass` that meets the FP-tolerant target (mean FPR ≤ 0.05), with median TPR ≈ 0.84 and mean FPR ≈ 4.6 %. See [`benchmarks/calibration_summary.json`](benchmarks/calibration_summary.json) for details.

This separation gives FP tolerance while keeping auto-learning's drift risk in check.

## Hangover

Prevents the gate from flipping OFF during transient silences (closure before plosives, breaths, etc.):

```
if gate(t-1) == ON and target_score(t) < θ_pass:
    if elapsed_off_duration < hangover_ms (recommended 200–500 ms):
        gate(t) = ON   # hold
    else:
        gate(t) = OFF
```

## Envelope

Applying binary ON/OFF directly produces audible clicks and choppiness, so attack/release smoothing is applied to the gate signal:

```
attack_coef  = 1 - exp(-1 / (attack_ms  × sr / 1000))
release_coef = 1 - exp(-1 / (release_ms × sr / 1000))

if target_gate(t) == ON:
    envelope(t) = envelope(t-1) + attack_coef × (1.0 - envelope(t-1))
else:
    envelope(t) = envelope(t-1) + release_coef × (0.0 - envelope(t-1))

output(t) = dfn3_output(t) × envelope(t)
```

Recommended values:
- `attack_ms = 15` (fast attack).
- `release_ms = 100` (gentle fadeout).

## Enrollment

### Explicit enrollment protocol

1. Ask the user for a 30 s – 1 min recording.
   - Varied utterance context: short and long sentences, backchannels, etc.
   - Clean recording with SNR > 20 dB.
2. Extract 5–10 embeddings from the recording.
   - Sliding window (e.g. 3 s chunks, 1.5 s shift).
   - Run ECAPA-TDNN on each chunk.
3. Also record F0 statistics.
   - Compute `μ_enroll`, `σ_enroll` from the speech segments only.
4. Persist the above as the `enrollment_pool`.

### Auto-learning

Continuously add embeddings from frames judged high-confidence "target speaker" during a call:

```
admission conditions:
    cos_sim_max(t) > θ_learn        (= 0.80)
    AND f0_match(t) > θ_f0          (= 0.7 recommended)
    AND continuous_speech > 1.0 sec
    AND anchor_distance(emb) < δ    (drift prevention; see below)
    AND auto_learn_pool.is_consistent()
```

### Anchor protection

The embeddings from explicit enrollment are kept permanently as **anchors** and are never removed by auto-learning:

```
struct EmbeddingPool:
    anchors: Vec<Embedding>          # explicit enrollment, immutable
    auto_learn: VecDeque<Embedding>  # auto-learned, FIFO with cap
    max_auto_learn_size: usize = 20
```

### Consistency check

Before adding to the auto-learn pool, check the distance from the anchors:

```
fn anchor_distance(emb: &Embedding, anchors: &[Embedding]) -> f32 {
    1.0 - anchors.iter()
                  .map(|a| cos_similarity(emb, a))
                  .max()
}

if anchor_distance(emb) > δ (= 0.4 recommended):
    reject  // too far from the anchors — drift signal
```

### Periodic anomaly detection

Monitor the pool-wide median; reset the auto-learn portion if it drifts far from the anchors:

```
period: every 5 minutes, or every N auto_learn_pool updates

if median(auto_learn_pool)'s anchor_distance > δ_reset (= 0.5):
    auto_learn_pool.clear()
    log_warning("auto-learn pool drifted, resetting")
```

## VAD-conditioned dynamic chunking

ECAPA-TDNN is essentially stable only on samples of 1 s or longer. However, holding a fixed 1-second buffer at all times mixes in silence regions and degrades accuracy.

Fix: **append only frames marked as speech by silero-vad to the internal buffer**:

```
let mut speech_buffer: VecDeque<f32> = VecDeque::new();
let mut last_emb_update: Instant = Instant::now();

for frame in input_stream {
    let dfn3_out = dfn3.process(frame);
    let downsampled = resample(dfn3_out, 48000, 16000);
    let vad_score = vad.process(downsampled);

    if vad_score > 0.5 {
        speech_buffer.extend(downsampled);
        if speech_buffer.len() > MAX_BUFFER {
            speech_buffer.drain(..speech_buffer.len() - MAX_BUFFER);
        }
    }

    // Update SV every 250 ms once the buffer holds at least 1 s
    if last_emb_update.elapsed() > Duration::from_millis(250)
       && speech_buffer.len() >= 16000 {
        let emb = ecapa.embed(&speech_buffer);
        let f0_mean = f0.estimate(&speech_buffer);
        update_target_score(emb, f0_mean);
        last_emb_update = Instant::now();
    }

    let envelope = update_envelope(target_gate);
    output_stream.push(dfn3_out * envelope);
}
```

## Why F0 as an auxiliary

ECAPA-TDNN alone may not always distinguish the target from a different speaker with a similar voiceprint. F0 range varies significantly between people and works as an auxiliary signal:

- Male mean F0: ~120 Hz (individual variation 80–180 Hz).
- Female mean F0: ~220 Hz (individual variation 150–300 Hz).
- Even within the same gender, differences at the σ level of F0 contribute to discrimination.

The system uses F0 as a *soft reinforcement* rather than a *hard filter*:

- F0 match score is computed as a Gaussian (0.0–1.0).
- Folded into the combined score with weight β = 0.2.
- A frame still passes outside the F0 range if cos sim is high enough.

This minimises false detections when speaking state differs between enrollment and inference (everyday voice vs excited voice).

## Future integration with classical methods

Beyond F0 matching, several signal-processing reinforcements are candidates (low priority):

- **Harmonic + Residual Model (HNM)**: decompose audio into periodic and aperiodic components; pass only the periodic part.
- **Computational Auditory Scene Analysis (CASA)**: a time–frequency mask following the harmonic structure.
- **Spectral envelope matching**: compare the MFCC / LPC envelope between enrollment and inference.

Skipped in the initial implementation. Revisit if F0 matching proves insufficient.

## Score normalisation with AS-Norm (Adaptive S-Norm)

See `docs/decisions.md` D-010 for full details.

### Motivation

The raw `target_score(emb, pool) = α·max_cos_sim + β·f0_match` shifts in absolute value with language, noise, and recording conditions. The Scenario 5 (PR #17) baseline measurement showed, with a single global `θ_pass = 0.30`:

- ja TPR = 0.59 (drops the target at low SNR).
- zh-CN FPR = 0.33 (lets other speakers through at low SNR).

These are symmetric failure modes: **no single threshold can jointly optimise multiple languages**.

### Mechanism

A pre-built "impostor cohort" (ECAPA embeddings of 30–50 multilingual non-target speakers) is used: at test time, the cosine similarity between the current embedding and each cohort entry is computed, and the target score is z-score-normalised by the mean / std of the top-K impostor scores:

```
S_norm = (S_target - μ_top-K(S_impostor)) / σ_top-K(S_impostor)
```

This removes the absolute-value distribution shift (per-language bias), so a global `θ_pass` can cover multiple languages.

### Cohort construction

`scripts/build_impostor_cohort.py` extracts ECAPA embeddings from a manifest (CommonVoice / MLS / Emilia-YODAS — any of these formats works) and saves them as `.npz`.

```bash
python scripts/build_impostor_cohort.py \
    --manifest en=$MELLONELLA_DATA_DIR/emilia_yodas/en/manifest.csv \
    --manifest de=$MELLONELLA_DATA_DIR/mls/de/manifest.csv \
    --manifest ja=$MELLONELLA_DATA_DIR/emilia_yodas/ja/manifest.csv \
    --per-language 5 \
    --output bench/data/cohorts/impostor_cohort_v1.npz
```

Roughly 5 speakers per language is enough — cohort size has a logarithmic effect on EER. The output file is ~38 KB (50 speakers × 192 dim × 4 B), small enough to bundle with the package.

### Phased rollout

Phase 1 ✅ (PR #18): cohort build script + docs.
Phase 2 ✅ (PR #19): AS-Norm implemented in `gating.py`, `GatingConfig` extended, `pipeline.process_offline` branched, CI auto-builds the cohort and feeds it into scenario_5.
**Phase 2 observation**: ja TPR rose from 0.67 to 0.85 (+18 pp), resolving the drop problem; however zh-CN FPR worsened from 0.23 to 0.42 (+19 pp), and de TPR showed a new regression (-8 pp). See `docs/decisions.md` D-010 "Phase 2 observation" and `docs/benchmarks.md` Scenario 5 "Phase 2: deltas after AS-Norm" for details. The heuristic default `theta_pass_as_norm = 1.5` is over- or under-tight on a per-language basis.

**Phase 4 (cohort-disjoint fix, in progress)**: cohort diagnostics in PR #21 revealed a structural bug. The cohort contained test speakers, and the default of 3 speakers made the cohort itself too small (18 embeddings, top-K = 10 = 56 %). The fix bumps the default to 10 speakers and uses `build_impostor_cohort.py --skip-top-n 2` to carve test speakers out of the cohort and enforce structural separation.

Phase 3 (calibrate.py extensions): starts once Phase 4 produces a clean cohort. Calibrating on top of a broken cohort is meaningless.
Phase 5 (optional): extensions C / D / E (Language-Dependent AS-Norm, etc.) are revisited after Phase 3.

### Implementation notes (Phase 2)

- Public API:
  - `mellonella_poc.gating.load_cohort(path) -> np.ndarray`
  - `mellonella_poc.gating.as_norm_score(emb, raw, cohort, top_k) -> float`
  - `mellonella_poc.gating.target_score_as_norm(emb, pool, cohort, config) -> float`
- New `GatingConfig` fields:
  - `use_as_norm: bool = False`
  - `as_norm_cohort_path: str | None = None`
  - `as_norm_top_k: int = 10`
  - `theta_pass_as_norm: float = 1.5` (z-score scale)
  - `theta_learn_as_norm: float = 2.5`
- `PipelineComponents` loads the cohort once at build time; `process_offline` references it per-frame (no per-call reload).
- On the AS-Norm path, the per-frame score formula switches from `α·cs + β·f0_match` to `as_norm(cs vs cohort)`. F0 is still used at the auto-learn admission gate via `theta_f0`.
- When `use_as_norm = False`, the legacy path is bit-identical (default `False`).

### CLI / CI integration

- `scripts/scenario_5_from_manifest.py --as-norm-cohort PATH` switches to the AS-Norm path (`--real-pipeline` required).
- `.github/workflows/scenario_5.yml` auto-builds the cohort with `scripts/build_impostor_cohort.py` after MLS + Emilia preparation and feeds it to scenario_5; no external cohort-artifact management is needed.
