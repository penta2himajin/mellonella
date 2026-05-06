# Design Decisions

This document records the alternatives considered during design, their rejection reasons, and the major design decisions.

## D-001: Hard-gating instead of true TSE

### Alternatives considered

A. Run ConVoiFilter (offline TSE) in-house.
B. Causal-ise ESPnet TD-SpeakerBeam (requires retraining).
C. Paper-based implementation of SpeakerBeam-SS / E3Net etc. (requires writing it ourselves).
D. Hard-gating (VAD + SV + NS, the Score Combination style of Personal VAD).

### Chosen: D (hard-gating)

Reasons:
- A is 5-second-chunk offline processing, unsuitable for real-time calls.
- B / C both require in-house training, which conflicts with the "no additional training" requirement.
- D can be built entirely from existing pretrained models.
- Given the underlying single-target-speaker requirement, the complexity of N-speaker separation is unnecessary.
- Minimal artifacts on the target voice (no mask-based unnaturalness, no spectral distortion from GAN-based generators).

### Trade-offs

- Full separation under simultaneous speech is not possible → addressed by the FP-tolerant policy.
- SV judgment is unstable on short utterances (backchannels, etc.) → addressed by dynamic chunking + temporal smoothing.

## D-002: Hybrid 48 kHz output + 16 kHz internal decision

### Alternatives considered

A. Unified 16 kHz (ConVoiFilter standalone).
B. Hybrid 48 kHz NS + 16 kHz TSE (MossFormer2_SE_48K + ConVoiFilter).
C. 48 kHz output + DFN3 + 16 kHz decision (the predecessor of the current configuration).
D. 3-stage cascade (DFN3 → TSE → MossFormer2_SR_48K).

### Chosen: C (DFN3 + 16 kHz decision)

Reasons:
- A is impossible in real-time due to ConVoiFilter's chunk constraint (5 s).
- B has high compute cost from MossFormer2_SE_48K + ConVoiFilter, and ConVoiFilter's real-time constraint remains.
- D is unsuitable for call use because MossFormer2_SR_48K uses 4-second chunks, GAN generation, and TTS training data ([details](references.md#mossformer2_sr_48k-evaluation)).
- C leverages DFN3's full-band 48 kHz processing while running the decision step on ECAPA-TDNN's native rate (16 kHz).

### Decisive reasons for rejecting MossFormer2_SR_48K

- Algorithmic latency of 4 seconds (`decode_window: 4`) → not real-time.
- Trained on TTS-synthesised audio → out-of-distribution.
- High-frequency generation via GAN → distortion of speaker characteristics.
- Artifact accumulation through the 3-stage cascade.

## D-003: Ordering is option A (NS → decision → output the post-DFN3 signal)

### Alternatives considered

A. input → DFN3 → decision → gate → output the post-DFN3 signal.
B. input → decision → gate → DFN3 → output.
C. input → DFN3 → decision; gate the original signal → output.

### Chosen: A

Reasons:
- The decision is made on the clean signal (improved accuracy).
- The output reflects the NS effect (improved call quality).
- DFN3 is computed only once and shared by the decision and output paths → minimum compute cost.

### How option C was considered

Mid-design, option C was temporarily preferred with the goal of "not letting DFN3 artifacts reach the final output." On reassessment, the NS benefit outweighs its artifacts for call use, so we returned to option A. For recording / music-style use cases, option C is worth revisiting.

## D-004: Combine explicit enrollment + auto-learning

### Alternatives considered

A. Explicit enrollment only.
B. Auto-learning only (zero-enrollment).
C. Explicit enrollment + auto-learning combined.

### Chosen: C

Reasons:
- A cannot track time-varying changes (cold, fatigue, mic change).
- B has low reliability at enrollment (no guarantee that the first session is purely the target speaker).
- C combines the reliability of explicit enrollment with the adaptability of auto-learning.

### Drift mitigation (important)

The combination of an FP-tolerant policy and auto-learning has high drift risk:

- FP tolerance lets frames with other-speaker mixing pass easily.
- Feeding those into auto-learning lets other speakers' voiceprints seep into the pool.

The following are made mandatory to mitigate this:

1. **Two-tier thresholds**: strictly enforce `θ_learn (0.80) > θ_pass (0.50)`.
2. **Anchor protection**: explicit-enrollment embeddings are kept permanently and are not removed by auto-learning.
3. **Consistency check**: validate distance from the anchors before adding to the auto-learn pool.
4. **Anomaly detection**: periodically monitor the pool's median deviation from the anchors; reset on excessive deviation.

## D-005: Add F0 as an auxiliary

### Alternatives considered

A. ECAPA-TDNN alone (judge by cosine similarity only).
B. ECAPA-TDNN + F0 match.
C. ECAPA-TDNN + F0 + harmonic-structure analysis.

### Chosen: B

Reasons:
- A alone may not always distinguish "a different speaker with a similar voiceprint."
- F0 has large individual variation and works as an auxiliary signal.
- C has high implementation cost and is overkill for the initial implementation.

### Design details

- F0 is folded into the combined score with weight β rather than used as a hard filter.
- A continuous score via Gaussian fitness avoids strict range checks.
- The design minimises false detections when speaking state at enrollment differs from inference.
- YIN (DSP-based, lightweight) is the primary candidate for the implementation; CREPE (ONNX) is the high-accuracy option.

### Calibration history for weights (α, β)

Initial values were placed by intuition at `α = 0.8, β = 0.2`. A joint sweep over α ∈ [0.0, 1.0] and θ ∈ [0.20, 0.55] using `scripts/calibrate_alpha_beta.py` on librosa libri1/2/3 × white/pink noise × SNR -5..20 dB selected **`α = 0.9, β = 0.1, θ_pass = 0.30`** as the operating point that meets the FP-tolerant target (mean FPR ≤ 0.05) while maximising TPR_median: at the same TPR_median = 0.84 as `α = 0.8`, FPR_mean drops from 0.046 to 0.017.

For reference:
- `α = 1.0` (cosine only, F0 disabled) gives TPR_median = 0.81 / FPR_mean = 0.008 — disabling F0 entirely costs ~3 points of TPR.
- `α = 0.9` is the "sparing F0 use" sweet spot in between.

For details, see [`benchmarks/calibration_alpha_beta_summary.json`](benchmarks/calibration_alpha_beta_summary.json) and [`../poc/notebooks/02_alpha_beta_sweep.py`](../poc/notebooks/02_alpha_beta_sweep.py).

> **Caveat**: the calibration speakers are all English LibriSpeech, so F0 distributions are close. With mixed-gender or cross-language data, the optimal β is likely higher. On the CI baseline (libri1-specific), `α = 0.9` lowers the low-SNR TPR slightly (e.g. 0.46 → 0.32 at SNR 5 dB) — `α = 0.9` wins on the aggregate mean but trades against per-speaker robustness. A full re-calibration with CommonVoice / VCTK and speaker diversity is planned for Phase 2.

## D-006: Not adopting VoiceFilter-Lite

### History

Google's VoiceFilter-Lite (2020) looked attractive at first thanks to its lightweight, streaming-friendly nature.

### Reason for rejection

Reading the paper directly, VoiceFilter-Lite has:
- Input: log-mel filterbank energies.
- Output: enhanced log-mel filterbank energies.

In other words, **it does not output waveforms**. It is designed exclusively as an ASR front-end and is fundamentally unusable for call applications. The SpeakerBeam-SS paper (Sato et al., Interspeech 2024) also notes this explicitly:

> "Since VoiceFilter-Lite enhances filterbank features for ASR, it is not suitable for communication applications."

The original VoiceFilter (2019) does output waveforms, but ConVoiFilter is a strict upgrade; even when rolling our own, basing it on ConVoiFilter is more reasonable.

## D-007: Handling the WHAM! dataset

### Situation

- ConVoiFilter is trained on WHAM! noise.
- WHAM! is CC BY-NC 4.0 (non-commercial).
- However, since this project is hard-gating, ConVoiFilter is not used.

### Conclusion

This configuration does not use any WHAM!-derived models, so no grey-zone issue arises:

- DFN3: DNS Challenge data (CC BY 4.0) + proprietary data.
- silero-vad: proprietary data.
- ECAPA-TDNN: VoxCeleb1+2 (public; some non-commercial caveats).

The VoxCeleb licence terms are not fully clean given the media is sourced from BBC / YouTube. At full commercial deployment, we may consider an ECAPA-TDNN alternative (e.g. one trained on CommonVoice).

## D-008: Adopt ECAPA-TDNN as the speaker embedding

### Alternatives considered

A. d-vector (standard in the VoiceFilter family).
B. x-vector (classic; used in ConVoiFilter).
C. ECAPA-TDNN (modern standard; published by SpeechBrain).
D. ECAPA2 (2024 improved version).
E. WavLM-based embedding.

### Chosen: C (ECAPA-TDNN)

Reasons:
- Currently the best balance of performance and efficiency.
- SpeechBrain's `spkrec-ecapa-voxceleb` is published under Apache 2.0 and works immediately.
- ONNX conversion is straightforward; mobile deployment is feasible.
- D (ECAPA2) is newer, but its published weights are less polished than C.
- E (WavLM) has a large model size, with real-time concerns.

### Future alternatives to consider

- Migrate to ECAPA2 (if a performance advantage is confirmed).
- TitaNet (NVIDIA; CC-BY-NC, non-commercial).
- Train our own (CommonVoice-based, to secure fully clean licensing).

## D-009: Keep the design language-independent

### Decision

DFN3, silero-vad, and ECAPA-TDNN are all language-independent. We do not run Japanese-specific fine-tuning.

### Reasons

- ECAPA-TDNN is trained on VoxCeleb (multilingual) and works on Japanese speakers.
- Japanese-specific speaker-embedding models in OSS are limited.
- Language-independence has the benefit of not being country-specific.

### Future revisit trigger

Only if device-level testing on Japanese speakers shows EER regression will we consider fine-tuning on Japanese data.

## D-010: Introduce score normalisation with AS-Norm (Adaptive S-Norm)

### Background

The initial baseline measurement of Scenario 5 (multilingual robustness) introduced in PR #17 (real pipeline, MLS + Emilia-YODAS, 6 languages × 4 SNRs) quantitatively confirmed that, with the global `θ_pass = 0.30`, score distributions skew across languages:

| Lang | TPR | FPR |
|---|---|---|
| de | 0.77 | 0.02 |
| en | 0.78 | 0.00 |
| fr | 0.80 | 0.00 |
| ko | 0.80 | 0.00 |
| **ja** | **0.67** | 0.07 |
| **zh-CN** | 0.86 | **0.23** |

ja falls to TPR = 0.59 at SNR ≤ 5 dB (FN bias), while zh-CN rises to FPR = 0.33 in the same SNR band (FP bias). **No single global `θ_pass` can jointly optimise both.**

### Alternatives considered

| Option | Decision | Reason |
|---|---|---|
| A. per-language `θ_pass` overrides | rejected | per-language manual tuning, hard to maintain |
| B. **AS-Norm (Adaptive S-Norm)** | **chosen** | industry standard (>20 years), light inference-time overhead (+30 cosine sims per call with cohort K = 30), one global threshold suffices |
| C. Language-Dependent AS-Norm (Thienpondt 2020) | future consideration | needs an LID head, more complex than B |
| D. TAS-Norm (2025, trainable) | future consideration | needs training data, exceeds the PoC scope |
| E. Discriminative condition-aware backend (Ferrer 2019) | future consideration | needs lots of calibration data and training |

### Chosen: B (AS-Norm)

Mechanism:

```
S_norm = (S_target - μ_top-K(S_impostor)) / σ_top-K(S_impostor)
```

- At inference time, in addition to computing the cosine sim between the target embedding and the enrollment, also compute the cosine sims against a pre-built **impostor cohort** (embeddings for 30–50 multilingual non-target speakers).
- Z-score-normalise by the mean and standard deviation of the top-K (K ≈ 10) impostor scores.
- Systematic bias in the score distribution that depends on language / noise / recording conditions disappears, so a global `θ_pass` covers multiple languages.

### Implementation phases

1. **Phase 1** ✅ (PR #18): cohort build script. `scripts/build_impostor_cohort.py` extracts ECAPA embeddings from MLS + Emilia manifests and outputs `.npz`.
2. **Phase 2** ✅ (PR #19): implement `as_norm_score` / `load_cohort` in `gating.py`; extend `GatingConfig` with `use_as_norm` / `as_norm_cohort_path` / `as_norm_top_k` / `theta_pass_as_norm` / `theta_learn_as_norm`; branch `pipeline.process_offline`'s score path; CI auto-builds the cohort and feeds it to scenario_5.
3. **Phase 3** ✅ (PR #24 + follow-up PR): add the AS-Norm extension to `scripts/calibrate.py`, fix `theta_pass_as_norm` data-drivenly via a per-language sweep, and document the baseline with CI observations. Tightening `scenario_5.yml --fpr-max` is **deferred** for the reason described below (run-to-run variance from the cohort scale would make the CI hard-fail flaky).
4. **Phase 4** ✅ (PR #22 + #23): cohort-disjoint fix + `actions/cache`-isation.
5. **Phase 5** ✅ (PR #27 + PR #28): cohort determinisation + pinning the HF stream input. PR #27 determinises the `mls.prepare` / `emilia.prepare` selection logic (speakers by upstream-id lex order; clips by `(-len, sha1)` order). PR #28 pins the HF stream input itself: `load_dataset(..., revision=<commit_sha>)` plus, for Emilia, fetch the shard list via `HfApi`, lex-sort it, and pass it explicitly via `data_files=`. The manifest is now bit-identical regardless of cache state. **Threshold tightening is split into a separate PR**: with the post-pin stable baseline observed once, set values under the `--fpr-max < --tpr-min` invariant.
6. **Phase 6** (in progress): cohort scale-up. **Part 1** (PR #40, merged): scale per-language 8 → 15, top-K stays at 10 in code (`DEFAULT_TOP_SPEAKERS = 18` in mls/emilia.prepare; `--per-language 15` in scenario_5.yml; `as_norm_top_k = 10` in GatingConfig; cache key v5 → v8 — v6 attempted top_speakers = 60 which CI failed on MLS de test = 30 spk; v7 attempted top_speakers = 30 which CI failed on MLS fr test = 18 spk; v8 saturates the smallest MLS test split at 18 spk). **Part 1.5** (this PR): scale per-language 15 → 50 by routing the de / fr cohort through Emilia-YODAS DE / FR shards instead of MLS — cross-source-disjoint from MLS test by construction (different upstream universes). The first attempt (v9) tried MLS train-split sourcing but failed in CI: per-speaker locality (~937 clips/spk in MLS train) + HF parquet streaming's ~20 samples/sec rate meant a 30 000-sample window only surfaced ~32 distinct speakers, and bumping the window further blows the CI timeout. v10 (this revision) pivots to Emilia for de / fr cohort — high speaker density (52 spk surface in seconds), zero new code surface beyond extending the existing Emilia prep step from 4 → 6 languages. emilia.py bumps `DEFAULT_TOP_SPEAKERS` 18 → 52 across all 6 languages; cohort `--per-language 15 → 50`; `as_norm_top_k 10 → 20`; cache key v8 → v10 (v9 was never cached). Total cohort = 6 langs × 50 spk = 300 embeddings (vs 90 in v8), reaching the literature 50-100 spk/lang lower bound. **Part 2** (gated on hosted-runner restoration): `theta_pass_as_norm` data-driven re-calibration on the v10 cohort, scenario_5 threshold tightening (combined with Phase 5 closeout follow-up), extending AS-Norm to scenarios 1 / 4 / 6. See "Phase 6: cohort scale-up" below.

### Phase 2 design notes

- On the AS-Norm path, F0 is removed from the per-frame gate decision; the judgment is made purely on the cohort-normalised SV similarity (F0 is still used at the auto-learn admission gate via `theta_f0`). Reason: AS-Norm literature applies it directly to SV similarity, and adding F0 to a z-score creates a scale mismatch.
- `theta_pass_as_norm = 1.5` / `theta_learn_as_norm = 2.5` are heuristic initial values. Phase 3 re-runs `scripts/calibrate.py` on the AS-Norm path to fix them data-drivenly.
- `use_as_norm = False` is kept as the default to ensure existing PoC + bench tests still pass bit-identically (same pattern as `enable_auto_learn`).

### Phase 2 observation (PR #19 initial CI run, real pipeline, MLS + Emilia-YODAS, 6 languages)

Comparing PR #17's legacy `α·cs + β·f0` baseline against PR #19's AS-Norm (default `theta_pass_as_norm = 1.5`, cohort 30 embeddings × 6 languages) per-language:

| Lang | TPR (legacy) | TPR (AS-Norm) | Δ TPR | FPR (legacy) | FPR (AS-Norm) | Δ FPR |
|---|---|---|---|---|---|---|
| de | 0.77 | 0.69 | **−0.08** | 0.02 | 0.04 | +0.02 |
| en | 0.78 | 0.80 | +0.02 | 0.00 | 0.00 | 0 |
| fr | 0.80 | 0.83 | +0.03 | 0.00 | 0.03 | +0.03 |
| ja | 0.67 | **0.85** | **+0.18** ✅ | 0.07 | 0.12 | +0.05 |
| ko | 0.80 | 0.86 | +0.06 | 0.00 | 0.00 | 0 |
| zh-CN | 0.86 | 0.87 | +0.01 | 0.23 | **0.42** | **+0.19** ❌ |
| **mean** | 0.78 | **0.82** | **+0.04** | 0.05 | 0.10 | +0.05 |
| **stddev** | 0.058 | 0.060 | +0.002 | 0.084 | 0.148 | +0.064 |

**Assessment**:

- **Major win**: ja TPR rose from 0.67 to 0.85 (+18 pp). At low SNR (0 / 5 dB) it improved 0.59 → 0.85, resolving the Japanese drop problem that motivated this decision. en / fr / ko also improved slightly, with aggregate TPR up +4 pp.
- **Two new regressions**:
  - **zh-CN FPR**: 0.23 → 0.42 (+19 pp); rose to 0.54 at SNR = 0. With a 30-embedding cohort (per-language 5), impostor discrimination for tonal languages is thin, and top-K = 10 occupies 33 % of the whole cohort, so normalisation may be weak.
  - **de TPR**: 0.77 → 0.69 (−8 pp), dropping to 0.48 at SNR = 0 — a new regression. AS-Norm may be unintentionally biasing the de cohort distribution.
- Aggregate FPR worsened 0.05 → 0.10 (+5 pp; zh-CN-driven), and cross-language stddev widened from 0.084 to 0.148.

**Implications for Phase 3**:

1. The heuristic `theta_pass_as_norm = 1.5` is too tight or too loose per language. Sweeping with the AS-Norm extension of `calibrate.py` to find a global optimum is the main thrust.
2. Only if that cannot absorb the issue do we consider Phase 4-style cohort scale-up (per-language 5 → 10) or higher top-K. We do not pre-commit.
3. Once `theta_pass_as_norm` is updated, also tighten the scenario_5 hard-fail thresholds (currently `--tpr-min 0.3 --fpr-max 0.7` is a 27 pp buffer over the legacy observed values — expected to shrink after Phase 3).

### Re-observation after Phase 2 (PR #21 cohort diagnostics + discovery of a structural bug)

After PR #21 turned the cohort summary into an artifact, comparing PR #19 and PR #21 cohorts revealed that the **variance was not random noise but caused by a structural bug**:

1. **The cohort contained test speakers** (a fatal algorithm violation): `scenario_5_from_manifest.py` selected target / other from the same manifest, and the cohort was built from the same manifest. Each manifest contained only 3 speakers, so the cohort was {speaker01, 02, 03} and the test was 2 of those. The "impostor cohort" for AS-Norm was actually including target / other themselves, breaking the premise of z-score normalisation.
2. **Cohort half the assumed size** (18 vs 30): `mls.prepare` / `emilia.prepare` defaulted to `top_speakers = 3`, putting only 3 speakers into each manifest. Even passing `--per-language 5` could only retrieve 3. top-K = 10 / 18 = 56 %, far exceeding the literature ceiling (10–30 %).
3. **The same `speaker01` referred to different upstream speakers across runs**: at the prepare stage, labels were assigned in "encounter order while streaming"; ordering jitter from HF datasets' parallel IO propagated directly into the cohort composition.

### Phase 4: cohort-disjoint fix ✅ (PR #22) + cohort cache stability ✅ (PR #23)

Before Phase 3 (calibrate.py extensions), **the structural bug had to be killed first**, so we ran Phase 4 as a priority:

- Bumped `mls.prepare` / `emilia.prepare`'s default `top_speakers` from 3 to 10. Each manifest now ships 10 speakers.
- Added `--skip-top-n N` to `scripts/build_impostor_cohort.py`: carve the ranks scenario_5 uses for testing out of the cohort.
- `.github/workflows/scenario_5.yml` now passes `--skip-top-n 2 --per-language 8`. Result: 8 cohort speakers per language = 48 embeddings, top-K = 10 = 21 % (within the literature range), and fully disjoint from the test set.

After PR #22 merged, running CI twice in a row on the disjoint cohort still showed zh-CN FPR fluctuating 0.76 → 0.85 between runs. The cause was that HF datasets streaming's non-deterministic order made the manifest itself regenerate differently on each cache miss. As an **additional Phase 4 measure**, we persisted the cohort with `actions/cache@v4` plus a skip-if-exists guard (PR #23):

- Added the hash of `scripts/build_impostor_cohort.py` to the cache key, so the cache auto-invalidates if the selection logic changes.
- Bumped the cache key v1 → v2 to force-discard the existing broken-cohort cache.
- Added a "skip if exists" guard to the workflow's cohort build step. Once generated, the `.npz` is reused as long as the cache hit continues — fully deterministic.
- To regenerate after adding a new language or changing `top_speakers`, just bump the cache key (v2 → v3); no need to commit it to the repo (~38 KB, but caching is operationally lighter than putting it into git every time).

This makes Phase 3's `calibrate.py` extensions ready to start. Calibrating without Phase 4's fix would just fit on top of a "broken cohort."

### Phase 3: calibrate.py AS-Norm extension (PR #24)

After PR #23 made the cohort fully deterministic via CI cache, we added an AS-Norm sweep to `scripts/calibrate.py`:

- New CLI flags `--use-as-norm --cohort PATH`. Both required.
- New sweep range `THETA_GRID_AS_NORM = (0.5, 0.75, ..., 3.0)` — 11 steps to match the z-score scale. The legacy `THETA_GRID` (0.20–0.55, cosine scale) is kept.
- Output files branch by mode:
  - legacy → `docs/benchmarks/calibration_{results.csv,summary.json}`.
  - as_norm → `docs/benchmarks/calibration_as_norm_{results.csv,summary.json}`.
- `recommend_theta()` is parametrised (`max_mean_fpr` / `min_tpr_floor`). The AS-Norm default is `MAX_MEAN_FPR_AS_NORM = 0.10`, loosened from legacy's 0.05 because the per-language FPR spread observed in PR #23 is wide.
- Added a `mode` column to the CSV / summary; schema_version 1 → 2.
- Added 10 lightweight unit tests in `bench/tests/test_scripts_calibrate.py`: theta grid range, the AS-Norm / legacy branch in `_simulate_gate`, each fallback in `recommend_theta`, and the check that `--cohort` is required when `--use-as-norm` is set.

How to run (locally):

```bash
# 1. Build the cohort (if not already)
python scripts/build_impostor_cohort.py \
    --manifest en=$MELLONELLA_DATA_DIR/emilia_yodas/en/manifest.csv \
    --manifest ja=$MELLONELLA_DATA_DIR/emilia_yodas/ja/manifest.csv \
    --manifest de=$MELLONELLA_DATA_DIR/mls/de/manifest.csv \
    --skip-top-n 2 --per-language 8 \
    --output bench/data/cohorts/scenario5_cohort_v1.npz

# 2. AS-Norm calibration sweep (repeat per language or use a concatenated manifest)
python scripts/calibrate.py \
    --use-as-norm \
    --cohort bench/data/cohorts/scenario5_cohort_v1.npz \
    --manifest ja=$MELLONELLA_DATA_DIR/emilia_yodas/ja/subset/manifest.csv \
    --language ja
```

A separate follow-up PR was planned to propagate `recommended_theta_pass` from `calibration_as_norm_summary.json` into the default of `GatingConfig.theta_pass_as_norm` (Phase 3 follow-up), with `scenario_5.yml --fpr-max` tightened afterwards. As described below, **the tightening is deferred** due to CI-observed variance.

### Phase 3 follow-up: CI baseline observation and decision to hold the threshold

After PR #24 merged, scenario_5 was run multiple times in a cohort-disjoint + cache-frozen state (PR #22 + #23) to observe how stable `theta_pass_as_norm = 1.5` is:

| Run | TPR mean | FPR mean | zh-CN FPR mean | zh-CN per-row max |
|---|---|---|---|---|
| Just after PR #23 | 0.79 | 0.15 | 0.31 | 0.62 |
| Just after PR #24 merge | 0.77 | 0.13 | 0.31 | ~0.6 |
| Follow-up run 1 | 0.74 | 0.11 | 0.34 | 0.71 |
| Follow-up run 2 | 0.77 | 0.10 | 0.31 | ~0.6 |

The aggregate values (TPR mean ~0.77, FPR mean ~0.12) converge to within ±2–3 pp across runs, but **the zh-CN per-row max fluctuates 0.6–0.85**. The cause is exactly the HF datasets streaming non-determinism discussed in Phase 4 recurring on each cache miss: the cohort composition (which 8 speakers fall into ranks 2-9) changes, AS-Norm's μ / σ shifts, and only certain zh-CN rows spike out.

**Decision**: tightening `--fpr-max` to match the observed baseline (e.g. 0.95 → 0.4) was demonstrated by PR #25 to hard-fail once every 1–2 runs, with CI then catching "noise from cohort cache generation differences" rather than "true AS-Norm regressions." We reinterpret "Phase 3 complete" as **"identify data-driven threshold candidates and document CI observations"** rather than **"tighten the threshold"**:

- `theta_pass_as_norm = 1.5` is fixed as the spec achieving the CI baseline (TPR mean 0.77 / FPR mean 0.13 across PR #23–25).
- `scenario_5.yml --fpr-max 0.95` is held. It still functions as a safety net catching catastrophic regressions (e.g. broken cohort with FPR > 0.9), but this looseness is a conscious choice; we do not tighten until the cohort scale-up tames the variance in Phase 4.
- A truly data-driven calibration (the original Phase 3 goal) is re-run only after the cohort scale reaches the literature recommendation of 50–100 spk/lang. The current per-language 8 spk = 48 cohort embeddings is too small to cut the variance.

### Phase 3 addendum: mechanism verification via local sweep

On the user's machine (mvenv: torch 2.4.1+cpu, speechbrain 1.1.0, DeepFilterNet 0.5.6), running `scripts/calibrate.py --use-as-norm --cohort cohort_v1.npz` over 108 cells × 11 θ for 8.5 min end-to-end-verified the AS-Norm path of `_simulate_gate`, the AS-Norm-specific `recommend_theta` budget (0.10), and the `mode = as_norm` output to CSV / summary. **However, the recommended θ from this sweep (= 3.0) is not adopted as a production value**:

- Local cohort = MLS de + fr (2 languages, 16 speakers); test = librosa libri (English).
- With cohort and test languages disjoint, AS-Norm's μ comes out low to begin with; FPR ≫ 0.10 at every θ, and `recommend_theta`'s fallback path (= the strictest θ) selects 3.0.
- The production cohort is 6 languages × 48 speakers and represents the real distribution including overlap with the test, so the meaning differs.

The role of the local sweep is limited to verifying **"the code runs end-to-end without crashing"**. Production values are decided in the Phase 3 follow-up section (the table above).

### Phase 5: cohort determinisation (cohort-determinism fix)

The zh-CN per-row FPR fluctuation of 0.6–0.85 observed in the Phase 3 follow-up persisted even after introducing `actions/cache` in Phase 4: the root cause is that **on each cache miss, the manifest is regenerated with different upstream speakers**. This is a structural bug in `mls.prepare` / `emilia.prepare`:

1. HF datasets streaming returns the same sample set for the same split, but **ordering is not guaranteed** (parallel IO, retry, shard interleave).
2. The old implementation labelled speakers in encounter order (`speaker01..N`), took the first K clips from each speaker, and early-broke once top-N speakers were collected — so ordering jitter propagated directly into manifest jitter.

**Fix** (`bench/mellonella_bench/datasets/mls.py` and `bench/mellonella_bench/datasets/emilia.py`):

- Remove the early-break. Scan the streaming window (`max_stream = 5000`) to the end.
- Raise the per-speaker bucket cap to `clips_per_speaker × 4` and select afterwards.
- Deterministic post-hoc selection:
  - Speaker selection: top-N by `(clips_count desc, speaker_id lex asc)`. Even when counts tie, the lex tiebreak fixes the order.
  - Label assignment: sort the selected set by **upstream `speaker_id` ascending** and assign `speaker01..N` in order. The same upstream speaker always lands in the same slot.
  - Clip selection: sort by `(-len(audio), sha1(audio.tobytes()))` and adopt the top K. Longer clips → richer information in the concat fed to ECAPA, which stabilises TPR. Tiebreak by content-hash, independent of arrival order. **A first revision (sha1 only) drew 1–2 s snippets from Emilia-YODAS, causing per-row TPR below 0.3 on ko / fr; we switched to length-first.**
- `select_speakers_for_language` in `scripts/build_impostor_cohort.py` also gains an `(-audio_size, speaker_id)` lex tiebreak. This plugs the remaining leak where selection among speakers of equal size depended on dict-iteration order.

**Contract test** (added `test_prepare_is_deterministic_under_streaming_reorder` in `bench/tests/test_datasets_{mls,emilia}.py`):

- Shuffle the same fixture into two different orderings via `random.Random(seed)` and run prepare twice.
- Compare manifest.csv as bytes and each wav with `filecmp.cmp(shallow=False)`. All require fully bit-identical results.

In addition, `test_select_speakers_uses_lex_tiebreak_for_equal_audio_lengths` was added to build_impostor_cohort, asserting that lex top-2 are selected from 4 speakers tied on audio length.

**Cache impact**: the old manifest is valid as CSV, but `speaker01..N` point to different upstream speakers, so reusing it via cache hit silently breaks AS-Norm's μ / σ. Bump `scenario_5.yml`'s cache key v2 → v3 to discard the old cache and regenerate via the new prep.

**Expected effect**:

- The manifest becomes fully invariant under ordering jitter, so the cache miss → re-prep → re-cohort-build chain becomes fully deterministic. The "cache miss = different cohort" weakness left over after Phase 4's `actions/cache` introduction goes away.
- This makes it possible to resume cohort scale-up (Phase 4's original goal) and `--fpr-max` tightening (deferred at Phase 3 follow-up) on a reproducible baseline.

### Phase 5 follow-up: HF revision pin (true idempotency)

After PR #27 merged, an attempt in PR #28 to tighten to `--tpr-min 0.60 --fpr-max 0.55` saw 2/48 row failures in CI (per-language: fr FPR mean 0.414, ja TPR mean 0.793; the gate as a whole was healthy at TPR mean 0.828 / FPR mean 0.118). Digging into why the baseline did not reproduce — given that Phase 5 had claimed to "determinise the `mls.prepare` / `emilia.prepare` selection logic" — revealed that **the HF stream input itself was not pinned**:

1. `load_dataset(...)` was called with `revision` unset, implicitly resolving upstream's `main` HEAD. The next time the dataset updates, you silently end up in a different universe.
2. Emilia was passing `data_files={"train": "Emilia-YODAS/JA/*.tar"}` — a glob — but HF datasets' internal `resolve_pattern` (`src/datasets/data_files.py`) consumes `fs.glob(...).items()` directly with **no `sorted()` immediately after**. The order of matching tars is implementation-defined by fsspec / HfFileSystem and contractually undefined.
3. PR #27 only fixed "same stream input → same manifest"; reproducibility of the stream input itself was out of scope.

**Fix** (PR #28):

- Add a `DATASET_REVISION` constant (commit SHA) to `bench/mellonella_bench/datasets/{mls,emilia}.py` and pin via `load_dataset(..., revision=DATASET_REVISION)`.
  - MLS: `facebook/multilingual_librispeech@2e83e61823b4c47dcbcb1980bb88601274127609` (2024-08-12).
  - Emilia: `amphion/Emilia-Dataset@d7f2f7340a6385696f3766c8049fa920a4707c07` (2025-02-28).
- Drop Emilia's glob. Fetch the shard list via `HfApi().list_repo_files(..., revision=DATASET_REVISION)`, lex-sort, and pass `data_files={"train": shards}` — explicitly pinned-and-sorted ordering into `load_dataset`. Shard input order is now a function only of `(DATASET_REPO, DATASET_REVISION, language)`.
- Bump `scenario_5.yml`'s cache key v4 → v5: if a v4 manifest cache built with the pre-pin unsorted glob is reused, the opening short-circuit of `prepare()` would let an outdated manifest through to the new code.

**Threshold tightening is deferred to a separate PR**:

PR #28's original `--tpr-min 0.60 --fpr-max 0.55` (with the `fpr_max < tpr_min` invariant) is to resume in a separate PR after observing one idempotent baseline on main. Decide values from the per-row max FPR / min TPR observed on the post-pin stable baseline.

### Phase 5 closeout: remaining tasks

- Run scenario_5 once after merge to populate the v5 cache on main (the merge event triggers the workflow, or kick it manually via `workflow_dispatch`).
- Run the same commit twice and verify per-row metrics are bit-identical.
- Record the per-row table at that point as the observed values, and land a threshold-tightening PR in Phase 5 follow-up part 2 (maintaining `fpr_max < tpr_min`).

**Status (2026-05-05)**: blocked on a hosted-runner allocation outage on `penta2himajin/mellonella` — every push-event scenario_5 run since PR #28 merged (run #34 onwards, 12 consecutive failures) terminates within ~11 s at the "Job is waiting for a hosted runner to come online" line, before any user step executes. The job's `system.txt` confirms the `if:` evaluates true; the failure is at GitHub's runner scheduler, not in the workflow. Until the runner allocation is restored (likely a billing / quota issue on the account side), the bit-identical observation cannot be made, and the threshold-tightening follow-up stays deferred. Phase 6 (cohort scale-up) was opened in parallel because it does not depend on the Phase 5 baseline observation — the threshold-tightening follow-up will then land combined Phase 5 + Phase 6 numbers in one PR once the runner is back.

### Phase 6: cohort scale-up

Phase 4 carved test target / other out of the cohort (cohort-disjoint) and Phase 5 made the manifest + cohort bit-identical across cache rebuilds. Both fixes were prerequisites to scaling the cohort up; on a non-disjoint or non-deterministic cohort, scaling would have just amplified the existing structural bugs. With those out of the way, Phase 6 moves the cohort from the Phase 4-5 working size (6 langs × 8 spk = 48 embeddings, top-K = 10 = 21 % of cohort) toward the literature recommendation (50–100 spk/lang, top-K 20–30) — see Thienpondt 2020 and the AS-Norm survey in Park 2025.

**Part 1 (v8) — MLS test split limit (discovered by two failed CI runs)**:

| cache | `DEFAULT_TOP_SPEAKERS` | `--per-language` | CI outcome |
|-------|------------------------|------------------|------------|
| v6    | 60                     | 50               | failed: `MLS(german, test) yielded only 30 speaker(s) after 3394 samples; need 60` |
| v7    | 30                     | 25               | failed: `MLS(french, test) yielded only 18 speaker(s) after 2426 samples; need 30` |
| v8 (PR #40, merged) | 18      | 15               | saturates MLS fr test (the smallest split) |

The MLS HF `test` split is small by design — it's a held-out evaluation set, not a source for impostor cohorts. Per-language sizes: German test = 30 spk / 3 394 samples; French test = 18 spk / 2 426 samples. **18 was the binding limit on the test path.** Part 1 (v8) saturated it: 6 langs × 15 spk = 90 embeddings, top-K = 10 (= 11 % of cohort). This was a ~2× cohort growth over Phase 4-5 — the most that MLS test could support without architectural changes.

**Part 1.5 — break past the MLS test split for the cohort**:

The MLS test split's 18-speaker French bound (Part 1, v8) is structural — `test` is a held-out evaluation split, not designed to source impostor cohorts from. To grow past v8's 90-embedding cohort we need a different source for de / fr cohort embeddings (en / ja / ko / zh-CN already source from Emilia-YODAS, which has plenty of speaker density).

The first attempt (v9) tried MLS train-split sourcing on the rationale that MLS train has 4 500+ speakers per language and is split-disjoint from MLS test by canonical LibriSpeech construction. The CI run failed at `RuntimeError: MLS(german, train) yielded only 32 speaker(s) after 30000 samples; need 52`: MLS train has heavy per-speaker locality (~937 clips per speaker laid out consecutively in the parquet shards) and HF streaming's per-row iteration rate is ~20 samples/sec, so surfacing 52 distinct speakers requires scanning ~50 000 samples ≈ 40 minutes per language — past the 60-min CI timeout for de + fr.

The second attempt (v10, this PR) routes the de / fr cohort through Emilia-YODAS DE / FR shards instead. Emilia-YODAS has CC-BY YouTube-derived shards for all 6 cohort languages (`Emilia-YODAS/{EN,ZH,JA,KO,DE,FR}/`) with high per-shard speaker density — the existing 5 000-sample scan window surfaces 52 speakers in seconds. Cross-source disjointness is automatic: MLS test (read audiobook from LibriVox) and Emilia-YODAS (CC-BY YouTube) share no upstream speakers by construction.

**Architecture (v10)**:

| role | data source | output dir | top_speakers | consumer |
|------|-------------|------------|--------------|----------|
| de / fr target / other | MLS test | `data/mls/<lang>/` | 18 | scenario_5_from_manifest.py |
| de / fr cohort | Emilia-YODAS DE / FR | `data/emilia_yodas/<lang>/` | 52 (ranks 2-51 used) | build_impostor_cohort.py |
| en / ja / ko / zh-CN target / other | Emilia-YODAS | `data/emilia_yodas/<lang>/` | 52 (top-2 used) | scenario_5_from_manifest.py |
| en / ja / ko / zh-CN cohort | Emilia-YODAS (same manifest) | `data/emilia_yodas/<lang>/` | 52 (ranks 2-51 used) | build_impostor_cohort.py |

For en / ja / ko / zh-CN the same 52-speaker manifest serves both target / other (top-2) and cohort (ranks 2-51 via `--skip-top-n 2`). For de / fr the Emilia manifest is cohort-only; target / other comes from the disjoint MLS test manifest, so `--skip-top-n 2` against the de / fr Emilia manifest is a uniform no-op (drops the lex-first 2 Emilia speakers; harmless, avoids per-manifest skip-top-n config in the cohort builder).

**Code changes (this PR — Part 1.5, v10)**:

- `bench/mellonella_bench/datasets/emilia.py`: bump `DEFAULT_TOP_SPEAKERS` 18 → 52 (header comment expanded to explain the dual-source coverage). Emilia shard speaker density is high (hundreds per shard); the existing 5 000-sample scan window comfortably surfaces 52 speakers per language. Peak memory ≈ 0.5 GB / language.
- `bench/mellonella_bench/datasets/mls.py`: header comment updated to flag MLS test as the binding bound for the test split; otherwise no behavioral change vs v8 (the v9 `max_speakers_seen` parameter and `--max-stream` / `--max-speakers-seen` CLI flags were reverted because the train-split path they supported is no longer wired up).
- `.github/workflows/scenario_5.yml`: extend the existing Emilia-YODAS prep step from 4 languages (en / ja / ko / zh-CN) to 6 (adds de, fr). The MLS test prep step (de, fr → `data/mls/<lang>/`) is unchanged. Cohort-build step's `--manifest` map is rewritten — all 6 languages now point at `data/emilia_yodas/<lang>/manifest.csv`. `--per-language` stays at 50; total cohort = 6 langs × 50 spk = 300 embeddings (same as the v9 plan). Cohort build is now gated on HF_TOKEN (because all 6 cohort sources require it); the scenario_5 run step omits `--as-norm-cohort` when HF_TOKEN is missing so the de / fr cosine-only path still produces a result. Bump cache key v9 → v10 (manifest layout changes — v9 artifacts have a `mls_cohort/` tree that v10 doesn't, and v10 needs new `emilia_yodas/{de,fr}/` trees v9 didn't have). `timeout-minutes` stays at 60 as a safety margin.
- `poc/mellonella_poc/config.py`: `as_norm_top_k = 20` (carried over from v9; matches the same 300-embedding cohort target).

**Cost projection (v10)**:

- Manifest prep (cold cache): MLS test path unchanged (~5 min total for de + fr). Emilia path now 6 languages × ~3-5 min each = ~18-30 min. Total cold-cache scenario_5 prep ≈ 25-35 min, comfortably inside the 60-min job timeout.
- Cohort build (cold cache): 300 ECAPA passes ≈ 5-7 min vs ~2 min for the v8 90-embedding cohort. Still negligible relative to manifest prep.
- Warm cache (steady state once main has cached the v10 artifacts): no measurable change vs v8.
- Cohort `.npz` size: 300 × 192 × 4 B ≈ 230 KB (vs 70 KB for v8). Still negligible for the artifact upload.
- Memory: Emilia prep peak ≈ 0.5 GB; ECAPA model resident ≈ 100 MB. Worst case ≈ 0.6 GB, well within the 7 GB CI runner budget.

**v9 attempt — what we learned**

The v9 commit (`feat(d-010): Phase 6 part 1.5 — MLS train-split cohort sourcing (v9)`, 6c44a40) attempted MLS train sourcing with `--max-stream 30000 --max-speakers-seen 60 --top-speakers 52`. Per-row HF parquet streaming over MLS train turned out to be the binding cost, not memory: ~20 samples/sec × 30 000 samples ≈ 23 min per language for de alone, and the German train shards' per-speaker locality meant that 30 000 samples surfaced only 32 distinct speakers (each speaker contributes ~937 consecutive clips to the iterator). Bumping `--max-stream` further to surface 52 speakers would have pushed the streaming step past the CI timeout. The v10 pivot to Emilia-YODAS for de / fr cohort skips the MLS-train rabbit hole entirely — Emilia's per-shard speaker density makes the same 5 000-sample window enough.

**Phase 6 follow-up part 2** (hosted-runner restored as of 2026-05-06; observation phase in progress):

1. **`theta_pass_as_norm` data-driven re-calibration** (in progress). The heuristic 1.5 σ default was empirically validated on the Phase 4-5 cohort (48 embeddings, top-K 10). After v10 the cohort grows to 300 embeddings with top-K 20 — both the candidate pool size AND the top-K shift, so the impostor-tail distribution moves materially. `scenario_5.yml` now runs `scripts/calibrate.py --use-as-norm --cohort data/cohorts/scenario5_cohort.npz` after the cohort build, staging `calibration_as_norm_summary.json` into the `scenario-5-results` artifact. Once one CI run lands a stable `recommended_theta_pass`, propagate it into the `GatingConfig.theta_pass_as_norm` default in a follow-up PR. Same deferral pattern as Phase 3 → Phase 5 closeout: heuristic shipped, data-driven update gated on one stable baseline observation.
2. **Threshold tightening** (`scenario_5.yml --tpr-min` / `--fpr-max`). Inherits the Phase 5 closeout deferral. Once the v10 cache is populated and we have one bit-identical CI baseline, both Phase 5 and Phase 6 follow-ups can land in the same PR.
3. **Extending AS-Norm to other scenarios** (scenario_1 / 4 / 6). Currently only scenario_5 wires the cohort through; the gating layer already supports it via `GatingConfig.use_as_norm` so the change is mostly per-scenario YAML wiring + per-scenario θ. Better done after the threshold landed because the new threshold is the shared default.

**Phase 6 closeout: remaining tasks** (gated on the same hosted-runner restoration as Phase 5 closeout):

- Once a Phase 6 push-event run completes successfully, observe the per-row TPR / FPR baseline with the v10 cohort.
- Run the same commit twice; bit-identical guarantee should still hold (Phase 5 fixes apply unchanged; the Emilia-YODAS scan is deterministic given DATASET_REVISION + sorted *.tar shard list).
- Run `scripts/calibrate.py --use-as-norm --cohort data/cohorts/scenario5_cohort.npz` against the v10 cohort, propagate the recommended `theta_pass_as_norm` into `GatingConfig` default.
- Tighten `scenario_5.yml` thresholds under the `--fpr-max < --tpr-min` invariant (combined Phase 5 + Phase 6 follow-up).
- Add `use_as_norm = True` to scenario_1 / 4 / 6 with the new global θ.

### References

- Thienpondt et al. (2020) "Cross-Lingual Speaker Verification with Domain-Balanced Hard Prototype Mining and Language-Dependent Score Normalization", https://arxiv.org/abs/2007.07689
- Park et al. (2025) "Trainable Adaptive Score Normalization for Automatic Speaker Verification", https://arxiv.org/abs/2504.04512
- Ferrer et al. (2019) "A Discriminative Condition-Aware Backend for Speaker Verification", https://arxiv.org/abs/1911.11622
- Klusáček et al. (2025) "On the influence of language similarity in non-target speaker verification trials", https://arxiv.org/abs/2506.02777
