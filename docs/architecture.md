# Architecture

## Full processing pipeline

```
input (arbitrary SR, mono)
  │
  ▼
[Stage 0] Resampler @ 48 kHz                      ── unify input SR
  │
  ▼
[Stage 1] DeepFilterNet 3 (48 kHz NS)             ── noise suppression
  │
  ├──→ resample to 16 kHz ─┐
  │                         │
  │                         ▼
  │                  [Stage 2] silero-vad         ── speech / non-speech
  │                         │
  │                         ▼
  │                  [Stage 3] dynamic chunking   ── accumulate speech frames only
  │                         │
  │                         ▼
  │                  [Stage 4]
  │                  ├─ ECAPA-TDNN: speaker embedding
  │                  └─ F0 extraction (auxiliary)
  │                         │
  │                         ▼
  │                  [Stage 5] combined decision  ── gate on/off
  │                         │
  │                         ▼
  │                  [Stage 6] auto-learn pool update (conditional)
  │                         │
  │                         │ gate signal
  │                         ▼
  └──→ [Stage 7] envelope ◀──────────────────────┘
        (attack 10–20 ms, release 50–200 ms)
        │
        ▼
        output (48 kHz, mono)
```

## Stage details

### Stage 0: Resampler

- Unifies the input sampling rate to 48 kHz.
- High-quality resampler (`soxr` recommended; `scipy.signal.resample_poly` works too).

### Stage 1: DeepFilterNet 3 (NS)

- Full-band processing at 48 kHz.
- Algorithmic latency: ~30 ms (20 ms frame + 20 ms lookahead, ~30 ms effective with internal overlap-add).
- Output is the cleaned target-speaker signal plus residual other-speaker audio.
- Downstream VAD / SV consume **this cleaned signal**, which improves decision accuracy.

### Stage 2: silero-vad

- Frame-level (30 ms) binary speech / non-speech decision.
- Lightweight ONNX implementation; callable from Rust via ONNX Runtime.
- Output is a confidence score in `[0, 1]`.

### Stage 3: Dynamic chunk accumulation (VAD-conditioned chunking)

- Append only frames that silero-vad marks as speech to an internal buffer.
- Silence frames are skipped, reducing SV compute cost.
- Once the buffer reaches a fixed length (e.g. 1 s), trigger Stage 4.
- During continuous speech, refresh on a sliding window (e.g. recompute the embedding over the most recent 1 s every 250 ms).

### Stage 4: Speaker feature extraction

#### ECAPA-TDNN (required)
- Input: 16 kHz, accumulated buffer (≥ 1 s).
- Output: 192-dim speaker embedding vector.
- Inference time: ~70 ms per 1 s chunk on CPU.

#### F0 extraction (auxiliary, recommended)
- Input: 16 kHz, accumulated buffer.
- Output: mean F0, F0 trajectory.
- Use: compared against the F0 range from enrollment to reinforce the SV decision.
- Candidates: YIN (DSP-native, lightweight) or CREPE (ONNX, higher accuracy).

### Stage 5: Combined decision

```
target_score = α × cos_sim_max(emb, enrollment_pool)
             + β × f0_match(f0_mean, enrollment_f0_range)

if target_score > θ_pass:
    gate = ON
else:
    gate = OFF
```

- `cos_sim_max`: max cosine similarity against each embedding in the enrollment pool.
- `f0_match`: 1.0 when mean F0 falls inside the enrolled F0 range, decaying as it moves outside.
- `α + β = 1` (recommended initial values: `α = 0.8, β = 0.2`).
- See [gating.md](gating.md) for details.

### Stage 6: Auto-learn pool update (conditional)

Add the embedding to the auto-learn pool only when Stage 5 judges the frame as the target speaker with high confidence:

```
if cos_sim_max > θ_learn          (high-confidence threshold)
   and f0_match > θ_f0
   and continuous_speech > 1.0 s
   and anchor_distance(emb) < δ:  (drift prevention)
        add(emb, auto_learn_pool)
```

See [gating.md](gating.md) for details.

### Stage 7: Envelope application

- Applying the binary gate directly produces click artifacts, so smooth it with an attack / release envelope.
- `attack`: ON transition (recommended 10–20 ms).
- `release`: OFF transition (recommended 50–200 ms).
- Applied to the **post-DFN3 48 kHz signal**.

## Why this ordering (option A)

Three orderings were considered.

### Option A (chosen): input → DFN3 → decision → gate → output the post-DFN3 signal

- Decision is made on the clean signal → high accuracy.
- Output is the NS-processed signal → better call quality.
- DFN3 runs only once and is shared by the decision and output paths.

### Option B (rejected): input → decision → gate → DFN3 → output

- Decision uses the noisy signal → SV accuracy drops at low SNR.
- Skipping DFN3 on gate-muted frames is an advantage, but the accuracy drop dominates.

### Option C (rejected): input → DFN3 → decision; gate the original signal → output

- Suitable for use cases where DFN3 artifacts must not appear in the output.
- For call use cases the NS benefit outweighs its artifacts, so rejected.
- Worth revisiting for recording / music-style use cases.

## Latency budget

| Stage | Latency contribution |
|---|---|
| Resampler | < 5 ms |
| DFN3 (NS) | ~20 ms (2-frame conv lookahead) + ~10 ms model = ~30 ms |
| silero-vad | < 10 ms |
| Envelope attack | 10–20 ms |
| **Total (output latency, NS off)** | **~15–35 ms** (resampler + VAD + envelope) |
| **Total (output latency, NS on)** | **~45–65 ms** (matches the architecture target) |

The ECAPA-TDNN embedding computation sets the decision-update interval (chunk shift = 250 ms) but does not affect output latency: the most recent decision is held until the next update.

**DFN3 streaming**: the ONNX exported by `scripts/export_dfn3_onnx.py` is the *stateful per-frame* variant — one STFT frame in, one enhanced frame out, with the three GRU hidden states threaded across calls via explicit ONNX inputs/outputs. The wrapper's only buffering is the 2-frame future-feature queue the model's `conv_lookahead` mechanism requires; that's the source of the algorithmic ~20 ms part of the DFN3 budget. (Earlier code shipped a 102-frame ONNX export that added ~1.02 s of buffering latency; step 13.5 replaces it with the stateful per-frame export.)

That is, *decision responsiveness* and *absolute output latency* are managed separately:

- Absolute latency: 50–65 ms (excellent for VoIP).
- Decision-update interval: 250–500 ms (governs how quickly the gate follows speaker changes).

## Sampling-rate policy

- Input / output: 48 kHz (DFN3's native rate, full-band quality).
- Internal decision: 16 kHz (ECAPA-TDNN's native rate).

The post-DFN3 48 kHz signal is downsampled to 16 kHz for the decision step; the output stays at 48 kHz. This preserves final output quality while keeping the decision-stage models on their training distribution.

### Implementation

In the Rust port this is realised by the dual-rate split inside `mellonella_core::pipeline`:

- `process_offline` (and its `async_refresh` variant) take an `audio_sample_rate: u32` parameter alongside the `PipelineConfig::sample_rate` field. The former is the rate of the audio path (the WAV the caller hands in and the WAV the caller gets back); the latter is the rate of the decision path (16 kHz, fixed by ECAPA / VAD training distribution).
- When the two rates differ, the pipeline resamples the input *once* to the decision rate, runs VAD / ECAPA / F0 on the downsampled copy, and applies the envelope to the **original-rate** audio. Decision-rate boundaries are mapped onto the audio-rate axis via integer scaling (`scale_to_audio_rate`) with half-up rounding to avoid drift on non-integer ratios.
- The `mellonella` CLI sets `audio_sample_rate = 48_000` for `process`, so output WAVs are always 48 kHz mono 16-bit signed. Inputs at any common sample rate are resampled to 48 kHz on the way in. The same rate is the default for the future audio-IO crate and the GUI.

## Streaming API (Phase 3.5 step 8 — design, step 9 — implementation)

The offline `mellonella_core::pipeline::process_offline` function consumes a contiguous `&[f32]`. For live microphone use, GUI integrations, and any caller that produces samples incrementally, the engine also exposes a stateful streaming counterpart in `mellonella_core::streaming`. It inherits the dual-rate split described above: callers push audio at the output rate (typically 48 kHz), the pipeline downsamples internally to 16 kHz for the decision path, and emits envelope-gated audio at the input rate.

### Surface

```text
StreamingConfig {
    pipeline: PipelineConfig,            // decision-side cadence (sample_rate = 16 kHz)
    gate: GateConfig,
    audio_sample_rate: u32,              // audio-path rate (default 48 kHz)
    diagnostics: bool,
}

StreamingPipeline::new(pool, config, components) -> Result<Self, PipelineError>
    .push_samples(&[f32])               -> Result<StreamingOutput, PipelineError>
    .flush()                            -> Result<StreamingOutput, PipelineError>
    .pool() / .pool_mut() / .reset() / .into_parts()

StreamingOutput { audio, events, [gate|score|cos_sim_max|f0_match]_per_frame }
```

The full Rustdoc — including the buffering model, parity contract against `process_offline`, and ownership rules — lives on the module itself. This section is the architectural-level summary; the source is the canonical spec.

### Buffering model

- **Input granularity**: any length, any cadence, at `audio_sample_rate` (default 48 kHz). Callers may push 1-sample or 100 000-sample chunks.
- **Internal alignment**: the audio is resampled to 16 kHz internally for the decision path; VAD frames are 512 samples (32 ms @ 16 kHz, `vad::CHUNK_SAMPLES_16K`). Sub-frame residue is buffered in a ring at the audio rate until enough samples accumulate.
- **Output granularity**: a multiple of one VAD frame's audio-rate equivalent per call (e.g. 1536 samples @ 48 kHz, the integer scaling of 512 @ 16 kHz), plus envelope attack/release tail. Sub-frame residue produces zero output until the next call.
- **Flush**: `flush()` zero-pads the residue to a full VAD frame so the trailing audio gets one last decision pass; intended for end-of-stream.

### Parity contract

At **identity rate** (`audio_sample_rate == pipeline.sample_rate`, 16 kHz throughout, resampler bypassed) and `async_refresh = false`:

- `StreamingPipeline::new → push_samples(audio) → flush` produces per-VAD-frame `gate_per_frame` and `score_per_frame` identical to `process_offline(audio, …)`. Verified by the `streaming_smoke` integration test `streaming_identity_rate_per_frame_matches_offline`.
- Chunking the same audio in different patterns (one shot, 333, 512, 997, 1024 samples) produces byte-identical concatenated output. Verified by `streaming_identity_rate_chunk_invariance`.

At **dual rate** (e.g. 48 kHz audio / 16 kHz decision):

- `StreamingPipeline` uses a stateful `rubato::SincFixedOut` resampler; `process_offline` uses a one-shot `resample_to`. The two paths agree on overall behaviour (gate decisions, length within ±5 % of input) but are **not byte-identical** because the streaming and offline resamplers have different startup-delay characteristics. A follow-up step will unify the offline path on top of the streaming engine; the existing `pipeline_parity` Rust↔Python fixture stays at identity rate so it's unaffected by either path.

With `async_refresh = true`:

- Phase 3.5 step 9 routes async only through `process_offline_async`. `StreamingPipeline::new` returns an error when the flag is set (`PipelineError::Embedding` with a "not yet supported" message). Live async streaming needs a persistent worker lifecycle and lands in a later step.

### Diagnostics

`StreamingConfig::diagnostics` (default `false`) gates the per-VAD-frame trace arrays on `StreamingOutput`. The live hot path skips the allocation; the GUI can opt in only while a "live status" panel is open.

### Migration path for the offline pipeline (deferred)

A follow-up step will re-express `process_offline` as a thin wrapper that builds a `StreamingPipeline`, calls `push_samples` once with the whole buffer, then `flush`. The wrapper will concatenate outputs into the existing `ProcessResult` shape while preserving the Rust↔Python parity fixture byte-for-byte. Until then both paths coexist; the streaming engine is the recommended target for new callers (CLI live mode, audio-IO crate, GUI).

### Latency budget (streaming)

The latency table above describes per-sample output latency, which the streaming path inherits unchanged. Additionally:

| Source | Streaming contribution |
|---|---|
| Internal VAD frame alignment | up to one VAD frame (32 ms) of buffering when the caller pushes mid-frame chunks |
| `flush()` zero-pad | one VAD frame of synthetic audio at end-of-stream |

The caller controls input cadence (audio device callback size). For typical 10 ms callbacks, alignment buffering averages ~16 ms (half a VAD frame), well inside the 50–65 ms total latency budget.
