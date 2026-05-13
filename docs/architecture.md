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
| DFN3 (NS) | ~30 ms |
| silero-vad | < 10 ms |
| Envelope attack | 10–20 ms |
| **Total (output latency)** | **~50–65 ms** |

The ECAPA-TDNN embedding computation sets the decision-update interval (chunk shift = 250 ms) but does not affect output latency: the most recent decision is held until the next update.

That is, *decision responsiveness* and *absolute output latency* are managed separately:

- Absolute latency: 50–65 ms (excellent for VoIP).
- Decision-update interval: 250–500 ms (governs how quickly the gate follows speaker changes).

## Sampling-rate policy

- Input / output: 48 kHz (DFN3's native rate, full-band quality).
- Internal decision: 16 kHz (ECAPA-TDNN's native rate).

The post-DFN3 48 kHz signal is downsampled to 16 kHz for the decision step; the output stays at 48 kHz. This preserves final output quality while keeping the decision-stage models on their training distribution.

## Streaming API (Phase 3.5 step 8 — design, step 9 — implementation)

The offline `mellonella_core::pipeline::process_offline` function consumes a contiguous `&[f32]`. For live microphone use, GUI integrations, and any caller that produces samples incrementally, the engine also exposes a stateful streaming counterpart in `mellonella_core::streaming`.

### Surface

```text
StreamingConfig { pipeline: PipelineConfig, gate: GateConfig, diagnostics: bool }

StreamingPipeline::new(pool, config, components) -> Result<Self, PipelineError>
    .push_samples(&[f32])               -> Result<StreamingOutput, PipelineError>
    .flush()                            -> Result<StreamingOutput, PipelineError>
    .pool() / .pool_mut() / .reset() / .into_parts()

StreamingOutput { audio, events, [gate|score|cos_sim_max|f0_match]_per_frame }
```

The full Rustdoc — including the buffering model, parity contract against `process_offline`, and ownership rules — lives on the module itself. This section is the architectural-level summary; the source is the canonical spec.

### Buffering model

- **Input granularity**: any length, any cadence. Callers may push 1-sample or 100 000-sample chunks.
- **Internal alignment**: VAD frames are 512 samples (32 ms @ 16 kHz, `vad::CHUNK_SAMPLES_16K`). Sub-frame residue is buffered in an internal ring until enough samples accumulate.
- **Output granularity**: a multiple of 512 samples per call, plus envelope attack/release tail. Sub-frame residue produces zero output until the next call.
- **Flush**: `flush()` zero-pads the residue to a full VAD frame so the trailing audio gets one last decision pass; intended for end-of-stream.

### Parity contract

With `async_refresh = false`, chunking the same audio differently must produce the **same concatenated output** as a single `process_offline` call. A `streaming_chunk_invariance` test (chunks of 100 / 333 / 512 / 1024 samples) enforces this in the implementation PR. The existing `pipeline_parity` Rust↔Python parity fixture continues to use `process_offline` directly.

With `async_refresh = true`, the chunking-invariance property still holds within a single chunking strategy, but per-frame scores trail their refresh point by ≤ 1 ECAPA inference (~44 ms on the dev VM) — the same delay documented on `PipelineConfig::async_refresh`. The gate's `hangover_ms` already absorbs delays of that order, so audible behaviour is unchanged.

### Diagnostics

`StreamingConfig::diagnostics` (default `false`) gates the per-VAD-frame trace arrays on `StreamingOutput`. The live hot path skips the allocation; the GUI can opt in only while a "live status" panel is open.

### Migration path for the offline pipeline

`process_offline` will be re-expressed as a thin wrapper that builds a `StreamingPipeline`, calls `push_samples` once with the whole buffer, then `flush`. The wrapper concatenates outputs into the existing `ProcessResult` shape, preserving the Rust↔Python parity fixture byte-for-byte. The migration lands in the same PR as the streaming implementation.

### Latency budget (streaming)

The latency table above describes per-sample output latency, which the streaming path inherits unchanged. Additionally:

| Source | Streaming contribution |
|---|---|
| Internal VAD frame alignment | up to one VAD frame (32 ms) of buffering when the caller pushes mid-frame chunks |
| `flush()` zero-pad | one VAD frame of synthetic audio at end-of-stream |

The caller controls input cadence (audio device callback size). For typical 10 ms callbacks, alignment buffering averages ~16 ms (half a VAD frame), well inside the 50–65 ms total latency budget.
