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
