# mellonella-tse — Stage C: causal target speaker extraction

`mellonella` currently *gates* (whole-frame pass/mute) but cannot
*separate* overlapping speakers. **Stage C** adds a small **causal
time-domain target speaker extraction (TSE)** network: conditioned on a
frozen enrolled-speaker embedding, it extracts only the target speaker's
voice from a mixture.

This package is the model + training harness. It is greenfield Python ML
code; the trained model exports to ONNX for the Rust core.

## Architecture

Causal Conv-TasNet TSE with SpeakerBeam-style FiLM conditioning:

- **Encoder** — 1-D conv, ReLU, `N=256` basis channels. Causal (left-pad).
- **Conditioning** — the frozen 192-dim ECAPA enrollment embedding feeds a
  trainable 2-layer MLP (`192 → 256 → 2·B`) producing FiLM `(γ, β)`,
  `B=128` each. The ECAPA model itself is *not* part of this — the 192-dim
  vector is a plain input.
- **Separator** — causal Temporal Convolutional Network, `R=2` repeats ×
  `X=6` depthwise-separable conv blocks, dilations `1,2,4,8,16,32`. Each
  block: 1×1 conv → causal dilated depthwise conv (left-pad only) → PReLU →
  cumulative (causal) layer norm → FiLM → 1×1 residual + skip. Bottleneck
  `B=128`, conv channels `H=256`, depthwise kernel `P=3`.
- **Mask + decoder** — skip-sum → 1×1 conv → sigmoid mask over the `N`
  basis → multiply with the encoder output → 1-D transposed-conv decoder.

~1.41 M parameters with the confirmed dims (a touch under the 1.5–2.5 M
design target; kept as-is because the dims are the approved architecture).

### Two forward modes (numerically equivalent)

1. **Full-sequence** — `model(mixture, cond)` processes a whole waveform at
   once, for training. Causal via left-padding + cumulative layer norm.
2. **Streaming** — `model.forward_streaming(chunk, cond, conv_states) ->
   (extracted_chunk, new_conv_states)` processes one fixed-size chunk at a
   time with explicit causal-conv state buffers (depthwise-conv ring
   buffers, cumulative-LN running stats, encoder/decoder overlap). This is
   what exports to ONNX.

The two modes agree to ~1e-6 (see `smoke.py` check 2 and the tests). The
streaming-state layout is documented at the top of `model.py`.

## PoC vs. production — a config swap, not a rewrite

| | PoC (default) | Production (later) |
|---|---|---|
| Sample rate | 16 kHz | 48 kHz |
| Encoder kernel / stride | 32 / 16 | 96 / 48 |
| Latent frame rate | 1 kHz | 1 kHz (held constant) |
| Datasets | LibriSpeech / LibriMix `train-100` + MUSAN | VCTK + DEMAND |
| Separator | identical | identical |

`TSEConfig.poc_16k()` is the default. `TSEConfig.prod_48k()` triples the
encoder kernel/stride so the latent rate stays at 1 kHz; everything
downstream is byte-identical. Production is a config + dataset swap.

## Running the local validation gate

`smoke.py` is the Phase 2 local-CPU gate. It (1) runs a forward+backward
and prints the param count, (2) asserts full-sequence == streaming, (3)
overfits one synthetic mixture for ~200 steps and asserts the SI-SDR loss
collapses, (4) exports to ONNX and round-trips a clip through
`onnxruntime`, checking per-chunk PyTorch↔ONNX parity. No datasets needed
— it runs on an in-memory synthetic fixture set.

```bash
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install -e training[dev,onnx]
python -m tse.smoke              # exits non-zero on any failure
pytest training/                 # unit tests
ruff check training scripts/export_tse_onnx.py
```

## Training

`train.py` runs from the same code path locally and on Kaggle:

```bash
# Local sanity: overfit one synthetic mixture (no data needed).
python -m tse.train --overfit-steps 200 --out build/tse

# Full training on the synthetic fixture set (scaffold check only).
python -m tse.train --epochs 10 --out build/tse

# Full training on real data (Phase 3 — see the data.py stub).
python -m tse.train --epochs 100 --data-dir data/ --out build/tse
```

It writes per-epoch checkpoints (`ckpt_epoch*.pt`) and a `metrics.json`,
and supports `--resume`.

### Fine-tuning to reduce overlap musical noise (composite loss)

Pure SI-SDR training leaves audible musical-noise / "ジャギジャギ"
artefacts on the extracted target during **overlapping speech** — SI-SDR
is a waveform-level, scale-invariant objective that is deaf to the
time-varying spectral structure humans hear as grit (Kolbæk 2019), and
masking in Conv-TasNet's learned basis makes that mask jitter decode into
broadband bursts. Post-hoc DSP (mask flooring, mask EMA, decision-directed
Wiener) was measured to barely move the artefact (1–10 %), because it isn't
separable additive noise — it's signal distortion baked in by the loss.

The fix is a **composite loss** (opt-in, default off so existing runs are
byte-identical):

* `--mr-stft-weight 0.3` — adds a multi-resolution STFT loss (Yamamoto
  2020 formulation: spectral convergence + log-mag L1 at FFT sizes
  512/1024/2048) on top of SI-SDR. This supervises the spectral domain
  SI-SDR ignores, suppressing the broadband mask jitter.
* `--mix-consist-weight 0.1` — adds a mixture-consistency penalty (Wisdom
  2019, single-source form): penalises correlation between the extracted
  target and its residual `mixture - est`, discouraging the
  over-suppression bursts that dominate overlap regions.

Recommended workflow — **fine-tune the shipped prod weights** rather than
training from scratch, using `--init-checkpoint` to warm-start (loads model
weights only; optimizer/epoch state stay fresh, unlike `--resume`):

```bash
# Warm-start from the shipped prod_48k checkpoint and fine-tune ~5-15
# epochs with the composite anti-artefact loss at a low LR.
python -m tse.train \
  --config prod_48k \
  --data-dir data/ --data-source vctk-demand \
  --init-checkpoint path/to/tse_prod_48k.weights.pt \
  --mr-stft-weight 0.3 --mix-consist-weight 0.1 \
  --lr 5e-5 --epochs 10 \
  --out build/tse-finetune

# Then export the streaming ONNX from the best checkpoint as usual.
python scripts/export_tse_onnx.py export-and-verify \
  --config prod_48k --chunk 480 \
  --checkpoint build/tse-finetune/ckpt_epoch*.pt \
  --output build/tse_prod_48k.onnx
```

The composite weights are tunable: raise `--mr-stft-weight` toward 0.5 for
more spectral smoothing (at a small SI-SDR cost), or set both to 0 to
recover the legacy SI-SDR-only objective. A 60-step overfit sanity check
showed the composite loss reaches within ~0.4 dB of the SI-SDR-only final
SI-SDR, confirming the extra terms don't degrade separation.

On Kaggle the same knobs are env vars: `POC_MR_STFT_WEIGHT`,
`POC_MIX_CONSIST_WEIGHT`, and `POC_CHECKPOINT` (warm-start path).

For the 48 kHz production model, pass `--config prod_48k` and point
`--data-dir` at a VCTK + DEMAND root (`--data-source vctk-demand`). The
`--clip-grad-norm` flag (default `5.0`, tighter values `1.0`/`0.5`) is
the recommended fp16-AMP stability mitigation — it prevents the
underflow-driven NaN cascade observed in PoC v3 at low LR while keeping
AMP on for the speed win.

### How this maps to Kaggle (Phase 3)

1. **Enrollment embeddings** — `prepare_enrollment_embeddings.py`
   precomputes per-utterance frozen ECAPA embeddings into an `.npz`, using
   the existing ECAPA ONNX (`$MELLONELLA_ECAPA_ONNX`). The ECAPA model is
   never in the training loop.
2. **Data** — `data.py::librispeech_musan_sources` builds `TSESourceItem`
   bundles from local LibriSpeech + MUSAN (the 16 kHz PoC path), and
   `data.py::vctk_demand_sources` does the same for VCTK + DEMAND (the
   48 kHz production path: VCTK speakers grouped by the leading `pXXX`
   prefix of the filename; DEMAND noise canonicalised on `ch01.wav` per
   category so the 16-channel arrays aren't duplicated). Audio is **not**
   loaded eagerly (sources hold `Path`\ s); `TSEMixtureDataset` decodes
   lazily per access (and resamples VCTK 44.1 → 48 kHz on the fly via
   librosa) so a 6 GB train-clean-100 never has to live in RAM. The same
   dataset then does on-the-fly target+interferer(+noise) mixing.
3. **Train** — `tse.kaggle_train` is a self-contained kernel entrypoint
   (`python -m tse.kaggle_train` on Kaggle): clones the repo, installs
   `training[onnx]` + SpeechBrain, stitches `/kaggle/input/` datasets into
   the layout the loader expects, runs `prepare_enrollment_embeddings`,
   and then `train.py` on GPU. Knobs via env vars: `POC_CONFIG` (`poc_16k`
   default or `prod_48k` for the VCTK + DEMAND path), `POC_EPOCHS`,
   `POC_BATCH`, `POC_N_PAIRS`, `POC_LR`, `POC_CLIP_GRAD_NORM` (max global
   gradient norm — set to `1.0` or `0.5` for tighter fp16-AMP stability),
   `ENROLL_LIMIT`, `LIBRISPEECH_DATA` / `MUSAN_DATA` / `VCTK_DATA` /
   `DEMAND_DATA` / `ECAPA_ONNX` (path overrides).
4. **Export** — `scripts/export_tse_onnx.py export-and-verify
   --checkpoint <ckpt>` produces the streaming ONNX for the Rust core.

## Files

- `config.py` — `TSEConfig` frozen dataclass + `poc_16k()` / `prod_48k()`.
- `model.py` — `CausalConvTasNetTSE` (two forward modes), FiLM MLP,
  cumulative LN, causal TCN blocks, param-count helper.
- `data.py` — `TSEMixtureDataset` (lazy `Path` decoding) +
  `synthetic_fixture_dataset` (in-memory, no downloads) +
  `librispeech_musan_sources` (real LibriSpeech + MUSAN loader, 16 kHz
  PoC path) + `vctk_demand_sources` (VCTK + DEMAND loader, 48 kHz
  production path).
- `kaggle_train.py` — Kaggle kernel entrypoint (env-var knobs, idempotent
  re-runs).
- `loss.py` — negative SI-SDR loss, consistent with `bench`'s `si_sdr`.
- `train.py` — overfit-one-batch + full training loop, CLI, checkpointing.
- `prepare_enrollment_embeddings.py` — offline ECAPA embedding precompute
  (validated in Phase 3).
- `smoke.py` — the local-CPU validation gate (four checks).
- `tests/test_model.py` — param count, causality, full-vs-streaming, FiLM,
  loss sign/scale, dataset.
- `../../scripts/export_tse_onnx.py` — stateful per-chunk ONNX export +
  PyTorch↔ONNX parity verification.
