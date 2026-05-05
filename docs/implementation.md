# Implementation Plan

## Tech stack

### Core language

**Rust** is the implementation language for the production build:

- DeepFilterNet 3 ships an official Rust implementation (the `deep_filter` crate).
- The Rust ONNX Runtime binding (`ort`) is mature, so PyTorch models can be consumed via ONNX.
- A single binary that targets both desktop and mobile is realistic.
- It also matches the project owner's Rust proficiency.

Python is restricted to PoC / validation work; the production implementation lives in Rust.

### Inference runtime

**ONNX Runtime** is the unified inference runtime. Reasons:

- silero-vad, ECAPA-TDNN, and CREPE all have ONNX-converted models published.
- INT8 quantization is straightforward (helpful for mobile size reduction).
- CPU + GPU support, with platform accelerators reachable via CoreML / NNAPI.

DFN3 is the only exception: it uses its own standalone Rust implementation (the official `deep_filter` crate).

### Per-component implementation

| Component | Implementation | Form |
|---|---|---|
| Resampler | `rubato` crate | Rust-native |
| DeepFilterNet 3 | `deep_filter` crate | Rust-native |
| silero-vad | ONNX | via `ort` |
| ECAPA-TDNN | ONNX (SpeechBrain → ONNX) | via `ort` |
| F0 (YIN) | hand-rolled or `pitch-detection` crate | Rust-native |
| F0 (CREPE, optional) | ONNX | via `ort` |

## Platform support

### Desktop

- **Linux**: `x86_64-unknown-linux-gnu`.
- **macOS**: `aarch64-apple-darwin` (Apple Silicon), `x86_64-apple-darwin`.
- **Windows**: `x86_64-pc-windows-msvc`.

Integration:
- Shipped as a library (`.so` / `.dylib` / `.dll`).
- Can also run standalone as a CLI tool.
- Virtual-audio-device integration (PipeWire / CoreAudio / WASAPI) is planned for later.

### Mobile

- **iOS**: `aarch64-apple-ios`.
- **Android**: `aarch64-linux-android`.

Optimizations:
- INT8 quantization shrinks ECAPA-TDNN from 23 MB to ~6 MB.
- CoreML / NNAPI back-ends are used to reach the on-device accelerator.
- Model files are bundled ahead of time to keep startup latency low.

Estimated binary size:
- DFN3: 6 MB.
- silero-vad: 2 MB.
- ECAPA-TDNN (INT8): 6 MB.
- F0 (YIN): 0 MB (DSP code only).
- Runtime + supporting code: ~10 MB.
- **Total**: ~25 MB.

## Implementation phases

### Phase 1: PoC (Python + PyTorch)

Goal: validate the algorithm and do initial threshold / parameter tuning.

Tasks:
- Build the Python pipeline: silero-vad + ECAPA-TDNN + DFN3.
- Implement the explicit enrollment mechanism.
- Implement the gate logic (combined decision + hangover + envelope).
- Smoke-test on the developer's own voice mixed with ambient noise.
- Validate initial values for `θ_pass` and `θ_learn`.

Estimated time: 1–2 weeks.
Deliverable: a working Jupyter notebook plus a minimal CLI.

### Phase 2: Feature extensions (Python)

Goal: validate the F0 auxiliary decision and auto-learning.

Tasks:
- Add F0 extraction (YIN or CREPE).
- Verify that the F0 match improves the combined decision.
- Implement the auto-learn pool.
- Implement anchor protection and drift detection.
- Stability test with a long-call simulation.

Estimated time: 1–2 weeks.
Deliverable: a feature-complete Python implementation.

### Phase 3: Rust port (desktop)

Goal: production desktop implementation, performance optimization.

Tasks:
- ONNX conversion: ECAPA-TDNN (SpeechBrain → ONNX); silero-vad already ships in ONNX form.
- Design the Rust crate layout (`mellonella-core`, `mellonella-cli`, `mellonella-ffi`, etc.).
- Streaming processing (ring buffer, frame sync).
- Integrate the Rust DFN3 implementation.
- Integrate ONNX Runtime (the `ort` crate).
- Benchmarks (perf parity with the Python version, CPU usage).

Estimated time: 2–3 weeks.
Deliverable: a CLI plus library that runs on Linux / macOS / Windows.

### Phase 4: Mobile support

Goal: binaries that run on iOS and Android.

Tasks:
- iOS: call the Rust library from Swift (`cbindgen` + Swift Package).
- Android: call via JNI from Kotlin.
- Apply INT8 quantization.
- Verify the CoreML / NNAPI back-ends.
- Measure battery consumption.

Estimated time: 2–3 weeks.
Deliverable: iOS / Android SDKs plus sample apps.

### Phase 5: Virtual-audio-device integration (optional)

System-wide integration so call apps (Zoom, Google Meet, etc.) can use it:

- macOS: via BlackHole / Loopback, or a CoreAudio HAL plugin.
- Linux: PipeWire filter-chain (DFN3 already ships in this form, useful as a reference).
- Windows: VB-Cable + WASAPI, or a custom APO.

Given the complexity, Phase 5 may be split out as a separate project.

## Benchmarking policy

Benchmark dataset selection, evaluation scenarios, the minimal eval set, and the metrics list are detailed in [benchmarks.md](benchmarks.md). The evaluation protocol, pass/fail criteria, and result-recording / management policy are detailed in [evaluation.md](evaluation.md).

Per-phase evaluation cadence:

- **End of Phase 1**: scenario 1 (solo + noise) + scenario 5 (multilingual).
- **End of Phase 2**: the above + scenario 6 (drift validation).
- **End of Phase 3**: all scenarios + measured latency / CPU.
- **End of Phase 4**: measurement on real mobile devices (latency / battery).

Phase progression is gated on clearing the minimum criteria for the corresponding scenarios. Concrete thresholds are fixed by the Phase 1 initial measurement.

## Development layout and directory structure (tentative)

```
mellonella/
├── docs/                          # this design spec
├── poc/                           # Phase 1–2: Python PoC
│   ├── notebooks/
│   ├── mellonella_poc/
│   └── pyproject.toml
├── crates/                        # Phase 3: Rust production implementation
│   ├── mellonella-core/           # core logic
│   ├── mellonella-cli/            # CLI
│   ├── mellonella-ffi/            # FFI (mobile bindings)
│   └── mellonella-bench/          # benchmarks
├── models/                        # ONNX-converted models (git-lfs)
├── mobile/                        # Phase 4
│   ├── ios/
│   └── android/
└── tests/                         # integration-test / benchmark audio
```

## Dependencies (tentative list)

### Rust

- `deep_filter` (official DFN3).
- `ort` (ONNX Runtime binding).
- `rubato` (resampling).
- `ndarray` (tensor ops).
- `pitch-detection` (YIN) or hand-rolled.
- `crossbeam` or `flume` (streaming channel).
- `serde` + `serde_json` (config files).

### Python (PoC)

- `torch`, `torchaudio`.
- `speechbrain` (ECAPA-TDNN).
- `silero-vad`.
- `deepfilternet`.
- `librosa` (audio pre-processing).
- `numpy`, `scipy`.
