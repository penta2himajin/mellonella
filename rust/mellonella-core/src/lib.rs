//! Mellonella core engine — Phase 3 WIP.
//!
//! Targets feature-parity with the Python PoC at `poc/mellonella_poc/`:
//! ECAPA-TDNN embedding (via ONNX Runtime) → AS-Norm gating → DFN3
//! noise suppression → auto-learn pool. The public API will be filled in
//! once the ONNX export from `scripts/export_ecapa_onnx.py` is parity-
//! verified (handoff issue #66).

pub mod dfn3;
pub mod embedding;
pub mod enrollment;
pub mod f0;
pub mod features;
pub mod gating;
pub mod ort_threads;
pub mod pipeline;
pub mod resample;
pub mod streaming;
pub mod vad;
