//! Mellonella core engine — Phase 3 WIP.
//!
//! Targets feature-parity with the Python PoC at `poc/mellonella_poc/`:
//! ECAPA-TDNN embedding (via ONNX Runtime) → AS-Norm gating → DFN3
//! noise suppression → auto-learn pool. The public API will be filled in
//! once the ONNX export from `scripts/export_ecapa_onnx.py` is parity-
//! verified (handoff issue #66).

pub mod embedding {
    //! ECAPA-TDNN inference wrapper. Will load the ONNX produced by
    //! `scripts/export_ecapa_onnx.py` and expose a `(samples_16k) → [f32; 192]`
    //! function. Backed by `ort` once added.
}

pub mod enrollment;
pub mod f0;
pub mod gating;
