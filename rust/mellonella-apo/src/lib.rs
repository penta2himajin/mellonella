//! Mellonella Windows APO plugin — target-speaker hard-gating +
//! DeepFilterNet 3 noise suppression exposed as a Windows Audio
//! Processing Object, the Windows-side counterpart to
//! `mellonella-ladspa`.
//!
//! Windows-only by construction: every item lives under
//! `#[cfg(windows)]`. On non-Windows targets the crate compiles to an
//! empty `cdylib`, mirroring the upstream `tympan-apo` framework's
//! cross-platform compile policy so the workspace can still
//! `cargo check --workspace` from a Linux runner.
//!
//! ## Sample-rate policy
//!
//! 48 kHz / mono / float32 only. Anything else triggers
//! [`FormatNegotiation::Suggest`] back at the audio engine, which
//! transparently inserts its built-in SRC at the graph edge — same
//! "zero added latency in practice" trick the LADSPA build relies on
//! PipeWire for.
//!
//! ## Configuration
//!
//! All non-numeric configuration arrives out-of-band, identical to the
//! LADSPA build:
//!
//! * **Enrollment**: `$MELLONELLA_ENROLLMENT` → otherwise
//!   `<dirs::config_dir>/mellonella/enrollment.json` (`mellonella-gui`'s
//!   auto-save path).
//! * **ONNX models**: `mellonella_core::hf_fetch::ensure_*_onnx`
//!   (env var → on-disk cache → first-run HuggingFace download).
//!
//! Without enrollment the plugin falls back to DFN3-only NS so it
//! stays useful before the user has enrolled — same UX as the LADSPA
//! plugin.

#![cfg_attr(not(windows), allow(unused))]

#[cfg(windows)]
mod plugin;

#[cfg(windows)]
tympan_apo::register_apo!(plugin::MellonellaApo);
