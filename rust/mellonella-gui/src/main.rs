//! Mellonella GUI — egui + tray-icon front-end for the live filter.
//!
//! See `app::MellonellaApp` for the UI behaviour and `state::AppState`
//! for the data model. `tray::TrayHandles` is best-effort: it falls
//! through to a window-only experience on platforms where building
//! the system tray icon fails.
//!
//! Required env vars before launching (same as the CLI):
//!
//! * `MELLONELLA_ECAPA_ONNX` — ECAPA-TDNN ONNX path
//! * `MELLONELLA_VAD_ONNX`   — silero-vad ONNX path
//! * `ORT_DYLIB_PATH`        — ONNX Runtime dylib (used by `ort`'s
//!   `load-dynamic` feature)

#![forbid(unsafe_code)]
// Pedantic lints that fight GUI code idioms:
//
// * `cast_precision_loss` / `cast_possible_truncation` / `cast_sign_loss`:
//   pixel coordinates, sample counts and frame durations all fit
//   comfortably in f32. The casts are intentional.
// * `must_use_candidate`: trivially true for tiny getters; tagging
//   each one is noise.
#![allow(
    clippy::cast_precision_loss,
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss,
    clippy::must_use_candidate
)]

mod app;
mod state;
mod tray;

use eframe::{egui, NativeOptions};

use crate::app::MellonellaApp;
use crate::state::AppState;
use crate::tray::TrayHandles;

fn main() -> Result<(), eframe::Error> {
    let tray = TrayHandles::try_new();
    if tray.is_none() {
        eprintln!("[gui] tray-icon init failed; running with window only");
    }

    let native_options = NativeOptions {
        viewport: egui::ViewportBuilder::default()
            .with_inner_size([420.0, 280.0])
            .with_min_inner_size([320.0, 200.0])
            .with_title("Mellonella"),
        ..Default::default()
    };

    eframe::run_native(
        "Mellonella",
        native_options,
        Box::new(move |_cc| {
            let state = AppState::default();
            Ok(Box::new(MellonellaApp::new(state, tray)))
        }),
    )
}
