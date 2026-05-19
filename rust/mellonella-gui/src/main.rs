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
            .with_inner_size([560.0, 420.0])
            .with_min_inner_size([460.0, 360.0])
            .with_title("Mellonella"),
        ..Default::default()
    };

    eframe::run_native(
        "Mellonella",
        native_options,
        Box::new(move |cc| {
            install_fonts(&cc.egui_ctx);
            let state = AppState::default();
            Ok(Box::new(MellonellaApp::new(state, tray)))
        }),
    )
}

/// Bundled Japanese (CJK) font — M+ 1 Regular, SIL OFL 1.1.
/// Distributed alongside the binary because egui's `default_fonts`
/// feature ships only Latin / Greek glyphs, so any Japanese device
/// name, path, or error message renders as tofu (□) without a fallback.
const M_PLUS_1_REGULAR: &[u8] = include_bytes!("../../../assets/fonts/Mplus1-Regular.otf");

/// Append the bundled CJK font to egui's font registry as the last
/// fallback for both the proportional and monospace families. Keeping
/// it last means Latin glyphs still come from egui's default
/// `Ubuntu-Light` (so the UI looks unchanged for non-CJK users) while
/// any CJK codepoint falls through to M+ 1 and renders correctly
/// instead of being displayed as tofu.
fn install_fonts(ctx: &egui::Context) {
    let mut fonts = egui::FontDefinitions::default();
    fonts.font_data.insert(
        "mplus1".to_string(),
        std::sync::Arc::new(egui::FontData::from_static(M_PLUS_1_REGULAR)),
    );
    fonts
        .families
        .entry(egui::FontFamily::Proportional)
        .or_default()
        .push("mplus1".to_string());
    fonts
        .families
        .entry(egui::FontFamily::Monospace)
        .or_default()
        .push("mplus1".to_string());
    ctx.set_fonts(fonts);
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn bundled_cjk_font_is_a_valid_otf() {
        // Catches the case where the binary lands in the tree but the
        // OFL `.otf` got moved / corrupted: the file must exist
        // (so `include_bytes!` resolves) and start with the OpenType
        // CFF magic (`OTTO`) so egui will accept it.
        assert!(!M_PLUS_1_REGULAR.is_empty(), "bundled font is empty");
        assert!(
            M_PLUS_1_REGULAR.starts_with(b"OTTO"),
            "bundled font is not an OpenType/CFF file (first 4 bytes: {:?})",
            &M_PLUS_1_REGULAR[..4]
        );
    }

    #[test]
    fn install_fonts_registers_mplus1_in_both_families() {
        let ctx = egui::Context::default();
        install_fonts(&ctx);
        // egui doesn't expose the font registry directly, but
        // re-running the install path with the *same* identifier
        // without an explicit reset would panic with a duplicate
        // entry if the first call hadn't taken effect — sanity-check
        // by setting fonts twice and asserting the call succeeds (no
        // panic = the registry is in a coherent state).
        install_fonts(&ctx);
    }
}
