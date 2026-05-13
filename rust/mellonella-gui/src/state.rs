//! Mutable state owned by the eframe app.
//!
//! Modelled as one flat struct rather than a `SessionState` enum
//! because the UI inspects most fields independently of whether a
//! live session is currently running (enrollment path, device
//! selection, last error are all sticky across start/stop cycles).

use std::path::{Path, PathBuf};

use mellonella_audio_io::{
    list_input_devices, list_output_devices, AudioDevice, LiveSession, LiveSessionStats,
    SessionConfig, SessionEvent,
};
use mellonella_core::embedding::EcapaTdnn;
use mellonella_core::enrollment::{EmbeddingPool, EmbeddingPoolConfig};
use mellonella_core::features::Fbank;
use mellonella_core::gating::GateConfig;
use mellonella_core::pipeline::{PipelineComponents, PipelineConfig};
use mellonella_core::streaming::StreamingConfig;
use mellonella_core::vad::SileroVad;

/// Output sample rate used end-to-end (matches the CLI's offline
/// constant and `StreamingConfig::default().audio_sample_rate`).
pub const OUTPUT_SAMPLE_RATE: u32 = 48_000;

/// Decision sample rate for VAD / ECAPA / F0 inside the pipeline.
pub const DECISION_SAMPLE_RATE: u32 = 16_000;

pub struct AppState {
    pub enrollment_path: Option<PathBuf>,
    pub pool_anchors: usize,
    pub pool_f0_mu: f32,
    pub pool_f0_sigma: f32,
    pub available_inputs: Vec<AudioDevice>,
    pub available_outputs: Vec<AudioDevice>,
    /// Selected input device name; `None` → host default. Stored as
    /// `String` rather than indexes into `available_inputs` so the
    /// selection survives a device-list refresh.
    pub selected_input: Option<String>,
    pub selected_output: Option<String>,
    pub session: Option<LiveSession>,
    pub last_error: Option<String>,
    pub last_stats: LiveSessionStats,
}

impl Default for AppState {
    fn default() -> Self {
        let available_inputs = list_input_devices().unwrap_or_default();
        let available_outputs = list_output_devices().unwrap_or_default();
        Self {
            enrollment_path: None,
            pool_anchors: 0,
            pool_f0_mu: 0.0,
            pool_f0_sigma: 0.0,
            available_inputs,
            available_outputs,
            selected_input: None,
            selected_output: None,
            session: None,
            last_error: None,
            last_stats: LiveSessionStats::default(),
        }
    }
}

impl AppState {
    /// Try to load the enrollment JSON at `path`. On success records
    /// metadata used by the UI (anchor count, F0 stats). On failure
    /// clears any cached metadata and surfaces the error to the user.
    pub fn load_enrollment(&mut self, path: &Path) {
        match EmbeddingPool::load(path, EmbeddingPoolConfig::default()) {
            Ok(pool) => {
                let m = pool.metadata();
                self.enrollment_path = Some(path.to_path_buf());
                self.pool_anchors = pool.anchors().len();
                self.pool_f0_mu = m.f0_mu;
                self.pool_f0_sigma = m.f0_sigma;
                self.last_error = None;
            }
            Err(e) => {
                self.enrollment_path = None;
                self.pool_anchors = 0;
                self.pool_f0_mu = 0.0;
                self.pool_f0_sigma = 0.0;
                self.last_error = Some(format!("load enrollment: {e}"));
            }
        }
    }

    /// Re-enumerate input/output devices from cpal. Preserves the
    /// existing selection if the named device is still present.
    pub fn refresh_devices(&mut self) {
        self.available_inputs = list_input_devices().unwrap_or_default();
        self.available_outputs = list_output_devices().unwrap_or_default();
        if let Some(name) = &self.selected_input {
            if !self.available_inputs.iter().any(|d| &d.name == name) {
                self.selected_input = None;
            }
        }
        if let Some(name) = &self.selected_output {
            if !self.available_outputs.iter().any(|d| &d.name == name) {
                self.selected_output = None;
            }
        }
    }

    /// Build a fresh `PipelineComponents` from the ONNX env vars.
    /// Used on every start (one ECAPA / VAD session per live run so
    /// stopping fully releases ORT resources).
    fn build_components() -> Result<PipelineComponents, String> {
        let ecapa_path = std::env::var("MELLONELLA_ECAPA_ONNX")
            .map_err(|_| "MELLONELLA_ECAPA_ONNX env var not set".to_string())?;
        let vad_path = std::env::var("MELLONELLA_VAD_ONNX")
            .map_err(|_| "MELLONELLA_VAD_ONNX env var not set".to_string())?;
        let fbank = Fbank::with_speechbrain_filterbank().map_err(|e| format!("Fbank init: {e}"))?;
        let ecapa =
            EcapaTdnn::from_onnx_path(&ecapa_path).map_err(|e| format!("ECAPA load: {e}"))?;
        let vad = SileroVad::from_onnx_path(&vad_path, DECISION_SAMPLE_RATE)
            .map_err(|e| format!("VAD load: {e}"))?;
        Ok(PipelineComponents {
            vad,
            fbank,
            ecapa,
            cohort: Vec::new(),
        })
    }

    /// Spin up a `LiveSession`. Requires an enrollment to have been
    /// loaded first. On failure records the message and leaves the
    /// session field unchanged.
    pub fn start(&mut self) {
        if self.session.is_some() {
            return;
        }
        let Some(path) = self.enrollment_path.clone() else {
            self.last_error = Some("load an enrollment JSON first".into());
            return;
        };
        let pool = match EmbeddingPool::load(&path, EmbeddingPoolConfig::default()) {
            Ok(p) => p,
            Err(e) => {
                self.last_error = Some(format!("reload enrollment: {e}"));
                return;
            }
        };
        let components = match Self::build_components() {
            Ok(c) => c,
            Err(e) => {
                self.last_error = Some(e);
                return;
            }
        };
        let cfg = SessionConfig {
            input_device: self.selected_input.clone(),
            output_device: self.selected_output.clone(),
            streaming: StreamingConfig {
                pipeline: PipelineConfig::default(),
                gate: GateConfig::default(),
                audio_sample_rate: OUTPUT_SAMPLE_RATE,
                diagnostics: false,
            },
            ring_capacity_samples: 0,
        };
        match LiveSession::new(pool, components, cfg) {
            Ok(s) => {
                self.session = Some(s);
                self.last_error = None;
            }
            Err(e) => {
                self.last_error = Some(format!("open live session: {e}"));
            }
        }
    }

    /// Tear down the live session and capture its final stats.
    pub fn stop(&mut self) {
        let Some(session) = self.session.take() else {
            return;
        };
        match session.stop() {
            Ok(stats) => {
                self.last_stats = stats;
                self.last_error = None;
            }
            Err(e) => {
                self.last_error = Some(format!("stop session: {e}"));
            }
        }
    }

    /// Poll the live session for stats + events. Call once per UI
    /// frame so the displayed counters stay fresh and worker-side
    /// errors propagate into `last_error`.
    pub fn poll_session(&mut self) {
        let Some(session) = self.session.as_ref() else {
            return;
        };
        self.last_stats = session.stats_snapshot();
        if let Some(SessionEvent::Error(msg)) = session.try_recv_event() {
            self.last_error = Some(format!("pipeline error: {msg}"));
            // Worker is dead; tear down the session shell so the UI
            // returns to the Ready state.
            self.stop();
        }
    }

    #[must_use]
    pub fn is_running(&self) -> bool {
        self.session.is_some()
    }

    #[must_use]
    pub fn can_start(&self) -> bool {
        self.enrollment_path.is_some() && self.session.is_none()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_state_is_idle() {
        let s = AppState::default();
        assert!(s.enrollment_path.is_none());
        assert_eq!(s.pool_anchors, 0);
        assert!(!s.is_running());
        assert!(!s.can_start());
        assert!(s.last_error.is_none());
    }

    #[test]
    fn loading_a_missing_enrollment_records_an_error() {
        let mut s = AppState::default();
        s.load_enrollment(std::path::Path::new(
            "/no/such/enrollment-mellonella-gui-test.json",
        ));
        assert!(s.last_error.is_some(), "expected an error to be recorded");
        assert!(s.enrollment_path.is_none());
        assert!(!s.can_start());
    }

    #[test]
    fn refresh_devices_clears_invalid_selection() {
        let mut s = AppState {
            // A device name that won't be in any real enumeration.
            selected_input: Some("__not-a-real-device__".into()),
            ..AppState::default()
        };
        s.refresh_devices();
        // The double-underscore-bracketed name is reserved for tests;
        // no real cpal device will share it, so after refresh it must
        // be cleared.
        assert!(s.selected_input.is_none());
    }

    #[test]
    fn sample_rates_match_streaming_and_audio_io_constants() {
        assert_eq!(
            OUTPUT_SAMPLE_RATE,
            mellonella_audio_io::INTERNAL_SAMPLE_RATE,
            "OUTPUT_SAMPLE_RATE must match mellonella-audio-io's INTERNAL_SAMPLE_RATE",
        );
    }
}
