//! Mutable state owned by the eframe app.
//!
//! Modelled as one flat struct rather than a `SessionState` enum
//! because the UI inspects most fields independently of whether a
//! live session is currently running (enrollment, device selection,
//! last error are all sticky across start/stop cycles).
//!
//! Enrollment is held as an **in-memory `EmbeddingPool`** rather
//! than a path: the GUI offers WAV-file and mic-recording flows
//! that build the pool directly without round-tripping through the
//! enrollment JSON on disk. The JSON load / save buttons are
//! exposed as power-user controls for CLI interop.

use std::path::{Path, PathBuf};

use mellonella_audio_io::{
    list_input_devices, list_output_devices, AudioDevice, LiveSession, LiveSessionStats, Recorder,
    SessionConfig, SessionEvent,
};
use mellonella_core::embedding::EcapaTdnn;
use mellonella_core::enrollment::{EmbeddingPool, EmbeddingPoolConfig};
use mellonella_core::features::Fbank;
use mellonella_core::gating::GateConfig;
use mellonella_core::pipeline::{enroll_from_recording, PipelineComponents, PipelineConfig};
use mellonella_core::resample::resample_to;
use mellonella_core::streaming::StreamingConfig;
use mellonella_core::vad::SileroVad;

/// Output sample rate used end-to-end (matches the CLI's offline
/// constant and `StreamingConfig::default().audio_sample_rate`).
pub const OUTPUT_SAMPLE_RATE: u32 = 48_000;

/// Decision sample rate for VAD / ECAPA / F0 inside the pipeline.
pub const DECISION_SAMPLE_RATE: u32 = 16_000;

/// Default recording duration for the "Record" button. ECAPA's
/// `EmbeddingPoolConfig::default()` accepts a 1 s window minimum,
/// so 5 s is a comfortable margin that gives the model a chance to
/// see speech variability without making the user wait.
pub const DEFAULT_RECORD_SECS: f32 = 5.0;

/// Where the current enrollment came from. Surfaced in the UI so
/// users see "Recorded 5.0 s" vs "Loaded from voice.wav" vs
/// "Loaded enrollment.json".
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum EnrollmentOrigin {
    None,
    Wav(PathBuf),
    Mic { secs: u32 },
    Json(PathBuf),
}

pub struct AppState {
    /// Current speaker pool. `None` until the user enrolls a voice.
    /// Held in-memory so `Start` doesn't have to re-read JSON.
    pub pool: Option<EmbeddingPool>,
    pub origin: EnrollmentOrigin,
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
    /// User-toggled "Enable noise suppression". Honoured only when
    /// `MELLONELLA_DFN3_ONNX` is set; UI disables the checkbox
    /// otherwise (see [`Self::dfn3_available`]).
    pub enable_dfn3: bool,
    pub session: Option<LiveSession>,
    pub recorder: Option<Recorder>,
    pub last_error: Option<String>,
    pub last_stats: LiveSessionStats,
}

impl Default for AppState {
    fn default() -> Self {
        let available_inputs = list_input_devices().unwrap_or_default();
        let available_outputs = list_output_devices().unwrap_or_default();
        Self {
            pool: None,
            origin: EnrollmentOrigin::None,
            pool_anchors: 0,
            pool_f0_mu: 0.0,
            pool_f0_sigma: 0.0,
            available_inputs,
            available_outputs,
            selected_input: None,
            selected_output: None,
            enable_dfn3: false,
            session: None,
            recorder: None,
            last_error: None,
            last_stats: LiveSessionStats::default(),
        }
    }
}

/// `Some(path)` when `MELLONELLA_DFN3_ONNX` is set and the file
/// exists. Used by the UI to decide whether to enable the "Enable
/// noise suppression" checkbox.
#[must_use]
pub fn dfn3_path_from_env() -> Option<PathBuf> {
    let raw = std::env::var_os("MELLONELLA_DFN3_ONNX")?;
    let p = PathBuf::from(raw);
    if p.exists() {
        Some(p)
    } else {
        None
    }
}

impl AppState {
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

    fn store_pool(&mut self, pool: EmbeddingPool, origin: EnrollmentOrigin) {
        let m = pool.metadata();
        self.pool_anchors = pool.anchors().len();
        self.pool_f0_mu = m.f0_mu;
        self.pool_f0_sigma = m.f0_sigma;
        self.origin = origin;
        self.pool = Some(pool);
        self.last_error = None;
    }

    fn clear_pool(&mut self) {
        self.pool = None;
        self.origin = EnrollmentOrigin::None;
        self.pool_anchors = 0;
        self.pool_f0_mu = 0.0;
        self.pool_f0_sigma = 0.0;
    }

    /// Load a pre-computed enrollment JSON (the CLI's
    /// `mellonella enroll` output). Power-user path for parity with
    /// the CLI workflow.
    pub fn load_enrollment_json(&mut self, path: &Path) {
        match EmbeddingPool::load(path, EmbeddingPoolConfig::default()) {
            Ok(pool) => self.store_pool(pool, EnrollmentOrigin::Json(path.to_path_buf())),
            Err(e) => {
                self.clear_pool();
                self.last_error = Some(format!("load enrollment: {e}"));
            }
        }
    }

    /// Save the current enrollment to JSON so it can be re-used
    /// from the CLI or a future session.
    pub fn save_enrollment_json(&mut self, path: &Path) {
        let Some(pool) = self.pool.as_ref() else {
            self.last_error = Some("nothing to save — enrol a voice first".into());
            return;
        };
        match pool.save(path) {
            Ok(()) => {
                self.last_error = None;
            }
            Err(e) => {
                self.last_error = Some(format!("save enrollment: {e}"));
            }
        }
    }

    /// Read a clean voice recording from `path` (16-bit signed mono
    /// WAV at any sample rate; resampled to 16 kHz internally), run
    /// `enroll_from_recording`, and store the resulting pool.
    pub fn enroll_from_wav(&mut self, path: &Path) {
        match read_wav_to_16k_mono(path) {
            Ok(audio) => self.run_enrollment(&audio, EnrollmentOrigin::Wav(path.to_path_buf())),
            Err(e) => {
                self.clear_pool();
                self.last_error = Some(format!("read WAV: {e}"));
            }
        }
    }

    /// Kick off a mic recording of `secs` seconds at 16 kHz mono.
    /// Call `poll_recorder` once per frame to detect completion.
    pub fn start_recording(&mut self, secs: f32) {
        if self.recorder.is_some() {
            return;
        }
        match Recorder::start(self.selected_input.clone(), DECISION_SAMPLE_RATE, secs) {
            Ok(r) => {
                self.recorder = Some(r);
                self.last_error = None;
            }
            Err(e) => self.last_error = Some(format!("start recording: {e}")),
        }
    }

    /// Cancel an in-flight recording. The worker still returns
    /// whatever it has collected, which the next `poll_recorder`
    /// converts into an enrollment.
    pub fn cancel_recording(&mut self) {
        if let Some(r) = self.recorder.as_ref() {
            r.cancel();
        }
    }

    /// Poll the active recorder for completion. On success runs
    /// enrolment on the captured buffer and stores the resulting
    /// pool. Returns silently when no recorder is active or it's
    /// still capturing.
    pub fn poll_recorder(&mut self) {
        let Some(recorder) = self.recorder.as_mut() else {
            return;
        };
        let Some(result) = recorder.try_finish() else {
            return;
        };
        // Recorder finished; pull it out and consume the result.
        let target_secs = recorder.target_seconds().round() as u32;
        self.recorder = None;
        match result {
            Ok(audio) => {
                if audio.len() < DECISION_SAMPLE_RATE as usize {
                    self.last_error =
                        Some("recording too short for ECAPA enrolment (need ≥ 1 s)".to_string());
                    return;
                }
                self.run_enrollment(&audio, EnrollmentOrigin::Mic { secs: target_secs });
            }
            Err(e) => self.last_error = Some(format!("recording failed: {e}")),
        }
    }

    fn run_enrollment(&mut self, audio_16k: &[f32], origin: EnrollmentOrigin) {
        let mut components = match Self::build_components() {
            Ok(c) => c,
            Err(e) => {
                self.last_error = Some(e);
                return;
            }
        };
        match enroll_from_recording(audio_16k, &mut components, EmbeddingPoolConfig::default()) {
            Ok(pool) => self.store_pool(pool, origin),
            Err(e) => self.last_error = Some(format!("enrol: {e}")),
        }
    }

    /// Spin up a `LiveSession` using the current in-memory pool.
    pub fn start(&mut self) {
        if self.session.is_some() {
            return;
        }
        let Some(pool) = self.pool.clone() else {
            self.last_error = Some("enrol a voice first".into());
            return;
        };
        let components = match Self::build_components() {
            Ok(c) => c,
            Err(e) => {
                self.last_error = Some(e);
                return;
            }
        };
        let dfn3_onnx_path = if self.enable_dfn3 {
            if let Some(p) = dfn3_path_from_env() {
                Some(p)
            } else {
                self.last_error = Some(
                    "MELLONELLA_DFN3_ONNX env var not set — can't enable noise suppression".into(),
                );
                return;
            }
        } else {
            None
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
            dfn3_onnx_path,
            // GUI uses the safe default; multi-channel mic users
            // who want a specific channel use the CLI's
            // `mellonella live --input-channel N` for now. A GUI
            // dropdown is a small follow-up.
            input_channel: mellonella_audio_io::ChannelStrategy::default(),
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
            self.stop();
        }
    }

    #[must_use]
    pub fn is_running(&self) -> bool {
        self.session.is_some()
    }

    #[must_use]
    pub fn is_recording(&self) -> bool {
        self.recorder.is_some()
    }

    #[must_use]
    pub fn can_start(&self) -> bool {
        self.pool.is_some() && self.session.is_none() && self.recorder.is_none()
    }

    /// Whether DFN3 noise suppression is reachable in this process
    /// (env var set, file exists). The UI greys the toggle when
    /// this returns `false`.
    #[must_use]
    #[allow(clippy::unused_self)] // method form is more discoverable from app.rs
    pub fn dfn3_available(&self) -> bool {
        dfn3_path_from_env().is_some()
    }

    /// Latest input RMS from the worker (0.0 when no session is
    /// running). Used by the GUI's level meter.
    #[must_use]
    pub fn input_rms(&self) -> f32 {
        self.session.as_ref().map_or(0.0, LiveSession::input_rms)
    }

    /// Latest output (gate × envelope) RMS from the worker.
    #[must_use]
    pub fn output_rms(&self) -> f32 {
        self.session.as_ref().map_or(0.0, LiveSession::output_rms)
    }

    /// Latest gate state — `true` when audio is currently being
    /// passed through. `false` for both "gated off" and "no session".
    #[must_use]
    pub fn gate_on(&self) -> bool {
        self.session.as_ref().is_some_and(LiveSession::gate_on)
    }

    /// Estimated end-to-end output latency for the current
    /// configuration, in milliseconds. Used by the GUI status row
    /// so users can verify the live filter is in the right
    /// ballpark for their use case (e.g. ≪ 100 ms for headphone
    /// monitoring, < 200 ms for call apps).
    ///
    /// Breakdown:
    ///
    /// * resampler: ~5 ms
    /// * VAD: < 10 ms (assume 8 ms)
    /// * envelope attack: `gate.attack_ms`
    /// * DFN3: ~30 ms when `enable_dfn3` is on, else 0
    ///
    /// This is the architecture doc's published budget — values
    /// are conservative upper bounds, not measured per-frame on
    /// the host. Useful as a sanity check, not a benchmark.
    #[must_use]
    pub fn estimated_latency_ms(&self) -> f32 {
        let gate_cfg = GateConfig::default();
        let mut total = 5.0_f32 + 8.0 + gate_cfg.attack_ms;
        if self.enable_dfn3 && self.dfn3_available() {
            total += 30.0;
        }
        total
    }
}

/// Decode a 16-bit signed mono WAV at any common rate into f32
/// samples at 16 kHz mono — the rate `enroll_from_recording`
/// expects. Mirrors the CLI's `read_wav_mono` + resample helper but
/// inline so the GUI doesn't pull the CLI binary in as a dep.
fn read_wav_to_16k_mono(path: &Path) -> Result<Vec<f32>, String> {
    let mut reader = hound::WavReader::open(path).map_err(|e| e.to_string())?;
    let spec = reader.spec();
    if spec.channels != 1 {
        return Err(format!("expected mono, got {} channels", spec.channels));
    }
    if spec.sample_format != hound::SampleFormat::Int || spec.bits_per_sample != 16 {
        return Err(format!(
            "expected 16-bit signed int, got {:?} {}-bit",
            spec.sample_format, spec.bits_per_sample
        ));
    }
    let scale = 1.0_f32 / f32::from(i16::MAX);
    let samples: Vec<f32> = reader
        .samples::<i16>()
        .map(|s| s.map(|v| f32::from(v) * scale))
        .collect::<Result<Vec<f32>, _>>()
        .map_err(|e| e.to_string())?;
    if spec.sample_rate == DECISION_SAMPLE_RATE {
        return Ok(samples);
    }
    resample_to(&samples, spec.sample_rate, DECISION_SAMPLE_RATE).map_err(|e| e.to_string())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_state_is_idle() {
        let s = AppState::default();
        assert!(s.pool.is_none());
        assert_eq!(s.pool_anchors, 0);
        assert!(matches!(s.origin, EnrollmentOrigin::None));
        assert!(!s.is_running());
        assert!(!s.is_recording());
        assert!(!s.can_start());
        assert!(s.last_error.is_none());
    }

    #[test]
    fn loading_a_missing_enrollment_records_an_error() {
        let mut s = AppState::default();
        s.load_enrollment_json(std::path::Path::new(
            "/no/such/enrollment-mellonella-gui-test.json",
        ));
        assert!(s.last_error.is_some(), "expected an error to be recorded");
        assert!(s.pool.is_none());
        assert!(matches!(s.origin, EnrollmentOrigin::None));
        assert!(!s.can_start());
    }

    #[test]
    fn enroll_from_missing_wav_records_an_error() {
        let mut s = AppState::default();
        s.enroll_from_wav(std::path::Path::new(
            "/no/such/audio-mellonella-gui-test.wav",
        ));
        assert!(s.last_error.is_some());
        assert!(s.pool.is_none());
    }

    #[test]
    fn refresh_devices_clears_invalid_selection() {
        let mut s = AppState {
            selected_input: Some("__not-a-real-device__".into()),
            ..AppState::default()
        };
        s.refresh_devices();
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

    #[test]
    fn estimated_latency_without_dfn3_is_under_50_ms() {
        let s = AppState::default();
        let latency = s.estimated_latency_ms();
        assert!(
            (15.0..=50.0).contains(&latency),
            "without DFN3 the budget should be in the 15–50 ms range, got {latency}"
        );
    }

    #[test]
    fn estimated_latency_increases_with_dfn3() {
        let mut s = AppState::default();
        let without = s.estimated_latency_ms();
        s.enable_dfn3 = true;
        let with = s.estimated_latency_ms();
        // DFN3 only counts when the env var also points at a real
        // file — which `default()` doesn't guarantee. Either the
        // values are equal (env unset) or DFN3 adds ~30 ms.
        if with > without {
            let delta = with - without;
            assert!(
                (25.0..=35.0).contains(&delta),
                "DFN3 should add ~30 ms, got {delta}"
            );
        }
    }
}
