//! Mutable state owned by the eframe app.
//!
//! Modelled as one flat struct rather than a `SessionState` enum
//! because the UI inspects most fields independently of whether a
//! live session is currently running (enrollment, device selection,
//! last error are all sticky across start/stop cycles).
//!
//! Enrollment is held as an **in-memory `EmbeddingPool`** built from
//! the mic recording flow and updated by the auto-learn pool during
//! a live session. The pool is persisted to
//! `~/.config/mellonella/enrollment.json` and auto-loaded on next
//! launch, so file-level import / export controls are unnecessary.

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
use mellonella_core::tse_stage::TseStageConfig;
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
/// users see "Recorded 5.0 s" vs the auto-loaded persistent pool.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum EnrollmentOrigin {
    None,
    Mic {
        secs: u32,
    },
    /// Auto-loaded from `default_enrollment_path()` on launch. Path
    /// retained for display so the user can see which file backs the
    /// in-memory pool.
    AutoLoaded(PathBuf),
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
    /// Path to the TSE streaming ONNX (Stage C). Picked via a file
    /// dialog in the Settings panel; persisted in-memory only for this
    /// session.
    pub tse_onnx_path: Option<PathBuf>,
    /// Mic-enrollment recording duration in seconds. Step 20 made
    /// this user-configurable from the GUI (a slider next to the
    /// Record button); default matches the previous fixed
    /// [`DEFAULT_RECORD_SECS`] value.
    pub record_duration_secs: f32,
    /// User-adjustable gate / envelope parameters. Sliders in the
    /// Settings panel mutate these in place; `start()` reads them
    /// when building the `SessionConfig`. Defaults match
    /// `GateConfig::default()`.
    pub gate_cfg: GateConfig,
    /// User-adjustable pipeline cadence (currently just
    /// `sv_update_samples` — ECAPA refresh interval). Sliders in
    /// the Settings panel mutate this; defaults match
    /// `PipelineConfig::default()`.
    pub pipeline_cfg: PipelineConfig,
    pub session: Option<LiveSession>,
    pub recorder: Option<Recorder>,
    pub last_error: Option<String>,
    pub last_stats: LiveSessionStats,
}

impl Default for AppState {
    fn default() -> Self {
        let available_inputs = list_input_devices().unwrap_or_default();
        let available_outputs = list_output_devices().unwrap_or_default();
        let mut state = Self {
            pool: None,
            origin: EnrollmentOrigin::None,
            pool_anchors: 0,
            pool_f0_mu: 0.0,
            pool_f0_sigma: 0.0,
            available_inputs,
            available_outputs,
            selected_input: None,
            selected_output: None,
            tse_onnx_path: None,
            record_duration_secs: DEFAULT_RECORD_SECS,
            gate_cfg: GateConfig::default(),
            pipeline_cfg: default_live_pipeline_cfg(),
            session: None,
            recorder: None,
            last_error: None,
            last_stats: LiveSessionStats::default(),
        };
        // Auto-load the persistent enrollment if one exists. The user
        // shouldn't have to re-enrol on every launch; the first-run
        // wizard in `crate::app` keeps prompting only until this fires.
        if let Some(path) = default_enrollment_path() {
            if path.exists() {
                state.load_enrollment_json(&path);
            }
        }
        state
    }
}

/// Default on-disk location for the auto-saved enrollment:
/// `<dirs::config_dir>/mellonella/enrollment.json`. Returns `None` on
/// platforms where `dirs::config_dir()` is unavailable (rare).
#[must_use]
pub fn default_enrollment_path() -> Option<PathBuf> {
    let dir = dirs::config_dir()?.join("mellonella");
    Some(dir.join("enrollment.json"))
}

/// Live-tuned `PipelineConfig` for GUI sessions: takes the library
/// defaults and overrides the fields whose library defaults are
/// backward-compat no-ops (so existing tests against
/// `PipelineConfig::default()` keep passing) but that matter for
/// real-time mic use:
///
/// * `silence_force_off_ms = 400` — close the gate immediately after
///   400 ms of continuous VAD-silence. Longer than a typical
///   inter-word pause (~250 ms) so normal speech doesn't trip it,
///   but short enough that the gate closes within ~500 ms of the
///   user actually stopping (400 ms + 100 ms envelope `release_ms`).
/// * `score_ema_alpha = 0.7` — smooth `last_score` updates across
///   refreshes to ride out one-refresh dips at speech onset.
/// * `sv_min_new_samples_after_silence = 1600` — fire the
///   post-silence early refresh after only 100 ms of new speech
///   (instead of the library-default 250 ms) so `last_score` catches
///   up to the current speaker faster on resume. Cheap for a
///   single-target system since there's no cross-speaker risk.
#[must_use]
pub fn default_live_pipeline_cfg() -> PipelineConfig {
    PipelineConfig {
        silence_force_off_ms: 400.0,
        score_ema_alpha: 0.7,
        sv_min_new_samples_after_silence: 1_600,
        ..PipelineConfig::default()
    }
}

/// `Some(path)` when the DFN3 ONNX is reachable — either via the
/// `MELLONELLA_DFN3_ONNX` env var or the on-disk cache populated by
/// [`mellonella_core::hf_fetch::ensure_dfn3_onnx`]. Used by the UI's
/// status row and by [`AppState::start`] to decide whether to wire
/// DFN3 into the live engine. Cheap (no network) — the actual fetch
/// happens elsewhere; this is just a "is the file there yet?" probe.
#[must_use]
pub fn dfn3_path_from_env() -> Option<PathBuf> {
    if let Some(raw) = std::env::var_os("MELLONELLA_DFN3_ONNX") {
        let p = PathBuf::from(raw);
        if p.exists() {
            return Some(p);
        }
    }
    let cached = mellonella_core::hf_fetch::cached_path(
        mellonella_core::hf_fetch::DFN3_REPO,
        mellonella_core::hf_fetch::DFN3_FILE,
    )
    .ok()?;
    cached.exists().then_some(cached)
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

    /// Build a fresh `PipelineComponents`, resolving ONNX paths through
    /// the cache-first fallback chain in [`mellonella_core::hf_fetch`]
    /// (env var → cache → HuggingFace fetch). Falls back gracefully to
    /// the legacy env-var-only setup for models without an HF mirror.
    fn build_components() -> Result<PipelineComponents, String> {
        let ecapa_path = mellonella_core::hf_fetch::ensure_ecapa_onnx(|_, _| {})
            .map_err(|e| format!("ECAPA ONNX: {e}"))?;
        let vad_path = mellonella_core::hf_fetch::ensure_vad_onnx(|_, _| {})
            .map_err(|e| format!("VAD ONNX: {e}"))?;
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
            tse: None,
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

    /// Auto-load the persistent enrollment from
    /// `default_enrollment_path()` on launch. The enrollment pool is
    /// otherwise managed entirely in-memory: mic recording builds it,
    /// auto-learn updates it during a live session, and
    /// [`Self::persist_enrollment_to_default_path`] writes it back.
    fn load_enrollment_json(&mut self, path: &Path) {
        match EmbeddingPool::load(path, EmbeddingPoolConfig::default()) {
            Ok(pool) => self.store_pool(pool, EnrollmentOrigin::AutoLoaded(path.to_path_buf())),
            Err(e) => {
                self.clear_pool();
                self.last_error = Some(format!("load enrollment: {e}"));
            }
        }
    }

    /// Kick off a mic recording of `secs` seconds at 48 kHz mono —
    /// matches the live audio path's rate so an optional DFN3
    /// pre-pass during enrollment runs on the same distribution as
    /// the live ECAPA refresh path. Call `poll_recorder` once per
    /// frame to detect completion.
    pub fn start_recording(&mut self, secs: f32) {
        if self.recorder.is_some() {
            return;
        }
        match Recorder::start(self.selected_input.clone(), OUTPUT_SAMPLE_RATE, secs) {
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
                if audio.len() < OUTPUT_SAMPLE_RATE as usize {
                    self.last_error =
                        Some("recording too short for ECAPA enrolment (need ≥ 1 s)".to_string());
                    return;
                }
                self.run_enrollment(&audio, EnrollmentOrigin::Mic { secs: target_secs });
            }
            Err(e) => self.last_error = Some(format!("recording failed: {e}")),
        }
    }

    /// Enroll from 48 kHz mono audio. Downsamples to 16 kHz and runs
    /// ECAPA on the raw signal, matching the live engine's decision
    /// path: after the Phase 5 refactor, DFN3 lives post-TSE in the
    /// audio chain, while VAD / ECAPA / F0 see the raw mic input.
    /// Keeping enrollment on raw audio matches that distribution so
    /// anchor embeddings live in the same space as runtime refreshes.
    fn run_enrollment(&mut self, audio_48k: &[f32], origin: EnrollmentOrigin) {
        let audio_16k = match resample_to(audio_48k, OUTPUT_SAMPLE_RATE, DECISION_SAMPLE_RATE) {
            Ok(a) => a,
            Err(e) => {
                self.last_error = Some(format!("resample 48 kHz → 16 kHz for ECAPA: {e}"));
                return;
            }
        };
        let mut components = match Self::build_components() {
            Ok(c) => c,
            Err(e) => {
                self.last_error = Some(e);
                return;
            }
        };
        match enroll_from_recording(&audio_16k, &mut components, EmbeddingPoolConfig::default()) {
            Ok(pool) => {
                self.store_pool(pool, origin);
                self.persist_enrollment_to_default_path();
            }
            Err(e) => self.last_error = Some(format!("enrol: {e}")),
        }
    }

    /// Save the current enrollment pool to `default_enrollment_path()`
    /// so the next launch auto-loads it. No-op when no pool is loaded
    /// or no platform config dir is configured.
    fn persist_enrollment_to_default_path(&mut self) {
        let Some(pool) = self.pool.as_ref() else {
            return;
        };
        let Some(path) = default_enrollment_path() else {
            return;
        };
        if let Some(parent) = path.parent() {
            if let Err(e) = std::fs::create_dir_all(parent) {
                self.last_error = Some(format!("create config dir: {e}"));
                return;
            }
        }
        if let Err(e) = pool.save(&path) {
            self.last_error = Some(format!("auto-save enrollment: {e}"));
        }
    }

    /// Spin up a `LiveSession` using the current in-memory pool.
    ///
    /// DFN3 and TSE are always enabled when their ONNX models are
    /// reachable (DFN3 via `MELLONELLA_DFN3_ONNX`, TSE via
    /// [`Self::tse_onnx_path`]). The GUI no longer exposes toggles
    /// for them — the engine handles dual-stage absence gracefully
    /// when an ONNX is unavailable.
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
        let dfn3_onnx_path = dfn3_path_from_env();
        let mut pipeline_cfg = self.pipeline_cfg.clone();
        // Auto-resolve TSE ONNX if the user hasn't picked one — the
        // fetcher's cache hit path makes this cheap on every launch
        // after the first.
        if self.tse_onnx_path.is_none() {
            if let Ok(p) = mellonella_core::hf_fetch::ensure_tse_prod_48k_onnx(|_, _, _| {}) {
                self.tse_onnx_path = Some(p);
            }
        }
        if let Some(onnx) = self.tse_onnx_path.as_ref() {
            if !onnx.exists() {
                self.last_error = Some(format!("TSE ONNX path does not exist: {}", onnx.display()));
                return;
            }
            pipeline_cfg.tse = Some(TseStageConfig::new_prod_48k(onnx.clone()));
        }
        let cfg = SessionConfig {
            input_device: self.selected_input.clone(),
            output_device: self.selected_output.clone(),
            streaming: StreamingConfig {
                pipeline: pipeline_cfg,
                gate: self.gate_cfg,
                audio_sample_rate: OUTPUT_SAMPLE_RATE,
                diagnostics: false,
                // `LiveSession::new` overwrites this from `dfn3_onnx_path`
                // on `SessionConfig` below — keep `None` here as the
                // construction-time default.
                dfn3_onnx_path: None,
            },
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

    /// Whether DFN3 noise suppression is reachable in this process
    /// (env var set, file exists). Used by the status row to indicate
    /// "NS active" and by [`Self::estimated_latency_ms`] to factor in
    /// the DFN3 lookahead.
    #[must_use]
    #[allow(clippy::unused_self)] // method form is more discoverable from app.rs
    pub fn dfn3_available(&self) -> bool {
        dfn3_path_from_env().is_some()
    }

    /// Whether a TSE ONNX path has been configured and still exists
    /// on disk. Used by the status row.
    #[must_use]
    pub fn tse_available(&self) -> bool {
        self.tse_onnx_path.as_deref().is_some_and(Path::exists)
    }

    /// Download the canonical Stage C TSE prod_48k model from
    /// HuggingFace into the local cache (synchronous; the UI freezes
    /// briefly during the ~10 MB download) and update
    /// [`Self::tse_onnx_path`] to point at the cached file. Reuses an
    /// already-cached copy on subsequent calls.
    ///
    /// Surfaces failures via [`Self::last_error`] rather than
    /// `Result` so the egui call site stays click-handler-shaped.
    pub fn fetch_tse_from_hf(&mut self) {
        match mellonella_core::hf_fetch::fetch_tse_prod_48k(|_, _, _| {}) {
            Ok(path) => {
                self.tse_onnx_path = Some(path);
                self.last_error = None;
            }
            Err(e) => {
                self.last_error = Some(format!("HuggingFace fetch failed: {e}"));
            }
        }
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
        assert!(s.pool.is_none());
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
    fn default_enrollment_path_lives_under_config_dir() {
        let Some(p) = default_enrollment_path() else {
            eprintln!("[skip] no config dir on this platform");
            return;
        };
        assert!(
            p.ends_with("mellonella/enrollment.json") || p.ends_with("mellonella\\enrollment.json"),
            "unexpected suffix: {}",
            p.display()
        );
    }

    // ----------------------------------------------------------------
    // ONNX-backed end-to-end checks. Gated on the same env vars as the
    // mellonella-core integration tests (`MELLONELLA_ECAPA_ONNX`,
    // `MELLONELLA_VAD_ONNX`, `ORT_DYLIB_PATH`) so a contributor without
    // the model artefacts still gets a green `cargo test`. The
    // persistence helpers below stash and restore any pre-existing
    // `default_enrollment_path()` file so they don't clobber a real
    // profile.
    // ----------------------------------------------------------------

    fn skip_if_no_onnx() -> Option<(String, String)> {
        let Ok(ecapa) = std::env::var("MELLONELLA_ECAPA_ONNX") else {
            eprintln!("[skip] MELLONELLA_ECAPA_ONNX not set");
            return None;
        };
        let Ok(vad) = std::env::var("MELLONELLA_VAD_ONNX") else {
            eprintln!("[skip] MELLONELLA_VAD_ONNX not set");
            return None;
        };
        if std::env::var("ORT_DYLIB_PATH").is_err() {
            eprintln!("[skip] ORT_DYLIB_PATH not set");
            return None;
        }
        Some((ecapa, vad))
    }

    fn enroll_pool_from_fixture(ecapa: &str, vad: &str) -> EmbeddingPool {
        use mellonella_core::embedding::EcapaTdnn;
        use mellonella_core::features::Fbank;
        use mellonella_core::vad::SileroVad;
        let fixture = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("..")
            .join("mellonella-core")
            .join("tests")
            .join("fixtures")
            .join("pipeline_input.bin");
        let bytes = std::fs::read(&fixture).expect("read pipeline_input.bin");
        assert!(bytes.len().is_multiple_of(4));
        let audio_16k: Vec<f32> = bytes
            .chunks_exact(4)
            .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect();
        let mut components = PipelineComponents {
            vad: SileroVad::from_onnx_path(vad, 16_000).expect("load VAD"),
            fbank: Fbank::with_speechbrain_filterbank().expect("Fbank from speechbrain filterbank"),
            ecapa: EcapaTdnn::from_onnx_path(ecapa).expect("load ECAPA"),
            cohort: Vec::new(),
            tse: None,
        };
        enroll_from_recording(&audio_16k, &mut components, EmbeddingPoolConfig::default())
            .expect("enroll_from_recording")
    }

    /// Swap the on-disk `default_enrollment_path()` with `pool` for
    /// the duration of `body`, restoring whatever was there.
    fn with_test_enrollment(pool: &EmbeddingPool, body: impl FnOnce()) {
        let path = default_enrollment_path().expect("config dir available");
        if let Some(parent) = path.parent() {
            std::fs::create_dir_all(parent).expect("create config dir");
        }
        let backup = path.with_extension("json.testbackup");
        let had_backup = path.exists();
        if had_backup {
            std::fs::rename(&path, &backup).expect("backup existing enrollment");
        }
        pool.save(&path).expect("save test pool");
        body();
        let _ = std::fs::remove_file(&path);
        if had_backup {
            std::fs::rename(&backup, &path).expect("restore existing enrollment");
        }
    }

    #[test]
    fn default_app_state_auto_loads_persisted_enrollment() {
        let Some((ecapa, vad)) = skip_if_no_onnx() else {
            return;
        };
        let pool = enroll_pool_from_fixture(&ecapa, &vad);
        let expected_dim = pool.anchor_centroid().expect("anchor dim").len();
        with_test_enrollment(&pool, || {
            let state = AppState::default();
            assert!(state.pool.is_some(), "should auto-load enrollment.json");
            assert!(
                matches!(state.origin, EnrollmentOrigin::AutoLoaded(_)),
                "expected AutoLoaded origin, got {:?}",
                state.origin
            );
            assert!(state.pool_anchors >= 1);
            let loaded_dim = state
                .pool
                .as_ref()
                .unwrap()
                .anchor_centroid()
                .unwrap()
                .len();
            assert_eq!(loaded_dim, expected_dim);
        });
    }

    /// Drive the GUI's exact pipeline configuration end-to-end on an
    /// offline buffer, no cpal involved. This is the headless analog
    /// of the user pressing Live monitor — it exercises the same TSE
    /// → DFN3 → gate → envelope chain that the worker runs.
    ///
    /// Gated on `MELLONELLA_TSE_PROD_48K_ONNX` and
    /// `MELLONELLA_DFN3_ONNX` *in addition* to the ECAPA / VAD pair —
    /// the GUI auto-enables both stages when the models are present,
    /// so the test follows the same rule.
    ///
    /// When `MELLONELLA_DUMP_OFFLINE_WAV` is set, the input (resampled)
    /// and chain output are also written to `/tmp/mellonella_offline_*.wav`
    /// so a developer reproducing a "weird audio output" complaint
    /// can listen to what the pipeline actually produced.
    #[test]
    #[allow(clippy::too_many_lines)]
    fn streaming_pipeline_runs_clean_on_offline_speech() {
        use mellonella_core::resample::resample_to;
        use mellonella_core::streaming::{StreamingConfig, StreamingPipeline};
        use mellonella_core::tse_stage::TseStageConfig;

        let Some((ecapa, vad)) = skip_if_no_onnx() else {
            return;
        };
        let Ok(tse_path) = std::env::var("MELLONELLA_TSE_PROD_48K_ONNX") else {
            eprintln!("[skip] MELLONELLA_TSE_PROD_48K_ONNX not set");
            return;
        };
        let Ok(dfn3_path) = std::env::var("MELLONELLA_DFN3_ONNX") else {
            eprintln!("[skip] MELLONELLA_DFN3_ONNX not set");
            return;
        };

        // 1) Enroll a pool on the canonical 16 kHz fixture and grab
        //    the pipeline components.
        let pool = enroll_pool_from_fixture(&ecapa, &vad);
        let components = PipelineComponents {
            vad: mellonella_core::vad::SileroVad::from_onnx_path(&vad, 16_000).unwrap(),
            fbank: mellonella_core::features::Fbank::with_speechbrain_filterbank().unwrap(),
            ecapa: mellonella_core::embedding::EcapaTdnn::from_onnx_path(&ecapa).unwrap(),
            cohort: Vec::new(),
            tse: None,
        };

        // 2) Load the audio path. By default uses the 2 s 16 kHz
        //    `pipeline_input.bin` fixture and upsamples to 48 kHz.
        //    Set `MELLONELLA_OFFLINE_INPUT_WAV=<path.wav>` to point
        //    the test at a longer, real-world recording for
        //    repro-grade ear-checks; the test reads it as 16-bit
        //    signed mono and resamples to 48 kHz internally.
        let audio_48k: Vec<f32> = if let Ok(p) = std::env::var("MELLONELLA_OFFLINE_INPUT_WAV") {
            let (samples_native, native_sr) = read_pcm16_mono_wav(&p);
            if native_sr == OUTPUT_SAMPLE_RATE {
                samples_native
            } else {
                resample_to(&samples_native, native_sr, OUTPUT_SAMPLE_RATE)
                    .expect("input WAV → 48 kHz resample")
            }
        } else {
            let fixture = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
                .join("..")
                .join("mellonella-core")
                .join("tests")
                .join("fixtures")
                .join("pipeline_input.bin");
            let bytes = std::fs::read(&fixture).unwrap();
            let audio_16k: Vec<f32> = bytes
                .chunks_exact(4)
                .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
                .collect();
            resample_to(&audio_16k, 16_000, OUTPUT_SAMPLE_RATE)
                .expect("16k → 48k resample for offline streaming run")
        };

        // 3) Build a StreamingConfig that mirrors the GUI's
        //    `AppState::start()` exactly — same gate, same pipeline
        //    cadence, TSE Prod48k enabled, DFN3 enabled.
        let mut pipeline_cfg = default_live_pipeline_cfg();
        pipeline_cfg.tse = Some(TseStageConfig::new_prod_48k(PathBuf::from(&tse_path)));
        let cfg = StreamingConfig {
            pipeline: pipeline_cfg,
            gate: GateConfig::default(),
            audio_sample_rate: OUTPUT_SAMPLE_RATE,
            diagnostics: false,
            dfn3_onnx_path: Some(PathBuf::from(&dfn3_path)),
        };

        let mut pipeline = StreamingPipeline::new(pool, cfg, components)
            .expect("StreamingPipeline accepts the GUI config");

        // 4) Push the audio in 480-sample (10 ms) chunks, mirroring
        //    the worker's actual cadence. Track diagnostics that
        //    the user can use to decide whether the chain is doing
        //    something pathological on real speech.
        let chunk_size: usize = 480;
        let mut all_output: Vec<f32> = Vec::with_capacity(audio_48k.len() + chunk_size);
        let mut nan_count = 0_usize;
        let mut inf_count = 0_usize;
        let mut gate_transitions = 0_usize;
        let mut last_gate_state: Option<bool> = None;
        let mut gate_on_samples = 0_u64;
        let mut chunks_pushed = 0_u64;
        let mut zero_output_chunks = 0_u64;
        let t0 = std::time::Instant::now();
        for chunk in audio_48k.chunks(chunk_size) {
            let out = pipeline
                .push_samples(chunk)
                .expect("push_samples in offline streaming run");
            chunks_pushed += 1;
            if out.audio.is_empty() {
                zero_output_chunks += 1;
            }
            for &s in &out.audio {
                if s.is_nan() {
                    nan_count += 1;
                } else if !s.is_finite() {
                    inf_count += 1;
                }
            }
            for &(_, is_on) in &out.gate_decisions {
                if Some(is_on) != last_gate_state {
                    gate_transitions += 1;
                    last_gate_state = Some(is_on);
                }
            }
            if let Some(true) = last_gate_state {
                gate_on_samples += out.audio.len() as u64;
            }
            all_output.extend_from_slice(&out.audio);
        }
        let tail = pipeline.flush().expect("flush");
        for &s in &tail.audio {
            if s.is_nan() {
                nan_count += 1;
            } else if !s.is_finite() {
                inf_count += 1;
            }
        }
        all_output.extend_from_slice(&tail.audio);
        let wall_ms = t0.elapsed().as_millis();
        let audio_ms = audio_48k.len() as f64 / f64::from(OUTPUT_SAMPLE_RATE) * 1000.0;
        let realtime_factor = audio_ms / wall_ms.max(1) as f64;

        let rms_in = (audio_48k.iter().map(|s| s * s).sum::<f32>() / audio_48k.len() as f32).sqrt();
        let rms_out =
            (all_output.iter().map(|s| s * s).sum::<f32>() / all_output.len().max(1) as f32).sqrt();
        let peak_in = audio_48k.iter().map(|s| s.abs()).fold(0.0_f32, f32::max);
        let peak_out = all_output.iter().map(|s| s.abs()).fold(0.0_f32, f32::max);
        let total_out = all_output.len();
        let rms_db = 20.0 * (rms_out.max(1e-12) / rms_in.max(1e-12)).log10();
        let peak_db = 20.0 * (peak_out.max(1e-12) / peak_in.max(1e-12)).log10();

        eprintln!(
            "[offline] {chunks_pushed} chunks pushed, {zero_output_chunks} produced no output, \
             {wall_ms} ms wall ({realtime_factor:.2}× realtime)"
        );
        eprintln!(
            "[offline] gate transitions: {gate_transitions}, samples gate-on / total: \
             {gate_on_samples} / {total_out}"
        );
        eprintln!("[offline] RMS  in={rms_in:.4}  out={rms_out:.4}  ({rms_db:+.1} dB)");
        eprintln!("[offline] PEAK in={peak_in:.4}  out={peak_out:.4}  ({peak_db:+.1} dB)");

        // 5) Assertions: the chain must produce finite samples and a
        //    length within shouting distance of the input (chain
        //    buffering can trim or pad by a few frames; we allow
        //    ±200 ms = ±9600 samples of slack at 48 kHz).
        assert_eq!(
            nan_count, 0,
            "pipeline emitted {nan_count} NaN samples — TSE / DFN3 state is going non-finite"
        );
        assert_eq!(
            inf_count, 0,
            "pipeline emitted {inf_count} Inf samples — clipping or division-by-zero somewhere"
        );
        let slack = OUTPUT_SAMPLE_RATE as usize / 5;
        let len_delta = all_output.len().abs_diff(audio_48k.len());
        let total_in = audio_48k.len();
        assert!(
            len_delta <= slack,
            "output length {total_out} too far from input {total_in} \
             (delta {len_delta}, slack {slack})"
        );

        // 6) Optional artefact dump for ear-checking the chain.
        if std::env::var_os("MELLONELLA_DUMP_OFFLINE_WAV").is_some() {
            let in_path = "/tmp/mellonella_offline_input_48k.wav";
            let out_path = "/tmp/mellonella_offline_output_48k.wav";
            write_f32_wav(in_path, &audio_48k, hound_lite_spec());
            write_f32_wav(out_path, &all_output, hound_lite_spec());
            eprintln!(
                "[offline] wrote {} ({} samples) and {} ({} samples)",
                in_path,
                audio_48k.len(),
                out_path,
                all_output.len()
            );
        }
    }

    /// Minimal 16-bit / signed / mono WAV reader. Returns
    /// `(samples in [-1, 1], native sample rate)`. Panics on
    /// malformed input — fine for a developer harness.
    fn read_pcm16_mono_wav(path: &str) -> (Vec<f32>, u32) {
        let bytes = std::fs::read(path).expect("read input WAV");
        assert!(bytes.len() > 44, "WAV too short: {}", bytes.len());
        assert_eq!(&bytes[..4], b"RIFF");
        assert_eq!(&bytes[8..12], b"WAVE");
        // Find the `fmt ` and `data` chunks (the standard offsets
        // are only valid when there are no extension chunks before
        // `data`, which is overwhelmingly common but not guaranteed).
        let mut i = 12_usize;
        let mut fmt_off: Option<usize> = None;
        let mut data_off: Option<(usize, usize)> = None;
        while i + 8 <= bytes.len() {
            let id = &bytes[i..i + 4];
            let sz = u32::from_le_bytes([bytes[i + 4], bytes[i + 5], bytes[i + 6], bytes[i + 7]])
                as usize;
            match id {
                b"fmt " => fmt_off = Some(i + 8),
                b"data" => data_off = Some((i + 8, sz)),
                _ => {}
            }
            i += 8 + sz;
        }
        let fmt = fmt_off.expect("WAV has no fmt chunk");
        let (data, data_len) = data_off.expect("WAV has no data chunk");
        let audio_format = u16::from_le_bytes([bytes[fmt], bytes[fmt + 1]]);
        let channels = u16::from_le_bytes([bytes[fmt + 2], bytes[fmt + 3]]);
        let sample_rate = u32::from_le_bytes([
            bytes[fmt + 4],
            bytes[fmt + 5],
            bytes[fmt + 6],
            bytes[fmt + 7],
        ]);
        let bits = u16::from_le_bytes([bytes[fmt + 14], bytes[fmt + 15]]);
        assert_eq!(audio_format, 1, "expected PCM (1), got {audio_format}");
        assert_eq!(channels, 1, "expected mono, got {channels} channels");
        assert_eq!(bits, 16, "expected 16-bit, got {bits}-bit");
        let scale = 1.0_f32 / f32::from(i16::MAX);
        let samples: Vec<f32> = bytes[data..data + data_len]
            .chunks_exact(2)
            .map(|c| f32::from(i16::from_le_bytes([c[0], c[1]])) * scale)
            .collect();
        (samples, sample_rate)
    }

    /// Minimal 16-bit / 48 kHz / mono WAV writer config. Inlined so
    /// the test stays dependency-free (the GUI crate no longer pulls
    /// `hound` after the latest cleanup).
    #[derive(Clone, Copy)]
    struct WavSpec {
        channels: u16,
        sample_rate: u32,
    }
    fn hound_lite_spec() -> WavSpec {
        WavSpec {
            channels: 1,
            sample_rate: OUTPUT_SAMPLE_RATE,
        }
    }
    fn write_f32_wav(path: &str, samples: &[f32], spec: WavSpec) {
        use std::io::Write;
        let bits_per_sample = 16_u16;
        let byte_rate =
            spec.sample_rate * u32::from(spec.channels) * u32::from(bits_per_sample / 8);
        let block_align = spec.channels * (bits_per_sample / 8);
        let data_bytes: u32 = samples.len() as u32 * u32::from(bits_per_sample / 8);
        let mut f = std::fs::File::create(path).expect("create wav");
        f.write_all(b"RIFF").unwrap();
        f.write_all(&(36 + data_bytes).to_le_bytes()).unwrap();
        f.write_all(b"WAVE").unwrap();
        f.write_all(b"fmt ").unwrap();
        f.write_all(&16_u32.to_le_bytes()).unwrap();
        f.write_all(&1_u16.to_le_bytes()).unwrap(); // PCM
        f.write_all(&spec.channels.to_le_bytes()).unwrap();
        f.write_all(&spec.sample_rate.to_le_bytes()).unwrap();
        f.write_all(&byte_rate.to_le_bytes()).unwrap();
        f.write_all(&block_align.to_le_bytes()).unwrap();
        f.write_all(&bits_per_sample.to_le_bytes()).unwrap();
        f.write_all(b"data").unwrap();
        f.write_all(&data_bytes.to_le_bytes()).unwrap();
        for &s in samples {
            let q = (s.clamp(-1.0, 1.0) * f32::from(i16::MAX)) as i16;
            f.write_all(&q.to_le_bytes()).unwrap();
        }
    }

    #[test]
    fn start_records_an_error_when_no_audio_device_is_available() {
        let Some((ecapa, vad)) = skip_if_no_onnx() else {
            return;
        };
        let pool = enroll_pool_from_fixture(&ecapa, &vad);
        with_test_enrollment(&pool, || {
            let mut state = AppState::default();
            assert!(state.pool.is_some(), "precondition: pool auto-loaded");
            state.start();
            // Headless container has no cpal device → start() must
            // surface an error and leave no half-constructed session.
            // (If audio is somehow available, the session is fine —
            // stop it to keep the test hermetic.)
            let has_err = state.last_error.is_some();
            let has_session = state.session.is_some();
            assert!(
                has_err ^ has_session,
                "expected exactly one of last_error / session after start(); \
                 err={:?}, has_session={has_session}",
                state.last_error
            );
            if has_session {
                state.stop();
                assert!(state.session.is_none());
            }
        });
    }
}
