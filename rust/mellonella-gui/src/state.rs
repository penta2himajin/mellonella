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
/// users see "Recorded 5.0 s" vs "Loaded from voice.wav" vs
/// "Loaded enrollment.json".
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum EnrollmentOrigin {
    None,
    Wav(PathBuf),
    Mic { secs: u32 },
    Json(PathBuf),
}

/// Distinguishes between the two reasons the GUI can run the mic
/// `Recorder`: rebuilding the speaker pool (`Enroll`) or measuring how
/// the loaded pool would respond to the current mic (`Test`). They
/// share the same audio capture path because they're mutually
/// exclusive — you can't be enrolling and verifying at the same time.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum RecordingMode {
    #[default]
    Enroll,
    Test,
}

/// Outcome of a "Test voice" recording: the cosine match score against
/// the currently-loaded pool, the gate threshold it would be compared
/// to, and the resulting pass/fail decision. Surfaced inline next to
/// the profile pill so the user can sanity-check their enrollment
/// without having to start a live session.
#[derive(Debug, Clone, Copy)]
pub struct TestResult {
    /// `pool.match_score(test_embedding)` — cosine against the
    /// anchor centroid maxed with cosine against the adapted embedding.
    pub match_score: f32,
    /// Threshold used for the pass/fail decision (the legacy
    /// `theta_pass`; we don't surface an AS-Norm variant in this UI).
    pub theta_pass: f32,
    /// `match_score >= theta_pass`.
    pub would_pass: bool,
    /// Mean f0 of the test recording's voiced frames.
    pub f0_mu: f32,
    /// Anchors the score was compared against (informational).
    pub anchors_checked: usize,
}

/// TSE model variant — mirrors the CLI's `--tse-config` enum so the
/// GUI's dropdown has the same options.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum TseVariant {
    /// 16 kHz PoC. Requires both the audio path and decision path to
    /// be at 16 kHz; will fail to start on the GUI's default 48 kHz
    /// audio path (kept selectable so users running a custom build
    /// at 16 kHz can still pick it).
    Poc16k,
    /// 48 kHz production model — the canonical release at
    /// <https://huggingface.co/penta2himajin/tse-conv-tasnet-48k>.
    /// Matches the GUI's 48 kHz audio path exactly.
    #[default]
    Prod48k,
}

impl TseVariant {
    /// Display label for the dropdown.
    #[must_use]
    pub const fn label(self) -> &'static str {
        match self {
            Self::Poc16k => "poc_16k (16 kHz)",
            Self::Prod48k => "prod_48k (48 kHz)",
        }
    }
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
    /// Latch flipped by `maybe_auto_start` when a session-construction
    /// attempt errors, so the failure doesn't loop on every frame.
    /// `store_pool` clears it on the next successful enrollment / load
    /// so the user only has to fix what was wrong (e.g. enroll, set
    /// `MELLONELLA_DFN3_ONNX`) for the auto-start path to retry —
    /// there's no Pause / Resume button to clear it manually any more.
    pub user_paused: bool,
    /// Path to the TSE streaming ONNX (Stage C). Picked via a file
    /// dialog in the Settings panel; persisted in-memory only for this
    /// session.
    pub tse_onnx_path: Option<PathBuf>,
    /// TSE model variant. `Prod48k` matches the production 48 kHz
    /// model on HuggingFace; `Poc16k` matches the original 16 kHz PoC.
    /// Defaults to `Prod48k` — the 48 kHz audio path the GUI uses
    /// end-to-end matches that variant's expected SR exactly.
    pub tse_variant: TseVariant,
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
    /// Distinguishes whether the active `recorder` is collecting audio
    /// for an enrollment (`Enroll`) or a verification ("Test voice")
    /// (`Test`). `poll_recorder` branches on this when the recorder
    /// completes.
    pub recording_mode: RecordingMode,
    /// Latest "Test voice" outcome, shown inline next to the profile
    /// pill. Cleared on the next test or by starting an enrollment.
    pub test_result: Option<TestResult>,
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
            user_paused: false,
            tse_onnx_path: None,
            tse_variant: TseVariant::default(),
            record_duration_secs: DEFAULT_RECORD_SECS,
            gate_cfg: GateConfig::default(),
            pipeline_cfg: default_live_pipeline_cfg(),
            session: None,
            recorder: None,
            recording_mode: RecordingMode::default(),
            test_result: None,
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
        // A successful enrollment / load is the natural moment to
        // release the auto-start latch that `maybe_auto_start` flips
        // on a previous failed start. Without the Pause/Resume button
        // there's no other surface that clears it, so a stale latch
        // would otherwise wedge the app in "never starts" mode after
        // the user fixes whatever was wrong.
        self.user_paused = false;
        // A fresh pool invalidates any previous Test verdict.
        self.test_result = None;
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
    /// WAV at any sample rate; resampled to 48 kHz internally so the
    /// optional DFN3 pre-pass can run at its native rate), then route
    /// through `run_enrollment`.
    pub fn enroll_from_wav(&mut self, path: &Path) {
        match read_wav_to_48k_mono(path) {
            Ok(audio) => self.run_enrollment(&audio, EnrollmentOrigin::Wav(path.to_path_buf())),
            Err(e) => {
                self.clear_pool();
                self.last_error = Some(format!("read WAV: {e}"));
            }
        }
    }

    /// Kick off a mic recording of `secs` seconds at 48 kHz mono —
    /// matches the live audio path's rate so an optional DFN3
    /// pre-pass during enrollment runs on the same distribution as
    /// the live ECAPA refresh path. Call `poll_recorder` once per
    /// frame to detect completion.
    pub fn start_recording(&mut self, secs: f32) {
        self.start_recording_with_mode(secs, RecordingMode::Enroll);
    }

    /// Like [`Self::start_recording`] but the resulting audio is
    /// scored against the currently-loaded pool instead of replacing
    /// it. Surfaces the cosine score and a "would pass" verdict in
    /// [`Self::test_result`]; no-op when no pool is loaded.
    pub fn start_test_recording(&mut self, secs: f32) {
        if self.pool.is_none() {
            self.last_error = Some("test voice: enroll first".to_string());
            return;
        }
        self.start_recording_with_mode(secs, RecordingMode::Test);
    }

    fn start_recording_with_mode(&mut self, secs: f32, mode: RecordingMode) {
        if self.recorder.is_some() {
            return;
        }
        if mode == RecordingMode::Test {
            // Clear stale result so the UI doesn't show last
            // session's verdict while the new recording is in flight.
            self.test_result = None;
        }
        match Recorder::start(self.selected_input.clone(), OUTPUT_SAMPLE_RATE, secs) {
            Ok(r) => {
                self.recorder = Some(r);
                self.recording_mode = mode;
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

    /// Poll the active recorder for completion. On success either
    /// runs enrolment on the captured buffer (`RecordingMode::Enroll`)
    /// or scores it against the loaded pool (`RecordingMode::Test`).
    /// Returns silently when no recorder is active or it's still
    /// capturing.
    pub fn poll_recorder(&mut self) {
        let Some(recorder) = self.recorder.as_mut() else {
            return;
        };
        let Some(result) = recorder.try_finish() else {
            return;
        };
        // Recorder finished; pull it out and consume the result.
        let target_secs = recorder.target_seconds().round() as u32;
        let mode = self.recording_mode;
        self.recorder = None;
        self.recording_mode = RecordingMode::default();
        match result {
            Ok(audio) => {
                if audio.len() < OUTPUT_SAMPLE_RATE as usize {
                    let needed = match mode {
                        RecordingMode::Enroll => "ECAPA enrolment",
                        RecordingMode::Test => "voice test",
                    };
                    self.last_error =
                        Some(format!("recording too short for {needed} (need ≥ 1 s)"));
                    return;
                }
                match mode {
                    RecordingMode::Enroll => {
                        self.run_enrollment(&audio, EnrollmentOrigin::Mic { secs: target_secs });
                    }
                    RecordingMode::Test => self.run_voice_test(&audio),
                }
            }
            Err(e) => self.last_error = Some(format!("recording failed: {e}")),
        }
    }

    /// Score 48 kHz mono audio against the loaded pool — same code
    /// path as `run_enrollment` up to the ECAPA call, but instead of
    /// replacing the pool we take the resulting centroid as a single
    /// test embedding and compute `pool.match_score`. Mirrors what the
    /// live streaming engine would do on each refresh while the user
    /// speaks into the mic.
    fn run_voice_test(&mut self, audio_48k: &[f32]) {
        let Some(pool) = self.pool.clone() else {
            self.last_error = Some("test voice: no pool loaded".to_string());
            return;
        };
        let audio_16k = match resample_to(audio_48k, OUTPUT_SAMPLE_RATE, DECISION_SAMPLE_RATE) {
            Ok(a) => a,
            Err(e) => {
                self.last_error = Some(format!("resample 48 kHz → 16 kHz for test: {e}"));
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
        // `enroll_from_recording` is overkill — it builds a multi-anchor
        // pool. We only need the centroid embedding to score against the
        // user's existing pool, so the throw-away pool's anchor_centroid
        // is what the live refresh path would have produced for the
        // same audio. Reusing the helper keeps the windowing /
        // ECAPA-feature pipeline byte-identical to enrollment.
        let probe_pool = match enroll_from_recording(
            &audio_16k,
            &mut components,
            EmbeddingPoolConfig::default(),
        ) {
            Ok(p) => p,
            Err(e) => {
                self.last_error = Some(format!("test voice: {e}"));
                return;
            }
        };
        let Some(centroid) = probe_pool.anchor_centroid() else {
            self.last_error = Some("test voice: ECAPA produced no embedding".to_string());
            return;
        };
        let match_score = pool.match_score(centroid);
        let theta_pass = self.gate_cfg.theta_pass;
        let f0_mu = probe_pool.metadata().f0_mu;
        self.test_result = Some(TestResult {
            match_score,
            theta_pass,
            would_pass: match_score >= theta_pass,
            f0_mu,
            anchors_checked: pool.anchors().len(),
        });
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
                self.tse_variant = TseVariant::Prod48k;
            }
        }
        if let Some(onnx) = self.tse_onnx_path.as_ref() {
            if !onnx.exists() {
                self.last_error = Some(format!("TSE ONNX path does not exist: {}", onnx.display()));
                return;
            }
            pipeline_cfg.tse = Some(match self.tse_variant {
                TseVariant::Poc16k => TseStageConfig::new(onnx.clone()),
                TseVariant::Prod48k => TseStageConfig::new_prod_48k(onnx.clone()),
            });
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
                self.tse_variant = TseVariant::Prod48k;
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

/// Decode a 16-bit signed mono WAV at any common rate into f32
/// samples at 48 kHz mono — the rate the enrollment pipeline
/// operates at (DFN3 is native here, and we resample down to 16 kHz
/// before ECAPA inside `run_enrollment`).
fn read_wav_to_48k_mono(path: &Path) -> Result<Vec<f32>, String> {
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
    if spec.sample_rate == OUTPUT_SAMPLE_RATE {
        return Ok(samples);
    }
    resample_to(&samples, spec.sample_rate, OUTPUT_SAMPLE_RATE).map_err(|e| e.to_string())
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
    fn default_state_is_not_user_paused() {
        let s = AppState::default();
        assert!(
            !s.user_paused,
            "fresh state should auto-start once enrollment + models are ready"
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
}
