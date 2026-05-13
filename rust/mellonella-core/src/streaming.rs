//! Streaming / online pipeline API — stateful counterpart of
//! [`crate::pipeline::process_offline`].
//!
//! Intended for live microphone use, GUI integrations, and any
//! caller that produces samples incrementally rather than as one
//! contiguous buffer. Runs alongside the offline
//! [`crate::pipeline::process_offline`] (step 8) — a follow-up
//! step (10) will unify them so the offline path becomes a thin
//! wrapper over this engine.
//!
//! Inherits the **dual-rate split** from `process_offline` (Phase
//! 3.5 step 8): callers push audio at
//! `StreamingConfig::audio_sample_rate` (default 48 kHz, full-band
//! quality), the pipeline resamples internally to 16 kHz for the
//! VAD / ECAPA / F0 decision path, and emits envelope-gated audio
//! at the input rate.
//!
//! # Buffering model
//!
//! * **Input granularity**: any length, any cadence, at
//!   `StreamingConfig::audio_sample_rate`. Callers may push
//!   1-sample or 100 000-sample chunks and the pipeline does the
//!   right thing.
//! * **Internal alignment**: VAD frames are 512 samples (32 ms @
//!   16 kHz, [`crate::vad::CHUNK_SAMPLES_16K`]). The audio is
//!   resampled to 16 kHz on the decision path via a stateful
//!   `rubato::SincFixedOut`; sub-VAD-frame audio-rate residue is
//!   held in an internal ring buffer until enough samples
//!   accumulate to drive one more VAD frame.
//! * **Output granularity**: a multiple of one VAD frame's
//!   audio-rate equivalent per `push_samples` call (e.g. 1536
//!   samples @ 48 kHz, the integer scaling of 512 @ 16 kHz). When
//!   rates don't divide evenly the resampler determines the exact
//!   audio-rate input count per VAD frame via `input_frames_next()`.
//! * **Flush**: [`StreamingPipeline::flush`] zero-pads any residual
//!   audio-rate samples to the resampler's expected input size so
//!   the trailing audio gets one last decision pass.
//!
//! # Algorithm parity with `process_offline`
//!
//! With `StreamingConfig::pipeline.async_refresh == false` and
//! `audio_sample_rate == pipeline.sample_rate` (identity rate, no
//! internal resample), `StreamingPipeline::new → push_samples(audio)
//! → flush` produces per-VAD-frame `gate_per_frame` and
//! `score_per_frame` identical to `process_offline(audio, …)`.
//! Verified by the `streaming_smoke` integration test
//! (`streaming_identity_rate_per_frame_matches_offline`), gated on
//! the same ONNX env vars as the existing parity fixtures.
//!
//! At dual-rate (`audio_sample_rate != pipeline.sample_rate`), the
//! streaming engine uses a stateful `SincFixedOut` resampler and
//! `process_offline` uses a one-shot `resample_to`; outputs will
//! differ slightly because the two resamplers have different
//! startup-delay characteristics. A follow-up step will unify the
//! offline path to use the streaming engine.
//!
//! # Async refresh
//!
//! `StreamingPipeline` supports both sync (`async_refresh = false`)
//! and async (`async_refresh = true`) modes:
//!
//! * **Sync**: Fbank / ECAPA / F0 run inline on the caller's thread
//!   inside `push_samples`. Simple but blocks for ~30–50 ms per
//!   refresh.
//! * **Async**: at construction time, `fbank` + `ecapa` are moved
//!   into a persistent worker thread; the main thread sends speech
//!   windows over a channel and reads back `(embedding, f0_mu)`
//!   results, applying them via `apply_refresh_result` on the
//!   next frame after they arrive. Mirrors the cadence model of
//!   `process_offline_async` (at most one inference outstanding,
//!   one queued window so a burst doesn't drop work).
//!
//! `into_parts` joins the worker (waking it via channel close)
//! and reconstructs the original [`PipelineComponents`] from the
//! moved `fbank` + `ecapa` plus the main-thread `vad` + `cohort`.
//! `process_offline` still uses its own dedicated
//! `process_offline_async` for one-shot async runs; rewiring that
//! to `StreamingPipeline::new + push_samples + flush` is a
//! follow-up step.
//!
//! # Ownership
//!
//! `StreamingPipeline::new` takes ownership of the
//! [`crate::enrollment::EmbeddingPool`] and
//! [`crate::pipeline::PipelineComponents`]. They can be recovered
//! on shutdown via [`StreamingPipeline::into_parts`] — useful for
//! persisting the post-run pool (auto-learn updates) and for
//! tearing the ONNX sessions down deterministically.

#![allow(clippy::too_many_arguments, clippy::too_many_lines)]

use std::collections::VecDeque;
use std::sync::mpsc::{channel, Receiver, Sender, TryRecvError};
use std::thread::JoinHandle;

use rubato::{
    Resampler, SincFixedOut, SincInterpolationParameters, SincInterpolationType, WindowFunction,
};

use crate::embedding::{EcapaTdnn, EmbeddingError};
use crate::enrollment::EmbeddingPool;
use crate::f0::{estimate_f0_track, f0_statistics, DEFAULT_F_MAX, DEFAULT_F_MIN};
use crate::features::{Fbank, N_MELS};
use crate::gating::{
    as_norm_score, cos_sim_max_iter, f0_match, should_admit_auto_learn, EnvelopeState, GateConfig,
    GateState,
};
use crate::pipeline::{
    apply_refresh_result, fbank_ecapa_one, AutoLearnEvent, AutoLearnKind, PipelineComponents,
    PipelineConfig, PipelineError,
};
use crate::vad::{SileroVad, CHUNK_SAMPLES_16K};

/// Pair returned by [`StreamingState::drain_one_frame`]: the
/// audio-rate samples consumed and the matching decision-rate VAD
/// frame. Aliased to satisfy `clippy::type_complexity`.
type DrainedFrame = (Vec<f32>, Vec<f32>);

/// Configuration for [`StreamingPipeline`].
///
/// `pipeline` and `gate` are the same structs the offline pipeline
/// consumes; `audio_sample_rate` and `diagnostics` are streaming-
/// specific.
#[derive(Debug, Clone)]
pub struct StreamingConfig {
    /// Pipeline-side cadence (window / refresh / VAD threshold /
    /// auto-learn switch / async refresh). `pipeline.sample_rate`
    /// is the **decision** rate (16 kHz) — see the module
    /// "Buffering model" doc for the dual-rate split.
    pub pipeline: PipelineConfig,
    /// Gate-side parameters (hangover, attack/release, score
    /// threshold, F0 weight).
    pub gate: GateConfig,
    /// Sample rate of the audio path — the rate the caller pushes
    /// into [`StreamingPipeline::push_samples`] and the rate the
    /// returned envelope-gated audio is at. Default 48 000 Hz
    /// (DFN3's native, full-band rate, per the
    /// `docs/architecture.md` Sampling-rate policy). The pipeline
    /// resamples internally to `pipeline.sample_rate` (16 kHz) for
    /// the decision path.
    pub audio_sample_rate: u32,
    /// When `true`, [`StreamingOutput`] populates
    /// `gate_per_frame` / `score_per_frame` /
    /// `cos_sim_max_per_frame` / `f0_match_per_frame`. Default
    /// `false` to avoid the per-call allocation in the live path —
    /// the GUI can opt in only while a "live status" panel is open.
    pub diagnostics: bool,
}

impl Default for StreamingConfig {
    fn default() -> Self {
        Self {
            pipeline: PipelineConfig::default(),
            gate: GateConfig::default(),
            audio_sample_rate: 48_000,
            diagnostics: false,
        }
    }
}

/// Output of a single [`StreamingPipeline::push_samples`] or
/// [`StreamingPipeline::flush`] call.
///
/// All vectors describe **only what was produced by this call** —
/// they are *not* cumulative across calls. Callers writing to a sink
/// (audio device, WAV file, GUI ring) should consume `audio` every
/// call and forget it; the pipeline does not retain a copy.
///
/// `gate_per_frame` / `score_per_frame` / `cos_sim_max_per_frame` /
/// `f0_match_per_frame` are populated only when
/// [`StreamingConfig::diagnostics`] is `true`.
#[derive(Debug, Default, Clone)]
pub struct StreamingOutput {
    /// Envelope-gated audio at the pipeline's configured
    /// [`StreamingConfig::audio_sample_rate`] (default 48 kHz),
    /// mono, for the just-pushed chunk. Length is a multiple of one
    /// VAD frame's audio-rate equivalent (e.g. 1536 samples @
    /// 48 kHz, the integer scaling of
    /// [`crate::vad::CHUNK_SAMPLES_16K`] = 512 @ 16 kHz).
    pub audio: Vec<f32>,
    /// Audio-rate `(start_sample, is_on)` decisions consumed by the
    /// envelope. Indices are *cumulative since
    /// [`StreamingPipeline::new`] / `reset`*, so callers stitching
    /// successive outputs share a coherent timeline.
    pub gate_decisions: Vec<(usize, bool)>,
    /// Auto-learn admission / rejection / reset events that
    /// occurred during this call, in chronological order. `frame_idx`
    /// is cumulative since `new` / `reset`.
    pub events: Vec<AutoLearnEvent>,
    /// Per-VAD-frame gate state. Empty unless
    /// [`StreamingConfig::diagnostics`] is true.
    pub gate_per_frame: Vec<bool>,
    /// Per-VAD-frame integrated score. Empty unless diagnostics on.
    pub score_per_frame: Vec<f32>,
    /// Per-VAD-frame `cos_sim_max`. Empty unless diagnostics on.
    pub cos_sim_max_per_frame: Vec<f32>,
    /// Per-VAD-frame F0 match. Empty unless diagnostics on.
    pub f0_match_per_frame: Vec<f32>,
}

/// Private engine state — the carry-over data the streaming loop
/// needs between `push_samples` / `flush` calls. Doesn't hold
/// `EmbeddingPool` or `PipelineComponents`; both
/// [`StreamingPipeline`] (owns) and the offline wrapper (borrows)
/// pass those in by mutable reference.
pub(crate) struct StreamingState {
    /// Audio-rate ring; sub-decision-frame residue waiting for
    /// enough samples to drive the next resampler call.
    audio_ring: VecDeque<f32>,
    /// Stateful sinc resampler from `audio_sample_rate` →
    /// `decision_sample_rate`. `None` when the two rates match
    /// (identity path — no resample).
    resampler: Option<SincFixedOut<f32>>,
    /// Scratch input/output buffers reused across resampler calls
    /// to avoid per-call allocation. `None` mirrors `resampler`.
    resampler_in: Option<Vec<f32>>,
    resampler_out: Option<Vec<Vec<f32>>>,
    /// SV-rate (16 kHz) speech accumulator — frames with
    /// `speech_prob > vad_threshold` flow in here.
    speech_buffer: VecDeque<f32>,
    /// Reusable contiguous staging slice for Fbank / F0 input.
    sv_window_scratch: Vec<f32>,
    /// SV refresh cadence + VAD-edge early-refresh state — mirrors
    /// `process_offline`'s locals.
    samples_since_update: usize,
    silence_seen_since_refresh: bool,
    new_speech_samples_after_silence: usize,
    prev_speech: bool,
    consecutive_speech_ms: f32,
    last_score: f32,
    last_cs: f32,
    last_fm: f32,
    gate_state: GateState,
    envelope_state: EnvelopeState,
    /// Last emitted `(start_sample, is_on)` decision (at audio
    /// rate). Used to detect runs and emit the new boundary only on
    /// transitions.
    current_decision: Option<bool>,
    /// Monotonic VAD-frame counter since construction / reset — used
    /// as `AutoLearnEvent.frame_idx`.
    frame_idx: usize,
    /// Monotonic audio-rate sample counter — used as
    /// `gate_decisions[i].0` so successive call outputs share a
    /// timeline.
    audio_samples_emitted: usize,
    /// Audio-rate samples per decision-rate VAD frame at identity
    /// rate (= `CHUNK_SAMPLES_16K`). Cached only for the identity
    /// branch; dual-rate uses `resampler.input_frames_next()`.
    identity_input_per_frame: usize,
    /// Persistent ECAPA / Fbank / F0 worker thread, present only
    /// when `config.pipeline.async_refresh = true`. The worker owns
    /// the `Fbank` + `EcapaTdnn` (moved at construction) and
    /// returns them via `shutdown` so [`StreamingPipeline::into_parts`]
    /// can reconstruct the full [`PipelineComponents`].
    pub(crate) async_worker: Option<AsyncWorker>,
}

/// Persistent worker thread for `async_refresh = true` streaming.
///
/// The worker owns the `Fbank` + `EcapaTdnn` for the lifetime of
/// the streaming pipeline. Refresh windows arrive over `work_tx`;
/// each window is run through Fbank → ECAPA + F0 and the resulting
/// `(embedding, f0_mu)` flows back over `result_rx`.
///
/// Bookkeeping (`outstanding`, `pending`, `refresh_frame_indices`)
/// mirrors `process_offline_async`: at most one outstanding
/// inference at a time, one queued window so a burst of two
/// refreshes within one ECAPA wall time doesn't drop work, and
/// frame indices in FIFO order so [`AutoLearnEvent.frame_idx`]
/// reflects the *trigger* frame rather than the result-arrival
/// frame.
pub(crate) struct AsyncWorker {
    work_tx: Sender<Vec<f32>>,
    result_rx: Receiver<Result<(Vec<f32>, f32), EmbeddingError>>,
    join: Option<JoinHandle<(Fbank, EcapaTdnn)>>,
    outstanding: u32,
    pending: Option<Vec<f32>>,
    refresh_frame_indices: VecDeque<usize>,
}

impl AsyncWorker {
    fn spawn(
        mut fbank: Fbank,
        mut ecapa: EcapaTdnn,
        decision_sr: u32,
    ) -> Result<Self, PipelineError> {
        let (work_tx, work_rx) = channel::<Vec<f32>>();
        let (result_tx, result_rx) = channel::<Result<(Vec<f32>, f32), EmbeddingError>>();
        let join = std::thread::Builder::new()
            .name("mellonella-streaming-async-worker".into())
            .spawn(move || {
                while let Ok(window) = work_rx.recv() {
                    let msg = match fbank_ecapa_one(&window, &mut fbank, &mut ecapa) {
                        Ok(embedding) => {
                            let f0_track = estimate_f0_track(
                                &window,
                                decision_sr,
                                2048,
                                512,
                                DEFAULT_F_MIN,
                                DEFAULT_F_MAX,
                            );
                            let (f0_mu, _) = f0_statistics(&f0_track);
                            Ok((embedding, f0_mu))
                        }
                        Err(e) => Err(e),
                    };
                    if result_tx.send(msg).is_err() {
                        break;
                    }
                }
                (fbank, ecapa)
            })
            .map_err(|e| {
                PipelineError::Embedding(EmbeddingError::Ort(format!("spawn async worker: {e}")))
            })?;
        Ok(Self {
            work_tx,
            result_rx,
            join: Some(join),
            outstanding: 0,
            pending: None,
            refresh_frame_indices: VecDeque::new(),
        })
    }

    /// Submit a refresh window. If the worker is idle (`outstanding
    /// == 0`) the window is sent immediately; otherwise it queues
    /// as the single pending window (overwriting any previous
    /// pending — the cadence guarantees that won't happen in
    /// normal use, but the fallback keeps the state machine well-
    /// defined under burst load).
    fn submit(&mut self, window: Vec<f32>, frame_idx: usize) {
        self.refresh_frame_indices.push_back(frame_idx);
        if self.outstanding == 0 {
            if self.work_tx.send(window).is_ok() {
                self.outstanding = 1;
            }
        } else {
            self.pending = Some(window);
        }
    }

    /// Non-blocking poll for the next completed inference.
    fn try_recv_result(&mut self) -> Result<Option<(usize, Vec<f32>, f32)>, EmbeddingError> {
        if self.outstanding == 0 {
            return Ok(None);
        }
        match self.result_rx.try_recv() {
            Ok(Ok((emb, f0_mu))) => {
                let frame_idx = self.refresh_frame_indices.pop_front().unwrap_or(0);
                self.outstanding -= 1;
                if let Some(next) = self.pending.take() {
                    if self.work_tx.send(next).is_ok() {
                        self.outstanding = 1;
                    }
                }
                Ok(Some((frame_idx, emb, f0_mu)))
            }
            Ok(Err(e)) => Err(e),
            Err(TryRecvError::Empty) => Ok(None),
            Err(TryRecvError::Disconnected) => {
                Err(EmbeddingError::Ort("async worker disconnected".into()))
            }
        }
    }

    /// Blocking drain of outstanding work — used by `flush_async`.
    fn drain_blocking(&mut self) -> Result<Vec<(usize, Vec<f32>, f32)>, EmbeddingError> {
        let mut results = Vec::new();
        while self.outstanding > 0 {
            let msg = self
                .result_rx
                .recv()
                .map_err(|_| EmbeddingError::Ort("async worker disconnected".into()))?;
            let (emb, f0_mu) = msg?;
            let frame_idx = self.refresh_frame_indices.pop_front().unwrap_or(0);
            self.outstanding -= 1;
            results.push((frame_idx, emb, f0_mu));
            if let Some(next) = self.pending.take() {
                if self.work_tx.send(next).is_ok() {
                    self.outstanding = 1;
                }
            }
        }
        Ok(results)
    }

    /// Tear the worker down and recover the owned components.
    /// Called from [`StreamingPipeline::into_parts`].
    fn shutdown(mut self) -> Result<(Fbank, EcapaTdnn), PipelineError> {
        drop(self.work_tx); // Triggers the worker's `recv()` to return Err and exit.
        let join = self
            .join
            .take()
            .expect("worker join handle present at shutdown");
        join.join().map_err(|_| {
            PipelineError::Embedding(EmbeddingError::Ort("async worker panicked".into()))
        })
    }
}

/// Owning storage for `PipelineComponents` inside a
/// [`StreamingPipeline`]. Sync stores the full struct; async splits
/// out `fbank` + `ecapa` into the persistent worker thread, keeping
/// only `vad` + `cohort` on the main thread.
///
/// The Sync variant is heap-sized (PipelineComponents holds ONNX
/// session handles); boxing keeps the enum compact even though we
/// only ever hold one variant per pipeline lifetime.
pub(crate) enum ComponentsStorage {
    Sync(Box<PipelineComponents>),
    Async {
        vad: SileroVad,
        cohort: Vec<Vec<f32>>,
    },
}

impl StreamingState {
    /// Sync constructor — `async_refresh` must be `false`. For
    /// async streaming, call [`Self::new_async`] (and route through
    /// [`StreamingPipeline`] which owns the worker lifecycle).
    pub(crate) fn new(config: &StreamingConfig) -> Result<Self, PipelineError> {
        if config.pipeline.async_refresh {
            return Err(PipelineError::Embedding(EmbeddingError::Ort(
                "StreamingState::new requires async_refresh = false; \
                 use StreamingState::new_async or StreamingPipeline::new for async"
                    .into(),
            )));
        }
        let decision_sr = config.pipeline.sample_rate;
        let audio_sr = config.audio_sample_rate;
        let (resampler, resampler_in, resampler_out) = if audio_sr == decision_sr {
            (None, None, None)
        } else {
            let ratio = f64::from(decision_sr) / f64::from(audio_sr);
            let params = SincInterpolationParameters {
                sinc_len: 256,
                f_cutoff: 0.95,
                interpolation: SincInterpolationType::Linear,
                oversampling_factor: 256,
                window: WindowFunction::BlackmanHarris2,
            };
            let r = SincFixedOut::<f32>::new(ratio, 1.1, params, CHUNK_SAMPLES_16K, 1).map_err(
                |e| {
                    PipelineError::Embedding(EmbeddingError::Ort(format!(
                        "streaming resampler init failed: {e}"
                    )))
                },
            )?;
            let in_cap = r.input_frames_max();
            (
                Some(r),
                Some(Vec::with_capacity(in_cap)),
                Some(vec![vec![0.0_f32; CHUNK_SAMPLES_16K]]),
            )
        };
        Ok(Self {
            audio_ring: VecDeque::with_capacity(audio_sr as usize), // ~1 s
            resampler,
            resampler_in,
            resampler_out,
            speech_buffer: VecDeque::with_capacity(config.pipeline.sv_window_samples),
            sv_window_scratch: Vec::with_capacity(config.pipeline.sv_window_samples),
            samples_since_update: 0,
            silence_seen_since_refresh: false,
            new_speech_samples_after_silence: 0,
            prev_speech: false,
            consecutive_speech_ms: 0.0,
            last_score: 0.0,
            last_cs: 0.0,
            last_fm: 1.0,
            gate_state: GateState::new(config.gate),
            envelope_state: EnvelopeState::new(config.gate, audio_sr),
            current_decision: None,
            frame_idx: 0,
            audio_samples_emitted: 0,
            identity_input_per_frame: CHUNK_SAMPLES_16K,
            async_worker: None,
        })
    }

    /// Async constructor — spawns the persistent ECAPA / Fbank / F0
    /// worker thread, moves `fbank` + `ecapa` into it. The
    /// resulting state's `step_one_frame_async` dispatches refresh
    /// windows to the worker via channels instead of calling
    /// fbank/ecapa inline.
    pub(crate) fn new_async(
        config: &StreamingConfig,
        fbank: Fbank,
        ecapa: EcapaTdnn,
    ) -> Result<Self, PipelineError> {
        if !config.pipeline.async_refresh {
            return Err(PipelineError::Embedding(EmbeddingError::Ort(
                "StreamingState::new_async requires async_refresh = true".into(),
            )));
        }
        // Build a sync-shaped state first by pretending async_refresh
        // is off, then attach the worker. This sidesteps duplicating
        // ~60 lines of resampler / buffer / gate / envelope init.
        let sync_cfg = StreamingConfig {
            pipeline: PipelineConfig {
                async_refresh: false,
                ..config.pipeline
            },
            ..config.clone()
        };
        let mut state = Self::new(&sync_cfg)?;
        let worker = AsyncWorker::spawn(fbank, ecapa, config.pipeline.sample_rate)?;
        state.async_worker = Some(worker);
        Ok(state)
    }

    /// Reset the carry-over state without touching pool or
    /// components. Used by [`StreamingPipeline::reset`].
    pub(crate) fn reset(&mut self, config: &StreamingConfig) {
        self.audio_ring.clear();
        if let Some(r) = self.resampler.as_mut() {
            r.reset();
        }
        self.speech_buffer.clear();
        self.sv_window_scratch.clear();
        self.samples_since_update = 0;
        self.silence_seen_since_refresh = false;
        self.new_speech_samples_after_silence = 0;
        self.prev_speech = false;
        self.consecutive_speech_ms = 0.0;
        self.last_score = 0.0;
        self.last_cs = 0.0;
        self.last_fm = 1.0;
        self.gate_state = GateState::new(config.gate);
        self.envelope_state = EnvelopeState::new(config.gate, config.audio_sample_rate);
        self.current_decision = None;
        self.frame_idx = 0;
        self.audio_samples_emitted = 0;
    }

    /// Audio-rate samples needed to drive one VAD frame at the
    /// decision rate. Identity case: 512. Dual-rate: variable, asks
    /// the resampler.
    fn input_per_frame(&self) -> usize {
        match self.resampler.as_ref() {
            Some(r) => r.input_frames_next(),
            None => self.identity_input_per_frame,
        }
    }

    /// Drive one VAD frame's worth of audio-rate samples through
    /// the decision path. Pops samples from `audio_ring`, produces
    /// one decision-rate VAD frame, and returns the audio-rate
    /// slice that was consumed (for envelope application by the
    /// caller). `None` means not enough samples in the ring yet.
    fn drain_one_frame(&mut self) -> Result<Option<DrainedFrame>, PipelineError> {
        let n_input = self.input_per_frame();
        if self.audio_ring.len() < n_input {
            return Ok(None);
        }
        let mut audio_chunk = Vec::with_capacity(n_input);
        for _ in 0..n_input {
            audio_chunk.push(self.audio_ring.pop_front().unwrap_or(0.0));
        }
        let decision_chunk = match self.resampler.as_mut() {
            None => audio_chunk.clone(),
            Some(r) => {
                let buf_in = self
                    .resampler_in
                    .as_mut()
                    .expect("resampler_in present alongside resampler");
                buf_in.clear();
                buf_in.extend_from_slice(&audio_chunk);
                let buf_out = self
                    .resampler_out
                    .as_mut()
                    .expect("resampler_out present alongside resampler");
                let in_slices = [buf_in.as_slice()];
                r.process_into_buffer(&in_slices, buf_out, None)
                    .map_err(|e| {
                        PipelineError::Embedding(EmbeddingError::Ort(format!(
                            "streaming resampler step failed: {e}"
                        )))
                    })?;
                buf_out[0].clone()
            }
        };
        debug_assert_eq!(decision_chunk.len(), CHUNK_SAMPLES_16K);
        Ok(Some((audio_chunk, decision_chunk)))
    }

    /// Single VAD-frame iteration: runs VAD, accumulates speech,
    /// maybe runs ECAPA, advances gate + envelope, appends the
    /// envelope-gated audio-rate samples to `out`.
    fn step_one_frame(
        &mut self,
        audio_chunk: &[f32],
        decision_frame: &[f32],
        pool: &mut EmbeddingPool,
        components: &mut PipelineComponents,
        config: &StreamingConfig,
        out: &mut StreamingOutput,
    ) -> Result<(), PipelineError> {
        let vad_frame = CHUNK_SAMPLES_16K;
        let pipeline_cfg = &config.pipeline;
        let gate_cfg = &config.gate;
        let dt_ms = pipeline_cfg.vad_frame_ms();

        let speech_prob = components.vad.score(decision_frame)?;
        let now_speech = speech_prob > pipeline_cfg.vad_threshold;
        if now_speech {
            for &sample in decision_frame {
                if self.speech_buffer.len() == pipeline_cfg.sv_window_samples {
                    self.speech_buffer.pop_front();
                }
                self.speech_buffer.push_back(sample);
            }
            self.consecutive_speech_ms += dt_ms;
            if self.silence_seen_since_refresh {
                self.new_speech_samples_after_silence += vad_frame;
            }
        } else {
            self.consecutive_speech_ms = 0.0;
        }
        if self.prev_speech && !now_speech {
            self.silence_seen_since_refresh = true;
            self.new_speech_samples_after_silence = 0;
        }
        self.prev_speech = now_speech;
        self.samples_since_update += vad_frame;

        let due_normal = self.samples_since_update >= pipeline_cfg.sv_update_samples;
        let due_early = self.silence_seen_since_refresh
            && now_speech
            && self.new_speech_samples_after_silence
                >= pipeline_cfg.sv_min_new_samples_after_silence;
        if (due_normal || due_early) && self.speech_buffer.len() >= pipeline_cfg.sv_window_samples {
            self.samples_since_update = 0;
            self.silence_seen_since_refresh = false;
            self.new_speech_samples_after_silence = 0;
            self.sv_window_scratch.clear();
            self.sv_window_scratch
                .extend(self.speech_buffer.iter().copied());
            let window: &[f32] = &self.sv_window_scratch;

            let feats = components.fbank.compute(window);
            let n_frames = feats.len() / N_MELS;
            let embedding = components.ecapa.embed_features(&feats, n_frames, N_MELS)?;

            let f0_track = estimate_f0_track(
                window,
                pipeline_cfg.sample_rate,
                2048,
                512,
                DEFAULT_F_MIN,
                DEFAULT_F_MAX,
            );
            let (f0_mu, _) = f0_statistics(&f0_track);

            let cs = cos_sim_max_iter(
                &embedding,
                pool.anchors()
                    .iter()
                    .chain(pool.auto_learn().iter())
                    .map(Vec::as_slice),
            );
            let fm = f0_match(f0_mu, pool.metadata().f0_mu, pool.metadata().f0_sigma);
            self.last_cs = cs;
            self.last_fm = fm;
            self.last_score = if gate_cfg.use_as_norm && !components.cohort.is_empty() {
                as_norm_score(&embedding, cs, &components.cohort, 20)
            } else {
                cs
            };

            if pipeline_cfg.enable_auto_learn
                && should_admit_auto_learn(
                    self.last_score,
                    fm,
                    self.consecutive_speech_ms,
                    gate_cfg,
                )
            {
                let admitted = pool.add_auto_learn(embedding);
                let kind = if admitted {
                    AutoLearnKind::Admit
                } else {
                    AutoLearnKind::RejectAnchorDistance
                };
                out.events.push(AutoLearnEvent {
                    frame_idx: self.frame_idx,
                    kind,
                    score: self.last_score,
                    f0_match: fm,
                });
                if admitted && pool.maybe_reset() {
                    out.events.push(AutoLearnEvent {
                        frame_idx: self.frame_idx,
                        kind: AutoLearnKind::Reset,
                        score: self.last_score,
                        f0_match: fm,
                    });
                }
            }
        }

        let is_on = self.gate_state.update(self.last_score, dt_ms);
        if config.diagnostics {
            out.gate_per_frame.push(is_on);
            out.score_per_frame.push(self.last_score);
            out.cos_sim_max_per_frame.push(self.last_cs);
            out.f0_match_per_frame.push(self.last_fm);
        }

        // Decision boundary: record (audio_rate_index, is_on) on
        // transition. The index is cumulative across calls.
        let block_start_audio = self.audio_samples_emitted;
        if self.current_decision != Some(is_on) {
            out.gate_decisions.push((block_start_audio, is_on));
            self.current_decision = Some(is_on);
        }

        // Apply envelope at audio rate for this frame's audio-rate
        // span. `EnvelopeState::advance` returns one gain per
        // sample.
        let gain = self.envelope_state.advance(is_on, audio_chunk.len());
        for (k, &g) in gain.iter().enumerate() {
            out.audio.push(audio_chunk[k] * g);
        }
        self.audio_samples_emitted += audio_chunk.len();
        self.frame_idx += 1;
        Ok(())
    }

    /// Async-mode counterpart of `step_one_frame`. Identical except
    /// that an ECAPA refresh **submits** the speech window to the
    /// persistent worker instead of running `Fbank` / `EcapaTdnn`
    /// inline, and any ready results are applied via
    /// [`apply_refresh_result`] before the gate update.
    ///
    /// This matches `process_offline_async`'s "at most one
    /// inference in flight + one queued window" cadence so live
    /// behaviour is consistent with the offline async path.
    fn step_one_frame_async(
        &mut self,
        audio_chunk: &[f32],
        decision_frame: &[f32],
        pool: &mut EmbeddingPool,
        vad: &mut SileroVad,
        cohort: &[Vec<f32>],
        config: &StreamingConfig,
        out: &mut StreamingOutput,
    ) -> Result<(), PipelineError> {
        let vad_frame = CHUNK_SAMPLES_16K;
        let pipeline_cfg = &config.pipeline;
        let gate_cfg = &config.gate;
        let dt_ms = pipeline_cfg.vad_frame_ms();

        let speech_prob = vad.score(decision_frame)?;
        let now_speech = speech_prob > pipeline_cfg.vad_threshold;
        if now_speech {
            for &sample in decision_frame {
                if self.speech_buffer.len() == pipeline_cfg.sv_window_samples {
                    self.speech_buffer.pop_front();
                }
                self.speech_buffer.push_back(sample);
            }
            self.consecutive_speech_ms += dt_ms;
            if self.silence_seen_since_refresh {
                self.new_speech_samples_after_silence += vad_frame;
            }
        } else {
            self.consecutive_speech_ms = 0.0;
        }
        if self.prev_speech && !now_speech {
            self.silence_seen_since_refresh = true;
            self.new_speech_samples_after_silence = 0;
        }
        self.prev_speech = now_speech;
        self.samples_since_update += vad_frame;

        // Submit a refresh window to the worker when due — same
        // cadence rule as the sync path.
        let due_normal = self.samples_since_update >= pipeline_cfg.sv_update_samples;
        let due_early = self.silence_seen_since_refresh
            && now_speech
            && self.new_speech_samples_after_silence
                >= pipeline_cfg.sv_min_new_samples_after_silence;
        if (due_normal || due_early) && self.speech_buffer.len() >= pipeline_cfg.sv_window_samples {
            self.samples_since_update = 0;
            self.silence_seen_since_refresh = false;
            self.new_speech_samples_after_silence = 0;
            let window: Vec<f32> = self.speech_buffer.iter().copied().collect();
            if let Some(worker) = self.async_worker.as_mut() {
                worker.submit(window, self.frame_idx);
            }
        }

        // Drain at most one ready result per frame so the gate
        // score updates as soon as the worker is done.
        if let Some(worker) = self.async_worker.as_mut() {
            if let Some((trigger_frame, embedding, f0_mu)) = worker.try_recv_result()? {
                apply_refresh_result(
                    embedding,
                    f0_mu,
                    trigger_frame,
                    self.consecutive_speech_ms,
                    pool,
                    cohort,
                    gate_cfg,
                    pipeline_cfg.enable_auto_learn,
                    &mut self.last_score,
                    &mut self.last_cs,
                    &mut self.last_fm,
                    &mut out.events,
                );
            }
        }

        let is_on = self.gate_state.update(self.last_score, dt_ms);
        if config.diagnostics {
            out.gate_per_frame.push(is_on);
            out.score_per_frame.push(self.last_score);
            out.cos_sim_max_per_frame.push(self.last_cs);
            out.f0_match_per_frame.push(self.last_fm);
        }

        let block_start_audio = self.audio_samples_emitted;
        if self.current_decision != Some(is_on) {
            out.gate_decisions.push((block_start_audio, is_on));
            self.current_decision = Some(is_on);
        }

        let gain = self.envelope_state.advance(is_on, audio_chunk.len());
        for (k, &g) in gain.iter().enumerate() {
            out.audio.push(audio_chunk[k] * g);
        }
        self.audio_samples_emitted += audio_chunk.len();
        self.frame_idx += 1;
        Ok(())
    }

    /// Drain as many full VAD frames as the audio_ring currently
    /// supports, returning the accumulated [`StreamingOutput`].
    pub(crate) fn push_block(
        &mut self,
        samples: &[f32],
        pool: &mut EmbeddingPool,
        components: &mut PipelineComponents,
        config: &StreamingConfig,
    ) -> Result<StreamingOutput, PipelineError> {
        self.audio_ring.extend(samples.iter().copied());
        let mut out = StreamingOutput::default();
        while let Some((audio_chunk, decision_chunk)) = self.drain_one_frame()? {
            self.step_one_frame(
                &audio_chunk,
                &decision_chunk,
                pool,
                components,
                config,
                &mut out,
            )?;
        }
        Ok(out)
    }

    /// Async counterpart of [`Self::push_block`]. Takes the
    /// async-friendly subset of components (vad + cohort) since
    /// fbank + ecapa live in the worker thread.
    pub(crate) fn push_block_async(
        &mut self,
        samples: &[f32],
        pool: &mut EmbeddingPool,
        vad: &mut SileroVad,
        cohort: &[Vec<f32>],
        config: &StreamingConfig,
    ) -> Result<StreamingOutput, PipelineError> {
        self.audio_ring.extend(samples.iter().copied());
        let mut out = StreamingOutput::default();
        while let Some((audio_chunk, decision_chunk)) = self.drain_one_frame()? {
            self.step_one_frame_async(
                &audio_chunk,
                &decision_chunk,
                pool,
                vad,
                cohort,
                config,
                &mut out,
            )?;
        }
        Ok(out)
    }

    /// Zero-pad any residual audio-rate samples to the resampler's
    /// next-expected input size so the trailing audio gets one last
    /// decision pass. Idempotent on a fully-drained state.
    pub(crate) fn flush(
        &mut self,
        pool: &mut EmbeddingPool,
        components: &mut PipelineComponents,
        config: &StreamingConfig,
    ) -> Result<StreamingOutput, PipelineError> {
        if self.audio_ring.is_empty() {
            return Ok(StreamingOutput::default());
        }
        let n_input = self.input_per_frame();
        if self.audio_ring.len() < n_input {
            // Pad with silence so one more frame can flow through.
            let pad = n_input - self.audio_ring.len();
            self.audio_ring.extend(std::iter::repeat(0.0_f32).take(pad));
        }
        let mut out = StreamingOutput::default();
        // Drain remaining whole frames (may be > 1 if the caller
        // pushed a chunk that's just shy of multiple frames).
        // `drain_one_frame` itself returns `None` when the ring no
        // longer has a full frame's worth of samples.
        while let Some((audio_chunk, decision_chunk)) = self.drain_one_frame()? {
            self.step_one_frame(
                &audio_chunk,
                &decision_chunk,
                pool,
                components,
                config,
                &mut out,
            )?;
        }
        Ok(out)
    }

    /// Async counterpart of [`Self::flush`]. Drains any residue +
    /// also blocks on outstanding worker inferences so the trailing
    /// auto-learn events / score updates are captured before the
    /// caller tears the pipeline down.
    pub(crate) fn flush_async(
        &mut self,
        pool: &mut EmbeddingPool,
        vad: &mut SileroVad,
        cohort: &[Vec<f32>],
        config: &StreamingConfig,
    ) -> Result<StreamingOutput, PipelineError> {
        let mut out = StreamingOutput::default();
        // Same audio-frame zero-pad path as the sync flush.
        if !self.audio_ring.is_empty() {
            let n_input = self.input_per_frame();
            if self.audio_ring.len() < n_input {
                let pad = n_input - self.audio_ring.len();
                self.audio_ring.extend(std::iter::repeat(0.0_f32).take(pad));
            }
            while let Some((audio_chunk, decision_chunk)) = self.drain_one_frame()? {
                self.step_one_frame_async(
                    &audio_chunk,
                    &decision_chunk,
                    pool,
                    vad,
                    cohort,
                    config,
                    &mut out,
                )?;
            }
        }
        // Drain any in-flight ECAPA work so its scores + auto-learn
        // events make it into the output. Per-frame audio was
        // already emitted using whatever `last_score` was at that
        // frame; these results only affect `pool` /
        // `auto_learn_events`.
        if let Some(worker) = self.async_worker.as_mut() {
            let results = worker.drain_blocking()?;
            for (trigger_frame, embedding, f0_mu) in results {
                apply_refresh_result(
                    embedding,
                    f0_mu,
                    trigger_frame,
                    self.consecutive_speech_ms,
                    pool,
                    cohort,
                    &config.gate,
                    config.pipeline.enable_auto_learn,
                    &mut self.last_score,
                    &mut self.last_cs,
                    &mut self.last_fm,
                    &mut out.events,
                );
            }
        }
        Ok(out)
    }

    /// Take ownership of the worker (only meaningful in async
    /// mode). Returns the recovered `Fbank` + `EcapaTdnn` after
    /// joining the thread. Used by
    /// [`StreamingPipeline::into_parts`].
    pub(crate) fn shutdown_worker(&mut self) -> Result<Option<(Fbank, EcapaTdnn)>, PipelineError> {
        let Some(worker) = self.async_worker.take() else {
            return Ok(None);
        };
        worker.shutdown().map(Some)
    }
}

/// Stateful, single-target speaker gating pipeline driven by
/// incremental sample pushes.
///
/// Supports both sync mode (`async_refresh = false`, inline
/// Fbank / ECAPA on the main thread) and async mode
/// (`async_refresh = true`, persistent worker thread running
/// Fbank / ECAPA / F0 in parallel with VAD + gating on the
/// caller's thread). See the module-level docs for the buffering
/// model, parity contract, and ownership rules.
pub struct StreamingPipeline {
    state: StreamingState,
    config: StreamingConfig,
    pool: EmbeddingPool,
    components: ComponentsStorage,
}

impl StreamingPipeline {
    /// Build a streaming pipeline. The pool and components are
    /// moved in; recover them via [`Self::into_parts`].
    ///
    /// When `config.pipeline.async_refresh = true`, the components'
    /// `fbank` + `ecapa` move into a persistent worker thread for
    /// the lifetime of the pipeline; `into_parts` re-joins the
    /// worker to reconstruct the original [`PipelineComponents`]
    /// struct.
    ///
    /// # Errors
    ///
    /// Returns `PipelineError` if the resampler can't be built
    /// (only when `audio_sample_rate != pipeline.sample_rate`) or
    /// if spawning the async worker fails.
    pub fn new(
        pool: EmbeddingPool,
        config: StreamingConfig,
        components: PipelineComponents,
    ) -> Result<Self, PipelineError> {
        if config.pipeline.async_refresh {
            let PipelineComponents {
                vad,
                fbank,
                ecapa,
                cohort,
            } = components;
            let state = StreamingState::new_async(&config, fbank, ecapa)?;
            Ok(Self {
                state,
                config,
                pool,
                components: ComponentsStorage::Async { vad, cohort },
            })
        } else {
            let state = StreamingState::new(&config)?;
            Ok(Self {
                state,
                config,
                pool,
                components: ComponentsStorage::Sync(Box::new(components)),
            })
        }
    }

    /// Push an arbitrary-length chunk of `audio_sample_rate` Hz f32
    /// mono samples (range `[-1.0, 1.0]`).
    ///
    /// Returns the gated output, at the same sample rate,
    /// corresponding to all VAD frames that could be completed with
    /// the new samples. Sub-frame residue at the audio rate is
    /// buffered and produces no output until the next call (or a
    /// [`Self::flush`]).
    ///
    /// # Errors
    ///
    /// Returns `PipelineError` if an underlying ONNX inference
    /// fails, the resampler step fails, or (in async mode) the
    /// worker thread disconnects.
    pub fn push_samples(&mut self, samples: &[f32]) -> Result<StreamingOutput, PipelineError> {
        match &mut self.components {
            ComponentsStorage::Sync(c) => {
                self.state
                    .push_block(samples, &mut self.pool, c.as_mut(), &self.config)
            }
            ComponentsStorage::Async { vad, cohort } => {
                self.state
                    .push_block_async(samples, &mut self.pool, vad, cohort, &self.config)
            }
        }
    }

    /// Flush any residual sub-VAD-frame samples by zero-padding to
    /// the resampler's next-expected input size so the trailing
    /// audio gets one last decision pass. In async mode, also
    /// blocks on any in-flight ECAPA inferences so trailing
    /// scores / auto-learn events make it into the output.
    ///
    /// Call this once at end-of-stream (e.g. when the audio device
    /// closes) to avoid losing the tail.
    ///
    /// # Errors
    ///
    /// Same as [`Self::push_samples`].
    pub fn flush(&mut self) -> Result<StreamingOutput, PipelineError> {
        match &mut self.components {
            ComponentsStorage::Sync(c) => {
                self.state.flush(&mut self.pool, c.as_mut(), &self.config)
            }
            ComponentsStorage::Async { vad, cohort } => {
                self.state
                    .flush_async(&mut self.pool, vad, cohort, &self.config)
            }
        }
    }

    /// Read-only access to the owned pool.
    #[must_use]
    pub fn pool(&self) -> &EmbeddingPool {
        &self.pool
    }

    /// Mutable access to the owned pool. Safe between
    /// `push_samples` / `flush` calls.
    pub fn pool_mut(&mut self) -> &mut EmbeddingPool {
        &mut self.pool
    }

    /// Reset stateful pieces (rings, gate, envelope, frame index)
    /// **without** rebuilding ONNX sessions or tearing down the
    /// async worker thread (if any). Pool is preserved.
    pub fn reset(&mut self) {
        self.state.reset(&self.config);
    }

    /// Tear the pipeline down, returning the owned pool and
    /// components. In async mode, this joins the worker thread
    /// (waking it via a channel close) and recombines `fbank` +
    /// `ecapa` with the main-thread `vad` + `cohort` into the
    /// original [`PipelineComponents`].
    ///
    /// # Errors
    ///
    /// Returns `PipelineError` if the async worker panicked.
    ///
    /// # Panics
    ///
    /// Panics if the storage is in `Async` mode but no worker is
    /// present — this is an unreachable invariant violation
    /// (`Async` is only ever constructed with a freshly-spawned
    /// worker).
    pub fn into_parts(mut self) -> Result<(EmbeddingPool, PipelineComponents), PipelineError> {
        let components = match self.components {
            ComponentsStorage::Sync(c) => *c,
            ComponentsStorage::Async { vad, cohort } => {
                let (fbank, ecapa) = self
                    .state
                    .shutdown_worker()?
                    .expect("async storage always holds a worker");
                PipelineComponents {
                    vad,
                    fbank,
                    ecapa,
                    cohort,
                }
            }
        };
        Ok((self.pool, components))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn streaming_config_default_is_dual_rate() {
        let cfg = StreamingConfig::default();
        assert_eq!(cfg.audio_sample_rate, 48_000);
        assert_eq!(cfg.pipeline.sample_rate, 16_000);
        assert!(!cfg.diagnostics);
    }

    #[test]
    fn state_new_identity_rate_has_no_resampler() {
        let mut cfg = StreamingConfig::default();
        cfg.audio_sample_rate = cfg.pipeline.sample_rate;
        let state = StreamingState::new(&cfg).expect("identity-rate state");
        assert!(state.resampler.is_none());
        assert_eq!(state.identity_input_per_frame, CHUNK_SAMPLES_16K);
    }

    #[test]
    fn state_new_dual_rate_builds_resampler() {
        let cfg = StreamingConfig::default();
        let state = StreamingState::new(&cfg).expect("dual-rate state");
        assert!(state.resampler.is_some());
    }

    #[test]
    fn state_new_rejects_async_refresh_on_sync_path() {
        // Sync `new` is for `async_refresh = false` only. Async
        // mode requires `new_async` (which routes through the
        // worker-spawning path; not exercised here because spawning
        // needs real Fbank + EcapaTdnn).
        let mut cfg = StreamingConfig::default();
        cfg.pipeline.async_refresh = true;
        match StreamingState::new(&cfg) {
            Err(PipelineError::Embedding(_)) => {}
            Err(other) => panic!("expected Embedding error, got: {other}"),
            Ok(_) => panic!("sync `new` must reject async_refresh = true"),
        }
    }
}
