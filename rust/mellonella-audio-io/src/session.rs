//! Live session: mic → `StreamingPipeline` → speaker.
//!
//! Owns the three threads the data flow needs (cpal input cb, worker,
//! cpal output cb) and the two bounded queues between them. Drop the
//! [`LiveSession`] to stop; the cpal streams are torn down and the
//! worker exits on the next iteration.

use std::path::PathBuf;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;
use std::thread::JoinHandle;

use cpal::traits::{DeviceTrait, HostTrait, StreamTrait};
use cpal::{Device, Sample, SampleFormat, Stream, StreamConfig};
use crossbeam_channel::{bounded, Receiver, Sender, TrySendError};
use mellonella_core::dfn3::Dfn3Streamer;
use mellonella_core::enrollment::EmbeddingPool;
use mellonella_core::pipeline::PipelineComponents;
use mellonella_core::streaming::{StreamingConfig, StreamingPipeline};
use rubato::{
    Resampler, SincFixedIn, SincInterpolationParameters, SincInterpolationType, WindowFunction,
};

use crate::devices::DeviceKind;
use crate::{AudioIoError, INTERNAL_SAMPLE_RATE};

/// Caller-tunable knobs for [`LiveSession::new`].
///
/// Most users want `SessionConfig::default()`; named device fields
/// let the CLI / GUI honour user choice from the device picker.
#[derive(Debug, Clone, Default)]
pub struct SessionConfig {
    /// Input device name (as returned by [`crate::list_input_devices`]).
    /// `None` → host default input.
    pub input_device: Option<String>,
    /// Output device name. `None` → host default output.
    pub output_device: Option<String>,
    /// Streaming pipeline configuration. `audio_sample_rate` is
    /// overridden internally to [`INTERNAL_SAMPLE_RATE`] (48 kHz) —
    /// the cpal-side resamplers convert from the device rate.
    pub streaming: StreamingConfig,
    /// Internal ring buffer capacity, measured in 48 kHz mono
    /// samples. Default 24_000 (= 0.5 s of buffering) which keeps
    /// glitches absorbing nicely without paying a huge memory bill.
    /// The cpal callbacks drop chunks (input) / emit silence
    /// (output) when this is exhausted.
    pub ring_capacity_samples: usize,
    /// When `Some(path)`, the worker runs DFN3 noise suppression on
    /// the captured audio (at 48 kHz) before handing it to the
    /// streaming pipeline. `None` → no NS (current behaviour from
    /// step 12).
    ///
    /// **Latency trade-off**: DFN3's patched ONNX export is locked
    /// to 102 STFT frames per inference, which means the worker
    /// has to buffer up to ~1.02 s of audio before the first
    /// enhanced sample is available. Surface this trade-off to
    /// users in the UI — the GUI's "Enable noise suppression"
    /// checkbox tooltip and the CLI's `--enable-dfn3` flag both
    /// document it.
    pub dfn3_onnx_path: Option<PathBuf>,
}

/// Stats surfaced when [`LiveSession::stop`] returns. Useful as a
/// post-run "did anything weird happen?" summary for the CLI.
#[derive(Debug, Default, Clone, Copy)]
pub struct LiveSessionStats {
    /// Total samples that flowed through the worker (at 48 kHz mono).
    pub samples_processed: u64,
    /// Input chunks dropped because the input ring was full (worker
    /// can't keep up).
    pub input_overruns: u64,
    /// Output chunks where the output ring was empty and silence
    /// was emitted (worker fell behind).
    pub output_underruns: u64,
}

/// Async event the worker can emit. Step 12 uses only `Error` for
/// pipeline failures; later steps will add diagnostics (gate state,
/// auto-learn events) for the GUI.
#[derive(Debug, Clone)]
pub enum SessionEvent {
    /// Worker-side pipeline error. The worker exits after emitting
    /// this; subsequent `stop()` returns the partial stats.
    Error(String),
}

/// Live audio session. Construct, then optionally drain
/// [`SessionEvent`]s from `events()`, then call `stop()`.
pub struct LiveSession {
    // Streams must outlive the session for cpal to keep callbacks
    // firing. Held as Option so `stop()` can drop them
    // deterministically before joining the worker.
    input_stream: Option<Stream>,
    output_stream: Option<Stream>,
    worker: Option<JoinHandle<()>>,
    events_rx: Receiver<SessionEvent>,
    samples_processed: Arc<AtomicU64>,
    input_overruns: Arc<AtomicU64>,
    output_underruns: Arc<AtomicU64>,
}

impl LiveSession {
    /// Open input + output devices, spawn the worker thread, start
    /// streaming. Returns once everything is running; the caller
    /// holds the session and calls `stop()` when done.
    ///
    /// # Errors
    ///
    /// Surfaces device-query, format-not-supported, stream-construction
    /// and worker-spawn failures as `AudioIoError`.
    pub fn new(
        pool: EmbeddingPool,
        components: PipelineComponents,
        config: SessionConfig,
    ) -> Result<Self, AudioIoError> {
        let host = cpal::default_host();
        let input_dev = pick_device(&host, DeviceKind::Input, config.input_device.as_deref())?;
        let output_dev = pick_device(&host, DeviceKind::Output, config.output_device.as_deref())?;

        let input_cfg = input_dev
            .default_input_config()
            .map_err(|e| AudioIoError::DeviceQuery(e.to_string()))?;
        let output_cfg = output_dev
            .default_output_config()
            .map_err(|e| AudioIoError::DeviceQuery(e.to_string()))?;

        let input_format = input_cfg.sample_format();
        let output_format = output_cfg.sample_format();
        if input_format != SampleFormat::F32 {
            return Err(AudioIoError::UnsupportedSampleFormat {
                format: format!("input {input_format:?}"),
            });
        }
        if output_format != SampleFormat::F32 {
            return Err(AudioIoError::UnsupportedSampleFormat {
                format: format!("output {output_format:?}"),
            });
        }

        let input_sr = input_cfg.sample_rate().0;
        let output_sr = output_cfg.sample_rate().0;
        let input_channels = input_cfg.channels();
        let output_channels = output_cfg.channels();

        eprintln!(
            "[audio-io] input  : {} ({} Hz, {} ch, f32)",
            input_dev.name().unwrap_or_else(|_| "?".into()),
            input_sr,
            input_channels
        );
        eprintln!(
            "[audio-io] output : {} ({} Hz, {} ch, f32)",
            output_dev.name().unwrap_or_else(|_| "?".into()),
            output_sr,
            output_channels
        );

        let ring_cap = config
            .ring_capacity_samples
            .max(INTERNAL_SAMPLE_RATE as usize / 10);
        let (input_tx, input_rx) = bounded::<Vec<f32>>(ring_capacity_chunks(ring_cap));
        let (output_tx, output_rx) = bounded::<Vec<f32>>(ring_capacity_chunks(ring_cap));
        let (events_tx, events_rx) = bounded::<SessionEvent>(64);

        let samples_processed = Arc::new(AtomicU64::new(0));
        let input_overruns = Arc::new(AtomicU64::new(0));
        let output_underruns = Arc::new(AtomicU64::new(0));

        let input_stream = build_input_stream(
            &input_dev,
            input_cfg.clone().into(),
            input_channels,
            input_sr,
            input_tx,
            input_overruns.clone(),
        )?;
        let output_stream = build_output_stream(
            &output_dev,
            output_cfg.clone().into(),
            output_channels,
            output_sr,
            output_rx,
            output_underruns.clone(),
        )?;

        // Override audio_sample_rate so the pipeline sees the
        // post-resample 48 kHz the worker actually feeds it.
        let mut streaming_cfg = config.streaming.clone();
        streaming_cfg.audio_sample_rate = INTERNAL_SAMPLE_RATE;
        let pipeline = StreamingPipeline::new(pool, streaming_cfg, components)
            .map_err(|e| AudioIoError::Pipeline(e.to_string()))?;

        let dfn3 = match config.dfn3_onnx_path.as_deref() {
            Some(path) => Some(
                Dfn3Streamer::from_onnx_path(path)
                    .map_err(|e| AudioIoError::Pipeline(format!("DFN3 load: {e}")))?,
            ),
            None => None,
        };
        if dfn3.is_some() {
            eprintln!("[audio-io] noise suppression: ENABLED (+ ~1.02 s buffering latency)");
        }

        let worker = spawn_worker(
            pipeline,
            dfn3,
            input_rx,
            output_tx,
            events_tx,
            samples_processed.clone(),
        )?;

        input_stream
            .play()
            .map_err(|e| AudioIoError::Stream(e.to_string()))?;
        output_stream
            .play()
            .map_err(|e| AudioIoError::Stream(e.to_string()))?;

        Ok(Self {
            input_stream: Some(input_stream),
            output_stream: Some(output_stream),
            worker: Some(worker),
            events_rx,
            samples_processed,
            input_overruns,
            output_underruns,
        })
    }

    /// Try to receive a pending session event without blocking.
    /// Returns `None` when no event is queued.
    pub fn try_recv_event(&self) -> Option<SessionEvent> {
        self.events_rx.try_recv().ok()
    }

    /// Live snapshot of the running counters. Useful for the CLI to
    /// print a periodic "still going" line.
    #[must_use]
    pub fn stats_snapshot(&self) -> LiveSessionStats {
        LiveSessionStats {
            samples_processed: self.samples_processed.load(Ordering::Relaxed),
            input_overruns: self.input_overruns.load(Ordering::Relaxed),
            output_underruns: self.output_underruns.load(Ordering::Relaxed),
        }
    }

    /// Stop the session and return the final stats. Drops the cpal
    /// streams first (so the input queue stops growing), then joins
    /// the worker (which exits when the input channel closes).
    ///
    /// # Errors
    ///
    /// Returns `WorkerDied` if the worker thread panicked.
    pub fn stop(mut self) -> Result<LiveSessionStats, AudioIoError> {
        self.input_stream.take();
        // Output stream stays alive until the worker has flushed —
        // we just drop it after joining so trailing audio still
        // flows out.
        if let Some(worker) = self.worker.take() {
            worker
                .join()
                .map_err(|_| AudioIoError::WorkerDied("panicked".into()))?;
        }
        self.output_stream.take();
        Ok(self.stats_snapshot())
    }
}

impl Drop for LiveSession {
    fn drop(&mut self) {
        // If the caller forgot `stop()`, do the same shutdown
        // sequence on Drop. Errors are swallowed — there's nowhere
        // useful to report them from a destructor.
        self.input_stream.take();
        if let Some(worker) = self.worker.take() {
            let _ = worker.join();
        }
        self.output_stream.take();
    }
}

/// One worker iteration drains as much input as is available, runs
/// the pipeline, sends output downstream. Falls out of the loop when
/// the input channel disconnects (i.e. all senders dropped because
/// the input cpal stream was torn down).
fn spawn_worker(
    mut pipeline: StreamingPipeline,
    mut dfn3: Option<Dfn3Streamer>,
    input_rx: Receiver<Vec<f32>>,
    output_tx: Sender<Vec<f32>>,
    events_tx: Sender<SessionEvent>,
    samples_processed: Arc<AtomicU64>,
) -> Result<JoinHandle<()>, AudioIoError> {
    std::thread::Builder::new()
        .name("mellonella-audio-io-worker".into())
        .spawn(move || {
            while let Ok(chunk) = input_rx.recv() {
                // DFN3 (if enabled) runs first — it buffers up to
                // ~1.02 s and emits whole chunks. When the buffer
                // isn't full yet, `enhanced` is empty and we skip
                // straight to the next input.
                let enhanced = if let Some(streamer) = dfn3.as_mut() {
                    match streamer.push_samples(&chunk) {
                        Ok(buf) => buf,
                        Err(e) => {
                            let _ = events_tx.send(SessionEvent::Error(format!("DFN3: {e}")));
                            return;
                        }
                    }
                } else {
                    chunk
                };
                if enhanced.is_empty() {
                    continue;
                }
                match pipeline.push_samples(&enhanced) {
                    Ok(out) => {
                        let n = out.audio.len() as u64;
                        if n > 0 && output_tx.send(out.audio).is_err() {
                            // Output side disconnected — session is
                            // shutting down. Exit cleanly.
                            break;
                        }
                        samples_processed.fetch_add(n, Ordering::Relaxed);
                    }
                    Err(e) => {
                        let _ = events_tx.send(SessionEvent::Error(e.to_string()));
                        return;
                    }
                }
            }
            // Input closed — flush DFN3 first so any sub-chunk
            // residue gets enhanced, then flush the streaming
            // pipeline.
            if let Some(streamer) = dfn3.as_mut() {
                match streamer.flush() {
                    Ok(buf) if !buf.is_empty() => {
                        if let Ok(out) = pipeline.push_samples(&buf) {
                            let n = out.audio.len() as u64;
                            if n > 0 {
                                let _ = output_tx.send(out.audio);
                            }
                            samples_processed.fetch_add(n, Ordering::Relaxed);
                        }
                    }
                    Ok(_) => {}
                    Err(e) => {
                        let _ = events_tx.send(SessionEvent::Error(format!("DFN3 flush: {e}")));
                    }
                }
            }
            if let Ok(tail) = pipeline.flush() {
                let n = tail.audio.len() as u64;
                if n > 0 {
                    let _ = output_tx.send(tail.audio);
                }
                samples_processed.fetch_add(n, Ordering::Relaxed);
            }
        })
        .map_err(|e| AudioIoError::Stream(format!("spawn worker: {e}")))
}

/// Convert a sample-count ring capacity to a sensible chunk count.
/// `crossbeam_channel::bounded` takes a slot count, not a sample
/// count; chunks vary in size by device callback period, but a
/// 1-ms-equivalent slot count is a fine ceiling: 24 000 samples ≈
/// 500 slots of 1 ms each.
fn ring_capacity_chunks(ring_cap_samples: usize) -> usize {
    let slots = ring_cap_samples / (INTERNAL_SAMPLE_RATE as usize / 1000).max(1);
    slots.max(16)
}

fn pick_device(
    host: &cpal::Host,
    kind: DeviceKind,
    name: Option<&str>,
) -> Result<Device, AudioIoError> {
    if let Some(name) = name {
        let iter = match kind {
            DeviceKind::Input => host
                .input_devices()
                .map_err(|e| AudioIoError::DeviceQuery(e.to_string()))?,
            DeviceKind::Output => host
                .output_devices()
                .map_err(|e| AudioIoError::DeviceQuery(e.to_string()))?,
        };
        for dev in iter {
            if dev.name().ok().as_deref() == Some(name) {
                return Ok(dev);
            }
        }
        Err(AudioIoError::DeviceNotFound {
            kind,
            name: name.to_string(),
        })
    } else {
        let dev = match kind {
            DeviceKind::Input => host.default_input_device(),
            DeviceKind::Output => host.default_output_device(),
        };
        dev.ok_or(AudioIoError::NoDefaultDevice(kind))
    }
}

fn build_input_stream(
    device: &Device,
    config: StreamConfig,
    channels: u16,
    device_sr: u32,
    input_tx: Sender<Vec<f32>>,
    overruns: Arc<AtomicU64>,
) -> Result<Stream, AudioIoError> {
    let mut resampler = build_resampler(device_sr, INTERNAL_SAMPLE_RATE)?;
    let channels_usize = channels as usize;
    let err_fn = |e: cpal::StreamError| eprintln!("[audio-io] input stream error: {e}");
    device
        .build_input_stream(
            &config,
            move |data: &[f32], _: &cpal::InputCallbackInfo| {
                // Downmix interleaved frames to mono by averaging.
                let mut mono = Vec::with_capacity(data.len() / channels_usize);
                let mut sum = 0.0_f32;
                let mut count = 0_usize;
                for (i, &s) in data.iter().enumerate() {
                    sum += s;
                    count += 1;
                    if (i + 1) % channels_usize == 0 {
                        mono.push(sum / count as f32);
                        sum = 0.0;
                        count = 0;
                    }
                }
                let processed = match &mut resampler {
                    Some(r) => resample_one(r, &mono),
                    None => Ok(mono),
                };
                match processed {
                    Ok(chunk) if !chunk.is_empty() => {
                        if let Err(TrySendError::Full(_)) = input_tx.try_send(chunk) {
                            overruns.fetch_add(1, Ordering::Relaxed);
                        }
                        // Disconnected means the session is shutting
                        // down; the dropped chunk is fine.
                    }
                    Ok(_) => {}
                    Err(e) => eprintln!("[audio-io] input resample: {e}"),
                }
            },
            err_fn,
            None,
        )
        .map_err(|e| AudioIoError::Stream(e.to_string()))
}

fn build_output_stream(
    device: &Device,
    config: StreamConfig,
    channels: u16,
    device_sr: u32,
    output_rx: Receiver<Vec<f32>>,
    underruns: Arc<AtomicU64>,
) -> Result<Stream, AudioIoError> {
    let mut resampler = build_resampler(INTERNAL_SAMPLE_RATE, device_sr)?;
    let channels_usize = channels as usize;
    // Carry-over mono samples that didn't fit in the previous
    // callback's device buffer. cpal's output callback hands us a
    // fixed-size scratch slice each time; remaining samples wait
    // for the next call.
    let mut carry: Vec<f32> = Vec::new();
    let err_fn = |e: cpal::StreamError| eprintln!("[audio-io] output stream error: {e}");
    device
        .build_output_stream(
            &config,
            move |data: &mut [f32], _: &cpal::OutputCallbackInfo| {
                let frames_needed = data.len() / channels_usize;
                // Top up the carry buffer until it has enough mono
                // samples to fill `data`, draining the output ring.
                while carry.len() < frames_needed {
                    let Ok(chunk) = output_rx.try_recv() else {
                        break;
                    };
                    let resampled = match &mut resampler {
                        Some(r) => resample_one(r, &chunk).unwrap_or_default(),
                        None => chunk,
                    };
                    carry.extend_from_slice(&resampled);
                }
                if carry.len() < frames_needed {
                    underruns.fetch_add(1, Ordering::Relaxed);
                }
                // Broadcast mono → all output channels. Silence on
                // underrun.
                for frame_idx in 0..frames_needed {
                    let mono = carry.get(frame_idx).copied().unwrap_or(0.0);
                    let base = frame_idx * channels_usize;
                    for ch in 0..channels_usize {
                        data[base + ch] = Sample::from_sample(mono);
                    }
                }
                if carry.len() > frames_needed {
                    carry.drain(..frames_needed);
                } else {
                    carry.clear();
                }
            },
            err_fn,
            None,
        )
        .map_err(|e| AudioIoError::Stream(e.to_string()))
}

/// Build a streaming sinc resampler `src_sr → dst_sr`, or `None` if
/// the rates already match (skip the no-op cost).
fn build_resampler(src_sr: u32, dst_sr: u32) -> Result<Option<SincFixedIn<f32>>, AudioIoError> {
    if src_sr == dst_sr {
        return Ok(None);
    }
    let ratio = f64::from(dst_sr) / f64::from(src_sr);
    let params = SincInterpolationParameters {
        sinc_len: 256,
        f_cutoff: 0.95,
        interpolation: SincInterpolationType::Linear,
        oversampling_factor: 256,
        window: WindowFunction::BlackmanHarris2,
    };
    // 1024 mono frames per call → ~21 ms @ 48 kHz; bigger than any
    // realistic cpal callback so the resampler can absorb a whole
    // callback at once.
    let r = SincFixedIn::<f32>::new(ratio, 1.1, params, 1024, 1)
        .map_err(|e| AudioIoError::Resample(e.to_string()))?;
    Ok(Some(r))
}

/// One resample step. Splits the input into the resampler's
/// fixed-size chunks, processing each in turn; the residue is
/// padded with zeros for simplicity (acceptable artifact at chunk
/// edges for the live demo, fixed in a follow-up step).
fn resample_one(resampler: &mut SincFixedIn<f32>, mono: &[f32]) -> Result<Vec<f32>, String> {
    let frames = resampler.input_frames_next();
    let mut out = Vec::new();
    let mut idx = 0;
    while idx < mono.len() {
        let end = (idx + frames).min(mono.len());
        let mut chunk = vec![0.0_f32; frames];
        chunk[..end - idx].copy_from_slice(&mono[idx..end]);
        let input = [chunk.as_slice()];
        let mut output = vec![vec![0.0_f32; resampler.output_frames_max()]];
        resampler
            .process_into_buffer(&input, &mut output, None)
            .map_err(|e| e.to_string())?;
        let produced = resampler.output_frames_next();
        out.extend_from_slice(&output[0][..produced]);
        idx = end;
    }
    Ok(out)
}
