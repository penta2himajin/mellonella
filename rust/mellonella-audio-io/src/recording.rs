//! Short-duration microphone capture used by the GUI's enrollment
//! flow: opens an input device, captures N seconds of mono f32 at a
//! target sample rate (typically 16 kHz for ECAPA), and hands the
//! buffer back through a channel.
//!
//! Threading mirrors [`crate::LiveSession`] on the input side: the
//! cpal callback runs on its own thread, downmixes + resamples, and
//! pushes chunks into a bounded `crossbeam_channel`. A worker
//! thread accumulates until the target length is reached or the
//! caller cancels.

use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::Arc;
use std::thread::JoinHandle;
use std::time::Duration;

use cpal::traits::{DeviceTrait, HostTrait, StreamTrait};
use cpal::SampleFormat;
use crossbeam_channel::{bounded, Receiver};
use rubato::{
    Resampler, SincFixedIn, SincInterpolationParameters, SincInterpolationType, WindowFunction,
};

use crate::devices::DeviceKind;
use crate::AudioIoError;

/// Single-shot microphone recorder. Construct with [`Recorder::start`],
/// poll [`Recorder::try_finish`] each UI frame, optionally call
/// [`Recorder::cancel`] for early termination.
pub struct Recorder {
    thread: Option<JoinHandle<()>>,
    result_rx: Receiver<Result<Vec<f32>, AudioIoError>>,
    samples_collected: Arc<AtomicUsize>,
    target_samples: usize,
    target_sample_rate: u32,
    cancel: Arc<AtomicBool>,
}

impl Recorder {
    /// Open `input_device` (or host default if `None`), spawn the
    /// capture worker, and start recording.
    ///
    /// Records until either `max_duration_secs` worth of samples
    /// have been collected at `target_sample_rate` or the caller
    /// invokes [`Self::cancel`].
    ///
    /// # Errors
    ///
    /// Surfaces device-query / unsupported-format / stream-construction
    /// failures as `AudioIoError`. Currently accepts `f32` input
    /// devices only — same constraint as [`crate::LiveSession`].
    pub fn start(
        input_device: Option<String>,
        target_sample_rate: u32,
        max_duration_secs: f32,
    ) -> Result<Self, AudioIoError> {
        let host = cpal::default_host();
        let device = pick_device(&host, input_device.as_deref())?;
        let cfg = device
            .default_input_config()
            .map_err(|e| AudioIoError::DeviceQuery(e.to_string()))?;
        if cfg.sample_format() != SampleFormat::F32 {
            return Err(AudioIoError::UnsupportedSampleFormat {
                format: format!("input {:?}", cfg.sample_format()),
            });
        }

        let device_sr = cfg.sample_rate().0;
        let channels = cfg.channels();
        let stream_cfg = cfg.into();

        let target_samples = (max_duration_secs * target_sample_rate as f32).round() as usize;
        let samples_collected = Arc::new(AtomicUsize::new(0));
        let cancel = Arc::new(AtomicBool::new(false));
        let (result_tx, result_rx) = bounded::<Result<Vec<f32>, AudioIoError>>(1);

        let samples_collected_for_worker = samples_collected.clone();
        let cancel_for_worker = cancel.clone();
        let thread = std::thread::Builder::new()
            .name("mellonella-recorder".into())
            .spawn(move || {
                let result = run_recording(
                    &device,
                    &stream_cfg,
                    channels,
                    device_sr,
                    target_sample_rate,
                    target_samples,
                    &samples_collected_for_worker,
                    &cancel_for_worker,
                );
                let _ = result_tx.send(result);
            })
            .map_err(|e| AudioIoError::Stream(format!("spawn recorder thread: {e}")))?;

        Ok(Self {
            thread: Some(thread),
            result_rx,
            samples_collected,
            target_samples,
            target_sample_rate,
            cancel,
        })
    }

    /// Try to fetch the recording result without blocking. Returns
    /// `None` while the worker is still collecting samples; returns
    /// `Some(Ok(buf))` when capture is complete or `Some(Err(_))`
    /// on failure. After a successful poll the join handle is
    /// drained so a subsequent call returns `None`.
    pub fn try_finish(&mut self) -> Option<Result<Vec<f32>, AudioIoError>> {
        match self.result_rx.try_recv() {
            Ok(result) => {
                if let Some(t) = self.thread.take() {
                    let _ = t.join();
                }
                Some(result)
            }
            Err(_) => None,
        }
    }

    /// Signal the worker to stop as soon as it processes the next
    /// chunk. The worker still returns whatever it has collected so
    /// far (a partial recording).
    pub fn cancel(&self) {
        self.cancel.store(true, Ordering::Relaxed);
    }

    /// Captured-seconds-so-far, used by the UI for a progress
    /// indicator. Lock-free atomic read.
    #[must_use]
    pub fn elapsed_seconds(&self) -> f32 {
        self.samples_collected.load(Ordering::Relaxed) as f32 / self.target_sample_rate as f32
    }

    /// Capture progress in `[0, 1]`. Clamped at 1.0 so a small
    /// over-shoot in the final chunk doesn't show > 100 %.
    #[must_use]
    pub fn progress(&self) -> f32 {
        if self.target_samples == 0 {
            return 1.0;
        }
        (self.samples_collected.load(Ordering::Relaxed) as f32 / self.target_samples as f32)
            .min(1.0)
    }

    /// Total seconds the recorder was asked to capture (not the
    /// elapsed seconds — that's [`Self::elapsed_seconds`]).
    #[must_use]
    pub fn target_seconds(&self) -> f32 {
        self.target_samples as f32 / self.target_sample_rate as f32
    }
}

impl Drop for Recorder {
    fn drop(&mut self) {
        self.cancel.store(true, Ordering::Relaxed);
        if let Some(t) = self.thread.take() {
            let _ = t.join();
        }
    }
}

fn pick_device(host: &cpal::Host, name: Option<&str>) -> Result<cpal::Device, AudioIoError> {
    if let Some(n) = name {
        let iter = host
            .input_devices()
            .map_err(|e| AudioIoError::DeviceQuery(e.to_string()))?;
        for dev in iter {
            if dev.name().ok().as_deref() == Some(n) {
                return Ok(dev);
            }
        }
        Err(AudioIoError::DeviceNotFound {
            kind: DeviceKind::Input,
            name: n.to_string(),
        })
    } else {
        host.default_input_device()
            .ok_or(AudioIoError::NoDefaultDevice(DeviceKind::Input))
    }
}

fn run_recording(
    device: &cpal::Device,
    stream_cfg: &cpal::StreamConfig,
    channels: u16,
    device_sr: u32,
    target_sr: u32,
    target_samples: usize,
    samples_collected: &AtomicUsize,
    cancel: &AtomicBool,
) -> Result<Vec<f32>, AudioIoError> {
    let (chunk_tx, chunk_rx) = bounded::<Vec<f32>>(64);
    let channels_usize = channels as usize;
    let mut resampler = build_resampler(device_sr, target_sr)?;

    let err_fn = |e: cpal::StreamError| eprintln!("[recorder] input stream error: {e}");
    let stream = device
        .build_input_stream(
            stream_cfg,
            move |data: &[f32], _: &cpal::InputCallbackInfo| {
                let mono = downmix_to_mono(data, channels_usize);
                let resampled = match &mut resampler {
                    Some(r) => resample_chunk(r, &mono).unwrap_or(mono),
                    None => mono,
                };
                let _ = chunk_tx.try_send(resampled);
            },
            err_fn,
            None,
        )
        .map_err(|e| AudioIoError::Stream(e.to_string()))?;
    stream
        .play()
        .map_err(|e| AudioIoError::Stream(e.to_string()))?;

    let mut buffer: Vec<f32> = Vec::with_capacity(target_samples);
    while buffer.len() < target_samples && !cancel.load(Ordering::Relaxed) {
        match chunk_rx.recv_timeout(Duration::from_millis(50)) {
            Ok(chunk) => {
                buffer.extend_from_slice(&chunk);
                samples_collected.store(buffer.len(), Ordering::Relaxed);
            }
            Err(crossbeam_channel::RecvTimeoutError::Timeout) => {}
            Err(crossbeam_channel::RecvTimeoutError::Disconnected) => break,
        }
    }

    // Drop the stream first so the cpal callback stops trying to
    // push into a now-unreceived channel.
    drop(stream);

    if buffer.len() > target_samples {
        buffer.truncate(target_samples);
    }
    samples_collected.store(buffer.len(), Ordering::Relaxed);
    Ok(buffer)
}

fn downmix_to_mono(data: &[f32], channels: usize) -> Vec<f32> {
    // Recorder is shared between the GUI's mic-enrollment flow and
    // potential future uses; it always averages because the
    // typical "register my voice" scenario doesn't carry a
    // channel-selection UI.
    crate::ChannelStrategy::Average.downmix(data, channels)
}

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
    let r = SincFixedIn::<f32>::new(ratio, 1.1, params, 1024, 1)
        .map_err(|e| AudioIoError::Resample(e.to_string()))?;
    Ok(Some(r))
}

fn resample_chunk(resampler: &mut SincFixedIn<f32>, mono: &[f32]) -> Result<Vec<f32>, String> {
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
