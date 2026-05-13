//! Mellonella desktop audio I/O — Phase B.
//!
//! Bridges the operating system's audio devices (via `cpal`) and
//! [`mellonella_core::streaming::StreamingPipeline`]. The goal is to
//! let the CLI and a future GUI use the live filter end-to-end —
//! microphone in, envelope-gated audio out the default output device
//! — without each caller having to wrestle cpal's threading model
//! or sample-rate / channel-format conversion.
//!
//! # Status
//!
//! Phase B step 12 (this crate's introduction):
//!
//! * **In**: device enumeration, [`LiveSession`] that opens a
//!   default-input → pipeline → default-output route, runs until
//!   stopped, and surfaces basic counters on shutdown.
//! * **Constraints (intentional)**: f32 sample format only — most
//!   modern systems expose f32 by default; we reject otherwise with
//!   a clear error rather than implementing every cpal format
//!   shim. Multi-channel inputs are downmixed to mono by averaging;
//!   multi-channel outputs broadcast the same sample to every
//!   channel.
//! * **Out of scope (deferred)**: DFN3 in the live path, hot-swap on
//!   device disconnect, latency display, channel-aware (non-average)
//!   downmixing.
//!
//! # Threading model
//!
//! ```text
//! [cpal input callback]                      [worker thread]                       [cpal output callback]
//!   capture device frames                       <- recv from input ring                <- recv from output ring
//!   downmix to f32 mono                         StreamingPipeline.push_samples         broadcast mono to channels
//!   resample device_sr -> 48 kHz                StreamingPipeline.flush?               resample 48 kHz -> device_sr
//!   send to input ring (drop-on-full)           send to output ring                    write into device buffer
//! ```
//!
//! The rings are bounded `crossbeam_channel` queues; the input side
//! drops chunks when full (better than blocking a real-time
//! callback), the output side emits silence when empty (better than
//! holding up the speaker). Both behaviours are logged so the caller
//! knows the pipeline is falling behind.

#![forbid(unsafe_code)]
// Pedantic lints that fight an audio-I/O crate's idioms:
//
// * `similar_names`: paired `input_*` / `output_*` variables are
//   intentional everywhere.
// * `cast_precision_loss` / `cast_possible_truncation` /
//   `cast_sign_loss`: device frame counts and sample counts are
//   small (≤ 1 million); the casts are fine in practice.
// * `needless_pass_by_value`: cpal's `StreamConfig` and
//   `SessionConfig` are small `Clone` structs; taking by value at
//   public boundaries is idiomatic and cheap.
// * `must_use_candidate`: trivially true for getter-style methods;
//   tagging every one is noise.
// * `missing_errors_doc`: error variants are documented on
//   `AudioIoError`; per-function rehash is redundant.
#![allow(
    clippy::similar_names,
    clippy::cast_precision_loss,
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss,
    clippy::needless_pass_by_value,
    clippy::must_use_candidate,
    clippy::missing_errors_doc,
    clippy::too_many_arguments
)]

mod devices;
mod recording;
mod session;

pub use devices::{list_input_devices, list_output_devices, AudioDevice, DeviceKind};
pub use recording::Recorder;
pub use session::{LiveSession, LiveSessionStats, SessionConfig, SessionEvent};

/// Errors surfaced by the audio-IO crate.
#[derive(Debug)]
pub enum AudioIoError {
    /// `cpal` reported no default input or output device.
    NoDefaultDevice(DeviceKind),
    /// Could not query device configuration (host I/O failure or
    /// permission denied).
    DeviceQuery(String),
    /// The named device wasn't found in the host enumeration.
    DeviceNotFound { kind: DeviceKind, name: String },
    /// The device's native sample format isn't `f32`. Step 12
    /// rejects other formats with a clear error rather than
    /// implementing every cpal format shim.
    UnsupportedSampleFormat { format: String },
    /// cpal stream construction / start failure.
    Stream(String),
    /// Resampler init / step failure (rubato).
    Resample(String),
    /// Underlying core pipeline failure surfaced from the worker.
    Pipeline(String),
    /// The worker thread panicked or exited unexpectedly.
    WorkerDied(String),
}

impl std::fmt::Display for AudioIoError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::NoDefaultDevice(kind) => write!(f, "no default {kind} device available"),
            Self::DeviceQuery(msg) => write!(f, "device query failed: {msg}"),
            Self::DeviceNotFound { kind, name } => {
                write!(f, "no {kind} device named {name:?}")
            }
            Self::UnsupportedSampleFormat { format } => {
                write!(
                    f,
                    "unsupported sample format {format:?} — only f32 is accepted in step 12"
                )
            }
            Self::Stream(msg) => write!(f, "cpal stream error: {msg}"),
            Self::Resample(msg) => write!(f, "resampler error: {msg}"),
            Self::Pipeline(msg) => write!(f, "pipeline error: {msg}"),
            Self::WorkerDied(msg) => write!(f, "worker thread died: {msg}"),
        }
    }
}

impl std::error::Error for AudioIoError {}

/// The rate the live pipeline operates on between cpal-side
/// resamplers and `StreamingPipeline`. Matches the CLI's offline
/// `OUTPUT_SAMPLE_RATE` and `StreamingConfig::audio_sample_rate`
/// default so all stages share the same audio-path rate.
pub const INTERNAL_SAMPLE_RATE: u32 = 48_000;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn device_kind_display_is_lowercase() {
        assert_eq!(DeviceKind::Input.to_string(), "input");
        assert_eq!(DeviceKind::Output.to_string(), "output");
    }

    #[test]
    fn audio_io_error_messages_are_informative() {
        let err = AudioIoError::NoDefaultDevice(DeviceKind::Input);
        assert_eq!(err.to_string(), "no default input device available");
        let err = AudioIoError::UnsupportedSampleFormat {
            format: "I16".into(),
        };
        assert!(err.to_string().contains("I16"));
        let err = AudioIoError::DeviceNotFound {
            kind: DeviceKind::Output,
            name: "Studio Mic".into(),
        };
        assert!(err.to_string().contains("Studio Mic"));
        assert!(err.to_string().contains("output"));
    }

    #[test]
    fn internal_sample_rate_matches_streaming_default() {
        // The whole crate is built around this rate matching the
        // streaming pipeline's audio-path default; the offline CLI
        // (`mellonella-cli`) also uses 48 kHz. Lock the invariant.
        assert_eq!(
            INTERNAL_SAMPLE_RATE,
            mellonella_core::streaming::StreamingConfig::default().audio_sample_rate
        );
    }

    #[test]
    fn device_enumeration_does_not_panic() {
        // Some CI runners have zero audio devices; either the
        // enumeration returns Ok(vec![]) or surfaces a clean
        // `DeviceQuery` — both are fine. The only outcome we reject
        // is a panic, which would mean the cpal init path is
        // unsound.
        let _ = list_input_devices();
        let _ = list_output_devices();
    }
}
