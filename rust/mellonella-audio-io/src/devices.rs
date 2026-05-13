//! Device enumeration. Returns the host's input / output device list
//! plus their default configurations so the CLI / GUI can offer a
//! picker without exposing raw cpal types.

use cpal::traits::{DeviceTrait, HostTrait};

use crate::AudioIoError;

/// Whether a device is an input (microphone) or output (speaker).
/// Used for error messages and to pick which host enumeration to
/// walk.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DeviceKind {
    Input,
    Output,
}

impl std::fmt::Display for DeviceKind {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Input => f.write_str("input"),
            Self::Output => f.write_str("output"),
        }
    }
}

/// Device descriptor exposed to callers.
///
/// `is_default` is set on whichever device the host reports as the
/// default input/output, so the CLI can star it in `mellonella
/// devices`.
#[derive(Debug, Clone)]
pub struct AudioDevice {
    pub kind: DeviceKind,
    pub name: String,
    pub is_default: bool,
    pub default_sample_rate: u32,
    pub default_channels: u16,
    pub default_sample_format: String,
}

impl std::fmt::Display for AudioDevice {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let marker = if self.is_default { "* " } else { "  " };
        write!(
            f,
            "{marker}[{kind}] {name} ({sr} Hz, {ch} ch, {fmt})",
            kind = self.kind,
            name = self.name,
            sr = self.default_sample_rate,
            ch = self.default_channels,
            fmt = self.default_sample_format,
        )
    }
}

fn enumerate(kind: DeviceKind) -> Result<Vec<AudioDevice>, AudioIoError> {
    let host = cpal::default_host();
    let default_name = match kind {
        DeviceKind::Input => host.default_input_device().and_then(|d| d.name().ok()),
        DeviceKind::Output => host.default_output_device().and_then(|d| d.name().ok()),
    };

    let iter: Box<dyn Iterator<Item = cpal::Device>> = match kind {
        DeviceKind::Input => Box::new(
            host.input_devices()
                .map_err(|e| AudioIoError::DeviceQuery(e.to_string()))?,
        ),
        DeviceKind::Output => Box::new(
            host.output_devices()
                .map_err(|e| AudioIoError::DeviceQuery(e.to_string()))?,
        ),
    };

    let mut out = Vec::new();
    for dev in iter {
        let name = dev
            .name()
            .map_err(|e| AudioIoError::DeviceQuery(e.to_string()))?;
        let cfg = match kind {
            DeviceKind::Input => dev.default_input_config(),
            DeviceKind::Output => dev.default_output_config(),
        };
        let Ok(cfg) = cfg else {
            // Some hosts list devices that can't be queried for a
            // default config (e.g. busy by another app). Skip
            // silently — listing should still mostly succeed.
            continue;
        };
        let is_default = default_name.as_deref() == Some(name.as_str());
        out.push(AudioDevice {
            kind,
            name,
            is_default,
            default_sample_rate: cfg.sample_rate().0,
            default_channels: cfg.channels(),
            default_sample_format: format!("{:?}", cfg.sample_format()),
        });
    }
    Ok(out)
}

/// List input (microphone) devices visible to the default host. The
/// host's default input device is marked `is_default = true`.
pub fn list_input_devices() -> Result<Vec<AudioDevice>, AudioIoError> {
    enumerate(DeviceKind::Input)
}

/// List output (speaker) devices visible to the default host. The
/// host's default output device is marked `is_default = true`.
pub fn list_output_devices() -> Result<Vec<AudioDevice>, AudioIoError> {
    enumerate(DeviceKind::Output)
}
