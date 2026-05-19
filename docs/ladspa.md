# LADSPA plugin

`mellonella-ladspa` packages the streaming mellonella engine —
[`StreamingPipeline`](../rust/mellonella-core/src/streaming.rs) with
DeepFilterNet 3 noise suppression plus the optional target-speaker
hard gate — as a single LADSPA `.so` that PipeWire, JACK, or
Audacity can load directly.

Sample-rate policy: **48 kHz only.** DFN3 is 48 kHz-native; the
plugin refuses any other rate at `instantiate()` time rather than
trying to inline a resampler. PipeWire / Pulse / JACK do high-quality
SR conversion at the graph edge anyway, so this costs zero extra
latency in practice.

## Build

```sh
cd rust
cargo build -p mellonella-ladspa --release
```

The build artifact is `rust/target/release/libmellonella_ladspa.so`
(~4 MB). It dynamically loads ONNX Runtime via `ort`'s
`load-dynamic` feature, so the system also needs `libonnxruntime.so`
on the loader path (the same dependency mellonella-gui already has).

## Install

LADSPA hosts scan `$LADSPA_PATH` (colon-separated) and fall back to
`/usr/lib/ladspa:/usr/local/lib/ladspa`. The simplest per-user
install:

```sh
scripts/install-ladspa.sh         # → ~/.ladspa/libmellonella_ladspa.so
export LADSPA_PATH="$HOME/.ladspa:${LADSPA_PATH:-/usr/lib/ladspa}"
```

Verify the host can see it:

```sh
$ analyseplugin ~/.ladspa/libmellonella_ladspa.so
Plugin Name: "Mellonella Target-Speaker Gate"
Plugin Label: "mellonella_gate"
...
```

## Configuration

LADSPA control ports only carry `f32`, so non-numeric configuration
arrives out-of-band:

| Source | Override env var | Default |
|---|---|---|
| Enrollment JSON | `MELLONELLA_ENROLLMENT` | `<dirs::config_dir>/mellonella/enrollment.json` (where mellonella-gui auto-saves) |
| DFN3 ONNX | `MELLONELLA_DFN3_ONNX` | HuggingFace cache |
| VAD ONNX | (cache only) | HuggingFace cache |
| ECAPA ONNX | (cache only) | HuggingFace cache |

If no enrollment is found, the plugin runs in **DFN3-only +
bootstrap-auto-learn** mode: the speaker gate stays fully open, but
the streaming engine still observes refreshes and seeds the first
anchor from the first second of voiced speech it hears. Subsequent
admissions accumulate into the adapted embedding, and on
`deactivate()` the pool is persisted back to the enrollment path —
so the next session loads the learned profile and runs the full
gate. Existing enrollment files are never overwritten by this path
(persistence only fires when the session started with an empty
pool).

This means the LADSPA plugin can be dropped in cold: the first call
or two will pass through unfiltered (gate open, DFN3 cleaning), and
within a few minutes of normal use the gate becomes effective on
subsequent sessions. If you want immediate gating, enroll explicitly
via `mellonella-gui` first.

### Control ports

| # | Name | Direction | Range | Default | Notes |
|---|---|---|---|---|---|
| 0 | In | audio in | — | — | Mono |
| 1 | Out | audio out | — | — | Mono |
| 2 | Bypass | control in | 0 / 1 | 0 | When ≥0.5 the dry input passes through unchanged |
| 3 | Dry/Wet | control in | 0.0 – 1.0 | 1.0 | Linear mix |
| 4 | VAD Threshold | control in | 0.0 – 1.0 | 0.5 | Reserved — engine setter is WIP |
| 5 | Gate | control out | 0 / 1 | — | 1 while the gate is open (visualisation) |
| 6 | Score | control out | 0.0 – 1.0 | — | Speaker-match score (visualisation) |

## Using it in PipeWire (recommended)

`pipewire`'s `module-filter-chain` instantiates a LADSPA plugin and
exposes it as a virtual source + sink so any app can pick the
filtered signal as its microphone.

`~/.config/pipewire/pipewire.conf.d/99-mellonella.conf`:

```conf
context.modules = [
  { name = libpipewire-module-filter-chain
    args = {
      node.description = "Mellonella Filter"
      media.name       = "Mellonella Filter"
      filter.graph = {
        nodes = [
          { type   = ladspa
            name   = mellonella
            plugin = mellonella_ladspa
            label  = mellonella_gate
            control = { "Bypass" = 0.0, "Dry/Wet" = 1.0 }
          }
        ]
      }
      capture.props = {
        node.name       = "mellonella_input"
        media.class     = "Audio/Sink"
        audio.position  = "MONO"
        audio.rate      = 48000
      }
      playback.props = {
        node.name       = "mellonella_output"
        media.class     = "Audio/Source"
        audio.position  = "MONO"
        audio.rate      = 48000
      }
    }
  }
]
```

Then `systemctl --user restart pipewire wireplumber pipewire-pulse`,
and route your real mic into the **Mellonella Filter (sink)** via
`pavucontrol`. Apps (Zoom, Meet, Discord, OBS) will see
**Mellonella Filter (source)** as a regular microphone — selecting it
is enough.

### Loopback shortcut

If you just want to listen to yourself through the filter:

```sh
pw-loopback --capture-props='node.name=mic_in' \
            --playback-props='node.name=mellonella_input'
```

## Using it in JACK / Audacity

Both load `LADSPA_PATH` plugins automatically. In Audacity the
plugin shows up under **Effect → LADSPA**; in QJackCtl /
`jack-rack` it's available as a regular plugin block.

## Troubleshooting

* `failed to activate ... sample rate 44100 Hz is not supported`
  → set the PipeWire / JACK graph rate to 48000 Hz (e.g.
  `default.clock.rate = 48000` in
  `~/.config/pipewire/pipewire.conf.d/10-clock.conf`).
* No suppression, only passthrough → check stderr from PipeWire
  (`journalctl --user -u pipewire -f`) for the activation banner.
  "no enrollment.json found" means the plugin is in NS-only mode by
  design; run `mellonella-gui` once and re-trigger `activate()` (a
  PipeWire restart is enough).
* `libonnxruntime.so: cannot open shared object` → install ONNX
  Runtime system-wide or set `ORT_DYLIB_PATH=/path/to/libonnxruntime.so`
  before launching the audio server.
