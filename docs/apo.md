# Windows APO plugin

`mellonella-apo` packages the streaming mellonella engine —
[`StreamingPipeline`](../rust/mellonella-core/src/streaming.rs) with
DeepFilterNet 3 noise suppression plus the optional target-speaker
hard gate — as a Windows
[Audio Processing Object](https://learn.microsoft.com/en-us/windows-hardware/drivers/audio/audio-processing-object-architecture)
(an `Sfx` per-stream effect, so it sits on the capture path of a
single audio endpoint).

The implementation is the Windows-side counterpart to
[`mellonella-ladspa`](./ladspa.md); both consume the same
mellonella-core pipeline and the same `enrollment.json` auto-saved
by `mellonella-gui`, so a single enrollment works everywhere.

Sample-rate policy: **48 kHz / mono / float32 only.** Anything else
triggers a `FormatNegotiation::Suggest` reply, and the Windows Audio
Engine then inserts its built-in SRC at the graph edge — same "graph
end resamples, no added latency" trick the LADSPA build relies on
PipeWire for. No external resampler / Voicemeeter / VB-Cable is
needed.

## Build

You'll need either a Windows machine with the
[Build Tools for Visual Studio](https://visualstudio.microsoft.com/downloads/#build-tools-for-visual-studio-2022)
(MSVC toolchain) **or** any host with `x86_64-pc-windows-gnu` and
the `mingw-w64` cross toolchain installed.

### Native (MSVC)

```pwsh
cd rust
cargo build -p mellonella-apo --release
# → rust\target\release\mellonella_apo.dll
```

### Cross from Linux/macOS (mingw-w64)

```sh
rustup target add x86_64-pc-windows-gnu
sudo apt-get install mingw-w64        # or: brew install mingw-w64
cd rust
cargo build -p mellonella-apo --release --target x86_64-pc-windows-gnu
# → rust/target/x86_64-pc-windows-gnu/release/mellonella_apo.dll
```

The artefact is ~4 MB. It dynamically loads ONNX Runtime via `ort`'s
`load-dynamic` feature, so the system also needs `onnxruntime.dll`
reachable on the loader path (same dependency `mellonella-gui` has;
the simplest answer is to drop it in the same directory as
`mellonella_apo.dll`).

## Install (per-endpoint capture stream effect)

> Heads up: editing the Windows audio APO registry is a fiddly,
> reversible-but-reboot-y workflow. Test on a non-critical machine
> first, and keep a screenshot of the registry keys you change so you
> can roll back.

1. Copy `mellonella_apo.dll` and `onnxruntime.dll` somewhere stable,
   e.g. `C:\Program Files\Mellonella\`.

2. Register the COM class as an Administrator:

   ```pwsh
   regsvr32 "C:\Program Files\Mellonella\mellonella_apo.dll"
   ```

   This calls `DllRegisterServer`, populating
   `HKLM\SOFTWARE\Classes\CLSID\{A1B2C3D4-E5F6-4789-8A1B-2C3D4E5F6071}`
   under the Mellonella CLSID with the in-proc server pointer.

3. Attach it to the desired capture endpoint. Open the registry
   editor and find your microphone's endpoint key under
   `HKLM\SOFTWARE\Microsoft\Windows\CurrentVersion\MMDevices\Audio\Capture\`.
   Inside it, navigate to
   `FxProperties\{D04E05A6-594B-4FB6-A80D-01AF5EED7D1D},5` — the
   `PKEY_FX_StreamEffectClsidList` property — and add the Mellonella
   CLSID `{A1B2C3D4-E5F6-4789-8A1B-2C3D4E5F6071}` to its REG_SZ list
   (or create the value if it doesn't exist).

4. Restart the Windows Audio service:

   ```pwsh
   Restart-Service Audiosrv -Force
   ```

5. Open the device's properties → Advanced → make sure **Default
   Format** is set to `1 channel, 16-bit, 48000 Hz` (or higher) — if
   the device's preferred mix format isn't 48 kHz, the engine
   inserts its built-in SRC ahead of our APO transparently, which is
   fine but you might as well save a step.

Verify it loaded by opening Sound Settings → microphone → checking
that the level meter still moves. Any APO crash on `lock_for_process`
will leave the endpoint silent until you un-set the FX list.

## Uninstall

```pwsh
# 1. Remove the CLSID from the endpoint's PKEY_FX_StreamEffectClsidList
#    entry (manual registry edit).
# 2. Unregister the COM class:
regsvr32 /u "C:\Program Files\Mellonella\mellonella_apo.dll"
# 3. Delete the DLL.
Restart-Service Audiosrv -Force
```

## Configuration

LADSPA-equivalent out-of-band channels, identical defaults:

| Source | Override env var | Default |
|---|---|---|
| Enrollment JSON | `MELLONELLA_ENROLLMENT` | `%APPDATA%\mellonella\enrollment.json` (where `mellonella-gui` auto-saves) |
| DFN3 ONNX | `MELLONELLA_DFN3_ONNX` | `mellonella` cache under `%LOCALAPPDATA%` |
| VAD ONNX | (cache only) | same cache |
| ECAPA ONNX | (cache only) | same cache |

If no enrollment is found, the plugin runs in **DFN3-only +
bootstrap-auto-learn** mode: the speaker gate stays fully open, but
the streaming engine still observes refreshes and seeds the first
anchor from the first second of voiced speech it hears. Subsequent
admissions accumulate into the adapted embedding, and on
`unlock_for_process` the pool is persisted back to the enrollment
path — so the next time the audio engine reopens the stream the APO
loads the learned profile and runs the full gate. Existing
enrollment files are never overwritten by this path (persistence
only fires when the session started with an empty pool).

If you want immediate gating, enroll explicitly through
`mellonella-gui` before registering the APO.

## Troubleshooting

* `lock_for_process` errors in the Windows Event Log
  (Applications and Services → Microsoft → Windows → Audio) usually
  mean either `onnxruntime.dll` isn't reachable or the cached models
  haven't been downloaded yet. Run `mellonella-gui` once to seed the
  cache (the APO and the GUI share the same `hf_fetch` cache
  directory).
* Microphone goes completely silent after registering the APO →
  unregister with `regsvr32 /u`, restart `Audiosrv`, and inspect the
  Event Log for the failed `lock_for_process` reason.
* Default Format is locked to 16 kHz on a USB headset → leave it as
  is; Windows inserts its built-in SRC before the APO and the result
  is still 48 kHz at our input.

## Limitations vs LADSPA

| Aspect | LADSPA | APO |
|---|---|---|
| Control ports | Bypass / Dry-Wet / VAD threshold runtime knobs | None — APO has no equivalent of LADSPA control ports for `Sfx` |
| Bypass | live, via control port | toggle the device-level "Enhancements" checkbox in Sound settings (or unregister) |
| Per-stream tuning | live | requires re-registration with different defaults |

A control surface for the APO (PropertyStore-backed live tuning) is a
plausible follow-up but out of scope for the initial build.
