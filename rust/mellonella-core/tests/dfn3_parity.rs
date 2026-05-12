//! End-to-end parity check between [`mellonella_core::dfn3::Dfn3Pipeline`]
//! and the Python DFN3 reference defined in
//! `scripts/dump_dfn3_fixture.py`.
//!
//! Gated on `MELLONELLA_DFN3_ONNX` (path to the patched DFN3 ONNX)
//! and `ORT_DYLIB_PATH` (libonnxruntime).

#![allow(clippy::cast_precision_loss)]

use mellonella_core::dfn3::{Dfn3Pipeline, SAMPLES_PER_CHUNK};

const INPUT: &[u8] = include_bytes!("fixtures/dfn3_input.bin");
const EXPECTED: &[u8] = include_bytes!("fixtures/dfn3_expected_audio.bin");

// Per-sample tolerance for the audio waveform. STFT round-trip + ONNX
// f32 accumulation pushes the per-sample delta into the mid-1e-2 range
// at peak; 5e-2 is the worst case empirically observed, comfortably
// inaudible.
const TOL: f32 = 5e-2;

fn read_f32_buffer(bytes: &[u8]) -> Vec<f32> {
    bytes
        .chunks_exact(4)
        .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect()
}

#[test]
fn dfn3_pipeline_matches_python_reference() {
    let Ok(onnx_path) = std::env::var("MELLONELLA_DFN3_ONNX") else {
        eprintln!("[skip] MELLONELLA_DFN3_ONNX not set");
        return;
    };
    if !std::path::Path::new(&onnx_path).exists() {
        eprintln!("[skip] MELLONELLA_DFN3_ONNX={onnx_path} does not exist");
        return;
    }

    let audio_in = read_f32_buffer(INPUT);
    let expected = read_f32_buffer(EXPECTED);
    assert_eq!(audio_in.len(), SAMPLES_PER_CHUNK, "input length");
    assert_eq!(expected.len(), SAMPLES_PER_CHUNK, "expected length");

    let mut pipeline = Dfn3Pipeline::from_onnx_path(&onnx_path).expect("DFN3 pipeline load");
    let actual = pipeline.process(&audio_in).expect("DFN3 process");
    assert_eq!(actual.len(), SAMPLES_PER_CHUNK, "output length");

    let mut max_delta = 0.0_f32;
    let mut argmax = 0_usize;
    for (i, (&a, &b)) in actual.iter().zip(expected.iter()).enumerate() {
        let d = (a - b).abs();
        if d > max_delta {
            max_delta = d;
            argmax = i;
        }
    }
    assert!(
        max_delta <= TOL,
        "audio parity: max|Δ|={max_delta:.3e} at sample={argmax} (rust={}, python={}) exceeds tol={TOL:.3e}",
        actual[argmax],
        expected[argmax],
    );
}
