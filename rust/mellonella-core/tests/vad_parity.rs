//! Parity check between [`mellonella_core::vad::SileroVad`] and the
//! Python silero-vad ONNX wrapper for a deterministic synthesised
//! speech-like waveform.
//!
//! Gated on `MELLONELLA_VAD_ONNX` (path to `silero_vad.onnx`) and
//! `ORT_DYLIB_PATH` (libonnxruntime). Without those, the test prints
//! a skip notice and passes — keeps CI green without vendoring the
//! 2.3 MB ONNX file.

use mellonella_core::vad::{SileroVad, CHUNK_SAMPLES_16K};

const INPUT: &[u8] = include_bytes!("fixtures/vad_input.bin");
const EXPECTED: &[u8] = include_bytes!("fixtures/vad_expected.bin");

const TOL: f32 = 1e-3;

fn read_f32_buffer(bytes: &[u8]) -> Vec<f32> {
    bytes
        .chunks_exact(4)
        .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect()
}

#[test]
fn vad_matches_silero_fixture() {
    let Ok(path) = std::env::var("MELLONELLA_VAD_ONNX") else {
        eprintln!("[skip] MELLONELLA_VAD_ONNX not set");
        return;
    };
    if !std::path::Path::new(&path).exists() {
        eprintln!("[skip] MELLONELLA_VAD_ONNX={path} does not exist");
        return;
    }

    let audio = read_f32_buffer(INPUT);
    let expected = read_f32_buffer(EXPECTED);
    let n_chunks = audio.len() / CHUNK_SAMPLES_16K;
    assert_eq!(
        n_chunks,
        expected.len(),
        "chunk count mismatch: {} vs {}",
        n_chunks,
        expected.len()
    );

    let mut vad = SileroVad::from_onnx_path(&path, 16_000).expect("load VAD ONNX");
    let mut max_delta = 0.0_f32;
    let mut argmax = 0_usize;
    for i in 0..n_chunks {
        let chunk = &audio[i * CHUNK_SAMPLES_16K..(i + 1) * CHUNK_SAMPLES_16K];
        let actual = vad.score(chunk).expect("VAD inference");
        let delta = (actual - expected[i]).abs();
        if delta > max_delta {
            max_delta = delta;
            argmax = i;
        }
    }
    assert!(
        max_delta <= TOL,
        "max|Δ|={max_delta:.3e} at chunk={argmax} exceeds tol={TOL:.3e}"
    );
}
