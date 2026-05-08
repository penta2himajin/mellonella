//! Byte-level parity check between [`mellonella_core::features::Fbank`]
//! and SpeechBrain's `Fbank` for the ECAPA-TDNN preset.
//!
//! Fixtures are produced by ``scripts/dump_fbank_fixture.py``.

use mellonella_core::features::{Fbank, N_MELS, N_STFT};

const FILTERBANK: &[u8] = include_bytes!("fixtures/fbank_filterbank.bin");
const INPUT: &[u8] = include_bytes!("fixtures/fbank_input.bin");
const EXPECTED: &[u8] = include_bytes!("fixtures/fbank_expected.bin");

// Single-precision FFT + matmul + log10 chains accumulate enough rounding
// noise that the per-bin delta can exceed 1e-3 at high mels. 5e-3 dB is
// well under any perceptual threshold (~0.005 dB) so it's a comfortable
// bound for byte-equivalent purposes.
const TOL: f32 = 5e-3;

fn read_f32_buffer(bytes: &[u8]) -> Vec<f32> {
    assert!(
        bytes.len() % 4 == 0,
        "fixture byte length not a multiple of 4"
    );
    bytes
        .chunks_exact(4)
        .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect()
}

#[test]
fn fbank_matches_speechbrain_fixture() {
    let fb_matrix = read_f32_buffer(FILTERBANK);
    assert_eq!(fb_matrix.len(), N_STFT * N_MELS, "filterbank shape");

    let audio = read_f32_buffer(INPUT);
    let expected = read_f32_buffer(EXPECTED);
    assert_eq!(
        expected.len() % N_MELS,
        0,
        "expected output not aligned to N_MELS"
    );
    let n_frames_expected = expected.len() / N_MELS;

    let mut fbank = Fbank::new(&fb_matrix).expect("Fbank::new accepts the fixture");
    let actual = fbank.compute(&audio);
    let n_frames = actual.len() / N_MELS;

    assert_eq!(
        n_frames, n_frames_expected,
        "frame count mismatch: rust {n_frames} vs python {n_frames_expected}"
    );

    let mut max_delta = 0.0_f32;
    let mut argmax = (0_usize, 0_usize);
    for f in 0..n_frames {
        for m in 0..N_MELS {
            let i = f * N_MELS + m;
            let delta = (actual[i] - expected[i]).abs();
            if delta > max_delta {
                max_delta = delta;
                argmax = (f, m);
            }
        }
    }
    assert!(
        max_delta <= TOL,
        "max|Δ|={max_delta:.3e} at (frame={}, mel={}) exceeds tol={TOL:.3e}",
        argmax.0,
        argmax.1,
    );
}
