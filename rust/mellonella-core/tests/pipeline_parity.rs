//! End-to-end parity check between [`mellonella_core::pipeline::process_offline`]
//! and the Python reference defined in `scripts/dump_pipeline_fixture.py`.
//!
//! The reference runs a deliberately Rust-equivalent path in Python
//! (no DFN3, no resample, raw `cos_sim_max` instead of `α·cs + β·f0`)
//! so the per-frame outputs should match byte-for-byte up to f32
//! rounding.
//!
//! Gated on `MELLONELLA_ECAPA_ONNX`, `MELLONELLA_VAD_ONNX`, and
//! `ORT_DYLIB_PATH`. Without those, the test prints a skip notice and
//! passes so CI stays green without vendoring the ONNX files.

#![allow(
    clippy::cast_precision_loss,
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss
)]

use mellonella_core::embedding::EcapaTdnn;
use mellonella_core::enrollment::{EmbeddingPool, EmbeddingPoolConfig};
use mellonella_core::features::Fbank;
use mellonella_core::gating::GateConfig;
use mellonella_core::pipeline::{process_offline, PipelineComponents, PipelineConfig};
use mellonella_core::vad::SileroVad;

const FILTERBANK: &[u8] = include_bytes!("fixtures/fbank_filterbank.bin");
const INPUT: &[u8] = include_bytes!("fixtures/pipeline_input.bin");
const ANCHOR: &[u8] = include_bytes!("fixtures/pipeline_anchor.bin");
const EXPECTED_SCORE: &[u8] = include_bytes!("fixtures/pipeline_score_per_frame.bin");
const EXPECTED_GATE: &[u8] = include_bytes!("fixtures/pipeline_gate_per_frame.bin");

// Score parity: ECAPA path accumulates ~6e-4 raw embedding delta, which
// after cos_sim_max can move the score by a few thousandths. 5e-3 is
// well under the gate threshold's tolerance (0.30 ± 0.005 doesn't flip
// decisions in this fixture — see the test assertion that gate bytes
// match exactly).
const SCORE_TOL: f32 = 5e-3;

fn read_f32_buffer(bytes: &[u8]) -> Vec<f32> {
    bytes
        .chunks_exact(4)
        .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect()
}

#[test]
fn process_offline_matches_python_reference() {
    let Ok(ecapa_path) = std::env::var("MELLONELLA_ECAPA_ONNX") else {
        eprintln!("[skip] MELLONELLA_ECAPA_ONNX not set");
        return;
    };
    let Ok(vad_path) = std::env::var("MELLONELLA_VAD_ONNX") else {
        eprintln!("[skip] MELLONELLA_VAD_ONNX not set");
        return;
    };
    if !std::path::Path::new(&ecapa_path).exists() || !std::path::Path::new(&vad_path).exists() {
        eprintln!("[skip] ONNX file(s) missing");
        return;
    }

    let audio = read_f32_buffer(INPUT);
    let anchor = read_f32_buffer(ANCHOR);
    let expected_score = read_f32_buffer(EXPECTED_SCORE);
    let expected_gate: Vec<bool> = EXPECTED_GATE.iter().map(|&b| b != 0).collect();
    assert_eq!(anchor.len(), 192);
    assert_eq!(expected_score.len(), expected_gate.len());

    let fb_matrix = read_f32_buffer(FILTERBANK);
    let fbank = Fbank::new(&fb_matrix).expect("Fbank from fixture");
    let ecapa = EcapaTdnn::from_onnx_path(&ecapa_path).expect("ECAPA load");
    let vad = SileroVad::from_onnx_path(&vad_path, 16_000).expect("VAD load");

    let mut components = PipelineComponents {
        vad,
        fbank,
        ecapa,
        cohort: Vec::new(),
    };

    let mut pool = EmbeddingPool::new(EmbeddingPoolConfig::default());
    pool.add_anchors([anchor]);

    let pipeline_cfg = PipelineConfig {
        vad_threshold: -1.0, // mirrors the dump script's always-accept
        // The fixture was generated against sv_update_samples = 4 000
        // (250 ms refresh cadence). Pin the test to that value even
        // though the live default is now 8 000 (Phase 3.5 step 3) —
        // otherwise the Rust run would refresh on a different schedule
        // than the Python reference and the per-frame scores diverge.
        sv_update_samples: 4_000,
        enable_auto_learn: false,
        // The dump script has no EMA smoothing; the live default is now
        // 0.6 (#117). Pin to 1.0 so the per-frame scores stay byte-equal
        // with the reference. (The #117 centroid scoring needs no pin —
        // the fixture pool has a single anchor, whose centroid is that
        // anchor verbatim, so `match_score` reduces to the old
        // `cos_sim_max` exactly.)
        score_ema_alpha: 1.0,
        ..PipelineConfig::default()
    };
    let gate_cfg = GateConfig::default();
    let result = process_offline(
        &audio,
        16_000,
        &mut pool,
        &pipeline_cfg,
        &gate_cfg,
        &mut components,
    )
    .expect("pipeline runs");

    assert_eq!(
        result.score_per_frame.len(),
        expected_score.len(),
        "frame count mismatch: rust={} python={}",
        result.score_per_frame.len(),
        expected_score.len()
    );

    // Per-frame score parity.
    let mut max_score_delta = 0.0_f32;
    let mut argmax_score = 0_usize;
    for (i, (&a, &b)) in result
        .score_per_frame
        .iter()
        .zip(expected_score.iter())
        .enumerate()
    {
        let delta = (a - b).abs();
        if delta > max_score_delta {
            max_score_delta = delta;
            argmax_score = i;
        }
    }
    assert!(
        max_score_delta <= SCORE_TOL,
        "score parity: max|Δ|={max_score_delta:.3e} at frame={argmax_score} (rust={}, python={}) exceeds tol={SCORE_TOL:.3e}",
        result.score_per_frame[argmax_score],
        expected_score[argmax_score],
    );

    // Per-frame gate-state parity — must be byte-equal.
    for (i, (&a, &b)) in result
        .gate_per_frame
        .iter()
        .zip(expected_gate.iter())
        .enumerate()
    {
        assert_eq!(
            a, b,
            "gate state diverges at frame={i}: rust={a} python={b} (score rust={}, python={})",
            result.score_per_frame[i], expected_score[i],
        );
    }
}
