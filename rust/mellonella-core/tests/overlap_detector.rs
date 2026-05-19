//! Smoke + behavioural test for [`mellonella_core::overlap::OverlapDetector`].
//!
//! Gated on `MELLONELLA_OVERLAP_SEG_ONNX` and `ORT_DYLIB_PATH`.
//! Optionally consumes a developer-supplied WAV via
//! `MELLONELLA_OVERLAP_TEST_SOLO_WAV` and
//! `MELLONELLA_OVERLAP_TEST_MIX_WAV` to spot-check on real audio.
//! When the optional WAVs aren't set, the test synthesises a clean
//! sine on a 1-s buffer (which should score `mean_overlap_prob ≈ 0`)
//! so the detector path is exercised even in a minimal CI lane.

#![allow(
    clippy::cast_precision_loss,
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss
)]

use std::path::PathBuf;

use mellonella_core::overlap::{OverlapDetector, OVERLAP_WINDOW_SAMPLES};
use mellonella_core::resample::resample_to;

fn need_onnx() -> Option<PathBuf> {
    if std::env::var("ORT_DYLIB_PATH").is_err() {
        eprintln!("[skip] ORT_DYLIB_PATH not set");
        return None;
    }
    let path = std::env::var_os("MELLONELLA_OVERLAP_SEG_ONNX").map(PathBuf::from)?;
    if !path.exists() {
        eprintln!(
            "[skip] MELLONELLA_OVERLAP_SEG_ONNX → {} not found",
            path.display()
        );
        return None;
    }
    Some(path)
}

fn read_pcm16_mono_wav(path: &str) -> Option<(Vec<f32>, u32)> {
    let bytes = std::fs::read(path).ok()?;
    if bytes.len() <= 44 {
        return None;
    }
    let mut i = 12_usize;
    let mut fmt_off: Option<usize> = None;
    let mut data_off: Option<(usize, usize)> = None;
    while i + 8 <= bytes.len() {
        let sz =
            u32::from_le_bytes([bytes[i + 4], bytes[i + 5], bytes[i + 6], bytes[i + 7]]) as usize;
        match &bytes[i..i + 4] {
            b"fmt " => fmt_off = Some(i + 8),
            b"data" => data_off = Some((i + 8, sz)),
            _ => {}
        }
        i += 8 + sz;
    }
    let fmt = fmt_off?;
    let (data, dlen) = data_off?;
    let sr = u32::from_le_bytes([
        bytes[fmt + 4],
        bytes[fmt + 5],
        bytes[fmt + 6],
        bytes[fmt + 7],
    ]);
    let scale = 1.0_f32 / f32::from(i16::MAX);
    let samples: Vec<f32> = bytes[data..data + dlen]
        .chunks_exact(2)
        .map(|c| f32::from(i16::from_le_bytes([c[0], c[1]])) * scale)
        .collect();
    Some((samples, sr))
}

fn to_16k(samples: Vec<f32>, sr: u32) -> Vec<f32> {
    if sr == 16_000 {
        samples
    } else {
        resample_to(&samples, sr, 16_000).expect("resample to 16k")
    }
}

#[test]
fn detector_runs_and_returns_a_decision_after_one_second_of_audio() {
    let Some(onnx) = need_onnx() else {
        return;
    };
    let mut det = OverlapDetector::from_onnx_path(&onnx).expect("load overlap seg");

    // First push: buffer still filling, no decision yet.
    let half_window = vec![0.01_f32; OVERLAP_WINDOW_SAMPLES / 2];
    assert!(
        det.push(&half_window).expect("push half").is_none(),
        "no decision while the buffer is half-full"
    );

    // Second push completes the buffer; the cadence counter has
    // also accumulated past the default 4 000-sample cadence so the
    // detector fires on this call.
    let decision = det
        .push(&half_window)
        .expect("push other half")
        .expect("first inference fires once the buffer fills");
    eprintln!(
        "[smoke] constant-DC 1 s: mean={:.3} max={:.3} frames {}/{} overlap",
        decision.mean_overlap_prob,
        decision.max_overlap_prob,
        decision.overlap_frame_count,
        decision.total_frames
    );
    // DC signal isn't speech — model should not see overlap.
    assert!(
        decision.mean_overlap_prob < 0.1,
        "DC signal should not classify as overlap (got {:.3})",
        decision.mean_overlap_prob
    );
}

#[test]
fn detector_separates_solo_voice_from_two_speaker_overlap() {
    let Some(onnx) = need_onnx() else {
        return;
    };
    let solo = std::env::var("MELLONELLA_OVERLAP_TEST_SOLO_WAV").ok();
    let mix = std::env::var("MELLONELLA_OVERLAP_TEST_MIX_WAV").ok();
    let (Some(solo), Some(mix)) = (solo, mix) else {
        eprintln!(
            "[skip] both MELLONELLA_OVERLAP_TEST_SOLO_WAV and \
             MELLONELLA_OVERLAP_TEST_MIX_WAV must be set"
        );
        return;
    };
    let solo = read_pcm16_mono_wav(&solo).expect("solo WAV");
    let mix = read_pcm16_mono_wav(&mix).expect("mix WAV");

    let solo_16k = to_16k(solo.0, solo.1);
    let mix_16k = to_16k(mix.0, mix.1);

    let run_once = |audio_16k: &[f32]| {
        let mut det = OverlapDetector::from_onnx_path(&onnx).expect("load");
        // Feed in 1-second windows, force a run on the first full window.
        let _ = det.push(audio_16k).expect("push");
        det.force_run()
            .expect("force_run")
            .expect("decision after enough samples")
    };

    let s = run_once(&solo_16k);
    let m = run_once(&mix_16k);
    eprintln!(
        "[real ] solo  : mean={:.3} max={:.3} overlap-frames={}/{}",
        s.mean_overlap_prob, s.max_overlap_prob, s.overlap_frame_count, s.total_frames
    );
    eprintln!(
        "[real ] mix   : mean={:.3} max={:.3} overlap-frames={}/{}",
        m.mean_overlap_prob, m.max_overlap_prob, m.overlap_frame_count, m.total_frames
    );

    // Empirically: solo ≲ 0.05, mix ≳ 0.30. Use 0.1 / 0.2 as a
    // generous bracket so the test doesn't pin to a single fixture.
    assert!(
        s.mean_overlap_prob < 0.1,
        "solo voice should not classify as overlap (got {:.3})",
        s.mean_overlap_prob
    );
    assert!(
        m.mean_overlap_prob > 0.2,
        "two-speaker mix should classify as overlap (got {:.3})",
        m.mean_overlap_prob
    );
}
