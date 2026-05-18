//! End-to-end parity check between [`mellonella_core::tse::TseSession`]
//! and the Python ONNX Runtime reference defined in
//! `scripts/export_tse_onnx.py dump-fixture`.
//!
//! The test:
//!
//! 1. Builds the streaming TSE ONNX (`build/tse_smoke.onnx`) by
//!    invoking `python scripts/export_tse_onnx.py export ...` if the
//!    fixture isn't already on disk. The export also drops a
//!    `*.weights.pt` sidecar that the dump step uses.
//! 2. Generates a deterministic test clip (8 × 160-sample chunks =
//!    1280 samples) and a deterministic 192-dim cond embedding,
//!    writing both as little-endian `float32` to `build/`.
//! 3. Invokes `python scripts/export_tse_onnx.py dump-fixture ...` to
//!    thread the same clip + cond through the ONNX model in Python.
//!    Result lands as a flat `float32` binary in `build/`.
//! 4. Loads the ONNX into [`TseSession`] from Rust, processes the
//!    same chunks, asserts per-chunk `max|Δ|` ≤ 1e-4 vs Python.
//!
//! The test prints a clear skip message and passes if Python+torch+
//! onnxruntime aren't available (so the cargo CI lane doesn't have to
//! provision them). The `test (training)` lane already has the full
//! Python stack.
//!
//! `ORT_DYLIB_PATH` must point at a libonnxruntime the Rust ort crate
//! can load; the test skips with a notice if it isn't set or the file
//! is missing.

#![allow(
    clippy::cast_precision_loss,
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss,
    clippy::too_many_lines,
    clippy::similar_names
)]

use std::path::{Path, PathBuf};
use std::process::Command;

use mellonella_core::tse::{TseConfig, TseSession, TSE_COND_DIM};

const CHUNK: usize = 160;
const N_CHUNKS: usize = 8;
const TOTAL: usize = CHUNK * N_CHUNKS;
const TOL: f32 = 1e-4;

/// Locate the repo root by walking up from `CARGO_MANIFEST_DIR` until
/// we find `scripts/export_tse_onnx.py`. The test is run with
/// `CARGO_MANIFEST_DIR == rust/mellonella-core`, so the repo root sits
/// two levels up.
fn repo_root() -> PathBuf {
    let mut here = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    for _ in 0..4 {
        if here.join("scripts/export_tse_onnx.py").exists() {
            return here;
        }
        if !here.pop() {
            break;
        }
    }
    panic!(
        "cannot locate repo root from CARGO_MANIFEST_DIR={CARGO_MANIFEST_DIR}",
        CARGO_MANIFEST_DIR = env!("CARGO_MANIFEST_DIR")
    );
}

/// Run a `python scripts/export_tse_onnx.py ...` subcommand. Returns
/// `Ok(())` on success, `Err(skip_message)` if Python isn't available
/// (so the test can skip cleanly rather than fail in environments
/// without the training Python stack).
fn run_python(args: &[&str], root: &Path) -> Result<(), String> {
    let python = std::env::var("PYTHON").unwrap_or_else(|_| "python3".to_string());
    let script = root.join("scripts").join("export_tse_onnx.py");
    let output = Command::new(&python)
        .arg(&script)
        .args(args)
        .current_dir(root)
        .output();
    let output = match output {
        Ok(o) => o,
        Err(e) => return Err(format!("failed to spawn '{python}': {e}")),
    };
    if !output.status.success() {
        // Check stderr for missing-import patterns and fold them into a
        // skip message; otherwise treat it as a hard failure.
        let stderr = String::from_utf8_lossy(&output.stderr);
        if stderr.contains("ModuleNotFoundError") || stderr.contains("No module named") {
            return Err(format!(
                "python missing required module — {python} stderr: {stderr}"
            ));
        }
        let script_display = script.display();
        panic!(
            "python {script_display} {args:?} failed:\n--- stderr ---\n{stderr}\n--- stdout ---\n{}",
            String::from_utf8_lossy(&output.stdout),
        );
    }
    Ok(())
}

/// Deterministic LCG so the test is hermetic without a `rand`
/// dev-dependency. Same constants as `glibc`'s `rand`. Returns values
/// uniformly in `[-1.0, 1.0)` after scaling.
fn lcg_clip(n: usize, seed: u64) -> Vec<f32> {
    let mut state: u64 = seed;
    let mut out = Vec::with_capacity(n);
    for _ in 0..n {
        // 32-bit Lehmer-style step.
        state = state.wrapping_mul(1_103_515_245).wrapping_add(12_345);
        // Take the high bits to get a roughly-uniform 32-bit number.
        let bits = (state >> 16) as u32 & 0x7FFF_FFFF;
        // Scale to [-0.5, 0.5) — staying well inside the model's
        // expected dynamic range; the parity bound is independent of
        // signal scale but small inputs keep the ONNX accumulator
        // happy.
        let val = (bits as f32 / 0x7FFF_FFFF as f32) - 0.5;
        out.push(val);
    }
    out
}

/// Read a binary blob as little-endian `f32`.
fn read_f32_buffer(bytes: &[u8]) -> Vec<f32> {
    bytes
        .chunks_exact(4)
        .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect()
}

/// Write a slice of `f32` to `path` as little-endian raw bytes.
fn write_f32_file(path: &Path, data: &[f32]) {
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent).expect("create_dir_all build/");
    }
    let mut bytes = Vec::with_capacity(data.len() * 4);
    for &v in data {
        bytes.extend_from_slice(&v.to_le_bytes());
    }
    std::fs::write(path, &bytes).expect("write fixture file");
}

#[test]
fn tse_session_matches_python_onnxruntime() {
    // Allow the test to short-circuit when ORT can't be dynamically
    // loaded — the cargo CI lane doesn't necessarily set this up.
    let Ok(dylib) = std::env::var("ORT_DYLIB_PATH") else {
        eprintln!("[skip] ORT_DYLIB_PATH not set");
        return;
    };
    if !Path::new(&dylib).exists() {
        eprintln!("[skip] ORT_DYLIB_PATH={dylib} does not exist");
        return;
    }

    let root = repo_root();
    let build = root.join("build");

    let onnx_path = build.join("tse_smoke.onnx");
    let weights_path = build.join("tse_smoke.onnx.weights.pt");
    let clip_path = build.join("tse_smoke_input.bin");
    let cond_path = build.join("tse_smoke_cond.bin");
    let expected_path = build.join("tse_smoke_expected.bin");

    // 1) Export the ONNX (+ weights sidecar) if missing. Skip with a
    //    clear notice if Python/torch isn't available.
    if !onnx_path.exists() || !weights_path.exists() {
        let chunk_arg = CHUNK.to_string();
        let out_arg = onnx_path.to_string_lossy().into_owned();
        match run_python(
            &[
                "export", "--config", "poc_16k", "--chunk", &chunk_arg, "--output", &out_arg,
            ],
            &root,
        ) {
            Ok(()) => {}
            Err(msg) => {
                eprintln!("[skip] cannot export TSE ONNX: {msg}");
                return;
            }
        }
    }

    // 2) Deterministic test clip + cond. The cond is also LCG-derived
    //    (different seed) so non-zero FiLM gammas/betas exercise every
    //    block.
    let clip = lcg_clip(TOTAL, 0x00C0_FFEE);
    let cond = lcg_clip(TSE_COND_DIM, 0xCAFE_F00D);
    write_f32_file(&clip_path, &clip);
    write_f32_file(&cond_path, &cond);

    // 3) Have Python ONNX Runtime produce the expected output. If the
    //    Python stack is broken we surface the failure (we already
    //    succeeded in exporting above, so any breakage from here is a
    //    real bug).
    let chunk_arg = CHUNK.to_string();
    let onnx_arg = onnx_path.to_string_lossy().into_owned();
    let clip_arg = clip_path.to_string_lossy().into_owned();
    let cond_arg = cond_path.to_string_lossy().into_owned();
    let expected_arg = expected_path.to_string_lossy().into_owned();
    run_python(
        &[
            "dump-fixture",
            "--config",
            "poc_16k",
            "--chunk",
            &chunk_arg,
            "--onnx",
            &onnx_arg,
            "--clip",
            &clip_arg,
            "--cond",
            &cond_arg,
            "--output",
            &expected_arg,
        ],
        &root,
    )
    .expect("python dump-fixture");

    let expected_bytes = std::fs::read(&expected_path).expect("read expected fixture");
    let expected = read_f32_buffer(&expected_bytes);
    assert_eq!(
        expected.len(),
        TOTAL,
        "expected fixture length: got {}, want {TOTAL}",
        expected.len(),
    );

    // 4) Rust side: load the ONNX, process the same chunks, compare.
    let cfg = TseConfig::poc_16k();
    assert_eq!(cfg.n_state_tensors(), 89);

    let mut session = TseSession::from_onnx_path(&onnx_path).expect("load TSE ONNX");
    let mut cond_arr = [0.0_f32; TSE_COND_DIM];
    cond_arr.copy_from_slice(&cond);

    let mut max_per_chunk: f32 = 0.0;
    let mut rust_concat = Vec::with_capacity(TOTAL);
    for i in 0..N_CHUNKS {
        let chunk = &clip[i * CHUNK..(i + 1) * CHUNK];
        let out = session
            .process_chunk(chunk, &cond_arr)
            .expect("process_chunk");
        assert_eq!(out.len(), CHUNK, "chunk {i} length");
        let expected_chunk = &expected[i * CHUNK..(i + 1) * CHUNK];
        let mut chunk_max = 0.0_f32;
        for (a, b) in out.iter().zip(expected_chunk) {
            let d = (a - b).abs();
            if d > chunk_max {
                chunk_max = d;
            }
        }
        eprintln!("[parity] chunk {i:2} max|Δ| = {chunk_max:.3e}");
        if chunk_max > max_per_chunk {
            max_per_chunk = chunk_max;
        }
        rust_concat.extend_from_slice(&out);
    }

    let overall = rust_concat
        .iter()
        .zip(&expected)
        .map(|(a, b)| (a - b).abs())
        .fold(0.0_f32, f32::max);
    eprintln!("[parity] per-chunk max|Δ| = {max_per_chunk:.3e}");
    eprintln!("[parity] overall  max|Δ|  = {overall:.3e}");
    eprintln!("[parity] tolerance        = {TOL:.3e}");

    assert!(
        max_per_chunk <= TOL,
        "TSE per-chunk parity: max|Δ|={max_per_chunk:.3e} > tol={TOL:.3e}",
    );
}
