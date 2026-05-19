//! Rust↔Python ONNX parity for the **prod_48k** TSE model.
//!
//! Mirrors `tse_parity.rs` (the poc_16k variant) but uses the 48 kHz
//! production config (`enc_stride=48`, chunk=480) and reuses a
//! pre-exported ONNX at `build/tse_prod_48k.onnx` if present.
//!
//! Skip conditions (same as `tse_parity.rs`):
//! * `ORT_DYLIB_PATH` unset or missing — the cargo CI lane doesn't ship
//!   onnxruntime.
//! * `build/tse_prod_48k.onnx` and `.weights.pt` sidecar missing AND
//!   no Python torch stack available — we can't generate the fixture.

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

const CHUNK: usize = 480;
const N_CHUNKS: usize = 8;
const TOTAL: usize = CHUNK * N_CHUNKS;
const TOL: f32 = 1e-4;

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

fn lcg_clip(n: usize, seed: u64) -> Vec<f32> {
    let mut state: u64 = seed;
    let mut out = Vec::with_capacity(n);
    for _ in 0..n {
        state = state.wrapping_mul(1_103_515_245).wrapping_add(12_345);
        let bits = (state >> 16) as u32 & 0x7FFF_FFFF;
        let val = (bits as f32 / 0x7FFF_FFFF as f32) - 0.5;
        out.push(val);
    }
    out
}

fn read_f32_buffer(bytes: &[u8]) -> Vec<f32> {
    bytes
        .chunks_exact(4)
        .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect()
}

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
fn tse_session_matches_python_onnxruntime_prod_48k() {
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

    let onnx_path = build.join("tse_prod_48k.onnx");
    let weights_path = build.join("tse_prod_48k.onnx.weights.pt");
    let clip_path = build.join("tse_prod_48k_input.bin");
    let cond_path = build.join("tse_prod_48k_cond.bin");
    let expected_path = build.join("tse_prod_48k_expected.bin");

    if !onnx_path.exists() || !weights_path.exists() {
        eprintln!(
            "[skip] missing pre-built prod_48k fixtures at {} (run scripts/export_tse_onnx.py \
             export-and-verify --config prod_48k --chunk 480 --output build/tse_prod_48k.onnx \
             --checkpoint <ckpt> first)",
            onnx_path.display(),
        );
        return;
    }

    let clip = lcg_clip(TOTAL, 0x00C0_FFEE);
    let cond = lcg_clip(TSE_COND_DIM, 0xCAFE_F00D);
    write_f32_file(&clip_path, &clip);
    write_f32_file(&cond_path, &cond);

    let chunk_arg = CHUNK.to_string();
    let onnx_arg = onnx_path.to_string_lossy().into_owned();
    let clip_arg = clip_path.to_string_lossy().into_owned();
    let cond_arg = cond_path.to_string_lossy().into_owned();
    let expected_arg = expected_path.to_string_lossy().into_owned();
    match run_python(
        &[
            "dump-fixture",
            "--config",
            "prod_48k",
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
    ) {
        Ok(()) => {}
        Err(msg) => {
            eprintln!("[skip] cannot dump TSE prod_48k fixture: {msg}");
            return;
        }
    }

    let expected_bytes = std::fs::read(&expected_path).expect("read expected fixture");
    let expected = read_f32_buffer(&expected_bytes);
    assert_eq!(
        expected.len(),
        TOTAL,
        "expected fixture length: got {}, want {TOTAL}",
        expected.len(),
    );

    let cfg = TseConfig::prod_48k();
    assert_eq!(cfg.n_state_tensors(), 89);

    let mut session = TseSession::from_onnx_path_with_config(&onnx_path, cfg)
        .expect("load prod_48k TSE ONNX");
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
        "TSE prod_48k per-chunk parity: max|Δ|={max_per_chunk:.3e} > tol={TOL:.3e}",
    );
}
