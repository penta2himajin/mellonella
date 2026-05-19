//! HuggingFace Hub model fetcher.
//!
//! Downloads model files from `https://huggingface.co/<repo>/resolve/main/<file>`
//! into a local cache directory and returns the on-disk path. Built so
//! both the CLI (`--tse-from-hf`) and the GUI ("Download from HuggingFace"
//! button) can pull the canonical Stage C model
//! ([`penta2himajin/tse-conv-tasnet-48k`](https://huggingface.co/penta2himajin/tse-conv-tasnet-48k))
//! without the user manually pointing at a file.
//!
//! Cache layout:
//!
//! ```text
//! $XDG_CACHE_HOME/mellonella/models/<owner>/<repo>/<file>
//! ```
//!
//! On Linux that's typically `~/.cache/mellonella/models/...`; on
//! macOS `~/Library/Caches/mellonella/models/...`; on Windows
//! `%LOCALAPPDATA%\mellonella\models\...`. The cache is shared
//! between mellonella's CLI and GUI binaries.
//!
//! The fetcher is intentionally minimal — synchronous, no retries
//! beyond what the underlying HTTP stack does, no partial-resume.
//! A failed download leaves nothing in cache; the next call retries
//! from scratch.

use std::ffi::OsStr;
use std::fs;
use std::io::{self, Read, Write};
use std::path::{Path, PathBuf};

fn has_onnx_extension(name: &str) -> bool {
    Path::new(name)
        .extension()
        .and_then(OsStr::to_str)
        .is_some_and(|ext| ext.eq_ignore_ascii_case("onnx"))
}

/// Canonical TSE production model repo on HuggingFace.
pub const TSE_PROD_48K_REPO: &str = "penta2himajin/tse-conv-tasnet-48k";

/// Files that together make up the TSE production export. The `.onnx`
/// references `.onnx.data` by relative path, so both must live side by
/// side in the cache directory.
pub const TSE_PROD_48K_FILES: &[&str] = &["tse_prod_48k.onnx", "tse_prod_48k.onnx.data"];

/// HuggingFace mirror of the Silero VAD ONNX. `onnx-community/silero-vad`
/// is a community-maintained mirror of the model snakers4 ships with the
/// `silero-vad` Python package; the ONNX file is byte-equal to what
/// `scripts/download_models.sh` would copy out of the Python package.
pub const VAD_REPO: &str = "onnx-community/silero-vad";
pub const VAD_FILE: &str = "onnx/model.onnx";

/// HuggingFace repo hosting the mellonella-side ONNX exports for DFN3
/// and ECAPA-TDNN. These models aren't natively distributed as ONNX
/// — DFN3 ships as a torch checkpoint via `df.enhance`, and ECAPA-TDNN
/// ships as a SpeechBrain checkpoint — so the user has to convert
/// once via `scripts/export_dfn3_onnx.py` / `scripts/export_ecapa_onnx.py`
/// and upload the result. The repo path here is the canonical home;
/// `ensure_dfn3_onnx` / `ensure_ecapa_onnx` fall back to env vars when
/// the HF copy isn't reachable yet.
pub const MELLONELLA_MODELS_REPO: &str = "penta2himajin/mellonella-models";
pub const DFN3_FILE: &str = "dfn3.onnx";
pub const ECAPA_FILE: &str = "ecapa_tdnn.onnx";

/// Errors produced by [`fetch_file`] / [`fetch_tse_prod_48k`].
#[derive(Debug)]
pub enum FetchError {
    /// Cache directory couldn't be determined or created.
    Cache(io::Error),
    /// Underlying HTTP transport failure. Boxed because
    /// `ureq::Error` is large (~272 B) and bloats the enum.
    Http(Box<ureq::Error>),
    /// File write failure during download.
    Io(io::Error),
    /// Server returned a non-200 response.
    Status { url: String, code: u16 },
}

impl std::fmt::Display for FetchError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Cache(e) => write!(f, "cache directory: {e}"),
            Self::Http(e) => write!(f, "HTTP transport: {e}"),
            Self::Io(e) => write!(f, "I/O: {e}"),
            Self::Status { url, code } => write!(f, "GET {url} returned HTTP {code}"),
        }
    }
}

impl std::error::Error for FetchError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Cache(e) | Self::Io(e) => Some(e),
            Self::Http(e) => Some(e),
            Self::Status { .. } => None,
        }
    }
}

impl From<ureq::Error> for FetchError {
    fn from(e: ureq::Error) -> Self {
        Self::Http(Box::new(e))
    }
}

/// Resolve the cache root used for HuggingFace artifacts:
/// `<dirs::cache_dir>/mellonella/models`. The directory is created on
/// first use.
///
/// # Errors
/// [`FetchError::Cache`] when `dirs::cache_dir()` returns `None` (no
/// platform cache directory configured) or when the directory can't be
/// created.
pub fn cache_root() -> Result<PathBuf, FetchError> {
    let base = dirs::cache_dir().ok_or_else(|| {
        FetchError::Cache(io::Error::other(
            "platform cache directory not configured (no XDG_CACHE_HOME etc.)",
        ))
    })?;
    let dir = base.join("mellonella").join("models");
    fs::create_dir_all(&dir).map_err(FetchError::Cache)?;
    Ok(dir)
}

/// Local cache path for `<owner>/<repo>/<file>` under [`cache_root`].
/// Does not download; intended for `if !path.exists() { fetch_file(...) }`
/// style checks.
///
/// # Errors
/// As for [`cache_root`].
pub fn cached_path(repo: &str, file: &str) -> Result<PathBuf, FetchError> {
    Ok(cache_root()?.join(repo).join(file))
}

/// Download `https://huggingface.co/<repo>/resolve/main/<file>` into the
/// local cache and return its on-disk path. If the file is already
/// cached, returns the cached path without contacting the server.
///
/// `progress` is invoked periodically with `(bytes_so_far, total_bytes)`;
/// `total_bytes` is `None` when the server doesn't send `Content-Length`.
/// Pass `|_, _| {}` if progress isn't needed.
///
/// # Errors
/// * [`FetchError::Cache`] — cache directory unavailable.
/// * [`FetchError::Http`] — connection / TLS / DNS failure.
/// * [`FetchError::Status`] — non-200 response.
/// * [`FetchError::Io`] — local write failed.
pub fn fetch_file(
    repo: &str,
    file: &str,
    mut progress: impl FnMut(u64, Option<u64>),
) -> Result<PathBuf, FetchError> {
    let dst = cached_path(repo, file)?;
    if dst.exists() {
        return Ok(dst);
    }
    if let Some(parent) = dst.parent() {
        fs::create_dir_all(parent).map_err(FetchError::Cache)?;
    }
    let url = format!("https://huggingface.co/{repo}/resolve/main/{file}");
    let response = ureq::get(&url).call()?;
    if response.status() != 200 {
        return Err(FetchError::Status {
            url,
            code: response.status(),
        });
    }
    let total: Option<u64> = response
        .header("Content-Length")
        .and_then(|s| s.parse::<u64>().ok());
    // Download to a `.partial` neighbour and rename on success so an
    // interrupted run never leaves a half-written file in the cache.
    let tmp = dst.with_extension(format!(
        "{}.partial",
        dst.extension().and_then(|s| s.to_str()).unwrap_or(""),
    ));
    let mut reader = response.into_reader();
    let mut writer = fs::File::create(&tmp).map_err(FetchError::Io)?;
    let mut buf = vec![0_u8; 64 * 1024];
    let mut so_far: u64 = 0;
    loop {
        let n = reader.read(&mut buf).map_err(FetchError::Io)?;
        if n == 0 {
            break;
        }
        writer.write_all(&buf[..n]).map_err(FetchError::Io)?;
        so_far += n as u64;
        progress(so_far, total);
    }
    writer.flush().map_err(FetchError::Io)?;
    drop(writer);
    fs::rename(&tmp, &dst).map_err(FetchError::Io)?;
    Ok(dst)
}

/// Fetch the canonical Stage C TSE prod_48k bundle
/// ([`TSE_PROD_48K_REPO`]) — both `tse_prod_48k.onnx` and its external
/// weights sidecar `tse_prod_48k.onnx.data` — and return the path to
/// the `.onnx`. The `.data` lives next to it (referenced by relative
/// path inside the graph), so `TseSession::from_onnx_path` can load
/// from the returned path directly.
///
/// # Errors
/// As for [`fetch_file`].
pub fn fetch_tse_prod_48k(
    mut progress: impl FnMut(&str, u64, Option<u64>),
) -> Result<PathBuf, FetchError> {
    let mut onnx_path = None;
    for file in TSE_PROD_48K_FILES {
        let path = fetch_file(TSE_PROD_48K_REPO, file, |so_far, total| {
            progress(file, so_far, total);
        })?;
        if has_onnx_extension(file) {
            onnx_path = Some(path);
        }
    }
    onnx_path.ok_or_else(|| {
        FetchError::Cache(io::Error::other(
            "internal error: TSE_PROD_48K_FILES has no .onnx entry",
        ))
    })
}

/// Identifies one of the four ONNX models the live engine needs. Used
/// by [`EnsureProgress`] callbacks so a single UI progress bar can
/// label which model is currently being fetched.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ModelKind {
    Tse,
    Vad,
    Dfn3,
    Ecapa,
}

impl ModelKind {
    /// Human-readable label for status displays.
    #[must_use]
    pub const fn label(self) -> &'static str {
        match self {
            Self::Tse => "TSE (target-speaker extraction)",
            Self::Vad => "Silero VAD",
            Self::Dfn3 => "DeepFilterNet 3",
            Self::Ecapa => "ECAPA-TDNN",
        }
    }
}

/// Resolve a model ONNX path with the fallback chain:
///   1. `env_var` (when set and the file exists) — current dev / CI flow
///   2. Local cache (when previously downloaded) — second-and-later launches
///   3. HuggingFace fetch (when a canonical `(repo, file)` is known) —
///      first-launch auto-setup
///
/// Returns the first hit. `progress` is invoked during step 3 only.
/// Pass `|_, _| {}` if progress isn't needed.
///
/// # Errors
/// * Forwards any [`FetchError`] from a failed HF download.
/// * Returns `FetchError::Cache` with a descriptive message when the
///   env var is unset **and** no HF canonical URL is configured (i.e.
///   the model can't be auto-fetched yet).
fn resolve_model_path(
    env_var: &str,
    canonical: Option<(&str, &str)>,
    progress: impl FnMut(u64, Option<u64>),
) -> Result<PathBuf, FetchError> {
    if let Some(raw) = std::env::var_os(env_var) {
        let p = PathBuf::from(raw);
        if p.exists() {
            return Ok(p);
        }
    }
    if let Some((repo, file)) = canonical {
        let cached = cached_path(repo, file)?;
        if cached.exists() {
            return Ok(cached);
        }
        return fetch_file(repo, file, progress);
    }
    Err(FetchError::Cache(io::Error::other(format!(
        "{env_var} not set and no HuggingFace mirror is configured for this model — \
         run scripts/export_*.py to generate the ONNX and point {env_var} at it"
    ))))
}

/// Ensure the Silero VAD ONNX is available locally and return its path.
/// Falls back to `MELLONELLA_VAD_ONNX` when set.
///
/// # Errors
/// As for [`fetch_file`].
pub fn ensure_vad_onnx(progress: impl FnMut(u64, Option<u64>)) -> Result<PathBuf, FetchError> {
    resolve_model_path("MELLONELLA_VAD_ONNX", Some((VAD_REPO, VAD_FILE)), progress)
}

/// Ensure the DFN3 ONNX is available locally and return its path.
/// Falls back to `MELLONELLA_DFN3_ONNX` when set.
///
/// # Errors
/// As for [`fetch_file`].
pub fn ensure_dfn3_onnx(progress: impl FnMut(u64, Option<u64>)) -> Result<PathBuf, FetchError> {
    resolve_model_path(
        "MELLONELLA_DFN3_ONNX",
        Some((MELLONELLA_MODELS_REPO, DFN3_FILE)),
        progress,
    )
}

/// Ensure the ECAPA-TDNN ONNX is available locally and return its path.
/// Falls back to `MELLONELLA_ECAPA_ONNX` when set.
///
/// # Errors
/// As for [`fetch_file`].
pub fn ensure_ecapa_onnx(progress: impl FnMut(u64, Option<u64>)) -> Result<PathBuf, FetchError> {
    resolve_model_path(
        "MELLONELLA_ECAPA_ONNX",
        Some((MELLONELLA_MODELS_REPO, ECAPA_FILE)),
        progress,
    )
}

/// Ensure the TSE prod_48k ONNX is available locally and return its path.
/// Two-file bundle (`.onnx` + `.onnx.data`); both are pulled together when
/// fetching. Falls back to the existing `MELLONELLA_TSE_PROD_48K_ONNX` env
/// var when set.
///
/// # Errors
/// As for [`fetch_file`].
pub fn ensure_tse_prod_48k_onnx(
    mut progress: impl FnMut(&str, u64, Option<u64>),
) -> Result<PathBuf, FetchError> {
    if let Some(raw) = std::env::var_os("MELLONELLA_TSE_PROD_48K_ONNX") {
        let p = PathBuf::from(raw);
        if p.exists() {
            return Ok(p);
        }
    }
    fetch_tse_prod_48k(|file, so_far, total| progress(file, so_far, total))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cache_root_succeeds_or_skips() {
        // Smoke test only — cache_root() returns Err on platforms
        // where dirs::cache_dir() is None. CI should always have one,
        // but the test stays lenient.
        match cache_root() {
            Ok(p) => assert!(p.is_absolute(), "cache root must be absolute: {p:?}"),
            Err(e) => eprintln!("[skip] no cache dir on this platform: {e}"),
        }
    }

    #[test]
    fn cached_path_includes_repo_and_file() {
        let Ok(p) = cached_path("foo/bar", "baz.onnx") else {
            eprintln!("[skip] no cache dir");
            return;
        };
        let s = p.to_string_lossy();
        assert!(s.contains("foo/bar") || s.contains("foo\\bar"), "{s}");
        assert!(s.ends_with("baz.onnx"), "{s}");
    }

    #[test]
    fn tse_prod_48k_constants_are_consistent() {
        assert!(TSE_PROD_48K_FILES.iter().any(|f| has_onnx_extension(f)));
        assert!(TSE_PROD_48K_FILES.iter().any(|f| f.ends_with(".onnx.data")));
    }

    #[test]
    fn resolve_model_path_errors_when_nothing_configured() {
        // Env var unset and no canonical URL → should return a
        // descriptive cache error.
        let err = resolve_model_path("MELLONELLA_NEVER_SET_VAR_XYZ", None, |_, _| {}).unwrap_err();
        let msg = format!("{err}");
        assert!(
            msg.contains("MELLONELLA_NEVER_SET_VAR_XYZ"),
            "error should name the env var: {msg}"
        );
    }

    #[test]
    fn model_kind_labels_are_non_empty() {
        for k in [
            ModelKind::Tse,
            ModelKind::Vad,
            ModelKind::Dfn3,
            ModelKind::Ecapa,
        ] {
            assert!(!k.label().is_empty());
        }
    }
}
