//! Shared `ort` threading policy for ECAPA / VAD / DFN3 sessions.
//!
//! The default `Session::builder()` configuration sets both
//! intra-op and inter-op pools to `num_cores`, which thrashes on
//! small (2-vCPU) hosts: every op pool tries to spread the same op
//! across all cores while the inter-op pool also tries to run ops in
//! parallel, and the OS has nowhere to schedule the actual work.
//!
//! For single-batch offline inference (one audio chunk at a time):
//!
//! * `intra_op_num_threads` — the physical core count, but clamped at
//!   2 so we don't spawn a pool of 16 threads on a small VM that has
//!   to schedule everything onto 2 vCPUs.
//! * `inter_op_num_threads` — fixed at 1. We never benefit from
//!   parallelising ops within a single inference call because the
//!   graph is essentially linear and small.
//!
//! Override via the `MELLONELLA_ORT_INTRA_THREADS` env var when the
//! defaults don't fit (e.g. dedicated server with a wide CPU).

/// Returns the intra-op thread count to pin on each `Session`.
#[must_use]
pub fn intra_op_threads() -> usize {
    if let Ok(s) = std::env::var("MELLONELLA_ORT_INTRA_THREADS") {
        if let Ok(n) = s.parse::<usize>() {
            return n.max(1);
        }
    }
    std::thread::available_parallelism()
        .map_or(1, std::num::NonZeroUsize::get)
        .min(2)
}
