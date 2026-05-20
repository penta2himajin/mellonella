//! Shared `ort` threading policy for ECAPA / VAD / DFN3 / TSE /
//! overlap sessions.
//!
//! The default `Session::builder()` configuration sets both
//! intra-op and inter-op pools to `num_cores`, which thrashes on
//! small (2-vCPU) hosts: every op pool tries to spread the same op
//! across all cores while the inter-op pool also tries to run ops in
//! parallel, and the OS has nowhere to schedule the actual work.
//!
//! For single-batch realtime inference (one audio chunk at a time,
//! and the live worker only ever has **one** ONNX session active at
//! a moment because it runs the chain stages sequentially):
//!
//! * `intra_op_num_threads` — the physical core count, clamped at 4.
//!   The clamp used to be 2 (tuned for 2-vCPU CI VMs), but on a real
//!   4-8 core laptop that left the heavier per-chunk inferences
//!   (TSE Conv-TasNet especially) unable to keep up at realtime —
//!   the live monitor logged a steady ~10-25 underruns/s. Since the
//!   worker serialises the stages, giving the single active session
//!   up to 4 threads doesn't oversubscribe the way a 16-thread pool
//!   on a 2-vCPU box did. Still capped so a 32-core workstation
//!   doesn't spawn a 32-thread pool for a tiny linear graph.
//! * `inter_op_num_threads` — fixed at 1. We never benefit from
//!   parallelising ops within a single inference call because the
//!   graph is essentially linear and small.
//!
//! Override via the `MELLONELLA_ORT_INTRA_THREADS` env var when the
//! defaults don't fit (e.g. a 2-vCPU CI box that wants the old
//! clamp of 2, or a dedicated server with a wide CPU).

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
        .min(4)
}
