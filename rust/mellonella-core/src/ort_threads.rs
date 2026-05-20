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
//! * `intra_op_num_threads` — the physical core count, clamped at 2.
//!   #179 briefly raised this to 4 on the theory that the heavier
//!   per-chunk inferences (TSE Conv-TasNet) needed more cores. A
//!   bench on a 4-vCPU box proved the opposite — ORT's intra-op pool
//!   busy-waits, so giving it 4 threads on a 4-core machine starves
//!   the worker / cpal / OS threads and makes the chain *slower*:
//!
//!   | intra | Solo RTF | Overlap (TSE) RTF |
//!   |------:|---------:|------------------:|
//!   |     1 |    2.44× |             1.54× |
//!   |   **2** | **2.75×** |        **1.64×** |
//!   |     3 |    2.02× |             1.35× |
//!   |     4 |    1.62× |             1.06× |
//!
//!   2 is the clear optimum; everything above it regresses. The clamp
//!   is back to 2. The real win for an underrunning laptop is a
//!   lighter TSE (int8 export) or moving ECAPA/TSE off the realtime
//!   thread, not more threads.
//! * `inter_op_num_threads` — fixed at 1. We never benefit from
//!   parallelising ops within a single inference call because the
//!   graph is essentially linear and small.
//!
//! Override via the `MELLONELLA_ORT_INTRA_THREADS` env var when the
//! defaults don't fit (e.g. a dedicated server with a wide CPU and
//! nothing else competing for cores).

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
