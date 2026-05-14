"""Unit tests for the Stage C causal Conv-TasNet TSE model.

Covers: parameter-count band, causality (no dependence on future input),
full-sequence vs. streaming equivalence, FiLM conditioning actually
changing the output, and the sign/scale of the SI-SDR loss.
"""

from __future__ import annotations

import numpy as np
import torch

from tse.config import TSEConfig
from tse.data import TSEMixtureDataset, synthetic_fixture_dataset
from tse.loss import neg_si_sdr_loss, si_sdr
from tse.model import CausalConvTasNetTSE, count_parameters


def _model(seed: int = 0) -> CausalConvTasNetTSE:
    torch.manual_seed(seed)
    return CausalConvTasNetTSE(TSEConfig.poc_16k()).eval()


# ---------------------------------------------------------------------------
# parameter count
# ---------------------------------------------------------------------------


def test_param_count_in_expected_range() -> None:
    model = _model()
    n = count_parameters(model)
    # The confirmed architecture (R=2, X=6, B=128, H=256, N=256) lands near
    # the lower edge of the 1.5-2.5 M design target. Keep a band that
    # brackets it while still catching gross structural mistakes.
    assert 1_000_000 <= n <= 3_000_000, f"unexpected param count {n:,}"


def test_prod_config_keeps_latent_rate_and_separator_size() -> None:
    poc = TSEConfig.poc_16k()
    prod = TSEConfig.prod_48k()
    # Latent frame rate is held constant across PoC and production.
    assert poc.latent_rate_hz == prod.latent_rate_hz == 1000.0
    # The separator is identical — only the encoder/decoder geometry scales.
    poc_params = count_parameters(CausalConvTasNetTSE(poc))
    prod_params = count_parameters(CausalConvTasNetTSE(prod))
    # The decoder/encoder grow with the larger kernel; everything else is
    # the same, so the totals differ only modestly.
    assert prod_params > poc_params
    assert prod_params < poc_params * 1.5


# ---------------------------------------------------------------------------
# causality
# ---------------------------------------------------------------------------


def test_causality_no_dependence_on_future() -> None:
    """Perturbing a future input sample must not change earlier outputs."""
    model = _model(1)
    torch.manual_seed(2)
    total = 16 * 60
    mix = torch.randn(1, total)
    cond = torch.randn(1, 192)

    with torch.no_grad():
        base = model(mix, cond)
        # Perturb a sample well into the second half.
        t_pert = total - 16 * 5
        mix2 = mix.clone()
        mix2[0, t_pert] += 5.0
        perturbed = model(mix2, cond)

    # Outputs strictly before the perturbation must be untouched.
    head_delta = float((base[:, :t_pert] - perturbed[:, :t_pert]).abs().max())
    assert head_delta < 1e-5, f"future leaked into the past: head Δ={head_delta:.2e}"
    # And the perturbation must actually have *some* downstream effect.
    tail_delta = float((base[:, t_pert:] - perturbed[:, t_pert:]).abs().max())
    assert tail_delta > 1e-4, "perturbation had no effect at all — model may be dead"


# ---------------------------------------------------------------------------
# full vs streaming
# ---------------------------------------------------------------------------


def test_full_vs_streaming_equivalence() -> None:
    model = _model(3)
    torch.manual_seed(4)
    chunk_len = 16 * 8
    n_chunks = 7
    total = chunk_len * n_chunks
    mix = torch.randn(2, total)
    cond = torch.randn(2, 192)

    with torch.no_grad():
        full = model(mix, cond)
        state = model.make_initial_state(batch_size=2)
        outs = []
        for i in range(n_chunks):
            out, state = model.forward_streaming(
                mix[:, i * chunk_len : (i + 1) * chunk_len], cond, state
            )
            outs.append(out)
        streamed = torch.cat(outs, dim=1)

    assert full.shape == streamed.shape
    delta = float((full - streamed).abs().max())
    assert delta < 1e-4, f"full vs streaming diverged: {delta:.2e}"


def test_streaming_rejects_bad_chunk_length() -> None:
    model = _model()
    state = model.make_initial_state(batch_size=1)
    bad_chunk = torch.randn(1, 16 * 3 + 1)  # not a stride multiple
    try:
        model.forward_streaming(bad_chunk, torch.randn(1, 192), state)
    except ValueError:
        pass
    else:  # pragma: no cover
        raise AssertionError("expected ValueError for non-stride-multiple chunk")


def test_state_tensor_count_matches_layout() -> None:
    model = _model()
    state = model.make_initial_state(batch_size=1)
    assert len(state) == model.n_state_tensors
    # 1 enc-overlap + 3 input-norm + 7 per block + 1 dec-overlap.
    assert model.n_state_tensors == 1 + 3 + 7 * len(model.blocks) + 1


# ---------------------------------------------------------------------------
# FiLM conditioning
# ---------------------------------------------------------------------------


def test_film_conditioning_changes_output() -> None:
    """Different enrollment embeddings must yield different extractions."""
    model = _model(5)
    torch.manual_seed(6)
    mix = torch.randn(1, 16 * 80)
    cond_a = torch.randn(1, 192)
    cond_b = torch.randn(1, 192)

    with torch.no_grad():
        out_a = model(mix, cond_a)
        out_b = model(mix, cond_b)
        out_a2 = model(mix, cond_a)

    # Same conditioning → identical output.
    assert float((out_a - out_a2).abs().max()) < 1e-6
    # Different conditioning → meaningfully different output.
    delta = float((out_a - out_b).abs().max())
    assert delta > 1e-3, f"FiLM conditioning had ~no effect: Δ={delta:.2e}"


def test_film_zero_embedding_is_finite() -> None:
    model = _model()
    mix = torch.randn(1, 16 * 40)
    out = model(mix, torch.zeros(1, 192))
    assert torch.isfinite(out).all()


# ---------------------------------------------------------------------------
# loss
# ---------------------------------------------------------------------------


def test_si_sdr_perfect_estimate_is_high() -> None:
    torch.manual_seed(7)
    ref = torch.randn(2, 4000)
    # An exact copy → very high SI-SDR, near-zero loss magnitude.
    value = si_sdr(ref, ref.clone())
    assert torch.all(value > 100.0)
    loss = neg_si_sdr_loss(ref.clone(), ref)
    assert loss < -100.0  # loss is negative SI-SDR


def test_si_sdr_scale_invariance() -> None:
    torch.manual_seed(8)
    ref = torch.randn(1, 4000)
    # A non-trivial estimate (so the score is finite and meaningful);
    # scaling it must not change SI-SDR.
    est = ref + 0.2 * torch.randn(1, 4000)
    a = si_sdr(ref, est * 3.7)
    b = si_sdr(ref, est)
    assert float((a - b).abs().max()) < 1e-3


def test_si_sdr_worse_estimate_has_lower_score() -> None:
    torch.manual_seed(9)
    ref = torch.randn(1, 4000)
    good = ref + 0.01 * torch.randn(1, 4000)
    bad = ref + 1.0 * torch.randn(1, 4000)
    assert float(si_sdr(ref, good)) > float(si_sdr(ref, bad))
    # Loss ordering is the reverse.
    assert float(neg_si_sdr_loss(good, ref)) < float(neg_si_sdr_loss(bad, ref))


def test_loss_matches_bench_si_sdr_reference() -> None:
    """The torch loss must agree with bench's NumPy si_sdr definition."""
    rng = np.random.default_rng(0)
    ref_np = rng.standard_normal(8000).astype(np.float32)
    est_np = (ref_np + 0.3 * rng.standard_normal(8000)).astype(np.float32)

    # Inline copy of bench/mellonella_bench/metrics/ns_quality.py::si_sdr.
    eps = 1e-12
    s = ref_np.astype(np.float64)
    s_hat = est_np.astype(np.float64)
    s = s - s.mean()
    s_hat = s_hat - s_hat.mean()
    denom = float(np.dot(s, s)) + eps
    alpha = float(np.dot(s_hat, s)) / denom
    s_target = alpha * s
    e_noise = s_hat - s_target
    num = float(np.dot(s_target, s_target)) + eps
    den = float(np.dot(e_noise, e_noise)) + eps
    bench_value = 10.0 * float(np.log10(num / den))

    torch_value = float(si_sdr(torch.from_numpy(ref_np), torch.from_numpy(est_np)))
    assert abs(torch_value - bench_value) < 1e-2, (
        f"torch SI-SDR {torch_value:.4f} disagrees with bench {bench_value:.4f}"
    )


# ---------------------------------------------------------------------------
# dataset
# ---------------------------------------------------------------------------


def test_synthetic_fixture_dataset_shapes() -> None:
    ds = synthetic_fixture_dataset(n=4, sample_rate=16_000, duration_sec=1.0)
    assert len(ds) == 4
    mix, cond, target = ds[0]
    assert mix.shape == target.shape == (16_000,)
    assert cond.shape == (192,)
    assert torch.isfinite(mix).all() and torch.isfinite(target).all()


def test_dataset_deterministic_when_not_random_cropping() -> None:
    ds = synthetic_fixture_dataset(n=2, sample_rate=16_000, duration_sec=1.0)
    a1 = ds[0][0]
    a2 = ds[0][0]
    assert torch.equal(a1, a2)


def test_dataset_requires_nonempty_sources() -> None:
    try:
        TSEMixtureDataset([])
    except ValueError:
        pass
    else:  # pragma: no cover
        raise AssertionError("expected ValueError for empty source list")
