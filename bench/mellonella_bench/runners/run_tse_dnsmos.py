"""DNSMOS evaluation runner for the Stage C TSE model.

Runs a small fixed test set (clean target / target+noise / target+interferer
/ target+interferer+noise) through the 48 kHz ``tse-conv-tasnet-48k`` ONNX
and scores the input mixture and the extracted output with DNSMOS P.835
(SIG / BAK / OVRL) and P.808.

Purpose: a regression gate for the cross-corpus generalisation of the
Stage C TSE. The HF model card reports +9.90 dB SI-SDR on its VCTK + DEMAND
validation; this runner measures the same model on a different corpus
(LibriTTS-R or any user-supplied 48 kHz mono WAV pairs) so that
:doc:`docs/decisions.md` D-014's "scenario-level evaluation" trigger has a
concrete data point.

Inputs
------
The runner consumes a directory layout::

    <audio_dir>/
      target_0.wav .. target_{N-1}.wav   # clean target speech, 48 kHz mono
      inter_0.wav  .. inter_{N-1}.wav    # different-speaker interferer
      noise.wav                          # ambient noise, at least as long
                                         # as the longest target clip
      enroll_cond.npy                    # 192-dim float32 ECAPA embedding
                                         # of the target speaker

ONNX paths
----------
* TSE: ``penta2himajin/tse-conv-tasnet-48k`` (``tse_prod_48k.onnx`` +
  ``.data`` sidecar)
* DNSMOS P.835: ``DNSMOS/sig_bak_ovr.onnx`` from the Microsoft
  DNS-Challenge repo
* DNSMOS P.808: ``DNSMOS/model_v8.onnx`` from the same repo

CLI
---
::

    python -m mellonella_bench.runners.run_tse_dnsmos \\
        --tse-onnx /path/to/tse_prod_48k.onnx \\
        --dnsmos-sbo /path/to/sig_bak_ovr.onnx \\
        --dnsmos-p808 /path/to/model_v8.onnx \\
        --audio-dir /path/to/audio \\
        --output /path/to/results.json \\
        [--n-utts 4]

Limitations
-----------
* DNSMOS is an objective MOS estimator; treat absolute values as
  references only. Compare ``input`` vs ``tse`` deltas, not raw scores.
* The TSE model NaNs on pure-silence inputs (the cumulative LN epsilon is
  1e-8); the runner adds 1e-6 RMS Gaussian dither (~-120 dBFS, inaudible)
  to every frame.
* SI-SDR is intrusive (vs the clean target reference); only meaningful in
  scenarios where a clean reference is available.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import soundfile as sf

CHUNK = 480
SR = 48_000
DNSMOS_SR = 16_000
DNSMOS_INPUT_LEN_S = 9.01
SILENCE_DITHER_RMS = 1e-6


@dataclass
class Scenario:
    name: str
    mixture: np.ndarray
    reference: np.ndarray  # clean target for intrusive SI-SDR


@dataclass
class ScenarioResult:
    scenario: str
    utt: int
    input_dnsmos: dict[str, float]
    tse_dnsmos: dict[str, float]
    si_sdr_input: float
    si_sdr_tse: float


# ---------------------------------------------------------------------------
# TSE inference
# ---------------------------------------------------------------------------
class TseSession:
    """Streaming wrapper around ``tse_prod_48k.onnx`` (89 state tensors).

    Holds nothing between ``run`` calls — each call starts from a fresh
    zero state, since the runner exercises one utterance at a time.
    """

    def __init__(self, onnx_path: Path):
        import onnxruntime as ort

        opts = ort.SessionOptions()
        opts.intra_op_num_threads = 2
        opts.inter_op_num_threads = 1
        self.sess = ort.InferenceSession(str(onnx_path), opts, providers=["CPUExecutionProvider"])
        self.input_names = [i.name for i in self.sess.get_inputs()]
        self.output_names = [o.name for o in self.sess.get_outputs()]
        self.state_in_shapes = {
            i.name: i.shape for i in self.sess.get_inputs() if i.name.startswith("state_in_")
        }

    def _fresh_state(self) -> dict[str, np.ndarray]:
        return {n: np.zeros(s, dtype=np.float32) for n, s in self.state_in_shapes.items()}

    def run(self, audio: np.ndarray, cond: np.ndarray) -> np.ndarray:
        if cond.ndim == 1:
            cond_in = cond.reshape(1, -1).astype(np.float32)
        else:
            cond_in = cond.astype(np.float32)
        rng = np.random.default_rng(42)
        dithered = (
            audio + rng.standard_normal(len(audio)).astype(np.float32) * SILENCE_DITHER_RMS
        ).astype(np.float32)
        n_chunks = len(dithered) // CHUNK
        state = self._fresh_state()
        chunks: list[np.ndarray] = []
        for i in range(n_chunks):
            chunk = dithered[i * CHUNK : (i + 1) * CHUNK].reshape(1, CHUNK).astype(np.float32)
            feeds = {"audio_chunk": chunk, "cond_embedding": cond_in, **state}
            results = self.sess.run(self.output_names, feeds)
            chunks.append(results[0].flatten())
            new_state = {}
            for name, val in zip(self.output_names[1:], results[1:], strict=True):
                idx = name.rsplit("_", 1)[-1]
                new_state[f"state_in_{idx}"] = val
            state = new_state
        if not chunks:
            return np.zeros(0, dtype=np.float32)
        return np.concatenate(chunks)


# ---------------------------------------------------------------------------
# DNSMOS
# ---------------------------------------------------------------------------
class DnsmosSession:
    """DNSMOS P.835 (SIG/BAK/OVR) + P.808 single-MOS estimator.

    Follows the Microsoft DNS-Challenge reference (``dnsmos_local.py``):
    9.01-second 16 kHz windows, polynomial fit on the raw P.835 outputs
    (non-personalised coefficients), 120-mel-band log-spectrogram feed for
    the P.808 model.
    """

    def __init__(self, sbo_path: Path, p808_path: Path):
        import onnxruntime as ort

        opts = ort.SessionOptions()
        opts.intra_op_num_threads = 2
        self.sbo = ort.InferenceSession(str(sbo_path), opts, providers=["CPUExecutionProvider"])
        self.p808 = ort.InferenceSession(str(p808_path), opts, providers=["CPUExecutionProvider"])

    @staticmethod
    def _melspec(audio: np.ndarray) -> np.ndarray:
        import librosa

        m = librosa.feature.melspectrogram(
            y=audio, sr=DNSMOS_SR, n_fft=321, hop_length=160, n_mels=120
        )
        m = (librosa.power_to_db(m, ref=np.max) + 40) / 40
        return m.T.astype(np.float32)

    @staticmethod
    def _polyfit(sig: float, bak: float, ovr: float) -> tuple[float, float, float]:
        p_ovr = np.poly1d([-0.06766283, 1.11546468, 0.04602535])
        p_sig = np.poly1d([-0.08397278, 1.22083953, 0.0052439])
        p_bak = np.poly1d([-0.13166888, 1.60915514, -0.39604546])
        return float(p_sig(sig)), float(p_bak(bak)), float(p_ovr(ovr))

    def __call__(self, audio48: np.ndarray) -> dict[str, float]:
        import scipy.signal as sps

        a = sps.resample_poly(audio48.astype(np.float32), DNSMOS_SR, SR).astype(np.float32)
        target_len = int(DNSMOS_SR * DNSMOS_INPUT_LEN_S)
        while len(a) < target_len:
            a = np.concatenate([a, a])
        n_hops = int(np.floor(len(a) / DNSMOS_SR) - DNSMOS_INPUT_LEN_S) + 1
        sigs, baks, ovrs, p808s = [], [], [], []
        for k in range(n_hops):
            seg = a[k * DNSMOS_SR : k * DNSMOS_SR + target_len]
            if len(seg) < target_len:
                continue
            feats = seg.reshape(1, -1).astype(np.float32)
            mel = self._melspec(seg[:-160])[np.newaxis, :, :]
            raw = self.sbo.run(None, {"input_1": feats})[0][0]
            p808 = float(self.p808.run(None, {"input_1": mel})[0][0][0])
            sig_p, bak_p, ovr_p = self._polyfit(*raw)
            sigs.append(sig_p)
            baks.append(bak_p)
            ovrs.append(ovr_p)
            p808s.append(p808)
        return {
            "SIG": float(np.mean(sigs)),
            "BAK": float(np.mean(baks)),
            "OVRL": float(np.mean(ovrs)),
            "P808": float(np.mean(p808s)),
        }


# ---------------------------------------------------------------------------
# Metrics + mixing helpers
# ---------------------------------------------------------------------------
def si_sdr(reference: np.ndarray, estimate: np.ndarray) -> float:
    n = min(len(reference), len(estimate))
    s = reference[:n].astype(np.float64) - reference[:n].mean()
    s_hat = estimate[:n].astype(np.float64) - estimate[:n].mean()
    eps = 1e-12
    alpha = float(np.dot(s_hat, s)) / (float(np.dot(s, s)) + eps)
    s_t = alpha * s
    e_n = s_hat - s_t
    num = float(np.dot(s_t, s_t)) + eps
    den = float(np.dot(e_n, e_n)) + eps
    return 10.0 * float(np.log10(num / den))


def _mix_at_db(s1: np.ndarray, s2: np.ndarray, db: float) -> np.ndarray:
    s1f = np.asarray(s1, dtype=np.float64)
    s2f = np.asarray(s2, dtype=np.float64)
    e1 = float(np.sqrt(np.mean(s1f**2)) + 1e-12)
    e2 = float(np.sqrt(np.mean(s2f**2)) + 1e-12)
    return (s2f * ((e1 / e2) * (10 ** (-db / 20.0)))).astype(np.float32)


def _build_scenarios(
    target: np.ndarray, interferer: np.ndarray, noise: np.ndarray
) -> list[Scenario]:
    n = min(len(target), len(interferer), len(noise))
    n = (n // CHUNK) * CHUNK
    t = target[:n].astype(np.float32)
    ifr = interferer[:n].astype(np.float32)
    ns = noise[:n].astype(np.float32)
    return [
        Scenario("A_clean", t.copy(), t.copy()),
        Scenario("B_t_noise_10dB", (t + _mix_at_db(t, ns, 10.0)).astype(np.float32), t.copy()),
        Scenario("C_t_inter_0dB", (t + _mix_at_db(t, ifr, 0.0)).astype(np.float32), t.copy()),
        Scenario(
            "D_t_inter_n_5dB",
            (t + _mix_at_db(t, ifr, 5.0) + _mix_at_db(t, ns, 5.0)).astype(np.float32),
            t.copy(),
        ),
    ]


def _align(x: np.ndarray, n: int) -> np.ndarray:
    if len(x) >= n:
        return x[:n]
    return np.pad(x, (0, n - len(x)))


def _clip_for_overload(x: np.ndarray) -> np.ndarray:
    peak = float(np.max(np.abs(x)))
    if peak <= 0.95:
        return x.astype(np.float32)
    return (x / peak * 0.95).astype(np.float32)


# ---------------------------------------------------------------------------
# Eval
# ---------------------------------------------------------------------------
def evaluate(
    audio_dir: Path,
    tse: TseSession,
    mos: DnsmosSession,
    cond: np.ndarray,
    *,
    n_utts: int = 4,
) -> dict:
    targets = [
        sf.read(str(audio_dir / f"target_{i}.wav"))[0].astype(np.float32) for i in range(n_utts)
    ]
    inters = [
        sf.read(str(audio_dir / f"inter_{i}.wav"))[0].astype(np.float32) for i in range(n_utts)
    ]
    noise = sf.read(str(audio_dir / "noise.wav"))[0].astype(np.float32)

    results: list[ScenarioResult] = []
    for utt_idx in range(n_utts):
        scenarios = _build_scenarios(targets[utt_idx], inters[utt_idx], noise)
        for sc in scenarios:
            mixture_clipped = _clip_for_overload(sc.mixture)
            extracted = tse.run(mixture_clipped, cond)
            ref_aligned = _align(sc.reference, len(extracted))
            results.append(
                ScenarioResult(
                    scenario=sc.name,
                    utt=utt_idx,
                    input_dnsmos=mos(mixture_clipped),
                    tse_dnsmos=mos(extracted),
                    si_sdr_input=si_sdr(ref_aligned, mixture_clipped[: len(extracted)]),
                    si_sdr_tse=si_sdr(ref_aligned, extracted),
                )
            )

    def _agg(name: str) -> dict:
        rows = [r for r in results if r.scenario == name]
        keys = ("SIG", "BAK", "OVRL", "P808")
        return {
            "input": {k: float(np.mean([r.input_dnsmos[k] for r in rows])) for k in keys},
            "tse": {k: float(np.mean([r.tse_dnsmos[k] for r in rows])) for k in keys},
            "si_sdr_input": float(np.mean([r.si_sdr_input for r in rows])),
            "si_sdr_tse": float(np.mean([r.si_sdr_tse for r in rows])),
            "delta_OVRL": float(
                np.mean([r.tse_dnsmos["OVRL"] - r.input_dnsmos["OVRL"] for r in rows])
            ),
            "delta_si_sdr": float(np.mean([r.si_sdr_tse - r.si_sdr_input for r in rows])),
        }

    return {
        "per_utt": [asdict(r) for r in results],
        "summary": {
            sc: _agg(sc) for sc in ("A_clean", "B_t_noise_10dB", "C_t_inter_0dB", "D_t_inter_n_5dB")
        },
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--tse-onnx", required=True, type=Path)
    p.add_argument("--dnsmos-sbo", required=True, type=Path)
    p.add_argument("--dnsmos-p808", required=True, type=Path)
    p.add_argument("--audio-dir", required=True, type=Path)
    p.add_argument(
        "--cond-npy",
        type=Path,
        default=None,
        help="ECAPA cond embedding .npy (float32[192]). Defaults to "
        "<audio-dir>/enroll_cond.npy.",
    )
    p.add_argument("--output", required=True, type=Path, help="results.json output path")
    p.add_argument("--n-utts", type=int, default=4)
    return p


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    cond_path = args.cond_npy or (args.audio_dir / "enroll_cond.npy")
    if not cond_path.exists():
        print(f"[tse-dnsmos] ECAPA cond not found at {cond_path}", file=sys.stderr)
        return 2
    cond = np.load(cond_path).astype(np.float32)
    if cond.shape != (192,):
        print(f"[tse-dnsmos] cond shape must be (192,), got {cond.shape}", file=sys.stderr)
        return 2

    tse = TseSession(args.tse_onnx)
    mos = DnsmosSession(args.dnsmos_sbo, args.dnsmos_p808)
    results = evaluate(args.audio_dir, tse, mos, cond, n_utts=args.n_utts)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(results, indent=2))
    print(f"[tse-dnsmos] wrote {args.output}")

    print("\nScenario             | input OVRL → tse OVRL | Δ      | input SI-SDR → tse SI-SDR | Δ")
    print("-" * 100)
    for sc, v in results["summary"].items():
        print(
            f"{sc:20s} | {v['input']['OVRL']:.2f}  →  {v['tse']['OVRL']:.2f}        "
            f"| {v['delta_OVRL']:+.2f}  | "
            f"{v['si_sdr_input']:+6.2f} → {v['si_sdr_tse']:+7.2f} dB    "
            f"| {v['delta_si_sdr']:+.2f}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
