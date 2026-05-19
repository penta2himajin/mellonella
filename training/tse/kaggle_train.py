"""Kaggle kernel entrypoint for the Stage C training runs.

Designed to run as a Kaggle Python kernel. Supports two model presets via
the ``POC_CONFIG`` env var:

* ``POC_CONFIG=poc_16k`` (default) — the 16 kHz PoC path:
  LibriSpeech + MUSAN, ``TSEConfig.poc_16k()``.
* ``POC_CONFIG=prod_48k`` — the 48 kHz production path:
  VCTK + DEMAND, ``TSEConfig.prod_48k()`` (encoder kernel/stride 96/48 so
  the latent rate stays at 1 kHz; the separator is byte-identical).

Expected Kaggle inputs (override via env vars below):

================  ============================================================
Preset            Expected datasets
================  ============================================================
poc_16k           LibriSpeech (``LibriSpeech/train-clean-100/<spk>/<ch>/*.flac``)
                  + MUSAN noise (``noise/.../*.wav``)
                  + ECAPA ONNX (``ecapa_tdnn.onnx``)
prod_48k          VCTK (``<pXXX>_<utt>.wav`` somewhere under the dataset root)
                  + DEMAND (``<CATEGORY>/ch01.wav`` per noise category)
                  + ECAPA ONNX
================  ============================================================

The kernel:

1. ``git clone`` mellonella (branch configurable via ``MELLONELLA_REF``).
2. ``pip install -e training[onnx]`` plus SpeechBrain for the ECAPA Fbank
   front-end.
3. Symlinks the input datasets into a single ``data/`` root that the
   loader understands.
4. Runs :mod:`tse.prepare_enrollment_embeddings` (skipped if the
   ``.npz`` already exists from a previous run).
5. Runs :mod:`tse.train` on GPU.
6. Writes checkpoints + ``metrics.json`` to ``/kaggle/working/tse_poc/``.

Env-var overrides (set in the kernel UI under "Settings → Environment"):

* ``MELLONELLA_REF`` — git ref to clone (default ``main``).
* ``POC_CONFIG`` — ``poc_16k`` (default) or ``prod_48k``.
* ``LIBRISPEECH_DATA``, ``MUSAN_DATA``, ``ECAPA_ONNX`` — override the
  ``/kaggle/input/<slug>/...`` paths (poc_16k).
* ``VCTK_DATA``, ``DEMAND_DATA`` — same idea for the 48 kHz path.
* ``POC_EPOCHS``, ``POC_BATCH``, ``POC_N_PAIRS``, ``POC_LR``,
  ``POC_LR_SCHEDULE`` (``none`` / ``cosine`` / ``step``) — training knobs.
* ``POC_CLIP_GRAD_NORM`` — max global gradient norm (default ``5.0`` to
  preserve v3/v4; lower values like ``1.0`` / ``0.5`` are the standard
  fp16-AMP stability mitigation).
* ``ENROLL_LIMIT``, ``ENROLL_DEVICE`` (``auto`` / ``cuda`` / ``cpu``) —
  enrollment-embedding precompute knobs. ``auto`` uses CUDA when
  ``onnxruntime-gpu`` is available, otherwise CPU.

This file is also locally importable for development (it is just a
module); the heavy lifting only runs from ``main`` so ``import`` is cheap.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
from pathlib import Path

DEFAULT_REPO = "https://github.com/penta2himajin/mellonella.git"
DEFAULT_REF = "main"

KAGGLE_INPUT = Path("/kaggle/input")
KAGGLE_WORKING = Path("/kaggle/working")


def _env_path(key: str, default: Path) -> Path:
    raw = os.environ.get(key)
    return Path(raw).expanduser().resolve() if raw else default


def _run(cmd: list[str], *, cwd: Path | None = None, env: dict[str, str] | None = None) -> None:
    print(f"$ {' '.join(map(str, cmd))}", flush=True)
    subprocess.run(cmd, check=True, cwd=str(cwd) if cwd else None, env=env)


def _symlink(src: Path, dst: Path) -> None:
    if dst.exists() or dst.is_symlink():
        return
    dst.parent.mkdir(parents=True, exist_ok=True)
    dst.symlink_to(src)
    print(f"[kaggle] symlink {dst} -> {src}", flush=True)


def main() -> int:
    ref = os.environ.get("MELLONELLA_REF", DEFAULT_REF)
    repo_url = os.environ.get("MELLONELLA_REPO", DEFAULT_REPO)
    repo_dir = KAGGLE_WORKING / "mellonella"

    # 1. Clone the repo (or refresh).
    if not (repo_dir / ".git").exists():
        if repo_dir.exists():
            shutil.rmtree(repo_dir)
        _run(["git", "clone", "--branch", ref, "--depth", "1", repo_url, str(repo_dir)])
    else:
        _run(["git", "fetch", "--depth", "1", "origin", ref], cwd=repo_dir)
        _run(["git", "checkout", "FETCH_HEAD"], cwd=repo_dir)

    # 2. Install the training package + SpeechBrain (for the ECAPA Fbank
    # front-end used by prepare_enrollment_embeddings).
    _run([sys.executable, "-m", "pip", "install", "-q", "-e", f"{repo_dir}/training[onnx]"])
    _run([sys.executable, "-m", "pip", "install", "-q", "speechbrain>=1.0", "librosa>=0.10"])

    # 3. Stitch the Kaggle input datasets into the data layout the loaders
    # expect. The poc_16k path uses LibriSpeech + MUSAN; prod_48k uses
    # VCTK + DEMAND. ECAPA ONNX is required either way for enrollment
    # embedding precompute.
    poc_config = os.environ.get("POC_CONFIG", "poc_16k")
    if poc_config not in ("poc_16k", "prod_48k"):
        raise ValueError(f"POC_CONFIG must be 'poc_16k' or 'prod_48k', got {poc_config!r}")

    ecapa_onnx = _env_path("ECAPA_ONNX", KAGGLE_INPUT / "ecapa-onnx" / "ecapa_tdnn.onnx")
    if not ecapa_onnx.exists():
        raise FileNotFoundError(f"required kaggle input missing: {ecapa_onnx}")

    data_root = KAGGLE_WORKING / "data"

    if poc_config == "poc_16k":
        libri_src = _env_path("LIBRISPEECH_DATA", KAGGLE_INPUT / "librispeech-train-clean-100")
        musan_src = _env_path("MUSAN_DATA", KAGGLE_INPUT / "musan-subset")
        for required in (libri_src, musan_src):
            if not required.exists():
                raise FileNotFoundError(f"required kaggle input missing: {required}")
        # librispeech_musan_sources looks for data_root/LibriSpeech/<split>;
        # users commonly upload the split directory directly, so accept either.
        libri_dst = data_root / "LibriSpeech" / "train-clean-100"
        if (libri_src / "train-clean-100").is_dir():
            _symlink(libri_src / "train-clean-100", libri_dst)
        else:
            _symlink(libri_src, libri_dst)
        # MUSAN: _scan_musan_noise looks in extracted/musan/<subset>,
        # subset/<subset>, or musan/<subset>. We materialise the first.
        musan_dst = data_root / "musan" / "extracted" / "musan" / "noise"
        if (musan_src / "noise").is_dir():
            _symlink(musan_src / "noise", musan_dst)
        else:
            _symlink(musan_src, musan_dst)
        enroll_audio_dir = libri_dst
        data_source = "librispeech-musan"
    else:  # prod_48k
        vctk_src = _env_path("VCTK_DATA", KAGGLE_INPUT / "vctk")
        demand_src = _env_path("DEMAND_DATA", KAGGLE_INPUT / "demand")
        for required in (vctk_src, demand_src):
            if not required.exists():
                raise FileNotFoundError(f"required kaggle input missing: {required}")
        # vctk_demand_sources expects data_root/VCTK-Corpus and data_root/demand.
        vctk_dst = data_root / "VCTK-Corpus"
        _symlink(vctk_src, vctk_dst)
        demand_dst = data_root / "demand"
        _symlink(demand_src, demand_dst)
        enroll_audio_dir = vctk_dst
        data_source = "vctk-demand"

    # 4. Precompute enrollment embeddings (idempotent — skip if up-to-date).
    emb_out = KAGGLE_WORKING / "enroll_embeddings.npz"
    if not emb_out.exists():
        enroll_limit = os.environ.get("ENROLL_LIMIT", "2000")
        enroll_device = os.environ.get("ENROLL_DEVICE", "auto")
        env = os.environ.copy()
        env["MELLONELLA_ECAPA_ONNX"] = str(ecapa_onnx)
        _run(
            [
                sys.executable,
                "-m",
                "tse.prepare_enrollment_embeddings",
                "--audio-dir",
                str(enroll_audio_dir),
                "--out",
                str(emb_out),
                "--limit",
                str(enroll_limit),
                "--device",
                enroll_device,
            ],
            cwd=repo_dir / "training",
            env=env,
        )
    else:
        print(f"[kaggle] reusing existing embeddings at {emb_out}", flush=True)

    # 5. Train.
    out_dir = KAGGLE_WORKING / "tse_poc"
    epochs = os.environ.get("POC_EPOCHS", "20")
    batch = os.environ.get("POC_BATCH", "16")
    lr = os.environ.get("POC_LR", "1e-3")
    lr_schedule = os.environ.get("POC_LR_SCHEDULE", "none")
    n_pairs = os.environ.get("POC_N_PAIRS", "5000")
    optimizer = os.environ.get("POC_OPTIMIZER", "adam")
    weight_decay = os.environ.get("POC_WEIGHT_DECAY", "0.0")
    warmup_epochs = os.environ.get("POC_WARMUP_EPOCHS", "0")
    ema_decay = os.environ.get("POC_EMA_DECAY", "0.0")
    amp = os.environ.get("POC_AMP", "auto")
    clip_grad_norm = os.environ.get("POC_CLIP_GRAD_NORM", "5.0")
    poc_device = os.environ.get("POC_DEVICE", "cuda")
    num_workers = os.environ.get("POC_NUM_WORKERS", "2")
    val_speakers = os.environ.get("POC_VAL_SPEAKERS", "0")
    test_speakers = os.environ.get("POC_TEST_SPEAKERS", "0")
    val_pairs = os.environ.get("POC_VAL_PAIRS", "500")
    cmd = [
        sys.executable,
        "-m",
        "tse.train",
        "--config",
        poc_config,
        "--data-dir",
        str(data_root),
        "--data-source",
        data_source,
        "--embeddings-npz",
        str(emb_out),
        "--n-pairs",
        n_pairs,
        "--epochs",
        epochs,
        "--batch-size",
        batch,
        "--lr",
        lr,
        "--lr-schedule",
        lr_schedule,
        "--warmup-epochs",
        warmup_epochs,
        "--optimizer",
        optimizer,
        "--weight-decay",
        weight_decay,
        "--ema-decay",
        ema_decay,
        "--amp",
        amp,
        "--clip-grad-norm",
        clip_grad_norm,
        "--device",
        poc_device,
        "--num-workers",
        num_workers,
        "--val-speakers",
        val_speakers,
        "--test-speakers",
        test_speakers,
        "--val-pairs",
        val_pairs,
        "--out",
        str(out_dir),
    ]
    if os.environ.get("POC_COMPILE", "0") not in ("0", "", "false", "False"):
        cmd.append("--compile")
    _run(cmd, cwd=repo_dir / "training")

    print(f"[kaggle] done — outputs under {out_dir}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
