# ---
# jupyter:
#   jupytext:
#     formats: py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.16.0
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Phase 1 — θ_pass threshold sweep
#
# Validates the calibrated `θ_pass = 0.30` from `docs/gating.md` D-004 against
# real speaker recordings. (The notebook predates `scripts/calibrate.py`; the
# multi-speaker / multi-noise sweep there is now the authoritative calibration.)
#
# **Procedure**
# 1. Build an enrollment pool from the first half of speaker A.
# 2. Sweep θ_pass over `[0.30, 0.85]`.
# 3. For each θ:
#    - Run the second half of speaker A through the gate → measure **TPR**
#      (frame-level pass rate when the target is speaking).
#    - Run speaker B through the same gate → measure **FPR** (false-acceptance
#      rate, target ≠ test).
# 4. Plot TPR/FPR curves and pick an operating point.
#
# The curves are noisy on a single ~10 s sample per speaker, so treat the
# numbers as an order-of-magnitude check, not a final calibration. Re-run with
# multiple recordings per speaker (set `MELLONELLA_NOTEBOOK_DATA` to a custom
# directory) to tighten the analysis.

# %%
from __future__ import annotations

import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import soundfile as sf

from mellonella_poc.config import Config, GatingConfig
from mellonella_poc.pipeline import (
    PipelineComponents,
    enroll_from_recording,
    process_offline,
)

# %% [markdown]
# ## Audio sources
#
# Default: `librosa` ships short LibriSpeech samples (`libri1` = speaker A,
# `libri2` = speaker B). Override with `MELLONELLA_NOTEBOOK_DATA` pointing at a
# directory containing `target_speaker.wav` and `other_speaker.wav`.

# %%
data_dir = Path(os.environ.get("MELLONELLA_NOTEBOOK_DATA") or ".")
target_path = data_dir / "target_speaker.wav"
other_path = data_dir / "other_speaker.wav"

if not (target_path.exists() and other_path.exists()):
    import librosa

    target_path = Path(librosa.example("libri1"))
    other_path = Path(librosa.example("libri2"))
    print(f"Using librosa samples:\n  target: {target_path}\n  other:  {other_path}")


def _load_mono(path: Path) -> tuple[np.ndarray, int]:
    audio, sr = sf.read(str(path), dtype="float32", always_2d=False)
    if audio.ndim == 2:
        audio = audio.mean(axis=1)
    return np.asarray(audio, dtype=np.float32), int(sr)


target_audio, target_sr = _load_mono(target_path)
other_audio, other_sr = _load_mono(other_path)
print(f"target: {target_audio.size / target_sr:.2f}s @ {target_sr} Hz")
print(f"other:  {other_audio.size / other_sr:.2f}s @ {other_sr} Hz")

# %% [markdown]
# ## Enrollment
#
# Use the first half of speaker A as the enrollment recording, and the second
# half as the held-out target test signal.

# %%
midpoint = target_audio.size // 2
enrollment_audio = target_audio[:midpoint]
target_test_audio = target_audio[midpoint:]
print(f"enrollment: {enrollment_audio.size / target_sr:.2f}s")
print(f"target test: {target_test_audio.size / target_sr:.2f}s")

# %%
base_config = Config()
components = PipelineComponents.build_default(base_config)
pool = enroll_from_recording(enrollment_audio, target_sr, base_config, components)
print(
    f"Built pool: {len(pool.anchors)} anchors, "
    f"f0_mu={pool.metadata.f0_mu:.1f} Hz, "
    f"f0_sigma={pool.metadata.f0_sigma:.1f} Hz"
)

# %% [markdown]
# ## θ_pass sweep
#
# `θ_learn` is held just above each candidate `θ_pass` (or 0.80, whichever is
# greater) so the auto-learn admission rule never fires during the sweep —
# we want a clean read on the gate threshold alone.


# %%
def gate_pass_rate(theta_pass: float, audio: np.ndarray, sr: int) -> float:
    cfg = Config(
        audio=base_config.audio,
        gating=GatingConfig(
            theta_pass=theta_pass,
            theta_learn=max(theta_pass + 0.01, 0.80),
        ),
    )
    result = process_offline(audio, sr, pool, cfg, components)
    if result.gate_per_frame.size == 0:
        return 0.0
    return float(result.gate_per_frame.mean())


thetas = np.linspace(0.30, 0.85, 12)
rows: list[dict[str, float]] = []
for theta in thetas:
    tpr = gate_pass_rate(float(theta), target_test_audio, target_sr)
    fpr = gate_pass_rate(float(theta), other_audio, other_sr)
    rows.append({"theta_pass": float(theta), "tpr": tpr, "fpr": fpr})
    print(f"theta={theta:.2f}  TPR={tpr:.3f}  FPR={fpr:.3f}")

# %%
df = pd.DataFrame(rows)
df

# %% [markdown]
# ## Operating-point plot

# %%
fig, ax = plt.subplots(figsize=(8, 5))
ax.plot(df["theta_pass"], df["tpr"], "o-", label="TPR (target → pass)")
ax.plot(df["theta_pass"], df["fpr"], "s-", label="FPR (other → pass)")
ax.axvline(
    GatingConfig().theta_pass,
    color="gray",
    linestyle="--",
    label=r"$\theta_{pass}$ = 0.30 (calibrated default)",
)
ax.set_xlabel(r"$\theta_{pass}$")
ax.set_ylabel("rate")
ax.set_ylim(-0.05, 1.05)
ax.set_title("Gate operating point as a function of " + r"$\theta_{pass}$")
ax.grid(True, alpha=0.3)
ax.legend()
fig.tight_layout()
plt.show()

# %% [markdown]
# ## Take-away
#
# A reasonable operating point keeps TPR > 0.85 while FPR < 0.20. Per
# `docs/gating.md` we explicitly accept FP > FN ("FP 許容方針") for the
# single-speaker target case, so a slightly tighter `θ_pass` than the EER
# crossing is usually preferred.
#
# **Next steps**
# - Re-run with several enrollment / test pairs per speaker to estimate
#   variance.
# - Sweep `α` (cos-sim weight) and `β` (F0-match weight) jointly to verify the
#   `α=0.8, β=0.2` initial split.
# - Plot the same curves under noise (mix in MUSAN; covered by Scenario 1 in
#   `bench/`).
