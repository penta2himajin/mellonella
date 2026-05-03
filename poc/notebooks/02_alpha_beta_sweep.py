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
# # Phase 1 — α / β joint sweep (F0-aux ablation)
#
# Visualises the joint sweep produced by `scripts/calibrate_alpha_beta.py`
# (results in `docs/benchmarks/calibration_alpha_beta_results.csv`,
# summary in `calibration_alpha_beta_summary.json`). The goal is to
# verify the `α=0.8, β=0.2` initial split from `docs/decisions.md` D-005.
#
# **Procedure (already executed by the script)**
# 1. Capture per-frame `cos_sim_max` + `f0_match` from each pipeline run
#    (3 enrollments × 3 test speakers × 2 noise types × 6 SNRs = 108 cells).
# 2. For each cell, post-hoc reconstruct `score = α · cs + β · fm` for every
#    `α ∈ {0.0, 0.1, …, 1.0}`, then replay `GateState` for every
#    `θ_pass ∈ {0.20, 0.25, …, 0.55}`.
# 3. Aggregate over (SNR ≥ 5 dB) cells: median TPR (target=enroll match)
#    and mean FPR (target≠enroll).
#
# **Plots in this notebook**
# - TPR / FPR vs α at several θ_pass values (line plot)
# - TPR_median heatmap over (α, θ)
# - FPR_mean heatmap over (α, θ)
# - Operating curve at the recommended θ_pass

# %%
from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

REPO_ROOT = Path("..").resolve().parent  # poc/notebooks → poc → repo root
RESULTS_CSV = REPO_ROOT / "docs" / "benchmarks" / "calibration_alpha_beta_results.csv"
SUMMARY_JSON = REPO_ROOT / "docs" / "benchmarks" / "calibration_alpha_beta_summary.json"

df = pd.read_csv(RESULTS_CSV)
summary = json.loads(SUMMARY_JSON.read_text())
print(f"loaded {len(df)} rows; recommended = {summary.get('recommended')}")

# %% [markdown]
# ## Aggregate per (α, θ) at SNR ≥ 5 dB

# %%
df_repr = df[df["snr_db"] >= summary["config"]["min_representative_snr_db"]]
agg = (
    df_repr.groupby(["alpha", "theta_pass", "kind"])["rate"].agg(["median", "mean"]).unstack("kind")
)
# Flatten multi-index for readability
agg.columns = [f"{stat}_{kind}" for stat, kind in agg.columns]
agg = agg.reset_index()
agg

# %% [markdown]
# ## TPR / FPR vs α at the recommended θ_pass and a few neighbours

# %%
recommended = summary.get("recommended") or {}
recommended_theta = float(recommended.get("theta_pass") or 0.30)
nearby_thetas = [t for t in summary["config"]["theta_grid"] if abs(t - recommended_theta) <= 0.10]

fig, ax = plt.subplots(figsize=(9, 5))
for theta in nearby_thetas:
    subset = agg[agg["theta_pass"] == theta]
    ax.plot(subset["alpha"], subset["median_tpr"], "-o", label=f"TPR_med (θ={theta:.2f})")
    ax.plot(subset["alpha"], subset["mean_fpr"], "--s", label=f"FPR_mean (θ={theta:.2f})")
ax.axvline(0.8, color="gray", linestyle=":", label=r"$\alpha=0.8$ (D-005 initial)")
ax.set_xlabel(r"$\alpha$  (cos-sim weight; $\beta = 1 - \alpha$)")
ax.set_ylabel("rate")
ax.set_ylim(-0.05, 1.05)
ax.set_title(r"TPR / FPR vs $\alpha$ at neighbouring $\theta_{pass}$")
ax.grid(True, alpha=0.3)
ax.legend(fontsize=8, ncol=2)
fig.tight_layout()
plt.show()

# %% [markdown]
# ## Heatmap: TPR_median over (α, θ_pass)

# %%
heat_tpr = agg.pivot(index="theta_pass", columns="alpha", values="median_tpr")
heat_fpr = agg.pivot(index="theta_pass", columns="alpha", values="mean_fpr")

fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))
im0 = axes[0].imshow(
    heat_tpr.values,
    aspect="auto",
    origin="lower",
    extent=(
        heat_tpr.columns.min(),
        heat_tpr.columns.max(),
        heat_tpr.index.min(),
        heat_tpr.index.max(),
    ),
    cmap="viridis",
    vmin=0.0,
    vmax=1.0,
)
axes[0].set_title("TPR_median")
axes[0].set_xlabel(r"$\alpha$")
axes[0].set_ylabel(r"$\theta_{pass}$")
fig.colorbar(im0, ax=axes[0])

im1 = axes[1].imshow(
    heat_fpr.values,
    aspect="auto",
    origin="lower",
    extent=(
        heat_fpr.columns.min(),
        heat_fpr.columns.max(),
        heat_fpr.index.min(),
        heat_fpr.index.max(),
    ),
    cmap="magma",
    vmin=0.0,
    vmax=max(0.2, float(heat_fpr.values.max())),
)
axes[1].set_title("FPR_mean")
axes[1].set_xlabel(r"$\alpha$")
axes[1].set_ylabel(r"$\theta_{pass}$")
fig.colorbar(im1, ax=axes[1])

if recommended:
    for ax in axes:
        ax.scatter(
            [recommended["alpha"]],
            [recommended["theta_pass"]],
            marker="x",
            color="red",
            label="recommended",
        )
        ax.legend(loc="upper right", fontsize=8)

fig.tight_layout()
plt.show()

# %% [markdown]
# ## Take-away
#
# - The TPR vs α line at the recommended θ tells us whether the F0 component
#   (lower α) actually helps. If TPR is essentially flat across α, F0 is
#   contributing little and the cosine-only configuration (`α=1.0`) would
#   work just as well.
# - The FPR rise at low α (high β) shows whether F0 alone over-accepts other
#   speakers — a real risk for same-pitch interlocutors.
# - The recommended marker is the (α, θ) pair that maximises TPR_median
#   under the FPR ≤ 5 % budget; comparing it to D-005's `α=0.8` gives a
#   data-driven endorsement (or revision) of that initial split.
#
# Re-run with `python scripts/calibrate_alpha_beta.py --from-csv` after
# changing the recommendation policy to refresh the summary without
# re-running the pipeline.
