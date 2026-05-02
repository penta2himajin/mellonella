# Phase 1 notebooks

Threshold validation and ad-hoc analysis notebooks for the Phase 1 PoC.
Files are stored as **jupytext-style `.py`** (each `# %%` marker is a cell)
so they review cleanly in PRs and lint with ruff. Convert to `.ipynb` with
jupytext when you want a runnable notebook.

## Setup

```bash
pip install -e poc[models,notebook]
```

`[notebook]` adds jupytext / matplotlib / pandas / ipykernel; `[models]`
brings in the actual ML stack (torch/speechbrain/silero-vad/deepfilternet).

## Running

```bash
cd poc/notebooks
jupytext --to notebook 01_threshold_sweep.py
jupyter nbconvert --execute --to html 01_threshold_sweep.ipynb
```

Or open the `.py` file directly in VS Code / JupyterLab — both render the
`# %%` markers as cells natively.

## Test data

By default the notebooks use `librosa`'s LibriSpeech samples
(`libri1` = target speaker, `libri2` = other). To use real recordings,
set `MELLONELLA_NOTEBOOK_DATA` to a directory containing
`target_speaker.wav` and `other_speaker.wav` (each ≥ 60 s of clean speech).

## Index

| Notebook | What it does |
|---|---|
| [01_threshold_sweep.py](01_threshold_sweep.py) | θ_pass sweep on target vs other speaker; plots TPR/FPR trade-off and recommends an operating point |
