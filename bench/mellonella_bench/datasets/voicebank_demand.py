"""Download + verify VoiceBank+DEMAND test set (Scenario 1 NS quality).

License:
    VoiceBank (VCTK)  CC BY 4.0 / ODC-By 1.0
    DEMAND            CC BY-SA 3.0  (ShareAlike applies to derived data)

The official mirror lives at https://datashare.ed.ac.uk/. Sizes:
    clean_testset_wav.zip  ~290 MB, 824 utterances, 2 unseen speakers
    noisy_testset_wav.zip  ~340 MB, 824 utterances, 5 noise types × 4 SNRs

This module only handles fetch + integrity. Mixing into Scenario 1 input is
done in ``mellonella_bench.scenarios.scenario_1``.
"""

from __future__ import annotations

import argparse
import urllib.request
from pathlib import Path

from .common import DatasetSpec, default_data_dir, ensure_clean_dir, verify_archive

CLEAN = DatasetSpec(
    name="voicebank_demand_clean_testset",
    url="https://datashare.ed.ac.uk/bitstream/handle/10283/2791/clean_testset_wav.zip",
    archive_sha256="",  # filled in once we've fetched once and recorded the hash
    license="CC BY 4.0 / ODC-By 1.0",
    notes="VoiceBank-DEMAND clean test split (824 utterances, 2 unseen speakers).",
)

NOISY = DatasetSpec(
    name="voicebank_demand_noisy_testset",
    url="https://datashare.ed.ac.uk/bitstream/handle/10283/2791/noisy_testset_wav.zip",
    archive_sha256="",
    license="CC BY-SA 3.0",
    notes="VoiceBank-DEMAND noisy test split (824 utterances, 5 noise × 4 SNR).",
)


def download_archive(spec: DatasetSpec, target: Path) -> Path:
    """Stream ``spec.url`` to ``target``. Skips if the file already exists."""
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists():
        return target
    tmp = target.with_suffix(target.suffix + ".part")
    print(f"[voicebank_demand] downloading {spec.name} → {target}")
    with urllib.request.urlopen(spec.url) as src, tmp.open("wb") as dst:  # noqa: S310 - HTTPS only
        while True:
            chunk = src.read(1 << 20)
            if not chunk:
                break
            dst.write(chunk)
    tmp.rename(target)
    if spec.archive_sha256:
        verify_archive(target, spec.archive_sha256)
    return target


def extract_zip(archive: Path, dest: Path) -> Path:
    """Extract ``archive`` (zip) into ``dest``. Idempotent: skips if dest exists and is non-empty."""
    import zipfile

    if dest.exists() and any(dest.iterdir()):
        return dest
    ensure_clean_dir(dest)
    with zipfile.ZipFile(archive) as zf:
        zf.extractall(dest)
    return dest


def fetch(data_dir: Path | None = None) -> dict[str, Path]:
    """Download + extract the test set. Returns ``{name: extracted_path}``."""
    root = data_dir if data_dir is not None else default_data_dir() / "voicebank_demand"
    root.mkdir(parents=True, exist_ok=True)
    out: dict[str, Path] = {}
    for spec in (CLEAN, NOISY):
        archive = download_archive(spec, root / f"{spec.name}.zip")
        out[spec.name] = extract_zip(archive, root / spec.name)
    return out


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Fetch the VoiceBank+DEMAND test split.")
    parser.add_argument("--data-dir", type=Path, default=None)
    args = parser.parse_args(argv)
    paths = fetch(args.data_dir)
    for name, path in paths.items():
        print(f"  {name}: {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
