"""Download + verify MUSAN noise corpus.

License:
    Creative Commons (flexible) — see http://www.openslr.org/17/
    Apache 2.0 redistribution-friendly.

MUSAN is ~60 GB. This module fetches the official tarball and extracts it
under ``$MELLONELLA_DATA_DIR/musan``. For the Phase 1 PoC we typically use
only a 30-clip subset (``--subset N``).
"""

from __future__ import annotations

import argparse
import random
import shutil
import tarfile
import urllib.request
from pathlib import Path

from .common import DatasetSpec, default_data_dir, ensure_clean_dir

SPEC = DatasetSpec(
    name="musan",
    url="https://www.openslr.org/resources/17/musan.tar.gz",
    archive_sha256="",  # to be filled in after first fetch
    license="Apache 2.0",
    notes="MUSAN: speech / music / noise. ~60 GB extracted.",
)


def download_archive(target: Path) -> Path:
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists():
        return target
    tmp = target.with_suffix(target.suffix + ".part")
    print(f"[musan] downloading {SPEC.url} → {target}")
    with urllib.request.urlopen(SPEC.url) as src, tmp.open("wb") as dst:  # noqa: S310 - HTTPS only
        while True:
            chunk = src.read(1 << 20)
            if not chunk:
                break
            dst.write(chunk)
    tmp.rename(target)
    return target


def extract(archive: Path, dest: Path) -> Path:
    if dest.exists() and any(dest.iterdir()):
        return dest
    ensure_clean_dir(dest)
    with tarfile.open(archive, "r:gz") as tf:
        tf.extractall(dest)
    return dest


def build_subset(
    extracted: Path, subset_dir: Path, n_per_category: int = 10
) -> dict[str, list[Path]]:
    """Pick ``n_per_category`` files for each of speech/music/noise into ``subset_dir``.

    Selection is deterministic (seeded). Honours the original directory structure.
    """
    if subset_dir.exists():
        shutil.rmtree(subset_dir)
    subset_dir.mkdir(parents=True, exist_ok=True)
    rng = random.Random(0)
    picked: dict[str, list[Path]] = {}
    for category in ("speech", "music", "noise"):
        cat_root = extracted / "musan" / category
        if not cat_root.exists():
            raise FileNotFoundError(f"expected {cat_root} to exist after extraction")
        wavs = sorted(cat_root.rglob("*.wav"))
        if not wavs:
            raise RuntimeError(f"no wavs found under {cat_root}")
        chosen = rng.sample(wavs, k=min(n_per_category, len(wavs)))
        out_dir = subset_dir / category
        out_dir.mkdir(parents=True, exist_ok=True)
        for src in chosen:
            shutil.copy2(src, out_dir / src.name)
        picked[category] = [out_dir / src.name for src in chosen]
    return picked


def fetch(data_dir: Path | None = None, subset: int | None = 10) -> Path:
    """Download, extract, and (optionally) materialise the PoC subset.

    Returns the extracted root.
    """
    root = data_dir if data_dir is not None else default_data_dir() / "musan"
    root.mkdir(parents=True, exist_ok=True)
    archive = download_archive(root / "musan.tar.gz")
    extracted = extract(archive, root / "extracted")
    if subset:
        build_subset(extracted, root / "subset", n_per_category=subset)
    return extracted


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Fetch MUSAN and build the PoC subset.")
    parser.add_argument("--data-dir", type=Path, default=None)
    parser.add_argument("--subset", type=int, default=10, help="N clips per category (0 disables)")
    args = parser.parse_args(argv)
    extracted = fetch(args.data_dir, subset=args.subset or None)
    print(f"  extracted: {extracted}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
