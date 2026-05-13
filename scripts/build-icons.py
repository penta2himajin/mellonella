#!/usr/bin/env python3
"""Render PNG + favicon.ico from assets/icon.svg.

Run from repo root:
    python3 scripts/build-icons.py

Outputs (regenerated each run):
    assets/icon-dark.svg          black fill, for light backgrounds
    assets/icon-light.svg         white fill, for dark backgrounds
    assets/icon-{32,64,128,256,512,1024}.png    dark-on-transparent
    assets/favicon.ico            16/32/48 multi-resolution
"""
from __future__ import annotations

import io
from pathlib import Path

import cairosvg
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
ASSETS = ROOT / "assets"
MASTER = ASSETS / "icon.svg"

PNG_SIZES = (32, 64, 128, 256, 512, 1024)
ICO_SIZES = (16, 32, 48)
DARK = "#111111"
LIGHT = "#f5f5f5"


def render_png(svg: str, size: int) -> bytes:
    return cairosvg.svg2png(
        bytestring=svg.encode("utf-8"),
        output_width=size,
        output_height=size,
    )


def main() -> None:
    master = MASTER.read_text(encoding="utf-8")
    if "currentColor" not in master:
        raise SystemExit(f"{MASTER} must use currentColor as the fill/stroke")

    dark_svg = master.replace("currentColor", DARK)
    light_svg = master.replace("currentColor", LIGHT)
    (ASSETS / "icon-dark.svg").write_text(dark_svg, encoding="utf-8")
    (ASSETS / "icon-light.svg").write_text(light_svg, encoding="utf-8")

    for size in PNG_SIZES:
        out = ASSETS / f"icon-{size}.png"
        out.write_bytes(render_png(dark_svg, size))
        print(f"wrote {out.relative_to(ROOT)}")

    frames = [
        Image.open(io.BytesIO(render_png(dark_svg, s))).convert("RGBA")
        for s in ICO_SIZES
    ]
    ico_path = ASSETS / "favicon.ico"
    frames[0].save(ico_path, format="ICO", sizes=[(s, s) for s in ICO_SIZES])
    print(f"wrote {ico_path.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
