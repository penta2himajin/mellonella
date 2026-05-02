"""Minimal CLI entry point for the Phase 1 PoC.

Subcommands:
* ``enroll``   build an enrollment.json from a clean recording
* ``process``  run the gating pipeline against an utterance
* ``info``     print the active configuration

Audio I/O uses `soundfile`. The heavy model wrappers are imported lazily,
so `--help` and `info` work without torch installed.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

from .config import Config


def _read_audio(path: Path) -> tuple[np.ndarray, int]:
    import soundfile as sf

    audio, sr = sf.read(str(path), dtype="float32", always_2d=False)
    if audio.ndim == 2:
        audio = audio.mean(axis=1)
    return np.asarray(audio, dtype=np.float32), int(sr)


def _write_audio(path: Path, audio: np.ndarray, sample_rate: int) -> None:
    import soundfile as sf

    sf.write(str(path), audio, sample_rate)


def cmd_enroll(args: argparse.Namespace) -> int:
    from .pipeline import PipelineComponents, enroll_from_recording

    config = Config()
    components = PipelineComponents.build_default(config)
    audio, sr = _read_audio(args.input)
    pool = enroll_from_recording(audio, sr, config, components)
    pool.save(args.output)
    print(
        f"Wrote enrollment to {args.output}: "
        f"{len(pool.anchors)} anchors, "
        f"f0_mu={pool.metadata.f0_mu:.1f} Hz, "
        f"f0_sigma={pool.metadata.f0_sigma:.1f} Hz"
    )
    return 0


def cmd_process(args: argparse.Namespace) -> int:
    from .enrollment import EmbeddingPool
    from .pipeline import PipelineComponents, process_offline

    config = Config()
    components = PipelineComponents.build_default(config)
    pool = EmbeddingPool.load(args.enrollment, config.gating)
    audio, sr = _read_audio(args.input)
    out = process_offline(audio, sr, pool, config, components)
    _write_audio(args.output, out, config.audio.output_sr)
    print(
        f"Wrote {args.output} ({out.size / config.audio.output_sr:.2f}s @ {config.audio.output_sr}Hz)"
    )
    return 0


def cmd_info(_: argparse.Namespace) -> int:
    config = Config()
    payload = {
        "audio": {
            "output_sr": config.audio.output_sr,
            "sv_sr": config.audio.sv_sr,
            "frame_ms": config.audio.frame_ms,
            "sv_window_sec": config.audio.sv_window_sec,
            "sv_update_ms": config.audio.sv_update_ms,
        },
        "gating": {
            "alpha": config.gating.alpha,
            "beta": config.gating.beta,
            "theta_pass": config.gating.theta_pass,
            "theta_learn": config.gating.theta_learn,
            "theta_f0": config.gating.theta_f0,
            "hangover_ms": config.gating.hangover_ms,
            "attack_ms": config.gating.attack_ms,
            "release_ms": config.gating.release_ms,
            "anchor_distance_threshold": config.gating.anchor_distance_threshold,
            "anchor_reset_threshold": config.gating.anchor_reset_threshold,
            "auto_learn_max_size": config.gating.auto_learn_max_size,
        },
    }
    json.dump(payload, sys.stdout, indent=2)
    sys.stdout.write("\n")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="mellonella-poc")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_enroll = sub.add_parser("enroll", help="build an enrollment pool from a recording")
    p_enroll.add_argument(
        "--input", type=Path, required=True, help="clean target speaker recording"
    )
    p_enroll.add_argument("--output", type=Path, required=True, help="enrollment json path")
    p_enroll.set_defaults(func=cmd_enroll)

    p_process = sub.add_parser("process", help="run the gating pipeline against an utterance")
    p_process.add_argument("--enrollment", type=Path, required=True)
    p_process.add_argument("--input", type=Path, required=True)
    p_process.add_argument("--output", type=Path, required=True)
    p_process.set_defaults(func=cmd_process)

    p_info = sub.add_parser("info", help="print default configuration as JSON")
    p_info.set_defaults(func=cmd_info)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
