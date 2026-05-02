"""CLI smoke tests that don't require model dependencies."""

from __future__ import annotations

import io
import json
from contextlib import redirect_stdout

import pytest

from mellonella_poc.cli import build_parser, main


def test_help_does_not_require_models(capsys):
    parser = build_parser()
    with pytest.raises(SystemExit) as exc:
        parser.parse_args(["--help"])
    assert exc.value.code == 0
    captured = capsys.readouterr()
    assert "mellonella-poc" in captured.out
    assert "enroll" in captured.out
    assert "process" in captured.out


def test_info_subcommand_prints_config():
    buf = io.StringIO()
    with redirect_stdout(buf):
        rc = main(["info"])
    assert rc == 0
    payload = json.loads(buf.getvalue())
    assert payload["audio"]["output_sr"] == 48_000
    assert payload["gating"]["theta_pass"] < payload["gating"]["theta_learn"]
