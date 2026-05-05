# Contributing to mellonella

Thanks for your interest in mellonella. This is a single-developer
research project (Phase 1 PoC stage); contributions are welcome but
review may be slow.

## Status

The project is at **Phase 1 PoC**: API, thresholds, and pipeline
internals are still moving. Backwards compatibility is **not**
guaranteed until a 1.0 release.

The Python implementation under [`poc/`](poc/) is the validation
sandbox. The Rust + ONNX Runtime port (Phase 3) is future work and
no contributions are expected for it yet.

## Filing issues

Please use the issue templates:

- **Bug report**: include OS / Python version, model versions,
  pipeline configuration, and reproduction steps. Audio attachments
  are appreciated when relevant — clean speech only, please remove
  any PII before attaching.
- **Feature request**: describe the use case before proposing a
  solution.

For security issues, see [SECURITY.md](SECURITY.md) instead.

## Filing pull requests

Before opening a PR:

1. **Open an issue first** for anything beyond a typo or trivial fix.
   The architecture is documented under [`docs/`](docs/); please read
   [`docs/decisions.md`](docs/decisions.md) before proposing
   structural changes.
2. **Run the local test suite** (`pytest` from `poc/` or `bench/`).
   See [`poc/README.md`](poc/README.md) and
   [`bench/README.md`](bench/README.md) for setup.
3. **Match the existing code style**: `ruff` is the formatter and
   linter; `mypy` is the type checker. The CI workflow under
   [`.github/workflows/ci.yml`](.github/workflows/ci.yml) runs both.
4. **Use Conventional Commits** for the commit subject (`feat:`,
   `fix:`, `docs:`, `chore:`, etc.). The PR template covers the body.
5. **Keep PRs scoped**. One concern per PR — split unrelated changes.

## Code style

- Python: `ruff` for formatting + linting (config in `pyproject.toml`),
  `mypy --strict` for type-checking.
- All public APIs must have type hints.
- New public functions need at least one test.
- No emojis in code or commit messages.

## Licensing

By contributing, you agree that your contribution will be licensed
under the [Apache License 2.0](LICENSE), the same as the rest of the
project. You retain copyright on your contributions.

There is no CLA at this time. If a CLA is added in the future,
existing contributions will not be retroactively required to sign it.

## Review expectations

- This is a single-developer project; review turnaround is best-effort.
- Reasonable to expect 1-2 weeks for a substantive PR review.
- Trivial fixes (typos, broken links, etc.) often land within days.
- If a PR sits untouched for more than a month, feel free to ping
  with a comment.
