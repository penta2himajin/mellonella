# Security Policy

## Supported versions

| Version | Status |
|---|---|
| Phase 1 PoC (current) | Best-effort; no production guarantees |

The project is pre-1.0 and not intended for production use. There are
no LTS branches.

## Reporting a vulnerability

If you discover a security issue, please **do not** open a public
issue. Instead, report it privately through GitHub's
[Security Advisories](https://github.com/penta2himajin/mellonella/security/advisories/new)
interface.

Please include:

- A description of the issue and its impact.
- Reproduction steps or a proof-of-concept.
- The affected commit / version.
- Your contact information for follow-up.

Acknowledgement: best-effort, typically within a week. Fix turnaround
depends on severity and the single-developer nature of the project.

## Scope

- The Python PoC (`poc/`), evaluation harness (`bench/`), helper
  scripts (`scripts/`), and CI workflows (`.github/workflows/`) are
  in scope.
- Bundled pretrained models (DFN3, silero-vad, ECAPA-TDNN) are out of
  scope; please report issues with those upstream.
