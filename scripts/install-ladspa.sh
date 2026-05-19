#!/usr/bin/env bash
# Install the mellonella LADSPA plugin to ~/.ladspa for per-user use.
# Run `cargo build -p mellonella-ladspa --release` first.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
ARTIFACT="$REPO_ROOT/rust/target/release/libmellonella_ladspa.so"
TARGET_DIR="${LADSPA_INSTALL_DIR:-$HOME/.ladspa}"
TARGET="$TARGET_DIR/libmellonella_ladspa.so"

if [[ ! -f "$ARTIFACT" ]]; then
  echo "error: build artefact not found at $ARTIFACT" >&2
  echo "       run 'cargo build -p mellonella-ladspa --release' from $REPO_ROOT/rust first." >&2
  exit 1
fi

mkdir -p "$TARGET_DIR"
install -m 0755 "$ARTIFACT" "$TARGET"
echo "installed → $TARGET"

case ":${LADSPA_PATH:-}:" in
  *":$TARGET_DIR:"*) ;;
  *)
    echo
    echo "note: \$LADSPA_PATH does not contain $TARGET_DIR."
    echo "      add to your shell rc:"
    echo "        export LADSPA_PATH=\"$TARGET_DIR:\${LADSPA_PATH:-/usr/lib/ladspa}\""
    ;;
esac
