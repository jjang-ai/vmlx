#!/usr/bin/env bash
# Local install: package from existing bundled-python (no full rebundle).
# Prefer from repo root:  make install
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PANEL_DIR="$(dirname "$SCRIPT_DIR")"
REPO_DIR="$(dirname "$PANEL_DIR")"

cd "$REPO_DIR"
echo "==> build-and-install: using Makefile install path (staged .app, no DMG unless needed)"
echo "    Tip: make help  |  resume: make app  |  engine edits: make engine-and-install"
echo

if [[ -x "$PANEL_DIR/bundled-python/python/bin/python3" ]]; then
  make install
else
  echo "==> bundled-python missing — running first-dmg then install (one-time full setup)"
  make first-dmg
  make install
fi
