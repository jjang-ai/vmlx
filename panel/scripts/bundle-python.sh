#!/bin/bash
set -euo pipefail

# Build a relocatable Python environment with all vmlx-engine dependencies.
# Run once on dev machine before `npm run dist`.
# Output: panel/bundled-python/python/ (~1-2 GB)

PYTHON_VERSION="3.12.12"
BUILD_DATE="20260211"
ARCH="aarch64-apple-darwin"
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PANEL_DIR="$(dirname "$SCRIPT_DIR")"
REPO_DIR="$(dirname "$PANEL_DIR")"
FINAL_BUNDLE_DIR="$PANEL_DIR/bundled-python"
BUNDLE_DIR="$PANEL_DIR/.bundled-python.staging.$$"
PREVIOUS_BUNDLE_DIR="$PANEL_DIR/.bundled-python.previous.$$"
BUNDLE_PUBLISHED=0
STANDALONE_TARBALL=""
JANG_LOCAL="${VMLX_JANG_TOOLS_SOURCE:-${VMLINUX_JANG_TOOLS_SOURCE:-$HOME/jang/jang-tools}}"
JANG_MIN_VERSION="2.5.48"
JANG_SOURCE_COMMIT=""
JANG_SOURCE_VERSION=""
VMLX_SOURCE_COMMIT=""

# A host user-site package can make pip report a dependency as satisfied even
# though it is absent from the relocatable bundle. Keep every install and
# verification action isolated from the build machine's Python configuration.
export PYTHONNOUSERSITE=1
unset PYTHONPATH PYTHONHOME VIRTUAL_ENV

remove_bundle_tree_with_retry() {
  local target="$1"
  local attempt

  # APFS can transiently return ENOTEMPTY while retiring the large, freshly
  # renamed site-packages tree.  The EXIT trap's immediate second rm has
  # consistently completed that same cleanup, so make the retry explicit and
  # bounded instead of failing a verified build after it has been published.
  # This helper is only called with this script's PID-scoped staging/previous
  # paths.  Exhausting the bounded attempts remains a hard release failure.
  for attempt in 1 2 3; do
    if /bin/rm -rf "$target" \
      && [ ! -e "$target" ] \
      && [ ! -L "$target" ]; then
      return 0
    fi
    if [ "$attempt" -lt 3 ]; then
      echo "WARNING: retrying transient bundled-Python cleanup ($attempt/3): $target" >&2
      /bin/sleep 1
    fi
  done

  echo "ERROR: bundled-Python cleanup did not converge after 3 attempts: $target" >&2
  return 1
}

cleanup_bundle_build() {
  local status=$?
  trap - EXIT
  if [ -n "$STANDALONE_TARBALL" ]; then
    rm -f "$STANDALONE_TARBALL"
  fi
  if [ "$BUNDLE_PUBLISHED" -ne 1 ]; then
    remove_bundle_tree_with_retry "$BUNDLE_DIR" || true
  fi
  if [ -e "$PREVIOUS_BUNDLE_DIR" ]; then
    if [ ! -e "$FINAL_BUNDLE_DIR" ] && [ ! -L "$FINAL_BUNDLE_DIR" ]; then
      mv "$PREVIOUS_BUNDLE_DIR" "$FINAL_BUNDLE_DIR" || true
    else
      remove_bundle_tree_with_retry "$PREVIOUS_BUNDLE_DIR" || true
    fi
  fi
  exit "$status"
}
trap cleanup_bundle_build EXIT

echo "==> Bundling Python $PYTHON_VERSION for standalone vMLX distribution"

check_local_vmlx_source_clean() {
  if ! git -C "$REPO_DIR" rev-parse --is-inside-work-tree >/dev/null 2>&1; then
    echo "ERROR: RELEASE BLOCKED — vMLX source is not a Git checkout: $REPO_DIR" >&2
    exit 1
  fi
  if [ "${VMLX_ALLOW_DIRTY_SOURCE:-${VMLINUX_ALLOW_DIRTY_SOURCE:-0}}" = "1" ]; then
    echo "    WARNING: VMLX_ALLOW_DIRTY_SOURCE=1 — bundling dirty vMLX source" >&2
  else
    local package_status
    package_status="$(
      git -C "$REPO_DIR" status --porcelain --untracked-files=all -- \
        pyproject.toml uv.lock vmlx_engine \
        panel/package.json panel/package-lock.json panel/scripts panel/src
    )"
    if [ -n "$package_status" ]; then
      echo "ERROR: RELEASE BLOCKED — vMLX package source is dirty: $REPO_DIR" >&2
      echo "       Release provenance cannot identify uncommitted runtime or packaging code." >&2
      echo "       Package-scope status:" >&2
      printf '%s\n' "$package_status" >&2
      echo >&2
      echo "       Commit/stash/drop those changes, or set VMLX_ALLOW_DIRTY_SOURCE=1" >&2
      echo "       only for local smoke builds." >&2
      exit 1
    fi
  fi
  VMLX_SOURCE_COMMIT="$(git -C "$REPO_DIR" rev-parse HEAD)"
  echo "    vMLX source commit: $VMLX_SOURCE_COMMIT"
}

check_local_jang_source_clean() {
  if [ ! -f "$JANG_LOCAL/pyproject.toml" ]; then
    return 0
  fi
  if ! git -C "$JANG_LOCAL" rev-parse --is-inside-work-tree >/dev/null 2>&1; then
    if [ "${VMLX_ALLOW_UNVERSIONED_JANG_SOURCE:-${VMLINUX_ALLOW_UNVERSIONED_JANG_SOURCE:-0}}" = "1" ]; then
      echo "    WARNING: VMLX_ALLOW_UNVERSIONED_JANG_SOURCE=1 — JANG provenance is unavailable" >&2
      return 0
    fi
    echo "ERROR: RELEASE BLOCKED — local jang-tools source is not a Git checkout: $JANG_LOCAL" >&2
    echo "       Release bundles require an exact source commit. Set" >&2
    echo "       VMLX_ALLOW_UNVERSIONED_JANG_SOURCE=1 only for local smoke builds." >&2
    exit 1
  fi
  if [ "${VMLX_ALLOW_DIRTY_JANG_SOURCE:-${VMLINUX_ALLOW_DIRTY_JANG_SOURCE:-0}}" = "1" ]; then
    echo "    WARNING: VMLX_ALLOW_DIRTY_JANG_SOURCE=1 — bundling dirty jang-tools" >&2
  else
    local package_status
    package_status="$(
      git -C "$JANG_LOCAL" status --porcelain --untracked-files=all -- \
        pyproject.toml jang_tools
    )"
    if [ -n "$package_status" ]; then
      echo "ERROR: RELEASE BLOCKED — local jang-tools package source is dirty: $JANG_LOCAL" >&2
      echo "       Release bundles must not silently package tracked or untracked JANG runtime changes." >&2
      echo "       Package-scope status:" >&2
      printf '%s\n' "$package_status" >&2
      echo >&2
      echo "       Commit/stash/drop those changes, point VMLX_JANG_TOOLS_SOURCE at a clean" >&2
      echo "       checkout, or set VMLX_ALLOW_DIRTY_JANG_SOURCE=1 only for local smoke builds." >&2
      exit 1
    fi
  fi

  # Dirty is not the only way to ship the wrong runtime. A CLEAN checkout that
  # is simply behind origin/main passes every check above and silently bundles
  # a stale JANG — that is how a release went out with the pre-fix DSV4 decode
  # path, from a long-lived feature branch that looked perfectly healthy.
  #
  # VMLX_JANG_TOOLS_RELEASE_PIN is the release-grade alternative to the
  # generic stale bypass: it names ONE exact full commit that this release
  # deliberately bundles even though origin/main has moved on (e.g. 1.6.44
  # pins 3a5101e — the runtime already shipped as jang 2.5.46 — while the
  # unreleased Aug-27 sync on origin/main awaits its own 2.5.47 release).
  # The pin only passes when HEAD matches it exactly, so it cannot silently
  # bless an arbitrary stale checkout the way ALLOW_STALE does.
  if [ -n "${VMLX_JANG_TOOLS_RELEASE_PIN:-}" ]; then
    _jang_head="$(git -C "$JANG_LOCAL" rev-parse HEAD)"
    if [ "$_jang_head" != "$VMLX_JANG_TOOLS_RELEASE_PIN" ]; then
      echo "ERROR: RELEASE BLOCKED — jang-tools HEAD $_jang_head does not match" >&2
      echo "       VMLX_JANG_TOOLS_RELEASE_PIN=$VMLX_JANG_TOOLS_RELEASE_PIN" >&2
      exit 1
    fi
    echo "    jang-tools release pin matched: $_jang_head (freshness gate satisfied by exact pin)" >&2
  elif [ "${VMLX_ALLOW_STALE_JANG_SOURCE:-0}" = "1" ]; then
    echo "    WARNING: VMLX_ALLOW_STALE_JANG_SOURCE=1 — not checking jang-tools freshness" >&2
  elif git -C "$JANG_LOCAL" rev-parse --verify --quiet origin/main >/dev/null; then
    local behind
    behind="$(git -C "$JANG_LOCAL" rev-list --count HEAD..origin/main 2>/dev/null || echo 0)"
    if [ "${behind:-0}" -gt 0 ]; then
      echo "ERROR: RELEASE BLOCKED — jang-tools source is $behind commit(s) behind origin/main" >&2
      echo "       source: $JANG_LOCAL" >&2
      echo "       branch: $(git -C "$JANG_LOCAL" rev-parse --abbrev-ref HEAD)" >&2
      echo "       Bundling a stale checkout ships an old JANG runtime while every" >&2
      echo "       version string still reads correct. Point VMLX_JANG_TOOLS_SOURCE at a" >&2
      echo "       clean checkout at origin/main, or set VMLX_ALLOW_STALE_JANG_SOURCE=1" >&2
      echo "       for local smoke builds only." >&2
      exit 1
    fi
  else
    echo "    WARNING: no origin/main ref in $JANG_LOCAL — cannot verify jang-tools freshness" >&2
  fi

  JANG_SOURCE_COMMIT="$(git -C "$JANG_LOCAL" rev-parse HEAD)"
  JANG_SOURCE_VERSION="$(
    sed -n 's/^version[[:space:]]*=[[:space:]]*"\([^"]*\)".*/\1/p' \
      "$JANG_LOCAL/pyproject.toml" | head -1
  )"
  if [ -z "$JANG_SOURCE_VERSION" ]; then
    echo "ERROR: RELEASE BLOCKED — cannot read JANG version from $JANG_LOCAL/pyproject.toml" >&2
    exit 1
  fi
  echo "    JANG source commit: $JANG_SOURCE_COMMIT"
  echo "    JANG source version: $JANG_SOURCE_VERSION"
}

write_bundle_provenance() {
  local installed_jang_version
  installed_jang_version="$(
    "$PYTHON" -c 'from importlib.metadata import version; print(version("jang"))'
  )"
  if [ -n "$JANG_SOURCE_VERSION" ] && [ "$installed_jang_version" != "$JANG_SOURCE_VERSION" ]; then
    echo "ERROR: RELEASE BLOCKED — installed JANG version does not match source" >&2
    echo "       source version   : $JANG_SOURCE_VERSION" >&2
    echo "       installed version: $installed_jang_version" >&2
    exit 1
  fi
  "$PYTHON" - "$installed_jang_version" "$JANG_MIN_VERSION" <<'PY'
import sys
from packaging.version import Version

installed, minimum = map(Version, sys.argv[1:3])
if installed < minimum:
    raise SystemExit(
        f"RELEASE BLOCKED — bundled JANG {installed} is below required {minimum}"
    )
PY
  BUNDLE_PROVENANCE_PATH="$BUNDLE_DIR/vmlx-bundle-provenance.json" \
  BUNDLE_VMLX_COMMIT="$VMLX_SOURCE_COMMIT" \
  BUNDLE_VMLX_VERSION="$("$PYTHON" -c 'import vmlx_engine; print(vmlx_engine.__version__)')" \
  BUNDLE_JANG_COMMIT="${JANG_SOURCE_COMMIT:-UNVERSIONED}" \
  BUNDLE_JANG_VERSION="$installed_jang_version" \
  BUNDLE_JANG_RELEASE_PIN="${VMLX_JANG_TOOLS_RELEASE_PIN:-}" \
  BUNDLE_MLX_WHEEL_PLATFORM="$MLX_WHEEL_PLATFORM" \
  "$PYTHON" - <<'PY'
import json
import os
from pathlib import Path

payload = {
    "schema_version": 1,
    "vmlx": {
        "commit": os.environ["BUNDLE_VMLX_COMMIT"],
        "version": os.environ["BUNDLE_VMLX_VERSION"],
    },
    "jang": {
        "commit": os.environ["BUNDLE_JANG_COMMIT"],
        "version": os.environ["BUNDLE_JANG_VERSION"],
    },
    "mlx_wheel_platform": os.environ["BUNDLE_MLX_WHEEL_PLATFORM"],
}
# A pinned release deliberately bundles one exact jang commit even though
# origin/main has moved on. Record the pin so the verifier can check the
# bundle against the INTENDED commit instead of a moving upstream ref.
release_pin = os.environ.get("BUNDLE_JANG_RELEASE_PIN", "")
if release_pin:
    payload["jang"]["release_pin"] = release_pin
Path(os.environ["BUNDLE_PROVENANCE_PATH"]).write_text(
    json.dumps(payload, indent=2, sort_keys=True) + "\n",
    encoding="utf-8",
)
PY
  echo "    wrote bundle provenance: $BUNDLE_DIR/vmlx-bundle-provenance.json"
}

check_local_vmlx_source_clean
check_local_jang_source_clean

# Build in a private sibling directory. The currently verified bundle remains
# untouched unless this entire build and verifier both succeed.
remove_bundle_tree_with_retry "$BUNDLE_DIR"
remove_bundle_tree_with_retry "$PREVIOUS_BUNDLE_DIR"
mkdir -p "$BUNDLE_DIR"

# Download python-build-standalone (Astral's relocatable Python builds)
TARBALL="cpython-${PYTHON_VERSION}+${BUILD_DATE}-${ARCH}-install_only.tar.gz"
URL="https://github.com/astral-sh/python-build-standalone/releases/download/${BUILD_DATE}/${TARBALL}"
# BSD mktemp randomizes only trailing Xs; adding .tar.gz creates a literal
# shared filename and makes concurrent flavor builds collide.
STANDALONE_TARBALL="$(mktemp "${TMPDIR:-/tmp}/vmlx-python-standalone.XXXXXX")"

restore_python_runtime_files() {
  local RESTORE_TMP
  RESTORE_TMP="$(mktemp -d)"
  tar xzf "$STANDALONE_TARBALL" -C "$RESTORE_TMP" \
    "./python/bin/python3" \
    "./python/bin/python3.12" \
    "./python/lib/libpython3.12.dylib"
  mkdir -p "$BUNDLE_DIR/python/bin" "$BUNDLE_DIR/python/lib"
  cp -f "$RESTORE_TMP/python/bin/python3.12" "$BUNDLE_DIR/python/bin/python3.12"
  cp -f "$RESTORE_TMP/python/lib/libpython3.12.dylib" "$BUNDLE_DIR/python/lib/libpython3.12.dylib"
  chmod +x "$BUNDLE_DIR/python/bin/python3.12"
  ln -sf python3.12 "$BUNDLE_DIR/python/bin/python3"
  rm -rf "$RESTORE_TMP"
}

echo "==> Downloading Python ${PYTHON_VERSION}..."
curl -L "$URL" -o "$STANDALONE_TARBALL"
tar xzf "$STANDALONE_TARBALL" -C "$BUNDLE_DIR"

PYTHON="$BUNDLE_DIR/python/bin/python3"

# Verify Python works
"$PYTHON" --version

# Upgrade pip
echo "==> Upgrading pip..."
"$PYTHON" -m pip install --upgrade pip

# Install ALL dependencies (lean: no gradio, no dev tools, no pytz).
# mflux requires the standard opencv-python distribution. Install exactly one
# OpenCV wheel: the 4.13 macOS ARM wheel is available, while the newer 4.14
# release currently has no macOS ARM wheel and would trigger a non-reproducible
# source build. OpenCV's packages share the cv2 namespace and must not coexist.
#
MLX_VERSION="0.32.2"
MLX_LM_VERSION="0.31.3"
MLX_VLM_VERSION="0.5.0"
MFLUX_VERSION="0.19.0"
OPENCV_VERSION="4.13.0.92"
MLX_AUDIO_VERSION="0.4.6"
# Keep the downloader/tokenizer dependency on the version qualified with the
# bundled runtime. A lower bound silently upgraded it during a source-only
# rebuild, invalidating the retained dependency comparison.
HUGGINGFACE_HUB_VERSION="1.32.0"
# The MCP SDK was installed as an unpinned "mcp>=1.0.0", so every re-bundle took
# whatever PyPI had latest — which is how the shipped bundle silently moved to a
# MAJOR version (2.0.0) while uv.lock still records 1.26.0. Nothing broke,
# because vmlx_engine/mcp/client.py only uses ClientSession,
# StdioServerParameters, stdio_client and sse_client, all of which exist in both
# — but a future 3.x would land the same way, unchosen and unverified. Pin it to
# the version we actually ship and test, like every other dependency here.
MCP_VERSION="2.0.0"

detect_mlx_wheel_platform() {
  local requested="${VMLX_BUNDLE_MLX_PLATFORM:-${VMLINUX_BUNDLE_MLX_PLATFORM:-compat}}"
  case "$requested" in
    auto|compat|sonoma|sequoia|"")
      echo "macosx_14_0_arm64"
      ;;
    tahoe|native|m5)
      echo "macosx_26_0_arm64"
      ;;
    macosx_*_arm64)
      echo "$requested"
      ;;
    *)
      echo "ERROR: unsupported VMLX_BUNDLE_MLX_PLATFORM=$requested" >&2
      echo "       Use auto, compat, native, or an explicit macosx_*_arm64 wheel tag." >&2
      exit 1
      ;;
  esac
}

# MLX publishes multiple macOS-tagged wheels for the same version. The older
# macosx_14 wheel preserves Sonoma/Sequoia compatibility; the macosx_26 wheel
# carries newer Metal kernels but cannot be loaded on macOS 15 (#169). The
# release DMG script builds both flavors explicitly; this lower-level bundler
# defaults to compat so ad-hoc packaging from Tahoe/M5 hardware is not
# accidentally Sequoia-hostile. Native/Tahoe wheels stay explicit opt-in:
#   VMLX_BUNDLE_MLX_PLATFORM=native ./panel/scripts/bundle-python.sh
# The legacy misspelled VMLINUX_BUNDLE_MLX_PLATFORM is still accepted so older
# local scripts fail closed to their intended wheel flavor instead of silently
# drifting back to compat.
MLX_WHEEL_PLATFORM="$(detect_mlx_wheel_platform)"
WHEELHOUSE="$BUNDLE_DIR/wheelhouse"
mkdir -p "$WHEELHOUSE"
echo "==> Installing MLX $MLX_VERSION wheels for $MLX_WHEEL_PLATFORM..."
"$PYTHON" -m pip download --only-binary=:all: --no-deps \
  --dest "$WHEELHOUSE" \
  --platform "$MLX_WHEEL_PLATFORM" \
  --implementation cp --python-version 312 --abi cp312 \
  "mlx==$MLX_VERSION"
"$PYTHON" -m pip download --only-binary=:all: --no-deps \
  --dest "$WHEELHOUSE" \
  --platform "$MLX_WHEEL_PLATFORM" \
  --implementation py --python-version 312 --abi none \
  "mlx-metal==$MLX_VERSION"
"$PYTHON" -m pip install "$WHEELHOUSE"/mlx-"$MLX_VERSION"-*.whl "$WHEELHOUSE"/mlx_metal-"$MLX_VERSION"-*.whl

echo "==> Installing dependencies..."
"$PYTHON" -m pip install --only-binary=:all: \
  "mlx==$MLX_VERSION" "mlx-lm==$MLX_LM_VERSION" "mlx-vlm==$MLX_VLM_VERSION" \
  "transformers>=4.40.0,<5.13" "tokenizers>=0.19.0" "huggingface-hub==$HUGGINGFACE_HUB_VERSION" \
  "numpy>=1.24.0" "pillow>=10.0.0" \
  "opencv-python==$OPENCV_VERSION" \
  "fastapi>=0.100.0" "uvicorn>=0.23.0" \
  "mcp==$MCP_VERSION" "jsonschema>=4.0.0" "referencing>=0.37.0" \
  "psutil>=5.9.0" "tqdm>=4.66.0" "pyyaml>=6.0" \
  "requests>=2.28.0" "tabulate>=0.9.0" "mlx-embeddings>=0.0.5" \
  "tiktoken>=0.7.0" \
  "soundfile>=0.12" \
  "mflux==$MFLUX_VERSION" \
  "timm>=1.0.20" \
  "dflash==0.1.0" "dflash-mlx==0.1.8" \
  "einops>=0.8.0"  # Kimi K2.6 tokenizer + Nemotron-Omni RADIO/ViT deps
  # dflash: the z-lab DFlash2 draft runtime (Qwen3.8 matched drafter path);
  # dflash-mlx: the Apache-2.0 four-row affine-q4 Metal verify kernel.

restore_python_runtime_files
PYTHON_BIN="$BUNDLE_DIR/python/bin/python3.12"
if [ -f "$PYTHON_BIN" ]; then
  chmod +x "$PYTHON_BIN"
  ln -sf python3.12 "$PYTHON"
fi
if [ ! -x "$PYTHON" ]; then
  echo "ERROR: bundled Python executable disappeared after dependency install: $PYTHON" >&2
  find "$BUNDLE_DIR/python/bin" -maxdepth 1 \( -name 'python*' -o -name 'pip*' \) -ls >&2 || true
  exit 1
fi

# Install the exact MLX-Audio release used by this bundle. Its runtime
# requirements are already part of the dependency transaction above; --no-deps
# prevents a second resolver pass from changing the verified MLX stack.
echo "==> Installing mlx-audio (STT/TTS)..."
"$PYTHON" -m pip install --no-deps "mlx-audio==$MLX_AUDIO_VERSION"
# Install mlx-audio's transitive deps that we don't already have
"$PYTHON" -m pip install \
  librosa sounddevice miniaudio pyloudnorm numba

# Install vmlx-engine from PyPI (the local repo's pyproject.toml has a
# `setuptools.package-dir = engine` declaration that breaks `pip install
# .` here because the panel dir lives at panel/, not engine/. PyPI 1.4.3
# is a clean wheel with the same source so this is equivalent.)
#
# jang_tools — install from LOCAL source path, not PyPI. The published
# `jang` wheel lags the local development of jang_tools.dsv4 modules
# (pool_quant_cache.py, fused_pool_attn.py, fused_pool_attn_kernel.py,
# build_role_codebooks.py landed in jang 2.5.10 but only jang 2.5.9 was
# on PyPI as of v1.5.2 build, causing
# `ModuleNotFoundError: No module named 'jang_tools.dsv4.pool_quant_cache'`
# the moment a DSV4-Flash bundle was loaded). Pinning to local source
# ensures every DMG ships with whatever DSV4 runtime the engine actually
# needs. Falls back to PyPI if the local path doesn't exist (CI builds).
echo "==> Installing vmlx-engine + jang_tools (local source)..."
# Install vmlx-engine from local source so the bundle ships current fixes,
# not the lagging PyPI release. Falls back to PyPI if the local path is
# absent (CI builds). 2026-05-03: previous pin to PyPI vmlx==1.4.3
# silently overwrote shipped F1-F13 fixes on every rebuild.
VMLX_LOCAL="$(cd "$(dirname "$0")/../.." && pwd)"
if [ -f "$VMLX_LOCAL/pyproject.toml" ] && [ -d "$VMLX_LOCAL/vmlx_engine" ]; then
  echo "    using local vmlx at $VMLX_LOCAL"
  # Setuptools can reuse stale wheel scratch under build/lib when rebuilding
  # the local wheel.  Clean only packaging-owned subdirectories: this repo's
  # build/ root also contains tracked release-gate evidence and must never be
  # erased by a bundle operation.
  rm -rf \
    "$VMLX_LOCAL/build/lib" \
    "$VMLX_LOCAL/build"/bdist.* \
    "$VMLX_LOCAL/build"/temp.* \
    "$VMLX_LOCAL/build"/scripts-* \
    "$VMLX_LOCAL"/*.egg-info
  "$PYTHON" -m pip install --force-reinstall --no-deps --no-cache-dir "$VMLX_LOCAL"
else
  echo "    local vmlx missing, falling back to PyPI"
  "$PYTHON" -m pip install --no-deps "vmlx>=1.5.24"
fi
if [ -f "$JANG_LOCAL/pyproject.toml" ]; then
  echo "    using local jang-tools at $JANG_LOCAL"
  "$PYTHON" -m pip install --force-reinstall --no-deps --no-cache-dir "$JANG_LOCAL"
else
  if [ "${VMLX_ALLOW_PYPI_JANG:-${VMLINUX_ALLOW_PYPI_JANG:-0}}" = "1" ]; then
    echo "    local jang-tools missing; VMLX_ALLOW_PYPI_JANG=1 so using PyPI fallback"
    "$PYTHON" -m pip install --no-deps "jang>=2.5.48"
  else
    echo "ERROR: RELEASE BLOCKED — local jang-tools source missing: $JANG_LOCAL" >&2
    echo "       vMLX release builds must bundle the checked-out JANG runtime," >&2
    echo "       not whatever PyPI happens to have. Set VMLX_ALLOW_PYPI_JANG=1" >&2
    echo "       only for non-release CI smoke builds." >&2
    exit 1
  fi
fi
write_bundle_provenance

# Local source installs generate the vmlx/jang console entrypoints after the
# dependency install pass. Relocate their shebangs immediately so a later
# cleanup/import step cannot leave build-machine paths in the packaged app.
for SCRIPT in "$BUNDLE_DIR/python/bin/"vmlx* "$BUNDLE_DIR/python/bin/"jang*; do
  if [ ! -f "$SCRIPT" ]; then
    continue
  fi
  FIRST_LINE="$(LC_ALL=C head -n 1 "$SCRIPT" 2>/dev/null || true)"
  if [[ "$FIRST_LINE" == '#!'*python* ]] && [[ "$FIRST_LINE" != '#!/bin/sh'* ]]; then
    TMP_SCRIPT="$(mktemp "${SCRIPT}.XXXXXX")"
    {
      printf '%s\n' '#!/bin/sh'
      printf '%s\n' "'''exec' \"\$(dirname \"\$0\")/python3\" -B -s \"\$0\" \"\$@\""
      printf '%s\n' "' '''"
      tail -n +2 "$SCRIPT"
    } > "$TMP_SCRIPT"
    chmod --reference="$SCRIPT" "$TMP_SCRIPT" 2>/dev/null || chmod +x "$TMP_SCRIPT"
    mv "$TMP_SCRIPT" "$SCRIPT"
  fi
done

# Clean up to reduce size
echo "==> Cleaning up..."
SITE="$BUNDLE_DIR/python/lib/python3.12/site-packages"

# Python bytecode (regenerated on import)
find "$BUNDLE_DIR" -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
find "$BUNDLE_DIR" -name "*.pyc" -delete 2>/dev/null || true

# Unused stdlib modules
rm -rf "$BUNDLE_DIR/python/lib/python3.12/test"
rm -rf "$BUNDLE_DIR/python/lib/python3.12/ensurepip"
rm -rf "$BUNDLE_DIR/python/lib/python3.12/idlelib"
rm -rf "$BUNDLE_DIR/python/lib/python3.12/tkinter"
rm -rf "$BUNDLE_DIR/python/lib/python3.12/turtle"*
rm -rf "$BUNDLE_DIR/python/share" 2>/dev/null || true
# Unused .so for removed stdlib (tkinter)
rm -f "$BUNDLE_DIR/python/lib/python3.12/lib-dynload/_tkinter"*.so 2>/dev/null || true

# Third-party test suites in site-packages (~80+ MB of data never used at
# runtime).  Keep vmlx_engine/tests: those runtime diagnostics are explicit
# package data and the staged-app parity gate binds the complete engine package
# to the release source.
find "$SITE" -type d -name "tests" \
  ! -path "$SITE/vmlx_engine/tests" \
  -exec rm -rf {} + 2>/dev/null || true
find "$SITE" -type d -name "test" -exec rm -rf {} + 2>/dev/null || true

# Third-party packages can ship agent/skill metadata that is not used at
# runtime and can be mistaken for release-workspace artifacts.
find "$SITE" -type d -name ".agents" -exec rm -rf {} + 2>/dev/null || true

# Packages not needed at runtime (transitive deps / dev-only tools)
# torch/torchvision/torchgen: KEEP (2026-05-03 reversal of earlier strip).
# Earlier comment claimed "vmlx-engine uses MLX, not PyTorch" — true for the
# core LLM path, but VL bundles tell another story:
#   - Nemotron-Omni Stage-1 (default) bridges PT encoders to MLX LLM via
#     `jang_tools.nemotron_omni_chat.OmniChat` which calls
#     `AutoModel.from_pretrained(..., torch_dtype=bfloat16)`.
#   - Qwen3-VL / Qwen3.5-VL: transformers loads `Qwen3VLVideoProcessor`
#     eagerly via `AutoProcessor.from_pretrained`; that class hard-requires
#     torch+torchvision and raises a confusing `ImportError` upstream
#     that gets re-wrapped as "mlx-vlm is required" in mllm.py.
#   - mlx_vlm itself imports torch helpers in some processor paths.
# Stripping torch broke every VL bundle the moment a video processor or
# PT encoder loaded. Keep them. Cost: ~700 MB extra bundle weight.
# pyproject.toml already declares them as hard deps (`torch>=2.3.0`,
# `torchvision>=0.18.0`) — pip install -e . pulls them in cleanly.
# soundfile: KEEP. Required for Nemotron-3-Nano-Omni audio path
# (jang_tools.nemotron_omni_chat → ParakeetEncoder mel features).
# Modern pip install ships libsndfile_arm64.dylib alongside _soundfile_data,
# so the older "missing libsndfile.dylib" runtime crash no longer applies.
# Verified working at bundle time: see verify-bundled-python.sh soundfile gate.
# setuptools: KEEP. Torch declares it as a runtime requirement, and removing it
# leaves the packaged environment dependency-incomplete even when imports seem
# to work on the build machine.
rm -rf "$WHEELHOUSE" 2>/dev/null || true

# Keep pip intact (needed for engine auto-update at runtime via python3 -m pip)
# NOTE: Do NOT remove pip/_vendor/* — pip 26+ requires cachecontrol, pygments,
# rich, and other vendored modules. Removing them breaks `python3 -m pip install`.
# Only safe to remove: pip's test directories.
find "$SITE/pip" -type d -name "tests" -exec rm -rf {} + 2>/dev/null || true

# ====================================================================
# Patches for bundled dependencies (apply AFTER pip install, AFTER cleanup)
# These fix issues in transformers/mlx-vlm for torch-free environments.
# ====================================================================
echo "==> Applying bundled dependency patches..."

# 1. transformers/processing_utils.py: Allow None sub-processors (video_processor)
#    Without torchvision, Qwen2VL's video_processor loads as None. The type check
#    must allow None so image-only VLM usage works.
sed -i '' 's/if not isinstance(argument, proper_class):/if argument is not None and not isinstance(argument, proper_class):/' \
  "$SITE/transformers/processing_utils.py"

# 2. transformers/processing_utils.py: Skip ImportError when loading sub-processors
#    Video processor requires torchvision; gracefully skip when unavailable.
"$PYTHON" -c "
import re
path = '$SITE/transformers/processing_utils.py'
with open(path, 'r') as f:
    content = f.read()
# Wrap the auto_processor_class.from_pretrained call in try/except ImportError
old = '''            elif is_primary:
                # Primary non-tokenizer sub-processor: load via Auto class
                auto_processor_class = MODALITY_TO_AUTOPROCESSOR_MAPPING[sub_processor_type]
                sub_processor = auto_processor_class.from_pretrained(
                    pretrained_model_name_or_path, subfolder=subfolder, **kwargs
                )
                args.append(sub_processor)'''
new = '''            elif is_primary:
                # Primary non-tokenizer sub-processor: load via Auto class
                auto_processor_class = MODALITY_TO_AUTOPROCESSOR_MAPPING[sub_processor_type]
                try:
                    sub_processor = auto_processor_class.from_pretrained(
                        pretrained_model_name_or_path, subfolder=subfolder, **kwargs
                    )
                    args.append(sub_processor)
                except ImportError:
                    # Skip sub-processors that need unavailable backends (e.g. video needs torchvision)
                    pass'''
if old in content:
    content = content.replace(old, new)
    with open(path, 'w') as f:
        f.write(content)
    print('  Patched: processing_utils.py sub-processor ImportError handling')
else:
    print('  Already patched or structure changed: processing_utils.py sub-processor')
"

# 3. transformers/models/auto/video_processing_auto.py: Null check for extractors
#    transformers 5.2.0 bug where extractors can be None
sed -i '' 's/if class_name in extractors:/if extractors is not None and class_name in extractors:/' \
  "$SITE/transformers/models/auto/video_processing_auto.py" 2>/dev/null || true

# 4. mlx_vlm/utils.py: Lazy-import soundfile (defense-in-depth)
#    Even after removing the soundfile package, patch the import to be lazy
#    in case soundfile gets pulled back in as a transitive dep.
sed -i '' 's/^import soundfile as sf$/# import soundfile as sf  # lazy-loaded: see _get_sf()/' \
  "$SITE/mlx_vlm/utils.py" 2>/dev/null || true

# 5. mlx_vlm/models/qwen3_5/language.py: Fix mRoPE dimension mismatch for MoE
#    mlx-vlm 0.3.12 bug: broadcasting with cos/sin can produce 5D tensors
"$PYTHON" -c "
path = '$SITE/mlx_vlm/models/qwen3_5/language.py'
try:
    with open(path, 'r') as f:
        content = f.read()
    old = '''    q_embed = (q_rot * cos) + (rotate_half(q_rot) * sin)
    k_embed = (k_rot * cos) + (rotate_half(k_rot) * sin)

    q_embed = mx.concatenate([q_embed, q_pass], axis=-1)'''
    new = '''    q_embed = (q_rot * cos) + (rotate_half(q_rot) * sin)
    k_embed = (k_rot * cos) + (rotate_half(k_rot) * sin)

    # Fix mRoPE dimension mismatch for MoE models: broadcasting with cos/sin
    # can produce 5D tensors when q_pass is 4D (mlx-vlm 0.3.12 bug)
    if q_embed.ndim > q_pass.ndim and q_embed.ndim == 5:
        q_embed = q_embed[0]
        k_embed = k_embed[0]

    q_embed = mx.concatenate([q_embed, q_pass], axis=-1)'''
    if old in content:
        content = content.replace(old, new)
        with open(path, 'w') as f:
            f.write(content)
        print('  Patched: qwen3_5/language.py mRoPE dimension fix')
    else:
        print('  Already patched or structure changed: qwen3_5/language.py')
except FileNotFoundError:
    print('  Skipped: qwen3_5/language.py not found (model not in this mlx-vlm version)')
	"
	
# --- Patch: mlx_vlm/models/gemma4/vision.py (Gemma 4 multi-image input) ---
# mlxstudio#88: some Gemma 4 processors return `pixel_values` as a Python
# list containing numpy arrays. mlx.concatenate requires mx.array inputs.
echo "  Patching mlx_vlm/models/gemma4/vision.py (pixel_values list coercion)..."
"$PYTHON" -c "
path = '$SITE/mlx_vlm/models/gemma4/vision.py'
try:
    with open(path, 'r') as f:
        content = f.read()
    old = '''    def __call__(self, pixel_values: mx.array) -> mx.array:
        if isinstance(pixel_values, list):
            pixel_values = mx.concatenate(pixel_values, axis=0)

        B, C, H, W = pixel_values.shape'''
    new = '''    def __call__(self, pixel_values: mx.array) -> mx.array:
        if isinstance(pixel_values, list):
            # mlxstudio#88: multi-image processors can hand us a Python list
            # containing numpy arrays instead of MLX arrays. Upstream
            # mx.concatenate only accepts mx.array inputs, so coerce each
            # element before concatenating.
            pixel_values = [
                v if isinstance(v, mx.array) else mx.array(v)
                for v in pixel_values
            ]
            pixel_values = mx.concatenate(pixel_values, axis=0)
        elif not isinstance(pixel_values, mx.array):
            pixel_values = mx.array(pixel_values)

        B, C, H, W = pixel_values.shape'''
    if old in content:
        content = content.replace(old, new)
        with open(path, 'w') as f:
            f.write(content)
        print('  Patched: gemma4 vision pixel_values list coercion')
    elif 'mlxstudio#88' in content and 'isinstance(v, mx.array)' in content:
        print('  Already patched: gemma4 vision pixel_values list coercion')
    elif all(
        needle in content
        for needle in (
            'if isinstance(pixel_values, list):',
            'if not isinstance(img, mx.array):',
            'img = mx.array(img)',
        )
    ):
        print('  Native support: gemma4 vision handles mixed and different-sized image lists')
    else:
        raise SystemExit(
            '  ERROR: gemma4 vision mixed-list support is neither patchable nor native'
        )
except FileNotFoundError:
    raise SystemExit('  ERROR: gemma4 vision.py not found')
"

# --- Patch: mlx_lm MLA fp32 SDPA absorb (DeepSeek V3 / V3.2 / Mistral 4) ---
# The L==1 decode path for MLA-family models needs the q/k/v/mask cast to
# fp32 before scaled_dot_product_attention or the absorb branch produces
# logit drift. Upstream mlx_lm has carried this fix in some 0.31.x
# releases but regressed it in 0.31.3, so vMLX vendors the patched files
# at panel/scripts/patches/*.patched.py and copies them in at bundle time.
# Rationale: this is a CONTROLLED FORK of three upstream files (Apache-2.0,
# attribution preserved), not a runtime monkeypatch. Each rebuild lands
# the deterministic patched source.
PATCH_DIR="$REPO_DIR/panel/scripts/patches"
for entry in \
  "deepseek_v3.patched.py:mlx_lm/models/deepseek_v3.py:Kimi K2.6 fp32 MLA decode" \
  "deepseek_v32.patched.py:mlx_lm/models/deepseek_v32.py:DSV3.2 fp32 MLA decode" \
  "mistral4.patched.py:mlx_lm/models/mistral4.py:Mistral 4 fp32 MLA decode"; do
  IFS=":" read -r src rel desc <<< "$entry"
  src_path="$PATCH_DIR/$src"
  dst_path="$SITE/$rel"
  if [ ! -f "$src_path" ]; then
    echo "ERROR: required patch source missing: $src_path" >&2
    exit 1
  fi
  echo "  Installing $rel ($desc)..."
  mkdir -p "$(dirname "$dst_path")"
  cp "$src_path" "$dst_path"
  if ! grep -q 'q_sdpa.*astype(mx\.float32)' "$dst_path"; then
    echo "ERROR: $rel post-install does not contain q_sdpa fp32 cast" >&2
    exit 1
  fi
done
# Stage a copy under bundle research/ for build-traceability (matches the
# v3 historical layout so existing diagnostics keep working).
mkdir -p "$BUNDLE_DIR/python/lib/python3.12/research"
cp "$PATCH_DIR/deepseek_v3.patched.py" \
   "$BUNDLE_DIR/python/lib/python3.12/research/deepseek_v3_patched.py"

# --- New model class: bailing_hybrid (Ling-2.6-flash / Bailing-V2.5) ---
# mlx-lm 0.31.3 doesn't ship a model class for `bailing_hybrid` model_type
# yet; vMLX vendors the validated implementation under panel/scripts/patches/.
# This is a NEW file, not a patch over an upstream one — bundle-python.sh
# just copies it in. Hard-fail if the file is missing.
BAILING_SRC="$PATCH_DIR/bailing_hybrid.patched.py"
BAILING_DST="$SITE/mlx_lm/models/bailing_hybrid.py"
if [ ! -f "$BAILING_SRC" ]; then
  echo "ERROR: required source missing: $BAILING_SRC" >&2
  exit 1
fi
echo "  Installing mlx_lm/models/bailing_hybrid.py (Ling-2.6 / Bailing-V2.5 hybrid)..."
cp "$BAILING_SRC" "$BAILING_DST"
if ! grep -q 'class LanguageModel' "$BAILING_DST"; then
  echo "ERROR: bailing_hybrid post-install missing LanguageModel class" >&2
  exit 1
fi

# --- Python 3.12 compatibility: distutils.version shim for RADIO remote code ---
# NVIDIA C-RADIO's HF remote code still imports
# `from distutils.version import LooseVersion`. Python 3.12 removed distutils,
# and the bundled runtime intentionally does not include setuptools. Provide the
# tiny compatibility surface that RADIO needs so Nemotron-Omni media startup
# works in the installed app without pulling dev-machine site-packages.
echo "  Installing distutils.version compatibility shim (Python 3.12 RADIO support)..."
mkdir -p "$SITE/distutils"
cat > "$SITE/distutils/__init__.py" <<'PY'
"""Minimal distutils compatibility package for bundled Python 3.12."""
PY
cat > "$SITE/distutils/version.py" <<'PY'
"""Subset of distutils.version used by legacy HF remote-code modules."""

from packaging.version import parse as _parse


class LooseVersion:
    def __init__(self, vstring=None):
        self.vstring = "" if vstring is None else str(vstring)
        self._version = _parse(self.vstring)

    def __repr__(self):
        return f"LooseVersion({self.vstring!r})"

    def __str__(self):
        return self.vstring

    def _coerce(self, other):
        if isinstance(other, LooseVersion):
            return other._version
        return _parse(str(other))

    def __lt__(self, other):
        return self._version < self._coerce(other)

    def __le__(self, other):
        return self._version <= self._coerce(other)

    def __eq__(self, other):
        return self._version == self._coerce(other)

    def __ne__(self, other):
        return not self == other

    def __gt__(self, other):
        return self._version > self._coerce(other)

    def __ge__(self, other):
        return self._version >= self._coerce(other)
PY
"$PYTHON" -s - <<'PY'
from distutils.version import LooseVersion
assert LooseVersion("1.2") < LooseVersion("1.3")
PY

# --- Patch: mlx_lm/models/ssm.py (Mamba/Nemotron-H hybrid state space model) ---
# Fix 1: mx.clip(dt, ...) upper-clips dt values, corrupting Mamba state transitions.
#         Replace with mx.maximum(dt, time_step_limit[0]) — only lower-clip.
# Fix 2: output_dtypes=[input_type, input_type] stores SSM state in bfloat16,
#         causing precision loss. State must be float32.
echo "  Patching mlx_lm/models/ssm.py (Mamba state fixes)..."
"$PYTHON" -c "
import os, glob
base = '$BUNDLE_DIR/python/lib/python3.*/site-packages/mlx_lm/models/ssm.py'
paths = glob.glob(base)
if not paths:
    print('  Skipped: ssm.py not found')
else:
    path = paths[0]
    with open(path, 'r') as f:
        content = f.read()
    changed = False
    # Fix 1: clip -> maximum (line 10)
    old1 = 'return mx.clip(dt, time_step_limit[0], time_step_limit[1])'
    new1 = 'return mx.maximum(dt, time_step_limit[0])'
    if old1 in content:
        content = content.replace(old1, new1)
        changed = True
        print('  Patched: ssm.py dt clip -> maximum')
    else:
        print('  Already patched or structure changed: ssm.py dt fix')
    # Fix 2: state output dtype must be float32
    old2 = 'output_dtypes=[input_type, input_type]'
    new2 = 'output_dtypes=[input_type, mx.float32]'
    if old2 in content:
        content = content.replace(old2, new2)
        changed = True
        print('  Patched: ssm.py state dtype -> float32')
    else:
        print('  Already patched or structure changed: ssm.py dtype fix')
    if changed:
        with open(path, 'w') as f:
            f.write(content)
"

echo "==> Patches applied."

# Verify imports only after dependency patches are baked in. Importing
# vmlx_engine/CLI earlier can trigger runtime patch checks against a half-built
# bundle and produce false "patched source not found" diagnostics.
echo "==> Verifying installation..."
"$PYTHON" -c "import vmlx_engine; print(f'vmlx_engine {vmlx_engine.__version__} imported OK')"
"$PYTHON" -m vmlx_engine.cli --help > /dev/null 2>&1 && echo "CLI OK"

# ====================================================================
# Critical: Verify the Python shared library exists (prevents broken bundles)
# The bundled Python MUST include libpython3.12.dylib for the app to work.
# Without it, the app falls back to system Python which may have outdated or
# missing packages (e.g., mlx_vlm without qwen3_5_moe support).
# ====================================================================
echo "==> Verifying Python shared library..."
LIBPYTHON="$BUNDLE_DIR/python/lib/libpython3.12.dylib"
if [ -f "$LIBPYTHON" ]; then
  echo "  libpython3.12.dylib OK ($(du -h "$LIBPYTHON" | cut -f1))"
else
  # Check if it exists elsewhere in the bundle (some builds put it in different locations)
  FOUND=$(find "$BUNDLE_DIR" -name "libpython3.12*.dylib" 2>/dev/null | head -1)
  if [ -n "$FOUND" ]; then
    echo "  Found at: $FOUND — creating symlink"
    ln -sf "$FOUND" "$LIBPYTHON"
  else
    echo "ERROR: libpython3.12.dylib NOT FOUND in bundle!"
    echo "  The app will fall back to system Python, which may have outdated packages."
    echo "  This is a critical build issue — the bundle is incomplete."
    exit 1
  fi
fi

# Post-cleanup verification: ensure pip still works (catches vendor stripping bugs)
echo "==> Verifying pip is functional (needed for engine auto-update)..."
"$PYTHON" -s -m pip --version > /dev/null 2>&1 || { echo "ERROR: pip is broken after cleanup! Check vendor removals."; exit 1; }
"$PYTHON" -s -m pip check
echo "  pip OK"

# The verification imports above run after the first size cleanup and can
# recreate bytecode caches. Remove them again before packaging so a signed app
# never ships generated Python cache files as sealed Resources.
echo "==> Removing Python bytecode generated by verification imports..."
find "$BUNDLE_DIR" -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
find "$BUNDLE_DIR" -name "*.pyc" -delete 2>/dev/null || true

# Critical: reject editable installs (prevents shipping dev-machine paths to users)
echo "==> Checking for editable installs..."
EDITABLE_PTH=$(find "$SITE" -maxdepth 1 -name "__editable__.*" -o -name "__editable___*" 2>/dev/null)
if [ -n "$EDITABLE_PTH" ]; then
  echo "ERROR: Editable install detected in bundled Python!"
  echo "  Found: $EDITABLE_PTH"
  echo "  This would ship with hardcoded paths to your dev machine."
  echo "  Fix: re-run bundle-python.sh from scratch (it cleans the bundle dir)."
  exit 1
fi
echo "  No editable installs (good)"

# Verify path isolation
echo "==> Verifying path isolation..."
ENABLE_USER_SITE=$("$PYTHON" -s -c "import site; print(site.ENABLE_USER_SITE)" 2>&1)
if [ "$ENABLE_USER_SITE" = "False" ]; then
  echo "  ENABLE_USER_SITE=False with -s flag OK"
else
  echo "WARNING: -s flag did not suppress user site-packages"
fi

echo ""
# Post-bundle: rewrite shebangs in console scripts to a relocatable sibling
# Python trampoline.
# pip bakes the source bundled-python path into shebangs, which would ship to
# users and never resolve. Older builds rewrote to
# `/Applications/vMLX.app/...`, but that breaks repo-local test apps and users
# who run the app from another location. Keep every console script anchored to
# its own `python/bin/python3` instead.
#
# The shell/Python polyglot preserves `-B -s`; without `-B`, direct
# console-script use writes __pycache__ into the signed .app bundle and
# invalidates the sealed Resources signature after first launch.
echo "==> Rewriting console-script shebangs to relocatable bundled Python..."
SHEBANG_FIXED=0
for SCRIPT in "$BUNDLE_DIR/python/bin/"*; do
  if [ ! -f "$SCRIPT" ]; then
    continue
  fi
  FIRST_LINE="$(LC_ALL=C head -n 1 "$SCRIPT" 2>/dev/null || true)"
  if [[ "$FIRST_LINE" == '#!'*python* ]] && [[ "$FIRST_LINE" != '#!/bin/sh'* ]]; then
    TMP_SCRIPT="$(mktemp "${SCRIPT}.XXXXXX")"
    {
      printf '%s\n' '#!/bin/sh'
      printf '%s\n' "'''exec' \"\$(dirname \"\$0\")/python3\" -B -s \"\$0\" \"\$@\""
      printf '%s\n' "' '''"
      tail -n +2 "$SCRIPT"
    } > "$TMP_SCRIPT"
    chmod --reference="$SCRIPT" "$TMP_SCRIPT" 2>/dev/null || chmod +x "$TMP_SCRIPT"
    mv "$TMP_SCRIPT" "$SCRIPT"
    SHEBANG_FIXED=$((SHEBANG_FIXED + 1))
  fi
done
echo "  rewrote $SHEBANG_FIXED console-script shebangs"
# Sanity check — no script shebang should still reference a dev or absolute app path.
LEAKED=$(
  find "$BUNDLE_DIR/python/bin" -maxdepth 1 -type f -perm -111 -print 2>/dev/null \
    | while read -r SCRIPT; do
        FIRST_LINE="$(LC_ALL=C head -n 1 "$SCRIPT" 2>/dev/null || true)"
        if [[ "$FIRST_LINE" == '#!'*python* ]] \
          || [[ "$FIRST_LINE" == *"$BUNDLE_DIR"* ]] \
          || [[ "$FIRST_LINE" == *"/Users/"* ]] \
          || [[ "$FIRST_LINE" == *"/Applications/vMLX.app"* ]]; then
          printf '%s: %s\n' "$SCRIPT" "$FIRST_LINE"
        fi
      done \
    | head -20
)
if [ -n "$LEAKED" ]; then
  echo "ERROR: non-relocatable console-script shebangs after rewrite:"
  echo "$LEAKED"
  exit 1
fi
echo "  console-script shebangs are relocatable (good)"

echo ""
echo "==> Bundle size:"
du -sh "$BUNDLE_DIR"
echo ""
echo "==> Running the release verifier against the staged bundle..."
VMLX_BUNDLED_PYTHON_DIR="$BUNDLE_DIR" \
VMLX_EXPECTED_MLX_WHEEL_PLATFORM="$MLX_WHEEL_PLATFORM" \
  "$SCRIPT_DIR/verify-bundled-python.sh"

if [ -L "$FINAL_BUNDLE_DIR" ]; then
  echo "ERROR: refusing to replace symlinked bundled-Python destination: $FINAL_BUNDLE_DIR" >&2
  exit 1
fi
if [ -e "$FINAL_BUNDLE_DIR" ] && [ ! -d "$FINAL_BUNDLE_DIR" ]; then
  echo "ERROR: bundled-Python destination is not a directory: $FINAL_BUNDLE_DIR" >&2
  exit 1
fi

echo "==> Publishing verified bundled Python..."
if [ -d "$FINAL_BUNDLE_DIR" ]; then
  mv "$FINAL_BUNDLE_DIR" "$PREVIOUS_BUNDLE_DIR"
fi
if ! mv "$BUNDLE_DIR" "$FINAL_BUNDLE_DIR"; then
  echo "ERROR: failed to publish verified bundled Python" >&2
  if [ -d "$PREVIOUS_BUNDLE_DIR" ] && [ ! -e "$FINAL_BUNDLE_DIR" ]; then
    mv "$PREVIOUS_BUNDLE_DIR" "$FINAL_BUNDLE_DIR" || true
  fi
  exit 1
fi
BUNDLE_PUBLISHED=1
remove_bundle_tree_with_retry "$PREVIOUS_BUNDLE_DIR"

echo "==> Done! Bundled Python ready at: $FINAL_BUNDLE_DIR"
echo "    Next: npm run build && npx electron-builder --mac"
