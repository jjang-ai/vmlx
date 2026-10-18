#!/bin/bash
# Release-time sanity check: bundled-python must have all critical model
# modules that vMLX depends on. Runs before electron-builder packages the
# .app so we never ship a DMG that instantly ModuleNotFoundErrors on a
# model the user tries to load.
#
# Added after a user reported `ModuleNotFoundError: No module named
# 'mlx_vlm.models.gemma4'` on a fresh install — the bundled mlx_vlm 0.4.0
# had the gemma4 dir cherry-picked in at some point and we want to make
# sure we never regress the cherry-pick on a future rebuild.
set -euo pipefail
R20_FIXED_PATH="/opt/homebrew/bin:/usr/bin:/bin:/usr/sbin:/sbin"
export PATH="$R20_FIXED_PATH"
export PYTHONNOUSERSITE=1
unset PYTHONPATH PYTHONHOME VIRTUAL_ENV

HERE="$(cd -P "$(dirname "$0")" && pwd -P)"
PANEL="$(cd -P "$HERE/.." && pwd -P)"
BUNDLE_ROOT="${VMLX_BUNDLED_PYTHON_DIR:-$PANEL/bundled-python}"
if [ ! -d "$BUNDLE_ROOT" ]; then
  echo "❌ bundled-python root missing: $BUNDLE_ROOT"
  exit 1
fi
BUNDLE_ROOT="$(cd -P "$BUNDLE_ROOT" && pwd -P)"
case "$BUNDLE_ROOT" in
  "$PANEL/bundled-python"|"$PANEL"/.bundled-python.staging.*)
    ;;
  *)
    echo "❌ RELEASE BLOCKED — bundled-python root is outside the panel: $BUNDLE_ROOT"
    exit 1
    ;;
esac
PY="$BUNDLE_ROOT/python/bin/python3"
RELEASE_SCOPE="${VMLX_RELEASE_SCOPE:-${VMLINUX_RELEASE_SCOPE:-}}"
NODE_BIN="${VMLX_R20_TOOL_NODE_REALPATH:-/opt/homebrew/bin/node}"
GIT_BIN="${VMLX_R20_TOOL_GIT_REALPATH:-/usr/bin/git}"
SHASUM_BIN="${VMLX_R20_TOOL_SHASUM_REALPATH:-/usr/bin/shasum}"
AWK_BIN="${VMLX_R20_TOOL_AWK_REALPATH:-/usr/bin/awk}"
FIND_BIN="${VMLX_R20_TOOL_FIND_REALPATH:-/usr/bin/find}"

if [ ! -x "$PY" ]; then
  echo "❌ bundled python missing: $PY"
  exit 1
fi

run_bundled_python() {
  cd /tmp
  PYTHONDONTWRITEBYTECODE=1 PYTHONNOUSERSITE=1 PYTHONPATH= "$PY" -B -s "$@"
}

echo "==> Verifying bundled dependency closure..."
run_bundled_python -m pip check
run_bundled_python - <<'PYEOF'
from importlib.metadata import PackageNotFoundError, version

expected_distribution = "4.13.0.92"
installed_distribution = version("opencv-python")
if installed_distribution != expected_distribution:
    raise SystemExit(
        "RELEASE BLOCKED — bundled opencv-python version drift: "
        f"{installed_distribution} != {expected_distribution}"
    )
try:
    headless_version = version("opencv-python-headless")
except PackageNotFoundError:
    headless_version = None
if headless_version is not None:
    raise SystemExit(
        "RELEASE BLOCKED — overlapping opencv-python-headless distribution "
        f"is installed ({headless_version})"
    )
import cv2

if not cv2.__version__.startswith("4.13.0"):
    raise SystemExit(
        f"RELEASE BLOCKED — bundled cv2 version drift: {cv2.__version__}"
    )
print(
    "  ok   opencv-python 4.13.0.92 is the sole cv2 distribution "
    f"(cv2 {cv2.__version__})"
)
PYEOF

assert_r20_pinned_tool() {
  local name="$1"
  local path="$2"
  local expected_sha_var="VMLX_R20_TOOL_${name}_SHA256"
  local expected_sha256="${!expected_sha_var:-}"
  if [ "$RELEASE_SCOPE" != "r20_production" ]; then
    return 0
  fi
  if [ -z "$expected_sha256" ]; then
    echo "❌ RELEASE BLOCKED — missing pinned $name SHA-256"
    exit 1
  fi
  run_bundled_python - "$path" "$expected_sha256" "$name" <<'PYEOF'
import hashlib
import os
import pathlib
import stat
import sys

path = pathlib.Path(sys.argv[1]).resolve(strict=True)
expected = sys.argv[2]
name = sys.argv[3]
metadata = path.stat()
if not stat.S_ISREG(metadata.st_mode) or not os.access(path, os.X_OK):
    raise SystemExit(f"pinned {name} is not an executable regular file")
digest = hashlib.sha256(path.read_bytes()).hexdigest()
if digest != expected:
    raise SystemExit(f"pinned {name} SHA-256 changed")
PYEOF
}

# Hard guard against the 1.5.9→1.5.12 ship-stale-engine class of bug:
# if package.json bumps but bundle-python.sh wasn't re-run, the bundled
# vmlx_engine still reports the old version. Refuse to package the .app
# in that case so no DMG ever ships an installer/runtime version mismatch.
assert_r20_pinned_tool NODE "$NODE_BIN"
assert_r20_pinned_tool GIT "$GIT_BIN"
assert_r20_pinned_tool SHASUM "$SHASUM_BIN"
assert_r20_pinned_tool AWK "$AWK_BIN"
assert_r20_pinned_tool FIND "$FIND_BIN"
assert_r20_pinned_tool NODE "$NODE_BIN"
PKG_VERSION="$("$NODE_BIN" -p "require('$PANEL/package.json').version")"
BUNDLED_VERSION="$(run_bundled_python -c 'import vmlx_engine; print(vmlx_engine.__version__)' 2>/dev/null || echo "MISSING")"
if [ "$PKG_VERSION" != "$BUNDLED_VERSION" ]; then
  echo "❌ RELEASE BLOCKED — bundled-python vmlx_engine version drift"
  echo "   package.json version : $PKG_VERSION"
  echo "   bundled-python ships : $BUNDLED_VERSION"
  echo
  echo "   Re-run ./scripts/bundle-python.sh so the bundled site-packages"
  echo "   match the version users will see in the .app's Info.plist."
  exit 1
fi
echo "  ok   bundled vmlx_engine version matches package.json ($PKG_VERSION)"

# Version parity alone is not enough: a stale bundled-python can carry the
# current __version__ string while still shipping old runtime code. Block the
# removed DSV4 force-flip env vars explicitly so direct/off rails cannot be
# silently flipped back on in a packaged build.
BUNDLED_ENGINE_DIR="$(run_bundled_python -c 'import pathlib, vmlx_engine; print(pathlib.Path(vmlx_engine.__file__).resolve().parent)' 2>/dev/null || true)"
if [ -z "$BUNDLED_ENGINE_DIR" ] || [ ! -d "$BUNDLED_ENGINE_DIR" ]; then
  echo "❌ RELEASE BLOCKED — cannot locate bundled vmlx_engine package"
  echo "   resolved path: ${BUNDLED_ENGINE_DIR:-<empty>}"
  exit 1
fi
if grep -R -n --include='*.py' -E 'VMLX_DSV4_ALLOW_CHAT|VMLX_DSV4_ALLOW_THINKING|VMLX_DSV4_FORCE_DIRECT_RAIL' "$BUNDLED_ENGINE_DIR"; then
  echo
  echo "❌ RELEASE BLOCKED — bundled-python contains removed DSV4 env-var force-flips"
  echo "   Forbidden flags: VMLX_DSV4_ALLOW_CHAT, VMLX_DSV4_ALLOW_THINKING, VMLX_DSV4_FORCE_DIRECT_RAIL"
  echo "   Bundled package : $BUNDLED_ENGINE_DIR"
  echo
  echo "   Re-run ./scripts/bundle-python.sh from a clean source checkout."
  exit 1
fi
echo "  ok   bundled vmlx_engine has no removed DSV4 env-var force-flips"

check_console_script_shebangs() {
  local bin_dir="$1"
  local label="$2"
  if [ ! -d "$bin_dir" ]; then
    echo "❌ RELEASE BLOCKED — $label console-script dir missing: $bin_dir"
    exit 1
  fi
  local leaks
  leaks=$(
    "$FIND_BIN" "$bin_dir" -maxdepth 1 -type f -perm -111 -print 2>/dev/null \
      | while read -r script; do
          first_line="$(LC_ALL=C head -n 1 "$script" 2>/dev/null || true)"
          if [[ "$first_line" == '#!'*python* ]] \
            || [[ "$first_line" == *"$BUNDLE_ROOT"* ]] \
            || [[ "$first_line" == *"/Users/"* ]] \
            || [[ "$first_line" == *"/Applications/vMLX.app"* ]]; then
            printf '%s: %s\n' "$script" "$first_line"
          fi
        done
  )
  if [ -n "$leaks" ]; then
    echo "❌ RELEASE BLOCKED — $label has non-relocatable console-script shebangs"
    echo "$leaks" | head -40
    echo
    echo "   Re-run ./scripts/bundle-python.sh, or rewrite the shebangs to the"
    echo "   bundled sibling-python trampoline before packaging."
    exit 1
  fi
  echo "  ok   $label console-script shebangs are relocatable"
}

check_console_script_shebangs "$BUNDLE_ROOT/python/bin" "bundled-python"

SOURCE_ENGINE_DIR="$PANEL/../vmlx_engine"
HASH_GATED_ENGINE_FILES=(
  "__init__.py"
  "server.py"
  "api/utils.py"
  "api/tool_calling.py"
  "api/anthropic_adapter.py"
  "api/ollama_adapter.py"
  "block_disk_store.py"
  "cli.py"
  "disk_cache.py"
  "engine/batched.py"
  "engine/simple.py"
  "cache_record_validator.py"
  "loaders/load_jangtq_dsv4.py"
  "memory_cache.py"
  "mllm_batch_generator.py"
  "mllm_scheduler.py"
  "model_configs.py"
  "model_config_registry.py"
  "output_collector.py"
  "reasoning/minimax_m3_parser.py"
  "speculative.py"
  "models/llm.py"
  "models/mllm.py"
  "models/mimo_v26.py"
  "models/mimo_v26_contract.py"
  "models/mimo_v26_cache.py"
  "tool_parsers/xml_function_tool_parser.py"
  "models/minimax_m3/cache.py"
  "models/step3p7_mlx_vlm.py"
  "models/gemma4_unified_register.py"
  "models/gemma4_unified/__init__.py"
  "models/gemma4_unified/config.py"
  "models/gemma4_unified/gemma4_unified.py"
  "models/gemma4_unified/processing_gemma4_unified.py"
  "omni_multimodal.py"
  "paged_cache.py"
  "prefix_cache.py"
  "runtime_patches/gemma4_vision.py"
  "runtime_patches/gemma4_processing.py"
  "sampling.py"
  "scheduler.py"
  "tool_parsers/dsml_tool_parser.py"
  "patches/mlx_vlm_mtp/qwen35_vl.py"
  "utils/single_batch_generator.py"
  "utils/chat_template_kwargs.py"
  "utils/head_dim_detection.py"
  "utils/mlx_vlm_compat.py"
  "utils/ssm_companion_cache.py"
  "utils/ssm_companion_disk_store.py"
  "utils/jang_loader.py"
  "utils/nanbeige_runtime.py"
  "utils/tokenizer.py"
  "chat_templates/gemma4.jinja"
  "config/defaults.yaml"
  "metal/codebook_matvec.metal"
  "metal/codebook_moe.metal"
)
for rel in "${HASH_GATED_ENGINE_FILES[@]}"; do
  if [ ! -f "$SOURCE_ENGINE_DIR/$rel" ] || [ ! -f "$BUNDLED_ENGINE_DIR/$rel" ]; then
    echo "❌ RELEASE BLOCKED — cannot compare source and bundled vmlx_engine/$rel"
    echo "   source : $SOURCE_ENGINE_DIR/$rel"
    echo "   bundled: $BUNDLED_ENGINE_DIR/$rel"
    exit 1
  fi
  SOURCE_SHA="$("$SHASUM_BIN" -a 256 "$SOURCE_ENGINE_DIR/$rel" | "$AWK_BIN" '{print $1}')"
  BUNDLED_SHA="$("$SHASUM_BIN" -a 256 "$BUNDLED_ENGINE_DIR/$rel" | "$AWK_BIN" '{print $1}')"
  if [ "$SOURCE_SHA" != "$BUNDLED_SHA" ]; then
    echo "❌ RELEASE BLOCKED — bundled vmlx_engine/$rel content drift"
    echo "   source sha256 : $SOURCE_SHA"
    echo "   bundled sha256: $BUNDLED_SHA"
    echo
    echo "   Re-run ./scripts/bundle-python.sh from this checkout."
    exit 1
  fi
done
echo "  ok   bundled critical vmlx_engine files match source content"

BUNDLED_JANG_TOOLS_DIR="$(run_bundled_python -c 'import pathlib, jang_tools; print(pathlib.Path(jang_tools.__file__).resolve().parent)' 2>/dev/null || true)"
JANG_TOOLS_SOURCE_ROOT="${VMLX_JANG_TOOLS_SOURCE:-${VMLINUX_JANG_TOOLS_SOURCE:-$HOME/jang/jang-tools}}"
JANG_TOOLS_SOURCE_DIR="$JANG_TOOLS_SOURCE_ROOT/jang_tools"
JANG_MIN_VERSION="2.5.50"
PROVENANCE_FILE="$BUNDLE_ROOT/vmlx-bundle-provenance.json"
HASH_GATED_JANG_TOOLS_FILES=(
  "capabilities.py"
  "convert.py"
  "convert_hy3_jangtq.py"
  "loader.py"
  "load_jangtq.py"
  "load_jangtq_vlm.py"
  "load_jangtq_kimi_vlm.py"
  "nemotron_omni_chat.py"
  "dsv4/mlx_model.py"
  "dsv4/pool_quant_cache.py"
  "hy3/__init__.py"
  "hy3/model.py"
  "hy3/runtime.py"
  "laguna/runtime.py"
  "kimi_prune/generate_vl.py"
  "kimi_prune/runtime_patch.py"
  "mimo_v2/mlx_model.py"
  "mimo_v2/mlx_register.py"
  "mimo_v2/v26_model.py"
  "mimo_v2/v26_vision.py"
  "mimo_v2/v26_audio.py"
  "mimo_v2/v26_omni.py"
  "nanbeige/__init__.py"
  "nanbeige/model.py"
  "nanbeige/mlx_register.py"
  "step37/__init__.py"
  "step37/nvfp4_codec.py"
  "step37/step3p7_mlx.py"
  "topk_override.py"
  "turboquant/fused_gate_up_kernel.py"
  "turboquant/gather_tq_kernel.py"
  "turboquant/hadamard_kernel.py"
  "turboquant/mpp_nax_kernel.py"
  "turboquant/tq_kernel.py"
)
if [ -z "$BUNDLED_JANG_TOOLS_DIR" ] || [ ! -d "$BUNDLED_JANG_TOOLS_DIR" ]; then
  echo "❌ RELEASE BLOCKED — cannot locate bundled jang_tools package"
  echo "   resolved path: ${BUNDLED_JANG_TOOLS_DIR:-<empty>}"
  exit 1
fi
if [ ! -f "$PROVENANCE_FILE" ]; then
  echo "❌ RELEASE BLOCKED — bundled-Python provenance manifest is missing"
  echo "   expected: $PROVENANCE_FILE"
  exit 1
fi
EXPECTED_MLX_WHEEL_PLATFORM="${VMLX_EXPECTED_MLX_WHEEL_PLATFORM:-${VMLINUX_EXPECTED_MLX_WHEEL_PLATFORM:-}}"
if [ "$RELEASE_SCOPE" = "r20_production" ] && [ -z "$EXPECTED_MLX_WHEEL_PLATFORM" ]; then
  echo "❌ RELEASE BLOCKED — production bundle verification requires the exact MLX wheel platform"
  exit 1
fi
EXPECTED_MLX_VERSION="$(
  sed -n 's/^MLX_VERSION="\([^"]*\)"/\1/p' "$PANEL/scripts/bundle-python.sh" \
    | head -1
)"
if [ -z "$EXPECTED_MLX_VERSION" ]; then
  echo "❌ RELEASE BLOCKED — cannot resolve the pinned MLX version from bundle-python.sh"
  exit 1
fi
run_bundled_python - \
  "$PROVENANCE_FILE" \
  "$EXPECTED_MLX_WHEEL_PLATFORM" \
  "$EXPECTED_MLX_VERSION" <<'PYEOF'
import importlib.metadata
import json
import pathlib
import sys

provenance = json.loads(pathlib.Path(sys.argv[1]).read_text(encoding="utf-8"))
expected = sys.argv[2] or provenance.get("mlx_wheel_platform", "")
expected_version = sys.argv[3]
if expected not in {"macosx_14_0_arm64", "macosx_26_0_arm64"}:
    raise SystemExit(f"invalid expected MLX wheel platform: {expected!r}")
if provenance.get("mlx_wheel_platform") != expected:
    raise SystemExit(
        "bundle provenance MLX wheel platform differs from the requested release flavor"
    )
records = {}
for distribution_name in ("mlx", "mlx-metal"):
    distribution = importlib.metadata.distribution(distribution_name)
    wheel_path = pathlib.Path(distribution._path) / "WHEEL"
    tags = sorted(
        {
            line.split(":", 1)[1].strip()
            for line in wheel_path.read_text(encoding="utf-8").splitlines()
            if line.startswith("Tag:")
        }
    )
    platforms = {tag.rsplit("-", 1)[-1] for tag in tags}
    if (
        not tags
        or any(tag.count("-") < 2 for tag in tags)
        or platforms != {expected}
    ):
        raise SystemExit(
            f"installed {distribution_name} wheel tags do not exactly target "
            f"{expected}: {tags}"
        )
    records[distribution_name] = {
        "version": distribution.version,
        "tags": tags,
    }
if records["mlx"]["version"] != records["mlx-metal"]["version"]:
    raise SystemExit(
        "installed mlx and mlx-metal distribution versions do not match"
    )
if records["mlx"]["version"] != expected_version:
    raise SystemExit(
        "RELEASE BLOCKED — bundled MLX version drift: "
        f"{records['mlx']['version']} != {expected_version}"
    )
print(
    "  ok   bundled MLX wheel contract "
    f"{records['mlx']['version']} @ {expected}"
)
PYEOF
SOURCE_JANG_VERSION="$(
  sed -n 's/^version[[:space:]]*=[[:space:]]*"\([^"]*\)".*/\1/p' \
    "$JANG_TOOLS_SOURCE_ROOT/pyproject.toml" 2>/dev/null | head -1
)"
BUNDLED_JANG_VERSION="$(
  run_bundled_python -c 'from importlib.metadata import version; print(version("jang"))'
)"
if [ -z "$SOURCE_JANG_VERSION" ] || [ "$SOURCE_JANG_VERSION" != "$BUNDLED_JANG_VERSION" ]; then
  echo "❌ RELEASE BLOCKED — bundled JANG distribution version drift"
  echo "   source version : ${SOURCE_JANG_VERSION:-<missing>}"
  echo "   bundled version: $BUNDLED_JANG_VERSION"
  exit 1
fi
run_bundled_python - "$BUNDLED_JANG_VERSION" "$JANG_MIN_VERSION" <<'PYEOF'
import sys
from packaging.version import Version

installed, minimum = map(Version, sys.argv[1:3])
if installed < minimum:
    raise SystemExit(
        f"RELEASE BLOCKED — bundled JANG {installed} is below required {minimum}"
    )
PYEOF
# Prefer origin/main over the local HEAD as the expected commit. Comparing the
# bundle against "whichever jang checkout happens to be default" is weak in both
# directions: it BLOCKS a correct bundle when the local checkout is a stale
# feature branch, and it PASSES a stale bundle whenever the same stale branch was
# used for both bundling and verifying — which is exactly how the pre-fix DSV4
# decode path shipped. origin/main is the same reference no matter which machine
# or worktree runs this.
# `git show <rev>:<path>` resolves <path> relative to the CURRENT DIRECTORY when
# that directory is below the repo root, and the jang checkout may sit in a
# subdirectory of its repo. Capture the prefix so the lookups below are anchored
# correctly either way.
JANG_GIT_PREFIX="$("$GIT_BIN" -C "$JANG_TOOLS_SOURCE_ROOT" rev-parse --show-prefix 2>/dev/null || true)"
EXPECTED_JANG_COMMIT="$("$GIT_BIN" -C "$JANG_TOOLS_SOURCE_ROOT" rev-parse origin/main 2>/dev/null || true)"
if [ -z "$EXPECTED_JANG_COMMIT" ]; then
  echo "    WARNING: no origin/main in $JANG_TOOLS_SOURCE_ROOT — falling back to local HEAD" >&2
  EXPECTED_JANG_COMMIT="$("$GIT_BIN" -C "$JANG_TOOLS_SOURCE_ROOT" rev-parse HEAD 2>/dev/null || true)"
fi
# A pinned release bundle records jang.release_pin in its provenance (written
# by bundle-python.sh under VMLX_JANG_TOOLS_RELEASE_PIN). Such a bundle
# deliberately ships one exact commit while origin/main has moved on — the
# mismatch this check must catch is bundle != pin, not pin != upstream. The
# pin is only honored when it matches the bundled commit itself, so a stale
# or tampered provenance cannot use it to bless an arbitrary bundle.
PROVENANCE_JANG_PIN="$(
  run_bundled_python - "$PROVENANCE_FILE" <<'PYEOF'
import json
import sys

print(json.load(open(sys.argv[1], encoding="utf-8"))["jang"].get("release_pin", ""))
PYEOF
)"
if [ -n "$PROVENANCE_JANG_PIN" ]; then
  echo "  ok   bundle declares jang release pin $PROVENANCE_JANG_PIN — verifying against the pin"
  EXPECTED_JANG_COMMIT="$PROVENANCE_JANG_PIN"
fi
PROVENANCE_JANG_COMMIT="$(
  run_bundled_python - "$PROVENANCE_FILE" <<'PYEOF'
import json
import sys

print(json.load(open(sys.argv[1], encoding="utf-8"))["jang"]["commit"])
PYEOF
)"
PROVENANCE_JANG_VERSION="$(
  run_bundled_python - "$PROVENANCE_FILE" <<'PYEOF'
import json
import sys

print(json.load(open(sys.argv[1], encoding="utf-8"))["jang"]["version"])
PYEOF
)"
if [ -z "$EXPECTED_JANG_COMMIT" ] \
  || [ "$PROVENANCE_JANG_COMMIT" != "$EXPECTED_JANG_COMMIT" ] \
  || [ "$PROVENANCE_JANG_VERSION" != "$BUNDLED_JANG_VERSION" ]; then
  echo "❌ RELEASE BLOCKED — bundled JANG provenance mismatch"
  echo "   source commit     : ${EXPECTED_JANG_COMMIT:-<missing>}"
  echo "   provenance commit : $PROVENANCE_JANG_COMMIT"
  echo "   bundled version   : $BUNDLED_JANG_VERSION"
  echo "   provenance version: $PROVENANCE_JANG_VERSION"
  exit 1
fi
echo "  ok   bundled JANG provenance matches source ($BUNDLED_JANG_VERSION @ $EXPECTED_JANG_COMMIT)"
if [ ! -d "$JANG_TOOLS_SOURCE_DIR" ]; then
  if [ "${VMLX_ALLOW_MISSING_JANG_SOURCE_HASH:-${VMLINUX_ALLOW_MISSING_JANG_SOURCE_HASH:-0}}" = "1" ]; then
    echo "⚠️  skipping jang_tools content hash parity; local source not present: $JANG_TOOLS_SOURCE_DIR"
  else
    echo "❌ RELEASE BLOCKED — local jang_tools source unavailable for hash parity"
    echo "   expected source: $JANG_TOOLS_SOURCE_DIR"
    echo
    echo "   Release builds must compare bundled jang_tools against local source."
    echo "   Set VMLX_ALLOW_MISSING_JANG_SOURCE_HASH=1 only for non-release CI smoke builds."
    exit 1
  fi
else
  for rel in "${HASH_GATED_JANG_TOOLS_FILES[@]}"; do
    if [ ! -f "$JANG_TOOLS_SOURCE_DIR/$rel" ] || [ ! -f "$BUNDLED_JANG_TOOLS_DIR/$rel" ]; then
      echo "❌ RELEASE BLOCKED — cannot compare source and bundled jang_tools/$rel"
      echo "   source : $JANG_TOOLS_SOURCE_DIR/$rel"
      echo "   bundled: $BUNDLED_JANG_TOOLS_DIR/$rel"
      exit 1
    fi
    # Use the exact commit accepted by the provenance gate above, including an
    # explicit release pin. Never compare a pinned bundle to unrelated main
    # bytes or fall back to an uncommitted working tree.
    SOURCE_SHA=""
    JANG_BLOB_REF="${EXPECTED_JANG_COMMIT}:${JANG_GIT_PREFIX}jang_tools/$rel"
    # Pipe the blob straight into shasum. Capturing it in a variable first would
    # strip trailing newlines (command substitution does), and every file would
    # then look like content drift.
    if "$GIT_BIN" -C "$JANG_TOOLS_SOURCE_ROOT" cat-file -e "$JANG_BLOB_REF" 2>/dev/null; then
      SOURCE_SHA="$(
        "$GIT_BIN" -C "$JANG_TOOLS_SOURCE_ROOT" show "$JANG_BLOB_REF" \
          | "$SHASUM_BIN" -a 256 | "$AWK_BIN" '{print $1}'
      )"
    fi
    if [ -z "$SOURCE_SHA" ]; then
      echo "❌ RELEASE BLOCKED — jang_tools/$rel is missing at $EXPECTED_JANG_COMMIT" >&2
      exit 1
    fi
    BUNDLED_SHA="$("$SHASUM_BIN" -a 256 "$BUNDLED_JANG_TOOLS_DIR/$rel" | "$AWK_BIN" '{print $1}')"
    if [ "$SOURCE_SHA" != "$BUNDLED_SHA" ]; then
      echo "❌ RELEASE BLOCKED — bundled jang_tools/$rel content drift"
      echo "   source sha256 : $SOURCE_SHA"
      echo "   bundled sha256: $BUNDLED_SHA"
      echo
      echo "   Re-run ./scripts/bundle-python.sh from this checkout."
      exit 1
    fi
  done
  echo "  ok   bundled critical jang_tools files match source content"
fi

# Isolated imports — no user site, no PYTHONPATH leakage (same env as the
# running engine). -s suppresses user site-packages the way sessions.ts does.
run_bundled_python - <<'PYEOF'
import sys

REQUIRED = [
    # (import name, human label, remediation hint)
    ("mlx", "mlx core", "bundled mlx package broken"),
    ("mlx.nn", "mlx.nn", "bundled mlx package broken"),
    ("mlx_lm", "mlx-lm", "bundled mlx-lm package broken"),
    ("mlx_vlm", "mlx-vlm", "bundled mlx-vlm package broken"),
    ("mlx_vlm.models.gemma4", "mlx-vlm gemma4", "cherry-picked gemma4 dir missing or incomplete — re-sync from an mlx-vlm wheel that has it"),
    ("mlx_vlm.models.gemma3", "mlx-vlm gemma3", "bundled mlx-vlm gemma3 missing"),
    ("mlx_vlm.models.qwen3_vl", "mlx-vlm qwen3_vl", "bundled mlx-vlm qwen3_vl missing"),
    ("mflux", "mflux image runtime", "bundled mflux package missing — Image tab and Server-tab image models will fail before ready"),
    ("mflux.models.qwen21.variants.edit.qwen_image_21_edit", "Qwen Image 2.1", "bundled mflux lacks the native Qwen Image 2.1 adapter"),
    ("mflux.models.ideogram4.variants.txt2img.ideogram4", "Ideogram 4", "bundled mflux lacks the native Ideogram 4 adapter"),
    ("mflux.models.common.config.model_config", "mflux ModelConfig", "mflux install incomplete — cannot resolve local image model configs"),
    ("timm", "timm vision backbone", "bundled timm package missing — Nemotron-Omni image/video media dispatch will fail before generation"),
    ("einops", "einops tensor rearrange", "bundled einops package missing — Nemotron-Omni vision remote code will fail before generation"),
    ("librosa", "librosa audio features", "bundled librosa package missing — Nemotron-Omni session construction will fail before generation"),
    ("jang_tools", "jang-tools", "bundled jang-tools package missing"),
    ("jang_tools.capabilities", "jang_tools.capabilities", "JANG capability stamping/verification helpers missing from bundled jang-tools"),
    ("jang_tools.hy3", "jang_tools.hy3", "Hy3 model-family registration missing from bundled jang-tools"),
    ("mlx_lm.models.hy_v3", "mlx_lm.models.hy_v3", "Hy3 model-family mlx-lm registration missing after importing jang_tools.hy3"),
    ("mlx_lm.models.bailing_hybrid", "mlx_lm.models.bailing_hybrid", "Ling/Bailing hybrid mlx-lm runtime missing — bundle-python.sh must install bailing_hybrid.patched.py"),
    ("jang_tools.hy3.runtime", "jang_tools.hy3.runtime", "Hy3 runtime loader missing from bundled jang-tools"),
    ("jang_tools.laguna.runtime", "jang_tools.laguna.runtime", "Laguna mixed-affine runtime missing from bundled jang-tools"),
    ("jang_tools.mimo_v2.mlx_register", "jang_tools.mimo_v2.mlx_register", "MiMo-V2.5 runtime registration missing from bundled jang-tools"),
    ("mlx_lm.models.mimo_v2", "mlx_lm.models.mimo_v2", "MiMo-V2.5 mlx-lm registration missing after importing jang_tools.mimo_v2.mlx_register"),
    ("vmlx_engine.models.mimo_v26", "vmlx_engine.models.mimo_v26", "MiMo-V2.6 text/vision/video/audio runtime missing from bundle"),
    ("jang_tools.nanbeige", "jang_tools.nanbeige", "Nanbeige package missing from bundled jang-tools"),
    ("jang_tools.nanbeige.model", "jang_tools.nanbeige.model", "Nanbeige looped-transformer model runtime missing from bundled jang-tools"),
    ("jang_tools.nanbeige.mlx_register", "jang_tools.nanbeige.mlx_register", "Nanbeige mlx-lm registration hook missing from bundled jang-tools"),
    ("mlx_lm.models.nanbeige", "mlx_lm.models.nanbeige", "Nanbeige mlx-lm registration missing after importing jang_tools.nanbeige.mlx_register"),
    ("jang_tools.step37.step3p7_mlx", "jang_tools.step37.step3p7_mlx", "Step3p7 source VLM runtime missing from bundled jang-tools"),
    ("jang_tools.load_jangtq", "jang_tools.load_jangtq", "JANGTQ fast-path loader missing from bundled jang-tools"),
    ("jang_tools.topk_override", "jang_tools.topk_override", "JANGTQ top-k runtime override helper missing from bundled jang-tools"),
    ("jang_tools.turboquant.tq_kernel", "jang_tools.turboquant.tq_kernel", "TQ Metal kernel runtime missing from bundled jang-tools"),
    ("jang_tools.turboquant.hadamard_kernel", "hadamard_kernel", "P3 Hadamard kernel missing"),
    ("jang_tools.turboquant.fused_gate_up_kernel", "fused_gate_up_kernel", "P17 fused kernel missing"),
    ("jang_tools.turboquant.gather_tq_kernel", "gather_tq_kernel", "P17 gather kernel missing"),
    ("jang_tools.turboquant.mpp_nax_kernel", "mpp_nax_kernel", "JANGTQ acceleration kernel missing"),
    # Kimi K2.6 runtime — research/KIMI-K2.6-VMLX-INTEGRATION.md §1.1
    ("jang_tools.load_jangtq_kimi_vlm", "jang_tools.load_jangtq_kimi_vlm", "Kimi VL loader missing (kimi_k25 remap + wired_limit + command-buffer split)"),
    ("jang_tools.kimi_prune.generate_vl", "jang_tools.kimi_prune.generate_vl", "Kimi chunked VL generate path missing — required by vmlx_engine.vlm.generate_vl"),
    ("vmlx_engine", "vmlx_engine", "bundled vmlx_engine missing"),
    ("vmlx_engine.models.step3p7_mlx_vlm", "vmlx_engine Step3p7 VLM runtime", "Step3p7 source VLM runtime missing from bundled vmlx_engine"),
    ("vmlx_engine.models.gemma4_unified_register", "vmlx_engine Gemma 4 Unified VLM/audio runtime register", "Gemma 4 Unified source VLM/audio runtime missing from bundled vmlx_engine"),
    ("vmlx_engine.utils.jang_loader", "vmlx_engine jang_loader", "bundled jang_loader missing"),
    ("vmlx_engine.api.ollama_adapter", "vmlx_engine ollama_adapter", "bundled ollama_adapter missing"),
    # Doc §1.3 + §1.4 import paths — shipping these means the Kimi integration
    # doc's code examples work verbatim on the shipped DMG, not just in dev.
    ("vmlx_engine.loaders.load_jangtq", "vmlx_engine.loaders.load_jangtq", "§1.3 text loader re-export missing"),
    ("vmlx_engine.loaders.load_jangtq_vlm", "vmlx_engine.loaders.load_jangtq_vlm", "§1.1 shared VLM loader re-export missing"),
    ("vmlx_engine.loaders.load_jangtq_kimi_vlm", "vmlx_engine.loaders.load_jangtq_kimi_vlm", "§1.4 Kimi VL loader re-export missing"),
    ("vmlx_engine.vlm.generate_vl", "vmlx_engine.vlm.generate_vl", "§1.4 chunked-prefill generate_vl re-export missing"),
    ("vmlx_engine.runtime_patches.kimi_k25_mla", "vmlx_engine.runtime_patches.kimi_k25_mla", "§1.2 Kimi MLA fp32-SDPA runtime-patch installer missing"),
]

failures = []
for mod, label, hint in REQUIRED:
    try:
        __import__(mod)
        print(f"  ok   {label:<40}  ({mod})")
    except Exception as e:
        failures.append((mod, label, hint, e))
        print(f"  FAIL {label:<40}  ({mod})  {type(e).__name__}: {e}")

if failures:
    print()
    print("RELEASE BLOCKED — bundled-python is missing critical modules:")
    for mod, label, hint, e in failures:
        print(f"  - {label}: {hint}")
    sys.exit(1)

# Laguna XS 2.1 JANG_6M has 6-bit per-module weights inside an artifact whose
# top-level affine fallback is 8-bit. Import-only verification cannot catch an
# old runtime that applies the fallback globally and then fails in MLX
# dequantization. Pin the exact packed embedding shape contract used by the
# shipped artifact.
try:
    from jang_tools.laguna.runtime import (
        LAGUNA_MIXED_AFFINE_RUNTIME_VERSION,
        infer_affine_bits_from_shapes,
    )

    if LAGUNA_MIXED_AFFINE_RUNTIME_VERSION < 1:
        raise RuntimeError(
            "LAGUNA_MIXED_AFFINE_RUNTIME_VERSION must be at least 1"
        )
    _laguna_bits = infer_affine_bits_from_shapes(
        (100352, 384),
        (100352, 32),
        group_size=64,
        fallback_bits=8,
    )
    if _laguna_bits != 6:
        raise RuntimeError(
            f"XS 2.1 JANG_6M embedding must derive 6 bits, got {_laguna_bits}"
        )
    print("  ok   Laguna mixed-affine 6-bit runtime contract")
except Exception as e:
    print(f"  FAIL Laguna mixed-affine runtime contract: {type(e).__name__}: {e}")
    sys.exit(1)

# Step-3.7 VLM registration is source-owned until mlx-vlm ships native support.
# Importing the runtime module alone is insufficient; the app loader must be
# able to register both the mlx_vlm model package and processor submodule.
try:
    import importlib.util
    from vmlx_engine.models.mllm import _register_step3p7_mlx_vlm_runtime

    _register_step3p7_mlx_vlm_runtime()
    for _name in (
        "mlx_vlm.models.step3p7",
        "mlx_vlm.models.step3p7.processing_step3",
    ):
        if importlib.util.find_spec(_name) is None:
            print(f"  FAIL Step3p7 mlx-vlm registration missing: {_name}")
            sys.exit(1)
    print("  ok   Step3p7 mlx-vlm registration")
except Exception as e:
    print(f"  FAIL Step3p7 mlx-vlm registration check: {type(e).__name__}: {e}")
    sys.exit(1)

# Gemma 4 Unified source runtime is registered under mlx-vlm until upstream
# ships native support. This catches missing vendored files and broken relative
# imports before a packaged app claims Gemma 4 12B media support.
try:
    import importlib.util
    from vmlx_engine.models.gemma4_unified_register import _register_gemma4_unified_runtime

    _register_gemma4_unified_runtime()
    for _name in (
        "mlx_vlm.models.gemma4_unified",
        "mlx_vlm.models.gemma4_unified.processing_gemma4_unified",
    ):
        if importlib.util.find_spec(_name) is None:
            print(f"  FAIL Gemma 4 Unified mlx-vlm registration missing: {_name}")
            sys.exit(1)
    print("  ok   Gemma 4 Unified mlx-vlm registration")
except Exception as e:
    print(f"  FAIL Gemma 4 Unified mlx-vlm registration check: {type(e).__name__}: {e}")
    sys.exit(1)

# Extra spot-check: load the gemma4 Model class (catches broken relative
# imports that package-level __import__ won't catch).
try:
    from mlx_vlm.models.gemma4 import Model, LanguageModel, VisionModel  # noqa: F401
    print("  ok   gemma4 Model/LanguageModel/VisionModel classes")
except Exception as e:
    print(f"  FAIL gemma4 class import: {type(e).__name__}: {e}")
    sys.exit(1)

try:
    import importlib
    import vmlx_engine

    vmlx_engine._install_mlx_vlm_registry_patches()
    _assistant = importlib.import_module("mlx_vlm.models.gemma4_assistant")
    if not hasattr(_assistant, "Model") or not hasattr(_assistant, "ModelConfig"):
        print("  FAIL Gemma 4 assistant alias missing Model/ModelConfig")
        sys.exit(1)
    _unified_assistant = importlib.import_module(
        "mlx_vlm.speculative.drafters.gemma4_unified_assistant"
    )
    if (
        not hasattr(_unified_assistant, "Model")
        or not hasattr(_unified_assistant, "ModelConfig")
    ):
        print("  FAIL Gemma 4 Unified assistant alias missing Model/ModelConfig")
        sys.exit(1)
    print("  ok   Gemma 4 assistant mlx_vlm.models alias")
except Exception as e:
    print(f"  FAIL Gemma 4 assistant alias check: {type(e).__name__}: {e}")
    sys.exit(1)

# mlxstudio#88: bundled Gemma 4 vision must either carry the compatibility
# wrapper or mlx-vlm's newer native mixed/different-sized-image list path.
try:
    import inspect
    import mlx_vlm.models.gemma4.vision as _g4v
    _src = inspect.getsource(_g4v.VisionModel.__call__)
    _compat = "mlxstudio#88" in _src and "isinstance(v, mx.array)" in _src
    _native = all(
        needle in _src
        for needle in (
            "if isinstance(pixel_values, list):",
            "if not isinstance(img, mx.array):",
            "img = mx.array(img)",
        )
    )
    if _compat or _native:
        implementation = "compatibility wrapper" if _compat else "native mlx-vlm"
        print(
            "  ok   Gemma 4 vision mixed/different-sized image lists "
            f"({implementation})"
        )
    else:
        print(
            "  FAIL Gemma 4 vision mixed-list support missing from bundled "
            "mlx_vlm/models/gemma4/vision.py"
        )
        print("       re-run bundle-python.sh (mlxstudio#88)")
        sys.exit(1)
except Exception as e:
    print(f"  FAIL Gemma 4 vision patch check: {type(e).__name__}: {e}")
    sys.exit(1)

# Kimi K2.6 §1.2 fp32 MLA L==1 SDPA patch must be baked into bundled mlx_lm.
# If this fails, the bundled deepseek_v3.py wasn't patched at build time and
# Kimi K2.6 decode will produce repetition loops after ~14 tokens.
try:
    import inspect
    import mlx_lm.models.deepseek_v3 as _dv3
    _src = inspect.getsource(_dv3.DeepseekV3Attention.__call__)
    if "JANG fast fix" in _src and "q_sdpa" in _src:
        print("  ok   Kimi K2.6 fp32 MLA L==1 SDPA patch in bundled mlx_lm")
    else:
        print("  FAIL Kimi K2.6 fp32 MLA L==1 SDPA patch missing from bundled mlx_lm/models/deepseek_v3.py")
        print("       re-apply research/deepseek_v3_patched.py over it")
        sys.exit(1)
except Exception as e:
    print(f"  FAIL Kimi K2.6 MLA patch check: {type(e).__name__}: {e}")
    sys.exit(1)

# DeepSeek V3.2 / GLM-5.1 and Mistral4 use the same MLA L==1 absorb branch.
# bundle-python.sh vendors deterministic patched source for all three files;
# verify all three so a future mlx-lm reinstall cannot silently drop one.
for _mod_name, _class_name, _label in [
    ("mlx_lm.models.deepseek_v32", "DeepseekV32Attention", "DeepSeek V3.2 / GLM-5.1"),
    ("mlx_lm.models.mistral4", "Mistral4Attention", "Mistral 4"),
]:
    try:
        import importlib
        import inspect

        _mod = importlib.import_module(_mod_name)
        _cls = getattr(_mod, _class_name)
        _src = inspect.getsource(_cls.__call__)
        if "q_sdpa" in _src and "astype(mx.float32)" in _src:
            print(f"  ok   {_label} fp32 MLA L==1 SDPA patch in bundled mlx_lm")
        else:
            print(f"  FAIL {_label} fp32 MLA L==1 SDPA patch missing from bundled {_mod_name}")
            sys.exit(1)
    except Exception as e:
        print(f"  FAIL {_label} MLA patch check: {type(e).__name__}: {e}")
        sys.exit(1)

# mlx_vlm kimi_k25 / MiniCPM-V-4.6 dispatch remaps must be live (installed by
# vmlx_engine.__init__ at import time). Catches a silent regression if the
# remap block ever gets removed.
try:
    import vmlx_engine  # triggers remap install
    from mlx_vlm.utils import MODEL_REMAPPING, get_model_and_args
    from mlx_vlm.prompt_utils import MODEL_CONFIG
    if MODEL_REMAPPING.get("kimi_k25") != "kimi_vl":
        print("  FAIL kimi_k25 → kimi_vl remap missing in mlx_vlm.utils.MODEL_REMAPPING")
        sys.exit(1)
    if "kimi_k25" not in MODEL_CONFIG:
        print("  FAIL kimi_k25 missing in mlx_vlm.prompt_utils.MODEL_CONFIG")
        sys.exit(1)
    if MODEL_REMAPPING.get("minicpmv4_6") != "minicpmo":
        print("  FAIL MiniCPM-V-4.6 → minicpmo remap missing in mlx_vlm.utils.MODEL_REMAPPING")
        sys.exit(1)
    if "minicpmv4_6" not in MODEL_CONFIG:
        print("  FAIL minicpmv4_6 missing in mlx_vlm.prompt_utils.MODEL_CONFIG")
        sys.exit(1)
    _arch, _model_type = get_model_and_args({"model_type": "minicpmv4_6"})
    if _model_type != "minicpmo" or not hasattr(_arch, "Model"):
        print("  FAIL MiniCPM-V-4.6 remap did not resolve mlx_vlm.models.minicpmo")
        sys.exit(1)
    print("  ok   Kimi K2.6 mlx_vlm remap + prompt_utils config")
    print("  ok   MiniCPM-V-4.6 mlx_vlm remap + prompt_utils config")
except Exception as e:
    print(f"  FAIL mlx_vlm remap check: {type(e).__name__}: {e}")
    sys.exit(1)

# MiniCPM-V-4.7 text can load through tokenizer fallback even if its native
# processor is missing. Qualify the actual media processor before signing.
try:
    from transformers import MiniCPMV4_7Processor
    from vmlx_engine.models.minicpmv4_7.processing import MiniCPMV47ProcessorLoader
    assert callable(MiniCPMV4_7Processor.from_pretrained)
    assert callable(MiniCPMV47ProcessorLoader.from_pretrained)
    print("  ok   MiniCPM-V-4.7 native media processor available")
except Exception as e:
    print(f"  FAIL MiniCPM-V-4.7 native media processor: {type(e).__name__}: {e}")
    sys.exit(1)

print()
print("bundled-python: all critical imports ok")
PYEOF
