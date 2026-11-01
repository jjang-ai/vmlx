#!/bin/bash
set -euo pipefail
R20_FIXED_PATH="/opt/homebrew/bin:/usr/bin:/bin:/usr/sbin:/sbin"
export PATH="$R20_FIXED_PATH"

# Build the two public macOS DMG flavors for the same source checkout.
#
# vmlx#169: macosx_26 MLX wheels ship Metal language 4.0 kernels that are
# valid on Tahoe but fail on Sequoia. Release packaging must therefore produce
# two clearly named DMGs from the same source:
#   - sequoia: macosx_14 wheels, works on Sonoma 14.5+, Sequoia 15, and Tahoe
#   - tahoe: native macosx_26 wheels, Tahoe-only
#
# This script only builds local artifacts. It does not tag, upload, publish,
# notarize, update the updater manifest, or create a GitHub release.
umask 077

SCRIPT_DIR="$(cd -P "$(dirname "$0")" && pwd -P)"
PANEL_DIR="$(cd -P "$SCRIPT_DIR/.." && pwd -P)"
ROOT_DIR="$(cd -P "$PANEL_DIR/.." && pwd -P)"
REQUESTED_RELEASE_SCOPE="${VMLX_RELEASE_SCOPE:-${VMLINUX_RELEASE_SCOPE:-}}"
RELEASE_SCOPE="$REQUESTED_RELEASE_SCOPE"
REQUESTED_FLAVOR="${1:-all}"
DEV_UNSIGNED="${VMLX_DEV_UNSIGNED:-0}"
AUTHORITATIVE_PYTHON="$ROOT_DIR/.venv/bin/python"
PYTHON_ACTION_HELPER="$PANEL_DIR/scripts/release-python-action.cjs"

GIT_BIN="/usr/bin/git"
NODE_BIN="/opt/homebrew/bin/node"
NPM_BIN="/opt/homebrew/bin/npm"
NPX_BIN="/opt/homebrew/bin/npx"
SHASUM_BIN="/usr/bin/shasum"
AWK_BIN="/usr/bin/awk"
FILE_BIN="/usr/bin/file"
FIND_BIN="/usr/bin/find"
ASAR_BIN="$PANEL_DIR/node_modules/@electron/asar/bin/asar.js"
APP_BUILDER_BIN="$PANEL_DIR/node_modules/app-builder-bin/mac/app-builder_arm64"
ELECTRON_BUILDER_BIN="$PANEL_DIR/node_modules/electron-builder/cli.js"

cd "$PANEL_DIR"

remove_owned_release_tree_with_retry() {
  local target="$1"
  local attempt

  for attempt in 1 2 3 4 5 6 7 8; do
    /bin/rm -rf -- "$target" || true
    if [[ ! -e "$target" && ! -L "$target" ]]; then
      return 0
    fi
    if [[ "$attempt" -lt 8 ]]; then
      echo "WARNING: retrying transient owned release cleanup ($attempt/8): $target" >&2
      /bin/sleep 0.1
    fi
  done

  echo "ERROR: owned release cleanup did not converge after 8 attempts: $target" >&2
  return 1
}

assert_exact_release_tool_path() {
  local name="$1"
  local expected="$2"
  local resolved
  resolved="$(type -P "$name" || true)"
  if [[ "$(type -t "$name" || true)" != "file" || "$resolved" != "$expected" ]]; then
    echo "ERROR: release tool $name path changed; expected $expected, found ${resolved:-missing}" >&2
    exit 1
  fi
  if [[ ! -x "$expected" ]]; then
    echo "ERROR: required release tool is unavailable: $expected" >&2
    exit 1
  fi
}

assert_exact_release_tool_path git "$GIT_BIN"
assert_exact_release_tool_path node "$NODE_BIN"
assert_exact_release_tool_path npm "$NPM_BIN"
assert_exact_release_tool_path npx "$NPX_BIN"
assert_exact_release_tool_path shasum "$SHASUM_BIN"
assert_exact_release_tool_path awk "$AWK_BIN"
assert_exact_release_tool_path file "$FILE_BIN"
assert_exact_release_tool_path find "$FIND_BIN"

if [[ -n "${VMLX_RELEASE_SCOPE:-}" && -n "${VMLINUX_RELEASE_SCOPE:-}" ]] \
  && [[ "$VMLX_RELEASE_SCOPE" != "$VMLINUX_RELEASE_SCOPE" ]]; then
  echo "ERROR: VMLX_RELEASE_SCOPE and VMLINUX_RELEASE_SCOPE disagree" >&2
  exit 1
fi

VERSION="$("$NODE_BIN" -p "require('./package.json').version")"
if [[ "$REQUESTED_RELEASE_SCOPE" == "production" ]]; then
  # Accept any well-formed release version. Pinning this to a single literal
  # meant every release after that one silently built through the unhardened
  # scope. Tree-wide version agreement is enforced by the Python preflight.
  if [[ "$VERSION" =~ ^[0-9]+\.[0-9]+\.[0-9]+$ ]]; then
    RELEASE_SCOPE="r20_production"
  else
    echo "ERROR: public production packaging requires a release version like 1.2.3, got $VERSION" >&2
    exit 1
  fi
fi

if [[ "$DEV_UNSIGNED" != "0" && "$DEV_UNSIGNED" != "1" ]]; then
  echo "ERROR: VMLX_DEV_UNSIGNED must be 0 or 1" >&2
  exit 1
fi
if [[ "$DEV_UNSIGNED" == "1" && "$RELEASE_SCOPE" != "codex_ui_only" ]]; then
  echo "ERROR: unsigned DMGs are allowed only with VMLX_RELEASE_SCOPE=codex_ui_only" >&2
  exit 1
fi
if [[ "$DEV_UNSIGNED" == "1" ]]; then
  export CSC_IDENTITY_AUTO_DISCOVERY=false
fi

if [[ "$RELEASE_SCOPE" == "r20_production" ]]; then
  if [[ "$REQUESTED_FLAVOR" != "all" ]]; then
    echo "ERROR: vMLX production packaging must build both Sequoia and Tahoe via flavor=all" >&2
    exit 1
  fi
  if [[ -n "${VMLX_RELEASE_OUTPUT_DIR:-}" || -n "${VMLINUX_RELEASE_OUTPUT_DIR:-}" ]]; then
    echo "ERROR: vMLX production output is fixed at $PANEL_DIR/release; release output overrides are forbidden" >&2
    exit 1
  fi
  DIST_DIR="$PANEL_DIR/release"
else
  DIST_DIR="${VMLX_RELEASE_OUTPUT_DIR:-${VMLINUX_RELEASE_OUTPUT_DIR:-$PANEL_DIR/release}}"
  if [[ "$DIST_DIR" != /* ]]; then
    DIST_DIR="$PANEL_DIR/$DIST_DIR"
  fi
fi

# Release builds must never borrow dependencies from another checkout. A
# symlinked node_modules can silently package stale Electron/native code and,
# in particular, makes electron-builder rebuild better-sqlite3 in the wrong
# tree. Require a checkout-local install so the signed artifact has one
# auditable source/dependency root.
if [[ -L "$PANEL_DIR/node_modules" ]]; then
  echo "ERROR: release node_modules must not be a symlink: $PANEL_DIR/node_modules" >&2
  echo "       Unlink it and run npm ci in this release checkout." >&2
  exit 1
fi
if [[ ! -d "$PANEL_DIR/node_modules" ]]; then
  echo "ERROR: release node_modules is missing: $PANEL_DIR/node_modules" >&2
  echo "       Run npm ci in this release checkout before packaging." >&2
  exit 1
fi
NODE_MODULES_REAL="$(cd "$PANEL_DIR/node_modules" && pwd -P)"
if [[ "$NODE_MODULES_REAL" != "$PANEL_DIR/node_modules" ]]; then
  echo "ERROR: release node_modules resolves outside this checkout: $NODE_MODULES_REAL" >&2
  exit 1
fi

PREPACKAGE_READY_MANIFEST_OUT="${VMLX_PREPACKAGE_READY_MANIFEST_OUT:-${VMLINUX_PREPACKAGE_READY_MANIFEST_OUT:-$ROOT_DIR/build/current-release-regression-manifest-pre-dmg-release-build.json}}"
PRIVATE_EVIDENCE_ROOT="${VMLX_R20_PRIVATE_EVIDENCE_ROOT:-${VMLINUX_R20_PRIVATE_EVIDENCE_ROOT:-}}"
R20_PRE_NOTARY_MANIFEST_OUT="${VMLX_R20_PRE_NOTARY_MANIFEST_OUT:-${VMLINUX_R20_PRE_NOTARY_MANIFEST_OUT:-}}"
EXPECTED_APPLE_TEAM_ID="55KGF2S5AY"
EXPECTED_CODESIGN_IDENTITY="Developer ID Application: ShieldStack LLC (55KGF2S5AY)"
EXPECTED_CSC_NAME="ShieldStack LLC (55KGF2S5AY)"
ARTIFACT_CHAIN_HELPER="$ROOT_DIR/tests/cross_matrix/run_packaged_integrity_contract.py"
RELEASE_CODESIGN_IDENTITY="${VMLX_RELEASE_CODESIGN_IDENTITY:-${VMLINUX_RELEASE_CODESIGN_IDENTITY:-$EXPECTED_CODESIGN_IDENTITY}}"
APPLE_CODESIGN="/usr/bin/codesign"
APPLE_HDIUTIL="/usr/bin/hdiutil"

assert_unshadowed_tool() {
  local name="$1"
  local expected="$2"
  local resolved
  resolved="$(type -P "$name" || true)"
  if [[ "$(type -t "$name" || true)" != "file" || "$resolved" != "$expected" ]]; then
    echo "ERROR: release tool $name is shadowed; expected $expected, found ${resolved:-missing}" >&2
    exit 1
  fi
  if [[ ! -x "$expected" ]]; then
    echo "ERROR: required release tool is unavailable: $expected" >&2
    exit 1
  fi
}

assert_unshadowed_tool codesign "$APPLE_CODESIGN"
assert_unshadowed_tool hdiutil "$APPLE_HDIUTIL"

if [[ -n "${VMLX_R20_PRE_NOTARY_MANIFEST_OUT:-}" ]] \
  && [[ -n "${VMLINUX_R20_PRE_NOTARY_MANIFEST_OUT:-}" ]] \
  && [[ "$VMLX_R20_PRE_NOTARY_MANIFEST_OUT" != "$VMLINUX_R20_PRE_NOTARY_MANIFEST_OUT" ]]; then
  echo "ERROR: VMLX_R20_PRE_NOTARY_MANIFEST_OUT and VMLINUX_R20_PRE_NOTARY_MANIFEST_OUT disagree" >&2
  exit 1
fi
if [[ -n "${VMLX_R20_PRIVATE_EVIDENCE_ROOT:-}" ]] \
  && [[ -n "${VMLINUX_R20_PRIVATE_EVIDENCE_ROOT:-}" ]] \
  && [[ "$VMLX_R20_PRIVATE_EVIDENCE_ROOT" != "$VMLINUX_R20_PRIVATE_EVIDENCE_ROOT" ]]; then
  echo "ERROR: VMLX_R20_PRIVATE_EVIDENCE_ROOT and VMLINUX_R20_PRIVATE_EVIDENCE_ROOT disagree" >&2
  exit 1
fi
export VMLX_RELEASE_SCOPE="$RELEASE_SCOPE"

if [[ "$RELEASE_SCOPE" == "r20_production" ]]; then
  if [[ -n "${VMLX_R20_EXPECTED_VERSION:-}" ]] \
    && [[ "$VMLX_R20_EXPECTED_VERSION" != "$VERSION" ]]; then
    echo "ERROR: VMLX_R20_EXPECTED_VERSION does not match package version $VERSION" >&2
    exit 1
  fi
  export VMLX_R20_EXPECTED_VERSION="$VERSION"
  if [[ ! "$VERSION" =~ ^[0-9]+\.[0-9]+\.[0-9]+$ ]]; then
    echo "ERROR: VMLX_RELEASE_SCOPE=r20_production requires a release version like 1.2.3, found $VERSION" >&2
    exit 1
  fi
  if [[ -n "${PYTHON:-}" || -n "${PYTHONHOME:-}" || -n "${PYTHONPATH:-}" ]]; then
    echo "ERROR: vMLX release Python cannot be overridden by PYTHON, PYTHONHOME, or PYTHONPATH" >&2
    exit 1
  fi
  if [[ -n "${VIRTUAL_ENV:-}" && "$VIRTUAL_ENV" != "$ROOT_DIR/.venv" ]]; then
    echo "ERROR: vMLX release VIRTUAL_ENV is not the authoritative repository venv" >&2
    exit 1
  fi
  if [[ ! -x "$AUTHORITATIVE_PYTHON" ]]; then
    echo "ERROR: missing authoritative vMLX release Python: $AUTHORITATIVE_PYTHON" >&2
    exit 1
  fi
  PYTHON_BIN="$AUTHORITATIVE_PYTHON"
  if [[ "$RELEASE_CODESIGN_IDENTITY" != "$EXPECTED_CODESIGN_IDENTITY" ]]; then
    echo "ERROR: vMLX production packaging requires $EXPECTED_CODESIGN_IDENTITY" >&2
    exit 1
  fi
  CONFIGURED_APPLE_TEAM_ID="$(
    "$NODE_BIN" -p "require('./package.json').build.mac.notarize.teamId"
  )"
  if [[ "$CONFIGURED_APPLE_TEAM_ID" != "$EXPECTED_APPLE_TEAM_ID" ]]; then
    echo "ERROR: vMLX package notarization team must be $EXPECTED_APPLE_TEAM_ID" >&2
    exit 1
  fi
  # electron-builder treats CSC_NAME as a certificate selector and rejects the
  # full codesign Authority value when it includes the certificate-type prefix.
  export CSC_NAME="$EXPECTED_CSC_NAME"
else
  PYTHON_BIN="${PYTHON:-$AUTHORITATIVE_PYTHON}"
  if [[ ! -x "$PYTHON_BIN" ]]; then
    PYTHON_BIN="${PYTHON:-python3}"
  fi
fi

if [[ "$VERSION" == "1.6.20" && "$RELEASE_SCOPE" != "r20_production" ]]; then
  echo "ERROR: vMLX 1.6.20 may only be packaged with VMLX_RELEASE_SCOPE=r20_production" >&2
  echo "       Legacy, empty, and codex_ui_only scopes cannot certify the .19 release." >&2
  exit 1
fi

bind_release_python_action() {
  if [[ "$RELEASE_SCOPE" != "r20_production" ]]; then
    return 0
  fi
  if [[ ! -f "$PYTHON_ACTION_HELPER" ]]; then
    echo "ERROR: missing independent release Python action helper: $PYTHON_ACTION_HELPER" >&2
    exit 1
  fi
  local binding
  binding="$("$NODE_BIN" "$PYTHON_ACTION_HELPER" bind --python "$AUTHORITATIVE_PYTHON")"
  export VMLX_R20_RELEASE_PYTHON_PLAN="$(
    "$NODE_BIN" -e 'process.stdout.write(JSON.parse(process.argv[1]).planPath)' "$binding"
  )"
  export VMLX_R20_RELEASE_PYTHON_PLAN_SHA256="$(
    "$NODE_BIN" -e 'process.stdout.write(JSON.parse(process.argv[1]).planSha256)' "$binding"
  )"
  export VMLX_R20_RELEASE_PYTHON_ACTION="$(
    "$NODE_BIN" -e 'process.stdout.write(JSON.parse(process.argv[1]).actionPath)' "$binding"
  )"
  export VMLX_R20_RELEASE_PYTHON_SOURCE_SHA256="$(
    "$NODE_BIN" -e 'process.stdout.write(JSON.parse(process.argv[1]).sourceSha256)' "$binding"
  )"
  export VMLX_R20_RELEASE_PYTHON_PYVENV_SHA256="$(
    "$NODE_BIN" -e 'process.stdout.write(JSON.parse(process.argv[1]).pyvenvSha256)' "$binding"
  )"
  export VMLX_R20_RELEASE_PYTHON="$AUTHORITATIVE_PYTHON"
  if [[ ! "$VMLX_R20_RELEASE_PYTHON_PLAN_SHA256" =~ ^[0-9a-f]{64}$ ]] \
    || [[ ! "$VMLX_R20_RELEASE_PYTHON_SOURCE_SHA256" =~ ^[0-9a-f]{64}$ ]] \
    || [[ ! "$VMLX_R20_RELEASE_PYTHON_PYVENV_SHA256" =~ ^[0-9a-f]{64}$ ]]; then
    echo "ERROR: independent release Python binding is malformed" >&2
    exit 1
  fi
}

run_release_python() {
  if [[ "$RELEASE_SCOPE" == "r20_production" ]]; then
    "$NODE_BIN" "$PYTHON_ACTION_HELPER" run --cwd "$PWD" -- "$@"
  else
    "$PYTHON_BIN" "$@"
  fi
}

cleanup_release_python_action() {
  if [[ "$RELEASE_SCOPE" == "r20_production" ]] \
    && [[ -n "${VMLX_R20_RELEASE_PYTHON_PLAN:-}" ]] \
    && [[ -n "${VMLX_R20_RELEASE_PYTHON_PLAN_SHA256:-}" ]]; then
    "$NODE_BIN" "$PYTHON_ACTION_HELPER" cleanup >/dev/null 2>&1 || true
  fi
}

bind_release_python_action
if [[ "$RELEASE_SCOPE" == "r20_production" ]]; then
  trap cleanup_release_python_action EXIT
fi

assert_r20_release_output_safe() {
  if [[ "$RELEASE_SCOPE" != "r20_production" ]]; then
    return 0
  fi
  run_release_python -I - "$ROOT_DIR" "$PANEL_DIR" "$DIST_DIR" <<'PY'
import pathlib
import stat
import sys

root, panel, output = map(pathlib.Path, sys.argv[1:])
if root.resolve(strict=True) != root or panel.resolve(strict=True) != panel:
    raise SystemExit("release checkout root/panel is not a canonical physical path")
expected = panel / "release"
if output != expected:
    raise SystemExit(f"production release output is not canonical: {output}")
for candidate in (root, panel):
    for component in list(reversed(candidate.parents)) + [candidate]:
        try:
            metadata = component.lstat()
        except FileNotFoundError:
            continue
        if stat.S_ISLNK(metadata.st_mode):
            raise SystemExit(f"release output has symlinked ancestor: {component}")
if output.exists() or output.is_symlink():
    metadata = output.lstat()
    if stat.S_ISLNK(metadata.st_mode) or not stat.S_ISDIR(metadata.st_mode):
        raise SystemExit(f"production release output is not a real directory: {output}")
    if output.resolve(strict=True) != expected:
        raise SystemExit(f"production release output resolves outside panel/release: {output}")
PY
}

assert_r20_release_output_safe

GIT_REALPATH="$GIT_BIN"
GIT_SHA256=""
NODE_REALPATH="$NODE_BIN"
NODE_SHA256=""
NPM_REALPATH="$NPM_BIN"
NPM_SHA256=""
NPX_REALPATH="$NPX_BIN"
NPX_SHA256=""
SHASUM_REALPATH="$SHASUM_BIN"
SHASUM_SHA256=""
AWK_REALPATH="$AWK_BIN"
AWK_SHA256=""
FILE_REALPATH="$FILE_BIN"
FILE_SHA256=""
FIND_REALPATH="$FIND_BIN"
FIND_SHA256=""
ASAR_REALPATH="$ASAR_BIN"
ASAR_SHA256=""
APP_BUILDER_REALPATH="$APP_BUILDER_BIN"
APP_BUILDER_SHA256=""
ELECTRON_BUILDER_REALPATH="$ELECTRON_BUILDER_BIN"
ELECTRON_BUILDER_SHA256=""
R20_TOOLCHAIN_PLAN_PATH="$ROOT_DIR/build/r20-release-toolchain-plan.json"
R20_TOOLCHAIN_PLAN_SHA256=""
R20_BUILD_PLAN_PATH="$ROOT_DIR/build/r20-release-driver-plan.json"

establish_r20_build_toolchain_provenance() {
  if [[ "$RELEASE_SCOPE" != "r20_production" ]]; then
    return 0
  fi

  local name
  local path
  local identity
  local realpath
  local digest
  while IFS=$'\t' read -r name path; do
    identity="$(
      run_release_python -I - "$path" <<'PY'
import hashlib
import os
import pathlib
import stat
import sys

invocation = pathlib.Path(sys.argv[1])
realpath = invocation.resolve(strict=True)
metadata = realpath.stat()
if not stat.S_ISREG(metadata.st_mode) or not os.access(realpath, os.X_OK):
    raise SystemExit(f"release tool is not an executable regular file: {realpath}")
digest = hashlib.sha256()
with realpath.open("rb") as handle:
    for chunk in iter(lambda: handle.read(1024 * 1024), b""):
        digest.update(chunk)
print(f"{realpath}\t{digest.hexdigest()}")
PY
    )"
    IFS=$'\t' read -r realpath digest <<<"$identity"
    if [[ -z "$realpath" || ! "$digest" =~ ^[0-9a-f]{64}$ ]]; then
      echo "ERROR: could not pin release tool identity for $name" >&2
      exit 1
    fi
    printf -v "${name}_REALPATH" '%s' "$realpath"
    printf -v "${name}_SHA256" '%s' "$digest"
    export "VMLX_R20_TOOL_${name}_PATH=$path"
    export "VMLX_R20_TOOL_${name}_REALPATH=$realpath"
    export "VMLX_R20_TOOL_${name}_SHA256=$digest"
  done <<EOF
GIT	$GIT_BIN
NODE	$NODE_BIN
NPM	$NPM_BIN
NPX	$NPX_BIN
SHASUM	$SHASUM_BIN
AWK	$AWK_BIN
FILE	$FILE_BIN
FIND	$FIND_BIN
ASAR	$ASAR_BIN
APP_BUILDER	$APP_BUILDER_BIN
ELECTRON_BUILDER	$ELECTRON_BUILDER_BIN
EOF
  export VMLX_R20_FIXED_PATH="$R20_FIXED_PATH"
}

assert_r20_tool_identity() {
  local name="$1"
  if [[ "$RELEASE_SCOPE" != "r20_production" ]]; then
    return 0
  fi
  local realpath_var="${name}_REALPATH"
  local sha_var="${name}_SHA256"
  local expected_realpath="${!realpath_var}"
  local expected_sha256="${!sha_var}"
  run_release_python -I - "$expected_realpath" "$expected_sha256" "$name" <<'PY'
import hashlib
import os
import pathlib
import stat
import sys

path = pathlib.Path(sys.argv[1])
expected_sha256 = sys.argv[2]
name = sys.argv[3]
realpath = path.resolve(strict=True)
if realpath != path:
    raise SystemExit(f"pinned {name} realpath changed: {path} -> {realpath}")
metadata = realpath.stat()
if not stat.S_ISREG(metadata.st_mode) or not os.access(realpath, os.X_OK):
    raise SystemExit(f"pinned {name} is not an executable regular file")
digest = hashlib.sha256()
with realpath.open("rb") as handle:
    for chunk in iter(lambda: handle.read(1024 * 1024), b""):
        digest.update(chunk)
if digest.hexdigest() != expected_sha256:
    raise SystemExit(f"pinned {name} SHA-256 changed")
PY
}

assert_r20_toolchain_identity() {
  local name
  for name in \
    GIT NODE NPM NPX SHASUM AWK FILE FIND \
    ASAR APP_BUILDER ELECTRON_BUILDER; do
    assert_r20_tool_identity "$name"
  done
}

write_r20_toolchain_plan() {
  if [[ "$RELEASE_SCOPE" != "r20_production" ]]; then
    return 0
  fi
  mkdir -p "$(dirname "$R20_TOOLCHAIN_PLAN_PATH")"
  rm -f "$R20_TOOLCHAIN_PLAN_PATH"
  R20_TOOLCHAIN_PLAN_SHA256="$(
    run_release_python -I - "$R20_TOOLCHAIN_PLAN_PATH" "$VERSION" <<'PY'
import hashlib
import json
import os
import sys
from pathlib import Path

output = Path(sys.argv[1]).absolute()
version = sys.argv[2]
tool_names = (
    "GIT", "NODE", "NPM", "NPX", "SHASUM", "AWK", "FILE", "FIND",
    "ASAR", "APP_BUILDER", "ELECTRON_BUILDER",
)
payload = {
    "schema_version": 1,
    "scope": "r20_production_toolchain",
    "version": version,
    "fixed_path": os.environ["VMLX_R20_FIXED_PATH"],
    "tools": {
        name.lower(): {
            "path": os.environ[f"VMLX_R20_TOOL_{name}_PATH"],
            "realpath": os.environ[f"VMLX_R20_TOOL_{name}_REALPATH"],
            "sha256": os.environ[f"VMLX_R20_TOOL_{name}_SHA256"],
        }
        for name in tool_names
    },
}
encoded = (json.dumps(payload, sort_keys=True, separators=(",", ":")) + "\n").encode()
temporary = output.with_name(f".{output.name}.{os.getpid()}.tmp")
fd = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
try:
    with os.fdopen(fd, "wb") as handle:
        handle.write(encoded)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, output)
    os.chmod(output, 0o400)
finally:
    try:
        temporary.unlink()
    except FileNotFoundError:
        pass
print(hashlib.sha256(encoded).hexdigest())
PY
  )"
  if [[ ! "$R20_TOOLCHAIN_PLAN_SHA256" =~ ^[0-9a-f]{64}$ ]]; then
    echo "ERROR: vMLX toolchain plan hash is invalid" >&2
    exit 1
  fi
}

run_bound_release_action() {
  local document="$1"
  local expected_document_sha256="$2"
  local action="$3"
  shift 3
  run_release_python "$ARTIFACT_CHAIN_HELPER" artifact-chain \
    run-bound-tool-action \
    --binding-kind plan \
    --document "$document" \
    --expected-document-sha256 "$expected_document_sha256" \
    --action "$action" \
    --cwd "$PANEL_DIR" \
    -- "$@"
}

capture_bound_release_action() {
  local document="$1"
  local expected_document_sha256="$2"
  local action="$3"
  shift 3
  local payload
  payload="$(
    run_release_python "$ARTIFACT_CHAIN_HELPER" artifact-chain \
      run-bound-tool-action \
      --binding-kind plan \
      --document "$document" \
      --expected-document-sha256 "$expected_document_sha256" \
      --action "$action" \
      --cwd "$PANEL_DIR" \
      --capture-output \
      -- "$@"
  )"
  # Captured `find` output can describe thousands of bundled files. Passing
  # the JSON envelope as a command-line argument can exceed macOS ARG_MAX.
  # Stream it over stdin so verification remains bounded by memory, not argv.
  printf '%s' "$payload" |
    run_release_python -I -c \
      'import json,sys; sys.stdout.buffer.write(json.load(sys.stdin)["stdout"].encode("utf-8", "surrogateescape"))'
}

run_toolchain_action() {
  local action="$1"
  shift
  if [[ "$RELEASE_SCOPE" != "r20_production" ]]; then
    case "$action" in
      npm) "$NPM_BIN" "$@" ;;
      find) "$FIND_BIN" "$@" ;;
      *)
        echo "ERROR: unsupported non-production toolchain action: $action" >&2
        return 2
        ;;
    esac
    return
  fi
  run_bound_release_action \
    "$R20_TOOLCHAIN_PLAN_PATH" \
    "$R20_TOOLCHAIN_PLAN_SHA256" \
    "$action" \
    "$@"
}

capture_toolchain_action() {
  local action="$1"
  shift
  if [[ "$RELEASE_SCOPE" != "r20_production" ]]; then
    case "$action" in
      git) "$GIT_BIN" "$@" ;;
      shasum) "$SHASUM_BIN" "$@" ;;
      awk) "$AWK_BIN" "$@" ;;
      file) "$FILE_BIN" "$@" ;;
      find) "$FIND_BIN" "$@" ;;
      *)
        echo "ERROR: unsupported non-production captured action: $action" >&2
        return 2
        ;;
    esac
    return
  fi
  capture_bound_release_action \
    "$R20_TOOLCHAIN_PLAN_PATH" \
    "$R20_TOOLCHAIN_PLAN_SHA256" \
    "$action" \
    "$@"
}

toolchain_sha256() {
  local path="$1"
  local output
  local digest
  local remainder
  output="$(capture_toolchain_action shasum -a 256 "$path")"
  IFS=' ' read -r digest remainder <<<"$output"
  if [[ ! "$digest" =~ ^[0-9a-f]{64}$ ]]; then
    echo "ERROR: pinned shasum returned an invalid digest for $path" >&2
    exit 1
  fi
  printf '%s\n' "$digest"
}

run_driver_plan_action() {
  local action="$1"
  shift
  if [[ -n "${USE_SYSTEM_APP_BUILDER:-}" ]]; then
    echo "ERROR: vMLX production packaging forbids USE_SYSTEM_APP_BUILDER" >&2
    exit 1
  fi
  run_bound_release_action \
    "$R20_BUILD_PLAN_PATH" \
    "${VMLX_R20_RELEASE_PLAN_SHA256:?missing release-driver plan digest}" \
    "$action" \
    "$@"
}

capture_driver_plan_action() {
  local action="$1"
  shift
  capture_bound_release_action \
    "$R20_BUILD_PLAN_PATH" \
    "${VMLX_R20_RELEASE_PLAN_SHA256:?missing release-driver plan digest}" \
    "$action" \
    "$@"
}

assert_asar_excludes_finder_metadata() {
  local archive="$1"
  local entry
  local listing

  listing="$(capture_driver_plan_action asar list "$archive")"
  while IFS= read -r entry; do
    case "$entry" in
      */.DS_Store|.DS_Store)
        echo "ERROR: app.asar contains forbidden Finder metadata: $archive" >&2
        return 1
        ;;
    esac
  done <<<"$listing"
}

run_electron_builder_action() {
  if [[ "$RELEASE_SCOPE" == "r20_production" ]]; then
    run_driver_plan_action electron-builder "$@"
  else
    "$NODE_BIN" "$ELECTRON_BUILDER_BIN" "$@"
  fi
}

cleanup_r20_release_plans() {
  if [[ "$RELEASE_SCOPE" == "r20_production" ]]; then
    rm -f "$R20_TOOLCHAIN_PLAN_PATH" "$R20_BUILD_PLAN_PATH"
    if [[ -n "${R20_ATTESTATION_EXTRACT_ROOT:-}" ]]; then
      remove_owned_release_tree_with_retry "$R20_ATTESTATION_EXTRACT_ROOT"
    fi
  fi
}

establish_r20_release_python_provenance() {
  if [[ "$RELEASE_SCOPE" != "r20_production" ]]; then
    return 0
  fi

  local provenance
  local init_path
  local server_path
  local prefix_path
  provenance="$(
    run_release_python -I - \
      "$ROOT_DIR" \
      "$AUTHORITATIVE_PYTHON" \
      "$VERSION" <<'PY'
import importlib.util
import pathlib
import sys

root = pathlib.Path(sys.argv[1]).resolve()
expected_python = pathlib.Path(sys.argv[2]).absolute()
expected_prefix = (root / ".venv").resolve()
pyvenv_config = expected_prefix / "pyvenv.cfg"
expected_version = sys.argv[3]
actual_python = pathlib.Path(sys.executable).absolute()
if actual_python != expected_python:
    raise SystemExit(
        f"release Python executable mismatch: {actual_python} != {expected_python}"
    )
if pathlib.Path(sys.prefix).resolve() != expected_prefix:
    raise SystemExit(
        f"release Python prefix mismatch: {pathlib.Path(sys.prefix).resolve()} "
        f"!= {expected_prefix}"
    )
if not pyvenv_config.is_file():
    raise SystemExit(f"release Python is missing {pyvenv_config}")

import vmlx_engine

init_path = pathlib.Path(vmlx_engine.__file__).resolve()
server_spec = importlib.util.find_spec("vmlx_engine.server")
server_path = (
    pathlib.Path(server_spec.origin).resolve()
    if server_spec is not None and server_spec.origin
    else None
)
expected_init = (root / "vmlx_engine" / "__init__.py").resolve()
expected_server = (root / "vmlx_engine" / "server.py").resolve()
if init_path != expected_init or server_path != expected_server:
    raise SystemExit("release Python imports vmlx_engine outside the release checkout")
if getattr(vmlx_engine, "__version__", None) != expected_version:
    raise SystemExit("release Python imported vmlx_engine version does not match package")

print("\t".join((str(init_path), str(server_path), str(pathlib.Path(sys.prefix).resolve()))))
PY
  )"
  IFS=$'\t' read -r init_path server_path prefix_path <<<"$provenance"
  if [[ "$init_path" != "$ROOT_DIR/vmlx_engine/__init__.py" ]] \
    || [[ "$server_path" != "$ROOT_DIR/vmlx_engine/server.py" ]] \
    || [[ "$prefix_path" != "$ROOT_DIR/.venv" ]]; then
    echo "ERROR: release Python import/prefix attestation is not source-exact" >&2
    exit 1
  fi
  VMLX_R20_RELEASE_PYTHON_INIT_SHA256="$(toolchain_sha256 "$init_path")"
  VMLX_R20_RELEASE_PYTHON_SERVER_SHA256="$(toolchain_sha256 "$server_path")"
  VMLX_R20_RELEASE_PYTHON_EXECUTABLE_SHA256="$VMLX_R20_RELEASE_PYTHON_SOURCE_SHA256"
  if [[ ! "$VMLX_R20_RELEASE_PYTHON_INIT_SHA256" =~ ^[0-9a-f]{64}$ ]] \
    || [[ ! "$VMLX_R20_RELEASE_PYTHON_SERVER_SHA256" =~ ^[0-9a-f]{64}$ ]] \
    || [[ ! "$VMLX_R20_RELEASE_PYTHON_EXECUTABLE_SHA256" =~ ^[0-9a-f]{64}$ ]] \
    || [[ ! "$VMLX_R20_RELEASE_PYTHON_PYVENV_SHA256" =~ ^[0-9a-f]{64}$ ]]; then
    echo "ERROR: vMLX release Python provenance hashes are invalid" >&2
    exit 1
  fi
  export VMLX_R20_RELEASE_PYTHON="$AUTHORITATIVE_PYTHON"
  export VMLX_R20_RELEASE_PYTHON_INIT_SHA256
  export VMLX_R20_RELEASE_PYTHON_SERVER_SHA256
  export VMLX_R20_RELEASE_PYTHON_EXECUTABLE_SHA256
  export VMLX_R20_RELEASE_PYTHON_PYVENV_SHA256
}

establish_r20_build_toolchain_provenance
assert_r20_toolchain_identity
write_r20_toolchain_plan
establish_r20_release_python_provenance
if [[ "$RELEASE_SCOPE" == "r20_production" ]]; then
  cleanup_r20_release_state() {
    cleanup_r20_release_plans
    cleanup_release_python_action
  }
  trap cleanup_r20_release_state EXIT
fi

echo "==> Checking pre-package release ledger before public DMG build"
case "$RELEASE_SCOPE" in
  codex_ui_only)
    # v1.6.0 release path: live validation is Codex-driven at the END of the
    # release chain (drives dev-build UI over CDP against real engine on
    # test-host.local), not via the offline proof-artifact manifest gate.
    # The historical ledger tracks proof-artifacts from named live matrix
    # runs; those aren't produced under this workflow. Fail-open the ledger
    # check for this scope; Codex validation is the substantive gate.
    echo "    RELEASE_SCOPE=codex_ui_only: skipping offline manifest gate."
    echo "    Codex UI validation on the built DMG is the substantive gate."
    ;;
  mm3_gemma_vl)
    (
      cd "$ROOT_DIR"
      if [[ "$VERSION" == "1.5.66" || "$VERSION" == "1.5.67" ]]; then
        run_release_python "panel/scripts/scoped-release-preflight-66.py" \
          --expected-version "$VERSION" \
          --out "$PREPACKAGE_READY_MANIFEST_OUT"
      elif [[ "$VERSION" == "1.5.65" ]]; then
        run_release_python "panel/scripts/scoped-release-preflight-65.py" \
          --out "$PREPACKAGE_READY_MANIFEST_OUT"
      else
        run_release_python "panel/scripts/scoped-release-preflight.py" \
          --scope mm3_gemma_vl \
          --out "$PREPACKAGE_READY_MANIFEST_OUT"
      fi
    )
    ;;
  r16_parser_cache)
    (
      cd "$ROOT_DIR"
      run_release_python "panel/scripts/scoped-release-preflight-16.py" \
        --expected-version "$VERSION" \
        --out "$PREPACKAGE_READY_MANIFEST_OUT"
    )
    ;;
  r17_consolidation)
    (
      cd "$ROOT_DIR"
      run_release_python "panel/scripts/scoped-release-preflight-17.py" \
        --expected-version "$VERSION" \
        --out "$PREPACKAGE_READY_MANIFEST_OUT"
    )
    ;;
  r20_production)
    (
      cd "$ROOT_DIR"
      if [[ -z "$PRIVATE_EVIDENCE_ROOT" ]]; then
        echo "ERROR: VMLX_R20_PRIVATE_EVIDENCE_ROOT must identify the external private evidence root" >&2
        exit 1
      fi
      if [[ -z "${VMLX_JANG_TOOLS_SOURCE:-}" ]]; then
        echo "ERROR: VMLX_JANG_TOOLS_SOURCE must identify the exact clean JANG release source" >&2
        exit 1
      fi
      for bypass in \
        VMLX_ALLOW_DIRTY_SOURCE VMLINUX_ALLOW_DIRTY_SOURCE \
        VMLX_ALLOW_DIRTY_JANG_SOURCE VMLINUX_ALLOW_DIRTY_JANG_SOURCE \
        VMLX_ALLOW_UNVERSIONED_JANG_SOURCE VMLINUX_ALLOW_UNVERSIONED_JANG_SOURCE \
        VMLX_ALLOW_PYPI_JANG VMLINUX_ALLOW_PYPI_JANG \
        VMLX_ALLOW_MISSING_JANG_SOURCE_HASH VMLINUX_ALLOW_MISSING_JANG_SOURCE_HASH; do
        if [[ "${!bypass:-0}" == "1" ]]; then
          echo "ERROR: $bypass=1 is forbidden for the .19 production release" >&2
          exit 1
        fi
      done
      if [[ -n "${VMLX_R20_RELEASE_ATTESTATION:-}" ]]; then
        echo "ERROR: obsolete VMLX_R20_RELEASE_ATTESTATION is forbidden for the SSD-only release path" >&2
        exit 1
      fi
      scoped_prepackage_args=()
      if [[ -n "${VMLX_SCOPED_PREPACKAGE_RECEIPT:-}" ]]; then
        scoped_prepackage_args=(--scoped-prepackage-receipt "$VMLX_SCOPED_PREPACKAGE_RECEIPT")
      fi
      run_release_python "tests/cross_matrix/run_release_regression_manifest.py" \
        --scope "$RELEASE_SCOPE" \
        --require-prepackage-ready \
        --require-production-provenance \
        --expected-version "$VERSION" \
        --jang-source "$VMLX_JANG_TOOLS_SOURCE" \
        "${scoped_prepackage_args[@]}" \
        --out "$PREPACKAGE_READY_MANIFEST_OUT"
    )
    ;;
  "")
    (
      cd "$ROOT_DIR"
      run_release_python "tests/cross_matrix/run_release_regression_manifest.py" \
        --require-prepackage-ready \
        --out "$PREPACKAGE_READY_MANIFEST_OUT"
    )
    ;;
  *)
    echo "ERROR: unsupported release scope: $RELEASE_SCOPE" >&2
    echo "Set VMLX_RELEASE_SCOPE=r20_production for the production checkpoint," >&2
    echo "or VMLX_RELEASE_SCOPE=r17_consolidation for the 1.6.17 usable checkpoint," >&2
    echo "or VMLX_RELEASE_SCOPE=r16_parser_cache for the 1.6.16 emergency parser/cache scope," >&2
    echo "or VMLX_RELEASE_SCOPE=mm3_gemma_vl (or VMLINUX_RELEASE_SCOPE=mm3_gemma_vl)," >&2
    echo "or VMLX_RELEASE_SCOPE=codex_ui_only for Codex-driven UI validation flow." >&2
    echo "Supported scoped release values: r20_production, r17_consolidation, r16_parser_cache, mm3_gemma_vl, codex_ui_only" >&2
    exit 2
    ;;
esac

R20_EXPECTED_VMLX_COMMIT=""
R20_EXPECTED_VMLX_TREE=""
R20_EXPECTED_VMLX_UPSTREAM_COMMIT=""
R20_EXPECTED_VMLX_REMOTE_MAIN_COMMIT=""
R20_EXPECTED_VMLX_REMOTE_IDENTITY=""
R20_EXPECTED_JANG_COMMIT=""
R20_EXPECTED_JANG_TREE=""
R20_EXPECTED_JANG_UPSTREAM_COMMIT=""
R20_EXPECTED_JANG_REMOTE_MAIN_COMMIT=""
R20_EXPECTED_JANG_REMOTE_IDENTITY=""
R20_PREFLIGHT_MANIFEST_SHA256=""
R20_BUILD_DRIVER_NONCE=""
R20_BUILD_ATTESTATION_OUT=""
R20_ATTESTATION_EXTRACT_ROOT=""
R20_HOOK_ATTESTATION_DIR=""
R20_CURRENT_BUNDLE_RUNTIME_PATH=""
R20_CURRENT_BUNDLE_RUNTIME_SHA256=""
R20_CURRENT_MLX_WHEEL_PLATFORM=""
R20_CURRENT_MINIMUM_SYSTEM_VERSION=""

read_r20_manifest_value() {
  local dotted_path="$1"
  run_release_python - "$PREPACKAGE_READY_MANIFEST_OUT" "$dotted_path" <<'PY'
import json
import sys
from pathlib import Path

manifest_path = Path(sys.argv[1])
dotted_path = sys.argv[2]
manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
if manifest.get("status") != "pass":
    raise SystemExit("ERROR: vMLX preflight manifest status is not pass")
value = manifest
for component in dotted_path.split("."):
    if not isinstance(value, dict) or component not in value:
        raise SystemExit(
            f"ERROR: vMLX preflight manifest is missing {dotted_path}"
        )
    value = value[component]
if not isinstance(value, str) or not value.strip():
    raise SystemExit(
        f"ERROR: vMLX preflight manifest has invalid {dotted_path}"
    )
print(value)
PY
}

canonical_origin_identity() {
  local repo="$1"
  local remote_url
  remote_url="$(capture_toolchain_action git -C "$repo" remote get-url origin)"
  run_release_python - "$remote_url" <<'PY'
import re
import sys
from urllib.parse import urlsplit

normalized = sys.argv[1].strip()
match = re.fullmatch(
    r"git@github\.com:([^/:\s]+/[^/:\s]+?)(?:\.git)?",
    normalized,
    re.IGNORECASE,
)
if match is not None:
    print(match.group(1).lower())
    raise SystemExit(0)
try:
    parsed = urlsplit(normalized)
except ValueError as exc:
    raise SystemExit(
        "ERROR: release origin is not a canonical GitHub repository"
    ) from exc
path = re.sub(r"\.git$", "", parsed.path.strip("/"), flags=re.IGNORECASE)
scheme = parsed.scheme.lower()
https_ok = (
    scheme == "https"
    and parsed.username is None
    and parsed.password is None
    and parsed.port is None
)
ssh_ok = (
    scheme == "ssh"
    and parsed.username == "git"
    and parsed.password is None
    and parsed.port is None
)
if (
    not (https_ok or ssh_ok)
    or (parsed.hostname or "").lower() != "github.com"
    or parsed.query
    or parsed.fragment
    or re.fullmatch(r"[^/:\s]+/[^/:\s]+", path) is None
):
    raise SystemExit("ERROR: release origin is not a canonical GitHub repository")
print(path.lower())
PY
}

live_origin_main_commit() {
  local repo="$1"
  local remote_main
  local remote_commit
  remote_main="$(
    capture_toolchain_action git -C "$repo" ls-remote --exit-code origin refs/heads/main
  )"
  IFS=$' \t' read -r remote_commit _ <<<"$remote_main"
  if [[ ! "$remote_commit" =~ ^[0-9a-f]{40}$ ]]; then
    echo "ERROR: pinned git returned an invalid origin/main commit" >&2
    exit 1
  fi
  printf '%s\n' "$remote_commit"
}

if [[ "$RELEASE_SCOPE" == "r20_production" ]]; then
  R20_PREFLIGHT_MANIFEST_SHA256="$(toolchain_sha256 "$PREPACKAGE_READY_MANIFEST_OUT")"
  R20_EXPECTED_VMLX_COMMIT="$(read_r20_manifest_value source.commit)"
  R20_EXPECTED_VMLX_TREE="$(read_r20_manifest_value source.tree)"
  R20_EXPECTED_VMLX_UPSTREAM_COMMIT="$(
    read_r20_manifest_value source.upstream_commit
  )"
  R20_EXPECTED_VMLX_REMOTE_MAIN_COMMIT="$(
    read_r20_manifest_value source.remote_main_commit
  )"
  R20_EXPECTED_VMLX_REMOTE_IDENTITY="$(
    read_r20_manifest_value source.remote_identity
  )"
  R20_EXPECTED_JANG_COMMIT="$(read_r20_manifest_value jang.commit)"
  R20_EXPECTED_JANG_TREE="$(read_r20_manifest_value jang.tree)"
  R20_EXPECTED_JANG_UPSTREAM_COMMIT="$(
    read_r20_manifest_value jang.upstream_commit
  )"
  R20_EXPECTED_JANG_REMOTE_MAIN_COMMIT="$(
    read_r20_manifest_value jang.remote_main_commit
  )"
  R20_EXPECTED_JANG_REMOTE_IDENTITY="$(
    read_r20_manifest_value jang.remote_identity
  )"
  if [[ "$(toolchain_sha256 "$PREPACKAGE_READY_MANIFEST_OUT")" != "$R20_PREFLIGHT_MANIFEST_SHA256" ]]; then
    echo "ERROR: vMLX preflight manifest changed while its identity was read" >&2
    exit 1
  fi
  if [[ "$R20_EXPECTED_VMLX_REMOTE_IDENTITY" != "jjang-ai/vmlx" ]] \
    || [[ "$R20_EXPECTED_JANG_REMOTE_IDENTITY" != "jjang-ai/jangq" ]]; then
    echo "ERROR: vMLX preflight manifest does not attest canonical release repositories" >&2
    exit 1
  fi
  if [[ -z "$R20_PRE_NOTARY_MANIFEST_OUT" ]]; then
    R20_PRE_NOTARY_MANIFEST_OUT="$PRIVATE_EVIDENCE_ROOT/artifact-handoffs/vMLX-${VERSION}-${R20_EXPECTED_VMLX_COMMIT:0:12}-pre-notary.json"
  fi
  R20_BUILD_DRIVER_NONCE="$(
    run_release_python -I -c 'import secrets; print(secrets.token_hex(32))'
  )"
  if [[ ! "$R20_BUILD_DRIVER_NONCE" =~ ^[0-9a-f]{64}$ ]]; then
    echo "ERROR: could not create unpredictable release build-driver nonce" >&2
    exit 1
  fi
  R20_BUILD_ATTESTATION_OUT="$PRIVATE_EVIDENCE_ROOT/artifact-handoffs/vMLX-${VERSION}-${R20_EXPECTED_VMLX_COMMIT:0:12}-build-driver.json"
  R20_HOOK_ATTESTATION_DIR="$PRIVATE_EVIDENCE_ROOT/hook-completions/vMLX-${VERSION}-${R20_EXPECTED_VMLX_COMMIT:0:12}"
  if [[ -e "$R20_HOOK_ATTESTATION_DIR" || -L "$R20_HOOK_ATTESTATION_DIR" ]]; then
    echo "ERROR: refusing reused r20 hook-completion attestation directory" >&2
    exit 1
  fi
  mkdir -p "$R20_HOOK_ATTESTATION_DIR"
  chmod 0700 "$R20_HOOK_ATTESTATION_DIR"
  export VMLX_R20_OFFICIAL_PACKAGING="1"
  export VMLX_R20_EXPECTED_TEAM_ID="$EXPECTED_APPLE_TEAM_ID"
  export VMLX_R20_EXPECTED_CODESIGN_IDENTITY="$EXPECTED_CODESIGN_IDENTITY"
  export VMLX_R20_PREPACKAGE_MANIFEST="$PREPACKAGE_READY_MANIFEST_OUT"
  export VMLX_R20_PREPACKAGE_MANIFEST_SHA256="$R20_PREFLIGHT_MANIFEST_SHA256"
  export VMLX_R20_RELEASE_DRIVER_PID="$$"
  export VMLX_R20_RELEASE_DRIVER_NONCE="$R20_BUILD_DRIVER_NONCE"
  export VMLX_R20_RELEASE_REQUESTED_FLAVOR="$REQUESTED_FLAVOR"
  export VMLX_R20_HOOK_ATTESTATION_DIR="$R20_HOOK_ATTESTATION_DIR"
fi

write_r20_build_plan() {
  local flavor="$1"
  local phase="$2"
  local expected_artifact="$3"
  local plan_hash
  local staged_app="$DIST_DIR/${flavor}-app/mac-arm64/vMLX.app"
  local hook_attestation="$R20_HOOK_ATTESTATION_DIR/${flavor}.completion.json"

  if [[ "$RELEASE_SCOPE" != "r20_production" ]]; then
    return 0
  fi
  if [[ "$R20_CURRENT_BUNDLE_RUNTIME_PATH" != "$R20_HOOK_ATTESTATION_DIR/${flavor}.bundle-runtime.json" ]] \
    || [[ ! "$R20_CURRENT_BUNDLE_RUNTIME_SHA256" =~ ^[0-9a-f]{64}$ ]]; then
    echo "ERROR: ${flavor} build plan is missing sealed bundle runtime provenance" >&2
    exit 1
  fi

  mkdir -p "$(dirname "$R20_BUILD_PLAN_PATH")"
  rm -f "$R20_BUILD_PLAN_PATH"
  plan_hash="$(
    run_release_python -I - \
      "$R20_BUILD_PLAN_PATH" \
      "$VERSION" \
      "$R20_EXPECTED_VMLX_COMMIT" \
      "$R20_EXPECTED_VMLX_TREE" \
      "$R20_PREFLIGHT_MANIFEST_SHA256" \
      "$REQUESTED_FLAVOR" \
      "$flavor" \
      "$phase" \
      "$expected_artifact" \
      "$staged_app" \
      "$hook_attestation" \
      "$R20_CURRENT_BUNDLE_RUNTIME_PATH" \
      "$R20_CURRENT_BUNDLE_RUNTIME_SHA256" \
      "$R20_CURRENT_MLX_WHEEL_PLATFORM" \
      "$R20_CURRENT_MINIMUM_SYSTEM_VERSION" \
      "$R20_BUILD_DRIVER_NONCE" \
      "$$" <<'PY'
import hashlib
import json
import os
import sys
from pathlib import Path

(
    output_raw,
    version,
    source_commit,
    source_tree,
    manifest_sha256,
    requested_flavor,
    current_flavor,
    phase,
    expected_artifact_raw,
    staged_app_raw,
    hook_attestation_raw,
    bundle_runtime_raw,
    bundle_runtime_sha256,
    mlx_wheel_platform,
    minimum_system_version,
    driver_nonce,
    driver_pid_raw,
) = sys.argv[1:]
output = Path(output_raw).absolute()
expected_artifact = Path(expected_artifact_raw).absolute()
staged_app = Path(staged_app_raw).absolute()
hook_attestation = Path(hook_attestation_raw).absolute()
bundle_runtime = Path(bundle_runtime_raw).absolute()
tool_names = (
    "GIT", "NODE", "NPM", "NPX", "SHASUM", "AWK", "FILE", "FIND",
    "ASAR", "APP_BUILDER", "ELECTRON_BUILDER",
)
tools = {
    name.lower(): {
        "path": os.environ[f"VMLX_R20_TOOL_{name}_PATH"],
        "realpath": os.environ[f"VMLX_R20_TOOL_{name}_REALPATH"],
        "sha256": os.environ[f"VMLX_R20_TOOL_{name}_SHA256"],
    }
    for name in tool_names
}
payload = {
    "schema_version": 3,
    "scope": "r20_production",
    "version": version,
    "source_commit": source_commit,
    "source_tree": source_tree,
    "manifest_sha256": manifest_sha256,
    "requested_flavor": requested_flavor,
    "current_flavor": current_flavor,
    "phase": phase,
    "expected_artifact": str(expected_artifact),
    "staged_app": str(staged_app),
    "hook_attestation": str(hook_attestation),
    "bundle_runtime": {
        "path": str(bundle_runtime),
        "sha256": bundle_runtime_sha256,
    },
    "flavor_contract": {
        "mlx_wheel_platform": mlx_wheel_platform,
        "minimum_system_version": minimum_system_version,
    },
    "driver_pid": int(driver_pid_raw),
    "nonce": driver_nonce,
    "fixed_path": os.environ["VMLX_R20_FIXED_PATH"],
    "tools": tools,
}
encoded = (json.dumps(payload, sort_keys=True, separators=(",", ":")) + "\n").encode()
temporary = output.with_name(f".{output.name}.{os.getpid()}.tmp")
fd = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
try:
    with os.fdopen(fd, "wb") as handle:
        handle.write(encoded)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, output)
    os.chmod(output, 0o400)
finally:
    try:
        temporary.unlink()
    except FileNotFoundError:
        pass
print(hashlib.sha256(encoded).hexdigest())
PY
  )"
  if [[ ! "$plan_hash" =~ ^[0-9a-f]{64}$ ]]; then
    echo "ERROR: vMLX release driver plan hash is invalid" >&2
    exit 1
  fi
  export VMLX_R20_RELEASE_PLAN="$R20_BUILD_PLAN_PATH"
  export VMLX_R20_RELEASE_PLAN_SHA256="$plan_hash"
  export VMLX_R20_RELEASE_CURRENT_FLAVOR="$flavor"
  export VMLX_R20_RELEASE_PHASE="$phase"
  export VMLX_EXPECTED_MLX_WHEEL_PLATFORM="$R20_CURRENT_MLX_WHEEL_PLATFORM"
  export VMLX_R20_RELEASE_EXPECTED_ARTIFACT="$(
    run_release_python -I -c \
      'import pathlib, sys; print(pathlib.Path(sys.argv[1]).absolute())' \
      "$expected_artifact"
  )"
}

assert_r20_source_identity() {
  local phase="$1"
  local vmlx_remote_identity
  local vmlx_remote_main_commit
  local jang_remote_identity
  local jang_remote_main_commit
  local env_name
  if [[ "$RELEASE_SCOPE" != "r20_production" ]]; then
    return 0
  fi
  if [[ "$(toolchain_sha256 "$PREPACKAGE_READY_MANIFEST_OUT")" != "$R20_PREFLIGHT_MANIFEST_SHA256" ]]; then
    echo "ERROR: vMLX preflight manifest changed ${phase}" >&2
    exit 1
  fi
  if [[ "$(capture_toolchain_action git -C "$ROOT_DIR" rev-parse HEAD)" != "$R20_EXPECTED_VMLX_COMMIT" ]] \
    || [[ "$(capture_toolchain_action git -C "$ROOT_DIR" rev-parse HEAD^{tree})" != "$R20_EXPECTED_VMLX_TREE" ]] \
    || [[ "$(capture_toolchain_action git -C "$ROOT_DIR" rev-parse '@{upstream}')" != "$R20_EXPECTED_VMLX_UPSTREAM_COMMIT" ]] \
    || [[ -n "$(capture_toolchain_action git -C "$ROOT_DIR" status --porcelain --untracked-files=all)" ]]; then
    echo "ERROR: vMLX source identity changed ${phase}" >&2
    exit 1
  fi
  vmlx_remote_identity="$(canonical_origin_identity "$ROOT_DIR")"
  vmlx_remote_main_commit="$(live_origin_main_commit "$ROOT_DIR")"
  if [[ "$vmlx_remote_identity" != "$R20_EXPECTED_VMLX_REMOTE_IDENTITY" ]] \
    || [[ "$vmlx_remote_main_commit" != "$R20_EXPECTED_VMLX_REMOTE_MAIN_COMMIT" ]]; then
    echo "ERROR: live canonical vMLX origin/main changed ${phase}" >&2
    exit 1
  fi
  if [[ "$(capture_toolchain_action git -C "$VMLX_JANG_TOOLS_SOURCE" rev-parse HEAD)" != "$R20_EXPECTED_JANG_COMMIT" ]] \
    || [[ "$(capture_toolchain_action git -C "$VMLX_JANG_TOOLS_SOURCE" rev-parse HEAD^{tree})" != "$R20_EXPECTED_JANG_TREE" ]] \
    || [[ "$(capture_toolchain_action git -C "$VMLX_JANG_TOOLS_SOURCE" rev-parse '@{upstream}')" != "$R20_EXPECTED_JANG_UPSTREAM_COMMIT" ]] \
    || [[ -n "$(capture_toolchain_action git -C "$VMLX_JANG_TOOLS_SOURCE" status --porcelain --untracked-files=all)" ]]; then
    echo "ERROR: JANG source identity changed ${phase}" >&2
    exit 1
  fi
  jang_remote_identity="$(canonical_origin_identity "$VMLX_JANG_TOOLS_SOURCE")"
  jang_remote_main_commit="$(live_origin_main_commit "$VMLX_JANG_TOOLS_SOURCE")"
  if [[ "$jang_remote_identity" != "$R20_EXPECTED_JANG_REMOTE_IDENTITY" ]] \
    || [[ "$jang_remote_main_commit" != "$R20_EXPECTED_JANG_REMOTE_MAIN_COMMIT" ]]; then
    echo "ERROR: live canonical JANG origin/main changed ${phase}" >&2
    exit 1
  fi
  for env_file in \
    "$PANEL_DIR/.env" \
    "$PANEL_DIR/.env.local" \
    "$PANEL_DIR/.env.production" \
    "$PANEL_DIR/.env.production.local"; do
    if [[ -e "$env_file" || -L "$env_file" ]]; then
      echo "ERROR: release build input ${env_file##*/} exists ${phase}" >&2
      exit 1
    fi
  done
  while IFS= read -r env_name; do
    if [[ "$env_name" == VITE_* ]]; then
      echo "ERROR: untracked VITE_ environment overrides exist ${phase}" >&2
      exit 1
    fi
  done < <(compgen -e)
}

if [[ "$RELEASE_SCOPE" == "r20_production" ]]; then
  assert_r20_source_identity "before npm ci"
  echo "==> Reinstalling exact panel dependencies from package-lock.json"
  run_toolchain_action npm ci
  assert_r20_source_identity "after npm ci"
  # The fail-closed retained-evidence gate above requires prepackage clearance
  # plus clean, pushed, canonical vMLX and JANG origin/main provenance. Running
  # the heavy suites again here would duplicate immutable rows and can fail only
  # because a prior packaging bundle is stale. A source or remote-main change
  # invalidates the sealed manifest and is rejected before every build phase.
  echo "==> Reusing exact-head retained prepackage evidence"
  assert_r20_source_identity "after retained prepackage evidence reuse"
fi

is_macho_file() {
  local file_path="$1"
  local description
  description="$(capture_toolchain_action file "$file_path")"
  [[ "$description" == *"Mach-O"* ]]
}

sign_bundled_python_native_files() {
  local bundled_python="$1"
  local identity="$2"

  if [[ ! -d "$bundled_python" ]]; then
    echo "ERROR: missing bundled Python at $bundled_python" >&2
    exit 1
  fi

  echo "==> Signing bundled Python native files with release identity"
  local signed_count=0
  while IFS= read -r native_file; do
    if is_macho_file "$native_file"; then
      "$APPLE_CODESIGN" --force --timestamp --options runtime --sign "$identity" "$native_file" >/dev/null
      signed_count=$((signed_count + 1))
    fi
  done < <(capture_toolchain_action find "$bundled_python" -type f \( -name "*.dylib" -o -name "*.so" -o -perm +111 \))
  echo "  signed $signed_count bundled Python native files"
}

sign_remaining_app_macho_leaves() {
  local app_path="$1"
  local identity="$2"
  local bundled_python="$app_path/Contents/Resources/bundled-python"
  local signed_count=0
  local signature

  echo "==> Signing remaining ad-hoc or unsigned app Mach-O leaves"
  while IFS= read -r native_file; do
    if ! is_macho_file "$native_file"; then
      continue
    fi

    signature=""
    if ! signature="$("$APPLE_CODESIGN" -dv --verbose=4 "$native_file" 2>&1)"; then
      :
    elif ! grep -Eq "Signature=adhoc|flags=.*adhoc|TeamIdentifier=not set" <<< "$signature"; then
      continue
    fi

    "$APPLE_CODESIGN" --force --timestamp --options runtime --sign "$identity" "$native_file" >/dev/null
    signed_count=$((signed_count + 1))
  done < <(
    capture_toolchain_action find "$app_path/Contents" \
      -path "$bundled_python" -prune -o \
      -type f -print
  )
  echo "  signed $signed_count remaining app Mach-O leaves"
}

verify_release_macho_leaves() {
  local app_path="$1"
  local failed=0
  local checked_count=0
  local native_file
  local signature

  echo "==> Verifying every app Mach-O leaf has Developer ID, timestamp, and hardened runtime"
  while IFS= read -r native_file; do
    checked_count=$((checked_count + 1))
    signature="$("$APPLE_CODESIGN" -dv --verbose=4 "$native_file" 2>&1 || true)"
    # Here-strings, not printf pipes: grep -q exits on its first match, printf
    # then fails with EPIPE, and under pipefail a correct signature became a
    # release failure (measured 2026-09-05: six correctly signed leaves).
    if ! grep -Fqx "Authority=$EXPECTED_CODESIGN_IDENTITY" <<< "$signature" ||
      ! grep -Fqx "TeamIdentifier=$EXPECTED_APPLE_TEAM_ID" <<< "$signature" ||
      ! grep -q "^Timestamp=" <<< "$signature" ||
      ! grep -Eq "^CodeDirectory .*flags=.*runtime" <<< "$signature"; then
      echo "ERROR: release Mach-O leaf is not signed by the exact configured Apple team: $native_file" >&2
      printf '%s\n' "$signature" >&2
      failed=1
    fi
  # Classify the complete tree in one pinned-Python pass. Spawning the sealed
  # action wrapper once per file makes this audit take hours for bundled Python.
  # Magic inspection keeps full coverage without relying on filename suffixes.
  done < <(run_release_python -I - "$app_path/Contents" <<'PY'
import os
import sys

MACHO_MAGICS = {
    bytes.fromhex(value)
    for value in (
        "feedface",
        "cefaedfe",
        "feedfacf",
        "cffaedfe",
        "cafebabe",
        "bebafeca",
        "cafebabf",
        "bfbafeca",
    )
}

root = os.path.realpath(sys.argv[1])
for directory, _, filenames in os.walk(root, followlinks=False):
    for filename in filenames:
        path = os.path.join(directory, filename)
        if os.path.islink(path) or not os.path.isfile(path):
            continue
        try:
            with open(path, "rb") as handle:
                magic = handle.read(4)
        except OSError:
            continue
        if magic in MACHO_MAGICS:
            print(path)
PY
  )

  if [[ "$failed" -ne 0 ]]; then
    exit 1
  fi
  echo "  verified $checked_count app Mach-O leaves"
}

verify_release_signature_identity() {
  local target="$1"
  local signature

  signature="$("$APPLE_CODESIGN" -dv --verbose=4 "$target" 2>&1 || true)"
  if ! printf '%s\n' "$signature" | grep -Fqx "Authority=$EXPECTED_CODESIGN_IDENTITY" ||
    ! printf '%s\n' "$signature" | grep -Fqx "TeamIdentifier=$EXPECTED_APPLE_TEAM_ID" ||
    ! printf '%s\n' "$signature" | grep -q "^Timestamp="; then
    echo "ERROR: release artifact is not signed by the exact configured Apple team: $target" >&2
    printf '%s\n' "$signature" >&2
    exit 1
  fi
}

finalize_release_app_signature() {
  local app_path="$1"
  local identity="${2:-$RELEASE_CODESIGN_IDENTITY}"
  local entitlements="$PANEL_DIR/build/entitlements.mac.plist"

  assert_r20_release_output_safe
  if [[ ! -d "$app_path" ]]; then
    echo "ERROR: missing staged app at $app_path" >&2
    exit 1
  fi
  if [[ ! -f "$entitlements" ]]; then
    echo "ERROR: missing release entitlements at $entitlements" >&2
    exit 1
  fi

  local bundled_python="$app_path/Contents/Resources/bundled-python"
  if [[ -d "$bundled_python" ]]; then
    echo "==> Removing Python bytecode before release app seal"
    run_toolchain_action find "$bundled_python" -name "*.pyc" -type f -delete
    run_toolchain_action find "$bundled_python" -name "__pycache__" -type d -prune -exec rm -rf {} +
  fi

  sign_bundled_python_native_files "$bundled_python" "$identity"
  sign_remaining_app_macho_leaves "$app_path" "$identity"
  echo "==> Final release app seal/signature: $app_path"
  "$APPLE_CODESIGN" --force --deep --timestamp --options runtime --entitlements "$entitlements" --sign "$identity" "$app_path"
  "$APPLE_CODESIGN" --verify --deep --strict --verbose=2 "$app_path"
  verify_release_signature_identity "$app_path"
  verify_release_macho_leaves "$app_path"
}

artifact_chain_build() {
  run_release_python "$ARTIFACT_CHAIN_HELPER" artifact-chain "$@"
}

artifact_json_field() {
  local payload="$1"
  local field="$2"
  run_release_python -I -c \
    'import json,sys; value=json.loads(sys.argv[1]); print(value[sys.argv[2]])' \
    "$payload" "$field"
}

find_staged_app() {
  local staged_output="$1"
  local result
  result="$(artifact_chain_build find-staged-app --staged-output "$staged_output")"
  artifact_json_field "$result" app
}

seal_current_bundle_runtime() {
  local flavor="$1"
  local wheel_platform="$2"
  local minimum_system_version="$3"
  local output="$R20_HOOK_ATTESTATION_DIR/${flavor}.bundle-runtime.json"
  local result

  if [[ "$RELEASE_SCOPE" != "r20_production" ]]; then
    return 0
  fi
  result="$(
    artifact_chain_build write-bundle-runtime \
      --root "$ROOT_DIR" \
      --bundle-root "$PANEL_DIR/bundled-python" \
      --version "$VERSION" \
      --private-root "$PRIVATE_EVIDENCE_ROOT" \
      --flavor "$flavor" \
      --out "$output"
  )"
  R20_CURRENT_BUNDLE_RUNTIME_PATH="$output"
  R20_CURRENT_BUNDLE_RUNTIME_SHA256="$(
    artifact_json_field "$result" sha256
  )"
  R20_CURRENT_MLX_WHEEL_PLATFORM="$wheel_platform"
  R20_CURRENT_MINIMUM_SYSTEM_VERSION="$minimum_system_version"
  if [[ ! "$R20_CURRENT_BUNDLE_RUNTIME_SHA256" =~ ^[0-9a-f]{64}$ ]]; then
    echo "ERROR: ${flavor} bundle runtime attestation digest is invalid" >&2
    exit 1
  fi
}

verify_staged_app_parity() {
  local flavor="$1"
  local staged_output="$2"
  local app_path

  if [[ "$RELEASE_SCOPE" != "r20_production" ]]; then
    return 0
  fi
  app_path="$(find_staged_app "$staged_output")"
  if [[ ! -f "$app_path/Contents/Resources/app.asar" ]]; then
    echo "ERROR: staged ${flavor} app is missing app.asar" >&2
    exit 1
  fi
  (
    set -euo pipefail
    local extracted
    extracted="$(mktemp -d "$PRIVATE_EVIDENCE_ROOT/.staged-${flavor}-asar.XXXXXX")"
    trap 'remove_owned_release_tree_with_retry "$extracted"' EXIT
    assert_asar_excludes_finder_metadata \
      "$app_path/Contents/Resources/app.asar"
    run_driver_plan_action asar extract \
      "$app_path/Contents/Resources/app.asar" \
      "$extracted"
    artifact_chain_build check-staged-app \
      --root "$ROOT_DIR" \
      --staged-output "$staged_output" \
      --extracted-asar "$extracted" \
      --version "$VERSION" \
      --flavor "$flavor"
  )
}

write_mounted_dmg_payload_parity() (
  set -euo pipefail
  local flavor="$1"
  local dmg="$DIST_DIR/vMLX-${VERSION}-${flavor}-arm64.dmg"
  local hook_attestation="$R20_HOOK_ATTESTATION_DIR/${flavor}.completion.json"
  local parity_attestation="$R20_HOOK_ATTESTATION_DIR/${flavor}.dmg-parity.json"
  local hook_record
  local dmg_record
  local operation_root
  local snapshot_dmg
  local mount_dir
  local extracted_asar
  local attached=0

  hook_record="$(artifact_chain_build file-record \
    --path "$hook_attestation" \
    --label "${flavor} hook completion attestation")"
  dmg_record="$(artifact_chain_build file-record \
    --path "$dmg" \
    --label "${flavor} generated DMG")"
  operation_root="$(mktemp -d "$PRIVATE_EVIDENCE_ROOT/.pre-notary-${flavor}.XXXXXX")"
  chmod 0700 "$operation_root"
  snapshot_dmg="$operation_root/$(basename "$dmg")"
  mount_dir="$operation_root/mount"
  extracted_asar="$operation_root/extracted-asar"
  mkdir -p "$mount_dir" "$extracted_asar"

  cleanup_pre_notary_mount() {
    if [[ "$attached" == "1" ]]; then
      "$APPLE_HDIUTIL" detach "$mount_dir" >/dev/null 2>&1 \
        || "$APPLE_HDIUTIL" detach -force "$mount_dir" >/dev/null 2>&1 \
        || true
    fi
    remove_owned_release_tree_with_retry "$operation_root"
  }
  trap cleanup_pre_notary_mount EXIT

  artifact_chain_build copy-operation-file \
    --private-root "$PRIVATE_EVIDENCE_ROOT" \
    --source "$dmg" \
    --out "$snapshot_dmg" \
    --expected-sha256 "$(artifact_json_field "$dmg_record" sha256)" \
    --label "pre-notary ${flavor} DMG snapshot" >/dev/null
  "$APPLE_HDIUTIL" attach \
    -readonly \
    -nobrowse \
    -noautoopen \
    -mountpoint "$mount_dir" \
    "$snapshot_dmg" >/dev/null
  attached=1
  if [[ ! -d "$mount_dir/vMLX.app" || -L "$mount_dir/vMLX.app" ]]; then
    echo "ERROR: mounted ${flavor} DMG does not contain the exact vMLX.app" >&2
    exit 1
  fi
  assert_asar_excludes_finder_metadata \
    "$mount_dir/vMLX.app/Contents/Resources/app.asar"
  run_driver_plan_action asar extract \
    "$mount_dir/vMLX.app/Contents/Resources/app.asar" \
    "$extracted_asar"
  artifact_chain_build write-dmg-payload-parity \
    --root "$ROOT_DIR" \
    --dist "$DIST_DIR" \
    --version "$VERSION" \
    --private-root "$PRIVATE_EVIDENCE_ROOT" \
    --flavor "$flavor" \
    --hook-attestation "$hook_attestation" \
    --expected-hook-sha256 "$(artifact_json_field "$hook_record" sha256)" \
    --expected-nonce "$R20_BUILD_DRIVER_NONCE" \
    --expected-driver-pid "$$" \
    --mounted-app "$mount_dir/vMLX.app" \
    --extracted-asar "$extracted_asar" \
    --out "$parity_attestation" >/dev/null
  "$APPLE_HDIUTIL" detach "$mount_dir" >/dev/null
  attached=0
  trap - EXIT
  remove_owned_release_tree_with_retry "$operation_root"
)

build_one() {
  local flavor="$1"
  local platform="$2"
  local wheel_tag
  local minimum_system_version
  local staged_output="$DIST_DIR/${flavor}-app"
  local app_path
  local dmg_path="$DIST_DIR/vMLX-${VERSION}-${flavor}-arm64.dmg"
  local identity_args=()

  if [[ "$DEV_UNSIGNED" == "1" ]]; then
    identity_args=(--config.mac.identity=null)
  fi

  case "$flavor" in
    sequoia)
      wheel_tag="macosx_14_0_arm64"
      ;;
    tahoe)
      wheel_tag="macosx_26_0_arm64"
      ;;
    *)
      echo "ERROR: unsupported release flavor: $flavor" >&2
      exit 1
      ;;
  esac

  minimum_system_version="$(python3 ./scripts/mlx-runtime.py minimum-os --platform "$wheel_tag")"

  assert_r20_source_identity "before ${flavor} bundle"
  echo "==> Building vMLX ${VERSION} ${flavor} DMG (${wheel_tag})"
  VMLX_BUNDLE_MLX_PLATFORM="$platform" ./scripts/bundle-python.sh
  VMLX_EXPECTED_MLX_WHEEL_PLATFORM="$wheel_tag" \
    ./scripts/verify-bundled-python.sh
  assert_r20_source_identity "after ${flavor} Python bundle verification"
  seal_current_bundle_runtime \
    "$flavor" "$wheel_tag" "$minimum_system_version"
  assert_r20_release_output_safe
  remove_owned_release_tree_with_retry "$staged_output"
  # Let electron-builder perform its proven inside-out Developer-ID signing of
  # Electron and Squirrel framework leaves. Its mandatory beforePack hook owns
  # the single renderer build for this flavor, so the driver must not build the
  # renderer a second time. The controlled finalizer below then re-signs bundled
  # Python, repairs any remaining ad-hoc Mach-O leaves, audits every leaf, and
  # applies the final outer app seal.
  write_r20_build_plan "$flavor" "stage" "$staged_output"
  run_electron_builder_action --mac --dir \
    --config.directories.output="$staged_output" \
    --config.mac.minimumSystemVersion="$minimum_system_version" \
    ${identity_args[@]+"${identity_args[@]}"}
  assert_r20_source_identity "after ${flavor} app staging"
  app_path="$(find_staged_app "$staged_output")"
  if [[ "$DEV_UNSIGNED" == "1" ]]; then
    echo "==> Unsigned development build: skipping Developer ID finalization"
  else
    finalize_release_app_signature "$app_path" "$RELEASE_CODESIGN_IDENTITY"
    verify_staged_app_parity "$flavor" "$staged_output"
  fi
  assert_r20_source_identity "after ${flavor} staged app parity"
  write_r20_build_plan "$flavor" "dmg" "$dmg_path"
  run_electron_builder_action --mac dmg \
    --prepackaged "$app_path" \
    --config.directories.output="$DIST_DIR" \
    --config.mac.minimumSystemVersion="$minimum_system_version" \
    --config.mac.artifactName="vMLX-\${version}-${flavor}-\${arch}.\${ext}" \
    ${identity_args[@]+"${identity_args[@]}"}
  if [[ ! -s "$dmg_path" ]]; then
    echo "ERROR: missing expected ${flavor} release DMG: $dmg_path" >&2
    exit 1
  fi
  if [[ "$DEV_UNSIGNED" != "1" ]]; then
    verify_release_signature_identity "$dmg_path"
  fi
  assert_r20_source_identity "after ${flavor} DMG"
}

case "$REQUESTED_FLAVOR" in
  all)
    assert_r20_release_output_safe
    remove_owned_release_tree_with_retry "$DIST_DIR"
    build_one "sequoia" "compat"
    build_one "tahoe" "native"
    ;;
  sequoia)
    build_one "sequoia" "compat"
    ;;
  tahoe)
    build_one "tahoe" "native"
    ;;
  *)
    echo "Usage: $0 [all|sequoia|tahoe]" >&2
    exit 2
    ;;
esac

if [[ "$RELEASE_SCOPE" == "r20_production" ]]; then
  expected_sequoia="$DIST_DIR/vMLX-${VERSION}-sequoia-arm64.dmg"
  expected_tahoe="$DIST_DIR/vMLX-${VERSION}-tahoe-arm64.dmg"
  if [[ ! -s "$expected_sequoia" || ! -s "$expected_tahoe" ]]; then
    echo "ERROR: vMLX production packaging did not produce both required DMGs" >&2
    exit 1
  fi
  run_driver_plan_action node - "$DIST_DIR" "$expected_sequoia" "$expected_tahoe" <<'NODE'
const {
  verifyExactDmgDirectory,
} = require("./scripts/electron-builder-before-pack.cjs");

const [distDirRaw, ...expected] = process.argv.slice(2);
verifyExactDmgDirectory(
  distDirRaw,
  expected,
  "vMLX production packaging produced an unexpected DMG set",
);
NODE
  # Re-run exact-one staged-app and full source/runtime/renderer parity after
  # both flavors exist so the private handoff cannot inherit an earlier stage.
  verify_staged_app_parity "sequoia" "$DIST_DIR/sequoia-app"
  verify_staged_app_parity "tahoe" "$DIST_DIR/tahoe-app"
  echo "==> Proving each exact DMG matches its hook-completed staged payload"
  write_mounted_dmg_payload_parity "sequoia"
  write_mounted_dmg_payload_parity "tahoe"
  sequoia_hook_sha256="$(
    artifact_json_field "$(
      artifact_chain_build file-record \
        --path "$R20_HOOK_ATTESTATION_DIR/sequoia.completion.json" \
        --label "sequoia hook completion"
    )" sha256
  )"
  tahoe_hook_sha256="$(
    artifact_json_field "$(
      artifact_chain_build file-record \
        --path "$R20_HOOK_ATTESTATION_DIR/tahoe.completion.json" \
        --label "tahoe hook completion"
    )" sha256
  )"
  sequoia_parity_sha256="$(
    artifact_json_field "$(
      artifact_chain_build file-record \
        --path "$R20_HOOK_ATTESTATION_DIR/sequoia.dmg-parity.json" \
        --label "sequoia DMG parity"
    )" sha256
  )"
  tahoe_parity_sha256="$(
    artifact_json_field "$(
      artifact_chain_build file-record \
        --path "$R20_HOOK_ATTESTATION_DIR/tahoe.dmg-parity.json" \
        --label "tahoe DMG parity"
    )" sha256
  )"
  assert_r20_source_identity "before pre-notary artifact manifest"
  R20_ATTESTATION_EXTRACT_ROOT="$(
    mktemp -d "$PRIVATE_EVIDENCE_ROOT/.build-attestation-asar.XXXXXX"
  )"
  chmod 0700 "$R20_ATTESTATION_EXTRACT_ROOT"
  sequoia_app="$(find_staged_app "$DIST_DIR/sequoia-app")"
  tahoe_app="$(find_staged_app "$DIST_DIR/tahoe-app")"
  assert_asar_excludes_finder_metadata \
    "$sequoia_app/Contents/Resources/app.asar"
  assert_asar_excludes_finder_metadata \
    "$tahoe_app/Contents/Resources/app.asar"
  run_driver_plan_action asar extract \
    "$sequoia_app/Contents/Resources/app.asar" \
    "$R20_ATTESTATION_EXTRACT_ROOT/sequoia"
  run_driver_plan_action asar extract \
    "$tahoe_app/Contents/Resources/app.asar" \
    "$R20_ATTESTATION_EXTRACT_ROOT/tahoe"
  echo "==> Writing no-clobber build-driver attestation"
  build_attestation_result="$(
    artifact_chain_build write-build-attestation \
      --root "$ROOT_DIR" \
      --dist "$DIST_DIR" \
      --version "$VERSION" \
      --preflight "$PREPACKAGE_READY_MANIFEST_OUT" \
      --private-root "$PRIVATE_EVIDENCE_ROOT" \
      --out "$R20_BUILD_ATTESTATION_OUT" \
      --nonce "$R20_BUILD_DRIVER_NONCE" \
      --driver-pid "$$" \
      --sequoia-staged-output "$DIST_DIR/sequoia-app" \
      --sequoia-extracted-asar "$R20_ATTESTATION_EXTRACT_ROOT/sequoia" \
      --sequoia-hook-attestation "$R20_HOOK_ATTESTATION_DIR/sequoia.completion.json" \
      --sequoia-hook-attestation-sha256 "$sequoia_hook_sha256" \
      --sequoia-dmg-parity-attestation "$R20_HOOK_ATTESTATION_DIR/sequoia.dmg-parity.json" \
      --sequoia-dmg-parity-attestation-sha256 "$sequoia_parity_sha256" \
      --tahoe-staged-output "$DIST_DIR/tahoe-app" \
      --tahoe-extracted-asar "$R20_ATTESTATION_EXTRACT_ROOT/tahoe" \
      --tahoe-hook-attestation "$R20_HOOK_ATTESTATION_DIR/tahoe.completion.json" \
      --tahoe-hook-attestation-sha256 "$tahoe_hook_sha256" \
      --tahoe-dmg-parity-attestation "$R20_HOOK_ATTESTATION_DIR/tahoe.dmg-parity.json" \
      --tahoe-dmg-parity-attestation-sha256 "$tahoe_parity_sha256"
  )"
  build_attestation_sha256="$(
    artifact_json_field "$build_attestation_result" sha256
  )"
  echo "==> Binding exact Sequoia/Tahoe DMGs and blockmaps to the release source"
  pre_notary_result="$(
    artifact_chain_build write-pre-from-driver \
    --root "$ROOT_DIR" \
    --dist "$DIST_DIR" \
    --version "$VERSION" \
    --private-root "$PRIVATE_EVIDENCE_ROOT" \
    --build-attestation "$R20_BUILD_ATTESTATION_OUT" \
    --expected-build-attestation-sha256 "$build_attestation_sha256" \
    --expected-nonce "$R20_BUILD_DRIVER_NONCE" \
    --expected-driver-pid "$$" \
    --out "$R20_PRE_NOTARY_MANIFEST_OUT"
  )"
  remove_owned_release_tree_with_retry "$R20_ATTESTATION_EXTRACT_ROOT"
  R20_ATTESTATION_EXTRACT_ROOT=""
  assert_r20_source_identity "after pre-notary artifact manifest"
  pre_notary_sha256="$(artifact_json_field "$pre_notary_result" sha256)"
  echo "==> Private no-clobber pre-notary artifact handoff"
  printf 'VMLX_R20_PRE_NOTARY_MANIFEST=%s\n' "$R20_PRE_NOTARY_MANIFEST_OUT"
  printf 'VMLX_R20_PRE_NOTARY_MANIFEST_SHA256=%s\n' "$pre_notary_sha256"
  printf 'VMLX_R20_BUILD_ATTESTATION=%s\n' "$R20_BUILD_ATTESTATION_OUT"
  printf 'VMLX_R20_BUILD_ATTESTATION_SHA256=%s\n' "$build_attestation_sha256"
  printf 'VMLX_R20_EXPECTED_SOURCE_COMMIT=%s\n' "$R20_EXPECTED_VMLX_COMMIT"
  printf 'VMLX_R20_EXPECTED_SOURCE_TREE=%s\n' "$R20_EXPECTED_VMLX_TREE"
  printf 'VMLX_R20_EXPECTED_PREFLIGHT_SHA256=%s\n' "$R20_PREFLIGHT_MANIFEST_SHA256"
fi

echo "==> Built DMG artifacts:"
capture_toolchain_action find "$DIST_DIR" -maxdepth 1 -type f -name "vMLX-${VERSION}-*.dmg" -print
