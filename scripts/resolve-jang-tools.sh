#!/usr/bin/env bash
# Resolve the local jang-tools checkout for bundling and Makefile targets.
#
# Search order (after explicit env):
#   1. REPO_ROOT/../jang/jang-tools   (sibling clone — typical dev layout)
#   2. REPO_ROOT/../jangq/jang-tools  (alternate clone directory name)
#   3. $HOME/jang/jang-tools            (legacy default)
#
# Override: export VMLX_JANG_TOOLS_SOURCE=/path/to/jang-tools
set -euo pipefail

usage() {
  cat <<'EOF'
Usage: resolve-jang-tools.sh [--try] [REPO_ROOT]

Prints the jang-tools directory path on stdout.

  --try       exit 0 with no output when not found (for Makefile discovery)
  REPO_ROOT   vmlx repository root (default: parent of scripts/)

Override search paths:
  export VMLX_JANG_TOOLS_SOURCE=/path/to/jang-tools
EOF
}

TRY=0
if [[ "${1:-}" == "--try" ]]; then
  TRY=1
  shift
fi

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  usage
  exit 0
fi

resolve_explicit() {
  local path="$1"
  if [[ -f "$path/pyproject.toml" ]]; then
    echo "$path"
    return 0
  fi
  if [[ "$TRY" == "1" ]]; then
    return 1
  fi
  echo "ERROR: jang-tools not found at VMLX_JANG_TOOLS_SOURCE/VMLINUX_JANG_TOOLS_SOURCE: $path" >&2
  echo "       Expected pyproject.toml under that directory." >&2
  return 1
}

if [[ -n "${VMLX_JANG_TOOLS_SOURCE:-}" ]]; then
  resolve_explicit "$VMLX_JANG_TOOLS_SOURCE"
  exit $?
fi

if [[ -n "${VMLINUX_JANG_TOOLS_SOURCE:-}" ]]; then
  resolve_explicit "$VMLINUX_JANG_TOOLS_SOURCE"
  exit $?
fi

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "${1:-$SCRIPT_DIR/..}" && pwd)"

candidates=(
  "$REPO_ROOT/../jang/jang-tools"
  "$REPO_ROOT/../jangq/jang-tools"
  "$HOME/jang/jang-tools"
)

for candidate in "${candidates[@]}"; do
  if [[ -f "$candidate/pyproject.toml" ]]; then
    echo "$(cd "$candidate" && pwd)"
    exit 0
  fi
done

if [[ "$TRY" == "1" ]]; then
  exit 0
fi

cat >&2 <<EOF
ERROR: jang-tools checkout not found.

Searched:
  $REPO_ROOT/../jang/jang-tools
  $REPO_ROOT/../jangq/jang-tools
  $HOME/jang/jang-tools

Recommended layout (clone beside vmlx):
  git clone https://github.com/jjang-ai/jangq.git $REPO_ROOT/../jang
  cd $(dirname "$REPO_ROOT") && make freshen-jang

Or set an explicit path:
  export VMLX_JANG_TOOLS_SOURCE=/path/to/jang-tools
EOF
exit 1
