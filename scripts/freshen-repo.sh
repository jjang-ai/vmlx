#!/usr/bin/env bash
# Fetch from origin and check out the requested ref (latest semver tag, main, or branch).
set -euo pipefail

usage() {
  cat <<'EOF'
Usage: freshen-repo.sh [--force] REPO_DIR [REF]

REF (default: latest-tag):
  latest-tag   newest origin tag matching v* (fallback: origin/main)
  main         origin/main
  BRANCH       any remote branch name (e.g. codex/mimo-v25-cache-contract)

Refuses to run with uncommitted tracked changes unless --force is passed.
EOF
}

FORCE=0
if [[ "${1:-}" == "--force" ]]; then
  FORCE=1
  shift
fi

REPO_DIR="${1:?REPO_DIR required}"
REF="${2:-latest-tag}"

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  usage
  exit 0
fi

cd "$REPO_DIR"

if ! git rev-parse --is-inside-work-tree >/dev/null 2>&1; then
  echo "ERROR: not a git repository: $REPO_DIR" >&2
  exit 1
fi

if [[ "$FORCE" != "1" ]]; then
  if ! git diff --quiet --ignore-submodules -- \
    || ! git diff --cached --quiet --ignore-submodules --; then
    echo "ERROR: $REPO_DIR has uncommitted changes." >&2
    echo "       Commit or stash first, or re-run with --force." >&2
    git status --short --untracked-files=no >&2 || true
    exit 1
  fi
fi

REMOTE="${VMLX_FRESHEN_REMOTE:-origin}"
echo "==> Fetching $REMOTE in $REPO_DIR"
git fetch --tags "$REMOTE"

resolve_ref() {
  case "$1" in
    latest-tag)
      local tag
      tag="$(git tag -l 'v*' --sort=-v:refname | head -1 || true)"
      if [[ -n "$tag" ]]; then
        echo "$tag"
      else
        echo "main"
      fi
      ;;
    main)
      echo "main"
      ;;
    *)
      echo "$1"
      ;;
  esac
}

TARGET="$(resolve_ref "$REF")"
echo "==> Checking out $TARGET"

if git show-ref --verify --quiet "refs/tags/$TARGET"; then
  git checkout --force "$TARGET"
  echo "    now at tag $TARGET ($(
    git rev-parse --short HEAD
  ))"
  exit 0
fi

if git show-ref --verify --quiet "refs/remotes/$REMOTE/$TARGET"; then
  git checkout -B "$TARGET" "$REMOTE/$TARGET"
  git pull --ff-only "$REMOTE" "$TARGET"
  echo "    now at branch $TARGET ($(
    git rev-parse --short HEAD
  ))"
  exit 0
fi

if git show-ref --verify --quiet "refs/heads/$TARGET"; then
  git checkout "$TARGET"
  git pull --ff-only "$REMOTE" "$TARGET" 2>/dev/null || true
  echo "    now at branch $TARGET ($(
    git rev-parse --short HEAD
  ))"
  exit 0
fi

echo "ERROR: ref not found on $REMOTE: $TARGET" >&2
exit 1
