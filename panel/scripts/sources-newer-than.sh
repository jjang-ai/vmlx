#!/usr/bin/env bash
# Exit 0 if any path is newer than REF (or REF is missing). Exit 1 if REF is current.
set -euo pipefail

REF="${1:?usage: sources-newer-than.sh REF PATH [PATH ...]}"
shift

if [[ ! -f "$REF" ]]; then
  exit 0
fi

ref_mtime="$(/usr/bin/stat -f %m "$REF")"

path_is_newer() {
  local path="$1"

  if [[ ! -e "$path" ]]; then
    return 1
  fi

  if [[ -f "$path" ]]; then
    [[ "$(/usr/bin/stat -f %m "$path")" -gt "$ref_mtime" ]]
    return
  fi

  while IFS= read -r -d '' file; do
    if [[ "$(/usr/bin/stat -f %m "$file")" -gt "$ref_mtime" ]]; then
      return 0
    fi
  done < <(
    find "$path" -type f \
      -not -path '*/.*' \
      -not -path '*/__pycache__/*' \
      -not -name '*.pyc' \
      -print0 2>/dev/null
  )
  return 1
}

for path in "$@"; do
  if path_is_newer "$path"; then
    exit 0
  fi
done

exit 1
