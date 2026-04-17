#!/usr/bin/env bash
# audit-disk.sh — enumerate every model on disk with real weight shards.
#
# Walks the HF cache (~/.cache/huggingface/hub/models--*) plus the
# user-added dirs registered in `~/Library/Application Support/vmlx/
# models.sqlite3` and prints, for each directory that has a config.json
# AND at least one weight file:
#
#   org/name | shards=N | size=X.YGB | source
#
# Follows symlinks to resolve HF blob sizes correctly — otherwise every
# HF snapshot shows 0 bytes because the safetensors/bin files are
# relative symlinks into the blobs/ subdir.
#
# Intended use: pre-flight a test matrix run with
#     tests/e2e/audit-disk.sh > /tmp/models-today.txt
# and diff against the previous run to see what moved on disk.

set -u

audit_dir() {
    local dir="$1"
    local src="$2"
    [ -f "$dir/config.json" ] || return
    local shards=0
    local total=0
    # Weight files that count: safetensors / bin / gguf.
    for f in "$dir"/*.safetensors "$dir"/*.bin "$dir"/*.gguf; do
        [ -f "$f" ] || continue
        shards=$((shards + 1))
        local real
        real=$(readlink -f "$f")
        [ -f "$real" ] || continue
        local sz
        sz=$(stat -f "%z" "$real" 2>/dev/null || echo 0)
        total=$((total + sz))
    done
    # Also recurse one level for pipeline dirs (Flux/Z-Image): their
    # config.json sits at the top but the transformer/ vae/ etc. carry
    # the actual weights.
    for sub in "$dir"/*/; do
        [ -d "$sub" ] || continue
        for f in "$sub"*.safetensors "$sub"*.bin; do
            [ -f "$f" ] || continue
            shards=$((shards + 1))
            local real
            real=$(readlink -f "$f")
            [ -f "$real" ] || continue
            local sz
            sz=$(stat -f "%z" "$real" 2>/dev/null || echo 0)
            total=$((total + sz))
        done
    done
    [ "$shards" -eq 0 ] && return
    local gb
    gb=$(python3 -c "print(f'{$total/1024/1024/1024:.1f}')")
    printf "%-70s shards=%-3d size=%sGB  source=%s\n" \
        "${dir#$HOME/}" "$shards" "$gb" "$src"
}

# 1. HF cache — one snapshot per `models--<org>--<repo>` (prefer refs/main).
if [ -d "$HOME/.cache/huggingface/hub" ]; then
    for repo in "$HOME/.cache/huggingface/hub"/models--*/; do
        name=$(basename "$repo")
        snap_dir="$repo/snapshots"
        [ -d "$snap_dir" ] || continue
        # Pick the refs/main target if it exists; else any snapshot.
        picked=""
        if [ -f "$repo/refs/main" ]; then
            sha=$(cat "$repo/refs/main" | tr -d '\n')
            [ -d "$snap_dir/$sha" ] && picked="$snap_dir/$sha"
        fi
        [ -z "$picked" ] && picked=$(ls -d "$snap_dir"/*/ 2>/dev/null | head -1)
        [ -n "$picked" ] && audit_dir "$picked" "hfCache:$name"
    done
fi

# 2. User dirs registered in the library DB.
DB="$HOME/Library/Application Support/vmlx/models.sqlite3"
if [ -f "$DB" ] && command -v sqlite3 >/dev/null 2>&1; then
    while IFS= read -r root; do
        [ -d "$root" ] || continue
        # Walk depth 3 to catch org/name layout used by MLXModels.
        find "$root" -maxdepth 3 -type d | while read -r d; do
            [ -f "$d/config.json" ] && audit_dir "$d" "userDir:${root#$HOME/}"
        done
    done < <(sqlite3 "$DB" "SELECT url FROM user_dirs" 2>/dev/null)
fi
