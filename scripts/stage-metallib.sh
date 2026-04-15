#!/bin/zsh
# Stage the metallib so it's colocated with built binaries. Runs after
# `swift build` so the SwiftPM debug/release binaries can find the
# Metal kernels at `<exec_dir>/mlx.metallib` (the first path mlx-swift's
# `load_colocated_library` tries in device.cpp). Without this the CLI
# fails every model load with "Failed to load the default metallib."
#
# Usage: scripts/stage-metallib.sh [debug|release]
set -e
CFG="${1:-debug}"
cd "$(dirname "$0")/.."
SRC=".build/arm64-apple-macosx/${CFG}/vmlx_Cmlx.bundle/default.metallib"
DST=".build/arm64-apple-macosx/${CFG}/mlx.metallib"
if [[ ! -f "$SRC" ]]; then
    echo "stage-metallib: source not found at $SRC (run 'swift build -c $CFG' first)"
    exit 1
fi
cp "$SRC" "$DST"
echo "stage-metallib: $DST ($(du -h "$DST" | cut -f1))"
