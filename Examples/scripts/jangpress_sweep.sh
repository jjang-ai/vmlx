#!/usr/bin/env bash
# JangPress A/B sweep — runs JANGPressMultiTurn across compressPct
# values and aggregates the EVOLUTION + ROUTER ADVISOR blocks into a
# single summary table.
#
# Usage:
#   jangpress_sweep.sh <bundle> [pcts="0 50 70 100"] [turns=3] [outdir]
#
# Each run is a fresh process so JangPress prestack cache + page cache
# state from the previous run can be cold-flushed by the kernel; the
# kernel still keeps file pages in the unified page cache, so order
# matters: lower pcts run first to bias against JangPress (cache is
# warm by the high-pct runs).
#
# Emits:
#   <outdir>/pct=<N>.log                    — full bench stdout/stderr
#   <outdir>/SUMMARY.md                     — aggregated comparison

set -euo pipefail

BUNDLE="${1:-}"
PCTS="${2:-0 50 70 100}"
TURNS="${3:-3}"
OUTDIR="${4:-/Users/eric/vmlx/sweep-out/$(date +%Y%m%d-%H%M%S)}"

if [ -z "$BUNDLE" ] || [ ! -d "$BUNDLE" ]; then
    echo "usage: $0 <bundle-dir> [pcts] [turns] [outdir]" >&2
    echo "  e.g. $0 /Volumes/EricsLLMDrive/jangq-ai/Laguna-XS.2-JANGTQ" >&2
    exit 2
fi

mkdir -p "$OUTDIR"
BENCH_BIN="/Users/eric/vmlx/swift/.build/debug/JANGPressMultiTurn"

if [ ! -x "$BENCH_BIN" ]; then
    echo "[sweep] building JANGPressMultiTurn…" >&2
    (cd /Users/eric/vmlx/swift && swift build --product JANGPressMultiTurn) >&2
fi

echo "# JangPress sweep — $(basename "$BUNDLE")" > "$OUTDIR/SUMMARY.md"
echo "" >> "$OUTDIR/SUMMARY.md"
echo "Bundle: \`$BUNDLE\`  " >> "$OUTDIR/SUMMARY.md"
echo "Turns: $TURNS  " >> "$OUTDIR/SUMMARY.md"
echo "Date: $(date '+%Y-%m-%d %H:%M:%S')  " >> "$OUTDIR/SUMMARY.md"
echo "" >> "$OUTDIR/SUMMARY.md"
echo "| pct | load_ms | post-load footprint | turn1 ttft | turn1 tps | turn2 ttft | turn2 tps | turn3 ttft | turn3 tps | post-T3 footprint | rewarms | thrash |" >> "$OUTDIR/SUMMARY.md"
echo "|---|---|---|---|---|---|---|---|---|---|---|---|" >> "$OUTDIR/SUMMARY.md"

for PCT in $PCTS; do
    LOG="$OUTDIR/pct=$PCT.log"
    echo "[sweep] === pct=$PCT === → $LOG" >&2
    "$BENCH_BIN" "$BUNDLE" "$PCT" "$TURNS" 2>&1 | tee "$LOG" >/dev/null

    # Pull values from the log:
    LOAD_MS=$(grep -E '^\[load\] [0-9]+ ms' "$LOG" | head -1 | sed -E 's/.*\[load\] ([0-9]+) ms.*/\1/' || echo "?")
    PL_FP=$(grep -E '^\[load\] footprint post-load:' "$LOG" | head -1 | sed -E 's/.*footprint post-load: ([0-9.]+ GB).*/\1/' || echo "?")

    # Turn rows: "Turn 1: ttft= 1234 ms, decode=5678 ms, tps= 9.99, ..."
    T1_TTFT=$(grep -E '^Turn 1: ttft=' "$LOG" | sed -E 's/.*ttft= *([0-9]+) ms.*/\1/' || echo "?")
    T2_TTFT=$(grep -E '^Turn 2: ttft=' "$LOG" | sed -E 's/.*ttft= *([0-9]+) ms.*/\1/' || echo "?")
    T3_TTFT=$(grep -E '^Turn 3: ttft=' "$LOG" | sed -E 's/.*ttft= *([0-9]+) ms.*/\1/' || echo "?")
    T1_TPS=$(grep -E '^Turn 1: ' "$LOG" | sed -E 's/.*tps= *([0-9.]+).*/\1/' || echo "?")
    T2_TPS=$(grep -E '^Turn 2: ' "$LOG" | sed -E 's/.*tps= *([0-9.]+).*/\1/' || echo "?")
    T3_TPS=$(grep -E '^Turn 3: ' "$LOG" | sed -E 's/.*tps= *([0-9.]+).*/\1/' || echo "?")
    T3_FP=$(grep -E '^Turn 3: ' "$LOG" | sed -E 's/.*footprint=([0-9.]+ GB).*/\1/' || echo "?")

    REWARMS=$(grep -E 'warmCalls=.*rewarms=' "$LOG" | sed -E 's/.*rewarms=([0-9]+).*/\1/' || echo "?")
    THRASH=$(grep -E 'warmCalls=.*thrashRatio=' "$LOG" | sed -E 's/.*thrashRatio=([0-9.]+).*/\1/' || echo "?")

    printf '| %s | %s | %s | %s ms | %s | %s ms | %s | %s ms | %s | %s | %s | %s |\n' \
        "$PCT" "$LOAD_MS" "$PL_FP" \
        "$T1_TTFT" "$T1_TPS" "$T2_TTFT" "$T2_TPS" "$T3_TTFT" "$T3_TPS" \
        "$T3_FP" "$REWARMS" "$THRASH" >> "$OUTDIR/SUMMARY.md"
done

echo "" >> "$OUTDIR/SUMMARY.md"
echo "## Notes" >> "$OUTDIR/SUMMARY.md"
echo "" >> "$OUTDIR/SUMMARY.md"
echo "- pct=0 baseline: JangPress logically a no-op (no DONTNEED issued)." >> "$OUTDIR/SUMMARY.md"
echo "- pct=100 stress: every routed expert range advised cold." >> "$OUTDIR/SUMMARY.md"
echo "- Run order matters: lower pcts first so OS page cache state biases against JangPress." >> "$OUTDIR/SUMMARY.md"
echo "- footprint = phys_footprint (Activity Monitor 'Memory' column)." >> "$OUTDIR/SUMMARY.md"

echo "" >&2
echo "[sweep] done. SUMMARY.md:" >&2
cat "$OUTDIR/SUMMARY.md" >&2
