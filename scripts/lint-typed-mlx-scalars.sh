#!/usr/bin/env bash
# lint-typed-mlx-scalars.sh — catch the MoE half-speed regression.
#
# Scans model forward-pass files for untyped MLXArray scalar
# constructors that are KNOWN to silently promote bfloat16/fp16
# activations to Float32 on every decode step. This is the single
# biggest cause of MoE half-speed slowdowns.
#
# See:
#   - docs/PERF-MOE-HALF-SPEED-ROOT-CAUSE.md  (WHY)
#   - SWIFT-NO-REGRESSION-CHECKLIST.md §27     (PROCESS GATE)
#
# Exit code 0 = clean, 1 = violations found.
#
# Two modes:
#   ./lint-typed-mlx-scalars.sh            # strict mode, zero FP target
#   ./lint-typed-mlx-scalars.sh --scan     # loose mode, triage-first
#
# Strict mode is wired into:
#   - scripts/build-release.sh (blocks release builds)
#   - Tests/vMLXTests/PerfScalarLintTests.swift (blocks swift test)

set -eu

SWIFT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
MODEL_DIRS=(
    "$SWIFT_DIR/Sources/vMLXLLM/Models"
    "$SWIFT_DIR/Sources/vMLXVLM/Models"
    "$SWIFT_DIR/Sources/vMLXLMCommon"
)

MODE="${1:-strict}"

if ! command -v rg >/dev/null 2>&1; then
    echo "lint-typed-mlx-scalars: ripgrep (rg) not installed — skipping." >&2
    exit 0
fi

FAIL=0
TMP=$(mktemp)
trap 'rm -f "$TMP"' EXIT

# ============================================================
# STRICT PATTERNS — every hit is a real bug. No false positives.
# ============================================================
#
# 1. Scalar negation via `MLXArray(0) - <expr>`. Always replace with
#    unary `-<expr>`. Saves one AsType + one broadcast dispatch.
STRICT_NEG='MLXArray\(\s*0\s*\)\s*-\s*\w'
#
# 2. `maximum(x, MLXArray(0))` or `maximum(MLXArray(0), x)` — relu
#    without dtype passthrough. Silently promotes x to fp32.
STRICT_MAX='(MLX\.)?maximum\([^)]*MLXArray\(\s*-?[0-9]+(\.[0-9]+)?\s*\)[^)]*\)'
#
# 3. `compiledLogitSoftcap(*, MLXArray(<literal>))` — softcap with
#    untyped scalar cap. Pass `dtype: <first-arg>.dtype`.
STRICT_SOFTCAP='compiledLogitSoftcap\([^)]*MLXArray\(\s*[^,)]+\s*\)\s*\)'
#
# 4. `MLX.where(*, *, MLXArray(Float.leastNormalMagnitude))` —
#    mask-fill with untyped floor. Pass `dtype: scores.dtype`.
STRICT_WHERE='MLX\.where\([^)]*MLXArray\(\s*-?Float\.(leastNormalMagnitude|infinity)\s*\)\s*\)'
#
# 5. `-MLXArray(Float.infinity)` or `MLXArray(-Float.infinity)` as
#    operand — untyped. SSM time-step floors are a common hit.
STRICT_INF='MLXArray\(\s*-?Float\.(infinity|leastNormalMagnitude)\s*\)'

# ============================================================
# Loose scan: everything that LOOKS like MLXArray(<literal>) without
# `dtype:`. Human triages; don't run in CI.
# ============================================================
LOOSE='MLXArray\(\s*[^,)]+\s*\)'

if [[ "$MODE" == "--scan" ]]; then
    echo "[lint] LOOSE scan — human triage mode"
    for dir in "${MODEL_DIRS[@]}"; do
        [[ -d "$dir" ]] || continue
        rg --no-heading --line-number --type swift \
            -e "$LOOSE" "$dir" 2>/dev/null \
            | grep -Ev 'dtype:|wrappedValue = MLXArray|MLXArray\(Int32\(|MLXArray\(Array\(|MLXArray\(from:|MLXArray\(converting:|MLXArray\(pow|MLXArray\[|MLXArray\(\)\s|asType' \
            >> "$TMP" || true
    done
    if [[ -s "$TMP" ]]; then
        echo ""; cat "$TMP"
        exit 0   # scan mode never fails the build
    fi
    echo "[lint] no scan hits."
    exit 0
fi

# ============================================================
# STRICT mode (default)
# ============================================================
echo "[lint] scanning model hot-path files for untyped MLXArray scalars (strict)…"
for dir in "${MODEL_DIRS[@]}"; do
    [[ -d "$dir" ]] || continue
    rg --no-heading --line-number --type swift \
        -e "$STRICT_NEG" \
        -e "$STRICT_MAX" \
        -e "$STRICT_SOFTCAP" \
        -e "$STRICT_WHERE" \
        -e "$STRICT_INF" \
        "$dir" \
        2>/dev/null \
        >> "$TMP" || true
done

# Strip lines that ARE typed (the match boundary can still include `dtype:` if
# the regex grabbed too far; be safe). Also drop anything within 3 lines of a
# `lint-ok:` suppression comment so we can annotate genuine false positives
# inline without blanket-excluding whole files.
grep -v 'dtype:' "$TMP" > "${TMP}.typed" || true
awk 'BEGIN{FS=":"}
    NR==FNR { if ($0 ~ /lint-ok:/) skip[$1":"$2]=1; next }
    { ok=1; f=$1; n=$2+0;
      for (i=1; i<=3; i++) if ((f":"(n-i)) in skip) ok=0;
      if (ok) print }
' <(for d in "${MODEL_DIRS[@]}"; do [[ -d "$d" ]] && rg --no-heading --line-number --type swift 'lint-ok:' "$d" 2>/dev/null; done) \
  "${TMP}.typed" > "${TMP}.filtered"
mv "${TMP}.filtered" "$TMP"
rm -f "${TMP}.typed"

if [[ -s "$TMP" ]]; then
    echo "" >&2
    echo "[lint] FAIL — untyped MLXArray scalars in hot-path files:" >&2
    echo "       (See SWIFT-NO-REGRESSION-CHECKLIST.md §27 and" >&2
    echo "        docs/PERF-MOE-HALF-SPEED-ROOT-CAUSE.md for WHY." >&2
    echo "        Fix: pass \`dtype: <activation>.dtype\` to the constructor," >&2
    echo "        or rewrite \`MLXArray(0) - x\` as \`-x\`.)" >&2
    echo "" >&2
    cat "$TMP" >&2
    echo "" >&2
    echo "       Hit count: $(wc -l < "$TMP" | tr -d ' ')" >&2
    FAIL=1
fi

if [[ $FAIL -eq 0 ]]; then
    echo "[lint] OK — no untyped MLXArray scalars in model forward paths."
fi

exit $FAIL
