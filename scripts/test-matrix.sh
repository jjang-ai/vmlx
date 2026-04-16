#!/bin/zsh
# Live-model regression matrix for the Swift vMLX engine.
#
# Drives `vmlxctl chat` across a curated set of small cached HuggingFace
# checkpoints spanning different families (Llama 3, Qwen 3, Gemma 4), sends
# a canned prompt, and asserts that the engine produced a non-empty
# response. Tests the ENTIRE load → tokenize → generate → decode path,
# end-to-end, against real model weights.
#
# Does NOT measure quality. Does NOT benchmark tokens/sec. The only thing
# it proves is "the engine successfully loaded this model family and
# emitted coherent text." That's the regression signal we've been missing.
#
# Usage:
#   scripts/test-matrix.sh
#   scripts/test-matrix.sh --fast           # single-model smoke
#   MODELS_ROOT=~/custom/cache scripts/test-matrix.sh   # override cache dir
#
# Prereqs:
#   - swift build -c release (or debug) done
#   - scripts/stage-metallib.sh run (otherwise every load fails)
#   - target model cached under $MODELS_ROOT (default:
#     ~/.cache/huggingface/hub/). The runner skips models that aren't
#     found on disk — download them yourself with
#     `vmlxctl pull <repo>` if you want that row in the matrix.

set -u

cd "$(dirname "$0")/.."

MODELS_ROOT="${MODELS_ROOT:-$HOME/.cache/huggingface/hub}"
CFG="${CFG:-debug}"
BIN=".build/arm64-apple-macosx/${CFG}/vmlxctl"

if [[ ! -x "$BIN" ]]; then
    echo "test-matrix: binary not found at $BIN"
    echo "  run 'swift build -c ${CFG}' and 'scripts/stage-metallib.sh ${CFG}' first"
    exit 1
fi

# Model repo → prompt → expected-non-empty keyword. The keyword is a
# loose sanity check: we accept ANY output containing even a scrap of it
# (case-insensitive). For greeting prompts we just check output isn't
# empty. Keep the prompt short — loading Gemma 4 e2b already eats ~6s
# per row; long prompts just slow the matrix without adding signal.
typeset -A MATRIX_REPOS MATRIX_PROMPTS MATRIX_KEYWORDS
MATRIX_REPOS[llama3]="mlx-community/Llama-3.2-1B-Instruct-4bit"
MATRIX_PROMPTS[llama3]="Say hi in five words."
MATRIX_KEYWORDS[llama3]=""

MATRIX_REPOS[qwen3]="mlx-community/Qwen3-0.6B-8bit"
MATRIX_PROMPTS[qwen3]="What is 2 + 2?"
MATRIX_KEYWORDS[qwen3]="4"

MATRIX_REPOS[gemma4]="mlx-community/gemma-4-e2b-it-4bit"
MATRIX_PROMPTS[gemma4]="Respond with only the capital of France."
MATRIX_KEYWORDS[gemma4]="paris"

# Order defines matrix execution. Smallest first so a broken
# build fails fast on Llama 3.2 1B before sinking 60s into Gemma.
ROWS=(llama3 qwen3 gemma4)

if [[ "${1:-}" == "--fast" ]]; then
    ROWS=(llama3)
fi

PASS=0
FAIL=0
SKIP=0
typeset -A RESULTS

for row in $ROWS; do
    repo="${MATRIX_REPOS[$row]}"
    prompt="${MATRIX_PROMPTS[$row]}"
    keyword="${MATRIX_KEYWORDS[$row]}"

    # Resolve repo → local snapshot dir.
    cache_name="models--${repo//\//--}"
    snapshot_parent="${MODELS_ROOT}/${cache_name}/snapshots"
    if [[ ! -d "$snapshot_parent" ]]; then
        echo "[${row}] SKIP — ${repo} not cached at ${snapshot_parent}"
        RESULTS[$row]=SKIP
        SKIP=$((SKIP + 1))
        continue
    fi
    snapshot=$(ls -d "$snapshot_parent"/*/ 2>/dev/null | head -1)
    if [[ -z "$snapshot" ]]; then
        echo "[${row}] SKIP — no snapshot under $snapshot_parent"
        RESULTS[$row]=SKIP
        SKIP=$((SKIP + 1))
        continue
    fi

    echo
    echo "================================================================"
    echo "[${row}] ${repo}"
    echo "  prompt: ${prompt}"
    echo "================================================================"

    # Run chat REPL with a single prompt. Feed via stdin, exit after
    # the response is emitted. Timeout belt-and-suspenders to avoid
    # a wedged load hanging the matrix.
    tmp_out=$(mktemp)
    printf '%s\n/quit\n' "$prompt" | \
        "$BIN" chat --model "$snapshot" --system "Be extremely brief." \
            > "$tmp_out" 2>&1 &
    pid=$!
    # Max 180s per model (Gemma 4 e2b fresh-load is ~60s).
    (sleep 180 && kill -9 $pid 2>/dev/null) &
    watchdog=$!
    wait $pid 2>/dev/null
    rc=$?
    kill $watchdog 2>/dev/null

    # Strip the "Loading" + "Ready" + marker lines so the assertion only
    # sees actual model output.
    output=$(grep -v -E '^Loading |^Ready\. Type|^\[vMLX\]' "$tmp_out" | head -40)
    rm -f "$tmp_out"

    if [[ -z "${output// }" ]]; then
        echo "[${row}] FAIL — empty output (rc=$rc)"
        RESULTS[$row]=FAIL
        FAIL=$((FAIL + 1))
        continue
    fi

    # Keyword check (case-insensitive). Empty keyword = "just needs
    # SOMETHING coherent", which we already confirmed by non-empty.
    if [[ -n "$keyword" ]]; then
        if ! echo "$output" | grep -qi "$keyword"; then
            echo "[${row}] FAIL — expected keyword '${keyword}' not in output:"
            echo "${output}" | head -5 | sed 's/^/    /'
            RESULTS[$row]=FAIL
            FAIL=$((FAIL + 1))
            continue
        fi
    fi

    echo "[${row}] PASS"
    echo "  output: $(echo "$output" | head -2 | tr '\n' ' ' | cut -c1-120)"
    RESULTS[$row]=PASS
    PASS=$((PASS + 1))
done

echo
echo "================================================================"
echo "MATRIX SUMMARY"
echo "================================================================"
for row in $ROWS; do
    printf "  %-10s %s\n" "$row" "${RESULTS[$row]:-unknown}"
done
echo "  ------"
echo "  PASS: $PASS   FAIL: $FAIL   SKIP: $SKIP"

# Non-zero exit if any model failed — don't exit on skips (a CI box
# that doesn't have the weights cached is a different failure mode).
if [[ $FAIL -gt 0 ]]; then
    exit 1
fi
exit 0
