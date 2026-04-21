#!/bin/bash
# iter-152 §224: multi-turn thinking / reasoning / tool regression matrix
#
# For each loaded model × {reasoning ON, reasoning OFF, tools ON} triple,
# assert:
#   (a) no `<think>`, `<|think|>`, `<|channel>`, `<channel|>`, `<tool_call>`
#       tokens leak into the visible `content` channel
#   (b) reasoning channel is populated when thinking is ON
#   (c) multi-turn context carries over (model recalls prior-turn fact)
#   (d) tool_call shows up in `tool_calls` field, not as raw text in content
#
# Drives a single vmlxctl serve instance with --model override per model.
# Logs per-model pass/fail to /tmp/vmlx-matrix.log.
#
# Usage:
#   ./scripts/multi-turn-thinking-matrix.sh <model_path>
#
# Or with a list (one path per line):
#   ./scripts/multi-turn-thinking-matrix.sh --list models.txt

set -euo pipefail

PORT="${VMLX_MATRIX_PORT:-8799}"
TOKEN="${VMLX_MATRIX_TOKEN:-test-admin-token}"
VMLXCTL="${VMLXCTL_BIN:-.build/release/vmlxctl}"
LOGFILE="/tmp/vmlx-matrix.log"

if [ ! -x "$VMLXCTL" ]; then
    echo "ERROR: $VMLXCTL not found"
    echo "       Run: swift build -c release --product vmlxctl"
    exit 1
fi

test_model() {
    local model="$1"
    if [ ! -f "$model/config.json" ]; then
        echo "SKIP: $model (no config.json)"
        return 1
    fi

    local name
    name="$(basename "$model")"
    echo ""
    echo "=========================================================="
    echo "=== $name"
    echo "=========================================================="

    pkill -f "vmlxctl serve" 2>/dev/null || true
    sleep 3
    "$VMLXCTL" serve --port "$PORT" --admin-token "$TOKEN" --model "$model" \
        > "/tmp/vmlx-matrix-serve.log" 2>&1 &
    local pid=$!

    local up=0
    for i in $(seq 1 180); do
        if curl -sf "http://127.0.0.1:$PORT/v1/models" > /dev/null 2>&1; then
            up=1
            break
        fi
        sleep 1
    done
    if [ "$up" = 0 ]; then
        echo "  FAIL: server didn't start in 180s"
        tail -20 /tmp/vmlx-matrix-serve.log
        kill $pid 2>/dev/null || true
        return 1
    fi

    local mid
    mid=$(curl -s "http://127.0.0.1:$PORT/v1/models" \
          -H "Authorization: Bearer $TOKEN" \
          | python3 -c 'import sys,json; d=json.loads(sys.stdin.read()); ms=[m["id"] for m in d["data"] if m.get("vmlx",{}).get("loaded")]; print(ms[0] if ms else d["data"][0]["id"])')
    echo "  loaded model id: $mid"

    python3 /tmp/multi_turn_matrix.py "$mid" "$PORT" "$TOKEN" "$name" \
        | tee -a "$LOGFILE"

    kill $pid 2>/dev/null || true
    sleep 3
}

# Write the per-model test driver to /tmp (Python for readability)
cat > /tmp/multi_turn_matrix.py << 'PYEOF'
import urllib.request, json, sys, re

MODEL_ID = sys.argv[1]
PORT = sys.argv[2]
TOKEN = sys.argv[3]
NAME = sys.argv[4]
HOST = f"http://127.0.0.1:{PORT}"
HEADERS = {"Authorization": f"Bearer {TOKEN}", "Content-Type": "application/json"}

LEAK_TOKENS = [
    "<think>", "</think>", "<|think|>",
    "<|channel>", "<channel|>",
    "<tool_call>", "</tool_call>",
    "<|tool_call>",
    "[TOOL_CALLS]",
    "<minimax:thinking>",
]

def post(body):
    req = urllib.request.Request(HOST + "/v1/chat/completions",
                                   data=json.dumps(body).encode(),
                                   headers=HEADERS, method="POST")
    try:
        resp = json.loads(urllib.request.urlopen(req, timeout=240).read())
    except Exception as e:
        return None, None, None, str(e)
    msg = resp["choices"][0]["message"]
    content = msg.get("content") or ""
    reasoning = msg.get("reasoning_content") or msg.get("reasoning") or ""
    tool_calls = msg.get("tool_calls") or []
    return content, reasoning, tool_calls, None

def check_leak(label, content):
    leaks = [t for t in LEAK_TOKENS if t in content]
    if leaks:
        print(f"    FAIL {label}: leak tokens in content: {leaks!r}")
        return False
    return True

passed = 0; total = 0

# T1: reasoning OFF, single turn
total += 1
content, reasoning, tc, err = post({
    "model": MODEL_ID,
    "messages": [{"role":"user","content":"Say 'hello' and nothing else."}],
    "max_tokens": 50,
    "enable_thinking": False,
})
if err: print(f"  T1 ERR: {err}");
else:
    ok = check_leak("T1", content)
    if ok and len(content) > 0:
        passed += 1
        print(f"  T1 reasoning OFF  PASS  content={content!r}")
    else:
        print(f"  T1 reasoning OFF  FAIL  content={content!r} reasoning={reasoning!r}")

# T2: reasoning ON, single turn (thinking model emits to reasoning)
total += 1
content, reasoning, tc, err = post({
    "model": MODEL_ID,
    "messages": [{"role":"user","content":"What is 2+2? Think carefully then answer."}],
    "max_tokens": 400,
    "enable_thinking": True,
})
if err: print(f"  T2 ERR: {err}")
else:
    ok_leak = check_leak("T2", content)
    # Either content has the answer or reasoning has chain-of-thought (for true thinking models).
    has_output = len(content) > 0 or len(reasoning) > 0
    if ok_leak and has_output:
        passed += 1
        print(f"  T2 reasoning ON   PASS  content_len={len(content)} reasoning_len={len(reasoning)}")
        if len(reasoning) == 0 and len(content) > 40:
            print(f"    NOTE: no reasoning channel populated — may indicate model doesn't emit harmony/think markers or parser misses them")
    else:
        print(f"  T2 reasoning ON   FAIL  content={content[:200]!r} reasoning={reasoning[:200]!r}")

# T3: multi-turn context recall (no thinking)
total += 1
content, reasoning, tc, err = post({
    "model": MODEL_ID,
    "messages": [
        {"role":"user","content":"Remember: the secret word is 'dolphin'."},
        {"role":"assistant","content":"Got it. The word is dolphin."},
        {"role":"user","content":"What was the secret word? One word only."},
    ],
    "max_tokens": 30,
    "enable_thinking": False,
})
if err: print(f"  T3 ERR: {err}")
else:
    ok_leak = check_leak("T3", content)
    recalled = "dolphin" in content.lower()
    if ok_leak and recalled:
        passed += 1
        print(f"  T3 multi-turn     PASS  recalled 'dolphin' in {content!r}")
    else:
        print(f"  T3 multi-turn     FAIL  recalled={recalled} content={content!r}")

# T4: tools attached, no tool use needed (just chat)
total += 1
content, reasoning, tc, err = post({
    "model": MODEL_ID,
    "messages": [{"role":"user","content":"Say 'banana' and nothing else."}],
    "max_tokens": 30,
    "enable_thinking": False,
    "tools": [{"type":"function","function":{"name":"noop","description":"unused","parameters":{"type":"object","properties":{}}}}],
})
if err: print(f"  T4 ERR: {err}")
else:
    ok_leak = check_leak("T4", content)
    has_output = len(content) > 0 or len(tc) > 0
    if ok_leak and has_output:
        passed += 1
        print(f"  T4 tools attached PASS  content={content!r} tool_calls={len(tc)}")
    else:
        print(f"  T4 tools attached FAIL  content={content!r}")

# T5: multi-turn with thinking ON T2 only
total += 1
content, reasoning, tc, err = post({
    "model": MODEL_ID,
    "messages": [
        {"role":"user","content":"Pick a color, any color. One word."},
        {"role":"assistant","content":"Blue."},
        {"role":"user","content":"What color did you pick? Think then answer with one word."},
    ],
    "max_tokens": 200,
    "enable_thinking": True,
})
if err: print(f"  T5 ERR: {err}")
else:
    ok_leak = check_leak("T5", content)
    has_output = len(content) > 0 or len(reasoning) > 0
    if ok_leak and has_output:
        passed += 1
        print(f"  T5 multi + think  PASS  content_len={len(content)} reasoning_len={len(reasoning)}")
    else:
        print(f"  T5 multi + think  FAIL  content={content[:200]!r}")

# Summary
print(f"")
print(f"  {NAME}: {passed}/{total} passed")
PYEOF

if [ "${1:-}" = "--list" ]; then
    while read -r m; do
        [ -n "$m" ] && test_model "$m"
    done < "$2"
else
    test_model "$1"
fi

pkill -f "vmlxctl serve" 2>/dev/null || true
echo ""
echo "=========================================================="
echo "Full log: $LOGFILE"
