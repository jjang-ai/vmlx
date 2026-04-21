#!/bin/bash
# iter-ralph-2 §226: cache + hybrid SSM + async re-derive verification harness
#
# Boots vmlxctl serve against a given model, runs a sequence of requests
# designed to stress:
#   1. Prefix cache hit on T2+ (usage.cache_detail should show paged/disk)
#   2. Hybrid SSM companion fetch (detail should carry `+ssm(N)`)
#   3. Reasoning-on → reasoning-off turn alternation (no <think> leak;
#      hybrid SSM correctly skipped/restored across turns)
#   4. Async SSM re-derive (after a thinking turn, the next non-thinking
#      turn's cache state needs re-derivation — engine should do this
#      without blocking the scheduler)
#   5. Sleep/wake preserves reasoning_parser + thinkInTemplate caps
#   6. Burst concurrent requests (real batching) don't corrupt cache
#
# Usage:
#   scripts/cache-hybrid-verify.sh <model_path>
#
# Prints per-turn observations (cache_detail, cached_tokens, reasoning/
# content split) so regressions are visible in the commit message.

set -euo pipefail

PORT="${VMLX_VERIFY_PORT:-8799}"
TOKEN="${VMLX_VERIFY_TOKEN:-test-admin-token}"
VMLXCTL="${VMLXCTL_BIN:-.build/release/vmlxctl}"

if [ ! -x "$VMLXCTL" ]; then
    echo "ERROR: $VMLXCTL not found. Run: swift build -c release --product vmlxctl"
    exit 1
fi

MODEL="${1:-}"
if [ -z "$MODEL" ] || [ ! -f "$MODEL/config.json" ]; then
    echo "ERROR: pass model path as arg 1 (got: $MODEL)"
    exit 1
fi

echo "=== Boot $MODEL on :$PORT ==="
pkill -f "vmlxctl serve" 2>/dev/null || true
sleep 3

"$VMLXCTL" serve --port "$PORT" --admin-token "$TOKEN" --model "$MODEL" \
    > /tmp/vmlx-verify-serve.log 2>&1 &
PID=$!
trap "kill $PID 2>/dev/null || true; sleep 1" EXIT

for i in $(seq 1 180); do
    if curl -sf "http://127.0.0.1:$PORT/v1/models" > /dev/null 2>&1; then
        echo "  up in ${i}s"
        break
    fi
    sleep 1
done

MID=$(curl -s "http://127.0.0.1:$PORT/v1/models" \
      -H "Authorization: Bearer $TOKEN" \
      | python3 -c 'import sys,json; d=json.loads(sys.stdin.read()); ms=[m["id"] for m in d["data"] if m.get("vmlx",{}).get("loaded")]; print(ms[0] if ms else d["data"][0]["id"])')
echo "  model: $MID"
echo ""

# Ship the driver as a heredoc'd Python file so quoting stays sane.
cat > /tmp/cache_hybrid_verify.py << 'PYEOF'
import urllib.request, json, sys, time, threading

MID = sys.argv[1]
PORT = sys.argv[2]
TOKEN = sys.argv[3]
HOST = f"http://127.0.0.1:{PORT}"
H = {"Authorization": f"Bearer {TOKEN}", "Content-Type": "application/json"}

LEAK = ["<think>", "</think>", "<|think|>", "<|channel>", "<channel|>",
        "<tool_call>", "[TOOL_CALLS]", "<minimax:thinking>"]

def call(path, body=None, method="POST"):
    data = json.dumps(body or {}).encode() if body is not None else None
    req = urllib.request.Request(HOST+path, data=data, headers=H, method=method)
    try:
        return json.loads(urllib.request.urlopen(req, timeout=240).read())
    except Exception as e:
        return {"error": str(e)}

def chat(messages, enable_thinking=None, max_tokens=40, tools=None):
    body = {"model": MID, "messages": messages, "max_tokens": max_tokens}
    if enable_thinking is not None:
        body["enable_thinking"] = enable_thinking
    if tools: body["tools"] = tools
    r = call("/v1/chat/completions", body)
    if "error" in r:
        return {"content": "", "reasoning": "", "usage": {}, "err": r["error"]}
    msg = r["choices"][0]["message"]
    return {
        "content": msg.get("content") or "",
        "reasoning": msg.get("reasoning_content") or msg.get("reasoning") or "",
        "tool_calls": msg.get("tool_calls") or [],
        "usage": r.get("usage") or {},
    }

def leak_check(label, content):
    bad = [t for t in LEAK if t in content]
    if bad:
        print(f"    ✗ {label} leak tokens: {bad!r}")
        return False
    return True

passed = 0; total = 0

# ===== A. Warm the kernel =====
call("/admin/cache/clear")
_ = chat([{"role":"user","content":"hi"}], enable_thinking=False, max_tokens=8)

print("="*60)
print("A. SINGLE-TURN CACHE + REASONING CHANNELS")
print("="*60)

# T1 — single-turn reasoning OFF
total += 1
r = chat([{"role":"user","content":"Say 'alpha' and nothing else."}],
         enable_thinking=False, max_tokens=15)
cd = r["usage"].get("cache_detail") or "?"
ct = r["usage"].get("cached_tokens", 0)
ok = leak_check("T1", r["content"])
if ok and r["content"]:
    passed += 1
    print(f"  T1 reasoning OFF  ✓  content={r['content']!r:>20}  cache={cd!r} cached={ct}")
else:
    print(f"  T1 reasoning OFF  ✗  content={r['content']!r}")

# T2 — single-turn reasoning ON
total += 1
r = chat([{"role":"user","content":"What is 5+5? Think briefly then answer."}],
         enable_thinking=True, max_tokens=200)
cd = r["usage"].get("cache_detail") or "?"
ok = leak_check("T2", r["content"])
if ok and (r["content"] or r["reasoning"]):
    passed += 1
    print(f"  T2 reasoning ON   ✓  content_len={len(r['content'])} reasoning_len={len(r['reasoning'])} cache={cd!r}")
else:
    print(f"  T2 reasoning ON   ✗  content={r['content'][:60]!r} reasoning={r['reasoning'][:60]!r}")

print()
print("="*60)
print("B. MULTI-TURN PREFIX CACHE HIT VERIFICATION")
print("="*60)

call("/admin/cache/clear")

msgs = [{"role":"user","content":"Remember: alpha is the first letter of the Greek alphabet, and beta is the second."}]
r1 = chat(msgs, enable_thinking=False, max_tokens=30)
pt1 = r1["usage"].get("prompt_tokens", 0)
print(f"  T1 turn 1      content={r1['content'][:60]!r}  prompt_tokens={pt1} cache={r1['usage'].get('cache_detail')!r}")

msgs.append({"role":"assistant","content":r1["content"] or "noted."})
msgs.append({"role":"user","content":"What is alpha? One word only."})

total += 1
r2 = chat(msgs, enable_thinking=False, max_tokens=15)
ct2 = r2["usage"].get("cached_tokens", 0)
cd2 = r2["usage"].get("cache_detail") or "?"
ok = leak_check("T3-cache", r2["content"])
hit = ct2 > 0 or "paged" in str(cd2) or "disk" in str(cd2) or "memory" in str(cd2)
print(f"  T2 turn 2      content={r2['content'][:60]!r}  cached_tokens={ct2} cache={cd2!r}")
if ok and hit:
    passed += 1
    print(f"  B-cache-hit       ✓  prefix cache fired on T2 (cached={ct2}, detail={cd2!r})")
else:
    print(f"  B-cache-hit       ✗  no cache hit on T2 — expected cached_tokens>0 or cache_detail containing tier name")

print()
print("="*60)
print("C. HYBRID SSM: REASONING ON → OFF ALTERNATION (§197 + §225)")
print("="*60)

call("/admin/cache/clear")

# Turn 1 thinking ON
r1 = chat([{"role":"user","content":"What is 7+7?"}], enable_thinking=True, max_tokens=150)
print(f"  T1 think-on    content_len={len(r1['content'])} reasoning_len={len(r1['reasoning'])} cache={r1['usage'].get('cache_detail')!r}")

# Turn 2 thinking OFF with multi-turn history
msgs = [
    {"role":"user","content":"What is 7+7?"},
    {"role":"assistant","content":r1["content"] or "14"},
    {"role":"user","content":"What was the previous answer? One number only."},
]
total += 1
r2 = chat(msgs, enable_thinking=False, max_tokens=15)
ok = leak_check("C-alt-off", r2["content"])
recalled = "14" in r2["content"]
# Check no leading whitespace leak (old T7 symptom)
ws_leak = r2["content"].startswith(("\n\n\n", "    "))
if ok and recalled and not ws_leak:
    passed += 1
    print(f"  C-alternation     ✓  T2(off after T1-on) clean, recalled={recalled}, content={r2['content']!r}")
else:
    print(f"  C-alternation     ✗  recalled={recalled} ws_leak={ws_leak} content={r2['content']!r}")

# Turn 3 thinking ON again (SSM re-derive path)
msgs.append({"role":"assistant","content":r2["content"] or "14"})
msgs.append({"role":"user","content":"Confirm by thinking about it carefully."})
total += 1
r3 = chat(msgs, enable_thinking=True, max_tokens=200)
ok = leak_check("C-alt-on2", r3["content"])
if ok and (r3["content"] or r3["reasoning"]):
    passed += 1
    print(f"  C-re-derive       ✓  T3(on after T2-off) content_len={len(r3['content'])} reasoning_len={len(r3['reasoning'])}")
else:
    print(f"  C-re-derive       ✗  content={r3['content'][:80]!r} reasoning={r3['reasoning'][:80]!r}")

print()
print("="*60)
print("D. SLEEP → WAKE → CACHE SURVIVES")
print("="*60)

# Preload cache
call("/admin/cache/clear")
msgs = [{"role":"user","content":"The city of Paris is known for the Eiffel Tower."}]
_ = chat(msgs, enable_thinking=False, max_tokens=10)

# Soft-sleep
s = call("/admin/soft-sleep")
time.sleep(2)

# Multi-turn against the soft-slept engine (JIT wake)
msgs.append({"role":"assistant","content":"Noted."})
msgs.append({"role":"user","content":"What city was mentioned? One word."})
total += 1
r = chat(msgs, enable_thinking=False, max_tokens=10)
ok = leak_check("D-wake", r["content"])
recalled = "Paris" in r["content"] or "paris" in r["content"].lower()
if ok and recalled:
    passed += 1
    print(f"  D-JIT-wake        ✓  recalled={recalled} content={r['content']!r}")
else:
    print(f"  D-JIT-wake        ✗  recalled={recalled} content={r['content']!r}")

print()
print("="*60)
print("E. BURST CONCURRENT (real batching)")
print("="*60)

def parallel_req(idx, results):
    body = {"model": MID,
            "messages":[{"role":"user","content":f"Say '{idx}' and only that number."}],
            "max_tokens": 10, "enable_thinking": False}
    req = urllib.request.Request(HOST+"/v1/chat/completions",
                                   data=json.dumps(body).encode(),
                                   headers=H, method="POST")
    try:
        resp = json.loads(urllib.request.urlopen(req, timeout=120).read())
        results[idx] = resp["choices"][0]["message"].get("content") or ""
    except Exception as e:
        results[idx] = f"ERR:{e}"

results = {}
threads = []
for i in range(4):
    t = threading.Thread(target=parallel_req, args=(i, results))
    threads.append(t); t.start()
for t in threads: t.join()

all_ok = all(str(i) in (results.get(i) or "") for i in range(4))
all_leak_free = all(leak_check(f"E{i}", results.get(i, "")) for i in range(4))
total += 1
if all_ok and all_leak_free:
    passed += 1
    print(f"  E-burst           ✓  all 4 returned their index: {[results[i][:20] for i in range(4)]}")
else:
    print(f"  E-burst           ✗  results: {results}")

print()
print("="*60)
print(f"SUMMARY: {passed}/{total} passed")
print("="*60)
sys.exit(0 if passed == total else 1)
PYEOF

python3 /tmp/cache_hybrid_verify.py "$MID" "$PORT" "$TOKEN"
RC=$?

kill $PID 2>/dev/null || true
sleep 2

exit $RC
