#!/bin/bash
# Per-Model Production Checklist (PMC) — one script, one model, all checks.
#
# Boots vmlxctl serve against the given model and runs every assertion that
# must pass for the model to count as "fully working" for streaming + cache
# + reasoning + tools.
#
# Exit code 0 only when ALL checks pass. Anything else = the model is NOT
# production-ready and the Ralph loop must investigate + fix + retest.
#
# Usage:
#   scripts/per-model-pmc.sh <model_path>
#
# Outputs a single-line JSON summary at the end so the loop can grep it
# and decide pass/fail without re-parsing prose.

set -euo pipefail

PORT="${VMLX_PMC_PORT:-8799}"
TOKEN="${VMLX_PMC_TOKEN:-pmc-tok}"
VMLXCTL="${VMLXCTL_BIN:-.build/release/vmlxctl}"
LOG_TAIL=/tmp/vmlx-pmc-tail.log

if [ ! -x "$VMLXCTL" ]; then
    echo "ERROR: $VMLXCTL not found. Run: swift build -c release --product vmlxctl"
    exit 2
fi

MODEL="${1:-}"
if [ -z "$MODEL" ] || [ ! -f "$MODEL/config.json" ]; then
    echo "ERROR: pass a model directory as arg 1 (got: $MODEL)"
    exit 2
fi

NAME="$(basename "$MODEL")"
echo "================================================================"
echo "PMC: $NAME"
echo "================================================================"

pkill -f "vmlxctl serve" 2>/dev/null || true
sleep 3

"$VMLXCTL" serve --port "$PORT" --admin-token "$TOKEN" --model "$MODEL" \
    > "$LOG_TAIL" 2>&1 &
PID=$!
trap "kill $PID 2>/dev/null || true; sleep 1" EXIT

# Wait up to 5 min for load.
UP=0
for i in $(seq 1 300); do
    if curl -sf "http://127.0.0.1:$PORT/v1/models" > /dev/null 2>&1; then
        UP=1
        echo "  boot: ready in ${i}s"
        break
    fi
    sleep 1
done
if [ "$UP" = 0 ]; then
    echo "  boot: TIMEOUT after 300s"
    tail -20 "$LOG_TAIL"
    echo "{\"model\":\"$NAME\",\"pass\":0,\"total\":8,\"reason\":\"boot_timeout\"}"
    exit 1
fi

MID=$(curl -s "http://127.0.0.1:$PORT/v1/models" \
      -H "Authorization: Bearer $TOKEN" \
      | python3 -c 'import sys,json; d=json.loads(sys.stdin.read()); ms=[m["id"] for m in d["data"] if m.get("vmlx",{}).get("loaded")]; print(ms[0] if ms else d["data"][0]["id"])')
echo "  model id: $MID"

# Driver script — emits structured pass/fail JSON.
cat > /tmp/pmc_driver.py << 'PYEOF'
import urllib.request, json, sys, threading, time

MID, PORT, TOKEN, NAME = sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4]
HOST = f"http://127.0.0.1:{PORT}"
H = {"Authorization": f"Bearer {TOKEN}", "Content-Type": "application/json"}

LEAK = ["<think>", "</think>", "<|think|>", "<|channel>", "<channel|>",
        "<tool_call>", "</tool_call>", "[TOOL_CALLS]", "<minimax:thinking>",
        "<|tool_call|>", "<turn|>", "<|turn>"]

def post(path, body):
    req = urllib.request.Request(HOST+path, data=json.dumps(body).encode(),
                                   headers=H, method="POST")
    try:
        return json.loads(urllib.request.urlopen(req, timeout=240).read())
    except Exception as e:
        return {"_err": str(e)}

def post_empty(path):
    req = urllib.request.Request(HOST+path, data=b"{}", headers=H, method="POST")
    try:
        urllib.request.urlopen(req, timeout=30).read()
    except: pass

def chat(messages, **kw):
    body = {"model": MID, "messages": messages, "max_tokens": kw.pop("max_tokens", 50)}
    body.update(kw)
    r = post("/v1/chat/completions", body)
    if "_err" in r:
        return {"err": r["_err"], "content": "", "reasoning": "", "tool_calls": [], "usage": {}}
    msg = r["choices"][0]["message"]
    return {
        "content": msg.get("content") or "",
        "reasoning": msg.get("reasoning_content") or msg.get("reasoning") or "",
        "tool_calls": msg.get("tool_calls") or [],
        "usage": r.get("usage") or {},
        "finish_reason": r["choices"][0].get("finish_reason"),
    }

def chat_stream(messages, **kw):
    body = {"model": MID, "messages": messages, "stream": True,
            "max_tokens": kw.pop("max_tokens", 30)}
    body.update(kw)
    req = urllib.request.Request(HOST+"/v1/chat/completions",
                                   data=json.dumps(body).encode(),
                                   headers=H, method="POST")
    content, reasoning, finish = "", "", None
    try:
        for line in urllib.request.urlopen(req, timeout=180):
            line = line.decode().strip()
            if line.startswith("data: "):
                p = line[6:]
                if p == "[DONE]": break
                try:
                    d = json.loads(p)
                    for ch in d.get("choices", []):
                        delta = ch.get("delta") or {}
                        if delta.get("content"): content += delta["content"]
                        rc = delta.get("reasoning_content") or delta.get("reasoning")
                        if rc: reasoning += rc
                        if ch.get("finish_reason"): finish = ch["finish_reason"]
                except: pass
    except Exception as e:
        return {"err": str(e), "content": content, "reasoning": reasoning, "finish": finish}
    return {"content": content, "reasoning": reasoning, "finish": finish}

def leak_check(label, content):
    bad = [t for t in LEAK if t in content]
    return (len(bad) == 0, bad)

passed, total = 0, 0
fails = []

# C1: boot — already done (we're past it)
total += 1; passed += 1

# C2: single-turn reasoning OFF, content non-empty, no leaks
total += 1
post_empty("/admin/cache/clear")
r = chat([{"role":"user","content":"Say 'hello' and nothing else."}],
         enable_thinking=False, max_tokens=20)
ok_leak, bad = leak_check("C2", r["content"])
if r.get("err"):
    fails.append(f"C2 server-error: {r['err']}")
elif not ok_leak:
    fails.append(f"C2 leak tokens: {bad} content={r['content'][:100]!r}")
elif not r["content"].strip():
    fails.append(f"C2 empty content")
else:
    passed += 1

# C3: single-turn reasoning ON, either content or reasoning populated
total += 1
post_empty("/admin/cache/clear")
r = chat([{"role":"user","content":"What is 2+2? Think briefly then answer."}],
         enable_thinking=True, max_tokens=300)
ok_leak, bad = leak_check("C3", r["content"])
if r.get("err"):
    fails.append(f"C3 server-error: {r['err']}")
elif not ok_leak:
    fails.append(f"C3 leak tokens: {bad}")
elif not (r["content"] or r["reasoning"]):
    fails.append(f"C3 both channels empty")
else:
    passed += 1

# C4: multi-turn recall (no thinking)
# Some model families (MiniMax M2.7, Nemotron-H always-reasoning) ignore
# enable_thinking=False and unconditionally emit <think> blocks, so we
# budget 300 tokens (plenty for either a short non-reasoning answer or a
# reasoning pass + answer) and check for recall in BOTH `content` AND
# `reasoning_content` — the secret word might legitimately land in
# either field depending on whether the model reasoned.
total += 1
post_empty("/admin/cache/clear")
r1 = chat([{"role":"user","content":"Remember: the secret word is 'banana'."}],
          enable_thinking=False, max_tokens=300)
r2 = chat([
    {"role":"user","content":"Remember: the secret word is 'banana'."},
    {"role":"assistant","content":r1.get("content") or "OK."},
    {"role":"user","content":"What was the secret word? One word only."},
], enable_thinking=False, max_tokens=300)
combined = (r2.get("content") or "") + " " + (r2.get("reasoning") or "")
ok_leak, bad = leak_check("C4", r2.get("content") or "")
recalled = "banana" in combined.lower()
if r2.get("err"):
    fails.append(f"C4 server-error: {r2['err']}")
elif not ok_leak:
    fails.append(f"C4 leak tokens: {bad}")
elif not recalled:
    fails.append(f"C4 didn't recall: content={(r2.get('content') or '')[:80]!r} reasoning={(r2.get('reasoning') or '')[:80]!r}")
else:
    passed += 1

# C5: multi-turn cache hit on T2 (cache_detail shows tier or cached_tokens > 0)
total += 1
post_empty("/admin/cache/clear")
msgs = [{"role":"user","content":"The capital of France is Paris. Acknowledge briefly."}]
r1 = chat(msgs, enable_thinking=False, max_tokens=15)
msgs.append({"role":"assistant","content":r1.get("content") or "Got it."})
msgs.append({"role":"user","content":"What is the capital? One word."})
r2 = chat(msgs, enable_thinking=False, max_tokens=15)
cd = str(r2["usage"].get("cache_detail",""))
ct = r2["usage"].get("cached_tokens", 0)
hit = ct > 0 or any(tier in cd for tier in ["paged","memory","disk"])
if not hit:
    fails.append(f"C5 cache miss on T2: cached_tokens={ct} cache_detail={cd!r}")
else:
    passed += 1

# C6: tools attached, no marker leak, content or tool_calls populated.
# max_tokens=300 for thinking-model parity — Gemma 4 always-reasons when
# tools are attached and the template injects `<|channel>thought` even
# under enable_thinking=False. Short budgets truncate mid-marker, which
# causes the reasoning parser's partial-prefix emit to leak fragments
# into content.
total += 1
post_empty("/admin/cache/clear")
r = chat([{"role":"user","content":"Just say 'orange' and nothing else."}],
         enable_thinking=False, max_tokens=300,
         tools=[{"type":"function","function":{"name":"noop","description":"unused","parameters":{"type":"object","properties":{}}}}])
ok_leak, bad = leak_check("C6", r["content"])
if r.get("err"):
    fails.append(f"C6 server-error: {r['err']}")
elif not ok_leak:
    fails.append(f"C6 marker leak: {bad}")
elif not (r["content"] or r["tool_calls"]):
    fails.append(f"C6 empty (no content + no tool_calls)")
else:
    passed += 1

# C7: streaming SSE completes with finish_reason
total += 1
sr = chat_stream([{"role":"user","content":"Say one word."}],
                 enable_thinking=False, max_tokens=15)
if sr.get("err"):
    fails.append(f"C7 stream-error: {sr['err']}")
elif not sr.get("finish"):
    fails.append(f"C7 stream ended without finish_reason")
else:
    passed += 1

# C8: alternating reasoning ON → OFF → ON multi-turn doesn't leak </think>
total += 1
post_empty("/admin/cache/clear")
r1 = chat([{"role":"user","content":"What is 5+5? Think."}], enable_thinking=True, max_tokens=200)
msgs = [
    {"role":"user","content":"What is 5+5? Think."},
    {"role":"assistant","content": r1.get("content") or "10"},
    {"role":"user","content":"Repeat the answer. One number."},
]
r2 = chat(msgs, enable_thinking=False, max_tokens=15)
ok_leak, bad = leak_check("C8", r2["content"])
if r2.get("err"):
    fails.append(f"C8 server-error: {r2['err']}")
elif not ok_leak:
    fails.append(f"C8 reasoning-off-after-on leaked tokens: {bad} content={r2['content']!r}")
else:
    passed += 1

# Summary
print("\n--- PMC results ---")
for f in fails: print(f"  ✗ {f}")
status = "PASS" if passed == total else "FAIL"
print(f"\n  {NAME}: {passed}/{total} {status}")

# Single-line JSON for grep
import json
print(json.dumps({
    "model": NAME,
    "pass": passed,
    "total": total,
    "fails": fails,
}))
sys.exit(0 if passed == total else 1)
PYEOF

python3 /tmp/pmc_driver.py "$MID" "$PORT" "$TOKEN" "$NAME"
RC=$?

kill $PID 2>/dev/null || true
sleep 2
exit $RC
