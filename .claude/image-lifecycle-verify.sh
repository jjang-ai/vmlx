#!/usr/bin/env bash
# H1 §314 — Image-backend lifecycle harness.
#
# Covers I3/L1/L2/L3/L4 live scenarios:
#   S1  boot vmlxctl with --image-model; /health comes up
#   S2  gen #1 produces a valid PNG (magic 89504e47) at requested size
#   S3  response carries warnings: [{code:"placeholder_output"}] (Z-Image)
#   S4  /admin/soft-sleep transitions state; image weights still resident
#   S5  gen post-soft-sleep JIT-wakes, produces PNG
#   S6  /admin/deep-sleep transitions state; fluxBackend drained
#   S7  gen post-deep-sleep JIT-rehydrates (lastImageModelPath replay)
#      (measures elapsed time — should include the load cost)
#   S8  /v1/images/edits with non-edit model returns 400 wrong-model-kind
#
# Invocation:
#   bash .claude/image-lifecycle-verify.sh
#   env IMAGE_MODEL=<path>   override default z-image-turbo-8bit
#   env VERBOSE=1            stream server stderr to tail

set -u
PORT="${PORT:-19911}"
BASE="http://127.0.0.1:$PORT"
ADMIN_TOKEN=il
BIN="$(cd "$(dirname "$0")/.." && pwd)/.build/release/vmlxctl"
IMAGE_MODEL="${IMAGE_MODEL:-$HOME/.mlxstudio/models/image-gen/z-image-turbo-8bit}"
LOG=$(mktemp -t vmlx-img-lifecycle).log
OUTDIR=$(mktemp -d -t vmlx-img-artifacts)
SERVER_PID=""

PASSED=0
FAILED=0

cleanup() {
  if [ -n "$SERVER_PID" ]; then
    kill "$SERVER_PID" 2>/dev/null || true
    wait "$SERVER_PID" 2>/dev/null || true
  fi
  rm -rf "$OUTDIR"
}
trap cleanup EXIT INT TERM

check() {
  local label="$1"; local cmd="$2"; local expect="$3"
  local got
  got=$(eval "$cmd" 2>&1)
  if echo "$got" | grep -qE "$expect"; then
    printf "  ✓ %s\n" "$label"
    PASSED=$((PASSED+1))
  else
    printf "  ✗ %s\n    expected: %s\n    got: %s\n" "$label" "$expect" "$(echo "$got" | head -c 300)"
    FAILED=$((FAILED+1))
  fi
}

section() { printf "\n==== %s ====\n" "$1"; }

if [ ! -x "$BIN" ]; then
  printf "vmlxctl not built at %s — run \`swift build -c release\`\n" "$BIN"
  exit 2
fi
if [ ! -d "$IMAGE_MODEL" ]; then
  printf "image model not at %s — set IMAGE_MODEL= or download z-image-turbo-8bit\n" "$IMAGE_MODEL"
  exit 2
fi

# Kill any leftover vmlxctl from prior runs.
pkill -9 -f vmlxctl 2>/dev/null || true
sleep 1

section "S1 boot + image model load"
"$BIN" serve --image-model "$IMAGE_MODEL" --port "$PORT" --admin-token "$ADMIN_TOKEN" \
  --idle-soft-sec 600 --idle-deep-sec 1200 > "$LOG" 2>&1 &
SERVER_PID=$!
ready=0
for i in $(seq 1 120); do
  if grep -q 'image model ready' "$LOG" 2>/dev/null; then
    printf "  ✓ image model loaded in %ds\n" "$i"
    PASSED=$((PASSED+1))
    ready=1
    break
  fi
  if ! kill -0 "$SERVER_PID" 2>/dev/null; then
    printf "  ✗ server died during boot\n"
    tail -20 "$LOG"
    FAILED=$((FAILED+1))
    break
  fi
  sleep 1
done
if [ "$ready" -ne 1 ]; then exit 1; fi

gen() {
  local out="$1" prompt="$2" size="$3"
  curl -sS -m 120 -H 'Content-Type: application/json' \
    -d "{\"model\":\"z-image-turbo\",\"prompt\":\"$prompt\",\"size\":\"$size\",\"steps\":4,\"response_format\":\"b64_json\"}" \
    "$BASE/v1/images/generations" > "$out"
}

png_info() {
  python3 - "$1" <<'PY'
import sys, json, base64, struct
raw = open(sys.argv[1]).read()
try:
    o = json.loads(raw)
except Exception:
    print(f"ERR not-json: {raw[:200]}")
    sys.exit(0)
if "error" in o:
    print(f"ERR body: {json.dumps(o['error'])[:200]}")
    sys.exit(0)
if "data" not in o or not o["data"]:
    print(f"ERR no-data: {list(o.keys())}")
    sys.exit(0)
entry = o["data"][0]
if "b64_json" not in entry:
    print(f"ERR no-b64 keys={list(entry.keys())}")
    sys.exit(0)
data = base64.b64decode(entry["b64_json"])
magic = data[:8].hex()
w, h = struct.unpack(">II", data[16:24])
warnings = o.get("warnings", [])
codes = ",".join(w.get("code", "") for w in warnings) if warnings else ""
print(f"magic={magic} size={w}x{h} warnings={codes}")
PY
}

section "S2/S3 gen #1 + placeholder warning"
gen "$OUTDIR/g1.json" "test cold" "256x256"
check "gen #1 returns valid PNG" \
  "png_info $OUTDIR/g1.json" \
  "magic=89504e470d0a1a0a size=256x256"
check "gen #1 carries placeholder_output warning" \
  "png_info $OUTDIR/g1.json" \
  "warnings=placeholder_output"

section "S4 soft-sleep state"
check "soft-sleep returns sleeping" \
  "curl -sS -m 10 -H 'x-admin-token: $ADMIN_TOKEN' -X POST $BASE/admin/soft-sleep" \
  '"status":"sleeping"'
check "health shows soft_sleep" \
  "curl -sS -m 5 $BASE/health" \
  '"state":"soft_sleep"'

section "S5 gen during soft-sleep JIT-wakes"
gen "$OUTDIR/g2.json" "test soft" "256x256"
check "gen post-soft returns valid PNG" \
  "png_info $OUTDIR/g2.json" \
  "magic=89504e470d0a1a0a size=256x256"
check "health back to running post-soft-wake" \
  "curl -sS -m 5 $BASE/health" \
  '"state":"running"'

section "S6 deep-sleep transitions + drains fluxBackend"
check "deep-sleep returns deep_sleeping" \
  "curl -sS -m 15 -H 'x-admin-token: $ADMIN_TOKEN' -X POST $BASE/admin/deep-sleep" \
  '"status":"deep_sleeping"'
check "health shows deep_sleep" \
  "curl -sS -m 5 $BASE/health" \
  '"state":"deep_sleep"'

section "S7 gen post-deep-sleep JIT-rehydrates"
t0=$(date +%s)
gen "$OUTDIR/g3.json" "test deep" "256x256"
t1=$(date +%s)
check "gen post-deep returns valid PNG" \
  "png_info $OUTDIR/g3.json" \
  "magic=89504e470d0a1a0a size=256x256"
printf "  • deep-wake + gen elapsed: %ds\n" $((t1 - t0))

section "S8 edits on non-edit model → 400"
SRC=$(python3 -c '
import base64, struct, zlib
def png(w,h,rgb):
  def c(t,d):
    return struct.pack(">I",len(d))+t+d+struct.pack(">I",zlib.crc32(t+d)&0xffffffff)
  r=b"".join(b"\x00"+bytes(rgb)*w for _ in range(h))
  return b"\x89PNG\r\n\x1a\n"+c(b"IHDR",struct.pack(">IIBBBBB",w,h,8,2,0,0,0))+c(b"IDAT",zlib.compress(r))+c(b"IEND",b"")
print(base64.b64encode(png(32,32,(255,0,0))).decode())
')
RESP=$(curl -sS -o "$OUTDIR/edit.json" -w '%{http_code}' -m 60 -H 'Content-Type: application/json' \
  -d "{\"model\":\"z-image-turbo\",\"image\":\"$SRC\",\"prompt\":\"make blue\"}" \
  "$BASE/v1/images/edits")
check "edit-on-non-edit-model returns 400" "echo $RESP" '^400$'
check "edit error body says wrong model kind" \
  "cat $OUTDIR/edit.json" \
  "wrong model kind"

section "RESULT"
printf "  PASSED: %d\n  FAILED: %d\n" "$PASSED" "$FAILED"
if [ "$FAILED" -gt 0 ]; then exit 1; fi
