#!/usr/bin/env bash
# verify.sh — one-command production verification.
#
# 1. Audits on-disk models (tests/e2e/audit-disk.sh)
# 2. Builds vmlxctl if not already built
# 3. Runs the tier-1 matrix (tests/e2e/run-matrix.sh 1)
# 4. Compares the per-model pass/fail + tps against the last run's
#    baseline (tests/e2e/results/baseline.json) and prints a delta.
# 5. Exits non-zero on regression (any case that was passing is now
#    failing, or tps dropped > 30% from baseline).
#
# Flags:
#   --update-baseline    save this run as the new baseline
#   --skip-build         assume vmlxctl is fresh
#   --tier <N>           default 1; passes through to run-matrix.sh

set -u
ROOT=$(cd "$(dirname "$0")/../.." && pwd)
cd "$ROOT"

TIER=1
UPDATE_BASELINE=0
SKIP_BUILD=0
while [ $# -gt 0 ]; do
    case "$1" in
        --update-baseline) UPDATE_BASELINE=1; shift ;;
        --skip-build)      SKIP_BUILD=1; shift ;;
        --tier)            TIER="$2"; shift 2 ;;
        -h|--help)
            grep '^#' "$0" | sed 's/^# \?//'
            exit 0 ;;
        *) echo "unknown flag: $1" >&2; exit 1 ;;
    esac
done

BASELINE="$ROOT/tests/e2e/results/baseline.json"
STAMP=$(date +%Y%m%d-%H%M%S)

section() { printf '\n\033[1;34m== %s ==\033[0m\n' "$1"; }

# ---- 1. audit ---------------------------------------------------------
section "Disk audit"
tests/e2e/audit-disk.sh | tee "tests/e2e/results/disk-$STAMP.txt"

# ---- 2. build ---------------------------------------------------------
if [ "$SKIP_BUILD" -eq 0 ]; then
    section "Build vmlxctl"
    swift build --product vmlxctl -c release 2>&1 | tail -3
fi

if [ ! -f ".build/arm64-apple-macosx/release/vmlxctl" ]; then
    echo "vmlxctl binary missing — build failed or --skip-build used too early" >&2
    exit 2
fi

# ---- 3. run matrix ---------------------------------------------------
section "Tier-$TIER matrix"
tests/e2e/run-matrix.sh "$TIER" 2>&1
# Find the just-created matrix dir (most recent in results/)
RUN_DIR=$(ls -1td tests/e2e/results/matrix-*/ 2>/dev/null | head -1)
[ -z "$RUN_DIR" ] && { echo "no matrix output produced" >&2; exit 3; }
RUN_DIR=${RUN_DIR%/}

# ---- 4. compare against baseline + build delta -----------------------
section "Regression check"
python3 - "$RUN_DIR" "$BASELINE" "$UPDATE_BASELINE" <<'PY'
import json, os, sys, glob
run_dir, baseline_path, update = sys.argv[1], sys.argv[2], sys.argv[3] == "1"

# Aggregate this run per model.
current = {}
for jsonl in sorted(glob.glob(f"{run_dir}/*.jsonl")):
    label = os.path.basename(jsonl).replace(".jsonl", "")
    cases = {}
    tps = None
    for line in open(jsonl):
        line = line.strip()
        if not line.startswith("{"): continue
        try: d = json.loads(line)
        except Exception: continue
        if d.get("name"): cases[d["name"]] = bool(d.get("ok", False))
        if d.get("name") == "sse_stream":
            tps = d.get("tps")
    current[label] = {"cases": cases, "tps": tps}

# Print per-model summary
def line(label, passed, failed, tps, tag=""):
    print(f"  {label:30s} {passed:3d} pass / {failed:3d} fail  tps={tps}  {tag}")

for label, data in current.items():
    p = sum(1 for v in data["cases"].values() if v)
    f = sum(1 for v in data["cases"].values() if not v)
    line(label, p, f, data.get("tps"))

# Compare against baseline
regressions = []
if os.path.exists(baseline_path):
    baseline = json.load(open(baseline_path))
    print("\n  ── deltas vs baseline ──")
    for label, data in current.items():
        b = baseline.get(label, {})
        b_cases = b.get("cases", {})
        b_tps = b.get("tps") or 0
        # Case regressions: any case that was ok=True in baseline and is now ok=False
        for case, ok in data["cases"].items():
            was = b_cases.get(case)
            if was is True and ok is False:
                regressions.append(f"{label}:{case} (was pass, now FAIL)")
        # tps regression >30%
        cur_tps = data.get("tps") or 0
        if b_tps and cur_tps and cur_tps < b_tps * 0.7:
            regressions.append(f"{label}:tps {cur_tps} < baseline {b_tps} (-{100*(b_tps-cur_tps)/b_tps:.0f}%)")
            print(f"  {label}: tps {cur_tps} vs {b_tps} — DROP")
        elif b_tps and cur_tps and cur_tps > b_tps * 1.1:
            print(f"  {label}: tps {cur_tps} vs {b_tps} — IMPROVE")
else:
    print("  (no baseline.json — first run)")

if update:
    with open(baseline_path, "w") as f:
        json.dump(current, f, indent=2)
    print(f"\n  ✓ baseline written to {baseline_path}")

if regressions:
    print(f"\n\033[1;31m✗ {len(regressions)} regression(s):\033[0m")
    for r in regressions: print(f"   - {r}")
    sys.exit(1)
print("\n\033[1;32m✓ no regressions vs baseline\033[0m")
PY
