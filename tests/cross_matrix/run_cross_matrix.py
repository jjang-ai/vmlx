#!/usr/bin/env python3
"""Cross-matrix cache regression runner (NO-REGRESSION-CHECKLIST.md §14).

Iterates cache backend × KV quant × model family rows from §14.2,
spawns a fresh vmlx_engine server per row, runs the 7-turn protocol
from run_big3.py, asserts coherence + cache-hit growth, writes
JSON + a printable summary.

Exit code != 0 on any failure. Row filter: --row X4,X8 runs only
those rows. --fast runs odd-numbered rows. --skip-slow drops the
119B Mistral row.

All paths come from §13 Model Registry. Missing models are reported
as SKIP (not silently passed).
"""
import argparse
import json
import os
import subprocess
import sys
import time

TEST_DIR = "/tmp/vmlx_mt_test"
os.makedirs(TEST_DIR, exist_ok=True)

# run_big3.run_model does one full 7-turn harness on an already-started
# server. We reuse its post/build helpers but start the server ourselves
# with per-row flags.
sys.path.insert(0, TEST_DIR)
if "run_big3" in sys.modules:
    del sys.modules["run_big3"]
from run_big3 import (  # noqa: E402
    BASE,
    build_image_turn,
    build_text_turn,
    cache_hits,
    coherence_fail,
    extract_reply,
    post,
    solid_png,
    wait_loaded,
)

# --- Cache backend presets ---------------------------------------------

BACKENDS = {
    "none": [
        "--disable-prefix-cache",
        # intentionally no paged
    ],
    "legacy_prefix": [
        "--enable-prefix-cache",
        "--no-memory-aware-cache",
    ],
    "mem_aware": [
        "--enable-prefix-cache",
    ],
    "paged": [
        "--enable-prefix-cache",
        "--use-paged-cache",
    ],
    "paged_blockdisk": [
        "--enable-prefix-cache",
        "--use-paged-cache",
        "--enable-block-disk-cache",
    ],
    "paged_promptdisk": [
        "--enable-prefix-cache",
        "--use-paged-cache",
        "--enable-disk-cache",
    ],
}

# Backends that are expected to produce cache hits on turn 2+.
# "none" must NOT grow hits (and the harness only checks coherence).
CACHEABLE = {"legacy_prefix", "mem_aware", "paged", "paged_blockdisk", "paged_promptdisk"}


def server_log_had_mixed_attention(log_path):
    """Return True if mixed-attention diagnostics fired in this server log."""
    try:
        with open(log_path, "r") as f:
            data = f.read()
        return "mixed-attention model detected" in data
    except Exception:
        return False


# --- Row definitions (§14.2) -------------------------------------------

ROWS = [
    dict(
        id="X1",
        label="Qwen3.5-4B-JANG_4K + paged+L2",
        path="/Users/eric/jang/models/Qwen3.5-4B-JANG_4K",
        is_vl=False,
        backend="paged_blockdisk",
        kv_quant="none",
        notes="uniform JANG baseline",
    ),
    dict(
        id="X2",
        label="Qwen3.5-4B-JANG_4K + mem_aware (TQ native)",
        path="/Users/eric/jang/models/Qwen3.5-4B-JANG_4K",
        is_vl=False,
        backend="mem_aware",
        kv_quant="none",
        notes="TurboQuant auto-enabled (JANG default). Tests TQ + memory-aware prefix cache interaction.",
    ),
    dict(
        id="X3",
        label="Qwen3.5-4B-JANG_4K + NO CACHE",
        path="/Users/eric/jang/models/Qwen3.5-4B-JANG_4K",
        is_vl=False,
        backend="none",
        kv_quant="none",
        notes="no-cache sanity",
    ),
    dict(
        id="X4",
        label="Gemma-4-26B-JANG_4M + paged+L2",
        path="/Users/eric/jang/models/Gemma-4-26B-A4B-it-JANG_4M",
        is_vl=False,
        backend="paged_blockdisk",
        kv_quant="none",
        notes="mixed attention + RotatingKVCache paged/L2",
    ),
    dict(
        id="X5",
        label="Gemma-4-26B-JANG_4M + mem_aware (TQ + mixed attn)",
        path="/Users/eric/jang/models/Gemma-4-26B-A4B-it-JANG_4M",
        is_vl=False,
        backend="mem_aware",
        kv_quant="none",
        notes="TQ + mixed attention. TurboQuant auto-enabled.",
    ),
    dict(
        id="X6",
        label="Qwen3.5-VL-4B-JANG_4S-CRACK + paged+L2 (VL)",
        path="/Users/eric/.mlxstudio/models/Qwen3.5-VL-4B-JANG_4S-CRACK",
        is_vl=True,
        backend="paged_blockdisk",
        kv_quant="none",
        notes="VL + image alternation",
    ),
    dict(
        id="X7",
        label="Qwen3.5-VL-4B-JANG_4S-CRACK + mem_aware (VL)",
        path="/Users/eric/.mlxstudio/models/Qwen3.5-VL-4B-JANG_4S-CRACK",
        is_vl=True,
        backend="mem_aware",
        kv_quant="none",
        notes="VL on non-paged path",
    ),
    dict(
        id="X8",
        label="Nemotron-Cascade-2-30B-JANG_2L + paged+L2",
        path="/Users/eric/.mlxstudio/models/Nemotron-Cascade-2-30B-A3B-JANG_2L-CRACK",
        is_vl=False,
        backend="paged_blockdisk",
        kv_quant="none",
        notes="hybrid SSM + SSM companion (paged L2)",
    ),
    dict(
        id="X9",
        label="Nemotron-Cascade-2-30B-JANG_2L + legacy prefix",
        path="/Users/eric/.mlxstudio/models/Nemotron-Cascade-2-30B-A3B-JANG_2L-CRACK",
        is_vl=False,
        backend="legacy_prefix",
        kv_quant="none",
        notes="hybrid SSM on legacy path",
    ),
    dict(
        id="X10",
        label="Mistral-Small-4-119B-JANG_4M-CRACK + paged+L2",
        path="/Users/eric/models/Mistral-Small-4-119B-JANG_4M-CRACK",
        is_vl=False,
        backend="paged_blockdisk",
        kv_quant="none",
        notes="MLA §5.1 + §8d regression model",
        slow=True,
    ),
    dict(
        id="X11",
        label="MiniMax-M2.5-JANG_2L-CRACK + paged",
        path="/Users/eric/.mlxstudio/models/MiniMax-M2.5-JANG_2L-CRACK",
        is_vl=False,
        backend="paged",
        kv_quant="none",
        notes="thinking model, uniform attn",
    ),
    dict(
        id="X12",
        label="gemma-4-e2b-it-4bit + paged+L2 (stock)",
        path="/Users/eric/osaurus_models/finished/gemma-4-e2b-it-4bit",
        is_vl=False,
        backend="paged_blockdisk",
        kv_quant="none",
        notes="stock MLX smoke",
    ),
    dict(
        id="X13",
        label="gemma-4-e2b-it-4bit + NO CACHE (stock MLX, no TQ)",
        path="/Users/eric/osaurus_models/finished/gemma-4-e2b-it-4bit",
        is_vl=False,
        backend="none",
        kv_quant="none",
        notes="stock MLX smoke — no TQ (not JANG), no cache",
    ),
    dict(
        id="X14",
        label="Mistral-Small-4-119B-JANG_4M-CRACK + mem_aware (MLA, TQ skipped)",
        path="/Users/eric/models/Mistral-Small-4-119B-JANG_4M-CRACK",
        is_vl=False,
        backend="mem_aware",
        kv_quant="none",
        notes="MLA model — TQ auto-skipped (CacheList). Verify bf16 KV works on mem_aware.",
        slow=True,
    ),
    dict(
        id="X15",
        label="Qwen3.5-VL-4B-JANG_4S-CRACK + paged (VL, reasoning OFF)",
        path="/Users/eric/.mlxstudio/models/Qwen3.5-VL-4B-JANG_4S-CRACK",
        is_vl=True,
        backend="paged",
        kv_quant="none",
        notes="VL + TQ + reasoning OFF — tests §15 streaming fix",
    ),
]


# --- Server start ------------------------------------------------------


def start_server(row):
    """Spawn a fresh vmlx_engine server configured for this row. Returns Popen."""
    log = os.path.join(TEST_DIR, f"server_{row['id']}.log")
    subprocess.run(["pkill", "-f", "vmlx_engine.cli"], capture_output=True)
    time.sleep(3)
    cmd = [
        sys.executable, "-m", "vmlx_engine.cli", "serve",
        row["path"],
        "--port", "9999",
        "--host", "127.0.0.1",
        "--continuous-batching",
    ]
    cmd += BACKENDS[row["backend"]]
    if row["kv_quant"] != "none":
        cmd += ["--kv-cache-quantization", row["kv_quant"]]
    with open(log, "w") as lf:
        lf.write("# cmd: " + " ".join(cmd) + "\n\n")
        lf.flush()
        p = subprocess.Popen(
            cmd,
            stdout=lf,
            stderr=subprocess.STDOUT,
            env=os.environ.copy(),
            cwd="/Users/eric/mlx/vllm-mlx",
        )
    return p, log


# --- 7-turn harness (thin wrapper around run_big3 helpers) -------------


def run_row(row):
    """One full 7-turn run against the already-started server."""
    results = {
        "id": row["id"],
        "label": row["label"],
        "path": row["path"],
        "is_vl": row["is_vl"],
        "backend": row["backend"],
        "kv_quant": row["kv_quant"],
        "turns": [],
        "final": {},
    }
    print(f"\n{'='*78}")
    print(f"=== {row['id']}  {row['label']}")
    print(f"    backend={row['backend']}  kv_quant={row['kv_quant']}  vl={row['is_vl']}")
    print(f"    path: {row['path']}")
    print(f"{'='*78}")
    if not os.path.isdir(row["path"]):
        print(f"  SKIP — path missing")
        results["skipped"] = True
        return results

    proc, log = start_server(row)
    t0 = time.time()
    ok = wait_loaded(max_s=900)
    load_s = time.time() - t0
    print(f"  load: {'OK' if ok else 'TIMEOUT'} in {load_s:.0f}s  log={log}")
    if not ok:
        results["turns"] = [{"name": "load", "status": "FAIL", "fail": "timeout", "reply": ""}]
        proc.terminate()
        return results

    pre_hits, backend_name = cache_hits()
    history = []
    red = solid_png((255, 0, 0))
    blue = solid_png((0, 0, 255))

    def record(turn_name, body, resp, http_code, expect_refs=None):
        reply = extract_reply(resp) if http_code == 200 else ""
        fail = None
        if http_code != 200:
            fail = f"http_{http_code}: {str(resp)[:100]}"
        else:
            fail = coherence_fail(reply)
            if not fail and expect_refs:
                lower = reply.lower()
                if not any(kw.lower() in lower for kw in expect_refs):
                    fail = f"no_ref: missing {expect_refs}"
        status = "FAIL" if fail else "OK"
        short = reply[:140].replace("\n", " ")
        print(f"  {turn_name}: {status}  {short}")
        if fail:
            print(f"        reason: {fail}")
        results["turns"].append({
            "name": turn_name,
            "status": status,
            "fail": fail,
            "reply": reply[:300],
        })
        return reply

    is_vl = row["is_vl"]

    # T1
    body = build_text_turn("What is the capital of France? Answer in one word.", history)
    code, resp = post(f"{BASE}/v1/chat/completions", body)
    reply = record("T1 capital", body, resp, code, expect_refs=["paris"])
    history += [
        {"role": "user", "content": "What is the capital of France? Answer in one word."},
        {"role": "assistant", "content": reply.strip()[:100]},
    ]

    # T2
    if is_vl:
        body = build_image_turn("What color is the image?", red, history)
        code, resp = post(f"{BASE}/v1/chat/completions", body)
        reply = record("T2 vl:red", body, resp, code, expect_refs=["red"])
        history += [
            {"role": "user", "content": "What color is the image?"},
            {"role": "assistant", "content": reply.strip()[:100]},
        ]
    else:
        body = build_text_turn("What is its most famous landmark?", history)
        code, resp = post(f"{BASE}/v1/chat/completions", body)
        reply = record("T2 landmark", body, resp, code)
        history += [
            {"role": "user", "content": "What is its most famous landmark?"},
            {"role": "assistant", "content": reply.strip()[:100]},
        ]

    # T3
    if is_vl:
        body = build_text_turn("The first city I asked about — is it in Europe?", history)
        code, resp = post(f"{BASE}/v1/chat/completions", body)
        reply = record("T3 recall city", body, resp, code, expect_refs=["yes", "europe", "france", "paris"])
    else:
        body = build_text_turn("Approximately how tall is it?", history)
        code, resp = post(f"{BASE}/v1/chat/completions", body)
        reply = record("T3 how tall", body, resp, code, expect_refs=["meter", "feet", "ft", "tall", "300", "330", " m "])
    history += [
        {"role": "user", "content": body["messages"][-1]["content"] if isinstance(body["messages"][-1]["content"], str) else "..."},
        {"role": "assistant", "content": reply.strip()[:100]},
    ]

    # T4
    if is_vl:
        body = build_image_turn("And this image, what color is it?", blue, history)
        code, resp = post(f"{BASE}/v1/chat/completions", body)
        reply = record("T4 vl:blue", body, resp, code, expect_refs=["blue"])
        history += [
            {"role": "user", "content": "And this image, what color is it?"},
            {"role": "assistant", "content": reply.strip()[:100]},
        ]
    else:
        body = build_text_turn("Who designed it?", history)
        code, resp = post(f"{BASE}/v1/chat/completions", body)
        reply = record("T4 who designed", body, resp, code)
        history += [
            {"role": "user", "content": "Who designed it?"},
            {"role": "assistant", "content": reply.strip()[:100]},
        ]

    # T5
    if is_vl:
        body = build_text_turn("What two colors have I shown you so far?", history)
        code, resp = post(f"{BASE}/v1/chat/completions", body)
        reply = record("T5 recall colors", body, resp, code, expect_refs=["red", "blue"])
    else:
        body = build_text_turn("Summarize what we've discussed about the capital.", history)
        code, resp = post(f"{BASE}/v1/chat/completions", body)
        reply = record("T5 summarize", body, resp, code, expect_refs=["paris", "france"])
    history += [
        {"role": "user", "content": body["messages"][-1]["content"] if isinstance(body["messages"][-1]["content"], str) else "..."},
        {"role": "assistant", "content": reply.strip()[:100]},
    ]

    # T6
    body = build_text_turn("What was the very first question I asked you in this conversation?", history)
    code, resp = post(f"{BASE}/v1/chat/completions", body)
    reply = record("T6 first-q", body, resp, code, expect_refs=["capital", "france", "paris"])
    history += [
        {"role": "user", "content": body["messages"][-1]["content"]},
        {"role": "assistant", "content": reply.strip()[:100]},
    ]

    mid_hits, _ = cache_hits()

    # T7 bypass
    import uuid
    body = build_text_turn(
        "What is the capital of France? Answer in one word.",
        [],
        cache_salt=f"xmat-{row['id']}-{uuid.uuid4().hex[:6]}",
    )
    t_before = time.time()
    code, resp = post(f"{BASE}/v1/chat/completions", body)
    t_elapsed = time.time() - t_before
    _ = record("T7 bypass", body, resp, code)
    post_hits, _ = cache_hits()

    results["final"] = {
        "pre_session_hits": pre_hits,
        "mid_session_hits": mid_hits,
        "post_bypass_hits": post_hits,
        "hits_grew_during_session": mid_hits > pre_hits,
        "bypass_added_hits": post_hits - mid_hits,
        "cache_backend": backend_name,
        "load_seconds": round(load_s, 0),
        "t7_elapsed_sec": round(t_elapsed, 2),
    }
    print(
        f"  hits: pre={pre_hits} mid={mid_hits} post={post_hits}"
        f"  backend={backend_name}  load={load_s:.0f}s  t7={t_elapsed:.2f}s"
    )

    # Extra assertion for cacheable rows: mid must exceed pre. Mixed-attention
    # models are no longer exempt; RotatingKVCache preservation is part of the
    # production cache contract.
    mixed_attention = server_log_had_mixed_attention(log)
    results["final"]["mixed_attention_detected"] = mixed_attention
    if row["backend"] in CACHEABLE and not (mid_hits > pre_hits):
        results["turns"].append({
            "name": "cache_growth",
            "status": "FAIL",
            "fail": (
                f"hits did not grow ({pre_hits} -> {mid_hits}) on cacheable "
                f"backend {row['backend']}"
                + (" (mixed-attention diagnostic was present)" if mixed_attention else "")
            ),
            "reply": "",
        })

    proc.terminate()
    time.sleep(2)
    return results


# --- Main --------------------------------------------------------------


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--row", help="Comma-separated row IDs to run (e.g. X4,X8)")
    ap.add_argument("--fast", action="store_true", help="Odd-numbered rows only")
    ap.add_argument("--skip-slow", action="store_true", help="Drop rows tagged slow=True")
    ap.add_argument("--dry-run", action="store_true", help="Print selected rows and exit")
    ap.add_argument("--out", default=os.path.join(TEST_DIR, "results_cross_matrix.json"))
    args = ap.parse_args()

    rows = list(ROWS)
    if args.row:
        wanted = set(s.strip() for s in args.row.split(","))
        rows = [r for r in rows if r["id"] in wanted]
    if args.fast:
        def idx(r):
            return int(r["id"][1:])
        rows = [r for r in rows if idx(r) % 2 == 1]
    if args.skip_slow:
        rows = [r for r in rows if not r.get("slow")]

    print(f"\nCross-matrix rows selected: {len(rows)} / {len(ROWS)}")
    for r in rows:
        missing = "" if os.path.isdir(r["path"]) else "  [MISSING]"
        slow = "  [SLOW]" if r.get("slow") else ""
        print(f"  {r['id']}  {r['label']}  (backend={r['backend']} kv={r['kv_quant']}){slow}{missing}")

    if args.dry_run:
        print("\n--dry-run: exiting before launching any server")
        return

    all_results = []
    for r in rows:
        try:
            res = run_row(r)
        except Exception as e:
            print(f"  ❌ CRASHED {r['id']}: {e}")
            res = {"id": r["id"], "label": r["label"], "error": str(e)}
        all_results.append(res)
        subprocess.run(["pkill", "-f", "vmlx_engine.cli"], capture_output=True)
        time.sleep(3)

    with open(args.out, "w") as f:
        json.dump(all_results, f, indent=2)

    # Summary
    print(f"\n\n{'='*78}\nCROSS-MATRIX RESULTS\n{'='*78}")
    passed = failed = skipped = 0
    for r in all_results:
        if r.get("skipped"):
            print(f"⚠️  {r['id']}  {r['label']}  SKIPPED")
            skipped += 1
            continue
        if r.get("error"):
            print(f"❌ {r['id']}  {r['label']}  CRASHED: {r['error'][:80]}")
            failed += 1
            continue
        turns = r.get("turns", [])
        n_ok = sum(1 for t in turns if t.get("status") == "OK")
        n_tot = len(turns)
        pass_all = n_ok == n_tot and n_tot >= 7
        icon = "✅" if pass_all else "❌"
        print(f"{icon} {r['id']}  {r['label']}  {n_ok}/{n_tot}")
        if not pass_all:
            for t in turns:
                if t.get("status") != "OK":
                    print(f"    ✗ {t.get('name')}  {t.get('fail') or ''}")
            failed += 1
        else:
            passed += 1

    print(f"\n{passed} passed, {failed} failed, {skipped} skipped of {len(rows)}")
    print(f"results → {args.out}")

    if failed > 0:
        print("\n🚨 CROSS-MATRIX FAIL — block release")
        sys.exit(1)
    print("\n✅ CROSS-MATRIX GREEN")


if __name__ == "__main__":
    main()
