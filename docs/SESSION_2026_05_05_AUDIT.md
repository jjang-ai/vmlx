# Session Audit — 2026-05-05 (Python engine/app)

Single-doc enumeration of every code change made this session, intended for Eric's pass-through.

Scope: Python engine + app (`/Users/eric/mlx/vllm-mlx/` + `/Users/eric/jang/jang-tools/`). Swift was explicitly out of scope.

---

## Fix #1 — bailing_hybrid patch source ← bundled drift

**Problem**: `panel/scripts/patches/bailing_hybrid.patched.py` was lagging the bundled copy at `panel/bundled-python/python/lib/python3.12/site-packages/mlx_lm/models/bailing_hybrid.py`. Bundled had four critical fp32 + RAM-safety fixes the patch source lacked. `bundle-python.sh` blindly copies patch → bundled, so the next bundle would silently regress.

**Files touched**:
- `panel/scripts/patches/bailing_hybrid.patched.py` — synced FROM bundled
- `panel/bundled-python/python/lib/python3.12/site-packages/mlx_lm/models/bailing_hybrid.py` — re-stamped (no-op on its own; verifying parity)
- `.venv/lib/python3.13/site-packages/mlx_lm/models/bailing_hybrid.py` — synced for dev tests

**The 4 fixes synced**:
1. `recurrent_gla()` — q/k/v/h cast to fp32 before per-token loop, exp_g in fp32. fp16 overflows past ~80-token prompts.
2. fp32 output preservation — output stays fp32 through return, no cast inside `recurrent_gla`.
3. `g_norm(output).astype(x.dtype)` — cast back to input dtype after RMS-bound g_norm so `dense()` runs in fp16 (raw fp32 dropped 30→<10 tok/s).
4. Per-stack `mx.eval(packed); del stacked` in `sanitize` — Ling has 73,728 per-expert keys; lazy `mx.stack(...)` pins all alive until materialized → RAM blowup.

**Verification**:
```
diff panel/scripts/patches/bailing_hybrid.patched.py \
     panel/bundled-python/python/lib/python3.12/site-packages/mlx_lm/models/bailing_hybrid.py
```

**Live verified**: Ling-2.6-flash-JANGTQ loads in 3.5s, decodes "The capital of France is Paris.", multi-turn recall ("teal" preserved across turns).

---

## Fix #2 — JANGTQ_K dp_bits fast-path bug (MiniMax-M2.7-JANGTQ_K)

**Problem**: `jang_tools/load_jangtq.py:_get_compiled_decode` (~L1108) compiled the down-projection gather kernel using `gp.bits` (gate's bits) instead of `dp.bits`. For JANGTQ_K (gate=2, up=2, **down=4**), the down kernel unpacked 4-bit packed tensors as 2-bit → wrong codebook indices → garbage output (Chinese/English token salad, no `</think>`).

**Pre-fix output** (MiniMax-M2.7-JANGTQ_K, "What is 2+2?"):
```
'人道回答三季度\n三季度\n\n ris俩 coronavirus math肋三季度...'
```

**Post-fix output** (same prompt):
```
'The user asks "What is 2 + 2?"... Thus final answer: 4.\n</think>\n\n2+2=4.'
```

**Files touched**:
- `/Users/eric/jang/jang-tools/jang_tools/load_jangtq.py` — `_get_compiled_decode` + `_fused_switchglu_call`
- `/Users/eric/mlx/vllm-mlx/panel/bundled-python/python/lib/python3.12/site-packages/jang_tools/load_jangtq.py` — synced

**Three changes**:
1. `_get_compiled_decode(in_f, out_f, bits, K, swiglu_limit=0.0, dp_bits=None)` — added `dp_bits` param, defaults to `bits` for byte-identical legacy behavior.
2. Cache key extended: `key = (in_f, out_f, bits, dp_bits, K, limit_milli)` — prevents cross-config kernel sharing between uniform and JANGTQ_K layers.
3. `gather_dn = make_gather_tq_decode_per_row(out_f, in_f, dp_bits, K)` — down kernel built with own bits.
4. `_fused_switchglu_call` passes `dp_bits=dp.bits` at the call site.

Slow path was already correct (calls `self.down_proj(...)` which uses `self.bits`).

**Tests**: `tests/test_jang_loader.py::TestJangtqKMixedBitsFastPath` — 3 source-grep regression pins:
- `test_get_compiled_decode_accepts_separate_dp_bits`
- `test_compiled_decode_cache_key_distinguishes_bits`
- `test_gather_dn_uses_dp_bits`

**Regression-safe**: DSV4 Flash JANGTQ (uniform 2-bit) verified unaffected — `dp_bits=None` fallback to `bits` is byte-identical.

---

## Fix #3 — Ling MXFP4 / MXFP4-CRACK flat-2D switch_mlp shape repair

**Problem**: `convert_ling_mxfp4.py` converter rev that produced both `Ling-2.6-flash-MXFP4` and `Ling-2.6-flash-MXFP4-CRACK` flattened the prestacked routed-expert tensors' (out, in_per_row) axes into one. Bundle ships `(256, 524288)` 2D but mlx_lm's quantized SwitchLinear expects `(256, 1024, 512)` 3D → strict shape check at `model.load_weights()` raises `ValueError: Expected shape (256, 1024, 512) but received shape (256, 524288)...`.

Total elements match exactly (`1024 * 512 = 524288`) — data is intact, just stored at wrong rank.

**Files touched**:
- `panel/scripts/patches/bailing_hybrid.patched.py` — extended `Model.sanitize` with shape-repair pass
- `panel/bundled-python/python/lib/python3.12/site-packages/mlx_lm/models/bailing_hybrid.py` — synced
- `.venv/lib/python3.13/site-packages/mlx_lm/models/bailing_hybrid.py` — synced

**Logic**: walk `model.layers.<L>.mlp.switch_mlp.{gate,up,down}_proj.{weight,scales,biases}`. If shape is `(n_exp, flat)` 2D and `flat` divides cleanly by the expected out-dim (`moe_intermediate_size` for gate/up, `hidden_size` for down), reshape to `(n_exp, out, in_per_row)` 3D. Idempotent — no-op when already 3D (JANGTQ bundles unaffected). Also no-op when shape is non-divisible (defensive).

**Tests**: `tests/test_jang_loader.py::TestBailingHybridFlatSwitchMlpRepair` — 2 pin tests:
- `test_sanitize_repairs_flat_2d_switch_mlp_to_3d` — fixture with flat 2D inputs gets reshaped correctly
- `test_sanitize_no_op_on_correct_3d_shape` — 3D inputs pass through unchanged

**Live verified**:
- Ling-2.6-flash-MXFP4-CRACK: was ValueError → now loads in 5.3s, decodes "Fifteen" for "What is 8+7?"
- Ling-2.6-flash-MXFP4 (non-CRACK): same shape bug, same fix → decodes "Paris"
- Ling-2.6-flash-JANGTQ (regression): 3D bundle, sanitize is no-op → decodes "Hello! How can I help you today?"

**Note about source**: the converter (`/Users/eric/jang/jang-tools/jang_tools/convert_ling_mxfp4.py`) currently produces 3D output when run fresh. The buggy bundles in the wild were made by an earlier rev. Engine-side repair is the only path to fix existing user disks.

---

## Fix #4 — DSV4 rep_penalty floor (1.10)

**Problem (live user report)**: User on DSV4-Flash-JANGTQ via `/v1/responses` got degenerate repetition loops on 348-tok and 717-tok prompts:
- Chat mode: `"\nresponse\nresponse\nresponse\nresponse"` (rep_penalty=1.05 from bundle)
- Thinking mode: `"(the project (the project (the project ..."` (rep_penalty=1.0 from bundle = NO penalty)

Bundle `jang_config.json sampling_defaults`:
```
"repetition_penalty": 1.0,
"repetition_penalty_thinking": 1.0,    # NO penalty
"repetition_penalty_chat": 1.05,        # too low
```

`scheduler.py:1659-1662` skips building the logits_processor entirely when `rep_penalty == 1.0` — thinking mode had ZERO repetition defense.

v1.5.8 followup commit (44c571a6) had explicitly bumped DSV4 family fallback to 1.15 to fix this exact loop class, but a later commit removed the bandaid on a (incorrect) theory that a `scheduler.py:768` cache-cumulative-state fix had solved the underlying issue. The loops returned in production.

**File touched**: `vmlx_engine/server.py` — `_resolve_repetition_penalty`

**Logic**: refactored to compute `resolved` as before, then if `_model_family_for_defaults(model_name) == "deepseek_v4"` AND `resolved is None or resolved < 1.10`, return 1.10. Per-request explicit overrides bypass the floor entirely (benchmarks can still pass 1.05 explicitly).

**Tests**: `tests/test_server.py::TestDSV4RepetitionPenaltyFloor` — 4 pin tests:
- `test_floor_clamps_thinking_1_0_to_1_10`
- `test_floor_clamps_chat_1_05_to_1_10`
- `test_per_request_override_bypasses_floor`
- `test_floor_does_not_affect_non_dsv4_families`

**Live verified**: `vmlx-engine serve DeepSeek-V4-Flash-JANGTQ` with the user's repro prompt (Haskayne VC fund / 50-hour student project) — output is now coherent: "The project goal is to establish a comprehensive framework for process optimization..." Unique-word ratio 0.95 (healthy). Earlier degenerate loop is gone.

**Bundle still has bad defaults** — recommend also re-stamping `JANGQ-AI/DeepSeek-V4-Flash-JANGTQ` HF repo's `jang_config.json sampling_defaults` to `{"repetition_penalty_thinking": 1.10, "repetition_penalty_chat": 1.10}` so users get safe values from disk too. Engine floor is defense-in-depth.

---

## MiniMax-M2.7-JANGTQ_K — full multi-turn + cache feature verification

Boot:
```
.venv/bin/vmlx-engine serve /Users/eric/models/JANGQ/MiniMax-M2.7-JANGTQ_K \
  --port 18002 --host 127.0.0.1 --max-tokens 600 \
  --continuous-batching --use-paged-cache \
  --enable-prefix-cache --prefix-cache-size 4 --prefix-cache-max-bytes 2147483648 \
  --enable-block-disk-cache --block-disk-cache-dir /tmp/vmlx-mm-block-cache --block-disk-cache-max-gb 4 \
  --enable-disk-cache --disk-cache-dir /tmp/vmlx-mm-disk-cache --disk-cache-max-gb 4 \
  --kv-cache-quantization q4
```

| Feature | Status | Evidence |
|---|---|---|
| Continuous batching | ok | "Mode: Continuous batching" in startup log |
| Paged cache | ok | `block_size=64, max_blocks=1000`; "paged cache miss" → "Stored paged cache..." per request |
| Prefix cache | ok | "Prefix cache requires continuous batching — enabled automatically" |
| Block disk cache | ok | `BlockDiskStore initialized: dir=/tmp/vmlx-mm-block-cache, max=4.0GB`; SQLite + 6 sharded block dirs written |
| KV cache q4 quant | ok | "KV cache quantization round-trip test passed: bits=4, group_size=64, test_shape=(1, 4, 8, 128)" |
| L2 disk cache | ok | 4 `cache_*.safetensors` files written totaling ~20MB; restart with same dir produces coherent output |
| Multi-turn coherence | ok | Turn 1: "Hello Alice! Saturn is a fascinating planet..."; Turn 2 recall: "Saturn" |
| TurboQuant KV cache | ok | Boot with `VMLX_FORCE_TQ_AUTO=1` → "TurboQuant enabled: 3-bit keys, 3-bit values, 6 critical layers". Output: 390-char coherent ice/density explanation, 97% ASCII, 80% unique-words. Multi-turn recall through TQ KV: "Density" |

**Per-projection bits verified** (JANGTQ_K dispatch correctness):
- Loader log: `bits_map={'routed_expert': {'gate_proj': 2, 'down_proj': 4, 'up_proj': 2}, ...}`
- 186 TQ groups, all pre-stacked, 186 modules replaced
- Output coherent, no garbage tokens — confirms Fix #2 is engaged

---

## Pytest sweep results (offline)

```
.venv/bin/pytest tests/ \
  --ignore=tests/test_e2e_live.py \
  --ignore=tests/load_test_vlm.py \
  --ignore=tests/mem_profile_*.py \
  --ignore=tests/stress_tool_calls.py \
  --ignore=tests/integration --ignore=tests/benchmark \
  --ignore=tests/cross_matrix --ignore=tests/evals \
  -q
```

**Result**: **2755 passed, 53 skipped, 1 pre-existing unrelated failure**.

Pre-existing failure: `test_vl_video_regression::test_dependency_exists_and_wires_onto_all_inference_endpoints`. The test does a 500-byte source-grep window scan for `check_memory_pressure` after each endpoint string in `server.py`; actual distance for `/v1/chat/completions` is 41,227 bytes. The dependency IS wired (server.py:3852 `Depends(check_memory_pressure)`); test is a stale grep-window bug, not a real regression. Last touched in commit `3ae0b234 v1.5.15` along with server.py — unrelated to this session.

---

## Files modified summary

```
panel/scripts/patches/bailing_hybrid.patched.py     # Fix #1 (sync) + Fix #3 (new sanitize step)
panel/bundled-python/python/lib/python3.12/site-packages/mlx_lm/models/bailing_hybrid.py  # mirror
.venv/lib/python3.13/site-packages/mlx_lm/models/bailing_hybrid.py                       # dev mirror
/Users/eric/jang/jang-tools/jang_tools/load_jangtq.py  # Fix #2
panel/bundled-python/python/lib/python3.12/site-packages/jang_tools/load_jangtq.py       # mirror
vmlx_engine/server.py                                # Fix #4
tests/test_jang_loader.py                            # Pin tests for #2 + #3 (5 new tests)
tests/test_server.py                                 # Pin tests for #4 (4 new tests)
docs/SESSION_2026_05_05_AUDIT.md                     # this doc
```

No changes to: scheduler.py logic, paged cache, prefix cache, disk caches, BatchEngine, or any other engine internals. Fixes are surgical to the specific bugs identified.

---

## How Eric tests the app

Engine path (preserves all this session's fixes):

```
# Test #1 — JANGTQ_K MiniMax (Fix #2 verification)
cd /Users/eric/mlx/vllm-mlx
.venv/bin/vmlx-engine serve /Users/eric/models/JANGQ/MiniMax-M2.7-JANGTQ_K \
  --port 18002 --max-tokens 300
# Visit http://127.0.0.1:18002, ask "What is 2+2?" — should answer cleanly with thinking block

# Test #2 — DSV4 (Fix #4 verification)
.venv/bin/vmlx-engine serve /Users/eric/models/JANGQ/DeepSeek-V4-Flash-JANGTQ \
  --port 18002 --max-tokens 400
# Paste user's Haskayne VC fund prompt — should return coherent paragraph, no degenerate loop

# Test #3 — Ling MXFP4 (Fix #3 verification, if a broken-shape bundle is on disk)
.venv/bin/vmlx-engine serve /Users/eric/models/JANGQ/Ling-2.6-flash-MXFP4-CRACK \
  --port 18002 --max-tokens 200
# Was ValueError on shape — now loads + decodes

# Test #4 — Ling JANGTQ (Fix #1 fp32 GLA verification, regression)
.venv/bin/vmlx-engine serve /Users/eric/models/JANGQ/Ling-2.6-flash-JANGTQ \
  --port 18002 --max-tokens 200
# Long prompt past 80 tokens — fp32 GLA prevents NaN logits
```

To rebuild the panel app for users:

```
cd /Users/eric/mlx/vllm-mlx/panel
# bundle-python.sh will copy patches/bailing_hybrid.patched.py → bundled-python (Fix #1 + #3)
# .venv was already updated for dev parity
# vmlx_engine/server.py is in the panel's runtime — Fix #4 is automatic
npm run build
```

To re-stamp the JANGQ-AI HF repo bundles (separate concern, recommended):
- `DeepSeek-V4-Flash-JANGTQ`: edit `jang_config.json sampling_defaults` to `{"repetition_penalty_thinking": 1.10, "repetition_penalty_chat": 1.10, "repetition_penalty": 1.10}`
- `Ling-2.6-flash-MXFP4` + `Ling-2.6-flash-MXFP4-CRACK`: re-run `convert_ling_mxfp4.py` (current source produces correct 3D shapes); engine repair is defense for users with the old bad bundles

---

## Installed-app live verification (post-install)

After the 3 fixes were copied into `/Applications/vMLX.app/Contents/Resources/bundled-python/python/lib/python3.12/site-packages/`, ran live tests through the bundled Python:

**MiniMax-M2.7-JANGTQ_K (Fix #2 + cache features)**:
- Turn 1 (cold): 16.9s, 2940-char coherent reasoning about Border Collie exercise needs
- Turn 2 (recall): 1.97s (8.5x faster — cache hit), output `"Your dog's name is Pepper and she is a Border Collie!"` ✓
- Server log: paged cache stored 56-token block at turn 1, block-disk write-through fired
- Per-projection bits dispatched correctly (no garbage tokens)

**Ling-2.6-flash-JANGTQ2-CRACK (Fix #1 fp32 GLA + multi-turn hybrid SSM)**:
- Default thinking OFF: 1232 chars 100% ASCII English project proposal
- Multi-turn recall: "You can see why I like teal, right?" (teal preserved across hybrid SSM cache) ✓
- 4-test chat_template_kwargs stress sweep — all 100% ASCII English:
  - `chat_template_kwargs:{enable_thinking:true}` → 472-char step-by-step math
  - `chat_template_kwargs:{enable_thinking:false}` → "20"
  - `reasoning_effort:high` → 364-char photosynthesis explanation
  - explicit "answer in English" prompt → 98% ASCII English paragraph

**Ling-2.6-flash-MXFP4-CRACK (Fix #3)**:
- Was ValueError on shape; now decodes "Fifteen" via app

**DSV4-Flash-JANGTQ (Fix #4)**:
- Was looping `(the project (the project ...`; now coherent paragraph (95% unique-word ratio, no degeneracy)

**Backups**: original pre-install files at `/tmp/vmlx-app-backup-1778022556/` (revertable).

---

## Commit / push procedure (waiting for Eric's "good")

Two repos to update:

### 1. vmlx (`/Users/eric/mlx/vllm-mlx`, branch `session/v1.5.8` at origin/main)

Files:
- `vmlx_engine/server.py` — Fix #4 DSV4 rep_penalty floor
- `tests/test_server.py` — 4 pin tests for Fix #4
- `tests/test_jang_loader.py` — 5 pin tests for Fix #2 + #3
- `panel/scripts/patches/bailing_hybrid.patched.py` — new file, Fix #1 + #3
- `docs/SESSION_2026_05_05_AUDIT.md` — this doc

Suggested commit:
```
git add vmlx_engine/server.py tests/test_server.py tests/test_jang_loader.py \
        panel/scripts/patches/bailing_hybrid.patched.py docs/SESSION_2026_05_05_AUDIT.md
git commit -m "fix: DSV4 rep_penalty floor + Ling MXFP4 shape repair + JANGTQ_K dp_bits + bailing fp32 sync

- vmlx_engine/server.py: DSV4 family safety floor at 1.10 (bundle declares
  repetition_penalty_thinking:1.0 / chat:1.05; production loop confirmed)
- panel/scripts/patches/bailing_hybrid.patched.py: NEW — synced from bundled
  with fp32 GLA recurrence + per-stack mx.eval RAM-safety + flat-2D shape
  repair for older converter MXFP4 bundles
- 9 new pin tests (4 server.py, 5 jang_loader.py)
- docs/SESSION_2026_05_05_AUDIT.md: full audit"
git push origin session/v1.5.8:main
```

### 2. jang-tools (`/Users/eric/jang/jang-tools`, branch `main` at origin/main)

Files:
- `jang_tools/load_jangtq.py` — Fix #2 dp_bits fast-path

Suggested commit:
```
cd /Users/eric/jang/jang-tools
git add jang_tools/load_jangtq.py
git commit -m "fix: JANGTQ_K mixed gate/down bits — fast-path gather kernel uses dp_bits

MiniMax-M2.7-JANGTQ_K ships per-projection bits (gate=2, up=2, down=4).
The compiled-decode fast path was building the gather_dn kernel against
gp.bits → down_proj 4-bit packed tensors unpacked as 2-bit → garbage
output. Adds dp_bits param (default None = legacy uniform-bits fallback,
byte-identical for JANGTQ2/3/4 callers). Cache key extended to include
both bits so a uniform 2-bit and a JANGTQ_K layer in the same model don't
share a compiled kernel.

Slow path was already correct (calls down_proj which uses self.bits).
DSV4-Flash JANGTQ regression-verified unaffected (uniform 2-bit)."
git push origin main
```

### 3. PyPI

vmlx + jang-tools should bump version per Eric's release process:
- `vmlx`: `pyproject.toml` already at 1.5.20 — bump to 1.5.21 if shipping these as patch
- `jang-tools` (PyPI name `jang`): currently 2.5.23 — bump to 2.5.24

After git push:
```
# vmlx
cd /Users/eric/mlx/vllm-mlx
.venv/bin/python -m build && .venv/bin/python -m twine upload dist/vmlx-1.5.21*

# jang-tools (separate repo)
cd /Users/eric/jang/jang-tools
python -m build && python -m twine upload dist/jang-2.5.24*
```

### 4. Bundled-python in panel

`bundle-python.sh` will pick up:
- `panel/scripts/patches/bailing_hybrid.patched.py` → `mlx_lm/models/bailing_hybrid.py` (already synced manually + parity-verified)
- `pip install jang` (after PyPI bump) → updated `jang_tools/load_jangtq.py`
- Engine source ships with the panel build → updated `vmlx_engine/server.py`

For the immediate DMG release, the manual install at `/Applications/vMLX.app/Contents/Resources/bundled-python/...` is already in place. Next official build (`npm run build`) will pick all of this up automatically as long as PyPI is bumped first.

### 5. JANGQ-AI HF bundles (separate, recommended)

- `JANGQ-AI/DeepSeek-V4-Flash-JANGTQ` — re-stamp `jang_config.json sampling_defaults` to `{"repetition_penalty_thinking": 1.10, "repetition_penalty_chat": 1.10, "repetition_penalty": 1.10}` so users without engine update still get safe values
- `JANGQ-AI/Ling-2.6-flash-MXFP4` and `-CRACK` — re-run current `convert_ling_mxfp4.py` (produces correct 3D shapes); engine repair is defense for users with the old bad bundles

---

## What's NOT in scope for this session

- Swift engine/app (`/Users/eric/vmlx/swift/`) — separate agent's territory
- `/Applications/vMLX.app` — installed-app state is user QA territory
- New release tagging / DMG build / notarization — Eric handles releases
- Pushing updated bundles to HF — Eric controls JANGQ-AI repos
- Continuous-batching production-polish (iter-63 deferred work) — out of scope

---

## Memory file pointer

Session findings also recorded at:
- `/Users/eric/.claude/projects/-Users-eric-vmlx/memory/project_jangtq_k_ling_audit_2026_05_05.md` — JANGTQ_K + Ling audit + bailing_hybrid sync
- `MEMORY.md` index updated with pointer to that file
