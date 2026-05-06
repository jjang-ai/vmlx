# Codex 2026-05-05 jang-tools fixes

5 of 8 Codex items landed in `/Users/eric/jang/jang-tools/`. The remaining 3 are out of session scope (Swift, Ling Lightning attention port, Kimi expert-paging).

## #1 — `jang_tools.verify_jangtq_prestacked` module (NEW)

**File:** `jang_tools/verify_jangtq_prestacked.py`
**Spec ref:** `research/JANGTQ-PRESTACK-SPEC.md`

Hard-fails on:
- per-expert TQ keys (regex `\.experts\.\d+\.(?:w[123]|gate_proj|up_proj|down_proj)\.tq_(?:packed|norms|bits)`)
- missing tq_packed/tq_norms/tq_bits triplets per (prefix, projection)
- wrong ndim (packed 3D, norms 2D, bits 1D)
- wrong leading dim (must equal `n_routed_experts` from config.json)
- sidecar pollution (`jangtq_stacked.safetensors`, `jangtq_stacked.json` in bundle dir)
- module replacement count mismatch (when `--check-load` is set)

**CLI:** `python -m jang_tools.verify_jangtq_prestacked /path/to/bundle [--check-load]`
**Programmatic:** `from jang_tools.verify_jangtq_prestacked import verify, VerificationError; verify(p, check_load=False)`

## #3 — Hard-fail hydrate skips in `load_jangtq.py:977`

**File:** `jang_tools/load_jangtq.py` `_hydrate_jangtq_model`

Before: silently `Skip (not in model): {base}` then continue. A model could load with most TQ tensors silently fp16-fallback'd.

After: collect missing required modules into `_missed_required` list, raise `RuntimeError` at end of TQ replacement loop with first 8 names. Allowlist regexes match `\.mtp\.`, `\.eh_proj\.`, `\.shared_head\.`, `\.embed_tokens\.tq_` (documented non-inference paths). Override: `VMLX_JANGTQ_ALLOW_HYDRATE_SKIPS=1` for emergency loads.

## #4 — Sync `jangrt/switchglu_decode.py` with canonical loader

**File:** `jang_tools/jangrt/switchglu_decode.py`

Before:
- Imports wrapped in `try/except` and swallowed (silent install failure)
- `_get_compiled_decode(in_f, out_f, bits, K)` — no dp_bits, no swiglu_limit
- `gather_dn = make_gather_tq_decode_per_row(out_f, in_f, bits, K)` used same bits as gate
- Slow path called `fused_gate_up_swiglu_matmul(..., bits=gp.bits)` with no swiglu_limit

After:
- Imports raise (no silent failure)
- `_get_compiled_decode(in_f, out_f, bits, K, swiglu_limit=0.0, dp_bits=None)`
- Cache key `(in_f, out_f, bits, dp_bits, K, limit_milli)`
- `gather_dn = make_gather_tq_decode_per_row(out_f, in_f, dp_bits, K)`
- Pulls `_swiglu_limit = activation.swiglu_limit` from the live SwitchGLU
- Slow path passes `swiglu_limit=_swiglu_limit`
- Fast path passes both `swiglu_limit=` and `dp_bits=dp.bits`

## #8 — Normalize converter metadata (4 converters)

| Converter | Before | After |
|---|---|---|
| `convert_glm51_jangtq_2l.py:276` | `"bits": 2` (routed) | `"bits": 8` + `weight_format: "mxtq"` |
| `convert_mxtq.py:191` | `"bits": 2` (routed) | `"bits": 8` + `weight_format: "mxtq"` |
| `kimi_prune/convert_kimi_jangtq.py:297` | `"bits": 2 if profile != "3L" else 3` | `"bits": 8` + `weight_format: "mxtq"` |
| `convert_nemotron_jangtq.py:366` | `"bits": EXPERT_BITS` (routed) | `"bits": 8` |

`config["quantization"]["bits"]` is the affine control-plane fallback (attention/embed/head/shared_expert) — NOT routed expert bits. Routed bits live in `jang_config.mxtq_bits`. Wrong value caused mlx_lm to apply 2-bit dequantization to 8-bit-stored weights for control-plane → garbage output (per `JANG-ISSUES.md`).

## #2 — Converters now post-process via rebundle to be prestack-compliant

**File:** `jang_tools/rebundle_jangtq_stacked.py` — added programmatic `rebundle(src, dst, ...)` API alongside the existing CLI `main()`.

**Files:** `convert_minimax_jangtq.py`, `convert_ling_jangtq.py`, `convert_nemotron_jangtq.py`, `kimi_prune/convert_kimi_jangtq.py`

After the per-expert quantization loop completes, each converter:
1. Calls `rebundle(OUT, OUT.parent / (OUT.name + ".prestack_tmp"))`
2. Renames the original to `.per_expert_backup`
3. Renames the prestack tmp to OUT
4. Removes the backup

Default ON. `--no-prestack` opts out for debugging the raw per-expert output.

Result: SHIPPED bundles match the spec (no `.experts.<E>.<proj>.tq_*` keys, no sidecar pollution).

## NOT done (out of scope)

- **#5 Swift jang-runtime fixes** — explicitly out of this session's Python-only scope.
- **#6 Ling bailing_hybrid Swift port** — Swift work.
- **#7 Kimi/JangPress** — user said "don't bother" + already complex multi-day expert-paging work.

## Bonus — Gemma 4 31B JANG_4M-CRACK shape mismatch (HF discussion #25)

**File:** `vmlx_engine/utils/quant_shape_inference.py` `infer_quant_overrides_for_bundle`

User report: `dealignai/Gemma-4-31B-JANG_4M-CRACK` fails to load with
```
ValueError: Expected shape (8192, 672) but received shape (8192, 1344) for
parameter language_model.model.layers.0.self_attn.q_proj.weight
```

Root cause (per [HF discussion #25](https://huggingface.co/dealignai/Gemma-4-31B-JANG_4M-CRACK/discussions/25)):
1. The CRACK bundle's `config.json` `quantization` block has per-layer overrides keyed under HF naming `model.language_model.X`, but mlx_lm's `class_predicate` matches MLX module paths AFTER `sanitize()` which removes the `model.` prefix → predicate looks up `language_model.model.X` and finds nothing → falls back to global 8-bit default while the layer is actually 4-bit → shape doubles (672 vs 1344 = 8-bit assumed for a 4-bit layer).
2. Plus, 181 4-bit JANG layers were missing from the override map entirely.

**Fix:** `infer_quant_overrides_for_bundle` now writes each inferred override under BOTH the HF-prefixed name AND the MLX-sanitized name. Whichever path the predicate looks up wins. Conservative: doesn't overwrite existing entries.

The shape-inference logic was already correct (computed bits from `weight.shape[-1] * 32 / scales.shape[-1]`); this commit fixes the key naming so the inferred overrides actually take effect on VLM wrapper class hierarchies.

## Verifier usage example

```bash
# Build a bundle:
python -m jang_tools.convert_minimax_jangtq /path/to/src /tmp/minimax_jangtq2 JANGTQ2

# Verify (raises VerificationError on hard-fail):
python -m jang_tools.verify_jangtq_prestacked /tmp/minimax_jangtq2

# Verify + load test:
python -m jang_tools.verify_jangtq_prestacked /tmp/minimax_jangtq2 --check-load
```
