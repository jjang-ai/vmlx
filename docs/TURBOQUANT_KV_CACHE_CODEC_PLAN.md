# TurboQuant KV Cache Codec — Design Plan (Codex 2026-05-06)

## Why this exists

After v1.5.23, Codex live-probed `/v1/responses` on the installed app. Two findings:

1. **Hard crash on corrupt cache restore.** A stale or malformed L2 disk block could
   produce `[metal::malloc] Attempting to allocate 468462801024 bytes`. Root cause:
   `prefix_cache.reconstruct_cache` did `cache.state = state` with **zero shape
   validation**, then the next forward pass requested a corrupted-shape allocation.

   *Mitigation shipped (CR4):* `vmlx_engine/cache_record_validator.py` +
   `tests/test_cache_record_validator.py`. Hard-rejects records that violate per-tensor
   byte cap (4 GB), per-record byte cap (16 GB), per-dim cap (256K), layer-count
   match, or known tag set. Wired into `block_disk_store.read_block` and
   `prefix_cache.reconstruct_cache` (BlockAware). On any failure: log + cache miss.

2. **TurboQuant KV is not actually default.** Earlier docs claimed it was. Codex's
   live `/health` check showed `turboquant_kv_cache.enabled=false`. The DSV4
   `jang_config.json` has no `turboquant` block. CLI default is
   `--kv-cache-quantization none`.

   *Mitigation shipped (B5):* `DSV4_FIX_NUANCES.md` corrected to "OPT-IN, not
   default."

This doc is the design for **doing it for real** — a typed cache codec, not a flag.

## Contract (Codex's seven items)

Quoting the contract verbatim:

> 1. Prefix/paged/L2 disk cache must store a typed record per cache family:
>    - plain KVCache
>    - RotatingKVCache/SWA
>    - DeepseekV4Cache composite: SWA local + CSA/HSA pool state
>    - TurboQuantKVCache encoded form
>
> 2. DSV4 cache store must use a clean prompt-boundary snapshot, never
>    post-generation cache. Store N-1 prompt key if that is the current contract,
>    but validate it.
>
> 3. For DSV4 composite cache:
>    - quantize/encode only the SWA local KV if that is the intended safe path
>    - keep CSA/HSA compressed pool buffers native unless a real codec exists
>    - terminal block must contain the full composite state
>    - non-terminal blocks with pending markers must never be restored as usable cache
>
> 4. Restore path must validate before allocating: schema version, model/cache hash,
>    layer count == expected, block size/token count sane, local KV dims sane,
>    offsets ≤ max allowed, no decoded tensor may request hundreds of GB.
>    If validation fails: log and treat as cache miss. Never allocate.
>
> 5. TQ encode/decode must round-trip in tests:
>    - fresh prefill output == restored prefix output at temp=0
>    - paged L1 hit
>    - L2 disk hit after restart
>    - multi-turn recall stays coherent
>    - cache_salt bypass and cached path produce equivalent output
>
> 6. Make TurboQuant KV default only after `/health` reports it actually enabled.
>
> 7. Hard regression for the exact crash: restore malformed/bogus TQ/L2 cache
>    metadata and assert it misses instead of trying to allocate 468462801024 bytes.

## Status

### 2026-05-07 DSV4/ZAYA correction

This document's generic TurboQuant KV plan does **not** mean every model family
should receive generic TQ-KV by default.

- DeepSeek-V4 Flash uses a native heterogeneous cache: SWA state plus CSA/HCA
  compressed entries plus incomplete-tail state/recompute. Its production path
  must keep generic TurboQuant KV disabled and expose the native DSV4 pool
  compression status separately in `/health` and `/v1/cache/stats`.
- ZAYA uses CCA state (`KVCache` plus `conv_state`/`prev_hs`). Its
  prefix/paged/L2/runtime TQ-KV tiers stay disabled until a typed CCA restore
  record exists and live restore gates pass.
- Default-on generic TQ-KV remains scoped to compatible plain-KV families with
  live encode/decode/cache-hit proof. Hybrid families require typed partial
  codecs, not a generic wrapper.

### 2026-05-07 ZAYA typed CCA restore

ZAYA remains live-gated, but the Python engine now has a real typed CCA cache
record instead of treating ZAYA as a generic `CacheList`:

- ZAYA odd MoE layers are extracted and serialized as explicit `no_state`
  records so restore keeps the full 80-layer CCA/MoE schedule aligned.
- Prefix/paged/L2 store emits typed `zaya_cca` records. Each record carries the
  block-sliceable standard KV subcache plus terminal prompt-boundary CCA
  `conv_state`/`prev_hs`.
- Restore rejects ZAYA chains that have only KV pages and no terminal CCA state.
  That turns an unsafe partial hit into a clean miss.
- L2 block disk serialization/deserialization round-trips `zaya_cca_v1`.
- Cumulative state trees are copied on store and restore; an in-memory prefix
  hit cannot borrow mutable `conv_state`/`prev_hs` leaves from the request that
  created the block.
- `/health`, `/v1/cache/stats`, and `/v1/models/{id}/capabilities` now expose
  native cache status as `family=zaya`, `cache_type=typed_cca`,
  `schema=zaya_cca_v1`, with generic TurboQuant KV explicitly false.
- Local contract coverage now proves prompt snapshot -> restore -> continuation
  logits for the small ZAYA runtime, plus validator/prefix/block-disk and
  hybrid/DSV4/SSM cache suites.

This is not a production sign-off for all ZAYA bundles. The remaining release
gate is live full-model ZAYA cache-hit, L2-restart, multi-turn, and tool-row
coverage with `zaya_cca_v1` visible in health/cache stats.

| # | Item | Status |
|---|---|---|
| 1 | Typed records per family (`kv`, `rotating_kv`, `quantized_kv`, `deepseek_v4`, `cumulative`, `cache_list`, `no_state`, `zaya_cca`) | EXISTS. Add `turboquant_kv` after codec tests land. |
| 2 | DSV4 N-1 clean prompt-boundary snapshot | EXISTS (`scheduler.py` "DSV4 prefix cache store using clean prompt-boundary snapshot"). Add validator on the snapshot itself. |
| 3 | DSV4 composite: SWA quantize, CSA/HSA native, terminal-only restore | EXISTS partially. CSA/HSA stay native today. Pending markers rejected at fetch (`prefix_cache.py:1172`). |
| 4 | Restore-path validation, no-hundreds-of-GB allocation | **DONE in this commit (CR4).** |
| 5 | Round-trip tests | TBD — depends on codec design below. |
| 6 | Default-on after `/health` truthful | **NOT YET.** Default stays opt-in. |
| 7 | Regression for the 468 GB crash | **DONE in this commit (`tests/test_cache_record_validator.py`).** |

## Codec design (items 1, 5, 6)

### TurboQuantKVCache record shape

Add a new tag string and writer/reader pair:

```
("turboquant_kv",
   keys_packed,            # int8 array, shape (..., seq, ceil(d / pack_factor))
   keys_scales,            # float16, shape (..., seq, num_groups, 1)
   keys_zeros,             # float16, shape (..., seq, num_groups, 1)
   values_packed,
   values_scales,
   values_zeros,
   meta_dict,              # group_size, bits, codec_version, dtype_orig
)
```

Why this shape: keys/values quantize independently; per-token group quantization
preserves the linear-attention property (attention can dequantize per token without
materializing the full cache). Group size 64, bits ∈ {2, 4, 8}, default 4.

### Encoder

`vmlx_engine/cache_codecs/turboquant_kv.py`:
```python
def encode(keys: mx.array, values: mx.array, *, bits: int = 4, group_size: int = 64) -> dict:
    ...
def decode(record: dict) -> tuple[mx.array, mx.array]:
    ...
```

The encoder is a thin wrapper around `mx.fast.affine_quantize` (already used by
`QuantizedKVCache`); the wrapper's job is **typed serialization**: emit a record the
disk store can write and the validator can check.

### Disk-store integration

1. `block_disk_store._serialize_block` learns to flatten `turboquant_kv` records
   into the same `__metadata__` + `layer_*_*` flat scheme used today.
2. `block_disk_store._deserialize_block` learns to reconstruct the tag.
3. `cache_record_validator.validate_cache_record` validates `turboquant_kv` shapes
   (already extensible — add `elif tag == "turboquant_kv":` branch with cap checks).

### DSV4 composite cache integration

DSV4 has **mixed cache families per layer**:

- 5 layers: `KVCache` (plain attention)
- 28 layers: `DeepseekV4Cache` (SWA local + CSA pool + HSA pool)
- 10 layers: routing/MTP (excluded from prefix cache)

Current production policy for each `DeepseekV4Cache` layer:
- SWA local (RotatingKVCache) -> native DSV4 state or native DSV4 pool codec.
- CSA pool buffers (compressed latents) -> keep native (`deepseek_v4` tag).
- HCA pool buffers (heavily compressed latents; older local notes sometimes
  say `HSA`) -> keep native (`deepseek_v4` tag).

Do not wrap DSV4 in the generic `turboquant_kv` record unless a later
DSV4-specific codec proves that it can preserve the official heterogeneous
cache contract, including incomplete-tail recomputation/checkpoint policy.

This matches Codex's item #3 exactly.

### Round-trip tests (item 5)

Add `tests/test_turboquant_kv_codec.py`:

1. `test_encode_decode_roundtrip_close` — encode random KV, decode, compare to
   original at relative-tolerance 1e-2 (4-bit) / 1e-3 (8-bit).
2. `test_paged_l1_hit_equivalence` — prefill prompt, generate 50 tokens at temp=0,
   capture; reset; warm prefix from cache; generate 50 tokens at temp=0; assert
   token-id sequences equal.
3. `test_l2_disk_hit_after_restart` — same as #2 but `BlockDiskStore` is closed +
   reopened between prefill and warm.
4. `test_multi_turn_coherence` — 3 turns; verify turn 3's reasoning content
   references turn 2's content.
5. `test_cache_salt_bypass_equivalence` — same prompt twice, once with
   `cache_salt="X"`, once without; compare token sequences.
6. `test_dsv4_partial_swa_only` — verify CSA/HSA pool buffers in the restored DSV4
   cache are byte-identical to the pre-encode buffers (only SWA was quantized).

### CLI/health gating (item 6)

The current `--kv-cache-quantization {none,q4,q8}` flag is the right knob. Codex's
ask is that **default-on must be reflected truthfully in `/health`**:

```python
# server.py /health response
"turboquant_kv_cache": {
    "enabled": _kv_cache_bits != 0,        # currently False unless --kv-cache-quantization q4|q8
    "bits": _kv_cache_bits,
    "scope": ["swa_local"] if _is_dsv4 else ["all_kv_layers"],
}
```

Default flip happens **after**:
- All 6 round-trip tests pass
- The validator has been live for ≥1 release with zero false-rejections
- `_is_dsv4` carve-out demonstrably preserves CSA/HSA bit-exactness

Until then, the user opts in via CLI flag or per-bundle `jang_config["turboquant"]`.

## Out-of-scope for this commit

- Implementing the TurboQuantKVCache encoder/decoder kernels (multi-day work).
- Ling/MiniMax-specific cache families (separate codec design once DSV4 lands).
- Default-on flip (item 6) — gated on test coverage, not this commit.

## What this commit DOES ship

1. **`vmlx_engine/cache_record_validator.py`** — the typed validator with hard
   per-tensor / per-record / per-dim caps. Catches the 468 GB malformed-cache
   crash class. Codex contract item #4.
2. **`tests/test_cache_record_validator.py`** — 10 regression tests including
   `test_468gb_corruption_rejected` (item #7 hard regression).
3. **Wired into both restore paths** (`block_disk_store.read_block` +
   `prefix_cache.reconstruct_cache`) with model-derived `expected_num_layers`.
4. **This design doc** capturing the full plan for items #1, #2, #3, #5, #6.
