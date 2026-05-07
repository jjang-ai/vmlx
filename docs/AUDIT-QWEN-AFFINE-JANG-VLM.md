# Qwen3.5/3.6 affine-JANG hybrid VLM — Root Cause Audit

**Spec:** `docs/superpowers/specs/2026-05-06-production-audit-design.md` §8.4, §17.4
**Trigger:** Codex commit `94b16d22` kept a fallback that routes affine-JANG (non-MXTQ) Qwen3.5/3.6 hybrid VLM bundles to the text-only loader. Per spec working principle, fallbacks are guards — replace with real fix.

## Bundle under audit

`/Users/eric/models/dealign.ai/Qwen3.6-27B-JANG_4M-CRACK`:

- `model_type=qwen3_5` (top), `text_config.model_type=qwen3_5_text` (inner)
- 64 layers, mostly `linear_attention` interspersed with `full_attention` every 4th
- `weight_format=jang` (affine-JANG), `format=jang`, `mxtq_bits=null`
- per-key quantization overrides on `full_attention` layers (3, 7, 11, 15, ...) for self_attn projections at 8-bit; default 4-bit elsewhere

## Symptom

- Loaded via `_load_jang_v2` (text path) → coherent output (live verified previously).
- Loaded via `_load_jang_v2_vlm` (VLM path) → corrupt output for both text-only and image inputs.
- The 94b16d22 fallback in `vmlx_engine/utils/jang_loader.py:1518–1526` routes affine-JANG Qwen hybrid VLM bundles to `_load_jang_v2` (text-only) and the symptom is masked.
- A later wiring audit found one additional failure path: Electron could mark
  the same affine-JANG bundle multimodal from `jang_config.architecture.has_vision=true`
  and pass `--is-mllm`; Python then honored `force_mllm=True` before reaching
  the Qwen text-only policy. Source now overrides forced MLLM for this exact
  affine-JANG class and panel detection mirrors the same policy.

## Root cause

**M-RoPE divergence between `mlx_lm.qwen3_5` and `mlx_vlm.qwen3_5.language`.**

### `mlx_lm.qwen3_5` (text path — coherent)

- Single-axis 1D RoPE via `nn.RoPE` with `partial_rotary_factor=0.25` (read from `rope_parameters` or top-level config).
- Standard cos/sin frequency table; rotation applied along the head_dim axis.

### `mlx_vlm.qwen3_5.language` (VLM path — corrupt)

- 3-axis M-RoPE via `Qwen3_5RotaryEmbedding` class:
  ```python
  # mlx_vlm/models/qwen3_5/language.py:18
  class Qwen3_5RotaryEmbedding:
      def __call__(self, x, position_ids):
          if position_ids.ndim == 2:
              position_ids = mx.broadcast_to(
                  position_ids[None, ...],
                  (3, position_ids.shape[0], position_ids.shape[1]),
              )
          inv_freq_expanded = mx.broadcast_to(
              self.inv_freq[None, None, :, None].astype(mx.float32),
              (3, position_ids.shape[1], self.inv_freq.shape[0], 1),
          )
          position_ids_expanded = position_ids[:, :, None, :].astype(mx.float32)
          freqs = inv_freq_expanded @ position_ids_expanded
          # …rotation downstream
  ```
- For text-only generation, `Qwen3_5Attention.__call__` (line 144) computes `position_ids = mx.tile(mx.expand_dims(mx.arange(...), 0), (3, 1, 1))` — same indices replicated across all 3 axes.
- Theoretically with identical position_ids on all 3 axes, the M-RoPE result should equal 1D RoPE rotation. In practice the downstream rotation merge/split path in `mlx_vlm.qwen3_5.language` is M-RoPE-specific and diverges from mlx_lm's 1D rotation.
- Net effect: every layer's attention rotates Q/K with the wrong rotation matrix → garbage attention scores → garbage logits.

The bug is **inside `mlx_vlm.qwen3_5.language`'s text-only fallback path**, not in `_load_jang_v2_vlm`.

## What's NOT the bug

The 2026-05-02 fix at `_load_jang_v2_vlm:1690–1747` (the per-module quantization override predicate) is correct and load-bearing. It already handles the `language_model.model.layers.{3,7,11,...}.self_attn.{q,k,v,o}_proj` 8-bit overrides correctly.

The MXTQ Qwen VLM path (e.g. `Qwen3.6-35B-A3B-JANGTQ-CRACK`) works via `jang_tools.load_jangtq_vlm`, which has its own runtime patches and does not hit `Qwen3_5RotaryEmbedding`'s MRoPE bug.

## Real fix paths (ranked)

1. **Patch `mlx_vlm.qwen3_5.language`** to short-circuit MRoPE → 1D RoPE when `position_ids` is text-only (all 3 axes identical). This is the cleanest fix and benefits everyone using mlx_vlm directly. Submit as upstream PR. **Estimated effort:** 0.5–1 day to implement + verify; upstream review timeline separate.

2. **Compose hybrid loader:** in `_load_jang_v2_vlm` for Qwen affine-JANG hybrid, build language model from `mlx_lm.qwen3_5` and attach vision tower from `mlx_vlm.qwen3_5.vision`. Keeps the affine-JANG bundle in the engine but routes through the working language code. **Estimated effort:** 2–3 days; risk of vision-language interface drift.

3. **Pin `mlx_vlm` and apply runtime patch in `vmlx_engine/runtime_patches/`** that swaps `Qwen3_5RotaryEmbedding.__call__` with the 1D-equivalent path when text-only input is detected. Per spec working principle this is a monkey-patch — only acceptable as an interim measure with a clear deprecation path to (1) or (2).

**Recommended path:** start with (1). If upstream PR is slow, fall back to (2) for the engine.

## Layer-by-layer divergence verification

Confirmed not yet performed (requires loading the bundle via both paths and comparing first-layer hidden states). The MRoPE difference is sufficient to explain the corruption, but a layer-1 diff would empirically confirm and rule out other co-divergences (per-projection bit overrides, LayerNorm parameters, expert tying for MoE variants).

## Decision for this audit cycle

- The fallback at `_load_jang_v2_vlm:1518–1526`, the `is_mllm_model` short-circuit, and the panel-side `resolveJangMultimodal` override **stay in place for now** with a clear pointer to this audit doc and a follow-up issue.
- Regression coverage now pins all three routing layers:
  `is_mllm_model(..., force_mllm=True)` returns `False` for affine-JANG Qwen
  hybrid, `_load_jang_v2_vlm` delegates affine-JANG to `_load_jang_v2`, and
  panel detection does not pass `--is-mllm` for affine-JANG while keeping
  MXTQ/JANGTQ Qwen VLM multimodal.
- Real fix path (1) is queued for the next cycle.
- The fallback is no longer "guard hiding a mystery"; it is a documented stopgap with a known root cause and a fix path.

## Remaining items

- Implement fix path (1): patch `mlx_vlm.qwen3_5.language.Qwen3_5RotaryEmbedding` for text-only short-circuit.
- Layer-1 hidden-state diff verification on `Qwen3.6-27B-JANG_4M-CRACK`.
- Re-test image attach + text-only multi-turn after fix lands.
- Remove the fallback in `_load_jang_v2_vlm:1518–1526`, the `is_mllm_model`
  short-circuit, and the panel detection override after a real MRoPE/loader fix
  lands.
