# Flash MoE — SSD Expert Streaming

Port of `vmlx_engine/flash_moe_config.py` + `utils/flash_moe_loader.py` +
`models/flash_moe_integration.py` (~1150 LOC Python) into self-contained
Swift. Enables massive MoE models (35B–397B) to run on Macs with limited
RAM by keeping only a hot slot-bank of expert weights in memory and
streaming the rest from SSD on demand.

## Build order

### Phase 1 — foundation (LANDED 2026-04-13)
- `FlashMoEConfig.swift` — config struct, `from_dict`/`to_dict` parity
  with Python for CLI + settings + API round-trip.
- `ExpertIndex.swift` — safetensors header scanner. Reads only the
  first `8 + headerSize` bytes per file, extracts every
  `switch_mlp.*` tensor's absolute byte offset + shape + dtype, and
  normalizes Nemotron `fc1`/`fc2` → `up_proj`/`down_proj`. Handles
  all 4 key-pattern families (Nemotron, MiniMax, Gemma 4, Qwen/Mistral).
- `SlotBankCache.swift` — O(1) thread-safe LRU cache keyed by
  `"<layer>:<expert>"`. Doubly-linked list + hash map. Lock via
  `OSAllocatedUnfairLock`. Full `SlotBankStats` reporting.
- `ExpertWeightSet.swift` — data bundle for all tensors belonging to
  one expert in one layer. Class (not struct) to avoid copying MLX
  storage on cache promotion.
- `FlashMoEExpertLoader.swift` — on-demand expert loader with pread
  + parallel I/O queue. Reads raw bytes at precomputed offsets into
  typed `MLXArray`s. Handles F16/BF16/F32/U32/I32/U8 dtypes. Reports
  cache hit-rate + disk load latency via `stats()`.

**Phase 1 tests:** `Tests/vMLXTests/FlashMoETests.swift` — 14 tests
covering config round-trip, key-pattern matching (all 5 families
including the non-expert rejection path), and LRU semantics (promote,
evict, replace, clear).

### Phase 2a — layer-swap shims (LANDED 2026-04-13)
- `FlashMoEApplyExpert.swift` — Shared per-expert matmul helper,
  port of Python `_apply_expert_tensors`. `flashMoEApplyExpertTensors`
  runs gate/up/down projections for one expert with `quantizedMM`
  when scales are present, `matmul(weight.T)` for fp paths. Handles
  3-projection SwitchGLU and 2-projection SwitchMLP (Nemotron) via
  `FlashMoEActivation` enum. Includes `inferBitsFromShapes` for
  JANG mixed-precision.
- `FlashMoEBlock.swift` — Text-path MoE block replacement
  (Qwen/Mistral/MiniMax/Nemotron). `Module` subclass conforming to
  `UnaryLayer`. Injectable `router` closure so each family's native
  gate style works (softmax/sigmoid/pre-routed). Token-generation
  fast path (T=1) skips grouping; prefill path (T>1) groups tokens
  by expert per top-K slot for batched matmuls.
- `FlashMoESwitchGLUShim.swift` — Gemma 4 drop-in for
  `experts.switch_glu`. Matches `SwitchGLU.__call__(x, indices)` →
  `[..., K, H]` exactly so the sibling `Router` keeps handling
  routing. Scatter-adds per-slot outputs, stacks along K axis.
- `FlashMoEApply.swift` — Model-agnostic traversal. Defines
  `FlashMoEReplaceable` (model protocol) + `FlashMoELayer` (layer
  protocol) + `FlashMoELayout` enum (textPathSwitchGLU /
  textPathSwitchMLP / gemma4 / none). `FlashMoE.apply(to:loader:)`
  walks `model.flashMoELayers`, checks `loader.index.layers[layerIdx]`
  for coverage, and calls the layer's `replaceMoEBlock(with:)` or
  `replaceSwitchGLU(with:)` depending on layout. Returns
  `FlashMoEApplyResult` with per-layout counts for engine stats.

### Phase 2b — model-side conformance (NEXT SESSION)
- Make `Qwen3.swift` / `Qwen3Next.swift` / `MiniMaxText01.swift` /
  `NemotronH.swift` / `Mistral4.swift` / `Gemma4Text.swift` conform
  to `FlashMoEReplaceable` by exposing their decoder-layer list as
  `flashMoELayers`.
- Per-family `DecoderLayer.replaceMoEBlock(with:)` implementations
  that install the shim via `updateModule(key: "mlp", _)` (or the
  family's equivalent key), read the original gate's `num_experts_per_tok`
  + `routed_scaling_factor` into the shim, wire the `router` closure
  to call the original gate, and zero out the original expert
  projections to drop RAM.
- Gemma 4's `replaceSwitchGLU(with:)` for the sibling-router layout.
- Stats exposure through `Engine.cacheStats` so the CachePanel
  surfaces hit rate + disk load latency alongside paged cache.

### Phase 3 — engine integration (LAST)
- Honor `GlobalSettings.flashMoe` / `flashMoeSlotBank` /
  `flashMoePrefetch` / `flashMoeIoSplit` in `Engine.load`.
- CLI: `vmlxctl serve --flash-moe --flash-moe-slot-bank 256`.
- Temporal prefetch warm-up on first token (scan cached turns).
- CacheCoordinator hook so flash-moe stats land in
  `cacheStats()["flashMoe"]`.
- Mutually-exclusive checks with distributed and MCP tool use that
  could confuse the slot bank. (Smelt removed 2026-05-04 — see
  JangPress for cold-expert handling.)

## Key design parity with Python

- **Key patterns**: the four layouts match the Python regex exactly,
  including the Gemma 4 `language_model.layers.N.switch_mlp` path
  that must be matched before the generic `mlp.switch_mlp` path so
  it doesn't get absorbed.
- **Nemotron normalization**: `fc1` → `up_proj`, `fc2` → `down_proj`
  happens inside `matchExpertKey` so downstream code is uniform.
- **BF16 path**: raw bytes are read as `Float16` storage and then
  converted via `asType(.bfloat16)` — MLX preserves the 2-byte
  layout. Matches the Python `np.uint8` + `view(np.float16)` trick.
- **Slot key format**: `"<layer>:<expert>"` — same as Python for
  cross-logging interop.

## Non-goals

- Smelt removed entirely 2026-05-04 (Eric directive). JangPress is the cold-expert handler.
- Not porting speculative decoding (user directive — jang-spec
  replaces it).
- Not building a separate safetensors mmap layer — we use direct
  `FileHandle.seek` + `readData(ofLength:)`. pread-on-macOS is not
  trivially exposed without going through `Darwin`'s lower-level
  POSIX shims; for the slot bank pattern (a few MB per load) the
  overhead of `FileHandle` is negligible compared to the disk read.
