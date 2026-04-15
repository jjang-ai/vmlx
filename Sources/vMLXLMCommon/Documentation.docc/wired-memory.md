# Wired Memory and Weight Reservations

MLXLMCommon exposes wired memory policies and tickets to help coordinate a single process-wide
wired memory limit during inference. Policies decide whether work should be admitted and what
wired limit is needed; tickets represent active or reserved memory usage and are registered
with the `WiredMemoryManager`.

For a full system overview (manager, policies, tickets, hysteresis, and usage patterns), see the
MLX wired memory article in the upstream `mlx-swift` repository:
<https://github.com/ml-explore/mlx-swift/blob/main/Source/MLX/Documentation.docc/Articles/wired-memory.md>

This article focuses on **estimating model weight bytes** so you can create a **weight reservation
(ticket)** that reflects the actual cost of keeping weights resident.

## Measuring weight bytes at runtime

If you can afford to load the model, the most accurate approach is to sum `nbytes` across all
parameter arrays. Loading a model already materializes weights, so you get a reliable number
without running any extra inference.

```swift
let context = try await LLMModelFactory.shared.load(configuration: config)
let weightBytes = context.model
    .parameters()
    .flattened()
    .reduce(0) { $0 + $1.1.nbytes }
```

You can optionally sanity check with `Memory.snapshot()` before/after load. In practice, the
difference between the sum of `nbytes` and MLX active memory has been very small in our tests.

## Avoiding load: estimate from tensor files

If you want a **no-load estimate**, sum the tensor file sizes on disk (for example, all
`.safetensors` shards in the model directory). This is fast and avoids allocating the model,
but it includes file metadata and may slightly exceed the in-memory representation.

```swift
let tensorExtensions: Set<String> = ["safetensors", "bin", "gguf"]
let sizes = fileSizes(in: modelURL, tensorExtensions: tensorExtensions)
let estimatedBytes = sizes.tensorTotalBytes
```

### What affects the difference?

- **File metadata/headers**: safetensors includes a JSON header; shard totals usually exceed
  `nbytes` by a small amount.
- **Allocator alignment/overhead**: MLX active memory may be a tiny bit larger than the logical
  `nbytes` sum.
- **Format differences**: compressed or container formats can cause larger gaps between on-disk
  size and in-memory representation.

## Observed deltas (local measurements)

These measurements were taken on two local models using the runtime `nbytes` sum, tensor file
sizes, and MLX `Memory.snapshot()` right after load (no inference):

| Model | Sum of `nbytes` | Tensor file total | Active memory after load | Notes |
| --- | ---: | ---: | ---: | --- |
| Qwen3-4B-Sky-High-Hermes-4bit | 2,262,535,712 | 2,262,637,937 | 2,264,337,376 | +102,225 bytes vs files; +1,801,664 bytes vs active |
| Qwen3-Next-80B-A3B-Instruct-MLX-4bit | 44,844,060,160 | 44,844,286,608 | 44,844,101,616 | +226,448 bytes vs files; +41,456 bytes vs active |

These examples suggest that **`nbytes` is a reliable basis** for a reservation ticket when you
can load the model, and file-size estimates are a close approximation when you cannot.

## Diagnostic utilities

MLXLMCommon includes lightweight helpers to measure real memory usage so you can
model tickets based on observed behavior rather than only static estimates.
The utilities are policy-agnostic; use the measurements to size tickets or
validate a policy's budget assumptions.

Use `WiredMemoryUtils.tune(...)` to capture:

- `weightBytes` from `nbytes` (stable)
- `kvBytes` from actual cache arrays after prefill
- `workspaceBytes` from the prefill peak (transient)

The returned `WiredMemoryMeasurement` can be used to build a budget policy or to
validate manual calculations. For multimodal models, prefer the overload that
accepts a prepared `LMInput` or a `UserInput` so the measurement includes image
or video tensors.

## Practical guidance for tickets

- If you **can load**: compute `nbytes` once at load time and reuse it for the model's lifetime.
- If you **cannot load**: sum tensor file sizes as a proxy.
- Add a **small fixed margin** (e.g., 16-64 MB) to cover allocator overhead and minor variance.

For inference workloads, keep **weights**, **KV cache**, and **activation** budgets separate so
policies can scale the wired limit based on what is actually active.

## Policy-only budgeting on CPU

If wired memory control is unavailable (CPU-only execution), you can still use
policies for admission gating and budgeting by enabling policy-only mode on the
manager. This keeps ticket tracking and limit math active without attempting to
change the wired limit. Policy-only mode defaults to `true` on unsupported
backends.

```swift
await WiredMemoryManager.shared.updateConfiguration { configuration in
    configuration.policyOnlyWhenUnsupported = true
}
```

You can also provide `baselineOverride` (a fixed budget), or rely on
`GPU.maxRecommendedWorkingSetBytes()` when running on Apple Silicon with unified
memory.

## Estimating KV cache and attention workspace

Inference tickets are typically driven by **KV cache** (persistent) plus **prefill workspace**
(transient). Dense models are straightforward; MoE and hybrid models (like Qwen3-Next with
full-attention + linear/SSM layers) need a layer-by-layer sum using config values.

### Dense full-attention KV cache

For standard attention layers:

```
elements per token per layer = 2 * kvHeads * headDim
layer elements = tokens * elements per token per layer
layer bytes = layer elements * bytesPerElement
total KV bytes = layer bytes * numAttentionLayers
```

Where `bytesPerElement` is 2 for FP16/BF16, 1 for INT8, and 0.5 for INT4.

### Hybrid / MoE models with SSM (example: Qwen3-Next)

Qwen3-Next alternates full-attention layers with linear/SSM layers. Use the same KV math above
for **full-attention layers**, then add the SSM cache sizes for the linear layers.

For the SSM cache per linear layer, one workable approximation is:

```
convState elements = B * (convKernelSize - 1) * convDim
convDim = (keyHeadDim * numKeyHeads) * 2 + (valueHeadDim * numValueHeads)

state elements = B * numValueHeads * valueHeadDim * keyHeadDim

linear layer bytes = (convState elements + state elements) * bytesPerElement
total linear bytes = linear layer bytes * numLinearLayers
```

This yields a small but non-zero persistent cache budget for the linear/SSM layers.

### Prefill attention workspace (transient)

Prefill can allocate large temporary buffers proportional to the **prefill chunk size** `L`. A
simple upper bound for a single attention layer in FP16/BF16 is:

```
Q = B * H * L * D
K = B * Hkv * L * D
V = B * Hkv * L * D
Scores = B * H * L * L
Output = B * H * L * D
```

Multiply each by `bytesPerElement`, then sum to estimate peak transient workspace. If the model
uses an additional gating tensor, include it as `B * L * (H * D)`.

### Practical guidance

In MLXLMCommon, most callers will **create a single ticket** and run `generate()` inside the
ticket scope. In that case, budget the ticket for the **peak** expected usage
(weights + KV cache + prefill workspace). If you already created a **separate reservation
ticket** for weights, then the inference ticket should cover **KV cache + prefill workspace**
only.

If you need tighter control, you can split budgets by phase (e.g., a transient add-on for
prefill), but the common path is a single ticket.

- Compute **KV cache** separately from **weights**; KV persists for the duration of generation.
- Include **prefill workspace** in your peak estimate (it is transient, but can dominate memory).
- For hybrid models, sum all components (full-attention KV + linear/SSM cache + workspace).
- When using KV quantization, change `bytesPerElement` accordingly.
