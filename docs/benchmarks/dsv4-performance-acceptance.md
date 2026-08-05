# DeepSeek-V4 performance acceptance

DSV4 performance changes are accepted against the source runtime, not against
another experimental configuration. The suite runs the same prompts and greedy
generation cases twice: first with all promoted fast paths disabled, then with
only the candidate paths enabled.

The promoted profile intentionally contains two changes:

- a headroom-gated cache of the exact dequantized FP32 vocabulary matrix,
  preserving JANG's FP32 matmul while avoiding repeated dequantization. Its
  persistent bytes are deducted from the existing MLX allocator-cache ceiling
  so the optimization does not expand the runtime's memory envelope;
- bounded sharing of the existing manual RoPE cosine/sine tables across equal
  DSV4 RoPE instances.

The custom affine Metal MoE kernel remains an explicit hardware experiment.
Compiled MoE, unused-indexer mutation, native RoPE, and a raised layerwise
prefill threshold are disabled by the suite and are not part of promotion.

## Required commands

Use one runtime and one immutable model bundle for both runs:

```text
python bench/dsv4_performance_suite.py \
  --model /path/to/DeepSeek-V4-Flash-JANG \
  --profile native \
  --out dsv4-native.json

python bench/dsv4_performance_suite.py \
  --model /path/to/DeepSeek-V4-Flash-JANG \
  --profile optimized \
  --baseline-report dsv4-native.json \
  --acceptance \
  --out dsv4-optimized.json
```

The default matrix covers three 30-token samples at prompt targets 256, 400,
512, 1024, and 2048, plus a 1,000-token identity gate at prompt target 256. Every
acceptance run must include this complete matrix; custom subsets are diagnostic
only. Every
candidate hash must equal its native counterpart. Median decode must remain at
least 98% of native, the slowest decode sample at least 75% of the native
minimum, and median prefill at least 95%. Independently, the slowest repeat in
each profile must remain at least 75% of that profile's median; an unstable
native run cannot be used to inflate an optimized ratio. Every median decode
must also remain at least 40% of the warm 1,000-token anchor. This catches a
consistently pathological short case that a within-case ratio alone would call
stable. Improvements are reported as ratios rather than a machine-specific
absolute target.

The matrix runs sequentially without clearing MLX allocator state. This models
the warm multi-request runtime and makes allocator-residency cliffs part of the
acceptance result rather than hiding them with a synthetic cold-cache protocol.

Keep both JSON reports with the PR evidence. A report without
`--baseline-report` proves repeatability only and is not an acceptance result.
Reports record a bundle manifest fingerprint and a verified relative checkout
identity; they never publish machine-local model or source paths.
