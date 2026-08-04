# DeepSeek-V4 decode integration acceptance

The DSV4 optimizations are guarded by model-shape, dtype, cache-shape, and
request-length checks. Unsupported layouts and runtime failures must remain on
the native JANG/MLX implementation. A faster short run is not sufficient for
promotion: the output stream must remain stable through the long-run gate.

## Runtime policy

- The compiled native-gather MoE path is enabled by default for exact DSV4
  weighted-route layouts and accepts only single-token decode shapes.
- The quantized affine vocabulary-head path is enabled by default when the
  loaded head exposes the native MLX quantization contract.
- The exact RoPE table cache is enabled by default and is bounded. Native RoPE
  remains an explicit experiment with `VMLX_DSV4_ROPE_NATIVE=1`.
- The indexer bypass is request-bounded and applies only to fresh, short
  requests. Prefill and prefix-cache-hit paths retain the native indexer.
- The custom affine Metal path remains opt-in with
  `VMLX_DSV4_AFFINE_MOE_FASTPATH=1`.

Every optional path must be safe to disable through its environment control,
and a failed optimized dispatch must fall back to the original implementation
for the remainder of the process or request as appropriate.

## Required checks

Run the source checks first:

```text
git diff --check
python -m py_compile \
  vmlx_engine/loaders/load_jangtq_dsv4.py \
  vmlx_engine/utils/dsv4_batch_generator.py \
  vmlx_engine/models/dsv4_compiled_moe.py \
  vmlx_engine/models/dsv4_indexer_skip.py \
  vmlx_engine/models/dsv4_lm_head_cache.py \
  vmlx_engine/models/dsv4_rope_cache.py
python -m pip check
```

Run the no-model contract suite:

```text
python -m pytest -q \
  tests/test_dsv4_stability_fastpaths.py \
  tests/test_dsv4_affine_moe_fastpath.py \
  tests/test_dsv4_batch_generator_speed.py \
  tests/test_dsv4_contract_hardening.py \
  tests/test_dsv4_route_mode_code_exactness.py \
  tests/test_dsv4_bundle_integrity.py
```

Run the target harness with a real DSV4 bundle. Without
`--expected-token-hash`, the report is a repeatability result only:

```text
python bench/dsv4_target_harness.py \
  --model /path/to/DeepSeek-V4-Flash-JANG \
  --out dsv4-target-report.json \
  --repeats 3 \
  --prompt-targets 256
```

For an exact-reference gate, pass the known-good 30-token stream hash. The
same distinction applies to the 1,000-token harness:

```text
python bench/dsv4_longrun_harness.py \
  --model /path/to/DeepSeek-V4-Flash-JANG \
  --out dsv4-longrun-report.json \
  --repeats 1 \
  --prompt-targets 256 \
  --max-tokens 1000 \
  --expected-token-hash <reference-sha256>
```

Keep the raw JSON reports with the review artifacts. Report decode and prefill
as separate measurements, and do not promote a candidate that is exact but
slower or that only passes the short stream gate.
