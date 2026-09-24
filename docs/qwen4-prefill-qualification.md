# Opt-in Qwen4 prefill optimization qualification

These changes target `qwen4_exp` / `qwen4_exp_text` on Apple Silicon. They do
not change the default runtime paths. Device-specific performance must be
measured on the serving host.

| Flag (set to `1`) | Path | Eligibility / fallback |
| --- | --- | --- |
| `VMLX_QWEN4_GDN_BLOCKED_PREFILL` | Blocked gated-delta prefill | Supported head shapes and dtypes, at least 16 tokens; existing path otherwise |
| `VMLX_QWEN4_VERIFY_SDPA` | Sparse QK, stock softmax, tail-aligned sparse PV | Batch 1, 3–4 queries, context at least 8192, 24 query / 2 KV heads, head dimension 256, fp16/bf16 |
| `VMLX_QWEN4_PREFILL_DIRECT` | Native sparse QSA prefill | Evaluation prefills of at least 256 rows and context at least 8192, supported geometry, compatible built extension; existing path otherwise |
| `VMLX_QWEN4_COALESCE_PREFILL_CHECKPOINTS` | One lazy graph preserving stock segments and SSM/PLE checkpoints | Fresh batch-1 short requests with two supported boundaries through 4096; existing split prefill otherwise |
| `VMLX_QWEN4_ALIGNED_MOE_PREFILL` | Expert-aligned q4 prefill tiles | Stock evaluation SwitchGLU, 512 experts / top-10, 2560→640→2560 affine q4/group-64, fp16/bf16, 4096–10240 routed rows; qualified MLX 0.32.2 headers required, stock fallback otherwise |

The flags default to off. Prefix-cache identities include the effective math
flags and, for direct prefill, the native artifact content identity. Cache
state from incompatible configurations must not be reused. Coalescing errors
propagate after temporary checkpoint attributes are removed; they must not
retry the already mutated cache as another prefill.

## Build

### Experimental sparse online-softmax path

`VMLX_QWEN4_SPARSE_FUSED_PREFILL=1` is a separate default-off experiment,
not part of the historical qualification below. Unlike `PREFILL_DIRECT`, it
does not materialize a dense score/probability matrix. Its NAX consumer keeps
selected keys in their original 32-key tiles, preserves MLX's two independent
128-wide QK partial sums, and skips only tiles with no selected keys. Packing
the selected keys changed softmax/PV reduction order and failed full-model
numerical qualification; that implementation is not the active experiment.
Legacy materialized-score equality is not an oracle for this path.

Current eligibility is MLX 0.32.2 on `applegpu_g17s`, batch 1, FP16,
24 query / 2 KV heads, head dimension 256, 512 four-token selected blocks,
256–4096 query rows and 32768–131072 logical context tokens. Other shapes
retain their existing path; these are dispatch bounds, not context limits.
There is no change to decode, MTP, quantization, selection, or cache format.
The flag and native artifact identity separate prefix-cache namespaces.

An ABI check and nonzero, bit-exact current-SDPA preflight run before this
consumer can receive selected blocks. A failed preflight leaves stock active.
An execution error after cache mutation propagates; it never retries that
chunk against the mutated cache. Component thresholds do not replace
full-model quality, cache, UI/API and performance qualification.

Use the existing component benchmark with `--stock-path runtime --candidate
nax` to compare against today's fused masked path. `--candidate fused` denotes
the older SIMD implementation, which failed the full-model numerical gate.
Numerical differences
are reported, not silently treated as exactness or product acceptance.
The same-weight gate can isolate this experiment with `--arms sparse_fused
--prefill-step-size 1024 --continuation-step-size 1`. No whole-model speed or
production-admission claim is made for this experimental flag.

For numerical diagnosis, `--arms sparse_oracle` preserves the selected-block
route but reconstructs a mask for stock fused SDPA. It compares the native
consumer on the same real tensors without propagating its result, separating
consumer arithmetic from the surrounding route. Optional `--attention-fixture`
saves the first real attention inputs for component-only investigation.
`--arms repeat` checks stock-versus-stock repeatability. These controls are not
performance or product-admission receipts.

### Optional extension build

Use an isolated serving environment with the repository lockfile. For the
optional native QSA extension, follow
[`VMLX_INTEGRATION.md`](../native_extensions/qsa_kernels/VMLX_INTEGRATION.md).
The current extension requires MLX 0.32.2 and nanobind 2.15.0 at build time.
Build it with the serving Python; rebuild after an MLX/Python change. The main
vMLX wheel remains pure Python and does not implicitly install the extension.
Imported source notices and licenses accompany their respective packages.

Aligned MoE loads the installed MLX headers lazily and verifies their source
hashes before compiling a custom Metal kernel. It shares GPU routing metadata
within one SwitchGLU call and preserves stock activation, unsorting and weighted
reduction. Unsupported MLX versions or modified/missing headers use stock MoE.
The 16-row expert-aligned tile avoids redundant work when a stock tile crosses
expert boundaries; longer batches remain on stock based on measured results.

## Paired API protocol

Run the clean upstream checkout and candidate sequentially using the same
model weights, dependency environment, native-MTP sidecar, one-sequence limit,
and sampling settings. Set the five flags to `0` for the baseline and `1` for
the candidate. Never load two large model instances concurrently. Keep other
settings identical, including prefill chunk size and MTP priming policy.

For the qualification protocol, both arms use 4096-token prefill chunks,
fixed MTP depth 3, temperature 0.1, top-p 0.95 and top-k 20, with thinking off.
Both set `VMLX_NATIVE_MTP_PROMPT_PRIMING=0`,
`VMLX_QWEN4_HC_COMPILE_MAX_ROWS=4096`,
`VMLINUX_TIGHT_MEMORY_PREFILL_STEP_SIZE=4096`, and
`VMLX_METAL_WS_MAX_GB=120`. The working-set setting is host-specific.
Use separate disposable disk-cache directories with equal budgets. Preserve
raw request-matched cache telemetry and accept uncached measurements only
when their recorded cached-token count is zero.

```sh
.venv/bin/python bench/qwen4_speed_gate.py \
  --url http://127.0.0.1:18194/v1/chat/completions \
  --model qwen-next --label baseline --prompt-seed paired-unique-run-id \
  --tokenizer /path/to/model/tokenizer.json --output baseline-speed.json \
  --contexts 1024,4096,8192,16384,32768 --trials 3 --max-tokens 2048 \
  --exact-contexts --sustained-task --warmup-context 1024 \
  --temperature 0.1 --top-p 0.95 --top-k 20 --max-attempts 1
```

Repeat with the same prompt seed and arguments against the candidate, changing
only the output and label. The exact-context protocol assumes five template
tokens for this Qwen4 endpoint and rejects differing API counts. Each prompt
has an early unique nonce. The warmup is separate. Report all trials and
ranges; do not silently discard slow trials or substitute short completions.
Prefill is API prompt tokens divided by streamed time to first token. Decode
is API completion tokens divided by the remaining stream time. These are
resident-runtime API measurements, not cold model loading or kernel-only rates.

## Quality and numerical gates

```sh
.venv/bin/python bench/qwen4_quality_gate.py \
  --url http://127.0.0.1:18194/v1/chat/completions \
  --tokenizer /path/to/model/tokenizer.json --label candidate \
  --output candidate-quality.json
```

Run this against no-MTP and MTP configurations. `--extended` adds 64K and
approximately 131K retrieval cases; distinguish capacity rejections from
incorrect answers. The standard set checks exact answers, executable Python
behavior, parsed tool calls and long-context retrieval. It is a bounded
regression gate, not a broad capability benchmark.

After releasing the serving model, run the same-weight numerical gate:

```sh
.venv/bin/python bench/qwen4_logit_gate.py \
  --model /path/to/model --output logit-results.json
```

This compares each optimization and the combined path on identical
teacher-forced inputs at 1K/4K/8K/32K. Short contexts include both one-token
and eight-token final segments to cover the server generation suffix. Fixed thresholds are mean KL <= 0.01,
maximum row KL <= 0.05, and logit RMS <= 0.1. Top-1 agreement is also recorded.
The commands return a nonzero exit code on failed checks. Inspect every
recorded result as well as the exit code.

Run the recorded Qwen4, native-MTP, hybrid-prefix/SSM, batch-generator and disk
cache regression suite with Metal available and the optional extension built.
The manifest is the exact file selection used for the 1,101-test qualification:

```sh
xargs .venv/bin/python -m pytest -q --maxfail=10 \
  --junitxml=qwen4-regression-tests.xml < bench/qwen4-regression-tests.txt
```

Run this after releasing the full serving model, as in the recorded campaign.
Test counts may change when upstream adds cases; inspect the JUnit report for
failures, errors, and skips rather than treating the historical count as a gate.
Check cold/warm cache reuse, reuse after restart, cancellation followed by a
new request, incompatible native ABI fallback, and clean installed-wheel
execution. Record any skipped tests and untested device/OS combinations.

## Qualification scope

These paths are experimental and default off. Run the numerical and end-to-end protocols above on the target device before enabling them. A component benchmark does not establish application-level performance or model quality.
