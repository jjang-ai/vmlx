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

The flags default to off. Prefix-cache identities include the effective math
flags and, for direct prefill, the native artifact content identity. Cache
state from incompatible configurations must not be reused. Coalescing errors
propagate after temporary checkpoint attributes are removed; they must not
retry the already mutated cache as another prefill.

## Build

Use an isolated serving environment with the repository lockfile. For the
optional native QSA extension, follow
[`VMLX_INTEGRATION.md`](../native_extensions/qsa_kernels/VMLX_INTEGRATION.md).
The current extension requires MLX 0.32.2 and nanobind 2.15.0 at build time.
Build it with the serving Python; rebuild after an MLX/Python change. The main
vMLX wheel remains pure Python and does not implicitly install the extension.
Imported source notices and licenses accompany their respective packages.

## Paired API protocol

Run the clean upstream checkout and candidate sequentially using the same
model weights, dependency environment, native-MTP sidecar, one-sequence limit,
and sampling settings. Set the four flags to `0` for the baseline and `1` for
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
The manifest is the exact file selection used for the 1,078-test qualification:

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

## Recorded qualification status (2026-09-08)

**Quality and recovery pass; short-prefill target unmet, not merge-ready.**
The tail-aligned sparse-PV correction passed all 28 same-weight numerical
comparisons bit-for-bit: KL, logit RMS and maximum absolute error were zero,
with 100% top-1 agreement. All 1078 relevant regression tests passed without
skips. The thresholds were unchanged. The serving run passed 26/26 MTP quality
checks and cold/warm/restarted cache checks (two cases each). Request-matched
cache telemetry showed zero cold reuse and positive warm/restarted reuse.
Cancellation cleared in 59 ms, followed by a successful request.

Measured upstream engine: vMLX 1.6.56 `ca3dc2d8264647b79937e6002e5e47a32f239ced`.
Latest checked upstream `f9aa51b26db4bb79490353920ae35afc36330a3d` has identical
inference-engine and runtime dependency files; intervening changes affect
desktop UI, packaging and the release manifest. Host: M4 Max, 128 GiB, CRACK
JANG4M model, MLX 0.32.2, mlx-lm 0.31.3, mlx-vlm 0.5.0, Python 3.12.13.

| Prompt tokens | Baseline prefill | Candidate prefill | Change | Baseline decode | Candidate decode | Change |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1024 | 564.9 | 569.7 | +0.8% | 48.7 | 50.6 | +4.0% |
| 4096 | 696.1 | 703.5 | +1.1% | 49.0 | 49.5 | +1.2% |
| 8192 | 672.6 | 696.5 | +3.6% | 47.4 | 51.6 | +8.8% |
| 16384 | 650.8 | 694.3 | +6.7% | 40.8 | 49.2 | +20.3% |
| 32768 | 562.1 | 628.8 | +11.9% | 35.4 | 48.3 | +36.4% |

Units are API tokens/second, means of three uncached trials with exactly 2048
output tokens. The baseline comes from the preceding same-host campaign.
Its entire paired 1K cohort was repeated under exclusive load after a numerical
probe overlapped the original baseline 1K cohort; that decision was recorded
before any candidate timing. Other baseline sizes and all new candidate trials
were uncontended. No slow valid trial was omitted.

All fifteen candidate decode trials exceeded 40 tokens/s. All three 1K prefill
trials remained below 600 tokens/s; other prompt sizes met that target.
The JSON receipt retains ranges, standard deviations, source hashes and every
context summary. These results supersede earlier approximate-prototype and
full-PV speed tables; their gains must not be attributed to this revision.

The previous extended no-MTP controls passed 28/30 in both versions: every
admitted check through 64K passed, while two approximately 131K requests were
rejected for memory capacity in each version. Those extended controls were
not repeated in this PV-only serving run and are labeled as prior evidence.
The new full-model numerical and component checks cover the PV correction.

Native source and binaries did not change for this correction. They previously
built and executed from clean installed wheels on Python 3.11, 3.12 and 3.14 with
build/source locations hidden. The main engine qualification above used 3.12.
Other GPU families and macOS versions remain unmeasured. Production was restored
healthy with unchanged configuration after the serving campaign.

Machine-readable evidence:
[`qwen4-prefill-qualification-results.json`](qwen4-prefill-qualification-results.json).
Further prefill experiments are isolated and are not part of these results.
