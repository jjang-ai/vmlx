# Experimental upstream MLX GDN for Qwen4

`VMLX_QWEN4_MLX_FAST_GDN=1` selects `mx.fast.gated_delta_update` for an
unmasked GPU prefill of at least 16 tokens when the input is FP16 or BF16 and
the Qwen4 linear-attention geometry is 16 key heads, 48 value heads, and
128-wide key and value vectors. It takes precedence over
`VMLX_QWEN4_GDN_BLOCKED_PREFILL=1` for that shape. Decode, native-MTP short
verification segments, masked inputs, unsupported geometry, and the default
configuration retain the existing dispatch. Requesting the new path with an
MLX build that lacks `mx.fast.gated_delta_update` raises a clear error once
the eligible shape is reached.

The API arrived in [MLX #4020](https://github.com/ml-explore/mlx/pull/4020).
The related [routed `gather_qmm` scheduling update](https://github.com/ml-explore/mlx/pull/4572)
is in MLX itself and needs no vMLX call-site change. Both are on MLX `main`
at commit `64ea011cb65f14d9ce2737e60db9a4ae91ed7441`; the installed
production release remains MLX 0.32.2. No core dependency upgrade is made
by this opt-in vMLX change.

The optional native QSA extension must be rebuilt with the same MLX version
and nanobind ABI as the serving runtime. Its package metadata now records
the exact MLX version used at build time. The local MLX-main wheel used for
the September 29, 2026 experiment was built with nanobind 3.0.1 and a
macOS 26.6.2 minimum. This locally built wheel is not a portable release.
The extension's `pyproject.toml` still pins the supported MLX 0.32.2 build
for isolated package builds. For this main-commit experiment, install
nanobind 3.0.1 in the candidate environment and run `setup.py build_ext
--inplace` with that Python and `CMAKE_OSX_DEPLOYMENT_TARGET=26.6.2`.
The separate aligned MoE prefill kernel remains qualified only for MLX
0.32.2 and falls back to stock routing on the MLX-main build.

On an M4 Max, the direct GDN component comparison at the Qwen Next
16/48-head, 128/128 shape was 1.44x faster at 4K tokens than vMLX's existing
blocked prefill kernel, with a maximum absolute output difference of
`6.1e-5` in the synthetic FP16 probe. A matched full-model Qwen Next run
used the same 1K/4K/8K/16K prompts and 512-token completions, three trials
per size, fixed native MTP depth 3, and zero cached tokens. Both arms passed
26/26 bounded quality cases. Whole-model prefill median changes were -1.0%,
+1.8%, +1.2%, and -2.9% at those sizes; decode varied more. This does not
establish a whole-model speed gain or justify enabling the flag by default.

The raw build logs, component probes, request-level API receipts, quality
results, and production restoration record are in
`/Users/hermes/projects/ModelLab/artifacts/mlx-main-qwen-next-20260929/`.
