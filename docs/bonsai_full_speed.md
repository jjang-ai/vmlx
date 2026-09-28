# Optional Bonsai full-speed serving

This path targets Qwen3.5 Bonsai JANG bundles with explicit signed-Hadamard transforms and 2-bit/group-128 ternary affine weights. It keeps the default vMLX behavior unchanged. The optional branches are selected by environment variables so each can be measured against the same model and runtime.

| Variable | Effect |
| --- | --- |
| `VMLX_BONSAI_FUSED_HADAMARD=1` | Use the checked, 1024-wide fused rotation where supported. |
| `VMLX_BONSAI_FP16_PREFILL_ONLY=1` | Keep long prefill activations FP16 while decode retains the native precision policy. This policy has a distinct prefix-cache identity. |
| `VMLX_BONSAI_FP16_QMM=1` | Use FP16 inputs to qualifying packed projections, casting results back to the stream dtype. |
| `VMLX_BONSAI_PREFILL_DQ_GEMM=1` | Use a dequantized FP16 GEMM for eligible 2048+ row prefill projections. Requires FP16 QMM. |
| `VMLX_BONSAI_TERNARY_VERIFY=1` | Use the guarded 2-3-row exact-ternary verifier. Requires FP16 QMM and affine biases equal to negative scales. |
| `VMLX_BONSAI_GDN_FP32=1` | Use the guarded FP32 GDN recurrence on its supported geometry. |
| `VMLX_BONSAI_ANE_PREFILL=1` | Prepare a 4096-row ANE/CPU/GPU prefill split using oMLX custom kernels. Requires FP16 prefill-only and FP16 QMM. |

The ANE branch needs the **native** `omlx.custom_kernels.qwen35_prefill` extension in the same Python environment. A default Python-only oMLX installation does not provide it. The verified build used oMLX commit `a98d8c8c301a7b07683ad9959d695ed91af5aab8`, MLX `0.32.2`, nanobind `2.15.0`, Python `3.12`, and macOS 15. Build the native wheel and install it without oMLX's unrelated server dependencies:

```bash
git clone https://github.com/jundot/omlx.git
cd omlx
git checkout a98d8c8c301a7b07683ad9959d695ed91af5aab8
OMLX_WITH_CUSTOM_KERNEL=1 uv build --wheel --out-dir dist
uv pip install --python /path/to/vmlx/.venv/bin/python --no-deps dist/omlx-*.whl
```

Verify the capability in the vMLX environment before serving:

```bash
python - <<'PY'
from omlx.custom_kernels.qwen35_prefill import fast
assert fast.qwen35_ane_available()
assert fast.qwen35_cpu_shared_resource_available()
assert fast.qwen35_ane_bank_compiler_available()
PY
```

For a compatible JANG bundle with its Q4 native-MTP head overlay:

```bash
VMLX_BONSAI_FUSED_HADAMARD=1 \
VMLX_BONSAI_FP16_PREFILL_ONLY=1 \
VMLX_BONSAI_FP16_QMM=1 \
VMLX_BONSAI_PREFILL_DQ_GEMM=1 \
VMLX_BONSAI_TERNARY_VERIFY=1 \
VMLX_BONSAI_GDN_FP32=1 \
VMLX_BONSAI_ANE_PREFILL=1 \
vmlx serve /path/to/bonsai-jang-mtp-overlay --native-mtp-depth 1
```

The ANE adapter checks the loaded modules, affine layout, output alignment, projection bias, and native runtime before registering any projections. It applies only to prefill rows 2048–4096; other sizes use the regular projection path. If the built oMLX backend is unavailable when requested, model loading fails with a clear error. Set all flags before model loading so the prefix-cache identity reflects the selected numerical path.

The ANE branch uses approximate INT8 weights for part of each projection. Long-prompt outputs can differ from the ANE-off result, so validate model quality on the intended tasks before enabling it for a workload. On the measured M4 Max with the Dealign Bonsai 2 CRACK JANG/MTP overlay, model preparation compiled 34 ANE banks in about 41 seconds and the 4K benchmark reached about 26 GB peak Metal memory. These costs are hardware and model dependent. The MLX-Serve GDN and ternary kernels are used under their retained MIT license; the rotation retains its MTPLX Apache-2.0 license and NOTICE.
