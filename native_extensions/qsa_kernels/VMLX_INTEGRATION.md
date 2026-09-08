# VMLX Qwen4 sparse prefill integration

Vendored from MTPLX revision `21be78b3f51820eecef020e5e4855c0715eaf9a5`.
vMLX adds a separate materialized sparse-QK primitive; the original fused
primitive remains available for API compatibility. See `mtplx_qsa_kernels/NOTICE` and the
license files for oMLX, MLX, mlx-serve, and MTPLX provenance.

The optional Python adapter is `vmlx_engine/metal/qwen4_prefill_direct.py`.
`VMLX_QWEN4_PREFILL_DIRECT=1` requests it; the default is off. VMLX only
routes evaluation prefills with at least 256 rows, context at least 8192,
and the supported Qwen4 geometry. Decode and short suffixes retain their existing attention path.
The indexer preserves its selected blocks and sorts their IDs into the valid
prefix required by the native kernel. The serving adapter uses only the materialized-score primitive, which writes
selected scores directly into the stock layout after initializing masked
positions. It preserves
stock low-precision QK, absolute softmax positions, and PV accumulation order.
It still materializes dense score/probability arrays, but avoids gathered KV
arrays and dense QK computation. Older binaries without the new API are
disabled until rebuilt. No quality claim is made by speed runs.

Build using the exact serving interpreter and its MLX wheel:

```sh
uv pip install --python .venv/bin/python 'nanobind==2.15.0' setuptools
cd native_extensions/qsa_kernels
../../.venv/bin/python setup.py build_ext --inplace
```

Build products are local and ignored by Git. Rebuild after changing MLX or
Python. The adapter checks build receipts and the nanobind ABI, and proves
both packaged Metal dtype pipelines before enabling the path.

To run the isolated component speed comparison from the repository root:

```sh
.venv/bin/python benchmarks/qwen4_sparse_prefill.py --output /tmp/qsa-speed.json
```

Avoid running the component test concurrently with model timing requests.
The upstream README in this directory describes MTPLX integration. For vMLX
full-model reproduction and qualification, see
[`docs/qwen4-prefill-qualification.md`](../../docs/qwen4-prefill-qualification.md).
