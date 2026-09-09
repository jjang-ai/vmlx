"""Stock fallback and expert-boundary correctness for short Qwen4 prefill."""

import mlx.core as mx
import pytest
from mlx_lm.models.switch_layers import SwitchGLU

from vmlx_engine.metal import qwen4_aligned_moe_prefill as aligned


def test_default_off_does_not_inspect_or_compile(monkeypatch):
    monkeypatch.delenv("VMLX_QWEN4_ALIGNED_MOE_PREFILL", raising=False)
    assert aligned.aligned_switchglu(None, None, None) is None


@pytest.mark.parametrize("tokens", [1, 4, 63, 409, 1025, 4096])
def test_decode_small_and_long_batches_fall_back(tokens, monkeypatch):
    monkeypatch.setenv("VMLX_QWEN4_ALIGNED_MOE_PREFILL", "1")
    switch = SwitchGLU(16, 16, 2)
    switch.eval()
    x = mx.zeros((1, tokens, 2560), mx.bfloat16)
    ids = mx.zeros((1, tokens, 10), mx.int32)
    monkeypatch.setattr(aligned, "_kernel", lambda: pytest.fail("unexpected compile"))
    assert aligned.aligned_switchglu(switch, x, ids) is None


def test_unquantized_projection_falls_back(monkeypatch):
    monkeypatch.setenv("VMLX_QWEN4_ALIGNED_MOE_PREFILL", "1")
    switch = SwitchGLU(16, 16, 2)
    switch.eval()
    monkeypatch.setattr(aligned, "_kernel", lambda: pytest.fail("unexpected compile"))
    assert (
        aligned.aligned_switchglu(
            switch,
            mx.zeros((1, 512, 2560), mx.float16),
            mx.zeros((1, 512, 10), mx.int32),
        )
        is None
    )


@pytest.mark.parametrize("dtype", [mx.float16, mx.bfloat16])
@pytest.mark.parametrize("rows,k,n", [(4096, 2560, 640), (10240, 640, 2560)])
@pytest.mark.parametrize("distribution", ["uniform", "skewed"])
def test_expert_boundaries_match_stock_exactly(dtype, rows, k, n, distribution):
    kernel = aligned._kernel()
    if kernel is None:
        pytest.skip("requires qualified MLX 0.32.2 SIMD backend and Metal headers")
    mx.random.seed(rows + k)
    x = (mx.random.normal((rows, 1, k)) * 0.1).astype(dtype)
    w = mx.random.randint(0, 2**31, (512, n, k // 8), dtype=mx.uint32)
    scales = (mx.random.uniform(shape=(512, n, k // 64)) * 0.02).astype(dtype)
    biases = (-7 * scales).astype(dtype)
    if distribution == "uniform":
        ids = mx.sort(mx.random.randint(0, 512, (rows,), dtype=mx.int32))
    else:
        # Empty experts, partial 16-row tiles, and the last expert are exercised.
        ids = mx.concatenate(
            [
                mx.zeros((17,), mx.int32),
                mx.full((rows - 48,), 255, mx.int32),
                mx.full((31,), 511, mx.int32),
            ]
        )
    counts = mx.zeros((512,), mx.int32).at[ids].add(mx.ones((rows,), mx.int32))
    starts = mx.concatenate([mx.zeros((1,), mx.int32), mx.cumsum(counts)])
    ends = mx.cumsum((counts + 15) // 16)
    ref = mx.gather_qmm(
        x,
        w,
        scales,
        biases,
        rhs_indices=ids,
        transpose=True,
        group_size=64,
        bits=4,
        mode="affine",
        sorted_indices=True,
    )
    got = kernel(
        inputs=[x, w, scales, biases, starts, ends, n, k],
        template=[("T", dtype), ("BM", 16)],
        grid=(n // 32 * 64, (rows + 15) // 16 + 512, 1),
        threadgroup=(64, 1, 1),
        output_shapes=[(rows, 1, n)],
        output_dtypes=[dtype],
    )[0]
    mx.eval(ref, got)
    assert bool(mx.array_equal(ref, got))


@pytest.mark.parametrize("family", ["qwen4_exp", "qwen4_exp_text", "llama"])
def test_opt_in_cache_identity_scoped_to_qwen4(family, monkeypatch):
    from types import SimpleNamespace

    from vmlx_engine.prefix_cache import compute_model_cache_key

    model = SimpleNamespace(args=SimpleNamespace(model_type=family))
    monkeypatch.setenv("VMLX_QWEN4_ALIGNED_MOE_PREFILL", "0")
    before = compute_model_cache_key(model)
    monkeypatch.setenv("VMLX_QWEN4_ALIGNED_MOE_PREFILL", "1")
    after = compute_model_cache_key(model)
    assert (before != after) == family.startswith("qwen4_exp")


@pytest.mark.parametrize("enabled", [False, True])
def test_math_revision_invalidates_only_enabled_cache(enabled, monkeypatch):
    from types import SimpleNamespace

    from vmlx_engine.prefix_cache import compute_model_cache_key

    model = SimpleNamespace(args=SimpleNamespace(model_type="qwen4_exp_text"))
    monkeypatch.setenv("VMLX_QWEN4_ALIGNED_MOE_PREFILL", str(int(enabled)))
    before = compute_model_cache_key(model)
    monkeypatch.setattr(aligned, "MATH_ABI", "future-revision")
    assert (before != compute_model_cache_key(model)) == enabled


def test_unqualified_mlx_version_falls_back(monkeypatch):
    aligned._kernel.cache_clear()
    try:
        with monkeypatch.context() as m:
            m.setattr(mx, "__version__", "unqualified")
            assert aligned._kernel() is None
    finally:
        aligned._kernel.cache_clear()


def test_modified_headers_fall_back(monkeypatch):
    aligned._kernel.cache_clear()
    try:
        with monkeypatch.context() as m:
            # Exercise the header check even on a host where dispatch would
            # ordinarily retain stock NAX before reaching it.
            m.setattr(aligned, "_simd_backend_eligible", lambda: True)
            m.setattr(aligned, "_HEADER_HASHES", {})
            assert aligned._kernel() is None
    finally:
        aligned._kernel.cache_clear()


@pytest.mark.parametrize("architecture,eligible", [
    ("applegpu_g16s", True), ("applegpu_g17s", False),
    ("applegpu_g17p", True), ("applegpu_g18p", False),
    ("applegpu_g18s", False), ("unknown", False), (None, False),
])
def test_simd_kernel_does_not_override_nax_backend(architecture, eligible, monkeypatch):
    monkeypatch.setattr(mx, "device_info", lambda: {"architecture": architecture})
    assert aligned._simd_backend_eligible() is eligible


def test_nax_hardware_never_compiles_simd_kernel(monkeypatch):
    aligned._kernel.cache_clear()
    try:
        monkeypatch.setattr(mx, "device_info", lambda: {"architecture": "applegpu_g17s"})
        monkeypatch.setattr(mx.fast, "metal_kernel", lambda **kw: pytest.fail("SIMD compile on NAX hardware"))
        assert aligned._kernel() is None
    finally:
        aligned._kernel.cache_clear()
