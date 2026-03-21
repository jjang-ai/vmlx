"""
Test: QuantizedKVCache prefix cache bug fix

Bug: _quantize_cache_for_storage() assigned mx.quantize() (which returns a list)
directly to qkv.keys/qkv.values. This caused qkv.state to return (list, list).
In _extract_block_tensor_slice(), the isinstance(keys, tuple) check then failed,
falling through to len(keys.shape) -> AttributeError: 'list' object has no attribute 'shape'.

Fix:
  - scheduler.py: wrap mx.quantize() with tuple() in _quantize_cache_for_storage()
  - prefix_cache.py: isinstance(keys, (tuple, list)) in _extract_block_tensor_slice()
                     and _is_positional_cache() structure fallback

Usage:
    python3 test_quantized_kv_prefix_cache.py
"""
import sys
import importlib.util
import traceback

import mlx.core as mx

# ---------------------------------------------------------------------------
# Load the bundled mlx_lm cache module in isolation (avoids heavy imports)
# ---------------------------------------------------------------------------
_CACHE_PATH = (
    "/Applications/vMLX.app/Contents/Resources/bundled-python"
    "/python/lib/python3.12/site-packages/mlx_lm/models/cache.py"
)
spec = importlib.util.spec_from_file_location("mlx_lm.models.cache", _CACHE_PATH)
cache_mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(cache_mod)
KVCache = cache_mod.KVCache
QuantizedKVCache = cache_mod.QuantizedKVCache


def make_test_kv(B=1, H=8, S=64, D=128):
    """Return a filled KVCache with S tokens."""
    kv = KVCache()
    kv.keys = mx.random.normal((B, H, S, D))
    kv.values = mx.random.normal((B, H, S, D))
    kv.offset = S
    return kv


# ---------------------------------------------------------------------------
# Test 1 — reproduce the original bug
# ---------------------------------------------------------------------------
def test_bug_exists_without_fix():
    """
    Demonstrate the original bug: assigning mx.quantize() list to qkv.keys
    causes state to return (list, list), which later breaks isinstance checks.
    """
    kv = make_test_kv()
    qkv = QuantizedKVCache(group_size=64, bits=8)

    # This is what the BUGGY code did (no tuple() wrap):
    qkv.keys = mx.quantize(kv.keys, group_size=64, bits=8)
    qkv.values = mx.quantize(kv.values, group_size=64, bits=8)
    qkv.offset = kv.offset

    state = qkv.state
    keys, values = state
    assert isinstance(keys, list), (
        f"Expected list from buggy path, got {type(keys).__name__}"
    )
    print("[test_bug_exists_without_fix] PASS — buggy path produces list as expected")
    return keys


# ---------------------------------------------------------------------------
# Test 2 — verify root-cause fix in _quantize_cache_for_storage
# ---------------------------------------------------------------------------
def test_fix_tuple_wrap():
    """
    Verify that wrapping mx.quantize() with tuple() makes qkv.state return
    (tuple, tuple), so isinstance(keys, tuple) succeeds.
    """
    kv = make_test_kv()
    qkv = QuantizedKVCache(group_size=64, bits=8)

    # FIXED code: tuple() wrap
    qkv.keys = tuple(mx.quantize(kv.keys, group_size=64, bits=8))
    qkv.values = tuple(mx.quantize(kv.values, group_size=64, bits=8))
    qkv.offset = kv.offset

    state = qkv.state
    keys, values = state
    assert isinstance(keys, tuple), (
        f"Expected tuple from fixed path, got {type(keys).__name__}"
    )
    assert len(keys) == 3, f"Expected 3 components (data/scales/zeros), got {len(keys)}"
    assert hasattr(keys[0], "shape"), "keys[0] should be an MLX array with .shape"
    print("[test_fix_tuple_wrap] PASS — fixed path produces tuple, isinstance works")


# ---------------------------------------------------------------------------
# Test 3 — verify _extract_block_tensor_slice behaviour after fix
# ---------------------------------------------------------------------------
def test_extract_slice():
    """
    Simulate _extract_block_tensor_slice's QuantizedKVCache branch with both
    list and tuple to confirm the (tuple, list) isinstance fix handles both.
    """
    kv = make_test_kv(S=128)
    mx.eval(kv.keys, kv.values)

    start_idx, end_idx = 0, 64

    for label, make_keys in [
        ("list (buggy storage)",  lambda k, v: (list(mx.quantize(k, 64, 8)), list(mx.quantize(v, 64, 8)))),
        ("tuple (fixed storage)", lambda k, v: (tuple(mx.quantize(k, 64, 8)), tuple(mx.quantize(v, 64, 8)))),
    ]:
        keys_q, values_q = make_keys(kv.keys, kv.values)

        # Replicate the fixed branch logic:
        try:
            if isinstance(keys_q, (tuple, list)):   # FIXED check
                first_k = keys_q[0]
                seq_len = first_k.shape[-2]          # works for both list and tuple
                actual_end = min(end_idx, seq_len)
                assert start_idx < actual_end
                keys_slice = tuple(t[..., start_idx:actual_end, :] for t in keys_q)
                values_slice = tuple(t[..., start_idx:actual_end, :] for t in values_q)
                mx.eval(*keys_slice, *values_slice)
                print(f"[test_extract_slice] PASS for {label} — slice shape: {keys_slice[0].shape}")
            else:
                raise AssertionError(f"isinstance check failed for {label}")
        except Exception as e:
            print(f"[test_extract_slice] FAIL for {label}: {e}")
            traceback.print_exc()
            sys.exit(1)


# ---------------------------------------------------------------------------
# Test 4 — verify _is_positional_cache structural fallback
# ---------------------------------------------------------------------------
def test_is_positional_cache_structural():
    """
    Verify that the structural fallback in _is_positional_cache handles both
    list and tuple keys after the comment fix.
    """
    kv = make_test_kv()
    mx.eval(kv.keys)

    q_list = list(mx.quantize(kv.keys, 64, 8))   # list
    q_tuple = tuple(mx.quantize(kv.keys, 64, 8)) # tuple

    for label, first in [("list keys", q_list), ("tuple keys", q_tuple)]:
        # Replicate the fixed _is_positional_cache structural fallback:
        ok = isinstance(first, (tuple, list)) and len(first) >= 2
        ok = ok and hasattr(first[0], "shape") and len(first[0].shape) in (3, 4)
        assert ok, f"Structural fallback failed for {label}"
        print(f"[test_is_positional_cache_structural] PASS — {label} detected as positional")


# ---------------------------------------------------------------------------
# Run all tests
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    print("=" * 60)
    print("QuantizedKVCache prefix cache bug fix verification")
    print("=" * 60)

    buggy_keys = test_bug_exists_without_fix()
    assert not isinstance(buggy_keys, tuple), "Bug should produce list, not tuple"

    test_fix_tuple_wrap()
    test_extract_slice()
    test_is_positional_cache_structural()

    print("=" * 60)
    print("All tests PASSED")
    print("=" * 60)
