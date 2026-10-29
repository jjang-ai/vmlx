"""CPU control/lifetime checks for sparse index capacity, without importing MLX.

Execute the owning cache methods from AST against small immutable slice values.
These checks establish logical admission/serialization and growth behavior;
queued MLX consumers and completed GPU cost need the separate native gate.
"""

from __future__ import annotations

import ast
import itertools
import math
from pathlib import Path
import unittest

ROOT = Path(__file__).parents[1]


class Array:
    def __init__(self, shape, values=None, dtype="float32"):
        self.shape, self.dtype = tuple(shape), dtype
        self.data = list(values if values is not None else [0] * math.prod(shape))
        assert len(self.data) == math.prod(shape)

    @property
    def nbytes(self):
        return len(self.data) * 4

    def _selection(self, key):
        key = key if isinstance(key, tuple) else (key,)
        if Ellipsis in key:
            pos = key.index(Ellipsis)
            key = key[:pos] + (slice(None),) * (len(self.shape) - len(key) + 1) + key[pos + 1:]
        key += (slice(None),) * (len(self.shape) - len(key))
        axes, shape = [], []
        for size, index in zip(self.shape, key):
            if isinstance(index, slice):
                items = list(range(*index.indices(size)))
                shape.append(len(items))
            else:
                items = [index % size]
            axes.append(items)
        positions = []
        for indices in itertools.product(*axes):
            position = 0
            for index, size in zip(indices, self.shape):
                position = position * size + index
            positions.append(position)
        return tuple(shape), positions

    def __getitem__(self, key):
        shape, positions = self._selection(key)
        return Array(shape, [self.data[position] for position in positions], self.dtype)

    def __setitem__(self, key, value):
        shape, positions = self._selection(key)
        assert shape == value.shape
        for position, item in zip(positions, value.data):
            self.data[position] = item


class MX:
    array = Array
    uint8, bfloat16 = "uint8", "bfloat16"

    def __init__(self):
        self.concatenations = 0

    def zeros(self, shape, dtype="float32"):
        return Array(shape, dtype=dtype)

    def concatenate(self, arrays, axis):
        assert axis == 2 and all(array.shape[:2] == (1, 1) for array in arrays)
        self.concatenations += 1
        return Array((1, 1, sum(array.shape[2] for array in arrays), arrays[0].shape[3]),
                     list(itertools.chain.from_iterable(array.data for array in arrays)),
                     arrays[0].dtype)


class KV:
    """Only the inherited KV logical/capacity contract used by these methods."""
    def __init__(self):
        self.keys = self.values = None
        self.offset = 0

    @property
    def state(self):
        return (None, None) if self.keys is None else (
            self.keys[..., :self.offset, :], self.values[..., :self.offset, :]
        )

    @state.setter
    def state(self, values):
        self.keys, self.values = values
        self.offset = self.keys.shape[2]

    def trim(self, n):
        n = min(self.offset, n)
        self.offset -= n
        return n

    def is_trimmable(self):
        return True


class KQ:
    """Only the lane-format surface (models/minimax_m3/kv_quant.py) the cache uses for plain (bf16) lanes."""
    class Fmt:
        name, code, quantized = "bf16", 0, False

    BF16 = Fmt()

    @staticmethod
    def get(name):
        assert name in (None, "", "bf16") or name is KQ.BF16
        return KQ.BF16

    @staticmethod
    def configured():
        return KQ.BF16, KQ.BF16

    @staticmethod
    def lane_format(lane, fmt, head_dim=None):
        assert lane.dtype != "uint8"
        return KQ.BF16


def load_owners():
    mx = MX()
    ns = {"mx": mx, "KVCache": KV, "CACHE_TUPLE_TAG": "minimax_m3", "KQ": KQ}
    files = {
        "vmlx_engine/models/minimax_m3/cache.py": (
            "MiniMaxM3SparseCache", "restore_minimax_m3_sparse", "clone_minimax_m3_sparse"
        ),
        "vmlx_engine/mllm_batch_generator.py": ("_native_mtp_trim_head_chain",),
        "vmlx_engine/memory_cache.py": ("_estimate_state_memory", "estimate_kv_cache_memory"),
    }
    for name, wanted in files.items():
        tree = ast.parse((ROOT / name).read_text())
        nodes = [ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0)]
        nodes.extend(node for node in tree.body
                     if isinstance(node, (ast.ClassDef, ast.FunctionDef)) and node.name in wanted)
        assert len(nodes) == len(wanted) + 1
        module = ast.fix_missing_locations(ast.Module(body=nodes, type_ignores=[]))
        exec(compile(module, str(ROOT / name), "exec"), ns)
    return ns, mx


def rows(values, width=3):
    return Array((1, 1, len(values), width), [value for value in values for _ in range(width)])


def append(cache, values):
    # The real attention owner appends K/V immediately before the index lane.
    old = cache.offset
    capacity = max(256, old + len(values), 0 if cache.keys is None else cache.keys.shape[2])
    if cache.keys is None or capacity > cache.keys.shape[2]:
        keys, payload = rows([0] * capacity, 2), rows([0] * capacity, 2)
        if cache.keys is not None:
            keys[..., :old, :] = cache.keys[..., :old, :]
            payload[..., :old, :] = cache.values[..., :old, :]
        cache.keys, cache.values = keys, payload
    cache.keys[..., old:old + len(values), :] = rows(values, 2)
    cache.values[..., old:old + len(values), :] = rows(values, 2)
    cache.offset = old + len(values)
    return cache.update_index(rows(values))


def logical_values(array):
    return array.data[::array.shape[-1]]


class CapacityTests(unittest.TestCase):
    def setUp(self):
        self.ns, self.mx = load_owners()
        self.cache = self.ns["MiniMaxM3SparseCache"]()

    def assert_logical(self, expected):
        self.assertEqual(self.cache.offset, len(expected))
        self.assertEqual(self.cache._idx_offset, len(expected))
        for state in self.cache.state:
            self.assertEqual(state.shape[2], len(expected))
            self.assertEqual(logical_values(state), expected)
        self.assertEqual(logical_values(self.cache.to_cache_data()[3]), expected)

    def test_full_partial_and_rejected_d2_chains_preserve_capacity_and_history(self):
        class State:
            pass
        state = State()
        state.mtp_cache = [None, self.cache]
        expected = list(range(11))
        append(self.cache, expected)
        backing = self.cache.idx_keys
        for cycle, accepted in enumerate((2, 1, 0) * 6):
            # Depth 2 contributes one dependent-chain pair, regardless of
            # whether verification later accepts two, one or zero drafts.
            held = append(self.cache, [900 + cycle])
            state.head_chain_pairs = 1
            self.ns["_native_mtp_trim_head_chain"](state)
            self.assertEqual(state.head_chain_pairs, 0)
            self.assertIs(self.cache.idx_keys, backing)
            self.assertEqual(backing.shape[2], 256)
            self.assert_logical(expected)
            commit = [100 + cycle * 3 + index for index in range(accepted + 1)]
            current = append(self.cache, commit)
            expected.extend(commit)
            self.assertEqual(logical_values(current), expected)
            self.assertEqual(logical_values(held), expected[:-len(commit)] + [900 + cycle])
            self.assert_logical(expected)
        self.assertEqual(self.mx.concatenations, 0, "Every cycle must reuse spare capacity")

    def test_boundary_growth_then_trim_and_overwrite_never_exposes_stale_rows(self):
        expected = list(range(255))
        append(self.cache, expected)
        append(self.cache, [900, 901, 902])
        self.assertEqual(self.cache.idx_keys.shape[2], 511)
        backing = self.cache.idx_keys
        self.assertEqual(self.cache.trim(3), 3)
        self.assertIs(self.cache.idx_keys, backing)
        self.assert_logical(expected)
        append(self.cache, [-1, -2])
        self.assert_logical(expected + [-1, -2])
        self.assertEqual(self.mx.concatenations, 1)
        self.assertEqual(self.cache.trim(999), 257)
        self.assertEqual(self.cache.idx_keys.shape[2], 511)
        self.assert_logical([])
        append(self.cache, [7])
        self.assert_logical([7])
        self.assertEqual(self.mx.concatenations, 1)

    def test_noop_empty_and_overtrim_return_inherited_counts(self):
        self.assertEqual(self.cache.trim(3), 0)
        self.assertEqual(self.cache._idx_offset, 0)
        append(self.cache, [1, 2])
        backing = self.cache.idx_keys
        self.assertEqual(self.cache.trim(0), 0)
        self.assertIs(self.cache.idx_keys, backing)
        self.assert_logical([1, 2])
        self.assertEqual(self.cache.trim(20), 2)
        self.assertIs(self.cache.idx_keys, backing)
        self.assert_logical([])

    def test_derived_frontier_truncates_or_clears_on_error(self):
        class Frontier:
            nbytes = 16
            def __init__(self):
                self.offsets = []
            def truncate_to_tokens(self, offset):
                self.offsets.append(offset)
        frontier = Frontier()
        append(self.cache, [1, 2, 3])
        self.cache.derived["frontier"] = frontier
        self.cache.trim(1)
        self.assertEqual(frontier.offsets, [2])
        self.cache.trim(0)
        self.assertEqual(frontier.offsets, [2])
        self.assertEqual(self.cache.derived_nbytes, 16)
        class Broken:
            def truncate_to_tokens(self, offset):
                raise RuntimeError("invalid derived frontier")
        self.cache.derived["broken"] = Broken()
        self.cache.trim(1)
        self.assertEqual(self.cache.derived, {})

    def test_serialized_and_cloned_ownership_remains_logical_and_isolated(self):
        append(self.cache, [1, 2, 900])
        self.cache.trim(1)
        snapshot = self.cache.state
        restored = self.ns["restore_minimax_m3_sparse"](*snapshot)
        cloned = self.ns["clone_minimax_m3_sparse"](
            self.cache, copy_fn=lambda a: Array(a.shape, a.data, a.dtype)
        )
        for cache in (restored, cloned):
            self.assertEqual(cache.offset, 2)
            self.assertEqual(cache._idx_offset, 2)
            self.assertEqual(cache.idx_keys.shape[2], 2)
            self.assertEqual(cache.derived, {})
        resident = self.ns["estimate_kv_cache_memory"]([self.cache])
        serialized = self.ns["estimate_kv_cache_memory"]([{"state": snapshot}])
        self.assertEqual(resident, 256 * (2 + 2 + 3) * 4)
        self.assertEqual(serialized, 2 * (2 + 2 + 3) * 4)
        append(self.cache, [7, 8])
        self.assertEqual(logical_values(snapshot[2]), [1, 2])
        for cache in (restored, cloned):
            self.assertEqual(logical_values(cache.state[2]), [1, 2])
            current = append(cache, [6])
            self.assertEqual(logical_values(current), [1, 2, 6])
            self.assertEqual(cache._idx_offset, cache.offset)
        self.assertEqual(logical_values(snapshot[2]), [1, 2])

    def test_desynchronised_appends_still_fail_closed(self):
        append(self.cache, [1, 2])
        self.cache.trim(1)
        self.cache.offset += 2  # owner forgot one index row
        with self.assertRaisesRegex(ValueError, "desynchronised"):
            self.cache.update_index(rows([3]))


if __name__ == "__main__":
    unittest.main()
