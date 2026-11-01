"""One device round-trip per verify cycle, with the accept decision on device.

The verify cycle reads back a single small "decision bundle" holding sampled
tokens, drafts and accept flags rather than pulling ids to the host and
comparing in Python. vMLX paid two blocking syncs
per cycle (sampled ids, then draft ids) and compared host-side with the GPU
idle -- and raw-harness timing put the engine at ~66ms/cycle against the same
model's ~55ms/cycle without engine machinery.

The decisive property here is EQUIVALENCE: the fused path must produce exactly
the accepted count the host-side comparison produces, for every prefix shape.
"""

import mlx.core as mx
import pytest

from vmlx_engine import mllm_batch_generator as gen


def _bundle(sampled_ids, draft_ids):
    sampled = mx.array(sampled_ids, dtype=mx.uint32)
    drafts = [mx.array([d], dtype=mx.uint32) for d in draft_ids]
    bundle, _ = gen._native_mtp_decision_bundle(sampled, drafts, len(draft_ids))
    mx.eval(bundle)
    return [int(v) for v in bundle.tolist()]


def _accepted(sampled_ids, draft_ids):
    flat = _bundle(sampled_ids, draft_ids)
    return flat[-1]


def _host_accepted(sampled_ids, draft_ids):
    """The original host-side comparison, as the reference."""
    n = 0
    for idx, d in enumerate(draft_ids):
        if int(sampled_ids[idx]) != int(d):
            break
        n += 1
    return n


class TestBundleLayout:
    def test_layout_is_sampled_then_drafts_then_count(self):
        flat = _bundle([7, 8], [7])
        assert flat[:2] == [7, 8]      # sampled, depth+1 entries
        assert flat[2:3] == [7]        # drafts, depth entries
        assert flat[3] == 1            # accepted count

    def test_depth_three_layout(self):
        flat = _bundle([1, 2, 3, 4], [1, 2, 9])
        assert flat[:4] == [1, 2, 3, 4]
        assert flat[4:7] == [1, 2, 9]
        assert flat[7] == 2


class TestAcceptedCount:
    def test_all_match(self):
        assert _accepted([5, 6, 7, 8], [5, 6, 7]) == 3

    def test_none_match(self):
        assert _accepted([5, 6, 7, 8], [9, 9, 9]) == 0

    def test_leading_run_only(self):
        """A later match after a mismatch must NOT be counted."""
        assert _accepted([1, 2, 3, 4], [1, 9, 3]) == 1

    def test_single_draft_match(self):
        assert _accepted([42, 43], [42]) == 1

    def test_single_draft_mismatch(self):
        assert _accepted([42, 43], [41]) == 0


class TestEquivalenceWithHostPath:
    """The fused device decision must equal the host comparison exactly."""

    @pytest.mark.parametrize(
        "sampled,drafts",
        [
            ([1, 2], [1]),
            ([1, 2], [2]),
            ([1, 2, 3], [1, 2]),
            ([1, 2, 3], [1, 9]),
            ([1, 2, 3], [9, 2]),
            ([1, 2, 3, 4], [1, 2, 3]),
            ([1, 2, 3, 4], [1, 2, 9]),
            ([1, 2, 3, 4], [1, 9, 3]),
            ([1, 2, 3, 4], [9, 2, 3]),
            ([0, 0, 0, 0], [0, 0, 0]),
            ([248319, 5, 6, 7], [248319, 5, 6]),
        ],
    )
    def test_matches_host_comparison(self, sampled, drafts):
        assert _accepted(sampled, drafts) == _host_accepted(sampled, drafts)


def test_large_token_ids_survive_the_roundtrip():
    """Vocab is 248320 - ids must not be truncated by the uint32 packing."""
    flat = _bundle([248319, 248318], [248319])
    assert flat[0] == 248319
    assert flat[2] == 248319
    assert flat[-1] == 1
