"""Failed recurrent rollback leaves the replay fallback its full KV state."""
import pytest


class _KV:
    def __init__(self):
        self.offset = 100

    def is_trimmable(self):
        return True

    def trim(self, n):
        self.offset -= n


class _Recurrent:
    def __init__(self, failure=None):
        self.cache = ["before-conv", "before-ssm"]
        self.rollback_state = ("confirmed-conv", "confirmed-ssm")
        self.failure = failure

    def rollback_to(self, accepted):
        if self.failure == "throw":
            raise ValueError("injected late rollback failure")
        if self.failure == "malformed":
            return ("only-one-state",)
        return ("accepted-conv", "accepted-ssm")

    def __setitem__(self, index, value):
        self.cache[index] = value

    def is_trimmable(self):
        return False


@pytest.mark.parametrize("failure", ["throw", "malformed"])
def test_late_rollback_failure_does_not_double_trim_kv(failure):
    from vmlx_engine.mllm_batch_generator import (
        _native_mtp_snapshot_replay_cache, _native_mtp_restore_replay_cache,
        _native_mtp_rollback_to_confirmed_impl,
    )
    kv1, kv2 = _KV(), _KV()
    early, late = _Recurrent(), _Recurrent(failure)
    cache = [kv1, early, kv2, late]
    snapshot = _native_mtp_snapshot_replay_cache(cache, advance_len=4)
    kv1.offset = kv2.offset = 104
    early.cache = late.cache = ["verified-conv", "verified-ssm"]
    assert not _native_mtp_rollback_to_confirmed_impl(cache, 2, 1)
    assert (kv1.offset, kv2.offset) == (104, 104)
    assert _native_mtp_restore_replay_cache(cache, snapshot, 4)
    assert (kv1.offset, kv2.offset) == (100, 100)
    assert early.cache == late.cache == ["before-conv", "before-ssm"]


def test_successful_rollback_keeps_accepted_state_and_trims_once():
    from vmlx_engine.mllm_batch_generator import _native_mtp_rollback_to_confirmed_impl
    kv, recurrent = _KV(), _Recurrent()
    kv.offset = 104
    assert _native_mtp_rollback_to_confirmed_impl([kv, recurrent], 2, 1)
    assert kv.offset == 102
    assert recurrent.cache == ["accepted-conv", "accepted-ssm"]
    assert recurrent.rollback_state is None and recurrent.rollback_to is None
