"""Request-local speculative state for the patched Qwen GatedDeltaNet.

Only the owning layer opts a cache instance in. Nothing is added to serialized
cache.state, and unrelated ArraysCache families do not gain this capability.
"""
from types import MethodType


def _can_rollback(cache, count):
    record = getattr(cache, "_vmlx_qwen_rollback", None)
    return record is not None and isinstance(count, int) and 0 < count <= record[0]


def _commit(cache):
    cache._vmlx_qwen_rollback = None
    cache.rollback_state = None


def _rollback(cache, count):
    if not _can_rollback(cache, count):
        return False
    drafts, restore = cache._vmlx_qwen_rollback
    # Compute before mutating this cache. Errors are fatal to the request;
    # a caller must never resume AR from a partially restored layer set.
    conv, ssm = restore(drafts - count)
    cache[0], cache[1] = conv, ssm
    cache.advance(-count)
    _commit(cache)
    return True


def prepare_qwen_rollback(cache):
    """Called on every owning-layer forward, including after SSD restore."""
    cache.supports_partial_rollback = True
    cache.can_rollback_speculative = MethodType(_can_rollback, cache)
    cache.rollback_speculative = MethodType(_rollback, cache)
    cache.commit_speculative = MethodType(_commit, cache)
    _commit(cache)


def record_qwen_rollback(cache, drafts, restore):
    cache._vmlx_qwen_rollback = (drafts, restore)
