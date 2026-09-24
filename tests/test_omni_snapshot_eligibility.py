"""Native checkpoints must represent the next template's actual prefix."""
import threading
from types import SimpleNamespace
import pytest
from vmlx_engine.omni_multimodal import OmniMultimodalDispatcher


@pytest.mark.parametrize("thinking,reply,finish,should_store", [
    (False, "Answer.", "stop", True),
    (True, "Inspect the frames.</think>Answer.", "stop", False),
    (None, "Inspect the frames.</think>Answer.", "stop", False),
    (True, "</think>Answer.", "stop", False),
    (False, "Unfinished answer", "length", False),
    (True, "Unfinished reasoning", "length", False),
])
def test_only_replayable_completed_prefix_is_published(tmp_path, thinking, reply, finish, should_store):
    events=[]
    class Session:
        _cache=None
        def reset(self):
            self._cache=None
            events.append('released')
        def turn(self, **kwargs):
            self._cache=[object()]
            self._last_finish_reason=finish
            return reply
    d=OmniMultimodalDispatcher.__new__(OmniMultimodalDispatcher)
    d._session=Session();d._backend='stage1';d._lock=threading.Lock()
    d._last_signature=None;d._scratch_dir=tmp_path;d._disk_cache_enabled=True
    d._try_restore_session_snapshot=lambda signature: False
    d._persist_session_snapshot=lambda: events.append('stored') or True
    result=d.chat([{'role':'user','content':'Inspect.'}],enable_thinking=thinking)
    events.clear()
    d.finish_request_cache()
    assert events==(['stored','released'] if should_store else ['released'])
    assert result['finish_reason']==finish
    assert d._session._cache is None and d._last_signature is None


def test_old_checkpoint_schema_is_not_silently_reused(tmp_path):
    import mlx.core as mx
    from mlx_lm.models.cache import ArraysCache,save_prompt_cache
    from vmlx_engine.utils.omni_session_disk_store import OmniSessionDiskStore
    state=ArraysCache(1);state[0]=mx.ones((1,2,3))
    store=OmniSessionDiskStore(root=tmp_path,model_key='bundle',max_size_bytes=10000000)
    store.save('prefix',lambda p:save_prompt_cache(str(p),[state],{
        'schema':'nemotron_omni_session_v1','bundle_fingerprint':'bundle','signature':'prefix','history_json':'[]'}))
    d=OmniMultimodalDispatcher.__new__(OmniMultimodalDispatcher)
    d._disk_cache_enabled=True;d._backend='stage1';d._last_signature=None
    d._session_l2_fingerprint='bundle';d._session_l2_store=store
    d._session_l2_stats={'hits':0,'misses':0,'last_error':None}
    d._session=SimpleNamespace(_cache=None,mlx_model=SimpleNamespace(backbone=SimpleNamespace(layers=[SimpleNamespace(block_type='M')])))
    assert d._try_restore_session_snapshot('prefix') is False
    assert d._session._cache is None
    assert d._session_l2_stats['hits']==0


def test_stage2_session_is_not_subject_to_stage1_snapshot_policy(tmp_path):
    class Session:
        def reset(self): pass
        def turn(self, **kwargs):
            self._last_finish_reason='length'
            return 'Thinking'
    d=OmniMultimodalDispatcher.__new__(OmniMultimodalDispatcher)
    d._session=Session();d._backend='stage2';d._lock=threading.Lock()
    d._last_signature=None;d._scratch_dir=tmp_path
    d.chat([{'role':'user','content':'Inspect.'}],enable_thinking=True)
    assert d._last_signature is not None
    assert d._last_snapshot_skip_reason is None
