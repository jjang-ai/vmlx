"""Text hybrid SSD completion must publish both state lanes before the next call."""
from types import SimpleNamespace
from unittest.mock import Mock
import ast
import inspect
import textwrap
import pytest
from vmlx_engine.scheduler import Scheduler

def fixture():
    s = Scheduler.__new__(Scheduler)
    s._is_hybrid = True
    s._uses_dsv4_cache = s._uses_zaya_cache = False
    s._mixed_attention_cache_model = False
    s._hybrid_kv_positions = [0]
    s._hybrid_num_layers = 2
    disk = SimpleNamespace(wait_for_write=Mock(return_value=True))
    s._ssm_state_cache = SimpleNamespace(
        disk_enabled=True, _disk=disk, has_complete=Mock(return_value=False),
        store=Mock(), _key=Mock(return_value="key"))
    s.block_aware_cache = SimpleNamespace(reconstruct_cache=Mock(return_value=["kv"]))
    s._kv_cache_bits = 0
    s._seed_cache_from_ssm_checkpoint = Mock(return_value=(["seed"], 2))
    s._prefill_for_prompt_only_cache = Mock(return_value=["kv", "ssm"])
    s._ssm_rederive_queue = [([0,1,2,3,4],5,"r"),([9],1,"other")]
    r = SimpleNamespace(request_id="r", _cache_extra_keys=None)
    return s,r

def test_persists_exact_retained_boundary_and_waits_for_its_write():
    s,r=fixture()
    s._persist_hybrid_ssd_companion(r,[0,1,2,3,4],SimpleNamespace(num_tokens=4))
    s._seed_cache_from_ssm_checkpoint.assert_called_once_with(
        r,reconstructed=["kv"],ssm_tokens=[0,1,2,3],fetch_num=4)
    s._prefill_for_prompt_only_cache.assert_called_once_with(
        [0,1,2,3],base_cache=["seed"],base_token_count=2)
    s._ssm_state_cache.store.assert_called_once_with([0,1,2,3],4,["ssm"])
    s._ssm_state_cache._disk.wait_for_write.assert_called_once_with("key",timeout=30.0)
    assert s._ssm_rederive_queue == [([9],1,"other")]

def test_existing_complete_checkpoint_does_not_recompute():
    s,r=fixture(); s._ssm_state_cache.has_complete.return_value=True
    s._persist_hybrid_ssd_companion(r,[1,2,3,4],SimpleNamespace(num_tokens=4))
    s._prefill_for_prompt_only_cache.assert_not_called()
    s._ssm_state_cache.store.assert_not_called()

@pytest.mark.parametrize("field",["_uses_dsv4_cache","_uses_zaya_cache","_mixed_attention_cache_model"])
def test_other_native_state_contracts_unchanged(field):
    s,r=fixture(); setattr(s,field,True)
    s._persist_hybrid_ssd_companion(r,[1,2,3,4],SimpleNamespace(num_tokens=4))
    s._prefill_for_prompt_only_cache.assert_not_called()

def test_disabled_ssd_is_unchanged():
    s,r=fixture(); s._ssm_state_cache.disk_enabled=False
    s._persist_hybrid_ssd_companion(r,[1,2,3,4],SimpleNamespace(num_tokens=4))
    s._prefill_for_prompt_only_cache.assert_not_called()

@pytest.mark.parametrize("failure",["derive","write","layout","media"])
def test_incomplete_companion_cannot_report_stored(failure):
    s,r=fixture()
    if failure=="derive": s._prefill_for_prompt_only_cache.return_value=None
    elif failure=="layout": s._prefill_for_prompt_only_cache.return_value=["kv"]
    elif failure=="media": r.pixel_values=object()
    else: s._ssm_state_cache._disk.wait_for_write.return_value=False
    with pytest.raises(RuntimeError):
        s._persist_hybrid_ssd_companion(r,[1,2,3,4],SimpleNamespace(num_tokens=4))

def test_cleanup_owns_companion_before_success_record():
    tree=ast.parse(textwrap.dedent(inspect.getsource(Scheduler._cleanup_finished)))
    calls=[n for n in ast.walk(tree) if isinstance(n,ast.Call)]
    stores=[n.lineno for n in calls if isinstance(n.func,ast.Attribute) and n.func.attr=="store_cache"]
    companions=[n.lineno for n in calls if isinstance(n.func,ast.Attribute) and n.func.attr=="_persist_hybrid_ssd_companion"]
    assert len(companions)==1 and any(line<companions[0] for line in stores)

def test_text_cache_discriminators_are_not_media_inputs():
    s,r=fixture(); r._cache_extra_keys={"enable_thinking":True}
    s._persist_hybrid_ssd_companion(r,[1,2,3,4],SimpleNamespace(num_tokens=4))
    s._ssm_state_cache.store.assert_called_once()
