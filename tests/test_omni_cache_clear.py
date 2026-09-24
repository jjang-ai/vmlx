import asyncio
from types import SimpleNamespace

import pytest

from vmlx_engine.omni_multimodal import OmniMultimodalDispatcher
from vmlx_engine.utils.omni_session_disk_store import OmniSessionDiskStore


@pytest.mark.parametrize('cache_type', ['prefix', 'all'])
def test_server_clear_includes_native_snapshots_before_native_model_load(tmp_path, monkeypatch, cache_type):
    from vmlx_engine import server
    bundle=tmp_path/'bundle';bundle.mkdir()
    policy={'root':str(tmp_path/'pool'),'max_size_bytes':1000000,'ttl_minutes':0.0}
    store=OmniSessionDiskStore(root=policy['root'],model_key=OmniMultimodalDispatcher._bundle_fingerprint(bundle),max_size_bytes=1000000)
    try:
        snapshot=store.save('history',lambda p:p.write_bytes(b'native cached tensors'))
        monkeypatch.setattr(server,'_model_path',str(bundle))
        monkeypatch.setattr(server,'_engine',None)
        monkeypatch.setattr(server,'_get_scheduler',lambda:None)
        monkeypatch.setattr(server,'_loaded_omni_disk_cache_policy',lambda:policy)
        monkeypatch.setattr('vmlx_engine.omni_multimodal.is_omni_multimodal_bundle',lambda p:True)
        monkeypatch.setattr(OmniMultimodalDispatcher,'_instance',None)
        result=asyncio.run(server.clear_cache(cache_type))
        assert not snapshot.exists(),result
        assert 'omni_session_disk' in result['caches']
        assert not result.get('skipped'),result
    finally:store.close()


def test_native_clear_preserves_other_model_and_ram_only_request(tmp_path,monkeypatch):
    from vmlx_engine import server
    bundle=tmp_path/'bundle';bundle.mkdir()
    policy={'root':str(tmp_path/'pool'),'max_size_bytes':1000000,'ttl_minutes':0.0}
    a=OmniSessionDiskStore(root=policy['root'],model_key=OmniMultimodalDispatcher._bundle_fingerprint(bundle),max_size_bytes=1000000)
    b=OmniSessionDiskStore(root=policy['root'],model_key='unrelated',max_size_bytes=1000000)
    try:
        pa=a.save('mine',lambda p:p.write_bytes(b'one'));pb=b.save('other',lambda p:p.write_bytes(b'two'))
        monkeypatch.setattr(server,'_model_path',str(bundle));monkeypatch.setattr(server,'_engine',None)
        monkeypatch.setattr(server,'_get_scheduler',lambda:None)
        monkeypatch.setattr(server,'_loaded_omni_disk_cache_policy',lambda:policy)
        monkeypatch.setattr('vmlx_engine.omni_multimodal.is_omni_multimodal_bundle',lambda p:True)
        monkeypatch.setattr(OmniMultimodalDispatcher,'_instance',None)
        asyncio.run(server.clear_cache('ram'));assert pa.exists() and pb.exists()
        asyncio.run(server.clear_cache('prefix'));assert not pa.exists() and pb.exists()
    finally:a.close();b.close()


@pytest.mark.asyncio
async def test_native_clear_waits_for_queued_publication(tmp_path,monkeypatch):
    import threading
    from concurrent.futures import ThreadPoolExecutor
    owner=ThreadPoolExecutor(max_workers=1);release=threading.Event();started=threading.Event()
    policy={'root':str(tmp_path),'max_size_bytes':1000000,'ttl_minutes':0.0}
    store=OmniSessionDiskStore(root=tmp_path,model_key='test',max_size_bytes=1000000)
    bundle=str((tmp_path/'bundle').resolve());events=[]
    instance=SimpleNamespace(bundle_path=bundle,_session_l2_policy=policy,submit=owner.submit)
    def clear():
        events.append('clear');return store.clear()
    instance._clear_native_disk_cache=clear
    monkeypatch.setattr(OmniMultimodalDispatcher,'_instance',instance)
    def write():
        started.set();assert release.wait(5)
        store.save('request',lambda p:p.write_bytes(b'cached'));events.append('written')
    pending=owner.submit(write);assert started.wait(2)
    task=asyncio.create_task(OmniMultimodalDispatcher.clear_disk_cache_for(bundle,disk_cache_policy=policy))
    try:
        await asyncio.sleep(.02);assert not task.done()
        release.set();assert await asyncio.wait_for(task,3)>0
        assert events==['written','clear']
        assert not any(p.exists() for p in store.paths('request'))
    finally:
        release.set();await asyncio.gather(task,return_exceptions=True);owner.shutdown(wait=True);store.close()


def test_clear_error_is_reported_as_skipped_tier(monkeypatch):
    from vmlx_engine import server
    monkeypatch.setattr(server,'_model_path','/not-loaded')
    monkeypatch.setattr(server,'_get_scheduler',lambda:None)
    monkeypatch.setattr('vmlx_engine.omni_multimodal.is_omni_multimodal_bundle',lambda p:True)
    async def fail(*a,**k):raise OSError('cache root unavailable')
    monkeypatch.setattr(OmniMultimodalDispatcher,'clear_disk_cache_for',fail)
    result=asyncio.run(server.clear_cache('prefix'))
    assert result['status']=='clear_failed'
    assert result['skipped']==['omni_session_disk:clear_failed']
    assert 'omni_session_disk' not in result.get('caches',[])
