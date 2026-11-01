"""Native GLM cache administration must not address the Qwen SSD namespace."""
import asyncio
import threading
from collections import OrderedDict
from types import SimpleNamespace

import pytest

pytest.importorskip('mlx.core')
from vmlx_engine import glm_dflash2 as runtime


def test_stats_are_native_metadata_with_shared_budget(tmp_path, monkeypatch):
    disk = runtime.PrefixDiskTier(tmp_path, 65536)
    try:
        disk._catalog.save('a' * 40, 3, lambda path: path.write_bytes(b'x' * 123))
        disk._load_index()
        disk.writes = 2
        disk.hits = 1
        store = SimpleNamespace(entries=OrderedDict([((1, 2, 3), object())]), max_entries=12, disk=disk)
        monkeypatch.setattr(runtime, '_PREFIX_STORES', {1: store})
        monkeypatch.setattr(runtime, '_PREFIX_DISK', {'root': str(tmp_path), 'max_bytes': 65536})
        stats = runtime.session_cache_stats()
        assert stats['ram']['tokens'] == 3
        assert stats['ram']['entries'] == 1
        assert stats['ssd']['writes'] == 2
        assert stats['ssd']['hits'] == 1
        assert stats['ssd']['global_budget']['bytes_after'] >= 123
        assert stats['ssd']['global_budget']['root'] == str(tmp_path)
        assert stats['ssd']['global_budget']['compliant'] is True
        assert stats['ssd']['shared_pool'] is True
        assert 'bundle_budget' not in stats['ssd']
    finally:
        disk.close()


def test_clear_drains_writer_and_only_removes_owned_snapshots(tmp_path, monkeypatch):
    entered, release = threading.Event(), threading.Event()
    def write(self, name, key, snap):
        entered.set()
        assert release.wait(3)
        self._catalog.save(name, len(key), lambda path: path.write_bytes(b'owned'))
        self._load_index()
        return {'durable': True}
    monkeypatch.setattr(runtime.PrefixDiskTier, '_write', write)
    disk = runtime.PrefixDiskTier(tmp_path, 65536)
    (tmp_path / 'NOTES.md').write_text('preserve')
    receipt = disk.put((1, 2), None)
    assert entered.wait(1)
    result = []
    thread = threading.Thread(target=lambda: result.append(disk.clear()))
    thread.start()
    try:
        thread.join(.05)
        assert thread.is_alive()
    finally:
        release.set()
        thread.join(3)
        disk.close()
    assert receipt.result()['durable']
    assert result == [1]
    assert disk.index == {}
    assert not list(disk.dir.glob('*.safetensors'))
    assert (tmp_path / 'NOTES.md').read_text() == 'preserve'


def test_ram_clear_preserves_disk(monkeypatch):
    disk = SimpleNamespace(clear=lambda: pytest.fail('RAM clear touched SSD'))
    store = SimpleNamespace(entries={1: object()}, disk=disk)
    monkeypatch.setattr(runtime, '_PREFIX_STORES', {1: store})
    assert runtime.clear_sessions(False) == 0
    assert store.entries == {}


@pytest.mark.parametrize('include_ssd', [False, True])
def test_clear_before_first_request_after_restart(tmp_path, monkeypatch, include_ssd):
    from vmlx_engine.utils.glm_prefix_disk_store import GlmPrefixDiskStore

    owned = GlmPrefixDiskStore(tmp_path, 'loaded-bundle', 65536)
    neighbor = GlmPrefixDiskStore(tmp_path, 'other-bundle', 65536)
    try:
        owned.save('a' * 40, 3, lambda path: path.write_bytes(b'owned'))
        neighbor.save('b' * 40, 3, lambda path: path.write_bytes(b'neighbor'))
        owned.close()
        # Startup configures the disk namespace, but opens no PrefixStore
        # until the first inference request.
        monkeypatch.setattr(runtime, '_PREFIX_STORES', {})
        monkeypatch.setattr(runtime, '_PREFIX_DISK', {
            'root': str(tmp_path), 'max_bytes': 65536,
            'model_key': 'loaded-bundle',
        })
        assert runtime.clear_sessions(include_ssd) == int(include_ssd)
        reopened = GlmPrefixDiskStore(tmp_path, 'loaded-bundle', 65536)
        try:
            assert reopened.contains('a' * 40, 3) is (not include_ssd)
            assert neighbor.contains('b' * 40, 3)
            assert runtime._PREFIX_STORES == {}
        finally:
            reopened.close()
    finally:
        owned.close()
        neighbor.close()


def test_loaded_bundle_selects_native_cache_owner(monkeypatch):
    from vmlx_engine import server, dflash2_runtime
    for family in ('glm5_next', 'glm5_next_text'):
        monkeypatch.setattr(server, '_current_model_config', lambda: SimpleNamespace(family_name=family))
        assert server._active_dflash2_cache_runtime() is runtime
    monkeypatch.setattr(server, '_current_model_config', lambda: SimpleNamespace(family_name='qwen3_5'))
    assert server._active_dflash2_cache_runtime() is dflash2_runtime


@pytest.mark.asyncio
async def test_native_clear_excludes_generation_and_does_not_clear_busy_engine(monkeypatch):
    from vmlx_engine import server
    lock = asyncio.Lock()
    calls = []
    monkeypatch.setattr(runtime, 'clear_sessions', lambda include_ssd: calls.append(include_ssd))
    async def worker(fn):
        assert lock.locked()
        fn()
    monkeypatch.setattr(server, '_engine', SimpleNamespace(_generation_lock=lock, _run_model_call=worker))
    async with lock:
        assert await server._clear_glm_dflash2_cache(True) is False
    assert calls == []
    assert await server._clear_glm_dflash2_cache(True) is True
    assert calls == [True]
    assert not lock.locked()

@pytest.mark.asyncio
async def test_shared_pool_clear_drains_and_preserves_native_ram(tmp_path, monkeypatch):
    import os
    from vmlx_engine import server, speculative
    disk = runtime.PrefixDiskTier(tmp_path, 65536)
    resident = {(1, 2): object()}
    try:
        disk._catalog.save('b' * 40, 2, lambda path: path.write_bytes(b'owned'))
        disk._load_index()
        monkeypatch.setattr(runtime, '_PREFIX_STORES', {1: SimpleNamespace(entries=resident, disk=disk)})
        lock = asyncio.Lock()
        calls = []
        original_flush = disk.flush
        def flush():
            assert lock.locked()
            calls.append('drained')
            original_flush()
        monkeypatch.setattr(disk, 'flush', flush)
        async def worker(fn):
            assert lock.locked()
            return fn()
        monkeypatch.setattr(server, '_engine', SimpleNamespace(_generation_lock=lock, _run_model_call=worker))
        monkeypatch.setattr(server, '_current_model_config', lambda: SimpleNamespace(family_name='glm5_next'))
        monkeypatch.setattr(speculative, 'is_dflash2_enabled', lambda: True)
        async with lock:
            with pytest.raises(server.HTTPException) as exc:
                await server.clear_cache('ssd_pool', str(tmp_path), os.getpid())
            assert exc.value.status_code == 409
        with pytest.raises(server.HTTPException) as exc:
            await server.clear_cache('ssd_pool', str(tmp_path), os.getpid()+1)
        assert exc.value.status_code == 409
        assert disk._catalog.contains('b' * 40, 2)
        result = await server.clear_cache('ssd_pool', str(tmp_path), os.getpid())
        assert result['status'] == 'eligible_cleared'
        assert result['freed_bytes'] > 0 and result['remaining_bytes'] == 0
        assert result['resident_cache_preserved'] and resident
        assert calls == ['drained']
        assert disk.index == {}
    finally:
        disk.close()

@pytest.mark.asyncio
async def test_shared_pool_clear_keeps_admission_locked_on_disconnect(monkeypatch):
    from vmlx_engine import server
    lock = asyncio.Lock()
    entered, release = asyncio.Event(), asyncio.Event()
    async def worker(fn):
        entered.set()
        await release.wait()
        return {'status': 'eligible_cleared'}
    monkeypatch.setattr(server, '_engine', SimpleNamespace(_generation_lock=lock, _run_model_call=worker))
    task = asyncio.create_task(server._clear_glm_shared_ssd_pool('/unused', 1))
    await entered.wait()
    task.cancel()
    await asyncio.sleep(0)
    assert lock.locked() and not task.done()
    release.set()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert not lock.locked()
