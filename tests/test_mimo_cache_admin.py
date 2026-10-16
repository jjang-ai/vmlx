"""MiMo DFlash pool administration must address its own session store."""
import asyncio
import os
from types import SimpleNamespace

import pytest


@pytest.mark.asyncio
@pytest.mark.parametrize("condition", ["idle", "busy", "wrong_pid", "wrong_root", "pending_write"])
async def test_mimo_pool_clear_owns_store_and_preserves_busy_entries(tmp_path, monkeypatch, condition):
    from vmlx_engine import server, speculative, mimo_session
    from vmlx_engine.utils.omni_session_disk_store import OmniSessionDiskStore

    store = OmniSessionDiskStore(root=tmp_path, model_key="mimo-admin", max_size_bytes=65536)
    lock = asyncio.Lock()
    flushed = []

    def flush(timeout):
        assert lock.locked()
        flushed.append(True)
        return condition != "pending_write"

    async def worker(fn):
        assert lock.locked()
        return fn()

    async def wrong_owner(*args, **kwargs):
        pytest.fail("MiMo clear reached the Qwen cache owner")

    monkeypatch.setattr(server, "_engine", SimpleNamespace(_generation_lock=lock, _run_model_call=worker))
    monkeypatch.setattr(server, "_current_model_config", lambda: SimpleNamespace(family_name="mimo_v2"))
    monkeypatch.setattr(speculative, "is_dflash2_enabled", lambda: True)
    monkeypatch.setattr(server, "_clear_dflash2_shared_ssd_pool", wrong_owner)
    monkeypatch.setattr(mimo_session, "SSD", SimpleNamespace(store=store, flush=flush))
    resident = {"preserve": object()}
    monkeypatch.setattr(mimo_session, "STORE", resident)
    try:
        store.save("entry", lambda path: path.write_bytes(b"mimo-state"))
        (tmp_path / "NOTES.md").write_text("preserve")
        root = str(tmp_path) if condition != "wrong_root" else str(tmp_path / "other")
        pid = os.getpid() + (condition == "wrong_pid")
        if condition == "busy":
            await lock.acquire()
        if condition == "idle":
            result = await server.clear_cache("ssd_pool", root, pid)
            assert result["freed_bytes"] > 0 and result["remaining_bytes"] == 0
            assert result["resident_cache_preserved"] and resident
            assert flushed == [True]
        else:
            with pytest.raises(server.HTTPException) as exc:
                await server.clear_cache("ssd_pool", root, pid)
            assert exc.value.status_code == 409
            assert store.load("entry", lambda path: path.read_bytes()) == b"mimo-state"
        assert (tmp_path / "NOTES.md").read_text() == "preserve"
    finally:
        if lock.locked():
            lock.release()
        store.close()
