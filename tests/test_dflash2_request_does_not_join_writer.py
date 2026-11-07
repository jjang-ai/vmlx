"""A DFlash2 request never joins the session SSD writer queue (its entries are RAM-served)."""
import inspect

from vmlx_engine import dflash2_runtime


def test_resumable_stream_does_not_wait_for_the_ssd_writer():
    source = inspect.getsource(dflash2_runtime._stream_generate_resumable)
    assert "wait_pending_writes" not in source
    assert "_q.join" not in source


def test_session_entries_are_marked_ram_served(monkeypatch):
    """VMLX_SSD_ONLY_RAM_RELEASE=0 (the RAM tier kept): every entry is served from RAM while its SSD write runs."""
    from concurrent.futures import Future

    class FakeSSD:
        def put(self, entry):
            f = Future()
            f.set_running_or_notify_cancel()
            return f

    monkeypatch.setenv("VMLX_SSD_ONLY_RAM_RELEASE", "0")
    puts = []
    monkeypatch.setattr(dflash2_runtime, "_SESSION_SSD", FakeSSD())
    monkeypatch.setattr(dflash2_runtime._SESSION_STORE, "put", lambda entry: puts.append(entry))
    receipt = dflash2_runtime._store_session({"model_key": "m", "tokens": [1], "kind": "turn"})
    assert puts and getattr(receipt, "vmlx_lookup_served_from_ram", False) is True
    assert dflash2_runtime._store_session({"model_key": "m", "tokens": [1]}, ram_only=True) is None


def test_ssd_only_text_entries_skip_the_ram_store_unless_refused(monkeypatch):
    """SSD caching (default): a prefill snapshot is not kept in RAM (its cache views would make the next append copy the
    whole live KV buffer); the next lookup waits for the receipt and restores from SSD. The finished turn stays hot. A
    write refused on the spot keeps the RAM entry; media (ram_only) entries are RAM-only as before."""
    from concurrent.futures import Future

    class FakeSSD:
        refuse = False

        def put(self, entry):
            f = Future()
            f.set_running_or_notify_cancel()
            if self.refuse:
                f.set_result({"outcome": "refused", "durable": False})
            return f

    monkeypatch.delenv("VMLX_SSD_ONLY_RAM_RELEASE", raising=False)
    ssd = FakeSSD()
    puts = []
    monkeypatch.setattr(dflash2_runtime, "_SESSION_SSD", ssd)
    monkeypatch.setattr(dflash2_runtime._SESSION_STORE, "put", lambda entry: puts.append(entry))
    receipt = dflash2_runtime._store_session({"model_key": "m", "tokens": [1], "kind": "boundary"})
    assert not puts and receipt.vmlx_lookup_served_from_ram is False
    # The finished turn is the live cache (no view to hold): kept hot for an immediate continuation, while its receipt
    # still fences the next admission (the writer drops its arrays before signalling, so no buffer is shared).
    receipt = dflash2_runtime._store_session({"model_key": "m", "tokens": [1, 9], "kind": "turn"})
    assert len(puts) == 1 and receipt.vmlx_lookup_served_from_ram is False
    ssd.refuse = True
    receipt = dflash2_runtime._store_session({"model_key": "m", "tokens": [2], "kind": "chunk"})
    assert len(puts) == 2 and receipt.vmlx_lookup_served_from_ram is True
    assert dflash2_runtime._store_session({"model_key": "m", "tokens": [3]}, ram_only=True) is None
    assert len(puts) == 3
