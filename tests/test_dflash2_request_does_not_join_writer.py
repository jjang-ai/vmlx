"""A DFlash2 request never joins the session SSD writer queue (its entries are RAM-served)."""
import inspect

from vmlx_engine import dflash2_runtime


def test_resumable_stream_does_not_wait_for_the_ssd_writer():
    source = inspect.getsource(dflash2_runtime._stream_generate_resumable)
    assert "wait_pending_writes" not in source
    assert "_q.join" not in source


def test_session_entries_are_marked_ram_served(monkeypatch):
    from concurrent.futures import Future

    class FakeSSD:
        def put(self, entry):
            f = Future()
            f.set_running_or_notify_cancel()
            return f

    puts = []
    monkeypatch.setattr(dflash2_runtime, "_SESSION_SSD", FakeSSD())
    monkeypatch.setattr(dflash2_runtime._SESSION_STORE, "put", lambda entry: puts.append(entry))
    receipt = dflash2_runtime._store_session({"model_key": "m", "tokens": [1], "kind": "turn"})
    assert puts and getattr(receipt, "vmlx_lookup_served_from_ram", False) is True
    assert dflash2_runtime._store_session({"model_key": "m", "tokens": [1]}, ram_only=True) is None
