"""Tests for critical cache fixes — memory leak, thread safety, locking."""
import pytest
import threading
import tempfile
import os


class TestSchedulerRequestsCleanup:
    """Verify self.requests is cleaned in _cleanup_finished."""

    def test_requests_pop_is_present_in_cleanup(self):
        """_cleanup_finished source contains self.requests.pop."""
        import inspect
        from vmlx_engine.scheduler import Scheduler
        source = inspect.getsource(Scheduler._cleanup_finished)
        assert "self.requests.pop(" in source, "requests.pop should be in _cleanup_finished"

    def test_abort_request_handles_missing(self):
        """abort_request safely handles already-removed requests."""
        import inspect
        from vmlx_engine.scheduler import Scheduler
        source = inspect.getsource(Scheduler.abort_request)
        assert ".pop(request_id, None)" in source or ".get(request_id)" in source


class TestDiskCacheThreadSafety:
    """Verify SQLite check_same_thread=False."""

    def test_connection_created_with_cross_thread_flag(self):
        """Connection pool uses check_same_thread=False."""
        import inspect
        import vmlx_engine.disk_cache as dc_mod
        source = inspect.getsource(dc_mod)
        assert "check_same_thread=False" in source

    def test_cross_thread_sqlite_works(self):
        """SQLite connection with check_same_thread=False works across threads."""
        import sqlite3
        with tempfile.TemporaryDirectory() as tmp:
            db_path = os.path.join(tmp, "test.db")
            conn = sqlite3.connect(db_path, check_same_thread=False)
            conn.execute("CREATE TABLE t (x INTEGER)")
            conn.execute("INSERT INTO t VALUES (1)")
            conn.commit()

            result = [None]
            def read_from_thread():
                try:
                    row = conn.execute("SELECT x FROM t").fetchone()
                    result[0] = row[0]
                except Exception as e:
                    result[0] = str(e)

            t = threading.Thread(target=read_from_thread)
            t.start()
            t.join()
            assert result[0] == 1
            conn.close()


class TestMemoryCacheRemoveLock:
    """Verify remove() method has proper locking."""

    def test_remove_source_has_lock(self):
        """remove() should use self._lock."""
        import inspect
        from vmlx_engine.memory_cache import MemoryAwarePrefixCache
        source = inspect.getsource(MemoryAwarePrefixCache.remove)
        assert "self._lock" in source, "remove() should acquire self._lock"


class TestStreamingTimeoutSupport:
    """Verify _stream_with_keepalive accepts total_timeout."""

    def test_keepalive_has_total_timeout_param(self):
        """_stream_with_keepalive signature includes total_timeout."""
        import inspect
        from vmlx_engine.server import _stream_with_keepalive
        sig = inspect.signature(_stream_with_keepalive)
        assert 'total_timeout' in sig.parameters, \
            f"Missing total_timeout param. Params: {list(sig.parameters.keys())}"
