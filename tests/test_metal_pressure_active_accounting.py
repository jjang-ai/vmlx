"""Active Metal bytes already exclude the allocator's unused free-list."""
import asyncio
from unittest.mock import Mock

import mlx.core as mx
import pytest
from fastapi import HTTPException


@pytest.mark.parametrize('clear_raises', [False, True])
def test_unused_allocator_cache_is_not_subtracted_from_active(monkeypatch, clear_raises):
    from vmlx_engine import server
    from vmlx_engine.utils import memory_limits
    monkeypatch.setattr(memory_limits, 'is_metal_ws_guard_enabled', lambda: True)
    monkeypatch.setattr(memory_limits, 'get_metal_ws_guard_threshold', lambda _d: 99.0)
    monkeypatch.setattr(memory_limits, 'get_effective_metal_working_set_bytes', lambda _mx: (1000, 1000))
    monkeypatch.setattr(server, '_model_path', 'pressure-accounting-fixture')
    monkeypatch.setattr(server, '_model_name', 'pressure-accounting-fixture')
    monkeypatch.setattr(server, '_engine', None)
    monkeypatch.setattr(server, '_last_metal_ws_log', 0)
    monkeypatch.setattr(mx, 'get_cache_memory', lambda: 200)
    clear = Mock(side_effect=RuntimeError('controlled reclaim failure') if clear_raises else None)
    monkeypatch.setattr(mx, 'clear_cache', clear)
    with pytest.raises(HTTPException) as exc:
        asyncio.run(server.check_metal_working_set_pressure(None))
    assert exc.value.status_code == 503
    assert exc.value.headers['Retry-After'] == '5'
    clear.assert_called_once()


def test_actual_active_reduction_after_reclaim_is_respected(monkeypatch):
    from vmlx_engine import server
    from vmlx_engine.utils import memory_limits
    monkeypatch.setattr(memory_limits, 'is_metal_ws_guard_enabled', lambda: True)
    monkeypatch.setattr(memory_limits, 'get_metal_ws_guard_threshold', lambda _d: 99.0)
    readings = iter([(1000, 1000), (900, 1000)])
    monkeypatch.setattr(memory_limits, 'get_effective_metal_working_set_bytes', lambda _mx: next(readings))
    monkeypatch.setattr(server, '_model_path', 'pressure-accounting-fixture')
    monkeypatch.setattr(server, '_model_name', 'pressure-accounting-fixture')
    monkeypatch.setattr(server, '_engine', None)
    clear = Mock()
    monkeypatch.setattr(mx, 'clear_cache', clear)
    asyncio.run(server.check_metal_working_set_pressure(None))
    clear.assert_called_once()
