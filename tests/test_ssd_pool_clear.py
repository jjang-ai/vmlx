"""The new explicit SSD action cannot silently clear an active or replaced engine."""
import asyncio
import os
from types import SimpleNamespace
from unittest.mock import Mock

import pytest


@pytest.fixture
def context(monkeypatch):
    import vmlx_engine.server as server
    engine_stats = {"engine_collector_request_ids": [], "engine_collector_count": 0,
                    "terminal_cleanup_pending": False}
    scheduler_stats = {"waiting_request_ids": [], "num_waiting": 0,
                       "running_request_ids": [], "num_running": 0}
    result = SimpleNamespace(evicted_entries=2, evicted_bytes=123, bytes_after=50,
                             max_size_bytes=1000, protected_temp_files=0, protected_recent_orphans=0)
    budget = SimpleNamespace(root="/isolated/cache", clear_eligible=Mock(return_value=result))
    scheduler = SimpleNamespace(get_stats=lambda: scheduler_stats,
        paged_cache_manager=SimpleNamespace(_disk_store=SimpleNamespace(global_budget=budget)))
    monkeypatch.setattr(server, "_engine", SimpleNamespace(get_stats=lambda: engine_stats))
    monkeypatch.setattr(server, "_get_scheduler", lambda: scheduler)
    return server, engine_stats, scheduler_stats, budget


@pytest.mark.parametrize("state", ["running", "waiting", "collector", "fence", "unknown", "wrong_root", "wrong_pid"])
def test_busy_or_changed_identity_never_clears(context, state):
    server, engine, scheduler, budget = context
    root, pid = str(budget.root), os.getpid()
    if state in ("running", "waiting"):
        scheduler[state + "_request_ids"] = ["request-1"]
        scheduler["num_" + state] = 1
        if state == "running": scheduler["running_requests"] = [{"request_id": "request-1", "status": "running"}]
    elif state == "collector":
        engine["engine_collector_request_ids"] = ["request-1"]; engine["engine_collector_count"] = 1
    elif state == "fence": engine["terminal_cleanup_pending"] = True
    elif state == "unknown": engine.clear()
    elif state == "wrong_root": root = "/different/cache"
    elif state == "wrong_pid": pid += 1
    with pytest.raises(server.HTTPException) as exc:
        asyncio.run(server.clear_cache("ssd_pool", root, pid))
    assert exc.value.status_code == 409
    budget.clear_eligible.assert_not_called()


def test_clear_reports_actual_bytes_not_empty_directory(context):
    server, _, _, budget = context
    response = asyncio.run(server.clear_cache("ssd_pool", str(budget.root), os.getpid()))
    assert response["status"] == "eligible_cleared"
    assert response["remaining_bytes"] == 50
    assert response["effective_cap_bytes"] == 1000
    assert response["resident_cache_preserved"] is True


def test_other_engine_refusal_is_typed_and_not_success(context):
    server, _, _, budget = context
    budget.clear_eligible.side_effect = BlockingIOError("other owner")
    with pytest.raises(server.HTTPException) as exc:
        asyncio.run(server.clear_cache("ssd_pool", str(budget.root), os.getpid()))
    assert exc.value.status_code == 409
