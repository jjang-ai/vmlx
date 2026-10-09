# SPDX-License-Identifier: Apache-2.0
"""Sleep endpoints must refuse to tear down caches while requests are live.

Observed live (2026-08-07): auto-sleep measured idle from request *submission*,
so a >10min DSV4 reasoning turn triggered /admin/soft-sleep mid-generation.
The deep_reset() destroyed the running request's KV state, the stream stalled
silently, and the client timed out after 900s with empty content.
"""

import asyncio
from unittest.mock import MagicMock, patch

from vmlx_engine import server


def _run(coro):
    # Python 3.13 no longer creates a main-thread event loop implicitly.
    return asyncio.run(coro)


def _scheduler(num_running=0, num_waiting=0):
    scheduler = MagicMock()
    scheduler.get_stats.return_value = {
        "num_running": num_running,
        "num_waiting": num_waiting,
    }
    return scheduler


def test_soft_sleep_refuses_while_request_running():
    scheduler = _scheduler(num_running=1)
    with (
        patch.object(server, "_standby_state", "active"),
        patch.object(server, "_get_scheduler", return_value=scheduler),
        patch("vmlx_engine.utils.gpu_keepalive.stop_global") as stop_heartbeat,
    ):
        res = _run(server.admin_soft_sleep())

    assert res.status_code == 409
    assert b'"busy"' in res.body
    scheduler.deep_reset.assert_not_called()
    stop_heartbeat.assert_not_called()


def test_deep_sleep_refuses_while_request_waiting():
    scheduler = _scheduler(num_waiting=2)
    with (
        patch.object(server, "_standby_state", "active"),
        patch.object(server, "_get_scheduler", return_value=scheduler),
        patch("vmlx_engine.utils.gpu_keepalive.stop_global") as stop_heartbeat,
    ):
        res = _run(server.admin_deep_sleep())

    assert res.status_code == 409
    assert b'"busy"' in res.body
    scheduler.deep_reset.assert_not_called()
    stop_heartbeat.assert_not_called()


def test_soft_sleep_proceeds_when_idle():
    scheduler = _scheduler()
    with (
        patch.object(server, "_standby_state", "active"),
        patch.object(server, "_get_scheduler", return_value=scheduler),
        patch("vmlx_engine.utils.gpu_keepalive.stop_global") as stop_heartbeat,
    ):
        res = _run(server.admin_soft_sleep())

    assert res == {"status": "soft_sleep"}
    scheduler.deep_reset.assert_called_once()
    stop_heartbeat.assert_called_once()


def test_busy_guard_tolerates_stats_failure():
    scheduler = MagicMock()
    scheduler.get_stats.side_effect = RuntimeError("no stats")
    with patch.object(server, "_get_scheduler", return_value=scheduler):
        assert server._sleep_busy_response("soft") is None
