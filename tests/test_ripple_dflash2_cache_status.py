"""Ripple's scheduler-free session cache must report its actual active tiers."""
from types import SimpleNamespace

import pytest


@pytest.fixture
def active_lane(monkeypatch):
    from vmlx_engine import server, dflash2_runtime

    monkeypatch.setattr(server, "_engine", SimpleNamespace(_llm_dflash2_lane=lambda _: True))
    monkeypatch.setattr(dflash2_runtime, "_SESSION_SSD", object())
    monkeypatch.setattr(dflash2_runtime, "_SESSION_STORE", SimpleNamespace(max_entries=4))

    def forbidden():
        raise AssertionError("health must not scan SSD or touch tensors")

    monkeypatch.setattr(dflash2_runtime, "session_cache_stats", forbidden)
    return server, dflash2_runtime


@pytest.mark.parametrize("ram,ssd", [(4, True), (4, False), (0, True), (0, False)])
def test_active_tiers_and_topology(active_lane, monkeypatch, ram, ssd):
    server, runtime = active_lane
    monkeypatch.setattr(runtime._SESSION_STORE, "max_entries", ram)
    monkeypatch.setattr(runtime, "_SESSION_SSD", object() if ssd else None)
    status = server._native_cache_status(family="ricursive")
    assert status["cache_type"] == "dflash2_native_session"
    assert status["native_checkpoint_ssd"] is ssd
    assert status["prefix"] is bool(ram or ssd)
    assert status["paged"] is False
    assert status["block_disk_l2"] is False
    topology = server._cache_topology_configuration(None, {"native_cache": status})
    assert topology["native_cache"]["schema"] == status["schema"]
    assert topology["native_cache"]["prefix"] is bool(ram or ssd)
    assert topology["instantiated"]["paged_ram_enabled"] is False


def test_disabled_lane_does_not_advertise_stale_store(active_lane, monkeypatch):
    server, _ = active_lane
    monkeypatch.setattr(server, "_engine", SimpleNamespace(_llm_dflash2_lane=lambda _: False))
    assert not server._native_cache_status(family="ricursive").get("native_checkpoint_ssd")


def test_scheduler_and_other_families_keep_their_contract(active_lane):
    server, _ = active_lane
    assert server._native_cache_status(SimpleNamespace(), family="ricursive").get("cache_backend") != "dflash2-text"
    assert server._native_cache_status(family="qwen3").get("cache_backend") != "dflash2-text"
