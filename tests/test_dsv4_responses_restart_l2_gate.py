# SPDX-License-Identifier: Apache-2.0
"""Contracts for the DSV4 Responses restart/L2 gate preflight."""

from types import SimpleNamespace
import sys

from tests.cross_matrix import run_dsv4_responses_restart_l2_gate as gate


def test_dsv4_responses_restart_l2_resource_snapshot_labels_binary_units(monkeypatch):
    monkeypatch.setitem(
        sys.modules,
        "psutil",
        SimpleNamespace(
            virtual_memory=lambda: SimpleNamespace(
                total=128 * 1024**3,
                available=74 * 1024**3,
                percent=42.0,
            )
        ),
    )

    snapshot = gate.resource_snapshot("preflight")

    assert snapshot["system_memory"]["unit"] == "GiB"
    assert snapshot["system_memory"]["total_gib"] == 128.0
    assert snapshot["system_memory"]["available_gib"] == 74.0
    assert snapshot["system_memory"]["total_gb"] == 128.0
    assert snapshot["system_memory"]["available_gb"] == 74.0


def test_dsv4_responses_restart_l2_memory_preflight_labels_binary_units(monkeypatch):
    monkeypatch.setattr(
        gate,
        "resource_snapshot",
        lambda name: {
            "name": name,
            "system_memory": {
                "unit": "GiB",
                "available_gib": 74.0,
                "available_gb": 74.0,
                "total_gib": 128.0,
                "total_gb": 128.0,
            },
        },
    )
    args = SimpleNamespace(min_free_gb=80.0)

    artifact = gate.blocked_by_memory_preflight(args)

    assert artifact is not None
    assert artifact["status"] == "skipped"
    assert artifact["reason"] == "insufficient_free_memory"
    assert artifact["unit"] == "GiB"
    assert artifact["required_available_gib"] == 80.0
    assert artifact["required_available_gb"] == 80.0
    assert artifact["available_gib"] == 74.0
    assert artifact["available_gb"] == 74.0
    assert artifact["memory_gap_gib"] == 6.0
    assert artifact["memory_gap_gb"] == 6.0
    assert artifact["telemetry"][0]["system_memory"]["available_gib"] == 74.0
