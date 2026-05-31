# SPDX-License-Identifier: Apache-2.0
"""Contracts for the #175 installed-app admin sleep lifecycle probe."""

from tests.cross_matrix import run_issue175_admin_sleep_probe as probe


def test_classify_admin_sleep_probe_requires_sleep_wake_and_visible_content():
    result = probe.classify_probe(
        {
            "initial_request": {"ok": True, "visible": True},
            "soft_sleep": {"code": 200, "body": {"status": "soft_sleep"}},
            "soft_wake": {"code": 200, "body": {"status": "active"}},
            "after_soft_request": {"ok": True, "visible": True},
            "deep_sleep": {"code": 200, "body": {"status": "deep_sleep"}},
            "deep_wake": {"code": 200, "body": {"status": "active"}},
            "after_deep_request": {"ok": True, "visible": True},
        }
    )

    assert result["status"] == "pass"
    assert result["checks"]["soft_sleep_entered"] is True
    assert result["checks"]["deep_sleep_entered"] is True
    assert result["checks"]["visible_after_deep_wake"] is True


def test_classify_admin_sleep_probe_rejects_missing_deep_wake_content():
    result = probe.classify_probe(
        {
            "initial_request": {"ok": True, "visible": True},
            "soft_sleep": {"code": 200, "body": {"status": "soft_sleep"}},
            "soft_wake": {"code": 200, "body": {"status": "active"}},
            "after_soft_request": {"ok": True, "visible": True},
            "deep_sleep": {"code": 200, "body": {"status": "deep_sleep"}},
            "deep_wake": {"code": 200, "body": {"status": "active"}},
            "after_deep_request": {"ok": True, "visible": False},
        }
    )

    assert result["status"] == "fail"
    assert "visible_after_deep_wake" in result["failures"]
