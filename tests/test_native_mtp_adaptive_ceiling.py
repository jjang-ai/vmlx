# SPDX-License-Identifier: Apache-2.0
"""Adaptive native MTP governs the draft depth: advisory depths only choose where it starts.

Ornith-1.5-122B's converter stamps jang_config.mtp.recommended_num_drafts = 1; the runtime used that stamp as the
Adaptive ladder's TOP, pinning every Ornith request at D1 while D2 measured faster. Every advisory source (default,
tuning sidecar, bundle stamp, family default) must leave the ladder top at the verifier ceiling; only an explicit
env / CLI depth (a diagnostic lever) limits it.
"""

import pytest

from vmlx_engine.native_mtp import (
    native_mtp_adaptive_ceiling,
    native_mtp_depth_is_explicit,
    native_mtp_max_depth,
)


@pytest.mark.parametrize("source", [
    "default", "bundle:recommended_num_drafts", "family_default:hy_v3",
    "/x/vmlx_mtp_tuning.json:best_depth", None, "",
])
def test_advisory_sources_do_not_cap_adaptive(source, monkeypatch):
    monkeypatch.delenv("VMLX_NATIVE_MTP_MAX_DEPTH", raising=False)
    monkeypatch.delenv("VMLINUX_NATIVE_MTP_MAX_DEPTH", raising=False)
    assert not native_mtp_depth_is_explicit(source)
    for depth in (1, 2, 3):
        assert native_mtp_adaptive_ceiling(depth, source) == native_mtp_max_depth() == 3


@pytest.mark.parametrize("source", ["VMLX_NATIVE_MTP_DEPTH", "VMLINUX_NATIVE_MTP_DEPTH"])
def test_explicit_env_depth_stays_a_limit(source):
    assert native_mtp_depth_is_explicit(source)
    assert native_mtp_adaptive_ceiling(1, source) == 1
    assert native_mtp_adaptive_ceiling(2, source) == 2
    assert native_mtp_adaptive_ceiling(9, source) == native_mtp_max_depth()


def test_raised_verifier_ceiling_is_the_adaptive_top(monkeypatch):
    monkeypatch.setenv("VMLX_NATIVE_MTP_MAX_DEPTH", "5")
    assert native_mtp_adaptive_ceiling(1, "bundle:recommended_num_drafts") == native_mtp_max_depth() == 5
