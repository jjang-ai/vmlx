# SPDX-License-Identifier: Apache-2.0
"""The L2 cache identity must change when engine sources change.

Package versions alone cannot separate two runtimes during development: the
version string sits still while cache/model math changes underneath it, so L2
blocks written by the previous build keep matching and get replayed as valid.

That is not theoretical. Reproduced live on Muse-Glimmer-30B: with a stale
block-cache a warm hit answered a DIFFERENT question than the identical cold
request — "Nice to meet you, Eric!" instead of "Oslo" — confidently and
deterministically, 6 runs out of 6. Quarantining the cache made warm agree with
cold again (5/5 correct at cached=106).

Released installs do not need this (a release bumps the version) and must not pay
the ~180 ms hash, so the source id is computed ONLY inside a git checkout.
"""

import pathlib
from types import SimpleNamespace
import pytest

from vmlx_engine import prefix_cache


@pytest.mark.parametrize("name", ["VMLX_M3_FUSED_ATTN", "VMLX_JANGT_FAST", "VMLX_M3_FUSED_MOE", "VMLX_MINIMAX_MSA"])
def test_minimax_numerical_overrides_isolate_persistent_state(monkeypatch, name):
    monkeypatch.delenv(name, raising=False)
    baseline = prefix_cache._resolve_runtime_cache_fingerprint()
    monkeypatch.setenv(name, "0")
    disabled = prefix_cache._resolve_runtime_cache_fingerprint()
    monkeypatch.setenv(name, "1")
    enabled = prefix_cache._resolve_runtime_cache_fingerprint()
    assert len({baseline, disabled, enabled}) == 3
    assert enabled == prefix_cache._resolve_runtime_cache_fingerprint()
    monkeypatch.delenv(name)
    assert prefix_cache._resolve_runtime_cache_fingerprint() == baseline


def test_minimax_policy_is_frozen_for_a_running_process(monkeypatch):
    baseline = prefix_cache.runtime_cache_fingerprint()
    monkeypatch.setenv("VMLX_M3_FUSED_ATTN", "0")
    assert prefix_cache.runtime_cache_fingerprint() == baseline


def test_full_precision_cache_identity_isolated_from_default(monkeypatch):
    monkeypatch.delenv("MLX_ENABLE_TF32", raising=False)
    default = prefix_cache._resolve_runtime_cache_fingerprint()
    monkeypatch.setenv("MLX_ENABLE_TF32", "0")
    precise = prefix_cache._resolve_runtime_cache_fingerprint()
    assert precise != default
    assert precise == prefix_cache._resolve_runtime_cache_fingerprint()
    monkeypatch.setenv("MLX_ENABLE_TF32", "1")
    assert prefix_cache._resolve_runtime_cache_fingerprint() == default


def test_precision_environment_does_not_rekey_running_process(monkeypatch):
    baseline = prefix_cache.runtime_cache_fingerprint()
    monkeypatch.setenv("MLX_ENABLE_TF32", "0")
    assert prefix_cache.runtime_cache_fingerprint() == baseline


def test_same_version_different_metal_wheels_have_distinct_cache_identity(monkeypatch):
    platform = {"mlx": "14", "mlx-metal": "14"}
    monkeypatch.setattr(prefix_cache.importlib.metadata, "version", lambda _: "0.32.2")
    monkeypatch.setattr(prefix_cache.importlib.metadata, "distribution", lambda name:
        SimpleNamespace(read_text=lambda _: f"Tag: py3-none-macosx_{platform[name]}_0_arm64\n"))
    compat = prefix_cache._resolve_runtime_cache_fingerprint()
    platform["mlx-metal"] = "26"
    metal_changed = prefix_cache._resolve_runtime_cache_fingerprint()
    assert metal_changed != compat
    platform["mlx"] = "26"
    native = prefix_cache._resolve_runtime_cache_fingerprint()
    assert native != metal_changed
    assert native == prefix_cache._resolve_runtime_cache_fingerprint()
    platform.update({"mlx": "14", "mlx-metal": "14"})
    assert prefix_cache._resolve_runtime_cache_fingerprint() == compat


def test_wheel_metadata_failure_does_not_change_running_identity(monkeypatch):
    baseline = prefix_cache.runtime_cache_fingerprint()
    def unavailable(_):
        raise OSError("metadata unavailable")
    monkeypatch.setattr(prefix_cache.importlib.metadata, "distribution", unavailable)
    assert prefix_cache.runtime_cache_fingerprint() == baseline
    assert "mlx_wheel=unknown" in prefix_cache._resolve_runtime_cache_fingerprint()


def test_source_id_is_deterministic():
    assert prefix_cache._resolve_source_checkout_id() == (
        prefix_cache._resolve_source_checkout_id()
    ), "cache identity must not change without a source change"


def test_source_id_present_in_a_checkout_and_absent_otherwise():
    package_dir = pathlib.Path(prefix_cache.__file__).resolve().parent
    in_checkout = (package_dir.parent / ".git").exists()
    source_id = prefix_cache._resolve_source_checkout_id()
    if in_checkout:
        assert source_id, "a source checkout must contribute a content id"
        assert len(source_id) == 12
    else:
        assert source_id == "", "a released install must skip source hashing"


def test_fingerprint_carries_the_source_id_when_present():
    fingerprint = prefix_cache._resolve_runtime_cache_fingerprint()
    source_id = prefix_cache._resolve_source_checkout_id()
    assert fingerprint.startswith("runtime_cache=")
    if source_id:
        assert f"src={source_id}" in fingerprint


def test_source_id_tracks_file_content(tmp_path, monkeypatch):
    """Editing a source file must change the id; reverting must restore it."""
    package_dir = pathlib.Path(prefix_cache.__file__).resolve().parent
    if not (package_dir.parent / ".git").exists():
        return  # released install: nothing to track

    target = package_dir / "paged_cache.py"
    original = target.read_bytes()
    before = prefix_cache._resolve_source_checkout_id()
    try:
        target.write_bytes(original + b"\n# fingerprint probe\n")
        assert prefix_cache._resolve_source_checkout_id() != before
    finally:
        target.write_bytes(original)
    assert prefix_cache._resolve_source_checkout_id() == before
