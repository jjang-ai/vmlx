# SPDX-License-Identifier: Apache-2.0
"""Contracts for the DSV4 native-versus-optimized acceptance report."""

from __future__ import annotations

import importlib.util
import json
import os
import sys
from pathlib import Path


def _module():
    path = Path(__file__).parents[1] / "bench" / "dsv4_performance_suite.py"
    spec = importlib.util.spec_from_file_location("dsv4_performance_suite", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _case(*, token_hash="hash", decode=20.0, min_decode=None, prefill=200.0, passed=True):
    return {
        "passed": passed,
        "token_ids_sha256": token_hash,
        "median_decode_tok_s": decode,
        "min_decode_tok_s": decode if min_decode is None else min_decode,
        "median_prefill_tok_s": prefill,
    }


def test_case_parser_requires_unique_positive_triplets():
    mod = _module()
    cases = mod._parse_cases("256:30:3,256:1000:1")
    assert [case.key for case in cases] == ["p256-g30", "p256-g1000"]
    for invalid in ("", "256:30", "256:0:1", "256:30:1,256:30:2"):
        try:
            mod._parse_cases(invalid)
        except ValueError:
            pass
        else:
            raise AssertionError(f"accepted invalid case list {invalid!r}")


def test_profile_controls_isolate_candidate_features(monkeypatch):
    mod = _module()
    monkeypatch.setattr(os, "environ", {})
    head = mod._configure_profile("optimized", "lm-head")
    assert head["VMLX_DSV4_LM_HEAD_MODE"] == "exact-cache"
    assert head["VMLX_DSV4_ROPE_CACHE"] == "0"
    rope = mod._configure_profile("optimized", "rope-cache")
    assert rope["VMLX_DSV4_LM_HEAD_MODE"] == "native"
    assert rope["VMLX_DSV4_ROPE_CACHE"] == "1"


def test_acceptance_requires_exact_hash_and_bounded_throughput():
    mod = _module()
    baseline = {"results": {"p256-g30": _case()}}
    candidate = {
        "results": {"p256-g30": _case(decode=19.8, prefill=195.0)}
    }
    accepted = mod._compare(
        candidate,
        baseline,
        min_decode_ratio=0.98,
        min_prefill_ratio=0.95,
    )
    assert accepted["passed"] is True

    candidate["results"]["p256-g30"]["token_ids_sha256"] = "different"
    assert (
        mod._compare(
            candidate,
            baseline,
            min_decode_ratio=0.98,
            min_prefill_ratio=0.95,
        )["passed"]
        is False
    )


def test_acceptance_fails_closed_for_missing_or_slow_cases():
    mod = _module()
    baseline = {"results": {"p256-g30": _case(), "p256-g1000": _case()}}
    missing = {"results": {"p256-g30": _case()}}
    missing_baseline = {"results": {"p256-g1000": _case()}}
    assert (
        mod._compare(
            missing_baseline,
            {"results": {}},
            min_decode_ratio=0.98,
            min_prefill_ratio=0.95,
        )["passed"]
        is False
    )
    assert (
        mod._compare(
            missing,
            baseline,
            min_decode_ratio=0.98,
            min_prefill_ratio=0.95,
        )["cases"]["p256-g1000"]["reason"]
        == "missing candidate case"
    )
    slow = {"results": {"p256-g30": _case(decode=15.0)}}
    assert (
        mod._compare(
            slow,
            missing,
            min_decode_ratio=0.98,
            min_prefill_ratio=0.95,
        )["passed"]
        is False
    )
    assert set(baseline["results"]) == {"p256-g30", "p256-g1000"}


def test_acceptance_rejects_incompatible_baseline_metadata():
    mod = _module()
    candidate = {
        "schema": mod.SCHEMA,
        "profile": "optimized",
        "passed": True,
        "feature_state_passed": True,
        "model": {"bundle_name": "model", "manifest_sha256": "bundle"},
        "source": {"checkout_verified": True, "repository_root": "."},
        "git_revision": "candidate",
        "hardware": {"machine": "arm64"},
        "runtime": {"mlx": "1"},
        "cases": [{"key": "p256-g30"}],
        "benchmark_controls": {
            "min_within_profile_decode_ratio": 0.75,
            "min_anchor_decode_ratio": 0.40,
        },
    }
    baseline = candidate | {"profile": "native"}
    assert mod._baseline_compatibility(candidate, baseline)["passed"] is True

    stale = baseline | {"git_revision": "stale"}
    compatibility = mod._baseline_compatibility(candidate, stale)
    assert compatibility["passed"] is False
    assert compatibility["checks"]["git_revision"] is False


def test_model_identity_is_public_and_changes_with_manifest(tmp_path):
    mod = _module()
    (tmp_path / "config.json").write_text('{"model_type":"deepseek_v4"}')
    (tmp_path / "model-00001-of-00001.safetensors").write_bytes(b"weights")
    (tmp_path / "model.safetensors.index.json").write_text(
        '{"weight_map":{"model.weight":"model-00001-of-00001.safetensors"}}'
    )

    first = mod._model_identity(tmp_path)
    assert first["bundle_name"] == tmp_path.name
    assert first["shard_count"] == 1
    assert first["shard_bytes"] == 7
    assert str(tmp_path) not in json.dumps(first)

    (tmp_path / "config.json").write_text('{"model_type":"changed"}')
    second = mod._model_identity(tmp_path)
    assert second["manifest_sha256"] != first["manifest_sha256"]


def test_acceptance_rejects_catastrophic_tail_hidden_by_median():
    mod = _module()
    baseline = {"results": {"p512-g30": _case(decode=12.0, min_decode=11.0)}}
    candidate = {"results": {"p512-g30": _case(decode=12.1, min_decode=0.8)}}
    comparison = mod._compare(
        candidate,
        baseline,
        min_decode_ratio=0.98,
        min_prefill_ratio=0.95,
        min_sample_decode_ratio=0.75,
    )
    assert comparison["passed"] is False


def test_case_summary_rejects_unstable_profile_tail():
    mod = _module()
    case = mod.Case(prompt_target=512, generation_target=30, repeats=3)
    samples = [
        {
            "new_tokens": 30,
            "token_ids_sha256": "same",
            "decode_tok_s": decode,
            "prefill_tok_s": 200.0,
        }
        for decode in (12.0, 11.5, 0.7)
    ]

    summary = mod._summarize(case, samples)
    assert summary["passed"] is False
    assert summary["stable_decode_tail"] is False
    assert summary["min_to_median_decode_ratio"] < 0.1


def test_cross_case_decode_floor_rejects_consistently_cold_short_case():
    mod = _module()
    results = {
        "p256-g30": _case(decode=0.75),
        "p400-g30": _case(decode=10.0),
        "p256-g1000": _case(decode=18.5),
    }

    sanity = mod._decode_floor_sanity(results, min_anchor_ratio=0.40)
    assert sanity["passed"] is False
    assert sanity["cases"]["p256-g30"]["passed"] is False

    targeted = mod._decode_floor_sanity({"p400-g30": _case(decode=10.0)})
    assert targeted["passed"] is True
    assert targeted["applied"] is False
