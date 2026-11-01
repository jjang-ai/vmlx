# SPDX-License-Identifier: Apache-2.0
"""MTP telemetry edge-case regression pins.

Issue: per probe on 2026-05-10, three real bugs in `_model_mtp_status()`:

1. Negative `num_nextn_predict_layers` was silently treated as
   `configured_without_runtime` instead of `metadata_inconsistent`. A bundle
   author who ships `-1` (typo, signed-int wrap, manual edit) gets a
   misleading "configured but waiting for runtime wiring" status in /health
   and /v1/models/{id}/capabilities.

2. `drop_mtp` was checked with strict identity (`drop_mtp is True`), so
   truthy non-bool values like `"yes"` or `1` slipped past and were silently
   treated as False (not dropped). The bundle then claimed
   `weights_present_runtime_unwired` even when the author intended to disable
   MTP.

3. When `num_nextn_predict_layers` says N but the bundle index has only k<N
   distinct `mtp.N.*` layers (partial converter output, edited config), the
   prior code reported `artifact_available=True` because *some* mtp.* tensors
   existed. That would silently let a future MTP runtime decode use the
   wrong layer count.

Behavior: all three cases now raise `metadata_inconsistent` with explicit
issue text, and `artifact_available=False` so downstream surfaces don't
claim the bundle is healthy.

Fix: see `vmlx_engine/server.py::_model_mtp_status` — added
`_bundle_index_mtp_layer_count` helper + 3 new validation branches in the
issue list.
"""

import ast
import json
from pathlib import Path
import pytest


def test_native_mtp_cache_snapshot_is_bounded_and_never_reads_tensor_properties():
    from vmlx_engine.native_mtp_cache_telemetry import (
        MAX_MTP_CACHE_LAYERS_SCANNED,
        native_mtp_cache_snapshot,
    )

    class _KVCache:
        def __init__(self, offset, max_size=None):
            self.offset = offset
            if max_size is not None:
                self.max_size = max_size

    class _TensorOffsetTrap:
        @property
        def offset(self):
            raise AssertionError("telemetry must not read tensor-like properties")

        def size(self):
            raise AssertionError("telemetry must not call cache size()")

    uniform = native_mtp_cache_snapshot([_KVCache(17), _KVCache(17)])
    assert uniform == {
        "layers": 2,
        "layers_scanned": 2,
        "introspectable_layers": 2,
        "truncated": False,
        "offset": 17,
        "offset_min": 17,
        "offset_max": 17,
        "length": 17,
        "length_min": 17,
        "length_max": 17,
    }

    rotating = native_mtp_cache_snapshot([_KVCache(80, max_size=32)])
    assert rotating["offset"] == 80
    assert rotating["length"] == 32

    bounded = native_mtp_cache_snapshot(
        [_KVCache(index) for index in range(MAX_MTP_CACHE_LAYERS_SCANNED + 7)]
        + [_TensorOffsetTrap()]
    )
    assert bounded["layers"] == MAX_MTP_CACHE_LAYERS_SCANNED + 8
    assert bounded["layers_scanned"] == MAX_MTP_CACHE_LAYERS_SCANNED
    assert bounded["introspectable_layers"] == MAX_MTP_CACHE_LAYERS_SCANNED
    assert bounded["truncated"] is True
    assert all(not isinstance(value, (list, tuple)) for value in bounded.values())

    unknown = native_mtp_cache_snapshot([_TensorOffsetTrap()])
    assert unknown["introspectable_layers"] == 0
    assert unknown["offset"] is None
    assert unknown["length"] is None


def test_native_mtp_cache_lifecycle_metrics_are_scalar_and_bounded():
    from vmlx_engine.native_mtp_cache_telemetry import (
        MAX_MTP_CACHE_METRIC_VALUE,
        native_mtp_cache_lifecycle_snapshot,
    )

    lifecycle = native_mtp_cache_lifecycle_snapshot(
        head_cache={
            "layers": MAX_MTP_CACHE_METRIC_VALUE + 100,
            "layers_scanned": MAX_MTP_CACHE_METRIC_VALUE + 100,
            "introspectable_layers": MAX_MTP_CACHE_METRIC_VALUE + 100,
            "truncated": True,
            "offset": MAX_MTP_CACHE_METRIC_VALUE + 100,
            "length": -1,
            "offset_min": {"nested": ["must", "be", "dropped"]},
            "length_max": ["must", "be", "dropped"],
            "unexpected": {"unbounded": ["payload"] * 1000},
        },
        recreated_on_rejects=MAX_MTP_CACHE_METRIC_VALUE + 100,
        retained_on_rejects=-10,
    )

    assert lifecycle["recreated_on_rejects"] == MAX_MTP_CACHE_METRIC_VALUE
    assert lifecycle["retained_on_rejects"] == 0
    assert lifecycle["head_cache"]["layers"] == MAX_MTP_CACHE_METRIC_VALUE
    assert lifecycle["head_cache"]["layers_scanned"] == 32
    assert lifecycle["head_cache"]["introspectable_layers"] == 32
    assert lifecycle["head_cache"]["truncated"] is True
    assert lifecycle["head_cache"]["offset"] == MAX_MTP_CACHE_METRIC_VALUE
    assert lifecycle["head_cache"]["offset_min"] is None
    assert lifecycle["head_cache"]["length"] == 0
    assert lifecycle["head_cache"]["length_max"] is None
    assert "unexpected" not in lifecycle["head_cache"]
    assert set(lifecycle["head_cache"]) == {
        "layers",
        "layers_scanned",
        "introspectable_layers",
        "truncated",
        "offset",
        "offset_min",
        "offset_max",
        "length",
        "length_min",
        "length_max",
    }
    assert all(
        value is None or isinstance(value, (bool, int))
        for value in lifecycle["head_cache"].values()
    )
    assert len(json.dumps(lifecycle)) < 512
    assert set(lifecycle) == {
        "head_cache",
        "recreated_on_rejects",
        "retained_on_rejects",
    }


def test_native_mtp_cache_snapshot_stays_off_the_draft_cycle_hot_path():
    """The bounded cache scan belongs only to terminal telemetry publication."""

    roots = {
        Path("vmlx_engine/text_mtp/lane_stats.py"): {
            "finish_record"
        },
        Path("vmlx_engine/mllm_batch_generator.py"): {
            "_native_mtp_log_stats",
            "_native_mtp_capture_head_cache_before_discard",
        },
    }

    class _SnapshotCallVisitor(ast.NodeVisitor):
        def __init__(self):
            self.functions = []
            self.calls = []

        def visit_FunctionDef(self, node):
            self.functions.append(node.name)
            self.generic_visit(node)
            self.functions.pop()

        visit_AsyncFunctionDef = visit_FunctionDef

        def visit_Call(self, node):
            if (
                isinstance(node.func, ast.Name)
                and node.func.id == "native_mtp_cache_snapshot"
            ):
                self.calls.append(self.functions[-1] if self.functions else None)
            self.generic_visit(node)

    for path, expected_functions in roots.items():
        visitor = _SnapshotCallVisitor()
        visitor.visit(ast.parse(path.read_text(encoding="utf-8"), filename=str(path)))
        assert set(visitor.calls) == expected_functions
        assert len(visitor.calls) == len(expected_functions)


def test_native_mtp_stats_snapshot_exposes_acceptance_depth_and_timings(monkeypatch):
    from vmlx_engine.mllm_batch_generator import MLLMNativeMTPStats

    monkeypatch.delenv("VMLINUX_NATIVE_MTP_TRACE", raising=False)
    stats = MLLMNativeMTPStats()
    stats.cycles = 10
    stats.accepts = 6
    stats.rejects = 4
    stats.drafted_tokens = 27
    stats.accepted_tokens = 21
    stats.margin_truncated_cycles = 3
    stats.accepted_by_depth = [10, 8, 3]
    stats.drafted_by_depth = [10, 10, 7]
    stats.seed_main_forwards = 1
    stats.verify_main_forwards = 10
    stats.replay_main_forwards = 4
    stats.mtp_forwards = 31
    stats.verify_ms = 120.0
    stats.sample_ms = 6.0
    stats.draft_ms = 40.0
    stats.snapshot_ms = 5.0
    stats.restore_ms = 3.0
    stats.replay_ms = 22.0
    stats.materialize_ms = 4.0
    stats.draft_head_calls = 31
    stats.draft_head = {
        "configured": True,
        "available": True,
        "source_bits": 8,
        "draft_bits": 4,
        "calls": 900,
    }
    stats.mtp_cache_recreated_on_rejects = 4
    stats.mtp_head_cache = {
        "layers": 1,
        "layers_scanned": 1,
        "introspectable_layers": 1,
        "truncated": False,
        "offset": 9,
        "offset_min": 9,
        "offset_max": 9,
        "length": 9,
        "length_min": 9,
        "length_max": 9,
    }
    snapshot = stats.to_dict(
        request_id="req-test",
        finish_reason="length",
        final_depth=2,
        fallback_reason="d3_acceptance=0.429<min=0.850",
    )

    assert snapshot["request_id"] == "req-test"
    assert snapshot["finish_reason"] == "length"
    assert snapshot["final_depth"] == 2
    assert snapshot["cycles"] == 10
    assert snapshot["accepted_tokens"] == 21
    assert snapshot["drafted_tokens"] == 27
    assert snapshot["margin_truncated_cycles"] == 3
    assert snapshot["acceptance_rate"] == pytest.approx(21 / 27)
    assert snapshot["depth_acceptance_rates"] == {
        "d1": pytest.approx(1.0),
        "d2": pytest.approx(0.8),
        "d3": pytest.approx(3 / 7),
    }
    assert snapshot["forwards"] == {
        "seed_main": 1,
        "verify_main": 10,
        "replay_main": 4,
        "mtp": 31,
    }
    assert snapshot["draft_head"]["calls"] == 31
    assert snapshot["draft_head"]["active_observed"] is True
    assert snapshot["timings_ms"]["total"] == pytest.approx(200.0)
    assert snapshot["timings_ms"]["avg_cycle"] == pytest.approx(20.0)
    assert snapshot["cache_lifecycle"] == {
        "head_cache": stats.mtp_head_cache,
        "recreated_on_rejects": 4,
        "retained_on_rejects": 0,
    }
    assert snapshot["profiled_phase_timing"] is False
    assert snapshot["fallback_reason"] == "d3_acceptance=0.429<min=0.850"


def test_native_mtp_draft_head_telemetry_records_only_request_delta():
    from vmlx_engine.mllm_batch_generator import (
        MLLMNativeMTPStats,
        _record_native_mtp_draft_head_delta,
    )

    stats = MLLMNativeMTPStats()
    before = {"calls": 700, "available": True, "source_bits": 8}
    after = {"calls": 703, "available": True, "source_bits": 8}
    _record_native_mtp_draft_head_delta(stats, before, after)
    _record_native_mtp_draft_head_delta(stats, after, {**after, "calls": 702})

    assert stats.draft_head_calls == 3
    assert stats.draft_head["calls"] == 702


def test_mllm_native_mtp_stats_identify_synchronized_phase_timing(monkeypatch):
    from vmlx_engine.mllm_batch_generator import (
        MLLMBatchStats,
        MLLMNativeMTPStats,
    )

    monkeypatch.setenv("VMLINUX_NATIVE_MTP_TRACE", "1")
    batch_stats = MLLMBatchStats()
    batch_stats.record_native_mtp(
        request_id="profiled-row",
        stats=MLLMNativeMTPStats(),
        finish_reason="stop",
        final_depth=1,
    )

    snapshot = batch_stats.to_dict()["last_native_mtp"]

    assert snapshot["profiled_phase_timing"] is True


def test_mllm_adaptive_discard_preserves_last_head_cache_snapshot():
    from vmlx_engine.mllm_batch_generator import (
        MLLMNativeMTPStats,
        _native_mtp_capture_head_cache_before_discard,
        _native_mtp_log_stats,
    )

    class _KVCache:
        def __init__(self, offset):
            self.offset = offset

    stats = MLLMNativeMTPStats()
    cache = [_KVCache(41)]
    _native_mtp_capture_head_cache_before_discard(stats, cache)
    _native_mtp_log_stats("fallback-row", stats, "fallback_to_ar", None)

    assert stats.mtp_head_cache["offset"] == 41
    assert stats.mtp_head_cache["length"] == 41


def test_mllm_remove_publishes_active_mtp_cancellation_once():
    from types import SimpleNamespace

    from vmlx_engine.mllm_batch_generator import (
        MLLMBatchGenerator,
        MLLMBatchStats,
        MLLMNativeMTPState,
    )

    class _KVCache:
        def __init__(self, offset):
            self.offset = offset

    state = MLLMNativeMTPState(
        mtp_cache=[_KVCache(23)],
        depth=1,
    )
    request = SimpleNamespace(
        uid=7,
        request_id="cancel-mtp-row",
        _native_mtp_state=state,
    )
    generator = MLLMBatchGenerator.__new__(MLLMBatchGenerator)
    generator.active_batch = SimpleNamespace(
        uids=[7],
        requests=[request],
    )
    generator.unprocessed_requests = []
    generator._stats = MLLMBatchStats()

    generator.remove([7])

    assert generator.active_batch is None
    assert not hasattr(request, "_native_mtp_state")
    assert generator._stats.last_native_mtp["finish_reason"] == "cancelled"
    assert (
        generator._stats.last_native_mtp["cache_lifecycle"]["head_cache"]["offset"]
        == 23
    )


def test_native_mtp_light_timing_records_without_sync_trace(monkeypatch):
    from vmlx_engine.mllm_batch_generator import (
        MLLMNativeMTPStats,
        _native_mtp_trace_start,
        _native_mtp_trace_stop,
    )

    monkeypatch.delenv("VMLINUX_NATIVE_MTP_TRACE", raising=False)
    stats = MLLMNativeMTPStats()
    start = _native_mtp_trace_start()
    assert start > 0

    _native_mtp_trace_stop(stats, "draft_ms", start)

    assert stats.draft_ms > 0


def test_text_lane_native_mtp_stats_publish_acceptance_depth_and_forwards():
    from vmlx_engine.text_mtp.lane_stats import LaneStats; from vmlx_engine.text_mtp.lane_stats import publish; from vmlx_engine.text_mtp.lane_stats import native_mtp_stats_snapshot

    stats = LaneStats(
        cycles=5,
        accepts=3,
        rejects=2,
        init_emits=2,
        draft_emits=3,
        bonus_emits=3,
        verify_emits=2,
        depth=1,
        draft_tokens_proposed=5,
        draft_tokens_accepted=3,
        accepted_by_depth=[3, 0, 0],
        drafted_by_depth=[5, 0, 0],
        seed_main_forwards=1,
        verify_main_forwards=5,
        mtp_forwards=6,
        backbone_ms=20.0,
        mtp_head_ms=5.0,
        sample_ms=2.0,
        cache_ops_ms=1.0,
        mtp_cache_retained_on_rejects=2,
        mtp_head_cache={
            "layers": 1,
            "layers_scanned": 1,
            "introspectable_layers": 1,
            "truncated": False,
            "offset": 7,
            "offset_min": 7,
            "offset_max": 7,
            "length": 7,
            "length_min": 7,
            "length_max": 7,
        },
    )

    published = publish("hy3-test", stats, "stop")
    snapshot = native_mtp_stats_snapshot()

    assert published["request_id"] == "hy3-test"
    assert published["final_depth"] == 1
    assert published["accepted_tokens"] == 3
    assert published["drafted_tokens"] == 5
    assert published["acceptance_rate"] == pytest.approx(0.6)
    assert published["depth_acceptance_rates"] == {
        "d1": pytest.approx(0.6),
        "d2": None,
        "d3": None,
    }
    assert published["forwards"] == {
        "seed_main": 1,
        "verify_main": 5,
        "replay_main": 0,
        "mtp": 6,
    }
    assert published["timings_ms"]["total"] == pytest.approx(28.0)
    assert published["cache_lifecycle"] == {
        "head_cache": stats.mtp_head_cache,
        "recreated_on_rejects": 0,
        "retained_on_rejects": 2,
    }
    assert snapshot["last_native_mtp"] == published
    assert snapshot["native_mtp_totals"]["requests"] >= 1
    assert snapshot["native_mtp_totals"]["accepted_tokens"] >= 3
    assert snapshot["native_mtp_totals"]["mtp_cache_retained_on_rejects"] >= 2


def test_mllm_batch_stats_projects_mtp_cache_lifecycle_to_scheduler_stats():
    from vmlx_engine.mllm_batch_generator import (
        MLLMBatchStats,
        MLLMNativeMTPStats,
    )

    batch_stats = MLLMBatchStats()
    mtp_stats = MLLMNativeMTPStats(
        rejects=2,
        mtp_cache_recreated_on_rejects=2,
    )
    batch_stats.record_native_mtp(
        request_id="qwen-vl",
        stats=mtp_stats,
        finish_reason="stop",
        final_depth=1,
    )

    lifecycle = batch_stats.to_dict()["last_native_mtp"]["cache_lifecycle"]
    assert lifecycle["recreated_on_rejects"] == 2
    assert lifecycle["retained_on_rejects"] == 0


class TestMtpTelemetryEdgeCases:
    def test_negative_layer_count_flagged_metadata_inconsistent(self, tmp_path):
        from vmlx_engine.server import _model_mtp_status

        (tmp_path / "config.json").write_text(
            '{"model_type":"deepseek_v4","num_nextn_predict_layers":-1}'
        )
        status = _model_mtp_status(str(tmp_path))
        assert status["status"] == "metadata_inconsistent"
        assert status["artifact_available"] is False
        assert status["runtime_available"] is False
        assert any(
            "negative" in issue.lower() for issue in status["issues"]
        ), status["issues"]

    def test_drop_mtp_string_yes_flagged_invalid_type(self, tmp_path):
        from vmlx_engine.server import _model_mtp_status

        (tmp_path / "config.json").write_text(
            '{"model_type":"deepseek_v4","num_nextn_predict_layers":1}'
        )
        (tmp_path / "jang_config.json").write_text(
            '{"weight_format":"mxtq","drop_mtp":"yes"}'
        )
        (tmp_path / "model.safetensors.index.json").write_text(
            '{"weight_map":{"mtp.0.layers.0.self_attn.q_proj.weight":"model.safetensors"}}'
        )
        status = _model_mtp_status(str(tmp_path))
        assert status["status"] == "metadata_inconsistent"
        assert status["artifact_available"] is False
        assert any(
            "drop_mtp" in issue and "boolean" in issue.lower()
            for issue in status["issues"]
        ), status["issues"]

    def test_drop_mtp_int_one_flagged_invalid_type(self, tmp_path):
        """Integer 1 is also truthy non-bool; must be flagged like the string case."""
        from vmlx_engine.server import _model_mtp_status

        (tmp_path / "config.json").write_text(
            '{"model_type":"deepseek_v4","num_nextn_predict_layers":1}'
        )
        (tmp_path / "jang_config.json").write_text(
            '{"weight_format":"mxtq","drop_mtp":1}'
        )
        status = _model_mtp_status(str(tmp_path))
        assert any(
            "drop_mtp" in issue and "boolean" in issue.lower()
            for issue in status["issues"]
        ), status["issues"]

    def test_partial_indexed_layers_flagged(self, tmp_path):
        """Config says 3 MTP layers, only mtp.0 in index → metadata_inconsistent."""
        from vmlx_engine.server import _model_mtp_status

        (tmp_path / "config.json").write_text(
            '{"model_type":"deepseek_v4","num_nextn_predict_layers":3}'
        )
        (tmp_path / "jang_config.json").write_text(
            '{"weight_format":"mxtq","drop_mtp":false}'
        )
        (tmp_path / "model.safetensors.index.json").write_text(
            '{"weight_map":{"mtp.0.layers.0.self_attn.q_proj.weight":"model.safetensors"}}'
        )
        status = _model_mtp_status(str(tmp_path))
        assert status["status"] == "metadata_inconsistent"
        assert status["artifact_available"] is False
        assert any(
            "1 distinct" in issue.lower() for issue in status["issues"]
        ), status["issues"]

    def test_correct_layer_count_match_stays_healthy(self, tmp_path):
        """Healthy regression: when config_layers == indexed_mtp_layer_count,
        no "distinct mtp.N" issue is raised (paired with the partial-layer
        test above)."""
        from vmlx_engine.server import _model_mtp_status

        (tmp_path / "config.json").write_text(
            '{"model_type":"deepseek_v4","num_nextn_predict_layers":3}'
        )
        (tmp_path / "jang_config.json").write_text(
            '{"weight_format":"mxtq","drop_mtp":false}'
        )
        (tmp_path / "model.safetensors.index.json").write_text(
            '{"weight_map":{'
            '"mtp.0.layers.0.self_attn.q_proj.weight":"model.safetensors",'
            '"mtp.1.layers.0.self_attn.q_proj.weight":"model.safetensors",'
            '"mtp.2.layers.0.self_attn.q_proj.weight":"model.safetensors"'
            '}}'
        )
        status = _model_mtp_status(str(tmp_path))
        assert status["status"] == "weights_present_runtime_unwired"
        assert status["artifact_available"] is True
        assert not any(
            "distinct mtp" in issue.lower() for issue in status["issues"]
        ), status["issues"]

    def test_drop_mtp_true_strict_bool_still_works(self, tmp_path):
        """Healthy regression: literal bool True is the canonical 'drop' signal."""
        from vmlx_engine.server import _model_mtp_status

        (tmp_path / "config.json").write_text(
            '{"model_type":"deepseek_v4","num_nextn_predict_layers":0}'
        )
        (tmp_path / "jang_config.json").write_text(
            '{"weight_format":"mxtq","drop_mtp":true}'
        )
        status = _model_mtp_status(str(tmp_path))
        assert status["status"] == "dropped"
        assert status["issues"] == []

    def test_drop_mtp_false_strict_bool_still_works(self, tmp_path):
        """Healthy regression: literal bool False is treated as 'don't drop'."""
        from vmlx_engine.server import _model_mtp_status

        (tmp_path / "config.json").write_text(
            '{"model_type":"deepseek_v4","num_nextn_predict_layers":1}'
        )
        (tmp_path / "jang_config.json").write_text(
            '{"weight_format":"mxtq","drop_mtp":false}'
        )
        (tmp_path / "model.safetensors.index.json").write_text(
            '{"weight_map":{"mtp.0.layers.0.self_attn.q_proj.weight":"model.safetensors"}}'
        )
        status = _model_mtp_status(str(tmp_path))
        # Accepts either weights_present_runtime_unwired (codex's runtime_supported
        # branch) — main contract is no metadata_inconsistent issues.
        assert status["status"] != "metadata_inconsistent"
        assert status["artifact_available"] is True
        assert status["issues"] == []

    def test_bundle_index_mtp_layer_count_helper(self, tmp_path):
        """Direct unit on the helper used for the partial-layer issue."""
        from vmlx_engine.server import _bundle_index_mtp_layer_count

        # No bundle path → None.
        assert _bundle_index_mtp_layer_count(None) is None
        # Empty bundle → None.
        assert _bundle_index_mtp_layer_count(str(tmp_path)) is None
        # Index without mtp.* keys → None.
        (tmp_path / "model.safetensors.index.json").write_text(
            '{"weight_map":{"layers.0.self_attn.q_proj.weight":"model.safetensors"}}'
        )
        assert _bundle_index_mtp_layer_count(str(tmp_path)) is None
        # Index with mtp.0/1/2 keys → 3.
        (tmp_path / "model.safetensors.index.json").write_text(
            '{"weight_map":{'
            '"mtp.0.layers.0.q.weight":"a",'
            '"mtp.0.layers.0.k.weight":"a",'
            '"mtp.1.layers.0.q.weight":"a",'
            '"mtp.2.layers.0.q.weight":"a"'
            '}}'
        )
        assert _bundle_index_mtp_layer_count(str(tmp_path)) == 3


def test_mllm_native_mtp_skip_reaches_health_snapshot(monkeypatch):
    """The MLLM gate must publish WHY MTP was skipped, like the text lane.

    PerformancePanel reads batch_generator.last_native_mtp_skip; without this
    key an MLLM session that only ever ran sampled requests showed a null MTP
    tile with no way to tell "skipped by policy" from "broken".
    """
    from vmlx_engine import mllm_batch_generator as mllm_gen
    from vmlx_engine.mllm_batch_generator import (
        MLLMBatchGenerator,
        MLLMBatchStats,
    )

    class _Req:
        request_id = "sampled-row"
        temperature = 0.7
        repetition_penalty = 1.0

    class _ModelWithHead:
        mtp = object()

        def mtp_forward(self):
            pass

        def make_mtp_cache(self):
            return []

    monkeypatch.setattr(mllm_gen, "_NATIVE_MTP_STOCHASTIC_ACCEPT", False)
    gen = object.__new__(MLLMBatchGenerator)
    gen._stats = MLLMBatchStats()
    gen.language_model = _ModelWithHead()

    assert gen._native_mtp_enabled_for_request(_Req()) is False

    snapshot = gen._stats.to_dict()
    skip = snapshot["last_native_mtp_skip"]
    assert skip is not None
    assert skip["uid"] == "sampled-row"
    assert skip["reason"]
    # Engagement snapshots stay independent of skip snapshots (text-lane parity).
    assert snapshot["last_native_mtp"] is None


def test_mllm_sampled_request_uses_shared_stochastic_acceptance(monkeypatch):
    """A nonzero-temperature MLLM request stays on MTP when rejection
    sampling is enabled; the environment kill switch alone restores the old
    deterministic-only gate.
    """
    from vmlx_engine import mllm_batch_generator as mllm_gen
    from vmlx_engine.mllm_batch_generator import MLLMBatchGenerator, MLLMBatchStats

    class _Req:
        request_id = "sampled-mtp-row"
        temperature = 1.0
        repetition_penalty = 1.0

    class _ModelWithHead:
        mtp = object()

        def mtp_forward(self):
            pass

        def make_mtp_cache(self):
            return []

    monkeypatch.setattr(mllm_gen, "_NATIVE_MTP_STOCHASTIC_ACCEPT", True)
    gen = object.__new__(MLLMBatchGenerator)
    gen._stats = MLLMBatchStats()
    gen.language_model = _ModelWithHead()

    assert gen._native_mtp_disabled_reason_for_request(_Req()) is None
    assert gen._native_mtp_enabled_for_request(_Req()) is True
    assert gen._stats.to_dict()["last_native_mtp_skip"] is None


def test_both_native_mtp_env_spellings_reach_every_reader(monkeypatch):
    """VMLX_NATIVE_MTP=0 must disable MTP for the MLLM gate and /health too.

    native_mtp.py's kill switch already accepted both prefixes, but the MLLM
    per-request gate and the /health status bit each kept their own copy that
    only knew the legacy VMLINUX_ spelling. So `VMLX_NATIVE_MTP=0` switched the
    runtime off while those two carried on as if MTP were on: an A/B run that
    way silently compares MTP against itself, and /health reports the wrong
    state. Pin that all readers now agree, for BOTH spellings.
    """
    from vmlx_engine.native_mtp import native_mtp_disabled_by_env
    from vmlx_engine.mllm_batch_generator import MLLMBatchGenerator, MLLMBatchStats

    class _Req:
        request_id = "env-gate"
        temperature = 0.0
        repetition_penalty = 1.0

    def gate_reason():
        gen = object.__new__(MLLMBatchGenerator)
        gen._stats = MLLMBatchStats()
        gen.language_model = _ModelWithHead()
        return gen._native_mtp_disabled_reason_for_request(_Req())

    class _ModelWithHead:
        """Passes the head check so the env gate is what decides."""

        mtp = object()

    for name in ("VMLX_NATIVE_MTP", "VMLINUX_NATIVE_MTP"):
        monkeypatch.delenv("VMLX_NATIVE_MTP", raising=False)
        monkeypatch.delenv("VMLINUX_NATIVE_MTP", raising=False)
        monkeypatch.setenv(name, "0")

        assert native_mtp_disabled_by_env() is True, name
        reason = gate_reason()
        assert reason is not None, f"MLLM gate ignored {name}=0"
        assert "disable" in reason.lower(), reason

    monkeypatch.delenv("VMLX_NATIVE_MTP", raising=False)
    monkeypatch.delenv("VMLINUX_NATIVE_MTP", raising=False)
    assert native_mtp_disabled_by_env() is False


def test_mllm_reentry_record_explicitly_reports_only_last_phase():
    from vmlx_engine.mllm_batch_generator import MLLMBatchStats, MLLMNativeMTPStats
    publisher = MLLMBatchStats()
    first = MLLMNativeMTPStats(cycles=10, drafted_tokens=20, accepted_tokens=12)
    publisher.record_native_mtp(request_id="same-request", stats=first,
                                finish_reason="fallback_to_ar", final_depth=2)
    original = first.to_dict(request_id="same-request", finish_reason="fallback_to_ar",
                             final_depth=2)
    second = MLLMNativeMTPStats(cycles=2, drafted_tokens=4, accepted_tokens=3)
    publisher.record_native_mtp(request_id="same-request", stats=second,
                                finish_reason="ar_calibration", final_depth=1)
    record = publisher.to_dict()["last_native_mtp"]
    assert record["scope"] == "last_native_mtp_phase"
    assert record["cycles"] == 2 and record["drafted_tokens"] == 4
    assert record["finish_reason"] == "ar_calibration"
    assert first.to_dict(request_id="same-request", finish_reason="fallback_to_ar",
                         final_depth=2) == original
