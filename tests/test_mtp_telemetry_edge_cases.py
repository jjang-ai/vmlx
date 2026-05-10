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

import json
import pytest


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
