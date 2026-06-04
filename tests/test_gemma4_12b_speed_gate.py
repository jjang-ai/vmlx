from tests.cross_matrix.run_gemma4_12b_speed_gate import summarize_results


def test_gemma4_speed_gate_passes_on_default_median_target():
    result = summarize_results(
        [
            {"variant": "default_bundle_sampling", "http_code": 200, "wall_tps": 46.5},
            {"variant": "default_bundle_sampling", "http_code": 200, "wall_tps": 45.5},
            {"variant": "temp1_topk0", "http_code": 200, "wall_tps": 45.0},
            {"variant": "temp1_topk64", "http_code": 200, "wall_tps": 45.2},
        ],
        45.0,
    )

    assert result["status"] == "pass"
    assert result["default_median_tps"] == 46.0
    assert result["top_k_primary_regression"] is False


def test_gemma4_speed_gate_fails_when_default_median_is_below_target():
    result = summarize_results(
        [
            {"variant": "default_bundle_sampling", "http_code": 200, "wall_tps": 44.0},
            {"variant": "default_bundle_sampling", "http_code": 200, "wall_tps": 44.8},
            {"variant": "temp1_topk0", "http_code": 200, "wall_tps": 46.0},
            {"variant": "temp1_topk64", "http_code": 200, "wall_tps": 46.2},
        ],
        45.0,
    )

    assert result["status"] == "fail"
    assert result["default_median_tps"] == 44.4


def test_gemma4_speed_gate_flags_large_top_k_delta():
    result = summarize_results(
        [
            {"variant": "default_bundle_sampling", "http_code": 200, "wall_tps": 45.0},
            {"variant": "default_bundle_sampling", "http_code": 200, "wall_tps": 46.0},
            {"variant": "temp1_topk0", "http_code": 200, "wall_tps": 52.0},
            {"variant": "temp1_topk64", "http_code": 200, "wall_tps": 40.0},
        ],
        45.0,
    )

    assert result["status"] == "pass"
    assert result["top_k_primary_regression"] is True
