import numpy as np

from tests.cross_matrix.dsv4_real_activation_expert_probe import (
    _cosine,
    _last_token_rows,
    _relative_l2,
    _score_weighted_metrics,
    _token_rows,
)


def test_relative_l2_and_cosine_metrics_are_stable():
    actual = np.array([1.0, 2.0, 2.0], dtype=np.float32)
    expected = np.array([1.0, 1.0, 2.0], dtype=np.float32)

    assert np.isclose(_relative_l2(actual, expected), 1.0 / np.linalg.norm(expected))
    assert np.isclose(
        _cosine(actual, expected),
        float(np.dot(actual, expected) / (np.linalg.norm(actual) * np.linalg.norm(expected))),
    )


def test_last_token_rows_extracts_each_hc_lane_and_expert_slot():
    x = np.arange(1 * 3 * 2 * 4, dtype=np.float32).reshape(1, 3, 2, 4)
    indices = np.array([[[[1, 2], [3, 4]], [[5, 6], [7, 8]], [[9, 10], [11, 12]]]], dtype=np.uint32)
    y = np.arange(1 * 3 * 2 * 2 * 4, dtype=np.float32).reshape(1, 3, 2, 2, 4)
    scores = np.full((1, 3, 2, 2), 0.25, dtype=np.float32)

    rows = _last_token_rows(x=x, indices=indices, runtime_y=y, scores=scores)

    assert [row["expert"] for row in rows] == [9, 10, 11, 12]
    assert [row["hc_lane"] for row in rows] == [0, 0, 1, 1]
    assert [row["expert_slot"] for row in rows] == [0, 1, 0, 1]
    assert [row["score"] for row in rows] == [0.25, 0.25, 0.25, 0.25]
    assert np.array_equal(rows[0]["x"], x[0, -1, 0])
    assert np.array_equal(rows[-1]["runtime_y"], y[0, -1, 1, 1])


def test_token_rows_can_extract_all_positions_from_non_laned_shape():
    x = np.arange(1 * 2 * 3, dtype=np.float32).reshape(1, 2, 3)
    indices = np.array([[[1, 2], [3, 4]]], dtype=np.uint32)
    y = np.arange(1 * 2 * 2 * 3, dtype=np.float32).reshape(1, 2, 2, 3)

    rows = _token_rows(x=x, indices=indices, runtime_y=y, scores=None, token_indices=None)

    assert [(row["token_index"], row["expert"]) for row in rows] == [
        (0, 1),
        (0, 2),
        (1, 3),
        (1, 4),
    ]
    assert all(row["hc_lane"] == 0 for row in rows)


def test_score_weighted_metrics_groups_rows_by_token_and_lane():
    rows = [
        {
            "batch": 0,
            "token_index": 0,
            "hc_lane": 0,
            "score": 0.25,
            "runtime_y": np.array([2.0, 0.0], dtype=np.float32),
            "source_y": np.array([1.0, 0.0], dtype=np.float32),
        },
        {
            "batch": 0,
            "token_index": 0,
            "hc_lane": 0,
            "score": 0.75,
            "runtime_y": np.array([0.0, 2.0], dtype=np.float32),
            "source_y": np.array([0.0, 1.0], dtype=np.float32),
        },
    ]

    metrics = _score_weighted_metrics(rows)

    assert metrics["group_count"] == 1
    assert np.isclose(metrics["rel_l2_mean"], 1.0)
    assert np.isclose(metrics["cosine_mean"], 1.0)
