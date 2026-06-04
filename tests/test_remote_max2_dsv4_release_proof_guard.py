from tests.cross_matrix.run_remote_max2_dsv4_release_proof_guard import aggregate_status


def test_remote_max2_release_proof_guard_reports_blocked_no_launch():
    result = aggregate_status(
        {
            "readiness": {"status": "prepared_but_blocked", "launch_decision": "do_not_launch"},
            "exactness": {"status": "skipped_not_ready", "launch_decision": "do_not_launch"},
            "real_ui": {"status": "skipped_not_ready", "launch_decision": "do_not_launch"},
        }
    )

    assert result["status"] == "blocked_no_launch"
    assert result["release_ready"] is False


def test_remote_max2_release_proof_guard_reports_ready_when_children_ready():
    result = aggregate_status(
        {
            "readiness": {"status": "ready", "launch_decision": "launch_allowed"},
            "exactness": {"status": "ready_to_launch", "launch_decision": "launch_allowed"},
            "real_ui": {"status": "ready_to_launch", "launch_decision": "launch_allowed"},
        }
    )

    assert result["status"] == "ready_or_pass"
    assert result["release_ready"] is True


def test_remote_max2_release_proof_guard_detects_partial_launch_attempt():
    result = aggregate_status(
        {
            "readiness": {"status": "prepared_but_blocked", "launch_decision": "do_not_launch"},
            "exactness": {"status": "ready_to_launch", "launch_decision": "launch_allowed"},
            "real_ui": {"status": "skipped_not_ready", "launch_decision": "do_not_launch"},
        }
    )

    assert result["status"] == "partial_launch_attempted"
    assert result["release_ready"] is False
