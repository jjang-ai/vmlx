from tests.cross_matrix.run_remote_max2_dsv4_readiness import (
    free_plus_speculative_purgeable_gb,
    launch_decision,
    parse_vm_stat,
)


def test_remote_max2_dsv4_readiness_parses_vm_stat_memory():
    parsed = parse_vm_stat(
        """
Mach Virtual Memory Statistics: (page size of 16384 bytes)
Pages free:                             1000.
Pages speculative:                       200.
Pages purgeable:                          50.
Pages active:                           9999.
"""
    )

    assert parsed["page_size"] == 16384
    assert parsed["pages"]["free"] == 1000
    assert parsed["pages"]["speculative"] == 200
    assert parsed["pages"]["purgeable"] == 50
    assert free_plus_speculative_purgeable_gb(parsed) == round(1250 * 16384 / (1024**3), 2)


def test_remote_max2_dsv4_readiness_blocks_when_memory_below_threshold():
    decision = launch_decision(
        version_ok=True,
        import_ok=True,
        model_present=True,
        available_gb=76.78,
        required_available_gb=120.0,
    )

    assert decision["launch_allowed"] is False
    assert decision["launch_decision"] == "do_not_launch"
    assert decision["launch_blockers"] == ["insufficient_memory"]


def test_remote_max2_dsv4_readiness_blocks_stale_or_unimportable_source():
    decision = launch_decision(
        version_ok=False,
        import_ok=False,
        model_present=True,
        available_gb=140.0,
        required_available_gb=120.0,
    )

    assert decision["launch_allowed"] is False
    assert decision["launch_blockers"] == ["version_mismatch", "source_import_failed"]
