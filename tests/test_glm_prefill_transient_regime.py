import pytest

from vmlx_engine.utils.prefill_admission import (
    PrefillAdmissionError,
    hybrid_chunk_valve_check,
    replace_chunk_transient_observation,
)


@pytest.mark.parametrize("family", ["glm5_next", "glm5_next_text"])
def test_completed_smaller_glm_chunk_replaces_wider_peak(family):
    assert replace_chunk_transient_observation(family, 373, 512, 656, 1024)
    assert not replace_chunk_transient_observation(family, 373, 1024, 656, 1024)
    assert not replace_chunk_transient_observation(family, 373, 2048, 656, 1024)
    assert not replace_chunk_transient_observation(family, 0, 512, 656, 1024)
    assert replace_chunk_transient_observation(family, 700, 512, 656, 1024)


@pytest.mark.parametrize("family", ["qwen3_5", "qwen3_next", "dots3", "unknown"])
def test_other_families_keep_maximum(family):
    assert not replace_chunk_transient_observation(family, 373, 512, 656, 1024)
    assert replace_chunk_transient_observation(family, 700, 512, 656, 1024)


def test_measured_glm_receipt_retains_admission_guard():
    gib = 1024**3
    def check(transient, context, active=100.78):
        hybrid_chunk_valve_check(
            int(active*gib), int(108.4*gib), int(transient*gib), context,
            4096, 0, chunk_start=3584, chunk_end=4096,
        )
    with pytest.raises(PrefillAdmissionError):
        check(6.56, 3072)
    check(3.73, 3584)
    with pytest.raises(PrefillAdmissionError):
        check(3.73, 3584, active=105)
