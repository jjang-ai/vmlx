from vmlx_engine.scheduler import Scheduler


def test_generated_only_logits_processor_trims_prefill_prompt_context():
    seen = []

    def processor(tokens, logits):
        seen.append(list(tokens))
        return logits

    wrapped = Scheduler._wrap_generated_only_logits_processor(
        processor, skip_prefix_tokens=4
    )

    assert wrapped([10, 11, 12, 13, 14], "logits") == "logits"
    assert wrapped([10, 11, 12, 13, 14, 15], "logits") == "logits"

    # First decode step keeps the final prompt token. Later steps keep that
    # final prompt token plus generated tokens, matching mlx_lm.generate_step.
    assert seen == [[14], [14, 15]]


def test_generated_only_logits_processor_preserves_one_token_when_skip_is_too_large():
    seen = []

    def processor(tokens, logits):
        seen.append(list(tokens))
        return logits

    wrapped = Scheduler._wrap_generated_only_logits_processor(
        processor, skip_prefix_tokens=99
    )

    wrapped([1, 2, 3], "logits")

    assert seen == [[3]]
