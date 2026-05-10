import numpy as np
import mlx.core as mx

from vmlx_engine.request import Request, RequestStatus, SamplingParams
from vmlx_engine.scheduler import Scheduler, SchedulerConfig


class _TinyCache:
    def __init__(self):
        self.tokens = []
        self.offset = 0

    @property
    def state(self):
        return (mx.array([self.offset], dtype=mx.int32),)

    @property
    def nbytes(self):
        return 0

    def extract(self, idx):
        if idx != 0:
            raise IndexError(idx)
        return self

    def filter(self, keep):
        if list(keep) != [0]:
            raise NotImplementedError("tiny cache only supports one sequence")

    def prepare(self, *args, **kwargs):
        return None

    def finalize(self):
        return None


class _TinyModel:
    def __init__(self):
        self.calls = []

    def make_cache(self):
        return [_TinyCache()]

    def __call__(self, input_ids, cache):
        tokens = np.array(input_ids).reshape(-1).astype(int).tolist()
        cache[0].tokens.extend(tokens)
        cache[0].offset += len(tokens)
        self.calls.append(tokens)
        batch, seq_len = input_ids.shape
        logits = mx.zeros((batch, seq_len, 8), dtype=mx.float32)
        logits = logits + mx.array([0, 1, 2, 7, 3, 2, 1, 0], dtype=mx.float32)
        return logits


class _TinyTokenizer:
    clean_up_tokenization_spaces = False

    def decode(self, tokens):
        return "".join(str(int(t)) for t in tokens)


class TestSingleActiveBatchGenerator:
    def test_scheduler_uses_owned_single_active_generator_for_max_one(self):
        scheduler = Scheduler(
            _TinyModel(),
            tokenizer=_TinyTokenizer(),
            config=SchedulerConfig(
                max_num_seqs=1,
                enable_prefix_cache=False,
            ),
        )

        generator = scheduler._create_batch_generator(SamplingParams())

        assert generator.__class__.__module__ == "vmlx_engine.utils.single_batch_generator"
        assert generator.__class__.__name__ == "SingleBatchGenerator"

    def test_scheduler_keeps_mlx_lm_batch_generator_for_real_multi_sequence(self):
        scheduler = Scheduler(
            _TinyModel(),
            tokenizer=_TinyTokenizer(),
            config=SchedulerConfig(
                max_num_seqs=2,
                enable_prefix_cache=False,
            ),
        )

        generator = scheduler._create_batch_generator(SamplingParams())

        assert generator.__class__.__module__ == "mlx_lm.generate"
        assert generator.__class__.__name__ == "BatchGenerator"

    def test_single_active_generator_preserves_native_cache_object(self):
        from vmlx_engine.utils.single_batch_generator import SingleBatchGenerator

        model = _TinyModel()
        native_cache = [_TinyCache()]
        generator = SingleBatchGenerator(
            model=model,
            max_tokens=2,
            stop_tokens={9},
            sampler=lambda logits: mx.argmax(logits, axis=-1),
        )

        [uid] = generator.insert([[11, 12]], caches=[native_cache], max_tokens=[2])
        prompt_responses, generation_responses = generator.next()

        assert uid == 0
        assert generation_responses == []
        assert len(prompt_responses) == 1
        assert prompt_responses[0].token == 3
        assert prompt_responses[0].prompt_cache[0] is native_cache[0]
        assert native_cache[0].tokens == [11, 12, 3]

    def test_single_active_generator_finishes_on_stop_token(self):
        from vmlx_engine.utils.single_batch_generator import SingleBatchGenerator

        generator = SingleBatchGenerator(
            model=_TinyModel(),
            max_tokens=8,
            stop_tokens={3},
            sampler=lambda logits: mx.argmax(logits, axis=-1),
        )

        generator.insert([[1, 2]], max_tokens=[8])
        prompt_responses, _ = generator.next()

        assert prompt_responses[0].token == 3
        assert prompt_responses[0].finish_reason == "stop"

    def test_scheduler_step_uses_single_active_generator_end_to_end(self):
        request = Request(
            request_id="r1",
            prompt=[1, 2],
            sampling_params=SamplingParams(max_tokens=1, temperature=0.0),
        )
        request.prompt_token_ids = [1, 2]
        request.num_prompt_tokens = 2
        scheduler = Scheduler(
            _TinyModel(),
            tokenizer=_TinyTokenizer(),
            config=SchedulerConfig(
                max_num_seqs=1,
                enable_prefix_cache=False,
            ),
        )

        scheduler.add_request(request)
        output = scheduler.step()

        assert scheduler.batch_generator.__class__.__name__ == "SingleBatchGenerator"
        stats = scheduler.get_stats()
        assert stats["engine_path"] == "single_active"
        assert stats["batch_generator"]["single_active_decode"] is True
        assert output.outputs[0].new_token_ids == [3]
        assert output.outputs[0].finish_reason == "length"
        assert request.status == RequestStatus.FINISHED_LENGTH_CAPPED
