from types import SimpleNamespace
import pytest
from vmlx_engine.reasoning.usage import count_reasoning_tokens

class Parser:
    _think_in_prompt=False
    def reasoning_tag_token_seqs(self, tokenizer):
        return {'start':[[10,11]],'end':[[12,13]]}

@pytest.mark.parametrize('tokens,expected',[
    ([10,11,1,2,12,13,3],2),
    ([3,10,11,1,12,13,4,10,11,2,3,12,13],3),
    ([10,11,1,2],2),
    ([10,11,1,12],1),
    ([1,2,3],0),
])
def test_counts_original_ids_excluding_delimiters(tokens,expected):
    assert count_reasoning_tokens(tokens,Parser(),None)==expected

def test_prompt_open_reasoning_not_charged_as_generated_tokens():
    parser=Parser();parser._think_in_prompt=True
    assert count_reasoning_tokens([1,2,12,13,3],parser,None)==2

@pytest.mark.parametrize('text,expected', [
    ('private</mm:think>answer', 7),
    ('private</think>answer', 7),
    ('plain answer', 0),
    ('prefix<mm:think>private</mm:think>answer', 7),
    ('<think>private</think>answer', 7),
    ('private</mm:thi', 0),
])
def test_minimax_end_only_usage_matches_its_reasoning_contract(text, expected):
    from vmlx_engine.reasoning.minimax_m3_parser import MiniMaxM3ReasoningParser
    tokenizer = SimpleNamespace(encode=lambda text, **kwargs: list(map(ord, text)))
    parser = MiniMaxM3ReasoningParser()
    parser.reset_state(adaptive_mode=True)
    assert count_reasoning_tokens(list(map(ord, text)), parser, tokenizer) == expected

def test_other_parsers_do_not_gain_an_implicit_end_only_contract():
    assert count_reasoning_tokens([1, 2, 12, 13, 3], Parser(), None) == 0

def test_unknown_is_not_a_fabricated_zero():
    assert count_reasoning_tokens(None,Parser(),None) is None
    parser=SimpleNamespace(reasoning_tag_token_seqs=lambda _: {})
    assert count_reasoning_tokens([1,2],parser,None) is None

def test_eos_is_not_reasoning_when_generation_ends_inside_think():
    tokenizer=SimpleNamespace(eos_token_id=99,eos_token_ids={99})
    assert count_reasoning_tokens([10,11,1,99],Parser(),tokenizer)==1

@pytest.mark.parametrize("parser", [object(), SimpleNamespace(reasoning_tag_token_seqs=None)])
def test_missing_optional_parser_contract_omits_usage(parser):
    assert count_reasoning_tokens([1, 2], parser, None) is None

def test_failed_optional_parser_contract_omits_usage():
    def unavailable(_):
        raise RuntimeError("tokenizer does not expose reasoning delimiters")
    assert count_reasoning_tokens([1, 2], SimpleNamespace(reasoning_tag_token_seqs=unavailable), None) is None

@pytest.mark.asyncio
async def test_batched_stream_exports_ids_only_at_terminal(monkeypatch):
    from vmlx_engine.engine import batched
    from vmlx_engine.request import RequestOutput

    ids = [10, 11, 7, 12, 13, 8]
    class Scheduler:
        async def add_request(self, **kwargs):
            return 'usage'
        async def stream_outputs(self, request_id):
            yield RequestOutput(request_id=request_id, output_token_ids=ids[:3],
                                output_text='<think>x', new_text='<think>x', finished=False)
            yield RequestOutput(request_id=request_id, output_token_ids=ids,
                                output_text='<think>x</think>y', new_text='</think>y',
                                completion_tokens=len(ids), finished=True)
    monkeypatch.setattr(batched, '_gpu_keepalive_touch', lambda: None)
    engine = batched.BatchedEngine.__new__(batched.BatchedEngine)
    engine._loaded = True
    engine._is_mllm = False
    engine._engine = Scheduler()
    outputs = [out async for out in engine.stream_generate('hi')]
    assert outputs[0].tokens == []
    assert outputs[-1].tokens == ids
    assert outputs[-1].tokens is not ids
    assert count_reasoning_tokens(outputs[-1].tokens, Parser(), None) == 1


def test_int_eos_token_ids_does_not_raise():
    """GLM's tokenizer wrapper exposes eos_token_ids as an int; non-streaming chat completions returned HTTP 500."""
    from types import SimpleNamespace
    from vmlx_engine.reasoning.usage import count_reasoning_tokens

    tok = SimpleNamespace(eos_token_ids=2, eos_token_id=2)
    parser = SimpleNamespace(reasoning_tag_token_seqs=lambda t: {"start": [[10]], "end": [[11]]}, _think_in_prompt=False)
    assert count_reasoning_tokens([10, 5, 6, 11, 7, 2], parser, tok) == 2


def test_record_usage_never_raises(monkeypatch):
    from types import SimpleNamespace
    import vmlx_engine.server as server
    import vmlx_engine.reasoning.usage as U
    monkeypatch.setattr(U, "count_reasoning_tokens", lambda *a, **k: (_ for _ in ()).throw(RuntimeError("boom")))
    out = SimpleNamespace(output_token_ids=[1, 2, 3], reasoning_tokens=None)
    monkeypatch.setattr(server, "_output_token_ids_for_reasoning", lambda o: [1, 2, 3])
    server._record_reasoning_token_usage(out, object(), object())     # must not raise
    assert out.reasoning_tokens is None
