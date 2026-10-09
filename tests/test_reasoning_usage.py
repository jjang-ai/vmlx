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

def test_unknown_is_not_a_fabricated_zero():
    assert count_reasoning_tokens(None,Parser(),None) is None
    parser=SimpleNamespace(reasoning_tag_token_seqs=lambda _: {})
    assert count_reasoning_tokens([1,2],parser,None) is None

def test_eos_is_not_reasoning_when_generation_ends_inside_think():
    tokenizer=SimpleNamespace(eos_token_id=99,eos_token_ids={99})
    assert count_reasoning_tokens([10,11,1,99],Parser(),tokenizer)==1
