import json
from pathlib import Path

import mlx.core as mx
import pytest

from vmlx_engine.models.zaya import Model, ModelArgs, register_mlx_lm_zaya


def _small_args():
    return ModelArgs(
        hidden_size=16,
        num_hidden_layers=4,
        ffn_hidden_size=32,
        num_attention_heads=4,
        num_key_value_heads=2,
        num_query_groups=2,
        cca_num_q_heads=4,
        kv_channels=4,
        vocab_size=64,
        num_experts=3,
        zaya_mlp_expansion=8,
        max_position_embeddings=128,
    )


def test_zaya_runtime_registers_as_mlx_lm_model():
    register_mlx_lm_zaya()

    import mlx_lm.models.zaya as zaya

    assert zaya.Model is Model


def test_zaya_cca_cache_carries_kv_conv_and_prev_hidden_state():
    model = Model(_small_args())
    cache = model.make_cache()

    out = model(mx.array([[1, 2, 3]]), cache=cache)
    mx.eval(out)

    assert out.shape == (1, 3, 64)
    assert cache[0][0].offset == 3
    assert [None if v is None else tuple(v.shape) for v in cache[0][1].cache] == [
        (1, 2, 24),
        (1, 1, 16),
    ]

    out2 = model(mx.array([[4]]), cache=cache)
    mx.eval(out2)

    assert out2.shape == (1, 1, 64)
    assert cache[0][0].offset == 4
    assert [None if v is None else tuple(v.shape) for v in cache[0][1].cache] == [
        (1, 2, 24),
        (1, 1, 16),
    ]


def test_local_zaya_mxfp4_loads_strictly_when_present():
    model_dir = Path("/Users/eric/jang/models/Zyphra/ZAYA1-8B-MXFP4")
    if not model_dir.exists():
        pytest.skip("local ZAYA MXFP4 bundle is not present")

    from vmlx_engine.loaders.load_zaya import load_zaya_model

    model, tokenizer = load_zaya_model(model_dir, lazy=True)

    assert len(model.layers) == 80
    assert type(model.layers[0].self_attn).__name__ == "ZayaCCAAttention"
    assert type(model.layers[1].zaya_block).__name__ == "ZayaMoE"
    assert tokenizer is not None

    cfg = json.loads((model_dir / "config.json").read_text())
    assert cfg["model_type"] == "zaya"


def test_zaya_xml_tool_parser_registered_and_parses_native_format():
    from vmlx_engine.tool_parsers import ToolParserManager

    parser = ToolParserManager.get_tool_parser("zaya_xml")()
    result = parser.extract_tool_calls(
        "Plan first.\n"
        "<zyphra_tool_call>\n"
        "<function=search_docs>\n"
        "<parameter=query>\n"
        "\"zaya cache\"\n"
        "</parameter>\n"
        "<parameter=limit>\n"
        "3\n"
        "</parameter>\n"
        "</function>\n"
        "</zyphra_tool_call>"
    )

    assert result.tools_called is True
    assert result.content == "Plan first."
    assert result.tool_calls[0]["name"] == "search_docs"
    assert json.loads(result.tool_calls[0]["arguments"]) == {
        "query": "zaya cache",
        "limit": 3,
    }
    assert ToolParserManager.get_tool_parser("zaya").supports_native_format()


def test_zaya_rendered_chat_prompt_encoding_does_not_append_eos():
    """Rendered chat templates already carry BOS/turn markers.

    ZAYA uses GemmaTokenizerFast with add_eos_token=True. Calling encode()
    with tokenizer defaults appends <|im_end|> after the assistant generation
    prompt, so the server stops after one EOS token. The chat path must encode
    rendered templates with add_special_tokens=False.
    """

    class ZayaLikeTokenizer:
        def encode(self, text, add_special_tokens=True):
            ids = [2] + [ord(c) % 1000 for c in text]
            if add_special_tokens:
                ids.append(106)
            return ids

    from vmlx_engine.engine.batched import BatchedEngine
    from vmlx_engine.scheduler import Scheduler

    prompt = "<bos><|im_start|>assistant\n<think>\n</think>\n\n"
    tok = ZayaLikeTokenizer()

    assert Scheduler._encode_prompt_text(tok, prompt, None)[-1] == 106
    assert Scheduler._encode_prompt_text(tok, prompt, False)[-1] != 106
    assert BatchedEngine._encode_rendered_prompt(tok.encode, prompt)[-1] != 106
