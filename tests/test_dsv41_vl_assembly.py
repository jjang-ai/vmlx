"""DeepSeek-V4.1 VL prompt assembly and chunked image splicing (CPU only, no model weights).

- images stay in the user turn the client sent them in (the old code moved every history image into
  the last user turn);
- tools are attached like the text path's apply_chat_template shim;
- the media salt separates different pixels and is stable for the same pixels;
- the SingleBatchGenerator splices each image into the prefill chunk holding its span, with a
  chunk-relative start, and never splits a span across chunks (the old code passed every image only
  with the pos==0 chunk, using absolute starts).
"""

from __future__ import annotations

import importlib.util
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace

import mlx.core as mx
import pytest


from vmlx_engine.models.deepseek_v4_1 import dsv41_vl_preprocess as P  # noqa: E402
from vmlx_engine.utils.single_batch_generator import SingleBatchGenerator  # noqa: E402

mlx_model = pytest.importorskip("jang_tools.dsv4_1.mlx_model")


@pytest.fixture(autouse=True)
def _cpu_only():
    """CPU only (another process owns the GPU); restore the previous default afterwards."""
    prev = mx.default_device()
    mx.set_default_device(mx.cpu)
    yield
    mx.set_default_device(prev)

IMG_A = "data:image/png;base64,AAAA"
IMG_B = "data:image/png;base64,BBBB"


def _source():
    return [
        {"role": "system", "content": "sys"},
        {"role": "user", "content": [
            {"type": "text", "text": "what is this?"},
            {"type": "image_url", "image_url": {"url": IMG_A}},
        ]},
        {"role": "assistant", "content": "a cat"},
        {"role": "user", "content": "and its colour?"},
    ]


def _stripped():  # what extract_multimodal_content hands the engine
    return [
        {"role": "system", "content": "sys"},
        {"role": "user", "content": "what is this?"},
        {"role": "assistant", "content": "a cat"},
        {"role": "user", "content": "and its colour?"},
    ]


def test_images_stay_in_their_original_turn():
    out, placed = P.anchor_images_to_turns(_stripped(), _source())
    assert placed == 1
    assert out[1]["content"] == [
        {"type": "text", "text": "what is this?"},
        {"type": "image_url", "image_url": {"url": IMG_A}},
    ]
    assert out[3]["content"] == "and its colour?"  # the follow-up turn carries no image
    # old behaviour, for contrast: everything lands in the last user turn
    old = P._inject_extra_images(_stripped(), [IMG_A])
    assert isinstance(old[3]["content"], list) and isinstance(old[1]["content"], str)


def test_rewritten_text_keeps_image_in_turn_and_mismatch_declines():
    stripped = _stripped()
    stripped[1]["content"] = "what is this? (rewritten)"
    out, placed = P.anchor_images_to_turns(stripped, _source())
    assert placed == 1 and out[1]["content"][0]["type"] == "image_url"
    assert out[1]["content"][1] == {"type": "text", "text": "what is this? (rewritten)"}
    assert P.anchor_images_to_turns(stripped[:2], _source()) is None


def test_tools_attached_like_text_shim():
    tools = [{"type": "function", "function": {"name": "f"}}]
    with_sys = P._attach_tools(_stripped(), tools)
    assert with_sys[0]["tools"] == tools and with_sys[0]["content"] == "sys"
    no_sys = P._attach_tools(_stripped()[1:], tools)
    assert no_sys[0] == {"role": "system", "content": "", "tools": tools}


@dataclass
class Img:
    start: int
    patches: mx.array
    n_vit_h: int
    n_vit_w: int
    types: list


def test_media_salt_tracks_pixels():
    a = Img(3, mx.ones((4, 4), mx.bfloat16), 2, 2, [1, 0, 0, 2])
    a2 = Img(9, mx.ones((4, 4), mx.bfloat16), 2, 2, [1, 0, 0, 2])
    b = Img(3, mx.zeros((4, 4), mx.bfloat16), 2, 2, [1, 0, 0, 2])
    assert P.media_cache_extra_keys([a]) == P.media_cache_extra_keys([a2])
    assert P.media_cache_extra_keys([a]) != P.media_cache_extra_keys([b])
    assert P.media_cache_extra_keys([]) is None


_ENC = Path.home() / "models/DeepSeek-V4.1-Flash-1017-JANG-Q2Q3-IMATRIX/encoding/encoding.py"


@pytest.mark.skipif(not _ENC.exists(), reason="bundle encoder not present")
def test_bundle_encoder_renders_image_in_first_turn():
    spec = importlib.util.spec_from_file_location("dsv41_enc_test", _ENC)
    enc = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(enc)
    msgs, _ = P.anchor_images_to_turns(_stripped(), _source())
    prompt, mm = enc.encode_messages(
        P._normalize_messages_for_encoding(msgs), thinking_mode="thinking",
        reasoning_effort="high", return_multi_modal_data=True,
    )
    assert len(mm["images"]) == 1
    first, second = prompt.split("and its colour?")[0], prompt.split("and its colour?")[1]
    assert enc.IMAGE_PLACEHOLDER in first and enc.IMAGE_PLACEHOLDER not in second
    assert prompt.index(enc.IMAGE_PLACEHOLDER) < prompt.index("a cat")


class RecordingRuntime:
    def __init__(self):
        self.args = SimpleNamespace(num_hidden_layers=2, compress_ratios=[0, 1])
        self.calls = []

    def forward(self, ids, sess, all_logits=False, return_hidden=False, images=None, token_types=None,
                engram_lookahead=None):
        start = sess.pos
        self.calls.append((start, len(ids), [(im.start, len(im.types), im.tag) for im in images or []],
                           None if token_types is None else list(token_types)))
        sess.pos += len(ids)
        logits = mx.zeros((len(ids), 11))
        return logits if all_logits else logits[-1]


@dataclass
class TagImg:
    start: int
    types: list
    tag: str
    patches: object = None
    n_vit_h: int = 1
    n_vit_w: int = 1


def test_images_are_spliced_chunk_relative_and_never_split():
    rt = RecordingRuntime()
    model = mlx_model.Model(mlx_model.ModelArgs()).attach(rt)
    n = 40
    tt = [-1] * n
    imgs = [TagImg(3, [1, 0, 0, 2], "a"), TagImg(14, [1, 0, 0, 0, 0, 2], "b"), TagImg(30, [1, 0, 2], "c")]
    for im in imgs:
        tt[im.start:im.start + len(im.types)] = im.types
    gen = SingleBatchGenerator(model, max_tokens=1, prefill_step_size=8)
    gen.insert([list(range(100, 100 + n))], max_tokens=[1], dsv41_vl_images=[imgs], dsv41_vl_token_types=[tt])
    gen.next()
    seen = {}
    for start, length, passed, ttypes in rt.calls:
        if ttypes is not None:
            assert ttypes == tt[start:start + length]
        for rel, span, tag in passed:
            assert 0 <= rel and rel + span <= length  # inside this chunk
            seen[tag] = start + rel
        for im in imgs:  # no chunk boundary inside an image span
            s, e = im.start, im.start + len(im.types)
            assert not (s < start < e) and not (s < start + length < e)
    assert seen == {"a": 3, "b": 14, "c": 30}
