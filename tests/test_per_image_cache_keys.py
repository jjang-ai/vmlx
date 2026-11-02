# SPDX-License-Identifier: Apache-2.0
"""A new image only prefills the new part of the chat, and rows of one image can never answer for another.

One key covered every image in the request, so adding an image changed the key of every row and each screenshot
re-prefilled the whole conversation. Each image now has its own key, active from its first placeholder.
"""

from __future__ import annotations

import pytest

import mlx.core as mx

from vmlx_engine.cache_key import cache_extra_keys_for_token_range, merge_cache_extra_keys, per_item_key_fallbacks
from vmlx_engine.models.minimax_m3 import prefill_geometry as G

IMG = G.MEDIA_TOKEN_IDS[0]


def _px(value, patches=64):
    return mx.full((patches, 4), float(value), dtype=mx.float32)


def _tokens(*runs):
    out = [1, 2, 3]
    for n in runs:
        out += [IMG] * n + [7, 8, 9]
    return out + [5]


def _keys(tokens, *values):
    pv = mx.concatenate([_px(v) for v in values], axis=0)
    grid = mx.array([[1, 8, 8]] * len(values), dtype=mx.int32)
    return G.per_image_cache_keys(tokens, pv, grid)


def test_rows_before_a_new_image_keep_their_keys():
    two = _tokens(16, 16)
    one = two[: two.index(IMG) + 16 + 3]
    a, b = _keys(one + [5], 1), _keys(two, 1, 2)
    second_image = two.index(IMG) + 16 + 3
    assert cache_extra_keys_for_token_range(a, 0, second_image) == cache_extra_keys_for_token_range(b, 0, second_image)
    assert cache_extra_keys_for_token_range(b, 0, second_image + 1) != cache_extra_keys_for_token_range(a, 0, second_image + 1)


def test_another_image_changes_rows_from_that_image_on_and_not_before():
    t = _tokens(16, 16)
    first, second = t.index(IMG), t.index(IMG) + 16 + 3
    a, b = _keys(t, 1, 2), _keys(t, 1, 3)
    assert cache_extra_keys_for_token_range(a, 0, second) == cache_extra_keys_for_token_range(b, 0, second)
    assert cache_extra_keys_for_token_range(a, 0, second + 1) != cache_extra_keys_for_token_range(b, 0, second + 1)
    c = _keys(t, 9, 2)
    assert cache_extra_keys_for_token_range(a, 0, first) == cache_extra_keys_for_token_range(c, 0, first)
    assert cache_extra_keys_for_token_range(a, 0, first + 1) != cache_extra_keys_for_token_range(c, 0, first + 1)


def test_swapping_two_images_is_a_different_chat():
    t = _tokens(16, 16)
    assert _keys(t, 1, 2) != _keys(t, 2, 1)


@pytest.mark.parametrize("why, tokens, pv, grid", [
    ("fewer runs than images", _tokens(16), mx.concatenate([_px(1), _px(2)]), mx.array([[1, 8, 8], [1, 8, 8]], dtype=mx.int32)),
    ("run length does not fit", _tokens(16, 9), mx.concatenate([_px(1), _px(2)]), mx.array([[1, 8, 8], [1, 8, 8]], dtype=mx.int32)),
    ("pixels do not fit the grid", _tokens(16), _px(1, 60), mx.array([[1, 8, 8]], dtype=mx.int32)),
    ("no placeholders", [1, 2, 3], _px(1), mx.array([[1, 8, 8]], dtype=mx.int32)),
])
def test_anything_that_does_not_line_up_gives_no_per_image_keys(why, tokens, pv, grid):
    assert G.per_image_cache_keys(tokens, pv, grid) is None, why


def test_reuse_never_stops_inside_an_image():
    t = _tokens(16, 16)
    first = t.index(IMG)
    second = first + 16 + 3
    boundary = len(t) - 1
    assert G.media_reuse(t, first, boundary) == first
    assert G.media_reuse(t, first + 5, boundary) == first
    assert G.media_reuse(t, first + 16, boundary) == first + 16
    assert G.media_reuse(t, second + 1, boundary) == second
    assert G.media_reuse(t, second + 16, boundary) == second + 16
    assert G.media_reuse(t, boundary + 9, boundary) == boundary
    assert G.media_reuse([1, 2, 3], 3, 3) == 0, "no placeholder found proves nothing"


def test_tail_gets_only_the_images_it_holds():
    t = _tokens(16, 4)
    pv = mx.concatenate([_px(1, 64), _px(2, 16)], axis=0)
    grid = mx.array([[1, 8, 8], [1, 4, 4]], dtype=mx.int32)
    first = t.index(IMG)
    second = first + 16 + 3
    tail_pv, tail_grid = G.tail_pixels(t, second, pv, grid)
    assert tail_pv.shape == (16, 4) and tail_grid.tolist() == [[1, 4, 4]]
    assert mx.all(tail_pv == 2).item()
    tail_pv, tail_grid = G.tail_pixels(t, first, pv, grid)
    assert tail_pv.shape == (80, 4) and tail_grid.shape == (2, 3)
    assert G.tail_pixels(t, first + 1, pv, grid) is None, "inside an image"
    assert G.tail_pixels(t, second + 4, pv, grid) is None, "nothing left to splice: the caller replays text"


def test_fallbacks_drop_the_newest_image_first():
    t = _tokens(16, 16)
    keys = _keys(t, 1, 2)
    fb = list(per_item_key_fallbacks(keys))
    assert len(fb) == 2
    one = t[: t.index(IMG) + 16 + 3] + [5]
    assert fb[0] == _keys(one, 1), "the earlier turn's whole-entry RAM key"
    assert "media_item_0" not in fb[1] and "media_item_1" not in fb[1]
    assert list(per_item_key_fallbacks({"a": 1})) == []
    assert list(per_item_key_fallbacks(None)) == []


def test_merge_keeps_both_sets_of_scopes():
    a = {"x": 1, "__vmlx_cache_extra_scopes_v1__": {"x": 5}}
    b = {"y": 2, "__vmlx_cache_extra_scopes_v1__": {"y": 9}}
    m = merge_cache_extra_keys(a, b)
    assert m["x"] == 1 and m["y"] == 2 and m["__vmlx_cache_extra_scopes_v1__"] == {"x": 5, "y": 9}
    assert a == {"x": 1, "__vmlx_cache_extra_scopes_v1__": {"x": 5}}


def test_deepseek_keys_are_per_image():
    from types import SimpleNamespace as NS

    from vmlx_engine.models.deepseek_v4_1.dsv41_vl_preprocess import media_cache_extra_keys

    def img(start, value):
        return NS(n_vit_h=8, n_vit_w=8, types=[1] * 16, start=start, patches=mx.full((1, 4), float(value), dtype=mx.float32))

    a, b = media_cache_extra_keys([img(10, 1)]), media_cache_extra_keys([img(10, 1), img(60, 2)])
    assert cache_extra_keys_for_token_range(a, 0, 60) == cache_extra_keys_for_token_range(b, 0, 60)
    assert cache_extra_keys_for_token_range(b, 0, 61) != cache_extra_keys_for_token_range(a, 0, 61)
    c = media_cache_extra_keys([img(10, 3), img(60, 2)])
    assert cache_extra_keys_for_token_range(b, 0, 11) != cache_extra_keys_for_token_range(c, 0, 11)
    assert cache_extra_keys_for_token_range(b, 0, 10) == cache_extra_keys_for_token_range(c, 0, 10)
