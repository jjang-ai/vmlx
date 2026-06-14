"""Regression gate for MiniMax-M3 runtime decode scheduling."""

import contextlib
import io
import os
import sys

import mlx.core as mx

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from runtime import generate


class _CacheEntry:
    def __init__(self):
        self.offset = 0


class _FakeModel:
    def __init__(self):
        self.cache_entry = _CacheEntry()
        self.calls = 0

    def make_cache(self):
        return [self.cache_entry] * 4

    def __call__(self, inputs, cache=None):
        self.calls += 1
        if cache is not None:
            cache[3].offset += inputs.shape[-1]
        token = min(self.calls, 4)
        logits = mx.full((1, inputs.shape[-1], 8), -1000.0, dtype=mx.float32)
        logits[:, -1, token] = 1000.0
        return logits


class _FakeTokenizer:
    chat_template = None

    def encode(self, prompt):
        return [0, 0]

    def decode(self, tokens):
        return "".join(str(t) for t in tokens)


def main():
    model = _FakeModel()
    tok = _FakeTokenizer()
    with contextlib.redirect_stdout(io.StringIO()):
        text = generate(model, tok, eos_ids=set(), prompt="x", max_tokens=3, stream=False)
    assert text == "123", text
    assert model.cache_entry.offset == 5, model.cache_entry.offset
    assert model.calls == 4, model.calls
    print("OK: runtime generate schedules every emitted token into the cache")


if __name__ == "__main__":
    main()
