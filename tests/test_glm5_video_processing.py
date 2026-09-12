"""Native GLM temporal packing and mixed-modality placeholder ownership."""

from types import SimpleNamespace
import re

import mlx.core as mx
import numpy as np
import pytest
from PIL import Image

from vmlx_engine.models.glm5_next import processing


class Tokenizer:
    image_token = "<|image|>"
    video_token = "<|video|>"
    tokens = {
        "<|image|>": 10, "<|video|>": 11,
        "<|begin_of_video|>": 12, "<|end_of_video|>": 13,
        "<|begin_of_image|>": 14, "<|end_of_image|>": 15,
    }

    def convert_tokens_to_ids(self, token):
        return self.tokens[token]

    def __call__(self, text, **kwargs):
        self.text = text
        self.kwargs = kwargs
        rows = [[self.tokens.get(t, 99) for t in re.findall(r"<\|.*?\|>|.", s)] for s in text]
        width = max(map(len, rows))
        return {"input_ids": np.array([r + [0] * (width - len(r)) for r in rows]),
                "attention_mask": np.array([[1] * len(r) + [0] * (width - len(r)) for r in rows])}


def make_processor(cls=None):
    from vmlx_engine.models.glm5_next.processing import Glm5NextVideoProcessor

    processor = object.__new__(cls or processing.Glm5NextProcessor)
    processor.tokenizer = Tokenizer()
    processor.image_processor = processing.Glm5NextImageProcessor(min_image_tokens=1)
    processor.video_processor = Glm5NextVideoProcessor(min_image_tokens=1)
    processor.image_token, processor.video_token = "<|image|>", "<|video|>"
    processor.image_token_id, processor.video_token_id = 10, 11
    return processor


def test_real_processor_mixin_initialization_accepts_native_video_component():
    from tokenizers import Tokenizer as FastTokenizer
    from tokenizers.models import WordLevel
    from transformers import PreTrainedTokenizerFast
    from vmlx_engine.models.glm5_next.processing import Glm5NextVideoProcessor

    tokenizer = PreTrainedTokenizerFast(tokenizer_object=FastTokenizer(WordLevel({"[UNK]": 0}, unk_token="[UNK]")))
    tokenizer.add_special_tokens({"additional_special_tokens": list(Tokenizer.tokens)})
    video = Glm5NextVideoProcessor()
    processor = processing.Glm5NextProcessor(
        image_processor=processing.Glm5NextImageProcessor(), tokenizer=tokenizer,
        video_processor=video,
    )
    assert processor.video_processor is video
    assert processor.image_token_id == tokenizer.convert_tokens_to_ids("<|image|>")


@pytest.mark.parametrize("count", [1, 2, 3, 4, 5])
@pytest.mark.parametrize("layout", ["tchw", "thwc", "frames"])
def test_temporal_pairs_patch_order_and_last_frame_padding(count, layout):
    from vmlx_engine.models.glm5_next.processing import Glm5NextVideoProcessor

    processor = Glm5NextVideoProcessor(min_image_tokens=1, do_rescale=False, do_normalize=False)
    frames = np.arange(count * 28 * 28 * 3, dtype=np.uint32).reshape(count, 28, 28, 3).astype(np.uint8)
    clip = np.moveaxis(frames, -1, 1) if layout == "tchw" else frames
    if layout == "frames":
        clip = [Image.fromarray(f) for f in frames]
    output = processor(videos=[clip])
    padded = list(frames) + ([frames[-1]] if count % 2 else [])
    # Independent scalar indexing in the native (t, merged spatial, C, temporal, h, w) order.
    reference = []
    for start in range(0, len(padded), 2):
        for gh in range(2):
            for gw in range(2):
                reference.append([padded[start + dt][gh * 14 + ph, gw * 14 + pw, c]
                                  for c in range(3) for dt in range(2)
                                  for ph in range(14) for pw in range(14)])
    np.testing.assert_array_equal(output["pixel_values_videos"], np.array(reference, dtype=np.float32))
    assert output["video_grid_thw"].tolist() == [[len(padded) // 2, 2, 2]]


def test_native_video_timestamps_and_mixed_image_token_types():
    processor = make_processor()
    result = processor(
        text="<|begin_of_image|><|image|><|end_of_image|> then <|begin_of_video|><|video|><|end_of_video|>",
        images=[Image.new("RGB", (28, 28), "red")],
        videos=[np.zeros((4, 3, 28, 28), dtype=np.uint8)],
        fps=[1], video_timestamps=[[0.0, 1.25, 2.5, 3.75]],
    )
    assert processor.tokenizer.text == [
        "<|begin_of_image|><|image|><|end_of_image|> then <|begin_of_video|>"
        "<|begin_of_image|><|image|><|end_of_image|>0.0 seconds"
        "<|begin_of_image|><|image|><|end_of_image|>2.5 seconds<|end_of_video|>"
    ]
    ids, kinds = np.array(result["input_ids"]), np.array(result["mm_token_type_ids"])
    assert kinds[ids == 10].tolist() == [1, 2, 2]
    assert not np.any(ids == 11)
    assert "fps" not in processor.tokenizer.kwargs
    assert "video_timestamps" not in processor.tokenizer.kwargs
    assert result["image_grid_thw"].tolist() == [[1, 2, 2]]
    assert result["video_grid_thw"].tolist() == [[2, 2, 2]]


def test_jang_wrapper_retains_native_processor_and_explicit_timestamps():
    from jang_tools.load_jangtq_vlm import _install_video_fallback
    from vmlx_engine.mllm_batch_generator import _call_processor_direct_unscoped

    class Wrapped(processing.Glm5NextProcessor):
        pass

    processor = make_processor(Wrapped)
    _install_video_fallback(processor)
    try:
        result = _call_processor_direct_unscoped(
            processor, prompts="<|begin_of_video|><|video|><|end_of_video|>", images=[],
            videos=[np.zeros((4, 3, 28, 28), dtype=np.uint8)], video_fps=[1],
            video_timestamps=[[0, 1.25, 2.5, 3.75]],
            add_special_tokens=False,
        )
        assert "2.5 seconds" in processor.tokenizer.text[0]
        assert np.asarray(result["video_grid_thw"]).tolist() == [[2, 2, 2]]
    finally:
        del Wrapped.__call__


def test_two_clips_keep_native_temporal_groups_and_per_clip_timestamps():
    processor = make_processor()
    result = processor(
        text="<|begin_of_video|><|video|><|end_of_video|> then <|begin_of_video|><|video|><|end_of_video|>",
        videos=[np.zeros((3, 3, 28, 28), dtype=np.uint8), np.full((2, 3, 56, 28), 255, dtype=np.uint8)],
        video_timestamps=[[0, 0.5, 1.0], [3.25, 4.0]],
    )
    assert result["video_grid_thw"].tolist() == [[2, 2, 2], [1, 4, 2]]
    assert result["pixel_values_videos"].shape == (16, 1176)
    assert processor.tokenizer.text[0].count("0.0 seconds") == 1
    assert "1.0 seconds" in processor.tokenizer.text[0]
    assert "3.2 seconds" in processor.tokenizer.text[0]
    assert np.count_nonzero(np.array(result["mm_token_type_ids"]) == 2) == 4


def test_video_canvas_budget_counts_padded_temporal_frames_and_preserves_config():
    from vmlx_engine.models.glm5_next.processing import Glm5NextVideoProcessor

    processor = Glm5NextVideoProcessor(min_image_tokens=1, max_image_tokens=16)
    result = processor([np.zeros((5, 3, 256, 256), dtype=np.uint8)])
    assert int(np.prod(result["video_grid_thw"][0])) // processor.merge_size**2 <= 16
    assert processor.max_image_tokens == 16
    assert result["video_grid_thw"][0, 0] == 3


@pytest.mark.parametrize("clip", [np.zeros((3, 28, 28)), np.zeros((0, 3, 28, 28)), []])
def test_invalid_video_shape_is_not_silently_an_image(clip):
    processor = make_processor()
    with pytest.raises(ValueError, match="GLM video"):
        processor.video_processor([clip])


def test_video_configuration_is_loaded_without_changing_image_configuration(tmp_path, monkeypatch):
    import json

    (tmp_path / "processor_config.json").write_text(json.dumps({
        "image_processor": {"patch_size": 14, "min_image_tokens": 16, "max_image_tokens": 8000},
        "video_processor": {"video_processor_type": "Glm5NextVideoProcessor", "patch_size": 7,
                            "temporal_patch_size": 4, "merge_size": 2, "max_image_tokens": 400, "fps": 3},
    }))
    monkeypatch.setattr(processing.AutoTokenizer, "from_pretrained", lambda *a, **k: Tokenizer())
    monkeypatch.setattr(processing, "load_chat_template", lambda *a: None)
    monkeypatch.setattr(processing.GlmOcrProcessor, "__init__", lambda self, **kw: self.__dict__.update(kw))
    result = processing.Glm5NextProcessor.from_pretrained(tmp_path)
    assert result.image_processor.max_image_tokens == 8000
    assert result.video_processor.max_image_tokens == 400
    assert result.video_processor.patch_size == 7
    assert result.video_processor.temporal_patch_size == 4
    assert result.video_processor.fps == 3


def test_mixed_video_features_scatter_only_into_video_segment():
    from vmlx_engine.models.glm5_next.vlm import Model

    class Vision:
        patch_embed = SimpleNamespace(proj=SimpleNamespace(weight=mx.zeros((1,), dtype=mx.float32)))

        def __call__(self, pixels, grid):
            return pixels

    owner = SimpleNamespace(
        language_model=SimpleNamespace(model=SimpleNamespace(embed_tokens=lambda ids: mx.zeros((*ids.shape, 2)))),
        vision_tower=Vision(), config=SimpleNamespace(image_token_id=10, video_token_id=11,
                                                    video_start_token_id=12, video_end_token_id=13),
        _scatter_features=Model._scatter_features,
    )
    result = Model.get_input_embeddings(
        owner, mx.array([[10, 12, 10, 10, 13, 99]]),
        pixel_values=mx.array([[1.0, 2.0]]), image_grid_thw=mx.array([[1, 2, 2]]),
        pixel_values_videos=mx.array([[3.0, 4.0], [5.0, 6.0]]), video_grid_thw=mx.array([[2, 2, 2]]),
    )
    mx.eval(result.inputs_embeds)
    assert result.inputs_embeds.tolist() == [[[1, 2], [0, 0], [3, 4], [5, 6], [0, 0], [0, 0]]]
