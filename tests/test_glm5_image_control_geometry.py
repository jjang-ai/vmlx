"""Native GLM canvas diagnostics must not consume its loader size sentinel."""

from types import SimpleNamespace

import pytest
from PIL import Image

from vmlx_engine import image_controls as ic
from vmlx_engine.models.glm5_next.processing import Glm5NextImageProcessor


def test_native_geometry_ignores_loader_longest_edge_sentinel():
    processor = Glm5NextImageProcessor()
    # The compatibility sentinel is not a one-pixel image ceiling.
    assert processor.size == {"longest_edge": 1}
    assert ic.image_processor_geometry(processor) == (784, 12544, 6272000)


@pytest.mark.parametrize("height,width", [(28, 28), (112, 112), (256, 256), (280, 280), (307, 489), (800, 800)])
@pytest.mark.parametrize("wrapped", [False, True])
def test_diagnostics_match_actual_native_canvas(height, width, wrapped):
    processor = Glm5NextImageProcessor(max_image_tokens=256)
    owner = SimpleNamespace(image_processor=processor) if wrapped else processor
    output = processor(Image.new("RGB", (width, height), "red"))
    t, gh, gw = output["image_grid_thw"][0].tolist()
    canvas = gh * processor.patch_size, gw * processor.patch_size
    tokens = t * gh * gw // processor.merge_size**2
    geometry = ic.image_processor_geometry(owner)
    assert processor.image_control_size(height, width) == canvas
    reports, unmet = ic.image_controls_diagnostics(
        ic.ImageControls(resized_height=height, resized_width=width),
        before=(height, width), after=(height, width),
        token_pixels=geometry[0], pixel_floor=geometry[1], pixel_ceiling=geometry[2],
        processor=owner,
    )
    if canvas == (height, width):
        assert reports == unmet == []
    else:
        assert len(reports) == len(unmet) == 1
        assert f"effective {canvas[0]}x{canvas[1]}" in reports[0]
        assert f"{tokens} tokens" in reports[0]
        assert "(1 px)" not in reports[0]


def test_native_bounds_follow_loaded_patch_merge_and_token_values():
    processor = Glm5NextImageProcessor(
        patch_size=7, merge_size=2, temporal_patch_size=4,
        min_image_tokens=7, max_image_tokens=32,
    )
    assert ic.image_processor_geometry(processor) == (196, 1372, 6272)
    processor.min_image_tokens = 9
    processor.max_image_tokens = 48
    assert ic.image_processor_geometry(processor) == (196, 1764, 9408)


def test_request_callsite_uses_native_canvas_for_strict_controls(tmp_path):
    from vmlx_engine.errors import MediaControlsUnmeetableError
    from vmlx_engine.mllm_batch_generator import MLLMBatchGenerator, _release_derived_media_files

    generator = object.__new__(MLLMBatchGenerator)
    generator.processor = SimpleNamespace(image_processor=Glm5NextImageProcessor())
    source = tmp_path / "badge.png"
    Image.new("RGB", (512, 512), "red").save(source)
    request = SimpleNamespace(request_id="glm-native-canvas-test")
    try:
        with pytest.raises(MediaControlsUnmeetableError, match="effective 280x280.*100 tokens"):
            generator._apply_image_controls(
                request, str(source), ic.ImageControls(resized_height=256, resized_width=256), strict=True,
            )
        path = generator._apply_image_controls(
            request, str(source), ic.ImageControls(resized_height=280, resized_width=280), strict=True,
        )
        with Image.open(path) as result:
            assert result.size == (280, 280)
        with Image.open(source) as original:
            assert original.size == (512, 512)
    finally:
        assert _release_derived_media_files(request) == 2
