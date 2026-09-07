"""Per-request image controls: validation, request-local bounding, processor-floor reporting, strict mode, and
cache-identity coverage — the image counterpart of the video controls, plus the explicit unsupported diagnostic for
image_token_budget on non-Gemma processors (never silently ignored)."""
import inspect
import os

import pytest
from PIL import Image

from vmlx_engine import image_controls as ic
from vmlx_engine.errors import MediaControlsUnmeetableError


def test_validation_names_the_field():
    with pytest.raises(ValueError, match="image_max_pixels"):
        ic.validate_image_controls({"image_max_pixels": 0})
    with pytest.raises(ValueError, match="image_min_pixels must be <= image_max_pixels"):
        ic.validate_image_controls({"image_min_pixels": 200, "image_max_pixels": 100})
    with pytest.raises(ValueError, match="together"):
        ic.validate_image_controls({"image_resized_height": 224})
    with pytest.raises(ValueError, match="exceeds image_max_pixels"):
        ic.validate_image_controls({"image_resized_height": 400, "image_resized_width": 400, "image_max_pixels": 1000})
    c = ic.validate_image_controls({"image_max_pixels": 100_000, "image_min_pixels": 50_000})
    assert (c.max_pixels, c.min_pixels, c.resize) == (100_000, 50_000, None)
    assert ic.validate_image_controls({}).is_unset


def test_target_size_precedence_explicit_then_max_then_min():
    assert ic.target_size(480, 640, ic.ImageControls(resized_height=100, resized_width=200, max_pixels=10)) == (100, 200)
    h, w = ic.target_size(480, 640, ic.ImageControls(max_pixels=100_000))
    assert h * w <= 100_000 and abs(h / w - 480 / 640) < 0.01
    h, w = ic.target_size(48, 64, ic.ImageControls(min_pixels=100_000))
    assert h * w >= 100_000
    assert ic.target_size(480, 640, ic.ImageControls(max_pixels=10_000_000)) == (480, 640)  # never upscaled by max


def test_bound_image_file_writes_a_request_local_copy_and_leaves_the_source(tmp_path):
    src = tmp_path / "src.png"
    Image.new("RGB", (640, 480), "red").save(src)
    new_path, before, after = ic.bound_image_file(str(src), ic.ImageControls(max_pixels=100_000), str(tmp_path / "req-1"))
    assert before == (480, 640) and after[0] * after[1] <= 100_000 and new_path != str(src)
    assert Image.open(src).size == (640, 480)  # source untouched
    assert Image.open(new_path).size == (after[1], after[0])
    same, b2, a2 = ic.bound_image_file(str(src), ic.ImageControls(max_pixels=10_000_000), str(tmp_path / "req-2"))
    assert same == str(src) and b2 == a2  # nothing to do: original path


def test_diagnostics_report_the_processor_floor_and_flag_unmeetable():
    # Qwen image processor: 1024 px/token, floor 65536 px
    reports, unmeetable = ic.image_controls_diagnostics(ic.ImageControls(max_pixels=20_000), before=(480, 640), after=(122, 163), token_pixels=1024, pixel_floor=65536, pixel_ceiling=16_777_216)
    assert len(reports) == 1 and "below the image processor floor (65536 px)" in reports[0] and "effective" in reports[0]
    assert unmeetable == reports
    reports, unmeetable = ic.image_controls_diagnostics(ic.ImageControls(max_pixels=200_000), before=(480, 640), after=(387, 516), token_pixels=1024, pixel_floor=65536, pixel_ceiling=16_777_216)
    assert reports == [] and unmeetable == []  # honoured: nothing to report


def test_fragment_and_cache_identity_cover_every_control():
    a = ic.ImageControls(max_pixels=100_000).fragment(); b = ic.ImageControls(max_pixels=100_001).fragment()
    c = ic.ImageControls(resized_height=224, resized_width=224).fragment()
    assert len({a, b, c}) == 3
    from vmlx_engine import mllm_batch_generator as g

    src = inspect.getsource(g._pixel_cache_prompt_key)
    assert "vmlx:image_pixels=" in src and "image_controls.fragment()" in src
    ident = inspect.getsource(g._mllm_media_item_identity) if hasattr(g, "_mllm_media_item_identity") else open(g.__file__).read()
    assert '"image_controls"' in ident and "_ic.fragment()" in ident


def test_request_models_accept_and_validate_image_controls_on_every_dialect():
    from vmlx_engine.api.models import ChatCompletionRequest
    from vmlx_engine.api import anthropic_adapter, ollama_adapter

    req = ChatCompletionRequest(model="m", messages=[{"role": "user", "content": "hi"}], image_max_pixels=100_000, media_controls_strict=True)
    assert req.image_max_pixels == 100_000 and req.media_controls_strict is True
    with pytest.raises(Exception, match="image_min_pixels must be <= image_max_pixels"):
        ChatCompletionRequest(model="m", messages=[{"role": "user", "content": "hi"}], image_min_pixels=5, image_max_pixels=1)
    from vmlx_engine.video_controls import video_control_kwargs

    kw = video_control_kwargs(req)
    assert kw["image_max_pixels"] == 100_000 and kw["media_controls_strict"] is True and kw["image_controls"].max_pixels == 100_000
    assert "image_max_pixels" in inspect.getsource(anthropic_adapter) and "IMAGE_CONTROL_FIELDS" in inspect.getsource(ollama_adapter)


def test_strict_error_maps_to_a_400_on_every_lane_and_never_a_500():
    from vmlx_engine import server
    from vmlx_engine.engine import batched

    src = open(server.__file__).read()
    assert src.count("except MediaControlsUnmeetableError as e:") >= 4
    assert "_media_controls_unmeetable_response_from_error" in src and 'status_code=400' in inspect.getsource(server._media_controls_unmeetable_response_from_error)
    assert "MediaControlsUnmeetableError.code" in inspect.getsource(batched._raise_prompt_too_long_from_output)
    assert issubclass(MediaControlsUnmeetableError, ValueError) and MediaControlsUnmeetableError.code == "media_controls_unmeetable"


def test_image_token_budget_is_reported_as_unsupported_off_gemma_and_strict_rejects():
    from vmlx_engine import mllm_batch_generator as g

    src = open(g.__file__).read()
    assert "image_token_budget_support(self.processor)" in src and "is not supported by this " in src
    assert "raise MediaControlsUnmeetableError(f\"media_controls_strict: {_msg}\"" in src

    class _Qwen:  # not gemma
        model_type = "qwen3_vl"
    class _Gemma:
        model_type = "gemma4"
    assert ic.image_token_budget_support(_Qwen()) is False and ic.image_token_budget_support(_Gemma()) is True


def test_native_clip_budget_and_fallback_plan_raise_in_strict_mode():
    from vmlx_engine import mllm_batch_generator as g
    from vmlx_engine.engine import batched

    clip = inspect.getsource(g._apply_clip_pixel_budget)
    assert "strict and \"cannot be met\" in _msg" in clip and "except MediaControlsUnmeetableError:" in clip
    fb = inspect.getsource(batched.BatchedEngine._video_frame_fallback_messages)
    assert "strict and (\"cannot be met\" in _msg or \"below the image processor floor\" in _msg)" in fb
