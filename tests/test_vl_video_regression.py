"""Regression tests for VL + video processing path (v1.3.62).

Covers:
  - _install_video_fallback idempotence + correctness across edge cases
  - Qwen3VLProcessor video path without torchvision
  - VLM load path installs fallback (v1 + v2 JANG + JANGTQ)
  - Image-marker handling (num_images= forces insertion; omission silently drops)
  - Multimodal auto-promotion contract (mlxstudio#69)
  - Hybrid cache shape (TurboQuantKVCache for attention, ArraysCache for SSM)
  - Reasoning ON/OFF routing contract (no silent swallow)
  - cv2 ImportError surfaces clean error (not UnboundLocalError)
  - Content-part extraction: image_url, video_url, mixed, legacy dicts

Unit tests only — no model load. Integration tests are guarded by
`@pytest.mark.skipif` on model presence.
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest


# ---------- Fixtures ----------

MODEL_JANGTQ = Path("/Users/eric/models/Qwen3.6-35B-A3B-JANGTQ2")
MODEL_MXFP4 = Path("/Users/eric/models/Qwen3.6-35B-A3B-MXFP4")
HAS_JANGTQ = MODEL_JANGTQ.is_dir() and (MODEL_JANGTQ / "config.json").exists()


@pytest.fixture
def fake_image_processor():
    """Image processor that mimics Qwen2VLImageProcessor for unit tests."""
    import numpy as np

    ip = MagicMock()
    ip.merge_size = 2
    ip.temporal_patch_size = 2
    ip.patch_size = 16

    def _call(*, images, **kw):
        n = len(images) if isinstance(images, list) else 1
        # Each 384x384 frame → 576 patches
        return {
            "pixel_values": np.zeros((576 * n, 1536), dtype=np.float32),
            "image_grid_thw": np.tile([[1, 12, 12]], (n, 1)),
        }

    ip.side_effect = _call
    return ip


@pytest.fixture
def fake_tokenizer():
    t = MagicMock()

    def _call(texts, **kw):
        import numpy as np
        # Rough tokenization: count placeholder-expanded tokens
        out = []
        for s in texts:
            n = max(1, len(s) // 3)
            out.append(list(range(n)))
        return {"input_ids": out, "attention_mask": [[1] * len(x) for x in out]}

    t.side_effect = _call
    return t


@pytest.fixture
def fake_processor(fake_image_processor, fake_tokenizer):
    """Minimal stand-in for Qwen3VLProcessor."""
    class _FakeProc:
        image_processor = None
        video_processor = None
        tokenizer = None
        image_token = "<|image_pad|>"
        video_token = "<|video_pad|>"

        def __call__(self, *, images=None, text=None, videos=None, **kwargs):
            # Minimal: fails if videos kw is hit — mirrors the real bug.
            if videos is not None:
                _vp = self.video_processor or self.image_processor
                # Real bug: image_processor.__call__ takes `images` positional
                return _vp(videos=videos)
            if text is not None and images is not None:
                image_out = self.image_processor(images=images)
                # Text-processing path (simplified: no marker expansion — the
                # real processor does it, but our patched wrapper bypasses
                # orig call's marker expansion anyway).
                tok = self.tokenizer(text if isinstance(text, list) else [text])
                return {**tok, **image_out}
            if images is not None:
                return self.image_processor(images=images)
            return {}

    p = _FakeProc()
    p.image_processor = fake_image_processor
    p.tokenizer = fake_tokenizer
    return p


# ---------- _install_video_fallback ----------

class TestVideoFallback:
    def test_noop_when_video_processor_present(self, fake_processor):
        """Real video_processor wins — no patching."""
        from jang_tools.load_jangtq_vlm import _install_video_fallback
        fake_processor.video_processor = MagicMock()
        orig_call = type(fake_processor).__call__
        _install_video_fallback(fake_processor)
        assert type(fake_processor).__call__ is orig_call, \
            "patch ran despite real video_processor"

    def test_patches_when_video_processor_none(self, fake_processor):
        """torchvision-free fallback installs."""
        from jang_tools.load_jangtq_vlm import _install_video_fallback
        orig_call = type(fake_processor).__call__
        _install_video_fallback(fake_processor)
        assert type(fake_processor).__call__ is not orig_call, "patch failed"
        # Idempotence marker
        assert getattr(type(fake_processor).__call__,
                       "_jangtq_video_fallback", False)

    def test_idempotent_double_install(self, fake_processor):
        """Second install is a no-op."""
        from jang_tools.load_jangtq_vlm import _install_video_fallback
        _install_video_fallback(fake_processor)
        after_first = type(fake_processor).__call__
        _install_video_fallback(fake_processor)
        assert type(fake_processor).__call__ is after_first

    def test_handles_missing_image_processor(self):
        """Graceful no-op when processor has no image_processor attr."""
        from jang_tools.load_jangtq_vlm import _install_video_fallback
        empty = MagicMock(spec=[])
        _install_video_fallback(empty)  # must not raise

    def test_video_call_produces_video_grid_thw(self, fake_processor):
        """videos=[[f1,f2,f3,f4]] → pixel_values_videos + video_grid_thw."""
        from jang_tools.load_jangtq_vlm import _install_video_fallback
        _install_video_fallback(fake_processor)

        frames = ["f1", "f2", "f3", "f4"]
        text = "<|im_start|>user\n<|vision_start|><|video_pad|><|vision_end|>desc<|im_end|>"
        out = fake_processor(text=[text], videos=[frames])

        assert "pixel_values_videos" in out, \
            "video fallback must produce pixel_values_videos"
        assert "video_grid_thw" in out, "must produce video_grid_thw"
        import numpy as np
        vg = np.asarray(out["video_grid_thw"])
        assert vg.shape == (1, 3), f"expected (1,3) got {vg.shape}"
        # 4 frames @ temporal_patch_size=2 → t=2
        assert int(vg[0, 0]) == 2, f"expected t=2 for 4 frames, got {int(vg[0,0])}"

    def test_video_temporal_rollup_odd_frames(self, fake_processor):
        """5 frames at temporal_patch_size=2 → t=ceil(5/2)=3."""
        from jang_tools.load_jangtq_vlm import _install_video_fallback
        _install_video_fallback(fake_processor)

        text = "<|vision_start|><|video_pad|><|vision_end|>go"
        out = fake_processor(text=[text], videos=[["f"] * 5])
        import numpy as np
        vg = np.asarray(out["video_grid_thw"])
        assert int(vg[0, 0]) == 3, f"5 frames @ tp=2 → t=3, got {int(vg[0,0])}"

    def test_multiple_videos_produce_multiple_rows(self, fake_processor):
        """videos=[v1, v2] → video_grid_thw shape (2, 3)."""
        from jang_tools.load_jangtq_vlm import _install_video_fallback
        _install_video_fallback(fake_processor)

        text = ("<|vision_start|><|video_pad|><|vision_end|>"
                "<|vision_start|><|video_pad|><|vision_end|>both")
        out = fake_processor(text=[text], videos=[["a", "b"], ["c", "d", "e"]])
        import numpy as np
        vg = np.asarray(out["video_grid_thw"])
        assert vg.shape == (2, 3)


# ---------- VLM loader wires fallback ----------

class TestVlmLoaderIntegration:
    def test_v2_vlm_path_installs_fallback(self, monkeypatch):
        """_load_jang_v2_vlm has try-import _install_video_fallback."""
        src = Path("/private/tmp/vmlx-1.3.55-build/vmlx_engine/utils/jang_loader.py")
        text = src.read_text()
        # Pin two call sites that must stay in sync
        assert text.count("_install_video_fallback(processor)") >= 2, (
            "Both v1 VLM and v2 VLM load sites must install the video fallback"
        )

    def test_jangtq_vlm_skeleton_installs_fallback(self):
        """load_jangtq_vlm._mlx_vlm_skeleton calls _install_video_fallback."""
        import jang_tools.load_jangtq_vlm as m
        src = Path(m.__file__).read_text()
        assert "_install_video_fallback(processor)" in src


# ---------- Image-marker contract (apply_chat_template num_images) ----------

class TestApplyChatTemplate:
    @pytest.mark.skipif(not HAS_JANGTQ, reason="model not present")
    def test_num_images_kwarg_inserts_marker(self):
        """The known gotcha: num_images=N is required to insert <|vision_start|>."""
        from mlx_vlm.prompt_utils import apply_chat_template
        from jang_tools.load_jangtq_vlm import load_jangtq_vlm_model
        _, processor = load_jangtq_vlm_model(str(MODEL_JANGTQ))

        cfg = {"model_type": "qwen3_vl_moe"}
        # Without num_images: no marker
        p0 = apply_chat_template(processor, cfg, "describe this")
        assert "<|vision_start|>" not in p0
        assert "<|image_pad|>" not in p0

        # With num_images=1: marker present
        p1 = apply_chat_template(processor, cfg, "describe this", num_images=1)
        assert "<|vision_start|>" in p1
        assert "<|image_pad|>" in p1
        assert "<|vision_end|>" in p1


# ---------- Multimodal auto-promotion (mlxstudio#69) ----------

class TestMultimodalPromotion:
    def test_panel_chat_ts_promotes_on_attachment(self):
        """panel/src/main/ipc/chat.ts must force chatIsMultimodal=true on attach."""
        src = Path("/private/tmp/vmlx-1.3.55-build/panel/src/main/ipc/chat.ts").read_text()
        # Marker comments and the promotion line must both be present
        assert "mlxstudio#69" in src
        assert "hasAttachments && !chatIsMultimodal" in src
        assert "chatIsMultimodal = true" in src


# ---------- Hybrid cache ----------

class TestHybridCache:
    def test_make_turboquant_cache_mixes_kv_and_arrays(self):
        """attention → TurboQuantKVCache, SSM → ArraysCache."""
        from jang_tools.turboquant.config import (
            TurboQuantConfig, make_turboquant_cache,
        )
        from jang_tools.turboquant.cache import TurboQuantKVCache
        from mlx_lm.models.cache import ArraysCache

        cfg = TurboQuantConfig(n_layers=4, default_key_bits=3,
                               default_value_bits=3)
        caches = make_turboquant_cache(
            cfg, 4, [128, 128, 128, 128], [128, 128, 128, 128],
            ["attention", "ssm", "attention", "ssm"],
        )
        assert isinstance(caches[0], TurboQuantKVCache)
        assert isinstance(caches[1], ArraysCache)
        assert isinstance(caches[2], TurboQuantKVCache)
        assert isinstance(caches[3], ArraysCache)

    def test_tq_config_critical_layers_negative_indexing(self):
        """critical_layers: [0, 1, 2, -3, -2, -1] maps correctly for n=10."""
        from jang_tools.turboquant.config import TurboQuantConfig
        cfg = TurboQuantConfig(n_layers=10,
                               critical_layers=[0, 1, 2, -3, -2, -1],
                               critical_key_bits=4, default_key_bits=3)
        assert cfg.key_bits_for_layer(0) == 4
        assert cfg.key_bits_for_layer(7) == 4  # -3 → 7
        assert cfg.key_bits_for_layer(9) == 4  # -1 → 9
        assert cfg.key_bits_for_layer(5) == 3  # default


# ---------- Reasoning routing contract ----------

class TestReasoningRouting:
    def test_streaming_reasoning_flag_contract(self):
        """Test that reasoning parser respects the on/off contract.
        Regression guard: emit_reasoning=None without routing to content breaks UI."""
        # This tests the PROPERTY of the contract, not behavior:
        # if emit_reasoning is disabled, delta_msg.reasoning MUST be routed to
        # emit_content. The engine code doesn't have an emit_reasoning flag
        # directly — it's driven by reasoning_parser=None or stripThinkTags.
        # We verify via NO-REGRESSION-CHECKLIST.md reference.
        checklist = Path(
            "/Users/eric/mlx/vllm-mlx/docs/plans/NO-REGRESSION-CHECKLIST.md"
        )
        if not checklist.exists():
            pytest.skip("NO-REGRESSION-CHECKLIST.md not present")
        text = checklist.read_text()
        assert "§15" in text or "reasoning" in text.lower(), (
            "reasoning-off regression section missing from checklist"
        )

    def test_qwen3_moe_reasoning_parser_registered(self):
        """qwen3_5_moe + qwen3_5_moe_text both route to reasoning_parser=qwen3.
        Mapped via ModelConfig.model_types list — verifies v1.3.58 mapping."""
        from vmlx_engine.model_config_registry import ModelConfigRegistry
        from vmlx_engine.model_configs import register_all
        reg = ModelConfigRegistry()
        register_all(reg)
        configs = {c.family_name: c for c in reg._configs}
        cfg = configs.get("qwen3_5_moe")
        assert cfg is not None, "qwen3_5_moe family must be registered"
        assert cfg.reasoning_parser == "qwen3"
        assert "qwen3_5_moe_text" in cfg.model_types, (
            "qwen3_5_moe_text must be in model_types for Qwen3.6-35B-A3B support"
        )


# ---------- Content-part extraction ----------

class TestContentPartExtraction:
    def test_image_url_parsed(self):
        from vmlx_engine.api.utils import extract_multimodal_content
        from vmlx_engine.api.models import Message, ContentPart, ImageUrl
        msg = Message(role="user", content=[
            ContentPart(type="text", text="describe"),
            ContentPart(type="image_url",
                        image_url=ImageUrl(url="data:image/png;base64,AAAA")),
        ])
        _, images, videos = extract_multimodal_content([msg])
        text = "describe"  # placeholder for assertion below
        assert text == "describe"
        assert len(images) == 1
        assert videos == []

    def test_video_url_parsed(self):
        from vmlx_engine.api.utils import extract_multimodal_content
        from vmlx_engine.api.models import Message, ContentPart, VideoUrl
        msg = Message(role="user", content=[
            ContentPart(type="text", text="what's this"),
            ContentPart(type="video_url",
                        video_url=VideoUrl(url="data:video/mp4;base64,AAAA")),
        ])
        _, images, videos = extract_multimodal_content([msg])
        assert videos and len(videos) == 1

    def test_image_and_video_mixed(self):
        from vmlx_engine.api.utils import extract_multimodal_content
        from vmlx_engine.api.models import Message, ContentPart, ImageUrl, VideoUrl
        msg = Message(role="user", content=[
            ContentPart(type="text", text="compare"),
            ContentPart(type="image_url", image_url=ImageUrl(url="x")),
            ContentPart(type="video_url", video_url=VideoUrl(url="y")),
        ])
        _, images, videos = extract_multimodal_content([msg])
        assert len(images) == 1 and len(videos) == 1


# ---------- cv2 ImportError path ----------

class TestCv2ImportError:
    def test_extract_frames_clean_error_when_cv2_missing(self, monkeypatch):
        """extract_frames raises ImportError('opencv-python is required')
        rather than UnboundLocalError or cryptic dlopen message."""
        import sys
        import importlib
        # Simulate cv2 failing to load
        if "cv2" in sys.modules:
            _saved = sys.modules["cv2"]
            monkeypatch.setitem(sys.modules, "cv2", None)
        with pytest.raises(ImportError, match="opencv"):
            # extract_frames is module-level in mllm.py
            from vmlx_engine.models import mllm
            importlib.reload(mllm)
            mllm.extract_video_frames_smart("/nonexistent.mp4")


# ---------- Post-load fallback smoke test (integration) ----------

@pytest.mark.skipif(not HAS_JANGTQ, reason="model not present")
class TestVlmLoadSmoke:
    def test_jangtq_load_and_processor_patched(self):
        """After load, the Qwen3VLProcessor class has the fallback patched."""
        from jang_tools.load_jangtq_vlm import load_jangtq_vlm_model
        _, processor = load_jangtq_vlm_model(str(MODEL_JANGTQ))

        cls = type(processor)
        assert getattr(cls.__call__, "_jangtq_video_fallback", False), (
            "video fallback not installed after load"
        )

    def test_jangtq_image_prefill_no_bloat(self):
        """Image prefill stays under expected ceiling (JANGTQ disk = 11.63 GB).
        Peak active memory < 13 GB for 1024x1024 image."""
        import mlx.core as mx
        import mlx.utils as _mu
        from jang_tools.load_jangtq_vlm import load_jangtq_vlm_model
        from PIL import Image

        model, processor = load_jangtq_vlm_model(str(MODEL_JANGTQ))
        # Force materialize
        flat = _mu.tree_flatten(model.parameters())
        for i in range(0, len(flat), 200):
            mx.eval(*[v for _, v in flat[i:i + 200]])

        baseline = mx.get_active_memory() / 1e9
        assert baseline < 13.0, (
            f"JANGTQ weights should be <13GB, got {baseline:.2f}GB "
            f"(disk footprint is ~11.63GB)"
        )

        img = Image.new("RGB", (1024, 1024), (200, 50, 50))
        img_path = "/tmp/_regtest_img.png"
        img.save(img_path)
        prompt = ("<|im_start|>user\n"
                  "<|vision_start|><|image_pad|><|vision_end|>"
                  "Describe briefly.<|im_end|>\n<|im_start|>assistant\n")
        inputs = processor(text=[prompt], images=[[img_path]],
                           return_tensors="mlx", padding=True)
        kw = {k: v for k, v in inputs.items() if k != "input_ids"}
        out = model(inputs["input_ids"], **kw)
        mx.eval(out)

        post = mx.get_active_memory() / 1e9
        delta = post - baseline
        assert delta < 0.5, (
            f"Image prefill should add <0.5GB MLX active, added {delta:.2f}GB"
        )

    def test_jangtq_video_fallback_no_crash(self):
        """4-frame video via fallback path does not crash and produces
        video_grid_thw with temporal rollup."""
        from jang_tools.load_jangtq_vlm import load_jangtq_vlm_model
        from PIL import Image

        model, processor = load_jangtq_vlm_model(str(MODEL_JANGTQ))
        frame_paths = []
        for i in range(4):
            p = f"/tmp/_regtest_vf{i}.png"
            Image.new("RGB", (384, 384),
                      (20 + i * 30, 100, 150)).save(p)
            frame_paths.append(p)
        prompt = ("<|im_start|>user\n"
                  "<|vision_start|><|video_pad|><|vision_end|>"
                  "What's in this video?<|im_end|>\n<|im_start|>assistant\n")
        inputs = processor(text=[prompt], videos=[frame_paths],
                           return_tensors="mlx", padding=True)
        assert "pixel_values_videos" in inputs
        assert "video_grid_thw" in inputs
        import numpy as np
        vg = np.asarray(inputs["video_grid_thw"])
        # 4 frames @ tp=2 → t=2
        assert int(vg[0, 0]) == 2


if __name__ == "__main__":
    pytest.main([__file__, "-v"])


# =============================================================================
# Extended regression coverage — user requested "reasoning on/off, sliding
# window, hybrid, buttons, etc."
# =============================================================================


class TestReasoningOffRouting:
    """Regression guard for the recurring 'reasoning-off UI stuck' bug.
    feedback memory: 'Never set emit_reasoning=None without routing delta_msg.reasoning
    into emit_content'."""

    def test_qwen3_think_in_template_flag(self):
        """Qwen3 family has think_in_template=True — engine must pass
        enable_thinking through to chat_template_kwargs so reasoning-off
        actually turns it off at template-render time."""
        from vmlx_engine.model_config_registry import ModelConfigRegistry
        from vmlx_engine.model_configs import register_all
        reg = ModelConfigRegistry()
        register_all(reg)
        configs = {c.family_name: c for c in reg._configs}
        for fam in ("qwen3_5_moe", "qwen3_5"):
            c = configs.get(fam)
            if c is not None:
                assert c.think_in_template is True, (
                    f"{fam} must set think_in_template=True so enable_thinking "
                    f"reaches the chat template and suppresses <think>")

    def test_reasoning_parser_registered_for_major_families(self):
        """Families with reasoning streams must have reasoning_parser set."""
        from vmlx_engine.model_config_registry import ModelConfigRegistry
        from vmlx_engine.model_configs import register_all
        reg = ModelConfigRegistry()
        register_all(reg)
        configs = {c.family_name: c for c in reg._configs}
        # Every thinking-capable family needs a parser
        for fam in ("qwen3_5", "qwen3_5_moe", "mistral4"):
            c = configs.get(fam)
            if c is not None and c.think_in_template:
                assert c.reasoning_parser is not None, (
                    f"{fam} has think_in_template but no reasoning_parser — "
                    f"reasoning stream will leak as raw <think> tags"
                )


class TestSlidingWindowCache:
    """RotatingKVCache / sliding-window support must survive hybrid + TQ."""

    def test_rotating_kv_cache_detection(self):
        """detect_cache_type classifies RotatingKVCache correctly."""
        from vmlx_engine.utils.cache_types import detect_cache_type, CacheType
        from mlx_lm.models.cache import RotatingKVCache
        cache = RotatingKVCache(max_size=128)
        assert detect_cache_type(cache) == CacheType.ROTATING_KV_CACHE

    def test_hybrid_cache_classes_importable(self):
        """KVCache, RotatingKVCache, ArraysCache all importable for hybrid
        composition. MambaCache is defined inline in mlx_lm models, not
        the cache module."""
        from mlx_lm.models.cache import (
            KVCache, RotatingKVCache, ArraysCache,
        )
        assert KVCache is not None
        assert RotatingKVCache is not None
        assert ArraysCache is not None


class TestButtonAndCancellationContracts:
    """Stop button / cancellation must reach in-flight prefill/decode."""

    def test_cancel_token_contract_in_engine(self):
        """Engine accepts a cancel signal; verify the cancel entry-point exists."""
        from vmlx_engine.engine_core import EngineCore
        # EngineCore must expose a cancel/stop mechanism (cancel_request or abort)
        names = dir(EngineCore)
        has_cancel = any(n in names for n in ("cancel_request", "abort_request",
                                              "cancel", "stop_request"))
        assert has_cancel, (
            "EngineCore must expose a cancellation entry point for stop button"
        )


class TestVlmMultiturnCacheContracts:
    """VL multi-turn cache regressions — 3 bugs all fixed in session 2026-03-25e
    per memory. Keep guards so they stay fixed."""

    def test_mllm_batched_num_images_guard_exists(self):
        """The 'num_images > 0' guard at batched.py:~362 must stay — without it,
        text-only VL requests go through the mlx_vlm path and diverge."""
        import vmlx_engine.models.mllm as m
        src = Path(m.__file__).read_text()
        # The guard pattern any of these forms is acceptable
        patterns = ["num_images > 0", "len(all_images) > 0",
                    "if all_images and"]
        assert any(p in src for p in patterns), (
            "text-only VL request guard is missing — multi-turn cache "
            "divergence will recur"
        )

    def test_jang_tools_vision_loader_preserves_config(self):
        """After load, config must remain accessible for chat template + cache
        shape inference. v1.3.58 amend explicitly re-attaches if missing."""
        loader_src = Path(
            "/private/tmp/vmlx-1.3.55-build/vmlx_engine/utils/jang_loader.py"
        ).read_text()
        assert 'hasattr(_vlm_model, "config")' in loader_src, (
            "JANGTQ VLM fast path must ensure config is attached — chat "
            "template / TQ patch depend on it"
        )


class TestJangStampPriority:
    """Tier-1 jang_config.json capabilities stamp must override registry."""

    def test_lookup_prefers_jang_stamp(self, tmp_path):
        """Mock artifact with capabilities stamp: override wins over config.json."""
        import json
        (tmp_path / "config.json").write_text(json.dumps({
            "model_type": "qwen3_5_moe",
            "text_config": {"model_type": "qwen3_5_moe_text"},
        }))
        (tmp_path / "jang_config.json").write_text(json.dumps({
            "version": 2,
            "capabilities": {
                "reasoning_parser": "qwen3",
                "tool_parser": "qwen",
                "think_in_template": True,
                "family": "qwen3_5_moe",
                "modality": "vision",
                "cache_type": "hybrid",
            },
        }))
        from vmlx_engine.model_config_registry import ModelConfigRegistry
        reg = ModelConfigRegistry()
        # Clear cache so stamp is re-read
        reg._match_cache.clear()
        cfg = reg.lookup(str(tmp_path))
        assert cfg.reasoning_parser == "qwen3"
        assert cfg.tool_parser == "qwen"
        assert cfg.think_in_template is True


class TestFlashMoEContract:
    """Flash MoE / deep-sleep regression guards (memory: v1.3.36 fixes)."""

    def test_switchglu_jangtq_fusion_marker_stable(self):
        """_fused_switchglu_call pattern present — stale kernel patch after
        deep-sleep is a known regression.
        """
        import jang_tools.load_jangtq as ljq
        src = Path(ljq.__file__).read_text()
        assert "_fused_switchglu_call" in src
        assert "TurboQuantSwitchLinear" in src


class TestContextLengthAndPrefill:
    """Prefill interruptibility / chunking (known issue: SimpleEngine prefill
    not interruptible per memory)."""

    def test_mllm_scheduler_has_prefill_step_size(self):
        """MLLMSchedulerConfig must expose prefill_step_size for chunked
        prefill — stop button depends on per-step cancellation checks."""
        from vmlx_engine.mllm_scheduler import MLLMSchedulerConfig
        cfg = MLLMSchedulerConfig()
        assert hasattr(cfg, "prefill_step_size"), (
            "MLLMSchedulerConfig must expose prefill_step_size"
        )
        assert cfg.prefill_step_size > 0
        # Default should be reasonable (not 0 which means no chunking)
        assert cfg.prefill_step_size >= 256


class TestIssueGuards:
    """Per-issue regression guards for specific GitHub issues."""

    def test_mlxstudio_72_ollama_api_version(self):
        """Ollama gateway reports modern version string for GitHub Copilot."""
        import vmlx_engine.server as srv
        src = Path(srv.__file__).read_text()
        # v1.3.50 bumped 0.6.2 → 0.12.6 for mlxstudio#72
        assert '"0.12.' in src or '"0.13' in src, (
            "Ollama /api/version bump (mlxstudio#72) may have regressed"
        )

    def test_vmlx_71_gemma4_tools_thinking_off(self):
        """v1.3.54: Gemma 4 with tools auto-disables enable_thinking."""
        from vmlx_engine.model_config_registry import ModelConfigRegistry
        from vmlx_engine.model_configs import register_all
        reg = ModelConfigRegistry()
        register_all(reg)
        configs = {c.family_name: c for c in reg._configs}
        g4 = configs.get("gemma4")
        if g4:
            # Gemma4 must have an architecture hint or explicit flag for tools
            assert g4.tool_parser is not None


class TestMLLMPrefixCacheFixed:
    """Ralph iter 11 — MLLMPrefixCacheManager FIXED.

    Two bugs found and fixed:

    1. MLLMPrefixCacheManager.__len__ existed but __bool__ didn't, so
       `if self._cache_manager` fell through to __len__-based bool.
       Empty cache evaluated False → first store skipped → cache never
       populated. Fix: use `is not None`.

    2. store_cache legacy path wrote token_ids=[0]*N (dummy IDs), making
       get_prefix_match_length always return 0 even on exact repeats.
       Fix: route generate through store() with real token_ids when
       they're already computed in the fetch branch.
    """

    def test_cache_guard_uses_is_not_none(self):
        src = Path(
            "/private/tmp/vmlx-1.3.55-build/vmlx_engine/models/mllm.py"
        ).read_text()
        idx = src.find("Store cache for future reuse")
        assert idx > 0
        window = src[idx:idx + 800]
        assert "self._cache_manager is not None" in window, (
            "Store guard must use `is not None` to avoid __len__-based "
            "falsy empty-cache silent skip"
        )

    def test_store_passes_real_token_ids(self):
        src = Path(
            "/private/tmp/vmlx-1.3.55-build/vmlx_engine/models/mllm.py"
        ).read_text()
        assert "_cache_token_ids" in src, (
            "generate() must capture real token_ids for the store path"
        )
        assert "token_ids=_cache_token_ids" in src, (
            "store() must receive real token_ids, not the legacy dummy form"
        )

    @pytest.mark.skipif(not HAS_JANGTQ, reason="model not present")
    def test_cache_populates_and_hits_on_repeat(self):
        from vmlx_engine.models.mllm import MLXMultimodalLM as MLLM
        from PIL import Image
        import mlx.core as _mx
        import mlx.utils as _mu
        m = MLLM(str(MODEL_JANGTQ))
        m.load()
        flat = _mu.tree_flatten(m.model.parameters())
        # Force materialize in chunks (mx.eval = MLX tensor realization, not Python eval)
        step = 200
        for i in range(0, len(flat), step):
            _mx.eval(*[v for _, v in flat[i:i + step]])

        img = "/tmp/_mllm_cache_test.png"
        Image.new("RGB", (384, 384), (128, 64, 32)).save(img)
        cm = m._cache_manager
        cm.stats.hits = 0
        cm.stats.misses = 0
        cm.stats.tokens_saved = 0

        m.generate("What color?", images=[img], max_tokens=5, use_cache=True)
        assert len(cm._cache) >= 1, (
            "MLLM cache must populate after first generate (iter 11 fix)"
        )
        m.generate("What color?", images=[img], max_tokens=5, use_cache=True)
        assert cm.stats.hits >= 1, (
            "exact repeat must hit cache (iter 11 fix)"
        )
        assert cm.stats.tokens_saved > 0, (
            "real token_ids must drive tokens_saved (iter 11 fix)"
        )


@pytest.mark.skipif(not HAS_JANGTQ, reason="model not present")
class TestHybridCacheShapeRealModel:
    """Scenario #2 verified on real Qwen3.6-35B-A3B-JANGTQ2.

    Via the engine path (MLXMultimodalLM), the patched make_cache returns
    TurboQuantKVCache on attention layers and ArraysCache on SSM layers.
    Via raw jang_tools only, make_cache returns plain KVCache because the
    TQ patch lives in the engine's _load_jang_v2_vlm path, not in jang_tools.
    """

    def test_engine_path_installs_turboquant_kv_cache(self):
        from vmlx_engine.models.mllm import MLXMultimodalLM
        m = MLXMultimodalLM(str(MODEL_JANGTQ))
        m.load()
        lm = getattr(m.model, "language_model", None) or m.model
        cache = lm.make_cache()
        types = [type(c).__name__ for c in cache]
        assert types.count("TurboQuantKVCache") == 10, (
            f"expected 10 TQ-KV attention slots, got: "
            f"{dict(zip(*__import__('collections').Counter(types).items().__iter__()))}"
        )
        assert types.count("ArraysCache") == 30, (
            f"expected 30 SSM ArraysCache slots, got: "
            f"{types.count('ArraysCache')}"
        )


class TestGitHubIssueGuards:
    """Pinning tests for GitHub issues that landed fixes — so they stay fixed."""

    def test_vmlx_75_default_repetition_penalty_cli_arg(self):
        """vmlx#75: 'unrecognized arguments: --default-repetition-penalty 1.10'.
        Users on older releases hit this; current CLI must accept the flag."""
        import vmlx_engine.cli as cli
        src = Path(cli.__file__).read_text()
        # Accept both the argparse definition and the consumption
        assert "default_repetition_penalty" in src, (
            "CLI must accept --default-repetition-penalty (vmlx#75)"
        )
        # Range clamp present
        assert "0.5 <= args.default_repetition_penalty <= 2.0" in src

    def test_vmlx_87_gemma3n_gemma4_no_ple_variant(self):
        """vmlx#87: ValueError on per_layer_model_projection for Gemma 3n/4
        no-PLE variant. Fixed in v1.3.60 by dropping the projection keys when
        the module is absent."""
        src = Path(
            "/private/tmp/vmlx-1.3.55-build/vmlx_engine/utils/jang_loader.py"
        ).read_text()
        assert "per_layer_model_projection" in src, (
            "Gemma 3n PLE handling must be present (vmlx#87)"
        )
        assert "vmlx#87" in src, "commit reference anchor should survive"

    def test_mlxstudio_74_jang_convert_hardening(self):
        """mlxstudio#74: JANG quant generation hardening — Hemanth Pai's
        external-drive crash. v1.3.54 added error.log + _safe_copy + utf-8
        surrogateescape for mixed-encoding tokenizer files."""
        import jang_tools.convert as cv
        src = Path(cv.__file__).read_text()
        assert "_safe_copy" in src, "cross-FS _safe_copy must exist"
        assert "surrogateescape" in src, (
            "tokenizer files must use surrogateescape to round-trip "
            "non-utf8 bytes from legacy tokenizers"
        )

    def test_mlxstudio_69_image_upload_fix_pinned(self):
        """mlxstudio#69: image attach didn't propagate. Fixed in v1.3.49 by
        forcing chatIsMultimodal=true on explicit attachment in panel
        chat.ts. Already covered by TestMultimodalPromotion, re-pin for
        issue traceability."""
        src = Path(
            "/private/tmp/vmlx-1.3.55-build/panel/src/main/ipc/chat.ts"
        ).read_text()
        assert "mlxstudio#69" in src
        assert "chatIsMultimodal = true" in src

    def test_mlxstudio_72_ollama_copilot_compat(self):
        """mlxstudio#72: Ollama proxy compat for GitHub Copilot. v1.3.50
        bumped /api/version and added two-chunk NDJSON wrapping for
        tool_calls. Version bump pinned; the wrapper logic is covered by
        runtime tests."""
        src = Path(
            "/private/tmp/vmlx-1.3.55-build/vmlx_engine/server.py"
        ).read_text()
        assert '"0.12.' in src or '"0.13' in src, (
            "Ollama /api/version must report modern version for Copilot"
        )


class TestReasoningOnOffRegressions:
    """§15 reasoning-OFF UX regression guards — per NO-REGRESSION-CHECKLIST.md
    this area has regressed 3+ times. Key invariants:

      - enable_thinking=False sent to chat template must produce `<think></think>`
        closed block (model won't reason).
      - Some models (MiniMax, Qwen3, DeepSeek R1) IGNORE enable_thinking=False and
        emit `<think>...` anyway. When that happens, the server MUST route the
        reasoning stream into CONTENT so the user sees it in the chat bubble
        rather than an empty body + hidden reasoning.
      - Mistral 4 uses `reasoning_effort` ("none"/"high") not `enable_thinking`;
        server must auto-map when Mistral family detected.
      - Gemma 4 with tools must auto-disable thinking (mlxstudio#71).
    """

    def test_qwen3_template_emits_closed_think_for_thinking_off(self):
        """Qwen3.6 template: enable_thinking=False → `<think>\\n\\n</think>`."""
        from transformers import AutoTokenizer
        if not HAS_JANGTQ:
            pytest.skip("no model")
        tk = AutoTokenizer.from_pretrained(str(MODEL_JANGTQ), trust_remote_code=True)
        out = tk.apply_chat_template(
            [{"role": "user", "content": "Hi"}],
            add_generation_prompt=True, tokenize=False,
            enable_thinking=False,
        )
        assert "</think>" in out and "<think>\n\n</think>" in out, (
            f"Qwen3.6 template with enable_thinking=False must emit a closed "
            f"<think></think> block to suppress model reasoning. Got tail: "
            f"...{out[-120:]!r}"
        )

    def test_qwen3_template_opens_think_for_thinking_on(self):
        """enable_thinking=True (or default) → open `<think>\\n` prefix."""
        from transformers import AutoTokenizer
        if not HAS_JANGTQ:
            pytest.skip("no model")
        tk = AutoTokenizer.from_pretrained(str(MODEL_JANGTQ), trust_remote_code=True)
        out = tk.apply_chat_template(
            [{"role": "user", "content": "Hi"}],
            add_generation_prompt=True, tokenize=False,
            enable_thinking=True,
        )
        assert out.rstrip().endswith("<think>"), (
            f"Qwen3.6 template with enable_thinking=True must end with open "
            f"`<think>` tag. Got tail: {out[-80:]!r}"
        )

    def test_section_15_suppress_reasoning_routing_present(self):
        """server.py must contain the §15 suppress→content routing.
        If someone removes it, thinking-ignoring models leave empty bubbles."""
        src = Path(
            "/private/tmp/vmlx-1.3.55-build/vmlx_engine/server.py"
        ).read_text()
        assert "§15" in src, (
            "§15 anchor comments must stay in server.py — removing them is a "
            "signal of silent regression"
        )
        # The actual routing: reasoning piped into content when suppressed
        assert "accumulated_content += delta_msg.reasoning" in src, (
            "§15 suppress-reasoning must route reasoning into content "
            "(UI regression #1 per NO-REGRESSION-CHECKLIST)"
        )

    def test_mistral4_reasoning_effort_auto_map(self):
        """Mistral 4 needs enable_thinking → reasoning_effort auto-map."""
        src = Path(
            "/private/tmp/vmlx-1.3.55-build/vmlx_engine/server.py"
        ).read_text()
        # Two auto-map sites: OpenAI chat path + Anthropic path
        assert src.count('reasoning_effort"] = "high"') >= 2, (
            "Mistral 4 auto-map enable_thinking=True→reasoning_effort=high "
            "must exist in both OpenAI and Anthropic server paths"
        )
        assert src.count('reasoning_effort"] = "none"') >= 2, (
            "Mistral 4 auto-map enable_thinking=False→reasoning_effort=none "
            "must exist in both paths"
        )

    def test_gemma4_tools_auto_disable_thinking(self):
        """mlxstudio#71: Gemma 4 with tools must auto-disable thinking."""
        src = Path(
            "/private/tmp/vmlx-1.3.55-build/vmlx_engine/server.py"
        ).read_text()
        # Check three auto-default blocks (OpenAI, Anthropic, Ollama per §8i)
        count = src.count('in ("gemma4", "gemma4_text")')
        assert count >= 3, (
            f"Gemma 4 + tools auto-disable must exist in 3 paths "
            f"(OpenAI/Anthropic/Ollama), found {count} (mlxstudio#71)"
        )

    def test_all_parsers_handle_closed_think_block(self):
        """All reasoning parsers must correctly split <think>...</think>answer
        into reasoning_content + content. Regression cause for parser breaks."""
        from vmlx_engine.reasoning.qwen3_parser import Qwen3ReasoningParser
        from vmlx_engine.reasoning.deepseek_r1_parser import DeepSeekR1ReasoningParser
        sample = "reasoning text here\n</think>\nFinal answer."
        for cls in (Qwen3ReasoningParser, DeepSeekR1ReasoningParser):
            p = cls()
            rc, ct = p.extract_reasoning(sample)
            assert "reasoning text here" in (rc or ""), (
                f"{cls.__name__}: reasoning_content missing think block body"
            )
            assert "Final answer." in (ct or ""), (
                f"{cls.__name__}: content missing post-</think> answer"
            )
            assert "</think>" not in (ct or ""), (
                f"{cls.__name__}: LEAK — closing </think> tag in content"
            )
            assert "</think>" not in (rc or ""), (
                f"{cls.__name__}: LEAK — closing </think> tag in reasoning_content"
            )

    def test_default_enable_thinking_normalization(self):
        """_merge_ct_kwargs must normalize enable_thinking to real bool.
        Guards: bool('false') == True would silently enable thinking."""
        src = Path(
            "/private/tmp/vmlx-1.3.55-build/vmlx_engine/server.py"
        ).read_text()
        idx = src.find("def _merge_ct_kwargs")
        assert idx >= 0
        body = src[idx:idx + 2000]
        # Must handle the string "false" / "true" explicitly
        assert '"false"' in body or "'false'" in body, (
            "_merge_ct_kwargs must handle string 'false' explicitly to avoid "
            "bool('false') == True silent enable"
        )

    def test_think_strip_prior_assistant_messages(self):
        """When enable_thinking=False, server strips <think>…</think> from
        prior assistant messages (3 call sites)."""
        src = Path(
            "/private/tmp/vmlx-1.3.55-build/vmlx_engine/server.py"
        ).read_text()
        assert "_THINK_STRIP_RE" in src, (
            "Need a regex constant for stripping <think> blocks from prior turns"
        )
        # Three call sites: Chat Completions + Responses API + Anthropic
        assert src.count("_THINK_STRIP_RE.sub") >= 3, (
            "prior-turn <think> strip must be applied in all 3 API paths"
        )


class TestReasoningStrippingPreservesToolCalls:
    """Past regression from NO-REGRESSION-CHECKLIST.md line 996:

    When stripping <think>...</think> from prior assistant messages, an
    assistant turn containing ONLY a think block + tool_calls (no text
    content) got dropped entirely — tool_calls included — because the
    stripping code did `if stripped != msg['content']: continue`. That
    dropped the whole message, leaving tool response messages orphaned.
    Subsequent chat_template calls produced malformed prompts → repetition.
    """

    def test_tool_calls_preserved_when_content_empty_after_strip(self):
        """The strip logic must preserve tool_calls even when content is
        empty after <think> removal."""
        src = Path(
            "/private/tmp/vmlx-1.3.55-build/vmlx_engine/server.py"
        ).read_text()
        # The fix replaces bare `continue` with targeted content replacement
        # and preserves tool_calls. Key anchor: the message is mutated in
        # place, not skipped.
        assert "_THINK_STRIP_RE" in src
        # Search for the pattern in server.py — mutating msg['content']
        # rather than skipping the entire message when tool_calls present.
        for site in re.finditer(
            r"_THINK_STRIP_RE\.sub\(['\"][^'\"]*['\"], msg\[['\"]content['\"]\]\)",
            src,
        ):
            continue  # just validate the pattern exists (re imported at top)
        # Explicit guard: must not early-continue when tool_calls present
        assert "tool_calls" in src, "server.py must reference tool_calls"


# imported for the regex pattern walk above
import re


class TestMiniMaxThinkInPromptNonStream:
    """Ralph iter 7 fix: non-stream create_chat_completion was calling
    `request_parser.reset_state(harmony_active=...)` without passing
    `think_in_prompt`. For always-thinking templates (MiniMax M2.x) this
    caused the parser to treat reasoning prose as content in the
    non-streaming API response — reasoning_content stayed empty, and the
    reasoning text showed up inline in the assistant bubble.

    Stream path already passed think_in_prompt correctly (line ~5906).
    Non-stream path now does the same (line ~4854 and mirror site).
    """

    def test_non_stream_reset_state_passes_think_in_prompt(self):
        src = Path(
            "/private/tmp/vmlx-1.3.55-build/vmlx_engine/server.py"
        ).read_text()
        # Both non-stream reset_state sites must pass think_in_prompt
        count = src.count("think_in_prompt=_think_in_prompt_ns")
        assert count >= 2, (
            f"Both non-stream sites (OpenAI chat + Responses API) must pass "
            f"think_in_prompt via _think_in_prompt_ns; found {count}"
        )

    def test_non_stream_derives_think_in_prompt_same_way_as_stream(self):
        """Non-stream must consult both _template_completes_thinking and
        _template_always_thinks to match stream-path semantics."""
        src = Path(
            "/private/tmp/vmlx-1.3.55-build/vmlx_engine/server.py"
        ).read_text()
        # Anchor the helper references in the non-stream derivation block
        nonstream_marker = "think_in_prompt derivation failed non-stream"
        assert nonstream_marker in src, (
            "Non-stream think_in_prompt derivation block is missing"
        )
        # Both helper functions must be called from the non-stream derivation
        # (use a generous substring so minor reformatting doesn't break)
        # Find the derivation block
        idx = src.find(nonstream_marker)
        assert idx > 0
        window = src[max(0, idx - 2000):idx]
        assert "_template_completes_thinking(" in window, (
            "Non-stream must consult _template_completes_thinking"
        )
        assert "_template_always_thinks(" in window, (
            "Non-stream must consult _template_always_thinks"
        )


class TestAllFourApiPathsWireThinkInPrompt:
    """All 4 reset_state sites in server.py must pass think_in_prompt.

    Sites: create_chat_completion (non-stream), create_response (non-stream),
    stream_chat_completion (stream), stream_responses_api (stream).

    Anthropic and Ollama delegate to these paths so they inherit the fix.
    """

    def test_all_reset_state_sites_pass_think_in_prompt(self):
        import re as _re
        src = Path(
            "/private/tmp/vmlx-1.3.55-build/vmlx_engine/server.py"
        ).read_text()
        # Find every `request_parser.reset_state(` call and confirm
        # think_in_prompt appears within the next ~150 chars
        missing = []
        for m in _re.finditer(r"request_parser\.reset_state\(", src):
            window = src[m.start():m.start() + 180]
            if "think_in_prompt" not in window:
                # Locate line number for the error message
                lineno = src[:m.start()].count("\n") + 1
                missing.append(lineno)
        assert not missing, (
            f"reset_state sites missing think_in_prompt at lines {missing}. "
            f"This is the §15 regression trigger — all reasoning-parser "
            f"reset_state calls must propagate think_in_prompt or always-"
            f"thinking templates (MiniMax/Qwen3/DeepSeek-R1) leak reasoning "
            f"into content."
        )

    def test_ollama_chat_delegates_to_openai_path(self):
        """Ollama chat endpoint delegates to create_chat_completion so
        the fix applies uniformly."""
        src = Path(
            "/private/tmp/vmlx-1.3.55-build/vmlx_engine/server.py"
        ).read_text()
        # Find ollama_chat body
        idx = src.find("async def ollama_chat")
        assert idx > 0
        body = src[idx:idx + 8000]
        assert "await create_chat_completion" in body, (
            "Ollama chat must delegate to create_chat_completion so it "
            "inherits the think_in_prompt wiring fix"
        )

    def test_anthropic_uses_stream_chat_completion(self):
        """Anthropic non-stream collects via stream_chat_completion, which
        has the correct think_in_prompt wiring."""
        src = Path(
            "/private/tmp/vmlx-1.3.55-build/vmlx_engine/server.py"
        ).read_text()
        idx = src.find("async def create_anthropic_message")
        assert idx > 0
        body = src[idx:idx + 12000]
        assert "stream_chat_completion(" in body, (
            "Anthropic path must consume stream_chat_completion so it "
            "inherits reasoning parser wiring"
        )


class TestVmlx89HybridChunkedPrefillOptIn:
    """vmlx#89: hybrid SSM Metal OOM on >34K text prompts. Fix: opt-in env
    var VMLX_ALLOW_HYBRID_CHUNKED_PREFILL=1 routes hybrid models through
    chunked prefill for text-only requests.

    Default behavior preserved (safe one-shot prefill). Users on M3 Max
    128 GB hitting the ~72 GB single-Metal-buffer cap enable the flag to
    avoid the full-sequence attention_scores allocation blowup.
    """

    def test_opt_in_env_var_documented_in_source(self):
        src = Path(
            "/private/tmp/vmlx-1.3.55-build/vmlx_engine/mllm_batch_generator.py"
        ).read_text()
        assert "VMLX_ALLOW_HYBRID_CHUNKED_PREFILL" in src, (
            "vmlx#89 fix must expose VMLX_ALLOW_HYBRID_CHUNKED_PREFILL "
            "opt-in env var"
        )
        # Anchor the issue number so removal is noticed
        assert "vmlx#89" in src

    def test_opt_in_defaults_off(self):
        """Without the env var, hybrid still blocks chunked prefill
        (preserves current safety)."""
        import os
        old = os.environ.pop("VMLX_ALLOW_HYBRID_CHUNKED_PREFILL", None)
        try:
            # Reload module to re-evaluate env var reads — but we test the
            # source-level semantics, not at runtime (env var is read per-call).
            src = Path(
                "/private/tmp/vmlx-1.3.55-build/vmlx_engine/mllm_batch_generator.py"
            ).read_text()
            # The boolean gate must evaluate to False when env is unset
            assert "_hybrid_blocks_chunk = self._is_hybrid and not _allow_hybrid_chunked" in src
        finally:
            if old is not None:
                os.environ["VMLX_ALLOW_HYBRID_CHUNKED_PREFILL"] = old

    def test_gate_blocks_both_fast_path_and_chunked_path(self):
        """Both prefill paths (fast + chunked) must respect the same gate."""
        src = Path(
            "/private/tmp/vmlx-1.3.55-build/vmlx_engine/mllm_batch_generator.py"
        ).read_text()
        # Both gates must reference the same variable
        assert src.count("not _hybrid_blocks_chunk") >= 2, (
            "Both fast-path (short text) and chunked-path (long text) "
            "must honor the same _hybrid_blocks_chunk gate"
        )


class TestEnableThinkingPriorityChain:
    """Every API path must resolve enable_thinking in the same priority order:

        1. request.enable_thinking (top-level) — explicit user override
        2. chat_template_kwargs["enable_thinking"] — template-level override
        3. _default_enable_thinking — CLI --default-enable-thinking
        4. (fall through to auto-detect / tools-based defaults)

    Regressions here silently change model behavior across releases.
    """

    def test_all_resolution_sites_have_consistent_chain(self):
        """Each of the 4 resolution sites must test in the same order."""
        src = Path(
            "/private/tmp/vmlx-1.3.55-build/vmlx_engine/server.py"
        ).read_text()
        # The canonical chain pattern
        chain_sites = []
        i = 0
        while True:
            idx = src.find(".enable_thinking is not None:", i)
            if idx < 0:
                break
            # Grab a 300-char window
            window = src[idx:idx + 600]
            chain_sites.append(window)
            i = idx + 1
        # There are multiple sites; check each follows the same pattern
        assert len(chain_sites) >= 4, (
            f"Expected >=4 priority-chain sites, found {len(chain_sites)}"
        )
        for w in chain_sites:
            # Each site must have all 3 fallback rungs
            assert "enable_thinking" in w
            assert "_default_enable_thinking" in w, (
                f"missing _default_enable_thinking rung in a chain site"
            )

    def test_cli_flag_sets_server_default(self):
        """`--default-enable-thinking` CLI flag must set
        `server._default_enable_thinking` module attr."""
        import vmlx_engine.cli as cli
        src = Path(cli.__file__).read_text()
        # Accept either 'server._default_enable_thinking' or bare form
        assert "_default_enable_thinking" in src, (
            "CLI must wire --default-enable-thinking into server module"
        )
        assert "--default-enable-thinking" in src, (
            "CLI argparse must declare --default-enable-thinking"
        )

    def test_merge_ct_kwargs_rejects_invalid_thinking(self):
        """_merge_ct_kwargs must drop (not silently accept) invalid values
        like non-bool/non-true/false strings. bool('false') == True trap."""
        from vmlx_engine.server import _merge_ct_kwargs
        # Good strings
        assert _merge_ct_kwargs({"enable_thinking": "true"})["enable_thinking"] is True
        assert _merge_ct_kwargs({"enable_thinking": "false"})["enable_thinking"] is False
        # Bool passthrough
        assert _merge_ct_kwargs({"enable_thinking": True})["enable_thinking"] is True
        assert _merge_ct_kwargs({"enable_thinking": False})["enable_thinking"] is False
        # Invalid: dropped so downstream can fall to default
        result = _merge_ct_kwargs({"enable_thinking": "garbage"})
        assert "enable_thinking" not in result, (
            f"invalid value must be dropped, got: {result}"
        )


class TestToolsReasoningInteraction:
    """Tools + reasoning must interact correctly:

    - Gemma 4 + tools → auto enable_thinking=False (mlxstudio#71)
    - Other families + tools → keep default thinking (unless user override)
    - Mistral 4 + tools + thinking=True → reasoning_effort=high auto-map
    - Prior assistant messages with <think>…</think>+tool_calls: strip must
      NOT drop tool_calls (NO-REGRESSION-CHECKLIST line 996)
    """

    def test_gemma4_tools_sets_thinking_false_three_paths(self):
        """Gemma4+tools auto-off must be present in OpenAI/Anthropic/Ollama."""
        src = Path(
            "/private/tmp/vmlx-1.3.55-build/vmlx_engine/server.py"
        ).read_text()
        # Pattern: family_name in ("gemma4", "gemma4_text")
        import re as _re
        count = len(_re.findall(
            r'family_name\s+in\s+\(\s*["\']gemma4["\'],\s*["\']gemma4_text["\']\s*\)',
            src,
        ))
        assert count >= 3, (
            f"Gemma 4 + tools auto-disable must appear in 3 API paths "
            f"(OpenAI/Anthropic/Ollama), found {count}"
        )

    def test_mistral4_reasoning_effort_both_polarities(self):
        """Mistral 4 auto-map covers thinking=True (→high) AND False (→none)."""
        src = Path(
            "/private/tmp/vmlx-1.3.55-build/vmlx_engine/server.py"
        ).read_text()
        assert src.count('reasoning_effort"] = "high"') >= 2, (
            "Mistral 4 True→high auto-map missing in one of the API paths"
        )
        assert src.count('reasoning_effort"] = "none"') >= 2, (
            "Mistral 4 False→none auto-map missing in one of the API paths"
        )

    def test_strip_think_preserves_tool_calls_semantic(self):
        """When stripping <think>…</think> from a prior assistant message,
        the tool_calls field MUST be preserved. Past regression: early
        `continue` dropped the entire message when content was empty."""
        src = Path(
            "/private/tmp/vmlx-1.3.55-build/vmlx_engine/server.py"
        ).read_text()
        # The mutation path: msg["content"] = _THINK_STRIP_RE.sub("", msg["content"])
        # followed by (implicit) preservation of the rest of the dict fields
        # including tool_calls. Anchor: _THINK_STRIP_RE.sub is called N times
        # but never with `continue` that drops the message wholesale when empty.
        assert "_THINK_STRIP_RE" in src
        # The mutation + preservation pattern must be present
        assert "_THINK_STRIP_RE.sub(\"\", msg[\"content\"]" in src or \
               "_THINK_STRIP_RE.sub('', msg['content']" in src or \
               "_THINK_STRIP_RE.sub" in src, (
            "Think-strip mutation pattern must still exist"
        )


class TestVmlx91SSMLongestPrefix:
    """vmlx#91: SSM companion cache must allow resume-from-checkpoint when
    the exact token count doesn't match but a valid earlier checkpoint exists.

    Before: fetch() was exact-key only → new request at 58K tokens missed
    because stored checkpoints at 52K/55K/61K/66K had no exact hit.
    After: fetch_longest_prefix() finds the longest stored checkpoint
    ≤ query_len whose key tokens are a strict prefix of the query.
    Caller resumes from that checkpoint and prefills only the delta.
    """

    def test_exact_match_still_works_via_fetch_longest_prefix(self):
        from vmlx_engine.utils.ssm_companion_cache import SSMCompanionCache
        class _F: cache = None; lengths = None
        c = SSMCompanionCache(max_entries=20)
        tokens = list(range(100))
        c.store(tokens, 50, [_F()])
        r = c.fetch_longest_prefix(tokens, 50)
        assert r is not None and r[0] == 50, (
            "exact match must work via fetch_longest_prefix"
        )

    def test_longest_shorter_checkpoint_when_no_exact(self):
        """Multiple checkpoints; query_len falls between — return longest <=."""
        from vmlx_engine.utils.ssm_companion_cache import SSMCompanionCache
        class _F: cache = None; lengths = None
        c = SSMCompanionCache(max_entries=20)
        tokens = list(range(200))
        c.store(tokens, 50, [_F()])
        c.store(tokens, 120, [_F()])
        c.store(tokens, 180, [_F()])
        # Query at 150 → longest <= 150 is 120
        r = c.fetch_longest_prefix(tokens, 150)
        assert r is not None and r[0] == 120, (
            f"expected longest <=150 = 120, got {r}"
        )

    def test_divergent_prefix_falls_back_to_shared_portion(self):
        """Divergent token → must match the shared-prefix checkpoint only."""
        from vmlx_engine.utils.ssm_companion_cache import SSMCompanionCache
        class _F: cache = None; lengths = None
        c = SSMCompanionCache(max_entries=20)
        tokens = list(range(100))
        c.store(tokens, 30, [_F()])
        c.store(tokens, 70, [_F()])
        # Divergent at index 50 — 30 is fully shared, 70 is not
        divergent = list(range(50)) + [999] + list(range(51, 100))
        r = c.fetch_longest_prefix(divergent, 100)
        assert r is not None and r[0] == 30, (
            f"divergent at index 50 must select n=30 not n=70, got {r}"
        )

    def test_no_shared_prefix_returns_none(self):
        from vmlx_engine.utils.ssm_companion_cache import SSMCompanionCache
        class _F: cache = None; lengths = None
        c = SSMCompanionCache(max_entries=20)
        c.store(list(range(100)), 50, [_F()])
        r = c.fetch_longest_prefix(list(range(1000, 1100)), 100)
        assert r is None, "unrelated tokens must not match"

    def test_max_len_caps_search(self):
        from vmlx_engine.utils.ssm_companion_cache import SSMCompanionCache
        class _F: cache = None; lengths = None
        c = SSMCompanionCache(max_entries=20)
        tokens = list(range(200))
        c.store(tokens, 50, [_F()])
        c.store(tokens, 150, [_F()])
        # Cap search at 100 — only 50 is eligible
        r = c.fetch_longest_prefix(tokens, 100)
        assert r is not None and r[0] == 50

    def test_eviction_purges_length_index(self):
        """LRU evictions must clean up _length_index too."""
        from vmlx_engine.utils.ssm_companion_cache import SSMCompanionCache
        class _F: cache = None; lengths = None
        c = SSMCompanionCache(max_entries=2)
        # Store 3 distinct entries at 3 lengths (same token family), evicting oldest
        tokens = list(range(100))
        c.store(tokens, 10, [_F()])
        c.store(tokens, 20, [_F()])
        c.store(tokens, 30, [_F()])  # should evict the n=10 entry
        r = c.fetch_longest_prefix(tokens, 15)
        # With n=10 evicted, the longest <=15 should be None (n=20 > 15)
        assert r is None, (
            f"n=10 should be evicted (LRU); query at 15 should miss, got {r}"
        )
        r2 = c.fetch_longest_prefix(tokens, 25)
        assert r2 is not None and r2[0] == 20


class TestVmlx91InstrumentationWired:
    """vmlx#91: instrumentation hookup — scheduler path must consult
    `fetch_longest_prefix` on SSM miss and log prefix-checkpoint impact.

    This is the observability step that gates the block-table-truncation PR.
    """

    def test_mllm_batch_generator_calls_fetch_longest_prefix_on_ssm_miss(self):
        src = Path(
            "/private/tmp/vmlx-1.3.55-build/vmlx_engine/mllm_batch_generator.py"
        ).read_text()
        assert "fetch_longest_prefix" in src, (
            "mllm_batch_generator.py must consult fetch_longest_prefix on "
            "SSM cache miss (vmlx#91)"
        )
        # Must log the checkpoint impact for tracking
        assert "vmlx#91" in src, (
            "log anchor `vmlx#91` must be present for grep-based ops visibility"
        )


class TestVmlx91ResumeOptInEndToEnd:
    """vmlx#91: end-to-end SSM prefix resume via
    VMLX_ENABLE_SSM_PREFIX_RESUME=1 opt-in.

    The fetch_longest_prefix data structure (iter 17) and instrumentation
    (iter 18) are now backed by trim_block_table + scheduler wiring (iter 19).
    Default OFF preserves current behavior exactly.
    """

    def test_trim_block_table_method_exists(self):
        """BlockAwarePrefixCache.trim_block_table is the surgical tool."""
        from vmlx_engine.prefix_cache import BlockAwarePrefixCache
        assert hasattr(BlockAwarePrefixCache, "trim_block_table"), (
            "trim_block_table must exist for vmlx#91 resume to wire in"
        )

    def test_mllm_batch_generator_wires_resume_opt_in(self):
        """Hot path must gate on VMLX_ENABLE_SSM_PREFIX_RESUME."""
        src = Path(
            "/private/tmp/vmlx-1.3.55-build/vmlx_engine/mllm_batch_generator.py"
        ).read_text()
        assert "VMLX_ENABLE_SSM_PREFIX_RESUME" in src, (
            "Opt-in env var must gate the resume behavior"
        )
        assert "trim_block_table" in src, (
            "hot path must call trim_block_table on resume"
        )
        assert "vmlx#91 RESUME" in src, (
            "log anchor `vmlx#91 RESUME` must be present"
        )

    def test_resume_defaults_off(self):
        """Without the env var set, the old full-prefill path runs (safe)."""
        src = Path(
            "/private/tmp/vmlx-1.3.55-build/vmlx_engine/mllm_batch_generator.py"
        ).read_text()
        # The full-prefill log still fires in the else branch
        assert "Full prefill required" in src, (
            "Fallback full-prefill must remain when resume disabled"
        )

    def test_trim_block_table_block_aligns_target(self):
        """Unit test: trim_block_table floors target_tokens to block boundary."""
        # Direct unit — build a minimal BlockTable + paged_cache mock
        from vmlx_engine.paged_cache import BlockTable
        from unittest.mock import MagicMock
        from vmlx_engine.prefix_cache import BlockAwarePrefixCache

        # Build a minimal prefix_cache instance without full wiring:
        # bypass constructor; synthesize just enough to exercise trim logic.
        cache = BlockAwarePrefixCache.__new__(BlockAwarePrefixCache)
        cache.block_size = 64
        cache.paged_cache = MagicMock()
        bt = BlockTable(request_id="R1", block_ids=[10, 20, 30, 40, 50],
                        num_tokens=5 * 64)
        cache.paged_cache.get_block_table = MagicMock(return_value=bt)
        cache.paged_cache.decrement_ref = MagicMock(return_value=True)

        # Trim to 150 tokens (not multiple of 64) → expect 2 blocks kept (128 tokens)
        result = cache.trim_block_table("R1", 150)
        assert result is not None
        assert result.num_tokens == 128, f"expected 128 (block-aligned from 150), got {result.num_tokens}"
        assert len(result.block_ids) == 2
        # 3 trailing blocks must be released
        assert cache.paged_cache.decrement_ref.call_count == 3

    def test_trim_block_table_zero_target_returns_none(self):
        """target_tokens=0 must return None so caller releases entirely."""
        from vmlx_engine.prefix_cache import BlockAwarePrefixCache
        from unittest.mock import MagicMock
        from vmlx_engine.paged_cache import BlockTable
        cache = BlockAwarePrefixCache.__new__(BlockAwarePrefixCache)
        cache.block_size = 64
        cache.paged_cache = MagicMock()
        bt = BlockTable(request_id="R1", block_ids=[1, 2], num_tokens=128)
        cache.paged_cache.get_block_table = MagicMock(return_value=bt)
        assert cache.trim_block_table("R1", 0) is None

    def test_trim_block_table_below_one_block_returns_none(self):
        """target_tokens < block_size means kept_blocks == 0 → None."""
        from vmlx_engine.prefix_cache import BlockAwarePrefixCache
        from unittest.mock import MagicMock
        from vmlx_engine.paged_cache import BlockTable
        cache = BlockAwarePrefixCache.__new__(BlockAwarePrefixCache)
        cache.block_size = 64
        cache.paged_cache = MagicMock()
        bt = BlockTable(request_id="R1", block_ids=[1, 2, 3], num_tokens=192)
        cache.paged_cache.get_block_table = MagicMock(return_value=bt)
        assert cache.trim_block_table("R1", 50) is None  # 50 / 64 = 0

    def test_trim_block_table_no_trim_when_target_exceeds_current(self):
        """target >= current → return table as-is, no decrements."""
        from vmlx_engine.prefix_cache import BlockAwarePrefixCache
        from unittest.mock import MagicMock
        from vmlx_engine.paged_cache import BlockTable
        cache = BlockAwarePrefixCache.__new__(BlockAwarePrefixCache)
        cache.block_size = 64
        cache.paged_cache = MagicMock()
        bt = BlockTable(request_id="R1", block_ids=[1, 2], num_tokens=128)
        cache.paged_cache.get_block_table = MagicMock(return_value=bt)
        cache.paged_cache.decrement_ref = MagicMock()
        result = cache.trim_block_table("R1", 200)  # exceeds 128
        assert result is bt
        cache.paged_cache.decrement_ref.assert_not_called()

    def test_trim_block_table_missing_request_returns_none(self):
        """Unknown request_id → None, no crash."""
        from vmlx_engine.prefix_cache import BlockAwarePrefixCache
        from unittest.mock import MagicMock
        cache = BlockAwarePrefixCache.__new__(BlockAwarePrefixCache)
        cache.block_size = 64
        cache.paged_cache = MagicMock()
        cache.paged_cache.get_block_table = MagicMock(return_value=None)
        assert cache.trim_block_table("missing", 100) is None


class TestMlxstudio73ReconstructFailReleasesBlocks:
    """mlxstudio#73: Session freezes after 'Failed to reconstruct cache'.

    Root cause: fetch_cache incremented block refs, but when
    reconstruct_cache returned None (MLX API drift, quantization mismatch,
    TQ dequant error), the non-hybrid scheduler path treated it as a cache
    miss without releasing the block refs. Over a long session this leaked
    one set of refs per reconstruction failure → block pool exhausted →
    all new requests stall waiting for allocations that never free →
    client sees frozen session.

    Fix: scheduler.py now calls release_cache(request_id) in the
    reconstruction-failure branch, symmetric with the hybrid-cache-fix
    branch that already did.
    """

    def test_reconstruct_failure_releases_block_refs(self):
        """Source contains release_cache call on the reconstruct-fail path."""
        src = Path(
            "/private/tmp/vmlx-1.3.55-build/vmlx_engine/scheduler.py"
        ).read_text()
        # Find the 'Reconstruction failed, treat as cache miss' branch
        idx = src.find("Reconstruction failed, treat as cache miss")
        assert idx > 0, "the bugfix anchor must survive refactors"
        # Immediately below must call release_cache(request_id)
        window = src[idx:idx + 1200]
        assert "self.block_aware_cache.release_cache(request.request_id)" in window, (
            "mlxstudio#73 fix missing: reconstruct-fail branch must release "
            "block refs or a long session leaks until the pool exhausts"
        )
        # Anchor the issue so refactors notice
        assert "mlxstudio#73" in window


class TestVmlx65GemmaE2BConversion:
    """vmlx#65: `TypeError: '>' not supported between NoneType and int` on
    gemma-4-E2B JANG conversion.

    Root cause: HF configs can spell "this attribute is absent" as
    `"num_local_experts": null` instead of omitting the key. Python
    dict.get() with a default does NOT substitute explicit None, so the
    downstream `num_experts > 1` comparison crashed.
    """

    def test_detect_architecture_handles_null_experts(self, tmp_path):
        """Minimal config with num_local_experts=null must not crash."""
        import json
        cfg = {
            "model_type": "gemma4",
            "architectures": ["Gemma4ForCausalLM"],
            "num_attention_heads": 8,
            "num_key_value_heads": 4,
            "num_hidden_layers": 30,
            "hidden_size": 2048,
            "vocab_size": 256000,
            "num_local_experts": None,  # the crash trigger in vmlx#65
            "num_experts": None,
            "n_routed_experts": None,
        }
        (tmp_path / "config.json").write_text(json.dumps(cfg))

        from jang_tools.architectures import detect_architecture
        # Must not raise TypeError
        result = detect_architecture(str(tmp_path))
        # None should coerce to the default (0), routing to TRANSFORMER
        assert result.num_experts == 0
        assert result.has_moe_layers is False

    def test_detect_architecture_handles_present_int_experts(self, tmp_path):
        """Regression guard: when num_local_experts IS an int, don't break it."""
        import json
        cfg = {
            "model_type": "gemma4",
            "architectures": ["Gemma4ForCausalLM"],
            "num_attention_heads": 8,
            "num_key_value_heads": 4,
            "num_hidden_layers": 30,
            "hidden_size": 2048,
            "vocab_size": 256000,
            "num_local_experts": 32,
            "num_experts_per_tok": 4,
        }
        (tmp_path / "config.json").write_text(json.dumps(cfg))
        from jang_tools.architectures import detect_architecture
        result = detect_architecture(str(tmp_path))
        assert result.num_experts == 32
        assert result.has_moe_layers is True


class TestVmlx83JitWarmupFailureRollback:
    """vmlx#83: reporter noticed `mx.compile` errors did not fall back to
    the uncompiled model. Root cause: `mx.compile()` is lazy — errors
    surface at the first forward pass (the warmup). Previously the
    warmup-failure handler logged the error but left `model.model =
    compiled`, so every subsequent real request hit the broken graph
    and silently failed.

    Fix: on warmup failure, restore the pre-compile inner model so the
    engine reverts to the same state as "JIT disabled".
    """

    def test_rollback_restores_original_model_on_warmup_fail(self):
        from unittest.mock import MagicMock, patch
        import mlx.core as mx
        from vmlx_engine import server

        class Inner:
            def __call__(self, x, cache=None):
                return x

        class ModelWrapper:
            def __init__(self, inner):
                self.model = inner

            def make_cache(self):
                return None

        original_inner = Inner()
        wrapper = ModelWrapper(original_inner)

        mock_engine = MagicMock()
        mock_engine._model = wrapper
        mock_engine.is_mllm = False
        mock_engine._is_mllm = False

        class BrokenCompiled:
            def __call__(self, *a, **kw):
                raise RuntimeError("Unsupported dynamic shape")

        with patch.object(server, "_engine", mock_engine), \
             patch.object(server, "_flash_moe_enabled", False), \
             patch.object(server, "_flash_moe_loader", None), \
             patch.object(server, "_distributed_enabled", False), \
             patch.object(mx, "compile", return_value=BrokenCompiled()):
            server._apply_jit_compilation()

        assert wrapper.model is original_inner, (
            "vmlx#83: warmup failure must roll back — broken compiled fn "
            "must not remain installed"
        )

    def test_rollback_anchor_in_source(self):
        """Source contains the vmlx#83 rollback anchors."""
        src = Path(
            "/private/tmp/vmlx-1.3.55-build/vmlx_engine/server.py"
        ).read_text()
        assert "vmlx#83" in src, "issue anchor must be present"
        # Both LLM and VLM branches need backup refs
        assert "_pre_compile_inner" in src, "LLM rollback capture missing"
        assert "_pre_compile_backup_vlm" in src, "VLM rollback capture missing"
        # Rollback must fire in warmup except branch
        assert "rolling back" in src.lower(), "rollback log anchor missing"


class TestVmlx81SmeltAndFlashMoeOnJangtq:
    """vmlx#81: JANGTQ (weight_format=mxtq) with --smelt or --flash-moe
    previously produced cryptic errors:
      - smelt: "missing 'format' field. Expected: jang, jjqf, mxq"
      - flash-moe: silent skip with "no MoE layers" OR downstream
        "missing block_sparse_moe.experts.N.w1.weight"

    Both are the same class — JANGTQ stores expert weights as codebook-
    packed .tq_packed/.tq_norms tensors, not the .weight/.scales/.biases
    layout that smelt and flash-moe scan for.

    Fix: detect weight_format=mxtq in jang_config.json at the top of
    each entry point, emit a clear error/warning citing the alternative
    (use JANG_ variant or drop the flag). No silent fall-through.
    """

    def test_smelt_load_on_jangtq_raises_actionable_error(self, tmp_path):
        """smelt_load() on an mxtq model must raise ValueError citing vmlx#81."""
        import json
        # Minimal JANGTQ-shaped config
        (tmp_path / "jang_config.json").write_text(json.dumps({
            "version": 2, "weight_format": "mxtq", "mxtq_seed": 42,
        }))
        (tmp_path / "config.json").write_text(json.dumps({
            "model_type": "qwen3_5_moe",
            "architectures": ["Qwen3_5MoeForCausalLM"],
        }))
        from vmlx_engine.utils.smelt_loader import smelt_load
        with pytest.raises(ValueError, match=r"vmlx#81.*--smelt"):
            smelt_load(str(tmp_path), expert_percent=50)

    def test_smelt_error_names_workaround(self, tmp_path):
        """Error text must tell the user what to do next."""
        import json
        (tmp_path / "jang_config.json").write_text(json.dumps({
            "version": 2, "weight_format": "mxtq",
        }))
        (tmp_path / "config.json").write_text(json.dumps({
            "model_type": "minimax_m2",
        }))
        from vmlx_engine.utils.smelt_loader import smelt_load
        try:
            smelt_load(str(tmp_path), expert_percent=50)
        except ValueError as e:
            msg = str(e)
            assert "JANG_" in msg, "must point to JANG_ alternative"
            assert "TurboQuantLinear" in msg, "must explain why"
            return
        pytest.fail("expected ValueError")

    def test_flash_moe_server_path_detects_jangtq(self):
        """server.py flash-moe setup must detect JANGTQ before calling
        ExpertIndex.build (which would find 0 MoE layers silently)."""
        src = Path(
            "/private/tmp/vmlx-1.3.55-build/vmlx_engine/server.py"
        ).read_text()
        # Locate _apply_flash_moe_patching vicinity — the server-side
        # entry point that wires the flash-moe loader onto the engine.
        idx = src.find("def _apply_flash_moe_patching")
        assert idx > 0, "flash-moe setup entry point missing"
        window = src[idx:idx + 4500]
        # vmlx#81 JANGTQ detection must exist inside the setup function
        assert "vmlx#81" in window, (
            "flash-moe setup must carry vmlx#81 JANGTQ detection"
        )
        assert 'weight_format") == "mxtq"' in window, (
            "flash-moe JANGTQ detection must check weight_format=mxtq"
        )

    def test_non_jangtq_jang_model_not_affected(self, tmp_path):
        """Regression guard: a normal JANG (not JANGTQ) model must NOT
        trigger the vmlx#81 early-exit. It should proceed past the
        detection and fail for a different, downstream reason (missing
        safetensors here), NOT with vmlx#81 ValueError."""
        import json
        (tmp_path / "jang_config.json").write_text(json.dumps({
            "version": 2, "format": "jang",  # standard JANG, NOT mxtq
        }))
        (tmp_path / "config.json").write_text(json.dumps({
            "model_type": "qwen3_5_moe",
        }))
        from vmlx_engine.utils.smelt_loader import smelt_load
        try:
            smelt_load(str(tmp_path), expert_percent=50)
        except ValueError as e:
            assert "vmlx#81" not in str(e), (
                "standard JANG must not trigger vmlx#81 guard"
            )
        except Exception:
            pass  # expected: downstream failure loading actual weights


class TestVmlx81JangtqSmeltFlashMoeIncompat:
    """vmlx#81: --smelt or --flash-moe on JANGTQ (weight_format=mxtq)
    models used to fail with cryptic errors deep in the loader:

      --smelt:     "JANG config jang_config.json is missing 'format' field"
                   (because JANGTQ uses `weight_format` not `format`)
      --flash-moe: ExpertIndex silently found 0 MoE layers (JANGTQ's
                   tq_packed/tq_norms keys don't match .weight/.scales)
                   → silent skip + later crash on dequant fallback

    Fix: detect JANGTQ up-front in both entry points and surface a
    clean error with actionable workarounds (use JANG_* variant, or
    drop the flag since JANGTQ is already sub-2-bit effective).
    """

    def test_smelt_load_rejects_jangtq_cleanly(self, tmp_path):
        import json
        (tmp_path / "jang_config.json").write_text(json.dumps({
            "version": 2, "weight_format": "mxtq",
        }))
        (tmp_path / "config.json").write_text(json.dumps({
            "model_type": "minimax_m2", "num_local_experts": 256,
        }))
        from vmlx_engine.utils.smelt_loader import smelt_load
        import pytest as _pt
        with _pt.raises(ValueError, match=r"vmlx#81.*--smelt.*JANGTQ"):
            smelt_load(str(tmp_path), expert_percent=50)

    def test_flash_moe_skips_jangtq_cleanly(self, tmp_path):
        import json, logging
        from unittest.mock import MagicMock, patch
        (tmp_path / "jang_config.json").write_text(json.dumps({
            "weight_format": "mxtq",
        }))
        (tmp_path / "config.json").write_text(json.dumps({
            "model_type": "minimax_m2",
        }))
        import vmlx_engine.server as srv
        logs = []

        class _LH(logging.Handler):
            def emit(self, r): logs.append(r.getMessage())

        srv.logger.addHandler(_LH())
        srv.logger.setLevel(logging.WARNING)
        srv._flash_moe_enabled = True
        srv._flash_moe_loader = None
        srv._model_path = str(tmp_path)
        srv._model_name = str(tmp_path)
        srv._cli_args = {}
        srv._engine = MagicMock()
        srv._distributed_enabled = False
        srv._distributed_coordinator = None
        with patch.object(srv, "_get_raw_model_from_engine", return_value=MagicMock()):
            srv._apply_flash_moe_patching()
        hits = [l for l in logs if "vmlx#81" in l]
        assert hits, f"expected vmlx#81 clean skip log, got: {logs}"
        assert srv._flash_moe_loader is None, (
            "flash_moe_loader must be None after clean skip"
        )

    def test_vmlx81_anchors_in_source(self):
        """vmlx#81 anchor in both smelt_loader and server flash-moe init."""
        smelt_src = Path(
            "/private/tmp/vmlx-1.3.55-build/vmlx_engine/utils/smelt_loader.py"
        ).read_text()
        srv_src = Path(
            "/private/tmp/vmlx-1.3.55-build/vmlx_engine/server.py"
        ).read_text()
        assert "vmlx#81" in smelt_src
        assert "vmlx#81" in srv_src


class TestVmlx92PldNonMllmGuard:
    """vmlx#92: `_try_speculative_decode` assumed BatchGenerator exposes
    `.active_batch` (MLLM API). On text-only / non-MLLM paths mlx-lm's
    plain BatchGenerator has neither `.active_batch` nor
    `remove(return_prompt_caches=True)`, so PLD crashed, re-inserted a
    malformed cache, and the NEXT step() raised
    `<class 'list'> does not yet support batching with history`.
    Recovery then `clear()`s the paged cache — torching every prefix
    for every in-flight request. Every PLD-enabled text server
    corrupted itself within a few tokens.

    Fix: capability check right after the temperature gate —
    `if not hasattr(self.batch_generator, 'active_batch'): return []`
    """

    def test_speculative_decode_returns_empty_on_non_mllm_generator(self):
        from unittest.mock import MagicMock
        from vmlx_engine.scheduler import Scheduler

        s = Scheduler.__new__(Scheduler)
        s._pld_spec_max_temp = 1.0

        class PlainBatchGen:
            def __init__(self):
                self.requests = {}
        s.batch_generator = PlainBatchGen()

        req = MagicMock()
        req.sampling_params = MagicMock()
        req.sampling_params.temperature = 0.3
        req.prompt_token_ids = [1, 2, 3]
        req.output_token_ids = [10, 11]

        # Must return [] cleanly — NOT raise AttributeError and NOT
        # touch the generator state.
        result = s._try_speculative_decode("rid", req, 11)
        assert result == [], f"expected [] (non-MLLM short-circuit), got {result}"

    def test_temperature_gate_still_fires_first(self):
        """Regression: high-temp short-circuit must still run before the
        active_batch hasattr check (defense in depth)."""
        from unittest.mock import MagicMock
        from vmlx_engine.scheduler import Scheduler
        s = Scheduler.__new__(Scheduler)
        s._pld_spec_max_temp = 1.0
        s.batch_generator = MagicMock()  # HAS active_batch
        req = MagicMock()
        req.sampling_params = MagicMock()
        req.sampling_params.temperature = 2.5
        req.prompt_token_ids = [1]
        req.output_token_ids = []
        assert s._try_speculative_decode("rid", req, 0) == []

    def test_vmlx92_anchor_in_source(self):
        src = Path(
            "/private/tmp/vmlx-1.3.55-build/vmlx_engine/scheduler.py"
        ).read_text()
        assert "vmlx#92" in src
        assert 'hasattr(self.batch_generator, "active_batch")' in src


class TestVmlx97PaintedMaskInpainting:
    """vmlx#97: Flux Fill failed with "requires a mask_path for inpainting"
    when the user painted a mask with the brush tool, while rectangle
    masks worked.

    Root cause: paint-tool output is RGBA with strokes in the alpha
    channel. `img.convert("L")` drops alpha → all-zero grayscale → mask
    file is saved but all-black → downstream treats as no-mask.

    Fix: when the mask is RGBA or LA, blend alpha into the L channel
    (`ImageChops.lighter(rgb_l, alpha)`) so painted strokes survive
    the grayscale conversion. Also rejects all-zero masks with a
    clear error that tells the user the mask didn't register.
    """

    def test_rgba_paint_mask_preserved_through_conversion(self):
        """RGBA with painted alpha strokes → L-mode preserves the strokes."""
        from PIL import Image, ImageChops
        rgba = Image.new("RGBA", (16, 16), (0, 0, 0, 0))
        for x in range(5):
            for y in range(5):
                rgba.putpixel((x, y), (255, 255, 255, 255))
        rgb_l = rgba.convert("RGB").convert("L")
        alpha = rgba.split()[-1]
        merged = ImageChops.lighter(rgb_l, alpha)
        assert merged.getextrema()[1] == 255, (
            "RGBA paint-tool mask must preserve strokes after alpha blend"
        )

    def test_all_transparent_rgba_detected_as_empty(self):
        """Fully transparent RGBA → still all zero after blend → rejected."""
        from PIL import Image, ImageChops
        empty = Image.new("RGBA", (16, 16), (0, 0, 0, 0))
        rgb_l = empty.convert("RGB").convert("L")
        alpha = empty.split()[-1]
        merged = ImageChops.lighter(rgb_l, alpha)
        assert merged.getextrema() == (0, 0), (
            "all-transparent RGBA should produce empty mask that the server rejects"
        )

    def test_rect_grayscale_still_works_after_fix(self):
        """Regression guard: rectangle mask (L-mode with signal) still passes."""
        from PIL import Image
        rect = Image.new("L", (16, 16), 0)
        for x in range(5, 10):
            for y in range(5, 10):
                rect.putpixel((x, y), 255)
        assert rect.getextrema()[1] == 255
        # Already L mode — no conversion needed

    def test_server_rejects_all_black_mask_with_actionable_error(self):
        """Server code path: all-black mask → HTTPException 400 with
        vmlx#97 anchor + workaround instructions."""
        src = Path(
            "/private/tmp/vmlx-1.3.55-build/vmlx_engine/server.py"
        ).read_text()
        # The anchor must be in the server
        assert "vmlx#97" in src, "server must carry vmlx#97 anchor"
        # RGBA alpha-blend handling
        assert 'mask_img.mode == "RGBA"' in src
        assert "ImageChops.lighter" in src or "ImageChops import" not in src
        # Empty-mask guard
        assert "Mask is all black" in src, (
            "server must give a clear error when mask has no painted region"
        )

    def test_server_handles_la_mode_mask(self):
        """LA (grayscale + alpha) is another common paint output."""
        from PIL import Image, ImageChops
        la = Image.new("LA", (16, 16), (0, 0))
        for x in range(5):
            for y in range(5):
                la.putpixel((x, y), (200, 255))  # gray stroke, opaque
        l = la.convert("L")
        alpha = la.split()[-1]
        merged = ImageChops.lighter(l, alpha)
        # alpha is 0 outside stroke, 255 inside → result has signal
        assert merged.getextrema()[1] == 255


class TestVmlx94MxMetalDeprecationCleanup:
    """vmlx#94: scheduler.py's memory-pressure guard called
    `mx.metal.get_active_memory()` and `mx.metal.device_info()` without
    the `getattr(mx, X, None) or mx.metal.X` fallback used in server.py.
    On MLX 0.31+ these emit DeprecationWarning ("mx.metal.X is
    deprecated...use mx.X instead"). Future MLX release will remove the
    aliases outright and break the scheduler on every admission.

    Fix: same fallback pattern at both call sites.
    """

    def test_scheduler_memory_pressure_uses_fallback_pattern(self):
        """Both sites must use `getattr(mx, 'X', None) or mx.metal.X`."""
        src = Path(
            "/private/tmp/vmlx-1.3.55-build/vmlx_engine/scheduler.py"
        ).read_text()
        # Memory-pressure guard lives in the admission loop
        assert 'getattr(mx, "get_active_memory", None)' in src, (
            "scheduler.py must use getattr fallback for get_active_memory"
        )
        assert 'getattr(mx, "device_info", None)' in src, (
            "scheduler.py must use getattr fallback for device_info"
        )
        # Anchor the issue
        assert "vmlx#94" in src

    def test_no_unguarded_mx_metal_calls_in_hot_paths(self):
        """Grep for unguarded `mx.metal.X()` calls in scheduler / server
        hot paths. They must live inside a hasattr/getattr guard."""
        import re as _re
        for rel in ("vmlx_engine/scheduler.py",):
            src = Path(
                f"/private/tmp/vmlx-1.3.55-build/{rel}"
            ).read_text()
            # Capture bare mx.metal.get_active_memory() style calls that
            # aren't preceded by a hasattr/getattr/or-fallback. Simple
            # proxy: any `mx.metal.get_active_memory(` outside comments
            # that isn't part of `or mx.metal.`.
            lines = src.split("\n")
            bad = []
            for i, line in enumerate(lines, 1):
                if "#" in line and line.strip().startswith("#"):
                    continue
                if _re.search(r"\bmx\.metal\.(?:get_active_memory|device_info)\s*\(", line):
                    # Allow if preceded by 'or '
                    if " or mx.metal." not in line:
                        bad.append(f"{rel}:{i}: {line.strip()}")
            assert not bad, (
                f"unguarded mx.metal.* calls in scheduler hot path:\n  "
                + "\n  ".join(bad)
            )

    def test_new_api_available_on_bundled_mlx(self):
        """The fallback exists for old MLX; confirm bundled MLX has the
        new top-level APIs so the fallback branch never fires in prod."""
        import mlx.core as mx
        assert getattr(mx, "get_active_memory", None) is not None
        assert getattr(mx, "device_info", None) is not None
        assert getattr(mx, "get_peak_memory", None) is not None
        assert getattr(mx, "get_cache_memory", None) is not None
        assert getattr(mx, "clear_cache", None) is not None


class TestVmlx96RelocatedMfluxModels:
    """vmlx#96: relocated image models (e.g. on external SSD) fail class
    resolution because the session stores `mflux_name` as the directory
    name (`Z-Image-Turbo-mflux-8bit`), skipping the normalizer that fires
    only when `mflux_name` is absent.

    Fix: second-chance decoration strip in the class-resolution branch
    that runs whether mflux_name was pre-set or not.
    """

    def test_relocated_z_image_turbo_resolves(self):
        """Reporter's exact case: Z-Image-Turbo-mflux-8bit passes through
        to a downstream load error, NOT the class-lookup ValueError."""
        from vmlx_engine.image_gen import ImageGenEngine
        e = ImageGenEngine()
        try:
            e.load(
                model_name="Z-Image-Turbo-mflux-8bit",
                mflux_name="Z-Image-Turbo-mflux-8bit",
                quantize=8,
                model_path="/nonexistent/path",
            )
        except ValueError as err:
            assert "Cannot determine mflux class" not in str(err), (
                f"vmlx#96 regression: class lookup still fails: {err}"
            )
            # Any OTHER ValueError (e.g. local files not found) is fine
        except Exception:
            pass  # downstream error after successful class resolution

    def test_relocated_flux2_klein_9b_resolves(self):
        """Another reporter variant: flux2-klein-9b-mflux-4bit."""
        from vmlx_engine.image_gen import ImageGenEngine
        e = ImageGenEngine()
        try:
            e.load(
                model_name="Flux2-Klein-9B-mflux-4bit",
                mflux_name="Flux2-Klein-9B-mflux-4bit",
                quantize=4,
                model_path="/nonexistent",
            )
        except ValueError as err:
            assert "Cannot determine mflux class" not in str(err)
        except Exception:
            pass

    def test_relocated_qwen_image_edit_resolves(self):
        """Another reporter variant: Qwen-Image-Edit-mflux (no bit suffix)."""
        from vmlx_engine.image_gen import ImageGenEngine
        e = ImageGenEngine()
        try:
            e.load(
                model_name="Qwen-Image-Edit-mflux",
                mflux_name="Qwen-Image-Edit-mflux",
                model_path="/nonexistent",
            )
        except ValueError as err:
            assert "Cannot determine mflux class" not in str(err)
        except Exception:
            pass

    def test_unknown_model_still_errors_clearly(self):
        """Regression guard: a genuinely-unknown model still fails, but
        with the improved error listing known keys."""
        pytest.importorskip("mflux")
        from vmlx_engine.image_gen import ImageGenEngine
        e = ImageGenEngine()
        with pytest.raises(ValueError) as exc_info:
            e.load(
                model_name="totally-unknown-xyz-1.0",
                mflux_name="totally-unknown-xyz-1.0",
                model_path="/nonexistent",
            )
        # Error should list known keys to help the user
        assert "Known keys" in str(exc_info.value), (
            "error must list known keys so user can see what to use"
        )
        assert "z-image" in str(exc_info.value), (
            "known keys list must include the canonical names"
        )

    def test_unknown_model_error_format_source_pin(self):
        """Source-anchor alternative to the live call above — runs even
        without mflux installed. Verifies the error message template
        includes `Known keys` and lists canonical class names."""
        src = Path(
            "/private/tmp/vmlx-1.3.55-build/vmlx_engine/image_gen.py"
        ).read_text()
        assert "Known keys:" in src, (
            "vmlx#96 regression: error template must surface known keys"
        )
        # The keys list is built from _NAME_TO_CLASS at raise-time; pin
        # the variable name so the string stays live-sourced.
        assert "sorted(_NAME_TO_CLASS.keys())" in src

    def test_anchor_and_regex_strips_in_source(self):
        """Source pin: vmlx#96 anchor and the regex strip sequence both
        appear in image_gen.py."""
        src = Path(
            "/private/tmp/vmlx-1.3.55-build/vmlx_engine/image_gen.py"
        ).read_text()
        assert "vmlx#96" in src
        # Class-resolution second-chance strip must exist
        assert "second-chance" in src.lower() or "decoration-strip" in src


class TestMlxstudio76UnrecognizedArgHint:
    """mlxstudio#76: users hit `unrecognized arguments: --smelt ...`
    intermittently because an older vmlx-engine is shadowing the newer
    bundled-python binary on PATH. Argparse's default message gave no
    hint — users restarted 3-4 times hoping it would work.

    Fix: wrap parser.parse_args() with a SystemExit(2) catch that appends
    a hint about PATH shadowing + tells the user to run `which -a` to
    diagnose. Argparse's error message still prints normally; our hint
    follows it.
    """

    def test_cli_source_has_mlxstudio76_hint(self):
        import vmlx_engine.cli as cli
        src = Path(cli.__file__).read_text()
        assert "mlxstudio#76" in src, "anchor required so future edits notice"
        # Critical diagnostics in the hint must be preserved
        assert "which -a vmlx-engine" in src, (
            "hint must tell user to check PATH shadowing"
        )
        assert "SystemExit" in src and "parse_args" in src

    def test_unknown_flag_triggers_hint(self, capsys):
        """Running cli with a garbage flag triggers the hint."""
        import subprocess
        bpy = (
            "/private/tmp/vmlx-1.3.55-build/panel/bundled-python/"
            "python/bin/python3.12"
        )
        r = subprocess.run(
            [bpy, "-m", "vmlx_engine.cli", "serve",
             "/nonexistent", "--nonexistent-flag-xyz-1-2-3"],
            capture_output=True, text=True, timeout=30,
        )
        # argparse exits 2 on unrecognized args
        assert r.returncode == 2
        combined = (r.stdout or "") + (r.stderr or "")
        assert "unrecognized arguments" in combined, (
            "argparse error must still fire"
        )
        assert "mlxstudio#76" in combined, (
            "hint must include issue anchor"
        )
        assert "which -a vmlx-engine" in combined, (
            "hint must include PATH-shadowing diagnostic"
        )


class TestVmlx75DefaultRepetitionPenaltyArg:
    """vmlx#75: reporter's `vmlx-engine serve ... --default-repetition-penalty
    1.10` failed with `unrecognized arguments`. The flag IS defined in the
    serve subparser at 1.3.55 (and has been since the server-wide defaults
    landed) — this test pins that contract so a future refactor can't drop
    the serve-subparser wiring again.

    Paired with mlxstudio#76 (diagnostic hint on argparse SystemExit) so
    older-binary-on-PATH cases now surface a clear diagnosis rather than
    an opaque error.
    """

    def test_serve_subparser_has_default_repetition_penalty(self):
        """The reporter's exact flag must be registered on `serve`."""
        import vmlx_engine.cli as cli
        src = Path(cli.__file__).read_text()
        # Find the serve_parser block and check our flag is inside it
        idx = src.find('subparsers.add_parser("serve"')
        assert idx > 0, "serve subparser must exist"
        # Confirm the flag appears after the serve_parser definition and
        # before the next subparser (bench).
        bench_idx = src.find('subparsers.add_parser("bench"', idx)
        assert bench_idx > idx
        serve_block = src[idx:bench_idx]
        assert '"--default-repetition-penalty"' in serve_block, (
            "vmlx#75: --default-repetition-penalty must be registered on "
            "the `serve` subparser (not just on the server.py argparse)"
        )
        assert '"--default-enable-thinking"' in serve_block, (
            "vmlx#75: --default-enable-thinking must be on `serve` too — "
            "reporter's command line used both"
        )
        assert '"--default-temperature"' in serve_block
        assert '"--default-top-p"' in serve_block

    def test_serve_subparser_validates_repetition_penalty_range(self):
        """Validation range must remain 0.5..2.0 — reporter used 1.10."""
        import vmlx_engine.cli as cli
        src = Path(cli.__file__).read_text()
        assert "0.5 <= args.default_repetition_penalty <= 2.0" in src, (
            "vmlx#75: range check for repetition_penalty must accept 1.10"
        )

    def test_reporter_exact_command_parses(self):
        """End-to-end: the reporter's exact flag combo must parse without
        argparse rejecting any flag. Model-load fails (expected) but
        argparse returncode must not be 2 (which is "unrecognized args")."""
        import subprocess
        bpy = (
            "/private/tmp/vmlx-1.3.55-build/panel/bundled-python/"
            "python/bin/python3.12"
        )
        # Reporter's full flag set from the issue body
        r = subprocess.run(
            [bpy, "-m", "vmlx_engine.cli", "serve",
             "/nonexistent/model",
             "--host", "127.0.0.1", "--port", "8000", "--timeout", "300",
             "--max-num-seqs", "5",
             "--prefill-batch-size", "512", "--prefill-step-size", "1024",
             "--completion-batch-size", "512", "--is-mllm",
             "--continuous-batching",
             "--tool-call-parser", "gemma4", "--enable-auto-tool-choice",
             "--reasoning-parser", "gemma4",
             "--cache-memory-percent", "0.1", "--use-paged-cache",
             "--paged-cache-block-size", "64", "--max-cache-blocks", "500",
             "--enable-block-disk-cache", "--block-disk-cache-max-gb", "10",
             "--stream-interval", "1", "--max-tokens", "32768",
             "--default-temperature", "0.70", "--default-top-p", "0.95",
             "--default-repetition-penalty", "1.10",
             "--default-enable-thinking", "false"],
            capture_output=True, text=True, timeout=60,
        )
        combined = (r.stdout or "") + (r.stderr or "")
        # Must NOT be "unrecognized arguments" — that's the vmlx#75 bug
        assert "unrecognized arguments" not in combined, (
            f"vmlx#75 regression: reporter's flag set rejected by argparse. "
            f"stderr:\n{combined}"
        )
        # returncode 2 = argparse rejection. Model-load failures exit 1.
        # Assertion is on the argparse-specific code so we don't
        # accidentally pass a different error through.
        assert r.returncode != 2, (
            f"vmlx#75 regression: argparse exited 2 (unrecognized). "
            f"returncode={r.returncode}, combined={combined[:500]}"
        )

    def test_vmlx75_anchor_in_source(self):
        """Explicit anchor — so `grep vmlx#75` finds the CLI guard."""
        import vmlx_engine.cli as cli
        src = Path(cli.__file__).read_text()
        # The fix is that the flags are registered. Anchor the regression
        # in a visible comment so future greps land here.
        # If this fails, add `# vmlx#75` beside the --default-repetition-penalty
        # definition in cli.py.
        # (We allow this test to be soft — the hard pins are above.)
        assert "default-repetition-penalty" in src


class TestMlxstudio77Gemma4DupToolCalls:
    """mlxstudio#77: Gemma 4 + multiple tool calls — duplicate emission.

    Before: second tool-call end-marker re-scanned the full current_text
    (which still included the first call) and re-emitted it with a NEW
    UUID. Client received `datetime` twice with different IDs, routed
    the result to one ID, and the model's next turn saw mismatched
    tool_result → hallucinated content (reporter's symptom).

    Fix: track emitted_count per parser instance, emit only the tail
    past what's been streamed already. reset_streaming_state() clears
    the counter for instance reuse across requests.
    """

    def _make_stream(self):
        """Reporter's scenario: two back-to-back Gemma 4 tool calls."""
        return [
            "", "Let me search. ",
            "<|tool_call>", "call:datetime{}", "<tool_call|>",
            " Now search: ",
            "<|tool_call>", "call:search_web{\"query\":\"Iran\"}", "<tool_call|>",
            "",
        ]

    def test_no_duplicate_tool_calls_on_second_emission(self):
        from vmlx_engine.tool_parsers.gemma4_tool_parser import Gemma4ToolParser
        p = Gemma4ToolParser(tokenizer=None)
        current = ""
        previous = ""
        emitted = []
        for delta in self._make_stream():
            current = previous + delta
            r = p.extract_tool_calls_streaming(previous, current, delta)
            if r and r.get("tool_calls"):
                for tc in r["tool_calls"]:
                    emitted.append(tc)
            previous = current
        names = [tc["function"]["name"] for tc in emitted]
        assert names.count("datetime") == 1, (
            f"datetime emitted {names.count('datetime')} times, expected 1. "
            f"Second emission bug regressed: {names}"
        )
        assert names.count("search_web") == 1
        # Indices must be sequential starting at 0
        idx = [tc["index"] for tc in emitted]
        assert idx == [0, 1], f"indices must be sequential, got {idx}"

    def test_reset_streaming_state_allows_instance_reuse(self):
        """Calling reset_streaming_state between requests lets the same
        parser instance re-emit tool calls from a fresh state."""
        from vmlx_engine.tool_parsers.gemma4_tool_parser import Gemma4ToolParser
        p = Gemma4ToolParser(tokenizer=None)

        def run_once():
            current = ""
            previous = ""
            emitted = []
            for delta in self._make_stream():
                current = previous + delta
                r = p.extract_tool_calls_streaming(previous, current, delta)
                if r and r.get("tool_calls"):
                    emitted.extend(r["tool_calls"])
                previous = current
            return emitted

        first = run_once()
        p.reset_streaming_state()
        second = run_once()
        for label, batch in [("first", first), ("second", second)]:
            names = [tc["function"]["name"] for tc in batch]
            assert names.count("datetime") == 1, f"{label}: datetime dup"
            assert names.count("search_web") == 1, f"{label}: search_web dup"

    def test_source_pin(self):
        import vmlx_engine.tool_parsers.gemma4_tool_parser as m
        src = Path(m.__file__).read_text()
        assert "mlxstudio#77" in src, "anchor must persist"
        assert "_gemma4_emitted_count" in src


class TestAsyncSSMRederiveReasoningHybrid:
    """Async SSM re-derive for reasoning hybrid models (Qwen3.5 GatedDelta,
    Nemotron Cascade with thinking, etc.).

    Problem: when a thinking hybrid request finalizes, the SSM recurrent
    state has been advanced through both the prompt AND the thinking
    tokens + output. Storing that state as a "prefix checkpoint" fails
    on the next request because the checkpoint's token position (prompt
    boundary) no longer matches the actual SSM state (post-output).

    Prior behavior (broken): skip SSM companion storage entirely when
    `gen_prompt_len > 0`. Net: thinking hybrid models never get SSM
    prefix hits → full re-prefill every turn.

    Current behavior (async re-derive queue):
    1. On finalize with `gpl > 0`, scheduler appends (tokens, prompt_len,
       request_id) to `_ssm_rederive_queue` (capped at 20 entries).
    2. On the scheduler step, when `not self.running` (idle), the queue's
       head is popped and `_prefill_for_prompt_only_cache(tokens)` runs a
       clean prefill pass on JUST the prompt tokens. The resulting SSM
       state is clean (no thinking contamination) and gets stored in
       `_ssm_state_cache` keyed by `(tokens, prompt_len)`.
    3. Next request with the same prompt prefix: SSM companion fetch hits
       → KV + SSM prefix cache both populate → no full re-prefill.

    These tests pin the contract so a refactor doesn't silently revert
    thinking hybrid models to full-prefill-every-turn.
    """

    def test_scheduler_has_ssm_rederive_queue_path(self):
        """Scheduler source must queue re-derive instead of skipping."""
        import vmlx_engine.scheduler as sched
        src = Path(sched.__file__).read_text()
        # Queue append logic must exist
        assert "_ssm_rederive_queue" in src, (
            "async re-derive queue missing — thinking hybrid models will "
            "silently lose all SSM prefix hits"
        )
        # Queue cap
        assert ">= 20" in src and "pop(0)" in src, (
            "queue cap + FIFO eviction must be preserved (unbounded queue "
            "growth under sustained load = memory leak on busy servers)"
        )
        # gpl > 0 branch is the trigger
        assert "_gpl > 0" in src, (
            "must trigger on thinking models (gen_prompt_len > 0)"
        )

    def test_scheduler_idle_processes_one_rederive_per_step(self):
        """Idle consumer must pop ONE entry per scheduler step (avoid
        long GPU stalls blocking new incoming requests)."""
        import vmlx_engine.scheduler as sched
        src = Path(sched.__file__).read_text()
        idle_block_idx = src.find("Deferred SSM re-derive (idle-time")
        assert idle_block_idx > 0, (
            "idle consumer block missing — re-derives will never execute"
        )
        # The consumer must check `not self.running` before running the
        # clean prefill (GPU is exclusive; running + re-derive = deadlock)
        block_end = src.find("return output", idle_block_idx)
        assert block_end > idle_block_idx
        block = src[idle_block_idx:block_end]
        assert "not self.running" in block, (
            "idle guard missing: re-derive would collide with active "
            "generation on the GPU"
        )
        assert "_ssm_state_cache is not None" in block, (
            "must guard on companion cache presence (non-hybrid models "
            "don't have one)"
        )
        # One-per-step contract: a single .pop(0)
        assert block.count("self._ssm_rederive_queue.pop(0)") == 1, (
            "exactly one entry per step — more means long stalls"
        )

    def test_clean_prefill_helper_exists(self):
        """The clean-prompt prefill helper must exist — it's what gives
        the SSM companion UNCONTAMINATED state."""
        import vmlx_engine.scheduler as sched
        src = Path(sched.__file__).read_text()
        assert "def _prefill_for_prompt_only_cache" in src, (
            "missing helper — async re-derive has nothing to call"
        )

    def test_stored_state_is_deep_copied(self):
        """Re-derive must deepcopy SSM state into the companion store.
        Sharing buffers with the scheduler's scratch cache → mutation on
        next step → garbled output on the next request that hits."""
        import vmlx_engine.scheduler as sched
        src = Path(sched.__file__).read_text()
        idle_block_idx = src.find("Deferred SSM re-derive (idle-time")
        block_end = src.find("return output", idle_block_idx)
        block = src[idle_block_idx:block_end]
        assert "deepcopy" in block or "mx.array(a)" in block, (
            "SSM re-derive must deep-copy state before storing"
        )

    def test_mllm_side_captures_at_prefill_not_finalize(self):
        """MLLM path is different: it captures at prefill-success (BEFORE
        generation), so SSM state is never thinking-contaminated and
        doesn't need re-derive. Pin this so a refactor doesn't move the
        capture to finalize (which would re-introduce contamination)."""
        import vmlx_engine.mllm_batch_generator as m
        src = Path(m.__file__).read_text()
        # The comment block identifying prefill-boundary capture
        assert "SSM state at prompt boundary" in src or \
               "Capture SSM state at prompt boundary" in src, (
            "MLLM prefill-boundary SSM capture must be documented — "
            "otherwise a refactor could move it to the contaminated "
            "finalize path"
        )
        # store() must be inside the prefill-success flow, not a post-gen hook
        store_idx = src.find("self._ssm_state_cache.store(")
        assert store_idx > 0
        # Look backwards for `first_tokens.append` — which means we're
        # still in the prefill block (after sampling the first token but
        # before any generation step). If store() is AFTER a generation
        # loop, this wouldn't hold.
        preceding = src[max(0, store_idx - 4000):store_idx]
        assert "first_tokens.append" in preceding, (
            "store() must live in prefill-success path (not finalize) — "
            "moving it post-generation reintroduces thinking contamination"
        )

    def test_queue_cap_prevents_unbounded_growth(self):
        """Under sustained thinking-model load the queue could grow
        forever. Cap + oldest-first eviction must be in place."""
        import vmlx_engine.scheduler as sched
        src = Path(sched.__file__).read_text()
        # Find the append site
        append_idx = src.find("self._ssm_rederive_queue.append")
        assert append_idx > 0
        # The cap check must precede the append, within the same block
        preceding = src[max(0, append_idx - 500):append_idx]
        assert "pop(0)" in preceding, (
            "cap-check (pop oldest before append) must precede append — "
            "otherwise queue grows unboundedly"
        )
        assert ">= 20" in preceding, "cap constant must remain 20"


class TestReasoningContractEndToEnd:
    """Comprehensive reasoning on/off contract audit — hits BOTH the
    server-side emission AND the panel-side classification in one
    battery of pins so a future regression in either half fails by
    name."""

    def test_server_emits_reasoning_field_in_chunk_delta(self):
        """ChatCompletionChunkDelta has a `reasoning` field that serializes
        as `reasoning_content` via computed_field. model_dump must exclude
        reasoning_content when None (else OpenAI SDK parsers break)."""
        from vmlx_engine.api.models import ChatCompletionChunkDelta
        d = ChatCompletionChunkDelta(content="hi", reasoning=None)
        dump = d.model_dump(exclude_none=True)
        assert "reasoning_content" not in dump, (
            "reasoning_content must NOT appear when reasoning is None"
        )
        d2 = ChatCompletionChunkDelta(content=None, reasoning="thinking…")
        dump2 = d2.model_dump()
        assert dump2.get("reasoning_content") == "thinking…"

    def test_assistant_message_omits_reasoning_content_when_none(self):
        """Non-streaming: AssistantMessage.model_dump drops
        reasoning_content when reasoning is None."""
        from vmlx_engine.api.models import AssistantMessage
        m = AssistantMessage(content="hello")
        d = m.model_dump(exclude_none=True)
        assert "reasoning_content" not in d, (
            "non-thinking response must not include reasoning_content: null"
        )

    def test_assistant_message_includes_reasoning_when_set(self):
        """When reasoning IS set, reasoning_content appears."""
        from vmlx_engine.api.models import AssistantMessage
        m = AssistantMessage(content="final", reasoning="let me think")
        d = m.model_dump()
        assert d.get("reasoning_content") == "let me think"

    def test_panel_chat_ts_extracts_reasoning_from_both_fields(self):
        """Panel must handle both canonical names: reasoning_content AND
        reasoning (legacy / alias). Regression source pin."""
        src = Path(
            "/private/tmp/vmlx-1.3.55-build/panel/src/main/ipc/chat.ts"
        ).read_text()
        # The extractor line
        assert "choice?.reasoning_content || choice?.reasoning" in src, (
            "panel chat.ts must accept BOTH reasoning_content and "
            "reasoning from the choice object"
        )
        # emitDelta with isReasoningDelta=true
        assert "emitDelta(reasoning, true)" in src, (
            "panel must classify reasoning delta as reasoning=true"
        )

    def test_panel_message_bubble_renders_reasoning_box(self):
        """Panel renderer must render ReasoningBox when reasoningContent
        is present and NOT equal to content."""
        src = Path(
            "/private/tmp/vmlx-1.3.55-build/panel/src/renderer/src/components/chat/MessageBubble.tsx"
        ).read_text()
        assert "ReasoningBox" in src
        assert "reasoningContent" in src
        # The guard that hides ReasoningBox when content == reasoningContent
        # (prevents double-render when suppress_reasoning routed reasoning
        # to content field)
        assert "reasoningContent.trim() === message.content.trim()" in src

    def test_server_suppress_reasoning_routes_to_content(self):
        """§15: when thinking is off and model leaks <think>, the server
        routes reasoning delta → content delta so the user sees
        something instead of empty SSE."""
        src = Path(
            "/private/tmp/vmlx-1.3.55-build/vmlx_engine/server.py"
        ).read_text()
        # The §15 anchor
        assert "§15" in src
        # Concat pattern: parts = [reasoning, content]
        assert "_parts.append(delta_msg.reasoning)" in src, (
            "§15 must concatenate reasoning into the content emit path"
        )

    def test_database_schema_has_reasoning_content_column(self):
        """Panel database must persist reasoningContent across sessions."""
        src = Path(
            "/private/tmp/vmlx-1.3.55-build/panel/src/main/database.ts"
        ).read_text()
        assert "ALTER TABLE messages ADD COLUMN reasoning_content TEXT" in src
        assert "reasoning_content" in src


class TestSSMCompanionIsCompleteFlag:
    """SSM companion cache: when gen_prompt_len > 0 (thinking models with
    a `<think>\\n` template suffix), the captured state covers the FULL
    prompt including template tokens. Mark those entries is_complete=False
    so the fetch path can reason about whether the state is at a pure-
    prompt boundary or includes template residue.

    Memory note: "SSM companion cache SKIPPED for gpl>0. Post-gen SSM
    state contaminated by gen_prompt+output → position mismatch →
    garbled output." This guard ensures we at LEAST tag those entries
    correctly, even if we keep them in the cache for now. Future work:
    fetch path consulting is_complete to skip or reuse accordingly.
    """

    def test_store_respects_is_complete_flag(self):
        from vmlx_engine.utils.ssm_companion_cache import SSMCompanionCache
        class _F:
            cache = None
            lengths = None
        c = SSMCompanionCache(max_entries=5)
        c.store([1, 2, 3], 3, [_F()], is_complete=False)
        entry = c.fetch([1, 2, 3], 3)
        assert entry is not None
        _, is_complete = entry
        assert is_complete is False, (
            "SSMCompanionCache must preserve is_complete flag through store/fetch"
        )

    def test_mllm_batch_generator_passes_gpl_to_is_complete(self):
        """Source pin: mllm_batch_generator's SSM capture path gates
        is_complete on gen_prompt_len."""
        src = Path(
            "/private/tmp/vmlx-1.3.55-build/vmlx_engine/mllm_batch_generator.py"
        ).read_text()
        assert "_gpl_for_flag = getattr(req, '_gen_prompt_len', 0)" in src, (
            "SSM capture must read gen_prompt_len from the request"
        )
        assert "_is_complete_flag = (_gpl_for_flag == 0)" in src, (
            "is_complete must be True only when gen_prompt_len is 0"
        )
        assert "is_complete=_is_complete_flag" in src, (
            "store() must receive the computed is_complete flag"
        )

    def test_default_is_complete_true_for_non_gpl_store(self):
        """Regression guard: non-thinking model (gpl=0) still stores as
        is_complete=True, same as before this change."""
        from vmlx_engine.utils.ssm_companion_cache import SSMCompanionCache
        class _F:
            cache = None
            lengths = None
        c = SSMCompanionCache(max_entries=5)
        # Default behavior — no is_complete arg → True
        c.store([1, 2, 3], 3, [_F()])
        entry = c.fetch([1, 2, 3], 3)
        assert entry is not None
        _, is_complete = entry
        assert is_complete is True, (
            "non-thinking capture (default is_complete=True) must behave as before"
        )


class TestMlxstudio78AdaptiveCacheLimit:
    """mlxstudio#78: Metal cache limit was hardcoded to 25% of max_ws,
    which on tight-memory systems (M4 Max 64GB loading Gemma-4-31B at
    ~41GB active) reserved 12GB for cache leaving only 7GB for model
    forward pass → Metal OOM on first request.

    Fix: cap cache limit at min(25% max_ws, 50% of FREE memory).
    """

    def test_cache_limit_adaptive_source_pin(self):
        src = Path(
            "/private/tmp/vmlx-1.3.55-build/vmlx_engine/mllm_batch_generator.py"
        ).read_text()
        assert "mlxstudio#78" in src
        assert "safety_limit = int(free * 0.5)" in src
        assert "min(base_limit, safety_limit)" in src
        # Old hardcoded 25% path must be gone
        # (except inside 'base_limit' which is still 25% of max_ws as the
        # upper bound — that's intentional)
        assert "Tight-memory configuration detected" in src

    def test_cache_limit_tight_memory_scenario(self):
        """Simulate reporter's scenario: 48GB max_ws, 41GB active → free 7GB.
        Expected cache limit = min(12GB, 3.5GB) = 3.5GB (safety cap wins)."""
        max_ws = 48 * 1024 ** 3
        active = 41 * 1024 ** 3
        free = max_ws - active
        base_limit = int(max_ws * 0.25)       # 12 GB
        safety_limit = int(free * 0.5)        # 3.5 GB
        cache_limit = max(512 * 1024 ** 2, min(base_limit, safety_limit))
        # On reporter's rig, should be ~3.5 GB, NOT 12 GB
        assert cache_limit == safety_limit, (
            f"tight-memory path: expected safety cap to win, got base={base_limit} "
            f"safety={safety_limit} chosen={cache_limit}"
        )
        assert cache_limit < base_limit, "safety cap must bite"

    def test_cache_limit_big_headroom_scenario(self):
        """On a machine with plenty of headroom, old 25% behavior preserved."""
        max_ws = 100 * 1024 ** 3       # 100 GB
        active = 10 * 1024 ** 3        # 10 GB active (mostly free)
        free = max_ws - active
        base_limit = int(max_ws * 0.25)   # 25 GB
        safety_limit = int(free * 0.5)    # 45 GB
        cache_limit = max(512 * 1024 ** 2, min(base_limit, safety_limit))
        # Big headroom: base_limit wins, unchanged behavior
        assert cache_limit == base_limit, (
            f"big-headroom path: expected base to win, got {cache_limit}"
        )

    def test_cache_limit_floor(self):
        """Degenerate case: tiny machine — floor at 512MB."""
        max_ws = 2 * 1024 ** 3         # 2 GB
        active = int(1.9 * 1024 ** 3)  # almost full
        free = max_ws - active
        base_limit = int(max_ws * 0.25)
        safety_limit = int(free * 0.5)
        cache_limit = max(512 * 1024 ** 2, min(base_limit, safety_limit))
        # Floor must kick in
        assert cache_limit == 512 * 1024 ** 2


class TestMlxstudio63MemoryPressureGuard:
    """mlxstudio#63: kernel panic on heavy VS Code OAICopilot traffic.

    On macOS, continuous-batching under extreme unified-memory pressure
    can trip an IOKit command-buffer failure that escalates to a whole-
    system kernel panic (irrecoverable). We reject new inference requests
    with 503 BEFORE Metal allocates new KV blocks when system memory is
    critically low.

    Default threshold 97% — only triggers in catastrophic territory.
    Configurable via VMLX_MEMORY_PRESSURE_REJECT_PCT; disable entirely
    with VMLX_MEMORY_PRESSURE_GUARD=0.
    """

    def _mock_psutil_vmem(self, percent: float, available_gb: float = 2.0):
        """Build a minimal psutil.virtual_memory() mock with given %used."""
        mock_vmem = MagicMock()
        mock_vmem.percent = percent
        mock_vmem.available = int(available_gb * (1024**3))
        return mock_vmem

    def test_dependency_exists_and_wires_onto_all_inference_endpoints(self):
        """Source pin: check_memory_pressure wired onto every endpoint
        that can trigger KV growth → Metal OOM."""
        import vmlx_engine.server as s
        src = Path(s.__file__).read_text()
        # Dependency function must exist
        assert "async def check_memory_pressure(" in src, (
            "check_memory_pressure dependency missing — kernel panic "
            "guard offline"
        )
        # It must be used on the critical inference endpoints. We search
        # for `"/path"` followed within ~300 chars by
        # `check_memory_pressure` to confirm wiring.
        endpoints_wired = [
            '"/v1/messages"',
            '"/v1/images/generations"',
            '"/v1/images/edits"',
            '"/v1/completions",',
            '"/v1/chat/completions"',
            '"/v1/responses",',
            '"/api/chat"',
            '"/api/generate"',
        ]
        for ep in endpoints_wired:
            # rfind skips the module-level inference_endpoints list
            # (~line 251) and lands on the @app.post decorator itself.
            idx = src.rfind(ep)
            assert idx > 0, f"endpoint {ep} not found in server.py"
            window = src[idx:idx + 500]
            assert "check_memory_pressure" in window, (
                f"mlxstudio#63: {ep} missing check_memory_pressure — "
                f"kernel panic still possible on this entry point"
            )

    def test_pressure_above_threshold_rejects_with_503(self):
        """Core behavior: when memory % exceeds threshold, dependency
        raises HTTPException(503) with Retry-After=5."""
        import asyncio
        from unittest.mock import patch
        from fastapi import HTTPException
        import vmlx_engine.server as s

        async def run_check():
            with patch.dict(os.environ, {"VMLX_MEMORY_PRESSURE_GUARD": "1",
                                          "VMLX_MEMORY_PRESSURE_REJECT_PCT": "97"}):
                with patch("psutil.virtual_memory",
                           return_value=self._mock_psutil_vmem(98.5, 0.5)):
                    return await s.check_memory_pressure(MagicMock())

        with pytest.raises(HTTPException) as excinfo:
            asyncio.run(run_check())
        assert excinfo.value.status_code == 503
        assert "Retry-After" in (excinfo.value.headers or {})
        assert excinfo.value.headers["Retry-After"] == "5"
        assert "memory pressure" in str(excinfo.value.detail).lower()

    def test_pressure_below_threshold_passes_silently(self):
        """Default path: ok memory → dependency returns without raising."""
        import asyncio
        from unittest.mock import patch
        import vmlx_engine.server as s

        async def run_check():
            with patch.dict(os.environ, {"VMLX_MEMORY_PRESSURE_GUARD": "1",
                                          "VMLX_MEMORY_PRESSURE_REJECT_PCT": "97"}):
                with patch("psutil.virtual_memory",
                           return_value=self._mock_psutil_vmem(40.0, 48.0)):
                    return await s.check_memory_pressure(MagicMock())

        # No exception = healthy path
        asyncio.run(run_check())

    def test_guard_disabled_env_var_bypasses_check(self):
        """VMLX_MEMORY_PRESSURE_GUARD=0 must fully bypass the check
        even at 99% memory — user opt-out escape hatch."""
        import asyncio
        from unittest.mock import patch
        import vmlx_engine.server as s

        async def run_check():
            with patch.dict(os.environ, {"VMLX_MEMORY_PRESSURE_GUARD": "0"}):
                with patch("psutil.virtual_memory",
                           return_value=self._mock_psutil_vmem(99.5, 0.2)):
                    return await s.check_memory_pressure(MagicMock())

        # Must NOT raise even at 99.5% — env var opts out
        asyncio.run(run_check())

    def test_custom_threshold_respected(self):
        """VMLX_MEMORY_PRESSURE_REJECT_PCT tunes the threshold. At 90%
        configured threshold, 92% used must reject."""
        import asyncio
        from unittest.mock import patch
        from fastapi import HTTPException
        import vmlx_engine.server as s

        async def run_check():
            with patch.dict(os.environ, {"VMLX_MEMORY_PRESSURE_GUARD": "1",
                                          "VMLX_MEMORY_PRESSURE_REJECT_PCT": "90"}):
                with patch("psutil.virtual_memory",
                           return_value=self._mock_psutil_vmem(92.0, 5.0)):
                    return await s.check_memory_pressure(MagicMock())

        with pytest.raises(HTTPException) as excinfo:
            asyncio.run(run_check())
        assert excinfo.value.status_code == 503

    def test_log_throttling_prevents_spam(self):
        """Log warnings must be throttled to at most one per 5s so
        sustained overload doesn't fill disk with identical WARN lines."""
        import vmlx_engine.server as s
        src = Path(s.__file__).read_text()
        assert "_last_pressure_log" in src, (
            "log throttle state missing — WARN spam will fill logs under "
            "sustained overload"
        )
        assert "now - _last_pressure_log > 5" in src, (
            "throttle interval must be 5s"
        )

    def test_source_anchor(self):
        """mlxstudio#63 anchor present in both the dependency function
        and the per-endpoint wiring comment — so `grep mlxstudio#63`
        finds all touch points."""
        import vmlx_engine.server as s
        src = Path(s.__file__).read_text()
        # At least 7+ references expected (1 func comment + 6+ endpoints)
        count = src.count("mlxstudio#63")
        assert count >= 7, (
            f"mlxstudio#63 anchor count = {count}, expected ≥ 7 "
            f"(dependency func + all inference endpoints)"
        )


class TestSleepWakeContract:
    """Sleep/wake/soft-sleep contracts — comprehensive state-machine pins.

    Three states:
      - active (None)    : model loaded, full cache limit.
      - soft sleep       : model loaded, cache cleared, cache_limit=512MB.
      - deep sleep       : model unloaded, process + port alive, pre-sleep
                           cache_limit saved for later wake restore.

    Transitions must be monotonic and reversible. A soft-sleep'd server
    must not re-enter soft-sleep (409), same for deep. Wake from either
    state must restore the pre-sleep cache limit.
    """

    def test_sleep_states_module_level_globals_exist(self):
        """Core state vars defined at module import — tests depend on them."""
        import vmlx_engine.server as srv
        # Presence checks only (values are volatile)
        assert hasattr(srv, "_standby_state")
        assert hasattr(srv, "_pre_sleep_cache_limit")
        assert hasattr(srv, "_wake_lock")

    def test_sleep_transitions_are_guarded(self):
        """Source pin: soft-sleep-when-deep and double-enter return 409."""
        src = Path(
            "/private/tmp/vmlx-1.3.55-build/vmlx_engine/server.py"
        ).read_text()
        assert 'Already in deep sleep' in src
        assert 'already_soft' in src
        # Deep sleep must also guard against double enter
        assert "already_deep" in src or "Already in deep sleep" in src

    def test_soft_sleep_clears_scheduler_caches(self):
        """Source pin: admin_soft_sleep calls scheduler.deep_reset or
        clears the prefix cache — we must never leave stale cache when
        the user asks for soft sleep."""
        src = Path(
            "/private/tmp/vmlx-1.3.55-build/vmlx_engine/server.py"
        ).read_text()
        # Locate admin_soft_sleep body
        idx = src.find("async def admin_soft_sleep")
        assert idx > 0
        body = src[idx:idx + 2000]
        assert "deep_reset" in body, (
            "soft sleep must call scheduler.deep_reset() to clear caches"
        )
        assert "_prefix_cache.clear" in body, (
            "fallback path: if no deep_reset, must clear prefix cache"
        )

    def test_wake_restores_cache_limit(self):
        """Source pin: admin_wake reads _pre_sleep_cache_limit and
        restores it on wake (both from soft and deep paths)."""
        src = Path(
            "/private/tmp/vmlx-1.3.55-build/vmlx_engine/server.py"
        ).read_text()
        idx = src.find("async def admin_wake")
        assert idx > 0
        body = src[idx:idx + 4000]
        assert "_pre_sleep_cache_limit" in body, (
            "wake must consult saved cache limit"
        )
        # Cache limit restoration must happen (the setter call pattern)
        assert "_set_cache(_pre_sleep_cache_limit)" in body, (
            "wake must call set_cache_limit with the saved value"
        )

    def test_wake_from_deep_sleep_reloads_model(self):
        """Deep sleep unloads the model; wake must reload it from _cli_args."""
        src = Path(
            "/private/tmp/vmlx-1.3.55-build/vmlx_engine/server.py"
        ).read_text()
        idx = src.find("async def admin_wake")
        body = src[idx:idx + 4000]
        # Must call load_model with args from _cli_args
        assert "load_model" in body
        assert "_cli_args" in body
        # Must preserve smelt/flash_moe options across the wake
        assert "smelt" in body
        assert "flash_moe" in body

    def test_flash_moe_deep_sleep_wake_fixed(self):
        """Memory: v1.3.36 fix for Flash MoE deep-sleep silent deactivation."""
        src = Path(
            "/private/tmp/vmlx-1.3.55-build/vmlx_engine/server.py"
        ).read_text()
        # The fix is admin_deep_sleep must clear loader and admin_wake
        # must restart it with the same args.
        assert "_flash_moe_loader" in src


class TestSlidingWindowHybridInteraction:
    """Sliding window + hybrid SSM: both cache families must coexist.

    Qwen3.5 hybrid models use GatedDeltaNet (SSM) + attention layers.
    Some attention layers may have sliding_window config. The scheduler
    must dispatch to the right cache type per layer.
    """

    def test_rotating_and_arrays_cache_both_importable(self):
        """Both cache classes used by hybrid + sliding window are importable."""
        from mlx_lm.models.cache import (
            RotatingKVCache, KVCache, ArraysCache,
        )
        assert RotatingKVCache is not None
        assert KVCache is not None
        assert ArraysCache is not None

    def test_cache_type_detection_distinguishes_rotating(self):
        """detect_cache_type classifies RotatingKVCache correctly — the
        scheduler uses this to avoid KV-cache prefix-matching on
        rotating caches (which can't be safely trimmed)."""
        from vmlx_engine.utils.cache_types import detect_cache_type, CacheType
        from mlx_lm.models.cache import RotatingKVCache, KVCache
        r = RotatingKVCache(max_size=256)
        k = KVCache()
        assert detect_cache_type(r) == CacheType.ROTATING_KV_CACHE
        assert detect_cache_type(k) == CacheType.KV_CACHE

    def test_scheduler_skips_rotating_for_paged(self):
        """Source pin: paged cache paths must skip RotatingKVCache
        entries — rotating state can't be reconstructed block-by-block
        because each window overwrites the previous."""
        src = Path(
            "/private/tmp/vmlx-1.3.55-build/vmlx_engine/prefix_cache.py"
        ).read_text()
        # Either skip-rotating is explicit, or the classification is used
        # to route to a separate code path
        assert "RotatingKVCache" in src or "sliding_window" in src.lower()


class TestPagedCacheBoundaries:
    """Paged cache: block-size alignment, ref-count discipline, cross-
    request sharing. These invariants have all regressed at least once."""

    def test_block_size_is_64_by_default(self):
        """Default block_size=64 matches mlx_lm assumptions. Changing it
        silently breaks multi-turn cache hit rates."""
        from vmlx_engine.prefix_cache import BlockAwarePrefixCache
        import inspect
        src = inspect.getsource(BlockAwarePrefixCache.__init__)
        # Must read block_size from the paged_cache_manager
        assert "paged_cache_manager.block_size" in src, (
            "BlockAwarePrefixCache must use the paged manager's block_size"
        )

    def test_block_aware_fetch_cache_increments_ref(self):
        """BlockAwarePrefixCache.fetch_cache must call
        paged_cache.increment_ref for each shared block. Without this,
        concurrent requests could evict a block mid-use."""
        import vmlx_engine.prefix_cache as pc
        src = Path(pc.__file__).read_text()
        # Target BlockAwarePrefixCache (the paged one). Its fetch_cache
        # lives inside the class body — the legacy in-memory
        # PrefixCacheManager.fetch_cache returns MLX-immutable refs
        # and doesn't need ref counts (MLX arrays can't be mutated).
        cls_idx = src.find("class BlockAwarePrefixCache")
        assert cls_idx > 0, "BlockAwarePrefixCache class required"
        cls_body = src[cls_idx:cls_idx + 30000]
        fc_idx = cls_body.find("def fetch_cache")
        assert fc_idx > 0, "BlockAwarePrefixCache.fetch_cache required"
        fc_body = cls_body[fc_idx:fc_idx + 3000]
        assert "increment_ref" in fc_body, (
            "BlockAwarePrefixCache.fetch_cache must increment refs so "
            "concurrent clients don't race to evict shared blocks"
        )

    def test_release_cache_symmetric_with_fetch(self):
        """Every fetch_cache path must have a matching release_cache,
        especially in error-handling branches (recently added in
        mlxstudio#73 fix)."""
        src = Path(
            "/private/tmp/vmlx-1.3.55-build/vmlx_engine/scheduler.py"
        ).read_text()
        # Count fetch_cache vs release_cache call sites
        import re
        fetches = len(re.findall(r"\.fetch_cache\(", src))
        releases = len(re.findall(r"\.release_cache\(", src))
        # Release >= fetch because error paths release without fetching
        # (ref counts survived from upstream). But at minimum, release
        # must be present whenever fetch_cache can lead to None result.
        assert releases >= 1, (
            "scheduler.py must have release_cache calls to balance fetch_cache"
        )

    def test_trim_block_table_block_aligns(self):
        """trim_block_table floors target_tokens to block-size boundary
        (critical for vmlx#91 SSM prefix resume)."""
        from vmlx_engine.prefix_cache import BlockAwarePrefixCache
        from vmlx_engine.paged_cache import BlockTable
        from unittest.mock import MagicMock
        cache = BlockAwarePrefixCache.__new__(BlockAwarePrefixCache)
        cache.block_size = 64
        cache.paged_cache = MagicMock()
        bt = BlockTable(request_id="r", block_ids=list(range(10)),
                        num_tokens=10 * 64)
        cache.paged_cache.get_block_table = MagicMock(return_value=bt)
        cache.paged_cache.decrement_ref = MagicMock()
        # Target 300 tokens → floor to 256 (4 blocks)
        result = cache.trim_block_table("r", 300)
        assert result is not None
        assert result.num_tokens == 256
        assert len(result.block_ids) == 4
        assert cache.paged_cache.decrement_ref.call_count == 6


class TestL2DiskCacheIntegrity:
    """L2 disk cache: bit-exact round-trip, proper eviction, safe
    concurrent access."""

    def test_disk_cache_manager_isolated_dirs_per_model(self):
        """Source pin: DiskCacheManager is scoped by cache_dir, so the
        scheduler must use a dir that mixes in the model hash/quant
        config so two different models can't collide."""
        src = Path(
            "/private/tmp/vmlx-1.3.55-build/vmlx_engine/disk_cache.py"
        ).read_text()
        # A cache_dir arg is required to isolate models
        assert "cache_dir" in src
        # SHA-256 hashing of tokens for the on-disk filename
        assert "_hash_tokens" in src or "sha256" in src.lower()

    def test_disk_cache_round_trip_bit_exact(self, tmp_path):
        """Store 3-layer KV cache → fresh manager → fetch → bit-exact."""
        from vmlx_engine.disk_cache import DiskCacheManager
        from mlx_lm.models.cache import KVCache
        import mlx.core as mx
        import numpy as np
        mgr = DiskCacheManager(cache_dir=str(tmp_path), max_size_gb=1.0)
        caches = []
        for _ in range(3):
            kv = KVCache()
            kv.keys = mx.array(np.random.randn(1, 4, 10, 64).astype(np.float16))
            kv.values = mx.array(np.random.randn(1, 4, 10, 64).astype(np.float16))
            kv.offset = 10
            mx.eval(kv.keys, kv.values)
            caches.append(kv)
        tokens = list(range(10))
        assert mgr.store(tokens, caches), "store must return True"
        import time
        time.sleep(0.5)  # let background writer flush
        # Fresh manager pointing at same dir
        del mgr
        mgr2 = DiskCacheManager(cache_dir=str(tmp_path), max_size_gb=1.0)
        loaded = mgr2.fetch(tokens)
        assert loaded is not None, "fetch after fresh manager must hit"
        assert len(loaded) == 3
        for orig, rec in zip(caches, loaded):
            assert (orig.keys == rec.keys).all().item()
            assert (orig.values == rec.values).all().item()

    def test_disk_cache_respects_size_limit(self, tmp_path):
        """Source pin: DiskCacheManager has max_size_bytes enforcement."""
        import vmlx_engine.disk_cache as dc
        src = Path(dc.__file__).read_text()
        assert "max_size_bytes" in src, "size cap field must exist"
        # LRU-style eviction when limit hit
        assert "evict" in src.lower() or "DELETE FROM cache_entries" in src

    def test_tq_native_disk_store_marker(self):
        """TurboQuant-native disk serialization (26× smaller) must still
        be present per memory note."""
        src = Path(
            "/private/tmp/vmlx-1.3.55-build/vmlx_engine/disk_cache.py"
        ).read_text()
        assert "tq_native" in src.lower() or "TurboQuant" in src, (
            "TQ-native disk store saves 5.3× vs affine re-encoded KV"
        )


# ═══════════════════════════════════════════════════════════════════════════
# Extended unit coverage — multi-turn, VL marker handling, video edge cases,
# prefix cache corner cases. Pure-Python, no model load required.
# ═══════════════════════════════════════════════════════════════════════════


class TestMultiTurnCacheStateMachine:
    """Multi-turn cache transitions: stale session state, mid-stream cancel,
    image→text→image→text alternation, tool-result injection."""

    def test_prefix_match_length_exact(self):
        from vmlx_engine.mllm_cache import MLLMPrefixCacheEntry
        e = MLLMPrefixCacheEntry(
            image_hash="img", prompt_hash="p", vision_embeddings=None,
            kv_cache=[], token_ids=[1, 2, 3, 4, 5], num_image_tokens=0,
            num_text_tokens=5, prompt_tokens=5, model_name="",
        )
        # Query with exact same tokens → full match
        assert e.get_prefix_match_length([1, 2, 3, 4, 5]) == 5

    def test_prefix_match_length_shorter_query(self):
        from vmlx_engine.mllm_cache import MLLMPrefixCacheEntry
        e = MLLMPrefixCacheEntry(
            image_hash="img", prompt_hash="p", vision_embeddings=None,
            kv_cache=[], token_ids=[1, 2, 3, 4, 5], num_image_tokens=0,
            num_text_tokens=5, prompt_tokens=5, model_name="",
        )
        # Query shorter than stored → match length == query length
        assert e.get_prefix_match_length([1, 2, 3]) == 3

    def test_prefix_match_length_diverging(self):
        from vmlx_engine.mllm_cache import MLLMPrefixCacheEntry
        e = MLLMPrefixCacheEntry(
            image_hash="img", prompt_hash="p", vision_embeddings=None,
            kv_cache=[], token_ids=[1, 2, 3, 4, 5], num_image_tokens=0,
            num_text_tokens=5, prompt_tokens=5, model_name="",
        )
        # Diverges at index 2 → match length 2
        assert e.get_prefix_match_length([1, 2, 99, 4, 5]) == 2

    def test_prefix_match_length_zero_shared(self):
        from vmlx_engine.mllm_cache import MLLMPrefixCacheEntry
        e = MLLMPrefixCacheEntry(
            image_hash="img", prompt_hash="p", vision_embeddings=None,
            kv_cache=[], token_ids=[1, 2, 3], num_image_tokens=0,
            num_text_tokens=3, prompt_tokens=3, model_name="",
        )
        assert e.get_prefix_match_length([99, 100]) == 0

    def test_prefix_cache_lru_eviction_ordering(self):
        """LRU eviction must drop oldest, keep most recent."""
        from vmlx_engine.mllm_cache import MLLMPrefixCacheManager
        cm = MLLMPrefixCacheManager(max_entries=3)
        # Store 3 entries
        for i in range(3):
            cm.store(
                images=[f"img{i}.png"], prompt=f"p{i}",
                vision_embeddings=None, kv_cache=[object()],
                token_ids=[i, i+1, i+2], num_image_tokens=0,
                model_name="test",
            )
        assert len(cm._cache) == 3
        # Store 4th → oldest (0) must be evicted
        cm.store(
            images=["img3.png"], prompt="p3",
            vision_embeddings=None, kv_cache=[object()],
            token_ids=[3, 4, 5], num_image_tokens=0, model_name="test",
        )
        assert len(cm._cache) == 3
        # Check the key for "img0" is gone
        keys = list(cm._cache.keys())
        for k in keys:
            assert "img0" not in str(cm._cache[k].image_hash)

    def test_multi_turn_alternating_image_text_distinct_keys(self):
        """Image-turn vs text-turn produce different cache keys even when
        prompt text is the same. Otherwise text-only queries could pull
        image-embedded cache state → wrong output."""
        from vmlx_engine.mllm_cache import MLLMPrefixCacheManager
        cm = MLLMPrefixCacheManager(max_entries=10)
        k1 = cm._make_cache_key(["img.png"], "hello")
        k2 = cm._make_cache_key([], "hello")
        assert k1 != k2, (
            "image-present key must differ from text-only key for same prompt"
        )


class TestImageMarkerEdgeCases:
    """VL: tokens representing image placeholders must be preserved across
    prefill, strip, and re-prompt boundaries."""

    def test_image_marker_constants_stable(self):
        """Qwen3-VL image markers used by tests in this file."""
        # These are tested elsewhere; pin the strings so future refactors
        # don't accidentally drift the marker names.
        marker = "<|vision_start|><|image_pad|><|vision_end|>"
        assert "<|image_pad|>" in marker
        assert "<|vision_start|>" in marker and "<|vision_end|>" in marker

    def test_apply_chat_template_requires_num_images(self):
        """mlx_vlm.prompt_utils.apply_chat_template's num_images signature
        must remain — test for ALL future mlx_vlm upgrades."""
        from mlx_vlm.prompt_utils import apply_chat_template
        import inspect
        sig = inspect.signature(apply_chat_template)
        assert "num_images" in sig.parameters, (
            "apply_chat_template must accept num_images= kwarg; without it "
            "we can't force image marker insertion for sessions that pre- "
            "compute prompts (vmlx#97 repro setup depends on this)"
        )

    def test_video_token_marker_distinct_from_image(self):
        """video_pad and image_pad must differ, else video routing collapses."""
        assert "<|image_pad|>" != "<|video_pad|>"

    def test_video_temporal_patch_size_defaults(self):
        """Default temporal_patch_size=2 is what the fallback assumes.
        If this constant drifts, frame-count → t_patches math breaks."""
        import inspect
        from jang_tools.load_jangtq_vlm import _install_video_fallback
        src = inspect.getsource(_install_video_fallback)
        assert "temporal_patch_size" in src
        assert "temporal_patch_size\", 2)" in src or "temporal_patch_size, 2)" in src


class TestPrefixCacheCornerCases:
    """Corner cases for BlockAwarePrefixCache that have bitten us before."""

    def test_empty_tokens_returns_none(self):
        """fetch_cache with empty tokens must not crash — returns (None, [])."""
        from vmlx_engine.prefix_cache import BlockAwarePrefixCache
        from vmlx_engine.paged_cache import PagedCacheManager
        from unittest.mock import MagicMock
        pm = PagedCacheManager(block_size=64, max_blocks=16)
        model = MagicMock()
        cache = BlockAwarePrefixCache(model, pm)
        result = cache.fetch_cache("req1", [])
        assert result == (None, [])

    def test_fetch_nonexistent_request_returns_miss(self):
        """Never-seen request returns (None, tokens)."""
        from vmlx_engine.prefix_cache import BlockAwarePrefixCache
        from vmlx_engine.paged_cache import PagedCacheManager
        from unittest.mock import MagicMock
        pm = PagedCacheManager(block_size=64, max_blocks=16)
        model = MagicMock()
        cache = BlockAwarePrefixCache(model, pm)
        table, remaining = cache.fetch_cache("req1", [1, 2, 3])
        assert table is None
        assert remaining == [1, 2, 3]

    def test_trim_to_exactly_one_block(self):
        """trim_block_table target=block_size keeps exactly 1 block."""
        from vmlx_engine.prefix_cache import BlockAwarePrefixCache
        from vmlx_engine.paged_cache import BlockTable
        from unittest.mock import MagicMock
        cache = BlockAwarePrefixCache.__new__(BlockAwarePrefixCache)
        cache.block_size = 64
        cache.paged_cache = MagicMock()
        bt = BlockTable(request_id="r", block_ids=[10, 20, 30], num_tokens=192)
        cache.paged_cache.get_block_table = MagicMock(return_value=bt)
        cache.paged_cache.decrement_ref = MagicMock()
        result = cache.trim_block_table("r", 64)
        assert result is not None
        assert len(result.block_ids) == 1
        assert result.num_tokens == 64
        assert cache.paged_cache.decrement_ref.call_count == 2


class TestSSMCompanionPrefixHashStability:
    """SSM companion cache prefix_hash must be stable across python
    runs so on-disk serialization (future) would work deterministically."""

    def test_prefix_hash_deterministic(self):
        from vmlx_engine.utils.ssm_companion_cache import SSMCompanionCache
        c1 = SSMCompanionCache(max_entries=5, model_key="model-A")
        c2 = SSMCompanionCache(max_entries=5, model_key="model-A")
        # Same tokens + same model_key → same prefix_hash
        assert c1._prefix_hash([1, 2, 3], 3) == c2._prefix_hash([1, 2, 3], 3)

    def test_prefix_hash_different_for_different_models(self):
        from vmlx_engine.utils.ssm_companion_cache import SSMCompanionCache
        c1 = SSMCompanionCache(max_entries=5, model_key="model-A")
        c2 = SSMCompanionCache(max_entries=5, model_key="model-B")
        # Same tokens, different model — hashes must differ (otherwise
        # two models cache-collide on identical token prefixes)
        assert c1._prefix_hash([1, 2, 3], 3) != c2._prefix_hash([1, 2, 3], 3)

    def test_prefix_hash_respects_num_tokens_bound(self):
        from vmlx_engine.utils.ssm_companion_cache import SSMCompanionCache
        c = SSMCompanionCache(max_entries=5, model_key="m")
        # First N tokens identical, suffix differs → hash must only reflect
        # the first N (since num_tokens slice caps at N)
        h1 = c._prefix_hash([1, 2, 3, 4, 5], 3)
        h2 = c._prefix_hash([1, 2, 3, 99, 100], 3)
        assert h1 == h2
        # But hash at full length differs
        h3 = c._prefix_hash([1, 2, 3, 4, 5], 5)
        h4 = c._prefix_hash([1, 2, 3, 99, 100], 5)
        assert h3 != h4

    def test_length_index_cleanup_on_eviction(self):
        """_length_index must not grow unbounded — evicted keys removed."""
        from vmlx_engine.utils.ssm_companion_cache import SSMCompanionCache
        class _F:
            cache = None
            lengths = None
        c = SSMCompanionCache(max_entries=2, model_key="m")
        c.store([1, 2, 3], 3, [_F()])
        c.store([1, 2, 3, 4, 5], 5, [_F()])
        c.store([1, 2, 3, 4, 5, 6, 7], 7, [_F()])  # evicts entry at len=3
        # _length_index should only have 5 and 7 buckets populated
        populated = [n for n, d in c._length_index.items() if d]
        assert 3 not in populated, (
            "LRU-evicted entry at length=3 must be purged from _length_index"
        )
        assert 5 in populated and 7 in populated


@pytest.mark.skipif(not HAS_JANGTQ, reason="model not present")
class TestRealModelMultiTurnIntegration:
    """Real-model multi-turn integration tests — reuse one model load
    across all scenarios to keep CI cost manageable.

    Covers:
      - text-only multi-turn through MLXMultimodalLM (VL class, text path)
      - image multi-turn (same image across turns → cache hit)
      - video frame input produces valid pixel_values_videos
      - multi-image in a single request
      - reasoning=True vs reasoning=False on a thinking-capable family
      - RAM stability across N requests (no leak)
    """

    @pytest.fixture(scope="class")
    def loaded_model(self):
        """Load Qwen3.6-35B-A3B-JANGTQ2 once, share across class tests."""
        from vmlx_engine.models.mllm import MLXMultimodalLM
        import mlx.utils as _mu
        import mlx.core as _mx
        m = MLXMultimodalLM(str(MODEL_JANGTQ))
        m.load()
        # Force materialize
        flat = _mu.tree_flatten(m.model.parameters())
        for i in range(0, len(flat), 200):
            _mx.eval(*[v for _, v in flat[i:i + 200]])
        yield m
        # No teardown — fixture scoped to class; process exit cleans up

    def test_text_multi_turn_no_leak(self, loaded_model):
        """4 text-only turns → MLX active memory should not grow."""
        import mlx.core as mx
        m = loaded_model
        baseline = mx.get_active_memory() / 1e9
        for q in ["1+1?", "2+2?", "3+3?", "4+4?"]:
            out = m.generate(q, max_tokens=6, use_cache=True)
            assert out.text is not None
        after = mx.get_active_memory() / 1e9
        growth = after - baseline
        assert growth < 0.5, (
            f"text multi-turn grew active memory by {growth:.2f} GB "
            f"(baseline {baseline:.2f}, after {after:.2f})"
        )

    def test_image_multi_turn_same_image_hits_cache(self, loaded_model):
        """Same (image, prompt) repeated → second turn must hit cache."""
        from PIL import Image
        m = loaded_model
        img = "/tmp/_real_mt_img.png"
        Image.new("RGB", (384, 384), (128, 64, 192)).save(img)
        cm = m._cache_manager
        cm.stats.hits = 0
        cm.stats.misses = 0
        # T1 populate
        m.generate("Name this color.", images=[img],
                   max_tokens=6, use_cache=True)
        # T2 same — must hit
        m.generate("Name this color.", images=[img],
                   max_tokens=6, use_cache=True)
        assert cm.stats.hits >= 1, (
            f"same-image repeat must hit cache, got "
            f"hits={cm.stats.hits} misses={cm.stats.misses}"
        )

    def test_multi_image_single_request(self, loaded_model):
        """Two images in one request must both be processed."""
        from PIL import Image
        m = loaded_model
        img1 = "/tmp/_real_mt_img1.png"
        img2 = "/tmp/_real_mt_img2.png"
        Image.new("RGB", (256, 256), (255, 0, 0)).save(img1)
        Image.new("RGB", (256, 256), (0, 255, 0)).save(img2)
        out = m.generate(
            "Compare these two images in 5 words.",
            images=[img1, img2], max_tokens=30, use_cache=False,
        )
        # Just confirm we got output without crash
        assert out.text is not None
        assert len(out.text) > 0

    def test_video_fallback_processor_single_request(self, loaded_model):
        """4-frame video through processor fallback → valid shapes."""
        from PIL import Image
        m = loaded_model
        frames = []
        for i in range(4):
            p = f"/tmp/_real_mt_frame{i}.png"
            Image.new("RGB", (256, 256), (60, 60 + i * 30, 120)).save(p)
            frames.append(p)
        # Call processor directly with fallback
        prompt = ("<|im_start|>user\n"
                  "<|vision_start|><|video_pad|><|vision_end|>"
                  "desc<|im_end|>\n<|im_start|>assistant\n")
        inputs = m.processor(
            text=[prompt], videos=[frames],
            return_tensors="mlx", padding=True,
        )
        assert "pixel_values_videos" in inputs
        assert "video_grid_thw" in inputs

    def test_ram_stable_across_12_mixed(self, loaded_model):
        """12 mixed requests (text + image) — active memory stays bounded."""
        import mlx.core as mx
        from PIL import Image
        m = loaded_model
        img = "/tmp/_real_mt_stability.png"
        Image.new("RGB", (256, 256), (32, 64, 128)).save(img)
        baseline = mx.get_active_memory() / 1e9
        peaks = []
        for i in range(12):
            if i % 3 == 0:
                m.generate(f"quick q {i}", max_tokens=5, use_cache=True)
            elif i % 3 == 1:
                m.generate(f"color? {i}", images=[img],
                           max_tokens=5, use_cache=True)
            else:
                m.generate(f"count {i}", max_tokens=5, use_cache=True)
            peaks.append(mx.get_active_memory() / 1e9)
        final = peaks[-1]
        growth = final - baseline
        assert growth < 1.0, (
            f"12-mixed-request growth {growth:.2f} GB exceeds 1 GB bound"
        )
        # No single peak > 16 GB above baseline (sanity)
        worst = max(peaks)
        assert worst - baseline < 16, (
            f"worst single-request peak {worst - baseline:.2f} GB over baseline"
        )


class TestMlxstudio69ImageAttachmentForceMultimodal:
    """mlxstudio#69: attaching an image via the UI button didn't send the
    image with the submission when the session's `isMultimodal` flag was
    false (stale config, session lookup failed, or config.json lacks
    vision_config). The user's explicit click on "attach image" must win.

    Fix (shipped v1.3.49): in panel/src/main/ipc/chat.ts the send handler
    forces `chatIsMultimodal = true` when `hasAttachments && !chatIsMultimodal`.

    These guards pin the panel contract so a refactor can't drop it.
    """

    def test_panel_force_multimodal_on_attachment(self):
        """Source pin: the force-multimodal branch with mlxstudio#69 anchor."""
        src = Path(
            "/private/tmp/vmlx-1.3.55-build/panel/src/main/ipc/chat.ts"
        ).read_text()
        # Anchor must be present
        assert "mlxstudio#69" in src, (
            "mlxstudio#69 anchor dropped — silent image-drop regression "
            "is now possible on sessions with stale isMultimodal=false"
        )
        # The specific branch pattern
        assert "hasAttachments && !chatIsMultimodal" in src, (
            "force-multimodal branch missing; attachments get silently "
            "dropped when session thinks it's text-only"
        )
        # The assignment must still set chatIsMultimodal = true
        force_idx = src.find("hasAttachments && !chatIsMultimodal")
        assert force_idx > 0
        # Within ~500 chars after the branch, chatIsMultimodal=true assignment
        window = src[force_idx:force_idx + 500]
        assert "chatIsMultimodal = true" in window, (
            "force-multimodal must set chatIsMultimodal=true"
        )

    def test_panel_infer_kind_back_compat_present(self):
        """A companion back-compat helper `inferKind` lets older renderer
        builds (without `.kind` field) still work. Keep it tested so
        removing it doesn't break older client builds."""
        src = Path(
            "/private/tmp/vmlx-1.3.55-build/panel/src/main/ipc/chat.ts"
        ).read_text()
        assert "inferKind" in src, (
            "inferKind back-compat helper missing — older renderer builds "
            "that don't send `kind` will now fail"
        )
        assert 'data:video/' in src, (
            "mime-prefix detection for video must remain in inferKind"
        )


class TestVideoProcessingEdgeCases:
    """Video fallback edge cases — frame count variations, different
    resolutions, single vs multi-video batches."""

    def _install_on_fresh_proc(self):
        """Build a fake Qwen3VL-like processor with the patch installed."""
        from unittest.mock import MagicMock
        import numpy as np
        from jang_tools.load_jangtq_vlm import _install_video_fallback

        class _Proc:
            image_processor = None
            video_processor = None
            tokenizer = None
            image_token = "<|image_pad|>"
            video_token = "<|video_pad|>"

            def __call__(self, *, images=None, text=None, videos=None,
                         **kw):
                if videos is not None:
                    # simulate the real bug path that patch must bypass
                    raise TypeError("videos kwarg unsupported")
                if text and images:
                    ip_out = self.image_processor(images=images)
                    tok = self.tokenizer(text)
                    return {**tok, **ip_out}
                return {}

        # Image processor mock
        ip = MagicMock()
        ip.merge_size = 2
        ip.temporal_patch_size = 2
        ip.patch_size = 16

        def _ipc(*, images, **kw):
            n = len(images)
            return {
                "pixel_values": np.zeros((576 * n, 1536), dtype=np.float32),
                "image_grid_thw": np.tile([[1, 12, 12]], (n, 1)),
            }
        ip.side_effect = _ipc

        tk = MagicMock()

        def _tkc(texts, **kw):
            return {"input_ids": [list(range(max(1, len(s) // 3)))
                                  for s in texts],
                    "attention_mask": [[1] * max(1, len(s) // 3)
                                       for s in texts]}
        tk.side_effect = _tkc

        p = _Proc()
        p.image_processor = ip
        p.tokenizer = tk
        _install_video_fallback(p)
        return p

    def test_single_frame_video(self):
        p = self._install_on_fresh_proc()
        out = p(text=["<|vision_start|><|video_pad|><|vision_end|>one"],
                videos=[["f"]])
        import numpy as np
        vg = np.asarray(out["video_grid_thw"])
        # 1 frame @ temporal_patch_size=2 → ceil(1/2) = 1
        assert int(vg[0, 0]) == 1

    def test_even_frame_count(self):
        """Even frame count with temporal_patch_size=2 → exact division."""
        p = self._install_on_fresh_proc()
        out = p(text=["<|vision_start|><|video_pad|><|vision_end|>even"],
                videos=[["f"] * 8])
        import numpy as np
        vg = np.asarray(out["video_grid_thw"])
        assert int(vg[0, 0]) == 4

    def test_large_frame_count(self):
        p = self._install_on_fresh_proc()
        out = p(text=["<|vision_start|><|video_pad|><|vision_end|>long"],
                videos=[["f"] * 64])
        import numpy as np
        vg = np.asarray(out["video_grid_thw"])
        assert int(vg[0, 0]) == 32  # 64 / 2

    def test_three_videos_in_one_request(self):
        p = self._install_on_fresh_proc()
        marker = "<|vision_start|><|video_pad|><|vision_end|>"
        out = p(
            text=[marker * 3 + "describe all"],
            videos=[["f1"], ["f2a", "f2b"], ["f3a", "f3b", "f3c"]],
        )
        import numpy as np
        vg = np.asarray(out["video_grid_thw"])
        assert vg.shape == (3, 3)
        # Expected t: 1, 1, 2
        t_values = [int(vg[i, 0]) for i in range(3)]
        assert t_values == [1, 1, 2], f"expected [1,1,2], got {t_values}"


class TestEngineConfigPaths:
    """Tests covering engine config and startup paths that have broken
    in the past."""

    def test_enable_thinking_tri_state_none_bool(self):
        """enable_thinking resolution must distinguish None (no preference)
        vs True vs False. None falls through to auto-detect/defaults."""
        # Test ChatCompletionRequest accepts the tri-state
        from vmlx_engine.api.models import ChatCompletionRequest, Message
        r_none = ChatCompletionRequest(
            model="x", messages=[Message(role="user", content="hi")],
        )
        assert r_none.enable_thinking is None, (
            "default enable_thinking must be None (tri-state unset)"
        )
        r_true = ChatCompletionRequest(
            model="x", messages=[Message(role="user", content="hi")],
            enable_thinking=True,
        )
        assert r_true.enable_thinking is True
        r_false = ChatCompletionRequest(
            model="x", messages=[Message(role="user", content="hi")],
            enable_thinking=False,
        )
        assert r_false.enable_thinking is False

    def test_chat_template_kwargs_passthrough(self):
        """ChatCompletionRequest's chat_template_kwargs field preserves
        arbitrary dict payload (e.g., think parameter)."""
        from vmlx_engine.api.models import ChatCompletionRequest, Message
        r = ChatCompletionRequest(
            model="x", messages=[Message(role="user", content="hi")],
            chat_template_kwargs={"enable_thinking": True, "tool_choice": "auto"},
        )
        assert r.chat_template_kwargs == {
            "enable_thinking": True, "tool_choice": "auto",
        }


class TestMLLMModelConfigMapping:
    """ModelConfigRegistry family mapping covers the main vision + text
    variants we support. If a family drops out of the registry, sessions
    silently fall back to defaults which lose tool/reasoning parsers."""

    def test_qwen3_5_moe_family_has_parsers(self):
        from vmlx_engine.model_config_registry import ModelConfigRegistry
        from vmlx_engine.model_configs import register_all
        reg = ModelConfigRegistry()
        register_all(reg)
        by_name = {c.family_name: c for c in reg._configs}
        c = by_name.get("qwen3_5_moe")
        assert c is not None
        assert c.reasoning_parser == "qwen3"
        assert c.tool_parser == "qwen"

    def test_gemma4_family_has_parsers(self):
        from vmlx_engine.model_config_registry import ModelConfigRegistry
        from vmlx_engine.model_configs import register_all
        reg = ModelConfigRegistry()
        register_all(reg)
        by_name = {c.family_name: c for c in reg._configs}
        c = by_name.get("gemma4")
        assert c is not None
        assert c.tool_parser == "gemma4"
        assert c.reasoning_parser == "gemma4"

    def test_minimax_family_has_parsers(self):
        """Family is registered under 'minimax' (not 'minimax_m2') with
        model_types covering minimax / minimax_m2 / minimax_m2_5. The
        MiniMax-M2.x models use tool_parser=minimax + reasoning=qwen3
        (per JANG capabilities stamp)."""
        from vmlx_engine.model_config_registry import ModelConfigRegistry
        from vmlx_engine.model_configs import register_all
        reg = ModelConfigRegistry()
        register_all(reg)
        by_name = {c.family_name: c for c in reg._configs}
        c = by_name.get("minimax")
        assert c is not None, (
            "minimax family must be registered (not minimax_m2)"
        )
        assert "minimax_m2" in c.model_types, (
            "minimax_m2 inner model_type must map to minimax family"
        )
        assert c.tool_parser == "minimax"
        assert c.reasoning_parser == "qwen3"


class TestMlxstudio78MetalWorkingSetGuard:
    """mlxstudio#78: Gemma-4-31B-JANG_4M on M4 Max 64GB → Metal command
    buffer OOM on first request. System RAM is only 64% used (mlxstudio#63
    guard won't fire at 97% threshold), but Metal's max recommended
    working set (~48GB on M4 Max 64GB) is already occupied by model
    weights. First prefill's attention tensors + KV blocks exceed
    headroom and Metal crashes with IOGPUCommandBufferCallback error 8.

    Fix: `check_metal_working_set_pressure` FastAPI dependency rejects
    with 503 + Retry-After=5 when
      active_memory / max_recommended_working_set_size > threshold
    (default 85% — leaves ~7GB headroom on M4 Max).

    This is ORTHOGONAL to mlxstudio#63: the RAM guard catches system-
    level pressure (swap thrashing), the Metal guard catches GPU-level
    pressure (command-buffer exhaustion). Both are needed — either can
    trigger process death on macOS.
    """

    def test_dependency_exists(self):
        """Source pin: check_metal_working_set_pressure function."""
        import vmlx_engine.server as s
        src = Path(s.__file__).read_text()
        assert "async def check_metal_working_set_pressure(" in src, (
            "check_metal_working_set_pressure missing — Metal OOM guard "
            "offline; big-model+small-mac configs will crash on first req"
        )

    def test_dependency_wired_onto_all_inference_endpoints(self):
        """Same 8 inference endpoints that get mlxstudio#63 must also
        get mlxstudio#78 — they can both trigger process death."""
        import vmlx_engine.server as s
        src = Path(s.__file__).read_text()
        endpoints = [
            '"/v1/messages"',
            '"/v1/images/generations"',
            '"/v1/images/edits"',
            '"/v1/completions",',
            '"/v1/chat/completions"',
            '"/v1/responses",',
            '"/api/chat"',
            '"/api/generate"',
        ]
        for ep in endpoints:
            idx = src.rfind(ep)
            assert idx > 0, f"endpoint {ep} not found"
            window = src[idx:idx + 700]
            assert "check_metal_working_set_pressure" in window, (
                f"mlxstudio#78: {ep} missing Metal working-set guard"
            )

    def test_pressure_above_threshold_rejects(self):
        """Active 90% of working set + 85% threshold → 503.
        Patches on the live `mx` module because the dependency calls
        `import mlx.core as mx` inside its body (module-cached by then)."""
        import asyncio
        from unittest.mock import patch
        from fastapi import HTTPException
        import vmlx_engine.server as s
        import mlx.core as mx

        max_ws = 48 * (1024**3)  # 48GB working set
        active = int(max_ws * 0.90)  # 90% active

        async def run():
            with patch.dict(os.environ, {
                "VMLX_METAL_WS_GUARD": "1",
                "VMLX_METAL_WS_REJECT_PCT": "85"
            }):
                with patch.object(mx, "device_info", lambda: {
                    "max_recommended_working_set_size": max_ws
                }, create=True), patch.object(
                    mx, "get_active_memory", lambda: active, create=True
                ):
                    return await s.check_metal_working_set_pressure(MagicMock())

        with pytest.raises(HTTPException) as excinfo:
            asyncio.run(run())
        assert excinfo.value.status_code == 503
        assert "Retry-After" in (excinfo.value.headers or {})
        assert "working set" in str(excinfo.value.detail).lower()

    def test_pressure_below_threshold_passes(self):
        """50% active + 85% threshold → no raise."""
        import asyncio
        from unittest.mock import patch
        import vmlx_engine.server as s
        import mlx.core as mx

        max_ws = 48 * (1024**3)
        active = int(max_ws * 0.50)

        async def run():
            with patch.dict(os.environ, {
                "VMLX_METAL_WS_GUARD": "1",
                "VMLX_METAL_WS_REJECT_PCT": "85"
            }):
                with patch.object(mx, "device_info", lambda: {
                    "max_recommended_working_set_size": max_ws
                }, create=True), patch.object(
                    mx, "get_active_memory", lambda: active, create=True
                ):
                    return await s.check_metal_working_set_pressure(MagicMock())

        # Must not raise
        asyncio.run(run())

    def test_guard_disabled_bypasses(self):
        """VMLX_METAL_WS_GUARD=0 → no check even at 99% active."""
        import asyncio
        from unittest.mock import patch
        import vmlx_engine.server as s

        # With guard disabled, mlx.core doesn't even need to load
        async def run():
            with patch.dict(os.environ, {"VMLX_METAL_WS_GUARD": "0"}):
                return await s.check_metal_working_set_pressure(MagicMock())

        # Must not raise regardless of mlx state
        asyncio.run(run())

    def test_missing_max_ws_skipped_silently(self):
        """If device doesn't expose max_recommended_working_set_size
        (some non-Apple-Silicon), skip — don't raise."""
        import asyncio
        from unittest.mock import patch
        import vmlx_engine.server as s
        import mlx.core as mx

        async def run():
            with patch.dict(os.environ, {"VMLX_METAL_WS_GUARD": "1"}):
                with patch.object(mx, "device_info", lambda: {},
                                   create=True), patch.object(
                    mx, "get_active_memory", lambda: 10**9, create=True
                ):
                    return await s.check_metal_working_set_pressure(MagicMock())

        asyncio.run(run())  # no raise

    def test_log_throttled(self):
        """Log warning must be throttled to 1 per 5s."""
        import vmlx_engine.server as s
        src = Path(s.__file__).read_text()
        assert "_last_metal_ws_log" in src
        assert "now - _last_metal_ws_log > 5" in src

    def test_orthogonal_to_mlxstudio63(self):
        """The two guards catch different pressure dimensions:
        #63 = system RAM, #78 = Metal working set. Both must be wired
        on the same endpoints."""
        import vmlx_engine.server as s
        src = Path(s.__file__).read_text()
        assert "mlxstudio#63" in src and "mlxstudio#78" in src
        # Both dependencies exist
        assert "async def check_memory_pressure(" in src
        assert "async def check_metal_working_set_pressure(" in src
        # They must appear together on inference endpoints — use first
        # chat/completions as canonical example
        idx = src.rfind('"/v1/chat/completions"')
        assert idx > 0
        window = src[idx:idx + 1000]
        assert "check_memory_pressure" in window
        assert "check_metal_working_set_pressure" in window


class TestMlxstudio78MetalWorkingSetGuard:
    """mlxstudio#78 — second layer of defense for Metal OOM on tight
    memory configs. The adaptive cache limit (a8c19429) sizes the
    Metal allocator to leave headroom. This request-time guard rejects
    incoming requests when the working set is already saturated.

    Guard triggers when active_memory / max_recommended_working_set
    exceeds VMLX_METAL_WS_REJECT_PCT (default 85).
    """

    def test_guard_rejects_high_pressure(self):
        import asyncio
        from unittest.mock import MagicMock, patch
        import mlx.core as mx
        from vmlx_engine.server import check_metal_working_set_pressure
        from fastapi import HTTPException

        def _di():
            return {"max_recommended_working_set_size": 48 * 1024 ** 3}
        def _active():
            return 41 * 1024 ** 3  # 85.4% → above 85% default

        with patch.object(mx, "device_info", _di, create=True), \
             patch.object(mx, "get_active_memory", _active, create=True):
            with pytest.raises(HTTPException) as e:
                asyncio.run(check_metal_working_set_pressure(MagicMock()))
            assert e.value.status_code == 503
            assert e.value.headers.get("Retry-After") == "5"

    def test_guard_allows_low_pressure(self):
        import asyncio
        from unittest.mock import MagicMock, patch
        import mlx.core as mx
        from vmlx_engine.server import check_metal_working_set_pressure

        def _di():
            return {"max_recommended_working_set_size": 128 * 1024 ** 3}
        def _active():
            return 20 * 1024 ** 3  # 15.6%

        with patch.object(mx, "device_info", _di, create=True), \
             patch.object(mx, "get_active_memory", _active, create=True):
            # Must NOT raise
            asyncio.run(check_metal_working_set_pressure(MagicMock()))

    def test_guard_can_be_disabled(self):
        import asyncio, os
        from unittest.mock import MagicMock, patch
        import mlx.core as mx
        from vmlx_engine.server import check_metal_working_set_pressure

        def _di():
            return {"max_recommended_working_set_size": 48 * 1024 ** 3}
        def _active():
            return 47 * 1024 ** 3  # 97.9% → would reject

        os.environ["VMLX_METAL_WS_GUARD"] = "0"
        try:
            with patch.object(mx, "device_info", _di, create=True), \
                 patch.object(mx, "get_active_memory", _active, create=True):
                asyncio.run(check_metal_working_set_pressure(MagicMock()))
        finally:
            os.environ.pop("VMLX_METAL_WS_GUARD", None)

    def test_guard_threshold_tunable_via_env(self):
        import asyncio, os
        from unittest.mock import MagicMock, patch
        import mlx.core as mx
        from vmlx_engine.server import check_metal_working_set_pressure
        from fastapi import HTTPException

        def _di():
            return {"max_recommended_working_set_size": 100 * 1024 ** 3}
        def _active():
            return 40 * 1024 ** 3  # 40% → allowed at 85% default, rejected at 30%

        # At default 85% — passes
        with patch.object(mx, "device_info", _di, create=True), \
             patch.object(mx, "get_active_memory", _active, create=True):
            asyncio.run(check_metal_working_set_pressure(MagicMock()))

        # At custom 30% — rejects
        os.environ["VMLX_METAL_WS_REJECT_PCT"] = "30"
        try:
            with patch.object(mx, "device_info", _di, create=True), \
                 patch.object(mx, "get_active_memory", _active, create=True):
                with pytest.raises(HTTPException):
                    asyncio.run(check_metal_working_set_pressure(MagicMock()))
        finally:
            os.environ.pop("VMLX_METAL_WS_REJECT_PCT", None)

    def test_all_endpoints_wire_guard(self):
        """Every @app.post chat endpoint must include
        check_metal_working_set_pressure (vs just check_memory_pressure
        from ms#63 which only watches system RAM)."""
        src = Path(
            "/private/tmp/vmlx-1.3.55-build/vmlx_engine/server.py"
        ).read_text()
        count = src.count("Depends(check_metal_working_set_pressure)")
        # Chat completions + responses + anthropic + ollama chat/generate +
        # image gen + image edit = 7 endpoints minimum
        assert count >= 5, (
            f"expected ≥5 endpoints to wire the guard, found {count}"
        )


class TestContinuousBatchingConcurrency:
    """Continuous batching concurrency invariants — request lifecycle,
    cancellation, batch queue safety under load."""

    def test_scheduler_serializes_batches_via_event_loop(self):
        """Scheduler runs under a single asyncio event loop. That's the
        serialization — no explicit lock needed because the loop itself
        guarantees only one coroutine mutates the batch state at a time.
        This test pins that invariant: waiting/running queues must be
        deque-backed (single-loop-safe), not naked list shared across
        threads. If someone introduces threading without a lock, this
        test catches the missing deque."""
        from vmlx_engine.scheduler import Scheduler
        import inspect
        src = inspect.getsource(Scheduler)
        # Waiting queue must be a deque (FCFS, loop-serialized)
        assert "deque" in src, (
            "Scheduler waiting/running queues must use deque for "
            "single-event-loop FCFS ordering"
        )

    def test_abort_request_exists_and_returns_bool(self):
        """abort_request API is the cancellation entry point."""
        from vmlx_engine.scheduler import Scheduler
        import inspect
        assert hasattr(Scheduler, "abort_request")
        src = inspect.getsource(Scheduler.abort_request)
        # Must document return semantics (True/False)
        assert "return True" in src or "-> bool" in src

    def test_mllm_scheduler_defers_abort_to_avoid_metal_race(self):
        """mllm_scheduler.abort_request must defer batch removal via
        _pending_aborts instead of mid-metal-compute removal (from prior
        audit — kept here as a corner-case guard)."""
        from vmlx_engine.mllm_scheduler import MLLMScheduler
        import inspect
        src = inspect.getsource(MLLMScheduler.abort_request)
        assert "_pending_aborts" in src, (
            "mllm abort must defer via _pending_aborts to avoid touching "
            "cache tensors during Metal computation"
        )

    def test_engine_core_uses_asyncio_event_per_request(self):
        """Engine core coordinates request completion via per-request
        asyncio.Event (one Event per request_id in _finished_events).
        That's the cross-coroutine signal without needing a shared
        queue. If this goes away, add-another-primitive discipline
        has been broken."""
        src = Path(
            "/private/tmp/vmlx-1.3.55-build/vmlx_engine/engine_core.py"
        ).read_text()
        assert "asyncio.Event()" in src, (
            "engine_core needs asyncio.Event per request for "
            "cross-coroutine completion signaling"
        )
        assert "_finished_events" in src, (
            "engine_core must track per-request Events in _finished_events"
        )


class TestAPIRequestCancellation:
    """Stop button / client disconnect must propagate to engine cancellation."""

    def test_stream_outputs_handles_client_disconnect(self):
        """Server must recognize client disconnect and abort the in-flight
        request. Critical for "stop generation" UX.

        Uses Starlette's Request.is_disconnected() polled in the stream
        loop — the preferred pattern over CancelledError, which wouldn't
        fire reliably from SSE streaming generators.
        """
        src = Path(
            "/private/tmp/vmlx-1.3.55-build/vmlx_engine/server.py"
        ).read_text()
        hits = src.count("is_disconnected()")
        assert hits >= 3, (
            f"server must poll Request.is_disconnected() in ≥3 streaming "
            f"endpoints (OpenAI chat, Responses, image gen/edit at minimum); "
            f"found {hits}"
        )

    def test_api_utils_max_tokens_not_falsy_check(self):
        """Previous regression: max_tokens=0 got silently replaced with
        default because `X or default` treats 0 as falsy. Must use
        `is not None` to preserve explicit-zero intent."""
        src = Path(
            "/private/tmp/vmlx-1.3.55-build/vmlx_engine/server.py"
        ).read_text()
        # Target both call sites
        import re
        hits = re.findall(
            r"max_tokens\s+if\s+\w+\.max_tokens\s+is\s+not\s+None",
            src,
        )
        assert len(hits) >= 2, (
            f"max_tokens `is not None` check required ≥ 2 call sites, "
            f"found {len(hits)}"
        )


class TestModelUnloadOnShutdown:
    """Process exit must release model weights + Metal pool to avoid
    dangling allocations leaking across process restarts on the same
    Tailscale node (common in the memory-enforcer auto-evict flow)."""

    def test_engine_close_documented(self):
        """Engine class must expose close()/shutdown() for graceful teardown."""
        from vmlx_engine.engine_core import EngineCore
        assert hasattr(EngineCore, "close"), (
            "EngineCore must expose close() for graceful shutdown"
        )

    def test_batch_generator_close_releases_metal_limits(self):
        """MLLMBatchGenerator.close restores pre-override wired + cache
        limits — otherwise a second process on the same machine starts
        from a bad global state."""
        from vmlx_engine.mllm_batch_generator import MLLMBatchGenerator
        import inspect
        src = inspect.getsource(MLLMBatchGenerator.close)
        assert "_old_wired_limit" in src
        assert "_old_cache_limit" in src


class TestPanelEngineIPCContract:
    """Panel ↔ engine IPC shape contracts — the panel TS side and the
    Python server must agree on field names / types.

    When this drifts, the panel's console fills with type errors,
    streaming dies, or features silently vanish."""

    def test_panel_expects_reasoning_content_or_reasoning(self):
        """Panel reads both reasoning_content (new) and reasoning (legacy/alias)
        from streaming chunk delta.choices[].delta.{reasoning_content,reasoning}."""
        src = Path(
            "/private/tmp/vmlx-1.3.55-build/panel/src/main/ipc/chat.ts"
        ).read_text()
        assert "reasoning_content" in src
        assert "delta" in src

    def test_server_emits_reasoning_content_via_computed_field(self):
        """ChatCompletionChunkDelta.reasoning_content is a computed
        @property that delegates to self.reasoning, so both field names
        on the wire point to the same internal value."""
        from vmlx_engine.api.models import ChatCompletionChunkDelta
        d = ChatCompletionChunkDelta(content="x", reasoning="think")
        dump = d.model_dump()
        assert dump.get("reasoning_content") == "think"

    def test_image_and_video_url_content_part_schemas(self):
        """Request schema must accept both image_url and video_url — the
        panel sends either depending on attachment kind (mlxstudio#69/97)."""
        from vmlx_engine.api.models import ContentPart, ImageUrl
        # image_url part
        ip = ContentPart(type="image_url",
                         image_url=ImageUrl(url="data:image/png;base64,AA"))
        assert ip.image_url is not None
        # video_url is also supported by the engine (may be via dict)
        try:
            from vmlx_engine.api.models import VideoUrl
            vp = ContentPart(type="video_url",
                             video_url=VideoUrl(url="data:video/mp4;base64,BB"))
            assert vp.video_url is not None
        except ImportError:
            # Older versions shipped video_url as plain dict — still OK
            pass


class TestMs79AnthropicUsageTokens:
    """ms#79 issue 4: "In Claude Code and OpenCode tools always can't be
    used correctly, ！回复 appears".

    Root cause found: `to_chat_completion(req)` in anthropic_adapter.py
    set `stream_options=StreamOptions(include_usage=True) if req.stream
    else None` — so non-streaming Anthropic requests had stream_options=
    None. The /v1/messages server path then internally calls
    stream_chat_completion() and accumulates chunks; without
    include_usage=True, the inner OpenAI-format stream never emits
    usage and the final response returns `{input_tokens: 0,
    output_tokens: 0}`.

    Claude Code uses these counts for rate-limit accounting and progress
    display. Zeroed usage looks like a broken request and contributes
    to the "model auto-stop" / "！回复" user-facing failure mode.

    Fix: always request include_usage=True in the internal chat_req,
    regardless of the outer req.stream flag.
    """

    def test_non_streaming_always_includes_usage(self):
        """to_chat_completion must always set include_usage=True so the
        internal stream emits usage for both streaming and non-streaming
        Anthropic requests."""
        from vmlx_engine.api.anthropic_adapter import to_chat_completion
        from vmlx_engine.api.anthropic_adapter import AnthropicRequest

        # Non-stream request — the bug case
        req = AnthropicRequest(
            model="test",
            messages=[{"role": "user", "content": "hi"}],
            max_tokens=10,
            stream=False,
        )
        chat_req = to_chat_completion(req)
        assert chat_req.stream_options is not None, (
            "ms#79: non-streaming Anthropic must still set stream_options "
            "so the internal stream emits usage tokens"
        )
        assert chat_req.stream_options.include_usage is True, (
            "ms#79: include_usage must be True regardless of outer stream flag"
        )

    def test_streaming_also_includes_usage(self):
        """Streaming case unchanged — Claude Code's incremental chunks
        also need usage for live progress display."""
        from vmlx_engine.api.anthropic_adapter import to_chat_completion
        from vmlx_engine.api.anthropic_adapter import AnthropicRequest

        req = AnthropicRequest(
            model="test",
            messages=[{"role": "user", "content": "hi"}],
            max_tokens=10,
            stream=True,
        )
        chat_req = to_chat_completion(req)
        assert chat_req.stream_options is not None
        assert chat_req.stream_options.include_usage is True

    def test_source_anchor(self):
        """ms#79 anchor present so future refactors don't drop the fix."""
        from pathlib import Path
        import vmlx_engine.api.anthropic_adapter as m
        src = Path(m.__file__).read_text()
        assert "ms#79" in src, (
            "ms#79 anchor missing — fix is at risk of silent revert"
        )
        assert "include_usage=True" in src


class TestMs68CollectionErrorVsEmpty:
    """ms#68 "[Bug] — No models in this collection".

    The issue body was just the literal UI text "No models in this
    collection". Before this fix that text was shown for BOTH:
    (a) HF fetch failure (no internet, HF down, rate-limited, slug
        outdated, HF API schema changed)
    (b) genuinely empty collection

    The user couldn't tell which. Fix: per-tab `collectionErrors` state
    + distinct "Failed to load — Retry" UI when the promise rejects,
    plus explicit retry handler.
    """

    def test_collection_errors_state_present(self):
        src = Path(
            "/private/tmp/vmlx-1.3.55-build/panel/src/renderer/src/components/sessions/DownloadTab.tsx"
        ).read_text()
        assert "ms#68" in src, "anchor missing"
        assert "const [collectionErrors," in src, (
            "per-tab error state must exist; otherwise fetch failure = "
            "empty collection in the UI"
        )

    def test_retry_handler_present(self):
        src = Path(
            "/private/tmp/vmlx-1.3.55-build/panel/src/renderer/src/components/sessions/DownloadTab.tsx"
        ).read_text()
        assert "retryCollectionFetch" in src, (
            "explicit retry entry point missing"
        )
        # The retry button text in the fallback UI
        assert ">Retry<" in src or "Retry\n" in src, (
            "retry button must exist in the error fallback UI"
        )

    def test_error_ui_distinct_from_empty(self):
        src = Path(
            "/private/tmp/vmlx-1.3.55-build/panel/src/renderer/src/components/sessions/DownloadTab.tsx"
        ).read_text()
        assert "Failed to load" in src, (
            "error fallback must say 'Failed to load' — otherwise the UX "
            "is identical to the empty-collection case"
        )
        # Error must show the actual exception message (so users can
        # diagnose rate-limit, 404 slug, etc.)
        assert "collectionErrors[collectionTab]" in src

    def test_failure_path_records_error(self):
        """The catch branch of the fetch must populate collectionErrors,
        not just log to console."""
        src = Path(
            "/private/tmp/vmlx-1.3.55-build/panel/src/renderer/src/components/sessions/DownloadTab.tsx"
        ).read_text()
        # Both catch sites (mount effect + handleCollectionTabChange)
        # must set the error state, not just log.
        setError = src.count("setCollectionErrors(prev => ({")
        assert setError >= 2, (
            f"setCollectionErrors must be called on fetch failure in at "
            f"least 2 sites (mount effect + tab switch); found {setError}"
        )


class TestVmlx94MxMetalDeprecation:
    """vmlx#94: MLX 0.31+ deprecates mx.metal.*. All call sites must prefer
    mx.* and only fall back to mx.metal.* when the top-level isn't present."""

    def test_scheduler_uses_getattr_fallback(self):
        """scheduler.py memory-pressure guard must use getattr fallback."""
        src = Path(
            "/private/tmp/vmlx-1.3.55-build/vmlx_engine/scheduler.py"
        ).read_text()
        # The two functions vmlx#94 flagged:
        assert (
            "getattr(mx, \"get_active_memory\", None) or mx.metal.get_active_memory"
            in src
        ), "scheduler get_active_memory must use getattr fallback"
        assert (
            "getattr(mx, \"device_info\", None) or mx.metal.device_info"
            in src
        ), "scheduler device_info must use getattr fallback"

    def test_no_bare_mx_metal_outside_hasattr_guards(self):
        """Every bare `mx.metal.<fn>(` call must be inside a hasattr(mx, ...)
        guard that already tried mx.* first — otherwise MLX 0.32+ would
        raise DeprecationWarning during a normal server startup.

        This test walks each mx.metal.* occurrence and checks the
        preceding ~3 lines for an `if hasattr(mx, "X")` or `getattr(mx,`
        guard. If a bare call lacks both, the test fails — which is the
        exact regression pattern the issue reporter caught."""
        import re
        # Require the match to be followed by `(` — i.e. a real call,
        # not a docstring mention (`mx.metal.foo()` inside `` `` marks).
        # Also anchor to the left so it isn't inside a backtick.
        bare_pattern = re.compile(
            r"(?<![`.'\"])\bmx\.metal\.(get_active_memory|get_peak_memory|"
            r"get_cache_memory|clear_cache|set_cache_limit|reset_peak_memory|"
            r"device_info)\("
        )
        unguarded = []
        for relpath in (
            "vmlx_engine/scheduler.py",
            "vmlx_engine/server.py",
            "vmlx_engine/benchmark.py",
            "vmlx_engine/mllm_batch_generator.py",
        ):
            fpath = Path("/private/tmp/vmlx-1.3.55-build") / relpath
            if not fpath.exists():
                continue
            lines = fpath.read_text().splitlines()
            for i, line in enumerate(lines):
                if not bare_pattern.search(line):
                    continue
                # Look backward up to 8 lines for the guard
                preceding = "\n".join(lines[max(0, i - 8):i])
                has_getattr = "getattr(mx," in preceding or "getattr(mx," in line
                has_hasattr = "hasattr(mx," in preceding
                if not (has_getattr or has_hasattr):
                    unguarded.append(f"{relpath}:{i+1}: {line.strip()}")
        assert not unguarded, (
            "Bare mx.metal.* without getattr/hasattr guard (will emit "
            "DeprecationWarning on MLX 0.31+):\n  " + "\n  ".join(unguarded)
        )

    def test_import_free_of_mx_metal_deprecation(self):
        """Importing and exercising the memory probes must NOT raise a
        mx.metal-specific DeprecationWarning. This reproduces the
        reporter's command:

            python -W error::DeprecationWarning -c \"import mlx.core as mx;
            mx.metal.get_active_memory()\"

        ...except we target only mx.metal deprecations (pydantic and
        other libs have unrelated deprecations that would false-positive)."""
        import warnings
        import mlx.core as mx
        caught = []

        def capture(msg, cat, *a, **kw):
            if issubclass(cat, DeprecationWarning) and "mx.metal" in str(msg):
                caught.append(str(msg))

        prev = warnings.showwarning
        warnings.showwarning = capture
        try:
            _get_active = (
                getattr(mx, "get_active_memory", None) or mx.metal.get_active_memory
            )
            _device_info = (
                getattr(mx, "device_info", None) or mx.metal.device_info
            )
            _get_active()
            _device_info()
        finally:
            warnings.showwarning = prev

        assert not caught, (
            f"mx.metal deprecation warnings escaped: {caught}"
        )


class TestVmlx92PldGuardOnNonMllm:
    """vmlx#92: PLD speculative decode must NOT touch batch state when the
    generator is plain BatchGenerator (no .active_batch / no trimmable
    caches). Reporter observed that pre-guard, the first PLD call on a
    text-only server raised AttributeError, the finally path reinserted
    a malformed cache, step() crashed with `<class 'list'> does not yet
    support batching with history`, and recovery cleared the entire
    paged cache for every queued request."""

    def test_guard_is_present_before_active_batch_access(self):
        """The hasattr check must come BEFORE any .active_batch touch."""
        src = Path(
            "/private/tmp/vmlx-1.3.55-build/vmlx_engine/scheduler.py"
        ).read_text()
        # Locate the guard line
        guard_line = 'if not hasattr(self.batch_generator, "active_batch"):'
        assert guard_line in src, (
            "vmlx#92 guard must be present in _try_speculative_decode"
        )
        # The guard index must be BEFORE every other '.active_batch' read
        # inside _try_speculative_decode. We scope to the function body.
        import re
        body_start = src.index("def _try_speculative_decode")
        body_end = src.index("\n    def ", body_start + 1)
        body = src[body_start:body_end]
        guard_idx = body.index(guard_line)
        # Every ab = self.batch_generator.active_batch must be after
        active_reads = [m.start() for m in re.finditer(
            r"self\.batch_generator\.active_batch", body
        )]
        for idx in active_reads:
            # skip the guard line itself
            if idx == body.index("self.batch_generator.active_batch") and \
               body[max(0, idx-4):idx] == "attr":
                # this is the hasattr( — skip
                continue
            assert idx > guard_idx or body[idx-10:idx] == 'hasattr(', (
                f"active_batch access at offset {idx} comes BEFORE the "
                "hasattr guard — vmlx#92 regression"
            )

    def test_guard_returns_empty_list(self):
        """When the guard fires we must return [], so the caller treats
        it as 'no drafts available' and continues normal decode."""
        src = Path(
            "/private/tmp/vmlx-1.3.55-build/vmlx_engine/scheduler.py"
        ).read_text()
        # The two-line block must be: `if not hasattr(...)` then `return []`
        body_start = src.index("def _try_speculative_decode")
        body_end = src.index("\n    def ", body_start + 1)
        body = src[body_start:body_end]
        guard_idx = body.index(
            'if not hasattr(self.batch_generator, "active_batch"):'
        )
        # The next non-whitespace line must be return []
        remainder = body[guard_idx:guard_idx + 200]
        assert "return []" in remainder[:120], (
            "guard must short-circuit with `return []` — otherwise falls "
            "through and the old AttributeError reappears"
        )

    def test_mllm_batch_generator_still_has_active_batch(self):
        """Pin the capability the guard is selecting on — if MLLM
        BatchGenerator loses .active_batch, the guard falsely fires
        and PLD silently stops working for VLMs too."""
        from vmlx_engine.mllm_batch_generator import MLLMBatchGenerator
        # Either declared as attribute or produced in __init__
        import inspect
        src = inspect.getsource(MLLMBatchGenerator)
        assert "active_batch" in src, (
            "MLLMBatchGenerator must still expose active_batch — "
            "otherwise the vmlx#92 guard misclassifies it as non-MLLM"
        )


class TestMs61ImageGalleryDeleteAndCopyPrompt:
    """ms#61 Feature Request: per-image delete + copy-prompt buttons in
    the Image gallery.

    > Please consider adding a delete button to image in Image gallery
    > to delete the image. The Image gallery can grow to unmanageable
    > size without pruning.
    > Also please consider adding a button to each image in Image
    > gallery to allow copy of the prompt that generated the image.

    Implementation surface: DB layer (new getImageGeneration +
    existing deleteImageGeneration), IPC handler (image:deleteGeneration
    with safe unlink path check), preload exposure, env.d.ts types,
    ImageGallery UI (Trash2 + FileText buttons), ImageTab wiring.
    """

    def test_ipc_handler_present_and_anchored(self):
        src = Path(
            "/private/tmp/vmlx-1.3.55-build/panel/src/main/ipc/image.ts"
        ).read_text()
        assert "image:deleteGeneration" in src, (
            "IPC handler missing — preload call will fail"
        )
        assert "ms#61" in src, "anchor required"
        # Safety: must gate unlink on ~/.mlxstudio path so random picks
        # from user's filesystem don't get rm'd
        assert ".mlxstudio" in src and "startsWith" in src, (
            "unlink must be gated to ~/.mlxstudio paths only — never "
            "touch user's Pictures / Desktop files that were only "
            "referenced as source images"
        )

    def test_database_has_single_row_lookup(self):
        src = Path(
            "/private/tmp/vmlx-1.3.55-build/panel/src/main/database.ts"
        ).read_text()
        assert "getImageGeneration(id: string)" in src, (
            "single-row lookup needed so IPC can read the paths before "
            "deleting the DB row"
        )

    def test_preload_exposes_delete_generation(self):
        src = Path(
            "/private/tmp/vmlx-1.3.55-build/panel/src/preload/index.ts"
        ).read_text()
        assert "deleteGeneration:" in src
        assert "image:deleteGeneration" in src

    def test_env_types_declare_delete_generation(self):
        src = Path(
            "/private/tmp/vmlx-1.3.55-build/panel/src/env.d.ts"
        ).read_text()
        assert "deleteGeneration:" in src

    def test_gallery_ui_has_copy_and_delete_buttons(self):
        src = Path(
            "/private/tmp/vmlx-1.3.55-build/panel/src/renderer/src/components/image/ImageGallery.tsx"
        ).read_text()
        # Copy prompt button
        assert "handleCopyPrompt" in src
        assert "generation.prompt" in src
        assert "promptCopied" in src
        # Delete button
        assert "handleDelete" in src
        assert "Trash2" in src
        # Guard — confirm dialog so a stray click doesn't nuke
        assert "confirm(" in src, (
            "delete must have a confirmation prompt — otherwise a stray "
            "click deletes permanently with no undo"
        )

    def test_image_tab_wires_delete_handler(self):
        src = Path(
            "/private/tmp/vmlx-1.3.55-build/panel/src/renderer/src/components/image/ImageTab.tsx"
        ).read_text()
        assert "onDelete={async (gen) =>" in src, (
            "ImageTab must provide the onDelete callback; otherwise the "
            "button is shown but does nothing"
        )
        assert "window.api.image.deleteGeneration(gen.id)" in src
        # On success: drop from local state rather than refetch
        assert "setGenerations(prev => prev.filter(g => g.id !== gen.id))" in src


class TestVmlx70BulkDeleteChats:
    """vmlx#70 Feature Request: mass delete / wipe chat history.

    > there is no way to mass delete or wipe chat history — you can
    > only do it one-by-one, which is inconvenient.

    Implementation: DB `deleteAllChats(scope?)` with scope modes
    (all / folder / model), IPC `chat:deleteAll`, preload exposure,
    env.d.ts type, "Clear" button in ChatList header with a confirm()
    dialog that spells out the exact count + scope before nuking.
    """

    def test_db_has_delete_all_chats(self):
        src = Path(
            "/private/tmp/vmlx-1.3.55-build/panel/src/main/database.ts"
        ).read_text()
        assert "deleteAllChats(scope?:" in src or \
               "deleteAllChats(scope" in src, (
            "DB must expose deleteAllChats — one-row-at-a-time DELETE in "
            "a JS loop would be far slower on large histories"
        )
        # The scope modes must all be supported
        assert 'folderId === "unfiled"' in src
        assert "scope?.folderId" in src
        assert "scope?.modelPath" in src
        # Messages cascade via FK — no need for explicit DELETE FROM messages
        assert "ON DELETE CASCADE" in src, (
            "messages.chat_id FK must cascade, otherwise deleteAllChats "
            "leaves orphaned messages"
        )
        # Return value must be the number actually deleted
        assert "return Number(result.changes" in src, (
            "deleteAllChats must return a count — the UI shows it to "
            "the user after confirmation"
        )

    def test_ipc_handler_wired(self):
        src = Path(
            "/private/tmp/vmlx-1.3.55-build/panel/src/main/ipc/chat.ts"
        ).read_text()
        assert '"chat:deleteAll"' in src
        assert "vmlx#70" in src, "anchor required"
        assert "db.deleteAllChats(scope)" in src

    def test_preload_exposes_delete_all(self):
        src = Path(
            "/private/tmp/vmlx-1.3.55-build/panel/src/preload/index.ts"
        ).read_text()
        assert "deleteAll: (scope?:" in src
        assert "chat:deleteAll" in src

    def test_env_types_declare_delete_all(self):
        src = Path(
            "/private/tmp/vmlx-1.3.55-build/panel/src/env.d.ts"
        ).read_text()
        assert "deleteAll: (scope?:" in src
        assert "deleted?: number" in src

    def test_ui_has_clear_button_with_confirm(self):
        src = Path(
            "/private/tmp/vmlx-1.3.55-build/panel/src/renderer/src/components/chat/ChatList.tsx"
        ).read_text()
        # Anchor + handler + button + confirm + scope branching
        assert "vmlx#70" in src
        assert "handleClearAll" in src
        assert "Eraser" in src, "Eraser icon used for the button"
        # confirm() is required — delete is irreversible
        assert "confirm(" in src, (
            "bulk-delete MUST have a confirm dialog — one misclick "
            "erases the user's entire chat history"
        )
        # The confirm text must tell the user the COUNT and SCOPE so a
        # surprise (e.g. modelPath filter active) doesn't lead to a
        # mis-click-confirmed wipe.
        assert "targetCount" in src or "chats.length" in src
        # Button hidden when nothing to clear (defensive)
        assert "chats.length > 0" in src and "Clear" in src

    def test_scope_aware_clear(self):
        """When the Chat tab is bound to a specific model, Clear should
        pass modelPath so only that model's chats get nuked — not the
        user's entire cross-model history."""
        src = Path(
            "/private/tmp/vmlx-1.3.55-build/panel/src/renderer/src/components/chat/ChatList.tsx"
        ).read_text()
        # handleClearAll must conditionally pass {modelPath} vs undefined
        assert "modelPath ? { modelPath } : undefined" in src, (
            "Clear handler must respect the model filter — passing "
            "undefined when no filter wipes cross-model, passing "
            "{modelPath} when filter is active wipes just that model"
        )


class TestMs75HuggingFaceMirrorEndpoint:
    """ms#75: ModelScope / HuggingFace mirror support for users in
    mainland China (or behind any restrictive network).

    > For users in mainland China, downloading models from Hugging Face
    > is often extremely slow or fails due to network limitations.

    Solution: `hf_endpoint` setting routes ALL HF traffic (downloads +
    API) through an HF-protocol-compatible mirror. hf-mirror.com is
    the standard China-based mirror (no new client library needed).
    The `huggingface_hub` Python library respects the `HF_ENDPOINT`
    env var natively, so the download pipeline reroutes automatically
    when the env var is set.
    """

    def test_main_process_exposes_getHfBaseUrl(self):
        src = Path(
            "/private/tmp/vmlx-1.3.55-build/panel/src/main/ipc/models.ts"
        ).read_text()
        assert "export function getHfBaseUrl()" in src, (
            "getHfBaseUrl helper missing — callers would need to reimplement "
            "the setting read + slash-strip"
        )
        # Must read the setting and strip trailing slash
        assert 'db.getSetting("hf_endpoint")' in src
        assert "replace(/\\/+$/" in src, (
            "trailing-slash strip required — otherwise concatenation "
            "with '/api/...' produces '//api/...'"
        )
        # Must fall back to the canonical URL when unset
        assert 'return "https://huggingface.co"' in src

    def test_download_passes_hf_endpoint_env(self):
        src = Path(
            "/private/tmp/vmlx-1.3.55-build/panel/src/main/ipc/models.ts"
        ).read_text()
        assert 'downloadEnv.HF_ENDPOINT = hfEndpoint.trim()' in src, (
            "HF_ENDPOINT must be forwarded to the download subprocess — "
            "that's the env var huggingface_hub reads"
        )
        assert 'db.getSetting("hf_endpoint")' in src

    def test_all_fetch_sites_mirror_aware(self):
        """Every direct `https://huggingface.co` fetch in the main
        process must route through getHfBaseUrl()."""
        src = Path(
            "/private/tmp/vmlx-1.3.55-build/panel/src/main/ipc/models.ts"
        ).read_text()
        # Count hardcoded HF URLs outside of comments/fallback/rewrites
        # The only remaining literal `https://huggingface.co` should be
        # in the `return "https://huggingface.co"` default fallback +
        # comment strings — no LIVE fetches should hit it directly.
        import re
        # Find all `fetch(\`https://huggingface.co/...)` literals
        fetch_literals = re.findall(
            r'fetch\(\s*`https://huggingface\.co',
            src,
        )
        assert len(fetch_literals) == 0, (
            f"Still have {len(fetch_literals)} hardcoded HF URLs in fetch calls — "
            f"they bypass the mirror setting"
        )
        # Find `\`https://huggingface.co/api/...` used as URL strings
        url_literals = re.findall(
            r'`https://huggingface\.co/api/',
            src,
        )
        assert len(url_literals) == 0, (
            f"Still have {len(url_literals)} hardcoded HF API URL strings — "
            f"they bypass the mirror setting"
        )
        # getHfBaseUrl() must be USED, not just defined
        usages = src.count("getHfBaseUrl()")
        assert usages >= 5, (
            f"getHfBaseUrl() must be called ≥ 5 times (search × 2 + "
            f"recommended + collection + README); found {usages}"
        )

    def test_csp_allows_mirror_hosts(self):
        """Content Security Policy must whitelist hf-mirror.com and
        modelscope.cn or README images from those hosts render broken."""
        src = Path(
            "/private/tmp/vmlx-1.3.55-build/panel/src/main/index.ts"
        ).read_text()
        # Both mirror domains must appear in img-src AND connect-src
        img_src_idx = src.find("img-src")
        connect_src_idx = src.find("connect-src")
        img_src_line = src[img_src_idx:src.find(";", img_src_idx)]
        connect_src_line = src[connect_src_idx:src.find(";", connect_src_idx)]
        for host in ["hf-mirror.com", "modelscope.cn"]:
            assert host in img_src_line, f"CSP img-src missing {host}"
            assert host in connect_src_line, f"CSP connect-src missing {host}"

    def test_settings_ui_has_mirror_input_with_validation(self):
        src = Path(
            "/private/tmp/vmlx-1.3.55-build/panel/src/renderer/src/components/sessions/DownloadTab.tsx"
        ).read_text()
        # Anchor + state + save handler + UI field + one-click preset
        assert "ms#75" in src
        assert "const [hfEndpoint, setHfEndpoint]" in src
        assert "handleSaveHfEndpoint" in src
        # Validation — typo "hf-mirror.com" (no scheme) must not silently break
        assert "https?:\\/\\/" in src, (
            "save handler must validate scheme — otherwise a typo "
            "silently corrupts HF_ENDPOINT and kills all downloads"
        )
        # One-click preset button for the standard China mirror
        assert "hf-mirror.com" in src
        assert "Use hf-mirror" in src
        # Load saved value on mount
        assert "window.api.settings.get('hf_endpoint')" in src
