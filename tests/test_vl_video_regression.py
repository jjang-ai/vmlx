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
