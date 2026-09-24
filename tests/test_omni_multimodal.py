import json
import threading
from concurrent.futures import Future, ThreadPoolExecutor

import pytest
from fastapi import HTTPException

from vmlx_engine.omni_multimodal import (
    _OmniIncrementalRailSplitter,
    OmniMultimodalDispatcher,
    _build_omni_turn_prompt_with_thinking,
    dispatch_omni_chat_completion,
    _extract_parts,
    _hash_user_texts,
    _user_turn_signatures,
    is_omni_multimodal_bundle,
    omni_multimodal_component_status,
    request_has_multimodal,
    request_modalities,
)


class _FakeTokenizer:
    def __init__(self):
        self.calls = []

    def apply_chat_template(self, messages, **kwargs):
        self.calls.append((messages, kwargs))
        rail = "<think></think>" if kwargs.get("enable_thinking") is False else "<think>\n"
        return f"rendered:{messages[-1]['content']}:{rail}"


def _write_omni_bundle(
    tmp_path,
    *,
    radio: bool = True,
    parakeet: bool = True,
    projector: bool = True,
    video_preprocessor: bool = True,
    model_type: str = "nemotron_h",
    sound_model_type: str = "parakeet",
):
    tmp_path.mkdir(parents=True, exist_ok=True)
    (tmp_path / "config.json").write_text(json.dumps({"model_type": model_type}))
    (tmp_path / "config_omni.json").write_text(
        json.dumps({"sound_config": {"model_type": sound_model_type}})
    )
    (tmp_path / "configuration_radio.py").write_text("# radio config placeholder\n")
    import numpy as np
    from safetensors.numpy import save_file
    shapes = {}
    if radio:
        shapes["vision_model.radio_model.model.blocks.0.attn.qkv.weight"] = (12, 4)
        shapes["vision_model.radio_model.model.patch_generator.embedder.weight"] = (4, 3)
    if parakeet:
        shapes["sound_encoder.encoder.layers.0.conv.depthwise_conv.weight"] = (3, 3)
    if projector:
        shapes.update({"mlp1.0.weight": (4,), "mlp1.1.weight": (8, 4), "mlp1.3.weight": (6, 8),
                       "sound_projection.norm.weight": (3,), "sound_projection.linear1.weight": (5, 3),
                       "sound_projection.linear2.weight": (6, 5)})
    save_file({k: np.zeros(shape, dtype=np.float16) for k,shape in shapes.items()}, str(tmp_path / "model.safetensors"))
    (tmp_path / "model.safetensors.index.json").write_text(
        json.dumps({"weight_map": {k: "model.safetensors" for k in shapes}})
    )
    if video_preprocessor:
        (tmp_path / "video_preprocessor_config.json").write_text(
            json.dumps({"video_processor_type": "FakeVideoProcessor"})
        )
    return tmp_path


def test_omni_component_status_requires_radio_parakeet_and_projector(tmp_path):
    bundle = _write_omni_bundle(tmp_path / "omni")

    status = omni_multimodal_component_status(bundle)

    assert status["bundle_compatible"] is True
    assert status["config_model_type"] == "nemotron_h"
    assert status["sound_config_model_type"] == "parakeet"
    assert status["has_radio_weights"] is True
    assert status["has_parakeet_weights"] is True
    assert status["has_media_projector"] is True
    assert status["has_video_preprocessor_config"] is True
    assert status["video_bridge_supported"] is True
    assert status["modalities"] == ["text", "audio", "image", "video"]
    assert status["missing"] == []
    assert is_omni_multimodal_bundle(bundle) is True


def test_omni_component_status_accepts_nemotron_h_v2_spelling(tmp_path):
    # nemotron_h_v2 is the same hybrid architecture; a v2 bundle carrying the
    # full RADIO/Parakeet sidecar set must not be rejected on the model_type
    # alias alone (the requirements row used to compare against the literal
    # "nemotron_h").
    bundle = _write_omni_bundle(tmp_path / "omni", model_type="nemotron_h_v2")

    status = omni_multimodal_component_status(bundle)

    assert status["config_model_type"] == "nemotron_h_v2"
    assert status["bundle_compatible"] is True
    assert status["missing"] == []
    assert is_omni_multimodal_bundle(bundle) is True


def test_omni_component_status_omits_video_without_runtime_bridge(tmp_path):
    bundle = _write_omni_bundle(tmp_path / "omni", video_preprocessor=False)

    status = omni_multimodal_component_status(bundle)

    assert status["bundle_compatible"] is True
    assert status["has_radio_weights"] is True
    assert status["has_video_preprocessor_config"] is False
    assert status["video_bridge_supported"] is False
    assert status["modalities"] == ["text", "audio", "image", "video"]
    assert status["missing"] == []
    assert is_omni_multimodal_bundle(bundle) is True


def test_omni_component_status_rejects_vision_only_bundle(tmp_path):
    bundle = _write_omni_bundle(tmp_path / "omni", parakeet=False)

    status = omni_multimodal_component_status(bundle)

    assert status["bundle_compatible"] is False
    assert status["has_radio_weights"] is True
    assert status["has_parakeet_weights"] is False
    assert "parakeet weights" in status["missing"]
    assert is_omni_multimodal_bundle(bundle) is False


def test_omni_prompt_builder_forwards_enable_thinking_false_on_first_turn():
    tok = _FakeTokenizer()

    prompt = _build_omni_turn_prompt_with_thinking(
        tok,
        "Describe the image directly.",
        n_image_tokens=2,
        is_first=True,
        enable_thinking=False,
    )

    assert [m["role"] for m in tok.calls[0][0]] == ["user"]
    assert "<img><image><image></img>" in tok.calls[0][0][0]["content"]
    assert "Answer directly" not in tok.calls[0][0][0]["content"]
    assert tok.calls[0][1]["enable_thinking"] is False
    assert prompt.endswith("<think></think>")


def test_omni_prompt_builder_omits_enable_thinking_when_request_is_auto():
    tok = _FakeTokenizer()

    _build_omni_turn_prompt_with_thinking(
        tok,
        "Describe the image naturally.",
        n_image_tokens=1,
        is_first=True,
        enable_thinking=None,
    )

    assert [m["role"] for m in tok.calls[0][0]] == ["user"]
    assert "Answer directly" not in tok.calls[0][0][0]["content"]
    assert "enable_thinking" not in tok.calls[0][1]


def test_omni_prompt_builder_forwards_enable_thinking_false_on_followup_turn():
    tok = _FakeTokenizer()

    _build_omni_turn_prompt_with_thinking(
        tok,
        "Now answer directly.",
        is_first=False,
        enable_thinking=False,
    )

    assert tok.calls[0][1]["enable_thinking"] is False
    assert "Answer directly" not in tok.calls[0][0][-1]["content"]


def test_omni_prompt_builder_uses_image_placeholders_for_video_tokens():
    tok = _FakeTokenizer()

    _build_omni_turn_prompt_with_thinking(
        tok,
        "Describe the video.",
        n_video_tokens=3,
        is_first=True,
        enable_thinking=False,
    )

    content = tok.calls[0][0][-1]["content"]
    assert "<img><image><image><image></img>" in content
    assert "<video>" not in content


def test_omni_dispatcher_sets_thinking_flag_on_first_session_turn(tmp_path):
    class _FakeSession:
        def __init__(self):
            self.reset_count = 0

        def reset(self):
            self.reset_count += 1

        def turn(self, **kwargs):
            return "direct answer"

    dispatcher = OmniMultimodalDispatcher.__new__(OmniMultimodalDispatcher)
    dispatcher.bundle_path = "/fake"
    dispatcher._backend = "stage1"
    dispatcher._session = _FakeSession()
    dispatcher._lock = __import__("threading").Lock()
    dispatcher._last_signature = None
    dispatcher._scratch_dir = tmp_path

    dispatcher.chat(
        [{"role": "user", "content": "Describe directly."}],
        enable_thinking=False,
    )

    assert dispatcher._session._vmlx_enable_thinking is False


def test_omni_stage2_dispatcher_forwards_reasoning_and_decode_callback(tmp_path):
    captured = {}

    class _FakeSession:
        _last_prompt_tokens = 7
        _last_completion_tokens = 3
        _last_finish_reason = "stop"

        def reset(self):
            pass

        def turn(self, **kwargs):
            captured.update(kwargs)
            return "visible"

    dispatcher = OmniMultimodalDispatcher.__new__(OmniMultimodalDispatcher)
    dispatcher.bundle_path = "/fake"
    dispatcher._backend = "stage2"
    dispatcher._session = _FakeSession()
    dispatcher._lock = __import__("threading").Lock()
    dispatcher._last_signature = None
    dispatcher._scratch_dir = tmp_path
    callback = lambda token_id, text: None

    result = dispatcher.chat(
        [{"role": "user", "content": "Describe directly."}],
        enable_thinking=False,
        token_callback=callback,
    )

    assert captured["enable_thinking"] is False
    assert captured["token_callback"] is callback
    assert result["prompt_tokens"] == 7
    assert result["completion_tokens"] == 3
    assert result["finish_reason"] == "stop"


def test_omni_incremental_splitter_handles_fragmented_markers():
    splitter = _OmniIncrementalRailSplitter(explicit_thinking_off=False)
    events = []
    for delta in ("<thi", "nk>private", " work</thi", "nk>visible", " answer"):
        events.extend(splitter.feed(delta))
    events.extend(splitter.feed("", final=True))

    reasoning = "".join(text for rail, text in events if rail == "reasoning")
    content = "".join(text for rail, text in events if rail == "content")
    assert reasoning == "private work"
    assert content == "visible answer"
    assert "<think>" not in reasoning + content
    assert "</think>" not in reasoning + content


@pytest.mark.asyncio
async def test_omni_stream_emits_generation_time_reasoning_content_and_usage(
    tmp_path, monkeypatch
):
    bundle = _write_omni_bundle(tmp_path / "omni")
    captured = {}

    class _Dispatcher:
        persist_calls = 0

        def chat(self, *, token_callback, **kwargs):
            captured.update(kwargs)
            for token_id, delta in enumerate(
                ("<think>", "private", "</think>", "visible", " answer"),
                start=1,
            ):
                token_callback(token_id, delta)
            return {
                "content": "<think>private</think>visible answer",
                "n_images": 1,
                "has_audio": False,
                "has_video": False,
                "prompt_tokens": 23,
                "completion_tokens": 5,
                "finish_reason": "stop",
            }

        def reset(self):
            pass

        def finish_request_cache(self):
            self.persist_calls += 1
            return None

        def submit(self, fn, /, *args, **kwargs):
            future = Future()
            try:
                future.set_result(fn(*args, **kwargs))
            except BaseException as exc:
                future.set_exception(exc)
            return future

    dispatcher = _Dispatcher()
    monkeypatch.setattr(
        OmniMultimodalDispatcher,
        "get",
        classmethod(lambda cls, path, **kwargs: dispatcher),
    )

    class _Request:
        model = "omni"
        stream = True
        messages = [{
            "role": "user",
            "content": [
                {"type": "text", "text": "Describe."},
                {
                    "type": "image_url",
                    "image_url": {"url": "data:image/png;base64,AAAA"},
                },
            ],
        }]
        max_tokens = 8
        max_completion_tokens = None
        temperature = 0
        top_p = 1
        chat_template_kwargs = {}
        enable_thinking = True
        skip_prefix_cache = True
        cache_salt = "isolated-request"

    response = await dispatch_omni_chat_completion(
        _Request(),
        str(bundle),
        effective_max_tokens=16_384,
        effective_temperature=0.6,
        effective_top_p=0.95,
    )
    payloads = []
    async for raw in response.body_iterator:
        for line in str(raw).splitlines():
            if line.startswith("data: ") and line != "data: [DONE]":
                payloads.append(json.loads(line[6:]))

    reasoning = "".join(
        (payload.get("choices") or [{}])[0].get("delta", {}).get(
            "reasoning_content", ""
        )
        for payload in payloads
    )
    content = "".join(
        (payload.get("choices") or [{}])[0].get("delta", {}).get("content", "")
        for payload in payloads
    )
    terminal = next(payload for payload in payloads if payload.get("usage"))

    assert reasoning == "private"
    assert content == "visible answer"
    assert terminal["choices"][0]["finish_reason"] == "stop"
    assert terminal["usage"] == {
        "prompt_tokens": 23,
        "completion_tokens": 5,
        "total_tokens": 28,
    }
    # Both bypass spellings above prohibit publication as well as reuse.
    assert dispatcher.persist_calls == 0
    assert captured["max_tokens"] == 16_384
    assert captured["temperature"] == 0.6
    assert captured["top_p"] == 0.95

    assert captured["force_reset"] is True
    assert captured["cache_salt"] == "isolated-request"


def test_omni_dispatcher_uses_one_persistent_native_runtime_owner_thread():
    dispatcher = OmniMultimodalDispatcher.__new__(OmniMultimodalDispatcher)
    dispatcher._executor = ThreadPoolExecutor(
        max_workers=1,
        thread_name_prefix="test-omni-owner",
    )
    try:
        first = dispatcher.submit(threading.get_ident).result(timeout=2)
        second = dispatcher.submit(threading.get_ident).result(timeout=2)
    finally:
        dispatcher._executor.shutdown(wait=True)

    assert first == second
    assert first != threading.get_ident()


def test_omni_session_l2_roundtrips_the_NATIVE_representation(tmp_path):
    """The session snapshot must store what the live cache holds, nothing more.

    This test used to assert the opposite -- that the attention slots came
    back as QuantizedKVCache at 4 bits. That was an IMPOSED codec: the Omni
    backbone's own make_cache returns a plain float KVCache for '*' blocks, so
    nothing about the model asked for q4. Worse, the restore assigns the
    loaded objects straight back into the live session and mlx_lm has no
    dequant API, so a restored session decoded through q4 attention and
    appended q4 keys from then on -- permanently, and only on warm turns.
    A natively quantized cache (DSV4 pool state) is already quantized when it
    arrives here and still round-trips unchanged.
    """
    import mlx.core as mx
    from mlx_lm.models.cache import (
        ArraysCache,
        KVCache,
        QuantizedKVCache,
        load_prompt_cache,
    )
    from types import SimpleNamespace

    attention = KVCache()
    attention.update_and_fetch(
        mx.ones((1, 2, 3, 64)),
        mx.ones((1, 2, 3, 64)) * 2,
    )
    ssm = ArraysCache(2)
    ssm[0] = mx.ones((1, 4, 8))
    ssm[1] = mx.ones((1, 3, 5))
    backbone = SimpleNamespace(
        layers=[SimpleNamespace(block_type="*"), SimpleNamespace(block_type="M")]
    )

    dispatcher = OmniMultimodalDispatcher.__new__(OmniMultimodalDispatcher)
    dispatcher._disk_cache_enabled = True
    dispatcher._backend = "stage1"
    dispatcher._last_signature = "media-prefix-a"
    dispatcher._session_l2_fingerprint = "bundle-a"
    dispatcher._session_l2_path = tmp_path / "latest.safetensors"
    dispatcher._session_l2_policy = {"root": str(tmp_path), "max_size_bytes": 10000000}
    dispatcher._session_l2_store = None
    dispatcher._session_l2_stats = {
        "stores": 0,
        "hits": 0,
        "misses": 0,
        "last_store_seconds": None,
        "last_restore_seconds": None,
        "last_error": None,
    }
    dispatcher._session = SimpleNamespace(
        mlx_model=SimpleNamespace(backbone=backbone),
        _cache=[attention, ssm],
        _history_text=[{"role": "user", "content": "remember blue"}],
    )

    assert dispatcher._persist_session_snapshot() is True
    stored, metadata = load_prompt_cache(
        str(dispatcher._session_l2_path), return_metadata=True
    )
    assert isinstance(stored[0], KVCache), (
        "attention slot came back as %s -- a codec was imposed on a natively "
        "full-precision cache" % type(stored[0]).__name__
    )
    assert not isinstance(stored[0], QuantizedKVCache)
    assert isinstance(stored[1], ArraysCache)
    assert metadata["signature"] == "media-prefix-a"

    restored = OmniMultimodalDispatcher.__new__(OmniMultimodalDispatcher)
    restored._disk_cache_enabled = True
    restored._backend = "stage1"
    restored._last_signature = None
    restored._session_l2_fingerprint = "bundle-a"
    restored._session_l2_path = dispatcher._session_l2_path
    restored._session_l2_policy = dispatcher._session_l2_policy
    restored._session_l2_store = None
    restored._session_l2_stats = {
        "stores": 0,
        "hits": 0,
        "misses": 0,
        "last_store_seconds": None,
        "last_restore_seconds": None,
        "last_error": None,
    }
    restored._session = SimpleNamespace(
        mlx_model=SimpleNamespace(backbone=backbone),
        _cache=None,
        _history_text=[],
    )

    assert restored._try_restore_session_snapshot("media-prefix-a") is True
    assert isinstance(restored._session._cache[0], KVCache)
    assert not isinstance(restored._session._cache[0], QuantizedKVCache), (
        "a restored session must not continue decoding through a codec the "
        "model never used"
    )
    assert isinstance(restored._session._cache[1], ArraysCache)
    assert restored._session._history_text == [
        {"role": "user", "content": "remember blue"}
    ]
    assert restored._session_l2_stats["hits"] == 1


def test_omni_session_l2_rejects_a_different_media_prefix(tmp_path):
    import mlx.core as mx
    from mlx_lm.models.cache import ArraysCache, save_prompt_cache
    from types import SimpleNamespace

    ssm = ArraysCache(1)
    ssm[0] = mx.ones((1, 2, 3))

    dispatcher = OmniMultimodalDispatcher.__new__(OmniMultimodalDispatcher)
    dispatcher._disk_cache_enabled = True
    dispatcher._backend = "stage1"
    dispatcher._last_signature = None
    dispatcher._session_l2_fingerprint = "bundle-a"
    dispatcher._session_l2_path = tmp_path / "latest.safetensors"
    dispatcher._session_l2_policy = {"root": str(tmp_path), "max_size_bytes": 10000000}
    dispatcher._session_l2_store = None
    dispatcher._session_l2_stats = {
        "stores": 0,
        "hits": 0,
        "misses": 0,
        "last_store_seconds": None,
        "last_restore_seconds": None,
        "last_error": None,
    }
    dispatcher._session = SimpleNamespace(
        mlx_model=SimpleNamespace(
            backbone=SimpleNamespace(layers=[SimpleNamespace(block_type="M")])
        ),
        _cache=None,
        _history_text=[],
    )
    save_prompt_cache(
        str(dispatcher._session_l2_path),
        [ssm],
        {
            "schema": "nemotron_omni_session_v1",
            "bundle_fingerprint": "bundle-a",
            "signature": "orange-media-prefix",
            "history_json": "[]",
        },
    )

    assert dispatcher._try_restore_session_snapshot("blue-media-prefix") is False
    assert dispatcher._session_l2_stats["hits"] == 0
    assert dispatcher._session_l2_stats["misses"] == 1


def test_omni_extracts_input_audio_shape_emitted_by_panel(tmp_path):
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "Transcribe the sound."},
                {
                    "type": "input_audio",
                    "input_audio": {"data": "UklGRg==", "format": "wav"},
                },
            ],
        }
    ]

    assert request_has_multimodal(messages) is True
    assert request_modalities(messages) == {"audio"}
    text, images, audio, video = _extract_parts(messages, tmp_path)

    assert text == "Transcribe the sound."
    assert images == []
    assert video is None
    assert audio is not None
    assert audio.suffix == ".wav"
    assert audio.read_bytes() == b"RIFF"


def test_omni_extracts_audio_url_data_shape_from_adapters(tmp_path):
    """The Ollama and Anthropic adapters emit audio as {"type": "audio_url"}.
    Before audio_url was recognized, request_has_multimodal returned False,
    the request skipped omni dispatch, and audio was silently answered
    text-only. It must now round-trip like input_audio."""
    import base64

    data_url = "data:audio/wav;base64," + base64.b64encode(b"RIFFxx").decode()
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "Transcribe the sound."},
                {"type": "audio_url", "audio_url": {"url": data_url}},
            ],
        }
    ]

    assert request_has_multimodal(messages) is True
    assert request_modalities(messages) == {"audio"}
    text, images, audio, video = _extract_parts(messages, tmp_path)
    assert text == "Transcribe the sound."
    assert images == []
    assert video is None
    assert audio is not None
    assert audio.read_bytes() == b"RIFFxx"


def test_omni_backend_reads_both_env_spellings(monkeypatch):
    """--omni-backend used to set VMLINUX_OMNI_BACKEND while the picker read
    only VMLX_OMNI_BACKEND, so the flag was inert. Both spellings now work."""
    from vmlx_engine.omni_multimodal import OmniMultimodalDispatcher

    monkeypatch.delenv("VMLX_OMNI_BACKEND", raising=False)
    monkeypatch.delenv("VMLINUX_OMNI_BACKEND", raising=False)
    assert OmniMultimodalDispatcher._pick_backend() == "stage1"

    monkeypatch.setenv("VMLINUX_OMNI_BACKEND", "stage2")
    assert OmniMultimodalDispatcher._pick_backend() == "stage2"

    monkeypatch.delenv("VMLINUX_OMNI_BACKEND")
    monkeypatch.setenv("VMLX_OMNI_BACKEND", "mlx")
    assert OmniMultimodalDispatcher._pick_backend() == "stage2"


def test_omni_conversation_signature_salts_audio_bytes():
    def messages(audio_data):
        return [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "Remember the marker."},
                    {
                        "type": "input_audio",
                        "input_audio": {"data": audio_data, "format": "wav"},
                    },
                ],
            }
        ]

    orange = _user_turn_signatures(messages("T1JBTkdF"))
    orange_again = _user_turn_signatures(messages("T1JBTkdF"))
    blue = _user_turn_signatures(messages("QkxVRQ=="))

    assert orange == orange_again
    assert _hash_user_texts(orange) == _hash_user_texts(orange_again)
    assert orange != blue
    assert _hash_user_texts(orange) != _hash_user_texts(blue)


def test_omni_dispatcher_resets_when_replayed_prefix_media_changes(tmp_path, monkeypatch):
    class _Session:
        def __init__(self):
            self.reset_count = 0
            self.turns = []
            self._last_prompt_tokens = 1
            self._last_completion_tokens = 1
            self._last_finish_reason = "stop"

        def reset(self):
            self.reset_count += 1

        def turn(self, **kwargs):
            self.turns.append(kwargs)
            return "READY"

    dispatcher = OmniMultimodalDispatcher.__new__(OmniMultimodalDispatcher)
    dispatcher.bundle_path = str(tmp_path)
    dispatcher._session = _Session()
    dispatcher._lock = threading.Lock()
    dispatcher._last_signature = None
    dispatcher._scratch_dir = tmp_path
    dispatcher._backend = "stage1"

    def first_turn(audio_data):
        return [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "Remember the marker."},
                    {
                        "type": "input_audio",
                        "input_audio": {"data": audio_data, "format": "wav"},
                    },
                ],
            }
        ]

    dispatcher.chat(first_turn("T1JBTkdF"))
    assert dispatcher._session.reset_count == 1

    changed_media_history = first_turn("QkxVRQ==") + [
        {"role": "assistant", "content": "READY"},
        {"role": "user", "content": "Repeat the marker."},
    ]
    # This fixture has no encoders. Capture the full-history handoff while
    # preserving the original changed-media extraction assertion below.
    replayed = []
    def replay(session, messages, **kwargs):
        replayed.append(messages)
        text, images, audio, video = _extract_parts(
            messages, tmp_path, rehydrate_history_media=True,
        )
        return session.turn(text=text, images=images, audio=audio, video=video)
    monkeypatch.setattr("vmlx_engine.omni_multimodal._run_omni_full_history", replay)
    dispatcher.chat(changed_media_history)

    assert replayed == [changed_media_history]
    assert dispatcher._session.reset_count == 2
    assert dispatcher._session.turns[-1]["audio"] is not None
    assert dispatcher._session.turns[-1]["audio"].read_bytes() == b"BLUE"


def test_request_modalities_detects_image_audio_video():
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "Use all media."},
                {"type": "image_url", "image_url": {"url": "data:image/png;base64,AAAA"}},
                {"type": "input_audio", "input_audio": {"data": "UklGRg==", "format": "wav"}},
                {"type": "video_url", "video_url": {"url": "data:video/mp4;base64,AAAA"}},
            ],
        }
    ]

    assert request_has_multimodal(messages) is True
    assert request_modalities(messages) == {"image", "audio", "video"}


@pytest.mark.asyncio
async def test_omni_dispatch_rejects_unsupported_video_before_session_load(tmp_path):
    bundle = _write_omni_bundle(tmp_path / "omni", video_preprocessor=False)
    OmniMultimodalDispatcher._instance = None

    class _Request:
        model = "omni"
        stream = False
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "Describe this video."},
                    {"type": "video_url", "video_url": {"url": "data:video/mp4;base64,AAAA"}},
                ],
            }
        ]
        max_tokens = 8
        max_completion_tokens = None
        temperature = 0
        top_p = 1
        chat_template_kwargs = {}
        enable_thinking = False

    # Video is no longer rejected before session load: omni radio/vision
    # bundles now advertise a frame-fallback video path, so the pre-load
    # modality gate no longer returns 400 for a video-present request. Dispatch
    # proceeds past the gate and only fails later at load time (500) because
    # this fake tmp bundle has no loadable model. The contract under test is
    # that there is NO 400 reject-before-load for video.
    with pytest.raises(HTTPException) as exc:
        await dispatch_omni_chat_completion(_Request(), str(bundle))

    assert exc.value.status_code != 400
    assert exc.value.status_code == 500


class TestOmniSplitMatchesStreaming:
    """Non-stream must split the reply the same way the stream splitter does.

    MEASURED on the live Omni bundle (identical image request, 243 chars of
    thinking, cut by max_tokens):
        STREAM      content_len=0   reasoning_len=243   (correct)
        NON-STREAM  content_len=243 reasoning_len=0     (raw thinking rendered
                                                         as the visible answer)
    The Omni template OPENS the thinking rail in the prompt, so a reply that
    never emits </think> is still entirely inside that rail.
    _OmniIncrementalRailSplitter defaults to mode="reasoning" and flushes an
    unclosed tail as reasoning; _split_omni_reply fell through to
    "everything is content".
    """

    def test_truncated_prompt_opened_rail_is_reasoning_not_content(self):
        from vmlx_engine.omni_multimodal import _split_omni_reply

        reasoning, content = _split_omni_reply(
            "Thinking Process:\n\n1. Analyze the request", explicit_thinking_off=False
        )
        assert content == ""
        assert reasoning == "Thinking Process:\n\n1. Analyze the request"

    def test_unclosed_self_opened_rail_splits_and_hides_the_marker(self):
        from vmlx_engine.omni_multimodal import _split_omni_reply

        reasoning, content = _split_omni_reply(
            "visible so far<think>plan cut off here", explicit_thinking_off=False
        )
        assert content == "visible so far"
        assert reasoning == "plan cut off here"
        assert "<think" not in content

    def test_completed_rails_are_unchanged(self):
        from vmlx_engine.omni_multimodal import _split_omni_reply

        assert _split_omni_reply(
            "private plan</think>The image is blue.", explicit_thinking_off=False
        ) == ("private plan", "The image is blue.")
        assert _split_omni_reply(
            "pre<think>plan</think>The image is blue.", explicit_thinking_off=False
        ) == ("plan", "preThe image is blue.")

    def test_thinking_off_keeps_everything_visible(self):
        from vmlx_engine.omni_multimodal import _split_omni_reply

        reasoning, content = _split_omni_reply(
            "The image is blue.", explicit_thinking_off=True
        )
        assert reasoning is None
        assert content == "The image is blue."

    def test_no_shape_leaks_a_marker_into_content(self):
        from vmlx_engine.omni_multimodal import _split_omni_reply

        for raw, off in [
            ("Thinking Process:\n\n1. Analyze", False),
            ("private plan</think>Answer.", False),
            ("pre<think>plan</think>Answer.", False),
            ("visible<think>cut", False),
            ("<think>cut", False),
            ("Answer.", True),
            ("<think>plan</think>Answer.", True),
        ]:
            _, content = _split_omni_reply(raw, explicit_thinking_off=off)
            assert "<think" not in (content or "")
            assert "</think" not in (content or "")
