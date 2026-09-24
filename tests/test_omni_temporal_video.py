"""Weight-backed native video must not be converted into a still contact sheet."""
import threading
from pathlib import Path

from vmlx_engine import omni_multimodal as omni


class NativeSession:
    def reset(self):
        pass

    def turn(self, **kwargs):
        self.last_turn = kwargs
        return "answer"


def dispatcher(tmp_path, temporal):
    d = omni.OmniMultimodalDispatcher.__new__(omni.OmniMultimodalDispatcher)
    d._session = NativeSession()
    d._backend = "stage1"
    d._lock = threading.Lock()
    d._last_signature = None
    d._scratch_dir = tmp_path
    d._native_video_spec = {"temporal_patch_size": 2} if temporal else None
    return d


def test_temporal_bundle_routes_clip_to_native_video_encoder(tmp_path, monkeypatch):
    clip = tmp_path / "clip.mp4"
    clip.write_bytes(b"video")
    sheets = []
    monkeypatch.setattr(omni, "_extract_omni_video_frames", lambda *a, **kw: sheets.append(a) or [tmp_path / "sheet.jpg"])
    d = dispatcher(tmp_path, True)
    d.chat([{"role": "user", "content": [{"type": "video_url", "video_url": {"url": str(clip)}}]}], enable_thinking=False)
    assert d._session.last_turn['video'] == clip
    assert not d._session.last_turn['images']
    assert sheets == []


def test_native_temporal_state_cannot_match_legacy_contact_sheet_state(tmp_path, monkeypatch):
    clip = tmp_path / "clip.mp4"
    clip.write_bytes(b"video")
    monkeypatch.setattr(omni, "_extract_omni_video_frames", lambda *a, **kw: [tmp_path / "sheet.jpg"])
    messages = [{"role": "user", "content": [{"type": "video_url", "video_url": {"url": str(clip)}}]}]
    native, legacy = dispatcher(tmp_path, True), dispatcher(tmp_path, False)
    native.chat(messages, enable_thinking=False)
    legacy.chat(messages, enable_thinking=False)
    assert native._last_signature != legacy._last_signature


import json
import numpy as np
import pytest
from types import SimpleNamespace
from vmlx_engine.omni_native_video import temporal_video_spec, encode_temporal_video, _PREFIX


def projection_bundle(root, *, image_shape=(4, 3), video_shape=(4, 6), declared=2,
                      include_video=True, video_dtype=np.float16):
    from safetensors.numpy import save_file
    tensors = {_PREFIX + "embedder.weight": np.zeros(image_shape, dtype=np.float16)}
    if include_video:
        tensors[_PREFIX + "video_embedder.weight"] = np.zeros(video_shape, dtype=video_dtype)
    save_file(tensors, str(root / "model-00001-of-00001.safetensors"))
    (root / "model.safetensors.index.json").write_text(json.dumps({"weight_map": {
        key: "model-00001-of-00001.safetensors" for key in tensors}}))
    (root / "config_omni.json").write_text(json.dumps({"video_temporal_patch_size": declared}))


def test_temporal_capability_reads_real_shard_shapes(tmp_path):
    projection_bundle(tmp_path)
    assert temporal_video_spec(tmp_path) == {
        "temporal_patch_size": 2, "weight": _PREFIX + "video_embedder.weight",
        "shape": [4, 6], "dtype": "F16"}


def test_config_alone_does_not_establish_temporal_capability(tmp_path):
    projection_bundle(tmp_path, include_video=False)
    assert temporal_video_spec(tmp_path) is None


@pytest.mark.parametrize("overrides", [
    {"declared": 3}, {"video_shape": (5, 6)}, {"video_shape": (4, 7)},
    {"video_dtype": np.int32},
])
def test_temporal_projection_disagreement_fails_closed(tmp_path, overrides):
    projection_bundle(tmp_path, **overrides)
    with pytest.raises(ValueError):
        temporal_video_spec(tmp_path)


@pytest.mark.parametrize("fault", ["missing_shard", "missing_tensor", "truncated"])
def test_index_cannot_substitute_for_real_temporal_tensor(tmp_path, fault):
    projection_bundle(tmp_path)
    shard = tmp_path / "model-00001-of-00001.safetensors"
    if fault == "missing_shard":
        shard.unlink()
    elif fault == "truncated":
        shard.write_bytes(b"bad")
    else:
        from safetensors.numpy import save_file
        save_file({_PREFIX + "embedder.weight": np.zeros((4, 3), dtype=np.float16)}, str(shard))
    from vmlx_engine.model_bundle_integrity import BundleIntegrityError
    with pytest.raises((ValueError, FileNotFoundError, BundleIntegrityError)):
        temporal_video_spec(tmp_path)


@pytest.mark.parametrize("cap", [1, 2, 3])
def test_native_video_obeys_cap_and_temporal_groups(tmp_path, monkeypatch, cap):
    import torch
    from vmlx_engine.models import mllm
    observed = {}
    # Include duplicates: every selected frame must reach the temporal encoder.
    frames = [np.full((2, 2, 3), value, dtype=np.uint8) for value in [0, 0, 2, 3]]
    def sample(path, **kwargs):
        observed['sampling'] = kwargs
        return frames
    monkeypatch.setattr(mllm, 'extract_video_frames_smart', sample)
    class Processor:
        _is_video_mode = False
        def __call__(self, *, images, return_tensors):
            assert self._is_video_mode is True
            observed['images'] = images
            observed['pixels'] = [np.array(image) for image in images]
            return {'pixel_values': torch.zeros((len(images), 3, 2, 2))}
    class Model:
        video_temporal_patch_dim = 2
        def extract_video_feature(self, pixels):
            observed['dtype'] = pixels.dtype
            observed['frame_count'] = len(pixels)
            return torch.ones(((len(pixels) + 1) // 2, 3, 4))
    processor = Processor()
    session = SimpleNamespace(pt_model=Model(), processor=SimpleNamespace(image_processor=processor),
                              device='cpu', torch_dtype=torch.bfloat16)
    embeds, prompt = encode_temporal_video(session, tmp_path/'clip.mp4',
        controls={'fps': 2.0, 'max_frames': cap}, temporal_patch_size=2)
    assert observed['sampling'] == {'fps': 2.0, 'max_frames': cap}
    assert observed['frame_count'] == cap
    assert observed['dtype'] == torch.bfloat16
    assert processor._is_video_mode is False
    assert embeds.shape == ((cap + 1)//2, 3, 4)
    assert prompt.count('<image>') == embeds.shape[0] * 3
    assert f'frame {cap + 1}' not in prompt.lower()
    assert prompt.startswith('Frame 1' + (' and frame 2' if cap > 1 else '') + ': <img>')
    for image in observed['images']:
        with pytest.raises(ValueError):
            image.getpixel((0, 0))


def test_native_video_processor_mode_restored_on_failure(tmp_path, monkeypatch):
    from vmlx_engine.models import mllm
    monkeypatch.setattr(mllm, 'extract_video_frames_smart', lambda *a, **kw: [np.zeros((2, 2, 3), dtype=np.uint8)])
    class Processor:
        _is_video_mode = 'original'
        def __call__(self, **kwargs):
            assert self._is_video_mode is True
            raise RuntimeError('bad pixels')
    processor = Processor()
    session = SimpleNamespace(pt_model=SimpleNamespace(video_temporal_patch_dim=2),
                              processor=SimpleNamespace(image_processor=processor))
    with pytest.raises(RuntimeError, match='bad pixels'):
        encode_temporal_video(session, 'clip', controls={'fps': 2.0, 'max_frames': 2}, temporal_patch_size=2)
    assert processor._is_video_mode == 'original'


@pytest.mark.parametrize("has_history", [False, True])
def test_multiple_clips_use_full_assembly_without_incremental_overwrite(tmp_path, monkeypatch, has_history):
    clips = []
    for name in ('one.mp4', 'two.mp4'):
        path = tmp_path / name
        path.write_bytes(b'clip')
        clips.append({'type': 'video_url', 'video_url': {'url': str(path)}})
    messages = ([{'role': 'user', 'content': 'old'}, {'role': 'assistant', 'content': 'answer'}]
                if has_history else []) + [{'role': 'user', 'content': clips}]
    observed = []
    monkeypatch.setattr(omni, '_run_omni_full_history', lambda session, supplied, **kwargs: observed.append(supplied) or 'answer')
    d = dispatcher(tmp_path, True)
    d._last_signature = omni._conversation_signature(messages[:-1], False, video_policy=omni._omni_video_policy(temporal_patch_size=2))
    result = d.chat(messages, enable_thinking=False)
    assert observed == [messages]
    assert result['cached_tokens'] == 0
    assert result['has_video'] is True
    assert not hasattr(d._session, 'last_turn')


def test_cold_native_history_preserves_video_labels_and_embedding_order(tmp_path):
    from functools import partial
    from PIL import Image
    from vmlx_engine.omni_native_prompt import run_full_history
    image_path = tmp_path/'still.png'
    Image.new('RGB', (2, 2)).save(image_path)
    for name in ('one.mp4', 'two.mp4'):
        (tmp_path/name).write_bytes(b'clip')
    observed = {}
    class Tokenizer:
        def apply_chat_template(self, transcript, **kwargs):
            observed['transcript'] = transcript
            return 'prompt'
        def __call__(self, text, **kwargs):
            return {'input_ids': np.array([[1, 2, 3]])}
    class Session:
        tokenizer = Tokenizer()
        mx = SimpleNamespace(array=np.array)
        mlx_model = SimpleNamespace(backbone=SimpleNamespace(embeddings=lambda ids: np.zeros((1, 3, 4))))
        def _ensure_cache(self): pass
        def _extract_image_embeddings(self, images):
            return np.full((1, 1, 4), 2)
        def _extract_video_embeddings(self, path):
            value = 1 if Path(path).name == 'one.mp4' else 3
            self._vmlx_video_prompt = 'Frame 1 and frame 2: <img><image><image></img>\n'
            return np.full((1, 2, 4), value)
        def _inject_embeddings(self, ids, text, visual, video, audio):
            observed['visual'] = visual
            return text
        def _decode_turn(self, embeds, **kwargs): return 'reply'
    messages = [{'role': 'user', 'content': [
        {'type': 'video_url', 'video_url': {'url': str(tmp_path/'one.mp4')}},
        {'type': 'image_url', 'image_url': {'url': str(image_path)}},
        {'type': 'video_url', 'video_url': {'url': str(tmp_path/'two.mp4')}},
        {'type': 'text', 'text': 'Compare.'}]}]
    run_full_history(Session(), messages, scratch_dir=tmp_path,
        extract_parts=partial(omni._extract_parts, native_video=True),
        enable_thinking=False, max_tokens=32, temperature=0, top_p=1)
    assert observed['transcript'][0]['content'] == (
        'Frame 1 and frame 2: <img><image><image></img>\n'
        '<img><image></img>\n'
        'Frame 1 and frame 2: <img><image><image></img>\nCompare.')
    assert observed['visual'][0, :, 0].tolist() == [1, 1, 2, 3, 3]
    assert isinstance(messages[0]['content'], list)
