"""Unsupported native media sources must never become text-only requests."""
from types import SimpleNamespace
import pytest
from fastapi import HTTPException
from vmlx_engine import omni_multimodal as omni


@pytest.mark.asyncio
@pytest.mark.parametrize('part', [
    {'type': 'image_url', 'image_url': {'url': 'https://example.com/changing.png'}},
    {'type': 'video_url', 'video_url': {'url': 'https://example.com/changing.mp4'}},
    {'type': 'audio_url', 'audio_url': {'url': 'https://example.com/changing.wav'}},
    {'type': 'image_url', 'image_url': {}},
    {'type': 'video_url', 'video_url': {'url': '/missing/native-media.mp4'}},
    {'type': 'input_audio', 'input_audio': {'format': 'wav'}},
])
@pytest.mark.parametrize('thinking', [True, False])
async def test_invalid_sources_rejected_before_dispatcher_or_cache(monkeypatch, part, thinking):
    monkeypatch.setattr(omni, 'omni_multimodal_component_status', lambda _: {'modalities': ['text', 'image', 'audio', 'video']})
    monkeypatch.setattr(omni.OmniMultimodalDispatcher, 'get', lambda *a, **kw: pytest.fail('invalid source reached model/cache owner'))
    request = SimpleNamespace(messages=[{'role': 'user', 'content': [part]}], enable_thinking=thinking)
    with pytest.raises(HTTPException) as error:
        await omni.dispatch_omni_chat_completion(request, '/unused', disk_cache_enabled=True)
    assert error.value.status_code == 400
    assert 'media source' in error.value.detail.lower()
    assert 'example.com' not in error.value.detail  # No signed URL/query reflection.


@pytest.mark.parametrize('part', [
    {'type': 'image_url', 'image_url': {'url': 'data:image/png;base64,AA=='}},
    {'type': 'video', 'video': 'data:video/mp4;base64,AA=='},
    {'type': 'input_audio', 'input_audio': {'data': 'AA==', 'format': 'wav'}},
    {'type': 'audio_url', 'audio_url': 'data:audio/wav;base64,AA=='},
])
def test_supported_data_sources_keep_existing_native_path(part):
    omni._validate_native_media_sources([{'role': 'user', 'content': [part]}])


def test_local_file_source_is_valid_but_directory_is_not(tmp_path):
    path = tmp_path / 'image.png'; path.write_bytes(b'image')
    omni._validate_native_media_sources([{'role': 'user', 'content': [{'type': 'image_url', 'image_url': {'url': str(path)}}]}])
    with pytest.raises(HTTPException):
        omni._validate_native_media_sources([{'role': 'user', 'content': [{'type': 'image_url', 'image_url': {'url': str(tmp_path)}}]}])


@pytest.mark.parametrize('value', ['data:image/png;base64,', 'data:image/png;base64,%%%', 'data:image/png,raw', 'data:image/png;base64'])
def test_malformed_data_envelopes_fail_before_native_decode(value):
    with pytest.raises(HTTPException) as error:
        omni._validate_native_media_sources([{'role': 'user', 'content': [{'type': 'image_url', 'image_url': {'url': value}}]}])
    assert error.value.status_code == 400


def test_malformed_audio_data_does_not_hide_behind_valid_fallback_url(tmp_path):
    path = tmp_path / 'audio.wav';path.write_bytes(b'audio')
    with pytest.raises(HTTPException):
        omni._validate_native_media_sources([{'role': 'user', 'content': [
            {'type': 'audio_url', 'audio_url': {'data': 123, 'url': str(path)}}]}])
