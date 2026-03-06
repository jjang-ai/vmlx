# SPDX-License-Identifier: Apache-2.0
"""
Tests for audio support (STT, TTS, audio processing).

Note: Some tests require mlx-audio to be installed.
"""

import pytest
import numpy as np


class TestSTTEngine:
    """Tests for Speech-to-Text engine."""

    def test_init_whisper(self):
        """Test STT engine initialization with Whisper."""
        from vmlx_engine.audio.stt import STTEngine

        engine = STTEngine("mlx-community/whisper-large-v3-mlx")
        assert engine.model_name == "mlx-community/whisper-large-v3-mlx"
        assert engine._is_parakeet is False
        assert engine._loaded is False

    def test_init_parakeet(self):
        """Test STT engine initialization with Parakeet."""
        from vmlx_engine.audio.stt import STTEngine

        engine = STTEngine("mlx-community/parakeet-tdt-0.6b-v2")
        assert engine._is_parakeet is True

    def test_default_models(self):
        """Test default model constants."""
        from vmlx_engine.audio.stt import DEFAULT_WHISPER_MODEL, DEFAULT_PARAKEET_MODEL

        assert "whisper" in DEFAULT_WHISPER_MODEL.lower()
        assert "parakeet" in DEFAULT_PARAKEET_MODEL.lower()

    def test_transcription_result(self):
        """Test TranscriptionResult dataclass."""
        from vmlx_engine.audio.stt import TranscriptionResult

        result = TranscriptionResult(
            text="Hello world",
            language="en",
            duration=2.5,
        )
        assert result.text == "Hello world"
        assert result.language == "en"
        assert result.duration == 2.5


class TestTTSEngine:
    """Tests for Text-to-Speech engine."""

    def test_init_kokoro(self):
        """Test TTS engine initialization with Kokoro."""
        from vmlx_engine.audio.tts import TTSEngine

        engine = TTSEngine("mlx-community/Kokoro-82M-bf16")
        assert engine.model_name == "mlx-community/Kokoro-82M-bf16"
        assert engine._model_family == "kokoro"
        assert engine._loaded is False

    def test_init_chatterbox(self):
        """Test TTS engine initialization with Chatterbox."""
        from vmlx_engine.audio.tts import TTSEngine

        engine = TTSEngine("mlx-community/chatterbox-turbo-fp16")
        assert engine._model_family == "chatterbox"

    def test_init_vibevoice(self):
        """Test TTS engine initialization with VibeVoice."""
        from vmlx_engine.audio.tts import TTSEngine

        engine = TTSEngine("mlx-community/VibeVoice-Realtime-0.5B-4bit")
        assert engine._model_family == "vibevoice"

    def test_init_voxcpm(self):
        """Test TTS engine initialization with VoxCPM."""
        from vmlx_engine.audio.tts import TTSEngine

        engine = TTSEngine("mlx-community/VoxCPM1.5")
        assert engine._model_family == "voxcpm"

    def test_available_voices(self):
        """Test voice lists."""
        from vmlx_engine.audio.tts import KOKORO_VOICES, CHATTERBOX_VOICES

        assert "af_heart" in KOKORO_VOICES
        assert len(KOKORO_VOICES) > 5
        assert "default" in CHATTERBOX_VOICES

    def test_get_voices(self):
        """Test get_voices method."""
        from vmlx_engine.audio.tts import TTSEngine

        kokoro = TTSEngine("mlx-community/Kokoro-82M-bf16")
        voices = kokoro.get_voices()
        assert "af_heart" in voices

    def test_audio_output(self):
        """Test AudioOutput dataclass."""
        from vmlx_engine.audio.tts import AudioOutput

        audio = np.zeros(24000, dtype=np.float32)
        output = AudioOutput(
            audio=audio,
            sample_rate=24000,
            duration=1.0,
        )
        assert output.sample_rate == 24000
        assert output.duration == 1.0
        assert len(output.audio) == 24000


class TestAudioProcessor:
    """Tests for audio processor (SAM-Audio)."""

    def test_init(self):
        """Test audio processor initialization."""
        from vmlx_engine.audio.processor import AudioProcessor

        processor = AudioProcessor("mlx-community/sam-audio-large-fp16")
        assert processor.model_name == "mlx-community/sam-audio-large-fp16"
        assert processor._loaded is False

    def test_default_model(self):
        """Test default SAM-Audio model."""
        from vmlx_engine.audio.processor import DEFAULT_SAM_MODEL

        assert "sam-audio" in DEFAULT_SAM_MODEL.lower()

    def test_separation_result(self):
        """Test SeparationResult dataclass."""
        from vmlx_engine.audio.processor import SeparationResult

        target = np.zeros(44100, dtype=np.float32)
        residual = np.zeros(44100, dtype=np.float32)

        result = SeparationResult(
            target=target,
            residual=residual,
            sample_rate=44100,
            peak_memory=1.5,
        )
        assert result.sample_rate == 44100
        assert result.peak_memory == 1.5
        assert len(result.target) == 44100


class TestAPIModels:
    """Tests for audio API models."""

    def test_audio_url(self):
        """Test AudioUrl model."""
        from vmlx_engine.api.models import AudioUrl

        url = AudioUrl(url="file://test.mp3")
        assert url.url == "file://test.mp3"

    def test_content_part_audio(self):
        """Test ContentPart with audio."""
        from vmlx_engine.api.models import ContentPart

        part = ContentPart(type="audio_url", audio_url={"url": "test.mp3"})
        assert part.type == "audio_url"
        # Pydantic converts dict to AudioUrl model
        assert part.audio_url.url == "test.mp3"

    def test_speech_request(self):
        """Test AudioSpeechRequest model."""
        from vmlx_engine.api.models import AudioSpeechRequest

        req = AudioSpeechRequest(
            model="kokoro",
            input="Hello world",
            voice="af_heart",
            speed=1.2,
        )
        assert req.model == "kokoro"
        assert req.input == "Hello world"
        assert req.voice == "af_heart"
        assert req.speed == 1.2

class TestAudioImports:
    """Test that all audio modules can be imported."""

    def test_import_audio_module(self):
        """Test importing main audio module."""
        from vmlx_engine.audio import (
            STTEngine,
            TTSEngine,
            AudioProcessor,
        )

        assert STTEngine is not None
        assert TTSEngine is not None
        assert AudioProcessor is not None

    def test_import_api_models(self):
        """Test importing audio API models."""
        from vmlx_engine.api import (
            AudioUrl,
            AudioSpeechRequest,
        )

        assert AudioUrl is not None
        assert AudioSpeechRequest is not None


# Integration tests (require mlx-audio installed)
@pytest.mark.skip(reason="Requires mlx-audio and models downloaded")
class TestAudioIntegration:
    """Integration tests for audio (require models)."""

    def test_whisper_transcription(self):
        """Test Whisper transcription."""
        from vmlx_engine.audio import transcribe_audio

        result = transcribe_audio(
            "test_audio.wav",
            model_name="mlx-community/whisper-small-mlx",
        )
        assert result.text is not None

    def test_kokoro_tts(self):
        """Test Kokoro TTS generation."""
        from vmlx_engine.audio import generate_speech

        audio = generate_speech(
            "Hello world",
            model_name="mlx-community/Kokoro-82M-bf16",
            voice="af_heart",
        )
        assert audio.audio is not None
        assert audio.sample_rate > 0

    def test_sam_audio_separation(self):
        """Test SAM-Audio voice separation."""
        from vmlx_engine.audio import separate_voice

        target, residual = separate_voice(
            "test_audio.wav",
            model_name="mlx-community/sam-audio-small",
        )
        assert target is not None
        assert residual is not None


class TestTTSErrorMessages:
    """Tests for improved TTS/STT error messages that distinguish
    'mlx-audio not installed' from transitive dependency failures."""

    def test_tts_load_catches_import_error(self):
        """TTSEngine.load() re-raises ImportError with context."""
        from vmlx_engine.audio.tts import TTSEngine
        from unittest.mock import patch

        engine = TTSEngine("mlx-community/Kokoro-82M-bf16")

        # Simulate mlx-audio itself missing
        with patch.dict("sys.modules", {"mlx_audio": None, "mlx_audio.tts": None, "mlx_audio.tts.generate": None}):
            try:
                engine.load()
                assert False, "Should have raised ImportError"
            except ImportError as e:
                assert "mlx-audio" in str(e).lower() or "mlx_audio" in str(e).lower()

    def test_tts_transitive_dep_error_preserved(self):
        """TTSEngine.load() preserves transitive dependency error messages."""
        from vmlx_engine.audio.tts import TTSEngine
        from unittest.mock import patch

        engine = TTSEngine("mlx-community/Kokoro-82M-bf16")

        # Simulate a transitive dep failure (e.g., misaki, num2words)
        def fake_load_model(*args, **kwargs):
            raise ImportError("No module named 'num2words'")

        with patch("vmlx_engine.audio.tts.TTSEngine.load", side_effect=ImportError("TTS dependency missing: No module named 'num2words'")):
            try:
                engine.load()
                assert False, "Should have raised ImportError"
            except ImportError as e:
                msg = str(e)
                assert "num2words" in msg
                # Should NOT say "mlx-audio not installed" for transitive deps
                assert "mlx-audio is required" not in msg

    def test_stt_load_catches_import_error(self):
        """STTEngine.load() re-raises ImportError with context."""
        from vmlx_engine.audio.stt import STTEngine
        from unittest.mock import patch

        engine = STTEngine("mlx-community/whisper-large-v3-mlx")

        with patch.dict("sys.modules", {"mlx_audio": None, "mlx_audio.stt": None, "mlx_audio.stt.utils": None}):
            try:
                engine.load()
                assert False, "Should have raised ImportError"
            except ImportError as e:
                assert "mlx-audio" in str(e).lower() or "mlx_audio" in str(e).lower() or "stt" in str(e).lower()


class TestMambaCacheCompat:
    """Tests for BatchMambaCache compatibility across mlx-lm versions."""

    def test_batch_mamba_cache_init(self):
        """BatchMambaCache initializes regardless of MambaCache signature."""
        from vmlx_engine.utils.mamba_cache import BatchMambaCache

        cache = BatchMambaCache(size=3, left_padding=[0, 0])
        assert cache._batch_size == 2
        assert len(cache.cache) == 3

    def test_batch_mamba_cache_init_no_padding(self):
        """BatchMambaCache handles None left_padding."""
        from vmlx_engine.utils.mamba_cache import BatchMambaCache

        cache = BatchMambaCache(size=2, left_padding=None)
        assert cache._batch_size == 0

    def test_batch_mamba_cache_extract(self):
        """BatchMambaCache.extract returns a MambaCache with correct structure."""
        import mlx.core as mx
        from vmlx_engine.utils.mamba_cache import BatchMambaCache

        cache = BatchMambaCache(size=2, left_padding=[0, 0])
        cache.cache = [
            mx.zeros((2, 4, 8)),
            mx.zeros((2, 4, 8)),
        ]
        extracted = cache.extract(0)
        assert extracted.cache[0].shape == (1, 4, 8)
        assert extracted.cache[1].shape == (1, 4, 8)

    def test_batch_mamba_cache_merge(self):
        """BatchMambaCache.merge concatenates caches along batch dim."""
        import mlx.core as mx
        from vmlx_engine.utils.mamba_cache import BatchMambaCache, MambaCache

        try:
            c1 = MambaCache(2)
        except TypeError:
            c1 = MambaCache()
            c1.cache = [None, None]
        c1.cache = [mx.zeros((1, 4, 8)), mx.zeros((1, 4, 8))]

        try:
            c2 = MambaCache(2)
        except TypeError:
            c2 = MambaCache()
            c2.cache = [None, None]
        c2.cache = [mx.ones((1, 4, 8)), mx.ones((1, 4, 8))]

        merged = BatchMambaCache.merge([c1, c2])
        assert merged.cache[0].shape == (2, 4, 8)
        assert merged._batch_size == 2


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
