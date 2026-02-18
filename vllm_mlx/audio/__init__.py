# SPDX-License-Identifier: Apache-2.0
"""
Audio support for vllm-mlx using mlx-audio.

Provides:
- STT (Speech-to-Text): Whisper, Parakeet
- TTS (Text-to-Speech): Kokoro, Chatterbox, VibeVoice, VoxCPM
- Audio Processing: SAM-Audio (voice separation)
"""

from .stt import STTEngine, transcribe_audio
from .tts import TTSEngine, generate_speech
from .processor import AudioProcessor, separate_voice

__all__ = [
    # STT
    "STTEngine",
    "transcribe_audio",
    # TTS
    "TTSEngine",
    "generate_speech",
    # Processing
    "AudioProcessor",
    "separate_voice",
]
