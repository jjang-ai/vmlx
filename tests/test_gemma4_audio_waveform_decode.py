import math
import wave

import numpy as np


def test_gemma4_audio_waveforms_from_paths_decodes_wav_to_float32(tmp_path):
    from vmlx_engine.mllm_batch_generator import _gemma4_audio_waveforms_from_paths

    wav_path = tmp_path / "blue.wav"
    sample_rate = 16000
    frames = bytearray()
    for i in range(sample_rate // 20):
        value = int(12000 * math.sin(2 * math.pi * 440 * i / sample_rate))
        frames.extend(value.to_bytes(2, "little", signed=True))

    with wave.open(str(wav_path), "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(sample_rate)
        handle.writeframes(bytes(frames))

    class _AudioProcessor:
        sampling_rate = sample_rate

    class _Processor:
        audio_processor = _AudioProcessor()

    waveforms = _gemma4_audio_waveforms_from_paths([str(wav_path)], _Processor())

    assert len(waveforms) == 1
    assert isinstance(waveforms[0], np.ndarray)
    assert waveforms[0].dtype == np.float32
    assert waveforms[0].ndim == 1
    assert waveforms[0].size > 0
