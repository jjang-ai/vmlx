"""Cold native Omni requests must honor the transcript supplied by the client."""
from pathlib import Path
import threading

from vmlx_engine.omni_multimodal import OmniMultimodalDispatcher


class Session:
    def reset(self):
        pass

    def turn(self, **kwargs):
        self.last_turn = kwargs
        return "old answer"


def dispatcher(tmp_path):
    d = OmniMultimodalDispatcher.__new__(OmniMultimodalDispatcher)
    d._session = Session()
    d._backend = "stage1"
    d._lock = threading.Lock()
    d._last_signature = None
    d._scratch_dir = tmp_path
    return d


def test_cold_media_history_is_replayed_as_supplied_not_last_user_only(tmp_path, monkeypatch):
    import vmlx_engine.omni_multimodal as omni
    observed = []
    monkeypatch.setattr(omni, "_run_omni_full_history", lambda session, messages, **kw: observed.append(messages) or "answer", raising=False)
    messages = [
        {"role": "system", "content": "Remember the provided history."},
        {"role": "user", "content": "My code is CEDAR."},
        {"role": "assistant", "content": "I will remember CEDAR."},
        {"role": "user", "content": "What code did I give?"},
    ]
    d = dispatcher(tmp_path)
    d.chat(messages, enable_thinking=False)
    assert observed == [messages]


def test_changed_assistant_context_cannot_reuse_same_user_prefix(tmp_path, monkeypatch):
    import vmlx_engine.omni_multimodal as omni
    observed = []
    monkeypatch.setattr(omni, "_run_omni_full_history", lambda session, messages, **kw: observed.append(messages) or "answer", raising=False)
    d = dispatcher(tmp_path)
    first = [{"role": "user", "content": "Choose a code."}]
    d.chat(first, enable_thinking=False)
    corrected = first + [{"role": "assistant", "content": "New supplied code BLUE."}, {"role": "user", "content": "Repeat it."}]
    d.chat(corrected, enable_thinking=False)
    assert observed == [corrected]


def test_full_history_prefill_encodes_prior_media_and_keeps_supplied_roles(tmp_path):
    import base64
    from io import BytesIO
    from types import SimpleNamespace
    import numpy as np
    from PIL import Image
    from vmlx_engine.omni_native_prompt import run_full_history
    from vmlx_engine.omni_multimodal import _extract_parts

    png = BytesIO()
    Image.new('RGB', (2, 2), 'green').save(png, format='PNG')
    image = {'type': 'image_url', 'image_url': {'url': 'data:image/png;base64,' + base64.b64encode(png.getvalue()).decode()}}
    audio = {'type': 'input_audio', 'input_audio': {'data': 'AA==', 'format': 'wav'}}
    messages = [
        {'role': 'system', 'content': 'Keep the exact supplied history.'},
        {'role': 'user', 'content': [{'type': 'text', 'text': 'Remember this image.'}, image]},
        {'role': 'assistant', 'content': 'The code is CEDAR.', 'reasoning_content': 'Supplied reasoning.'},
        {'role': 'user', 'content': [{'type': 'text', 'text': 'Transcribe and recall my code.'}, audio]},
    ]
    observed = {}

    class Tokenizer:
        def apply_chat_template(self, transcript, **kwargs):
            observed['transcript'] = transcript
            observed['kwargs'] = kwargs
            return 'full-transcript'
        def __call__(self, text, **kwargs):
            return {'input_ids': np.array([[1, 2, 3, 4]])}

    class Native:
        tokenizer = Tokenizer()
        mx = SimpleNamespace(array=np.array)
        mlx_model = SimpleNamespace(backbone=SimpleNamespace(embeddings=lambda ids: np.zeros((1, 4, 8))))
        def _ensure_cache(self):
            pass
        def _extract_image_embeddings(self, images):
            observed['image_count'] = len(images)
            return np.ones((1, 2, 8))
        def _extract_audio_embeddings(self, path):
            observed['audio_bytes'] = Path(path).read_bytes()
            return np.full((1, 1, 8), 2)
        def _inject_embeddings(self, ids, text, visual, video, sound):
            observed['visual'] = visual
            observed['sound'] = sound
            assert video is None
            return text
        def _decode_turn(self, embeds, **kwargs):
            observed['decodes'] = observed.get('decodes', 0) + 1
            return 'new reply'

    session = Native()
    result = run_full_history(session, messages, scratch_dir=tmp_path, extract_parts=_extract_parts,
        enable_thinking=False, max_tokens=64, temperature=0, top_p=1)
    assert result == 'new reply'
    assert observed['decodes'] == 1  # Earlier replies were not regenerated.
    assert observed['transcript'][0] == messages[0]
    assert observed['transcript'][2] == messages[2]
    assert observed['transcript'][1]['content'] == '<img><image><image></img>\nRemember this image.'
    assert observed['transcript'][3]['content'] == '<sound><so_embedding></sound>\nTranscribe and recall my code.'
    assert observed['image_count'] == 1 and observed['audio_bytes'] == b'\0'
    assert np.all(observed['visual'] == 1) and np.all(observed['sound'] == 2)
    assert observed['kwargs']['enable_thinking'] is False
    assert messages[1]['content'][1] == image  # No wire transcript mutation.
    assert session._last_prompt_tokens == 4


def test_causal_signature_includes_system_assistant_and_cache_salt():
    from vmlx_engine.omni_multimodal import _conversation_signature
    base = [{'role': 'system', 'content': 'A'}, {'role': 'user', 'content': 'Q'}, {'role': 'assistant', 'content': 'Answer'}]
    key = _conversation_signature(base, False, 'one')
    assert key != _conversation_signature([{'role': 'system', 'content': 'B'}] + base[1:], False, 'one')
    assert key != _conversation_signature(base[:-1] + [{'role': 'assistant', 'content': 'Changed'}], False, 'one')
    assert key != _conversation_signature(base, True, 'one')
    assert key != _conversation_signature(base, False, 'two')


def test_bypass_rebuilds_supplied_history_even_with_matching_state(tmp_path, monkeypatch):
    import vmlx_engine.omni_multimodal as omni
    d = dispatcher(tmp_path)
    first = [{'role': 'user', 'content': 'Choose a code.'}]
    d.chat(first, enable_thinking=False)
    messages = first + [{'role': 'assistant', 'content': 'old answer'}, {'role': 'user', 'content': 'Repeat.'}]
    observed = []
    monkeypatch.setattr(omni, '_run_omni_full_history', lambda session, supplied, **kw: observed.append(supplied) or 'reply')
    d.chat(messages, enable_thinking=False, force_reset=True)
    assert observed == [messages]
