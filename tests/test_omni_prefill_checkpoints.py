"""Native prompt checkpoints must survive template reasoning truncation safely."""
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace
import json
import threading

import numpy as np
import pytest

from vmlx_engine.omni_native_prefix import NativePrefillCheckpoints, prefill_state, source_prefix_keys
from vmlx_engine.omni_native_prompt import run_full_history
from vmlx_engine.omni_multimodal import OmniMultimodalDispatcher, _OMNI_SESSION_L2_SCHEMA


def tiny_model():
    import mlx.core as mx
    from mlx_lm.models.nemotron_h import Model, ModelArgs
    mx.random.seed(19)
    return Model(ModelArgs(
        model_type='nemotron_h', vocab_size=128, hidden_size=32,
        intermediate_size=64, num_hidden_layers=3, max_position_embeddings=1024,
        num_attention_heads=4, num_key_value_heads=2, head_dim=8,
        attention_bias=False, mamba_num_heads=4, mamba_head_dim=8,
        mamba_proj_bias=False, ssm_state_size=8, conv_kernel=4, n_groups=1,
        mlp_bias=False, layer_norm_epsilon=1e-5, use_bias=False,
        use_conv_bias=True, hybrid_override_pattern=['M', '-', '*'],
    ))


def owner(tmp_path):
    from vmlx_engine.utils.omni_session_disk_store import OmniSessionDiskStore
    model = tiny_model()
    d = OmniMultimodalDispatcher.__new__(OmniMultimodalDispatcher)
    d._session = SimpleNamespace(mlx_model=model, _cache=model.make_cache(), _history_text=[])
    d._session_l2_fingerprint = 'tiny-native-model'
    d._session_l2_store = OmniSessionDiskStore(root=tmp_path, model_key='tiny-native-model', max_size_bytes=10_000_000)
    d._session_l2_stats = {'hits': 0, 'misses': 0, 'stores': 0}
    return d


def put_checkpoint(d, messages, tokens, *, metadata_change=None):
    import mlx.core as mx
    from mlx_lm.models.cache import save_prompt_cache
    checkpoint = NativePrefillCheckpoints(d, messages, {}, publish=True)
    d._session._cache = d._session.mlx_model.make_cache()
    prefill_state(d._session, d._session.mlx_model.backbone.embeddings(mx.array([tokens])))
    checkpoint.publish(np.array([tokens]), messages)
    if metadata_change:
        from mlx_lm.models.cache import load_prompt_cache
        path = d._session_l2_path
        cache, metadata = load_prompt_cache(str(path), return_metadata=True)
        metadata.update(metadata_change)
        d._native_disk_store().save(checkpoint.keys[-1][1], lambda p: save_prompt_cache(str(p), cache, metadata))
    d._session._cache = None
    return checkpoint


def test_native_hybrid_split_prefill_roundtrip_matches_cold_logits(tmp_path):
    import mlx.core as mx
    d = owner(tmp_path)
    messages = [{'role': 'user', 'content': 'media prompt'}]
    tokens = [1, 2, 3, 4, 5, 6, 7]
    cold_cache = d._session.mlx_model.make_cache()
    cold = d._session.mlx_model(mx.array([tokens]), cache=cold_cache)
    mx.eval(cold)
    put_checkpoint(d, messages, tokens[:4])
    loaded = NativePrefillCheckpoints(d, messages, {}, publish=False)
    assert loaded.find()
    assert loaded.accept(np.array([tokens])) == 4
    warm = d._session.mlx_model(mx.array([tokens[4:]]), cache=d._session._cache)
    mx.eval(warm)
    assert mx.allclose(cold[:, 4:], warm, atol=1e-4, rtol=1e-4).item()
    for expected, actual in zip(cold_cache, d._session._cache):
        for left, right in zip(expected.state, actual.state):
            assert left.dtype == right.dtype
            assert mx.allclose(left, right, atol=1e-4, rtol=1e-4).item()
    assert d._session_l2_stats['hits'] == 1
    assert d._session_l2_stats['last_restore_seconds'] >= 0


def test_longest_available_user_checkpoint_and_no_duplicate_write(tmp_path):
    d = owner(tmp_path)
    first = [{'role': 'user', 'content': 'one'}]
    second = first + [{'role': 'assistant', 'content': 'answer'}, {'role': 'user', 'content': 'two'}]
    put_checkpoint(d, first, [1, 2])
    put_checkpoint(d, second, [1, 2, 3, 4])
    checkpoint = NativePrefillCheckpoints(d, second, {}, publish=True)
    assert checkpoint.find()['source_count'] == 3
    assert checkpoint.accept(np.array([[1, 2, 3, 4, 5]])) == 4
    before = d._session_l2_path.stat().st_mtime_ns
    checkpoint.publish(np.array([[1, 2, 3, 4]]), second)
    assert d._session_l2_stats['stores'] == 2
    assert d._session_l2_path.stat().st_mtime_ns == before


@pytest.mark.parametrize('change', [
    {'schema': 'old'}, {'kind': 'completed_turn'}, {'bundle_fingerprint': 'other'},
    {'source_message_count': '99'}, {'token_ids': '[1, true]'},
    {'token_ids': '[1]'}, {'history_json': '{}'},
])
def test_incompatible_checkpoint_never_assigns_native_state(tmp_path, change):
    d = owner(tmp_path)
    messages = [{'role': 'user', 'content': 'one'}]
    put_checkpoint(d, messages, [1, 2], metadata_change=change)
    checkpoint = NativePrefillCheckpoints(d, messages, {}, publish=False)
    assert checkpoint.find() is None
    assert d._session._cache is None
    assert d._session_l2_stats['hits'] == 0


def test_changed_template_tokens_reject_before_assigning_recurrent_state(tmp_path):
    d = owner(tmp_path)
    messages = [{'role': 'user', 'content': 'one'}]
    put_checkpoint(d, messages, [1, 2])
    checkpoint = NativePrefillCheckpoints(d, messages, {}, publish=False)
    assert checkpoint.find()
    assert checkpoint.accept(np.array([[1, 9, 3]])) is None
    assert d._session._cache is None
    assert checkpoint.find() is None  # A cold retry must not select it again.
    assert d._session_l2_stats['hits'] == 0


def test_source_key_binds_media_bytes_preprocessing_and_causal_history(tmp_path):
    path = tmp_path / 'video.mp4'
    path.write_bytes(b'first-video')
    messages = [{'role': 'user', 'content': [{'type': 'video_url', 'video_url': {'url': str(path)}}]}]
    original = source_prefix_keys(messages, {'fps': 2})
    assert original != source_prefix_keys(messages, {'fps': 3})
    path.write_bytes(b'other-video')
    assert original != source_prefix_keys(messages, {'fps': 2})
    altered = [{'role': 'system', 'content': 'Different policy'}] + messages
    assert original[-1][1] != source_prefix_keys(altered, {'fps': 2})[-1][1]


def test_ssd_publication_failure_is_not_hidden(tmp_path, monkeypatch):
    d = owner(tmp_path)
    checkpoint = NativePrefillCheckpoints(d, [{'role': 'user', 'content': 'one'}], {}, publish=True)
    def fail(*a):
        raise OSError('disk full')
    monkeypatch.setattr(d._session_l2_store, 'save', fail)
    with pytest.raises(OSError, match='disk full'):
        checkpoint.publish(np.array([[1, 2]]), [{'role': 'user', 'content': 'one'}])
    assert d._session_l2_stats['stores'] == 0
    assert d._session_l2_stats['last_error'] == 'disk full'


@pytest.mark.parametrize('force_reset,enabled', [(False, True), (True, True), (False, False)])
def test_dispatcher_uses_earlier_checkpoint_only_when_cache_allowed(tmp_path, monkeypatch, force_reset, enabled):
    from vmlx_engine import omni_multimodal as omni
    calls = []
    class Session:
        _vmlx_prefill_checkpoints = True
        def reset(self): self._cache = None
        def turn(self, **kw): return 'answer'
    d = OmniMultimodalDispatcher.__new__(OmniMultimodalDispatcher)
    d._session = Session(); d._backend = 'stage1'; d._lock = threading.Lock()
    d._scratch_dir = tmp_path; d._last_signature = None; d._disk_cache_enabled = enabled
    d._try_restore_session_snapshot = lambda signature: False
    def assemble(session, *args, **kwargs):
        calls.append(kwargs['checkpoints'])
        session._last_prompt_tokens = 8
        session._vmlx_restored_prefix_tokens = 20
        return 'Thinking.</think>answer'
    monkeypatch.setattr(omni, '_run_omni_full_history', assemble)
    result = d.chat([{'role': 'user', 'content': 'question'}], enable_thinking=True, force_reset=force_reset)
    if enabled and not force_reset:
        assert len(calls) == 1 and calls[0].publish_enabled
        assert result['cached_tokens'] == 20 and result['prompt_tokens'] == 28
    else:
        assert not calls and result['cached_tokens'] == 0


def history_session(model):
    import mlx.core as mx
    class Tokenizer:
        changed = False
        def apply_chat_template(self, messages, **kwargs):
            # Previous reasoning is omitted, as in the native default template.
            text = ('changed:' if self.changed else '') + ''.join(
                m['role'] + ':' + m['content'] + '\n' for m in messages)
            return text + ('assistant:<think>' if kwargs.get('enable_thinking', True) else 'assistant:') if kwargs['add_generation_prompt'] else text
        def __call__(self, text, **kwargs):
            text = text.replace('<image>', '\x12')
            return {'input_ids': np.array([[ord(c) for c in text]])}
    class Session:
        tokenizer = Tokenizer()
        mlx_model = model
        _cache = None
        _history_text = []
        video_encodes = 0
        injected = []
        def _ensure_cache(self):
            if self._cache is None: self._cache = model.make_cache()
        def _extract_video_embeddings(self, path):
            self.video_encodes += 1
            return np.ones((1, 2, 32), dtype=np.float32) * 0.25
        def _inject_embeddings(self, ids, embeds, visuals, video, audio):
            if visuals is not None:
                positions = np.where(ids[0] == 18)[0].tolist()
                assert len(positions) == visuals.shape[1]
                embeds = mx.array(embeds)
                embeds[:, positions, :] = mx.array(visuals)
                self.injected.append(len(positions))
            return embeds
        def _decode_turn(self, embeds, **kwargs):
            # Run exactly the native backbone path; capture final logits rather
            # than using a synthetic sampler as a parity oracle.
            from mlx_lm.models.base import create_attention_mask, create_ssm_mask
            b = model.backbone
            attention = create_attention_mask(embeds, self._cache[b.fa_idx])
            ssm = create_ssm_mask(embeds, self._cache[b.ssm_idx])
            hidden = embeds; ci = 0
            for layer in b.layers:
                if layer.block_type in ('M', '*'):
                    hidden = layer(hidden, mask=attention if layer.block_type == '*' else ssm, cache=self._cache[ci]); ci += 1
                else: hidden = layer(hidden)
            self.logits = model.lm_head(b.norm_f(hidden))[:, -1]
            mx.eval(self.logits)
            return 'reasoning</think>answer'
    session = Session(); session.mx = mx
    return session


@pytest.mark.parametrize('template_changed,new_media', [(False, False), (False, True), (True, False)])
def test_full_history_reuses_media_and_matches_cold_native_logits(tmp_path, template_changed, new_media):
    import mlx.core as mx
    d = owner(tmp_path)
    d._session = history_session(d._session.mlx_model)
    video = {'type': 'video_url', 'video_url': {'url': 'data:video/mp4;base64,AA=='}}
    first = [{'role': 'user', 'content': [video, {'type': 'text', 'text': 'Describe.'}]}]
    extracts = []
    def extract(messages, scratch_dir):
        extracts.append(deepcopy(messages))
        return '', [], None, Path('clip.mp4')
    kwargs = dict(scratch_dir=tmp_path, extract_parts=extract, enable_thinking=True,
                  max_tokens=4, temperature=0, top_p=1)
    run_full_history(d._session, first, checkpoints=NativePrefillCheckpoints(d, first, {}, publish=True), **kwargs)
    assert d._session_l2_stats['stores'] == 1
    assert d._session.video_encodes == 1
    d._session._cache = None
    d._session.tokenizer.changed = template_changed
    next_content = [video, {'type': 'text', 'text': 'And this?'}] if new_media else 'Recall.'
    messages = first + [{'role': 'assistant', 'content': 'answer', 'reasoning_content': 'removed thinking'},
                        {'role': 'user', 'content': next_content}]
    checkpoint = NativePrefillCheckpoints(d, messages, {}, publish=True)
    run_full_history(d._session, messages, checkpoints=checkpoint, **kwargs)
    warm = d._session.logits
    assert d._session._vmlx_restored_prefix_tokens > 0 if not template_changed else d._session._vmlx_restored_prefix_tokens == 0
    assert d._session.video_encodes == 1 + int(new_media or template_changed)
    assert d._session_l2_stats['hits'] == int(not template_changed)
    cold = history_session(d._session.mlx_model)
    cold.tokenizer.changed = template_changed
    run_full_history(cold, messages, **kwargs)
    assert mx.allclose(warm, cold.logits, atol=1e-4, rtol=1e-4).item()
    # Only complete, native prefix caches went to disk; no stored reasoning.
    from mlx_lm.models.cache import load_prompt_cache
    cache, metadata = load_prompt_cache(str(d._session_l2_path), return_metadata=True)
    assert 'removed thinking' in metadata['history_json']  # Source context is preserved.
    assert metadata['kind'] == 'prompt_prefill_v1'
    assert len(cache) == 2


def test_invalid_longest_token_prefix_falls_back_to_earlier_checkpoint(tmp_path):
    d = owner(tmp_path)
    first = [{'role': 'user', 'content': 'one'}]
    second = first + [{'role': 'assistant', 'content': 'answer'}, {'role': 'user', 'content': 'two'}]
    put_checkpoint(d, first, [1, 2])
    put_checkpoint(d, second, [1, 2, 9, 9])
    checkpoint = NativePrefillCheckpoints(d, second, {}, publish=False)
    assert checkpoint.find()['source_count'] == 3
    assert checkpoint.accept(np.array([[1, 2, 3, 4, 5]])) is None
    assert checkpoint.find()['source_count'] == 1
    assert checkpoint.accept(np.array([[1, 2, 3, 4, 5]])) == 2


def test_wrong_native_cache_topology_is_rejected(tmp_path):
    from mlx_lm.models.cache import load_prompt_cache, save_prompt_cache
    d = owner(tmp_path)
    messages = [{'role': 'user', 'content': 'one'}]
    checkpoint = put_checkpoint(d, messages, [1, 2])
    cache, metadata = load_prompt_cache(str(d._session_l2_path), return_metadata=True)
    d._native_disk_store().save(checkpoint.keys[-1][1], lambda p: save_prompt_cache(str(p), list(reversed(cache)), metadata))
    assert NativePrefillCheckpoints(d, messages, {}, publish=False).find() is None
    assert d._session._cache is None


def test_complete_tool_result_checkpoint_restores_native_hybrid_state(tmp_path):
    import mlx.core as mx
    from mlx_lm.models.cache import load_prompt_cache
    d = owner(tmp_path)
    messages = [
        {'role': 'user', 'content': 'image'},
        {'role': 'assistant', 'content': '', 'tool_calls': [
            {'id': 'a', 'function': {'name': 'inspect', 'arguments': {}}}]},
        {'role': 'tool', 'tool_call_id': 'a', 'content': 'green circle'},
    ]
    checkpoint = put_checkpoint(d, messages, [1, 2, 3, 4])
    _, metadata = load_prompt_cache(str(d._session_l2_path), return_metadata=True)
    assert metadata['source_message_count'] == '3'
    next_messages = messages + [
        {'role': 'assistant', 'content': '', 'tool_calls': [
            {'id': 'b', 'function': {'name': 'inspect', 'arguments': {}}}]},
        {'role': 'tool', 'tool_call_id': 'b', 'content': 'saved'},
    ]
    restore = NativePrefillCheckpoints(d, next_messages, {}, publish=True)
    assert restore.find()['source_count'] == 3
    tokens = [1, 2, 3, 4, 5, 6]
    assert restore.accept(np.array([tokens])) == 4
    warm = d._session.mlx_model(mx.array([tokens[4:]]), cache=d._session._cache)
    cold = d._session.mlx_model(mx.array([tokens]), cache=d._session.mlx_model.make_cache())
    mx.eval(warm, cold)
    assert mx.allclose(warm, cold[:, 4:], atol=1e-4, rtol=1e-4).item()
    restore.publish(np.array([tokens]), next_messages)
    assert d._session_l2_stats['stores'] == 2
    latest = NativePrefillCheckpoints(d, next_messages, {}, publish=False)
    assert latest.find()['source_count'] == 5
    assert checkpoint.keys[-1][0] == 3
