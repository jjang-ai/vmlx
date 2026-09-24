"""Native persisted state must not survive in-place bundle changes."""
import json
import os
from pathlib import Path

import numpy as np
import pytest
from safetensors.numpy import save_file

from vmlx_engine.omni_multimodal import OmniMultimodalDispatcher as Dispatcher


@pytest.fixture
def bundle(tmp_path):
    tmp_path = tmp_path / 'bundle'
    tmp_path.mkdir()
    save_file({'vision.projection.weight': np.zeros((2, 2), dtype=np.float16)}, str(tmp_path / 'model-00001-of-00001.safetensors'))
    (tmp_path / 'model.safetensors.index.json').write_text(json.dumps({
        'weight_map': {'vision.projection.weight': 'model-00001-of-00001.safetensors'},
    }))
    (tmp_path / 'config.json').write_text('{"model_type":"nemotron_h"}')
    for name in ['tokenizer.json', 'tokenizer_config.json', 'chat_template.jinja',
                 'preprocessor_config.json', 'audio/processor_config.json']:
        p = tmp_path / name
        p.parent.mkdir(exist_ok=True)
        p.write_text('original')
    return tmp_path


def fingerprint(root, *, fresh=False):
    # Distinguish a missing artifact dependency from the legacy path-only LRU.
    fn = Dispatcher._bundle_fingerprint
    if fresh and hasattr(fn, 'cache_clear'):
        fn.cache_clear()
    return fn(root)


@pytest.mark.parametrize('name', [
    'tokenizer.json', 'tokenizer_config.json', 'chat_template.jinja',
    'preprocessor_config.json', 'audio/processor_config.json',
])
def test_prompt_and_processor_changes_invalidate_persisted_namespace(bundle, name):
    before = fingerprint(bundle, fresh=True)
    (bundle / name).write_text('modified')
    assert fingerprint(bundle, fresh=True) != before


def test_path_memoization_does_not_hide_changes_in_same_process(bundle):
    before = fingerprint(bundle, fresh=True)
    (bundle / 'config.json').write_text('{"model_type":"nemotron_h","changed":true}')
    assert fingerprint(bundle) != before


@pytest.mark.parametrize('atomic', [False, True])
def test_weight_replacement_with_unchanged_index_size_and_mtime_invalidates(bundle, atomic):
    shard = bundle / 'model-00001-of-00001.safetensors'
    original = shard.stat()
    before = fingerprint(bundle, fresh=True)
    data = bytearray(shard.read_bytes())
    data[-2:] = np.float16(1).tobytes()
    if atomic:
        replacement = bundle / 'replacement.tmp'
        replacement.write_bytes(data)
        replacement.replace(shard)
    else:
        shard.write_bytes(data)
    os.utime(shard, ns=(original.st_atime_ns, original.st_mtime_ns))
    assert shard.stat().st_size == original.st_size
    assert shard.stat().st_mtime_ns == original.st_mtime_ns
    assert fingerprint(bundle, fresh=True) != before


def test_unindexed_quantization_sidecar_changes_namespace(bundle):
    before = fingerprint(bundle, fresh=True)
    save_file({'codebook': np.ones((4,), dtype=np.float16)}, str(bundle / 'quantization.safetensors'))
    assert fingerprint(bundle, fresh=True) != before


def test_deleted_processor_changes_namespace(bundle):
    before = fingerprint(bundle, fresh=True)
    (bundle / 'audio/processor_config.json').unlink()
    assert fingerprint(bundle, fresh=True) != before


def test_linked_processor_directory_is_bound_without_following_cycles(bundle, tmp_path):
    linked = tmp_path / 'processor-source'
    linked.mkdir()
    config = linked / 'processor_config.json'
    config.write_text('original')
    (bundle / 'linked-processor').symlink_to(linked, target_is_directory=True)
    (linked / 'cycle').symlink_to(bundle, target_is_directory=True)
    before = fingerprint(bundle)
    config.write_text('changed')
    assert fingerprint(bundle) != before


def test_stable_bundle_survives_process_cache_clear_and_ignores_readme(bundle):
    before = fingerprint(bundle, fresh=True)
    (bundle / 'README.md').write_text('A new description')
    assert fingerprint(bundle, fresh=True) == before


def test_same_path_rebinds_native_owner_when_artifact_changes(bundle):
    class Owner(Dispatcher):
        _instance = None
        def __init__(self, bundle_path, **kwargs):
            self.bundle_path = bundle_path
            self._session_l2_policy = {}
            self._session_l2_fingerprint = self._bundle_fingerprint(bundle_path)
            self.closed = False
        def close(self): self.closed = True
    old = Owner.get(bundle)
    (bundle / 'chat_template.jinja').write_text('new template')
    new = Owner.get(bundle)
    assert new is not old and old.closed
    assert new._session_l2_fingerprint != old._session_l2_fingerprint


def test_fingerprinting_does_not_read_weights_or_rehash_unchanged_metadata(bundle, monkeypatch):
    original_open = Path.open
    opens = []
    def guarded_open(path, *args, **kwargs):
        assert path.suffix != '.safetensors', 'Cache identity read a weight payload'
        opens.append(str(path))
        return original_open(path, *args, **kwargs)
    monkeypatch.setattr(Path, 'open', guarded_open)
    first = fingerprint(bundle)
    assert opens
    opens.clear()
    assert fingerprint(bundle) == first
    assert not opens


def test_repaired_bundle_cannot_read_prior_native_ssd_namespace(bundle, tmp_path):
    from safetensors.numpy import load_file
    from vmlx_engine.utils.omni_session_disk_store import OmniSessionDiskStore
    root = tmp_path / 'ssd'
    old = OmniSessionDiskStore(root=root, model_key=fingerprint(bundle), max_size_bytes=1000000)
    payload = {'state': np.ones((2, 2), dtype=np.float16)}
    path = old.save('same-messages', lambda target: save_file(payload, str(target)))
    try:
        (bundle / 'chat_template.jinja').write_text('repaired template')
        new = OmniSessionDiskStore(root=root, model_key=fingerprint(bundle), max_size_bytes=1000000)
        try:
            assert new.load('same-messages', lambda p: load_file(str(p))) is None
            assert path.is_file(), 'Identity invalidation must not delete unrelated records'
            assert np.array_equal(old.load('same-messages', lambda p: load_file(str(p)))['state'], payload['state'])
        finally:
            new.close()
    finally:
        old.close()
