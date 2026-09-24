import json
import os
import time

import pytest

from vmlx_engine.utils.omni_session_disk_store import OmniSessionDiskStore


def store(tmp_path, model='omni', cap=1000000, ttl=0):
    return OmniSessionDiskStore(root=tmp_path, model_key=model, max_size_bytes=cap, ttl_minutes=ttl)


def test_distinct_causal_snapshots_survive_restart_and_other_namespace(tmp_path):
    s = store(tmp_path)
    try:
        a = s.save('image-chat', lambda p: p.write_bytes(b'native-f16-and-f32-a'))
        s.save('other-chat', lambda p: p.write_bytes(b'native-f16-and-f32-b'))
        assert a.is_relative_to(tmp_path)
    finally:
        s.close()
    s = store(tmp_path)
    other = store(tmp_path, model='different-artifact')
    try:
        assert s.load('image-chat', lambda p: p.read_bytes()) == b'native-f16-and-f32-a'
        assert s.load('other-chat', lambda p: p.read_bytes()) == b'native-f16-and-f32-b'
        assert other.load('image-chat', lambda p: p.read_bytes()) is None
        assert s.load('changed-assistant', lambda p: pytest.fail('read wrong history')) is None
    finally:
        s.close(); other.close()


def test_native_snapshot_participates_in_global_lru_and_manual_clear(tmp_path):
    s = store(tmp_path, cap=400000)
    other = store(tmp_path, model='other', cap=400000)
    try:
        a = s.save('old', lambda p: p.write_bytes(b'a' * 240000))
        for path in s.paths('old'): os.utime(path, (time.time()-100,)*2)
        b = other.save('new', lambda p: p.write_bytes(b'b' * 240000))
        assert not a.exists() and b.exists()
        assert other.budget.refresh_health().bytes_after <= 400000
        other.budget.clear_eligible()
        assert not b.exists() and not b.with_suffix('.json').exists()
    finally:
        s.close(); other.close()


def test_native_snapshot_obeys_ttl_without_loading_stale_payload(tmp_path):
    s = store(tmp_path, ttl=1)
    try:
        s.save('expired', lambda p: p.write_bytes(b'old'))
        for path in s.paths('expired'): os.utime(path, (time.time()-61,)*2)
        assert s.load('expired', lambda p: pytest.fail('stale load')) is None
        s.save('fresh', lambda p: p.write_bytes(b'new'))
        assert not any(p.exists() for p in s.paths('expired'))
        assert s.load('fresh', lambda p:p.read_bytes()) == b'new'
    finally: s.close()


def test_native_snapshot_fsyncs_and_write_failure_never_publishes(tmp_path, monkeypatch):
    s = store(tmp_path)
    calls=[];real=os.fsync
    monkeypatch.setattr(os, 'fsync', lambda fd:(calls.append(fd),real(fd))[1])
    try:
        s.save('good', lambda p:p.write_bytes(b'ok'))
        assert len(calls)>=3
        def failed(path):
            path.write_bytes(b'partial')
            raise OSError('disk full')
        with pytest.raises(OSError, match='disk full'): s.save('bad', failed)
        assert not any(p.exists() for p in s.paths('bad'))
        assert not list(s.directory.glob('*.tmp.*'))
        assert s.load('good', lambda p:p.read_bytes()) == b'ok'
    finally:s.close()


def test_oversized_snapshot_refused_without_evicting_useful_entries(tmp_path):
    s = store(tmp_path, cap=10000)
    try:
        a = s.save('good', lambda p:p.write_bytes(b'ok'))
        with pytest.raises(OSError,match='exceeds configured SSD cap'):
            s.save('too-big', lambda p:p.write_bytes(b'x'*11000))
        assert a.exists()
        assert not list(s.directory.glob('*.tmp.*'))
    finally:s.close()


def test_native_snapshot_corrupt_sidecar_cannot_match_and_symlinks_rejected(tmp_path):
    s=store(tmp_path)
    try:
        s.save('good',lambda p:p.write_bytes(b'ok'))
        data,side=s.paths('good')
        side.write_text(json.dumps({'schema':'wrong','signature':'good'}))
        assert s.load('good',lambda p:pytest.fail('invalid sidecar read')) is None
        data.unlink();data.symlink_to(tmp_path/'outside')
        with pytest.raises(OSError,match='symlink'):s.save('good',lambda p:p.write_bytes(b'new'))
    finally:s.close()
