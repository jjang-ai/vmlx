"""Cheap local artifact identity for persistent native Omni state.

Payloads stay mmap-backed: weight files are identified by filesystem change
identity, not read into memory or hashed on each request. Metadata content is
hashed once per observed manifest. Conservative misses after a faithful file
replacement are preferable to reusing state from repaired/replaced weights.
"""
from functools import lru_cache
import hashlib
import os
from pathlib import Path

_WEIGHTS = {".safetensors", ".bin", ".pt", ".pth", ".npy", ".npz"}
_METADATA = {".json", ".jinja", ".py", ".model", ".tiktoken", ".bpe"}


def _file_identity(path):
    st = path.stat()
    return st.st_dev, st.st_ino, st.st_size, st.st_mtime_ns, st.st_ctime_ns


def _manifest(root):
    entries = []
    visited = set()
    for directory, folders, files in os.walk(root, followlinks=True):
        directory_stat = Path(directory).stat()
        directory_id = directory_stat.st_dev, directory_stat.st_ino
        if directory_id in visited:
            folders[:] = []
            continue
        visited.add(directory_id)
        folders[:] = [name for name in folders if not name.startswith(".") and name != "__pycache__"]
        for name in files:
            path = Path(directory) / name
            if name.startswith("._"):
                continue
            if path.suffix not in _WEIGHTS | _METADATA and name not in {"merges.txt", "vocab.txt", ".vmlx-downloading"}:
                continue
            entries.append((path.relative_to(root).as_posix(), *_file_identity(path)))
    return tuple(sorted(entries))


@lru_cache(maxsize=8)
def _digest(root, manifest):
    digest = hashlib.sha256(b"omni-native-artifact-v1\0")
    digest.update(root.encode())
    for entry in manifest:
        name, *identity = entry
        digest.update(repr(entry).encode())
        path = Path(root) / name
        if path.suffix not in _WEIGHTS:
            with path.open("rb") as handle:
                while chunk := handle.read(1024 * 1024):
                    digest.update(chunk)
            if _file_identity(path) != tuple(identity):
                raise ValueError("Omni bundle changed while computing its cache identity")
    return digest.hexdigest()[:32]


def bundle_fingerprint(bundle_path):
    root = Path(bundle_path).resolve()
    # Cache by observed files, never by path alone. ctime/inode catch repairs
    # that retain shard size, mtime and byte-identical index/metadata files.
    return _digest(str(root), _manifest(root))
