"""Block payload files are written with gathered writes; bytes and order are unchanged, short writes resume."""

import os

import numpy as np
import pytest

from vmlx_engine.block_disk_store import BlockDiskStore


def _buffers():
    rng = np.random.default_rng(0)
    return [
        b"\x08\x00\x00\x00\x00\x00\x00\x00",
        b'{"a":1}   ',
        rng.integers(0, 2**16, size=(3, 5, 7), dtype=np.uint16),
        np.zeros((0, 4), dtype=np.float32),  # zero-element tensor
        rng.standard_normal((2, 9)).astype(np.float16),
        rng.standard_normal((1, 1, 64, 32)).astype(np.float32),
    ]


def _expected(buffers):
    return b"".join(bytes(memoryview(b).cast("B")) if memoryview(b).nbytes else b"" for b in buffers)


@pytest.mark.parametrize("cap", [None, 1, 7, 100, 4096])
def test_writev_writes_every_byte_in_order_even_with_short_writes(tmp_path, monkeypatch, cap):
    real = os.writev

    def short_writev(fd, bufs):
        if cap is None:
            return real(fd, bufs)
        data = b"".join(bytes(b) for b in bufs)[:cap]
        return os.write(fd, data)

    monkeypatch.setattr(os, "writev", short_writev)
    path = tmp_path / "f.bin"
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    try:
        BlockDiskStore._writev_all_fd(fd, _buffers())
    finally:
        os.close(fd)
    assert path.read_bytes() == _expected(_buffers())


def test_many_buffers_beyond_iov_max(tmp_path):
    bufs = [np.full((3,), i % 251, dtype=np.uint8) for i in range(3000)]
    path = tmp_path / "g.bin"
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    try:
        BlockDiskStore._writev_all_fd(fd, bufs)
    finally:
        os.close(fd)
    assert path.read_bytes() == b"".join(b.tobytes() for b in bufs)
