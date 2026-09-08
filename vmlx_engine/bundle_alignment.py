"""Bounded, byte-preserving atomic safetensors alignment repair.

This changes container layout, never tensor values. Each shard replacement is
atomic; a multi-shard bundle is resumable, not one filesystem transaction.
"""
from __future__ import annotations

import hashlib
import copy
import json
import os
from pathlib import Path
import shutil
import stat
import struct
import sys
import tempfile
import time

CHUNK_BYTES = 4 * 1024 * 1024
JOURNAL = ".vmlx-alignment-transaction.json"


def _file_digest(path: Path) -> str:
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def _sync_directory(path: Path) -> None:
    fd = os.open(path, os.O_RDONLY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def recover_alignment_transaction(root: Path) -> None:
    """Finish a validated shard/stamp transaction after interruption, under lock."""
    from .model_bundle_integrity import BundleIntegrityError, _atomic_json_write
    journal_path = root / JOURNAL
    if not journal_path.exists():
        return
    if journal_path.is_symlink():
        raise BundleIntegrityError("alignment recovery journal must not be a symlink")
    state = json.loads(journal_path.read_text())
    if state.get("schema") != "vmlx-alignment-transaction-v1":
        raise BundleIntegrityError("unknown alignment recovery journal")

    def local(name: str) -> Path:
        relative = Path(name)
        if relative.is_absolute() or ".." in relative.parts:
            raise BundleIntegrityError("unsafe alignment recovery path")
        result = root / relative
        if not result.parent.resolve().is_relative_to(root.resolve()):
            raise BundleIntegrityError("alignment recovery path escapes bundle")
        return result

    shard = local(state["shard"])
    temporary = local(state["temporary"])
    if temporary.parent != shard.parent or not temporary.name.startswith(f".{shard.name}.vmlx-align-") or not temporary.name.endswith(".tmp"):
        raise BundleIntegrityError("invalid alignment recovery temporary name")
    digest = _file_digest(shard)
    if digest == state["new_sha256"]:
        stamp = state.get("stamp")
        if stamp:
            target = local(stamp["path"])
            expected = copy.deepcopy(stamp["before"])
            artifact = expected.get("draft_artifact", {})
            if (target != root / "vmlx_mtp_proposal_head.json"
                or artifact.get("file") != state["shard"]
                or artifact.get("sha256") != state["old_sha256"]):
                raise BundleIntegrityError("unsupported alignment recovery stamp")
            artifact["sha256"] = state["new_sha256"]
            if stamp["after"] != expected:
                raise BundleIntegrityError("alignment recovery would change non-hash metadata")
            if target.is_symlink():
                raise BundleIntegrityError("alignment stamp must not be a symlink")
            current = json.loads(target.read_text())
            if current not in (stamp["before"], stamp["after"]):
                raise BundleIntegrityError("alignment stamp changed during repair; recovery refused")
            if current != stamp["after"]:
                _atomic_json_write(target, stamp["after"])
                _sync_directory(target.parent)
        _sync_directory(shard.parent)
        _event("TRANSACTION_COMMITTED", shard, sha256=digest)
    elif digest == state["old_sha256"]:
        _event("INTERRUPTED_COPY_DISCARDED", shard, original_preserved=True)
    else:
        raise BundleIntegrityError("alignment recovery found an externally changed shard; original state cannot be inferred")
    temporary.unlink(missing_ok=True)
    journal_path.unlink()
    _sync_directory(root)


def _event(stage: str, path: Path, **fields) -> None:
    print("[BUNDLE-ALIGNMENT] " + json.dumps(
        {"stage": stage, "shard": str(path), **fields}, sort_keys=True
    ), file=sys.stderr, flush=True)


def _identity(path: Path) -> tuple[int, ...]:
    s = path.stat()
    link = path.lstat()
    return (s.st_dev, s.st_ino, s.st_size, s.st_mtime_ns, s.st_ctime_ns,
            link.st_ino, link.st_mtime_ns, link.st_ctime_ns)


def _tensor_digests(handle, header: dict) -> dict[str, str]:
    result = {}
    for key, tensor in header["tensors"].items():
        handle.seek(header["data_start"] + tensor["start"])
        remaining = tensor["nbytes"]
        digest = hashlib.sha256()
        while remaining:
            chunk = handle.read(min(CHUNK_BYTES, remaining))
            if not chunk:
                raise OSError(f"unexpected end of tensor {key!r}")
            digest.update(chunk)
            remaining -= len(chunk)
        result[key] = digest.hexdigest()
    return result


def _digest_bound_stamp(root: Path, path: Path) -> dict | None:
    """A container rewrite cannot silently invalidate a signed/file-hash contract.

    Only the explicit local proposal-head stamp has a supported migration.
    Unknown signatures/hash manifests remain fail-closed.
    """
    from .model_bundle_integrity import BundleIntegrityError

    relative = path.relative_to(root).as_posix()
    stamp = None
    for metadata in root.rglob("*.json"):
        if metadata.name.startswith("._") or metadata.name == JOURNAL or metadata.name.endswith(".safetensors.index.json"):
            continue
        if metadata.stat().st_size > 16 * 1024 * 1024:
            # Tokenizer vocabularies can be large without referring to weight
            # hashes. Scan bounded text chunks first; never reject those solely
            # because of their size, or load an unbounded manifest into RAM.
            filename_seen = hash_seen = False
            tail = ""
            overlap = max(len(relative), len(path.name), 16)
            with metadata.open(encoding="utf-8") as stream:
                while chunk := stream.read(64 * 1024):
                    text = tail + chunk
                    filename_seen |= relative in text or path.name in text
                    hash_seen |= any(word in text.lower() for word in (
                        "sha256", "sha512", '"checksum"', '"signature"'))
                    if filename_seen and hash_seen:
                        raise BundleIntegrityError(f"{metadata}: oversized file-hash manifest requires explicit migration")
                    tail = text[-overlap:]
            continue
        raw = metadata.read_text(encoding="utf-8")
        if (relative in raw or path.name in raw) and any(
            word in raw.lower() for word in ("sha256", "sha512", '"checksum"', '"signature"')
        ):
            value = json.loads(raw)
            artifact = value.get("draft_artifact", {})
            if (metadata == root / "vmlx_mtp_proposal_head.json"
                and not metadata.is_symlink()
                and isinstance(artifact, dict) and artifact.get("file") == relative
                and set(k for k in artifact if "sha" in k or "checksum" in k) == {"sha256"}
                and not any("signature" in k.lower() for k in value)):
                if artifact.get("sha256") != _file_digest(path):
                    raise BundleIntegrityError(f"{path}: existing proposal-head digest mismatch; cannot repair by blessing it")
                stamp = {"path": metadata.relative_to(root).as_posix(), "before": value}
                continue
            raise BundleIntegrityError(
                f"{path}: alignment repair refused: file-hash reference in {metadata}; "
                "original preserved (hash metadata requires explicit migration)"
            )
    return stamp


def repair_shard_alignment(root: Path, path: Path, header: dict) -> dict:
    from .model_bundle_integrity import (
        BundleIntegrityError, _DTYPE_BYTES, _read_safetensors_header,
    )

    before = _identity(path)
    header = _read_safetensors_header(path)
    if _identity(path) != before:
        raise BundleIntegrityError(f"{path}: shard changed during header validation")
    if not header["misaligned"]:
        return header
    started = time.monotonic()
    _event("MISALIGNED_DETECTED", path, count=len(header["misaligned"]),
           examples=header["misaligned"][:3])
    temporary: Path | None = None
    replaced = False
    try:
        # HF snapshots use file symlinks to shared blobs: read their target,
        # replace the LINK, never the shared blob. Refuse symlinked directories
        # escaping the bundle; the publication parent must belong to this root.
        if not path.parent.resolve().is_relative_to(root.resolve()):
            raise BundleIntegrityError(f"{path}: publication directory escapes the bundle")
        original_stat = path.stat()
        if not stat.S_ISREG(original_stat.st_mode):
            raise BundleIntegrityError(f"{path}: not a regular shard")
        unknown = {t["dtype"] for t in header["tensors"].values()
                   if t["dtype"].upper() not in _DTYPE_BYTES}
        if unknown:
            raise BundleIntegrityError(f"{path}: unknown alignment contract for dtypes {sorted(unknown)}")
        stamp = _digest_bound_stamp(root, path)
        # Largest natural alignment first makes a contiguous payload aligned
        # without inserting forbidden holes between tensor ranges.
        keys = sorted(header["tensors"], key=lambda k: (
            -_DTYPE_BYTES[header["tensors"][k]["dtype"].upper()], k
        ))
        container = {}
        if "__metadata__" in header["container"]:
            container["__metadata__"] = header["container"]["__metadata__"]
        offset = 0
        for key in keys:
            tensor = header["tensors"][key]
            entry = dict(header["container"][key])
            entry["data_offsets"] = [offset, offset + tensor["nbytes"]]
            container[key] = entry
            offset += tensor["nbytes"]
        encoded = json.dumps(container, ensure_ascii=False, separators=(",", ":")).encode()
        encoded += b" " * (-len(encoded) % 8)
        needed = 8 + len(encoded) + offset
        if shutil.disk_usage(path.parent).free < needed + CHUNK_BYTES:
            raise BundleIntegrityError(f"{path}: alignment repair needs {needed + CHUNK_BYTES} free bytes beside the original")
        fd, name = tempfile.mkstemp(prefix=f".{path.name}.vmlx-align-", suffix=".tmp", dir=path.parent)
        temporary = Path(name)
        # Publish intent before copying so even SIGKILL during the copy leaves
        # an owned, discoverable temp that recovery can discard safely.
        from .model_bundle_integrity import _atomic_json_write
        transaction = {
            "schema": "vmlx-alignment-transaction-v1",
            "shard": path.relative_to(root).as_posix(),
            "temporary": temporary.relative_to(root).as_posix(),
            "old_sha256": _file_digest(path), "new_sha256": None, "stamp": None,
        }
        try:
            _atomic_json_write(root / JOURNAL, transaction)
            _sync_directory(root)
        except Exception:
            os.close(fd)
            raise
        _event("COPYING", path, bytes=needed, original_preserved=True)
        original_hashes = {}
        copied = 0
        progress_at = time.monotonic()
        with path.open("rb") as source, os.fdopen(fd, "wb") as destination:
            destination.write(struct.pack("<Q", len(encoded)))
            destination.write(encoded)
            for key in keys:
                tensor = header["tensors"][key]
                source.seek(header["data_start"] + tensor["start"])
                remaining = tensor["nbytes"]
                digest = hashlib.sha256()
                while remaining:
                    chunk = source.read(min(CHUNK_BYTES, remaining))
                    if not chunk:
                        raise OSError(f"unexpected end of tensor {key!r}")
                    destination.write(chunk)
                    digest.update(chunk)
                    remaining -= len(chunk)
                    copied += len(chunk)
                    if time.monotonic() - progress_at >= 5:
                        _event("COPYING", path, copied_bytes=copied, payload_bytes=offset)
                        progress_at = time.monotonic()
                original_hashes[key] = digest.hexdigest()
            destination.flush()
            os.fchmod(destination.fileno(), stat.S_IMODE(original_stat.st_mode))
            os.fsync(destination.fileno())
        candidate = _read_safetensors_header(temporary)
        if candidate["misaligned"] or set(candidate["tensors"]) != set(header["tensors"]):
            raise BundleIntegrityError(f"{path}: candidate alignment/key validation failed")
        if candidate["container"].get("__metadata__") != header["container"].get("__metadata__"):
            raise BundleIntegrityError(f"{path}: candidate metadata changed")
        for key, tensor in header["tensors"].items():
            if any(candidate["tensors"][key][field] != tensor[field]
                   for field in ("dtype", "shape", "nbytes")):
                raise BundleIntegrityError(f"{path}: candidate tensor descriptor changed: {key}")
        with temporary.open("rb") as validated:
            if _tensor_digests(validated, candidate) != original_hashes:
                raise BundleIntegrityError(f"{path}: candidate tensor-byte validation failed")
        if _identity(path) != before:
            raise BundleIntegrityError(f"{path}: original changed during repair; refusing replacement")
        _event("VALIDATED", path, tensors=len(keys), payload_bytes=copied)
        new_digest = _file_digest(temporary)
        if stamp:
            stamp["after"] = copy.deepcopy(stamp["before"])
            stamp["after"]["draft_artifact"]["sha256"] = new_digest
        transaction.update(new_sha256=new_digest, stamp=stamp)
        if _identity(path) != before:
            raise BundleIntegrityError(f"{path}: original changed during validation")
        _atomic_json_write(root / JOURNAL, transaction)
        _sync_directory(root)
        # Check directory fsync support before irreversible publication as well.
        directory_fd = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory_fd)
            os.replace(temporary, path)
            replaced = True
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
        recover_alignment_transaction(root)
        _event("REPAIRED_ON_DISK", path, tensors=len(keys), bytes=needed,
               elapsed_seconds=round(time.monotonic() - started, 3))
        candidate["relative_path"] = path.name
        return candidate
    except Exception as exc:
        _event("REPAIR_FAILED", path, replacement_published=replaced, error=str(exc))
        raise BundleIntegrityError(f"{path}: alignment repair failed: {exc}") from exc
    finally:
        # A prepared journal owns the temp until recovery resolves whether the
        # shard was replaced. Before that, failed copies can be removed now.
        if temporary is not None and not (root / JOURNAL).exists():
            temporary.unlink(missing_ok=True)
