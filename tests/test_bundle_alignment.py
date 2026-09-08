"""Real temporary safetensors, no model/GPU allocation or production writes."""
import hashlib
import json
import os
from pathlib import Path
import struct
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor

import pytest

from vmlx_engine import bundle_alignment as repair
from vmlx_engine.model_bundle_integrity import (
    BundleIntegrityError, _read_safetensors_header, check_model_bundle,
)


def fixture(root: Path, name="model.safetensors") -> Path:
    path = root / name
    path.parent.mkdir(parents=True, exist_ok=True)
    header = {"__metadata__": {"format": "mlx", "quantization": "affine", "note": "日本語"}}
    data = b""
    # Physical dtypes, not model families/bits, determine alignment. Arbitrary
    # packed words and scale bytes must survive without numeric conversion.
    for key, dtype, raw in [
        ("padding", "U8", b"x"), ("mtp.scales", "F16", b"\x01\x7c"),
        ("vision.biases", "BF16", b"\x80\x7f"),
        ("experts.weight", "U32", b"\x01\x23\x45\x67"),
        ("audio.weight", "F32", struct.pack("<f", -1.5)),
        ("index", "I64", struct.pack("<q", 123)),
        ("empty", "F16", b""),
    ]:
        header[key] = {"dtype": dtype, "shape": [0 if not raw else 1],
                       "data_offsets": [len(data), len(data) + len(raw)]}
        data += raw
    encoded = json.dumps(header).encode()
    encoded += b" " * (-len(encoded) % 8)
    path.write_bytes(struct.pack("<Q", len(encoded)) + encoded + data)
    return path


def tensor_bytes(path):
    h = _read_safetensors_header(path)
    raw = path.read_bytes()
    return {k: (v["dtype"], v["shape"], raw[h["data_start"] + v["start"]:h["data_start"] + v["end"]])
            for k, v in h["tensors"].items()}


def test_mixed_nested_configless_roundtrip_and_noop(tmp_path, capsys):
    root = tmp_path / "model"
    shard = fixture(root, "vision/nested.safetensors")
    original = tensor_bytes(shard)
    metadata = _read_safetensors_header(shard)["container"]["__metadata__"]
    result = check_model_bundle(root, cache_dir=tmp_path / "stamps")
    assert result["misaligned_tensors"] == 0
    assert result["repairs"] == ["vision/nested.safetensors"]
    assert tensor_bytes(shard) == original
    assert _read_safetensors_header(shard)["container"]["__metadata__"] == metadata
    identity = repair._identity(shard)
    assert check_model_bundle(root, cache_dir=tmp_path / "stamps")["cache_hit"]
    assert repair._identity(shard) == identity
    assert not list(root.rglob("*.tmp"))
    events = [json.loads(line.split("] ", 1)[1])["stage"] for line in capsys.readouterr().err.splitlines()]
    assert events == ["MISALIGNED_DETECTED", "COPYING", "VALIDATED", "TRANSACTION_COMMITTED", "REPAIRED_ON_DISK"]


def test_large_tokenizer_metadata_is_bounded_and_not_a_hash_manifest(tmp_path):
    root = tmp_path / "model"
    shard = fixture(root)
    before = tensor_bytes(shard)
    tokenizer = root / "tokenizer.json"
    tokenizer.write_text(json.dumps({"vocab": "x" * (17 * 1024 * 1024)}))
    assert check_model_bundle(root, cache_dir=tmp_path / "stamps")["misaligned_tensors"] == 0
    assert tensor_bytes(shard) == before


@pytest.mark.parametrize("weight_format,bits,group_size", [
    ("mxtq", 2, 64), ("mxtq", 4, 32), ("affine", 3, 32),
    ("affine", 6, 128), ("affine", 8, 64),
])
def test_quantized_and_nested_media_bytes_and_configs_preserved(
    tmp_path, weight_format, bits, group_size,
):
    """Container repair must not reinterpret codebooks or quantization policy.

    A deliberately unaligned, structurally valid miniature bundle exercises the
    actual packed-word/norm/codebook dtypes, not a fabricated model inference.
    """
    root = tmp_path / "bundle"
    root.mkdir()
    configs = {
        "config.json": {"model_type": "fixture", "quantization": {
            "bits": bits, "group_size": group_size}, "vision_config": {"width": 32}},
        "jang_config.json": {"weight_format": weight_format, "mxtq_seed": 42,
            "mxtq_bits": {"routed_expert": bits, "attention": 8, "norms_router": 16}},
        "generation_config.json": {"eos_token_id": [1, 2], "temperature": 0.7},
        "processor_config.json": {"min_pixels": 1024, "max_pixels": 65536},
        "vision/config.json": {"dtype": "bfloat16"},
        "audio/config.json": {"sample_rate": 24000},
        "mtp/config.json": {"num_hidden_layers": 1},
    }
    for name, config in configs.items():
        path = root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(config, indent=2) + "\n")

    groups = {
        "model-00001-of-00001.safetensors": [
            ("expert.tq_packed", "U32", bytes.fromhex("ffffffff01234567")),
            ("expert.tq_indices", "I32", struct.pack("<ii", -1, 17)),
            ("expert.tq_norms", "F16", bytes.fromhex("007c017e")),
            ("attention.scales", "BF16", bytes.fromhex("803f807f")),
        ],
        "jangtq_runtime.safetensors": [
            ("codebook.32.2", "F32", struct.pack("<ff", -0.125, 0.375)),
            ("signs.32.42", "F32", struct.pack("<ff", -1, 1)),
        ],
        "vision/encoder.safetensors": [("vision.weight", "BF16", bytes.fromhex("803f8000"))],
        "audio/encoder.safetensors": [("audio.weight", "F32", struct.pack("<ff", -0.0, 1.5))],
        "mtp/head.safetensors": [("mtp.weight", "U32", bytes.fromhex("89abcdef01234567"))],
    }
    widths = {"U8": 1, "F16": 2, "BF16": 2, "I32": 4, "U32": 4, "F32": 4}
    original = {}
    for name, rows in groups.items():
        path = root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        header = {"__metadata__": {"format": "mlx", "weight_format": weight_format}}
        data = b""
        for key, dtype, raw in [("alignment_byte", "U8", b"x"), *rows]:
            header[key] = {"dtype": dtype, "shape": [len(raw) // widths[dtype]],
                           "data_offsets": [len(data), len(data) + len(raw)]}
            data += raw
        encoded = json.dumps(header).encode()
        encoded += b" " * (-len(encoded) % 8)
        path.write_bytes(struct.pack("<Q", len(encoded)) + encoded + data)
        assert _read_safetensors_header(path)["misaligned"]
        original[name] = tensor_bytes(path)

    index_path = root / "model.safetensors.index.json"
    index_path.write_text(json.dumps({"metadata": {"custom": "preserve"},
        "weight_map": {k: "model-00001-of-00001.safetensors"
                       for k in original["model-00001-of-00001.safetensors"]}}))
    metadata_before = {p.relative_to(root).as_posix(): p.read_bytes()
                       for p in root.rglob("*.json")}
    report = check_model_bundle(root, cache_dir=tmp_path / "stamps")
    assert set(report["repairs"]) == set(groups)
    assert report["misaligned_tensors"] == 0
    for name, expected in original.items():
        assert tensor_bytes(root / name) == expected
        assert _read_safetensors_header(root / name)["container"]["__metadata__"] == {
            "format": "mlx", "weight_format": weight_format}
    assert {name: (root / name).read_bytes() for name in metadata_before} == metadata_before
    before = {name: repair._identity(root / name) for name in groups}
    assert check_model_bundle(root, cache_dir=tmp_path / "stamps")["cache_hit"]
    assert {name: repair._identity(root / name) for name in groups} == before


def test_large_hash_manifest_is_refused_without_replacing_shard(tmp_path):
    root = tmp_path / "model"
    shard = fixture(root)
    before = shard.read_bytes()
    (root / "hashes.json").write_text(json.dumps({"padding": "x" * (17 * 1024 * 1024),
        "model.safetensors": {"sha256": "unknown"}}))
    with pytest.raises(BundleIntegrityError, match="oversized file-hash manifest"):
        check_model_bundle(root, cache_dir=tmp_path / "stamps")
    assert shard.read_bytes() == before


def test_inspection_never_rewrites_or_bypasses_later_repair(tmp_path):
    root = tmp_path / "model"
    path = fixture(root)
    before = path.read_bytes()
    inspected = check_model_bundle(root, repair=False, cache_dir=tmp_path / "cache")
    assert inspected["misaligned_tensors"] > 0
    assert path.read_bytes() == before
    assert check_model_bundle(root, cache_dir=tmp_path / "cache")["repairs"] == [path.name]


@pytest.mark.parametrize("failure", ["space", "fsync", "replace", "validation", "source_changed"])
def test_failure_preserves_original_no_success_stamp(tmp_path, monkeypatch, capsys, failure):
    root = tmp_path / "model"
    path = fixture(root)
    before = path.read_bytes()
    if failure == "space":
        monkeypatch.setattr(repair.shutil, "disk_usage", lambda _: type("Disk", (), {"free": 0})())
    elif failure == "fsync":
        monkeypatch.setattr(repair.os, "fsync", lambda _: (_ for _ in ()).throw(OSError("fsync failed")))
    elif failure == "replace":
        original = repair.os.replace
        def replace(src, dst):
            if Path(dst) == path:
                raise OSError("replace failed")
            return original(src, dst)
        monkeypatch.setattr(repair.os, "replace", replace)
    else:
        original = repair._tensor_digests
        def digests(handle, header):
            value = original(handle, header)
            if failure == "validation":
                return {}
            os.utime(path, ns=(1, 1))
            return value
        monkeypatch.setattr(repair, "_tensor_digests", digests)
    with pytest.raises(BundleIntegrityError):
        check_model_bundle(root, cache_dir=tmp_path / "cache")
    assert path.read_bytes() == before
    assert not list((tmp_path / "cache").glob("*.json"))
    assert '"stage": "REPAIRED_ON_DISK"' not in capsys.readouterr().err
    monkeypatch.undo()
    check_model_bundle(root, cache_dir=tmp_path / "cache")
    assert not list(root.rglob("*.tmp"))
    assert not (root / repair.JOURNAL).exists()


def test_hardlink_original_inode_unchanged(tmp_path):
    root = tmp_path / "model"
    path = fixture(root)
    alias = tmp_path / "original.safetensors"
    os.link(path, alias)
    original = alias.read_bytes()
    check_model_bundle(root, cache_dir=tmp_path / "cache")
    assert alias.read_bytes() == original
    assert path.stat().st_ino != alias.stat().st_ino


def test_hf_file_symlink_replaced_without_mutating_shared_blob(tmp_path):
    path = fixture(tmp_path / "original")
    root = tmp_path / "model"
    root.mkdir()
    (root / "model.safetensors").symlink_to(path)
    before = path.read_bytes()
    check_model_bundle(root, cache_dir=tmp_path / "cache")
    assert path.read_bytes() == before
    assert not (root / "model.safetensors").is_symlink()
    assert _read_safetensors_header(root / "model.safetensors")["misaligned"] == []


def test_proposal_sidecar_digest_migrated_without_tensor_changes(tmp_path):
    root = tmp_path / "model"
    path = fixture(root, "mtp_draft/head.safetensors")
    before = path.read_bytes()
    (root / "vmlx_mtp_proposal_head.json").write_text(json.dumps({"draft_artifact": {
        "file": "mtp_draft/head.safetensors", "sha256": hashlib.sha256(before).hexdigest()}}))
    tensors = tensor_bytes(path)
    check_model_bundle(root, cache_dir=tmp_path / "cache")
    stamp = json.loads((root / "vmlx_mtp_proposal_head.json").read_text())
    assert stamp["draft_artifact"]["sha256"] == hashlib.sha256(path.read_bytes()).hexdigest()
    assert tensor_bytes(path) == tensors


def test_unknown_hash_manifest_refused(tmp_path):
    root = tmp_path / "model"
    path = fixture(root)
    before = path.read_bytes()
    (root / "signed_manifest.json").write_text(json.dumps({"file": path.name, "signature": "unknown"}))
    with pytest.raises(BundleIntegrityError, match="file-hash reference"):
        check_model_bundle(root, cache_dir=tmp_path / "cache")
    assert path.read_bytes() == before


@pytest.mark.parametrize("stage", ["copy", "replace"])
def test_crash_during_copy_or_after_replace_before_stamp_recovers(tmp_path, stage):
    root = tmp_path / "model"
    path = fixture(root)
    before = tensor_bytes(path)
    stamp = root / "vmlx_mtp_proposal_head.json"
    stamp.write_text(json.dumps({"draft_artifact": {"file": path.name,
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(), "bits": 4, "group_size": 64}}))
    script = '''
import os, sys
from pathlib import Path
from vmlx_engine.model_bundle_integrity import check_model_bundle
from vmlx_engine import bundle_alignment
original = os.replace
def interrupted(src, dst):
    original(src, dst)
    if sys.argv[3] == 'replace' and Path(dst).name == 'model.safetensors': os._exit(91)
os.replace = interrupted
original_event = bundle_alignment._event
def event(stage, *args, **kwargs):
    original_event(stage, *args, **kwargs)
    if sys.argv[3] == 'copy' and stage == 'COPYING': os._exit(91)
bundle_alignment._event = event
check_model_bundle(sys.argv[1], cache_dir=sys.argv[2])
'''
    result = subprocess.run([sys.executable, "-c", script, str(root), str(tmp_path / "cache"), stage], capture_output=True)
    assert result.returncode == 91, result.stderr
    assert (root / repair.JOURNAL).exists()
    assert not list((tmp_path / "cache").glob("*.json"))
    assert tensor_bytes(path) == before
    check_model_bundle(root, cache_dir=tmp_path / "cache")
    assert json.loads(stamp.read_text())["draft_artifact"] == {
        "file": path.name, "bits": 4, "group_size": 64,
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
    assert not (root / repair.JOURNAL).exists()
    assert not list(root.rglob("*.tmp"))


def test_two_callers_with_distinct_stamp_dirs_serialize(tmp_path):
    root = tmp_path / "model"
    path = fixture(root)
    original = tensor_bytes(path)
    with ThreadPoolExecutor(2) as pool:
        results = list(pool.map(lambda n: check_model_bundle(root, cache_dir=tmp_path / f"cache{n}"), range(2)))
    assert all(r["misaligned_tensors"] == 0 for r in results)
    assert tensor_bytes(path) == original
    assert sum(path.name in r["repairs"] for r in results) == 1
