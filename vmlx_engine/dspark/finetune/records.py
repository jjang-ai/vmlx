"""On-disk training records for DSpark fine-tuning: sharded, bounded memory, resumable.

A RECORD is one sequence the target ran over (absolute positions 0..n-1):

- ``tokens``  int32 [n]  the token ids the target consumed;
- ``labels``  int32 [n]  ``labels[j]`` = the token the target produces at position j given positions < j (its greedy
  pick from row j - 1; for a greedy generation, the generated token itself);
- ``gen``     uint8 [n]  1 where the token was produced by the target in serving (a generated continuation, or an
  assistant span of a teacher-forced conversation): anchors and labels are drawn from these positions only;
- ``taps``    bf16 [n - tap_start, n_taps * hidden]  the residual stream after each drafter tap layer, concatenated in
  the drafter's ``target_layer_ids`` order, for positions ``tap_start .. n-1`` (positions before ``tap_start`` are
  older than the drafter's sliding window needs and are not stored);
- ``topk_ids`` int32 / ``topk_logprobs`` float16 [n, K]  optional: the target's top-K at each row (soft targets).

Shards are plain safetensors files (each array of all records concatenated) plus a JSON index of record offsets.
Writing: records are buffered up to ``shard_bytes``, then the shard is written to a temporary name, renamed, its index
written, and the record ids appended to ``manifest.jsonl`` (the resume point). A crash loses at most the open buffer.
Reading: numpy memory maps, so a training step reads only the rows it uses.
"""

from __future__ import annotations

import json
import os
import struct
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Optional

import numpy as np

from . import FORMAT

_DT = {"BF16": (np.uint16, 2), "F16": (np.float16, 2), "F32": (np.float32, 4), "I32": (np.int32, 4),
       "U8": (np.uint8, 1), "F8_E4M3": (np.uint8, 1), "I64": (np.int64, 8), "U16": (np.uint16, 2)}


# ----------------------------------------------------------------------------------------------- safetensors I/O
def write_safetensors(path: str | Path, tensors: dict, *, metadata: Optional[dict] = None, order=None) -> int:
    """Write ``{name: (dtype_tag, np.ndarray)}`` as one safetensors file, data contiguous in ``order`` (default:
    sorted names), header padded to 8 bytes. ``dtype_tag`` is the safetensors tag (``BF16`` arrays are passed as their
    uint16 bit patterns, ``F8_E4M3`` as uint8 bytes). Returns the data size in bytes. Written to ``path`` directly; the
    caller handles atomic renames."""
    names = list(order) if order is not None else sorted(tensors)
    header, offset = {}, 0
    if metadata is not None:
        header["__metadata__"] = {str(k): str(v) for k, v in metadata.items()}
    for name in names:
        tag, arr = tensors[name]
        arr = np.ascontiguousarray(arr)
        if arr.dtype != _DT[tag][0]:
            raise TypeError(f"{name}: {tag} expects {_DT[tag][0]}, got {arr.dtype}")
        header[name] = {"dtype": tag, "shape": list(arr.shape), "data_offsets": [offset, offset + arr.nbytes]}
        offset += arr.nbytes
    raw = json.dumps(header, separators=(",", ":")).encode()
    raw += b" " * ((8 - len(raw) % 8) % 8)
    with open(path, "wb") as f:
        f.write(struct.pack("<Q", len(raw)))
        f.write(raw)
        for name in names:
            f.write(np.ascontiguousarray(tensors[name][1]).tobytes())
    return offset


def read_safetensors_header(path: str | Path) -> tuple[dict, int]:
    """(tensor specs without ``__metadata__``, byte offset of the data section)."""
    with open(path, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        header = json.loads(f.read(n))
    header.pop("__metadata__", None)
    return header, 8 + n


class SafetensorsMap:
    """Memory-mapped read access to one safetensors file: ``arr(name)`` is a numpy view (BF16 as uint16 bits)."""

    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.header, self.base = read_safetensors_header(self.path)
        self._mm = np.memmap(self.path, dtype=np.uint8, mode="r")

    def arr(self, name: str) -> np.ndarray:
        spec = self.header[name]
        dt, _ = _DT[spec["dtype"]]
        a, b = spec["data_offsets"]
        return self._mm[self.base + a:self.base + b].view(dt).reshape(spec["shape"])


# ----------------------------------------------------------------------------------------------- records
@dataclass
class Record:
    id: str
    n: int                      # positions 0..n-1
    tap_start: int              # first position with a stored tap row
    shard: str = ""
    tok_off: int = 0            # offset into the shard's per-position arrays
    tap_off: int = 0            # offset into the shard's tap rows
    domain: str = ""
    set: str = ""
    mode: str = ""
    n_gen: int = 0
    extra: dict = field(default_factory=dict)

    def index_entry(self) -> dict:
        return {"id": self.id, "n": self.n, "tap_start": self.tap_start, "tok_off": self.tok_off,
                "tap_off": self.tap_off, "domain": self.domain, "set": self.set, "mode": self.mode,
                "n_gen": self.n_gen, "extra": self.extra}


class ShardWriter:
    """Buffered, resumable record writer (see the module docstring)."""

    def __init__(self, out_dir: str | Path, meta: dict, *, shard_bytes: int = 1 << 30):
        self.dir = Path(out_dir)
        (self.dir / "shards").mkdir(parents=True, exist_ok=True)
        meta_path = self.dir / "meta.json"
        meta = {"format": FORMAT, **meta}
        if meta_path.is_file():
            old = json.loads(meta_path.read_text())
            keys = ("format", "taps", "hidden", "tap_dtype", "topk", "vocab_size")
            clash = {k: (old.get(k), meta.get(k)) for k in keys if old.get(k) != meta.get(k)}
            if clash:
                raise ValueError(f"{self.dir} holds records of another layout: {clash}")
        else:
            meta_path.write_text(json.dumps(meta, indent=1))
        self.meta = meta
        self.width = int(meta["hidden"]) * len(meta["taps"])
        self.topk = int(meta.get("topk") or 0)
        self.shard_bytes = int(shard_bytes)
        self.manifest = self.dir / "manifest.jsonl"
        self._buf: list[dict] = []
        self._bytes = 0
        self._next = len(list((self.dir / "shards").glob("shard-*.json")))

    def done_ids(self) -> set:
        if not self.manifest.is_file():
            return set()
        return {json.loads(line)["id"] for line in self.manifest.read_text().splitlines() if line.strip()}

    def add(self, rec: Record, *, tokens, labels, gen, taps, picks=None, topk_ids=None, topk_logprobs=None) -> None:
        n = int(rec.n)
        tokens, labels = np.asarray(tokens, np.int32), np.asarray(labels, np.int32)
        gen = np.asarray(gen, np.uint8)
        picks = tokens * 0 - 1 if picks is None else np.asarray(picks, np.int32)
        taps = np.asarray(taps)
        if taps.dtype != np.uint16:
            raise TypeError("taps must be bf16 bit patterns (uint16)")
        if not (len(tokens) == len(labels) == len(gen) == len(picks) == n):
            raise ValueError(f"record {rec.id}: per-position arrays disagree with n={n}")
        if taps.shape != (n - rec.tap_start, self.width):
            raise ValueError(f"record {rec.id}: taps {taps.shape} != {(n - rec.tap_start, self.width)}")
        item = {"rec": rec, "tokens": tokens, "labels": labels, "gen": gen, "picks": picks, "taps": taps}
        if self.topk:
            item["topk_ids"] = np.asarray(topk_ids, np.int32).reshape(n, self.topk)
            item["topk_logprobs"] = np.asarray(topk_logprobs, np.float16).reshape(n, self.topk)
        self._buf.append(item)
        self._bytes += taps.nbytes + 13 * n + self.topk * 6 * n
        if self._bytes >= self.shard_bytes:
            self.flush()

    def flush(self) -> Optional[Path]:
        if not self._buf:
            return None
        name = f"shard-{self._next:05d}"
        tok_off = tap_off = 0
        index = []
        for item in self._buf:
            rec = item["rec"]
            rec.shard, rec.tok_off, rec.tap_off = name + ".safetensors", tok_off, tap_off
            tok_off += rec.n
            tap_off += rec.n - rec.tap_start
            index.append(rec.index_entry())
        cat = lambda k: np.concatenate([it[k] for it in self._buf])  # noqa: E731
        tensors = {"tokens": ("I32", cat("tokens")), "labels": ("I32", cat("labels")), "gen": ("U8", cat("gen")),
                   "picks": ("I32", cat("picks")), "taps": ("BF16", cat("taps"))}
        if self.topk:
            tensors["topk_ids"] = ("I32", cat("topk_ids"))
            tensors["topk_logprobs"] = ("F16", cat("topk_logprobs"))
        final = self.dir / "shards" / f"{name}.safetensors"
        tmp = final.with_suffix(".safetensors.tmp")
        write_safetensors(tmp, tensors, metadata={"format": FORMAT})
        os.replace(tmp, final)
        (self.dir / "shards" / f"{name}.json").write_text(json.dumps({"records": index}))
        with open(self.manifest, "a") as f:
            for entry in index:
                f.write(json.dumps({"id": entry["id"], "shard": name, "n": entry["n"], "n_gen": entry["n_gen"],
                                    "t": round(time.time(), 1)}) + "\n")
        self._next += 1
        self._buf, self._bytes = [], 0
        return final

    def close(self) -> None:
        self.flush()


class Dataset:
    """Every record of one or more data directories (``data_dir``: a path, a comma-separated string or a list; e.g.
    generated on two machines), optionally only some ``sets`` / ``domains`` / ``ids`` / ``modes``, memory-mapped.
    All directories must share the record layout (taps, hidden size, dtype, top-K)."""

    def __init__(self, data_dir, *, sets: Optional[Iterable[str]] = None,
                 domains: Optional[Iterable[str]] = None, ids: Optional[Iterable[str]] = None,
                 modes: Optional[Iterable[str]] = None):
        if isinstance(data_dir, (list, tuple)):
            dirs = [Path(d) for d in data_dir]
        else:
            dirs = [Path(d) for d in str(data_dir).split(",") if d]
        self.dirs = dirs
        self.dir = dirs[0]
        self.meta = json.loads((self.dir / "meta.json").read_text())
        for d in dirs[1:]:
            other = json.loads((d / "meta.json").read_text())
            clash = {k: (self.meta.get(k), other.get(k)) for k in ("taps", "hidden", "tap_dtype", "topk", "vocab_size")
                     if self.meta.get(k) != other.get(k)}
            if clash:
                raise ValueError(f"{d} holds records of another layout than {self.dir}: {clash}")
        self.width = int(self.meta["hidden"]) * len(self.meta["taps"])
        self.topk = int(self.meta.get("topk") or 0)
        sets = set(sets) if sets else None
        domains = set(domains) if domains else None
        ids = set(ids) if ids else None
        modes = set(modes) if modes else None
        self.records: list[Record] = []
        self._maps: dict[str, SafetensorsMap] = {}
        seen = set()
        for d in dirs:
            for idx in sorted((d / "shards").glob("shard-*.json")):
                for e in json.loads(idx.read_text())["records"]:
                    if sets and e["set"] not in sets or domains and e["domain"] not in domains \
                            or ids and e["id"] not in ids or modes and e["mode"] not in modes:
                        continue
                    if e["id"] in seen:
                        raise ValueError(f"record {e['id']} appears twice across {dirs}")
                    seen.add(e["id"])
                    self.records.append(Record(id=e["id"], n=e["n"], tap_start=e["tap_start"],
                                               shard=str(d / "shards" / (idx.stem + ".safetensors")),
                                               tok_off=e["tok_off"], tap_off=e["tap_off"], domain=e["domain"],
                                               set=e["set"], mode=e["mode"], n_gen=e["n_gen"],
                                               extra=e.get("extra") or {}))

    def _map(self, rec: Record) -> SafetensorsMap:
        m = self._maps.get(rec.shard)
        if m is None:
            m = self._maps[rec.shard] = SafetensorsMap(rec.shard)
        return m

    def per_position(self, rec: Record, name: str) -> np.ndarray:
        return self._map(rec).arr(name)[rec.tok_off:rec.tok_off + rec.n]

    def tokens(self, rec: Record) -> np.ndarray:
        return self.per_position(rec, "tokens")

    def labels(self, rec: Record) -> np.ndarray:
        return self.per_position(rec, "labels")

    def gen(self, rec: Record) -> np.ndarray:
        return self.per_position(rec, "gen")

    def taps(self, rec: Record, a: int, b: int) -> np.ndarray:
        """Tap rows of absolute positions [a, b) (bf16 bits, uint16)."""
        if not rec.tap_start <= a <= b <= rec.n:
            raise IndexError(f"{rec.id}: taps [{a}, {b}) outside the stored rows [{rec.tap_start}, {rec.n})")
        base = rec.tap_off - rec.tap_start
        return self._map(rec).arr("taps")[base + a:base + b]

    def summary(self) -> dict:
        by = {}
        for r in self.records:
            d = by.setdefault(r.domain or "?", {"records": 0, "tokens": 0, "gen": 0})
            d["records"] += 1
            d["tokens"] += r.n
            d["gen"] += r.n_gen
        return {"records": len(self.records), "gen": sum(r.n_gen for r in self.records), "by_domain": by}


def anchor_positions(gen: np.ndarray) -> np.ndarray:
    """Positions p that can anchor a draft round: the target produced the token at p (it is a committed output) and
    the next one (there is at least one label)."""
    g = np.asarray(gen, bool)
    if len(g) < 2:
        return np.zeros(0, np.int64)
    return np.nonzero(g[:-1] & g[1:])[0]
