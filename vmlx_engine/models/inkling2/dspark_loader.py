"""Load the native BF16 Inkling head from a sidecar or indexed target shard."""

import json
from collections import defaultdict
from pathlib import Path

import mlx.core as mx

from .dspark_config import DSparkConfig
from .dspark_head import InklingDraft

_DRAFT_PREFIX = "mtp_drafter."


def load_inkling_dspark(bundle: Path, target) -> tuple[InklingDraft, str]:
    """Return the trained head and the source of its weights."""
    raw_config = json.loads((bundle / "config.json").read_text())
    text_config = dict(raw_config["text_config"])
    text_config["mtp_config"] = raw_config.get("mtp_config") or text_config.get("mtp_config")
    drafter = InklingDraft(DSparkConfig(text_config=text_config))
    drafter.validate_target_compatibility(target)

    sidecar = bundle / "mtp.safetensors"
    index_path = bundle / "model.safetensors.index.json"
    index = json.loads(index_path.read_text())["weight_map"]
    shards = defaultdict(list)
    for key, filename in index.items():
        if key.startswith(_DRAFT_PREFIX):
            shard_name = Path(filename)
            if shard_name.name != filename or shard_name.suffix != ".safetensors":
                raise ValueError(f"Invalid Inkling DSpark shard name {filename!r}")
            shards[bundle / shard_name].append(key)

    if sidecar.is_file() and shards:
        raise ValueError("The Inkling bundle specifies both sidecar and indexed DSpark weights")
    if sidecar.is_file():
        weights = list(mx.load(sidecar).items())
        source = "sidecar"
    elif shards:
        weights = []
        for shard, keys in shards.items():
            tensors = mx.load(shard)
            weights.extend(
                (key.removeprefix(_DRAFT_PREFIX), tensors[key]) for key in keys
            )
        source = "indexed"
    else:
        raise FileNotFoundError("The Inkling bundle has no native DSpark weights")
    if not weights or any(weight.dtype != mx.bfloat16 for _, weight in weights):
        raise ValueError("Inkling DSpark requires its complete native BF16 weight set")
    drafter.load_weights(weights, strict=True)
    mx.eval(drafter.parameters())
    return drafter.eval(), source
