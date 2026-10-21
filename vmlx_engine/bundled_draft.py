"""Header-only admission for bundled drafters, before selecting a serving lane."""
import json
from pathlib import Path

from .model_bundle_integrity import _read_safetensors_header


def validate_bundled_draft(draft_path, target_path) -> None:
    """Reject missing/corrupt weights and incompatible target geometry.

    This is admission, not a substitute for strict runtime weight loading or
    generation verification. It never repairs or rewrites model artifacts.
    """
    draft, target = Path(draft_path), Path(target_path)
    cfg = json.loads((draft / "config.json").read_text())
    target_cfg = json.loads((target / "config.json").read_text())
    text = target_cfg.get("text_config") or target_cfg
    hidden, vocab, layers = (int(text[k]) for k in ("hidden_size", "vocab_size", "num_hidden_layers"))
    if min(hidden, vocab, layers) <= 0:
        raise ValueError("Invalid target geometry")
    if int(cfg["hidden_size"]) != hidden or not 0 < int(cfg["vocab_size"]) <= vocab:
        raise ValueError("Bundled drafter hidden size or vocabulary does not fit target")
    taps = cfg["dflash_config"]["target_layer_ids"]
    if not isinstance(taps, list) or not taps or any(type(i) is not int or not 0 <= i < layers for i in taps):
        raise ValueError("Bundled drafter target layers do not fit target")
    if "num_target_layers" in cfg and int(cfg["num_target_layers"]) != layers:
        raise ValueError("Bundled drafter target layer count does not match target")
    tensors, owners = {}, {}
    shards = sorted(p for p in draft.glob("*.safetensors") if not p.name.startswith("._"))
    if not shards:
        raise ValueError("Bundled drafter has no safetensors weights")
    for shard in shards:
        header = _read_safetensors_header(shard)["tensors"]
        if tensors.keys() & header.keys():
            raise ValueError("Duplicate bundled drafter tensors")
        tensors.update(header)
        owners.update({name: shard.name for name in header})
    fc = tensors.get("fc.weight", {})
    quant = cfg.get("quantization") or cfg.get("quantization_config") or {}
    packed = "fc.scales" in tensors
    width = hidden * len(taps)
    if packed:
        bits, group = quant.get("bits"), quant.get("group_size")
        if quant.get("mode", "affine") != "affine" or bits not in (2, 3, 4, 5, 6, 8) or group not in (32, 64, 128):
            raise ValueError("Unsupported bundled draft quantization")
        if width * bits % 32 or width % group or fc.get("dtype") != "U32":
            raise ValueError("Invalid packed bundled draft projection")
        for key in ("fc.scales", "fc.biases"):
            if tensors.get(key, {}).get("shape") != [hidden, width // group]:
                raise ValueError("Invalid bundled draft quantization scales/biases")
        width = width * bits // 32
    elif fc.get("dtype") not in ("F16", "BF16", "F32"):
        raise ValueError("Invalid floating bundled draft projection")
    if fc.get("shape") != [hidden, width]:
        raise ValueError("Bundled draft projection does not fit target taps")
    draft_layers = cfg["num_hidden_layers"]
    if type(draft_layers) is not int or not 0 < draft_layers <= len(tensors):
        raise ValueError("Invalid bundled drafter layer count")
    for i in range(draft_layers):
        if f"layers.{i}.self_attn.q_proj.weight" not in tensors:
            raise ValueError("Bundled drafter is missing attention weights")
    for index_path in draft.glob("*.safetensors.index.json"):
        mapping = json.loads(index_path.read_text())["weight_map"]
        for name, filename in mapping.items():
            if Path(filename).name != filename or owners.get(name) != filename:
                raise ValueError("Invalid bundled drafter shard index")
