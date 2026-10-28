"""Header-only admission for bundled drafters, before selecting a serving lane."""
import json
from pathlib import Path

from .model_bundle_integrity import _read_safetensors_header

_FLOAT = ("F16", "BF16", "F32")
_E4M3 = ("U8", "F8_E4M3")


def _draft_geometry(cfg: dict) -> dict:
    """hidden / vocab / taps / layer counts of a drafter config: the flat DFlash / DSpark layout (MiniCPM-V-4.7,
    SpecForge) or the MiniMax-M3.1 ``DSparkMiniMaxDraftModel`` layout (geometry under ``text_config``, taps in
    ``dspark_target_layer_ids``)."""
    from .dspark.drafter import is_minimax_config

    if is_minimax_config(cfg):
        text = cfg.get("text_config") or cfg
        dsc = text.get("dspark_config") or {}
        taps = text.get("dspark_target_layer_ids", dsc.get("target_layer_ids"))
        return {"hidden": text["hidden_size"], "vocab": text["vocab_size"], "taps": taps,
                "layers": text["num_hidden_layers"], "minimax": True,
                "num_target_layers": text.get("num_target_layers", cfg.get("num_target_layers"))}
    return {"hidden": cfg["hidden_size"], "vocab": cfg["vocab_size"], "taps": cfg["dflash_config"]["target_layer_ids"],
            "layers": cfg["num_hidden_layers"], "minimax": False, "num_target_layers": cfg.get("num_target_layers")}


def _fc_entry(quant: dict) -> dict:
    """The quantization entry of the drafter's ``fc``: its per-module entry (mlx-lm's per-module form, keyed by the
    live module path) when present, else the block's top-level group_size / bits / mode."""
    own = quant.get("fc")
    if isinstance(own, dict):
        return own
    return {k: quant[k] for k in ("group_size", "bits", "mode") if k in quant}


def _check_fc(tensors: dict, quant: dict, hidden: int, width: int) -> None:
    fc = tensors.get("fc.weight", {})
    if "fc.weight_scale_inv" in tensors:
        # MXFP8 as released: e4m3 bytes + one e8m0 scale per 32 inputs.
        if fc.get("dtype") not in _E4M3 or fc.get("shape") != [hidden, width] or width % 32:
            raise ValueError("Invalid MXFP8 bundled draft projection")
        if tensors["fc.weight_scale_inv"].get("shape") != [hidden, width // 32]:
            raise ValueError("Invalid bundled draft MXFP8 scales")
        return
    if "fc.scales" in tensors:
        entry = _fc_entry(quant)
        mode = entry.get("mode", "affine")
        bits, group = entry.get("bits"), entry.get("group_size")
        if mode == "mxfp8":
            if bits not in (None, 8) or group not in (None, 32) or width % 32 or fc.get("dtype") != "U32":
                raise ValueError("Invalid packed MXFP8 bundled draft projection")
            if tensors["fc.scales"].get("shape") != [hidden, width // 32] or "fc.biases" in tensors:
                raise ValueError("Invalid bundled draft MXFP8 scales")
            width = width * 8 // 32
        elif mode == "affine":
            if bits not in (2, 3, 4, 5, 6, 8) or group not in (32, 64, 128):
                raise ValueError("Unsupported bundled draft quantization")
            if width * bits % 32 or width % group or fc.get("dtype") != "U32":
                raise ValueError("Invalid packed bundled draft projection")
            for key in ("fc.scales", "fc.biases"):
                if tensors.get(key, {}).get("shape") != [hidden, width // group]:
                    raise ValueError("Invalid bundled draft quantization scales/biases")
            width = width * bits // 32
        else:
            raise ValueError("Unsupported bundled draft quantization")
    elif fc.get("dtype") not in _FLOAT:
        raise ValueError("Invalid floating bundled draft projection")
    if fc.get("shape") != [hidden, width]:
        raise ValueError("Bundled draft projection does not fit target taps")


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
    geo = _draft_geometry(cfg)
    if int(geo["hidden"]) != hidden or not 0 < int(geo["vocab"]) <= vocab:
        raise ValueError("Bundled drafter hidden size or vocabulary does not fit target")
    taps = geo["taps"]
    if not isinstance(taps, list) or not taps or any(type(i) is not int or not 0 <= i < layers for i in taps):
        raise ValueError("Bundled drafter target layers do not fit target")
    if geo["num_target_layers"] is not None and int(geo["num_target_layers"]) != layers:
        raise ValueError("Bundled drafter target layer count does not match target")
    tensors, owners = {}, {}
    shards = sorted(p for p in draft.glob("*.safetensors") if not p.name.startswith("._"))
    if not shards:
        raise ValueError("Bundled drafter has no safetensors weights")
    rename = (lambda n: n)
    if geo["minimax"]:
        from .dspark.drafter import minimax_tensor_name as rename
    for shard in shards:
        header = _read_safetensors_header(shard)["tensors"]
        names = {rename(name): spec for name, spec in header.items()}
        if len(names) != len(header) or tensors.keys() & names.keys():
            raise ValueError("Duplicate bundled drafter tensors")
        tensors.update(names)
        owners.update({name: shard.name for name in header})
    quant = cfg.get("quantization") or cfg.get("quantization_config") or {}
    _check_fc(tensors, quant, hidden, hidden * len(taps))
    draft_layers = geo["layers"]
    if type(draft_layers) is not int or not 0 < draft_layers <= len(tensors):
        raise ValueError("Invalid bundled drafter layer count")
    for i in range(draft_layers):
        if f"layers.{i}.self_attn.q_proj.weight" not in tensors:
            raise ValueError("Bundled drafter is missing attention weights")
    for index_path in draft.glob("*.safetensors.index.json"):
        if index_path.name.startswith("._"):
            continue
        mapping = json.loads(index_path.read_text())["weight_map"]
        for name, filename in mapping.items():
            if Path(filename).name != filename or owners.get(name) != filename:
                raise ValueError("Invalid bundled drafter shard index")
