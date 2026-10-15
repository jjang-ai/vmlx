"""Identify the measured MiMo mixed-format bundle without loading a runtime.

Preserved media tensors do not establish that a serving adapter is available.
In particular, capability discovery must not import the legacy MiMo adapter:
that registration also patches shared mlx-lm classes process-wide.
"""

from __future__ import annotations

import json
import os
import hashlib
from pathlib import Path


def canonicalize_mimo_v26_tool_results(messages: list[dict]) -> list[dict]:
    """Preserve ID associations through the native template's positional format.

    The vendor template renders neither call IDs nor result IDs. Complete
    ID-bearing result batches can be reordered without altering their meaning.
    Partial/unknown/duplicate IDs cannot be represented safely and are rejected.
    Native histories with no result-ID fields retain their supplied order.
    """
    output = []
    calls = []
    index = 0
    while index < len(messages):
        message = messages[index]
        if message.get("role") != "tool":
            output.append(message)
            if message.get("role") == "assistant":
                calls = message.get("tool_calls") or []
            elif message.get("role") not in ("system", "developer"):
                calls = []
            index += 1
            continue
        end = index + 1
        while end < len(messages) and messages[end].get("role") == "tool":
            end += 1
        results = messages[index:end]
        if any("tool_call_id" in result for result in results):
            call_ids = [call.get("id") for call in calls if isinstance(call, dict)]
            result_ids = [result.get("tool_call_id") for result in results]
            valid_calls = (
                len(call_ids) == len(calls) and bool(call_ids)
                and all(isinstance(value, str) and value for value in call_ids)
                and len(set(call_ids)) == len(call_ids)
            )
            valid_results = (
                all(isinstance(value, str) and value for value in result_ids)
                and len(set(result_ids)) == len(result_ids)
            )
            if not (valid_calls and valid_results and set(result_ids) == set(call_ids)):
                raise ValueError(
                    "MiMo-V2.6 requires a complete, unique tool-result batch "
                    "matching the preceding assistant tool-call IDs; its native "
                    "template cannot represent ambiguous or partial ID associations"
                )
            by_id = dict(zip(result_ids, results))
            results = [by_id[call_id] for call_id in call_ids]
        output.extend(results)
        calls = []
        index = end
    return output


def mimo_v26_runtime_source_identity(files: dict[str, Path]) -> str:
    """Freeze external runtime sources at import, independently of package version."""
    digest = hashlib.sha256()
    for name, path in sorted(files.items()):
        digest.update(name.encode() + b"\0")
        digest.update(Path(path).read_bytes())
    return digest.hexdigest()


def mimo_v26_cache_identity(bundle: str | Path, runtime_identity: str,
                           settings: dict) -> str:
    """Bind lazy media dependencies omitted by the main weight index.

    Weight stat identity follows the main-shard policy; it is not a substitute
    for bundle integrity verification. Bundles must remain immutable while loaded.
    Missing optional media files get explicit markers; unreadable files fail closed.
    """
    bundle = Path(bundle)
    digest = hashlib.sha256(b"mimo-v26-media-runtime-v1\0")
    digest.update(runtime_identity.encode() + b"\0")
    digest.update(json.dumps(settings, sort_keys=True).encode())
    for name in ("config.json", "audio_tokenizer/config.json"):
        path = bundle / name
        digest.update(b"\0" + name.encode() + b"\0")
        try:
            digest.update(path.read_bytes())
        except FileNotFoundError:
            digest.update(b"missing")
    path = bundle / "audio_tokenizer/model.safetensors"
    try:
        stat = path.stat()
        digest.update(f"audio_weights:{stat.st_size}:{stat.st_mtime_ns}".encode())
    except FileNotFoundError:
        digest.update(b"audio_weights:missing")
    return digest.hexdigest()


def read_mimo_v26_contract(bundle_path: str | Path | None) -> dict | None:
    if not bundle_path:
        return None
    path = Path(bundle_path)
    try:
        cfg = json.loads((path / "config.json").read_text())
        if cfg.get("model_type") != "mimo_v2":
            return None
        meta = json.loads((path / "jang_config.json").read_text())
    except (OSError, ValueError, AttributeError):
        return None
    if not isinstance(meta, dict):
        return None
    if meta.get("weight_format") == "mixed_affine_mxfp4":           # the RL JANG_2L family
        return meta
    if meta.get("format") == "jangtq2" and _routed_jangt_contract(cfg):  # JANGHT / JANGH: trellis or codebook routed experts
        return meta
    return None


def _routed_jangt_contract(cfg: dict) -> bool:
    """config.json carries the JANGT runtime contract: a jangt block and routed projections quantized jangt / jangtq2."""
    q = cfg.get("quantization")
    if not isinstance(cfg.get("jangt"), dict) or not isinstance(q, dict):
        return False
    return any(".mlp.switch_mlp." in k and isinstance(v, dict) and v.get("mode") in ("jangt", "jangtq2") for k, v in q.items())


def mimo_v26_media_requested(contract: dict) -> bool:
    caps = contract.get("capabilities") or {}
    modalities = caps.get("modalities") or {}
    return any(
        contract.get(f"has_{name}") is True or modalities.get(name) is True
        for name in ("vision", "video", "audio")
    )


def mimo_v26_media_enabled(contract: dict) -> bool:
    # Qualification uses an explicit process-local opt-in; bundle capability
    # stamps remain false until real image/video/audio generation is verified.
    return mimo_v26_media_requested(contract) or os.environ.get("VMLX_MIMO26_MEDIA_TEST") == "1"


def mimo_v26_modalities(bundle_path, contract):
    if not mimo_v26_media_enabled(contract):
        return ["text"]
    path = Path(bundle_path)
    cfg = json.loads((path / "config.json").read_text())
    weights = json.loads((path / "model.safetensors.index.json").read_text())["weight_map"]
    result = ["text"]
    if cfg.get("vision_config") and any(k.startswith("visual.") for k in weights):
        if cfg.get("image_token_id") is not None:
            result.append("vision")
        if cfg.get("video_token_id") is not None:
            result.append("video")
    if (cfg.get("audio_config") and (path / "audio_tokenizer/model.safetensors").is_file()
            and any(k.startswith("audio_encoder.") for k in weights)
            and any(k.startswith("speech_embeddings.") for k in weights)):
        result.append("audio")
    return result
