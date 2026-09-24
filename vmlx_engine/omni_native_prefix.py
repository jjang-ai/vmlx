"""SSD checkpoints before native Omni's assistant generation suffix.

Recurrent state cannot be cropped or stitched. Restore only a complete native
state whose token IDs and source media are an exact prefix of this request.
"""
from __future__ import annotations

from copy import deepcopy
import hashlib
import json
import logging
import time

logger = logging.getLogger(__name__)
_KIND = "prompt_prefill_v1"


def source_prefix_keys(messages, video_policy, tools=None):
    """Hash each source message once, including decoded media identity."""
    from .omni_multimodal import _media_part_identity

    policy = {"contract": _KIND, "video_policy": video_policy}
    if tools:
        policy["tools"] = tools
    digest = hashlib.sha256(json.dumps(
        policy,
        sort_keys=True, separators=(",", ":"),
    ).encode())
    result = []
    for count, message in enumerate(messages, 1):
        item = {key: value for key, value in message.items() if value is not None}
        if isinstance(item.get("content"), list):
            parts = []
            for part in item["content"]:
                identity = _media_part_identity(part)
                parts.append({"media_identity": identity} if identity is not None else part)
            item["content"] = parts
        encoded = json.dumps(item, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()
        digest.update(str(len(encoded)).encode() + b":" + encoded)
        # Admission has checked complete call/result batches. The native
        # template groups contiguous tool results in one user-role segment;
        # only its closing boundary is a reusable prompt checkpoint.
        tool_boundary = message.get("role") == "tool" and (
            count == len(messages) or messages[count].get("role") != "tool"
        )
        if message.get("role") == "user" or tool_boundary:
            result.append((count, "prefill:" + digest.hexdigest()))
    return result


def prefill_state(session, embeds):
    """Advance the native hybrid cache without sampling or changing its dtype."""
    if embeds.shape[1] == 0:
        return
    from mlx_lm.models.base import create_attention_mask, create_ssm_mask

    backbone = session.mlx_model.backbone
    cache = session._cache
    attention_mask = create_attention_mask(embeds, cache[backbone.fa_idx])
    ssm_mask = create_ssm_mask(embeds, cache[backbone.ssm_idx])
    hidden = embeds
    cache_index = 0
    for layer in backbone.layers:
        if layer.block_type in ("M", "*"):
            layer_cache = cache[cache_index]
            cache_index += 1
            mask = attention_mask if layer.block_type == "*" else ssm_mask
            hidden = layer(hidden, mask=mask, cache=layer_cache)
        else:
            hidden = layer(hidden)
    # The serialization boundary materializes cache state before publishing.
    # Norm/head are unnecessary: the existing decoder consumes the remaining
    # assistant suffix and computes its normal next-token logits.


class NativePrefillCheckpoints:
    def __init__(self, dispatcher, messages, video_policy, *, publish, tools=None):
        self.dispatcher = dispatcher
        self.keys = source_prefix_keys(messages, video_policy, tools)
        self.message_count = len(messages)
        self.publish_enabled = publish
        self.candidate = None
        self._searched = False
        self._remaining = list(reversed(self.keys))

    def find(self):
        if self._searched:
            return self.candidate
        self._searched = True
        owner = self.dispatcher
        started = time.monotonic()
        store = owner._native_disk_store()
        from .omni_multimodal import _OMNI_SESSION_L2_SCHEMA
        from mlx_lm.models.cache import load_prompt_cache

        while self._remaining:
            count, signature = self._remaining.pop(0)
            def read(path):
                import mlx.core as mx
                cache, metadata = load_prompt_cache(str(path), return_metadata=True)
                if (metadata.get("schema") != _OMNI_SESSION_L2_SCHEMA
                        or metadata.get("kind") != _KIND
                        or metadata.get("bundle_fingerprint") != owner._session_l2_fingerprint
                        or metadata.get("signature") != signature
                        or metadata.get("source_message_count") != str(count)):
                    return None
                tokens = json.loads(metadata.get("token_ids") or "null")
                rendered = json.loads(metadata.get("history_json") or "null")
                if (not isinstance(tokens, list) or not tokens
                        or any(type(token) is not int or token < 0 for token in tokens)
                        or not isinstance(rendered, list) or len(rendered) != count
                        or any(not isinstance(message, dict) for message in rendered)):
                    return None
                expected = owner._session.mlx_model.make_cache()
                if len(cache) != len(expected):
                    return None
                for loaded, native in zip(cache, expected):
                    if type(loaded) is not type(native):
                        return None
                    if hasattr(native, "offset") and int(loaded.offset) != len(tokens):
                        return None
                mx.eval([entry.state for entry in cache])
                return {"cache": cache, "tokens": tokens, "rendered": rendered,
                        "source_count": count, "signature": signature, "path": path}
            try:
                candidate = store.load(signature, read)
            except Exception as error:
                logger.warning("Omni prompt checkpoint rejected: %s", error)
                continue
            if candidate is not None:
                self.candidate = candidate
                candidate["restore_seconds"] = round(time.monotonic() - started, 6)
                return candidate
        return None

    def accept(self, input_ids):
        """Validate tokens before assigning any restored recurrent state."""
        candidate = self.candidate
        if candidate is None:
            return 0
        tokens = candidate["tokens"]
        if len(tokens) >= input_ids.shape[-1] or input_ids[0, :len(tokens)].tolist() != tokens:
            logger.info("Omni prompt checkpoint token prefix changed; rebuilding supplied history")
            candidate["cache"] = None
            self.candidate = None
            self._searched = False  # Try an earlier valid user boundary first.
            return None
        owner = self.dispatcher
        owner._session._cache = candidate["cache"]
        owner._session._history_text = deepcopy(candidate["rendered"])
        owner._session_l2_path = candidate["path"]
        owner._session_l2_stats["hits"] += 1
        owner._session_l2_stats["last_error"] = None
        owner._session_l2_stats["last_restore_seconds"] = candidate["restore_seconds"]
        owner._session_l2_stats["last_snapshot_kind"] = _KIND
        logger.info("Omni restored native prompt checkpoint: %d tokens, %d source messages",
                    len(tokens), candidate["source_count"])
        return len(tokens)

    def publish(self, input_ids, rendered):
        if not self.publish_enabled or not self.keys or self.keys[-1][0] != self.message_count:
            return
        count, signature = self.keys[-1]
        if (self.candidate is not None and self.candidate["signature"] == signature
                and self.candidate["tokens"] == input_ids[0].tolist()):
            return  # An identical durable checkpoint already exists.
        from .omni_multimodal import _OMNI_SESSION_L2_SCHEMA
        from mlx_lm.models.cache import save_prompt_cache

        owner = self.dispatcher
        started = time.monotonic()
        metadata = {
            "schema": _OMNI_SESSION_L2_SCHEMA, "kind": _KIND,
            "bundle_fingerprint": owner._session_l2_fingerprint,
            "signature": signature, "source_message_count": str(count),
            "token_ids": json.dumps(input_ids[0].tolist(), separators=(",", ":")),
            "history_json": json.dumps(rendered, ensure_ascii=False, separators=(",", ":")),
        }
        try:
            path = owner._native_disk_store().save(signature, lambda destination:
                save_prompt_cache(str(destination), owner._cache_for_persistence(), metadata))
        except Exception as error:
            owner._session_l2_stats["last_error"] = str(error)
            raise
        owner._session_l2_path = path
        owner._session_l2_stats["stores"] += 1
        owner._session_l2_stats["last_store_seconds"] = round(time.monotonic() - started, 6)
        owner._session_l2_stats["last_error"] = None
        owner._session_l2_stats["last_snapshot_kind"] = _KIND
        logger.info("Omni persisted native prompt checkpoint: %d tokens, %d bytes",
                    input_ids.shape[-1], path.stat().st_size)
