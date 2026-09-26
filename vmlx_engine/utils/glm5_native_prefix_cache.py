"""Exact GLM native checkpoints over the shared, SSD-only cache pool.

This does not split recurrent state into generic KV blocks or retain an L1
payload. The serving caller owns the causal checkpoint and media admission.
"""

from __future__ import annotations

import hashlib
import logging
import time
from pathlib import Path

from ..global_disk_cache_budget import (
    ensure_managed_block_cache_namespace,
    get_global_disk_cache_budget,
)
from ..models.glm5_next.glm5_next import Glm5KDACache, Glm5MLACache
from .ssm_companion_cache import SSMCompanionCache
from .ssm_companion_disk_store import SSMCompanionDiskStore

logger = logging.getLogger(__name__)
NATIVE_GLM_SSD_SCHEMA = "glm5_native_ssd_v1"


def glm5_native_layout(layers):
    """Observe an entire typed layout; a family name alone is not admission."""
    types = tuple(type(layer) for layer in layers)
    if not types or not {Glm5KDACache, Glm5MLACache}.issubset(types):
        return None
    if any(kind not in (Glm5KDACache, Glm5MLACache) for kind in types):
        return None
    return tuple((type(layer), tuple(layer.meta_state)[:3]) for layer in layers)


class Glm5NativePrefixCache:
    def __init__(self, *, root, max_size_bytes, model_key, layout,
                 allow_legacy_hashed_namespaces=False,
                 allow_legacy_direct_namespace=False, activity_probe=None):
        if not model_key or not layout or any(
            kind not in (Glm5KDACache, Glm5MLACache) for kind, _ in layout
        ):
            raise ValueError("native GLM SSD requires model identity and layout")
        self.layout = tuple(layout)
        self._activity_probe = activity_probe
        self._last_activity = time.monotonic()
        self.model_key = f"{model_key}:{NATIVE_GLM_SSD_SCHEMA}"
        cache_root = Path(root).expanduser().resolve()
        namespace = hashlib.sha256(self.model_key.encode()).hexdigest()[:16]
        directory = ensure_managed_block_cache_namespace(cache_root / namespace)
        if not directory.is_relative_to(cache_root):
            raise ValueError("native GLM cache escaped configured root")
        self.budget = get_global_disk_cache_budget(
            cache_root, int(max_size_bytes),
            allow_legacy_hashed_namespaces=allow_legacy_hashed_namespaces,
            allow_legacy_direct_namespace=allow_legacy_direct_namespace,
        )
        try:
            self.disk = SSMCompanionDiskStore(
                directory=directory / "ssm_companion",
                budget_bytes=int(max_size_bytes), global_budget=self.budget,
                idle_maintenance=self._idle_maintenance,
            )
        except Exception:
            self.budget.close()
            raise
        self.lookup = SSMCompanionCache(
            max_entries=0, max_bytes=0, model_key=self.model_key, disk_store=self.disk,
        )
        self.last_store = None
        self.last_fetch = None

    def _valid_boundary(self, layers, boundary):
        if type(boundary) is not int or boundary <= 0:
            return False
        try:
            if glm5_native_layout(layers) != self.layout:
                return False
            for layer in layers:
                if isinstance(layer, Glm5MLACache) and layer.offset != boundary:
                    return False
                if isinstance(layer, Glm5KDACache) and any(
                    value is None for value in layer.cache
                ):
                    return False
                type(layer).from_state(layer.state, layer.meta_state)
        except (ValueError, TypeError, AttributeError):
            return False
        return True

    def _idle_maintenance(self):
        # No block writer exists in this native-only backend. Own its owed
        # aggregate janitor work without scanning during inference or stores.
        if time.monotonic() - self._last_activity < 1.0:
            return
        if self._activity_probe is not None:
            try:
                if self._activity_probe():
                    return
            except Exception:
                return  # unknown engine activity is not an idle signal
        if self.budget.deferred_reconcile_due:
            self.budget.run_deferred_reconcile()
        elif self.budget.idle_reconcile_due():
            self.budget.run_idle_reconcile()

    def store(self, tokens, boundary, layers, *, extra_keys=None, request_id=None):
        """Freeze then wait for the exact fsynced pair, not queue admission."""
        started = time.perf_counter()
        self._last_activity = time.monotonic()
        key = None
        outcome, detail, durable = "refused", "invalid native boundary", False
        admissible = self._valid_boundary(layers, boundary) and boundary <= len(tokens)
        if admissible:
            payload_bytes = sum(
                int(value.nbytes) for layer in layers for value in layer.state
                if value is not None
            )
            # A record larger than this writer's configured pool cannot fit,
            # even before metadata. Refuse without freezing or evicting useful
            # entries. A smaller concurrent lease is still enforced at publish.
            if self.disk.budget_bytes > 0 and payload_bytes > self.disk.budget_bytes:
                admissible = False
                detail = (
                    f"native checkpoint bytes={payload_bytes} exceed configured "
                    f"SSD cap={self.disk.budget_bytes}"
                )
        if admissible:
            key = self.lookup._key(tokens, boundary, cache_extra_keys=extra_keys)
            try:
                if self.disk.has_complete(key):
                    # Validate the typed record, not merely file presence,
                    # before claiming that a duplicate is already durable.
                    found = self.disk.fetch(key)
                    if found and found[1] and self._valid_boundary(found[0], boundary):
                        outcome, detail, durable = "already_durable", "existing typed checkpoint", True
                if not durable:
                    admitted = self.disk.store(key, layers, True, tokens, boundary)
                    if admitted:
                        durable = self.disk.wait_for_write(key, timeout=30.0)
                        outcome = "stored" if durable else "failed"
                        detail = "typed checkpoint fsynced" if durable else "native SSD publication incomplete"
                    else:
                        detail = "native SSD write refused by queue or pool admission"
            except Exception as exc:
                outcome, detail = "failed", f"native SSD write: {type(exc).__name__}"
                logger.warning("GLM native SSD store failed for %s: %s", request_id, exc)
        receipt = {
            "request_id": request_id, "outcome": outcome, "detail": detail,
            "durable": durable, "retained_tokens": boundary if durable else 0,
            "key": key, "seconds": time.perf_counter() - started,
        }
        self.last_store = receipt
        self._last_activity = time.monotonic()
        logger.info(
            "GLM native SSD publication for %s: outcome=%s N=%d key=%s durable=%s seconds=%.4f",
            request_id, outcome, receipt["retained_tokens"], key, durable, receipt["seconds"],
        )
        return receipt

    def fetch(self, tokens, *, extra_keys=None, request_id=None):
        """Longest exact stored boundary; never rewind a longer native state."""
        started = time.perf_counter()
        self._last_activity = time.monotonic()
        token_ids = list(tokens)
        ceiling = len(token_ids) - 1
        found = None
        while ceiling > 0:
            found = self.lookup.fetch_longest_prefix(
                token_ids, max_len=ceiling, cache_extra_keys=extra_keys,
            )
            if found is None:
                break
            boundary, layers, complete = found
            # The transport validates its record, while this facade owns the
            # architecture's exact state boundary. A rejected longest record
            # must not hide a shorter valid checkpoint with the same identity.
            if type(boundary) is not int or not 0 < boundary <= ceiling:
                found = None
                break
            if complete and self._valid_boundary(layers, boundary):
                break
            ceiling = boundary - 1
            found = None
            del layers  # Release a large rejected payload before the next read.
        boundary = found[0] if found is not None else 0
        key = self.lookup._key(tokens, boundary, cache_extra_keys=extra_keys) if boundary else None
        self.last_fetch = {
            "request_id": request_id, "cached_tokens": boundary, "key": key,
            "seconds": time.perf_counter() - started,
        }
        logger.info("GLM native SSD %s for %s: N=%d key=%s seconds=%.4f",
                    "HIT" if found else "MISS", request_id, boundary, key,
                    self.last_fetch["seconds"])
        return (boundary, found[1]) if found is not None else None

    def close(self):
        # Do not remove the writer's lease until its last publication ends.
        if self.disk.shutdown(timeout=None):
            self.budget.close()
