"""Capture an exact singleton native boundary without retaining a KV mirror."""

import logging

logger = logging.getLogger(__name__)


def prompt_with_hybrid_capture(batch, tokens, prompt):
    """Run the owning prefill, detaching companion state before its suffix.

    The scheduler supplies the target in this batch's token coordinates.
    Unsupported/padded multi-row batches retain the original safe path.
    Never retry a failed model forward: native state may already have advanced.
    """
    resolve = getattr(batch, "_vmlx_hybrid_boundary_target", None)
    capture = getattr(batch, "_vmlx_hybrid_boundary_store", None)
    if not callable(resolve) or not callable(capture) or len(tokens) != 1 or len(batch.uids) != 1:
        return prompt(batch, tokens)
    uid = batch.uids[0]
    prior = list(batch.tokens[0])
    try:
        target = resolve(uid, prior)
    except Exception:
        logger.warning("Hybrid prefill boundary resolution failed for uid=%s", uid, exc_info=True)
        return prompt(batch, tokens)
    if target is None or not len(prior) <= target <= len(prior) + len(tokens[0]):
        return prompt(batch, tokens)
    cut = target - len(prior)
    if cut:
        prompt(batch, [tokens[0][:cut]])
    try:
        capture(uid, list(batch.tokens[0]), batch.extract_cache(0))
    except Exception:
        # Terminal persistence still requires a complete matching companion;
        # failed capture therefore falls back to the existing clean rederive.
        logger.warning("Hybrid prefill boundary capture failed for uid=%s", uid, exc_info=True)
    if cut < len(tokens[0]):
        return prompt(batch, [tokens[0][cut:]])
