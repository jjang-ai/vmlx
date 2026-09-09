"""Optional Qwen4 short prefill with inline recurrent/PLE checkpoints.

Each layer processes the original prompt segments in one lazy model graph.
This preserves stock matrix shapes and cache boundaries while avoiding
separate graph evaluations and unnecessary final-layer pointwise outputs.
"""
from __future__ import annotations
from copy import copy
import logging
import os

logger = logging.getLogger(__name__)
_MARKS = ('prefill_checkpoint_states', 'prefill_checkpoint_aux_states')


def coalesce_qwen4_prefill_checkpoints(
    generator, request, lm, input_ids, cache, all_tokens, boundaries, kwargs_for
):
    """Return last-token model output, or None before any mutation."""
    if os.environ.get('VMLX_QWEN4_COALESCE_PREFILL_CHECKPOINTS', '0') != '1':
        return None
    if str(getattr(generator, '_model_type', '')) not in {'qwen4_exp', 'qwen4_exp_text'}:
        return None
    if int(getattr(request, '_cached_tokens', 0) or 0) != 0:
        return None
    if input_ids.ndim != 2 or input_ids.shape[0] != 1 or len(boundaries) != 2:
        return None
    first, final = boundaries
    if not (0 < first < final <= 4096 and final < input_ids.shape[1] <= 4112):
        return None
    kv_positions = set(getattr(generator, '_hybrid_kv_positions', ()) or ())
    recurrent = [(i, c) for i, c in enumerate(cache) if i not in kv_positions]
    if not recurrent or any(
        type(c).__name__ != 'ArraysCache'
        or not isinstance(getattr(c, 'cache', None), list)
        or len(c.cache) not in (2, 4)
        or getattr(c, 'lengths', None) is not None
        for _, c in recurrent
    ):
        return None
    kwargs = kwargs_for(0, input_ids.shape[1])
    kwargs.update(prefill_checkpoint_steps=(first, final), prefill_last_logits_only=True)
    try:
        output = lm(input_ids, **kwargs)
        for boundary in (first, final):
            checkpoint_cache = list(cache)
            for i, current in recurrent:
                state = getattr(current, 'prefill_checkpoint_states', {}).get(boundary)
                aux = getattr(current, 'prefill_checkpoint_aux_states', {}).get(boundary)
                if not isinstance(state, tuple) or len(state) != 2:
                    raise RuntimeError('Qwen4 prefill did not provide recurrent checkpoint')
                if len(current.cache) == 4 and (not isinstance(aux, tuple) or len(aux) != 2):
                    raise RuntimeError('Qwen4 prefill did not provide PLE checkpoint')
                checkpoint = copy(current)
                checkpoint.cache = list(state) + (list(aux) if len(current.cache) == 4 else [])
                for name in _MARKS:
                    checkpoint.__dict__.pop(name, None)
                checkpoint_cache[i] = checkpoint
            if not generator._maybe_capture_clean_ssm_boundary(
                request, checkpoint_cache, all_tokens, boundary
            ):
                raise RuntimeError('Qwen4 prefill checkpoint was not captured')
    finally:
        for _, current in recurrent:
            for name in _MARKS:
                current.__dict__.pop(name, None)
    logger.info(
        'Qwen4 coalesced full prefill: request=%s first=%d final=%d total=%d recurrent_layers=%d',
        request.request_id, first, final, input_ids.shape[1], len(recurrent),
    )
    return output
