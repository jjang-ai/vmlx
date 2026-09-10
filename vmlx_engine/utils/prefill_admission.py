# SPDX-License-Identifier: Apache-2.0
"""Family-agnostic prefill admission control and allocation-error triage.

DSV4 already had a prefill valve: it projects the next chunk's peak Metal
working set and aborts the request BEFORE submitting GPU work, because a Metal
allocation failure inside libc++ can take the process down. Every other family
had nothing, so a prompt the device cannot serve was discovered by failing
rather than by checking.

Two pure pieces live here so both are unit-testable without Metal:

* :func:`prefill_valve_check` — the projection, generalised from DSV4's.
* :func:`classify_allocation_error` — tells a PERMANENT allocation failure
  (one a retry can never satisfy) apart from transient exhaustion.

The distinction matters because the recovery for the two is opposite. Clearing
the cache and rescheduling fixes transient exhaustion; against a request for
more than the device's maximum buffer size it re-runs the identical doomed
allocation forever. The scheduler already learned this the hard way with
``DSV4PrefillMemoryError``, which it excludes from recovery BY NAME.
"""

from __future__ import annotations

import os

_GIB = 1024**3

# MLX reports an over-maximum buffer request like:
#   [metal::malloc] Attempting to allocate 4398046511104 bytes which is greater
#   than the maximum allowed buffer size of 86586540032 bytes.
# Measured on-device: this is a CATCHABLE Python RuntimeError, not a process
# abort — the failure mode is that nothing classifies it, not that it escapes.
_PERMANENT_SIGNATURES = (
    "greater than the maximum allowed buffer size",
    "maximum allowed buffer size",
)

# The command-buffer variant is a different animal and MUST NOT be treated as a
# recoverable Python error. MEASURED on Qwen3.6-27B at a 100,935-token fresh
# span, after the prefill had correctly chunked:
#
#   libc++abi: terminating due to uncaught exception of type std::runtime_error:
#   [METAL] Command buffer execution failed: Insufficient Memory
#   (00000008:kIOGPUCommandBufferCallbackErrorOutOfMemory)
#
# libc++ TERMINATES — the process dies, so unlike [metal::malloc] there is no
# exception to catch and nothing downstream ever runs. Listing it here is only
# for triage of logs after the fact; the sole defence is the admission check,
# which declines before the command buffer is submitted.
_PROCESS_FATAL_SIGNATURES = (
    "command buffer execution failed",
    "kiogpucommandbuffercallbackerroroutofmemory",
)


def is_process_fatal_allocation_signature(text: str) -> bool:
    """Did this log line come from a Metal failure that ABORTS the process?"""
    lowered = (text or "").lower()
    return any(sig in lowered for sig in _PROCESS_FATAL_SIGNATURES)

# Exhaustion that a cache clear can genuinely relieve.
_TRANSIENT_SIGNATURES = (
    "out of memory",
    "cannot allocate memory",
    "allocation failed",
    "insufficient memory",
)


def wired_limit_advisory(max_ws_bytes: int) -> str:
    """User-facing guidance appended to every admission rejection.

    The rejections fire at the macOS Metal wired-memory ceiling (default
    ~84% of RAM, ``iogpu.wired_limit_mb=0``), which an administrator can
    raise — that is often the difference between "cannot serve this
    context" and serving it comfortably on the same machine. The advisory
    must carry the caution too: setting it too high can stall or panic
    macOS, and the sysctl resets on reboot.
    """
    return (
        f" Advanced: this rejection is against the macOS GPU wired-memory "
        f"limit ({max_ws_bytes / _GIB:.2f}GB; macOS defaults to ~84% of "
        f"RAM). An administrator can raise it with 'sudo sysctl "
        f"iogpu.wired_limit_mb=<MB>' — leave at least 8GB of RAM for macOS "
        f"or the system may become unstable, and note the setting resets "
        f"on reboot."
    )


class PrefillAdmissionError(RuntimeError):
    """The request cannot be served on this device — rejected before GPU work.

    The message deliberately avoids every scheduler ``CACHE_CORRUPTION_PATTERNS``
    substring ("out of memory", "Allocation failed", ...). Matching one would
    route this into cache-clear + reschedule, which would re-run the doomed
    prefill in a loop. Same contract as ``DSV4PrefillMemoryError``.
    """


def prefill_valve_enabled() -> bool:
    """Admission control is ON by default; set VMLX_PREFILL_ADMISSION=0 to disable."""
    raw = os.environ.get("VMLX_PREFILL_ADMISSION", "1").strip().lower()
    return raw not in {"0", "false", "no", "off"}


def prefill_keep_alloc_enabled() -> bool:
    """Skip mx.clear_cache() between prefill chunks; OFF by default.

    ONE reader for every generator. Before this helper the two generators
    disagreed: single_batch_generator read VMLINUX_PREFILL_KEEP_ALLOC with a
    VMLX_ fallback while mllm_batch_generator read only VMLX_, so the
    VMLINUX_ spelling toggled the text path but silently not the MLLM path.
    (cli.py --prefill-keep-alloc sets the VMLX_ spelling, so only direct env
    users hit the divergence.) One MLLM site also tested raw string
    truthiness, which made VMLX_PREFILL_KEEP_ALLOC=0 KEEP allocations there;
    the parsed-boolean contract below is the one every other site already
    used. VMLINUX_ wins when both are set, matching the original text path.
    """
    raw = os.environ.get(
        "VMLINUX_PREFILL_KEEP_ALLOC",
        os.environ.get("VMLX_PREFILL_KEEP_ALLOC", ""),
    )
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def prefill_valve_min_margin_bytes(default_gb: float = 2.0) -> int:
    """Transient-headroom floor used before any per-chunk peak is observed.

    Mirrors the DSV4 valve's default: measured transient peak-over-active on
    DSV4-Flash reaches ~3.4GB at 430k context, and the valve adapts upward from
    observed peaks, so this only has to cover the first chunks.
    """
    try:
        gb = float(os.environ.get("VMLX_PREFILL_ADMISSION_MIN_MARGIN_GB", default_gb))
    except (TypeError, ValueError):
        gb = default_gb
    return max(0, int(gb * _GIB))


def project_chunk_peak_bytes(
    active_bytes: int,
    observed_transient_bytes: int,
    min_margin_bytes: int,
) -> int:
    """Projected peak for the next chunk: active + 1.25x the largest transient seen."""
    margin = max(int(observed_transient_bytes * 1.25), int(min_margin_bytes))
    return int(active_bytes) + margin


def prefill_valve_check(
    active_bytes: int,
    max_ws_bytes: int,
    observed_transient_bytes: int,
    min_margin_bytes: int,
    *,
    chunk_start: int,
    chunk_end: int,
    model_label: str = "model",
) -> None:
    """Raise :class:`PrefillAdmissionError` when the next chunk cannot fit.

    Pure: no MLX calls, so the threshold is unit-testable without a GPU. A
    non-positive limit or active reading means "unknown", and unknown must never
    reject a request that would have worked.
    """
    if max_ws_bytes <= 0 or active_bytes <= 0:
        return
    projected = project_chunk_peak_bytes(
        active_bytes, observed_transient_bytes, min_margin_bytes
    )
    if projected <= max_ws_bytes:
        return
    raise PrefillAdmissionError(
        f"{model_label}: prefill admission rejected chunk "
        f"[{chunk_start}:{chunk_end}) — active Metal working set "
        f"{active_bytes / _GIB:.2f}GB plus projected transient "
        f"{(projected - active_bytes) / _GIB:.2f}GB exceeds the device "
        f"working-set limit {max_ws_bytes / _GIB:.2f}GB. A context of this "
        f"length cannot be served on this hardware; reduce the prompt or "
        f"context size." + wired_limit_advisory(max_ws_bytes)
    )


def classify_allocation_error(error: BaseException | str) -> str | None:
    """Return ``"permanent"``, ``"transient"``, or ``None`` (not allocation-related).

    ``"permanent"`` means a retry cannot succeed — the request asked for more
    than the device's maximum buffer size. Recovering by clearing caches would
    re-run the identical allocation, so callers must fail the request instead.
    """
    text = str(error).lower()
    if not text:
        return None
    for signature in _PERMANENT_SIGNATURES:
        if signature in text:
            return "permanent"
    if isinstance(error, PrefillAdmissionError):
        return "permanent"
    for signature in _TRANSIENT_SIGNATURES:
        if signature in text:
            return "transient"
    if "[metal::malloc]" in text:
        # A metal::malloc failure with no clearer signal: treat as transient so
        # the existing cache-clear recovery gets one chance at it.
        return "transient"
    return None


def is_permanent_allocation_error(error: BaseException | str) -> bool:
    return classify_allocation_error(error) == "permanent"


# Per-chunk attention budget. NOT the same as the hybrid guard's 8 GiB one-shot
# threshold: that decides whether to chunk at all, this decides how big a chunk
# may be, and the two must not share a number.
#
# Empirical, from the failure itself: Qwen3.6-27B died with a command-buffer OOM
# at 30 heads x 2048 x 67292 x 2 = 8.27 GB in a single chunk, against a ~9.5 GB
# Metal cap. So the scores buffer alone reaching ~8.3 GB is already fatal —
# whatever else is in flight during the forward pass does not leave room. Half
# the one-shot guard keeps a wide margin under that measured failure point.
# VMLX_PREFILL_CHUNK_ATTN_BUDGET_GB tunes it.
def _chunk_attention_budget_bytes() -> int:
    try:
        gb = float(os.environ.get("VMLX_PREFILL_CHUNK_ATTN_BUDGET_GB", "4"))
    except (TypeError, ValueError):
        gb = 4.0
    return max(1, int(gb * _GIB))


METAL_SINGLE_BUFFER_BUDGET_BYTES = _chunk_attention_budget_bytes()


def max_prefill_chunk_tokens(
    num_heads: int,
    context_tokens: int,
    budget_bytes: int | None = None,
    bytes_per_score: int = 2,
) -> int:
    """Largest chunk whose attention scores still fit one Metal buffer.

    A chunked prefill computes ``chunk x context`` attention scores, NOT
    ``chunk^2`` — so the per-chunk buffer grows with the CONTEXT even though the
    chunk size is fixed. A step size that is safe at 10k is fatal at 100k.

    MEASURED: Qwen3.6-27B (~30 heads) at a 67,292-token context chunked at the
    configured step and still died with a command-buffer OOM, which aborts the
    process. 30 x 2048 x 67292 x 2 = 8.3 GB — over the cap, in ONE chunk.

    Returns a token count >= 1; callers clamp their step size to it.
    """
    if budget_bytes is None:
        budget_bytes = _chunk_attention_budget_bytes()
    heads = max(1, int(num_heads or 1))
    context = max(1, int(context_tokens or 1))
    per_token = heads * context * max(1, int(bytes_per_score))
    if per_token <= 0:
        return 1
    return max(1, int(budget_bytes) // per_token)


def project_span_peak_bytes(
    measured_transient_bytes: int,
    measured_at_context: int,
    final_context: int,
) -> int:
    """Extrapolate a measured per-chunk transient to the END of the span.

    The hybrid prefill's transient scales with the CONTEXT being attended over,
    not with how many query tokens are in flight — established by three chunk
    sizing experiments that all aborted at the same 73-75k context whether the
    chunk was 2048 or 556 tokens. So the honest projection is linear in context:
    a transient of T at context C becomes T * (final / C) at the end.

    Pure, so the threshold is testable without a GPU.
    """
    transient = max(0, int(measured_transient_bytes))
    at = max(1, int(measured_at_context))
    final = max(1, int(final_context))
    if final <= at:
        return transient
    return int(transient * (final / at))


def replace_chunk_transient_observation(
    model_type: str,
    measured_bytes: int,
    measured_chunk_tokens: int,
    retained_bytes: int,
    retained_chunk_tokens: int,
) -> bool:
    """Keep the maximum, except after a measured GLM chunk-width reduction.

    A completed smaller GLM chunk establishes a new allocation regime; do not
    extrapolate the previous wider chunk forever. This does not predict an
    unexecuted smaller chunk or change admission safety margins. Other hybrid
    families retain their existing maximum-observation policy.
    """
    if measured_bytes <= 0:
        return False
    return measured_bytes >= retained_bytes or (
        model_type in {"glm5_next", "glm5_next_text"}
        and 0 < measured_chunk_tokens < retained_chunk_tokens
    )


def hybrid_chunk_valve_check(
    active_bytes: int,
    max_ws_bytes: int,
    observed_transient_bytes: int,
    observed_at_context: int,
    next_context: int,
    min_margin_bytes: int,
    *,
    chunk_start: int,
    chunk_end: int,
    model_label: str = "model",
    safety: float = 1.10,
    observed_chunk_tokens: int = 0,
    next_chunk_tokens: int = 0,
    chunk_scaled: bool = False,
) -> None:
    """Per-chunk admission for the hybrid prefill, projected along the context.

    Why this and not :func:`prefill_valve_check`: that one pads the largest
    observed transient by a flat 1.25x, which suits DSV4 where the transient is
    volatile. The hybrid transient is not volatile — MEASURED on Qwen3.6-27B it
    is a straight line in context (2.82GB + 0.00015*ctx, residuals <=0.02GB)
    that grows only ~0.31GB per 2048-token chunk. Against a real 107.52GB
    device limit the flat pad refuses chunks that demonstrably run: chunk 46
    peaked at 104.41GB and would have been declined at 108.65GB projected.

    Scaling the last observed transient by the context ratio instead predicts
    the measured peaks to within 0.06GB:

        ctx 92,160 -> projected 98.21GB, actual 98.21GB
        ctx 94,208 -> projected 104.47GB, actual 104.41GB   (ran, admitted)
        ctx 96,256 -> projected 110.80GB                     (aborted, declined)

    ``safety`` sits inside the window those two chunks define (0.81 < s < 1.18);
    1.10 keeps ~10% headroom while still separating them.

    ``chunk_scaled`` is for attention paths whose transient is the scores
    buffer itself and therefore proportional to ``chunk x context`` (dots3's
    absorbed-latent path), unlike the GDN hybrid measured above where halving
    the chunk did not move the abort point. Without it, a transient observed
    at a 1024-token chunk is projected unchanged onto a 64-token retry, so
    the halving ladder can never rescue the span: MEASURED live, dots3 at a
    17k context was refused at every chunk size down to the floor with the
    same ~8.5GB projection, capping the model at ~17k on hardware that
    serves it. When enabled and both chunk sizes are known, the projection
    scales by their ratio.

    Unknown readings never reject.
    """
    if max_ws_bytes <= 0 or active_bytes <= 0 or observed_transient_bytes <= 0:
        return
    projected = project_span_peak_bytes(
        observed_transient_bytes, observed_at_context, next_context
    )
    if chunk_scaled and observed_chunk_tokens > 0 and next_chunk_tokens > 0:
        projected = int(projected * (next_chunk_tokens / observed_chunk_tokens))
    projected = max(int(projected * safety), int(min_margin_bytes))
    if active_bytes + projected <= max_ws_bytes:
        return
    raise PrefillAdmissionError(
        f"{model_label}: prefill admission rejected chunk "
        f"[{chunk_start}:{chunk_end}) at a context of {next_context} tokens — "
        f"active Metal working set {active_bytes / _GIB:.2f}GB plus a projected "
        f"{projected / _GIB:.2f}GB for this chunk exceeds the device "
        f"working-set limit {max_ws_bytes / _GIB:.2f}GB. A context of this "
        f"length cannot be served on this hardware; shorten the conversation "
        f"or reduce the prompt." + wired_limit_advisory(max_ws_bytes)
    )


def fit_peak_model(
    samples: "list[tuple[int, int]]",
) -> "tuple[float, float] | None":
    """Least-squares fit of ``absolute peak bytes = intercept + slope * context``.

    THE QUANTITY MATTERS. An earlier version of the whole-span check projected
    only the per-chunk TRANSIENT and added the active reading taken at span
    START. That is what made it unwireable, and ``test_prefill_admission.py``
    pinned the failure: fed the real numbers from the span that actually died
    (18.32GB transient at ~65k context, final context 100,935, 21GB active at
    start, ~107GB limit) it projects ~56GB and ADMITS the fatal span. Active is
    not a constant across a span — it climbs as KV accumulates, and the observed
    active at the death point was ~95GB, not 21GB.

    Fitting the measured ``mx.get_peak_memory()`` per chunk sidesteps the whole
    problem: peak is already absolute (weights + KV + transient), so there is
    nothing to add and nothing to forget. Against the same measurements the
    affine peak model projects ~123GB at the final context and declines.

    Affine, not proportional: the per-chunk valve
    (:func:`hybrid_chunk_valve_check`) projects one chunk ahead where scaling by
    the context ratio is accurate to 0.06GB, but a whole-span decision
    extrapolates much further, and proportional scaling drops the constant term.
    The generator already has scar tissue from exactly that error — feeding
    chunk 0's one-time full-width allocation into a proportional projection
    "extrapolated to ~425GB at 100k and declined every request".

    Two parameters need two distinct contexts. Returns ``None`` when the
    samples cannot determine them, and None must never reject.
    """
    points = [
        (float(ctx), float(transient))
        for ctx, transient in samples
        if ctx > 0 and transient > 0
    ]
    if len(points) < 2:
        return None
    xs = [x for x, _ in points]
    ys = [y for _, y in points]
    n = float(len(points))
    mean_x = sum(xs) / n
    mean_y = sum(ys) / n
    sxx = sum((x - mean_x) ** 2 for x in xs)
    if sxx <= 0.0:  # every sample at the same context — slope undetermined
        return None
    sxy = sum((x - mean_x) * (y - mean_y) for x, y in points)
    slope = sxy / sxx
    intercept = mean_y - slope * mean_x
    return (intercept, slope)


def project_peak_affine(
    intercept: float, slope: float, context: int, floor_bytes: int = 0
) -> int:
    """Projected absolute peak at ``context`` from a fitted affine model.

    ``floor_bytes`` keeps the projection from falling below a peak that was
    already OBSERVED — a fit is an estimate, a measurement is not, and the
    larger of the two is the honest input to an admission decision.
    """
    projected = intercept + slope * max(0, int(context))
    return max(int(floor_bytes), int(projected) if projected > 0 else 0)


def turn_peak_admission_check(
    max_ws_bytes: int,
    walk_model: "tuple[float, float] | None",
    final_context: int,
    *,
    last_observed_peak_bytes: int = 0,
    allowance_bytes: int = 0,
    fitted_max_context: int = 0,
    max_extrapolation: float = 2.0,
    model_label: str = "model",
) -> None:
    """Decline a hybrid-hit delta forward whose CROSS-TURN peak walk says it dies.

    Why a third valve. On an incrementally growing conversation the absolute
    Metal peak walks UP between turns — measured on Qwen3.8 at ~5.0-5.6GB per
    ~5.6k-token turn (allocator/fragmentation growth that a fresh process does
    not carry: the identical request replayed cold survives). Neither existing
    valve can see that walk:

    * :func:`span_admission_check` fits peaks WITHIN one span; a ~5.6k delta
      yields 2-3 chunk peaks at near-identical contexts, so the fitted slope is
      ~0 and the check passes silently — confirmed over seven crash runs that
      aborted with zero admission refusals.
    * :func:`hybrid_chunk_valve_check` projects from an observed transient, and
      the first chunk of each span has no observation — precisely the chunk
      whose command buffer aborted (one ~32s silent buffer, then SIGABRT).

    ``walk_model`` is an affine fit of (final context, absolute turn peak)
    across PREVIOUS turns of this process, from :func:`fit_peak_model`. Because
    it is fitted within the current process and config, it self-calibrates to
    whatever is resident — the cross-config intercept shift measured between the
    stock pool (wall at ~96k) and a 4GB-bounded pool (wall at ~101.6k) is
    captured automatically, where any constant threshold misclassifies one of
    the two.

    ``allowance_bytes`` defaults to ZERO — refuse at the reported limit — and
    that is a measured decision, not caution by reflex. The limit is advisory
    (a turn peaking at 109.2GB against the 107.5GB limit demonstrably
    survived), which tempts a positive allowance; but across both measured
    configs the last-surviving turn and the first-aborting turn PROJECT to the
    same ~109.2GB from their own walks — the fatal turn's actual peak
    overshoots its linear projection by ~3-5GB (the walk is slightly
    super-linear at the wall), so no threshold separates them. When the
    boundary cannot be split, the house rule inverts: the alternative to a
    refusal here is not a served request, it is a process abort the per-chunk
    valve cannot catch (blind first chunk). Refusing at the limit costs at
    most one turn that survives only by EXCEEDING the device's stated budget;
    the conversation continues past the refusal via engine restart +
    L2 restore, which a fresh process serves with ~90GB of headroom
    (measured fresh-replay immunity at the same span).

    Unknown readings never reject: no fit, non-positive limit, non-positive
    slope, or a projection beyond ``max_extrapolation`` x the fitted context
    range all return without raising.
    """
    if max_ws_bytes <= 0 or walk_model is None:
        return
    intercept, slope = walk_model
    if slope <= 0:
        return
    if fitted_max_context > 0 and final_context > fitted_max_context * max_extrapolation:
        return
    projected = project_peak_affine(
        intercept, slope, final_context, floor_bytes=max(0, int(last_observed_peak_bytes))
    )
    if projected <= 0:
        return
    threshold = int(max_ws_bytes) + max(0, int(allowance_bytes))
    if projected <= threshold:
        return
    raise PrefillAdmissionError(
        f"{model_label}: turn-peak admission declined a delta forward at a "
        f"context of {final_context} tokens — the cross-turn Metal peak walk "
        f"projects {projected / _GIB:.1f}GB against a refusal threshold of "
        f"{threshold / _GIB:.1f}GB (device limit {max_ws_bytes / _GIB:.1f}GB "
        f"+ {max(0, int(allowance_bytes)) / _GIB:.1f}GB measured overshoot "
        f"allowance; walk {intercept / _GIB:.2f}GB + "
        f"{slope * 1000 / _GIB:.4f}GB per 1k tokens). Growing this "
        f"conversation further in-process would abort the engine; start a new "
        f"conversation, restart the engine to continue this one, or reduce "
        f"the prefix-cache memory settings."
    )


def span_admission_check(
    max_ws_bytes: int,
    peak_model: "tuple[float, float] | None",
    final_context: int,
    *,
    fresh_tokens: int,
    model_label: str = "model",
    safety: float = 1.10,
    fitted_max_context: int = 0,
    max_extrapolation: float = 4.0,
    degradable_chunks: bool = False,
) -> None:
    """Decline a whole prefill span that cannot finish, BEFORE burning the work.

    Per-chunk admission declines correctly but late: only once memory has
    climbed near the ceiling, which on a ~100k span is ~46 chunks of GPU work
    already spent. Deciding once, up front, converts that into an immediate
    clean per-request error.

    ``peak_model`` is a fit from :func:`fit_peak_model`, learned from a PREVIOUS
    span — a span cannot measure itself before it starts, which is why this
    check needs cross-span state and the per-chunk valve does not.

    Note there is no ``active_bytes`` parameter, deliberately. The fitted peak
    is already absolute, and the previous signature's ``active at span start``
    term is precisely what made this check admit the span that died.

    ``safety`` is 1.10 rather than the per-chunk valve's 1.25 because an affine
    fit over many measured chunks is a far tighter estimator than one scaled
    observation; the same 10% margin the per-chunk valve uses internally.

    Unknown readings never reject: a missing model, a non-positive limit, or a
    non-increasing fit (slope <= 0 — peak not growing with context, so there is
    nothing to extrapolate) all return without raising.
    """
    if max_ws_bytes <= 0 or peak_model is None:
        return
    intercept, slope = peak_model
    if slope <= 0:
        return
    # Do not extrapolate a fit far past the contexts it was measured over.
    # REJECTING A SPAN THAT WOULD HAVE RUN IS THE WORSE FAILURE: the per-chunk
    # valve is a real backstop that still declines correctly (just later), so
    # missing a decline costs wasted work, while a false decline breaks a
    # request the device serves. Beyond `max_extrapolation` x the measured
    # range, defer to the per-chunk valve rather than guess.
    if fitted_max_context > 0 and final_context > fitted_max_context * max_extrapolation:
        return
    # NO historical floor. Flooring this span's projection at a PREVIOUS span's
    # observed peak imports that span's baseline, and mx.get_peak_memory() is
    # global absolute active memory — it includes whatever else was resident at
    # the time. Independent review caught the false rejection this produces:
    # against the committed measured row the per-chunk valve projects 106.17GiB
    # and RUNS the chunk (limit 107.52GiB), while a 104.41GiB historical floor
    # carried into this check projects 114.85GiB and rejects the same work.
    # Rejecting a span the device would have served is the failure this valve
    # must never cause, so the fit alone decides.
    projected = project_peak_affine(intercept, slope, final_context)
    if projected <= 0:
        return
    padded = int(projected * safety)
    if padded <= max_ws_bytes:
        return
    # The fit was learned at the chunk size the previous span used. When the
    # caller's per-chunk path can DEGRADE — halving a declined chunk and
    # retrying rather than failing the request — the projection is an upper
    # bound on work that will not actually be attempted at that width, and
    # this check's own rule applies with full force: rejecting a span the
    # device would have served is the failure it must never cause. So defer
    # to the adaptive per-chunk path unless the span is over the limit by a
    # wide margin. Measured: an 8,033-token prefill that this check would
    # have refused completes at 357 pp/s once the chunk path can adapt.
    if degradable_chunks and padded < max_ws_bytes * 2:
        return
    raise PrefillAdmissionError(
        f"{model_label}: prefill admission declined a {fresh_tokens}-token span "
        f"before starting it — at the final context of {final_context} tokens the "
        f"projected Metal working set is {padded / _GIB:.1f}GB against a device "
        f"limit of {max_ws_bytes / _GIB:.1f}GB (fitted peak model "
        f"{intercept / _GIB:.2f}GB + {slope * 1000 / _GIB:.4f}GB per 1k tokens). "
        f"A context of this length cannot be served on this hardware; "
        f"shorten the conversation or reduce the prompt."
        + wired_limit_advisory(max_ws_bytes)
    )
