# SPDX-License-Identifier: Apache-2.0
"""DeepSeek-V4.1-Flash DSpark speculative-decode generator for the vMLX engine.

Selected (scheduler) only when a bundle's dSpark draft is attached as ``model._dspark`` (default-on
for bundles that ship the draft; ``VMLX_DSV41_DSPARK=0`` disables it) and max_num_seqs == 1. It subclasses
``SingleBatchGenerator`` to reuse its request lifecycle (insert/remove/Response/stop machinery) and
overrides only the decode path: it prefills the request's ENGINE cache with
``jang_tools.dsv41.dspark.DSparkRunner.prefill(session=<engine DeepseekV41Cache list>)`` (so the prompt
boundary snapshot and the prefix cache see the same state as the AR path) and then drives
``DSparkRunner.decode_stream`` on that session (no-refeed fast path, DECODE_DSPARK_METHODS M1).

Greedy requests (temperature == 0) produce the backbone's greedy tokens; speculation only changes
throughput. A temperature > 0 request runs exact sampled speculation (jang_tools.dsv41.dspark.SpecSampler:
the AR sampler's temperature / top_p / min_p / top_k distribution, Gumbel noise keyed by the request seed and
the absolute position, coupled drafting); the adaptive governor's plain AR steps draw from the same
position-keyed noise, so a fixed ``seed`` reproduces the same tokens whatever the governor chooses.
A request carrying vision tensors, logits processors (repetition / frequency / presence penalty, logit_bias)
or a logprobs request falls back to the inherited AR path (per request).
"""

from __future__ import annotations

import logging
import os
import time

import mlx.core as mx

from .single_batch_generator import Response, SingleBatchGenerator

logger = logging.getLogger(__name__)


def _adaptive_enabled() -> bool:
    return os.environ.get("VMLX_DSV41_DSPARK_ADAPTIVE", "1").strip().lower() not in {"", "0", "false", "off", "no"}


# --- DSpark prefill chunk width -----------------------------------------------------------------------------------
# The width trades compute against MLX's allocator working set.
#
# Compute: each chunk passes over every routed expert once, so wider chunks cost less per row, up to a width that
# depends on the GPU: on an M5 Max 2048 rows beat 1536 and 1024 while 2176-2432 lost the gain; on an M5 Ultra (two
# dies) 4096 still beat 2048 (JANGHT1.4, 32k tokens, memory not binding: 1024 1,515 pp/s, 2048 1,715, 4096 1,895). The
# widest width is _DIE_MAX_CHUNK per GPU die.
#
# Memory: MLX's allocator keeps freed buffers for reuse until active + cached memory reaches its GC line, 0.95 x
# Metal's recommended working set (or the memory limit, if lower); there it releases cached buffers on every cache
# miss and allocates fresh Metal buffers. Measured (DSV4.1, draft resident, cold prefill): a chunk runs at full speed
# while its PEAK active memory stays >= ~2.8 GiB under that line and slows 5-27 % closer to it; moving only the GC
# line (mx.set_memory_limit) reproduces it. JANG_2K on a 256 GB M5 Ultra rests 6.2 GiB under the line, JANGHT1.4 on a
# 128 GB M5 Max 7.2 GiB. A chunk's transient memory grows with its rows and with the context.
#
# So the width is CHOSEN per request from the live headroom (GC line - active memory now): the widest width up to the
# GPU's widest (or up to --prefill-step-size when that is set below vMLX's default 2048, as before) whose predicted
# peak keeps _GC_RESERVE_BYTES under the line, never below the fixed rule's 1024 (narrower chunks were slower in every
# measurement, including 1M-token prompts where no width keeps the reserve: 1024 417 pp/s, 512 350). A prompt that
# fits runs as one chunk; a longer one in the fewest chunks of at most that width, a multiple of 128 rows. The
# transient is predicted from a measured DSV4.1 model times a learned scale: after each DSpark prefill the measured
# peak (at the N-1 boundary, before the prompt snapshot) gives a measured / predicted ratio, and the scale is the
# median of the last 8. The chooser only picks a width: it never refuses or shortens a prompt. When the allocator
# numbers cannot be read it keeps the fixed rule of vMLX <= 1.6.82 (1024; one chunk for a prompt of <= 2048 tokens on
# a short session). VMLX_DSV41_DSPARK_PREFILL_STEP pins the width (diagnostics).
_GiB = 1 << 30
_DIE_MAX_CHUNK = 2048      # widest width per GPU die (an M5 Ultra has two)
_ALIGN = 128               # a split prompt's chunk width is a multiple of this
_DEFAULT_STEP = 2048       # vMLX's --prefill-step-size default; a smaller configured step caps the width
_GC_RESERVE_BYTES = int(2.8 * _GiB)
# Transient of one chunk (peak active minus active before it), draft resident:
#   _T_BASE + rows * _T_ROW + context * _T_CTX
# (upper envelope of the measured chunks, both bundles).
_T_BASE = int(1.0 * _GiB)
_T_ROW = int(1.4 * _GiB) // 1024
_T_CTX = int(0.9 * _GiB) // 100_000
_SCALE_WINDOW = 8          # observations the learned scale keeps (their median is the scale)
_OBSERVE_MIN_ROWS = 256    # shorter prefills are dominated by the base term; not used for learning
_SCALE_FLOOR = 0.5

# The fixed rule of vMLX <= 1.6.82, kept as the fallback when the allocator numbers cannot be read; its step is also
# the chooser's narrowest width.
_FALLBACK_STEP = 1024
_MIN_CHUNK = _FALLBACK_STEP
_SHORT_PROMPT_ONE_CHUNK = 2048
_SHORT_CHUNK_MAX_CONTEXT = 65536


def _prefill_override() -> int:
    try:
        return max(0, int(os.environ.get("VMLX_DSV41_DSPARK_PREFILL_STEP", "0")))
    except ValueError:
        return 0


def _draft_prefill_step(configured: int) -> int:
    """Step of this generator's inherited AR prefill (VL turns), which keeps the fixed rule."""
    override = _prefill_override()
    if override > 0:
        return override
    return min(int(configured), _FALLBACK_STEP)


def _fallback_chunk(n_tokens: int, start: int, step: int) -> int:
    """The fixed rule of vMLX <= 1.6.82: the step, or one chunk for a short prompt on a short session."""
    if step < n_tokens <= _SHORT_PROMPT_ONE_CHUNK and start + n_tokens <= _SHORT_CHUNK_MAX_CONTEXT:
        return n_tokens
    return step


def _device_info() -> dict:
    return mx.device_info() if hasattr(mx, "device_info") else mx.metal.device_info()


def _gpu_max_chunk() -> int:
    """_DIE_MAX_CHUNK x the GPU's dies: two for an Ultra (two Max dies), else one; one when the name is unreadable."""
    try:
        name = str(_device_info().get("device_name", ""))
    except Exception:  # noqa: BLE001
        name = ""
    return _DIE_MAX_CHUNK * (2 if "ultra" in name.lower() else 1)


def _gc_line_bytes() -> int | None:
    """MLX's allocator GC line: 0.95 x max_recommended_working_set_size, or the memory limit if one is set lower."""
    line = int(0.95 * int(_device_info()["max_recommended_working_set_size"]))
    get_limit = getattr(mx, "get_memory_limit", None)
    if callable(get_limit):
        limit = int(get_limit())
        if limit > 0:
            line = min(line, limit)
    return line if line > 0 else None


def _active_bytes() -> int:
    get = getattr(mx, "get_active_memory", None) or mx.metal.get_active_memory
    return int(get())


def _split(body: int, widest: int) -> int:
    """Width for body rows: one chunk if they fit in widest, else the fewest chunks of at most widest rows, the width
    the near-equal share rounded up to a multiple of _ALIGN (an unaligned 950 + 949 split of 1,899 rows ran 2 % slower
    than 1024 + 875 on JANG_2K and JANGHT alike)."""
    widest = max(1, int(widest))
    if body <= widest:
        return body
    top = widest // _ALIGN * _ALIGN or widest
    k = -(-body // top)
    share = -(-body // k)
    return min(top, -(-share // _ALIGN) * _ALIGN)


class PrefillChunkChooser:
    """Chooses the DSpark prefill chunk width of each request from the allocator headroom (see the notes above).

    choose(n_tokens, start) -> width for jang_tools DSparkRunner.prefill(chunk_size=...); begin() / end() bracket the
    prefill so the measured peak teaches the transient scale. The learned state is shared by every chooser in the
    process (the scheduler rebuilds its generator when sampling parameters change; the model and memory stay)."""

    _ratios: list = []      # measured / predicted transient of recent prefills (process-wide)

    def __init__(self, configured_step: int | None = None, *, gc_line=_gc_line_bytes, active=_active_bytes,
                 max_chunk=None):
        cfg = int(configured_step or _DEFAULT_STEP)
        widest = int(max_chunk) if max_chunk else _gpu_max_chunk()
        self.cap = max(1, cfg) if cfg < _DEFAULT_STEP else widest
        self.configured_step = cfg
        self._gc_line = gc_line
        self._active = active
        self.last = None

    # --- model -----------------------------------------------------------------------------------------------------
    @classmethod
    def scale(cls) -> float:
        """Median measured / predicted ratio of the recent prefills (1.0 before any), never below _SCALE_FLOOR. The
        median, not the largest: a request's peak is its worst chunk (the transient is a sawtooth over the context), so
        one ratio over-states the typical chunk, and the largest would keep every later request narrower than needed."""
        if not cls._ratios:
            return 1.0
        r = sorted(cls._ratios)
        mid = len(r) // 2
        med = r[mid] if len(r) % 2 else 0.5 * (r[mid - 1] + r[mid])
        return max(_SCALE_FLOOR, med)

    @staticmethod
    def predicted_transient(rows: int, ctx: int, scale: float = 1.0) -> float:
        return scale * (_T_BASE + rows * _T_ROW + ctx * _T_CTX)

    def widest(self, headroom: int, ctx: int) -> int:
        """Widest width (_MIN_CHUNK .. cap) whose predicted peak keeps _GC_RESERVE_BYTES under the GC line; _MIN_CHUNK
        when none does."""
        s = self.scale()
        room = headroom - _GC_RESERVE_BYTES
        rows = int((room / s - _T_BASE - ctx * _T_CTX) // _T_ROW)
        return max(min(_MIN_CHUNK, self.cap), min(self.cap, rows))

    # --- choice ----------------------------------------------------------------------------------------------------
    def choose(self, n_tokens: int, start: int = 0) -> int:
        n_tokens, start = int(n_tokens), max(0, int(start))
        override = _prefill_override()
        if override > 0:
            self.last = {"rule": "override", "chunk": override}
            return override
        body = n_tokens - 1  # the last prompt token runs alone after the N-1 boundary
        try:
            line = self._gc_line()
            active = self._active()
            if not line or active <= 0:
                raise ValueError("no allocator numbers")
        except Exception as exc:  # noqa: BLE001 - failed detection keeps the fixed rule, never something stricter
            step = min(self.configured_step, _FALLBACK_STEP)
            chunk = _fallback_chunk(n_tokens, start, step)
            self.last = {"rule": "fallback", "reason": str(exc), "chunk": chunk}
            return chunk
        ctx = start + n_tokens
        widest = self.widest(line - active, ctx)
        chunk = _split(body, widest) if body > 0 else widest
        self.last = {"rule": "adaptive", "chunk": chunk, "widest": widest,
                     "headroom_gib": round((line - active) / _GiB, 2), "scale": round(self.scale(), 3),
                     "n_tokens": n_tokens, "start": start}
        return chunk

    # --- learning --------------------------------------------------------------------------------------------------
    def begin(self):
        """Before the prefill: active bytes now (None if unreadable); resets MLX's peak counter so the boundary read is
        this prefill's peak."""
        try:
            a0 = self._active()
            reset = getattr(mx, "reset_peak_memory", None) or mx.metal.reset_peak_memory
            reset()
            return a0
        except Exception:  # noqa: BLE001 - learning is optional
            return None

    def end(self, mark, rows: int, n_tokens: int, start: int = 0, peak: int | None = None) -> None:
        """After the body chunks (at the N-1 boundary): record measured / predicted transient for the width used."""
        if mark is None or rows < _OBSERVE_MIN_ROWS or (self.last or {}).get("rule") != "adaptive":
            return
        try:
            if peak is None:
                get = getattr(mx, "get_peak_memory", None) or mx.metal.get_peak_memory
                peak = int(get())
        except Exception:  # noqa: BLE001
            return
        measured = int(peak) - int(mark)
        if measured <= 0:
            return
        ratio = measured / self.predicted_transient(int(rows), int(start) + int(n_tokens))
        cls = type(self)
        cls._ratios.append(ratio)
        del cls._ratios[:-_SCALE_WINDOW]
        self.last = dict(self.last or {}, measured_gib=round(measured / _GiB, 3), ratio=round(ratio, 3))

    def describe(self) -> dict:
        return dict(self.last or {}, scale=round(self.scale(), 3), ratios=[round(r, 3) for r in self._ratios])


class DSV41DSparkGenerator(SingleBatchGenerator):
    """SingleBatchGenerator that decodes V4.1 via DSpark speculation when a draft is attached."""

    def __init__(self, model, *args, **kwargs):
        super().__init__(model, *args, **kwargs)
        from jang_tools.dsv41.dspark import DSparkRunner

        # The DSpark prefill chooses its width per request (PrefillChunkChooser); this generator's inherited AR
        # prefill (VL turns), which runs with the same draft resident, keeps the fixed step.
        self._chunk_chooser = PrefillChunkChooser(self.prefill_step_size)
        self.prefill_step_size = _draft_prefill_step(self.prefill_step_size)

        self._dspark = getattr(model, "_dspark", None)
        self._runtime = getattr(model, "runtime", None)
        self._runner = None
        if self._dspark is not None and self._runtime is not None:
            self._runner = DSparkRunner(self._runtime, self._dspark, adaptive=_adaptive_enabled())
            logger.info("DSV4.1 DSpark generator active (adaptive=%s, bs=%s)",
                        _adaptive_enabled(), getattr(self._dspark, "bs", "?"))
        else:
            logger.warning("DSV4.1 DSpark generator constructed without a draft/runtime; AR fallback")

    def request_progress(self, uid: int):
        """Positions written into this request's DSV4.1 session (``_SeqState.pos``), or None.

        DSpark prefills the whole prompt inside one ``DSparkRunner.prefill`` call, which advances the session
        chunk by chunk while no output token exists. Without this the server's stream guard read 0 for a long
        prefill and killed it as wedged once its grace windows ran out: a 1.04M-token prompt (512-token chunks)
        died at ~2,760 s, three 900 s windows, just short of its first token. Integer reads only: the liveness
        probe calls this from outside the model thread. Covers the inherited AR prefill (VL turns) too."""
        req = self._find(uid)
        if req is None:
            return None
        try:
            return max(0, int(req.cache[0]._seq.pos))
        except (AttributeError, IndexError, TypeError, ValueError):
            return None

    # --- per-request routing -------------------------------------------------
    @staticmethod
    def _is_vl(req) -> bool:
        return bool(getattr(req, "pixel_values", None) is not None
                    or getattr(req, "pixel_values_videos", None) is not None
                    or getattr(req, "dsv41_vl_images", None) is not None)

    @staticmethod
    def _sampler_temperature(sampler):
        """The temperature the scheduler built ``sampler`` with, or None when it is unrecorded."""
        spec = getattr(sampler, "_vmlx_sampling", None)
        if isinstance(spec, dict) and spec.get("temperature") is not None:
            return float(spec["temperature"])
        temp = getattr(sampler, "temp", None)  # vmlx_engine.sampling stamps every sampler it builds
        if isinstance(temp, (int, float)) and not isinstance(temp, bool):
            return float(temp)
        return None

    def _is_greedy(self, req) -> bool:
        """True only for a temperature == 0 request without logits processors or logprobs.

        Greedy means argmax: the scheduler tags it ``_vmlx_greedy`` (vmlx_engine.sampling tags it
        ``_vmlx_is_greedy``) and records temperature 0. ``_vmlx_accepts_logits`` is NOT a greedy marker:
        make_sampler also sets it on the compact top-k sampler (temperature > 0, top_k > 0, min_p == 0),
        and treating that as greedy decoded a sampled top_k request greedily. Sampled requests
        (temperature > 0 with top-p / top-k / min-p) take the exact sampled path (_spec_sampler); logits
        processors (repetition / frequency / presence penalty, logit_bias) and logprobs requests take the
        inherited AR path, which applies them per token."""
        sampler = req.sampler or self.sampler
        if getattr(sampler, "_vmlx_greedy", False) is True or getattr(sampler, "_vmlx_is_greedy", False) is True:
            greedy = True
        else:
            greedy = self._sampler_temperature(sampler) == 0.0
        if not greedy:
            return False
        if req.logits_processors:
            return False
        if self._logprobs_required(req):
            return False
        return True

    def _spec_sampler(self, req):
        """(ok, SpecSampler | None): greedy -> (True, None); a temperature > 0 sampler built by the scheduler's
        make_sampler (tagged _vmlx_sampling) without logits processors / logprobs -> (True, SpecSampler);
        anything else -> (False, None), the AR path."""
        if self._is_greedy(req):
            return True, None
        if os.environ.get("VMLX_DSV41_DSPARK_SAMPLED", "1").strip().lower() in {"0", "false", "off", "no"}:
            return False, None
        sampler = req.sampler or self.sampler
        spec = getattr(sampler, "_vmlx_sampling", None)
        if not isinstance(spec, dict) or not float(spec.get("temperature", 0.0)) > 0:
            return False, None
        if req.logits_processors or self._logprobs_required(req):
            return False, None
        try:
            from jang_tools.dsv41.dspark import SpecSampler
        except ImportError:  # a jang_tools without the exact sampled path (before jangq #30): decode AR
            logger.warning("DSV4.1 DSpark: installed jang_tools has no SpecSampler; sampled requests decode AR")
            return False, None

        return True, SpecSampler(spec["temperature"], top_p=spec.get("top_p", 0.0), min_p=spec.get("min_p", 0.0),
                                 top_k=spec.get("top_k", 0), seed=spec.get("seed"))

    def _use_dspark(self, req) -> bool:
        if self._runner is None or self._is_vl(req):
            return False
        ok, sampler = self._spec_sampler(req)
        req._dspark_sampler = sampler
        return ok

    # --- decode override ------------------------------------------------------
    def _start_request(self, req) -> Response:
        if not self._use_dspark(req):
            return super()._start_request(req)  # VL / sampling / processors / logprobs -> AR path
        if req.cache is None:
            req.cache = self._make_new_cache()
        self._refresh_thread_stream()
        req.prefill_usage = None
        req.prefill_started = time.perf_counter()
        req.prompt_processed = True
        # DSpark drives the ENGINE cache (req.cache duck-types the runtime Session) for BOTH cold and
        # prefix-HIT requests, so DSpark output populates the prefix cache AND a restored prefix is
        # CONTINUED by DSpark (not re-prefilled). On a hit the scheduler restores req.cache and passes
        # only the uncached tail as prompt_tokens; prefill() forwards it from the session's absolute
        # pos, in chunks of the chosen width, and calls back at the exact N-1 prompt boundary -- the
        # state the scheduler stores under prompt[:-1] (same contract as the AR path).
        from jang_tools.dsv4_1.mlx_model import _relink

        seq = _relink(req.cache)
        boundary = {}
        prompt = list(req.prompt_tokens)
        start = int(getattr(seq, "pos", 0) or 0)
        chooser = self._chunk_chooser
        chunk = chooser.choose(len(prompt), start)
        mark = chooser.begin()

        def _on_boundary(_sess):
            # Peak of the body chunks, read before the prompt snapshot allocates.
            chooser.end(mark, min(chunk, len(prompt) - 1), len(prompt), start)
            snap = self._clone_dsv41_prompt_boundary(req)
            # Carry the draft-window seed of the boundary's last positions, so a later prefix hit
            # on this entry drafts with a full window (DSparkRunner.prefill reseeds from it).
            tail_fn = getattr(self._dspark, "main_tail", None)
            tail = tail_fn() if callable(tail_fn) else None
            if snap and tail is not None and int(tail[1]) + int(tail[0].shape[1]) == int(snap[0].offset):
                snap[0]._seq.dspark_tail = (mx.array(tail[0]), int(tail[1]))
                mx.eval(snap[0]._seq.dspark_tail[0])
            boundary["snapshot"] = snap

        with self._stream_context():
            sess, pending, c = self._runner.prefill(prompt, session=seq, on_boundary=_on_boundary, chunk_size=chunk)
            mx.eval(pending)
        logger.info("DSV4.1 DSpark prefill %s: %s tokens from pos %s, chunk %s (%s)", req.uid, len(prompt), start,
                    chunk, chooser.describe())
        req.prompt_cache_snapshot = boundary.get("snapshot")
        req.context_tokens.extend(prompt)
        req.prefill_usage = {
            "tokens": len(prompt),
            "seconds": time.perf_counter() - req.prefill_started,
            "scope": "model_prefill_and_prompt_state",
        }
        req.prefill_started = None
        req._dspark_stats = {}
        # SingleBatchGenerator emits at least one token whatever max_tokens says (its length check runs
        # after the first token); keep that contract so the stream always yields a first token.
        req._dspark_stream = self._runner.decode_stream(
            sess, pending, c, max(1, int(req.max_tokens)), eos_ids=set(t for t in self.stop_tokens if isinstance(t, int)),
            stats=req._dspark_stats, sampler=getattr(req, "_dspark_sampler", None),
        )
        req._dspark_on = True
        req.next_token = self._pull(req)  # first committed token (always an int, max_new >= 1)
        return self._emit_dspark(req)

    def _pull(self, req):
        gen = getattr(req, "_dspark_stream", None)
        if gen is None:
            return None
        try:
            with self._stream_context():  # same stream the prefill ran on
                return int(next(gen))
        except StopIteration:
            return None

    def _emit_dspark(self, req) -> Response:
        current = req.next_token
        if current is None:
            # decode_stream runs with max_new >= 1, so its first pull always yields a token, and a later
            # call happens only after a token whose lookahead pull returned a token (a None lookahead
            # finishes the request below). Never fabricate a token here.
            raise RuntimeError(f"DSV4.1 DSpark stream for request {req.uid} produced no token")
        # Pull the NEXT token now (one-token lookahead parity with SBG) to know if the stream ended.
        req.next_token = self._pull(req)
        req.output_tokens.append(current)
        req.context_tokens.append(current)
        req.finish_reason = self._finish_reason_after_token(req, current)
        if req.finish_reason is None and req.next_token is None:
            req.finish_reason = "stop"  # stream ended without a stop/length trigger
        resp = Response(uid=req.uid, token=current, logprobs=None, finish_reason=req.finish_reason,
                        current_state=req.current_state, match_sequence=req.match_sequence,
                        prompt_cache=req.cache, all_tokens=list(req.context_tokens),
                        prompt_cache_snapshot=getattr(req, 'prompt_cache_snapshot', None), prefill_usage=req.prefill_usage,
                        mixed_swa_boundary=None, stable_root_snapshot=None, boundary_checkpoints=None)
        if req.finish_reason is not None:
            self._request = None
            self._log_dspark_stats(req)
        return resp

    @staticmethod
    def _log_dspark_stats(req) -> None:
        st = getattr(req, "_dspark_stats", None) or {}
        if not st:
            return
        logger.info(
            "DSV4.1 DSpark request %s: emitted=%s steps=%s spec_steps=%s ar_steps=%s accept=%s/%s "
            "spec_ms_per_tok=%s ar_ms_per_tok=%s",
            req.uid, st.get("emitted"), st.get("steps"), st.get("spec_steps"), st.get("ar_steps"),
            st.get("accepted"), st.get("drafted"),
            None if st.get("spec_s_per_tok") is None else round(st["spec_s_per_tok"] * 1000, 2),
            None if st.get("ar_s_per_tok") is None else round(st["ar_s_per_tok"] * 1000, 2),
        )

    def _yield_current_and_schedule_next(self, req, *, prompt_cache_snapshot=None) -> Response:
        if getattr(req, "_dspark_on", False):
            return self._emit_dspark(req)
        return super()._yield_current_and_schedule_next(req, prompt_cache_snapshot=prompt_cache_snapshot)
