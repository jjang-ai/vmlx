# SPDX-License-Identifier: Apache-2.0
"""
Simple engine for maximum single-user throughput.

This engine wraps mlx-lm directly with zero overhead for optimal
performance when serving a single user at a time.
"""

import asyncio
import functools
import logging
from collections.abc import AsyncIterator
from concurrent.futures import ThreadPoolExecutor
from typing import Any

from ..api.tool_calling import check_and_inject_fallback_tools, convert_tools_for_template
from ..api.utils import clean_output_text, is_mllm_model
from ..errors import PromptTooLongError
from ..mlx_memory import clear_mlx_memory_cache
from ..model_config_registry import get_model_config_registry
from ..utils.chat_template_kwargs import (
    build_chat_template_kwargs,
    ensure_thinking_off_sentinel,
)
from .base import BaseEngine, GenerationOutput

logger = logging.getLogger(__name__)


class SimpleEngine(BaseEngine):
    """
    Simple engine for direct model calls.

    This engine provides maximum throughput for single-user scenarios
    by calling mlx-lm/mlx-vlm directly without batching overhead.
    """

    def __init__(
        self,
        model_name: str,
        trust_remote_code: bool = True,
        enable_cache: bool = True,
        force_mllm: bool = False,
    ):
        """
        Initialize the simple engine.

        Args:
            model_name: HuggingFace model name or local path
            trust_remote_code: Whether to trust remote code
            enable_cache: Enable VLM cache for multimodal models
            force_mllm: Force loading as MLLM even if not auto-detected
        """
        self._model_name = model_name
        self._trust_remote_code = trust_remote_code
        self._enable_cache = enable_cache
        self._is_mllm = is_mllm_model(model_name, force_mllm=force_mllm)

        self._model = None
        self._loaded = False

        # Lock to serialize MLX operations (prevents Metal command buffer conflicts)
        self._generation_lock = asyncio.Lock()
        # Abort flag — checked between tokens in stream_generate
        self._abort_requested = False
        self._current_request_id: str | None = None
        self._model_executor: ThreadPoolExecutor | None = None

    def _ensure_model_executor(self) -> ThreadPoolExecutor:
        executor = getattr(self, "_model_executor", None)
        if executor is None:
            executor = ThreadPoolExecutor(
                max_workers=1,
                thread_name_prefix="simple-engine-model",
            )
            self._model_executor = executor
        return executor

    async def _run_model_call(self, fn, /, *args, **kwargs):
        """Run all MLX model work on SimpleEngine's one model thread.

        MLX streams are thread-local. Loading on one thread and later calling
        generate()/next() from arbitrary default-executor threads can crash
        JANG/JANGTQ/VLM/hybrid kernels with `There is no Stream(gpu,N) in
        current thread`. BatchedEngine already pins load+step to one executor;
        SimpleEngine needs the same invariant for direct/no-CB mode.
        """
        loop = asyncio.get_running_loop()
        call = functools.partial(fn, *args, **kwargs)
        return await loop.run_in_executor(self._ensure_model_executor(), call)

    @property
    def model_name(self) -> str:
        """Get the model name."""
        return self._model_name

    @property
    def is_mllm(self) -> bool:
        """Check if this is a multimodal model."""
        return self._is_mllm

    def _model_family_name(self) -> str | None:
        try:
            return get_model_config_registry().lookup(self._model_name).family_name
        except Exception:
            return None

    def _model_tool_parser_name(self) -> str | None:
        try:
            return get_model_config_registry().lookup(self._model_name).tool_parser
        except Exception:
            return None

    def _chat_template_enable_thinking_for_render(
        self, requested_enable_thinking: bool | None
    ) -> bool | None:
        """Return the template kwarg that should be used for prompt rendering.

        Do not silently flip a request's thinking mode at render time. MiMo's
        shipped template owns a distinct thinking-off prompt rail, and live
        JANG_2L continuous-batching proof showed that rendering the thinking-on
        rail and suppressing tags at decode leaks reasoning text into visible
        content. First-token EOS handling belongs at the logits boundary, not in
        a hidden prompt-mode rewrite.
        """

        return requested_enable_thinking

    def _messages_are_text_only(self, messages: list[dict[str, Any]]) -> bool:
        for message in messages:
            content = message.get("content")
            if isinstance(content, str) or content is None:
                continue
            if not isinstance(content, list):
                return False
            for part in content:
                if isinstance(part, str):
                    continue
                if not isinstance(part, dict):
                    return False
                part_type = str(part.get("type", "") or "").lower()
                if part_type and part_type != "text":
                    return False
                if any(key in part for key in ("image", "image_url", "video", "video_url", "audio", "audio_url")):
                    return False
        return True

    def _mimo_text_only_prompt(
        self,
        messages: list[dict[str, Any]],
        *,
        enable_thinking: bool | None,
        template_tools: list[dict] | None,
    ) -> tuple[str, Any]:
        processor = getattr(self._model, "processor", None)
        tokenizer = (
            getattr(processor, "tokenizer", None)
            if processor is not None
            else None
        ) or processor
        if tokenizer is None:
            raise RuntimeError("MiMo text-only route requires a processor tokenizer")
        apply_template = getattr(self._model, "_apply_chat_template", None)
        if not callable(apply_template):
            raise RuntimeError("MiMo text-only route requires MLLM chat template support")
        return apply_template(
            messages,
            enable_thinking=enable_thinking,
            tools=template_tools,
        ), tokenizer

    def _mimo_text_only_generate(
        self,
        *,
        prompt: str,
        tokenizer: Any,
        max_tokens: int,
        temperature: float,
        top_p: float,
        stop: list[str] | None,
        enable_thinking: bool | None,
        kwargs: dict[str, Any],
    ) -> GenerationOutput:
        language_model = getattr(getattr(self._model, "model", None), "language_model", None)
        if language_model is None:
            raise RuntimeError("MiMo text-only route requires language_model")

        from mlx_lm import generate
        from mlx_lm.sample_utils import make_logits_processors
        from ..sampling import make_sampler

        top_k = int(kwargs.pop("top_k", 0) or 0)
        min_p = float(kwargs.pop("min_p", 0.0) or 0.0)
        repetition_penalty = float(kwargs.pop("repetition_penalty", 1.0) or 1.0)
        kwargs.pop("use_cache", None)
        kwargs.pop("tools", None)

        sampler = make_sampler(temp=temperature, top_p=top_p, min_p=min_p, top_k=top_k)
        logits_processors = self._mimo_thinking_off_logits_processors(
            tokenizer,
            enable_thinking=enable_thinking,
        )
        if repetition_penalty and repetition_penalty != 1.0:
            logits_processors.extend(
                make_logits_processors(
                    repetition_penalty=repetition_penalty,
                )
            )
        if not logits_processors:
            logits_processors = None

        output_text = generate(
            language_model,
            tokenizer,
            prompt=prompt,
            max_tokens=max_tokens,
            sampler=sampler,
            logits_processors=logits_processors,
            verbose=False,
        )
        finish_reason = "length"
        if stop and output_text:
            for stop_seq in stop:
                index = output_text.find(stop_seq)
                if index >= 0:
                    output_text = output_text[:index]
                    finish_reason = "stop"
                    break
        try:
            tokens = tokenizer.encode(output_text)
        except Exception:
            tokens = []
        if finish_reason != "stop":
            finish_reason = "length" if len(tokens) >= max_tokens else "stop"
        try:
            prompt_tokens = len(tokenizer.encode(prompt))
        except Exception:
            prompt_tokens = 0
        return GenerationOutput(
            text=clean_output_text(output_text),
            raw_text=output_text,
            tokens=tokens,
            prompt_tokens=prompt_tokens,
            completion_tokens=len(tokens),
            finish_reason=finish_reason,
        )

    def _mimo_thinking_off_logits_processors(
        self,
        tokenizer: Any,
        *,
        enable_thinking: bool | None,
    ) -> list[Any]:
        """Suppress MiMo native thinking tags when API thinking is disabled.

        MiMo's stable text prompt for thinking-off serving is the plain
        assistant prefix; its native ``enable_thinking=False`` template closes
        ``<think></think>`` before generation and can first-token-stop. Because
        the stable render does not itself prevent tag emission, the decode loop
        must make the native thinking delimiters unavailable for the requested
        API rail. Current first-token probes also show the primary EOS marker
        as the top token for failing system+long rows, so EOS is suppressed
        only for the first generated token. This is a logits-boundary policy,
        not output injection.
        """

        if enable_thinking is not False:
            return []

        token_ids: set[int] = set()
        for token in ("<think>", "</think>"):
            try:
                encoded = tokenizer.encode(token, add_special_tokens=False)
            except TypeError:
                encoded = tokenizer.encode(token)
            except Exception:
                encoded = []
            if len(encoded) == 1:
                try:
                    token_ids.add(int(encoded[0]))
                except Exception:
                    continue

        processors = []
        if token_ids:

            def _suppress_mimo_thinking_tags(_, logits):
                import mlx.core as mx

                indices = mx.array(sorted(token_ids))
                return logits.at[:, indices].add(-float("inf"))

            processors.append(_suppress_mimo_thinking_tags)

        eos_token_ids: set[int] = set()
        for token in ("<|im_end|>",):
            try:
                encoded = tokenizer.encode(token, add_special_tokens=False)
            except TypeError:
                encoded = tokenizer.encode(token)
            except Exception:
                encoded = []
            if len(encoded) == 1:
                try:
                    eos_token_ids.add(int(encoded[0]))
                except Exception:
                    continue
        eos_id = getattr(tokenizer, "eos_token_id", None)
        if isinstance(eos_id, int):
            eos_token_ids.add(eos_id)

        if eos_token_ids:

            def _suppress_mimo_first_token_eos(tokens, logits):
                import mlx.core as mx

                try:
                    first_generation_token = int(tokens.shape[0]) <= 1
                except Exception:
                    first_generation_token = False
                if not first_generation_token:
                    return logits
                indices = mx.array(sorted(eos_token_ids))
                return logits.at[:, indices].add(-float("inf"))

            processors.append(_suppress_mimo_first_token_eos)

        return processors

    @property
    def tokenizer(self) -> Any:
        """Get the tokenizer."""
        if not self._loaded or self._model is None:
            return None
        if self._is_mllm:
            return getattr(self._model, "processor", None)
        return self._model.tokenizer

    def _estimate_mllm_chat_prompt_tokens(
        self,
        messages: list[dict[str, Any]],
        enable_thinking: bool | None = None,
    ) -> int:
        """Best-effort prompt-token count for direct MLLM streaming usage.

        Some mlx-vlm stream chunks do not carry prompt_tokens. The LLM direct
        stream path already falls back to tokenizer.encode(prompt); keep MLLM
        usage accounting equivalent by applying the same chat template the
        MLLM wrapper uses and counting that formatted prompt.
        """
        if self._model is None:
            return 0
        processor = getattr(self._model, "processor", None)
        tokenizer = (
            getattr(processor, "tokenizer", None)
            if processor is not None
            else None
        ) or processor
        if tokenizer is None or not hasattr(tokenizer, "encode"):
            return 0
        formatted_prompt = None
        apply_template = getattr(self._model, "_apply_chat_template", None)
        if callable(apply_template):
            try:
                formatted_prompt = apply_template(messages, enable_thinking)
            except Exception:
                formatted_prompt = None
        if not formatted_prompt:
            parts: list[str] = []
            for msg in messages:
                content = msg.get("content", "")
                if isinstance(content, str):
                    parts.append(content)
                elif isinstance(content, list):
                    for part in content:
                        if isinstance(part, dict) and isinstance(part.get("text"), str):
                            parts.append(part["text"])
            formatted_prompt = "\n".join(parts)
        try:
            return len(tokenizer.encode(formatted_prompt))
        except Exception:
            return 0

    async def start(self) -> None:
        """Start the engine (load model if not loaded)."""
        if self._loaded:
            return

        def _construct_and_load():
            if self._is_mllm:
                from ..models.mllm import MLXMultimodalLM

                model = MLXMultimodalLM(
                    self._model_name,
                    trust_remote_code=self._trust_remote_code,
                    enable_cache=self._enable_cache,
                )
            else:
                from ..models.llm import MLXLanguageModel

                model = MLXLanguageModel(
                    self._model_name,
                    trust_remote_code=self._trust_remote_code,
                )
            model.load()
            return model

        self._model = await self._run_model_call(_construct_and_load)
        self._loaded = True
        logger.info(f"SimpleEngine loaded: {self._model_name} (MLLM={self._is_mllm})")

        # Check for hybrid SSM model (MambaCache + KVCache layers)
        raw_model = getattr(self._model, "model", None)
        if raw_model is not None and hasattr(raw_model, "make_cache"):
            try:
                cache = raw_model.make_cache()
                cache_types = {type(c).__name__ for c in cache}
                kv_only = {"KVCache", "RotatingKVCache", "QuantizedKVCache", "TurboQuantKVCache"}
                if cache_types and not cache_types.issubset(kv_only):
                    logger.warning(
                        "Hybrid SSM model detected in simple mode — batch SSM "
                        "features not available. Use --continuous-batching for "
                        "full hybrid support."
                    )
            except Exception:
                pass

    async def stop(self) -> None:
        """Stop the engine and cleanup resources."""
        self._model = None
        self._loaded = False
        try:
            from ..models.mllm import reset_vlm_stream

            reset_vlm_stream()
        except Exception as e:
            logger.debug("Direct VLM stream reset skipped during SimpleEngine stop: %s", e)
        executor = getattr(self, "_model_executor", None)
        self._model_executor = None
        if executor is not None:
            executor.shutdown(wait=False, cancel_futures=True)
        logger.info("SimpleEngine stopped")

    def _prompt_token_count(self, prompt: str) -> int:
        tokenizer = getattr(getattr(self, "_model", None), "tokenizer", None)
        if tokenizer is None:
            return 0
        try:
            return len(tokenizer.encode(prompt, add_special_tokens=False))
        except TypeError:
            return len(tokenizer.encode(prompt))
        except Exception:
            return 0

    def _raise_if_prompt_over_limit(
        self,
        prompt: str,
        max_prompt_tokens: int,
        *,
        source: str,
    ) -> None:
        if max_prompt_tokens <= 0:
            return
        token_count = self._prompt_token_count(prompt)
        if token_count > max_prompt_tokens:
            raise PromptTooLongError(
                token_count,
                max_prompt_tokens,
                source=source,
            )

    async def generate(
        self,
        prompt: str,
        max_tokens: int = 256,
        temperature: float = 0.7,
        top_p: float = 0.9,
        stop: list[str] | None = None,
        **kwargs,
    ) -> GenerationOutput:
        """
        Generate a complete response (non-streaming).

        Args:
            prompt: Input text
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            top_p: Top-p sampling
            stop: Stop sequences
            **kwargs: Additional model-specific parameters

        Returns:
            GenerationOutput with complete text
        """
        if not self._loaded:
            await self.start()
        # SimpleEngine has no prefix cache — eat the bypass kwarg so it
        # doesn't leak into self._model.generate which would reject it.
        kwargs.pop("_bypass_prefix_cache", None)
        max_prompt_tokens = int(kwargs.pop("max_prompt_tokens", 0) or 0)
        self._raise_if_prompt_over_limit(
            prompt,
            max_prompt_tokens,
            source="tokenized prompt",
        )

        async with self._generation_lock:
            output = await self._run_model_call(
                self._model.generate,
                prompt=prompt,
                max_tokens=max_tokens,
                temperature=temperature,
                top_p=top_p,
                stop=stop,
                **kwargs,
            )

            # Clean output text. Preserve raw (pre-clean) so reasoning
            # parsers can still see marker tokens like Gemma 4's
            # `<|channel>thought\n...<channel|>` that clean_output_text
            # strips for display.
            raw_text = output.text
            text = clean_output_text(raw_text)

            return GenerationOutput(
                text=text,
                raw_text=raw_text,
                tokens=getattr(output, "tokens", []),
                logprobs=getattr(output, "logprobs", None),
                prompt_tokens=getattr(output, "prompt_tokens", 0),
                completion_tokens=getattr(
                    output, "completion_tokens", len(getattr(output, "tokens", []))
                ),
                finish_reason=output.finish_reason,
            )

    async def stream_generate(
        self,
        prompt: str,
        max_tokens: int = 256,
        temperature: float = 0.7,
        top_p: float = 0.9,
        stop: list[str] | None = None,
        **kwargs,
    ) -> AsyncIterator[GenerationOutput]:
        """
        Stream generation token by token.

        Args:
            prompt: Input text
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            top_p: Top-p sampling
            stop: Stop sequences
            **kwargs: Additional model-specific parameters

        Yields:
            GenerationOutput with incremental text
        """
        if not self._loaded:
            await self.start()
        # SimpleEngine has no prefix cache — eat the bypass kwarg so it
        # doesn't leak into self._model.generate which would reject it.
        kwargs.pop("_bypass_prefix_cache", None)
        max_prompt_tokens = int(kwargs.pop("max_prompt_tokens", 0) or 0)
        self._raise_if_prompt_over_limit(
            prompt,
            max_prompt_tokens,
            source="tokenized prompt",
        )

        async with self._generation_lock:
            accumulated_text = ""
            prompt_tokens = 0
            completion_tokens = 0
            finished = False
            self._abort_requested = False  # Reset at start of generation
            self._current_request_id = kwargs.pop("request_id", None)

            try:
                stream_iter = await self._run_model_call(
                    lambda: iter(
                        self._model.stream_generate(
                            prompt=prompt,
                            max_tokens=max_tokens,
                            temperature=temperature,
                            top_p=top_p,
                            stop=stop,
                            **kwargs,
                        )
                    )
                )
            except Exception as gen_err:
                logger.error(f"stream_generate failed to start: {gen_err}", exc_info=True)
                raise

            # Offload each next() call to thread pool so the event loop
            # stays responsive for health checks during long prefills
            _sentinel = object()
            def _next():
                try:
                    return next(stream_iter)
                except StopIteration:
                    return _sentinel

            while True:
                chunk = await self._run_model_call(_next)
                if chunk is _sentinel:
                    break

                # Check abort flag between tokens
                if self._abort_requested:
                    self._abort_requested = False
                    logger.info("SimpleEngine: generation aborted by request")
                    yield GenerationOutput(
                        text=accumulated_text,
                        new_text="",
                        prompt_tokens=prompt_tokens,
                        completion_tokens=completion_tokens,
                        finished=True,
                        finish_reason="abort",
                    )
                    break

                prompt_tokens = (
                    chunk.prompt_tokens
                    if hasattr(chunk, "prompt_tokens")
                    else prompt_tokens
                )
                completion_tokens += 1
                new_text = chunk.text if hasattr(chunk, "text") else str(chunk)
                accumulated_text += new_text

                finished = (
                    getattr(chunk, "finished", False) or completion_tokens >= max_tokens
                )
                finish_reason = None
                if finished:
                    finish_reason = getattr(chunk, "finish_reason", "stop")
                    if prompt_tokens == 0:
                        try:
                            prompt_tokens = len(self._model.tokenizer.encode(prompt))
                        except Exception:
                            prompt_tokens = 0

                yield GenerationOutput(
                    text=accumulated_text,
                    new_text=new_text,
                    prompt_tokens=prompt_tokens,
                    completion_tokens=completion_tokens,
                    finished=finished,
                    finish_reason=finish_reason,
                )

                if finished:
                    break

            if not finished:
                if completion_tokens == 0:
                    logger.warning("stream_generate yielded zero tokens — model may have failed silently")
                if prompt_tokens == 0:
                    try:
                        prompt_tokens = len(self._model.tokenizer.encode(prompt))
                    except Exception:
                        prompt_tokens = 0
                yield GenerationOutput(
                    text=accumulated_text,
                    new_text="",
                    prompt_tokens=prompt_tokens,
                    completion_tokens=completion_tokens,
                    finished=True,
                    finish_reason="stop" if completion_tokens > 0 else None,
                )

            self._current_request_id = None

    async def chat(
        self,
        messages: list[dict[str, Any]],
        max_tokens: int = 256,
        temperature: float = 0.7,
        top_p: float = 0.9,
        tools: list[dict] | None = None,
        images: list[str] | None = None,
        videos: list[str] | None = None,
        **kwargs,
    ) -> GenerationOutput:
        """
        Chat completion (non-streaming).

        Args:
            messages: List of chat messages
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            top_p: Top-p sampling
            tools: Optional tool definitions
            images: Optional image URLs/paths
            videos: Optional video URLs/paths
            **kwargs: Additional model-specific parameters

        Returns:
            GenerationOutput with assistant response
        """
        if not self._loaded:
            await self.start()
        # SimpleEngine has no prefix cache — eat the bypass kwarg so it
        # doesn't leak into self._model.generate which would reject it.
        kwargs.pop("_bypass_prefix_cache", None)

        # Convert tools for template if provided
        template_tools = convert_tools_for_template(tools) if tools else None

        # Pop vmlx-engine-specific kwargs early to prevent leaking to mlx-lm calls
        extra_ct_kwargs = kwargs.pop("chat_template_kwargs", None)
        reasoning_effort = kwargs.pop("reasoning_effort", None)
        kwargs.pop("request_id", None)
        thinking_enabled = kwargs.pop("enable_thinking", True)
        prompt_suffix = kwargs.pop("prompt_suffix", None)
        skip_gen_prompt = kwargs.pop("skip_generation_prompt", False)
        max_prompt_tokens = int(kwargs.pop("max_prompt_tokens", 0) or 0)

        async with self._generation_lock:
            if self._is_mllm:
                if (
                    self._model_family_name() == "mimo_v2"
                    and self._messages_are_text_only(messages)
                    and not images
                    and not videos
                ):
                    prompt, tokenizer = self._mimo_text_only_prompt(
                        messages,
                        enable_thinking=thinking_enabled,
                        template_tools=template_tools,
                    )
                    if prompt_suffix:
                        prompt += prompt_suffix
                    self._raise_if_prompt_over_limit(
                        prompt,
                        max_prompt_tokens,
                        source="rendered MiMo text-only chat prompt",
                    )
                    output = await self._run_model_call(
                        self._mimo_text_only_generate,
                        prompt=prompt,
                        tokenizer=tokenizer,
                        max_tokens=max_tokens,
                        temperature=temperature,
                        top_p=top_p,
                        stop=None,
                        enable_thinking=thinking_enabled,
                        kwargs=dict(kwargs),
                    )
                    return output

                # For MLLM, use the chat method which handles images/videos
                # Run in thread pool to allow asyncio timeout to work
                mllm_kwargs = dict(kwargs)
                if thinking_enabled is not None:
                    mllm_kwargs["enable_thinking"] = thinking_enabled
                if reasoning_effort:
                    mllm_kwargs["reasoning_effort"] = reasoning_effort
                if extra_ct_kwargs:
                    # Strip reserved keys (Concern #14): see above.
                    mllm_kwargs.update({
                        k: v for k, v in extra_ct_kwargs.items()
                        if k not in ("tokenize", "add_generation_prompt")
                    })
                if template_tools:
                    mllm_kwargs["tools"] = template_tools
                output = await self._run_model_call(
                    self._model.chat,
                    messages=messages,
                    max_tokens=max_tokens,
                    temperature=temperature,
                    top_p=top_p,
                    **mllm_kwargs,
                )
                raw_text = output.text
                text = clean_output_text(raw_text)
                return GenerationOutput(
                    text=text,
                    raw_text=raw_text,
                    prompt_tokens=output.prompt_tokens,
                    completion_tokens=output.completion_tokens,
                    finish_reason=output.finish_reason,
                )
            else:
                # Apply template manually so enable_thinking and reasoning_effort
                # are properly passed to tokenizer.apply_chat_template()
                tokenizer = self._model.tokenizer
                if hasattr(tokenizer, "apply_chat_template"):
                    # Ensure tool_calls arguments are dicts, not JSON strings.
                    for msg in messages:
                        for tc in msg.get("tool_calls") or []:
                            fn = tc.get("function", {})
                            args = fn.get("arguments")
                            if isinstance(args, str):
                                try:
                                    import json
                                    fn["arguments"] = json.loads(args)
                                except (json.JSONDecodeError, TypeError):
                                    fn["arguments"] = {}

                    tpl_kwargs = build_chat_template_kwargs(
                        enable_thinking=self._chat_template_enable_thinking_for_render(
                            thinking_enabled
                        ),
                        extra=extra_ct_kwargs,
                        tokenize=False,
                        add_generation_prompt=not skip_gen_prompt,
                    )
                    if template_tools:
                        tpl_kwargs["tools"] = template_tools
                    if reasoning_effort:
                        tpl_kwargs["reasoning_effort"] = reasoning_effort

                    try:
                        prompt = tokenizer.apply_chat_template(messages, **tpl_kwargs)
                    except Exception as template_err:
                        # Progressively strip non-essential kwargs to preserve tools/thinking
                        strip_order = [
                            k for k in tpl_kwargs
                            if k not in ("tokenize", "add_generation_prompt")
                        ]
                        prompt = None
                        for key in reversed(strip_order):
                            del tpl_kwargs[key]
                            try:
                                prompt = tokenizer.apply_chat_template(messages, **tpl_kwargs)
                                stripped = [k for k in strip_order if k not in tpl_kwargs]
                                logger.warning(
                                    f"Chat template succeeded after stripping: {stripped} "
                                    f"(original error: {template_err})"
                                )
                                break
                            except Exception:
                                continue
                        if prompt is None:
                            prompt = tokenizer.apply_chat_template(messages, **tpl_kwargs)

                    prompt = check_and_inject_fallback_tools(
                        prompt,
                        messages,
                        template_tools,
                        tokenizer,
                        tpl_kwargs,
                        tool_parser_id=self._model_tool_parser_name(),
                    )

                    if thinking_enabled is False:
                        prompt = ensure_thinking_off_sentinel(
                            prompt,
                            family_name=self._model_family_name(),
                            model_name=self._model_name,
                            tools_present=bool(template_tools),
                        )
                else:
                    prompt = "\n".join(f"{m['role']}: {m['content']}" for m in messages)
                    prompt += "\nassistant:"

                # Append prompt suffix (e.g. Harmony analysis prefix for GPT-OSS)
                if prompt_suffix:
                    prompt += prompt_suffix

                self._raise_if_prompt_over_limit(
                    prompt,
                    max_prompt_tokens,
                    source="rendered chat prompt",
                )

                # Generate inline (don't call self.generate() — we already hold
                # _generation_lock and asyncio.Lock is not reentrant).
                output = await self._run_model_call(
                    self._model.generate,
                    prompt=prompt,
                    max_tokens=max_tokens,
                    temperature=temperature,
                    top_p=top_p,
                    **kwargs,
                )
                raw_text = output.text
                text = clean_output_text(raw_text)
                _prompt_tokens = getattr(output, "prompt_tokens", 0)
                # Concern #11 (audit 2026-04-08): non-streaming path was missing
                # the prompt_tokens fallback that the streaming path has at
                # line ~282. If the model wrapper reports 0 (e.g. because the
                # internal token counter wasn't wired for this model family),
                # re-tokenize the prompt to recover a real count. This also
                # closes the surface that surfaced ISSUE-A4-004 (nemotron_h
                # HTTP serve reported prompt_tokens=0 with all-<unk> output).
                if _prompt_tokens == 0:
                    try:
                        _prompt_tokens = len(tokenizer.encode(prompt))
                    except Exception:
                        _prompt_tokens = 0
                return GenerationOutput(
                    text=text,
                    raw_text=raw_text,
                    tokens=getattr(output, "tokens", []),
                    logprobs=getattr(output, "logprobs", None),
                    prompt_tokens=_prompt_tokens,
                    completion_tokens=getattr(
                        output, "completion_tokens", len(getattr(output, "tokens", []))
                    ),
                    finish_reason=output.finish_reason,
                )

    async def stream_chat(
        self,
        messages: list[dict[str, Any]],
        max_tokens: int = 256,
        temperature: float = 0.7,
        top_p: float = 0.9,
        tools: list[dict] | None = None,
        images: list[str] | None = None,
        videos: list[str] | None = None,
        request_id: str | None = None,
        **kwargs,
    ) -> AsyncIterator[GenerationOutput]:
        """
        Stream chat completion token by token.

        Note: SimpleEngine doesn't use request_id (synchronous processing),
        but parameter is here for API compatibility with BaseEngine.

        Args:
            messages: List of chat messages
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            top_p: Top-p sampling
            tools: Optional tool definitions
            images: Optional image URLs/paths
            videos: Optional video URLs/paths
            request_id: Optional custom request ID (not used in SimpleEngine)
            **kwargs: Additional model-specific parameters

        Yields:
            GenerationOutput with incremental text
        """
        if not self._loaded:
            await self.start()
        # SimpleEngine has no prefix cache — eat the bypass kwarg so it
        # doesn't leak into self._model.generate which would reject it.
        kwargs.pop("_bypass_prefix_cache", None)

        # Convert tools for template
        template_tools = convert_tools_for_template(tools) if tools else None

        # Pop vmlx-engine-specific kwargs early to prevent leaking to mlx-lm calls
        reasoning_effort = kwargs.pop("reasoning_effort", None)
        thinking_enabled = kwargs.pop("enable_thinking", True)
        extra_ct_kwargs = kwargs.pop("chat_template_kwargs", None)
        prompt_suffix = kwargs.pop("prompt_suffix", None)
        skip_gen_prompt = kwargs.pop("skip_generation_prompt", False)
        max_prompt_tokens = int(kwargs.pop("max_prompt_tokens", 0) or 0)

        # Build prompt using tokenizer
        if self._is_mllm:
            # For MLLM, stream tokens one at a time on the dedicated model
            # worker. This mirrors the LLM stream_generate pattern for true
            # per-token streaming while preserving MLX stream thread affinity.
            accumulated_text = ""
            token_count = 0
            last_prompt_tokens = 0
            finished = False
            self._abort_requested = False
            self._current_request_id = request_id

            # Pass enable_thinking to MLLM for models that support it (Qwen3-VL, etc.)
            mllm_kwargs = dict(kwargs)
            if thinking_enabled is not None:
                mllm_kwargs["enable_thinking"] = thinking_enabled
            if reasoning_effort:
                mllm_kwargs["reasoning_effort"] = reasoning_effort
            if extra_ct_kwargs:
                # Strip reserved keys (Concern #14): see above.
                mllm_kwargs.update({
                    k: v for k, v in extra_ct_kwargs.items()
                    if k not in ("tokenize", "add_generation_prompt")
                })
            if template_tools:
                mllm_kwargs["tools"] = template_tools

            async with self._generation_lock:
                try:
                    # Create and consume the synchronous iterator on the
                    # same MLX worker thread that loaded the model.
                    stream_iter = await self._run_model_call(
                        lambda: iter(self._model.stream_chat(
                            messages=messages,
                            max_tokens=max_tokens,
                            temperature=temperature,
                            top_p=top_p,
                            **mllm_kwargs,
                        ))
                    )
                except Exception as e:
                    logger.error(f"MLLM stream_chat failed to start: {type(e).__name__}: {e}")
                    clear_mlx_memory_cache(log=logger)
                    raise

                # Per-token iteration: offload each next() to thread pool
                _sentinel = object()
                def _next():
                    try:
                        return next(stream_iter)
                    except StopIteration:
                        return _sentinel

                while True:
                    try:
                        chunk = await self._run_model_call(_next)
                    except Exception as e:
                        logger.error(f"MLLM generation error: {type(e).__name__}: {e}")
                        clear_mlx_memory_cache(log=logger)
                        raise

                    if chunk is _sentinel:
                        break

                    # Check abort flag between tokens
                    if self._abort_requested:
                        self._abort_requested = False
                        logger.info("SimpleEngine: MLLM generation aborted by request")
                        yield GenerationOutput(
                            text=accumulated_text,
                            new_text="",
                            prompt_tokens=0,
                            completion_tokens=token_count,
                            finished=True,
                            finish_reason="abort",
                        )
                        return

                    token_count += 1
                    new_text = chunk.text if hasattr(chunk, "text") else str(chunk)
                    accumulated_text += new_text

                    chunk_prompt_tokens = getattr(chunk, "prompt_tokens", 0)
                    if chunk_prompt_tokens:
                        last_prompt_tokens = chunk_prompt_tokens

                    finished = chunk.finish_reason is not None
                    if finished and last_prompt_tokens == 0:
                        last_prompt_tokens = self._estimate_mllm_chat_prompt_tokens(
                            messages,
                            thinking_enabled,
                        )

                    yield GenerationOutput(
                        text=accumulated_text,
                        new_text=new_text,
                        prompt_tokens=last_prompt_tokens,
                        completion_tokens=token_count,
                        finished=finished,
                        finish_reason=chunk.finish_reason if finished else None,
                    )

                    if finished:
                        break

                # If stream ended without explicit finish, emit final chunk
                if not finished:
                    if last_prompt_tokens == 0:
                        last_prompt_tokens = self._estimate_mllm_chat_prompt_tokens(
                            messages,
                            thinking_enabled,
                        )
                    yield GenerationOutput(
                        text=accumulated_text,
                        new_text="",
                        prompt_tokens=last_prompt_tokens,
                        completion_tokens=token_count,
                        finished=True,
                        finish_reason="stop",
                    )
            self._current_request_id = None
            return

        # For LLM, apply chat template and stream
        tokenizer = self._model.tokenizer
        if hasattr(tokenizer, "apply_chat_template"):
            # Ensure tool_calls arguments are dicts, not JSON strings.
            # Chat templates (Qwen3, Llama, etc.) call .items() on arguments.
            for msg in messages:
                for tc in msg.get("tool_calls") or []:
                    fn = tc.get("function", {})
                    args = fn.get("arguments")
                    if isinstance(args, str):
                        try:
                            import json
                            fn["arguments"] = json.loads(args)
                        except (json.JSONDecodeError, TypeError):
                            fn["arguments"] = {}

            # Use enable_thinking and reasoning_effort popped at function entry
            # skip_gen_prompt: when prompt_suffix provides the full assistant prefix
            # (e.g. Harmony analysis channel), don't let the template add its own
            template_kwargs = build_chat_template_kwargs(
                enable_thinking=self._chat_template_enable_thinking_for_render(
                    thinking_enabled
                ),
                extra=extra_ct_kwargs,
                tokenize=False,
                add_generation_prompt=not skip_gen_prompt,
            )
            if template_tools:
                template_kwargs["tools"] = template_tools
            if reasoning_effort:
                template_kwargs["reasoning_effort"] = reasoning_effort

            try:
                prompt = tokenizer.apply_chat_template(messages, **template_kwargs)
            except Exception as template_err:
                # Progressively strip non-essential kwargs to preserve tools/thinking.
                # Strip order: extra kwargs first, then tools, then enable_thinking.
                logger.warning(f"Chat template first attempt failed (will retry with fewer kwargs): {template_err}")
                strip_order = [
                    k for k in template_kwargs
                    if k not in ("tokenize", "add_generation_prompt")
                ]
                prompt = None
                for key in reversed(strip_order):
                    del template_kwargs[key]
                    try:
                        prompt = tokenizer.apply_chat_template(messages, **template_kwargs)
                        stripped = [k for k in strip_order if k not in template_kwargs]
                        logger.warning(
                            f"Chat template succeeded after stripping: {stripped} "
                            f"(original error: {template_err})"
                        )
                        break
                    except Exception:
                        continue

                if prompt is None:
                    try:
                        prompt = tokenizer.apply_chat_template(messages, **template_kwargs)
                    except Exception as retry_err:
                        logger.error(f"Chat template failed even after stripping kwargs: {retry_err}")
                        # Last resort: manual prompt formatting
                        prompt = "\n".join(f"{m['role']}: {m['content']}" for m in messages)
                        prompt += "\nassistant:"

            prompt = check_and_inject_fallback_tools(
                prompt,
                messages,
                template_tools,
                tokenizer,
                template_kwargs,
                tool_parser_id=self._model_tool_parser_name(),
            )

            if thinking_enabled is False:
                prompt = ensure_thinking_off_sentinel(
                    prompt,
                    family_name=self._model_family_name(),
                    model_name=self._model_name,
                    tools_present=bool(template_tools),
                )
        else:
            prompt = "\n".join(f"{m['role']}: {m['content']}" for m in messages)
            prompt += "\nassistant:"

        # Append prompt suffix (e.g. Harmony analysis prefix for GPT-OSS)
        if prompt_suffix:
            prompt += prompt_suffix

        self._raise_if_prompt_over_limit(
            prompt,
            max_prompt_tokens,
            source="rendered chat prompt",
        )

        # Stream generate
        async for output in self.stream_generate(
            prompt=prompt,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            request_id=request_id,
            max_prompt_tokens=max_prompt_tokens,
            **kwargs,
        ):
            yield output

    async def abort_request(self, request_id: str) -> bool:
        """Abort the current generation if request_id matches (or unconditionally if no tracking)."""
        if self._current_request_id and request_id != self._current_request_id:
            logger.info(f"SimpleEngine: abort_request({request_id}) ignored — current request is {self._current_request_id}")
            return False
        self._abort_requested = True
        logger.info(f"SimpleEngine: abort_request({request_id}) — flagging for abort")
        return True

    def get_stats(self) -> dict[str, Any]:
        """Get engine statistics."""
        return {
            "engine_type": "simple",
            "model_name": self._model_name,
            "is_mllm": self._is_mllm,
            "loaded": self._loaded,
        }

    def get_cache_stats(self) -> dict[str, Any] | None:
        """Get cache statistics (for MLLM models)."""
        if self._is_mllm and self._model is not None:
            return self._model.get_cache_stats()
        return None
