# SPDX-License-Identifier: Apache-2.0
"""
Simple engine for maximum single-user throughput.

This engine wraps mlx-lm directly with zero overhead for optimal
performance when serving a single user at a time.
"""

import asyncio
import logging
from collections.abc import AsyncIterator
from typing import Any

from ..api.tool_calling import convert_tools_for_template
from ..api.utils import clean_output_text, is_mllm_model
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

    @property
    def model_name(self) -> str:
        """Get the model name."""
        return self._model_name

    @property
    def is_mllm(self) -> bool:
        """Check if this is a multimodal model."""
        return self._is_mllm

    @property
    def tokenizer(self) -> Any:
        """Get the tokenizer."""
        if not self._loaded or self._model is None:
            return None
        if self._is_mllm:
            return getattr(self._model, "processor", None)
        return self._model.tokenizer

    async def start(self) -> None:
        """Start the engine (load model if not loaded)."""
        if self._loaded:
            return

        if self._is_mllm:
            from ..models.mllm import MLXMultimodalLM

            self._model = MLXMultimodalLM(
                self._model_name,
                trust_remote_code=self._trust_remote_code,
                enable_cache=self._enable_cache,
            )
        else:
            from ..models.llm import MLXLanguageModel

            self._model = MLXLanguageModel(
                self._model_name,
                trust_remote_code=self._trust_remote_code,
            )

        self._model.load()
        self._loaded = True
        logger.info(f"SimpleEngine loaded: {self._model_name} (MLLM={self._is_mllm})")

    async def stop(self) -> None:
        """Stop the engine and cleanup resources."""
        self._model = None
        self._loaded = False
        logger.info("SimpleEngine stopped")

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

        async with self._generation_lock:
            # Run in thread pool to allow asyncio timeout to work
            output = await asyncio.to_thread(
                self._model.generate,
                prompt=prompt,
                max_tokens=max_tokens,
                temperature=temperature,
                top_p=top_p,
                stop=stop,
                **kwargs,
            )

            # Clean output text
            text = clean_output_text(output.text)

            return GenerationOutput(
                text=text,
                tokens=getattr(output, "tokens", []),
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

        async with self._generation_lock:
            accumulated_text = ""
            prompt_tokens = 0
            completion_tokens = 0
            finished = False
            self._abort_requested = False  # Reset at start of generation

            try:
                stream_iter = self._model.stream_generate(
                    prompt=prompt,
                    max_tokens=max_tokens,
                    temperature=temperature,
                    top_p=top_p,
                    stop=stop,
                    **kwargs,
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
                chunk = await asyncio.to_thread(_next)
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

        # Convert tools for template if provided
        template_tools = convert_tools_for_template(tools) if tools else None

        # Pop vllm-mlx-specific kwargs early to prevent leaking to mlx-lm calls
        extra_ct_kwargs = kwargs.pop("chat_template_kwargs", None)
        reasoning_effort = kwargs.pop("reasoning_effort", None)
        kwargs.pop("request_id", None)
        thinking_enabled = kwargs.pop("enable_thinking", True)
        prompt_suffix = kwargs.pop("prompt_suffix", None)
        skip_gen_prompt = kwargs.pop("skip_generation_prompt", False)

        async with self._generation_lock:
            if self._is_mllm:
                # For MLLM, use the chat method which handles images/videos
                # Run in thread pool to allow asyncio timeout to work
                mllm_kwargs = dict(kwargs)
                if thinking_enabled is not None:
                    mllm_kwargs["enable_thinking"] = thinking_enabled
                if reasoning_effort:
                    mllm_kwargs["reasoning_effort"] = reasoning_effort
                if extra_ct_kwargs:
                    mllm_kwargs.update(extra_ct_kwargs)
                output = await asyncio.to_thread(
                    self._model.chat,
                    messages=messages,
                    max_tokens=max_tokens,
                    temperature=temperature,
                    top_p=top_p,
                    **mllm_kwargs,
                )
                text = clean_output_text(output.text)
                return GenerationOutput(
                    text=text,
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

                    tpl_kwargs = {
                        "tokenize": False,
                        "add_generation_prompt": not skip_gen_prompt,
                    }
                    if thinking_enabled is not None:
                        tpl_kwargs["enable_thinking"] = thinking_enabled
                    if template_tools:
                        tpl_kwargs["tools"] = template_tools
                    if reasoning_effort:
                        tpl_kwargs["reasoning_effort"] = reasoning_effort
                    if extra_ct_kwargs:
                        tpl_kwargs.update(extra_ct_kwargs)

                    try:
                        prompt = tokenizer.apply_chat_template(messages, **tpl_kwargs)
                    except (TypeError, Exception) as template_err:
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
                            except (TypeError, Exception):
                                continue
                        if prompt is None:
                            prompt = tokenizer.apply_chat_template(messages, **tpl_kwargs)
                            
                    from ..api.tool_calling import check_and_inject_fallback_tools
                    prompt = check_and_inject_fallback_tools(
                        prompt, messages, template_tools, tokenizer, tpl_kwargs
                    )
                else:
                    prompt = "\n".join(f"{m['role']}: {m['content']}" for m in messages)
                    prompt += "\nassistant:"

                # Append prompt suffix (e.g. Harmony analysis prefix for GPT-OSS)
                if prompt_suffix:
                    prompt += prompt_suffix

                # Use generate() with the manually-built prompt
                return await self.generate(
                    prompt=prompt,
                    max_tokens=max_tokens,
                    temperature=temperature,
                    top_p=top_p,
                    **kwargs,
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

        # Convert tools for template
        template_tools = convert_tools_for_template(tools) if tools else None

        # Pop vllm-mlx-specific kwargs early to prevent leaking to mlx-lm calls
        reasoning_effort = kwargs.pop("reasoning_effort", None)
        thinking_enabled = kwargs.pop("enable_thinking", True)
        extra_ct_kwargs = kwargs.pop("chat_template_kwargs", None)
        prompt_suffix = kwargs.pop("prompt_suffix", None)
        skip_gen_prompt = kwargs.pop("skip_generation_prompt", False)

        # Build prompt using tokenizer
        if self._is_mllm:
            # For MLLM, stream tokens one at a time via asyncio.to_thread(_next).
            # This mirrors the LLM stream_generate pattern for true per-token
            # streaming — each next() call is offloaded individually so the
            # event loop stays responsive between tokens.
            accumulated_text = ""
            token_count = 0
            finished = False
            self._abort_requested = False

            # Pass enable_thinking to MLLM for models that support it (Qwen3-VL, etc.)
            mllm_kwargs = dict(kwargs)
            if thinking_enabled is not None:
                mllm_kwargs["enable_thinking"] = thinking_enabled
            if reasoning_effort:
                mllm_kwargs["reasoning_effort"] = reasoning_effort
            if extra_ct_kwargs:
                mllm_kwargs.update(extra_ct_kwargs)

            async with self._generation_lock:
                try:
                    # Create the synchronous iterator in a thread
                    stream_iter = await asyncio.to_thread(
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
                    try:
                        import mlx.core as mx
                        mx.clear_memory_cache()
                    except Exception:
                        pass
                    yield GenerationOutput(
                        text=f"[Generation error: {type(e).__name__}: {e}]",
                        new_text=f"[Generation error: {type(e).__name__}: {e}]",
                        prompt_tokens=0,
                        completion_tokens=0,
                        finished=True,
                        finish_reason="error",
                    )
                    return

                # Per-token iteration: offload each next() to thread pool
                _sentinel = object()
                def _next():
                    try:
                        return next(stream_iter)
                    except StopIteration:
                        return _sentinel

                while True:
                    try:
                        chunk = await asyncio.to_thread(_next)
                    except Exception as e:
                        logger.error(f"MLLM generation error: {type(e).__name__}: {e}")
                        try:
                            import mlx.core as mx
                            mx.clear_memory_cache()
                        except Exception:
                            pass
                        yield GenerationOutput(
                            text=accumulated_text + f"\n[Generation error: {type(e).__name__}: {e}]",
                            new_text=f"\n[Generation error: {type(e).__name__}: {e}]",
                            prompt_tokens=0,
                            completion_tokens=token_count,
                            finished=True,
                            finish_reason="error",
                        )
                        return

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

                    finished = chunk.finish_reason is not None

                    yield GenerationOutput(
                        text=accumulated_text,
                        new_text=new_text,
                        prompt_tokens=getattr(chunk, "prompt_tokens", 0),
                        completion_tokens=token_count,
                        finished=finished,
                        finish_reason=chunk.finish_reason if finished else None,
                    )

                    if finished:
                        break

                # If stream ended without explicit finish, emit final chunk
                if not finished:
                    yield GenerationOutput(
                        text=accumulated_text,
                        new_text="",
                        prompt_tokens=0,
                        completion_tokens=token_count,
                        finished=True,
                        finish_reason="stop",
                    )
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
            template_kwargs = {
                "tokenize": False,
                "add_generation_prompt": not skip_gen_prompt,
            }
            if thinking_enabled is not None:
                template_kwargs["enable_thinking"] = thinking_enabled
            if template_tools:
                template_kwargs["tools"] = template_tools
            if reasoning_effort:
                template_kwargs["reasoning_effort"] = reasoning_effort
            # Merge extra chat_template_kwargs (e.g. thinking_budget)
            if extra_ct_kwargs:
                template_kwargs.update(extra_ct_kwargs)

            try:
                prompt = tokenizer.apply_chat_template(messages, **template_kwargs)
            except (TypeError, Exception) as template_err:
                # Progressively strip non-essential kwargs to preserve tools/thinking.
                # Strip order: extra kwargs first, then tools, then enable_thinking.
                logger.warning(f"Chat template first attempt failed (will retry with fewer kwargs): {template_err}")
                had_tools = "tools" in template_kwargs
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
                    except (TypeError, Exception):
                        continue

                if prompt is None:
                    try:
                        prompt = tokenizer.apply_chat_template(messages, **template_kwargs)
                    except Exception as retry_err:
                        logger.error(f"Chat template failed even after stripping kwargs: {retry_err}")
                        # Last resort: manual prompt formatting
                        prompt = "\n".join(f"{m['role']}: {m['content']}" for m in messages)
                        prompt += "\nassistant:"
                        
            from ..api.tool_calling import check_and_inject_fallback_tools
            prompt = check_and_inject_fallback_tools(
                prompt, messages, template_tools, tokenizer, template_kwargs
            )
        else:
            prompt = "\n".join(f"{m['role']}: {m['content']}" for m in messages)
            prompt += "\nassistant:"

        # Append prompt suffix (e.g. Harmony analysis prefix for GPT-OSS)
        if prompt_suffix:
            prompt += prompt_suffix

        # Stream generate
        async for output in self.stream_generate(
            prompt=prompt,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            **kwargs,
        ):
            yield output

    async def abort_request(self, request_id: str) -> bool:
        """Abort the current generation. Sets a flag checked between tokens."""
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
