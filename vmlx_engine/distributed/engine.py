# SPDX-License-Identifier: Apache-2.0
"""Distributed inference engine.

Implements BaseEngine interface for distributed pipeline-parallel inference.
Routes all generate/chat calls through the MeshManager's coordinator,
which orchestrates forward passes across worker nodes.

When distributed mode is active and peers are found, this engine replaces
SimpleEngine/BatchedEngine. If no peers are found (single-node), falls
back to standard local engine loading.
"""

from __future__ import annotations

import asyncio
import logging
import time
import uuid
from collections.abc import AsyncIterator
from typing import Any, Optional

import mlx.core as mx
import mlx.nn as nn

from ..engine.base import BaseEngine, GenerationOutput

logger = logging.getLogger(__name__)


class DistributedEngine(BaseEngine):
    """Engine that performs inference across multiple nodes via pipeline parallelism.

    Wraps a MeshManager's coordinator to provide the standard BaseEngine
    interface (generate, stream_generate, chat, stream_chat).

    The coordinator handles:
    - Embedding (embed_tokens)
    - Pipeline forward through local + remote layers
    - lm_head projection to logits
    This engine handles:
    - Tokenization
    - Autoregressive generate loop (sampling, KV cache, stop sequences)
    - Streaming
    """

    def __init__(self, mesh_manager):
        """
        Args:
            mesh_manager: A started MeshManager instance with coordinator role.
        """
        self._mesh = mesh_manager
        self._model_name_str = mesh_manager.model_path
        self._loaded = False
        self._generation_lock = asyncio.Lock()
        self._abort_requested = False
        self._current_request_id: str | None = None

    @property
    def model_name(self) -> str:
        return self._model_name_str

    @property
    def is_mllm(self) -> bool:
        return False  # Distributed mode is text-only for now

    @property
    def tokenizer(self) -> Any:
        return self._mesh.get_tokenizer()

    async def start(self) -> None:
        if self._loaded:
            return
        # MeshManager.start() was already called — just verify
        if not self._mesh.is_ready:
            logger.info("Waiting for distributed mesh to become ready...")
            for _ in range(60):  # Wait up to 60s
                await asyncio.sleep(1)
                if self._mesh.is_ready:
                    break
            if not self._mesh.is_ready:
                raise RuntimeError("Distributed mesh did not become ready in time")
        self._loaded = True
        logger.info("DistributedEngine ready (%d nodes)", len(self._mesh.topology.active_nodes))

    async def stop(self) -> None:
        if self._mesh:
            await self._mesh.shutdown()
        self._loaded = False
        logger.info("DistributedEngine stopped")

    async def generate(
        self,
        prompt: str,
        max_tokens: int = 256,
        temperature: float = 0.7,
        top_p: float = 0.9,
        stop: list[str] | None = None,
        **kwargs,
    ) -> GenerationOutput:
        if not self._loaded:
            await self.start()

        result_text = ""
        output = None
        async for chunk in self._generate_impl(
            prompt, max_tokens, temperature, top_p, stop, **kwargs
        ):
            result_text += chunk.new_text
            output = chunk

        if output is None:
            return GenerationOutput(text="", finished=True)

        output.text = result_text
        output.finished = True
        return output

    async def stream_generate(
        self,
        prompt: str,
        max_tokens: int = 256,
        temperature: float = 0.7,
        top_p: float = 0.9,
        stop: list[str] | None = None,
        **kwargs,
    ) -> AsyncIterator[GenerationOutput]:
        if not self._loaded:
            await self.start()

        async for chunk in self._generate_impl(
            prompt, max_tokens, temperature, top_p, stop, **kwargs
        ):
            yield chunk

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
        prompt = self._apply_chat_template(messages, tools)
        return await self.generate(prompt, max_tokens, temperature, top_p, **kwargs)

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
        prompt = self._apply_chat_template(messages, tools)
        self._current_request_id = request_id
        async for chunk in self.stream_generate(
            prompt, max_tokens, temperature, top_p, **kwargs
        ):
            yield chunk
        self._current_request_id = None

    async def abort_request(self, request_id: str) -> bool:
        if self._current_request_id == request_id:
            self._abort_requested = True
            return True
        return False

    def get_stats(self) -> dict[str, Any]:
        return {
            "engine_type": "distributed",
            "mode": self._mesh.mode,
            "num_nodes": len(self._mesh.topology.active_nodes),
            "is_coordinator": self._mesh._is_coordinator,
            "total_ram_gb": self._mesh.topology.total_ram_gb,
        }

    def get_cache_stats(self) -> dict[str, Any] | None:
        return None  # Distributed cache stats TBD

    # ------------------------------------------------------------------
    # Internal generate loop
    # ------------------------------------------------------------------

    async def _generate_impl(
        self,
        prompt: str,
        max_tokens: int,
        temperature: float,
        top_p: float,
        stop: list[str] | None = None,
        **kwargs,
    ) -> AsyncIterator[GenerationOutput]:
        """Core autoregressive generate loop using distributed forward."""
        async with self._generation_lock:
            self._abort_requested = False

            tokenizer = self.tokenizer
            if tokenizer is None:
                raise RuntimeError("Tokenizer not available")

            # Tokenize
            input_ids = mx.array(tokenizer.encode(prompt))[None]  # [1, seq_len]
            prompt_tokens = input_ids.shape[1]

            coordinator = self._mesh.coordinator
            if coordinator is None:
                raise RuntimeError("No coordinator available")

            # Request tracking
            request_id = kwargs.get("request_id", str(uuid.uuid4())[:8])

            generated_tokens = []
            generated_text = ""
            eos_token_id = _get_eos_token_id(tokenizer)
            stop_sequences = stop or []

            # Prefill: run full prompt through pipeline
            logits = await coordinator.forward(input_ids)
            mx.async_eval(logits)

            # Sample first token
            next_token = _sample(logits[:, -1, :], temperature, top_p)
            generated_tokens.append(next_token.item())

            token_text = tokenizer.decode([next_token.item()])
            generated_text += token_text

            yield GenerationOutput(
                text="",
                new_text=token_text,
                tokens=[next_token.item()],
                prompt_tokens=prompt_tokens,
                completion_tokens=1,
                finished=False,
            )

            # Decode loop
            for step in range(1, max_tokens):
                if self._abort_requested:
                    break

                # Check EOS
                if next_token.item() in eos_token_id:
                    break

                # Check stop sequences
                if _check_stop(generated_text, stop_sequences):
                    break

                # Forward single token through pipeline
                token_input = next_token.reshape(1, 1)
                logits = await coordinator.forward(token_input)
                mx.async_eval(logits)

                # Sample
                next_token = _sample(logits[:, -1, :], temperature, top_p)
                generated_tokens.append(next_token.item())

                token_text = tokenizer.decode([next_token.item()])
                generated_text += token_text

                yield GenerationOutput(
                    text="",
                    new_text=token_text,
                    tokens=generated_tokens.copy(),
                    prompt_tokens=prompt_tokens,
                    completion_tokens=step + 1,
                    finished=False,
                )

            # Final chunk
            finish_reason = "stop"
            if self._abort_requested:
                finish_reason = "abort"
            elif len(generated_tokens) >= max_tokens:
                finish_reason = "length"

            yield GenerationOutput(
                text=generated_text,
                new_text="",
                tokens=generated_tokens,
                prompt_tokens=prompt_tokens,
                completion_tokens=len(generated_tokens),
                finish_reason=finish_reason,
                finished=True,
            )

    def _apply_chat_template(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict] | None = None,
    ) -> str:
        """Apply chat template to messages."""
        tokenizer = self.tokenizer
        if tokenizer is None:
            raise RuntimeError("Tokenizer not available")

        try:
            kwargs = {}
            if tools:
                kwargs["tools"] = tools
            return tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
                **kwargs,
            )
        except Exception as e:
            logger.warning("Chat template failed: %s — falling back to simple format", e)
            parts = []
            for msg in messages:
                role = msg.get("role", "user")
                content = msg.get("content", "")
                parts.append(f"{role}: {content}")
            parts.append("assistant:")
            return "\n".join(parts)


# ---------------------------------------------------------------------------
# Sampling helpers
# ---------------------------------------------------------------------------

def _sample(logits: mx.array, temperature: float, top_p: float) -> mx.array:
    """Sample a token from logits with temperature and top-p."""
    if temperature <= 0:
        return mx.argmax(logits, axis=-1)

    # Temperature scaling
    logits = logits / temperature

    # Top-p (nucleus) sampling
    if top_p < 1.0:
        sorted_indices = mx.argsort(-logits, axis=-1)
        sorted_logits = mx.take_along_axis(logits, sorted_indices, axis=-1)
        probs = mx.softmax(sorted_logits, axis=-1)
        cumsum = mx.cumsum(probs, axis=-1)

        # Zero out tokens beyond top-p threshold
        mask = cumsum - probs > top_p
        sorted_logits = mx.where(mask, mx.array(float("-inf")), sorted_logits)

        # Unsort
        unsort_indices = mx.argsort(sorted_indices, axis=-1)
        logits = mx.take_along_axis(sorted_logits, unsort_indices, axis=-1)

    probs = mx.softmax(logits, axis=-1)
    return mx.random.categorical(mx.log(probs + 1e-10))


def _get_eos_token_id(tokenizer) -> set:
    """Get EOS token IDs from tokenizer."""
    eos_ids = set()
    if hasattr(tokenizer, "eos_token_id"):
        eid = tokenizer.eos_token_id
        if isinstance(eid, int):
            eos_ids.add(eid)
        elif isinstance(eid, (list, tuple)):
            eos_ids.update(eid)
    # Also check for additional stop tokens in generation_config
    if hasattr(tokenizer, "generation_config"):
        gc = tokenizer.generation_config
        if hasattr(gc, "eos_token_id"):
            eid = gc.eos_token_id
            if isinstance(eid, int):
                eos_ids.add(eid)
            elif isinstance(eid, (list, tuple)):
                eos_ids.update(eid)
    return eos_ids


def _check_stop(text: str, stop_sequences: list[str]) -> bool:
    """Check if any stop sequence appears in generated text."""
    for seq in stop_sequences:
        if seq and seq in text:
            return True
    return False
