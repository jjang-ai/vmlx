# SPDX-License-Identifier: Apache-2.0
"""
MLX Language Model wrapper.

This module provides a wrapper around mlx-lm for LLM inference,
integrating with vLLM's model execution system.
"""

import logging
from dataclasses import dataclass
from typing import Iterator

logger = logging.getLogger(__name__)


@dataclass
class GenerationOutput:
    """Output from text generation."""

    text: str
    tokens: list[int]
    finish_reason: str | None = None


@dataclass
class StreamingOutput:
    """Streaming output chunk."""

    text: str
    token: int
    finished: bool = False
    finish_reason: str | None = None


class MLXLanguageModel:
    """
    Wrapper around mlx-lm for LLM inference.

    This class provides a unified interface for loading and running
    inference on language models using Apple's MLX framework.

    Example:
        >>> model = MLXLanguageModel("mlx-community/Llama-3.2-3B-Instruct-4bit")
        >>> output = model.generate("Hello, how are you?", max_tokens=100)
        >>> print(output.text)
    """

    def __init__(
        self,
        model_name: str,
        tokenizer_name: str | None = None,
        trust_remote_code: bool = False,
    ):
        """
        Initialize the MLX language model.

        Args:
            model_name: HuggingFace model name or local path
            tokenizer_name: Optional separate tokenizer name
            trust_remote_code: Whether to trust remote code
        """
        self.model_name = model_name
        self.tokenizer_name = tokenizer_name or model_name
        self.trust_remote_code = trust_remote_code

        self.model = None
        self.tokenizer = None
        self._loaded = False

        # SSD disk-streaming state (set by server.py after model load)
        self._stream_from_disk = False
        self._model_path = None
        self._weight_index = None
        self._temp_weight_dir = None

    def load(self) -> None:
        """Load the model and tokenizer."""
        if self._loaded:
            return

        try:
            from ..utils.tokenizer import load_model_with_fallback

            logger.info(f"Loading model: {self.model_name}")

            # Build tokenizer config
            tokenizer_config = {"trust_remote_code": self.trust_remote_code}

            # Use model config registry for EOS token overrides
            from ..model_config_registry import get_model_config_registry

            registry = get_model_config_registry()
            model_config = registry.lookup(self.model_name)
            if model_config.eos_tokens:
                tokenizer_config["eos_token"] = model_config.eos_tokens[0]
                logger.info(
                    f"{model_config.family_name} detected: "
                    f"setting eos_token to {model_config.eos_tokens[0]}"
                )

            self.model, self.tokenizer = load_model_with_fallback(
                self.model_name,
                tokenizer_config=tokenizer_config,
            )

            self._loaded = True
            logger.info(f"Model loaded successfully: {self.model_name}")

        except ImportError:
            raise ImportError(
                "mlx-lm is required for LLM inference. "
                "Install with: pip install mlx-lm"
            )
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            raise

    def _create_sampler(
        self,
        temperature: float = 0.7,
        top_p: float = 0.9,
        min_p: float = 0.0,
        top_k: int = 0,
    ):
        """Create a sampler for text generation."""
        from mlx_lm.sample_utils import make_sampler

        return make_sampler(
            temp=temperature,
            top_p=top_p,
            min_p=min_p,
            top_k=top_k,
        )

    def generate(
        self,
        prompt: str,
        max_tokens: int = 256,
        temperature: float = 0.7,
        top_p: float = 0.9,
        repetition_penalty: float = 1.0,
        stop: list[str] | None = None,
        top_k: int = 0,
        min_p: float = 0.0,
        **kwargs,
    ) -> GenerationOutput:
        """
        Generate text from a prompt.

        Args:
            prompt: Input prompt text
            max_tokens: Maximum number of tokens to generate
            temperature: Sampling temperature (0 = greedy)
            top_p: Top-p (nucleus) sampling parameter
            repetition_penalty: Penalty for repeating tokens
            stop: List of stop sequences
            top_k: Top-k sampling (0 = disabled)
            min_p: Min-p sampling threshold

        Returns:
            GenerationOutput with generated text and tokens
        """
        if not self._loaded:
            self.load()

        # SSD disk-streaming: use custom generate loop
        if self._stream_from_disk and self._model_path:
            from ..utils.ssd_generate import ssd_generate
            output_text = ssd_generate(
                self.model, self.tokenizer, prompt, self._model_path,
                weight_index=self._weight_index,
                temp_weight_dir=self._temp_weight_dir,
                max_tokens=max_tokens,
                temperature=temperature,
                top_p=top_p,
                repetition_penalty=repetition_penalty,
            )
            tokens = self.tokenizer.encode(output_text)
            finish_reason = "length" if len(tokens) >= max_tokens else "stop"
            return GenerationOutput(text=output_text, tokens=tokens, finish_reason=finish_reason)

        from mlx_lm import generate

        # Create sampler with all sampling parameters
        sampler = self._create_sampler(temperature, top_p, min_p=min_p, top_k=top_k)

        # Build logits processors for repetition penalty
        logits_processors = None
        if repetition_penalty and repetition_penalty != 1.0:
            from mlx_lm.sample_utils import make_logits_processors
            logits_processors = make_logits_processors(
                repetition_penalty=repetition_penalty,
            )

        # Check for speculative decoding
        draft_model_arg = None
        num_draft = 0
        try:
            from ..speculative import get_draft_model, get_num_draft_tokens, is_speculative_enabled, validate_draft_tokenizer
            if is_speculative_enabled():
                draft_model_arg = get_draft_model()
                num_draft = get_num_draft_tokens()
                if draft_model_arg is not None:
                    validate_draft_tokenizer(self.tokenizer)
        except ImportError:
            pass

        if draft_model_arg is not None:
            # mlx_lm.generate() doesn't accept draft_model, so we use
            # stream_generate with speculative decoding and collect all output
            from mlx_lm import stream_generate as sg

            output_text = ""
            for resp in sg(
                self.model,
                self.tokenizer,
                prompt=prompt,
                max_tokens=max_tokens,
                sampler=sampler,
                logits_processors=logits_processors,
                draft_model=draft_model_arg,
                num_draft_tokens=num_draft,
            ):
                # Each resp.text is a new segment from the streaming detokenizer
                output_text += resp.text
        else:
            # Standard non-speculative generation
            output_text = generate(
                self.model,
                self.tokenizer,
                prompt=prompt,
                max_tokens=max_tokens,
                sampler=sampler,
                logits_processors=logits_processors,
                verbose=False,
            )

        # Truncate at first stop sequence (mlx_lm.generate doesn't support stop natively)
        # Note: output_text is generated tokens only (no prompt echo)
        finish_reason = "length"
        if stop and output_text:
            for stop_seq in stop:
                idx = output_text.find(stop_seq)
                if idx != -1:
                    output_text = output_text[:idx]
                    finish_reason = "stop"
                    break

        # Tokenize after truncation to get accurate token count
        tokens = self.tokenizer.encode(output_text)
        if finish_reason != "stop":
            finish_reason = "length" if len(tokens) >= max_tokens else "stop"

        return GenerationOutput(
            text=output_text,
            tokens=tokens,
            finish_reason=finish_reason,
        )

    def stream_generate(
        self,
        prompt: str,
        max_tokens: int = 256,
        temperature: float = 0.7,
        top_p: float = 0.9,
        repetition_penalty: float = 1.0,
        stop: list[str] | None = None,
        top_k: int = 0,
        min_p: float = 0.0,
        **kwargs,
    ) -> Iterator[StreamingOutput]:
        """
        Stream text generation token by token.

        Args:
            prompt: Input prompt text
            max_tokens: Maximum number of tokens to generate
            temperature: Sampling temperature (0 = greedy)
            top_p: Top-p (nucleus) sampling parameter
            repetition_penalty: Penalty for repeating tokens
            stop: List of stop sequences
            top_k: Top-k sampling (0 = disabled)
            min_p: Min-p sampling threshold

        Yields:
            StreamingOutput for each generated token
        """
        if not self._loaded:
            self.load()

        # SSD disk-streaming: use custom generate loop
        if self._stream_from_disk and self._model_path:
            from ..utils.ssd_generate import ssd_stream_generate

            token_count = 0
            accumulated_text = ""
            for response in ssd_stream_generate(
                self.model, self.tokenizer, prompt, self._model_path,
                weight_index=self._weight_index,
                temp_weight_dir=self._temp_weight_dir,
                max_tokens=max_tokens,
                temperature=temperature,
                top_p=top_p,
                repetition_penalty=repetition_penalty,
            ):
                token_count += 1
                new_text = response.text
                accumulated_text += new_text
                should_stop = False
                if stop:
                    for stop_seq in stop:
                        if stop_seq in accumulated_text:
                            should_stop = True
                            break
                finished = should_stop or token_count >= max_tokens or response.finish_reason is not None
                finish_reason = None
                if finished:
                    finish_reason = "stop" if should_stop else (response.finish_reason or "length")
                yield StreamingOutput(
                    text=new_text,
                    token=response.token if hasattr(response, "token") else 0,
                    finished=finished,
                    finish_reason=finish_reason,
                )
                if finished:
                    break
            return

        from mlx_lm import stream_generate

        # Create sampler with all sampling parameters
        sampler = self._create_sampler(temperature, top_p, min_p=min_p, top_k=top_k)

        # Build logits processors for repetition penalty
        logits_processors = None
        if repetition_penalty and repetition_penalty != 1.0:
            from mlx_lm.sample_utils import make_logits_processors
            logits_processors = make_logits_processors(
                repetition_penalty=repetition_penalty,
            )

        token_count = 0
        accumulated_text = ""

        # Check for speculative decoding
        spec_kwargs = {}
        try:
            from ..speculative import get_draft_model, get_num_draft_tokens, is_speculative_enabled, validate_draft_tokenizer
            if is_speculative_enabled():
                draft_model = get_draft_model()
                if draft_model is not None:
                    spec_kwargs["draft_model"] = draft_model
                    spec_kwargs["num_draft_tokens"] = get_num_draft_tokens()
                    # Validate tokenizer compatibility on first use
                    validate_draft_tokenizer(self.tokenizer)
                    logger.info(
                        f"Speculative decoding active: draft_tokens={get_num_draft_tokens()}"
                    )
        except ImportError:
            pass

        for response in stream_generate(
            self.model,
            self.tokenizer,
            prompt=prompt,
            max_tokens=max_tokens,
            sampler=sampler,
            logits_processors=logits_processors,
            **spec_kwargs,
        ):
            token_count += 1
            # response.text is the new token text (not accumulated)
            new_text = response.text
            accumulated_text += new_text

            # Check for stop sequences
            should_stop = False
            if stop:
                for stop_seq in stop:
                    if stop_seq in accumulated_text:
                        should_stop = True
                        break

            finished = should_stop or token_count >= max_tokens
            finish_reason = None
            if finished:
                finish_reason = "stop" if should_stop else "length"

            yield StreamingOutput(
                text=new_text,
                token=response.token if hasattr(response, "token") else 0,
                finished=finished,
                finish_reason=finish_reason,
            )

            if finished:
                break

    def get_model_info(self) -> dict:
        """Get information about the loaded model."""
        if not self._loaded:
            return {"loaded": False, "model_name": self.model_name}

        info = {
            "loaded": True,
            "model_name": self.model_name,
            "tokenizer_name": self.tokenizer_name,
        }

        # Try to get model config
        if hasattr(self.model, "config"):
            config = self.model.config
            info.update(
                {
                    "vocab_size": getattr(config, "vocab_size", None),
                    "hidden_size": getattr(config, "hidden_size", None),
                    "num_layers": getattr(config, "num_hidden_layers", None),
                    "num_heads": getattr(config, "num_attention_heads", None),
                }
            )

        return info

    def __repr__(self) -> str:
        status = "loaded" if self._loaded else "not loaded"
        return f"<MLXLanguageModel model={self.model_name} status={status}>"
