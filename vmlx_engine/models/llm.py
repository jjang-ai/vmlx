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

        self._model_path = None

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

            # ─────────────────────────────────────────────────────────────
            # UNIVERSAL multi-eos stop set installation
            # ─────────────────────────────────────────────────────────────
            # Common bug class observed across laguna, dsv4, qwen3.6,
            # nemotron-h, kimi, minimax, mistral3, gemma4: bundles ship
            # `generation_config.json::eos_token_id` as a LIST (e.g.
            # `[2, 24]`, `[1, 128803]`, `[248046, 248044]`), but
            # `mlx_lm.generate.stream_generate` re-wraps the tokenizer
            # with `TokenizerWrapper(tok)` — no `eos_token_ids` arg —
            # falling back to the singleton `{tokenizer.eos_token_id}`.
            # The model emits a token from the list that ISN'T the
            # singleton primary, the engine doesn't recognise it as a
            # stop, and decoding rolls past the natural turn boundary.
            # User symptoms range from emitting `</assistant>` literal
            # text (laguna) to looping in `<think>` mode (qwen3.6
            # hybrid SSM, dsv4 reasoning) to fake user turns.
            #
            # The PRE-WRAP pattern below collects all stop ids from:
            #   1. `generation_config.json::eos_token_id` (authoritative —
            #      bundle's own declared stop list)
            #   2. `model_config.eos_tokens` from the registry (engine
            #      overrides, e.g. dsv4's `<｜User｜>` defensive stop)
            #   3. `tokenizer.eos_token_id` (the existing primary)
            # …then wraps with `TokenizerWrapper(tok, eos_token_ids=...)`.
            # mlx_lm's `isinstance(tokenizer, TokenizerWrapper)` check
            # in `stream_generate` short-circuits the re-wrap, so our
            # full id list survives.
            try:
                from mlx_lm.tokenizer_utils import TokenizerWrapper
                resolved: list[int] = []

                # Source 1: tokenizer's own primary eos
                primary = getattr(self.tokenizer, "eos_token_id", None)
                if isinstance(primary, int):
                    resolved.append(primary)

                # Source 2: bundle's generation_config.json eos_token_id
                # (list or int). This is the most authoritative — it's
                # what HF / vLLM / llama.cpp use as the model's declared
                # stop set.
                try:
                    from pathlib import Path as _Path
                    import json as _json
                    _gen_cfg_path = _Path(self.model_name) / "generation_config.json"
                    if _gen_cfg_path.is_file():
                        _gen_cfg = _json.loads(_gen_cfg_path.read_text())
                        _gen_eos = _gen_cfg.get("eos_token_id")
                        if isinstance(_gen_eos, list):
                            for t in _gen_eos:
                                if isinstance(t, int) and t not in resolved:
                                    resolved.append(t)
                        elif isinstance(_gen_eos, int):
                            if _gen_eos not in resolved:
                                resolved.append(_gen_eos)
                except Exception:
                    pass  # missing / unreadable generation_config — skip

                # Source 3: registry-level eos_tokens (engine overrides
                # like dsv4's `<｜User｜>` defensive stop). Resolve each
                # string against the tokenizer's vocab.
                unresolved: list[str] = []
                if model_config.eos_tokens:
                    for tok_str in model_config.eos_tokens:
                        try:
                            tid = int(tok_str)
                        except ValueError:
                            tid = self.tokenizer.convert_tokens_to_ids(tok_str) \
                                if hasattr(self.tokenizer, "convert_tokens_to_ids") \
                                else None
                        if tid is None or tid < 0:
                            unresolved.append(tok_str)
                            continue
                        # `convert_tokens_to_ids` returns `unk_token_id`
                        # for strings not in vocab; treat that as
                        # unresolved unless the user really IS asking for
                        # the unk token.
                        unk_id = getattr(self.tokenizer, "unk_token_id", None)
                        if (
                            unk_id is not None and tid == unk_id
                            and tok_str != getattr(self.tokenizer, "unk_token", "")
                        ):
                            unresolved.append(tok_str)
                            continue
                        if tid not in resolved:
                            resolved.append(tid)

                # Only wrap when there's > 1 stop id — singleton equals
                # the default, no need to interfere.
                if len(resolved) > 1:
                    if isinstance(self.tokenizer, TokenizerWrapper):
                        # Already wrapped (Laguna early route or some
                        # custom path). Add any missing ids directly to
                        # the wrapper's stop set via add_eos_token.
                        for tid in resolved:
                            try:
                                # add_eos_token accepts int or string; pass
                                # int directly to avoid round-tripping
                                # through vocab.
                                self.tokenizer.add_eos_token(tid)
                            except Exception:
                                pass
                        # Telemetry: inspect the post-add set
                        existing = getattr(self.tokenizer, "_eos_token_ids", None)
                        logger.info(
                            f"{model_config.family_name}: TokenizerWrapper "
                            f"existing wrapper extended; eos_token_ids={existing}"
                        )
                    else:
                        # Raw HF tokenizer — pre-wrap with full id list
                        # so it survives mlx_lm's re-wrap.
                        self.tokenizer = TokenizerWrapper(
                            self.tokenizer, eos_token_ids=resolved,
                        )
                        logger.info(
                            f"{model_config.family_name}: tokenizer "
                            f"pre-wrapped with eos_token_ids={resolved}"
                        )

                if unresolved:
                    logger.warning(
                        f"{model_config.family_name}: unresolved eos "
                        f"strings {unresolved} (not in vocab as single "
                        f"tokens) — fix the registry entry or bundle "
                        f"tokenizer_config special_tokens"
                    )
            except Exception as _wrap_err:
                logger.warning(
                    f"{model_config.family_name}: universal multi-eos "
                    f"install failed: {_wrap_err}"
                )

            self._loaded = True
            logger.info(f"Model loaded successfully: {self.model_name}")

        except ImportError as _ie:
            # ISSUE-A4-003: previously this raised a generic message that masked
            # the real ImportError (e.g. missing jang_tools, missing turboquant,
            # etc.). Preserve the original cause via `from _ie` so debugging is
            # not blocked by the friendly message.
            raise ImportError(
                "mlx-lm (or one of its plugins) failed to import for LLM "
                f"inference. Original error: {_ie}. Install/fix with: "
                "pip install mlx-lm"
            ) from _ie
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
