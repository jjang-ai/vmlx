# SPDX-License-Identifier: Apache-2.0
"""Shared runtime errors surfaced by API routes."""


class PromptTooLongError(ValueError):
    """Raised when an exact tokenized prompt exceeds the configured context cap."""

    def __init__(
        self,
        prompt_tokens: int,
        max_prompt_tokens: int,
        *,
        source: str = "prompt",
        request_id: str | None = None,
    ):
        self.prompt_tokens = int(prompt_tokens)
        self.max_prompt_tokens = int(max_prompt_tokens)
        self.source = source
        self.request_id = request_id
        super().__init__(
            f"prompt_too_long: {source} has {self.prompt_tokens} tokens, "
            f"max prompt/context tokens is {self.max_prompt_tokens}"
        )


class VLMImagePrefillBudgetError(RuntimeError):
    """Raised when media-expanded VLM prefill would exceed a safe Metal budget."""

    code = "vlm_image_prefill_too_large"

    def __init__(self, detail: str, *, request_id: str | None = None):
        self.detail = str(detail)
        self.request_id = request_id
        super().__init__(self.detail)
