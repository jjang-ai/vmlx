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
