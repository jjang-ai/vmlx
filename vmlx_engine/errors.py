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


class UnsupportedMediaModalityError(RuntimeError):
    """Raised when a loaded family advertises media but lacks that runtime path."""

    code = "unsupported_media_modality"

    def __init__(
        self,
        modality: str,
        detail: str,
        *,
        family: str | None = None,
        request_id: str | None = None,
    ):
        self.modality = str(modality or "media")
        self.family = family
        self.detail = str(detail)
        self.request_id = request_id
        prefix = f"unsupported media modality {self.modality}"
        if family:
            prefix += f" for {family}"
        super().__init__(f"{prefix}: {self.detail}")


class MediaControlsUnmeetableError(ValueError):
    """Raised with ``media_controls_strict`` when a video/image control cannot
    be honoured as sent (the processor's floor/ceiling/grid, an unsupported
    control on this processor). Without strict mode the engine does the closest
    thing and reports the effective settings in ``warnings`` instead."""

    code = "media_controls_unmeetable"

    def __init__(self, detail: str, *, request_id: str | None = None):
        self.detail = str(detail)
        self.request_id = request_id
        super().__init__(self.detail)


class MediaInputError(ValueError):
    """Raised when a media part of the request cannot be used: a media-typed
    content part with no source under an accepted key, or an image/video
    source the loader cannot open. Never silently dropped into a text-only
    answer (live: a Responses ``input_video`` under the key ``video`` ran
    text-only with a 200)."""

    code = "media_input_invalid"

    def __init__(self, detail: str, *, request_id: str | None = None):
        self.detail = str(detail)
        self.request_id = request_id
        super().__init__(self.detail)
