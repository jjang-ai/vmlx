# SPDX-License-Identifier: Apache-2.0
"""
Pydantic models for OpenAI-compatible API.

These models define the request and response schemas for:
- Chat completions
- Text completions
- Tool calling
- MCP (Model Context Protocol) integration
"""

import time
import uuid
from typing import Any

from pydantic import BaseModel, Field, computed_field, field_validator, model_validator

from ..video_controls import VIDEO_CONTROL_FIELDS, validate_video_controls
from ..image_controls import IMAGE_CONTROL_FIELDS, validate_image_controls


_NO_REASONING_EFFORTS = {"none", "off", "false", "disabled", "disable", "0"}


def _validate_openai_penalty(value: float | None, field_name: str) -> float | None:
    if value is not None and not (-2.0 <= value <= 2.0):
        raise ValueError(f"{field_name} must be between -2 and 2")
    return value


def _validate_logit_bias(value):
    if value is None:
        return None
    from ..utils.token_logits_processors import normalize_logit_bias

    return {str(token_id): bias for token_id, bias in normalize_logit_bias(value).items()}


def _is_no_reasoning_effort(value: str | None) -> bool:
    return isinstance(value, str) and value.strip().lower() in _NO_REASONING_EFFORTS


def _normalize_prompt_context_aliases(obj):
    """Normalize vMLX max prompt/context aliases onto max_prompt_tokens."""
    if getattr(obj, "max_prompt_tokens", None) is not None:
        return obj
    for alias in ("max_context_tokens", "max_context"):
        value = getattr(obj, alias, None)
        if value is not None:
            obj.max_prompt_tokens = value
            break
    return obj


def _validate_prompt_context_limit(v):
    if v is not None and v < 1:
        raise ValueError("max prompt/context tokens must be at least 1")
    return v


_GEMMA4_IMAGE_TOKEN_BUDGETS = {70, 140, 280, 560, 1120}


def _validate_image_token_budget(v):
    if v is not None and v not in _GEMMA4_IMAGE_TOKEN_BUDGETS:
        allowed = ", ".join(str(value) for value in sorted(_GEMMA4_IMAGE_TOKEN_BUDGETS))
        raise ValueError(f"image_token_budget must be one of: {allowed}")
    return v


# =============================================================================
# Content Types (for multimodal messages)
# =============================================================================


class ImageUrl(BaseModel):
    """Image URL with optional detail level."""

    url: str
    detail: str | None = None


class VideoUrl(BaseModel):
    """Video URL."""

    url: str


class AudioUrl(BaseModel):
    """Audio URL for audio content."""

    url: str


class ContentPart(BaseModel):
    """
    A part of a multimodal message content.

    Supports:
    - text: Plain text content
    - image_url / image: Image from URL or base64
    - video / video_url: Video from local path or URL/base64
    - audio_url: Audio from URL or base64
    - input_audio / audio: Audio dict {data: <base64>, format: "wav"|"mp3"|...}
      — the OpenAI audio chat schema used by Nemotron-3-Nano-Omni.
    """
    # Allow extra fields so future content-part types pass through to the
    # downstream dispatcher without Pydantic stripping them.
    model_config = {"extra": "allow"}

    type: str  # "text", "image_url", "image", "video", "video_url", "audio_url", "input_audio", "audio"
    text: str | None = None
    image_url: ImageUrl | dict | str | None = None
    image: dict | None = None
    video: str | None = None
    video_url: VideoUrl | dict | str | None = None
    audio_url: AudioUrl | dict | str | None = None
    input_audio: dict | None = None
    audio: dict | None = None


# =============================================================================
# Messages
# =============================================================================


class Message(BaseModel):
    """
    A message in a chat conversation.

    Supports:
    - Simple text messages (role + content string)
    - Multimodal messages (role + content list with text/images/videos)
    - Tool call messages (assistant with tool_calls)
    - Tool response messages (role="tool" with tool_call_id)
    """

    role: str
    content: str | list[ContentPart] | list[dict] | None = None
    # Private assistant reasoning carried forward for local multi-turn
    # templates.  This stays separate from visible ``content`` so OpenAI Chat,
    # Ollama ``message.thinking``, and Anthropic thinking blocks can round-trip
    # through the shared prompt renderer without polluting the answer rail.
    reasoning_content: str | None = None
    # For assistant messages with tool calls
    tool_calls: list[dict] | None = None
    # For tool response messages (role="tool")
    tool_call_id: str | None = None
    # Ollama replays tool results as ``role:"tool"`` with ``tool_name`` and no
    # tool_call_id. Without this field pydantic silently discarded the name
    # and the result rendered as an anonymous "[Tool Result ()]" (dialect F5).
    tool_name: str | None = None


# =============================================================================
# Tool Calling
# =============================================================================


class FunctionCall(BaseModel):
    """A function call with name and arguments."""

    name: str
    arguments: str  # JSON string


class ToolCall(BaseModel):
    """A tool call from the model."""

    id: str
    type: str = "function"
    function: FunctionCall


class ToolDefinition(BaseModel):
    """Definition of a tool that can be called by the model."""

    type: str = "function"
    function: dict


# =============================================================================
# Structured Output (JSON Schema)
# =============================================================================


class ResponseFormatJsonSchema(BaseModel):
    """JSON Schema definition for structured output."""

    name: str
    description: str | None = None
    schema_: dict = Field(alias="schema")  # JSON Schema specification
    strict: bool | None = False

    class Config:
        populate_by_name = True


class ResponseFormat(BaseModel):
    """
    Response format specification for structured output.

    Supports:
    - "text": Default text output (no structure enforcement)
    - "json_object": Forces valid JSON output
    - "json_schema": Forces JSON matching a specific schema
    """

    model_config = {"extra": "allow"}

    type: str = "text"  # "text", "json_object", "json_schema"
    json_schema: ResponseFormatJsonSchema | None = None


# =============================================================================
# Chat Completion
# =============================================================================


class StreamOptions(BaseModel):
    """Options for streaming responses."""

    # Chat Completions option. Responses usage is already carried by its
    # terminal response object; this field does not enable a Responses event.
    include_usage: bool = False
    # Standard Responses option. vMLX does not add padding today, but accepts
    # the field so OpenAI-compatible clients can send their normal shape.
    include_obfuscation: bool | None = None


class ChatCompletionRequest(BaseModel):
    """Request for chat completion."""

    model: str
    messages: list[Message] = Field(..., min_length=1)
    temperature: float | None = None
    top_p: float | None = None
    max_tokens: int | None = None
    # OpenAI deprecated `max_tokens` for chat in favour of this, so current
    # SDKs send it and nothing else. The model ignores unknown fields, so
    # before it was declared here a client that set only
    # `max_completion_tokens` silently got the default output cap — a
    # truncated or runaway generation with no error anywhere.
    max_completion_tokens: int | None = None
    stream: bool = False
    stream_options: StreamOptions | None = (
        None  # Streaming options (include_usage, etc.)
    )
    stop: str | list[str] | None = None
    # Extended sampling parameters
    top_k: int | None = None  # Top-k sampling (0 = disabled)
    min_p: float | None = None  # Min-p sampling threshold
    repetition_penalty: float | None = None  # Repetition penalty (1.0 = disabled)
    seed: int | None = None  # Per-request MLX sampling seed
    frequency_penalty: float | None = None
    presence_penalty: float | None = None
    logit_bias: dict[str, float] | None = None
    # OpenAI-compatible per-token logprobs. Chat uses a boolean switch plus
    # optional top_logprobs count (0-20).
    logprobs: bool | None = None
    top_logprobs: int | None = None
    # Tool calling
    tools: list[ToolDefinition] | None = None
    tool_choice: str | dict | None = None  # "auto", "none", or specific tool
    # Number of completions (only n=1 supported; rejects n>1 with validation error)
    n: int | None = None
    # Structured output
    response_format: ResponseFormat | dict | None = None
    # MLLM-specific parameters
    # Gemma 4 supports these exact visual soft-token budgets. Higher values
    # improve OCR/small-text fidelity at a proportional prefill cost.
    image_token_budget: int | None = None
    video_fps: float | None = None
    video_max_frames: int | None = None
    # Per-frame / per-clip pixel budgets and an explicit frame size (both
    # dimensions or neither). Validated together in validate_video_controls.
    video_max_pixels: int | None = None
    video_min_pixels: int | None = None
    video_total_pixels: int | None = None
    video_resized_height: int | None = None
    video_resized_width: int | None = None
    # Whole-clip vision-token budget (derived into the loader's per-clip pixel budget).
    video_token_budget: int | None = None
    # vMLX extension: per-request IMAGE preprocessing controls (request-local, cache-keyed);
    # media_controls_strict rejects a control the processor cannot honour instead of best-effort + warnings
    image_max_pixels: int | None = None
    image_min_pixels: int | None = None
    image_resized_height: int | None = None
    image_resized_width: int | None = None
    media_controls_strict: bool | None = None
    # Request timeout in seconds (None = use server default)
    timeout: float | None = None
    # vMLX extension: per-request prompt/context admission cap. This can
    # lower the server/session --max-prompt-tokens ceiling for one request,
    # but server.py will not let it raise above the session ceiling.
    max_prompt_tokens: int | None = None
    max_context_tokens: int | None = None
    max_context: int | None = None
    # Thinking/reasoning toggle (None = auto from model config, True/False = explicit)
    enable_thinking: bool | None = None
    # Reasoning effort level for models that support it (e.g., GPT-OSS: low/medium/high)
    reasoning_effort: str | None = None
    max_thinking_tokens: int | None = None
    # Canonical vMLX shorthand for discrete UI/API modes:
    # instruct/chat/off -> enable_thinking=False
    # reasoning/thinking/on -> enable_thinking=True, reasoning_effort=medium when absent
    # max/max_thinking -> enable_thinking=True, reasoning_effort=max when absent
    thinking_mode: str | None = None
    # Extra kwargs passed directly to tokenizer.apply_chat_template()
    # Standard vLLM convention: {"enable_thinking": true/false, ...}
    # enable_thinking here is used as fallback when top-level enable_thinking is None
    chat_template_kwargs: dict | None = None
    # Cache isolation / bypass control.
    # When cache_salt is non-empty OR skip_prefix_cache is True, the server
    # guarantees the request BYPASSES every prefix-cache layer:
    #   - Paged cache (block_aware_cache)
    #   - Memory-aware cache (MemoryAwarePrefixCache)
    #   - Legacy PrefixCacheManager
    #   - L2 disk cache
    #   - Block disk store
    #   - SSM companion cache (hybrid SSM models)
    #   - MLLM vision / pixel_values caches
    # Neither fetch nor store runs for the tagged request. Use for benchmark
    # runs that need guaranteed fresh execution without pollution from prior
    # requests — e.g., set `cache_salt: str(uuid.uuid4())` per run.
    #
    # Note: this is a per-request BYPASS. For multi-turn chats within a run,
    # pass cache_salt only on the first turn (or none) if you want cache hits
    # within the run; pass a new salt on every turn for strict isolation.
    cache_salt: str | None = None
    skip_prefix_cache: bool | None = None
    # mlxstudio#100 — Continue (VS Code) and other Anthropic-style clients
    # send the reasoning toggle as a nested object: `reasoning: {"effort": "..."}`.
    # Accept that shape too and normalize into `reasoning_effort` via a
    # model_validator below.
    reasoning: dict | None = None

    @model_validator(mode="after")
    def _normalize_reasoning_alias(self):
        _normalize_prompt_context_aliases(self)
        # If caller sent `reasoning: {"effort": "..."}` and didn't set
        # `reasoning_effort`, lift real effort names up so downstream code
        # sees them. Treat explicit no-reasoning aliases (`none`, `off`, ...)
        # as direct-rail requests; otherwise an OpenAI Responses payload like
        # `reasoning: {"effort": "none"}` is misread as "reasoning object
        # exists, enable thinking".
        #
        # Anthropic-style `reasoning: {"type": "enabled", "budget_tokens": N}`
        # still opts into thinking when no explicit effort is present.
        if self.reasoning is not None and self.reasoning_effort is None:
            eff = self.reasoning.get("effort")
            budget_tokens = self.reasoning.get("budget_tokens")
            if self.max_thinking_tokens is None and isinstance(budget_tokens, int):
                self.max_thinking_tokens = budget_tokens
            if _is_no_reasoning_effort(eff):
                if self.enable_thinking is None:
                    self.enable_thinking = False
            elif isinstance(eff, str) and eff:
                self.reasoning_effort = eff
            elif self.enable_thinking is None:
                # No effort but reasoning object present → opt-in to thinking
                self.enable_thinking = True
        if _is_no_reasoning_effort(self.reasoning_effort):
            self.reasoning_effort = None
            if self.enable_thinking is None:
                self.enable_thinking = False
        if self.thinking_mode is not None:
            mode = self.thinking_mode.strip().lower().replace("-", "_").replace(" ", "_")
            if mode in ("instruct", "instruction", "chat", "off", "none", "false"):
                if self.enable_thinking is None:
                    self.enable_thinking = False
                if self.reasoning_effort is None:
                    self.reasoning_effort = None
            elif mode in ("reasoning", "thinking", "think", "on", "true"):
                if self.enable_thinking is None:
                    self.enable_thinking = True
                if self.reasoning_effort is None:
                    self.reasoning_effort = "medium"
            elif mode in ("medium", "high"):
                if self.enable_thinking is None:
                    self.enable_thinking = True
                if self.reasoning_effort is None:
                    self.reasoning_effort = mode
            elif mode in ("max", "max_thinking", "maximum"):
                if self.enable_thinking is None:
                    self.enable_thinking = True
                if self.reasoning_effort is None:
                    self.reasoning_effort = "max"
            else:
                raise ValueError(
                    "thinking_mode must be one of: instruct, reasoning, max"
                )
        if self.top_logprobs is not None and self.logprobs is False:
            raise ValueError("top_logprobs requires logprobs=true")
        if self.top_logprobs is not None and self.logprobs is None:
            # OpenAI clients often send only top_logprobs. Treat that as a
            # request for logprobs instead of silently dropping it.
            self.logprobs = True
        return self

    @field_validator("max_prompt_tokens", "max_context_tokens", "max_context")
    @classmethod
    def validate_prompt_context_limit(cls, v):
        return _validate_prompt_context_limit(v)

    @field_validator("image_token_budget")
    @classmethod
    def validate_image_token_budget(cls, v):
        return _validate_image_token_budget(v)

    @model_validator(mode="after")
    def validate_video_controls(self):
        validate_video_controls({f: getattr(self, f, None) for f in VIDEO_CONTROL_FIELDS})
        validate_image_controls({f: getattr(self, f, None) for f in IMAGE_CONTROL_FIELDS})
        return self

    @field_validator("temperature")
    @classmethod
    def validate_temperature(cls, v):
        if v is not None and (v < 0 or v > 2):
            raise ValueError("temperature must be between 0 and 2")
        return v

    @field_validator("top_p")
    @classmethod
    def validate_top_p(cls, v):
        if v is not None and (v <= 0 or v > 1):
            raise ValueError("top_p must be between 0 (exclusive) and 1")
        return v

    @field_validator("max_tokens", "max_completion_tokens")
    @classmethod
    def validate_max_tokens(cls, v):
        if v is not None and v < 1:
            raise ValueError("max_tokens must be at least 1")
        return v

    @model_validator(mode="after")
    def fold_max_completion_tokens(self):
        """Collapse OpenAI's newer spelling onto ``max_tokens``.

        Everything downstream reads ``max_tokens``; folding here means the
        output cap, the projected-output guard and the reasoning-aware
        fallback all see the value without each having to know both spellings.
        Sending both with DIFFERENT values is a client bug worth surfacing
        rather than silently picking one.
        """
        if self.max_completion_tokens is None:
            return self
        if self.max_tokens is None:
            self.max_tokens = self.max_completion_tokens
        elif self.max_tokens != self.max_completion_tokens:
            raise ValueError(
                "max_tokens and max_completion_tokens both set and disagree "
                f"({self.max_tokens} vs {self.max_completion_tokens}); send one."
            )
        return self

    @field_validator("max_thinking_tokens")
    @classmethod
    def validate_max_thinking_tokens(cls, v):
        if v is not None and v < 1:
            raise ValueError("max_thinking_tokens must be at least 1")
        return v

    @field_validator("top_k")
    @classmethod
    def validate_top_k(cls, v):
        if v is not None and v < 0:
            raise ValueError("top_k must be >= 0")
        return v

    @field_validator("top_logprobs")
    @classmethod
    def validate_top_logprobs(cls, v):
        if v is not None and (v < 0 or v > 20):
            raise ValueError("top_logprobs must be between 0 and 20")
        return v

    @field_validator("min_p")
    @classmethod
    def validate_min_p(cls, v):
        if v is not None and (v < 0 or v > 1):
            raise ValueError("min_p must be between 0 and 1")
        return v

    @field_validator("repetition_penalty")
    @classmethod
    def validate_repetition_penalty(cls, v):
        if v is not None and v <= 0:
            raise ValueError("repetition_penalty must be > 0")
        return v

    @field_validator("frequency_penalty")
    @classmethod
    def validate_frequency_penalty(cls, v):
        return _validate_openai_penalty(v, "frequency_penalty")

    @field_validator("presence_penalty")
    @classmethod
    def validate_presence_penalty(cls, v):
        return _validate_openai_penalty(v, "presence_penalty")

    @field_validator("logit_bias")
    @classmethod
    def validate_logit_bias(cls, v):
        return _validate_logit_bias(v)

    @field_validator("stop")
    @classmethod
    def normalize_stop(cls, v):
        """Normalize bare string to list for consistent iteration."""
        if isinstance(v, str):
            return [v]
        return v

    @field_validator("n")
    @classmethod
    def validate_n(cls, v):
        if v is not None and v != 1:
            raise ValueError("Only n=1 is supported. Multiple completions are not implemented.")
        return v


class AssistantMessage(BaseModel):
    """Response message from the assistant."""

    role: str = "assistant"
    content: str | None = None
    reasoning: str | None = Field(
        default=None, exclude=True  # Internal storage; excluded from JSON
    )
    tool_calls: list[ToolCall] | None = None

    @computed_field
    @property
    def reasoning_content(self) -> str | None:
        """OpenAI O1-style reasoning field. Only present when thinking is enabled."""
        return self.reasoning

    def model_dump(self, **kwargs) -> dict:
        """Override to exclude reasoning_content when None."""
        d = super().model_dump(**kwargs)
        if d.get("reasoning_content") is None:
            d.pop("reasoning_content", None)
        return d


class ChatCompletionChoice(BaseModel):
    """A single choice in chat completion response."""

    index: int = 0
    message: AssistantMessage
    logprobs: dict | None = None
    finish_reason: str | None = "stop"


class PromptTokensDetails(BaseModel):
    """Breakdown of prompt token usage."""

    cached_tokens: int = 0
    # Tier/composition of the reused cache. Base values: "memory", "prefix",
    # "disk", "paged", "block-disk". Composite suffixes encode which extra state
    # was restored: "+ssm" (SSM companion), "+disk"/"+ssm+disk" (L2 spill),
    # "+mixed_swa", "+zaya_cca", "+tq-native", "+tq" (TurboQuant). DeepSeek V4
    # reports its native delta match as "paged"/"paged+disk".
    cache_detail: str | None = None


class Usage(BaseModel):
    """Token usage statistics."""

    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    prompt_tokens_details: PromptTokensDetails | None = None


class ChatCompletionResponse(BaseModel):
    """Response for chat completion."""

    id: str = Field(default_factory=lambda: f"chatcmpl-{uuid.uuid4().hex[:8]}")
    object: str = "chat.completion"
    created: int = Field(default_factory=lambda: int(time.time()))
    model: str
    choices: list[ChatCompletionChoice]
    usage: Usage = Field(default_factory=Usage)
    warnings: list[str] | None = None
    # Present only when the admission-time context clamp bound this request:
    # {prompt_tokens, requested_max_tokens, clamped_max_tokens,
    #  declared_context_tokens, exhausted}. `exhausted` is true when the
    # generation ran to the clamped budget — a finish_reason=length that
    # means CONTEXT EXHAUSTION, not an ordinary output-limit stop. Additive;
    # OpenAI clients ignore unknown fields.
    context_exhaustion: dict | None = None


# =============================================================================
# Text Completion
# =============================================================================


class CompletionRequest(BaseModel):
    """Request for text completion."""

    model: str
    prompt: str | list[str]
    temperature: float | None = None
    top_p: float | None = None
    max_tokens: int | None = None
    stream: bool = False
    stop: str | list[str] | None = None
    # Extended sampling parameters
    top_k: int | None = None
    min_p: float | None = None
    repetition_penalty: float | None = None
    seed: int | None = None  # Per-request MLX sampling seed
    frequency_penalty: float | None = None
    presence_penalty: float | None = None
    logit_bias: dict[str, float] | None = None
    # Legacy OpenAI Completions logprobs count. None disables logprobs;
    # 0 returns sampled token logprobs without alternate top tokens; 1-20
    # returns that many top alternatives per generated token.
    logprobs: int | None = None
    # Request timeout in seconds (None = use server default)
    timeout: float | None = None
    max_prompt_tokens: int | None = None
    max_context_tokens: int | None = None
    max_context: int | None = None
    # Cache bypass (see ChatCompletionRequest.cache_salt for semantics).
    cache_salt: str | None = None
    skip_prefix_cache: bool | None = None
    # Raw completions on a chat-template-only family (any is_mllm engine) are
    # wrapped through the chat rail with enable_thinking forced off by
    # default -- these let a caller explicitly opt into reasoning instead of
    # having it silently dropped with no error. None preserves the existing
    # forced-off default; only an explicit value changes behavior.
    reasoning_effort: str | None = None
    enable_thinking: bool | None = None

    @model_validator(mode="after")
    def _normalize_prompt_context_alias(self):
        return _normalize_prompt_context_aliases(self)

    @field_validator("temperature")
    @classmethod
    def validate_temperature(cls, v):
        if v is not None and (v < 0 or v > 2):
            raise ValueError("temperature must be between 0 and 2")
        return v

    @field_validator("top_p")
    @classmethod
    def validate_top_p(cls, v):
        if v is not None and (v <= 0 or v > 1):
            raise ValueError("top_p must be between 0 (exclusive) and 1")
        return v

    @field_validator("max_tokens")
    @classmethod
    def validate_max_tokens(cls, v):
        if v is not None and v < 1:
            raise ValueError("max_tokens must be at least 1")
        return v

    @field_validator("max_prompt_tokens", "max_context_tokens", "max_context")
    @classmethod
    def validate_prompt_context_limit(cls, v):
        return _validate_prompt_context_limit(v)

    @field_validator("top_k")
    @classmethod
    def validate_top_k(cls, v):
        if v is not None and v < 0:
            raise ValueError("top_k must be >= 0")
        return v

    @field_validator("min_p")
    @classmethod
    def validate_min_p(cls, v):
        if v is not None and (v < 0 or v > 1):
            raise ValueError("min_p must be between 0 and 1")
        return v

    @field_validator("repetition_penalty")
    @classmethod
    def validate_repetition_penalty(cls, v):
        if v is not None and v <= 0:
            raise ValueError("repetition_penalty must be > 0")
        return v

    @field_validator("frequency_penalty")
    @classmethod
    def validate_frequency_penalty(cls, v):
        return _validate_openai_penalty(v, "frequency_penalty")

    @field_validator("presence_penalty")
    @classmethod
    def validate_presence_penalty(cls, v):
        return _validate_openai_penalty(v, "presence_penalty")

    @field_validator("logit_bias")
    @classmethod
    def validate_logit_bias(cls, v):
        return _validate_logit_bias(v)

    @field_validator("logprobs")
    @classmethod
    def validate_logprobs(cls, v):
        if v is not None and (v < 0 or v > 20):
            raise ValueError("logprobs must be between 0 and 20")
        return v

    @field_validator("stop")
    @classmethod
    def normalize_stop(cls, v):
        if isinstance(v, str):
            return [v]
        return v


class CompletionChoice(BaseModel):
    """A single choice in text completion response."""

    index: int = 0
    text: str
    logprobs: dict | None = None
    finish_reason: str | None = "stop"


class CompletionResponse(BaseModel):
    """Response for text completion."""

    id: str = Field(default_factory=lambda: f"cmpl-{uuid.uuid4().hex[:8]}")
    object: str = "text_completion"
    created: int = Field(default_factory=lambda: int(time.time()))
    model: str
    choices: list[CompletionChoice]
    usage: Usage = Field(default_factory=Usage)


# =============================================================================
# Models List
# =============================================================================


class ModelInfo(BaseModel):
    """Information about an available model."""

    id: str
    object: str = "model"
    created: int = Field(default_factory=lambda: int(time.time()))
    owned_by: str = "vmlx-engine"


class ModelsResponse(BaseModel):
    """Response for listing models."""

    object: str = "list"
    data: list[ModelInfo]


# =============================================================================
# MCP (Model Context Protocol)
# =============================================================================


class MCPToolInfo(BaseModel):
    """Information about an MCP tool."""

    name: str
    description: str
    server: str
    parameters: dict = Field(default_factory=dict)
    enabled: bool = True
    effective: bool = True
    source: str = "mcp"
    transport: str | None = None
    server_state: str | None = None
    error: str | None = None


class MCPToolsResponse(BaseModel):
    """Response for listing MCP tools."""

    tools: list[MCPToolInfo]
    count: int


class MCPServerInfo(BaseModel):
    """Information about an MCP server."""

    name: str
    state: str
    transport: str
    tools_count: int
    error: str | None = None
    enabled: bool = True
    configured: bool = True
    command_redacted: str | None = None
    url_redacted: str | None = None
    last_connected: float | None = None
    env_keys: list[str] = Field(default_factory=list)
    header_keys: list[str] = Field(default_factory=list)


class MCPServersResponse(BaseModel):
    """Response for listing MCP servers."""

    servers: list[MCPServerInfo]


class MCPExecuteRequest(BaseModel):
    """Request to execute an MCP tool."""

    tool_name: str
    arguments: dict = Field(default_factory=dict)
    model: str | None = None


class MCPExecuteResponse(BaseModel):
    """Response from executing an MCP tool."""

    tool_name: str
    content: str | list | dict | None = None
    is_error: bool = False
    error_message: str | None = None


# =============================================================================
# Audio (STT/TTS)
# =============================================================================


class AudioSpeechRequest(BaseModel):
    """Request for text-to-speech."""

    model: str = "kokoro"
    input: str
    voice: str = "af_heart"
    speed: float = 1.0
    response_format: str = "wav"

    @field_validator("speed")
    @classmethod
    def validate_speed(cls, v):
        if v <= 0 or v > 4.0:
            raise ValueError("speed must be between 0 (exclusive) and 4.0")
        return v


# =============================================================================
# Embeddings
# =============================================================================


class EmbeddingRequest(BaseModel):
    """Request for text embeddings (OpenAI compatible)."""

    input: str | list[str]
    model: str
    encoding_format: str | None = "float"  # "float" or "base64"


class EmbeddingData(BaseModel):
    """A single embedding result."""

    object: str = "embedding"
    index: int
    embedding: list[float]


class EmbeddingUsage(BaseModel):
    """Token usage for embedding requests."""

    prompt_tokens: int = 0
    total_tokens: int = 0


class EmbeddingResponse(BaseModel):
    """Response for embeddings endpoint (OpenAI compatible)."""

    object: str = "list"
    data: list[EmbeddingData]
    model: str
    usage: EmbeddingUsage = Field(default_factory=EmbeddingUsage)


# =============================================================================
# Streaming (for SSE responses)
# =============================================================================


# =============================================================================
# Responses API (OpenAI /v1/responses format)
# =============================================================================


class ResponsesOutputText(BaseModel):
    """Text content in a Responses API output message."""

    type: str = "output_text"
    text: str = ""
    annotations: list = Field(default_factory=list)


class ResponsesOutputMessage(BaseModel):
    """A message in the Responses API output array."""

    type: str = "message"
    id: str = Field(default_factory=lambda: f"item_{uuid.uuid4().hex[:12]}")
    status: str = "completed"
    role: str = "assistant"
    content: list[ResponsesOutputText] = Field(default_factory=list)


class ResponsesReasoningSummaryText(BaseModel):
    """A summary part in a Responses API reasoning output item."""

    type: str = "summary_text"
    text: str = ""


class ResponsesReasoningText(BaseModel):
    """Raw reasoning content in a Responses API reasoning output item."""

    type: str = "reasoning_text"
    text: str = ""


class ResponsesReasoningItem(BaseModel):
    """A reasoning item in the Responses API output array."""

    type: str = "reasoning"
    id: str = Field(default_factory=lambda: f"rs_{uuid.uuid4().hex[:12]}")
    status: str = "completed"
    summary: list[ResponsesReasoningSummaryText] = Field(default_factory=list)
    content: list[ResponsesReasoningText] = Field(default_factory=list)


class ResponsesFunctionCall(BaseModel):
    """A function call in the Responses API output array."""

    type: str = "function_call"
    id: str = Field(default_factory=lambda: f"fc_{uuid.uuid4().hex[:12]}")
    call_id: str = Field(default_factory=lambda: f"call_{uuid.uuid4().hex[:12]}")
    name: str = ""
    arguments: str = ""
    status: str = "completed"


class ResponsesTextFormat(BaseModel):
    """Text format specification for Responses API."""

    model_config = {"extra": "allow"}

    type: str = "text"  # "text", "json_object", "json_schema"
    json_schema: dict | None = None  # Schema definition for json_schema type


class ResponsesToolDefinition(BaseModel):
    """Tool definition in Responses API flat format.

    Responses API uses: {"type":"function","name":"...","parameters":{...}}
    Chat Completions uses: {"type":"function","function":{"name":"...","parameters":{...}}}
    """

    type: str = "function"
    name: str
    description: str | None = None
    parameters: dict | None = None
    strict: bool | None = None

    def to_chat_completions_format(self) -> dict:
        """Convert flat Responses format to nested Chat Completions format."""
        func = {"name": self.name}
        if self.description:
            func["description"] = self.description
        if self.parameters:
            func["parameters"] = self.parameters
        if self.strict is not None:
            func["strict"] = self.strict
        return {"type": "function", "function": func}


class ResponsesRequest(BaseModel):
    """Request for OpenAI Responses API (POST /v1/responses)."""

    model_config = {"extra": "ignore"}

    model: str
    input: str | list[dict] | list[Message]
    instructions: str | None = None
    temperature: float | None = None
    top_p: float | None = None
    top_k: int | None = None
    min_p: float | None = None
    repetition_penalty: float | None = None
    seed: int | None = None  # Per-request MLX sampling seed
    frequency_penalty: float | None = None
    presence_penalty: float | None = None
    max_output_tokens: int | None = None
    stop: str | list[str] | None = None
    stream: bool = False
    stream_options: StreamOptions | None = None  # Responses: include_obfuscation
    # Accept both flat (Responses API) and nested (Chat Completions) tool formats, plus built-in tools
    tools: list[dict] | None = None
    tool_choice: str | dict | None = None
    text: ResponsesTextFormat | dict | None = None
    # For multi-turn chaining
    previous_response_id: str | None = None
    store: bool = False
    # Thinking/reasoning toggle (None = auto from model config, True/False = explicit)
    enable_thinking: bool | None = None
    # Reasoning effort level for models that support it (e.g., GPT-OSS: low/medium/high)
    reasoning_effort: str | None = None
    max_thinking_tokens: int | None = None
    thinking_mode: str | None = None
    reasoning: dict | None = None
    # Extra kwargs passed directly to tokenizer.apply_chat_template()
    chat_template_kwargs: dict | None = None
    # Request timeout in seconds (None = use server default)
    timeout: float | None = None
    max_prompt_tokens: int | None = None
    max_context_tokens: int | None = None
    max_context: int | None = None
    # Video processing controls (MLLM models)
    image_token_budget: int | None = None
    video_fps: float | None = None
    video_max_frames: int | None = None
    # Per-frame / per-clip pixel budgets and an explicit frame size (both
    # dimensions or neither). Validated together in validate_video_controls.
    video_max_pixels: int | None = None
    video_min_pixels: int | None = None
    video_total_pixels: int | None = None
    video_resized_height: int | None = None
    video_resized_width: int | None = None
    # Whole-clip vision-token budget (derived into the loader's per-clip pixel budget).
    video_token_budget: int | None = None
    # vMLX extension: per-request IMAGE preprocessing controls (request-local, cache-keyed);
    # media_controls_strict rejects a control the processor cannot honour instead of best-effort + warnings
    image_max_pixels: int | None = None
    image_min_pixels: int | None = None
    image_resized_height: int | None = None
    image_resized_width: int | None = None
    media_controls_strict: bool | None = None
    # Cache bypass — parity with ChatCompletionRequest.cache_salt /
    # skip_prefix_cache. Without these fields, `model_config={"extra":
    # "ignore"}` silently drops them, and Responses-API clients (Claude
    # Code, OpenAI SDK) cannot bypass cache for testing or for
    # explicitly requested fresh state. Wired through to the same
    # `_compute_bypass_prefix_cache` plumbing as Chat/Completions.
    cache_salt: str | None = None
    skip_prefix_cache: bool | None = None

    @model_validator(mode="after")
    def _normalize_reasoning_alias(self):
        _normalize_prompt_context_aliases(self)
        if self.reasoning is not None and self.reasoning_effort is None:
            eff = self.reasoning.get("effort")
            budget_tokens = self.reasoning.get("budget_tokens")
            if self.max_thinking_tokens is None and isinstance(budget_tokens, int):
                self.max_thinking_tokens = budget_tokens
            if _is_no_reasoning_effort(eff):
                if self.enable_thinking is None:
                    self.enable_thinking = False
            elif isinstance(eff, str) and eff:
                self.reasoning_effort = eff
            elif self.enable_thinking is None:
                self.enable_thinking = True
        if _is_no_reasoning_effort(self.reasoning_effort):
            self.reasoning_effort = None
            if self.enable_thinking is None:
                self.enable_thinking = False
        if self.thinking_mode is not None:
            mode = self.thinking_mode.strip().lower().replace("-", "_").replace(" ", "_")
            if mode in ("instruct", "instruction", "chat", "off", "none", "false"):
                if self.enable_thinking is None:
                    self.enable_thinking = False
                if self.reasoning_effort is None:
                    self.reasoning_effort = None
            elif mode in ("reasoning", "thinking", "think", "on", "true"):
                if self.enable_thinking is None:
                    self.enable_thinking = True
                if self.reasoning_effort is None:
                    self.reasoning_effort = "medium"
            elif mode in ("medium", "high"):
                if self.enable_thinking is None:
                    self.enable_thinking = True
                if self.reasoning_effort is None:
                    self.reasoning_effort = mode
            elif mode in ("max", "max_thinking", "maximum"):
                if self.enable_thinking is None:
                    self.enable_thinking = True
                if self.reasoning_effort is None:
                    self.reasoning_effort = "max"
            else:
                raise ValueError(
                    "thinking_mode must be one of: instruct, reasoning, max"
                )
        return self

    @field_validator("max_prompt_tokens", "max_context_tokens", "max_context")
    @classmethod
    def validate_prompt_context_limit(cls, v):
        return _validate_prompt_context_limit(v)

    @field_validator("image_token_budget")
    @classmethod
    def validate_image_token_budget(cls, v):
        return _validate_image_token_budget(v)

    @model_validator(mode="after")
    def validate_video_controls(self):
        validate_video_controls({f: getattr(self, f, None) for f in VIDEO_CONTROL_FIELDS})
        validate_image_controls({f: getattr(self, f, None) for f in IMAGE_CONTROL_FIELDS})
        return self

    @field_validator("temperature")
    @classmethod
    def validate_temperature(cls, v):
        if v is not None and (v < 0 or v > 2):
            raise ValueError("temperature must be between 0 and 2")
        return v

    @field_validator("top_p")
    @classmethod
    def validate_top_p(cls, v):
        if v is not None and (v <= 0 or v > 1):
            raise ValueError("top_p must be between 0 (exclusive) and 1")
        return v

    @field_validator("max_output_tokens")
    @classmethod
    def validate_max_output_tokens(cls, v):
        if v is not None and v < 1:
            raise ValueError("max_output_tokens must be at least 1")
        return v

    @field_validator("max_thinking_tokens")
    @classmethod
    def validate_max_thinking_tokens(cls, v):
        if v is not None and v < 1:
            raise ValueError("max_thinking_tokens must be at least 1")
        return v

    @field_validator("top_k")
    @classmethod
    def validate_top_k(cls, v):
        if v is not None and v < 0:
            raise ValueError("top_k must be >= 0")
        return v

    @field_validator("min_p")
    @classmethod
    def validate_min_p(cls, v):
        if v is not None and (v < 0 or v > 1):
            raise ValueError("min_p must be between 0 and 1")
        return v

    @field_validator("repetition_penalty")
    @classmethod
    def validate_repetition_penalty(cls, v):
        if v is not None and v <= 0:
            raise ValueError("repetition_penalty must be > 0")
        return v

    @field_validator("frequency_penalty")
    @classmethod
    def validate_frequency_penalty(cls, v):
        return _validate_openai_penalty(v, "frequency_penalty")

    @field_validator("presence_penalty")
    @classmethod
    def validate_presence_penalty(cls, v):
        return _validate_openai_penalty(v, "presence_penalty")

    @field_validator("stop")
    @classmethod
    def normalize_stop(cls, v):
        """Normalize bare string to list for consistent iteration."""
        if isinstance(v, str):
            return [v]
        return v


class InputTokensDetails(BaseModel):
    """Breakdown of input token usage (Responses API format)."""

    cached_tokens: int = 0
    cache_detail: str | None = None


class ResponsesUsage(BaseModel):
    """Usage for Responses API (uses input_tokens/output_tokens per spec)."""

    input_tokens: int = 0
    output_tokens: int = 0
    total_tokens: int = 0
    input_tokens_details: InputTokensDetails | None = None


class ResponsesObject(BaseModel):
    """Response for Responses API."""

    id: str = Field(default_factory=lambda: f"resp_{uuid.uuid4().hex[:12]}")
    object: str = "response"
    created_at: int = Field(default_factory=lambda: int(time.time()))
    status: str = "completed"
    model: str
    output: list[
        ResponsesOutputMessage | ResponsesReasoningItem | ResponsesFunctionCall
    ] = Field(default_factory=list)
    usage: ResponsesUsage = Field(default_factory=ResponsesUsage)
    previous_response_id: str | None = None
    # Values are not uniformly strings: the length terminal attaches a nested
    # context_exhaustion record (dict of ints) next to the spec "reason" key.
    # dict[str, str] rejected that nested dict at pydantic validation on the
    # NON-stream door only (the stream terminal is a raw dict).
    incomplete_details: dict[str, Any] | None = None
    error: dict | None = None
    # Additive (#175): set when an out-of-set reasoning_effort was coerced to a
    # stamped tier — {requested_effort, effective_effort, stamped_levels}.
    effort_substitution: dict | None = None
    # Non-fatal warnings surfaced to the client. Set when the response is
    # technically valid but a chain/coherence/cache-prefix risk applies — e.g.,
    # `previous_response_id` chained a prior response that produced reasoning
    # only (no visible message, no tool calls).
    warnings: list[str] | None = None

    @computed_field
    @property
    def output_text(self) -> str | None:
        """Convenience concatenation of visible Responses output text."""
        chunks: list[str] = []
        for item in self.output:
            if not isinstance(item, ResponsesOutputMessage):
                continue
            for part in item.content:
                if part.type in ("output_text", "text") and part.text:
                    chunks.append(part.text)
        return "".join(chunks) if chunks else None


# =============================================================================
# Streaming (for SSE responses)
# =============================================================================


class ChatCompletionChunkDelta(BaseModel):
    """Delta content in a streaming chunk."""

    role: str | None = None
    content: str | None = None
    reasoning: str | None = Field(
        default=None, exclude=True  # Internal storage; excluded from JSON
    )
    tool_calls: list[dict] | None = None

    @computed_field
    @property
    def reasoning_content(self) -> str | None:
        """OpenAI O1-style reasoning field. Only present when thinking is enabled."""
        return self.reasoning

    def model_dump(self, **kwargs) -> dict:
        """Override to exclude reasoning_content when None (#46).

        Pydantic's computed_field is not excluded by exclude_none=True,
        which causes 'reasoning_content: null' to leak into every SSE
        chunk — breaking strict OpenAI SDK parsers (Claude Code, etc.).
        """
        d = super().model_dump(**kwargs)
        if d.get("reasoning_content") is None:
            d.pop("reasoning_content", None)
        return d


class ChatCompletionChunkChoice(BaseModel):
    """A single choice in a streaming chunk."""

    index: int = 0
    delta: ChatCompletionChunkDelta
    logprobs: dict | None = None
    finish_reason: str | None = None


class ChatCompletionChunk(BaseModel):
    """A streaming chunk for chat completion."""

    id: str = Field(default_factory=lambda: f"chatcmpl-{uuid.uuid4().hex[:8]}")
    object: str = "chat.completion.chunk"
    created: int = Field(default_factory=lambda: int(time.time()))
    model: str
    choices: list[ChatCompletionChunkChoice]
    usage: Usage | None = None  # Included when stream_options.include_usage=true
    warnings: list[str] | None = None
    tool_call_generating: bool | None = None  # vMLX UI hint while native XML is buffered
