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

from pydantic import BaseModel, Field, computed_field, field_validator

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
    - image_url: Image from URL or base64
    - video: Video from local path
    - video_url: Video from URL or base64
    - audio_url: Audio from URL or base64
    """

    type: str  # "text", "image_url", "video", "video_url", "audio_url"
    text: str | None = None
    image_url: ImageUrl | dict | str | None = None
    video: str | None = None
    video_url: VideoUrl | dict | str | None = None
    audio_url: AudioUrl | dict | str | None = None


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
    # For assistant messages with tool calls
    tool_calls: list[dict] | None = None
    # For tool response messages (role="tool")
    tool_call_id: str | None = None


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

    type: str = "text"  # "text", "json_object", "json_schema"
    json_schema: ResponseFormatJsonSchema | None = None


# =============================================================================
# Chat Completion
# =============================================================================


class StreamOptions(BaseModel):
    """Options for streaming responses."""

    include_usage: bool = False  # Include usage stats in final chunk


class ChatCompletionRequest(BaseModel):
    """Request for chat completion."""

    model: str
    messages: list[Message] = Field(..., min_length=1)
    temperature: float | None = None
    top_p: float | None = None
    max_tokens: int | None = None
    stream: bool = False
    stream_options: StreamOptions | None = (
        None  # Streaming options (include_usage, etc.)
    )
    stop: str | list[str] | None = None
    # Extended sampling parameters
    top_k: int | None = None  # Top-k sampling (0 = disabled)
    min_p: float | None = None  # Min-p sampling threshold
    repetition_penalty: float | None = None  # Repetition penalty (1.0 = disabled)
    frequency_penalty: float | None = None  # Accepted for API compat (not implemented)
    presence_penalty: float | None = None  # Accepted for API compat (not implemented)
    # Tool calling
    tools: list[ToolDefinition] | None = None
    tool_choice: str | dict | None = None  # "auto", "none", or specific tool
    # Number of completions (only n=1 supported; rejects n>1 with validation error)
    n: int | None = None
    # Structured output
    response_format: ResponseFormat | dict | None = None
    # MLLM-specific parameters
    video_fps: float | None = None
    video_max_frames: int | None = None
    # Request timeout in seconds (None = use server default)
    timeout: float | None = None
    # Thinking/reasoning toggle (None = auto from model config, True/False = explicit)
    enable_thinking: bool | None = None
    # Reasoning effort level for models that support it (e.g., GPT-OSS: low/medium/high)
    reasoning_effort: str | None = None
    # Extra kwargs passed directly to tokenizer.apply_chat_template()
    # Standard vLLM convention: {"enable_thinking": true/false, ...}
    # enable_thinking here is used as fallback when top-level enable_thinking is None
    chat_template_kwargs: dict | None = None

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


class ChatCompletionChoice(BaseModel):
    """A single choice in chat completion response."""

    index: int = 0
    message: AssistantMessage
    finish_reason: str | None = "stop"


class PromptTokensDetails(BaseModel):
    """Breakdown of prompt token usage."""

    cached_tokens: int = 0
    cache_detail: str | None = None  # e.g. "paged", "paged+ssm(23)+tq", "disk"


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
    # Request timeout in seconds (None = use server default)
    timeout: float | None = None

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


class MCPServersResponse(BaseModel):
    """Response for listing MCP servers."""

    servers: list[MCPServerInfo]


class MCPExecuteRequest(BaseModel):
    """Request to execute an MCP tool."""

    tool_name: str
    arguments: dict = Field(default_factory=dict)


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
    frequency_penalty: float | None = None  # Accepted for API compat (not implemented)
    presence_penalty: float | None = None  # Accepted for API compat (not implemented)
    max_output_tokens: int | None = None
    stop: str | list[str] | None = None
    stream: bool = False
    stream_options: StreamOptions | None = None  # Streaming options (include_usage)
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
    # Extra kwargs passed directly to tokenizer.apply_chat_template()
    chat_template_kwargs: dict | None = None
    # Request timeout in seconds (None = use server default)
    timeout: float | None = None
    # Video processing controls (MLLM models)
    video_fps: float | None = None
    video_max_frames: int | None = None

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
    output: list[ResponsesOutputMessage | ResponsesFunctionCall] = Field(default_factory=list)
    usage: ResponsesUsage = Field(default_factory=ResponsesUsage)
    previous_response_id: str | None = None
    error: dict | None = None


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


class ChatCompletionChunkChoice(BaseModel):
    """A single choice in a streaming chunk."""

    index: int = 0
    delta: ChatCompletionChunkDelta
    finish_reason: str | None = None


class ChatCompletionChunk(BaseModel):
    """A streaming chunk for chat completion."""

    id: str = Field(default_factory=lambda: f"chatcmpl-{uuid.uuid4().hex[:8]}")
    object: str = "chat.completion.chunk"
    created: int = Field(default_factory=lambda: int(time.time()))
    model: str
    choices: list[ChatCompletionChunkChoice]
    usage: Usage | None = None  # Included when stream_options.include_usage=true
