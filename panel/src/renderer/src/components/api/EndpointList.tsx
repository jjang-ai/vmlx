import type { ApiFormat } from "./ApiDashboard";
import { useTranslation } from "../../i18n";

// Maps English category/description strings from the static endpoint data
// tables above to their i18n key. Falls through to the English value when
// no key is mapped — safe default that keeps the wire labels accurate even
// if translations drift.
const CATEGORY_KEYS: Record<string, string> = {
  "Chat & Completions": "endpoints.chatCompletions",
  "Models": "endpoints.models",
  "Image Generation": "endpoints.imageGeneration",
  "Embeddings & Reranking": "endpoints.embeddingsReranking",
  "Audio": "endpoints.audio",
  "Cache": "endpoints.cache",
  "MCP Tools": "endpoints.mcpTools",
  "Cancel": "endpoints.cancelCategory",
  "Health": "endpoints.health",
  "Anthropic": "endpoints.anthropic",
};

const DESCRIPTION_KEYS: Record<string, string> = {
  "OpenAI-compatible chat and text completion endpoints": "endpoints.chatCompletionsDesc",
  "Model information and management": "endpoints.modelsDesc",
  "Text-to-image generation and image editing (requires mflux)": "endpoints.imageGenerationDesc",
  "Vector embeddings and reranking (requires embedding model)": "endpoints.embeddingsRerankingDesc",
  "Speech-to-text and text-to-speech (requires mlx-audio)": "endpoints.audioDesc",
  "KV cache management for prefix caching": "endpoints.cacheDesc",
  "Model Context Protocol tool integration": "endpoints.mcpToolsDesc",
  "Cancel in-flight streaming requests": "endpoints.cancelDesc",
  "Server health and status": "endpoints.healthDesc",
  "Anthropic Messages API — use with Claude Code, Anthropic SDK, or any Anthropic-compatible client": "endpoints.anthropicDesc",
};

const ENDPOINT_DESC_KEYS: Record<string, string> = {
  "Chat completions (OpenAI format)": "endpoints.chatCompletionsEndpoint",
  "Responses API (OpenAI format)": "endpoints.responsesEndpoint",
  "Text completions (legacy format)": "endpoints.textCompletionsEndpoint",
  "Anthropic Messages API": "endpoints.anthropicMessagesEndpoint",
  "List loaded models": "endpoints.listModelsEndpoint",
  "Generate images from text prompts": "endpoints.generateImagesEndpoint",
  "Generate images from text prompts (OpenAI format)": "endpoints.generateImagesEndpoint",
  "Edit images with text instructions": "endpoints.editImagesEndpoint",
  "Edit images with text instructions (Qwen-Image-Edit)": "endpoints.editImagesEndpoint",
  "Generate text embeddings": "endpoints.embeddingsEndpoint",
  "Rerank documents by relevance": "endpoints.rerankEndpoint",
  "Transcribe audio (Whisper STT)": "endpoints.transcriptionsEndpoint",
  "Generate speech (Kokoro TTS)": "endpoints.speechEndpoint",
  "List available TTS voices": "endpoints.voicesEndpoint",
  "Cache statistics (hit rate, size)": "endpoints.cacheStatsEndpoint",
  "List cached prefix entries": "endpoints.cacheEntriesEndpoint",
  "Pre-warm cache with a prompt": "endpoints.cacheWarmEndpoint",
  "Clear all cached entries": "endpoints.cacheClearEndpoint",
  "List available MCP tools": "endpoints.mcpToolsListEndpoint",
  "MCP server connection status": "endpoints.mcpServersEndpoint",
  "Execute an MCP tool": "endpoints.mcpExecuteEndpoint",
  "Cancel a chat completion stream": "endpoints.cancelChatEndpoint",
  "Cancel a responses stream": "endpoints.cancelResponsesEndpoint",
  "Cancel a text completion stream": "endpoints.cancelCompletionsEndpoint",
  "Health check — model status, memory, MCP info": "endpoints.healthEndpoint",
};

interface Endpoint {
  method: "GET" | "POST" | "DELETE";
  path: string;
  description: string;
  stream?: boolean;
  auth?: boolean;
}

interface EndpointGroup {
  category: string;
  description: string;
  endpoints: Endpoint[];
}

const OPENAI_GROUPS: EndpointGroup[] = [
  {
    category: "Chat & Completions",
    description: "OpenAI-compatible chat and text completion endpoints",
    endpoints: [
      {
        method: "POST",
        path: "/v1/chat/completions",
        description: "Chat completions (OpenAI format)",
        stream: true,
        auth: true,
      },
      {
        method: "POST",
        path: "/v1/responses",
        description: "Responses API (OpenAI format)",
        stream: true,
        auth: true,
      },
      {
        method: "POST",
        path: "/v1/completions",
        description: "Text completions (legacy format)",
        stream: true,
        auth: true,
      },
    ],
  },
  {
    category: "Models",
    description: "Model information and management",
    endpoints: [
      {
        method: "GET",
        path: "/v1/models",
        description: "List loaded models",
        auth: true,
      },
    ],
  },
  {
    category: "Image Generation",
    description: "Text-to-image generation and image editing (requires mflux)",
    endpoints: [
      {
        method: "POST",
        path: "/v1/images/generations",
        description: "Generate images from text prompts",
        auth: true,
      },
      {
        method: "POST",
        path: "/v1/images/edits",
        description: "Edit images with text instructions",
        auth: true,
      },
    ],
  },
  {
    category: "Embeddings & Reranking",
    description: "Vector embeddings and reranking (requires embedding model)",
    endpoints: [
      {
        method: "POST",
        path: "/v1/embeddings",
        description: "Generate text embeddings",
        auth: true,
      },
      {
        method: "POST",
        path: "/v1/rerank",
        description: "Rerank documents by relevance",
        auth: true,
      },
    ],
  },
  {
    category: "Audio",
    description: "Speech-to-text and text-to-speech (requires mlx-audio)",
    endpoints: [
      {
        method: "POST",
        path: "/v1/audio/transcriptions",
        description: "Transcribe audio (Whisper STT)",
        auth: true,
      },
      {
        method: "POST",
        path: "/v1/audio/speech",
        description: "Generate speech (Kokoro TTS)",
        stream: true,
        auth: true,
      },
      {
        method: "GET",
        path: "/v1/audio/voices",
        description: "List available TTS voices",
        auth: true,
      },
    ],
  },
  {
    category: "Cache",
    description: "KV cache management for prefix caching",
    endpoints: [
      {
        method: "GET",
        path: "/v1/cache/stats",
        description: "Cache statistics (hit rate, size)",
        auth: true,
      },
      {
        method: "GET",
        path: "/v1/cache/entries",
        description: "List cached prefix entries",
        auth: true,
      },
      {
        method: "POST",
        path: "/v1/cache/warm",
        description: "Pre-warm cache with a prompt",
        auth: true,
      },
      {
        method: "DELETE",
        path: "/v1/cache",
        description: "Clear all cached entries",
        auth: true,
      },
    ],
  },
  {
    category: "MCP Tools",
    description: "Model Context Protocol tool integration",
    endpoints: [
      {
        method: "GET",
        path: "/v1/mcp/tools",
        description: "List available MCP tools",
        auth: true,
      },
      {
        method: "GET",
        path: "/v1/mcp/servers",
        description: "MCP server connection status",
        auth: true,
      },
      {
        method: "POST",
        path: "/v1/mcp/execute",
        description: "Execute an MCP tool",
        auth: true,
      },
    ],
  },
  {
    category: "Cancel",
    description: "Cancel in-flight streaming requests",
    endpoints: [
      {
        method: "POST",
        path: "/v1/chat/completions/{id}/cancel",
        description: "Cancel a chat completion stream",
      },
      {
        method: "POST",
        path: "/v1/responses/{id}/cancel",
        description: "Cancel a responses stream",
      },
      {
        method: "POST",
        path: "/v1/completions/{id}/cancel",
        description: "Cancel a text completion stream",
      },
    ],
  },
  {
    category: "Health",
    description: "Server health and status",
    endpoints: [
      {
        method: "GET",
        path: "/health",
        description: "Health check — model status, memory, MCP info",
      },
    ],
  },
];

const ANTHROPIC_GROUPS: EndpointGroup[] = [
  {
    category: "Messages API",
    description:
      "Anthropic Messages API — use with Claude Code, Anthropic SDK, or any Anthropic-compatible client",
    endpoints: [
      {
        method: "POST",
        path: "/v1/messages",
        description: "Create a message (streaming or non-streaming)",
        stream: true,
        auth: true,
      },
    ],
  },
  {
    category: "Models",
    description: "Model information",
    endpoints: [
      {
        method: "GET",
        path: "/v1/models",
        description: "List loaded models",
        auth: true,
      },
    ],
  },
  {
    category: "Health",
    description: "Server health and status",
    endpoints: [
      { method: "GET", path: "/health", description: "Health check" },
    ],
  },
];

const OLLAMA_GROUPS: EndpointGroup[] = [
  {
    category: "Chat & Generate",
    description:
      "Ollama-compatible endpoints — use with ollama CLI or any Ollama client. Responses use NDJSON streaming.",
    endpoints: [
      {
        method: "POST",
        path: "/api/chat",
        description: "Chat completion (NDJSON streaming)",
        stream: true,
      },
      {
        method: "POST",
        path: "/api/generate",
        description: "Text generation (NDJSON streaming)",
        stream: true,
      },
    ],
  },
  {
    category: "Models & Info",
    description: "Model information and embeddings",
    endpoints: [
      { method: "GET", path: "/api/tags", description: "List loaded models" },
      { method: "POST", path: "/api/show", description: "Show model details" },
      {
        method: "POST",
        path: "/api/embeddings",
        description: "Generate embeddings",
      },
      {
        method: "POST",
        path: "/api/embed",
        description: "Generate embeddings (alias)",
      },
    ],
  },
  {
    category: "Liveness",
    description: "Ollama compatibility check",
    endpoints: [
      {
        method: "GET",
        path: "/",
        description: 'Liveness check — returns "vMLX Gateway is running"',
      },
    ],
  },
];

const FORMAT_GROUPS: Record<ApiFormat, EndpointGroup[]> = {
  openai: OPENAI_GROUPS,
  anthropic: ANTHROPIC_GROUPS,
  ollama: OLLAMA_GROUPS,
};

const METHOD_COLORS: Record<string, string> = {
  GET: "bg-blue-500/15 text-blue-500",
  POST: "bg-green-500/15 text-green-500",
  DELETE: "bg-red-500/15 text-red-500",
};

interface EndpointListProps {
  format?: ApiFormat;
  isImage?: boolean;
  isEdit?: boolean;
}

export function EndpointList({ format = "openai" }: EndpointListProps) {
  const { t } = useTranslation();
  const groups = FORMAT_GROUPS[format] || OPENAI_GROUPS;
  const formatLabel =
    format === "openai"
      ? "OpenAI"
      : format === "anthropic"
        ? "Anthropic"
        : "Ollama";

  return (
    <div className="space-y-4">
      <h3 className="text-sm font-medium">
        {formatLabel} {t('endpoints.title')}
      </h3>
      {groups.map((group) => {
        const catKey = CATEGORY_KEYS[group.category];
        const descKey = DESCRIPTION_KEYS[group.description];
        return (
        <div
          key={group.category}
          className="border border-border rounded-lg overflow-hidden"
        >
          <div className="px-3 py-2 bg-muted/50 border-b border-border">
            <h4 className="text-xs font-medium">{catKey ? t(catKey) : group.category}</h4>
            <p className="text-[10px] text-muted-foreground">
              {descKey ? t(descKey) : group.description}
            </p>
          </div>
          <div className="divide-y divide-border">
            {group.endpoints.map((ep) => {
              const epKey = ENDPOINT_DESC_KEYS[ep.description];
              return (
              <div key={ep.path} className="px-3 py-2 flex items-center gap-2">
                <span
                  className={`text-[10px] font-mono font-bold px-1.5 py-0.5 rounded ${METHOD_COLORS[ep.method] || "bg-muted text-foreground"}`}
                >
                  {ep.method}
                </span>
                <code className="text-xs font-mono flex-1 truncate">
                  {ep.path}
                </code>
                {ep.stream && (
                  <span className="text-[9px] px-1 py-0.5 rounded bg-violet-500/15 text-violet-400">
                    {t('common.stream')}
                  </span>
                )}
                <span className="text-[10px] text-muted-foreground hidden sm:inline">
                  {epKey ? t(epKey) : ep.description}
                </span>
              </div>
              );
            })}
          </div>
        </div>
        );
      })}
    </div>
  );
}
