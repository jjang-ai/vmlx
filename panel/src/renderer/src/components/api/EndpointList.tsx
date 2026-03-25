import type { ApiFormat } from './ApiDashboard'

interface Endpoint {
  method: 'GET' | 'POST' | 'DELETE'
  path: string
  description: string
  stream?: boolean
  auth?: boolean
}

interface EndpointGroup {
  category: string
  description: string
  endpoints: Endpoint[]
}

const OPENAI_GROUPS: EndpointGroup[] = [
  {
    category: 'Chat & Completions',
    description: 'OpenAI-compatible chat and text completion endpoints',
    endpoints: [
      { method: 'POST', path: '/v1/chat/completions', description: 'Chat completions (OpenAI format)', stream: true, auth: true },
      { method: 'POST', path: '/v1/responses', description: 'Responses API (OpenAI format)', stream: true, auth: true },
      { method: 'POST', path: '/v1/completions', description: 'Text completions (legacy format)', stream: true, auth: true },
    ],
  },
  {
    category: 'Models',
    description: 'Model information and management',
    endpoints: [
      { method: 'GET', path: '/v1/models', description: 'List loaded models', auth: true },
    ],
  },
  {
    category: 'Image Generation',
    description: 'Text-to-image generation and image editing (requires mflux)',
    endpoints: [
      { method: 'POST', path: '/v1/images/generations', description: 'Generate images from text prompts', auth: true },
      { method: 'POST', path: '/v1/images/edits', description: 'Edit images with text instructions', auth: true },
    ],
  },
  {
    category: 'Embeddings & Reranking',
    description: 'Vector embeddings and reranking (requires embedding model)',
    endpoints: [
      { method: 'POST', path: '/v1/embeddings', description: 'Generate text embeddings', auth: true },
      { method: 'POST', path: '/v1/rerank', description: 'Rerank documents by relevance', auth: true },
    ],
  },
  {
    category: 'Audio',
    description: 'Speech-to-text and text-to-speech (requires mlx-audio)',
    endpoints: [
      { method: 'POST', path: '/v1/audio/transcriptions', description: 'Transcribe audio (Whisper STT)', auth: true },
      { method: 'POST', path: '/v1/audio/speech', description: 'Generate speech (Kokoro TTS)', stream: true, auth: true },
      { method: 'GET', path: '/v1/audio/voices', description: 'List available TTS voices', auth: true },
    ],
  },
  {
    category: 'Cache',
    description: 'KV cache management for prefix caching',
    endpoints: [
      { method: 'GET', path: '/v1/cache/stats', description: 'Cache statistics (hit rate, size)', auth: true },
      { method: 'GET', path: '/v1/cache/entries', description: 'List cached prefix entries', auth: true },
      { method: 'POST', path: '/v1/cache/warm', description: 'Pre-warm cache with a prompt', auth: true },
      { method: 'DELETE', path: '/v1/cache', description: 'Clear all cached entries', auth: true },
    ],
  },
  {
    category: 'MCP Tools',
    description: 'Model Context Protocol tool integration',
    endpoints: [
      { method: 'GET', path: '/v1/mcp/tools', description: 'List available MCP tools', auth: true },
      { method: 'GET', path: '/v1/mcp/servers', description: 'MCP server connection status', auth: true },
      { method: 'POST', path: '/v1/mcp/execute', description: 'Execute an MCP tool', auth: true },
    ],
  },
  {
    category: 'Cancel',
    description: 'Cancel in-flight streaming requests',
    endpoints: [
      { method: 'POST', path: '/v1/chat/completions/{id}/cancel', description: 'Cancel a chat completion stream' },
      { method: 'POST', path: '/v1/responses/{id}/cancel', description: 'Cancel a responses stream' },
      { method: 'POST', path: '/v1/completions/{id}/cancel', description: 'Cancel a text completion stream' },
    ],
  },
  {
    category: 'Health',
    description: 'Server health and status',
    endpoints: [
      { method: 'GET', path: '/health', description: 'Health check — model status, memory, MCP info' },
    ],
  },
]

const ANTHROPIC_GROUPS: EndpointGroup[] = [
  {
    category: 'Messages API',
    description: 'Anthropic Messages API — use with Claude Code, Anthropic SDK, or any Anthropic-compatible client',
    endpoints: [
      { method: 'POST', path: '/v1/messages', description: 'Create a message (streaming or non-streaming)', stream: true, auth: true },
    ],
  },
  {
    category: 'Models',
    description: 'Model information',
    endpoints: [
      { method: 'GET', path: '/v1/models', description: 'List loaded models', auth: true },
    ],
  },
  {
    category: 'Health',
    description: 'Server health and status',
    endpoints: [
      { method: 'GET', path: '/health', description: 'Health check' },
    ],
  },
]

const OLLAMA_GROUPS: EndpointGroup[] = [
  {
    category: 'Chat & Generate',
    description: 'Ollama-compatible endpoints — use with ollama CLI or any Ollama client. Responses use NDJSON streaming.',
    endpoints: [
      { method: 'POST', path: '/api/chat', description: 'Chat completion (NDJSON streaming)', stream: true },
      { method: 'POST', path: '/api/generate', description: 'Text generation (NDJSON streaming)', stream: true },
    ],
  },
  {
    category: 'Models & Info',
    description: 'Model information and embeddings',
    endpoints: [
      { method: 'GET', path: '/api/tags', description: 'List loaded models' },
      { method: 'POST', path: '/api/show', description: 'Show model details' },
      { method: 'POST', path: '/api/embeddings', description: 'Generate embeddings' },
      { method: 'POST', path: '/api/embed', description: 'Generate embeddings (alias)' },
    ],
  },
  {
    category: 'Liveness',
    description: 'Ollama compatibility check',
    endpoints: [
      { method: 'GET', path: '/', description: 'Liveness check — returns "vMLX Gateway is running"' },
    ],
  },
]

const FORMAT_GROUPS: Record<ApiFormat, EndpointGroup[]> = {
  openai: OPENAI_GROUPS,
  anthropic: ANTHROPIC_GROUPS,
  ollama: OLLAMA_GROUPS,
}

const METHOD_COLORS: Record<string, string> = {
  GET: 'bg-blue-500/15 text-blue-500',
  POST: 'bg-green-500/15 text-green-500',
  DELETE: 'bg-red-500/15 text-red-500',
}

interface EndpointListProps {
  format?: ApiFormat
  isImage?: boolean
  isEdit?: boolean
}

export function EndpointList({ format = 'openai', isImage, isEdit }: EndpointListProps) {
  const groups = FORMAT_GROUPS[format] || OPENAI_GROUPS

  return (
    <div className="space-y-4">
      <h3 className="text-sm font-medium">
        {format === 'openai' ? 'OpenAI Endpoints' : format === 'anthropic' ? 'Anthropic Endpoints' : 'Ollama Endpoints'}
      </h3>
      {groups.map(group => (
        <div key={group.category} className="border border-border rounded-lg overflow-hidden">
          <div className="px-3 py-2 bg-muted/50 border-b border-border">
            <h4 className="text-xs font-medium">{group.category}</h4>
            <p className="text-[10px] text-muted-foreground">{group.description}</p>
          </div>
          <div className="divide-y divide-border">
            {group.endpoints.map(ep => (
              <div key={ep.path} className="px-3 py-2 flex items-center gap-2">
                <span className={`text-[10px] font-mono font-bold px-1.5 py-0.5 rounded ${METHOD_COLORS[ep.method] || 'bg-muted text-foreground'}`}>
                  {ep.method}
                </span>
                <code className="text-xs font-mono flex-1 truncate">{ep.path}</code>
                {ep.stream && (
                  <span className="text-[9px] px-1 py-0.5 rounded bg-violet-500/15 text-violet-400">stream</span>
                )}
                <span className="text-[10px] text-muted-foreground hidden sm:inline">{ep.description}</span>
              </div>
            ))}
          </div>
        </div>
      ))}
    </div>
  )
}
