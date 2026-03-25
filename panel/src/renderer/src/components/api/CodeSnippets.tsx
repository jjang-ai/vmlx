import { useState, useMemo } from 'react'
import { Copy, Check } from 'lucide-react'
import type { ApiFormat } from './ApiDashboard'

type Lang = 'curl' | 'python-openai' | 'python-anthropic' | 'python-requests' | 'javascript' | 'ollama-cli'

const OPENAI_LANGS: { key: Lang; label: string }[] = [
  { key: 'curl', label: 'curl' },
  { key: 'python-openai', label: 'Python (OpenAI)' },
  { key: 'javascript', label: 'JavaScript' },
]

const ANTHROPIC_LANGS: { key: Lang; label: string }[] = [
  { key: 'curl', label: 'curl' },
  { key: 'python-anthropic', label: 'Python (Anthropic)' },
]

const OLLAMA_LANGS: { key: Lang; label: string }[] = [
  { key: 'ollama-cli', label: 'CLI' },
  { key: 'curl', label: 'curl' },
  { key: 'python-requests', label: 'Python' },
]

interface CodeSnippetsProps {
  baseUrl: string
  apiKey: string | null
  modelId: string | null
  isImage?: boolean
  isEdit?: boolean
  format?: ApiFormat
}

// ── OpenAI snippets ──

function buildCurl(baseUrl: string, apiKey: string | null, model: string): string {
  const authHeader = apiKey ? `\n  -H "Authorization: Bearer ${apiKey}" \\` : ''
  return `curl ${baseUrl}/v1/chat/completions \\
  -H "Content-Type: application/json" \\${authHeader}
  -d '{
    "model": "${model}",
    "messages": [
      {"role": "user", "content": "Hello!"}
    ],
    "stream": true
  }'`
}

function buildPythonOpenAI(baseUrl: string, apiKey: string | null, model: string): string {
  const key = apiKey ? `"${apiKey}"` : '"not-needed"'
  return `from openai import OpenAI

client = OpenAI(
    base_url="${baseUrl}/v1",
    api_key=${key},
)

response = client.chat.completions.create(
    model="${model}",
    messages=[
        {"role": "user", "content": "Hello!"}
    ],
    stream=True,
)

for chunk in response:
    if chunk.choices[0].delta.content:
        print(chunk.choices[0].delta.content, end="", flush=True)
print()`
}

function buildJavaScript(baseUrl: string, apiKey: string | null, model: string): string {
  const key = apiKey ? `"${apiKey}"` : '"not-needed"'
  return `import OpenAI from "openai";

const client = new OpenAI({
  baseURL: "${baseUrl}/v1",
  apiKey: ${key},
});

const stream = await client.chat.completions.create({
  model: "${model}",
  messages: [
    { role: "user", content: "Hello!" }
  ],
  stream: true,
});

for await (const chunk of stream) {
  const content = chunk.choices[0]?.delta?.content;
  if (content) process.stdout.write(content);
}
console.log();`
}

// ── Anthropic snippets ──

function buildAnthropicCurl(baseUrl: string, apiKey: string | null, model: string): string {
  const key = apiKey || 'not-needed'
  return `curl ${baseUrl}/v1/messages \\
  -H "Content-Type: application/json" \\
  -H "x-api-key: ${key}" \\
  -H "anthropic-version: 2023-06-01" \\
  -d '{
    "model": "${model}",
    "max_tokens": 1024,
    "messages": [
      {"role": "user", "content": "Hello!"}
    ]
  }'`
}

function buildPythonAnthropic(baseUrl: string, apiKey: string | null, model: string): string {
  const key = apiKey ? `"${apiKey}"` : '"not-needed"'
  return `import anthropic

client = anthropic.Anthropic(
    base_url="${baseUrl}",
    api_key=${key},
)

# The Anthropic SDK appends /v1/messages automatically
message = client.messages.create(
    model="${model}",
    max_tokens=1024,
    messages=[
        {"role": "user", "content": "Hello!"}
    ],
)

print(message.content[0].text)`
}

// ── Ollama snippets ──

function buildOllamaCli(baseUrl: string, _apiKey: string | null, model: string): string {
  return `# Set the Ollama host to your vMLX gateway
export OLLAMA_HOST=${baseUrl}

# Chat with a model
ollama run ${model}

# Or use ollama API directly
ollama list`
}

function buildOllamaCurl(baseUrl: string, _apiKey: string | null, model: string): string {
  return `# Streaming chat (NDJSON)
curl ${baseUrl}/api/chat -d '{
  "model": "${model}",
  "messages": [
    {"role": "user", "content": "Hello!"}
  ]
}'

# Text generation
curl ${baseUrl}/api/generate -d '{
  "model": "${model}",
  "prompt": "Hello!"
}'

# List models
curl ${baseUrl}/api/tags`
}

function buildOllamaPython(baseUrl: string, _apiKey: string | null, model: string): string {
  return `import requests, json

# Streaming chat via Ollama API
response = requests.post(
    "${baseUrl}/api/chat",
    json={
        "model": "${model}",
        "messages": [
            {"role": "user", "content": "Hello!"}
        ],
    },
    stream=True,
)

for line in response.iter_lines():
    if line:
        chunk = json.loads(line)
        print(chunk.get("message", {}).get("content", ""), end="", flush=True)
        if chunk.get("done"):
            break
print()`
}

// ── Builder maps ──

const OPENAI_BUILDERS: Record<string, (b: string, k: string | null, m: string) => string> = {
  'curl': buildCurl,
  'python-openai': buildPythonOpenAI,
  'javascript': buildJavaScript,
}

const ANTHROPIC_BUILDERS: Record<string, (b: string, k: string | null, m: string) => string> = {
  'curl': buildAnthropicCurl,
  'python-anthropic': buildPythonAnthropic,
}

const OLLAMA_BUILDERS: Record<string, (b: string, k: string | null, m: string) => string> = {
  'ollama-cli': buildOllamaCli,
  'curl': buildOllamaCurl,
  'python-requests': buildOllamaPython,
}

const FORMAT_LANGS: Record<ApiFormat, { key: Lang; label: string }[]> = {
  openai: OPENAI_LANGS,
  anthropic: ANTHROPIC_LANGS,
  ollama: OLLAMA_LANGS,
}

const FORMAT_BUILDERS: Record<ApiFormat, Record<string, (b: string, k: string | null, m: string) => string>> = {
  openai: OPENAI_BUILDERS,
  anthropic: ANTHROPIC_BUILDERS,
  ollama: OLLAMA_BUILDERS,
}

export function CodeSnippets({ baseUrl, apiKey, modelId, format = 'openai' }: CodeSnippetsProps) {
  const availableLangs = FORMAT_LANGS[format]
  const [lang, setLang] = useState<Lang>(availableLangs[0].key)
  const [copied, setCopied] = useState(false)

  // Reset lang when format changes
  const validKeys = availableLangs.map(l => l.key)
  if (!validKeys.includes(lang)) {
    setLang(availableLangs[0].key)
  }

  const model = modelId || 'your-model-name'
  const builders = FORMAT_BUILDERS[format]
  const snippet = useMemo(
    () => (builders[lang] || builders[availableLangs[0].key])(baseUrl, apiKey, model),
    [lang, baseUrl, apiKey, model, format]
  )

  const handleCopy = () => {
    navigator.clipboard.writeText(snippet)
    setCopied(true)
    setTimeout(() => setCopied(false), 2000)
  }

  return (
    <div className="space-y-2">
      <div className="flex items-center justify-between">
        <h3 className="text-sm font-medium">Quick Start</h3>
        <div className="flex gap-1">
          {availableLangs.map(l => (
            <button
              key={l.key}
              onClick={() => setLang(l.key)}
              className={`px-2 py-1 text-[10px] rounded transition-colors ${
                lang === l.key
                  ? 'bg-primary/15 text-primary font-medium'
                  : 'text-muted-foreground hover:bg-accent'
              }`}
            >
              {l.label}
            </button>
          ))}
        </div>
      </div>
      <div className="relative">
        <pre className="p-4 rounded-lg border border-border bg-background text-xs font-mono overflow-x-auto whitespace-pre leading-relaxed">
          {snippet}
        </pre>
        <button
          onClick={handleCopy}
          className="absolute top-2 right-2 p-1.5 rounded bg-muted/80 hover:bg-muted text-muted-foreground hover:text-foreground transition-colors"
          title="Copy"
        >
          {copied ? <Check className="h-3.5 w-3.5 text-green-500" /> : <Copy className="h-3.5 w-3.5" />}
        </button>
      </div>
    </div>
  )
}
