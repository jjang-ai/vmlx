import { useState, useMemo } from 'react'
import { Copy, Check } from 'lucide-react'

type Lang = 'curl' | 'python-openai' | 'python-anthropic' | 'javascript'

const LANGS: { key: Lang; label: string }[] = [
  { key: 'curl', label: 'curl' },
  { key: 'python-openai', label: 'Python (OpenAI)' },
  { key: 'python-anthropic', label: 'Python (Anthropic)' },
  { key: 'javascript', label: 'JavaScript' },
]

interface CodeSnippetsProps {
  baseUrl: string
  apiKey: string | null
  modelId: string | null
  isImage?: boolean
  isEdit?: boolean
}

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

const BUILDERS: Record<Lang, (baseUrl: string, apiKey: string | null, model: string) => string> = {
  'curl': buildCurl,
  'python-openai': buildPythonOpenAI,
  'python-anthropic': buildPythonAnthropic,
  'javascript': buildJavaScript,
}

// Image-specific snippets
function buildImageCurl(baseUrl: string, apiKey: string | null, model: string): string {
  const authHeader = apiKey ? `\n  -H "Authorization: Bearer ${apiKey}" \\` : ''
  return `curl ${baseUrl}/v1/images/generations \\
  -H "Content-Type: application/json" \\${authHeader}
  -d '{
    "model": "${model}",
    "prompt": "A cat astronaut floating in space",
    "size": "1024x1024",
    "steps": 4,
    "guidance": 3.5
  }'`
}

function buildImagePython(baseUrl: string, apiKey: string | null, model: string): string {
  const key = apiKey ? `"${apiKey}"` : '"not-needed"'
  return `from openai import OpenAI
import base64

client = OpenAI(
    base_url="${baseUrl}/v1",
    api_key=${key},
)

response = client.images.generate(
    model="${model}",
    prompt="A cat astronaut floating in space",
    size="1024x1024",
    n=1,
    response_format="b64_json",
)

# Save the image
image_data = base64.b64decode(response.data[0].b64_json)
with open("output.png", "wb") as f:
    f.write(image_data)
print("Saved output.png")`
}

// Image edit snippets
function buildImageEditCurl(baseUrl: string, apiKey: string | null, model: string): string {
  const authHeader = apiKey ? `\n  -H "Authorization: Bearer ${apiKey}" \\` : ''
  return `curl ${baseUrl}/v1/images/edits \\
  -H "Content-Type: application/json" \\${authHeader}
  -d '{
    "model": "${model}",
    "prompt": "Make the sky sunset orange",
    "image": "<base64-encoded-image>",
    "size": "1024x1024",
    "steps": 24,
    "guidance": 3.5,
    "strength": 0.8
  }'`
}

function buildImageEditPython(baseUrl: string, apiKey: string | null, model: string): string {
  return `import requests, base64

# Read source image
with open("input.jpg", "rb") as f:
    image_b64 = base64.b64encode(f.read()).decode()

response = requests.post(
    "${baseUrl}/v1/images/edits",
    headers={
        "Content-Type": "application/json",
        ${apiKey ? `"Authorization": "Bearer ${apiKey}",` : ''}
    },
    json={
        "model": "${model}",
        "prompt": "Make the sky sunset orange",
        "image": image_b64,
        "size": "1024x1024",
        "steps": 24,
        "guidance": 3.5,
        "strength": 0.8,
    },
)

result = response.json()
image_data = base64.b64decode(result["data"][0]["b64_json"])
with open("edited.png", "wb") as f:
    f.write(image_data)
print("Saved edited.png")`
}

const IMAGE_BUILDERS: Record<string, (baseUrl: string, apiKey: string | null, model: string) => string> = {
  'curl': buildImageCurl,
  'python-openai': buildImagePython,
}

const IMAGE_EDIT_BUILDERS: Record<string, (baseUrl: string, apiKey: string | null, model: string) => string> = {
  'curl': buildImageEditCurl,
  'python-openai': buildImageEditPython,
}

const IMAGE_LANGS: { key: Lang; label: string }[] = [
  { key: 'curl', label: 'curl' },
  { key: 'python-openai', label: 'Python' },
]

export function CodeSnippets({ baseUrl, apiKey, modelId, isImage, isEdit }: CodeSnippetsProps) {
  const availableLangs = isImage ? IMAGE_LANGS : LANGS
  const [lang, setLang] = useState<Lang>(availableLangs[0].key)
  const [copied, setCopied] = useState(false)

  // Reset lang when switching between image/text servers to avoid stale tab
  const validKeys = availableLangs.map(l => l.key)
  if (!validKeys.includes(lang)) {
    setLang(availableLangs[0].key)
  }

  const model = modelId || 'your-model-name'
  const builders = isImage ? (isEdit ? IMAGE_EDIT_BUILDERS : IMAGE_BUILDERS) : BUILDERS
  const snippet = useMemo(() => (builders[lang] || builders[availableLangs[0].key])(baseUrl, apiKey, model), [lang, baseUrl, apiKey, model, isImage, isEdit])

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
