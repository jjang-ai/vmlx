import { ipcMain } from 'electron'
import { randomUUID } from 'crypto'
import { db } from '../database'
import { resolveUrl, connectHost } from '../sessions'
import { getAuthHeaders } from './utils'

/**
 * Benchmark IPC handlers.
 * Sends test prompts to the running session and measures TTFT, TPS, and throughput.
 */

interface BenchPrompt {
  label: string
  messages: Array<{ role: string; content: string }>
  maxTokens: number
}

const BENCH_PROMPTS: BenchPrompt[] = [
  {
    label: 'Short generation',
    messages: [
      { role: 'user', content: 'Write a haiku about silicon.' }
    ],
    maxTokens: 64
  },
  {
    label: 'Medium generation',
    messages: [
      { role: 'user', content: 'Explain how a transformer neural network processes a sentence, step by step.' }
    ],
    maxTokens: 256
  },
  {
    label: 'Long generation',
    messages: [
      { role: 'user', content: 'Write a detailed technical blog post about the advantages and challenges of running large language models on Apple Silicon. Cover memory bandwidth, unified memory architecture, and the role of quantization.' }
    ],
    maxTokens: 512
  },
  {
    label: 'Long prompt (prefill test)',
    messages: [
      { role: 'system', content: 'You are a helpful assistant that summarizes text concisely.' },
      { role: 'user', content: `Summarize the following passage in 2 sentences:\n\n${'The development of artificial intelligence has progressed through several distinct phases. In the early days, researchers focused on symbolic AI, attempting to encode human knowledge into explicit rules and logical frameworks. This approach showed promise in narrow domains but struggled with the complexity and ambiguity of real-world problems. The emergence of machine learning shifted the paradigm, allowing systems to learn patterns from data rather than following hand-crafted rules. Deep learning, powered by neural networks with many layers, further revolutionized the field by enabling automatic feature extraction from raw data. The introduction of the transformer architecture in 2017 marked another watershed moment, leading to large language models that could generate coherent text, translate languages, and answer questions with unprecedented accuracy. Today, the focus has shifted to making these models more efficient, more aligned with human values, and more accessible to a broader range of users and applications. '.repeat(3)}` }
    ],
    maxTokens: 128
  }
]

interface PromptResult {
  label: string
  ttft: number      // seconds
  tps: number       // tokens per second
  promptTokens: number
  completionTokens: number
  totalTime: number  // seconds
  ppSpeed: number    // prompt processing tokens/sec
}

async function runSingleBenchmark(
  baseUrl: string,
  prompt: BenchPrompt,
  authHeaders: Record<string, string> = {}
): Promise<PromptResult> {
  const fetchStart = Date.now()
  let firstTokenTime: number | null = null
  let tokenCount = 0
  let promptTokens = 0

  const res = await fetch(`${baseUrl}/v1/chat/completions`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json', ...authHeaders },
    body: JSON.stringify({
      model: 'default',
      messages: prompt.messages,
      max_tokens: prompt.maxTokens,
      temperature: 0.7,
      stream: true,
      stream_options: { include_usage: true }
    }),
    signal: AbortSignal.timeout(120000)
  })

  if (!res.ok) {
    throw new Error(`Benchmark request failed: ${res.status} ${res.statusText}`)
  }

  const reader = res.body?.getReader()
  if (!reader) throw new Error('No response body')

  const decoder = new TextDecoder()
  let buffer = ''

  while (true) {
    const { done, value } = await reader.read()
    if (done) break

    buffer += decoder.decode(value, { stream: true })
    const lines = buffer.split('\n')
    buffer = lines.pop() || ''

    for (const line of lines) {
      const trimmed = line.trim()
      if (!trimmed || !trimmed.startsWith('data: ')) continue
      const data = trimmed.slice(6)
      if (data === '[DONE]') continue

      try {
        const parsed = JSON.parse(data)

        // Usage info (final chunk with usage) — server count is authoritative
        let serverUsageThisChunk = false
        if (parsed.usage) {
          promptTokens = parsed.usage.prompt_tokens || promptTokens
          if (parsed.usage.completion_tokens != null) {
            tokenCount = parsed.usage.completion_tokens
            serverUsageThisChunk = true
          }
        }

        // Content delta — only client-count when server didn't provide usage this chunk
        const delta = parsed.choices?.[0]?.delta
        if (delta?.content) {
          if (!firstTokenTime) firstTokenTime = Date.now()
          if (!serverUsageThisChunk) tokenCount++
        }
      } catch { /* ignore parse errors */ }
    }
  }

  const totalTime = (Date.now() - fetchStart) / 1000
  const ttft = firstTokenTime ? (firstTokenTime - fetchStart) / 1000 : totalTime
  const generationTime = firstTokenTime ? (Date.now() - firstTokenTime) / 1000 : totalTime
  const tps = generationTime > 0.01 ? tokenCount / generationTime : 0
  const ppSpeed = ttft > 0.001 && promptTokens > 0 ? promptTokens / ttft : 0

  return {
    label: prompt.label,
    ttft,
    tps,
    promptTokens,
    completionTokens: tokenCount,
    totalTime,
    ppSpeed
  }
}

export function registerBenchmarkHandlers(getWindow: () => Electron.BrowserWindow | null): void {
  ipcMain.handle('benchmark:run', async (_, sessionId: string, endpoint: { host: string; port: number }, modelPath: string, modelName?: string, options?: { flushCache?: boolean }) => {
    const baseUrl = await resolveUrl(`http://${connectHost(endpoint.host)}:${endpoint.port}`)
    const authHeaders = getAuthHeaders(sessionId)
    const results: PromptResult[] = []
    const win = getWindow()

    // Optionally flush prefix cache before benchmark to get clean results
    if (options?.flushCache) {
      try {
        const cacheRes = await fetch(`${baseUrl}/v1/cache`, {
          method: 'DELETE',
          headers: authHeaders,
          signal: AbortSignal.timeout(10000)
        })
        if (cacheRes.ok) {
          console.log('[BENCHMARK] Prefix cache flushed before benchmark run')
        }
      } catch (err: any) {
        console.warn('[BENCHMARK] Cache flush failed (non-fatal):', err.message)
      }
    }

    for (let i = 0; i < BENCH_PROMPTS.length; i++) {
      const prompt = BENCH_PROMPTS[i]

      // Notify progress
      if (win && !win.isDestroyed()) {
        win.webContents.send('benchmark:progress', {
          sessionId,
          current: i + 1,
          total: BENCH_PROMPTS.length,
          label: prompt.label
        })
      }

      try {
        const result = await runSingleBenchmark(baseUrl, prompt, authHeaders)
        results.push(result)
      } catch (err: any) {
        results.push({
          label: prompt.label,
          ttft: 0,
          tps: 0,
          promptTokens: 0,
          completionTokens: 0,
          totalTime: 0,
          ppSpeed: 0
        })
        console.error(`[BENCHMARK] Prompt "${prompt.label}" failed:`, err.message)
      }
    }

    // Save to database
    const benchmark = {
      id: randomUUID(),
      sessionId,
      modelPath,
      modelName,
      resultsJson: JSON.stringify(results),
      createdAt: Date.now()
    }
    db.saveBenchmark(benchmark)

    return { id: benchmark.id, results, createdAt: benchmark.createdAt }
  })

  ipcMain.handle('benchmark:history', async (_, modelPath?: string) => {
    const benchmarks = db.getBenchmarks(modelPath)
    return benchmarks.map(b => ({
      id: b.id,
      sessionId: b.sessionId,
      modelPath: b.modelPath,
      modelName: b.modelName,
      results: JSON.parse(b.resultsJson),
      createdAt: b.createdAt
    }))
  })

  ipcMain.handle('benchmark:delete', async (_, id: string) => {
    db.deleteBenchmark(id)
    return { success: true }
  })
}
