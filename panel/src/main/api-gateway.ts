// SPDX-License-Identifier: Apache-2.0
// API Gateway — single-port reverse proxy for all model sessions.
// Routes OpenAI, Anthropic, and Ollama requests to the correct backend
// by model name. Supports JIT auto-load for stopped models.
//
// Architecture:
//   Client → [Gateway :8080] → route by model field → [Session A :52431]
//                                                    → [Session B :52432]
//   Ollama /api/* endpoints translated to OpenAI format before forwarding.

import { createServer, IncomingMessage, ServerResponse, request as httpRequest, Server } from 'http'
import { db } from './database'
import { sessionManager } from './sessions'
import { EventEmitter } from 'events'

const DEFAULT_PORT = 8080
const JIT_TIMEOUT_MS = 120_000
const HEALTH_POLL_MS = 2_000
const PROXY_TIMEOUT_MS = 300_000  // 5 min max for a single proxied request

interface ResolvedSession {
  id: string
  host: string
  port: number
  status: string
  modelName: string
  modelPath: string
  servedModelName?: string
}

export class ApiGateway extends EventEmitter {
  private server: Server | null = null
  private port: number = DEFAULT_PORT
  private _running = false
  /** Track in-flight JIT loads to prevent duplicate starts */
  private jitPending = new Map<string, Promise<boolean>>()

  get running(): boolean { return this._running }
  get activePort(): number { return this.port }

  // ═══════════════════════════════════════════════════════════════
  // Lifecycle
  // ═══════════════════════════════════════════════════════════════

  async start(port?: number): Promise<void> {
    if (this._running) return
    this.port = port ?? parseInt(db.getSetting('gateway_port') || String(DEFAULT_PORT), 10)

    const maxRetries = 10
    for (let attempt = 0; attempt < maxRetries; attempt++) {
      try {
        await this._tryListen(this.port)
        return
      } catch (err: any) {
        if (err.code === 'EADDRINUSE' && attempt < maxRetries - 1) {
          const nextPort = this.port + 1
          console.warn(`[gateway] Port ${this.port} in use, trying ${nextPort}`)
          this.port = nextPort
        } else {
          throw err
        }
      }
    }
  }

  private _tryListen(port: number): Promise<void> {
    return new Promise((resolve, reject) => {
      this.server = createServer((req, res) => {
        this.handleRequest(req, res).catch(err => {
          console.error('[gateway] Unhandled request error:', err)
          if (!res.headersSent) {
            this.sendJson(res, 500, { error: { message: 'Internal gateway error', type: 'server_error' } })
          }
        })
      })

      this.server.on('error', (err: NodeJS.ErrnoException) => {
        reject(err)
      })

      this.server.listen(port, '0.0.0.0', () => {
        this._running = true
        db.setSetting('gateway_port', String(port))
        console.log(`[gateway] Listening on 0.0.0.0:${port}`)
        this.emit('started', port)
        resolve()
      })
    })
  }

  async stop(): Promise<void> {
    if (!this.server) return
    return new Promise(resolve => {
      this.server!.close(() => {
        this._running = false
        this.server = null
        this.jitPending.clear()
        console.log('[gateway] Stopped')
        this.emit('stopped')
        resolve()
      })
    })
  }

  async restart(port: number): Promise<void> {
    await this.stop()
    await this.start(port)
  }

  // ═══════════════════════════════════════════════════════════════
  // Request Router
  // ═══════════════════════════════════════════════════════════════

  private async handleRequest(req: IncomingMessage, res: ServerResponse): Promise<void> {
    const url = req.url || '/'
    const method = req.method || 'GET'

    // ── CORS preflight (Open WebUI, browser clients) ──
    if (method === 'OPTIONS') {
      res.writeHead(204, {
        'Access-Control-Allow-Origin': '*',
        'Access-Control-Allow-Methods': 'GET, POST, PUT, DELETE, OPTIONS, HEAD',
        'Access-Control-Allow-Headers': 'Content-Type, Authorization, X-Requested-With',
        'Access-Control-Max-Age': '86400',
      })
      res.end()
      return
    }

    // ── Ollama liveness check (HEAD / or GET /) ──
    if (url === '/' && (method === 'HEAD' || method === 'GET')) {
      res.writeHead(200, { 'Content-Type': 'text/plain' })
      if (method === 'GET') res.write('vMLX Gateway is running')
      res.end()
      return
    }

    // ── Gateway meta endpoints (no proxy needed) ──
    if (url === '/health' && method === 'GET') return this.handleHealth(res)
    if (url === '/v1/models' && method === 'GET') return this.handleListModels(res)

    // ── Ollama endpoints ──
    if (url.startsWith('/api/')) return this.handleOllamaRoute(req, res, url, method)

    // ── All other routes: read body, resolve model, proxy ──
    const body = await this.readBody(req)
    let modelName: string | undefined

    // Extract model from body (POST) or query param (GET/DELETE)
    if (method === 'POST' && body) {
      try {
        const parsed = JSON.parse(body)
        modelName = parsed.model
      } catch (_) { /* not JSON — forward raw */ }
    }
    if (!modelName) {
      // Support ?model=X query parameter for GET/DELETE endpoints (cache, MCP, audio voices)
      const qIdx = url.indexOf('?')
      if (qIdx >= 0) {
        const params = new URLSearchParams(url.slice(qIdx))
        modelName = params.get('model') || undefined
      }
    }

    // Cancel requests without model field: broadcast to all running backends.
    // Only the backend holding that request ID will actually cancel.
    const isCancel = method === 'POST' && /\/cancel\/?$/.test(url)
    if (isCancel && !modelName) {
      const sessions = db.getSessions().filter((s: any) => s.status === 'running')
      if (sessions.length === 0) return this.sendJson(res, 404, { error: 'No running models' })
      let accepted = false
      for (const s of sessions) {
        const host = s.host === '0.0.0.0' ? '127.0.0.1' : s.host
        try {
          const cancelRes = await new Promise<number>((resolve) => {
            const cancelReq = httpRequest({ hostname: host, port: s.port, path: url, method: 'POST', headers: { 'Content-Type': 'application/json' }, timeout: 5000 }, (r) => {
              r.resume()
              resolve(r.statusCode || 500)
            })
            cancelReq.on('error', () => resolve(500))
            cancelReq.on('timeout', () => { cancelReq.destroy(); resolve(500) })
            if (body) cancelReq.write(body)
            cancelReq.end()
          })
          if (cancelRes >= 200 && cancelRes < 300) accepted = true
        } catch {}
      }
      return this.sendJson(res, accepted ? 200 : 404, accepted ? { status: 'cancelled' } : { error: 'Request ID not found on any backend' })
    }

    const session = this.resolveSession(modelName)
    if (!session) {
      const available = this.getAvailableModelNames()
      return this.sendJson(res, 404, {
        error: {
          message: `Model '${modelName || 'unknown'}' not found. Available: [${available.join(', ')}]`,
          type: 'invalid_request_error',
          code: 'model_not_found',
        }
      })
    }

    // JIT auto-load if not running
    if (session.status !== 'running') {
      const ok = await this.jitLoad(session.id)
      if (!ok) {
        return this.sendJson(res, 503, {
          error: {
            message: `Model '${session.modelName}' failed to load within ${JIT_TIMEOUT_MS / 1000}s`,
            type: 'server_error',
            code: 'model_load_timeout',
          },
          retry_after: 30,
        })
      }
      // Re-read session to get updated port (may have changed on restart)
      const fresh = db.getSession(session.id)
      if (fresh) {
        session.port = fresh.port
        session.host = fresh.host === '0.0.0.0' ? '127.0.0.1' : fresh.host
        session.status = fresh.status
      }
    }

    // Touch session to prevent idle sleep
    sessionManager.touchSession(session.id)

    return this.proxyRequest(req, res, session, body)
  }

  // ═══════════════════════════════════════════════════════════════
  // Model Resolution
  // ═══════════════════════════════════════════════════════════════

  private resolveSession(modelName?: string): ResolvedSession | undefined {
    const sessions = db.getSessions()
    if (!sessions.length) return undefined

    // Filter out remote sessions — they proxy through renderer, not this gateway
    const localSessions = sessions.filter(s => s.type !== 'remote')
    if (!localSessions.length) return undefined

    const candidates: ResolvedSession[] = localSessions.map(s => {
      let config: any = {}
      try { config = JSON.parse(s.config || '{}') } catch (_) {}
      return {
        id: s.id,
        host: s.host === '0.0.0.0' ? '127.0.0.1' : s.host,
        port: s.port,
        status: s.status,
        modelName: s.modelName || s.modelPath.split('/').pop() || '',
        modelPath: s.modelPath,
        servedModelName: config.servedModelName || undefined,
      }
    })

    // No model specified — prefer running session, then first available
    if (!modelName) {
      return candidates.find(c => c.status === 'running')
        || candidates.find(c => c.status === 'standby')
        || candidates[0]
    }

    const lower = modelName.toLowerCase()
    // Strip Ollama :tag suffix (e.g., "qwen3.5:latest" → "qwen3.5")
    const baseName = lower.split(':')[0]

    // 1. Exact match on servedModelName (user alias — highest priority)
    const byAlias = candidates.find(c => c.servedModelName === modelName)
    if (byAlias) return byAlias

    // 2. Exact match on servedModelName (case-insensitive)
    const byAliasCI = candidates.find(c => c.servedModelName?.toLowerCase() === lower)
    if (byAliasCI) return byAliasCI

    // 3. Exact match on modelName (basename of path)
    const byName = candidates.find(c => c.modelName === modelName)
    if (byName) return byName

    // 4. Exact match on full modelPath
    const byPath = candidates.find(c => c.modelPath === modelName)
    if (byPath) return byPath

    // 5. Case-insensitive modelName match
    const byNameCI = candidates.find(c => c.modelName.toLowerCase() === lower)
    if (byNameCI) return byNameCI

    // 6. Partial — model name contains query or vice versa
    const byPartial = candidates.find(c =>
      c.modelName.toLowerCase().includes(baseName) || baseName.includes(c.modelName.toLowerCase())
    )
    if (byPartial) return byPartial

    // 7. Partial on servedModelName
    const byAliasPartial = candidates.find(c =>
      c.servedModelName && (c.servedModelName.toLowerCase().includes(baseName) || baseName.includes(c.servedModelName.toLowerCase()))
    )
    if (byAliasPartial) return byAliasPartial

    // 8. Single-model fallback — if only one session exists, route to it
    if (candidates.length === 1) return candidates[0]

    return undefined
  }

  private getAvailableModelNames(): string[] {
    const sessions = db.getSessions()
    const names: string[] = []
    for (const s of sessions) {
      let config: any = {}
      try { config = JSON.parse(s.config || '{}') } catch (_) {}
      names.push(config.servedModelName || s.modelName || s.modelPath.split('/').pop() || 'unknown')
    }
    return names
  }

  // ═══════════════════════════════════════════════════════════════
  // JIT Auto-Load
  // ═══════════════════════════════════════════════════════════════

  private async jitLoad(sessionId: string): Promise<boolean> {
    // Deduplicate concurrent JIT loads for the same session
    const existing = this.jitPending.get(sessionId)
    if (existing) return existing

    const promise = this._doJitLoad(sessionId)
    this.jitPending.set(sessionId, promise)
    try {
      return await promise
    } finally {
      this.jitPending.delete(sessionId)
    }
  }

  private async _doJitLoad(sessionId: string): Promise<boolean> {
    console.log(`[gateway] JIT loading session ${sessionId}`)
    try {
      await sessionManager.startSession(sessionId)
    } catch (err) {
      console.error(`[gateway] Failed to start session ${sessionId}: ${err}`)
      return false
    }

    const deadline = Date.now() + JIT_TIMEOUT_MS
    while (Date.now() < deadline) {
      const s = db.getSession(sessionId)
      if (s?.status === 'running') return true
      if (s?.status === 'error') return false
      await new Promise(r => setTimeout(r, HEALTH_POLL_MS))
    }
    console.error(`[gateway] JIT load timeout for session ${sessionId}`)
    return false
  }

  // ═══════════════════════════════════════════════════════════════
  // Reverse Proxy
  // ═══════════════════════════════════════════════════════════════

  private proxyRequest(
    clientReq: IncomingMessage,
    clientRes: ServerResponse,
    session: ResolvedSession,
    body: string
  ): void {
    const options = {
      hostname: session.host,
      port: session.port,
      path: clientReq.url,
      method: clientReq.method,
      headers: {
        ...clientReq.headers,
        host: `${session.host}:${session.port}`,
      },
      timeout: PROXY_TIMEOUT_MS,
    }

    const proxyReq = httpRequest(options, (proxyRes) => {
      // Forward status + headers verbatim (preserves SSE Content-Type)
      clientRes.writeHead(proxyRes.statusCode || 502, proxyRes.headers)
      // Pipe response directly — works for SSE streaming and regular JSON
      proxyRes.pipe(clientRes)
    })

    proxyReq.on('error', (err) => {
      console.error(`[gateway] Proxy error → ${session.host}:${session.port}${clientReq.url}: ${err.message}`)
      if (!clientRes.headersSent) {
        this.sendJson(clientRes, 502, {
          error: { message: `Backend unavailable: ${err.message}`, type: 'server_error' }
        })
      }
    })

    proxyReq.on('timeout', () => {
      proxyReq.destroy()
      if (!clientRes.headersSent) {
        this.sendJson(clientRes, 504, {
          error: { message: 'Backend request timed out', type: 'server_error' }
        })
      }
    })

    if (body) proxyReq.write(body)
    proxyReq.end()

    // Abort backend inference when client disconnects mid-stream
    clientReq.on('close', () => { if (!proxyReq.destroyed) proxyReq.destroy() })
  }

  // ═══════════════════════════════════════════════════════════════
  // /v1/models — Aggregate all sessions
  // ═══════════════════════════════════════════════════════════════

  private handleListModels(res: ServerResponse): void {
    const sessions = db.getSessions()
    const models: any[] = []
    const seen = new Set<string>()

    for (const s of sessions) {
      let config: any = {}
      try { config = JSON.parse(s.config || '{}') } catch (_) {}

      // Primary name: alias if set, otherwise basename
      const primaryName = config.servedModelName || s.modelName || s.modelPath.split('/').pop() || 'unknown'
      if (!seen.has(primaryName)) {
        seen.add(primaryName)
        models.push({
          id: primaryName,
          object: 'model',
          created: Math.floor((s.createdAt || Date.now()) / 1000),
          owned_by: 'vmlx-engine',
        })
      }

      // Also list actual model name if alias differs
      const actualName = s.modelName || s.modelPath.split('/').pop() || ''
      if (config.servedModelName && actualName && config.servedModelName !== actualName && !seen.has(actualName)) {
        seen.add(actualName)
        models.push({
          id: actualName,
          object: 'model',
          created: Math.floor((s.createdAt || Date.now()) / 1000),
          owned_by: 'vmlx-engine',
        })
      }
    }

    this.sendJson(res, 200, { object: 'list', data: models })
  }

  // ═══════════════════════════════════════════════════════════════
  // /health — Gateway + backend status
  // ═══════════════════════════════════════════════════════════════

  private handleHealth(res: ServerResponse): void {
    const sessions = db.getSessions()
    this.sendJson(res, 200, {
      status: 'ok',
      gateway_port: this.port,
      backends: sessions.map(s => {
        let config: any = {}
        try { config = JSON.parse(s.config || '{}') } catch (_) {}
        return {
          id: s.id,
          model: config.servedModelName || s.modelName,
          status: s.status,
          port: s.port,
        }
      }),
    })
  }

  // ═══════════════════════════════════════════════════════════════
  // Ollama API Compatibility
  // ═══════════════════════════════════════════════════════════════

  private async handleOllamaRoute(req: IncomingMessage, res: ServerResponse, url: string, method: string): Promise<void> {
    // GET endpoints (no body)
    if (method === 'GET') {
      if (url === '/api/tags') return this.handleOllamaTags(res)
      if (url === '/api/ps') return this.handleOllamaPs(res)
      if (url === '/api/version') return this.sendJson(res, 200, { version: '0.6.2' })  // Compat shim
      return this.sendJson(res, 404, { error: 'Unknown endpoint' })
    }

    // POST endpoints
    if (method === 'POST') {
      if (url === '/api/chat') return this.handleOllamaChat(req, res)
      if (url === '/api/generate') return this.handleOllamaGenerate(req, res)
      if (url === '/api/show') return this.handleOllamaShow(req, res)
      if (url === '/api/embeddings' || url === '/api/embed') return this.handleOllamaEmbed(req, res)
      // Unsupported but don't error — return empty success for compat
      if (url === '/api/pull') return this.sendJson(res, 200, { status: 'success' })
      if (url === '/api/delete') return this.sendJson(res, 200, { status: 'success' })
      if (url === '/api/copy') return this.sendJson(res, 200, { status: 'success' })
      if (url === '/api/create') return this.sendJson(res, 200, { status: 'success' })
      return this.sendJson(res, 404, { error: 'Unknown endpoint' })
    }

    // HEAD for health checks from some clients
    if (method === 'HEAD' && url === '/') {
      res.writeHead(200)
      res.end()
      return
    }

    this.sendJson(res, 405, { error: 'Method not allowed' })
  }

  // ── /api/tags ──

  private handleOllamaTags(res: ServerResponse): void {
    const sessions = db.getSessions()
    const models = sessions.map(s => {
      let config: any = {}
      try { config = JSON.parse(s.config || '{}') } catch (_) {}
      return {
        name: config.servedModelName || s.modelName || 'unknown',
        model: s.modelPath,
        modified_at: new Date(s.updatedAt || Date.now()).toISOString(),
        size: 0,
        digest: '',
        details: { format: 'mlx', family: '', parameter_size: '', quantization_level: '' },
      }
    })
    this.sendJson(res, 200, { models })
  }

  // ── /api/ps ──

  private handleOllamaPs(res: ServerResponse): void {
    const sessions = db.getSessions().filter(s => s.status === 'running')
    const models = sessions.map(s => {
      let config: any = {}
      try { config = JSON.parse(s.config || '{}') } catch (_) {}
      return {
        name: config.servedModelName || s.modelName || 'unknown',
        model: s.modelPath,
        size: 0,
        digest: '',
        expires_at: new Date(Date.now() + 300_000).toISOString(),
      }
    })
    this.sendJson(res, 200, { models })
  }

  // ── /api/chat ──

  private async handleOllamaChat(req: IncomingMessage, res: ServerResponse): Promise<void> {
    const body = await this.readBody(req)
    if (!body) return this.sendJson(res, 400, { error: 'Empty request body' })

    let parsed: any
    try { parsed = JSON.parse(body) } catch (_) {
      return this.sendJson(res, 400, { error: 'Invalid JSON' })
    }

    const session = this.resolveSession(parsed.model)
    if (!session) {
      return this.sendJson(res, 404, { error: `model '${parsed.model || 'unknown'}' not found` })
    }

    if (session.status !== 'running') {
      const ok = await this.jitLoad(session.id)
      if (!ok) return this.sendJson(res, 503, { error: 'Model failed to load' })
      const fresh = db.getSession(session.id)
      if (fresh) { session.port = fresh.port; session.host = fresh.host === '0.0.0.0' ? '127.0.0.1' : fresh.host }
    }

    sessionManager.touchSession(session.id)

    // Translate Ollama → OpenAI
    const opts = parsed.options || {}
    const openaiBody: any = {
      model: parsed.model || session.modelName,
      messages: parsed.messages || [],
      stream: parsed.stream !== false,
    }
    if (opts.num_predict != null) openaiBody.max_tokens = opts.num_predict
    if (opts.temperature != null) openaiBody.temperature = opts.temperature
    if (opts.top_p != null) openaiBody.top_p = opts.top_p
    if (opts.top_k != null) openaiBody.top_k = opts.top_k
    if (opts.stop) openaiBody.stop = opts.stop
    if (opts.repeat_penalty != null) openaiBody.repetition_penalty = opts.repeat_penalty

    const isStreaming = parsed.stream !== false
    const modelForResponse = parsed.model || session.modelName

    const proxyOpts = {
      hostname: session.host,
      port: session.port,
      path: '/v1/chat/completions',
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      timeout: PROXY_TIMEOUT_MS,
    }

    const proxyReq = httpRequest(proxyOpts, (proxyRes) => {
      if (!isStreaming) {
        // Non-streaming: buffer, translate, send
        let data = ''
        proxyRes.on('data', (chunk: Buffer) => { data += chunk.toString() })
        proxyRes.on('end', () => {
          try {
            const openai = JSON.parse(data)
            this.sendJson(res, 200, {
              model: modelForResponse,
              created_at: new Date().toISOString(),
              message: { role: 'assistant', content: openai.choices?.[0]?.message?.content || '' },
              done: true,
              done_reason: openai.choices?.[0]?.finish_reason || 'stop',
              total_duration: 0,
              eval_count: openai.usage?.completion_tokens || 0,
              prompt_eval_count: openai.usage?.prompt_tokens || 0,
            })
          } catch (_) {
            this.sendJson(res, 502, { error: 'Failed to parse backend response' })
          }
        })
      } else {
        // Streaming: SSE → NDJSON
        res.writeHead(200, { 'Content-Type': 'application/x-ndjson', 'Transfer-Encoding': 'chunked' })

        let buffer = ''
        proxyRes.on('data', (chunk: Buffer) => {
          buffer += chunk.toString()
          const lines = buffer.split('\n')
          buffer = lines.pop() || ''

          for (const line of lines) {
            const trimmed = line.trim()
            if (!trimmed.startsWith('data: ')) continue
            const payload = trimmed.slice(6)

            if (payload === '[DONE]') {
              res.write(JSON.stringify({
                model: modelForResponse, created_at: new Date().toISOString(),
                message: { role: 'assistant', content: '' }, done: true, done_reason: 'stop',
              }) + '\n')
              res.end()
              return
            }

            try {
              const chunk = JSON.parse(payload)
              const content = chunk.choices?.[0]?.delta?.content || ''
              const finishReason = chunk.choices?.[0]?.finish_reason
              const done = finishReason != null

              const ollamaChunk: any = {
                model: modelForResponse,
                created_at: new Date().toISOString(),
                message: { role: 'assistant', content },
                done,
              }
              if (done) {
                ollamaChunk.done_reason = finishReason || 'stop'
                if (chunk.usage) {
                  ollamaChunk.eval_count = chunk.usage.completion_tokens || 0
                  ollamaChunk.prompt_eval_count = chunk.usage.prompt_tokens || 0
                }
              }
              res.write(JSON.stringify(ollamaChunk) + '\n')
              if (done) { res.end(); return }
            } catch (_) { /* skip malformed chunks */ }
          }
        })

        proxyRes.on('end', () => {
          if (!res.writableEnded) {
            res.write(JSON.stringify({
              model: modelForResponse, created_at: new Date().toISOString(),
              message: { role: 'assistant', content: '' }, done: true, done_reason: 'stop',
            }) + '\n')
            res.end()
          }
        })
      }
    })

    proxyReq.on('error', (err) => {
      if (!res.headersSent) this.sendJson(res, 502, { error: `Backend unavailable: ${err.message}` })
    })
    proxyReq.on('timeout', () => {
      proxyReq.destroy()
      if (!res.headersSent) this.sendJson(res, 504, { error: 'Request timed out' })
    })
    proxyReq.write(JSON.stringify(openaiBody))
    proxyReq.end()
    req.on('close', () => { if (!proxyReq.destroyed) proxyReq.destroy() })
  }

  // ── /api/generate ──

  private async handleOllamaGenerate(req: IncomingMessage, res: ServerResponse): Promise<void> {
    const body = await this.readBody(req)
    if (!body) return this.sendJson(res, 400, { error: 'Empty request body' })

    let parsed: any
    try { parsed = JSON.parse(body) } catch (_) {
      return this.sendJson(res, 400, { error: 'Invalid JSON' })
    }

    const session = this.resolveSession(parsed.model)
    if (!session) return this.sendJson(res, 404, { error: `model '${parsed.model || 'unknown'}' not found` })

    if (session.status !== 'running') {
      const ok = await this.jitLoad(session.id)
      if (!ok) return this.sendJson(res, 503, { error: 'Model failed to load' })
      const fresh = db.getSession(session.id)
      if (fresh) { session.port = fresh.port; session.host = fresh.host === '0.0.0.0' ? '127.0.0.1' : fresh.host }
    }

    sessionManager.touchSession(session.id)

    const opts = parsed.options || {}
    const isStreaming = parsed.stream !== false
    const openaiBody: any = {
      model: parsed.model || session.modelName,
      prompt: parsed.prompt || '',
      stream: isStreaming,
    }
    if (opts.num_predict != null) openaiBody.max_tokens = opts.num_predict
    if (opts.temperature != null) openaiBody.temperature = opts.temperature
    if (opts.top_p != null) openaiBody.top_p = opts.top_p
    if (opts.stop) openaiBody.stop = opts.stop

    const modelForResponse = parsed.model || session.modelName

    const proxyOpts = {
      hostname: session.host,
      port: session.port,
      path: '/v1/completions',
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      timeout: PROXY_TIMEOUT_MS,
    }

    const proxyReq = httpRequest(proxyOpts, (proxyRes) => {
      if (!isStreaming) {
        // Non-streaming: buffer, translate, send
        let data = ''
        proxyRes.on('data', (chunk: Buffer) => { data += chunk.toString() })
        proxyRes.on('end', () => {
          try {
            const openai = JSON.parse(data)
            this.sendJson(res, 200, {
              model: modelForResponse,
              created_at: new Date().toISOString(),
              response: openai.choices?.[0]?.text || '',
              done: true,
              done_reason: openai.choices?.[0]?.finish_reason || 'stop',
            })
          } catch (_) {
            this.sendJson(res, 502, { error: 'Failed to parse backend response' })
          }
        })
      } else {
        // Streaming: SSE → NDJSON (same pattern as handleOllamaChat)
        res.writeHead(200, { 'Content-Type': 'application/x-ndjson', 'Transfer-Encoding': 'chunked' })

        let buffer = ''
        proxyRes.on('data', (chunk: Buffer) => {
          buffer += chunk.toString()
          const lines = buffer.split('\n')
          buffer = lines.pop() || ''

          for (const line of lines) {
            const trimmed = line.trim()
            if (!trimmed.startsWith('data: ')) continue
            const payload = trimmed.slice(6)

            if (payload === '[DONE]') {
              res.write(JSON.stringify({
                model: modelForResponse, created_at: new Date().toISOString(),
                response: '', done: true, done_reason: 'stop',
              }) + '\n')
              res.end()
              return
            }

            try {
              const chunk = JSON.parse(payload)
              const text = chunk.choices?.[0]?.text || ''
              const finishReason = chunk.choices?.[0]?.finish_reason
              const done = finishReason != null

              const ollamaChunk: any = {
                model: modelForResponse,
                created_at: new Date().toISOString(),
                response: text,
                done,
              }
              if (done) {
                ollamaChunk.done_reason = finishReason || 'stop'
                if (chunk.usage) {
                  ollamaChunk.eval_count = chunk.usage.completion_tokens || 0
                  ollamaChunk.prompt_eval_count = chunk.usage.prompt_tokens || 0
                }
              }
              res.write(JSON.stringify(ollamaChunk) + '\n')
              if (done) { res.end(); return }
            } catch (_) { /* skip malformed chunks */ }
          }
        })

        proxyRes.on('end', () => {
          if (!res.writableEnded) {
            res.write(JSON.stringify({
              model: modelForResponse, created_at: new Date().toISOString(),
              response: '', done: true, done_reason: 'stop',
            }) + '\n')
            res.end()
          }
        })
      }
    })

    proxyReq.on('error', (err) => {
      if (!res.headersSent) this.sendJson(res, 502, { error: `Backend unavailable: ${err.message}` })
    })
    proxyReq.on('timeout', () => { proxyReq.destroy(); if (!res.headersSent) this.sendJson(res, 504, { error: 'Timed out' }) })
    proxyReq.write(JSON.stringify(openaiBody))
    proxyReq.end()
    req.on('close', () => { if (!proxyReq.destroyed) proxyReq.destroy() })
  }

  // ── /api/show ──

  private async handleOllamaShow(req: IncomingMessage, res: ServerResponse): Promise<void> {
    const body = await this.readBody(req)
    let parsed: any
    try { parsed = JSON.parse(body || '{}') } catch (_) {
      return this.sendJson(res, 400, { error: 'Invalid JSON' })
    }

    const session = this.resolveSession(parsed.name || parsed.model)
    if (!session) return this.sendJson(res, 404, { error: 'model not found' })

    this.sendJson(res, 200, {
      modelfile: '',
      parameters: '',
      template: '',
      details: { format: 'mlx', family: '', parameter_size: '', quantization_level: '' },
      model_info: { name: session.servedModelName || session.modelName },
    })
  }

  // ── /api/embeddings ──

  private async handleOllamaEmbed(req: IncomingMessage, res: ServerResponse): Promise<void> {
    const body = await this.readBody(req)
    let parsed: any
    try { parsed = JSON.parse(body || '{}') } catch (_) {
      return this.sendJson(res, 400, { error: 'Invalid JSON' })
    }

    const session = this.resolveSession(parsed.model)
    if (!session) return this.sendJson(res, 404, { error: 'model not found' })

    if (session.status !== 'running') {
      const ok = await this.jitLoad(session.id)
      if (!ok) return this.sendJson(res, 503, { error: 'Model failed to load' })
      const fresh = db.getSession(session.id)
      if (fresh) { session.port = fresh.port; session.host = fresh.host === '0.0.0.0' ? '127.0.0.1' : fresh.host }
    }

    sessionManager.touchSession(session.id)

    // Translate Ollama embeddings → OpenAI
    const openaiBody = JSON.stringify({
      model: parsed.model,
      input: parsed.input || parsed.prompt || '',
    })

    const proxyOpts = {
      hostname: session.host,
      port: session.port,
      path: '/v1/embeddings',
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
    }

    const proxyReq = httpRequest(proxyOpts, (proxyRes) => {
      let data = ''
      proxyRes.on('data', (chunk: Buffer) => { data += chunk.toString() })
      proxyRes.on('end', () => {
        try {
          const openai = JSON.parse(data)
          // Ollama format: { embeddings: [[...]], model: "...", total_duration: ... }
          const embeddings = openai.data?.map((d: any) => d.embedding) || []
          this.sendJson(res, 200, { model: parsed.model, embeddings, total_duration: 0 })
        } catch (_) {
          this.sendJson(res, 502, { error: 'Failed to parse backend response' })
        }
      })
    })

    proxyReq.on('error', () => { if (!res.headersSent) this.sendJson(res, 502, { error: 'Backend unavailable' }) })
    proxyReq.write(openaiBody)
    proxyReq.end()
    req.on('close', () => { if (!proxyReq.destroyed) proxyReq.destroy() })
  }

  // ═══════════════════════════════════════════════════════════════
  // Utilities
  // ═══════════════════════════════════════════════════════════════

  private readBody(req: IncomingMessage): Promise<string> {
    return new Promise(resolve => {
      const chunks: Buffer[] = []
      req.on('data', (chunk: Buffer) => chunks.push(chunk))
      req.on('end', () => resolve(Buffer.concat(chunks).toString()))
      req.on('error', () => resolve(''))
    })
  }

  private sendJson(res: ServerResponse, status: number, data: any): void {
    const json = JSON.stringify(data)
    res.writeHead(status, {
      'Content-Type': 'application/json',
      'Content-Length': Buffer.byteLength(json),
      'Access-Control-Allow-Origin': '*',
    })
    res.end(json)
  }
}

// Singleton
export const apiGateway = new ApiGateway()
