// MLX Studio Chat — eric@mlx.studio
import { ipcMain, BrowserWindow, net } from 'electron'
import { v4 as uuidv4 } from 'uuid'
import { request as httpsRequest } from 'node:https'
import { request as httpRequest } from 'node:http'
import { db, Chat, Message, Folder } from '../database'
import { sessionManager, resolveUrl, connectHost } from '../sessions'
import { BUILTIN_TOOLS, isBuiltinTool, AGENTIC_SYSTEM_PROMPT } from '../tools/registry'
import { executeBuiltinTool } from '../tools/executor'
import { readGenerationDefaults } from './models'
import { detectModelConfigFromDir } from '../model-config-registry'
import { getAuthHeaders } from './utils'

// Default connection config (fallback values)
const DEFAULT_HOST = '0.0.0.0'
const DEFAULT_PORT = 8000

/**
 * SSE-streaming fetch using Node.js http/https directly.
 * Electron 28's global fetch() uses Chromium's net module which buffers
 * SSE chunks instead of delivering them immediately. Node.js http/https
 * streams data as it arrives from the socket.
 */
async function streamingFetch(url: string, init: {
  method: string
  headers: Record<string, string>
  body: string
  signal?: AbortSignal
}): Promise<{ ok: boolean; status: number; statusText: string; body: ReadableStream<Uint8Array> | null; text: () => Promise<string> }> {
  const parsed = new URL(url)
  const isHttps = parsed.protocol === 'https:'
  const reqFn = isHttps ? httpsRequest : httpRequest
  const bodyBuf = Buffer.from(init.body, 'utf-8')

  return new Promise((resolve, reject) => {
    if (init.signal?.aborted) {
      reject(Object.assign(new Error('The operation was aborted.'), { name: 'AbortError' }))
      return
    }

    let settled = false
    const settle = (fn: () => void) => { if (!settled) { settled = true; fn() } }

    const req = reqFn({
      hostname: parsed.hostname,
      port: parsed.port || (isHttps ? 443 : 80),
      path: parsed.pathname + parsed.search,
      method: init.method,
      // Disable connection pooling — each SSE stream gets a fresh TCP connection.
      // Prevents stale keep-alive connections from causing ECONNRESET/"aborted" errors.
      agent: false,
      headers: {
        ...init.headers,
        'Content-Length': bodyBuf.length.toString()
      }
    }, (res) => {
      const ok = (res.statusCode ?? 0) >= 200 && (res.statusCode ?? 0) < 300

      if (!ok) {
        let data = ''
        res.on('data', (chunk) => { data += chunk.toString() })
        res.on('end', () => {
          settle(() => resolve({
            ok, status: res.statusCode ?? 0, statusText: res.statusMessage ?? '',
            body: null, text: () => Promise.resolve(data)
          }))
        })
        res.on('error', () => {
          settle(() => resolve({
            ok, status: res.statusCode ?? 0, statusText: res.statusMessage ?? '',
            body: null, text: () => Promise.resolve(data)
          }))
        })
        return
      }

      // Wrap Node.js stream in Web ReadableStream for compatibility with streamSSE
      const stream = new ReadableStream<Uint8Array>({
        start(controller) {
          res.on('data', (chunk: Buffer) => { controller.enqueue(new Uint8Array(chunk)) })
          res.on('end', () => { try { controller.close() } catch (_) { } })
          res.on('error', (err) => {
            console.error(`[streamingFetch] stream error: message="${(err as any)?.message}" code="${(err as any)?.code}"`)
            try { controller.error(err) } catch (_) { }
          })
          // Handle premature close (server drops connection before response completes)
          res.on('close', () => {
            if (!res.complete) {
              try { controller.error(new Error('Connection closed before response completed')) } catch (_) { }
            }
          })
        },
        cancel() { res.destroy() }
      })

      settle(() => resolve({
        ok: true, status: res.statusCode ?? 200, statusText: res.statusMessage ?? 'OK',
        body: stream, text: () => Promise.reject(new Error('Cannot read text from streaming response'))
      }))
    })

    req.on('error', (err) => {
      settle(() => reject(err))
    })

    if (init.signal) {
      const onAbort = () => {
        req.destroy()
        settle(() => reject(Object.assign(new Error('The operation was aborted.'), { name: 'AbortError' })))
      }
      init.signal.addEventListener('abort', onAbort, { once: true })
    }

    req.end(bodyBuf)
  })
}

// Common chat template stop tokens that models may generate
const TEMPLATE_STOP_TOKENS = [
  '<|im_end|>', '<|im_start|>',           // ChatML (Qwen, etc.)
  '<|eot_id|>', '<|start_header_id|>',     // Llama 3
  '<|end|>', '<|user|>', '<|assistant|>',   // Phi-3
  '</s>', '<s>',                            // Llama 2, Mistral
  '<|endoftext|>',                          // GPT-NeoX, StableLM
  '[/INST]', '[INST]',                      // Mistral instruct
  '<end_of_turn>',                          // Gemma
  '<minimax:tool_call>',                    // MiniMax tool call open tag
  '</minimax:tool_call>',                   // MiniMax tool call close tag
  '<|start|>', '<|channel|>', '<|message|>', // Harmony/GPT-OSS protocol (GLM-4.7, GPT-OSS)
]

// Regex to strip any leaked template tokens from output
const TEMPLATE_TOKEN_REGEX = new RegExp(
  TEMPLATE_STOP_TOKENS.map(t => t.replace(/[.*+?^${}()|[\]\\]/g, '\\$&')).join('|'),
  'g'
)

/**
 * Use Electron's net.fetch for remote sessions — Chromium's network stack handles
 * HTTPS certificates, system proxies, and SSE streaming properly.
 * Local sessions use streamingFetch (Node.js http/https) to avoid Electron 28's
 * global fetch buffering SSE chunks.
 */
const remoteFetch: typeof globalThis.fetch = (input, init?) => net.fetch(input as any, init as any)

// Tool category definitions for per-category filtering
const FILE_TOOLS = new Set(['read_file', 'write_file', 'edit_file', 'patch_file', 'batch_edit', 'copy_file', 'move_file', 'delete_file', 'create_directory', 'list_directory', 'insert_text', 'replace_lines', 'apply_regex', 'read_image'])
const SEARCH_TOOLS = new Set(['search_files', 'find_files', 'file_info', 'get_diagnostics', 'get_tree', 'diff_files'])
const SHELL_TOOLS = new Set(['run_command', 'spawn_process', 'get_process_output'])
const DDG_SEARCH_TOOLS = new Set(['ddg_search'])
const FETCH_TOOLS = new Set(['fetch_url'])
const GIT_TOOLS = new Set(['git'])
const UTILITY_TOOLS = new Set(['count_tokens', 'clipboard_read', 'clipboard_write', 'get_current_datetime'])
// ask_user is intentionally excluded from UTILITY_TOOLS — it's a core IPC tool that should
// always be available regardless of the utilityToolsEnabled toggle.

/** Build set of disabled tool names based on per-category toggle overrides */
function getDisabledTools(overrides: any): Set<string> {
  const disabled = new Set<string>()
  if (overrides.fileToolsEnabled === false) FILE_TOOLS.forEach(t => disabled.add(t))
  if (overrides.searchToolsEnabled === false) SEARCH_TOOLS.forEach(t => disabled.add(t))
  if (overrides.shellEnabled === false) SHELL_TOOLS.forEach(t => disabled.add(t))
  if (overrides.webSearchEnabled === false) DDG_SEARCH_TOOLS.forEach(t => disabled.add(t))
  if (overrides.fetchUrlEnabled === false) FETCH_TOOLS.forEach(t => disabled.add(t))
  if (overrides.gitEnabled === false) GIT_TOOLS.forEach(t => disabled.add(t))
  if (overrides.utilityToolsEnabled === false) UTILITY_TOOLS.forEach(t => disabled.add(t))
  // Brave web_search requires API key — always disable if no key configured
  // (user must explicitly enable Brave search via braveSearchEnabled toggle)
  if (overrides.braveSearchEnabled === false) {
    disabled.add('web_search')
  } else {
    const braveKey = db.getSetting('braveApiKey')
    if (!braveKey && !process.env.BRAVE_API_KEY) {
      disabled.add('web_search')
    }
  }
  return disabled
}

/** Filter BUILTIN_TOOLS based on per-category toggle overrides */
function filterTools(overrides: any): any[] {
  const disabled = getDisabledTools(overrides)
  if (disabled.size === 0) return BUILTIN_TOOLS
  return BUILTIN_TOOLS.filter((t: any) => !disabled.has(t.function.name))
}

// Track active requests per chat for abort/concurrency (B5/B6)
const activeRequests = new Map<string, { controller: AbortController; startedAt: number; timeoutMs: number; responseId?: string; endpoint?: { host: string; port: number }; baseUrl?: string; authHeaders?: Record<string, string> }>()
// Stale lock: each request stores its timeoutMs; stale check uses timeoutMs + 30s buffer

// ask_user: single global listener with Map-based resolver (prevents listener accumulation)
const askUserResolvers = new Map<string, (answer: string) => void>()
ipcMain.on('chat:answerUser', (_, chatId: string, answer: string) => {
  const resolve = askUserResolvers.get(chatId)
  if (resolve) {
    askUserResolvers.delete(chatId)
    resolve(answer)
  }
})

/** Abort all active chat requests targeting a specific endpoint (called when session stops) */
export function abortByEndpoint(host: string, port: number): number {
  let count = 0
  for (const [chatId, entry] of activeRequests) {
    if (entry.endpoint?.host === host && entry.endpoint?.port === port) {
      console.log(`[CHAT] Aborting chat ${chatId} — session endpoint ${host}:${port} stopped`)
      // Send server cancel if we have a response ID (fire-and-forget)
      if (entry.responseId && (entry.baseUrl || entry.endpoint)) {
        const cancelPath = entry.responseId.startsWith('resp_')
          ? `/v1/responses/${entry.responseId}/cancel`
          : `/v1/chat/completions/${entry.responseId}/cancel`
        const cancelBase = entry.baseUrl || `http://${host}:${port}`
        fetch(`${cancelBase}${cancelPath}`, {
          method: 'POST', headers: entry.authHeaders || {}, signal: AbortSignal.timeout(1000)
        }).catch(() => { /* server may already be stopped */ })
      }
      try { entry.controller.abort() } catch (_) { }
      activeRequests.delete(chatId)
      count++
    }
  }
  return count
}

/** Resolved endpoint info including optional session reference */
interface ResolvedEndpoint {
  host: string
  port: number
  session?: import('../database').Session
}

/** Resolve endpoint for a chat: use modelPath to find session, fallback to detection */
async function resolveServerEndpoint(modelPath?: string): Promise<ResolvedEndpoint> {
  // 1. If chat has modelPath, find its session (normalize to handle trailing slash)
  if (modelPath) {
    const session = sessionManager.getSessionByModelPath(modelPath.replace(/\/+$/, ''))
    if (session && session.status === 'running') {
      return { host: session.host, port: session.port, session }
    }
  }

  // 2. Detect any running processes
  const processes = await sessionManager.detect()
  const healthy = processes.find(p => p.healthy)
  if (healthy) {
    // Use 127.0.0.1 for connection (0.0.0.0 is a bind address, not connectable)
    return { host: '127.0.0.1', port: healthy.port }
  }

  return { host: '127.0.0.1', port: DEFAULT_PORT }
}

export function registerChatHandlers(getWindow: () => BrowserWindow | null): void {
  // Folders
  ipcMain.handle('chat:createFolder', async (_, name: string, parentId?: string) => {
    const folder: Folder = {
      id: uuidv4(),
      name,
      parentId,
      createdAt: Date.now()
    }
    db.createFolder(folder)
    return folder
  })

  ipcMain.handle('chat:getFolders', async () => {
    return db.getFolders()
  })

  ipcMain.handle('chat:deleteFolder', async (_, id: string) => {
    db.deleteFolder(id)
    return { success: true }
  })

  // Chats
  ipcMain.handle('chat:create', async (_, title: string, modelId: string, folderId?: string, modelPath?: string) => {
    const chat: Chat = {
      id: uuidv4(),
      title,
      folderId,
      createdAt: Date.now(),
      updatedAt: Date.now(),
      modelId,
      modelPath
    }
    db.createChat(chat)

    // Apply model defaults from generation_config.json (not from sibling chats)
    if (modelPath) {
      try {
        const defaults = await readGenerationDefaults(modelPath)
        if (defaults) {
          db.setChatOverrides({
            chatId: chat.id,
            temperature: defaults.temperature,
            topP: defaults.topP,
            topK: defaults.topK,
            minP: defaults.minP,
            repeatPenalty: defaults.repeatPenalty
          })
          console.log(`[CHAT] Applied generation defaults for ${chat.id}:`, defaults)
        }
      } catch (e) {
        console.error('[CHAT] Failed to read generation defaults:', e)
      }
    }

    return chat
  })

  ipcMain.handle('chat:getByModel', async (_, modelPath: string) => {
    return db.getChatsByModelPath(modelPath)
  })

  ipcMain.handle('chat:getRecent', async (_, limit: number) => {
    return db.getRecentChats(limit)
  })

  ipcMain.handle('chat:getAll', async (_, folderId?: string) => {
    return db.getChats(folderId)
  })

  ipcMain.handle('chat:get', async (_, id: string) => {
    return db.getChat(id)
  })

  ipcMain.handle('chat:update', async (_, id: string, updates: Partial<Chat>) => {
    db.updateChat(id, updates)
    return { success: true }
  })

  ipcMain.handle('chat:delete', async (_, id: string) => {
    db.deleteChat(id)
    return { success: true }
  })

  ipcMain.handle('chat:search', async (_, query: string) => {
    return db.searchChats(query)
  })

  // Messages
  ipcMain.handle('chat:getMessages', async (_, chatId: string) => {
    return db.getMessages(chatId)
  })

  ipcMain.handle('chat:addMessage', async (_, chatId: string, role: string, content: string) => {
    // Ensure chat exists (FK constraint on messages.chat_id)
    const chat = db.getChat(chatId)
    if (!chat) {
      throw new Error(`Cannot add message: chat ${chatId} not found`)
    }
    const message: Message = {
      id: uuidv4(),
      chatId,
      role: role as 'system' | 'user' | 'assistant',
      content,
      timestamp: Date.now()
    }
    db.addMessage(message)
    return message
  })

  // Send message and get streaming response
  // Optional 4th arg: endpoint override { host, port } for multi-server support
  // Optional 5th arg: image attachments for vision/multimodal models
  ipcMain.handle('chat:sendMessage', async (_, chatId: string, content: string, endpoint?: { host: string; port: number }, attachments?: Array<{ dataUrl: string; name: string }>) => {
    // B6: Concurrency guard — reject if a request is already active for this chat
    // B6: Concurrency guard with stale lock recovery
    const existing = activeRequests.get(chatId)
    if (existing) {
      const age = Date.now() - existing.startedAt
      // Use the timeout configured when that request started, plus 30s buffer
      // Cap at 10 minutes to prevent indefinite lock (e.g., when serverTimeout is 86400s)
      const staleLockMs = Math.min(existing.timeoutMs + 30_000, 30 * 60 * 1000)
      if (age > staleLockMs) {
        // Lock is stale — abort and clear it
        console.log(`[CHAT] Clearing stale lock for ${chatId} (${Math.round(age / 1000)}s old, limit ${Math.round(staleLockMs / 1000)}s)`)
        try { existing.controller.abort() } catch (_) { }
        activeRequests.delete(chatId)
      } else {
        throw new Error('A message is already being generated for this chat')
      }
    }

    // B5: Create AbortController for this request
    const abortController = new AbortController()
    let timedOut = false

    const chat = db.getChat(chatId)
    if (!chat) {
      throw new Error('Chat not found')
    }

    // Look up session for this chat — needed for timeout, reasoning parser,
    // AND for endpoint resolution (remote sessions need remoteUrl/apiKey/type)
    let timeoutSeconds = 300
    let sessionHasReasoningParser = false
    let isHarmonyModel = false
    let chatSession: import('../database').Session | undefined
    if (chat.modelPath) {
      chatSession = sessionManager.getSessionByModelPath(chat.modelPath.replace(/\/+$/, ''))
      if (chatSession) {
        // Touch session to reset idle timer — prevents premature sleep during active chat
        sessionManager.touchSession(chatSession.id)
        try {
          const sessionConfig = JSON.parse(chatSession.config)
          if (sessionConfig.timeout && sessionConfig.timeout > 0) {
            timeoutSeconds = sessionConfig.timeout
          }
          // Check if model has a reasoning parser (for enable_thinking default)
          if (sessionConfig.reasoningParser && sessionConfig.reasoningParser !== 'auto') {
            sessionHasReasoningParser = true
            isHarmonyModel = sessionConfig.reasoningParser === 'openai_gptoss'
          } else if (sessionConfig.reasoningParser === 'auto' && chat.modelPath) {
            // "auto" means use detection
            const detected = detectModelConfigFromDir(chat.modelPath)
            sessionHasReasoningParser = !!detected.reasoningParser
            isHarmonyModel = detected.reasoningParser === 'openai_gptoss'
          }
        } catch (_) { }
      }
    }
    const fetchTimeout = setTimeout(() => { timedOut = true; abortController.abort() }, timeoutSeconds * 1000)
    activeRequests.set(chatId, { controller: abortController, startedAt: Date.now(), timeoutMs: timeoutSeconds * 1000, endpoint: undefined, responseId: undefined })

    // Resolve actual server endpoint: explicit endpoint > session by modelPath > detect > default
    // CRITICAL: When endpoint is passed from the renderer, attach the chatSession
    // so remote sessions get proper remoteUrl, auth headers, and health check path.
    // SECURITY: Validate renderer-provided endpoint is localhost or matches a known session
    if (endpoint) {
      const isLocalhost = endpoint.host === '127.0.0.1' || endpoint.host === 'localhost' || endpoint.host === '::1' || endpoint.host === '0.0.0.0'
      const isKnownSession = chatSession && chatSession.host === endpoint.host && chatSession.port === endpoint.port
      if (!isLocalhost && !isKnownSession) {
        throw new Error(`Endpoint ${endpoint.host}:${endpoint.port} not allowed — must be localhost or match a configured session`)
      }
    }
    const resolved = endpoint
      ? { host: endpoint.host, port: endpoint.port, session: chatSession } as ResolvedEndpoint
      : await resolveServerEndpoint(chat.modelPath)

    // Detect remote session and compute base URL + auth headers
    const resolvedSession = resolved.session
    const isRemote = resolvedSession?.type === 'remote'
    const rawBaseUrl = isRemote && resolvedSession?.remoteUrl
      ? resolvedSession.remoteUrl.replace(/\/+$/, '')
      : `http://${connectHost(resolved.host)}:${resolved.port}`
    // Resolve .local mDNS hostnames to IPv4 — Node.js fetch resolves them to
    // unreachable IPv6 link-local addresses (fe80::...) causing "fetch failed"
    const baseUrl = await resolveUrl(rawBaseUrl)
    console.log(`[CHAT] Endpoint resolution: isRemote=${isRemote}, rawBaseUrl=${rawBaseUrl}, baseUrl=${baseUrl}, session=${resolvedSession?.id ?? 'none'}, type=${resolvedSession?.type ?? 'none'}`)
    const authHeaders: Record<string, string> = resolvedSession?.id ? getAuthHeaders(resolvedSession.id) : {}
    // Update active request entry with resolved baseUrl and auth for cancel support
    const activeEntry = activeRequests.get(chatId)
    if (activeEntry) {
      activeEntry.endpoint = { host: resolved.host, port: resolved.port }
      activeEntry.baseUrl = baseUrl
      if (Object.keys(authHeaders).length > 0) activeEntry.authHeaders = authHeaders
    }

    // Health check with retry — wait for server to become ready instead of
    // failing immediately. This prevents orphaned user messages and allows
    // chatting as soon as the server finishes loading.
    //
    // OPTIMIZATION: If the global health monitor confirmed this session healthy
    // within the last 15 seconds, skip the per-message health check entirely.
    // The global monitor runs every 5s, so 15s gives a generous window.
    // This avoids adding 100-500ms+ RTT on every single message for remote sessions.
    const recentlyHealthy = resolvedSession?.id
      ? (Date.now() - sessionManager.getLastHealthyAt(resolvedSession.id) < 15_000)
      : false

    // Remote sessions: 1 quick attempt then proceed (the request itself has a timeout).
    // Local sessions: 15 retries with 2s delays (30s total — JANG models need longer to dequantize).
    const maxHealthRetries = isRemote ? 1 : 15
    const healthRetryDelay = isRemote ? 500 : 2000
    let healthOk = recentlyHealthy
    if (recentlyHealthy) {
      console.log(`[CHAT] Skipping health check — global monitor confirmed healthy within 15s`)
    } else {
      const healthUrl = isRemote ? `${baseUrl}/v1/models` : `${baseUrl}/health`
      console.log(`[CHAT] Health check URL: ${healthUrl} (${isRemote ? 'remote' : 'local'}, max ${maxHealthRetries} attempts)`)
      for (let attempt = 0; attempt < maxHealthRetries; attempt++) {
        try {
          const healthRes = await fetch(healthUrl, { headers: authHeaders, signal: AbortSignal.timeout(isRemote ? 3000 : 5000) })
          if (healthRes.ok) {
            healthOk = true
            console.log(`[CHAT] Health check passed on attempt ${attempt + 1}`)
            break
          }
          if (attempt < maxHealthRetries - 1) {
            console.log(`[CHAT] Server not ready (HTTP ${healthRes.status}), retrying in ${healthRetryDelay}ms...`)
            await new Promise(r => setTimeout(r, healthRetryDelay))
          }
        } catch (healthErr: any) {
          console.log(`[CHAT] Health check failed (attempt ${attempt + 1}/${maxHealthRetries}): ${healthErr.message || healthErr.cause?.message || healthErr}`)
          if (attempt < maxHealthRetries - 1) {
            await new Promise(r => setTimeout(r, healthRetryDelay))
          }
        }
      }
    }
    // For remote sessions: proceed even if health check failed — the request has
    // its own timeout and the server may just be busy with another generation.
    if (!healthOk && isRemote) {
      console.log(`[CHAT] Remote health check failed but proceeding anyway — request will use its own timeout`)
      healthOk = true
    }
    if (!healthOk) {
      activeRequests.delete(chatId)
      clearTimeout(fetchTimeout)
      throw new Error(`Cannot reach server on port ${resolved.port} after ${maxHealthRetries} attempts (${maxHealthRetries * healthRetryDelay / 1000}s). The model may still be loading — wait for the status indicator to turn green, then try again.`)
    }

    // Add user message AFTER health check passes — this prevents orphaned
    // user messages when the server isn't ready yet.
    // When images are attached, store content as JSON array of content parts
    const hasAttachments = attachments && attachments.length > 0
    const userContentForDb = hasAttachments
      ? JSON.stringify([
        ...(content.trim() ? [{ type: 'text', text: content }] : []),
        ...attachments.map(a => ({ type: 'image_url', image_url: { url: a.dataUrl } })),
      ])
      : content
    const userMessage: Message = {
      id: uuidv4(),
      chatId,
      role: 'user',
      content: userContentForDb,
      timestamp: Date.now()
    }
    db.addMessage(userMessage)

    // Generate assistant message ID upfront so typing indicator can reference it
    const assistantMessageId = uuidv4()

    // Signal to renderer that the model is processing (typing indicator during TTFT)
    try {
      const win = getWindow()
      if (win && !win.isDestroyed()) {
        win.webContents.send('chat:typing', { chatId, messageId: assistantMessageId })
      }
    } catch (_) { }

    // Get messages for context
    const messages = db.getMessages(chatId)

    // Get overrides if any
    const overrides = db.getChatOverrides(chatId)

    // Build request messages with system prompt if set
    // Using any[] to support tool_calls and tool_call_id fields
    const requestMessages: any[] = []

    // Add system prompt from overrides if available, or agentic prompt when built-in tools enabled
    const hasSystemPrompt = !!overrides?.systemPrompt
    if (hasSystemPrompt && overrides?.builtinToolsEnabled) {
      const toolRule = '\n\nIMPORTANT: After using any tools, you MUST always provide a substantive response explaining what you found or did. Never stop after just executing tools.'
      requestMessages.push({ role: 'system', content: overrides!.systemPrompt! + toolRule })
    } else if (hasSystemPrompt) {
      requestMessages.push({ role: 'system', content: overrides!.systemPrompt! })
    } else if (overrides?.builtinToolsEnabled) {
      requestMessages.push({ role: 'system', content: AGENTIC_SYSTEM_PROMPT })
    }
    // No default system prompt injected — let the model's native template handle defaults.
    // Injecting "You are a helpful assistant." reinforces safety behavior in abliterated/CRACK models.

    // Add conversation messages (skip any existing system messages to avoid duplicates)
    // Messages with JSON content arrays (multimodal) are parsed back to content parts for the API
    for (const m of messages) {
      if (m.role === 'system' && (hasSystemPrompt || overrides?.builtinToolsEnabled)) continue
      let msgContent: any = m.content
      // Strip "[Generation interrupted]" markers from previous assistant messages —
      // these are UI-only annotations saved to DB on abort, not meant for the model
      if (m.role === 'assistant' && typeof msgContent === 'string') {
        msgContent = msgContent.replace(/\n\n\[Generation interrupted\]$/, '').replace(/^\[Generation interrupted\]$/, '')
        // Strip any leaked <think> blocks from prior messages when thinking is OFF.
        // These can leak if server didn't catch them or model was mid-think on abort.
        // Without stripping, the model sees prior thinking in context and mimics it.
        if (overrides?.enableThinking === false) {
          msgContent = msgContent.replace(/<think>[\s\S]*?<\/think>\s*/g, '')
        }
        if (!msgContent.trim()) continue // Skip entirely empty aborted messages
      }
      // Detect JSON content arrays (multimodal messages with images)
      if (m.role === 'user' && m.content.startsWith('[')) {
        try {
          const parsed = JSON.parse(m.content)
          if (Array.isArray(parsed) && parsed.length > 0 && parsed[0].type) {
            msgContent = parsed
          }
        } catch { /* not JSON, use as plain string */ }
      }
      requestMessages.push({ role: m.role, content: msgContent })
    }

    // Prepare assistant message placeholder
    const assistantMessage: Message = {
      id: assistantMessageId,
      chatId,
      role: 'assistant',
      content: '',
      timestamp: Date.now()
    }

    // Metrics tracking
    const startTime = Date.now()
    let fetchStartTime = startTime // Updated just before the API fetch (for accurate TTFT)
    let tokenCount = 0
    let promptTokens = 0
    let cachedTokens = 0
    let firstTokenTime: number | null = null
    // Track actual generation time (excludes PP and tool execution pauses)
    let generationMs = 0
    let lastTokenTime: number | null = null
    // Rolling window for live TPS: circular buffer of (timestamp, tokenCount) snapshots.
    // Uses actual token count deltas for accurate throughput — handles multi-token SSE chunks
    // correctly (e.g., reasoning batches where each chunk may contain 2+ tokens).
    const TPS_BUFFER_SIZE = 30
    const tpsSnapshots: Array<[number, number]> = [] // [timestamp, relative tokenCount]
    let liveTps = 0
    let tpsTokenBase = 0 // re-anchor point for tpsSnapshots after iteration reset
    // No streaming throttle — emit every token. Renderer-side useTypewriter
    // in MessageBubble.tsx handles smooth character reveal via rAF.
    let reader: ReadableStreamDefaultReader<Uint8Array> | undefined
    // (thinkingTimer removed — "Thinking silently" indicator disabled)
    let fullContent = ''
    let reasoningContent = ''
    // Accumulates content across tool iterations so abort during tool execution can recover
    // earlier content that would otherwise be lost when fullContent is reset between iterations
    let allGeneratedContent = ''
    // Per-iteration token count for auto-continue threshold (tokenCount is cumulative)
    let iterationTokenCount = 0
    let iterationTokenBase = 0 // tokenCount at start of iteration (for server-usage delta)
    // Cumulative token offset: tracks total tokens from completed iterations.
    // Server restarts completion_tokens from 0 on each new HTTP request, so
    // raw tokenCount only reflects the current iteration. This offset + iterationTokenCount
    // gives the true total across all tool iterations.
    let cumulativeTokenOffset = 0
    // Collect tool statuses for DB persistence (mirrors what's emitted to renderer)
    const collectedToolStatuses: Array<{ phase: string; toolName: string; detail?: string; iteration?: number; contentOffset?: number }> = []
    // Declared outside try so catch block can access them for error recovery
    let isReasoning = false
    let lastFinishReason: string | undefined
    // Periodic DB save interval — saves content every 5s so it survives navigation/crashes
    let periodicSaveInterval: ReturnType<typeof setInterval> | null = null

    // Pre-insert assistant message to DB immediately so periodic updates have a row to update.
    // Uses INSERT OR REPLACE so the final addMessage at completion overwrites cleanly.
    db.addMessage(assistantMessage)

    const startPeriodicSave = () => {
      if (periodicSaveInterval) return
      periodicSaveInterval = setInterval(() => {
        const saveContent = allGeneratedContent
          ? (fullContent.trim() ? allGeneratedContent + '\n\n' + fullContent.trim() : allGeneratedContent)
          : fullContent
        if (saveContent || reasoningContent) {
          try {
            db.updateMessageContent(assistantMessage.id, saveContent, reasoningContent || undefined)
          } catch (_) { }
        }
      }, 5000)
    }
    const stopPeriodicSave = () => {
      if (periodicSaveInterval) {
        clearInterval(periodicSaveInterval)
        periodicSaveInterval = null
      }
    }

    try {
      // Determine wire format: 'responses' or 'completions' (default)
      const wireApi = overrides?.wireApi || 'completions'
      const useResponsesApi = wireApi === 'responses'

      // Call API (local vMLX Engine or remote OpenAI-compatible endpoint)
      const apiUrl = useResponsesApi
        ? `${baseUrl}/v1/responses`
        : `${baseUrl}/v1/chat/completions`
      console.log(`[CHAT] Sending to: ${apiUrl} (wire: ${wireApi}, remote: ${isRemote})`)

      // Get model name: remote uses configured model, local reads from health endpoint
      let modelName = isRemote
        ? (resolvedSession?.remoteModel || chat.modelId || 'default')
        : (chat.modelId || 'default')
      if (!isRemote) {
        try {
          const healthRes = await fetch(`${baseUrl}/health`, { signal: AbortSignal.timeout(1000) })
          if (healthRes.ok) {
            const health = await healthRes.json()
            if (health.model_name) modelName = health.model_name
          }
        } catch (_) { /* use fallback */ }
      }

      // Only send stop sequences when the user explicitly sets them in chat settings.
      // The server already handles stop tokens via the model's chat template — sending
      // all template tokens for every model risks false-positive stops (e.g. Qwen hitting </s>).
      const stopSequences = overrides?.stopSequences
        ? overrides.stopSequences.split(',').map((s: string) => s.trim()).filter(Boolean)
        : undefined

      // Build request body — shared between initial request and tool follow-ups
      const buildRequestBody = (): Record<string, any> => {
        if (useResponsesApi) {
          const systemMessages = requestMessages.filter((m: any) => m.role === 'system')
          const instructions = overrides?.systemPrompt || (systemMessages.length > 0 ? systemMessages.map((m: any) => m.content).join('\n') : undefined)
          const inputMessages = requestMessages.filter((m: any) => m.role !== 'system')
          const obj: Record<string, any> = {
            model: modelName,
            input: inputMessages,
            instructions,
            // Only send temperature/top_p when explicitly set in chat overrides.
            // When omitted, the server uses its --default-temperature/--default-top-p CLI args.
            ...(overrides?.temperature != null ? { temperature: overrides.temperature } : {}),
            ...(overrides?.topP != null ? { top_p: overrides.topP } : {}),
            ...(overrides?.maxTokens ? { max_output_tokens: overrides.maxTokens } : {}),
            stream: true,
            stream_options: { include_usage: true }
          }
          if (stopSequences) obj.stop = stopSequences
          if (overrides?.topK != null && overrides.topK > 0) obj.top_k = overrides.topK
          if (overrides?.minP != null && overrides.minP > 0) obj.min_p = overrides.minP
          if (overrides?.repeatPenalty != null && overrides.repeatPenalty !== 1.0) obj.repetition_penalty = overrides.repeatPenalty
          if (overrides?.builtinToolsEnabled) {
            obj.tools = filterTools(overrides).map(t => ({
              type: 'function',
              name: t.function.name,
              description: t.function.description,
              parameters: t.function.parameters
            }))
          }
          // enable_thinking: explicit user override sent to both local and remote.
          // When undefined (auto), local server auto-detects from model config; remote gets sessionHasReasoningParser as hint.
          if (overrides?.enableThinking !== undefined) {
            obj.enable_thinking = overrides.enableThinking
          } else if (isRemote) {
            obj.enable_thinking = sessionHasReasoningParser
          }
          // chat_template_kwargs: local only (vMLX Engine internal, no remote provider supports this)
          if (!isRemote && obj.enable_thinking !== undefined) obj.chat_template_kwargs = { enable_thinking: obj.enable_thinking }
          if (overrides?.reasoningEffort) obj.reasoning_effort = overrides.reasoningEffort
          return obj
        } else {
          const obj: Record<string, any> = {
            model: modelName,
            messages: requestMessages,
            // Only send temperature/top_p when explicitly set in chat overrides.
            // When omitted, the server uses its --default-temperature/--default-top-p CLI args.
            ...(overrides?.temperature != null ? { temperature: overrides.temperature } : {}),
            ...(overrides?.topP != null ? { top_p: overrides.topP } : {}),
            ...(overrides?.maxTokens ? { max_tokens: overrides.maxTokens } : {}),
            stream: true,
            stream_options: { include_usage: true }
          }
          if (stopSequences) obj.stop = stopSequences
          if (overrides?.topK != null && overrides.topK > 0) obj.top_k = overrides.topK
          if (overrides?.minP != null && overrides.minP > 0) obj.min_p = overrides.minP
          if (overrides?.repeatPenalty != null && overrides.repeatPenalty !== 1.0) obj.repetition_penalty = overrides.repeatPenalty
          if (overrides?.builtinToolsEnabled) {
            // Chat Completions API: tools must be in OpenAI format with "function" wrapper
            // e.g. {"type": "function", "function": {"name": ..., "parameters": ...}}
            obj.tools = filterTools(overrides)
          }
          // enable_thinking: explicit user override sent to both local and remote.
          // When undefined (auto), local server auto-detects from model config; remote gets sessionHasReasoningParser as hint.
          if (overrides?.enableThinking !== undefined) {
            obj.enable_thinking = overrides.enableThinking
          } else if (isRemote) {
            obj.enable_thinking = sessionHasReasoningParser
          }
          // chat_template_kwargs: local only (vMLX Engine internal, no remote provider supports this)
          if (!isRemote && obj.enable_thinking !== undefined) obj.chat_template_kwargs = { enable_thinking: obj.enable_thinking }
          if (overrides?.reasoningEffort) obj.reasoning_effort = overrides.reasoningEffort
          return obj
        }
      }
      const requestBody = JSON.stringify(buildRequestBody())

      fetchStartTime = Date.now() // Capture just before fetch for accurate TTFT
      // Remote: use Electron's net.fetch (Chromium stack — proper HTTPS certs, proxies, SSE).
      // Local: use streamingFetch (Node.js http/https) to avoid Electron 28's SSE buffering bug.
      const response = isRemote
        ? await remoteFetch(apiUrl, {
          method: 'POST',
          headers: { 'Content-Type': 'application/json', ...authHeaders },
          body: requestBody,
          signal: abortController.signal
        })
        : await streamingFetch(apiUrl, {
          method: 'POST',
          headers: { 'Content-Type': 'application/json', ...authHeaders },
          body: requestBody,
          signal: abortController.signal
        })

      if (!response.ok) {
        const errorText = await response.text()
        // Try to extract structured error detail from JSON responses
        let errorDetail = errorText
        try {
          const parsed = JSON.parse(errorText)
          if (parsed.detail) {
            errorDetail = typeof parsed.detail === 'string' ? parsed.detail
              : Array.isArray(parsed.detail) ? parsed.detail.map((d: any) => d.msg || JSON.stringify(d)).join('; ')
              : JSON.stringify(parsed.detail)
          }
        } catch { /* use raw text */ }
        throw new Error(`API error: ${response.status} - ${errorDetail}`)
      }

      // Stream response
      reader = response.body?.getReader()
      if (!reader) throw new Error('Response body is null')

      fullContent = ''
      reasoningContent = ''
      isReasoning = false
      let currentEventType = '' // Track SSE event type for Responses API

      // Track whether server sends real token counts (via usage in each SSE chunk)
      let serverSendsUsage = false


      // Track tool calls received during streaming for MCP auto-execution
      let receivedToolCalls: Array<{ id: string; function: { name: string; arguments: string } }> = []
      // Track finish_reason from server to detect truncation (length), content filter, etc.
      // (declared outside try block so catch can access it for abort recovery)
      // Track tool iteration count (declared here so processLine closure can access it)
      const MAX_TOOL_ITERATIONS = overrides?.maxToolIterations ?? 10
      let toolIteration = 0

      // Track the length of content last emitted to renderer (for inline tool call positioning)
      let lastEmittedContentLength = 0

      // Helper: emit tool call status to renderer (separate from content stream)
      const emitToolStatus = (phase: string, toolName: string, detail?: string, iteration?: number) => {
        const contentOffset = phase === 'calling' ? lastEmittedContentLength : undefined
        // Collect for persistence — include detail for calling, result, and error phases
        // so tool results are visible after reload (truncate large results to 4KB)
        const persistDetail = (phase === 'calling' || phase === 'result' || phase === 'error')
          ? (detail && detail.length > 4096 ? detail.slice(0, 4096) + '...' : detail)
          : undefined
        collectedToolStatuses.push({
          phase, toolName, iteration, contentOffset,
          detail: persistDetail
        })
        try {
          const win = getWindow()
          if (win && !win.isDestroyed()) {
            win.webContents.send('chat:toolStatus', {
              chatId,
              messageId: assistantMessage.id,
              phase,
              toolName,
              detail,
              iteration,
              contentOffset
            })
          }
        } catch (_) { }
      }

      // Client-side tool call buffering: suppress content when leaked tool call XML detected.
      // Must check RAW content before template token stripping, since markers like
      // <minimax:tool_call> get stripped by TEMPLATE_TOKEN_REGEX and never reach fullContent.
      let clientToolCallBuffering = false
      let rawAccumulated = '' // Tracks unstripped content for tool call detection
      // Client-side <think> tag extraction: tracks whether we're inside a <think> block
      // when the server doesn't provide reasoning_content (fallback for all parser types)
      let clientSideThinkParsing = false

      // Helper: emit streaming delta to renderer
      // skipClientCount: when true, skip client-side token counting/TPS (used when
      // a single SSE chunk is split into multiple emitDelta calls by think-tag extraction,
      // so we only count once per SSE chunk, not once per emitDelta call)
      const emitDelta = (delta: string, isReasoningDelta: boolean, skipClientCount = false) => {
        // Skip emission if abort already fired — prevents stale tokens from reaching renderer
        if (abortController.signal.aborted) return
        // Track raw content BEFORE stripping for tool call marker detection
        if (!isReasoningDelta) {
          rawAccumulated += delta
          // Only activate buffering when tool call markers appear at the start of a line,
          // not when the model is explaining tool syntax in prose (e.g., "I'll use <tool_call>...")
          if (!clientToolCallBuffering) {
            // Catch real tool call formats AND hallucinated Claude-style tool calls
            // Use trailing window (last 200 chars) to avoid O(n) regex on full response
            const lineStartPattern = /(?:^|\n)\s*(?:<minimax:tool_call|<tool_call>|\[Calling tool:|<invoke name=|<read_file\b|<write_file\b|<run_command\b|<search_files\b|<edit_file\b|<list_directory\b|<execute_command\b|<bash\b)/
            const searchWindow = rawAccumulated.length > 200 ? rawAccumulated.slice(-200) : rawAccumulated
            if (lineStartPattern.test(searchWindow)) {
              clientToolCallBuffering = true
              console.log(`[CHAT] Client-side tool call buffering activated`)
            }
          }
        }

        // Strip any leaked chat template tokens from the delta
        delta = delta.replace(TEMPLATE_TOKEN_REGEX, '')
        if (!delta) return
        // Strip Harmony protocol residue — only for GLM/GPT-OSS models that use the
        // Harmony <|start|><|channel|><|message|> protocol. Without this guard, these
        // regexes would strip legitimate prose like "assistant analysis" from all models.
        if (isHarmonyModel) {
          delta = delta.replace(/<\/?(?:assistant|analysis|final)+/gi, '')
          delta = delta.replace(/(?:assistant\s*){1,3}(?:analysis|final)/gi, '')
          delta = delta.replace(/(?:analysis|final)\s*(?:assistant\s*){1,3}/gi, '')
          if (!delta) return
        }
        // Strip U+FFFD replacement characters
        delta = delta.replace(/\uFFFD/g, '')
        if (!delta) return

        // === State updates (always, no throttle) ===
        const now = Date.now()
        if (firstTokenTime === null) { firstTokenTime = now; startPeriodicSave() }
        // Track generation-only time: count time between consecutive tokens.
        // Gaps > 5s (e.g., tool execution, follow-up PP) are excluded.
        // Threshold is 5s (not 2s) to handle slow big models at ~0.5 tok/s.
        if (lastTokenTime !== null) {
          const gap = now - lastTokenTime
          if (gap < 5000) generationMs += gap
        }
        lastTokenTime = now

        if (isReasoningDelta) {
          isReasoning = true
          reasoningContent += delta
        } else {
          if (isReasoning) {
            isReasoning = false
            try {
              const win = getWindow()
              if (win && !win.isDestroyed()) {
                win.webContents.send('chat:reasoningDone', { chatId, messageId: assistantMessage.id, reasoningContent })
              }
            } catch (_) { }
          }
          fullContent += delta
          // Update content offset immediately (not throttled) for accurate tool call positioning
          lastEmittedContentLength = allGeneratedContent
            ? allGeneratedContent.length + 2 + fullContent.length
            : fullContent.length
        }
        // Client-side counting (fallback when server doesn't send usage in each chunk).
        // Must happen BEFORE TPS snapshot so the rolling window uses accurate counts.
        // skipClientCount prevents inflation when think-tag splitting calls emitDelta
        // multiple times for a single SSE chunk.
        if (!serverSendsUsage && !skipClientCount) { tokenCount++; iterationTokenCount++ }

        // Rolling TPS: snapshot (timestamp, relative tokenCount) for accurate throughput.
        // Uses tpsTokenBase-relative count to avoid negative deltas at iteration boundaries
        // (server restarts completion_tokens from 0 on each new HTTP request).
        tpsSnapshots.push([now, tokenCount - tpsTokenBase])
        if (tpsSnapshots.length > TPS_BUFFER_SIZE) tpsSnapshots.shift()
        if (tpsSnapshots.length >= 2) {
          const [oldT, oldN] = tpsSnapshots[0]
          const [newT, newN] = tpsSnapshots[tpsSnapshots.length - 1]
          const span = (newT - oldT) / 1000
          const tpsDelta = newN - oldN
          liveTps = (span > 0.01 && tpsDelta > 0) ? tpsDelta / span : (tpsDelta <= 0 ? 0 : liveTps)
        }

        // Suppress rendering (but not counting/TPS) when tool call content is detected
        if (!isReasoningDelta && clientToolCallBuffering) return

        // === IPC emission — every token emitted immediately ===
        // Renderer-side useTypewriter handles smooth character reveal via rAF.

        // Live generation TPS from rolling window (real-time speed of incoming tokens).
        // Cumulative TPS (tokenCount / generationMs) is used for final saved metrics only.
        const streamTps = liveTps
        // Cumulative generation time for elapsed display
        const genSec = generationMs / 1000
        const wallSec = (now - (firstTokenTime || fetchStartTime)) / 1000
        const elapsed = genSec > 0.05 ? genSec : wallSec
        // TTFT measured from fetchStartTime (excludes health check and message building overhead)
        const ttft = Math.max(0, firstTokenTime ? (firstTokenTime - fetchStartTime) / 1000 : 0)
        const ppSpeed = (promptTokens > 0 && ttft > 0.001) ? (promptTokens / ttft).toFixed(1) : undefined

        try {
          const win = getWindow()
          if (win && !win.isDestroyed()) {
            // Include pre-tool content so UI doesn't lose earlier text when fullContent resets
            const displayContent = (!isReasoningDelta && allGeneratedContent)
              ? allGeneratedContent + '\n\n' + fullContent
              : (isReasoningDelta ? reasoningContent : fullContent)
            win.webContents.send('chat:stream', {
              chatId,
              messageId: assistantMessage.id,
              fullContent: displayContent,
              isReasoning: isReasoningDelta,
              metrics: {
                tokenCount: cumulativeTokenOffset + iterationTokenCount,
                promptTokens,
                tokensPerSecond: streamTps.toFixed(1),
                ppSpeed,
                ttft: ttft.toFixed(2),
                elapsed: elapsed.toFixed(1)
              }
            })
          }
        } catch (_) { }
      }

      // Process a single SSE data line (with event type context)
      const processLine = (trimmed: string) => {
        // Track SSE event type (Responses API uses "event:" lines)
        if (trimmed.startsWith('event: ')) {
          currentEventType = trimmed.slice(7)
          return
        }
        if (!trimmed) { currentEventType = ''; return }  // Blank line = SSE event boundary, reset type
        if (!trimmed.startsWith('data: ')) return
        const data = trimmed.slice(6)
        if (data === '[DONE]') { currentEventType = ''; return }

        try {
          const parsed = JSON.parse(data)

          if (useResponsesApi) {
            // ── Responses API SSE parsing ──
            // Track response ID from response.created event
            // Server wraps in { response: { id: "resp_..." } }
            const respId = parsed.response?.id || parsed.id
            if (currentEventType === 'response.created' && respId) {
              const entry = activeRequests.get(chatId)
              if (entry && !entry.responseId) {
                entry.responseId = respId
                entry.endpoint = { host: resolved.host, port: resolved.port }
              }
            }

            // Reasoning delta from response.reasoning.delta (custom event for thinking models)
            if (currentEventType === 'response.reasoning.delta' && parsed.delta) {
              emitDelta(parsed.delta, true)
            }

            // Reasoning done — triggers reasoningDone event in emitDelta (isReasoning=true→false transition)
            if (currentEventType === 'response.reasoning.done') {
              // Force the reasoning→content transition so reasoningDone fires
              if (isReasoning) {
                isReasoning = false
                try {
                  const win = getWindow()
                  if (win && !win.isDestroyed()) {
                    win.webContents.send('chat:reasoningDone', { chatId, messageId: assistantMessage.id, reasoningContent })
                  }
                } catch (_) { }
              }
            }

            // Delta text from response.output_text.delta
            // Server sends { delta: "text" }, not { text: "..." }
            if (currentEventType === 'response.output_text.delta' && (parsed.delta || parsed.text)) {
              emitDelta(parsed.delta || parsed.text, false)
            }

            // Handle function_call items (tool calls) from Responses API
            // response.output_item.done carries the complete tool call: { item: { type, call_id, name, arguments } }
            if (currentEventType === 'response.output_item.done' && parsed.item?.type === 'function_call') {
              const item = parsed.item
              receivedToolCalls.push({
                id: item.call_id || `call_${uuidv4().replace(/-/g, '').slice(0, 16)}`,
                function: { name: item.name, arguments: item.arguments || '{}' }
              })
              emitToolStatus('calling', item.name, item.arguments || '{}', toolIteration)
            }

            // Real-time usage from response.usage events (per-chunk, for live TPS accuracy)
            if (currentEventType === 'response.usage' && parsed.usage) {
              if (parsed.usage.output_tokens != null) {
                tokenCount = parsed.usage.output_tokens
                // Detect server token count restart (new HTTP request resets completion_tokens to 0)
                if (tokenCount < iterationTokenBase) iterationTokenBase = 0
                iterationTokenCount = Math.max(0, tokenCount - iterationTokenBase)
                // Clear contaminated client-counted entries when transitioning to server usage
                if (!serverSendsUsage) { tpsSnapshots.length = 0; tpsTokenBase = tokenCount }
                serverSendsUsage = true
              }
              if (parsed.usage.input_tokens != null) promptTokens = parsed.usage.input_tokens
              if (parsed.usage.input_tokens_details?.cached_tokens) cachedTokens = parsed.usage.input_tokens_details.cached_tokens
            }

            // Handle error events from Responses API
            // Server may emit "error", "response.error", or "response.failed" event types
            if (currentEventType === 'error' || currentEventType === 'response.error' || currentEventType === 'response.failed') {
              const errDetail = parsed.error?.message || parsed.error?.code || parsed.detail || JSON.stringify(parsed)
              console.error(`[CHAT] Responses API error event: ${errDetail}`)
              throw new Error(`Server error: ${errDetail}`)
            }

            // Final usage from response.completed event
            // Server wraps in { response: { usage: { input_tokens, output_tokens } } }
            const respUsage = parsed.response?.usage || parsed.usage
            if (currentEventType === 'response.completed') {
              // Track status for truncation detection
              const respStatus = parsed.response?.status
              if (respStatus === 'incomplete') lastFinishReason = 'length'
              else if (respStatus === 'completed') lastFinishReason = 'stop'
              else if (respStatus) lastFinishReason = respStatus
            }
            if (currentEventType === 'response.completed' && respUsage) {
              if (respUsage.output_tokens != null) {
                tokenCount = respUsage.output_tokens
                if (tokenCount < iterationTokenBase) iterationTokenBase = 0
                iterationTokenCount = Math.max(0, tokenCount - iterationTokenBase)
                if (!serverSendsUsage) { tpsSnapshots.length = 0; tpsTokenBase = tokenCount }
                serverSendsUsage = true
              }
              if (respUsage.input_tokens != null) promptTokens = respUsage.input_tokens
              if (respUsage.input_tokens_details?.cached_tokens) cachedTokens = respUsage.input_tokens_details.cached_tokens
            }
          } else {
            // ── Chat Completions SSE parsing ──
            const choice = parsed.choices?.[0]?.delta

            // Track response ID for server-side cancel
            if (parsed.id) {
              const entry = activeRequests.get(chatId)
              if (entry && !entry.responseId) {
                entry.responseId = parsed.id
                entry.endpoint = { host: resolved.host, port: resolved.port }
              }
            }

            // Update usage BEFORE emitting delta so metrics use real server counts
            if (parsed.usage) {
              if (parsed.usage.completion_tokens != null) {
                tokenCount = parsed.usage.completion_tokens
                // Detect server token count restart (new HTTP request resets completion_tokens to 0)
                if (tokenCount < iterationTokenBase) iterationTokenBase = 0
                iterationTokenCount = Math.max(0, tokenCount - iterationTokenBase)
                // Clear contaminated client-counted entries when transitioning to server usage
                if (!serverSendsUsage) { tpsSnapshots.length = 0; tpsTokenBase = tokenCount }
                serverSendsUsage = true
              }
              if (parsed.usage.prompt_tokens != null) promptTokens = parsed.usage.prompt_tokens
              if (parsed.usage.prompt_tokens_details?.cached_tokens) cachedTokens = parsed.usage.prompt_tokens_details.cached_tokens
            }

            // Track finish_reason (length = truncated, content_filter = filtered)
            const finishReason = parsed.choices?.[0]?.finish_reason
            if (finishReason) lastFinishReason = finishReason

            // Handle error chunks from Chat Completions (tool_choice/JSON schema failures)
            if (parsed.error) {
              const errDetail = parsed.error.message || parsed.error.code || JSON.stringify(parsed.error)
              console.error(`[CHAT] Chat completions error chunk: ${errDetail}`)
              throw new Error(`Server error: ${errDetail}`)
            }

            // Handle reasoning_content from reasoning parser
            const reasoning = choice?.reasoning_content || choice?.reasoning
            if (reasoning) {
              emitDelta(reasoning, true)
            }

            if (choice?.content) {
              // Client-side fallback: if server didn't provide reasoning_content
              // but content contains <think> tags, extract them client-side.
              // This handles servers without a reasoning parser, remote endpoints,
              // and older server versions.
              // chunkCounted tracks whether we've already counted this SSE chunk's token
              // to prevent inflation from think-tag splitting into multiple emitDelta calls.
              if (!reasoning) {
                const content = choice.content as string
                let chunkCounted = !!reasoning // if reasoning was emitted above, counting already happened
                const emitWithCount = (text: string, isR: boolean) => {
                  emitDelta(text, isR, chunkCounted)
                  chunkCounted = true // subsequent calls skip counting
                }
                if (clientSideThinkParsing) {
                  // We're inside a <think> block — check for closing tag
                  const endIdx = content.indexOf('</think>')
                  if (endIdx >= 0) {
                    const reasoningPart = content.slice(0, endIdx)
                    const contentPart = content.slice(endIdx + 8) // 8 = '</think>'.length
                    clientSideThinkParsing = false
                    if (reasoningPart) emitWithCount(reasoningPart, true)
                    if (contentPart) emitWithCount(contentPart, false)
                  } else {
                    // Still in reasoning block
                    emitWithCount(content, true)
                  }
                } else if (content.includes('<think>')) {
                  // Start of think block found in this delta
                  const startIdx = content.indexOf('<think>')
                  const preContent = content.slice(0, startIdx)
                  const afterStart = content.slice(startIdx + 7) // 7 = '<think>'.length
                  if (preContent) emitWithCount(preContent, false)
                  // Check if closing tag is also in this delta
                  const endIdx = afterStart.indexOf('</think>')
                  if (endIdx >= 0) {
                    const reasoningPart = afterStart.slice(0, endIdx)
                    const postContent = afterStart.slice(endIdx + 8)
                    if (reasoningPart) emitWithCount(reasoningPart, true)
                    if (postContent) emitWithCount(postContent, false)
                  } else {
                    clientSideThinkParsing = true
                    if (afterStart) emitWithCount(afterStart, true)
                  }
                } else {
                  emitWithCount(content, false)
                }
              } else {
                emitDelta(choice.content, false, !!reasoning)
              }
            }

            // Detect server-side tool call buffering signal (TPS keeps counting, show status)
            if (parsed.tool_call_generating) {
              if (!clientToolCallBuffering) clientToolCallBuffering = true
              console.log(`[CHAT] Server signaled tool call generation in progress`)
              emitToolStatus('generating', '', 'Generating tool call...', toolIteration)
            }

            // Handle tool_calls from streaming response
            // Supports both complete tool calls (vmlx-engine default) and incremental argument
            // streaming (OpenAI-style: first chunk has name, subsequent chunks append arguments)
            if (choice?.tool_calls && Array.isArray(choice.tool_calls)) {
              for (const tc of choice.tool_calls) {
                const fn = tc.function
                const idx = tc.index ?? -1
                if (fn?.name) {
                  // New tool call: initialize (use index for positional tracking)
                  const toolCall = {
                    id: tc.id || `call_${uuidv4().replace(/-/g, '').slice(0, 16)}`,
                    function: { name: fn.name, arguments: fn.arguments || '' }
                  }
                  if (idx >= 0) {
                    receivedToolCalls[idx] = toolCall
                  } else {
                    receivedToolCalls.push(toolCall)
                  }
                  console.log(`[CHAT] Tool call detected: ${fn.name}(${(fn.arguments || '').slice(0, 100)})`)
                  // Don't emit arguments here — during incremental streaming,
                  // arguments may be empty/partial. Final args shown after execution.
                  emitToolStatus('calling', fn.name, '', toolIteration)
                } else if (fn?.arguments && idx >= 0) {
                  // Incremental argument chunk: accumulate arguments for existing tool call
                  if (receivedToolCalls[idx]) {
                    receivedToolCalls[idx].function.arguments += fn.arguments
                  } else {
                    // Out-of-order index: initialize a placeholder to prevent sparse array crash
                    receivedToolCalls[idx] = { id: tc.id || `call_${uuidv4().replace(/-/g, '').slice(0, 16)}`, function: { name: '', arguments: fn.arguments } }
                  }
                }
              }
            }
          }

        } catch (e) {
          // Skip malformed JSON lines — log at debug level for troubleshooting
          if (e instanceof SyntaxError) {
            // Expected: malformed SSE data line
          } else {
            console.warn('[CHAT] Error processing SSE line:', (e as Error).message)
          }
        }
      }

      // ─── Helper: stream SSE response through processLine ──────────────
      const streamSSE = async (rdr: ReadableStreamDefaultReader<Uint8Array>) => {
        const dec = new TextDecoder()
        let buf = ''
        while (true) {
          // Check abort before each read — fast models can buffer many chunks
          if (abortController.signal.aborted) break
          const { value, done } = await rdr.read()
          if (done) break
          buf += dec.decode(value, { stream: true })
          const lines = buf.split('\n')
          buf = lines.pop() || ''
          for (let li = 0; li < lines.length; li++) {
            if (abortController.signal.aborted) break
            processLine(lines[li].trim())
          }
        }
        if (abortController.signal.aborted) return
        const rem = dec.decode() // flush TextDecoder streaming buffer
        if (rem) buf += rem
        // Process remaining lines (may contain multiple newline-separated events)
        if (buf.trim()) {
          for (const line of buf.split('\n')) {
            if (abortController.signal.aborted) break
            if (line.trim()) processLine(line.trim())
          }
        }
      }

      await streamSSE(reader)

      // ─── Helper: send follow-up request and stream response ────────────
      const sendFollowUp = async (): Promise<boolean> => {
        // Reset SSE parser state from previous stream
        currentEventType = ''
        // Reset fetchStartTime so TTFT for follow-up is measured correctly
        fetchStartTime = Date.now()
        firstTokenTime = null
        // Use the same wire API format as the initial request
        const url = useResponsesApi
          ? `${baseUrl}/v1/responses`
          : `${baseUrl}/v1/chat/completions`
        // Remote: net.fetch (Chromium); Local: streamingFetch (Node.js)
        const followUpInit = {
          method: 'POST',
          headers: { 'Content-Type': 'application/json', ...authHeaders },
          body: JSON.stringify(buildRequestBody()),
          signal: abortController.signal
        }
        const res = isRemote
          ? await remoteFetch(url, followUpInit)
          : await streamingFetch(url, followUpInit as any)
        if (!res.ok) {
          const errText = await res.text()
          console.log(`[CHAT] Follow-up failed: ${res.status} ${errText}`)
          emitToolStatus('error', '', `Follow-up error: ${res.status} ${errText}`, toolIteration)
          return false
        }
        const followUpReader = res.body?.getReader()
        if (!followUpReader) return false
        await streamSSE(followUpReader)
        return true
      }

      // ─── Helper: execute tool calls and push results to messages ───────
      const executeToolCalls = async () => {
        if (useResponsesApi) {
          // Responses API: push individual output items (not Chat Completions format)
          if (fullContent) {
            requestMessages.push({ type: 'output_text', text: fullContent })
          }
          for (const tc of receivedToolCalls) {
            requestMessages.push({
              type: 'function_call',
              call_id: tc.id,
              name: tc.function.name,
              arguments: tc.function.arguments
            })
          }
        } else {
          // Chat Completions: push assistant message with tool_calls array
          requestMessages.push({
            role: 'assistant',
            content: fullContent || null,
            tool_calls: receivedToolCalls.map(tc => ({
              id: tc.id,
              type: 'function' as const,
              function: { name: tc.function.name, arguments: tc.function.arguments }
            }))
          })
        }

        const pendingImageDataUrls: string[] = []
        for (const tc of receivedToolCalls) {
          // Check abort between each tool — don't make user wait for all tools to finish
          if (abortController.signal.aborted) throw Object.assign(new Error('AbortError'), { name: 'AbortError' })
          let resultText = ''
          try {
            let toolArgs: Record<string, any>
            try {
              toolArgs = JSON.parse(tc.function.arguments || '{}')
            } catch (parseErr) {
              resultText = `Invalid tool arguments: ${(parseErr as Error).message}`
              emitToolStatus('error', tc.function.name, resultText, toolIteration)
              requestMessages.push(useResponsesApi
                ? { type: 'function_call_output', call_id: tc.id, output: resultText }
                : { role: 'tool', tool_call_id: tc.id, content: resultText })
              continue
            }
            emitToolStatus('executing', tc.function.name, undefined, toolIteration)

            if (tc.function.name === 'ask_user') {
              // Special handling: ask_user needs IPC to renderer, not executor
              const question = toolArgs.question || 'What would you like to do?'
              emitToolStatus('asking', 'ask_user', question, toolIteration)
              resultText = await new Promise<string>((resolve) => {
                const win = getWindow()
                if (!win || win.isDestroyed()) { resolve('(User interface not available)'); return }
                if (abortController.signal.aborted) { resolve('(Generation was stopped)'); return }
                win.webContents.send('chat:askUser', { chatId, question })
                // Use Map-based resolver (single global listener, no per-call listener accumulation)
                const cleanup = () => {
                  askUserResolvers.delete(chatId)
                  clearTimeout(askTimeout)
                  abortController.signal.removeEventListener('abort', onAbort)
                }
                askUserResolvers.set(chatId, (answer: string) => {
                  cleanup()
                  resolve(answer)
                })
                const onAbort = () => {
                  cleanup()
                  resolve('(Generation was stopped)')
                }
                abortController.signal.addEventListener('abort', onAbort, { once: true })
                const askTimeout = setTimeout(() => {
                  cleanup()
                  resolve('(User did not respond within 5 minutes)')
                }, 300000)
              })
              emitToolStatus('result', 'ask_user', resultText, toolIteration)
            } else if (isBuiltinTool(tc.function.name)) {
              // Enforce tool category toggles at execution time (defense-in-depth:
              // filterTools removes disabled tools from definitions sent to model,
              // but models can hallucinate tool calls not in the provided list)
              const disabledSet = getDisabledTools(overrides || {})
              if (disabledSet.has(tc.function.name)) {
                resultText = `Tool "${tc.function.name}" is disabled in chat settings.`
                emitToolStatus('error', tc.function.name, resultText, toolIteration)
              } else if (!(overrides?.workingDirectory)) {
                resultText = 'Error: Working directory not set. Configure it in Chat Settings.'
                emitToolStatus('error', tc.function.name, resultText, toolIteration)
              } else {
                const workDir = overrides.workingDirectory
                console.log(`[CHAT] Builtin tool: ${tc.function.name}`)
                const result = await executeBuiltinTool(tc.function.name, toolArgs, workDir, overrides?.toolResultMaxChars)
                resultText = result.content
                // For read_image: inject image as multimodal content for VLM follow-ups
                if (result.imageDataUrl) {
                  pendingImageDataUrls.push(result.imageDataUrl)
                }
                emitToolStatus(result.is_error ? 'error' : 'result', tc.function.name, resultText, toolIteration)
              }
            } else if (isRemote) {
              // MCP tool passthrough is only available on local vmlx-engine servers
              resultText = `MCP tool "${tc.function.name}" is only available with local vmlx-engine sessions.`
              emitToolStatus('error', tc.function.name, resultText, toolIteration)
            } else {
              const execRes = await fetch(`${baseUrl}/v1/mcp/execute`, {
                method: 'POST',
                headers: {
                  'Content-Type': 'application/json',
                  ...authHeaders
                },
                body: JSON.stringify({ tool_name: tc.function.name, arguments: toolArgs }),
                signal: abortController.signal
              })
              if (!execRes.ok) {
                const errText = await execRes.text()
                resultText = `Error (${execRes.status}): ${errText}`
                emitToolStatus('error', tc.function.name, resultText, toolIteration)
              } else {
                const result = await execRes.json()
                if (result.is_error) {
                  resultText = `Error: ${result.error_message || 'Unknown error'}`
                  emitToolStatus('error', tc.function.name, resultText, toolIteration)
                } else {
                  resultText = typeof result.content === 'string'
                    ? result.content
                    : JSON.stringify(result.content, null, 2)
                  // Apply same truncation as built-in tools to prevent context overflow
                  const mcpMaxChars = overrides?.toolResultMaxChars || 50000
                  if (resultText.length > mcpMaxChars) {
                    resultText = resultText.slice(0, mcpMaxChars) + `\n\n[Truncated — showing first ${mcpMaxChars} of ${resultText.length} characters]`
                  }
                  emitToolStatus('result', tc.function.name, resultText, toolIteration)
                }
              }
            }
          } catch (err: any) {
            if (err?.name === 'AbortError') throw err
            resultText = `Tool execution error: ${err.message}`
            emitToolStatus('error', tc.function.name, err.message, toolIteration)
          }

          requestMessages.push(useResponsesApi
            ? { type: 'function_call_output', call_id: tc.id, output: resultText }
            : { role: 'tool', tool_call_id: tc.id, content: resultText })
        }

        // Inject images from read_image tool results as multimodal content parts.
        // VL models can only process images in content arrays, not in tool result strings.
        // Text FIRST, then images — Qwen3.5-VL expects this order.
        if (pendingImageDataUrls.length > 0) {
          const contentParts: any[] = [
            { type: 'text', text: 'Here are the images from the tool results above.' },
            ...pendingImageDataUrls.map(url => ({
              type: 'image_url',
              image_url: { url }
            }))
          ]
          requestMessages.push({ role: 'user', content: contentParts })
          console.log(`[CHAT] Injected ${pendingImageDataUrls.length} image(s) as multimodal content for VLM`)
        }
      }

      console.log(`[CHAT] Stream ended — content: ${fullContent.length} chars, reasoning: ${reasoningContent.length} chars, tool calls: ${receivedToolCalls.length}, buffered: ${clientToolCallBuffering}`)

      // ─── Unified Tool Execution + Auto-Continue Loop ───────────────────
      // Handles both tool call execution and auto-continuation for models
      // that stop after tool use without providing a response.
      // Auto-continue is limited to MAX_AUTO_CONTINUES consecutive attempts.
      // Resets after each successful tool call round.
      const AUTO_CONTINUE_TOKEN_THRESHOLD = 100
      const MAX_AUTO_CONTINUES = 3
      let autoContinueCount = 0
      while (toolIteration < MAX_TOOL_ITERATIONS) {
        // Compact sparse array: parallel tool calls at non-contiguous indices create holes
        // that for...of silently skips. Filter to only real entries.
        if (receivedToolCalls.length > 0) {
          receivedToolCalls = receivedToolCalls.filter(Boolean)
        }
        if (receivedToolCalls.length > 0) {
          // ── Model made tool calls: execute and send follow-up ──
          toolIteration++
          autoContinueCount = 0 // reset — model is making progress
          console.log(`[CHAT] Tool execution iteration ${toolIteration} (${receivedToolCalls.length} tool calls)`)
          // Preserve content before tool execution so abort can recover it
          if (fullContent.trim()) {
            allGeneratedContent += (allGeneratedContent ? '\n\n' : '') + fullContent.trim()
          }
          // Flush accumulated content to renderer before blocking on tool execution
          try {
            const win = getWindow()
            if (win && !win.isDestroyed() && allGeneratedContent.trim()) {
              win.webContents.send('chat:stream', {
                chatId,
                messageId: assistantMessage.id,
                fullContent: allGeneratedContent,
                isReasoning: false,
                metrics: {
                  tokenCount: cumulativeTokenOffset + iterationTokenCount,
                  promptTokens,
                  tokensPerSecond: liveTps.toFixed(1),
                  ttft: firstTokenTime ? ((firstTokenTime - fetchStartTime) / 1000).toFixed(2) : '0',
                  elapsed: (generationMs / 1000).toFixed(1)
                }
              })
            }
          } catch (_) { }
          // Reset idle timer before tool execution — builtin tools run locally
          // without server contact, so the model could sleep during long tool runs
          if (chatSession) sessionManager.touchSession(chatSession.id)
          await executeToolCalls()
          receivedToolCalls = []
          fullContent = ''
          rawAccumulated = ''
          lastFinishReason = undefined // Reset for next iteration
          // Reset content offset tracker to match the accumulated content position
          lastEmittedContentLength = allGeneratedContent.length ? allGeneratedContent.length + 2 : 0
          clientToolCallBuffering = false
          clientSideThinkParsing = false
          cumulativeTokenOffset += iterationTokenCount // Save completed iteration tokens for cumulative total
          iterationTokenBase = tokenCount // Save cumulative base for server-usage delta
          iterationTokenCount = 0
          tpsSnapshots.length = 0; liveTps = 0; tpsTokenBase = tokenCount // Reset rolling TPS for fresh generation phase
          serverSendsUsage = false // Re-detect for new HTTP request (server restarts completion_tokens from 1)
          // Fire reasoningDone if model was still in reasoning mode when tool calls appeared
          if (isReasoning && reasoningContent) {
            try {
              const win = getWindow()
              if (win && !win.isDestroyed()) {
                win.webContents.send('chat:reasoningDone', { chatId, messageId: assistantMessage.id, reasoningContent })
              }
            } catch (_) { }
          }
          isReasoning = false // Reset reasoning state for new iteration
          reasoningContent = '' // Reset so next iteration's reasoning doesn't accumulate with previous
          // (thinking indicator removed)
          emitToolStatus('processing', '', undefined, toolIteration)
          // Reset idle timer before follow-up — tools may have consumed minutes
          if (chatSession) sessionManager.touchSession(chatSession.id)
          if (!await sendFollowUp()) break

        } else if (
          toolIteration > 0 &&
          autoContinueCount < MAX_AUTO_CONTINUES &&
          (fullContent.trim().length === 0 || iterationTokenCount < AUTO_CONTINUE_TOKEN_THRESHOLD)
        ) {
          // ── Auto-continue: model stopped without a substantive response after tool use ──
          // This handles two cases:
          // 1. Model generated ZERO content after tool results (just stopped)
          // 2. Model generated a brief/incomplete response (< threshold tokens)
          autoContinueCount++
          const hasContent = fullContent.trim().length > 0
          console.log(`[CHAT] Auto-continue ${autoContinueCount}/${MAX_AUTO_CONTINUES}: model stopped with ${iterationTokenCount} tokens (iteration), content=${hasContent}`)
          if (hasContent) {
            allGeneratedContent += (allGeneratedContent ? '\n\n' : '') + fullContent.trim()
            if (useResponsesApi) {
              requestMessages.push({ type: 'output_text', text: fullContent })
            } else {
              requestMessages.push({ role: 'assistant', content: fullContent })
            }
          }
          const continuePrompt = 'Based on the tool results above, provide your complete response. Summarize what you found, explain the results, and address my original request.'
          if (useResponsesApi) {
            requestMessages.push({ type: 'message', role: 'user', content: continuePrompt })
          } else {
            requestMessages.push({ role: 'user', content: continuePrompt })
          }
          fullContent = ''
          rawAccumulated = ''
          lastFinishReason = undefined // Reset for next iteration
          clientToolCallBuffering = false
          clientSideThinkParsing = false
          receivedToolCalls = []
          // Reset content offset tracker to match the accumulated content position
          lastEmittedContentLength = allGeneratedContent.length ? allGeneratedContent.length + 2 : 0
          cumulativeTokenOffset += iterationTokenCount // Save completed iteration tokens for cumulative total
          iterationTokenBase = tokenCount // Save cumulative base for server-usage delta
          iterationTokenCount = 0
          tpsSnapshots.length = 0; liveTps = 0; tpsTokenBase = tokenCount // Reset rolling TPS for fresh generation phase
          serverSendsUsage = false // Re-detect for new HTTP request (server restarts completion_tokens from 1)
          // Fire reasoningDone if model was still in reasoning mode at auto-continue boundary
          if (isReasoning && reasoningContent) {
            try {
              const win = getWindow()
              if (win && !win.isDestroyed()) {
                win.webContents.send('chat:reasoningDone', { chatId, messageId: assistantMessage.id, reasoningContent })
              }
            } catch (_) { }
          }
          isReasoning = false // Reset reasoning state for new iteration
          reasoningContent = '' // Reset so next iteration's reasoning doesn't accumulate with previous
          // (thinking indicator removed)
          emitToolStatus('processing', '', 'Generating response...', toolIteration)
          // Reset idle timer before auto-continue follow-up
          if (chatSession) sessionManager.touchSession(chatSession.id)
          if (!await sendFollowUp()) break

        } else {
          break
        }
      }

      if (toolIteration > 0) {
        console.log(`[CHAT] Tool loop completed after ${toolIteration} iteration(s)`)
        emitToolStatus('done', '', undefined, toolIteration)
      }

      // Fire reasoningDone if stream ended while still in reasoning mode
      // (e.g., model only produced analysis channel, never transitioned to final)
      if (isReasoning) {
        isReasoning = false
        try {
          const win = getWindow()
          if (win && !win.isDestroyed()) {
            win.webContents.send('chat:reasoningDone', { chatId, messageId: assistantMessage.id, reasoningContent })
          }
        } catch (_) { }
      }

      // Calculate final metrics — use generation-only time for t/s, fallback to wall clock
      const totalTime = (Date.now() - startTime) / 1000
      const genTimeSec = generationMs > 0 ? generationMs / 1000 : 0
      const wallTimeSec = firstTokenTime && lastTokenTime && lastTokenTime > firstTokenTime
        ? (lastTokenTime - firstTokenTime) / 1000
        : (firstTokenTime ? (Date.now() - firstTokenTime) / 1000 : totalTime)
      const finalGenSec = genTimeSec > 0.05 ? genTimeSec : wallTimeSec
      // Use cumulative total across all tool iterations (server restarts completion_tokens per request)
      const totalTokenCount = cumulativeTokenOffset + iterationTokenCount
      const finalTps = finalGenSec > 0 ? totalTokenCount / finalGenSec : 0
      // TTFT measured from fetchStartTime (excludes health check and message building overhead)
      const ttft = Math.max(0, firstTokenTime ? (firstTokenTime - fetchStartTime) / 1000 : 0)
      // Guard against Infinity when TTFT is near zero (e.g., prefix cache hit)
      const finalPpSpeed = (promptTokens > 0 && ttft > 0.001)
        ? (promptTokens / ttft).toFixed(1)
        : undefined

      // Combine content from all tool iterations into the final message
      if (allGeneratedContent && fullContent.trim()) {
        fullContent = allGeneratedContent + '\n\n' + fullContent
      } else if (allGeneratedContent && !fullContent.trim()) {
        fullContent = allGeneratedContent
      }

      // Strip any remaining template tokens and leaked tool call XML
      fullContent = fullContent.replace(TEMPLATE_TOKEN_REGEX, '')
      // Strip Harmony protocol residue (concatenated protocol words after template token removal)
      fullContent = fullContent.replace(/<\/?(?:assistant|analysis|final)+/gi, '')
      fullContent = fullContent.replace(/(?:assistant\s*){1,3}(?:analysis|final)/gi, '')
      fullContent = fullContent.replace(/(?:analysis|final)\s*(?:assistant\s*){1,3}/gi, '')
      // Strip leaked tool call blocks that server didn't parse (various model formats)
      fullContent = fullContent.replace(/<minimax:tool_call>[\s\S]*?<\/minimax:tool_call>/g, '')
      fullContent = fullContent.replace(/<tool_call>[\s\S]*?<\/tool_call>/g, '')
      fullContent = fullContent.replace(/\[Calling tool:\s*\w+\(\{[\s\S]*?\}\)\]/g, '')
      fullContent = fullContent.replace(/<invoke\b[^>]*>[\s\S]*?<\/invoke>/g, '')
      fullContent = fullContent.replace(/<parameter\b[^>]*>[\s\S]*?<\/parameter>/g, '')
      // Strip hallucinated Claude-style tool calls (models trained on Anthropic data)
      fullContent = fullContent.replace(/<(?:read_file|write_file|run_command|search_files|edit_file|list_directory|execute_command|bash)\b[^>]*>[\s\S]*?(?:<\/(?:read_file|write_file|run_command|search_files|edit_file|list_directory|execute_command|bash)>|$)/g, '')
      // Strip self-closing hallucinated tool calls like <read_file path="..." />
      fullContent = fullContent.replace(/<(?:read_file|write_file|run_command|search_files|edit_file|list_directory|execute_command|bash)\b[^>]*\/>/g, '')
      // Strip leaked Harmony protocol channel markers (GLM, GPT-OSS)
      fullContent = fullContent.replace(/<\|start\|>assistant/g, '')
      fullContent = fullContent.replace(/<\|channel\|>(?:analysis|final)<\|message\|>/g, '')
      stopPeriodicSave() // Stop periodic saves — final save below overwrites with complete content
      fullContent = fullContent.trim()
      // If no main content but reasoning was produced, keep them separate.
      // Reasoning stays in reasoningContent for the reasoning box; content stays empty.
      // (Previously this did fullContent = reasoningContent which triggered the anti-dup
      // check in MessageBubble, hiding the reasoning box.)
      if (!fullContent && reasoningContent) {
        console.log(`[CHAT] No main content — reasoning only (${reasoningContent.length} chars)`)
      }
      assistantMessage.content = fullContent
      assistantMessage.tokens = totalTokenCount
      assistantMessage.metricsJson = JSON.stringify({
        tokenCount: totalTokenCount,
        promptTokens: promptTokens || undefined,
        cachedTokens: cachedTokens || undefined,
        tokensPerSecond: finalTps.toFixed(1),
        ppSpeed: finalPpSpeed,
        ttft: ttft.toFixed(2),
        totalTime: totalTime.toFixed(1)
      })
      if (collectedToolStatuses.length > 0) {
        assistantMessage.toolCallsJson = JSON.stringify(collectedToolStatuses)
      }
      if (reasoningContent) {
        assistantMessage.reasoningContent = reasoningContent
      }
      db.addMessage(assistantMessage)

      // Send final metrics
      try {
        const win = getWindow()
        if (win && !win.isDestroyed()) {
          win.webContents.send('chat:complete', {
            chatId,
            messageId: assistantMessage.id,
            content: fullContent,
            reasoningContent: reasoningContent || undefined,
            finishReason: lastFinishReason,
            metrics: {
              tokenCount: totalTokenCount,
              promptTokens,
              cachedTokens,
              tokensPerSecond: finalTps.toFixed(1),
              ppSpeed: finalPpSpeed,
              ttft: ttft.toFixed(2),
              totalTime: totalTime.toFixed(1)
            }
          })
        }
      } catch (_) { }

      console.log(`[CHAT] Response complete: ${totalTokenCount} tokens in ${totalTime.toFixed(1)}s (${finalTps.toFixed(1)} t/s, live=${liveTps.toFixed(1)} t/s, TTFT: ${ttft.toFixed(2)}s${promptTokens ? `, pp: ${promptTokens} tokens${cachedTokens ? ` (${cachedTokens} cached)` : ''}, ${finalPpSpeed} pp/s` : ''}, usage=${serverSendsUsage ? 'server' : 'client'})`)

      return assistantMessage
    } catch (error) {
      stopPeriodicSave()
      // Release the SSE reader if it was acquired
      try { reader?.cancel() } catch (_) { }

      const _err = error as any
      console.error('[CHAT] Error caught:', {
        message: _err?.message,
        name: _err?.name,
        code: _err?.code,
        type: _err?.constructor?.name,
        stack: _err?.stack?.split('\n').slice(0, 5).join('\n'),
        abortSignal: abortController.signal.aborted,
        timedOut,
        fullContentLen: fullContent?.length,
        readerAcquired: !!reader
      })

      // Fire reasoningDone if interrupted during reasoning mode
      if (isReasoning) {
        isReasoning = false
        try {
          const win = getWindow()
          if (win && !win.isDestroyed()) {
            win.webContents.send('chat:reasoningDone', { chatId, messageId: assistantMessage.id, reasoningContent })
          }
        } catch (_) { }
      }

      // Save partial response: combine all content from previous tool iterations + current.
      // allGeneratedContent holds text from completed iterations; fullContent has current iteration.
      const abortFinishReason = lastFinishReason ?? null
      let partialContent = ''
      if (allGeneratedContent.trim() && fullContent.trim()) {
        partialContent = allGeneratedContent.trim() + '\n\n' + fullContent.trim()
      } else if (allGeneratedContent.trim()) {
        partialContent = allGeneratedContent.trim()
      } else {
        partialContent = fullContent.trim()
      }
      if (partialContent) {
        partialContent = partialContent.replace(TEMPLATE_TOKEN_REGEX, '')
        partialContent = partialContent.replace(/<minimax:tool_call>[\s\S]*?<\/minimax:tool_call>/g, '')
        partialContent = partialContent.replace(/<tool_call>[\s\S]*?<\/tool_call>/g, '')
        partialContent = partialContent.replace(/\[Calling tool:\s*\w+\(\{[\s\S]*?\}\)\]/g, '')
        partialContent = partialContent.replace(/<invoke\b[^>]*>[\s\S]*?<\/invoke>/g, '')
        partialContent = partialContent.replace(/<parameter\b[^>]*>[\s\S]*?<\/parameter>/g, '')
        partialContent = partialContent.replace(/<(?:read_file|write_file|run_command|search_files|edit_file|list_directory|execute_command|bash)\b[^>]*>[\s\S]*?(?:<\/(?:read_file|write_file|run_command|search_files|edit_file|list_directory|execute_command|bash)>|$)/g, '')
        partialContent = partialContent.replace(/<(?:read_file|write_file|run_command|search_files|edit_file|list_directory|execute_command|bash)\b[^>]*\/>/g, '')
        partialContent = partialContent.replace(/<\|start\|>assistant/g, '')
        partialContent = partialContent.replace(/<\|channel\|>(?:analysis|final)<\|message\|>/g, '')
        partialContent = partialContent.trim()
      }
      // Save message if we have any content OR reasoning (reasoning stays separate)
      if (partialContent || reasoningContent.trim()) {
        assistantMessage.content = partialContent
          ? partialContent + '\n\n[Generation interrupted]'
          : '[Generation interrupted]'
        const abortTotalTokens = cumulativeTokenOffset + iterationTokenCount
        assistantMessage.tokens = abortTotalTokens

        // Calculate real metrics for the partial generation (not hardcoded zeros)
        const abortTotalTime = (Date.now() - startTime) / 1000
        const abortGenSec = generationMs > 50 ? generationMs / 1000
          : (firstTokenTime ? (Date.now() - firstTokenTime) / 1000 : abortTotalTime)
        const abortTps = (abortGenSec > 0 && abortTotalTokens > 0) ? abortTotalTokens / abortGenSec : 0
        // Use fetchStartTime for TTFT (consistent with non-abort path)
        const abortTtft = firstTokenTime ? (firstTokenTime - fetchStartTime) / 1000 : 0
        const abortPpSpeed = (promptTokens > 0 && abortTtft > 0.001)
          ? (promptTokens / abortTtft).toFixed(1)
          : undefined

        const abortMetrics = {
          tokenCount: abortTotalTokens,
          promptTokens: promptTokens || undefined,
          cachedTokens: cachedTokens || undefined,
          tokensPerSecond: abortTps.toFixed(1),
          ppSpeed: abortPpSpeed,
          ttft: abortTtft.toFixed(2),
          totalTime: abortTotalTime.toFixed(1)
        }

        // Persist metricsJson to DB so reloading the chat shows real stats
        assistantMessage.metricsJson = JSON.stringify(abortMetrics)
        if (collectedToolStatuses.length > 0) {
          assistantMessage.toolCallsJson = JSON.stringify(collectedToolStatuses)
        }
        if (reasoningContent) {
          assistantMessage.reasoningContent = reasoningContent
        }
        db.addMessage(assistantMessage)

        try {
          const win = getWindow()
          if (win && !win.isDestroyed()) {
            win.webContents.send('chat:complete', {
              chatId,
              messageId: assistantMessage.id,
              content: assistantMessage.content,
              reasoningContent: reasoningContent || undefined,
              finishReason: abortFinishReason,
              metrics: abortMetrics
            })
          }
        } catch (_) { }
      } else {
        // No content generated — remove the pre-inserted empty placeholder row
        try { db.deleteMessage(assistantMessage.id) } catch (_) { }
      }

      // Distinguish timeout from user-initiated abort for better error messages.
      // CRITICAL: Check abortController.signal.aborted FIRST — when abort fires during
      // reader.read(), the error message can be 'terminated' instead of 'AbortError',
      // which would be misclassified as "server connection lost".
      const wasAborted = abortController.signal.aborted
      const errMsg = (error as Error).message || ''
      if (timedOut) {
        throw new Error(`Request timed out after ${timeoutSeconds}s. Increase the Timeout setting in Session Config, or the model may be overloaded.`)
      }
      if (wasAborted) {
        // User-initiated abort: return normally so the renderer's success path handles it.
        // Content (if any) was already saved to DB and chat:complete event sent above.
        console.log(`[CHAT] Abort complete — saved ${partialContent ? partialContent.length : 0} chars`)
        return (partialContent || reasoningContent.trim()) ? assistantMessage : null
      }
      // Check both error message AND error code — Node.js ConnResetException has
      // message "aborted" but code "ECONNRESET", which the message-only check missed.
      const errCode = (error as any)?.code || ''
      if (errMsg === 'terminated' || errMsg === 'aborted'
        || errMsg.includes('ECONNREFUSED') || errMsg.includes('ECONNRESET')
        || errCode === 'ECONNRESET' || errCode === 'ECONNREFUSED'
        || errMsg.includes('Connection closed before response completed')
        || errMsg.includes('socket hang up')) {
        throw new Error(`Server connection lost. The model server may have crashed or stopped. Try restarting the session.`)
      }
      throw new Error(`Failed to send message: ${errMsg}`)
    } finally {
      // Always clean up the active request tracker and periodic save
      stopPeriodicSave()
      clearTimeout(fetchTimeout)
      activeRequests.delete(chatId)
    }
  })

  // B5: Abort active generation for a chat
  ipcMain.handle('chat:abort', async (_, chatId: string) => {
    const entry = activeRequests.get(chatId)
    if (entry) {
      console.log(`[CHAT] Aborting generation for chat ${chatId}`)
      // 1. Abort the SSE fetch stream
      try { entry.controller.abort() } catch (_) { }

      // 2. Tell the server to cancel inference (frees GPU immediately)
      if (entry.responseId && (entry.endpoint || entry.baseUrl)) {
        try {
          // Route to correct cancel endpoint based on response ID prefix
          const cancelPath = entry.responseId.startsWith('resp_')
            ? `/v1/responses/${entry.responseId}/cancel`
            : `/v1/chat/completions/${entry.responseId}/cancel`
          const cancelBase = entry.baseUrl || `http://${connectHost(entry.endpoint!.host)}:${entry.endpoint!.port}`
          const cancelRes = await fetch(
            `${cancelBase}${cancelPath}`,
            { method: 'POST', headers: entry.authHeaders || {}, signal: AbortSignal.timeout(2000) }
          )
          console.log(`[CHAT] Server cancel sent for ${entry.responseId} — status ${cancelRes.status}`)
        } catch (cancelErr: any) {
          console.log(`[CHAT] Server cancel failed for ${entry.responseId}: ${cancelErr.message || cancelErr}`)
        }
      } else if (!entry.responseId) {
        // Abort during prefill: responseId not assigned yet. The fetch abort (step 1)
        // closes the connection; the server will detect disconnect via is_disconnected()
        // on the next token yield. No explicit cancel needed — prefill is typically <2s.
        console.log(`[CHAT] Abort during prefill (no responseId yet) — connection closed, server will detect disconnect`)
      }

      activeRequests.delete(chatId)
      return { success: true }
    }
    return { success: false, error: 'No active request for this chat' }
  })

  // Check if a chat has an active streaming generation (used for re-sync on tab switch)
  ipcMain.handle('chat:isStreaming', (_, chatId: string) => {
    return activeRequests.has(chatId)
  })

  // Clear all active locks (called on window reload/close)
  ipcMain.handle('chat:clearAllLocks', async () => {
    const count = activeRequests.size
    for (const [chatId, entry] of activeRequests) {
      // 1. Abort the SSE fetch stream
      try { entry.controller.abort() } catch (_) { }
      // 2. Send server-side cancel to free GPU (same logic as chat:abort)
      if (entry.responseId && (entry.endpoint || entry.baseUrl)) {
        try {
          const cancelPath = entry.responseId.startsWith('resp_')
            ? `/v1/responses/${entry.responseId}/cancel`
            : `/v1/chat/completions/${entry.responseId}/cancel`
          const cancelBase = entry.baseUrl || `http://${connectHost(entry.endpoint!.host)}:${entry.endpoint!.port}`
          fetch(`${cancelBase}${cancelPath}`, {
            method: 'POST', headers: entry.authHeaders || {}, signal: AbortSignal.timeout(2000)
          }).catch(() => {}) // Fire-and-forget, don't block window close
          console.log(`[CHAT] clearAllLocks: cancel sent for ${chatId} (${entry.responseId})`)
        } catch (_) { }
      }
    }
    activeRequests.clear()
    return { cleared: count }
  })

  // Overrides — validate numeric bounds to prevent garbage values from reaching the engine
  ipcMain.handle('chat:setOverrides', async (_, chatId: string, overrides: any) => {
    const clamp = (v: any, lo: number, hi: number) => typeof v === 'number' ? Math.max(lo, Math.min(hi, v)) : v
    const sanitized = { ...overrides }
    if (sanitized.temperature != null) sanitized.temperature = clamp(sanitized.temperature, 0, 10)
    if (sanitized.topP != null) sanitized.topP = clamp(sanitized.topP, 0, 1)
    if (sanitized.topK != null) sanitized.topK = clamp(sanitized.topK, 0, 1000)
    if (sanitized.minP != null) sanitized.minP = clamp(sanitized.minP, 0, 1)
    if (sanitized.maxTokens != null) sanitized.maxTokens = clamp(sanitized.maxTokens, 1, 1000000)
    if (sanitized.repeatPenalty != null) sanitized.repeatPenalty = clamp(sanitized.repeatPenalty, 0, 10)
    if (sanitized.maxToolIterations != null) sanitized.maxToolIterations = clamp(sanitized.maxToolIterations, 1, 100)
    if (sanitized.toolResultMaxChars != null) sanitized.toolResultMaxChars = clamp(sanitized.toolResultMaxChars, 100, 500000)
    db.setChatOverrides({ chatId, ...sanitized })
    return { success: true }
  })

  ipcMain.handle('chat:getOverrides', async (_, chatId: string) => {
    return db.getChatOverrides(chatId)
  })

  ipcMain.handle('chat:clearOverrides', async (_, chatId: string) => {
    db.clearChatOverrides(chatId)
    return { success: true }
  })
}
