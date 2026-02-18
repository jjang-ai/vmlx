import { ipcMain, BrowserWindow } from 'electron'
import { v4 as uuidv4 } from 'uuid'
import { db, Chat, Message, Folder } from '../database'
import { readFileSync, existsSync } from 'fs'
import { join } from 'path'
import { app } from 'electron'
import { sessionManager, resolveUrl } from '../sessions'
import { BUILTIN_TOOLS, isBuiltinTool, AGENTIC_SYSTEM_PROMPT } from '../tools/registry'
import { executeBuiltinTool } from '../tools/executor'
import { readGenerationDefaults } from './models'
import { detectModelConfigFromDir } from '../model-config-registry'

const CONFIG_DIR = join(app.getPath('userData'), 'config')
const CONFIG_FILE = join(CONFIG_DIR, 'server-config.json')

// Default config if file doesn't exist
const DEFAULT_CONFIG = {
  host: '127.0.0.1',
  port: 8093,
  apiKey: '',
  maxTokens: 4096
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

// Tool category definitions for per-category filtering
const FILE_TOOLS = new Set(['read_file', 'write_file', 'edit_file', 'patch_file', 'batch_edit', 'copy_file', 'move_file', 'delete_file', 'create_directory', 'list_directory'])
const SEARCH_TOOLS = new Set(['search_files', 'find_files', 'file_info', 'get_diagnostics'])
const SHELL_TOOLS = new Set(['run_command'])
const DDG_SEARCH_TOOLS = new Set(['ddg_search'])
const FETCH_TOOLS = new Set(['fetch_url'])

/** Filter BUILTIN_TOOLS based on per-category toggle overrides */
function filterTools(overrides: any): any[] {
  const disabled = new Set<string>()
  if (overrides.fileToolsEnabled === false) FILE_TOOLS.forEach(t => disabled.add(t))
  if (overrides.searchToolsEnabled === false) SEARCH_TOOLS.forEach(t => disabled.add(t))
  if (overrides.shellEnabled === false) SHELL_TOOLS.forEach(t => disabled.add(t))
  if (overrides.webSearchEnabled === false) DDG_SEARCH_TOOLS.forEach(t => disabled.add(t))
  if (overrides.fetchUrlEnabled === false) FETCH_TOOLS.forEach(t => disabled.add(t))
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
  if (disabled.size === 0) return BUILTIN_TOOLS
  return BUILTIN_TOOLS.filter((t: any) => !disabled.has(t.function.name))
}

// Track active requests per chat for abort/concurrency (B5/B6)
const activeRequests = new Map<string, { controller: AbortController; startedAt: number; timeoutMs: number; responseId?: string; endpoint?: { host: string; port: number }; baseUrl?: string; authHeaders?: Record<string, string> }>()
// Stale lock: each request stores its timeoutMs; stale check uses timeoutMs + 30s buffer

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
    return { host: '127.0.0.1', port: healthy.port }
  }

  // 3. Fallback to config file
  try {
    if (existsSync(CONFIG_FILE)) {
      const config = JSON.parse(readFileSync(CONFIG_FILE, 'utf-8'))
      return { host: config.host || '127.0.0.1', port: config.port || 8093 }
    }
  } catch (_) { }

  return { host: '127.0.0.1', port: 8093 }
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

    // Inherit chat overrides: prefer most recent sibling chat's settings, else model defaults
    if (modelPath) {
      let inherited = false
      try {
        const siblings = db.getChatsByModelPath(modelPath)
        // Find the most recent OTHER chat (sorted by updatedAt DESC) that has overrides
        for (const sib of siblings) {
          if (sib.id === chat.id) continue
          const sibOverrides = db.getChatOverrides(sib.id)
          if (sibOverrides) {
            db.setChatOverrides({
              chatId: chat.id,
              temperature: sibOverrides.temperature,
              topP: sibOverrides.topP,
              topK: sibOverrides.topK,
              minP: sibOverrides.minP,
              maxTokens: sibOverrides.maxTokens,
              repeatPenalty: sibOverrides.repeatPenalty,
              systemPrompt: sibOverrides.systemPrompt,
              stopSequences: sibOverrides.stopSequences,
              wireApi: sibOverrides.wireApi,
              maxToolIterations: sibOverrides.maxToolIterations,
              builtinToolsEnabled: sibOverrides.builtinToolsEnabled,
              workingDirectory: sibOverrides.workingDirectory,
              enableThinking: sibOverrides.enableThinking,
              reasoningEffort: sibOverrides.reasoningEffort,
              hideToolStatus: sibOverrides.hideToolStatus,
              webSearchEnabled: sibOverrides.webSearchEnabled,
              braveSearchEnabled: sibOverrides.braveSearchEnabled,
              fetchUrlEnabled: sibOverrides.fetchUrlEnabled,
              fileToolsEnabled: sibOverrides.fileToolsEnabled,
              searchToolsEnabled: sibOverrides.searchToolsEnabled,
              shellEnabled: sibOverrides.shellEnabled
            })
            console.log(`[CHAT] Inherited overrides from chat ${sib.id} for ${chat.id}`)
            inherited = true
            break
          }
        }
      } catch (e) {
        console.error('[CHAT] Failed to inherit overrides:', e)
      }

      // Fallback: read from model's generation_config.json
      if (!inherited) {
        try {
          const defaults = await readGenerationDefaults(modelPath)
          if (defaults) {
            db.setChatOverrides({
              chatId: chat.id,
              temperature: defaults.temperature,
              topP: defaults.topP,
              topK: defaults.topK,
              repeatPenalty: defaults.repeatPenalty
            })
            console.log(`[CHAT] Applied generation defaults for ${chat.id}:`, defaults)
          }
        } catch (e) {
          console.error('[CHAT] Failed to read generation defaults:', e)
        }
      }
    }

    return chat
  })

  ipcMain.handle('chat:getByModel', async (_, modelPath: string) => {
    return db.getChatsByModelPath(modelPath)
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
  ipcMain.handle('chat:sendMessage', async (_, chatId: string, content: string, endpoint?: { host: string; port: number }) => {
    // B6: Concurrency guard — reject if a request is already active for this chat
    // B6: Concurrency guard with stale lock recovery
    const existing = activeRequests.get(chatId)
    if (existing) {
      const age = Date.now() - existing.startedAt
      // Use the timeout configured when that request started, plus 30s buffer
      const staleLockMs = existing.timeoutMs + 30_000
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
    let chatSession: import('../database').Session | undefined
    if (chat.modelPath) {
      chatSession = sessionManager.getSessionByModelPath(chat.modelPath.replace(/\/+$/, ''))
      if (chatSession) {
        try {
          const sessionConfig = JSON.parse(chatSession.config)
          if (sessionConfig.timeout && sessionConfig.timeout > 0) {
            timeoutSeconds = sessionConfig.timeout
          }
          // Check if model has a reasoning parser (for enable_thinking default)
          if (sessionConfig.reasoningParser && sessionConfig.reasoningParser !== 'auto') {
            sessionHasReasoningParser = true
          } else if (sessionConfig.reasoningParser === 'auto' && chat.modelPath) {
            // "auto" means use detection
            const detected = detectModelConfigFromDir(chat.modelPath)
            sessionHasReasoningParser = !!detected.reasoningParser
          }
        } catch (_) { }
      }
    }
    const fetchTimeout = setTimeout(() => { timedOut = true; abortController.abort() }, timeoutSeconds * 1000)
    activeRequests.set(chatId, { controller: abortController, startedAt: Date.now(), timeoutMs: timeoutSeconds * 1000, endpoint: undefined, responseId: undefined })

    // Get config from file or defaults
    let config = DEFAULT_CONFIG
    try {
      if (existsSync(CONFIG_FILE)) {
        config = { ...DEFAULT_CONFIG, ...JSON.parse(readFileSync(CONFIG_FILE, 'utf-8')) }
      }
    } catch (e) {
      console.log('[CHAT] Using default config:', e)
    }

    // Resolve actual server endpoint: explicit endpoint > session by modelPath > detect > config
    // CRITICAL: When endpoint is passed from the renderer, attach the chatSession
    // so remote sessions get proper remoteUrl, auth headers, and health check path.
    const resolved = endpoint
      ? { host: endpoint.host, port: endpoint.port, session: chatSession } as ResolvedEndpoint
      : await resolveServerEndpoint(chat.modelPath)
    const server = resolved
    config = { ...config, host: server.host, port: server.port }

    // Detect remote session and compute base URL + auth headers
    const resolvedSession = resolved.session
    const isRemote = resolvedSession?.type === 'remote'
    const rawBaseUrl = isRemote && resolvedSession?.remoteUrl
      ? resolvedSession.remoteUrl.replace(/\/+$/, '')
      : `http://${config.host}:${config.port}`
    // Resolve .local mDNS hostnames to IPv4 — Node.js fetch resolves them to
    // unreachable IPv6 link-local addresses (fe80::...) causing "fetch failed"
    const baseUrl = await resolveUrl(rawBaseUrl)
    console.log(`[CHAT] Endpoint resolution: isRemote=${isRemote}, rawBaseUrl=${rawBaseUrl}, baseUrl=${baseUrl}, session=${resolvedSession?.id ?? 'none'}, type=${resolvedSession?.type ?? 'none'}`)
    const authHeaders: Record<string, string> = {}
    if (isRemote && resolvedSession?.remoteApiKey) {
      authHeaders['Authorization'] = `Bearer ${resolvedSession.remoteApiKey}`
      if (resolvedSession.remoteOrganization) {
        authHeaders['OpenAI-Organization'] = resolvedSession.remoteOrganization
      }
    } else if (config.apiKey) {
      authHeaders['Authorization'] = `Bearer ${config.apiKey}`
    }
    // Update active request entry with resolved baseUrl and auth for cancel support
    const activeEntry = activeRequests.get(chatId)
    if (activeEntry) {
      activeEntry.baseUrl = baseUrl
      if (Object.keys(authHeaders).length > 0) activeEntry.authHeaders = authHeaders
    }

    // Health check with retry — wait for server to become ready instead of
    // failing immediately. This prevents orphaned user messages and allows
    // chatting as soon as the server finishes loading.
    const maxHealthRetries = 5
    const healthRetryDelay = 2000  // 2 seconds between retries
    let healthOk = false
    const healthUrl = isRemote ? `${baseUrl}/v1/models` : `${baseUrl}/health`
    console.log(`[CHAT] Health check URL: ${healthUrl}`)
    for (let attempt = 0; attempt < maxHealthRetries; attempt++) {
      try {
        const healthRes = await fetch(healthUrl, { headers: authHeaders, signal: AbortSignal.timeout(5000) })
        if (healthRes.ok) {
          healthOk = true
          console.log(`[CHAT] Health check passed on attempt ${attempt + 1}`)
          break
        }
        // Server responded but not healthy — wait and retry
        if (attempt < maxHealthRetries - 1) {
          console.log(`[CHAT] Server not ready (HTTP ${healthRes.status}), retrying in ${healthRetryDelay}ms...`)
          await new Promise(r => setTimeout(r, healthRetryDelay))
        }
      } catch (healthErr: any) {
        // Connection failed — wait and retry
        console.log(`[CHAT] Health check failed (attempt ${attempt + 1}/${maxHealthRetries}): ${healthErr.message || healthErr.cause?.message || healthErr}`)
        if (attempt < maxHealthRetries - 1) {
          await new Promise(r => setTimeout(r, healthRetryDelay))
        }
      }
    }
    if (!healthOk) {
      activeRequests.delete(chatId)
      clearTimeout(fetchTimeout)
      throw new Error(`Cannot reach server${isRemote ? ' at ' + baseUrl : ' on port ' + config.port} after ${maxHealthRetries} attempts. Make sure the session is started and the model is loaded.`)
    }

    // Add user message AFTER health check passes — this prevents orphaned
    // user messages when the server isn't ready yet.
    const userMessage: Message = {
      id: uuidv4(),
      chatId,
      role: 'user',
      content,
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

    // Inject current date/time so the model knows when "now" is.
    // Placed at the END of the system prompt to maximize prefix cache hits
    // (the stable prompt prefix stays identical across days).
    const now = new Date()
    const dateStr = now.toLocaleDateString('en-US', { weekday: 'long', year: 'numeric', month: 'long', day: 'numeric' })
    const timeStr = now.toLocaleTimeString('en-US', { hour: '2-digit', minute: '2-digit', hour12: true })
    const dateSuffix = `\n\nCurrent date: ${dateStr}, ${timeStr}`

    // Add system prompt from overrides if available, or agentic prompt when built-in tools enabled
    const hasSystemPrompt = !!overrides?.systemPrompt
    if (hasSystemPrompt && overrides?.builtinToolsEnabled) {
      const toolRule = '\n\nIMPORTANT: After using any tools, you MUST always provide a substantive response explaining what you found or did. Never stop after just executing tools.'
      requestMessages.push({ role: 'system', content: overrides!.systemPrompt! + toolRule + dateSuffix })
    } else if (hasSystemPrompt) {
      requestMessages.push({ role: 'system', content: overrides!.systemPrompt! + dateSuffix })
    } else if (overrides?.builtinToolsEnabled) {
      requestMessages.push({ role: 'system', content: AGENTIC_SYSTEM_PROMPT + dateSuffix })
    } else {
      // No system prompt at all — inject a minimal one with just the date
      requestMessages.push({ role: 'system', content: `You are a helpful assistant.${dateSuffix}` })
    }

    // Add conversation messages (skip any existing system messages to avoid duplicates)
    for (const m of messages) {
      if (m.role === 'system' && (hasSystemPrompt || overrides?.builtinToolsEnabled)) continue
      requestMessages.push({ role: m.role, content: m.content })
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
    const tpsSnapshots: Array<[number, number]> = [] // [timestamp, cumulative tokenCount]
    let liveTps = 0
    // Throttle IPC emission to renderer (~30 fps for smooth streaming)
    let lastStreamEmitTime = 0
    const STREAM_THROTTLE_MS = 32
    let reader: ReadableStreamDefaultReader<Uint8Array> | undefined
    let fullContent = ''
    let reasoningContent = ''
    // Accumulates content across tool iterations so abort during tool execution can recover
    // earlier content that would otherwise be lost when fullContent is reset between iterations
    let allGeneratedContent = ''
    // Per-iteration token count for auto-continue threshold (tokenCount is cumulative)
    let iterationTokenCount = 0
    let iterationTokenBase = 0 // tokenCount at start of iteration (for server-usage delta)

    try {
      // Determine wire format: 'responses' or 'completions' (default)
      const wireApi = overrides?.wireApi || 'completions'
      const useResponsesApi = wireApi === 'responses'

      // Call API (local vLLM-MLX or remote OpenAI-compatible endpoint)
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

      // Build request body based on wire format
      let requestBody: string
      if (useResponsesApi) {
        // Responses API format: separate instructions from input messages
        // Use overrides system prompt, or extract system messages from conversation history
        const systemMessages = requestMessages.filter(m => m.role === 'system')
        const instructions = overrides?.systemPrompt || (systemMessages.length > 0 ? systemMessages.map(m => m.content).join('\n') : undefined)
        // Filter out system messages (instructions are separate in Responses API)
        const inputMessages = requestMessages.filter(m => m.role !== 'system')
        const responsesObj: Record<string, any> = {
          model: modelName,
          input: inputMessages,
          instructions,
          temperature: overrides?.temperature ?? 0.7,
          top_p: overrides?.topP ?? 0.9,
          max_output_tokens: overrides?.maxTokens ?? config.maxTokens ?? 4096,
          stream: true,
          stream_options: { include_usage: true }
        }
        if (stopSequences) responsesObj.stop = stopSequences
        if (overrides?.topK != null && overrides.topK > 0) responsesObj.top_k = overrides.topK
        if (overrides?.minP != null && overrides.minP > 0) responsesObj.min_p = overrides.minP
        if (overrides?.repeatPenalty != null && overrides.repeatPenalty !== 1.0) responsesObj.repetition_penalty = overrides.repeatPenalty
        if (overrides?.builtinToolsEnabled) {
          responsesObj.tools = filterTools(overrides)
        }
        // enable_thinking & reasoning_effort are vllm-mlx extensions — only send to local,
        // or when user explicitly enables them for a remote endpoint
        if (!isRemote) {
          responsesObj.enable_thinking = overrides?.enableThinking ?? sessionHasReasoningParser
          responsesObj.chat_template_kwargs = { enable_thinking: responsesObj.enable_thinking }
          if (overrides?.reasoningEffort) responsesObj.reasoning_effort = overrides.reasoningEffort
        } else if (overrides?.enableThinking != null) {
          // User explicitly toggled thinking for remote — send it (some providers support it)
          responsesObj.enable_thinking = overrides.enableThinking
          if (overrides?.reasoningEffort) responsesObj.reasoning_effort = overrides.reasoningEffort
        }
        requestBody = JSON.stringify(responsesObj)
      } else {
        const bodyObj: Record<string, any> = {
          model: modelName,
          messages: requestMessages,
          temperature: overrides?.temperature ?? 0.7,
          top_p: overrides?.topP ?? 0.9,
          max_tokens: overrides?.maxTokens ?? config.maxTokens ?? 4096,
          stream: true,
          stream_options: { include_usage: true }
        }
        if (stopSequences) bodyObj.stop = stopSequences
        if (overrides?.topK != null && overrides.topK > 0) bodyObj.top_k = overrides.topK
        if (overrides?.minP != null && overrides.minP > 0) bodyObj.min_p = overrides.minP
        if (overrides?.repeatPenalty != null && overrides.repeatPenalty !== 1.0) bodyObj.repetition_penalty = overrides.repeatPenalty
        if (overrides?.builtinToolsEnabled) {
          bodyObj.tools = filterTools(overrides)
        }
        // enable_thinking & reasoning_effort are vllm-mlx extensions — only send to local,
        // or when user explicitly enables them for a remote endpoint
        if (!isRemote) {
          bodyObj.enable_thinking = overrides?.enableThinking ?? sessionHasReasoningParser
          bodyObj.chat_template_kwargs = { enable_thinking: bodyObj.enable_thinking }
          if (overrides?.reasoningEffort) bodyObj.reasoning_effort = overrides.reasoningEffort
        } else if (overrides?.enableThinking != null) {
          bodyObj.enable_thinking = overrides.enableThinking
          if (overrides?.reasoningEffort) bodyObj.reasoning_effort = overrides.reasoningEffort
        }
        requestBody = JSON.stringify(bodyObj)
      }

      fetchStartTime = Date.now() // Capture just before fetch for accurate TTFT
      const response = await fetch(apiUrl, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          ...authHeaders
        },
        body: requestBody,
        signal: abortController.signal
      })

      if (!response.ok) {
        const errorText = await response.text()
        throw new Error(`API error: ${response.status} ${response.statusText} - ${errorText}`)
      }

      // Stream response
      reader = response.body?.getReader()
      if (!reader) throw new Error('Response body is null')

      const decoder = new TextDecoder()
      fullContent = ''
      reasoningContent = ''
      let isReasoning = false
      let lineBuffer = '' // Buffer for incomplete SSE lines across chunks
      let currentEventType = '' // Track SSE event type for Responses API

      // Track whether server sends real token counts (via usage in each SSE chunk)
      let serverSendsUsage = false

      // Track tool calls received during streaming for MCP auto-execution
      let receivedToolCalls: Array<{ id: string; function: { name: string; arguments: string } }> = []
      // Track tool iteration count (declared here so processLine closure can access it)
      const MAX_TOOL_ITERATIONS = overrides?.maxToolIterations ?? 10
      let toolIteration = 0

      // Helper: emit tool call status to renderer (separate from content stream)
      const emitToolStatus = (phase: string, toolName: string, detail?: string, iteration?: number) => {
        try {
          const win = getWindow()
          if (win && !win.isDestroyed()) {
            win.webContents.send('chat:toolStatus', {
              chatId,
              messageId: assistantMessage.id,
              phase,
              toolName,
              detail,
              iteration
            })
          }
        } catch (_) { }
      }

      // Client-side tool call buffering: suppress content when leaked tool call XML detected.
      // Must check RAW content before template token stripping, since markers like
      // <minimax:tool_call> get stripped by TEMPLATE_TOKEN_REGEX and never reach fullContent.
      let clientToolCallBuffering = false
      let rawAccumulated = '' // Tracks unstripped content for tool call detection

      // Helper: emit streaming delta to renderer
      const emitDelta = (delta: string, isReasoningDelta: boolean) => {
        // Track raw content BEFORE stripping for tool call marker detection
        if (!isReasoningDelta) {
          rawAccumulated += delta
          // Only activate buffering when tool call markers appear at the start of a line,
          // not when the model is explaining tool syntax in prose (e.g., "I'll use <tool_call>...")
          if (!clientToolCallBuffering) {
            // Catch real tool call formats AND hallucinated Claude-style tool calls
            const lineStartPattern = /(?:^|\n)\s*(?:<minimax:tool_call|<tool_call>|\[Calling tool:|<invoke name=|<read_file\b|<write_file\b|<run_command\b|<search_files\b|<edit_file\b|<list_directory\b|<execute_command\b|<bash\b)/
            if (lineStartPattern.test(rawAccumulated)) {
              clientToolCallBuffering = true
              console.log(`[CHAT] Client-side tool call buffering activated`)
            }
          }
        }

        // Strip any leaked chat template tokens from the delta
        delta = delta.replace(TEMPLATE_TOKEN_REGEX, '')
        if (!delta) return
        // Strip Harmony protocol residue: after <|start|>, <|channel|>, <|message|> are
        // removed above, words like "assistant", "analysis", "final" may remain concatenated.
        // Catches all garbled forms: </assistantassistantanalysis, assistantfinal, etc.
        delta = delta.replace(/<\/?(?:assistant|analysis|final)+/gi, '')
        delta = delta.replace(/(?:assistant\s*){1,3}(?:analysis|final)/gi, '')
        delta = delta.replace(/(?:analysis|final)\s*(?:assistant\s*){1,3}/gi, '')
        if (!delta) return
        // Strip U+FFFD replacement characters
        delta = delta.replace(/\uFFFD/g, '')
        if (!delta) return

        // === State updates (always, no throttle) ===
        const now = Date.now()
        if (firstTokenTime === null) firstTokenTime = now
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
        }
        // Client-side counting (fallback when server doesn't send usage in each chunk).
        // Must happen BEFORE TPS snapshot so the rolling window uses accurate counts.
        if (!serverSendsUsage) { tokenCount++; iterationTokenCount++ }

        // Rolling TPS: snapshot (timestamp, tokenCount) for accurate throughput.
        // Uses real token count deltas — handles multi-token SSE chunks correctly
        // (e.g., reasoning batches where server sends 2+ tokens per chunk).
        tpsSnapshots.push([now, tokenCount])
        if (tpsSnapshots.length > TPS_BUFFER_SIZE) tpsSnapshots.shift()
        if (tpsSnapshots.length >= 2) {
          const [oldT, oldN] = tpsSnapshots[0]
          const [newT, newN] = tpsSnapshots[tpsSnapshots.length - 1]
          const span = (newT - oldT) / 1000
          liveTps = span > 0.01 ? (newN - oldN) / span : liveTps
        }

        // Suppress rendering (but not counting/TPS) when tool call content is detected
        if (!isReasoningDelta && clientToolCallBuffering) return

        // === Throttled IPC emission (~31 fps, STREAM_THROTTLE_MS=32ms) ===
        // First token always emits immediately; subsequent tokens throttled to STREAM_THROTTLE_MS
        const isFirstContent = now - (firstTokenTime || now) < 50
        if (!isFirstContent && now - lastStreamEmitTime < STREAM_THROTTLE_MS) return
        lastStreamEmitTime = now

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
                tokenCount,
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
        if (!trimmed || !trimmed.startsWith('data: ')) return
        const data = trimmed.slice(6)
        if (data === '[DONE]') return

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
                entry.endpoint = { host: config.host, port: config.port }
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
                iterationTokenCount = tokenCount - iterationTokenBase
                serverSendsUsage = true
              }
              if (parsed.usage.input_tokens != null) promptTokens = parsed.usage.input_tokens
              if (parsed.usage.input_tokens_details?.cached_tokens) cachedTokens = parsed.usage.input_tokens_details.cached_tokens
            }

            // Final usage from response.completed event
            // Server wraps in { response: { usage: { input_tokens, output_tokens } } }
            const respUsage = parsed.response?.usage || parsed.usage
            if (currentEventType === 'response.completed' && respUsage) {
              if (respUsage.output_tokens != null) {
                tokenCount = respUsage.output_tokens
                iterationTokenCount = tokenCount - iterationTokenBase
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
                entry.endpoint = { host: config.host, port: config.port }
              }
            }

            // Update usage BEFORE emitting delta so metrics use real server counts
            if (parsed.usage) {
              if (parsed.usage.completion_tokens != null) {
                tokenCount = parsed.usage.completion_tokens
                iterationTokenCount = tokenCount - iterationTokenBase
                serverSendsUsage = true
              }
              if (parsed.usage.prompt_tokens != null) promptTokens = parsed.usage.prompt_tokens
              if (parsed.usage.prompt_tokens_details?.cached_tokens) cachedTokens = parsed.usage.prompt_tokens_details.cached_tokens
            }

            // Handle reasoning_content from reasoning parser
            const reasoning = choice?.reasoning_content || choice?.reasoning
            if (reasoning) {
              emitDelta(reasoning, true)
            }

            if (choice?.content) {
              emitDelta(choice.content, false)
            }

            // Handle tool_calls from streaming response
            // Supports both complete tool calls (vllm-mlx default) and incremental argument
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
                  emitToolStatus('calling', fn.name, fn.arguments || '{}', toolIteration)
                } else if (fn?.arguments && idx >= 0 && receivedToolCalls[idx]) {
                  // Incremental argument chunk: accumulate arguments for existing tool call
                  receivedToolCalls[idx].function.arguments += fn.arguments
                }
              }
            }
          }

          // Reset event type after processing data
          currentEventType = ''
        } catch (e) {
          // Skip malformed JSON — reset event type to avoid stale context
          currentEventType = ''
        }
      }

      while (true) {
        const { done, value } = await reader.read()
        if (done) break

        const chunk = decoder.decode(value, { stream: true })
        lineBuffer += chunk
        const lines = lineBuffer.split('\n')

        // Keep the last element as it may be incomplete
        lineBuffer = lines.pop() || ''

        for (const line of lines) {
          processLine(line.trim())
        }
      }

      // Flush remaining decoder bytes and process any leftover buffer
      const remaining = decoder.decode()
      if (remaining) lineBuffer += remaining
      if (lineBuffer.trim()) {
        processLine(lineBuffer.trim())
      }

      // ─── Helper: stream SSE response through processLine ──────────────
      const streamSSE = async (reader: ReadableStreamDefaultReader<Uint8Array>) => {
        const dec = new TextDecoder()
        let buf = ''
        while (true) {
          const { done, value } = await reader.read()
          if (done) break
          buf += dec.decode(value, { stream: true })
          const lines = buf.split('\n')
          buf = lines.pop() || ''
          for (const line of lines) processLine(line.trim())
        }
        const rem = dec.decode()
        if (rem) buf += rem
        if (buf.trim()) processLine(buf.trim())
      }

      // ─── Helper: build follow-up request body ──────────────────────────
      const buildFollowUpBody = (): Record<string, any> => {
        if (useResponsesApi) {
          // Responses API format
          const systemMessages = requestMessages.filter((m: any) => m.role === 'system')
          const instructions = overrides?.systemPrompt || (systemMessages.length > 0 ? systemMessages.map((m: any) => m.content).join('\n') : undefined)
          const inputMessages = requestMessages.filter((m: any) => m.role !== 'system')
          const obj: Record<string, any> = {
            model: modelName,
            input: inputMessages,
            instructions,
            temperature: overrides?.temperature ?? 0.7,
            top_p: overrides?.topP ?? 0.9,
            max_output_tokens: overrides?.maxTokens ?? config.maxTokens ?? 4096,
            stream: true,
            stream_options: { include_usage: true }
          }
          if (stopSequences) obj.stop = stopSequences
          if (overrides?.topK != null && overrides.topK > 0) obj.top_k = overrides.topK
          if (overrides?.minP != null && overrides.minP > 0) obj.min_p = overrides.minP
          if (overrides?.repeatPenalty != null && overrides.repeatPenalty !== 1.0) obj.repetition_penalty = overrides.repeatPenalty
          if (overrides?.builtinToolsEnabled) {
            obj.tools = filterTools(overrides)
          }
          if (!isRemote) {
            obj.enable_thinking = overrides?.enableThinking ?? sessionHasReasoningParser
            obj.chat_template_kwargs = { enable_thinking: obj.enable_thinking }
            if (overrides?.reasoningEffort) obj.reasoning_effort = overrides.reasoningEffort
          } else if (overrides?.enableThinking != null) {
            obj.enable_thinking = overrides.enableThinking
            if (overrides?.reasoningEffort) obj.reasoning_effort = overrides.reasoningEffort
          }
          return obj
        } else {
          // Chat Completions format
          const obj: Record<string, any> = {
            model: modelName,
            messages: requestMessages,
            temperature: overrides?.temperature ?? 0.7,
            top_p: overrides?.topP ?? 0.9,
            max_tokens: overrides?.maxTokens ?? config.maxTokens ?? 4096,
            stream: true,
            stream_options: { include_usage: true }
          }
          if (stopSequences) obj.stop = stopSequences
          if (overrides?.topK != null && overrides.topK > 0) obj.top_k = overrides.topK
          if (overrides?.minP != null && overrides.minP > 0) obj.min_p = overrides.minP
          if (overrides?.repeatPenalty != null && overrides.repeatPenalty !== 1.0) obj.repetition_penalty = overrides.repeatPenalty
          if (overrides?.builtinToolsEnabled) {
            obj.tools = filterTools(overrides)
          }
          if (!isRemote) {
            obj.enable_thinking = overrides?.enableThinking ?? sessionHasReasoningParser
            obj.chat_template_kwargs = { enable_thinking: obj.enable_thinking }
            if (overrides?.reasoningEffort) obj.reasoning_effort = overrides.reasoningEffort
          } else if (overrides?.enableThinking != null) {
            obj.enable_thinking = overrides.enableThinking
            if (overrides?.reasoningEffort) obj.reasoning_effort = overrides.reasoningEffort
          }
          return obj
        }
      }

      // ─── Helper: send follow-up request and stream response ────────────
      const sendFollowUp = async (): Promise<boolean> => {
        // Reset SSE parser state from previous stream
        currentEventType = ''
        // Use the same wire API format as the initial request
        const url = useResponsesApi
          ? `${baseUrl}/v1/responses`
          : `${baseUrl}/v1/chat/completions`
        const res = await fetch(url, {
          method: 'POST',
          headers: {
            'Content-Type': 'application/json',
            ...authHeaders
          },
          body: JSON.stringify(buildFollowUpBody()),
          signal: abortController.signal
        })
        if (!res.ok) {
          const errText = await res.text()
          console.log(`[CHAT] Follow-up failed: ${res.status} ${errText}`)
          emitToolStatus('error', '', `Follow-up error: ${res.status} ${errText}`, toolIteration)
          return false
        }
        const reader = res.body?.getReader()
        if (!reader) return false
        await streamSSE(reader)
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

            if (isBuiltinTool(tc.function.name)) {
              const workDir = overrides?.workingDirectory
              if (!workDir) {
                resultText = 'Error: Working directory not set. Configure it in Chat Settings.'
                emitToolStatus('error', tc.function.name, resultText, toolIteration)
              } else {
                console.log(`[CHAT] Builtin tool: ${tc.function.name}`)
                const result = await executeBuiltinTool(tc.function.name, toolArgs, workDir)
                resultText = result.content
                emitToolStatus(result.is_error ? 'error' : 'result', tc.function.name, resultText, toolIteration)
              }
            } else if (isRemote) {
              // MCP tool passthrough is only available on local vllm-mlx servers
              resultText = `MCP tool "${tc.function.name}" is only available with local vllm-mlx sessions.`
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
                  tokenCount,
                  promptTokens,
                  tokensPerSecond: liveTps.toFixed(1),
                  ttft: firstTokenTime ? ((firstTokenTime - fetchStartTime) / 1000).toFixed(2) : '0',
                  elapsed: (generationMs / 1000).toFixed(1)
                }
              })
            }
          } catch (_) { }
          await executeToolCalls()
          receivedToolCalls = []
          fullContent = ''
          rawAccumulated = ''
          clientToolCallBuffering = false
          iterationTokenBase = tokenCount // Save cumulative base for server-usage delta
          iterationTokenCount = 0
          tpsSnapshots.length = 0; liveTps = 0 // Reset rolling TPS for fresh generation phase
          emitToolStatus('processing', '', undefined, toolIteration)
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
            requestMessages.push({ role: 'assistant', content: fullContent })
          }
          requestMessages.push({
            role: 'user',
            content: 'Based on the tool results above, provide your complete response. Summarize what you found, explain the results, and address my original request.'
          })
          fullContent = ''
          rawAccumulated = ''
          clientToolCallBuffering = false
          receivedToolCalls = []
          iterationTokenBase = tokenCount // Save cumulative base for server-usage delta
          iterationTokenCount = 0
          tpsSnapshots.length = 0; liveTps = 0 // Reset rolling TPS for fresh generation phase
          emitToolStatus('processing', '', 'Generating response...', toolIteration)
          if (!await sendFollowUp()) break

        } else {
          break
        }
      }

      if (toolIteration > 0) {
        console.log(`[CHAT] Tool loop completed after ${toolIteration} iteration(s)`)
        emitToolStatus('done', '', undefined, toolIteration)
      }

      // Calculate final metrics — use generation-only time for t/s, fallback to wall clock
      const totalTime = (Date.now() - startTime) / 1000
      const genTimeSec = generationMs > 0 ? generationMs / 1000 : 0
      const wallTimeSec = firstTokenTime && lastTokenTime && lastTokenTime > firstTokenTime
        ? (lastTokenTime - firstTokenTime) / 1000
        : (firstTokenTime ? (Date.now() - firstTokenTime) / 1000 : totalTime)
      const finalGenSec = genTimeSec > 0.05 ? genTimeSec : wallTimeSec
      const finalTps = finalGenSec > 0 ? tokenCount / finalGenSec : 0
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
      fullContent = fullContent.trim()
      // If no main content but reasoning was produced, use reasoning as content
      // (e.g., model only produced <think>...</think> without content after)
      if (!fullContent && reasoningContent) {
        fullContent = reasoningContent
        console.log(`[CHAT] No main content produced — using reasoning content as fallback (${reasoningContent.length} chars)`)
      }
      assistantMessage.content = fullContent
      assistantMessage.tokens = tokenCount
      assistantMessage.metricsJson = JSON.stringify({
        tokenCount,
        promptTokens: promptTokens || undefined,
        cachedTokens: cachedTokens || undefined,
        tokensPerSecond: finalTps.toFixed(1),
        ppSpeed: finalPpSpeed,
        ttft: ttft.toFixed(2),
        totalTime: totalTime.toFixed(1)
      })
      db.addMessage(assistantMessage)

      // Send final metrics
      try {
        const win = getWindow()
        if (win && !win.isDestroyed()) {
          win.webContents.send('chat:complete', {
            chatId,
            messageId: assistantMessage.id,
            content: fullContent,
            metrics: {
              tokenCount,
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

      console.log(`[CHAT] Response complete: ${tokenCount} tokens in ${totalTime.toFixed(1)}s (${finalTps.toFixed(1)} t/s, live=${liveTps.toFixed(1)} t/s, TTFT: ${ttft.toFixed(2)}s${promptTokens ? `, pp: ${promptTokens} tokens${cachedTokens ? ` (${cachedTokens} cached)` : ''}, ${finalPpSpeed} pp/s` : ''}, usage=${serverSendsUsage ? 'server' : 'client'})`)

      return assistantMessage
    } catch (error) {
      // Release the SSE reader if it was acquired
      try { reader?.cancel() } catch (_) { }

      console.error('[CHAT] Error:', error)

      // Save partial response if any content or reasoning was generated before the error.
      // Use allGeneratedContent (content from before tool iterations) as fallback when
      // fullContent is empty (e.g., abort during tool execution resets fullContent to '').
      let partialContent = fullContent.trim()
      if (!partialContent && allGeneratedContent.trim()) {
        partialContent = allGeneratedContent.trim()
        console.log(`[CHAT] Using pre-tool-iteration content as fallback (${partialContent.length} chars)`)
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
      if (!partialContent && reasoningContent.trim()) {
        partialContent = reasoningContent.trim()
        console.log(`[CHAT] No content on abort — using reasoning as fallback (${partialContent.length} chars)`)
      }

      if (partialContent) {
        assistantMessage.content = partialContent + '\n\n[Generation interrupted]'
        assistantMessage.tokens = tokenCount

        // Calculate real metrics for the partial generation (not hardcoded zeros)
        const abortTotalTime = (Date.now() - startTime) / 1000
        const abortGenSec = generationMs > 50 ? generationMs / 1000
          : (firstTokenTime ? (Date.now() - firstTokenTime) / 1000 : abortTotalTime)
        const abortTps = (abortGenSec > 0 && tokenCount > 0) ? tokenCount / abortGenSec : 0
        // Use fetchStartTime for TTFT (consistent with non-abort path)
        const abortTtft = firstTokenTime ? (firstTokenTime - fetchStartTime) / 1000 : 0
        const abortPpSpeed = (promptTokens > 0 && abortTtft > 0.001)
          ? (promptTokens / abortTtft).toFixed(1)
          : undefined

        const abortMetrics = {
          tokenCount,
          promptTokens: promptTokens || undefined,
          cachedTokens: cachedTokens || undefined,
          tokensPerSecond: abortTps.toFixed(1),
          ppSpeed: abortPpSpeed,
          ttft: abortTtft.toFixed(2),
          totalTime: abortTotalTime.toFixed(1)
        }

        // Persist metricsJson to DB so reloading the chat shows real stats
        assistantMessage.metricsJson = JSON.stringify(abortMetrics)
        db.addMessage(assistantMessage)

        try {
          const win = getWindow()
          if (win && !win.isDestroyed()) {
            win.webContents.send('chat:complete', {
              chatId,
              messageId: assistantMessage.id,
              content: assistantMessage.content,
              metrics: abortMetrics
            })
          }
        } catch (_) { }
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
        return partialContent ? assistantMessage : null
      }
      if (errMsg === 'terminated' || errMsg.includes('ECONNREFUSED') || errMsg.includes('ECONNRESET')) {
        throw new Error(`Server connection lost. The model server may have crashed or stopped. Try restarting the session.`)
      }
      throw new Error(`Failed to send message: ${errMsg}`)
    } finally {
      // Always clean up the active request tracker
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
          const cancelBase = entry.baseUrl || `http://${entry.endpoint!.host}:${entry.endpoint!.port}`
          await fetch(
            `${cancelBase}${cancelPath}`,
            { method: 'POST', headers: entry.authHeaders || {}, signal: AbortSignal.timeout(2000) }
          )
          console.log(`[CHAT] Server cancel sent for ${entry.responseId}`)
        } catch (_) { /* server may already be done */ }
      }

      activeRequests.delete(chatId)
      return { success: true }
    }
    return { success: false, error: 'No active request for this chat' }
  })

  // Clear all active locks (called on window reload/close)
  ipcMain.handle('chat:clearAllLocks', async () => {
    const count = activeRequests.size
    for (const [, entry] of activeRequests) {
      try { entry.controller.abort() } catch (_) { }
    }
    activeRequests.clear()
    return { cleared: count }
  })

  // Overrides
  ipcMain.handle('chat:setOverrides', async (_, chatId: string, overrides: any) => {
    db.setChatOverrides({ chatId, ...overrides })
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
