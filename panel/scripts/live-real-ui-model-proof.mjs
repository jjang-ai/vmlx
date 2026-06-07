#!/usr/bin/env node
import { execFile, spawn } from 'node:child_process'
import crypto from 'node:crypto'
import { createServer } from 'node:http'
import net from 'node:net'
import { existsSync, mkdirSync, mkdtempSync, readFileSync, rmSync, writeFileSync } from 'node:fs'
import { tmpdir } from 'node:os'
import path from 'node:path'
import { promisify } from 'node:util'

const panelDir = path.resolve(new URL('..', import.meta.url).pathname)
const repoDir = path.resolve(panelDir, '..')
const modelPath = process.env.VMLINUX_REAL_UI_MODEL_PATH || process.env.VMLX_REAL_UI_MODEL_PATH
const proofBasename = process.env.VMLINUX_REAL_UI_PROOF_BASENAME
  || process.env.VMLX_REAL_UI_PROOF_BASENAME
  || 'current-real-ui-live-model-zaya-text-20260526'
const python = process.env.VMLINUX_REAL_UI_PYTHON
  || process.env.VMLX_REAL_UI_PYTHON
  || path.join(repoDir, '.venv', 'bin', 'python')
const installedAppPath = process.env.VMLINUX_REAL_UI_APP_PATH
  || process.env.VMLX_REAL_UI_APP_PATH
  || ''
const servedModel = process.env.VMLINUX_REAL_UI_SERVED_MODEL
  || process.env.VMLX_REAL_UI_SERVED_MODEL
  || path.basename(modelPath || 'real-ui-model').replace(/[^A-Za-z0-9_.-]+/g, '-')
const modelName = path.basename(modelPath || servedModel)
const wireApi = process.env.VMLINUX_REAL_UI_WIRE_API
  || process.env.VMLX_REAL_UI_WIRE_API
  || 'chat'
const promptOneOverride = process.env.VMLINUX_REAL_UI_PROMPT_1
  || process.env.VMLX_REAL_UI_PROMPT_1
const promptTwoOverride = process.env.VMLINUX_REAL_UI_PROMPT_2
  || process.env.VMLX_REAL_UI_PROMPT_2
const requestMaxTokens = Number(process.env.VMLINUX_REAL_UI_MAX_TOKENS || process.env.VMLX_REAL_UI_MAX_TOKENS || '96')
const requestMaxPromptTokensRaw = process.env.VMLINUX_REAL_UI_MAX_PROMPT_TOKENS
  || process.env.VMLX_REAL_UI_MAX_PROMPT_TOKENS
  || process.env.VMLINUX_REAL_UI_MAX_CONTEXT_TOKENS
  || process.env.VMLX_REAL_UI_MAX_CONTEXT_TOKENS
  || ''
const requestMaxPromptTokens = requestMaxPromptTokensRaw
  ? Number(requestMaxPromptTokensRaw)
  : null

function envBool(name, fallback = false) {
  const value = process.env[name] ?? process.env[name.replace('VMLINUX_', 'VMLX_')]
  if (value == null || value === '') return fallback
  return /^(1|true|yes|on)$/i.test(value)
}

const builtinToolsEnabled = envBool('VMLINUX_REAL_UI_BUILTIN_TOOLS', false)
const defaultPromptOne = builtinToolsEnabled
  ? [
      'Use the run_command tool exactly once to create a file named real_ui_tool_probe_1.txt in the configured working directory.',
      'Write the text REAL_UI_LIVE_TOOL_ONE into that file.',
      'After the tool result is returned, reply briefly in English and include REAL_UI_LIVE_TOOL_ONE once.',
    ].join(' ')
  : 'Reply briefly in English. Include the phrase REAL_UI_LIVE once.'
const defaultPromptTwo = builtinToolsEnabled
  ? [
      'Use the run_command tool exactly once to read real_ui_tool_probe_1.txt and create real_ui_tool_probe_2.txt in the same working directory.',
      'Write REAL_UI_LIVE_TOOL_TWO into the second file.',
      'Do not copy the first file into the second file; the second file must contain REAL_UI_LIVE_TOOL_TWO.',
      'After the tool result is returned, reply briefly in English with REAL_UI_LIVE_TOOL_TWO once and mention this is the second UI turn.',
    ].join(' ')
  : 'Repeat the phrase REAL_UI_LIVE once and mention that this is the second UI turn.'
const promptOne = promptOneOverride || defaultPromptOne
const promptTwo = promptTwoOverride || defaultPromptTwo
const checkServerCacheControls = envBool('VMLINUX_REAL_UI_CHECK_SERVER_CACHE_CONTROLS', false)
const checkMedia = envBool('VMLINUX_REAL_UI_CHECK_MEDIA', false)
const checkVideo = envBool('VMLINUX_REAL_UI_CHECK_VIDEO', false)
const expectPagedCacheLocked = envBool('VMLINUX_REAL_UI_EXPECT_PAGED_CACHE_LOCKED', false)
const enableThinkingOverride = (
  process.env.VMLINUX_REAL_UI_ENABLE_THINKING != null
  || process.env.VMLX_REAL_UI_ENABLE_THINKING != null
)
  ? envBool('VMLINUX_REAL_UI_ENABLE_THINKING', false)
  : undefined
const maxToolIterations = Number(process.env.VMLINUX_REAL_UI_MAX_TOOL_ITERATIONS || process.env.VMLX_REAL_UI_MAX_TOOL_ITERATIONS || '4')
const toolResultMaxChars = Number(process.env.VMLINUX_REAL_UI_TOOL_RESULT_MAX_CHARS || process.env.VMLX_REAL_UI_TOOL_RESULT_MAX_CHARS || '12345')
const imageDataUrl = process.env.VMLINUX_REAL_UI_IMAGE_DATA_URL
  || process.env.VMLX_REAL_UI_IMAGE_DATA_URL
  || 'data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAIAAACQd1PeAAAADElEQVR4nGP4z8AAAAMBAQDJ/pLvAAAAAElFTkSuQmCC'
const imageExpectRegex = process.env.VMLINUX_REAL_UI_IMAGE_EXPECT_REGEX
  || process.env.VMLX_REAL_UI_IMAGE_EXPECT_REGEX
  || '\\bred\\b'
const videoDataUrl = process.env.VMLINUX_REAL_UI_VIDEO_DATA_URL
  || process.env.VMLX_REAL_UI_VIDEO_DATA_URL
  || ''
const videoExpectRegex = process.env.VMLINUX_REAL_UI_VIDEO_EXPECT_REGEX
  || process.env.VMLX_REAL_UI_VIDEO_EXPECT_REGEX
  || ''
const cacheExpectRegex = process.env.VMLINUX_REAL_UI_CACHE_EXPECT_REGEX
  || process.env.VMLX_REAL_UI_CACHE_EXPECT_REGEX
  || ''

if (!modelPath) {
  console.error('Set VMLINUX_REAL_UI_MODEL_PATH or VMLX_REAL_UI_MODEL_PATH')
  process.exit(2)
}

const sleep = (ms) => new Promise((resolve) => setTimeout(resolve, ms))
const execFileAsync = promisify(execFile)

async function removeTemporaryTree(target, { maxRetries = 8 } = {}) {
  for (let attempt = 0; attempt <= maxRetries; attempt += 1) {
    try {
      rmSync(target, { recursive: true, force: true })
      return
    } catch (error) {
      if (!['ENOTEMPTY', 'EBUSY', 'EPERM'].includes(error?.code) || attempt === maxRetries) {
        throw error
      }
      await sleep(50 * (attempt + 1))
    }
  }
}

async function freePort() {
  return await new Promise((resolve, reject) => {
    const server = createServer()
    server.listen(0, '127.0.0.1', () => {
      const port = server.address().port
      server.close(() => resolve(port))
    })
    server.on('error', reject)
  })
}

async function requestJson(url, timeoutMs = 1000) {
  const controller = new AbortController()
  const timer = setTimeout(() => controller.abort(), timeoutMs)
  try {
    const res = await fetch(url, { signal: controller.signal })
    if (!res.ok) throw new Error(`${res.status} ${res.statusText}`)
    return await res.json()
  } finally {
    clearTimeout(timer)
  }
}

function isSocketDisconnectError(error) {
  const code = String(error?.code || '')
  const message = String(error?.message || error || '')
  const cause = error?.cause
  const nestedErrors = Array.isArray(error?.errors) ? error.errors : []
  return (
    code === 'EPIPE'
    || code === 'ECONNRESET'
    || code === 'ERR_STREAM_DESTROYED'
    || code === 'ERR_STREAM_WRITE_AFTER_END'
    || /EPIPE|write EPIPE|broken pipe|socket hang up|connection reset|premature close|stream.*destroyed|write after end/i.test(message)
    || (cause ? isSocketDisconnectError(cause) : false)
    || nestedErrors.some((nested) => isSocketDisconnectError(nested))
  )
}

function attachChildProcessStreamErrorGuard(stream, logs) {
  stream?.on('error', (error) => {
    if (isSocketDisconnectError(error)) return
    logs.push(`child stdio stream error: ${error?.message || String(error)}`)
  })
}

class CdpSocket {
  constructor(socket) {
    this.socket = socket
    this.buffer = Buffer.alloc(0)
    this.nextId = 1
    this.pending = new Map()
    this.closed = false
    socket.on('data', (chunk) => this.onData(chunk))
    socket.on('error', (error) => {
      if (isSocketDisconnectError(error)) this.closed = true
      this.rejectPending(error)
    })
    socket.on('close', () => {
      this.closed = true
      this.rejectPending(new Error('CDP socket closed before response'))
    })
    socket.on('end', () => {
      this.closed = true
      this.rejectPending(new Error('CDP socket ended before response'))
    })
  }

  static async connect(wsUrl) {
    const url = new URL(wsUrl)
    const key = crypto.randomBytes(16).toString('base64')
    const socket = net.connect(Number(url.port || 80), url.hostname)
    await new Promise((resolve, reject) => {
      socket.once('connect', resolve)
      socket.once('error', reject)
    })
    socket.write([
      `GET ${url.pathname}${url.search} HTTP/1.1`,
      `Host: ${url.host}`,
      'Upgrade: websocket',
      'Connection: Upgrade',
      `Sec-WebSocket-Key: ${key}`,
      'Sec-WebSocket-Version: 13',
      '\r\n',
    ].join('\r\n'))
    let handshake = Buffer.alloc(0)
    return await new Promise((resolve, reject) => {
      const onData = (chunk) => {
        handshake = Buffer.concat([handshake, chunk])
        const idx = handshake.indexOf('\r\n\r\n')
        if (idx < 0) return
        socket.off('data', onData)
        const header = handshake.slice(0, idx).toString('utf8')
        if (!header.includes(' 101 ')) {
          reject(new Error(`WebSocket upgrade failed: ${header.split('\r\n')[0]}`))
          return
        }
        const rest = handshake.slice(idx + 4)
        const cdp = new CdpSocket(socket)
        if (rest.length) cdp.onData(rest)
        resolve(cdp)
      }
      socket.on('data', onData)
      socket.once('error', reject)
    })
  }

  send(method, params = {}, timeoutMs = 60_000) {
    const id = this.nextId++
    const payload = JSON.stringify({ id, method, params })
    return new Promise((resolve, reject) => {
      const timer = setTimeout(() => {
        if (this.pending.has(id)) {
          this.pending.delete(id)
          reject(new Error(`CDP timeout: ${method}`))
        }
      }, timeoutMs)
      this.pending.set(id, { resolve, reject, timer })
      try {
        this.writeClientFrame(payload)
      } catch (error) {
        clearTimeout(timer)
        this.pending.delete(id)
        reject(error)
      }
    })
  }

  close() {
    this.closed = true
    try { this.socket.end() } catch {}
    try { this.socket.destroy() } catch {}
  }

  rejectPending(error) {
    for (const { reject, timer } of this.pending.values()) {
      clearTimeout(timer)
      reject(error)
    }
    this.pending.clear()
  }

  writeClientFrame(payload) {
    if (this.socket.destroyed || this.closed) {
      const error = new Error('CDP socket closed before write')
      error.code = 'ERR_STREAM_DESTROYED'
      this.rejectPending(error)
      return false
    }
    try {
      this.socket.write(encodeClientFrame(payload))
      return true
    } catch (error) {
      if (isSocketDisconnectError(error)) {
        this.closed = true
        this.rejectPending(error)
        return false
      }
      throw error
    }
  }

  onData(chunk) {
    this.buffer = Buffer.concat([this.buffer, chunk])
    while (this.buffer.length >= 2) {
      const opcode = this.buffer[0] & 0x0f
      let len = this.buffer[1] & 0x7f
      let offset = 2
      if (len === 126) {
        if (this.buffer.length < 4) return
        len = this.buffer.readUInt16BE(2)
        offset = 4
      } else if (len === 127) {
        if (this.buffer.length < 10) return
        const high = this.buffer.readUInt32BE(2)
        const low = this.buffer.readUInt32BE(6)
        if (high !== 0) throw new Error('CDP frame too large')
        len = low
        offset = 10
      }
      if (this.buffer.length < offset + len) return
      const payload = this.buffer.slice(offset, offset + len)
      this.buffer = this.buffer.slice(offset + len)
      if (opcode === 1) {
        const msg = JSON.parse(payload.toString('utf8'))
        if (msg.id && this.pending.has(msg.id)) {
          const { resolve, reject, timer } = this.pending.get(msg.id)
          this.pending.delete(msg.id)
          clearTimeout(timer)
          if (msg.error) reject(new Error(JSON.stringify(msg.error)))
          else resolve(msg.result)
        }
      } else if (opcode === 8) {
        this.close()
      }
    }
  }
}

function encodeClientFrame(text) {
  const payload = Buffer.from(text, 'utf8')
  const len = payload.length
  const headerLen = len < 126 ? 2 : len < 65536 ? 4 : 10
  const header = Buffer.alloc(headerLen + 4)
  header[0] = 0x81
  if (len < 126) {
    header[1] = 0x80 | len
  } else if (len < 65536) {
    header[1] = 0x80 | 126
    header.writeUInt16BE(len, 2)
  } else {
    header[1] = 0x80 | 127
    header.writeUInt32BE(0, 2)
    header.writeUInt32BE(len, 6)
  }
  const mask = crypto.randomBytes(4)
  mask.copy(header, headerLen)
  const out = Buffer.alloc(header.length + payload.length)
  header.copy(out, 0)
  for (let i = 0; i < payload.length; i++) {
    out[header.length + i] = payload[i] ^ mask[i % 4]
  }
  return out
}

async function waitForTarget(debugPort, appLogs) {
  const started = Date.now()
  while (Date.now() - started < 60_000) {
    if (appLogs.some((line) => line.includes('Failed to get lock') || line.includes('second-instance'))) {
      throw new Error('Electron app did not acquire single-instance lock')
    }
    try {
      const targets = await requestJson(`http://127.0.0.1:${debugPort}/json/list`, 1000)
      const page = targets.find((t) => t.type === 'page' && t.webSocketDebuggerUrl)
      if (page) return page
    } catch {}
    await sleep(250)
  }
  throw new Error(`Timed out waiting for DevTools target on ${debugPort}`)
}

async function waitForServer(baseUrl, server) {
  const serverLogs = server.logs
  const started = Date.now()
  while (Date.now() - started < 900_000) {
    try {
      const models = await requestJson(`${baseUrl}/v1/models`, 2000)
      return models
    } catch {}
    if (server.proc.exitCode != null || server.proc.signalCode != null) {
      const tail = server.logs.slice(-80).join('\n')
      throw new Error(
        `server process exited before health: code=${server.proc.exitCode} signal=${server.proc.signalCode}\n${tail}`,
      )
    }
    if (serverLogs.some((line) => /Traceback|ERROR|Exception/.test(line))) {
      const tail = serverLogs.slice(-80).join('\n')
      if (/Address already in use/.test(tail)) throw new Error(tail)
    }
    await sleep(1000)
  }
  throw new Error(`Timed out waiting for real model server at ${baseUrl}`)
}

async function evaluate(cdp, expression, timeoutMs = 120_000) {
  const result = await cdp.send('Runtime.evaluate', {
    expression,
    awaitPromise: true,
    returnByValue: true,
    timeout: timeoutMs,
  }, timeoutMs + 5_000)
  if (result.exceptionDetails) {
    throw new Error(JSON.stringify(result.exceptionDetails, null, 2))
  }
  return result.result?.value
}

async function capturePng(cdp, filePath) {
  const shot = await cdp.send('Page.captureScreenshot', {
    format: 'png',
    captureBeyondViewport: true,
  })
  writeFileSync(filePath, Buffer.from(shot.data, 'base64'))
  return filePath
}

function startRealServer(port, outDir) {
  const conservativeRuntime = envBool('VMLINUX_REAL_UI_CONSERVATIVE_RUNTIME', false)
  const runtimeArgs = conservativeRuntime
    ? [
        '--no-continuous-batching',
        '--disable-prefix-cache',
        '--kv-cache-quantization',
        'none',
        '--disable-native-mtp',
      ]
    : [
        '--continuous-batching',
        '--use-paged-cache',
        '--paged-cache-block-size',
        '64',
        '--max-cache-blocks',
        '1000',
        '--enable-block-disk-cache',
        '--block-disk-cache-dir',
        path.join(outDir, 'block-cache'),
        '--block-disk-cache-max-gb',
        '2',
      ]
  const args = [
    '-B',
    '-s',
    '-m',
    'vmlx_engine.cli',
    'serve',
    modelPath,
    '--host',
    '127.0.0.1',
    '--port',
    String(port),
    '--served-model-name',
    servedModel,
    '--timeout',
    '240',
    '--max-num-seqs',
    '1',
    '--prefill-batch-size',
    '512',
    '--prefill-step-size',
    '1024',
    '--completion-batch-size',
    '128',
    ...runtimeArgs,
    '--ssm-state-cache-mb',
    '512',
    '--max-tokens',
    process.env.VMLINUX_REAL_UI_MAX_TOKENS || '96',
    '--log-level',
    'INFO',
    '--default-enable-thinking',
    'false',
  ]
  if (Number.isFinite(requestMaxPromptTokens) && requestMaxPromptTokens > 0) {
    args.push('--max-prompt-tokens', String(Math.floor(requestMaxPromptTokens)))
  }
  if (builtinToolsEnabled) {
    args.push('--enable-auto-tool-choice', '--tool-call-parser', 'auto')
  }
  if (process.env.VMLINUX_REAL_UI_IS_MLLM === '1' || process.env.VMLX_REAL_UI_IS_MLLM === '1') {
    args.push('--is-mllm')
  }
  const logs = []
  const proc = spawn(python, args, {
    cwd: repoDir,
    env: {
      ...process.env,
      PYTHONUNBUFFERED: '1',
      PYTHONPATH: repoDir,
    },
    detached: true,
    stdio: ['ignore', 'pipe', 'pipe'],
  })
  proc.stdout.on('data', (d) => logs.push(...d.toString().split(/\r?\n/).filter(Boolean)))
  proc.stderr.on('data', (d) => logs.push(...d.toString().split(/\r?\n/).filter(Boolean)))
  attachChildProcessStreamErrorGuard(proc.stdout, logs)
  attachChildProcessStreamErrorGuard(proc.stderr, logs)
  return { proc, logs, command: [python, ...args] }
}

function startUiApp(userDataDir, debugPort) {
  if (installedAppPath) {
    const exe = path.join(installedAppPath, 'Contents', 'MacOS', 'vMLX')
    if (!existsSync(exe)) {
      throw new Error(`Installed vMLX executable not found: ${exe}`)
    }
    const args = [
      `--user-data-dir=${userDataDir}`,
      `--remote-debugging-port=${debugPort}`,
    ]
    const logs = []
    const proc = spawn(exe, args, {
      cwd: tmpdir(),
      env: { ...process.env, VMLX_SKIP_UPDATE_CHECK: '1' },
      detached: true,
      stdio: ['ignore', 'pipe', 'pipe'],
    })
    proc.stdout.on('data', (d) => logs.push(...d.toString().split(/\r?\n/).filter(Boolean)))
    proc.stderr.on('data', (d) => logs.push(...d.toString().split(/\r?\n/).filter(Boolean)))
    attachChildProcessStreamErrorGuard(proc.stdout, logs)
    attachChildProcessStreamErrorGuard(proc.stderr, logs)
    return {
      proc,
      logs,
      uiLaunchMode: 'installed-app',
      command: [exe, ...args],
      appPath: installedAppPath,
    }
  }

  const args = [
    'run',
    'dev',
    '--',
    '--',
    `--user-data-dir=${userDataDir}`,
    `--remote-debugging-port=${debugPort}`,
  ]
  const logs = []
  const proc = spawn('npm', args, {
    cwd: panelDir,
    env: { ...process.env, VMLX_SKIP_UPDATE_CHECK: '1' },
    detached: true,
    stdio: ['ignore', 'pipe', 'pipe'],
  })
  proc.stdout.on('data', (d) => logs.push(...d.toString().split(/\r?\n/).filter(Boolean)))
  proc.stderr.on('data', (d) => logs.push(...d.toString().split(/\r?\n/).filter(Boolean)))
  attachChildProcessStreamErrorGuard(proc.stdout, logs)
  attachChildProcessStreamErrorGuard(proc.stderr, logs)
  return {
    proc,
    logs,
    uiLaunchMode: 'electron-dev',
    command: ['npm', ...args],
    appPath: '',
  }
}

async function childProcessTree(rootPid) {
  if (!rootPid) return []
  let stdout = ''
  try {
    const result = await execFileAsync('ps', ['-axo', 'pid=,ppid='])
    stdout = result.stdout || ''
  } catch (_) {
    return []
  }
  const childrenByParent = new Map()
  for (const line of stdout.split(/\r?\n/)) {
    const trimmed = line.trim()
    if (!trimmed) continue
    const [pidText, ppidText] = trimmed.split(/\s+/, 2)
    const pid = Number(pidText)
    const ppid = Number(ppidText)
    if (!Number.isInteger(pid) || !Number.isInteger(ppid)) continue
    const children = childrenByParent.get(ppid) || []
    children.push(pid)
    childrenByParent.set(ppid, children)
  }
  const descendants = []
  const stack = [...(childrenByParent.get(rootPid) || [])]
  while (stack.length) {
    const pid = stack.pop()
    descendants.push(pid)
    stack.push(...(childrenByParent.get(pid) || []))
  }
  return descendants
}

async function terminateProcessTree(proc, signal) {
  if (!proc?.pid) return
  const descendants = await childProcessTree(proc.pid)
  for (const pid of descendants.reverse()) {
    try { process.kill(pid, signal) } catch {}
  }
  try { process.kill(-proc.pid, signal) } catch {}
  try { process.kill(proc.pid, signal) } catch {}
}

async function terminateProcess(proc) {
  if (!proc?.pid) return
  try { proc.stdout?.destroy() } catch {}
  try { proc.stderr?.destroy() } catch {}
  await terminateProcessTree(proc, 'SIGTERM')
  await sleep(1500)
  if (proc.exitCode == null && proc.signalCode == null) {
    await terminateProcessTree(proc, 'SIGKILL')
  }
}

function visibleAssistantAfterEachUser(turns) {
  if (!Array.isArray(turns)) return false
  let sawUser = false
  for (let i = 0; i < turns.length; i += 1) {
    const turn = turns[i]
    if (!turn || turn.role !== 'user') continue
    sawUser = true
    let nextUserIndex = turns.length
    for (let j = i + 1; j < turns.length; j += 1) {
      if (turns[j]?.role === 'user') {
        nextUserIndex = j
        break
      }
    }
    const hasVisibleAssistant = turns
      .slice(i + 1, nextUserIndex)
      .some((candidate) => candidate?.role === 'assistant' && String(candidate?.content || '').trim())
    if (!hasVisibleAssistant) return false
  }
  return sawUser
}

function assertResult(result) {
  const failures = []
  const chat = result.chat || {}
  const cacheTelemetryExpected = (
    Array.isArray(result.serverCommand)
    && !result.serverCommand.includes('--disable-prefix-cache')
    && !result.serverCommand.includes('--no-continuous-batching')
  )
  if (!result.server?.models?.data?.length) failures.push('real server /v1/models returned no models')
  if (result.remoteSessionStarted !== true) failures.push('remote session did not start through Electron UI API')
  if (!chat.turns?.some((m) => m.role === 'assistant' && m.content)) failures.push('assistant content is empty')
  if (!chat.finalVisibleText) failures.push('final visible assistant content is empty')
  const visibleAssistantTurnsComplete = visibleAssistantAfterEachUser(chat.turns)
  if (!visibleAssistantTurnsComplete) {
    failures.push('UI turn ended with empty visible assistant content')
  }
  if (chat.rawParserTagLeak) failures.push('raw parser/reasoning/tool markup leaked into UI content')
  if (chat.reasoningRawParserTagLeak) failures.push('raw parser/reasoning/tool markup leaked into reasoning segments')
  if ((chat.reasoningCjkLeakCount || 0) > 0 || (chat.reasoningKoreanLeakCount || 0) > 0) {
    failures.push('wrong-language text leaked into reasoning segments')
  }
  if ((chat.reasoningNumericRunCount || 0) > 0) {
    failures.push('numeric/list-like garbage leaked into reasoning segments')
  }
  if ((result.eventCounts?.complete || 0) < 2) failures.push('expected two completed UI chat turns')
  if ((result.eventCounts?.stream || 0) < 1) failures.push('expected streaming events from real model')
  if ((chat.turns?.length || 0) < 4) failures.push(`expected at least four persisted chat messages, got ${chat.turns?.length || 0}`)
  if (result.sendErrors?.length) failures.push(`renderer send errors: ${result.sendErrors.join('; ')}`)
  if (cacheTelemetryExpected && (result.cache?.cacheHitTokens || 0) <= 0) failures.push('expected real cache-hit token telemetry after repeated UI turns')
  if (cacheTelemetryExpected && !result.provenSurfaces?.includes('cache_hit_telemetry')) {
    failures.push('live proof did not record clean cache-hit telemetry')
  }
  if (
    result.requestedServerCacheControls === true
    && result.server?.health?.native_cache?.block_disk_l2 === true
    && !l2DiskStorageSeen(result.cache?.after)
  ) {
    failures.push('expected cache endpoint L2 disk storage telemetry for native block_disk_l2 cache')
  }
  if (result.requestedWireApi === 'responses' && !result.provenSurfaces?.includes('responses_api')) {
    failures.push('requested Responses API mode but proof did not record responses_api surface')
  }
  if (result.requestedWireApi === 'responses' && !result.provenSurfaces?.includes('responses_delta_streaming')) {
    failures.push('requested Responses API mode but proof did not record responses_delta_streaming surface')
  }
  if (
    result.requestedWireApi === 'responses'
    && result.requestedServerCacheControls === true
    && !result.provenSurfaces?.includes('responses_cache_detail_usage')
  ) {
    failures.push('requested Responses API cache controls but proof did not record responses_cache_detail_usage surface')
  }
  if (!result.provenSurfaces?.includes('generation_defaults_applied')) {
    failures.push('live proof did not record model-owned generation defaults / request max_tokens resolution')
  }
  if (!result.provenSurfaces?.includes('language_leak_check')) {
    failures.push('live proof did not record clean visible/reasoning language leak check')
  }
  if (result.requestedBuiltinTools === true && !result.provenSurfaces?.includes('long_tool_loop')) {
    failures.push('requested real built-in tools but proof did not record long_tool_loop surface')
  }
  if (result.requestedEnableThinking === true && !result.provenSurfaces?.includes('reasoning_display')) {
    failures.push('requested real reasoning but proof did not record reasoning_display surface')
  }
  if (result.requestedServerCacheControls === true && !result.provenSurfaces?.includes('server_cache_controls')) {
    failures.push('requested real server cache controls but proof did not record server_cache_controls surface')
  }
  if (result.requestedMedia === true && !result.provenSurfaces?.includes('vl_image')) {
    failures.push('requested real image media but proof did not record vl_image surface')
  }
  if (result.requestedVideo === true && !result.provenSurfaces?.includes('video_where_supported')) {
    failures.push('requested real video media but proof did not record video_where_supported surface')
  }
  if (failures.length) {
    const error = new Error(`Real UI live-model proof failed:\n- ${failures.join('\n- ')}`)
    error.failures = failures
    throw error
  }
}

function deriveProvenSurfaces(result) {
  const surfaces = new Set()
  const chat = result.chat || {}
  const health = result.server?.health || {}
  if (result.uiLaunchMode === 'installed-app') {
    surfaces.add('installed_app_ui')
  } else if (result.appLogTail?.length) {
    surfaces.add('current_electron_dev_build')
  }
  if (health.status === 'healthy' && health.model_loaded === true) surfaces.add('real_loaded_model')
  if (chat.turns?.length) surfaces.add('chat_completions')
  if (chat.rawParserTagLeak === false && chat.reasoningRawParserTagLeak === false) surfaces.add('parser_leak_check')
  if (
    chat.cjkLeakCount === 0
    && chat.koreanLeakCount === 0
    && chat.reasoningCjkLeakCount === 0
    && chat.reasoningKoreanLeakCount === 0
    && chat.reasoningNumericRunCount === 0
  ) surfaces.add('language_leak_check')
  if ((result.cache?.cacheHitTokens || 0) > 0 && cacheReconstructionClean(result)) {
    surfaces.add('cache_hit_telemetry')
  }
  if (hasNativeCacheStatus(health)) surfaces.add('native_cache_status')
  if (hasCacheEndpointStats(result.cache)) surfaces.add('cache_endpoint_stats')
  if (l2DiskStorageSeen(result.cache?.after)) surfaces.add('l2_disk_storage')
  if (result.rendererWireApi === 'responses' && (result.eventCounts?.complete || 0) > 0) {
    surfaces.add('responses_api')
  }
  if (responsesDeltaStreamingSeen(result)) {
    surfaces.add('responses_delta_streaming')
  }
  if (responsesCacheDetailUsageSeen(result)) {
    surfaces.add('responses_cache_detail_usage')
  }
  if (generationDefaultsAppliedSeen(result)) {
    surfaces.add('generation_defaults_applied')
  }
  if (liveSpeedFloorSeen(result)) {
    surfaces.add('live_speed_floor')
  }
  if (
    (result.eventCounts?.tool || 0) >= 3
    && namedToolResultCount(result) >= 2
    && namedToolErrorCount(result) === 0
    && namedToolProbeSemanticsOk(result)
  ) {
    surfaces.add('long_tool_loop')
  }
  if (
    ((result.eventCounts?.reasoningDone || 0) > 0 || (result.persistedReasoningCount || 0) > 0)
    && (result.requestedEnableThinking !== true || visibleAssistantAfterEachUser(chat.turns))
  ) {
    surfaces.add('reasoning_display')
  }
  if (result.chatOverrides?.builtinToolsEnabled === result.requestedBuiltinTools) {
    surfaces.add('settings_persistence')
  }
  if (result.serverCacheControls?.verified === true) {
    surfaces.add('server_cache_controls')
  }
  if (
    surfaces.has('cache_hit_telemetry')
    && surfaces.has('l2_disk_storage')
    && surfaces.has('long_tool_loop')
    && surfaces.has('server_cache_controls')
  ) {
    surfaces.add('tool_l2_cache_integrated')
  }
  if (result.media?.imageVerified === true) {
    surfaces.add('vl_image')
  }
  if (result.media?.videoVerified === true) {
    surfaces.add('video_where_supported')
  }
  return [...surfaces].sort()
}

function extractLiveSpeedSamples(result) {
  const samples = []
  const lines = []
  for (const key of ['appLogTail', 'serverLogTail']) {
    if (Array.isArray(result?.[key])) {
      lines.push(...result[key].map((line) => String(line)))
    }
  }
  const speedRe = /Response complete:\s+(\d+)\s+tokens.*?live=(\d+(?:\.\d+)?)\s+t\/s,\s+TTFT:\s+(\d+(?:\.\d+)?)s.*?usage=server/
  for (const line of lines) {
    const match = line.match(speedRe)
    if (!match) continue
    samples.push({
      tokens: Number(match[1]),
      liveTokensPerSecond: Number(match[2]),
      ttftSeconds: Number(match[3]),
      line,
    })
  }
  return samples
}

function liveSpeedFloorForResult(result) {
  const identity = `${result?.modelName || ''} ${result?.modelPath || ''}`.toLowerCase()
  if (identity.includes('lfm2.5') || identity.includes('lfm25')) return 100
  if (identity.includes('step-3.7') || identity.includes('step37')) return 45
  return null
}

function liveSpeedFloorSeen(result) {
  const floor = liveSpeedFloorForResult(result)
  if (!floor) return false
  const samples = Array.isArray(result?.liveSpeedSamples)
    ? result.liveSpeedSamples
    : extractLiveSpeedSamples(result)
  let passing = 0
  for (const sample of samples) {
    if (
      Number(sample?.tokens || 0) > 0
      && Number(sample?.liveTokensPerSecond || 0) >= floor
      && Number(sample?.ttftSeconds || 0) > 0
      && Number(sample?.ttftSeconds || 0) <= 5
    ) {
      passing += 1
    }
  }
  return passing >= 2
}

function hasNativeCacheStatus(health) {
  const nativeCache = health && health.native_cache
  return !!(
    nativeCache
    && typeof nativeCache === 'object'
    && nativeCache.family
    && nativeCache.schema
    && nativeCache.cache_type
    && Array.isArray(nativeCache.components)
    && nativeCache.prefix === true
    && nativeCache.paged === true
    && nativeCache.block_disk_l2 === true
  )
}

function hasCacheEndpointStats(cache) {
  const before = cache?.before
  const after = cache?.after
  return !!(
    before
    && after
    && typeof before.scheduler_cache === 'object'
    && typeof after.scheduler_cache === 'object'
    && typeof after.block_disk_cache === 'object'
    && typeof after.cache_totals === 'object'
  )
}

function l2DiskStorageSeen(cache) {
  const blockDisk = cache?.block_disk_cache || {}
  const totals = cache?.cache_totals || {}
  for (const value of [
    blockDisk.blocks_on_disk,
    blockDisk.total_tokens_on_disk,
    blockDisk.total_cached_tokens,
    blockDisk.disk_writes,
    totals.l2_tokens_on_disk,
    totals.l2_block_tokens_on_disk,
    totals.l2_ssm_tokens_on_disk,
    totals.l2_tokens_on_disk_store_sum,
  ]) {
    if (typeof value === 'number' && value > 0) return true
  }
  return false
}

function responsesDeltaStreamingSeen(result) {
  if (result?.rendererWireApi !== 'responses') return false
  if ((result.eventCounts?.stream || 0) < 2) return false
  const traces = Array.isArray(result.streamTrace)
    ? result.streamTrace
    : (Array.isArray(result.streamTraceByMessage) ? result.streamTraceByMessage : [])
  const qualifyingTraceIds = new Set()
  let qualifyingTraceCount = 0
  for (const trace of traces) {
    if (
      trace
      && typeof trace === 'object'
      && (trace.count || 0) >= 2
      && typeof trace.firstFullContent === 'string'
      && typeof trace.lastFullContent === 'string'
      && trace.firstFullContent.length > 0
      && trace.lastFullContent.length > 0
      && trace.firstFullContent !== trace.lastFullContent
    ) {
      if (typeof trace.messageId === 'string' && trace.messageId) {
        qualifyingTraceIds.add(trace.messageId)
      } else {
        qualifyingTraceCount += 1
      }
    }
  }
  return qualifyingTraceIds.size + qualifyingTraceCount >= 2
}

function responsesCacheDetailUsageSeen(result) {
  if (result?.rendererWireApi !== 'responses') return false
  const walk = (value) => {
    if (!value || typeof value !== 'object') return false
    if (Array.isArray(value)) return value.some((child) => walk(child))
    const cacheDetail = value.cache_detail ?? value.cacheDetail
    const cachedTokens = value.cached_tokens ?? value.cachedTokens
    if (
      typeof cacheDetail === 'string'
      && cacheDetail.trim()
      && typeof cachedTokens === 'number'
      && cachedTokens > 0
    ) {
      return true
    }
    return Object.values(value).some((child) => walk(child))
  }
  return walk(result)
}

function generationDefaultsAppliedSeen(result) {
  const chatOverrides = result?.chatOverrides && typeof result.chatOverrides === 'object'
    ? result.chatOverrides
    : {}
  for (const field of ['temperature', 'topP', 'topK', 'minP', 'repeatPenalty']) {
    if (chatOverrides[field] != null) return false
  }
  const requestMaxTokens = result?.requestContract?.requestMaxTokens
  const overrideMaxTokens = chatOverrides.maxTokens
  if (
    typeof requestMaxTokens === 'number'
    && typeof overrideMaxTokens === 'number'
    && requestMaxTokens !== overrideMaxTokens
  ) {
    return false
  }
  const logText = Array.isArray(result?.serverLogTail)
    ? result.serverLogTail.map((line) => String(line)).join('\n')
    : ''
  if (!logText.includes('Resolved sampling kwargs route=')) return false
  if (!logText.includes('kwargs=') || !logText.includes('max_tokens')) return false
  if (result?.rendererWireApi === 'responses') return logText.includes('/v1/responses')
  if (result?.rendererWireApi === 'chat') return logText.includes('/v1/chat/completions')
  return Boolean(result?.rendererWireApi)
}

function cacheReconstructionClean(result) {
  for (const key of ['serverLogTail', 'appLogTail']) {
    const lines = Array.isArray(result?.[key]) ? result[key] : []
    for (const line of lines) {
      const text = String(line)
      if (
        text.includes('worker-side paged cache reconstruction failed')
        || text.includes('reconstruction failed, treating as cache miss')
        || text.includes('hybrid paged MISS')
        || text.includes('no usable SSM companion')
      ) {
        return false
      }
    }
  }
  return true
}

function namedToolResultCount(result) {
  const groups = Array.isArray(result.persistedToolsByMessage)
    ? result.persistedToolsByMessage
    : []
  let count = 0
  for (const group of groups) {
    if (!Array.isArray(group)) continue
    for (const item of group) {
      if (!item || typeof item !== 'object') continue
      if (item.phase === 'result' && typeof item.toolName === 'string' && item.toolName.trim()) {
        count += 1
      }
    }
  }
  return count
}

function namedToolErrorCount(result) {
  const groups = Array.isArray(result.persistedToolsByMessage)
    ? result.persistedToolsByMessage
    : []
  let count = 0
  for (const group of groups) {
    if (!Array.isArray(group)) continue
    for (const item of group) {
      if (!item || typeof item !== 'object') continue
      if (item.phase === 'error' && typeof item.toolName === 'string' && item.toolName.trim()) {
        count += 1
      }
    }
  }
  return count
}

function namedToolProbeSemanticsOk(result) {
  const turnsText = Array.isArray(result.chat?.turns)
    ? result.chat.turns.map((turn) => String(turn?.content || '')).join('\n')
    : ''
  const probeRequested = (
    turnsText.includes('REAL_UI_LIVE_TOOL_ONE')
    || turnsText.includes('REAL_UI_LIVE_TOOL_TWO')
  )
  if (!probeRequested) return true

  const groups = Array.isArray(result.persistedToolsByMessage)
    ? result.persistedToolsByMessage
    : []
  const resultDetails = groups.map((group) => {
    if (!Array.isArray(group)) return ''
    return group
      .filter((item) =>
        item
        && typeof item === 'object'
        && item.phase === 'result'
        && typeof item.toolName === 'string'
        && item.toolName.trim()
      )
      .map((item) => String(item.detail || item.message || item.text || ''))
      .join('\n')
  })
  const files = result.toolProbeFiles || {}
  const fileSemanticsOk = (
    String(files['real_ui_tool_probe_1.txt'] || '').trimEnd() === 'REAL_UI_LIVE_TOOL_ONE'
    && String(files['real_ui_tool_probe_2.txt'] || '').trimEnd() === 'REAL_UI_LIVE_TOOL_TWO'
  )
  const commandSemanticsOk = (
    resultDetails.some((detail) =>
      detail.includes('real_ui_tool_probe_1.txt')
      || detail.includes('REAL_UI_LIVE_TOOL_ONE')
    )
    && resultDetails.some((detail) =>
      detail.includes('real_ui_tool_probe_2.txt')
      || detail.includes('REAL_UI_LIVE_TOOL_TWO')
    )
  )
  const visibleToolSemanticsOk = (() => {
    const turns = Array.isArray(result.chat?.turns) ? result.chat.turns : []
    for (const turn of turns) {
      if (!turn || turn.role !== 'assistant') continue
      const content = String(turn.content || '')
      const lower = content.toLowerCase()
      const secondFile = lower.indexOf('real_ui_tool_probe_2.txt')
      const firstTokenAfterSecondFile = secondFile >= 0
        && /real[\s_\\-]*ui[\s_\\-]*live[\s_\\-]*tool[\s_\\-]*one/i.test(
          content.slice(secondFile, secondFile + 160),
        )
      const malformedSecondToken = (
        content.includes('RE:AL_UI_LIVE_TOOL_TWO')
        || /\bre\s*:\s*al[\s_\\-]*ui[\s_\\-]*live[\s_\\-]*tool[\s_\\-]*two\b/i.test(content)
      )
      if (firstTokenAfterSecondFile || malformedSecondToken) return false
    }
    return true
  })()
  const strictExactReplyOk = (() => {
    const turns = Array.isArray(result.chat?.turns) ? result.chat.turns : []
    const exactReplyRe = /reply exactly:\s*["'“”`]?([A-Za-z0-9_=-]+)["'“”`]?/i
    for (let i = 0; i < turns.length; i += 1) {
      const turn = turns[i]
      if (!turn || turn.role !== 'user') continue
      const match = String(turn.content || '').match(exactReplyRe)
      if (!match) continue
      const expected = match[1]
      const nextAssistant = turns.slice(i + 1).find((candidate) => candidate?.role === 'assistant')
      const actual = String(nextAssistant?.content || '').trim()
      if (actual !== expected) return false
    }
    return true
  })()
  return (
    commandSemanticsOk
    && fileSemanticsOk
    && visibleToolSemanticsOk
    && strictExactReplyOk
  )
}

function countMatches(text, regex) {
  return (text.match(regex) || []).length
}

function maxRecursiveNumber(value, keyPattern) {
  let best = 0
  const visit = (node, key = '') => {
    if (typeof node === 'number' && keyPattern.test(key)) {
      best = Math.max(best, node)
    } else if (Array.isArray(node)) {
      node.forEach((item) => visit(item, key))
    } else if (node && typeof node === 'object') {
      for (const [childKey, childValue] of Object.entries(node)) {
        visit(childValue, childKey)
      }
    }
  }
  visit(value)
  return best
}

async function main() {
  const runDir = mkdtempSync(path.join(tmpdir(), 'vmlx-real-ui-live-'))
  const userDataDir = mkdtempSync(path.join(tmpdir(), 'vmlx-real-ui-userdata-'))
  const workingDirectory = process.env.VMLINUX_REAL_UI_WORKING_DIRECTORY
    || process.env.VMLX_REAL_UI_WORKING_DIRECTORY
    || mkdtempSync(path.join(tmpdir(), 'vmlx-real-ui-tools-'))
  mkdirSync(workingDirectory, { recursive: true })
  const proofDir = path.join(repoDir, 'docs', 'internal', 'agent-notes')
  mkdirSync(proofDir, { recursive: true })

  const serverPort = Number(process.env.VMLINUX_REAL_UI_SERVER_PORT || await freePort())
  const debugPort = await freePort()
  const baseUrl = `http://127.0.0.1:${serverPort}`
  const server = startRealServer(serverPort, runDir)
  let app
  let cdp
  let appLogs = []
  let serverModels = {}
  let healthBefore = {}
  try {
    serverModels = await waitForServer(baseUrl, server)
    app = startUiApp(userDataDir, debugPort)
    appLogs = app.logs

    const target = await waitForTarget(debugPort, appLogs)
    cdp = await CdpSocket.connect(target.webSocketDebuggerUrl)
    await cdp.send('Runtime.enable')
    await cdp.send('Page.enable')
    await cdp.send('Emulation.setDeviceMetricsOverride', {
      width: 1440,
      height: 1000,
      deviceScaleFactor: 1,
      mobile: false,
    })
    await evaluate(cdp, `
      new Promise((resolve, reject) => {
        const started = Date.now();
        const check = () => {
          if (window.api?.chat && window.api?.sessions) resolve(true);
          else if (Date.now() - started > 30000) reject(new Error('window.api not ready'));
          else setTimeout(check, 100);
        };
        check();
      })
    `)

    healthBefore = await requestJson(`${baseUrl}/health`, 5000).catch((error) => ({ error: error.message }))
    let rendererResult
    try {
      rendererResult = await evaluate(cdp, `
      (async () => {
        const baseUrl = ${JSON.stringify(baseUrl)};
        const servedModel = ${JSON.stringify(servedModel)};
        const wireApi = ${JSON.stringify(wireApi)};
        const builtinToolsEnabled = ${JSON.stringify(builtinToolsEnabled)};
        const enableThinking = ${enableThinkingOverride === undefined ? 'undefined' : JSON.stringify(enableThinkingOverride)};
        const checkMedia = ${JSON.stringify(checkMedia)};
        const checkVideo = ${JSON.stringify(checkVideo)};
        const imageDataUrl = ${JSON.stringify(imageDataUrl)};
        const imageExpectRegex = ${JSON.stringify(imageExpectRegex)};
        const videoDataUrl = ${JSON.stringify(videoDataUrl)};
        const videoExpectRegex = ${JSON.stringify(videoExpectRegex)};
        const workingDirectory = ${JSON.stringify(workingDirectory)};
        const endpoint = { host: '127.0.0.1', port: ${JSON.stringify(serverPort)} };
        const l2DiskStorageSeen = ${l2DiskStorageSeen.toString()};
        const waitForCacheEndpointStorage = async (initial, sessionId) => {
          if (!${JSON.stringify(checkServerCacheControls)}) return initial;
          let latest = initial;
          if (l2DiskStorageSeen(latest)) return latest;
          const started = Date.now();
          while (Date.now() - started < 15000) {
            await new Promise((resolve) => setTimeout(resolve, 250));
            latest = await window.api.cache.stats(endpoint, sessionId)
              .catch((error) => ({ error: String(error?.message || error) }));
            if (l2DiskStorageSeen(latest)) return latest;
          }
          return latest;
        };
        await new Promise((resolve, reject) => {
          const started = Date.now();
          const check = () => {
            if (document.getElementById('root')?.children.length) resolve(true);
            else if (Date.now() - started > 30000) reject(new Error('React root not mounted'));
            else setTimeout(check, 100);
          };
          check();
        });
        await window.api.engine.checkInstallation().catch(() => null);
        await window.api.chat.clearAllLocks().catch(() => null);
        const events = { stream: [], tool: [], reasoningDone: [], complete: [] };
        const cleanup = [
          window.api.chat.onStream((data) => events.stream.push({ t: performance.now(), ...data })),
          window.api.chat.onToolStatus((data) => events.tool.push({ t: performance.now(), ...data })),
          window.api.chat.onReasoningDone((data) => events.reasoningDone.push({ t: performance.now(), ...data })),
          window.api.chat.onComplete((data) => events.complete.push({ t: performance.now(), ...data })),
        ];
        try {
          const remote = await window.api.sessions.createRemote({
            remoteUrl: baseUrl,
            remoteModel: servedModel,
          });
          if (!remote.success) throw new Error(remote.error || 'remote session create failed');
          await window.api.sessions.start(remote.session.id);
          const preloadHealthBefore = await window.api.performance.health(endpoint)
            .catch((error) => ({ error: String(error?.message || error) }));
          const cacheBefore = await window.api.cache.stats(endpoint, remote.session.id)
            .catch((error) => ({ error: String(error?.message || error) }));
          const chat = await window.api.chat.create('Real UI live model proof', servedModel, undefined, remote.session.modelPath);
          const overrides = {
            chatId: chat.id,
            wireApi,
            builtinToolsEnabled,
            shellEnabled: builtinToolsEnabled,
            fileToolsEnabled: builtinToolsEnabled,
            searchToolsEnabled: false,
            gitEnabled: false,
            utilityToolsEnabled: builtinToolsEnabled,
            webSearchEnabled: false,
            braveSearchEnabled: false,
            fetchUrlEnabled: false,
            workingDirectory,
            maxToolIterations: ${JSON.stringify(maxToolIterations)},
            toolResultMaxChars: ${JSON.stringify(toolResultMaxChars)},
            maxTokens: ${JSON.stringify(requestMaxTokens)},
          };
          if (enableThinking !== undefined) overrides.enableThinking = enableThinking;
          await window.api.chat.setOverrides(chat.id, overrides);
          const chatOverrides = await window.api.chat.getOverrides(chat.id).catch((error) => ({ error: String(error?.message || error) }));
          const sendErrors = [];
          let rendererFailureStage = null;
          const sendMessageWithCapture = async (turn, stage, prompt, attachments) => {
            try {
              await window.api.chat.sendMessage(chat.id, prompt, undefined, attachments);
              return true;
            } catch (error) {
              rendererFailureStage = stage;
              sendErrors.push({ turn, stage, message: String(error?.message || error) });
              return false;
            }
          };
          const firstSent = await sendMessageWithCapture(1, 'first_send_message', ${JSON.stringify(promptOne)});
          if (firstSent) {
            await sendMessageWithCapture(2, 'second_send_message', ${JSON.stringify(promptTwo)});
          }
          if (checkMedia && !rendererFailureStage) {
            await sendMessageWithCapture(3, 'image_send_message', 'What is the dominant color of the attached image? Reply with one color word in English.', [
              {
                name: 'real-ui-proof-image.png',
                type: 'image/png',
                kind: 'image',
                dataUrl: imageDataUrl,
              },
            ]);
          }
          if (checkVideo && !rendererFailureStage) {
            if (!videoDataUrl) {
              rendererFailureStage = 'video_data_url_missing';
              sendErrors.push({
                turn: 4,
                stage: 'video_data_url_missing',
                message: 'VMLINUX_REAL_UI_CHECK_VIDEO requires VMLINUX_REAL_UI_VIDEO_DATA_URL',
              });
            } else {
              await sendMessageWithCapture(4, 'video_send_message', 'Describe the attached video briefly in English.', [
                {
                  name: 'real-ui-proof-video.mp4',
                  type: 'video/mp4',
                  kind: 'video',
                  dataUrl: videoDataUrl,
                },
              ]);
            }
          }
          const preloadHealthAfter = await window.api.performance.health(endpoint)
            .catch((error) => ({ error: String(error?.message || error) }));
          const cacheAfter = await window.api.cache.stats(endpoint, remote.session.id)
            .catch((error) => ({ error: String(error?.message || error) }));
          const cacheAfterSettled = await waitForCacheEndpointStorage(cacheAfter, remote.session.id);
          const messages = await window.api.chat.getMessages(chat.id);
          const assistants = messages.filter((m) => m.role === 'assistant');
          const first = assistants[0]?.content || '';
          const second = assistants[assistants.length - 1]?.content || '';
          const parsePersistedArray = (value) => {
            if (!value) return [];
            try {
              const parsed = JSON.parse(value);
              return Array.isArray(parsed) ? parsed : [];
            } catch (_) {
              return [];
            }
          };
          const persistedReasoningByMessage = assistants.map((m) =>
            parsePersistedArray(m.reasoningSegmentsJson)
          );
          const persistedToolsByMessage = assistants.map((m) =>
            parsePersistedArray(m.toolCallsJson)
          );
          const persistedReasoningSegments = persistedReasoningByMessage.flat();
          const persistedTools = persistedToolsByMessage.flat();
          const allAssistantText = assistants.map((m) => m.content || '').join('\\n');
          const visible = first + '\\n' + second;
          const streamTraceByMessage = Object.values(events.stream.reduce((acc, event) => {
            const key = event?.messageId || 'unknown';
            const row = acc[key] || {
              messageId: key,
              count: 0,
              firstFullContent: '',
              lastFullContent: '',
              firstReasoningContent: '',
              lastReasoningContent: '',
              firstMetrics: null,
              lastMetrics: null,
            };
            row.count += 1;
            const fullContent = typeof event?.fullContent === 'string' ? event.fullContent : '';
            const reasoningContent = event?.isReasoning && fullContent ? fullContent : '';
            if (fullContent && !row.firstFullContent) row.firstFullContent = fullContent;
            if (fullContent) row.lastFullContent = fullContent;
            if (reasoningContent && !row.firstReasoningContent) row.firstReasoningContent = reasoningContent;
            if (reasoningContent) row.lastReasoningContent = reasoningContent;
            if (event?.metrics && !row.firstMetrics) row.firstMetrics = event.metrics;
            if (event?.metrics) row.lastMetrics = event.metrics;
            acc[key] = row;
            return acc;
          }, {}));
          const reasoningText = persistedReasoningSegments
            .map((segment) => typeof segment?.text === 'string' ? segment.text : '')
            .filter(Boolean)
            .join('\\n');
          const rawParserLeakRegex = /<think>|<\\/think>|<tool_call>|<\\/tool_call>|<function>|<invoke>|<minimax:tool_call>|<zyphra_tool_call>|<\\|point_start\\|>|<\\|point_end\\|>|<\\|box_start\\|>|<\\|box_end\\|>|<\\|tool_call_start\\|>|<\\|tool_call_end\\|>/;
          const countRegex = (text, regex) => (text.match(regex) || []).length;
          const numericRunRegex = /(?:^|[\\s([{,;:])(?:\\d{1,4}[\\s,;:|\\-/.]+){8,}\\d{1,4}(?=$|[\\s)\\]},;:.])/gm;
          const contentPartsByMessage = messages.map((m) => {
            if (typeof m.content !== 'string') return [];
            try {
              const parsed = JSON.parse(m.content);
              return Array.isArray(parsed) ? parsed : [];
            } catch (_) {
              return [];
            }
          });
          const hasImageAttachment = contentPartsByMessage.some((parts) =>
            parts.some((part) => part?.type === 'image_url' && part?.image_url?.url)
          );
          const hasVideoAttachment = contentPartsByMessage.some((parts) =>
            parts.some((part) => part?.type === 'video_url' && part?.video_url?.url)
          );
          const imageSemanticVerified = checkMedia && new RegExp(imageExpectRegex, 'i').test(allAssistantText);
          const videoSemanticVerified = checkVideo && !!videoExpectRegex && new RegExp(videoExpectRegex, 'i').test(allAssistantText);
          const mediaEvidence = {
            requestedImage: checkMedia,
            requestedVideo: checkVideo,
            imageExpectedRegex: imageExpectRegex,
            videoExpectedRegex: videoExpectRegex,
            imageSemanticVerified,
            videoSemanticVerified,
            imageVerified: checkMedia && hasImageAttachment && imageSemanticVerified && !sendErrors.some((item) => item.turn === 3),
            videoVerified: checkVideo && hasVideoAttachment && videoSemanticVerified && !sendErrors.some((item) => item.turn === 4),
            persistedImageAttachment: hasImageAttachment,
            persistedVideoAttachment: hasVideoAttachment,
          };
          return {
            rendererWireApi: wireApi,
            rendererBuiltinToolsEnabled: builtinToolsEnabled,
            rendererEnableThinking: enableThinking,
            workingDirectory,
            remoteSessionId: remote.session.id,
            remoteSessionStarted: true,
            chatId: chat.id,
            chatOverrides,
            sendErrors,
            rendererFailureStage,
            media: mediaEvidence,
            messageCount: messages.length,
            assistantCount: assistants.length,
            firstAssistantContent: first,
            secondAssistantContent: second,
            persistedReasoningByMessage,
            persistedToolsByMessage,
            persistedReasoningText: reasoningText,
            persistedReasoningCount: persistedReasoningSegments.length,
            persistedToolCount: persistedTools.length,
            streamTraceByMessage,
            turns: messages.map((m) => ({ role: m.role, content: m.content || '' })),
            rawParserLeak: rawParserLeakRegex.test(visible) || rawParserLeakRegex.test(reasoningText),
            reasoningRawParserLeak: rawParserLeakRegex.test(reasoningText),
            reasoningCjkLeakCount: countRegex(reasoningText, /[\\u3400-\\u9FFF]/g),
            reasoningKoreanLeakCount: countRegex(reasoningText, /[\\uAC00-\\uD7AF]/g),
            reasoningNumericRunCount: countRegex(reasoningText, numericRunRegex),
            preloadHealthBefore,
            preloadHealthAfter,
            cacheBefore,
            cacheAfter: cacheAfterSettled,
            eventCounts: {
              stream: events.stream.length,
              tool: events.tool.length,
              reasoningDone: events.reasoningDone.length,
              complete: events.complete.length,
            },
          };
        } finally {
          cleanup.forEach((fn) => { try { fn(); } catch (_) {} });
        }
      })()
    `, 300_000)
    } catch (error) {
      const healthAfter = await requestJson(`${baseUrl}/health`, 5000)
        .catch((healthError) => ({ error: healthError.message }))
      let chatScreenshot = null
      try {
        chatScreenshot = await capturePng(
          cdp,
          path.join(proofDir, `${proofBasename}-chat.png`),
        )
      } catch {}
      const result = {
        generatedAt: new Date().toISOString(),
        status: 'fail',
        failureStage: 'renderer_real_ui_chat',
        error: error?.stack || error?.message || String(error),
        repoDir,
        panelDir,
        script: 'panel/scripts/live-real-ui-model-proof.mjs',
        modelPath,
        modelName,
        servedModel,
        requestedWireApi: wireApi,
        requestedBuiltinTools: builtinToolsEnabled,
        requestedEnableThinking: enableThinkingOverride,
        requestedServerCacheControls: checkServerCacheControls,
        requestedMedia: checkMedia,
        requestedVideo: checkVideo,
        requestContract: {
          promptOne,
          promptTwo,
          requestMaxTokens,
          requestMaxPromptTokens,
          maxToolIterations,
          toolResultMaxChars,
          wireApi,
          builtinToolsEnabled,
          enableThinking: enableThinkingOverride ?? null,
          checkServerCacheControls,
          checkMedia,
          checkVideo,
          expectPagedCacheLocked,
          imageExpectRegex,
          videoExpectRegex,
          cacheExpectRegex,
        },
        baseUrl,
        python,
        runDir,
        userDataDir,
        workingDirectory,
        serverCommand: server.command,
        server: {
          baseUrl,
          healthBefore,
          health: healthAfter,
          models: serverModels,
        },
        cache: {
          before: healthBefore,
          after: healthAfter,
          cacheHitTokens: 0,
        },
        chat: {
          turns: [],
          finalVisibleText: '',
          rawParserTagLeak: false,
          cjkLeakCount: 0,
          koreanLeakCount: 0,
        },
        screenshots: {
          chat: chatScreenshot ? path.resolve(chatScreenshot) : null,
        },
        eventCounts: {
          stream: 0,
          tool: 0,
          reasoningDone: 0,
          complete: 0,
        },
        sendErrors: [error?.message || String(error)],
        rendererFailureStage: 'renderer_real_ui_chat',
        appLogTail: appLogs.slice(-120),
        serverLogTail: server.logs.slice(-160),
      }
      result.visibleAssistantTurnsComplete = visibleAssistantAfterEachUser(result.chat?.turns || [])
      result.liveSpeedSamples = extractLiveSpeedSamples(result)
      result.provenSurfaces = deriveProvenSurfaces(result)
      writeFileSync(
        path.join(proofDir, `${proofBasename}-proof.json`),
        JSON.stringify(result, null, 2),
      )
      if (process.env.VMLINUX_REAL_UI_ALLOW_FAIL === '1' || process.env.VMLX_REAL_UI_ALLOW_FAIL === '1') {
        console.log(JSON.stringify({ ok: false, failures: [result.failureStage], result }, null, 2))
        return
      }
      throw error
    }
    const chatScreenshot = await capturePng(
      cdp,
      path.join(proofDir, `${proofBasename}-chat.png`),
    )
    let serverCacheControls = { requested: false, verified: false }
    if (checkServerCacheControls) {
      try {
        serverCacheControls = await evaluate(cdp, `
        (async () => {
          const wait = (predicate, timeoutMs = 15000) => new Promise((resolve, reject) => {
            const started = Date.now();
            const tick = () => {
              try {
                const value = predicate();
                if (value) return resolve(value);
              } catch (_) {}
              if (Date.now() - started > timeoutMs) {
                return reject(new Error('timeout waiting for server cache UI condition: ' + document.body.innerText.slice(0, 4000)));
              }
              setTimeout(tick, 100);
            };
            tick();
          });
          const modelPath = ${JSON.stringify(modelPath)};
          const cacheExpectRegex = ${JSON.stringify(cacheExpectRegex)};
          const expectPagedCacheLocked = ${JSON.stringify(expectPagedCacheLocked)};
          const updateDismiss = [...document.querySelectorAll('button')]
            .find((b) => b.innerText.includes("Got it"));
          if (updateDismiss) {
            updateDismiss.click();
            await new Promise((resolve) => setTimeout(resolve, 150));
          }
          window.dispatchEvent(new CustomEvent('vmlx:navigate', {
            detail: { mode: 'server', panel: 'create', modelPath }
          }));
          await wait(() => {
            const text = document.body.innerText;
            return text.includes('Create Session')
              && text.includes('Step 2: Configure')
              && text.includes(modelPath)
              && text.includes('Server Settings');
          });
          const sectionClickResults = [];
          const clickSection = async (title) => {
            const sectionButtons = [...document.querySelectorAll('button')];
            const clickable = sectionButtons.find((button) => {
              const text = (button.innerText || '').trim();
              const normalized = text.replace(/\\s+/g, ' ').trim();
              const titleWithoutDisclosure = normalized.replace(/^[▶▸▾▹]\\s*/, '').trim();
              return text === title
                || normalized === title
                || titleWithoutDisclosure === title
                || normalized.includes(title);
            });
            const before = document.body.innerText.includes('Block Disk Cache (L2)');
            if (clickable) {
              clickable.scrollIntoView({ block: 'center' });
              clickable.dispatchEvent(new MouseEvent('mousedown', { bubbles: true, cancelable: true, view: window }));
              clickable.dispatchEvent(new MouseEvent('mouseup', { bubbles: true, cancelable: true, view: window }));
              clickable.dispatchEvent(new MouseEvent('click', { bubbles: true, cancelable: true, view: window }));
              await new Promise((resolve) => setTimeout(resolve, 150));
            }
            const after = document.body.innerText.includes('Block Disk Cache (L2)');
            sectionClickResults.push({
              title,
              found: !!clickable,
              text: (clickable?.innerText || '').trim(),
              blockDiskVisibleBefore: before,
              blockDiskVisibleAfter: after,
            });
            return !!clickable;
          };
          await clickSection('Prefix Cache');
          await clickSection('Paged KV Cache');
          await clickSection('KV Cache Quantization');
          await clickSection('Disk Cache (Persistent)');
          if (!document.body.innerText.includes('Block Disk Cache (L2)')) {
            throw new Error(JSON.stringify({
              message: 'server cache sections did not expand',
              sectionClickResults,
              buttons: [...document.querySelectorAll('button')]
                .map((button) => (button.innerText || '').trim())
                .filter(Boolean),
              textHead: document.body.innerText.slice(0, 1800),
            }, null, 2));
          }
          const labelFor = (text) => [...document.querySelectorAll('label')]
            .find((label) => label.innerText.includes(text));
          const inputFor = (text) => labelFor(text)?.querySelector('input[type="checkbox"]');
          const checkedState = () => ({
            enablePrefixCache: !!inputFor('Enable Prefix Cache')?.checked,
            usePagedCache: !!inputFor('Use Paged KV Cache')?.checked,
            enableBlockDiskCache: !!inputFor('Block Disk Cache (L2)')?.checked,
            enableDiskCache: !!inputFor('Enable Disk Cache')?.checked,
            usePagedCacheDisabled: !!inputFor('Use Paged KV Cache')?.disabled,
            enableDiskCacheDisabled: !!inputFor('Enable Disk Cache')?.disabled,
            blockDiskCachePresent: !!inputFor('Block Disk Cache (L2)'),
          });
          const initialCacheControls = checkedState();
          const bodyText = document.body.innerText;
          const labels = ['Enable Prefix Cache', 'Use Paged KV Cache', 'Block Disk Cache (L2)', 'Enable Disk Cache', 'Stored Cache Quantization']
            .filter((label) => bodyText.includes(label));
          const cacheExpectationMatches = !cacheExpectRegex || new RegExp(cacheExpectRegex, 'i').test(bodyText);
          const verified = labels.length === 5
            && initialCacheControls.enablePrefixCache === true
            && initialCacheControls.usePagedCache === true
            && (!expectPagedCacheLocked || initialCacheControls.usePagedCacheDisabled === true)
            && initialCacheControls.enableDiskCacheDisabled === true
            && initialCacheControls.blockDiskCachePresent === true
            && cacheExpectationMatches;
          return {
            requested: true,
            verified,
            cacheExpectRegex,
            expectPagedCacheLocked,
            cacheExpectationMatches,
            labels,
            initialCacheControls,
            sectionClickResults,
            textHead: bodyText.slice(0, 1600),
          };
        })()
      `, 60_000)
      } catch (error) {
        serverCacheControls = {
          requested: true,
          verified: false,
          error: error?.message || String(error),
        }
      }
    }
    const healthAfter = await requestJson(`${baseUrl}/health`, 5000).catch((error) => ({ error: error.message }))
    const visibleText = rendererResult.secondAssistantContent || rendererResult.firstAssistantContent || ''
    const cacheHitTokens = maxRecursiveNumber(rendererResult, /cached|cacheHit|cache_hit/i)
      || maxRecursiveNumber(healthAfter, /cached|cacheHit|cache_hit/i)
    const toolProbeFiles = {}
    for (const name of ['real_ui_tool_probe_1.txt', 'real_ui_tool_probe_2.txt']) {
      const filePath = path.join(workingDirectory, name)
      if (existsSync(filePath)) {
        toolProbeFiles[name] = readFileSync(filePath, 'utf8')
      }
    }
    const result = {
      generatedAt: new Date().toISOString(),
      status: rendererResult.rendererFailureStage ? 'fail' : 'pass',
      failureStage: rendererResult.rendererFailureStage || undefined,
      repoDir,
      panelDir,
      script: 'panel/scripts/live-real-ui-model-proof.mjs',
      modelPath,
      modelName,
      servedModel,
      requestedWireApi: wireApi,
      requestedBuiltinTools: builtinToolsEnabled,
      requestedEnableThinking: enableThinkingOverride,
      requestedServerCacheControls: checkServerCacheControls,
      requestedMedia: checkMedia,
      requestedVideo: checkVideo,
      requestContract: {
        promptOne,
        promptTwo,
        requestMaxTokens,
        requestMaxPromptTokens,
        maxToolIterations,
        toolResultMaxChars,
        wireApi,
        builtinToolsEnabled,
        enableThinking: enableThinkingOverride ?? null,
        checkServerCacheControls,
        checkMedia,
        checkVideo,
        expectPagedCacheLocked,
        imageExpectRegex,
        videoExpectRegex,
        cacheExpectRegex,
      },
      baseUrl,
      python,
      runDir,
      userDataDir,
      workingDirectory,
      uiLaunchMode: app.uiLaunchMode,
      uiCommand: app.command,
      installedAppPath: app.appPath || undefined,
      serverCommand: server.command,
      server: {
        baseUrl,
        health: healthAfter,
        models: serverModels,
      },
      cache: {
        before: rendererResult.cacheBefore || healthBefore,
        after: rendererResult.cacheAfter || healthAfter,
        cacheHitTokens,
      },
      chat: {
        turns: rendererResult.turns || [],
        finalVisibleText: visibleText,
        rawParserTagLeak: rendererResult.rawParserLeak === true,
        cjkLeakCount: countMatches(visibleText, /[\u3400-\u9FFF]/g),
        koreanLeakCount: countMatches(visibleText, /[\uAC00-\uD7AF]/g),
        reasoningText: rendererResult.persistedReasoningText || '',
        reasoningRawParserTagLeak: rendererResult.reasoningRawParserLeak === true,
        reasoningCjkLeakCount: rendererResult.reasoningCjkLeakCount || 0,
        reasoningKoreanLeakCount: rendererResult.reasoningKoreanLeakCount || 0,
        reasoningNumericRunCount: rendererResult.reasoningNumericRunCount || 0,
      },
      screenshots: {
        chat: path.resolve(chatScreenshot),
      },
      ...rendererResult,
      serverCacheControls,
      toolProbeFiles,
      streamTrace: rendererResult.streamTraceByMessage || [],
      appLogTail: appLogs.slice(-80),
      serverLogTail: server.logs.slice(-120),
    }
    result.visibleAssistantTurnsComplete = visibleAssistantAfterEachUser(result.chat?.turns || [])
    result.liveSpeedSamples = extractLiveSpeedSamples(result)
    result.provenSurfaces = deriveProvenSurfaces(result)
    writeFileSync(
      path.join(proofDir, `${proofBasename}-proof.json`),
      JSON.stringify(result, null, 2),
    )
    if (process.env.VMLINUX_REAL_UI_ALLOW_FAIL === '1' || process.env.VMLX_REAL_UI_ALLOW_FAIL === '1') {
      try {
        assertResult(result)
        console.log(JSON.stringify({ ok: true, result }, null, 2))
      } catch (error) {
        console.log(JSON.stringify({ ok: false, failures: error.failures || [error.message], result }, null, 2))
      }
    } else {
      assertResult(result)
      console.log(JSON.stringify({ ok: true, result }, null, 2))
    }
  } finally {
    if (cdp) cdp.close()
    await terminateProcess(app?.proc)
    await terminateProcess(server.proc)
    await removeTemporaryTree(userDataDir)
    await removeTemporaryTree(runDir)
  }
}

main().catch((error) => {
  console.error(error.stack || error.message || String(error))
  process.exit(1)
})
