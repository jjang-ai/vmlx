#!/usr/bin/env node
import { createServer } from 'node:http'
import net from 'node:net'
import crypto from 'node:crypto'
import { spawn } from 'node:child_process'
import { mkdirSync, writeFileSync, mkdtempSync } from 'node:fs'
import { tmpdir } from 'node:os'
import path from 'node:path'

const panelDir = path.resolve(new URL('..', import.meta.url).pathname)
const repoDir = path.resolve(panelDir, '..')
const proofBasename = process.env.VMLX_LIVE_PROOF_BASENAME
  || `${new Date().toISOString().slice(0, 10)}-live-chat-tools-reasoning`

const sleep = (ms) => new Promise((resolve) => setTimeout(resolve, ms))

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

function safeHttpWrite(res, chunk) {
  if (res.destroyed || res.writableEnded) return false
  try {
    return res.write(chunk)
  } catch (error) {
    if (isSocketDisconnectError(error)) return false
    throw error
  }
}

function safeHttpEnd(res, chunk) {
  if (res.destroyed || res.writableEnded) return false
  try {
    res.end(chunk)
    return true
  } catch (error) {
    if (isSocketDisconnectError(error)) return false
    throw error
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

function collectRequestBody(req) {
  return new Promise((resolve, reject) => {
    const chunks = []
    req.on('data', (chunk) => chunks.push(chunk))
    req.on('end', () => {
      const raw = Buffer.concat(chunks).toString('utf8')
      if (!raw) return resolve({})
      try {
        resolve(JSON.parse(raw))
      } catch (error) {
        reject(error)
      }
    })
    req.on('error', reject)
  })
}

function writeSseEvent(res, event, data) {
  if (event && !safeHttpWrite(res, `event: ${event}\n`)) return false
  return safeHttpWrite(res, `data: ${JSON.stringify(data)}\n\n`)
}

async function startMockServer(workDir) {
  const requests = []
  const port = await freePort()
  let responsesCount = 0
  const server = createServer(async (req, res) => {
    try {
      if (req.method === 'GET' && req.url === '/v1/models') {
        res.writeHead(200, { 'content-type': 'application/json' })
        safeHttpEnd(res, JSON.stringify({ object: 'list', data: [{ id: 'vmlx-live-mock', object: 'model' }] }))
        return
      }
      if (req.method === 'POST' && req.url === '/v1/responses') {
        const body = await collectRequestBody(req)
        responsesCount += 1
        requests.push({ url: req.url, body })
        res.writeHead(200, {
          'content-type': 'text/event-stream; charset=utf-8',
          'cache-control': 'no-cache',
          connection: 'keep-alive',
        })

        const hasToolOutputs = JSON.stringify(body).includes('function_call_output')
        if (!hasToolOutputs && responsesCount === 1) {
          writeSseEvent(res, 'response.created', {
            sequence_number: 1,
            response: { id: 'resp_live_initial', status: 'in_progress' },
          })
          await sleep(25)
          writeSseEvent(res, 'response.reasoning_summary_text.delta', {
            sequence_number: 2,
            delta: 'First plan before tools. ',
          })
          await sleep(25)
          writeSseEvent(res, 'response.reasoning_summary_text.done', { sequence_number: 3 })
          const calls = [
            ['call_live_run', 'run_command', { command: 'sleep 1; echo LIVE_TOOL_OK > tool-output.txt; echo LIVE_TOOL_OK' }],
            ['call_live_list', 'list_directory', { path: '.', recursive: false }],
            ['call_live_image', 'read_image', { path: 'tiny.png' }],
            ['call_live_video', 'read_video', { path: 'clip.mp4' }],
          ]
          let seq = 4
          for (const [callId, name, args] of calls) {
            writeSseEvent(res, 'response.output_item.done', {
              sequence_number: seq++,
              item: {
                type: 'function_call',
                call_id: callId,
                name,
                arguments: JSON.stringify(args),
              },
            })
            await sleep(20)
          }
          writeSseEvent(res, 'response.completed', {
            sequence_number: 20,
            response: {
              id: 'resp_live_initial',
              status: 'completed',
              usage: { input_tokens: 42, output_tokens: 5 },
            },
          })
          safeHttpWrite(res, 'data: [DONE]\n\n')
          safeHttpEnd(res)
          return
        }

        writeSseEvent(res, 'response.created', {
          sequence_number: 30,
          response: { id: 'resp_live_followup', status: 'in_progress' },
        })
        await sleep(25)
        writeSseEvent(res, 'response.reasoning_summary_text.delta', {
          sequence_number: 31,
          delta: 'Second plan after tool results. ',
        })
        await sleep(25)
        writeSseEvent(res, 'response.reasoning_summary_text.done', { sequence_number: 32 })
        await sleep(25)
        writeSseEvent(res, 'response.output_text.delta', {
          sequence_number: 33,
          delta: 'Done after tools.',
        })
        await sleep(25)
        writeSseEvent(res, 'response.completed', {
          sequence_number: 34,
          response: {
            id: 'resp_live_followup',
            status: 'completed',
            usage: { input_tokens: 64, output_tokens: 9 },
          },
        })
        safeHttpWrite(res, 'data: [DONE]\n\n')
        safeHttpEnd(res)
        return
      }

      res.writeHead(404, { 'content-type': 'application/json' })
      safeHttpEnd(res, JSON.stringify({ error: `Unhandled ${req.method} ${req.url}` }))
    } catch (error) {
      if (isSocketDisconnectError(error)) {
        safeHttpEnd(res)
        return
      }
      res.writeHead(500, { 'content-type': 'application/json' })
      safeHttpEnd(res, JSON.stringify({ error: error.message }))
    }
  })

  await new Promise((resolve, reject) => {
    server.listen(port, '127.0.0.1', resolve)
    server.on('error', reject)
  })
  return {
    port,
    requests,
    close: () => new Promise((resolve) => server.close(resolve)),
  }
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
    try {
      socket.write([
        `GET ${url.pathname}${url.search} HTTP/1.1`,
        `Host: ${url.host}`,
        'Upgrade: websocket',
        'Connection: Upgrade',
        `Sec-WebSocket-Key: ${key}`,
        'Sec-WebSocket-Version: 13',
        '\r\n',
      ].join('\r\n'))
    } catch (error) {
      try { socket.destroy() } catch {}
      throw error
    }
    let handshake = Buffer.alloc(0)
    const connected = await new Promise((resolve, reject) => {
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
    return connected
  }

  send(method, params = {}, timeoutMs = 30_000) {
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
      const b0 = this.buffer[0]
      const opcode = b0 & 0x0f
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
  const maskOffset = headerLen
  const mask = crypto.randomBytes(4)
  mask.copy(header, maskOffset)
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

async function evaluate(cdp, expression) {
  const result = await cdp.send('Runtime.evaluate', {
    expression,
    awaitPromise: true,
    returnByValue: true,
    timeout: 30_000,
  })
  if (result.exceptionDetails) {
    throw new Error(JSON.stringify(result.exceptionDetails, null, 2))
  }
  return result.result?.value
}

function assertResult(result) {
  const failures = []
  const b = result.chatBOverrides || {}
  const g = result.generationDefaults || {}
  const expectEq = (actual, expected, label) => {
    if (actual !== expected) failures.push(`${label}: expected ${expected}, got ${actual}`)
  }
  expectEq(g.temperature, 0.42, 'generation default temperature')
  expectEq(g.topP, 0.77, 'generation default topP')
  expectEq(g.topK, 33, 'generation default topK')
  expectEq(g.maxNewTokens, 2048, 'generation default maxNewTokens')
  expectEq(g.repeatPenalty, 1.07, 'generation default repeatPenalty')
  expectEq(g.source, 'generation_config', 'generation default source')
  for (const key of ['temperature', 'topP', 'topK', 'minP', 'maxTokens', 'repeatPenalty']) {
    if (b[key] != null) failures.push(`sampler override carried over ${key}: ${b[key]}`)
  }
  expectEq(b.builtinToolsEnabled, true, 'tool top-level toggle')
  expectEq(b.shellEnabled, true, 'shell toggle')
  expectEq(b.fileToolsEnabled, false, 'file tools disabled toggle')
  expectEq(b.searchToolsEnabled, true, 'search toggle')
  expectEq(b.gitEnabled, false, 'git disabled toggle')
  expectEq(b.utilityToolsEnabled, true, 'utility toggle')
  expectEq(b.workingDirectory, result.workDir, 'working directory')
  expectEq(b.maxToolIterations, 4, 'max tool iterations')
  expectEq(b.toolResultMaxChars, 12345, 'tool result max chars')
  if (b.wireApi != null) failures.push(`wireApi carried over: ${b.wireApi}`)
  if (b.stopSequences != null) failures.push(`stopSequences carried over: ${b.stopSequences}`)
  if (b.systemPrompt != null) failures.push(`systemPrompt carried over: ${b.systemPrompt}`)
  if (b.enableThinking !== undefined) failures.push(`enableThinking carried over: ${b.enableThinking}`)
  if (b.reasoningEffort !== undefined) failures.push(`reasoningEffort carried over: ${b.reasoningEffort}`)

  const phases = result.toolPhasesById || {}
  for (const id of ['call_live_run', 'call_live_list', 'call_live_image', 'call_live_video']) {
    const seen = phases[id] || []
    for (const phase of ['calling', 'executing', 'result']) {
      if (!seen.includes(phase)) failures.push(`${id} missing ${phase}`)
    }
  }
  if (!(result.runToolExecutingToResultMs >= 800)) {
    failures.push(`run_command executing->result gap too small: ${result.runToolExecutingToResultMs}`)
  }
  if (result.mockResponsesRequests !== 2) {
    failures.push(`expected exactly 2 responses requests, got ${result.mockResponsesRequests}`)
  }
  if (result.finalAssistantContent !== 'Done after tools.') {
    failures.push(`final assistant content mismatch: ${result.finalAssistantContent}`)
  }
  const segments = result.persistedReasoningSegments || []
  if (segments.length !== 2 || !segments[0].includes('First plan') || !segments[1].includes('Second plan')) {
    failures.push(`reasoning segments mismatch: ${JSON.stringify(segments)}`)
  }
  const tailTypes = result.followupTailContentTypes || []
  if (JSON.stringify(tailTypes) !== JSON.stringify(['text', 'image_url', 'video_url'])) {
    failures.push(`follow-up media content types mismatch: ${JSON.stringify(tailTypes)}`)
  }
  const payload = result.initialPayloadKeys || {}
  for (const key of ['temperature', 'top_p', 'top_k', 'max_output_tokens']) {
    if (payload[key]) failures.push(`unexpected stale sampler key in initial request: ${key}`)
  }
  if (result.rawThinkTagLeak) failures.push('raw think/tool tag leaked into final content')

  const chatUi = result.chatSettingsUi || {}
  if (!chatUi.visible) failures.push('chat settings drawer was not visible in renderer')
  for (const label of ['Enable Built-in Coding Tools', 'Working Directory', 'Shell', 'Search', 'Utilities', 'Hide Tool Status']) {
    if (!chatUi.labels?.includes(label)) failures.push(`chat settings missing visible label: ${label}`)
  }
  if (chatUi.checked?.builtinToolsEnabled !== true) failures.push('chat settings builtin tools checkbox is not checked')
  if (chatUi.checked?.shellEnabled !== true) failures.push('chat settings shell checkbox is not checked')
  if (chatUi.checked?.fileToolsEnabled !== false) failures.push('chat settings file tools checkbox should be unchecked')
  if (chatUi.checked?.gitEnabled !== false) failures.push('chat settings git checkbox should be unchecked')
  if (chatUi.checked?.hideToolStatus !== true) failures.push('chat settings hide tool status checkbox is not checked')

  const serverUi = result.serverCacheUi || {}
  if (!serverUi.visible) failures.push('server cache settings UI was not visible in renderer')
  for (const label of ['Enable Prefix Cache', 'Use Paged KV Cache', 'Block Disk Cache (L2)', 'Enable Disk Cache', 'Stored Cache Quantization']) {
    if (!serverUi.labels?.includes(label)) failures.push(`server cache UI missing visible label: ${label}`)
  }
  if (serverUi.afterBlockDiskToggle?.enablePrefixCache !== true) failures.push('block disk toggle did not enable prefix cache')
  if (serverUi.afterBlockDiskToggle?.usePagedCache !== true) failures.push('block disk toggle did not enable paged cache')
  if (serverUi.afterBlockDiskToggle?.enableBlockDiskCache !== true) failures.push('block disk toggle did not enable block disk cache')
  if (serverUi.afterDiskToggle?.enablePrefixCache !== true) failures.push('legacy disk toggle did not enable prefix cache')
  if (serverUi.afterDiskToggle?.usePagedCache !== false) failures.push('legacy disk toggle did not clear paged cache')
  if (serverUi.afterDiskToggle?.enableDiskCache !== true) failures.push('legacy disk toggle did not enable disk cache')

  if (failures.length) {
    const error = new Error(`Live proof failed:\n- ${failures.join('\n- ')}`)
    error.failures = failures
    throw error
  }
}

async function capturePng(cdp, filePath) {
  const shot = await cdp.send('Page.captureScreenshot', {
    format: 'png',
    captureBeyondViewport: true,
  })
  writeFileSync(filePath, Buffer.from(shot.data, 'base64'))
  return filePath
}

async function main() {
  const workDir = mkdtempSync(path.join(tmpdir(), 'vmlx-live-tools-'))
  const modelDir = mkdtempSync(path.join(tmpdir(), 'vmlx-live-model-'))
  const userDataDir = mkdtempSync(path.join(tmpdir(), 'vmlx-live-userdata-'))
  mkdirSync(workDir, { recursive: true })
  writeFileSync(path.join(workDir, 'tiny.png'), Buffer.from(
    'iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/x8AAwMCAO+/p9sAAAAASUVORK5CYII=',
    'base64',
  ))
  writeFileSync(path.join(workDir, 'clip.mp4'), Buffer.from('live proof video placeholder\n'))
  writeFileSync(path.join(modelDir, 'config.json'), JSON.stringify({ model_type: 'qwen3_5' }, null, 2))
  writeFileSync(path.join(modelDir, 'generation_config.json'), JSON.stringify({
    temperature: 0.42,
    top_p: 0.77,
    top_k: 33,
    max_new_tokens: 2048,
    repetition_penalty: 1.07,
  }, null, 2))

  const mock = await startMockServer(workDir)
  const debugPort = await freePort()
  const appLogs = []
  const app = spawn('npm', [
    'run',
    'dev',
    '--',
    '--',
    `--user-data-dir=${userDataDir}`,
    `--remote-debugging-port=${debugPort}`,
  ], {
    cwd: panelDir,
    env: { ...process.env, VMLX_SKIP_UPDATE_CHECK: '1' },
    detached: true,
    stdio: ['ignore', 'pipe', 'pipe'],
  })
  app.stdout.on('data', (d) => appLogs.push(...d.toString().split(/\r?\n/).filter(Boolean)))
  app.stderr.on('data', (d) => appLogs.push(...d.toString().split(/\r?\n/).filter(Boolean)))
  attachChildProcessStreamErrorGuard(app.stdout, appLogs)
  attachChildProcessStreamErrorGuard(app.stderr, appLogs)

  let cdp
  try {
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
    const waitExpression = `
      new Promise((resolve, reject) => {
        const started = Date.now();
        const check = () => {
          if (window.api?.chat && window.api?.sessions) resolve(true);
          else if (Date.now() - started > 30000) reject(new Error('window.api not ready'));
          else setTimeout(check, 100);
        };
        check();
      })
    `
    await evaluate(cdp, waitExpression)
    const expression = `
      (async () => {
        const mockPort = ${JSON.stringify(mock.port)};
        const modelDir = ${JSON.stringify(modelDir)};
        const workDir = ${JSON.stringify(workDir)};
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
        await new Promise((resolve) => setTimeout(resolve, 1500));
        await window.api.chat.clearAllLocks().catch(() => null);
        await new Promise((resolve) => setTimeout(resolve, 250));
        const events = { stream: [], tool: [], reasoningDone: [], complete: [] };
        const cleanup = [
          window.api.chat.onStream((data) => events.stream.push({ t: performance.now(), ...data })),
          window.api.chat.onToolStatus((data) => events.tool.push({ t: performance.now(), ...data })),
          window.api.chat.onReasoningDone((data) => events.reasoningDone.push({ t: performance.now(), ...data })),
          window.api.chat.onComplete((data) => events.complete.push({ t: performance.now(), ...data })),
        ];
        try {
          const chatA = await window.api.chat.create('Settings A', 'live-local-model', undefined, modelDir);
          await window.api.chat.setOverrides(chatA.id, {
            chatId: chatA.id,
            temperature: 1.0,
            topP: 1.0,
            topK: 1,
            minP: 0.5,
            maxTokens: 128,
            repeatPenalty: 1.3,
            stopSequences: '<bad>',
            wireApi: 'completions',
            systemPrompt: 'sticky prompt that must not survive',
            enableThinking: true,
            reasoningEffort: 'max',
            builtinToolsEnabled: true,
            shellEnabled: true,
            fileToolsEnabled: false,
            searchToolsEnabled: true,
            gitEnabled: false,
            utilityToolsEnabled: true,
            webSearchEnabled: true,
            braveSearchEnabled: false,
            fetchUrlEnabled: true,
            workingDirectory: workDir,
            hideToolStatus: true,
            maxToolIterations: 4,
            toolResultMaxChars: 12345,
          });
          const chatB = await window.api.chat.create('Settings B', 'live-local-model', undefined, modelDir);
          const chatBOverrides = await window.api.chat.getOverrides(chatB.id);
          const generationDefaults = await window.api.models.getGenerationDefaults(modelDir);

          const remote = await window.api.sessions.createRemote({
            remoteUrl: 'http://127.0.0.1:' + mockPort,
            remoteModel: 'vmlx-live-mock',
          });
          if (!remote.success) throw new Error(remote.error || 'remote session create failed');
          await window.api.sessions.start(remote.session.id);
          const toolChat = await window.api.chat.create('Live tool proof', 'vmlx-live-mock', undefined, remote.session.modelPath);
          await window.api.chat.setOverrides(toolChat.id, {
            chatId: toolChat.id,
            wireApi: 'responses',
            builtinToolsEnabled: true,
            shellEnabled: true,
            fileToolsEnabled: true,
            searchToolsEnabled: true,
            gitEnabled: false,
            utilityToolsEnabled: true,
            webSearchEnabled: false,
            braveSearchEnabled: false,
            fetchUrlEnabled: false,
            workingDirectory: workDir,
            maxToolIterations: 4,
            toolResultMaxChars: 12345,
          });
          const assistant = await window.api.chat.sendMessage(toolChat.id, 'Use the available tools, inspect the local media, then answer.');
          const messages = await window.api.chat.getMessages(toolChat.id);
          const finalAssistant = [...messages].reverse().find((m) => m.role === 'assistant');
          const persistedReasoningSegments = finalAssistant?.reasoningSegmentsJson
            ? JSON.parse(finalAssistant.reasoningSegmentsJson)
            : [];
          const persistedTools = finalAssistant?.toolCallsJson
            ? JSON.parse(finalAssistant.toolCallsJson)
            : [];
          const uiChat = await window.api.chat.create('UI settings proof', 'vmlx-live-mock', undefined, remote.session.modelPath);
          await window.api.chat.setOverrides(uiChat.id, {
            chatId: uiChat.id,
            wireApi: 'responses',
            builtinToolsEnabled: true,
            shellEnabled: true,
            fileToolsEnabled: false,
            searchToolsEnabled: true,
            gitEnabled: false,
            utilityToolsEnabled: true,
            webSearchEnabled: true,
            braveSearchEnabled: false,
            fetchUrlEnabled: true,
            workingDirectory: workDir,
            hideToolStatus: true,
            maxToolIterations: 4,
            toolResultMaxChars: 12345,
          });
          const toolPhasesById = {};
          for (const event of events.tool) {
            if (!event.toolCallId) continue;
            toolPhasesById[event.toolCallId] ||= [];
            toolPhasesById[event.toolCallId].push(event.phase);
          }
          const runExecuting = events.tool.find((e) => e.toolCallId === 'call_live_run' && e.phase === 'executing');
          const runResult = events.tool.find((e) => e.toolCallId === 'call_live_run' && e.phase === 'result');
          const rawFinal = finalAssistant?.content || '';
          return {
            workDir,
            modelDir,
            userDataDir: ${JSON.stringify(userDataDir)},
            remoteSessionId: remote.session.id,
            uiChatId: uiChat.id,
            chatBOverrides,
            generationDefaults,
            assistantId: assistant?.id,
            finalAssistantContent: rawFinal,
            persistedReasoningSegments,
            persistedToolCount: persistedTools.length,
            eventCounts: {
              stream: events.stream.length,
              tool: events.tool.length,
              reasoningDone: events.reasoningDone.length,
              complete: events.complete.length,
            },
            toolPhasesById,
            runToolExecutingToResultMs: runExecuting && runResult ? Math.round(runResult.t - runExecuting.t) : null,
            rawThinkTagLeak: /<think>|<\\/think>|<tool_call>|<minimax:tool_call>/.test(rawFinal),
          };
        } finally {
          cleanup.forEach((fn) => { try { fn(); } catch (_) {} });
        }
      })()
    `
    const rendererResult = await evaluate(cdp, expression)
    const proofDir = path.join(repoDir, 'docs', 'internal', 'agent-notes')
    mkdirSync(proofDir, { recursive: true })
    const chatSettingsUi = await evaluate(cdp, `
      (async () => {
        const wait = (predicate, timeoutMs = 10000) => new Promise((resolve, reject) => {
          const started = Date.now();
          const tick = () => {
            try {
              const value = predicate();
              if (value) return resolve(value);
            } catch (_) {}
            if (Date.now() - started > timeoutMs) {
              return reject(new Error('timeout waiting for UI condition: ' + document.body.innerText.slice(0, 4000)));
            }
            setTimeout(tick, 100);
          };
          tick();
        });
        const updateDismiss = [...document.querySelectorAll('button')]
          .find((b) => b.innerText.includes("Got it"));
        if (updateDismiss) {
          updateDismiss.click();
          await new Promise((resolve) => setTimeout(resolve, 150));
        }
        window.dispatchEvent(new CustomEvent('vmlx:navigate', {
          detail: { mode: 'server', panel: 'session', sessionId: ${JSON.stringify(rendererResult.remoteSessionId)} }
        }));
        const chatButton = await wait(() => document.querySelector('button[title="Chat inference settings"]'));
        if (!chatButton) throw new Error('Chat settings button not found');
        chatButton.click();
        await wait(() => document.body.innerText.includes('Enable Built-in Coding Tools'));
        const bodyText = document.body.innerText;
        const labelFor = (text) => [...document.querySelectorAll('label')]
          .find((label) => label.innerText.includes(text));
        const inputFor = (text) => labelFor(text)?.querySelector('input[type="checkbox"]');
        const labels = ['Enable Built-in Coding Tools', 'Working Directory', 'Shell', 'Search', 'Utilities', 'Hide Tool Status']
          .filter((label) => bodyText.includes(label));
        labelFor('Enable Built-in Coding Tools')?.scrollIntoView({ block: 'center' });
        await new Promise((resolve) => setTimeout(resolve, 150));
        return {
          visible: true,
          labels,
          checked: {
            builtinToolsEnabled: !!inputFor('Enable Built-in Coding Tools')?.checked,
            shellEnabled: !!inputFor('Shell')?.checked,
            fileToolsEnabled: !!inputFor('File I/O')?.checked,
            gitEnabled: !!inputFor('Git')?.checked,
            hideToolStatus: !!inputFor('Hide Tool Status')?.checked,
          },
          textHead: bodyText.slice(0, 1200),
        };
      })()
    `)
    const chatSettingsScreenshot = await capturePng(
      cdp,
      path.join(proofDir, `${proofBasename}-chat-settings.png`),
    )
    const serverCacheUi = await evaluate(cdp, `
      (async () => {
        const wait = (predicate, timeoutMs = 15000) => new Promise((resolve, reject) => {
          const started = Date.now();
          const tick = () => {
            try {
              const value = predicate();
              if (value) return resolve(value);
            } catch (_) {}
            if (Date.now() - started > timeoutMs) {
              return reject(new Error('timeout waiting for UI condition: ' + document.body.innerText.slice(0, 4000)));
            }
            setTimeout(tick, 100);
          };
          tick();
        });
        const modelPath = ${JSON.stringify(modelDir)};
        window.dispatchEvent(new CustomEvent('vmlx:navigate', {
          detail: { mode: 'server', panel: 'create', modelPath }
        }));
        await wait(() => document.body.innerText.includes('Server Settings'));
        const clickSection = (title) => {
          const button = [...document.querySelectorAll('button')]
            .find((b) => b.innerText.includes(title));
          if (button) button.click();
          return !!button;
        };
        clickSection('Prefix Cache');
        clickSection('Paged KV Cache');
        clickSection('KV Cache Quantization');
        clickSection('Disk Cache (Persistent)');
        await wait(() => document.body.innerText.includes('Block Disk Cache (L2)'));
        const labelFor = (text) => [...document.querySelectorAll('label')]
          .find((label) => label.innerText.includes(text));
        const inputFor = (text) => labelFor(text)?.querySelector('input[type="checkbox"]');
        const checkedState = () => ({
          enablePrefixCache: !!inputFor('Enable Prefix Cache')?.checked,
          usePagedCache: !!inputFor('Use Paged KV Cache')?.checked,
          enableBlockDiskCache: !!inputFor('Block Disk Cache (L2)')?.checked,
          enableDiskCache: !!inputFor('Enable Disk Cache')?.checked,
        });
        const prefix = inputFor('Enable Prefix Cache');
        if (prefix?.checked) prefix.click();
        await new Promise((resolve) => setTimeout(resolve, 100));
        const paged = inputFor('Use Paged KV Cache');
        if (paged?.checked) paged.click();
        await new Promise((resolve) => setTimeout(resolve, 100));
        const block = inputFor('Block Disk Cache (L2)');
        if (!block) throw new Error('Block Disk Cache checkbox not found');
        if (!block.checked) block.click();
        await wait(() => inputFor('Enable Prefix Cache')?.checked && inputFor('Use Paged KV Cache')?.checked && inputFor('Block Disk Cache (L2)')?.checked);
        const afterBlockDiskToggle = checkedState();
        if (inputFor('Block Disk Cache (L2)')?.checked) inputFor('Block Disk Cache (L2)').click();
        await wait(() => !inputFor('Block Disk Cache (L2)')?.checked);
        if (inputFor('Use Paged KV Cache')?.checked) inputFor('Use Paged KV Cache').click();
        await wait(() => !inputFor('Use Paged KV Cache')?.checked);
        if (inputFor('Enable Prefix Cache')?.checked) inputFor('Enable Prefix Cache').click();
        await wait(() => !inputFor('Enable Prefix Cache')?.checked);
        const legacyDisk = inputFor('Enable Disk Cache');
        if (!legacyDisk) throw new Error('Enable Disk Cache checkbox not found');
        await wait(() => !inputFor('Enable Disk Cache')?.disabled);
        if (!legacyDisk.checked) legacyDisk.click();
        await wait(() => inputFor('Enable Prefix Cache')?.checked && !inputFor('Use Paged KV Cache')?.checked && inputFor('Enable Disk Cache')?.checked);
        const afterDiskToggle = checkedState();
        const bodyText = document.body.innerText;
        const labels = ['Enable Prefix Cache', 'Use Paged KV Cache', 'Block Disk Cache (L2)', 'Enable Disk Cache', 'Stored Cache Quantization']
          .filter((label) => bodyText.includes(label));
        labelFor('Enable Disk Cache')?.scrollIntoView({ block: 'center' });
        await new Promise((resolve) => setTimeout(resolve, 150));
        return {
          visible: true,
          labels,
          afterBlockDiskToggle,
          afterDiskToggle,
          textHead: bodyText.slice(0, 1600),
        };
      })()
    `)
    const serverCacheScreenshot = await capturePng(
      cdp,
      path.join(proofDir, `${proofBasename}-server-cache-settings.png`),
    )
    const firstBody = mock.requests[0]?.body || {}
    const secondBody = mock.requests[1]?.body || {}
    const followupTail = Array.isArray(secondBody.input) ? secondBody.input[secondBody.input.length - 1] : undefined
    const result = {
      generatedAt: new Date().toISOString(),
      repoDir,
      panelDir,
      mockPort: mock.port,
      debugPort,
      ...rendererResult,
      chatSettingsUi,
      serverCacheUi,
      screenshots: {
        chatSettings: chatSettingsScreenshot,
        serverCacheSettings: serverCacheScreenshot,
      },
      mockResponsesRequests: mock.requests.length,
      initialPayloadKeys: {
        temperature: Object.prototype.hasOwnProperty.call(firstBody, 'temperature'),
        top_p: Object.prototype.hasOwnProperty.call(firstBody, 'top_p'),
        top_k: Object.prototype.hasOwnProperty.call(firstBody, 'top_k'),
        max_output_tokens: Object.prototype.hasOwnProperty.call(firstBody, 'max_output_tokens'),
      },
      followupTailContentTypes: Array.isArray(followupTail?.content)
        ? followupTail.content.map((part) => part.type)
        : [],
      mockRequestSummaries: mock.requests.map((r, index) => ({
        index,
        inputCount: Array.isArray(r.body.input) ? r.body.input.length : null,
        hasTools: Array.isArray(r.body.tools),
        hasToolOutput: JSON.stringify(r.body).includes('function_call_output'),
        hasImageUrl: JSON.stringify(r.body).includes('"image_url"'),
        hasVideoUrl: JSON.stringify(r.body).includes('"video_url"'),
      })),
      appLogTail: appLogs.slice(-80),
    }
    writeFileSync(
      path.join(proofDir, `${proofBasename}-proof.json`),
      JSON.stringify(result, null, 2),
    )
    if (process.env.VMLX_LIVE_PROOF_ALLOW_FAIL === '1') {
      try {
        assertResult(result)
        console.log(JSON.stringify({ ok: true, result }, null, 2))
      } catch (error) {
        console.log(JSON.stringify({
          ok: false,
          failures: error.failures || [error.message],
          result,
        }, null, 2))
      }
    } else {
      assertResult(result)
      console.log(JSON.stringify({ ok: true, result }, null, 2))
    }
  } finally {
    if (cdp) cdp.close()
    await mock.close()
    try { process.kill(-app.pid, 'SIGTERM') } catch {}
    await sleep(1500)
    try { process.kill(-app.pid, 'SIGKILL') } catch {}
  }
}

main().catch((error) => {
  console.error(error.stack || error.message || String(error))
  process.exit(1)
})
