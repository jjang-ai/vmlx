#!/usr/bin/env node
import { spawn, spawnSync } from 'node:child_process'
import { createServer } from 'node:http'
import { existsSync, mkdirSync, mkdtempSync, readFileSync, rmSync, writeFileSync } from 'node:fs'
import { tmpdir } from 'node:os'
import path from 'node:path'
import { fileURLToPath } from 'node:url'
import { deflateSync } from 'node:zlib'
import { chromium } from 'playwright-core'

const __dirname = path.dirname(fileURLToPath(import.meta.url))
const panelDir = path.resolve(__dirname, '..')
const repoDir = path.resolve(panelDir, '..')

const rowName = process.env.VMLX_GEMMA_ROW || 'gemma4-e2b-mxfp4'
const modelPath = process.env.VMLX_GEMMA_MODEL_PATH
  || '/Users/example/models/OsaurusAI--gemma-4-E2B-it-qat-MXFP4'
const appPath = process.env.VMLX_APP_PATH || '/Applications/vMLX.app'
const appExe = path.join(appPath, 'Contents', 'MacOS', 'vMLX')
// Gemma audio is intentionally out of the current release gate. Keep audio off
// unless a future audio-specific rerun explicitly opts in.
const expectAudio = process.env.VMLX_GEMMA_EXPECT_AUDIO === '1'
const expectImage = process.env.VMLX_GEMMA_EXPECT_IMAGE !== '0'
const startedAt = new Date()
const stamp = startedAt.toISOString().replace(/[:.]/g, '-')
const proofDir = path.resolve(process.env.VMLX_GEMMA_PROOF_DIR || path.join(repoDir, 'build', `live-gemma4-media-${rowName}-${stamp}`))
const outJson = path.join(proofDir, 'gemma4-media-proof.json')
const shotPath = path.join(proofDir, 'gemma4-media-final.png')

const sleep = (ms) => new Promise((resolve) => setTimeout(resolve, ms))

function freePort() {
  return new Promise((resolve, reject) => {
    const server = createServer()
    server.listen(0, '127.0.0.1', () => {
      const port = server.address().port
      server.close(() => resolve(port))
    })
    server.on('error', reject)
  })
}

function words(text) {
  return String(text || '')
    .toLowerCase()
    .replace(/[^a-z0-9_]+/g, ' ')
    .trim()
    .split(/\s+/)
    .filter(Boolean)
}

function maxAdjacentPhraseRepeats(text) {
  const w = words(text)
  const best = { count: 0, n: 0, phrase: '' }
  for (let n = 2; n <= 10; n += 1) {
    for (let i = 0; i + n * 2 <= w.length; i += 1) {
      const phrase = w.slice(i, i + n).join(' ')
      let count = 1
      let j = i + n
      while (j + n <= w.length && w.slice(j, j + n).join(' ') === phrase) {
        count += 1
        j += n
      }
      if (count > best.count) {
        best.count = count
        best.n = n
        best.phrase = phrase
      }
    }
  }
  return best
}

function scoreText(text) {
  const value = String(text || '')
  const lines = value.split(/\n+/).map((line) => line.trim().replace(/\s+/g, ' ')).filter(Boolean)
  let maxLineRun = 0
  let previous = ''
  let run = 0
  for (const line of lines) {
    if (line === previous) run += 1
    else {
      previous = line
      run = 1
    }
    maxLineRun = Math.max(maxLineRun, run)
  }
  const phrase = maxAdjacentPhraseRepeats(value)
  return {
    chars: value.length,
    wordCount: words(value).length,
    empty: value.trim().length === 0,
    leakedReasoningTags: /<\|channel\>|<channel\|>|<\|start\>|<\|end\>|<think>|<\/think>|<mm:think/i.test(value),
    maxLineRun,
    adjacentPhraseRepeat: phrase,
    loopSuspect: maxLineRun >= 4 || phrase.count >= 6,
    preview: value.trim().slice(0, 800),
  }
}

function labelText(text) {
  return String(text || '')
    .replace(/\\_/g, '_')
    .replace(/&lowbar;/gi, '_')
}

function startsWithExactMarker(text, expected) {
  return labelText(text).trimStart().startsWith(expected)
}

function hasExactMarker(text, expected) {
  return labelText(text).includes(expected)
}

function toolCallingCount(row, toolName = '') {
  try {
    const events = JSON.parse(row?.toolCallsJson || '[]')
    return events.filter((event) => (
      event?.phase === 'calling'
      && (!toolName || event?.toolName === toolName)
    )).length
  } catch {
    return 0
  }
}

function parseMetrics(message) {
  try {
    return message?.metricsJson ? JSON.parse(message.metricsJson) : {}
  } catch {
    return {}
  }
}

function responseTextFromResponsesObject(obj) {
  if (!obj || typeof obj !== 'object') return ''
  if (typeof obj.output_text === 'string' && obj.output_text) return obj.output_text
  const chunks = []
  for (const item of obj.output || []) {
    if (typeof item?.content === 'string') chunks.push(item.content)
    for (const part of item?.content || []) {
      if (typeof part?.text === 'string') chunks.push(part.text)
    }
  }
  return chunks.join('\n')
}

function cacheTokensFromUsage(obj) {
  const details = obj?.usage?.prompt_tokens_details || obj?.usage?.input_tokens_details || {}
  return Number(details.cached_tokens || details.cache_read_input_tokens || 0)
}

function readJsonMaybe(file) {
  try {
    return JSON.parse(readFileSync(file, 'utf8'))
  } catch {
    return null
  }
}

function modelGenerationDefaults(modelPath) {
  const gen = readJsonMaybe(path.join(modelPath, 'generation_config.json')) || {}
  return {
    source: existsSync(path.join(modelPath, 'generation_config.json')) ? 'generation_config' : 'missing',
    do_sample: gen.do_sample,
    temperature: typeof gen.temperature === 'number' ? gen.temperature : null,
    top_p: typeof gen.top_p === 'number' ? gen.top_p : null,
    top_k: typeof gen.top_k === 'number' ? gen.top_k : null,
    min_p: typeof gen.min_p === 'number' ? gen.min_p : null,
    eos_token_id: gen.eos_token_id ?? null,
  }
}

function sessionGenerationDefaults(config) {
  return {
    temperature: typeof config?.defaultTemperature === 'number' ? config.defaultTemperature / 100 : null,
    top_p: typeof config?.defaultTopP === 'number' ? config.defaultTopP / 100 : null,
    top_k: typeof config?.defaultTopK === 'number' ? config.defaultTopK : null,
    declared: config?.defaultSamplingDefaultsDeclared === true,
    source: config?.generationStartupDefaultsSource || config?.source || null,
    generationStartupDefaultsVersion: config?.generationStartupDefaultsVersion ?? null,
  }
}

function defaultsMatch(modelValue, sessionValue) {
  if (modelValue == null || sessionValue == null) return modelValue == null && sessionValue == null
  return Math.abs(Number(modelValue) - Number(sessionValue)) < 1e-6
}

async function captureGenerationDefaults(page, modelPath, sessionConfig) {
  const model = modelGenerationDefaults(modelPath)
  const session = sessionGenerationDefaults(sessionConfig)
  const uiSteps = { noticeDismissed: false, navigated: false, panelClicked: false, error: null }
  await page.evaluate(async ({ sessionId }) => {
    const bounded = async (promise, ms = 750) => {
      try {
        return await Promise.race([
          Promise.resolve(promise),
          new Promise((resolve) => setTimeout(() => resolve(null), ms)),
        ])
      } catch {
        return null
      }
    }
    await bounded(window.api?.settings?.set?.('notice_dismissed_version', '1.5.45'))
    for (const button of Array.from(document.querySelectorAll('button'))) {
      if ((button.textContent || '').includes("Got it")) {
        button.click()
        break
      }
    }
    window.dispatchEvent(new CustomEvent('vmlx:navigate', {
      detail: { mode: 'server', panel: 'settings', sessionId },
    }))
    await new Promise((resolve) => setTimeout(resolve, 1200))
    let clicked = false
    for (const el of Array.from(document.querySelectorAll('button, [role="button"]'))) {
      if ((el.textContent || '').includes('Performance & Generation')) {
        el.click()
        clicked = true
        break
      }
    }
    await new Promise((resolve) => setTimeout(resolve, 800))
    return { noticeDismissed: true, navigated: true, panelClicked: clicked }
  }, { sessionId: sessionConfig?.id || null }).catch(() => null)
    .then((value) => {
      if (value && typeof value === 'object') Object.assign(uiSteps, value)
    }, (error) => {
      uiSteps.error = String(error?.message || error)
    })
  await page.waitForFunction(() => /Generation defaults are resolved/i.test(document.body.innerText || ''), null, { timeout: 10_000 }).catch(() => null)
  const bodyText = await page.evaluate(() => document.body.innerText || '').catch(() => '')
  const parity = {
    temperature: defaultsMatch(model.temperature, session.temperature),
    top_p: defaultsMatch(model.top_p, session.top_p),
    top_k: defaultsMatch(model.top_k, session.top_k),
  }
  const expectedTemperature = session.temperature != null ? `temperature ${Number(session.temperature).toFixed(2)}` : null
  const expectedTopP = session.top_p != null ? `top-p ${Number(session.top_p).toFixed(2)}` : null
  const expectedTopK = session.top_k && session.top_k > 0 ? `top-k ${Math.floor(session.top_k)}` : 'top-k off'
  const settingsTextVisible = /Generation defaults are resolved/i.test(bodyText) && /Current model-declared values/i.test(bodyText)
  return {
    model,
    session,
    parity,
    uiVisibleProbe: {
      navigation: uiSteps,
      bodyTextCaptured: bodyText.length > 0,
      settingsTextVisible,
      expectedTemperature,
      expectedTopP,
      expectedTopK,
      temperatureValueVisible: expectedTemperature ? bodyText.includes(expectedTemperature) : true,
      topPValueVisible: expectedTopP ? bodyText.includes(expectedTopP) : true,
      topKValueVisible: expectedTopK ? bodyText.includes(expectedTopK) : true,
      bodyPreview: bodyText.slice(0, 1200),
    },
  }
}

function crc32(buf) {
  let crc = 0xffffffff
  for (const byte of buf) {
    crc ^= byte
    for (let i = 0; i < 8; i += 1) crc = (crc >>> 1) ^ (0xedb88320 & -(crc & 1))
  }
  return (crc ^ 0xffffffff) >>> 0
}

function pngChunk(type, data) {
  const typeBuf = Buffer.from(type, 'ascii')
  const len = Buffer.alloc(4)
  len.writeUInt32BE(data.length, 0)
  const sum = Buffer.alloc(4)
  sum.writeUInt32BE(crc32(Buffer.concat([typeBuf, data])), 0)
  return Buffer.concat([len, typeBuf, data, sum])
}

function solidPngDataUrl(width, height, [r, g, b]) {
  const ihdr = Buffer.alloc(13)
  ihdr.writeUInt32BE(width, 0)
  ihdr.writeUInt32BE(height, 4)
  ihdr[8] = 8
  ihdr[9] = 2
  ihdr[10] = 0
  ihdr[11] = 0
  ihdr[12] = 0
  const rows = []
  for (let y = 0; y < height; y += 1) {
    const row = Buffer.alloc(1 + width * 3)
    row[0] = 0
    for (let x = 0; x < width; x += 1) {
      const offset = 1 + x * 3
      row[offset] = r
      row[offset + 1] = g
      row[offset + 2] = b
    }
    rows.push(row)
  }
  const png = Buffer.concat([
    Buffer.from([0x89, 0x50, 0x4e, 0x47, 0x0d, 0x0a, 0x1a, 0x0a]),
    pngChunk('IHDR', ihdr),
    pngChunk('IDAT', deflateSync(Buffer.concat(rows))),
    pngChunk('IEND', Buffer.alloc(0)),
  ])
  return `data:image/png;base64,${png.toString('base64')}`
}

function labeledRedPngDataUrl() {
  const width = 220
  const height = 96
  const bg = [220, 0, 0]
  const fg = [255, 255, 255]
  const font = {
    R: ['11110', '10001', '10001', '11110', '10100', '10010', '10001'],
    E: ['11111', '10000', '10000', '11110', '10000', '10000', '11111'],
    D: ['11110', '10001', '10001', '10001', '10001', '10001', '11110'],
  }
  const scale = 10
  const pixels = Array.from({ length: height }, () => Array.from({ length: width }, () => bg.slice()))
  let cursorX = 22
  const y0 = 13
  for (const ch of 'RED') {
    const glyph = font[ch]
    for (let gy = 0; gy < glyph.length; gy += 1) {
      for (let gx = 0; gx < glyph[gy].length; gx += 1) {
        if (glyph[gy][gx] !== '1') continue
        for (let yy = 0; yy < scale; yy += 1) {
          for (let xx = 0; xx < scale; xx += 1) {
            const py = y0 + gy * scale + yy
            const px = cursorX + gx * scale + xx
            if (py >= 0 && py < height && px >= 0 && px < width) pixels[py][px] = fg.slice()
          }
        }
      }
    }
    cursorX += 6 * scale
  }
  const ihdr = Buffer.alloc(13)
  ihdr.writeUInt32BE(width, 0)
  ihdr.writeUInt32BE(height, 4)
  ihdr[8] = 8
  ihdr[9] = 2
  ihdr[10] = 0
  ihdr[11] = 0
  ihdr[12] = 0
  const rows = []
  for (let y = 0; y < height; y += 1) {
    const row = Buffer.alloc(1 + width * 3)
    row[0] = 0
    for (let x = 0; x < width; x += 1) {
      const offset = 1 + x * 3
      row[offset] = pixels[y][x][0]
      row[offset + 1] = pixels[y][x][1]
      row[offset + 2] = pixels[y][x][2]
    }
    rows.push(row)
  }
  const png = Buffer.concat([
    Buffer.from([0x89, 0x50, 0x4e, 0x47, 0x0d, 0x0a, 0x1a, 0x0a]),
    pngChunk('IHDR', ihdr),
    pngChunk('IDAT', deflateSync(Buffer.concat(rows))),
    pngChunk('IEND', Buffer.alloc(0)),
  ])
  return `data:image/png;base64,${png.toString('base64')}`
}

function anthropicText(obj) {
  const chunks = []
  for (const part of obj?.content || []) {
    if (typeof part?.text === 'string') chunks.push(part.text)
  }
  return chunks.join('\n')
}

function ollamaText(obj) {
  return String(obj?.message?.content || obj?.response || '')
}

function makeSpeechAudio() {
  const aiff = path.join(tmpdir(), `vmlx-gemma-audio-${Date.now()}-${Math.random().toString(16).slice(2)}.aiff`)
  const wav = aiff.replace(/\.aiff$/, '.wav')
  try {
    const say = spawnSync('say', ['-o', aiff, 'audio present'], { timeout: 20_000 })
    if (say.status !== 0) throw new Error(String(say.stderr || say.error || 'say failed'))
    const af = spawnSync('afconvert', ['-f', 'WAVE', '-d', 'LEI16@16000', aiff, wav], { timeout: 20_000 })
    if (af.status !== 0) throw new Error(String(af.stderr || af.error || 'afconvert failed'))
    const buf = readFileSync(wav)
    return {
      base64: buf.toString('base64'),
      dataUrl: `data:audio/wav;base64,${buf.toString('base64')}`,
      format: 'wav',
      expectedPhrase: 'audio present',
      source: 'macos-say',
    }
  } catch (error) {
    const sampleRate = 16000
    const seconds = 0.8
    const samples = Math.floor(sampleRate * seconds)
    const data = Buffer.alloc(44 + samples * 2)
    data.write('RIFF', 0)
    data.writeUInt32LE(36 + samples * 2, 4)
    data.write('WAVEfmt ', 8)
    data.writeUInt32LE(16, 16)
    data.writeUInt16LE(1, 20)
    data.writeUInt16LE(1, 22)
    data.writeUInt32LE(sampleRate, 24)
    data.writeUInt32LE(sampleRate * 2, 28)
    data.writeUInt16LE(2, 32)
    data.writeUInt16LE(16, 34)
    data.write('data', 36)
    data.writeUInt32LE(samples * 2, 40)
    for (let i = 0; i < samples; i += 1) {
      const amp = Math.floor(0.25 * 32767 * Math.sin(2 * Math.PI * 440 * i / sampleRate))
      data.writeInt16LE(amp, 44 + i * 2)
    }
    return {
      base64: data.toString('base64'),
      dataUrl: `data:audio/wav;base64,${data.toString('base64')}`,
      format: 'wav',
      expectedPhrase: 'tone',
      source: `tone-fallback:${String(error?.message || error).slice(0, 200)}`,
    }
  } finally {
    rmSync(aiff, { force: true })
    rmSync(wav, { force: true })
  }
}

async function postJson(url, body, timeoutMs = 300_000) {
  const controller = new AbortController()
  const timer = setTimeout(() => controller.abort(), timeoutMs)
  try {
    const res = await fetch(url, {
      method: 'POST',
      headers: { 'content-type': 'application/json' },
      body: JSON.stringify(body),
      signal: controller.signal,
    })
    const text = await res.text()
    let json
    try {
      json = text ? JSON.parse(text) : null
    } catch {
      json = { raw: text }
    }
    return { ok: res.ok, status: res.status, json, raw: text.slice(0, 3000) }
  } finally {
    clearTimeout(timer)
  }
}

async function getJson(url, timeoutMs = 20_000) {
  const controller = new AbortController()
  const timer = setTimeout(() => controller.abort(), timeoutMs)
  try {
    const res = await fetch(url, { signal: controller.signal })
    const text = await res.text()
    try {
      return JSON.parse(text)
    } catch {
      return { status: res.status, raw: text.slice(0, 3000) }
    }
  } finally {
    clearTimeout(timer)
  }
}

async function streamSse(url, body, kind, timeoutMs = 300_000) {
  const controller = new AbortController()
  const timer = setTimeout(() => controller.abort(), timeoutMs)
  const events = []
  let content = ''
  let reasoning = ''
  let toolArgs = ''
  let responseToolDeltaArgs = ''
  let toolCallSignals = 0
  const completedToolCalls = []
  try {
    const res = await fetch(url, {
      method: 'POST',
      headers: { 'content-type': 'application/json' },
      body: JSON.stringify({ ...body, stream: true }),
      signal: controller.signal,
    })
    const text = await res.text()
    const chunks = text.split(/\n\n+/)
    for (const chunk of chunks) {
      const event = (chunk.match(/^event:\s*(.+)$/m) || [])[1] || ''
      for (const line of chunk.split(/\n/)) {
        if (!line.startsWith('data:')) continue
        const data = line.slice(5).trim()
        if (!data || data === '[DONE]') {
          events.push({ event: event || 'done', done: true })
          continue
        }
        let obj
        try { obj = JSON.parse(data) } catch { obj = { raw: data } }
        events.push({ event, obj })
        if (kind === 'chat') {
          const delta = obj?.choices?.[0]?.delta || {}
          content += delta.content || ''
          reasoning += delta.reasoning_content || delta.reasoning || ''
          for (const tc of delta.tool_calls || []) {
            toolCallSignals += 1
            toolArgs += tc?.function?.arguments || ''
          }
        } else if (kind === 'responses') {
          if (obj?.type === 'response.output_text.delta') content += obj.delta || ''
          if (obj?.type === 'response.reasoning.delta') reasoning += obj.delta || ''
          if (String(obj?.type || '').includes('function_call')) toolCallSignals += 1
          if (obj?.type === 'response.function_call_arguments.delta') {
            responseToolDeltaArgs += obj.delta || ''
            toolArgs += obj.delta || ''
          }
          const item = obj?.item || obj?.output_item || obj?.response?.output?.[0]
          if (item?.type === 'function_call') {
            toolCallSignals += 1
            completedToolCalls.push({
              name: item.name || '',
              arguments: item.arguments || '',
              call_id: item.call_id || '',
              status: item.status || '',
            })
            if (!responseToolDeltaArgs || !responseToolDeltaArgs.includes(item.arguments || '')) {
              toolArgs += item.arguments || ''
            }
          }
        }
      }
    }
    return {
      ok: res.ok,
      status: res.status,
      content,
      reasoning,
      toolArgs,
      completedToolCalls,
      hasToolCallSignal: toolCallSignals > 0,
      eventTypes: [...new Set(events.map((e) => e.event || e.obj?.type).filter(Boolean))],
      eventsTail: events.slice(-20),
      rawHead: text.slice(0, 3000),
      textScore: scoreText(content),
    }
  } finally {
    clearTimeout(timer)
  }
}

async function waitForCdp(debugPort) {
  const endpoint = `http://127.0.0.1:${debugPort}`
  const started = Date.now()
  while (Date.now() - started < 60_000) {
    try {
      const res = await fetch(`${endpoint}/json/version`)
      if (res.ok) return endpoint
    } catch {}
    await sleep(500)
  }
  throw new Error(`CDP endpoint did not open on ${debugPort}`)
}

async function getAppPage(browser) {
  for (const ctx of browser.contexts()) {
    for (const page of ctx.pages()) {
      const url = page.url()
      if (url.includes('index.html') || url.startsWith('file:') || url.includes('5173')) return page
    }
  }
  return browser.contexts()[0]?.pages()[0]
}

async function waitForWindowApi(page) {
  await page.evaluate(async () => {
    const started = Date.now()
    while (Date.now() - started < 45_000) {
      if (window.api?.sessions && window.api?.chat && window.api?.cache) return true
      await new Promise((resolve) => setTimeout(resolve, 100))
    }
    throw new Error('window.api not ready')
  })
}

function writeResult(result) {
  writeFileSync(outJson, JSON.stringify(result, null, 2))
}

async function waitForHealth(port, sessionId, page, result) {
  const endpoint = { host: '127.0.0.1', port }
  const started = Date.now()
  let last = null
  while (Date.now() - started < 900_000) {
    last = await page.evaluate(async ({ endpoint, sessionId }) => {
      const [health, logs] = await Promise.all([
        window.api.performance.health(endpoint).catch((error) => ({ error: String(error?.message || error) })),
        window.api.sessions.getLogs(sessionId).catch(() => []),
      ])
      return { health, logsTail: Array.isArray(logs) ? logs.slice(-100) : [] }
    }, { endpoint, sessionId })
    result.loadPolls.push({ elapsedMs: Date.now() - started, health: last.health, logsTail: last.logsTail.slice(-16) })
    writeResult(result)
    if (last.health?.status === 'healthy' && last.health?.model_loaded === true) return last.health
    await sleep(3000)
  }
  throw new Error(`Gemma session did not become healthy: ${JSON.stringify(last)}`)
}

async function sendUiTurn(page, chatId, prompt, endpoint, attachments = undefined) {
  const before = await page.evaluate(async ({ chatId }) => window.api.chat.getMessages(chatId), { chatId })
  const started = Date.now()
  let sendError = null
  try {
    await page.evaluate(
      async ({ chatId, prompt, endpoint, attachments }) => window.api.chat.sendMessage(chatId, prompt, endpoint, attachments),
      { chatId, prompt, endpoint, attachments },
    )
  } catch (error) {
    sendError = String(error?.message || error)
  }
  const after = await page.evaluate(async ({ chatId }) => window.api.chat.getMessages(chatId), { chatId })
  await sleep(2500)
  const afterQuiet = await page.evaluate(async ({ chatId }) => window.api.chat.getMessages(chatId), { chatId })
  const assistants = after.filter((m) => m.role === 'assistant')
  const last = assistants[assistants.length - 1] || null
  const beforeUsers = before.filter((m) => m.role === 'user').length
  const afterQuietUsers = afterQuiet.filter((m) => m.role === 'user').length
  const afterAssistants = after.filter((m) => m.role === 'assistant').length
  const afterQuietAssistants = afterQuiet.filter((m) => m.role === 'assistant').length
  return {
    prompt,
    sendError,
    elapsedMs: Date.now() - started,
    beforeCount: before.length,
    afterCount: after.length,
    afterQuietCount: afterQuiet.length,
    autonomousAssistantTurn: afterQuietUsers === beforeUsers + 1 && afterQuietAssistants > afterAssistants,
    content: last?.content || '',
    reasoningContent: last?.reasoningContent || '',
    reasoningChars: (last?.reasoningContent || '').length,
    hiddenOnly: !(last?.content || '').trim() && !!(last?.reasoningContent || '').trim(),
    metrics: parseMetrics(last),
    score: scoreText(last?.content || ''),
    toolCallsJson: last?.toolCallsJson || '',
    toolCallsOaiJson: last?.toolCallsOaiJson || '',
    messagesAfterQuiet: afterQuiet.map((m) => ({
      role: m.role,
      content: String(m.content || '').slice(0, 500),
      reasoningChars: String(m.reasoningContent || '').length,
    })),
  }
}

async function setChatOverrides(page, chatId, overrides) {
  await page.evaluate(async ({ chatId, overrides }) => {
    await window.api.chat.setOverrides(chatId, overrides)
  }, { chatId, overrides })
  return page.evaluate(async ({ chatId }) => window.api.chat.getOverrides(chatId), { chatId })
}

async function runMixedUiSession(page, modelPath, endpoint, proofDir, audioFixture) {
  const chat = await page.evaluate(async ({ modelPath, rowName }) => {
    return window.api.chat.create(`Gemma4 ${rowName} mixed reasoning-media-tool-cache stress`, modelPath.split('/').pop(), undefined, modelPath)
  }, { modelPath, rowName })
  const toolDir = path.join(proofDir, 'mixed-tool-workdir')
  mkdirSync(toolDir, { recursive: true })
  const redDataUrl = labeledRedPngDataUrl()
  const turns = []
  async function step(label, overrides, prompt, attachments = undefined) {
    const overridesAfterSet = await setChatOverrides(page, chat.id, overrides)
    const row = await sendUiTurn(page, chat.id, prompt, endpoint, attachments)
    row.label = label
    row.overridesAfterSet = overridesAfterSet
    turns.push(row)
    return row
  }

  await step('text_reasoning_off', {
    wireApi: 'responses',
    maxTokens: 140,
    temperature: 0.3,
    topP: 0.95,
    topK: 64,
    enableThinking: false,
    builtinToolsEnabled: false,
  }, 'Mixed turn 1. Reasoning OFF. Define Oracle EBS in one sentence and include exact label GEMMA_MIX_TEXT_OFF.')

  if (expectImage) {
    await step('image_reasoning_on', {
      wireApi: 'responses',
      maxTokens: 420,
      maxThinkingTokens: 320,
      temperature: 0,
      topP: 0.95,
      topK: 64,
      enableThinking: true,
      builtinToolsEnabled: false,
    }, 'Mixed turn 2. Reasoning ON. The attached image has a red background and a large white three-letter word. If the visible word is RED, reply visibly with exactly GEMMA_MIX_IMAGE_RED. Do not abbreviate the label.', [
      { dataUrl: redDataUrl, name: 'gemma-mixed-red-label.png', kind: 'image', type: 'image/png', size: redDataUrl.length },
    ])
  }

  await step('text_reasoning_auto', {
    wireApi: 'responses',
    maxTokens: 420,
    maxThinkingTokens: 320,
    temperature: 0.1,
    topP: 0.95,
    topK: 64,
    builtinToolsEnabled: false,
  }, 'Mixed turn 3. Reasoning AUTO. Begin the visible answer with exact label GEMMA_MIX_AUTO_TEXT, then explain profile options in one sentence.')

  if (expectAudio) {
    await step('audio_reasoning_auto', {
      wireApi: 'responses',
      maxTokens: 200,
      maxThinkingTokens: 180,
      temperature: 0.1,
      topP: 0.95,
      topK: 64,
      builtinToolsEnabled: false,
    }, 'Mixed turn 4. Reasoning AUTO with audio. Transcribe the attached audio. If it says audio present, include exact label GEMMA_MIX_AUDIO_PRESENT.', [
      { dataUrl: audioFixture.dataUrl, name: 'gemma-mixed-audio-present.wav', kind: 'audio', type: 'audio/wav', size: audioFixture.base64.length },
    ])
  }

  await step('tool_reasoning_on', {
    wireApi: 'responses',
    maxTokens: 800,
    maxThinkingTokens: 220,
    temperature: 0.2,
    topP: 0.95,
    topK: 64,
    enableThinking: true,
    builtinToolsEnabled: true,
    shellEnabled: true,
    fileToolsEnabled: true,
    searchToolsEnabled: false,
    webSearchEnabled: false,
    braveSearchEnabled: false,
    fetchUrlEnabled: false,
    gitEnabled: false,
    utilityToolsEnabled: true,
    workingDirectory: toolDir,
    maxToolIterations: 1,
    toolResultMaxChars: 8000,
  }, 'Mixed tool turn. Reasoning ON and tool required. Use run_command exactly once to create gemma_mixed_tool_on.txt containing exactly GEMMA_MIX_TOOL_ON. Do not read, cat, verify, or call any second tool. After the tool result, reply with GEMMA_MIX_TOOL_ON_DONE.')

  await step('tool_reasoning_auto', {
    wireApi: 'responses',
    maxTokens: 800,
    maxThinkingTokens: 220,
    temperature: 0.2,
    topP: 0.95,
    topK: 64,
    builtinToolsEnabled: true,
    shellEnabled: true,
    fileToolsEnabled: true,
    searchToolsEnabled: false,
    webSearchEnabled: false,
    braveSearchEnabled: false,
    fetchUrlEnabled: false,
    gitEnabled: false,
    utilityToolsEnabled: true,
    workingDirectory: toolDir,
    maxToolIterations: 1,
    toolResultMaxChars: 8000,
  }, 'Mixed tool turn. Reasoning AUTO and tool required. Use run_command exactly once to create gemma_mixed_tool_auto.txt containing exactly GEMMA_MIX_TOOL_AUTO. Do not read, cat, verify, or call any second tool. After the tool result, reply with GEMMA_MIX_TOOL_AUTO_DONE.')

  const finalLabels = ['GEMMA_MIX_TEXT_OFF', 'GEMMA_MIX_AUTO_TEXT', 'GEMMA_MIX_TOOL_ON', 'GEMMA_MIX_TOOL_AUTO']
  if (expectImage) finalLabels.splice(1, 0, 'GEMMA_MIX_IMAGE_RED')
  if (expectAudio) finalLabels.splice(expectImage ? 3 : 2, 0, 'GEMMA_MIX_AUDIO_PRESENT')
  await step('final_recall_cache', {
    wireApi: 'responses',
    maxTokens: 300,
    temperature: 0.2,
    topP: 0.95,
    topK: 64,
    enableThinking: false,
    // Keep the native-tool schema stable with the preceding tool turns so this
    // final recall is a real same-prefix cache-hit proof.
    builtinToolsEnabled: true,
    shellEnabled: true,
    fileToolsEnabled: true,
    searchToolsEnabled: false,
    webSearchEnabled: false,
    braveSearchEnabled: false,
    fetchUrlEnabled: false,
    gitEnabled: false,
    utilityToolsEnabled: true,
    workingDirectory: toolDir,
    maxToolIterations: 2,
    toolResultMaxChars: 8000,
  }, `Mixed final recall. Do not call any tools. Reply with these exact labels from this same chat: ${finalLabels.join(', ')}. No extra bullets.`)

  await step('post_mixed_cache_prime', {
    wireApi: 'responses',
    maxTokens: 120,
    temperature: 0.2,
    topP: 0.95,
    topK: 64,
    enableThinking: false,
    builtinToolsEnabled: false,
  }, 'Post mixed cache prime. We are still in the same chat after image, audio, and tools. Reply with exact label GEMMA_MIX_CACHE_PRIME.')

  await step('post_mixed_cache_hit', {
    wireApi: 'responses',
    maxTokens: 120,
    temperature: 0.2,
    topP: 0.95,
    topK: 64,
    enableThinking: false,
    builtinToolsEnabled: false,
  }, 'Post mixed cache hit. We are still in the same chat after image, audio, and tools. Reply with exact label GEMMA_MIX_CACHE_HIT.')

  let toolOnFile = null
  let toolAutoFile = null
  try { toolOnFile = readFileSync(path.join(toolDir, 'gemma_mixed_tool_on.txt'), 'utf8').trim() } catch {}
  try { toolAutoFile = readFileSync(path.join(toolDir, 'gemma_mixed_tool_auto.txt'), 'utf8').trim() } catch {}
  return {
    chatId: chat.id,
    toolDir,
    attachment: { name: 'gemma-mixed-red-label.png', color: 'red', text: 'RED', bytesBase64Chars: redDataUrl.length },
    audio: expectAudio ? { name: 'gemma-mixed-audio-present.wav', expectedPhrase: audioFixture.expectedPhrase, base64Chars: audioFixture.base64.length } : null,
    expectedLabels: finalLabels,
    toolFiles: { on: toolOnFile, auto: toolAutoFile },
    turns,
  }
}

function audioLooksUsed(text, expectedPhrase) {
  const lower = String(text || '').toLowerCase()
  if (expectedPhrase === 'audio present') {
    return lower.includes('audio') && lower.includes('present')
  }
  return lower.includes('tone') || lower.includes('sound') || lower.includes('audio')
}

function rejectsRedImageLabel(text) {
  return /\b(?:cannot|can't|do not|don't)\s+(?:see|view|access|inspect)|\bno\s+(?:attached\s+)?image\b|\b(?:not|does not|doesn't|do not|don't|will not|won't)\b[\s\S]{0,90}\b(?:red|label|use)\b|\b(?:gray|grey|pale\s+pink)\b/i.test(String(text || ''))
}

function deriveVerdict(result) {
  const failures = []
  const logs = (result.sessionLogsEnd || []).join('\n')
  if (!/family=gemma4|Model family: gemma4|gemma4/i.test(logs)) failures.push('Gemma4 autodetect/log evidence missing')
  const cfg = result.sessionConfigAfterStart || {}
  if (cfg.toolCallParser !== 'gemma4') failures.push(`session toolCallParser=${cfg.toolCallParser}`)
  if (cfg.reasoningParser !== 'gemma4') failures.push(`session reasoningParser=${cfg.reasoningParser}`)
  if (cfg.isMultimodal !== true) failures.push(`session isMultimodal=${cfg.isMultimodal}`)
  if (cfg.usePagedCache !== false) failures.push(`session usePagedCache=${cfg.usePagedCache}`)
  if (cfg.enableDiskCache !== true) failures.push(`session enableDiskCache=${cfg.enableDiskCache}`)
  if (cfg.kvCacheQuantization !== 'auto') failures.push(`session kvCacheQuantization=${cfg.kvCacheQuantization}`)
  const defaults = result.generationDefaults
  if (!defaults) failures.push('generation defaults proof missing')
  else {
    for (const key of ['temperature', 'top_p', 'top_k']) {
      if (defaults.parity?.[key] !== true) failures.push(`generation default parity failed for ${key}`)
    }
    if (defaults.session?.declared !== true) failures.push('session did not declare generation defaults')
    if (!defaults.uiVisibleProbe?.settingsTextVisible || !defaults.uiVisibleProbe?.temperatureValueVisible || !defaults.uiVisibleProbe?.topPValueVisible || !defaults.uiVisibleProbe?.topKValueVisible) {
      failures.push('UI did not visibly expose concrete generation defaults temperature/top-p/top-k')
    }
  }

  const modalities = new Set((result.capabilities?.modalities || []).map((m) => String(m).toLowerCase()))
  if (!modalities.has('text')) failures.push('capabilities missing text modality')
  if (expectImage && ![...modalities].some((m) => m === 'vision' || m === 'image')) failures.push('capabilities missing vision modality')
  if (expectAudio && !modalities.has('audio')) failures.push('capabilities missing audio modality')

  const turns = result.ui?.multiturn10?.turns || []
  if (turns.length !== 10) failures.push(`10-turn UI count mismatch: ${turns.length}`)
  if (turns.some((t) => t.sendError)) failures.push('UI text send error')
  if (turns.some((t) => t.score.empty || t.hiddenOnly)) failures.push('UI text empty or hidden-only')
  if (turns.some((t) => t.score.leakedReasoningTags)) failures.push('UI text raw reasoning/channel tag leak')
  if (turns.some((t) => t.score.loopSuspect)) failures.push('UI text loop suspect')
  if (turns.some((t) => t.autonomousAssistantTurn)) failures.push('UI autonomous assistant turn after completion')
  if (!turns.some((t) => Number(t.metrics?.cachedTokens || 0) > 0)) failures.push('UI text cachedTokens never exceeded 0')
  const final = labelText(turns[turns.length - 1]?.content || '')
  for (const label of ['GEMMA_EBS', 'GEMMA_AP', 'GEMMA_GL', 'GEMMA_PROFILE']) {
    if (!new RegExp(label, 'i').test(final)) failures.push(`final recall missing ${label}`)
  }
  const mixed = result.ui?.mixedSession
  if (!mixed) failures.push('mixed UI session missing')
  else {
    const mixedTurns = mixed.turns || []
    const minimumTurns = 7 + (expectImage ? 1 : 0) + (expectAudio ? 1 : 0)
    if (mixedTurns.length !== minimumTurns) failures.push(`mixed UI turn count mismatch: ${mixedTurns.length} expected ${minimumTurns}`)
    if (mixedTurns.some((t) => t.sendError)) failures.push('mixed UI send error')
    if (mixedTurns.some((t) => t.score.empty || t.hiddenOnly)) failures.push('mixed UI empty or hidden-only assistant turn')
    if (mixedTurns.some((t) => t.score.leakedReasoningTags)) failures.push('mixed UI raw reasoning/channel tag leak')
    if (mixedTurns.some((t) => t.score.loopSuspect)) failures.push('mixed UI loop suspect')
    if (mixedTurns.some((t) => t.autonomousAssistantTurn)) failures.push('mixed UI autonomous assistant turn after completion')
    if (mixed.toolFiles?.on !== 'GEMMA_MIX_TOOL_ON') failures.push('mixed UI reasoning-on tool file proof missing')
    if (mixed.toolFiles?.auto !== 'GEMMA_MIX_TOOL_AUTO') failures.push('mixed UI reasoning-auto tool file proof missing')
    const byLabel = Object.fromEntries(mixedTurns.map((t) => [t.label, t]))
    if ((byLabel.text_reasoning_off?.reasoningChars || 0) > 0) failures.push('mixed reasoning-off turn emitted reasoning')
    if (!/GEMMA_MIX_TEXT_OFF/i.test(byLabel.text_reasoning_off?.content || '')) failures.push('mixed text reasoning-off label missing')
    if (!/GEMMA_MIX_AUTO_TEXT/i.test(byLabel.text_reasoning_auto?.content || '')) failures.push('mixed text reasoning-auto label missing')
    if (expectImage && (byLabel.image_reasoning_on?.reasoningChars || 0) <= 0) failures.push('mixed reasoning-on image turn emitted no reasoning')
    if (!hasExactMarker(byLabel.tool_reasoning_on?.content || '', 'GEMMA_MIX_TOOL_ON_DONE')) failures.push('mixed reasoning-on tool final label missing')
    if (!hasExactMarker(byLabel.tool_reasoning_auto?.content || '', 'GEMMA_MIX_TOOL_AUTO_DONE')) failures.push('mixed reasoning-auto tool final label missing')
    const onToolCalls = toolCallingCount(byLabel.tool_reasoning_on, 'run_command')
    const autoToolCalls = toolCallingCount(byLabel.tool_reasoning_auto, 'run_command')
    if (onToolCalls !== 1) failures.push(`mixed reasoning-on expected exactly 1 run_command call, saw ${onToolCalls}`)
    if (autoToolCalls !== 1) failures.push(`mixed reasoning-auto expected exactly 1 run_command call, saw ${autoToolCalls}`)
    // Gemma4's native template tells required-tool turns to "Output only the
    // tool call"; reasoning-on is still exercised by sending enableThinking=true
    // with tools, but separated reasoning text is not mandatory for that row.
    const mixedImageContent = byLabel.image_reasoning_on?.content || ''
    if (
      expectImage
      && (
        !/GEMMA_MIX_IMAGE_RED/i.test(mixedImageContent)
        || rejectsRedImageLabel(mixedImageContent)
      )
    ) failures.push('mixed image turn did not identify red image')
    if (expectAudio && !audioLooksUsed(byLabel.audio_reasoning_auto?.content || '', result.audioFixture?.expectedPhrase)) failures.push('mixed audio turn did not acknowledge audio')
    const postMediaCacheHit = ['tool_reasoning_on', 'tool_reasoning_auto', 'final_recall_cache', 'post_mixed_cache_hit']
      .some((label) => Number(byLabel[label]?.metrics?.cachedTokens || 0) > 0)
    if (!postMediaCacheHit) failures.push('mixed post-media/tool cachedTokens never exceeded 0')
    const finalMixed = labelText(byLabel.final_recall_cache?.content || '')
    for (const label of mixed.expectedLabels || []) {
      if (!new RegExp(label, 'i').test(finalMixed)) failures.push(`mixed final recall missing ${label}`)
    }
    if (!/GEMMA_MIX_CACHE_PRIME/i.test(byLabel.post_mixed_cache_prime?.content || '')) failures.push('mixed post-media cache prime label missing')
    if (!/GEMMA_MIX_CACHE_HIT/i.test(byLabel.post_mixed_cache_hit?.content || '')) failures.push('mixed post-media cache hit label missing')
  }

  for (const row of result.ui?.reasoningModes || []) {
    if (row.score.empty || row.hiddenOnly) failures.push(`reasoning ${row.mode} empty or hidden-only`)
    if (row.score.leakedReasoningTags) failures.push(`reasoning ${row.mode} raw tag leak`)
  }
  const off = result.ui?.reasoningModes?.find((row) => row.mode === 'off')
  const on = result.ui?.reasoningModes?.find((row) => row.mode === 'on')
  if (off && off.reasoningChars > 0) failures.push(`reasoning off produced reasoningChars=${off.reasoningChars}`)
  if (on && on.reasoningChars <= 0) failures.push('reasoning on produced no reasoning content')

  if (expectAudio) {
    const audioRows = [
      ['uiAudio', result.ui?.audio],
      ['chatAudio', result.api?.chatAudio],
      ['responsesAudio', result.api?.responsesAudio],
    ]
    for (const [name, row] of audioRows) {
      if (!row) failures.push(`${name} missing`)
      else if (row.sendError || row.ok === false) failures.push(`${name} failed: ${row.sendError || row.status || row.raw || ''}`)
      else if (row.score?.empty || row.textScore?.empty || row.hiddenOnly) failures.push(`${name} empty or hidden-only`)
      else if (row.score?.leakedReasoningTags || row.textScore?.leakedReasoningTags) failures.push(`${name} raw tag leak`)
      else if (!audioLooksUsed(row.content || row.text || '', result.audioFixture?.expectedPhrase)) failures.push(`${name} did not acknowledge expected audio phrase/tone`)
    }
    const noAudio = result.ui?.postAudioText
    if (!noAudio || noAudio.sendError || noAudio.score.empty || noAudio.hiddenOnly) failures.push('post-audio text recovery failed')
  }

  for (const [name, row] of Object.entries(result.api || {})) {
    if (!row || typeof row !== 'object' || !Object.prototype.hasOwnProperty.call(row, 'ok')) continue
    if (!row.ok) failures.push(`API ${name} HTTP failed`)
    if (row.textScore?.empty) failures.push(`API ${name} visible text empty`)
    if (row.textScore?.loopSuspect) failures.push(`API ${name} loop suspect`)
    if (row.textScore?.leakedReasoningTags) failures.push(`API ${name} raw tag leak`)
  }
  for (const [name, expected] of Object.entries({
    chat: 'GEMMA_API_CHAT_OK',
    responsesText: 'GEMMA_RESP_OK',
    anthropicMessages: 'GEMMA_ANTHROPIC_OK',
    ollamaChat: 'GEMMA_OLLAMA_CHAT_OK',
    ollamaGenerate: 'GEMMA_OLLAMA_GENERATE_OK',
  })) {
    const row = result.api?.[name]
    if (row?.ok && !startsWithExactMarker(row.text || '', expected)) {
      failures.push(`API ${name} missing exact marker ${expected}`)
    }
  }
  if (!result.api?.chatTool?.hasToolCalls) failures.push('API Chat required tool did not return tool_calls')
  const apiToolArgs = JSON.stringify(result.api?.chatTool?.message?.tool_calls || [])
  if (result.api?.chatTool?.hasToolCalls && !hasExactMarker(apiToolArgs, 'GEMMA_TOOL_OK')) failures.push('API Chat tool arguments missing GEMMA_TOOL_OK')
  for (const [name, row] of Object.entries(result.streaming || {})) {
    if (!row?.ok) failures.push(`stream ${name} HTTP failed`)
    if (row?.textScore?.empty && !row?.hasToolCallSignal) failures.push(`stream ${name} reconstructed empty content/tool signal`)
    if (row?.textScore?.leakedReasoningTags) failures.push(`stream ${name} raw tag leak`)
  }
  for (const [name, expected] of Object.entries({
    chatText: 'GEMMA_STREAM_CHAT_OK',
    responsesText: 'GEMMA_STREAM_RESP_OK',
  })) {
    const row = result.streaming?.[name]
    if (row?.ok && !startsWithExactMarker(row.content || '', expected)) {
      failures.push(`stream ${name} missing exact marker ${expected}`)
    }
  }
  if (!result.streaming?.chatTool?.hasToolCallSignal) failures.push('streaming Chat tool call signal missing')
  if (!result.streaming?.responsesTool?.hasToolCallSignal) failures.push('streaming Responses tool call signal missing')
  if (result.streaming?.chatTool?.hasToolCallSignal && !hasExactMarker(result.streaming.chatTool.toolArgs || '', 'GEMMA_STREAM_CHAT_TOOL')) failures.push('streaming Chat tool arguments missing GEMMA_STREAM_CHAT_TOOL')
  if (result.streaming?.responsesTool?.hasToolCallSignal && !hasExactMarker(result.streaming.responsesTool.toolArgs || '', 'GEMMA_STREAM_RESP_TOOL')) failures.push('streaming Responses tool arguments missing GEMMA_STREAM_RESP_TOOL')
  const responseStreamToolCalls = result.streaming?.responsesTool?.completedToolCalls || []
  const exactResponseStreamToolCalls = responseStreamToolCalls.filter((call) => (
    call?.name === 'record_gemma_stream_response_label'
    && hasExactMarker(call?.arguments || '', 'GEMMA_STREAM_RESP_TOOL')
  ))
  if (result.streaming?.responsesTool?.hasToolCallSignal && exactResponseStreamToolCalls.length !== 1) {
    failures.push(`streaming Responses expected exactly 1 completed record_gemma_stream_response_label call with GEMMA_STREAM_RESP_TOOL, saw ${exactResponseStreamToolCalls.length}`)
  }
  if (expectImage && (!/GEMMA_STREAM_IMAGE_RED/i.test(result.streaming?.responsesImage?.content || '') || rejectsRedImageLabel(result.streaming?.responsesImage?.content || ''))) failures.push('streaming Responses image did not identify red image')
  return { status: failures.length ? 'fail' : 'pass', failures }
}

async function main() {
  if (!existsSync(appExe)) throw new Error(`Installed app executable missing: ${appExe}`)
  if (!existsSync(modelPath)) throw new Error(`Gemma model path missing: ${modelPath}`)
  mkdirSync(proofDir, { recursive: true })
  const userDataDir = mkdtempSync(path.join(tmpdir(), 'vmlx-gemma-media-userdata-'))
  const appLog = []
  const debugPort = await freePort()
  const sessionPort = Number(process.env.VMLX_GEMMA_PORT || await freePort())
  const audioFixture = makeSpeechAudio()
  const result = {
    generatedAt: startedAt.toISOString(),
    status: 'running',
    rowName,
    proofDir,
    outJson,
    shotPath,
    repoDir,
    panelDir,
    appPath,
    modelPath,
    debugPort,
    sessionPort,
    expectAudio,
    expectImage,
    audioFixture: {
      source: audioFixture.source,
      format: audioFixture.format,
      expectedPhrase: audioFixture.expectedPhrase,
      base64Chars: audioFixture.base64.length,
    },
    loadPolls: [],
    sourceTrace: {
      appGemmaRegistry: 'panel/src/main/model-config-registry.ts:167-168, 803-811',
      appCacheLaunchPolicy: 'panel/src/main/sessions.ts:2859-2966',
      serverModalities: 'vmlx_engine/server.py:2256-2314',
      mllmAudioPath: 'vmlx_engine/mllm_batch_generator.py:4202-4404',
      gemmaUnifiedRuntime: 'vmlx_engine/models/gemma4_unified_register.py',
    },
  }
  writeResult(result)

  const app = spawn(appExe, [`--user-data-dir=${userDataDir}`, `--remote-debugging-port=${debugPort}`], {
    cwd: tmpdir(),
    env: { ...process.env, VMLX_SKIP_UPDATE_CHECK: '1' },
    detached: true,
    stdio: ['ignore', 'pipe', 'pipe'],
  })
  app.stdout.on('data', (d) => appLog.push(...d.toString().split(/\r?\n/).filter(Boolean)))
  app.stderr.on('data', (d) => appLog.push(...d.toString().split(/\r?\n/).filter(Boolean)))
  result.appPid = app.pid
  result.appCommand = [appExe, `--user-data-dir=${userDataDir}`, `--remote-debugging-port=${debugPort}`]
  writeResult(result)

  let browser
  let page
  try {
    const cdp = await waitForCdp(debugPort)
    browser = await chromium.connectOverCDP(cdp)
    page = await getAppPage(browser)
    if (!page) throw new Error('No Electron renderer page found')
    page.setDefaultTimeout(900_000)
    await page.waitForLoadState('domcontentloaded').catch(() => {})
    await waitForWindowApi(page)

    const sessionResult = await page.evaluate(async ({ modelPath, port }) => {
      await window.api.chat.clearAllLocks().catch(() => null)
      const config = {
        host: '127.0.0.1',
        port,
        continuousBatching: true,
        maxNumSeqs: 1,
        prefillBatchSize: 512,
        prefillStepSize: 2048,
        completionBatchSize: 512,
        cacheMemoryPercent: 15,
        timeout: 300,
        maxTokens: 512,
      }
      const created = await window.api.sessions.create(modelPath, config)
      if (!created?.success || !created?.session?.id) {
        throw new Error(`sessions.create failed: ${created?.error || JSON.stringify(created)}`)
      }
      const started = await window.api.sessions.start(created.session.id)
      if (!started?.success) {
        throw new Error(`sessions.start failed: ${started?.error || JSON.stringify(started)}`)
      }
      const session = await window.api.sessions.get(created.session.id)
      return { created, started, session }
    }, { modelPath, port: sessionPort })
    result.sessionCreateResult = sessionResult.created
    result.sessionStartResult = sessionResult.started
    const session = sessionResult.session
    if (!session?.id) throw new Error(`sessions.get returned no session: ${JSON.stringify(sessionResult)}`)
    result.session = session
    result.sessionConfigAfterCreate = JSON.parse(session.config || '{}')
    writeResult(result)

    const endpoint = { host: '127.0.0.1', port: sessionPort }
    result.healthReady = await waitForHealth(sessionPort, session.id, page, result)
    result.sessionAfterStart = await page.evaluate(async ({ id }) => window.api.sessions.get(id), { id: session.id })
    result.sessionConfigAfterStart = JSON.parse(result.sessionAfterStart.config || '{}')
    result.sessionConfigAfterStart.id = session.id
    result.generationDefaults = await captureGenerationDefaults(page, modelPath, result.sessionConfigAfterStart)
    result.sessionLogsStart = await page.evaluate(async ({ id }) => window.api.sessions.getLogs(id), { id: session.id })
    result.cacheStart = await page.evaluate(async ({ endpoint, id }) => window.api.cache.stats(endpoint, id), { endpoint, id: session.id }).catch((error) => ({ error: String(error?.message || error) }))
    result.models = await getJson(`http://127.0.0.1:${sessionPort}/v1/models`).catch((error) => ({ error: String(error?.message || error) }))
    const servedModel = result.models?.data?.[0]?.id || result.sessionAfterStart?.modelName || path.basename(modelPath)
    result.servedModel = servedModel
    result.capabilities = await getJson(`http://127.0.0.1:${sessionPort}/v1/models/${encodeURIComponent(servedModel)}/capabilities`).catch((error) => ({ error: String(error?.message || error) }))
    writeResult(result)

    result.ui = { multiturn10: { turns: [] }, reasoningModes: [], audio: null, postAudioText: null, mixedSession: null }
    const chat = await page.evaluate(async ({ modelPath, rowName }) => {
      return window.api.chat.create(`Gemma4 ${rowName} 10-turn stress`, modelPath.split('/').pop(), undefined, modelPath)
    }, { modelPath, rowName })
    await page.evaluate(async ({ chatId }) => window.api.chat.setOverrides(chatId, {
      wireApi: 'responses',
      maxTokens: 220,
      temperature: 0.7,
      topP: 0.95,
      enableThinking: false,
    }), { chatId: chat.id })
    const prompts = [
      'Set anchor GEMMA_EBS=Oracle EBS. Define it in one short sentence.',
      'Set anchor GEMMA_AP=payables. Explain AP in one concise bullet.',
      'Set anchor GEMMA_GL=ledger. Explain GL in one concise bullet.',
      'Set anchor GEMMA_RESP=responsibilities. Explain responsibilities in one concise bullet.',
      'Set anchor GEMMA_CP=concurrent programs. Explain concurrent programs in one concise bullet.',
      'Set anchor GEMMA_PROFILE=profile options. Explain profile options in one concise bullet.',
      'Set anchor GEMMA_PATCH=patching. Explain patching in one concise bullet.',
      'Compare GEMMA_AP and GEMMA_GL in one short paragraph.',
      'What are GEMMA_RESP, GEMMA_CP, and GEMMA_PROFILE? Use those labels.',
      'Recall GEMMA_EBS, GEMMA_AP, GEMMA_GL, and GEMMA_PROFILE. Use each label once.',
    ]
    for (let i = 0; i < prompts.length; i += 1) {
      const turn = await sendUiTurn(page, chat.id, prompts[i], endpoint)
      turn.turn = i + 1
      result.ui.multiturn10.turns.push(turn)
      writeResult(result)
    }

    for (const mode of [
      { mode: 'off', overrides: { wireApi: 'responses', maxTokens: 220, temperature: 0.2, topP: 0.95, enableThinking: false }, prompt: 'Reasoning OFF test: reply visibly with exactly GEMMA_OFF_VISIBLE_OK.' },
      { mode: 'on', overrides: { wireApi: 'responses', maxTokens: 700, maxThinkingTokens: 220, temperature: 0.3, topP: 0.95, enableThinking: true }, prompt: 'Reasoning ON test: think briefly, then visible final answer must contain GEMMA_ON_VISIBLE_OK.' },
      { mode: 'auto', overrides: { wireApi: 'responses', maxTokens: 700, maxThinkingTokens: 220, temperature: 0.3, topP: 0.95 }, prompt: 'Reasoning AUTO test: if useful think briefly, then visible final answer must contain GEMMA_AUTO_VISIBLE_OK.' },
    ]) {
      const modeChat = await page.evaluate(async ({ modelPath, mode }) => {
        return window.api.chat.create(`Gemma4 reasoning ${mode}`, modelPath.split('/').pop(), undefined, modelPath)
      }, { modelPath, mode: mode.mode })
      await page.evaluate(async ({ chatId, overrides }) => window.api.chat.setOverrides(chatId, overrides), { chatId: modeChat.id, overrides: mode.overrides })
      const row = await sendUiTurn(page, modeChat.id, mode.prompt, endpoint)
      row.mode = mode.mode
      row.chatId = modeChat.id
      result.ui.reasoningModes.push(row)
      writeResult(result)
    }

    if (expectAudio) {
      const audioChat = await page.evaluate(async ({ modelPath }) => {
        return window.api.chat.create('Gemma4 UI audio stress', modelPath.split('/').pop(), undefined, modelPath)
      }, { modelPath })
      await page.evaluate(async ({ chatId }) => window.api.chat.setOverrides(chatId, {
        wireApi: 'responses',
        maxTokens: 180,
        temperature: 0.1,
        topP: 0.95,
        enableThinking: false,
      }), { chatId: audioChat.id })
      const uiAudio = await sendUiTurn(
        page,
        audioChat.id,
        'Transcribe the attached audio. If it says audio present, include the exact words audio present in the visible answer.',
        endpoint,
        [{ dataUrl: audioFixture.dataUrl, name: 'audio-present.wav', kind: 'audio', type: 'audio/wav', size: audioFixture.base64.length }],
      )
      result.ui.audio = { chatId: audioChat.id, ...uiAudio }
      writeResult(result)
      const postAudio = await sendUiTurn(
        page,
        audioChat.id,
        'This follow-up has no audio attachment. Reply with GEMMA_POST_AUDIO_TEXT_OK and do not claim a new audio file is attached.',
        endpoint,
      )
      result.ui.postAudioText = { chatId: audioChat.id, ...postAudio }
      writeResult(result)
    }

    result.ui.mixedSession = await runMixedUiSession(page, modelPath, endpoint, proofDir, audioFixture)
    writeResult(result)

    result.api = {}
    const chatResp = await postJson(`http://127.0.0.1:${sessionPort}/v1/chat/completions`, {
      model: servedModel,
      messages: [{ role: 'user', content: 'API Chat text check: start the visible answer with the exact marker GEMMA_API_CHAT_OK, then add one short fact about Oracle EBS.' }],
      max_tokens: 140,
      temperature: 0,
      top_p: 0.95,
      enable_thinking: false,
    })
    const chatText = chatResp.json?.choices?.[0]?.message?.content || ''
    result.api.chat = { ...chatResp, text: chatText, textScore: scoreText(chatText), cachedTokens: cacheTokensFromUsage(chatResp.json) }
    writeResult(result)

    const toolResp = await postJson(`http://127.0.0.1:${sessionPort}/v1/chat/completions`, {
      model: servedModel,
      messages: [{ role: 'user', content: 'Call the provided function with label GEMMA_TOOL_OK.' }],
      tools: [{
        type: 'function',
        function: {
          name: 'record_gemma_label',
          description: 'Record a label for the Gemma API stress test.',
          parameters: {
            type: 'object',
            properties: { label: { type: 'string' } },
            required: ['label'],
          },
        },
      }],
      tool_choice: { type: 'function', function: { name: 'record_gemma_label' } },
      max_tokens: 260,
      temperature: 0.2,
      top_p: 0.95,
      enable_thinking: false,
    })
    const toolMessage = toolResp.json?.choices?.[0]?.message || {}
    result.api.chatTool = {
      ...toolResp,
      message: toolMessage,
      text: toolMessage.content || '',
      textScore: scoreText(toolMessage.content || JSON.stringify(toolMessage.tool_calls || [])),
      hasToolCalls: Array.isArray(toolMessage.tool_calls) && toolMessage.tool_calls.length > 0,
    }
    writeResult(result)

    const responsesResp = await postJson(`http://127.0.0.1:${sessionPort}/v1/responses`, {
      model: servedModel,
      input: 'Responses text check: start the visible answer with the exact marker GEMMA_RESP_OK, then add one short fact about Oracle EBS.',
      max_output_tokens: 140,
      temperature: 0,
      top_p: 0.95,
      enable_thinking: false,
    })
    const responsesText = responseTextFromResponsesObject(responsesResp.json)
    result.api.responsesText = { ...responsesResp, text: responsesText, textScore: scoreText(responsesText), cachedTokens: cacheTokensFromUsage(responsesResp.json) }
    writeResult(result)

    if (expectAudio) {
      const chatAudioResp = await postJson(`http://127.0.0.1:${sessionPort}/v1/chat/completions`, {
        model: servedModel,
        messages: [{
          role: 'user',
          content: [
            { type: 'text', text: 'Transcribe the attached audio. If it says audio present, include the exact words audio present.' },
            { type: 'input_audio', input_audio: { data: audioFixture.base64, format: audioFixture.format } },
          ],
        }],
        max_tokens: 160,
        temperature: 0.1,
        top_p: 0.95,
        enable_thinking: false,
      })
      const chatAudioText = chatAudioResp.json?.choices?.[0]?.message?.content || ''
      result.api.chatAudio = { ...chatAudioResp, text: chatAudioText, textScore: scoreText(chatAudioText), cachedTokens: cacheTokensFromUsage(chatAudioResp.json) }
      writeResult(result)

      const responsesAudioResp = await postJson(`http://127.0.0.1:${sessionPort}/v1/responses`, {
        model: servedModel,
        input: [{
          type: 'message',
          role: 'user',
          content: [
            { type: 'input_text', text: 'Transcribe the attached audio. If it says audio present, include the exact words audio present.' },
            { type: 'input_audio', input_audio: { data: audioFixture.base64, format: audioFixture.format } },
          ],
        }],
        max_output_tokens: 160,
        temperature: 0.1,
        top_p: 0.95,
        enable_thinking: false,
      })
      const responsesAudioText = responseTextFromResponsesObject(responsesAudioResp.json)
      result.api.responsesAudio = { ...responsesAudioResp, text: responsesAudioText, textScore: scoreText(responsesAudioText), cachedTokens: cacheTokensFromUsage(responsesAudioResp.json) }
      writeResult(result)
    }

    const anthropicResp = await postJson(`http://127.0.0.1:${sessionPort}/v1/messages`, {
      model: servedModel,
      max_tokens: 120,
      messages: [{ role: 'user', content: 'Anthropic route check: start the visible answer with the exact marker GEMMA_ANTHROPIC_OK, then add one short fact about Oracle EBS.' }],
      thinking: { type: 'disabled' },
    })
    const anthropicVisible = anthropicText(anthropicResp.json)
    result.api.anthropicMessages = { ...anthropicResp, text: anthropicVisible, textScore: scoreText(anthropicVisible), cachedTokens: cacheTokensFromUsage(anthropicResp.json) }
    writeResult(result)

    const ollamaChatResp = await postJson(`http://127.0.0.1:${sessionPort}/api/chat`, {
      model: servedModel,
      messages: [{ role: 'user', content: 'Ollama chat check: start the visible answer with the exact marker GEMMA_OLLAMA_CHAT_OK, then add one short fact about Oracle EBS.' }],
      stream: false,
      think: false,
      options: { temperature: 0, top_p: 0.95, top_k: 64, num_predict: 120 },
    })
    const ollamaChatVisible = ollamaText(ollamaChatResp.json)
    result.api.ollamaChat = { ...ollamaChatResp, text: ollamaChatVisible, textScore: scoreText(ollamaChatVisible), cachedTokens: cacheTokensFromUsage(ollamaChatResp.json) }
    writeResult(result)

    const ollamaGenerateResp = await postJson(`http://127.0.0.1:${sessionPort}/api/generate`, {
      model: servedModel,
      prompt: 'Ollama generate check: start the visible answer with the exact marker GEMMA_OLLAMA_GENERATE_OK, then add one short fact about Oracle EBS.',
      stream: false,
      think: false,
      options: { temperature: 0, top_p: 0.95, top_k: 64, num_predict: 120 },
    })
    const ollamaGenerateVisible = ollamaText(ollamaGenerateResp.json)
    result.api.ollamaGenerate = { ...ollamaGenerateResp, text: ollamaGenerateVisible, textScore: scoreText(ollamaGenerateVisible), cachedTokens: cacheTokensFromUsage(ollamaGenerateResp.json) }
    writeResult(result)

    result.streaming = {}
    result.streaming.chatText = await streamSse(`http://127.0.0.1:${sessionPort}/v1/chat/completions`, {
      model: servedModel,
      messages: [{ role: 'user', content: 'Streaming Chat check: start the visible answer with the exact marker GEMMA_STREAM_CHAT_OK, then add one short fact about Oracle EBS.' }],
      max_tokens: 120,
      temperature: 0,
      top_p: 0.95,
      enable_thinking: false,
    }, 'chat')
    writeResult(result)
    result.streaming.responsesText = await streamSse(`http://127.0.0.1:${sessionPort}/v1/responses`, {
      model: servedModel,
      input: 'Streaming Responses check: start the visible answer with the exact marker GEMMA_STREAM_RESP_OK, then add one short fact about Oracle EBS.',
      max_output_tokens: 120,
      temperature: 0,
      top_p: 0.95,
      enable_thinking: false,
    }, 'responses')
    writeResult(result)
    if (expectImage) {
      const redDataUrl = labeledRedPngDataUrl()
      result.streaming.responsesImage = await streamSse(`http://127.0.0.1:${sessionPort}/v1/responses`, {
        model: servedModel,
        input: [{
          type: 'message',
          role: 'user',
          content: [
            { type: 'input_text', text: 'Streaming Responses image check: the attached image has a red background and a large white three-letter word. Reply visibly with exact label GEMMA_STREAM_IMAGE_RED only if the visible word is RED.' },
            { type: 'input_image', image_url: redDataUrl },
          ],
        }],
        max_output_tokens: 160,
        temperature: 0.1,
        top_p: 0.95,
        enable_thinking: false,
      }, 'responses')
      writeResult(result)
    }
    result.streaming.chatTool = await streamSse(`http://127.0.0.1:${sessionPort}/v1/chat/completions`, {
      model: servedModel,
      messages: [{ role: 'user', content: 'Streaming Chat tool check: call the function with label GEMMA_STREAM_CHAT_TOOL.' }],
      tools: [{
        type: 'function',
        function: {
          name: 'record_gemma_stream_label',
          description: 'Record a Gemma streaming label.',
          parameters: {
            type: 'object',
            properties: { label: { type: 'string' } },
            required: ['label'],
          },
        },
      }],
      tool_choice: { type: 'function', function: { name: 'record_gemma_stream_label' } },
      max_tokens: 220,
      temperature: 0.2,
      top_p: 0.95,
      enable_thinking: false,
    }, 'chat')
    writeResult(result)
    result.streaming.responsesTool = await streamSse(`http://127.0.0.1:${sessionPort}/v1/responses`, {
      model: servedModel,
      input: 'Streaming Responses tool check: call the function with label GEMMA_STREAM_RESP_TOOL.',
      tools: [{
        type: 'function',
        name: 'record_gemma_stream_response_label',
        description: 'Record a Gemma Responses streaming label.',
        parameters: {
          type: 'object',
          properties: { label: { type: 'string' } },
          required: ['label'],
        },
      }],
      tool_choice: { type: 'function', name: 'record_gemma_stream_response_label' },
      max_output_tokens: 220,
      temperature: 0.2,
      top_p: 0.95,
      enable_thinking: false,
    }, 'responses')
    writeResult(result)

    result.cacheEnd = await page.evaluate(async ({ endpoint, id }) => window.api.cache.stats(endpoint, id), { endpoint, id: session.id }).catch((error) => ({ error: String(error?.message || error) }))
    result.healthEnd = await getJson(`http://127.0.0.1:${sessionPort}/health`).catch((error) => ({ error: String(error?.message || error) }))
    result.capabilitiesEnd = await getJson(`http://127.0.0.1:${sessionPort}/v1/models/${encodeURIComponent(servedModel)}/capabilities`).catch((error) => ({ error: String(error?.message || error) }))
    result.sessionLogsEnd = await page.evaluate(async ({ id }) => window.api.sessions.getLogs(id), { id: session.id })
    result.appLogTail = appLog.slice(-200)
    writeResult(result)

    await page.evaluate(async ({ chatId, sessionId }) => {
      await window.api.settings?.set?.('appMode', 'chat').catch(() => null)
      await window.api.settings?.set?.('lastActiveChatId', chatId).catch(() => null)
      await window.api.settings?.set?.('lastActiveSessionId', sessionId).catch(() => null)
      location.reload()
      return true
    }, { chatId: chat.id, sessionId: session.id }).catch(() => null)
    await sleep(2500)
    await page.screenshot({ path: shotPath, fullPage: false }).catch(() => null)
    result.screenshot = shotPath
    result.finishedAt = new Date().toISOString()
    Object.assign(result, deriveVerdict(result))
    writeResult(result)

    await page.evaluate(async ({ id }) => window.api.sessions.stop(id).catch(() => null), { id: session.id }).catch(() => null)
    console.log(JSON.stringify({
      status: result.status,
      failures: result.failures,
      rowName,
      proofDir,
      outJson,
      shotPath,
      sessionPort,
      capabilities: result.capabilities?.modalities,
      cachedTurns: result.ui.multiturn10.turns.map((t) => ({ turn: t.turn, cachedTokens: t.metrics?.cachedTokens, tps: t.metrics?.tokensPerSecond })),
      reasoning: result.ui.reasoningModes.map((r) => ({ mode: r.mode, contentChars: r.content.length, reasoningChars: r.reasoningChars, hiddenOnly: r.hiddenOnly })),
      uiAudio: result.ui.audio?.score?.preview,
      postAudio: result.ui.postAudioText?.score?.preview,
      mixed: result.ui.mixedSession?.turns?.map((t) => ({ label: t.label, cachedTokens: t.metrics?.cachedTokens, content: t.score.preview, reasoningChars: t.reasoningChars })),
      generationDefaults: result.generationDefaults,
      apiAudio: {
        chat: result.api.chatAudio?.textScore?.preview,
        responses: result.api.responsesAudio?.textScore?.preview,
      },
      streaming: {
        chat: result.streaming.chatText?.textScore?.preview,
        responses: result.streaming.responsesText?.textScore?.preview,
        responsesImage: result.streaming.responsesImage?.textScore?.preview,
        chatTool: result.streaming.chatTool?.hasToolCallSignal,
        responsesTool: result.streaming.responsesTool?.hasToolCallSignal,
      },
    }, null, 2))
  } catch (error) {
    result.status = 'fail'
    result.failures = [String(error?.stack || error?.message || error)]
    result.appLogTail = appLog.slice(-200)
    try {
      if (page) result.bodyText = await page.evaluate(() => document.body.innerText.slice(0, 6000))
      if (page) await page.screenshot({ path: shotPath, fullPage: false }).catch(() => null)
    } catch {}
    writeResult(result)
    console.error(JSON.stringify({ status: 'fail', failures: result.failures, proofDir, outJson, shotPath }, null, 2))
    process.exitCode = 1
  } finally {
    await browser?.close().catch(() => null)
    if (app.pid) {
      try { process.kill(-app.pid, 'SIGTERM') } catch {}
      await sleep(1500)
      try { process.kill(-app.pid, 'SIGKILL') } catch {}
    }
    if (!process.env.VMLX_KEEP_GEMMA_USER_DATA) {
      rmSync(userDataDir, { recursive: true, force: true })
    }
  }
}

main().catch((error) => {
  console.error(error?.stack || error?.message || String(error))
  process.exit(1)
})
