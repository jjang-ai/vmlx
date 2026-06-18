#!/usr/bin/env node
import { spawn } from 'node:child_process'
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

const modelPath = process.env.VMLX_MM3_MODEL_PATH
  || '/Users/example/.mlxstudio/models/JANGQ-AI/MiniMax-M3-REAP40-d3-JANG_2L'
const appPath = process.env.VMLX_APP_PATH || '/Applications/vMLX.app'
const appExe = path.join(appPath, 'Contents', 'MacOS', 'vMLX')
const startedAt = new Date()
const stamp = startedAt.toISOString().replace(/[:.]/g, '-')
const proofDir = path.resolve(process.env.VMLX_MM3_PROOF_DIR || path.join(repoDir, 'build', `live-mm3-stress-${stamp}`))
const outJson = path.join(proofDir, 'mm3-stress-proof.json')
const shotPath = path.join(proofDir, 'mm3-stress-final.png')

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
    leakedThinkTags: /<mm:think|<think>|<\/think>|\[THINK\]|\[\/THINK\]/i.test(value),
    maxLineRun,
    adjacentPhraseRepeat: phrase,
    loopSuspect: maxLineRun >= 4 || phrase.count >= 6,
    preview: value.trim().slice(0, 600),
  }
}

function labelText(text) {
  return String(text || '')
    .replace(/\\_/g, '_')
    .replace(/&lowbar;/gi, '_')
}

function hasExactMarker(text, expected) {
  return labelText(text).includes(expected)
}

function equalsText(text, expected) {
  return labelText(text).trim().toLowerCase() === String(expected).toLowerCase()
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

function defaultsMatch(modelValue, sessionValue, key) {
  if (modelValue == null && key === 'top_k') return sessionValue === 0 || sessionValue == null
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
    temperature: defaultsMatch(model.temperature, session.temperature, 'temperature'),
    top_p: defaultsMatch(model.top_p, session.top_p, 'top_p'),
    top_k: defaultsMatch(model.top_k, session.top_k, 'top_k'),
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

function rejectsRedImageLabel(text) {
  return /\b(?:cannot|can't|do not|don't)\s+(?:see|view|access|inspect)|\bno\s+(?:attached\s+)?image\b|\b(?:not|does not|doesn't|do not|don't|will not|won't)\b[\s\S]{0,90}\b(?:red|label|use)\b|\b(?:gray|grey|pale\s+pink)\b/i.test(String(text || ''))
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

async function postJson(url, body, timeoutMs = 240_000) {
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
    return { ok: res.ok, status: res.status, json, raw: text.slice(0, 2000) }
  } finally {
    clearTimeout(timer)
  }
}

async function getJson(url, timeoutMs = 10_000) {
  const controller = new AbortController()
  const timer = setTimeout(() => controller.abort(), timeoutMs)
  try {
    const res = await fetch(url, { signal: controller.signal })
    return await res.json()
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
      eventsTail: events.slice(-24),
      rawHead: text.slice(0, 3000),
      textScore: scoreText(content || toolArgs),
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
      return { health, logsTail: Array.isArray(logs) ? logs.slice(-80) : [] }
    }, { endpoint, sessionId })
    result.loadPolls.push({ elapsedMs: Date.now() - started, health: last.health, logsTail: last.logsTail.slice(-12) })
    writeResult(result)
    if (last.health?.status === 'healthy' && last.health?.model_loaded === true) return last.health
    await sleep(3000)
  }
  throw new Error(`MM3 session did not become healthy: ${JSON.stringify(last)}`)
}

function writeResult(result) {
  writeFileSync(outJson, JSON.stringify(result, null, 2))
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
    toolResultsOaiJson: last?.toolResultsOaiJson || '',
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

async function runMixedUiSession(page, modelPath, endpoint, proofDir) {
  const chat = await page.evaluate(async ({ modelPath }) => {
    return window.api.chat.create('MM3 mixed reasoning-media-tool-cache stress', modelPath.split('/').pop(), undefined, modelPath)
  }, { modelPath })
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
    topP: 0.9,
    enableThinking: false,
    builtinToolsEnabled: false,
  }, 'Mixed turn 1. Reasoning OFF. Define Oracle EBS in one sentence and include exact label M3_MIX_TEXT_OFF.')

  await step('image_reasoning_on', {
    wireApi: 'responses',
    maxTokens: 260,
    maxThinkingTokens: 180,
    temperature: 0.2,
    topP: 0.9,
    enableThinking: true,
    builtinToolsEnabled: false,
  }, 'Mixed turn 2. Reasoning ON. The attached image has a red background and a large white three-letter word. Reply visibly with exact label MM3_MIX_IMAGE_RED only if the visible word is RED.', [
    { dataUrl: redDataUrl, name: 'mixed-red-label.png', kind: 'image', type: 'image/png', size: redDataUrl.length },
  ])

  await step('text_reasoning_auto', {
    wireApi: 'responses',
    maxTokens: 160,
    maxThinkingTokens: 180,
    temperature: 0.3,
    topP: 0.9,
    builtinToolsEnabled: false,
  }, 'Mixed turn 3. Reasoning AUTO. Set anchor M3_MIX_AUTO_TEXT=profile options and explain profile options in one sentence using that exact label.')

  await step('tool_reasoning_on', {
    wireApi: 'responses',
    maxTokens: 800,
    maxThinkingTokens: 220,
    temperature: 0.2,
    topP: 0.9,
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
  }, 'Mixed turn 4. Reasoning ON and tool required. Use run_command exactly once to create m3_mixed_tool_on.txt containing exactly M3_MIX_TOOL_ON. Do not read, cat, verify, or call any second tool. After the tool result, reply with M3_MIX_TOOL_ON_DONE.')

  await step('tool_reasoning_auto', {
    wireApi: 'responses',
    maxTokens: 800,
    maxThinkingTokens: 220,
    temperature: 0.2,
    topP: 0.9,
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
  }, 'Mixed turn 5. Reasoning AUTO and tool required. Use run_command exactly once to create m3_mixed_tool_auto.txt containing exactly M3_MIX_TOOL_AUTO. Do not read, cat, verify, or call any second tool. After the tool result, reply with M3_MIX_TOOL_AUTO_DONE.')

  await step('final_recall_cache', {
    wireApi: 'responses',
    maxTokens: 260,
    temperature: 0.2,
    topP: 0.9,
    enableThinking: false,
    // Keep the same native-tool schema visible as the preceding tool turns.
    // Turning tools off changes the rendered prompt/tool schema, so the final
    // recall turn cannot be a valid same-prefix cache-hit proof.
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
  }, 'Mixed turn 6. Do not call any tools. Recall this same chat. Reply with the exact labels M3_MIX_TEXT_OFF, MM3_MIX_IMAGE_RED, M3_MIX_AUTO_TEXT, M3_MIX_TOOL_ON, and M3_MIX_TOOL_AUTO. No extra bullets.')

  let toolOnFile = null
  let toolAutoFile = null
  try { toolOnFile = readFileSync(path.join(toolDir, 'm3_mixed_tool_on.txt'), 'utf8').trim() } catch {}
  try { toolAutoFile = readFileSync(path.join(toolDir, 'm3_mixed_tool_auto.txt'), 'utf8').trim() } catch {}
  return {
    chatId: chat.id,
    toolDir,
    attachment: { name: 'mixed-red-label.png', color: 'red', text: 'RED', bytesBase64Chars: redDataUrl.length },
    toolFiles: { on: toolOnFile, auto: toolAutoFile },
    turns,
  }
}

function deriveVerdict(result) {
  const failures = []
  const logs = (result.sessionLogsEnd || []).join('\n')
  for (const [label, regex] of [
    ['M3 autodetect log missing', /MiniMax-M3 AUTODETECTED/i],
    ['paged cache off log missing', /paged_cache=OFF/i],
    ['TurboQuant KV skip log missing', /tq_kv=SKIP\(native MSA\)/i],
    ['M3 VL route log missing', /vl_route=ON/i],
    ['M3 tool parser log missing', /tool_parser=minimax_m3/i],
    ['M3 reasoning parser log missing', /reasoning_parser=minimax_m3/i],
    ['M3 JIT off log missing', /jit=off|jit=OFF/i],
  ]) {
    if (!regex.test(logs)) failures.push(label)
  }
  const turns = result.ui?.multiturn10?.turns || []
  if (turns.length !== 10) failures.push(`10-turn UI count mismatch: ${turns.length}`)
  if (turns.some((t) => t.sendError)) failures.push('UI send error')
  if (turns.some((t) => t.score.empty || t.hiddenOnly)) failures.push('UI hidden-only or empty assistant turn')
  if (turns.some((t) => t.score.leakedThinkTags)) failures.push('UI raw think tag leak')
  if (turns.some((t) => t.score.loopSuspect)) failures.push('UI loop suspect')
  if (turns.some((t) => t.autonomousAssistantTurn)) failures.push('UI autonomous assistant turn after completion')
  const final = turns[turns.length - 1]?.content || ''
  for (const label of ['EBS', 'AP', 'GL', 'RESP', 'CP', 'PROFILE', 'PATCH']) {
    if (!new RegExp(`\\b${label}\\b`, 'i').test(final)) failures.push(`final recall missing ${label}`)
  }
  const cachedTurns = turns.filter((t) => Number(t.metrics?.cachedTokens || 0) > 0)
  if (!cachedTurns.length) failures.push('UI multiturn cachedTokens never exceeded 0')
  for (const row of result.ui?.reasoningModes || []) {
    if (row.score.empty || row.hiddenOnly) failures.push(`reasoning ${row.mode} empty or hidden-only`)
    if (row.score.leakedThinkTags) failures.push(`reasoning ${row.mode} raw tag leak`)
  }
  const off = result.ui?.reasoningModes?.find((row) => row.mode === 'off')
  const on = result.ui?.reasoningModes?.find((row) => row.mode === 'on')
  if (off && off.reasoningChars > 0) failures.push(`reasoning off produced reasoningChars=${off.reasoningChars}`)
  if (on && on.reasoningChars <= 0) failures.push('reasoning on produced no reasoning content')
  const tool = result.ui?.toolUse
  if (!tool || tool.fileContent !== 'M3_TOOL_OK') failures.push('UI tool execution file proof missing')
  if (tool?.score?.empty || tool?.hiddenOnly) failures.push('UI tool final answer empty or hidden-only')
  if (tool && !hasExactMarker(tool.content || '', 'M3_TOOL_OK_DONE')) failures.push('UI tool final marker missing M3_TOOL_OK_DONE')
  const long = result.ui?.longContextPrefix
  if (!/PROFILE_OPTION_SENTINEL_ZETA_173/i.test(long?.turn2?.content || '')) failures.push('long-context sentinel recall failed')
  if (Number(long?.turn2?.metrics?.cachedTokens || 0) <= 0) failures.push('long-context turn2 cachedTokens=0')
  const image = result.ui?.imageVl
  if (!image || image.sendError) failures.push(`MM3 image UI send failed: ${image?.sendError || 'missing image row'}`)
  if (image?.score?.empty || image?.hiddenOnly) failures.push('MM3 image UI response empty or hidden-only')
  if (image?.score?.leakedThinkTags) failures.push('MM3 image UI raw think tag leak')
  if (image?.score?.loopSuspect) failures.push('MM3 image UI loop suspect')
  if (image && (!/MM3_IMAGE_RED/i.test(image.content || '') || rejectsRedImageLabel(image.content || ''))) failures.push('MM3 image UI did not identify red image')
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
  const mixed = result.ui?.mixedSession
  if (!mixed) failures.push('mixed UI session missing')
  else {
    const labels = mixed.turns || []
    if (labels.length !== 6) failures.push(`mixed UI turn count mismatch: ${labels.length}`)
    if (labels.some((t) => t.sendError)) failures.push('mixed UI send error')
    if (labels.some((t) => t.score.empty || t.hiddenOnly)) failures.push('mixed UI empty or hidden-only assistant turn')
    if (labels.some((t) => t.score.leakedThinkTags)) failures.push('mixed UI raw think tag leak')
    if (labels.some((t) => t.score.loopSuspect)) failures.push('mixed UI loop suspect')
    if (labels.some((t) => t.autonomousAssistantTurn)) failures.push('mixed UI autonomous assistant turn after completion')
    if (mixed.toolFiles?.on !== 'M3_MIX_TOOL_ON') failures.push('mixed UI reasoning-on tool file proof missing')
    if (mixed.toolFiles?.auto !== 'M3_MIX_TOOL_AUTO') failures.push('mixed UI reasoning-auto tool file proof missing')
    const byLabel = Object.fromEntries(labels.map((t) => [t.label, t]))
    if ((byLabel.text_reasoning_off?.reasoningChars || 0) > 0) failures.push('mixed reasoning-off turn emitted reasoning')
    if ((byLabel.image_reasoning_on?.reasoningChars || 0) <= 0) failures.push('mixed reasoning-on image turn emitted no reasoning')
    if ((byLabel.tool_reasoning_on?.reasoningChars || 0) <= 0) failures.push('mixed reasoning-on tool turn emitted no reasoning')
    if (!hasExactMarker(byLabel.tool_reasoning_on?.content || '', 'M3_MIX_TOOL_ON_DONE')) failures.push('mixed reasoning-on tool final marker missing M3_MIX_TOOL_ON_DONE')
    if (!hasExactMarker(byLabel.tool_reasoning_auto?.content || '', 'M3_MIX_TOOL_AUTO_DONE')) failures.push('mixed reasoning-auto tool final marker missing M3_MIX_TOOL_AUTO_DONE')
    const onToolCalls = toolCallingCount(byLabel.tool_reasoning_on, 'run_command')
    const autoToolCalls = toolCallingCount(byLabel.tool_reasoning_auto, 'run_command')
    if (onToolCalls !== 1) failures.push(`mixed reasoning-on expected exactly 1 run_command call, saw ${onToolCalls}`)
    if (autoToolCalls !== 1) failures.push(`mixed reasoning-auto expected exactly 1 run_command call, saw ${autoToolCalls}`)
    const mixedImageContent = byLabel.image_reasoning_on?.content || ''
    if (
      !/MM3_MIX_IMAGE_RED/i.test(mixedImageContent)
      || rejectsRedImageLabel(mixedImageContent)
    ) failures.push('mixed image turn did not identify red image')
    const postMediaCacheHit = ['tool_reasoning_on', 'tool_reasoning_auto', 'final_recall_cache']
      .some((label) => Number(byLabel[label]?.metrics?.cachedTokens || 0) > 0)
    if (!postMediaCacheHit) failures.push('mixed post-media/tool cachedTokens never exceeded 0')
    const finalMixed = byLabel.final_recall_cache?.content || ''
    for (const label of ['M3_MIX_TEXT_OFF', 'MM3_MIX_IMAGE_RED', 'M3_MIX_AUTO_TEXT', 'M3_MIX_TOOL_ON', 'M3_MIX_TOOL_AUTO']) {
      if (!new RegExp(label, 'i').test(finalMixed)) failures.push(`mixed final recall missing ${label}`)
    }
  }
  for (const [name, row] of Object.entries(result.api || {})) {
    if (!row || typeof row !== 'object' || !Object.prototype.hasOwnProperty.call(row, 'ok')) continue
    if (!row?.ok) failures.push(`API ${name} HTTP failed`)
    if (row?.textScore?.empty) failures.push(`API ${name} visible text empty`)
    if (row?.textScore?.loopSuspect) failures.push(`API ${name} loop suspect`)
    if (row?.textScore?.leakedThinkTags) failures.push(`API ${name} raw think tag leak`)
  }
  if (!result.api?.chatTool?.hasToolCalls) failures.push('API Chat required tool did not return tool_calls')
  if (result.api?.chat?.ok && !hasExactMarker(result.api.chat.text || '', 'API_CHAT_OK')) failures.push('API Chat missing exact marker API_CHAT_OK')
  if (result.api?.responses1?.ok && !hasExactMarker(result.api.responses1.text || '', 'API_RESP_OK')) failures.push('API Responses turn1 missing exact marker API_RESP_OK')
  if (result.api?.responses2?.ok && !equalsText(result.api.responses2.text || '', 'violet')) failures.push(`API Responses previous_response_id recall mismatch: ${result.api.responses2.text || ''}`)
  if (result.api?.anthropicMessages?.ok && !/route ok/i.test(result.api.anthropicMessages.text || '')) failures.push('Anthropic route missing exact phrase route ok')
  if (result.api?.ollamaChat?.ok && !hasExactMarker(result.api.ollamaChat.text || '', 'OLLAMA_CHAT_OK')) failures.push('Ollama chat missing exact marker OLLAMA_CHAT_OK')
  if (result.api?.ollamaGenerate?.ok && !hasExactMarker(result.api.ollamaGenerate.text || '', 'OLLAMA_GENERATE_OK')) failures.push('Ollama generate missing exact marker OLLAMA_GENERATE_OK')
  const apiToolArgs = JSON.stringify(result.api?.chatTool?.message?.tool_calls || [])
  if (result.api?.chatTool?.hasToolCalls && !hasExactMarker(apiToolArgs, 'API_TOOL_OK')) failures.push('API Chat tool arguments missing API_TOOL_OK')
  for (const [name, row] of Object.entries(result.streaming || {})) {
    if (!row?.ok) failures.push(`stream ${name} HTTP failed`)
    if (row?.textScore?.empty && !row?.hasToolCallSignal) failures.push(`stream ${name} reconstructed empty content/tool signal`)
    if (row?.textScore?.leakedThinkTags) failures.push(`stream ${name} raw think tag leak`)
  }
  if (!result.streaming?.chatTool?.hasToolCallSignal) failures.push('streaming Chat tool call signal missing')
  if (!result.streaming?.responsesTool?.hasToolCallSignal) failures.push('streaming Responses tool call signal missing')
  if (result.streaming?.chatText?.ok && !hasExactMarker(result.streaming.chatText.content || '', 'MM3_STREAM_CHAT_OK')) failures.push('streaming Chat missing exact marker MM3_STREAM_CHAT_OK')
  if (result.streaming?.responsesImage?.ok && (!hasExactMarker(result.streaming.responsesImage.content || '', 'MM3_STREAM_IMAGE_RED') || rejectsRedImageLabel(result.streaming.responsesImage.content || ''))) failures.push('streaming Responses image missing exact MM3_STREAM_IMAGE_RED')
  if (result.streaming?.chatTool?.hasToolCallSignal && !hasExactMarker(result.streaming.chatTool.toolArgs || '', 'MM3_STREAM_CHAT_TOOL')) failures.push('streaming Chat tool arguments missing MM3_STREAM_CHAT_TOOL')
  if (result.streaming?.responsesTool?.hasToolCallSignal && !hasExactMarker(result.streaming.responsesTool.toolArgs || '', 'MM3_STREAM_RESP_TOOL')) failures.push('streaming Responses tool arguments missing MM3_STREAM_RESP_TOOL')
  const responseStreamToolCalls = result.streaming?.responsesTool?.completedToolCalls || []
  const exactResponseStreamToolCalls = responseStreamToolCalls.filter((call) => (
    call?.name === 'record_mm3_stream_response_label'
    && hasExactMarker(call?.arguments || '', 'MM3_STREAM_RESP_TOOL')
  ))
  if (result.streaming?.responsesTool?.hasToolCallSignal && exactResponseStreamToolCalls.length !== 1) {
    failures.push(`streaming Responses expected exactly 1 completed record_mm3_stream_response_label call with MM3_STREAM_RESP_TOOL, saw ${exactResponseStreamToolCalls.length}`)
  }
  return { status: failures.length ? 'fail' : 'pass', failures }
}

async function main() {
  if (!existsSync(appExe)) throw new Error(`Installed app executable missing: ${appExe}`)
  if (!existsSync(modelPath)) throw new Error(`MM3 model path missing: ${modelPath}`)
  mkdirSync(proofDir, { recursive: true })
  const userDataDir = mkdtempSync(path.join(tmpdir(), 'vmlx-mm3-stress-userdata-'))
  const appLog = []
  const debugPort = await freePort()
  const sessionPort = Number(process.env.VMLX_MM3_PORT || await freePort())
  const result = {
    generatedAt: startedAt.toISOString(),
    status: 'running',
    proofDir,
    outJson,
    shotPath,
    repoDir,
    panelDir,
    appPath,
    modelPath,
    debugPort,
    sessionPort,
    loadPolls: [],
    sourceTrace: {
      appFamilyDefaults: 'panel/src/main/sessions.ts:127-194',
      appM3LaunchPolicy: 'panel/src/main/sessions.ts:1364-1380, 1496-1500, 2858-2885',
      panelRegistry: 'panel/src/main/model-config-registry.ts:211-214, 880-891',
      cliM3Transparency: 'vmlx_engine/cli.py:504-543',
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
    writeResult(result)

    const chat = await page.evaluate(async ({ modelPath }) => {
      return window.api.chat.create('MM3 1.5.64 10-turn stress', modelPath.split('/').pop(), undefined, modelPath)
    }, { modelPath })
    await page.evaluate(async ({ chatId }) => {
      await window.api.chat.setOverrides(chatId, {
        wireApi: 'responses',
        maxTokens: 220,
        temperature: 0.7,
        topP: 0.9,
        enableThinking: false,
      })
    }, { chatId: chat.id })
    const prompts = [
      'Define Oracle EBS in one sentence. Use label EBS in the answer.',
      'Set anchor AP=payables. Explain the AP module in two concise bullets and include label AP.',
      'Set anchor GL=ledger. Explain the GL module in two concise bullets and include label GL.',
      'Set anchor RESP=responsibility. Explain EBS responsibilities in two concise bullets and include label RESP.',
      'Set anchor CP=concurrent programs. Explain concurrent programs in two concise bullets and include label CP.',
      'Set anchor PROFILE=profile options. Explain profile options in two concise bullets and include label PROFILE.',
      'Set anchor PATCH=patching. Explain EBS patching basics in two concise bullets and include label PATCH.',
      'Compare AP and GL in one short paragraph. Use labels AP and GL explicitly.',
      'What are the three admin anchors RESP, CP, PROFILE? Use those labels and one short phrase for each.',
      'Recall all seven anchors EBS, AP, GL, RESP, CP, PROFILE, PATCH. Give one short phrase for each and use every label exactly once.',
    ]
    result.ui = { multiturn10: { chatId: chat.id, turns: [] }, reasoningModes: [], toolUse: null, longContextPrefix: null, mixedSession: null }
    for (let i = 0; i < prompts.length; i += 1) {
      const turn = await sendUiTurn(page, chat.id, prompts[i], endpoint)
      turn.turn = i + 1
      result.ui.multiturn10.turns.push(turn)
      writeResult(result)
    }

    const reasoningModes = [
      {
        mode: 'off',
        overrides: { wireApi: 'responses', maxTokens: 260, temperature: 0.2, topP: 0.9, enableThinking: false },
        prompt: 'Reasoning OFF test: reply visibly with exactly OFF_VISIBLE_OK and no explanation.',
      },
      {
        mode: 'on',
        overrides: { wireApi: 'responses', maxTokens: 700, maxThinkingTokens: 220, temperature: 0.3, topP: 0.9, enableThinking: true },
        prompt: 'Reasoning ON test: think briefly, then visible final answer must contain ON_VISIBLE_OK.',
      },
      {
        mode: 'auto',
        overrides: { wireApi: 'responses', maxTokens: 700, maxThinkingTokens: 220, temperature: 0.3, topP: 0.9 },
        prompt: 'Reasoning AUTO test: if useful think briefly, then visible final answer must contain AUTO_VISIBLE_OK.',
      },
    ]
    for (const mode of reasoningModes) {
      const modeChat = await page.evaluate(async ({ modelPath, mode }) => {
        return window.api.chat.create(`MM3 reasoning ${mode}`, modelPath.split('/').pop(), undefined, modelPath)
      }, { modelPath, mode: mode.mode })
      await page.evaluate(async ({ chatId, overrides }) => window.api.chat.setOverrides(chatId, overrides), { chatId: modeChat.id, overrides: mode.overrides })
      const row = await sendUiTurn(page, modeChat.id, mode.prompt, endpoint)
      row.mode = mode.mode
      row.chatId = modeChat.id
      result.ui.reasoningModes.push(row)
      writeResult(result)
    }

    const toolDir = path.join(proofDir, 'tool-workdir')
    mkdirSync(toolDir, { recursive: true })
    const toolChat = await page.evaluate(async ({ modelPath }) => {
      return window.api.chat.create('MM3 UI builtin tool stress', modelPath.split('/').pop(), undefined, modelPath)
    }, { modelPath })
    await page.evaluate(async ({ chatId, toolDir }) => window.api.chat.setOverrides(chatId, {
      wireApi: 'responses',
      maxTokens: 800,
      temperature: 0.2,
      topP: 0.9,
      enableThinking: false,
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
      maxToolIterations: 4,
      toolResultMaxChars: 8000,
    }), { chatId: toolChat.id, toolDir })
    const toolRun = await sendUiTurn(
      page,
      toolChat.id,
      'Use the run_command tool exactly once to create a file named m3_tool_probe.txt in the working directory containing exactly M3_TOOL_OK. After the tool result, reply with M3_TOOL_OK_DONE.',
      endpoint,
    )
    let toolFile = null
    try {
      toolFile = readFileSync(path.join(toolDir, 'm3_tool_probe.txt'), 'utf8').trim()
    } catch {}
    result.ui.toolUse = { chatId: toolChat.id, toolDir, fileContent: toolFile, ...toolRun }
    writeResult(result)

    const facts = []
    for (let i = 1; i <= 180; i += 1) {
      facts.push(`MM3_CONTEXT_PAD_${String(i).padStart(3, '0')}: Oracle EBS admin note ${i} covers AP, GL, responsibilities, concurrent programs, profile options, patching, and diagnostics.`)
    }
    facts.push('MM3_TARGET_SENTINEL_LINE: The exact answer to the future sentinel question is PROFILE_OPTION_SENTINEL_ZETA_173.')
    const longChat = await page.evaluate(async ({ modelPath }) => {
      return window.api.chat.create('MM3 long-context prefix cache stress', modelPath.split('/').pop(), undefined, modelPath)
    }, { modelPath })
    await page.evaluate(async ({ chatId }) => window.api.chat.setOverrides(chatId, {
      wireApi: 'responses',
      maxTokens: 100,
      temperature: 0.2,
      topP: 0.9,
      enableThinking: false,
    }), { chatId: longChat.id })
    const long1 = await sendUiTurn(page, longChat.id, `${facts.join('\n')}\n\nAcknowledge by replying exactly LONG_CONTEXT_READY.`, endpoint)
    const cacheBeforeTurn2 = await page.evaluate(async ({ endpoint, id }) => window.api.cache.stats(endpoint, id), { endpoint, id: session.id }).catch((error) => ({ error: String(error?.message || error) }))
    const long2 = await sendUiTurn(page, longChat.id, 'Using the earlier long context, what is the exact sentinel? Reply with only the sentinel and no prose.', endpoint)
    const cacheAfterTurn2 = await page.evaluate(async ({ endpoint, id }) => window.api.cache.stats(endpoint, id), { endpoint, id: session.id }).catch((error) => ({ error: String(error?.message || error) }))
    result.ui.longContextPrefix = { chatId: longChat.id, turn1: long1, turn2: long2, cacheBeforeTurn2, cacheAfterTurn2 }
    writeResult(result)

    const redDataUrl = labeledRedPngDataUrl()
    const imageChat = await page.evaluate(async ({ modelPath }) => {
      return window.api.chat.create('MM3 VL image stress', modelPath.split('/').pop(), undefined, modelPath)
    }, { modelPath })
    await page.evaluate(async ({ chatId }) => window.api.chat.setOverrides(chatId, {
      wireApi: 'responses',
      maxTokens: 80,
      temperature: 0.1,
      topP: 0.9,
      enableThinking: false,
    }), { chatId: imageChat.id })
    const imageTurn = await sendUiTurn(
      page,
      imageChat.id,
      'The attached image has a red background and a large white three-letter word. Reply with exactly MM3_IMAGE_RED only if the visible word is RED.',
      endpoint,
      [{ dataUrl: redDataUrl, name: 'red-label.png', kind: 'image', type: 'image/png', size: redDataUrl.length }],
    )
    result.ui.imageVl = { chatId: imageChat.id, attachment: { name: 'red-label.png', color: 'red', text: 'RED', bytesBase64Chars: redDataUrl.length }, ...imageTurn }
    writeResult(result)

    result.ui.mixedSession = await runMixedUiSession(page, modelPath, endpoint, proofDir)
    writeResult(result)

    const models = await getJson(`http://127.0.0.1:${sessionPort}/v1/models`).catch((error) => ({ error: String(error?.message || error) }))
    const servedModel = models?.data?.[0]?.id || result.sessionAfterStart?.modelName || path.basename(modelPath)
    result.api = { models, servedModel }

    const chatResp = await postJson(`http://127.0.0.1:${sessionPort}/v1/chat/completions`, {
      model: servedModel,
      messages: [{ role: 'user', content: 'API Chat check: define Oracle EBS in one sentence and include API_CHAT_OK.' }],
      max_tokens: 160,
      temperature: 0.2,
      top_p: 0.9,
      enable_thinking: false,
    })
    const chatText = chatResp.json?.choices?.[0]?.message?.content || ''
    result.api.chat = { ...chatResp, text: chatText, textScore: scoreText(chatText), cachedTokens: cacheTokensFromUsage(chatResp.json) }
    writeResult(result)

    const toolResp = await postJson(`http://127.0.0.1:${sessionPort}/v1/chat/completions`, {
      model: servedModel,
      messages: [{ role: 'user', content: 'Call the provided function with label API_TOOL_OK.' }],
      tools: [{
        type: 'function',
        function: {
          name: 'record_mm3_label',
          description: 'Record a label for the MM3 API stress test.',
          parameters: {
            type: 'object',
            properties: { label: { type: 'string' } },
            required: ['label'],
          },
        },
      }],
      tool_choice: { type: 'function', function: { name: 'record_mm3_label' } },
      max_tokens: 240,
      temperature: 0.2,
      top_p: 0.9,
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

    const resp1 = await postJson(`http://127.0.0.1:${sessionPort}/v1/responses`, {
      model: servedModel,
      input: 'Responses check: remember API_RESP_ANCHOR=violet and reply with API_RESP_OK.',
      max_output_tokens: 160,
      temperature: 0.2,
      top_p: 0.9,
      enable_thinking: false,
    })
    const resp1Text = responseTextFromResponsesObject(resp1.json)
    result.api.responses1 = { ...resp1, text: resp1Text, textScore: scoreText(resp1Text), cachedTokens: cacheTokensFromUsage(resp1.json) }
    writeResult(result)

    const resp2 = await postJson(`http://127.0.0.1:${sessionPort}/v1/responses`, {
      model: servedModel,
      previous_response_id: resp1.json?.id,
      input: 'What was API_RESP_ANCHOR? Reply with only the value.',
      max_output_tokens: 80,
      temperature: 0.2,
      top_p: 0.9,
      enable_thinking: false,
    })
    const resp2Text = responseTextFromResponsesObject(resp2.json)
    result.api.responses2 = { ...resp2, text: resp2Text, textScore: scoreText(resp2Text), cachedTokens: cacheTokensFromUsage(resp2.json) }
    writeResult(result)

    result.streaming = {}
    result.streaming.chatText = await streamSse(`http://127.0.0.1:${sessionPort}/v1/chat/completions`, {
      model: servedModel,
      messages: [{ role: 'user', content: 'Streaming Chat mixed check: with reasoning on, reply visibly with MM3_STREAM_CHAT_OK and define Oracle EBS in one sentence.' }],
      max_tokens: 180,
      temperature: 0.2,
      top_p: 0.9,
      enable_thinking: true,
      max_thinking_tokens: 120,
    }, 'chat')
    writeResult(result)
    result.streaming.responsesImage = await streamSse(`http://127.0.0.1:${sessionPort}/v1/responses`, {
      model: servedModel,
      input: [{
        type: 'message',
        role: 'user',
        content: [
          { type: 'input_text', text: 'Streaming Responses image check: if the attached image is red, reply visibly with MM3_STREAM_IMAGE_RED.' },
          { type: 'input_image', image_url: redDataUrl },
        ],
      }],
      max_output_tokens: 160,
      temperature: 0.1,
      top_p: 0.9,
      enable_thinking: false,
    }, 'responses')
    writeResult(result)
    result.streaming.chatTool = await streamSse(`http://127.0.0.1:${sessionPort}/v1/chat/completions`, {
      model: servedModel,
      messages: [{ role: 'user', content: 'Streaming Chat tool check: call the function with label MM3_STREAM_CHAT_TOOL.' }],
      tools: [{
        type: 'function',
        function: {
          name: 'record_mm3_stream_label',
          description: 'Record an MM3 streaming label.',
          parameters: {
            type: 'object',
            properties: { label: { type: 'string' } },
            required: ['label'],
          },
        },
      }],
      tool_choice: { type: 'function', function: { name: 'record_mm3_stream_label' } },
      max_tokens: 220,
      temperature: 0.2,
      top_p: 0.9,
      enable_thinking: false,
    }, 'chat')
    writeResult(result)
    result.streaming.responsesTool = await streamSse(`http://127.0.0.1:${sessionPort}/v1/responses`, {
      model: servedModel,
      input: 'Streaming Responses tool check: call the function with label MM3_STREAM_RESP_TOOL.',
      tools: [{
        type: 'function',
        name: 'record_mm3_stream_response_label',
        description: 'Record an MM3 Responses streaming label.',
        parameters: {
          type: 'object',
          properties: { label: { type: 'string' } },
          required: ['label'],
        },
      }],
      tool_choice: { type: 'function', name: 'record_mm3_stream_response_label' },
      max_output_tokens: 220,
      temperature: 0.2,
      top_p: 0.9,
      enable_thinking: false,
    }, 'responses')
    writeResult(result)

    const anthropicResp = await postJson(`http://127.0.0.1:${sessionPort}/v1/messages`, {
      model: servedModel,
      max_tokens: 120,
      messages: [{ role: 'user', content: 'Anthropic route check: define Oracle EBS in one short sentence, then add the words route ok.' }],
      thinking: { type: 'disabled' },
    })
    const anthropicVisible = anthropicText(anthropicResp.json)
    result.api.anthropicMessages = { ...anthropicResp, text: anthropicVisible, textScore: scoreText(anthropicVisible), cachedTokens: cacheTokensFromUsage(anthropicResp.json) }
    writeResult(result)

    const ollamaChatResp = await postJson(`http://127.0.0.1:${sessionPort}/api/chat`, {
      model: servedModel,
      messages: [{ role: 'user', content: 'Ollama chat check: reply with OLLAMA_CHAT_OK and define Oracle EBS in one short sentence.' }],
      stream: false,
      think: false,
      options: { temperature: 0.2, top_p: 0.9, num_predict: 120 },
    })
    const ollamaChatVisible = ollamaText(ollamaChatResp.json)
    result.api.ollamaChat = { ...ollamaChatResp, text: ollamaChatVisible, textScore: scoreText(ollamaChatVisible), cachedTokens: cacheTokensFromUsage(ollamaChatResp.json) }
    writeResult(result)

    const ollamaGenerateResp = await postJson(`http://127.0.0.1:${sessionPort}/api/generate`, {
      model: servedModel,
      prompt: 'Ollama generate check: reply with OLLAMA_GENERATE_OK and define Oracle EBS in one short sentence.',
      stream: false,
      think: false,
      options: { temperature: 0.2, top_p: 0.9, num_predict: 120 },
    })
    const ollamaGenerateVisible = ollamaText(ollamaGenerateResp.json)
    result.api.ollamaGenerate = { ...ollamaGenerateResp, text: ollamaGenerateVisible, textScore: scoreText(ollamaGenerateVisible), cachedTokens: cacheTokensFromUsage(ollamaGenerateResp.json) }
    result.cacheEnd = await page.evaluate(async ({ endpoint, id }) => window.api.cache.stats(endpoint, id), { endpoint, id: session.id }).catch((error) => ({ error: String(error?.message || error) }))
    result.sessionLogsEnd = await page.evaluate(async ({ id }) => window.api.sessions.getLogs(id), { id: session.id })
    result.appLogTail = appLog.slice(-160)
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
      proofDir,
      outJson,
      shotPath,
      sessionPort,
      cachedTurns: result.ui.multiturn10.turns.map((t) => ({ turn: t.turn, cachedTokens: t.metrics?.cachedTokens, tps: t.metrics?.tokensPerSecond })),
      reasoning: result.ui.reasoningModes.map((r) => ({ mode: r.mode, contentChars: r.content.length, reasoningChars: r.reasoningChars, hiddenOnly: r.hiddenOnly })),
      toolFile,
      longTurn2: result.ui.longContextPrefix.turn2.content,
      imageVl: result.ui.imageVl?.content,
      mixed: result.ui.mixedSession?.turns?.map((t) => ({ label: t.label, cachedTokens: t.metrics?.cachedTokens, content: t.score.preview, reasoningChars: t.reasoningChars })),
      generationDefaults: result.generationDefaults,
      api: {
        chat: result.api.chat.textScore.preview,
        chatToolCalls: result.api.chatTool.hasToolCalls,
        responses1: result.api.responses1.textScore.preview,
        responses2: result.api.responses2.textScore.preview,
        anthropicMessages: result.api.anthropicMessages.textScore.preview,
        ollamaChat: result.api.ollamaChat.textScore.preview,
        ollamaGenerate: result.api.ollamaGenerate.textScore.preview,
      },
      streaming: {
        chatText: result.streaming.chatText?.textScore?.preview,
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
    if (!process.env.VMLX_KEEP_MM3_USER_DATA) {
      rmSync(userDataDir, { recursive: true, force: true })
    }
  }
}

main().catch((error) => {
  console.error(error?.stack || error?.message || String(error))
  process.exit(1)
})
