#!/usr/bin/env node
import { spawn } from 'node:child_process'
import { createServer } from 'node:http'
import { cpSync, existsSync, mkdirSync, mkdtempSync, rmSync, writeFileSync } from 'node:fs'
import { homedir, tmpdir } from 'node:os'
import path from 'node:path'
import { fileURLToPath } from 'node:url'
import { chromium } from 'playwright-core'

const __dirname = path.dirname(fileURLToPath(import.meta.url))
const panelDir = path.resolve(__dirname, '..')
const repoDir = path.resolve(panelDir, '..')

const rowName = process.env.VMLX_CLEAN_ROW || 'gemma4-e2b-mxfp4-clean-start'
const modelPath = process.env.VMLX_CLEAN_MODEL_PATH || '/Users/example/models/OsaurusAI--gemma-4-E2B-it-qat-MXFP4'
const expectedFamily = process.env.VMLX_CLEAN_EXPECT_FAMILY || 'gemma4'
const appPath = process.env.VMLX_APP_PATH || '/Applications/vMLX.app'
const appExe = path.join(appPath, 'Contents', 'MacOS', 'vMLX')
const useRealProfile = /^(1|true|yes)$/i.test(process.env.VMLX_CLEAN_USE_REAL_PROFILE || '')
const realUserDataDir = process.env.VMLX_CLEAN_REAL_USER_DATA_DIR || path.join(homedir(), 'Library', 'Application Support', 'vMLX')
const startedAt = new Date()
const stamp = startedAt.toISOString().replace(/[:.]/g, '-')
const proofDir = path.resolve(process.env.VMLX_CLEAN_PROOF_DIR || path.join(repoDir, 'build', `live-clean-start-${rowName}-${stamp}`))
const outJson = path.join(proofDir, 'clean-start-proof.json')
const shotPath = path.join(proofDir, 'clean-start-final.png')

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

function writeResult(result) {
  writeFileSync(outJson, JSON.stringify(result, null, 2))
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
      if (window.api?.sessions && window.api?.chat && window.api?.cache && window.api?.performance) return true
      await new Promise((resolve) => setTimeout(resolve, 100))
    }
    throw new Error('window.api not ready')
  })
}

async function waitForHealth(page, port, sessionId, result) {
  const endpoint = { host: '127.0.0.1', port }
  const started = Date.now()
  let last = null
  while (Date.now() - started < 900_000) {
    last = await page.evaluate(async ({ endpoint, sessionId }) => {
      const [health, logs] = await Promise.all([
        window.api.performance.health(endpoint).catch((error) => ({ error: String(error?.message || error) })),
        window.api.sessions.getLogs(sessionId).catch(() => []),
      ])
      return { health, logsTail: Array.isArray(logs) ? logs.slice(-120) : [] }
    }, { endpoint, sessionId })
    result.loadPolls.push({ elapsedMs: Date.now() - started, health: last.health, logsTail: last.logsTail.slice(-24) })
    writeResult(result)
    if (last.health?.status === 'healthy' && last.health?.model_loaded === true) return last.health
    await sleep(3000)
  }
  throw new Error(`session did not become healthy: ${JSON.stringify(last)}`)
}

function parseConfig(session) {
  try {
    return JSON.parse(session?.config || '{}')
  } catch {
    return {}
  }
}

function scoreText(text) {
  const value = String(text || '')
  const words = value.toLowerCase().replace(/[^a-z0-9_]+/g, ' ').trim().split(/\s+/).filter(Boolean)
  let maxRun = 0
  let prev = ''
  let run = 0
  for (const line of value.split(/\n+/).map((s) => s.trim()).filter(Boolean)) {
    if (line === prev) run += 1
    else {
      prev = line
      run = 1
    }
    maxRun = Math.max(maxRun, run)
  }
  return {
    chars: value.length,
    words: words.length,
    empty: value.trim().length === 0,
    leakedReasoningTags: /<\|channel\>|<channel\|>|<think>|<\/think>|<mm:think/i.test(value),
    loopSuspect: maxRun >= 4,
    preview: value.trim().slice(0, 500),
  }
}

function extractLaunchCommand(logLines) {
  const lines = Array.isArray(logLines) ? logLines : []
  return [...lines].reverse().find((line) => /\b(?:vmlx_engine\.cli|vmlx-engine)\b/.test(line) && /\bserve\b/.test(line)) || ''
}

function commandHasFlag(command, flag) {
  return new RegExp(`(?:^|\\s)${flag.replace(/[.*+?^${}()|[\]\\]/g, '\\$&')}(?:\\s|$)`).test(command)
}

function commandHasFlagValue(command, flag, value) {
  return new RegExp(`(?:^|\\s)${flag.replace(/[.*+?^${}()|[\]\\]/g, '\\$&')}\\s+${String(value).replace(/[.*+?^${}()|[\]\\]/g, '\\$&')}(?:\\s|$)`).test(command)
}

async function postJson(url, body) {
  const res = await fetch(url, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(body),
  })
  const text = await res.text()
  let json = null
  try {
    json = text ? JSON.parse(text) : null
  } catch {}
  return { ok: res.ok, status: res.status, text, json }
}

function extractChatText(json) {
  const message = json?.choices?.[0]?.message || {}
  const content = message.content
  if (typeof content === 'string') return content
  if (Array.isArray(content)) {
    return content.map((part) => part?.text || part?.content || '').join('')
  }
  return ''
}

function extractResponsesText(json) {
  if (typeof json?.output_text === 'string') return json.output_text
  const out = []
  for (const item of json?.output || []) {
    for (const part of item?.content || []) {
      if (typeof part?.text === 'string') out.push(part.text)
    }
  }
  return out.join('')
}

async function readSseText(res, kind) {
  const raw = await res.text()
  const lines = raw.split(/\r?\n/)
  const chunks = []
  for (const line of lines) {
    if (!line.startsWith('data:')) continue
    const data = line.slice(5).trim()
    if (!data || data === '[DONE]') continue
    let event
    try {
      event = JSON.parse(data)
    } catch {
      continue
    }
    if (kind === 'chat') {
      const delta = event?.choices?.[0]?.delta || {}
      if (typeof delta.content === 'string') chunks.push(delta.content)
      if (Array.isArray(delta.content)) {
        chunks.push(...delta.content.map((part) => part?.text || part?.content || '').filter(Boolean))
      }
    } else {
      if (event?.type === 'response.output_text.delta' && typeof event?.delta === 'string') chunks.push(event.delta)
      if (event?.type === 'response.completed' && typeof event?.response?.output_text === 'string') chunks.push(event.response.output_text)
    }
  }
  return { raw: raw.slice(0, 5000), text: chunks.join('') }
}

async function fetchGatewayText(gatewayBase, servedModel) {
  const chatBody = {
    model: servedModel,
    messages: [
      { role: 'user', content: 'Reply with exactly: GATEWAY_CHAT_OK' },
    ],
    temperature: 0,
    max_tokens: 64,
    enable_thinking: false,
  }
  const responsesBody = {
    model: servedModel,
    input: 'Reply with exactly: GATEWAY_RESP_OK',
    temperature: 0,
    max_output_tokens: 64,
    enable_thinking: false,
  }
  const chat = await postJson(`${gatewayBase}/v1/chat/completions`, chatBody)
  const responses = await postJson(`${gatewayBase}/v1/responses`, responsesBody)
  const streamChatRes = await fetch(`${gatewayBase}/v1/chat/completions`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({
      ...chatBody,
      messages: [{ role: 'user', content: 'Reply with exactly: GATEWAY_STREAM_CHAT_OK' }],
      stream: true,
    }),
  })
  const streamResponsesRes = await fetch(`${gatewayBase}/v1/responses`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({
      ...responsesBody,
      input: 'Reply with exactly: GATEWAY_STREAM_RESP_OK',
      stream: true,
    }),
  })
  const streamChat = await readSseText(streamChatRes, 'chat')
  const streamResponses = await readSseText(streamResponsesRes, 'responses')
  return {
    chat: { ok: chat.ok, status: chat.status, text: extractChatText(chat.json), raw: chat.text.slice(0, 2000) },
    responses: { ok: responses.ok, status: responses.status, text: extractResponsesText(responses.json), raw: responses.text.slice(0, 2000) },
    streamChat: { ok: streamChatRes.ok, status: streamChatRes.status, ...streamChat },
    streamResponses: { ok: streamResponsesRes.ok, status: streamResponsesRes.status, ...streamResponses },
  }
}

async function sendDefaultUiTurn(page, session, port, modelPath) {
  const endpoint = { host: '127.0.0.1', port }
  const chat = await page.evaluate(async ({ modelPath }) => {
    return window.api.chat.create('Clean-start default chat', modelPath.split('/').pop(), undefined, modelPath)
  }, { modelPath })
  const before = await page.evaluate(async ({ chatId }) => window.api.chat.getMessages(chatId), { chatId: chat.id })
  await page.evaluate(async ({ chatId, endpoint }) => {
    await window.api.chat.sendMessage(
      chatId,
      'Clean-start default UI turn. Define Oracle EBS in one sentence and include exact marker CLEAN_START_VISIBLE_OK.',
      endpoint,
    )
  }, { chatId: chat.id, endpoint })
  const after = await page.evaluate(async ({ chatId }) => window.api.chat.getMessages(chatId), { chatId: chat.id })
  await sleep(2500)
  const afterQuiet = await page.evaluate(async ({ chatId }) => window.api.chat.getMessages(chatId), { chatId: chat.id })
  const assistants = after.filter((m) => m.role === 'assistant')
  const last = assistants[assistants.length - 1] || {}
  return {
    chatId: chat.id,
    beforeCount: before.length,
    afterCount: after.length,
    afterQuietCount: afterQuiet.length,
    content: last.content || '',
    reasoningChars: String(last.reasoningContent || '').length,
    metricsJson: last.metricsJson || '',
    score: scoreText(last.content || ''),
    hiddenOnly: !(last.content || '').trim() && !!String(last.reasoningContent || '').trim(),
  }
}

function verdict(result) {
  const failures = []
  const cfg = result.sessionConfigAfterStart || {}
  const logs = (result.sessionLogsEnd || result.sessionLogsStart || []).join('\n')
  const launchCommand = result.launchCommand || extractLaunchCommand(result.sessionLogsStart || result.sessionLogsEnd || [])
  const nativeCache = result.healthEnd?.native_cache || result.healthReady?.native_cache || {}
  const modalities = new Set(result.capabilities?.modalities || [])
  const hasDefault = (key) => Object.prototype.hasOwnProperty.call(cfg, key) && typeof cfg[key] === 'number'

  if (result.sessionsAfterDelete?.length !== 0) failures.push(`sessions not fully deleted: ${result.sessionsAfterDelete?.length}`)
  if (!result.sessionCreateResult?.success) failures.push('session create failed')
  if (!result.sessionStartResult?.success) failures.push('session start failed')
  if (!result.healthReady?.model_loaded) failures.push('model did not load')
  if (!result.defaultUiTurn || result.defaultUiTurn.score?.empty || result.defaultUiTurn.hiddenOnly) failures.push('default UI turn empty or hidden-only')
  if (result.defaultUiTurn?.score?.leakedReasoningTags) failures.push('default UI turn leaked reasoning tags')
  if (!/CLEAN_START_VISIBLE_OK/i.test(result.defaultUiTurn?.content || '')) failures.push('default UI turn missing exact marker')
  if (!result.gateway?.status?.running) failures.push('gateway not running')
  if (!result.gateway?.health?.status) failures.push('gateway health missing')
  if (!result.gateway?.models?.data?.length) failures.push('gateway models missing')
  if (!result.gateway?.capabilities?.modalities?.includes('text')) failures.push('gateway capabilities missing text')
  if (!/GATEWAY_CHAT_OK/i.test(result.gatewayText?.chat?.text || '')) failures.push('gateway chat missing exact marker')
  if (!/GATEWAY_RESP_OK/i.test(result.gatewayText?.responses?.text || '')) failures.push('gateway responses missing exact marker')
  if (!/GATEWAY_STREAM_CHAT_OK/i.test(result.gatewayText?.streamChat?.text || '')) failures.push('gateway streaming chat missing exact marker')
  if (!/GATEWAY_STREAM_RESP_OK/i.test(result.gatewayText?.streamResponses?.text || '')) failures.push('gateway streaming responses missing exact marker')

  if (!cfg.port) failures.push('session config missing generated port')
  if (cfg.usePagedCache !== false) failures.push(`usePagedCache=${cfg.usePagedCache}`)
  if (cfg.enableDiskCache !== true) failures.push(`enableDiskCache=${cfg.enableDiskCache}`)
  if (cfg.kvCacheQuantization !== 'auto') failures.push(`kvCacheQuantization=${cfg.kvCacheQuantization}`)
  if (
    cfg.defaultSamplingDefaultsDeclared !== true ||
    !hasDefault('defaultTemperature') ||
    !hasDefault('defaultTopP') ||
    !hasDefault('defaultTopK')
  ) {
    failures.push('model generation defaults missing from session config')
  }

  if (expectedFamily === 'gemma4') {
    if (cfg.toolCallParser !== 'gemma4') failures.push(`toolCallParser=${cfg.toolCallParser}`)
    if (cfg.reasoningParser !== 'gemma4') failures.push(`reasoningParser=${cfg.reasoningParser}`)
    if (cfg.isMultimodal !== true) failures.push(`isMultimodal=${cfg.isMultimodal}`)
    if (!modalities.has('text') || !modalities.has('vision')) failures.push(`Gemma4 modalities missing text/vision: ${[...modalities].join(',')}`)
    if (nativeCache.schema !== 'mixed_swa_kv_v1') failures.push(`Gemma4 native cache schema=${nativeCache.schema}`)
    if (nativeCache.generic_turboquant_kv?.enabled !== false) failures.push('Gemma4 generic TQ-KV unexpectedly enabled')
    if (!/mixed-SWA|mixed_swa|RotatingKVCache|VLM mixed-attention/i.test(logs)) failures.push('Gemma4 mixed-SWA log evidence missing')
  }

  if (expectedFamily === 'minimax_m3') {
    if (cfg.toolCallParser !== 'minimax_m3') failures.push(`toolCallParser=${cfg.toolCallParser}`)
    if (cfg.reasoningParser !== 'minimax_m3') failures.push(`reasoningParser=${cfg.reasoningParser}`)
    if (cfg.isMultimodal !== true) failures.push(`isMultimodal=${cfg.isMultimodal}`)
    if (!modalities.has('vision')) failures.push(`MM3-VL capability missing vision: ${[...modalities].join(',')}`)
    if (nativeCache.schema !== 'minimax_m3_msa_v1') failures.push(`MM3 native cache schema=${nativeCache.schema}`)
    const components = nativeCache.components || []
    if (!components.includes('msa_idx_keys')) failures.push(`MM3 native cache missing msa_idx_keys: ${components.join(',')}`)
    if (nativeCache.generic_turboquant_kv?.enabled !== false) failures.push('MM3 generic TQ-KV unexpectedly enabled')
    if (!/MiniMax-M3 AUTODETECTED|tq_kv=SKIP|paged_cache=OFF|jit=OFF|msa_per_step_sync=ON/i.test(logs)) failures.push('MM3 autodetect/cache log evidence missing')
    if (!launchCommand) failures.push('MM3 launch command missing from UI logs')
    if (launchCommand && !commandHasFlag(launchCommand, '--enable-disk-cache')) failures.push('MM3 launch argv missing --enable-disk-cache')
    if (launchCommand && commandHasFlag(launchCommand, '--disable-prefix-cache')) failures.push('MM3 launch argv incorrectly disabled prefix cache')
    if (launchCommand && commandHasFlag(launchCommand, '--use-paged-cache')) failures.push('MM3 launch argv incorrectly enabled generic paged KV cache')
    if (launchCommand && commandHasFlag(launchCommand, '--enable-block-disk-cache')) failures.push('MM3 launch argv incorrectly enabled generic block disk cache')
    if (launchCommand && commandHasFlag(launchCommand, '--kv-cache-quantization')) failures.push('MM3 launch argv incorrectly passed generic --kv-cache-quantization')
    if (launchCommand && commandHasFlag(launchCommand, '--enable-jit')) failures.push('MM3 launch argv incorrectly passed --enable-jit')
    if (launchCommand && commandHasFlag(launchCommand, '--is-mllm')) failures.push('MM3 launch argv incorrectly passed generic --is-mllm')
    if (launchCommand && !commandHasFlagValue(launchCommand, '--tool-call-parser', 'minimax_m3')) failures.push('MM3 launch argv missing --tool-call-parser minimax_m3')
    if (launchCommand && !commandHasFlagValue(launchCommand, '--reasoning-parser', 'minimax_m3')) failures.push('MM3 launch argv missing --reasoning-parser minimax_m3')
    if (launchCommand && !commandHasFlag(launchCommand, '--enable-auto-tool-choice')) failures.push('MM3 launch argv missing --enable-auto-tool-choice')
    if (launchCommand && !commandHasFlagValue(launchCommand, '--timeout', '900')) failures.push('MM3 launch argv missing --timeout 900 long-generation default')
    if (launchCommand && commandHasFlag(launchCommand, '--max-tokens')) failures.push('MM3 launch argv incorrectly forced --max-tokens; default must remain model-owned')
  }

  return { status: failures.length ? 'fail' : 'pass', failures }
}

async function main() {
  if (!existsSync(appExe)) throw new Error(`Installed app executable missing: ${appExe}`)
  if (!existsSync(modelPath)) throw new Error(`model path missing: ${modelPath}`)
  mkdirSync(proofDir, { recursive: true })
  const userDataDir = useRealProfile
    ? realUserDataDir
    : mkdtempSync(path.join(tmpdir(), 'vmlx-clean-start-userdata-'))
  const debugPort = await freePort()
  const appArgs = [
    ...(useRealProfile ? [] : [`--user-data-dir=${userDataDir}`]),
    `--remote-debugging-port=${debugPort}`,
  ]
  const appLog = []
  const result = {
    generatedAt: startedAt.toISOString(),
    status: 'running',
    rowName,
    modelPath,
    expectedFamily,
    proofDir,
    outJson,
    shotPath,
    appPath,
    profileMode: useRealProfile ? 'real' : 'temporary',
    userDataDir,
    realUserDataDir,
    appCommand: [appExe, ...appArgs],
    loadPolls: [],
    sourceTrace: {
      sessionIpc: 'panel/src/preload/index.ts:348-360, panel/src/main/ipc/sessions.ts:105-192',
      sessionDefaults: 'panel/src/main/sessions.ts',
      gemmaCachePolicy: 'vmlx_engine/mllm_scheduler.py',
      mm3CachePolicy: 'vmlx_engine/cli.py, vmlx_engine/models/minimax_m3/cache.py',
    },
  }
  if (useRealProfile && existsSync(realUserDataDir)) {
    const backupDir = path.join(proofDir, 'real-user-data-backup')
    cpSync(realUserDataDir, backupDir, { recursive: true, force: true })
    result.realProfileBackupDir = backupDir
  }
  writeResult(result)

  const app = spawn(appExe, appArgs, {
    cwd: tmpdir(),
    env: { ...process.env, VMLX_SKIP_UPDATE_CHECK: '1' },
    detached: true,
    stdio: ['ignore', 'pipe', 'pipe'],
  })
  app.stdout.on('data', (d) => appLog.push(...d.toString().split(/\r?\n/).filter(Boolean)))
  app.stderr.on('data', (d) => appLog.push(...d.toString().split(/\r?\n/).filter(Boolean)))
  result.appPid = app.pid
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

    result.deletePass = await page.evaluate(async () => {
      await window.api.chat.clearAllLocks().catch(() => null)
      const before = await window.api.sessions.list()
      const deleted = []
      for (const session of before || []) {
        await window.api.sessions.stop(session.id).catch(() => null)
        const res = await window.api.sessions.delete(session.id).catch((error) => ({ success: false, error: String(error?.message || error) }))
        deleted.push({ id: session.id, modelName: session.modelName, status: session.status, result: res })
      }
      const after = await window.api.sessions.list()
      return { before, deleted, after }
    })
    result.sessionsBeforeDelete = result.deletePass.before || []
    result.sessionsAfterDelete = result.deletePass.after || []
    writeResult(result)

    const sessionResult = await page.evaluate(async ({ modelPath }) => {
      const created = await window.api.sessions.create(modelPath, {})
      if (!created?.success || !created?.session?.id) {
        throw new Error(`sessions.create({}, default config) failed: ${created?.error || JSON.stringify(created)}`)
      }
      const sessionAfterCreate = await window.api.sessions.get(created.session.id)
      const started = await window.api.sessions.start(created.session.id)
      if (!started?.success) {
        throw new Error(`sessions.start failed: ${started?.error || JSON.stringify(started)}`)
      }
      const sessionAfterStart = await window.api.sessions.get(created.session.id)
      return { created, started, sessionAfterCreate, sessionAfterStart }
    }, { modelPath })
    result.sessionCreateResult = sessionResult.created
    result.sessionStartResult = sessionResult.started
    result.sessionAfterCreate = sessionResult.sessionAfterCreate
    result.sessionAfterStart = sessionResult.sessionAfterStart
    result.sessionConfigAfterCreate = parseConfig(sessionResult.sessionAfterCreate)
    result.sessionConfigAfterStart = parseConfig(sessionResult.sessionAfterStart)
    const sessionId = sessionResult.sessionAfterStart.id
    const port = Number(result.sessionConfigAfterStart.port)
    result.sessionId = sessionId
    result.sessionPort = port
    writeResult(result)

    result.healthReady = await waitForHealth(page, port, sessionId, result)
    result.sessionLogsStart = await page.evaluate(async ({ id }) => window.api.sessions.getLogs(id), { id: sessionId })
    result.launchCommand = extractLaunchCommand(result.sessionLogsStart)
    result.cacheStart = await page.evaluate(async ({ endpoint, id }) => window.api.cache.stats(endpoint, id), { endpoint: { host: '127.0.0.1', port }, id: sessionId }).catch((error) => ({ error: String(error?.message || error) }))
    const modelsRes = await fetch(`http://127.0.0.1:${port}/v1/models`).then((r) => r.json()).catch((error) => ({ error: String(error?.message || error) }))
    result.models = modelsRes
    const servedModel = modelsRes?.data?.[0]?.id || sessionResult.sessionAfterStart.modelName || path.basename(modelPath)
    result.servedModel = servedModel
    result.capabilities = await fetch(`http://127.0.0.1:${port}/v1/models/${encodeURIComponent(servedModel)}/capabilities`).then((r) => r.json()).catch((error) => ({ error: String(error?.message || error) }))
    writeResult(result)

    result.gateway = {}
    result.gateway.status = await page.evaluate(async () => window.api.gateway.getStatus())
    if (result.gateway.status?.running) {
      const gatewayHost = result.gateway.status.host === '0.0.0.0'
        ? '127.0.0.1'
        : (result.gateway.status.host || '127.0.0.1')
      const gatewayPort = Number(result.gateway.status.port || 8080)
      const gatewayBase = `http://${gatewayHost}:${gatewayPort}`
      result.gateway.base = gatewayBase
      result.gateway.health = await fetch(`${gatewayBase}/health`).then((r) => r.json()).catch((error) => ({ error: String(error?.message || error) }))
      result.gateway.models = await fetch(`${gatewayBase}/v1/models`).then((r) => r.json()).catch((error) => ({ error: String(error?.message || error) }))
      result.gateway.capabilities = await fetch(`${gatewayBase}/v1/models/${encodeURIComponent(servedModel)}/capabilities`).then((r) => r.json()).catch((error) => ({ error: String(error?.message || error) }))
      result.gatewayText = await fetchGatewayText(gatewayBase, servedModel)
    }
    writeResult(result)

    result.defaultUiTurn = await sendDefaultUiTurn(page, sessionResult.sessionAfterStart, port, modelPath)
    writeResult(result)

    result.healthEnd = await fetch(`http://127.0.0.1:${port}/health`).then((r) => r.json()).catch((error) => ({ error: String(error?.message || error) }))
    result.cacheEnd = await page.evaluate(async ({ endpoint, id }) => window.api.cache.stats(endpoint, id), { endpoint: { host: '127.0.0.1', port }, id: sessionId }).catch((error) => ({ error: String(error?.message || error) }))
    result.sessionLogsEnd = await page.evaluate(async ({ id }) => window.api.sessions.getLogs(id), { id: sessionId })
    result.launchCommand = result.launchCommand || extractLaunchCommand(result.sessionLogsEnd)
    result.appLogTail = appLog.slice(-200)
    await page.screenshot({ path: shotPath, fullPage: false }).catch(() => null)
    result.screenshot = shotPath
    result.finishedAt = new Date().toISOString()
    Object.assign(result, verdict(result))
    writeResult(result)

    await page.evaluate(async ({ id }) => window.api.sessions.stop(id).catch(() => null), { id: sessionId }).catch(() => null)
    console.log(JSON.stringify({
      status: result.status,
      failures: result.failures,
      rowName,
      outJson,
      shotPath,
      sessionPort: port,
      deletedSessions: result.sessionsBeforeDelete.length,
      sessionConfig: result.sessionConfigAfterStart,
      capabilities: result.capabilities?.modalities,
      nativeCache: result.healthEnd?.native_cache,
      defaultUiTurn: result.defaultUiTurn,
    }, null, 2))
  } catch (error) {
    result.status = 'fail'
    result.failures = [String(error?.stack || error?.message || error)]
    result.appLogTail = appLog.slice(-200)
    if (page) {
      result.bodyText = await page.evaluate(() => document.body.innerText.slice(0, 5000)).catch(() => '')
      await page.screenshot({ path: shotPath, fullPage: false }).catch(() => null)
    }
    writeResult(result)
    console.error(JSON.stringify({ status: 'fail', failures: result.failures, outJson, shotPath }, null, 2))
    process.exitCode = 1
  } finally {
    await browser?.close().catch(() => null)
    if (app.pid) {
      try { process.kill(-app.pid, 'SIGTERM') } catch {}
      await sleep(1500)
      try { process.kill(-app.pid, 'SIGKILL') } catch {}
    }
    if (!useRealProfile && !process.env.VMLX_KEEP_CLEAN_USER_DATA) rmSync(userDataDir, { recursive: true, force: true })
  }
}

main().catch((error) => {
  console.error(error?.stack || error?.message || String(error))
  process.exit(1)
})
