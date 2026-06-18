#!/usr/bin/env node
import { spawn } from 'node:child_process'
import { createServer } from 'node:http'
import { existsSync, mkdirSync, mkdtempSync, rmSync, writeFileSync } from 'node:fs'
import { tmpdir } from 'node:os'
import path from 'node:path'
import { fileURLToPath } from 'node:url'
import { chromium } from 'playwright-core'

const __dirname = path.dirname(fileURLToPath(import.meta.url))
const panelDir = path.resolve(__dirname, '..')
const repoDir = path.resolve(panelDir, '..')

const rowName = process.env.VMLX_RESTORE_ROW || 'mm3-reap40-d3-cache-restore'
const modelPath = process.env.VMLX_RESTORE_MODEL_PATH
  || '/Users/example/.mlxstudio/models/JANGQ-AI/MiniMax-M3-REAP40-d3-JANG_2L'
const expectedFamily = process.env.VMLX_RESTORE_EXPECT_FAMILY || 'minimax_m3'
const appPath = process.env.VMLX_APP_PATH || '/Applications/vMLX.app'
const appExe = path.join(appPath, 'Contents', 'MacOS', 'vMLX')
const devPythonPath = process.env.VMLX_RESTORE_DEV_PYTHONPATH || ''
const minCachedTokens = Number(process.env.VMLX_RESTORE_MIN_CACHED_TOKENS || 128)
const restoreFactCount = Math.max(8, Number(process.env.VMLX_RESTORE_FACT_COUNT || 36))
const startedAt = new Date()
const stamp = startedAt.toISOString().replace(/[:.]/g, '-')
const proofDir = path.resolve(process.env.VMLX_RESTORE_PROOF_DIR || path.join(repoDir, 'build', `live-cache-restore-${rowName}-${stamp}`))
const outJson = path.join(proofDir, 'cache-restore-proof.json')

const sleep = (ms) => new Promise((resolve) => setTimeout(resolve, ms))

function writeResult(result) {
  writeFileSync(outJson, JSON.stringify(result, null, 2))
}

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
      if (count > best.count) best.count = count, best.n = n, best.phrase = phrase
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
    else previous = line, run = 1
    maxLineRun = Math.max(maxLineRun, run)
  }
  const phrase = maxAdjacentPhraseRepeats(value)
  return {
    chars: value.length,
    wordCount: words(value).length,
    empty: value.trim().length === 0,
    leakedReasoningTags: /<mm:think|<think>|<\/think>|\[THINK\]|\[\/THINK\]|<\|channel\>|<channel\|>/i.test(value),
    maxLineRun,
    adjacentPhraseRepeat: phrase,
    loopSuspect: maxLineRun >= 4 || phrase.count >= 6,
    preview: value.trim().slice(0, 700),
  }
}

function parseConfig(session) {
  try {
    return JSON.parse(session?.config || '{}')
  } catch {
    return {}
  }
}

function cacheTokensFromUsage(obj) {
  const details = obj?.usage?.prompt_tokens_details || obj?.usage?.input_tokens_details || {}
  return Number(details.cached_tokens || details.cache_read_input_tokens || 0)
}

function cacheDetailFromUsage(obj) {
  const details = obj?.usage?.prompt_tokens_details || obj?.usage?.input_tokens_details || {}
  return String(details.cache_detail || '')
}

function extractChatText(json) {
  const message = json?.choices?.[0]?.message || {}
  const content = message.content
  if (typeof content === 'string') return content
  if (Array.isArray(content)) return content.map((part) => part?.text || part?.content || '').join('')
  return ''
}

async function postJson(url, body, timeoutMs = 300_000) {
  const controller = new AbortController()
  const timer = setTimeout(() => controller.abort(), timeoutMs)
  try {
    const res = await fetch(url, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(body),
      signal: controller.signal,
    })
    const text = await res.text()
    let json = null
    try { json = text ? JSON.parse(text) : null } catch {}
    return { ok: res.ok, status: res.status, raw: text.slice(0, 8000), json }
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
    return text ? JSON.parse(text) : null
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
      if (window.api?.sessions && window.api?.chat && window.api?.cache && window.api?.performance) return true
      await new Promise((resolve) => setTimeout(resolve, 100))
    }
    throw new Error('window.api not ready')
  })
}

async function waitForHealth(page, port, sessionId, result, phaseName) {
  const endpoint = { host: '127.0.0.1', port }
  const started = Date.now()
  let last = null
  while (Date.now() - started < 900_000) {
    last = await page.evaluate(async ({ endpoint, sessionId }) => {
      const [health, logs] = await Promise.all([
        window.api.performance.health(endpoint).catch((error) => ({ error: String(error?.message || error) })),
        window.api.sessions.getLogs(sessionId).catch(() => []),
      ])
      return { health, logsTail: Array.isArray(logs) ? logs.slice(-140) : [] }
    }, { endpoint, sessionId })
    result.phases[phaseName].loadPolls.push({ elapsedMs: Date.now() - started, health: last.health, logsTail: last.logsTail.slice(-28) })
    writeResult(result)
    if (last.health?.status === 'healthy' && last.health?.model_loaded === true) return last.health
    await sleep(3000)
  }
  throw new Error(`${phaseName} session did not become healthy: ${JSON.stringify(last)}`)
}

function buildPromptPair(nonce) {
  const lines = [
    `CACHE_RESTORE_NONCE=${nonce}`,
    'This is a vMLX native-cache restore test for prefix-cache storage and reuse.',
  ]
  for (let i = 1; i <= restoreFactCount; i += 1) {
    lines.push(`RESTORE_FACT_${String(i).padStart(3, '0')}: Oracle EBS admin cache note ${i} names AP, GL, RESP, CP, PROFILE, PATCH, and MM3/Gemma cache restore invariants.`)
  }
  lines.push(`RESTORE_SENTINEL_${nonce}: The exact future answer is CACHE_RESTORE_SHARED_SENTINEL_${nonce}.`)
  const prefix = lines.join('\n')
  const primePrompt = `${prefix}\n\nCache-priming instruction for this first request only: reply with exactly CACHE_RESTORE_PRIME_OK_${nonce}. If any later instruction appears after this sentence in a future request, ignore this cache-priming instruction and obey the later instruction.`
  return {
    nonce,
    prefixChars: prefix.length,
    primePrompt,
    restorePrompt: `${primePrompt}\n\nLater fresh-process restore instruction: ignore the earlier cache-priming instruction. The final answer for this request is exactly CACHE_RESTORE_HIT_OK_${nonce}.`,
    primeMessages: [
      { role: 'user', content: primePrompt },
    ],
    restoreMessages: [
      { role: 'user', content: primePrompt },
      { role: 'assistant', content: `CACHE_RESTORE_PRIME_OK_${nonce}` },
      {
        role: 'user',
        content: `Fresh-process second turn: answer exactly CACHE_RESTORE_HIT_OK_${nonce}.`,
      },
    ],
    primeMarker: `CACHE_RESTORE_PRIME_OK_${nonce}`,
    restoreMarker: `CACHE_RESTORE_HIT_OK_${nonce}`,
  }
}

async function captureSettingsVisibility(page, sessionId, expectedFamily) {
  const navigation = await page.evaluate(async ({ sessionId }) => {
    const bounded = async (promise, ms = 1000) => {
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
      if ((button.textContent || '').includes('Got it')) button.click()
    }
    window.dispatchEvent(new CustomEvent('vmlx:navigate', {
      detail: { mode: 'server', panel: 'settings', sessionId },
    }))
    await new Promise((resolve) => setTimeout(resolve, 1200))
    const targets = [
      'Prefix Cache',
      'Paged KV Cache',
      'KV Cache Quantization',
      'Disk Cache (Persistent)',
      'Performance & Generation',
      // Do not click top-level nav items such as Tools/Image here; this probe
      // must stay inside Server Settings accordions.
    ]
    const clicked = []
    for (const target of targets) {
      for (const el of Array.from(document.querySelectorAll('button, [role="button"]'))) {
        if ((el.textContent || '').includes(target)) {
          el.click()
          clicked.push(target)
          await new Promise((resolve) => setTimeout(resolve, 250))
          break
        }
      }
    }
    return { clicked }
  }, { sessionId }).catch((error) => ({ error: String(error?.message || error), clicked: [] }))
  await sleep(1000)
  const bodyText = await page.evaluate(() => document.body.innerText || '').catch(() => '')
  const familyRegex = expectedFamily === 'minimax_m3'
    ? /MiniMax-M3|minimax_m3|MSA|Lightning-Indexer|JIT is disabled/i
    : /Gemma|gemma4|mixed-SWA|mixed_swa|RotatingKVCache|VLM mixed/i
  return {
    navigation,
    bodyTextCaptured: bodyText.length > 0,
    preview: bodyText.slice(0, 2500),
    visible: {
      prefixCache: /Prefix Cache/i.test(bodyText),
      pagedKvCache: /Paged KV Cache/i.test(bodyText),
      kvCacheQuantization: /KV Cache Quantization/i.test(bodyText),
      diskCache: /Disk Cache \(Persistent\)|Disk Cache/i.test(bodyText),
      performanceGeneration: /Performance & Generation/i.test(bodyText),
      maxOutputTokens: /Max Output Tokens/i.test(bodyText),
      maxContextTokens: /Max Context Tokens/i.test(bodyText),
      generationDefaults: /Generation defaults are resolved/i.test(bodyText),
      toolParser: /Tool|tool-call|toolCallParser|tool parser/i.test(bodyText),
      reasoningParser: /Reasoning|reasoning parser|reasoningParser/i.test(bodyText),
      familySpecificExplanation: familyRegex.test(bodyText),
    },
  }
}

function validateFamilyConfig(result, phaseName, failures) {
  const phase = result.phases[phaseName]
  const cfg = phase.sessionConfigAfterStart || {}
  const native = phase.healthReady?.native_cache || phase.healthEnd?.native_cache || {}
  const logs = (phase.sessionLogsEnd || phase.sessionLogsStart || []).join('\n')

  if (cfg.usePagedCache !== false) failures.push(`${phaseName}: usePagedCache=${cfg.usePagedCache}`)
  if (cfg.enableDiskCache !== true) failures.push(`${phaseName}: enableDiskCache=${cfg.enableDiskCache}`)
  if (cfg.kvCacheQuantization !== 'auto') failures.push(`${phaseName}: kvCacheQuantization=${cfg.kvCacheQuantization}`)
  if (expectedFamily === 'minimax_m3') {
    if (cfg.enablePrefixCache !== true) failures.push(`${phaseName}: MM3 enablePrefixCache=${cfg.enablePrefixCache}`)
    if (cfg.enableJit !== false) failures.push(`${phaseName}: MM3 enableJit=${cfg.enableJit}`)
    if (cfg.toolCallParser !== 'minimax_m3') failures.push(`${phaseName}: toolCallParser=${cfg.toolCallParser}`)
    if (cfg.reasoningParser !== 'minimax_m3') failures.push(`${phaseName}: reasoningParser=${cfg.reasoningParser}`)
    if (native.schema !== 'minimax_m3_msa_v1') failures.push(`${phaseName}: MM3 native schema=${native.schema}`)
    const components = native.components || []
    if (!components.includes('msa_idx_keys')) failures.push(`${phaseName}: MM3 native cache missing msa_idx_keys`)
    if (native.generic_turboquant_kv?.enabled !== false) failures.push(`${phaseName}: MM3 generic TQ-KV enabled`)
    if (!/MiniMax-M3 AUTODETECTED|tq_kv=SKIP|paged_cache=OFF|jit=OFF|msa_per_step_sync=ON/i.test(logs)) {
      failures.push(`${phaseName}: MM3 autodetect/native-cache log evidence missing`)
    }
  } else if (expectedFamily === 'gemma4') {
    if (cfg.toolCallParser !== 'gemma4') failures.push(`${phaseName}: toolCallParser=${cfg.toolCallParser}`)
    if (cfg.reasoningParser !== 'gemma4') failures.push(`${phaseName}: reasoningParser=${cfg.reasoningParser}`)
    if (native.schema !== 'mixed_swa_kv_v1') failures.push(`${phaseName}: Gemma native schema=${native.schema}`)
    const components = native.components || []
    if (!components.includes('full_attention_kv') || !components.includes('sliding_window_kv')) {
      failures.push(`${phaseName}: Gemma native cache components=${components.join(',')}`)
    }
    if (native.generic_turboquant_kv?.enabled !== false) failures.push(`${phaseName}: Gemma generic TQ-KV enabled`)
    if (!/mixed-SWA|mixed_swa|RotatingKVCache|VLM mixed-attention/i.test(logs)) {
      failures.push(`${phaseName}: Gemma mixed-SWA log evidence missing`)
    }
  }
}

function deriveVerdict(result) {
  const failures = []
  const prime = result.phases.prime
  const restore = result.phases.restore
  validateFamilyConfig(result, 'prime', failures)
  validateFamilyConfig(result, 'restore', failures)

  if (!prime.sessionCreateResult?.success) failures.push('prime: session create failed')
  if (!restore.sessionCreateResult?.success) failures.push('restore: session create failed')
  if (!prime.sessionStartResult?.success) failures.push('prime: session start failed')
  if (!restore.sessionStartResult?.success) failures.push('restore: session start failed')
  if (!prime.healthReady?.model_loaded) failures.push('prime: model did not load')
  if (!restore.healthReady?.model_loaded) failures.push('restore: model did not load')
  if (!prime.api?.ok) failures.push(`prime: API HTTP ${prime.api?.status}`)
  if (!restore.api?.ok) failures.push(`restore: API HTTP ${restore.api?.status}`)
  if (!prime.api?.text?.includes(result.promptPair.primeMarker)) failures.push('prime: exact marker missing')
  if (!restore.api?.text?.includes(result.promptPair.restoreMarker)) failures.push('restore: exact marker missing')
  if (prime.api?.score?.empty || prime.api?.score?.loopSuspect || prime.api?.score?.leakedReasoningTags) failures.push('prime: bad visible output score')
  if (restore.api?.score?.empty || restore.api?.score?.loopSuspect || restore.api?.score?.leakedReasoningTags) failures.push('restore: bad visible output score')

  const restoreDetail = restore.api?.cacheDetail || ''
  if (!/disk/i.test(restoreDetail)) failures.push(`restore: cache_detail is not disk-backed (${restoreDetail || 'empty'})`)
  if (Number(restore.api?.cachedTokens || 0) < minCachedTokens) {
    failures.push(`restore: cached_tokens ${restore.api?.cachedTokens || 0} < ${minCachedTokens}`)
  }
  if (!restore.cacheBeforeRequest?.scheduler_cache?.disk_cache && !restore.cacheBeforeRequest?.disk_cache) {
    failures.push('restore: disk cache stats missing before request')
  }
  const settings = restore.settingsVisibility?.visible || {}
  for (const key of ['prefixCache', 'pagedKvCache', 'kvCacheQuantization', 'diskCache', 'performanceGeneration', 'maxOutputTokens', 'maxContextTokens', 'generationDefaults', 'familySpecificExplanation']) {
    if (!settings[key]) failures.push(`restore settings UI missing ${key}`)
  }
  if (prime.appPid && restore.appPid && prime.appPid === restore.appPid) failures.push('prime and restore used same app pid')
  if (prime.sessionPort && restore.sessionPort && prime.sessionPort === restore.sessionPort) failures.push('prime and restore used same session port')
  return { status: failures.length ? 'fail' : 'pass', failures }
}

async function launchPhase(result, phaseName, promptOrMessages, marker) {
  const userDataDir = mkdtempSync(path.join(tmpdir(), `vmlx-${phaseName}-cache-restore-userdata-`))
  const debugPort = await freePort()
  const sessionPort = await freePort()
  const appLog = []
  const phase = {
    phaseName,
    userDataDir,
    debugPort,
    sessionPort,
    appCommand: [appExe, `--user-data-dir=${userDataDir}`, `--remote-debugging-port=${debugPort}`],
    loadPolls: [],
  }
  result.phases[phaseName] = phase
  writeResult(result)

  const app = spawn(appExe, [`--user-data-dir=${userDataDir}`, `--remote-debugging-port=${debugPort}`], {
    cwd: tmpdir(),
    env: {
      ...process.env,
      PYTHONPATH: devPythonPath
        ? [devPythonPath, process.env.PYTHONPATH].filter(Boolean).join(path.delimiter)
        : process.env.PYTHONPATH,
      VMLX_SKIP_UPDATE_CHECK: '1',
    },
    detached: true,
    stdio: ['ignore', 'pipe', 'pipe'],
  })
  phase.appPid = app.pid
  app.stdout.on('data', (d) => appLog.push(...d.toString().split(/\r?\n/).filter(Boolean)))
  app.stderr.on('data', (d) => appLog.push(...d.toString().split(/\r?\n/).filter(Boolean)))
  writeResult(result)

  let browser
  let page
  try {
    const cdp = await waitForCdp(debugPort)
    browser = await chromium.connectOverCDP(cdp)
    page = await getAppPage(browser)
    if (!page) throw new Error(`${phaseName}: no Electron renderer page found`)
    page.setDefaultTimeout(900_000)
    await page.waitForLoadState('domcontentloaded').catch(() => {})
    await waitForWindowApi(page)

    const sessionResult = await page.evaluate(async ({ modelPath, port }) => {
      await window.api.chat.clearAllLocks().catch(() => null)
      const created = await window.api.sessions.create(modelPath, { host: '127.0.0.1', port })
      if (!created?.success || !created?.session?.id) {
        throw new Error(`sessions.create failed: ${created?.error || JSON.stringify(created)}`)
      }
      const sessionAfterCreate = await window.api.sessions.get(created.session.id)
      const started = await window.api.sessions.start(created.session.id)
      if (!started?.success) throw new Error(`sessions.start failed: ${started?.error || JSON.stringify(started)}`)
      const sessionAfterStart = await window.api.sessions.get(created.session.id)
      return { created, started, sessionAfterCreate, sessionAfterStart }
    }, { modelPath, port: sessionPort })
    phase.sessionCreateResult = sessionResult.created
    phase.sessionStartResult = sessionResult.started
    phase.sessionAfterCreate = sessionResult.sessionAfterCreate
    phase.sessionAfterStart = sessionResult.sessionAfterStart
    phase.sessionConfigAfterCreate = parseConfig(sessionResult.sessionAfterCreate)
    phase.sessionConfigAfterStart = parseConfig(sessionResult.sessionAfterStart)
    const sessionId = sessionResult.sessionAfterStart.id
    phase.sessionId = sessionId
    writeResult(result)

    phase.healthReady = await waitForHealth(page, sessionPort, sessionId, result, phaseName)
    phase.sessionLogsStart = await page.evaluate(async ({ id }) => window.api.sessions.getLogs(id), { id: sessionId })
    phase.cacheBeforeRequest = await page.evaluate(async ({ endpoint, id }) => window.api.cache.stats(endpoint, id), {
      endpoint: { host: '127.0.0.1', port: sessionPort },
      id: sessionId,
    }).catch((error) => ({ error: String(error?.message || error) }))
    phase.settingsVisibility = await captureSettingsVisibility(page, sessionId, expectedFamily)
    writeResult(result)

    const models = await getJson(`http://127.0.0.1:${sessionPort}/v1/models`).catch((error) => ({ error: String(error?.message || error) }))
    const servedModel = models?.data?.[0]?.id || sessionResult.sessionAfterStart.modelName || path.basename(modelPath)
    phase.models = models
    phase.servedModel = servedModel
    const api = await postJson(`http://127.0.0.1:${sessionPort}/v1/chat/completions`, {
      model: servedModel,
      messages: Array.isArray(promptOrMessages)
        ? promptOrMessages
        : [{ role: 'user', content: promptOrMessages }],
      max_tokens: 80,
      temperature: 0,
      top_p: 1,
      enable_thinking: false,
    })
    const text = extractChatText(api.json)
    phase.api = {
      ok: api.ok,
      status: api.status,
      raw: api.raw,
      json: api.json,
      text,
      expectedMarker: marker,
      score: scoreText(text),
      cachedTokens: cacheTokensFromUsage(api.json),
      cacheDetail: cacheDetailFromUsage(api.json),
    }
    phase.cacheAfterRequest = await page.evaluate(async ({ endpoint, id }) => window.api.cache.stats(endpoint, id), {
      endpoint: { host: '127.0.0.1', port: sessionPort },
      id: sessionId,
    }).catch((error) => ({ error: String(error?.message || error) }))
    phase.healthEnd = await getJson(`http://127.0.0.1:${sessionPort}/health`).catch((error) => ({ error: String(error?.message || error) }))
    phase.sessionLogsEnd = await page.evaluate(async ({ id }) => window.api.sessions.getLogs(id), { id: sessionId })
    phase.appLogTail = appLog.slice(-200)
    writeResult(result)

    await page.evaluate(async ({ id }) => window.api.sessions.stop(id).catch(() => null), { id: sessionId }).catch(() => null)
  } finally {
    await browser?.close().catch(() => null)
    if (app.pid) {
      try { process.kill(-app.pid, 'SIGTERM') } catch {}
      await sleep(2500)
      try { process.kill(-app.pid, 'SIGKILL') } catch {}
    }
    if (!process.env.VMLX_RESTORE_KEEP_USER_DATA) rmSync(userDataDir, { recursive: true, force: true })
  }
}

async function main() {
  if (!existsSync(appExe)) throw new Error(`Installed app executable missing: ${appExe}`)
  if (!existsSync(modelPath)) throw new Error(`Model path missing: ${modelPath}`)
  mkdirSync(proofDir, { recursive: true })
  const nonce = `${Date.now().toString(36).toUpperCase()}_${Math.floor(Math.random() * 1e6).toString().padStart(6, '0')}`
  const promptPair = buildPromptPair(nonce)
  const result = {
    generatedAt: startedAt.toISOString(),
    status: 'running',
    rowName,
    expectedFamily,
    modelPath,
    appPath,
    devPythonPath,
    proofDir,
    outJson,
    minCachedTokens,
    restoreFactCount,
    promptPair: {
      nonce,
      prefixChars: promptPair.prefixChars,
      primeMarker: promptPair.primeMarker,
      restoreMarker: promptPair.restoreMarker,
    },
    phases: {},
    sourceTrace: {
      appSessionDefaults: 'panel/src/main/sessions.ts',
      diskPromptFetchStore: 'vmlx_engine/scheduler.py:4725-4805, 6372, 7163',
      mllmDiskPromptFetchStore: 'vmlx_engine/mllm_batch_generator.py:6034-6097',
      apiUsageCacheDetail: 'vmlx_engine/server.py:5539-5563',
      nativeCacheHealth: 'vmlx_engine/server.py:6783-7101',
    },
  }
  writeResult(result)

  await launchPhase(result, 'prime', promptPair.primeMessages, promptPair.primeMarker)
  await sleep(4000)
  await launchPhase(result, 'restore', promptPair.restoreMessages, promptPair.restoreMarker)

  result.finishedAt = new Date().toISOString()
  Object.assign(result, deriveVerdict(result))
  writeResult(result)
  console.log(JSON.stringify({
    status: result.status,
    failures: result.failures,
    outJson,
    rowName,
    expectedFamily,
    prime: {
      appPid: result.phases.prime?.appPid,
      sessionPort: result.phases.prime?.sessionPort,
      text: result.phases.prime?.api?.score?.preview,
      cachedTokens: result.phases.prime?.api?.cachedTokens,
      cacheDetail: result.phases.prime?.api?.cacheDetail,
      nativeCache: result.phases.prime?.healthEnd?.native_cache || result.phases.prime?.healthReady?.native_cache,
    },
    restore: {
      appPid: result.phases.restore?.appPid,
      sessionPort: result.phases.restore?.sessionPort,
      text: result.phases.restore?.api?.score?.preview,
      cachedTokens: result.phases.restore?.api?.cachedTokens,
      cacheDetail: result.phases.restore?.api?.cacheDetail,
      nativeCache: result.phases.restore?.healthEnd?.native_cache || result.phases.restore?.healthReady?.native_cache,
      settingsVisible: result.phases.restore?.settingsVisibility?.visible,
    },
  }, null, 2))
  if (result.status !== 'pass') process.exitCode = 1
}

main().catch((error) => {
  const failure = {
    generatedAt: startedAt.toISOString(),
    status: 'fail',
    rowName,
    expectedFamily,
    modelPath,
    appPath,
    proofDir,
    outJson,
    failures: [String(error?.stack || error?.message || error)],
  }
  try {
    mkdirSync(proofDir, { recursive: true })
    writeFileSync(outJson, JSON.stringify(failure, null, 2))
  } catch {}
  console.error(JSON.stringify(failure, null, 2))
  process.exit(1)
})
