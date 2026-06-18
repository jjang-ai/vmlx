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

const rowName = process.env.VMLX_LIFECYCLE_ROW || 'gemma4-e2b-mxfp4-lifecycle'
const modelPath = process.env.VMLX_LIFECYCLE_MODEL_PATH || '/Users/example/models/OsaurusAI--gemma-4-E2B-it-qat-MXFP4'
const appPath = process.env.VMLX_APP_PATH || '/Applications/vMLX.app'
const appExe = path.join(appPath, 'Contents', 'MacOS', 'vMLX')
const startedAt = new Date()
const stamp = startedAt.toISOString().replace(/[:.]/g, '-')
const proofDir = path.resolve(process.env.VMLX_LIFECYCLE_PROOF_DIR || path.join(repoDir, 'build', `live-lifecycle-${rowName}-${stamp}`))
const outJson = path.join(proofDir, 'lifecycle-proof.json')
const shotPath = path.join(proofDir, 'lifecycle-final.png')

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

async function waitForWindowApi(page) {
  await page.evaluate(async () => {
    const started = Date.now()
    while (Date.now() - started < 45_000) {
      if (window.api?.sessions && window.api?.chat && window.api?.performance) return true
      await new Promise((resolve) => setTimeout(resolve, 100))
    }
    throw new Error('window.api not ready')
  })
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
      return { health, logsTail: Array.isArray(logs) ? logs.slice(-80) : [] }
    }, { endpoint, sessionId })
    result.loadPolls.push({ elapsedMs: Date.now() - started, health: last.health, logsTail: last.logsTail.slice(-20) })
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

async function waitUntilNotStreaming(page, chatId, timeoutMs = 45_000) {
  const started = Date.now()
  let streaming = false
  while (Date.now() - started < timeoutMs) {
    streaming = await page.evaluate(async ({ chatId }) => window.api.chat.isStreaming(chatId), { chatId }).catch(() => false)
    if (!streaming) return { streaming: false, elapsedMs: Date.now() - started }
    await sleep(500)
  }
  return { streaming, elapsedMs: Date.now() - started }
}

async function quietMessageCounts(page, chatId, delayMs = 4000) {
  const before = await page.evaluate(async ({ chatId }) => window.api.chat.getMessages(chatId), { chatId })
  await sleep(delayMs)
  const after = await page.evaluate(async ({ chatId }) => window.api.chat.getMessages(chatId), { chatId })
  const assistantBefore = before.filter((m) => m.role === 'assistant').length
  const assistantAfter = after.filter((m) => m.role === 'assistant').length
  return {
    beforeCount: before.length,
    afterCount: after.length,
    assistantBefore,
    assistantAfter,
    changed: after.length !== before.length || assistantAfter !== assistantBefore,
  }
}

async function runAbortTurn(page, port, modelPath) {
  const endpoint = { host: '127.0.0.1', port }
  const chat = await page.evaluate(async ({ modelPath }) => {
    const chat = await window.api.chat.create('Lifecycle abort proof', modelPath.split('/').pop(), undefined, modelPath)
    await window.api.chat.setOverrides(chat.id, {
      maxTokens: 1200,
      enableThinking: false,
      builtinToolsEnabled: false,
    })
    return chat
  }, { modelPath })
  const prompt = 'Lifecycle abort proof. Write a very long numbered reference list about Oracle EBS modules. Keep writing until stopped.'
  const sendPromise = page.evaluate(async ({ chatId, prompt, endpoint }) => {
    try {
      const value = await window.api.chat.sendMessage(chatId, prompt, endpoint)
      return { resolved: true, value }
    } catch (error) {
      return { resolved: false, error: String(error?.message || error) }
    }
  }, { chatId: chat.id, prompt, endpoint })
  await sleep(2500)
  const streamingBeforeAbort = await page.evaluate(async ({ chatId }) => window.api.chat.isStreaming(chatId), { chatId: chat.id }).catch(() => false)
  const abortResult = await page.evaluate(async ({ chatId }) => window.api.chat.abort(chatId), { chatId: chat.id })
  const notStreaming = await waitUntilNotStreaming(page, chat.id)
  const sendResult = await sendPromise
  const quiet = await quietMessageCounts(page, chat.id)
  const messages = await page.evaluate(async ({ chatId }) => window.api.chat.getMessages(chatId), { chatId: chat.id })
  return {
    chatId: chat.id,
    streamingBeforeAbort,
    abortResult,
    notStreaming,
    sendResult,
    quiet,
    messageCount: messages.length,
    assistantCount: messages.filter((m) => m.role === 'assistant').length,
    assistantTail: messages.filter((m) => m.role === 'assistant').slice(-2),
  }
}

async function runStopTurn(page, sessionId, port, modelPath) {
  const endpoint = { host: '127.0.0.1', port }
  const chat = await page.evaluate(async ({ modelPath }) => {
    const chat = await window.api.chat.create('Lifecycle session stop proof', modelPath.split('/').pop(), undefined, modelPath)
    await window.api.chat.setOverrides(chat.id, {
      maxTokens: 1200,
      enableThinking: false,
      builtinToolsEnabled: false,
    })
    return chat
  }, { modelPath })
  const prompt = 'Lifecycle session-stop proof. Write a long numbered technical checklist about Oracle EBS patching and keep going until stopped.'
  const sendPromise = page.evaluate(async ({ chatId, prompt, endpoint }) => {
    try {
      const value = await window.api.chat.sendMessage(chatId, prompt, endpoint)
      return { resolved: true, value }
    } catch (error) {
      return { resolved: false, error: String(error?.message || error) }
    }
  }, { chatId: chat.id, prompt, endpoint })
  await sleep(2500)
  const streamingBeforeStop = await page.evaluate(async ({ chatId }) => window.api.chat.isStreaming(chatId), { chatId: chat.id }).catch(() => false)
  const stopResult = await page.evaluate(async ({ sessionId }) => window.api.sessions.stop(sessionId), { sessionId })
  const notStreaming = await waitUntilNotStreaming(page, chat.id)
  const sendResult = await sendPromise
  await sleep(3500)
  const sessionAfterStop = await page.evaluate(async ({ sessionId }) => window.api.sessions.get(sessionId), { sessionId }).catch((error) => ({ error: String(error?.message || error) }))
  const healthAfterStop = await fetch(`http://127.0.0.1:${port}/health`, { signal: AbortSignal.timeout(2000) })
    .then(async (r) => ({ ok: r.ok, status: r.status, text: (await r.text()).slice(0, 300) }))
    .catch((error) => ({ ok: false, error: String(error?.message || error) }))
  const quiet = await quietMessageCounts(page, chat.id)
  const messages = await page.evaluate(async ({ chatId }) => window.api.chat.getMessages(chatId), { chatId: chat.id })
  return {
    chatId: chat.id,
    streamingBeforeStop,
    stopResult,
    notStreaming,
    sendResult,
    sessionAfterStop,
    healthAfterStop,
    quiet,
    messageCount: messages.length,
    assistantCount: messages.filter((m) => m.role === 'assistant').length,
    assistantTail: messages.filter((m) => m.role === 'assistant').slice(-2),
  }
}

function verdict(result) {
  const failures = []
  if (!result.sessionStartResult?.success) failures.push('session start failed')
  if (!result.healthReady?.model_loaded) failures.push('model did not load')
  if (!result.abortTurn?.streamingBeforeAbort) failures.push('abort turn was not streaming before abort')
  if (!result.abortTurn?.abortResult?.success) failures.push(`chat abort failed: ${JSON.stringify(result.abortTurn?.abortResult)}`)
  if (result.abortTurn?.notStreaming?.streaming) failures.push('chat abort left streaming lock active')
  if (result.abortTurn?.quiet?.changed) failures.push('chat abort produced extra messages after quiet wait')
  if (!result.stopTurn?.streamingBeforeStop) failures.push('stop turn was not streaming before session stop')
  if (!result.stopTurn?.stopResult?.success) failures.push(`session stop failed: ${JSON.stringify(result.stopTurn?.stopResult)}`)
  if (result.stopTurn?.notStreaming?.streaming) failures.push('session stop left streaming lock active')
  if (result.stopTurn?.healthAfterStop?.ok) failures.push('session health still reachable after stop')
  if (!['stopped', 'error'].includes(result.stopTurn?.sessionAfterStop?.status)) failures.push(`session status after stop=${result.stopTurn?.sessionAfterStop?.status}`)
  if (result.stopTurn?.quiet?.changed) failures.push('session stop produced extra messages after quiet wait')
  return { status: failures.length ? 'fail' : 'pass', failures }
}

async function main() {
  if (!existsSync(appExe)) throw new Error(`Installed app executable missing: ${appExe}`)
  if (!existsSync(modelPath)) throw new Error(`model path missing: ${modelPath}`)
  mkdirSync(proofDir, { recursive: true })
  const userDataDir = mkdtempSync(path.join(tmpdir(), 'vmlx-lifecycle-userdata-'))
  const debugPort = await freePort()
  const appArgs = [`--user-data-dir=${userDataDir}`, `--remote-debugging-port=${debugPort}`]
  const appLog = []
  const result = {
    generatedAt: startedAt.toISOString(),
    status: 'running',
    rowName,
    modelPath,
    proofDir,
    outJson,
    shotPath,
    appPath,
    userDataDir,
    appCommand: [appExe, ...appArgs],
    loadPolls: [],
    sourceTrace: {
      chatAbortIpc: 'panel/src/preload/index.ts:134, panel/src/main/ipc/chat.ts:4089-4134',
      abortByEndpoint: 'panel/src/main/ipc/chat.ts:633-667',
      sessionStopAbort: 'panel/src/main/sessions.ts:2647, panel/src/main/sessions.ts:2350-2355',
      serverCancelEndpoints: 'vmlx_engine/server.py:8317-8383',
      schedulerAbort: 'vmlx_engine/mllm_scheduler.py:2122-2149, vmlx_engine/scheduler.py:4905-4935',
    },
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
    browser = await chromium.connectOverCDP(await waitForCdp(debugPort))
    page = await getAppPage(browser)
    if (!page) throw new Error('No Electron renderer page found')
    page.setDefaultTimeout(900_000)
    await page.waitForLoadState('domcontentloaded').catch(() => {})
    await waitForWindowApi(page)

    const sessionResult = await page.evaluate(async ({ modelPath }) => {
      const created = await window.api.sessions.create(modelPath, {})
      if (!created?.success || !created?.session?.id) throw new Error(`create failed: ${created?.error || JSON.stringify(created)}`)
      const started = await window.api.sessions.start(created.session.id)
      if (!started?.success) throw new Error(`start failed: ${started?.error || JSON.stringify(started)}`)
      const sessionAfterStart = await window.api.sessions.get(created.session.id)
      return { created, started, sessionAfterStart }
    }, { modelPath })
    result.sessionCreateResult = sessionResult.created
    result.sessionStartResult = sessionResult.started
    result.sessionAfterStart = sessionResult.sessionAfterStart
    result.sessionConfigAfterStart = parseConfig(sessionResult.sessionAfterStart)
    result.sessionId = sessionResult.sessionAfterStart.id
    result.sessionPort = Number(result.sessionConfigAfterStart.port)
    writeResult(result)

    result.healthReady = await waitForHealth(page, result.sessionPort, result.sessionId, result)
    result.abortTurn = await runAbortTurn(page, result.sessionPort, modelPath)
    writeResult(result)
    result.stopTurn = await runStopTurn(page, result.sessionId, result.sessionPort, modelPath)
    result.sessionLogsEnd = await page.evaluate(async ({ id }) => window.api.sessions.getLogs(id), { id: result.sessionId }).catch(() => [])
    result.appLogTail = appLog.slice(-200)
    await page.screenshot({ path: shotPath, fullPage: false }).catch(() => null)
    result.screenshot = shotPath
    result.finishedAt = new Date().toISOString()
    Object.assign(result, verdict(result))
    writeResult(result)
    console.log(JSON.stringify({
      status: result.status,
      failures: result.failures,
      rowName,
      outJson,
      shotPath,
      sessionPort: result.sessionPort,
      abortTurn: {
        streamingBeforeAbort: result.abortTurn.streamingBeforeAbort,
        abortResult: result.abortTurn.abortResult,
        notStreaming: result.abortTurn.notStreaming,
        quiet: result.abortTurn.quiet,
      },
      stopTurn: {
        streamingBeforeStop: result.stopTurn.streamingBeforeStop,
        stopResult: result.stopTurn.stopResult,
        notStreaming: result.stopTurn.notStreaming,
        sessionAfterStop: result.stopTurn.sessionAfterStop,
        healthAfterStop: result.stopTurn.healthAfterStop,
        quiet: result.stopTurn.quiet,
      },
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
    if (!process.env.VMLX_KEEP_LIFECYCLE_USER_DATA) rmSync(userDataDir, { recursive: true, force: true })
  }
}

main().catch((error) => {
  console.error(error?.stack || error?.message || String(error))
  process.exit(1)
})
