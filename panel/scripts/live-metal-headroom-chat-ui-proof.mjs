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
const startedAt = new Date()
const stamp = startedAt.toISOString().replace(/[:.]/g, '-')
const proofDir = path.resolve(
  process.env.VMLX_METAL_CHAT_UI_PROOF_DIR ||
  path.join(repoDir, 'build', `live-metal-headroom-chat-ui-${stamp}`),
)
const outJson = path.join(proofDir, 'metal-headroom-chat-ui-proof.json')
const shotPath = path.join(proofDir, 'metal-headroom-chat-ui-proof.png')
const installedAppPath = process.env.VMLX_APP_PATH || process.env.VMLINUX_REAL_UI_APP_PATH || ''
const headroomDetail = 'Requested max output tokens exceed projected safe Metal headroom: requested=8192, safe_cap=1. Reduce max_tokens/max_output_tokens, context length, paged cache blocks, or load a smaller model. Set VMLX_METAL_PROJECTED_OUTPUT_GUARD=0 only for explicit developer diagnostics; disabling it accepts Metal OOM / kernel-panic risk.'

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

function collectBody(req) {
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

async function startMockServer() {
  const requests = []
  const port = await freePort()
  const server = createServer(async (req, res) => {
    try {
      if (req.method === 'GET' && req.url === '/v1/models') {
        res.writeHead(200, { 'content-type': 'application/json' })
        res.end(JSON.stringify({ object: 'list', data: [{ id: 'vmlx-metal-headroom-mock', object: 'model' }] }))
        return
      }
      if (req.method === 'POST' && (req.url === '/v1/chat/completions' || req.url === '/v1/responses')) {
        const body = await collectBody(req)
        requests.push({ url: req.url, body })
        res.writeHead(413, { 'content-type': 'application/json' })
        res.end(JSON.stringify({ detail: headroomDetail }))
        return
      }
      if (req.method === 'GET' && req.url === '/health') {
        res.writeHead(200, { 'content-type': 'application/json' })
        res.end(JSON.stringify({ status: 'ok', model_name: 'vmlx-metal-headroom-mock' }))
        return
      }
      res.writeHead(404, { 'content-type': 'application/json' })
      res.end(JSON.stringify({ error: `unhandled ${req.method} ${req.url}` }))
    } catch (error) {
      res.writeHead(500, { 'content-type': 'application/json' })
      res.end(JSON.stringify({ error: String(error?.message || error) }))
    }
  })
  await new Promise((resolve, reject) => {
    server.listen(port, '127.0.0.1', resolve)
    server.on('error', reject)
  })
  return { server, port, requests }
}

async function waitForCdp(debugPort) {
  const endpoint = `http://127.0.0.1:${debugPort}`
  const started = Date.now()
  while (Date.now() - started < 90_000) {
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
    while (Date.now() - started < 60_000) {
      if (window.api?.sessions && window.api?.chat) return true
      await new Promise((resolve) => setTimeout(resolve, 100))
    }
    throw new Error('window.api chat/sessions not ready')
  })
}

function startApp(userDataDir, debugPort) {
  const args = [`--user-data-dir=${userDataDir}`, `--remote-debugging-port=${debugPort}`]
  const logs = []
  if (installedAppPath) {
    const exe = path.join(installedAppPath, 'Contents', 'MacOS', 'vMLX')
    if (!existsSync(exe)) throw new Error(`App executable missing: ${exe}`)
    const proc = spawn(exe, args, {
      cwd: tmpdir(),
      env: { ...process.env, VMLX_SKIP_UPDATE_CHECK: '1' },
      detached: true,
      stdio: ['ignore', 'pipe', 'pipe'],
    })
    proc.stdout.on('data', (d) => logs.push(...d.toString().split(/\r?\n/).filter(Boolean)))
    proc.stderr.on('data', (d) => logs.push(...d.toString().split(/\r?\n/).filter(Boolean)))
    return { proc, logs, command: [exe, ...args], mode: 'installed-app' }
  }
  const npmArgs = ['run', 'dev', '--', '--', ...args]
  const proc = spawn('npm', npmArgs, {
    cwd: panelDir,
    env: { ...process.env, VMLX_SKIP_UPDATE_CHECK: '1' },
    detached: true,
    stdio: ['ignore', 'pipe', 'pipe'],
  })
  proc.stdout.on('data', (d) => logs.push(...d.toString().split(/\r?\n/).filter(Boolean)))
  proc.stderr.on('data', (d) => logs.push(...d.toString().split(/\r?\n/).filter(Boolean)))
  return { proc, logs, command: ['npm', ...npmArgs], mode: 'electron-dev' }
}

async function main() {
  mkdirSync(proofDir, { recursive: true })
  const userDataDir = mkdtempSync(path.join(tmpdir(), 'vmlx-metal-chat-ui-userdata-'))
  const debugPort = await freePort()
  const mock = await startMockServer()
  const app = startApp(userDataDir, debugPort)
  const result = {
    generatedAt: startedAt.toISOString(),
    status: 'running',
    proofDir,
    outJson,
    shotPath,
    appCommand: app.command,
    appMode: app.mode,
    mockPort: mock.port,
    expectedVisibleText: 'Generation blocked',
  }
  writeResult(result)

  let browser
  let page
  try {
    const cdp = await waitForCdp(debugPort)
    browser = await chromium.connectOverCDP(cdp)
    page = await getAppPage(browser)
    if (!page) throw new Error('No Electron renderer page found')
    await page.waitForLoadState('domcontentloaded').catch(() => {})
    await waitForWindowApi(page)

    const renderer = await page.evaluate(async ({ port }) => {
      await window.api.chat.clearAllLocks().catch(() => null)
      const remote = await window.api.sessions.createRemote({
        remoteUrl: `http://127.0.0.1:${port}`,
        remoteModel: 'vmlx-metal-headroom-mock',
      })
      if (!remote.success) throw new Error(remote.error || 'remote create failed')
      await window.api.sessions.start(remote.session.id)
      const chat = await window.api.chat.create('Metal headroom UI proof', 'vmlx-metal-headroom-mock', undefined, remote.session.modelPath)
      await window.api.chat.setOverrides(chat.id, {
        chatId: chat.id,
        wireApi: 'completions',
        maxTokens: 8192,
      })
      const sendResult = await window.api.chat.sendMessage(chat.id, 'Trigger projected Metal headroom guard.')
      const messages = await window.api.chat.getMessages(chat.id)
      const streamingAfter = await window.api.chat.isStreaming(chat.id)
      const assistant = [...messages].reverse().find((m) => m.role === 'assistant')
      return {
        remoteSessionId: remote.session.id,
        chatId: chat.id,
        sendResult,
        messages,
        assistant,
        streamingAfter,
      }
    }, { port: mock.port })

    result.renderer = renderer
    result.mockRequests = mock.requests
    result.appLogTail = app.logs.slice(-200)
    const assistantContent = String(renderer.assistant?.content || '')
    const failures = []
    if (!assistantContent.includes('Generation blocked')) failures.push('visible Generation blocked assistant content missing')
    if (!assistantContent.includes('requested=8192')) failures.push('assistant content missing requested=8192')
    if (!assistantContent.includes('safe_cap=1')) failures.push('assistant content missing safe_cap=1')
    if (!assistantContent.includes('Metal OOM / kernel-panic risk')) failures.push('assistant content missing kernel-panic risk')
    if (renderer.streamingAfter !== false) failures.push('chat streaming lock did not clear')
    if (app.logs.some((line) => line.includes('[CHAT] Error caught'))) {
      failures.push('projected Metal headroom safety block was logged as an unexpected chat error')
    }
    if (!mock.requests.some((r) => r.url === '/v1/chat/completions' && r.body?.max_tokens === 8192 && r.body?.stream === true)) {
      failures.push('mock did not receive streaming Chat Completions max_tokens=8192 request')
    }
    result.status = failures.length ? 'fail' : 'pass'
    result.failures = failures
    await page.screenshot({ path: shotPath, fullPage: false }).catch(() => null)
    result.screenshot = shotPath
    result.finishedAt = new Date().toISOString()
    writeResult(result)
    console.log(JSON.stringify({ status: result.status, failures, outJson, shotPath }, null, 2))
    process.exitCode = result.status === 'pass' ? 0 : 1
  } catch (error) {
    result.status = 'fail'
    result.failures = [String(error?.stack || error?.message || error)]
    result.mockRequests = mock.requests
    result.appLogTail = app.logs.slice(-200)
    if (page) await page.screenshot({ path: shotPath, fullPage: false }).catch(() => null)
    writeResult(result)
    console.error(JSON.stringify({ status: 'fail', failures: result.failures, outJson, shotPath }, null, 2))
    process.exitCode = 1
  } finally {
    await browser?.close().catch(() => null)
    await new Promise((resolve) => mock.server.close(resolve)).catch(() => null)
    if (app.proc.pid) {
      try { process.kill(-app.proc.pid, 'SIGTERM') } catch {}
      await sleep(1000)
      try { process.kill(-app.proc.pid, 'SIGKILL') } catch {}
    }
    rmSync(userDataDir, { recursive: true, force: true })
  }
}

main().catch((error) => {
  console.error(error?.stack || error?.message || String(error))
  process.exit(1)
})
