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
const proofDir = path.resolve(process.env.VMLX_METAL_UI_PROOF_DIR || path.join(repoDir, 'build', `live-metal-headroom-ui-${stamp}`))
const outJson = path.join(proofDir, 'metal-headroom-ui-proof.json')
const shotPath = path.join(proofDir, 'metal-headroom-ui-proof.png')
const settingsShotPath = path.join(proofDir, 'metal-headroom-settings-proof.png')
const m3SettingsShotPath = path.join(proofDir, 'metal-headroom-m3-settings-proof.png')
const appPath = process.env.VMLX_APP_PATH || '/Applications/vMLX.app'
const appExe = path.join(appPath, 'Contents', 'MacOS', 'vMLX')
const useElectronDev = process.env.VMLINUX_ELECTRON_DEV === '1' || process.env.VMLX_ELECTRON_DEV === '1'

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
      if (window.api?.sessions) return true
      await new Promise((resolve) => setTimeout(resolve, 100))
    }
    throw new Error('window.api.sessions not ready')
  })
}

async function suppressUpdateNotice(page) {
  await page.evaluate(async () => {
    try {
      await window.api?.settings?.set?.('notice_dismissed_version', '1.5.45')
    } catch {}
  }).catch(() => {})
}

function startApp(userDataDir, debugPort) {
  const args = [`--user-data-dir=${userDataDir}`, `--remote-debugging-port=${debugPort}`]
  const appLog = []
  if (useElectronDev) {
    const proc = spawn('npm', ['run', 'dev', '--', '--', ...args], {
      cwd: panelDir,
      env: { ...process.env, VMLX_SKIP_UPDATE_CHECK: '1' },
      detached: true,
      stdio: ['ignore', 'pipe', 'pipe'],
    })
    proc.stdout.on('data', (d) => appLog.push(...d.toString().split(/\r?\n/).filter(Boolean)))
    proc.stderr.on('data', (d) => appLog.push(...d.toString().split(/\r?\n/).filter(Boolean)))
    return { proc, appLog, command: ['npm', 'run', 'dev', '--', '--', ...args], appMode: 'electron-dev' }
  }
  if (!existsSync(appExe)) throw new Error(`App executable missing: ${appExe}`)
  const proc = spawn(appExe, args, {
    cwd: tmpdir(),
    env: { ...process.env, VMLX_SKIP_UPDATE_CHECK: '1' },
    detached: true,
    stdio: ['ignore', 'pipe', 'pipe'],
  })
  proc.stdout.on('data', (d) => appLog.push(...d.toString().split(/\r?\n/).filter(Boolean)))
  proc.stderr.on('data', (d) => appLog.push(...d.toString().split(/\r?\n/).filter(Boolean)))
  return { proc, appLog, command: [appExe, ...args], appMode: 'installed-app' }
}

async function collectPagedCacheSettingsUi(page, fakeModelDir) {
  return await page.evaluate(async ({ modelPath }) => {
    const wait = (predicate, timeoutMs = 20_000) => new Promise((resolve, reject) => {
      const started = Date.now()
      const tick = () => {
        try {
          const value = predicate()
          if (value) return resolve(value)
        } catch {}
        if (Date.now() - started > timeoutMs) {
          return reject(new Error(`timeout waiting for settings UI: ${document.body.innerText.slice(0, 2000)}`))
        }
        setTimeout(tick, 100)
      }
      tick()
    })
    const dismissButton = [...document.querySelectorAll('button')]
      .find((button) => button.innerText.includes('Got it'))
    if (dismissButton) {
      dismissButton.click()
      await new Promise((resolve) => setTimeout(resolve, 250))
    }
    window.dispatchEvent(new CustomEvent('vmlx:navigate', {
      detail: { mode: 'server', panel: 'create', modelPath },
    }))
    await wait(() => document.body.innerText.includes('Server Settings'))
    await wait(() => document.body.innerText.includes('Prefix Cache'))
    const clickSection = async (title, visibleAfterClick) => {
      const candidates = [...document.querySelectorAll('button, [role="button"], div, span')]
        .map((element) => {
          const rect = element.getBoundingClientRect()
          return { element, rect, text: element.innerText || element.textContent || '' }
        })
        .filter(({ rect, text }) => text.includes(title) && rect.width > 0 && rect.height > 0 && rect.height < 90)
        .sort((a, b) => (a.rect.height * a.rect.width) - (b.rect.height * b.rect.width))
      const target = candidates[0]?.element
      if (!target) throw new Error(`settings section not found: ${title}`)
      const clickable = target.closest('button') || target.closest('[role="button"]') || target
      clickable.scrollIntoView({ block: 'center' })
      await new Promise((resolve) => setTimeout(resolve, 100))
      clickable.dispatchEvent(new MouseEvent('mousedown', { bubbles: true, cancelable: true }))
      clickable.dispatchEvent(new MouseEvent('mouseup', { bubbles: true, cancelable: true }))
      clickable.dispatchEvent(new MouseEvent('click', { bubbles: true, cancelable: true }))
      if (visibleAfterClick) {
        await wait(() => document.body.innerText.includes(visibleAfterClick))
      } else {
        await new Promise((resolve) => setTimeout(resolve, 150))
      }
      return true
    }
    await clickSection('Prefix Cache', 'Enable Prefix Cache')
    await clickSection('Paged KV Cache', 'Use Paged KV Cache')

    const labelFor = (text) => [...document.querySelectorAll('label')]
      .find((label) => label.innerText.includes(text))
    const checkboxFor = (text) => labelFor(text)?.querySelector('input[type="checkbox"]')
    const sliderRootFor = (text) => {
      const span = [...document.querySelectorAll('span')]
        .find((node) => node.innerText?.includes(text))
      return span?.closest('.block')
    }
    const disabledStateFor = (text) => {
      const root = sliderRootFor(text)
      const inputs = root ? [...root.querySelectorAll('input')] : []
      return {
        found: !!root && inputs.length > 0,
        disabled: inputs.length > 0 && inputs.every((input) => input.disabled),
      }
    }
    const ensurePrefix = checkboxFor('Enable Prefix Cache')
    if (!ensurePrefix) throw new Error('Enable Prefix Cache checkbox not found')
    if (!ensurePrefix.checked) ensurePrefix.click()
    await wait(() => checkboxFor('Enable Prefix Cache')?.checked)
    const paged = checkboxFor('Use Paged KV Cache')
    if (!paged) throw new Error('Use Paged KV Cache checkbox not found')
    if (!paged.checked) paged.click()
    await wait(() => document.body.innerText.includes('Effective paged capacity:'))
    const onState = {
      prefixChecked: !!checkboxFor('Enable Prefix Cache')?.checked,
      pagedChecked: !!checkboxFor('Use Paged KV Cache')?.checked,
      bodyIncludesCapacity: document.body.innerText.includes('Effective paged capacity: 64 tokens/block x 1000 blocks = 64,000 tokens'),
      bodyIncludesIgnoredText: document.body.innerText.includes('Cache Memory Limit, Cache Memory %, and Cache TTL are ignored while paged cache is active'),
      cacheMemoryLimit: disabledStateFor('Cache Memory Limit'),
      cacheMemoryPercent: disabledStateFor('Cache Memory %'),
      cacheTtl: disabledStateFor('Cache TTL'),
    }
    paged.click()
    await wait(() => !checkboxFor('Use Paged KV Cache')?.checked)
    const offState = {
      prefixChecked: !!checkboxFor('Enable Prefix Cache')?.checked,
      pagedChecked: !!checkboxFor('Use Paged KV Cache')?.checked,
      bodyIncludesCapacity: document.body.innerText.includes('Effective paged capacity:'),
      cacheMemoryLimit: disabledStateFor('Cache Memory Limit'),
      cacheMemoryPercent: disabledStateFor('Cache Memory %'),
      cacheTtl: disabledStateFor('Cache TTL'),
    }
    sliderRootFor('Cache Memory Limit')?.scrollIntoView({ block: 'center' })
    await new Promise((resolve) => setTimeout(resolve, 150))
    return {
      visible: true,
      onState,
      offState,
      textHead: document.body.innerText.slice(0, 1800),
    }
  }, { modelPath: fakeModelDir })
}

async function collectMinimaxM3SettingsUi(page, fakeM3ModelDir) {
  return await page.evaluate(async ({ modelPath }) => {
    const wait = (predicate, timeoutMs = 20_000) => new Promise((resolve, reject) => {
      const started = Date.now()
      const tick = () => {
        try {
          const value = predicate()
          if (value) return resolve(value)
        } catch {}
        if (Date.now() - started > timeoutMs) {
          return reject(new Error(`timeout waiting for M3 settings UI: ${document.body.innerText.slice(0, 2400)}`))
        }
        setTimeout(tick, 100)
      }
      tick()
    })
    const dismissButton = [...document.querySelectorAll('button')]
      .find((button) => button.innerText.includes('Got it'))
    if (dismissButton) {
      dismissButton.click()
      await new Promise((resolve) => setTimeout(resolve, 250))
    }
    window.dispatchEvent(new CustomEvent('vmlx:navigate', {
      detail: { mode: 'server', panel: 'create', modelPath },
    }))
    await wait(() => document.body.innerText.includes('Server Settings'))
    await wait(() => document.body.innerText.includes('Prefix Cache'))

    const clickSection = async (title, visibleAfterClick) => {
      const candidates = [...document.querySelectorAll('button, [role="button"], div, span')]
        .map((element) => {
          const rect = element.getBoundingClientRect()
          return { element, rect, text: element.innerText || element.textContent || '' }
        })
        .filter(({ rect, text }) => text.includes(title) && rect.width > 0 && rect.height > 0 && rect.height < 100)
        .sort((a, b) => (a.rect.height * a.rect.width) - (b.rect.height * b.rect.width))
      const target = candidates[0]?.element
      if (!target) throw new Error(`settings section not found: ${title}`)
      const clickable = target.closest('button') || target.closest('[role="button"]') || target
      clickable.scrollIntoView({ block: 'center' })
      await new Promise((resolve) => setTimeout(resolve, 100))
      clickable.dispatchEvent(new MouseEvent('mousedown', { bubbles: true, cancelable: true }))
      clickable.dispatchEvent(new MouseEvent('mouseup', { bubbles: true, cancelable: true }))
      clickable.dispatchEvent(new MouseEvent('click', { bubbles: true, cancelable: true }))
      if (visibleAfterClick) {
        await wait(() => document.body.innerText.includes(visibleAfterClick))
      } else {
        await new Promise((resolve) => setTimeout(resolve, 150))
      }
      return true
    }

    await clickSection('Paged KV Cache', 'Use Paged KV Cache')
    await wait(() => document.body.innerText.includes('MiniMax-M3 uses paged-off SSD prefix cache with native MSA idx_keys'))
    await clickSection('KV Cache Quantization', 'Stored Cache Quantization')
    await wait(() => document.body.innerText.includes('MiniMax-M3 keeps generic KV q4/q8 disabled'))
    await wait(() => document.body.innerText.includes('generic stored-KV codecs cannot preserve that cache format'))

    const isVisible = (element) => {
      if (!element) return false
      const rect = element.getBoundingClientRect()
      return rect.width > 0 && rect.height > 0
    }
    const labelFor = (text) => [...document.querySelectorAll('label')]
      .find((label) => label.innerText.includes(text) && isVisible(label))
    const checkboxFor = (text) => labelFor(text)?.querySelector('input[type="checkbox"]')
    const blockFor = (text) => [...document.querySelectorAll('.block')]
      .find((block) => block.innerText?.includes(text) && isVisible(block))
    const storedQuantSelect = blockFor('Stored Cache Quantization')?.querySelector('select')
    const pagedCheckbox = checkboxFor('Use Paged KV Cache')
    const bodyText = document.body.innerText
    const inputIncludesModelPath = [...document.querySelectorAll('input, textarea')]
      .some((input) => String(input.value || '').includes(modelPath) && isVisible(input))

    return {
      visible: true,
      modelPath,
      activeModelPathFound: bodyText.includes(modelPath) || inputIncludesModelPath,
      pagedCheckboxFound: !!pagedCheckbox,
      pagedChecked: !!pagedCheckbox?.checked,
      pagedDisabled: !!pagedCheckbox?.disabled,
      storedQuantSelectFound: !!storedQuantSelect,
      storedQuantValue: storedQuantSelect?.value || null,
      storedQuantDisabled: !!storedQuantSelect?.disabled,
      bodyIncludesM3PagedOff: bodyText.includes('MiniMax-M3 uses paged-off SSD prefix cache with native MSA idx_keys'),
      bodyIncludesM3StoredCodec: bodyText.includes('MiniMax-M3 keeps generic KV q4/q8 disabled'),
      bodyIncludesM3NativeCacheFormat: bodyText.includes('generic stored-KV codecs cannot preserve native MSA') || bodyText.includes('generic stored-KV codecs cannot preserve that cache format'),
      textHead: bodyText.slice(0, 2400),
    }
  }, { modelPath: fakeM3ModelDir })
}

async function main() {
  mkdirSync(proofDir, { recursive: true })
  const userDataDir = mkdtempSync(path.join(tmpdir(), 'vmlx-metal-ui-userdata-'))
  const fakeModelDir = mkdtempSync(path.join(tmpdir(), 'vmlx-metal-ui-model-'))
  const fakeM3ModelDir = mkdtempSync(path.join(tmpdir(), 'vmlx-metal-ui-m3-model-'))
  const debugPort = await freePort()
  const appArgs = [`--user-data-dir=${userDataDir}`, `--remote-debugging-port=${debugPort}`]
  const app = startApp(userDataDir, debugPort)
  const result = {
    generatedAt: startedAt.toISOString(),
    status: 'running',
    appPath,
    appMode: app.appMode,
    appCommand: app.command,
    proofDir,
    outJson,
    shotPath,
    settingsShotPath,
    m3SettingsShotPath,
    userDataDir,
    fakeModelDir,
    fakeM3ModelDir,
    expectedLog: 'Paged cache capacity: 64 tokens/block x 64 blocks = 4096 tokens.',
  }
  writeFileSync(path.join(fakeModelDir, 'config.json'), JSON.stringify({
    model_type: 'qwen3',
    architectures: ['Qwen3ForCausalLM'],
    num_hidden_layers: 1,
    num_key_value_heads: 1,
    num_attention_heads: 1,
    head_dim: 64,
    hidden_size: 64,
  }, null, 2))
  writeFileSync(path.join(fakeM3ModelDir, 'config.json'), JSON.stringify({
    model_type: 'minimax_m3_vl',
    architectures: ['MiniMaxM3ForConditionalGeneration'],
    num_hidden_layers: 1,
    num_key_value_heads: 1,
    num_attention_heads: 1,
    head_dim: 128,
    hidden_size: 128,
  }, null, 2))
  writeResult(result)

  result.appPid = app.proc.pid
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
    await suppressUpdateNotice(page)
    await page.reload({ waitUntil: 'domcontentloaded' }).catch(() => {})
    await waitForWindowApi(page)
    const minimaxM3SettingsUi = await collectMinimaxM3SettingsUi(page, fakeM3ModelDir)
    result.minimaxM3SettingsUi = minimaxM3SettingsUi
    await page.screenshot({ path: m3SettingsShotPath, fullPage: false }).catch(() => null)
    await page.reload({ waitUntil: 'domcontentloaded' }).catch(() => {})
    await waitForWindowApi(page)
    const settingsUi = await collectPagedCacheSettingsUi(page, fakeModelDir)
    result.settingsUi = settingsUi
    await page.screenshot({ path: settingsShotPath, fullPage: false }).catch(() => null)

    const sessionResult = await page.evaluate(async ({ modelPath }) => {
      const created = await window.api.sessions.create(modelPath, {
        enablePrefixCache: true,
        usePagedCache: true,
        pagedCacheBlockSize: 64,
        maxCacheBlocks: 64,
        enableDiskCache: false,
        enableBlockDiskCache: false,
        maxTokens: 8192,
        maxContextLength: 8192,
      })
      if (!created?.success || !created?.session?.id) {
        throw new Error(`create failed: ${created?.error || JSON.stringify(created)}`)
      }
      const started = await window.api.sessions.start(created.session.id)
      const logs = await window.api.sessions.getLogs(created.session.id)
      const session = await window.api.sessions.get(created.session.id)
      return { created, started, logs, session }
    }, { modelPath: fakeModelDir })

    result.session = sessionResult.session
    result.startResult = sessionResult.started
    result.logs = sessionResult.logs
    result.appLogTail = app.appLog.slice(-200)
    const joined = (sessionResult.logs || []).join('\n')
    result.pagedCapacityLogFound = joined.includes(result.expectedLog)
    result.cacheMemoryIgnoredTextFound = joined.includes('--cache-memory-mb/--cache-memory-percent are ignored while paged cache is active')
    result.failures = []
    const onState = settingsUi.onState || {}
    const offState = settingsUi.offState || {}
    if (!settingsUi.visible) result.failures.push('settings UI was not visible')
    if (!onState.pagedChecked) result.failures.push('paged cache on-state checkbox was not checked')
    if (!onState.bodyIncludesCapacity) result.failures.push('paged cache on-state effective capacity text missing')
    if (!onState.bodyIncludesIgnoredText) result.failures.push('paged cache on-state ignored MB/%/TTL text missing')
    for (const [label, state] of Object.entries({
      'Cache Memory Limit': onState.cacheMemoryLimit,
      'Cache Memory %': onState.cacheMemoryPercent,
      'Cache TTL': onState.cacheTtl,
    })) {
      if (!state?.found) result.failures.push(`${label} control not found in paged on-state`)
      if (!state?.disabled) result.failures.push(`${label} control not disabled in paged on-state`)
    }
    if (offState.pagedChecked) result.failures.push('paged cache off-state checkbox was still checked')
    if (offState.bodyIncludesCapacity) result.failures.push('paged cache off-state still showed effective capacity text')
    for (const [label, state] of Object.entries({
      'Cache Memory Limit': offState.cacheMemoryLimit,
      'Cache Memory %': offState.cacheMemoryPercent,
      'Cache TTL': offState.cacheTtl,
    })) {
      if (!state?.found) result.failures.push(`${label} control not found in paged off-state`)
      if (state?.disabled) result.failures.push(`${label} control still disabled in paged off-state`)
    }
    if (!result.pagedCapacityLogFound) result.failures.push('paged cache capacity line missing')
    if (!result.cacheMemoryIgnoredTextFound) result.failures.push('cache-memory ignored text missing')
    if (!minimaxM3SettingsUi.visible) result.failures.push('M3 settings UI was not visible')
    if (!minimaxM3SettingsUi.activeModelPathFound) result.failures.push('M3 fake model path was not visible/active in settings UI')
    if (!minimaxM3SettingsUi.pagedCheckboxFound) result.failures.push('M3 Use Paged KV Cache checkbox not found')
    if (minimaxM3SettingsUi.pagedChecked) result.failures.push('M3 Use Paged KV Cache checkbox was checked')
    if (!minimaxM3SettingsUi.pagedDisabled) result.failures.push('M3 Use Paged KV Cache checkbox was not disabled')
    if (!minimaxM3SettingsUi.storedQuantSelectFound) result.failures.push('M3 Stored Cache Quantization select not found')
    if (minimaxM3SettingsUi.storedQuantValue !== 'auto') result.failures.push(`M3 Stored Cache Quantization value was ${minimaxM3SettingsUi.storedQuantValue}`)
    if (!minimaxM3SettingsUi.storedQuantDisabled) result.failures.push('M3 Stored Cache Quantization select was not disabled')
    if (!minimaxM3SettingsUi.bodyIncludesM3PagedOff) result.failures.push('M3 paged-off explanation missing')
    if (!minimaxM3SettingsUi.bodyIncludesM3StoredCodec) result.failures.push('M3 stored-codec explanation missing')
    if (!minimaxM3SettingsUi.bodyIncludesM3NativeCacheFormat) result.failures.push('M3 native cache-format explanation missing')
    result.status = result.failures.length === 0 ? 'pass' : 'fail'
    await page.screenshot({ path: shotPath, fullPage: false }).catch(() => null)
    result.screenshot = shotPath
    result.settingsScreenshot = settingsShotPath
    result.finishedAt = new Date().toISOString()
    writeResult(result)
    console.log(JSON.stringify({ status: result.status, failures: result.failures, outJson, shotPath }, null, 2))
    process.exitCode = result.status === 'pass' ? 0 : 1
  } catch (error) {
    result.status = 'fail'
    result.failures = [String(error?.stack || error?.message || error)]
    result.appLogTail = app.appLog.slice(-200)
    if (page) await page.screenshot({ path: shotPath, fullPage: false }).catch(() => null)
    writeResult(result)
    console.error(JSON.stringify({ status: 'fail', failures: result.failures, outJson, shotPath }, null, 2))
    process.exitCode = 1
  } finally {
    await browser?.close().catch(() => null)
    if (app.proc.pid) {
      try { process.kill(-app.proc.pid, 'SIGTERM') } catch {}
      await sleep(1000)
      try { process.kill(-app.proc.pid, 'SIGKILL') } catch {}
    }
    rmSync(userDataDir, { recursive: true, force: true })
    rmSync(fakeModelDir, { recursive: true, force: true })
    rmSync(fakeM3ModelDir, { recursive: true, force: true })
  }
}

main().catch((error) => {
  console.error(error?.stack || error?.message || String(error))
  process.exit(1)
})
