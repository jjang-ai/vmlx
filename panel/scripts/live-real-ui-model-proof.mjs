#!/usr/bin/env node
import { execFile, spawn } from 'node:child_process'
import crypto from 'node:crypto'
import { createServer, get as httpGet } from 'node:http'
import net from 'node:net'
import {
  chmodSync,
  closeSync,
  constants as fsConstants,
  existsSync,
  fstatSync,
  lstatSync,
  mkdirSync,
  mkdtempSync,
  openSync,
  readFileSync,
  readSync,
  readdirSync,
  realpathSync,
  rmSync,
  writeFileSync,
} from 'node:fs'
import { tmpdir } from 'node:os'
import path from 'node:path'
import { fileURLToPath } from 'node:url'
import { promisify } from 'node:util'
import vm from 'node:vm'

const panelDir = path.resolve(new URL('..', import.meta.url).pathname)
const repoDir = path.resolve(panelDir, '..')
const proofFormat = 'vmlx-electron-ui-proof-v2'
const ownedRunIntentSchema = 'vmlx-r20-owned-run-intent-v5'
const ownedUiReleaseSchema = 'vmlx-r20-owned-ui-release-v5'
const ownedUiSessionAttestationSchema = 'vmlx-r20-owned-ui-session-attestation-v5'
const executableIdentityMaxBytes = 512 * 1024 * 1024
const installedReleaseManifestSchema = 'vmlx-installed-release-manifest-v1'
const installedReleaseManifestFields = [
  'app_asar_sha256',
  'bundled_provenance_sha256',
  'bundled_python_executable_fingerprint_sha256',
  'bundled_python_executable_sha256',
  'electron_executable_sha256',
  'schema',
  'source_commit',
  'source_tree',
]
const installedBundledPythonRelativePath = path.join(
  'Contents',
  'Resources',
  'bundled-python',
  'python',
  'bin',
  'python3',
)
const installedAppArtifactPaths = {
  app_asar: {
    manifestField: 'app_asar_sha256',
    relativePath: path.join('Contents', 'Resources', 'app.asar'),
  },
  electron_executable: {
    manifestField: 'electron_executable_sha256',
    relativePath: path.join('Contents', 'MacOS', 'vMLX'),
  },
  bundled_provenance: {
    manifestField: 'bundled_provenance_sha256',
    relativePath: path.join(
      'Contents',
      'Resources',
      'bundled-python',
      'vmlx-bundle-provenance.json',
    ),
  },
}
const proofDirInput = process.env.VMLINUX_REAL_UI_PROOF_DIR
  || process.env.VMLX_REAL_UI_PROOF_DIR
  || process.env.VMLX_PRIVATE_EVIDENCE_ROOT
  || ''
const runId = process.env.VMLINUX_REAL_UI_RUN_ID
  || process.env.VMLX_REAL_UI_RUN_ID
  || `vmlx-ui-${new Date().toISOString().replace(/[^0-9TZ]/g, '')}-${crypto.randomUUID()}`
const modelPath = process.env.VMLINUX_REAL_UI_MODEL_PATH || process.env.VMLX_REAL_UI_MODEL_PATH
const installedAppPath = process.env.VMLINUX_REAL_UI_APP_PATH
  || process.env.VMLX_REAL_UI_APP_PATH
  || ''
const servedModel = process.env.VMLINUX_REAL_UI_SERVED_MODEL
  || process.env.VMLX_REAL_UI_SERVED_MODEL
  || path.basename(modelPath || 'real-ui-model').replace(/[^A-Za-z0-9_.-]+/g, '-')
const modelName = path.basename(modelPath || servedModel)
const requestedProofBasename = process.env.VMLINUX_REAL_UI_PROOF_BASENAME
  || process.env.VMLX_REAL_UI_PROOF_BASENAME
  || ''
const releaseSentinelPath = process.env.VMLINUX_REAL_UI_RELEASE_SENTINEL
  || process.env.VMLX_REAL_UI_RELEASE_SENTINEL
  || ''
const releaseSentinelNonce = process.env.VMLINUX_REAL_UI_NONCE
  || process.env.VMLX_REAL_UI_NONCE
  || ''
const releaseRunIntentPath = (
  process.env.VMLINUX_REAL_UI_RUN_INTENT_PATH
  || process.env.VMLX_REAL_UI_RUN_INTENT_PATH
  || ''
).trim()
const releaseRunIntentSha256 = (
  process.env.VMLINUX_REAL_UI_RUN_INTENT_SHA256
  || process.env.VMLX_REAL_UI_RUN_INTENT_SHA256
  || ''
).trim()
const releaseActivePhaseIndexRaw = (
  process.env.VMLINUX_REAL_UI_ACTIVE_PHASE_INDEX
  || process.env.VMLX_REAL_UI_ACTIVE_PHASE_INDEX
  || ''
).trim()
const releaseActivePhaseIndex = releaseActivePhaseIndexRaw === ''
  ? null
  : Number(releaseActivePhaseIndexRaw)
const releaseSessionAttestationPath = (
  process.env.VMLINUX_REAL_UI_SESSION_ATTESTATION_PATH
  || process.env.VMLX_REAL_UI_SESSION_ATTESTATION_PATH
  || ''
).trim()
const releaseGatewayPidRaw = (
  process.env.VMLINUX_REAL_UI_GATEWAY_PID
  || process.env.VMLX_REAL_UI_GATEWAY_PID
  || ''
).trim()
const releaseGatewayPid = releaseGatewayPidRaw === ''
  ? null
  : Number(releaseGatewayPidRaw)
const releaseGatewayBaseUrl = (
  process.env.VMLINUX_REAL_UI_GATEWAY_BASE_URL
  || process.env.VMLX_REAL_UI_GATEWAY_BASE_URL
  || ''
).trim()
const attachCdpUrl = (
  process.env.VMLINUX_REAL_UI_ATTACH_CDP_URL
  || process.env.VMLX_REAL_UI_ATTACH_CDP_URL
  || ''
).trim()
const expectedElectronPidRaw = (
  process.env.VMLINUX_REAL_UI_EXPECTED_ELECTRON_PID
  || process.env.VMLX_REAL_UI_EXPECTED_ELECTRON_PID
  || ''
).trim()
const expectedElectronPid = expectedElectronPidRaw === ''
  ? null
  : Number(expectedElectronPidRaw)
const releaseRetainedPidsRaw = (
  process.env.VMLINUX_REAL_UI_RETAINED_PIDS
  || process.env.VMLX_REAL_UI_RETAINED_PIDS
  || ''
).trim()
const releaseRetainedPids = parseExplicitPidList(
  releaseRetainedPidsRaw,
  'Real UI retained PIDs',
)
const lifecycleOwner = (
  process.env.VMLINUX_REAL_UI_LIFECYCLE_OWNER
  || process.env.VMLX_REAL_UI_LIFECYCLE_OWNER
  || ''
).trim()
const allowTeardown = envBool('VMLINUX_REAL_UI_ALLOW_TEARDOWN', true)
const pairedCacheArtifactPath = (
  process.env.VMLINUX_REAL_UI_PAIRED_CACHE_ARTIFACT
  || process.env.VMLX_REAL_UI_PAIRED_CACHE_ARTIFACT
  || ''
).trim()
const reuseSessionId = (
  process.env.VMLINUX_REAL_UI_REUSE_SESSION_ID
  || process.env.VMLX_REAL_UI_REUSE_SESSION_ID
  || ''
).trim()
const reuseSessionAttestationPath = (
  process.env.VMLINUX_REAL_UI_REUSE_SESSION_ATTESTATION_PATH
  || process.env.VMLX_REAL_UI_REUSE_SESSION_ATTESTATION_PATH
  || ''
).trim()
const privateCacheAttestationTokenFile = (
  process.env.VMLINUX_PRIVATE_CACHE_ATTESTATION_TOKEN_FILE
  || process.env.VMLX_PRIVATE_CACHE_ATTESTATION_TOKEN_FILE
  || ''
).trim()

function safeArtifactComponent(value, fallback) {
  const safe = String(value || '')
    .trim()
    .replace(/[^A-Za-z0-9_.-]+/g, '-')
    .replace(/^-+|-+$/g, '')
  return (safe || fallback).slice(0, 120)
}

export function uniqueProofBasename({
  requested = '',
  model = 'real-ui-model',
  run = 'run',
} = {}) {
  const tuple = {
    requested: String(requested),
    model: String(model),
    run: String(run),
  }
  const suffix = sha256Text(canonicalJson(tuple)).slice(0, 20)
  const root = safeArtifactComponent(requested, 'real-ui-proof')
  const modelComponent = safeArtifactComponent(model, 'model')
  const runComponent = safeArtifactComponent(run, 'run')
  const prefix = `${root}-${modelComponent}-${runComponent}`
  return `${prefix.slice(0, 240 - suffix.length - 1)}-${suffix}`
}

export function ownedUiProducerPid({
  orchestrated = false,
  harnessPid = process.pid,
  parentPid = process.ppid,
} = {}) {
  const pid = orchestrated ? Number(parentPid) : Number(harnessPid)
  if (!Number.isInteger(pid) || pid <= 1) {
    throw new Error('Owned UI producer PID is invalid')
  }
  return pid
}

export function waitForCurrentSessionStart({
  sessions,
  sessionId,
  baselineLastStartedAt = 0,
  click,
  timeoutMs = 900_000,
  pollMs = 100,
} = {}) {
  if (
    !sessions
    || typeof sessions.get !== 'function'
    || typeof sessions.getLogs !== 'function'
    || typeof sessions.onStarting !== 'function'
    || typeof sessions.onReady !== 'function'
    || typeof sessions.onError !== 'function'
    || typeof click !== 'function'
    || !String(sessionId || '').trim()
  ) {
    throw new Error('Current-session Start waiter has an invalid contract')
  }
  const baseline = Number.isFinite(Number(baselineLastStartedAt))
    ? Number(baselineLastStartedAt)
    : 0

  return new Promise((resolve, reject) => {
    let settled = false
    let currentAttemptObserved = false
    let pollTimer = null
    let deadlineTimer = null
    const unsubs = []

    const cleanup = () => {
      if (pollTimer != null) clearTimeout(pollTimer)
      if (deadlineTimer != null) clearTimeout(deadlineTimer)
      for (const unsubscribe of unsubs.splice(0)) {
        try {
          if (typeof unsubscribe === 'function') unsubscribe()
        } catch {
          // Cleanup must not replace the authoritative Start result.
        }
      }
    }
    const startedAt = (session) => {
      const value = Number(
        session?.lastStartedAt
        ?? session?.last_started_at
        ?? 0,
      )
      return Number.isFinite(value) ? value : 0
    }
    const isTarget = (data) => String(data?.sessionId || '') === sessionId
    const resolveCurrent = () => {
      if (settled) return
      settled = true
      cleanup()
      Promise.resolve(sessions.get(sessionId)).then((current) => {
        if (current?.status !== 'running') {
          reject(new Error(
            'Current UI Start emitted ready without a running session row',
          ))
          return
        }
        resolve(current)
      }, reject)
    }
    const rejectCurrent = (data) => {
      if (settled) return
      settled = true
      cleanup()
      Promise.resolve(sessions.getLogs(sessionId)).catch(() => []).then((logs) => {
        reject(new Error(
          'UI Start control left session in error state: '
          + JSON.stringify({
            error: String(data?.error || ''),
            logs: Array.isArray(logs) ? logs.slice(-80) : [],
          }),
        ))
      })
    }
    const check = async () => {
      if (settled) return
      try {
        const current = await sessions.get(sessionId)
        if (settled) return
        // Polling may return a snapshot requested just before onStarting.  A
        // lifecycle event is current-attempt evidence, but it must not relabel
        // that older snapshot.  The persisted lastStartedAt written with the
        // loading transition is the polling-path ownership boundary; ready and
        // error events remain independently authoritative above.
        const belongsToCurrentAttempt = startedAt(current) > baseline
        if (belongsToCurrentAttempt && current?.status === 'running') {
          resolveCurrent()
          return
        }
        if (belongsToCurrentAttempt && current?.status === 'error') {
          rejectCurrent({ error: 'persisted current-attempt error' })
          return
        }
      } catch (error) {
        if (!settled) {
          settled = true
          cleanup()
          reject(error)
        }
        return
      }
      pollTimer = setTimeout(check, pollMs)
    }

    unsubs.push(
      sessions.onStarting((data) => {
        if (isTarget(data)) currentAttemptObserved = true
      }),
      sessions.onReady((data) => {
        if (!isTarget(data)) return
        currentAttemptObserved = true
        resolveCurrent()
      }),
      sessions.onError((data) => {
        if (!isTarget(data)) return
        currentAttemptObserved = true
        rejectCurrent(data)
      }),
    )
    deadlineTimer = setTimeout(() => {
      if (settled) return
      settled = true
      cleanup()
      reject(new Error(
        'Timed out waiting for the current UI Start lifecycle; sawStarting='
        + String(currentAttemptObserved),
      ))
    }, timeoutMs)
    try {
      click()
    } catch (error) {
      settled = true
      cleanup()
      reject(error)
      return
    }
    void check()
  })
}

const proofBasename = uniqueProofBasename({
  requested: requestedProofBasename,
  model: servedModel,
  run: runId,
})
const wireApi = process.env.VMLINUX_REAL_UI_WIRE_API
  || process.env.VMLX_REAL_UI_WIRE_API
  || 'chat'
const promptOneOverride = process.env.VMLINUX_REAL_UI_PROMPT_1
  || process.env.VMLX_REAL_UI_PROMPT_1
const promptTwoOverride = process.env.VMLINUX_REAL_UI_PROMPT_2
  || process.env.VMLX_REAL_UI_PROMPT_2
const promptThreeOverride = process.env.VMLINUX_REAL_UI_PROMPT_3
  || process.env.VMLX_REAL_UI_PROMPT_3
const requestMaxTokensRaw = process.env.VMLINUX_REAL_UI_MAX_TOKENS
  || process.env.VMLX_REAL_UI_MAX_TOKENS
  || ''
const requestMaxTokens = requestMaxTokensRaw
  ? Number(requestMaxTokensRaw)
  : undefined
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

function envNumber(name) {
  const value = process.env[name] ?? process.env[name.replace('VMLINUX_', 'VMLX_')]
  if (value == null || value === '') return undefined
  const parsed = Number(value)
  return Number.isFinite(parsed) ? parsed : undefined
}

export function parseOptionalPort(value, label) {
  if (value == null || String(value).trim() === '') return undefined
  const port = Number(value)
  if (!Number.isInteger(port) || port < 1 || port > 65535) {
    throw new Error(`${label} must be an integer from 1 to 65535`)
  }
  return port
}

export function parseExplicitPidList(value, label = 'PID list') {
  const raw = String(value ?? '').trim()
  if (!raw) return []
  const parts = raw.split(/[\s,]+/).filter(Boolean)
  const pids = parts.map((part) => Number(part))
  if (pids.some((pid) => !Number.isInteger(pid) || pid <= 1)) {
    throw new Error(`${label} must contain only integer PIDs greater than 1`)
  }
  if (new Set(pids).size !== pids.length) {
    throw new Error(`${label} must not contain duplicate PIDs`)
  }
  return pids
}

const requestedCdpPort = parseOptionalPort(
  process.env.VMLINUX_REAL_UI_CDP_PORT
    || process.env.VMLX_REAL_UI_CDP_PORT,
  'Real UI CDP port',
)
const requestedGatewayPort = parseOptionalPort(
  process.env.VMLINUX_REAL_UI_GATEWAY_PORT
    || process.env.VMLX_REAL_UI_GATEWAY_PORT,
  'Real UI gateway port',
)

const builtinToolsEnabled = envBool('VMLINUX_REAL_UI_BUILTIN_TOOLS', false)
const samplingOverrides = {
  temperature: envNumber('VMLINUX_REAL_UI_TEMPERATURE'),
  topP: envNumber('VMLINUX_REAL_UI_TOP_P'),
  topK: envNumber('VMLINUX_REAL_UI_TOP_K'),
  minP: envNumber('VMLINUX_REAL_UI_MIN_P'),
  repeatPenalty: envNumber('VMLINUX_REAL_UI_REPEAT_PENALTY'),
}
const defaultPromptOne = builtinToolsEnabled
  ? [
      'This is an authorized release-validation task in an isolated temporary workspace.',
      'Create real_ui_tool_probe_1.txt with exactly the contents REAL_UI_LIVE_TOOL_ONE by calling the built-in run_command tool once.',
      'Use this command:',
      'printf %s REAL_UI_LIVE_TOOL_ONE > real_ui_tool_probe_1.txt && cat real_ui_tool_probe_1.txt',
      'After the tool result returns, confirm completion briefly in English and include REAL_UI_LIVE_TOOL_ONE once.',
    ].join(' ')
  : 'Reply briefly in English. Include the phrase REAL_UI_LIVE once.'
const defaultPromptTwo = builtinToolsEnabled
  ? [
      'Continue the authorized release-validation task in the same isolated temporary workspace.',
      'Verify the first file and create real_ui_tool_probe_2.txt with exactly the contents REAL_UI_LIVE_TOOL_TWO by calling the built-in run_command tool once.',
      'Use this command:',
      'test "$(cat real_ui_tool_probe_1.txt)" = REAL_UI_LIVE_TOOL_ONE && printf %s REAL_UI_LIVE_TOOL_TWO > real_ui_tool_probe_2.txt && cat real_ui_tool_probe_2.txt',
      'After the tool result returns, confirm completion briefly in English with REAL_UI_LIVE_TOOL_TWO once and mention this is the second UI turn.',
    ].join(' ')
  : 'Repeat the phrase REAL_UI_LIVE once and mention that this is the second UI turn.'
const defaultPromptThree = builtinToolsEnabled
  ? [
      'The authorized release-validation task is complete; no tool is needed for this final response.',
      'Return this exact three-line rendering receipt and nothing else.',
      'Copy every character literally, including the dollar sign and both backslashes.',
      'Do not wrap the answer in markdown or code fences:',
      'Third UI turn: REAL_UI_LIVE_TOOL_ONE REAL_UI_LIVE_TOOL_TWO',
      'Currency: $43',
      'Math: \\(2 + 2 = 4\\)',
    ].join('\n')
  : [
      'Your complete visible answer must be exactly the following three lines and nothing else.',
      'Copy every character literally, including the dollar sign and both backslashes.',
      'Do not wrap the answer in markdown or code fences:',
      'Third UI turn: REAL_UI_LIVE',
      'Currency: $43',
      'Math: \\(2 + 2 = 4\\)',
    ].join('\n')
const promptOne = promptOneOverride || defaultPromptOne
const promptTwo = promptTwoOverride || defaultPromptTwo
const promptThree = promptThreeOverride || defaultPromptThree
const checkServerCacheControls = envBool('VMLINUX_REAL_UI_CHECK_SERVER_CACHE_CONTROLS', false)
// Point this at an mcp-config.json to prove MCP servers and tools are actually
// DISCOVERED in the app, not merely that the section renders. Default empty, so
// existing rows are untouched.
const mcpConfigPath = process.env.VMLINUX_REAL_UI_MCP_CONFIG
  || process.env.VMLX_REAL_UI_MCP_CONFIG
  || ''
const checkMedia = envBool('VMLINUX_REAL_UI_CHECK_MEDIA', false)
const checkVideo = envBool('VMLINUX_REAL_UI_CHECK_VIDEO', false)
const checkAudio = envBool('VMLINUX_REAL_UI_CHECK_AUDIO', false)
// Flip the reasoning mode mid-conversation (after turn 1). Default OFF so
// every existing row is byte-unchanged.
const toggleThinkingMidConv = envBool('VMLINUX_REAL_UI_TOGGLE_THINKING_MIDCONV', false)
const allowedReasoningEfforts = new Set(['low', 'medium', 'high', 'xhigh', 'max'])
const reasoningEffortOverride = (
  process.env.VMLINUX_REAL_UI_REASONING_EFFORT
  || process.env.VMLX_REAL_UI_REASONING_EFFORT
  || ''
).trim().toLowerCase() || undefined
const midConvReasoningEffortOverride = (
  process.env.VMLINUX_REAL_UI_MIDCONV_REASONING_EFFORT
  || process.env.VMLX_REAL_UI_MIDCONV_REASONING_EFFORT
  || ''
).trim().toLowerCase() || undefined
for (const [label, effort] of [
  ['initial', reasoningEffortOverride],
  ['mid-conversation', midConvReasoningEffortOverride],
]) {
  if (effort && !allowedReasoningEfforts.has(effort)) {
    throw new Error(`Real UI ${label} reasoning effort is invalid: ${effort}`)
  }
}
const expectPagedCacheLocked = envBool('VMLINUX_REAL_UI_EXPECT_PAGED_CACHE_LOCKED', false)
const expectPagedCache = envBool('VMLINUX_REAL_UI_EXPECT_PAGED_CACHE', false)
// Select the SSD-only lane (block-disk L2 WITHOUT the RAM paged pool) before
// Start. Until this existed the harness could only ASSERT the lane
// (EXPECT_PAGED_CACHE) and never CHOOSE it, so every live-UI proof ran the
// app-default L1+L2 lane and the SSD-only lane — what disk-only users actually
// run, and the lane where MiniMax-M3 once scored 9 stores / 0 hits while every
// L1+L2 family looked healthy — was never exercised through the UI at all.
// Default OFF so existing rows are byte-unchanged.
const forceSsdOnlyLane = envBool('VMLINUX_REAL_UI_FORCE_SSD_ONLY_LANE', false)
const nativeMtpModeOverride = (
  process.env.VMLINUX_REAL_UI_NATIVE_MTP_MODE
  || process.env.VMLX_REAL_UI_NATIVE_MTP_MODE
  || ''
).trim().toLowerCase() || undefined
const nativeMtpDepthOverride = envNumber('VMLINUX_REAL_UI_NATIVE_MTP_DEPTH')
if (
  nativeMtpModeOverride
  && !['auto', 'deterministic', 'off'].includes(nativeMtpModeOverride)
) {
  throw new Error(
    'Real UI Native MTP mode must be auto, deterministic, or off',
  )
}
if (
  nativeMtpDepthOverride != null
  && (
    !Number.isInteger(nativeMtpDepthOverride)
    || nativeMtpDepthOverride < 1
    || nativeMtpDepthOverride > 3
  )
) {
  throw new Error('Real UI Native MTP depth must be an integer from 1 through 3')
}
if (nativeMtpDepthOverride != null && nativeMtpModeOverride === 'off') {
  throw new Error(
    'Real UI fixed Native MTP depth cannot be combined with MTP Off',
  )
}
const blockDiskCacheMaxPercentOverride = envNumber(
  'VMLINUX_REAL_UI_BLOCK_DISK_CACHE_MAX_PERCENT',
)
if (
  blockDiskCacheMaxPercentOverride != null
  && (
    !Number.isInteger(blockDiskCacheMaxPercentOverride)
    || blockDiskCacheMaxPercentOverride < 0
    || blockDiskCacheMaxPercentOverride > 90
  )
) {
  throw new Error('Real UI SSD cache percent must be an integer from 0 through 90')
}
// Whether the paged expectation was stated at all. Paged cache is DEFAULT ON
// for every autodetected family except M3/openpangu_v2, so a hardcoded
// default-false expectation asserts the operator's memory rather than the
// product: it fails any paged-on family whose invocation forgets the flag,
// which is exactly how the zaya_text row on disk became unreproducible from
// its own documented command. When the flag is absent, the expectation is
// taken from the engine's live /health native_cache instead, which makes this
// a real visible-UI vs running-engine parity assertion.
const expectPagedCacheExplicit = [
  process.env.VMLINUX_REAL_UI_EXPECT_PAGED_CACHE,
  process.env.VMLX_REAL_UI_EXPECT_PAGED_CACHE,
].some((value) => value != null && value !== '')
const expectDsv4PoolQuant = (
  process.env.VMLINUX_REAL_UI_EXPECT_DSV4_POOL_QUANT != null
  || process.env.VMLX_REAL_UI_EXPECT_DSV4_POOL_QUANT != null
)
  ? envBool('VMLINUX_REAL_UI_EXPECT_DSV4_POOL_QUANT', false)
  : undefined
const enableThinkingOverride = (
  process.env.VMLINUX_REAL_UI_ENABLE_THINKING != null
  || process.env.VMLX_REAL_UI_ENABLE_THINKING != null
)
  ? envBool('VMLINUX_REAL_UI_ENABLE_THINKING', false)
  : undefined
const reasoningExpectation = (
  process.env.VMLINUX_REAL_UI_REASONING_EXPECTATION
  || process.env.VMLX_REAL_UI_REASONING_EXPECTATION
  || (enableThinkingOverride === true
    ? 'required'
    : enableThinkingOverride === false
      ? 'none'
      : 'optional')
).toLowerCase()
const maxToolIterations = Number(process.env.VMLINUX_REAL_UI_MAX_TOOL_ITERATIONS || process.env.VMLX_REAL_UI_MAX_TOOL_ITERATIONS || '4')
const toolResultMaxChars = Number(process.env.VMLINUX_REAL_UI_TOOL_RESULT_MAX_CHARS || process.env.VMLX_REAL_UI_TOOL_RESULT_MAX_CHARS || '12500')
const imageDataUrl = process.env.VMLINUX_REAL_UI_IMAGE_DATA_URL
  || process.env.VMLX_REAL_UI_IMAGE_DATA_URL
  || 'data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAIAAACQd1PeAAAADElEQVR4nGP4z8AAAAMBAQDJ/pLvAAAAAElFTkSuQmCC'
const imageExpectRegex = process.env.VMLINUX_REAL_UI_IMAGE_EXPECT_REGEX
  || process.env.VMLX_REAL_UI_IMAGE_EXPECT_REGEX
  || '\\bred\\b'
const videoDataUrl = process.env.VMLINUX_REAL_UI_VIDEO_DATA_URL
  || process.env.VMLX_REAL_UI_VIDEO_DATA_URL
  || ''
// Audio attachments are a real product capability — the renderer carries
// kind: 'audio' through InputBox/ChatInterface/chat-utils/MessageBubble — but
// this harness had NO audio path at all, so audio through the Electron UI had
// never been proven for any omni model.
const audioDataUrl = process.env.VMLINUX_REAL_UI_AUDIO_DATA_URL
  || process.env.VMLX_REAL_UI_AUDIO_DATA_URL
  || ''
const audioExpectRegex = process.env.VMLINUX_REAL_UI_AUDIO_EXPECT_REGEX
  || process.env.VMLX_REAL_UI_AUDIO_EXPECT_REGEX
  || ''
const videoExpectRegex = process.env.VMLINUX_REAL_UI_VIDEO_EXPECT_REGEX
  || process.env.VMLX_REAL_UI_VIDEO_EXPECT_REGEX
  || ''
const cacheExpectRegex = process.env.VMLINUX_REAL_UI_CACHE_EXPECT_REGEX
  || process.env.VMLX_REAL_UI_CACHE_EXPECT_REGEX
  || ''
const pairedApiHoldSeconds = Math.max(
  0,
  envNumber('VMLINUX_REAL_UI_PAIRED_API_HOLD_SECONDS') ?? 0,
)
const releaseSentinelTimeoutSeconds = Math.max(
  1,
  envNumber('VMLINUX_REAL_UI_RELEASE_TIMEOUT_SECONDS')
    ?? Math.max(pairedApiHoldSeconds, 900),
)
// How long to wait for the composer's Send control to become enabled before a
// turn. The default 30s is fine for ordinary bundles but too tight for a very
// large one still settling after a tool loop.
const sendReadyTimeoutMs = Math.max(
  5_000,
  envNumber('VMLINUX_REAL_UI_SEND_READY_TIMEOUT_MS') ?? 120_000,
)
const installedReleaseManifestPath = (
  process.env.VMLINUX_REAL_UI_RELEASE_MANIFEST
  || process.env.VMLX_REAL_UI_RELEASE_MANIFEST
  || ''
).trim()
const pairedApiArtifactPath = (
  process.env.VMLINUX_REAL_UI_PAIRED_API_ARTIFACT
  || process.env.VMLX_REAL_UI_PAIRED_API_ARTIFACT
  || ''
).trim()

const sleep = (ms) => new Promise((resolve) => setTimeout(resolve, ms))
const execFileAsync = promisify(execFile)

function sha256Text(value) {
  return crypto.createHash('sha256').update(String(value)).digest('hex')
}

function sha256Json(value) {
  return sha256Text(JSON.stringify(value))
}

function canonicalJson(value) {
  const normalize = (node) => {
    if (Array.isArray(node)) return node.map(normalize)
    if (!node || typeof node !== 'object') return node
    return Object.fromEntries(
      Object.keys(node)
        .sort()
        .map((key) => [key, normalize(node[key])]),
    )
  }
  // Match Python json.dumps(..., sort_keys=True, separators=(',', ':'),
  // ensure_ascii=True), which owns the API-v2 backend identity contract.
  return JSON.stringify(normalize(value)).replace(
    /[^\x20-\x7e]/g,
    (character) => `\\u${character.charCodeAt(0).toString(16).padStart(4, '0')}`,
  )
}

// Browser selectors are recorded after CSS.escape(), so validator-side
// ownership checks must use the same CSSOM serialization instead of comparing
// an escaped selector with a raw UUID.
export function cssEscapeIdentifier(value) {
  const string = String(value)
  const length = string.length
  let index = -1
  let output = ''
  const firstCodeUnit = string.charCodeAt(0)
  while (++index < length) {
    const codeUnit = string.charCodeAt(index)
    if (codeUnit === 0x0000) {
      output += '\uFFFD'
      continue
    }
    if (
      (codeUnit >= 0x0001 && codeUnit <= 0x001f)
      || codeUnit === 0x007f
      || (index === 0 && codeUnit >= 0x0030 && codeUnit <= 0x0039)
      || (
        index === 1
        && codeUnit >= 0x0030
        && codeUnit <= 0x0039
        && firstCodeUnit === 0x002d
      )
    ) {
      output += `\\${codeUnit.toString(16)} `
      continue
    }
    if (index === 0 && codeUnit === 0x002d && length === 1) {
      output += '\\-'
      continue
    }
    if (
      codeUnit >= 0x0080
      || codeUnit === 0x002d
      || codeUnit === 0x005f
      || (codeUnit >= 0x0030 && codeUnit <= 0x0039)
      || (codeUnit >= 0x0041 && codeUnit <= 0x005a)
      || (codeUnit >= 0x0061 && codeUnit <= 0x007a)
    ) {
      output += string.charAt(index)
      continue
    }
    output += `\\${string.charAt(index)}`
  }
  return output
}

function canonicalSha256(value) {
  return sha256Text(canonicalJson(value))
}

function validSha256(value) {
  return /^[0-9a-f]{64}$/i.test(String(value || ''))
}

function pythonCanonicalJsonMatchesParsed(canonicalText, parsedValue, expectedSha256) {
  if (typeof canonicalText !== 'string' || !validSha256(expectedSha256)) return false
  try {
    return (
      sha256Text(canonicalText) === expectedSha256
      && canonicalJson(JSON.parse(canonicalText)) === canonicalJson(parsedValue)
    )
  } catch {
    return false
  }
}

function sha256File(filePath) {
  return crypto.createHash('sha256').update(readFileSync(filePath)).digest('hex')
}

/**
 * The never-empty notices the panel substitutes when a turn produced no
 * renderable model answer (all-markup Responses payload, or a tool loop the
 * model never followed up on). Such a turn is a MODEL failure that must still
 * be legible to the user, so it is asserted non-empty, stream-equal to its
 * persisted record, and visibly rendered — but it carries no streamed model
 * answer, so the progressive-delta assertion cannot apply to it.
 *
 * Read out of the shared source instead of duplicated here: a copied string
 * would silently stop matching the moment the notice is reworded, and this
 * exemption must never widen by accident.
 */
function readNeverEmptyNotices() {
  const sourcePath = path.join(
    repoDir,
    'panel/src/shared/responsesStreamRecovery.ts',
  )
  const source = readFileSync(sourcePath, 'utf8')
  const notices = []
  for (const name of [
    'REJECTED_CONTROL_MARKUP_NOTICE',
    'TOOL_WITHOUT_ANSWER_NOTICE',
    'REASONING_WITHOUT_ANSWER_NOTICE',
  ]) {
    const match = source.match(
      new RegExp(`export const ${name}\\s*=\\s*\\n?\\s*"((?:[^"\\\\]|\\\\.)*)";`),
    )
    if (!match) {
      throw new Error(
        `Could not read ${name} from ${sourcePath}; the never-empty notice exemption would silently stop matching`,
      )
    }
    notices.push(JSON.parse(`"${match[1]}"`))
  }
  return notices
}

const NEVER_EMPTY_NOTICES = readNeverEmptyNotices()

function isNeverEmptyNoticeTurn(content) {
  const trimmed = String(content || '').trim()
  if (!trimmed) return false
  return NEVER_EMPTY_NOTICES.some((notice) => notice.trim() === trimmed)
}

function pythonSourceTreeDigest(root, relativeBase = repoDir) {
  const files = []
  const walk = (directory) => {
    for (const entry of readdirSync(directory, { withFileTypes: true })) {
      const fullPath = path.join(directory, entry.name)
      if (entry.isDirectory()) {
        walk(fullPath)
      } else if (entry.isFile() && entry.name.endsWith('.py')) {
        files.push(fullPath)
      }
    }
  }
  walk(root)
  files.sort()
  const digest = crypto.createHash('sha256')
  let fileCount = 0
  let readErrorCount = 0
  for (const filePath of files) {
    const relative = path.relative(relativeBase, filePath).split(path.sep).join('/')
    try {
      const contents = readFileSync(filePath)
      digest.update(relative)
      digest.update('\0')
      digest.update(contents)
      digest.update('\0')
      fileCount += 1
    } catch {
      digest.update(relative)
      digest.update('\0UNREADABLE\0')
      readErrorCount += 1
    }
  }
  return {
    python_source_tree_sha256: digest.digest('hex'),
    python_source_file_count: fileCount,
    python_source_read_error_count: readErrorCount,
  }
}

function sourceFilesDigest(roots) {
  const files = []
  const allowed = /\.(?:[cm]?[jt]sx?|json|css|html)$/
  const walk = (candidate) => {
    if (!existsSync(candidate)) return
    const stat = readdirSync(candidate, { withFileTypes: true })
    for (const entry of stat) {
      if (entry.name === 'node_modules' || entry.name === 'dist' || entry.name === 'out') continue
      const fullPath = path.join(candidate, entry.name)
      if (entry.isDirectory()) walk(fullPath)
      else if (entry.isFile() && allowed.test(entry.name)) files.push(fullPath)
    }
  }
  for (const root of roots) {
    if (!existsSync(root)) continue
    if (path.extname(root)) files.push(root)
    else walk(root)
  }
  files.sort()
  const digest = crypto.createHash('sha256')
  for (const filePath of files) {
    const relative = path.relative(repoDir, filePath).split(path.sep).join('/')
    digest.update(relative)
    digest.update('\0')
    digest.update(readFileSync(filePath))
    digest.update('\0')
  }
  return {
    renderer_source_tree_sha256: digest.digest('hex'),
    renderer_source_file_count: files.length,
  }
}

function objectRecord(value) {
  return value && typeof value === 'object' && !Array.isArray(value) ? value : undefined
}

function finiteNumber(value) {
  return typeof value === 'number' && Number.isFinite(value) ? value : undefined
}

function positiveInteger(value) {
  const number = finiteNumber(value)
  return number != null && number > 0 ? Math.floor(number) : undefined
}

export function resolveIndependentBundleGenerationDefaults(
  generationConfig,
  jangConfig,
  modelConfig,
) {
  const defaults = {}
  const generation = objectRecord(generationConfig)
  if (generation) {
    const samplingDisabled = generation.do_sample === false
    if (typeof generation.do_sample === 'boolean') defaults.doSample = generation.do_sample
    const temperature = finiteNumber(generation.temperature)
    if (temperature != null) defaults.temperature = samplingDisabled ? 0 : temperature
    const topP = finiteNumber(generation.top_p)
    if (topP != null) defaults.topP = samplingDisabled ? 1 : topP
    const topK = finiteNumber(generation.top_k)
    if (topK != null) defaults.topK = samplingDisabled ? 0 : Math.max(0, Math.round(topK))
    const minP = finiteNumber(generation.min_p)
    if (minP != null) defaults.minP = minP
    const repeatPenalty = finiteNumber(generation.repetition_penalty)
    if (repeatPenalty != null) defaults.repeatPenalty = repeatPenalty
    const maxNewTokens = positiveInteger(generation.max_new_tokens)
    if (maxNewTokens != null) defaults.maxNewTokens = maxNewTokens
    if (Object.keys(defaults).length) defaults.source = 'generation_config'
  }

  const jang = objectRecord(jangConfig)
  const sampling = objectRecord(jang?.chat?.sampling_defaults)
  if (sampling) {
    delete defaults.doSample
    for (const [sourceKey, targetKey] of [
      ['temperature', 'temperature'],
      ['top_p', 'topP'],
      ['min_p', 'minP'],
    ]) {
      const value = finiteNumber(sampling[sourceKey])
      if (value != null) defaults[targetKey] = value
    }
    const topK = finiteNumber(sampling.top_k)
    if (topK != null) defaults.topK = Math.max(0, Math.round(topK))
    const defaultMode = jang?.chat?.reasoning?.default_mode
    const repThinking = finiteNumber(sampling.repetition_penalty_thinking)
    const repChat = finiteNumber(sampling.repetition_penalty_chat)
    const repScalar = finiteNumber(sampling.repetition_penalty)
    const modelType = objectRecord(modelConfig)?.model_type
    const repeatPenalty = modelType === 'deepseek_v4'
      ? (defaultMode === 'thinking'
        ? (repThinking ?? repScalar ?? repChat)
        : (repScalar ?? repChat ?? repThinking))
      : defaultMode === 'thinking'
        ? (repThinking ?? repChat ?? repScalar)
        : (repChat ?? repThinking ?? repScalar)
    if (repeatPenalty != null) defaults.repeatPenalty = repeatPenalty
    const maxNewTokens = positiveInteger(sampling.max_new_tokens)
    if (maxNewTokens != null) defaults.maxNewTokens = maxNewTokens
    defaults.source = 'jang_config'
  }
  return Object.keys(defaults).some((key) => key !== 'source') ? defaults : null
}

function readOptionalJson(bundlePath, name) {
  const filePath = path.join(bundlePath, name)
  if (!existsSync(filePath)) {
    return {
      name,
      present: false,
      value: undefined,
      sha256: null,
      sizeBytes: null,
    }
  }
  const raw = readFileSync(filePath, 'utf8')
  return {
    name,
    present: true,
    value: JSON.parse(raw),
    sha256: sha256Text(raw),
    sizeBytes: Buffer.byteLength(raw),
  }
}

export function captureBundleGenerationContract(bundlePath) {
  const files = Object.fromEntries(
    ['config.json', 'generation_config.json', 'jang_config.json', 'tokenizer_config.json']
      .map((name) => {
        const record = readOptionalJson(bundlePath, name)
        return [name, record]
      }),
  )
  const templatePath = path.join(bundlePath, 'chat_template.jinja')
  const templateText = existsSync(templatePath) ? readFileSync(templatePath, 'utf8') : ''
  const tokenizerTemplate = files['tokenizer_config.json']?.value?.chat_template
  const includeStub = (
    typeof tokenizerTemplate === 'string'
    && /{%\s*include\s+['"][^'"]+['"]\s*%}/.test(tokenizerTemplate)
  )
  const usableTemplate = (
    (typeof tokenizerTemplate === 'string' && tokenizerTemplate.trim() && !includeStub)
    || (includeStub && templateText.trim())
    || templateText.trim()
  )
  const modelType = files['config.json']?.value?.model_type
  const chatContract = files['jang_config.json']?.value?.chat
  const nativeEncoder = {
    encoder: chatContract?.encoder,
    encoder_fn: chatContract?.encoder_fn,
    source: chatContract?.chat_template_source,
  }
  const usableNativeEncoder = Boolean(
    modelType === 'deepseek_v4'
    && nativeEncoder.encoder === 'encoding_dsv4'
    && nativeEncoder.encoder_fn === 'encode_messages'
    && nativeEncoder.source === 'official_python_encoder'
  )
  const usablePromptRenderer = Boolean(usableTemplate || usableNativeEncoder)
  const attestedFiles = {}
  for (const name of [
    'config.json',
    'generation_config.json',
    'jang_config.json',
    'tokenizer_config.json',
    'chat_template.jinja',
  ]) {
    if (name === 'chat_template.jinja') {
      const data = existsSync(templatePath) ? readFileSync(templatePath) : null
      attestedFiles[name] = data
        ? {
            state: 'present',
            size_bytes: data.length,
            sha256: crypto.createHash('sha256').update(data).digest('hex'),
          }
        : { state: 'missing' }
      continue
    }
    const record = files[name]
    attestedFiles[name] = record?.present
      ? {
          state: 'present',
          size_bytes: record.sizeBytes,
          sha256: record.sha256,
        }
      : { state: 'missing' }
  }
  const healthObserved = {
    schema: 'vmlx-bundle-config-v1',
    directory_state: 'available',
    files: attestedFiles,
  }
  const aggregateSha256 = canonicalSha256(healthObserved)
  return {
    bundle_path: realpathSync(bundlePath),
    files: Object.fromEntries(
      Object.entries(files).map(([name, record]) => [
        name,
        { present: record.present, sha256: record.sha256 },
      ]),
    ),
    defaults: resolveIndependentBundleGenerationDefaults(
      files['generation_config.json']?.value,
      files['jang_config.json']?.value,
      files['config.json']?.value,
    ),
    template: {
      tokenizer_chat_template_present:
        typeof tokenizerTemplate === 'string' && Boolean(tokenizerTemplate.trim()),
      tokenizer_chat_template_include_stub: includeStub,
      sidecar_present: Boolean(templateText.trim()),
      sidecar_sha256: templateText ? sha256Text(templateText) : null,
      mode: usableTemplate
        ? 'jinja'
        : usableNativeEncoder
          ? 'native_encoder'
          : null,
      native_encoder: nativeEncoder,
      usable: usablePromptRenderer,
      warning: usablePromptRenderer
        ? null
        : 'Bundle exposes neither a usable chat template nor a recognized native encoder contract',
    },
    health_attestation: {
      ...healthObserved,
      aggregate_sha256: aggregateSha256,
      fingerprint_sha256: aggregateSha256,
    },
  }
}

export function writePrivateArtifactFile(filePath, data) {
  writeFileSync(filePath, data, { flag: 'wx', mode: 0o600 })
  chmodSync(filePath, 0o600)
  return filePath
}

function readExternalFileBytes(
  filePath,
  label,
  {
    maxBytes = 64 * 1024 * 1024,
    requirePrivate = false,
    requireSingleLink = false,
    allowInsideRepo = true,
    retainRaw = true,
  } = {},
) {
  const absolute = path.resolve(filePath)
  const pathStat = lstatSync(absolute)
  if (!pathStat.isFile() || pathStat.isSymbolicLink()) {
    throw new Error(`${label} must be a regular, non-symlink file`)
  }
  if (requirePrivate && (pathStat.mode & 0o077) !== 0) {
    throw new Error(`${label} must not be group/world accessible (expected mode 0600)`)
  }
  if (requireSingleLink && pathStat.nlink !== 1) {
    throw new Error(`${label} must have exactly one filesystem link`)
  }
  if (pathStat.size > maxBytes) {
    throw new Error(`${label} exceeds the ${maxBytes}-byte safety limit`)
  }
  const canonical = realpathSync(absolute)
  if (!allowInsideRepo && isPathInside(canonical, realpathSync(repoDir))) {
    throw new Error(`${label} must stay outside the public Git worktree`)
  }
  const noFollow = Number(fsConstants.O_NOFOLLOW || 0)
  let fd
  let openedStat
  let raw
  let bytesRead = 0
  let sha256
  try {
    fd = openSync(canonical, fsConstants.O_RDONLY | noFollow)
    openedStat = fstatSync(fd)
    if (!openedStat.isFile()) {
      throw new Error(`${label} opened object is not a regular file`)
    }
    if (requireSingleLink && openedStat.nlink !== 1) {
      throw new Error(`${label} opened object must have exactly one filesystem link`)
    }
    if (requirePrivate && (openedStat.mode & 0o077) !== 0) {
      throw new Error(`${label} opened object is group/world accessible`)
    }
    if (
      openedStat.dev !== pathStat.dev
      || openedStat.ino !== pathStat.ino
      || openedStat.size !== pathStat.size
    ) {
      throw new Error(`${label} changed identity between path validation and open`)
    }
    if (openedStat.size > maxBytes) {
      throw new Error(`${label} exceeds the ${maxBytes}-byte safety limit`)
    }
    if (retainRaw) {
      raw = readFileSync(fd)
      bytesRead = raw.length
      sha256 = crypto.createHash('sha256').update(raw).digest('hex')
    } else {
      const digest = crypto.createHash('sha256')
      const chunk = Buffer.allocUnsafe(1024 * 1024)
      while (true) {
        const count = readSync(fd, chunk, 0, chunk.length, null)
        if (count === 0) break
        bytesRead += count
        if (bytesRead > openedStat.size) {
          throw new Error(`${label} grew while hashing`)
        }
        digest.update(chunk.subarray(0, count))
      }
      if (bytesRead !== openedStat.size) {
        throw new Error(`${label} changed size while hashing`)
      }
      sha256 = digest.digest('hex')
    }
    const afterReadStat = fstatSync(fd)
    if (
      afterReadStat.dev !== openedStat.dev
      || afterReadStat.ino !== openedStat.ino
      || afterReadStat.size !== openedStat.size
      || afterReadStat.mtimeMs !== openedStat.mtimeMs
      || afterReadStat.ctimeMs !== openedStat.ctimeMs
    ) {
      throw new Error(`${label} changed identity while reading`)
    }
  } finally {
    if (fd != null) closeSync(fd)
  }
  return {
    path: canonical,
    sha256,
    bytes: bytesRead,
    mode: openedStat.mode & 0o777,
    nlink: openedStat.nlink,
    opened_nofollow: true,
    ...(retainRaw ? { raw } : {}),
  }
}

function readExternalExecutableIdentity(filePath, label) {
  const absolute = path.resolve(filePath)
  // Python virtual environments deliberately expose python/python3/versioned
  // aliases as symlinks. Resolve that executable alias first, then retain the
  // existing regular-file + O_NOFOLLOW identity checks on the canonical target.
  // Private proof artifacts still call readExternalFileBytes directly and
  // therefore remain strictly non-symlink.
  const canonical = realpathSync(absolute)
  return readExternalFileBytes(canonical, label, {
    maxBytes: executableIdentityMaxBytes,
    retainRaw: false,
  })
}

function readPrivateExternalBytes(filePath, label, maxBytes = 64 * 1024 * 1024) {
  return readExternalFileBytes(filePath, label, {
    maxBytes,
    requirePrivate: true,
    requireSingleLink: true,
    allowInsideRepo: false,
  })
}

export function privateCacheAttestationSessionArgs(tokenFilePath) {
  const rawPath = String(tokenFilePath || '').trim()
  if (!rawPath) return ''
  if (!path.isAbsolute(rawPath)) {
    throw new Error('Private cache attestation token file must be absolute')
  }
  const opened = readPrivateExternalBytes(
    rawPath,
    'Private cache attestation token file',
    512,
  )
  if (opened.bytes < 32) {
    throw new Error('Private cache attestation token file is too small')
  }
  if (
    typeof process.getuid === 'function'
    && lstatSync(opened.path).uid !== process.getuid()
  ) {
    throw new Error('Private cache attestation token file has the wrong owner')
  }
  if ([...opened.raw].some((byte) => byte > 0x7f)) {
    throw new Error('Private cache attestation token file must contain ASCII')
  }
  const token = opened.raw.toString('ascii').trim()
  if (!/^[A-Za-z0-9_-]{32,512}$/.test(token)) {
    throw new Error('Private cache attestation token file has an invalid token')
  }
  if (!/^\/[A-Za-z0-9_./-]+$/.test(opened.path)) {
    throw new Error(
      'Private cache attestation token path contains characters unsupported '
      + 'by the Electron additional-argument transport',
    )
  }
  return '--enable-private-cache-attestation '
    + `--private-cache-attestation-token-file=${opened.path}`
}

export function readPrivateExternalJson(
  filePath,
  label,
  maxBytes = 64 * 1024 * 1024,
) {
  const opened = readPrivateExternalBytes(filePath, label, maxBytes)
  return {
    path: opened.path,
    sha256: opened.sha256,
    bytes: opened.bytes,
    mode: opened.mode,
    nlink: opened.nlink,
    opened_nofollow: opened.opened_nofollow,
    value: JSON.parse(opened.raw.toString('utf8')),
  }
}

const ownedRunIntentTopLevelFields = [
  'canonical_sha256',
  'created_at',
  'direct_base_url',
  'direct_health_url',
  'gateway_base_url',
  'gateway_health_url',
  'harnesses',
  'l2_size_eviction_requirements',
  'nonce',
  'native_direct_base_url',
  'native_direct_health_url',
  'phase_plan',
  'run_id',
  'schema',
  'source_commit',
  'source_tree',
]
const ownedRunIntentHarnessNames = ['api', 'cache', 'semantic', 'ui']
const ownedRunIntentHarnessFields = ['relative_path', 'sha256']
const ownedRunIntentL2RequirementFields = [
  'counter_only_evidence_allowed',
  'disk_bytes_within_saved_limit',
  'older_unused_prefix_eviction_required',
  'recent_target_survival_required',
  'restart_restore_required',
]
const ownedRunIntentPhaseFields = [
  'api_action_profile',
  'bundle_fingerprint_sha256',
  'bundle_role',
  'cache_policy',
  'kv_cache_quantization',
  'model',
  'model_bundle_path',
  'native_cache_policy',
  'operation',
  'paged_ram',
  'phase_index',
  'phase_name',
  'representative_id',
  'restart_required',
  'session_policy',
  'tq_policy',
  'ui_action_profile',
  'ui_turn_count',
]
const ownedRunIntentPhaseContract = [
  {
    phase_index: 0,
    phase_name: 'primary_ssd_only_store',
    representative_id: 'primary_tq_supported',
    bundle_role: 'primary',
    cache_policy: 'q4',
    paged_ram: false,
    operation: 'store',
    restart_required: false,
    session_policy: 'primary_stable_session',
    kv_cache_quantization: 'auto',
    tq_policy: 'auto-model-safe-required',
    ui_action_profile: 'primary-reasoning-render-store',
    ui_turn_count: 1,
    api_action_profile: 'full-agentic-plus-cache-store',
  },
  {
    phase_index: 1,
    phase_name: 'primary_ssd_only_restart_probe',
    representative_id: 'primary_tq_supported',
    bundle_role: 'primary',
    cache_policy: 'q4',
    paged_ram: false,
    operation: 'probe',
    restart_required: true,
    session_policy: 'primary_stable_session',
    kv_cache_quantization: 'auto',
    tq_policy: 'auto-model-safe-required',
    ui_action_profile: 'primary-tool-restart-probe',
    ui_turn_count: 1,
    api_action_profile: 'cache-probe',
  },
  {
    phase_index: 2,
    phase_name: 'primary_paged_on_store',
    representative_id: 'primary_tq_supported',
    bundle_role: 'primary',
    cache_policy: 'q4',
    paged_ram: true,
    operation: 'store-evict-refault',
    restart_required: true,
    session_policy: 'primary_stable_session',
    kv_cache_quantization: 'auto',
    tq_policy: 'auto-model-safe-required',
    ui_action_profile: 'primary-history-paged-evict-refault',
    ui_turn_count: 1,
    api_action_profile: 'cache-evict-refault',
  },
  {
    phase_index: 3,
    phase_name: 'primary_paged_on_restart_probe',
    representative_id: 'primary_tq_supported',
    bundle_role: 'primary',
    cache_policy: 'q4',
    paged_ram: true,
    operation: 'probe',
    restart_required: true,
    session_policy: 'primary_stable_session',
    kv_cache_quantization: 'auto',
    tq_policy: 'auto-model-safe-required',
    ui_action_profile: 'primary-restart-followup',
    ui_turn_count: 1,
    api_action_profile: 'cache-restart-probe',
  },
  {
    phase_index: 4,
    phase_name: 'primary_tq_off',
    representative_id: 'primary_tq_supported',
    bundle_role: 'primary',
    cache_policy: 'ssd-only',
    paged_ram: false,
    operation: 'store-probe',
    restart_required: true,
    session_policy: 'primary_stable_session',
    kv_cache_quantization: 'none',
    tq_policy: 'explicit-off',
    ui_action_profile: 'primary-tq-off-probe',
    ui_turn_count: 1,
    api_action_profile: 'cache-tq-off-store-probe',
  },
  {
    phase_index: 5,
    phase_name: 'native_exception',
    representative_id: 'secondary_native_exception',
    bundle_role: 'native',
    cache_policy: 'native',
    paged_ram: false,
    operation: 'switch-validate',
    restart_required: true,
    session_policy: 'distinct_native_session',
    kv_cache_quantization: 'none',
    tq_policy: 'native-suppressed',
    ui_action_profile: 'native-three-turn-switch',
    ui_turn_count: 3,
    api_action_profile: 'full-agentic-native-cache',
  },
]

function exactObjectFields(value, expected) {
  return value != null
    && !Array.isArray(value)
    && typeof value === 'object'
    && exactStringSet(Object.keys(value), expected)
}

function strictHttpUrl(value) {
  try {
    const parsed = new URL(String(value))
    if (!['http:', 'https:'].includes(parsed.protocol)) return null
    if (parsed.username || parsed.password || parsed.search || parsed.hash) return null
    return parsed
  } catch {
    return null
  }
}

function strictLoopbackHttpOrigin(value) {
  const parsed = strictHttpUrl(value)
  if (
    !parsed
    || parsed.protocol !== 'http:'
    || !['127.0.0.1', 'localhost', '[::1]'].includes(parsed.hostname)
    || !parsed.port
    || parsed.pathname !== '/'
    || value !== parsed.origin
  ) {
    return null
  }
  return parsed
}

function strictLoopbackHealthUrl(value, expectedOrigin) {
  const parsed = strictHttpUrl(value)
  if (
    !parsed
    || parsed.protocol !== 'http:'
    || !['127.0.0.1', 'localhost', '[::1]'].includes(parsed.hostname)
    || !parsed.port
    || parsed.origin !== expectedOrigin
    || parsed.pathname !== '/health'
    || value !== `${expectedOrigin}/health`
  ) {
    return null
  }
  return parsed
}

function ownedRunIntentDirectEndpoint(value, activePhaseIndex) {
  const native = activePhaseIndex === 5
  return {
    baseUrl: native ? value.native_direct_base_url : value.direct_base_url,
    healthUrl: native
      ? value.native_direct_health_url
      : value.direct_health_url,
  }
}

export function validateAttachOnlyLifecycle({
  cdpUrl,
  electronPid,
  owner,
  teardownAllowed,
} = {}) {
  const failures = []
  const parsed = strictHttpUrl(cdpUrl)
  if (
    !parsed
    || !['127.0.0.1', 'localhost', '[::1]'].includes(parsed.hostname)
    || !parsed.port
    || parsed.pathname !== '/'
    || cdpUrl !== parsed.origin
  ) {
    failures.push('attach-only CDP URL is not an exact loopback origin')
  }
  if (!Number.isInteger(electronPid) || electronPid <= 0) {
    failures.push('attach-only Electron PID is invalid')
  }
  if (owner !== 'parent') {
    failures.push('attach-only Electron lifecycle is not parent-owned')
  }
  if (teardownAllowed !== false) {
    failures.push('attach-only child is allowed to tear down the parent lifecycle')
  }
  return failures
}

function canonicalSha256WithoutField(value, omittedField) {
  const copy = { ...value }
  delete copy[omittedField]
  return canonicalSha256(copy)
}

export function validateOwnedRunIntent(
  opened,
  {
    runId: expectedRunId,
    nonce,
    expectedSha256,
    expectedSourceCommit,
    expectedSourceTree,
    expectedUiHarnessSha256,
    harnessRoot = repoDir,
    activePhaseIndex,
    activeModel,
    activeModelBundlePath,
    expectedDirectBaseUrl,
    expectedGatewayBaseUrl,
  } = {},
) {
  const failures = []
  const value = opened?.value
  if (
    opened?.opened_nofollow !== true
    || opened?.nlink !== 1
    || opened?.mode !== 0o600
  ) {
    failures.push('owned run intent was not safely opened as a private single-link file')
  }
  if (!validSha256(expectedSha256) || opened?.sha256 !== expectedSha256) {
    failures.push('owned run intent exact file hash does not match launch contract')
  }
  if (!exactObjectFields(value, ownedRunIntentTopLevelFields)) {
    failures.push('owned run intent top-level fields are missing or unexpected')
    return failures
  }
  if (
    value.schema !== ownedRunIntentSchema
    || value.run_id !== expectedRunId
    || value.nonce !== nonce
  ) {
    failures.push('owned run intent schema/run/nonce does not match')
  }
  if (
    !validSha256(value.canonical_sha256)
    || value.canonical_sha256
      !== canonicalSha256WithoutField(value, 'canonical_sha256')
  ) {
    failures.push('owned run intent canonical_sha256 is invalid')
  }
  if (
    !/^[0-9a-f]{40,64}$/i.test(String(value.source_commit || ''))
    || !/^[0-9a-f]{40,64}$/i.test(String(value.source_tree || ''))
    || (
      expectedSourceCommit
      && value.source_commit !== expectedSourceCommit
    )
    || (
      expectedSourceTree
      && value.source_tree !== expectedSourceTree
    )
  ) {
    failures.push('owned run intent source commit/tree does not match')
  }
  const createdAtMs = Date.parse(String(value.created_at || ''))
  if (!Number.isFinite(createdAtMs)) {
    failures.push('owned run intent created_at is invalid')
  }
  if (
    !exactObjectFields(
      value.l2_size_eviction_requirements,
      ownedRunIntentL2RequirementFields,
    )
    || value.l2_size_eviction_requirements.disk_bytes_within_saved_limit !== true
    || value.l2_size_eviction_requirements
      .older_unused_prefix_eviction_required !== true
    || value.l2_size_eviction_requirements
      .recent_target_survival_required !== true
    || value.l2_size_eviction_requirements.restart_restore_required !== true
    || value.l2_size_eviction_requirements.counter_only_evidence_allowed !== false
  ) {
    failures.push('owned run intent L2 size-eviction requirements are invalid')
  }

  if (!exactObjectFields(value.harnesses, ownedRunIntentHarnessNames)) {
    failures.push('owned run intent harness set is incomplete or unexpected')
  } else {
    for (const harnessName of ownedRunIntentHarnessNames) {
      const harness = value.harnesses[harnessName]
      if (
        !exactObjectFields(harness, ownedRunIntentHarnessFields)
        || !validSha256(harness?.sha256)
        || typeof harness?.relative_path !== 'string'
        || !harness.relative_path
        || path.isAbsolute(harness.relative_path)
        || harness.relative_path.split(/[\\/]/).includes('..')
      ) {
        failures.push(`owned run intent ${harnessName} harness binding is invalid`)
        continue
      }
      try {
        const canonicalRoot = realpathSync(harnessRoot)
        const lexicalPath = path.resolve(canonicalRoot, harness.relative_path)
        const pathStat = lstatSync(lexicalPath)
        const canonicalPath = realpathSync(lexicalPath)
        if (
          !isPathInside(lexicalPath, canonicalRoot)
          || !isPathInside(canonicalPath, canonicalRoot)
          || !pathStat.isFile()
          || pathStat.isSymbolicLink()
          || sha256File(canonicalPath) !== harness.sha256
        ) {
          failures.push(
            `owned run intent ${harnessName} harness bytes/path do not match`,
          )
        }
      } catch {
        failures.push(
          `owned run intent ${harnessName} harness bytes/path cannot be verified`,
        )
      }
    }
    if (
      expectedUiHarnessSha256
      && value.harnesses?.ui?.sha256 !== expectedUiHarnessSha256
    ) {
      failures.push('owned run intent UI harness hash does not match executing source')
    }
  }

  const directBase = strictLoopbackHttpOrigin(value.direct_base_url)
  const nativeDirectBase = strictLoopbackHttpOrigin(
    value.native_direct_base_url,
  )
  const gatewayBase = strictLoopbackHttpOrigin(value.gateway_base_url)
  const directHealth = strictLoopbackHealthUrl(
    value.direct_health_url,
    directBase?.origin,
  )
  const nativeDirectHealth = strictLoopbackHealthUrl(
    value.native_direct_health_url,
    nativeDirectBase?.origin,
  )
  const gatewayHealth = strictLoopbackHealthUrl(
    value.gateway_health_url,
    gatewayBase?.origin,
  )
  const activeDirect = ownedRunIntentDirectEndpoint(value, activePhaseIndex)
  if (
    !directBase
    || !nativeDirectBase
    || !gatewayBase
    || !directHealth
    || !nativeDirectHealth
    || !gatewayHealth
    || directBase.origin === gatewayBase.origin
    || directBase.origin === nativeDirectBase.origin
    || nativeDirectBase.origin === gatewayBase.origin
    || (
      expectedDirectBaseUrl
      && activeDirect.baseUrl
        !== strictLoopbackHttpOrigin(expectedDirectBaseUrl)?.origin
    )
    || (
      expectedGatewayBaseUrl
      && gatewayBase.origin
        !== strictLoopbackHttpOrigin(expectedGatewayBaseUrl)?.origin
    )
  ) {
    failures.push(
      'owned run intent primary/native/gateway endpoint binding is invalid',
    )
  }

  if (
    !Array.isArray(value.phase_plan)
    || value.phase_plan.length !== ownedRunIntentPhaseContract.length
  ) {
    failures.push('owned run intent must contain the exact six ordered phases')
  } else {
    for (let index = 0; index < ownedRunIntentPhaseContract.length; index += 1) {
      const phase = value.phase_plan[index]
      const expected = ownedRunIntentPhaseContract[index]
      if (!exactObjectFields(phase, ownedRunIntentPhaseFields)) {
        failures.push(`owned run intent phase ${index} fields are missing or unexpected`)
        continue
      }
      if (
        Object.entries(expected).some(([field, expectedValue]) => (
          phase[field] !== expectedValue
        ))
      ) {
        failures.push(`owned run intent phase ${index} policy/order does not match`)
      }
      if (
        typeof phase.model !== 'string'
        || !phase.model
        || typeof phase.model_bundle_path !== 'string'
        || !path.isAbsolute(phase.model_bundle_path)
        || !validSha256(phase.bundle_fingerprint_sha256)
        || typeof phase.native_cache_policy !== 'string'
        || !phase.native_cache_policy
        || phase.native_cache_policy === 'generic-kv-tq'
      ) {
        failures.push(`owned run intent phase ${index} model/bundle policy is invalid`)
      }
    }
    const primary = value.phase_plan[0]
    for (let index = 1; index < 5; index += 1) {
      const phase = value.phase_plan[index]
      if (
        phase.model !== primary.model
        || phase.model_bundle_path !== primary.model_bundle_path
        || phase.bundle_fingerprint_sha256 !== primary.bundle_fingerprint_sha256
        || phase.native_cache_policy !== primary.native_cache_policy
      ) {
        failures.push('owned run intent primary phases do not bind one stable bundle')
        break
      }
    }
    const native = value.phase_plan[5]
    if (
      native.model_bundle_path === primary.model_bundle_path
      || native.bundle_fingerprint_sha256 === primary.bundle_fingerprint_sha256
      || native.native_cache_policy === primary.native_cache_policy
    ) {
      failures.push('owned run intent native-exception representative is not distinct')
    }
  }

  if (
    !Number.isInteger(activePhaseIndex)
    || activePhaseIndex < 0
    || activePhaseIndex >= ownedRunIntentPhaseContract.length
  ) {
    failures.push('owned run intent active phase index is invalid')
  } else if (Array.isArray(value.phase_plan)) {
    const active = value.phase_plan[activePhaseIndex]
    let canonicalActiveBundle = null
    let canonicalRequestedBundle = null
    try {
      canonicalActiveBundle = realpathSync(active?.model_bundle_path || '')
      canonicalRequestedBundle = realpathSync(activeModelBundlePath || '')
    } catch {
      failures.push('owned run intent active model bundle cannot be resolved')
    }
    if (
      active?.model !== activeModel
      || (
        canonicalActiveBundle
        && canonicalRequestedBundle
        && canonicalActiveBundle !== canonicalRequestedBundle
      )
    ) {
      failures.push('owned run intent active model/bundle does not match UI launch')
    }
  }
  return failures
}

export function validateOwnedUiReleaseSentinel(
  opened,
  {
    runId: expectedRunId,
    nonce,
    sessionId,
    runIntentSha256,
    uiSessionAttestationSha256,
    activePhase,
    orchestrated = false,
    notBeforeMs,
  },
) {
  const failures = []
  const value = opened?.value || {}
  const expectedFields = [
    'api_capture_sha256',
    'api_action_profile',
    'cache_capture_sha256',
    'bundle_fingerprint_sha256',
    'bundle_role',
    'cache_policy',
    'model',
    'nonce',
    'paged_ram',
    'phase_index',
    'phase_name',
    'released_at',
    'representative_id',
    'run_id',
    'run_intent_sha256',
    'schema',
    'session_id',
    'ui_session_attestation_sha256',
    'ui_action_profile',
    'ui_turn_count',
  ]
  if (
    opened?.opened_nofollow !== true
    || opened?.nlink !== 1
    || opened?.mode !== 0o600
  ) {
    failures.push('owned UI release sentinel was not safely opened as a private single-link file')
  }
  if (
    !exactObjectFields(value, expectedFields)
    ||
    value.schema !== ownedUiReleaseSchema
    || value.run_id !== expectedRunId
    || value.nonce !== nonce
    || String(value.session_id || '') !== String(sessionId || '')
  ) {
    failures.push('owned UI release sentinel fields/run/nonce/session do not match')
  }
  if (
    orchestrated
    && (
      !activePhase
      || value.phase_index !== activePhase.phase_index
      || value.phase_name !== activePhase.phase_name
      || value.representative_id !== activePhase.representative_id
      || value.bundle_role !== activePhase.bundle_role
      || value.cache_policy !== activePhase.cache_policy
      || value.paged_ram !== activePhase.paged_ram
      || value.model !== activePhase.model
      || value.bundle_fingerprint_sha256
        !== activePhase.bundle_fingerprint_sha256
      || value.ui_action_profile !== activePhase.ui_action_profile
      || value.ui_turn_count !== activePhase.ui_turn_count
      || value.api_action_profile !== activePhase.api_action_profile
    )
  ) {
    failures.push('owned UI release sentinel active phase does not match run intent')
  }
  if (!validSha256(value.bundle_fingerprint_sha256)) {
    failures.push('owned UI release sentinel bundle_fingerprint_sha256 is invalid')
  }
  for (const field of [
    'run_intent_sha256',
    'ui_session_attestation_sha256',
    'api_capture_sha256',
    'cache_capture_sha256',
  ]) {
    if (!validSha256(value[field])) {
      failures.push(`owned UI release sentinel ${field} is invalid`)
    }
  }
  if (orchestrated) {
    if (!validSha256(runIntentSha256)) {
      failures.push(
        'owned UI release expected run-intent fingerprint is invalid',
      )
    }
    if (
      value.run_intent_sha256 !== runIntentSha256
    ) {
      failures.push(
        'owned UI release sentinel run_intent_sha256 is missing or unbound',
      )
    }
    if (
      !validSha256(uiSessionAttestationSha256)
      || value.ui_session_attestation_sha256
        !== uiSessionAttestationSha256
    ) {
      failures.push(
        'owned UI release sentinel ui_session_attestation_sha256 is missing or unbound',
      )
    }
  }
  const releasedAtMs = Date.parse(String(value.released_at || ''))
  if (!Number.isFinite(releasedAtMs) || releasedAtMs < Number(notBeforeMs || 0)) {
    failures.push('owned UI release sentinel is stale or predates the bound UI session')
  }
  return failures
}

export function validateOwnedReuseSessionAttestation(
  opened,
  {
    runId: expectedRunId,
    nonce,
    runIntentSha256,
    sessionId,
    activePhase,
    model,
    modelBundlePath,
    electronPid,
    cdpOrigin,
    gatewayPid,
    gatewayBaseUrl,
    sourceCommit,
    sourceTree,
  },
) {
  const failures = []
  const value = opened?.value || {}
  if (
    opened?.opened_nofollow !== true
    || opened?.nlink !== 1
    || opened?.mode !== 0o600
  ) {
    failures.push('reused UI session attestation was not safely opened')
  }
  if (
    !activePhase
    || ![1, 2, 3, 4].includes(activePhase.phase_index)
    || value.schema !== ownedUiSessionAttestationSchema
    || value.run_id !== expectedRunId
    || value.nonce !== nonce
    || value.run_intent_sha256 !== runIntentSha256
    || value.phase_index !== activePhase.phase_index - 1
    || value.representative_id !== activePhase.representative_id
    || value.session_id !== sessionId
    || value.model !== model
    || value.bundle_fingerprint_sha256
      !== activePhase.bundle_fingerprint_sha256
    || value.electron_pid !== electronPid
    || value.cdp_origin !== cdpOrigin
    || value.gateway_pid !== gatewayPid
    || value.gateway_base_url !== gatewayBaseUrl
    || value.lifecycle_owner !== 'parent'
    || value.source_commit !== sourceCommit
    || value.source_tree !== sourceTree
  ) {
    failures.push(
      'reused UI session attestation is stale, wrong-phase, wrong-model, '
      + 'or not owned by this parent lifecycle',
    )
  }
  try {
    if (
      realpathSync(value.model_bundle_path || '')
      !== realpathSync(modelBundlePath || '')
    ) {
      failures.push('reused UI session attestation model path does not match')
    }
  } catch {
    failures.push('reused UI session attestation model path is not resolvable')
  }
  if (!Number.isFinite(Date.parse(String(value.created_at || '')))) {
    failures.push('reused UI session attestation timestamp is invalid')
  }
  return failures
}

export async function waitForOwnedUiReleaseSentinel({
  filePath,
  runId: expectedRunId,
  nonce,
  sessionId,
  orchestrated = true,
  runIntentPath,
  runIntentSha256,
  uiSessionAttestationPath,
  uiSessionAttestationSha256,
  activePhase,
  apiArtifactPath,
  cacheArtifactPath,
  notBeforeMs,
  timeoutMs,
  pollMs = 100,
}) {
  if (!filePath || !nonce) {
    throw new Error('owned UI release sentinel requires both path and nonce')
  }
  if (
    orchestrated
    && (
      !runIntentPath
      || !validSha256(runIntentSha256)
      || !uiSessionAttestationPath
      || !validSha256(uiSessionAttestationSha256)
      || !apiArtifactPath
      || !cacheArtifactPath
    )
  ) {
    throw new Error(
      'orchestrated owned UI release requires exact run-intent and UI-session '
      + 'attestation path/hash bindings plus paired API/cache artifact paths',
    )
  }
  const absolute = path.resolve(filePath)
  if (existsSync(absolute)) {
    throw new Error('owned UI release sentinel existed before the UI hold became ready')
  }
  const deadline = Date.now() + timeoutMs
  while (Date.now() < deadline) {
    if (!existsSync(absolute)) {
      await sleep(pollMs)
      continue
    }
    const opened = readPrivateExternalJson(absolute, 'Owned UI release sentinel')
    const failures = validateOwnedUiReleaseSentinel(opened, {
      runId: expectedRunId,
      nonce,
      sessionId,
      orchestrated,
      runIntentSha256,
      uiSessionAttestationSha256,
      activePhase,
      notBeforeMs,
    })
    if (failures.length) {
      throw new Error(failures.join('; '))
    }
    let apiCapture = null
    let cacheCapture = null
    let runIntent = null
    let uiSessionAttestation = null
    if (orchestrated) {
      runIntent = readPrivateExternalBytes(
        runIntentPath,
        'Owned UI run intent',
      )
      uiSessionAttestation = readPrivateExternalBytes(
        uiSessionAttestationPath,
        'Owned UI session attestation',
      )
      apiCapture = readPrivateExternalBytes(
        apiArtifactPath,
        'Owned UI paired API artifact',
      )
      cacheCapture = readPrivateExternalBytes(
        cacheArtifactPath,
        'Owned UI paired cache artifact',
      )
      if (
        runIntent.path === uiSessionAttestation.path
        || runIntent.path === apiCapture.path
        || runIntent.path === cacheCapture.path
        || runIntent.path === opened.path
        || uiSessionAttestation.path === apiCapture.path
        || uiSessionAttestation.path === cacheCapture.path
        || uiSessionAttestation.path === opened.path
        ||
        apiCapture.path === cacheCapture.path
        || apiCapture.path === opened.path
        || cacheCapture.path === opened.path
      ) {
        throw new Error(
          'owned UI release sentinel and paired artifacts are not distinct files',
        )
      }
      if (
        runIntent.sha256 !== opened.value.run_intent_sha256
        || uiSessionAttestation.sha256
          !== opened.value.ui_session_attestation_sha256
        ||
        apiCapture.sha256 !== opened.value.api_capture_sha256
        || cacheCapture.sha256 !== opened.value.cache_capture_sha256
      ) {
        throw new Error(
          'owned UI release sentinel run-intent/UI-session/API/cache hashes '
          + 'do not match safely reopened exact artifacts',
        )
      }
    }
    return {
      path: opened.path,
      sha256: opened.sha256,
      schema: opened.value.schema,
      run_id: opened.value.run_id,
      nonce: opened.value.nonce,
      session_id: opened.value.session_id,
      phase_index: opened.value.phase_index,
      phase_name: opened.value.phase_name,
      representative_id: opened.value.representative_id,
      bundle_role: opened.value.bundle_role,
      cache_policy: opened.value.cache_policy,
      paged_ram: opened.value.paged_ram,
      model: opened.value.model,
      bundle_fingerprint_sha256:
        opened.value.bundle_fingerprint_sha256,
      ui_action_profile: opened.value.ui_action_profile,
      ui_turn_count: opened.value.ui_turn_count,
      api_action_profile: opened.value.api_action_profile,
      run_intent_sha256: opened.value.run_intent_sha256,
      ui_session_attestation_sha256:
        opened.value.ui_session_attestation_sha256,
      api_capture_sha256: opened.value.api_capture_sha256,
      cache_capture_sha256: opened.value.cache_capture_sha256,
      run_intent_path: runIntent?.path || null,
      run_intent_bytes: runIntent?.bytes || null,
      ui_session_attestation_path: uiSessionAttestation?.path || null,
      ui_session_attestation_bytes: uiSessionAttestation?.bytes || null,
      api_capture_path: apiCapture?.path || null,
      api_capture_bytes: apiCapture?.bytes || null,
      cache_capture_path: cacheCapture?.path || null,
      cache_capture_bytes: cacheCapture?.bytes || null,
      released_at: opened.value.released_at,
    }
  }
  throw new Error('timed out waiting for owned UI release sentinel')
}

function isPathInside(candidate, parent) {
  const relative = path.relative(parent, candidate)
  return relative === '' || (!relative.startsWith(`..${path.sep}`) && relative !== '..' && !path.isAbsolute(relative))
}

function nearestExistingDirectory(candidate) {
  let current = path.resolve(candidate)
  while (!existsSync(current)) {
    const parent = path.dirname(current)
    if (parent === current) break
    current = parent
  }
  return current
}

async function resolvePrivateProofDir() {
  if (!proofDirInput.trim()) {
    throw new Error(
      'Set VMLINUX_REAL_UI_PROOF_DIR, VMLX_REAL_UI_PROOF_DIR, or VMLX_PRIVATE_EVIDENCE_ROOT '
      + 'to a private directory outside every Git worktree',
    )
  }
  if (!proofBasename || path.basename(proofBasename) !== proofBasename || ['.', '..'].includes(proofBasename)) {
    throw new Error('VMLINUX_REAL_UI_PROOF_BASENAME must be a single safe file basename')
  }

  const requested = path.resolve(proofDirInput)
  const canonicalRepo = realpathSync(repoDir)
  const existingAncestor = realpathSync(nearestExistingDirectory(requested))
  if (isPathInside(requested, canonicalRepo) || isPathInside(existingAncestor, canonicalRepo)) {
    throw new Error('Real UI proof output must stay outside the public vMLX Git worktree')
  }

  mkdirSync(requested, { recursive: true })
  const canonical = realpathSync(requested)
  chmodSync(canonical, 0o700)
  if (isPathInside(canonical, canonicalRepo)) {
    throw new Error('Real UI proof output resolved inside the public vMLX Git worktree')
  }
  try {
    const { stdout } = await execFileAsync(
      'git',
      ['-C', canonical, 'rev-parse', '--show-toplevel'],
      { encoding: 'utf8' },
    )
    const gitRoot = stdout.trim()
    if (gitRoot) {
      throw new Error(`Real UI proof output must stay outside Git; resolved Git root: ${gitRoot}`)
    }
  } catch (error) {
    if (String(error?.message || '').startsWith('Real UI proof output must stay outside Git;')) {
      throw error
    }
    // git exits nonzero when the path is correctly outside every worktree.
  }
  return canonical
}

async function captureGitProvenance() {
  const runGit = async (args, fallback = '') => {
    try {
      const { stdout } = await execFileAsync('git', ['-C', repoDir, ...args], { encoding: 'utf8' })
      return stdout.trim()
    } catch {
      return fallback
    }
  }
  const [commit, tree, branch, upstream, status] = await Promise.all([
    runGit(['rev-parse', 'HEAD']),
    runGit(['rev-parse', 'HEAD^{tree}']),
    runGit(['branch', '--show-current']),
    runGit(['rev-parse', '--abbrev-ref', '--symbolic-full-name', '@{upstream}']),
    runGit(['status', '--porcelain=v1', '--untracked-files=all']),
  ])
  const sourceTree = pythonSourceTreeDigest(path.join(repoDir, 'vmlx_engine'))
  const rendererTree = sourceFilesDigest([
    path.join(panelDir, 'src'),
    path.join(panelDir, 'package.json'),
    path.join(panelDir, 'package-lock.json'),
    path.join(panelDir, 'electron.vite.config.ts'),
  ])
  return {
    observed_at: new Date().toISOString(),
    git_root: realpathSync(repoDir),
    branch,
    upstream: upstream || null,
    commit,
    tree,
    dirty: Boolean(status),
    status_porcelain: status ? status.split(/\r?\n/) : [],
    ...sourceTree,
    ...rendererTree,
    server_module_sha256: sha256File(path.join(repoDir, 'vmlx_engine', 'server.py')),
    package_init_sha256: sha256File(path.join(repoDir, 'vmlx_engine', '__init__.py')),
    harness_sha256: sha256Text(readFileSync(new URL(import.meta.url), 'utf8')),
  }
}

function readExternalReleaseManifest(manifestPath) {
  if (!manifestPath) return null
  const opened = readPrivateExternalJson(
    manifestPath,
    'Installed release manifest',
    1024 * 1024,
  )
  return {
    path: opened.path,
    sha256: opened.sha256,
    value: opened.value,
  }
}

async function listenerPidForPort(port) {
  const { stdout } = await execFileAsync(
    'lsof',
    ['-nP', `-iTCP:${port}`, '-sTCP:LISTEN', '-Fp'],
    { encoding: 'utf8' },
  )
  const pids = [...new Set(
    String(stdout)
      .split(/\r?\n/)
      .filter((line) => /^p\d+$/.test(line))
      .map((line) => Number(line.slice(1)))
      .filter((pid) => Number.isInteger(pid) && pid > 0),
  )]
  if (pids.length !== 1) {
    throw new Error(`expected exactly one TCP listener on ${port}, got ${pids.join(',') || 'none'}`)
  }
  return pids[0]
}

async function listenerPidsForPort(port) {
  let stdout = ''
  try {
    ;({ stdout } = await execFileAsync(
      'lsof',
      ['-nP', `-iTCP:${port}`, '-sTCP:LISTEN', '-Fp'],
      { encoding: 'utf8' },
    ))
  } catch (error) {
    if (Number(error?.code) === 1 && !String(error?.stdout || '').trim()) {
      return []
    }
    throw error
  }
  return [...new Set(
    String(stdout)
      .split(/\r?\n/)
      .filter((line) => /^p\d+$/.test(line))
      .map((line) => Number(line.slice(1)))
      .filter((pid) => Number.isInteger(pid) && pid > 0),
  )]
}

async function exactPidIsAlive(pid) {
  let stdout = ''
  try {
    ;({ stdout } = await execFileAsync(
      'ps',
      ['-p', String(Number(pid)), '-o', 'pid='],
      { encoding: 'utf8' },
    ))
  } catch (error) {
    if (Number(error?.code) === 1 && !String(error?.stdout || '').trim()) {
      return false
    }
    throw error
  }
  return String(stdout)
    .split(/\s+/)
    .filter(Boolean)
    .some((value) => Number(value) === Number(pid))
}

async function waitForExactProofBackendTeardown({
  backendPid,
  port,
  timeoutMs = 120_000,
  pollMs = 100,
}) {
  const expectedPid = Number(backendPid)
  const expectedPort = Number(port)
  if (
    !Number.isInteger(expectedPid)
    || expectedPid <= 0
    || !Number.isInteger(expectedPort)
    || expectedPort <= 0
  ) {
    throw new Error('Final proof-session teardown requires one exact backend PID and port')
  }
  const started = Date.now()
  let lastAlive = true
  let lastListenerPids = []
  while (Date.now() - started < timeoutMs) {
    lastAlive = await exactPidIsAlive(expectedPid)
    lastListenerPids = await listenerPidsForPort(expectedPort)
    const unexpected = lastListenerPids.filter((pid) => pid !== expectedPid)
    if (unexpected.length) {
      throw new Error(
        `Final proof port ${expectedPort} was rebound by unexpected PID(s): ${unexpected.join(',')}`,
      )
    }
    if (!lastAlive && lastListenerPids.length === 0) {
      return {
        backend_pid: expectedPid,
        port: expectedPort,
        backend_process_gone: true,
        listener_gone: true,
        observed_listener_pids: [],
        elapsed_ms: Date.now() - started,
      }
    }
    await sleep(pollMs)
  }
  throw new Error(
    'Timed out waiting for the exact proof backend to stop: '
    + `pid=${expectedPid} alive=${lastAlive} port=${expectedPort} `
    + `listeners=${lastListenerPids.join(',') || 'none'}`,
  )
}

export async function runPostSentinelWorkWithCleanup({
  work,
  cleanup,
}) {
  let value
  let originalError = null
  let cleanupError = null
  try {
    value = await work()
  } catch (error) {
    originalError = error
  } finally {
    try {
      await cleanup()
    } catch (error) {
      cleanupError = error
    }
  }
  if (originalError) {
    if (cleanupError && typeof originalError === 'object') {
      try {
        originalError.cleanupError = cleanupError
      } catch {}
    }
    throw originalError
  }
  if (cleanupError) throw cleanupError
  return value
}

async function attestExactSurvivorPids({
  backendPid,
  electronPid,
  gatewayPid,
  retainedPids,
  stage,
}) {
  const expected = [
    { role: 'parent_electron', pid: Number(electronPid) },
    { role: 'parent_gateway', pid: Number(gatewayPid) },
    ...retainedPids.map((pid, index) => ({
      role: `explicit_retained_${index + 1}`,
      pid: Number(pid),
    })),
  ]
  const pids = expected.map((entry) => entry.pid)
  const explicitPids = retainedPids.map(Number)
  const parentPid = Number(electronPid)
  const proofBackendPid = Number(backendPid)
  if (
    expected.some((entry) => !Number.isInteger(entry.pid) || entry.pid <= 1)
    || !Number.isInteger(proofBackendPid)
    || proofBackendPid <= 1
    || parentPid !== Number(gatewayPid)
    || new Set(explicitPids).size !== explicitPids.length
    || explicitPids.includes(parentPid)
    || pids.includes(proofBackendPid)
  ) {
    throw new Error(
      `Final proof-session ${stage} survivor attestation requires one shared `
      + 'Electron/gateway PID and disjoint explicit/backend PIDs',
    )
  }
  const observed = await Promise.all(expected.map(async (entry) => ({
    ...entry,
    alive: await exactPidIsAlive(entry.pid),
  })))
  const missing = observed.filter((entry) => entry.alive !== true)
  if (missing.length) {
    throw new Error(
      `Final proof-session ${stage} survivor PID(s) are not alive: `
      + missing.map((entry) => `${entry.role}=${entry.pid}`).join(','),
    )
  }
  return {
    stage,
    expected_retained_pids: [...retainedPids],
    processes: observed,
  }
}

async function executablePathForPid(pid) {
  try {
    const { stdout: commandLine } = await execFileAsync(
      'ps',
      ['-ww', '-p', String(pid), '-o', 'command='],
      { encoding: 'utf8' },
    )
    const match = String(commandLine).trim().match(/^(?:"([^"]+)"|'([^']+)'|(\S+))/)
    const invokedPath = match?.[1] || match?.[2] || match?.[3] || ''
    if (path.isAbsolute(invokedPath) && existsSync(invokedPath)) {
      return path.resolve(invokedPath)
    }
  } catch {}
  const { stdout } = await execFileAsync(
    'lsof',
    ['-a', '-p', String(pid), '-d', 'txt', '-Fn'],
    { encoding: 'utf8' },
  )
  const candidates = String(stdout)
    .split(/\r?\n/)
    .filter((line) => line.startsWith('n/'))
    .map((line) => line.slice(1))
    .filter((candidate) => existsSync(candidate))
  if (!candidates.length) {
    throw new Error(`could not resolve executable text mapping for PID ${pid}`)
  }
  return path.resolve(candidates[0])
}

async function captureListenerProcessBinding({
  port,
  expectedRootPid,
  expectedHealthPid,
  kind,
}) {
  const listenerPid = await listenerPidForPort(port)
  const descendants = expectedRootPid ? await childProcessTree(expectedRootPid) : []
  const belongsToRoot = !expectedRootPid
    || listenerPid === expectedRootPid
    || descendants.includes(listenerPid)
  if (!belongsToRoot) {
    throw new Error(
      `${kind} listener PID ${listenerPid} is not the launched PID ${expectedRootPid} or its descendant`,
    )
  }
  if (expectedHealthPid && listenerPid !== Number(expectedHealthPid)) {
    throw new Error(
      `${kind} listener PID ${listenerPid} does not match /health PID ${expectedHealthPid}`,
    )
  }
  const executablePath = await executablePathForPid(listenerPid)
  const executableIdentity = readExternalExecutableIdentity(
    executablePath,
    `${kind} listener executable`,
  )
  return {
    kind,
    port,
    launched_root_pid: expectedRootPid || null,
    process_tree_pids: expectedRootPid
      ? [expectedRootPid, ...descendants]
      : [listenerPid],
    listener_pid: listenerPid,
    health_pid: expectedHealthPid || null,
    belongs_to_launched_process_tree: belongsToRoot,
    invoked_executable_path: path.resolve(executablePath),
    invoked_executable_path_fingerprint_sha256: sha256Text(
      path.resolve(executablePath),
    ),
    executable_path: executableIdentity.path,
    executable_sha256: executableIdentity.sha256,
    executable_path_fingerprint_sha256: sha256Text(executableIdentity.path),
  }
}

export function viteRendererSourceSeen(resources) {
  return (Array.isArray(resources) ? resources : []).some((url) =>
    /\/src\/(?:main|App)\.tsx(?:\?|$)/.test(String(url))
    || /\/src\/renderer\/src\/(?:main|App)\.tsx(?:\?|$)/.test(String(url))
  )
}

function captureUiRuntimeProvenance(
  app,
  rendererResources,
  git,
  {
    cdpProcessBinding = null,
    backendProcessBinding = null,
    releaseManifest = null,
  } = {},
) {
  const mode = app?.uiLaunchMode || ''
  const executable = mode === 'installed-app'
    ? path.join(app.appPath, 'Contents', 'MacOS', 'vMLX')
    : path.join(
      panelDir,
      'node_modules',
      'electron',
      'dist',
      'Electron.app',
      'Contents',
      'MacOS',
      'Electron',
    )
  const asarPath = mode === 'installed-app'
    ? path.join(app.appPath, 'Contents', 'Resources', 'app.asar')
    : ''
  const resourcesRoot = mode === 'installed-app'
    ? path.join(app.appPath, 'Contents', 'Resources')
    : ''
  const bundledSourceRoot = resourcesRoot
    ? path.join(resourcesRoot, 'vmlx-engine-source', 'vmlx_engine')
    : ''
  const bundledProvenancePath = resourcesRoot
    ? path.join(resourcesRoot, 'bundled-python', 'vmlx-bundle-provenance.json')
    : ''
  let bundledProvenance = null
  let bundledProvenanceError = null
  if (bundledProvenancePath && existsSync(bundledProvenancePath)) {
    try {
      bundledProvenance = JSON.parse(readFileSync(bundledProvenancePath, 'utf8'))
    } catch (error) {
      bundledProvenanceError = error?.message || String(error)
    }
  }
  const bundledSource = (
    bundledSourceRoot
    && existsSync(path.join(bundledSourceRoot, 'server.py'))
    && existsSync(path.join(bundledSourceRoot, '__init__.py'))
  )
    ? {
        ...pythonSourceTreeDigest(
          bundledSourceRoot,
          path.dirname(bundledSourceRoot),
        ),
        server_module_sha256: sha256File(
          path.join(bundledSourceRoot, 'server.py'),
        ),
        package_init_sha256: sha256File(
          path.join(bundledSourceRoot, '__init__.py'),
        ),
      }
    : null
  const resources = Array.isArray(rendererResources?.resources)
    ? rendererResources.resources.map(String)
    : []
  const scripts = Array.isArray(rendererResources?.scripts)
    ? rendererResources.scripts.map(String)
    : []
  const allResources = [...resources, ...scripts]
  return {
    mode,
    source_commit: git?.commit || null,
    source_tree: git?.tree || null,
    renderer_source_tree_sha256: git?.renderer_source_tree_sha256 || null,
    renderer_source_file_count: git?.renderer_source_file_count ?? null,
    renderer_build_source_commit: rendererResources?.buildSourceCommit || null,
    page_url: rendererResources?.pageUrl || null,
    renderer_resources: allResources,
    vite_client_seen: allResources.some((url) => /(?:@vite\/client|@vite\/client)/.test(url)),
    vite_renderer_source_seen: viteRendererSourceSeen(allResources),
    electron_executable: executable,
    electron_executable_sha256:
      existsSync(executable) ? sha256File(executable) : null,
    cdp_process_binding: cdpProcessBinding,
    backend_python_process_binding: backendProcessBinding,
    served_renderer_modules: rendererResources?.servedModules || [],
    served_renderer_source_sha256:
      rendererResources?.servedRendererSourceSha256 || null,
    app_asar: asarPath || null,
    app_asar_sha256:
      asarPath && existsSync(asarPath) ? sha256File(asarPath) : null,
    external_release_manifest_path: releaseManifest?.path || null,
    external_release_manifest_sha256: releaseManifest?.sha256 || null,
    external_release_manifest: releaseManifest?.value || null,
    bundled_provenance_path: bundledProvenancePath || null,
    bundled_provenance_sha256:
      bundledProvenancePath && existsSync(bundledProvenancePath)
        ? sha256File(bundledProvenancePath)
        : null,
    bundled_provenance: bundledProvenance,
    bundled_provenance_error: bundledProvenanceError,
    bundled_source_root: bundledSourceRoot || null,
    bundled_source: bundledSource,
  }
}

export function runtimeBindingFromHealth(health) {
  const runtime = health?.runtime_provenance || {}
  const modelBundle = health?.model_bundle_provenance || {}
  const cacheTopology = health?.cache_topology_provenance || {}
  const identity = {
    backend_pid: Number(runtime.pid || 0) || null,
    runtime_source_hashes: {
      server_module_sha256: String(runtime.server_module_sha256 || ''),
      package_init_sha256: String(runtime.package_init_sha256 || ''),
      python_source_tree_sha256: String(runtime.python_source_tree_sha256 || ''),
      python_executable_fingerprint_sha256:
        String(runtime.python_executable_fingerprint_sha256 || ''),
    },
    python_source_file_count: runtime.python_source_file_count ?? null,
    python_source_read_error_count: runtime.python_source_read_error_count ?? null,
    model_name: String(health?.model_name || ''),
    loaded_model_name: String(health?.loaded_model_name || ''),
    model_bundle_fingerprint_sha256:
      String(modelBundle.fingerprint_sha256 || ''),
    model_bundle_files:
      modelBundle?.files && typeof modelBundle.files === 'object'
        ? modelBundle.files
        : {},
    cache_topology_fingerprint_sha256:
      String(cacheTopology.fingerprint_sha256 || ''),
  }
  return {
    ...identity,
    // Compatibility alias for the typed Electron artifact reader. It is not
    // part of the canonical API-v2 identity fingerprint below.
    pid: identity.backend_pid,
    fingerprint_sha256: canonicalSha256(identity),
  }
}

function createHealthSnapshot(url, health) {
  return {
    observed_at: new Date().toISOString(),
    url,
    sha256: sha256Json(health),
    binding: runtimeBindingFromHealth(health),
    raw: health,
  }
}

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

async function freePortExcluding(excluded) {
  for (let attempt = 0; attempt < 32; attempt += 1) {
    const port = await freePort()
    if (!excluded.has(port)) return port
  }
  throw new Error('Unable to allocate a distinct loopback port')
}

export async function requestJson(url, timeoutMs = 1000) {
  return await new Promise((resolve, reject) => {
    let settled = false
    const finish = (callback, value) => {
      if (settled) return
      settled = true
      clearTimeout(timer)
      callback(value)
    }
    const request = httpGet(url, { headers: { Connection: 'close' } }, (response) => {
      const chunks = []
      response.on('data', (chunk) => chunks.push(Buffer.from(chunk)))
      response.on('error', (error) => finish(reject, error))
      response.on('end', () => {
        if (response.statusCode == null || response.statusCode < 200 || response.statusCode >= 300) {
          finish(reject, new Error(`${response.statusCode || 0} ${response.statusMessage || ''}`.trim()))
          return
        }
        try {
          finish(resolve, JSON.parse(Buffer.concat(chunks).toString('utf8')))
        } catch (error) {
          finish(reject, error)
        }
      })
    })
    const timer = setTimeout(() => {
      request.destroy(new Error(`Timed out requesting ${url} after ${timeoutMs}ms`))
    }, timeoutMs)
    request.on('error', (error) => finish(reject, error))
  })
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

export function assertCdpExpressionSyntax(expression, label = 'cdp-evaluate') {
  if (typeof expression !== 'string' || !expression.trim()) {
    throw new TypeError(`${label} expression must be a non-empty string`)
  }
  try {
    new vm.Script(expression, { filename: `${label}.generated.js` })
  } catch (error) {
    const detail = error instanceof Error ? error.message : String(error)
    throw new SyntaxError(`${label} generated an invalid CDP expression: ${detail}`)
  }
  return expression
}

async function evaluate(cdp, expression, timeoutMs = 120_000) {
  assertCdpExpressionSyntax(expression)
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
  writePrivateArtifactFile(filePath, Buffer.from(shot.data, 'base64'))
  return filePath
}

const requiredScreenshotMaxBytes = 64 * 1024 * 1024
const requiredPngSignature = Buffer.from([
  0x89, 0x50, 0x4e, 0x47, 0x0d, 0x0a, 0x1a, 0x0a,
])

export function attestPrivatePngScreenshot(filePath) {
  const exactPath = path.resolve(filePath)
  const pathStat = lstatSync(exactPath)
  if (!pathStat.isFile() || pathStat.isSymbolicLink()) {
    throw new Error('Required real UI screenshot must be a regular, non-symlink file')
  }
  if ((pathStat.mode & 0o077) !== 0) {
    throw new Error('Required real UI screenshot must not be group/world accessible')
  }
  if (pathStat.nlink !== 1) {
    throw new Error('Required real UI screenshot must have exactly one filesystem link')
  }
  if (pathStat.size <= 0) {
    throw new Error('required screenshot artifact is empty')
  }
  if (pathStat.size > requiredScreenshotMaxBytes) {
    throw new Error(
      `Required real UI screenshot exceeds ${requiredScreenshotMaxBytes} bytes`,
    )
  }

  const noFollow = Number(fsConstants.O_NOFOLLOW || 0)
  let fd
  let openedStat
  let raw
  try {
    // Open the exact requested path, not a realpath-resolved target. O_NOFOLLOW
    // plus the path/fd identity comparisons below close final-component swap
    // and symlink races between lstat, open, and the completed read.
    fd = openSync(exactPath, fsConstants.O_RDONLY | noFollow)
    openedStat = fstatSync(fd)
    if (!openedStat.isFile()) {
      throw new Error('Required real UI screenshot opened object is not a regular file')
    }
    if ((openedStat.mode & 0o077) !== 0) {
      throw new Error('Required real UI screenshot opened object is group/world accessible')
    }
    if (openedStat.nlink !== 1) {
      throw new Error('Required real UI screenshot opened object must have exactly one filesystem link')
    }
    if (
      openedStat.dev !== pathStat.dev
      || openedStat.ino !== pathStat.ino
      || openedStat.size !== pathStat.size
    ) {
      throw new Error('Required real UI screenshot changed identity between lstat and exact-path open')
    }
    raw = readFileSync(fd)
    if (raw.length !== openedStat.size) {
      throw new Error('Required real UI screenshot changed size while reading')
    }
    if (
      raw.length < requiredPngSignature.length
      || !raw.subarray(0, requiredPngSignature.length).equals(requiredPngSignature)
    ) {
      throw new Error('required screenshot artifact does not have a valid PNG signature')
    }
    const afterReadStat = fstatSync(fd)
    if (
      afterReadStat.dev !== openedStat.dev
      || afterReadStat.ino !== openedStat.ino
      || afterReadStat.size !== openedStat.size
      || afterReadStat.mtimeMs !== openedStat.mtimeMs
      || afterReadStat.ctimeMs !== openedStat.ctimeMs
    ) {
      throw new Error('Required real UI screenshot changed identity while reading')
    }
    const pathAfterRead = lstatSync(exactPath)
    if (
      !pathAfterRead.isFile()
      || pathAfterRead.isSymbolicLink()
      || pathAfterRead.dev !== openedStat.dev
      || pathAfterRead.ino !== openedStat.ino
      || pathAfterRead.size !== openedStat.size
      || pathAfterRead.mtimeMs !== openedStat.mtimeMs
      || pathAfterRead.ctimeMs !== openedStat.ctimeMs
    ) {
      throw new Error('Required real UI screenshot path changed identity while reading')
    }
    if ((pathAfterRead.mode & 0o077) !== 0 || pathAfterRead.nlink !== 1) {
      throw new Error('Required real UI screenshot path lost its private single-link state')
    }
  } finally {
    if (fd != null) closeSync(fd)
  }

  const canonicalPath = realpathSync(exactPath)
  if (isPathInside(canonicalPath, realpathSync(repoDir))) {
    throw new Error('Required real UI screenshot must stay outside the public Git worktree')
  }
  const finalPathStat = lstatSync(exactPath)
  if (
    !finalPathStat.isFile()
    || finalPathStat.isSymbolicLink()
    || finalPathStat.dev !== openedStat.dev
    || finalPathStat.ino !== openedStat.ino
    || finalPathStat.size !== openedStat.size
    || finalPathStat.mtimeMs !== openedStat.mtimeMs
    || finalPathStat.ctimeMs !== openedStat.ctimeMs
  ) {
    throw new Error('Required real UI screenshot path changed identity during canonicalization')
  }
  return {
    path: exactPath,
    canonicalPath,
    device: openedStat.dev,
    inode: openedStat.ino,
    byteSize: raw.length,
    sha256: crypto.createHash('sha256').update(raw).digest('hex'),
    mode: openedStat.mode & 0o777,
    nlink: openedStat.nlink,
    openedNoFollow: true,
    regularFile: true,
    nonSymlink: true,
    privatePermissions: (openedStat.mode & 0o077) === 0,
    exactRequestedPath: true,
    pngSignatureValid: true,
  }
}

export function screenshotAttestationsMatch(captureAttestation, finalAttestation) {
  return Boolean(
    captureAttestation
    && finalAttestation
    && captureAttestation.path === finalAttestation.path
    && captureAttestation.canonicalPath === finalAttestation.canonicalPath
    && captureAttestation.device === finalAttestation.device
    && captureAttestation.inode === finalAttestation.inode
    && captureAttestation.byteSize === finalAttestation.byteSize
    && captureAttestation.sha256 === finalAttestation.sha256
    && finalAttestation.openedNoFollow === true
    && finalAttestation.regularFile === true
    && finalAttestation.nonSymlink === true
    && finalAttestation.privatePermissions === true
    && finalAttestation.exactRequestedPath === true
    && finalAttestation.pngSignatureValid === true
  )
}

export async function captureRequiredScreenshot(capture, filePath) {
  const attemptedPath = path.resolve(filePath)
  try {
    const capturedPath = await capture(filePath)
    if (!capturedPath) {
      throw new Error('required screenshot capture returned no artifact path')
    }
    const resolvedCapturedPath = path.resolve(capturedPath)
    if (resolvedCapturedPath !== attemptedPath) {
      throw new Error(
        'required screenshot capture substituted a different artifact path: '
        + `${resolvedCapturedPath} != ${attemptedPath}`,
      )
    }
    const attestation = attestPrivatePngScreenshot(resolvedCapturedPath)
    return {
      status: 'captured',
      path: attestation.path,
      attemptedPath,
      attestation,
      finalAttestation: null,
      error: null,
    }
  } catch (error) {
    return {
      status: 'failed',
      path: null,
      attemptedPath,
      attestation: null,
      finalAttestation: null,
      error: {
        stage: 'chat_screenshot_capture',
        name: error?.name || 'Error',
        message: error?.message || String(error),
      },
    }
  }
}

export function reattestRequiredScreenshot(screenshotCapture) {
  if (
    screenshotCapture?.status !== 'captured'
    || !screenshotCapture?.path
    || !screenshotCapture?.attestation
  ) return screenshotCapture
  try {
    const finalAttestation = attestPrivatePngScreenshot(
      screenshotCapture.attemptedPath,
    )
    if (!screenshotAttestationsMatch(
      screenshotCapture.attestation,
      finalAttestation,
    )) {
      throw new Error(
        'required screenshot artifact identity changed after capture '
        + '(device/inode/size/SHA-256 mismatch)',
      )
    }
    return {
      ...screenshotCapture,
      path: finalAttestation.path,
      finalAttestation,
      error: null,
    }
  } catch (error) {
    return {
      ...screenshotCapture,
      status: 'failed',
      path: null,
      finalAttestation: null,
      error: {
        stage: 'chat_screenshot_reattestation',
        name: error?.name || 'Error',
        message: error?.message || String(error),
      },
    }
  }
}

export function mergeRequiredScreenshotOutcome(rendererResult, screenshotCapture) {
  const screenshotFailed = (
    screenshotCapture?.status !== 'captured'
    || !screenshotCapture?.path
    || screenshotCapture?.error
  )
  return {
    ...rendererResult,
    // A renderer/chat failure is the primary defect. A later CDP screenshot
    // timeout is retained as secondary metadata and must not mask that stage.
    rendererFailureStage: rendererResult?.rendererFailureStage
      || (
        screenshotFailed
          ? screenshotCapture?.error?.stage || 'chat_screenshot_capture'
          : null
      ),
    screenshotCapture,
  }
}

const rendererProofModulePaths = [
  'src/renderer/src/main.tsx',
  'src/renderer/src/App.tsx',
  'src/renderer/src/components/chat/MessageBubble.tsx',
  'src/renderer/src/components/chat/ReasoningBox.tsx',
  'src/renderer/src/components/chat/InlineToolCall.tsx',
  'src/renderer/src/components/chat/ToolCallStatus.tsx',
]

export function viteRawRendererModulePath(relativePath, runId) {
  const rendererRootPrefix = 'src/renderer/'
  if (!relativePath.startsWith(rendererRootPrefix)) {
    throw new Error(`Renderer proof module is outside the Vite renderer root: ${relativePath}`)
  }
  const servedPath = relativePath.slice(rendererRootPrefix.length)
  if (!servedPath || servedPath.startsWith('/') || servedPath.includes('..')) {
    throw new Error(`Renderer proof module has an unsafe served path: ${relativePath}`)
  }
  // The renderer entry is already cached by Vite as `/src/main.tsx`, and Vite
  // reuses that transform even when `?raw` is added. Importing that URL then
  // parses the TSX entry instead of returning source text. A physical `/@fs/`
  // URL gives the proof import its own raw-transform identity for entry and
  // non-entry modules alike.
  const absolutePath = path.resolve(panelDir, relativePath)
  const rendererRoot = path.resolve(panelDir, 'src/renderer')
  if (!absolutePath.startsWith(`${rendererRoot}${path.sep}`)) {
    throw new Error(`Renderer proof module escapes the Vite renderer root: ${relativePath}`)
  }
  const viteFsPath = absolutePath.split(path.sep).join('/')
  return `/@fs${encodeURI(viteFsPath)}?raw&vmlx_proof=${encodeURIComponent(runId)}`
}

export function localRendererModuleEvidence() {
  return rendererProofModulePaths.map((relativePath) => {
    const absolutePath = path.join(panelDir, relativePath)
    const data = readFileSync(absolutePath)
    return {
      relative_path: relativePath,
      size_bytes: data.length,
      sha256: crypto.createHash('sha256').update(data).digest('hex'),
    }
  })
}

function startUiApp(userDataDir, debugPort, gatewayPort) {
  const proofEnv = {
    ...process.env,
    VMLX_SKIP_UPDATE_CHECK: '1',
    VMLX_ALLOW_SECONDARY_INSTANCE: '1',
    VMLX_PROOF_OWNED_ENGINE_LIFECYCLE: '1',
    VMLX_PROOF_GATEWAY_PORT: String(gatewayPort),
    VMLINUX_PROOF_OWNED_ENGINE_LIFECYCLE: '1',
    VMLINUX_PROOF_GATEWAY_PORT: String(gatewayPort),
  }
  if (installedAppPath) {
    const exe = path.join(installedAppPath, 'Contents', 'MacOS', 'vMLX')
    if (!existsSync(exe)) {
      throw new Error(`Installed vMLX executable not found: ${exe}`)
    }
    const args = [
      `--vmlx-user-data-dir=${userDataDir}`,
      '--vmlx-allow-secondary-instance',
      `--remote-debugging-port=${debugPort}`,
    ]
    const logs = []
    const proc = spawn(exe, args, {
      cwd: tmpdir(),
      env: proofEnv,
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
      gatewayPort,
    }
  }

  const args = [
    'run',
    'dev',
    '--',
    '--',
    `--vmlx-user-data-dir=${userDataDir}`,
    '--vmlx-allow-secondary-instance',
    `--remote-debugging-port=${debugPort}`,
  ]
  const logs = []
  const proc = spawn('npm', args, {
    cwd: panelDir,
    env: proofEnv,
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
    gatewayPort,
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

function approximatelyEqual(left, right, tolerance = 1e-6) {
  return typeof left === 'number'
    && typeof right === 'number'
    && Number.isFinite(left)
    && Number.isFinite(right)
    && Math.abs(left - right) <= tolerance
}

function numericField(value, ...keys) {
  for (const key of keys) {
    const candidate = value?.[key]
    if (typeof candidate === 'number' && Number.isFinite(candidate)) return candidate
  }
  return undefined
}

function parseJsonObject(value) {
  if (value && typeof value === 'object' && !Array.isArray(value)) return value
  if (typeof value !== 'string' || !value.trim()) return null
  try {
    const parsed = JSON.parse(value)
    return parsed && typeof parsed === 'object' && !Array.isArray(parsed) ? parsed : null
  } catch {
    return null
  }
}

function extractToolCommand(detail) {
  const parsed = parseJsonObject(detail)
  if (!parsed) return ''
  for (const key of ['command', 'cmd', 'script']) {
    if (typeof parsed[key] === 'string') return parsed[key]
  }
  const nested = parseJsonObject(parsed.arguments)
  if (nested) return extractToolCommand(nested)
  return ''
}

export function validateModelBundleBinding(result) {
  const failures = []
  const requestedPath = String(
    result?.bundleGenerationContract?.bundle_path
    || result?.modelPath
    || '',
  )
  const requestedModel = String(result?.servedModel || '')
  const expectedAttestation = result?.bundleGenerationContract?.health_attestation
  const healthSnapshots = [
    result?.healthProvenance?.before?.raw,
    result?.healthProvenance?.after?.raw,
  ]
  const listedIds = (result?.server?.models?.data || [])
    .map((entry) => String(entry?.id || ''))
    .filter(Boolean)
  if (!requestedPath) failures.push('requested local bundle path is missing')
  if (!requestedModel) failures.push('requested served model name is missing')
  const boundIds = listedIds.filter(
    (id) => (
      id === requestedModel
      || id.endsWith(`/${requestedModel}`)
      || requestedPath.endsWith(`/${id}`)
    ),
  )
  if (
    listedIds.length < 1
    || boundIds.length !== listedIds.length
    || !listedIds.includes(requestedModel)
  ) {
    failures.push(
      `/v1/models is not exactly bound to requested served model ${requestedModel || 'missing'}`,
    )
  }
  for (const [index, health] of healthSnapshots.entries()) {
    const phase = index === 0 ? 'before' : 'after'
    const healthModel = String(health?.model_name || '')
    if (!health || health?.model_loaded !== true) {
      failures.push(`${phase} /health did not report a loaded model`)
      continue
    }
    if (
      requestedModel
      && healthModel !== requestedModel
    ) {
      failures.push(
        `${phase} /health model ${healthModel || 'missing'} does not match requested served model ${requestedModel}`,
      )
    }
    const attestation = health?.model_bundle_provenance
    if (!expectedAttestation || canonicalJson(attestation) !== canonicalJson(expectedAttestation)) {
      failures.push(`${phase} /health bundle attestation does not match requested bundle bytes`)
    }
    if (
      health?.runtime_provenance?.model_bundle_provenance
      && canonicalJson(health.runtime_provenance.model_bundle_provenance)
        !== canonicalJson(attestation)
    ) {
      failures.push(`${phase} nested /health bundle attestation disagrees with top-level attestation`)
    }
  }
  return failures
}

function normalizeProofText(value) {
  return String(value || '').replace(/\s+/g, ' ').trim()
}

function normalizeVisibleLinkText(value) {
  return normalizeProofText(
    String(value || '')
      .replace(/^\s*(?:[-+*]|\d+[.)])\s+/gm, '')
      .replace(/[•◦]\s*/g, '')
      .replace(/\[([^\]]+)]\(([^)]+)\)/g, '$1'),
  )
    .replace(/\\([×÷·≈≤≥≠±→←∞π])/g, '$1')
    // A DANGLING TeX delimiter is markdown-escaped punctuation by the time it
    // reaches the rail, so unescape it on both sides of the comparison exactly
    // as the symbol rules above do. Terminated math never reaches here — it has
    // already been swapped for a VMLXPROOFMATH placeholder — so this only ever
    // touches an opener or closer the model never paired.
    //
    // Found on three consecutive LFM2.5 runs whose ONLY failure was
    // "normalized visible reasoning rail segments are not linked to persisted
    // reasoning segments". The whole diff was five backslashes and two
    // newlines: the model's reasoning wrote `Math: \(2 + 2 = 4` with no closing
    // `\)`, markdown rendered the escaped paren as a literal `(`, and the
    // persisted text kept the backslash. Same failure on the gemma4
    // cachecontrols row.
    .replace(/\\([()[\]])/g, '$1')
    .replace(/\\times\b/g, '×')
    .replace(/\\div\b/g, '÷')
    .replace(/\\cdot\b/g, '·')
    .replace(/\\approx\b/g, '≈')
    .replace(/\\leq?\b/g, '≤')
    .replace(/\\geq?\b/g, '≥')
    .replace(/\\neq\b/g, '≠')
    .replace(/\\pm\b/g, '±')
    .replace(/[*_`#>~]/g, '')
    .replace(/\s+/g, ' ')
    .trim()
}

const PROOF_CODE_RE = /```[\s\S]*?```|~~~[\s\S]*?~~~|`[^`\n]*`/g

function looksLikeProofSingleDollarMath(text) {
  const trimmed = text.trim()
  if (!trimmed || trimmed !== text) return false
  if (/^[+\-*/=<>]/.test(trimmed) || /[+\-*/=<>]$/.test(trimmed)) return false
  if (/^\d+(?:[.,]\d{2})?$/.test(trimmed)) return false
  const lexicalView = trimmed
    .replace(/\\text\s*\{[^{}]*\}/g, '')
    .replace(/\\[A-Za-z]+/g, '')
  const proseWords = lexicalView.match(/[A-Za-z]{2,}/g) || []
  if (
    proseWords.some(
      (word) =>
        !/^(?:sin|cos|tan|cot|sec|csc|log|ln|exp|lim|max|min|mod|gcd|lcm|det)$/i.test(
          word,
        ),
    )
  ) {
    return false
  }
  if (/\\[A-Za-z]+/.test(trimmed)) return true
  if (/[{}_^=<>]/.test(trimmed)) return true
  if (/(?:[\dA-Za-z])\s*[+\-*/]\s*(?:[\dA-Za-z])/.test(trimmed)) return true
  if (/^[A-Za-z]$/.test(trimmed)) return true
  return false
}

function isProofDollarEscaped(text, index) {
  let precedingBackslashes = 0
  for (let cursor = index - 1; cursor >= 0 && text[cursor] === '\\'; cursor--) {
    precedingBackslashes += 1
  }
  return precedingBackslashes % 2 === 1
}

function findNextProofDollar(text, start) {
  for (let index = start; index < text.length; index++) {
    if (text[index] === '$' && !isProofDollarEscaped(text, index)) return index
  }
  return -1
}

function replaceProofSingleDollarMathLine(line, captureMath) {
  let output = ''
  let unchangedStart = 0
  let opener = findNextProofDollar(line, 0)
  while (opener >= 0) {
    let closer = findNextProofDollar(line, opener + 1)
    while (closer >= 0) {
      const body = line.slice(opener + 1, closer)
      if (looksLikeProofSingleDollarMath(body)) {
        output += line.slice(unchangedStart, opener)
        output += captureMath(body, 'single-dollar', 'inline')
        unchangedStart = closer + 1
        opener = findNextProofDollar(line, unchangedStart)
        break
      }
      // A rejected currency pair may end at the opener of a following valid
      // expression (`$43 and $47 \\times 19$`). Preserve the literal dollar
      // and retry from that candidate exactly as the product renderer does.
      output += line.slice(unchangedStart, opener + 1)
      unchangedStart = opener + 1
      opener = closer
      closer = findNextProofDollar(line, opener + 1)
    }
    if (closer < 0) return output + line.slice(unchangedStart)
  }
  return output + line.slice(unchangedStart)
}

function normalizeProofRepeatedMathDelimiters(markdown) {
  return String(markdown || '')
    .replace(/(?:\\\(\s*){2,}/g, '\\(')
    .replace(/(?:\\\)\s*){2,}/g, '\\)')
    .replace(/(?:\\\[\s*){2,}/g, '\\[')
    .replace(/(?:\\\]\s*){2,}/g, '\\]')
}

/**
 * Replace only renderer-recognized TeX spans with stable markers while
 * retaining each canonical source-span identity: trimmed body, delimiter kind, and
 * inline/display intent. Raw stream-to-SQLite equality is attested separately.
 * The DOM collector applies the same markers to completed sanitized KaTeX
 * nodes, proving that rendering did not hide changed, missing, or reordered
 * model text.
 */
export function extractPersistedReasoningMathLinkage(value) {
  const protectedSegments = []
  const mathSources = []
  let linkedText = normalizeProofRepeatedMathDelimiters(
    String(value || '').replace(PROOF_CODE_RE, (segment) => {
      const index = protectedSegments.push(segment) - 1
      return `\u0000PROOFCODE${index}\u0000`
    }),
  )
  const captureMath = (body, delimiter, displayMode) => {
    const source = String(body || '').trim()
    if (!source) return ''
    const index = mathSources.push({ source, delimiter, displayMode }) - 1
    return `VMLXPROOFMATH${index}`
  }
  linkedText = linkedText
    .replace(/\\\[([\s\S]*?)\\\]/g, (_match, body) =>
      captureMath(body, 'bracket', 'display'))
    .replace(/\$\$([\s\S]*?)\$\$/g, (_match, body) =>
      captureMath(body, 'double-dollar', 'display'))
    .replace(/\\\(([^\n]*?)\\\)/g, (_match, body) =>
      captureMath(body, 'paren', 'inline'))
  linkedText = linkedText
    .split('\n')
    .map((line) => replaceProofSingleDollarMathLine(line, captureMath))
    .join('\n')
    .replace(/\u0000PROOFCODE(\d+)\u0000/g, (_match, indexText) => (
      protectedSegments[Number(indexText)] || ''
    ))
  return { linkedText, mathSources }
}

const rawProtocolMarkerRegex =
  /<think>|<\/think>|<tool_call>|<\/tool_call>|<function>|<invoke>|<minimax:tool_call>|<zyphra_tool_call>|<\|point_start\|>|<\|point_end\|>|<\|box_start\|>|<\|box_end\|>|<\|tool_call_start\|>|<\|tool_call_end\|>|\[THINK\]|\[\/THINK\]|<mm:think>|<\/mm:think>/i

const protocolMarkers = [
  '<think>',
  '</think>',
  '<tool_call>',
  '</tool_call>',
  '<function>',
  '<invoke>',
  '<minimax:tool_call>',
  '<zyphra_tool_call>',
  '<|point_start|>',
  '<|point_end|>',
  '<|box_start|>',
  '<|box_end|>',
  '<|tool_call_start|>',
  '<|tool_call_end|>',
  '[THINK]',
  '[/THINK]',
  '<mm:think>',
  '</mm:think>',
]

function containsTransientProtocolMarker(value) {
  const text = String(value || '')
  if (rawProtocolMarkerRegex.test(text)) return true
  return protocolMarkers.some((marker) => {
    for (let length = 3; length < marker.length; length += 1) {
      if (text.includes(marker.slice(0, length))) return true
    }
    return false
  })
}

function expectedUiTurnCount(result) {
  const configured = Number(result?.requestContract?.uiTurnCount)
  return Number.isInteger(configured) && configured >= 1 && configured <= 3
    ? configured
    : 3
}

// Share of the reasoning text that long numeric runs must occupy before they
// count as degenerate spew rather than deliberate arithmetic. Step37 counted
// "1 2 3 ... 21" to check `wc -c` against the length of REAL_UI_LIVE_TOOL_ONE:
// one 55-character run inside 8,536 characters of coherent English, 0.6% of the
// text, which the raw count flagged as garbage. Real spew is the bulk of what it
// appears in, not a rounding error.
export const REASONING_NUMERIC_SPEW_SHARE = 0.15

export function reasoningNumericRunIsSpew(chat) {
  const count = Number(chat?.reasoningNumericRunCount || 0)
  if (count <= 0) return false
  const runChars = Number(chat?.reasoningNumericRunChars || 0)
  const textLength = Number(chat?.reasoningTextLength || 0)
  // An older artifact records the count but neither length: fall back to the
  // original count-only rule rather than silently passing something unmeasured.
  if (!runChars || !textLength) return true
  return runChars / textLength >= REASONING_NUMERIC_SPEW_SHARE
}

export function expectedUiToolCallCount(result) {
  const profile = String(result?.requestContract?.uiActionProfile || '')
  if (profile === 'primary-tool-restart-probe') return 1
  if (profile === 'native-three-turn-switch') return 2
  // The legacy three-turn prompt contract predates named release profiles but
  // still explicitly requests one built-in run_command call on each of the
  // first two turns. Do not let the presence of its profile name downgrade the
  // expected count to the generic no-tool default.
  if (
    profile === 'legacy-three-turn'
    && result?.requestedBuiltinTools === true
  ) return 2
  if (profile) return 0
  return result?.requestedBuiltinTools === true ? 2 : 0
}

export function uiProfileRequiresPositiveCacheReuse(result) {
  return new Set([
    'primary-tool-restart-probe',
    'primary-restart-followup',
    'native-three-turn-switch',
  ]).has(String(result?.requestContract?.uiActionProfile || ''))
}

// The restart probes must share at least one complete 64-token cache block
// with their corresponding store phase. Keep this deliberately longer than
// one block even for tokenizers that split the numbered anchors efficiently.
export const releasePrimarySharedPrefix = [
  'R19_PRIMARY_SHARED_PREFIX',
  'cache-anchor-9f4b7d2a',
  ...Array.from(
    { length: 96 },
    (_, index) => `cache-token-${String(index).padStart(3, '0')}`,
  ),
  'Keep the response coherent and finite.',
].join(' ')

export function validateRenderedDomEvidence(result) {
  const failures = []
  const expectedTurns = expectedUiTurnCount(result)
  const dom = result?.renderedDom || {}
  const messages = Array.isArray(dom.messages) ? dom.messages : []
  const assistantIds = Array.isArray(result?.assistantMessageIds)
    ? result.assistantMessageIds.slice(0, expectedTurns)
    : []
  const traceIds = new Set(
    (result?.messageEventTrace || [])
      .filter((row) => row?.events?.some((event) => event?.event === 'terminal'))
      .map((row) => String(row.messageId || '')),
  )
  const traceById = new Map(
    (result?.messageEventTrace || []).map((row) => [
      String(row?.messageId || ''),
      row,
    ]),
  )
  const persistedById = new Map(
    (result?.assistantRecords || []).map((message) => [
      String(message?.id || ''),
      String(message?.content || ''),
    ]),
  )
  if (assistantIds.length !== expectedTurns) {
    failures.push(`expected exactly ${expectedTurns} primary assistant message IDs, got ${assistantIds.length}`)
  }
  for (const [assistantIndex, messageId] of assistantIds.entries()) {
    const rendered = messages.find((row) => String(row?.messageId || '') === String(messageId))
    if (!rendered) {
      failures.push(`assistant message ${messageId} has no rendered DOM evidence`)
      continue
    }
    if (rendered.role !== 'assistant') {
      failures.push(`message ${messageId} rendered with role ${rendered.role || 'missing'}`)
    }
    if (!String(rendered.answerText || '').trim()) {
      failures.push(`assistant message ${messageId} has empty rendered answer text`)
    }
    if (
      rendered.answerState !== 'complete'
      || !Number.isInteger(rendered.answerFullLength)
      || rendered.answerFullLength < 0
      || rendered.answerRenderedLength !== rendered.answerFullLength
    ) {
      failures.push(
        `assistant message ${messageId} visible typewriter did not drain to the persisted source length`,
      )
    }
    if (!String(persistedById.get(String(messageId)) || '').trim()) {
      failures.push(`assistant message ${messageId} has no matching persisted content record`)
    }
    const persisted = String(persistedById.get(String(messageId)) || '')
    if (
      assistantIndex < 2
      && normalizeVisibleLinkText(rendered.answerText)
        !== normalizeVisibleLinkText(persisted)
    ) {
      failures.push(
        `assistant message ${messageId} normalized visible answer is not linked to persisted content`,
      )
    }
    const finalContent = traceChannelText(
      traceById.get(String(messageId))?.events || [],
      'content',
    )
    // The persistence writer (main/ipc/chat.ts periodic/final save) trims
    // leading/trailing whitespace when stitching tool-loop segments, so the
    // wire-faithful delta trace can legitimately differ by edge whitespace
    // (e.g. a leading "\n" content delta after a tool result). Mid-content
    // must still match byte-exact.
    if (finalContent.trim() !== persisted.trim()) {
      failures.push(`assistant message ${messageId} final stream does not equal persisted content`)
    }
    if (!traceIds.has(String(messageId))) {
      failures.push(`assistant message ${messageId} is not linked to a terminal event trace`)
    }
    if (rendered.visible !== true) {
      failures.push(`assistant message ${messageId} was not visibly rendered`)
    }
    if ((rendered.toolCards || []).some((card) => card?.visible !== true)) {
      failures.push(`assistant message ${messageId} contains a non-visible tool card`)
    }
  }
  if (Array.isArray(dom.rawI18nKeys) && dom.rawI18nKeys.length) {
    failures.push(`raw i18n keys are visible: ${dom.rawI18nKeys.join(', ')}`)
  }
  if (Array.isArray(dom.visibleErrors) && dom.visibleErrors.length) {
    failures.push(`visible UI errors were observed: ${dom.visibleErrors.join(' | ')}`)
  }
  if (Array.isArray(dom.transientAlerts) && dom.transientAlerts.length) {
    failures.push(`transient UI alerts were observed: ${dom.transientAlerts.join(' | ')}`)
  }
  const configuredPrompts = [
    result?.requestContract?.promptOne,
    result?.requestContract?.promptTwo,
    result?.requestContract?.promptThree,
  ].slice(0, expectedTurns).map(String)
  const renderingPromptIndex = configuredPrompts.findIndex(
    (prompt) => prompt.includes('$43') || prompt.includes('\\times'),
  )
  if (renderingPromptIndex >= 0) {
    const expectedUserMessageId = String(
      result?.uiTurnEvidence?.[renderingPromptIndex]?.userMessageId || '',
    )
    const renderedUserMathMessage = (dom.userMessages || []).find(
      (row) => String(row?.messageId || '') === expectedUserMessageId,
    )
    const userCurrencyOccurrences = Array.isArray(
      renderedUserMathMessage?.currencyOccurrences,
    ) ? renderedUserMathMessage.currencyOccurrences : []
    if (
      !expectedUserMessageId
      || renderedUserMathMessage?.role !== 'user'
      || renderedUserMathMessage?.visible !== true
      || !String(renderedUserMathMessage?.text || '').includes('$43')
      || userCurrencyOccurrences.length !== 1
      || userCurrencyOccurrences[0]?.insideKatex === true
      || (renderedUserMathMessage?.katexCount || 0) < 1
      || (renderedUserMathMessage?.katexErrorCount || 0) > 0
      || /\\(?:times|frac|div|approx)\b/.test(
        String(renderedUserMathMessage?.text || ''),
      )
    ) {
      failures.push(
        'deterministic visible user prompt did not preserve currency and render its TeX through KaTeX',
      )
    }
    const renderedMathMessage = messages.find(
      (row) => String(row?.messageId || '')
        === String(assistantIds[renderingPromptIndex] || ''),
    )
    if (!String(renderedMathMessage?.answerText || '').includes('$43')) {
      failures.push('rendered answer did not preserve literal currency $43')
    }
    const currencyOccurrences = Array.isArray(renderedMathMessage?.currencyOccurrences)
      ? renderedMathMessage.currencyOccurrences
      : []
    if (
      currencyOccurrences.length !== 1
      || currencyOccurrences[0]?.insideKatex === true
    ) {
      failures.push('literal currency $43 was not preserved exactly once outside KaTeX')
    }
    if ((renderedMathMessage?.katexErrorCount || 0) > 0) {
      failures.push('rendered answer contains a KaTeX error')
    }
    if (
      (
        /\\(?:\(|\[)/.test(String(renderedMathMessage?.answerText || ''))
        || /\\(?:times|frac|div|approx)\b/.test(
          String(renderedMathMessage?.answerText || ''),
        )
      )
      || String(renderedMathMessage?.html || '').includes('katex-error')
    ) {
      failures.push('rendered answer exposed raw TeX/parser syntax')
    }
    const expectedTex = '47 \\times 19 = 893 < 920 = 46 \\times 20'
    const persistedMathSource = String(
      persistedById.get(String(assistantIds[renderingPromptIndex] || '')) || '',
    )
    const exactSingleDollarSource = `$${expectedTex}$`
    // A live model may legally answer with Unicode math rather than echoing
    // the requested TeX source. Require KaTeX only when the persisted
    // assistant bytes actually contain the delimited TeX; otherwise validate
    // the truthful plain rendering without inventing a renderer failure.
    if (persistedMathSource.includes(exactSingleDollarSource)) {
      if ((renderedMathMessage?.katexCount || 0) < 1) {
        failures.push('persisted TeX source did not produce a KaTeX-rendered expression')
      }
      const expectedVisibleMath = normalizeProofText(
        '47 × 19 = 893 < 920 = 46 × 20',
      )
      if (
        !normalizeProofText(renderedMathMessage?.answerText)
          .includes(expectedVisibleMath)
      ) {
        failures.push('rendered answer does not visibly contain the expected KaTeX expression')
      }
      // KaTeX output:'html' intentionally omits MathML annotations. If an
      // annotation is present, it remains useful corroboration and must be exact.
      const annotations = Array.isArray(renderedMathMessage?.katexAnnotations)
        ? renderedMathMessage.katexAnnotations.map((value) => normalizeProofText(value))
        : []
      if (
        annotations.length > 0
        && (
          annotations.length !== 1
          || annotations[0] !== expectedTex
        )
      ) {
        failures.push('rendered answer KaTeX source does not exactly match the persisted expression')
      }
    }
  }
  return failures
}

function traceChannelText(events, channel) {
  return (Array.isArray(events) ? events : [])
    .filter((event) => event?.event === 'stream' && event?.channel === channel)
    .map((event) => String(event?.delta || ''))
    .join('')
}

function traceReasoningSegments(events) {
  const segments = new Map()
  let inferredSegment = 0
  for (const event of Array.isArray(events) ? events : []) {
    if (event?.event === 'reasoningDone' || event?.event === 'reasoning_terminal') {
      inferredSegment += 1
      continue
    }
    if (event?.event !== 'stream' || event?.channel !== 'reasoning') continue
    const explicit = Number(event?.segmentIndex)
    const index = Number.isInteger(explicit) && explicit >= 0
      ? explicit
      : inferredSegment
    segments.set(index, `${segments.get(index) || ''}${String(event?.delta || '')}`)
  }
  return [...segments.entries()]
    .sort(([left], [right]) => left - right)
    .map(([, value]) => value)
    .filter(Boolean)
}

export function upsertBoundedDomSample(samples, state, sample, maxSamples, force = false) {
  if (!Array.isArray(samples) || !state || !sample) return false
  const limit = Math.max(1, Number.parseInt(String(maxSamples), 10) || 1)
  const otherLimit = limit >= 4 ? 2 : 0
  const primaryLimit = Math.max(1, Math.floor((limit - otherLimit) / 2))
  const channelLimits = {
    content: primaryLimit,
    reasoning: primaryLimit,
    other: otherLimit,
  }
  const answerText = String(sample.answerText || '')
  const reasoningText = String(sample.reasoningText || '')
  const contentChanged = answerText !== String(state.lastAnswerText || '')
  const reasoningChanged = reasoningText !== String(state.lastReasoningText || '')
  const channel = contentChanged
    ? 'content'
    : reasoningChanged
      ? 'reasoning'
      : 'other'
  state.channelCounts ||= { content: 0, reasoning: 0, other: 0 }
  state.channelLastStoredIndex ||= {}
  const signature = [
    answerText.length,
    reasoningText.length,
    Number(sample.katexCount || 0),
    Array.isArray(sample.toolCards) ? sample.toolCards.length : 0,
  ].join(':')
  if (!force && signature === state.lastSignature) return false
  const channelHasRoom = Number(state.channelCounts[channel] || 0)
    < Number(channelLimits[channel] || 0)
  if (state.count < limit && channelHasRoom) {
    samples.push(sample)
    state.count += 1
    state.channelCounts[channel] = Number(state.channelCounts[channel] || 0) + 1
    state.lastStoredIndex = samples.length - 1
    state.channelLastStoredIndex[channel] = samples.length - 1
  } else if (
    force
    && Number.isInteger(
      state.channelLastStoredIndex[channel] ?? state.lastStoredIndex,
    )
  ) {
    const replacementIndex =
      state.channelLastStoredIndex[channel] ?? state.lastStoredIndex
    if (replacementIndex < 0 || replacementIndex >= samples.length) return false
    samples[replacementIndex] = sample
    state.lastStoredIndex = replacementIndex
    state.channelLastStoredIndex[channel] = replacementIndex
  } else {
    state.lastAnswerText = answerText
    state.lastReasoningText = reasoningText
    state.lastSignature = signature
    return false
  }
  state.lastAnswerText = answerText
  state.lastReasoningText = reasoningText
  state.lastSignature = signature
  return true
}

export function validateReasoningEvidence(result, expectation = 'optional') {
  const failures = []
  // Recorded on the result so a run that leaned on never-empty notices is
  // visible in the artifact rather than reading as a clean pass.
  const neverEmptyNoticeTurns = []
  const expectedTurns = expectedUiTurnCount(result)
  const assistantIds = new Set(
    (Array.isArray(result?.assistantMessageIds)
      ? result.assistantMessageIds.slice(0, expectedTurns)
      : [])
      .map((messageId) => String(messageId || '')),
  )
  const traces = (Array.isArray(result?.messageEventTrace)
    ? result.messageEventTrace
    : [])
    .filter((row) => assistantIds.has(String(row?.messageId || '')))
  const domMessages = Array.isArray(result?.renderedDom?.messages)
    ? result.renderedDom.messages
    : []
  const domSamples = Array.isArray(result?.renderedDom?.samples)
    ? result.renderedDom.samples
    : []
  let reasoningMessageCount = 0
  if (assistantIds.size !== expectedTurns || traces.length !== expectedTurns) {
    failures.push(
      `expected reasoning/content traces for exactly ${expectedTurns} primary assistant turns, got ${traces.length}`,
    )
  }
  for (const row of traces) {
    const events = Array.isArray(row?.events) ? row.events : []
    const reasoningEvents = events.filter(
      (event) => event?.event === 'stream' && event?.channel === 'reasoning',
    )
    const contentEvents = events.filter(
      (event) => event?.event === 'stream' && event?.channel === 'content',
    )
    const progressiveContentDeltaCount = contentEvents
      .filter((event) => String(event?.delta || '').length > 0)
      .length
    const messageDomSamples = domSamples.filter(
      (sample) => String(sample?.messageId || '') === String(row?.messageId || ''),
    )
    const renderedAnswerStates = new Set(
      messageDomSamples.map((sample) => String(sample?.answerText || '')).filter(Boolean),
    )
    const orderedSequences = events.map((event) => Number(event?.sequence))
    if (
      orderedSequences.some((sequence) => !Number.isFinite(sequence))
      || orderedSequences.some((sequence, index) => index > 0 && sequence <= orderedSequences[index - 1])
    ) {
      failures.push(`message ${row?.messageId || 'unknown'} event sequence is not strictly monotonic`)
    }
    for (const channel of ['reasoning', 'content']) {
      let previousLength = 0
      let previousSegmentIndex = null
      let accumulatedSegmentText = ''
      let previousSegmentIndexForText = null
      let previousReasoningSegments = []
      const channelEvents = events.filter(
        (event) => event?.event === 'stream' && event?.channel === channel,
      )
      for (const event of channelEvents) {
        const segmentIndex = channel === 'reasoning'
          ? (Number.isInteger(Number(event?.segmentIndex))
              ? Number(event.segmentIndex)
              : 0)
          : 0
        if (
          channel === 'reasoning'
          && previousSegmentIndex !== null
          && segmentIndex !== previousSegmentIndex
        ) {
          previousLength = 0
        }
        previousSegmentIndex = segmentIndex
        const delta = String(event?.delta || '')
        const retainedFullContent = event?.payload?.fullContent
        const retainedReasoningSegments = channel === 'reasoning'
          ? event?.payload?.reasoningSegments
          : null
        if (channel === 'reasoning') {
          accumulatedSegmentText = segmentIndex === previousSegmentIndexForText
            ? accumulatedSegmentText + delta
            : delta
          previousSegmentIndexForText = segmentIndex
        }
        // The panel's snapshot carries VISIBLE segments (whitespace-only
        // segments are filtered by design — LFM reasoning opens with a bare
        // "\n" delta), so an empty/short snapshot is CORRECT until the
        // current segment has non-whitespace content.
        const segmentHasVisibleText = channel === 'reasoning'
          && accumulatedSegmentText.trim().length > 0
        if (
          channel === 'reasoning'
          && segmentHasVisibleText
          && (
            !Array.isArray(retainedReasoningSegments)
            || retainedReasoningSegments.length <= segmentIndex
            || previousReasoningSegments.some(
              // One legal transition for an EARLIER segment: the panel's
              // reasoning_summary_text.done reconcile may EXTEND a completed
              // segment when wire deltas were dropped (the streamed text is a
              // strict prefix of the server's authoritative text) — that
              // repair is byte-faithful to the server. Any other rewrite of
              // an earlier segment remains corruption. Observed live on LFM
              // tool-loop turns where each iteration opens a new segment
              // after the previous one's done-reconcile.
              (value, index) => index < segmentIndex
                && String(retainedReasoningSegments[index] || '') !== value
                && !String(retainedReasoningSegments[index] || '').startsWith(value),
            )
          )
        ) {
          failures.push(
            `message ${row?.messageId || 'unknown'} reasoning segment history changed or was omitted`,
          )
          break
        }
        const fullContentLength = Number(
          (channel === 'reasoning'
            ? event?.payload?.reasoningSegmentLength
            : event?.payload?.fullContentLength)
            ?? (typeof retainedFullContent === 'string'
              ? retainedFullContent.length
              : Number.NaN),
        )
        if (
          event?.cumulativeReset === true
          || !Number.isFinite(fullContentLength)
          || fullContentLength !== previousLength + delta.length
        ) {
          failures.push(`message ${row?.messageId || 'unknown'} ${channel} stream reset or shrank`)
          break
        }
        if (
          containsTransientProtocolMarker(delta)
        ) {
          failures.push(`message ${row?.messageId || 'unknown'} leaked parser markers in transient ${channel} stream`)
          break
        }
        previousLength = fullContentLength
        if (channel === 'reasoning') {
          previousReasoningSegments = retainedReasoningSegments.map(
            (value) => String(value || ''),
          )
        }
      }
      if (containsTransientProtocolMarker(traceChannelText(channelEvents, channel))) {
        failures.push(`message ${row?.messageId || 'unknown'} leaked parser markers across ${channel} stream boundaries`)
      }
    }
    // A never-empty notice is the panel telling the user the model produced no
    // usable answer. It is one substituted sentence, not a generated one, so
    // "streamed in at least two deltas" is not a property it can have. Every
    // other assertion still binds it (non-empty, stream-equals-persisted,
    // visibly rendered), and the surfaces the model failed to prove —
    // responses_delta_streaming, tool_loop — stay unrecorded, so a family
    // cannot pass the gate by emitting notices.
    const noticeTurn = isNeverEmptyNoticeTurn(
      result?.assistantRecords?.[
        (result?.assistantMessageIds || [])
          .map(String)
          .indexOf(String(row?.messageId || ''))
      ]?.content,
    )
    if (noticeTurn) neverEmptyNoticeTurns.push(String(row?.messageId || ''))
    if (progressiveContentDeltaCount < 2 && !noticeTurn) {
      failures.push(`message ${row?.messageId || 'unknown'} content was not progressively streamed`)
    }
    // The DOM sampler runs at a 125ms floor; a short answer from a fast model
    // can legitimately paint fully inside one window (seen: gemma4 26B at
    // ~104 t/s on a 3-line receipt). Only require two distinct rendered
    // answer states when the wire content stream spanned enough time for the
    // sampler to observe growth; wire-level progressiveness is asserted above.
    const contentStreamTimes = events
      .filter((event) => event?.event === 'stream' && event?.channel === 'content')
      .map((event) => Number(event?.t))
      .filter((value) => Number.isFinite(value))
    const contentStreamSpanMs = contentStreamTimes.length >= 2
      ? Math.max(...contentStreamTimes) - Math.min(...contentStreamTimes)
      : 0
    if (renderedAnswerStates.size < 2 && contentStreamSpanMs >= 750) {
      failures.push(`message ${row?.messageId || 'unknown'} visible answer DOM was not progressively updated`)
    }
    const terminalIndexes = events
      .map((event, index) => event?.event === 'terminal' ? index : -1)
      .filter((index) => index >= 0)
    const terminalIndex = terminalIndexes[0] ?? -1
    if (terminalIndexes.length !== 1) {
      failures.push(
        `message ${row?.messageId || 'unknown'} expected exactly one terminal event, got ${terminalIndexes.length}`,
      )
    }
    if (terminalIndex < 0) {
      failures.push(`message ${row?.messageId || 'unknown'} has no terminal event`)
      continue
    }
    if (terminalIndex !== events.length - 1) {
      failures.push(`message ${row?.messageId || 'unknown'} terminal event was not final`)
    }
    const assistantIndex = (result?.assistantMessageIds || [])
      .map(String)
      .indexOf(String(row?.messageId || ''))
    const persistedContent = String(result?.assistantRecords?.[assistantIndex]?.content || '')
    const finalContent = traceChannelText(events, 'content')
    // Same edge-whitespace normalization as the persistence writer
    // (main/ipc/chat.ts trims at tool-loop segment joins); see the paired
    // assertion above. Mid-content stays byte-exact.
    if (finalContent.trim() !== persistedContent.trim()) {
      failures.push(`message ${row?.messageId || 'unknown'} final content stream does not equal persisted final`)
    }
    if (!reasoningEvents.length) {
      // Some families (gemma4 via Responses) deliver reasoning as a single
      // terminal item rather than progressive deltas — the API probe accepts
      // the same shape (summary_text). Count the message as reasoning-bearing
      // when the reasoning is BOTH persisted and visibly rendered; the
      // progressive-stream checks below only bind delta-streamed reasoning.
      const renderedNonDelta = domMessages.find(
        (message) => String(message?.messageId || '') === String(row?.messageId || ''),
      )
      const persistedNonDelta = (result?.persistedReasoningByMessage?.[assistantIndex] || [])
        .filter((segment) => typeof segment === 'string')
        .join('')
        .trim()
      if (persistedNonDelta && String(renderedNonDelta?.reasoningText || '').trim()) {
        reasoningMessageCount += 1
      }
      continue
    }
    reasoningMessageCount += 1
    const progressiveReasoningDeltaCount = reasoningEvents
      .filter((event) => String(event?.delta || '').length > 0)
      .length
    const completeReasoningText = traceReasoningSegments(events).join('\n').trim()
    const singleTokenReasoning = /^\S+$/.test(completeReasoningText)
    if (progressiveReasoningDeltaCount < 2 && !singleTokenReasoning) {
      failures.push(`message ${row.messageId} reasoning was not progressively streamed`)
    }
    // Same never-empty-notice exemption as the content-channel check above: a
    // reasoning-only turn whose ANSWER is a substituted notice reaches THIS
    // branch (it is reasoning-bearing), and one substituted sentence cannot
    // arrive in two progressive deltas. Exempting only the other site left the
    // notice turn still failing here.
    if (progressiveContentDeltaCount < 2 && !noticeTurn) {
      failures.push(`message ${row.messageId} visible content was not progressively streamed`)
    }
    if (reasoningEvents.some((event) => event?.payload?.isReasoning !== true)) {
      failures.push(`message ${row.messageId} reasoning delta was not labeled separately`)
    }
    if (contentEvents.some((event) => event?.payload?.isReasoning === true)) {
      failures.push(`message ${row.messageId} content delta was mislabeled as reasoning`)
    }
    const rendered = domMessages.find(
      (message) => String(message?.messageId || '') === String(row.messageId || ''),
    )
    if (!String(rendered?.reasoningText || '').trim()) {
      failures.push(`message ${row.messageId} reasoning streamed but no visible reasoning rail was rendered`)
    }
    const renderedReasoningStates = new Set(
      messageDomSamples.map((sample) => String(sample?.reasoningText || '')).filter(Boolean),
    )
    if (renderedReasoningStates.size < 2) {
      failures.push(`message ${row.messageId} visible reasoning rail was not progressively updated`)
    }
    const persistedReasoningSegments = (
      result?.persistedReasoningByMessage?.[assistantIndex] || []
    )
    if (persistedReasoningSegments.some((segment) => typeof segment !== 'string')) {
      failures.push(`message ${row.messageId} persisted reasoning segments are not strings`)
    }
    const persistedReasoning = persistedReasoningSegments
      .filter((segment) => typeof segment === 'string')
      .filter(Boolean)
      .join('\n')
    const persistedSegments = persistedReasoningSegments
      .filter((segment) => typeof segment === 'string')
      .filter(Boolean)
    const finalReasoningSegments = traceReasoningSegments(events)
    if (canonicalJson(finalReasoningSegments) !== canonicalJson(persistedSegments)) {
      failures.push(
        `message ${row.messageId} final reasoning stream segments do not equal persisted reasoning segments`,
      )
    }
    const renderedReasoningSegments = Array.isArray(rendered?.reasoningSegments)
      ? rendered.reasoningSegments.map(String).filter(Boolean)
      : persistedSegments.length <= 1 && String(rendered?.reasoningText || '').trim()
        ? [String(rendered.reasoningText)]
        : []
    const persistedMathLinkage = persistedSegments.map(
      extractPersistedReasoningMathLinkage,
    )
    const persistedMathSources = persistedMathLinkage.map(
      (segment) => segment.mathSources,
    )
    const persistedLinkedSegments = persistedMathLinkage.map(
      (segment) => segment.linkedText,
    )
    const hasPersistedMath = persistedMathSources.some((sources) => sources.length > 0)
    const renderedLinkedSegments = Array.isArray(rendered?.reasoningLinkedSegments)
      ? rendered.reasoningLinkedSegments.map(String).filter(Boolean)
      : []
    const renderedMathSources = Array.isArray(rendered?.reasoningMathSources)
      ? rendered.reasoningMathSources.map((sources) => (
          Array.isArray(sources)
            ? sources.map((identity) => {
                const value = identity && typeof identity === 'object' ? identity : {}
                return {
                  source: String(value.source || ''),
                  delimiter: String(value.delimiter || ''),
                  displayMode: String(value.displayMode || ''),
                }
              })
            : []
        ))
      : []
    const renderedMathDisplayModes = Array.isArray(rendered?.reasoningMathDisplayModes)
      ? rendered.reasoningMathDisplayModes.map((modes) => (
          Array.isArray(modes)
            ? modes.map((mode) => {
                const value = mode && typeof mode === 'object' ? mode : {}
                return {
                  wrapperDisplayMode: String(value.wrapperDisplayMode || ''),
                  katexDisplayMode: String(value.katexDisplayMode || ''),
                }
              })
            : []
        ))
      : []
    const renderedMathIdentityCount = renderedMathSources
      .reduce((total, sources) => total + sources.length, 0)
    const renderedKatexCount = Array.isArray(rendered?.reasoningKatexCounts)
      ? rendered.reasoningKatexCounts.reduce((total, count) => total + Number(count || 0), 0)
      : 0
    const renderedKatexErrorCount = Array.isArray(rendered?.reasoningKatexErrorCounts)
      ? rendered.reasoningKatexErrorCounts.reduce(
          (total, count) => total + Number(count || 0),
          0,
        )
      : 0
    if (
      !hasPersistedMath
      && (renderedMathIdentityCount > 0 || renderedKatexCount > 0 || renderedKatexErrorCount > 0)
    ) {
      failures.push(
        `message ${row.messageId} rendered unexpected reasoning math absent from persisted reasoning`,
      )
    }
    const linkedSegmentsForComparison = hasPersistedMath
      ? renderedLinkedSegments
      : renderedReasoningSegments
    const persistedSegmentsForComparison = hasPersistedMath
      ? persistedLinkedSegments
      : persistedSegments
    if (
      hasPersistedMath
      && canonicalJson(renderedMathSources) !== canonicalJson(persistedMathSources)
    ) {
      failures.push(
        `message ${row.messageId} visible reasoning math identity is not exactly linked to persisted canonical TeX`,
      )
    }
    const hasMathDisplayMismatch = persistedMathSources.some((sources, segmentIndex) => {
      const renderedModes = renderedMathDisplayModes[segmentIndex]
      if (!Array.isArray(renderedModes) || renderedModes.length !== sources.length) return true
      return sources.some((identity, mathIndex) => {
        const renderedMode = renderedModes[mathIndex] || {}
        return (
          renderedMode.wrapperDisplayMode !== identity.displayMode
          || renderedMode.katexDisplayMode !== identity.displayMode
        )
      })
    })
    if (hasPersistedMath && hasMathDisplayMismatch) {
      failures.push(
        `message ${row.messageId} visible reasoning math inline/display mode does not match persisted delimiters`,
      )
    }
    if (
      hasPersistedMath
      && (
        !Array.isArray(rendered?.reasoningKatexCounts)
        || rendered.reasoningKatexCounts.length !== persistedMathSources.length
        || rendered.reasoningKatexCounts.some(
          (count, index) => Number(count) !== persistedMathSources[index].length,
        )
        || !Array.isArray(rendered?.reasoningKatexErrorCounts)
        || rendered.reasoningKatexErrorCounts.length !== persistedMathSources.length
        || rendered.reasoningKatexErrorCounts.some((count) => Number(count) > 0)
      )
    ) {
      failures.push(
        `message ${row.messageId} persisted reasoning TeX did not render cleanly through KaTeX`,
      )
    }
    if (
      canonicalJson(linkedSegmentsForComparison.map(normalizeVisibleLinkText))
      !== canonicalJson(persistedSegmentsForComparison.map(normalizeVisibleLinkText))
    ) {
      failures.push(
        `message ${row.messageId} normalized visible reasoning rail segments are not linked to persisted reasoning segments`,
      )
    }
  }
  if (expectation === 'required' && reasoningMessageCount !== assistantIds.size) {
    failures.push(
      `reasoning was required on all ${expectedTurns} turns but separate reasoning appeared on ${reasoningMessageCount}`,
    )
  }
  if (expectation === 'none' && reasoningMessageCount > 0) {
    failures.push('reasoning was disabled but reasoning deltas were observed')
  }
  if (expectation === 'none' && domMessages.some((message) => String(message?.reasoningText || '').trim())) {
    failures.push('reasoning was disabled but a visible reasoning rail was rendered')
  }
  if (result && typeof result === 'object') {
    result.neverEmptyNoticeTurns = neverEmptyNoticeTurns
  }
  // A run where EVERY visible answer is a never-empty notice produced no real
  // answers at all, and must not read as healthy. dots3-note tools-off did
  // exactly that across three budgets — reasoning of 11.4k then 23.9k
  // characters with every answer exactly the 98-character notice — and the run
  // still looked reasonable enough that I called it clean. The notice is the
  // correct product behaviour (it replaces a blank bubble); a run made ENTIRELY
  // of them is not a proof of anything the model said.
  const answeredTurns = (result?.assistantRecords || []).filter(
    (record) => String(record?.content || '').trim(),
  ).length
  if (answeredTurns > 0 && neverEmptyNoticeTurns.length >= answeredTurns) {
    failures.push(
      `every visible answer (${answeredTurns}) was a never-empty notice, so the model produced no answer text`,
    )
  }
  return failures
}

function flattenedPersistedTools(result) {
  return (Array.isArray(result?.persistedToolsByMessage)
    ? result.persistedToolsByMessage
    : [])
    .flatMap((group, messageIndex) => (
      Array.isArray(group)
        ? group.map((item) => ({ ...item, messageIndex }))
        : []
    ))
}

export function validateExactToolLoopEvidence(result) {
  const failures = []
  const expectedToolCalls = expectedUiToolCallCount(result)
  const statuses = flattenedPersistedTools(result)
  const statusCalls = statuses.filter((item) => item?.phase === 'calling')
  const errors = statuses.filter((item) => item?.phase === 'error')
  const calls = (result?.persistedOaiCallsByMessage || [])
    .flatMap((group, messageIndex) => (
      Array.isArray(group)
        ? group.map((item) => ({ ...item, messageIndex }))
        : []
    ))
  const results = (result?.persistedOaiResultsByMessage || [])
    .flatMap((group, messageIndex) => (
      Array.isArray(group)
        ? group.map((item) => ({ ...item, messageIndex }))
        : []
    ))
  const domCards = (result?.renderedDom?.messages || [])
    .flatMap((message) => message?.toolCards || [])
  if (expectedToolCalls === 0) {
    if (calls.length || results.length || statuses.length || domCards.length) {
      failures.push('tool call/result/status residue was observed in a no-tool UI proof')
    }
    return failures
  }
  // A FLOOR, not an equality. Settled by live evidence under the DEFAULT prompt
  // (which asks for the tool "once" per turn): LFM2.5-8B splits the prescribed
  // compound command into two calls — `printf %s REAL_UI_LIVE_TOOL_ONE >
  // real_ui_tool_probe_1.txt` then `cat real_ui_tool_probe_1.txt` — doing
  // exactly the prescribed work, every card successful, probe files byte-exact,
  // and turn 2's dependent command correct. Demanding equality withheld
  // tool_loop/long_tool_loop from a chain that demonstrably worked.
  //
  // (The stored LFM2.5/Step37 artifacts with five to seven calls are a separate
  // matter: those runs used a VMLINUX_REAL_UI_PROMPT_1 override that demanded
  // four calls on turn 1 while expectedUiToolCallCount derives 2 from the
  // profile. Those rows are regenerated with the default prompt.)
  //
  // What still fails, and must: a step missing from its own turn. The qwen36
  // case (ledger 220) made NO call on turn 1 and put both calls on turn 2, one
  // a single batched command doing both steps, so the per-turn chain never
  // happened. Every call — protocol or extra — must still have its own result,
  // a unique id and a visible card, no call may reach ahead to a later step's
  // probe, and each protocol step must show successful completion.
  if (calls.length < expectedToolCalls) {
    failures.push(`expected at least ${expectedToolCalls} tool calls, got ${calls.length}`)
  }
  const expected = [
    {
      file: 'real_ui_tool_probe_1.txt',
      token: 'REAL_UI_LIVE_TOOL_ONE',
      forbidden: 'real_ui_tool_probe_2.txt',
    },
    {
      file: 'real_ui_tool_probe_2.txt',
      token: 'REAL_UI_LIVE_TOOL_TWO',
      requiredAlso: 'real_ui_tool_probe_1.txt',
    },
  ]
  const expectedExecuted = expected.slice(0, expectedToolCalls)
  const probeFiles = result?.toolProbeFiles || {}
  for (const spec of expectedExecuted) {
    if (String(probeFiles?.[spec.file] || '') !== spec.token) {
      failures.push(`tool probe ${spec.file} did not contain exactly ${spec.token}`)
    }
  }
  // Resolve each protocol step to the call that satisfied it ON ITS OWN TURN
  // rather than by list position. With the default prompt (one run_command per
  // turn) this is equivalent to positional matching, but it reports the real
  // failure instead of a cascade: the qwen36 case (ledger 220) made NO call on
  // turn 1 and put both calls on turn 2 — one a single batched command doing
  // both steps — and positional matching turned that into ten separate
  // failures that hid which step was actually missing.
  //
  // The exact-count rule above stays. The stored LFM2.5/Step37 artifacts that
  // carried five to seven calls were produced with a VMLINUX_REAL_UI_PROMPT_1
  // override that demanded four separate calls on turn 1, while
  // expectedUiToolCallCount derives 2 from the profile — an operator/expectation
  // mismatch, not models adding calls of their own accord. Those rows are
  // regenerated with the default prompt, so tolerating surplus calls would
  // weaken the gate for no real case.
  const callsByTurn = new Map()
  for (const call of calls) {
    const turn = call.messageIndex ?? -1
    if (!callsByTurn.has(turn)) callsByTurn.set(turn, [])
    callsByTurn.get(turn).push(call)
  }
  const commandOf = (call) => extractToolCommand(
    call?.function?.arguments || call.detail || call.arguments,
  )
  const protocolCalls = expectedExecuted.map((spec, index) => {
    const turnCalls = callsByTurn.get(index) || []
    return turnCalls.find((call) => {
      const command = commandOf(call)
      if (!command.includes(spec.file) || !command.includes(spec.token)) return false
      if (spec.forbidden && command.includes(spec.forbidden)) return false
      if (spec.requiredAlso && !command.includes(spec.requiredAlso)) return false
      return true
    }) || null
  })
  // The ordering guarantee has to hold for EVERY call on the step's turn, not
  // only the one chosen: a model that reaches ahead to the second probe on turn
  // one has broken the dependency the chain is meant to prove.
  expectedExecuted.forEach((spec, index) => {
    if (!spec.forbidden) return
    for (const call of (callsByTurn.get(index) || [])) {
      if (commandOf(call).includes(spec.forbidden)) {
        failures.push(`tool call ${index + 1} referenced the second-turn probe prematurely`)
        break
      }
    }
  })
  const protocolCallIds = new Set(
    protocolCalls.filter(Boolean).map((call) => String(call.id || call.toolCallId || call.callId || '')),
  )
  const extraCalls = calls.filter(
    (call) => !protocolCallIds.has(String(call.id || call.toolCallId || call.callId || '')),
  )
  const protocolErrors = errors.filter(
    (status) => protocolCallIds.has(String(status?.toolCallId || '')),
  )
  if (protocolErrors.length) {
    failures.push(`tool loop contains ${protocolErrors.length} error status entries`)
  }
  // Correspondence is by ID, never by list position. An extensive-churn row can
  // emit dozens of calls and the visible status stream does not interleave in
  // the same order as the persisted arrays, so positional pairing produced a
  // cascade of meaningless "call ID/order does not match" failures on a 30-call
  // run — and buried the one finding that mattered: a VISIBLE calling status
  // (call_ee7e26d2) whose call was never persisted, i.e. the user saw a tool
  // call that is absent from the saved conversation.
  const persistedCallIds = new Set(
    calls.map((call) => String(call.id || call.toolCallId || call.callId || '')),
  )
  for (const status of statusCalls) {
    const statusId = String(status.toolCallId || '')
    if (!statusId || !persistedCallIds.has(statusId)) {
      failures.push(`visible tool status ${statusId || '(no id)'} has no persisted tool call`)
    }
  }
  for (const call of calls) {
    const id = String(call.id || call.toolCallId || call.callId || '')
    const own = statuses.filter((status) => String(status?.toolCallId || '') === id)
    if (!own.some((status) => status?.phase === 'calling')) {
      failures.push(`tool call ${id} has no visible calling status`)
    }
    const terminal = own.filter(
      (status) => status?.phase === 'result' || status?.phase === 'error',
    )
    if (terminal.length !== 1) {
      failures.push(`tool call ${id} did not reach exactly one terminal status, got ${terminal.length}`)
    }
    const statusName = String(
      own.find((status) => status?.phase === 'calling')?.toolName || '',
    )
    if (statusName !== String(call?.function?.name || call?.toolName || call?.name || '')) {
      failures.push(`tool call ${id} visible status name does not match the persisted call`)
    }
    if (!protocolCallIds.has(id)) {
      const matching = results.filter(
        (item) => String(item.tool_call_id || item.toolCallId || item.callId || '') === id,
      )
      if (matching.length !== 1) {
        failures.push(`tool call ${id} expected exactly one persisted result, got ${matching.length}`)
      }
    }
  }
  protocolCalls.forEach((call, index) => {
    if (!call) {
      failures.push(`tool call ${index + 1} was not persisted on assistant turn ${index + 1}`)
      return
    }
    const name = String(call?.function?.name || call.toolName || call.name || '')
    if (name !== 'run_command') failures.push(`tool call ${index + 1} used ${name || 'missing name'}`)
    const callId = String(call.id || call.toolCallId || call.callId || '')
    if (!callId) failures.push(`tool call ${index + 1} has no call ID`)
    const command = extractToolCommand(
      call?.function?.arguments || call.detail || call.arguments,
    )
    const spec = expected[index]
    // The resolver already required file/token/forbidden/requiredAlso to match,
    // so re-checking them here would be dead weight — except for the ordering
    // rule, which is enforced above across every call on the turn.
    if (!command.includes(spec.file) || !command.includes(spec.token)) {
      failures.push(`tool call ${index + 1} arguments do not contain its exact probe file/token`)
    }
    const matchingResults = results.filter(
      (item) => String(item.tool_call_id || item.toolCallId || item.callId || '') === callId,
    )
    if (matchingResults.length !== 1) {
      failures.push(`tool call ${index + 1} expected exactly one matching result, got ${matchingResults.length}`)
    } else {
      const matchingResult = matchingResults[0]
      if ((matchingResult.messageIndex ?? -1) !== (call.messageIndex ?? -1)) {
        failures.push(`tool call ${index + 1} result was not persisted on the same assistant turn`)
      } else if (!JSON.stringify(matchingResult).includes(spec.token)) {
        failures.push(`tool call ${index + 1} result did not preserve ${spec.token}`)
      }
    }
    const callStatuses = statuses.filter(
      (status) => String(status?.toolCallId || '') === callId,
    )
    if (callStatuses.filter((status) => status?.phase === 'calling').length !== 1) {
      failures.push(`tool call ${index + 1} did not have exactly one calling status`)
    }
    if (callStatuses.filter((status) => status?.phase === 'result').length !== 1) {
      failures.push(`tool call ${index + 1} did not have exactly one successful result status`)
    }
    if (callStatuses.some((status) => status?.phase === 'error')) {
      failures.push(`tool call ${index + 1} contains an error status`)
    }
    const trace = (result?.messageEventTrace || []).find(
      (row) => String(row?.messageId || '') === String(result?.assistantMessageIds?.[index] || ''),
    )
    const toolEvents = (trace?.events || []).filter(
      (event) => (
        event?.event === 'tool'
        && String(event?.payload?.toolCallId || '') === callId
      ),
    )
    const callingIndex = toolEvents.findIndex((event) => event?.payload?.phase === 'calling')
    const resultIndex = toolEvents.findIndex((event) => event?.payload?.phase === 'result')
    if (callingIndex < 0 || resultIndex <= callingIndex) {
      failures.push(`tool call ${index + 1} live status order was not calling then result`)
    }
    const continuation = String(result?.assistantRecords?.[index]?.content || '')
    if (!continuation.includes(spec.token)) {
      failures.push(`tool call ${index + 1} had no visible result continuation containing ${spec.token}`)
    }
  })
  const callIds = calls.map(
    (call) => String(call.id || call.toolCallId || call.callId || ''),
  ).filter(Boolean)
  if (new Set(callIds).size !== callIds.length) failures.push('tool call IDs are not unique')
  const resultIds = results.map(
    (item) => String(item.tool_call_id || item.toolCallId || item.callId || ''),
  ).filter(Boolean)
  if (new Set(resultIds).size !== resultIds.length) failures.push('tool result IDs are not unique')
  if (resultIds.some((callId) => !callIds.includes(callId))) {
    failures.push('tool result exists without its exact persisted call')
  }
  // One card per call the model actually made, protocol or surplus — every
  // call must be VISIBLE to the user, but only the protocol calls have to show
  // successful completion. A surplus verification call that errored is the
  // model's business; hiding it would be the product's.
  if (domCards.length !== calls.length) {
    failures.push(`expected one rendered tool card per call (${calls.length}), got ${domCards.length}`)
  }
  for (const callId of callIds) {
    const matchingCards = domCards.filter((card) => String(card?.callId || '') === callId)
    if (matchingCards.length !== 1) {
      failures.push(`tool call ${callId} has no matching rendered tool card`)
      continue
    }
    const card = matchingCards[0]
    if (card.visible !== true) {
      failures.push(`tool call ${callId} rendered card was not visible`)
    }
    if (
      protocolCallIds.has(callId)
      && !['result', 'done', 'complete'].includes(String(card.phase || ''))
    ) {
      failures.push(`tool call ${callId} rendered card did not show successful completion`)
    }
    const result = results.find(
      (item) => String(item.tool_call_id || item.toolCallId || item.callId || '') === callId,
    )
    const call = calls.find(
      (item) => String(item.id || item.toolCallId || item.callId || '') === callId,
    )
    const callName = String(call?.function?.name || call?.toolName || call?.name || '')
    if (!result || String(card.name || '') !== callName) {
      failures.push(`tool call ${callId} rendered card ID/name is not linked to its persisted result`)
    }
  }
  if (result?.toolProbeCleanup?.removed !== true) {
    failures.push('tool probe files were not cleaned after evidence capture')
  }
  const cleanupPaths = Array.isArray(result?.toolProbeCleanup?.paths)
    ? result.toolProbeCleanup.paths.map((item) => path.resolve(String(item)))
    : []
  const expectedCleanupPaths = expected.map((spec) =>
    path.resolve(String(result?.workingDirectory || ''), spec.file),
  )
  if (
    cleanupPaths.length !== expectedCleanupPaths.length
    || canonicalJson(cleanupPaths.sort()) !== canonicalJson(expectedCleanupPaths.sort())
  ) {
    failures.push('tool probe cleanup paths are not bound to the exact configured working directory')
  }
  const thirdTools = Array.isArray(result?.persistedToolsByMessage?.[2])
    ? result.persistedToolsByMessage[2]
    : []
  if (expectedUiTurnCount(result) >= 3 && thirdTools.some((item) => item?.phase === 'calling')) {
    failures.push('third assistant turn emitted an unexpected tool call')
  }
  const thirdOaiCalls = Array.isArray(result?.persistedOaiCallsByMessage?.[2])
    ? result.persistedOaiCallsByMessage[2]
    : []
  if (expectedUiTurnCount(result) >= 3 && thirdOaiCalls.length) {
    failures.push('third assistant turn persisted an unexpected tool call')
  }
  return failures
}

export function validateGenerationDefaultsEvidence(result) {
  const failures = []
  const expectedTurns = expectedUiTurnCount(result)
  const defaults = result?.bundleGenerationContract?.defaults
  const rendererDefaults = result?.rendererGenerationDefaults
  const dom = result?.chatSettingsDom || {}
  const effective = result?.server?.health?.effective_defaults || {}
  const resolved = result?.resolvedSamplingKwargs || {}
  const resolvedRecords = (Array.isArray(result?.resolvedSamplingRecords)
    ? result.resolvedSamplingRecords
    : [])
    .filter((record) => record?.values && typeof record.values === 'object')
  const explicit = result?.requestContract?.samplingOverrides || {}
  const persistedNativeMtpMode = result?.nativeMtpSelection?.persistedMode
    ?? result?.effectiveSessionConfig?.nativeMtpMode
    ?? result?.serverCacheControls?.persistedConfig?.nativeMtpMode
    ?? 'auto'
  const nativeMtpGreedy = result?.server?.health?.mtp?.runtime_active === true
    && persistedNativeMtpMode === 'deterministic'
  const explicitFields = Object.entries(explicit).filter(([, value]) => value != null)
  const turnEvidence = Array.isArray(result?.uiTurnEvidence)
    ? result.uiTurnEvidence.slice(0, expectedTurns)
    : []
  const settingsInteraction = result?.chatSettingsInteraction || {}
  const requestCorrelationVerified = isServerRequestCorrelationVerified(result)
  const mapping = [
    ['temperature', 'temperature', 'temperature', 'temperature'],
    ['topP', 'top_p', 'topP', 'top_p'],
    ['topK', 'top_k', 'topK', 'top_k'],
    ['minP', 'min_p', 'minP', 'min_p'],
    ['repeatPenalty', 'repetition_penalty', 'repeatPenalty', 'repetition_penalty'],
    ['maxNewTokens', 'max_output_tokens', 'maxTokens', 'max_tokens'],
  ]
  if (!defaults) {
    // A bundle whose generation_config stamps no sampler fields has no
    // model-owned defaults BY DESIGN; the truthful evidence is the drawer's
    // engine-fallback banner plus verified request correlation (the engine's
    // resolved kwargs ARE the live defaults). Seen on Step-3.7-Flash-JANG_K.
    if (
      settingsInteraction.noSamplerDefaultsBanner === true
      && requestCorrelationVerified
    ) {
      return failures
    }
    failures.push('bundle has no independently resolved generation defaults')
    return failures
  }
  if (!requestCorrelationVerified) {
    failures.push(
      'resolved generation defaults lack exact proof/request/message correlation for every UI wire request',
    )
  }
  if (settingsInteraction.openedVisibly !== true) {
    failures.push('Chat Settings was not opened through its visible control')
  }
  if (
    (explicitFields.length
      || result?.requestContract?.requestMaxTokens != null
      || result?.requestedBuiltinTools === true
      || result?.requestedWireApi)
    && (
      settingsInteraction.savedViaVisibleControl !== true
      || settingsInteraction.reopenedAfterSave !== true
    )
  ) {
    failures.push('requested Chat Settings were not visibly changed, saved, and reopened')
  }
  if (requestCorrelationVerified && resolvedRecords.length < expectedTurns) {
    failures.push(
      `expected resolved sampling kwargs for at least ${expectedTurns} UI requests, got ${resolvedRecords.length}`,
    )
  }
  if (turnEvidence.length !== expectedTurns) {
    failures.push(`expected exactly ${expectedTurns} bound UI turn records, got ${turnEvidence.length}`)
  }
  const expectedRoute = result?.requestedWireApi === 'responses'
    ? '/v1/responses'
    : '/v1/chat/completions'
  // Sampling logs name the API-facing served model. The filesystem bundle is
  // independently bound by model_bundle_provenance and must never be compared
  // as though it were the served identifier.
  const expectedModel = String(result?.servedModel || '')
  const expectedWire = result?.requestedWireApi === 'responses'
    ? 'responses'
    : 'completions'
  const expectedPersistedThinking = result?.requestedEnableThinking === undefined
    && result?.requestedReasoningEffort != null
    ? true
    : result?.requestedEnableThinking
  const acceptableReasoningLabels = expectedPersistedThinking === true
    ? ['On', 'Reasoning']
    : expectedPersistedThinking === false
      ? ['Off', 'Instruct']
      : ['Auto']
  if (String(dom?.wireApi || '') !== expectedWire) {
    failures.push(`reopened visible wire format ${dom?.wireApi || 'missing'} does not match ${expectedWire}`)
  }
  // Notice families (thinkingNotConfigurable: LFM2.5, MiniMax) render the
  // honesty notice instead of mode buttons — a missing mode control there is
  // the CORRECT persisted state for Auto. An explicit requested override
  // against the notice is already fatal at interaction time.
  const noticeWithAutoRequested = dom?.thinkingNotice === true
    && result?.requestedEnableThinking !== true
    && result?.requestedEnableThinking !== false
    && !dom?.reasoningMode
  if (
    !noticeWithAutoRequested
    && !acceptableReasoningLabels.includes(String(dom?.reasoningMode || ''))
  ) {
    failures.push('reopened visible reasoning mode does not match the requested mode')
  }
  if (dom?.builtinToolsEnabled !== (result?.requestedBuiltinTools === true)) {
    failures.push('reopened visible built-in tools state does not match the requested state')
  }
  if (result?.requestedBuiltinTools === true) {
    const toolExpectations = [
      [dom?.workingDirectory, result?.workingDirectory, 'working directory'],
      [
        Number(dom?.maxToolIterations),
        Number(result?.requestContract?.maxToolIterations),
        'max tool iterations',
      ],
      [
        Number(dom?.toolResultMaxChars),
        Number(result?.requestContract?.toolResultMaxChars),
        'tool result limit',
      ],
    ]
    for (const [observed, expected, label] of toolExpectations) {
      if (observed !== expected) {
        failures.push(`reopened visible ${label} does not match the requested value`)
      }
    }
    const categoryExpected = {
      file: true,
      search: false,
      shell: true,
      webSearch: false,
      urlFetch: false,
      git: false,
      utilities: true,
    }
    for (const [key, expected] of Object.entries(categoryExpected)) {
      if (dom?.toolCategories?.[key] !== expected) {
        failures.push(`reopened visible tool category ${key} does not match the requested value`)
      }
    }
  }
  for (let turn = 1; turn <= expectedTurns; turn += 1) {
    const evidence = turnEvidence.find((row) => Number(row?.turn) === turn)
    if (
      !evidence?.userMessageId
      || !evidence?.assistantMessageId
      || String(evidence?.prompt || '') !== String(result?.requestContract?.[`prompt${['', 'One', 'Two', 'Three'][turn]}`] || '')
    ) {
      failures.push(`UI turn ${turn} is not bound to its prompt, user message, and assistant message`)
    }
    if (requestCorrelationVerified) {
      const records = resolvedRecords.filter(
        (record) => String(record?.proof_request_id || '') === String(evidence?.proofRequestId || ''),
      )
      if (!records.length) {
        failures.push(`UI turn ${turn} has no server-correlated resolved sampling record`)
        continue
      }
      for (const record of records) {
        if (
          String(record?.route || '') !== expectedRoute
          || String(record?.model || '') !== expectedModel
        ) {
          failures.push(`UI turn ${turn} resolved log route/model is not bound to the requested session`)
        }
        if (
          String(record?.proof_request_id || '') !== String(evidence?.proofRequestId || '')
          || String(record?.message_id || '') !== String(evidence?.assistantMessageId || '')
          || !String(record?.request_id || '')
        ) {
          failures.push(`UI turn ${turn} resolved log lacks exact server-emitted request/message correlation`)
        }
      }
    }
  }
  // A temperature-0 request is greedy, and the engine deliberately refuses to
  // forward omitted bundle sampler filters on it:
  // _normalize_deterministic_sampling_filters() forces top_p=1.0 and drops
  // top_k so logs, MTP policy checks, and exact-output gates see an
  // unambiguous deterministic request (pinned by
  // test_engine_audit.py::test_temperature_zero_omits_bundle_sampling_filters_at_route_layer).
  // Asserting bundle parity on those two fields for a greedy turn would
  // contradict the engine contract - and this proof's own manifest row runs at
  // temperature 0, so the assertion could never hold as specified. Explicit
  // request overrides are still asserted exactly; only omitted values relax.
  const explicitTemperatureEntry = explicitFields.find(
    ([key]) => key === 'temperature',
  )
  const effectiveTemperature = explicitTemperatureEntry
    ? explicitTemperatureEntry[1]
    : nativeMtpGreedy
      ? 0
      : defaults.temperature
  const greedyRequest = Number(effectiveTemperature) === 0
  for (const [bundleKey, engineKey, uiKey, resolvedKey] of mapping) {
    const expected = defaults[bundleKey]
    if (expected == null) continue
    const effectiveExpected = nativeMtpGreedy
      ? bundleKey === 'temperature'
        ? 0
        : bundleKey === 'topP'
          ? 1
          : bundleKey === 'topK' || bundleKey === 'minP'
            ? 0
            : expected
      : expected
    const overrideEntry = explicitFields.find(([key]) => key === uiKey)
    const maxTokensOverride = bundleKey === 'maxNewTokens'
      ? result?.requestContract?.requestMaxTokens
      : undefined
    const requestOverride = maxTokensOverride ?? (overrideEntry ? overrideEntry[1] : undefined)
    const uiExpected = requestOverride ?? effectiveExpected
    const rendererValue = rendererDefaults?.[bundleKey]
    if (!approximatelyEqual(Number(rendererValue), Number(expected))) {
      failures.push(`renderer default ${bundleKey}=${rendererValue} does not match bundle ${expected}`)
    }
    const uiValue = dom?.values?.[uiKey]
    if (bundleKey === 'maxNewTokens') {
      if (requestOverride == null) {
        if (String(dom?.maxTokens?.value || '') !== '') {
          failures.push('default run saved maxTokens instead of inheriting the model default')
        }
        if (!String(dom?.maxTokens?.placeholder || '').includes(String(expected))) {
          failures.push(`maxTokens placeholder does not expose model default ${expected}`)
        }
      } else if (!approximatelyEqual(Number(dom?.maxTokens?.value), Number(uiExpected))) {
        failures.push(`visible UI maxTokens=${dom?.maxTokens?.value} does not match override ${uiExpected}`)
      }
    } else if (!approximatelyEqual(Number(uiValue), Number(uiExpected))) {
      failures.push(
        `visible UI ${uiKey}=${uiValue} does not match ${requestOverride == null && nativeMtpGreedy ? 'Native-MTP effective default' : requestOverride == null ? 'bundle' : 'override'} ${uiExpected}`,
      )
    }
    const healthValue = numericField(effective, engineKey)
    if (!approximatelyEqual(Number(healthValue), Number(expected))) {
      failures.push(`health effective ${engineKey}=${healthValue} does not match bundle ${expected}`)
    }
    if (requestCorrelationVerified) {
      // Greedy neutralization applies only to values the request did NOT set.
      const greedyNeutralized = greedyRequest && requestOverride == null
      const resolvedExpected = greedyNeutralized && engineKey === 'top_p'
        ? 1
        : requestOverride ?? effectiveExpected
      const greedyOmitted = greedyNeutralized && engineKey === 'top_k'
      const resolvedValue = numericField(resolved, resolvedKey, engineKey, bundleKey, uiKey)
      const resolvedSentinelOmitted = (
        (engineKey === 'top_k' || engineKey === 'min_p')
        && Number(resolvedExpected) === 0
        && (resolvedValue === undefined || resolvedValue === null)
      )
      const resolvedGreedyOmitted = greedyOmitted
        && (resolvedValue === undefined || resolvedValue === null)
      if (
        !resolvedSentinelOmitted
        && !resolvedGreedyOmitted
        && !approximatelyEqual(Number(resolvedValue), Number(resolvedExpected))
      ) {
        failures.push(`resolved request ${engineKey}=${resolvedValue} does not match ${resolvedExpected}`)
      }
      resolvedRecords.forEach((record, index) => {
        const turnValue = numericField(
          record.values,
          resolvedKey,
          engineKey,
          bundleKey,
          uiKey,
        )
        const turnSentinelOmitted = (
          (engineKey === 'top_k' || engineKey === 'min_p')
          && Number(resolvedExpected) === 0
          && (turnValue === undefined || turnValue === null)
        )
        const turnGreedyOmitted = greedyOmitted
          && (turnValue === undefined || turnValue === null)
        if (
          !turnSentinelOmitted
          && !turnGreedyOmitted
          && !approximatelyEqual(Number(turnValue), Number(resolvedExpected))
        ) {
          failures.push(
            `resolved request record ${index + 1} ${engineKey}=${turnValue} does not match ${resolvedExpected}`,
          )
        }
      })
    }
  }
  const stored = result?.chatOverrides || {}
  if (stored.wireApi !== expectedWire) {
    failures.push(`persisted wire format ${stored.wireApi || 'missing'} does not match ${expectedWire}`)
  }
  if (Boolean(stored.builtinToolsEnabled) !== (result?.requestedBuiltinTools === true)) {
    failures.push('persisted built-in tools state does not match the requested state')
  }
  if (
    result?.requestedEnableThinking === undefined
      ? (result?.requestedReasoningEffort != null
        ? stored.enableThinking !== true
        : stored.enableThinking != null)
      : stored.enableThinking !== result.requestedEnableThinking
  ) {
    failures.push('persisted reasoning mode does not match the requested mode')
  }
  if (result?.requestedReasoningEffort != null) {
    if (stored.reasoningEffort !== result.requestedReasoningEffort) {
      failures.push('persisted reasoning effort does not match the requested effort')
    }
    if (dom?.reasoningEffort !== result.requestedReasoningEffort) {
      failures.push('reopened visible reasoning effort does not match the requested effort')
    }
  }
  if (result?.requestedMidConvReasoningEffort != null) {
    const flip = result?.midConvReasoningFlip || {}
    if (
      flip.buttonFound !== true
      || flip.saved !== true
      || flip.persistedReasoningEffortAfter !== result.requestedMidConvReasoningEffort
    ) {
      failures.push('mid-conversation reasoning effort was not visibly changed and persisted')
    }
  }
  if (result?.requestedBuiltinTools === true) {
    const storedToolExpectations = [
      [stored.workingDirectory, result?.workingDirectory, 'working directory'],
      [
        Number(stored.maxToolIterations),
        Number(result?.requestContract?.maxToolIterations),
        'max tool iterations',
      ],
      [
        Number(stored.toolResultMaxChars),
        Number(result?.requestContract?.toolResultMaxChars),
        'tool result limit',
      ],
    ]
    for (const [observed, expected, label] of storedToolExpectations) {
      if (observed !== expected) {
        failures.push(`persisted ${label} does not match the requested value`)
      }
    }
  }
  const hasMaxTokensOverride = result?.requestContract?.requestMaxTokens != null
  if (!explicitFields.length && !hasMaxTokensOverride) {
    for (const field of ['temperature', 'topP', 'topK', 'minP', 'repeatPenalty', 'maxTokens']) {
      if (stored[field] != null) failures.push(`default run persisted synthetic ${field} override`)
    }
  } else {
    for (const [field, value] of explicitFields) {
      const mappingRow = mapping.find(([, , uiKey]) => uiKey === field)
      const resolvedKey = mappingRow?.[3]
      const visibleMatches = approximatelyEqual(
        Number(dom?.values?.[field]),
        Number(value),
      )
      const everyResolvedRequestMatches = resolvedKey != null
        && resolvedRecords.length >= expectedTurns
        && resolvedRecords.every((record) => approximatelyEqual(
          Number(numericField(record.values, resolvedKey, field)),
          Number(value),
        ))
      const storedValue = stored[field]
      const effectivelyPersisted = storedValue == null
        ? visibleMatches && everyResolvedRequestMatches
        : approximatelyEqual(Number(storedValue), Number(value))
      if (!effectivelyPersisted) {
        failures.push(`explicit ${field} override=${value} was not persisted exactly`)
      }
    }
    if (
      hasMaxTokensOverride
      && !approximatelyEqual(
        Number(stored.maxTokens),
        Number(result.requestContract.requestMaxTokens),
      )
    ) {
      failures.push(
        `explicit maxTokens override=${result.requestContract.requestMaxTokens} was not persisted exactly`,
      )
    }
  }
  return failures
}

export function validateServerCacheEvidence(result) {
  const failures = []
  if (result?.requestedServerCacheControls !== true) return failures
  const evidence = result?.serverCacheControls || {}
  const config = result?.session?.effective_config || {}
  const health = result?.server?.health || {}
  const argv = Array.isArray(evidence.argv) ? evidence.argv.map(String) : []
  const visible = evidence.initialCacheControls || {}
  const nativeCache = health?.native_cache || {}
  const promptDiskOnly = nativeCache.prompt_disk_l2_configured === true
    && nativeCache.block_disk_l2_configured !== true
  const requestedBlockDiskPercent = result?.requestedBlockDiskCacheMaxPercent
  const expectedDsv4PoolQuant = result?.expectedDsv4PoolQuant
  if (evidence.runningSessionDrawer !== true) failures.push('cache controls were not inspected on the running session')
  if (evidence.controlScope !== 'running-session-toolbar') {
    failures.push('cache controls were not opened from the running-session toolbar')
  }
  for (const field of [
    'enablePrefixCache',
    'usePagedCache',
    promptDiskOnly ? 'enableDiskCache' : 'enableBlockDiskCache',
  ]) {
    if (typeof visible[field] !== 'boolean') {
      failures.push(`running-session visible cache control ${field} is missing`)
    }
  }
  if (visible.enablePrefixCache !== config.enablePrefixCache) {
    failures.push('visible prefix-cache state does not match persisted session config')
  }
  if (visible.usePagedCache !== config.usePagedCache) {
    failures.push('visible paged-cache state does not match persisted session config')
  }
  if (promptDiskOnly) {
    if (visible.enableDiskCache !== true) {
      failures.push('running-session exact prompt SSD/L2 control was not visibly enabled')
    }
    if (visible.enableDiskCache !== config.enableDiskCache) {
      failures.push('visible prompt SSD/L2 state does not match persisted session config')
    }
    if (visible.enableBlockDiskCache !== config.enableBlockDiskCache) {
      failures.push('visible block SSD/L2 state does not match persisted session config')
    }
    if (config.enableDiskCache !== true) {
      failures.push('persisted session config did not enable exact prompt disk cache')
    }
    if (config.enableBlockDiskCache !== false) {
      failures.push('persisted session config enabled generic block disk cache for an exact prompt-disk lane')
    }
    if (!argv.includes('--enable-disk-cache')) failures.push('engine argv omitted --enable-disk-cache')
    if (argv.includes('--enable-block-disk-cache')) {
      failures.push('engine argv enabled generic block disk cache for an exact prompt-disk lane')
    }
    if (nativeCache.prompt_disk_l2 !== true) {
      failures.push('/health did not report native_cache.prompt_disk_l2=true')
    }
    if (nativeCache.block_disk_l2 === true) {
      failures.push('/health reported generic block disk L2 on an exact prompt-disk lane')
    }
  } else {
    if (visible.enableBlockDiskCache !== true) failures.push('running-session SSD/L2 control was not visibly enabled')
    if (visible.enableBlockDiskCache !== config.enableBlockDiskCache) {
      failures.push('visible SSD/L2 state does not match persisted session config')
    }
    if (config.enableBlockDiskCache !== true) failures.push('persisted session config did not enable block disk cache')
    if (!argv.includes('--enable-block-disk-cache')) failures.push('engine argv omitted --enable-block-disk-cache')
    if (nativeCache.block_disk_l2 !== true) failures.push('/health did not report native_cache.block_disk_l2=true')
  }
  if (requestedBlockDiskPercent != null) {
    if (promptDiskOnly) {
      failures.push('block-disk percentage was requested for an exact prompt-disk lane')
    }
    if (!approximatelyEqual(
      Number(visible.blockDiskCacheMaxPercent),
      Number(requestedBlockDiskPercent),
    )) {
      failures.push('visible SSD cache percentage does not match the requested aggregate budget')
    }
    if (!approximatelyEqual(
      Number(config.blockDiskCacheMaxPercent),
      Number(requestedBlockDiskPercent),
    )) {
      failures.push('persisted SSD cache percentage does not match the requested aggregate budget')
    }
    const percentFlagIndex = argv.indexOf('--block-disk-cache-max-percent')
    const escapedPercentFlag = '--block-disk-cache-max-percent'
      .replace(/[.*+?^${}()|[\]\\]/g, '\\$&')
    const commandLinePercentMatch = String(evidence.commandLine || '').match(
      new RegExp(`(?:^|\\s)${escapedPercentFlag}(?:=|\\s+)([^\\s]+)`),
    )
    const argvPercentCandidate = percentFlagIndex >= 0
      ? argv[percentFlagIndex + 1]
      : undefined
    const percentFlagValue = argvPercentCandidate != null
      && !String(argvPercentCandidate).startsWith('--')
      ? argvPercentCandidate
      : commandLinePercentMatch?.[1]
    if (
      percentFlagIndex < 0
      || !approximatelyEqual(
        Number(percentFlagValue),
        Number(requestedBlockDiskPercent),
      )
    ) {
      failures.push('engine argv omitted or changed the requested aggregate SSD cache percentage')
    }
    if (argv.includes('--block-disk-cache-max-gb')) {
      failures.push('engine argv emitted a flat GB cap that overrides the requested SSD percentage')
    }
    const blockDiskStats = health?.cache?.block_disk_cache
      || health?.block_disk_cache
      || {}
    const finiteBudgetRecorded = Number(blockDiskStats.max_size_gb) > 0
      || Number(blockDiskStats.max_size_bytes) > 0
      || Number(blockDiskStats.global_budget?.max_size_bytes) > 0
    if (!finiteBudgetRecorded && Number(requestedBlockDiskPercent) > 0) {
      failures.push('/health cache telemetry omitted the finite resolved SSD cache budget')
    }
  }
  if (typeof config.enablePrefixCache !== 'boolean') {
    failures.push('persisted session config omitted enablePrefixCache')
  } else {
    if (config.enablePrefixCache) {
      if (argv.includes('--disable-prefix-cache')) {
        failures.push('engine argv contradicted enabled prefix cache with --disable-prefix-cache')
      }
    } else if (!argv.includes('--disable-prefix-cache')) {
      failures.push('engine argv omitted --disable-prefix-cache')
    }
    if (Boolean(nativeCache.prefix) !== config.enablePrefixCache) {
      failures.push('/health native prefix state does not match persisted session config')
    }
  }
  if (typeof config.usePagedCache !== 'boolean') {
    failures.push('persisted session config omitted usePagedCache')
  } else {
    const expectedFlag = config.usePagedCache ? '--use-paged-cache' : '--no-paged-cache'
    const contradictoryFlag = config.usePagedCache ? '--no-paged-cache' : '--use-paged-cache'
    if (!argv.includes(expectedFlag)) failures.push(`engine argv omitted ${expectedFlag}`)
    if (argv.includes(contradictoryFlag)) failures.push(`engine argv contains contradictory ${contradictoryFlag}`)
    if (Boolean(nativeCache.paged) !== config.usePagedCache) {
      failures.push('/health native paged state does not match persisted session config')
    }
    if (!promptDiskOnly && Boolean(nativeCache.block_disk_only) !== !config.usePagedCache) {
      failures.push('/health block-disk-only state does not match persisted paged-cache state')
    }
  }
  if (typeof expectedDsv4PoolQuant === 'boolean') {
    const poolQuant = nativeCache.pool_quant || {}
    if (poolQuant.requested !== expectedDsv4PoolQuant) {
      failures.push('/health DSV4 pool quant requested state does not match the bundle-derived expectation')
    }
    if (poolQuant.observed !== expectedDsv4PoolQuant) {
      failures.push('/health DSV4 pool quant observed cache class does not match the bundle-derived expectation')
    }
    if (poolQuant.enabled !== expectedDsv4PoolQuant) {
      failures.push('/health DSV4 pool quant enabled state does not match the bundle-derived expectation')
    }
    if (poolQuant.matches_request !== true) {
      failures.push('/health DSV4 pool quant requested/observed attestation did not match')
    }
    if (poolQuant.error) {
      failures.push(`/health DSV4 pool quant attestation reported an error: ${poolQuant.error}`)
    }
  }
  failures.push(...validateRequestCorrelatedCacheEvidence(result))
  failures.push(...validateNativeMtpSurfaceParity(result))
  return failures
}

// The Native MTP control must render exactly when the ENGINE says the bundle
// carries MTP weights — not when the bundle NAME happens to say so. A control
// on a model that cannot use it is the dead-toggle class of bug; a missing
// control on a model that can is an unreachable feature. Both arms were
// observed live before this was pinned:
//   Qwen3.6-35B-A3B-MXFP8-CRACK-MTP  index_has_mtp_tensors=true,  31 tensors
//     -> label visible, selector present, options [auto, deterministic, off]
//   Nemotron-Omni-Nano-JANGTQ-CRACK  index_has_mtp_tensors=false, 0 tensors
//     -> no label, no selector, no mention anywhere in the drawer
// Deliberately NOT asserted: the blocked-fallback direction. A bundle whose
// weights are present but whose compatibility gate fails has never been
// observed here, and pinning an unobserved expectation is what made two
// earlier rows unpassable. It is recorded instead, so the first real
// occurrence shows up in the artifact rather than as a mystery failure.
export function validateNativeMtpSurfaceParity(result) {
  const failures = []
  if (result?.requestedServerCacheControls !== true) return failures
  const surface = result?.serverCacheControls?.nativeMtpControl
  const mtp = result?.server?.health?.mtp
  const requestedMode = result?.requestContract?.nativeMtpMode
  const requestedDepth = result?.requestContract?.nativeMtpDepth
  const selection = result?.nativeMtpSelection
  if (!surface || typeof surface !== 'object') {
    failures.push('server cache controls were inspected but the Native MTP surface was not captured')
    return failures
  }
  if (!mtp || typeof mtp !== 'object') {
    failures.push('/health omitted the mtp block, so the Native MTP surface could not be checked against the engine')
    return failures
  }
  if (typeof mtp.index_has_mtp_tensors !== 'boolean') {
    failures.push('/health mtp.index_has_mtp_tensors is not a boolean, so UI/engine MTP parity is unverifiable')
    return failures
  }
  if (mtp.index_has_mtp_tensors === true) {
    if (surface.labelVisible !== true) {
      failures.push('engine reports MTP weights in the bundle but the UI rendered no Native MTP control')
    }
    if (surface.modeSelectPresent !== true) {
      failures.push('engine reports MTP weights in the bundle but the UI rendered no Native MTP mode selector')
    }
    const options = Array.isArray(surface.modeOptions) ? surface.modeOptions.map(String) : []
    for (const mode of ['auto', 'deterministic', 'off']) {
      if (options.length && !options.includes(mode)) {
        failures.push(`Native MTP mode selector is missing the ${mode} option`)
      }
    }
    if (mtp.runtime_active === true && surface.blockedFallbackNoticeShown === true) {
      failures.push('engine reports the native MTP runtime ACTIVE but the UI showed the blocked-fallback notice')
    }
  } else if (surface.labelVisible === true || surface.modeSelectPresent === true) {
    failures.push('engine reports no MTP weights in the bundle but the UI rendered a Native MTP control')
  }
  if (requestedMode != null) {
    if (!selection || selection.requested !== true) {
      failures.push('requested Native MTP selection was not captured before Start')
    } else {
      if (selection.selectedMode !== requestedMode) {
        failures.push(`visible Native MTP mode ${selection.selectedMode || 'missing'} did not match requested ${requestedMode}`)
      }
      if (selection.persistedMode !== requestedMode) {
        failures.push(`persisted Native MTP mode ${selection.persistedMode || 'missing'} did not match requested ${requestedMode}`)
      }
    }
    if (surface?.selectedMode !== requestedMode) {
      failures.push(`running-session Native MTP mode ${surface?.selectedMode || 'missing'} did not match requested ${requestedMode}`)
    }
  }
  if (requestedDepth != null) {
    if (selection?.selectedDepthPolicy !== 'fixed') {
      failures.push('visible Native MTP depth policy was not fixed before Start')
    }
    if (Number(selection?.selectedDepth) !== Number(requestedDepth)) {
      failures.push(`visible Native MTP depth ${selection?.selectedDepth ?? 'missing'} did not match requested D${requestedDepth}`)
    }
    if (
      selection?.persistedDepthOverride !== true
      || Number(selection?.persistedDepth) !== Number(requestedDepth)
    ) {
      failures.push(`persisted Native MTP depth did not retain fixed D${requestedDepth}`)
    }
    if (
      result?.effectiveSessionConfig?.nativeMtpDepthOverride !== true
      || Number(result?.effectiveSessionConfig?.nativeMtpDepth) !== Number(requestedDepth)
    ) {
      failures.push(`started session did not retain fixed Native MTP D${requestedDepth}`)
    }
    if (mtp.runtime_active !== true) {
      failures.push(`requested fixed Native MTP D${requestedDepth} but /health did not report the runtime active`)
    }
    if (Number(mtp.effective_depth) !== Number(requestedDepth)) {
      failures.push(`/health Native MTP effective depth ${mtp.effective_depth ?? 'missing'} did not match requested D${requestedDepth}`)
    }
  }
  return failures
}

function explicitCacheCounters(snapshot) {
  const stats = snapshot?.scheduler_stats || {}
  return {
    processed: Number(stats.num_requests_processed || 0),
    hitRequests: Number(stats.cache_hit_requests || 0),
    hitTokens: Number(stats.cache_hit_tokens || 0),
    partialTokens: Number(stats.cache_reuse_partial_tokens || 0),
    skippedTokens: Number(stats.cache_reuse_skip_tokens || 0),
  }
}

export function correlateTerminalResponseToCacheExecution({
  terminal,
  cacheSnapshot,
  turn,
  proofRequestId,
  userMessageId,
  assistantMessageId,
} = {}) {
  const schedulerStats = cacheSnapshot?.scheduler
    || cacheSnapshot?.scheduler_stats
    || {}
  const execution = schedulerStats.last_cache_execution
    || schedulerStats.batch_generator?.last_cache_execution
    || null
  const terminalResponseId = String(terminal?.responseId || '')
  const executionRequestId = String(execution?.request_id || '')
  const messageId = String(terminal?.messageId || '')
  const expectedAssistantMessageId = String(assistantMessageId || '')
  const exact = Boolean(
    terminalResponseId
    && executionRequestId
    && terminalResponseId === executionRequestId
    && messageId
    && messageId === expectedAssistantMessageId
  )
  const observation = execution && typeof execution === 'object'
    ? {
        ...execution,
        proof_request_id: String(proofRequestId || ''),
        request_id: executionRequestId,
        terminal_response_id: terminalResponseId,
        message_id: expectedAssistantMessageId,
        correlation_source:
          'chat_complete_response_id_to_scheduler_last_cache_execution',
      }
    : null
  return {
    turn: Number(turn),
    proofRequestId: String(proofRequestId || ''),
    userMessageId: String(userMessageId || ''),
    assistantMessageId: expectedAssistantMessageId,
    terminalResponseId: terminalResponseId || null,
    serverRequestId: terminalResponseId || null,
    executionRequestId: executionRequestId || null,
    correlationStatus: exact
      ? 'verified'
      : terminalResponseId && executionRequestId
        ? 'partial_request_identity_mismatch'
        : 'partial_product_support_missing',
    serverObservation: observation,
  }
}

export function isCacheRequestCorrelationVerified(result) {
  if (result?.requestedServerCacheControls !== true) return false
  const expectedTurns = expectedUiTurnCount(result)
  const turns = Array.isArray(result?.cacheRequestEvidence)
    ? result.cacheRequestEvidence.slice(0, expectedTurns)
    : []
  const uiTurns = Array.isArray(result?.uiTurnEvidence)
    ? result.uiTurnEvidence.slice(0, expectedTurns)
    : []
  if (turns.length !== expectedTurns || uiTurns.length !== expectedTurns) {
    return false
  }
  return uiTurns.every((uiTurn) => {
    const row = turns.find((item) => Number(item?.turn) === Number(uiTurn?.turn))
    const observation = row?.serverObservation || {}
    return Boolean(
      row
      && row.correlationStatus === 'verified'
      && row.serverRequestId
      && String(row.serverRequestId) === String(row.terminalResponseId || '')
      && String(row.serverRequestId) === String(row.executionRequestId || '')
      && String(row.serverRequestId) === String(observation.request_id || '')
      && String(row.serverRequestId) === String(observation.terminal_response_id || '')
      && String(row.assistantMessageId || '') === String(uiTurn.assistantMessageId || '')
      && String(row.userMessageId || '') === String(uiTurn.userMessageId || '')
      && String(observation.message_id || '') === String(uiTurn.assistantMessageId || '')
      && String(observation.proof_request_id || '') === String(uiTurn.proofRequestId || '')
      && observation.correlation_source
        === 'chat_complete_response_id_to_scheduler_last_cache_execution'
    )
  })
}

export function validateRequestCorrelatedCacheEvidence(result) {
  const failures = []
  if (result?.requestedServerCacheControls !== true) return failures
  const expectedTurns = expectedUiTurnCount(result)
  const turns = Array.isArray(result?.cacheRequestEvidence)
    ? result.cacheRequestEvidence.slice(0, expectedTurns)
    : []
  if (turns.length !== expectedTurns) {
    return [`expected exactly ${expectedTurns} request-correlated cache snapshots, got ${turns.length}`]
  }
  if (!isCacheRequestCorrelationVerified(result)) {
    failures.push(
      'UI terminal response IDs do not exactly match scheduler.last_cache_execution.request_id for every turn',
    )
  }
  let positiveReuseTurns = 0
  for (let turn = 1; turn <= expectedTurns; turn += 1) {
    const evidence = turns.find((row) => Number(row?.turn) === turn)
    const uiTurn = (result?.uiTurnEvidence || []).find(
      (row) => Number(row?.turn) === turn,
    )
    const observation = evidence?.serverObservation || {}
    if (
      !evidence?.assistantMessageId
      || String(evidence.assistantMessageId)
        !== String(result?.assistantMessageIds?.[turn - 1] || '')
      || String(evidence?.assistantMessageId || '')
        !== String(uiTurn?.assistantMessageId || '')
      || String(evidence?.userMessageId || '')
        !== String(uiTurn?.userMessageId || '')
    ) {
      failures.push(`cache snapshot for UI turn ${turn} is not bound to its user/assistant messages`)
    }
    if (
      String(observation.proof_request_id || '') !== String(uiTurn?.proofRequestId || '')
      || String(observation.request_id || '') !== String(evidence?.serverRequestId || '')
      || String(observation.terminal_response_id || '')
        !== String(evidence?.serverRequestId || '')
      || String(observation.message_id || '') !== String(evidence?.assistantMessageId || '')
    ) {
      failures.push(`cache snapshot for UI turn ${turn} lacks exact server-emitted request correlation`)
    }
    const cachedTokens = Number(
      observation.cached_tokens ?? observation.cache_hit_tokens ?? 0,
    )
    const promptTokens = Number(observation.prompt_tokens || 0)
    const prefillTokens = Number(observation.prefill_tokens || 0)
    const cacheReuseApplied = observation.cache_reuse_applied === true
      || observation.cache_outcome === 'hit'
      || cachedTokens > 0
    if (
      cacheReuseApplied
      && cachedTokens > 0
      && promptTokens > cachedTokens
      && prefillTokens > 0
    ) {
      positiveReuseTurns += 1
    }
    if (
      cachedTokens < 0
      || promptTokens < 0
      || prefillTokens < 0
      || (promptTokens > 0 && cachedTokens > promptTokens)
    ) {
      failures.push(`cache counters regressed during UI turn ${turn}`)
    }
  }
  if (uiProfileRequiresPositiveCacheReuse(result) && positiveReuseTurns < 1) {
    failures.push('no UI turn had request-correlated cache-hit and reused-token deltas')
  }
  return failures
}

export function isServerRequestCorrelationVerified(result) {
  const correlation = result?.requestCorrelation || {}
  const expectedTurns = expectedUiTurnCount(result)
  const turns = Array.isArray(correlation.turns)
    ? correlation.turns.slice(0, expectedTurns)
    : []
  const uiTurns = Array.isArray(result?.uiTurnEvidence)
    ? result.uiTurnEvidence.slice(0, expectedTurns)
    : []
  const records = Array.isArray(result?.resolvedSamplingRecords)
    ? result.resolvedSamplingRecords
    : []
  if (
    correlation.status !== 'verified'
    || turns.length !== expectedTurns
    || uiTurns.length !== expectedTurns
    || (correlation.turns || []).length !== expectedTurns
    || (result?.uiTurnEvidence || []).length !== expectedTurns
  ) {
    return false
  }
  const allRequestIds = []
  const verified = uiTurns.every((uiTurn) => {
    const row = turns.find((item) => Number(item?.turn) === Number(uiTurn?.turn))
    const uiRequestIds = [...new Set(
      (Array.isArray(uiTurn?.requestIds) ? uiTurn.requestIds : [])
        .map((value) => String(value || ''))
        .filter(Boolean),
    )]
    const rowRequestIds = [...new Set(
      (Array.isArray(row?.serverRequestIds) ? row.serverRequestIds : [])
        .map((value) => String(value || ''))
        .filter(Boolean),
    )]
    if (
      !row
      || !uiTurn?.proofRequestId
      || String(uiTurn.proofRequestId) !== String(uiTurn.userMessageId || '')
      || String(row.proofRequestId || '') !== String(uiTurn.proofRequestId)
      || String(row.userMessageId || '') !== String(uiTurn.userMessageId || '')
      || String(row.assistantMessageId || '') !== String(uiTurn.assistantMessageId || '')
      || String(row.serverProofRequestId || '') !== String(uiTurn.proofRequestId)
      || String(uiTurn.terminalProofRequestId || '') !== String(uiTurn.proofRequestId)
      || String(row.serverMessageId || '') !== String(uiTurn.assistantMessageId || '')
      || String(uiTurn.terminalMessageId || '') !== String(uiTurn.assistantMessageId || '')
      || uiTurn.logMatchMode !== 'exact_identity_ring_safe'
      || uiRequestIds.length === 0
      || rowRequestIds.length !== uiRequestIds.length
      || rowRequestIds.some((requestId) => !uiRequestIds.includes(requestId))
      || row.resolvedLogCorrelated !== true
    ) {
      return false
    }
    allRequestIds.push(...rowRequestIds)
    const matchingRecords = records.filter((record) => (
      String(record?.proof_request_id || '') === String(uiTurn.proofRequestId)
      && String(record?.message_id || '') === String(uiTurn.assistantMessageId)
    ))
    const recordRequestIds = matchingRecords.map((record) => String(record?.request_id || ''))
    return (
      matchingRecords.length === rowRequestIds.length
      && recordRequestIds.every(Boolean)
      && new Set(recordRequestIds).size === rowRequestIds.length
      && rowRequestIds.every((requestId) => recordRequestIds.includes(requestId))
      && matchingRecords.every((record) => record?.correlation_source === 'server_emitted')
    )
  })
  return (
    verified
    && allRequestIds.length === records.length
    && new Set(allRequestIds).size === allRequestIds.length
  )
}

export function validateUiRuntimeProvenance(result) {
  const failures = []
  const provenance = result?.uiRuntimeProvenance || {}
  const healthBinding = result?.healthProvenance?.after?.binding || {}
  const cdp = provenance.cdp_process_binding || {}
  const backend = provenance.backend_python_process_binding || {}
  try {
    const electron = readExternalExecutableIdentity(
      provenance.electron_executable,
      'Electron runtime executable',
    )
    if (
      electron.path !== realpathSync(cdp.executable_path || '')
      || electron.sha256 !== provenance.electron_executable_sha256
      || electron.sha256 !== cdp.executable_sha256
      || sha256Text(electron.path) !== cdp.executable_path_fingerprint_sha256
    ) {
      failures.push('Electron executable path/bytes are not independently bound to the CDP listener')
    }
  } catch (error) {
    failures.push(String(error?.message || error))
  }
  try {
    const python = readExternalExecutableIdentity(
      backend.executable_path,
      'Python backend runtime executable',
    )
    if (
      python.sha256 !== backend.executable_sha256
      || sha256Text(python.path) !== backend.executable_path_fingerprint_sha256
    ) {
      failures.push('Python backend executable path/bytes are not independently bound to its listener')
    }
  } catch (error) {
    failures.push(String(error?.message || error))
  }
  if (!validSha256(provenance.renderer_source_tree_sha256)) {
    failures.push('renderer source-tree fingerprint is missing')
  }
  if (
    provenance.renderer_source_tree_sha256 !== result?.gitProvenance?.before?.renderer_source_tree_sha256
    || provenance.renderer_source_tree_sha256 !== result?.gitProvenance?.after?.renderer_source_tree_sha256
  ) {
    failures.push('live renderer fingerprint does not match the source checkout before/after the proof')
  }
  if (provenance.source_commit !== result?.gitProvenance?.after?.commit) {
    failures.push('renderer source commit is not bound to the proof HEAD')
  }
  const buildSourceCommit = String(provenance.renderer_build_source_commit || '')
  if (
    (
      provenance.mode === 'installed-app'
      || Boolean(buildSourceCommit)
    )
    && (
      !/^[0-9a-f]{40}$/.test(buildSourceCommit)
      || buildSourceCommit !== result?.gitProvenance?.after?.commit
    )
  ) {
    failures.push('build-injected renderer source commit is not bound to the proof HEAD')
  }
  if (provenance.source_tree !== result?.gitProvenance?.after?.tree) {
    failures.push('renderer source tree is not bound to the proof tree')
  }
  if (
    !Number.isInteger(cdp.launched_root_pid)
    || cdp.launched_root_pid <= 0
    || !Array.isArray(cdp.process_tree_pids)
    || !cdp.process_tree_pids.includes(cdp.launched_root_pid)
    || !cdp.process_tree_pids.includes(cdp.listener_pid)
    || !Number.isInteger(cdp.listener_pid)
    || cdp.listener_pid <= 0
    || cdp.belongs_to_launched_process_tree !== true
    || !validSha256(cdp.executable_sha256)
    || cdp.executable_sha256 !== provenance.electron_executable_sha256
  ) {
    failures.push('CDP listener PID is not bound to the launched Electron executable bytes')
  }
  if (
    !Number.isInteger(backend.listener_pid)
    || backend.listener_pid !== healthBinding.backend_pid
    || backend.health_pid !== healthBinding.backend_pid
    || !validSha256(backend.executable_sha256)
    || !validSha256(backend.executable_path_fingerprint_sha256)
    || !validSha256(backend.invoked_executable_path_fingerprint_sha256)
    || (
      provenance.mode === 'installed-app'
        ? ![
          backend.invoked_executable_path_fingerprint_sha256,
          backend.executable_path_fingerprint_sha256,
        ].includes(
          healthBinding.runtime_source_hashes?.python_executable_fingerprint_sha256,
        )
        : backend.invoked_executable_path_fingerprint_sha256
          !== healthBinding.runtime_source_hashes?.python_executable_fingerprint_sha256
    )
    || sha256Text(path.resolve(backend.invoked_executable_path || ''))
      !== backend.invoked_executable_path_fingerprint_sha256
  ) {
    failures.push('backend TCP listener is not bound to /health PID and imported Python executable')
  }
  if (provenance.mode === 'electron-dev') {
    if (!validSha256(provenance.electron_executable_sha256)) {
      failures.push('dev Electron executable fingerprint is missing')
    }
    const served = Array.isArray(provenance.served_renderer_modules)
      ? provenance.served_renderer_modules
      : []
    const local = localRendererModuleEvidence()
    if (
      served.length !== local.length
      || canonicalJson(served) !== canonicalJson(local)
      || !validSha256(provenance.served_renderer_source_sha256)
      || provenance.served_renderer_source_sha256 !== canonicalSha256(local)
    ) {
      failures.push('dev renderer bytes served through CDP do not exactly match the source checkout')
    }
    if (provenance.vite_renderer_source_seen !== true || provenance.vite_client_seen !== true) {
      failures.push('dev renderer resources do not include the live Vite renderer')
    }
  } else if (provenance.mode === 'installed-app') {
    const manifest = provenance.external_release_manifest || {}
    if (
      !validSha256(provenance.external_release_manifest_sha256)
      || canonicalJson(Object.keys(manifest).sort())
        !== canonicalJson(installedReleaseManifestFields)
      || manifest.schema !== installedReleaseManifestSchema
      || manifest.source_commit !== result?.gitProvenance?.after?.commit
      || manifest.source_tree !== result?.gitProvenance?.after?.tree
    ) {
      failures.push('installed app lacks an independent release manifest bound to the proof HEAD')
    }
    if (
      !validSha256(provenance.app_asar_sha256)
      || provenance.app_asar_sha256 !== manifest.app_asar_sha256
    ) {
      failures.push('installed renderer ASAR is not bound to the external release manifest')
    }
    if (!validSha256(provenance.electron_executable_sha256)) {
      failures.push('installed Electron executable fingerprint is missing')
    }
    if (
      provenance.electron_executable_sha256 !== manifest.electron_executable_sha256
      || provenance.electron_executable_sha256 !== cdp.executable_sha256
    ) {
      failures.push('installed Electron executable does not match the external release manifest')
    }
    if (!validSha256(provenance.bundled_provenance_sha256)) {
      failures.push('installed app bundled provenance fingerprint is missing')
    }
    if (provenance.bundled_provenance_sha256 !== manifest.bundled_provenance_sha256) {
      failures.push('bundled-Python provenance does not match the external release manifest')
    }
    if (
      provenance.bundled_provenance?.vmlx?.commit
      !== result?.gitProvenance?.after?.commit
    ) {
      failures.push('installed app bundled provenance commit does not match proof HEAD')
    }
    const bundledSource = provenance.bundled_source || {}
    const gitSource = result?.gitProvenance?.after || {}
    const runtimeBinding = result?.healthProvenance?.after?.binding || {}
    const runtimeSource = runtimeBinding.runtime_source_hashes || {}
    for (const field of [
      'server_module_sha256',
      'package_init_sha256',
      'python_source_tree_sha256',
    ]) {
      if (
        !validSha256(bundledSource[field])
        || bundledSource[field] !== gitSource[field]
        || bundledSource[field] !== runtimeSource[field]
      ) {
        failures.push(`installed bundled/runtime/source identity does not match: ${field}`)
      }
    }
    for (const field of [
      'python_source_file_count',
      'python_source_read_error_count',
    ]) {
      if (
        bundledSource[field] !== gitSource[field]
        || bundledSource[field] !== runtimeBinding[field]
      ) {
        failures.push(`installed bundled/runtime/source count does not match: ${field}`)
      }
    }
    if (
      manifest.bundled_python_executable_fingerprint_sha256
        !== runtimeSource.python_executable_fingerprint_sha256
      || manifest.bundled_python_executable_fingerprint_sha256
        !== backend.executable_path_fingerprint_sha256
    ) {
      failures.push('installed backend did not import from the manifest-attested bundled Python')
    }
    if (
      !validSha256(manifest.bundled_python_executable_sha256)
      || manifest.bundled_python_executable_sha256 !== backend.executable_sha256
    ) {
      failures.push('installed backend Python bytes do not match the external release manifest')
    }
  } else {
    failures.push(`unknown UI launch mode ${provenance.mode || 'missing'}`)
  }
  return failures
}

const pairedApiProtocolRoutes = {
  chat: '/v1/chat/completions',
  responses: '/v1/responses',
  anthropic: '/v1/messages',
  ollama: '/api/chat',
}
const pairedCaptureLayer = 'requests.decompressed_response_parser_input'
const pairedCaptureSemantics = [
  'Exact decompressed response-body bytes delivered to protocol parsers: ',
  'streaming bytes before requests.iter_lines line splitting or Unicode ',
  'decoding, and nonstream response bytes before JSON decoding; excludes ',
  'HTTP transfer framing and compressed transport octets.',
].join('')
const pairedSafeCaptureHeaderNames = new Set([
  'content-encoding',
  'content-length',
  'content-type',
  'transfer-encoding',
])

export function expectedRawMatrixRoutes(contract, includeCancellation) {
  const streamLabels = [
    'stream-flow-round1',
    'stream-flow-round2',
    'stream-flow-round3',
    ...(includeCancellation ? ['stream-abort', 'stream-recovery'] : []),
  ]
  const routes = []
  for (const baseLabel of ['direct', 'gateway']) {
    for (const protocol of contract.protocols) {
      if (contract.modes.includes('stream')) {
        for (const captureLabel of streamLabels) {
          routes.push(`${baseLabel}\0${protocol}\0${captureLabel}`)
        }
      }
      if (contract.modes.includes('nonstream')) {
        for (const roundNumber of [1, 2, 3]) {
          routes.push(
            `${baseLabel}\0${protocol}\0nonstream-flow-round${roundNumber}`,
          )
        }
        if (includeCancellation) {
          routes.push(`${baseLabel}\0${protocol}\0nonstream-recovery`)
        }
      }
    }
  }
  return routes
}

function parseSseObjects(raw, label) {
  const objects = []
  const frames = []
  let doneSeen = false
  let postDoneFrames = 0
  for (const block of raw.toString('utf8').split(/\r?\n\r?\n/)) {
    const data = block
      .split(/\r?\n/)
      .filter((line) => line.startsWith('data:'))
      .map((line) => line.slice(5).trimStart())
      .join('\n')
      .trim()
    if (!data) continue
    if (data === '[DONE]') {
      if (doneSeen) postDoneFrames += 1
      doneSeen = true
      frames.push({ kind: 'done' })
      continue
    }
    try {
      const object = JSON.parse(data)
      if (doneSeen) postDoneFrames += 1
      objects.push(object)
      frames.push({ kind: 'data', object })
    } catch {
      throw new Error(`${label} contains malformed SSE JSON`)
    }
  }
  return { objects, frames, doneSeen, postDoneFrames }
}

function mergeToolNameFragment(existing, fragment) {
  const prior = String(existing || '')
  const next = String(fragment || '')
  if (!next) return prior
  if (!prior || next.startsWith(prior)) return next
  if (prior.endsWith(next)) return prior
  return `${prior}${next}`
}

function collectOpenAiChatStream(raw, label) {
  const { frames, doneSeen, postDoneFrames } = parseSseObjects(raw, label)
  const reasoning = []
  const content = []
  const toolCalls = new Map()
  const terminals = []
  const orderedChannels = []
  let semanticTerminalSeen = false
  let postTerminalEvents = postDoneFrames
  for (const frame of frames) {
    if (frame.kind === 'done') {
      orderedChannels.push('terminal:DONE')
      continue
    }
    const object = frame.object
    for (const choice of Array.isArray(object?.choices) ? object.choices : []) {
      const delta = choice?.delta || {}
      const reasoningDelta = delta.reasoning_content ?? delta.reasoning
      if (typeof reasoningDelta === 'string' && reasoningDelta) {
        if (semanticTerminalSeen) postTerminalEvents += 1
        reasoning.push(reasoningDelta)
        orderedChannels.push('reasoning')
      }
      if (typeof delta.content === 'string' && delta.content) {
        if (semanticTerminalSeen) postTerminalEvents += 1
        content.push(delta.content)
        orderedChannels.push('content')
      }
      for (const call of Array.isArray(delta.tool_calls) ? delta.tool_calls : []) {
        if (semanticTerminalSeen) postTerminalEvents += 1
        const key = String(call?.index ?? call?.id ?? toolCalls.size)
        const prior = toolCalls.get(key) || { id: '', name: '', arguments: '' }
        toolCalls.set(key, {
          id: String(call?.id || prior.id || ''),
          name: mergeToolNameFragment(prior.name, call?.function?.name),
          arguments: `${prior.arguments}${call?.function?.arguments || ''}`,
        })
        orderedChannels.push('tool')
      }
      if (choice?.finish_reason) {
        terminals.push(String(choice.finish_reason))
        semanticTerminalSeen = true
        orderedChannels.push(`terminal:${choice.finish_reason}`)
      }
    }
  }
  return {
    reasoning: reasoning.join(''),
    content: content.join(''),
    toolCalls: [...toolCalls.values()],
    terminal: doneSeen && terminals.length > 0,
    terminalReasons: [...terminals, ...(doneSeen ? ['DONE'] : [])],
    orderedChannels,
    postTerminalEvents,
  }
}

function collectResponsesStream(raw, label) {
  const { frames, postDoneFrames } = parseSseObjects(raw, label)
  const reasoning = []
  const content = []
  const toolCalls = new Map()
  let terminal = false
  const orderedChannels = []
  let postTerminalEvents = postDoneFrames
  for (const frame of frames) {
    if (frame.kind !== 'data') continue
    const object = frame.object
    if (terminal) postTerminalEvents += 1
    const type = String(object?.type || '')
    if (
      type === 'response.reasoning_summary_text.delta'
      || type === 'response.reasoning_text.delta'
    ) {
      reasoning.push(String(object?.delta || ''))
      orderedChannels.push('reasoning')
    } else if (type === 'response.output_text.delta') {
      content.push(String(object?.delta || ''))
      orderedChannels.push('content')
    }
    const item = object?.item || object?.output_item
    if (
      item?.type === 'function_call'
      || type === 'response.function_call_arguments.delta'
      || type === 'response.function_call_arguments.done'
    ) {
      const key = String(item?.id || item?.call_id || object?.item_id || object?.call_id || 'call')
      const prior = toolCalls.get(key) || { id: '', name: '', arguments: '' }
      const argumentValue = type === 'response.function_call_arguments.done'
        ? String(object?.arguments ?? item?.arguments ?? prior.arguments ?? '')
        : type === 'response.function_call_arguments.delta'
          ? `${prior.arguments}${object?.delta || ''}`
          : String(item?.arguments ?? prior.arguments ?? '')
      toolCalls.set(key, {
        id: String(item?.call_id || item?.id || object?.call_id || prior.id || ''),
        name: String(item?.name || object?.name || prior.name || ''),
        arguments: argumentValue,
      })
      orderedChannels.push('tool')
    }
    if (type === 'response.completed' && object?.response?.status === 'completed') {
      terminal = true
      orderedChannels.push('terminal:response.completed')
    }
  }
  return {
    reasoning: reasoning.join(''),
    content: content.join(''),
    toolCalls: [...toolCalls.values()],
    terminal,
    terminalReasons: terminal ? ['response.completed'] : [],
    orderedChannels,
    postTerminalEvents,
  }
}

function collectResponsesNonstream(raw, label) {
  const body = parseProtocolNonstreamBody(raw, label, 'Responses')
  const reasoning = []
  const content = []
  const toolCalls = []
  const orderedChannels = []
  for (const item of Array.isArray(body.output) ? body.output : []) {
    if (item?.type === 'reasoning') {
      for (const summary of Array.isArray(item.summary) ? item.summary : []) {
        const text = String(summary?.text || '')
        if (text) {
          reasoning.push(text)
          orderedChannels.push('reasoning')
        }
      }
    } else if (item?.type === 'message') {
      for (const part of Array.isArray(item.content) ? item.content : []) {
        if (['output_text', 'text'].includes(String(part?.type || ''))) {
          const text = String(part?.text || '')
          if (text) {
            content.push(text)
            orderedChannels.push('content')
          }
        }
      }
    } else if (item?.type === 'function_call') {
      toolCalls.push({
        id: String(item.call_id || item.id || ''),
        name: String(item.name || ''),
        arguments: item.arguments ?? '',
      })
      orderedChannels.push('tool')
    }
  }
  const status = typeof body.status === 'string' && body.status
    ? body.status
    : ''
  if (status) orderedChannels.push(`terminal:response.${status}`)
  return {
    reasoning: reasoning.join(''),
    content: content.join(''),
    toolCalls,
    terminal: status === 'completed',
    terminalReasons: status ? [`response.${status}`] : [],
    orderedChannels,
    postTerminalEvents: 0,
  }
}

function parseProtocolNonstreamBody(raw, label, protocolLabel) {
  let body
  try {
    body = JSON.parse(raw.toString('utf8'))
  } catch {
    throw new Error(`${label} contains malformed ${protocolLabel} nonstream JSON`)
  }
  if (!body || typeof body !== 'object' || Array.isArray(body)) {
    throw new Error(`${label} ${protocolLabel} nonstream JSON is not an object`)
  }
  return body
}

function collectOpenAiChatNonstream(raw, label) {
  const body = parseProtocolNonstreamBody(raw, label, 'Chat')
  const choice = Array.isArray(body.choices) ? body.choices[0] : null
  const message = choice?.message || {}
  const reasoning = String(
    message.reasoning_content ?? message.reasoning ?? '',
  )
  const content = String(message.content ?? '')
  const toolCalls = (Array.isArray(message.tool_calls) ? message.tool_calls : [])
    .map((call) => ({
      id: String(call?.id || ''),
      name: String(call?.function?.name || ''),
      arguments: call?.function?.arguments ?? '',
    }))
  const terminalReason = String(choice?.finish_reason || '')
  return {
    reasoning,
    content,
    toolCalls,
    terminal: Boolean(terminalReason),
    terminalReasons: terminalReason ? [terminalReason] : [],
    orderedChannels: [
      ...(reasoning ? ['reasoning'] : []),
      ...(content ? ['content'] : []),
      ...toolCalls.map(() => 'tool'),
      ...(terminalReason ? [`terminal:${terminalReason}`] : []),
    ],
    postTerminalEvents: 0,
  }
}

function collectAnthropicNonstream(raw, label) {
  const body = parseProtocolNonstreamBody(raw, label, 'Anthropic')
  const reasoning = []
  const content = []
  const toolCalls = []
  const orderedChannels = []
  for (const block of Array.isArray(body.content) ? body.content : []) {
    if (['thinking', 'reasoning'].includes(String(block?.type || ''))) {
      const text = String(
        block?.thinking ?? block?.reasoning ?? block?.text ?? '',
      )
      if (text) {
        reasoning.push(text)
        orderedChannels.push('reasoning')
      }
    } else if (block?.type === 'text') {
      const text = String(block?.text || '')
      if (text) {
        content.push(text)
        orderedChannels.push('content')
      }
    } else if (block?.type === 'tool_use') {
      toolCalls.push({
        id: String(block?.id || ''),
        name: String(block?.name || ''),
        arguments: block?.input ?? {},
      })
      orderedChannels.push('tool')
    }
  }
  const terminalReason = String(body.stop_reason || '')
  if (terminalReason) orderedChannels.push(`terminal:${terminalReason}`)
  return {
    reasoning: reasoning.join(''),
    content: content.join(''),
    toolCalls,
    terminal: Boolean(terminalReason),
    terminalReasons: terminalReason ? [terminalReason] : [],
    orderedChannels,
    postTerminalEvents: 0,
  }
}

function collectOllamaNonstream(raw, label) {
  const body = parseProtocolNonstreamBody(raw, label, 'Ollama')
  const message = body.message || {}
  const reasoning = String(message.thinking ?? message.reasoning ?? '')
  const content = String(message.content ?? '')
  const toolCalls = (Array.isArray(message.tool_calls) ? message.tool_calls : [])
    .map((call, index) => ({
      id: String(call?.id || `ollama_call_${index}`),
      name: String(call?.function?.name || ''),
      arguments: call?.function?.arguments ?? {},
    }))
  const terminalReason = body.done === true
    ? String(body.done_reason || 'stop')
    : ''
  return {
    reasoning,
    content,
    toolCalls,
    terminal: Boolean(terminalReason),
    terminalReasons: terminalReason ? [terminalReason] : [],
    orderedChannels: [
      ...(reasoning ? ['reasoning'] : []),
      ...(content ? ['content'] : []),
      ...toolCalls.map(() => 'tool'),
      ...(terminalReason ? [`terminal:${terminalReason}`] : []),
    ],
    postTerminalEvents: 0,
  }
}

function collectAnthropicStream(raw, label) {
  const { frames, postDoneFrames } = parseSseObjects(raw, label)
  const reasoning = []
  const content = []
  const toolCalls = new Map()
  const terminals = []
  let terminal = false
  let semanticTerminalSeen = false
  const orderedChannels = []
  let postTerminalEvents = postDoneFrames
  for (const frame of frames) {
    if (frame.kind !== 'data') continue
    const object = frame.object
    const type = String(object?.type || '')
    if (
      terminal
      || (semanticTerminalSeen && type !== 'message_stop')
    ) {
      postTerminalEvents += 1
    }
    const block = object?.content_block
    const delta = object?.delta || {}
    if (type === 'content_block_start' && block?.type === 'tool_use') {
      const key = String(object?.index ?? toolCalls.size)
      toolCalls.set(key, {
        id: String(block.id || ''),
        name: String(block.name || ''),
        arguments: canonicalJson(block.input || {}),
      })
      orderedChannels.push('tool')
    }
    if (type === 'content_block_start' && block?.type === 'thinking') {
      reasoning.push(String(block.thinking || ''))
      orderedChannels.push('reasoning')
    }
    if (type === 'content_block_start' && block?.type === 'text') {
      content.push(String(block.text || ''))
      orderedChannels.push('content')
    }
    if (delta?.type === 'thinking_delta') {
      reasoning.push(String(delta.thinking || ''))
      orderedChannels.push('reasoning')
    }
    if (delta?.type === 'text_delta') {
      content.push(String(delta.text || ''))
      orderedChannels.push('content')
    }
    if (delta?.type === 'input_json_delta') {
      const key = String(object?.index ?? 0)
      const prior = toolCalls.get(key) || { id: '', name: '', arguments: '' }
      toolCalls.set(key, {
        ...prior,
        arguments: `${prior.arguments === '{}' ? '' : prior.arguments}${delta.partial_json || ''}`,
      })
      orderedChannels.push('tool')
    }
    if (type === 'message_delta' && delta?.stop_reason) {
      terminals.push(String(delta.stop_reason))
      semanticTerminalSeen = true
      orderedChannels.push(`terminal:${delta.stop_reason}`)
    }
    if (type === 'message_stop') {
      terminal = true
      orderedChannels.push('terminal:message_stop')
    }
  }
  return {
    reasoning: reasoning.join(''),
    content: content.join(''),
    toolCalls: [...toolCalls.values()],
    terminal: terminal && terminals.length > 0,
    terminalReasons: [...terminals, ...(terminal ? ['message_stop'] : [])],
    orderedChannels,
    postTerminalEvents,
  }
}

export function collectOllamaStream(raw, label) {
  const reasoning = []
  const content = []
  const toolCalls = []
  const terminals = []
  const orderedChannels = []
  let terminalSeen = false
  let postTerminalEvents = 0
  for (const line of raw.toString('utf8').split(/\r?\n/).filter(Boolean)) {
    let object
    try {
      object = JSON.parse(line)
    } catch {
      throw new Error(`${label} contains malformed Ollama NDJSON`)
    }
    if (terminalSeen) postTerminalEvents += 1
    const message = object?.message || {}
    const reasoningDelta = message.thinking ?? message.reasoning
    if (typeof reasoningDelta === 'string' && reasoningDelta) {
      reasoning.push(reasoningDelta)
      orderedChannels.push('reasoning')
    }
    if (typeof message.content === 'string' && message.content) {
      content.push(message.content)
      orderedChannels.push('content')
    }
    for (const [index, call] of (Array.isArray(message.tool_calls) ? message.tool_calls : []).entries()) {
      toolCalls.push({
        id: String(call?.id || `ollama_call_${index}`),
        name: String(call?.function?.name || ''),
        arguments: canonicalJson(call?.function?.arguments || {}),
      })
      orderedChannels.push('tool')
    }
    if (object?.done === true) {
      terminals.push(String(object?.done_reason || 'stop'))
      terminalSeen = true
      orderedChannels.push(`terminal:${object?.done_reason || 'stop'}`)
    }
  }
  return {
    reasoning: reasoning.join(''),
    content: content.join(''),
    toolCalls,
    terminal: terminals.length > 0,
    terminalReasons: terminals,
    orderedChannels,
    postTerminalEvents,
  }
}

function collectProtocolStream(protocol, raw, label) {
  if (protocol === 'chat') return collectOpenAiChatStream(raw, label)
  if (protocol === 'responses') return collectResponsesStream(raw, label)
  if (protocol === 'anthropic') return collectAnthropicStream(raw, label)
  if (protocol === 'ollama') return collectOllamaStream(raw, label)
  throw new Error(`${label} has unsupported protocol ${protocol}`)
}

function collectProtocolNonstream(protocol, raw, label) {
  if (protocol === 'chat') return collectOpenAiChatNonstream(raw, label)
  if (protocol === 'responses') return collectResponsesNonstream(raw, label)
  if (protocol === 'anthropic') return collectAnthropicNonstream(raw, label)
  if (protocol === 'ollama') return collectOllamaNonstream(raw, label)
  throw new Error(`${label} has unsupported protocol ${protocol}`)
}

const expectedPairedApiProtocols = Object.keys(pairedApiProtocolRoutes)
const expectedPairedApiModes = ['stream', 'nonstream']
const fullPairedApiProfiles = new Set([
  'full-agentic',
  'full-agentic-plus-cache-store',
  'full-agentic-native-cache',
])
const scopedPairedApiProfiles = new Set([
  'cache-probe',
  'cache-evict-refault',
  'cache-restart-probe',
  'cache-tq-off-store-probe',
])
const pairedToolParameters = {
  file_info: {
    type: 'object',
    properties: { path: { type: 'string' } },
    required: ['path'],
    additionalProperties: false,
  },
  run_command: {
    type: 'object',
    properties: { command: { type: 'string' } },
    required: ['command'],
    additionalProperties: false,
  },
}

function exactStringSet(value, expected) {
  return Array.isArray(value)
    && value.length === expected.length
    && canonicalJson([...value].sort()) === canonicalJson([...expected].sort())
}

export function expectedPairedApiContract(result) {
  const topLevelProfile = String(result?.apiActionProfile || '')
  const requestProfile = String(
    result?.requestContract?.apiActionProfile || '',
  )
  const profilesAgree = (
    (!topLevelProfile && !requestProfile)
    || (
      Boolean(topLevelProfile)
      && Boolean(requestProfile)
      && topLevelProfile === requestProfile
    )
  )
  const profile = requestProfile || topLevelProfile
  if (!profilesAgree) {
    return {
      valid: false,
      profile,
      protocols: [],
      modes: [],
      requireFrozenChatParity: false,
    }
  }
  // Legacy/non-orchestrated proof callers predate named action profiles and
  // own the original full matrix contract. Named full profiles retain that
  // exact contract. Cache-only phases are deliberately scoped to Chat stream;
  // phases 0 and 5 independently own the full matrix, so inflating phases 1-4
  // would only repeat already-attested protocol work.
  if (!profile || fullPairedApiProfiles.has(profile)) {
    return {
      valid: true,
      profile,
      protocols: expectedPairedApiProtocols,
      modes: expectedPairedApiModes,
      requireFrozenChatParity: true,
    }
  }
  if (scopedPairedApiProfiles.has(profile)) {
    return {
      valid: true,
      profile,
      protocols: ['chat'],
      modes: ['stream'],
      requireFrozenChatParity: false,
    }
  }
  return {
    valid: false,
    profile,
    protocols: [],
    modes: [],
    requireFrozenChatParity: false,
  }
}

function expectedFlowTerminals(protocol, mode, roundIndex) {
  const toolRound = roundIndex < 2
  if (protocol === 'chat') {
    return [
      toolRound ? 'tool_calls' : 'stop',
      ...(mode === 'stream' ? ['DONE'] : []),
    ]
  }
  if (protocol === 'responses') return ['response.completed']
  if (protocol === 'anthropic') {
    return [
      toolRound ? 'tool_use' : 'end_turn',
      ...(mode === 'stream' ? ['message_stop'] : []),
    ]
  }
  return [toolRound ? 'tool_calls' : 'stop']
}

function expectedContractNames(protocol, stage) {
  if (protocol === 'ollama') {
    if (stage === 1) return ['file_info']
    if (stage === 2) return ['run_command']
    return []
  }
  return ['file_info', 'run_command']
}

function expectedToolChoice(protocol, mode, stage) {
  if (protocol === 'ollama') return null
  if (stage === 3) {
    return protocol === 'anthropic' ? { type: 'none' } : 'none'
  }
  const name = stage === 1 ? 'file_info' : 'run_command'
  if (stage === 1 && mode !== 'stream') {
    return protocol === 'anthropic' ? { type: 'any' } : 'required'
  }
  if (protocol === 'chat') {
    return { type: 'function', function: { name } }
  }
  if (protocol === 'responses') {
    return { type: 'function', name }
  }
  return { type: 'tool', name }
}

function normalizedToolArguments(value) {
  if (value && typeof value === 'object' && !Array.isArray(value)) return value
  if (typeof value !== 'string' || !value.trim()) return {}
  try {
    const parsed = JSON.parse(value)
    return parsed && typeof parsed === 'object' && !Array.isArray(parsed) ? parsed : {}
  } catch {
    return null
  }
}

function validatePublicToolContracts(request, protocol, stage, label) {
  const failures = []
  const contracts = Array.isArray(request?.tool_contracts) ? request.tool_contracts : []
  const expectedNames = expectedContractNames(protocol, stage)
  if (canonicalJson(contracts.map((item) => item?.name)) !== canonicalJson(expectedNames)) {
    failures.push(`${label} tool contract names/order are not exact`)
    return failures
  }
  for (const contract of contracts) {
    if (
      !pairedToolParameters[contract?.name]
      || canonicalJson(contract?.parameters) !== canonicalJson(pairedToolParameters[contract.name])
    ) {
      failures.push(`${label} ${contract?.name || 'unknown'} tool schema is not exact`)
    }
  }
  return failures
}

function toolLinkPresent(
  request,
  execution,
  { requireName = false, allowMissingCallId = false } = {},
) {
  const links = Array.isArray(request?.tool_history_linkage)
    ? request.tool_history_linkage
    : []
  return links.some((link) => (
    link?.kind === 'tool_result'
    && (
      String(link?.call_id || '') === String(execution?.call_id || '')
      || (allowMissingCallId && !String(link?.call_id || ''))
    )
    && (!requireName || String(link?.name || '') === String(execution?.name || ''))
    && Number(link?.output_chars) === Number(execution?.output_chars)
    && String(link?.output_sha256 || '') === String(execution?.output_sha256 || '')
  ))
}

function assistantToolLinkPresent(
  request,
  call,
  { allowMissingCallId = false } = {},
) {
  const links = Array.isArray(request?.tool_history_linkage)
    ? request.tool_history_linkage
    : []
  return links.some((link) => (
    link?.kind === 'assistant_tool_call'
    && (
      String(link?.call_id || '') === String(call?.id || '')
      || (allowMissingCallId && !String(link?.call_id || ''))
    )
    && String(link?.name || '') === String(call?.name || '')
  ))
}

export function validateRawChannelOrder(
  parsed,
  roundIndex,
  label,
  reasoningRequested = false,
) {
  const failures = []
  const channels = Array.isArray(parsed?.orderedChannels)
    ? parsed.orderedChannels
    : []
  const firstTerminal = channels.findIndex((channel) => (
    String(channel).startsWith('terminal:')
  ))
  const nonterminal = firstTerminal >= 0
    ? channels.slice(0, firstTerminal)
    : channels
  if (
    firstTerminal < 0
    || channels.slice(firstTerminal).some((channel) => (
      !String(channel).startsWith('terminal:')
    ))
  ) {
    failures.push(`${label} raw terminal framing is not a terminal-only suffix`)
  }
  if (roundIndex < 2) {
    const reasoningIndexes = []
    const toolIndexes = []
    for (let index = 0; index < nonterminal.length; index += 1) {
      if (nonterminal[index] === 'reasoning') reasoningIndexes.push(index)
      if (nonterminal[index] === 'tool') toolIndexes.push(index)
    }
    const reasoningOrderInvalid = reasoningIndexes.length === 0
      ? reasoningRequested
      : (
          reasoningIndexes.length < 2
          || Math.max(...reasoningIndexes) > Math.min(...toolIndexes)
        )
    if (
      reasoningOrderInvalid
      || toolIndexes.length < 1
      || nonterminal.some((channel) => channel === 'content')
      || nonterminal.some((channel) => (
        !['reasoning', 'tool'].includes(String(channel))
      ))
    ) {
      failures.push(
        `${label} raw stream does not preserve progressive reasoning before tool output`,
      )
    }
  } else if (
    nonterminal.length < 2
    || nonterminal.some((channel) => channel !== 'content')
  ) {
    failures.push(
      `${label} raw final answer is not a progressive content-only stream`,
    )
  }
  return failures
}

function validateFlowRoundEvents(round, protocol, mode, roundIndex, label) {
  const failures = []
  const events = Array.isArray(round?.events) ? round.events : []
  const expectedTerminals = expectedFlowTerminals(protocol, mode, roundIndex)
  const terminalEvents = events.filter((event) => event?.channel === 'terminal')
  const terminalValues = terminalEvents.map((event) => String(event?.kind || ''))
  if (
    canonicalJson(round?.terminals || []) !== canonicalJson(expectedTerminals)
    || canonicalJson(terminalValues) !== canonicalJson(expectedTerminals)
  ) {
    failures.push(`${label} terminal status/count/order is not protocol-native exact`)
  }
  const firstTerminal = events.findIndex((event) => event?.channel === 'terminal')
  if (
    firstTerminal < 0
    || events.slice(firstTerminal).some((event) => event?.channel !== 'terminal')
    || events.length - firstTerminal !== expectedTerminals.length
  ) {
    failures.push(`${label} has non-terminal output after terminalization`)
  }
  const times = events.map((event) => Number(event?.at_ms))
  if (
    times.some((value) => !Number.isFinite(value))
    || times.some((value, index) => index > 0 && value < times[index - 1])
  ) {
    failures.push(`${label} event timestamps are not monotonic`)
  }
  return failures
}

export function validateMatrixFlow(flow, protocol, mode, label, repoRoot) {
  const failures = []
  const requests = Array.isArray(flow?.requests) ? flow.requests : []
  const rounds = Array.isArray(flow?.rounds) ? flow.rounds : []
  const executions = Array.isArray(flow?.executions) ? flow.executions : []
  if (flow?.pass !== true || Object.values(flow?.checks || {}).some((value) => value !== true)) {
    failures.push(`${label} flow/checks do not attest pass`)
  }
  if (requests.length !== 3 || rounds.length !== 3 || executions.length !== 2) {
    failures.push(`${label} must contain exactly three requests/rounds and two executions`)
    return failures
  }
  const classifications = Array.isArray(flow?.terminal_classification)
    ? flow.terminal_classification
    : []
  if (
    classifications.length !== 3
    || classifications.some((item, index) => (
      item?.pass !== true
      || canonicalJson(item?.values) !== canonicalJson(
        expectedFlowTerminals(protocol, mode, index),
      )
    ))
  ) {
    failures.push(`${label} terminal classifications are not exact`)
  }
  const expectedTools = [
    { name: 'file_info', arguments: { path: 'panel/package.json' } },
    { name: 'run_command', arguments: { command: 'pwd' } },
  ]
  for (let index = 0; index < 3; index += 1) {
    const stage = index + 1
    const request = requests[index]
    const round = rounds[index]
    const roundLabel = `${label} round${stage}`
    if (Number(request?.stage) !== stage || request?.stream !== (mode === 'stream')) {
      failures.push(`${roundLabel} request stage/stream mode is not exact`)
    }
    if (
      canonicalJson(request?.tool_choice ?? null)
      !== canonicalJson(expectedToolChoice(protocol, mode, stage))
    ) {
      failures.push(`${roundLabel} protocol-native tool_choice is not exact`)
    }
    const expectedPreviousResponseId = (
      protocol === 'responses' && index > 0
        ? String(rounds[index - 1]?.response_id || '')
        : null
    )
    if (
      canonicalJson(request?.previous_response_id ?? null)
      !== canonicalJson(expectedPreviousResponseId)
    ) {
      failures.push(`${roundLabel} Responses continuation ID is not exact`)
    }
    failures.push(...validatePublicToolContracts(request, protocol, stage, roundLabel))
    if (Number(round?.status_code) !== 200 || (round?.errors || []).length) {
      failures.push(`${roundLabel} did not complete with HTTP 200 and zero protocol errors`)
    }
    failures.push(...validateFlowRoundEvents(round, protocol, mode, index, roundLabel))
    if (containsTransientProtocolMarker(round?.content || '')) {
      failures.push(`${roundLabel} leaked inline reasoning/tool markers into visible content`)
    }
    const calls = Array.isArray(round?.tool_calls) ? round.tool_calls : []
    if (index < 2) {
      const expected = expectedTools[index]
      if (
        calls.length !== 1
        || !String(calls[0]?.id || '')
        || calls[0]?.name !== expected.name
        || canonicalJson(calls[0]?.arguments) !== canonicalJson(expected.arguments)
        || calls[0]?.arguments_parse_error
      ) {
        failures.push(`${roundLabel} exact tool name/arguments/call ID are invalid`)
      }
      if (String(round?.content || '').trim()) {
        failures.push(`${roundLabel} tool round exposed visible answer prose`)
      }
      const reasoningChars = Number(round?.reasoning_chars)
      if (
        reasoningChars > 0
        && !String(round?.reasoning_sha256 || '')
      ) {
        failures.push(`${roundLabel} emitted reasoning without a separate payload hash`)
      }
      if (
        reasoningChars > 0
        && mode === 'stream'
        && Number(round?.reasoning_delta_count) < 2
      ) {
        failures.push(`${roundLabel} emitted reasoning that was not progressively streamed`)
      }
    } else {
      if (calls.length) failures.push(`${roundLabel} final synthesis emitted a tool call`)
      if (String(round?.content || '').trim() !== String(flow?.expected_final || '').trim()) {
        failures.push(`${roundLabel} visible final answer is not exact`)
      }
      if (mode === 'stream' && Number(round?.content_delta_count) < 2) {
        failures.push(`${roundLabel} visible answer was not progressively streamed`)
      }
    }
  }
  for (let index = 0; index < 2; index += 1) {
    const expected = expectedTools[index]
    const execution = executions[index]
    const call = rounds[index]?.tool_calls?.[0]
    if (
      execution?.name !== expected.name
      || String(execution?.call_id || '') !== String(call?.id || '')
      || canonicalJson(execution?.arguments) !== canonicalJson(expected.arguments)
      || !validSha256(execution?.output_sha256)
      || Number(execution?.output_chars) <= 0
    ) {
      failures.push(`${label} execution ${index + 1} is not exactly linked to its tool call`)
    }
    const continuation = requests[index + 1]
    if (protocol !== 'responses' && !assistantToolLinkPresent(
      continuation,
      call,
      { allowMissingCallId: protocol === 'ollama' },
    )) {
      failures.push(
        `${label} request${index + 2} is not linked to the prior assistant tool call`,
      )
    }
    if (!toolLinkPresent(continuation, execution, {
      requireName: protocol === 'ollama',
      allowMissingCallId: protocol === 'ollama',
    })) {
      failures.push(
        `${label} request${index + 2} is not linked to the real ${execution?.name || 'tool'} result`,
      )
    }
  }
  if (
    executions[0]?.result?.path !== 'panel/package.json'
    || !String(executions[0]?.result?.size_human || '')
    || String(executions[1]?.result?.stdout || '') !== String(repoRoot || '')
  ) {
    failures.push(`${label} real file_info/pwd results are not exact`)
  }
  const expectedHistoryKinds = protocol === 'responses'
    ? [[], ['tool_result'], ['tool_result']]
    : [
        [],
        ['assistant_tool_call', 'tool_result'],
        [
          'assistant_tool_call',
          'tool_result',
          'assistant_tool_call',
          'tool_result',
        ],
      ]
  for (let index = 0; index < requests.length; index += 1) {
    const actualKinds = (requests[index]?.tool_history_linkage || [])
      .map((link) => link?.kind)
    if (canonicalJson(actualKinds) !== canonicalJson(expectedHistoryKinds[index])) {
      failures.push(`${label} request${index + 1} tool history order is not exact`)
    }
  }
  if (
    protocol !== 'responses'
    && (
      !assistantToolLinkPresent(requests[2], rounds[0]?.tool_calls?.[0], {
        allowMissingCallId: protocol === 'ollama',
      })
      || !toolLinkPresent(requests[2], executions[0], {
        requireName: protocol === 'ollama',
        allowMissingCallId: protocol === 'ollama',
      })
    )
  ) {
    failures.push(`${label} request3 does not retain the first tool turn`)
  }
  const reasoningRequested = requests.slice(0, 2).some(
    (request) => request?.enable_thinking === true,
  )
  const reasoningObserved = rounds.slice(0, 2).some(
    (round) => Number(round?.reasoning_chars) > 0,
  )
  // The producer's contract permits a thinking-enabled model to call one of
  // the tools without a private chain, but requires at least one real
  // reasoning rail across the tool loop. Thinking-off bodies must not be
  // failed for truthfully omitting a reasoning channel.
  if (reasoningRequested && !reasoningObserved) {
    failures.push(`${label} requested reasoning but no tool round emitted it`)
  }
  return failures
}

function healthIdentityMatchesUi(identity, uiBinding) {
  return [
    'backend_pid',
    'python_source_file_count',
    'python_source_read_error_count',
    'model_name',
    'loaded_model_name',
    'model_bundle_fingerprint_sha256',
    'model_bundle_files',
    'cache_topology_fingerprint_sha256',
    'fingerprint_sha256',
  ].every((field) => (
    canonicalJson(identity?.[field] ?? null)
      === canonicalJson(uiBinding?.[field] ?? null)
  ))
    && canonicalJson(identity?.runtime_source_hashes)
      === canonicalJson(uiBinding?.runtime_source_hashes)
}

function validateMatrixIdentity(value, result) {
  const failures = []
  const identity = value?.identity || {}
  if (
    !Array.isArray(identity?.failures)
    || identity.failures.length !== 0
  ) {
    failures.push('paired matrix identity failure list is not exactly empty')
  }
  const sourceBefore = identity?.source?.before || {}
  const sourceAfter = identity?.source?.after || {}
  const uiSourceBefore = result?.gitProvenance?.before || {}
  const uiSourceAfter = result?.gitProvenance?.after || {}
  for (const field of [
    'head',
    'tree',
    'server_module_sha256',
    'package_init_sha256',
    'python_source_tree_sha256',
    'python_source_file_count',
    'python_source_read_error_count',
  ]) {
    if (
      canonicalJson(sourceBefore[field]) !== canonicalJson(uiSourceBefore[field === 'head' ? 'commit' : field])
      || canonicalJson(sourceAfter[field]) !== canonicalJson(uiSourceAfter[field === 'head' ? 'commit' : field])
    ) {
      failures.push(`paired matrix source identity mismatch: ${field}`)
    }
  }
  if (
    sourceBefore.clean !== true
    || sourceAfter.clean !== true
    || canonicalJson(sourceBefore) !== canonicalJson(sourceAfter)
    || identity?.source?.declared_head !== sourceAfter.head
  ) {
    failures.push('paired matrix source before/after/declared HEAD is not exact and clean')
  }
  const runnerBefore = identity?.runner?.before || {}
  const runnerAfter = identity?.runner?.after || {}
  const producerPath = 'tests/cross_matrix/run_agentic_protocol_matrix.py'
  const expectedHarnessPath = realpathSync(path.join(repoDir, producerPath))
  const uiRuntime = result?.uiRuntimeProvenance || {}
  const executionMode = String(runnerBefore.execution_mode || '')
  let producerExecutableSha256 = ''
  if (
    canonicalJson(runnerBefore) !== canonicalJson(runnerAfter)
    || runnerBefore.producer_harness_relative_path !== producerPath
    || path.resolve(runnerBefore.producer_harness_path || '') !== expectedHarnessPath
  ) {
    failures.push('paired matrix producer environment/harness identity is invalid')
  } else {
    try {
      if (
        !path.isAbsolute(String(runnerBefore.python_executable_path || ''))
        || sha256Text(runnerBefore.python_executable_path)
          !== runnerBefore.python_executable_fingerprint_sha256
        || !path.isAbsolute(String(runnerBefore.python_prefix_path || ''))
        || sha256Text(runnerBefore.python_prefix_path)
          !== runnerBefore.python_prefix_fingerprint_sha256
        || realpathSync(runnerBefore.python_executable_path)
          !== realpathSync(runnerBefore.producer_executable_path)
      ) {
        failures.push('paired matrix producer Python path/fingerprint binding is invalid')
      }
      const harness = readExternalFileBytes(
        runnerBefore.producer_harness_path,
        'Paired API producer harness',
      )
      const executable = readExternalExecutableIdentity(
        runnerBefore.producer_executable_path,
        'Paired API producer executable',
      )
      producerExecutableSha256 = executable.sha256
      if (
        harness.sha256 !== runnerBefore.producer_harness_sha256
        || harness.bytes !== Number(runnerBefore.producer_harness_size_bytes)
        || executable.sha256 !== runnerBefore.producer_executable_sha256
        || executable.bytes !== Number(runnerBefore.producer_executable_size_bytes)
      ) {
        failures.push('paired matrix producer executable/harness bytes do not match reopened files')
      }
    } catch (error) {
      failures.push(String(error?.message || error))
    }
  }
  const checkoutFingerprints = runnerBefore
    .checkout_python_invocation_fingerprints_sha256
  const installedFingerprints = runnerBefore
    .installed_python_invocation_fingerprints_sha256
  const acceptedFingerprints = runnerBefore
    .accepted_python_invocation_fingerprints_sha256
  const validFingerprintList = (value, allowEmpty = false) => (
    Array.isArray(value)
    && (allowEmpty || value.length > 0)
    && value.every((fingerprint) => validSha256(fingerprint))
  )
  if (uiRuntime.mode === 'electron-dev') {
    if (
      executionMode !== 'source-checkout-venv'
      || runnerBefore.repo_venv !== true
      || runnerBefore.repo_python !== true
      || runnerBefore.installed_runtime !== null
      || !validFingerprintList(checkoutFingerprints)
      || canonicalJson(acceptedFingerprints) !== canonicalJson(checkoutFingerprints)
      || !Array.isArray(installedFingerprints)
      || installedFingerprints.length !== 0
      || !checkoutFingerprints.includes(
        runnerBefore.python_executable_fingerprint_sha256,
      )
    ) {
      failures.push('paired matrix source-checkout runner identity is invalid')
    }
  } else if (uiRuntime.mode === 'installed-app') {
    const installed = runnerBefore.installed_runtime || {}
    const manifest = installed.manifest || {}
    const appPath = path.resolve(String(installed.app_path || ''))
    let canonicalAppPath = appPath
    try {
      canonicalAppPath = realpathSync(appPath)
    } catch {}
    const expectedAppPath = path.resolve(String(result?.installedAppPath || ''))
    const invokedPythonPath = path.resolve(
      String(installed.invoked_python_path || ''),
    )
    const expectedPythonInvocationPath = path.join(
      appPath,
      installedBundledPythonRelativePath,
    )
    let expectedPythonPath = expectedPythonInvocationPath
    try {
      expectedPythonPath = realpathSync(expectedPythonInvocationPath)
    } catch {}
    let expectedPythonPrefix = path.dirname(path.dirname(expectedPythonPath))
    try {
      expectedPythonPrefix = realpathSync(expectedPythonPrefix)
    } catch {}
    const sourceBinding = installed.source_binding || {}
    const expectedSourceBinding = {
      head: uiSourceAfter.commit,
      tree: uiSourceAfter.tree,
      server_module_sha256: uiSourceAfter.server_module_sha256,
      package_init_sha256: uiSourceAfter.package_init_sha256,
      python_source_tree_sha256: uiSourceAfter.python_source_tree_sha256,
      python_source_file_count: uiSourceAfter.python_source_file_count,
      python_source_read_error_count: uiSourceAfter.python_source_read_error_count,
    }
    if (
      executionMode !== 'installed-runtime'
      || runnerBefore.repo_venv !== false
      || runnerBefore.repo_python !== false
      || !validFingerprintList(checkoutFingerprints, true)
      || !validFingerprintList(installedFingerprints)
      || installedFingerprints.length !== 1
      || canonicalJson(acceptedFingerprints) !== canonicalJson(installedFingerprints)
      || installedFingerprints[0]
        !== runnerBefore.python_executable_fingerprint_sha256
      || installed.schema !== 'vmlx-agentic-installed-runtime-v1'
    ) {
      failures.push('paired matrix installed-runtime runner identity is invalid')
    }
    if (
      !expectedAppPath
      || appPath !== expectedAppPath
      || path.basename(appPath) !== 'vMLX.app'
      || invokedPythonPath !== expectedPythonPath
      || runnerBefore.python_executable_path !== expectedPythonPath
      || installed.python_prefix_path !== expectedPythonPrefix
      || runnerBefore.python_prefix_path !== expectedPythonPrefix
    ) {
      failures.push('paired matrix installed app/Python paths do not match the UI install')
    }
    if (
      canonicalJson(Object.keys(manifest).sort())
        !== canonicalJson(installedReleaseManifestFields)
      || manifest.schema !== installedReleaseManifestSchema
      || manifest.source_commit !== uiSourceAfter.commit
      || manifest.source_tree !== uiSourceAfter.tree
      || installed.manifest_opened_nofollow !== true
      || !validSha256(installed.manifest_sha256)
      || !Number.isInteger(Number(installed.manifest_size_bytes))
      || Number(installed.manifest_size_bytes) <= 0
      || Number(installed.manifest_size_bytes) > 1024 * 1024
      || installed.manifest_nlink !== 1
    ) {
      failures.push('paired matrix installed release manifest attestation is invalid')
    }
    try {
      const reopenedManifest = readPrivateExternalJson(
        installed.manifest_path,
        'Paired installed release manifest',
        1024 * 1024,
      )
      if (
        reopenedManifest.path !== uiRuntime.external_release_manifest_path
        || reopenedManifest.path !== installed.manifest_path
        || reopenedManifest.sha256 !== uiRuntime.external_release_manifest_sha256
        || reopenedManifest.sha256 !== installed.manifest_sha256
        || reopenedManifest.bytes !== Number(installed.manifest_size_bytes)
        || reopenedManifest.mode !== 0o600
        || reopenedManifest.nlink !== 1
        || reopenedManifest.opened_nofollow !== true
        || canonicalJson(reopenedManifest.value) !== canonicalJson(manifest)
        || canonicalJson(reopenedManifest.value)
          !== canonicalJson(uiRuntime.external_release_manifest)
      ) {
        failures.push('paired UI/API installed release manifests are not the same private file')
      }
    } catch (error) {
      failures.push(String(error?.message || error))
    }
    const bundledPython = installed.bundled_python || {}
    try {
      const reopenedPython = readExternalExecutableIdentity(
        expectedPythonInvocationPath,
        'Paired installed bundled Python',
      )
      if (
        bundledPython.path !== reopenedPython.path
        || !isPathInside(reopenedPython.path, canonicalAppPath)
      ) {
        failures.push('paired matrix bundled Python canonical path escapes or differs from the UI app')
      }
      if (
        bundledPython.sha256 !== reopenedPython.sha256
        || Number(bundledPython.size_bytes) !== reopenedPython.bytes
        || manifest.bundled_python_executable_sha256 !== reopenedPython.sha256
        || manifest.bundled_python_executable_sha256
          !== runnerBefore.producer_executable_sha256
      ) {
        failures.push('paired matrix bundled Python bytes are not manifest/producer bound')
      }
      if (
        manifest.bundled_python_executable_fingerprint_sha256
          !== runnerBefore.python_executable_fingerprint_sha256
        || manifest.bundled_python_executable_fingerprint_sha256
          !== uiRuntime.backend_python_process_binding
            ?.executable_path_fingerprint_sha256
      ) {
        failures.push('paired matrix bundled Python canonical path fingerprint is not manifest/UI bound')
      }
      if (
        manifest.bundled_python_executable_sha256
          !== uiRuntime.backend_python_process_binding?.executable_sha256
      ) {
        failures.push('paired matrix bundled Python bytes do not match the UI backend')
      }
    } catch (error) {
      failures.push(String(error?.message || error))
    }
    const installedArtifacts = installed.artifacts || {}
    if (
      canonicalJson(Object.keys(installedArtifacts).sort())
        !== canonicalJson(Object.keys(installedAppArtifactPaths).sort())
    ) {
      failures.push('paired matrix installed artifact set is not exact')
    }
    for (const [label, contract] of Object.entries(installedAppArtifactPaths)) {
      const record = installedArtifacts[label] || {}
      const expectedPath = path.join(appPath, contract.relativePath)
      try {
        const reopened = readExternalFileBytes(
          expectedPath,
          `Paired installed ${label.replaceAll('_', ' ')}`,
          { maxBytes: executableIdentityMaxBytes, retainRaw: false },
        )
        if (
          record.opened_nofollow !== true
          || record.requested_path !== expectedPath
          || record.path !== reopened.path
          || record.sha256 !== reopened.sha256
          || Number(record.size_bytes) !== reopened.bytes
          || record.sha256 !== manifest[contract.manifestField]
        ) {
          failures.push(`paired matrix installed ${label} bytes/path are invalid`)
        }
      } catch (error) {
        failures.push(String(error?.message || error))
      }
    }
    if (
      uiRuntime.app_asar !== path.join(
        appPath,
        installedAppArtifactPaths.app_asar.relativePath,
      )
      || manifest.app_asar_sha256 !== uiRuntime.app_asar_sha256
      || uiRuntime.electron_executable !== path.join(
        appPath,
        installedAppArtifactPaths.electron_executable.relativePath,
      )
      || manifest.electron_executable_sha256
        !== uiRuntime.electron_executable_sha256
      || uiRuntime.bundled_provenance_path !== path.join(
        appPath,
        installedAppArtifactPaths.bundled_provenance.relativePath,
      )
      || manifest.bundled_provenance_sha256
        !== uiRuntime.bundled_provenance_sha256
      || canonicalJson(installed.bundled_provenance)
        !== canonicalJson(uiRuntime.bundled_provenance)
      || canonicalJson(installed.bundled_source)
        !== canonicalJson(uiRuntime.bundled_source)
      || canonicalJson(sourceBinding) !== canonicalJson(expectedSourceBinding)
    ) {
      failures.push('paired matrix installed provenance does not match the exact UI package/source')
    }
  } else {
    failures.push('paired matrix producer cannot be bound to an unknown UI runtime mode')
  }
  const producerPid = Number(runnerBefore.producer_pid)
  const forbiddenPids = [
    result?.healthProvenance?.after?.binding?.backend_pid,
    result?.uiRuntimeProvenance?.cdp_process_binding?.launched_root_pid,
    result?.uiRuntimeProvenance?.cdp_process_binding?.listener_pid,
  ].map(Number)
  if (!Number.isInteger(producerPid) || producerPid <= 0 || forbiddenPids.includes(producerPid)) {
    failures.push('paired matrix producer PID is missing or aliases Electron/backend ownership')
  }
  const uiBinding = result?.healthProvenance?.after?.binding || {}
  if (
    !validSha256(producerExecutableSha256)
    || producerExecutableSha256
      !== result?.uiRuntimeProvenance?.backend_python_process_binding?.executable_sha256
  ) {
    failures.push(
      'paired matrix producer Python canonical executable identity does not match the UI backend Python',
    )
  }
  const bundleBefore = identity?.bundle?.before || {}
  const bundleAfter = identity?.bundle?.after || {}
  if (
    canonicalJson(bundleBefore) !== canonicalJson(bundleAfter)
    || bundleBefore.fingerprint_sha256 !== uiBinding.model_bundle_fingerprint_sha256
    || canonicalJson(bundleBefore.files)
      !== canonicalJson(result?.bundleGenerationContract?.health_attestation?.files)
  ) {
    failures.push('paired matrix bundle identity is not bound to the UI model bundle')
  }
  for (const baseLabel of ['direct', 'gateway']) {
    for (const phase of ['before', 'after']) {
      const row = identity?.health?.[baseLabel]?.[phase] || {}
      let healthOrigin = ''
      let requestOrigin = ''
      try {
        healthOrigin = new URL(row.url).origin
        requestOrigin = new URL(value?.bases?.[baseLabel]).origin
      } catch {
        failures.push(`${baseLabel} ${phase} health URL origin is invalid`)
      }
      if (!healthOrigin || healthOrigin !== requestOrigin) {
        failures.push(
          `${baseLabel} ${phase} health URL origin is not bound to its request base`,
        )
      }
      // Hash the producer's exact Python-canonical bytes. JS cannot reproduce
      // Python's numeric lexical distinction after JSON.parse (for example
      // 10.0 becomes 10), so the producer records both canonical bytes and the
      // parsed health object; both representations remain mutually bound.
      if (
        !row.full
        || !validSha256(row.full_sha256)
        || !pythonCanonicalJsonMatchesParsed(
          row.full_canonical_json,
          row.full,
          row.full_sha256,
        )
        || !healthIdentityMatchesUi(row.identity, uiBinding)
        || row.identity?.fingerprint_sha256 !== value?.backend_identity_fingerprint_sha256
      ) {
        failures.push(`${baseLabel} ${phase} health identity is not bound to the UI backend`)
      }
    }
  }
  return failures
}

function openedManifestArtifact(runDirectory, fileName, label) {
  if (
    !fileName
    || path.basename(fileName) !== fileName
    || ['.', '..'].includes(fileName)
  ) {
    throw new Error(`${label} filename is unsafe`)
  }
  const candidate = path.resolve(runDirectory, fileName)
  if (!isPathInside(candidate, runDirectory)) {
    throw new Error(`${label} escapes its private run directory`)
  }
  return readPrivateExternalBytes(candidate, label)
}

function comparableManifestSummary(rawCapture) {
  const copy = structuredClone(rawCapture || {})
  for (const key of [
    'manifest_file',
    'manifest_path',
    'manifest_sha256',
    'run_directory',
  ]) delete copy[key]
  return copy
}

function captureHeadersAreSanitized(headers) {
  return Array.isArray(headers) && headers.every((header) => {
    const name = String(header?.name || '')
      .trim()
      .toLowerCase()
      .replaceAll('_', '-')
    const value = String(header?.value ?? '')
    return name
      && (
        pairedSafeCaptureHeaderNames.has(name)
        || value === '<redacted>'
      )
  })
}

function validateRawMatrixCapture(value, result, contract) {
  const failures = []
  const rawCapture = value?.raw_capture || {}
  if (
    rawCapture.enabled !== true
    || rawCapture.complete !== true
    || Number(rawCapture.errors) !== 0
    || rawCapture.run_id !== result?.run_id
    || !validSha256(rawCapture.manifest_sha256)
  ) {
    return ['paired matrix raw parser-input capture is incomplete or unbound']
  }
  let manifestOpened
  let manifest
  try {
    manifestOpened = readPrivateExternalBytes(
      rawCapture.manifest_path,
      'Paired API raw capture manifest',
    )
    manifest = JSON.parse(manifestOpened.raw.toString('utf8'))
  } catch (error) {
    return [String(error?.message || error)]
  }
  let runDirectory
  try {
    const declaredRunDirectory = path.resolve(rawCapture.run_directory || '')
    const runDirectoryStat = lstatSync(declaredRunDirectory)
    runDirectory = realpathSync(declaredRunDirectory)
    if (
      !runDirectoryStat.isDirectory()
      || runDirectoryStat.isSymbolicLink()
      || (runDirectoryStat.mode & 0o077) !== 0
      || isPathInside(runDirectory, realpathSync(repoDir))
    ) {
      return ['paired matrix raw capture run directory is not private, canonical, and external']
    }
  } catch (error) {
    return [`paired matrix raw capture run directory is invalid: ${String(error?.message || error)}`]
  }
  if (
    manifestOpened.sha256 !== rawCapture.manifest_sha256
    || manifestOpened.path !== path.resolve(runDirectory, rawCapture.manifest_file)
    || path.dirname(manifestOpened.path) !== runDirectory
    || canonicalJson(manifest) !== canonicalJson(comparableManifestSummary(rawCapture))
  ) {
    failures.push('paired matrix raw capture manifest bytes/path/summary do not match')
  }
  const includeCancellation = value?.checks?.abort_recovery_skipped !== true
  const expectedRoutes = expectedRawMatrixRoutes(contract, includeCancellation)
  const routes = Array.isArray(manifest?.routes) ? manifest.routes : []
  const actualRoutes = routes.map((route) => (
    `${route?.base_label || ''}\0${route?.protocol || ''}\0${route?.capture_label || ''}`
  ))
  if (
    Number(manifest?.schema_version) !== 1
    || manifest?.capture_layer !== pairedCaptureLayer
    || manifest?.capture_semantics !== pairedCaptureSemantics
    || Number(manifest?.expected) !== expectedRoutes.length
    || Number(manifest?.started) !== expectedRoutes.length
    || Number(manifest?.finished) !== expectedRoutes.length
    || Number(manifest?.errors) !== 0
    || manifest?.complete !== true
    || manifest?.run_id !== result?.run_id
  ) {
    failures.push('paired matrix raw capture manifest contract/totals are not exact')
  }
  if (
    canonicalJson([...actualRoutes].sort()) !== canonicalJson([...expectedRoutes].sort())
    || new Set(actualRoutes).size !== actualRoutes.length
  ) {
    failures.push('paired matrix raw capture routes are missing, duplicated, or unexpected')
  }
  for (const route of routes) {
    const label = `${route?.base_label}/${route?.protocol}/${route?.capture_label}`
    const artifacts = Array.isArray(route?.artifacts) ? route.artifacts : []
    if (
      route?.expected !== 1
      || route?.started !== 1
      || route?.finished !== 1
      || (route?.errors || []).length
      || artifacts.length !== 1
      || artifacts[0]?.verified !== true
    ) {
      failures.push(`${label} raw capture lifecycle is not exact`)
      continue
    }
    const artifact = artifacts[0]
    let bodyOpened
    let metadataOpened
    let metadata
    try {
      bodyOpened = openedManifestArtifact(runDirectory, artifact.body_file, `${label} body`)
      metadataOpened = openedManifestArtifact(
        runDirectory,
        artifact.metadata_file,
        `${label} metadata`,
      )
      metadata = JSON.parse(metadataOpened.raw.toString('utf8'))
    } catch (error) {
      failures.push(String(error?.message || error))
      continue
    }
    if (
      Number(metadata?.schema_version) !== 1
      || metadata?.capture_layer !== pairedCaptureLayer
      || metadata?.capture_semantics !== pairedCaptureSemantics
      || bodyOpened.sha256 !== artifact.body_sha256
      || bodyOpened.bytes !== Number(artifact.body_bytes)
      || metadataOpened.sha256 !== artifact.metadata_sha256
      || metadata?.base_label !== route.base_label
      || metadata?.protocol !== route.protocol
      || metadata?.capture_label !== route.capture_label
      || metadata?.response?.body_file !== artifact.body_file
      || metadata?.response?.body_sha256 !== bodyOpened.sha256
      || Number(metadata?.response?.body_bytes) !== bodyOpened.bytes
      || Number(metadata?.response?.status_code) !== 200
      || metadata?.response?.capture_error_type
      || metadata?.request?.body_sha256 !== artifact.request_body_sha256
      || !validSha256(metadata?.request?.body_sha256)
      || metadata?.request?.prepared_payload_body_sha256
        !== artifact.prepared_payload_body_sha256
      || metadata?.request?.prepared_payload_canonical_body_sha256
        !== artifact.prepared_payload_canonical_body_sha256
      || metadata?.request?.prepared_payload_body_sha256
        !== metadata?.request?.payload?.body_sha256
      || metadata?.request?.prepared_payload_canonical_body_sha256
        !== metadata?.request?.payload?.canonical_body_sha256
      || !validSha256(metadata?.request?.prepared_payload_body_sha256)
      || !validSha256(
        metadata?.request?.prepared_payload_canonical_body_sha256,
      )
      || !captureHeadersAreSanitized(metadata?.request?.headers)
      || !captureHeadersAreSanitized(metadata?.response?.headers)
      || metadata?.request?.payload?.model !== value?.requested_model
      || metadata?.request?.url !== `${value?.bases?.[route.base_label]}${pairedApiProtocolRoutes[route.protocol]}`
    ) {
      failures.push(`${label} raw body/metadata/request/response binding is invalid`)
    }
    const streamRoundMatch = String(route.capture_label).match(
      /^stream-flow-round([123])$/,
    )
    const nonstreamRoundMatch = String(route.capture_label).match(
      /^nonstream-flow-round([123])$/,
    )
    if (!streamRoundMatch && !nonstreamRoundMatch) continue
    const mode = streamRoundMatch ? 'stream' : 'nonstream'
    const roundMatch = streamRoundMatch || nonstreamRoundMatch
    const roundIndex = Number(roundMatch[1]) - 1
    const flow = value?.flows?.[route.base_label]?.[route.protocol]?.[mode]
    const round = flow?.rounds?.[roundIndex]
    const request = flow?.requests?.[roundIndex]
    let parsed
    try {
      parsed = mode === 'stream'
        ? collectProtocolStream(route.protocol, bodyOpened.raw, label)
        : collectProtocolNonstream(route.protocol, bodyOpened.raw, label)
    } catch (error) {
      failures.push(String(error?.message || error))
      continue
    }
    if (mode === 'stream') {
      failures.push(...validateRawChannelOrder(
        parsed,
        roundIndex,
        label,
        request?.enable_thinking === true,
      ))
    }
    if (
      parsed.postTerminalEvents !== 0
      || parsed.terminal !== true
      || canonicalJson(parsed.terminalReasons) !== canonicalJson(round?.terminals)
      || sha256Text(parsed.reasoning) !== round?.reasoning_sha256
      || parsed.reasoning.length !== Number(round?.reasoning_chars)
      || sha256Text(parsed.content) !== round?.content_sha256
      || parsed.content.length !== Number(round?.content_chars)
      || metadata?.request?.payload?.body_sha256 !== request?.body_sha256
      || metadata?.request?.payload?.canonical_body_sha256 !== request?.canonical_body_sha256
      || canonicalJson(metadata?.request?.payload?.tool_contracts)
        !== canonicalJson(request?.tool_contracts)
      || canonicalJson(metadata?.request?.payload?.tool_history_linkage)
        !== canonicalJson(request?.tool_history_linkage)
    ) {
      failures.push(`${label} raw bytes do not reproduce the public flow/request evidence`)
    }
    const expectedCall = roundIndex < 2 ? round?.tool_calls?.[0] : null
    const parsedCall = parsed.toolCalls?.[0]
    if (
      expectedCall
      && (
        parsed.toolCalls.length !== 1
        || String(parsedCall?.id || '') !== String(expectedCall?.id || '')
        || parsedCall?.name !== expectedCall?.name
        || canonicalJson(normalizedToolArguments(parsedCall?.arguments))
          !== canonicalJson(expectedCall?.arguments)
      )
    ) {
      failures.push(`${label} raw tool call does not match the public exact tool call`)
    }
    if (!expectedCall && parsed.toolCalls.length) {
      failures.push(`${label} raw final answer unexpectedly emitted a tool call`)
    }
  }
  return failures
}

export function validateFrozenChatParity(value) {
  const failures = []
  for (const stage of [2, 3]) {
    const replay = value?.paired_replays?.[`chat_nonstream_round${stage}`]
    const request = replay?.request || {}
    const flowRequest = value?.flows?.direct?.chat?.nonstream?.requests?.[stage - 1] || {}
    const hashes = request?.leg_body_sha256 || {}
    const expectedHash = request?.prepared_body_sha256
    const expectedThinking = flowRequest?.enable_thinking
    if (
      replay?.schema !== 'vmlx-agentic-protocol-paired-replay-v1'
      || canonicalJson(replay?.target) !== canonicalJson({
        protocol: 'chat',
        mode: 'nonstream',
        stage,
      })
      || replay?.pass !== true
      || replay?.checks?.exact_body_sha_equal !== true
      || replay?.checks?.gateway_backend_lifecycle_pass !== true
    ) {
      failures.push(
        `Chat nonstream round ${stage} stochastic-history parity lacks a passing frozen paired replay`,
      )
    }
    if (
      !validSha256(expectedHash)
      || canonicalJson(Object.keys(hashes).sort()) !== canonicalJson(['a1', 'a2', 'b'])
      || Object.values(hashes).some((hash) => hash !== expectedHash)
    ) {
      failures.push(
        `Chat nonstream round ${stage} frozen paired replay did not transmit one exact body on all three legs`,
      )
    }
    if (
      request?.body_sha256 !== expectedHash
      || request?.body_sha256 !== flowRequest?.body_sha256
      || typeof expectedThinking !== 'boolean'
      || request?.enable_thinking !== expectedThinking
    ) {
      failures.push(
        `Chat nonstream round ${stage} frozen paired replay is not bound to the transmitted ${expectedThinking ? 'On' : 'Off'} flow body`,
      )
    }
  }
  return failures
}

export function validatePairedApiEvidence(result) {
  const failures = []
  const artifact = result?.pairedApiArtifact
  const claimsDualSurface = result?.surfaceStatus === 'dual_surface_attested'
  if (!artifact) {
    if (claimsDualSurface) {
      failures.push('dual-surface status was claimed without a separate raw API artifact')
    }
    return failures
  }
  let reopened
  try {
    reopened = readPrivateExternalJson(artifact.path, 'Paired raw API proof artifact')
  } catch (error) {
    return [String(error?.message || error)]
  }
  if (
    reopened.sha256 !== artifact.sha256
    || canonicalJson(reopened.value) !== canonicalJson(artifact.value)
    || reopened.opened_nofollow !== true
    || reopened.nlink !== 1
  ) {
    failures.push('paired raw API metadata does not match its safely reopened artifact bytes')
  }
  const value = reopened.value || {}
  const contract = expectedPairedApiContract(result)
  if (!contract.valid) {
    failures.push(
      `paired API action profile is missing, inconsistent, or unsupported: ${contract.profile || 'unnamed'}`,
    )
  }
  if (
    value.schema !== 'vmlx-agentic-protocol-matrix-v2'
    || Number(value.schema_version) !== 2
    || value.pass !== true
    || value.run_id !== result?.run_id
    || value.requested_model !== result?.servedModel
    || path.resolve(value.repo_root || '') !== realpathSync(repoDir)
    || value.second_tool_choice !== 'explicit'
    || !exactStringSet(value.protocols, contract.protocols)
    || !exactStringSet(value.modes, contract.modes)
  ) {
    failures.push('paired API artifact is not the exact passing vmlx-agentic-protocol-matrix-v2 run')
  }
  for (const [name, passed] of Object.entries(value?.checks || {})) {
    if (name === 'abort_recovery_skipped') continue
    if (passed !== true) failures.push(`paired API matrix check failed: ${name}`)
  }
  let directOrigin = ''
  let gatewayOrigin = ''
  let uiOrigin = ''
  try {
    directOrigin = new URL(value?.bases?.direct).origin
    gatewayOrigin = new URL(value?.bases?.gateway).origin
    uiOrigin = new URL(result?.baseUrl || result?.server?.baseUrl).origin
  } catch {
    failures.push('paired API matrix direct/gateway origins are invalid')
  }
  if (
    directOrigin !== uiOrigin
    || !gatewayOrigin
    || gatewayOrigin === directOrigin
  ) {
    failures.push('paired API direct/gateway origins are not bound to the UI session')
  }
  failures.push(...validateMatrixIdentity(value, result))
  const flowBases = value?.flows || {}
  if (!exactStringSet(Object.keys(flowBases), ['direct', 'gateway'])) {
    failures.push('paired API matrix flow bases are not exactly direct+gateway')
  } else {
    for (const baseLabel of ['direct', 'gateway']) {
      if (!exactStringSet(Object.keys(flowBases[baseLabel] || {}), contract.protocols)) {
        failures.push(`${baseLabel} matrix protocols are incomplete`)
        continue
      }
      for (const protocol of contract.protocols) {
        const modes = flowBases[baseLabel]?.[protocol] || {}
        if (!exactStringSet(Object.keys(modes), contract.modes)) {
          failures.push(`${baseLabel}/${protocol} matrix modes are incomplete`)
          continue
        }
        for (const mode of contract.modes) {
          failures.push(...validateMatrixFlow(
            modes[mode],
            protocol,
            mode,
            `${baseLabel}/${protocol}/${mode}`,
            value.repo_root,
          ))
        }
      }
    }
  }
  for (const protocol of contract.protocols) {
    for (const mode of contract.modes) {
      const directRequests = flowBases?.direct?.[protocol]?.[mode]?.requests || []
      const gatewayRequests = flowBases?.gateway?.[protocol]?.[mode]?.requests || []
      // Only the initial request is byte-comparable. Later direct and gateway
      // requests are independently generated conversations, so stochastic
      // reasoning and tool IDs legitimately make their histories differ.
      // validateMatrixFlow() already binds each continuation to its own
      // captured prior result and tool-result history.
      const parityIndexes = [0]
      if (
        directRequests.length !== 3
        || gatewayRequests.length !== 3
        || directRequests.some((request) => !validSha256(request?.canonical_body_sha256))
        || gatewayRequests.some((request) => !validSha256(request?.canonical_body_sha256))
        || parityIndexes.some((index) => (
          directRequests[index]?.canonical_body_sha256
          !== gatewayRequests[index]?.canonical_body_sha256
        ))
      ) {
        failures.push(
          `${protocol}/${mode} initial direct and gateway canonical request bodies are not byte-parity equivalent`,
        )
      }
    }
  }
  if (contract.requireFrozenChatParity) {
    failures.push(...validateFrozenChatParity(value))
  } else if (Object.keys(value?.paired_replays || {}).length !== 0) {
    failures.push('scoped paired API matrix unexpectedly contains frozen nonstream replay evidence')
  }
  failures.push(...validateRawMatrixCapture(value, result, contract))
  if (claimsDualSurface && !isServerRequestCorrelationVerified(result)) {
    failures.push('dual-surface status was claimed without request/cache/settings correlation')
  }
  return failures
}

export function applyTopLevelCorrelationStatus(
  result,
  { rendererFailed = false } = {},
) {
  if (rendererFailed) {
    result.status = 'fail'
    result.pass = false
    result.surfaceStatus = 'partial_ui_only'
    return result
  }
  const correlationVerified = isServerRequestCorrelationVerified(result)
  const pairedArtifactPresent = Boolean(result?.pairedApiArtifact)
  result.status = correlationVerified && pairedArtifactPresent ? 'pass' : 'partial'
  result.pass = correlationVerified && pairedArtifactPresent
  result.surfaceStatus = result?.pairedApiArtifact
    ? correlationVerified
      ? 'dual_surface_attested'
      : 'partial_dual_surface_uncorrelated'
    : 'partial_ui_only'
  return result
}

export function applyAssertionFailureStatus(result, error) {
  result.status = 'fail'
  result.pass = false
  result.failureStage = result.failureStage || 'release_assertions'
  result.assertionFailures = error?.failures || [
    error?.message || String(error),
  ]
  return result
}

export function validateGatewaySingleModelEvidence(result) {
  if (!result?.ownedRunIntent) return []
  const failures = []
  const evidence = result.gatewaySingleModelMode || {}
  const serverControl = evidence.serverSettingsControl || {}
  const toggleControl = evidence.toggleControl || {}
  const statusAfter = evidence.gatewayStatusAfterToggle || {}
  const statusBeforeStart = evidence.gatewayStatusImmediatelyBeforeStart || {}
  const persistedSetting =
    evidence.persistedSettingImmediatelyBeforeStart || {}
  if (
    serverControl.selector !== '[data-vmlx-control="server-settings"]'
    || serverControl.visible !== true
    || serverControl.ariaPressedAfterOpen !== 'true'
    || serverControl.ariaPressedAfterClose !== 'false'
  ) {
    failures.push(
      'V5 Single model evidence did not visibly open and close the real Server settings control',
    )
  }
  if (
    toggleControl.selector
      !== '[data-vmlx-control="gateway-single-model-mode"]'
    || toggleControl.visible !== true
    || toggleControl.ariaPressedAfter !== 'true'
  ) {
    failures.push(
      'V5 Single model evidence did not attest the visible Server settings toggle as enabled',
    )
  }
  const alreadyOnPath = (
    toggleControl.ariaPressedBefore === 'true'
    && toggleControl.observedAlreadyOn === true
    && toggleControl.clickedToEnable === false
  )
  const enabledByClickPath = (
    toggleControl.ariaPressedBefore !== 'true'
    && toggleControl.observedAlreadyOn === false
    && toggleControl.clickedToEnable === true
  )
  if (!alreadyOnPath && !enabledByClickPath) {
    failures.push(
      'V5 Single model evidence did not truthfully distinguish already-on state from a visible enable click',
    )
  }
  if (
    statusAfter.running !== true
    || statusAfter.singleModelMode !== true
    || statusBeforeStart.running !== true
    || statusBeforeStart.singleModelMode !== true
  ) {
    failures.push(
      'V5 Single model evidence did not bind the enabled DOM toggle to live gateway status before Start',
    )
  }
  if (
    persistedSetting.key !== 'gateway_single_model_mode'
    || persistedSetting.source !== 'window.api.settings.get'
    || persistedSetting.value !== 'true'
  ) {
    failures.push(
      'V5 Single model evidence did not independently read the persisted gateway setting before Start',
    )
  }
  return failures
}

export function validateFinalPhaseStopEvidence(result) {
  const phaseIndex = Number(result?.ownedRunIntent?.phase_index)
  const finalPhaseRequired = phaseIndex === 5 && Boolean(result?.releaseEvidence)
  if (!finalPhaseRequired) return []
  const failures = []
  const evidence = result.finalPhaseStopEvidence || {}
  const visibleControl = evidence.visibleControl || {}
  const session = evidence.session || {}
  const backend = evidence.backend || {}
  const survivors = evidence.survivors || {}
  let expectedBackendPort = null
  try {
    expectedBackendPort = Number(new URL(result?.baseUrl || '').port)
  } catch {}
  if (
    evidence.releaseSentinelConsumed !== true
    || evidence.phaseIndex !== 5
  ) {
    failures.push('final V5 proof-session Stop did not follow phase-5 sentinel consumption')
  }
  if (
    visibleControl.selector !== '[data-vmlx-control="session-stop"]'
    || visibleControl.clicked !== true
    || typeof visibleControl.label !== 'string'
    || !visibleControl.label.trim()
    || visibleControl.sessionId !== result?.session?.id
    || visibleControl.exactSelector
      !== (
        '[data-vmlx-control="session-stop"]'
        + `[data-vmlx-session-id="${cssEscapeIdentifier(result?.session?.id || '')}"]`
      )
  ) {
    failures.push('final V5 proof session was not stopped through its exact visible Stop control')
  }
  const before = session.before || {}
  const after = session.after || {}
  if (
    session.id !== result?.session?.id
    || !['running', 'standby'].includes(before.status)
    || before.pid !== result?.backend?.pid
    || before.port !== expectedBackendPort
    || after.status !== 'stopped'
    || after.pid !== null
    || after.port !== expectedBackendPort
    || session.pidClearSemantics !== 'nullable_pid_cleared'
    || session.portClearSemantics !== 'non_nullable_endpoint_retained'
  ) {
    failures.push(
      'final V5 proof-session database state was not bound to the backend PID/port and durably stopped',
    )
  }
  if (
    backend.backend_pid !== result?.backend?.pid
    || !Number.isInteger(expectedBackendPort)
    || expectedBackendPort <= 0
    || backend.port !== expectedBackendPort
    || backend.backend_process_gone !== true
    || backend.listener_gone !== true
    || !Array.isArray(backend.observed_listener_pids)
    || backend.observed_listener_pids.length !== 0
  ) {
    failures.push('final V5 proof backend process/listener teardown was not exactly attested')
  }
  const expectedRetainedPids = Array.isArray(result?.requestedRetainedPids)
    ? result.requestedRetainedPids
    : []
  const expectedElectronPid = Number(result?.uiBackendBinding?.electron_pid)
  const expectedGatewayPid = Number(
    result?.uiSessionAttestation?.value?.gateway_pid
    ?? result?.uiBackendBinding?.gateway_process_binding?.listener_pid,
  )
  const validateSnapshot = (snapshot, stage) => {
    if (
      snapshot?.stage !== stage
      || JSON.stringify(snapshot?.expected_retained_pids || [])
        !== JSON.stringify(expectedRetainedPids)
      || !Array.isArray(snapshot?.processes)
    ) return false
    const expected = [
      ['parent_electron', expectedElectronPid],
      ['parent_gateway', expectedGatewayPid],
      ...expectedRetainedPids.map((pid, index) => [
        `explicit_retained_${index + 1}`,
        Number(pid),
      ]),
    ]
    return (
      snapshot.processes.length === expected.length
      && expected.every(([role, pid]) =>
        snapshot.processes.some((process) =>
          process?.role === role
          && process?.pid === pid
          && process?.alive === true
        )
      )
    )
  }
  if (
    expectedRetainedPids.length === 0
    || expectedRetainedPids.some((pid) =>
      !Number.isInteger(Number(pid)) || Number(pid) <= 1
    )
    || new Set(expectedRetainedPids.map(Number)).size
      !== expectedRetainedPids.length
    || !Number.isInteger(expectedElectronPid)
    || expectedElectronPid <= 1
    || !Number.isInteger(expectedGatewayPid)
    || expectedGatewayPid <= 1
    || expectedElectronPid !== expectedGatewayPid
    || expectedRetainedPids.map(Number).some((pid) =>
      pid === expectedElectronPid
      || pid === expectedGatewayPid
      || pid === Number(result?.backend?.pid)
    )
    || !validateSnapshot(survivors.before, 'before_visible_stop')
    || !validateSnapshot(survivors.after, 'after_backend_teardown')
  ) {
    failures.push(
      'final V5 cleanup did not attest parent Electron, gateway, and explicit retained PIDs before and after Stop',
    )
  }
  return failures
}

function assertResult(result) {
  const failures = []
  const chat = result.chat || {}
  const healthBeforeBinding = result.healthProvenance?.before?.binding || {}
  const healthAfterBinding = result.healthProvenance?.after?.binding || {}
  const expectedTurns = expectedUiTurnCount(result)
  const sessionConfig = result.session?.effective_config || {}
  const cacheTelemetryExpected = (
    sessionConfig.enablePrefixCache !== false
    && sessionConfig.continuousBatching !== false
  )
  const positiveCacheReuseExpected = (
    cacheTelemetryExpected
    && uiProfileRequiresPositiveCacheReuse(result)
  )
  if (result.format !== proofFormat) failures.push(`expected proof format ${proofFormat}`)
  if (result.bundleGenerationContract?.template?.usable !== true) {
    failures.push(
      result.bundleGenerationContract?.template?.warning
      || 'bundle exposes no usable chat template',
    )
  }
  if (!result.run_id) failures.push('stable run_id was not recorded')
  if (result.uiStartControl?.clicked !== true) failures.push('session was not started through the visible Electron local Start control')
  if (
    result.uiStartControl?.selector !== '[data-vmlx-control="session-start"]'
    || result.uiStartControl?.sessionId !== result?.session?.id
    || result.uiStartControl?.exactSelector
      !== (
        '[data-vmlx-control="session-start"]'
        + `[data-vmlx-session-id="${cssEscapeIdentifier(result?.session?.id || '')}"]`
      )
  ) {
    failures.push('clicked control was not the exact session-bound local Start control')
  }
  if (
    result.preStartStopControl
    && (
      result.preStartStopControl.selector
        !== '[data-vmlx-control="session-stop"]'
      || result.preStartStopControl.sessionId !== result?.session?.id
      || result.preStartStopControl.exactSelector
        !== (
          '[data-vmlx-control="session-stop"]'
          + `[data-vmlx-session-id="${cssEscapeIdentifier(result?.session?.id || '')}"]`
        )
      || result.preStartStopControl.clicked !== true
      || result.preStartStopControl.statusAfter !== 'stopped'
    )
  ) {
    failures.push(
      'pre-Start active session was not stopped through its exact session-bound visible Stop/Cancel control',
    )
  }
  if (result.uiStartControl?.sessionStatusAfter !== 'running') {
    failures.push(`UI-started session did not reach running state: ${result.uiStartControl?.sessionStatusAfter || 'unknown'}`)
  }
  failures.push(...validateGatewaySingleModelEvidence(result))
  failures.push(...validateFinalPhaseStopEvidence(result))
  if (!result.server?.models?.data?.length) failures.push('real server /v1/models returned no models')
  if (result.localSessionStarted !== true) failures.push('local model session did not start through the Electron UI control')
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
  if (reasoningNumericRunIsSpew(chat)) {
    failures.push('numeric/list-like garbage leaked into reasoning segments')
  }
  if ((result.eventCounts?.complete || 0) < expectedTurns) {
    failures.push(`expected ${expectedTurns} completed UI chat turns`)
  }
  if ((result.eventCounts?.stream || 0) < 1) failures.push('expected streaming events from real model')
  if ((chat.turns?.length || 0) < expectedTurns * 2) {
    failures.push(`expected at least ${expectedTurns * 2} persisted chat messages, got ${chat.turns?.length || 0}`)
  }
  if (
    !Array.isArray(result.messageEventTrace)
    || result.messageEventTrace.length < expectedTurns
  ) {
    failures.push('per-message reasoning/content/tool/terminal stream trace is incomplete')
  }
  const tracedTerminalMessages = new Set(
    (result.messageEventTrace || [])
      .filter((row) => Array.isArray(row?.events) && row.events.some((event) => event?.event === 'terminal'))
      .map((row) => row.messageId),
  )
  if (tracedTerminalMessages.size < expectedTurns) {
    failures.push(`expected a terminal event for each of ${expectedTurns} UI turns`)
  }
  if (!result.healthProvenance?.before?.raw || !result.healthProvenance?.after?.raw) {
    failures.push('full /health provenance snapshots were not retained before and after')
  }
  if (!healthBeforeBinding.backend_pid || !healthAfterBinding.backend_pid) failures.push('/health runtime PID provenance is missing')
  for (const field of [
    'server_module_sha256',
    'package_init_sha256',
    'python_source_tree_sha256',
    'python_executable_fingerprint_sha256',
  ]) {
    if (
      !validSha256(healthBeforeBinding.runtime_source_hashes?.[field])
      || !validSha256(healthAfterBinding.runtime_source_hashes?.[field])
    ) failures.push(`/health runtime source hash provenance is invalid: ${field}`)
  }
  for (const field of [
    'server_module_sha256',
    'package_init_sha256',
    'python_source_tree_sha256',
  ]) {
    if (
      healthBeforeBinding.runtime_source_hashes?.[field]
        !== result.gitProvenance?.before?.[field]
      || healthAfterBinding.runtime_source_hashes?.[field]
        !== result.gitProvenance?.after?.[field]
    ) failures.push(`/health runtime source does not match observed checkout: ${field}`)
  }
  for (const field of [
    'python_source_file_count',
    'python_source_read_error_count',
  ]) {
    if (
      healthBeforeBinding[field] !== result.gitProvenance?.before?.[field]
      || healthAfterBinding[field] !== result.gitProvenance?.after?.[field]
    ) failures.push(`/health runtime source count does not match observed checkout: ${field}`)
  }
  if (
    !validSha256(healthBeforeBinding.model_bundle_fingerprint_sha256)
    || !validSha256(healthAfterBinding.model_bundle_fingerprint_sha256)
  ) failures.push('/health model-bundle fingerprint provenance is missing')
  if (
    !validSha256(healthBeforeBinding.cache_topology_fingerprint_sha256)
    || !validSha256(healthAfterBinding.cache_topology_fingerprint_sha256)
  ) failures.push('/health cache-topology fingerprint provenance is missing')
  if (
    healthBeforeBinding.backend_pid !== healthAfterBinding.backend_pid
    || healthBeforeBinding.backend_pid !== result.backend?.pid
  ) failures.push('backend PID changed or does not match /health runtime provenance')
  if (
    !validSha256(healthBeforeBinding.fingerprint_sha256)
    || healthBeforeBinding.fingerprint_sha256 !== healthAfterBinding.fingerprint_sha256
    || healthBeforeBinding.fingerprint_sha256
      !== result.backend_identity_fingerprint_sha256
  ) failures.push('canonical backend identity fingerprint is missing or changed')
  if (
    healthBeforeBinding.model_bundle_fingerprint_sha256
    !== healthAfterBinding.model_bundle_fingerprint_sha256
  ) failures.push('model-bundle fingerprint changed during the UI proof')
  if (
    healthBeforeBinding.cache_topology_fingerprint_sha256
    !== healthAfterBinding.cache_topology_fingerprint_sha256
  ) failures.push('cache-topology fingerprint changed during the UI proof')
  if (
    !result.gitProvenance?.before?.commit
    || !result.gitProvenance?.before?.tree
    || !result.gitProvenance?.after?.commit
    || !result.gitProvenance?.after?.tree
  ) failures.push('exact Git commit/tree provenance is missing')
  if (result.gitProvenance?.before?.dirty || result.gitProvenance?.after?.dirty) {
    failures.push('real UI proof requires one clean source checkout')
  }
  if (
    result.gitProvenance?.before?.commit !== result.gitProvenance?.after?.commit
    || result.gitProvenance?.before?.tree !== result.gitProvenance?.after?.tree
    || result.gitProvenance?.before?.harness_sha256 !== result.gitProvenance?.after?.harness_sha256
  ) failures.push('source or UI proof harness changed during the live run')
  if (
    result.screenshotCapture?.status !== 'captured'
    || !result.screenshotCapture?.path
    || result.screenshotCapture?.error
    || result.screenshotCapture?.attestation?.path !== result.screenshotCapture?.path
    || !Number.isInteger(result.screenshotCapture?.attestation?.byteSize)
    || result.screenshotCapture?.attestation?.byteSize <= 0
    || !validSha256(result.screenshotCapture?.attestation?.sha256)
    || result.screenshotCapture?.attestation?.openedNoFollow !== true
    || result.screenshotCapture?.attestation?.regularFile !== true
    || result.screenshotCapture?.attestation?.nonSymlink !== true
    || result.screenshotCapture?.attestation?.privatePermissions !== true
    || result.screenshotCapture?.attestation?.exactRequestedPath !== true
    || result.screenshotCapture?.attestation?.pngSignatureValid !== true
    || !screenshotAttestationsMatch(
      result.screenshotCapture?.attestation,
      result.screenshotCapture?.finalAttestation,
    )
    || result.screenshots?.chat !== result.screenshotCapture?.path
  ) failures.push('required real UI screenshot was not successfully captured')
  if (result.sendErrors?.length) failures.push(`renderer send errors: ${result.sendErrors.join('; ')}`)
  if (positiveCacheReuseExpected && (result.cache?.cacheHitTokens || 0) <= 0) failures.push('expected real cache-hit token telemetry after a reuse probe')
  if (positiveCacheReuseExpected && !result.provenSurfaces?.includes('cache_hit_telemetry')) {
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
  if (!result.provenSurfaces?.includes('generation_defaults_visible_ui')) {
    failures.push('live proof did not record visible model-owned generation defaults')
  }
  if (
    isServerRequestCorrelationVerified(result)
    && !result.provenSurfaces?.includes('generation_defaults_applied')
  ) {
    failures.push('server-correlated request did not record resolved generation defaults')
  }
  if (!result.provenSurfaces?.includes('language_leak_check')) {
    failures.push('live proof did not record clean visible/reasoning language leak check')
  }
  const expectedToolCalls = expectedUiToolCallCount(result)
  if (expectedToolCalls > 0 && !result.provenSurfaces?.includes('tool_loop')) {
    failures.push('UI action profile expected real built-in tools but proof did not record tool_loop surface')
  }
  if (expectedToolCalls >= 2 && !result.provenSurfaces?.includes('long_tool_loop')) {
    failures.push('UI action profile expected two real built-in tools but proof did not record long_tool_loop surface')
  }
  if (
    (result.requestedEnableThinking === true || result.requestedReasoningEffort != null)
    && !result.provenSurfaces?.includes('reasoning_display')
  ) {
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
  if (result.requestedAudio === true && !result.provenSurfaces?.includes('audio_where_supported')) {
    failures.push('requested real audio media but proof did not record audio_where_supported surface')
  }
  failures.push(...validateRenderedDomEvidence(result))
  failures.push(...validateReasoningEvidence(result, result.reasoningExpectation || 'optional'))
  failures.push(...validateExactToolLoopEvidence(result))
  failures.push(...validateGenerationDefaultsEvidence(result))
  failures.push(...validateServerCacheEvidence(result))
  failures.push(...validateUiRuntimeProvenance(result))
  failures.push(...validateModelBundleBinding(result))
  failures.push(...validatePairedApiEvidence(result))
  if (result.status === 'pass' && !isServerRequestCorrelationVerified(result)) {
    failures.push('top-level pass was claimed without request/cache/settings correlation')
  }
  if (failures.length) {
    const error = new Error(`Real UI live-model proof failed:\n- ${failures.join('\n- ')}`)
    error.failures = failures
    throw error
  }
}

export function deriveProvenSurfaces(result) {
  const surfaces = new Set()
  const chat = result.chat || {}
  const health = result.server?.health || {}
  if (
    result?.ownedRunIntent
    && validateGatewaySingleModelEvidence(result).length === 0
  ) {
    surfaces.add('gateway_single_model_visible_ui')
  }
  if (
    result?.ownedRunIntent?.phase_index === 5
    && result?.releaseEvidence
    && validateFinalPhaseStopEvidence(result).length === 0
  ) {
    surfaces.add('final_proof_session_visible_stop')
  }
  if (
    result.uiLaunchMode === 'installed-app'
    && validateUiRuntimeProvenance(result).length === 0
  ) {
    surfaces.add('installed_app_ui')
  } else if (
    result.uiLaunchMode === 'electron-dev'
    && validateUiRuntimeProvenance(result).length === 0
  ) {
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
    && !reasoningNumericRunIsSpew(chat)
  ) surfaces.add('language_leak_check')
  surfaces.add('electron_ui')
  if (
    result.requestedServerCacheControls === true
    && isCacheRequestCorrelationVerified(result)
    && validateRequestCorrelatedCacheEvidence(result).length === 0
    && (result.cache?.cacheHitTokens || 0) > 0
    && cacheReconstructionClean(result)
  ) {
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
  if (validateGenerationDefaultsEvidence(result).length === 0) {
    surfaces.add('generation_defaults_visible_ui')
  }
  if (
    isServerRequestCorrelationVerified(result)
    && generationDefaultsAppliedSeen(result)
  ) {
    surfaces.add('generation_defaults_applied')
  }
  if (liveSpeedFloorSeen(result)) {
    surfaces.add('live_speed_floor')
  }
  const expectedToolCalls = expectedUiToolCallCount(result)
  if (
    expectedToolCalls > 0
    && validateExactToolLoopEvidence(result).length === 0
  ) {
    surfaces.add('tool_loop')
    if (expectedToolCalls >= 2) surfaces.add('long_tool_loop')
  }
  if (
    validateReasoningEvidence(result, result.reasoningExpectation || 'optional').length === 0
    && ((result.eventCounts?.reasoningDone || 0) > 0 || (result.persistedReasoningCount || 0) > 0)
  ) {
    surfaces.add('reasoning_display')
  }
  if (validateRenderedDomEvidence(result).length === 0) surfaces.add('rendered_dom')
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
  if (result.media?.audioVerified === true) {
    surfaces.add('audio_where_supported')
  }
  const correlationVerified = isServerRequestCorrelationVerified(result)
  if (
    result.pairedApiArtifact
    && validatePairedApiEvidence({
      ...result,
      surfaceStatus: correlationVerified
        ? 'dual_surface_attested'
        : 'partial_dual_surface_uncorrelated',
    }).length === 0
  ) {
    surfaces.add('separate_raw_api')
    if (correlationVerified) surfaces.add('dual_surface')
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
  return validateGenerationDefaultsEvidence(result).length === 0
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

export function parseResolvedSamplingKwargs(lines) {
  const records = []
  for (const line of Array.isArray(lines) ? lines : []) {
    const text = String(line)
    const marker = text.indexOf('Resolved sampling kwargs route=')
    const kwargsMarker = text.indexOf(' kwargs=', marker)
    if (marker < 0 || kwargsMarker < 0) continue
    const routeModel = text.slice(marker, kwargsMarker)
    const routeModelMatch = routeModel.match(
      /^Resolved sampling kwargs route=(\S+)\s+model=(.*?)(?:\s+proof_request_id=(\S+)\s+request_id=(\S+)\s+message_id=(\S+))?$/,
    )
    if (!routeModelMatch) continue
    const raw = text.slice(kwargsMarker + ' kwargs='.length).trim()
    const values = {}
    for (const key of [
      'temperature',
      'top_p',
      'top_k',
      'min_p',
      'repetition_penalty',
      'max_tokens',
      'enable_thinking',
    ]) {
      const match = raw.match(new RegExp(`['"]${key}['"]\\s*:\\s*([^,}]+)`))
      if (!match) continue
      const token = match[1].trim().replace(/^['"]|['"]$/g, '')
      if (/^(?:true|false)$/i.test(token)) values[key] = /^true$/i.test(token)
      else if (/^(?:none|null)$/i.test(token)) values[key] = null
      else {
        const number = Number(token)
        values[key] = Number.isFinite(number) ? number : token
      }
    }
    const record = {
      route_model: routeModel,
      route: routeModelMatch[1],
      model: routeModelMatch[2],
      raw,
      values,
    }
    if (routeModelMatch[3]) {
      record.proof_request_id = routeModelMatch[3]
      record.request_id = routeModelMatch[4]
      record.message_id = routeModelMatch[5]
      record.correlation_source = 'server_emitted'
    }
    records.push(record)
  }
  return records
}

function countMatches(text, regex) {
  return (text.match(regex) || []).length
}

async function main() {
  if (!modelPath) {
    throw new Error('Set VMLINUX_REAL_UI_MODEL_PATH or VMLX_REAL_UI_MODEL_PATH')
  }
  if (requestMaxPromptTokens != null) {
    throw new Error(
      'The real UI proof cannot apply a per-chat max prompt/context override through Chat Settings; '
      + 'remove VMLINUX_REAL_UI_MAX_PROMPT_TOKENS/VMLX_REAL_UI_MAX_PROMPT_TOKENS '
      + 'instead of recording an unapplied setting',
    )
  }
  if ((pairedApiHoldSeconds > 0 || releaseSentinelPath) && !pairedApiArtifactPath) {
    throw new Error(
      'A paired API hold/release sentinel requires VMLINUX_REAL_UI_PAIRED_API_ARTIFACT '
      + 'to name the separate private raw-API artifact that will be written during the hold',
    )
  }
  if (releaseSentinelPath && !releaseSentinelNonce) {
    throw new Error(
      'VMLINUX_REAL_UI_RELEASE_SENTINEL requires a nonempty VMLINUX_REAL_UI_NONCE',
    )
  }
  const attachLifecycleFailures = validateAttachOnlyLifecycle({
    cdpUrl: attachCdpUrl,
    electronPid: expectedElectronPid,
    owner: lifecycleOwner,
    teardownAllowed: allowTeardown,
  })
  if (
    releaseSentinelPath
    && (
      !releaseRunIntentPath
      || !validSha256(releaseRunIntentSha256)
      || !Number.isInteger(releaseActivePhaseIndex)
      || releaseActivePhaseIndex < 0
      || releaseActivePhaseIndex >= ownedRunIntentPhaseContract.length
      || !releaseSessionAttestationPath
      || !Number.isInteger(releaseGatewayPid)
      || releaseGatewayPid <= 0
      || releaseGatewayPid !== expectedElectronPid
      || !releaseGatewayBaseUrl
      || !pairedCacheArtifactPath
      || (
        releaseActivePhaseIndex === 5
        && (
          releaseRetainedPids.length === 0
          || releaseRetainedPids.includes(expectedElectronPid)
          || releaseRetainedPids.includes(releaseGatewayPid)
        )
      )
      || attachLifecycleFailures.length > 0
    )
  ) {
    throw new Error(
      'An orchestrated release sentinel requires '
      + 'VMLINUX_REAL_UI_RUN_INTENT_PATH, VMLINUX_REAL_UI_RUN_INTENT_SHA256, '
      + 'VMLINUX_REAL_UI_ACTIVE_PHASE_INDEX, VMLINUX_REAL_UI_SESSION_ATTESTATION_PATH, '
      + 'VMLINUX_REAL_UI_GATEWAY_PID, VMLINUX_REAL_UI_GATEWAY_BASE_URL, '
      + 'VMLINUX_REAL_UI_PAIRED_CACHE_ARTIFACT, '
      + 'VMLINUX_REAL_UI_ATTACH_CDP_URL, VMLINUX_REAL_UI_EXPECTED_ELECTRON_PID, '
      + 'VMLINUX_REAL_UI_RETAINED_PIDS for phase 5, '
      + 'matching Electron/gateway PIDs with disjoint retained PIDs, '
      + 'VMLINUX_REAL_UI_LIFECYCLE_OWNER=parent, and VMLINUX_REAL_UI_ALLOW_TEARDOWN=0'
      + (
        attachLifecycleFailures.length
          ? `: ${attachLifecycleFailures.join('; ')}`
          : ''
      ),
    )
  }
  if (attachCdpUrl && !releaseSentinelPath) {
    throw new Error('Attach-only Electron mode is reserved for an orchestrated release run')
  }
  if (Boolean(reuseSessionId) !== Boolean(reuseSessionAttestationPath)) {
    throw new Error(
      'Owned UI session reuse requires both the exact session ID and prior '
      + 'session-attestation path',
    )
  }
  if (
    reuseSessionId
    && (
      !releaseSentinelPath
      || !attachCdpUrl
      || lifecycleOwner !== 'parent'
      || allowTeardown
    )
  ) {
    throw new Error(
      'Owned UI session reuse is restricted to attach-only parent lifecycle phases',
    )
  }
  if (
    builtinToolsEnabled
    && (
      !Number.isInteger(toolResultMaxChars)
      || toolResultMaxChars < 500
      || toolResultMaxChars > 50_000
      || toolResultMaxChars % 500 !== 0
    )
  ) {
    throw new Error(
      'Tool Result Limit must be an exact visible Chat Settings slider value '
      + '(integer 500..50000 in steps of 500)',
    )
  }
  const privateCacheAttestationArgs = privateCacheAttestationSessionArgs(
    privateCacheAttestationTokenFile,
  )
  const proofDir = await resolvePrivateProofDir()
  const gitBefore = await captureGitProvenance()
  const bundleGenerationContract = captureBundleGenerationContract(modelPath)
  const releaseManifest = installedAppPath
    ? readExternalReleaseManifest(installedReleaseManifestPath)
    : null
  if (installedAppPath && !releaseManifest) {
    throw new Error(
      'Installed-app proof requires VMLINUX_REAL_UI_RELEASE_MANIFEST pointing to an external release manifest',
    )
  }
  const userDataDir = mkdtempSync(path.join(tmpdir(), 'vmlx-real-ui-userdata-'))
  const configuredWorkingDirectory = process.env.VMLINUX_REAL_UI_WORKING_DIRECTORY
    || process.env.VMLX_REAL_UI_WORKING_DIRECTORY
    || ''
  const workingDirectory = configuredWorkingDirectory
    || mkdtempSync(path.join(tmpdir(), 'vmlx-real-ui-tools-'))
  mkdirSync(workingDirectory, { recursive: true })
  for (const name of ['real_ui_tool_probe_1.txt', 'real_ui_tool_probe_2.txt']) {
    rmSync(path.join(workingDirectory, name), { force: true })
  }

  const serverPort = Number(
    process.env.VMLINUX_REAL_UI_SERVER_PORT
    || process.env.VMLX_REAL_UI_SERVER_PORT
    || await freePort(),
  )
  const attachCdp = attachCdpUrl ? strictHttpUrl(attachCdpUrl) : null
  if (
    attachCdpUrl
    && (
      !attachCdp
      || !['127.0.0.1', 'localhost', '[::1]'].includes(attachCdp.hostname)
      || !attachCdp.port
      || attachCdp.pathname !== '/'
      || attachCdpUrl !== attachCdp.origin
    )
  ) {
    throw new Error('VMLINUX_REAL_UI_ATTACH_CDP_URL must be an exact loopback HTTP(S) origin')
  }
  const debugPort = attachCdp
    ? Number(attachCdp.port)
    : requestedCdpPort
      ?? await freePortExcluding(new Set([serverPort]))
  const gatewayPort = attachCdp
    ? null
    : requestedGatewayPort
      ?? await freePortExcluding(new Set([serverPort, debugPort]))
  if (
    !attachCdp
    && (
      gatewayPort === serverPort
      || gatewayPort === debugPort
      || debugPort === serverPort
    )
  ) {
    throw new Error('Real UI backend, CDP, and gateway ports must be distinct')
  }
  const baseUrl = `http://127.0.0.1:${serverPort}`
  let ownedRunIntent = null
  let activeReleasePhase = null
  let reuseSessionAttestation = null
  if (releaseSentinelPath) {
    ownedRunIntent = readPrivateExternalJson(
      releaseRunIntentPath,
      'Owned UI run intent',
    )
    const runIntentFailures = validateOwnedRunIntent(ownedRunIntent, {
      runId,
      nonce: releaseSentinelNonce,
      expectedSha256: releaseRunIntentSha256,
      expectedSourceCommit: gitBefore.commit,
      expectedSourceTree: gitBefore.tree,
      expectedUiHarnessSha256: sha256File(fileURLToPath(import.meta.url)),
      activePhaseIndex: releaseActivePhaseIndex,
      activeModel: servedModel,
      activeModelBundlePath: modelPath,
      expectedDirectBaseUrl: baseUrl,
      expectedGatewayBaseUrl: releaseGatewayBaseUrl,
    })
    if (runIntentFailures.length) {
      throw new Error(runIntentFailures.join('; '))
    }
    activeReleasePhase = ownedRunIntent.value.phase_plan[releaseActivePhaseIndex]
    const reusePhase = [1, 2, 3, 4].includes(
      activeReleasePhase.phase_index,
    )
    if (reusePhase !== Boolean(reuseSessionId)) {
      throw new Error(
        'Owned UI session reuse is required only for release phases 1-4',
      )
    }
    if (reusePhase) {
      reuseSessionAttestation = readPrivateExternalJson(
        reuseSessionAttestationPath,
        'Owned prior UI session attestation',
      )
      const failures = validateOwnedReuseSessionAttestation(
        reuseSessionAttestation,
        {
          runId,
          nonce: releaseSentinelNonce,
          runIntentSha256: releaseRunIntentSha256,
          sessionId: reuseSessionId,
          activePhase: activeReleasePhase,
          model: servedModel,
          modelBundlePath: modelPath,
          electronPid: expectedElectronPid,
          cdpOrigin: attachCdp.origin,
          gatewayPid: releaseGatewayPid,
          gatewayBaseUrl: releaseGatewayBaseUrl,
          sourceCommit: gitBefore.commit,
          sourceTree: gitBefore.tree,
        },
      )
      if (failures.length) throw new Error(failures.join('; '))
    }
  }
  const uiActionProfile = activeReleasePhase?.ui_action_profile
    || 'legacy-three-turn'
  const uiTurnCount = Number(activeReleasePhase?.ui_turn_count || 3)
  const apiActionProfile = activeReleasePhase?.api_action_profile
    || 'full-agentic'
  // Keep the release proof's paged-RAM tier deterministically below its fixed
  // 10 GiB SSD tier. The cache gate must first evict the recent target from L1 while its
  // exact chain is still readable in L2, then perform a real disk refault.
  // MiniMax M2.7's model-safe q8 cache uses enough bytes per block that the
  // normal 15% session default, and prior percentage overrides once
  // earlier phase entries were present, let bounded L2 evict the recent target
  // before its terminal L1 block was gone. A fixed 4096 MiB cap gives this
  // proof topology a deterministic margin below a fixed 10 GiB L2. The cache
  // gate independently attests that live margin and still fails closed.
  // This is proof-session configuration only; product defaults and
  // user-created sessions remain unchanged.
  const releasePagedCacheMemoryMb = activeReleasePhase?.paged_ram
    ? 4096
    : null
  const releaseBlockDiskCacheMaxGb = activeReleasePhase ? 10 : null
  const releaseBlockDiskCacheAnchorPhase = (() => {
    if (!activeReleasePhase) return null
    if (activeReleasePhase.operation !== 'probe') return activeReleasePhase
    const plan = Array.isArray(ownedRunIntent?.value?.phase_plan)
      ? ownedRunIntent.value.phase_plan
      : []
    const prior = [...plan]
      .filter((candidate) => (
        candidate
        && candidate.phase_index < activeReleasePhase.phase_index
        && candidate.representative_id === activeReleasePhase.representative_id
        && candidate.cache_policy === activeReleasePhase.cache_policy
        && candidate.kv_cache_quantization === activeReleasePhase.kv_cache_quantization
        && candidate.paged_ram === activeReleasePhase.paged_ram
        && candidate.operation !== 'probe'
      ))
      .sort((left, right) => right.phase_index - left.phase_index)[0]
    return prior || activeReleasePhase
  })()
  const releaseBlockDiskCacheDir = activeReleasePhase
    ? path.join(
      path.dirname(proofDir),
      'ui-shared-block-disk-cache',
      [
        String(releaseBlockDiskCacheAnchorPhase.phase_index).padStart(2, '0'),
        safeArtifactComponent(releaseBlockDiskCacheAnchorPhase.phase_name, 'phase'),
        safeArtifactComponent(releaseBlockDiskCacheAnchorPhase.representative_id, 'representative'),
      ].join('-'),
    )
    : ''
  if (releaseBlockDiskCacheDir) {
    mkdirSync(releaseBlockDiskCacheDir, { recursive: true })
    chmodSync(path.dirname(releaseBlockDiskCacheDir), 0o700)
    chmodSync(releaseBlockDiskCacheDir, 0o700)
  }
  const profilePromptOne = {
    'primary-reasoning-render-store': [
      releasePrimarySharedPrefix,
      'Do not call tools.',
      'Privately compare 47 times 19 with 46 times 20.',
      'Reply exactly two lines: R19-PRIMARY-STORE-DONE and',
      'The literal currency string is $43 and $47 \\times 19 = 893 < 920 = 46 \\times 20$.',
    ].join(' '),
    'primary-tool-restart-probe': [
      releasePrimarySharedPrefix,
      'Call the built-in run_command tool exactly once with this exact command:',
      'printf %s REAL_UI_LIVE_TOOL_ONE > real_ui_tool_probe_1.txt && cat real_ui_tool_probe_1.txt',
      'After the tool result, include REAL_UI_LIVE_TOOL_ONE in the visible answer.',
    ].join(' '),
    'primary-history-paged-evict-refault': [
      releasePrimarySharedPrefix,
      'Do not call tools.',
      'Reply exactly R19-PRIMARY-EVICT-REFAULT-DONE.',
    ].join(' '),
    'primary-restart-followup': [
      releasePrimarySharedPrefix,
      'Do not call tools.',
      'Reply exactly R19-PRIMARY-RESTART-FOLLOWUP-DONE.',
    ].join(' '),
    'primary-tq-off-probe': [
      releasePrimarySharedPrefix,
      'Do not call tools.',
      'Reply exactly R19-PRIMARY-TQ-OFF-DONE.',
    ].join(' '),
  }[uiActionProfile]
  const selectedPromptOne = promptOneOverride || profilePromptOne || promptOne
  const selectedPromptTwo = promptTwo
  const selectedPromptThree = promptThree
  let app
  let cdp
  let appLogs = []
  let serverModels = {}
  let healthBefore = {}
  let rendererResourceEvidence = {}
  let cdpProcessBinding = null
  let gatewayProcessBinding = null
  let backendProcessBinding = null
  let proofGatewayStatus = null
  try {
    app = attachCdp
      ? {
          proc: null,
          logs: [],
          uiLaunchMode: 'electron-dev',
          command: [],
          appPath: '',
          attached: true,
          lifecycleOwner,
          allowTeardown,
          cdpUrl: attachCdp.origin,
          expectedElectronPid,
        }
      : startUiApp(userDataDir, debugPort, gatewayPort)
    appLogs = app.logs

    const target = await waitForTarget(debugPort, appLogs)
    cdpProcessBinding = await captureListenerProcessBinding({
      port: debugPort,
      expectedRootPid: app.attached ? expectedElectronPid : app.proc.pid,
      expectedHealthPid: null,
      kind: 'electron-cdp',
    })
    if (
      releaseSentinelPath
      && (
        releaseGatewayPid !== expectedElectronPid
        || cdpProcessBinding.listener_pid !== expectedElectronPid
      )
    ) {
      throw new Error(
        'Owned release proof requires the gateway PID, expected Electron PID, '
        + 'and CDP-bound Electron PID to be identical',
      )
    }
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
    if (!app.attached) {
      proofGatewayStatus = await evaluate(cdp, `
        new Promise((resolve, reject) => {
          const started = Date.now();
          const check = async () => {
            try {
              const status = await window.api.gateway.getStatus();
              if (
                status?.running === true
                && Number(status.port) === ${JSON.stringify(gatewayPort)}
                && status.host === '127.0.0.1'
              ) {
                resolve(status);
                return;
              }
              if (
                status?.running === true
                && Number(status.port) !== ${JSON.stringify(gatewayPort)}
              ) {
                reject(new Error(
                  'Proof-owned gateway shifted from requested port '
                  + ${JSON.stringify(gatewayPort)}
                  + ' to '
                  + String(status.port)
                ));
                return;
              }
            } catch (_) {}
            if (Date.now() - started > 30000) {
              reject(new Error('Timed out waiting for the proof-owned gateway listener'));
              return;
            }
            setTimeout(check, 100);
          };
          check();
        })
      `)
      gatewayProcessBinding = await captureListenerProcessBinding({
        port: gatewayPort,
        expectedRootPid: app.proc.pid,
        expectedHealthPid: null,
        kind: 'electron-gateway',
      })
    }
    rendererResourceEvidence = await evaluate(cdp, `({
      pageUrl: location.href,
      scripts: [...document.scripts].map((script) => script.src).filter(Boolean),
      resources: performance.getEntriesByType('resource').map((entry) => entry.name),
      buildSourceCommit: globalThis.__VMLINUX_SOURCE_COMMIT__
        || document.documentElement.dataset.sourceCommit
        || '',
    })`)
    if (app.uiLaunchMode === 'electron-dev') {
      const servedModuleRequests = rendererProofModulePaths.map((relativePath) => ({
        relativePath,
        requestPath: viteRawRendererModulePath(relativePath, runId),
      }))
      const servedModules = await evaluate(cdp, `
        (async () => {
          const modules = ${JSON.stringify(servedModuleRequests)};
          const records = [];
          for (const record of modules) {
            const module = await import(record.requestPath);
            if (typeof module.default !== 'string') {
              throw new Error('Vite raw renderer module did not return source text: ' + record.relativePath);
            }
            records.push({
              relative_path: record.relativePath,
              source_text: module.default,
            });
          }
          return records;
        })()
      `)
      rendererResourceEvidence.servedModules = servedModules.map((record) => ({
        relative_path: record.relative_path,
        size_bytes: Buffer.byteLength(record.source_text),
        sha256: sha256Text(record.source_text),
      }))
      rendererResourceEvidence.servedRendererSourceSha256 = canonicalSha256(
        rendererResourceEvidence.servedModules,
      )
    }

    let rendererResult
    try {
      rendererResult = await evaluate(cdp, `
      (async () => {
        const baseUrl = ${JSON.stringify(baseUrl)};
        const modelPath = ${JSON.stringify(modelPath)};
        const servedModel = ${JSON.stringify(servedModel)};
        const wireApi = ${JSON.stringify(wireApi)};
        const uiActionProfile = ${JSON.stringify(uiActionProfile)};
        const uiTurnCount = ${JSON.stringify(uiTurnCount)};
          const apiActionProfile = ${JSON.stringify(apiActionProfile)};
          const builtinToolsEnabled = ${JSON.stringify(builtinToolsEnabled)};
          const enableThinking = ${enableThinkingOverride === undefined ? 'undefined' : JSON.stringify(enableThinkingOverride)};
          const requestedReasoningEffort = ${JSON.stringify(reasoningEffortOverride)};
          const requestedMidConvReasoningEffort = ${JSON.stringify(midConvReasoningEffortOverride)};
          const checkMedia = ${JSON.stringify(checkMedia)};
        const checkVideo = ${JSON.stringify(checkVideo)};
        const imageDataUrl = ${JSON.stringify(imageDataUrl)};
        const imageExpectRegex = ${JSON.stringify(imageExpectRegex)};
        const videoDataUrl = ${JSON.stringify(videoDataUrl)};
        const videoExpectRegex = ${JSON.stringify(videoExpectRegex)};
        const checkAudio = ${JSON.stringify(checkAudio)};
        const audioDataUrl = ${JSON.stringify(audioDataUrl)};
        const audioExpectRegex = ${JSON.stringify(audioExpectRegex)};
        const workingDirectory = ${JSON.stringify(workingDirectory)};
        const samplingOverrides = ${JSON.stringify(samplingOverrides)};
        const independentBundleDefaults = ${JSON.stringify(bundleGenerationContract.defaults)};
        const endpoint = { host: '127.0.0.1', port: ${JSON.stringify(serverPort)} };
        const l2DiskStorageSeen = ${l2DiskStorageSeen.toString()};
        const correlateTerminalResponseToCacheExecution =
          ${correlateTerminalResponseToCacheExecution.toString()};
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
        const processedRequestCount = (snapshot) =>
          Number(snapshot?.scheduler_stats?.num_requests_processed || 0);
        const waitForRequestCacheAdvance = async (initial, sessionId) => {
          if (!${JSON.stringify(checkServerCacheControls)}) return initial;
          const before = processedRequestCount(initial);
          let latest = initial;
          const started = Date.now();
          while (Date.now() - started < 15000) {
            latest = await window.api.cache.stats(endpoint, sessionId)
              .catch((error) => ({ error: String(error?.message || error) }));
            if (processedRequestCount(latest) > before) return latest;
            await new Promise((resolve) => setTimeout(resolve, 100));
          }
          return latest;
        };
        const waitForResolvedTurnLog = async (
          sessionId,
          { proofRequestId, messageId, requestIds },
        ) => {
          const expectedRoute = wireApi === 'responses'
            ? '/v1/responses'
            : '/v1/chat/completions';
          const expectedMarker = 'Resolved sampling kwargs route='
            + expectedRoute
            + ' model=';
          const expectedRequestIds = [...new Set(
            (Array.isArray(requestIds) ? requestIds : [])
              .map((value) => String(value || ''))
              .filter(Boolean),
          )];
          if (!proofRequestId || !messageId || expectedRequestIds.length === 0) {
            return { logs: [], matchedLines: [] };
          }
          let latest = [];
          const started = Date.now();
          while (Date.now() - started < 10000) {
            latest = await window.api.sessions.getLogs(sessionId).catch(() => []);
            const matchedLines = latest.filter((line) => {
              const text = String(line);
              return (
                text.includes(expectedMarker)
                && text.includes(' proof_request_id=' + proofRequestId + ' ')
                && text.includes(' message_id=' + messageId + ' kwargs=')
                && expectedRequestIds.some((requestId) =>
                  text.includes(' request_id=' + requestId + ' ')
                )
              );
            });
            const matchedRequestIds = new Set(matchedLines.map((line) => {
              const match = String(line).match(/\\srequest_id=(\\S+)\\smessage_id=/);
              return match?.[1] || '';
            }).filter(Boolean));
            if (
              matchedLines.length === expectedRequestIds.length
              && expectedRequestIds.every((requestId) => matchedRequestIds.has(requestId))
            ) {
              return { logs: latest, matchedLines };
            }
            await new Promise((resolve) => setTimeout(resolve, 100));
          }
          return {
            logs: latest,
            matchedLines: latest.filter((line) => {
              const text = String(line);
              return (
                text.includes(expectedMarker)
                && text.includes(' proof_request_id=' + proofRequestId + ' ')
                && text.includes(' message_id=' + messageId + ' kwargs=')
              );
            }),
          };
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
        const waitFor = (predicate, label, timeoutMs = 30000) => new Promise((resolve, reject) => {
          const started = Date.now();
          const check = () => {
            try {
              const value = predicate();
              if (value) return resolve(value);
            } catch (_) {}
            if (Date.now() - started > timeoutMs) {
              return reject(new Error(
                'Timed out waiting for ' + label + ': ' + document.body.innerText.slice(0, 4000)
              ));
            }
            setTimeout(check, 100);
          };
          check();
        });
        const isVisible = (element) => {
          if (!(element instanceof HTMLElement)) return false;
          const style = getComputedStyle(element);
          const rect = element.getBoundingClientRect();
          return style.display !== 'none'
            && style.visibility !== 'hidden'
            && Number(style.opacity || 1) !== 0
            && rect.width > 0
            && rect.height > 0;
        };
        const domSamples = [];
        const domSampleState = new Map();
        const maxDomSamplesPerMessage = 24;
        const minDomSampleIntervalMs = 125;
        const upsertBoundedDomSample = ${upsertBoundedDomSample.toString()};
        const transientAlerts = [];
        const observedAlertText = new Set();
        const captureAlerts = () => {
          for (const alert of document.querySelectorAll('[role="alert"]')) {
            if (!isVisible(alert)) continue;
            const text = (alert.textContent || '').replace(/\\s+/g, ' ').trim();
            const tone = alert.getAttribute('data-vmlx-tone');
            const isError = tone === 'error'
              || (!tone && String(alert.className || '').includes('destructive'))
              || /(?:failed|error|exception|traceback|invalid)/i.test(text);
            if (isError && text && !observedAlertText.has(text)) {
              observedAlertText.add(text);
              transientAlerts.push(text);
            }
          }
        };
        const alertObserver = new MutationObserver(captureAlerts);
        alertObserver.observe(document.body, { childList: true, subtree: true, characterData: true });
        const snapshotMessage = (messageId, cause) => {
          const escaped = CSS.escape(String(messageId || ''));
          const root = document.querySelector(
            '[data-vmlx-proof-message-id="' + escaped + '"]'
          );
          if (!root) return null;
          const answer = root.querySelector('[data-vmlx-proof-answer="true"]');
          const reasoningNodes = [...root.querySelectorAll(
            '[data-vmlx-proof-reasoning-content="true"]'
          )];
          const readMountedInnerText = (element) => {
            if (!element) return '';
            const probe = document.createElement('div');
            probe.setAttribute('aria-hidden', 'true');
            probe.style.cssText = [
              'position:fixed',
              'left:-100000px',
              'top:0',
              'width:1024px',
              'opacity:0',
              'pointer-events:none',
              'z-index:-2147483648',
            ].join(';');
            probe.appendChild(element);
            document.body.appendChild(probe);
            try {
              return element.innerText.trim();
            } finally {
              probe.remove();
            }
          };
          const reasoningEvidence = reasoningNodes.map((node) => {
            const linkedClone = node.cloneNode(true);
            const mathSources = [];
            const mathDisplayModes = [];
            linkedClone.querySelectorAll('[data-vmlx-math-source-codepoints]').forEach((mathNode) => {
              const encodedSource = mathNode.getAttribute(
                'data-vmlx-math-source-codepoints'
              ) || '';
              let source = '';
              try {
                if (!/^[0-9a-f]+(?:-[0-9a-f]+)*$/i.test(encodedSource)) {
                  throw new Error('invalid encoded math source');
                }
                source = encodedSource
                  .split('-')
                  .map((value) => String.fromCodePoint(Number.parseInt(value, 16)))
                  .join('');
              } catch (_) {
                source = '__INVALID_VMLX_MATH_SOURCE__';
              }
              const delimiter = mathNode.getAttribute(
                'data-vmlx-math-delimiter'
              ) || '';
              const displayMode = mathNode.getAttribute(
                'data-vmlx-math-display-mode'
              ) || '';
              const wrapperDisplayMode = mathNode.classList.contains('math-block')
                ? 'display'
                : mathNode.classList.contains('math-inline') ? 'inline' : 'unknown';
              const katexDisplayMode = mathNode.querySelector('.katex-display')
                ? 'display'
                : mathNode.querySelector('.katex') ? 'inline' : 'missing';
              const markerIndex = mathSources.push({
                source,
                delimiter,
                displayMode,
              }) - 1;
              mathDisplayModes.push({ wrapperDisplayMode, katexDisplayMode });
              mathNode.replaceWith(document.createTextNode('VMLXPROOFMATH' + markerIndex));
            });
            return {
              renderedText: (node.innerText || node.textContent || '').trim(),
              linkedText: readMountedInnerText(linkedClone),
              mathSources,
              mathDisplayModes,
              katexCount: node.querySelectorAll('.katex').length,
              katexErrorCount: node.querySelectorAll('.katex-error').length,
            };
          }).filter((segment) => segment.renderedText);
          const reasoningSegments = reasoningEvidence.map((segment) => segment.renderedText);
          const toolCards = [...root.querySelectorAll('[data-vmlx-proof-tool-card]')].map((card) => ({
            kind: card.getAttribute('data-vmlx-proof-tool-card') || '',
            name: card.getAttribute('data-vmlx-proof-tool-name') || '',
            callId: card.getAttribute('data-vmlx-proof-tool-call-id') || '',
            phase: card.getAttribute('data-vmlx-proof-tool-phase') || '',
            visible: isVisible(card),
            text: (card.textContent || '').replace(/\\s+/g, ' ').trim(),
          }));
          const currencyOccurrences = [];
          if (answer) {
            const walker = document.createTreeWalker(answer, NodeFilter.SHOW_TEXT);
            let node;
            while ((node = walker.nextNode())) {
              const text = node.nodeValue || '';
              let offset = text.indexOf('$43');
              while (offset >= 0) {
                currencyOccurrences.push({
                  text: '$43',
                  insideKatex: Boolean(node.parentElement?.closest('.katex')),
                });
                offset = text.indexOf('$43', offset + 3);
              }
            }
          }
          const proseAnswer = answer?.cloneNode(true);
          proseAnswer?.querySelectorAll(
            '[data-vmlx-proof-tool-card], [data-vmlx-proof-tool-container], .code-header'
          ).forEach((element) => element.remove());
          // The code-block language/copy header is renderer chrome, not model
          // output.  Keep the rendered <code> body in the persisted-content
          // linkage check while excluding that UI-owned label and button.
          // innerText only preserves rendered separators such as <br> and
          // block boundaries while a node participates in layout. The
          // scrubbed clone is detached, so falling back to textContent would
          // silently join visible lines (for example, "DONE<br>The" becomes
          // "DONEThe") and make byte-identical persisted content look
          // unlinked. Mount the clone offscreen for the synchronous read, then
          // always remove it. Opacity keeps it invisible without disabling
          // layout, and tool-card text remains excluded from the proof.
          let proseAnswerText = '';
          if (proseAnswer) {
            proseAnswerText = readMountedInnerText(proseAnswer);
          }
          return {
            cause,
            t: performance.now(),
            messageId: String(messageId || ''),
            role: root.getAttribute('data-vmlx-proof-message-role') || '',
            visible: isVisible(root),
            answerText: proseAnswerText,
            answerState: answer?.getAttribute('data-vmlx-proof-answer-state') || '',
            answerFullLength: Number(
              answer?.getAttribute('data-vmlx-proof-answer-full-length') || -1
            ),
            answerRenderedLength: Number(
              answer?.getAttribute('data-vmlx-proof-answer-rendered-length') || -1
            ),
            reasoningText: reasoningSegments.join('\\n'),
            reasoningSegments,
            reasoningLinkedSegments: reasoningEvidence.map((segment) => segment.linkedText),
            reasoningMathSources: reasoningEvidence.map((segment) => segment.mathSources),
            reasoningMathDisplayModes: reasoningEvidence.map(
              (segment) => segment.mathDisplayModes
            ),
            reasoningKatexCounts: reasoningEvidence.map((segment) => segment.katexCount),
            reasoningKatexErrorCounts: reasoningEvidence.map(
              (segment) => segment.katexErrorCount
            ),
            html: answer?.innerHTML || '',
            katexCount: answer?.querySelectorAll('.katex').length || 0,
            katexErrorCount: answer?.querySelectorAll('.katex-error').length || 0,
            katexAnnotations: [...(answer?.querySelectorAll(
              '.katex annotation[encoding="application/x-tex"]'
            ) || [])].map((annotation) => annotation.textContent || ''),
            currencyOccurrences,
            toolCards,
          };
        };
        const scheduleDomSample = (messageId, cause, force = false) => {
          const key = String(messageId || 'unknown');
          const now = performance.now();
          const state = domSampleState.get(key) || {
            count: 0,
            pending: false,
            forcePending: false,
            lastScheduledAt: -Infinity,
            lastSignature: '',
            lastStoredIndex: -1,
          };
          if (state.pending) {
            if (force) state.forcePending = true;
            domSampleState.set(key, state);
            return;
          }
          if (
            !force
            && (
              state.count >= maxDomSamplesPerMessage
              || now - state.lastScheduledAt < minDomSampleIntervalMs
            )
          ) return;
          state.pending = true;
          state.lastScheduledAt = now;
          domSampleState.set(key, state);
          requestAnimationFrame(() => requestAnimationFrame(() => {
            state.pending = false;
            const captureAsFinal = force || state.forcePending;
            state.forcePending = false;
            const sample = snapshotMessage(messageId, cause);
            if (sample) {
              upsertBoundedDomSample(
                domSamples,
                state,
                sample,
                maxDomSamplesPerMessage,
                captureAsFinal,
              );
            }
            captureAlerts();
          }));
        };
        await window.api.engine.checkInstallation().catch(() => null);
        await window.api.chat.clearAllLocks().catch(() => null);
        const events = { stream: [], tool: [], reasoningDone: [], complete: [] };
        const eventCounts = { stream: 0, tool: 0, reasoningDone: 0, complete: 0 };
        const streamTraceState = new Map();
        const eventTrace = [];
        const priorFullContent = new Map();
        const reasoningSegmentIndex = new Map();
        let eventSequence = 0;
        const recordEvent = (bucket, event, data) => {
          const captured = { t: performance.now(), ...data };
          eventCounts[bucket] += 1;
          const messageId = data?.messageId || 'unknown';
          const channel = event === 'stream'
            ? (data?.isReasoning ? 'reasoning' : 'content')
            : event === 'tool'
              ? 'tool'
              : event === 'reasoning_terminal'
                ? 'reasoning'
                : 'terminal';
          let delta = null;
          let cumulativeReset = false;
          let tracePayload = data;
          const segmentIndex = channel === 'reasoning'
            ? Number(reasoningSegmentIndex.get(messageId) || 0)
            : null;
          if (event === 'stream' && typeof data?.fullContent === 'string') {
            const reasoningSegments = (
              channel === 'reasoning' && Array.isArray(data?.reasoningSegments)
            )
              ? data.reasoningSegments.map((value) => String(value || ''))
              : null;
            const cumulativeContent = (
              reasoningSegments
              && Number.isInteger(segmentIndex)
              && segmentIndex >= 0
            )
              ? String(reasoningSegments[segmentIndex] || '')
              : data.fullContent;
            const key = messageId + ':' + channel
              + (channel === 'reasoning' ? ':' + segmentIndex : '');
            const previous = priorFullContent.get(key) || '';
            if (cumulativeContent.startsWith(previous)) {
              delta = cumulativeContent.slice(previous.length);
            } else {
              cumulativeReset = previous.length > 0;
              delta = cumulativeContent;
            }
            priorFullContent.set(key, cumulativeContent);
            tracePayload = {
              messageId: data?.messageId || null,
              isReasoning: data?.isReasoning === true,
              metrics: data?.metrics || null,
              fullContentLength: data.fullContent.length,
              reasoningSegments,
              reasoningSegmentLength:
                reasoningSegments && Number.isInteger(segmentIndex)
                  ? String(reasoningSegments[segmentIndex] || '').length
                  : null,
            };
            const summary = streamTraceState.get(messageId) || {
              messageId,
              count: 0,
              firstFullContent: '',
              lastFullContent: '',
              firstReasoningContent: '',
              lastReasoningContent: '',
              firstMetrics: null,
              lastMetrics: null,
            };
            summary.count += 1;
            if (data.fullContent && !summary.firstFullContent) {
              summary.firstFullContent = data.fullContent;
            }
            if (data.fullContent) summary.lastFullContent = data.fullContent;
            if (data.isReasoning && data.fullContent && !summary.firstReasoningContent) {
              summary.firstReasoningContent = data.fullContent;
            }
            if (data.isReasoning && data.fullContent) {
              summary.lastReasoningContent = data.fullContent;
            }
            if (data?.metrics && !summary.firstMetrics) summary.firstMetrics = data.metrics;
            if (data?.metrics) summary.lastMetrics = data.metrics;
            streamTraceState.set(messageId, summary);
          } else {
            events[bucket].push(captured);
          }
          eventTrace.push({
            sequence: ++eventSequence,
            t: captured.t,
            event,
            channel,
            messageId,
            delta,
            cumulativeReset,
            segmentIndex,
            payload: tracePayload,
          });
          if (event === 'reasoning_terminal') {
            reasoningSegmentIndex.set(messageId, segmentIndex + 1);
          }
          scheduleDomSample(messageId, event + ':' + channel, event !== 'stream');
        };
        const cleanup = [
          window.api.chat.onStream((data) => recordEvent('stream', 'stream', data)),
          window.api.chat.onToolStatus((data) => recordEvent('tool', 'tool', data)),
          window.api.chat.onReasoningDone((data) => recordEvent('reasoningDone', 'reasoning_terminal', data)),
          window.api.chat.onComplete((data) => recordEvent('complete', 'terminal', data)),
        ];
        let proofSessionId = null;
        let privateConfigRestoreAdditionalArgs = null;
        const privateCacheAttestationArgs = ${JSON.stringify(privateCacheAttestationArgs)};
        try {
          const stripPrivateCacheAttestationArgs = (raw) => {
            const tokens = String(raw || '').trim().split(/\\s+/).filter(Boolean);
            const kept = [];
            for (let index = 0; index < tokens.length; index += 1) {
              const token = tokens[index];
              if (token === '--enable-private-cache-attestation') continue;
              if (token === '--private-cache-attestation-token-file') {
                index += 1;
                continue;
              }
              if (token.startsWith('--private-cache-attestation-token-file=')) {
                continue;
              }
              kept.push(token);
            }
            return kept.join(' ');
          };
          const requestedSessionConfig = {
            host: '127.0.0.1',
            port: ${JSON.stringify(serverPort)},
            servedModelName: servedModel,
            logLevel: 'INFO',
            ...(${JSON.stringify(activeReleasePhase ? {
              enablePrefixCache: true,
              usePagedCache: Boolean(activeReleasePhase.paged_ram),
              enableBlockDiskCache: true,
              kvCacheQuantization: String(activeReleasePhase.kv_cache_quantization || 'none'),
              blockDiskCacheDir: releaseBlockDiskCacheDir,
              cacheMemoryPercent: 0,
              ...(releasePagedCacheMemoryMb == null
                ? {}
                : { cacheMemoryMb: releasePagedCacheMemoryMb }),
              blockDiskCacheMaxGb: releaseBlockDiskCacheMaxGb,
            } : {})}),
            ...(privateCacheAttestationArgs
              ? { additionalArgs: privateCacheAttestationArgs }
              : {}),
            ...(builtinToolsEnabled
              ? { enableAutoToolChoice: true, toolCallParser: 'auto' }
              : {}),
          };
          const requestedReuseSessionId = ${JSON.stringify(reuseSessionId)};
          let created;
          if (requestedReuseSessionId) {
            const existing = await window.api.sessions.get(
              requestedReuseSessionId,
            );
            const storedModelPath = existing?.modelPath
              || existing?.model_path
              || '';
            if (
              !existing
              || existing.id !== requestedReuseSessionId
              || existing.type !== 'local'
              || storedModelPath !== modelPath
              || existing.status === 'error'
            ) {
              throw new Error(
                'Parent-attested reused session is missing, errored, '
                + 'wrong-type, or bound to a different model path',
              );
            }
            created = { success: true, session: existing };
          } else {
            created = await window.api.sessions.create(
              modelPath,
              requestedSessionConfig,
            );
          }
          if (!created.success) {
            throw new Error(created.error || 'local session create failed');
          }
          proofSessionId = created.session?.id || null;
          if (created.session?.type !== 'local') {
            throw new Error('real UI proof requires an Electron-managed local session');
          }
          window.dispatchEvent(new CustomEvent('vmlx:navigate', {
            detail: {
              mode: 'server',
              panel: 'session',
              sessionId: created.session.id,
            },
          }));
          // The session/panel event stages the exact target, but a Sessions
          // context refresh can restore Chat mode before React commits it. Use
          // the same visible top-bar Server button a user clicks, then wait for
          // the session-bound settings control. The previous event-only path
          // timed out on the Chat quick-start screen without ever loading the
          // model, even though the session itself had been created correctly.
          const serverModeButton = await waitFor(() => {
            return [...document.querySelectorAll('button')].find((button) => (
              button instanceof HTMLButtonElement
              && isVisible(button)
              && (button.textContent || '').replace(/\\s+/g, ' ').trim() === 'Server'
            )) || null;
          }, 'visible top-bar Server mode control');
          serverModeButton.scrollIntoView({ block: 'center' });
          serverModeButton.click();
          let sessionBeforeStart = await window.api.sessions.get(created.session.id);
          let sessionConfigBeforeProof = {};
          try {
            sessionConfigBeforeProof = JSON.parse(sessionBeforeStart?.config || '{}');
          } catch (_) {}
          privateConfigRestoreAdditionalArgs = stripPrivateCacheAttestationArgs(
            sessionConfigBeforeProof.additionalArgs,
          );
          // sessions.create() intentionally deduplicates by model identity and may
          // return an already-active row even for the first phase of a proof run.
          // Stop every active state through the visible UI before requiring the
          // Start control, so the proof still attests a real user-driven restart.
          let preStartStopControl = null;
          if (['running', 'loading', 'standby'].includes(sessionBeforeStart?.status)) {
            const statusBeforeVisibleStop = sessionBeforeStart.status;
            const exactStopSelector =
              '[data-vmlx-control="session-stop"][data-vmlx-session-id="'
              + CSS.escape(created.session.id)
              + '"]';
            const stopButton = await new Promise((resolve, reject) => {
              const started = Date.now();
              const check = () => {
                const candidate = document.querySelector(exactStopSelector);
                if (
                  candidate instanceof HTMLButtonElement
                  && isVisible(candidate)
                  && !candidate.disabled
                ) return resolve(candidate);
                if (Date.now() - started > 30000) {
                  return reject(new Error(
                    'Timed out waiting for the exact session-bound visible '
                    + 'Stop/Cancel control for the active local session',
                  ));
                }
                setTimeout(check, 100);
              };
              check();
            });
            stopButton.scrollIntoView({ block: 'center' });
            stopButton.click();
            sessionBeforeStart = await new Promise((resolve, reject) => {
              const started = Date.now();
              const check = async () => {
                const current = await window.api.sessions.get(
                  created.session.id,
                );
                if (current?.status === 'stopped') return resolve(current);
                if (!current || current.status === 'error') {
                  return reject(new Error(
                    'Visible Stop/Cancel left the active local session missing '
                    + 'or errored',
                  ));
                }
                if (Date.now() - started > 120000) {
                  return reject(new Error(
                    'Timed out waiting for the visibly stopped active local session',
                  ));
                }
                setTimeout(check, 100);
              };
              check();
            });
            preStartStopControl = {
              selector: '[data-vmlx-control="session-stop"]',
              exactSelector: exactStopSelector,
              sessionId: created.session.id,
              label: (stopButton.textContent || '').replace(/\\s+/g, ' ').trim(),
              statusBefore: statusBeforeVisibleStop,
              statusAfter: sessionBeforeStart.status,
              visible: true,
              clicked: true,
            };
          }
          if (
            requestedReuseSessionId
            && sessionBeforeStart?.id !== requestedReuseSessionId
          ) {
            throw new Error('Visible restart selected a different local session');
          }
          if (requestedReuseSessionId) {
            const releasePhaseSessionConfig = ${JSON.stringify(activeReleasePhase ? {
              enablePrefixCache: true,
              usePagedCache: Boolean(activeReleasePhase.paged_ram),
              enableBlockDiskCache: true,
              kvCacheQuantization: String(activeReleasePhase.kv_cache_quantization || 'none'),
              blockDiskCacheDir: releaseBlockDiskCacheDir,
              cacheMemoryPercent: 0,
              ...(releasePagedCacheMemoryMb == null
                ? {}
                : { cacheMemoryMb: releasePagedCacheMemoryMb }),
              blockDiskCacheMaxGb: releaseBlockDiskCacheMaxGb,
            } : {})};
            if (Object.keys(releasePhaseSessionConfig).length) {
              const phaseConfigUpdate = await window.api.sessions.update(
                created.session.id,
                releasePhaseSessionConfig,
              );
              if (!phaseConfigUpdate?.success) {
                throw new Error(
                  phaseConfigUpdate?.error
                    || 'Failed to stage the release phase cache policy',
                );
              }
              sessionBeforeStart = await window.api.sessions.get(created.session.id);
            }
          }
          if (privateCacheAttestationArgs) {
            const launchAdditionalArgs = [
              privateConfigRestoreAdditionalArgs,
              privateCacheAttestationArgs,
            ].filter(Boolean).join(' ');
            const proofConfigUpdate = await window.api.sessions.update(
              created.session.id,
              { additionalArgs: launchAdditionalArgs },
            );
            if (!proofConfigUpdate?.success) {
              throw new Error(
                proofConfigUpdate?.error
                || 'Failed to stage private cache attestation launch arguments',
              );
            }
            sessionBeforeStart = await window.api.sessions.get(created.session.id);
          }
          const serverSettingsControl = await waitFor(() => {
            return [...document.querySelectorAll(
              '[data-vmlx-control="server-settings"]'
            )].find((button) =>
              button instanceof HTMLButtonElement && isVisible(button)
            ) || null;
          }, 'visible Server settings control before Start');
          const serverSettingsAriaBefore =
            serverSettingsControl.getAttribute('aria-pressed');
          if (serverSettingsAriaBefore !== 'true') {
            serverSettingsControl.scrollIntoView({ block: 'center' });
            serverSettingsControl.click();
          }
          await waitFor(() => {
            const drawer = document.querySelector(
              '[data-vmlx-surface="server-settings"]'
            );
            return (
              serverSettingsControl.getAttribute('aria-pressed') === 'true'
              && drawer instanceof HTMLElement
              && isVisible(drawer)
            ) ? drawer : null;
          }, 'visibly open Server settings drawer before Start');
          // Optional lane selection, before Start so the session is CREATED in
          // the SSD-only configuration rather than toggled afterwards.
          let ssdOnlyLaneSelection = null;
          let nativeMtpSelection = null;
          if (
            ${JSON.stringify(nativeMtpModeOverride)} != null
            || ${JSON.stringify(nativeMtpDepthOverride)} != null
          ) {
            const preDrawer = document.querySelector(
              '[data-vmlx-surface="server-settings"]'
            );
            const nativeSectionButton = [...(preDrawer?.querySelectorAll('button') || [])]
              .find((button) => (
                (button.innerText || '').replace(/\\s+/g, ' ').trim() === 'Native MTP'
              ));
            const labelFor = (text) => [...(preDrawer?.querySelectorAll('label') || [])]
              .find((label) => (
                (label.innerText || '').replace(/\\s+/g, ' ').trim().startsWith(text)
              ));
            if (!labelFor('Native MTP Mode')) {
              if (!(nativeSectionButton instanceof HTMLButtonElement)) {
                throw new Error('Visible Native MTP section was not available before Start');
              }
              nativeSectionButton.scrollIntoView({ block: 'center' });
              nativeSectionButton.click();
              await waitFor(
                () => labelFor('Native MTP Mode') || null,
                'visible Native MTP controls before Start',
              );
            }
            const setSelect = async (labelText, requestedValue) => {
              const select = labelFor(labelText)?.querySelector('select');
              if (!(select instanceof HTMLSelectElement) || select.disabled) {
                throw new Error('Visible ' + labelText + ' control was not editable');
              }
              const setter = Object.getOwnPropertyDescriptor(
                HTMLSelectElement.prototype,
                'value',
              )?.set;
              setter?.call(select, requestedValue);
              select.dispatchEvent(new Event('input', { bubbles: true }));
              select.dispatchEvent(new Event('change', { bubbles: true }));
              await waitFor(
                () => select.value === requestedValue ? select : null,
                'visible ' + labelText + ' to update',
              );
              await new Promise((resolve) => setTimeout(resolve, 150));
              return select;
            };
            const requestedMode = ${JSON.stringify(nativeMtpModeOverride)};
            const requestedDepth = ${JSON.stringify(nativeMtpDepthOverride)};
            if (requestedMode != null) {
              await setSelect('Native MTP Mode', requestedMode);
            }
            if (requestedDepth != null) {
              await setSelect('MTP Depth Policy', 'fixed');
              const depthSetting = preDrawer?.querySelector(
                '[data-setting-label="Native MTP Depth"]'
              );
              const depthInput = depthSetting?.querySelector('input[type="range"]');
              if (!(depthInput instanceof HTMLInputElement) || depthInput.disabled) {
                throw new Error('Visible Native MTP Depth control was not editable');
              }
              const setter = Object.getOwnPropertyDescriptor(
                HTMLInputElement.prototype,
                'value',
              )?.set;
              setter?.call(depthInput, String(requestedDepth));
              depthInput.dispatchEvent(new Event('input', { bubbles: true }));
              depthInput.dispatchEvent(new Event('change', { bubbles: true }));
              await waitFor(
                () => Number(depthSetting?.getAttribute('data-setting-value')) === requestedDepth,
                'visible Native MTP depth to update',
              );
            }
            const saveCandidates = [...(preDrawer?.querySelectorAll('button') || [])]
              .filter((button) => /^(save|save & restart|save and restart)$/i.test(
                (button.innerText || '').replace(/\\s+/g, ' ').trim(),
              ) && isVisible(button) && !button.disabled);
            const plainSave = saveCandidates.find(
              (button) => /^save$/i.test((button.innerText || '').trim()),
            ) || saveCandidates[0] || null;
            if (!(plainSave instanceof HTMLButtonElement)) {
              throw new Error('Visible Server Settings Save control was not available');
            }
            plainSave.scrollIntoView({ block: 'center' });
            plainSave.click();
            await new Promise((resolve) => setTimeout(resolve, 400));
            const reread = await window.api.sessions.get(created.session.id);
            const persisted = JSON.parse(reread?.config || '{}');
            if (
              requestedMode != null
              && persisted.nativeMtpMode !== requestedMode
            ) {
              throw new Error(
                'Native MTP mode did not persist: ' + (persisted.nativeMtpMode || 'missing')
              );
            }
            if (
              requestedDepth != null
              && (
                persisted.nativeMtpDepthOverride !== true
                || Number(persisted.nativeMtpDepth) !== requestedDepth
              )
            ) {
              throw new Error(
                'Native MTP fixed depth did not persist: override='
                  + persisted.nativeMtpDepthOverride
                  + ' depth=' + persisted.nativeMtpDepth
              );
            }
            nativeMtpSelection = {
              requested: true,
              requestedMode,
              requestedDepth,
              selectedMode: labelFor('Native MTP Mode')?.querySelector('select')?.value || null,
              selectedDepthPolicy: labelFor('MTP Depth Policy')?.querySelector('select')?.value || null,
              selectedDepth: Number(preDrawer?.querySelector(
                '[data-setting-label="Native MTP Depth"]'
              )?.getAttribute('data-setting-value')),
              savedVia: (plainSave.innerText || '').trim(),
              persistedMode: persisted.nativeMtpMode ?? null,
              persistedDepthOverride: persisted.nativeMtpDepthOverride === true,
              persistedDepth: persisted.nativeMtpDepth ?? null,
            };
            sessionBeforeStart = reread;
          }
          if (
            ${JSON.stringify(forceSsdOnlyLane)}
            || ${JSON.stringify(blockDiskCacheMaxPercentOverride)} != null
          ) {
            const preDrawer = document.querySelector(
              '[data-vmlx-surface="server-settings"]'
            );
            const expandSection = async (title) => {
              const btn = [...(preDrawer?.querySelectorAll('button') || [])]
                .find((b) => (b.innerText || '').replace(/\\s+/g, ' ').trim().includes(title));
              if (btn) {
                btn.scrollIntoView({ block: 'center' });
                btn.click();
                await new Promise((r) => setTimeout(r, 150));
              }
              return !!btn;
            };
            await expandSection('Prefix Cache');
            await expandSection('In-Memory Paged Cache');
            const preLabelFor = (text) => [...(preDrawer?.querySelectorAll('label') || [])]
              .find((label) => (label.innerText || '').includes(text));
            const preInputFor = (text) => preLabelFor(text)?.querySelector('input[type="checkbox"]');
            const pagedPre = preInputFor('In-Memory Paged Cache (RAM)');
            const blockPre = preInputFor('Block Disk Cache (SSD / L2)');
            const percentSetting = () => preDrawer?.querySelector(
              '[data-setting-label="SSD Cache Size (% of disk)"]'
            );
            const percentRange = () => percentSetting()?.querySelector(
              'input[type="range"]'
            );
            const before = {
              usePagedCache: !!pagedPre?.checked,
              enableBlockDiskCache: !!blockPre?.checked,
              pagedDisabled: !!pagedPre?.disabled,
              blockDiskCacheMaxPercent: percentSetting()
                ? Number(percentSetting().getAttribute('data-setting-value'))
                : null,
            };
            if (${JSON.stringify(forceSsdOnlyLane)} && blockPre && !blockPre.checked && !blockPre.disabled) {
              blockPre.scrollIntoView({ block: 'center' });
              blockPre.click();
              await new Promise((r) => setTimeout(r, 150));
            }
            if (${JSON.stringify(forceSsdOnlyLane)} && pagedPre && pagedPre.checked && !pagedPre.disabled) {
              pagedPre.scrollIntoView({ block: 'center' });
              pagedPre.click();
              await new Promise((r) => setTimeout(r, 150));
            }
            const requestedPercent = ${JSON.stringify(blockDiskCacheMaxPercentOverride)};
            if (requestedPercent != null) {
              const input = percentRange();
              if (!(input instanceof HTMLInputElement) || input.disabled) {
                throw new Error('Visible SSD cache percentage control was not editable');
              }
              const setter = Object.getOwnPropertyDescriptor(
                HTMLInputElement.prototype,
                'value',
              )?.set;
              setter?.call(input, String(requestedPercent));
              input.dispatchEvent(new Event('input', { bubbles: true }));
              input.dispatchEvent(new Event('change', { bubbles: true }));
              await waitFor(
                () => Number(percentSetting()?.getAttribute('data-setting-value')) === requestedPercent,
                'visible SSD cache percentage to update',
              );
            }
            // The session record already exists at this point, and the drawer
            // says as much ("Session is running. Save changes and use Save &
            // Restart to apply them"), so an unchecked box alone never reaches
            // the config Start uses. Commit it, then CONFIRM THE SIDE EFFECT by
            // re-reading the persisted session rather than trusting the click.
            let savedVia = null;
            const saveCandidates = [...(preDrawer?.querySelectorAll('button') || [])]
              .filter((b) => /^(save|save & restart|save and restart)$/i.test(
                (b.innerText || '').replace(/\\s+/g, ' ').trim(),
              ) && isVisible(b) && !b.disabled);
            // Prefer the plain Save that sits beside Reset (the Save & Restart
            // variant would tear the session down mid-proof).
            const plainSave = saveCandidates.find(
              (b) => /^save$/i.test((b.innerText || '').trim()),
            ) || saveCandidates[0] || null;
            if (plainSave) {
              plainSave.scrollIntoView({ block: 'center' });
              plainSave.click();
              savedVia = (plainSave.innerText || '').trim();
              await new Promise((r) => setTimeout(r, 400));
            }
            let persistedAfterSave = null;
            try {
              const reread = await window.api.sessions.get(created.session.id);
              persistedAfterSave = JSON.parse(reread?.config || '{}');
            } catch (_) {
              persistedAfterSave = null;
            }
            ssdOnlyLaneSelection = {
              requested: true,
              pagedControlFound: !!pagedPre,
              blockDiskControlFound: !!blockPre,
              before,
              after: {
                usePagedCache: !!preInputFor('In-Memory Paged Cache (RAM)')?.checked,
                enableBlockDiskCache: !!preInputFor('Block Disk Cache (SSD / L2)')?.checked,
                pagedDisabled: !!preInputFor('In-Memory Paged Cache (RAM)')?.disabled,
                blockDiskCacheMaxPercent: percentSetting()
                  ? Number(percentSetting().getAttribute('data-setting-value'))
                  : null,
              },
              savedVia,
              saveCandidateCount: saveCandidates.length,
              persistedUsePagedCacheAfterSave: persistedAfterSave
                ? persistedAfterSave.usePagedCache
                : null,
              persistedBlockDiskAfterSave: persistedAfterSave
                ? persistedAfterSave.enableBlockDiskCache
                : null,
              persistedBlockDiskCacheMaxPercentAfterSave: persistedAfterSave
                ? persistedAfterSave.blockDiskCacheMaxPercent
                : null,
            };
          }
          const singleModelToggle = await waitFor(() => {
            const candidate = document.querySelector(
              '[data-vmlx-control="gateway-single-model-mode"]'
            );
            return candidate instanceof HTMLButtonElement && isVisible(candidate)
              ? candidate
              : null;
          }, 'visible Gateway Single model toggle before Start');
          const gatewayStatusBeforeToggle = await window.api.gateway.getStatus();
          const toggleAriaBefore =
            singleModelToggle.getAttribute('aria-pressed');
          const observedAlreadyOn = toggleAriaBefore === 'true';
          let clickedToEnable = false;
          if (!observedAlreadyOn) {
            singleModelToggle.scrollIntoView({ block: 'center' });
            singleModelToggle.click();
            clickedToEnable = true;
          }
          const enabledGatewayState = await new Promise((resolve, reject) => {
            const started = Date.now();
            const check = async () => {
              const visibleToggle = document.querySelector(
                '[data-vmlx-control="gateway-single-model-mode"]'
              );
              const [status, persistedSetting] = await Promise.all([
                window.api.gateway.getStatus().catch(() => null),
                window.api.settings.get('gateway_single_model_mode')
                  .catch(() => null),
              ]);
              if (
                visibleToggle instanceof HTMLButtonElement
                && isVisible(visibleToggle)
                && visibleToggle.getAttribute('aria-pressed') === 'true'
                && status?.running === true
                && status?.singleModelMode === true
                && persistedSetting === 'true'
              ) {
                resolve({ status, persistedSetting });
                return;
              }
              if (Date.now() - started > 30000) {
                reject(new Error(
                  'Timed out binding the visible Gateway Single model toggle '
                  + 'to live gateway status before Start'
                ));
                return;
              }
              setTimeout(check, 100);
            };
            check();
          });
          const gatewayStatusAfterToggle = enabledGatewayState.status;
          const toggleAriaAfter =
            singleModelToggle.getAttribute('aria-pressed');
          const toggleVisibleAfter = isVisible(singleModelToggle);
          serverSettingsControl.click();
          await waitFor(
            () => serverSettingsControl.getAttribute('aria-pressed') === 'false',
            'visibly closed Server settings drawer before Start',
          );
          const [
            gatewayStatusImmediatelyBeforeStart,
            persistedGatewaySingleModelModeImmediatelyBeforeStart,
          ] = await Promise.all([
            window.api.gateway.getStatus(),
            window.api.settings.get('gateway_single_model_mode'),
          ]);
          if (
            gatewayStatusImmediatelyBeforeStart?.running !== true
            || gatewayStatusImmediatelyBeforeStart?.singleModelMode !== true
            || persistedGatewaySingleModelModeImmediatelyBeforeStart !== 'true'
          ) {
            throw new Error(
              'Gateway Single model mode was not independently persisted '
              + 'and live immediately before Start'
            );
          }
          const gatewaySingleModelMode = {
            requested: true,
            serverSettingsControl: {
              selector: '[data-vmlx-control="server-settings"]',
              visible: isVisible(serverSettingsControl),
              ariaPressedBefore: serverSettingsAriaBefore,
              ariaPressedAfterOpen: 'true',
              ariaPressedAfterClose:
                serverSettingsControl.getAttribute('aria-pressed'),
            },
            toggleControl: {
              selector:
                '[data-vmlx-control="gateway-single-model-mode"]',
              visible: toggleVisibleAfter,
              ariaPressedBefore: toggleAriaBefore,
              ariaPressedAfter: toggleAriaAfter,
              observedAlreadyOn,
              clickedToEnable,
            },
            gatewayStatusBeforeToggle,
            gatewayStatusAfterToggle,
            gatewayStatusImmediatelyBeforeStart,
            persistedSettingImmediatelyBeforeStart: {
              key: 'gateway_single_model_mode',
              value:
                persistedGatewaySingleModelModeImmediatelyBeforeStart,
              source: 'window.api.settings.get',
            },
          };
          const exactStartSelector =
            '[data-vmlx-control="session-start"][data-vmlx-session-id="'
            + CSS.escape(created.session.id)
            + '"]';
          const startButton = await new Promise((resolve, reject) => {
            const started = Date.now();
            const check = () => {
              const candidate = document.querySelector(exactStartSelector);
              if (
                candidate instanceof HTMLButtonElement
                && isVisible(candidate)
                && !candidate.disabled
              ) return resolve(candidate);
              if (Date.now() - started > 30000) {
                return reject(new Error(
                  'Timed out waiting for the exact session-bound visible '
                  + 'local-session Start control: '
                  + document.body.innerText.slice(0, 4000)
                ));
              }
              setTimeout(check, 100);
            };
            check();
          });
          const uiStartControl = {
            selector: '[data-vmlx-control="session-start"]',
            exactSelector: exactStartSelector,
            sessionId: created.session.id,
            label: (startButton.textContent || '').replace(/\\s+/g, ' ').trim(),
            clicked: false,
            sessionStatusBefore: sessionBeforeStart?.status || null,
            sessionStatusAfter: null,
          };
          const waitForCurrentSessionStart = ${waitForCurrentSessionStart.toString()};
          const startedSession = await waitForCurrentSessionStart({
            sessions: window.api.sessions,
            sessionId: created.session.id,
            baselineLastStartedAt:
              sessionBeforeStart?.lastStartedAt
              ?? sessionBeforeStart?.last_started_at
              ?? 0,
            click: () => {
              startButton.scrollIntoView({ block: 'center' });
              startButton.click();
              uiStartControl.clicked = true;
            },
          });
          uiStartControl.sessionStatusAfter = startedSession.status;
          const preloadHealthBefore = await window.api.performance.health(endpoint)
            .catch((error) => ({ error: String(error?.message || error) }));
          const cacheBefore = await window.api.cache.stats(endpoint, created.session.id)
            .catch((error) => ({ error: String(error?.message || error) }));
          const chat = await window.api.chat.create(
            'Real UI live model proof',
            servedModel,
            undefined,
            created.session.modelPath,
          );
          const requestedMaxTokens = ${JSON.stringify(requestMaxTokens ?? null)};
          const rendererGenerationDefaults = await window.api.models.getGenerationDefaults(modelPath)
            .catch((error) => ({ error: String(error?.message || error) }));

          window.dispatchEvent(new CustomEvent('vmlx:navigate', { detail: { mode: 'chat' } }));
          const chatRow = await waitFor(() => {
            const title = [...document.querySelectorAll('span')]
              .find((element) => (element.textContent || '').trim() === 'Real UI live model proof');
            return title?.closest('.cursor-pointer') || null;
          }, 'new chat row in the visible sidebar');
          chatRow.scrollIntoView({ block: 'center' });
          chatRow.click();
          await waitFor(
            () => document.querySelector('textarea:not([disabled])'),
            'active chat composer',
          );

          const chatSettingsButton = await waitFor(() => {
            return [...document.querySelectorAll('[data-vmlx-control="chat-settings"]')]
              .find((button) => button instanceof HTMLButtonElement && isVisible(button)) || null;
          }, 'visible Chat settings button');
          chatSettingsButton.click();
          const chatSettingsDrawer = await waitFor(() => {
            const drawer = document.querySelector('[data-vmlx-surface="chat-settings"]');
            return drawer instanceof HTMLElement && isVisible(drawer) ? drawer : null;
          }, 'Chat Settings drawer');
          const valueSetter = Object.getOwnPropertyDescriptor(
            HTMLInputElement.prototype,
            'value',
          )?.set;
          const selectSetter = Object.getOwnPropertyDescriptor(
            HTMLSelectElement.prototype,
            'value',
          )?.set;
          const setInput = async (input, value, controlLabel) => {
            if (!(input instanceof HTMLInputElement) || !valueSetter) {
              const visibleLabels = [...(chatSettingsDrawer?.querySelectorAll('label, div span') || [])]
                .map((node) => (node.textContent || '').trim())
                .filter((text) => text && text.length < 40)
                .slice(0, 40);
              throw new Error(
                'required visible Chat Settings input was not found: '
                + (controlLabel || 'unknown')
                + ' | visible labels: ' + JSON.stringify(visibleLabels),
              );
            }
            valueSetter.call(input, String(value));
            input.dispatchEvent(new Event('input', { bubbles: true }));
            input.dispatchEvent(new Event('change', { bubbles: true }));
            await new Promise((resolve) => setTimeout(resolve, 50));
          };
          const rangeValueFor = (label) => {
            const input = [...(chatSettingsDrawer?.querySelectorAll('input[type="range"]') || [])].find((candidate) => {
              const field = candidate.parentElement;
              const fieldLabel = field?.querySelector('div span');
              return (fieldLabel?.textContent || '').trim() === label;
            });
            return input ? Number(input.value) : null;
          };
          const rangeInputFor = (label) => [...(chatSettingsDrawer?.querySelectorAll('input[type="range"]') || [])].find((candidate) => {
            const fieldLabel = candidate.parentElement?.querySelector('div span');
            return (fieldLabel?.textContent || '').trim() === label;
          });
          const numberInputFor = (labelText) => [...(chatSettingsDrawer?.querySelectorAll('input[type="number"]') || [])].find((candidate) => {
            const label = candidate.parentElement?.querySelector('label');
            return (label?.textContent || '').trim() === labelText;
          });
          const maxTokenInputFor = () => numberInputFor('Max Tokens');
          const toolResultLimitInputFor = (root = chatSettingsDrawer) => {
            const label = [...(root?.querySelectorAll('label') || [])]
              .find((candidate) =>
                (candidate.textContent || '').replace(/\\s+/g, ' ').trim() === 'Tool Result Limit'
              );
            return label?.parentElement?.querySelector('input[type="range"]') || null;
          };
          const checkboxFor = (labelText) => [...(chatSettingsDrawer?.querySelectorAll('label') || [])]
            .find((label) =>
              (label.textContent || '').replace(/\\s+/g, ' ').trim().startsWith(labelText)
            )?.querySelector('input[type="checkbox"]');
          const chatSettingsInteraction = {
            openedVisibly: true,
            controlsChanged: [],
            savedViaVisibleControl: false,
            reopenedAfterSave: false,
            persistedAfterReopen: false,
          };
          for (const [label, value] of Object.entries({
            Temperature: samplingOverrides.temperature,
            'Top P': samplingOverrides.topP,
            'Top K': samplingOverrides.topK,
            'Min P': samplingOverrides.minP,
            'Repetition Penalty': samplingOverrides.repeatPenalty,
          })) {
            if (value == null) continue;
            // Controls hydrate asynchronously after the drawer opens on a
            // freshly-started session (observed: MiniMax drawer showed only
            // the session header when queried immediately) — wait for the
            // requested control instead of failing on the first paint.
            await setInput(
              await waitFor(() => rangeInputFor(label) || null, 'visible ' + label + ' slider'),
              value,
              label,
            );
            chatSettingsInteraction.controlsChanged.push(label);
          }
          if (requestedMaxTokens != null) {
            await setInput(
              await waitFor(() => maxTokenInputFor() || null, 'visible Max Tokens input'),
              requestedMaxTokens,
              'Max Tokens',
            );
            chatSettingsInteraction.controlsChanged.push('Max Tokens');
          }
          const thinkingLabel = enableThinking === true
            ? 'On'
            : enableThinking === false
              ? 'Off'
              : 'Auto';
          const acceptableThinkingLabels = enableThinking === true
            ? ['On', 'Reasoning']
            : enableThinking === false
              ? ['Off', 'Instruct']
              : ['Auto'];
          // Selecting a concrete effort calls updateThinkingMode(true,
          // effort). An effort-only run therefore begins at Auto but is
          // intentionally persisted and reopened as On.
          const acceptablePersistedThinkingLabels = (
            enableThinking === undefined && requestedReasoningEffort
          )
            ? ['On', 'Reasoning']
            : acceptableThinkingLabels;
          // A family whose template never reads enable_thinking renders an
          // honesty NOTICE instead of the Auto/On/Off button group
          // (ChatSettings thinkingNotConfigurable — LFM2.5, MiniMax). Wait for
          // either shape; interact only when the buttons exist. Requesting an
          // explicit on/off override against the notice is a row
          // misconfiguration and stays fatal.
          const findThinkingButton = () => [...(chatSettingsDrawer?.querySelectorAll('button') || [])]
            .find((button) =>
              isVisible(button)
              && !button.disabled
              && acceptablePersistedThinkingLabels.includes(
                (button.textContent || '').replace(/\\s+/g, ' ').trim()
              )
            ) || null;
          const thinkingNoticeShown = () =>
            /does not read a thinking toggle/i.test(chatSettingsDrawer?.innerText || '');
          const thinkingControl = await waitFor(
            () => findThinkingButton() || (thinkingNoticeShown() ? 'notice' : null),
            'visible reasoning control (' + thinkingLabel + ' button or not-configurable notice)',
          ).catch(() => null);
          if (thinkingControl === 'notice' || thinkingControl == null) {
            if (enableThinking === true || enableThinking === false) {
              throw new Error(
                'visible reasoning mode control missing: ' + thinkingLabel
                + (thinkingControl === 'notice'
                  ? ' (family renders the thinking-not-configurable notice)'
                  : ''),
              );
            }
            // Auto with no toggle rendered: family default applies untouched.
          } else {
            thinkingControl.click();
            chatSettingsInteraction.controlsChanged.push('Reasoning ' + thinkingLabel);
            await new Promise((resolve) => setTimeout(resolve, 50));
          }
          if (requestedReasoningEffort) {
            const effortControl = await waitFor(() => {
              const candidate = chatSettingsDrawer?.querySelector(
                '[data-reasoning-effort="' + CSS.escape(requestedReasoningEffort) + '"]'
              );
              return candidate instanceof HTMLButtonElement
                && isVisible(candidate)
                && !candidate.disabled
                ? candidate
                : null;
            }, 'visible native reasoning effort ' + requestedReasoningEffort);
            effortControl.click();
            chatSettingsInteraction.controlsChanged.push(
              'Reasoning Effort ' + requestedReasoningEffort,
            );
            await new Promise((resolve) => setTimeout(resolve, 50));
          }
          const wireSelect = await waitFor(
            () => [...(chatSettingsDrawer?.querySelectorAll('select') || [])]
              .find((candidate) =>
                [...candidate.options].some((option) => option.value === 'responses')
              ) || null,
            'visible API Wire Format control',
          ).catch(() => null);
          if (!(wireSelect instanceof HTMLSelectElement) || !selectSetter) {
            throw new Error('visible API Wire Format control missing');
          }
          const desiredWire = wireApi === 'responses' ? 'responses' : 'completions';
          const alternateWire = desiredWire === 'responses' ? 'completions' : 'responses';
          for (const nextWire of [alternateWire, desiredWire]) {
            selectSetter.call(wireSelect, nextWire);
            wireSelect.dispatchEvent(new Event('change', { bubbles: true }));
            await new Promise((resolve) => setTimeout(resolve, 50));
          }
          chatSettingsInteraction.controlsChanged.push('API Wire Format');
          const builtinInput = checkboxFor('Enable Built-in Coding Tools');
          if (!(builtinInput instanceof HTMLInputElement)) {
            throw new Error('visible built-in tools checkbox missing');
          }
          if (Boolean(builtinInput.checked) !== builtinToolsEnabled) {
            builtinInput.click();
            chatSettingsInteraction.controlsChanged.push('Enable Built-in Coding Tools');
            await waitFor(
              () => Boolean(builtinInput.checked) === builtinToolsEnabled,
              'visible built-in tools checkbox state',
            );
          }
          if (builtinToolsEnabled) {
            const workingInput = await waitFor(() =>
              [...(chatSettingsDrawer?.querySelectorAll('input[type="text"]') || [])]
                .find((candidate) =>
                  (candidate.getAttribute('placeholder') || '').includes('project directory')
                ) || null,
            'visible Working Directory input');
            await setInput(workingInput, workingDirectory, 'Working Directory');
            await setInput(numberInputFor('Max Tool Iterations'), ${JSON.stringify(maxToolIterations)}, 'Max Tool Iterations');
            await setInput(
              toolResultLimitInputFor(),
              ${JSON.stringify(toolResultMaxChars)},
              'Tool Result Limit',
            );
            chatSettingsInteraction.controlsChanged.push(
              'Working Directory',
              'Max Tool Iterations',
              'Tool Result Limit',
            );
            for (const [label, desired] of [
              ['File I/O', true],
              ['Search', false],
              ['Shell', true],
              ['Web Search', false],
              ['URL Fetch', false],
              ['Git', false],
              ['Utilities', true],
            ]) {
              const input = checkboxFor(label);
              if (input instanceof HTMLInputElement && Boolean(input.checked) !== desired) {
                input.click();
                chatSettingsInteraction.controlsChanged.push(label);
                await waitFor(
                  () => Boolean(input.checked) === desired,
                  'visible ' + label + ' tool checkbox state',
                );
              }
            }
          }
          const saveButton = await waitFor(() => {
            const buttons = [...(chatSettingsDrawer?.querySelectorAll(
              '[data-vmlx-control="chat-settings-save"]',
            ) || [])].filter((candidate) =>
              candidate instanceof HTMLButtonElement && isVisible(candidate)
            );
            const button = buttons.length === 1 ? buttons[0] : null;
            return button instanceof HTMLButtonElement
              && !button.disabled
              && button.getAttribute('data-vmlx-state') === 'dirty'
              ? button
              : null;
          },
          'enabled visible Chat Settings Save control');
          saveButton.click();
          chatSettingsInteraction.savedViaVisibleControl = true;
          await waitFor(() => {
            const buttons = [...(chatSettingsDrawer?.querySelectorAll(
              '[data-vmlx-control="chat-settings-save"]',
            ) || [])].filter((candidate) =>
              candidate instanceof HTMLButtonElement && isVisible(candidate)
            );
            const current = buttons.length === 1 ? buttons[0] : null;
            return current instanceof HTMLButtonElement
              && current.disabled
              && current.getAttribute('data-vmlx-state') === 'saved';
          }, 'Chat Settings save completion');
          // Auto can run native MTP with the bundle/request distribution via
          // stochastic verification. Only the explicit Deterministic mode
          // pins the visible Chat Settings tuple to greedy values.
          const persistedNativeMtpMode = nativeMtpSelection?.persistedMode
            ?? (() => {
              try {
                return JSON.parse(sessionBeforeStart?.config || '{}').nativeMtpMode;
              } catch (_) {
                return null;
              }
            })();
          const nativeMtpGreedyUi =
            preloadHealthBefore?.mtp?.runtime_active === true
            && persistedNativeMtpMode === 'deterministic';
          const expectedUiValues = {
            Temperature: nativeMtpGreedyUi
              ? 0
              : samplingOverrides.temperature ?? independentBundleDefaults?.temperature,
            'Top P': nativeMtpGreedyUi
              ? 1
              : samplingOverrides.topP ?? independentBundleDefaults?.topP,
            'Top K': nativeMtpGreedyUi
              ? 0
              : samplingOverrides.topK ?? independentBundleDefaults?.topK,
            'Min P': nativeMtpGreedyUi
              ? 0
              : samplingOverrides.minP ?? independentBundleDefaults?.minP,
            'Repetition Penalty':
              samplingOverrides.repeatPenalty ?? independentBundleDefaults?.repeatPenalty,
          };
          const bundleHasSamplerDefaults = independentBundleDefaults
            && Object.values(independentBundleDefaults).some((value) => value != null);
          await waitFor(() => {
            if (!bundleHasSamplerDefaults) {
              // A bundle with NO stamped sampler defaults (e.g. a
              // generation_config carrying only bos/eos ids —
              // Step-3.7-Flash-JANG_K-CRACK) renders the engine-fallback
              // banner instead of sliders. There are no model-derived
              // values to verify — accept once the drawer states it.
              if (!/No sampler defaults were available from this bundle/i
                .test(chatSettingsDrawer?.innerText || '')) return false;
              const fallbackMaxTokens = maxTokenInputFor();
              if (!fallbackMaxTokens) return false;
              chatSettingsInteraction.noSamplerDefaultsBanner = true;
              return requestedMaxTokens != null
                ? Number(fallbackMaxTokens.value) === Number(requestedMaxTokens)
                : true;
            }
            for (const [label, expected] of Object.entries(expectedUiValues)) {
              if (expected == null) continue;
              const observed = rangeValueFor(label);
              if (
                typeof observed !== 'number'
                || !Number.isFinite(observed)
                || Math.abs(observed - Number(expected)) > 1e-6
              ) return false;
            }
            const maxTokens = maxTokenInputFor();
            if (!maxTokens) return false;
            return requestedMaxTokens != null
              ? Number(maxTokens.value) === Number(requestedMaxTokens)
              : maxTokens.value === ''
                && (!independentBundleDefaults.maxNewTokens
                  || maxTokens.placeholder.includes(String(independentBundleDefaults.maxNewTokens)));
          }, 'model-derived values in the visible Chat Settings drawer');
          const chatSettingsClose = [...(chatSettingsDrawer?.querySelectorAll('button') || [])]
            .find((button) => button.getAttribute('aria-label') === 'Close');
          if (!chatSettingsClose) throw new Error('Chat Settings close control was not visible');
          chatSettingsClose.click();
          await waitFor(
            () => !document.querySelector('[data-vmlx-surface="chat-settings"]'),
            'Chat Settings drawer to close',
          );
          chatSettingsButton.click();
          const reopenedDrawer = await waitFor(() => {
            const drawer = document.querySelector('[data-vmlx-surface="chat-settings"]');
            return drawer instanceof HTMLElement && isVisible(drawer) ? drawer : null;
          }, 'reopened Chat Settings drawer');
          // The reopened drawer re-fetches saved settings + model defaults;
          // reading controls while "Loading saved chat settings and model
          // defaults…" is still showing races the hydration and reports every
          // field as null (observed live: MiniMax row reopened into the
          // loading state and failed persistence checks that had actually
          // saved). The first-open path already waits for hydration; the
          // reopen read must too.
          await waitFor(() => {
            const drawerNow = document.querySelector('[data-vmlx-surface="chat-settings"]');
            if (!(drawerNow instanceof HTMLElement)) return false;
            if (/Loading saved chat settings/i.test(drawerNow.textContent || '')) return false;
            return drawerNow.querySelectorAll('input[type="range"]').length > 0
              || drawerNow.querySelectorAll('select').length > 0;
          }, 'reopened Chat Settings drawer to hydrate');
          chatSettingsInteraction.reopenedAfterSave = true;
          const reopenedRangeValueFor = (label) => {
            const input = [...(reopenedDrawer?.querySelectorAll('input[type="range"]') || [])]
              .find((candidate) =>
                (candidate.parentElement?.querySelector('div span')?.textContent || '').trim() === label
              );
            return input ? Number(input.value) : null;
          };
          const reopenedMaxToken = [...(reopenedDrawer?.querySelectorAll('input[type="number"]') || [])]
            .find((candidate) =>
              (candidate.parentElement?.querySelector('label')?.textContent || '').trim() === 'Max Tokens'
            );
          const reopenedNumberValueFor = (labelText) => {
            const input = [...(reopenedDrawer?.querySelectorAll('input[type="number"]') || [])]
              .find((candidate) =>
                (candidate.parentElement?.querySelector('label')?.textContent || '').trim() === labelText
              );
            return input?.value ?? null;
          };
          const reopenedCheckboxValueFor = (labelText) => {
            const input = [...(reopenedDrawer?.querySelectorAll('label') || [])]
              .find((label) =>
                (label.textContent || '').replace(/\\s+/g, ' ').trim().startsWith(labelText)
              )?.querySelector('input[type="checkbox"]');
            return input instanceof HTMLInputElement ? input.checked : null;
          };
          const reopenedWireSelect = [...(reopenedDrawer?.querySelectorAll('select') || [])]
            .find((candidate) =>
              [...candidate.options].some((option) => option.value === 'responses')
            );
          const reopenedThinkingButton = [...(reopenedDrawer?.querySelectorAll('button') || [])]
            .find((button) => (
              isVisible(button)
              && acceptablePersistedThinkingLabels.includes(
                (button.textContent || '').replace(/\\s+/g, ' ').trim()
              )
              && (button.getAttribute('data-vmlx-state') === 'selected'
                || (!button.hasAttribute('data-vmlx-state') && String(button.className || '').includes('bg-primary')))
            ));
          const reopenedReasoningEffortButton = [...(reopenedDrawer?.querySelectorAll(
            '[data-reasoning-effort]'
          ) || [])].find((button) => (
            button instanceof HTMLButtonElement
            && isVisible(button)
            && (button.getAttribute('data-vmlx-state') === 'selected'
              || (!button.hasAttribute('data-vmlx-state') && String(button.className || '').includes('bg-primary')))
          ));
          const reopenedWorkingDirectory = [...(reopenedDrawer?.querySelectorAll('input[type="text"]') || [])]
            .find((candidate) =>
              (candidate.getAttribute('placeholder') || '').includes('project directory')
            )?.value ?? null;
          const reopenedToolResultLimit = toolResultLimitInputFor(reopenedDrawer);
          const chatSettingsDom = {
            values: {
              temperature: reopenedRangeValueFor('Temperature'),
              topP: reopenedRangeValueFor('Top P'),
              topK: reopenedRangeValueFor('Top K'),
              minP: reopenedRangeValueFor('Min P'),
              repeatPenalty: reopenedRangeValueFor('Repetition Penalty'),
            },
            maxTokens: {
              value: reopenedMaxToken?.value ?? null,
              placeholder: reopenedMaxToken?.getAttribute('placeholder') || '',
            },
            wireApi: reopenedWireSelect?.value ?? null,
            reasoningMode: (reopenedThinkingButton?.textContent || '')
              .replace(/\\s+/g, ' ')
              .trim() || null,
            reasoningEffort: reopenedReasoningEffortButton?.getAttribute(
              'data-reasoning-effort'
            ) || null,
            // Notice families (LFM2.5, MiniMax) render the honesty notice
            // instead of mode buttons — record the shape so the reopen
            // validator can accept a legitimately absent mode control.
            thinkingNotice: /does not read a thinking toggle/i.test(
              reopenedDrawer?.innerText || '',
            ),
            builtinToolsEnabled: reopenedCheckboxValueFor('Enable Built-in Coding Tools'),
            workingDirectory: reopenedWorkingDirectory,
            maxToolIterations: reopenedNumberValueFor('Max Tool Iterations'),
            toolResultMaxChars: reopenedToolResultLimit?.value ?? null,
            toolCategories: {
              file: reopenedCheckboxValueFor('File I/O'),
              search: reopenedCheckboxValueFor('Search'),
              shell: reopenedCheckboxValueFor('Shell'),
              webSearch: reopenedCheckboxValueFor('Web Search'),
              urlFetch: reopenedCheckboxValueFor('URL Fetch'),
              git: reopenedCheckboxValueFor('Git'),
              utilities: reopenedCheckboxValueFor('Utilities'),
            },
            textHead: reopenedDrawer?.innerText.slice(0, 3000) || '',
          };
          const chatOverrides = await window.api.chat.getOverrides(chat.id)
            .catch((error) => ({ error: String(error?.message || error) }));
          const samplingLabelByField = {
            temperature: 'Temperature',
            topP: 'Top P',
            topK: 'Top K',
            minP: 'Min P',
            repeatPenalty: 'Repetition Penalty',
          };
          const explicitSamplingPersisted = Object.entries(samplingOverrides)
            .every(([field, expected]) => {
              if (expected == null) return true;
              if (
                typeof chatOverrides?.[field] === 'number'
                && Math.abs(Number(chatOverrides[field]) - Number(expected)) <= 1e-6
              ) return true;
              const label = samplingLabelByField[field];
              const visible = label ? reopenedRangeValueFor(label) : null;
              return typeof visible === 'number'
                && Math.abs(Number(visible) - Number(expected)) <= 1e-6;
            });
          const requestedMaxTokensPersisted = requestedMaxTokens == null
            || Number(chatOverrides?.maxTokens) === Number(requestedMaxTokens);
          const visibleSamplingPersisted = Object.entries(expectedUiValues)
            .every(([label, expected]) => (
              expected == null
              || (
                typeof reopenedRangeValueFor(label) === 'number'
                && Math.abs(Number(reopenedRangeValueFor(label)) - Number(expected)) <= 1e-6
              )
            ));
          const visibleMaxTokensPersisted = requestedMaxTokens == null
            ? reopenedMaxToken?.value === ''
            : Number(reopenedMaxToken?.value) === Number(requestedMaxTokens);
          const visibleToolSettingsPersisted = !builtinToolsEnabled || (
            chatSettingsDom.builtinToolsEnabled === true
            && chatSettingsDom.workingDirectory === workingDirectory
            && Number(chatSettingsDom.maxToolIterations) === ${JSON.stringify(maxToolIterations)}
            && Number(chatSettingsDom.toolResultMaxChars) === ${JSON.stringify(toolResultMaxChars)}
            && chatSettingsDom.toolCategories.file === true
            && chatSettingsDom.toolCategories.search === false
            && chatSettingsDom.toolCategories.shell === true
            && chatSettingsDom.toolCategories.webSearch === false
            && chatSettingsDom.toolCategories.urlFetch === false
            && chatSettingsDom.toolCategories.git === false
            && chatSettingsDom.toolCategories.utilities === true
          );
          const toolSettingsPersisted = !builtinToolsEnabled || (
            chatOverrides?.workingDirectory === workingDirectory
            && Number(chatOverrides?.maxToolIterations) === ${JSON.stringify(maxToolIterations)}
            && Number(chatOverrides?.toolResultMaxChars) === ${JSON.stringify(toolResultMaxChars)}
            && chatOverrides?.fileToolsEnabled !== false
            && chatOverrides?.searchToolsEnabled === false
            && chatOverrides?.shellEnabled !== false
            && chatOverrides?.webSearchEnabled === false
            && chatOverrides?.fetchUrlEnabled === false
            && chatOverrides?.gitEnabled === false
            && chatOverrides?.utilityToolsEnabled !== false
          );
          chatSettingsInteraction.persistedAfterReopen = (
            chatOverrides?.wireApi === desiredWire
            && Boolean(chatOverrides?.builtinToolsEnabled) === builtinToolsEnabled
            && (enableThinking === undefined
              ? (requestedReasoningEffort
                ? chatOverrides?.enableThinking === true
                : chatOverrides?.enableThinking == null)
              : chatOverrides?.enableThinking === enableThinking)
            && (requestedReasoningEffort == null
              ? chatOverrides?.reasoningEffort == null
              : chatOverrides?.reasoningEffort === requestedReasoningEffort)
            && explicitSamplingPersisted
            && requestedMaxTokensPersisted
            && toolSettingsPersisted
            && visibleSamplingPersisted
            && visibleMaxTokensPersisted
            && chatSettingsDom.wireApi === desiredWire
            && (acceptablePersistedThinkingLabels.includes(chatSettingsDom.reasoningMode)
              || (chatSettingsDom.thinkingNotice === true
                && enableThinking === undefined
                && !chatSettingsDom.reasoningMode))
            && (requestedReasoningEffort == null
              ? chatSettingsDom.reasoningEffort == null
              : chatSettingsDom.reasoningEffort === requestedReasoningEffort)
            && chatSettingsDom.builtinToolsEnabled === builtinToolsEnabled
            && visibleToolSettingsPersisted
          );
          const reopenedClose = [...(reopenedDrawer?.querySelectorAll('button') || [])]
            .find((button) => button.getAttribute('aria-label') === 'Close');
          if (!reopenedClose) throw new Error('reopened Chat Settings close control was not visible');
          reopenedClose.click();
          await waitFor(
            () => !document.querySelector('[data-vmlx-surface="chat-settings"]'),
            'reopened Chat Settings drawer to close',
          );

          const sendErrors = [];
          let rendererFailureStage = null;
          const uiTurnEvidence = [];
          const cacheRequestEvidence = [];
          const sendMessageThroughVisibleComposer = async (turn, stage, prompt) => {
            try {
              const messagesBefore = await window.api.chat.getMessages(chat.id);
              const knownMessageIds = new Set(messagesBefore.map((message) => String(message.id)));
              const turnCacheBefore = await window.api.cache.stats(endpoint, created.session.id)
                .catch((error) => ({ error: String(error?.message || error) }));
              const textarea = await waitFor(
                () => document.querySelector('textarea:not([disabled])'),
                'enabled visible composer for turn ' + turn,
              );
              const setter = Object.getOwnPropertyDescriptor(
                HTMLTextAreaElement.prototype,
                'value',
              )?.set;
              setter.call(textarea, prompt);
              textarea.dispatchEvent(new Event('input', { bubbles: true }));
              textarea.dispatchEvent(new Event('change', { bubbles: true }));
              const completedBefore = events.complete.length;
              const sendButton = await waitFor(() => {
                const sibling = textarea.nextElementSibling;
                return sibling instanceof HTMLButtonElement
                  && !sibling.disabled
                  && isVisible(sibling)
                  ? sibling
                  : null;
              // The terminal-event wait below allows 10 minutes, but this one
              // used the 30s default, so a very large model that was still
              // settling (Stop still showing) failed the RUN rather than the
              // turn: dots3-note (280B) timed out here on turn 2 with
              // "Timed out waiting for enabled visible Send control".
              // Bounded, and still far below the completion wait, so a
              // genuinely wedged composer is still caught.
              }, 'enabled visible Send control for turn ' + turn, ${JSON.stringify(sendReadyTimeoutMs)});
              sendButton.scrollIntoView({ block: 'center' });
              sendButton.click();
              await waitFor(
                () => events.complete.length > completedBefore,
                'terminal event for visible UI turn ' + turn,
                600000,
              );
              const terminalAtCompletion =
                events.complete.slice(completedBefore).at(-1) || null;
              // Capture the singleton scheduler execution immediately after
              // this exact terminal event. The parent does not start any
              // paired API/cache producer until the UI attestation is written,
              // so a later unrelated request cannot be time-window attributed
              // to this turn.
              const turnCacheAfter = await waitForRequestCacheAdvance(
                turnCacheBefore,
                created.session.id,
              );
              const turnHealthAfter = await window.api.performance.health(endpoint)
                .catch((error) => ({ error: String(error?.message || error) }));
              await waitFor(
                () => document.querySelector('textarea:not([disabled])'),
                'composer recovery after turn ' + turn,
                30000,
              );
              const messagesAfter = await window.api.chat.getMessages(chat.id);
              const addedMessages = messagesAfter.filter(
                (message) => !knownMessageIds.has(String(message.id))
              );
              const userMessage = addedMessages.find(
                (message) => message.role === 'user' && String(message.content || '') === prompt
              );
              const assistantMessage = [...addedMessages]
                .reverse()
                .find((message) => message.role === 'assistant');
              const terminal = events.complete.slice(completedBefore)
                .find((event) => String(event?.messageId || '') === String(assistantMessage?.id || ''));
              const boundTerminal = terminal || terminalAtCompletion;
              const proofRequestId = String(userMessage?.id || '');
              const terminalProofRequestId = String(boundTerminal?.proofRequestId || '');
              const requestIds = [...new Set(
                (Array.isArray(boundTerminal?.requestIds) ? boundTerminal.requestIds : [])
                  .map((value) => String(value || ''))
                  .filter(Boolean),
              )];
              const resolvedLogEvidence = await waitForResolvedTurnLog(
                created.session.id,
                {
                  proofRequestId,
                  messageId: String(assistantMessage?.id || ''),
                  requestIds,
                },
              );
              uiTurnEvidence.push({
                turn,
                prompt,
                proofRequestId,
                terminalProofRequestId,
                requestIds,
                userMessageId: userMessage?.id || null,
                assistantMessageId: assistantMessage?.id || null,
                terminalMessageId: boundTerminal?.messageId || null,
                terminalResponseId: boundTerminal?.responseId || null,
                logMatchMode: 'exact_identity_ring_safe',
                logLines: resolvedLogEvidence.matchedLines,
              });
              let effectiveHealthAfter = turnHealthAfter;
              let cacheCorrelation =
                correlateTerminalResponseToCacheExecution({
                  terminal: boundTerminal,
                  cacheSnapshot: effectiveHealthAfter,
                  turn,
                  proofRequestId,
                  userMessageId: userMessage?.id || null,
                  assistantMessageId: assistantMessage?.id || null,
                });
              // The clean prefix-cache store runs AFTER the turn completes, so
              // scheduler.last_cache_execution can lag the terminal response id
              // briefly. Poll a few times before recording a partial mismatch.
              // A NULL last_cache_execution is the same lag one step earlier
              // (health read before the scheduler records the execution) and
              // reports partial_product_support_missing — live polling proved
              // the engine sets it seconds after each terminal completion, so
              // that status must retry too instead of being recorded as a
              // missing product capability.
              for (
                let retry = 0;
                retry < 6
                  && (cacheCorrelation.correlationStatus === 'partial_request_identity_mismatch'
                    || cacheCorrelation.correlationStatus === 'partial_product_support_missing');
                retry += 1
              ) {
                await new Promise((resolve) => setTimeout(resolve, 2000));
                effectiveHealthAfter = await window.api.performance.health(endpoint)
                  .catch((error) => ({ error: String(error?.message || error) }));
                cacheCorrelation = correlateTerminalResponseToCacheExecution({
                  terminal: boundTerminal,
                  cacheSnapshot: effectiveHealthAfter,
                  turn,
                  proofRequestId,
                  userMessageId: userMessage?.id || null,
                  assistantMessageId: assistantMessage?.id || null,
                });
              }
              cacheRequestEvidence.push({
                ...cacheCorrelation,
                before: turnCacheBefore,
                after: turnCacheAfter,
                healthAfter: effectiveHealthAfter,
              });
              return true;
            } catch (error) {
              rendererFailureStage = stage;
              sendErrors.push({ turn, stage, message: String(error?.message || error) });
              return false;
            }
          };
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
          const firstSent = await sendMessageThroughVisibleComposer(
            1,
            'first_visible_ui_send',
            ${JSON.stringify(selectedPromptOne)},
          );
          // MID-CONVERSATION reasoning flip. The harness otherwise sets the
          // thinking mode once per run, so "reasoning toggled mid-convo" — a
          // real user action, and one that interacts with the per-iteration
          // reasoning rail reset and the never-empty resolver's reasoning
          // input — had never been exercised here at all.
          let midConvReasoningFlip = null;
          if (
            (${JSON.stringify(toggleThinkingMidConv)} || requestedMidConvReasoningEffort)
            && firstSent
          ) {
            try {
              const openBtn = [...document.querySelectorAll('[data-vmlx-control="chat-settings"]')]
                .find((b) => b instanceof HTMLButtonElement && isVisible(b));
              if (!openBtn) throw new Error('chat-settings control not visible');
              openBtn.scrollIntoView({ block: 'center' });
              openBtn.click();
              const drawer = await waitFor(
                () => {
                  const d = document.querySelector('[data-vmlx-surface="chat-settings"]');
                  return d instanceof HTMLElement && isVisible(d) ? d : null;
                },
                'chat-settings drawer for mid-conversation reasoning flip',
              );
              // Flip to the OPPOSITE of what this run started with.
              const targetLabels = ${JSON.stringify(enableThinkingOverride === true
                ? ['Off', 'Instruct']
                : ['On', 'Reasoning'])};
              // The drawer renders "Loading saved chat settings and model
              // defaults…" before the thinking control exists, so reading it
              // immediately finds only Save/Reset and makes a present control
              // look missing — which is exactly how I first misread dots3-note.
              // Wait for the loading state to clear before deciding anything.
              await waitFor(
                () => !/loading saved chat settings/i.test(drawer?.innerText || ''),
                'chat settings to finish loading before the reasoning flip',
                60000,
              ).catch(() => null);
              // Record what this family ACTUALLY offers before trying to click
              // one. The target labels are a cross-family guess, and dots3-note
              // renders neither "Off" nor "Instruct", so buttonFound was false
              // and the flip silently did nothing. Capturing the real labels
              // turns that into a visible answer to "which reasoning
              // enforcement buttons does this model have".
              // Capture the drawer TEXT too: when a family's template does not
              // read enable_thinking the panel deliberately renders an honest
              // notice (chat.settings.thinkingNotConfigurable) INSTEAD of the
              // Auto/On/Off buttons. A button-only capture cannot tell that
              // correct behaviour apart from a missing control.
              // Whitespace escapes in here need DOUBLE backslashes: this is
              // inside the evaluate() template literal, so a single one
              // collapses and the page ends up matching the letter "s",
              // deleting every "s" from the captured text — it produced
              // "Loading aved chat etting" on the first run. Ledger row 217 is
              // the same trap; a Python heredoc collapsed the escape this time.
              // (The rule applies to COMMENTS too — the first version of this
              // note tripped the harness's own escape guard.)
              const drawerTextHead = (drawer?.innerText || '')
                .replace(/\\s+/g, ' ')
                .trim()
                .slice(0, 700);
              const drawerButtonLabels = [...(drawer?.querySelectorAll('button') || [])]
                .filter((b) => isVisible(b))
                .map((b) => (b.textContent || '').replace(/\\s+/g, ' ').trim())
                .filter((text) => text && text.length <= 24)
                .slice(0, 30);
              const btn = requestedMidConvReasoningEffort
                ? drawer?.querySelector(
                    '[data-reasoning-effort="'
                    + CSS.escape(requestedMidConvReasoningEffort)
                    + '"]'
                  )
                : [...(drawer?.querySelectorAll('button') || [])]
                    .find((b) => isVisible(b) && !b.disabled
                      && targetLabels.includes((b.textContent || '').replace(/\\s+/g, ' ').trim()));
              const pressedBefore = btn ? btn.getAttribute('aria-pressed') : null;
              if (btn) {
                btn.scrollIntoView({ block: 'center' });
                btn.click();
                await new Promise((r) => setTimeout(r, 120));
              }
              const saveBtn = document.querySelector('[data-vmlx-control="chat-settings-save"]');
              let saved = false;
              if (saveBtn instanceof HTMLButtonElement && isVisible(saveBtn) && !saveBtn.disabled) {
                saveBtn.click();
                saved = true;
                await new Promise((r) => setTimeout(r, 300));
              }
              // Confirm the SIDE EFFECT on the persisted chat overrides rather
              // than trusting the clicks.
              // Read the same source the harness already trusts for persisted
              // chat state (chat.getOverrides), not a guessed shape — my first
              // attempt used window.api.chat.get and returned null, which
              // proved nothing.
              let persistedAfter = null;
              let persistedReasoningEffortAfter = null;
              let overridesAfter = null;
              try {
                overridesAfter = await window.api.chat.getOverrides(chat.id);
                persistedAfter = overridesAfter?.enableThinking ?? null;
                persistedReasoningEffortAfter = overridesAfter?.reasoningEffort ?? null;
              } catch (_) {}
              // CLOSE the drawer. Leaving it open covers the composer, so the
              // NEXT turn's Send control never becomes reachable and the run
              // dies with "Timed out waiting for enabled visible Send control
              // for turn 2". That is what actually happened on every flip run
              // so far: both qwen36 mid-conv artifacts and the dots3-note one
              // recorded assistantRecords 1 / complete 1 / sendErrors 1, so
              // turns 2 and 3 never ran and the low reasoningDone count that I
              // previously read as "reasoning stopped after the flip" was just
              // the single turn that executed.
              let drawerClosed = false;
              const closeBtn = [...document.querySelectorAll('[data-vmlx-control="chat-settings"]')]
                .find((b) => b instanceof HTMLButtonElement && isVisible(b));
              if (closeBtn) {
                closeBtn.click();
                for (let i = 0; i < 40; i += 1) {
                  const still = document.querySelector('[data-vmlx-surface="chat-settings"]');
                  if (!(still instanceof HTMLElement) || !isVisible(still)) {
                    drawerClosed = true;
                    break;
                  }
                  await new Promise((r) => setTimeout(r, 100));
                }
              }
              midConvReasoningFlip = {
                requested: true,
                targetLabels,
                buttonFound: !!btn,
                drawerButtonLabels,
                drawerTextHead,
                ariaPressedBefore: pressedBefore,
                ariaPressedAfter: btn ? btn.getAttribute('aria-pressed') : null,
                saved,
                drawerClosed,
                persistedEnableThinkingAfterSave: persistedAfter,
                persistedReasoningEffortAfter,
                overrideKeysAfterSave: overridesAfter && typeof overridesAfter === 'object'
                  ? Object.keys(overridesAfter).sort()
                  : null,
              };
            } catch (error) {
              midConvReasoningFlip = {
                requested: true,
                error: String(error?.message || error),
              };
            }
          }
          let secondSent = false;
          if (firstSent && uiTurnCount >= 2) {
            secondSent = await sendMessageThroughVisibleComposer(
              2,
              'second_visible_ui_send',
              ${JSON.stringify(selectedPromptTwo)},
            );
          }
          if (secondSent && uiTurnCount >= 3) {
            await sendMessageThroughVisibleComposer(
              3,
              'third_visible_ui_send',
              ${JSON.stringify(selectedPromptThree)},
            );
          }
          if (checkMedia && !rendererFailureStage) {
            await sendMessageWithCapture(4, 'image_send_message', 'What is the dominant color of the attached image? Reply with one color word in English.', [
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
                turn: 5,
                stage: 'video_data_url_missing',
                message: 'VMLINUX_REAL_UI_CHECK_VIDEO requires VMLINUX_REAL_UI_VIDEO_DATA_URL',
              });
            } else {
              await sendMessageWithCapture(5, 'video_send_message', 'Describe the attached video briefly in English.', [
                {
                  name: 'real-ui-proof-video.mp4',
                  type: 'video/mp4',
                  kind: 'video',
                  dataUrl: videoDataUrl,
                },
              ]);
            }
          }
          // Audio turn. Same shape as the video turn, on turn 6 so it never
          // displaces an existing row's turn numbering.
          if (checkAudio && !rendererFailureStage) {
            if (!audioDataUrl) {
              rendererFailureStage = 'audio_data_url_missing';
              sendErrors.push({
                turn: 6,
                stage: 'audio_data_url_missing',
                message: 'VMLINUX_REAL_UI_CHECK_AUDIO requires VMLINUX_REAL_UI_AUDIO_DATA_URL',
              });
            } else {
              await sendMessageWithCapture(6, 'audio_send_message', 'Describe the attached audio briefly in English.', [
                {
                  name: 'real-ui-proof-audio.wav',
                  type: 'audio/wav',
                  kind: 'audio',
                  dataUrl: audioDataUrl,
                },
              ]);
            }
          }
          const preloadHealthAfter = await window.api.performance.health(endpoint)
            .catch((error) => ({ error: String(error?.message || error) }));
          const cacheAfter = await window.api.cache.stats(endpoint, created.session.id)
            .catch((error) => ({ error: String(error?.message || error) }));
          const cacheAfterSettled = await waitForCacheEndpointStorage(
            cacheAfter,
            created.session.id,
          );
          const messages = await window.api.chat.getMessages(chat.id);
          const assistants = messages.filter((m) => m.role === 'assistant');
          const assistantMessageIds = assistants.slice(0, uiTurnCount).map((message) => message.id);
          if (assistantMessageIds.length === uiTurnCount) {
            await waitFor(
              () => assistantMessageIds.every((messageId) =>
                document.querySelector(
                  '[data-vmlx-proof-message-id="' + CSS.escape(String(messageId)) + '"]'
                )
              ),
              uiTurnCount + ' assistant messages in the rendered chat DOM',
            );
            await waitFor(
              () => assistantMessageIds.every((messageId) => {
                const root = document.querySelector(
                  '[data-vmlx-proof-message-id="' + CSS.escape(String(messageId)) + '"]'
                );
                const answer = root?.querySelector('[data-vmlx-proof-answer="true"]');
                return (
                  answer?.getAttribute('data-vmlx-proof-answer-state') === 'complete'
                  && Number(answer.getAttribute('data-vmlx-proof-answer-rendered-length'))
                    === Number(answer.getAttribute('data-vmlx-proof-answer-full-length'))
                );
              }),
              uiTurnCount + ' assistant typewriter buffers to drain',
            );
          }
          for (const messageId of assistantMessageIds) {
            const root = document.querySelector(
              '[data-vmlx-proof-message-id="' + CSS.escape(String(messageId)) + '"]'
            );
            for (const rail of root?.querySelectorAll('[data-vmlx-proof-reasoning-rail="true"]') || []) {
              if (!rail.querySelector('[data-vmlx-proof-reasoning-content="true"]')) {
                rail.querySelector('button')?.click();
              }
            }
          }
          await new Promise((resolve) => setTimeout(resolve, 250));
          const renderedMessages = assistantMessageIds
            .map((messageId) => snapshotMessage(messageId, 'final'))
            .filter(Boolean);
          const renderedUserMessages = uiTurnEvidence
            .slice(0, uiTurnCount)
            .map((turn) => {
              const messageId = String(turn?.userMessageId || '');
              const root = messageId
                ? document.querySelector(
                    '[data-vmlx-proof-message-id="' + CSS.escape(messageId) + '"]'
                  )
                : null;
              if (!(root instanceof HTMLElement)) return null;
              const currencyOccurrences = [];
              const walker = document.createTreeWalker(root, NodeFilter.SHOW_TEXT);
              let node;
              while ((node = walker.nextNode())) {
                const text = node.nodeValue || '';
                let offset = text.indexOf('$43');
                while (offset >= 0) {
                  currencyOccurrences.push({
                    text: '$43',
                    insideKatex: Boolean(node.parentElement?.closest('.katex')),
                  });
                  offset = text.indexOf('$43', offset + 3);
                }
              }
              return {
                messageId,
                role: root.getAttribute('data-vmlx-proof-message-role') || '',
                visible: isVisible(root),
                text: (root.innerText || root.textContent || '').trim(),
                html: root.innerHTML || '',
                katexCount: root.querySelectorAll('.katex').length,
                katexErrorCount: root.querySelectorAll('.katex-error').length,
                currencyOccurrences,
              };
            })
            .filter(Boolean);
          const bodyText = document.body.innerText || '';
          const chromeClone = document.body.cloneNode(true);
          chromeClone.querySelectorAll(
            '[data-vmlx-proof-message-id], pre, code, script, style'
          ).forEach((element) => element.remove());
          const chromeText = chromeClone.innerText || '';
          const rawI18nKeys = [...new Set(
            chromeText.match(
              /\\b(?:[A-Za-z][A-Za-z0-9_-]*\\.){1,6}[A-Za-z][A-Za-z0-9_-]*\\b/g
            ) || []
          )];
          const visibleErrors = [...document.querySelectorAll('[role="alert"], .katex-error')]
            .filter(isVisible)
            .map((element) => (element.textContent || '').replace(/\\s+/g, ' ').trim())
            .filter((text) => /(?:failed|error|exception|traceback|invalid)/i.test(text));
          const renderedDom = {
            messages: renderedMessages,
            userMessages: renderedUserMessages,
            samples: domSamples,
            rawI18nKeys,
            visibleErrors,
            transientAlerts,
            chromeTextHead: chromeText.slice(0, 5000),
            bodyTextHead: bodyText.slice(0, 5000),
          };
          const first = assistants[0]?.content || '';
          const second = assistants[1]?.content || '';
          const third = assistants[2]?.content || '';
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
          const persistedOaiCallsByMessage = assistants.map((m) =>
            parsePersistedArray(m.toolCallsOaiJson)
          );
          const persistedOaiResultsByMessage = assistants.map((m) =>
            parsePersistedArray(m.toolResultsOaiJson)
          );
          const persistedReasoningSegments = persistedReasoningByMessage.flat();
          const persistedTools = persistedToolsByMessage.flat();
          const allAssistantText = assistants.map((m) => m.content || '').join('\\n');
          const visible = allAssistantText;
          // The media turns (image/video/audio) live PAST uiTurnCount, so they
          // appear in neither assistantRecords nor the DOM sample. Their answers
          // were therefore invisible in the artifact, which made an unmatched
          // media expect-regex impossible to diagnose without re-running the
          // model. Record the tail answers verbatim.
          const mediaTurnAnswers = assistants
            .slice(uiTurnCount)
            .map((m, index) => ({
              indexAfterUiTurns: index,
              messageId: m.id,
              length: String(m.content || '').length,
              text: String(m.content || '').slice(0, 600),
            }));
          const streamTraceByMessage = [...streamTraceState.values()];
          const messageEventTrace = Object.values(eventTrace.reduce((acc, event) => {
            const key = event.messageId || 'unknown';
            const row = acc[key] || { messageId: key, events: [] };
            row.events.push(event);
            acc[key] = row;
            return acc;
          }, {})).map((row) => ({
            ...row,
            events: row.events.slice().sort((left, right) => left.sequence - right.sequence),
          }));
          const persistedReasoningSchemaValid = persistedReasoningSegments
            .every((segment) => typeof segment === 'string');
          const reasoningText = persistedReasoningSegments
            .filter((segment) => typeof segment === 'string')
            .filter(Boolean)
            .join('\\n');
          // Closing tags are listed explicitly for every dialect, not just
          // think/tool_call. A Zaya-8B turn rendered </parameter>, </function>
          // and </zyphra_tool_call> as visible prose and this detector scored
          // it clean, because only the OPENING forms were listed. The paired
          // sanitizer never removed them either: it strips whole blocks, and an
          // orphan tail has no opening tag to pair with.
          const rawParserLeakRegex = /<think>|<\\/think>|<tool_calls?>|<\\/tool_calls?>|<function\\b|<\\/function>|<invoke\\b|<\\/invoke>|<parameter\\b|<\\/parameter>|<arg_key>|<\\/arg_key>|<arg_value>|<\\/arg_value>|<minimax:tool_call>|<\\/minimax:tool_call>|<zyphra_tool_call>|<\\/zyphra_tool_call>|<\\|point_start\\|>|<\\|point_end\\|>|<\\|box_start\\|>|<\\|box_end\\|>|<\\|tool_call_start\\|>|<\\|tool_call_end\\|>|\\[THINK\\]|\\[\\/THINK\\]|<mm:think>|<\\/mm:think>/i;
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
          // Audio rides as input_audio (or audio_url); see chat-utils.ts and the
          // main-process normaliser in ipc/chat.ts.
          const hasAudioAttachment = contentPartsByMessage.some((parts) =>
            parts.some((part) => (
              (part?.type === 'input_audio' && (part?.input_audio?.data || part?.audio?.data))
              || (part?.type === 'audio' && part?.audio?.data)
              || (part?.type === 'audio_url' && part?.audio_url?.url)
            ))
          );
          const imageSemanticVerified = checkMedia && new RegExp(imageExpectRegex, 'i').test(allAssistantText);
          const videoSemanticVerified = checkVideo && !!videoExpectRegex && new RegExp(videoExpectRegex, 'i').test(allAssistantText);
          // Same non-empty-regex rule as video: without an expectation there is
          // nothing to verify, and silently "passing" would be worse.
          const audioSemanticVerified = checkAudio && !!audioExpectRegex && new RegExp(audioExpectRegex, 'i').test(allAssistantText);
          const mediaEvidence = {
            requestedImage: checkMedia,
            requestedVideo: checkVideo,
            requestedAudio: checkAudio,
            imageExpectedRegex: imageExpectRegex,
            videoExpectedRegex: videoExpectRegex,
            audioExpectedRegex: audioExpectRegex,
            imageSemanticVerified,
            videoSemanticVerified,
            audioSemanticVerified,
            imageVerified: checkMedia && hasImageAttachment && imageSemanticVerified && !sendErrors.some((item) => item.turn === 4),
            videoVerified: checkVideo && hasVideoAttachment && videoSemanticVerified && !sendErrors.some((item) => item.turn === 5),
            audioVerified: checkAudio && hasAudioAttachment && audioSemanticVerified && !sendErrors.some((item) => item.turn === 6),
            persistedImageAttachment: hasImageAttachment,
            persistedVideoAttachment: hasVideoAttachment,
            persistedAudioAttachment: hasAudioAttachment,
            mediaTurnAnswers,
          };
          let effectiveSessionConfig = {};
          try {
            effectiveSessionConfig = JSON.parse(startedSession.config || '{}');
          } catch (_) {}
          const sessionLogs = await window.api.sessions.getLogs(created.session.id)
            .catch(() => []);
          return {
            rendererWireApi: wireApi,
            uiActionProfile,
            uiTurnCount,
            apiActionProfile,
            rendererBuiltinToolsEnabled: builtinToolsEnabled,
            rendererEnableThinking: enableThinking,
            workingDirectory,
            localSessionId: created.session.id,
            localSessionStarted: true,
            sessionType: created.session.type,
            effectiveSessionConfig,
            sessionLogs,
            preStartStopControl,
            serverModeControl: {
              label: (serverModeButton.textContent || '').replace(/\\s+/g, ' ').trim(),
              visible: true,
              clicked: true,
            },
            uiStartControl,
            gatewaySingleModelMode,
            ssdOnlyLaneSelection,
            nativeMtpSelection,
            chatId: chat.id,
            chatOverrides,
            rendererGenerationDefaults,
            chatSettingsDom,
            chatSettingsInteraction,
            midConvReasoningFlip,
            uiTurnEvidence,
            cacheRequestEvidence,
            sendErrors,
            rendererFailureStage,
            media: mediaEvidence,
            messageCount: messages.length,
            assistantCount: assistants.length,
            assistantMessageIds,
            assistantRecords: assistants.slice(0, uiTurnCount).map((message) => ({
              id: message.id,
              content: message.content || '',
            })),
            renderedDom,
            firstAssistantContent: first,
            secondAssistantContent: second,
            thirdAssistantContent: third,
            persistedReasoningByMessage,
            persistedToolsByMessage,
            persistedOaiCallsByMessage,
            persistedOaiResultsByMessage,
            persistedReasoningText: reasoningText,
            persistedReasoningCount: persistedReasoningSegments.length,
            persistedReasoningSchemaValid,
            persistedToolCount: persistedTools.length,
            streamTraceByMessage,
            messageEventTrace,
            turns: messages.map((m) => ({ role: m.role, content: m.content || '' })),
            rawParserLeak: rawParserLeakRegex.test(visible) || rawParserLeakRegex.test(reasoningText),
            reasoningRawParserLeak: rawParserLeakRegex.test(reasoningText),
            reasoningCjkLeakCount: countRegex(reasoningText, /[\\u3400-\\u9FFF]/g),
            reasoningKoreanLeakCount: countRegex(reasoningText, /[\\uAC00-\\uD7AF]/g),
            reasoningNumericRunCount: countRegex(reasoningText, numericRunRegex),
            // Degenerate numeric spew DOMINATES the text it appears in; a model
            // counting on purpose does not. Step37 wrote out "1 2 3 ... 21" to
            // check a byte count against the length of REAL_UI_LIVE_TOOL_ONE
            // inside 8.5k characters of otherwise coherent English, which
            // tripped the run detector on its own. Record how much of the
            // reasoning those runs actually occupy so the assertion can tell
            // spew from arithmetic instead of guessing from a raw count.
            // (No backticks in here: this whole block lives inside the in-page
            // evaluate() template literal and one would end the literal.)
            reasoningNumericRunChars: (() => {
              const matches = String(reasoningText || '').match(numericRunRegex);
              if (!matches) return 0;
              return matches.reduce((total, item) => total + item.length, 0);
            })(),
            reasoningTextLength: String(reasoningText || '').length,
            preloadHealthBefore,
            preloadHealthAfter,
            cacheBefore,
            cacheAfter: cacheAfterSettled,
            eventCounts: {
              stream: eventCounts.stream,
              tool: eventCounts.tool,
              reasoningDone: eventCounts.reasoningDone,
              complete: eventCounts.complete,
            },
          };
        } finally {
          captureAlerts();
          alertObserver.disconnect();
          cleanup.forEach((fn) => { try { fn(); } catch (_) {} });
          if (
            proofSessionId
            && privateConfigRestoreAdditionalArgs !== null
            && privateCacheAttestationArgs
          ) {
            const restored = await window.api.sessions.update(
              proofSessionId,
              { additionalArgs: privateConfigRestoreAdditionalArgs },
            );
            if (!restored?.success) {
              throw new Error(
                restored?.error
                || 'Failed to remove private cache attestation launch arguments',
              );
            }
          }
        }
      })()
    `, 1_200_000)
      healthBefore = rendererResult.preloadHealthBefore || {}
      proofGatewayStatus =
        rendererResult.gatewaySingleModelMode
          ?.gatewayStatusImmediatelyBeforeStart
        || proofGatewayStatus
      serverModels = await requestJson(`${baseUrl}/v1/models`, 5000)
    } catch (error) {
      const healthAfter = await requestJson(`${baseUrl}/health`, 5000)
        .catch((healthError) => ({ error: healthError.message }))
      const gitAfter = await captureGitProvenance()
      const healthProvenance = {
        before: createHealthSnapshot(`${baseUrl}/health`, healthBefore),
        after: createHealthSnapshot(`${baseUrl}/health`, healthAfter),
      }
      const uiRuntimeProvenance = captureUiRuntimeProvenance(
        app,
        rendererResourceEvidence,
        gitAfter,
        {
          cdpProcessBinding,
          backendProcessBinding,
          releaseManifest,
        },
      )
      let chatScreenshot = null
      try {
        chatScreenshot = await capturePng(
          cdp,
          path.join(proofDir, `${proofBasename}-chat.png`),
        )
      } catch {}
      const result = {
        format: proofFormat,
        run_id: runId,
        generatedAt: new Date().toISOString(),
        status: 'fail',
        surfaceStatus: 'partial_ui_only',
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
        requestedReasoningEffort: reasoningEffortOverride,
        requestedMidConvReasoningEffort: midConvReasoningEffortOverride,
        reasoningExpectation,
        requestedServerCacheControls: checkServerCacheControls,
        requestedBlockDiskCacheMaxPercent: blockDiskCacheMaxPercentOverride,
        requestedMedia: checkMedia,
        requestedVideo: checkVideo,
        requestedAudio: checkAudio,
        requestContract: {
          uiActionProfile,
          uiTurnCount,
          apiActionProfile,
          promptOne: selectedPromptOne,
          promptTwo: selectedPromptTwo,
          promptThree: selectedPromptThree,
          requestMaxTokens,
          requestMaxPromptTokens,
          maxToolIterations,
          toolResultMaxChars,
          wireApi,
          builtinToolsEnabled,
          enableThinking: enableThinkingOverride ?? null,
          reasoningExpectation,
          samplingOverrides: Object.fromEntries(
            Object.entries(samplingOverrides).filter(([, value]) => value !== undefined),
          ),
          checkServerCacheControls,
          nativeMtpMode: nativeMtpModeOverride ?? null,
          nativeMtpDepth: nativeMtpDepthOverride ?? null,
          checkMedia,
          checkVideo,
          expectPagedCacheLocked,
          imageExpectRegex,
          videoExpectRegex,
          cacheExpectRegex,
          pairedApiHoldSeconds,
        },
        baseUrl,
        userDataDir,
        workingDirectory,
        gitProvenance: {
          before: gitBefore,
          after: gitAfter,
        },
        healthProvenance,
        bundleGenerationContract,
        uiRuntimeProvenance,
        backend: {
          pid: healthProvenance.after.binding.backend_pid,
          base_url: baseUrl,
          model: servedModel,
          binding_before: healthProvenance.before.binding,
          binding_after: healthProvenance.after.binding,
        },
        backend_identity_fingerprint_sha256:
          healthProvenance.after.binding.fingerprint_sha256,
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
        messageEventTrace: [],
        sendErrors: [error?.message || String(error)],
        rendererFailureStage: 'renderer_real_ui_chat',
        appLogTail: appLogs.slice(-120),
        serverLogTail: [],
      }
      result.visibleAssistantTurnsComplete = visibleAssistantAfterEachUser(result.chat?.turns || [])
      result.liveSpeedSamples = extractLiveSpeedSamples(result)
      result.provenSurfaces = deriveProvenSurfaces(result)
      writePrivateArtifactFile(
        path.join(proofDir, `${proofBasename}-proof.json`),
        JSON.stringify(result, null, 2),
      )
      if (process.env.VMLINUX_REAL_UI_ALLOW_FAIL === '1' || process.env.VMLX_REAL_UI_ALLOW_FAIL === '1') {
        console.log(JSON.stringify({ ok: false, failures: [result.failureStage], result }, null, 2))
        return
      }
      throw error
    }
    const screenshotCapture = await captureRequiredScreenshot(
      (filePath) => capturePng(cdp, filePath),
      path.join(proofDir, `${proofBasename}-chat.png`),
    )
    rendererResult = mergeRequiredScreenshotOutcome(
      rendererResult,
      screenshotCapture,
    )
    let chatScreenshot = screenshotCapture.path
    let serverCacheControls = { requested: false, verified: false }
    if (checkServerCacheControls) {
      try {
        serverCacheControls = await evaluate(cdp, `
        (async () => {
          const isVisible = (element) => {
            if (!(element instanceof HTMLElement)) return false;
            const style = getComputedStyle(element);
            const rect = element.getBoundingClientRect();
            return style.display !== 'none'
              && style.visibility !== 'hidden'
              && Number(style.opacity || 1) !== 0
              && rect.width > 0
              && rect.height > 0;
          };
          const wait = (predicate, label, timeoutMs = 15000) => new Promise((resolve, reject) => {
            const started = Date.now();
            const tick = () => {
              try {
                const value = predicate();
                if (value) return resolve(value);
              } catch (_) {}
              if (Date.now() - started > timeoutMs) {
                return reject(new Error(
                  'timeout waiting for ' + label + ': ' + document.body.innerText.slice(0, 4000)
                ));
              }
              setTimeout(tick, 100);
            };
            tick();
          });
          const cacheExpectRegex = ${JSON.stringify(cacheExpectRegex)};
          const expectPagedCacheLocked = ${JSON.stringify(expectPagedCacheLocked)};
          const expectPagedCache = ${JSON.stringify(expectPagedCache)};
          const expectDsv4PoolQuant = ${JSON.stringify(expectDsv4PoolQuant ?? null)};
          const serverButton = await wait(() =>
            [...document.querySelectorAll(
              '[data-vmlx-control="server-settings"]'
            )].find((button) =>
              button instanceof HTMLButtonElement && isVisible(button)
            ) || null,
          'running-session Server settings control');
          serverButton.click();
          const drawer = await wait(() => {
            const candidate = document.querySelector(
              '[data-vmlx-surface="server-settings"]'
            );
            return candidate instanceof HTMLElement && isVisible(candidate)
              ? candidate
              : null;
          },
          'running-session Server Settings drawer');
          const sectionClickResults = [];
          const clickSection = async (title) => {
            const sectionButtons = [...(drawer?.querySelectorAll('button') || [])];
            const clickable = sectionButtons.find((button) => {
              const normalized = (button.innerText || '').replace(/\\s+/g, ' ').trim();
              return normalized === title || normalized.includes(title);
            });
            if (clickable) {
              clickable.scrollIntoView({ block: 'center' });
              clickable.click();
              await new Promise((resolve) => setTimeout(resolve, 150));
            }
            sectionClickResults.push({
              title,
              found: !!clickable,
              text: (clickable?.innerText || '').trim(),
            });
            return !!clickable;
          };
          await clickSection('Prefix Cache');
          await clickSection('In-Memory Paged Cache');
          await clickSection('Disk Cache');
          // MCP lives in its own collapsed section, so the first capture read
          // an unopened accordion and reported every MCP field absent. The
          // section's title is the i18n string 'Tool Integration (MCP)' — my
          // first guess, 'MCP Tools', is a different key entirely and matched
          // no button (sectionClickResults recorded found: false, which is why
          // that guess was visible rather than silent).
          await clickSection('Tool Integration (MCP)');
          // Optionally load a REAL MCP config and import it, so the proof shows
          // servers and tools actually discovered in the app rather than an
          // empty section. Typing the path is not enough — Import is what reads
          // the file and populates the discovered list, and a synthetic value
          // assignment without the native setter never reaches React.
          const mcpConfigPath = ${JSON.stringify(mcpConfigPath || '')};
          let mcpConfigLoad = null;
          if (mcpConfigPath) {
            const mcpInput = [...(drawer?.querySelectorAll('input') || [])]
              .find((i) => /mcp-config\\.json/i.test(i.placeholder || ''));
            const setter = Object.getOwnPropertyDescriptor(
              HTMLInputElement.prototype,
              'value',
            )?.set;
            let imported = false;
            if (mcpInput && setter) {
              setter.call(mcpInput, mcpConfigPath);
              mcpInput.dispatchEvent(new Event('input', { bubbles: true }));
              await new Promise((resolve) => setTimeout(resolve, 200));
              const importButton = [...(drawer?.querySelectorAll('button') || [])]
                .find((b) => /^import$/i.test((b.innerText || '').trim()));
              if (importButton) {
                importButton.scrollIntoView({ block: 'center' });
                importButton.click();
                imported = true;
                await new Promise((resolve) => setTimeout(resolve, 2500));
              }
            }
            mcpConfigLoad = {
              requestedPath: mcpConfigPath,
              inputFound: !!mcpInput,
              importClicked: imported,
              valueAfter: mcpInput ? String(mcpInput.value || '') : null,
            };
          }
          const labelFor = (text) => [...(drawer?.querySelectorAll('label') || [])]
            .find((label) => (label.innerText || '').includes(text));
          const inputFor = (text) => labelFor(text)?.querySelector('input[type="checkbox"]');
          const promptDiskInput = inputFor('Enable Disk Cache');
          const blockDiskInput = inputFor('Block Disk Cache (SSD / L2)');
          const pagedInput = inputFor('In-Memory Paged Cache (RAM)');
          const prefixInput = inputFor('Enable Prefix Cache')
            || inputFor('DSV4 Native Composite Prefix Cache');
          const initialCacheControls = {
            enablePrefixCache: !!prefixInput?.checked,
            usePagedCache: !!pagedInput?.checked,
            enableDiskCache: !!promptDiskInput?.checked,
            enableBlockDiskCache: !!blockDiskInput?.checked,
            usePagedCacheDisabled: !!pagedInput?.disabled,
            diskCachePresent: !!promptDiskInput,
            blockDiskCachePresent: !!blockDiskInput,
            blockDiskCacheMaxPercent: (() => {
              const setting = drawer?.querySelector(
                '[data-setting-label="SSD Cache Size (% of disk)"]'
              );
              return setting
                ? Number(setting.getAttribute('data-setting-value'))
                : null;
            })(),
          };
          const bodyText = drawer?.innerText || '';
          const labels = [
            'Enable Prefix Cache',
            'DSV4 Native Composite Prefix Cache',
            'In-Memory Paged Cache (RAM)',
            'Block Disk Cache (SSD / L2)',
            'Enable Disk Cache',
            'Stored Cache Quantization',
          ]
            .filter((label) => bodyText.includes(label));
          const cacheExpectationMatches = !cacheExpectRegex || new RegExp(cacheExpectRegex, 'i').test(bodyText);
          // The paged comparison is deliberately NOT folded in here: it is
          // settled outside this page context against the engine's live
          // /health native_cache, which is only available after this drawer
          // read. See the pagedCache parity block below.
          const verified = initialCacheControls.enablePrefixCache === true
            && (
              (initialCacheControls.enableBlockDiskCache === true
                && initialCacheControls.blockDiskCachePresent === true)
              || (initialCacheControls.enableDiskCache === true
                && initialCacheControls.diskCachePresent === true)
            )
            && !!prefixInput
            && !!pagedInput
            && (!expectPagedCacheLocked || initialCacheControls.usePagedCacheDisabled === true)
            && cacheExpectationMatches;
          const close = [...(drawer?.querySelectorAll('button') || [])]
            .find((button) => button.getAttribute('aria-label') === 'Close');
          close?.click();
          return {
            requested: true,
            verified,
            runningSessionDrawer: true,
            controlScope: 'running-session-toolbar',
            visibleBlockDiskChecked: initialCacheControls.enableBlockDiskCache,
            visiblePromptDiskChecked: initialCacheControls.enableDiskCache,
            cacheExpectRegex,
            expectPagedCacheLocked,
            expectPagedCache,
            expectDsv4PoolQuant,
            cacheExpectationMatches,
            labels,
            initialCacheControls,
            sectionClickResults,
            // Does the UI actually SHOW the MTP surface? An MTP bundle must
            // render the Native MTP controls; the harness never looked, so
            // "MTP visible in the app" had only ever been inferred from the
            // registry. Captured from the same open drawer as the cache
            // controls: whether the labelled control exists, which mode is
            // selected, and whether the blocked-fallback notice is up
            // (weights detected but the runtime compatibility gate not passed).
            nativeMtpControl: (() => {
              const label = [...(drawer?.querySelectorAll('label') || [])]
                .find((l) => /native mtp/i.test(l.innerText || ''));
              const select = [...(drawer?.querySelectorAll('select') || [])]
                .find((sel) => [...sel.options].some(
                  (o) => /auto|deterministic|off/i.test(o.value || o.textContent || ''),
                ) && /mtp/i.test((sel.closest('label,div')?.innerText || '')));
              return {
                labelVisible: !!label,
                // \\s, not \s: this string is inside the outer evaluate()
                // template literal, so a single backslash collapses and the
                // page runs /s+/g — which replaces every letter "s" with a
                // space. That produced "Determini tic override" and
                // "Auto (bundle default )" and nearly had me report a UI typo
                // that does not exist.
                labelText: (label?.innerText || '').replace(/\\s+/g, ' ').trim().slice(0, 120),
                modeSelectPresent: !!select,
                selectedMode: select ? String(select.value || '') : null,
                modeOptions: select ? [...select.options].map((o) => String(o.value)) : null,
                blockedFallbackNoticeShown: /native mtp weights were detected/i.test(bodyText),
                mentionedInDrawer: /native mtp/i.test(bodyText),
              };
            })(),
            // MCP had ZERO coverage in this harness — every MCP claim so far
            // came from the policy-contract artifact and the API, never from
            // the app. Same gap that audio and the SSD-only lane had. Captured
            // from the same open drawer as the cache and MTP controls: the
            // config-file field, both policy textareas (servers / tools) and
            // any discovered server rows with their transport, state and tool
            // count. Read-only for now; an assertion follows once the shape is
            // observed both with MCP configured and without.
            // (No backticks anywhere in here: this block is inside the in-page
            // evaluate() template literal and one would end the literal.)
            mcpConfigLoad,
            mcpControls: (() => {
              const labels = [...(drawer?.querySelectorAll('label') || [])];
              const configLabel = labels.find((l) => /mcp config file/i.test(l.innerText || ''));
              const serversLabel = labels.find((l) => /enabled mcp servers|mcp servers/i.test(l.innerText || ''));
              const toolsLabel = labels.find((l) => /enabled mcp tools|mcp tools/i.test(l.innerText || ''));
              const configInput = [...(drawer?.querySelectorAll('input') || [])]
                .find((i) => /mcp-config\\.json/i.test(i.placeholder || ''));
              const areas = [...(drawer?.querySelectorAll('textarea') || [])];
              const serversArea = areas.find((a) => /filesystem,github/i.test(a.placeholder || ''));
              const toolsArea = areas.find((a) => /filesystem__read_file/i.test(a.placeholder || ''));
              // The DISABLED lists are the ones the policy actually enforces
              // server-side (mcp_policy_filters_servers_tools_before_schema_merge),
              // so capture them too rather than only the allow-lists.
              const disabledServersArea = areas.find((a) => /browser_automation/i.test(a.placeholder || ''));
              const disabledToolsArea = areas.find((a) => /filesystem__write_file/i.test(a.placeholder || ''));
              const serverRows = [...(drawer?.querySelectorAll('span') || [])]
                .filter((s) => /\\u00b7/.test(s.innerText || '') && /(stdio|http|sse|mcp)/i.test(s.innerText || ''))
                .map((s) => (s.innerText || '').replace(/\\s+/g, ' ').trim())
                .slice(0, 6);
              return {
                sectionVisible: /mcp tools/i.test(bodyText),
                configFieldVisible: !!configLabel,
                configInputPresent: !!configInput,
                configValue: configInput ? String(configInput.value || '') : null,
                enabledServersFieldVisible: !!serversLabel,
                enabledServersPresent: !!serversArea,
                enabledServersValue: serversArea ? String(serversArea.value || '') : null,
                enabledToolsFieldVisible: !!toolsLabel,
                enabledToolsPresent: !!toolsArea,
                enabledToolsValue: toolsArea ? String(toolsArea.value || '') : null,
                disabledServersPresent: !!disabledServersArea,
                disabledServersValue: disabledServersArea ? String(disabledServersArea.value || '') : null,
                disabledToolsPresent: !!disabledToolsArea,
                disabledToolsValue: disabledToolsArea ? String(disabledToolsArea.value || '') : null,
                discoveredServerRows: serverRows,
              };
            })(),
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
    backendProcessBinding = await captureListenerProcessBinding({
      port: serverPort,
      expectedRootPid: null,
      expectedHealthPid: runtimeBindingFromHealth(healthAfter).backend_pid,
      kind: 'vmlx-python-backend',
    })
    const gitAfter = await captureGitProvenance()
    const healthProvenance = {
      before: createHealthSnapshot(`${baseUrl}/health`, healthBefore),
      after: createHealthSnapshot(`${baseUrl}/health`, healthAfter),
    }
    const uiRuntimeProvenance = captureUiRuntimeProvenance(
      app,
      rendererResourceEvidence,
      gitAfter,
      {
        cdpProcessBinding,
        backendProcessBinding,
        releaseManifest,
      },
    )
    rendererResult.cacheRequestEvidence = (
      rendererResult.cacheRequestEvidence || []
    ).map((row) => {
      const artifactValue = {
        schema: 'vmlx-ui-turn-health-cache-execution-v1',
        run_id: runId,
        turn: row.turn,
        proof_request_id: row.proofRequestId,
        user_message_id: row.userMessageId,
        assistant_message_id: row.assistantMessageId,
        terminal_response_id: row.terminalResponseId,
        correlation_status: row.correlationStatus,
        health: row.healthAfter,
      }
      const artifactBytes = Buffer.from(canonicalJson(artifactValue))
      const artifactPath = path.join(
        proofDir,
        `${proofBasename}-turn-${Number(row.turn)}-health.json`,
      )
      writePrivateArtifactFile(artifactPath, artifactBytes)
      return {
        ...row,
        healthArtifact: {
          path: path.resolve(artifactPath),
          sha256: sha256File(artifactPath),
          size_bytes: artifactBytes.length,
        },
      }
    })
    const resolvedSamplingRecords = (rendererResult.uiTurnEvidence || [])
      .flatMap((turn) => parseResolvedSamplingKwargs(turn.logLines || [])
        .map((record) => ({
          ...record,
          observed_window_turn: turn.turn,
        })))
    const resolvedSamplingKwargs = resolvedSamplingRecords.at(-1)?.values || {}
    const cacheRequestCorrelation = {
      status: (rendererResult.cacheRequestEvidence || []).length
        === expectedUiTurnCount({
          requestContract: { uiTurnCount },
        })
        && (rendererResult.cacheRequestEvidence || []).every(
          (row) => row.correlationStatus === 'verified',
        )
        ? 'verified'
        : 'partial',
      source:
        'chat:complete.responseId == health.scheduler.last_cache_execution.request_id',
      turns: (rendererResult.cacheRequestEvidence || []).map((row) => ({
        turn: row.turn,
        proofRequestId: row.proofRequestId,
        userMessageId: row.userMessageId,
        assistantMessageId: row.assistantMessageId,
        terminalResponseId: row.terminalResponseId,
        executionRequestId: row.executionRequestId,
        correlationStatus: row.correlationStatus,
        healthArtifact: row.healthArtifact,
      })),
    }
    const requestCorrelationTurns = (rendererResult.uiTurnEvidence || []).map((turn) => {
      const cacheRow = (rendererResult.cacheRequestEvidence || []).find(
        (row) => Number(row?.turn) === Number(turn?.turn),
      ) || {}
      const serverRequestIds = [...new Set(
        (Array.isArray(turn?.requestIds) ? turn.requestIds : [])
          .map((value) => String(value || ''))
          .filter(Boolean),
      )]
      const matchingRecords = resolvedSamplingRecords.filter((record) => (
        String(record?.proof_request_id || '') === String(turn?.proofRequestId || '')
        && String(record?.message_id || '') === String(turn?.assistantMessageId || '')
      ))
      const recordRequestIds = matchingRecords.map(
        (record) => String(record?.request_id || ''),
      )
      const resolvedLogCorrelated = Boolean(
        turn?.proofRequestId
        && String(turn.proofRequestId) === String(turn.userMessageId || '')
        && String(turn.terminalProofRequestId || '') === String(turn.proofRequestId)
        && String(turn.terminalMessageId || '') === String(turn.assistantMessageId || '')
        && turn.logMatchMode === 'exact_identity_ring_safe'
        && serverRequestIds.length > 0
        && matchingRecords.length === serverRequestIds.length
        && new Set(recordRequestIds).size === serverRequestIds.length
        && serverRequestIds.every((requestId) => recordRequestIds.includes(requestId))
      )
      return {
        turn: turn.turn,
        proofRequestId: turn.proofRequestId,
        userMessageId: turn.userMessageId,
        assistantMessageId: turn.assistantMessageId,
        serverProofRequestId: turn.terminalProofRequestId,
        serverRequestIds,
        serverMessageId: turn.terminalMessageId,
        resolvedLogCorrelated,
        cacheObservationCorrelated: cacheRow.correlationStatus === 'verified',
      }
    })
    const requestCorrelation = {
      status: (
        requestCorrelationTurns.length === uiTurnCount
        && requestCorrelationTurns.every((turn) => turn.resolvedLogCorrelated === true)
        && new Set(requestCorrelationTurns.flatMap((turn) => turn.serverRequestIds)).size
          === requestCorrelationTurns.flatMap((turn) => turn.serverRequestIds).length
      ) ? 'verified' : 'partial',
      source:
        'chat:complete proofRequestId/requestIds/messageId matched exactly to server-emitted sampling identities',
      turns: requestCorrelationTurns,
    }
    if (checkServerCacheControls) {
      const commandLine = [...(rendererResult.sessionLogs || [])]
        .reverse()
        .find((line) => String(line).includes('] $ '))
        || ''
      const argv = [...String(commandLine).matchAll(/--[A-Za-z0-9-]+/g)]
        .map((match) => match[0])
      serverCacheControls = {
        ...serverCacheControls,
        commandLine,
        argv,
        persistedConfig: rendererResult.effectiveSessionConfig || {},
        healthNativeCache: healthAfter?.native_cache || {},
      }
      // Visible paged toggle vs the engine that is actually running. An
      // explicit expectation still wins so a run can assert an intended
      // configuration; otherwise the engine's own report is the expectation,
      // so a paged-default-ON family cannot fail merely because the
      // invocation omitted a flag.
      const enginePagedCache = healthAfter?.native_cache?.paged === true
      const pagedCacheExpected = expectPagedCacheExplicit
        ? expectPagedCache
        : enginePagedCache
      const visiblePagedCache = (
        serverCacheControls.initialCacheControls?.usePagedCache === true
      )
      const nativeCacheAfter = healthAfter?.native_cache || {}
      const promptDiskOnly = nativeCacheAfter.prompt_disk_l2_configured === true
        && nativeCacheAfter.block_disk_l2_configured !== true
      const diskLaneVerified = promptDiskOnly
        ? rendererResult.effectiveSessionConfig?.enableDiskCache === true
          && rendererResult.effectiveSessionConfig?.enableBlockDiskCache === false
          && argv.includes('--enable-disk-cache')
          && !argv.includes('--enable-block-disk-cache')
          && nativeCacheAfter.prompt_disk_l2 === true
          && nativeCacheAfter.block_disk_l2 !== true
        : rendererResult.effectiveSessionConfig?.enableBlockDiskCache === true
          && argv.includes('--enable-block-disk-cache')
          && nativeCacheAfter.block_disk_l2 === true
      serverCacheControls = {
        ...serverCacheControls,
        promptDiskOnly,
        diskLaneVerified,
        pagedCacheExpected,
        pagedCacheExpectationSource: expectPagedCacheExplicit
          ? 'explicit_env'
          : 'engine_health',
        enginePagedCache,
        visiblePagedCache,
        pagedCacheParity: visiblePagedCache === pagedCacheExpected,
      }
      serverCacheControls.verified = (
        serverCacheControls.verified === true
        && serverCacheControls.pagedCacheParity === true
        && diskLaneVerified
      )
    }
    const visibleText = rendererResult.thirdAssistantContent
      || rendererResult.secondAssistantContent
      || rendererResult.firstAssistantContent
      || ''
    const cacheHitTokens = (rendererResult.cacheRequestEvidence || [])
      .reduce((total, evidence) => {
        const before = explicitCacheCounters(evidence?.before)
        const after = explicitCacheCounters(evidence?.after)
        return total + Math.max(0, after.hitTokens - before.hitTokens)
      }, 0)
    const toolProbeFiles = {}
    const toolProbePaths = []
    for (const name of ['real_ui_tool_probe_1.txt', 'real_ui_tool_probe_2.txt']) {
      const filePath = path.join(workingDirectory, name)
      toolProbePaths.push(filePath)
      if (existsSync(filePath)) {
        toolProbeFiles[name] = readFileSync(filePath, 'utf8')
      }
    }
    for (const filePath of toolProbePaths) rmSync(filePath, { force: true })
    const toolProbeCleanup = {
      removed: toolProbePaths.every((filePath) => !existsSync(filePath)),
      paths: toolProbePaths,
    }
    const uiBackendBinding = {
      format: 'vmlx-ui-backend-binding-v3',
      run_id: runId,
      generated_at: new Date().toISOString(),
      base_url: baseUrl,
      gateway_base_url: proofGatewayStatus
        ? `http://127.0.0.1:${proofGatewayStatus.port}`
        : releaseGatewayBaseUrl || null,
      gateway_status: proofGatewayStatus,
      gateway_process_binding: gatewayProcessBinding,
      backend_pid: healthProvenance.after.binding.backend_pid,
      backend_identity_fingerprint_sha256:
        healthProvenance.after.binding.fingerprint_sha256,
      binding: healthProvenance.after.binding,
      runtime_source_hashes:
        healthProvenance.after.binding.runtime_source_hashes,
      served_model: servedModel,
      local_session_id: rendererResult.localSessionId || null,
      chat_id: rendererResult.chatId || null,
      model_bundle_fingerprint_sha256:
        healthProvenance.after.binding.model_bundle_fingerprint_sha256,
      cache_topology_fingerprint_sha256:
        healthProvenance.after.binding.cache_topology_fingerprint_sha256,
      source_commit: gitAfter.commit,
      source_tree: gitAfter.tree,
      cdp_url: attachCdp?.origin || `http://127.0.0.1:${debugPort}`,
      electron_pid: cdpProcessBinding?.listener_pid || null,
      electron_process_binding: cdpProcessBinding,
      electron_lifecycle_owner: app.lifecycleOwner || 'ui-proof-child',
      electron_attached: app.attached === true,
      electron_teardown_allowed: app.allowTeardown !== false,
      hold_seconds: pairedApiHoldSeconds,
      release_sentinel_path: releaseSentinelPath
        ? path.resolve(releaseSentinelPath)
        : null,
      release_nonce_sha256: releaseSentinelNonce
        ? sha256Text(releaseSentinelNonce)
        : null,
      expected_run_intent_path: releaseRunIntentPath
        ? realpathSync(releaseRunIntentPath)
        : null,
      expected_run_intent_sha256: releaseRunIntentSha256 || null,
      expected_run_intent_canonical_sha256:
        ownedRunIntent?.value?.canonical_sha256 || null,
      expected_release_phase: activeReleasePhase,
      expected_session_attestation_path: releaseSessionAttestationPath
        ? path.resolve(releaseSessionAttestationPath)
        : null,
      expected_paired_cache_artifact_path: pairedCacheArtifactPath
        ? path.resolve(pairedCacheArtifactPath)
        : null,
      live_during_hold: pairedApiHoldSeconds > 0 || Boolean(releaseSentinelPath),
    }
    const proofPath = path.join(proofDir, `${proofBasename}-proof.json`)
    const bindingPath = path.join(proofDir, `${proofBasename}-ui-backend-binding.json`)
    writePrivateArtifactFile(bindingPath, JSON.stringify(uiBackendBinding, null, 2))
    const bindingSha256 = sha256File(bindingPath)
    let uiSessionAttestation = null
    if (releaseSentinelPath) {
      const attestationPath = path.resolve(releaseSessionAttestationPath)
      const existingAncestor = realpathSync(nearestExistingDirectory(attestationPath))
      if (
        isPathInside(attestationPath, realpathSync(repoDir))
        || isPathInside(existingAncestor, realpathSync(repoDir))
      ) {
        throw new Error('Owned UI session attestation must stay outside the public Git worktree')
      }
      if (existsSync(attestationPath)) {
        throw new Error('Owned UI session attestation path already exists')
      }
      const intentBundle = realpathSync(activeReleasePhase.model_bundle_path)
      const runtimeBundle = realpathSync(modelPath)
      const runtimeBundleFingerprint =
        healthProvenance.after.binding.model_bundle_fingerprint_sha256
      if (
        intentBundle !== runtimeBundle
        || activeReleasePhase.bundle_fingerprint_sha256
          !== runtimeBundleFingerprint
      ) {
        throw new Error(
          'Owned UI session attestation model bundle does not match live backend provenance',
        )
      }
      const sessionId = String(rendererResult.localSessionId || '')
      const backendPid = Number(healthProvenance.after.binding.backend_pid)
      const electronPid = Number(cdpProcessBinding?.listener_pid)
      if (
        !sessionId
        || !Number.isInteger(backendPid)
        || backendPid <= 0
        || !Number.isInteger(electronPid)
        || electronPid <= 0
      ) {
        throw new Error('Owned UI session attestation has incomplete PID/session binding')
      }
      const attestationValue = {
        schema: ownedUiSessionAttestationSchema,
        run_id: runId,
        nonce: releaseSentinelNonce,
        run_intent_sha256: releaseRunIntentSha256,
        phase_index: activeReleasePhase.phase_index,
        phase_name: activeReleasePhase.phase_name,
        representative_id: activeReleasePhase.representative_id,
        bundle_role: activeReleasePhase.bundle_role,
        cache_policy: activeReleasePhase.cache_policy,
        paged_ram: activeReleasePhase.paged_ram,
        ui_action_profile: activeReleasePhase.ui_action_profile,
        ui_turn_count: activeReleasePhase.ui_turn_count,
        api_action_profile: activeReleasePhase.api_action_profile,
        // The Python worker is the process owned and observed by the release
        // orchestrator. This Node harness is its directly spawned child, so
        // bind the independently written attestation to the worker's
        // kernel-observed PID rather than claiming the harness PID is the
        // orchestrator-owned producer.
        ui_producer_pid: ownedUiProducerPid({ orchestrated: true }),
        session_id: sessionId,
        model: activeReleasePhase.model,
        model_bundle_path: intentBundle,
        bundle_fingerprint_sha256:
          activeReleasePhase.bundle_fingerprint_sha256,
        backend_pid: backendPid,
        gateway_pid: releaseGatewayPid,
        direct_base_url: ownedRunIntentDirectEndpoint(
          ownedRunIntent.value,
          activeReleasePhase.phase_index,
        ).baseUrl,
        gateway_base_url: ownedRunIntent.value.gateway_base_url,
        electron_pid: electronPid,
        cdp_origin: attachCdp.origin,
        lifecycle_owner: lifecycleOwner,
        source_commit: gitAfter.commit,
        source_tree: gitAfter.tree,
        renderer_source_sha256:
          rendererResourceEvidence.servedRendererSourceSha256,
        session_binding_sha256: bindingSha256,
        created_at: new Date().toISOString(),
      }
      writePrivateArtifactFile(
        attestationPath,
        JSON.stringify(attestationValue),
      )
      uiSessionAttestation = readPrivateExternalJson(
        attestationPath,
        'Owned UI session attestation',
      )
    }
    const releaseNotBeforeMs = uiSessionAttestation
      ? lstatSync(uiSessionAttestation.path).mtimeMs
      : lstatSync(bindingPath).mtimeMs
    let releaseEvidence = null
    if (pairedApiHoldSeconds > 0 || releaseSentinelPath) {
      console.log(JSON.stringify({
        ui_only_ready_for_separate_api_run: true,
        run_id: runId,
        base_url: baseUrl,
        gateway_base_url: uiBackendBinding.gateway_base_url,
        backend_pid: healthProvenance.after.binding.backend_pid,
        backend_identity_fingerprint_sha256:
          healthProvenance.after.binding.fingerprint_sha256,
        ui_backend_binding_path: bindingPath,
        ui_backend_binding_sha256: bindingSha256,
        cdp_url: uiBackendBinding.cdp_url,
        electron_pid: uiBackendBinding.electron_pid,
        expected_paired_api_artifact_path: path.resolve(pairedApiArtifactPath),
        expected_paired_cache_artifact_path: pairedCacheArtifactPath
          ? path.resolve(pairedCacheArtifactPath)
          : null,
        expected_run_intent_path: ownedRunIntent?.path || null,
        expected_run_intent_sha256: releaseRunIntentSha256 || null,
        expected_ui_session_attestation_path:
          uiSessionAttestation?.path || null,
        expected_ui_session_attestation_sha256:
          uiSessionAttestation?.sha256 || null,
        expected_retained_pids: releaseRetainedPids,
        release_phase: activeReleasePhase,
        required_separate_api_artifact: true,
        hold_seconds: pairedApiHoldSeconds,
        release_sentinel_path: releaseSentinelPath
          ? path.resolve(releaseSentinelPath)
          : null,
        release_sentinel_timeout_seconds: releaseSentinelTimeoutSeconds,
      }))
      if (releaseSentinelPath) {
        releaseEvidence = await waitForOwnedUiReleaseSentinel({
          filePath: releaseSentinelPath,
          runId,
          nonce: releaseSentinelNonce,
          sessionId: rendererResult.localSessionId,
          orchestrated: true,
          runIntentPath: releaseRunIntentPath,
          runIntentSha256: releaseRunIntentSha256,
          uiSessionAttestationPath: releaseSessionAttestationPath,
          uiSessionAttestationSha256: uiSessionAttestation.sha256,
          activePhase: activeReleasePhase,
          apiArtifactPath: pairedApiArtifactPath,
          cacheArtifactPath: pairedCacheArtifactPath,
          notBeforeMs: releaseNotBeforeMs,
          timeoutMs: releaseSentinelTimeoutSeconds * 1000,
        })
      } else {
        await sleep(pairedApiHoldSeconds * 1000)
      }
    }
    let finalPhaseStopEvidence = null
    const finalPhaseStopRequired = Boolean(
      releaseEvidence
      && activeReleasePhase?.phase_index === 5
      && releaseActivePhaseIndex === 5
    )
    const pairedApiArtifact = await runPostSentinelWorkWithCleanup({
      work: async () => {
        const artifact = pairedApiArtifactPath
          ? readPrivateExternalJson(
              pairedApiArtifactPath,
              'Paired raw API proof artifact',
            )
          : null
        if (
          releaseEvidence
          && artifact?.sha256 !== releaseEvidence.api_capture_sha256
        ) {
          throw new Error(
            'paired API artifact changed after the owned release sentinel was consumed',
          )
        }
        return artifact
      },
      cleanup: async () => {
        if (!finalPhaseStopRequired) return
        const sessionId = String(rendererResult.localSessionId || '')
        const backendPid = Number(healthProvenance.after.binding.backend_pid)
        let survivorsBefore = null
        await runPostSentinelWorkWithCleanup({
          work: async () => {
            survivorsBefore = await attestExactSurvivorPids({
              backendPid,
              electronPid: expectedElectronPid,
              gatewayPid: releaseGatewayPid,
              retainedPids: releaseRetainedPids,
              stage: 'before_visible_stop',
            })
          },
          cleanup: async () => {
            const visibleStop = await evaluate(cdp, `
          (async () => {
            const sessionId = ${JSON.stringify(String(rendererResult.localSessionId || ''))};
            const expectedBackendPid = ${JSON.stringify(Number(healthProvenance.after.binding.backend_pid))};
            const expectedBackendPort = ${JSON.stringify(Number(serverPort))};
            const isVisible = (element) => {
              if (!(element instanceof HTMLElement)) return false;
              const style = getComputedStyle(element);
              const rect = element.getBoundingClientRect();
              return style.display !== 'none'
                && style.visibility !== 'hidden'
                && Number(style.opacity || 1) !== 0
                && rect.width > 0
                && rect.height > 0;
            };
            const waitFor = (predicate, label, timeoutMs = 120000) =>
              new Promise((resolve, reject) => {
                const started = Date.now();
                const check = async () => {
                  try {
                    const value = await predicate();
                    if (value) {
                      resolve(value);
                      return;
                    }
                  } catch (_) {}
                  if (Date.now() - started > timeoutMs) {
                    reject(new Error('Timed out waiting for ' + label));
                    return;
                  }
                  setTimeout(check, 100);
                };
                check();
              });
            window.dispatchEvent(new CustomEvent('vmlx:navigate', {
              detail: {
                mode: 'server',
                panel: 'session',
                sessionId,
              },
            }));
            const before = await window.api.sessions.get(sessionId);
            if (
              !before
              || !['running', 'standby'].includes(before.status)
              || Number(before.pid) !== expectedBackendPid
              || Number(before.port) !== expectedBackendPort
            ) {
              throw new Error(
                'Final V5 visible Stop requires the exact proof session '
                + 'to be running or standby and bound to its health PID/port'
              );
            }
            const exactSelector =
              '[data-vmlx-control="session-stop"][data-vmlx-session-id="'
              + CSS.escape(sessionId)
              + '"]';
            const control = await waitFor(() => {
              const candidate = document.querySelector(exactSelector);
              return candidate instanceof HTMLButtonElement && isVisible(candidate)
                ? candidate
                : null;
            }, 'the exact proof session visible Stop control');
            const label = (control.textContent || '').replace(/\\s+/g, ' ').trim();
            control.scrollIntoView({ block: 'center' });
            control.click();
            const after = await waitFor(async () => {
              const current = await window.api.sessions.get(sessionId);
              return current?.status === 'stopped' ? current : null;
            }, 'the exact proof session database state to become stopped');
            if (
              after.pid != null
              || Number(after.port) !== expectedBackendPort
            ) {
              throw new Error(
                'Final V5 stopped session did not clear its nullable PID '
                + 'while retaining its non-null endpoint port'
              );
            }
            return {
              visibleControl: {
                selector: '[data-vmlx-control="session-stop"]',
                exactSelector,
                sessionId,
                label,
                visible: true,
                clicked: true,
              },
              session: {
                id: sessionId,
                before: {
                  status: before.status,
                  pid: Number(before.pid),
                  port: Number(before.port),
                },
                after: {
                  status: after.status,
                  pid: after.pid ?? null,
                  port: Number(after.port),
                },
                pidClearSemantics: 'nullable_pid_cleared',
                portClearSemantics: 'non_nullable_endpoint_retained',
              },
            };
          })()
        `, 180_000)
        if (
          visibleStop?.session?.id !== sessionId
          || visibleStop?.session?.after?.status !== 'stopped'
        ) {
          throw new Error('Final V5 visible Stop did not attest the exact proof session')
        }
        const backendTeardown = await waitForExactProofBackendTeardown({
          backendPid,
          port: serverPort,
        })
        const survivorsAfter = await attestExactSurvivorPids({
          backendPid,
          electronPid: expectedElectronPid,
          gatewayPid: releaseGatewayPid,
          retainedPids: releaseRetainedPids,
          stage: 'after_backend_teardown',
        })
            finalPhaseStopEvidence = {
              releaseSentinelConsumed: true,
              phaseIndex: 5,
              ...visibleStop,
              backend: backendTeardown,
              survivors: {
                before: survivorsBefore,
                after: survivorsAfter,
              },
            }
          },
        })
      },
    })
    const result = {
      format: proofFormat,
      run_id: runId,
      generatedAt: new Date().toISOString(),
      status: rendererResult.rendererFailureStage ? 'fail' : 'partial',
      pass: false,
      surfaceStatus: pairedApiArtifact
        ? 'partial_dual_surface_uncorrelated'
        : 'partial_ui_only',
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
      requestedReasoningEffort: reasoningEffortOverride,
      requestedMidConvReasoningEffort: midConvReasoningEffortOverride,
      reasoningExpectation,
      requestedServerCacheControls: checkServerCacheControls,
      requestedBlockDiskCacheMaxPercent: blockDiskCacheMaxPercentOverride,
      expectedDsv4PoolQuant: expectDsv4PoolQuant,
      requestedMedia: checkMedia,
      requestedVideo: checkVideo,
      requestedAudio: checkAudio,
      requestedRetainedPids: releaseRetainedPids,
      requestContract: {
        uiActionProfile,
        uiTurnCount,
        apiActionProfile,
        promptOne: selectedPromptOne,
        promptTwo: selectedPromptTwo,
        promptThree: selectedPromptThree,
        requestMaxTokens,
        requestMaxPromptTokens,
        maxToolIterations,
        toolResultMaxChars,
        wireApi,
        builtinToolsEnabled,
        enableThinking: enableThinkingOverride ?? null,
        reasoningExpectation,
        samplingOverrides: Object.fromEntries(
          Object.entries(samplingOverrides).filter(([, value]) => value !== undefined),
        ),
        checkServerCacheControls,
        nativeMtpMode: nativeMtpModeOverride ?? null,
        nativeMtpDepth: nativeMtpDepthOverride ?? null,
        checkMedia,
        checkVideo,
        expectPagedCacheLocked,
        expectDsv4PoolQuant,
        imageExpectRegex,
        videoExpectRegex,
        cacheExpectRegex,
        pairedApiHoldSeconds,
      },
      baseUrl,
      userDataDir,
      workingDirectory,
      uiLaunchMode: app.uiLaunchMode,
      uiCommand: app.command,
      installedAppPath: app.appPath || undefined,
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
        // Carry the measurements too, or reasoningNumericRunIsSpew() falls back
        // to its count-only rule and the refinement is inert: the in-page
        // capture recorded chars 66 of textLength 10153 while this object still
        // handed the validator an undefined, so a step37 run kept failing with
        // the numbers that exonerate it sitting in the artifact.
        reasoningNumericRunChars: rendererResult.reasoningNumericRunChars || 0,
        reasoningTextLength: rendererResult.reasoningTextLength
          || String(rendererResult.persistedReasoningText || '').length,
      },
      screenshots: {
        chat: chatScreenshot,
      },
      ...rendererResult,
      gitProvenance: {
        before: gitBefore,
        after: gitAfter,
      },
      healthProvenance,
      bundleGenerationContract,
      uiRuntimeProvenance,
      resolvedSamplingRecords,
      resolvedSamplingKwargs,
      requestCorrelation,
      cacheRequestCorrelation,
      backend_identity_fingerprint_sha256:
        healthProvenance.after.binding.fingerprint_sha256,
      backend: {
        pid: healthProvenance.after.binding.backend_pid,
        base_url: baseUrl,
        model: servedModel,
        binding_before: healthProvenance.before.binding,
        binding_after: healthProvenance.after.binding,
      },
      uiBackendBinding,
      ownedRunIntent: ownedRunIntent
        ? {
            path: ownedRunIntent.path,
            sha256: ownedRunIntent.sha256,
            canonical_sha256: ownedRunIntent.value.canonical_sha256,
            phase_index: activeReleasePhase.phase_index,
            phase_name: activeReleasePhase.phase_name,
          }
        : null,
      uiSessionAttestation: uiSessionAttestation
        ? {
            path: uiSessionAttestation.path,
            sha256: uiSessionAttestation.sha256,
            value: uiSessionAttestation.value,
          }
        : null,
      releaseEvidence,
      finalPhaseStopEvidence,
      pairedApiArtifact,
      session: {
        id: rendererResult.localSessionId || null,
        type: rendererResult.sessionType || null,
        effective_config: rendererResult.effectiveSessionConfig || {},
      },
      serverCacheControls,
      toolProbeFiles,
      toolProbeCleanup,
      streamTrace: rendererResult.streamTraceByMessage || [],
      appLogTail: appLogs.slice(-80),
      serverLogTail: rendererResult.sessionLogs || [],
    }
    result.visibleAssistantTurnsComplete = visibleAssistantAfterEachUser(result.chat?.turns || [])
    result.liveSpeedSamples = extractLiveSpeedSamples(result)
    result.provenSurfaces = deriveProvenSurfaces(result)
    rendererResult = mergeRequiredScreenshotOutcome(
      rendererResult,
      reattestRequiredScreenshot(rendererResult.screenshotCapture),
    )
    chatScreenshot = rendererResult.screenshotCapture?.path || null
    result.screenshotCapture = rendererResult.screenshotCapture
    result.rendererFailureStage = rendererResult.rendererFailureStage
    result.failureStage = result.failureStage
      || rendererResult.rendererFailureStage
      || undefined
    result.screenshots.chat = chatScreenshot
    applyTopLevelCorrelationStatus(result, {
      rendererFailed: Boolean(rendererResult.rendererFailureStage),
    })
    let assertionError = null
    try {
      assertResult(result)
    } catch (error) {
      assertionError = error
      applyAssertionFailureStatus(result, error)
    }
    result.artifacts = {
      proof: proofPath,
      ui_backend_binding: bindingPath,
      owned_run_intent: ownedRunIntent?.path || null,
      ui_session_attestation: uiSessionAttestation?.path || null,
      paired_raw_api_proof: pairedApiArtifact?.path || null,
      owned_release_sentinel: releaseEvidence?.path || null,
      screenshot: chatScreenshot,
    }
    writePrivateArtifactFile(proofPath, JSON.stringify(result, null, 2))
    if (process.env.VMLINUX_REAL_UI_ALLOW_FAIL === '1' || process.env.VMLX_REAL_UI_ALLOW_FAIL === '1') {
      console.log(JSON.stringify({
        ok: assertionError === null && result.pass === true,
        failures: assertionError ? result.assertionFailures : undefined,
        result,
      }, null, 2))
    } else {
      if (assertionError) throw assertionError
      console.log(JSON.stringify({ ok: result.pass === true, result }, null, 2))
      if (result.pass !== true) process.exitCode = 2
    }
  } finally {
    if (cdp) cdp.close()
    if (!app?.attached && app?.allowTeardown !== false) {
      await terminateProcess(app?.proc)
    }
    await removeTemporaryTree(userDataDir)
    if (!configuredWorkingDirectory) {
      await removeTemporaryTree(workingDirectory)
    }
  }
}

const invokedAsMain = process.argv[1]
  && path.resolve(process.argv[1]) === fileURLToPath(import.meta.url)

if (invokedAsMain) {
  main().catch((error) => {
    console.error(error.stack || error.message || String(error))
    process.exit(1)
  })
}
