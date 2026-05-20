import { spawn, ChildProcess, execSync, execFileSync } from 'child_process'
import { lookup } from 'dns'
import { powerSaveBlocker } from 'electron'
import { EventEmitter } from 'events'
import { existsSync, readdirSync, readFileSync, statSync } from 'fs'
import { createServer } from 'net'
import { homedir, totalmem, freemem } from 'os'
import { join, basename } from 'path'
import { v4 as uuidv4 } from 'uuid'
import { db, Session } from './database'
import { resolveImageModelFromDirectoryName } from '../shared/imageModels'
import { dsv4EnvFromConfig } from '../shared/dsv4Env'
import { resolveCacheLaunchPolicy } from '../shared/cacheControlPolicy'
import { buildMcpPolicyArgs } from '../shared/mcpPolicy'
import { canonicalizeToolParserId } from '../shared/toolParserAliases'

export type { ServerConfig, DetectedProcess } from './server'
import type { ServerConfig, DetectedProcess } from './server'
import { detectModelConfigFromDir } from './model-config-registry'
import { getBundledPythonPath, verifyBundledEngineOnFilesystem } from './engine-manager'
import { app as electronApp } from 'electron'

/** Result of findEnginePath: either bundled Python or a system binary */
type EnginePath =
  | { type: 'bundled'; pythonPath: string }
  | { type: 'system'; binaryPath: string }

interface ManagedProcess {
  process: ChildProcess | null
  adoptedPid: number | null
  lastStderr?: string  // Last stderr line for error reporting
  exitCode?: number | null
  exitSignal?: string | null  // Signal that killed the process (e.g. SIGKILL for OOM)
  intentionalStop?: boolean   // Set true when stopSession sends SIGTERM — prevents crash misreport
}

/** Normalize model paths for consistent matching: resolve and strip trailing slashes */
function normalizePath(p: string): string {
  return p.replace(/\/+$/, '')
}

function shouldPassHfTokenToEngine(modelPath?: string): boolean {
  const value = String(modelPath || '').trim()
  if (!value || value.startsWith('remote://')) return false
  if (/^https?:\/\//i.test(value)) return true
  if (existsSync(value)) return false
  return /^[A-Za-z0-9][\w.-]*\/[\w./-]+$/.test(value)
}

interface BundleStartupDefaults {
  defaultTemperature?: number
  defaultTopP?: number
  defaultTopK?: number
  defaultMinP?: number
  defaultRepetitionPenalty?: number
  maxTokens?: number
  source?: 'generation_config' | 'jang_config'
}

function normalizeDetectedFamilyName(family?: string): string | undefined {
  if (!family) return undefined
  if (family === 'deepseek_v4') return 'deepseek-v4'
  if (family === 'zaya1_vl') return 'zaya1-vl'
  if (family === 'bailing_hybrid') return 'ling'
  return family
}

function isZayaCcaFamily(family?: string): boolean {
  const normalized = normalizeDetectedFamilyName(family)
  return normalized === 'zaya' || normalized === 'zaya1-vl'
}

function cacheTypeRequiresPaged(cacheType?: string): boolean {
  return cacheType === 'hybrid' || cacheType === 'mamba'
}

const DSV4_PAGED_CACHE_BLOCK_SIZE = 256
const GENERIC_DEFAULT_TIMEOUT_SECONDS = 300
const DSV4_DEFAULT_TIMEOUT_SECONDS = 900

function effectiveSessionTimeoutSeconds(config: Partial<ServerConfig>, family?: string): number {
  const configured = config.timeout
  if (configured != null && configured <= 0) return 86400
  const normalizedFamily = normalizeDetectedFamilyName(family)
  if (normalizedFamily === 'deepseek-v4' && (configured == null || configured === GENERIC_DEFAULT_TIMEOUT_SECONDS)) {
    return DSV4_DEFAULT_TIMEOUT_SECONDS
  }
  return configured != null && configured > 0 ? configured : GENERIC_DEFAULT_TIMEOUT_SECONDS
}

function applyFamilyStartupDefaults(config: Partial<ServerConfig>, modelPath?: string): void {
  if (!modelPath) return
  try {
    const detected = detectModelConfigFromDir(modelPath)
    if (
      normalizeDetectedFamilyName(detected.family) === 'deepseek-v4' &&
      (config.timeout == null || config.timeout === GENERIC_DEFAULT_TIMEOUT_SECONDS)
    ) {
      config.timeout = DSV4_DEFAULT_TIMEOUT_SECONDS
    }
  } catch {
    /* family defaults are best-effort; launch-time buildArgs repeats the guard */
  }
}

const ADDITIONAL_ARG_VALUE_FLAGS = new Set([
  '--block-disk-cache-dir',
  '--block-disk-cache-max-gb',
  '--allowed-origins',
  '--api-key',
  '--cache-memory-mb',
  '--cache-memory-percent',
  '--cache-ttl-minutes',
  '--chat-template',
  '--chat-template-kwargs',
  '--completion-batch-size',
  '--cluster-secret',
  '--default-enable-thinking',
  '--default-min-p',
  '--default-repetition-penalty',
  '--default-temperature',
  '--default-top-k',
  '--default-top-p',
  '--distributed-mode',
  '--disk-cache-dir',
  '--disk-cache-max-gb',
  '--embedding-model',
  '--flash-moe-io-split',
  '--flash-moe-prefetch',
  '--flash-moe-slot-bank',
  '--host',
  '--image-mode',
  '--image-quantize',
  '--kv-cache-group-size',
  '--kv-cache-quantization',
  '--log-level',
  '--max-cache-blocks',
  '--max-num-seqs',
  '--max-prompt-tokens',
  '--max-tokens',
  '--mcp-config',
  '--mcp-disabled-servers',
  '--mcp-disabled-tools',
  '--mcp-enabled-servers',
  '--mcp-enabled-tools',
  '--mflux-class',
  '--num-draft-tokens',
  '--native-mtp-depth',
  '--native-mtp-sampling-policy',
  '--paged-cache-block-size',
  '--pld-summary-interval',
  '--port',
  '--prefill-batch-size',
  '--prefill-step-size',
  '--prefix-cache-max-bytes',
  '--prefix-cache-size',
  '--rate-limit',
  '--reasoning-parser',
  '--served-model-name',
  '--smelt-experts',
  '--speculative-model',
  '--ssm-state-cache-mb',
  '--ssm-state-cache-size',
  '--stream-interval',
  '--timeout',
  '--tool-call-parser',
  '--worker-nodes',
])

const IMAGE_ADDITIONAL_ARG_BLOCKLIST = new Set([
  '--image-mode',
  '--image-quantize',
  '--served-model-name',
  '--mflux-class',
])

const DSV4_ADDITIONAL_ARG_BLOCKLIST = new Set([
  '--continuous-batching',
  '--no-continuous-batching',
  '--host',
  '--port',
  '--api-key',
  '--rate-limit',
  '--timeout',
  '--log-level',
  '--allowed-origins',
  '--served-model-name',
  '--enable-prefix-cache',
  '--disable-prefix-cache',
  '--use-paged-cache',
  '--paged-cache-block-size',
  '--max-cache-blocks',
  '--kv-cache-quantization',
  '--kv-cache-group-size',
  '--max-num-seqs',
  '--prefill-batch-size',
  '--prefill-step-size',
  '--completion-batch-size',
  '--max-tokens',
  '--max-prompt-tokens',
  '--stream-interval',
  '--ssm-state-cache-size',
  '--ssm-state-cache-mb',
  '--enable-jit',
  '--no-memory-aware-cache',
  '--prefix-cache-size',
  '--prefix-cache-max-bytes',
  '--cache-memory-mb',
  '--cache-memory-percent',
  '--cache-ttl-minutes',
  '--enable-disk-cache',
  '--disk-cache-dir',
  '--disk-cache-max-gb',
  '--smelt',
  '--smelt-experts',
  '--flash-moe',
  '--flash-moe-slot-bank',
  '--flash-moe-prefetch',
  '--flash-moe-io-split',
  '--distributed',
  '--distributed-mode',
  '--worker-nodes',
  '--cluster-secret',
  '--speculative-model',
  '--num-draft-tokens',
  '--native-mtp-depth',
  '--native-mtp-sampling-policy',
  '--disable-native-mtp',
  '--enable-pld',
  '--pld-summary-interval',
  '--is-mllm',
  '--mcp-config',
  '--mcp-disabled-servers',
  '--mcp-disabled-tools',
  '--mcp-enabled-servers',
  '--mcp-enabled-tools',
  '--enable-auto-tool-choice',
  '--tool-call-parser',
  '--reasoning-parser',
  '--embedding-model',
  '--default-temperature',
  '--default-top-p',
  '--default-top-k',
  '--default-min-p',
  '--default-repetition-penalty',
  '--default-enable-thinking',
  '--chat-template',
  '--chat-template-kwargs',
])

function filterAdditionalArgs(raw: string | undefined, blockedFlags: Set<string>): string[] {
  if (!raw?.trim()) return []
  const extra = raw.trim().split(/\s+/).filter(Boolean)
  const filtered: string[] = []
  for (let i = 0; i < extra.length; i++) {
    const flag = extra[i]
    if (blockedFlags.has(flag)) {
      if (ADDITIONAL_ARG_VALUE_FLAGS.has(flag)) i++
      continue
    }
    filtered.push(flag)
  }
  return filtered
}

function readBundleStartupDefaults(modelPath?: string): BundleStartupDefaults {
  if (!modelPath) return {}
  const out: BundleStartupDefaults = {}
  try {
    const gen = JSON.parse(readFileSync(join(modelPath, 'generation_config.json'), 'utf8'))
    if (typeof gen.temperature === 'number') out.defaultTemperature = Math.round(gen.temperature * 100)
    if (typeof gen.top_p === 'number') out.defaultTopP = Math.round(gen.top_p * 100)
    if (typeof gen.top_k === 'number') out.defaultTopK = Math.max(0, Math.round(gen.top_k))
    if (typeof gen.min_p === 'number') out.defaultMinP = Math.max(0, Math.round(gen.min_p * 100))
    if (typeof gen.repetition_penalty === 'number') out.defaultRepetitionPenalty = Math.round(gen.repetition_penalty * 100)
    if (typeof gen.max_new_tokens === 'number' && gen.max_new_tokens > 0) out.maxTokens = Math.round(gen.max_new_tokens)
    if (Object.keys(out).length > 0) out.source = 'generation_config'
  } catch { /* generation_config.json is optional */ }

  try {
    const jang = JSON.parse(readFileSync(join(modelPath, 'jang_config.json'), 'utf8'))
    const sampling = jang?.chat?.sampling_defaults
    if (sampling && typeof sampling === 'object') {
      if (typeof sampling.temperature === 'number') out.defaultTemperature = Math.round(sampling.temperature * 100)
      if (typeof sampling.top_p === 'number') out.defaultTopP = Math.round(sampling.top_p * 100)
      if (typeof sampling.top_k === 'number') out.defaultTopK = Math.max(0, Math.round(sampling.top_k))
      if (typeof sampling.min_p === 'number') out.defaultMinP = Math.max(0, Math.round(sampling.min_p * 100))
      // Pick mode-specific repetition penalty based on the bundle's
      // default reasoning mode. Bundles can split _thinking vs _chat because
      // the correct value is part of the bundle's chat contract. Falls back
      // to the unified scalar if either side is missing.
      const defaultMode = jang?.chat?.reasoning?.default_mode
      const repThinking = typeof sampling.repetition_penalty_thinking === 'number'
        ? sampling.repetition_penalty_thinking : undefined
      const repChat = typeof sampling.repetition_penalty_chat === 'number'
        ? sampling.repetition_penalty_chat : undefined
      const repScalar = typeof sampling.repetition_penalty === 'number'
        ? sampling.repetition_penalty : undefined
      const rep = defaultMode === 'thinking'
        ? (repThinking ?? repChat ?? repScalar)
        : (repChat ?? repThinking ?? repScalar)
      if (typeof rep === 'number') out.defaultRepetitionPenalty = Math.round(rep * 100)
      if (typeof sampling.max_new_tokens === 'number' && sampling.max_new_tokens > 0) out.maxTokens = Math.round(sampling.max_new_tokens)
      out.source = 'jang_config'
    }
  } catch { /* jang_config.json is optional */ }

  return out
}

function applyBundleStartupDefaults(config: Partial<ServerConfig>, modelPath?: string): void {
  const defs = readBundleStartupDefaults(modelPath)

  // Startup generation defaults are model-owned. Keep the saved/display config
  // aligned with bundle sampling metadata and clear old generic startup
  // overrides; buildArgs intentionally does not turn these values into
  // --default-* flags. max_new_tokens is also bundle-owned, but it must not be
  // copied into a hidden session-level --max-tokens default. Users change
  // output length per chat or per API request.
  config.defaultTemperature = defs.defaultTemperature ?? 0
  config.defaultTopP = defs.defaultTopP ?? 0
  ;(config as any).defaultTopK = defs.defaultTopK ?? 0
  ;(config as any).defaultMinP = defs.defaultMinP ?? 0
  ;(config as any).defaultRepetitionPenalty = defs.defaultRepetitionPenalty ?? 0
  ;(config as any).defaultMaxNewTokens = defs.maxTokens ?? 0
  const migrationKey = 'generationStartupDefaultsVersion'
  if ((config as any)[migrationKey] !== 3) {
    const oldHiddenMaxTokens =
      defs.maxTokens != null && Number(config.maxTokens) === Number(defs.maxTokens)
    const oldGenericMaxTokens = [4096, 12000, 12068, 32768].includes(Number(config.maxTokens))
    if (oldHiddenMaxTokens || oldGenericMaxTokens) {
      config.maxTokens = 0
    }
    ;(config as any)[migrationKey] = 3
  }
}

const CACHE_STACK_STARTUP_DEFAULTS_VERSION = 2

function markCacheStackStartupDefaultsCurrent(config: Partial<ServerConfig>): boolean {
  if (config.cacheStackStartupDefaultsVersion === CACHE_STACK_STARTUP_DEFAULTS_VERSION) return false
  config.cacheStackStartupDefaultsVersion = CACHE_STACK_STARTUP_DEFAULTS_VERSION
  return true
}

function isZayaCacheStackMigrationTarget(modelPath?: string): boolean {
  const lower = String(modelPath || '').toLowerCase()
  return lower.includes('zaya1') || lower.includes('zaya')
}

function applyCacheStackStartupDefaultMigration(config: Partial<ServerConfig>, modelPath?: string): boolean {
  if (Number(config.cacheStackStartupDefaultsVersion || 0) >= CACHE_STACK_STARTUP_DEFAULTS_VERSION) {
    return false
  }

  const zayaCacheMigrationTarget = isZayaCacheStackMigrationTarget(modelPath || config.modelPath)
  const staleContinuousDefaults =
    config.continuousBatching === true &&
    config.enablePrefixCache === true &&
    Number(config.maxNumSeqs) === 64 &&
    Number(config.prefillBatchSize) === 1024 &&
    Number(config.completionBatchSize) === 1024
  const staleNoPrefixBatchDefaults =
    config.continuousBatching === true &&
    config.enablePrefixCache === false &&
    Number(config.maxNumSeqs) > 0 &&
    Number(config.maxNumSeqs) <= 8 &&
    Number(config.prefillBatchSize) === 1024 &&
    Number(config.completionBatchSize) === 1024
  const stalePartialPagedCacheDefaults =
    zayaCacheMigrationTarget &&
    config.continuousBatching === true &&
    config.enablePrefixCache === true &&
    Number(config.maxNumSeqs) === 1 &&
    Number(config.prefillBatchSize) === 512 &&
    Number(config.completionBatchSize) === 512 &&
    config.usePagedCache === false
  const staleExplicitNoneCacheCodecDefaults =
    zayaCacheMigrationTarget &&
    config.continuousBatching === true &&
    config.enablePrefixCache === true &&
    Number(config.maxNumSeqs) === 1 &&
    Number(config.prefillBatchSize) === 512 &&
    Number(config.completionBatchSize) === 512 &&
    config.usePagedCache === true &&
    config.enableBlockDiskCache === true &&
    config.kvCacheQuantization === 'none'

  if (
    !staleContinuousDefaults &&
    !staleNoPrefixBatchDefaults &&
    !stalePartialPagedCacheDefaults &&
    !staleExplicitNoneCacheCodecDefaults
  ) return false

  config.continuousBatching = true
  config.enablePrefixCache = true
  config.maxNumSeqs = 1
  config.prefillBatchSize = 512
  config.prefillStepSize = 2048
  config.completionBatchSize = 512
  config.usePagedCache = true
  config.maxCacheBlocks = 1000
  config.kvCacheQuantization = 'auto'
  config.enableBlockDiskCache = true
  config.blockDiskCacheMaxGb = 10
  config.cacheMemoryPercent = 15
  markCacheStackStartupDefaultsCurrent(config)
  return true
}

/** Resolve bind address to connectable address (0.0.0.0 → 127.0.0.1) */
export function connectHost(host: string): string {
  return host === '0.0.0.0' ? '127.0.0.1' : host
}

/** Estimate model file bytes from local model files. Returns 0 if unknown. */
function estimateModelFileBytes(modelPath: string): number {
  try {
    const entries = readdirSync(modelPath, { withFileTypes: true })
    let totalBytes = 0
    for (const entry of entries) {
      const fullPath = join(modelPath, entry.name)
      if (entry.isDirectory()) {
        totalBytes += estimateModelFileBytes(fullPath)
      } else if (entry.isFile()) {
        totalBytes += statSync(fullPath).size
      }
    }
    return totalBytes
  } catch (_) {
    return 0
  }
}

/** Estimate model memory usage from safetensors file sizes. Returns bytes or 0 if unknown. */
function estimateModelMemory(modelPath: string): number {
  const fileBytes = estimateModelFileBytes(modelPath)
  if (fileBytes <= 0) return 0
  // Model size on disk + ~30% overhead for KV cache, activations, framework
  return Math.round(fileBytes * 1.3)
}

function formatGb(bytes: number): string {
  return (bytes / 1e9).toFixed(1)
}

/**
 * Resolve .local (mDNS/Bonjour) hostnames to IPv4 before fetch.
 * Node.js/undici's fetch resolves .local to IPv6 link-local (fe80::...)
 * which is unreachable without a zone ID, causing "fetch failed".
 * This replaces the hostname with the resolved IPv4 address.
 *
 * Results are cached for 60s to avoid redundant DNS lookups on every
 * message send and health check (previously added 50-100ms per call).
 */
const resolvedUrlCache = new Map<string, { url: string; timestamp: number }>()
const RESOLVE_URL_CACHE_TTL = 60_000 // 60 seconds

export async function resolveUrl(url: string): Promise<string> {
  const cached = resolvedUrlCache.get(url)
  if (cached && Date.now() - cached.timestamp < RESOLVE_URL_CACHE_TTL) {
    return cached.url
  }

  try {
    const parsed = new URL(url)
    if (parsed.hostname.endsWith('.local')) {
      const ip = await new Promise<string>((resolve, reject) => {
        lookup(parsed.hostname, { family: 4 }, (err, addr) => {
          if (err) reject(err); else resolve(addr)
        })
      })
      parsed.hostname = ip
      const resolved = parsed.toString().replace(/\/+$/, '')
      console.log(`[DNS] Resolved .local: ${url} → ${resolved}`)
      resolvedUrlCache.set(url, { url: resolved, timestamp: Date.now() })
      return resolved
    }
  } catch (e) {
    console.log(`[DNS] Failed to resolve ${url}:`, e)
  }
  resolvedUrlCache.set(url, { url, timestamp: Date.now() })
  return url
}

export class SessionManager extends EventEmitter {
  private processes = new Map<string, ManagedProcess>()
  private monitorInterval: ReturnType<typeof setInterval> | null = null
  private failCounts = new Map<string, number>()
  /** Per-session operation lock to prevent concurrent start/stop races */
  private operationLocks = new Map<string, Promise<void>>()
  /** Global creation lock to prevent port assignment races between concurrent createSession calls */
  private creationLock: Promise<void> = Promise.resolve()
  /** Timestamp of last successful health check per session (used to skip redundant per-message checks) */
  private lastHealthyAt = new Map<string, number>()
  /** Per-session ring buffer for log lines (capped at LOG_BUFFER_MAX_LINES) */
  private logBuffers = new Map<string, string[]>()
  private static readonly LOG_BUFFER_MAX_LINES = 2000
  // Allow up to 60 consecutive health check failures (5s * 60 = 5 min)
  // before marking session as down. Long prefill operations (e.g. 44k+
  // tokens) can block the server's event loop for 30+ seconds.
  private static readonly MAX_FAIL_COUNT = 60

  // ── System sleep prevention ──
  /** Electron powerSaveBlocker ID (-1 = not active) */
  private powerBlockerId: number = -1

  // ── Idle / Sleep tracking ──
  /** Timestamp of last API request per session (for idle detection) */
  private lastRequestAt = new Map<string, number>()
  /** Default idle timeouts in milliseconds */
  private static readonly DEFAULT_SOFT_TIMEOUT_TEXT_MS = 10 * 60 * 1000   // 10 min
  private static readonly DEFAULT_HARD_TIMEOUT_TEXT_MS = 30 * 60 * 1000   // 30 min
  private static readonly DEFAULT_SOFT_TIMEOUT_IMAGE_MS = 5 * 60 * 1000   // 5 min
  private static readonly DEFAULT_HARD_TIMEOUT_IMAGE_MS = 15 * 60 * 1000  // 15 min

  constructor() {
    super()
  }

  /** Get timestamp of last successful health check for a session (0 if never checked) */
  getLastHealthyAt(sessionId: string): number {
    return this.lastHealthyAt.get(sessionId) || 0
  }

  // Loading progress patterns — matched against engine stdout/stderr to detect loading phase
  private static readonly LOAD_PROGRESS_PATTERNS: Array<{ pattern: RegExp; label: string; progress: number }> = [
    { pattern: /Loading model:/, label: 'Initializing...', progress: 5 },
    // Phase 1: Process startup + config (0-25%)
    // For BatchedEngine, these fire BEFORE actual model loading (lifespan phase).
    // Keep progress low so real loading patterns (Phase 2) can advance the bar.
    { pattern: /System memory before load/, label: 'Checking memory...', progress: 5 },
    { pattern: /Loading model with (?:Simple|Batched)Engine/, label: 'Creating engine...', progress: 8 },
    { pattern: /\bmodel loaded \(batched mode\)/i, label: 'Starting server...', progress: 10 },
    { pattern: /Metal GPU memory after load/, label: 'Server initializing...', progress: 12 },
    { pattern: /Native tool format enabled/, label: 'Configuring tools...', progress: 14 },
    { pattern: /Default max tokens:/, label: 'Configuring limits...', progress: 16 },
    { pattern: /Uvicorn running on/, label: 'Server started, loading model...', progress: 20 },
    { pattern: /Waiting for application startup/, label: 'Starting model runtime...', progress: 22 },

    // Phase 2: Actual model loading (25-85%)
    // For BatchedEngine these fire DURING lifespan() (after Uvicorn starts).
    // For SimpleEngine these fire DURING load_model() (before Uvicorn starts).
    { pattern: /JANG v2 detected/, label: 'Loading JANG weights...', progress: 30 },
    { pattern: /Loading JANG v1 VLM:/, label: 'Loading JANG VL model...', progress: 30 },
    { pattern: /Loading MLLM:/, label: 'Loading vision model...', progress: 30 },
    { pattern: /Loading image model:/, label: 'Loading image model...', progress: 30 },
    { pattern: /Loading JANG VL model:/, label: 'Loading JANG VL...', progress: 30 },
    { pattern: /Loading \d+ safetensors shards/, label: 'Loading weights...', progress: 40 },
    { pattern: /Split kv_b_proj layer/, label: 'Processing MLA layers...', progress: 50 },
    { pattern: /bfloat16 enabled/, label: 'Converting to bfloat16...', progress: 55 },
    { pattern: /JANG v[12].{0,10}loaded in/, label: 'Weights loaded', progress: 65 },
    { pattern: /Model loaded successfully/, label: 'Model loaded', progress: 65 },
    { pattern: /MLLM loaded successfully/, label: 'Vision model loaded', progress: 65 },
    { pattern: /JANG VL model loaded/, label: 'JANG VL loaded', progress: 65 },
    { pattern: /Image model loaded in/, label: 'Image model loaded', progress: 65 },

    // Phase 3: Post-load config (85-92%)
    // SimpleEngine: fires during load_model(). BatchedEngine: fires during lifespan().
    { pattern: /\bmodel loaded \(simple mode\)/i, label: 'Engine ready', progress: 70 },
    { pattern: /Saved \d+\/\d+ layer weights to SSD/, label: 'Saving weights to SSD...', progress: 72 },
    { pattern: /SSD weight index:/, label: 'Building weight index...', progress: 73 },
    { pattern: /SSD per-layer weight recycling configured/, label: 'SSD streaming ready', progress: 74 },
    { pattern: /KV cache quantization/, label: 'Setting up KV cache...', progress: 78 },
    { pattern: /(?:Chat template loaded|Applied custom chat template)/, label: 'Loading chat template...', progress: 80 },
    { pattern: /PagedCacheManager initialized/, label: 'Configuring cache...', progress: 82 },
    { pattern: /Scheduler (?:initialized|started)/, label: 'Starting scheduler...', progress: 85 },
    { pattern: /BatchedEngine loaded/, label: 'Engine ready', progress: 88 },
    { pattern: /Application startup complete/, label: 'Almost ready...', progress: 92 },
  ]

  // Track last emitted progress per session to avoid duplicate events
  private loadProgressState = new Map<string, number>()
  private loadProgressMeta = new Map<string, {
    modelBytes?: number;
    lazyResident?: boolean;
    residentMb?: number;
    residentPercent?: number;
  }>()
  private loadResidentTimers = new Map<string, ReturnType<typeof setInterval>>()

  private stopLoadResidentMonitor(sessionId: string): void {
    const timer = this.loadResidentTimers.get(sessionId)
    if (timer) {
      clearInterval(timer)
      this.loadResidentTimers.delete(sessionId)
    }
  }

  private readProcessGroupResidentBytes(pid: number): number {
    const parseRssKb = (text: string): number =>
      text
        .split('\n')
        .map(s => parseInt(s.trim(), 10))
        .filter(n => Number.isFinite(n) && n > 0)
        .reduce((sum, kb) => sum + kb, 0)

    try {
      const group = execFileSync('ps', ['-o', 'rss=', '-g', String(pid)], {
        timeout: 1000,
      }).toString()
      const groupKb = parseRssKb(group)
      if (groupKb > 0) return groupKb * 1024
    } catch { /* fall back to root process RSS */ }

    try {
      const single = execFileSync('ps', ['-o', 'rss=', '-p', String(pid)], {
        timeout: 1000,
      }).toString()
      return parseRssKb(single) * 1024
    } catch {
      return 0
    }
  }

  private startLoadResidentMonitor(sessionId: string, pid: number, modelBytes: number): void {
    this.stopLoadResidentMonitor(sessionId)
    if (!pid || modelBytes <= 0) return

    const tick = () => {
      const session = db.getSession(sessionId)
      if (!session || session.status !== 'loading') {
        this.stopLoadResidentMonitor(sessionId)
        return
      }
      const residentBytes = this.readProcessGroupResidentBytes(pid)
      if (residentBytes <= 0) return

      const residentPercent = Math.min(100, Math.max(0, (residentBytes / modelBytes) * 100))
      const residentProgress = Math.min(90, Math.max(5, Math.round(25 + residentPercent * 0.60)))
      const current = this.loadProgressState.get(sessionId) ?? 0
      const progress = Math.max(current, residentProgress)
      const meta = {
        ...(this.loadProgressMeta.get(sessionId) || {}),
        modelBytes,
        lazyResident: true,
        residentMb: Math.round((residentBytes / 1048576) * 10) / 10,
        residentPercent: Math.round(residentPercent * 10) / 10,
      }
      this.loadProgressMeta.set(sessionId, meta)
      this.loadProgressState.set(sessionId, progress)
      this.emit('session:loadProgress', {
        sessionId,
        label: `Resident RAM ${formatGb(residentBytes)} / ${formatGb(modelBytes)} GB`,
        progress,
        ...meta,
      })
    }

    tick()
    this.loadResidentTimers.set(sessionId, setInterval(tick, 1000))
  }

  /** Check a log line for loading progress and emit event if phase advanced */
  private checkLoadProgress(sessionId: string, text: string): void {
    for (const { pattern, label, progress } of SessionManager.LOAD_PROGRESS_PATTERNS) {
      if (pattern.test(text)) {
        const current = this.loadProgressState.get(sessionId) ?? 0
        if (progress > current) {
          this.loadProgressState.set(sessionId, progress)
          this.emit('session:loadProgress', {
            sessionId,
            label,
            progress,
            ...(this.loadProgressMeta.get(sessionId) || {}),
          })
        }
        break
      }
    }
  }

  /** Append log data to the per-session ring buffer */
  pushLog(sessionId: string, data: string): void {
    let buffer = this.logBuffers.get(sessionId)
    if (!buffer) {
      buffer = []
      this.logBuffers.set(sessionId, buffer)
    }
    const timestamp = new Date().toISOString().slice(11, 23) // HH:mm:ss.SSS
    const lines = data.split('\n')
    for (const line of lines) {
      if (!line && lines.length > 1) continue // skip empty splits from trailing newline
      buffer.push(`[${timestamp}] ${line}`)
    }
    if (buffer.length > SessionManager.LOG_BUFFER_MAX_LINES) {
      buffer.splice(0, buffer.length - SessionManager.LOG_BUFFER_MAX_LINES)
    }
    // Parse log for loading progress indicators
    this.checkLoadProgress(sessionId, data)
  }

  /** Get all buffered log lines for a session */
  getLogs(sessionId: string): string[] {
    return this.logBuffers.get(sessionId) || []
  }

  /** Clear the log buffer for a session */
  clearLogs(sessionId: string): void {
    this.logBuffers.delete(sessionId)
  }

  /**
   * Acquire a per-session operation lock. Serializes start/stop operations
   * for the same session to prevent race conditions (e.g. stop during start,
   * start during stop, rapid start/stop/start).
   *
   * Uses promise-chaining: each caller atomically chains onto the tail of
   * the previous operation. No TOCTOU window between await and set.
   */
  private withSessionLock(sessionId: string, fn: () => Promise<void>): Promise<void> {
    const prev = this.operationLocks.get(sessionId) ?? Promise.resolve()
    const next = prev.catch(() => {}).then(() => fn())
    const tail = next.catch(() => {})
    // Store the chain tail so the next caller awaits us
    this.operationLocks.set(sessionId, tail)
    // Clean up once our operation settles (avoids unbounded map growth)
    tail.then(() => {
      if (this.operationLocks.get(sessionId) === tail) {
        this.operationLocks.delete(sessionId)
      }
    })
    return next
  }

  // ─── Process Detection (reused from ServerManager) ─────────────────

  async detect(): Promise<DetectedProcess[]> {
    const detected: DetectedProcess[] = []

    try {
      const output = execSync('ps aux', { encoding: 'utf-8', timeout: 5000 })
      const lines = output.split('\n')

      for (const line of lines) {
        if (line.includes('grep')) continue
        // Detect `vmlx-engine serve`, `python -m vmlx_engine.cli serve`, and `python -m vmlx_engine.server` processes
        const isCliServe = line.includes('vmlx-engine') && line.includes('serve')
        const isPythonModule = line.includes('vmlx_engine') && (line.includes('.cli') || line.includes('.server') || line.includes('--model'))
        if (!isCliServe && !isPythonModule) continue

        const parsed = this.parsePsLine(line)
        if (!parsed) continue

        let healthy = false
        let modelName: string | undefined
        let standbyDepth: 'soft' | 'deep' | null = null
        try {
          const res = await fetch(
            `http://127.0.0.1:${parsed.port}/health`,
            { signal: AbortSignal.timeout(2000) }
          )
          if (res.ok) {
            const data = await res.json()
            healthy = true
            modelName = data.model_name
            // Detect standby state for proper re-adoption
            if (data.status === 'standby_soft') standbyDepth = 'soft'
            else if (data.status === 'standby_deep') standbyDepth = 'deep'
          }
        } catch (_) { }

        detected.push({
          pid: parsed.pid,
          port: parsed.port,
          modelPath: parsed.modelPath,
          healthy,
          modelName,
          standbyDepth
        })
      }
    } catch (_) { }

    return detected
  }

  private parsePsLine(line: string): { pid: number; port: number; modelPath: string } | null {
    try {
      const parts = line.trim().split(/\s+/)
      const pid = parseInt(parts[1])
      if (isNaN(pid)) return null

      const cmdStart = parts.slice(10).join(' ')

      let modelPath = ''

      // Try `serve <model-path> --...` format first (vmlx-engine CLI)
      const serveIdx = cmdStart.indexOf('serve ')
      if (serveIdx !== -1) {
        const afterServe = cmdStart.substring(serveIdx + 6).trim()
        modelPath = afterServe.split(/\s+--/)[0].trim()
      }

      // Try `--model <path>` format (python -m vmlx_engine.server)
      if (!modelPath) {
        const modelMatch = cmdStart.match(/--model\s+(\S+)/)
        if (modelMatch) modelPath = modelMatch[1]
      }

      if (!modelPath) return null

      // Normalize: strip trailing slashes for consistent matching
      modelPath = normalizePath(modelPath)

      let port = 8000
      const portMatch = cmdStart.match(/--port\s+(\d+)/)
      if (portMatch) port = parseInt(portMatch[1])

      return { pid, port, modelPath }
    } catch (_) {
      return null
    }
  }

  // ─── Session Lifecycle ─────────────────────────────────────────────

  async createSession(modelPath: string, config: Partial<ServerConfig>): Promise<Session> {
    // Serialize all session creation to prevent port assignment race conditions.
    // Without this, concurrent createSession calls can both see the same DB snapshot
    // and assign the same port (TOCTOU race in findAvailablePort).
    let unlock!: () => void
    const prev = this.creationLock
    this.creationLock = new Promise<void>(r => { unlock = r })
    await prev
    try {
      return await this._createSessionInner(modelPath, config)
    } finally {
      unlock()
    }
  }

  private async _createSessionInner(modelPath: string, config: Partial<ServerConfig>): Promise<Session> {
    // Normalize path to prevent trailing-slash mismatches
    modelPath = normalizePath(modelPath)
    applyBundleStartupDefaults(config, modelPath)
    applyFamilyStartupDefaults(config, modelPath)
    markCacheStackStartupDefaultsCurrent(config)

    // Check if session already exists for this model path
    const existing = db.getSessionByModelPath(modelPath)
    if (existing) {
      // Merge new config into existing (don't overwrite unspecified fields)
      let existingConfig: Record<string, any> = {}
      try { existingConfig = JSON.parse(existing.config || '{}') } catch (_) { }
      const host = (config.host as string) || existing.host
      const port = (config.port as number) || existing.port
      applyCacheStackStartupDefaultMigration(existingConfig, modelPath)
      const merged = { ...existingConfig, ...config, modelPath, host, port }
      applyBundleStartupDefaults(merged, modelPath)
      applyFamilyStartupDefaults(merged, modelPath)
      markCacheStackStartupDefaultsCurrent(merged)
      db.updateSession(existing.id, {
        config: JSON.stringify(merged),
        host,
        port
      })
      return db.getSession(existing.id)!
    }

    const id = uuidv4()
    const host = config.host || '127.0.0.1'
    const port = config.port || await this.findAvailablePort()
    const now = Date.now()

    const session: Session = {
      id,
      modelPath,
      modelName: modelPath.split('/').pop() || modelPath,
      host,
      port,
      status: 'stopped',
      config: JSON.stringify({ ...config, modelPath, port, host }),
      createdAt: now,
      updatedAt: now,
      type: 'local'
    }

    db.createSession(session)
    this.emit('session:created', session)
    return session
  }

  async createRemoteSession(params: {
    remoteUrl: string
    remoteApiKey?: string
    remoteModel: string
    remoteOrganization?: string
  }): Promise<Session> {
    const url = new URL(params.remoteUrl)
    const modelPath = `remote://${params.remoteModel}@${url.host}`

    const existing = db.getSessionByModelPath(modelPath)
    if (existing) {
      db.updateSession(existing.id, {
        remoteUrl: params.remoteUrl,
        remoteApiKey: params.remoteApiKey,
        remoteModel: params.remoteModel,
        remoteOrganization: params.remoteOrganization
      })
      return db.getSession(existing.id)!
    }

    const id = uuidv4()
    const host = url.hostname
    // Remote sessions don't bind a local port — the port field is just a DB key.
    // Use findAvailablePort to avoid UNIQUE constraint conflicts when multiple
    // remote sessions point to different models on the same host (e.g., port 443).
    const port = await this.findAvailablePort()
    const now = Date.now()

    const session: Session = {
      id,
      modelPath,
      modelName: params.remoteModel,
      host,
      port,
      status: 'stopped',
      config: JSON.stringify({ timeout: 300 }),
      createdAt: now,
      updatedAt: now,
      type: 'remote',
      remoteUrl: params.remoteUrl,
      remoteApiKey: params.remoteApiKey,
      remoteModel: params.remoteModel,
      remoteOrganization: params.remoteOrganization
    }

    db.createSession(session)
    this.emit('session:created', session)
    return session
  }

  async startSession(sessionId: string): Promise<void> {
    // Remote sessions connect instead of starting a local process
    const session = db.getSession(sessionId)
    if (session?.type === 'remote') {
      // Guard: skip if already running or connecting
      if (session.status === 'running' || session.status === 'loading') {
        console.log(`[SESSIONS] Remote session ${sessionId} already ${session.status}, skipping connect`)
        return
      }
      return this._connectRemoteSession(session)
    }

    // Serialize start/stop operations per session to prevent races
    await this.withSessionLock(sessionId, () => this._startSessionInner(sessionId))
  }

  private async _startSessionInner(sessionId: string): Promise<void> {
    const session = db.getSession(sessionId)
    if (!session) throw new Error(`Session ${sessionId} not found`)

    const managed = this.processes.get(sessionId)
    if (managed?.process || managed?.adoptedPid) {
      throw new Error('Session is already running')
    }

    const config: ServerConfig = JSON.parse(session.config)
    config.modelPath = session.modelPath
    config.host = session.host
    config.port = session.port
    applyBundleStartupDefaults(config, config.modelPath)
    const migrated = applyCacheStackStartupDefaultMigration(config, config.modelPath)
    const markedCurrent = markCacheStackStartupDefaultsCurrent(config)
    if (migrated || markedCurrent) {
      // Persist the migrated config so the settings UI reflects the corrected
      // values on next render and the same migration doesn't have to re-fire
      // on every session start. Without this writeback the saved config keeps
      // showing the stale tuple even though the engine launches with the new
      // values.
      try {
        db.updateSession(session.id, { config: JSON.stringify(config) })
        console.log(`[SESSION] Persisted cache-stack startup defaults version for session ${session.id}`)
      } catch (e) {
        console.warn(`[SESSION] Failed to persist migration for ${session.id}: ${e}`)
      }
    }

    // Server startup must not carry per-chat thinking choices. The engine
    // resolves model defaults from its registry/bundle; explicit chat/API
    // requests pass enable_thinking per request.
    delete config.defaultEnableThinking

    const engineResult = this.findEnginePath()
    if (!engineResult) throw new Error('vmlx-engine not found. Please install it first.')

    // Image models may use mflux named models (e.g., "schnell") that are NOT filesystem paths
    // — skip path/format validation for image sessions, let the Python server handle it
    const isImageSession = config.modelType === 'image'

    if (!isImageSession) {
      if (!existsSync(config.modelPath)) throw new Error(`Model not found at: ${config.modelPath}`)

      // Block starting a session with an actively downloading model
      const downloadMarker = join(config.modelPath, '.vmlx-downloading')
      if (existsSync(downloadMarker)) {
        throw new Error('This model is still downloading. Please wait for the download to complete before starting a session.')
      }

      // Validate model format: vmlx-engine only supports MLX (safetensors) models
      try {
        const files = readdirSync(config.modelPath)
        const hasGGUF = files.some(f => f.endsWith('.gguf') || f.endsWith('.gguf.part'))
        const hasSafetensors = files.some(f => f.endsWith('.safetensors'))
        const hasConfig = files.includes('config.json')

        if (hasGGUF && !hasSafetensors) {
          throw new Error(
            'This model is in GGUF format, which is not supported by vmlx-engine. ' +
            'Please download an MLX-format version (safetensors) from HuggingFace Hub.'
          )
        }
        // Diffusers image models have model_index.json instead of config.json — that's valid
        const hasModelIndex = files.includes('model_index.json')
        const hasTransformerDir = files.includes('transformer')
        if (!hasConfig && !hasModelIndex && !hasTransformerDir) {
          throw new Error(
            'Model directory is missing config.json (text) or model_index.json (image). ' +
            'vmlx-engine requires MLX-format models with config.json and .safetensors files, ' +
            'or diffusers-format image models with model_index.json.'
          )
        }
      } catch (e) {
        if ((e as Error).message.includes('GGUF format') || (e as Error).message.includes('missing config.json') || (e as Error).message.includes('model_index.json')) throw e
        // Ignore filesystem errors — let the server handle them
      }
    } // end if (!isImageSession)

    // Re-detect model config from disk — handles case where model files were
    // replaced with a different model (same folder name, different model_type).
    // User-set overrides (port, host, apiKey, etc.) are preserved.
    let freshDetectedFamily: string | undefined
    if (!isImageSession) {
      try {
        const freshConfig = detectModelConfigFromDir(config.modelPath)
        if (freshConfig) {
          const freshFamily = normalizeDetectedFamilyName(freshConfig.family)
          freshDetectedFamily = freshFamily
          const oldFamily = config.toolCallParser
          const oldReasoningParser = config.reasoningParser
          // Update auto-detected fields only if user hasn't explicitly overridden them
          // Use === checks, not falsy — '' means "None/disabled" (explicit user choice)
          if (config.toolCallParser === undefined || config.toolCallParser === 'auto') {
            config.toolCallParser = freshConfig.toolParser || 'auto'
          }
          if (config.reasoningParser === undefined || config.reasoningParser === 'auto') {
            config.reasoningParser = freshConfig.reasoningParser || 'auto'
          }
          // v1.5.25 ZAYA recovery: early ZAYA builds were misdetected as Qwen
          // tool parsers. Preserve only explicit "None" (`''`) choices, but
          // keep reasoning on the registry path because ZAYA/ZAYA1-VL declare a
          // qwen3-compatible thinking rail and per-chat On/Off must work.
          if (isZayaCcaFamily(freshFamily)) {
            if (config.toolCallParser !== '') {
              config.toolCallParser = freshConfig.toolParser || 'auto'
            }
            if (config.reasoningParser !== '') {
              config.reasoningParser = freshConfig.reasoningParser || 'auto'
            }
          }
          if (freshFamily === 'deepseek-v4') {
            const dsv4Changed =
              config.continuousBatching !== true ||
              config.enablePrefixCache !== true ||
              config.usePagedCache !== true ||
              config.pagedCacheBlockSize !== DSV4_PAGED_CACHE_BLOCK_SIZE ||
              config.maxNumSeqs !== 1 ||
              config.kvCacheQuantization !== 'auto' ||
              config.enableJit === true ||
              (config as any).smelt === true ||
              (config as any).flashMoe === true ||
              (config as any).distributedEnabled === true ||
              !!config.speculativeModel ||
              config.isMultimodal !== false
            config.continuousBatching = true
            config.enablePrefixCache = true
            config.usePagedCache = true
            config.pagedCacheBlockSize = DSV4_PAGED_CACHE_BLOCK_SIZE
            config.maxNumSeqs = 1
            config.prefillBatchSize = 1
            config.completionBatchSize = 1
            config.kvCacheQuantization = 'auto'
            config.noMemoryAwareCache = false
            config.enableDiskCache = false
            config.enableJit = false
            ;(config as any).smelt = false
            ;(config as any).flashMoe = false
            ;(config as any).distributedEnabled = false
            config.speculativeModel = ''
            config.isMultimodal = false
            if (dsv4Changed) {
              this.pushLog(sessionId, '[INFO] DSV4-Flash detected; stale generic runtime settings were reset to the native SWA+CSA/HCA cache policy')
            }
          }
          // Refresh multimodal detection from disk. A detected VLM must win
          // over stale saved `isMultimodal=false` from older sessions, while a
          // forceTextOnly policy must clear stale true rows. The latter is used
          // for affine-JANG Qwen hybrid until its mlx_vlm M-RoPE path is fixed.
          // Smelt is handled later at launch time because its partial expert
          // support uses text-only loading.
          if (freshConfig.forceTextOnly === true) {
            config.isMultimodal = false
          } else if (freshConfig.isMultimodal === true) {
            config.isMultimodal = true
          } else if (config.isMultimodal === undefined) {
            config.isMultimodal = freshConfig.isMultimodal
          }
          if (
            freshConfig.usePagedCache === true &&
            cacheTypeRequiresPaged(freshConfig.cacheType) &&
            config.continuousBatching !== false &&
            config.enablePrefixCache !== false &&
            config.usePagedCache === false
          ) {
            config.usePagedCache = true
            this.pushLog(sessionId, `[INFO] ${freshConfig.family} ${freshConfig.cacheType} cache requires paged cache; stale saved Use Paged Cache=false was reset to auto-safe true`)
          }
          // Log if model type changed
          if (oldFamily && oldFamily !== 'auto' && freshConfig.toolParser && oldFamily !== freshConfig.toolParser) {
            this.pushLog(sessionId, `[INFO] Model config re-detected from disk (was: ${oldFamily}, now: ${freshConfig.toolParser})`)
          }
          if (oldReasoningParser && oldReasoningParser !== 'auto' && isZayaCcaFamily(freshFamily) && oldReasoningParser !== config.reasoningParser) {
            this.pushLog(sessionId, `[INFO] ZAYA reasoning parser reset from stale ${oldReasoningParser} to auto (no reasoning parser)`)
          }
          // Update DB with refreshed config
          db.updateSession(sessionId, { config: JSON.stringify(config) })
        }
      } catch (e) {
        this.pushLog(sessionId, `[WARN] Could not re-detect model config: ${(e as Error).message}`)
      }
    }

    // Memory estimation: warn if model is too large for available RAM
    const modelFileBytes = estimateModelFileBytes(config.modelPath)
    const modelSizeBytes = estimateModelMemory(config.modelPath)
    if (modelSizeBytes > 0) {
      const availableBytes = freemem()
      const totalBytes = totalmem()
      const usagePercent = ((totalBytes - availableBytes) / totalBytes) * 100
      const modelGB = formatGb(modelSizeBytes)
      const availGB = (availableBytes / 1e9).toFixed(1)
      const totalGB = (totalBytes / 1e9).toFixed(0)
      console.log(`[SESSION] Model estimate: ~${modelGB} GB | RAM: ${availGB} GB free / ${totalGB} GB total (${usagePercent.toFixed(0)}% used)`)
      this.emit('session:log', { sessionId, data: `Model estimate: ~${modelGB} GB | RAM: ${availGB} GB free / ${totalGB} GB total\n` })
      if (modelSizeBytes > availableBytes * 0.9) {
        console.warn(`[SESSION] WARNING: Model (~${modelGB} GB) may exceed available memory (${availGB} GB free). Risk of system instability.`)
        this.emit('session:log', { sessionId, data: `⚠️  Memory warning: Model requires ~${modelGB} GB but only ${availGB} GB free. Loading may cause system instability or swap.\n` })
      } else if (modelSizeBytes > availableBytes * 0.7) {
        console.log(`[SESSION] Model will use most of available RAM (${modelGB} GB / ${availGB} GB free)`)
        this.emit('session:log', { sessionId, data: `Note: Model (~${modelGB} GB) will use most available memory. KV cache may be limited.\n` })
      }
    }

    // Kill anything on this port first
    await this.killByPort(session.port)

    db.updateSession(sessionId, {
      status: 'loading',
      lastStartedAt: Date.now()
    })
    this.loadProgressState.delete(sessionId) // Reset loading progress for fresh start
    this.loadProgressMeta.delete(sessionId)
    this.stopLoadResidentMonitor(sessionId)
    if (modelFileBytes > 0) {
      const meta = { modelBytes: modelFileBytes, lazyResident: true }
      this.loadProgressMeta.set(sessionId, meta)
      this.loadProgressState.set(sessionId, 2)
      this.emit('session:loadProgress', {
        sessionId,
        label: 'Scanning model files...',
        progress: 2,
        ...meta,
      })
    }
    this.emit('session:starting', { sessionId, modelPath: session.modelPath })

    const args = this.buildArgs(config)

    // Ensure PATH includes pyenv/homebrew so the engine finds its Python
    const extraPath = [
      join(homedir(), '.pyenv', 'shims'),
      join(homedir(), '.pyenv', 'bin'),
      '/opt/homebrew/bin',
      '/usr/local/bin',
    ].join(':')
    const spawnEnv: Record<string, string | undefined> = { ...process.env, PATH: `${extraPath}:${process.env.PATH || ''}` }
    // Pass API key via env var (not CLI arg) to avoid exposure in ps aux
    if (config.apiKey) {
      spawnEnv.VLLM_API_KEY = config.apiKey
    }
    // Pass cluster secret via env var for distributed compute (same reason as API key)
    const clusterSecret = (config as any).distributedSecret
    if (clusterSecret) {
      spawnEnv.VMLX_CLUSTER_SECRET = clusterSecret
    }
    // Pass HuggingFace token only for remote repo IDs/URLs. Decrypting the
    // stored token is synchronous in Electron safeStorage and can block the
    // main thread; local bundles do not need HF_TOKEN at session startup.
    if (shouldPassHfTokenToEngine(config.modelPath)) {
      const hfToken = db.getSetting('hf_api_key')
      if (hfToken) {
        spawnEnv.HF_TOKEN = hfToken
      }
    }
    delete spawnEnv.JANGTQ_TOPK_OVERRIDE
    // Acceleration policy is internal and defaults to auto in the engine.
    // Do not let stale/debug parent env values force packaged app sessions
    // onto the legacy or strict experimental lane.
    delete spawnEnv.JANGTQ_MPP_NAX
    delete spawnEnv.JANGTQ_MPP_NAX_DISABLE
    delete spawnEnv.JANGTQ_MPP_NAX_STRICT
    delete spawnEnv.JANGTQ_MPP_DENSE
    delete spawnEnv.JANGTQ_MPP_DENSE_STRICT
    delete spawnEnv.JANGTQ_DISABLE_DSV4_STREAM_LOAD
    delete spawnEnv.JANGTQ_DISABLE_DSV4_FAST_LOAD
    delete spawnEnv.VMLX_DENSE_STRICT_LANE
    delete spawnEnv.VMLX_DSV4_FAST_LOAD_DISABLE
    delete spawnEnv.VMLINUX_DENSE_STRICT_LANE
    delete spawnEnv.VMLINUX_DSV4_FAST_LOAD_DISABLE
    // MCP config and policy must be session-owned. Inheriting these from a
    // developer shell would make the effective tool set differ from the UI.
    delete spawnEnv.VLLM_MLX_MCP_CONFIG
    delete spawnEnv.VLLM_MLX_MCP_ENABLED_SERVERS
    delete spawnEnv.VLLM_MLX_MCP_DISABLED_SERVERS
    delete spawnEnv.VLLM_MLX_MCP_ENABLED_TOOLS
    delete spawnEnv.VLLM_MLX_MCP_DISABLED_TOOLS
    // DSV4 Flash runtime knobs. Helper validates inputs and emits only the
    // env vars the engine reads.
    const dsv4Env = dsv4EnvFromConfig(config as any, {
      dsv4Active: freshDetectedFamily === 'deepseek-v4',
    })
    for (const [key, value] of Object.entries(dsv4Env)) {
      spawnEnv[key] = value
    }
    const scrubbedEnvProbeKeys = [
      'JANGTQ_MPP_NAX',
      'JANGTQ_MPP_NAX_DISABLE',
      'JANGTQ_MPP_NAX_STRICT',
      'JANGTQ_MPP_DENSE',
      'JANGTQ_MPP_DENSE_STRICT',
      'JANGTQ_DISABLE_DSV4_STREAM_LOAD',
      'JANGTQ_DISABLE_DSV4_FAST_LOAD',
      'VMLX_DENSE_STRICT_LANE',
      'VMLX_DSV4_FAST_LOAD_DISABLE',
      'VMLINUX_DENSE_STRICT_LANE',
      'VMLINUX_DSV4_FAST_LOAD_DISABLE',
    ]
    const scrubbedEnvProbe: Record<string, string | null> = {}
    for (const key of scrubbedEnvProbeKeys) {
      scrubbedEnvProbe[key] = spawnEnv[key] ?? null
    }
    this.pushLog(sessionId, `[ENV] engine_child_probe=${JSON.stringify(scrubbedEnvProbe)}`)
    // NOTE: We previously set HF_HUB_OFFLINE=1 for image models to prevent mflux from
    // silently downloading multi-GB models. This was removed because it also blocks mflux
    // from reading already-cached files in ~/.cache/huggingface/hub/. Instead, we rely on
    // validateImageModelCompleteness() to warn users about incomplete downloads before start,
    // and the logs panel shows startup progress if mflux does need to fetch missing components.

    let proc: ChildProcess
    if (engineResult.type === 'bundled') {
      // Bundled Python: spawn python3 -B -s -m vmlx_engine.cli serve <model> --host ... --port ...
      // -B: do not write __pycache__ into the signed app bundle at runtime
      // -s: suppress user site-packages (~/.local/lib/python3.12/site-packages)
      // This avoids shebang path issues with relocatable Python and ensures
      // the app uses ONLY its bundled engine, never system-installed mlx-lm/vmlx-engine.
      const bundledEnv: Record<string, string | undefined> = {
        ...spawnEnv,
        PYTHONDONTWRITEBYTECODE: '1',
        PYTHONNOUSERSITE: '1',  // Extra safety: disable user site-packages
        PYTHONPATH: undefined,  // Clear any inherited PYTHONPATH
        // vmlx#102/#116: brew-installed mlx/mlx-c can collide with bundled mlx
        // via DYLD_*. Clearing those forces the bundled libmlx to be the only
        // one loaded — fixes "duplicate key 'cpu' to enumeration mlx.core.DeviceType".
        DYLD_LIBRARY_PATH: undefined,
        DYLD_FALLBACK_LIBRARY_PATH: undefined,
        DYLD_INSERT_LIBRARIES: undefined,
      }
      const fullCmd = `${engineResult.pythonPath} -B -s -m vmlx_engine.cli ${args.join(' ')}`
      this.pushLog(sessionId, `$ ${fullCmd}`)
      this.emit('session:log', { sessionId, data: `$ ${fullCmd}\n` })
      proc = spawn(engineResult.pythonPath, ['-B', '-s', '-m', 'vmlx_engine.cli', ...args], {
        env: bundledEnv,
        stdio: ['ignore', 'pipe', 'pipe'],
        detached: true,  // Separate process group so we can kill entire group
      })
    } else {
      // System binary: spawn vmlx-engine directly
      const fullCmd = `${engineResult.binaryPath} ${args.join(' ')}`
      this.pushLog(sessionId, `$ ${fullCmd}`)
      this.emit('session:log', { sessionId, data: `$ ${fullCmd}\n` })
      proc = spawn(engineResult.binaryPath, args, {
        env: spawnEnv,
        stdio: ['ignore', 'pipe', 'pipe'],
        detached: true,
      })
    }

    this.processes.set(sessionId, { process: proc, adoptedPid: null })

    proc.stdout?.on('data', (data) => {
      const text = data.toString()
      this.pushLog(sessionId, text)
      this.emit('session:log', { sessionId, data: text })
    })
    proc.stderr?.on('data', (data) => {
      const text = data.toString()
      this.pushLog(sessionId, text)
      // Log errors to main console for diagnostics
      if (text.includes('ERROR') || text.includes('Traceback') || text.includes('Exception')) {
        console.error(`[SERVER] ${text.trimEnd()}`)
      }
      this.emit('session:log', { sessionId, data: text })
      // Capture most meaningful stderr line for error reporting.
      // Python exceptions print the error type several lines before the
      // final output (e.g., RuntimeError on line N, then "library not found"
      // on line N+3). Prefer exception lines over the last line.
      const managed = this.processes.get(sessionId)
      if (managed) {
        const lines = text.trim().split('\n').filter((l: string) => l.trim())
        // Look for Python exception lines (most informative)
        const exceptionLine = lines.find((l: string) =>
          /^(RuntimeError|ImportError|ModuleNotFoundError|OSError|ValueError|TypeError|MemoryError|FileNotFoundError):/.test(l.trim()) ||
          /^(mlx|mflux|torch|jax)\./.test(l.trim()) && l.includes('Error')
        )
        if (exceptionLine) {
          managed.lastStderr = exceptionLine.trim()
        } else if (!managed.lastStderr || !/^(RuntimeError|ImportError|ModuleNotFoundError|OSError|ValueError|TypeError|MemoryError|FileNotFoundError):/.test(managed.lastStderr)) {
          // Only overwrite if we haven't already captured an exception line
          const lastLine = lines.pop()
          if (lastLine) managed.lastStderr = lastLine
        }
      }
    })
    proc.stdout?.on('error', () => { })
    proc.stderr?.on('error', () => { })

    proc.on('exit', (code, signal) => {
      this.stopLoadResidentMonitor(sessionId)
      const managed = this.processes.get(sessionId)
      const lastStderr = managed?.lastStderr
      const intentional = managed?.intentionalStop === true
      this.processes.delete(sessionId)
      this.failCounts.delete(sessionId)
      const killed = signal === 'SIGKILL'
      const crashed = !intentional && (killed || (code !== null && code !== 0))
      db.updateSession(sessionId, {
        status: crashed ? 'error' : 'stopped',
        pid: undefined,
        lastStoppedAt: Date.now()
      })
      if (crashed) {
        let reason: string
        if (killed) {
          reason = 'Process was killed (SIGKILL) — likely out of memory. Try a smaller/more quantized model, reduce cache size, or close other apps.'
        } else if (lastStderr) {
          reason = `Process exited with code ${code}: ${lastStderr}`
        } else {
          reason = `Process exited with code ${code}`
        }
        this.pushLog(sessionId, `[ERROR] ${reason}`)
        this.emit('session:error', { sessionId, error: reason })
      } else {
        this.pushLog(sessionId, `[INFO] Process stopped (exit code ${code})`)
      }
      // Store exit info for waitForReady to access
      this.processes.set(sessionId, { process: null, adoptedPid: null, exitCode: code, exitSignal: signal, lastStderr })
      this.emit('session:stopped', { sessionId, code, signal })
      // Clean up the exit info after a delay so waitForReady can read it
      setTimeout(() => {
        const m = this.processes.get(sessionId)
        if (m && !m.process && !m.adoptedPid) this.processes.delete(sessionId)
      }, 5000)
    })

    proc.on('error', (error) => {
      this.processes.delete(sessionId)
      this.failCounts.delete(sessionId)
      db.updateSession(sessionId, {
        status: 'error',
        pid: undefined
      })
      this.emit('session:error', { sessionId, error: error.message })
    })

    if (proc.pid) {
      db.updateSession(sessionId, { pid: proc.pid })
      this.startLoadResidentMonitor(sessionId, proc.pid, modelFileBytes)
    }

    // Wait for health endpoint — use session timeout (min 120s for large models)
    const startupTimeoutMs = Math.max((config.timeout || 300) * 1000, 120000)
    try {
      await this.waitForReady(session.host, session.port, startupTimeoutMs, sessionId)
      this.stopLoadResidentMonitor(sessionId)
      db.updateSession(sessionId, { status: 'running' })
      this.touchSession(sessionId)  // Start idle timer from model-ready time
      this.emit('session:ready', { sessionId, port: session.port })
    } catch (err) {
      this.stopLoadResidentMonitor(sessionId)
      db.updateSession(sessionId, { status: 'error' })
      this.emit('session:error', { sessionId, error: (err as Error).message })
      throw err
    }
  }

  private async _connectRemoteSession(session: Session): Promise<void> {
    db.updateSession(session.id, { status: 'loading', lastStartedAt: Date.now() })
    this.emit('session:starting', { sessionId: session.id, modelPath: session.modelPath })
    this.pushLog(session.id, `[INFO] Connecting to remote endpoint...`)

    const baseUrl = session.remoteUrl!.replace(/\/+$/, '')
    const headers: Record<string, string> = {}
    if (session.remoteApiKey) headers['Authorization'] = `Bearer ${session.remoteApiKey}`
    if (session.remoteOrganization) headers['OpenAI-Organization'] = session.remoteOrganization

    const url = `${baseUrl}/v1/models`
    const resolvedUrl = await resolveUrl(url)
    this.pushLog(session.id, `[INFO] GET ${url}${resolvedUrl !== url ? ` (resolved: ${resolvedUrl})` : ''}`)
    console.log(`[SESSION] Connecting to remote: ${url}${resolvedUrl !== url ? ` (resolved: ${resolvedUrl})` : ''}`)

    // Retry up to 3 times with increasing delay to handle transient DNS/network issues
    let lastErr: Error | null = null
    for (let attempt = 1; attempt <= 3; attempt++) {
      try {
        const res = await fetch(resolvedUrl, {
          headers,
          signal: AbortSignal.timeout(10000)
        })
        if (!res.ok) throw new Error(`Server returned HTTP ${res.status}`)

        this.pushLog(session.id, `[INFO] Connected to remote endpoint (attempt ${attempt})`)
        console.log(`[SESSION] Remote connected: ${url} (attempt ${attempt})`)
        db.updateSession(session.id, { status: 'running' })
        this.lastHealthyAt.set(session.id, Date.now())
        this.emit('session:ready', { sessionId: session.id, port: session.port })
        return
      } catch (err) {
        lastErr = err as Error
        this.pushLog(session.id, `WARNING: Connect attempt ${attempt}/3 failed: ${lastErr.message}`)
        console.log(`[SESSION] Remote connect attempt ${attempt}/3 failed: ${lastErr.message}`)
        if (attempt < 3) await new Promise(r => setTimeout(r, attempt * 1000))
      }
    }

    this.pushLog(session.id, `[ERROR] Cannot connect to remote endpoint: ${lastErr!.message}`)
    db.updateSession(session.id, { status: 'error' })
    this.emit('session:error', { sessionId: session.id, error: `${lastErr!.message} (${url})` })
    throw new Error(`Cannot connect to remote endpoint ${url}: ${lastErr!.message}`)
  }

  async stopSession(sessionId: string): Promise<void> {
    const session = db.getSession(sessionId)
    if (!session) throw new Error(`Session ${sessionId} not found`)

    // Remote sessions just disconnect (no process to kill) — no lock needed
    if (session.type === 'remote') {
      this.pushLog(sessionId, '[INFO] Disconnected from remote endpoint')
      this.failCounts.delete(sessionId)
      db.updateSession(sessionId, { status: 'stopped', lastStoppedAt: Date.now() })
      this.emit('session:stopped', { sessionId })
      return
    }

    // Serialize start/stop operations per session to prevent races
    await this.withSessionLock(sessionId, async () => {
      this.failCounts.delete(sessionId)
      const managed = this.processes.get(sessionId)

      // Mark intentional stop on managed process to prevent crash misreport
      if (managed) managed.intentionalStop = true

      if (managed?.process) {
        await this.killChildProcess(managed.process)
        this.processes.delete(sessionId)
      } else if (managed?.adoptedPid) {
        this.killPid(managed.adoptedPid)
        await new Promise(r => setTimeout(r, 1500))
        try { process.kill(managed.adoptedPid, 0); this.killPid(managed.adoptedPid, 'SIGKILL') } catch (_) { }
        this.processes.delete(sessionId)
      } else if (session.pid) {
        // Fallback: kill by stored PID
        this.killPid(session.pid)
        await new Promise(r => setTimeout(r, 1500))
        try { process.kill(session.pid, 0); this.killPid(session.pid, 'SIGKILL') } catch (_) { }
      } else {
        // Last resort: kill by port
        await this.killByPort(session.port)
      }

      db.updateSession(sessionId, {
        status: 'stopped',
        pid: undefined,
        lastStoppedAt: Date.now(),
        standbyDepth: null
      })
      // Clear idle tracking and log buffer
      this.lastRequestAt.delete(sessionId)
      this.logBuffers.delete(sessionId)
      this.emit('session:stopped', { sessionId })
    })
  }

  async deleteSession(sessionId: string): Promise<void> {
    // Stop first if running
    const session = db.getSession(sessionId)
    if (session && (session.status === 'running' || session.status === 'loading' || session.status === 'standby')) {
      await this.stopSession(sessionId)
    }

    // Acquire lock to prevent race with concurrent startSession
    await this.withSessionLock(sessionId, async () => {
      this.processes.delete(sessionId)
      this.failCounts.delete(sessionId)
      this.logBuffers.delete(sessionId)
      db.deleteSession(sessionId)
      this.emit('session:deleted', { sessionId })
    })
  }

  /** Config keys that require a session restart to take effect (all CLI args). */
  private static readonly RESTART_REQUIRED_KEYS = new Set([
    'port', 'host', 'modelPath', 'continuousBatching', 'enablePrefixCache',
    'usePagedCache', 'pagedCacheBlockSize', 'maxCacheBlocks',
    'noMemoryAwareCache', 'cacheMemoryMb', 'cacheMemoryPercent',
    'kvCacheQuantization', 'kvCacheGroupSize',
    'enableDiskCache', 'diskCacheMaxGb', 'diskCacheDir',
    'enableBlockDiskCache', 'blockDiskCacheMaxGb', 'blockDiskCacheDir',
    'prefixCacheSize', 'prefixCacheMaxBytes', 'cacheTtlMinutes', 'isMultimodal',
    'toolCallParser', 'reasoningParser',
    'dsv4PoolQuant',
    'maxNumSeqs', 'prefillBatchSize', 'prefillStepSize', 'completionBatchSize',
    'streamInterval', 'apiKey', 'rateLimit',
    // NOTE: 'timeout' intentionally omitted — client sends per-request timeout
    // to server in the request body (chat.ts:818), so changes take effect immediately.
    'maxTokens', 'maxContextLength', 'mcpConfig',
    'mcpEnabledServers', 'mcpDisabledServers', 'mcpEnabledTools', 'mcpDisabledTools',
    'servedModelName',
    'speculativeModel', 'numDraftTokens', 'smelt', 'smeltExperts',
    'nativeMtpMode', 'nativeMtpDepth', 'nativeMtpDepthOverride',
    'flashMoe', 'flashMoeSlotBank', 'flashMoePrefetch', 'flashMoeIoSplit',
    'distributedEnabled', 'distributedMode', 'distributedSecret',
    'embeddingModel', 'additionalArgs', 'mfluxClass',
    'enableAutoToolChoice', 'chatTemplate',
    'logLevel', 'corsOrigins',
    'enableJit',
    'imageMode', 'imageQuantize',
    // VLM video sampling (Qwen 3.6 / Qwen3.5-VL) — no CLI-restart needed.
    // chat.ts reads sessionConfig.videoFps / videoMaxFrames and forwards as
    // video_fps / video_max_frames on each request body.
    'videoFps', 'videoMaxFrames',
  ])

  async updateSessionConfig(sessionId: string, config: Partial<ServerConfig>): Promise<{ restartRequired: boolean; changedKeys: string[] }> {
    const session = db.getSession(sessionId)
    if (!session) throw new Error(`Session ${sessionId} not found`)

    // Validate port if provided
    if (config.port !== undefined) {
      if (config.port < 1024 || config.port > 65535) {
        throw new Error(`Invalid port ${config.port}. Must be between 1024 and 65535.`)
      }
      // Check for port conflicts with other LOCAL sessions (remote sessions don't bind ports).
      // Only block if another session is actually running or loading on that port.
      const allSessions = db.getSessions()
      const conflicting = allSessions.find(s =>
        s.port === config.port &&
        s.id !== sessionId &&
        s.type === 'local' &&
        (s.status === 'running' || s.status === 'loading' || s.status === 'standby')
      )
      if (conflicting) {
        throw new Error(`Port ${config.port} is in use by running session "${conflicting.modelName || conflicting.modelPath}".`)
      }
      // Block if this port matches the API Gateway port (#44)
      const gwPort = parseInt(db.getSetting('gateway_port') || '8080', 10)
      if (config.port === gwPort) {
        throw new Error(`Port ${config.port} is in use by the API Gateway. Choose a different port.`)
      }
    }

    let currentConfig: Record<string, unknown> = {}
    try {
      currentConfig = JSON.parse(session.config)
    } catch {
      // Corrupted config in DB — start fresh
    }
    // Strip undefined values before merging — prevents config spread from
    // overwriting existing DB values with undefined (which JSON.stringify would then drop)
    const cleanConfig: Record<string, unknown> = {}
    for (const [k, v] of Object.entries(config as Record<string, unknown>)) {
      if (v !== undefined) cleanConfig[k] = v
    }
    const merged = { ...currentConfig, ...cleanConfig }
    markCacheStackStartupDefaultsCurrent(merged as Partial<ServerConfig>)

    // Log sleep config changes
    if ('idleTimeoutSoftMin' in cleanConfig || 'idleTimeoutHardMin' in cleanConfig || 'autoSleepEnabled' in cleanConfig) {
      console.log(`[SLEEP] Config saved for ${sessionId.slice(0, 8)}: soft=${merged.idleTimeoutSoftMin}min, hard=${merged.idleTimeoutHardMin}min, enabled=${merged.autoSleepEnabled}`)
    }

    // Always keep host/port in sync between JSON blob and DB columns
    // Extract from merged to ensure JSON blob and DB columns always agree
    const host = (merged.host as string) || session.host
    const port = (merged.port as number) || session.port

    db.updateSession(sessionId, {
      config: JSON.stringify(merged),
      host,
      port
    })

    // H6: Determine if changed keys require a restart
    const isRunning = session.status === 'running' || session.status === 'loading'
    const changedKeys = Object.keys(cleanConfig).filter(k =>
      SessionManager.RESTART_REQUIRED_KEYS.has(k) &&
      (cleanConfig as Record<string, unknown>)[k] !== currentConfig[k]
    )
    return {
      restartRequired: isRunning && changedKeys.length > 0,
      changedKeys,
    }
  }

  // ─── Discovery & Adoption ─────────────────────────────────────────

  async detectAndAdoptAll(): Promise<Session[]> {
    const processes = await this.detect()
    const adopted: Session[] = []

    for (const proc of processes) {
      if (!proc.healthy) continue

      // Normalize detected path for consistent DB matching
      proc.modelPath = normalizePath(proc.modelPath)

      // Check if we already have a session for this model path
      let session = db.getSessionByModelPath(proc.modelPath)

      // Determine correct session status from health response
      const adoptStatus = proc.standbyDepth ? 'standby' : 'running'
      const adoptStandbyDepth = proc.standbyDepth || null

      if (!session) {
        // Create a new session record for this detected process
        // Use full defaults so the settings page shows complete config
        const id = uuidv4()
        const now = Date.now()
        // Auto-detect model config for proper defaults (paged cache, parsers, etc.)
        const detected = detectModelConfigFromDir(proc.modelPath)
        // Defaults tuned for local single-user cache correctness. Continuous
        // batching is the backend path that enables prefix cache, paged KV,
        // block disk L2, and stored-cache codecs; maxNumSeqs=1 avoids a large
        // multi-user batch shape while keeping those features active.
        // Stream interval 1 = lowest latency per-token delivery.
        const defaultConfig: ServerConfig = {
          modelPath: proc.modelPath,
          host: '127.0.0.1',
          port: proc.port,
          timeout: normalizeDetectedFamilyName(detected.family) === 'deepseek-v4'
            ? DSV4_DEFAULT_TIMEOUT_SECONDS
            : GENERIC_DEFAULT_TIMEOUT_SECONDS,
          maxNumSeqs: 1,
          prefillBatchSize: 512,
          prefillStepSize: 2048,
          completionBatchSize: 512,
          continuousBatching: true,
          enablePrefixCache: true,
          prefixCacheSize: 100,
          prefixCacheMaxBytes: 0, // 0 = unlimited (bounded by cacheMemoryPercent)
          cacheMemoryMb: 0,
          cacheMemoryPercent: 15,
          noMemoryAwareCache: false,
          usePagedCache: detected.usePagedCache ?? true,
          pagedCacheBlockSize: normalizeDetectedFamilyName(detected.family) === 'deepseek-v4' ? DSV4_PAGED_CACHE_BLOCK_SIZE : 64,
          maxCacheBlocks: 1000,
          enableBlockDiskCache: true,
          blockDiskCacheMaxGb: 10,
          kvCacheQuantization: 'auto',
          cacheStackStartupDefaultsVersion: CACHE_STACK_STARTUP_DEFAULTS_VERSION,
          streamInterval: 1,
          maxTokens: 0,
          maxContextLength: 0,
          toolCallParser: 'auto',
          reasoningParser: 'auto',
          dsv4PoolQuant: false,
          defaultEnableThinking: undefined,
          nativeMtpMode: 'deterministic',
          nativeMtpDepth: (detected as any).nativeMtp?.depth ?? 3,
          nativeMtpDepthOverride: false,
          enableAutoToolChoice: detected.enableAutoToolChoice
        }
        session = {
          id,
          modelPath: proc.modelPath,
          modelName: proc.modelName || proc.modelPath.split('/').pop() || proc.modelPath,
          host: '127.0.0.1',
          port: proc.port,
          pid: proc.pid,
          status: adoptStatus,
          config: JSON.stringify(defaultConfig),
          createdAt: now,
          updatedAt: now,
          lastStartedAt: now,
          type: 'local',
          standbyDepth: adoptStandbyDepth
        }
        db.createSession(session)
      } else {
        // Update existing session with live process info (also normalize stored path)
        db.updateSession(session.id, {
          status: adoptStatus,
          standbyDepth: adoptStandbyDepth,
          pid: proc.pid,
          port: proc.port,
          modelPath: normalizePath(session.modelPath),
          modelName: proc.modelName || session.modelName,
          lastStartedAt: Date.now()
        })
        session = db.getSession(session.id)!
      }

      this.processes.set(session.id, { process: null, adoptedPid: proc.pid })
      adopted.push(session)
    }

    // Mark sessions that were running but no longer have a process
    const allSessions = db.getSessions()
    for (const s of allSessions) {
      if (s.status === 'running' || s.status === 'loading' || s.status === 'standby') {
        if (s.type === 'remote') {
          console.log(`[SESSIONS] Resetting stale remote session "${s.modelName}" to stopped (was ${s.status})`)
          db.updateSession(s.id, { status: 'stopped', standbyDepth: null })
          this.emit('session:stopped', { sessionId: s.id })
        } else if (!adopted.find(a => a.id === s.id)) {
          db.updateSession(s.id, { status: 'stopped', pid: undefined, standbyDepth: null })
        }
      }
    }

    return adopted
  }

  // ─── Global Health Monitor ─────────────────────────────────────────

  startGlobalMonitor(): void {
    if (this.monitorInterval) return

    this.monitorInterval = setInterval(async () => {
      const sessions = db.getSessions()

      for (const session of sessions) {
        // Skip stopped/error sessions. Standby sessions are monitored for health but not fail-counted.
        if (session.status === 'stopped' || session.status === 'error') continue

        // Standby sessions: just check process is alive, don't fail-count
        if (session.status === 'standby') {
          if (session.type !== 'remote' && session.pid) {
            const alive = this.isProcessAlive(session.id, session.pid)
            if (!alive) {
              db.updateSession(session.id, { status: 'stopped', standbyDepth: null })
              this.emit('session:stopped', { sessionId: session.id })
              this.pushLog(session.id, '[Sleep] Process died during standby')
              continue
            }
            // Check if model was woken externally (e.g., external curl triggered JIT wake)
            try {
              const res = await fetch(
                `http://${connectHost(session.host)}:${session.port}/health`,
                { signal: AbortSignal.timeout(3000) }
              )
              if (res.ok) {
                const data = await res.json()
                if (data.status === 'healthy') {
                  // Model woke externally — sync DB to running
                  db.updateSession(session.id, { status: 'running', standbyDepth: null })
                  this.touchSession(session.id)
                  this.emit('session:ready', { sessionId: session.id, port: session.port })
                  this.pushLog(session.id, '[Wake] Model woke externally — synced to running')
                }
              }
            } catch {
              // Health check failed — process alive but server unresponsive, keep standby
            }
          }
          continue
        }

        if (session.status !== 'running' && session.status !== 'loading') continue

        // Remote sessions: check /v1/models instead of /health
        if (session.type === 'remote') {
          if (!session.remoteUrl) {
            db.updateSession(session.id, { status: 'error' })
            this.emit('session:error', { sessionId: session.id, error: 'Missing remote URL' })
            continue
          }
          try {
            const remoteBase = session.remoteUrl.replace(/\/+$/, '')
            const remoteHeaders: Record<string, string> = {}
            if (session.remoteApiKey) remoteHeaders['Authorization'] = `Bearer ${session.remoteApiKey}`
            if (session.remoteOrganization) remoteHeaders['OpenAI-Organization'] = session.remoteOrganization
            const resolvedHealthUrl = await resolveUrl(`${remoteBase}/v1/models`)
            const pingStart = Date.now()
            const res = await fetch(resolvedHealthUrl, {
              headers: remoteHeaders,
              signal: AbortSignal.timeout(10000)
            })
            const latencyMs = Date.now() - pingStart
            if (res.ok) {
              this.failCounts.delete(session.id)
              this.lastHealthyAt.set(session.id, Date.now())
              if (session.status === 'loading') {
                this.loadProgressState.set(session.id, 100)
                this.emit('session:loadProgress', { sessionId: session.id, label: 'Connected', progress: 100 })
                db.updateSession(session.id, { status: 'running' })
                this.emit('session:ready', { sessionId: session.id, port: session.port })
              }
              this.emit('session:health', {
                sessionId: session.id,
                running: true,
                modelName: session.remoteModel,
                port: session.port,
                latencyMs
              })
            } else {
              this.incrementFailAndCheck(session.id)
            }
          } catch (_) {
            // Remote server unresponsive — likely busy with inference.
            // Use dampened counting (every 3rd failure) like local sessions,
            // since remote servers have no PID to check liveness.
            this.emit('session:health', {
              sessionId: session.id,
              running: true,
              busy: true,
              modelName: session.remoteModel,
              port: session.port
            })
            const count = this.failCounts.get(session.id) || 0
            if (count % 3 === 0) {
              this.incrementFailAndCheck(session.id)
            } else {
              this.failCounts.set(session.id, count + 1)
            }
          }
          continue
        }

        try {
          const res = await fetch(
            `http://${connectHost(session.host)}:${session.port}/health`,
            { signal: AbortSignal.timeout(10000) }
          )
          if (res.ok) {
            const data = await res.json()
            // Handle standby states from server
            const isStandby = data.status?.startsWith('standby_')
            // Only count as truly healthy if the model is loaded (status: "healthy")
            // The server returns "no_model" while still loading in lifespan()
            const modelReady = data.status === 'healthy'
            if (isStandby) {
              // Server is in standby — keep session alive, don't fail-count
              this.failCounts.delete(session.id)
            } else if (modelReady) {
              // Reset fail counter on success
              this.failCounts.delete(session.id)
              this.lastHealthyAt.set(session.id, Date.now())
              if (data.model_name && data.model_name !== session.modelName) {
                db.updateSession(session.id, { modelName: data.model_name })
              }
              if (session.status === 'loading') {
                // Emit 100% progress so bar completes before disappearing
                this.loadProgressState.set(session.id, 100)
                this.emit('session:loadProgress', { sessionId: session.id, label: 'Model ready', progress: 100 })
                db.updateSession(session.id, { status: 'running', standbyDepth: null })
                this.touchSession(session.id)
                this.emit('session:ready', { sessionId: session.id, port: session.port })
              }
              // Sync server-side last_request_time to idle timer — catches direct API
              // requests (curl, benchmarks, external tools) that bypass Electron IPC
              if (data.last_request_time) {
                const serverLastReq = Math.round(data.last_request_time * 1000) // Python epoch → JS epoch
                const electronLastReq = this.lastRequestAt.get(session.id) || 0
                if (serverLastReq > electronLastReq) {
                  this.lastRequestAt.set(session.id, serverLastReq)
                  db.updateSession(session.id, { lastRequestAt: serverLastReq })
                }
              }
            } else if (isStandby && session.status === 'loading') {
              // Wake failed — server reverted to standby but DB says loading.
              // Sync DB back to standby so user can retry.
              const depth = data.status === 'standby_deep' ? 'deep' : 'soft'
              db.updateSession(session.id, { status: 'standby', standbyDepth: depth })
              this.emit('session:standby', { sessionId: session.id, depth })
              this.pushLog(session.id, `[Wake] Model reload failed — reverted to ${depth} sleep`)
            } else if (session.status === 'loading') {
              // Server is up but model not loaded yet — update progress bar
              // to show we're past server startup, now waiting for model
              const current = this.loadProgressState.get(session.id) ?? 0
              if (current < 95) {
                this.loadProgressState.set(session.id, 95)
                this.emit('session:loadProgress', {
                  sessionId: session.id,
                  label: 'Model runtime still loading...',
                  progress: 95,
                  ...(this.loadProgressMeta.get(session.id) || {}),
                })
              }
            }
            this.emit('session:health', {
              sessionId: session.id,
              running: modelReady,
              status: modelReady ? 'ok' : 'loading',
              modelName: data.model_name,
              port: session.port,
              memory: data.memory  // { active_mb, peak_mb, cache_mb } from /health
            })
          } else {
            this.incrementFailAndCheck(session.id)
          }
        } catch (_) {
          // Health check timed out or failed — check if process is still alive
          // Long prefills block the event loop, so the server can't respond
          // but the process is still running fine
          if (this.isProcessAlive(session.id, session.pid)) {
            // Process alive but unresponsive (likely busy with long prefill)
            // Emit a "busy" health event so the UI knows the server isn't dead
            this.emit('session:health', {
              sessionId: session.id,
              running: true,
              busy: true,
              modelName: session.modelName,
              port: session.port
            })
            // Only count every 3rd failure to avoid false positives
            const count = this.failCounts.get(session.id) || 0
            if (count % 3 === 0) {
              this.incrementFailAndCheck(session.id)
            } else {
              this.failCounts.set(session.id, count + 1)
            }
          } else {
            // Process is truly dead — fast-track to marking down
            this.incrementFailAndCheck(session.id)
          }
        }
      }

      // Check for idle sessions that should enter sleep
      await this.checkIdleSessions()

      // Prevent macOS system sleep while any model is actively running
      this.updatePowerBlocker()
    }, 5000)
  }

  stopGlobalMonitor(): void {
    if (this.monitorInterval) {
      clearInterval(this.monitorInterval)
      this.monitorInterval = null
    }
    this.releasePowerBlocker()
  }

  /** Start/stop powerSaveBlocker based on whether any session is running */
  private updatePowerBlocker(): void {
    const sessions = db.getSessions()
    const hasActive = sessions.some(
      (s: any) => s.type !== 'remote' && (s.status === 'running' || s.status === 'loading')
    )
    if (hasActive && this.powerBlockerId === -1) {
      this.powerBlockerId = powerSaveBlocker.start('prevent-app-suspension')
      console.log(`[POWER] System sleep blocked (id=${this.powerBlockerId})`)
    } else if (!hasActive && this.powerBlockerId !== -1) {
      this.releasePowerBlocker()
    }
  }

  private releasePowerBlocker(): void {
    if (this.powerBlockerId !== -1) {
      powerSaveBlocker.stop(this.powerBlockerId)
      console.log(`[POWER] System sleep unblocked (id=${this.powerBlockerId})`)
      this.powerBlockerId = -1
    }
  }

  // ── Idle / Sleep Management ──

  /** Mark a session as having received a request (resets idle timer) */
  touchSession(sessionId: string): void {
    const now = Date.now()
    const prev = this.lastRequestAt.get(sessionId) || 0
    this.lastRequestAt.set(sessionId, now)
    db.updateSession(sessionId, { lastRequestAt: now })
    // Log touch events to help debug idle timer issues (only if previous touch was >10s ago)
    if (now - prev > 10000) {
      console.log(`[SLEEP] touchSession ${sessionId.slice(0, 8)} — idle timer reset`)
    }
  }

  /** Build auth headers for admin API calls (sleep/wake) when session has an API key */
  private _adminHeaders(session: import('./database').Session): Record<string, string> {
    try {
      const cfg = JSON.parse(session.config)
      if (cfg.apiKey) return { 'Authorization': `Bearer ${cfg.apiKey}` }
    } catch { /* no config or no key */ }
    return {}
  }

  /** Get idle timeouts for a session based on its model type */
  private getIdleTimeouts(session: import('./database').Session): { softMs: number; hardMs: number } {
    // Determine if this is an image session + read per-session overrides in one parse
    let isImage = false
    let perSessionSoft: number | undefined
    let perSessionHard: number | undefined
    try {
      const cfg = JSON.parse(session.config)
      isImage = cfg.modelType === 'image'
      // Accept each timeout independently (don't require BOTH to be set)
      if (typeof cfg.idleTimeoutSoftMin === 'number') perSessionSoft = cfg.idleTimeoutSoftMin
      if (typeof cfg.idleTimeoutHardMin === 'number') perSessionHard = cfg.idleTimeoutHardMin
    } catch {}

    // Defaults based on model type
    const defaultSoftMs = isImage ? SessionManager.DEFAULT_SOFT_TIMEOUT_IMAGE_MS : SessionManager.DEFAULT_SOFT_TIMEOUT_TEXT_MS
    const defaultHardMs = isImage ? SessionManager.DEFAULT_HARD_TIMEOUT_IMAGE_MS : SessionManager.DEFAULT_HARD_TIMEOUT_TEXT_MS

    // Check global settings
    const globalSoftStr = db.getSetting('idle_timeout_soft_min')
    const globalHardStr = db.getSetting('idle_timeout_hard_min')

    // Priority: per-session > global > model-type default (each timeout resolved independently)
    const softMs = perSessionSoft != null ? perSessionSoft * 60 * 1000
      : globalSoftStr ? parseInt(globalSoftStr) * 60 * 1000
      : defaultSoftMs
    const hardMs = perSessionHard != null ? perSessionHard * 60 * 1000
      : globalHardStr ? parseInt(globalHardStr) * 60 * 1000
      : defaultHardMs

    return { softMs, hardMs }
  }

  /** Check if auto-sleep is enabled (global setting, default true) */
  private isAutoSleepEnabled(): boolean {
    const setting = db.getSetting('auto_sleep_enabled')
    return setting !== '0' && setting !== 'false'
  }

  /** Trigger soft sleep on a session — clear caches, model stays loaded */
  async softSleep(sessionId: string): Promise<{ success: boolean; error?: string }> {
    const session = db.getSession(sessionId)
    if (!session || session.status !== 'running') {
      return { success: false, error: 'Session not running' }
    }
    if (session.type === 'remote') {
      return { success: false, error: 'Cannot sleep remote sessions' }
    }

    try {
      const host = connectHost(session.host)
      const headers = this._adminHeaders(session)
      const res = await fetch(`http://${host}:${session.port}/admin/soft-sleep`, {
        method: 'POST',
        headers,
        signal: AbortSignal.timeout(10000)
      })
      if (res.ok) {
        db.updateSession(sessionId, { status: 'standby', standbyDepth: 'soft' })
        this.emit('session:standby', { sessionId, depth: 'soft' })
        this.pushLog(sessionId, '[Sleep] Entered soft sleep — caches cleared, model loaded')
        return { success: true }
      }
      return { success: false, error: `Server returned ${res.status}` }
    } catch (e) {
      return { success: false, error: (e as Error).message }
    }
  }

  /** Trigger deep sleep on a session — unload model, process stays alive */
  async deepSleep(sessionId: string): Promise<{ success: boolean; error?: string }> {
    const session = db.getSession(sessionId)
    if (!session || (session.status !== 'running' && session.status !== 'standby')) {
      return { success: false, error: 'Session not running or standby' }
    }
    if (session.type === 'remote') {
      return { success: false, error: 'Cannot sleep remote sessions' }
    }

    try {
      const host = connectHost(session.host)
      const headers = this._adminHeaders(session)
      const res = await fetch(`http://${host}:${session.port}/admin/deep-sleep`, {
        method: 'POST',
        headers,
        signal: AbortSignal.timeout(10000)
      })
      if (res.ok) {
        db.updateSession(sessionId, { status: 'standby', standbyDepth: 'deep' })
        this.emit('session:standby', { sessionId, depth: 'deep' })
        this.pushLog(sessionId, '[Sleep] Entered deep sleep — model unloaded, port alive')
        return { success: true }
      }
      return { success: false, error: `Server returned ${res.status}` }
    } catch (e) {
      return { success: false, error: (e as Error).message }
    }
  }

  /** Wake a session from any sleep state — reload model */
  async wakeSession(sessionId: string): Promise<{ success: boolean; error?: string }> {
    const session = db.getSession(sessionId)
    if (!session || session.status !== 'standby') {
      return { success: false, error: 'Session not in standby' }
    }
    if (session.type === 'remote') {
      return { success: false, error: 'Cannot wake remote sessions' }
    }

    try {
      const host = connectHost(session.host)
      const headers = this._adminHeaders(session)
      // 120s timeout — admin/wake does synchronous model load (JANG mmap ~9s, large models 30-60s)
      const res = await fetch(`http://${host}:${session.port}/admin/wake`, {
        method: 'POST',
        headers,
        signal: AbortSignal.timeout(120000)
      })
      if (res.ok) {
        db.updateSession(sessionId, { status: 'loading', standbyDepth: null })
        this.loadProgressState.delete(sessionId)  // Reset progress for fresh wake
        this.loadProgressMeta.delete(sessionId)
        const modelFileBytes = estimateModelFileBytes(session.modelPath)
        const meta = modelFileBytes > 0 ? { modelBytes: modelFileBytes, lazyResident: true } : {}
        if (modelFileBytes > 0) this.loadProgressMeta.set(sessionId, meta)
        this.emit('session:loadProgress', { sessionId, label: 'Waking from sleep...', progress: 50, ...meta })
        if (session.pid && modelFileBytes > 0) {
          this.startLoadResidentMonitor(sessionId, session.pid, modelFileBytes)
        }
        this.emit('session:starting', { sessionId })
        this.pushLog(sessionId, '[Wake] Waking from sleep — reloading model...')
        // The global monitor will pick up the 'loading' status and wait for /health
        this.touchSession(sessionId)
        return { success: true }
      }
      return { success: false, error: `Server returned ${res.status}` }
    } catch (e) {
      return { success: false, error: (e as Error).message }
    }
  }

  /** Check idle sessions and trigger sleep transitions (called from monitor) */
  private async checkIdleSessions(): Promise<void> {
    if (!this.isAutoSleepEnabled()) return

    const sessions = db.getSessions()
    const now = Date.now()

    for (const session of sessions) {
      if (session.type === 'remote') continue
      if (session.status !== 'running' && session.status !== 'standby') continue

      // Check per-session autoSleepEnabled override
      let autoSleepDisabled = false
      try {
        const cfg = JSON.parse(session.config)
        if (cfg.autoSleepEnabled === false) autoSleepDisabled = true
      } catch {}
      if (autoSleepDisabled) continue

      const mapTs = this.lastRequestAt.get(session.id)
      const lastReq = mapTs || session.lastRequestAt || session.lastStartedAt || 0
      if (!lastReq) continue

      const idleMs = now - lastReq
      const { softMs, hardMs } = this.getIdleTimeouts(session)

      // Skip if timeouts are 0 (disabled)
      if (softMs <= 0 && hardMs <= 0) continue

      if (session.status === 'running' && softMs > 0 && idleMs >= softMs) {
        // Running and idle past soft timeout → soft sleep
        console.log(`[SLEEP] Session ${session.id.slice(0, 8)} idle ${Math.round(idleMs / 1000)}s >= soft ${Math.round(softMs / 1000)}s → soft sleep`)
        this.pushLog(session.id, `[Sleep] Idle for ${Math.round(idleMs / 60000)}min — entering soft sleep (timeout: ${Math.round(softMs / 60000)}min)`)
        await this.softSleep(session.id)
      } else if (session.status === 'running' && softMs <= 0 && hardMs > 0 && idleMs >= hardMs) {
        // Soft sleep disabled but hard sleep enabled → skip soft, go straight to deep
        console.log(`[SLEEP] Session ${session.id.slice(0, 8)} idle ${Math.round(idleMs / 1000)}s >= hard ${Math.round(hardMs / 1000)}s → deep sleep (soft disabled)`)
        this.pushLog(session.id, `[Sleep] Idle for ${Math.round(idleMs / 60000)}min — entering deep sleep (timeout: ${Math.round(hardMs / 60000)}min, soft sleep disabled)`)
        await this.deepSleep(session.id)
      } else if (session.status === 'standby' && session.standbyDepth === 'soft' && hardMs > 0 && idleMs >= hardMs) {
        // In soft sleep and idle past hard timeout → deep sleep
        console.log(`[SLEEP] Session ${session.id.slice(0, 8)} idle ${Math.round(idleMs / 1000)}s >= hard ${Math.round(hardMs / 1000)}s → deep sleep`)
        this.pushLog(session.id, `[Sleep] Idle for ${Math.round(idleMs / 60000)}min — entering deep sleep (timeout: ${Math.round(hardMs / 60000)}min)`)
        await this.deepSleep(session.id)
      }
    }
  }

  private incrementFailAndCheck(sessionId: string): void {
    const count = (this.failCounts.get(sessionId) || 0) + 1
    this.failCounts.set(sessionId, count)

    const session = db.getSession(sessionId)

    // For local sessions: if process is dead, mark down immediately
    // For remote sessions: skip this check — they have no PID, so isProcessAlive
    // always returns false. Use the normal fail-count threshold instead.
    if (session?.type !== 'remote' && session && !this.isProcessAlive(sessionId, session.pid)) {
      console.log(`[SESSIONS] Process dead for session ${sessionId} (fail #${count}), marking down`)
      this.failCounts.delete(sessionId)
      this.handleSessionDown(sessionId)
      return
    }

    // Scale max failures with session timeout (5s interval → timeout/5, min 60)
    let maxFails = SessionManager.MAX_FAIL_COUNT
    if (session) {
      try {
        const cfg = JSON.parse(session.config)
        if (cfg.timeout && cfg.timeout > 0) {
          maxFails = Math.max(60, Math.ceil(cfg.timeout / 5))
        }
      } catch (_) { }
    }

    if (count >= maxFails) {
      console.log(`[SESSIONS] Health check failed ${count}x for session ${sessionId} (limit ${maxFails}), marking down`)
      this.failCounts.delete(sessionId)
      this.handleSessionDown(sessionId)
    } else if (count % 10 === 0) {
      // Only log every 10th failure to reduce noise (process is likely doing a long prefill)
      console.log(`[SESSIONS] Health check failed ${count}/${maxFails} for session ${sessionId} (process alive, likely busy)`)
    }
  }

  private handleSessionDown(sessionId: string): void {
    const session = db.getSession(sessionId)
    if (session && (session.status === 'running' || session.status === 'loading')) {
      if (session.type === 'remote') {
        // Remote endpoint truly unreachable after sustained failures — mark as error.
        // Unlike local sessions, there's no process to kill. The user needs to know
        // the endpoint is down so they can fix it or restart.
        this.pushLog(sessionId, '[ERROR] Remote endpoint unreachable after sustained failures')
        console.log(`[SESSIONS] handleSessionDown: remote session ${sessionId} ("${session.modelName}") unreachable, marking error`)
        db.updateSession(sessionId, { status: 'error' })
        this.failCounts.delete(sessionId)
        this.emit('session:error', { sessionId, error: 'Remote endpoint unreachable' })
        return
      } else {
        // Kill the process before marking stopped — without this, the Python
        // process continues running as an orphan consuming RAM/CPU.
        const managed = this.processes.get(sessionId)
        const pid = managed?.adoptedPid ?? managed?.process?.pid ?? session.pid
        if (pid) {
          console.log(`[SESSIONS] handleSessionDown: killing PID ${pid} for session ${sessionId}`)
          this.killPid(pid, 'SIGTERM')
          // Schedule SIGKILL escalation after 3s (non-blocking)
          setTimeout(() => {
            try {
              process.kill(pid, 0) // Check if still alive
              console.log(`[SESSIONS] handleSessionDown: escalating to SIGKILL for PID ${pid}`)
              this.killPid(pid, 'SIGKILL')
            } catch (_) {
              // Already dead — good
            }
          }, 3000)
        } else if (session.port) {
          // Fallback: kill by port
          this.killByPort(session.port).catch(() => { })
        }
        this.processes.delete(sessionId)
      }
      this.failCounts.delete(sessionId)
      // Abort any in-flight SSE streams before marking down
      if (session.host && session.port) {
        this.emit('session:abortInference', { sessionId, host: session.host, port: session.port })
      }
      db.updateSession(sessionId, {
        status: 'error',
        pid: undefined,
        lastStoppedAt: Date.now()
      })
      this.emit('session:error', { sessionId, error: 'Session became unresponsive' })
    }
  }

  // ─── Stop All ──────────────────────────────────────────────────────

  async stopAll(): Promise<void> {
    this.stopGlobalMonitor()

    const processes = await this.detect()

    // B3: Send SIGTERM first to all processes for graceful shutdown
    for (const proc of processes) {
      this.killPid(proc.pid, 'SIGTERM')
    }
    for (const [, managed] of this.processes) {
      if (managed.process) {
        try { managed.process.kill('SIGTERM') } catch (_) { }
      } else if (managed.adoptedPid) {
        this.killPid(managed.adoptedPid, 'SIGTERM')
      }
    }

    // Wait for graceful shutdown, then SIGKILL any survivors
    if (processes.length > 0 || this.processes.size > 0) {
      await new Promise(r => setTimeout(r, 3000))
      for (const proc of processes) {
        try { process.kill(proc.pid, 'SIGKILL') } catch (_) { }
      }
      for (const [, managed] of this.processes) {
        if (managed.process) {
          try { managed.process.kill('SIGKILL') } catch (_) { }
        } else if (managed.adoptedPid) {
          try { process.kill(managed.adoptedPid, 'SIGKILL') } catch (_) { }
        }
      }
    }

    this.processes.clear()

    // Mark all sessions as stopped in DB (including standby — their processes were killed above)
    const sessions = db.getSessions()
    for (const s of sessions) {
      if (s.status === 'running' || s.status === 'loading' || s.status === 'standby') {
        db.updateSession(s.id, { status: 'stopped', pid: undefined, lastStoppedAt: Date.now(), standbyDepth: null })
      }
    }
  }

  // ─── Queries ───────────────────────────────────────────────────────

  getSessions(): Session[] {
    return db.getSessions()
  }

  getSession(id: string): Session | undefined {
    return db.getSession(id)
  }

  getSessionByModelPath(modelPath: string): Session | undefined {
    return db.getSessionByModelPath(normalizePath(modelPath))
  }

  // ─── Helpers (from ServerManager) ──────────────────────────────────

  buildArgs(config: ServerConfig): string[] {
    const args = ['serve', config.modelPath]
    const isImage = config.modelType === 'image'
    const detected = detectModelConfigFromDir(config.modelPath)

    // Server settings — always pass explicitly (both text and image)
    args.push('--host', config.host)
    args.push('--port', config.port.toString())
    args.push('--timeout', effectiveSessionTimeoutSeconds(config, detected.family).toString())

    if (config.rateLimit && config.rateLimit > 0) args.push('--rate-limit', config.rateLimit.toString())
    // API key passed via VLLM_API_KEY env var in spawn (not CLI arg) to avoid exposure in ps aux

    // Image models: skip all text-specific flags (parsers, batching, cache, etc.)
    // The Python server auto-detects image vs text from the model directory
    if (isImage) {
      // mlxstudio#82: for Server-tab image launches, config.mfluxClass /
      // config.servedModelName are often both empty (the Server-tab session
      // form doesn't require them). Fall back to fuzzy-matching the model
      // path's directory basename against IMAGE_MODELS so we emit the
      // right --mflux-class and --served-model-name flags automatically.
      // Without this, the engine sees just `/Volumes/.../FLUX.2-klein-9B`
      // and fails class resolution on startup — Mark's exact log path.
      let effectiveMfluxClass = config.mfluxClass
      let effectiveServedName = config.servedModelName
      if ((!effectiveMfluxClass || !effectiveServedName) && config.modelPath) {
        const dirBase = basename(config.modelPath.replace(/\/+$/, ''))
        const resolved = resolveImageModelFromDirectoryName(dirBase)
        if (resolved) {
          if (!effectiveMfluxClass) effectiveMfluxClass = resolved.mfluxClass
          if (!effectiveServedName) effectiveServedName = resolved.mfluxName
        }
      }
      // Image-specific settings (explicit flags, not via additionalArgs)
      if (config.imageMode === 'edit') args.push('--image-mode', 'edit')
      if (config.imageQuantize && config.imageQuantize > 0) args.push('--image-quantize', config.imageQuantize.toString())
      if (effectiveServedName) args.push('--served-model-name', effectiveServedName)
      if (effectiveMfluxClass) args.push('--mflux-class', effectiveMfluxClass)
      // Logging + CORS still apply to image servers
      if (config.logLevel && config.logLevel !== 'INFO') args.push('--log-level', config.logLevel)
      if (config.corsOrigins && config.corsOrigins !== '*') args.push('--allowed-origins', config.corsOrigins)
      // Strip image-specific flags from additionalArgs to prevent duplication
      // (stale additionalArgs may survive config merge from a previous session)
      if (config.additionalArgs?.trim()) {
        const filtered = filterAdditionalArgs(config.additionalArgs, IMAGE_ADDITIONAL_ARG_BLOCKLIST)
        if (filtered.length) args.push(...filtered)
      }
      return args
    }

    // === Text model flags below ===

    // Auto-detect tool/reasoning/cache behavior from config.json. This must
    // happen before concurrency flags because DSV4's custom generator is
    // single-batch even though the generic session profile defaults higher.
    const detectedFamily = normalizeDetectedFamilyName(detected.family)
    const dsv4Active = detectedFamily === 'deepseek-v4'

    // Concurrent processing
    // When value is 0 ("No limit" in UI), omit the flag so backend uses its default.
    // When value > 0, pass it explicitly to override the backend default.
    const effectiveMaxNumSeqs = dsv4Active ? 1 : config.maxNumSeqs
    if (dsv4Active && config.maxNumSeqs && config.maxNumSeqs !== 1) {
      console.log(`[SESSION] DSV4-Flash detected: overriding maxNumSeqs ${config.maxNumSeqs} -> 1 (DSV4BatchGenerator is single-batch only)`)
    }
    if (effectiveMaxNumSeqs && effectiveMaxNumSeqs > 0) {
      args.push('--max-num-seqs', effectiveMaxNumSeqs.toString())
    }
    if (!dsv4Active && config.prefillBatchSize && config.prefillBatchSize > 0) {
      args.push('--prefill-batch-size', config.prefillBatchSize.toString())
    }
    if (!dsv4Active && config.prefillStepSize && config.prefillStepSize > 0) {
      args.push('--prefill-step-size', config.prefillStepSize.toString())
    }
    if (!dsv4Active && config.completionBatchSize && config.completionBatchSize > 0) {
      args.push('--completion-batch-size', config.completionBatchSize.toString())
    }

    // VLM detection: tri-state — undefined=auto, true=force on, false=force off.
    // Only respect explicit user choice (true/false); undefined defers to auto-detect.
    // Smelt mutual exclusion: smelt's partial-expert loader doesn't wire the
    // vision tower, so image input on a smelt-loaded VLM produces garbage logits.
    // Suppress --is-mllm when smelt is active — the CLI also guards this, but
    // doing it here prevents misleading "Force MLLM mode enabled" log lines and
    // avoids the edge case where a saved session has isMultimodal=true from
    // before smelt was turned on.
    const effectiveSmelt = !!(config as any).smelt && !dsv4Active
    const isVLM = dsv4Active || effectiveSmelt || detected.forceTextOnly ? false
      : detected.isMultimodal ? true
        : config.isMultimodal === true ? true
          : config.isMultimodal === false ? false
            : false
    if (isVLM) args.push('--is-mllm')

    const cacheStackActive = dsv4Active ? true : config.continuousBatching !== false
    if (cacheStackActive) {
      args.push('--continuous-batching')
    } else {
      args.push('--no-continuous-batching')
    }

    // Parser resolution: User explicit choice -> Detected config -> Fallback logic
    // Empty string "" = user explicitly chose "None" (disabled) — always respected.
    const userToolParser = config.toolCallParser
    const effectiveToolParser = userToolParser === ''
      ? undefined                     // User explicitly chose "None"
      : canonicalizeToolParserId(userToolParser && userToolParser !== 'auto' ? userToolParser
        : detected.toolParser)       // Fallback to detection if auto or missing

    const effectiveAutoTool = config.enableAutoToolChoice ?? detected.enableAutoToolChoice

    const userReasoningParser = config.reasoningParser
    const effectiveReasoningParser = userReasoningParser === ''
      ? undefined                     // User explicitly chose "None"
      : (userReasoningParser && userReasoningParser !== 'auto' ? userReasoningParser
        : detected.reasoningParser)  // Fallback to detection if auto or missing

    // Pass resolved parsers directly to the CLI so backend doesn't guess.
    // When a tool parser is set, --enable-auto-tool-choice is required by the engine
    // (cli.py gates on both flags). Enable it unless user explicitly disabled auto-tool-choice.
    if (effectiveToolParser) {
      args.push('--tool-call-parser', effectiveToolParser)
      // Ensure --enable-auto-tool-choice is set when a parser is present
      if (effectiveAutoTool || config.enableAutoToolChoice === undefined) {
        args.push('--enable-auto-tool-choice')
      }
    } else if (effectiveAutoTool) {
      args.push('--enable-auto-tool-choice')
    }
    if (effectiveReasoningParser) {
      args.push('--reasoning-parser', effectiveReasoningParser)
    }
    if (detected.defaultEnableThinking === false) {
      args.push('--default-enable-thinking', 'false')
    }
    // Pass custom served model name if configured
    if (config.servedModelName) {
      args.push('--served-model-name', config.servedModelName)
    }
    // Pass custom chat template if configured
    if ((config as any).chatTemplate) {
      args.push('--chat-template', (config as any).chatTemplate)
    }

    console.log(`[SESSION] Model family: ${detected.family} | tool: ${effectiveToolParser || 'none'} (user=${userToolParser}, detected=${detected.toolParser || 'none'}) | reasoning: ${effectiveReasoningParser || 'none'} (user=${userReasoningParser}, detected=${detected.reasoningParser || 'none'}) | autoTool: ${effectiveAutoTool} | VLM: ${isVLM}`)

    // Prefix cache — requires --continuous-batching to take effect in vmlx-engine
    // Tool sessions benefit from prefix reuse, but an explicit user opt-out must
    // stay an opt-out; do not silently re-enable cache because tools are present.
    const zayaCcaActive = isZayaCcaFamily(detectedFamily)
    const hybridCacheActive = cacheTypeRequiresPaged(detected.cacheType)
    const architectureRequiresPagedCache =
      zayaCcaActive ||
      dsv4Active ||
      (hybridCacheActive && detected.usePagedCache === true)
    const cacheLaunchPolicy = resolveCacheLaunchPolicy({
      continuousBatching: cacheStackActive,
      enablePrefixCache: config.enablePrefixCache !== false,
      usePagedCache: config.usePagedCache ?? detected.usePagedCache ?? false,
      enableDiskCache: !!config.enableDiskCache,
      enableBlockDiskCache: !!config.enableBlockDiskCache,
      architectureRequiresPagedCache,
    })
    const prefixCacheOff = cacheLaunchPolicy.prefixCacheOff
    const usePagedCache = cacheLaunchPolicy.effectiveUsePagedCache

    if (prefixCacheOff) {
      args.push('--disable-prefix-cache')
    } else if (!dsv4Active) {
      if (config.noMemoryAwareCache) {
        args.push('--no-memory-aware-cache')
        if (config.prefixCacheSize && config.prefixCacheSize > 0) {
          args.push('--prefix-cache-size', config.prefixCacheSize.toString())
        }
        if (config.prefixCacheMaxBytes && config.prefixCacheMaxBytes > 0) {
          args.push('--prefix-cache-max-bytes', config.prefixCacheMaxBytes.toString())
        }
      } else {
        if (!usePagedCache && config.cacheMemoryMb && config.cacheMemoryMb > 0) {
          args.push('--cache-memory-mb', config.cacheMemoryMb.toString())
        }
        if (!usePagedCache && config.cacheMemoryPercent && config.cacheMemoryPercent > 0) {
          args.push('--cache-memory-percent', (config.cacheMemoryPercent / 100).toString())
        }
        // Cache TTL (time-to-live for cache entries) — only meaningful for memory-aware cache, not paged cache
        if (config.cacheTtlMinutes && config.cacheTtlMinutes > 0 && !usePagedCache) {
          args.push('--cache-ttl-minutes', config.cacheTtlMinutes.toString())
        }
      }
    }

    // Paged cache is a prefix cache backend — works for both LLMs and VLMs
    if (!prefixCacheOff && usePagedCache) {
      args.push('--use-paged-cache')
      const effectivePagedCacheBlockSize = detectedFamily === 'deepseek-v4'
        ? DSV4_PAGED_CACHE_BLOCK_SIZE
        : config.pagedCacheBlockSize
      if (detectedFamily === 'deepseek-v4' && config.pagedCacheBlockSize !== DSV4_PAGED_CACHE_BLOCK_SIZE) {
        console.log(`[SESSION] DSV4-Flash detected: overriding pagedCacheBlockSize ${config.pagedCacheBlockSize} -> ${DSV4_PAGED_CACHE_BLOCK_SIZE} (native SWA+CSA/HCA composite cache)`)
      }
      if (effectivePagedCacheBlockSize && effectivePagedCacheBlockSize > 0) {
        args.push('--paged-cache-block-size', effectivePagedCacheBlockSize.toString())
      }
      if (config.maxCacheBlocks && config.maxCacheBlocks > 0) {
        args.push('--max-cache-blocks', config.maxCacheBlocks.toString())
      }
    }

    // KV cache quantization for stored prefix cache entries
    // TurboQuant handles live generation cache compression (always active).
    // q8/q4 here is ADDITIONAL compression for stored cache entries.
    if (!prefixCacheOff && detectedFamily === 'deepseek-v4' && config.kvCacheQuantization && config.kvCacheQuantization !== 'auto') {
      console.log(`[SESSION] DSV4-Flash detected: ignoring generic kvCacheQuantization=${config.kvCacheQuantization}; native SWA+CSA/HCA cache policy owns compression`)
    }
    if (!prefixCacheOff && detectedFamily !== 'deepseek-v4' && config.kvCacheQuantization && config.kvCacheQuantization !== 'auto') {
      args.push('--kv-cache-quantization', config.kvCacheQuantization)
      if (config.kvCacheQuantization !== 'none' && config.kvCacheGroupSize && config.kvCacheGroupSize !== 64) {
        args.push('--kv-cache-group-size', config.kvCacheGroupSize.toString())
      }
    }

    // Disk cache (L2 persistent cache) — RE-ENABLED in v1.3.15
    // TQ-native serialization stores 3-bit compressed data directly (26x smaller).
    // Metal crash fix: all mx.save_safetensors calls now happen on main thread;
    // background writer only does atomic rename + SQLite update.
    if (cacheLaunchPolicy.enableLegacyDiskCache) {
      args.push('--enable-disk-cache')
      if (config.diskCacheDir) {
        args.push('--disk-cache-dir', config.diskCacheDir)
      }
      if (config.diskCacheMaxGb != null && config.diskCacheMaxGb >= 0) {
        args.push('--disk-cache-max-gb', config.diskCacheMaxGb.toString())
      }
    }

    // Block-level disk cache (L2 for paged cache blocks) — RE-ENABLED in v1.3.15
    // All serialization happens on main thread (same Metal safety as above).
    // Background writer only does atomic rename + SQLite index update.
    if (cacheLaunchPolicy.enableBlockDiskCache) {
      args.push('--enable-block-disk-cache')
      if (config.blockDiskCacheDir) {
        args.push('--block-disk-cache-dir', config.blockDiskCacheDir)
      }
      if (config.blockDiskCacheMaxGb != null && config.blockDiskCacheMaxGb >= 0) {
        args.push('--block-disk-cache-max-gb', config.blockDiskCacheMaxGb.toString())
      }
    }

    // Performance
    if (config.streamInterval && config.streamInterval > 0) {
      args.push('--stream-interval', config.streamInterval.toString())
    }
    // maxTokens: 0/unset = no session-level output override. Let the server
    // resolve explicit request > bundle max_new_tokens > engine fallback.
    if (config.maxTokens && config.maxTokens > 0) {
      args.push('--max-tokens', config.maxTokens.toString())
    }
    if (config.maxContextLength && config.maxContextLength > 0) {
      args.push('--max-prompt-tokens', config.maxContextLength.toString())
    }
    // Tool integration (parsers and --enable-auto-tool-choice already pushed above)
    if (config.mcpConfig) args.push('--mcp-config', config.mcpConfig)
    args.push(...buildMcpPolicyArgs(config))

    const requestedDistributed = !!(config as any).distributedEnabled
    const requestedFlashMoe = !!(config as any).flashMoe
    const turboQuantActive = !!(detected as any).isTurboQuant
    const effectiveDistributed = requestedDistributed && !dsv4Active
    const effectiveFlashMoe = requestedFlashMoe && !effectiveDistributed && !dsv4Active
    const effectiveEnableJit = !!config.enableJit && !isVLM && !effectiveFlashMoe && !effectiveDistributed && !dsv4Active && !zayaCcaActive && !turboQuantActive && !hybridCacheActive
    if (dsv4Active && ((config as any).smelt || requestedFlashMoe || requestedDistributed || config.speculativeModel)) {
      console.warn('[SESSION] DSV4-Flash detected: ignoring stale Smelt/Flash MoE/distributed/speculative flags; native DSV4 cache and expert hydration own this runtime')
    }
    if (requestedFlashMoe && !effectiveFlashMoe) {
      console.warn(`[SESSION] Ignoring stale Flash MoE flag because ${dsv4Active ? 'DSV4-Flash is active' : 'distributed mode is active'}`)
    }
    if (config.enableJit && !effectiveEnableJit) {
      const reason = dsv4Active
        ? 'DeepSeek-V4 uses native SWA+CSA/HCA composite cache'
        : zayaCcaActive
        ? 'ZAYA typed CCA cache is path-dependent and benchmarks faster on the uncompiled scheduler path'
        : isVLM
        ? 'multimodal/VLM models use the mlx-vlm streaming path, which is not mx.compile safe'
        : turboQuantActive
        ? 'TurboQuantKVCache uses custom cache objects that mx.compile cannot trace'
        : hybridCacheActive
        ? 'hybrid SSM/Mamba cache uses path-dependent Python cache objects that mx.compile cannot trace'
        : 'Flash MoE or distributed mode is active'
      console.warn(`[SESSION] Ignoring stale JIT flag because ${reason}`)
    }

    // Smelt mode (partial expert loading)
    if (effectiveSmelt) {
      args.push('--smelt')
      const pct = (config as any).smeltExperts ?? 50
      if (pct !== 50) {
        args.push('--smelt-experts', pct.toString())
      }
    }

    // Flash MoE (SSD expert streaming) — mutually exclusive with smelt/distributed/JIT.
    // Always pass the tunable values when Flash MoE is on so CLI reflects UI exactly
    // (no stale equality-with-default guard that drifts when DEFAULT_CONFIG changes).
    if (effectiveFlashMoe) {
      args.push('--flash-moe')
      const slotBank = (config as any).flashMoeSlotBank
      if (typeof slotBank === 'number' && slotBank > 0) {
        args.push('--flash-moe-slot-bank', slotBank.toString())
      }
      const prefetch = (config as any).flashMoePrefetch
      if (prefetch && prefetch !== 'none') {
        args.push('--flash-moe-prefetch', prefetch)
      }
      const ioSplit = (config as any).flashMoeIoSplit
      if (typeof ioSplit === 'number' && ioSplit > 0) {
        args.push('--flash-moe-io-split', ioSplit.toString())
      }
    }

    // Distributed compute
    if (effectiveDistributed) {
      args.push('--distributed')
      const mode = (config as any).distributedMode || 'pipeline'
      if (mode !== 'pipeline') {
        args.push('--distributed-mode', mode)
      }
      // Cluster secret passed via env var in _startSessionInner (same as API key)
    }

    // Speculative decoding
    const externalSpeculativeModel = config.speculativeModel || ''
    const compatibleExternalSpeculative = !dsv4Active && !isVLM && !cacheStackActive && !!externalSpeculativeModel
    if (externalSpeculativeModel && !compatibleExternalSpeculative) {
      const reason = dsv4Active
        ? 'DSV4-Flash has a native composite-cache runtime'
        : isVLM
        ? 'multimodal/VLM generation has no external draft verifier path'
        : cacheStackActive
        ? 'continuous batching is active'
        : 'this runtime does not support external draft decoding'
      console.warn(`[SESSION] Ignoring stale speculative model because ${reason}`)
    }
    if (compatibleExternalSpeculative) {
      args.push('--speculative-model', externalSpeculativeModel)
      if (config.numDraftTokens && config.numDraftTokens !== 3) {
        args.push('--num-draft-tokens', config.numDraftTokens.toString())
      }
    }

    // Native in-model MTP. This is separate from external speculative decoding:
    // Qwen3.6 preserved-MTP bundles carry their own draft head and the current
    // verified path is deterministic. The default app mode therefore launches
    // native-MTP bundles with the measured model-local depth when present
    // (D3 fallback) plus deterministic startup sampling so normal app/API
    // requests actually reach the native MTP runtime instead of silently taking
    // autoregressive decode through generation_config temperature=1.0.
    const nativeMtp = (detected as any).nativeMtp
    if (!dsv4Active && nativeMtp?.supported) {
      const mode = (config as any).nativeMtpMode || 'deterministic'
      if (mode === 'off') {
        args.push('--disable-native-mtp')
      } else {
        const configuredDepth = (config as any).nativeMtpDepthOverride === true
          ? (config as any).nativeMtpDepth
          : nativeMtp.depth
        const depth = Math.max(1, Math.min(3, Math.round(Number(configuredDepth || nativeMtp.depth || 3))))
        args.push('--native-mtp-depth', depth.toString())
        args.push('--native-mtp-sampling-policy', mode === 'deterministic' ? 'deterministic-defaults' : 'compatible-only')
      }
    }

    // Generation defaults are intentionally not passed as --default-* from the
    // panel. The engine resolves request > explicit API/chat value > bundle
    // jang_config/generation_config > family fallback in vmlx_engine.server.

    // Embedding model
    if (config.embeddingModel) {
      args.push('--embedding-model', config.embeddingModel)
    }

    // Do not pass server-level enable_thinking defaults from the panel. The
    // engine resolves model defaults, and chat/API requests carry explicit
    // enable_thinking per request.

    // JIT compilation
    if (effectiveEnableJit) args.push('--enable-jit')

    // Nemotron-Omni multimodal backend selector. Default stage1 (correct).
    // stage2 = native MLX RADIO + Parakeet, ~15-21x faster encoders + 82
    // tok/s decode on M4 Max — the JANGQ-AI banner numbers. User flips
    // this from the panel's "Omni Backend" select in the Server tab.
    if ((config as any).omniBackend && (config as any).omniBackend !== 'stage1') {
      args.push('--omni-backend', (config as any).omniBackend)
    }

    // Logging
    if (config.logLevel && config.logLevel !== 'INFO') {
      args.push('--log-level', config.logLevel)
    }

    // CORS
    if (config.corsOrigins && config.corsOrigins !== '*') {
      args.push('--allowed-origins', config.corsOrigins)
    }

    // Additional arguments — strip stale image-only flags from old session configs
    if (config.additionalArgs?.trim()) {
      const filtered = filterAdditionalArgs(
        config.additionalArgs,
        dsv4Active ? DSV4_ADDITIONAL_ARG_BLOCKLIST : IMAGE_ADDITIONAL_ARG_BLOCKLIST,
      )
      if (filtered.length) args.push(...filtered)
    }

    return args
  }

  findEnginePath(): EnginePath | null {
    const findProjectVenvEngine = (): EnginePath | null => {
      try {
        const sourceDir = join(__dirname, '..', '..', '..')
        const venvPython = join(sourceDir, '.venv', 'bin', 'python3')
        if (!existsSync(venvPython)) return null

        execFileSync(venvPython, ['-B', '-s', '-c', 'import vmlx_engine'], {
          encoding: 'utf-8',
          timeout: 10000,
          env: {
            ...process.env,
            PYTHONDONTWRITEBYTECODE: '1',
            PYTHONNOUSERSITE: '1',
            PYTHONPATH: '',
          },
        })
        console.log(`[SESSIONS] Using project venv: ${venvPython}`)
        return { type: 'bundled', pythonPath: venvPython }
      } catch (_) {
        return null
      }
    }

    // Bundled Python: use python3 -m vmlx_engine.cli instead of vmlx-engine binary.
    // This avoids shebang path issues in relocatable Python builds.
    //
    // mlxstudio#87 hotfix: previously we verified the bundle by spawning
    // `python3 -s -c "import vmlx_engine"` with a 10 s timeout. On a cold-disk
    // first launch, MLX + mlx_vlm shared libs take >10 s to import, the
    // subprocess times out, we fall through to the system-binary search,
    // find any stale user-installed `vmlx-engine` (old brew pip install, say
    // from months ago), and spawn it. That binary's Python has no
    // `vmlx_engine` / `jang_tools` → user sees "ModuleNotFoundError" and
    // blames the vMLX build.
    //
    // Fix: in a packaged app, bundled Python is authoritative. We verify
    // its presence via a filesystem dist-info read (no subprocess, no timeout),
    // and if it passes, we NEVER fall through to a system binary — a stale
    // user install will never win over a freshly-shipped DMG.
    const bundledPython = getBundledPythonPath()
    if (bundledPython) {
      if (verifyBundledEngineOnFilesystem()) {
        return { type: 'bundled', pythonPath: bundledPython }
      }
      if (electronApp.isPackaged) {
        // Bundled Python exists but dist-info is missing — the DMG is broken
        // OR a prior pip --force-reinstall corrupted the install. Refuse to
        // spawn a system binary that would almost certainly be older and
        // missing features: that path produced the ModuleNotFoundError
        // reports we've seen. Fail fast with a clear message instead.
        console.error(
          '[SESSIONS] Bundled Python present but vmlx_engine dist-info is missing. ' +
          'Reinstall vMLX from the latest DMG — spawning a system binary would ship ' +
          'outdated code.'
        )
        return null
      }
      console.log('[SESSIONS] Bundled Python missing vmlx_engine dist-info; trying system (dev mode)')
    }

    // Development builds must exercise the source tree they were launched
    // from. Prefer the project venv before any globally installed vmlx-engine,
    // otherwise UI smoke tests can silently run an old user binary.
    if (!electronApp.isPackaged) {
      const projectVenvEngine = findProjectVenvEngine()
      if (projectVenvEngine) return projectVenvEngine
    }

    // System binary search
    const home = homedir()
    const locations = [
      join(home, '.local', 'bin', 'vmlx-engine'),     // uv tool / pip --user
      '/opt/homebrew/bin/vmlx-engine',                  // Homebrew (Apple Silicon)
      '/usr/local/bin/vmlx-engine',                     // Homebrew (Intel) / system pip
      '/usr/bin/vmlx-engine',                           // System pip
      join(home, 'miniforge3', 'bin', 'vmlx-engine'),  // Miniforge
      join(home, 'anaconda3', 'bin', 'vmlx-engine'),   // Anaconda
      join(home, 'miniconda3', 'bin', 'vmlx-engine'),  // Miniconda
    ]

    // Scan pyenv versions (common on macOS)
    const pyenvRoot = join(home, '.pyenv', 'versions')
    try {
      if (existsSync(pyenvRoot)) {
        for (const ver of readdirSync(pyenvRoot)) {
          locations.push(join(pyenvRoot, ver, 'bin', 'vmlx-engine'))
        }
      }
    } catch (_) { }

    for (const loc of locations) {
      if (existsSync(loc)) return { type: 'system', binaryPath: loc }
    }

    // Fallback: check PATH via login shell (picks up pyenv, nvm, etc.)
    for (const shell of ['/bin/zsh', '/bin/bash']) {
      try {
        const result = execSync(
          `${shell} -lc "which vmlx-engine"`,
          { encoding: 'utf-8', timeout: 5000 }
        ).trim()
        if (result && existsSync(result)) return { type: 'system', binaryPath: result }
      } catch (_) { }
    }

    // Last resort: plain which
    try {
      const result = execSync('which vmlx-engine', { encoding: 'utf-8', timeout: 3000 }).trim()
      if (result && existsSync(result)) return { type: 'system', binaryPath: result }
    } catch (_) { }

    return null
  }

  private async findAvailablePort(): Promise<number> {
    const sessions = db.getSessions()
    // Check ALL session ports (DB has UNIQUE constraint on port column)
    const usedPorts = new Set(sessions.map(s => s.port))
    // Also exclude the API gateway port to prevent overlap crashes (#44)
    const gwPort = parseInt(db.getSetting('gateway_port') || '8080', 10)
    if (gwPort) usedPorts.add(gwPort)
    let port = 8000
    while (usedPorts.has(port) || !(await this.isPortFree(port))) {
      port++
      if (port > 65535) throw new Error('No available ports')
    }
    return port
  }

  private isPortFree(port: number): Promise<boolean> {
    return new Promise(resolve => {
      const server = createServer()
      server.once('error', () => resolve(false))
      server.once('listening', () => {
        server.close(() => setTimeout(() => resolve(true), 10))
      })
      server.listen(port, '127.0.0.1')
    })
  }

  /** Check if a session's process is still alive (not zombie) via PID probe. */
  private isProcessAlive(sessionId: string, dbPid?: number): boolean {
    // Check managed process first (spawned or adopted)
    const managed = this.processes.get(sessionId)
    const pid = managed?.adoptedPid ?? managed?.process?.pid ?? dbPid
    if (!pid) return false
    try {
      process.kill(pid, 0) // Signal 0: doesn't kill, just checks existence
    } catch (_) {
      return false
    }
    // M7: kill(pid, 0) succeeds for zombies. Check process state to filter them out.
    try {
      const state = execFileSync('ps', ['-o', 'state=', '-p', String(pid)],
        { timeout: 1000 }).toString().trim()
      if (state.startsWith('Z')) return false // Zombie process
    } catch (_) {
      // ps failed — process may have exited between checks
      return false
    }
    return true
  }

  private killPid(pid: number, signal: NodeJS.Signals = 'SIGTERM'): void {
    // Try process group kill first (negative PID kills entire group).
    // This ensures MCP subprocesses and uvicorn workers are also killed.
    // Falls back to single-PID kill if group kill fails (e.g., not a group leader).
    try { process.kill(-pid, signal) } catch (_) {
      try { process.kill(pid, signal) } catch (_) { }
    }
  }

  private async killByPort(port: number): Promise<void> {
    try {
      const output = execSync(`lsof -ti tcp:${port}`, { encoding: 'utf-8', timeout: 5000 }).trim()
      if (output) {
        const pids = output.split('\n').map(s => parseInt(s)).filter(n => !isNaN(n))
        for (const pid of pids) this.killPid(pid)
        await new Promise(r => setTimeout(r, 1500))
        // Escalate to SIGKILL if processes still hold the port
        try {
          const check = execSync(`lsof -ti tcp:${port}`, { encoding: 'utf-8', timeout: 3000 }).trim()
          if (check) {
            for (const pidStr of check.split('\n')) this.killPid(parseInt(pidStr), 'SIGKILL')
            await new Promise(r => setTimeout(r, 500))
          }
        } catch (_) { /* port freed */ }
      }
    } catch (_) { }
  }

  private async killChildProcess(proc: ChildProcess): Promise<void> {
    const pid = proc.pid
    return new Promise((resolve) => {
      // Escalate to SIGKILL after 10s if SIGTERM doesn't work
      const killTimeout = setTimeout(() => {
        // Kill entire process group (detached spawn)
        if (pid) { try { process.kill(-pid, 'SIGKILL') } catch (_) { } }
        try { proc.kill('SIGKILL') } catch (_) { }
      }, 10000)

      // B4: Final safety — resolve after 15s even if process never exits
      const hardTimeout = setTimeout(() => {
        clearTimeout(killTimeout)
        resolve()
      }, 15000)

      proc.once('exit', () => {
        clearTimeout(killTimeout)
        clearTimeout(hardTimeout)
        resolve()
      })

      // Send SIGTERM to process group first, then to the process directly
      if (pid) { try { process.kill(-pid, 'SIGTERM') } catch (_) { } }
      try { proc.kill('SIGTERM') } catch (_) {
        clearTimeout(killTimeout)
        clearTimeout(hardTimeout)
        resolve()
      }
    })
  }

  private async waitForReady(host: string, port: number, maxWait = 120000, sessionId?: string): Promise<void> {
    const startTime = Date.now()
    const healthUrl = `http://${connectHost(host)}:${port}/health`

    while (Date.now() - startTime < maxWait) {
      // Abort early if the process exited while we were waiting
      if (sessionId) {
        const managed = this.processes.get(sessionId)
        if (managed && !managed.process && !managed.adoptedPid) {
          // Process exited — include the reason if available
          let reason: string
          if (managed.exitSignal === 'SIGKILL') {
            reason = 'Process was killed (SIGKILL) — likely out of memory. Try a smaller/more quantized model, reduce cache size, or close other apps.'
          } else {
            reason = managed.lastStderr || `exit code ${managed.exitCode ?? 'unknown'}`
          }
          throw new Error(`Process exited before becoming ready: ${reason}`)
        }
        if (!managed) {
          throw new Error('Process exited before becoming ready')
        }
      }

      try {
        const response = await fetch(healthUrl, { signal: AbortSignal.timeout(1000) })
        if (response.ok) {
          // For BatchedEngine, the server starts (Uvicorn) BEFORE the model loads
          // in lifespan(). Just checking response.ok is not enough — we need to
          // verify the model is actually loaded by checking the response body.
          try {
            const data = await response.json() as { status?: string }
            if (data.status === 'healthy') return
            // Server is up but model still loading — keep polling
          } catch {
            // JSON parse failed — server is up but not ready, keep polling
          }
        }
      } catch (_) { }
      await new Promise(resolve => setTimeout(resolve, 500))
    }

    throw new Error('Server failed to start within timeout period')
  }
}

export const sessionManager = new SessionManager()
