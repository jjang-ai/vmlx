import { adoptNativeMtpConfig } from '../shared/nativeMtpAdoption'
import { markImageGenerationServerStopping, recordImageGenerationLog, requestImageGenerationServerStop } from './ipc/imageGenerationState'
import { GATEWAY_SINGLE_MODEL_MODE_KEY, isGatewaySettingEnabled } from '../shared/gatewaySettingsKeys'
import {
  healthFailureToleranceCount,
  shouldContinueStartupWait,
} from '../shared/enginePatienceWindows'
import { spawn, ChildProcess, execSync, execFileSync } from 'child_process'
import { lookup } from 'dns'
import { clipboard, dialog, powerSaveBlocker } from 'electron'
import { EventEmitter } from 'events'
import { existsSync, readdirSync, readFileSync } from 'fs'
import { createServer } from 'net'
import { homedir, totalmem, freemem } from 'os'
import { join, basename, dirname } from 'path'
import { v4 as uuidv4 } from 'uuid'
import { db, Session } from './database'
import { resolveImageModelFromDirectoryName } from '../shared/imageModels'
import { dsv4EnvFromConfig, resolveEffectiveModelFamily } from '../shared/dsv4Env'
import {
  DEFAULT_BLOCK_DISK_CACHE_PERCENT,
  LEGACY_BLOCK_DISK_CACHE_MAX_GB,
} from '../shared/cacheDefaults'
import { buildCacheLaunchArgs } from '../shared/cacheLaunchArgs'
import {
  applyLagunaJitDefaultEnvironment,
  DISABLE_JANG_AFFINE_JIT_DEFAULT_ENV,
  isLagunaMixedSwaTurboQuantEffective,
} from '../shared/lagunaCachePolicy'
import { buildMcpPolicyArgs } from '../shared/mcpPolicy'
import {
  canonicalizeToolParserId,
  resolveEffectiveToolParser,
} from '../shared/toolParserAliases'
import { buildToolLaunchArgs } from '../shared/toolLaunchArgs'
import { resolveEffectiveReasoningParser } from '../shared/reasoningParserAliases'
import {
  applyBundleGenerationDefaultsToSessionConfig,
  hasDeclaredBundleSamplingDefaults,
  resolveBundleGenerationDefaults,
} from '../shared/sessionGenerationDefaults'
import {
  GENERATION_STARTUP_DEFAULTS_VERSION,
  LEGACY_GENERIC_MAX_OUTPUT_TOKENS,
  MODEL_PARSER_DEFAULTS_VERSION,
  migrateModelParserDefaults,
} from '../shared/sessionConfigMigrations'
import { appendMetalWiredLimitGuidance, classifyLargeModelMemoryPreflight, classifyWiredLimitPreflight } from '../shared/metalWiredLimit'
import { sessionMatchesModelPath } from '../shared/sessionUtils'
import { normalizeHfTokenSetting } from '../shared/hfSettings'
import { shouldUseProofOwnedEngineLifecycle } from '../shared/userDataOverride'
import {
  classifySessionModelPaths,
  type SessionModelPathClassification,
  validateModelBundleDirectory,
} from './session-model-path'
import {
  estimateModelFileBytes,
  estimateModelLaunchAdmissionBytes,
  estimateModelLaunchResidentBytes,
  formatGb,
  launchResidentProfileForModel,
  estimateMacReclaimableMemoryBytes,
  effectiveLaunchAvailableBytes,
  unsafeModelLaunchReason,
  modelLaunchReserveWarning,
} from './modelLaunchMemory'
import {
  LifecycleSnapshot,
  lifecycleDisplay,
  lifecyclePhaseLabel,
  parseLifecycleSnapshot,
} from './lifecycleProgress'
import {
  BACKEND_STDERR_DISCONNECT_NORMALIZED_LINE,
  normalizeBackendStderrChunk,
} from './backend-stderr'
import { validateJangBundleMetadataForLaunch } from './model-bundle-validation'
import { runModelBundleIntegrityPreflight } from './model-bundle-integrity'
import { createBundleRepairProgressReporter } from './bundle-repair-progress'
import { sameLocalBundlePath } from './local-bundle-identity'

export type { ServerConfig, DetectedProcess } from './server'
import type { ServerConfig, DetectedProcess } from './server'
import { detectModelConfigFromDir } from './model-config-registry'
import {
  ENGINE_ENTRY_POINT_NAMES,
  ENGINE_SEARCH_DIRS,
  getBundledPythonPath,
  getDevelopmentProjectVenv,
  getDevelopmentSourceRoot,
  verifyBundledEngineOnFilesystem,
} from './engine-manager'
import { app as electronApp } from 'electron'
import { computeEffectiveJit } from '../shared/jitPolicy'
import {
  GENERIC_DEFAULT_TIMEOUT_SECONDS,
  SLOW_FAMILY_TIMEOUTS,
  resolveSlowFamilyTimeoutSeconds,
} from '../shared/slowFamilyTimeouts'
import {
  isZayaCcaFamily,
  normalizeDetectedFamilyName,
  usesExactTypedPromptDiskCache,
} from '../shared/detectedFamilyNames'
import { cacheTypeRequiresPaged } from '../shared/cacheTypeCapabilities'
import {
  filterAdditionalArgs,
  finitePositiveInteger,
} from '../shared/launchArgValues'
import { buildNativeMtpLaunchArgs } from '../shared/nativeMtpLaunchArgs'
import { planSessionConfigSave } from '../shared/sessionConfigLifecycle'

/** Result of findEnginePath: packaged Python, a source-bound dev venv, or a system binary. */
type EnginePath =
  | { type: 'bundled'; pythonPath: string }
  | { type: 'development'; pythonPath: string; sourceRoot: string }
  | { type: 'system'; binaryPath: string; sourceRoot?: string }

interface ManagedProcess {
  process: ChildProcess | null
  adoptedPid: number | null
  lastStderr?: string  // Last stderr line for error reporting
  backendStderrPending?: string
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
  doSample?: boolean
  defaultTemperature?: number
  defaultTopP?: number
  defaultTopK?: number
  defaultMinP?: number
  defaultRepetitionPenalty?: number
  maxTokens?: number
  samplingDefaultsDeclared?: boolean
  source?: 'generation_config' | 'jang_config'
}



function hasDsv4IdentityHint(modelPath: string): boolean {
  const isDsv4Value = (value: unknown): boolean => {
    const normalized = String(value || '').toLowerCase().replace(/[^a-z0-9]/g, '')
    return normalized.includes('deepseekv4') || normalized.includes('dsv4')
  }
  const visitConfig = (value: any): boolean => {
    if (!value || typeof value !== 'object') return false
    const candidates = [
      value.model_type,
      value._name_or_path,
      value.model_name,
      value?.capabilities?.family,
      value?.text_config?.model_type,
      ...(Array.isArray(value.architectures) ? value.architectures : []),
      ...(Array.isArray(value?.text_config?.architectures)
        ? value.text_config.architectures
        : []),
    ]
    return candidates.some(isDsv4Value)
  }

  for (const filename of ['config.json', 'jang_config.json']) {
    try {
      const path = join(modelPath, filename)
      if (existsSync(path) && visitConfig(JSON.parse(readFileSync(path, 'utf8')))) {
        return true
      }
    } catch {
      // Keep checking the independent file/path identity. A malformed bundle
      // config is exactly when adoption must not depend on the full detector.
    }
  }
  return isDsv4Value(basename(normalizePath(modelPath)))
}

/**
 * Existing DSV4 processes are not safe to adopt without exact provenance.
 *
 * An already-running engine can predate the current native composite-cache
 * contract or can use different cache controls. Its model path and port do not
 * attest either executable provenance or effective cache policy. Until health
 * exposes enough immutable provenance to prove both, DSV4 must be relaunched by
 * this Electron instance. Other families retain the existing adoption behavior.
 */
function canAdoptExistingLocalEngine(modelPath: string): boolean {
  const dsv4IdentityHint = hasDsv4IdentityHint(modelPath)
  try {
    const family = normalizeDetectedFamilyName(detectModelConfigFromDir(modelPath).family)
    return family !== 'deepseek-v4' && !dsv4IdentityHint
  } catch {
    // Detection failures remain adoptable for unrelated families, preserving
    // the established policy. A DSV4 identity from either bundle file or the
    // model directory fails closed because its executable/cache provenance is
    // not attested.
    return !dsv4IdentityHint
  }
}





const DSV4_PAGED_CACHE_BLOCK_SIZE = 256
const DSV4_MAX_CACHE_BLOCKS = 4097 // 4096 data blocks + reserved null block = 1,048,576 tokens

// --max-cache-blocks counts BLOCKS, so one number means very different token
// capacity at different block sizes. DSV4 was given an explicit 1M-token index
// above; every other family was left at a flat 1000, which at the generic
// 64-token block indexes only 63,936 tokens — far below the context window of
// models like Gemma 4. Measured on the box: a 77k-token Gemma prompt reported 0
// cached tokens on an EXACT repeat and ran slower than a cold prefill (82.5s vs
// 55.7s), while the same probe at 28k reused 28,199 tokens and cut TTFT from
// 9.10s to 0.98s. Size the generic default by target TOKENS instead.
// This bounds the index only; resident RAM stays governed by
// --cache-memory-mb/--cache-memory-percent.
const GENERIC_INDEX_TARGET_TOKENS = 262144

function indexBlocksForCapacity(
  blockSize: number | undefined,
  targetTokens: number = GENERIC_INDEX_TARGET_TOKENS,
): number {
  const size = Number.isFinite(Number(blockSize)) && Number(blockSize) > 0 ? Math.floor(Number(blockSize)) : 64
  // +1 for the reserved null block, matching DSV4_MAX_CACHE_BLOCKS' accounting.
  return Math.ceil(targetTokens / size) + 1
}
// These four numbers are the shared table's, not this file's. They are named
// here only because the family-startup-defaults writers below PERSIST a value
// into config.timeout (as opposed to resolving one), and a named constant reads
// better at those call sites than a table lookup. Sourcing them from the table
// keeps "what gets written" and "what gets resolved" from ever disagreeing.
const DSV4_DEFAULT_TIMEOUT_SECONDS = SLOW_FAMILY_TIMEOUTS['deepseek-v4']
const MINIMAX_M3_DEFAULT_TIMEOUT_SECONDS = SLOW_FAMILY_TIMEOUTS.minimax_m3
// openPangu-2.0-Flash: 92B MoE, typically 2-3 bit JANG — slow prefill/decode.
const OPENPANGU_V2_DEFAULT_TIMEOUT_SECONDS = SLOW_FAMILY_TIMEOUTS.openpangu_v2

function effectiveSessionTimeoutSeconds(config: Partial<ServerConfig>, family?: string): number {
  const configured = config.timeout
  if (configured != null && configured <= 0) return 86400
  // Shared table — see panel/src/shared/slowFamilyTimeouts.ts for why this rule
  // may only exist once. Three hand-written pre-checks (deepseek-v4,
  // minimax_m3, openpangu_v2) used to sit above this call. They returned the
  // same numbers the table does, so they were not a live bug — but they shadowed
  // it: raising any of those three in the shared table would have moved the
  // Settings CLI preview and left the launcher on the old value, re-creating
  // the exact preview-lies-about-launch defect the table was introduced to end.
  return resolveSlowFamilyTimeoutSeconds(configured, normalizeDetectedFamilyName(family))
}

/**
 * The timeout the ENGINE was actually given for this session.
 *
 * `config.timeout` is NOT that number. The slow families (hybrid SSM, DSV4,
 * MiniMax-M3, openPangu) keep the generic 300 in their stored config and get
 * lifted to 900 at launch by the table above, so any consumer that reads
 * `config.timeout` directly is reasoning about a value the engine never saw.
 *
 * That is exactly how the health poll came to kill sessions the engine was
 * still working on: it scaled its patience from the stored 300 while the
 * engine had been handed 900. Resolving the family requires the model
 * directory, which is why this lives next to the launcher rather than in the
 * shared module — but it is one function, and both the `--timeout` argument
 * and the health poll now call it.
 */
function resolvedEngineTimeoutSeconds(config: Partial<ServerConfig>): number {
  let detectedFamily: string | undefined
  try {
    detectedFamily = normalizeDetectedFamilyName(
      detectModelConfigFromDir(config.modelPath ?? '').family,
    )
  } catch (_) {
    detectedFamily = undefined
  }
  const effectiveFamily = normalizeDetectedFamilyName(
    resolveEffectiveModelFamily(config.modelFamily, detectedFamily),
  )
  return effectiveSessionTimeoutSeconds(config, effectiveFamily)
}




function setConfigValue(config: Record<string, any>, key: string, value: unknown): boolean {
  if (config[key] === value) return false
  config[key] = value
  return true
}

function applyFamilyStartupDefaults(config: Partial<ServerConfig>, modelPath?: string): boolean {
  if (!modelPath) return false
  try {
    const detected = detectModelConfigFromDir(modelPath)
    const detectedFamily = normalizeDetectedFamilyName(detected.family)
    const effectiveFamily = normalizeDetectedFamilyName(
      resolveEffectiveModelFamily(config.modelFamily, detectedFamily),
    )
    let changed = migrateModelParserDefaults(
      config as Record<string, any>,
      detectedFamily,
      detected.reasoningParser,
    )
    // Native MTP default: FIXED depth 3 for the Qwen3.8 MTP families
    // (Flash-Next qwen4-exp — every JANG tier and CRACK variant — and the
    // Qwen3.8-27B qwen3.5 D-series). Adaptive proved a wrong default for
    // fresh sessions (Eric, 2026-09-05: the session started adaptive when
    // it must start fixed D3). Fill ONLY missing values: an explicit user
    // choice (including turning the override off) always survives.
    if (effectiveFamily === 'qwen4-exp' || effectiveFamily === 'qwen3.5') {
      if ((config as any).nativeMtpDepthOverride === undefined) {
        ;(config as any).nativeMtpDepthOverride = true
        changed = true
      }
      if ((config as any).nativeMtpDepth === undefined) {
        ;(config as any).nativeMtpDepth = 3
        changed = true
      }
    }
    if (
      effectiveFamily === 'deepseek-v4' &&
      (config.timeout == null || config.timeout === GENERIC_DEFAULT_TIMEOUT_SECONDS)
    ) {
      config.timeout = DSV4_DEFAULT_TIMEOUT_SECONDS
      changed = true
    }
    if (effectiveFamily === 'deepseek-v4') {
      // DSV4 uses the normal product cache controls with a typed
      // SWA+CSA/HCA block record. Fill only missing user toggles here: explicit
      // post-migration Prefix/Paged/Block-Disk choices must survive restart.
      if (config.enablePrefixCache === undefined) {
        config.enablePrefixCache = true
        changed = true
      }
      if (config.usePagedCache === undefined) {
        config.usePagedCache = false
        changed = true
      }
      if (config.enableBlockDiskCache === undefined) {
        config.enableBlockDiskCache = true
        changed = true
      }
      // Legacy whole-prompt L2 and generic KV q4/q8 cannot represent DSV4's
      // native composite state. Block Disk L2 and the bundle pool codec own it.
      if (config.enableDiskCache !== false) {
        config.enableDiskCache = false
        changed = true
      }
      if (config.kvCacheQuantization !== 'auto') {
        config.kvCacheQuantization = 'auto'
        changed = true
      }
      if (config.pagedCacheBlockSize !== DSV4_PAGED_CACHE_BLOCK_SIZE) {
        config.pagedCacheBlockSize = DSV4_PAGED_CACHE_BLOCK_SIZE
        changed = true
      }
      if (config.maxCacheBlocks === undefined) {
        config.maxCacheBlocks = DSV4_MAX_CACHE_BLOCKS
        changed = true
      }
      const dsv4PrefixEnabled = config.enablePrefixCache !== false
      if (config.dsv4PrefixCache !== dsv4PrefixEnabled) {
        // Compatibility mirror only; launch uses the normal prefix flag.
        config.dsv4PrefixCache = dsv4PrefixEnabled
        changed = true
      }
      if (typeof detected.dsv4PoolQuantDefault === 'boolean') {
        if (config.dsv4PoolQuant !== detected.dsv4PoolQuantDefault) {
          config.dsv4PoolQuant = detected.dsv4PoolQuantDefault
          changed = true
        }
      } else if (config.dsv4PoolQuant !== undefined) {
        // No bundle stamp: omit the product override so the engine loader
        // derives the internal pool codec from jang_config.json.
        delete config.dsv4PoolQuant
        changed = true
      }
      if (config.dsv4ActivationQat === undefined) {
        // Activation QAT is a user-owned fidelity/overhead choice, not a
        // bundle-derived cache default. Existing sessions migrate to Off.
        config.dsv4ActivationQat = false
        changed = true
      }
    } else if (detectedFamily === 'minimax_m3') {
      if (config.timeout == null || config.timeout === GENERIC_DEFAULT_TIMEOUT_SECONDS) {
        config.timeout = MINIMAX_M3_DEFAULT_TIMEOUT_SECONDS
        changed = true
      }
      // Output-cap migration belongs to applyBundleStartupDefaults on the
      // stored baseline, never this per-launch family-default pass.
      // Defaults are applied only when a control has never been saved. M3's
      // typed MSA block records (K/V/idx_keys/offsets) support SSD-only L2,
      // so an explicit In-Memory Paged Cache=Off must survive restart while
      // Prefix Cache and Block Disk Cache remain enabled.
      if (config.enablePrefixCache === undefined) {
        config.enablePrefixCache = true
        changed = true
      }
      if (config.usePagedCache === undefined) {
        config.usePagedCache = false
        changed = true
      }
      if (config.enableDiskCache !== false) {
        config.enableDiskCache = false
        changed = true
      }
      if (config.enableBlockDiskCache === undefined) {
        config.enableBlockDiskCache = true
        changed = true
      }
      if (config.enableJit !== false) {
        config.enableJit = false
        changed = true
      }
    } else if (detectedFamily === 'glm5-next') {
      // GLM-5.3's KDA recurrent/conv state plus MLA/DSA indexer state is
      // persisted as one exact typed N-1 snapshot. The generic block store
      // cannot represent that boundary. Migrate the old generic default pair
      // while preserving an explicit all-off L2 choice.
      const staleGenericDiskPair =
        config.enableDiskCache !== true && config.enableBlockDiskCache === true
      if (config.enablePrefixCache === undefined) {
        config.enablePrefixCache = true
        changed = true
      }
      if (config.usePagedCache !== false) {
        config.usePagedCache = false
        changed = true
      }
      if (staleGenericDiskPair || config.enableDiskCache === undefined) {
        config.enableDiskCache = true
        changed = true
      }
      if (staleGenericDiskPair || config.enableDiskCache === true) {
        if (config.enableBlockDiskCache !== false) {
          config.enableBlockDiskCache = false
          changed = true
        }
      }
      if (config.noMemoryAwareCache !== false) {
        config.noMemoryAwareCache = false
        changed = true
      }
      if (config.kvCacheQuantization !== 'auto') {
        config.kvCacheQuantization = 'auto'
        changed = true
      }
    } else if (detectedFamily === 'openpangu_v2') {
      // openPangu-2.0-Flash: exact typed N-1 prompt snapshots preserve MLA KV,
      // DSA indexer, rotating-SWA metadata and all causal-conv states. Memory
      // prefix + prompt L2 are safe; generic paged/block reuse and legacy
      // entry-count sharing remain off. JIT stays off for dynamic DSA top-k.
      if (config.timeout == null || config.timeout === GENERIC_DEFAULT_TIMEOUT_SECONDS) {
        config.timeout = OPENPANGU_V2_DEFAULT_TIMEOUT_SECONDS
        changed = true
      }
      // Preserve explicit output caps; shared stored-baseline migration owns
      // historical generic defaults before current user edits are applied.
      if (config.enablePrefixCache !== true) {
        config.enablePrefixCache = true
        changed = true
      }
      if (config.usePagedCache !== false) {
        config.usePagedCache = false
        changed = true
      }
      if (config.enableDiskCache !== true) {
        config.enableDiskCache = true
        changed = true
      }
      if (config.enableBlockDiskCache !== false) {
        config.enableBlockDiskCache = false
        changed = true
      }
      if (config.noMemoryAwareCache !== false) {
        config.noMemoryAwareCache = false
        changed = true
      }
      if (config.kvCacheQuantization !== 'auto') {
        config.kvCacheQuantization = 'auto'
        changed = true
      }
      if (config.enableJit !== false) {
        config.enableJit = false
        changed = true
      }
    }
    return changed
  } catch {
    /* family defaults are best-effort; launch-time buildArgs repeats the guard */
    return false
  }
}


const IMAGE_ADDITIONAL_ARG_BLOCKLIST = new Set([
  '--host',
  '--port',
  '--uds',
  '--api-key',
  '--rate-limit',
  '--timeout',
  '--inference-endpoints',
  '--wake-timeout',
  '--log-level',
  '--allowed-origins',
  '--image-mode',
  '--image-quantize',
  '--served-model-name',
  '--mflux-class',
  '--mcp-config',
  '--mcp-disabled-servers',
  '--mcp-disabled-tools',
  '--mcp-enabled-servers',
  '--mcp-enabled-tools',
])

const TEXT_ADDITIONAL_ARG_BLOCKLIST = new Set([
  ...IMAGE_ADDITIONAL_ARG_BLOCKLIST,
  '--chat-template',
  '--chat-template-kwargs',
  '--continuous-batching',
  '--no-continuous-batching',
  '--enable-prefix-cache',
  '--disable-prefix-cache',
  '--use-paged-cache',
  '--no-paged-cache',
  '--enable-vision-memory-cache',
  '--no-vision-memory-cache',
  '--vision-memory-cache-size',
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
  '--no-jit',
  '--no-memory-aware-cache',
  '--prefix-cache-size',
  '--prefix-cache-max-bytes',
  '--cache-memory-mb',
  '--cache-memory-percent',
  '--cache-ttl-minutes',
  '--enable-disk-cache',
  '--disk-cache-dir',
  '--disk-cache-max-gb',
  '--enable-block-disk-cache',
  '--disable-block-disk-cache',
  '--block-disk-cache-dir',
  '--block-disk-cache-max-gb',
  '--block-disk-cache-max-percent',
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
  '--native-mtp-depth-policy',
  '--native-mtp-sampling-policy',
  '--disable-native-mtp',
  '--enable-pld',
  '--pld-summary-interval',
  '--is-mllm',
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
])

const DSV4_ADDITIONAL_ARG_BLOCKLIST = new Set([
  '--continuous-batching',
  '--no-continuous-batching',
  '--host',
  '--port',
  '--uds',
  '--api-key',
  '--rate-limit',
  '--timeout',
  '--inference-endpoints',
  '--wake-timeout',
  '--log-level',
  '--allowed-origins',
  '--served-model-name',
  '--dsv4-enable-prefix-cache',
  '--enable-prefix-cache',
  '--disable-prefix-cache',
  '--use-paged-cache',
  '--no-paged-cache',
  '--enable-vision-memory-cache',
  '--no-vision-memory-cache',
  '--vision-memory-cache-size',
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
  '--prefill-keep-alloc',
  '--no-state-machine-stops',
  '--ssm-state-cache-size',
  '--ssm-state-cache-mb',
  '--enable-jit',
  '--no-jit',
  '--no-memory-aware-cache',
  '--prefix-cache-size',
  '--prefix-cache-max-bytes',
  '--cache-memory-mb',
  '--cache-memory-percent',
  '--cache-ttl-minutes',
  '--enable-disk-cache',
  '--disk-cache-dir',
  '--disk-cache-max-gb',
  '--enable-block-disk-cache',
  '--disable-block-disk-cache',
  '--block-disk-cache-dir',
  '--block-disk-cache-max-gb',
  '--block-disk-cache-max-percent',
  '--image-mode',
  '--image-quantize',
  '--mflux-class',
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
  '--native-mtp-depth-policy',
  '--native-mtp-sampling-policy',
  '--disable-native-mtp',
  '--enable-pld',
  '--pld-summary-interval',
  '--is-mllm',
  '--model-family',
  '--text-only',
  '--mcp-config',
  '--mcp-disabled-servers',
  '--mcp-disabled-tools',
  '--mcp-enabled-servers',
  '--mcp-enabled-tools',
  '--omni-backend',
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


function readBundleStartupDefaults(modelPath?: string): BundleStartupDefaults {
  if (!modelPath) return {}
  let generationConfig: Record<string, any> | undefined
  let jangConfig: Record<string, any> | undefined
  let modelConfig: Record<string, any> | undefined
  try {
    generationConfig = JSON.parse(readFileSync(join(modelPath, 'generation_config.json'), 'utf8'))
  } catch { /* generation_config.json is optional */ }
  try {
    jangConfig = JSON.parse(readFileSync(join(modelPath, 'jang_config.json'), 'utf8'))
  } catch { /* jang_config.json is optional */ }
  try {
    modelConfig = JSON.parse(readFileSync(join(modelPath, 'config.json'), 'utf8'))
  } catch { /* config.json is optional */ }

  const defaults = resolveBundleGenerationDefaults(generationConfig, jangConfig, modelConfig)
  if (!defaults) return {}
  const mapped = applyBundleGenerationDefaultsToSessionConfig({}, defaults)
  return {
    doSample: mapped.defaultDoSample,
    defaultTemperature: mapped.defaultTemperature,
    defaultTopP: mapped.defaultTopP,
    defaultTopK: mapped.defaultTopK,
    defaultMinP: mapped.defaultMinP,
    defaultRepetitionPenalty: mapped.defaultRepetitionPenalty,
    maxTokens: mapped.defaultMaxNewTokens || undefined,
    samplingDefaultsDeclared: hasDeclaredBundleSamplingDefaults(defaults),
    source: defaults.source,
  }
}

function applyBundleStartupDefaults(
  config: Partial<ServerConfig>,
  modelPath?: string,
  migrateLegacyOutput = true,
): boolean {
  const defs = readBundleStartupDefaults(modelPath)
  const mutable = config as Record<string, any>
  let changed = false

  // Startup generation defaults are model-owned. Keep the saved/display config
  // aligned with bundle sampling metadata and clear old generic startup
  // overrides; buildArgs intentionally does not turn these values into
  // --default-* flags. max_new_tokens is also bundle-owned, but it must not be
  // copied into a hidden session-level --max-tokens default. Users change
  // output length per chat or per API request.
  changed = setConfigValue(mutable, 'defaultTemperature', defs.defaultTemperature ?? 0) || changed
  changed = setConfigValue(mutable, 'defaultTopP', defs.defaultTopP ?? 0) || changed
  changed = setConfigValue(mutable, 'defaultTopK', defs.defaultTopK ?? 0) || changed
  changed = setConfigValue(mutable, 'defaultMinP', defs.defaultMinP ?? 0) || changed
  changed = setConfigValue(mutable, 'defaultRepetitionPenalty', defs.defaultRepetitionPenalty ?? 0) || changed
  changed = setConfigValue(mutable, 'defaultMaxNewTokens', defs.maxTokens ?? 0) || changed
  changed = setConfigValue(mutable, 'defaultSamplingDefaultsDeclared', defs.samplingDefaultsDeclared === true) || changed
  changed = setConfigValue(mutable, 'defaultDoSample', defs.doSample) || changed
  const migrationKey = 'generationStartupDefaultsVersion'
  if (mutable[migrationKey] !== GENERATION_STARTUP_DEFAULTS_VERSION) {
    const oldHiddenMaxTokens =
      defs.maxTokens != null && Number(config.maxTokens) === Number(defs.maxTokens)
    const oldGenericMaxTokens = LEGACY_GENERIC_MAX_OUTPUT_TOKENS.has(Number(config.maxTokens))
    if (migrateLegacyOutput && (oldHiddenMaxTokens || oldGenericMaxTokens)) {
      changed = setConfigValue(mutable, 'maxTokens', 0) || changed
    }
    changed = setConfigValue(mutable, migrationKey, GENERATION_STARTUP_DEFAULTS_VERSION) || changed
  }
  return changed
}

const CACHE_STACK_STARTUP_DEFAULTS_VERSION = 17

function markCacheStackStartupDefaultsCurrent(
  config: Partial<ServerConfig>,
  modelPath?: string,
): boolean {
  if (config.cacheStackStartupDefaultsVersion === CACHE_STACK_STARTUP_DEFAULTS_VERSION) return false
  // Every migration from v12 onward decides what to do by DETECTING the family
  // from the bundle, so stamping the version while the bundle is unreachable
  // consumes the migration without performing it — permanently, because the
  // stamp then makes the session look already-migrated. Models live on an
  // external drive here, so "unreachable at launch" is routine, not exotic.
  // Retry instead: leave the version alone until the path is mounted again.
  // (This covers v12's DSV4 SSD-only case and the v16 paged-parity flip alike.)
  const pendingVersion = Number(config.cacheStackStartupDefaultsVersion || 0)
  if (pendingVersion >= 12 && pendingVersion < CACHE_STACK_STARTUP_DEFAULTS_VERSION) {
    const targetPath = modelPath || config.modelPath
    if (!targetPath) return false
    // The question here is ONLY "is the bundle reachable", so ask that directly.
    // Treating family==='unknown' as the answer conflates an unmounted drive
    // with a bundle that reads fine but is not in the registry — and the second
    // case then never migrates at all, silently keeping whatever defaults it
    // already had. Live example: an LFM2.5-VL HF snapshot (model_type lfm2_vl)
    // detects as 'unknown' and was the ONE session out of 21 that kept the
    // in-RAM paged cache on through the v17 flip.
    //
    // An unrecognised family is a real answer, not a failed detection. The
    // migrations that genuinely need a family are all gated on matching a
    // specific one, so they simply do not fire for it.
    try {
      if (!existsSync(join(targetPath, 'config.json'))) return false
    } catch {
      return false
    }
  }
  config.cacheStackStartupDefaultsVersion = CACHE_STACK_STARTUP_DEFAULTS_VERSION
  return true
}

function liftStaleFlatCacheIndex(
  config: Partial<ServerConfig>,
  modelPath?: string,
): boolean {
  // The flat 1000-block index addresses only 63,936 tokens at the 64-token
  // generic block, silently capping prefix reuse below the model's context
  // window (Gemma 4 at 77k reported ZERO reuse on an exact repeat and ran
  // slower than cold). The v14+ migrations lift it for EXISTING sessions, but
  // the Create Session form still ships 1000 in its DEFAULT_CONFIG and sends it
  // explicitly, and buildArgs then emits --max-cache-blocks 1000, which wins
  // over the engine's own default. A renderer-side default alone cannot be
  // trusted here, so this main-process backstop runs on every session-create
  // path: fresh, merged-over-existing, and adopted.
  //
  // DSV4 keeps its own 256-token block and 4097-block (1M token) contract, and
  // ZAYA deliberately runs a 1000-block index, so neither is lifted. If the
  // family cannot be resolved (bundle unreachable) we leave the value alone
  // rather than guess.
  if (Number(config.maxCacheBlocks) !== 1000) return false
  const target = modelPath || config.modelPath
  if (isZayaCacheStackMigrationTarget(target)) return false
  let family: string | undefined
  try {
    family = normalizeDetectedFamilyName(
      resolveEffectiveModelFamily(
        config.modelFamily,
        normalizeDetectedFamilyName(detectModelConfigFromDir(String(target || '')).family),
      ),
    )
  } catch {
    return false
  }
  if (!family || family === 'unknown' || family === 'deepseek-v4') return false
  config.maxCacheBlocks = indexBlocksForCapacity(config.pagedCacheBlockSize)
  return true
}

function normalizeCacheStackMutualExclusion(config: Partial<ServerConfig>): boolean {
  let changed = false
  // Persist the same hard-off RAM policy the shared launch builder enforces.
  // Leaving a stale true in SQLite made settings state disagree with the
  // disabled checkbox and with the actual --no-paged-cache argv.
  if (config.usePagedCache !== false) {
    config.usePagedCache = false
    changed = true
  }
  // Persist the same mutually-exclusive L2 lane that buildArgs will launch.
  // The legacy full-prompt store remains the exact typed exception lane
  // (notably openPangu). Never retain both disk formats for one session.
  if (config.enableBlockDiskCache === true && config.enableDiskCache === true) {
    config.enableDiskCache = false
    changed = true
  }
  // Production cache reuse is exact and architecture-native for every family.
  // Migrate old q4/q8/none values at every constructor/start/save choke point,
  // including sessions already stamped with the current defaults version.
  // `auto` omits the generic CLI codec flag; the engine keeps the model's own
  // QSA/GDN/SSM/sparse/rotating/composite cache objects unchanged.
  if (config.kvCacheQuantization !== 'auto') {
    config.kvCacheQuantization = 'auto'
    changed = true
  }
  return changed
}

function applyMissingCacheStackStartupDefaults(config: Partial<ServerConfig>, modelPath?: string): boolean {
  const targetPath = modelPath || config.modelPath
  let detectedFamily: string | undefined
  // No per-family paged capability is read here any more: in-RAM paged cache is
  // OFF for every family and SSD block-disk L2 is the only tier, so there is
  // nothing for a registry capability to decide.
  if (targetPath) {
    try {
      const detected = detectModelConfigFromDir(targetPath)
      detectedFamily = normalizeDetectedFamilyName(detected.family)
    } catch {
      /* detection is best-effort here; buildArgs repeats detection at launch */
    }
  }

  const effectiveFamily = normalizeDetectedFamilyName(
    resolveEffectiveModelFamily(config.modelFamily, detectedFamily),
  )
  const dsv4Active = effectiveFamily === 'deepseek-v4'
  const exactTypedPromptDiskCache = usesExactTypedPromptDiskCache(detectedFamily)
  // In-RAM paged cache is OFF for EVERY family, DSV4 included. SSD block-disk
  // L2 is the only cache tier. Seeding a saved `true` here (even though the
  // launch choke point forces --no-paged-cache) would persist a config that
  // disagrees with what actually runs.
  const defaultUsePagedCache = false
  // Every supported prefix-cache lane gets one SSD L2 by default. Exact
  // path-dependent snapshot families use prompt-level typed L2 instead of
  // generic content-addressed blocks.
  const defaultEnableDiskCache = exactTypedPromptDiskCache
  const defaultEnableBlockDiskCache = !exactTypedPromptDiskCache
  const mutable = config as Record<string, any>
  const staleV11Dsv4FailClosedCandidate =
    dsv4Active &&
    Number(config.cacheStackStartupDefaultsVersion || 0) === 11 &&
    config.dsv4PrefixCache === false &&
    config.enablePrefixCache === false &&
    config.usePagedCache === false &&
    config.enableBlockDiskCache === false
  let changed = false

  // Fill only missing values. Explicit user toggles and family-specific overrides
  // must survive, while fresh IPC-created sessions still need a complete config
  // for the Settings UI and live proof harness.
  if (mutable.enablePrefixCache === undefined) changed = setConfigValue(mutable, 'enablePrefixCache', true) || changed
  if (mutable.prefixCacheSize === undefined) changed = setConfigValue(mutable, 'prefixCacheSize', 100) || changed
  if (mutable.prefixCacheMaxBytes === undefined) changed = setConfigValue(mutable, 'prefixCacheMaxBytes', 0) || changed
  if (mutable.cacheMemoryMb === undefined) changed = setConfigValue(mutable, 'cacheMemoryMb', 0) || changed
  if (mutable.cacheMemoryPercent === undefined) changed = setConfigValue(mutable, 'cacheMemoryPercent', 15) || changed
  if (mutable.noMemoryAwareCache === undefined || exactTypedPromptDiskCache) changed = setConfigValue(mutable, 'noMemoryAwareCache', false) || changed
  if (mutable.usePagedCache === undefined || exactTypedPromptDiskCache) changed = setConfigValue(mutable, 'usePagedCache', exactTypedPromptDiskCache ? false : defaultUsePagedCache) || changed
  if (mutable.enableDiskCache === undefined) changed = setConfigValue(mutable, 'enableDiskCache', defaultEnableDiskCache) || changed
  if (mutable.diskCacheMaxGb === undefined) changed = setConfigValue(mutable, 'diskCacheMaxGb', 10) || changed
  if (mutable.pagedCacheBlockSize === undefined) changed = setConfigValue(mutable, 'pagedCacheBlockSize', dsv4Active ? DSV4_PAGED_CACHE_BLOCK_SIZE : 64) || changed
  if (mutable.maxCacheBlocks === undefined) {
    changed = setConfigValue(
      mutable,
      'maxCacheBlocks',
      staleV11Dsv4FailClosedCandidate
        ? 1000
        : dsv4Active
          ? DSV4_MAX_CACHE_BLOCKS
          : indexBlocksForCapacity(mutable.pagedCacheBlockSize),
    ) || changed
  }
  if (mutable.enableBlockDiskCache === undefined) changed = setConfigValue(mutable, 'enableBlockDiskCache', defaultEnableBlockDiskCache) || changed
  // Deliberately NOT seeded. Any number here is emitted and beats the percent;
  // 0 in particular means UNLIMITED to the engine, not "unset". Leaving the
  // field absent is what lets blockDiskCacheMaxPercent actually govern.
  if (mutable.blockDiskCacheMaxPercent === undefined) changed = setConfigValue(mutable, 'blockDiskCacheMaxPercent', DEFAULT_BLOCK_DISK_CACHE_PERCENT) || changed
  if (mutable.kvCacheQuantization === undefined) changed = setConfigValue(mutable, 'kvCacheQuantization', 'auto') || changed

  return changed
}

function numericArgValue(args: string[], flag: string): number | null {
  const idx = args.indexOf(flag)
  if (idx < 0 || idx + 1 >= args.length) return null
  const n = Number(args[idx + 1])
  return Number.isFinite(n) && n > 0 ? Math.floor(n) : null
}

function pagedCacheCapacityLogLine(args: string[]): string | null {
  const pagedActive = args.includes('--use-paged-cache')
  const blockDiskActive = args.includes('--enable-block-disk-cache')
  if (!pagedActive && !blockDiskActive) return null
  const blockSize = numericArgValue(args, '--paged-cache-block-size') ?? 64
  const maxBlocks = numericArgValue(args, '--max-cache-blocks') ?? 1000
  const usableBlocks = Math.max(0, maxBlocks - 1)
  const capacity = blockSize * usableBlocks
  if (!pagedActive) {
    return `Block disk-only index capacity: ${blockSize} tokens/block x ${usableBlocks} usable blocks (${maxBlocks} configured; 1 reserved) = ${capacity} indexed tokens. In-Memory Paged Cache (RAM) is disabled; KV payloads persist on SSD and restore transiently.\n`
  }
  return `In-memory paged-cache capacity: ${blockSize} tokens/block x ${usableBlocks} usable blocks (${maxBlocks} configured; 1 reserved) = ${capacity} tokens. Token capacity is sized by the usable Max Cache Blocks; --cache-memory-mb/--cache-memory-percent set the Apple unified-memory ceiling that evicts free blocks (they bound RAM, not token capacity).\n`
}

function isZayaCacheStackMigrationTarget(modelPath?: string): boolean {
  const lower = String(modelPath || '').toLowerCase()
  return lower.includes('zaya1') || lower.includes('zaya')
}

/**
 * v17 (2026-08-21): the in-RAM paged cache default flips back OFF, and the
 * block-disk budget moves from a flat GB number to a percent of the volume.
 * Existing installs must move too, otherwise an updated user silently keeps the
 * RAM mirror the new default exists to avoid — the parity gap that would make
 * "it ships off" true only for fresh installs.
 *
 * Measured basis: paged buys 2.68s vs 2.70s on a full hit and 3.48s vs 3.57s on
 * a partial (under 2%), while the mirror costs 15% of unified memory.
 *
 * This runs as a POST-pass, after every legacy migration has matched. Each of
 * those migrations is an exact-tuple fingerprint that includes `usePagedCache`,
 * so flipping it first silently rewrites near-miss tuples into exact matches and
 * re-fires migrations a user had deliberately escaped. Ordering is the whole
 * correctness argument here, not a style preference.
 */
function applySsdFirstCacheDefaults(
  config: Partial<ServerConfig>,
  _modelPath?: string,
  _detectedCacheType?: string,
  _detectedCacheSubtype?: string,
  detectedFamily?: string,
): boolean {
  let changed = false
  // No family may preserve a stale RAM-paged value in persisted state. ZAYA's
  // missing SSD reconstruction path is a real open runtime gap, not permission
  // for migrations to silently re-enable the retired tier.
  if (config.usePagedCache === true) {
    config.usePagedCache = false
    changed = true
  }
  // Exact typed prompt-snapshot families own a separate prompt-L2 format.
  // Apply the RAM-off normalization above, but do not rewrite their disk
  // format into the generic block tier below.
  if (usesExactTypedPromptDiskCache(detectedFamily)) return changed
  if (config.enableBlockDiskCache !== true) {
    // SSD-only is only cheap when the disk tier is actually on.
    config.enableBlockDiskCache = true
    changed = true
  }
  // A DELIBERATE unlimited must survive the move to percent. Shipped builds
  // offered a GB slider whose "Unlimited" position stored 0, so an upgrading
  // user may hold 0 on purpose. Dropping it and letting the 10% default apply
  // would cap someone who explicitly asked for no cap — an invented limit.
  // Percent 0 means unlimited too, so the intent translates exactly.
  //
  // Ordering matters: this must run BEFORE the percent default below, or the
  // default overwrites the preserved intent.
  if (config.blockDiskCacheMaxGb != null && Number(config.blockDiskCacheMaxGb) === 0) {
    config.blockDiskCacheMaxPercent = 0
    delete config.blockDiskCacheMaxGb
    changed = true
  }
  if (config.blockDiskCacheMaxPercent == null) {
    config.blockDiskCacheMaxPercent = DEFAULT_BLOCK_DISK_CACHE_PERCENT
    changed = true
  }
  if (Number(config.blockDiskCacheMaxGb) === LEGACY_BLOCK_DISK_CACHE_MAX_GB) {
    // The old flat default; let the percent take over. An explicitly chosen size
    // that happens to be 10 is indistinguishable from the default here, which is
    // why only the exact legacy value is migrated.
    //
    // DELETE it rather than writing 0 — 0 means UNLIMITED to the engine, so
    // "handing over to the percent" by writing 0 handed over the whole disk.
    delete config.blockDiskCacheMaxGb
    changed = true
  }
  return changed
}

function applyCacheStackStartupDefaultMigration(config: Partial<ServerConfig>, modelPath?: string): boolean {
  const cacheDefaultsVersion = Number(config.cacheStackStartupDefaultsVersion || 0)
  if (cacheDefaultsVersion >= CACHE_STACK_STARTUP_DEFAULTS_VERSION) {
    return false
  }
  const legacyChanged = applyLegacyCacheStackMigrations(config, modelPath)
  markCacheStackStartupDefaultsCurrent(config, modelPath || config.modelPath)
  if (
    Number(config.cacheStackStartupDefaultsVersion || 0) !==
    CACHE_STACK_STARTUP_DEFAULTS_VERSION
  ) {
    // The stamp declines while the bundle is unreachable, so the migration can
    // retry once the drive is mounted. The SSD-first pass has to honour that
    // too: writing usePagedCache / blockDiskCacheMaxGb on a pass that
    // deliberately did NOT complete rewrites the exact-tuple fingerprint the
    // retry matches on, and the retry then never fires. Models live on an
    // external drive here, so this path is routine.
    return legacyChanged
  }
  let detected: ReturnType<typeof detectModelConfigFromDir> | undefined
  try {
    detected = detectModelConfigFromDir(String(modelPath || config.modelPath || ''))
  } catch {
    /* detection best-effort; an unresolvable bundle just gets the generic flip */
  }
  const ssdFirstChanged = applySsdFirstCacheDefaults(
    config,
    modelPath,
    detected?.cacheType,
    detected?.cacheSubtype,
    detected?.family,
  )
  return legacyChanged || ssdFirstChanged
}

function applyLegacyCacheStackMigrations(config: Partial<ServerConfig>, modelPath?: string): boolean {
  const cacheDefaultsVersion = Number(config.cacheStackStartupDefaultsVersion || 0)
  if (cacheDefaultsVersion >= CACHE_STACK_STARTUP_DEFAULTS_VERSION) {
    return false
  }

  const zayaCacheMigrationTarget = isZayaCacheStackMigrationTarget(modelPath || config.modelPath)
  const isM3MigrateTarget = /minimax.?m3/i.test((modelPath || config.modelPath || "").toLowerCase())
  // v8 paged-default-ON: resolve the detected per-family paged capability so the
  // generic-family migration branch flips existing paged-OFF sessions to the new
  // detected default. Best-effort.
  let migrationDetectedFamily: string | undefined
  let migrationDetectedUsePaged: boolean | undefined
  try {
    const _migDetected = detectModelConfigFromDir(String(modelPath || config.modelPath || ''))
    migrationDetectedFamily = normalizeDetectedFamilyName(_migDetected.family)
    migrationDetectedUsePaged = _migDetected.usePagedCache
  } catch {
    /* detection best-effort; leave undefined -> paged-off in the generic branch */
  }
  const migrationEffectiveFamily = normalizeDetectedFamilyName(
    resolveEffectiveModelFamily(config.modelFamily, migrationDetectedFamily),
  )
  const migrationDsv4Active = migrationEffectiveFamily === 'deepseek-v4'
  // v12: DSV4 native composite block records became a product cache lane.
  // v13 restores the standard hot/warm/cold default now that typed RAM blocks,
  // bounded block-disk L2, and SSD refault share one native block index. Only
  // exact prior default tuples migrate; near-misses remain user-owned.
  const staleV11Dsv4FailClosed =
    migrationDsv4Active &&
    cacheDefaultsVersion === 11 &&
    config.continuousBatching === true &&
    Number(config.maxNumSeqs) === 1 &&
    Number(config.prefillBatchSize) === 512 &&
    Number(config.prefillStepSize) === 2048 &&
    Number(config.completionBatchSize) === 512 &&
    config.dsv4PrefixCache === false &&
    config.enablePrefixCache === false &&
    Number(config.prefixCacheSize) === 100 &&
    Number(config.prefixCacheMaxBytes) === 0 &&
    Number(config.cacheMemoryMb) === 0 &&
    Number(config.cacheMemoryPercent) === 15 &&
    Number(config.cacheTtlMinutes) === 0 &&
    config.noMemoryAwareCache === false &&
    config.usePagedCache === false &&
    config.enableDiskCache === false &&
    Number(config.diskCacheMaxGb) === 10 &&
    String(config.diskCacheDir || '') === '' &&
    config.enableBlockDiskCache === false &&
    Number(config.blockDiskCacheMaxGb) === 10 &&
    String(config.blockDiskCacheDir || '') === '' &&
    Number(config.maxCacheBlocks) === 1000 &&
    Number(config.pagedCacheBlockSize) === DSV4_PAGED_CACHE_BLOCK_SIZE &&
    config.kvCacheQuantization === 'auto' &&
    Number(config.kvCacheGroupSize) === 64
  const staleV12Dsv4SsdOnlyDefault =
    migrationDsv4Active &&
    cacheDefaultsVersion === 12 &&
    config.continuousBatching === true &&
    Number(config.maxNumSeqs) === 1 &&
    Number(config.prefillBatchSize) === 512 &&
    Number(config.prefillStepSize) === 2048 &&
    Number(config.completionBatchSize) === 512 &&
    config.dsv4PrefixCache === true &&
    config.enablePrefixCache === true &&
    Number(config.prefixCacheSize) === 100 &&
    Number(config.prefixCacheMaxBytes) === 0 &&
    Number(config.cacheMemoryMb) === 0 &&
    Number(config.cacheMemoryPercent) === 15 &&
    Number(config.cacheTtlMinutes) === 0 &&
    config.noMemoryAwareCache === false &&
    config.usePagedCache === false &&
    config.enableDiskCache === false &&
    Number(config.diskCacheMaxGb) === 10 &&
    String(config.diskCacheDir || '') === '' &&
    config.enableBlockDiskCache === true &&
    Number(config.blockDiskCacheMaxGb) === 10 &&
    String(config.blockDiskCacheDir || '') === '' &&
    Number(config.pagedCacheBlockSize) === DSV4_PAGED_CACHE_BLOCK_SIZE &&
    Number(config.maxCacheBlocks) === DSV4_MAX_CACHE_BLOCKS &&
    config.kvCacheQuantization === 'auto' &&
    Number(config.kvCacheGroupSize) === 64
  if (migrationDsv4Active) {
    if (!staleV11Dsv4FailClosed && !staleV12Dsv4SsdOnlyDefault) return false
    config.enablePrefixCache = true
    config.dsv4PrefixCache = true
    config.usePagedCache = false
    config.enableDiskCache = false
    config.enableBlockDiskCache = true
    config.pagedCacheBlockSize = DSV4_PAGED_CACHE_BLOCK_SIZE
    config.maxCacheBlocks = DSV4_MAX_CACHE_BLOCKS
    config.kvCacheQuantization = 'auto'
    markCacheStackStartupDefaultsCurrent(config, modelPath || config.modelPath)
    return true
  }
  // v16: a v13/v14/v15 session sitting at paged-OFF only because the multimodal
  // paged-off override used to clear it, while detection now resolves the same
  // bundle to paged-ON (muse-glimmer joined gemma4 on the typed mixed-SWA paged
  // lane). Must be evaluated BEFORE the v12 cutoff below, which exists to stop
  // older generic predicates re-firing and would otherwise make this dead code.
  // Gated on the detected value having actually flipped to true, so no other
  // family is touched, and on the untouched generic L2 tuple, so a deliberate
  // user paged-off is preserved. The stale flat 1000 index is lifted with it.
  if (
    cacheDefaultsVersion >= 13 &&
    cacheDefaultsVersion < CACHE_STACK_STARTUP_DEFAULTS_VERSION &&
    migrationDetectedUsePaged === true &&
    !migrationDsv4Active &&
    !zayaCacheMigrationTarget &&
    !isM3MigrateTarget &&
    config.usePagedCache === false &&
    config.continuousBatching === true &&
    config.enablePrefixCache === true &&
    config.enableDiskCache === false &&
    config.enableBlockDiskCache === true
  ) {
    // Paged RAM stays OFF for every family; this migration used to turn a saved
    // Off back On when the registry reported a per-family paged capability.
    config.usePagedCache = false
    if (config.maxCacheBlocks === undefined || Number(config.maxCacheBlocks) === 1000) {
      config.maxCacheBlocks = indexBlocksForCapacity(config.pagedCacheBlockSize)
    }
    markCacheStackStartupDefaultsCurrent(config, modelPath || config.modelPath)
    return true
  }
  // v13 changes only the DSV4 paged-RAM default. A v12 non-DSV4 session has
  // no migration work to do, and must not fall back through older generic
  // predicates merely because the global defaults version advanced.
  if (cacheDefaultsVersion >= 12) return false
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
  // v2 migration pushed GENERIC (non-path-dependent) sessions to paged-ON
  // (usePagedCache=true, blockDisk, kvq=auto, 512/512, maxSeqs=1). Phase-1 (v3-v5)
  // flipped to paged-OFF + legacy disk_cache. Phase-2 (v6, 2026-06-27) restored
  // paged-ON + block-disk
  // for generics after live RAM-soak proved no leak (Ornith 9B-MXFP4: 30 turns mixed
  // mode, +5.4% RSS plateaued by turn 11).
  const staleV2GenericPagedOn =
    !zayaCacheMigrationTarget &&
    config.continuousBatching === true &&
    config.enablePrefixCache === true &&
    Number(config.maxNumSeqs) === 1 &&
    Number(config.prefillBatchSize) === 512 &&
    Number(config.completionBatchSize) === 512 &&
    config.usePagedCache === true &&
    config.enableBlockDiskCache === true &&
    config.kvCacheQuantization === 'auto'
  // Phase-1 v3-v5 fingerprint: generic forced to paged-OFF + legacy disk_cache.
  // Phase-2 v6 flipped to paged-ON. Phase-3 v7 (Eric 2026-06-30) flips back to
  // paged-OFF as the safe default — see feedback_paged_cache_default_off_2026_06_30.
  const stalePhase1GenericPagedOff =
    !zayaCacheMigrationTarget &&
    !isM3MigrateTarget &&
    config.continuousBatching === true &&
    config.enablePrefixCache === true &&
    config.usePagedCache === false &&
    config.enableDiskCache === true &&
    config.enableBlockDiskCache === false
  // Phase-2 v6 fingerprint: paged-ON + block-disk-ON. Phase-3 v7 migration
  // resets these back to OFF for non-structural families.
  const stalePhase2GenericPagedOn =
    !zayaCacheMigrationTarget &&
    !isM3MigrateTarget &&
    config.continuousBatching === true &&
    config.enablePrefixCache === true &&
    config.usePagedCache === true &&
    config.enableBlockDiskCache === true &&
    config.enableDiskCache === false
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
  // A pre-v9 launch could preserve an adopted/pre-existing session with an
  // impossible UI tuple from older defaults: paged ON + legacy prompt L2 ON +
  // block L2 OFF. The UI cache policy cannot create that combination because
  // enabling paged clears legacy L2 and enables block L2. Migrate only this
  // exact stale tuple from any pre-v9 version so a valid explicit block-L2
  // opt-out (paged ON + both disk caches OFF) remains untouched.
  const stalePreV9PagedLegacyDiskWithoutBlockL2 =
    Number(config.cacheStackStartupDefaultsVersion || 0) < 9 &&
    !migrationDsv4Active &&
    migrationDetectedUsePaged === true &&
    config.continuousBatching === true &&
    config.enablePrefixCache === true &&
    Number(config.maxNumSeqs) === 1 &&
    Number(config.prefillBatchSize) === 512 &&
    Number(config.prefillStepSize) === 2048 &&
    Number(config.completionBatchSize) === 512 &&
    config.usePagedCache === true &&
    config.enableDiskCache === true &&
    config.enableBlockDiskCache === false &&
    config.kvCacheQuantization === 'auto' &&
    Number(config.maxCacheBlocks) === 1000 &&
    Number(config.pagedCacheBlockSize) === 64 &&
    Number(config.blockDiskCacheMaxGb) === 10
  // v10: M3 gained a typed paged/block-L2 serializer that preserves sparse MSA
  // keys, values, idx_keys, and absolute offsets. Migrate only the exact v9 M3
  // default tuple; any deviation is treated as an explicit user choice.
  const staleV9M3PagedOffWithLegacyL2 =
    isM3MigrateTarget &&
    migrationDetectedFamily === 'minimax_m3' &&
    migrationDetectedUsePaged === true &&
    Number(config.cacheStackStartupDefaultsVersion || 0) === 9 &&
    config.continuousBatching === true &&
    config.enablePrefixCache === true &&
    Number(config.maxNumSeqs) === 1 &&
    Number(config.prefillBatchSize) === 512 &&
    Number(config.prefillStepSize) === 2048 &&
    Number(config.completionBatchSize) === 512 &&
    config.usePagedCache === false &&
    config.enableDiskCache === true &&
    config.enableBlockDiskCache === false &&
    config.kvCacheQuantization === 'auto' &&
    Number(config.maxCacheBlocks) === 1000 &&
    Number(config.pagedCacheBlockSize) === 64 &&
    Number(config.blockDiskCacheMaxGb) === 10
  // v11: block L2 became independent of paged RAM. Migrate only the exact v10
  // untouched non-paged default tuple from prompt-L2 to disk-only block L2;
  // any divergent value remains an explicit user choice.
  const staleV10NonPagedPromptL2 =
    Number(config.cacheStackStartupDefaultsVersion || 0) === 10 &&
    migrationDetectedFamily !== 'openpangu_v2' &&
    !migrationDsv4Active &&
    migrationDetectedUsePaged !== true &&
    config.continuousBatching === true &&
    config.enablePrefixCache === true &&
    Number(config.maxNumSeqs) === 1 &&
    Number(config.prefillBatchSize) === 512 &&
    Number(config.prefillStepSize) === 2048 &&
    Number(config.completionBatchSize) === 512 &&
    config.usePagedCache === false &&
    config.enableDiskCache === true &&
    config.enableBlockDiskCache === false &&
    config.kvCacheQuantization === 'auto' &&
    Number(config.maxCacheBlocks) === 1000 &&
    Number(config.pagedCacheBlockSize) === 64 &&
    Number(config.blockDiskCacheMaxGb) === 10
  // v8 (2026-07-12): flip families that the v7 default left paged-OFF to their
  // detected paged default. Scoped to prefix-cache-on continuous-batching sessions.
  const staleV7GenericPagedOff =
    !zayaCacheMigrationTarget &&
    !isM3MigrateTarget &&
    migrationDetectedUsePaged === true &&
    Number(config.cacheStackStartupDefaultsVersion || 0) === 7 &&
    config.continuousBatching === true &&
    config.enablePrefixCache === true &&
    config.usePagedCache === false &&
    // EXACT v7 generic paged-OFF default tuple across EVERY field the migration
    // below overwrites/flips — any deviation means the user touched this session,
    // so we leave it untouched (preserve intentional user config).
    Number(config.maxNumSeqs) === 1 &&
    Number(config.prefillBatchSize) === 512 &&
    Number(config.prefillStepSize) === 2048 &&
    Number(config.completionBatchSize) === 512 &&
    config.enableDiskCache === false &&
    config.enableBlockDiskCache === false &&
    config.kvCacheQuantization === 'auto' &&
    Number(config.cacheMemoryPercent) === 15 &&
    Number(config.cacheMemoryMb) === 0 &&
    Number(config.maxCacheBlocks) === 1000 &&
    Number(config.pagedCacheBlockSize) === 64 &&
    Number(config.blockDiskCacheMaxGb) === 10 &&
    config.noMemoryAwareCache === false &&
    Number(config.prefixCacheSize) === 100 &&
    Number(config.prefixCacheMaxBytes) === 0

  if (
    !staleContinuousDefaults &&
    !staleNoPrefixBatchDefaults &&
    !stalePartialPagedCacheDefaults &&
    !staleExplicitNoneCacheCodecDefaults &&
    !stalePreV9PagedLegacyDiskWithoutBlockL2 &&
    !staleV9M3PagedOffWithLegacyL2 &&
    !staleV10NonPagedPromptL2 &&
    !staleV2GenericPagedOn &&
    !stalePhase1GenericPagedOff &&
    !stalePhase2GenericPagedOn &&
    !staleV7GenericPagedOff
  ) return false

  config.continuousBatching = true
  config.enablePrefixCache = true
  config.maxNumSeqs = 1
  config.prefillBatchSize = 512
  config.prefillStepSize = 2048
  config.completionBatchSize = 512
  config.kvCacheQuantization = 'auto'
  config.cacheMemoryPercent = 15
  if (zayaCacheMigrationTarget) {
    // Path-dependent (ZAYA CCA): recurrent state is not position-sliceable and
    // the SSD-only non-paged lane is not wired for it. Paged RAM is OFF for
    // every family regardless, so ZAYA gets NO prefix reuse and re-prefills
    // cleanly. That costs speed on ZAYA, never correctness -- the engine's
    // fetch side refuses a ZAYA chain with no terminal CCA state.
    config.usePagedCache = false
    config.maxCacheBlocks = 1000
    config.enableDiskCache = false
    config.enableBlockDiskCache = true
    config.blockDiskCacheMaxGb = 10
  } else {
    // Generic / Gemma-SWA / MoE / hybrid: paged cache OFF by default
    // (Eric directive 2026-06-30 v7, supersedes 2026-06-27 v6 which was ON).
    // Paged-off is the safe default; users who want the SSD prefix-hit
    // benefit toggle paged ON in the UI. The toggle must actually reach the
    // engine argv — verify --use-paged-cache appears when checkbox toggled.
    // SSD block-disk-cache is independent of paged RAM and remains default-on.
    // Structural families (DSV4/ZAYA CCA/hybrid SSM SWA subtype) still route
    // through cacheTypeRequiresPaged/cacheSubtypeRequiresPaged at spawn time
    // and stay paged-required — that's a runtime constraint, not a default.
    // v8 (Eric 2026-07-12, reverses v7): paged cache defaults ON for autodetected
    // families whose registry declares a safe generic or typed paged serializer.
    // Derive from the fully-resolved detected capability so existing generic
    // sessions inherit the detected paged default while block-disk L2 stays on.
    const migratedGenericPaged = migrationDetectedUsePaged ?? false
    config.usePagedCache = migratedGenericPaged
    // v14: the old flat 1000 indexes only 63,936 tokens at the generic
    // 64-token block, silently capping prefix reuse far below the model's
    // context window. Measured on Gemma 4: a 77k prompt reported 0 cached
    // tokens on an exact repeat and ran SLOWER than a cold prefill. Lift only
    // that exact stale value so a number the user chose is never overwritten.
    config.maxCacheBlocks =
      config.maxCacheBlocks === undefined || Number(config.maxCacheBlocks) === 1000
        ? indexBlocksForCapacity(config.pagedCacheBlockSize)
        : config.maxCacheBlocks
    config.enableDiskCache = false
    config.enableBlockDiskCache = true
    config.blockDiskCacheMaxGb = config.blockDiskCacheMaxGb ?? 10
  }
  markCacheStackStartupDefaultsCurrent(config, modelPath || config.modelPath)
  return true
}

/** Resolve bind address to connectable address (0.0.0.0 → 127.0.0.1) */
export function connectHost(host: string): string {
  return host === '0.0.0.0' ? '127.0.0.1' : host
}

// estimateModelFileBytes and formatGb are imported from ./modelLaunchMemory —
// this file used to carry byte-identical private copies (a #46 consolidation
// target). Keeping one definition means the recursive counter and the GB
// formatter can never drift between the admission preflight and the load bar.

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
  /** Refcounted user-requested stops that are actively terminating a backend. */
  private intentionalStops = new Map<string, number>()
  // Per-session lifecycle epoch: every EXPLICIT stop advances it immediately
  // (before taking the session lock), and any queued start that captured an
  // older epoch aborts instead of spawning. The session lock serializes
  // start/stop but serialization is not cancellation: Save & Restart issues
  // update -> stop -> start as separate IPC calls, and a user Stop landing
  // between them was executed BEFORE the queued start, which then spawned a
  // ~100GB engine the UI no longer tracked (installed 1.6.44 smoke, PID
  // 98861: UI=Stopped, port down, engine alive). Stop must win.
  private lifecycleEpochs = new Map<string, number>()
  /** Per-session operation lock to prevent concurrent start/stop races */
  private operationLocks = new Map<string, Promise<void>>()
  /** Global creation lock to prevent port assignment races between concurrent createSession calls */
  private creationLock: Promise<void> = Promise.resolve()
  /** Serialize manual/UI starts while single-model mode is replacing another local engine. */
  private singleModelStartTransitionPending: Promise<void> = Promise.resolve()
  /** Timestamp of last successful health check per session (used to skip redundant per-message checks) */
  private lastHealthyAt = new Map<string, number>()
  /** Per-session ring buffer for log lines (capped at LOG_BUFFER_MAX_LINES) */
  private logBuffers = new Map<string, string[]>()
  /** Successful preflight records awaiting this bundle's engine start. */
  private pendingBundlePreflightLogs = new Map<string, { modelPath: string; lines: string[] }>()
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
    // Migrate persisted cache defaults before the renderer first lists sessions.
    // Start-time migration alone makes the engine argv correct, but leaves the
    // Settings UI showing a stale cache stack until the model is launched. The
    // UI must reflect the configuration that launch will actually use.
    // Runs at module scope, before the app is ready — a list read must not
    // reach the keychain, or the main thread blocks before any window exists.
    for (const session of db.getSessions()) {
      let config: Partial<ServerConfig>
      try {
        config = JSON.parse(session.config || '{}')
      } catch {
        continue
      }
      const cacheDefaultsFilled = applyMissingCacheStackStartupDefaults(config, session.modelPath)
      const migrated = applyCacheStackStartupDefaultMigration(config, session.modelPath)
      const normalized = normalizeCacheStackMutualExclusion(config)
      const markedCurrent = markCacheStackStartupDefaultsCurrent(config, session.modelPath)
      if (cacheDefaultsFilled || migrated || normalized || markedCurrent) {
        db.updateSession(session.id, { config: JSON.stringify(config) })
        console.log(`[SESSION] Persisted startup cache defaults for session ${session.id}`)
      }
    }
  }

  /** Get timestamp of last successful health check for a session (0 if never checked) */
  getLastHealthyAt(sessionId: string): number {
    return this.lastHealthyAt.get(sessionId) || 0
  }

  // Loading progress patterns — matched against engine stdout/stderr to detect loading phase
  private static readonly LOAD_PROGRESS_PATTERNS: Array<{
    pattern: RegExp
    /** English text, kept so engine logs and older renderers still read. */
    label: string
    /** i18n key for the SAME text. The main process cannot translate — it has
     *  no locale catalog — so it ships both and the renderer resolves the key,
     *  falling back to `label` when a locale is missing the entry. */
    labelKey: string
    progress: number
  }> = [
    { pattern: /Loading model:/, label: 'Initializing...', labelKey: 'main.loadProgress.initializing', progress: 5 },
    // Phase 1: Process startup + config (0-25%)
    // For BatchedEngine, these fire BEFORE actual model loading (lifespan phase).
    // Keep progress low so real loading patterns (Phase 2) can advance the bar.
    { pattern: /System memory before load/, label: 'Checking memory...', labelKey: 'main.loadProgress.checkingMemory', progress: 5 },
    { pattern: /Loading model with (?:Simple|Batched)Engine/, label: 'Creating engine...', labelKey: 'main.loadProgress.creatingEngine', progress: 8 },
    { pattern: /\bmodel loaded \(batched mode\)/i, label: 'Starting server...', labelKey: 'main.loadProgress.startingServer', progress: 10 },
    { pattern: /Metal GPU memory after load/, label: 'Server initializing...', labelKey: 'main.loadProgress.serverInitializing', progress: 12 },
    { pattern: /Native tool format enabled/, label: 'Configuring tools...', labelKey: 'main.loadProgress.configuringTools', progress: 14 },
    { pattern: /Default max tokens:/, label: 'Configuring limits...', labelKey: 'main.loadProgress.configuringLimits', progress: 16 },
    { pattern: /Uvicorn running on/, label: 'Server started, loading model...', labelKey: 'main.loadProgress.serverStartedLoadingModel', progress: 20 },
    { pattern: /Waiting for application startup/, label: 'Starting model runtime...', labelKey: 'main.loadProgress.startingModelRuntime', progress: 22 },

    // Phase 2: Actual model loading (25-85%)
    // For BatchedEngine these fire DURING lifespan() (after Uvicorn starts).
    // For SimpleEngine these fire DURING load_model() (before Uvicorn starts).
    { pattern: /JANG v2 detected/, label: 'Loading JANG weights...', labelKey: 'main.loadProgress.loadingJangWeights', progress: 30 },
    { pattern: /Loading JANG v1 VLM:/, label: 'Loading JANG VL model...', labelKey: 'main.loadProgress.loadingJangVl', progress: 30 },
    { pattern: /Loading MLLM:/, label: 'Loading vision model...', labelKey: 'main.loadProgress.loadingVisionModel', progress: 30 },
    { pattern: /Loading image model:/, label: 'Loading image model...', labelKey: 'main.loadProgress.loadingImageModel', progress: 30 },
    { pattern: /Loading JANG VL model:/, label: 'Loading JANG VL...', labelKey: 'main.loadProgress.loadingJangVlShort', progress: 30 },
    { pattern: /Loading \d+ safetensors shards/, label: 'Loading weights...', labelKey: 'main.loadProgress.loadingWeights', progress: 40 },
    { pattern: /Split kv_b_proj layer/, label: 'Processing MLA layers...', labelKey: 'main.loadProgress.processingMla', progress: 50 },
    { pattern: /bfloat16 enabled/, label: 'Converting to bfloat16...', labelKey: 'main.loadProgress.convertingBfloat16', progress: 55 },
    { pattern: /JANG v[12].{0,10}loaded in/, label: 'Weights loaded', labelKey: 'main.loadProgress.weightsLoaded', progress: 65 },
    { pattern: /Model loaded successfully/, label: 'Model loaded', labelKey: 'main.loadProgress.modelLoaded', progress: 65 },
    { pattern: /MLLM loaded successfully/, label: 'Vision model loaded', labelKey: 'main.loadProgress.visionModelLoaded', progress: 65 },
    { pattern: /JANG VL model loaded/, label: 'JANG VL loaded', labelKey: 'main.loadProgress.jangVlLoaded', progress: 65 },
    { pattern: /Image model loaded in/, label: 'Image model loaded', labelKey: 'main.loadProgress.imageModelLoaded', progress: 65 },

    // Phase 3: Post-load config (85-92%)
    // SimpleEngine: fires during load_model(). BatchedEngine: fires during lifespan().
    { pattern: /\bmodel loaded \(simple mode\)/i, label: 'Engine ready', labelKey: 'main.loadProgress.engineReady', progress: 70 },
    { pattern: /Saved \d+\/\d+ layer weights to SSD/, label: 'Saving weights to SSD...', labelKey: 'main.loadProgress.savingWeights', progress: 72 },
    { pattern: /SSD weight index:/, label: 'Building weight index...', labelKey: 'main.loadProgress.buildingIndex', progress: 73 },
    { pattern: /SSD per-layer weight recycling configured/, label: 'SSD streaming ready', labelKey: 'main.loadProgress.ssdStreamingReady', progress: 74 },
    { pattern: /KV cache quantization/, label: 'Setting up KV cache...', labelKey: 'main.loadProgress.settingUpKvCache', progress: 78 },
    { pattern: /(?:Chat template loaded|Applied custom chat template)/, label: 'Loading chat template...', labelKey: 'main.loadProgress.loadingChatTemplate', progress: 80 },
    { pattern: /PagedCacheManager initialized/, label: 'Configuring cache...', labelKey: 'main.loadProgress.configuringCache', progress: 82 },
    { pattern: /Scheduler (?:initialized|started)/, label: 'Starting scheduler...', labelKey: 'main.loadProgress.startingScheduler', progress: 85 },
    { pattern: /BatchedEngine loaded/, label: 'Engine ready', labelKey: 'main.loadProgress.engineReady', progress: 88 },
    { pattern: /Application startup complete/, label: 'Almost ready...', labelKey: 'main.loadProgress.almostReady', progress: 92 },
  ]

  // Track last emitted progress per session to avoid duplicate events
  private loadProgressState = new Map<string, number>()
  private loadProgressMeta = new Map<string, {
    modelBytes?: number;
    expectedResidentBytes?: number;
    lazyResident?: boolean;
    residentMb?: number;
    residentPercent?: number;
    residentHighWaterBytes?: number;
    lastStartupProgressAt?: number;
  }>()
  private loadResidentTimers = new Map<string, ReturnType<typeof setInterval>>()
  // Engine-owned lifecycle progress (LOADPROGRESS stdout lines / /health
  // load_progress). Generation guards discard stale events after a
  // stop/restart/PID replacement; contractSessions marks engines that speak
  // the contract so the legacy log-pattern heuristics stay off for them.
  private lifecycleGenerations = new Map<string, number>()
  private contractSessions = new Set<string>()
  // Last emitted progress event per session — the hydration source for a
  // renderer that navigates into a page mid-load (events sent before the
  // page opened are gone; this snapshot is not).
  private lastLoadProgressEvents = new Map<string, Record<string, unknown>>()
  // 1s /health pollers driving wake progress for engines with no piped
  // stdout (adopted processes).
  private wakeHealthPollers = new Map<string, ReturnType<typeof setInterval>>()
  // Externally-triggered JIT wakes (API request or chat message) detected via
  // /health `wake_in_progress`. Distinct from wakePending, which marks wakes
  // this process initiated through /admin/wake.
  private externalWakes = new Set<string>()
  // admin/wake is synchronous, while the global health monitor continues to
  // poll.  During a legitimate deep reload Python still reports
  // `standby_deep`; keep explicit ownership so that transient state cannot be
  // mistaken for a failed wake and overwrite the visible loading state.
  private wakePending = new Set<string>()

  private stopLoadResidentMonitor(sessionId: string): void {
    const timer = this.loadResidentTimers.get(sessionId)
    if (timer) {
      clearInterval(timer)
      this.loadResidentTimers.delete(sessionId)
    }
  }

  private stopWakeHealthPoller(sessionId: string): void {
    const poller = this.wakeHealthPollers.get(sessionId)
    if (poller) {
      clearInterval(poller)
      this.wakeHealthPollers.delete(sessionId)
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

    // Normalize progress against what THIS family actually keeps resident:
    // ordinary loads copy every weight into Metal (expected ≈ file bytes), so
    // the bar tracks the real copy; expert-streaming families (DSV4-Flash,
    // MM3) plateau far below file size, and dividing by file bytes would
    // freeze their bar at ~30% for the whole load.
    const storedMeta = this.loadProgressMeta.get(sessionId)
    const expectedResidentBytes = storedMeta?.expectedResidentBytes && storedMeta.expectedResidentBytes > 0
      ? storedMeta.expectedResidentBytes
      : modelBytes
    const streamsWeights = storedMeta?.lazyResident === true

    // RSS is a DIAGNOSTIC readout, never the percentage oracle: the display
    // percentage comes exclusively from the engine's lifecycle-progress
    // contract (phases + shard units + ready). This monitor only refreshes
    // the resident-RAM line rendered under the bar, by re-emitting the last
    // authoritative progress event with fresh residency numbers.
    // mmap/SSD-streaming families never make every bundle byte resident,
    // which is exactly why residency cannot drive the bar.
    const tick = () => {
      const session = db.getSession(sessionId)
      if (!session || session.status !== 'loading') {
        this.stopLoadResidentMonitor(sessionId)
        return
      }
      const residentBytes = this.readProcessGroupResidentBytes(pid)
      if (residentBytes <= 0) return

      const residentPercent = Math.min(100, Math.max(0, (residentBytes / expectedResidentBytes) * 100))
      const previousMeta = this.loadProgressMeta.get(sessionId) || {}
      const previousHighWater = previousMeta.residentHighWaterBytes || 0
      const residentAdvanced = residentBytes >= previousHighWater + 1048576
      const meta = {
        ...previousMeta,
        modelBytes,
        expectedResidentBytes,
        lazyResident: streamsWeights,
        residentMb: Math.round((residentBytes / 1048576) * 10) / 10,
        residentPercent: Math.round(residentPercent * 10) / 10,
        residentHighWaterBytes: Math.max(previousHighWater, residentBytes),
        lastStartupProgressAt: residentAdvanced
          ? Date.now()
          : previousMeta.lastStartupProgressAt,
      }
      this.loadProgressMeta.set(sessionId, meta)
      const last = this.lastLoadProgressEvents.get(sessionId)
      if (last) {
        this.emitLoadProgress({ ...last, ...meta, sessionId })
      }
    }

    tick()
    this.loadResidentTimers.set(sessionId, setInterval(tick, 1000))
  }

  /**
   * A session that transitions to error must not leave stale progress
   * bookkeeping behind: a renderer reload would hydrate a bar for a dead
   * attempt, and a stale generation high-water mark would discard the
   * replacement engine's events.
   */
  private clearLoadProgressBookkeeping(sessionId: string): void {
    this.stopLoadResidentMonitor(sessionId)
    this.stopWakeHealthPoller(sessionId)
    this.loadProgressState.delete(sessionId)
    this.loadProgressMeta.delete(sessionId)
    this.lastLoadProgressEvents.delete(sessionId)
    this.lifecycleGenerations.delete(sessionId)
    this.contractSessions.delete(sessionId)
    this.externalWakes.delete(sessionId)
  }

  /** Store-and-emit so a page opened mid-load can hydrate the current state. */
  private emitLoadProgress(payload: { sessionId: string } & Record<string, unknown>): void {
    this.lastLoadProgressEvents.set(payload.sessionId, payload)
    this.emit('session:loadProgress', payload)
  }

  /** Hydration source for renderers that navigate into a page mid-load. */
  getLoadProgressSnapshot(): Record<string, Record<string, unknown>> {
    return Object.fromEntries(this.lastLoadProgressEvents)
  }

  /**
   * Terminal 100% for paths where the engine's own ready event cannot reach
   * us (adopted engines with no piped stdout, remote sessions). Spawned
   * engines normally deliver ready=true themselves through the contract.
   */
  private emitTerminalLoadProgress(sessionId: string): void {
    this.stopLoadResidentMonitor(sessionId)
    this.stopWakeHealthPoller(sessionId)
    const meta = this.loadProgressMeta.get(sessionId) || {}
    this.loadProgressState.set(sessionId, 100)
    this.emitLoadProgress({
      sessionId,
      label: 'Model ready',
      labelKey: 'main.loadProgress.modelReady',
      progress: 100,
      indeterminate: false,
      ...meta,
    })
  }

  /**
   * Apply one engine lifecycle snapshot (from a LOADPROGRESS stdout line or
   * /health `load_progress`). Generation-guarded: events from an older
   * load/wake attempt are discarded, so nothing stale can repaint the bar
   * after a Stop, restart, or PID replacement.
   */
  private applyLifecycleSnapshot(sessionId: string, snap: LifecycleSnapshot): void {
    const lastGeneration = this.lifecycleGenerations.get(sessionId) ?? -1
    if (snap.generation < lastGeneration) return
    if (snap.generation > lastGeneration) {
      this.lifecycleGenerations.set(sessionId, snap.generation)
      // New attempt: the monotonic display guard restarts with it.
      this.loadProgressState.set(sessionId, 0)
    }
    this.contractSessions.add(sessionId)
    if (snap.phase === 'idle' && !snap.ready) {
      // Sleep transition or failed attempt — standby/error events own the UI.
      return
    }
    // Determinate percentages exist only for measured units; phases without
    // a denominator render indeterminate. The monotonic guard applies to the
    // measured values so a shard count can never appear to run backwards.
    const display = lifecycleDisplay(snap)
    let progress = this.loadProgressState.get(sessionId) ?? 0
    if (!display.indeterminate) {
      progress = Math.max(progress, display.percent)
      this.loadProgressState.set(sessionId, progress)
    }
    const { label, labelKey, labelParams } = lifecyclePhaseLabel(snap)
    this.emitLoadProgress({
      sessionId,
      label,
      labelKey,
      ...(labelParams ? { labelParams } : {}),
      progress,
      indeterminate: display.indeterminate,
      progressGeneration: snap.generation,
      phase: snap.phase,
      ...(this.loadProgressMeta.get(sessionId) || {}),
    })
    if (snap.ready) {
      this.stopLoadResidentMonitor(sessionId)
      this.stopWakeHealthPoller(sessionId)
    }
  }

  /** Parse engine LOADPROGRESS lines out of a stdout chunk. */
  private checkLifecycleProgress(sessionId: string, text: string): boolean {
    if (!text.includes('LOADPROGRESS ')) return false
    let matched = false
    const re = /LOADPROGRESS (\{[^\n\r]*\})/g
    let match: RegExpExecArray | null
    while ((match = re.exec(text)) !== null) {
      const snap = parseLifecycleSnapshot(match[1])
      if (snap) {
        this.applyLifecycleSnapshot(sessionId, snap)
        matched = true
      }
    }
    return matched
  }

  /**
   * Shared presentation for EVERY wake path — the panel's /admin/wake button,
   * an externally-detected JIT wake (API request or chat message), and the
   * woke-externally race where ready arrives before we ever saw the wake.
   * Resets progress, publishes the waking event with the family's expected
   * resident bytes, and starts the RSS monitor so the bar tracks the reload
   * into RAM. With `settling` the session is already serving (status
   * 'running') and only the residual copy into RAM remains.
   */
  private beginWakeProgress(session: Session, logLine: string): void {
    const sessionId = session.id
    const modelFileBytes = estimateModelFileBytes(session.modelPath)
    const wakeProfile = launchResidentProfileForModel(session.modelPath)
    this.loadProgressState.delete(sessionId)
    this.loadProgressMeta.delete(sessionId)
    this.stopLoadResidentMonitor(sessionId)
    this.stopWakeHealthPoller(sessionId)
    const meta = modelFileBytes > 0
      ? {
          modelBytes: modelFileBytes,
          expectedResidentBytes: Math.round(modelFileBytes * wakeProfile.ratio),
          lazyResident: wakeProfile.streamsWeights,
        }
      : {}
    if (modelFileBytes > 0) this.loadProgressMeta.set(sessionId, meta)
    this.loadProgressState.set(sessionId, 0)
    this.emitLoadProgress({
      sessionId,
      label: 'Waking from sleep...',
      labelKey: 'main.loadProgress.wakingFromSleep',
      progress: 0,
      indeterminate: true,
      ...meta,
    })
    this.pushLog(sessionId, logLine)
    if (session.pid && modelFileBytes > 0) {
      this.startLoadResidentMonitor(sessionId, session.pid, modelFileBytes)
    }
    // Spawned engines deliver contract events over their piped stdout. An
    // adopted engine has no pipe — poll its /health snapshot instead so the
    // wake still moves the bar with engine-owned phases.
    const hasStdout = Boolean(this.processes.get(sessionId)?.process)
    if (!hasStdout) {
      const host = connectHost(session.host)
      const poller = setInterval(async () => {
        const current = db.getSession(sessionId)
        if (!current || current.status !== 'loading') {
          this.stopWakeHealthPoller(sessionId)
          return
        }
        try {
          const res = await fetch(`http://${host}:${session.port}/health`, {
            signal: AbortSignal.timeout(1500),
          })
          if (!res.ok) return
          const data = await res.json()
          if (data && typeof data.load_progress === 'object' && data.load_progress) {
            const snap = parseLifecycleSnapshot(JSON.stringify(data.load_progress))
            if (snap) this.applyLifecycleSnapshot(sessionId, snap)
          }
        } catch {
          // Server busy reloading — keep polling.
        }
      }, 1000)
      this.wakeHealthPollers.set(sessionId, poller)
    }
  }

  /**
   * Wait until a session leaves the 'loading' state (ready, failed, stopped,
   * or reverted to standby). Used to queue a chat message submitted while
   * the model is still loading — the blind fixed-retry health poll used to
   * give up after 30 s and fail the message while a multi-minute cold load
   * was legitimately in progress. The caller's AbortSignal wins immediately
   * (explicit Stop/cancel must clear the queued message).
   */
  async waitForSessionLifecycleSettled(
    sessionId: string,
    opts: { timeoutMs: number; signal?: AbortSignal },
  ): Promise<string> {
    const started = Date.now()
    // Bind to the process this wait began against: a PID swap mid-wait means
    // the attempt we queued behind is gone (Stop + new Start, crash restart),
    // and the queued message must not silently ride the replacement.
    let boundPid: number | null = db.getSession(sessionId)?.pid ?? null
    while (Date.now() - started < opts.timeoutMs) {
      if (opts.signal?.aborted) throw new Error('Request canceled')
      const session = db.getSession(sessionId)
      if (!session) return 'stopped'
      const pid = session.pid ?? null
      if (boundPid == null && pid != null) boundPid = pid
      else if (boundPid != null && pid != null && pid !== boundPid) return 'replaced'
      if (session.status === 'running') {
        // Engine-authoritative readiness: a contract-speaking engine must
        // have delivered ready (terminal 100) — the DB flip alone is not the
        // readiness barrier. Non-contract engines fall back to the health
        // monitor's healthy-gated flip.
        const entry = this.lastLoadProgressEvents.get(sessionId) as
          | { progress?: number }
          | undefined
        const engineReady =
          !this.contractSessions.has(sessionId) ||
          !entry ||
          (entry.progress ?? 0) >= 100
        if (engineReady) return 'running'
      } else if (session.status !== 'loading') {
        return session.status
      }
      await new Promise(resolve => setTimeout(resolve, 500))
    }
    return 'loading'
  }

  /** Check a log line for loading progress and emit event if phase advanced */
  private checkLoadProgress(sessionId: string, text: string): void {
    // The engine's own lifecycle contract is authoritative. Once a session's
    // engine has spoken it, the legacy log-pattern heuristics below stay off
    // — two oracles disagreeing is how bars jump backwards.
    if (this.checkLifecycleProgress(sessionId, text)) return
    if (this.contractSessions.has(sessionId)) return
    for (const { pattern, label, labelKey, progress } of SessionManager.LOAD_PROGRESS_PATTERNS) {
      if (pattern.test(text)) {
        const current = this.loadProgressState.get(sessionId) ?? 0
        if (progress > current) {
          this.loadProgressState.set(sessionId, progress)
          const meta = {
            ...(this.loadProgressMeta.get(sessionId) || {}),
            lastStartupProgressAt: Date.now(),
          }
          this.loadProgressMeta.set(sessionId, meta)
          this.emitLoadProgress({
            sessionId,
            label,
            labelKey,
            progress,
            ...meta,
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
    // Local time to match chat-bubble timestamps (was UTC via toISOString)
    const now = new Date()
    const timestamp =
      [now.getHours(), now.getMinutes(), now.getSeconds()]
        .map((n) => String(n).padStart(2, '0'))
        .join(':') + `.${String(now.getMilliseconds()).padStart(3, '0')}`
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

    // A source-owned live-proof Electron runs beside the user's real app with
    // its own userData directory and explicit secondary-instance permission.
    // That process must manage only engines it launches itself. Keeping the
    // guard at the global detection boundary prevents startup adoption,
    // single-model pruning, replacement adoption, and quit/crash cleanup from
    // ever claiming an unrelated vmlx-engine PID. Normal product launches do
    // not satisfy the three-part proof gate and retain the existing adoption
    // and orphan-cleanup behavior.
    if (shouldUseProofOwnedEngineLifecycle(process.argv, process.env)) {
      return detected
    }

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
        let nativeMtpSamplingPolicy = parsed.nativeMtpSamplingPolicy
        let nativeMtpDepthPolicy = parsed.nativeMtpDepthPolicy
        let nativeMtpDepth = parsed.nativeMtpDepth
        const nativeMtpDisabled = parsed.nativeMtpDisabled
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
            const healthMtpPolicy = data?.mtp?.request_policy
            if (
              healthMtpPolicy === 'compatible-only' ||
              healthMtpPolicy === 'deterministic-defaults' ||
              healthMtpPolicy === 'greedy-only' ||
              healthMtpPolicy === 'disabled'
            ) {
              nativeMtpSamplingPolicy = healthMtpPolicy
            }
            // The live engine reports its effective depth and depth policy;
            // adoption must recover them rather than guess a family default.
            const healthDepthPolicy = data?.mtp?.depth_policy
            if (healthDepthPolicy === 'fixed' || healthDepthPolicy === 'adaptive') {
              nativeMtpDepthPolicy = healthDepthPolicy
            }
            const healthDepth = data?.mtp?.effective_depth
            if (typeof healthDepth === 'number' && Number.isFinite(healthDepth)) {
              nativeMtpDepth = healthDepth
            }
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
          nativeMtpSamplingPolicy,
          nativeMtpDepthPolicy,
          nativeMtpDepth,
          nativeMtpDisabled,
          standbyDepth
        })
      }
    } catch (_) { }

    return detected
  }

  private parsePsLine(line: string): {
    pid: number
    port: number
    modelPath: string
    nativeMtpSamplingPolicy?: 'compatible-only' | 'deterministic-defaults' | 'greedy-only'
    nativeMtpDepthPolicy?: 'fixed' | 'adaptive'
    nativeMtpDepth?: number
    nativeMtpDisabled?: boolean
  } | null {
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

      const mtpPolicyMatch = cmdStart.match(
        /--native-mtp-sampling-policy\s+(compatible-only|deterministic-defaults|greedy-only)/,
      )
      const nativeMtpSamplingPolicy = mtpPolicyMatch?.[1] as
        | 'compatible-only'
        | 'deterministic-defaults'
        | 'greedy-only'
        | undefined

      const depthPolicyMatch = cmdStart.match(/--native-mtp-depth-policy\s+(fixed|adaptive)/)
      const nativeMtpDepthPolicy = depthPolicyMatch?.[1] as 'fixed' | 'adaptive' | undefined
      const depthMatch = cmdStart.match(/--native-mtp-depth\s+(\d+)/)
      const nativeMtpDepth = depthMatch ? parseInt(depthMatch[1]) : undefined
      const nativeMtpDisabled = /(^|\s)--disable-native-mtp(\s|$)/.test(cmdStart) || undefined

      return { pid, port, modelPath, nativeMtpSamplingPolicy, nativeMtpDepthPolicy, nativeMtpDepth, nativeMtpDisabled }
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
    // Incoming creation values are current user intent, not a persisted legacy
    // row. In particular, explicit4096 must not be mistaken for an old generic
    // default merely because it has the same numeric value.
    applyBundleStartupDefaults(config, modelPath, false)
    applyMissingCacheStackStartupDefaults(config, modelPath)
    applyFamilyStartupDefaults(config, modelPath)
    liftStaleFlatCacheIndex(config, modelPath)
    normalizeCacheStackMutualExclusion(config)
    markCacheStackStartupDefaultsCurrent(config, modelPath)

    // Reuse only the same actual directory (including genuine symlink aliases).
    // A basename is not a load identity: another drive may hold a different
    // quantization or a repaired/test copy with the exact same folder name.
    const existing =
      db.getSessionByModelPath(modelPath) ||
      db.getSessions().find(
        s => s.type !== 'remote' && sameLocalBundlePath(s.modelPath, modelPath)
      )
    if (existing) {
      // Creation reuses a bundle's session. Never rewrite a live session's
      // endpoint/config before startSession rejects its existing process: the
      // UI/gateway would advertise new settings while the old engine keeps
      // running. Active settings changes belong to Save & Restart instead.
      const managed = this.processes.get(existing.id)
      if (['running', 'loading', 'standby'].includes(existing.status) ||
          managed?.process || managed?.adoptedPid) {
        throw new Error(
          'This model already has an active session. Use its Server Settings and Save & Restart, or stop it before creating it again.'
        )
      }
      // Merge new config into existing (don't overwrite unspecified fields)
      let existingConfig: Record<string, any> = {}
      try { existingConfig = JSON.parse(existing.config || '{}') } catch (_) { }
      const host = (config.host as string) || existing.host
      const port = (config.port as number) || existing.port
      applyBundleStartupDefaults(existingConfig, modelPath)
      applyCacheStackStartupDefaultMigration(existingConfig, modelPath)
      const merged = { ...existingConfig, ...config, modelPath, host, port }
      applyBundleStartupDefaults(merged, modelPath)
      applyMissingCacheStackStartupDefaults(merged, modelPath)
      applyFamilyStartupDefaults(merged, modelPath)
      // The spread above lets an incoming renderer config re-introduce the flat
      // 1000 over a value the migration just lifted, so re-run the backstop on
      // the merged result rather than only on the incoming config.
      liftStaleFlatCacheIndex(merged, modelPath)
      normalizeCacheStackMutualExclusion(merged)
      markCacheStackStartupDefaultsCurrent(merged, modelPath)
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

  /**
   * Validate a local target before single-model replacement unloads the
   * currently healthy engine. This intentionally performs no process mutation.
   * `_startSessionInner` repeats the same shared validation immediately before
   * spawn so filesystem changes between preflight and launch still fail closed.
   */
  async preflightImageModelPath(modelPath: string): Promise<void> {
    // The Image picker replaces its previous session before createSession.
    // Validate the checkpoint before that replacement boundary, not only in
    // startSession after the previous healthy engine has already been stopped.
    const engine = this.findEnginePath()
    if (!engine) throw new Error('vmlx-engine not found. Please install it first.')
    console.log(`[IMAGE] Checking bundle integrity before replacement: ${modelPath}`)
    const report = await runModelBundleIntegrityPreflight(
      engine, modelPath, line => console.log(`[IMAGE] Bundle preflight: ${line}`),
    )
    console.log(
      `[IMAGE] Bundle integrity OK before replacement: ${modelPath}; ` +
      `${report.cache_hit ? 'one-time stamp' : 'fresh header scan'}, ${report.shards} shards`,
    )
  }

  async preflightSessionStart(sessionId: string): Promise<void> {
    const session = db.getSession(sessionId)
    if (!session) throw new Error(`Session ${sessionId} not found`)
    if (session.type === 'remote') return

    const config: ServerConfig = JSON.parse(session.config)
    config.modelPath = session.modelPath
    config.host = session.host
    config.port = session.port
    const engine = this.findEnginePath()
    if (!engine) {
      throw new Error('vmlx-engine not found. Please install it first.')
    }
    this.validateLocalSessionTarget(sessionId, session, config)
    if (existsSync(config.modelPath)) {
      const preflightLines: string[] = []
      const retainPreflightLine = (line: string) => {
        this.pushLog(sessionId, line)
        const stamped = this.logBuffers.get(sessionId)?.at(-1)
        if (stamped) preflightLines.push(stamped)
        if (preflightLines.length > SessionManager.LOG_BUFFER_MAX_LINES) preflightLines.shift()
      }
      const publishRepair = createBundleRepairProgressReporter((message, isNotice) => {
        retainPreflightLine(message.label)
        this.emit('session:log', {
          sessionId, data: message.label, labelKey: message.labelKey, labelParams: message.labelParams,
          bundleRepairNotice: isNotice,
        })
        if (!isNotice) this.emitLoadProgress({
          sessionId, ...message, progress: 0, indeterminate: true, phase: 'bundle_repair',
        })
      })
      const report = await runModelBundleIntegrityPreflight(
        engine, config.modelPath, line => {
          retainPreflightLine(line)
          publishRepair(line)
        },
      ).catch(error => {
        this.pendingBundlePreflightLogs.delete(sessionId)
        throw error
      })
      // Gateway/manual start may check the same stamped bundle twice. A
      // second no-op check must not erase the first check's repair receipt.
      if (preflightLines.length) this.pendingBundlePreflightLogs.set(sessionId, {
        modelPath: config.modelPath, lines: preflightLines,
      })
      const source = report.cache_hit ? 'one-time stamp' : 'fresh header scan'
      console.log(
        `[SESSIONS] bundle integrity OK for ${sessionId}: ${source}, ` +
        `${report.shards} shards, ${report.tensors} tensors, ` +
        `${report.misaligned_tensors} remaining misaligned tensors`,
      )
      for (const repaired of report.repairs) {
        console.log(`[SESSIONS] bundle integrity atomically repaired ${repaired}`)
      }
    }
  }

  private validateLocalSessionTarget(
    sessionId: string,
    session: Session,
    config: ServerConfig
  ): boolean {
    // Image models may use mflux named models (e.g., "schnell") that are NOT
    // filesystem paths. Let the image server validate those names at launch.
    const isImageSession = config.modelType === 'image'

    if (!isImageSession) {
      if (!existsSync(config.modelPath)) {
        // LE10: the saved path may be a stale symlink/alias (e.g. a
        // ~/.mlxstudio/models/X target that moved) while the SAME model is
        // present at a different real path (e.g. an external drive). Re-resolve
        // by model IDENTITY (basename) against other known sessions whose path
        // still exists — a real comparison, and only when exactly one distinct
        // valid path matches, never a guess. This lets Start succeed instead of
        // failing on a dead symlink when the model is plainly available.
        // When NO unambiguous twin exists, fail with the explicit repoint hint
        // (the session list surfaces the Repoint/Remove recovery actions).
        const resolved = Array.from(new Set(
          db.getSessions()
            .filter(s =>
              s.id !== sessionId &&
              sessionMatchesModelPath(s.modelPath, config.modelPath) &&
              existsSync(s.modelPath))
            .map(s => s.modelPath.replace(/\/+$/, ''))
        ))
        if (resolved.length === 1) {
          console.log(`[SESSION] modelPath missing (${config.modelPath}) — re-resolved by identity to ${resolved[0]}`)
          config.modelPath = resolved[0]
          // Persist so the UI + future starts use the valid path — but ONLY when
          // no other session already owns that path. sessions.model_path is
          // UNIQUE, and the valid path is typically owned by the very session we
          // re-resolved against, so a blind update would throw a UNIQUE
          // constraint error. In that collision case persistence is both
          // impossible and unnecessary: the identity re-resolution re-fires
          // cheaply on every start and the launch already uses the valid path.
          const pathOwnedElsewhere = db.getSessions().some(
            s => s.id !== session.id && s.modelPath?.replace(/\/+$/, '') === resolved[0],
          )
          if (!pathOwnedElsewhere) {
            try {
              db.updateSession(session.id, { modelPath: resolved[0] })
            } catch (e) {
              console.warn(`[SESSION] Failed to persist re-resolved modelPath for ${session.id}: ${e}`)
            }
          }
        } else {
          throw new Error(
            `Model not found at: ${config.modelPath}. Repoint the session to a valid model bundle before starting it.`,
          )
        }
      }

      // Block starting a session with an actively downloading model.
      const downloadMarker = join(config.modelPath, '.vmlx-downloading')
      if (existsSync(downloadMarker)) {
        throw new Error('This model is still downloading. Please wait for the download to complete before starting a session.')
      }

      // Validate model format: vmlx-engine only supports MLX (safetensors) models.
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
        // Diffusers image models have model_index.json instead of config.json — that's valid.
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
        // Ignore other filesystem errors — let the server handle them.
      }

      const metadata = validateJangBundleMetadataForLaunch(config.modelPath)
      if (!metadata.ok) {
        throw new Error(
          `Invalid model metadata. Repair or re-download this bundle before starting it: ${metadata.error}`,
        )
      }
    }

    return isImageSession
  }

  async startSession(
    sessionId: string,
    options?: {
      restartGeneration?: number
      launchOrigin?: 'manual' | 'gateway'
    },
  ): Promise<void> {
    // Captured at request entry: if an explicit Stop advances the epoch while
    // this start waits (gateway transition queue, session lock), the start is
    // superseded and must not spawn. Stop wins.
    //
    // A restart passes the generation it captured when the restart BEGAN
    // (before its own stop advanced the epoch). Capturing at our own entry
    // would be too late for that path: the restart's start is requested after
    // the user's Stop, so it would observe the post-Stop epoch and pass.
    const entryEpoch = options?.restartGeneration ?? (this.lifecycleEpochs.get(sessionId) ?? 0)

    const assertNotSuperseded = () => {
      const currentEpoch = this.lifecycleEpochs.get(sessionId) ?? 0
      if (currentEpoch !== entryEpoch) {
        console.log(
          `[SESSIONS] start of ${sessionId} superseded by an explicit stop ` +
          `(epoch ${entryEpoch} -> ${currentEpoch}); not spawning`
        )
        throw new Error('Start canceled: the session was stopped after this start was requested')
      }
    }

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

    const startLocalSession = async () => {
      // Before ANY side effect: single-model detection stops other engines and
      // adoption can mark this session running and return without ever
      // reaching the in-lock check. A superseded start must do nothing at all.
      assertNotSuperseded()
      if (isGatewaySettingEnabled(db.getSetting(GATEWAY_SINGLE_MODEL_MODE_KEY))) {
        // Validate before unloading the current model. Without this ordering a
        // stale/malformed target can strand the user with zero loaded models.
        await this.preflightSessionStart(sessionId)
        // Single-model mode is a RAM/process contract, not just a DB-state
        // contract. A prior Electron crash, gateway restart, or stale session
        // row can leave a healthy vmlx-engine alive while its DB row says
        // stopped (or points at an old port). Stop those detected engines
        // before launching the replacement, otherwise the UI can show one
        // running model while two engines are resident.
        await this.stopDetectedLocalEnginesForSingleModel(sessionId)
        // Detection awaited above — an explicit Stop may have landed while it
        // ran, and adoption below would mark this session running despite it.
        assertNotSuperseded()
        if (await this.adoptDetectedTargetProcessForStart(sessionId)) {
          return
        }
        const otherLocalSessions = db.getSessions().filter(other =>
          other.id !== sessionId &&
          other.type !== 'remote' &&
          ['running', 'loading', 'standby'].includes(other.status)
        )
        for (const other of otherLocalSessions) {
          console.log(
            `[SESSIONS] single-model mode: stopping session ${other.id} before starting ${sessionId}`
          )
          // Fail closed: if an old engine cannot unload, do not start another
          // one and silently violate the user-visible one-model RAM contract.
          await this.stopSession(other.id)
        }
      }

      // Serialize start/stop operations per session to prevent races.
      await this.withSessionLock(sessionId, () => {
        assertNotSuperseded()
        return this._startSessionInner(sessionId, options)
      })
    }

    if (!isGatewaySettingEnabled(db.getSetting(GATEWAY_SINGLE_MODEL_MODE_KEY))) {
      await startLocalSession()
      return
    }

    // Manual Start / Launch Session paths must obey the same replacement
    // ordering as gateway routing. Two concurrent starts resolve in request
    // order; the later target unloads the earlier one before it begins.
    const previous = this.singleModelStartTransitionPending.catch(() => {})
    let release: () => void = () => {}
    const current = new Promise<void>(resolve => { release = resolve })
    this.singleModelStartTransitionPending = previous.then(() => current)
    await previous
    try {
      await startLocalSession()
    } finally {
      release()
    }
  }

  /**
   * Save & Restart as ONE main-process lifecycle operation. The renderer used
   * to orchestrate update -> stop -> start over separate IPC calls; an
   * explicit user Stop landing between that pair advanced the epoch BEFORE
   * the restart's start was requested, so the start captured the post-Stop
   * epoch at entry and spawned an engine the UI no longer tracked.
   *
   * The generation is derived from the epoch read when the restart begins:
   * the restart-owned stop advances it exactly once for local sessions, so
   * any other advance — an explicit Stop landing before, during, or after
   * ours — makes this restart stale and its start must not run.
   */
  async restartSession(sessionId: string): Promise<void> {
    const session = db.getSession(sessionId)
    if (!session) throw new Error(`Session ${sessionId} not found`)
    const preStopEpoch = this.lifecycleEpochs.get(sessionId) ?? 0
    await this.stopSession(sessionId)
    // Remote stop paths do not advance the epoch (no process to spawn), so
    // their generation is the unchanged pre-stop value.
    const restartGeneration = session.type === 'remote' ? preStopEpoch : preStopEpoch + 1
    await this.startSession(sessionId, { restartGeneration })
  }

  async enforceSingleModelLocalProcessContract(
    targetSessionId: string,
    options: { preserveActiveTarget?: boolean } = {},
  ): Promise<boolean> {
    if (!isGatewaySettingEnabled(db.getSetting(GATEWAY_SINGLE_MODEL_MODE_KEY))) return false
    await this.preflightSessionStart(targetSessionId)
    await this.stopDetectedLocalEnginesForSingleModel(targetSessionId)
    if (options.preserveActiveTarget) {
      return this.adoptDetectedTargetProcessForStart(targetSessionId)
    }
    return false
  }

  private async stopDetectedLocalEnginesForSingleModel(targetSessionId: string): Promise<void> {
    const target = db.getSession(targetSessionId)
    if (!target || target.type === 'remote') return

    const targetPath = normalizePath(target.modelPath)
    const targetCanBeAdopted = canAdoptExistingLocalEngine(targetPath)
    const detected = await this.detect()
    const allSessions = db.getSessions()

    for (const proc of detected) {
      const livePath = normalizePath(proc.modelPath)
      const isHealthyTarget =
        targetCanBeAdopted &&
        proc.healthy &&
        livePath === targetPath &&
        proc.port === target.port
      if (isHealthyTarget) continue

      console.log(
        `[SESSIONS] single-model mode: stopping detected engine pid=${proc.pid} ` +
        `port=${proc.port} model=${livePath} before starting ${targetSessionId}`,
      )
      await this.terminateDetectedLocalEngine(proc, allSessions)
    }
  }

  private async terminateDetectedLocalEngine(proc: DetectedProcess, sessions = db.getSessions()): Promise<void> {
    const livePath = normalizePath(proc.modelPath)
    const owner = sessions.find(s =>
      s.type !== 'remote' &&
      (normalizePath(s.modelPath) === livePath || s.port === proc.port || s.pid === proc.pid)
    )
    if (owner) {
      markImageGenerationServerStopping(owner.id)
      try { await requestImageGenerationServerStop(owner.id) } catch (error) {
        this.pushLog(owner.id, `[WARNING] Image job cancel before Stop failed; terminating the server: ${error}`)
      }
    }
    this.killPid(proc.pid)
    await new Promise(r => setTimeout(r, 1500))
    try {
      process.kill(proc.pid, 0)
      this.killPid(proc.pid, 'SIGKILL')
    } catch (_) { }

    if (owner) {
      this.processes.delete(owner.id)
      db.updateSession(owner.id, {
        status: 'stopped',
        pid: undefined,
        lastStoppedAt: Date.now(),
        standbyDepth: null,
      })
      this.emit('session:stopped', { sessionId: owner.id })
    }
  }

  private async adoptDetectedTargetProcessForStart(sessionId: string): Promise<boolean> {
    const session = db.getSession(sessionId)
    if (!session || session.type === 'remote') return false

    const targetPath = normalizePath(session.modelPath)
    if (!canAdoptExistingLocalEngine(targetPath)) {
      console.warn(
        `[SESSIONS] Refusing to adopt an existing DSV4 engine for session ${sessionId}; ` +
        'the current Electron instance must relaunch it with source-matched executable and cache policy',
      )
      return false
    }
    const detected = await this.detect()
    const proc = detected.find(p =>
      p.healthy &&
      normalizePath(p.modelPath) === targetPath &&
      p.port === session.port
    )
    if (!proc) return false

    const status = proc.standbyDepth ? 'standby' : 'running'
    db.updateSession(session.id, {
      status,
      pid: proc.pid,
      port: proc.port,
      modelPath: targetPath,
      modelName: proc.modelName || session.modelName,
      lastStartedAt: Date.now(),
      standbyDepth: proc.standbyDepth || null,
    })
    this.processes.set(session.id, { process: null, adoptedPid: proc.pid })
    this.emitTerminalLoadProgress(session.id)
    this.emit('session:ready', { sessionId: session.id, port: proc.port, pid: proc.pid })
    return true
  }

  private async _startSessionInner(
    sessionId: string,
    options?: { launchOrigin?: 'manual' | 'gateway' },
  ): Promise<void> {
    const session = db.getSession(sessionId)
    if (!session) throw new Error(`Session ${sessionId} not found`)
    // Fresh log buffer per run — stop retains the previous buffer for
    // postmortems; preserve only the pending preflight for this exact bundle.
    this.logBuffers.delete(sessionId)
    const preflightLogs = this.pendingBundlePreflightLogs.get(sessionId)
    this.pendingBundlePreflightLogs.delete(sessionId)
    if (preflightLogs?.modelPath === session.modelPath) {
      this.logBuffers.set(sessionId, preflightLogs.lines)
    }

    const managed = this.processes.get(sessionId)
    if (managed?.process || managed?.adoptedPid) {
      throw new Error('Session is already running')
    }

    // Ordinary Save must never redirect an existing process to a future port/key.
    // Promote the saved config only here, under the start/restart lifecycle lock.
    const pendingApplied = db.applyPendingSessionConfig(sessionId)
    if (pendingApplied) Object.assign(session, pendingApplied)
    const config: ServerConfig = JSON.parse(session.config)
    // Detection may materialize an effective multimodal value for this launch,
    // but it must not turn an absent (Auto) setting into a persisted Force On
    // or Force Off override.  Capture user intent before applying defaults or
    // inspecting the current artifact.
    const hadExplicitMultimodalOverride = Object.prototype.hasOwnProperty.call(
      config,
      'isMultimodal',
    )
    const toolParserWasAuto = config.toolCallParser == null || config.toolCallParser === 'auto'
    const reasoningParserWasAuto = config.reasoningParser == null || config.reasoningParser === 'auto'
    config.modelPath = session.modelPath
    config.host = session.host
    config.port = session.port
    const bundleDefaultsChanged = applyBundleStartupDefaults(config, config.modelPath)
    const cacheDefaultsFilled = applyMissingCacheStackStartupDefaults(config, config.modelPath)
    const migrated = applyCacheStackStartupDefaultMigration(config, config.modelPath)
    const familyDefaultsChanged = applyFamilyStartupDefaults(config, config.modelPath)
    const normalized = normalizeCacheStackMutualExclusion(config)
    const markedCurrent = markCacheStackStartupDefaultsCurrent(config, config.modelPath)
    if (bundleDefaultsChanged || cacheDefaultsFilled || migrated || familyDefaultsChanged || normalized || markedCurrent) {
      // Persist the migrated config so the settings UI reflects the corrected
      // values on next render and the same migration doesn't have to re-fire
      // on every session start. Without this writeback the saved config keeps
      // showing the stale tuple even though the engine launches with the new
      // values.
      try {
        db.updateSession(session.id, { config: JSON.stringify(config) })
        console.log(`[SESSION] Persisted startup defaults for session ${session.id}`)
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
    const isImageSession = this.validateLocalSessionTarget(sessionId, session, config)

    // Re-detect model config from disk — handles case where model files were
    // replaced with a different model (same folder name, different model_type).
    // User-set overrides (port, host, apiKey, etc.) are preserved.
    let freshDetectedFamily: string | undefined
    let freshDsv4Active = normalizeDetectedFamilyName(
      resolveEffectiveModelFamily(config.modelFamily, freshDetectedFamily),
    ) === 'deepseek-v4'
    let freshDetectedConfig: ReturnType<typeof detectModelConfigFromDir> | undefined
    if (!isImageSession) {
      try {
        const freshConfig = detectModelConfigFromDir(config.modelPath)
        if (freshConfig) {
          freshDetectedConfig = freshConfig
          freshDetectedFamily = normalizeDetectedFamilyName(freshConfig.family)
          const freshFamily = freshDetectedFamily
          freshDsv4Active = normalizeDetectedFamilyName(
            resolveEffectiveModelFamily(config.modelFamily, freshDetectedFamily),
          ) === 'deepseek-v4'
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
          // keep reasoning on the registry path for text ZAYA; current ZAYA1-VL
          // plain-template bundles intentionally resolve reasoningParser to auto because
          // live proof shows their synthetic thinking rail is hidden-only.
          if (isZayaCcaFamily(freshFamily)) {
            if (config.toolCallParser !== '') {
              config.toolCallParser = freshConfig.toolParser || 'auto'
            }
            if (config.reasoningParser !== '') {
              config.reasoningParser = freshConfig.reasoningParser || 'auto'
            }
          }
          if (freshFamily === 'minimax' && config.reasoningParser !== '') {
            config.reasoningParser = freshConfig.reasoningParser || 'auto'
          }
          if (freshDsv4Active) {
            const dsv4PoolQuantDefault = freshConfig.dsv4PoolQuantDefault
            const dsv4PrefixEnabled = config.enablePrefixCache !== false
            const dsv4Changed =
              config.continuousBatching !== true ||
              config.enablePrefixCache === undefined ||
              config.usePagedCache !== false ||
              config.enableBlockDiskCache === undefined ||
              config.maxCacheBlocks === undefined ||
              config.dsv4PrefixCache !== dsv4PrefixEnabled ||
              (typeof dsv4PoolQuantDefault === 'boolean'
                ? config.dsv4PoolQuant !== dsv4PoolQuantDefault
                : config.dsv4PoolQuant !== undefined) ||
              config.dsv4ActivationQat === undefined ||
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
            if (config.enablePrefixCache === undefined) config.enablePrefixCache = true
            config.usePagedCache = false
            if (config.enableBlockDiskCache === undefined) config.enableBlockDiskCache = true
            if (config.maxCacheBlocks === undefined) config.maxCacheBlocks = DSV4_MAX_CACHE_BLOCKS
            config.dsv4PrefixCache = config.enablePrefixCache !== false
            if (typeof dsv4PoolQuantDefault === 'boolean') {
              config.dsv4PoolQuant = dsv4PoolQuantDefault
            } else {
              delete config.dsv4PoolQuant
            }
            if (config.dsv4ActivationQat === undefined) {
              config.dsv4ActivationQat = false
            }
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
              this.pushLog(
                sessionId,
                `[INFO] DSV4-Flash native cache policy: prefix=${config.enablePrefixCache !== false ? 'on' : 'off'}, paged_ram=off, block_disk_l2=${config.enableBlockDiskCache === true ? 'on' : 'off'}, block_size=${DSV4_PAGED_CACHE_BLOCK_SIZE}, generic_turboquant=off, pool_codec=bundle-derived, activation_qat=${config.dsv4ActivationQat === true ? 'on' : 'off'}`,
              )
            }
          } else if (freshFamily === 'minimax_m3') {
            const m3PrefixEnabled = config.enablePrefixCache !== false
            const m3BlockDiskEnabled = m3PrefixEnabled && config.enableBlockDiskCache !== false
            const m3Changed =
              config.enablePrefixCache === undefined ||
              config.usePagedCache !== false ||
              config.enableDiskCache !== false ||
              config.enableBlockDiskCache === undefined ||
              config.kvCacheQuantization !== 'auto' ||
              config.enableJit === true
            if (config.enablePrefixCache === undefined) config.enablePrefixCache = true
            config.usePagedCache = false
            config.enableDiskCache = false
            if (config.enableBlockDiskCache === undefined) config.enableBlockDiskCache = true
            config.kvCacheQuantization = 'auto'
            config.enableJit = false
            if (m3Changed) {
              this.pushLog(sessionId, !m3PrefixEnabled
                ? '[INFO] MiniMax-M3 detected; native typed prefix cache explicitly disabled for this session'
                : m3BlockDiskEnabled
                  ? '[INFO] MiniMax-M3 detected; using typed MSA SSD-only prefix cache with idx_keys, persistent RAM payloads disabled, generic KV quantization off, and JIT off'
                  : '[INFO] MiniMax-M3 detected; prefix cache enabled without paged RAM or block-disk L2')
            }
          } else if (freshFamily === 'glm5-next') {
            const staleGenericDiskPair =
              config.enableDiskCache !== true && config.enableBlockDiskCache === true
            const glmChanged =
              config.enablePrefixCache === undefined ||
              config.usePagedCache !== false ||
              staleGenericDiskPair ||
              config.noMemoryAwareCache !== false ||
              config.kvCacheQuantization !== 'auto'
            if (config.enablePrefixCache === undefined) config.enablePrefixCache = true
            config.usePagedCache = false
            if (staleGenericDiskPair || config.enableDiskCache === undefined) {
              config.enableDiskCache = true
            }
            if (staleGenericDiskPair || config.enableDiskCache === true) {
              config.enableBlockDiskCache = false
            }
            config.noMemoryAwareCache = false
            config.kvCacheQuantization = 'auto'
            if (glmChanged) {
              this.pushLog(sessionId, '[INFO] GLM-5.3 detected; exact full-precision typed KDA/MLA/DSA prefix + prompt-L2 cache enabled, generic paged/block/TurboQuant cache codecs disabled')
            }
          } else if (freshFamily === 'openpangu_v2') {
            const panguChanged =
              config.enablePrefixCache !== true ||
              config.usePagedCache !== false ||
              config.enableDiskCache !== true ||
              config.enableBlockDiskCache !== false ||
              config.noMemoryAwareCache !== false ||
              config.kvCacheQuantization !== 'auto' ||
              config.enableJit === true
            config.enablePrefixCache = true
            config.usePagedCache = false
            config.enableDiskCache = true
            config.enableBlockDiskCache = false
            config.noMemoryAwareCache = false
            config.kvCacheQuantization = 'auto'
            config.enableJit = false
            if (panguChanged) {
              this.pushLog(sessionId, '[INFO] openPangu detected; exact full-precision typed prefix + prompt-L2 cache enabled, generic paged/block/TurboQuant KV codecs disabled')
            }
          }
          // Repeat the all-family production normalization after detection so
          // an adopted stale session cannot put q4/q8 on the launch argv.
          if (config.kvCacheQuantization !== 'auto') {
            const priorStoredCodec = config.kvCacheQuantization
            config.kvCacheQuantization = 'auto'
            this.pushLog(
              sessionId,
              `[INFO] Saved stored-cache quantization=${priorStoredCodec} reset to auto; production prefix reuse preserves architecture-native cache state with no added codec`,
            )
          }
          // Refresh multimodal detection from disk. A detected VLM must win
          // over stale saved `isMultimodal=false` from older sessions, while a
          // forceTextOnly policy must clear stale true rows. Affine-JANG Qwen
          // hybrids now remain multimodal when the artifact has indexed vision
          // tensors and the vMLX-owned runtime is available; metadata-only or
          // explicitly text-only artifacts still land in forceTextOnly.
          // Smelt is handled later at launch time because its partial expert
          // support uses text-only loading.
          if (freshConfig.forceTextOnly === true) {
            // affine-JANG Qwen hybrid etc.: runtime forces text-only regardless of save.
            config.isMultimodal = false
          } else if (config.isMultimodal === undefined) {
            // Auto (no explicit user choice): take fresh detection.
            config.isMultimodal = freshConfig.isMultimodal
          }
          // Explicit user choice is RESPECTED: isMultimodal===true (Force On) or
          // ===false (Force Off) must survive — do NOT let detected VL clobber a
          // deliberate Force Off. buildArgs emits --text-only for a detected-VL model
          // the user forced off so the engine honors it (is_mllm_model force_text_only).
          // In-RAM paged cache is OFF for every family (SSD block-disk L2 is the
          // only tier), so a saved Off is never "stale" and is never reset to On.
          // This block used to silently flip a deliberate Off back to On on
          // re-detect, which is how paged RAM kept coming back after being
          // turned off.
          if (config.usePagedCache === true) {
            config.usePagedCache = false
            this.pushLog(sessionId, `[INFO] In-Memory Paged Cache (RAM) forced Off — SSD block-disk cache (L2) is the only cache tier`)
          }
          // Log if model type changed
          if (oldFamily && oldFamily !== 'auto' && freshConfig.toolParser && oldFamily !== freshConfig.toolParser) {
            this.pushLog(sessionId, `[INFO] Model config re-detected from disk (was: ${oldFamily}, now: ${freshConfig.toolParser})`)
          }
          if (oldReasoningParser && oldReasoningParser !== 'auto' && isZayaCcaFamily(freshFamily) && oldReasoningParser !== config.reasoningParser) {
            this.pushLog(sessionId, `[INFO] ZAYA reasoning parser reset from stale ${oldReasoningParser} to auto (no reasoning parser)`)
          }
          // Persist refreshed defaults without converting the tri-state Auto
          // selection into an explicit override.  The in-memory config keeps
          // the detected effective value for buildArgs below.
          const persistedConfig = { ...config }
          if (!hadExplicitMultimodalOverride) {
            delete persistedConfig.isMultimodal
          }
          if (toolParserWasAuto) {
            persistedConfig.toolCallParser = 'auto'
          }
          if (reasoningParserWasAuto) {
            persistedConfig.reasoningParser = 'auto'
          }
          db.updateSession(sessionId, { config: JSON.stringify(persistedConfig) })
        }
      } catch (e) {
        this.pushLog(sessionId, `[WARN] Could not re-detect model config: ${(e as Error).message}`)
      }
    }

    // Memory estimation: warn if model is too large for available RAM.
    // Residency is a property of the FAMILY's loader, not the weight format
    // (measured 2026-08-11, vmmap): ordinary loads COPY weights into dirty
    // Metal buffers — resident ≈ fileBytes + ~2 GB regardless of
    // affine/mxtq/mxfp8/plain (MM2.7 measured 0.96×). Only expert-streaming
    // families stay below file size (DSV4-Flash ~0.26×, MM3 ~0.80×). The old
    // "JANG = lazy mmap ×0.7" model under-admitted full-resident bundles by
    // ~45% while its ×1.3 arm over-refused big plain bundles. freemem() still
    // under-counts droppable page cache, so a bounded reclaimable credit is
    // applied for every launch.
    const modelFileBytes = estimateModelFileBytes(config.modelPath)
    const totalBytes = totalmem()
    const modelSizeBytes = estimateModelLaunchResidentBytes(config.modelPath, modelFileBytes, totalBytes)
    if (modelSizeBytes > 0) {
      const availableBytes = effectiveLaunchAvailableBytes(freemem(), {
        reclaimableBytes: estimateMacReclaimableMemoryBytes(),
        totalBytes,
      })
      const usagePercent = ((totalBytes - availableBytes) / totalBytes) * 100
      const modelGB = formatGb(modelSizeBytes)
      const availGB = (availableBytes / 1e9).toFixed(1)
      const totalGB = (totalBytes / 1e9).toFixed(0)
      console.log(`[SESSION] Model estimate: ~${modelGB} GB | RAM: ${availGB} GB free / ${totalGB} GB total (${usagePercent.toFixed(0)}% used)`)
      this.emit('session:log', { sessionId, data: `Model estimate: ~${modelGB} GB | RAM: ${availGB} GB free / ${totalGB} GB total\n` })
      // Admission ran on classifyLargeModelMemoryPreflight alone, whose block
      // arm needs modelSizeBytes >= 50GB AND availableBytes < 2GB AND >= 98%
      // used — by which point the machine is already dying, so in practice it
      // almost never fires. MEASURED on this box at 42.2 GB free: a 91.9 GB
      // Inkling and a 73.4 GB DSV4 both came back "warn", i.e. the app would
      // start a 92 GB model into 42 GB of free RAM and leave the outcome to the
      // OS.
      //
      // The calibrated rule already existed — unsafeModelLaunchReason refuses
      // when the estimated RESIDENT size exceeds effective free RAM, which is
      // the question that actually matters — but nothing called it, so it and
      // its override env were unreachable. Ask it first; keep the graded
      // warnings below for everything it admits.
      // Refuse on the MEASUREMENT-keyed estimate, not the conservative warning
      // bound. On 0.7, a 96 GB DSV4-Flash bundle estimated 69.2 GB and was
      // turned away on a box where it measures ~25 GB resident and runs fine.
      // The graded warnings below still use the conservative number.
      const admissionBytes = estimateModelLaunchAdmissionBytes(
        config.modelPath,
        modelFileBytes,
        totalBytes,
      )
      // 2026-08-17: this used to REFUSE the launch. It is now advisory only.
      //
      // The refusal turned users away from the exact models this app exists to
      // run. Two independent reasons it could not be trusted as a gate:
      //
      //  1. It keys off a per-family residency ratio resolved from the
      //     top-level config.json `model_type`. Any bundle that ratio does not
      //     recognise silently falls back to 1.0x, so a ~101 GB bundle
      //     estimated ~103.6 GB resident and was refused on a 128 GB box that
      //     runs it. A guard whose default is "assume the worst and block" is a
      //     guess with veto power.
      //  2. `freemem()` is not the capacity that matters on unified memory.
      //     macOS reclaims inactive, purgeable and file-cache pages on demand,
      //     so "currently free RAM" understates what a launch can actually use.
      //     The reclaimable credit below is capped at 15% of total, which does
      //     not cover a large page cache.
      //
      // The OS and the Metal allocator already fail loudly and recoverably when
      // a model genuinely does not fit. Predicting that failure badly, and
      // blocking on the prediction, is strictly worse than attempting the load.
      const admissionAdvisory = unsafeModelLaunchReason(admissionBytes, freemem(), process.env, {
        reclaimableBytes: estimateMacReclaimableMemoryBytes(),
        totalBytes,
      })
      if (admissionAdvisory) {
        const message = appendMetalWiredLimitGuidance(
          `Memory estimate: ${admissionAdvisory}. Starting anyway — this is an ` +
          `estimate, not a limit. If it does fail to load, closing other apps or ` +
          `stopping running vMLX sessions frees memory.`
        )
        console.warn(`[SESSION] ${message}`)
        this.emit('session:log', { sessionId, data: `⚠️  ${message}\n` })
      }
      const reserveWarning = modelLaunchReserveWarning(modelSizeBytes, availableBytes)
      if (reserveWarning) {
        console.warn(`[SESSION] ${reserveWarning}`)
        this.emit('session:log', { sessionId, data: `⚠️  ${reserveWarning}\n` })
      }
      // No `block` arm: classifyLargeModelMemoryPreflight can no longer return
      // one (the variant was deleted from its type on 2026-08-17). Preflight
      // advises; it never refuses.
      // Wired-limit recommendation (Eric, 2026-08-29): compare the model's
      // resident footprint to the user's EFFECTIVE Metal wired limit. Manual
      // UI starts show the exact sysctl command and explicitly wait for the
      // user's choice. API-gateway JIT starts must stay non-interactive: an
      // application-modal dialog blocks Electron's gateway and prevents the
      // engine child from spawning until someone clicks Continue. The warning
      // remains in the session log for those headless/API starts.
      try {
        let wiredLimitMb = 0
        try {
          wiredLimitMb = parseInt(
            execFileSync('sysctl', ['-n', 'iogpu.wired_limit_mb'], { timeout: 3000 })
              .toString().trim(), 10) || 0
        } catch { /* non-macOS or sysctl unavailable — default fraction used */ }
        const wiredPreflight = classifyWiredLimitPreflight({ modelSizeBytes, wiredLimitMb, totalBytes })
        if (wiredPreflight.action === 'recommend') {
          const logLine = `${wiredPreflight.message} ${wiredPreflight.detail.replace(/\n+/g, ' ')}`
          console.warn(`[SESSION] ${logLine}`)
          this.emit('session:log', { sessionId, data: `⚠️  ${logLine}\n` })
          if (options?.launchOrigin !== 'gateway') {
            // Advisory only: never await this dialog. Awaiting made the
            // application-modal sheet gate the engine spawn (and freeze the
            // whole window for automation) until someone clicked Continue -
            // the launch proceeded only after dismissal. The recommendation
            // stays visible while the load runs; Copy still works.
            const command = wiredPreflight.command
            void dialog.showMessageBox({
              type: 'warning',
              title: 'Metal wired-memory limit recommendation',
              message: wiredPreflight.message,
              detail: wiredPreflight.detail,
              buttons: ['Copy Command', 'Continue'],
              defaultId: 0,
              cancelId: 1,
            }).then((result) => {
              if (result.response === 0) clipboard.writeText(command)
            }).catch(() => { /* advisory-only path must never break a launch */ })
          }
        }
      } catch { /* advisory-only path must never break a launch */ }
      const memoryPreflight = classifyLargeModelMemoryPreflight({ modelSizeBytes, availableBytes, totalBytes })
      if (memoryPreflight.action === 'warn') {
        if (modelSizeBytes > availableBytes * 0.9) {
          console.warn(`[SESSION] WARNING: Model (~${modelGB} GB) may exceed available memory (${availGB} GB free). Risk of system instability.`)
          this.emit('session:log', { sessionId, data: `⚠️  ${memoryPreflight.message}\n` })
        } else {
          console.log(`[SESSION] Model will use most of available RAM (${modelGB} GB / ${availGB} GB free)`)
          this.emit('session:log', { sessionId, data: `${memoryPreflight.message}\n` })
        }
      }
    }

    // Never kill arbitrary processes by port. Terminate only a detected vMLX
    // engine for this exact model/session; otherwise fail closed.
    await this.ensureOwnedSessionPortAvailable(session)

    db.updateSession(sessionId, {
      status: 'loading',
      lastStartedAt: Date.now()
    })
    this.loadProgressState.delete(sessionId) // Reset loading progress for fresh start
    this.loadProgressMeta.delete(sessionId)
    this.stopLoadResidentMonitor(sessionId)
    this.stopWakeHealthPoller(sessionId)
    this.externalWakes.delete(sessionId)
    // A replacement engine process restarts its generation counter at 1; a
    // stale high-water mark here would discard EVERY event it ever sends.
    this.lifecycleGenerations.delete(sessionId)
    this.contractSessions.delete(sessionId)
    this.lastLoadProgressEvents.delete(sessionId)
    if (modelFileBytes > 0) {
      const residentProfile = launchResidentProfileForModel(config.modelPath)
      const meta = {
        modelBytes: modelFileBytes,
        expectedResidentBytes: Math.round(modelFileBytes * residentProfile.ratio),
        lazyResident: residentProfile.streamsWeights,
      }
      this.loadProgressMeta.set(sessionId, meta)
      this.loadProgressState.set(sessionId, 0)
      this.emitLoadProgress({
        sessionId,
        label: 'Scanning model files...',
        labelKey: 'main.loadProgress.scanningModelFiles',
        progress: 0,
        indeterminate: true,
        ...meta,
      })
    }
    this.emit('session:starting', { sessionId, modelPath: session.modelPath })

    const args = this.buildArgs(config)
    const pagedCapacityLine = pagedCacheCapacityLogLine(args)
    if (pagedCapacityLine) {
      this.pushLog(sessionId, pagedCapacityLine)
    }

    // Ensure PATH includes pyenv/homebrew so the engine finds its Python
    const extraPath = [
      join(homedir(), '.pyenv', 'shims'),
      join(homedir(), '.pyenv', 'bin'),
      '/opt/homebrew/bin',
      '/usr/local/bin',
    ].join(':')
    const spawnEnv: Record<string, string | undefined> = { ...process.env, PATH: `${extraPath}:${process.env.PATH || ''}` }
    // The Python CLI defaults mx.compile back ON for affine JANG bundles when
    // --enable-jit is absent. Production Auto now preserves Laguna's native
    // full/sliding cache and does not install selective TurboQuant, so cache
    // Auto must not suppress JIT. Preserve any deliberate parent-shell opt-out
    // and honor only Laguna's saved JIT-Off choice here.
    // `args` is authoritative for whether an explicit q4/q8/none cache mode
    // will reach the engine.
    const lagunaJitPolicyInput = {
      detected: freshDetectedConfig,
      kvCacheQuantization: config.kvCacheQuantization,
      explicitKvCacheQuantizationApplied: args.includes('--kv-cache-quantization'),
      enableJitRequested: !!config.enableJit,
    }
    const lagunaMixedSwaTurboQuantActive =
      isLagunaMixedSwaTurboQuantEffective(lagunaJitPolicyInput)
    const disableLagunaAffineJitDefault = applyLagunaJitDefaultEnvironment(
      spawnEnv,
      lagunaJitPolicyInput,
    )
    if (disableLagunaAffineJitDefault) {
      this.pushLog(
        sessionId,
        `[ENV] ${DISABLE_JANG_AFFINE_JIT_DEFAULT_ENV}=1 (${
          lagunaMixedSwaTurboQuantActive
            ? 'Laguna diagnostic TurboQuant cache is not mx.compile-safe'
            : 'Laguna JIT is Off; preventing the affine-JANG CLI auto-default'
        })`,
      )
    }
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
      // Normalize on READ, like the other consumers. Both UI save paths trim
      // today, so a whitespace-bearing token can only arrive from an older
      // build, a migration or a direct DB write -- but a stray newline here
      // becomes an invalid Authorization header inside the engine, which fails
      // as an auth error rather than as the formatting problem it is.
      const hfToken = normalizeHfTokenSetting(db.getSetting('hf_api_key'))
      if (hfToken) {
        spawnEnv.HF_TOKEN = hfToken
      }
    }
    // MiniMax-M3 VL route: the engine wires M3 vision through the text runtime only when
    // VMLX_M3_VL is truthy. Scope strictly to M3 so no other family's env changes.
    if (freshDetectedFamily === 'minimax_m3') {
      spawnEnv.VMLX_M3_VL = '1'
    }
    // Qwen3.5/3.6 hybrid affine-JANG VLM (for example Ornith): select the
    // vMLX-owned qwen3_5_family runtime, whose router-gate and 1D text-RoPE
    // patches replace the unsafe legacy mlx-vlm path. Scope the override to
    // these families; for a non-VL artifact the engine's media check makes it
    // a no-op. An explicit UI Force Off still emits --text-only and wins at
    // model classification.
    if (freshDetectedFamily === 'qwen3_5' || freshDetectedFamily === 'qwen3_5_moe') {
      spawnEnv.VMLX_QWEN_VL = '1'
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
    delete spawnEnv.DSV4_LONG_CTX
    delete spawnEnv.DSV4_POOL_QUANT
    delete spawnEnv.DSV4_ACTIVATION_QAT
    delete spawnEnv.VMLX_DENSE_STRICT_LANE
    delete spawnEnv.VMLX_DSV4_FAST_LOAD_DISABLE
    delete spawnEnv.VMLX_DSV4_ENABLE_PREFIX_CACHE
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
      dsv4Active: freshDsv4Active,
      dsv4PoolQuantDefault: freshDetectedConfig?.dsv4PoolQuantDefault,
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
      'DSV4_LONG_CTX',
      'DSV4_POOL_QUANT',
      'DSV4_ACTIVATION_QAT',
      'VMLX_DENSE_STRICT_LANE',
      'VMLX_DSV4_FAST_LOAD_DISABLE',
      'VMLX_DSV4_ENABLE_PREFIX_CACHE',
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
    if (engineResult.type === 'bundled' || engineResult.type === 'development') {
      // Bundled Python: spawn python3 -B -s -m vmlx_engine.cli serve <model> --host ... --port ...
      // -B: do not write __pycache__ into the signed app bundle at runtime
      // -s: suppress user site-packages (~/.local/lib/python3.12/site-packages)
      // This avoids shebang path issues with relocatable Python and ensures
      // the app uses ONLY its bundled engine, never system-installed mlx-lm/vmlx-engine.
      const developmentSourceRoot = engineResult.type === 'development'
        ? engineResult.sourceRoot
        : undefined
      const bundledEnv: Record<string, string | undefined> = {
        ...spawnEnv,
        PYTHONDONTWRITEBYTECODE: '1',
        PYTHONNOUSERSITE: '1',  // Extra safety: disable user site-packages
        // Do not prepend the launcher's cwd to sys.path. A packaged app started
        // from $HOME can otherwise treat a sibling repo directory named `mlx`
        // as a second package location and abort before model loading.
        PYTHONSAFEPATH: '1',
        // Packaged Python must not be shadowed. A development venv, however,
        // may be a symlink to a sibling checkout's dependency environment;
        // pinning the current source root is what prevents its editable .pth
        // from silently importing that sibling repository.
        PYTHONPATH: developmentSourceRoot,
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
        cwd: developmentSourceRoot || dirname(engineResult.pythonPath),
        stdio: ['ignore', 'pipe', 'pipe'],
        detached: true,  // Separate process group so we can kill entire group
      })
    } else {
      // System binary: spawn vmlx-engine directly
      const fullCmd = `${engineResult.binaryPath} ${args.join(' ')}`
      this.pushLog(sessionId, `$ ${fullCmd}`)
      this.emit('session:log', { sessionId, data: `$ ${fullCmd}\n` })
      const systemEnv = { ...spawnEnv }
      if (engineResult.sourceRoot) {
        // A development checkout may intentionally reuse dependencies from a
        // sibling venv. Pin the imported vmlx_engine package to this checkout;
        // otherwise the console script's editable install silently executes the
        // sibling repository and invalidates every current-source UI proof.
        systemEnv.PYTHONPATH = engineResult.sourceRoot
        this.pushLog(
          sessionId,
          `[SESSIONS] Development engine source: ${engineResult.sourceRoot}`,
        )
      }
      proc = spawn(engineResult.binaryPath, args, {
        env: systemEnv,
        cwd: dirname(engineResult.binaryPath),
        stdio: ['ignore', 'pipe', 'pipe'],
        detached: true,
      })
    }

    this.processes.set(sessionId, { process: proc, adoptedPid: null })

    proc.stdout?.on('data', (data) => {
      const text = data.toString()
      recordImageGenerationLog(sessionId, text, 'stdout')
      this.pushLog(sessionId, text)
      this.emit('session:log', { sessionId, data: text })
    })
    proc.stderr?.on('data', (data) => {
      const text = data.toString()
      // Consume original chunks before the display normalizer strips newlines.
      recordImageGenerationLog(sessionId, text, 'stderr')
      const managed = this.processes.get(sessionId)
      const normalized = normalizeBackendStderrChunk(
        managed?.backendStderrPending || '',
        text,
      )
      if (managed) managed.backendStderrPending = normalized.pending
      for (const event of normalized.events) {
        if (event.type === 'disconnect') {
          this.pushLog(sessionId, event.text)
          this.emit('session:log', {
            sessionId,
            data: BACKEND_STDERR_DISCONNECT_NORMALIZED_LINE,
          })
          continue
        }
        const stderrText = event.text
        this.pushLog(sessionId, stderrText)
        // Log errors to main console for diagnostics
        if (stderrText.includes('ERROR') || stderrText.includes('Traceback') || stderrText.includes('Exception')) {
          console.error(`[SERVER] ${stderrText.trimEnd()}`)
        }
        this.emit('session:log', { sessionId, data: stderrText })
        // Capture most meaningful stderr line for error reporting.
        // Python exceptions print the error type several lines before the
        // final output (e.g., RuntimeError on line N, then "library not found"
        // on line N+3). Prefer exception lines over the last line.
        if (!managed) continue
        const lines = stderrText.trim().split('\n').filter((l: string) => l.trim())
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
      const currentSession = db.getSession(sessionId)
      if (currentSession?.pid && currentSession.pid !== proc.pid) {
        console.log(
          `[SESSIONS] Ignoring stale child exit for session ${sessionId}; ` +
          `db now owns pid=${currentSession.pid}, exited pid=${proc.pid ?? 'unknown'}`,
        )
        return
      }
      if (managed && managed.process !== proc) {
        console.log(
          `[SESSIONS] Ignoring stale child exit for session ${sessionId}; ` +
          `current owner is pid=${managed.adoptedPid ?? managed.process?.pid ?? 'none'}`,
        )
        return
      }
      const lastStderr = managed?.lastStderr
      // The stop fence survives a concurrent health monitor deleting the
      // managed-process row before this exit callback runs.
      const intentional = (
        managed?.intentionalStop === true
        || this.intentionalStops.has(sessionId)
      )
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
        reason = appendMetalWiredLimitGuidance(reason)
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
      this.clearLoadProgressBookkeeping(sessionId)
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
      // Ready means serving, not resident: the settle phase keeps the bar
      // progressing until the weights are actually in RAM (display only —
      // the session is running and usable from this line on).
      this.emitTerminalLoadProgress(sessionId)
      db.updateSession(sessionId, { status: 'running' })
      this.touchSession(sessionId)  // Start idle timer from model-ready time
      this.emit('session:ready', {
        sessionId,
        port: session.port,
        ...(proc.pid ? { pid: proc.pid } : {})
      })
    } catch (err) {
      this.clearLoadProgressBookkeeping(sessionId)
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
        // Terminal progress comes from the main process on every path now —
        // the renderer no longer fabricates 100% on session:ready (that
        // fabrication is what used to end the bar before RAM was full).
        this.loadProgressState.set(session.id, 100)
        this.emitLoadProgress({ sessionId: session.id, label: 'Connected', labelKey: 'main.loadProgress.connected', progress: 100 })
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

    // A global health probe may already be in flight when visible Stop
    // terminates the backend. Keep this fence until `stopped` is durable so
    // that probe cannot race the intentional exit and rewrite it as `error`.
    markImageGenerationServerStopping(sessionId)
    this.intentionalStops.set(
      sessionId,
      (this.intentionalStops.get(sessionId) || 0) + 1,
    )

    // Advance the lifecycle epoch BEFORE queueing on the session lock so an
    // already-queued start (e.g. the second half of Save & Restart) observes
    // the stop and aborts instead of spawning after this stop completes.
    this.lifecycleEpochs.set(
      sessionId,
      (this.lifecycleEpochs.get(sessionId) ?? 0) + 1,
    )

    // Serialize start/stop operations per session to prevent races
    await this.withSessionLock(sessionId, async () => {
      this.failCounts.delete(sessionId)
      const managed = this.processes.get(sessionId)

      // Mark intentional stop on managed process to prevent crash misreport
      if (managed) managed.intentionalStop = true
      try { await requestImageGenerationServerStop(sessionId) } catch (error) {
        this.pushLog(sessionId, `[WARNING] Image job cancel before Stop failed; terminating the server: ${error}`)
      }

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
        // Ownership-scoped fallback. Never kill an unrelated app merely
        // because it happens to use this session's saved port.
        await this.terminateDetectedEngineForSession(session)
      }

      db.updateSession(sessionId, {
        status: 'stopped',
        pid: undefined,
        lastStoppedAt: Date.now(),
        standbyDepth: null
      })
      // Clear idle tracking. RETAIN the log buffer so a crash/stop
      // postmortem is possible (STOP-DESTROYS-SESSION-LOG-BUFFER): deleting
      // it here made every server death unexplainable after the fact. The
      // buffer is reset at the next start, and deleteSession still drops it.
      this.lastRequestAt.delete(sessionId)
      this.externalWakes.delete(sessionId)
      this.stopLoadResidentMonitor(sessionId)
      this.stopWakeHealthPoller(sessionId)
      this.lifecycleGenerations.delete(sessionId)
      this.contractSessions.delete(sessionId)
      this.lastLoadProgressEvents.delete(sessionId)
      this.pushLog(sessionId, '[INFO] Session stopped — log retained for postmortem until next start')
      this.emit('session:stopped', { sessionId })
    }).finally(() => {
      const remaining = (this.intentionalStops.get(sessionId) || 1) - 1
      if (remaining > 0) this.intentionalStops.set(sessionId, remaining)
      else this.intentionalStops.delete(sessionId)
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
      this.pendingBundlePreflightLogs.delete(sessionId)
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
    'enableBlockDiskCache', 'blockDiskCacheMaxGb', 'blockDiskCacheMaxPercent', 'blockDiskCacheDir',
    'prefixCacheSize', 'prefixCacheMaxBytes', 'cacheTtlMinutes', 'isMultimodal',
    'toolCallParser', 'reasoningParser',
    'dsv4PrefixCache', 'dsv4PoolQuant', 'dsv4ActivationQat',
    'maxNumSeqs', 'prefillBatchSize', 'prefillStepSize', 'completionBatchSize',
    'streamInterval', 'apiKey', 'rateLimit', 'timeout',
    // Timeout is both a per-request client/proxy deadline and a server launch
    // argument. Restart so Electron, gateway, argv, and backend agree after a
    // saved setting changes instead of leaving the old server timeout active.
    'maxTokens', 'maxContextLength', 'mcpConfig',
    'mcpEnabledServers', 'mcpDisabledServers', 'mcpEnabledTools', 'mcpDisabledTools',
    'servedModelName',
    'speculativeModel', 'numDraftTokens', 'smelt', 'smeltExperts',
    'nativeMtpMode', 'nativeMtpDepth', 'nativeMtpDepthOverride',
    'omniBackend',
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
      if (config.port !== session.port && !(await this.isPortFree(config.port))) {
        throw new Error(
          `Port ${config.port} is already in use by another application. ` +
          'vMLX will not stop or replace an unowned process.',
        )
      }
    }

    let effectiveConfig: Record<string, unknown> = {}
    try {
      effectiveConfig = JSON.parse(session.config)
    } catch {
      // Corrupted config in DB — start fresh
    }
    // Subsequent edits start from previously saved next-start settings, while
    // live transports continue reading session.config/host/port.
    const currentConfig: Record<string, unknown> = session.pendingConfig
      ? JSON.parse(session.pendingConfig)
      : effectiveConfig
    // IPC preserves own properties whose value is undefined. In the settings
    // UI that value means "Auto" for tri-state controls, so it must DELETE a
    // persisted override rather than be silently ignored. Omitted properties
    // remain untouched because Object.entries only visits properties present
    // on the submitted object.
    const cleanConfig: Record<string, unknown> = {}
    const explicitlyClearedKeys = new Set<string>()
    for (const [k, v] of Object.entries(config as Record<string, unknown>)) {
      if (v === undefined) {
        explicitlyClearedKeys.add(k)
      } else {
        cleanConfig[k] = v
      }
    }
    // Migrate the STORED baseline to the current cache-stack defaults version
    // FIRST, then layer the user's explicit edits on top so user intent wins.
    // Running the migration on the already-merged config (old order) let the
    // per-family default reset (gemma4/M3 paged-OFF, cache-memory 15%, block-L2
    // OFF) clobber the very cache fields the user just changed in this save —
    // the edits silently vanished while maxTokens (untouched by the migration)
    // persisted, so the launched engine received stale cache values. Live-found
    // via Codex Electron QA (2026-07-13): UI showed paged/L2/19%/2048 but the
    // engine launched --no-paged-cache --cache-memory-percent 0.15 with no
    // --max-tokens. A settings-save still can't stamp an un-migrated config as
    // current, because the baseline is migrated here before the merge.
    const migratedBaseline: Record<string, unknown> = { ...currentConfig }
    // Complete generation migration on the old baseline BEFORE merging an
    // explicit settings edit. Otherwise the next start clears that new value.
    applyBundleStartupDefaults(migratedBaseline as Partial<ServerConfig>, session.modelPath)
    applyCacheStackStartupDefaultMigration(migratedBaseline as Partial<ServerConfig>, (migratedBaseline.modelPath as string) || undefined)
    markCacheStackStartupDefaultsCurrent(migratedBaseline as Partial<ServerConfig>, session.modelPath)
    for (const key of explicitlyClearedKeys) delete migratedBaseline[key]
    const merged = { ...migratedBaseline, ...cleanConfig }
    normalizeCacheStackMutualExclusion(merged as Partial<ServerConfig>)
    markCacheStackStartupDefaultsCurrent(merged as Partial<ServerConfig>, session.modelPath)

    // Log sleep config changes
    if ('idleTimeoutSoftMin' in cleanConfig || 'idleTimeoutHardMin' in cleanConfig || 'autoSleepEnabled' in cleanConfig) {
      console.log(`[SLEEP] Config saved for ${sessionId.slice(0, 8)}: soft=${merged.idleTimeoutSoftMin}min, hard=${merged.idleTimeoutHardMin}min, enabled=${merged.autoSleepEnabled}`)
    }

    const savePlan = planSessionConfigSave(session, effectiveConfig, merged, SessionManager.RESTART_REQUIRED_KEYS)
    // Canonical live DB columns and config stay in sync; settings forms read
    // pendingConfig explicitly, never accidentally as an active endpoint.
    const host = (savePlan.config.host as string) || session.host
    const port = (savePlan.config.port as number) || session.port

    db.updateSession(sessionId, {
      config: JSON.stringify(savePlan.config),
      pendingConfig: savePlan.pendingConfig ? JSON.stringify(savePlan.pendingConfig) : null,
      host,
      port
    })

    const changedKeys = savePlan.changedKeys
    const updatedSession = db.getSession(sessionId)
    if (updatedSession) {
      this.emit('session:updated', {
        sessionId,
        session: updatedSession,
        changedKeys,
      })
    }
    return {
      restartRequired: savePlan.restartRequired,
      changedKeys,
    }
  }

  repointSessionModelPath(
    sessionId: string,
    candidatePath: string,
    options: { confirmIdentityChange?: boolean } = {},
  ): { session: Session; oldModelPath: string; reboundChatCount: number } {
    const session = db.getSession(sessionId)
    if (!session) throw new Error(`Session ${sessionId} not found`)
    if (session.type === 'remote') throw new Error('Remote sessions do not use local model paths.')

    const validation = validateModelBundleDirectory(candidatePath)
    if (!validation.valid) throw new Error(validation.error || 'Invalid model bundle directory.')

    const newModelPath = validation.path
    const identityChanged = !sessionMatchesModelPath(session.modelPath, newModelPath)
    if (identityChanged && !options.confirmIdentityChange) {
      throw new Error('The selected model identity differs from this session and requires confirmation.')
    }
    if (
      identityChanged &&
      (session.status === 'running' || session.status === 'loading' || session.status === 'standby')
    ) {
      throw new Error('Stop this session before repointing it to a different model.')
    }

    const pathOwner = db.getSessionByModelPath(newModelPath)
    if (pathOwner && pathOwner.id !== session.id) {
      throw new Error(
        `That model path is already used by session "${pathOwner.modelName || pathOwner.id}". Remove the stale session explicitly if the existing session should be kept.`,
      )
    }

    let config: Record<string, unknown> = {}
    try { config = JSON.parse(session.config || '{}') } catch { /* preserve a usable baseline below */ }
    config.modelPath = newModelPath

    const reboundChatCount = db.repointSessionModelPath(
      session.id,
      session.modelPath,
      {
        modelPath: newModelPath,
        modelName: basename(newModelPath) || session.modelName,
        config: JSON.stringify(config),
      },
    )
    const updated = db.getSession(session.id)!
    this.emit('session:updated', {
      sessionId: session.id,
      session: updated,
      oldModelPath: session.modelPath,
      newModelPath,
      reboundChatCount,
    })
    return { session: updated, oldModelPath: session.modelPath, reboundChatCount }
  }

  // ─── Discovery & Adoption ─────────────────────────────────────────

  async detectAndAdoptAll(): Promise<Session[]> {
    let processes = await this.detect()
    const nonAdoptableDsv4 = processes.filter(proc => !canAdoptExistingLocalEngine(proc.modelPath))
    const singleModelMode = isGatewaySettingEnabled(db.getSetting(GATEWAY_SINGLE_MODEL_MODE_KEY))
    const existingSessions = db.getSessions()
    for (const proc of nonAdoptableDsv4) {
      if (singleModelMode) {
        console.warn(
          `[SESSIONS] single-model mode: stopping non-adoptable DSV4 engine ` +
          `pid=${proc.pid} port=${proc.port}; executable provenance and ` +
          `effective native cache policy are not attested`,
        )
        await this.terminateDetectedLocalEngine(proc, existingSessions)
      } else {
        console.warn(
          `[SESSIONS] Skipping startup adoption for DSV4 engine pid=${proc.pid} ` +
          `port=${proc.port}; executable provenance and effective native cache policy are not attested`,
        )
      }
    }
    processes = processes.filter(proc => !nonAdoptableDsv4.includes(proc))
    processes = await this.pruneDetectedProcessesForSingleModel(processes)
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
        const detectedFamily = normalizeDetectedFamilyName(detected.family)
        const defaultConfig: ServerConfig = {
          modelPath: proc.modelPath,
          host: '127.0.0.1',
          port: proc.port,
          // Adopting an already-running engine used to hand-roll this as a
          // two-family ternary (deepseek-v4 / minimax_m3), so an adopted
          // openpangu_v2, qwen3.5, qwen3.5-moe, qwen3-next or nemotron-h
          // session persisted timeout=300 while the same model created through
          // the normal path persisted 900 — the same model showing two
          // different numbers in Settings depending on how the session came to
          // exist. Ask the shared table instead; it knows all seven.
          timeout: resolveSlowFamilyTimeoutSeconds(undefined, detectedFamily),
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
          // Paged RAM follows the model detector, while block SSD L2 defaults
          // on independently. For ordinary KV models with paged Off this is a
          // disk-only block-aware prefix backend; typed/path-dependent families
          // still force their detected paged contract. openPangu is the exact
          // typed exception and stays on prompt-level disk L2.
          // Paged RAM is OFF for every family, DSV4 included (SSD L2 only).
          usePagedCache: false,
          enableDiskCache: usesExactTypedPromptDiskCache(detectedFamily),
          pagedCacheBlockSize: detectedFamily === 'deepseek-v4' ? DSV4_PAGED_CACHE_BLOCK_SIZE : 64,
          // Size the index to the generic capacity target, never the old flat
          // 1000. At the 64-token generic block, 1000 indexes only 63,936
          // tokens, silently capping prefix reuse below the model's context
          // window — on Gemma 4 a 77k prompt then reported 0 cached tokens on
          // an exact repeat and ran slower than a cold prefill. The v14
          // migration already lifts existing sessions off that value; new
          // sessions must not be created at it in the first place.
          maxCacheBlocks: detectedFamily === 'deepseek-v4'
            ? DSV4_MAX_CACHE_BLOCKS
            : indexBlocksForCapacity(64),
          enableBlockDiskCache: !usesExactTypedPromptDiskCache(detectedFamily),
          // No GB cap: adopted sessions get the percent budget like everyone
          // else. This used to hardcode 10 AND stamp the defaults version
          // current, so the GB->percent migration could never reach it.
          blockDiskCacheMaxPercent: DEFAULT_BLOCK_DISK_CACHE_PERCENT,
          kvCacheQuantization: 'auto',
          cacheStackStartupDefaultsVersion: CACHE_STACK_STARTUP_DEFAULTS_VERSION,
          modelParserDefaultsVersion: MODEL_PARSER_DEFAULTS_VERSION,
          streamInterval: 8,
          maxTokens: 0,
          maxContextLength: 0,
          toolCallParser: 'auto',
          reasoningParser: 'auto',
          dsv4PrefixCache: detectedFamily === 'deepseek-v4',
          dsv4PoolQuant: detectedFamily === 'deepseek-v4'
            ? detected.dsv4PoolQuantDefault
            : undefined,
          dsv4ActivationQat: false,
          defaultEnableThinking: undefined,
          ...(() => {
            const adopted = adoptNativeMtpConfig(proc, detectedFamily, (detected as any).nativeMtp?.depth)
            return {
              nativeMtpMode: adopted.nativeMtpMode,
              nativeMtpDepth: adopted.nativeMtpDepth,
              nativeMtpDepthOverride: adopted.nativeMtpDepthOverride,
            }
          })(),
          enableAutoToolChoice: detected.enableAutoToolChoice
        }
        applyBundleStartupDefaults(defaultConfig, proc.modelPath)
        applyFamilyStartupDefaults(defaultConfig, proc.modelPath)
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

  private async pruneDetectedProcessesForSingleModel(processes: DetectedProcess[]): Promise<DetectedProcess[]> {
    if (!isGatewaySettingEnabled(db.getSetting(GATEWAY_SINGLE_MODEL_MODE_KEY))) return processes
    const healthy = processes.filter(proc => proc.healthy)
    if (healthy.length <= 1) return processes

    const sessions = db.getSessions()
    const score = (proc: DetectedProcess): [number, number, number, number] => {
      const livePath = normalizePath(proc.modelPath)
      const owner = sessions.find(s =>
        s.type !== 'remote' &&
        (s.pid === proc.pid || s.port === proc.port || normalizePath(s.modelPath) === livePath)
      )
      const active = owner && ['running', 'loading', 'standby'].includes(owner.status) ? 1 : 0
      const started = Number(owner?.lastStartedAt || 0)
      const updated = Number(owner?.updatedAt || 0)
      return [active, started, updated, proc.pid]
    }
    const better = (a: DetectedProcess, b: DetectedProcess) => {
      const sa = score(a)
      const sb = score(b)
      for (let i = 0; i < sa.length; i += 1) {
        if (sa[i] !== sb[i]) return sa[i] > sb[i]
      }
      return false
    }

    let keep = healthy[0]
    for (const proc of healthy.slice(1)) {
      if (better(proc, keep)) keep = proc
    }

    for (const proc of healthy) {
      if (proc === keep) continue
      console.log(
        `[SESSIONS] single-model mode: pruning detected engine pid=${proc.pid} ` +
        `port=${proc.port} model=${normalizePath(proc.modelPath)} during adoption; ` +
        `keeping pid=${keep.pid} port=${keep.port}`,
      )
      await this.terminateDetectedLocalEngine(proc, sessions)
    }

    return processes.filter(proc => proc === keep || !healthy.includes(proc))
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
            // Check if model was woken externally (e.g., an API request or a
            // chat message triggered the engine's JIT wake)
            try {
              const res = await fetch(
                `http://${connectHost(session.host)}:${session.port}/health`,
                { signal: AbortSignal.timeout(3000) }
              )
              if (res.ok) {
                const data = await res.json()
                if (data.status === 'healthy') {
                  // Model woke externally and finished serving-ready between
                  // monitor ticks — sync DB to running and let the settle
                  // phase carry the bar until the weights are in RAM.
                  this.externalWakes.delete(session.id)
                  db.updateSession(session.id, { status: 'running', standbyDepth: null })
                  this.emitTerminalLoadProgress(session.id)
                  this.touchSession(session.id)
                  this.emit('session:ready', {
                    sessionId: session.id,
                    port: session.port,
                    ...(session.pid ? { pid: session.pid } : {})
                  })
                  this.pushLog(session.id, '[Wake] Model woke externally — synced to running')
                } else if (data.wake_in_progress === true && !this.externalWakes.has(session.id)) {
                  // A JIT wake is reloading the model right now. Surface the
                  // same loading UI a button-triggered wake gets: status
                  // 'loading' plus the RSS resident monitor, on every surface.
                  this.externalWakes.add(session.id)
                  db.updateSession(session.id, { status: 'loading', standbyDepth: null })
                  this.beginWakeProgress(session, '[Wake] External wake detected — tracking model reload')
                  this.emit('session:starting', { sessionId: session.id })
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
            // The list read above carries no secret, so fetch this one
            // session's key. Only reached for a remote session that is
            // actually running, so idle app startup never hits the keychain.
            const remoteApiKey = db.getSession(session.id)?.remoteApiKey
            if (remoteApiKey) remoteHeaders['Authorization'] = `Bearer ${remoteApiKey}`
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
                this.emitLoadProgress({ sessionId: session.id, label: 'Connected', labelKey: 'main.loadProgress.connected', progress: 100 })
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
              await this.incrementFailAndCheck(session.id)
            }
          } catch (_) {
            // Remote server unresponsive — likely busy with inference.
            // Use dampened counting (every 3rd failure) like local sessions,
            // since remote servers have no PID to check liveness.
            this.emit('session:health', {
              sessionId: session.id,
              // EVERY exception lands here — DNS failure, connection refused,
              // timeout — not only "busy with inference", and unlike a local
              // session there is no PID to prove anything is alive. While the
              // session is still connecting, an unreachable remote must not
              // report itself running just because the request threw.
              running: session.status !== 'loading',
              busy: true,
              modelName: session.remoteModel,
              port: session.port
            })
            const count = this.failCounts.get(session.id) || 0
            if (count % 3 === 0) {
                await this.incrementFailAndCheck(session.id)
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
            if (
              isStandby &&
              session.status === 'loading' &&
              (this.wakePending.has(session.id) || data.wake_in_progress === true)
            ) {
              // A wake is still reloading the model — ours (admin/wake has
              // not returned) or an external JIT wake (health reports
              // wake_in_progress). RSS progress remains authoritative;
              // preserve Loading and its percentage.
              this.failCounts.delete(session.id)
            } else if (isStandby && session.status === 'loading') {
              // Wake failed — server reverted to standby but DB says loading.
              // Sync DB back to standby so the user can retry.
              //
              // This test MUST come before the general isStandby branch. It
              // used to sit third in the chain, after `if (isStandby)` had
              // already swallowed every standby reply, so it was unreachable:
              // a failed wake left the session pinned at status 'loading'
              // forever, with the progress bar frozen mid-fill. It never
              // errored either, because the branch it fell into deletes the
              // fail counter — so nothing could ever time the session out.
              const depth = data.status === 'standby_deep' ? 'deep' : 'soft'
              this.failCounts.delete(session.id)
              this.externalWakes.delete(session.id)
              this.stopLoadResidentMonitor(session.id)
              this.stopWakeHealthPoller(session.id)
              this.lastLoadProgressEvents.delete(session.id)
              db.updateSession(session.id, { status: 'standby', standbyDepth: depth })
              this.emit('session:standby', { sessionId: session.id, depth })
              this.pushLog(session.id, `[Wake] Model reload failed — reverted to ${depth} sleep`)
            } else if (isStandby) {
              // Server is in standby — keep session alive, don't fail-count
              this.failCounts.delete(session.id)
              if (session.status === 'running' && data.wake_in_progress !== true) {
                // The engine was put to sleep behind the panel's back (direct
                // /admin sleep call). A row that keeps saying "running" for a
                // sleeping engine blocks BOTH wake paths: chat cannot route
                // through wakeSession (it requires status standby) and the
                // external-wake detector only watches standby sessions — so a
                // message just failed the readiness poll forever.
                const depth = data.status === 'standby_deep' ? 'deep' : 'soft'
                db.updateSession(session.id, { status: 'standby', standbyDepth: depth })
                this.emit('session:standby', { sessionId: session.id, depth })
                this.pushLog(session.id, `[Sleep] Engine entered ${depth} sleep externally — synced`)
              }
            } else if (modelReady) {
              // Reset fail counter on success
              this.failCounts.delete(session.id)
              this.lastHealthyAt.set(session.id, Date.now())
              if (data.model_name && data.model_name !== session.modelName) {
                db.updateSession(session.id, { modelName: data.model_name })
              }
              if (session.status === 'loading') {
                // Server ready — keep the bar progressing through the settle
                // phase until the weights are actually resident in RAM
                // (immediate 100% when nothing is measurable).
                this.externalWakes.delete(session.id)
                this.emitTerminalLoadProgress(session.id)
                db.updateSession(session.id, { status: 'running', standbyDepth: null })
                this.touchSession(session.id)
                this.emit('session:ready', {
                  sessionId: session.id,
                  port: session.port,
                  ...(session.pid ? { pid: session.pid } : {})
                })
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
              // An actively generating engine is NOT idle — last_request_time only
              // marks submission, so a generation longer than the idle timeout
              // would get its caches deep-reset mid-stream by auto-sleep.
              const sched = data.scheduler
              if (sched && ((sched.num_running || 0) > 0 || (sched.num_waiting || 0) > 0)) {
                this.lastRequestAt.set(session.id, Date.now())
              }
            } else if (session.status === 'loading') {
              // Server is up but model not loaded yet — update progress bar
              // to show we're past server startup, now waiting for model
              const current = this.loadProgressState.get(session.id) ?? 0
              if (current < 95 && !this.contractSessions.has(session.id)) {
                // Legacy nudge only: a contract-speaking engine owns its own
                // percentage and a hard-coded 95 would jump the bar ahead.
                this.loadProgressState.set(session.id, 95)
                this.emitLoadProgress({
                  sessionId: session.id,
                  label: 'Model runtime still loading...',
                  labelKey: 'main.loadProgress.modelRuntimeStillLoading',
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
              memory: data.memory,  // { active_mb, peak_mb, cache_mb } from /health
              // Reuse the existing idle-cached health snapshot. A separate
              // cache-stats poll here would contend with active GPU decoding.
              enginePid: data.runtime_provenance?.pid,
              ssdPool: data.cache?.block_disk_cache?.global_budget,
            })
          } else {
            await this.incrementFailAndCheck(session.id)
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
              // Liveness is NOT readiness. A still-loading server has not opened
              // /health yet, so the fetch throws and the process is obviously
              // alive — reporting `running: true` there would tell the UI the
              // model is usable while it is still loading, hiding the progress
              // bar and offering Open mid-load. Consumers key on `running`.
              running: session.status !== 'loading',
              busy: true,
              modelName: session.modelName,
              port: session.port
            })
            // Only count every 3rd failure to avoid false positives
            const count = this.failCounts.get(session.id) || 0
            if (count % 3 === 0) {
              await this.incrementFailAndCheck(session.id)
            } else {
              this.failCounts.set(session.id, count + 1)
            }
          } else {
            // Process is truly dead — fast-track to marking down
            await this.incrementFailAndCheck(session.id)
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
  /** Engine refused sleep because requests are running/waiting (409 busy).
   *  Reset the idle clock so auto-sleep backs off a full idle window instead
   *  of retrying every monitor tick against a live generation. */
  private async _handleSleepBusy(sessionId: string, res: Response, depth: string): Promise<boolean> {
    if (res.status !== 409) return false
    try {
      const body = await res.json()
      if (body?.error !== 'busy') return false
      this.lastRequestAt.set(sessionId, Date.now())
      this.pushLog(
        sessionId,
        `[Sleep] ${depth} sleep deferred — engine busy (${body.num_running ?? '?'} running / ${body.num_waiting ?? '?'} waiting)`
      )
      return true
    } catch {
      return false
    }
  }

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
      if (await this._handleSleepBusy(sessionId, res, 'soft')) {
        return { success: false, error: 'Engine busy — generation in progress' }
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
      if (await this._handleSleepBusy(sessionId, res, 'deep')) {
        return { success: false, error: 'Engine busy — generation in progress' }
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

    const standbyDepth = session.standbyDepth === 'soft' ? 'soft' : 'deep'
    this.wakePending.add(sessionId)

    // admin/wake performs the reload synchronously.  Publishing "loading"
    // only after fetch() returned left the app visibly stuck at Deep Sleep for
    // the entire 30-90 second reload and started the RSS progress monitor after
    // there was nothing left to observe.  Enter loading before the request so
    // message-triggered and button-triggered wakes share the real progress UI.
    db.updateSession(sessionId, { status: 'loading', standbyDepth: null })
    this.beginWakeProgress(session, '[Wake] Waking from sleep — reloading model...')
    this.emit('session:starting', { sessionId })

    const restoreStandby = (error: string) => {
      this.wakePending.delete(sessionId)
      this.stopLoadResidentMonitor(sessionId)
      this.stopWakeHealthPoller(sessionId)
      this.loadProgressState.delete(sessionId)
      this.loadProgressMeta.delete(sessionId)
      this.lastLoadProgressEvents.delete(sessionId)
      db.updateSession(sessionId, { status: 'standby', standbyDepth })
      this.emit('session:standby', { sessionId, depth: standbyDepth })
      this.pushLog(sessionId, `[Wake] Reload failed — returned to ${standbyDepth} sleep: ${error}`)
      return { success: false, error }
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
        this.wakePending.delete(sessionId)
        // The global monitor will pick up the 'loading' status and wait for /health
        this.touchSession(sessionId)
        return { success: true }
      }
      return restoreStandby(`Server returned ${res.status}`)
    } catch (e) {
      return restoreStandby((e as Error).message)
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

      // Deep sleep is the stronger transition and may be configured earlier
      // than light sleep.  Check its deadline first: otherwise a valid
      // soft=10min / hard=1min configuration remains fully loaded until the
      // later soft deadline, then deep-sleeps only on a subsequent monitor
      // tick.
      if (session.status === 'running' && hardMs > 0 && idleMs >= hardMs) {
        console.log(`[SLEEP] Session ${session.id.slice(0, 8)} idle ${Math.round(idleMs / 1000)}s >= hard ${Math.round(hardMs / 1000)}s → deep sleep`)
        this.pushLog(session.id, `[Sleep] Idle for ${Math.round(idleMs / 60000)}min — entering deep sleep (timeout: ${Math.round(hardMs / 60000)}min)`)
        await this.deepSleep(session.id)
      } else if (session.status === 'running' && softMs > 0 && idleMs >= softMs) {
        // Running and idle past soft timeout → soft sleep
        console.log(`[SLEEP] Session ${session.id.slice(0, 8)} idle ${Math.round(idleMs / 1000)}s >= soft ${Math.round(softMs / 1000)}s → soft sleep`)
        this.pushLog(session.id, `[Sleep] Idle for ${Math.round(idleMs / 60000)}min — entering soft sleep (timeout: ${Math.round(softMs / 60000)}min)`)
        await this.softSleep(session.id)
      } else if (session.status === 'standby' && session.standbyDepth === 'soft' && hardMs > 0 && idleMs >= hardMs) {
        // In soft sleep and idle past hard timeout → deep sleep
        console.log(`[SLEEP] Session ${session.id.slice(0, 8)} idle ${Math.round(idleMs / 1000)}s >= hard ${Math.round(hardMs / 1000)}s → deep sleep`)
        this.pushLog(session.id, `[Sleep] Idle for ${Math.round(idleMs / 60000)}min — entering deep sleep (timeout: ${Math.round(hardMs / 60000)}min)`)
        await this.deepSleep(session.id)
      }
    }
  }

  private async adoptHealthyReplacementForSession(session: Session): Promise<boolean> {
    if (session.type === 'remote') return false

    const targetPath = normalizePath(session.modelPath)
    if (!canAdoptExistingLocalEngine(targetPath)) {
      console.warn(
        `[SESSIONS] Refusing to re-adopt a replacement DSV4 engine for session ${session.id}; ` +
        'its executable provenance and effective cache policy are not attested',
      )
      return false
    }
    const detected = await this.detect()
    const proc = detected.find(p =>
      p.healthy &&
      p.port === session.port &&
      (
        normalizePath(p.modelPath) === targetPath ||
        sessionMatchesModelPath(p.modelPath, targetPath)
      )
    )
    if (!proc) return false

    const status = proc.standbyDepth ? 'standby' : 'running'
    console.log(
      `[SESSIONS] Re-adopting healthy replacement pid=${proc.pid} ` +
      `for session ${session.id} on port=${proc.port} after stale pid=${session.pid}`,
    )
    db.updateSession(session.id, {
      status,
      pid: proc.pid,
      port: proc.port,
      modelPath: targetPath,
      modelName: proc.modelName || session.modelName,
      lastStartedAt: Date.now(),
      standbyDepth: proc.standbyDepth || null,
    })
    this.processes.set(session.id, { process: null, adoptedPid: proc.pid })
    this.failCounts.delete(session.id)
    if (status === 'standby') {
      this.emit('session:standby', { sessionId: session.id, depth: proc.standbyDepth })
    } else {
      this.emitTerminalLoadProgress(session.id)
      this.emit('session:ready', { sessionId: session.id, port: proc.port, pid: proc.pid })
    }
    return true
  }

  private async incrementFailAndCheck(sessionId: string): Promise<void> {
    if (
      this.intentionalStops.has(sessionId)
      || this.processes.get(sessionId)?.intentionalStop === true
    ) {
      this.failCounts.delete(sessionId)
      return
    }
    const count = (this.failCounts.get(sessionId) || 0) + 1
    this.failCounts.set(sessionId, count)

    const session = db.getSession(sessionId)

    // For local sessions: if process is dead, mark down immediately
    // For remote sessions: skip this check — they have no PID, so isProcessAlive
    // always returns false. Use the normal fail-count threshold instead.
    if (session?.type !== 'remote' && session && !this.isProcessAlive(sessionId, session.pid)) {
      if (await this.adoptHealthyReplacementForSession(session)) return
      // Adoption probes are asynchronous. A user can click Stop while one is
      // pending, so re-check the fence before classifying the dead backend.
      if (
        this.intentionalStops.has(sessionId)
        || this.processes.get(sessionId)?.intentionalStop === true
      ) {
        this.failCounts.delete(sessionId)
        return
      }
      console.log(`[SESSIONS] Process dead for session ${sessionId} (fail #${count}), marking down`)
      this.failCounts.delete(sessionId)
      this.handleSessionDown(sessionId)
      return
    }

    // Tolerate silence for at least as long as the ENGINE will. A large fresh
    // prefill blocks the engine's event loop, so /health does not answer at all
    // while it runs (measured: no response within 60s at 249k cached + 83k
    // fresh, process R at 77% CPU). The engine's streaming guard waits
    // `timeout` and then spends its unknown-progress grace windows, so this
    // used to declare the session down at 1x timeout while the engine was still
    // patiently computing for 3x.
    let maxFails = SessionManager.MAX_FAIL_COUNT
    if (session) {
      try {
        const cfg = JSON.parse(session.config)
        // NOT cfg.timeout — the slow families store 300 and are launched with
        // 900, so reading the stored value made this 3x less patient than the
        // engine for exactly the families whose prefills run longest.
        maxFails = healthFailureToleranceCount(resolvedEngineTimeoutSeconds(cfg))
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
    if (
      this.intentionalStops.has(sessionId)
      || this.processes.get(sessionId)?.intentionalStop === true
    ) {
      this.failCounts.delete(sessionId)
      return
    }
    const session = db.getSession(sessionId)
    // Abort in-flight streams FIRST, for EVERY downed session.
    //
    // This used to live inside the local branch below, which gave it two ways
    // to never run: a remote session returns before reaching it, and a session
    // already marked `error` never enters the branch at all. Either way the
    // chat's activeRequests entry outlives the engine, and chat:sendMessage
    // then rejects the user's NEXT message with "a message is already being
    // generated for this chat" — a toast that auto-dismisses, after which the
    // chat is simply dead until the 30-minute stale-lock cap expires.
    //
    // MEASURED in the dev app: a message sent into such a chat rendered in the
    // transcript and produced no reply, while the engine sat idle at 0.3% CPU.
    // A brand-new chat answered the same prompt in 4.95s.
    //
    // Safe to run unconditionally: abortByEndpoint only touches entries whose
    // endpoint matches, and aborting an already-settled request is a no-op.
    if (session?.host && session.port) {
      this.emit('session:abortInference', {
        sessionId,
        host: session.host,
        port: session.port,
      })
    }
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
          // Ownership-scoped fallback; unrelated listeners are left alone.
          this.terminateDetectedEngineForSession(session).catch(() => { })
        }
        this.processes.delete(sessionId)
      }
      this.failCounts.delete(sessionId)
      // (in-flight SSE streams were aborted above, for every downed session)
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

    // Discovery is not ownership. Another app/profile may have launched an
    // engine since our startup adoption; quitting this app must not kill it.
    // Explicitly adopted engines are already represented in this map.
    const owned = [...this.processes.entries()]
    for (const [, managed] of owned) {
      if (managed.process) {
        try { managed.process.kill('SIGTERM') } catch (_) { }
      } else if (managed.adoptedPid) {
        this.killPid(managed.adoptedPid, 'SIGTERM')
      }
    }

    // Wait for graceful shutdown, then SIGKILL any survivors
    if (owned.length > 0) {
      await new Promise(r => setTimeout(r, 3000))
      for (const [id, managed] of owned) {
        const current = this.processes.get(id)
        if (managed.process && current?.process === managed.process &&
            managed.process.exitCode == null && managed.process.signalCode == null) {
          try { managed.process.kill('SIGKILL') } catch (_) { }
        } else if (managed.adoptedPid && current?.adoptedPid === managed.adoptedPid) {
          this.killPid(managed.adoptedPid, 'SIGKILL')
        }
      }
    }

    // Do not claim an unowned or remote session stopped. Exit callbacks can
    // replace an owned entry with its terminal record while we await grace.
    const sessions = db.getSessions()
    for (const [id, managed] of owned) {
      const current = this.processes.get(id)
      if (current && current !== managed && (current.process || current.adoptedPid)) continue
      this.processes.delete(id)
      const s = sessions.find(session => session.id === id)
      if (s && s.type !== 'remote' &&
          (s.status === 'running' || s.status === 'loading' || s.status === 'standby')) {
        db.updateSession(s.id, { status: 'stopped', pid: undefined, lastStoppedAt: Date.now(), standbyDepth: null })
      }
    }
  }

  // ─── Queries ───────────────────────────────────────────────────────

  getSessions(): Array<Session & SessionModelPathClassification> {
    return classifySessionModelPaths(db.getSessions())
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
    const detectedFamily = normalizeDetectedFamilyName(detected.family)
    const effectiveFamily = normalizeDetectedFamilyName(
      resolveEffectiveModelFamily(config.modelFamily, detectedFamily),
    )

    // Server settings — always pass explicitly (both text and image)
    args.push('--host', config.host)
    args.push('--port', config.port.toString())
    args.push('--timeout', resolvedEngineTimeoutSeconds(config).toString())

    const rateLimit = finitePositiveInteger(config.rateLimit)
    if (rateLimit != null) args.push('--rate-limit', rateLimit.toString())
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
      const imageQuantize = finitePositiveInteger(config.imageQuantize)
      if (imageQuantize != null) args.push('--image-quantize', imageQuantize.toString())
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
    const dsv4Active = effectiveFamily === 'deepseek-v4'
    const m3Active = effectiveFamily === 'minimax_m3'

    // Concurrent processing
    // When value is 0 ("No limit" in UI), omit the flag so backend uses its default.
    // When value > 0, pass it explicitly to override the backend default.
    const effectiveMaxNumSeqs = dsv4Active ? 1 : finitePositiveInteger(config.maxNumSeqs)
    if (dsv4Active && config.maxNumSeqs && config.maxNumSeqs !== 1) {
      console.log(`[SESSION] DSV4-Flash detected: overriding maxNumSeqs ${config.maxNumSeqs} -> 1 (DSV4BatchGenerator is single-batch only)`)
    }
    if (effectiveMaxNumSeqs && effectiveMaxNumSeqs > 0) {
      args.push('--max-num-seqs', effectiveMaxNumSeqs.toString())
    }
    const prefillBatchSize = finitePositiveInteger(config.prefillBatchSize)
    if (!dsv4Active && prefillBatchSize != null) {
      args.push('--prefill-batch-size', prefillBatchSize.toString())
    }
    const prefillStepSize = finitePositiveInteger(config.prefillStepSize)
    // DSV4BatchGenerator is single-batch, so its batch-size controls remain
    // fixed at one. Its prefill *step* is a real bounded-memory control,
    // however, and must reach the CLI exactly as shown in the UI.
    if (prefillStepSize != null) {
      args.push('--prefill-step-size', prefillStepSize.toString())
    }
    const completionBatchSize = finitePositiveInteger(config.completionBatchSize)
    if (!dsv4Active && completionBatchSize != null) {
      args.push('--completion-batch-size', completionBatchSize.toString())
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
    // User explicitly toggled multimodal OFF (Force Off) — must beat detected VL.
    const userForceTextOnly = config.isMultimodal === false
    // Nemotron Omni advertises attachments through isMultimodal, but its
    // Parakeet/RADIO inputs are injected by the engine's Omni dispatcher into
    // the text decoder.  It is not a generic mlx_vlm --is-mllm route.
    const omniBackendActive = detectedFamily === 'nemotron-h' &&
      detected.isMultimodal === true &&
      !userForceTextOnly &&
      !detected.forceTextOnly
    // MiniMax-M3 VL route: vision is handled in-engine via SingleBatchGenerator behind
    // VMLX_M3_VL=1, so M3 must emit NEITHER --is-mllm NOR --text-only. Forcing isVLM=false
    // suppresses --is-mllm (the unpublished mlx_vlm.minimax_m3_vl path that crashes), and
    // excluding m3VlRoute from the --text-only branch keeps images flowing to the engine.
    const m3VlRoute = !!detected.m3VlRoute
    const isVLM = dsv4Active || effectiveSmelt || detected.forceTextOnly || userForceTextOnly || m3VlRoute || omniBackendActive ? false
      : detected.isMultimodal ? true
        : config.isMultimodal === true ? true
          : false
    if (isVLM) {
      args.push('--is-mllm')
    } else if (!dsv4Active && !effectiveSmelt && !m3VlRoute && !omniBackendActive && detected.isMultimodal && (userForceTextOnly || detected.forceTextOnly)) {
      // Model autodetects as VL but must run TEXT-ONLY (user Force-Off, or a family
      // whose VL runtime path isn't wired). Omitting --is-mllm is NOT enough — the
      // engine re-autodetects VL from config.json. --text-only forces is_mllm_model->False.
      args.push('--text-only')
    }

    const dflash2Speculative = /dflash2/i.test(config.speculativeModel || '')
    const cacheStackActive = dsv4Active
      ? true
      : dflash2Speculative
        ? false
        : config.continuousBatching !== false
    if (cacheStackActive) {
      args.push('--continuous-batching')
    } else {
      args.push('--no-continuous-batching')
    }

    // Parser resolution: User explicit choice -> Detected config -> Fallback logic
    // Empty string "" = user explicitly chose "None" (disabled) — always respected.
    const userToolParser = config.toolCallParser
    // Empty string = user explicitly chose "None". The engine only treats the
    // LITERAL "none" as a hard opt-out; an ABSENT flag makes it auto-configure the
    // detected parser from the registry (cli.py:785,1047-1054). So map "" -> "none"
    // and emit it — otherwise the "None (disable tool parsing)" option is inert.
    const effectiveToolParser = resolveEffectiveToolParser({
      configuredParser: userToolParser,
      detectedParser: detected.toolParser,
    })
    if (
      userToolParser &&
      userToolParser !== 'auto' &&
      !canonicalizeToolParserId(userToolParser)
    ) {
      console.warn(`[SESSION] Ignoring unsupported tool parser "${userToolParser}" for CLI launch; using detected parser "${effectiveToolParser || 'none'}"`)
    }

    const effectiveAutoTool = config.enableAutoToolChoice ?? detected.enableAutoToolChoice

    const userReasoningParser = config.reasoningParser
    const effectiveReasoningParser = resolveEffectiveReasoningParser({
      configuredParser: userReasoningParser,
      detectedParser: detected.reasoningParser,
      supportsThinking: detected.supportsThinking,
    })
    if (
      userReasoningParser &&
      userReasoningParser !== 'auto' &&
      !effectiveReasoningParser
    ) {
      console.warn(`[SESSION] Ignoring unsupported reasoning parser "${userReasoningParser}" for CLI launch`)
    }

    // Pass resolved parsers directly to the CLI so backend doesn't guess.
    // Auto tool choice is a separate tri-state: its already-resolved effective
    // boolean must be true. Auto with a detected-Off contract must not be turned
    // On merely because a parser exists.
    args.push(...buildToolLaunchArgs({
      toolParser: effectiveToolParser,
      enableAutoToolChoice: effectiveAutoTool,
    }))
    if (effectiveReasoningParser) {
      args.push('--reasoning-parser', effectiveReasoningParser)
    }
    // Manual MODEL-FAMILY override: when the user forces a family in the UI,
    // pass it through so the engine bypasses autodetect for the whole process.
    // 'auto'/undefined = keep autodetection (no flag emitted).
    const userModelFamily = (config as any).modelFamily as string | undefined
    if (userModelFamily && userModelFamily !== 'auto') {
      args.push('--model-family', userModelFamily)
      console.log(`[SESSION] Manual model-family override: ${userModelFamily} (autodetect bypassed)`)
    }
    // Thinking defaults are resolved by vmlx_engine.server from the registry
    // and explicit per-request UI/API controls. Do not turn detected
    // per-model defaults into hidden server-level startup overrides.
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
    const exactTypedPromptDiskCache = usesExactTypedPromptDiskCache(detectedFamily)
    const effectivePagedCacheBlockSize = dsv4Active
      ? DSV4_PAGED_CACHE_BLOCK_SIZE
      : config.pagedCacheBlockSize
    if (dsv4Active && config.pagedCacheBlockSize !== DSV4_PAGED_CACHE_BLOCK_SIZE) {
      console.log(`[SESSION] DSV4-Flash detected: overriding pagedCacheBlockSize ${config.pagedCacheBlockSize} -> ${DSV4_PAGED_CACHE_BLOCK_SIZE} (native SWA+CSA/HCA composite cache)`)
    }
    const cacheLaunch = buildCacheLaunchArgs({
      continuousBatching: cacheStackActive,
      enablePrefixCache: config.enablePrefixCache !== false,
      // Accepted for migration compatibility only. The shared builder always
      // emits --no-paged-cache and never its positive counterpart.
      usePagedCache: false,
      enableDiskCache: !!config.enableDiskCache,
      enableBlockDiskCache: exactTypedPromptDiskCache ? false : !!config.enableBlockDiskCache,
      noMemoryAwareCache: !!config.noMemoryAwareCache,
      forceMemoryAwareCache: exactTypedPromptDiskCache || dsv4Active,
      prefixCacheSize: config.prefixCacheSize,
      prefixCacheMaxBytes: config.prefixCacheMaxBytes,
      cacheMemoryMb: config.cacheMemoryMb,
      cacheMemoryPercent: config.cacheMemoryPercent,
      cacheTtlMinutes: config.cacheTtlMinutes,
      effectivePagedCacheBlockSize,
      maxCacheBlocks: config.maxCacheBlocks,
      diskCacheDir: config.diskCacheDir,
      diskCacheMaxGb: config.diskCacheMaxGb,
      blockDiskCacheDir: config.blockDiskCacheDir,
      blockDiskCacheMaxGb: config.blockDiskCacheMaxGb,
      blockDiskCacheMaxPercent: config.blockDiskCacheMaxPercent,
    })
    const cacheLaunchPolicy = cacheLaunch.policy
    const prefixCacheOff = cacheLaunchPolicy.prefixCacheOff
    const usePagedCache = cacheLaunchPolicy.effectiveUsePagedCache
    args.push(...cacheLaunch.args)
    if (dsv4Active) {
      console.log(`[SESSION] DSV4-Flash native cache policy: prefix=${prefixCacheOff ? 'off' : 'on'}, paged_ram=${usePagedCache ? 'on' : 'off'}, block_disk_l2=${cacheLaunchPolicy.enableBlockDiskCache ? 'on' : 'off'}, block_size=${DSV4_PAGED_CACHE_BLOCK_SIZE}, generic_turboquant=off, pool_codec=bundle-derived`)
    }

    // Production Electron sessions never add a generic stored-cache codec.
    // The engine persists each architecture's native cache representation.

    // Performance
    // Historical rows were lifted once by the owning database migration.
    // Preserve current explicit user values exactly so Settings and argv agree.
    const streamInterval = finitePositiveInteger(config.streamInterval)
    if (streamInterval != null) {
      args.push('--stream-interval', streamInterval.toString())
    }
    // maxTokens: 0/unset = no session-level output override. Let the server
    // resolve explicit request > bundle max_new_tokens > engine fallback.
    const maxTokens = finitePositiveInteger(config.maxTokens)
    if (maxTokens != null) {
      args.push('--max-tokens', maxTokens.toString())
    }
    const maxContextLength = finitePositiveInteger(config.maxContextLength)
    if (maxContextLength != null) {
      args.push('--max-prompt-tokens', maxContextLength.toString())
    }
    // Tool integration (parsers and --enable-auto-tool-choice already pushed above)
    if (config.mcpConfig) args.push('--mcp-config', config.mcpConfig)
    args.push(...buildMcpPolicyArgs(config))

    const requestedDistributed = !!(config as any).distributedEnabled
    const requestedFlashMoe = !!(config as any).flashMoe
    const zayaCcaActive = isZayaCcaFamily(detectedFamily)
    const hybridCacheActive = cacheTypeRequiresPaged(detected.cacheType)
    const turboQuantActive = !!(detected as any).isTurboQuant
    const lagunaMixedSwaTurboQuantActive = isLagunaMixedSwaTurboQuantEffective({
      detected,
      kvCacheQuantization: config.kvCacheQuantization,
      explicitKvCacheQuantizationApplied:
        !prefixCacheOff &&
        !!config.kvCacheQuantization &&
        config.kvCacheQuantization !== 'auto',
    })
    const effectiveDistributed = requestedDistributed && !dsv4Active
    const effectiveFlashMoe = requestedFlashMoe && !effectiveDistributed && !dsv4Active
    const effectiveEnableJit = computeEffectiveJit({
      enableJitRequested: !!config.enableJit,
      isMultimodal: isVLM,
      flashMoeActive: effectiveFlashMoe,
      distributedActive: effectiveDistributed,
      dsv4Active,
      m3Active,
      zayaCcaActive,
      turboQuantActive,
      lagunaMixedSwaTurboQuantActive,
      hybridCacheActive,
    })
    if (dsv4Active && ((config as any).smelt || requestedFlashMoe || requestedDistributed || config.speculativeModel)) {
      console.warn('[SESSION] DSV4-Flash detected: ignoring stale Smelt/Flash MoE/distributed/speculative flags; native DSV4 cache and expert hydration own this runtime')
    }
    if (requestedFlashMoe && !effectiveFlashMoe) {
      console.warn(`[SESSION] Ignoring stale Flash MoE flag because ${dsv4Active ? 'DSV4-Flash is active' : 'distributed mode is active'}`)
    }
    if (config.enableJit && !effectiveEnableJit) {
      const reason = dsv4Active
        ? 'DeepSeek-V4 full-model tracing is unsafe for native SWA+CSA/HCA state; native compiled router/SwiGLU and fused Metal mHC decode remain automatic'
        : m3Active
        ? 'MiniMax-M3 uses native MSA idx_keys and must stay on the uncompiled scheduler path'
        : zayaCcaActive
        ? 'ZAYA typed CCA cache is path-dependent and benchmarks faster on the uncompiled scheduler path'
        : isVLM
        ? 'multimodal/VLM models use the mlx-vlm streaming path, which is not mx.compile safe'
        : turboQuantActive
        ? 'TurboQuantKVCache uses custom cache objects that mx.compile cannot trace'
        : lagunaMixedSwaTurboQuantActive
        ? 'Laguna Auto cache mode uses selective TurboQuantKVCache for full-attention slots, which mx.compile cannot trace'
        : hybridCacheActive
        ? 'hybrid SSM/Mamba cache uses path-dependent Python cache objects that mx.compile cannot trace'
        : 'Flash MoE or distributed mode is active'
      console.warn(`[SESSION] Ignoring stale JIT flag because ${reason}`)
    }

    // Smelt mode (partial expert loading)
    if (effectiveSmelt) {
      args.push('--smelt')
      const pct = finitePositiveInteger((config as any).smeltExperts) ?? 50
      if (pct !== 50) {
        args.push('--smelt-experts', pct.toString())
      }
    }

    // Flash MoE (SSD expert streaming) — mutually exclusive with smelt/distributed/JIT.
    // Always pass the tunable values when Flash MoE is on so CLI reflects UI exactly
    // (no stale equality-with-default guard that drifts when DEFAULT_CONFIG changes).
    if (effectiveFlashMoe) {
      args.push('--flash-moe')
      const slotBank = finitePositiveInteger((config as any).flashMoeSlotBank)
      if (slotBank != null) {
        args.push('--flash-moe-slot-bank', slotBank.toString())
      }
      const prefetch = (config as any).flashMoePrefetch
      if (prefetch && prefetch !== 'none') {
        args.push('--flash-moe-prefetch', prefetch)
      }
      const ioSplit = finitePositiveInteger((config as any).flashMoeIoSplit)
      if (ioSplit != null) {
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
    const loopedNanbeige = detected.family === 'nanbeige' || detected.architectureHints?.cacheSchema === 'looped_kv_v1'
    const compatibleExternalSpeculative = !!externalSpeculativeModel && (
      dflash2Speculative
        ? !dsv4Active && isVLM
        : !dsv4Active && !isVLM && !cacheStackActive && !loopedNanbeige
    )
    if (externalSpeculativeModel && !compatibleExternalSpeculative) {
      const reason = dsv4Active
        ? 'DSV4-Flash has a native composite-cache runtime'
        : isVLM && !dflash2Speculative
        ? 'multimodal/VLM generation has no external draft verifier path'
        : cacheStackActive
        ? 'continuous batching is active'
        : loopedNanbeige
        ? 'Nanbeige has 44 looped KV slots for 22 shared layers'
        : 'this runtime does not support external draft decoding'
      console.warn(`[SESSION] Ignoring stale speculative model because ${reason}`)
    }
    if (compatibleExternalSpeculative) {
      args.push('--speculative-model', externalSpeculativeModel)
      const numDraftTokens = finitePositiveInteger(config.numDraftTokens)
      if (numDraftTokens != null && numDraftTokens !== 3) {
        args.push('--num-draft-tokens', numDraftTokens.toString())
      }
    }

    // Native in-model MTP. This is separate from external speculative decoding:
    // Qwen preserved-MTP bundles carry their own draft head. Auto preserves the
    // request/bundle sampler and uses rejection-sampling verification for
    // stochastic requests. Deterministic is the explicit greedy fast path.
    // Off restores ordinary AR with the same request/bundle sampler.
    const nativeMtp = (detected as any).nativeMtp
    if (!dsv4Active && nativeMtp?.supported) {
      const mode = (config as any).nativeMtpMode || 'auto'
      if (compatibleExternalSpeculative) {
        // An external drafter and the bundle's own MTP heads are two
        // speculative decoders bidding for the same decode step. Shipping both
        // is what made decode rates jump around on identical requests -- the
        // same code prompt measured 31.6, 44.5 and 45.3 t/s back to back with
        // a DFlash2 drafter and native MTP depth 3 both on the command line.
        // The drafter is the explicit per-session choice, so it wins, and MTP
        // is turned off LOUDLY rather than left to resolve its own depth from
        // the bundle underneath it.
        console.warn(
          `[SESSION] Native MTP disabled because an external speculative ` +
            `model is selected (${externalSpeculativeModel}). The two cannot ` +
            `share a decode step.`,
        )
      }
      args.push(...buildNativeMtpLaunchArgs({
        supported: true,
        detectedDepth: nativeMtp.depth,
        configuredDepth: (config as any).nativeMtpDepth,
        depthOverride: (config as any).nativeMtpDepthOverride === true,
        mode,
        modelDefaultMode: nativeMtp.defaultMode,
        externalSpeculativeActive: compatibleExternalSpeculative,
      }))
      // Auto emits `compatible-only`; Deterministic emits `greedy-only`.
      // Depth remains independently adaptive unless the user selects Fixed.
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

    // JIT compilation. Emitting nothing when the user turns JIT off is NOT the
    // same as off: the engine turns JIT on by itself for JANG-affine bundles
    // whenever --enable-jit is absent, so an unchecked box left JIT running and
    // the toggle was inert on exactly the models it mattered for. --no-jit is
    // the explicit off the engine now honours last.
    if (effectiveEnableJit) args.push('--enable-jit')
    else args.push('--no-jit')

    // Nemotron-Omni multimodal backend selector. Default stage1 (correct).
    // stage2 = native MLX RADIO + Parakeet, ~15-21x faster encoders + 82
    // tok/s decode on M4 Max — the JANGQ-AI banner numbers. User flips
    // this from the panel's "Omni Backend" select in the Server tab.
    if (omniBackendActive && (config as any).omniBackend && (config as any).omniBackend !== 'stage1') {
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
        dsv4Active ? DSV4_ADDITIONAL_ARG_BLOCKLIST : TEXT_ADDITIONAL_ARG_BLOCKLIST,
      )
      if (filtered.length) args.push(...filtered)
    }

    return args
  }

  findEnginePath(): EnginePath | null {
    const developmentSourceRoot = getDevelopmentSourceRoot() || undefined
    const systemEnginePath = (binaryPath: string): EnginePath => ({
      type: 'system',
      binaryPath,
      ...(developmentSourceRoot ? { sourceRoot: developmentSourceRoot } : {}),
    })

    const findProjectVenvEngine = (): EnginePath | null => {
      const projectVenv = getDevelopmentProjectVenv(developmentSourceRoot || null)
      if (!projectVenv) return null
      console.log(
        `[SESSIONS] Using project venv: ${projectVenv.pythonPath} ` +
        `(vmlx_engine ${projectVenv.version})`,
      )
      return {
        type: 'development',
        pythonPath: projectVenv.pythonPath,
        sourceRoot: developmentSourceRoot!,
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
    const locations = ENGINE_SEARCH_DIRS.flatMap((dir) =>
      ENGINE_ENTRY_POINT_NAMES.map((name) => join(dir, name)),
    )

    // Scan pyenv versions (common on macOS)
    const pyenvRoot = join(home, '.pyenv', 'versions')
    try {
      if (existsSync(pyenvRoot)) {
        for (const ver of readdirSync(pyenvRoot)) {
          for (const name of ENGINE_ENTRY_POINT_NAMES) {
            locations.push(join(pyenvRoot, ver, 'bin', name))
          }
        }
      }
    } catch (_) { }

    for (const loc of locations) {
      if (existsSync(loc)) return systemEnginePath(loc)
    }

    // Fallback: check PATH via login shell (picks up pyenv, nvm, etc.)
    for (const shell of ['/bin/zsh', '/bin/bash']) {
      for (const name of ENGINE_ENTRY_POINT_NAMES) {
        try {
          const result = execFileSync(
            shell,
            ['-lc', `which ${name}`],
            { encoding: 'utf-8', timeout: 5000 },
          ).trim()
          if (result && existsSync(result)) return systemEnginePath(result)
        } catch (_) { }
      }
    }

    // Last resort: plain which
    for (const name of ENGINE_ENTRY_POINT_NAMES) {
      try {
        const result = execFileSync('which', [name], { encoding: 'utf-8', timeout: 3000 }).trim()
        if (result && existsSync(result)) return systemEnginePath(result)
      } catch (_) { }
    }

    return null
  }

  /**
   * Return a port that is absent from both persisted sessions and the live OS
   * listener table. The renderer uses this for its initial form value so a
   * launch does not fail just because an unrelated launchd service owns the
   * next numerically-unused session port.
   */
  async getAvailablePort(): Promise<number> {
    return this.findAvailablePort()
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
    const managed = this.processes.get(sessionId)
    // DB pid is the freshest durable truth after a restart/adoption, while
    // this.processes can still briefly hold the previous ChildProcess until its
    // exit callback drains. Treat any current candidate as alive instead of
    // letting one stale managed pid mark a healthy replacement down.
    const candidates = [
      dbPid,
      managed?.adoptedPid,
      managed?.process?.pid,
    ].filter((pid): pid is number => typeof pid === 'number' && pid > 0)

    for (const pid of [...new Set(candidates)]) {
      try {
        process.kill(pid, 0) // Signal 0: doesn't kill, just checks existence
      } catch (_) {
        continue
      }
      // M7: kill(pid, 0) succeeds for zombies. Check process state to filter them out.
      try {
        const state = execFileSync('ps', ['-o', 'state=', '-p', String(pid)],
          { timeout: 1000 }).toString().trim()
        if (!state.startsWith('Z')) return true
      } catch (_) {
        // ps failed — process may have exited between checks
      }
    }
    return false
  }

  private killPid(pid: number, signal: NodeJS.Signals = 'SIGTERM'): void {
    // Try process group kill first (negative PID kills entire group).
    // This ensures MCP subprocesses and uvicorn workers are also killed.
    // Falls back to single-PID kill if group kill fails (e.g., not a group leader).
    try { process.kill(-pid, signal) } catch (_) {
      try { process.kill(pid, signal) } catch (_) { }
    }
  }

  private async terminateDetectedEngineForSession(session: Session): Promise<boolean> {
    const targetPath = normalizePath(session.modelPath)
    const detected = await this.detect()
    const proc = detected.find(candidate =>
      candidate.port === session.port &&
      normalizePath(candidate.modelPath) === targetPath
    )
    if (!proc) return false
    await this.terminateDetectedLocalEngine(proc)
    return true
  }

  private async ensureOwnedSessionPortAvailable(session: Session): Promise<void> {
    if (await this.isPortFree(session.port)) return
    const terminated = await this.terminateDetectedEngineForSession(session)
    if (terminated && await this.isPortFree(session.port)) return
    throw new Error(
      `Port ${session.port} is already in use by another application. ` +
      'vMLX did not terminate the unowned process; choose another port.',
    )
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

    while (true) {
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
          reason = appendMetalWiredLimitGuidance(reason)
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

      const now = Date.now()
      const lastProgressAt = sessionId
        ? this.loadProgressMeta.get(sessionId)?.lastStartupProgressAt
        : undefined
      const lastProgressAgeMs = lastProgressAt == null
        ? undefined
        : now - lastProgressAt
      if (!shouldContinueStartupWait(now - startTime, maxWait, lastProgressAgeMs)) {
        throw new Error('Server failed to start within timeout period')
      }
      await new Promise(resolve => setTimeout(resolve, 500))
    }
  }
}

export const sessionManager = new SessionManager()
