/**
 * Settings Flow Tests — verifies that ALL SessionConfig fields produce correct CLI flags
 * and that no settings are hardcoded. Tests use buildCommandPreview() which mirrors
 * the actual buildArgs() logic in sessions.ts exactly.
 *
 * Coverage: SessionConfig fields, context size detection, parser resolution,
 * VLM mode, cache feature gating, batching parameters, speculative decoding,
 * generation defaults, and embedding model.
 */
import { readFileSync } from 'node:fs'
import { resolve } from 'node:path'
import { describe, it, expect } from 'vitest'
import { resolveCacheLaunchPolicy } from '../src/shared/cacheControlPolicy'
import { buildMcpPolicyArgs } from '../src/shared/mcpPolicy'
import { canonicalizeToolParserId } from '../src/shared/toolParserAliases'
import { canonicalizeReasoningParserForCli } from '../src/shared/reasoningParserAliases'

// ─── SessionConfig replica (from SessionConfigForm.tsx) ──────────────────────

interface SessionConfig {
    host: string
    port: number
    apiKey: string
    rateLimit: number
    timeout: number
    maxNumSeqs: number
    prefillBatchSize: number
    prefillStepSize: number
    completionBatchSize: number
    continuousBatching: boolean
    enablePrefixCache: boolean
    prefixCacheSize: number
    prefixCacheMaxBytes: number
    cacheMemoryMb: number
    cacheMemoryPercent: number
    cacheTtlMinutes: number
    noMemoryAwareCache: boolean
    usePagedCache: boolean
    pagedCacheBlockSize: number
    maxCacheBlocks: number
    kvCacheQuantization: string
    kvCacheGroupSize: number
    omniBackend: 'stage1' | 'stage2'
    enableDiskCache: boolean
    diskCacheMaxGb: number
    diskCacheDir: string
    enableBlockDiskCache: boolean
    blockDiskCacheMaxGb: number
    blockDiskCacheDir: string
    streamInterval: number
    maxTokens: number
    mcpConfig: string
    mcpEnabledServers: string
    mcpDisabledServers: string
    mcpEnabledTools: string
    mcpDisabledTools: string
    enableAutoToolChoice?: boolean
    toolCallParser: string
    reasoningParser: string
    isMultimodal?: boolean
    servedModelName: string
    speculativeModel: string
    numDraftTokens: number
    smelt: boolean
    smeltExperts: number
    flashMoe: boolean
    flashMoeSlotBank: number
    flashMoePrefetch: 'none' | 'temporal'
    flashMoeIoSplit: number
    defaultTemperature: number
    defaultTopP: number
    defaultTopK?: number
    defaultMinP?: number
    defaultRepetitionPenalty: number
    defaultMaxNewTokens?: number
    defaultEnableThinking?: boolean
    dsv4PrefixCache?: boolean
    nativeMtpMode?: 'deterministic' | 'auto' | 'off'
    nativeMtpDepth?: number
    nativeMtpDepthOverride?: boolean
    embeddingModel: string
    additionalArgs: string
    enableJit: boolean
    logLevel: string
    corsOrigins: string
    maxContextLength: number
}

const DEFAULT_CONFIG: SessionConfig = {
    host: '127.0.0.1',
    port: 8000,
    apiKey: '',
    rateLimit: 0,
    timeout: 300,
    maxNumSeqs: 1,
    prefillBatchSize: 512,
    prefillStepSize: 2048,
    completionBatchSize: 512,
    continuousBatching: true,
    enablePrefixCache: true,
    prefixCacheSize: 100,
    prefixCacheMaxBytes: 0,
    cacheMemoryMb: 0,
    cacheMemoryPercent: 15,
    cacheTtlMinutes: 0,
    noMemoryAwareCache: false,
    usePagedCache: true,
    pagedCacheBlockSize: 64,
    maxCacheBlocks: 1000,
    kvCacheQuantization: 'auto',
    kvCacheGroupSize: 64,
    omniBackend: 'stage1',
    enableDiskCache: true,
    diskCacheMaxGb: 10,
    diskCacheDir: '',
    enableBlockDiskCache: true,
    blockDiskCacheMaxGb: 10,
    blockDiskCacheDir: '',
    streamInterval: 1,
    maxTokens: 0,
    mcpConfig: '',
    mcpEnabledServers: '',
    mcpDisabledServers: '',
    mcpEnabledTools: '',
    mcpDisabledTools: '',
    // enableAutoToolChoice intentionally omitted (undefined = auto-detect)
    toolCallParser: 'auto',
    reasoningParser: 'auto',
    isMultimodal: undefined,
    servedModelName: '',
    speculativeModel: '',
    numDraftTokens: 3,
    smelt: false,
    smeltExperts: 50,
    flashMoe: false,
    flashMoeSlotBank: 256,
    flashMoePrefetch: 'none',
    flashMoeIoSplit: 4,
    defaultTemperature: 0,
    defaultTopP: 0,
    defaultTopK: 0,
    defaultMinP: 0,
    defaultRepetitionPenalty: 0,
    defaultMaxNewTokens: 0,
    defaultEnableThinking: undefined,
    dsv4PrefixCache: false,
    nativeMtpMode: 'deterministic',
    nativeMtpDepth: 3,
    nativeMtpDepthOverride: false,
    embeddingModel: '',
    additionalArgs: '',
    enableJit: true,
    logLevel: 'INFO',
    corsOrigins: '*',
    maxContextLength: 0
}

// ─── buildCommandPreview (extracted from SessionSettings.tsx) ─────────────────
// This MUST mirror sessions.ts buildArgs() exactly.

type DetectedConfig = {
    toolParser?: string
    reasoningParser?: string
    isMultimodal?: boolean
    forceTextOnly?: boolean
    usePagedCache?: boolean
    enableAutoToolChoice?: boolean
    defaultEnableThinking?: boolean
    cacheType?: string
    family?: string
    isTurboQuant?: boolean
    nativeMtp?: {
        supported?: boolean
        depth?: number
        depthSource?: string
        runtimeScope?: string
        requiresDeterministicSampling?: boolean
    }
} | null

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

const ADDITIONAL_ARG_VALUE_FLAGS = new Set([
    '--allowed-origins',
    '--api-key',
    '--default-temperature',
    '--default-top-p',
    '--default-top-k',
    '--default-min-p',
    '--default-repetition-penalty',
    '--default-enable-thinking',
    '--log-level',
    '--max-tokens',
    '--max-prompt-tokens',
    '--native-mtp-depth',
    '--native-mtp-sampling-policy',
])

const IMAGE_ADDITIONAL_ARG_BLOCKLIST = new Set([
    '--image-mode',
    '--image-quantize',
    '--served-model-name',
    '--mflux-class',
])

const DSV4_ADDITIONAL_ARG_BLOCKLIST = new Set([
    '--dsv4-enable-prefix-cache',
    '--native-mtp-depth',
    '--native-mtp-sampling-policy',
    '--disable-native-mtp',
    '--default-temperature',
    '--default-top-p',
    '--default-top-k',
    '--default-min-p',
    '--default-repetition-penalty',
    '--default-enable-thinking',
    '--max-tokens',
    '--max-prompt-tokens',
    '--log-level',
])

function effectiveSessionTimeoutSeconds(config: Partial<SessionConfig>, family?: string): number {
    const configured = config.timeout
    if (configured != null && configured <= 0) return 86400
    const normalizedFamily = normalizeDetectedFamilyName(family)
    if (normalizedFamily === 'deepseek-v4' && (configured == null || configured === GENERIC_DEFAULT_TIMEOUT_SECONDS)) {
        return DSV4_DEFAULT_TIMEOUT_SECONDS
    }
    return configured != null && configured > 0 ? configured : GENERIC_DEFAULT_TIMEOUT_SECONDS
}

function finitePositiveNumber(value: unknown): number | undefined {
    return typeof value === 'number' && Number.isFinite(value) && value > 0 ? value : undefined
}

function finiteNonNegativeNumber(value: unknown): number | undefined {
    return typeof value === 'number' && Number.isFinite(value) && value >= 0 ? value : undefined
}

function finitePositiveInteger(value: unknown): number | undefined {
    const number = finitePositiveNumber(value)
    return number == null ? undefined : Math.max(1, Math.floor(number))
}

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

function buildCommandPreview(
    modelPath: string,
    config: SessionConfig,
    detected?: DetectedConfig
): string {
    const parts = ['vmlx-engine serve', modelPath]
    const detectedFamily = normalizeDetectedFamilyName(detected?.family)
    const turboQuantActive = !!detected?.isTurboQuant
    const dsv4Active = detectedFamily === 'deepseek-v4'
    const dsv4PrefixCacheOptIn = dsv4Active && config.dsv4PrefixCache === true
    const isVLM = dsv4Active || detected?.forceTextOnly ? false : (!!detected?.isMultimodal || config.isMultimodal === true)
    const zayaCcaActive = isZayaCcaFamily(detectedFamily)

    parts.push('--host', config.host)
    parts.push('--port', config.port.toString())
    parts.push('--timeout', effectiveSessionTimeoutSeconds(config, detectedFamily).toString())

    if (config.apiKey) parts.push('# VLLM_API_KEY=*** (env var)')
    const rateLimit = finitePositiveInteger(config.rateLimit)
    if (rateLimit != null) parts.push('--rate-limit', rateLimit.toString())

    const effectiveMaxNumSeqs = dsv4Active ? 1 : finitePositiveInteger(config.maxNumSeqs)
    if (effectiveMaxNumSeqs && effectiveMaxNumSeqs > 0) parts.push('--max-num-seqs', effectiveMaxNumSeqs.toString())
    const prefillBatchSize = finitePositiveInteger(config.prefillBatchSize)
    if (!dsv4Active && prefillBatchSize != null) parts.push('--prefill-batch-size', prefillBatchSize.toString())
    const prefillStepSize = finitePositiveInteger(config.prefillStepSize)
    if (!dsv4Active && prefillStepSize != null) parts.push('--prefill-step-size', prefillStepSize.toString())
    const completionBatchSize = finitePositiveInteger(config.completionBatchSize)
    if (!dsv4Active && completionBatchSize != null) parts.push('--completion-batch-size', completionBatchSize.toString())

    if (isVLM) parts.push('--is-mllm')
    const cacheStackActive = dsv4Active ? true : config.continuousBatching !== false
    if (cacheStackActive) parts.push('--continuous-batching')
    else parts.push('--no-continuous-batching')

    // Parser resolution: User explicit choice -> Detected config -> Fallback
    // (mirrors buildArgs: user choice wins over detection)
    const effectiveToolParser = config.toolCallParser === ''
        ? undefined
        : canonicalizeToolParserId(config.toolCallParser && config.toolCallParser !== 'auto' ? config.toolCallParser
            : detected?.toolParser)
    const effectiveAutoTool = config.enableAutoToolChoice ?? detected?.enableAutoToolChoice
    const requestedReasoningParser = config.reasoningParser === ''
        ? undefined
        : (config.reasoningParser && config.reasoningParser !== 'auto' ? config.reasoningParser
            : detected?.reasoningParser)
    const effectiveReasoningParser = canonicalizeReasoningParserForCli(requestedReasoningParser)

    const cacheLaunchPolicy = resolveCacheLaunchPolicy({
        continuousBatching: cacheStackActive,
        enablePrefixCache: dsv4Active
            ? dsv4PrefixCacheOptIn && config.enablePrefixCache !== false
            : config.enablePrefixCache !== false,
        usePagedCache: dsv4Active
            ? dsv4PrefixCacheOptIn
            : config.usePagedCache ?? detected?.usePagedCache ?? false,
        enableDiskCache: !!config.enableDiskCache,
        enableBlockDiskCache: dsv4Active
            ? dsv4PrefixCacheOptIn && !!config.enableBlockDiskCache
            : !!config.enableBlockDiskCache,
        architectureRequiresPagedCache:
            zayaCcaActive ||
            dsv4PrefixCacheOptIn ||
            (cacheTypeRequiresPaged(detected?.cacheType) && detected?.usePagedCache === true),
    })
    const prefixCacheOff = cacheLaunchPolicy.prefixCacheOff
    const usePagedCache = cacheLaunchPolicy.effectiveUsePagedCache

    if (prefixCacheOff) {
        parts.push('--disable-prefix-cache')
    } else if (!dsv4Active) {
        if (config.noMemoryAwareCache) {
            parts.push('--no-memory-aware-cache')
            const prefixCacheSize = finitePositiveInteger(config.prefixCacheSize)
            if (prefixCacheSize != null) parts.push('--prefix-cache-size', prefixCacheSize.toString())
            const prefixCacheMaxBytes = finitePositiveInteger(config.prefixCacheMaxBytes)
            if (prefixCacheMaxBytes != null) parts.push('--prefix-cache-max-bytes', prefixCacheMaxBytes.toString())
        } else {
            const cacheMemoryMb = finitePositiveInteger(config.cacheMemoryMb)
            if (!usePagedCache && cacheMemoryMb != null) parts.push('--cache-memory-mb', cacheMemoryMb.toString())
            const cacheMemoryPercent = finitePositiveNumber(config.cacheMemoryPercent)
            if (!usePagedCache && cacheMemoryPercent != null) parts.push('--cache-memory-percent', (cacheMemoryPercent / 100).toString())
            const cacheTtlMinutes = finitePositiveNumber(config.cacheTtlMinutes)
            if (cacheTtlMinutes != null && !usePagedCache) parts.push('--cache-ttl-minutes', cacheTtlMinutes.toString())
        }
    }

    if (!prefixCacheOff && usePagedCache) {
        parts.push('--use-paged-cache')
        const effectivePagedCacheBlockSize = dsv4Active
            ? DSV4_PAGED_CACHE_BLOCK_SIZE
            : config.pagedCacheBlockSize
        const pagedCacheBlockSize = finitePositiveInteger(effectivePagedCacheBlockSize)
        if (pagedCacheBlockSize != null) parts.push('--paged-cache-block-size', pagedCacheBlockSize.toString())
        const maxCacheBlocks = finitePositiveInteger(config.maxCacheBlocks)
        if (maxCacheBlocks != null) parts.push('--max-cache-blocks', maxCacheBlocks.toString())
    }

    if (!prefixCacheOff && !dsv4Active && config.kvCacheQuantization && config.kvCacheQuantization !== 'auto') {
        parts.push('--kv-cache-quantization', config.kvCacheQuantization)
        const kvCacheGroupSize = finitePositiveInteger(config.kvCacheGroupSize)
        if (config.kvCacheQuantization !== 'none' && kvCacheGroupSize != null && kvCacheGroupSize !== 64) {
            parts.push('--kv-cache-group-size', kvCacheGroupSize.toString())
        }
    }

    if (cacheLaunchPolicy.enableLegacyDiskCache) {
        parts.push('--enable-disk-cache')
        if (config.diskCacheDir) parts.push('--disk-cache-dir', config.diskCacheDir)
        const diskCacheMaxGb = finiteNonNegativeNumber(config.diskCacheMaxGb)
        if (diskCacheMaxGb != null) parts.push('--disk-cache-max-gb', diskCacheMaxGb.toString())
    }

    if (cacheLaunchPolicy.enableBlockDiskCache) {
        parts.push('--enable-block-disk-cache')
        if (config.blockDiskCacheDir) parts.push('--block-disk-cache-dir', config.blockDiskCacheDir)
        const blockDiskCacheMaxGb = finiteNonNegativeNumber(config.blockDiskCacheMaxGb)
        if (blockDiskCacheMaxGb != null) parts.push('--block-disk-cache-max-gb', blockDiskCacheMaxGb.toString())
    }

    const streamInterval = finitePositiveInteger(config.streamInterval)
    if (streamInterval != null) parts.push('--stream-interval', streamInterval.toString())
    const maxTokens = finitePositiveInteger(config.maxTokens)
    if (maxTokens != null) {
        parts.push('--max-tokens', maxTokens.toString())
    }
    // Pass resolved parsers directly (mirrors buildArgs lines 1139-1150)
    if (effectiveToolParser) {
        parts.push('--tool-call-parser', effectiveToolParser)
        if (effectiveAutoTool || config.enableAutoToolChoice === undefined) {
            parts.push('--enable-auto-tool-choice')
        }
    } else if (effectiveAutoTool) {
        parts.push('--enable-auto-tool-choice')
    }
    if (effectiveReasoningParser) parts.push('--reasoning-parser', effectiveReasoningParser)
    if (dsv4PrefixCacheOptIn) parts.push('--dsv4-enable-prefix-cache')

    if (config.mcpConfig) parts.push('--mcp-config', config.mcpConfig)
    parts.push(...buildMcpPolicyArgs(config))

    if (config.servedModelName) parts.push('--served-model-name', config.servedModelName)

    // Speculative decoding
    if (!dsv4Active && config.speculativeModel) {
        parts.push('--speculative-model', config.speculativeModel)
        const numDraftTokens = finitePositiveInteger(config.numDraftTokens)
        if (numDraftTokens != null && numDraftTokens !== 3) {
            parts.push('--num-draft-tokens', numDraftTokens.toString())
        }
    }

    if (!dsv4Active && detected?.nativeMtp?.supported) {
        const mode = config.nativeMtpMode || 'deterministic'
        if (mode === 'off') {
            parts.push('--disable-native-mtp')
        } else {
            const configuredDepth = config.nativeMtpDepthOverride === true
                ? config.nativeMtpDepth
                : detected.nativeMtp.depth
            const depth = Math.max(1, Math.min(3, finitePositiveInteger(configuredDepth) || finitePositiveInteger(detected.nativeMtp.depth) || 3))
            parts.push('--native-mtp-depth', String(depth))
            parts.push('--native-mtp-sampling-policy', mode === 'deterministic' ? 'deterministic-defaults' : 'compatible-only')
        }
    }

    // Generation defaults are resolved inside vmlx_engine.server from
    // jang_config/generation_config. The panel preview must not synthesize
    // --default-* flags that would override the engine's bundle lookup.

    // Embedding model
    if (config.embeddingModel) parts.push('--embedding-model', config.embeddingModel)

    // Thinking defaults are engine/model-owned. Chat/API requests carry
    // explicit enable_thinking; startup preview must not emit a server default.

    // JIT compilation
    if (config.enableJit && !isVLM && !dsv4Active && !zayaCcaActive && !turboQuantActive) parts.push('--enable-jit')

    if (config.omniBackend && config.omniBackend !== 'stage1') {
        parts.push('--omni-backend', config.omniBackend)
    }

    // Logging
    if (config.logLevel && config.logLevel !== 'INFO') parts.push('--log-level', config.logLevel)

    // CORS
    if (config.corsOrigins && config.corsOrigins !== '*') parts.push('--allowed-origins', config.corsOrigins)

    const maxContextLength = finitePositiveInteger(config.maxContextLength)
    if (maxContextLength != null) parts.push('--max-prompt-tokens', maxContextLength.toString())

    if (config.additionalArgs?.trim()) {
        const filtered = filterAdditionalArgs(
            config.additionalArgs,
            dsv4Active ? DSV4_ADDITIONAL_ARG_BLOCKLIST : IMAGE_ADDITIONAL_ARG_BLOCKLIST,
        )
        parts.push(...filtered)
    }

    return parts.join(' \\\n  ')
}

// ─── Helper ──────────────────────────────────────────────────────────────────

function preview(overrides: Partial<SessionConfig> = {}, detected?: DetectedConfig): string {
    return buildCommandPreview('/models/test-model', { ...DEFAULT_CONFIG, ...overrides }, detected)
}

function hasFlag(output: string, flag: string): boolean {
    // Normalize line continuations: "foo \\\n  bar" → "foo bar"
    const normalized = output.replace(/\s*\\\n\s*/g, ' ')
    return normalized.includes(flag)
}

function getFlagValue(output: string, flag: string): string | undefined {
    // Normalize line continuations: "foo \\\n  bar" → "foo bar"
    const normalized = output.replace(/\s*\\\n\s*/g, ' ')
    const idx = normalized.indexOf(flag)
    if (idx === -1) return undefined
    const rest = normalized.slice(idx + flag.length)
    const match = rest.match(/\s+(\S+)/)
    return match?.[1]
}

function expectNoInvalidNumericFlagValues(output: string) {
    const normalized = output.replace(/\s*\\\n\s*/g, ' ')
    expect(normalized).not.toMatch(/\s(?:NaN|Infinity|-Infinity)(?:\s|$)/)
}

// ─── Tests ───────────────────────────────────────────────────────────────────

describe('Server Settings', () => {
    it('sets host from config', () => {
        const out = preview({ host: '0.0.0.0' })
        expect(getFlagValue(out, '--host')).toBe('0.0.0.0')
    })

    it('sets port from config', () => {
        const out = preview({ port: 9999 })
        expect(getFlagValue(out, '--port')).toBe('9999')
    })

    it('sets timeout from config', () => {
        const out = preview({ timeout: 600 })
        expect(getFlagValue(out, '--timeout')).toBe('600')
    })

    it('uses 86400 for timeout when 0 (unlimited)', () => {
        const out = preview({ timeout: 0 })
        expect(getFlagValue(out, '--timeout')).toBe('86400')
    })

    it('deepseek-v4 uses a 900s default timeout for long reasoning turns', () => {
        const out = preview({ timeout: 300 }, { family: 'deepseek-v4' })
        expect(getFlagValue(out, '--timeout')).toBe('900')
    })

    it('deepseek-v4 preserves explicit non-default timeout values', () => {
        const out = preview({ timeout: 600 }, { family: 'deepseek-v4' })
        expect(getFlagValue(out, '--timeout')).toBe('600')
    })

    it('includes API key comment when set', () => {
        const out = preview({ apiKey: 'sk-test' })
        expect(hasFlag(out, 'VLLM_API_KEY=***')).toBe(true)
    })

    it('omits API key when empty', () => {
        const out = preview({ apiKey: '' })
        expect(hasFlag(out, 'VLLM_API_KEY')).toBe(false)
    })

    it('sets rate limit when > 0', () => {
        const out = preview({ rateLimit: 120 })
        expect(getFlagValue(out, '--rate-limit')).toBe('120')
    })

    it('omits rate limit when 0', () => {
        const out = preview({ rateLimit: 0 })
        expect(hasFlag(out, '--rate-limit')).toBe(false)
    })
})

describe('Concurrent Processing', () => {
    it('sets max-num-seqs from config', () => {
        const out = preview({ maxNumSeqs: 64 })
        expect(getFlagValue(out, '--max-num-seqs')).toBe('64')
    })

    it('sets prefill-batch-size from config', () => {
        const out = preview({ prefillBatchSize: 256 })
        expect(getFlagValue(out, '--prefill-batch-size')).toBe('256')
    })

    it('sets prefill-step-size from config', () => {
        const out = preview({ prefillStepSize: 1536 })
        expect(getFlagValue(out, '--prefill-step-size')).toBe('1536')
    })

    it('sets completion-batch-size from config', () => {
        const out = preview({ completionBatchSize: 128 })
        expect(getFlagValue(out, '--completion-batch-size')).toBe('128')
    })

    it('includes --continuous-batching when enabled (LLM)', () => {
        const out = preview({ continuousBatching: true })
        expect(hasFlag(out, '--continuous-batching')).toBe(true)
    })
})

describe('VLM Mode', () => {
    it('uses --is-mllm when isMultimodal=true', () => {
        const out = preview({ isMultimodal: true })
        expect(hasFlag(out, '--is-mllm')).toBe(true)
    })

    it('VLM gets --continuous-batching for BatchedEngine with MLLMScheduler', () => {
        const out = preview({ isMultimodal: true, continuousBatching: true })
        expect(hasFlag(out, '--is-mllm')).toBe(true)
        expect(hasFlag(out, '--continuous-batching')).toBe(true)
    })

    it('VLM continuous batching off emits explicit opt-out and suppresses cache stack', () => {
        const out = preview({ isMultimodal: true, continuousBatching: false, enablePrefixCache: true })
        expect(hasFlag(out, '--is-mllm')).toBe(true)
        expect(hasFlag(out, '--continuous-batching')).toBe(false)
        expect(hasFlag(out, '--no-continuous-batching')).toBe(true)
        expect(hasFlag(out, '--disable-prefix-cache')).toBe(true)
        expect(hasFlag(out, '--use-paged-cache')).toBe(false)
        expect(hasFlag(out, '--enable-block-disk-cache')).toBe(false)
    })

    it('detects VLM from model config', () => {
        const out = preview({}, { isMultimodal: true })
        expect(hasFlag(out, '--is-mllm')).toBe(true)
    })

    it('auto-detected VLM wins over stale isMultimodal=false', () => {
        const out = preview({ isMultimodal: false }, { isMultimodal: true })
        expect(hasFlag(out, '--is-mllm')).toBe(true)
    })

    it('manual isMultimodal=false is respected when detection is not VLM', () => {
        const out = preview({ isMultimodal: false }, { isMultimodal: false })
        expect(hasFlag(out, '--is-mllm')).toBe(false)
    })

    it('forceTextOnly detection wins over stale forced multimodal settings', () => {
        const out = preview({ isMultimodal: true }, { isMultimodal: false, forceTextOnly: true })
        expect(hasFlag(out, '--is-mllm')).toBe(false)
    })
})

describe('Prefix Cache', () => {
    it('disables prefix cache when enablePrefixCache=false', () => {
        const out = preview({ enablePrefixCache: false })
        expect(hasFlag(out, '--disable-prefix-cache')).toBe(true)
    })

    it('honors prefix cache off even when tools are configured', () => {
        const out = preview({ enablePrefixCache: false, enableAutoToolChoice: true, mcpConfig: '/path/mcp.json' })
        expect(hasFlag(out, '--disable-prefix-cache')).toBe(true)
    })

    it('does not contain a hidden tool-driven prefix cache override', () => {
        const source = readFileSync('src/main/sessions.ts', 'utf-8')
        expect(source).not.toContain('toolsNeedCache')
        expect(source).not.toContain('force prefix cache ON')
    })

    it('legacy mode: sets --no-memory-aware-cache and --prefix-cache-size', () => {
        const out = preview({ enablePrefixCache: true, noMemoryAwareCache: true, prefixCacheSize: 500 })
        expect(hasFlag(out, '--no-memory-aware-cache')).toBe(true)
        expect(getFlagValue(out, '--prefix-cache-size')).toBe('500')
    })

    it('memory-aware mode: sets --cache-memory-mb', () => {
        const out = preview({ enablePrefixCache: true, cacheMemoryMb: 4096, usePagedCache: false })
        expect(getFlagValue(out, '--cache-memory-mb')).toBe('4096')
    })

    it('session launch emits cache memory mb from a single call site', () => {
        const source = readFileSync('src/main/sessions.ts', 'utf-8')
        const matches = source.match(/args\.push\('--cache-memory-mb'/g) ?? []
        expect(matches).toHaveLength(1)
    })

    it('launch memory admission is warning-only for lazy-mmap bundles', () => {
        const source = readFileSync('src/main/sessions.ts', 'utf-8')
        const start = source.indexOf('// Memory estimation: warn if model is too large')
        const end = source.indexOf('// Kill anything on this port first')
        const block = source.slice(start, end)

        expect(start).toBeGreaterThanOrEqual(0)
        expect(end).toBeGreaterThan(start)
        expect(block).toContain('Model estimate')
        expect(block).toContain('Memory warning')
        expect(block).not.toContain('Launch blocked')
        expect(block).not.toContain('ALLOW_UNSAFE_MODEL_LAUNCH')
        expect(block).not.toContain("status: 'failed'")
        expect(block).not.toContain('throw new Error')
    })

    it('memory-aware mode: sets --cache-memory-percent as fraction', () => {
        const out = preview({ enablePrefixCache: true, cacheMemoryPercent: 30, usePagedCache: false })
        expect(getFlagValue(out, '--cache-memory-percent')).toBe('0.3')
    })

    it('sets --cache-ttl-minutes when > 0 and paged cache off', () => {
        const out = preview({ enablePrefixCache: true, cacheTtlMinutes: 60, usePagedCache: false })
        expect(getFlagValue(out, '--cache-ttl-minutes')).toBe('60')
    })

    it('suppresses --cache-ttl-minutes when paged cache is on', () => {
        const out = preview({ enablePrefixCache: true, cacheTtlMinutes: 60, usePagedCache: true })
        expect(hasFlag(out, '--cache-ttl-minutes')).toBe(false)
    })
})

describe('Paged KV Cache', () => {
    it('includes --use-paged-cache when enabled', () => {
        const out = preview({ enablePrefixCache: true, usePagedCache: true })
        expect(hasFlag(out, '--use-paged-cache')).toBe(true)
    })

    it('sets block size from config', () => {
        const out = preview({ enablePrefixCache: true, usePagedCache: true, pagedCacheBlockSize: 128 })
        expect(getFlagValue(out, '--paged-cache-block-size')).toBe('128')
    })

    it('sets max cache blocks from config', () => {
        const out = preview({ enablePrefixCache: true, usePagedCache: true, maxCacheBlocks: 2000 })
        expect(getFlagValue(out, '--max-cache-blocks')).toBe('2000')
    })

    it('omits paged cache when prefix cache is off', () => {
        const out = preview({ enablePrefixCache: false, usePagedCache: true })
        expect(hasFlag(out, '--use-paged-cache')).toBe(false)
    })

    it('paged cache from detected config', () => {
        const out = preview({ enablePrefixCache: true, usePagedCache: false }, { usePagedCache: true })
        // When config explicitly sets false, config wins over detected
        expect(hasFlag(out, '--use-paged-cache')).toBe(false)
    })

    it('ZAYA typed CCA forces paged cache when prefix cache is enabled', () => {
        const out = preview(
            {
                enablePrefixCache: true,
                usePagedCache: false,
                enableBlockDiskCache: true,
                cacheMemoryPercent: 30,
            },
            { family: 'zaya', usePagedCache: false },
        )

        expect(hasFlag(out, '--disable-prefix-cache')).toBe(false)
        expect(hasFlag(out, '--use-paged-cache')).toBe(true)
        expect(hasFlag(out, '--enable-block-disk-cache')).toBe(true)
        expect(hasFlag(out, '--cache-memory-percent')).toBe(false)
    })

    it('ZAYA typed CCA still honors explicit prefix-cache off', () => {
        const out = preview(
            {
                enablePrefixCache: false,
                usePagedCache: false,
                enableBlockDiskCache: false,
            },
            { family: 'zaya1-vl', usePagedCache: false },
        )

        expect(hasFlag(out, '--disable-prefix-cache')).toBe(true)
        expect(hasFlag(out, '--use-paged-cache')).toBe(false)
        expect(hasFlag(out, '--enable-block-disk-cache')).toBe(false)
    })
})

describe('KV Cache Quantization', () => {
    it('sets q8 quantization', () => {
        const out = preview({ enablePrefixCache: true, kvCacheQuantization: 'q8' })
        expect(getFlagValue(out, '--kv-cache-quantization')).toBe('q8')
    })

    it('sets q4 quantization', () => {
        const out = preview({ enablePrefixCache: true, kvCacheQuantization: 'q4' })
        expect(getFlagValue(out, '--kv-cache-quantization')).toBe('q4')
    })

    it('omits quantization in auto mode', () => {
        const out = preview({ enablePrefixCache: true, kvCacheQuantization: 'auto' })
        expect(hasFlag(out, '--kv-cache-quantization')).toBe(false)
    })

    it('passes explicit none to disable auto quantization', () => {
        const out = preview({ enablePrefixCache: true, kvCacheQuantization: 'none' })
        expect(getFlagValue(out, '--kv-cache-quantization')).toBe('none')
    })

    it('panel copy does not claim TurboQuant is always on', () => {
        const fs = require('fs')
        const source = fs.readFileSync(
            'src/renderer/src/components/sessions/SessionConfigForm.tsx',
            'utf-8',
        )

        expect(source).toContain('Engine-selected native cache')
        expect(source).toContain('Generic TurboQuant KV is disabled unless a tested override exists')
        expect(source).not.toContain('ON · Default')
        expect(source).not.toContain('TurboQuant only')
    })

    it('chat reasoning Auto copy avoids force-language', () => {
        const fs = require('fs')
        const locale = JSON.parse(
            fs.readFileSync('src/renderer/src/i18n/locales/en.json', 'utf-8'),
        )
        const help = locale.chat.settings.thinkingHelp

        expect(help).toContain('local vMLX')
        expect(help).toContain('model/runtime reasoning default')
        expect(help).toContain('request thinking')
        expect(help).toContain('Off')
        expect(help).not.toContain('force thinking')
        expect(help).not.toContain("others don't")
    })

    it('casual mode keeps cache codec on auto', () => {
        const fs = require('fs')
        const source = fs.readFileSync(
            'src/renderer/src/components/sessions/SessionConfigForm.tsx',
            'utf-8',
        )
        const casualStart = source.indexOf('export const CASUAL_CONFIG')
        const casualEnd = source.indexOf('interface SessionConfigFormProps', casualStart)
        const casualBlock = source.slice(casualStart, casualEnd)

        expect(casualBlock).toContain("kvCacheQuantization: 'auto'")
        expect(casualBlock).not.toContain("kvCacheQuantization: 'q4'")
    })

    it('cache panels surface native cache and TQ-KV status separately', () => {
        const fs = require('fs')
        const cachePanel = fs.readFileSync(
            'src/renderer/src/components/sessions/CachePanel.tsx',
            'utf-8',
        )
        const perfPanel = fs.readFileSync(
            'src/renderer/src/components/sessions/PerformancePanel.tsx',
            'utf-8',
        )

        expect(cachePanel).toContain('stats?.native_cache')
        expect(cachePanel).toContain('stats?.turboquant_kv_cache')
        expect(cachePanel).toContain('generic_turboquant_kv')
        expect(cachePanel).toContain('attention_kv_storage_quantization')
        expect(cachePanel).toContain('Attention KV L2')
        expect(cachePanel).toContain('SSM Policy')
        expect(cachePanel).toContain('single_sequence_only')
        expect(cachePanel).toContain('effective_max_num_seqs')
        expect(cachePanel).toContain('Cache Reuse Skips')
        expect(cachePanel).toContain('last_cache_reuse_skip')
        expect(cachePanel).toContain('needed_mb')
        expect(cachePanel).toContain('available_mb')
        expect(perfPanel).toContain('native_cache?:')
        expect(perfPanel).toContain('turboquant_kv_cache?:')
        expect(perfPanel).toContain('quantization?:')
        expect(perfPanel).toContain('acceleration?:')
        expect(perfPanel).toContain('mtp?:')
        expect(perfPanel).toContain('Weight Codec')
        expect(perfPanel).toContain('Metal NA')
        expect(perfPanel).toContain('MTP')
        expect(perfPanel).toContain('artifact_available')
        expect(perfPanel).toContain('runtime_available')
        expect(perfPanel).toContain('runtime_active')
        expect(perfPanel).toContain('runtime_reason')
        expect(perfPanel).toContain('effective_depth')
        expect(perfPanel).toContain('MTP Depth')
        expect(perfPanel).toContain('runtime_scope')
        expect(perfPanel).toContain('vl_runtime_available')
        expect(perfPanel).toContain('mtp_tensor_count')
        expect(perfPanel).toContain('vision_tensor_count')
        expect(perfPanel).toContain('MTP Scope')
        expect(perfPanel).toContain('MTP Tensors')
        expect(perfPanel).toContain('last_native_mtp')
        expect(perfPanel).toContain('MTP Accept')
        expect(perfPanel).toContain('MTP Depth Rates')
        expect(perfPanel).toContain('MTP Forwards')
        expect(perfPanel).toContain('MTP Timing')
        expect(perfPanel).toContain('effective_depth_source')
        expect(perfPanel).toContain('last_native_mtp_skip')
        expect(perfPanel).toContain('MTP Skip')
        expect(perfPanel).toContain('MTP Last')
        expect(perfPanel).toContain('weights present; runtime ready')
        expect(perfPanel).toContain('weights present; runtime unwired')
        expect(perfPanel).toContain('not used by JANGTQ')
        expect(perfPanel).toContain('Generic TQ-KV')
        expect(perfPanel).toContain('attention_kv_storage_quantization')
        expect(perfPanel).toContain('Attention KV L2')
        expect(perfPanel).toContain('SSM Policy')
        expect(perfPanel).toContain('Cache Stack')
        expect(perfPanel).toContain('Cache Components')
        expect(perfPanel).toContain('single_sequence_only')
        expect(perfPanel).toContain('scheduler?:')
        expect(perfPanel).toContain('Queue')
        expect(perfPanel).toContain('TTFT EWMA')
        expect(perfPanel).toContain('Cache Skips')
    })

    it('sets custom group size', () => {
        const out = preview({ enablePrefixCache: true, kvCacheQuantization: 'q8', kvCacheGroupSize: 32 })
        expect(getFlagValue(out, '--kv-cache-group-size')).toBe('32')
    })

    it('omits default group size 64', () => {
        const out = preview({ enablePrefixCache: true, kvCacheQuantization: 'q8', kvCacheGroupSize: 64 })
        expect(hasFlag(out, '--kv-cache-group-size')).toBe(false)
    })

    it('omits KV quant when prefix cache is off', () => {
        const out = preview({ enablePrefixCache: false, kvCacheQuantization: 'q8' })
        expect(hasFlag(out, '--kv-cache-quantization')).toBe(false)
    })
})

describe('Disk Cache', () => {
    it('enables disk cache', () => {
        const out = preview({ enablePrefixCache: true, enableDiskCache: true, usePagedCache: false })
        expect(hasFlag(out, '--enable-disk-cache')).toBe(true)
    })

    it('sets disk cache dir', () => {
        const out = preview({ enablePrefixCache: true, enableDiskCache: true, usePagedCache: false, diskCacheDir: '/tmp/cache' })
        expect(getFlagValue(out, '--disk-cache-dir')).toBe('/tmp/cache')
    })

    it('sets disk cache max gb', () => {
        const out = preview({ enablePrefixCache: true, enableDiskCache: true, usePagedCache: false, diskCacheMaxGb: 50 })
        expect(getFlagValue(out, '--disk-cache-max-gb')).toBe('50')
    })

    it('omits legacy disk cache when the disk toggle is off', () => {
        const out = preview({
            enablePrefixCache: false,
            usePagedCache: false,
            enableDiskCache: false,
            enableBlockDiskCache: false,
        })
        expect(hasFlag(out, '--enable-disk-cache')).toBe(false)
    })

    it('prefix cache off suppresses stale legacy disk cache at launch', () => {
        const out = preview({
            enablePrefixCache: false,
            usePagedCache: false,
            enableDiskCache: true,
            enableBlockDiskCache: false,
        })

        expect(hasFlag(out, '--disable-prefix-cache')).toBe(true)
        expect(hasFlag(out, '--enable-disk-cache')).toBe(false)
        expect(hasFlag(out, '--use-paged-cache')).toBe(false)
        expect(hasFlag(out, '--enable-block-disk-cache')).toBe(false)
    })

    it('session cache controls use shared policy so stale paged state cannot grey out disk cache', () => {
        const source = readFileSync('src/renderer/src/components/sessions/SessionConfigForm.tsx', 'utf8')
        expect(source).toContain('resolveCacheControlPolicy')
        expect(source).toContain('cacheControlUpdatesForDiskToggle')
        expect(source).toContain('cacheControlUpdatesForPagedToggle')
        expect(source).toContain('cacheControlUpdatesForBlockDiskToggle')
        expect(source).toContain('disabled={!dsv4Active && cachePolicy.pagedCacheDisabled}')
        expect(source).toContain('disabled={cachePolicy.legacyDiskCacheDisabled}')
        expect(source).toContain('checked={cachePolicy.legacyDiskCacheChecked}')
        expect(source).not.toContain('disabled={batchingOff || prefixOff || zayaTypedCacheRequiresPaged || dsv4CompositeRequiresPaged}')
        expect(source).not.toContain('disabled={batchingOff || prefixOff || effectiveUsePagedCache}')
    })
})

describe('Block Disk Cache', () => {
    it('enables block disk cache', () => {
        const out = preview({ enablePrefixCache: true, usePagedCache: true, enableBlockDiskCache: true })
        expect(hasFlag(out, '--enable-block-disk-cache')).toBe(true)
    })

    it('sets block disk cache dir', () => {
        const out = preview({ enablePrefixCache: true, usePagedCache: true, enableBlockDiskCache: true, blockDiskCacheDir: '/ssd/blocks' })
        expect(getFlagValue(out, '--block-disk-cache-dir')).toBe('/ssd/blocks')
    })

    it('sets block disk cache max gb', () => {
        const out = preview({ enablePrefixCache: true, usePagedCache: true, enableBlockDiskCache: true, blockDiskCacheMaxGb: 20 })
        expect(getFlagValue(out, '--block-disk-cache-max-gb')).toBe('20')
    })

    it('omits block disk cache when the block-disk toggle is off', () => {
        const out = preview({ enablePrefixCache: true, usePagedCache: true, enableBlockDiskCache: false })
        expect(hasFlag(out, '--enable-block-disk-cache')).toBe(false)
    })

    it('prefix cache off suppresses stale block-disk cache at launch', () => {
        const out = preview({
            enablePrefixCache: false,
            usePagedCache: false,
            enableDiskCache: true,
            enableBlockDiskCache: true,
        })

        expect(hasFlag(out, '--disable-prefix-cache')).toBe(true)
        expect(hasFlag(out, '--use-paged-cache')).toBe(false)
        expect(hasFlag(out, '--enable-block-disk-cache')).toBe(false)
        expect(hasFlag(out, '--enable-disk-cache')).toBe(false)
    })
})

describe('Performance & Generation', () => {
    it('sets stream interval from config', () => {
        const out = preview({ streamInterval: 5 })
        expect(getFlagValue(out, '--stream-interval')).toBe('5')
    })

    it('sets max tokens from config', () => {
        const out = preview({ maxTokens: 8192 })
        expect(getFlagValue(out, '--max-tokens')).toBe('8192')
    })

    it('surfaces Max Output Tokens separately from Max Context Tokens', () => {
        const formSource = readFileSync(resolve(__dirname, '../src/renderer/src/components/sessions/SessionConfigForm.tsx'), 'utf8')
        const maxOutputIndex = formSource.indexOf('label="Max Output Tokens"')
        const maxContextIndex = formSource.indexOf('label="Max Context Tokens"')

        expect(maxOutputIndex).toBeGreaterThan(-1)
        expect(maxContextIndex).toBeGreaterThan(-1)
        expect(maxOutputIndex).toBeLessThan(maxContextIndex)
        expect(formSource).toContain("onChange={v => onChange('maxTokens', v)}")
        expect(formSource).toContain('maps to --max-tokens')
        expect(formSource).toContain('does not change prompt/context length')
    })

    it('persists bundle/default migration so stale 32768 sessions do not keep relaunching huge output caps', () => {
        const source = readFileSync(resolve(__dirname, '../src/main/sessions.ts'), 'utf8')
        const helper = readFileSync(resolve(__dirname, '../src/shared/sessionConfigMigrations.ts'), 'utf8')
        expect(source).toContain('function applyBundleStartupDefaults(config: Partial<ServerConfig>, modelPath?: string): boolean')
        expect(source).toContain('const bundleDefaultsChanged = applyBundleStartupDefaults(config, config.modelPath)')
        expect(source).toContain('bundleDefaultsChanged || migrated || familyDefaultsChanged || markedCurrent')
        expect(helper).toContain('32768')
    })

    it('does not synthesize a huge max tokens flag when set to 0 (model/server default)', () => {
        const out = preview({ maxTokens: 0 })
        expect(getFlagValue(out, '--max-tokens')).toBeUndefined()
        expect(readFileSync(resolve(__dirname, '../src/main/sessions.ts'), 'utf8')).not.toContain("args.push('--max-tokens', '1000000')")
        expect(readFileSync(resolve(__dirname, '../src/renderer/src/components/sessions/SessionSettings.tsx'), 'utf8')).not.toContain("parts.push('--max-tokens', '1000000')")
    })

    it('custom max tokens is not overridden by default', () => {
        const out = preview({ maxTokens: 4096 })
        expect(getFlagValue(out, '--max-tokens')).toBe('4096')
    })

    it('casual preset maxTokens uses an explicit server output cap without changing model-owned defaults or context', () => {
        const formSource = readFileSync(resolve(__dirname, '../src/renderer/src/components/sessions/SessionConfigForm.tsx'), 'utf8')
        const casualStart = formSource.indexOf('export const CASUAL_CONFIG')
        const casualEnd = formSource.indexOf('\n}\n', casualStart)
        const casualBlock = formSource.slice(casualStart, casualEnd)

        expect(DEFAULT_CONFIG.maxTokens).toBe(0)
        expect(casualBlock).toContain('maxTokens: 8192')
        expect(casualBlock).not.toContain('maxContextLength')
        expect(formSource).toContain('limits runaway long replies')
        expect(formSource).not.toContain('prevents huge KV allocation')
    })

    it('JANGTQ router top-k override is not emitted by the app', () => {
        const out = preview({} as any, { family: 'minimax', isTurboQuant: true })
        expect(out.includes('JANGTQ_TOPK_OVERRIDE=')).toBe(false)
    })
})

describe('Tool Integration', () => {
    it('sets MCP config path', () => {
        const out = preview({ mcpConfig: '/path/mcp.json', enableAutoToolChoice: true })
        expect(getFlagValue(out, '--mcp-config')).toBe('/path/mcp.json')
    })

    it('sets session-level MCP policy flags', () => {
        const out = preview({
            mcpConfig: '/path/mcp.json',
            mcpEnabledServers: 'filesystem,github',
            mcpDisabledServers: 'browser_automation',
            mcpEnabledTools: 'filesystem__read_file\ngithub__search_repositories',
            mcpDisabledTools: 'filesystem__write_file',
            enableAutoToolChoice: true,
        })

        expect(getFlagValue(out, '--mcp-enabled-servers')).toBe('filesystem,github')
        expect(getFlagValue(out, '--mcp-disabled-servers')).toBe('browser_automation')
        expect(getFlagValue(out, '--mcp-enabled-tools')).toBe('filesystem__read_file,github__search_repositories')
        expect(getFlagValue(out, '--mcp-disabled-tools')).toBe('filesystem__write_file')
    })

    it('scrubs inherited MCP policy environment so UI settings own session policy', () => {
        const source = readFileSync('src/main/sessions.ts', 'utf8')

        for (const key of [
            'VLLM_MLX_MCP_CONFIG',
            'VLLM_MLX_MCP_ENABLED_SERVERS',
            'VLLM_MLX_MCP_DISABLED_SERVERS',
            'VLLM_MLX_MCP_ENABLED_TOOLS',
            'VLLM_MLX_MCP_DISABLED_TOOLS',
        ]) {
            expect(source).toContain(`delete spawnEnv.${key}`)
        }
    })

    it('enables auto tool choice', () => {
        const out = preview({ enableAutoToolChoice: true })
        expect(hasFlag(out, '--enable-auto-tool-choice')).toBe(true)
    })

    it('uses detected tool parser when user is auto', () => {
        const out = preview({ enableAutoToolChoice: true, toolCallParser: 'auto' }, { toolParser: 'qwen' })
        expect(getFlagValue(out, '--tool-call-parser')).toBe('qwen')
    })

    it('manual tool parser overrides when no detected', () => {
        const out = preview({ enableAutoToolChoice: true, toolCallParser: 'llama' })
        expect(getFlagValue(out, '--tool-call-parser')).toBe('llama')
    })

    it('canonicalizes legacy DSV4 and Hy3 parser aliases before launch', () => {
        expect(getFlagValue(preview({ enableAutoToolChoice: true, toolCallParser: 'deepseek_v4' }), '--tool-call-parser')).toBe('dsml')
        expect(getFlagValue(preview({ enableAutoToolChoice: true, toolCallParser: 'hy_v3' }), '--tool-call-parser')).toBe('hunyuan')
        expect(getFlagValue(preview({ enableAutoToolChoice: true, toolCallParser: 'auto' }, { toolParser: 'deepseek_v4' }), '--tool-call-parser')).toBe('dsml')
        expect(getFlagValue(preview({ enableAutoToolChoice: true, toolCallParser: 'auto' }, { toolParser: 'hy_v3' }), '--tool-call-parser')).toBe('hunyuan')
    })

    it('tool parser dropdown exposes DSV4 DSML, Hy3, and ZAYA parsers', () => {
        const formSource = readFileSync('src/renderer/src/components/sessions/SessionConfigForm.tsx', 'utf8')
        expect(formSource).toContain("value: 'dsml'")
        expect(formSource).toContain("value: 'hunyuan'")
        expect(formSource).toContain("value: 'zaya_xml'")
        expect(formSource).toContain('DeepSeek V4')
        expect(formSource).toContain('Hy3')
        expect(formSource).toContain('ZAYA')
    })

    it('tool parser dropdown exposes Gemma 3 tool_code parser separately from Hermes', () => {
        const formSource = readFileSync('src/renderer/src/components/sessions/SessionConfigForm.tsx', 'utf8')
        expect(formSource).toContain("value: 'gemma3'")
        expect(formSource).toContain('Gemma 3 / 3n')
        expect(formSource).toContain('tool_code')
        const reasoningTooltip = formSource.slice(
            formSource.indexOf('label="Reasoning Parser"'),
            formSource.indexOf('options={REASONING_PARSER_OPTIONS}', formSource.indexOf('label="Reasoning Parser"')),
        )
        expect(reasoningTooltip).not.toContain('Gemma 3')
    })

    it('tool parser dropdown covers every parser the panel registry can emit', () => {
        const registrySource = readFileSync('src/main/model-config-registry.ts', 'utf8')
        const formSource = readFileSync('src/renderer/src/components/sessions/SessionConfigForm.tsx', 'utf8')
        const emittedParsers = [...registrySource.matchAll(/toolParser: '([^']+)'/g)].map(match => match[1])
        const uiValues = new Set([...formSource.matchAll(/value: '([^']+)'/g)].map(match => match[1]))

        const missing = [...new Set(emittedParsers)].filter(parser => {
            return !uiValues.has(parser)
        })

        expect(missing).toEqual([])
    })

    it('reasoning parser dropdown covers every parser the panel registry can emit', () => {
        const registrySource = readFileSync('src/main/model-config-registry.ts', 'utf8')
        const formSource = readFileSync('src/renderer/src/components/sessions/SessionConfigForm.tsx', 'utf8')
        const emittedParsers = [...registrySource.matchAll(/reasoningParser: '([^']+)'/g)].map(match => match[1])
        const uiValues = new Set([...formSource.matchAll(/value: '([^']+)'/g)].map(match => match[1]))

        const missing = [...new Set(emittedParsers)].filter(parser => {
            return !uiValues.has(parser)
        })

        expect(missing).toEqual([])
    })

    it('manual tool parser takes priority over detected', () => {
        const out = preview({ enableAutoToolChoice: true, toolCallParser: 'llama' }, { toolParser: 'qwen' })
        expect(getFlagValue(out, '--tool-call-parser')).toBe('llama')
    })

    it('empty tool parser disables tool parsing', () => {
        const out = preview({ enableAutoToolChoice: true, toolCallParser: '' }, { toolParser: 'qwen' })
        expect(hasFlag(out, '--tool-call-parser')).toBe(false)
    })

    it('uses detected reasoning parser when user is auto', () => {
        const out = preview({ reasoningParser: 'auto' }, { reasoningParser: 'qwen3' })
        expect(getFlagValue(out, '--reasoning-parser')).toBe('qwen3')
    })

    it('manual reasoning parser when no detected', () => {
        const out = preview({ reasoningParser: 'deepseek_r1' })
        expect(getFlagValue(out, '--reasoning-parser')).toBe('deepseek_r1')
    })

    it('manual reasoning parser takes priority over detected', () => {
        const out = preview({ reasoningParser: 'deepseek_r1' }, { reasoningParser: 'qwen3' })
        expect(getFlagValue(out, '--reasoning-parser')).toBe('deepseek_r1')
    })

    it('passes MiniMax through the registered minimax_m2 reasoning parser', () => {
        const out = preview(
            { reasoningParser: 'minimax_m2', toolCallParser: 'minimax', enableAutoToolChoice: true },
            { family: 'minimax', reasoningParser: 'minimax_m2', toolParser: 'minimax', enableAutoToolChoice: true },
        )

        expect(getFlagValue(out, '--tool-call-parser')).toBe('minimax')
        expect(getFlagValue(out, '--reasoning-parser')).toBe('minimax_m2')
    })

    it('exposes MiniMax as its own reasoning parser option instead of under qwen3', () => {
        const formSource = readFileSync(resolve(__dirname, '../src/renderer/src/components/sessions/SessionConfigForm.tsx'), 'utf8')

        expect(formSource).toContain("value: 'minimax_m2'")
        expect(formSource).toContain('MiniMax M2')
        expect(formSource).not.toContain('Qwen / QwQ / MiniMax / StepFun')
    })

    // ── enableAutoToolChoice auto-detection regression tests ──
    // Bug: DEFAULT_CONFIG had enableAutoToolChoice: false, which blocked auto-detection
    // because ?? doesn't fall through on false (only null/undefined).
    // Fix: enableAutoToolChoice now defaults to undefined, allowing auto-detection.

    it('undefined enableAutoToolChoice allows auto-detection (the fix)', () => {
        // With undefined (new default) + detected enableAutoToolChoice: true
        // → --enable-auto-tool-choice MUST be emitted
        const out = preview({ toolCallParser: 'auto' }, { toolParser: 'qwen', enableAutoToolChoice: true })
        expect(hasFlag(out, '--enable-auto-tool-choice')).toBe(true)
        expect(getFlagValue(out, '--tool-call-parser')).toBe('qwen')
    })

    it('explicit false enableAutoToolChoice blocks auto-detection', () => {
        // User explicitly disabled → must NOT emit --enable-auto-tool-choice
        const out = preview({ enableAutoToolChoice: false, toolCallParser: 'auto' }, { toolParser: 'qwen', enableAutoToolChoice: true })
        expect(hasFlag(out, '--tool-call-parser')).toBe(true)
        expect(hasFlag(out, '--enable-auto-tool-choice')).toBe(false)
    })

    it('explicit true enableAutoToolChoice overrides detection', () => {
        // User explicitly enabled → must emit even without detection
        const out = preview({ enableAutoToolChoice: true, toolCallParser: 'llama' })
        expect(hasFlag(out, '--enable-auto-tool-choice')).toBe(true)
    })

    it('default config (no enableAutoToolChoice) with detected parser enables auto-tool-choice', () => {
        // This is the exact scenario from the bug report:
        // User creates session with default settings, model has tool support detected
        const out = preview({}, { toolParser: 'qwen', enableAutoToolChoice: true })
        expect(hasFlag(out, '--enable-auto-tool-choice')).toBe(true)
        expect(getFlagValue(out, '--tool-call-parser')).toBe('qwen')
    })

    it('default config without detected parser does not enable auto-tool-choice', () => {
        // Unknown model, no detection → no auto-tool-choice
        const out = preview({})
        expect(hasFlag(out, '--enable-auto-tool-choice')).toBe(false)
        expect(hasFlag(out, '--tool-call-parser')).toBe(false)
    })

    it('MCP config with auto-detected tools works with default settings', () => {
        // User sets MCP config path but doesn't touch enableAutoToolChoice
        // Should auto-detect and enable tool calling
        const out = preview({ mcpConfig: '/Volumes/Data/mcp.json' }, { toolParser: 'qwen', enableAutoToolChoice: true })
        expect(hasFlag(out, '--enable-auto-tool-choice')).toBe(true)
        expect(getFlagValue(out, '--mcp-config')).toBe('/Volumes/Data/mcp.json')
    })

    it('empty reasoning parser disables reasoning', () => {
        const out = preview({ reasoningParser: '' }, { reasoningParser: 'qwen3' })
        expect(hasFlag(out, '--reasoning-parser')).toBe(false)
    })
})

describe('Served Model Name', () => {
    it('sets served model name from config', () => {
        const out = preview({ servedModelName: 'my-custom-model' })
        expect(getFlagValue(out, '--served-model-name')).toBe('my-custom-model')
    })

    it('omits served model name when empty', () => {
        const out = preview({ servedModelName: '' })
        expect(hasFlag(out, '--served-model-name')).toBe(false)
    })
})

describe('Speculative Decoding', () => {
    it('sets speculative model from config', () => {
        const out = preview({ speculativeModel: 'mlx-community/Llama-3.2-1B-Instruct-4bit' })
        expect(getFlagValue(out, '--speculative-model')).toBe('mlx-community/Llama-3.2-1B-Instruct-4bit')
    })

    it('omits speculative model when empty', () => {
        const out = preview({ speculativeModel: '' })
        expect(hasFlag(out, '--speculative-model')).toBe(false)
    })

    it('omits --num-draft-tokens when default (3)', () => {
        const out = preview({ speculativeModel: 'draft-model', numDraftTokens: 3 })
        expect(hasFlag(out, '--speculative-model')).toBe(true)
        expect(hasFlag(out, '--num-draft-tokens')).toBe(false)
    })

    it('sets --num-draft-tokens when non-default', () => {
        const out = preview({ speculativeModel: 'draft-model', numDraftTokens: 5 })
        expect(getFlagValue(out, '--num-draft-tokens')).toBe('5')
    })

    it('omits --num-draft-tokens when no speculative model', () => {
        const out = preview({ speculativeModel: '', numDraftTokens: 10 })
        expect(hasFlag(out, '--num-draft-tokens')).toBe(false)
    })

    it('sets --num-draft-tokens=1 (minimum)', () => {
        const out = preview({ speculativeModel: 'draft-model', numDraftTokens: 1 })
        expect(getFlagValue(out, '--num-draft-tokens')).toBe('1')
    })

    it('sets --num-draft-tokens=20 (maximum)', () => {
        const out = preview({ speculativeModel: 'draft-model', numDraftTokens: 20 })
        expect(getFlagValue(out, '--num-draft-tokens')).toBe('20')
    })
})

describe('Native MTP', () => {
    const qwenMtpDetected: DetectedConfig = {
        family: 'qwen3.5',
        cacheType: 'hybrid',
        usePagedCache: true,
        isMultimodal: true,
        reasoningParser: 'qwen3',
        toolParser: 'qwen',
        enableAutoToolChoice: true,
        nativeMtp: {
            supported: true,
            depth: 2,
            depthSource: 'vmlx_mtp_tuning.json:native_mtp.best_depth',
            runtimeScope: 'text+vl',
            requiresDeterministicSampling: true,
        },
    }

    it('defaults native-MTP bundles to deterministic measured-depth launch policy without hidden sampler flags', () => {
        const out = preview({}, qwenMtpDetected)

        expect(getFlagValue(out, '--native-mtp-depth')).toBe('2')
        expect(getFlagValue(out, '--native-mtp-sampling-policy')).toBe('deterministic-defaults')
        expect(hasFlag(out, '--default-temperature')).toBe(false)
        expect(hasFlag(out, '--default-top-p')).toBe(false)
        expect(hasFlag(out, '--default-top-k')).toBe(false)
        expect(hasFlag(out, '--default-min-p')).toBe(false)
        expect(hasFlag(out, '--default-repetition-penalty')).toBe(false)
    })

    it('lets a manual native-MTP depth override win over the measured default', () => {
        const out = preview({ nativeMtpDepth: 3, nativeMtpDepthOverride: true }, qwenMtpDetected)

        expect(getFlagValue(out, '--native-mtp-depth')).toBe('3')
    })

    it('lets users disable native MTP without leaving deterministic sampling overrides behind', () => {
        const out = preview({ nativeMtpMode: 'off' }, qwenMtpDetected)

        expect(hasFlag(out, '--disable-native-mtp')).toBe(true)
        expect(hasFlag(out, '--native-mtp-depth')).toBe(false)
        expect(hasFlag(out, '--default-temperature')).toBe(false)
    })

    it('keeps non-MTP models on bundle-owned generation defaults', () => {
        const out = preview({ nativeMtpMode: 'deterministic', nativeMtpDepth: 3 }, { family: 'qwen3.5', cacheType: 'kv' })

        expect(hasFlag(out, '--native-mtp-depth')).toBe(false)
        expect(hasFlag(out, '--default-temperature')).toBe(false)
    })

    it('real session launcher and settings form expose native MTP controls', () => {
        const sessionsSource = readFileSync('src/main/sessions.ts', 'utf8')
        const formSource = readFileSync('src/renderer/src/components/sessions/SessionConfigForm.tsx', 'utf8')

        expect(sessionsSource).toContain('--native-mtp-depth')
        expect(sessionsSource).toContain('--native-mtp-sampling-policy')
        expect(sessionsSource).toContain('--disable-native-mtp')
        expect(formSource).toContain('Native MTP')
        expect(formSource).toContain('nativeMtpMode')
        expect(formSource).toContain('nativeMtpDepth')
    })
})

describe('Generation Defaults', () => {
    it('uses the published vmlx package name for PyPI install guidance while preserving vmlx-engine entrypoints', () => {
        const engineManager = readFileSync('src/main/engine-manager.ts', 'utf8')
        const createSource = readFileSync('src/renderer/src/components/sessions/CreateSession.tsx', 'utf8')

        expect(engineManager).toContain("const ENTRY_POINT_NAMES = ['vmlx-engine', 'vmlx-serve', 'vmlx']")
        expect(engineManager).toContain("const PYPI_PACKAGE_NAME = 'vmlx'")
        expect(engineManager).toContain('const pkg = bundledSource || PYPI_PACKAGE_NAME')
        expect(engineManager).toContain("['tool', 'upgrade', PYPI_PACKAGE_NAME]")
        expect(engineManager).not.toContain("const pkg = bundledSource || 'vmlx-engine'")
        expect(engineManager).not.toContain("['tool', 'upgrade', 'vmlx-engine']")

        expect(createSource).toContain('uv tool install vmlx')
        expect(createSource).toContain('pip3 install vmlx')
        expect(createSource).not.toContain('uv tool install vmlx-engine')
        expect(createSource).not.toContain('pip3 install vmlx-engine')
    })

    it('does not synthesize server --default sampling flags from UI/session config', () => {
        const out = preview({
            defaultTemperature: 80,
            defaultTopP: 90,
            defaultTopK: 40,
            defaultMinP: 5,
            defaultRepetitionPenalty: 110,
        })
        expect(hasFlag(out, '--default-temperature')).toBe(false)
        expect(hasFlag(out, '--default-top-p')).toBe(false)
        expect(hasFlag(out, '--default-top-k')).toBe(false)
        expect(hasFlag(out, '--default-min-p')).toBe(false)
        expect(hasFlag(out, '--default-repetition-penalty')).toBe(false)
    })

    it('does not copy model max_new_tokens into hidden startup maxTokens config', () => {
        const sessionsSource = readFileSync('src/main/sessions.ts', 'utf8')
        const createSource = readFileSync('src/renderer/src/components/sessions/CreateSession.tsx', 'utf8')
        const sessionSettingsSource = readFileSync('src/renderer/src/components/sessions/SessionSettings.tsx', 'utf8')
        const formSource = readFileSync('src/renderer/src/components/sessions/SessionConfigForm.tsx', 'utf8')

        expect(sessionsSource).toContain('max_new_tokens is also bundle-owned')
        expect(sessionsSource).toContain('defaultMaxNewTokens')
        expect(createSource).toContain('next.defaultMaxNewTokens')
        expect(createSource).toContain('stored.defaultMaxNewTokens')
        expect(sessionSettingsSource).toContain('next.defaultMaxNewTokens')
        expect(sessionsSource).not.toContain('config.maxTokens = defs.maxTokens')
        expect(createSource).not.toContain('next.maxTokens = gen.maxNewTokens')
        expect(createSource).not.toContain('stored.maxTokens = gen.maxNewTokens')
        expect(sessionSettingsSource).not.toContain('next.maxTokens = gen.maxNewTokens')
        expect(formSource).not.toContain('`max tokens ${config.maxTokens}`')
        expect(formSource).toContain('max output tokens')
    })

    it('chat settings default to neutral repeat penalty when bundle has no value', () => {
        const source = readFileSync('src/renderer/src/components/chat/ChatSettings.tsx', 'utf8')
        expect(source).toContain('value={overrides.repeatPenalty ?? modelDefaults.repeatPenalty ?? 1.0}')
        expect(source).toContain("onChange={v => update('repeatPenalty', v === 1.0 ? undefined : v)}")
        expect(source).not.toContain('value={overrides.repeatPenalty ?? 1.1}')
    })

    it('chat settings expose per-chat max tokens without hidden DSV4 floors', () => {
        const chatSettings = readFileSync('src/renderer/src/components/chat/ChatSettings.tsx', 'utf8')
        const chatIpc = readFileSync('src/main/ipc/chat.ts', 'utf8')
        expect(chatSettings).toContain("onChange={v => update('maxTokens', v)}")
        expect(chatSettings).toContain('model default')
        expect(chatSettings).not.toContain('next.maxTokens = Math.max')
        expect(chatSettings).not.toContain('4096')
        expect(chatIpc).toContain('dsv4OutputBudget(')
        expect(chatIpc).not.toContain('dsv4_finalizer_tokens')
        expect(chatIpc).not.toContain('Math.max(parsed ?? 0, 4096)')
    })

    it('new chat creation inherits only tool/workspace ergonomics, not stale sampling or prompts', () => {
        const source = readFileSync('src/main/ipc/chat.ts', 'utf8')
        const policy = readFileSync('src/main/chat-override-policy.ts', 'utf8')
        const createHandler = source.slice(
            source.indexOf('"chat:create"'),
            source.indexOf('ipcMain.handle("chat:getByModel"'),
        )

        expect(source).toContain('bundle generation defaults stay authoritative')
        expect(createHandler).toContain('getDefaultChatProfile')
        expect(createHandler).toContain('buildNewChatInheritedOverrides')
        expect(policy).toContain('NEW_CHAT_TOOL_INHERIT_KEYS')
        expect(policy).not.toContain("'systemPrompt'")
        expect(policy).not.toContain("'temperature'")
        expect(policy).not.toContain("'enableThinking'")
        expect(source).not.toContain('enableThinkingFromReasoningMode')
        expect(source).not.toContain('readGenerationDefaults')
        expect(source).not.toContain('Applied global model settings / generation defaults')
    })

    it('database migrates historical repeatPenalty 1.10 back to bundle defaults', () => {
        const source = readFileSync('src/main/database.ts', 'utf8')
        expect(source).toContain('migration_reset_stale_repeat_penalty_1_5_34')
        expect(source).toContain('UPDATE chat_overrides SET repeat_penalty = NULL')
        expect(source).toContain('delete parsed.repeatPenalty')
    })

    it('database clears historical generic sampling/model-setting rows once', () => {
        const source = readFileSync('src/main/database.ts', 'utf8')
        expect(source).toContain('migration_reset_stale_sampling_overrides_1_5_37')
        expect(source).toContain('temperature = CASE')
        expect(source).toContain('top_k = CASE WHEN top_k = 40 THEN NULL ELSE top_k END')
        expect(source).toContain('max_tokens IN (4096, 12000, 12068, 32768)')
        expect(source).toContain('migration_clear_model_settings_sampling_1_5_37')
        expect(source).toContain("reasoning_mode = 'auto'")
    })

    it('database clears legacy session maxTokens before settings UI or launch can reuse them', () => {
        const source = readFileSync('src/main/database.ts', 'utf8')
        const helper = readFileSync('src/shared/sessionConfigMigrations.ts', 'utf8')
        expect(source).toContain('migration_clear_legacy_session_max_output_1_5_45_2')
        expect(source).toMatch(/legacySessionMaxOutputKey[\s\S]*SELECT id, model_path, config FROM sessions/)
        expect(source).toContain('migrateLegacySessionStartupConfig(')
        expect(source).toContain('session.model_path')
        expect(helper).toContain('config.maxTokens = 0')
        expect(helper).toContain('config.generationStartupDefaultsVersion = GENERATION_STARTUP_DEFAULTS_VERSION')
        expect(helper).toContain("config.reasoningParser === 'qwen3'")
        expect(helper).toContain("config.reasoningParser = 'minimax_m2'")
    })

    it('saving chat overrides never syncs sampling or thinking back to model_settings', () => {
        const source = readFileSync('src/main/ipc/chat.ts', 'utf8')
        expect(source).not.toContain('Synced ${chatId} inference overrides back to global model_settings')
        expect(source).not.toContain('existingModelConfig.temperature')
        expect(source).not.toContain('existingModelConfig.top_p')
        expect(source).not.toContain('existingModelConfig.max_tokens')
        expect(source).not.toContain('existingModelConfig.reasoning_mode')
        expect(source).not.toContain('db.saveModelSettings(chat.modelPath')
    })

    it('model-settings IPC exposes launch metadata only, not reasoning or sampling rails', () => {
        const source = readFileSync('src/main/db/model-settings.ts', 'utf8')
        expect(source).toContain('Sampling and thinking')
        expect(source).not.toContain('reasoning_mode:')
        expect(source).not.toContain('settings.reasoning_mode')
        expect(source).not.toContain('sanitized.reasoning_mode')
        expect(source).not.toContain('temperature')
        expect(source).not.toContain('top_p')
        expect(source).not.toContain('max_tokens')
    })

    it('never emits user-saved server default thinking override from session startup', () => {
        expect(hasFlag(preview({ defaultEnableThinking: undefined }), '--default-enable-thinking')).toBe(false)
        expect(hasFlag(preview({ defaultEnableThinking: true }), '--default-enable-thinking')).toBe(false)
        expect(hasFlag(preview({ defaultEnableThinking: false }), '--default-enable-thinking')).toBe(false)
    })

    it('keeps detected family thinking defaults model-owned at startup', () => {
        const out = preview({}, {
            family: 'zaya',
            toolParser: 'zaya_xml',
            reasoningParser: 'qwen3',
            defaultEnableThinking: false,
            enableAutoToolChoice: true,
            cacheType: 'hybrid',
            usePagedCache: true,
        })
        expect(hasFlag(out, '--default-enable-thinking')).toBe(false)
        expect(getFlagValue(out, '--reasoning-parser')).toBe('qwen3')
    })
})

describe('Embedding Model', () => {
    it('sets embedding model from config', () => {
        const out = preview({ embeddingModel: 'mlx-community/embeddinggemma-300m-6bit' })
        expect(getFlagValue(out, '--embedding-model')).toBe('mlx-community/embeddinggemma-300m-6bit')
    })

    it('omits embedding model when empty', () => {
        const out = preview({ embeddingModel: '' })
        expect(hasFlag(out, '--embedding-model')).toBe(false)
    })
})

describe('Additional Arguments', () => {
    it('appends additional args to command', () => {
        const out = preview({ additionalArgs: '--log-level DEBUG' })
        expect(hasFlag(out, '--log-level DEBUG')).toBe(true)
    })

    it('DSV4 additional args cannot reenable native MTP or deterministic sampling policy', () => {
        const out = preview(
            {
                additionalArgs: [
                    '--native-mtp-depth 3',
                    '--native-mtp-sampling-policy deterministic-defaults',
                    '--disable-native-mtp',
                    '--dsv4-enable-prefix-cache',
                    '--default-temperature 0',
                    '--max-tokens 32768',
                    '--log-level DEBUG',
                ].join(' '),
            },
            { family: 'deepseek-v4', usePagedCache: false },
        )
        const normalized = out.replace(/\s*\\\n\s*/g, ' ')

        expect(normalized).not.toContain('--native-mtp-depth')
        expect(normalized).not.toContain('--native-mtp-sampling-policy')
        expect(normalized).not.toContain('deterministic-defaults')
        expect(normalized).not.toContain('--disable-native-mtp')
        expect(normalized).not.toContain('--dsv4-enable-prefix-cache')
        expect(normalized).not.toContain('--default-temperature')
        expect(normalized).not.toContain('--max-tokens')
        expect(normalized).not.toContain('--log-level DEBUG')

        const sessionsSource = readFileSync('src/main/sessions.ts', 'utf8')
        const settingsSource = readFileSync('src/renderer/src/components/sessions/SessionSettings.tsx', 'utf8')
        for (const source of [sessionsSource, settingsSource]) {
            expect(source).toContain("'--native-mtp-depth'")
            expect(source).toContain("'--native-mtp-sampling-policy'")
            expect(source).toContain("'--disable-native-mtp'")
            expect(source).toContain("'--dsv4-enable-prefix-cache'")
        }
    })

    it('omits additional args when empty', () => {
        const out = preview({ additionalArgs: '' })
        expect(out.trim()).toBe(out)
        expect(out).not.toContain('undefined')
    })
})

describe('No Hardcoded Values', () => {
    it('changing host produces different CLI output', () => {
        const a = preview({ host: '127.0.0.1' })
        const b = preview({ host: '192.168.1.1' })
        expect(a).not.toBe(b)
        expect(getFlagValue(a, '--host')).toBe('127.0.0.1')
        expect(getFlagValue(b, '--host')).toBe('192.168.1.1')
    })

    it('changing port produces different CLI output', () => {
        expect(getFlagValue(preview({ port: 8000 }), '--port')).toBe('8000')
        expect(getFlagValue(preview({ port: 9000 }), '--port')).toBe('9000')
    })

    it('changing maxTokens produces different CLI output', () => {
        expect(getFlagValue(preview({ maxTokens: 4096 }), '--max-tokens')).toBe('4096')
        expect(getFlagValue(preview({ maxTokens: 131072 }), '--max-tokens')).toBe('131072')
    })

    it('omits malformed persisted numeric launch overrides instead of emitting invalid CLI values', () => {
        const out = preview({
            rateLimit: Number.NaN,
            maxNumSeqs: Number.POSITIVE_INFINITY,
            prefillBatchSize: '512' as any,
            prefillStepSize: Number.NEGATIVE_INFINITY,
            completionBatchSize: Number.NaN,
            prefixCacheSize: Number.POSITIVE_INFINITY,
            prefixCacheMaxBytes: '4096' as any,
            cacheMemoryMb: Number.NaN,
            cacheMemoryPercent: Number.POSITIVE_INFINITY,
            cacheTtlMinutes: Number.NEGATIVE_INFINITY,
            pagedCacheBlockSize: Number.NaN,
            maxCacheBlocks: Number.POSITIVE_INFINITY,
            kvCacheQuantization: 'q4',
            kvCacheGroupSize: '64' as any,
            diskCacheMaxGb: Number.NaN,
            blockDiskCacheMaxGb: Number.POSITIVE_INFINITY,
            streamInterval: Number.NaN,
            maxTokens: Number.POSITIVE_INFINITY,
            maxContextLength: '32768' as any,
            speculativeModel: '/models/draft',
            numDraftTokens: Number.NaN,
            nativeMtpDepthOverride: true,
            nativeMtpDepth: Number.POSITIVE_INFINITY,
        }, {
            nativeMtp: { supported: true, depth: 3 },
        })

        expectNoInvalidNumericFlagValues(out)
        expect(getFlagValue(out, '--rate-limit')).toBeUndefined()
        expect(getFlagValue(out, '--max-num-seqs')).toBeUndefined()
        expect(getFlagValue(out, '--prefill-batch-size')).toBeUndefined()
        expect(getFlagValue(out, '--prefill-step-size')).toBeUndefined()
        expect(getFlagValue(out, '--completion-batch-size')).toBeUndefined()
        expect(getFlagValue(out, '--prefix-cache-size')).toBeUndefined()
        expect(getFlagValue(out, '--prefix-cache-max-bytes')).toBeUndefined()
        expect(getFlagValue(out, '--cache-memory-mb')).toBeUndefined()
        expect(getFlagValue(out, '--cache-memory-percent')).toBeUndefined()
        expect(getFlagValue(out, '--cache-ttl-minutes')).toBeUndefined()
        expect(getFlagValue(out, '--max-cache-blocks')).toBeUndefined()
        expect(getFlagValue(out, '--kv-cache-group-size')).toBeUndefined()
        expect(getFlagValue(out, '--disk-cache-max-gb')).toBeUndefined()
        expect(getFlagValue(out, '--block-disk-cache-max-gb')).toBeUndefined()
        expect(getFlagValue(out, '--stream-interval')).toBeUndefined()
        expect(getFlagValue(out, '--max-tokens')).toBeUndefined()
        expect(getFlagValue(out, '--max-prompt-tokens')).toBeUndefined()
        expect(getFlagValue(out, '--num-draft-tokens')).toBeUndefined()
        expect(getFlagValue(out, '--native-mtp-depth')).toBe('3')
    })

    it('floors positive decimal output/context launch overrides in the UI preview', () => {
        const out = preview({
            streamInterval: 1.9,
            maxTokens: 512.9,
            maxContextLength: 32768.9,
        })

        expect(getFlagValue(out, '--stream-interval')).toBe('1')
        expect(getFlagValue(out, '--max-tokens')).toBe('512')
        expect(getFlagValue(out, '--max-prompt-tokens')).toBe('32768')

        const settingsSource = readFileSync(resolve(__dirname, '../src/renderer/src/components/sessions/SessionSettings.tsx'), 'utf8')
        const previewBlock = settingsSource.slice(
            settingsSource.indexOf('function buildCommandPreview'),
            settingsSource.indexOf('const SettingsSection'),
        )
        expect(previewBlock).toContain('finitePositiveInteger(config.maxTokens)')
        expect(previewBlock).toContain('finitePositiveInteger(config.maxContextLength)')
        expect(previewBlock).toContain("parts.push('--max-prompt-tokens', maxContextLength.toString())")
    })

    it('changing prefillBatchSize produces different CLI output', () => {
        expect(getFlagValue(preview({ prefillBatchSize: 256 }), '--prefill-batch-size')).toBe('256')
        expect(getFlagValue(preview({ prefillBatchSize: 1024 }), '--prefill-batch-size')).toBe('1024')
    })

    it('changing completionBatchSize produces different CLI output', () => {
        expect(getFlagValue(preview({ completionBatchSize: 64 }), '--completion-batch-size')).toBe('64')
        expect(getFlagValue(preview({ completionBatchSize: 512 }), '--completion-batch-size')).toBe('512')
    })

    it('changing maxNumSeqs produces different CLI output', () => {
        expect(getFlagValue(preview({ maxNumSeqs: 32 }), '--max-num-seqs')).toBe('32')
        expect(getFlagValue(preview({ maxNumSeqs: 512 }), '--max-num-seqs')).toBe('512')
    })

    it('changing pagedCacheBlockSize produces different CLI output', () => {
        expect(getFlagValue(preview({ enablePrefixCache: true, pagedCacheBlockSize: 32 }), '--paged-cache-block-size')).toBe('32')
        expect(getFlagValue(preview({ enablePrefixCache: true, pagedCacheBlockSize: 256 }), '--paged-cache-block-size')).toBe('256')
    })

    it('deepseek-v4 disables composite prefix cache by default even with stale cache config', () => {
        const out = preview(
            {
                enablePrefixCache: true,
                continuousBatching: false,
                usePagedCache: false,
                pagedCacheBlockSize: 64,
                prefillBatchSize: 512,
                prefillStepSize: 2048,
                completionBatchSize: 512,
                kvCacheQuantization: 'q4',
                speculativeModel: '/tmp/draft',
                isMultimodal: true,
                nativeMtpDepth: 3,
            },
            { family: 'deepseek-v4', usePagedCache: false },
        )

        expect(hasFlag(out, '--dsv4-enable-prefix-cache')).toBe(false)
        expect(hasFlag(out, '--use-paged-cache')).toBe(false)
        expect(hasFlag(out, '--paged-cache-block-size')).toBe(false)
        expect(getFlagValue(out, '--max-num-seqs')).toBe('1')
        expect(hasFlag(out, '--no-continuous-batching')).toBe(false)
        expect(hasFlag(out, '--continuous-batching')).toBe(true)
        expect(hasFlag(out, '--disable-prefix-cache')).toBe(true)
        expect(hasFlag(out, '--prefill-batch-size')).toBe(false)
        expect(hasFlag(out, '--prefill-step-size')).toBe(false)
        expect(hasFlag(out, '--completion-batch-size')).toBe(false)
        expect(hasFlag(out, '--kv-cache-quantization')).toBe(false)
        expect(hasFlag(out, '--speculative-model')).toBe(false)
        expect(hasFlag(out, '--is-mllm')).toBe(false)
        expect(hasFlag(out, '--native-mtp-depth')).toBe(false)
    })

    it('deepseek-v4 diagnostic cache opt-in uses DS4 page-sized blocks', () => {
        const out = preview(
            {
                dsv4PrefixCache: true,
                enablePrefixCache: true,
                usePagedCache: false,
                enableBlockDiskCache: true,
                pagedCacheBlockSize: 64,
            },
            { family: 'deepseek-v4', usePagedCache: false },
        )

        expect(hasFlag(out, '--dsv4-enable-prefix-cache')).toBe(true)
        expect(hasFlag(out, '--disable-prefix-cache')).toBe(false)
        expect(hasFlag(out, '--use-paged-cache')).toBe(true)
        expect(getFlagValue(out, '--paged-cache-block-size')).toBe('256')
        expect(hasFlag(out, '--enable-block-disk-cache')).toBe(true)
    })

    it('deepseek-v4 respects explicit prefix cache disable', () => {
        const out = preview(
            {
                enablePrefixCache: false,
                usePagedCache: true,
                enableBlockDiskCache: true,
                kvCacheQuantization: 'q8',
            },
            { family: 'deepseek-v4', usePagedCache: true },
        )

        expect(hasFlag(out, '--disable-prefix-cache')).toBe(true)
        expect(hasFlag(out, '--use-paged-cache')).toBe(false)
        expect(hasFlag(out, '--enable-block-disk-cache')).toBe(false)
        expect(hasFlag(out, '--kv-cache-quantization')).toBe(false)
    })

    it('DSV4 pool quant and native prefix controls stay DSV4-only', () => {
        const staleDsv4Config = {
            dsv4PrefixCache: true,
            dsv4PoolQuant: true,
            enablePrefixCache: true,
            usePagedCache: true,
            enableBlockDiskCache: true,
            kvCacheQuantization: 'q8',
            kvCacheGroupSize: 32,
        } as const

        const nonDsv4 = preview(staleDsv4Config, {
            family: 'qwen3_5',
            usePagedCache: true,
        })
        expect(hasFlag(nonDsv4, '--dsv4-enable-prefix-cache')).toBe(false)
        expect(hasFlag(nonDsv4, '--use-paged-cache')).toBe(true)
        expect(hasFlag(nonDsv4, '--enable-block-disk-cache')).toBe(true)
        expect(getFlagValue(nonDsv4, '--kv-cache-quantization')).toBe('q8')
        expect(getFlagValue(nonDsv4, '--kv-cache-group-size')).toBe('32')

        const dsv4 = preview(staleDsv4Config, {
            family: 'deepseek-v4',
            usePagedCache: true,
        })
        expect(hasFlag(dsv4, '--dsv4-enable-prefix-cache')).toBe(true)
        expect(hasFlag(dsv4, '--use-paged-cache')).toBe(true)
        expect(getFlagValue(dsv4, '--paged-cache-block-size')).toBe('256')
        expect(hasFlag(dsv4, '--kv-cache-quantization')).toBe(false)
        expect(hasFlag(dsv4, '--kv-cache-group-size')).toBe(false)
    })

    it('deepseek-v4 family defaults preserve explicit pool quant and initialize missing false', () => {
        const source = readFileSync(resolve(__dirname, '../src/main/sessions.ts'), 'utf8')
        expect(source).toContain('config.dsv4PoolQuant == null')
        expect(source).toContain('dsv4PoolQuant: false')
    })

    it('detected Qwen3.6 hybrid cache forces paged cache over stale saved false', () => {
        const out = preview(
            {
                enablePrefixCache: true,
                usePagedCache: false,
                enableBlockDiskCache: true,
                cacheMemoryPercent: 15,
            },
            {
                family: 'qwen3.5-moe',
                cacheType: 'hybrid',
                usePagedCache: true,
                isMultimodal: true,
                reasoningParser: 'qwen3',
            },
        )

        expect(hasFlag(out, '--is-mllm')).toBe(true)
        expect(hasFlag(out, '--use-paged-cache')).toBe(true)
        expect(hasFlag(out, '--enable-block-disk-cache')).toBe(true)
        expect(hasFlag(out, '--cache-memory-percent')).toBe(false)
    })

    it('detected Mamba cache forces paged cache while regular KV respects saved false', () => {
        const mambaOut = preview(
            { enablePrefixCache: true, usePagedCache: false },
            { family: 'qwen3-next', cacheType: 'mamba', usePagedCache: true },
        )
        const kvOut = preview(
            { enablePrefixCache: true, usePagedCache: false },
            { family: 'qwen3', cacheType: 'kv', usePagedCache: true },
        )

        expect(hasFlag(mambaOut, '--use-paged-cache')).toBe(true)
        expect(hasFlag(kvOut, '--use-paged-cache')).toBe(false)
    })

    it('changing maxCacheBlocks produces different CLI output', () => {
        expect(getFlagValue(preview({ enablePrefixCache: true, maxCacheBlocks: 500 }), '--max-cache-blocks')).toBe('500')
        expect(getFlagValue(preview({ enablePrefixCache: true, maxCacheBlocks: 5000 }), '--max-cache-blocks')).toBe('5000')
    })

    it('changing startup generation defaults does not change CLI output', () => {
        expect(preview({ defaultTemperature: 50 })).toBe(preview({ defaultTemperature: 100 }))
        expect(preview({ defaultTopK: 40 })).toBe(preview({ defaultTopK: 0 }))
    })

    it('changing speculativeModel produces different CLI output', () => {
        const a = preview({ speculativeModel: 'model-a' })
        const b = preview({ speculativeModel: 'model-b' })
        expect(a).not.toBe(b)
        expect(getFlagValue(a, '--speculative-model')).toBe('model-a')
        expect(getFlagValue(b, '--speculative-model')).toBe('model-b')
    })

    it('changing logLevel produces different CLI output', () => {
        expect(hasFlag(preview({ logLevel: 'DEBUG' }), '--log-level')).toBe(true)
        expect(getFlagValue(preview({ logLevel: 'DEBUG' }), '--log-level')).toBe('DEBUG')
        expect(getFlagValue(preview({ logLevel: 'ERROR' }), '--log-level')).toBe('ERROR')
    })

    it('changing corsOrigins produces different CLI output', () => {
        const out = preview({ corsOrigins: 'http://localhost:3000' })
        expect(getFlagValue(out, '--allowed-origins')).toBe('http://localhost:3000')
    })

    it('maxContextLength emits max prompt/context CLI flag when explicitly set', () => {
        const out = preview({ maxContextLength: 8192 })
        expect(getFlagValue(out, '--max-prompt-tokens')).toBe('8192')
    })
})

describe('Default IP and New Settings', () => {
    it('default host is local-only 127.0.0.1', () => {
        expect(DEFAULT_CONFIG.host).toBe('127.0.0.1')
    })

    it('default host produces --host 127.0.0.1 in CLI output', () => {
        const out = preview()
        expect(getFlagValue(out, '--host')).toBe('127.0.0.1')
    })

    it('current startup defaults keep the single-user cache stack enabled', () => {
        const out = preview()

        expect(getFlagValue(out, '--max-num-seqs')).toBe('1')
        expect(getFlagValue(out, '--prefill-batch-size')).toBe('512')
        expect(getFlagValue(out, '--prefill-step-size')).toBe('2048')
        expect(getFlagValue(out, '--completion-batch-size')).toBe('512')
        expect(hasFlag(out, '--continuous-batching')).toBe(true)
        expect(hasFlag(out, '--disable-prefix-cache')).toBe(false)
        expect(hasFlag(out, '--cache-memory-percent')).toBe(false)
        expect(hasFlag(out, '--use-paged-cache')).toBe(true)
        expect(hasFlag(out, '--enable-block-disk-cache')).toBe(true)
        expect(hasFlag(out, '--default-temperature')).toBe(false)
        expect(hasFlag(out, '--default-top-p')).toBe(false)
        expect(hasFlag(out, '--default-repetition-penalty')).toBe(false)
        expect(hasFlag(out, '--enable-jit')).toBe(true)
    })

    it('session manager migrates the exact stale continuous-cache default tuple', () => {
        const source = readFileSync('src/main/sessions.ts', 'utf8')
        expect(source).toContain('function applyCacheStackStartupDefaultMigration')
        expect(source).toContain('const CACHE_STACK_STARTUP_DEFAULTS_VERSION = 2')
        expect(source).toContain('function markCacheStackStartupDefaultsCurrent')
        expect(source).toContain('config.cacheStackStartupDefaultsVersion = CACHE_STACK_STARTUP_DEFAULTS_VERSION')
        expect(source).toContain('config.continuousBatching === true')
        expect(source).toContain('config.enablePrefixCache === true')
        expect(source).toContain('Number(config.maxNumSeqs) === 64')
        expect(source).toContain('Number(config.prefillBatchSize) === 1024')
        expect(source).toContain('Number(config.completionBatchSize) === 1024')
        expect(source).toContain('config.continuousBatching = true')
        expect(source).toContain('config.enablePrefixCache = true')
        expect(source).toContain('config.maxNumSeqs = 1')
        expect(source).toContain('config.prefillBatchSize = 512')
        expect(source).toContain('config.prefillStepSize = 2048')
        expect(source).toContain('config.completionBatchSize = 512')
        expect(source).toContain('config.usePagedCache = true')
        expect(source).toContain('config.maxCacheBlocks = 1000')
        expect(source).toContain("config.kvCacheQuantization = 'auto'")
        expect(source).toContain('config.enableBlockDiskCache = true')
        expect(source).toContain('config.blockDiskCacheMaxGb = 10')
        expect(source).toContain('config.cacheMemoryPercent = 15')
    })

    it('cache-stack migration is one-time versioned so saved user toggles stick', () => {
        const source = readFileSync('src/main/sessions.ts', 'utf8')
        const serverSource = readFileSync('src/main/server.ts', 'utf8')
        expect(serverSource).toContain('cacheStackStartupDefaultsVersion?: number')
        expect(source).toContain('Number(config.cacheStackStartupDefaultsVersion || 0) >= CACHE_STACK_STARTUP_DEFAULTS_VERSION')
        expect(source).toContain('const markedCurrent = markCacheStackStartupDefaultsCurrent(config)')
        expect(source).toContain('const familyDefaultsChanged = applyFamilyStartupDefaults(config, config.modelPath)')
        expect(source).toContain('if (bundleDefaultsChanged || migrated || familyDefaultsChanged || markedCurrent)')
        expect(source).toContain('markCacheStackStartupDefaultsCurrent(merged as Partial<ServerConfig>)')
        expect(source).toContain('cacheStackStartupDefaultsVersion: CACHE_STACK_STARTUP_DEFAULTS_VERSION')
    })

    it('create-session stamps incoming settings current before saved legacy migration', () => {
        const source = readFileSync('src/main/sessions.ts', 'utf8')
        const start = source.indexOf('private async _createSessionInner')
        const existing = source.indexOf('const existing = db.getSessionByModelPath(modelPath)', start)
        const beforeExisting = source.slice(start, existing)
        const existingBlock = source.slice(existing, source.indexOf('const id = uuidv4()', existing))

        expect(beforeExisting).toContain('markCacheStackStartupDefaultsCurrent(config)')
        expect(beforeExisting).not.toContain('applyCacheStackStartupDefaultMigration(config')
        expect(existingBlock).toContain('applyCacheStackStartupDefaultMigration(existingConfig, modelPath)')
        expect(existingBlock).toContain('const merged = { ...existingConfig, ...config, modelPath, host, port }')
        expect(existingBlock).toContain('markCacheStackStartupDefaultsCurrent(merged)')
    })

    it('session manager migrates stale no-prefix MiniMax-style batch tuple', () => {
        const source = readFileSync('src/main/sessions.ts', 'utf8')
        expect(source).toContain('staleNoPrefixBatchDefaults')
        expect(source).toContain('config.enablePrefixCache === false')
        expect(source).toContain('Number(config.maxNumSeqs) <= 8')
        expect(source).toContain('Number(config.prefillBatchSize) === 1024')
        expect(source).toContain('Number(config.completionBatchSize) === 1024')
        expect(source).toContain('stalePartialPagedCacheDefaults')
        expect(source).toContain('config.usePagedCache === false')
        expect(source).toContain('!stalePartialPagedCacheDefaults')
    })

    it('session manager migrates stale explicit none cache codec defaults back to auto', () => {
        const source = readFileSync('src/main/sessions.ts', 'utf8')
        expect(source).toContain('const zayaCacheMigrationTarget = isZayaCacheStackMigrationTarget(modelPath || config.modelPath)')
        expect(source).toContain('staleExplicitNoneCacheCodecDefaults')
        expect(source).toContain('zayaCacheMigrationTarget &&')
        expect(source).toContain("config.kvCacheQuantization === 'none'")
        expect(source).toContain('!staleExplicitNoneCacheCodecDefaults')
        expect(source).toContain("config.kvCacheQuantization = 'auto'")
    })

    it('stale explicit none migration is limited to the single-user cache-stack default shape', () => {
        const source = readFileSync('src/main/sessions.ts', 'utf8')
        const start = source.indexOf('const staleExplicitNoneCacheCodecDefaults =')
        const end = source.indexOf('if (', start)
        const block = source.slice(start, end)

        expect(block).toContain('zayaCacheMigrationTarget &&')
        expect(block).toContain('config.continuousBatching === true')
        expect(block).toContain('config.enablePrefixCache === true')
        expect(block).toContain('Number(config.maxNumSeqs) === 1')
        expect(block).toContain('Number(config.prefillBatchSize) === 512')
        expect(block).toContain('Number(config.completionBatchSize) === 512')
        expect(block).toContain('config.usePagedCache === true')
        expect(block).toContain('config.enableBlockDiskCache === true')
        expect(block).toContain("config.kvCacheQuantization === 'none'")
    })

    it('ZAYA sessions keep the qwen3 reasoning parser and model-owned no-thinking default', () => {
        const source = readFileSync('src/main/sessions.ts', 'utf8')
        expect(source).toContain('function isZayaCcaFamily')
        expect(source).toContain('if (isZayaCcaFamily(freshFamily))')
        expect(source).toContain("config.reasoningParser = freshConfig.reasoningParser || 'auto'")
        expect(source).toContain('delete config.defaultEnableThinking')
        expect(source).not.toContain("args.push('--default-enable-thinking', 'false')")
        expect(source).not.toContain('ZAYA default thinking reset from stale on to off')

        const out = preview(
            { defaultEnableThinking: true },
            { family: 'zaya', cacheType: 'hybrid', usePagedCache: true, reasoningParser: 'qwen3', defaultEnableThinking: false }
        )
        expect(hasFlag(out, '--reasoning-parser')).toBe(true)
        expect(getFlagValue(out, '--reasoning-parser')).toBe('qwen3')
        expect(hasFlag(out, '--default-enable-thinking')).toBe(false)
    })

    it('logLevel INFO (default) does not emit --log-level flag', () => {
        const out = preview({ logLevel: 'INFO' })
        expect(hasFlag(out, '--log-level')).toBe(false)
    })

    it('logLevel DEBUG emits --log-level DEBUG', () => {
        const out = preview({ logLevel: 'DEBUG' })
        expect(hasFlag(out, '--log-level')).toBe(true)
        expect(getFlagValue(out, '--log-level')).toBe('DEBUG')
    })

    it('corsOrigins * (default) does not emit --allowed-origins flag', () => {
        const out = preview({ corsOrigins: '*' })
        expect(hasFlag(out, '--allowed-origins')).toBe(false)
    })

    it('corsOrigins custom value emits --allowed-origins', () => {
        const out = preview({ corsOrigins: 'http://example.com' })
        expect(getFlagValue(out, '--allowed-origins')).toBe('http://example.com')
    })

    it('maxContextLength emits max prompt/context CLI flag when set', () => {
        const out = preview({ maxContextLength: 32768 })
        expect(getFlagValue(out, '--max-prompt-tokens')).toBe('32768')
    })

    it('default config has all new fields', () => {
        expect(DEFAULT_CONFIG.logLevel).toBe('INFO')
        expect(DEFAULT_CONFIG.corsOrigins).toBe('*')
        expect(DEFAULT_CONFIG.maxContextLength).toBe(0)
        expect(DEFAULT_CONFIG.enableJit).toBe(true)
        expect(DEFAULT_CONFIG.defaultTemperature).toBe(0)
        expect(DEFAULT_CONFIG.defaultTopP).toBe(0)
        expect(DEFAULT_CONFIG.defaultTopK).toBe(0)
        expect(DEFAULT_CONFIG.defaultMinP).toBe(0)
        expect(DEFAULT_CONFIG.defaultRepetitionPenalty).toBe(0)
        expect(DEFAULT_CONFIG.maxTokens).toBe(0)
        expect(DEFAULT_CONFIG.omniBackend).toBe('stage1')
    })

    it('source defaults leave global sampling unset so bundle defaults can win', () => {
        const source = readFileSync('src/renderer/src/components/sessions/SessionConfigForm.tsx', 'utf8')
        expect(source).toContain('defaultTemperature: 0')
        expect(source).toContain('defaultTopP: 0')
        expect(source).toContain('defaultRepetitionPenalty: 0')
    })

    it('database migration resets only exact old generic sampling defaults', () => {
        const source = readFileSync('src/main/database.ts', 'utf8')
        expect(source).toContain('migration_reset_generic_sampling_defaults_1_5_39')
        expect(source).toContain('parsed.defaultTemperature === 70')
        expect(source).toContain('parsed.defaultTopP === 95')
        expect(source).toContain('parsed.defaultRepetitionPenalty === 110')
        expect(source).toContain('parsed.defaultRepetitionPenalty = 0')
    })
})

describe('JIT Toggle', () => {
    it('enableJit false does not emit --enable-jit flag', () => {
        const out = preview({ enableJit: false })
        expect(hasFlag(out, '--enable-jit')).toBe(false)
    })

    it('enableJit true emits --enable-jit flag', () => {
        const out = preview({ enableJit: true })
        expect(hasFlag(out, '--enable-jit')).toBe(true)
    })

    it('deepseek-v4 detection suppresses --enable-jit even when saved config requests it', () => {
        const out = preview(
            { enableJit: true, maxNumSeqs: 64 },
            { family: 'deepseek-v4' },
        )

        expect(hasFlag(out, '--enable-jit')).toBe(false)
        expect(getFlagValue(out, '--max-num-seqs')).toBe('1')
    })

    it('TurboQuant/JANGTQ detection suppresses --enable-jit because engine skips mx.compile', () => {
        const out = preview(
            { enableJit: true },
            { family: 'minimax', isTurboQuant: true },
        )

        expect(hasFlag(out, '--enable-jit')).toBe(false)
    })

    it('multimodal/VLM detection suppresses --enable-jit because mlx-vlm streaming is not compile-safe', () => {
        const out = preview(
            { enableJit: true, isMultimodal: false },
            { family: 'zaya1-vl', isMultimodal: true },
        )

        expect(hasFlag(out, '--is-mllm')).toBe(true)
        expect(hasFlag(out, '--enable-jit')).toBe(false)
    })

    it('ZAYA typed CCA detection suppresses --enable-jit because cache path is faster uncompiled', () => {
        const out = preview(
            { enableJit: true },
            { family: 'zaya' },
        )

        expect(hasFlag(out, '--enable-jit')).toBe(false)
    })

    it('manual multimodal mode suppresses --enable-jit even without detection', () => {
        const out = preview({ enableJit: true, isMultimodal: true })

        expect(hasFlag(out, '--is-mllm')).toBe(true)
        expect(hasFlag(out, '--enable-jit')).toBe(false)
    })

    it('settings form surfaces DeepSeek-V4 JIT as effectively disabled', () => {
        const fs = require('fs')
        const form = fs.readFileSync(
            'src/renderer/src/components/sessions/SessionConfigForm.tsx',
            'utf-8',
        )
        const settings = fs.readFileSync(
            'src/renderer/src/components/sessions/SessionSettings.tsx',
            'utf-8',
        )
        const create = fs.readFileSync(
            'src/renderer/src/components/sessions/CreateSession.tsx',
            'utf-8',
        )
        const drawer = fs.readFileSync(
            'src/renderer/src/components/sessions/ServerSettingsDrawer.tsx',
            'utf-8',
        )

        expect(form).toContain('detectedFamily')
        expect(form).toContain('DeepSeek-V4 native composite cache')
        expect(settings).toContain('detectedFamily={detectedConfig?.family}')
        expect(create).toContain('detectedFamily={detectedFamily}')
        expect(drawer).toContain('detectedFamily={detectedFamily}')
    })

    it('settings form surfaces TurboQuant/JANGTQ JIT as effectively disabled', () => {
        const fs = require('fs')
        const form = fs.readFileSync(
            'src/renderer/src/components/sessions/SessionConfigForm.tsx',
            'utf-8',
        )
        const sessions = fs.readFileSync('src/main/sessions.ts', 'utf-8')
        const registry = fs.readFileSync('src/main/model-config-registry.ts', 'utf-8')

        expect(registry).toContain('isTurboQuant')
        expect(registry).toContain("weight_format === 'mxtq'")
        expect(form).toContain('detectedIsTurboQuant')
        expect(form).toContain('TurboQuant KV')
        expect(sessions).toContain('turboQuantActive')
        expect(sessions).toContain('TurboQuantKVCache uses custom cache objects')
    })

    it('settings form surfaces multimodal JIT as effectively disabled', () => {
        const fs = require('fs')
        const form = fs.readFileSync(
            'src/renderer/src/components/sessions/SessionConfigForm.tsx',
            'utf-8',
        )
        const settings = fs.readFileSync(
            'src/renderer/src/components/sessions/SessionSettings.tsx',
            'utf-8',
        )
        const sessions = fs.readFileSync('src/main/sessions.ts', 'utf-8')

        expect(form).toContain('detectedIsMultimodal')
        expect(form).toContain('multimodal/VLM models')
        expect(settings).toContain('detectedIsMultimodal={detectedConfig?.isMultimodal}')
        expect(sessions).toContain('mlx-vlm streaming path')
    })

    it('settings form surfaces ZAYA typed CCA JIT as effectively disabled', () => {
        const fs = require('fs')
        const form = fs.readFileSync(
            'src/renderer/src/components/sessions/SessionConfigForm.tsx',
            'utf-8',
        )
        const settings = fs.readFileSync(
            'src/renderer/src/components/sessions/SessionSettings.tsx',
            'utf-8',
        )
        const sessions = fs.readFileSync('src/main/sessions.ts', 'utf-8')

        expect(form).toContain('zayaCcaActive')
        expect(form).toContain('ZAYA typed CCA cache')
        expect(settings).toContain('zayaCcaActive')
        expect(sessions).toContain('ZAYA typed CCA cache is path-dependent')
    })

    it('settings form and launch code surface ZAYA typed CCA paged-cache requirement', () => {
        const fs = require('fs')
        const form = fs.readFileSync(
            'src/renderer/src/components/sessions/SessionConfigForm.tsx',
            'utf-8',
        )
        const settings = fs.readFileSync(
            'src/renderer/src/components/sessions/SessionSettings.tsx',
            'utf-8',
        )
        const sessions = fs.readFileSync('src/main/sessions.ts', 'utf-8')

        expect(form).toContain('zayaTypedCacheRequiresPaged')
        expect(form).toContain('ZAYA typed CCA cache requires paged cache while prefix cache is enabled')
        expect(settings).toContain('zayaTypedCacheRequiresPaged')
        expect(sessions).toContain('resolveCacheLaunchPolicy')
        expect(sessions).toContain('architectureRequiresPagedCache')
        expect(sessions).toContain('zayaCcaActive ||')
    })

    it('settings form and launch code surface one DSV4 native composite cache switch', () => {
        const fs = require('fs')
        const form = fs.readFileSync(
            'src/renderer/src/components/sessions/SessionConfigForm.tsx',
            'utf-8',
        )
        const settings = fs.readFileSync(
            'src/renderer/src/components/sessions/SessionSettings.tsx',
            'utf-8',
        )
        const sessions = fs.readFileSync('src/main/sessions.ts', 'utf-8')

        expect(form).toContain('dsv4CompositeRequiresPaged')
        expect(form).toContain('dsv4CompositeCacheOptIn')
        expect(form).toContain('DSV4 Native Composite Prefix Cache')
        expect(form).toContain('DSV4 CSA/HCA Pool Codec')
        expect(form).toContain('native composite prefix cache is off for this session')
        expect(form).toContain('cacheControlUpdatesForDsv4CompositeToggle')
        expect(form).toContain('cacheControlUpdatesForDsv4BlockDiskToggle')
        expect(form).toContain('applyDsv4CompositeCacheToggle')
        expect(form).not.toContain('DSV4 Native Cache')
        expect(form).not.toContain('DSV4 Composite Prefix Cache')
        expect(form).not.toContain('DSV4 Pool Quantization')
        expect(form).not.toContain('DSV4 Flash composite prefix cache is disabled')
        expect(form).not.toContain("dsv4Active ? applyDsv4CompositeCacheToggle(v) : applyCacheControlUpdates(cacheControlUpdatesForPagedToggle")
        expect(form).toContain("dsv4Active ? cacheControlUpdatesForDsv4BlockDiskToggle(v) : cacheControlUpdatesForBlockDiskToggle")
        expect(form).toContain('disabled={!dsv4Active && cachePolicy.pagedCacheDisabled}')
        expect(form).toContain('block size is fixed to 256 tokens')
        expect(form).toContain('checked={!!config.dsv4PrefixCache}')
        expect(form).not.toContain('checked={dsv4Active ? true : config.enablePrefixCache}')
        expect(settings).toContain('DSV4_PAGED_CACHE_BLOCK_SIZE = 256')
        expect(settings).toContain('dsv4PrefixCacheOptIn')
        expect(sessions).toContain('DSV4_PAGED_CACHE_BLOCK_SIZE = 256')
        expect(sessions).toContain('--dsv4-enable-prefix-cache')
        expect(sessions).toContain('composite prefix/paged/L2 cache disabled by default')
        expect(sessions).not.toContain('const prefixCacheOff = dsv4Active ? false')
    })

    it('settings form hides generic paged-cache warnings for the DSV4 native cache path', () => {
        const fs = require('fs')
        const form = fs.readFileSync(
            'src/renderer/src/components/sessions/SessionConfigForm.tsx',
            'utf-8',
        )

        expect(form).toContain('{!dsv4Active && !dsv4CompositeRequiresPaged && config.enableDiskCache &&')
        expect(form).toContain('{!dsv4Active && !dsv4CompositeRequiresPaged && !batchingOff && prefixOff &&')
        expect(form).toContain('Persist DeepSeek-V4 native SWA+CSA/HCA composite cache records to SSD')
        expect(form).toContain("cachePolicy.legacyDiskCacheUnavailableReason === 'paged-cache-active'")
        expect(form).toContain("cachePolicy.legacyDiskCacheUnavailableReason === 'architecture-requires-paged-cache'")
        expect(form).toContain('This is not generic paged KV')
        expect(form).toContain('DSV4 Native Composite Prefix Cache')
    })

    it('enableJit does not affect other flags', () => {
        const without = preview({ enableJit: false })
        const withJit = preview({ enableJit: true })
        // Only difference should be the --enable-jit flag
        const normalized1 = without.replace(/\s*\\\n\s*/g, ' ')
        const normalized2 = withJit.replace(/\s*\\\n\s*/g, ' ')
        expect(normalized2).toContain('--enable-jit')
        expect(normalized1).not.toContain('--enable-jit')
        // Both should have the same host/port/timeout etc
        expect(getFlagValue(without, '--host')).toBe(getFlagValue(withJit, '--host'))
        expect(getFlagValue(without, '--port')).toBe(getFlagValue(withJit, '--port'))
    })
})

describe('connectHost Resolution', () => {
    // Test the connectHost logic (0.0.0.0 → 127.0.0.1 for connections)
    function connectHost(host: string): string {
        return host === '0.0.0.0' ? '127.0.0.1' : host
    }

    it('resolves 0.0.0.0 to 127.0.0.1', () => {
        expect(connectHost('0.0.0.0')).toBe('127.0.0.1')
    })

    it('passes through 127.0.0.1 unchanged', () => {
        expect(connectHost('127.0.0.1')).toBe('127.0.0.1')
    })

    it('passes through localhost unchanged', () => {
        expect(connectHost('localhost')).toBe('localhost')
    })

    it('passes through custom IPs unchanged', () => {
        expect(connectHost('192.168.1.100')).toBe('192.168.1.100')
    })

    it('passes through hostnames unchanged', () => {
        expect(connectHost('my-server.local')).toBe('my-server.local')
    })
})

describe('Feature Interaction', () => {
    it('continuous batching off is a real master switch for LLM cache flags', () => {
        const out = preview({ continuousBatching: false, enablePrefixCache: true })
        expect(hasFlag(out, '--continuous-batching')).toBe(false)
        expect(hasFlag(out, '--no-continuous-batching')).toBe(true)
        expect(hasFlag(out, '--disable-prefix-cache')).toBe(true)
        expect(hasFlag(out, '--use-paged-cache')).toBe(false)
        expect(hasFlag(out, '--enable-block-disk-cache')).toBe(false)
    })

    it('continuous batching off is a real master switch for VLM cache flags', () => {
        const out = preview({ isMultimodal: true, continuousBatching: false, enablePrefixCache: true })
        expect(hasFlag(out, '--is-mllm')).toBe(true)
        expect(hasFlag(out, '--continuous-batching')).toBe(false)
        expect(hasFlag(out, '--no-continuous-batching')).toBe(true)
        expect(hasFlag(out, '--disable-prefix-cache')).toBe(true)
        expect(hasFlag(out, '--use-paged-cache')).toBe(false)
    })

    it('prefillBatchSize 0 omits flag (uses backend default 512)', () => {
        const out = preview({ prefillBatchSize: 0, enablePrefixCache: true })
        expect(hasFlag(out, '--prefill-batch-size')).toBe(false)
    })

    it('VLM with all caching features works together', () => {
        const out = preview({
            isMultimodal: true,
            continuousBatching: true,
            enablePrefixCache: true,
            usePagedCache: true,
            kvCacheQuantization: 'q8',
            enableBlockDiskCache: true,
        })
        expect(hasFlag(out, '--is-mllm')).toBe(true)
        expect(hasFlag(out, '--continuous-batching')).toBe(true)
        expect(hasFlag(out, '--use-paged-cache')).toBe(true)
        expect(hasFlag(out, '--kv-cache-quantization')).toBe(true)
        expect(hasFlag(out, '--enable-block-disk-cache')).toBe(true)
    })

    it('disabling prefix cache disables all dependent features', () => {
        const out = preview({
            enablePrefixCache: false,
            usePagedCache: true,
            kvCacheQuantization: 'q8',
            enableDiskCache: true,
            enableBlockDiskCache: true,
        })
        expect(hasFlag(out, '--disable-prefix-cache')).toBe(true)
        expect(hasFlag(out, '--use-paged-cache')).toBe(false)
        expect(hasFlag(out, '--kv-cache-quantization')).toBe(false)
        expect(hasFlag(out, '--enable-disk-cache')).toBe(false)
        expect(hasFlag(out, '--enable-block-disk-cache')).toBe(false)
    })

    it('speculative decoding with all options set', () => {
        const out = preview({
            speculativeModel: 'draft-model',
            numDraftTokens: 7,
            defaultTemperature: 80,
            defaultTopP: 90,
            embeddingModel: 'embed-model',
            servedModelName: 'my-model',
        })
        expect(getFlagValue(out, '--speculative-model')).toBe('draft-model')
        expect(getFlagValue(out, '--num-draft-tokens')).toBe('7')
        expect(hasFlag(out, '--default-temperature')).toBe(false)
        expect(hasFlag(out, '--default-top-p')).toBe(false)
        expect(getFlagValue(out, '--embedding-model')).toBe('embed-model')
        expect(getFlagValue(out, '--served-model-name')).toBe('my-model')
    })

    it('optional model-specific features remain disabled by default', () => {
        const out = preview()
        expect(hasFlag(out, '--speculative-model')).toBe(false)
        expect(hasFlag(out, '--num-draft-tokens')).toBe(false)
        expect(hasFlag(out, '--embedding-model')).toBe(false)
        expect(hasFlag(out, '--served-model-name')).toBe(false)
        expect(hasFlag(out, '--default-enable-thinking')).toBe(false)
        expect(hasFlag(out, '--omni-backend')).toBe(false)
    })

    it('tool parser emitted without auto-tool-choice (matches buildArgs)', () => {
        // buildArgs emits --tool-call-parser independently of --enable-auto-tool-choice
        const out = preview(
            { enableAutoToolChoice: false, toolCallParser: 'llama' },
        )
        expect(hasFlag(out, '--tool-call-parser')).toBe(true)
        expect(getFlagValue(out, '--tool-call-parser')).toBe('llama')
        expect(hasFlag(out, '--enable-auto-tool-choice')).toBe(false)
    })

    it('detected tool parser emitted without auto-tool-choice', () => {
        const out = preview(
            { enableAutoToolChoice: false, toolCallParser: 'auto' },
            { toolParser: 'qwen' }
        )
        expect(getFlagValue(out, '--tool-call-parser')).toBe('qwen')
        expect(hasFlag(out, '--enable-auto-tool-choice')).toBe(false)
    })

    it('MCP tools honor explicit prefix cache disable', () => {
        const out = preview({
            enablePrefixCache: false,
            enableAutoToolChoice: true,
            mcpConfig: '/path/mcp.json',
            toolCallParser: 'hermes',
        })
        expect(hasFlag(out, '--disable-prefix-cache')).toBe(true)
        expect(hasFlag(out, '--mcp-config')).toBe(true)
    })

    it('noMemoryAwareCache suppresses memory-aware flags', () => {
        const out = preview({
            enablePrefixCache: true,
            noMemoryAwareCache: true,
            cacheMemoryMb: 2048,
            cacheMemoryPercent: 30,
            cacheTtlMinutes: 60,
            prefixCacheSize: 200,
        })
        expect(hasFlag(out, '--no-memory-aware-cache')).toBe(true)
        expect(hasFlag(out, '--prefix-cache-size')).toBe(true)
        // Memory-aware flags must NOT appear
        expect(hasFlag(out, '--cache-memory-mb')).toBe(false)
        expect(hasFlag(out, '--cache-memory-percent')).toBe(false)
        expect(hasFlag(out, '--cache-ttl-minutes')).toBe(false)
    })

    it('CLI preview includes prefixCacheMaxBytes like session launch args', () => {
        const fs = require('fs')
        const previewSource = fs.readFileSync(
            'src/renderer/src/components/sessions/SessionSettings.tsx',
            'utf-8',
        )
        const launchSource = fs.readFileSync('src/main/sessions.ts', 'utf-8')

        expect(previewSource).toContain('--prefix-cache-max-bytes')
        expect(launchSource).toContain('--prefix-cache-max-bytes')
    })

    it('cacheMemoryPercent default 15 emits 0.15 when legacy memory cache is active', () => {
        const out = preview({ enablePrefixCache: true, cacheMemoryPercent: 15, usePagedCache: false })
        expect(getFlagValue(out, '--cache-memory-percent')).toBe('0.15')
    })

    it('omits memory-aware cache budget flags when paged cache is active', () => {
        const out = preview({
            enablePrefixCache: true,
            usePagedCache: true,
            cacheMemoryMb: 4096,
            cacheMemoryPercent: 35,
        })
        expect(hasFlag(out, '--use-paged-cache')).toBe(true)
        expect(hasFlag(out, '--cache-memory-mb')).toBe(false)
        expect(hasFlag(out, '--cache-memory-percent')).toBe(false)
    })

    it('defaultTopP minimum boundary stays out of startup CLI', () => {
        const out = preview({ defaultTopP: 1 })
        expect(hasFlag(out, '--default-top-p')).toBe(false)
    })

    it('numDraftTokens 0 with speculative model omits draft tokens flag', () => {
        // numDraftTokens 0 is falsy → condition fails → flag omitted → Python uses default (3)
        const out = preview({ speculativeModel: 'draft-model', numDraftTokens: 0 })
        expect(hasFlag(out, '--speculative-model')).toBe(true)
        expect(hasFlag(out, '--num-draft-tokens')).toBe(false)
    })

    it('empty diskCacheDir with enableDiskCache does not emit --disk-cache-dir', () => {
        const out = preview({ enablePrefixCache: true, enableDiskCache: true, diskCacheDir: '', usePagedCache: false })
        expect(hasFlag(out, '--enable-disk-cache')).toBe(true)
        expect(hasFlag(out, '--disk-cache-dir')).toBe(false)
    })

    it('enableDiskCache suppressed when usePagedCache is on', () => {
        const out = preview({ enablePrefixCache: true, enableDiskCache: true, usePagedCache: true })
        expect(hasFlag(out, '--enable-disk-cache')).toBe(false)
    })

    it('VLM + speculative decoding both emit flags (Python gates server-side)', () => {
        const out = preview({
            isMultimodal: true,
            speculativeModel: 'draft-model',
            numDraftTokens: 5,
        })
        expect(hasFlag(out, '--is-mllm')).toBe(true)
        expect(hasFlag(out, '--speculative-model')).toBe(true)
        expect(getFlagValue(out, '--num-draft-tokens')).toBe('5')
    })

    it('speculative decoding + continuous batching + embedding model combined', () => {
        const out = preview({
            speculativeModel: 'draft-model',
            numDraftTokens: 4,
            continuousBatching: true,
            embeddingModel: 'embed-model',
            defaultTemperature: 80,
            defaultTopP: 90,
        })
        expect(hasFlag(out, '--continuous-batching')).toBe(true)
        expect(getFlagValue(out, '--speculative-model')).toBe('draft-model')
        expect(getFlagValue(out, '--num-draft-tokens')).toBe('4')
        expect(getFlagValue(out, '--embedding-model')).toBe('embed-model')
        expect(hasFlag(out, '--default-temperature')).toBe(false)
        expect(hasFlag(out, '--default-top-p')).toBe(false)
    })
})

describe('Update Checker', () => {
    // Tests for compareVersions logic (extracted from update-checker.ts)
    function compareVersions(current: string, latest: string): boolean {
        const a = current.split('.').map(Number)
        const b = latest.split('.').map(Number)
        for (let i = 0; i < Math.max(a.length, b.length); i++) {
            const av = a[i] || 0
            const bv = b[i] || 0
            if (bv > av) return true
            if (bv < av) return false
        }
        return false
    }

    it('detects newer major version', () => {
        expect(compareVersions('1.0.0', '2.0.0')).toBe(true)
    })

    it('detects newer minor version', () => {
        expect(compareVersions('1.0.0', '1.1.0')).toBe(true)
    })

    it('detects newer patch version', () => {
        expect(compareVersions('1.1.0', '1.1.1')).toBe(true)
    })

    it('returns false when versions are equal', () => {
        expect(compareVersions('1.1.0', '1.1.0')).toBe(false)
    })

    it('returns false when current is newer', () => {
        expect(compareVersions('2.0.0', '1.9.9')).toBe(false)
    })

    it('handles different version lengths', () => {
        expect(compareVersions('1.0', '1.0.1')).toBe(true)
        expect(compareVersions('1.0.1', '1.0')).toBe(false)
    })

    it('handles major version jump', () => {
        expect(compareVersions('0.3.0', '1.1.0')).toBe(true)
    })

    it('handles zero versions', () => {
        expect(compareVersions('0.0.0', '0.0.1')).toBe(true)
        expect(compareVersions('0.0.0', '0.0.0')).toBe(false)
    })
})

// =============================================================================
// Phase 4: connectHost and CORS verification
// =============================================================================

describe('URL construction uses connectHost', () => {
    // Replica of the connectHost function from sessions.ts
    function connectHost(host: string): string {
        return host === '0.0.0.0' ? '127.0.0.1' : host
    }

    it('all URL construction sites use connectHost — 0.0.0.0 maps to 127.0.0.1', () => {
        // The key invariant: 0.0.0.0 (bind-all) is never used in outgoing URLs
        expect(connectHost('0.0.0.0')).toBe('127.0.0.1')
    })

    it('connectHost preserves specific IPs', () => {
        expect(connectHost('127.0.0.1')).toBe('127.0.0.1')
        expect(connectHost('192.168.1.50')).toBe('192.168.1.50')
        expect(connectHost('10.0.0.1')).toBe('10.0.0.1')
    })

    it('connectHost preserves hostnames', () => {
        expect(connectHost('my-server.local')).toBe('my-server.local')
        expect(connectHost('localhost')).toBe('localhost')
    })

    it('health URL construction uses connectHost', () => {
        const host = '0.0.0.0'
        const port = 8092
        const healthUrl = `http://${connectHost(host)}:${port}/health`
        expect(healthUrl).toBe('http://127.0.0.1:8092/health')
        expect(healthUrl).not.toContain('0.0.0.0')
    })
})

describe('CORS credentials logic', () => {
    // Replica of the CORS logic from cli.py serve_command
    function corsConfig(allowedOrigins: string): { origins: string[], credentials: boolean } {
        const origins = allowedOrigins.split(',').map(o => o.trim()).filter(o => o.length > 0)
        const hasWildcard = origins.includes('*')
        return {
            origins,
            credentials: !hasWildcard,
        }
    }

    it('credentials are false when wildcard origin is used', () => {
        const config = corsConfig('*')
        expect(config.credentials).toBe(false)
        expect(config.origins).toEqual(['*'])
    })

    it('credentials are true when specific origins are listed', () => {
        const config = corsConfig('http://localhost:3000,http://example.com')
        expect(config.credentials).toBe(true)
        expect(config.origins).toEqual(['http://localhost:3000', 'http://example.com'])
    })

    it('credentials are false when wildcard is among specific origins', () => {
        const config = corsConfig('http://localhost:3000,*')
        expect(config.credentials).toBe(false)
    })

    it('empty string produces no origins', () => {
        const config = corsConfig('')
        expect(config.origins).toEqual([])
    })
})

// ═══════════════════════════════════════════════════════════════════════════════
// Phase 6: Settings → CLI Round-Trip Completeness
// ═══════════════════════════════════════════════════════════════════════════════

describe('Settings → CLI Round-Trip Completeness', () => {
    // All SessionConfig keys (from the interface defined at the top of this file)
    const ALL_CONFIG_KEYS: (keyof SessionConfig)[] = [
        'host', 'port', 'apiKey', 'rateLimit', 'timeout',
        'maxNumSeqs', 'prefillBatchSize', 'prefillStepSize', 'completionBatchSize',
        'continuousBatching', 'enablePrefixCache', 'prefixCacheSize', 'prefixCacheMaxBytes',
        'cacheMemoryMb', 'cacheMemoryPercent', 'cacheTtlMinutes', 'noMemoryAwareCache',
        'usePagedCache', 'pagedCacheBlockSize', 'maxCacheBlocks',
        'kvCacheQuantization', 'kvCacheGroupSize', 'omniBackend',
        'enableDiskCache', 'diskCacheMaxGb', 'diskCacheDir',
        'enableBlockDiskCache', 'blockDiskCacheMaxGb', 'blockDiskCacheDir',
        'streamInterval', 'maxTokens',
        'mcpConfig', 'mcpEnabledServers', 'mcpDisabledServers', 'mcpEnabledTools', 'mcpDisabledTools',
        'enableAutoToolChoice', 'toolCallParser', 'reasoningParser',
        'isMultimodal', 'servedModelName',
        'speculativeModel', 'numDraftTokens',
        'smelt', 'smeltExperts', 'flashMoe', 'flashMoeSlotBank', 'flashMoePrefetch', 'flashMoeIoSplit',
        'defaultTemperature', 'defaultTopP', 'defaultTopK', 'defaultMinP', 'defaultRepetitionPenalty', 'defaultMaxNewTokens', 'defaultEnableThinking',
        'dsv4PrefixCache',
        'nativeMtpMode', 'nativeMtpDepth', 'nativeMtpDepthOverride',
        'embeddingModel', 'additionalArgs',
        'enableJit', 'logLevel', 'corsOrigins', 'maxContextLength',
    ]

    // Collect all config keys that appear in at least one test in this file
    // by checking that setting them produces a CLI flag or expected behavior.
    // This is a structural meta-test: ensure coverage.
    it('every SessionConfig field is listed in the completeness check', () => {
        const interfaceKeys = Object.keys(DEFAULT_CONFIG) as (keyof SessionConfig)[]
        // Plus enableAutoToolChoice and isMultimodal which are optional (not in defaults)
        const fullSet = new Set([...interfaceKeys, 'enableAutoToolChoice', 'isMultimodal'])
        const checkedSet = new Set(ALL_CONFIG_KEYS)

        for (const key of fullSet) {
            expect(checkedSet.has(key), `SessionConfig key "${key}" missing from completeness list`).toBe(true)
        }
        for (const key of checkedSet) {
            expect(fullSet.has(key), `Completeness list has unknown key "${key}"`).toBe(true)
        }
    })

    it('default config produces the single-sequence cache-stack flags', () => {
        const out = preview()
        const normalized = out.replace(/\s*\\\n\s*/g, ' ')

        // Defaults should NOT produce these flags:
        expect(normalized).not.toContain('--api-key')
        expect(normalized).not.toContain('VLLM_API_KEY')  // apiKey is empty
        expect(normalized).not.toContain('--rate-limit')     // rateLimit is 0
        expect(normalized).not.toContain('--is-mllm')        // isMultimodal is undefined/false
        expect(normalized).not.toContain('--disable-prefix-cache')  // cache stack is enabled by default
        expect(normalized).not.toContain('--enable-disk-cache')     // paged cache uses block L2, not legacy prompt L2
        expect(normalized).not.toContain('--speculative-model')     // no speculative model
        expect(normalized).not.toContain('--embedding-model')       // empty
        expect(normalized).not.toContain('--log-level')             // INFO is default (not emitted)
        expect(normalized).not.toContain('--allowed-origins')       // * is default (not emitted)
        expect(normalized).not.toContain('--max-prompt-tokens')     // unset by default; explicit user value emits it
        expect(normalized).not.toContain('--default-temperature')   // request/CLI/bundle metadata resolve sampling
        expect(normalized).not.toContain('--default-top-p')         // do not poison bundles with generic UI defaults
        expect(normalized).not.toContain('--default-repetition-penalty')

        // Defaults SHOULD produce these flags:
        expect(normalized).toContain('--host')
        expect(normalized).toContain('--port')
        expect(normalized).toContain('--timeout')
        expect(normalized).not.toContain('--max-tokens')
        expect(normalized).toContain('--continuous-batching')
        expect(normalized).toContain('--use-paged-cache')
        expect(normalized).toContain('--enable-block-disk-cache')
        expect(normalized).not.toContain('--default-temperature')
        expect(normalized).not.toContain('--default-top-p')
        expect(normalized).not.toContain('--default-repetition-penalty')
        expect(normalized).toContain('--enable-jit')
    })

    it('server startup generation defaults are model-owned and not editable sliders', () => {
        const source = readFileSync('src/renderer/src/components/sessions/SessionConfigForm.tsx', 'utf8')
        expect(source).toContain('Generation defaults are resolved by the engine from generation_config.json/jang_config')
        expect(source).toContain('label="Max Context Tokens"')
        expect(source).not.toContain('label="Default Temperature"')
        expect(source).not.toContain('label="Default Top-P"')
        expect(source).not.toContain('label="Default Top-K"')
        expect(source).not.toContain('label="Default Min-P"')
        expect(source).not.toContain('label="Default Repetition Penalty"')
        expect(source).not.toContain('label="Default Max Tokens"')
    })

    it('Max Context Tokens can be manually typed while Auto is active', () => {
        const source = readFileSync('src/renderer/src/components/sessions/SessionConfigForm.tsx', 'utf8')
        const sliderStart = source.indexOf('export function SliderField')
        const sliderEnd = source.indexOf('\nexport ', sliderStart + 1)
        const sliderBody = source.slice(sliderStart, sliderEnd > 0 ? sliderEnd : undefined)
        const numberInputStart = sliderBody.indexOf('type="number"')
        const numberInputEnd = sliderBody.indexOf('/>', numberInputStart)
        const numberInput = sliderBody.slice(numberInputStart, numberInputEnd)

        expect(sliderBody).toContain('const isUnlimited = allowUnlimited && value === unlimitedValue')
        expect(numberInput).not.toContain('disabled || isUnlimited')
        expect(numberInput).toContain('disabled={disabled}')
        expect(sliderBody).toContain('onChange(isUnlimited ? unlimitedValue : defaultValue)')
    })

    it('all local session settings surfaces pass detected model context to Max Context Tokens', () => {
        const createSource = readFileSync('src/renderer/src/components/sessions/CreateSession.tsx', 'utf8')
        const drawerSource = readFileSync('src/renderer/src/components/sessions/ServerSettingsDrawer.tsx', 'utf8')
        const settingsSource = readFileSync('src/renderer/src/components/sessions/SessionSettings.tsx', 'utf8')

        expect(createSource).toContain('detectedMaxContext={detectedMaxContext}')
        expect(drawerSource).toContain('detectedMaxContext={detectedMaxContext}')
        expect(settingsSource).toContain('detectedMaxContext={detectedConfig?.maxContextLength}')
        expect(settingsSource).toContain('maxContextLength?: number')
    })

    it('JANGTQ router top-k override is not exposed through settings UI or launch env', () => {
        const formSource = readFileSync('src/renderer/src/components/sessions/SessionConfigForm.tsx', 'utf8')
        const settingsSource = readFileSync('src/renderer/src/components/sessions/SessionSettings.tsx', 'utf8')
        const sessionsSource = readFileSync('src/main/sessions.ts', 'utf8')
        expect(formSource).not.toContain('JANGTQ Active Experts Override')
        expect(formSource).not.toContain('jangtqTopKOverrideAllowed')
        expect(settingsSource).not.toContain('JANGTQ_TOPK_OVERRIDE')
        expect(settingsSource).not.toContain('jangtqTopKOverrideAllowed')
        expect(settingsSource).not.toContain('topKOverrideBlockedByFamily')
        expect(sessionsSource).toContain('delete spawnEnv.JANGTQ_TOPK_OVERRIDE')
        expect(sessionsSource).not.toContain('spawnEnv.JANGTQ_TOPK_OVERRIDE =')
    })

    it('chat settings renders disabled top-k sentinel as Off instead of raw 0 or -1', () => {
        const source = readFileSync('src/renderer/src/components/chat/ChatSettings.tsx', 'utf8')
        expect(source).toContain('formatTopK')
        expect(source).toContain("return 'Off'")
        expect(source).toContain('format={formatTopK}')
    })

    it('JANGTQ acceleration toggle is not exposed as a user setting', () => {
        const formSource = readFileSync('src/renderer/src/components/sessions/SessionConfigForm.tsx', 'utf8')
        const settingsSource = readFileSync('src/renderer/src/components/sessions/SessionSettings.tsx', 'utf8')
        const sessionsSource = readFileSync('src/main/sessions.ts', 'utf8')
        const perfSource = readFileSync('src/renderer/src/components/sessions/PerformancePanel.tsx', 'utf8')

        expect(formSource).not.toContain('JANGTQ MPP/NAX TensorOps')
        expect(formSource).not.toContain("onChange('jangtqMppNax'")
        expect(settingsSource).not.toContain('--jangtq-mpp-nax')
        expect(sessionsSource).not.toContain('--jangtq-mpp-nax')
        expect(sessionsSource).toContain('delete spawnEnv.JANGTQ_MPP_NAX')
        expect(sessionsSource).toContain('delete spawnEnv.JANGTQ_MPP_NAX_DISABLE')
        expect(sessionsSource).toContain('delete spawnEnv.JANGTQ_MPP_NAX_STRICT')
        expect(sessionsSource).toContain('delete spawnEnv.JANGTQ_MPP_DENSE')
        expect(sessionsSource).toContain('delete spawnEnv.JANGTQ_MPP_DENSE_STRICT')
        expect(sessionsSource).toContain('delete spawnEnv.JANGTQ_DISABLE_DSV4_STREAM_LOAD')
        expect(sessionsSource).toContain('delete spawnEnv.JANGTQ_DISABLE_DSV4_FAST_LOAD')
        expect(sessionsSource).toContain('delete spawnEnv.VMLX_DENSE_STRICT_LANE')
        expect(sessionsSource).toContain('delete spawnEnv.VMLX_DSV4_FAST_LOAD_DISABLE')
        expect(sessionsSource).toContain('delete spawnEnv.VMLINUX_DENSE_STRICT_LANE')
        expect(sessionsSource).toContain('delete spawnEnv.VMLINUX_DSV4_FAST_LOAD_DISABLE')
        expect(sessionsSource).toContain('engine_child_probe=')
        expect(perfSource).not.toContain('jangtq_mpp_nax?:')
        expect(perfSource).not.toContain('JANGTQ MPP/NAX')
    })

    it('DSV4 timeout default is wired through launch, chat IPC, and gateway proxy', () => {
        const sessionsSource = readFileSync('src/main/sessions.ts', 'utf8')
        const chatSource = readFileSync('src/main/ipc/chat.ts', 'utf8')
        const gatewaySource = readFileSync('src/main/api-gateway.ts', 'utf8')

        expect(sessionsSource).toContain('DSV4_DEFAULT_TIMEOUT_SECONDS = 900')
        expect(sessionsSource).toContain('effectiveSessionTimeoutSeconds')
        expect(chatSource).toContain('effectiveDsv4RequestTimeoutSeconds')
        expect(gatewaySource).toContain('effectiveGatewayProxyTimeoutMs')
    })

    it('mutual exclusion: disk cache NOT emitted when paged cache is active', () => {
        // enableDiskCache is gated by !(usePagedCache) in buildCommandPreview
        const out = preview({
            enablePrefixCache: true,
            enableDiskCache: true,
            diskCacheMaxGb: 20,
            diskCacheDir: '/tmp/cache',
            usePagedCache: true,
        })
        const normalized = out.replace(/\s*\\\n\s*/g, ' ')
        expect(normalized).not.toContain('--enable-disk-cache')
        expect(normalized).not.toContain('--disk-cache-dir')
        expect(normalized).not.toContain('--disk-cache-max-gb')
        // But paged cache flags should be present
        expect(normalized).toContain('--use-paged-cache')
    })

    it('mutual exclusion: block disk cache only emitted when paged cache is active', () => {
        // enableBlockDiskCache requires usePagedCache
        const out = preview({
            enablePrefixCache: true,
            enableBlockDiskCache: true,
            blockDiskCacheMaxGb: 50,
            blockDiskCacheDir: '/tmp/blocks',
            usePagedCache: true,
        })
        const normalized = out.replace(/\s*\\\n\s*/g, ' ')
        expect(normalized).toContain('--enable-block-disk-cache')
        expect(normalized).toContain('--block-disk-cache-dir')

        // Without paged cache, block disk cache should NOT appear. The UI
        // toggle writes paged=true before launch when the user turns block L2 on.
        const out2 = preview({
            enablePrefixCache: true,
            enableBlockDiskCache: true,
            blockDiskCacheMaxGb: 50,
            usePagedCache: false,
        })
        const normalized2 = out2.replace(/\s*\\\n\s*/g, ' ')
        expect(normalized2).not.toContain('--enable-block-disk-cache')
    })

    it('continuous batching off suppresses prefix cache even if prefix toggle is still on', () => {
        const out = preview({
            enablePrefixCache: true,
            continuousBatching: false,
        })
        const normalized = out.replace(/\s*\\\n\s*/g, ' ')
        expect(normalized).not.toContain('--continuous-batching')
        expect(normalized).toContain('--no-continuous-batching')
        expect(normalized).toContain('--disable-prefix-cache')
        expect(normalized).not.toContain('--use-paged-cache')
        expect(normalized).not.toContain('--enable-block-disk-cache')
    })

    it('prefix cache disabled suppresses all cache sub-flags', () => {
        const out = preview({
            enablePrefixCache: false,
            usePagedCache: true,        // should be suppressed
            kvCacheQuantization: 'q8',  // should be suppressed
            enableDiskCache: true,      // should be suppressed
        })
        const normalized = out.replace(/\s*\\\n\s*/g, ' ')
        expect(normalized).toContain('--disable-prefix-cache')
        expect(normalized).not.toContain('--use-paged-cache')
        expect(normalized).not.toContain('--kv-cache-quantization')
        expect(normalized).not.toContain('--enable-disk-cache')
    })

    it('cache TTL only emitted without paged cache', () => {
        // cacheTtlMinutes gated by !(usePagedCache)
        const withPaged = preview({
            enablePrefixCache: true,
            cacheTtlMinutes: 30,
            usePagedCache: true,
            noMemoryAwareCache: false,
        })
        expect(withPaged.replace(/\s*\\\n\s*/g, ' ')).not.toContain('--cache-ttl-minutes')

        const withoutPaged = preview({
            enablePrefixCache: true,
            cacheTtlMinutes: 30,
            usePagedCache: false,
            noMemoryAwareCache: false,
        })
        expect(withoutPaged.replace(/\s*\\\n\s*/g, ' ')).toContain('--cache-ttl-minutes')
    })
})
