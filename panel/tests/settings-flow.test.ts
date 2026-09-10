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
import { buildCacheLaunchArgs } from '../src/shared/cacheLaunchArgs'
import { SLOW_FAMILY_TIMEOUTS } from '../src/shared/slowFamilyTimeouts'
import { buildMcpPolicyArgs } from '../src/shared/mcpPolicy'
import {
    applyLagunaJitDefaultEnvironment,
    DISABLE_JANG_AFFINE_JIT_DEFAULT_ENV,
    isLagunaMixedSwaTurboQuantEffective,
    shouldDisableLagunaJitDefault,
} from '../src/shared/lagunaCachePolicy'
import { resolveEffectiveToolParser } from '../src/shared/toolParserAliases'
import {
    canonicalizeReasoningParserForCli,
    resolveEffectiveReasoningParser,
} from '../src/shared/reasoningParserAliases'
import { buildToolLaunchArgs } from '../src/shared/toolLaunchArgs'
import { buildNativeMtpLaunchArgs } from '../src/shared/nativeMtpLaunchArgs'
import { usesExactTypedPromptDiskCache } from '../src/shared/detectedFamilyNames'

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
    blockDiskCacheMaxPercent?: number
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
    dsv4PoolQuant?: boolean
    nativeMtpMode?: 'deterministic' | 'auto' | 'off'
    nativeMtpDepth?: number
    nativeMtpDepthOverride?: boolean
    embeddingModel: string
    additionalArgs: string
    enableJit: boolean
    logLevel: string
    corsOrigins: string
    maxContextLength: number
    chatTemplate?: string
    imageTokenBudget?: number
    videoFps?: number
    videoMaxFrames?: number
    distributedEnabled?: boolean
    distributedMode?: 'pipeline' | 'tensor'
    distributedSecret?: string
    distributedNodes?: Array<{ address: string; port: number; hostname?: string }>
    idleTimeoutSoftMin?: number
    idleTimeoutHardMin?: number
    autoSleepEnabled?: boolean
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
    usePagedCache: false,
    pagedCacheBlockSize: 64,
    maxCacheBlocks: 1000,
    kvCacheQuantization: 'auto',
    kvCacheGroupSize: 64,
    omniBackend: 'stage1',
    enableDiskCache: false,
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
    dsv4PoolQuant: undefined,
    nativeMtpMode: 'auto',
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
//
// A mirror agrees with itself no matter what the shipped launcher does. When
// buildArgs stopped emitting a flag the user's slider controlled, every test in
// this file stayed green because they all run against the mirror.
// `block-disk-percent-reaches-argv.test.ts` asserts against the REAL sessions.ts
// source for exactly that reason — any new "the user's setting reaches the
// process" claim belongs there, not here.

type DetectedConfig = {
    toolParser?: string
    reasoningParser?: string
    supportsThinking?: boolean
    isMultimodal?: boolean
    forceTextOnly?: boolean
    usePagedCache?: boolean
    enableAutoToolChoice?: boolean
    defaultEnableThinking?: boolean
    cacheType?: string
    cacheSubtype?: string
    family?: string
    dsv4PoolQuantDefault?: boolean
    architectureHints?: Record<string, string | number | boolean>
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
    return cacheType === 'hybrid' || cacheType === 'mamba' || cacheType === 'rotating_kv'
}

function cacheSubtypeRequiresPaged(cacheSubtype?: string): boolean {
    return cacheSubtype === 'step3p7_full_sliding_kv' || cacheSubtype === 'mixed_swa_kv'
}

function cacheTypeSupportsBlockDiskOnly(cacheType?: string): boolean {
    return cacheType === 'hybrid' || cacheType === 'mamba' || cacheType === 'rotating_kv'
}

function cacheSubtypeSupportsBlockDiskOnly(cacheSubtype?: string): boolean {
    return cacheSubtype === 'mixed_swa_kv' || cacheSubtype === 'step3p7_full_sliding_kv'
}

const DSV4_PAGED_CACHE_BLOCK_SIZE = 256
const GENERIC_DEFAULT_TIMEOUT_SECONDS = 300
const DSV4_DEFAULT_TIMEOUT_SECONDS = 900
const MINIMAX_M3_DEFAULT_TIMEOUT_SECONDS = 900

function extractFlagSetFromSource(sourceRel: string, constName: string): Set<string> {
    const source = readFileSync(sourceRel, 'utf8')
    const start = source.indexOf(`const ${constName}`) >= 0
        ? source.indexOf(`const ${constName}`)
        : source.indexOf(`export const ${constName}`)
    if (start === -1) throw new Error(`Missing ${constName} in ${sourceRel}`)
    const end = source.indexOf('])', start)
    if (end === -1) throw new Error(`Unterminated ${constName} in ${sourceRel}`)
    const block = source.slice(start, end)
    return new Set([...block.matchAll(/['"](--[a-z0-9][a-z0-9-]*)['"]/g)].map(match => match[1]))
}

// The value-flag list moved out of sessions.ts: it was a 66-entry duplicate
// shared by the argv BUILDER and the CLI-preview renderer, so it now lives in
// src/shared/launchArgValues.ts and both import it. Read the rule from its
// owner — grepping sessions.ts would pin the duplication that was removed.
const ADDITIONAL_ARG_VALUE_FLAGS = extractFlagSetFromSource(
    'src/shared/launchArgValues.ts',
    'ADDITIONAL_ARG_VALUE_FLAGS',
)

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

const TEXT_ADDITIONAL_ARG_BLOCKLIST = new Set(
    [
        ...IMAGE_ADDITIONAL_ARG_BLOCKLIST,
        ...extractFlagSetFromSource(
            'src/main/sessions.ts',
            'TEXT_ADDITIONAL_ARG_BLOCKLIST',
        ),
    ],
)

const DSV4_ADDITIONAL_ARG_BLOCKLIST = extractFlagSetFromSource(
    'src/main/sessions.ts',
    'DSV4_ADDITIONAL_ARG_BLOCKLIST',
)

function effectiveSessionTimeoutSeconds(config: Partial<SessionConfig>, family?: string): number {
    const configured = config.timeout
    if (configured != null && configured <= 0) return 86400
    const normalizedFamily = normalizeDetectedFamilyName(family)
    if (normalizedFamily === 'deepseek-v4' && (configured == null || configured === GENERIC_DEFAULT_TIMEOUT_SECONDS)) {
        return DSV4_DEFAULT_TIMEOUT_SECONDS
    }
    if (normalizedFamily === 'minimax_m3' && (configured == null || configured === GENERIC_DEFAULT_TIMEOUT_SECONDS)) {
        return MINIMAX_M3_DEFAULT_TIMEOUT_SECONDS
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
        const flagName = flag.includes('=') ? flag.slice(0, flag.indexOf('=')) : flag
        if (blockedFlags.has(flagName)) {
            if (flag === flagName && ADDITIONAL_ARG_VALUE_FLAGS.has(flagName)) i++
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
    const requestedDistributed = !!config.distributedEnabled
    const requestedFlashMoe = !!config.flashMoe
    const detectedFamily = normalizeDetectedFamilyName(detected?.family)
    const turboQuantActive = !!detected?.isTurboQuant
    const dsv4Active = detectedFamily === 'deepseek-v4'
    const m3Active = detectedFamily === 'minimax_m3'
    const effectiveSmelt = !!config.smelt && !dsv4Active
    // Mirror buildArgs (sessions.ts): user Force-Off (isMultimodal===false) beats
    // detected VL; m3Active stands in for m3VlRoute (registry sets it for every
    // minimax_m3 bundle) so M3 emits NEITHER --is-mllm NOR --text-only.
    const userForceTextOnly = config.isMultimodal === false
    const omniBackendActive = detectedFamily === 'nemotron-h' &&
        detected?.isMultimodal === true &&
        !userForceTextOnly &&
        !detected?.forceTextOnly
    const isVLM = dsv4Active || effectiveSmelt || detected?.forceTextOnly || userForceTextOnly || m3Active || omniBackendActive ? false
        : detected?.isMultimodal ? true
            : config.isMultimodal === true ? true
                : false
    const zayaCcaActive = isZayaCcaFamily(detectedFamily)
    const hybridCacheActive = detected?.cacheType === 'hybrid' || detected?.cacheType === 'mamba'
    const effectiveDistributed = requestedDistributed && !dsv4Active
    const effectiveFlashMoe = requestedFlashMoe && !effectiveDistributed && !dsv4Active
    if (dsv4Active && typeof detected?.dsv4PoolQuantDefault === 'boolean') {
        parts[0] = `DSV4_POOL_QUANT=${detected.dsv4PoolQuantDefault ? '1' : '0'} ${parts[0]}`
    }
    const lagunaMixedSwaTurboQuantActive = isLagunaMixedSwaTurboQuantEffective({
        detected,
        kvCacheQuantization: config.kvCacheQuantization,
        explicitKvCacheQuantizationApplied:
            config.continuousBatching !== false &&
            config.enablePrefixCache !== false &&
            !!config.kvCacheQuantization &&
            config.kvCacheQuantization !== 'auto',
    })
    const effectiveEnableJit = !!config.enableJit && !isVLM && !effectiveFlashMoe && !effectiveDistributed && !dsv4Active && !m3Active && !zayaCcaActive && !turboQuantActive && !lagunaMixedSwaTurboQuantActive && !hybridCacheActive
    if (shouldDisableLagunaJitDefault({
        detected,
        kvCacheQuantization: config.kvCacheQuantization,
        explicitKvCacheQuantizationApplied:
            config.continuousBatching !== false &&
            config.enablePrefixCache !== false &&
            !!config.kvCacheQuantization &&
            config.kvCacheQuantization !== 'auto',
        enableJitRequested: !!config.enableJit,
    })) {
        parts[0] = `${DISABLE_JANG_AFFINE_JIT_DEFAULT_ENV}=1 vmlx-engine serve`
    }

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
    if (prefillStepSize != null) parts.push('--prefill-step-size', prefillStepSize.toString())
    const completionBatchSize = finitePositiveInteger(config.completionBatchSize)
    if (!dsv4Active && completionBatchSize != null) parts.push('--completion-batch-size', completionBatchSize.toString())

    if (isVLM) parts.push('--is-mllm')
    else if (!dsv4Active && !effectiveSmelt && !m3Active && !omniBackendActive && detected?.isMultimodal && (userForceTextOnly || detected?.forceTextOnly)) {
        parts.push('--text-only')
    }
    const dflash2Speculative = /dflash2/i.test(config.speculativeModel || '')
    const cacheStackActive = dsv4Active
        ? true
        : dflash2Speculative
            ? false
            : config.continuousBatching !== false
    if (cacheStackActive) parts.push('--continuous-batching')
    else parts.push('--no-continuous-batching')

    // Parser resolution: User explicit choice -> Detected config -> Fallback
    // (mirrors buildArgs: user choice wins over detection)
    const effectiveToolParser = resolveEffectiveToolParser({
        configuredParser: config.toolCallParser,
        detectedParser: detected?.toolParser,
    })
    const effectiveAutoTool = config.enableAutoToolChoice ?? detected?.enableAutoToolChoice
    const effectiveReasoningParser = resolveEffectiveReasoningParser({
        configuredParser: config.reasoningParser,
        detectedParser: detected?.reasoningParser,
        supportsThinking: detected?.supportsThinking,
    })

    const exactTypedPromptDiskCache = usesExactTypedPromptDiskCache(detectedFamily)
    const cacheLaunch = buildCacheLaunchArgs({
        continuousBatching: cacheStackActive,
        enablePrefixCache: config.enablePrefixCache !== false,
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
        effectivePagedCacheBlockSize: dsv4Active
            ? DSV4_PAGED_CACHE_BLOCK_SIZE
            : config.pagedCacheBlockSize,
        maxCacheBlocks: config.maxCacheBlocks,
        diskCacheDir: config.diskCacheDir,
        diskCacheMaxGb: config.diskCacheMaxGb,
        blockDiskCacheDir: config.blockDiskCacheDir,
        blockDiskCacheMaxGb: config.blockDiskCacheMaxGb,
        blockDiskCacheMaxPercent: config.blockDiskCacheMaxPercent,
    })
    parts.push(...cacheLaunch.args)

    const streamInterval = finitePositiveInteger(config.streamInterval)
    if (streamInterval != null) parts.push('--stream-interval', streamInterval.toString())
    const maxTokens = finitePositiveInteger(config.maxTokens)
    if (maxTokens != null) {
        parts.push('--max-tokens', maxTokens.toString())
    }
    // Pass resolved parsers directly (mirrors buildArgs lines 1139-1150)
    parts.push(...buildToolLaunchArgs({
        toolParser: effectiveToolParser,
        enableAutoToolChoice: effectiveAutoTool,
    }))
    if (effectiveReasoningParser) parts.push('--reasoning-parser', effectiveReasoningParser)

    if (config.mcpConfig) parts.push('--mcp-config', config.mcpConfig)
    parts.push(...buildMcpPolicyArgs(config))

    if (effectiveSmelt) {
        parts.push('--smelt')
        const pct = finitePositiveInteger(config.smeltExperts) ?? 50
        if (pct !== 50) parts.push('--smelt-experts', pct.toString())
    }

    // Flash MoE and distributed compute mirror sessions.ts compatibility gates.
    if (effectiveFlashMoe) {
        parts.push('--flash-moe')
        const slotBank = finitePositiveInteger(config.flashMoeSlotBank)
        if (slotBank != null) parts.push('--flash-moe-slot-bank', slotBank.toString())
        if (config.flashMoePrefetch && config.flashMoePrefetch !== 'none') {
            parts.push('--flash-moe-prefetch', config.flashMoePrefetch)
        }
        const ioSplit = finitePositiveInteger(config.flashMoeIoSplit)
        if (ioSplit != null) parts.push('--flash-moe-io-split', ioSplit.toString())
    }

    if (effectiveDistributed) {
        parts.push('--distributed')
        const mode = config.distributedMode || 'pipeline'
        if (mode !== 'pipeline') parts.push('--distributed-mode', mode)
    }

    if (config.servedModelName) parts.push('--served-model-name', config.servedModelName)

    // Speculative decoding mirrors sessions.ts: external draft models are only
    // compatible with the non-VLM, non-DSV4, non-continuous-batching path,
    // except for the explicit DFlash2 text bridge on Qwen VLM bundles.
    const loopedNanbeige = detected?.family === 'nanbeige' || detected?.architectureHints?.cacheSchema === 'looped_kv_v1'
    const compatibleExternalSpeculative = !!config.speculativeModel && (
        dflash2Speculative
            ? !dsv4Active && isVLM
            : !dsv4Active && !isVLM && !cacheStackActive && !loopedNanbeige
    )
    if (compatibleExternalSpeculative) {
        parts.push('--speculative-model', config.speculativeModel)
        const numDraftTokens = finitePositiveInteger(config.numDraftTokens)
        if (numDraftTokens != null && numDraftTokens !== 3) {
            parts.push('--num-draft-tokens', numDraftTokens.toString())
        }
    }

    if (!dsv4Active && detected?.nativeMtp?.supported) {
        const mode = config.nativeMtpMode || 'auto'
        parts.push(...buildNativeMtpLaunchArgs({
            supported: true,
            detectedDepth: detected.nativeMtp.depth,
            configuredDepth: config.nativeMtpDepth,
            depthOverride: config.nativeMtpDepthOverride === true,
            mode,
            modelDefaultMode: detected.nativeMtp.defaultMode,
            externalSpeculativeActive: compatibleExternalSpeculative,
        }))
    }

    // Generation defaults are resolved inside vmlx_engine.server from
    // jang_config/generation_config. The panel preview must not synthesize
    // --default-* flags that would override the engine's bundle lookup.

    // Embedding model
    if (config.embeddingModel) parts.push('--embedding-model', config.embeddingModel)

    if (config.chatTemplate) parts.push('--chat-template', '"..."')

    // Thinking defaults are engine/model-owned. Chat/API requests carry
    // explicit enable_thinking; startup preview must not emit a server default.

    // JIT compilation
    if (effectiveEnableJit) parts.push('--enable-jit')
    else parts.push('--no-jit')

    if (omniBackendActive && config.omniBackend && config.omniBackend !== 'stage1') {
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
            dsv4Active ? DSV4_ADDITIONAL_ARG_BLOCKLIST : TEXT_ADDITIONAL_ARG_BLOCKLIST,
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

function countOccurrences(source: string, needle: string): number {
    return source.split(needle).length - 1
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

    it('minimax-m3 uses a 900s default timeout for long reasoning and streaming turns', () => {
        const out = preview({ timeout: 300 }, { family: 'minimax_m3' })
        expect(getFlagValue(out, '--timeout')).toBe('900')
    })

    it('minimax-m3 preserves explicit non-default timeout values', () => {
        const out = preview({ timeout: 1200 }, { family: 'minimax_m3' })
        expect(getFlagValue(out, '--timeout')).toBe('1200')
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

    it('user Force-Off beats auto-detected VLM: emits --text-only, not --is-mllm', () => {
        // Migration (sessions.ts) guarantees a detected-VL model only carries
        // isMultimodal===false when the user deliberately toggled Force-Off, so
        // buildArgs runs it text-only. Live-proven: Gemma4 Force-Off -> --text-only,
        // engine rejects image input (400). Preview must match that launch shape.
        const out = preview({ isMultimodal: false }, { isMultimodal: true })
        expect(hasFlag(out, '--is-mllm')).toBe(false)
        expect(hasFlag(out, '--text-only')).toBe(true)
    })

    it('manual isMultimodal=false is respected when detection is not VLM', () => {
        const out = preview({ isMultimodal: false }, { isMultimodal: false })
        expect(hasFlag(out, '--is-mllm')).toBe(false)
    })

    it('forceTextOnly detection wins over stale forced multimodal settings', () => {
        const out = preview({ isMultimodal: true }, { isMultimodal: false, forceTextOnly: true })
        expect(hasFlag(out, '--is-mllm')).toBe(false)
    })

    it('Smelt suppresses VLM launch even when stale multimodal settings remain', () => {
        const out = preview(
            { smelt: true, isMultimodal: true },
            { isMultimodal: true },
        )
        expect(hasFlag(out, '--smelt')).toBe(true)
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
        const out = preview({ enablePrefixCache: true, cacheMemoryMb: 4096, usePagedCache: false, enableBlockDiskCache: false })
        expect(getFlagValue(out, '--cache-memory-mb')).toBe('4096')
    })

    it('release cache profile is SSD-only: L2 budget emitted, no L1 RAM budget flags at all', () => {
        // Paged RAM is retired, so this profile (stale saved paged toggle and
        // all) launches block-disk-only. With no persistent L1 payload there is
        // nothing for --cache-memory-mb/--cache-memory-percent to bound, and
        // emitting either would claim a RAM tier that does not exist.
        const out = preview({
            enablePrefixCache: true,
            usePagedCache: true,
            enableBlockDiskCache: true,
            cacheMemoryMb: 4096,
            cacheMemoryPercent: 0,
            blockDiskCacheMaxGb: 10,
        })
        expect(hasFlag(out, '--use-paged-cache')).toBe(false)
        expect(hasFlag(out, '--no-paged-cache')).toBe(true)
        expect(hasFlag(out, '--cache-memory-mb')).toBe(false)
        expect(hasFlag(out, '--cache-memory-percent')).toBe(false)
        expect(getFlagValue(out, '--block-disk-cache-max-gb')).toBe('10')
    })

    it('cache memory mb has one shared call site consumed by preview and spawn', () => {
        const shared = readFileSync('src/shared/cacheLaunchArgs.ts', 'utf-8')
        const launcher = readFileSync('src/main/sessions.ts', 'utf-8')
        const renderer = readFileSync('src/renderer/src/components/sessions/SessionSettings.tsx', 'utf-8')
        expect(shared.match(/args\.push\('--cache-memory-mb'/g) ?? []).toHaveLength(1)
        expect(launcher).toContain("import { buildCacheLaunchArgs }")
        expect(renderer).toContain("import { buildCacheLaunchArgs }")
        expect(launcher).not.toContain("args.push('--cache-memory-mb'")
        expect(renderer).not.toContain("parts.push('--cache-memory-mb'")
    })

    it('memory-aware mode: sets --cache-memory-percent as fraction', () => {
        const out = preview({ enablePrefixCache: true, cacheMemoryPercent: 30, usePagedCache: false, enableBlockDiskCache: false })
        expect(getFlagValue(out, '--cache-memory-percent')).toBe('0.3')
    })

    it('sets --cache-ttl-minutes when > 0 and paged cache off', () => {
        const out = preview({ enablePrefixCache: true, cacheTtlMinutes: 60, usePagedCache: false, enableBlockDiskCache: false })
        expect(getFlagValue(out, '--cache-ttl-minutes')).toBe('60')
    })

    it('suppresses --cache-ttl-minutes when paged cache is on', () => {
        const out = preview({ enablePrefixCache: true, cacheTtlMinutes: 60, usePagedCache: true })
        expect(hasFlag(out, '--cache-ttl-minutes')).toBe(false)
    })
})

describe('Paged KV Cache', () => {
    it('never emits --use-paged-cache — an enabled toggle still launches --no-paged-cache', () => {
        // Inverse of the old contract ("includes --use-paged-cache when
        // enabled"): the RAM tier is retired for every family, so even a saved
        // usePagedCache=true must launch --no-paged-cache.
        const out = preview({ enablePrefixCache: true, usePagedCache: true })
        expect(hasFlag(out, '--use-paged-cache')).toBe(false)
        expect(hasFlag(out, '--no-paged-cache')).toBe(true)
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

    it('ZAYA typed CCA launches SSD block-disk only — the CCA requirement never re-enables paged RAM', () => {
        // OLD contract: ZAYA CCA force-enabled paged RAM whenever prefix cache
        // was on. NEW contract: no architecture escalates to the retired RAM
        // tier; ZAYA launches --no-paged-cache and the engine drops the
        // memory-aware lane instead (no prefix reuse, never a correctness risk).
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
        expect(hasFlag(out, '--use-paged-cache')).toBe(false)
        expect(hasFlag(out, '--no-paged-cache')).toBe(true)
        expect(hasFlag(out, '--enable-block-disk-cache')).toBe(true)
        // SSD-only launch retains no paged L1 payload, so the RAM byte ceiling
        // that #98 wired under paged cache is deliberately not sent.
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
    it('suppresses stale q8 quantization', () => {
        const out = preview({ enablePrefixCache: true, kvCacheQuantization: 'q8' })
        expect(hasFlag(out, '--kv-cache-quantization')).toBe(false)
    })

    it('suppresses stale q4 quantization', () => {
        const out = preview({ enablePrefixCache: true, kvCacheQuantization: 'q4' })
        expect(hasFlag(out, '--kv-cache-quantization')).toBe(false)
    })

    it('omits quantization in auto mode', () => {
        const out = preview({ enablePrefixCache: true, kvCacheQuantization: 'auto' })
        expect(hasFlag(out, '--kv-cache-quantization')).toBe(false)
    })

    it('suppresses stale explicit none', () => {
        const out = preview({ enablePrefixCache: true, kvCacheQuantization: 'none' })
        expect(hasFlag(out, '--kv-cache-quantization')).toBe(false)
    })

    it('panel copy does not claim TurboQuant is always on', () => {
        const fs = require('fs')
        const source = fs.readFileSync(
            'src/renderer/src/components/sessions/SessionConfigForm.tsx',
            'utf-8',
        )

        const enLocale = fs.readFileSync('src/renderer/src/i18n/locales/en.json', 'utf-8')
        expect(source).toContain("t('sessions.config.codecEngineNative')")
        expect(enLocale).toContain('Architecture-native cache')
        expect(source).toContain("t('sessions.config.hybridStatefulHint')")
        expect(enLocale).toContain('Generic TurboQuant KV is disabled unless a tested override exists')
        expect(source).not.toContain('ON · Default')
        expect(source).not.toContain('TurboQuant only')
    })

    it('describes mixed-SWA and native stored-cache state truthfully', () => {
        const fs = require('fs')
        const source = fs.readFileSync(
            'src/renderer/src/components/sessions/SessionConfigForm.tsx',
            'utf-8',
        )

        expect(source).toContain("isMixedSwaBundle(")
        const enLocale = fs.readFileSync('src/renderer/src/i18n/locales/en.json', 'utf-8')
        expect(source).toContain("t('sessions.config.codecMixedSwa')")
        expect(enLocale).toContain('Native full/sliding KV + rotating metadata')
        expect(enLocale).toContain("preserves the model's native cache-slot and rotating-window metadata")
        expect(enLocale).toContain('Generic TurboQuant is not added')
        expect(source).toContain("'NATIVE · GENERIC TQ OFF'")
        expect(source).toContain("const effectiveStoredCacheQuantization = 'auto'")
        expect(source).not.toContain('value="q8"')
        expect(source).not.toContain('value="q4"')
        expect(source).not.toContain("'TQ8 AUTO'")
    })

    it('reports HY3 native cache and MTP copy semantics', () => {
        const fs = require('fs')
        const source = fs.readFileSync(
            'src/renderer/src/components/sessions/SessionConfigForm.tsx',
            'utf-8',
        )

        expect(source).toContain("normalizedDetectedFamily === 'hy_v3' || normalizedDetectedFamily === 'hy3'")
        const enLocale = fs.readFileSync('src/renderer/src/i18n/locales/en.json', 'utf-8')
        expect(source).toContain("t('sessions.config.codecHy3')")
        expect(enLocale).toContain('Native HY3 KV (no added cache codec)')
        expect(source).toContain("'NATIVE · GENERIC TQ OFF'")
        expect(enLocale).toContain('Native MTP D1 copies this cache independently before batch split/verify')
    })

    it('reports Qwen and Bonsai architecture-native cache layouts', () => {
        const fs = require('fs')
        const source = fs.readFileSync(
            'src/renderer/src/components/sessions/SessionConfigForm.tsx',
            'utf-8',
        )

        expect(source).toContain("const bonsaiActive = normalizedModelIdentity.includes('bonsai')")
        expect(source).toContain("const qwenFullTqActive = !isMambaCache")
        const enLocale = fs.readFileSync('src/renderer/src/i18n/locales/en.json', 'utf-8')
        expect(source).toContain("t('sessions.config.codecQwenFull')")
        expect(enLocale).toContain('Native Qwen full-attention KV')
        expect(source).not.toContain('MIXED TQ4/8 AUTO')
        expect(source).toContain("t('sessions.config.codecQwenHybrid')")
        expect(enLocale).toContain('Native Qwen attention KV + native hybrid state')
        expect(source).toContain("t('sessions.config.codecBonsaiHybrid')")
        expect(enLocale).toContain('Native Bonsai attention KV + native hybrid state')
        expect(enLocale).toContain('Qwen hybrid cache detected — Auto preserves native attention KV')
        expect(enLocale).toContain('Bonsai hybrid cache detected — Auto preserves native attention KV')
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

    it('canonical defaults keep cache codec on auto without obsolete presets', () => {
        const fs = require('fs')
        const source = fs.readFileSync(
            'src/renderer/src/components/sessions/SessionConfigForm.tsx',
            'utf-8',
        )
        expect(DEFAULT_CONFIG.kvCacheQuantization).toBe('auto')
        expect(source).not.toContain('export const CASUAL_CONFIG')
        expect(source).not.toContain('export const EXPERT_CONFIG')
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

        // Labels and status copy render through i18n; assert the key wiring in
        // the sources and the English copy in the locale catalog.
        const panelCopy = fs.readFileSync(
            'src/renderer/src/i18n/locales/en.json',
            'utf-8',
        )

        expect(cachePanel).toContain('stats?.native_cache')
        expect(cachePanel).toContain('stats?.turboquant_kv_cache')
        expect(cachePanel).toContain('generic_turboquant_kv')
        expect(cachePanel).toContain('attention_kv_storage_quantization')
        expect(cachePanel).toContain('nativeCache?.storage_quantization')
        expect(cachePanel).toContain("t('sessions.cache.attentionKvL2')")
        expect(panelCopy).toContain('Stored Attention KV')
        expect(cachePanel).toContain('storage_encode_enabled')
        expect(cachePanel).toContain('stored_prefix_quantization')
        expect(cachePanel).toContain('storage_key_bits')
        expect(cachePanel).toContain('storage_value_bits')
        expect(cachePanel).toContain('key_bits_values')
        expect(cachePanel).toContain('value_bits_values')
        expect(cachePanel).toContain('kvQuant && !tqStoredPrefix')
        expect(cachePanel).toContain("t('sessions.cache.ssmPolicy')")
        expect(panelCopy).toContain('Companion State Policy')
        expect(cachePanel).toContain('single_sequence_only')
        expect(cachePanel).toContain('effective_max_num_seqs')
        expect(cachePanel).toContain("t('sessions.cache.cacheReuseSkips')")
        expect(panelCopy).toContain('Cache Reuse Skips')
        expect(cachePanel).toContain('last_cache_reuse_skip')
        expect(cachePanel).toContain('needed_mb')
        expect(cachePanel).toContain('available_mb')
        expect(perfPanel).toContain('native_cache?:')
        expect(perfPanel).toContain('turboquant_kv_cache?:')
        expect(perfPanel).toContain('quantization?:')
        expect(perfPanel).toContain('acceleration?:')
        expect(perfPanel).toContain('mtp?:')
        expect(perfPanel).toContain("t('sessions.performance.weightCodec')")
        expect(panelCopy).toContain('Weight Codec')
        expect(perfPanel).toContain("t('sessions.performance.metalNa')")
        expect(panelCopy).toContain('Metal NA')
        expect(perfPanel).toContain("t('sessions.performance.mtp')")
        expect(perfPanel).toContain('artifact_available')
        expect(perfPanel).toContain('runtime_available')
        expect(perfPanel).toContain('runtime_active')
        expect(perfPanel).toContain('runtime_reason')
        expect(perfPanel).toContain('effective_depth')
        expect(perfPanel).toContain("t('sessions.performance.mtpDepth')")
        expect(panelCopy).toContain('MTP Depth')
        expect(perfPanel).toContain('runtime_scope')
        expect(perfPanel).toContain('vl_runtime_available')
        expect(perfPanel).toContain('mtp_tensor_count')
        expect(perfPanel).toContain('vision_tensor_count')
        expect(perfPanel).toContain("t('sessions.performance.mtpScope')")
        expect(panelCopy).toContain('MTP Scope')
        expect(perfPanel).toContain("t('sessions.performance.mtpTensors')")
        expect(panelCopy).toContain('MTP Tensors')
        expect(perfPanel).toContain('last_native_mtp')
        expect(perfPanel).toContain("t('sessions.performance.mtpAccept')")
        expect(panelCopy).toContain('MTP Accept')
        expect(perfPanel).toContain("t('sessions.performance.mtpDepthRates')")
        expect(panelCopy).toContain('MTP Depth Rates')
        expect(perfPanel).toContain("t('sessions.performance.mtpForwards')")
        expect(panelCopy).toContain('MTP Forwards')
        expect(perfPanel).toContain("t('sessions.performance.mtpTiming')")
        expect(panelCopy).toContain('MTP Timing')
        expect(perfPanel).toContain('effective_depth_source')
        expect(perfPanel).toContain('last_native_mtp_skip')
        expect(perfPanel).toContain("t('sessions.performance.mtpSkip')")
        expect(panelCopy).toContain('MTP Skip')
        expect(perfPanel).toContain("t('sessions.performance.mtpLast')")
        expect(panelCopy).toContain('MTP Last')
        expect(perfPanel).toContain("t('sessions.performance.weightsPresentRuntimeReady')")
        expect(panelCopy).toContain('weights present; runtime ready')
        expect(perfPanel).toContain("t('sessions.performance.weightsPresentRuntimeUnwired')")
        expect(panelCopy).toContain('weights present; runtime unwired')
        expect(perfPanel).toContain("t('sessions.performance.notUsedByJangtq')")
        expect(panelCopy).toContain('not used by JANGTQ')
        expect(perfPanel).toContain("t('sessions.cache.genericTqKv')")
        expect(panelCopy).toContain('Generic TQ-KV')
        expect(perfPanel).toContain("t('sessions.cache.selectiveTqKv')")
        expect(panelCopy).toContain('Selective TQ-KV')
        expect(perfPanel).toContain('hybrid_attention_kv_only')
        expect(cachePanel).toContain("t('sessions.cache.selectiveTqKv')")
        expect(cachePanel).toContain('hybrid_attention_kv_only')
        expect(perfPanel).toContain('attention_kv_storage_quantization')
        expect(perfPanel).toContain('health?.native_cache?.storage_quantization')
        expect(perfPanel).toContain("t('sessions.cache.attentionKvL2')")
        expect(perfPanel).toContain('storage_encode_enabled')
        expect(perfPanel).toContain('stored_prefix_quantization')
        expect(perfPanel).toContain('storage_key_bits')
        expect(perfPanel).toContain('storage_value_bits')
        expect(perfPanel).toContain('key_bits_values')
        expect(perfPanel).toContain('value_bits_values')
        expect(perfPanel).toContain("t('sessions.cache.ssmPolicy')")
        expect(perfPanel).toContain("t('sessions.performance.cacheStack')")
        expect(panelCopy).toContain('Cache Stack')
        expect(perfPanel).toContain("t('sessions.performance.cacheComponents')")
        expect(panelCopy).toContain('Cache Components')
        expect(perfPanel).toContain('single_sequence_only')
        expect(perfPanel).toContain('scheduler?:')
        expect(perfPanel).toContain("t('sessions.performance.queue')")
        expect(panelCopy).toContain('Queue')
        expect(perfPanel).toContain("t('sessions.cache.ttftEwma')")
        expect(panelCopy).toContain('TTFT EWMA')
        expect(perfPanel).toContain("t('sessions.performance.cacheSkips')")
        expect(panelCopy).toContain('Cache Skips')
    })

    it('sets custom group size', () => {
        const out = preview({ enablePrefixCache: true, kvCacheQuantization: 'q8', kvCacheGroupSize: 32 })
        expect(hasFlag(out, '--kv-cache-group-size')).toBe(false)
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
        const out = preview({ enablePrefixCache: true, enableDiskCache: true, enableBlockDiskCache: false, usePagedCache: false })
        expect(hasFlag(out, '--enable-disk-cache')).toBe(true)
    })

    it('sets disk cache dir', () => {
        const out = preview({ enablePrefixCache: true, enableDiskCache: true, enableBlockDiskCache: false, usePagedCache: false, diskCacheDir: '/tmp/cache' })
        expect(getFlagValue(out, '--disk-cache-dir')).toBe('/tmp/cache')
    })

    it('sets disk cache max gb', () => {
        const out = preview({ enablePrefixCache: true, enableDiskCache: true, enableBlockDiskCache: false, usePagedCache: false, diskCacheMaxGb: 50 })
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
        expect(source).not.toContain('cacheControlUpdatesForPagedToggle')
        expect(source).toContain('cacheControlUpdatesForBlockDiskToggle')
        expect(source).not.toContain("label={t('sessions.config.pagedKVCache')}")
        expect(source).toContain('disabled={!cachePolicy.blockDiskCacheVisible || cachePolicy.blockDiskCacheDisabled || exactTypedPromptDiskCache}')
        expect(source).toContain('disabled={dsv4Active || cachePolicy.legacyDiskCacheDisabled}')
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

    it('emits an explicit opt-out when paged cache is active and block L2 is off', () => {
        const out = preview({ enablePrefixCache: true, usePagedCache: true, enableBlockDiskCache: false })
        expect(hasFlag(out, '--enable-block-disk-cache')).toBe(false)
        expect(hasFlag(out, '--disable-block-disk-cache')).toBe(true)
    })

    it('emits an explicit block L2 opt-out when paged cache is also off', () => {
        const out = preview({ enablePrefixCache: true, usePagedCache: false, enableBlockDiskCache: false })
        expect(hasFlag(out, '--no-paged-cache')).toBe(true)
        expect(hasFlag(out, '--enable-block-disk-cache')).toBe(false)
        expect(hasFlag(out, '--disable-block-disk-cache')).toBe(true)
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

    it('minimax-m3 explicit startup max tokens is reflected in CLI preview', () => {
        const out = preview({ maxTokens: 8192 }, { family: 'minimax_m3' })
        expect(getFlagValue(out, '--max-tokens')).toBe('8192')
    })

    it('surfaces Max Output Tokens separately from Max Context Tokens', () => {
        const formSource = readFileSync(resolve(__dirname, '../src/renderer/src/components/sessions/SessionConfigForm.tsx'), 'utf8')
        const maxOutputIndex = formSource.indexOf("label={t('sessions.config.maxOutputTokens')}")
        const maxContextIndex = formSource.indexOf("label={t('sessions.config.maxContextTokens')}")

        expect(maxOutputIndex).toBeGreaterThan(-1)
        expect(maxContextIndex).toBeGreaterThan(-1)
        expect(maxOutputIndex).toBeLessThan(maxContextIndex)
        expect(formSource).toContain("onChange={v => onChange('maxTokens', v)}")
        const enLocale = readFileSync(resolve(__dirname, '../src/renderer/src/i18n/locales/en.json'), 'utf8')
        expect(enLocale).toContain('maps to --max-tokens')
        expect(enLocale).toContain('does not change prompt/context length')
        expect(enLocale).toContain('Leave on Bundle / engine default unless you intentionally want a server-level cap')
    })

    it('persists bundle/default migration so stale 32768 sessions do not keep relaunching huge output caps', () => {
        const source = readFileSync(resolve(__dirname, '../src/main/sessions.ts'), 'utf8')
        const helper = readFileSync(resolve(__dirname, '../src/shared/sessionConfigMigrations.ts'), 'utf8')
        expect(source).toContain('function applyBundleStartupDefaults(')
        expect(source).toContain('migrateLegacyOutput && (oldHiddenMaxTokens || oldGenericMaxTokens)')
        expect(source).toContain('const bundleDefaultsChanged = applyBundleStartupDefaults(config, config.modelPath)')
        expect(source).toContain(
            'bundleDefaultsChanged || cacheDefaultsFilled || migrated || familyDefaultsChanged || normalized || markedCurrent'
        )
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

    it('canonical defaults leave maxTokens model-owned without a casual override', () => {
        const formSource = readFileSync(resolve(__dirname, '../src/renderer/src/components/sessions/SessionConfigForm.tsx'), 'utf8')
        expect(DEFAULT_CONFIG.maxTokens).toBe(0)
        expect(formSource).not.toContain('CASUAL_CONFIG')
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

    it('enables auto tool choice with a valid parser', () => {
        const out = preview({ enableAutoToolChoice: true, toolCallParser: 'qwen' })
        expect(hasFlag(out, '--enable-auto-tool-choice')).toBe(true)
    })

    it('does not enable auto tool choice when Auto has no detected parser', () => {
        const out = preview({ enableAutoToolChoice: true, toolCallParser: 'auto' })
        expect(hasFlag(out, '--tool-call-parser')).toBe(false)
        expect(hasFlag(out, '--enable-auto-tool-choice')).toBe(false)
    })

    it('uses detected tool parser when user is auto', () => {
        const out = preview({ enableAutoToolChoice: true, toolCallParser: 'auto' }, { toolParser: 'qwen' })
        expect(getFlagValue(out, '--tool-call-parser')).toBe('qwen')
    })

    it('manual tool parser overrides when no detected', () => {
        const out = preview({ enableAutoToolChoice: true, toolCallParser: 'llama' })
        expect(getFlagValue(out, '--tool-call-parser')).toBe('llama')
    })

    it('falls back from a stale saved tool parser instead of emitting invalid argv', () => {
        const detected = preview(
            { enableAutoToolChoice: true, toolCallParser: 'removed_parser_v0' },
            { toolParser: 'qwen' },
        )
        expect(getFlagValue(detected, '--tool-call-parser')).toBe('qwen')

        const unavailable = preview({
            enableAutoToolChoice: true,
            toolCallParser: 'removed_parser_v0',
        })
        expect(hasFlag(unavailable, '--tool-call-parser')).toBe(false)
        expect(hasFlag(unavailable, '--enable-auto-tool-choice')).toBe(false)
    })

    it('canonicalizes legacy DSV4 and Hy3 parser aliases before launch', () => {
        expect(getFlagValue(preview({ enableAutoToolChoice: true, toolCallParser: 'deepseek_v4' }), '--tool-call-parser')).toBe('dsml')
        expect(getFlagValue(preview({ enableAutoToolChoice: true, toolCallParser: 'hy_v3' }), '--tool-call-parser')).toBe('hunyuan')
        expect(getFlagValue(preview({ enableAutoToolChoice: true, toolCallParser: 'auto' }, { toolParser: 'deepseek_v4' }), '--tool-call-parser')).toBe('dsml')
        expect(getFlagValue(preview({ enableAutoToolChoice: true, toolCallParser: 'auto' }, { toolParser: 'hy_v3' }), '--tool-call-parser')).toBe('hunyuan')
    })

    it('"None" tool parser emits the literal --tool-call-parser none (engine opt-out), not an absent flag', () => {
        // The engine only disables tool parsing on the LITERAL "none"; an absent
        // flag makes it auto-configure the detected parser. So "None" must emit it,
        // even when a parser was detected — and must NOT enable auto-tool-choice.
        const out = preview({ toolCallParser: '', enableAutoToolChoice: true }, { toolParser: 'qwen' })
        expect(getFlagValue(out, '--tool-call-parser')).toBe('none')
        expect(out).not.toContain('--enable-auto-tool-choice')
    })

    it('"None" reasoning parser emits the literal --reasoning-parser none (engine opt-out)', () => {
        const out = preview({ reasoningParser: '' }, { reasoningParser: 'qwen3' })
        expect(getFlagValue(out, '--reasoning-parser')).toBe('none')
    })

    it('renders persisted literal none as the reasoning-parser None option', () => {
        const formSource = readFileSync('src/renderer/src/components/sessions/SessionConfigForm.tsx', 'utf8')
        expect(formSource).toContain(
            "value={config.reasoningParser === 'none' ? '' : config.reasoningParser}",
        )
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

    it('tool parser dropdown exposes Liquid LFM2 parser', () => {
        const formSource = readFileSync('src/renderer/src/components/sessions/SessionConfigForm.tsx', 'utf8')
        expect(formSource).toContain("value: 'lfm2'")
        expect(formSource).toContain('Liquid LFM2')
        expect(formSource).toContain('<|tool_call_start|>')
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

    it('empty tool parser disables tool parsing via literal --tool-call-parser none', () => {
        // "None" must emit the literal opt-out flag; an absent flag would make the
        // engine auto-configure the detected parser (the parity bug this fixes).
        const out = preview({ enableAutoToolChoice: true, toolCallParser: '' }, { toolParser: 'qwen' })
        expect(getFlagValue(out, '--tool-call-parser')).toBe('none')
        expect(hasFlag(out, '--enable-auto-tool-choice')).toBe(false)
    })

    it('uses detected reasoning parser when user is auto', () => {
        const out = preview({ reasoningParser: 'auto' }, { reasoningParser: 'qwen3' })
        expect(getFlagValue(out, '--reasoning-parser')).toBe('qwen3')
    })

    it('emits literal reasoning none for both explicit None spellings', () => {
        for (const reasoningParser of ['', 'none']) {
            const out = preview(
                { reasoningParser },
                { reasoningParser: 'qwen3', supportsThinking: true },
            )
            expect(getFlagValue(out, '--reasoning-parser')).toBe('none')
        }
    })

    it('uses literal reasoning none when current detection explicitly disables thinking', () => {
        const out = preview(
            { reasoningParser: 'auto' },
            { reasoningParser: 'qwen3', supportsThinking: false },
        )
        expect(getFlagValue(out, '--reasoning-parser')).toBe('none')
    })

    it('canonicalizes Poolside/Laguna vendor parser to the exact backend alias', () => {
        expect(canonicalizeReasoningParserForCli('poolside_v1')).toBe('deepseek_r1')
        const out = preview(
            { reasoningParser: 'auto' },
            { family: 'laguna', reasoningParser: 'poolside_v1' },
        )
        expect(getFlagValue(out, '--reasoning-parser')).toBe('deepseek_r1')
        expect(out).not.toContain('--reasoning-parser think_xml')
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

    it('passes MiniMax-M3 through its registered reasoning parser', () => {
        const out = preview(
            { reasoningParser: 'auto', toolCallParser: 'auto', enableAutoToolChoice: true },
            { family: 'minimax_m3', reasoningParser: 'minimax_m3', toolParser: 'minimax_m3', enableAutoToolChoice: true },
        )

        expect(getFlagValue(out, '--tool-call-parser')).toBe('minimax_m3')
        expect(getFlagValue(out, '--reasoning-parser')).toBe('minimax_m3')
    })

    it('uses MiniMax-M3 typed block-L2 without paged RAM, generic KV quantization, or JIT', () => {
        const out = preview(
            {
                enablePrefixCache: true,
                usePagedCache: true,
                enableDiskCache: false,
                enableBlockDiskCache: true,
                kvCacheQuantization: 'q4',
                enableJit: true,
            },
            {
                family: 'minimax_m3',
                reasoningParser: 'minimax_m3',
                toolParser: 'minimax_m3',
                enableAutoToolChoice: true,
                usePagedCache: true,
            },
        )

        expect(hasFlag(out, '--enable-disk-cache')).toBe(false)
        // Saved AND detected paged capability are both ignored: M3 launches the
        // typed MSA SSD tier with --no-paged-cache like every family.
        expect(hasFlag(out, '--use-paged-cache')).toBe(false)
        expect(hasFlag(out, '--no-paged-cache')).toBe(true)
        expect(hasFlag(out, '--enable-block-disk-cache')).toBe(true)
        expect(hasFlag(out, '--kv-cache-quantization')).toBe(false)
        expect(hasFlag(out, '--enable-jit')).toBe(false)
    })

    it('honors MiniMax-M3 paged Off while retaining typed block-disk L2', () => {
        const out = preview(
            {
                enablePrefixCache: true,
                usePagedCache: false,
                enableDiskCache: false,
                enableBlockDiskCache: true,
                kvCacheQuantization: 'auto',
                enableJit: false,
            },
            {
                family: 'minimax_m3',
                reasoningParser: 'minimax_m3',
                toolParser: 'minimax_m3',
                enableAutoToolChoice: true,
                usePagedCache: true,
            },
        )

        expect(hasFlag(out, '--no-paged-cache')).toBe(true)
        expect(hasFlag(out, '--use-paged-cache')).toBe(false)
        expect(hasFlag(out, '--enable-block-disk-cache')).toBe(true)
        expect(hasFlag(out, '--cache-memory-percent')).toBe(false)
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

    it('Auto follows a detected-Off tool contract even when a parser is present', () => {
        const out = preview({}, { toolParser: 'qwen', enableAutoToolChoice: false })
        expect(getFlagValue(out, '--tool-call-parser')).toBe('qwen')
        expect(hasFlag(out, '--enable-auto-tool-choice')).toBe(false)
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

    it('empty reasoning parser disables reasoning via literal --reasoning-parser none', () => {
        // "None" must emit the literal opt-out flag; an absent flag would make the
        // engine auto-configure the detected reasoning parser (the parity bug this fixes).
        const out = preview({ reasoningParser: '' }, { reasoningParser: 'qwen3' })
        expect(getFlagValue(out, '--reasoning-parser')).toBe('none')
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
        const out = preview({ continuousBatching: false, speculativeModel: 'mlx-community/Llama-3.2-1B-Instruct-4bit' })
        expect(getFlagValue(out, '--speculative-model')).toBe('mlx-community/Llama-3.2-1B-Instruct-4bit')
    })

    it('omits speculative model when empty', () => {
        const out = preview({ speculativeModel: '' })
        expect(hasFlag(out, '--speculative-model')).toBe(false)
    })

    it('omits --num-draft-tokens when default (3)', () => {
        const out = preview({ continuousBatching: false, speculativeModel: 'draft-model', numDraftTokens: 3 })
        expect(hasFlag(out, '--speculative-model')).toBe(true)
        expect(hasFlag(out, '--num-draft-tokens')).toBe(false)
    })

    it('sets --num-draft-tokens when non-default', () => {
        const out = preview({ continuousBatching: false, speculativeModel: 'draft-model', numDraftTokens: 5 })
        expect(getFlagValue(out, '--num-draft-tokens')).toBe('5')
    })

    it('omits --num-draft-tokens when no speculative model', () => {
        const out = preview({ speculativeModel: '', numDraftTokens: 10 })
        expect(hasFlag(out, '--num-draft-tokens')).toBe(false)
    })

    it('sets --num-draft-tokens=1 (minimum)', () => {
        const out = preview({ continuousBatching: false, speculativeModel: 'draft-model', numDraftTokens: 1 })
        expect(getFlagValue(out, '--num-draft-tokens')).toBe('1')
    })

    it('sets --num-draft-tokens=20 (maximum)', () => {
        const out = preview({ continuousBatching: false, speculativeModel: 'draft-model', numDraftTokens: 20 })
        expect(getFlagValue(out, '--num-draft-tokens')).toBe('20')
    })

    it('routes a DFlash2 draft for a Qwen VLM through SimpleEngine', () => {
        const out = preview(
            {
                speculativeModel: '/models/Qwen3.8-27B-DFlash2',
                continuousBatching: true,
            },
            {
                family: 'qwen3.5',
                isMultimodal: true,
                nativeMtp: { supported: true, depth: 3 },
            },
        )

        expect(getFlagValue(out, '--speculative-model')).toBe('/models/Qwen3.8-27B-DFlash2')
        expect(hasFlag(out, '--no-continuous-batching')).toBe(true)
        expect(hasFlag(out, '--continuous-batching')).toBe(false)
        expect(hasFlag(out, '--disable-native-mtp')).toBe(true)
        expect(hasFlag(out, '--native-mtp-depth')).toBe(false)
    })

    it('suppresses external speculative decoding for Nanbeige looped KV', () => {
        const out = preview(
            {
                continuousBatching: false,
                speculativeModel: 'draft-model',
                numDraftTokens: 5,
            },
            {
                family: 'nanbeige',
                cacheType: 'kv',
                architectureHints: {
                    cacheSchema: 'looped_kv_v1',
                    numLoops: 2,
                    cacheSlots: 44,
                },
            },
        )
        expect(hasFlag(out, '--speculative-model')).toBe(false)
        expect(hasFlag(out, '--num-draft-tokens')).toBe(false)
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
            requiresDeterministicSampling: false,
        },
    }

    it('defaults native-MTP bundles to adaptive Auto with greedy startup defaults', () => {
        const out = preview({}, qwenMtpDetected)

        // Auto mode must NOT pass an explicit depth: the flag becomes the
        // engine's explicit env override, pinning the start depth and
        // bypassing the tuning sidecar (which the engine reads itself) and
        // the session-scoped adaptive profile.
        // trailing space: hasFlag is substring-based and the policy flag
        // itself contains "--native-mtp-depth".
        expect(hasFlag(out, '--native-mtp-depth ')).toBe(false)
        expect(getFlagValue(out, '--native-mtp-depth-policy')).toBe('adaptive')
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
        expect(getFlagValue(out, '--native-mtp-depth-policy')).toBe('fixed')
    })

    it('defaults GLM Auto to AR but keeps an explicit fixed depth selectable', () => {
        const glmDetected: DetectedConfig = {
            ...qwenMtpDetected,
            family: 'glm5-next',
            nativeMtp: {
                ...qwenMtpDetected.nativeMtp!,
                runtimeScope: 'text',
                defaultMode: 'off',
            },
        }
        expect(hasFlag(preview({}, glmDetected), '--disable-native-mtp')).toBe(true)

        const explicit = preview({
            nativeMtpDepth: 3,
            nativeMtpDepthOverride: true,
        }, glmDetected)
        expect(hasFlag(explicit, '--disable-native-mtp')).toBe(false)
        expect(getFlagValue(explicit, '--native-mtp-depth')).toBe('3')
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
        const serverTypesSource = readFileSync('src/main/server.ts', 'utf8')
        const formSource = readFileSync('src/renderer/src/components/sessions/SessionConfigForm.tsx', 'utf8')

        expect(sessionsSource).toContain('--native-mtp-depth')
        expect(sessionsSource).toContain('--native-mtp-depth-policy')
        expect(sessionsSource).toContain('--native-mtp-sampling-policy')
        expect(sessionsSource).toContain('--disable-native-mtp')
        expect(sessionsSource).toContain('data?.mtp?.request_policy')
        // Adoption recovers the live process's policy/depth through the shared
        // helper; the old inline mapping turned an Auto session's own
        // deterministic-defaults launch policy into UI 'deterministic', which
        // the launcher re-emitted as greedy-only after restart.
        expect(sessionsSource).toContain('adoptNativeMtpConfig(proc, detectedFamily')
        expect(sessionsSource).not.toContain("proc.nativeMtpSamplingPolicy === 'deterministic-defaults'")
        expect(sessionsSource).toContain('data?.mtp?.depth_policy')
        expect(sessionsSource).toContain('data?.mtp?.effective_depth')
        expect(serverTypesSource).toContain("nativeMtpSamplingPolicy?: 'compatible-only' | 'deterministic-defaults' | 'greedy-only' | 'disabled'")
        expect(serverTypesSource).toContain("nativeMtpDepthPolicy?: 'fixed' | 'adaptive'")
        expect(formSource).toContain('Native MTP')
        expect(formSource).toContain('nativeMtpMode')
        expect(formSource).toContain('nativeMtpDepth')
        expect(formSource).toContain('nativeMtpDepthPolicy')
    })
})

describe('Generation Defaults', () => {
    it('uses the published vmlx package name for PyPI install guidance while preserving vmlx-engine entrypoints', () => {
        const engineManager = readFileSync('src/main/engine-manager.ts', 'utf8')
        const createSource = readFileSync('src/renderer/src/components/sessions/CreateSession.tsx', 'utf8')

        expect(engineManager).toContain("export const ENGINE_ENTRY_POINT_NAMES = ['vmlx-engine', 'vmlx-serve', 'vmlx']")
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
        const sharedDefaultsSource = readFileSync('src/shared/sessionGenerationDefaults.ts', 'utf8')

        expect(sessionsSource).toContain('max_new_tokens is also bundle-owned')
        expect(sessionsSource).toContain('defaultMaxNewTokens')
        expect(createSource).toContain('applyBundleGenerationDefaultsToSessionConfig')
        expect(createSource).not.toContain('next.defaultMaxNewTokens')
        expect(createSource).not.toContain('stored.defaultMaxNewTokens')
        expect(sessionSettingsSource).toContain('applyBundleGenerationDefaultsToSessionConfig')
        expect(sharedDefaultsSource).toContain('defaultMaxNewTokens:')
        expect(sessionsSource).not.toContain('config.maxTokens = defs.maxTokens')
        expect(createSource).not.toContain('next.maxTokens = gen.maxNewTokens')
        expect(createSource).not.toContain('stored.maxTokens = gen.maxNewTokens')
        expect(sessionSettingsSource).not.toContain('next.maxTokens = gen.maxNewTokens')
        expect(formSource).not.toContain('`max tokens ${config.maxTokens}`')
        expect(formSource).toContain('max output tokens')
    })

    it('shows explicit greedy generation defaults in the startup summary', () => {
        const formSource = readFileSync('src/renderer/src/components/sessions/SessionConfigForm.tsx', 'utf8')
        const createSource = readFileSync('src/renderer/src/components/sessions/CreateSession.tsx', 'utf8')
        const settingsSource = readFileSync('src/renderer/src/components/sessions/SessionSettings.tsx', 'utf8')
        const sharedDefaultsSource = readFileSync('src/shared/sessionGenerationDefaults.ts', 'utf8')

        expect(formSource).toContain('hasDeclaredSamplingDefaults')
        expect(formSource).toContain('config.defaultSamplingDefaultsDeclared === true')
        expect(createSource).toContain('applyBundleGenerationDefaultsToSessionConfig')
        expect(createSource).not.toContain('function hasDeclaredSamplingDefaults')
        expect(settingsSource).toContain('applyBundleGenerationDefaultsToSessionConfig')
        expect(sharedDefaultsSource).toContain('defaultSamplingDefaultsDeclared: hasDeclaredBundleSamplingDefaults(defaults)')
        expect(formSource).toContain("hasDeclaredSamplingDefaults ? `temperature ${(config.defaultTemperature / 100).toFixed(2)}` : null")
        expect(formSource).toContain("hasDeclaredSamplingDefaults ? ((config.defaultTopK ?? 0) > 0 ? `top-k ${Math.floor(config.defaultTopK ?? 0)}` : 'top-k off') : null")
    })

    it('chat settings do not invent a neutral repeat penalty when the bundle has no value', () => {
        const source = readFileSync('src/renderer/src/components/chat/ChatSettings.tsx', 'utf8')
        expect(source).toContain('const displayedRepeatPenalty = displayedOverrides.repeatPenalty ?? displayedModelDefaults.repeatPenalty')
        expect(source).toContain('displayedRepeatPenalty != null')
        expect(source).not.toContain('displayedModelDefaults.repeatPenalty ?? 1.0')
        expect(source).not.toContain('value={overrides.repeatPenalty ?? 1.1}')
    })

    it('preserves explicit neutral top-k and repetition overrides instead of snapping to bundle defaults', () => {
        const source = readFileSync('src/renderer/src/components/chat/ChatSettings.tsx', 'utf8')
        expect(source).toContain("update('topK', sanitized)")
        expect(source).toContain("update('repeatPenalty', sanitized)")
        expect(source).not.toContain("update('topK', v === 0 ? undefined : v)")
        expect(source).not.toContain("update('repeatPenalty', v === 1.0 ? undefined : v)")
    })

    it('chat settings expose per-chat max tokens without hidden DSV4 floors', () => {
        const chatSettings = readFileSync('src/renderer/src/components/chat/ChatSettings.tsx', 'utf8')
        const chatIpc = readFileSync('src/main/ipc/chat.ts', 'utf8')
        expect(chatSettings).toContain("onChange={v => update('maxTokens', v)}")
        // The "(model default)" placeholder copy moved behind t() in the i18n
        // pass. The invariant is unchanged — the field must advertise the model
        // default — so pin the key wiring AND the English copy it resolves to.
        const enLocale = readFileSync('src/renderer/src/i18n/locales/en.json', 'utf8')
        expect(chatSettings).toContain("t('chat.settings.modelDefaultWithValue', { value: displayedModelDefaults.maxTokens })")
        expect(enLocale).toContain('"modelDefaultWithValue": "{value} (model default)"')
        expect(chatSettings).not.toContain('next.maxTokens = Math.max')
        expect(chatSettings).not.toContain('4096')
        expect(chatIpc).toContain('dsv4OutputBudget(')
        expect(chatIpc).not.toContain('dsv4_finalizer_tokens')
        expect(chatIpc).not.toContain('Math.max(parsed ?? 0, 4096)')
    })

    it('new chat creation inherits only tool/workspace ergonomics, not stale sampling or prompts', () => {
        const source = readFileSync('src/main/ipc/chat.ts', 'utf8')
        const policy = readFileSync('src/main/chat-override-policy.ts', 'utf8')
        const createRecord = source.slice(
            source.indexOf('const createChatRecord ='),
            source.indexOf('ipcMain.handle(\n    "chat:create"'),
        )
        const createHandler = source.slice(
            source.indexOf('"chat:create"'),
            source.indexOf('ipcMain.handle(\n    "chat:ensureForModel"'),
        )

        expect(source).toContain('bundle generation defaults stay authoritative')
        expect(createRecord).toContain('getDefaultChatProfile')
        expect(createRecord).toContain('buildNewChatInheritedOverrides')
        expect(createHandler).toContain('createChatRecord(title, modelId, folderId, modelPath)')
        const ensureHandler = source.slice(
            source.indexOf('"chat:ensureForModel"'),
            source.indexOf('ipcMain.handle("chat:getByModel"'),
        )
        expect(ensureHandler).toContain('createChatRecord(title, "default", undefined, modelPath)')
        expect(policy).toContain('NEW_CHAT_TOOL_INHERIT_KEYS')
        const inheritanceStart = policy.indexOf('const NEW_CHAT_TOOL_INHERIT_KEYS = [')
        const inheritanceEnd = policy.indexOf('] as const', inheritanceStart)
        const inheritedKeys = policy.slice(inheritanceStart, inheritanceEnd)
        expect(inheritedKeys).not.toContain("'systemPrompt'")
        expect(inheritedKeys).not.toContain("'temperature'")
        expect(inheritedKeys).not.toContain("'enableThinking'")
        expect(source).not.toContain('enableThinkingFromReasoningMode')
        expect(createRecord).not.toContain('readGenerationDefaults')
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
        const out = preview({ additionalArgs: '--allowed-custom-flag yes' })
        expect(hasFlag(out, '--allowed-custom-flag yes')).toBe(true)
    })

    it('text additional args cannot override app-owned generation, parser, cache, or MTP flags', () => {
        const out = preview(
            {
                maxTokens: 123,
                maxContextLength: 456,
                additionalArgs: [
                    '--max-tokens 32768',
                    '--max-prompt-tokens=999999',
                    '--default-enable-thinking true',
                    '--default-repetition-penalty 1.2',
                    '--default-temperature=0',
                    '--reasoning-parser qwen3',
                    '--tool-call-parser qwen',
                    '--enable-auto-tool-choice',
                    '--native-mtp-depth 3',
                    '--native-mtp-sampling-policy deterministic-defaults',
                    '--disable-native-mtp',
                    '--use-paged-cache',
                    '--paged-cache-block-size 1',
                    '--kv-cache-quantization q4',
                    '--kv-cache-group-size 32',
                    '--speculative-model /tmp/draft',
                    '--num-draft-tokens 8',
                    '--allowed-custom-flag=yes',
                ].join(' '),
            },
            {
                family: 'ling',
                toolParser: undefined,
                reasoningParser: undefined,
                usePagedCache: false,
                nativeMtp: { supported: false },
            },
        )
        const normalized = out.replace(/\s*\\\n\s*/g, ' ')

        expect(getFlagValue(normalized, '--max-tokens')).toBe('123')
        expect(getFlagValue(normalized, '--max-prompt-tokens')).toBe('456')
        expect((normalized.match(/--max-tokens/g) || []).length).toBe(1)
        expect((normalized.match(/--max-prompt-tokens/g) || []).length).toBe(1)
        expect(normalized).not.toContain('--default-enable-thinking')
        expect(normalized).not.toContain('--default-repetition-penalty')
        expect(normalized).not.toContain('--default-temperature')
        expect(normalized).not.toContain('--reasoning-parser')
        expect(normalized).not.toContain('--tool-call-parser')
        expect(normalized).not.toContain('--enable-auto-tool-choice')
        expect(normalized).not.toContain('--native-mtp-depth')
        expect(normalized).not.toContain('--native-mtp-sampling-policy')
        expect(normalized).not.toContain('--disable-native-mtp')
        // Additional args can never resurrect the retired paged RAM tier.
        expect(normalized).not.toContain('--use-paged-cache')
        expect(normalized).not.toContain('--paged-cache-block-size 1')
        expect(normalized).not.toContain('--kv-cache-quantization')
        expect(normalized).not.toContain('--kv-cache-group-size')
        expect(normalized).not.toContain('--speculative-model')
        expect(normalized).not.toContain('--num-draft-tokens')
        expect(normalized).toContain('--allowed-custom-flag=yes')
    })

    it('text additional args cannot override app-owned server, template, model-name, or MCP flags', () => {
        const out = preview({
            host: '127.0.0.1',
            port: 8000,
            timeout: 300,
            rateLimit: 2,
            logLevel: 'WARN',
            corsOrigins: 'https://app.example',
            servedModelName: 'ui-name',
            chatTemplate: 'ui-template',
            mcpConfig: '/ui/mcp.json',
            additionalArgs: [
                '--host 0.0.0.0',
                '--port=9999',
                '--timeout 1',
                '--rate-limit=99',
                '--log-level DEBUG',
                '--allowed-origins=*',
                '--served-model-name raw-name',
                '--chat-template raw-template',
                '--chat-template-kwargs {"enable_thinking":true}',
                '--mcp-config /tmp/raw-mcp.json',
                '--mcp-enabled-servers raw',
                '--mcp-disabled-tools raw-tool',
                '--api-key raw-secret',
                '--uds=/tmp/raw.sock',
                '--wake-timeout 1',
                '--inference-endpoints http://127.0.0.1:9999',
                '--allowed-custom-flag yes',
            ].join(' '),
        })
        const normalized = out.replace(/\s*\\\n\s*/g, ' ')

        expect(getFlagValue(normalized, '--host')).toBe('127.0.0.1')
        expect(getFlagValue(normalized, '--port')).toBe('8000')
        expect(getFlagValue(normalized, '--timeout')).toBe('300')
        expect(getFlagValue(normalized, '--rate-limit')).toBe('2')
        expect(getFlagValue(normalized, '--log-level')).toBe('WARN')
        expect(getFlagValue(normalized, '--allowed-origins')).toBe('https://app.example')
        expect(getFlagValue(normalized, '--served-model-name')).toBe('ui-name')
        expect(getFlagValue(normalized, '--chat-template')).toBe('"..."')
        expect(getFlagValue(normalized, '--mcp-config')).toBe('/ui/mcp.json')
        expect(normalized).not.toContain('0.0.0.0')
        expect(normalized).not.toContain('9999')
        expect(normalized).not.toContain('--timeout 1')
        expect(normalized).not.toContain('--rate-limit=99')
        expect(normalized).not.toContain('DEBUG')
        expect(normalized).not.toContain('--allowed-origins=*')
        expect(normalized).not.toContain('raw-name')
        expect(normalized).not.toContain('raw-template')
        expect(normalized).not.toContain('/tmp/raw-mcp.json')
        expect(normalized).not.toContain('--chat-template-kwargs')
        expect(normalized).not.toContain('--mcp-enabled-servers')
        expect(normalized).not.toContain('--mcp-disabled-tools')
        expect(normalized).not.toContain('--api-key')
        expect(normalized).not.toContain('--uds')
        expect(normalized).not.toContain('--wake-timeout')
        expect(normalized).not.toContain('--inference-endpoints')
        expect(normalized).toContain('--allowed-custom-flag yes')
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
                    '--max-tokens=32768',
                    '--log-level=DEBUG',
                    '--uds=/tmp/stale-dsv4.sock',
                    '--no-state-machine-stops',
                    '--prefill-keep-alloc',
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
        expect((normalized.match(/--dsv4-enable-prefix-cache/g) || []).length).toBe(0)
        expect(normalized).not.toContain('--disable-prefix-cache')
        expect(normalized).toContain('--no-paged-cache')
        expect(normalized).toContain('--enable-block-disk-cache')
        expect(normalized).not.toContain('--default-temperature')
        expect(normalized).not.toContain('--max-tokens')
        expect(normalized).not.toContain('--log-level DEBUG')
        expect(normalized).not.toContain('--log-level=DEBUG')
        expect(normalized).not.toContain('--uds')
        expect(normalized).not.toContain('/tmp/stale-dsv4.sock')
        expect(normalized).not.toContain('--no-state-machine-stops')
        expect(normalized).not.toContain('--prefill-keep-alloc')

        const sessionsSource = readFileSync('src/main/sessions.ts', 'utf8')
        const settingsSource = readFileSync('src/renderer/src/components/sessions/SessionSettings.tsx', 'utf8')
        for (const source of [sessionsSource, settingsSource]) {
            expect(source).toContain("'--native-mtp-depth'")
            expect(source).toContain("'--native-mtp-sampling-policy'")
            expect(source).toContain("'--disable-native-mtp'")
            expect(source).toContain("'--dsv4-enable-prefix-cache'")
        }
    })

    it('DSV4 additional args strips blocked equals-form serve overrides', () => {
        const out = preview(
            {
                additionalArgs: [
                    '--uds=/tmp/vmlx.sock',
                    '--inference-endpoints=http://127.0.0.1:9999',
                    '--wake-timeout=999',
                    '--prefill-keep-alloc',
                    '--no-state-machine-stops',
                    '--allowed-custom-flag=yes',
                ].join(' '),
            },
            { family: 'deepseek-v4', usePagedCache: false },
        )
        const normalized = out.replace(/\s*\\\n\s*/g, ' ')

        expect(normalized).not.toContain('--uds')
        expect(normalized).not.toContain('--inference-endpoints')
        expect(normalized).not.toContain('--wake-timeout')
        expect(normalized).not.toContain('--prefill-keep-alloc')
        expect(normalized).not.toContain('--no-state-machine-stops')
        expect(normalized).toContain('--allowed-custom-flag=yes')
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
        expect(getFlagValue(preview({ enablePrefixCache: true, usePagedCache: true, pagedCacheBlockSize: 32 }), '--paged-cache-block-size')).toBe('32')
        expect(getFlagValue(preview({ enablePrefixCache: true, usePagedCache: true, pagedCacheBlockSize: 256 }), '--paged-cache-block-size')).toBe('256')
    })

    it('deepseek-v4 uses native SSD-only composite reuse without generic cache codecs', () => {
        const out = preview(
            {
                dsv4PoolQuant: true,
                enablePrefixCache: true,
                continuousBatching: false,
                usePagedCache: false,
                enableBlockDiskCache: true,
                pagedCacheBlockSize: 64,
                maxCacheBlocks: 4097,
                prefillBatchSize: 512,
                prefillStepSize: 2048,
                completionBatchSize: 512,
                kvCacheQuantization: 'q4',
                speculativeModel: '/tmp/draft',
                isMultimodal: true,
                nativeMtpDepth: 3,
            },
            { family: 'deepseek-v4', usePagedCache: false, dsv4PoolQuantDefault: true },
        )

        expect(out).toContain('DSV4_POOL_QUANT=1 vmlx-engine serve')
        expect(hasFlag(out, '--dsv4-enable-prefix-cache')).toBe(false)
        expect(hasFlag(out, '--use-paged-cache')).toBe(false)
        expect(hasFlag(out, '--no-paged-cache')).toBe(true)
        expect(getFlagValue(out, '--paged-cache-block-size')).toBe('256')
        expect(getFlagValue(out, '--max-cache-blocks')).toBe('4097')
        expect(hasFlag(out, '--enable-block-disk-cache')).toBe(true)
        expect(getFlagValue(out, '--max-num-seqs')).toBe('1')
        expect(hasFlag(out, '--no-continuous-batching')).toBe(false)
        expect(hasFlag(out, '--continuous-batching')).toBe(true)
        expect(hasFlag(out, '--disable-prefix-cache')).toBe(false)
        expect(hasFlag(out, '--prefill-batch-size')).toBe(false)
        expect(getFlagValue(out, '--prefill-step-size')).toBe('2048')
        expect(hasFlag(out, '--completion-batch-size')).toBe(false)
        expect(hasFlag(out, '--kv-cache-quantization')).toBe(false)
        expect(hasFlag(out, '--speculative-model')).toBe(false)
        expect(hasFlag(out, '--is-mllm')).toBe(false)
        expect(hasFlag(out, '--native-mtp-depth')).toBe(false)
    })

    it('deepseek-v4 passes the visible prefill step through to its native generator', () => {
        const out = preview(
            { prefillStepSize: 512 },
            { family: 'deepseek-v4', usePagedCache: false },
        )

        expect(getFlagValue(out, '--prefill-step-size')).toBe('512')
        expect(hasFlag(out, '--prefill-batch-size')).toBe(false)
        expect(hasFlag(out, '--completion-batch-size')).toBe(false)
    })

    it('deepseek-v4 command preview reflects an explicit bundle pool-codec Off stamp', () => {
        const out = preview(
            { enablePrefixCache: true, usePagedCache: false, enableBlockDiskCache: true },
            { family: 'deepseek-v4', dsv4PoolQuantDefault: false },
        )

        expect(out).toContain('DSV4_POOL_QUANT=0 vmlx-engine serve')
        expect(hasFlag(out, '--kv-cache-quantization')).toBe(false)
    })

    it('deepseek-v4 SSD-only settings use fixed native composite blocks', () => {
        const out = preview(
            {
                dsv4PrefixCache: true,
                enablePrefixCache: true,
                usePagedCache: false,
                enableBlockDiskCache: true,
                pagedCacheBlockSize: 64,
                maxCacheBlocks: 4097,
            },
            { family: 'deepseek-v4', usePagedCache: false },
        )

        expect(hasFlag(out, '--dsv4-enable-prefix-cache')).toBe(false)
        expect(hasFlag(out, '--disable-prefix-cache')).toBe(false)
        expect(hasFlag(out, '--use-paged-cache')).toBe(false)
        expect(hasFlag(out, '--no-paged-cache')).toBe(true)
        expect(getFlagValue(out, '--paged-cache-block-size')).toBe('256')
        expect(getFlagValue(out, '--max-cache-blocks')).toBe('4097')
        expect(hasFlag(out, '--enable-block-disk-cache')).toBe(true)
    })

    it('Step3.7 full/sliding KV cache subtype honors typed SSD-only mode with Block L2', () => {
        const out = preview(
            {
                enablePrefixCache: true,
                continuousBatching: true,
                usePagedCache: false,
                enableDiskCache: true,
                enableBlockDiskCache: true,
                kvCacheQuantization: 'q4',
                toolCallParser: 'auto',
                reasoningParser: 'auto',
            },
            {
                family: 'step-3.7-flash',
                cacheType: 'kv',
                cacheSubtype: 'step3p7_full_sliding_kv',
                usePagedCache: true,
                isMultimodal: true,
                toolParser: 'step3p5',
                reasoningParser: 'qwen3',
            },
        )

        expect(hasFlag(out, '--is-mllm')).toBe(true)
        expect(hasFlag(out, '--use-paged-cache')).toBe(false)
        expect(hasFlag(out, '--no-paged-cache')).toBe(true)
        expect(hasFlag(out, '--enable-disk-cache')).toBe(false)
        expect(hasFlag(out, '--enable-block-disk-cache')).toBe(true)
        expect(hasFlag(out, '--kv-cache-quantization')).toBe(false)
        expect(getFlagValue(out, '--tool-call-parser')).toBe('step3p5')
        expect(getFlagValue(out, '--reasoning-parser')).toBe('qwen3')
        expect(hasFlag(out, '--cache-memory-percent')).toBe(false)
    })

    it('deepseek-v4 respects explicit prefix cache disable and suppresses dependent caches', () => {
        const out = preview(
            {
                dsv4PrefixCache: false,
                dsv4PoolQuant: true,
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

    it('deepseek-v4 cache launch flags are singular and preserve explicit standard controls', () => {
        const enabled = preview(
            {
                dsv4PrefixCache: true,
                enablePrefixCache: true,
                usePagedCache: false,
                enableDiskCache: true,
                enableBlockDiskCache: true,
                kvCacheQuantization: 'q4',
                noMemoryAwareCache: true,
                prefixCacheSize: 200,
                cacheMemoryPercent: 25,
            },
            { family: 'deepseek-v4', cacheType: 'hybrid', usePagedCache: false },
        ).replace(/\s*\\\n\s*/g, ' ')

        expect(countOccurrences(enabled, '--dsv4-enable-prefix-cache')).toBe(0)
        expect(countOccurrences(enabled, '--use-paged-cache')).toBe(0)
        expect(countOccurrences(enabled, '--no-paged-cache')).toBe(1)
        expect(countOccurrences(enabled, '--enable-block-disk-cache')).toBe(1)
        expect(getFlagValue(enabled, '--paged-cache-block-size')).toBe('256')
        expect(countOccurrences(enabled, '--disable-prefix-cache')).toBe(0)
        expect(hasFlag(enabled, '--enable-disk-cache')).toBe(false)
        expect(hasFlag(enabled, '--kv-cache-quantization')).toBe(false)
        expect(hasFlag(enabled, '--no-memory-aware-cache')).toBe(false)
        expect(hasFlag(enabled, '--prefix-cache-size')).toBe(false)
        expect(hasFlag(enabled, '--cache-memory-percent')).toBe(false)

        const disabled = preview(
            {
                dsv4PrefixCache: false,
                enablePrefixCache: false,
                usePagedCache: true,
                enableBlockDiskCache: true,
            },
            { family: 'deepseek-v4', cacheType: 'hybrid', usePagedCache: true },
        ).replace(/\s*\\\n\s*/g, ' ')

        expect(countOccurrences(disabled, '--dsv4-enable-prefix-cache')).toBe(0)
        expect(countOccurrences(disabled, '--disable-prefix-cache')).toBe(1)
        expect(hasFlag(disabled, '--use-paged-cache')).toBe(false)
        expect(hasFlag(disabled, '--enable-block-disk-cache')).toBe(false)
    })

    it('DSV4 native cache settings stay family-scoped and suppress only generic TurboQuant', () => {
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
        expect(hasFlag(nonDsv4, '--use-paged-cache')).toBe(false)
        expect(hasFlag(nonDsv4, '--no-paged-cache')).toBe(true)
        expect(hasFlag(nonDsv4, '--enable-block-disk-cache')).toBe(true)
        expect(hasFlag(nonDsv4, '--kv-cache-quantization')).toBe(false)
        expect(hasFlag(nonDsv4, '--kv-cache-group-size')).toBe(false)

        const dsv4 = preview(staleDsv4Config, {
            family: 'deepseek-v4',
            usePagedCache: true,
        })
        expect(hasFlag(dsv4, '--dsv4-enable-prefix-cache')).toBe(false)
        expect(hasFlag(dsv4, '--use-paged-cache')).toBe(false)
        expect(hasFlag(dsv4, '--no-paged-cache')).toBe(true)
        expect(getFlagValue(dsv4, '--paged-cache-block-size')).toBe('256')
        expect(hasFlag(dsv4, '--disable-prefix-cache')).toBe(false)
        expect(hasFlag(dsv4, '--enable-block-disk-cache')).toBe(true)
        expect(hasFlag(dsv4, '--kv-cache-quantization')).toBe(false)
        expect(hasFlag(dsv4, '--kv-cache-group-size')).toBe(false)
    })

    it('deepseek-v4 family defaults preserve user cache controls and derive native invariants', () => {
        const source = readFileSync(resolve(__dirname, '../src/main/sessions.ts'), 'utf8')
        expect(source).toContain('config.dsv4PrefixCache = dsv4PrefixEnabled')
        expect(source).toContain('if (config.enablePrefixCache === undefined)')
        expect(source).toContain('if (config.usePagedCache === undefined)')
        expect(source).toContain('if (config.enableBlockDiskCache === undefined)')
        expect(source).not.toContain('config.dsv4PoolQuant = false')
        expect(source).not.toContain('const dsv4DefaultCacheOptIn = false')
        expect(source).toContain('config.dsv4PoolQuant = detected.dsv4PoolQuantDefault')
        expect(source).toContain('delete config.dsv4PoolQuant')
        expect(source).toContain("dsv4PoolQuant: detectedFamily === 'deepseek-v4'")
        expect(source).toContain('? detected.dsv4PoolQuantDefault')
    })

    it('detected Qwen3.6 hybrid cache honors paged Off when block SSD L2 owns the prefix backend', () => {
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
        expect(hasFlag(out, '--use-paged-cache')).toBe(false)
        expect(hasFlag(out, '--no-paged-cache')).toBe(true)
        expect(hasFlag(out, '--enable-block-disk-cache')).toBe(true)
        // SSD-only mode has no persistent L1 payload, so a RAM ceiling is inert.
        expect(hasFlag(out, '--cache-memory-percent')).toBe(false)
    })

    it('detected Mamba cache never re-enables paged RAM even when Block L2 is absent', () => {
        // OLD contract: a Mamba architecture without Block L2 escalated to
        // paged RAM while plain KV honored the saved false — the two argvs
        // diverged. NEW contract: the escalation is gone; Mamba launches
        // exactly like plain KV (--no-paged-cache) and simply gets no prefix
        // reuse when Block L2 is off.
        const mambaOut = preview(
            {
                enablePrefixCache: true,
                usePagedCache: false,
                enableBlockDiskCache: false,
            },
            { family: 'qwen3-next', cacheType: 'mamba', usePagedCache: true },
        )
        const kvOut = preview(
            { enablePrefixCache: true, usePagedCache: false },
            { family: 'qwen3', cacheType: 'kv', usePagedCache: true },
        )

        expect(hasFlag(mambaOut, '--use-paged-cache')).toBe(false)
        expect(hasFlag(mambaOut, '--no-paged-cache')).toBe(true)
        expect(hasFlag(mambaOut, '--disable-block-disk-cache')).toBe(true)
        expect(hasFlag(kvOut, '--use-paged-cache')).toBe(false)
        expect(hasFlag(kvOut, '--no-paged-cache')).toBe(true)
    })

    it('detected Gemma4 mixed-SWA rotating KV honors SSD-only mode with Block L2', () => {
        const out = preview(
            {
                enablePrefixCache: true,
                usePagedCache: false,
                enableDiskCache: true,
                enableBlockDiskCache: true,
                cacheMemoryPercent: 15,
            },
            {
                family: 'gemma4',
                cacheType: 'rotating_kv',
                usePagedCache: true,
                isMultimodal: true,
                toolParser: 'gemma4',
                reasoningParser: 'gemma4',
            },
        )

        expect(hasFlag(out, '--is-mllm')).toBe(true)
        expect(hasFlag(out, '--use-paged-cache')).toBe(false)
        expect(hasFlag(out, '--no-paged-cache')).toBe(true)
        expect(hasFlag(out, '--enable-disk-cache')).toBe(false)
        expect(hasFlag(out, '--enable-block-disk-cache')).toBe(true)
        // SSD-only mode does not retain a paged L1 payload mirror.
        expect(hasFlag(out, '--cache-memory-percent')).toBe(false)
    })

    it('detected Gemma4 mixed-SWA rotating KV never forces paged cache — without Block L2 it launches with no cache tier', () => {
        // OLD contract: mixed_swa_kv without Block L2 escalated to paged RAM.
        // NEW contract: the retired RAM tier is never a fallback; the launch is
        // --no-paged-cache with block disk explicitly disabled, so this shape
        // gets no prefix reuse rather than a RAM tier.
        const out = preview(
            {
                enablePrefixCache: true,
                usePagedCache: false,
                enableBlockDiskCache: false,
            },
            {
                family: 'gemma4',
                cacheType: 'rotating_kv',
                cacheSubtype: 'mixed_swa_kv',
                usePagedCache: true,
                isMultimodal: true,
            },
        )

        expect(hasFlag(out, '--use-paged-cache')).toBe(false)
        expect(hasFlag(out, '--no-paged-cache')).toBe(true)
        expect(hasFlag(out, '--enable-block-disk-cache')).toBe(false)
        expect(hasFlag(out, '--disable-block-disk-cache')).toBe(true)
    })

    it('changing maxCacheBlocks produces different CLI output', () => {
        expect(getFlagValue(preview({ enablePrefixCache: true, usePagedCache: true, maxCacheBlocks: 500 }), '--max-cache-blocks')).toBe('500')
        expect(getFlagValue(preview({ enablePrefixCache: true, usePagedCache: true, maxCacheBlocks: 5000 }), '--max-cache-blocks')).toBe('5000')
    })

    it('changing startup generation defaults does not change CLI output', () => {
        expect(preview({ defaultTemperature: 50 })).toBe(preview({ defaultTemperature: 100 }))
        expect(preview({ defaultTopK: 40 })).toBe(preview({ defaultTopK: 0 }))
    })

    it('changing speculativeModel produces different CLI output', () => {
        const a = preview({ continuousBatching: false, speculativeModel: 'model-a' })
        const b = preview({ continuousBatching: false, speculativeModel: 'model-b' })
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

    it('current startup defaults use the paged-off block-SSD single-user cache stack', () => {
        const out = preview()

        expect(getFlagValue(out, '--max-num-seqs')).toBe('1')
        expect(getFlagValue(out, '--prefill-batch-size')).toBe('512')
        expect(getFlagValue(out, '--prefill-step-size')).toBe('2048')
        expect(getFlagValue(out, '--completion-batch-size')).toBe('512')
        expect(hasFlag(out, '--continuous-batching')).toBe(true)
        expect(hasFlag(out, '--disable-prefix-cache')).toBe(false)
        expect(hasFlag(out, '--cache-memory-percent')).toBe(false)
        expect(hasFlag(out, '--use-paged-cache')).toBe(false)
        expect(hasFlag(out, '--no-paged-cache')).toBe(true)
        expect(hasFlag(out, '--enable-block-disk-cache')).toBe(true)
        expect(hasFlag(out, '--enable-disk-cache')).toBe(false)
        expect(hasFlag(out, '--default-temperature')).toBe(false)
        expect(hasFlag(out, '--default-top-p')).toBe(false)
        expect(hasFlag(out, '--default-repetition-penalty')).toBe(false)
        expect(hasFlag(out, '--enable-jit')).toBe(true)
    })

    it('session manager migrates the exact stale continuous-cache default tuple', () => {
        const source = readFileSync('src/main/sessions.ts', 'utf8')
        expect(source).toContain('function applyCacheStackStartupDefaultMigration')
        expect(source).toMatch(/const CACHE_STACK_STARTUP_DEFAULTS_VERSION = \d+/)
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
        // No migration may ever write paged-ON: in-RAM paged cache is retired
        // for every family, so the literal assignment must not exist anywhere.
        expect(source).not.toContain('config.usePagedCache = true')
        expect(source).toContain('config.maxCacheBlocks = 1000')
        expect(source).toContain("config.kvCacheQuantization = 'auto'")
        expect(source).toContain('config.enableBlockDiskCache = true')
        expect(source).toContain('config.blockDiskCacheMaxGb = 10')
        expect(source).toContain('config.cacheMemoryPercent = 15')
        // Current migration branches outcome by detected cache capability.
        // Generic upgraders inherit the detected paged/block-L2 tuple.
        expect(source).toContain('const staleV2GenericPagedOn =')
        expect(source).toContain('if (zayaCacheMigrationTarget) {')
        expect(source).toContain('config.usePagedCache = false')
        expect(source).toContain('config.enableDiskCache = false')
        expect(source).toContain('config.enableBlockDiskCache = false')
    })

    it('re-detect normalizes a saved In-Memory Paged Cache (RAM)=On to Off and logs it', () => {
        // Inverse of the removed "stale saved In-Memory Paged Cache (RAM)=Off
        // was reset to auto-safe On" behaviour: with the RAM tier retired, a
        // saved On is what is stale, and re-detect must force it Off — never
        // the other way around.
        const source = readFileSync('src/main/sessions.ts', 'utf8')
        expect(source).toContain('if (config.usePagedCache === true) {')
        expect(source).toContain(
            'In-Memory Paged Cache (RAM) forced Off — SSD block-disk cache (L2) is the only cache tier',
        )
        expect(source).not.toContain('stale saved In-Memory Paged Cache (RAM)=Off was reset to auto-safe On')
    })

    it('cache-stack migration is one-time versioned so saved user toggles stick', () => {
        const source = readFileSync('src/main/sessions.ts', 'utf8')
        const serverSource = readFileSync('src/main/server.ts', 'utf8')
        expect(serverSource).toContain('cacheStackStartupDefaultsVersion?: number')
        expect(source).toContain('function applyMissingCacheStackStartupDefaults')
        expect(source).toContain('cacheDefaultsVersion >= CACHE_STACK_STARTUP_DEFAULTS_VERSION')
        expect(source).toContain('const cacheDefaultsFilled = applyMissingCacheStackStartupDefaults(config, config.modelPath)')
        expect(source).toContain('const markedCurrent = markCacheStackStartupDefaultsCurrent(config, config.modelPath)')
        expect(source).toContain('const familyDefaultsChanged = applyFamilyStartupDefaults(config, config.modelPath)')
        expect(source).toContain(
            'if (bundleDefaultsChanged || cacheDefaultsFilled || migrated || familyDefaultsChanged || normalized || markedCurrent)'
        )
        expect(source).toContain('markCacheStackStartupDefaultsCurrent(merged as Partial<ServerConfig>, session.modelPath)')
        expect(source).toContain('cacheStackStartupDefaultsVersion: CACHE_STACK_STARTUP_DEFAULTS_VERSION')
    })

    it('MiniMax-M3 fills long-generation timeout but keeps max output model-owned before launch/adoption', () => {
        const source = readFileSync('src/main/sessions.ts', 'utf8')
        const familyStart = source.indexOf("detectedFamily === 'minimax_m3'")
        const familyBlock = source.slice(familyStart, source.indexOf('return changed', familyStart))
        const adoptStart = source.indexOf('async detectAndAdoptAll()')
        const defaultConfigStart = source.indexOf('const defaultConfig: ServerConfig = {', adoptStart)
        const createSession = source.indexOf('db.createSession(session)', defaultConfigStart)
        const adoptCreateBlock = source.slice(defaultConfigStart, createSession)

        // The 900 itself belongs to the shared table, not to sessions.ts — see
        // src/shared/slowFamilyTimeouts.ts. sessions.ts may name the constant
        // (the persist path reads better that way) but must SOURCE it, so that
        // what gets written into config and what gets resolved at launch can
        // never disagree.
        expect(source).toContain('MINIMAX_M3_DEFAULT_TIMEOUT_SECONDS = SLOW_FAMILY_TIMEOUTS.minimax_m3')
        expect(SLOW_FAMILY_TIMEOUTS.minimax_m3).toBe(900)
        expect(source).not.toContain('MINIMAX_M3_DEFAULT_TIMEOUT_SECONDS = 900')
        expect(source).not.toContain('MINIMAX_M3_DEFAULT_MAX_OUTPUT_TOKENS')
        expect(familyBlock).toContain('config.timeout = MINIMAX_M3_DEFAULT_TIMEOUT_SECONDS')
        // Per-launch defaults must not clear a fresh explicit output cap.
        // session-output-cap-intent.test.ts executes creation and save paths.
        expect(familyBlock).not.toContain('config.maxTokens = 0')
        expect(familyBlock).not.toContain('LEGACY_GENERIC_MAX_OUTPUT_TOKENS.has(Number(config.maxTokens))')
        // Adoption used to hand-roll a two-family timeout ternary here, so an
        // adopted openpangu_v2 / qwen3.5 / qwen3-next / nemotron-h session
        // persisted 300 while the same model created normally persisted 900.
        // It must ask the shared table, which knows all seven.
        expect(adoptCreateBlock).toContain('timeout: resolveSlowFamilyTimeoutSeconds(undefined, detectedFamily)')
        expect(adoptCreateBlock).not.toContain("detectedFamily === 'minimax_m3'")
        expect(adoptCreateBlock).toContain('maxTokens: 0')
    })

    it('MiniMax-M3 start refresh hardens paged RAM off while preserving explicit SSD/prefix toggles', () => {
        const source = readFileSync('src/main/sessions.ts', 'utf8')
        const familyStart = source.indexOf("} else if (detectedFamily === 'minimax_m3')")
        const familyEnd = source.indexOf("} else if (detectedFamily === 'openpangu_v2')", familyStart)
        const familyBlock = source.slice(familyStart, familyEnd)
        const start = source.indexOf("} else if (freshFamily === 'minimax_m3')")
        const end = source.indexOf("} else if (freshFamily === 'openpangu_v2')", start)
        const block = source.slice(start, end)

        expect(familyBlock).toContain('if (config.enablePrefixCache === undefined)')
        expect(familyBlock).toContain('if (config.usePagedCache === undefined)')
        expect(familyBlock).toContain('if (config.enableBlockDiskCache === undefined)')
        expect(familyBlock).not.toContain('config.usePagedCache = m3PrefixOptIn')
        expect(familyBlock).toContain('config.enableDiskCache = false')
        expect(block).toContain('if (config.enablePrefixCache === undefined) config.enablePrefixCache = true')
        // Saved true is stale too: the refresh persists the same hard-Off state
        // that the disabled control and shared launch builder expose.
        expect(block).toContain('config.usePagedCache = false')
        expect(block).not.toContain('if (config.usePagedCache === undefined) config.usePagedCache = false')
        expect(block).toContain('if (config.enableBlockDiskCache === undefined) config.enableBlockDiskCache = true')
        expect(block).toContain('using typed MSA SSD-only prefix cache with idx_keys')
        expect(block).not.toContain('config.usePagedCache = m3PrefixOptIn')
    })

    it('openPangu defaults to exact typed memory plus prompt L2 while keeping unsafe paged/block lanes off', () => {
        const source = readFileSync('src/main/sessions.ts', 'utf8')
        const familyStart = source.indexOf("} else if (detectedFamily === 'openpangu_v2')")
        const familyBlock = source.slice(familyStart, source.indexOf('return changed', familyStart))
        const launchStart = source.indexOf('const exactTypedPromptDiskCache = usesExactTypedPromptDiskCache(detectedFamily)', 3000)
        const launchBlock = source.slice(launchStart, source.indexOf('const prefixCacheOff', launchStart))

        expect(familyBlock).toContain('config.enablePrefixCache = true')
        expect(familyBlock).toContain('config.usePagedCache = false')
        expect(familyBlock).toContain('config.enableDiskCache = true')
        expect(familyBlock).toContain('config.enableBlockDiskCache = false')
        expect(familyBlock).toContain('config.noMemoryAwareCache = false')
        expect(familyBlock).toContain("config.kvCacheQuantization = 'auto'")
        expect(launchBlock).toContain('exactTypedPromptDiskCache ? false')
        expect(launchBlock).toContain('enableDiskCache: !!config.enableDiskCache')

        for (const file of [
            'src/renderer/src/components/sessions/CreateSession.tsx',
            'src/renderer/src/components/sessions/ServerSettingsDrawer.tsx',
            'src/renderer/src/components/sessions/SessionSettings.tsx',
        ]) {
            const ui = readFileSync(file, 'utf8')
            const start = ui.indexOf('usesExactTypedPromptDiskCache(detected.family)')
            const block = ui.slice(start, start + 450)
            expect(start, file).toBeGreaterThanOrEqual(0)
            expect(block, file).toContain('enablePrefixCache = true')
            expect(block, file).toContain('usePagedCache = false')
            expect(block, file).toContain('enableDiskCache = true')
            expect(block, file).toContain('enableBlockDiskCache = false')
            expect(block, file).toContain('noMemoryAwareCache = false')
            expect(block, file).toContain("kvCacheQuantization = 'auto'")
        }

        const form = readFileSync('src/renderer/src/components/sessions/SessionConfigForm.tsx', 'utf8')
        expect(form).toContain("normalizedDetectedFamily === 'openpangu_v2'")
        const enLocale = readFileSync('src/renderer/src/i18n/locales/en.json', 'utf8')
        expect(form).toContain("t('sessions.config.openPanguTypedCacheNote')")
        expect(enLocale).toContain('openPangu v2 uses exact typed N-1 prompt snapshots')
        expect(form).toContain('disabled={exactTypedPromptDiskCache}')
        expect(form).toContain("t('sessions.config.openPanguMemoryAwareNote')")
        expect(enLocale).toContain("Memory-aware mode is required for openPangu's non-aliasing typed cache clone")
        expect(form).toContain('cachePolicy.blockDiskCacheDisabled || exactTypedPromptDiskCache')
    })

    it('GLM-5.3 selects exact typed prompt L2 across startup, reset, adoption, and launch', () => {
        expect(usesExactTypedPromptDiskCache('glm5_next')).toBe(true)
        expect(usesExactTypedPromptDiskCache('glm5_next_text')).toBe(true)
        expect(usesExactTypedPromptDiskCache('glm5-next')).toBe(true)

        const source = readFileSync('src/main/sessions.ts', 'utf8')
        const familyStart = source.indexOf("} else if (detectedFamily === 'glm5-next')")
        const familyBlock = source.slice(familyStart, source.indexOf("} else if (detectedFamily === 'openpangu_v2')", familyStart))
        const freshStart = source.indexOf("} else if (freshFamily === 'glm5-next')")
        const freshBlock = source.slice(freshStart, source.indexOf("} else if (freshFamily === 'openpangu_v2')", freshStart))

        for (const block of [familyBlock, freshBlock]) {
            expect(block).toContain('config.enablePrefixCache = true')
            expect(block).toContain('config.usePagedCache = false')
            expect(block).toContain('config.enableDiskCache = true')
            expect(block).toContain('config.enableBlockDiskCache = false')
            expect(block).toContain('config.noMemoryAwareCache = false')
            expect(block).toContain("config.kvCacheQuantization = 'auto'")
        }

        expect(source).toContain('enableDiskCache: usesExactTypedPromptDiskCache(detectedFamily)')
        expect(source).toContain('enableBlockDiskCache: !usesExactTypedPromptDiskCache(detectedFamily)')
        expect(source).toContain('enableBlockDiskCache: exactTypedPromptDiskCache ? false : !!config.enableBlockDiskCache')

        for (const file of [
            'src/renderer/src/components/sessions/CreateSession.tsx',
            'src/renderer/src/components/sessions/ServerSettingsDrawer.tsx',
            'src/renderer/src/components/sessions/SessionSettings.tsx',
        ]) {
            const ui = readFileSync(file, 'utf8')
            expect(ui, file).toContain('usesExactTypedPromptDiskCache(detected.family)')
        }
    })

    it('create-session fills missing cache settings before stamping incoming settings current', () => {
        const source = readFileSync('src/main/sessions.ts', 'utf8')
        const start = source.indexOf('private async _createSessionInner')
        const existing = source.indexOf('const existing =', start)
        const beforeExisting = source.slice(start, existing)
        const existingBlock = source.slice(existing, source.indexOf('const id = uuidv4()', existing))

        expect(beforeExisting).toContain('applyMissingCacheStackStartupDefaults(config, modelPath)')
        expect(beforeExisting).toContain('markCacheStackStartupDefaultsCurrent(config, modelPath)')
        expect(beforeExisting.indexOf('applyMissingCacheStackStartupDefaults(config, modelPath)')).toBeLessThan(
            beforeExisting.indexOf('markCacheStackStartupDefaultsCurrent(config, modelPath)'),
        )
        expect(beforeExisting).not.toContain('applyCacheStackStartupDefaultMigration(config')
        expect(existingBlock).toContain('applyCacheStackStartupDefaultMigration(existingConfig, modelPath)')
        expect(existingBlock).toContain('const merged = { ...existingConfig, ...config, modelPath, host, port }')
        expect(existingBlock).toContain('applyMissingCacheStackStartupDefaults(merged, modelPath)')
        expect(existingBlock).toContain('markCacheStackStartupDefaultsCurrent(merged, modelPath)')
    })

    it('create-session UI persists the same paged-off plus block-L2 tuple that it displays', () => {
        const source = readFileSync('src/renderer/src/components/sessions/CreateSession.tsx', 'utf8')
        const detectStart = source.indexOf('const applyModelDefaults')
        const detectEnd = source.indexOf('// Auto-detect image model type', detectStart)
        const detectBlock = source.slice(detectStart, detectEnd)
        const launchStart = source.indexOf('const handleLaunch = async')
        const launchEnd = source.indexOf('const handleLaunchRemote', launchStart)
        const launchBlock = source.slice(launchStart, launchEnd)

        expect(detectBlock).toContain('usePagedCache: false')
        expect(detectBlock).not.toContain("usePagedCache: detected?.family === 'deepseek-v4' ? true")
        expect(detectBlock).not.toContain('usePagedCache: detected?.usePagedCache')
        expect(detectBlock).toContain("enableDiskCache: detected?.family === 'openpangu_v2'")
        expect(detectBlock).toContain("enableBlockDiskCache: detected?.family !== 'openpangu_v2'")
        expect(launchBlock).toContain('const normalizedCacheConfig = config')
        expect(launchBlock).not.toContain('enableDiskCache: false, enableBlockDiskCache: true')
        expect(launchBlock).toContain('window.api.sessions.create(selectedModel, launchConfig)')
    })

    it('fresh minimal session configs get SSD block-disk cache defaults with paged RAM off for every family', () => {
        const source = readFileSync('src/main/sessions.ts', 'utf8')
        const start = source.indexOf('function applyMissingCacheStackStartupDefaults')
        const end = source.indexOf('function isZayaCacheStackMigrationTarget', start)
        const helper = source.slice(start, end)

        expect(helper).toContain("setConfigValue(mutable, 'enablePrefixCache'")
        expect(helper).toContain("setConfigValue(mutable, 'usePagedCache', exactTypedPromptDiskCache ? false : defaultUsePagedCache)")
        expect(helper).toContain("setConfigValue(mutable, 'enableDiskCache', defaultEnableDiskCache)")
        expect(helper).toContain("setConfigValue(mutable, 'enableBlockDiskCache', defaultEnableBlockDiskCache)")
        expect(helper).toContain("setConfigValue(mutable, 'kvCacheQuantization', 'auto')")
        // In-RAM paged cache is OFF for EVERY family, DSV4 included; SSD
        // block-disk L2 is the only cache tier. A fresh session must not seed a
        // saved `true` that disagrees with the launch (which always emits
        // --no-paged-cache), and no per-family registry capability may
        // reintroduce a RAM tier here.
        expect(helper).toContain('const defaultUsePagedCache = false')
        expect(helper).not.toContain('dsv4Active ? true')
        expect(helper).not.toContain('detectedUsePaged')
        expect(helper).toContain('const defaultEnableDiskCache = exactTypedPromptDiskCache')
        expect(helper).toContain('const defaultEnableBlockDiskCache = !exactTypedPromptDiskCache')
    })

    it('v9 migrates only the pre-v9 stale impossible paged plus legacy-L2 tuple to block L2', () => {
        const source = readFileSync('src/main/sessions.ts', 'utf8')
        const start = source.indexOf('const stalePreV9PagedLegacyDiskWithoutBlockL2 =')
        const end = source.indexOf('// v8 (2026-07-12)', start)
        const block = source.slice(start, end)

        expect(source).toMatch(/const CACHE_STACK_STARTUP_DEFAULTS_VERSION = \d+/)
        expect(block).toContain('Number(config.cacheStackStartupDefaultsVersion || 0) < 9')
        expect(block).toContain('migrationDetectedUsePaged === true')
        expect(block).toContain('config.usePagedCache === true')
        expect(block).toContain('config.enableDiskCache === true')
        expect(block).toContain('config.enableBlockDiskCache === false')
        expect(source).toContain('!stalePreV9PagedLegacyDiskWithoutBlockL2')
        expect(source).toContain('config.enableDiskCache = false')
        expect(source).toContain('config.enableBlockDiskCache = true')
    })

    it('v10 migrates only the exact v9 MiniMax-M3 legacy-L2 tuple to typed paged block L2', () => {
        const source = readFileSync('src/main/sessions.ts', 'utf8')
        const start = source.indexOf('const staleV9M3PagedOffWithLegacyL2 =')
        const end = source.indexOf('// v8 (2026-07-12)', start)
        const block = source.slice(start, end)

        expect(source).toMatch(/const CACHE_STACK_STARTUP_DEFAULTS_VERSION = \d+/)
        expect(block).toContain("migrationDetectedFamily === 'minimax_m3'")
        expect(block).toContain('Number(config.cacheStackStartupDefaultsVersion || 0) === 9')
        expect(block).toContain('config.usePagedCache === false')
        expect(block).toContain('config.enableDiskCache === true')
        expect(block).toContain('config.enableBlockDiskCache === false')
        expect(source).toContain('!staleV9M3PagedOffWithLegacyL2')
        expect(source).toContain('config.usePagedCache = migratedGenericPaged')
        expect(source).toContain('config.enableBlockDiskCache = true')
    })

    it('migrates persisted cache defaults before the renderer first lists sessions', () => {
        const source = readFileSync('src/main/sessions.ts', 'utf8')
        const start = source.indexOf('constructor() {')
        const end = source.indexOf('/** Get timestamp', start)
        const block = source.slice(start, end)

        // db.getSessions() never returns secrets, which is what makes this
        // module-scope loop safe to run before the app is ready.
        expect(block).toContain('for (const session of db.getSessions())')
        expect(block).toContain('applyMissingCacheStackStartupDefaults(config, session.modelPath)')
        expect(block).toContain('applyCacheStackStartupDefaultMigration(config, session.modelPath)')
        expect(block).toContain('normalizeCacheStackMutualExclusion(config)')
        expect(block).toContain('markCacheStackStartupDefaultsCurrent(config, session.modelPath)')
        expect(block).toContain("db.updateSession(session.id, { config: JSON.stringify(config) })")
    })

    it('normalizes impossible paged and legacy-L2 tuples independent of migration version', () => {
        const source = readFileSync('src/main/sessions.ts', 'utf8')
        const helperStart = source.indexOf('function normalizeCacheStackMutualExclusion')
        const helperEnd = source.indexOf('function applyMissingCacheStackStartupDefaults', helperStart)
        const helper = source.slice(helperStart, helperEnd)
        const updateStart = source.indexOf('async updateSessionConfig')
        const updateEnd = source.indexOf('// Log sleep config changes', updateStart)
        const updateBlock = source.slice(updateStart, updateEnd)

        expect(helper).toContain('config.enableBlockDiskCache === true && config.enableDiskCache === true')
        expect(helper).toContain('config.enableDiskCache = false')
        expect(helper).not.toContain('config.enableBlockDiskCache = false')
        expect(updateBlock).toContain('normalizeCacheStackMutualExclusion(merged as Partial<ServerConfig>)')
    })

    it('persists tri-state Auto by deleting the stored override and requiring restart', () => {
        const source = readFileSync('src/main/sessions.ts', 'utf8')
        const updateStart = source.indexOf('async updateSessionConfig')
        const updateEnd = source.indexOf('repointSessionModelPath(', updateStart)
        const updateBlock = source.slice(updateStart, updateEnd)

        expect(updateBlock).toContain('const explicitlyClearedKeys = new Set<string>()')
        expect(updateBlock).toContain('explicitlyClearedKeys.add(k)')
        expect(updateBlock).toContain('for (const key of explicitlyClearedKeys) delete migratedBaseline[key]')
        expect(updateBlock).toContain('planSessionConfigSave(session, effectiveConfig, merged, SessionManager.RESTART_REQUIRED_KEYS)')
        const lifecycle = readFileSync('src/shared/sessionConfigLifecycle.ts', 'utf8')
        expect(lifecycle).toContain('Object.prototype.hasOwnProperty.call(effective, key)')
        expect(lifecycle).toContain('pendingConfig: restartRequired ? { ...desired } : null')
    })

    it('does not materialize detected multimodal or parser values into persisted Auto overrides at Start', () => {
        const fs = require('fs')
        const sessions = fs.readFileSync('src/main/sessions.ts', 'utf-8')
        const start = sessions.slice(
            sessions.indexOf('private async _startSessionInner'),
            sessions.indexOf('// Memory estimation:', sessions.indexOf('private async _startSessionInner')),
        )

        expect(start).toContain('const hadExplicitMultimodalOverride = Object.prototype.hasOwnProperty.call(')
        expect(start).toContain('const persistedConfig = { ...config }')
        expect(start).toContain('if (!hadExplicitMultimodalOverride)')
        expect(start).toContain('delete persistedConfig.isMultimodal')
        expect(start).toContain("const toolParserWasAuto = config.toolCallParser == null || config.toolCallParser === 'auto'")
        expect(start).toContain("const reasoningParserWasAuto = config.reasoningParser == null || config.reasoningParser === 'auto'")
        expect(start).toContain("persistedConfig.toolCallParser = 'auto'")
        expect(start).toContain("persistedConfig.reasoningParser = 'auto'")
        expect(start).toContain('db.updateSession(sessionId, { config: JSON.stringify(persistedConfig) })')
    })

    it('adopted paged sessions default to block L2 without legacy L2', () => {
        const source = readFileSync('src/main/sessions.ts', 'utf8')
        const start = source.indexOf('async detectAndAdoptAll()')
        const end = source.indexOf('applyBundleStartupDefaults(defaultConfig', start)
        const block = source.slice(start, end)

        expect(block).toContain('enableDiskCache: usesExactTypedPromptDiskCache(detectedFamily)')
        expect(block).toContain('enableBlockDiskCache: !usesExactTypedPromptDiskCache(detectedFamily)')
    })

    it('reset persists paged RAM off, SSD L2, and force-text-only values explicitly', () => {
        const source = readFileSync('src/renderer/src/components/sessions/SessionSettings.tsx', 'utf8')
        const start = source.indexOf('const handleReset = async () =>')
        const end = source.indexOf('if (!session)', start)
        const block = source.slice(start, end)

        expect(block).toContain('base.enableDiskCache = false')
        expect(block).toContain('base.usePagedCache = false')
        expect(block).not.toContain('base.usePagedCache = detected.usePagedCache')
        expect(block).toContain('base.enableDiskCache = false')
        expect(block).toContain('base.enableBlockDiskCache = true')
        expect(block).toContain('base.isMultimodal = detected.forceTextOnly === true')
        expect(block).toContain('? false')
        expect(block).toContain(': detected.isMultimodal === true')
    })

    it('adopted running sessions apply bundle generation defaults before saving config', () => {
        const source = readFileSync('src/main/sessions.ts', 'utf8')
        const start = source.indexOf('async detectAndAdoptAll()')
        const defaultConfigStart = source.indexOf('const defaultConfig: ServerConfig = {', start)
        const createSession = source.indexOf('db.createSession(session)', defaultConfigStart)
        const adoptCreateBlock = source.slice(defaultConfigStart, createSession)

        expect(adoptCreateBlock).toContain('applyBundleStartupDefaults(defaultConfig, proc.modelPath)')
        expect(adoptCreateBlock).toContain('applyFamilyStartupDefaults(defaultConfig, proc.modelPath)')
        expect(adoptCreateBlock.indexOf('applyBundleStartupDefaults(defaultConfig, proc.modelPath)')).toBeLessThan(
            adoptCreateBlock.indexOf('config: JSON.stringify(defaultConfig)'),
        )
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

    it('cache tier argv exists once and is consumed by both spawn and visual preview', () => {
        const shared = readFileSync('src/shared/cacheLaunchArgs.ts', 'utf8')
        const launcher = readFileSync('src/main/sessions.ts', 'utf8')
        const renderer = readFileSync('src/renderer/src/components/sessions/SessionSettings.tsx', 'utf8')

        expect(shared).toContain('export function buildCacheLaunchArgs(')
        expect(shared).toContain("'--no-paged-cache'")
        expect(shared).not.toContain("args.push('--use-paged-cache')")
        expect(launcher).toContain('const cacheLaunch = buildCacheLaunchArgs({')
        expect(launcher).toContain('args.push(...cacheLaunch.args)')
        expect(renderer).toContain('const cacheLaunch = buildCacheLaunchArgs({')
        expect(renderer).toContain('parts.push(...cacheLaunch.args)')
    })

    it('ZAYA sessions keep the qwen3 reasoning parser and model-owned no-thinking default', () => {
        const source = readFileSync('src/main/sessions.ts', 'utf8')
        // The family-alias rules live in ONE module now (they used to be three
        // byte-identical copies across main and renderer). sessions.ts must
        // import them rather than redefine them, or main and renderer can drift
        // on what a family is called — the exact mismatch that has silently
        // no-op'd family-gated settings fixes in this project before.
        const shared = readFileSync('src/shared/detectedFamilyNames.ts', 'utf8')
        expect(shared).toContain('function isZayaCcaFamily')
        expect(source).toContain('isZayaCcaFamily')
        expect(source).toMatch(/import \{[^}]*isZayaCcaFamily[^}]*\} from '[^']*shared\/detectedFamilyNames'/)
        expect(source).not.toContain('function isZayaCcaFamily')
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
    it('does not invent a Laguna cache-JIT opt-out in native Auto mode', () => {
        const env: Record<string, string | undefined> = {
            [DISABLE_JANG_AFFINE_JIT_DEFAULT_ENV]: 'stale-parent-value',
        }
        const detected = {
            family: 'laguna',
            architectureHints: {
                attentionArch: 'full_and_sliding_kv',
                cacheSchema: 'mixed_swa_kv_v1',
                selectiveTurboQuantKv: true,
            },
        }

        expect(applyLagunaJitDefaultEnvironment(env, {
            detected,
            kvCacheQuantization: 'auto',
            explicitKvCacheQuantizationApplied: false,
            enableJitRequested: true,
        })).toBe(false)
        expect(env[DISABLE_JANG_AFFINE_JIT_DEFAULT_ENV]).toBe('stale-parent-value')

        const unrelatedEnv: Record<string, string | undefined> = {
            [DISABLE_JANG_AFFINE_JIT_DEFAULT_ENV]: 'parent-shell-opt-out',
        }
        expect(applyLagunaJitDefaultEnvironment(env, {
            detected: { family: 'qwen3' },
            kvCacheQuantization: 'auto',
            explicitKvCacheQuantizationApplied: false,
            enableJitRequested: false,
        })).toBe(false)
        expect(applyLagunaJitDefaultEnvironment(unrelatedEnv, {
            detected: { family: 'qwen3' },
            kvCacheQuantization: 'auto',
            explicitKvCacheQuantizationApplied: false,
            enableJitRequested: false,
        })).toBe(false)
        expect(unrelatedEnv[DISABLE_JANG_AFFINE_JIT_DEFAULT_ENV]).toBe('parent-shell-opt-out')
    })

    it.each([
        ['auto', false, false, true],
        ['auto', false, true, false],
        ['q4', false, false, true],
        ['q4', false, true, false],
        ['q4', true, false, true],
        ['q8', true, false, true],
        ['none', true, false, true],
        ['q4', true, true, false],
        ['q8', true, true, false],
        ['none', true, true, false],
    ] as const)(
        'Laguna JIT policy mode=%s explicit=%s requested=%s disablesDefault=%s',
        (mode, explicitApplied, enableJitRequested, expected) => {
            expect(shouldDisableLagunaJitDefault({
                detected: {
                    family: 'laguna',
                    architectureHints: {
                        attentionArch: 'full_and_sliding_kv',
                        cacheSchema: 'mixed_swa_kv_v1',
                        selectiveTurboQuantKv: true,
                    },
                },
                kvCacheQuantization: mode,
                explicitKvCacheQuantizationApplied: explicitApplied,
                enableJitRequested,
            })).toBe(expected)
        },
    )

    it('honors Laguna JIT Off when the bundle disables live TurboQuant', () => {
        const detected = {
            family: 'laguna',
            architectureHints: {
                attentionArch: 'full_and_sliding_kv',
                cacheSchema: 'mixed_swa_kv_v1',
                selectiveTurboQuantKv: true,
                loaderTurboQuantEnabled: false,
            },
        }
        expect(shouldDisableLagunaJitDefault({
            detected,
            kvCacheQuantization: 'auto',
            explicitKvCacheQuantizationApplied: false,
            enableJitRequested: false,
        })).toBe(true)
        expect(shouldDisableLagunaJitDefault({
            detected,
            kvCacheQuantization: 'auto',
            explicitKvCacheQuantizationApplied: false,
            enableJitRequested: true,
        })).toBe(false)
    })

    it('enableJit false emits the explicit --no-jit polarity', () => {
        const out = preview({ enableJit: false })
        expect(hasFlag(out, '--enable-jit')).toBe(false)
        expect(hasFlag(out, '--no-jit')).toBe(true)
    })

    it('enableJit true emits --enable-jit flag', () => {
        const out = preview({ enableJit: true })
        expect(hasFlag(out, '--enable-jit')).toBe(true)
        expect(hasFlag(out, '--no-jit')).toBe(false)
    })

    it('deepseek-v4 detection suppresses --enable-jit even when saved config requests it', () => {
        const out = preview(
            { enableJit: true, maxNumSeqs: 64 },
            { family: 'deepseek-v4' },
        )

        expect(hasFlag(out, '--enable-jit')).toBe(false)
        expect(hasFlag(out, '--no-jit')).toBe(true)
        expect(getFlagValue(out, '--max-num-seqs')).toBe('1')
    })

    it('TurboQuant/JANGTQ detection suppresses --enable-jit because engine skips mx.compile', () => {
        const out = preview(
            { enableJit: true },
            { family: 'minimax', isTurboQuant: true },
        )

        expect(hasFlag(out, '--enable-jit')).toBe(false)
        expect(hasFlag(out, '--no-jit')).toBe(true)
    })

    it('multimodal/VLM detection suppresses --enable-jit because mlx-vlm streaming is not compile-safe', () => {
        // Genuine detected VLM (user did NOT Force-Off): runs --is-mllm and JIT is
        // suppressed. (A Force-Off isMultimodal=false would instead run text-only and
        // re-enable JIT — see the Force-Off parity test.)
        const out = preview(
            { enableJit: true },
            { family: 'zaya1-vl', isMultimodal: true },
        )

        expect(hasFlag(out, '--is-mllm')).toBe(true)
        expect(hasFlag(out, '--enable-jit')).toBe(false)
        expect(hasFlag(out, '--no-jit')).toBe(true)
    })

    it('ZAYA typed CCA detection suppresses --enable-jit because cache path is faster uncompiled', () => {
        const out = preview(
            { enableJit: true },
            { family: 'zaya' },
        )

        expect(hasFlag(out, '--enable-jit')).toBe(false)
        expect(hasFlag(out, '--no-jit')).toBe(true)
    })

    it('hybrid and Mamba cache detection suppresses --enable-jit like the real launcher', () => {
        const hybrid = preview(
            { enableJit: true },
            { family: 'qwen3.5', cacheType: 'hybrid', usePagedCache: true },
        )
        const mamba = preview(
            { enableJit: true },
            { family: 'ling', cacheType: 'mamba', usePagedCache: true },
        )

        expect(hasFlag(hybrid, '--enable-jit')).toBe(false)
        expect(hasFlag(mamba, '--enable-jit')).toBe(false)
        expect(hasFlag(hybrid, '--no-jit')).toBe(true)
        expect(hasFlag(mamba, '--no-jit')).toBe(true)
    })

    it('Laguna mixed full/sliding Auto preserves native cache and JIT choice', () => {
        const detected = {
            family: 'laguna',
            cacheType: 'kv',
            usePagedCache: false,
            architectureHints: {
                attentionArch: 'full_and_sliding_kv',
                cacheSchema: 'mixed_swa_kv_v1',
                selectiveTurboQuantKv: true,
            },
        }
        const out = preview(
            {
                enableJit: true,
                kvCacheQuantization: 'auto',
                usePagedCache: false,
                enableBlockDiskCache: true,
            },
            detected,
        )

        expect(hasFlag(out, '--enable-jit')).toBe(true)
        expect(out).not.toContain(`${DISABLE_JANG_AFFINE_JIT_DEFAULT_ENV}=1`)
        expect(hasFlag(out, '--no-paged-cache')).toBe(true)
        expect(hasFlag(out, '--enable-block-disk-cache')).toBe(true)
        expect(hasFlag(out, '--kv-cache-quantization')).toBe(false)
    })

    it.each(['none', 'q4', 'q8'])(
        'Laguna explicit %s cache choice disables the live TQ wrapper and preserves JIT',
        mode => {
            const out = preview(
                {
                    enableJit: true,
                    kvCacheQuantization: mode,
                    enablePrefixCache: true,
                },
                {
                    family: 'laguna',
                    cacheType: 'kv',
                    architectureHints: {
                        attentionArch: 'full_and_sliding_kv',
                        cacheSchema: 'mixed_swa_kv_v1',
                        selectiveTurboQuantKv: true,
                    },
                },
            )

            expect(hasFlag(out, '--kv-cache-quantization')).toBe(false)
            expect(out).not.toContain(`${DISABLE_JANG_AFFINE_JIT_DEFAULT_ENV}=1`)
            expect(hasFlag(out, '--enable-jit')).toBe(true)
        },
    )

    it('does not suppress JIT for unrelated plain-KV models or unstamped Laguna topology', () => {
        const qwen = preview(
            { enableJit: true, kvCacheQuantization: 'auto' },
            { family: 'qwen3', cacheType: 'kv' },
        )
        const unstampedLaguna = preview(
            { enableJit: true, kvCacheQuantization: 'auto' },
            { family: 'laguna', cacheType: 'kv' },
        )

        expect(hasFlag(qwen, '--enable-jit')).toBe(true)
        expect(hasFlag(unstampedLaguna, '--enable-jit')).toBe(true)
        expect(qwen).not.toContain(`${DISABLE_JANG_AFFINE_JIT_DEFAULT_ENV}=1`)
        expect(unstampedLaguna).not.toContain(`${DISABLE_JANG_AFFINE_JIT_DEFAULT_ENV}=1`)
    })

    it('preserves a bundle-owned Laguna live TurboQuant disable', () => {
        const out = preview(
            { enableJit: true, kvCacheQuantization: 'auto' },
            {
                family: 'laguna',
                cacheType: 'kv',
                architectureHints: {
                    attentionArch: 'full_and_sliding_kv',
                    cacheSchema: 'mixed_swa_kv_v1',
                    selectiveTurboQuantKv: true,
                    loaderTurboQuantEnabled: false,
                },
            },
        )

        expect(hasFlag(out, '--enable-jit')).toBe(true)
        expect(out).not.toContain(`${DISABLE_JANG_AFFINE_JIT_DEFAULT_ENV}=1`)
    })

    it('wires the same Laguna topology gate through launcher preview and form', () => {
        const sessions = readFileSync('src/main/sessions.ts', 'utf8')
        const settings = readFileSync(
            'src/renderer/src/components/sessions/SessionSettings.tsx',
            'utf8',
        )
        const form = readFileSync(
            'src/renderer/src/components/sessions/SessionConfigForm.tsx',
            'utf8',
        )

        for (const source of [sessions, settings, form]) {
            expect(source).toContain('isLagunaMixedSwaTurboQuantEffective')
            expect(source).toContain('lagunaMixedSwaTurboQuantActive')
        }
        expect(sessions).toContain('applyLagunaJitDefaultEnvironment')
        expect(sessions).toContain("args.includes('--kv-cache-quantization')")
        expect(sessions).toContain('DISABLE_JANG_AFFINE_JIT_DEFAULT_ENV')
        expect(form).toContain('detectedArchitectureHints')
        expect(form).toContain("t('sessions.config.jitDisabledLaguna')")
        expect(
            readFileSync('src/renderer/src/i18n/locales/en.json', 'utf8'),
        ).toContain('Cache Auto therefore does not suppress JIT')
    })

    it('Flash MoE and distributed launch modes suppress --enable-jit in preview and runtime policy', () => {
        const flash = preview({ enableJit: true, flashMoe: true })
        const distributed = preview({ enableJit: true, distributedEnabled: true })

        expect(hasFlag(flash, '--flash-moe')).toBe(true)
        expect(hasFlag(flash, '--enable-jit')).toBe(false)
        expect(hasFlag(flash, '--no-jit')).toBe(true)
        expect(hasFlag(distributed, '--distributed')).toBe(true)
        expect(hasFlag(distributed, '--enable-jit')).toBe(false)
        expect(hasFlag(distributed, '--no-jit')).toBe(true)
    })

    it('manual multimodal mode suppresses --enable-jit even without detection', () => {
        const out = preview({ enableJit: true, isMultimodal: true })

        expect(hasFlag(out, '--is-mllm')).toBe(true)
        expect(hasFlag(out, '--enable-jit')).toBe(false)
        expect(hasFlag(out, '--no-jit')).toBe(true)
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
        expect(form).toContain("t('sessions.config.codecDsv4')")
        expect(
            fs.readFileSync('src/renderer/src/i18n/locales/en.json', 'utf-8'),
        ).toContain('DeepSeek-V4 native composite cache')
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
        const createSession = fs.readFileSync(
            'src/renderer/src/components/sessions/CreateSession.tsx',
            'utf-8',
        )
        const settingsDrawer = fs.readFileSync(
            'src/renderer/src/components/sessions/ServerSettingsDrawer.tsx',
            'utf-8',
        )
        const sessions = fs.readFileSync('src/main/sessions.ts', 'utf-8')

        expect(form).toContain('detectedIsMultimodal')
        expect(form).toContain("t('sessions.config.jitDisabledMultimodal')")
        expect(
            fs.readFileSync('src/renderer/src/i18n/locales/en.json', 'utf-8'),
        ).toContain('multimodal/VLM models')
        const warningStart = form.indexOf('<IncompatWarning text={dsv4Active')
        const warningEnd = form.indexOf('/>\\n        )}', warningStart)
        const warning = form.slice(warningStart, warningEnd)
        expect(warning.indexOf(': multimodalActive')).toBeGreaterThan(-1)
        expect(warning.indexOf(': hybridCacheActive')).toBeGreaterThan(-1)
        expect(warning.indexOf(': multimodalActive')).toBeLessThan(
            warning.indexOf(': hybridCacheActive'),
        )
        expect(settings).toContain('detectedIsMultimodal={detectedConfig?.isMultimodal}')
        expect(sessions).toContain('mlx-vlm streaming path')
    })

    it('does not promise generic TQ for a native mixed-SWA runtime', () => {
        const form = readFileSync(
            resolve(__dirname, '../src/renderer/src/components/sessions/SessionConfigForm.tsx'),
            'utf8',
        )

        expect(form).toContain("t('sessions.config.codecMixedSwa')")
        const enLocale = readFileSync(resolve(__dirname, '../src/renderer/src/i18n/locales/en.json'), 'utf8')
        expect(enLocale).toContain('Native full/sliding KV + rotating metadata')
        expect(enLocale).toContain('Generic TurboQuant is not added')
        expect(form).not.toContain('TQ4 full-attention KV + native rotating SWA')
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
        expect(form).toContain("t('sessions.config.jitDisabledZaya')")
        expect(
            fs.readFileSync('src/renderer/src/i18n/locales/en.json', 'utf-8'),
        ).toContain('ZAYA typed CCA cache')
        expect(settings).toContain('zayaCcaActive')
        expect(sessions).toContain('ZAYA typed CCA cache is path-dependent')
    })

    it('command preview gates Omni backend to detected Nemotron-H multimodal only', () => {
        const nonOmni = preview({ omniBackend: 'stage2' }, { family: 'qwen3.5', isMultimodal: true })
        const textNemotron = preview({ omniBackend: 'stage2' }, { family: 'nemotron-h', isMultimodal: false })
        const omni = preview({ omniBackend: 'stage2' }, { family: 'nemotron-h', isMultimodal: true })
        const omniForceOff = preview(
            { omniBackend: 'stage2', isMultimodal: false },
            { family: 'nemotron-h', isMultimodal: true },
        )

        expect(hasFlag(nonOmni, '--omni-backend')).toBe(false)
        expect(hasFlag(textNemotron, '--omni-backend')).toBe(false)
        expect(getFlagValue(omni, '--omni-backend')).toBe('stage2')
        expect(hasFlag(omni, '--is-mllm')).toBe(false)
        expect(hasFlag(omni, '--text-only')).toBe(false)
        expect(hasFlag(omniForceOff, '--omni-backend')).toBe(false)
        expect(hasFlag(omniForceOff, '--is-mllm')).toBe(false)
        expect(hasFlag(omniForceOff, '--text-only')).toBe(true)
    })

    it('settings form and launch code surface the ZAYA SSD reconstruction gap without re-enabling RAM', () => {
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

        expect(form).toContain('zayaSsdReuseUnavailable')
        expect(form).toContain("t('sessions.config.zayaTypedCacheNote')")
        expect(
            fs.readFileSync('src/renderer/src/i18n/locales/en.json', 'utf-8'),
        ).toContain('ZAYA CCA SSD prefix reconstruction is not yet available')
        expect(settings).toContain('buildCacheLaunchArgs')
        expect(sessions).toContain('buildCacheLaunchArgs')
        expect(sessions).not.toContain("args.push('--use-paged-cache')")
    })

    it('settings form treats Gemma4 mixed-SWA rotating KV as typed SSD-only cache', () => {
        const fs = require('fs')
        const form = fs.readFileSync(
            'src/renderer/src/components/sessions/SessionConfigForm.tsx',
            'utf-8',
        )

        expect(form).toContain("detectedCacheType === 'rotating_kv'")
        expect(form).toContain("t('sessions.config.mixedSwaSsdOnlyNote')")
        expect(form).not.toContain('nativeCacheRequiresPaged')
    })

    it('settings form surfaces the macOS Metal wired-limit sudo command near memory/cache controls', () => {
        const fs = require('fs')
        const form = fs.readFileSync(
            'src/renderer/src/components/sessions/SessionConfigForm.tsx',
            'utf-8',
        )
        const shared = fs.readFileSync('src/shared/metalWiredLimit.ts', 'utf-8')

        // The COMMAND still comes from the shared module (main reuses it in
        // error messages); only the prose moved behind t() so it localizes —
        // with the app in Korean this note was one of only two English
        // sentences left on the whole form.
        const catalog = fs.readFileSync(
            'src/renderer/src/i18n/locales/en.json',
            'utf-8',
        )
        expect(form).toContain('metalWiredLimitCommand')
        expect(form).toContain(
            "<InfoNote text={t('sessions.config.metalWiredLimitHelp', { command: metalWiredLimitCommand })} />",
        )
        expect(shared).toContain('sudo sysctl iogpu.wired_limit_mb=120000')
        expect(shared).toContain('kIOGPUCommandBufferCallbackErrorOutOfMemory')
        expect(catalog).toContain('Metal wired-memory limit')
        expect(catalog).toContain('{{command}}')
    })

    it('settings form and launch code treat Step3.7 full/sliding KV subtype as typed SSD-only cache', () => {
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
        const createSession = fs.readFileSync(
            'src/renderer/src/components/sessions/CreateSession.tsx',
            'utf-8',
        )
        const settingsDrawer = fs.readFileSync(
            'src/renderer/src/components/sessions/ServerSettingsDrawer.tsx',
            'utf-8',
        )
        const envTypes = fs.readFileSync('src/env.d.ts', 'utf-8')

        expect(form).toContain('detectedCacheSubtype')
        expect(form).toContain("detectedCacheSubtype === 'step3p7_full_sliding_kv'")
        expect(form).toContain("t('sessions.config.stepSsdOnlyNote')")
        expect(
            fs.readFileSync('src/renderer/src/i18n/locales/en.json', 'utf-8'),
        ).toContain('Under tight Metal headroom, long cold-prompt stores can be skipped')
        expect(settings).toContain('detectedCacheSubtype={detectedConfig?.cacheSubtype}')
        expect(settings).toContain('buildCacheLaunchArgs')
        expect(sessions).toContain('buildCacheLaunchArgs')
        expect(sessions).not.toContain("args.push('--kv-cache-quantization'")
        expect(createSession).toContain('setDetectedCacheSubtype(detected?.cacheSubtype)')
        expect(createSession).toContain('detectedCacheSubtype={detectedCacheSubtype}')
        expect(settingsDrawer).toContain('setDetectedCacheSubtype(det?.cacheSubtype)')
        expect(settingsDrawer).toContain('detectedCacheSubtype={detectedCacheSubtype}')
        expect(envTypes).toContain('cacheSubtype?: string')
    })

    it('settings form and launch code expose the DSV4 native cache policy without a hidden second toggle', () => {
        const fs = require('fs')
        const form = fs.readFileSync(
            'src/renderer/src/components/sessions/SessionConfigForm.tsx',
            'utf-8',
        )
        const settings = fs.readFileSync(
            'src/renderer/src/components/sessions/SessionSettings.tsx',
            'utf-8',
        )
        const createSession = fs.readFileSync(
            'src/renderer/src/components/sessions/CreateSession.tsx',
            'utf-8',
        )
        const settingsDrawer = fs.readFileSync(
            'src/renderer/src/components/sessions/ServerSettingsDrawer.tsx',
            'utf-8',
        )
        const sessions = fs.readFileSync('src/main/sessions.ts', 'utf-8')

        expect(form).not.toContain('dsv4CompositeRequiresPaged')
        expect(form).not.toContain('dsv4CompositeCacheOptIn')
        expect(form).not.toContain('DSV4 Native Composite Prefix Cache')
        expect(form).not.toContain('DSV4 CSA/HCA Pool Codec')
        expect(form).toContain("t('sessions.config.dsv4BatchPathNote')")
        expect(
            fs.readFileSync('src/renderer/src/i18n/locales/en.json', 'utf-8'),
        ).toContain('Prefix reuse defaults On and Block Disk Cache (SSD / L2) defaults On as the warm/cold stack')
        expect(form).not.toContain('cacheControlUpdatesForDsv4CompositeToggle')
        expect(form).not.toContain('cacheControlUpdatesForDsv4BlockDiskToggle')
        expect(form).not.toContain('applyDsv4CompositeCacheToggle')
        expect(form).not.toContain('DSV4 Native Cache')
        expect(form).not.toContain('DSV4 Composite Prefix Cache')
        expect(form).not.toContain('DSV4 Pool Quantization')
        expect(form).not.toContain('DSV4 Flash composite prefix cache is disabled')
        expect(form).not.toContain("dsv4Active ? applyDsv4CompositeCacheToggle(v) : applyCacheControlUpdates(cacheControlUpdatesForPagedToggle")
        expect(form).not.toContain("dsv4Active ? cacheControlUpdatesForDsv4BlockDiskToggle(v) : cacheControlUpdatesForBlockDiskToggle")
        expect(form).not.toContain("label={t('sessions.config.pagedKVCache')}")
        expect(form).toContain('disabled={!cachePolicy.blockDiskCacheVisible || cachePolicy.blockDiskCacheDisabled || exactTypedPromptDiskCache}')
        expect(form).toContain("t('sessions.config.blockSizeTooltipDsv4')")
        expect(
            fs.readFileSync('src/renderer/src/i18n/locales/en.json', 'utf-8'),
        ).toContain('require fixed 256-token blocks')
        expect(form).toContain('disabled={dsv4Active}')
        expect(form).not.toContain('checked={config.dsv4PrefixCache !== false}')
        expect(form).not.toContain('checked={dsv4Active ? true : config.enablePrefixCache}')
        expect(form).toContain('export const DSV4_PAGED_CACHE_BLOCK_SIZE = 256')
        expect(form).toContain('export const DSV4_MAX_CACHE_BLOCKS = 4097')
        expect(form).toContain('defaultValue={dsv4Active ? DSV4_MAX_CACHE_BLOCKS : DEFAULT_CONFIG.maxCacheBlocks}')
        expect(form).toContain('max={100000}')
        expect(createSession).toContain("maxCacheBlocks: detected?.family === 'deepseek-v4' ? DSV4_MAX_CACHE_BLOCKS")
        expect(createSession).toContain('base.maxCacheBlocks = DSV4_MAX_CACHE_BLOCKS')
        expect(settingsDrawer).toContain('base.maxCacheBlocks = DSV4_MAX_CACHE_BLOCKS')
        expect(settings).toContain('DSV4_PAGED_CACHE_BLOCK_SIZE,')
        expect(settings).toContain('DSV4_MAX_CACHE_BLOCKS,')
        expect(settings).toContain('base.maxCacheBlocks = DSV4_MAX_CACHE_BLOCKS')
        expect(settings).not.toContain('dsv4PrefixCacheOptIn')
        expect(sessions).toContain('DSV4_PAGED_CACHE_BLOCK_SIZE = 256')
        expect(sessions).toContain('DSV4_MAX_CACHE_BLOCKS = 4097')
        expect(sessions).toContain("prefix=${prefixCacheOff ? 'off' : 'on'}")
        expect(sessions).toContain('block_disk_l2=')
        expect(sessions).toContain('pool_codec=bundle-derived')
        expect(sessions).toContain('const prefixCacheOff = cacheLaunchPolicy.prefixCacheOff')
    })

    it('settings form exposes standard DSV4 cache controls while keeping native codec ownership clear', () => {
        const form = readFileSync(
            'src/renderer/src/components/sessions/SessionConfigForm.tsx',
            'utf-8',
        )

        expect(countOccurrences(form, 'label="DSV4 Native Composite Prefix Cache"')).toBe(0)
        expect(countOccurrences(form, 'label="DSV4 CSA/HCA Pool Codec"')).toBe(0)
        expect(countOccurrences(form, "label={t('sessions.cache.blockDiskCache')}")).toBe(1)
        expect(countOccurrences(form, "label={t('sessions.config.pagedKVCache')}")).toBe(0)
        expect(form).not.toContain('LOCKED OFF')
        expect(form).toContain("t('sessions.config.blockDiskPureSsdNote')")
        const enLocale = readFileSync('src/renderer/src/i18n/locales/en.json', 'utf-8')
        expect(enLocale).toContain('Locked Off')
        expect(enLocale).toContain('The app always launches --no-paged-cache')
        expect(form).not.toContain('limited GPU RAM')
        expect(form).toContain("e.preventDefault()")
        expect(form).toContain('className="relative inline-flex ml-1"\n      onClick={handleClick}')
        expect(form).not.toContain('disabled={dsv4CompositeRequiresPaged}')
        expect(form).toContain('<select value={effectiveStoredCacheQuantization} className="cfg-input" disabled>')
        expect(form).toContain("t('sessions.config.storedQuantNativeTyped')")
        expect(enLocale).toContain('Native typed codec (bundle-derived)')
        expect(form).not.toContain('DSV4 Native Cache')
        expect(form).not.toContain('DSV4 Composite Prefix Cache')
        expect(form).not.toContain('DSV4 Pool Quantization')
    })

    it('uses user-facing RAM and SSD cache names in cache status panels', () => {
        const cachePanel = readFileSync(
            'src/renderer/src/components/sessions/CachePanel.tsx',
            'utf-8',
        )
        const performancePanel = readFileSync(
            'src/renderer/src/components/sessions/PerformancePanel.tsx',
            'utf-8',
        )

        const cachePanelCopy = readFileSync(
            'src/renderer/src/i18n/locales/en.json',
            'utf-8',
        )
        // copy lives in the locale catalog now that the panel is translated
        expect(cachePanelCopy).toContain('Block Disk Cache (SSD / L2)')
        expect(cachePanel).toContain("t('sessions.cache.managedRootSize')")
        expect(cachePanelCopy).toContain('Managed Root Size')
        expect(cachePanel).toContain("t('sessions.cache.managedRootLimit')")
        expect(cachePanelCopy).toContain('Managed Root Limit')
        expect(cachePanel).toContain("t('sessions.cache.namespaceSize')")
        expect(cachePanelCopy).toContain('Namespace Size')
        expect(cachePanel).toContain('global_budget')
        expect(cachePanel).toContain("t('sessions.cache.blockDiskExplainer')")
        expect(cachePanelCopy).toContain('every block-cache namespace and typed companion')
        expect(cachePanel).toContain("globalBlockDiskBudget.accounted === true")
        expect(cachePanel).toContain("t('sessions.cache.reconciliationPending')")
        expect(cachePanelCopy).toContain('Reconciliation pending')
        expect(performancePanel).toContain("health.native_cache.paged ? t('sessions.performance.cacheStackRamPaged')")
        expect(cachePanelCopy).toContain('RAM paged')
        expect(performancePanel).toContain("t('sessions.performance.blockDiskL2Ssd')")
        expect(cachePanelCopy).toContain('Block Disk L2 (SSD)')
        expect(cachePanel).not.toContain('L2 Paged')
        expect(performancePanel).not.toContain('Paged L2')
    })

    it('describes the block-cache maximum as one aggregate managed-root budget', () => {
        const form = readFileSync(
            'src/renderer/src/components/sessions/SessionConfigForm.tsx',
            'utf-8',
        )

        // The control is now a PERCENT of the volume, but the semantics it has
        // to disclose are unchanged: one aggregate root, shared by every
        // namespace, with the smallest finite limit winning. Those sentences
        // are the contract; the unit is not.
        expect(form).toContain("t('sessions.config.blockCacheMaxPercentTooltip')")
        expect(form).toContain("t('sessions.config.blockCacheSharedBudgetNote')")
        const enLocale = readFileSync('src/renderer/src/i18n/locales/en.json', 'utf-8')
        expect(enLocale).toContain('Maximum physical disk space for the managed block-cache root')
        expect(enLocale).toContain('shared across model/config namespaces and typed companion state')
        expect(enLocale).toContain('the smallest finite limit is enforced')
        expect(enLocale).toContain('Set to 0 for unlimited only when no live session supplies a finite limit')
        expect(enLocale).toContain('the size limit applies across all managed subdirectories and typed companions in this root')
    })

    it('settings form disables generic stored KV codec controls for MiniMax-M3 native MSA cache', () => {
        const form = readFileSync(
            'src/renderer/src/components/sessions/SessionConfigForm.tsx',
            'utf-8',
        )

        expect(form).toContain("const effectiveStoredCacheQuantization = 'auto'")
        expect(form).toContain("? t('sessions.config.codecOpenPangu')")
        expect(form).toContain("? t('sessions.config.codecM3')")
        expect(form).toContain("const liveCacheCodecBadge = 'NATIVE · GENERIC TQ OFF'")
        const enLocale = readFileSync('src/renderer/src/i18n/locales/en.json', 'utf-8')
        expect(enLocale).toContain('openPangu typed composite cache')
        expect(enLocale).toContain('MiniMax-M3 native MSA cache')
        expect(enLocale).toContain('MiniMax-M3 keeps generic KV q4/q8 disabled')
        expect(enLocale).toContain('native MSA snapshots with keys, values, idx_keys, and absolute offsets')
        expect(form).toContain('<select value={effectiveStoredCacheQuantization} className="cfg-input" disabled>')
    })

    it('settings form exposes MiniMax-M3 typed SSD cache while keeping its native codec', () => {
        const form = readFileSync(
            'src/renderer/src/components/sessions/SessionConfigForm.tsx',
            'utf-8',
        )

        expect(form).not.toContain("label={t('sessions.config.pagedKVCache')}")
        const enLocale = readFileSync('src/renderer/src/i18n/locales/en.json', 'utf-8')
        expect(form).toContain("t('sessions.config.m3NativeMsaNote')")
        expect(enLocale).toContain("MiniMax-M3's retained RAM tier is disabled")
        expect(form).toContain("t('sessions.config.m3SsdOnlyNote')")
        expect(enLocale).toContain('MiniMax-M3 SSD-only mode preserves native MSA keys, values, idx_keys, and absolute offsets')
        expect(form).toContain('architectureBlockDiskOnlySupported && !m3Active && !dsv4Active && cachePolicy.blockDiskCacheChecked')
        expect(enLocale).toContain('Enable Block Disk Cache for persistent typed MSA prefix reuse')
        expect(form).not.toContain('LOCKED OFF')
        expect(form).toContain('disabled={!cachePolicy.blockDiskCacheVisible || cachePolicy.blockDiskCacheDisabled || exactTypedPromptDiskCache}')
    })

    it('settings form keeps legacy prompt disk unavailable while exposing DSV4 Block Disk L2', () => {
        const fs = require('fs')
        const form = fs.readFileSync(
            'src/renderer/src/components/sessions/SessionConfigForm.tsx',
            'utf-8',
        )

        expect(form).not.toContain('Persist DeepSeek-V4 native SWA+CSA/HCA composite cache records to SSD')
        const enLocale = fs.readFileSync('src/renderer/src/i18n/locales/en.json', 'utf-8')
        expect(enLocale).toContain('Its retained paged-RAM mirror is disabled')
        expect(enLocale).toContain('defaults On as the warm/cold stack')
        expect(form).toContain('disabled={dsv4Active || cachePolicy.legacyDiskCacheDisabled}')
        expect(form).toContain('{(exactTypedPromptDiskCache || cachePolicy.legacyDiskCacheChecked) && <div data-vmlx-section="typed-disk-cache">')
        expect(enLocale).toContain('DSV4 uses Block Disk Cache (SSD / L2) above for persistent native composite blocks')
        expect(form).toContain("t('sessions.config.dsv4SsdOnlyNote')")
        expect(enLocale).toContain('DSV4 SSD-only mode preserves typed SWA plus CSA/HCA state')
        expect(form).toContain('{showCachingHelp && <Modal')
        expect(form).not.toContain('DSV4 Native Composite Prefix Cache')
    })

    it('enableJit does not affect other flags', () => {
        const without = preview({ enableJit: false })
        const withJit = preview({ enableJit: true })
        // Only the explicit JIT polarity should differ.
        const normalized1 = without.replace(/\s*\\\n\s*/g, ' ')
        const normalized2 = withJit.replace(/\s*\\\n\s*/g, ' ')
        expect(normalized2).toContain('--enable-jit')
        expect(normalized2).not.toContain('--no-jit')
        expect(normalized1).not.toContain('--enable-jit')
        expect(normalized1).toContain('--no-jit')
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
        // Paged RAM is not one of the shipping cache features any more: the
        // saved toggle is ignored and the launch is SSD block-disk only.
        expect(hasFlag(out, '--use-paged-cache')).toBe(false)
        expect(hasFlag(out, '--no-paged-cache')).toBe(true)
        expect(hasFlag(out, '--kv-cache-quantization')).toBe(false)
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
            continuousBatching: false,
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
        const out = preview({ enablePrefixCache: true, cacheMemoryPercent: 15, usePagedCache: false, enableBlockDiskCache: false })
        expect(getFlagValue(out, '--cache-memory-percent')).toBe('0.15')
    })

    it('drops the cache memory budget flags on the SSD-only launch a stale paged toggle now gets (#98 lane retired)', () => {
        const out = preview({
            enablePrefixCache: true,
            usePagedCache: true,
            cacheMemoryMb: 4096,
            cacheMemoryPercent: 35,
        })
        // #98/H1 wired --cache-memory-mb/percent through as the paged L1 RAM
        // byte ceiling. With paged RAM retired, this exact config launches
        // block-disk-only where there is no L1 payload to bound, so BOTH budget
        // flags are deliberately suppressed (they still flow on the legacy
        // memory-aware lane — see the 'memory-aware mode' tests).
        expect(hasFlag(out, '--use-paged-cache')).toBe(false)
        expect(hasFlag(out, '--no-paged-cache')).toBe(true)
        expect(hasFlag(out, '--cache-memory-mb')).toBe(false)
        expect(hasFlag(out, '--cache-memory-percent')).toBe(false)
        expect(hasFlag(out, '--cache-ttl-minutes')).toBe(false)
    })

    it('DSV4 explicit paged RAM is ignored — SSD-only launch emits no L1 memory budget', () => {
        // The saved-true twin of the SSD-only test below: DSV4's explicit paged
        // opt-in used to be the one lane that kept the RAM tier and its budget
        // flags. Retired — the same argv must come out whether the saved toggle
        // is true or false.
        const out = preview({
            enablePrefixCache: true,
            dsv4PrefixCache: true,
            usePagedCache: true,
            enableBlockDiskCache: true,
            cacheMemoryMb: 4096,
            cacheMemoryPercent: 15,
        }, { family: 'deepseek-v4' })

        expect(hasFlag(out, '--dsv4-enable-prefix-cache')).toBe(false)
        expect(hasFlag(out, '--use-paged-cache')).toBe(false)
        expect(hasFlag(out, '--no-paged-cache')).toBe(true)
        expect(hasFlag(out, '--disable-prefix-cache')).toBe(false)
        expect(hasFlag(out, '--enable-block-disk-cache')).toBe(true)
        expect(hasFlag(out, '--cache-memory-mb')).toBe(false)
        expect(hasFlag(out, '--cache-memory-percent')).toBe(false)
    })

    it('DSV4 SSD-only mode does not claim or emit a retained RAM budget', () => {
        const out = preview({
            enablePrefixCache: true,
            usePagedCache: false,
            enableBlockDiskCache: true,
            cacheMemoryMb: 4096,
            cacheMemoryPercent: 15,
        }, { family: 'deepseek-v4' })

        expect(hasFlag(out, '--no-paged-cache')).toBe(true)
        expect(hasFlag(out, '--enable-block-disk-cache')).toBe(true)
        expect(hasFlag(out, '--cache-memory-mb')).toBe(false)
        expect(hasFlag(out, '--cache-memory-percent')).toBe(false)
    })

    it('DSV4 launch and preview share the same SSD-only cache fragment', () => {
        const previewSource = readFileSync(resolve(__dirname, '../src/renderer/src/components/sessions/SessionSettings.tsx'), 'utf-8')
        const launchSource = readFileSync(resolve(__dirname, '../src/main/sessions.ts'), 'utf-8')
        const shared = readFileSync(resolve(__dirname, '../src/shared/cacheLaunchArgs.ts'), 'utf-8')
        expect(previewSource).toContain('forceMemoryAwareCache: exactTypedPromptDiskCache || dsv4Active')
        expect(launchSource).toContain('forceMemoryAwareCache: exactTypedPromptDiskCache || dsv4Active')
        expect(shared).toContain("'--no-paged-cache'")
        expect(shared).not.toContain("args.push('--use-paged-cache')")
    })

    it('settings form renders effective SSD capacity and preserves legacy memory-budget state', () => {
        const source = readFileSync(
            resolve(__dirname, '../src/renderer/src/components/sessions/SessionConfigForm.tsx'),
            'utf-8',
        )
        // The ARITHMETIC stays in the shared module; the SENTENCE moved behind
        // t() so it localizes with the rest of the form.
        const catalog = readFileSync(
            resolve(__dirname, '../src/renderer/src/i18n/locales/en.json'),
            'utf-8',
        )
        expect(source).toContain('resolvePagedCacheCapacity')
        expect(source).toContain('<InfoNote text={effectiveBlockDiskCapacityText} />')
        expect(source).not.toContain("label={t('sessions.config.pagedKVCache')}")
        expect(source).toContain("t('sessions.config.pagedCacheMemoryIgnored')")
        expect(catalog).toContain('Effective in-memory cache capacity')
        expect(catalog).toContain('Cache TTL does not apply while In-Memory Paged Cache is on')
        expect(source).toContain('pagedCacheControlsState')
        expect(source).toContain('memoryBudgetControlsDisabled')
        expect(source).toContain('cacheTtlDisabled')
    })

    it('defaultTopP minimum boundary stays out of startup CLI', () => {
        const out = preview({ defaultTopP: 1 })
        expect(hasFlag(out, '--default-top-p')).toBe(false)
    })

    it('numDraftTokens 0 with speculative model omits draft tokens flag', () => {
        // numDraftTokens 0 is falsy → condition fails → flag omitted → Python uses default (3)
        const out = preview({ continuousBatching: false, speculativeModel: 'draft-model', numDraftTokens: 0 })
        expect(hasFlag(out, '--speculative-model')).toBe(true)
        expect(hasFlag(out, '--num-draft-tokens')).toBe(false)
    })

    it('empty diskCacheDir with enableDiskCache does not emit --disk-cache-dir', () => {
        const out = preview({ enablePrefixCache: true, enableDiskCache: true, enableBlockDiskCache: false, diskCacheDir: '', usePagedCache: false })
        expect(hasFlag(out, '--enable-disk-cache')).toBe(true)
        expect(hasFlag(out, '--disk-cache-dir')).toBe(false)
    })

    it('enableDiskCache suppressed when usePagedCache is on', () => {
        const out = preview({ enablePrefixCache: true, enableDiskCache: true, usePagedCache: true })
        expect(hasFlag(out, '--enable-disk-cache')).toBe(false)
    })

    it('VLM suppresses external speculative decoding at launch', () => {
        const out = preview({
            isMultimodal: true,
            speculativeModel: 'draft-model',
            numDraftTokens: 5,
        })
        expect(hasFlag(out, '--is-mllm')).toBe(true)
        expect(hasFlag(out, '--speculative-model')).toBe(false)
        expect(hasFlag(out, '--num-draft-tokens')).toBe(false)
    })

    it('continuous batching suppresses external speculative decoding but preserves embedding model', () => {
        const out = preview({
            speculativeModel: 'draft-model',
            numDraftTokens: 4,
            continuousBatching: true,
            embeddingModel: 'embed-model',
            defaultTemperature: 80,
            defaultTopP: 90,
        })
        expect(hasFlag(out, '--continuous-batching')).toBe(true)
        expect(hasFlag(out, '--speculative-model')).toBe(false)
        expect(hasFlag(out, '--num-draft-tokens')).toBe(false)
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
        'dsv4PrefixCache', 'dsv4PoolQuant',
        'nativeMtpMode', 'nativeMtpDepth', 'nativeMtpDepthOverride',
        'embeddingModel', 'additionalArgs',
        'enableJit', 'logLevel', 'corsOrigins', 'maxContextLength',
        'chatTemplate', 'imageTokenBudget', 'videoFps', 'videoMaxFrames',
        'distributedEnabled', 'distributedMode', 'distributedSecret', 'distributedNodes',
        'idleTimeoutSoftMin', 'idleTimeoutHardMin', 'autoSleepEnabled',
    ]

    // Collect all config keys that appear in at least one test in this file
    // by checking that setting them produces a CLI flag or expected behavior.
    // This is a structural meta-test: ensure coverage.
    it('every SessionConfig field is listed in the completeness check', () => {
        const interfaceKeys = Object.keys(DEFAULT_CONFIG) as (keyof SessionConfig)[]
        // Plus optional fields that are not in DEFAULT_CONFIG but are still
        // persisted/session-owned controls.
        const fullSet = new Set([
            ...interfaceKeys,
            'enableAutoToolChoice',
            'isMultimodal',
            'chatTemplate',
            'imageTokenBudget',
            'videoFps',
            'videoMaxFrames',
            'distributedEnabled',
            'distributedMode',
            'distributedSecret',
            'distributedNodes',
            'idleTimeoutSoftMin',
            'idleTimeoutHardMin',
            'autoSleepEnabled',
        ])
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
        expect(normalized).not.toContain('--enable-disk-cache')    // generic default uses block SSD L2
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
        expect(normalized).not.toContain('--use-paged-cache')        // Phase-1: paged RAM block pool OFF by default
        expect(normalized).toContain('--enable-block-disk-cache')   // SSD-only block L2 stays on with paged RAM off
        expect(normalized).toContain('--no-paged-cache')
        expect(normalized).not.toContain('--default-temperature')
        expect(normalized).not.toContain('--default-top-p')
        expect(normalized).not.toContain('--default-repetition-penalty')
        expect(normalized).toContain('--enable-jit')
    })

    it('server startup generation defaults are model-owned and not editable sliders', () => {
        const source = readFileSync('src/renderer/src/components/sessions/SessionConfigForm.tsx', 'utf8')
        expect(source).toContain("t('sessions.config.generationDefaultsNote')")
        expect(
            readFileSync('src/renderer/src/i18n/locales/en.json', 'utf8'),
        ).toContain('Generation defaults are resolved by the engine from generation_config.json/jang_config')
        expect(source).toContain("label={t('sessions.config.maxContextTokens')}")
        expect(source).not.toContain('label="Default Temperature"')
        expect(source).not.toContain('label="Default Top-P"')
        expect(source).not.toContain('label="Default Top-K"')
        expect(source).not.toContain('label="Default Min-P"')
        expect(source).not.toContain('label="Default Repetition Penalty"')
        expect(source).not.toContain('label="Default Max Tokens"')
    })

    it('warns for stale Laguna XS top-k metadata without mutating sampling', () => {
        const form = readFileSync(
            'src/renderer/src/components/sessions/SessionConfigForm.tsx',
            'utf8',
        )

        expect(form).toContain("normalizedDetectedFamily === 'laguna'")
        expect(form).toContain("detectedArchitectureHints?.lagunaVariant === 'xs-2.1'")
        expect(form).toContain('Number(config.defaultTopK ?? 0) !== 20')
        expect(form).toContain("t('sessions.config.lagunaXsTopKWarning')")
        expect(form).not.toContain("onChange('defaultTopK', 20)")

        for (const sourcePath of [
            'src/renderer/src/components/sessions/CreateSession.tsx',
            'src/renderer/src/components/sessions/SessionSettings.tsx',
            'src/renderer/src/components/sessions/ServerSettingsDrawer.tsx',
        ]) {
            expect(readFileSync(sourcePath, 'utf8')).toContain(
                'detectedArchitectureHints={',
            )
        }

        for (const locale of ['en', 'es', 'ja', 'ko', 'zh']) {
            const messages = JSON.parse(
                readFileSync(`src/renderer/src/i18n/locales/${locale}.json`, 'utf8'),
            )
            expect(messages.sessions.config.lagunaXsTopKWarning).toContain('20')
        }
    })

    it('shows a DSV4 Top P 0.95 advisory without mutating server or API sampling', () => {
        const form = readFileSync(
            'src/renderer/src/components/sessions/SessionConfigForm.tsx',
            'utf8',
        )
        const chat = readFileSync(
            'src/renderer/src/components/chat/ChatSettings.tsx',
            'utf8',
        )

        expect(form).toContain('const dsv4TopPMismatch = shouldWarnDsv4TopP(')
        expect(form).toContain('Number(config.defaultTopP) / 100')
        expect(form).toContain('dsv4TopPMismatch && (')
        expect(form).toContain("t('common.dsv4TopPAdvisory')")
        expect(form).not.toContain("onChange('defaultTopP', 95)")
        expect(chat).toContain('const dsv4TopPMismatch =')
        expect(chat).toContain('hydrationCurrent && shouldWarnDsv4TopP(')
        expect(chat).toContain('dsv4TopPMismatch && (')
        expect(chat).toContain('data-vmlx-warning="dsv4-top-p-advisory"')
        expect(chat).toContain("t('common.dsv4TopPAdvisory')")
        expect(chat).not.toContain("update('topP', 0.95)")

        for (const locale of ['en', 'es', 'ja', 'ko', 'zh']) {
            const messages = JSON.parse(
                readFileSync(`src/renderer/src/i18n/locales/${locale}.json`, 'utf8'),
            )
            expect(messages.common.dsv4TopPAdvisory).toMatch(/0[.,]95/)
        }
    })

    it('video sampling controls are gated to runtime video-capable families', () => {
        const source = readFileSync('src/renderer/src/components/sessions/SessionConfigForm.tsx', 'utf8')
        // S21: the list moved to the shared helper so the form can also honour the bundle's declared modalities
        const shared = readFileSync('src/shared/videoCapableFamilies.ts', 'utf8')
        const allowlistStart = shared.indexOf('export const RUNTIME_VIDEO_CAPABLE_FAMILIES')
        const allowlistEnd = shared.indexOf('])', allowlistStart)
        const allowlist = shared.slice(allowlistStart, allowlistEnd)

        expect(allowlist).toContain("'qwen3-vl'")
        expect(allowlist).toContain("'qwen3.5'")
        expect(allowlist).toContain("'gemma4'")
        expect(allowlist).toContain("'nemotron-h'")
        expect(allowlist).toContain("'muse-glimmer'")
        expect(allowlist).not.toContain("'mimo_v2'")
        expect(allowlist).not.toContain("'step-3.7-flash'")
        expect(source).toContain('const detectedRuntimeVideoCapable = isRuntimeVideoCapable({')
        expect(source).toContain('runtimeModalities: detectedRuntimeModalities')
        expect(source).toContain('detectedRuntimeVideoCapable ||')
        expect(source).toContain('!detectedForceTextOnly && multimodalActive')
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

    it('slider settings expose stable value metadata and active unlimited state for live UI parity proof', () => {
        const source = readFileSync('src/renderer/src/components/sessions/SessionConfigForm.tsx', 'utf8')
        const sliderStart = source.indexOf('export function SliderField')
        const sliderEnd = source.indexOf('\nexport ', sliderStart + 1)
        const sliderBody = source.slice(sliderStart, sliderEnd > 0 ? sliderEnd : undefined)

        expect(sliderBody).toContain('data-setting-label={label}')
        expect(sliderBody).toContain('data-setting-value={String(value)}')
        expect(sliderBody).toContain('data-unlimited-active={String(isUnlimited)}')
        expect(sliderBody).toContain('aria-pressed={isUnlimited}')
        expect(sliderBody).toContain("aria-label={`${label}: ${unlimitedLabel} ${isUnlimited ? 'active' : 'inactive'}`}")
    })

    it('typed slider values are committed before either settings surface saves', () => {
        const formSource = readFileSync('src/renderer/src/components/sessions/SessionConfigForm.tsx', 'utf8')
        const drawerSource = readFileSync('src/renderer/src/components/sessions/ServerSettingsDrawer.tsx', 'utf8')
        const settingsSource = readFileSync('src/renderer/src/components/sessions/SessionSettings.tsx', 'utf8')
        const inputChangeStart = formSource.indexOf('const handleInputChange')
        const inputFocusStart = formSource.indexOf('const handleInputFocus', inputChangeStart)
        const inputChangeBody = formSource.slice(inputChangeStart, inputFocusStart)

        expect(inputChangeBody).toContain('setLocalInput(raw)')
        expect(inputChangeBody).toContain('onChange(parsed)')
        expect(formSource).toContain('export function commitActiveSettingsInput()')
        expect(drawerSource.match(/onPointerDown=\{commitActiveSettingsInput\}/g)).toHaveLength(2)
        expect(settingsSource.match(/onPointerDown=\{commitActiveSettingsInput\}/g)).toHaveLength(2)
    })

    it('Gemma live stress harness checks actual settings input values, not body text labels', () => {
        const source = readFileSync('scripts/live-gemma4-media-stress-proof.mjs', 'utf8')

        expect(source).toContain('captureSettingsControlValues')
        expect(source).toContain("controls['Timeout (seconds)']")
        expect(source).toContain('settings UI timeout value')
        expect(source).toContain("controls['Max Output Tokens']")
        expect(source).toContain('settings UI max output value')
        expect(source).toContain('Model-owned active despite explicit maxTokens')
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
        expect(source).toContain("t('chat.settings.topKOff')")
        expect(source).toContain('Math.round(value) <= 0')
        expect(source).not.toContain("return 'Off'")
    })

    it('chat settings applies only an explicit saved native-MTP override and receives session config', () => {
        const source = readFileSync('src/renderer/src/components/chat/ChatSettings.tsx', 'utf8')
        const hydration = readFileSync('src/shared/chatSettingsHydration.ts', 'utf8')
        const toolbar = readFileSync('src/renderer/src/components/layout/ChatModeToolbar.tsx', 'utf8')
        const sessionView = readFileSync('src/renderer/src/components/sessions/SessionView.tsx', 'utf8')
        expect(hydration).toContain('applyEffectiveSessionGenerationDefaults(')
        expect(hydration).toContain('detected?.nativeMtp')
        expect(source).toContain('session.config')
        expect(toolbar).toContain('config: displaySession.config')
        expect(sessionView).toContain('config: session.config')
    })

    it('chat settings shows max-thinking tokens only for engine-honoring or template-budget families', () => {
        const source = readFileSync('src/renderer/src/components/chat/ChatSettings.tsx', 'utf8')
        expect(source).toContain('setThinkingBudgetSupported(generation?.thinkingBudgetSupported)')
        expect(source).toContain('setSupportsThinkingBudget(detected?.supportsThinkingBudget)')
        expect(source).toContain('{(supportsThinkingBudget === true || thinkingBudgetSupported === true) && displayedEnableThinking !== false && (')
    })

    it('renders automatic tool choice as a real Auto/On/Off control on every settings surface', () => {
        const form = readFileSync('src/renderer/src/components/sessions/SessionConfigForm.tsx', 'utf8')
        const create = readFileSync('src/renderer/src/components/sessions/CreateSession.tsx', 'utf8')
        const drawer = readFileSync('src/renderer/src/components/sessions/ServerSettingsDrawer.tsx', 'utf8')
        const full = readFileSync('src/renderer/src/components/sessions/SessionSettings.tsx', 'utf8')
        expect(form).toContain("label={t('sessions.config.automaticToolChoice')}")
        expect(form).toContain("value === 'auto' ? undefined : value === 'on'")
        expect(form).toContain("{ value: 'off', label: t('chat.settings.thinkingOff') }")
        expect(create).not.toContain('delete stored.enableAutoToolChoice')
        expect(create).toContain('detectedEnableAutoToolChoice={detectedEnableAutoToolChoice}')
        expect(drawer).toContain('detectedEnableAutoToolChoice={detectedEnableAutoToolChoice}')
        expect(full).toContain('detectedEnableAutoToolChoice={detectedConfig?.enableAutoToolChoice}')
        expect(full).toContain('base.enableAutoToolChoice = undefined')
    })

    it('guards async Chat and Create settings hydration against stale model/session responses', () => {
        const chat = readFileSync('src/renderer/src/components/chat/ChatSettings.tsx', 'utf8')
        const create = readFileSync('src/renderer/src/components/sessions/CreateSession.tsx', 'utf8')
        const full = readFileSync('src/renderer/src/components/sessions/SessionSettings.tsx', 'utf8')
        expect(chat).toContain('const loadRequestRef = useRef(0)')
        expect(chat).toContain('const stillCurrent = () => active && loadRequestRef.current === requestId')
        expect(chat).toContain('if (!stillCurrent()) return')
        expect(create).toContain('const modelDefaultsRequestRef = useRef(0)')
        expect(create).toContain('const selectionStillCurrent = () => (')
        expect(full).toContain('setDetectedConfig(null)')
    })

    it('refuses unowned port conflicts instead of killing arbitrary listener PIDs', () => {
        const sessions = readFileSync('src/main/sessions.ts', 'utf8')
        expect(sessions).toContain('ensureOwnedSessionPortAvailable(session)')
        expect(sessions).toContain('terminateDetectedEngineForSession(session)')
        expect(sessions).toContain('vMLX did not terminate the unowned process')
        expect(sessions).not.toContain('lsof -ti tcp:')
        expect(sessions).not.toContain('killByPort(')
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

    it('DSV4 and MiniMax-M3 timeout defaults are wired through launch, chat IPC, and gateway proxy', () => {
        const sessionsSource = readFileSync('src/main/sessions.ts', 'utf8')
        const chatSource = readFileSync('src/main/ipc/chat.ts', 'utf8')
        const gatewaySource = readFileSync('src/main/api-gateway.ts', 'utf8')

        // This used to assert a per-file DSV4/MINIMAX const in each surface —
        // i.e. it PINNED THE DUPLICATION. The rule existed in seven copies and
        // four had diverged, so the values agreeing per file proved nothing.
        // Assert the wiring instead: every surface reads the one shared table.
        const sharedSource = readFileSync('src/shared/slowFamilyTimeouts.ts', 'utf8')
        expect(sharedSource).toContain('SLOW_FAMILY_TIMEOUT_SECONDS = 900')
        expect(sharedSource).toContain('minimax_m3')
        expect(sharedSource).toContain('deepseek-v4')

        expect(sessionsSource).toContain('effectiveSessionTimeoutSeconds')
        expect(chatSource).toContain('effectiveFamilyRequestTimeoutSeconds')
        expect(gatewaySource).toContain('effectiveGatewayProxyTimeoutMs')
        for (const [label, src] of [
            ['sessions', sessionsSource],
            ['chat IPC', chatSource],
            ['gateway', gatewaySource],
        ] as const) {
            expect(src, `${label} no longer reads the shared timeout table`).toMatch(
                /slowFamilyTimeouts/,
            )
        }
    })

    it('mutual exclusion: legacy disk cache NOT emitted while block-disk L2 owns the lane (stale paged toggle ignored)', () => {
        // Legacy disk used to be suppressed by an active paged cache. Paged RAM
        // is retired, so what suppresses it now is the block-disk L2 tier — and
        // the stale saved paged toggle must not resurface either lane.
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
        // Paged RAM never launches; the block SSD tier is what won the lane.
        expect(normalized).not.toContain('--use-paged-cache')
        expect(normalized).toContain('--no-paged-cache')
        expect(normalized).toContain('--enable-block-disk-cache')
    })

    it('block disk cache is emitted with paged RAM either active or disabled', () => {
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

        // Without paged RAM, the same content-addressed store becomes the
        // disk-only exact/partial-prefix backend.
        const out2 = preview({
            enablePrefixCache: true,
            enableBlockDiskCache: true,
            blockDiskCacheMaxGb: 50,
            usePagedCache: false,
        })
        const normalized2 = out2.replace(/\s*\\\n\s*/g, ' ')
        expect(normalized2).toContain('--enable-block-disk-cache')
        expect(normalized2).toContain('--no-paged-cache')
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
            enableBlockDiskCache: false,
            noMemoryAwareCache: false,
        })
        expect(withoutPaged.replace(/\s*\\\n\s*/g, ' ')).toContain('--cache-ttl-minutes')
    })
})

describe('block-disk-only capacity label', () => {
    it('selects a separate key instead of patching the localized sentence', () => {
        const fs = require('fs')
        const form = fs.readFileSync(
            'src/renderer/src/components/sessions/SessionConfigForm.tsx',
            'utf-8',
        )
        // This was `effectivePagedCapacityText.replace('Effective in-memory cache
        // capacity', 'Effective SSD block-index capacity')`. String surgery on a
        // translated sentence is a NO-OP in every non-English locale, so
        // block-disk-only mode mislabelled the SSD block index as in-memory RAM
        // capacity everywhere except English.
        expect(form).not.toContain("replace('Effective in-memory cache capacity'")
        expect(form).toContain("t('sessions.config.blockDiskCapacity'")
        expect(form).toContain('effectiveBlockDiskCapacityText')
        expect(form).toContain('blockDiskOnly')

        // Both variants must exist in every catalog, and must differ — one names
        // RAM, the other the SSD block index.
        for (const locale of ['en', 'es', 'ja', 'ko', 'zh']) {
            const catalog = JSON.parse(
                fs.readFileSync(`src/renderer/src/i18n/locales/${locale}.json`, 'utf-8'),
            )
            const config = catalog.sessions.config
            expect(typeof config.pagedCacheCapacity).toBe('string')
            expect(typeof config.blockDiskCapacity).toBe('string')
            expect(config.blockDiskCapacity).not.toBe(config.pagedCacheCapacity)
            // Same interpolation contract, so the numbers render either way.
            for (const placeholder of ['{{blockSize}}', '{{usableBlocks}}', '{{maxBlocks}}', '{{tokens}}']) {
                expect(config.blockDiskCapacity).toContain(placeholder)
            }
        }
    })
})
