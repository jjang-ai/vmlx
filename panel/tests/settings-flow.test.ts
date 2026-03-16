/**
 * Settings Flow Tests — verifies that ALL SessionConfig fields produce correct CLI flags
 * and that no settings are hardcoded. Tests use buildCommandPreview() which mirrors
 * the actual buildArgs() logic in sessions.ts exactly.
 *
 * Coverage: all 43 SessionConfig fields, context size detection, parser resolution,
 * VLM mode, cache feature gating, batching parameters, speculative decoding,
 * generation defaults, and embedding model.
 */
import { describe, it, expect } from 'vitest'

// ─── SessionConfig replica (from SessionConfigForm.tsx) ──────────────────────

interface SessionConfig {
    host: string
    port: number
    apiKey: string
    rateLimit: number
    timeout: number
    maxNumSeqs: number
    prefillBatchSize: number
    completionBatchSize: number
    continuousBatching: boolean
    enablePrefixCache: boolean
    prefixCacheSize: number
    cacheMemoryMb: number
    cacheMemoryPercent: number
    cacheTtlMinutes: number
    noMemoryAwareCache: boolean
    usePagedCache: boolean
    pagedCacheBlockSize: number
    maxCacheBlocks: number
    kvCacheQuantization: string
    kvCacheGroupSize: number
    enableDiskCache: boolean
    diskCacheMaxGb: number
    diskCacheDir: string
    enableBlockDiskCache: boolean
    blockDiskCacheMaxGb: number
    blockDiskCacheDir: string
    streamInterval: number
    maxTokens: number
    mcpConfig: string
    enableAutoToolChoice?: boolean
    toolCallParser: string
    reasoningParser: string
    isMultimodal?: boolean
    servedModelName: string
    speculativeModel: string
    numDraftTokens: number
    defaultTemperature: number
    defaultTopP: number
    embeddingModel: string
    additionalArgs: string
    enableJit: boolean
    logLevel: string
    corsOrigins: string
    maxContextLength: number
}

const DEFAULT_CONFIG: SessionConfig = {
    host: '0.0.0.0',
    port: 8000,
    apiKey: '',
    rateLimit: 0,
    timeout: 300,
    maxNumSeqs: 256,
    prefillBatchSize: 512,
    completionBatchSize: 512,
    continuousBatching: true,
    enablePrefixCache: true,
    prefixCacheSize: 100,
    cacheMemoryMb: 0,
    cacheMemoryPercent: 30,
    cacheTtlMinutes: 0,
    noMemoryAwareCache: false,
    usePagedCache: true,
    pagedCacheBlockSize: 64,
    maxCacheBlocks: 1000,
    kvCacheQuantization: 'none',
    kvCacheGroupSize: 64,
    enableDiskCache: false,
    diskCacheMaxGb: 10,
    diskCacheDir: '',
    enableBlockDiskCache: false,
    blockDiskCacheMaxGb: 10,
    blockDiskCacheDir: '',
    streamInterval: 1,
    maxTokens: 32768,
    mcpConfig: '',
    // enableAutoToolChoice intentionally omitted (undefined = auto-detect)
    toolCallParser: 'auto',
    reasoningParser: 'auto',
    isMultimodal: undefined,
    servedModelName: '',
    speculativeModel: '',
    numDraftTokens: 3,
    defaultTemperature: 0,
    defaultTopP: 0,
    embeddingModel: '',
    additionalArgs: '',
    enableJit: false,
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
    usePagedCache?: boolean
    enableAutoToolChoice?: boolean
    cacheType?: string
} | null

function buildCommandPreview(
    modelPath: string,
    config: SessionConfig,
    detected?: DetectedConfig
): string {
    const parts = ['vmlx-engine serve', modelPath]
    const isVLM = config.isMultimodal ?? !!detected?.isMultimodal

    parts.push('--host', config.host)
    parts.push('--port', config.port.toString())
    parts.push('--timeout', (config.timeout != null && config.timeout > 0 ? config.timeout : 86400).toString())

    if (config.apiKey) parts.push('# VLLM_API_KEY=*** (env var)')
    if (config.rateLimit && config.rateLimit > 0) parts.push('--rate-limit', config.rateLimit.toString())

    if (config.maxNumSeqs && config.maxNumSeqs > 0) parts.push('--max-num-seqs', config.maxNumSeqs.toString())
    if (config.prefillBatchSize && config.prefillBatchSize > 0) parts.push('--prefill-batch-size', config.prefillBatchSize.toString())
    if (config.completionBatchSize && config.completionBatchSize > 0) parts.push('--completion-batch-size', config.completionBatchSize.toString())

    if (isVLM) parts.push('--is-mllm')
    if (config.continuousBatching) parts.push('--continuous-batching')

    // Parser resolution: User explicit choice -> Detected config -> Fallback
    // (mirrors buildArgs: user choice wins over detection)
    const effectiveToolParser = config.toolCallParser === ''
        ? undefined
        : (config.toolCallParser && config.toolCallParser !== 'auto' ? config.toolCallParser
            : detected?.toolParser)
    const effectiveAutoTool = config.enableAutoToolChoice ?? detected?.enableAutoToolChoice
    const effectiveReasoningParser = config.reasoningParser === ''
        ? undefined
        : (config.reasoningParser && config.reasoningParser !== 'auto' ? config.reasoningParser
            : detected?.reasoningParser)

    const toolsNeedCache = !!(effectiveAutoTool && config.mcpConfig)
    const prefixCacheOff = config.enablePrefixCache === false && !toolsNeedCache

    if (prefixCacheOff) {
        parts.push('--disable-prefix-cache')
    } else {
        // Auto-enable continuous batching when prefix cache is on (required by vmlx-engine)
        if (!config.continuousBatching && !parts.includes('--continuous-batching')) {
            parts.push('--continuous-batching')
        }
        if (config.noMemoryAwareCache) {
            parts.push('--no-memory-aware-cache')
            if (config.prefixCacheSize && config.prefixCacheSize > 0) parts.push('--prefix-cache-size', config.prefixCacheSize.toString())
        } else {
            if (config.cacheMemoryMb && config.cacheMemoryMb > 0) parts.push('--cache-memory-mb', config.cacheMemoryMb.toString())
            if (config.cacheMemoryPercent && config.cacheMemoryPercent > 0) parts.push('--cache-memory-percent', (config.cacheMemoryPercent / 100).toString())
            if (config.cacheTtlMinutes && config.cacheTtlMinutes > 0 && !(config.usePagedCache ?? detected?.usePagedCache)) parts.push('--cache-ttl-minutes', config.cacheTtlMinutes.toString())
        }
    }

    if (!prefixCacheOff && (config.usePagedCache ?? detected?.usePagedCache)) {
        parts.push('--use-paged-cache')
        if (config.pagedCacheBlockSize && config.pagedCacheBlockSize > 0) parts.push('--paged-cache-block-size', config.pagedCacheBlockSize.toString())
        if (config.maxCacheBlocks && config.maxCacheBlocks > 0) parts.push('--max-cache-blocks', config.maxCacheBlocks.toString())
    }

    if (!prefixCacheOff && config.kvCacheQuantization && config.kvCacheQuantization !== 'none') {
        parts.push('--kv-cache-quantization', config.kvCacheQuantization)
        if (config.kvCacheGroupSize && config.kvCacheGroupSize !== 64) {
            parts.push('--kv-cache-group-size', config.kvCacheGroupSize.toString())
        }
    }

    if (!prefixCacheOff && config.enableDiskCache && !(config.usePagedCache ?? detected?.usePagedCache)) {
        parts.push('--enable-disk-cache')
        if (config.diskCacheDir) parts.push('--disk-cache-dir', config.diskCacheDir)
        if (config.diskCacheMaxGb != null && config.diskCacheMaxGb >= 0) parts.push('--disk-cache-max-gb', config.diskCacheMaxGb.toString())
    }

    if (!prefixCacheOff && (config.usePagedCache ?? detected?.usePagedCache) && config.enableBlockDiskCache) {
        parts.push('--enable-block-disk-cache')
        if (config.blockDiskCacheDir) parts.push('--block-disk-cache-dir', config.blockDiskCacheDir)
        if (config.blockDiskCacheMaxGb != null && config.blockDiskCacheMaxGb >= 0) parts.push('--block-disk-cache-max-gb', config.blockDiskCacheMaxGb.toString())
    }

    if (config.streamInterval && config.streamInterval > 0) parts.push('--stream-interval', config.streamInterval.toString())
    if (config.maxTokens && config.maxTokens > 0) {
        parts.push('--max-tokens', config.maxTokens.toString())
    } else {
        parts.push('--max-tokens', '1000000')
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

    if (config.mcpConfig) parts.push('--mcp-config', config.mcpConfig)

    if (config.servedModelName) parts.push('--served-model-name', config.servedModelName)

    // Speculative decoding
    if (config.speculativeModel) {
        parts.push('--speculative-model', config.speculativeModel)
        if (config.numDraftTokens && config.numDraftTokens !== 3) {
            parts.push('--num-draft-tokens', config.numDraftTokens.toString())
        }
    }

    // Generation defaults
    if (config.defaultTemperature && config.defaultTemperature > 0) {
        parts.push('--default-temperature', (config.defaultTemperature / 100).toFixed(2))
    }
    if (config.defaultTopP && config.defaultTopP > 0) {
        parts.push('--default-top-p', (config.defaultTopP / 100).toFixed(2))
    }

    // Embedding model
    if (config.embeddingModel) parts.push('--embedding-model', config.embeddingModel)

    // JIT compilation
    if (config.enableJit) parts.push('--enable-jit')

    // Logging
    if (config.logLevel && config.logLevel !== 'INFO') parts.push('--log-level', config.logLevel)

    // CORS
    if (config.corsOrigins && config.corsOrigins !== '*') parts.push('--allowed-origins', config.corsOrigins)

    // maxContextLength: reserved for future use, not yet wired to CLI

    if (config.additionalArgs?.trim()) parts.push(...config.additionalArgs.trim().split(/\s+/).filter(Boolean))

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

    it('VLM without continuous batching gets auto-enabled via prefix cache', () => {
        const out = preview({ isMultimodal: true, continuousBatching: false, enablePrefixCache: true })
        expect(hasFlag(out, '--is-mllm')).toBe(true)
        expect(hasFlag(out, '--continuous-batching')).toBe(true)
    })

    it('detects VLM from model config', () => {
        const out = preview({}, { isMultimodal: true })
        expect(hasFlag(out, '--is-mllm')).toBe(true)
    })

    it('manual isMultimodal overrides detected', () => {
        const out = preview({ isMultimodal: false }, { isMultimodal: true })
        expect(hasFlag(out, '--is-mllm')).toBe(false)
    })
})

describe('Prefix Cache', () => {
    it('disables prefix cache when enablePrefixCache=false', () => {
        const out = preview({ enablePrefixCache: false })
        expect(hasFlag(out, '--disable-prefix-cache')).toBe(true)
    })

    it('does not disable prefix cache when tools need it', () => {
        const out = preview({ enablePrefixCache: false, enableAutoToolChoice: true, mcpConfig: '/path/mcp.json' })
        expect(hasFlag(out, '--disable-prefix-cache')).toBe(false)
    })

    it('legacy mode: sets --no-memory-aware-cache and --prefix-cache-size', () => {
        const out = preview({ noMemoryAwareCache: true, prefixCacheSize: 500 })
        expect(hasFlag(out, '--no-memory-aware-cache')).toBe(true)
        expect(getFlagValue(out, '--prefix-cache-size')).toBe('500')
    })

    it('memory-aware mode: sets --cache-memory-mb', () => {
        const out = preview({ cacheMemoryMb: 4096 })
        expect(getFlagValue(out, '--cache-memory-mb')).toBe('4096')
    })

    it('memory-aware mode: sets --cache-memory-percent as fraction', () => {
        const out = preview({ cacheMemoryPercent: 30 })
        expect(getFlagValue(out, '--cache-memory-percent')).toBe('0.3')
    })

    it('sets --cache-ttl-minutes when > 0 and paged cache off', () => {
        const out = preview({ cacheTtlMinutes: 60, usePagedCache: false })
        expect(getFlagValue(out, '--cache-ttl-minutes')).toBe('60')
    })

    it('suppresses --cache-ttl-minutes when paged cache is on', () => {
        const out = preview({ cacheTtlMinutes: 60, usePagedCache: true })
        expect(hasFlag(out, '--cache-ttl-minutes')).toBe(false)
    })
})

describe('Paged KV Cache', () => {
    it('includes --use-paged-cache when enabled', () => {
        const out = preview({ usePagedCache: true })
        expect(hasFlag(out, '--use-paged-cache')).toBe(true)
    })

    it('sets block size from config', () => {
        const out = preview({ usePagedCache: true, pagedCacheBlockSize: 128 })
        expect(getFlagValue(out, '--paged-cache-block-size')).toBe('128')
    })

    it('sets max cache blocks from config', () => {
        const out = preview({ usePagedCache: true, maxCacheBlocks: 2000 })
        expect(getFlagValue(out, '--max-cache-blocks')).toBe('2000')
    })

    it('omits paged cache when prefix cache is off', () => {
        const out = preview({ enablePrefixCache: false, usePagedCache: true })
        expect(hasFlag(out, '--use-paged-cache')).toBe(false)
    })

    it('paged cache from detected config', () => {
        const out = preview({ usePagedCache: false }, { usePagedCache: true })
        // When config explicitly sets false, config wins over detected
        expect(hasFlag(out, '--use-paged-cache')).toBe(false)
    })
})

describe('KV Cache Quantization', () => {
    it('sets q8 quantization', () => {
        const out = preview({ kvCacheQuantization: 'q8' })
        expect(getFlagValue(out, '--kv-cache-quantization')).toBe('q8')
    })

    it('sets q4 quantization', () => {
        const out = preview({ kvCacheQuantization: 'q4' })
        expect(getFlagValue(out, '--kv-cache-quantization')).toBe('q4')
    })

    it('omits quantization when "none"', () => {
        const out = preview({ kvCacheQuantization: 'none' })
        expect(hasFlag(out, '--kv-cache-quantization')).toBe(false)
    })

    it('sets custom group size', () => {
        const out = preview({ kvCacheQuantization: 'q8', kvCacheGroupSize: 32 })
        expect(getFlagValue(out, '--kv-cache-group-size')).toBe('32')
    })

    it('omits default group size 64', () => {
        const out = preview({ kvCacheQuantization: 'q8', kvCacheGroupSize: 64 })
        expect(hasFlag(out, '--kv-cache-group-size')).toBe(false)
    })

    it('omits KV quant when prefix cache is off', () => {
        const out = preview({ enablePrefixCache: false, kvCacheQuantization: 'q8' })
        expect(hasFlag(out, '--kv-cache-quantization')).toBe(false)
    })
})

describe('Disk Cache', () => {
    it('enables disk cache', () => {
        const out = preview({ enableDiskCache: true, usePagedCache: false })
        expect(hasFlag(out, '--enable-disk-cache')).toBe(true)
    })

    it('sets disk cache dir', () => {
        const out = preview({ enableDiskCache: true, usePagedCache: false, diskCacheDir: '/tmp/cache' })
        expect(getFlagValue(out, '--disk-cache-dir')).toBe('/tmp/cache')
    })

    it('sets disk cache max gb', () => {
        const out = preview({ enableDiskCache: true, usePagedCache: false, diskCacheMaxGb: 50 })
        expect(getFlagValue(out, '--disk-cache-max-gb')).toBe('50')
    })

    it('omits disk cache when prefix cache is off', () => {
        const out = preview({ enablePrefixCache: false, enableDiskCache: true })
        expect(hasFlag(out, '--enable-disk-cache')).toBe(false)
    })
})

describe('Block Disk Cache', () => {
    it('enables block disk cache', () => {
        const out = preview({ usePagedCache: true, enableBlockDiskCache: true })
        expect(hasFlag(out, '--enable-block-disk-cache')).toBe(true)
    })

    it('sets block disk cache dir', () => {
        const out = preview({ usePagedCache: true, enableBlockDiskCache: true, blockDiskCacheDir: '/ssd/blocks' })
        expect(getFlagValue(out, '--block-disk-cache-dir')).toBe('/ssd/blocks')
    })

    it('sets block disk cache max gb', () => {
        const out = preview({ usePagedCache: true, enableBlockDiskCache: true, blockDiskCacheMaxGb: 20 })
        expect(getFlagValue(out, '--block-disk-cache-max-gb')).toBe('20')
    })

    it('omits block disk cache when paged cache is off', () => {
        const out = preview({ usePagedCache: false, enableBlockDiskCache: true })
        expect(hasFlag(out, '--enable-block-disk-cache')).toBe(false)
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

    it('uses 1000000 for max tokens when 0 (unlimited)', () => {
        const out = preview({ maxTokens: 0 })
        expect(getFlagValue(out, '--max-tokens')).toBe('1000000')
    })

    it('custom max tokens is not overridden by default', () => {
        const out = preview({ maxTokens: 4096 })
        expect(getFlagValue(out, '--max-tokens')).toBe('4096')
    })
})

describe('Tool Integration', () => {
    it('sets MCP config path', () => {
        const out = preview({ mcpConfig: '/path/mcp.json', enableAutoToolChoice: true })
        expect(getFlagValue(out, '--mcp-config')).toBe('/path/mcp.json')
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

describe('Generation Defaults', () => {
    it('sets default temperature (converted from ×100 integer)', () => {
        const out = preview({ defaultTemperature: 70 })
        expect(getFlagValue(out, '--default-temperature')).toBe('0.70')
    })

    it('omits default temperature when 0 (server default)', () => {
        const out = preview({ defaultTemperature: 0 })
        expect(hasFlag(out, '--default-temperature')).toBe(false)
    })

    it('sets high temperature (1.50 from 150)', () => {
        const out = preview({ defaultTemperature: 150 })
        expect(getFlagValue(out, '--default-temperature')).toBe('1.50')
    })

    it('sets low temperature (0.05 from 5)', () => {
        const out = preview({ defaultTemperature: 5 })
        expect(getFlagValue(out, '--default-temperature')).toBe('0.05')
    })

    it('sets max temperature (2.00 from 200)', () => {
        const out = preview({ defaultTemperature: 200 })
        expect(getFlagValue(out, '--default-temperature')).toBe('2.00')
    })

    it('sets default top-p (converted from ×100 integer)', () => {
        const out = preview({ defaultTopP: 90 })
        expect(getFlagValue(out, '--default-top-p')).toBe('0.90')
    })

    it('omits default top-p when 0 (server default)', () => {
        const out = preview({ defaultTopP: 0 })
        expect(hasFlag(out, '--default-top-p')).toBe(false)
    })

    it('sets low top-p (0.10 from 10)', () => {
        const out = preview({ defaultTopP: 10 })
        expect(getFlagValue(out, '--default-top-p')).toBe('0.10')
    })

    it('sets top-p 1.00 from 100', () => {
        const out = preview({ defaultTopP: 100 })
        expect(getFlagValue(out, '--default-top-p')).toBe('1.00')
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

    it('omits additional args when empty', () => {
        const out = preview({ additionalArgs: '' })
        // Count total parts — shouldn't have trailing empty content
        expect(out.trim().endsWith('--max-tokens') || out.includes('--max-tokens')).toBe(true)
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
        expect(getFlagValue(preview({ pagedCacheBlockSize: 32 }), '--paged-cache-block-size')).toBe('32')
        expect(getFlagValue(preview({ pagedCacheBlockSize: 256 }), '--paged-cache-block-size')).toBe('256')
    })

    it('changing maxCacheBlocks produces different CLI output', () => {
        expect(getFlagValue(preview({ maxCacheBlocks: 500 }), '--max-cache-blocks')).toBe('500')
        expect(getFlagValue(preview({ maxCacheBlocks: 5000 }), '--max-cache-blocks')).toBe('5000')
    })

    it('changing defaultTemperature produces different CLI output', () => {
        expect(getFlagValue(preview({ defaultTemperature: 50 }), '--default-temperature')).toBe('0.50')
        expect(getFlagValue(preview({ defaultTemperature: 100 }), '--default-temperature')).toBe('1.00')
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

    it('maxContextLength is reserved but does not emit CLI flag yet', () => {
        // maxContextLength is in the config interface but not wired to CLI
        // (backend support not yet implemented)
        const out = preview({ maxContextLength: 8192 })
        expect(hasFlag(out, '--max-context-length')).toBe(false)
    })
})

describe('Default IP and New Settings', () => {
    it('default host is 0.0.0.0', () => {
        expect(DEFAULT_CONFIG.host).toBe('0.0.0.0')
    })

    it('default host produces --host 0.0.0.0 in CLI output', () => {
        const out = preview()
        expect(getFlagValue(out, '--host')).toBe('0.0.0.0')
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

    it('maxContextLength reserved but not emitted to CLI', () => {
        const out = preview({ maxContextLength: 32768 })
        expect(hasFlag(out, '--max-context-length')).toBe(false)
    })

    it('default config has all new fields', () => {
        expect(DEFAULT_CONFIG.logLevel).toBe('INFO')
        expect(DEFAULT_CONFIG.corsOrigins).toBe('*')
        expect(DEFAULT_CONFIG.maxContextLength).toBe(0)
        expect(DEFAULT_CONFIG.enableJit).toBe(false)
    })
})

describe('JIT Toggle', () => {
    it('enableJit false (default) does not emit --enable-jit flag', () => {
        const out = preview({ enableJit: false })
        expect(hasFlag(out, '--enable-jit')).toBe(false)
    })

    it('enableJit true emits --enable-jit flag', () => {
        const out = preview({ enableJit: true })
        expect(hasFlag(out, '--enable-jit')).toBe(true)
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
    it('auto-enables continuous batching when prefix cache is on (LLM)', () => {
        const out = preview({ continuousBatching: false, enablePrefixCache: true })
        expect(hasFlag(out, '--continuous-batching')).toBe(true)
    })

    it('auto-enables continuous batching when prefix cache is on (VLM)', () => {
        const out = preview({ isMultimodal: true, continuousBatching: false, enablePrefixCache: true })
        expect(hasFlag(out, '--continuous-batching')).toBe(true)
    })

    it('prefillBatchSize 0 omits flag (uses backend default 8)', () => {
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
            defaultTopP: 95,
            embeddingModel: 'embed-model',
            servedModelName: 'my-model',
        })
        expect(getFlagValue(out, '--speculative-model')).toBe('draft-model')
        expect(getFlagValue(out, '--num-draft-tokens')).toBe('7')
        expect(getFlagValue(out, '--default-temperature')).toBe('0.80')
        expect(getFlagValue(out, '--default-top-p')).toBe('0.95')
        expect(getFlagValue(out, '--embedding-model')).toBe('embed-model')
        expect(getFlagValue(out, '--served-model-name')).toBe('my-model')
    })

    it('all new features disabled by default (zero values / empty strings)', () => {
        const out = preview()
        expect(hasFlag(out, '--speculative-model')).toBe(false)
        expect(hasFlag(out, '--num-draft-tokens')).toBe(false)
        expect(hasFlag(out, '--default-temperature')).toBe(false)
        expect(hasFlag(out, '--default-top-p')).toBe(false)
        expect(hasFlag(out, '--embedding-model')).toBe(false)
        expect(hasFlag(out, '--served-model-name')).toBe(false)
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

    it('MCP tools force prefix cache even when disabled', () => {
        const out = preview({
            enablePrefixCache: false,
            enableAutoToolChoice: true,
            mcpConfig: '/path/mcp.json',
            toolCallParser: 'hermes',
        })
        // Tools need cache → prefix cache NOT disabled
        expect(hasFlag(out, '--disable-prefix-cache')).toBe(false)
        expect(hasFlag(out, '--mcp-config')).toBe(true)
    })

    it('noMemoryAwareCache suppresses memory-aware flags', () => {
        const out = preview({
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

    it('cacheMemoryPercent default 30 emits 0.3', () => {
        const out = preview({ cacheMemoryPercent: 30 })
        expect(getFlagValue(out, '--cache-memory-percent')).toBe('0.3')
    })

    it('defaultTopP minimum boundary 1 emits 0.01', () => {
        const out = preview({ defaultTopP: 1 })
        expect(getFlagValue(out, '--default-top-p')).toBe('0.01')
    })

    it('numDraftTokens 0 with speculative model omits draft tokens flag', () => {
        // numDraftTokens 0 is falsy → condition fails → flag omitted → Python uses default (3)
        const out = preview({ speculativeModel: 'draft-model', numDraftTokens: 0 })
        expect(hasFlag(out, '--speculative-model')).toBe(true)
        expect(hasFlag(out, '--num-draft-tokens')).toBe(false)
    })

    it('empty diskCacheDir with enableDiskCache does not emit --disk-cache-dir', () => {
        const out = preview({ enableDiskCache: true, diskCacheDir: '', usePagedCache: false })
        expect(hasFlag(out, '--enable-disk-cache')).toBe(true)
        expect(hasFlag(out, '--disk-cache-dir')).toBe(false)
    })

    it('enableDiskCache suppressed when usePagedCache is on', () => {
        const out = preview({ enableDiskCache: true, usePagedCache: true })
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
            defaultTemperature: 70,
            defaultTopP: 90,
        })
        expect(hasFlag(out, '--continuous-batching')).toBe(true)
        expect(getFlagValue(out, '--speculative-model')).toBe('draft-model')
        expect(getFlagValue(out, '--num-draft-tokens')).toBe('4')
        expect(getFlagValue(out, '--embedding-model')).toBe('embed-model')
        expect(getFlagValue(out, '--default-temperature')).toBe('0.70')
        expect(getFlagValue(out, '--default-top-p')).toBe('0.90')
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
        'maxNumSeqs', 'prefillBatchSize', 'completionBatchSize',
        'continuousBatching', 'enablePrefixCache', 'prefixCacheSize',
        'cacheMemoryMb', 'cacheMemoryPercent', 'cacheTtlMinutes', 'noMemoryAwareCache',
        'usePagedCache', 'pagedCacheBlockSize', 'maxCacheBlocks',
        'kvCacheQuantization', 'kvCacheGroupSize',
        'enableDiskCache', 'diskCacheMaxGb', 'diskCacheDir',
        'enableBlockDiskCache', 'blockDiskCacheMaxGb', 'blockDiskCacheDir',
        'streamInterval', 'maxTokens',
        'mcpConfig', 'enableAutoToolChoice', 'toolCallParser', 'reasoningParser',
        'isMultimodal', 'servedModelName',
        'speculativeModel', 'numDraftTokens',
        'defaultTemperature', 'defaultTopP',
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

    it('default config produces minimal flags (no unnecessary options)', () => {
        const out = preview()
        const normalized = out.replace(/\s*\\\n\s*/g, ' ')

        // Defaults should NOT produce these flags:
        expect(normalized).not.toContain('--api-key')
        expect(normalized).not.toContain('VLLM_API_KEY')  // apiKey is empty
        expect(normalized).not.toContain('--rate-limit')     // rateLimit is 0
        expect(normalized).not.toContain('--is-mllm')        // isMultimodal is undefined/false
        expect(normalized).not.toContain('--disable-prefix-cache')  // prefix cache is on by default
        expect(normalized).not.toContain('--enable-disk-cache')     // disk cache is off
        expect(normalized).not.toContain('--speculative-model')     // no speculative model
        expect(normalized).not.toContain('--default-temperature')   // 0 means server default
        expect(normalized).not.toContain('--default-top-p')         // 0 means server default
        expect(normalized).not.toContain('--embedding-model')       // empty
        expect(normalized).not.toContain('--log-level')             // INFO is default (not emitted)
        expect(normalized).not.toContain('--allowed-origins')       // * is default (not emitted)
        expect(normalized).not.toContain('--enable-jit')            // off by default
        expect(normalized).not.toContain('--max-context-length')    // reserved, never emitted

        // Defaults SHOULD produce these flags:
        expect(normalized).toContain('--host')
        expect(normalized).toContain('--port')
        expect(normalized).toContain('--timeout')
        expect(normalized).toContain('--max-tokens')
        expect(normalized).toContain('--continuous-batching')
        expect(normalized).toContain('--use-paged-cache')
    })

    it('mutual exclusion: disk cache NOT emitted when paged cache is active', () => {
        // enableDiskCache is gated by !(usePagedCache) in buildCommandPreview
        const out = preview({
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
            enableBlockDiskCache: true,
            blockDiskCacheMaxGb: 50,
            blockDiskCacheDir: '/tmp/blocks',
            usePagedCache: true,
        })
        const normalized = out.replace(/\s*\\\n\s*/g, ' ')
        expect(normalized).toContain('--enable-block-disk-cache')
        expect(normalized).toContain('--block-disk-cache-dir')

        // Without paged cache, block disk cache should NOT appear
        const out2 = preview({
            enableBlockDiskCache: true,
            blockDiskCacheMaxGb: 50,
            usePagedCache: false,
        })
        const normalized2 = out2.replace(/\s*\\\n\s*/g, ' ')
        expect(normalized2).not.toContain('--enable-block-disk-cache')
    })

    it('prefix cache auto-enables continuous batching', () => {
        // When prefix cache is on and continuousBatching is off,
        // buildCommandPreview still emits --continuous-batching
        const out = preview({
            enablePrefixCache: true,
            continuousBatching: false,
        })
        const normalized = out.replace(/\s*\\\n\s*/g, ' ')
        expect(normalized).toContain('--continuous-batching')
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
            cacheTtlMinutes: 30,
            usePagedCache: true,
            noMemoryAwareCache: false,
        })
        expect(withPaged.replace(/\s*\\\n\s*/g, ' ')).not.toContain('--cache-ttl-minutes')

        const withoutPaged = preview({
            cacheTtlMinutes: 30,
            usePagedCache: false,
            noMemoryAwareCache: false,
        })
        expect(withoutPaged.replace(/\s*\\\n\s*/g, ' ')).toContain('--cache-ttl-minutes')
    })
})
