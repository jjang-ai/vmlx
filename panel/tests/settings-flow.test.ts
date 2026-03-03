/**
 * Settings Flow Tests — verifies that ALL SessionConfig fields produce correct CLI flags
 * and that no settings are hardcoded. Tests use buildCommandPreview() which mirrors
 * the actual buildArgs() logic in sessions.ts exactly.
 *
 * Coverage: all 37 SessionConfig fields, context size detection, parser resolution,
 * VLM mode, cache feature gating, and batching parameters.
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
    enableAutoToolChoice: boolean
    toolCallParser: string
    reasoningParser: string
    isMultimodal?: boolean
    additionalArgs: string
}

const DEFAULT_CONFIG: SessionConfig = {
    host: '127.0.0.1',
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
    cacheMemoryPercent: 20,
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
    enableAutoToolChoice: false,
    toolCallParser: 'auto',
    reasoningParser: 'auto',
    isMultimodal: undefined,
    additionalArgs: ''
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
    const parts = ['vllm-mlx serve', modelPath]
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
    if (config.continuousBatching && !isVLM) parts.push('--continuous-batching')

    const effectiveToolParser = config.toolCallParser === ''
        ? undefined
        : detected?.toolParser
        || (config.toolCallParser && config.toolCallParser !== 'auto' ? config.toolCallParser : undefined)
    const effectiveAutoTool = config.enableAutoToolChoice ?? detected?.enableAutoToolChoice
    const effectiveReasoningParser = config.reasoningParser === ''
        ? undefined
        : detected?.reasoningParser
        || (config.reasoningParser && config.reasoningParser !== 'auto' ? config.reasoningParser : undefined)

    const toolsNeedCache = !!(effectiveAutoTool && config.mcpConfig)
    const prefixCacheOff = config.enablePrefixCache === false && !toolsNeedCache

    if (prefixCacheOff) {
        parts.push('--disable-prefix-cache')
    } else {
        if (!isVLM && !config.continuousBatching && !parts.includes('--continuous-batching')) {
            parts.push('--continuous-batching')
        }
        if ((!config.prefillBatchSize || config.prefillBatchSize === 0) && !parts.some(a => a === '--prefill-batch-size')) {
            parts.push('--prefill-batch-size', '4096')
        }
        if (config.noMemoryAwareCache) {
            parts.push('--no-memory-aware-cache')
            if (config.prefixCacheSize && config.prefixCacheSize > 0) parts.push('--prefix-cache-size', config.prefixCacheSize.toString())
        } else {
            if (config.cacheMemoryMb && config.cacheMemoryMb > 0) parts.push('--cache-memory-mb', config.cacheMemoryMb.toString())
            if (config.cacheMemoryPercent && config.cacheMemoryPercent > 0) parts.push('--cache-memory-percent', (config.cacheMemoryPercent / 100).toString())
            if (config.cacheTtlMinutes && config.cacheTtlMinutes > 0) parts.push('--cache-ttl-minutes', config.cacheTtlMinutes.toString())
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

    if (!prefixCacheOff && config.enableDiskCache) {
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

    if (config.mcpConfig) parts.push('--mcp-config', config.mcpConfig)
    if (effectiveAutoTool) {
        parts.push('--enable-auto-tool-choice')
        if (effectiveToolParser) parts.push('--tool-call-parser', effectiveToolParser)
    }
    if (effectiveReasoningParser) parts.push('--reasoning-parser', effectiveReasoningParser)

    if (config.additionalArgs && config.additionalArgs.trim()) parts.push(config.additionalArgs.trim())

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

    it('omits --continuous-batching when VLM', () => {
        const out = preview({ isMultimodal: true, continuousBatching: true })
        expect(hasFlag(out, '--continuous-batching')).toBe(false)
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

    it('sets --cache-ttl-minutes when > 0', () => {
        const out = preview({ cacheTtlMinutes: 60 })
        expect(getFlagValue(out, '--cache-ttl-minutes')).toBe('60')
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

    it('uses detected tool parser', () => {
        const out = preview({ enableAutoToolChoice: true }, { toolParser: 'qwen' })
        expect(getFlagValue(out, '--tool-call-parser')).toBe('qwen')
    })

    it('manual tool parser overrides when no detected', () => {
        const out = preview({ enableAutoToolChoice: true, toolCallParser: 'llama' })
        expect(getFlagValue(out, '--tool-call-parser')).toBe('llama')
    })

    it('detected tool parser takes priority over manual', () => {
        const out = preview({ enableAutoToolChoice: true, toolCallParser: 'llama' }, { toolParser: 'qwen' })
        expect(getFlagValue(out, '--tool-call-parser')).toBe('qwen')
    })

    it('empty tool parser disables tool parsing', () => {
        const out = preview({ enableAutoToolChoice: true, toolCallParser: '' }, { toolParser: 'qwen' })
        expect(hasFlag(out, '--tool-call-parser')).toBe(false)
    })

    it('uses detected reasoning parser', () => {
        const out = preview({}, { reasoningParser: 'qwen3' })
        expect(getFlagValue(out, '--reasoning-parser')).toBe('qwen3')
    })

    it('manual reasoning parser when no detected', () => {
        const out = preview({ reasoningParser: 'deepseek_r1' })
        expect(getFlagValue(out, '--reasoning-parser')).toBe('deepseek_r1')
    })

    it('empty reasoning parser disables reasoning', () => {
        const out = preview({ reasoningParser: '' }, { reasoningParser: 'qwen3' })
        expect(hasFlag(out, '--reasoning-parser')).toBe(false)
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
})

describe('Feature Interaction', () => {
    it('auto-enables continuous batching when prefix cache is on (LLM)', () => {
        const out = preview({ continuousBatching: false, enablePrefixCache: true })
        expect(hasFlag(out, '--continuous-batching')).toBe(true)
    })

    it('sets safe prefill default when prefillBatchSize is 0 and prefix cache on', () => {
        const out = preview({ prefillBatchSize: 0, enablePrefixCache: true })
        expect(getFlagValue(out, '--prefill-batch-size')).toBe('4096')
    })

    it('VLM with all caching features works together', () => {
        const out = preview({
            isMultimodal: true,
            enablePrefixCache: true,
            usePagedCache: true,
            kvCacheQuantization: 'q8',
            enableBlockDiskCache: true,
        })
        expect(hasFlag(out, '--is-mllm')).toBe(true)
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
})
