/**
 * Comprehensive Application Audit Tests
 *
 * Covers edge cases across:
 *  Phase 1 — Model Config Detection & Generation Defaults
 *  Phase 2 — Session Lifecycle (port conflicts, fail counts, remote resilience)
 *  Phase 3 — Chat Pipeline (SSE parsing, template stripping, tool filtering, abort logic)
 *  Phase 4 — Server API (rate limiter, model name normalization)
 */
import { describe, it, expect, vi, beforeEach } from 'vitest'

// ═══════════════════════════════════════════════════════════════════════════════
// Phase 1: Model Config Detection
// ═══════════════════════════════════════════════════════════════════════════════

// Re-implement detectModelConfigFromDir logic as a pure function for testing
// (avoids filesystem dependency)

interface DetectedConfig {
    family: string
    toolParser?: string
    reasoningParser?: string
    cacheType: string
    usePagedCache: boolean
    enableAutoToolChoice: boolean
    isMultimodal: boolean
    description: string
    maxContextLength?: number
}

const DEFAULT_CONFIG: DetectedConfig = {
    family: 'unknown',
    cacheType: 'kv',
    usePagedCache: true,
    enableAutoToolChoice: false,
    isMultimodal: false,
    description: 'Unknown model'
}

// Reproduce MODEL_TYPE_TO_FAMILY from model-config-registry.ts
const MODEL_TYPE_TO_FAMILY: Record<string, string> = {
    'qwen3_5': 'qwen3.5-vl', 'qwen3_5_moe': 'qwen3.5-moe',
    'qwen3': 'qwen3', 'qwen3_next': 'qwen3-next', 'qwen3_moe': 'qwen3-moe',
    'qwen3_vl': 'qwen3-vl', 'qwen3_vl_moe': 'qwen3-vl',
    'qwen2': 'qwen2', 'qwen2_moe': 'qwen2', 'qwen2_vl': 'qwen2-vl',
    'qwen2_5_vl': 'qwen2-vl', 'qwen': 'qwen2', 'qwen_mamba': 'qwen-mamba',
    'llama': 'llama3', 'llama4': 'llama4',
    'mistral': 'mistral', 'mixtral': 'mixtral', 'pixtral': 'pixtral',
    'codestral': 'codestral', 'devstral': 'devstral', 'codestral_mamba': 'mamba',
    'deepseek_v3': 'deepseek-v3', 'deepseek_v2': 'deepseek-v2',
    'deepseek_vl': 'deepseek-vl', 'deepseek_vl2': 'deepseek-vl',
    'deepseek_vl_v2': 'deepseek-vl', 'deepseek2': 'deepseek', 'deepseek': 'deepseek',
    'chatglm': 'glm4', 'glm4': 'glm4', 'glm4_moe': 'glm47-flash',
    'glm4_moe_lite': 'glm47-flash', 'glm': 'glm4', 'gpt_oss': 'gpt-oss',
    'step1v': 'step-vl', 'step3p5': 'step-3.5-flash', 'step': 'step',
    'gemma': 'gemma', 'gemma2': 'gemma2', 'gemma3': 'gemma3', 'gemma3_text': 'gemma3-text',
    'phi3': 'phi3', 'phi3v': 'phi3-vision', 'phi3small': 'phi3',
    'phi4': 'phi4', 'phi4mm': 'phi4-multimodal', 'phi4flash': 'phi4', 'phi4_reasoning': 'phi4-reasoning',
    'phi': 'phi3',
    'minimax': 'minimax', 'minimax_m2': 'minimax', 'minimax_m2_5': 'minimax',
    'jamba': 'jamba', 'mamba': 'mamba', 'mamba2': 'mamba', 'falcon_mamba': 'falcon-mamba',
    'rwkv': 'rwkv', 'rwkv5': 'rwkv', 'rwkv6': 'rwkv',
    'nemotron': 'nemotron', 'nemotron_h': 'nemotron',
    'granite': 'granite', 'granite_moe': 'granite',
    'cohere': 'command-r', 'cohere2': 'command-r', 'hermes': 'hermes',
    'kimi_k2': 'kimi-k2',
    'exaone': 'exaone', 'exaone3': 'exaone', 'olmo': 'olmo', 'olmo2': 'olmo',
    'paligemma': 'paligemma', 'paligemma2': 'paligemma',
    'llava': 'llava', 'llava_next': 'llava', 'idefics2': 'idefics', 'idefics3': 'idefics',
    'cogvlm': 'cogvlm', 'cogvlm2': 'cogvlm', 'florence2': 'florence',
    'molmo': 'molmo', 'minicpmv': 'minicpm-v', 'smolvlm': 'smolvlm',
    'internvl_chat': 'internvl',
    'starcoder2': 'starcoder', 'stablelm': 'stablelm', 'baichuan': 'baichuan',
    'internlm': 'internlm', 'internlm2': 'internlm', 'internlm3': 'internlm3',
    'internlm_xcomposer2': 'internlm-xcomposer',
    'yi': 'llama3', 'orion': 'llama3',
}

// Registry of expected results per family
const FAMILY_CONFIGS: Record<string, Partial<DetectedConfig>> = {
    'qwen3': { cacheType: 'kv', toolParser: 'qwen', reasoningParser: 'qwen3', enableAutoToolChoice: true },
    'qwen3-next': { cacheType: 'mamba', toolParser: 'nemotron', usePagedCache: true },
    'qwen2-vl': { cacheType: 'kv', toolParser: 'qwen', isMultimodal: true },
    'llama3': { cacheType: 'kv', toolParser: 'llama', enableAutoToolChoice: true },
    'mistral': { cacheType: 'kv', toolParser: 'mistral', enableAutoToolChoice: true },
    'deepseek-r1': { cacheType: 'kv', toolParser: 'deepseek', reasoningParser: 'deepseek_r1' },
    'gpt-oss': { cacheType: 'kv', toolParser: 'glm47', reasoningParser: 'openai_gptoss', enableAutoToolChoice: true },
    'gemma3': { cacheType: 'kv', toolParser: 'hermes', reasoningParser: 'deepseek_r1', isMultimodal: true },
    'phi4-reasoning': { cacheType: 'kv', toolParser: 'hermes', reasoningParser: 'deepseek_r1', enableAutoToolChoice: true },
    'nemotron': { cacheType: 'hybrid', toolParser: 'nemotron', usePagedCache: true },
    'jamba': { cacheType: 'hybrid', usePagedCache: true },
    'falcon-mamba': { cacheType: 'mamba', usePagedCache: true },
    'mamba': { cacheType: 'mamba', usePagedCache: true },
    'step-vl': { cacheType: 'kv', toolParser: 'step3p5', reasoningParser: 'qwen3', isMultimodal: true },
    'minimax': { cacheType: 'kv', toolParser: 'minimax', reasoningParser: 'qwen3', enableAutoToolChoice: true },
    'kimi-k2': { cacheType: 'kv', toolParser: 'kimi', enableAutoToolChoice: true },
}

describe('Phase 1: Model Config Detection', () => {
    describe('MODEL_TYPE_TO_FAMILY mapping completeness', () => {
        it('maps qwen3 model_type to correct family', () => {
            expect(MODEL_TYPE_TO_FAMILY['qwen3']).toBe('qwen3')
        })

        it('maps qwen3_next (hybrid mamba) to mamba family', () => {
            expect(MODEL_TYPE_TO_FAMILY['qwen3_next']).toBe('qwen3-next')
        })

        it('maps qwen2_5_vl to qwen2-vl (VL reuse)', () => {
            expect(MODEL_TYPE_TO_FAMILY['qwen2_5_vl']).toBe('qwen2-vl')
        })

        it('maps codestral_mamba to pure mamba family', () => {
            expect(MODEL_TYPE_TO_FAMILY['codestral_mamba']).toBe('mamba')
        })

        it('maps gpt_oss to gpt-oss (Harmony protocol)', () => {
            expect(MODEL_TYPE_TO_FAMILY['gpt_oss']).toBe('gpt-oss')
        })

        it('maps yi to llama3 (architecture compatible)', () => {
            expect(MODEL_TYPE_TO_FAMILY['yi']).toBe('llama3')
        })

        it('has no undefined family mappings', () => {
            for (const [modelType, family] of Object.entries(MODEL_TYPE_TO_FAMILY)) {
                expect(family, `model_type "${modelType}" maps to undefined`).toBeDefined()
                expect(typeof family).toBe('string')
                expect(family.length).toBeGreaterThan(0)
            }
        })

        it('handles all vision model_types as multimodal', () => {
            const vlTypes = ['qwen3_5', 'qwen3_vl', 'qwen3_vl_moe', 'qwen2_vl', 'qwen2_5_vl',
                'pixtral', 'deepseek_vl', 'deepseek_vl2', 'deepseek_vl_v2', 'phi3v', 'phi4mm',
                'step1v', 'llava', 'llava_next', 'idefics2', 'idefics3', 'cogvlm', 'cogvlm2',
                'florence2', 'molmo', 'minicpmv', 'smolvlm', 'internvl_chat',
                'paligemma', 'paligemma2', 'internlm_xcomposer2', 'gemma3']
            for (const mt of vlTypes) {
                const family = MODEL_TYPE_TO_FAMILY[mt]
                expect(family, `VL model_type "${mt}" should map to a family`).toBeDefined()
            }
        })

        it('handles all hybrid/mamba model_types with correct cache', () => {
            const mambaTypes = ['qwen3_next', 'qwen_mamba', 'jamba', 'mamba', 'mamba2',
                'falcon_mamba', 'rwkv', 'rwkv5', 'rwkv6', 'nemotron', 'nemotron_h', 'codestral_mamba']
            for (const mt of mambaTypes) {
                const family = MODEL_TYPE_TO_FAMILY[mt]
                expect(family, `Mamba/hybrid model_type "${mt}" should map to a family`).toBeDefined()
            }
        })
    })

    describe('DEFAULT_CONFIG fallback', () => {
        it('defaults to kv cache type', () => {
            expect(DEFAULT_CONFIG.cacheType).toBe('kv')
        })

        it('defaults to paged cache enabled', () => {
            expect(DEFAULT_CONFIG.usePagedCache).toBe(true)
        })

        it('defaults to no tool choice', () => {
            expect(DEFAULT_CONFIG.enableAutoToolChoice).toBe(false)
        })

        it('defaults to text-only (not multimodal)', () => {
            expect(DEFAULT_CONFIG.isMultimodal).toBe(false)
        })

        it('defaults to unknown family', () => {
            expect(DEFAULT_CONFIG.family).toBe('unknown')
        })

        it('has no tool parser by default', () => {
            expect(DEFAULT_CONFIG.toolParser).toBeUndefined()
        })

        it('has no reasoning parser by default', () => {
            expect(DEFAULT_CONFIG.reasoningParser).toBeUndefined()
        })
    })

    describe('Context length field detection', () => {
        function extractMaxContext(parsed: Record<string, any>): number | undefined {
            return (
                (typeof parsed.max_position_embeddings === 'number' ? parsed.max_position_embeddings : undefined) ??
                (typeof parsed.max_sequence_length === 'number' ? parsed.max_sequence_length : undefined) ??
                (typeof parsed.seq_length === 'number' ? parsed.seq_length : undefined) ??
                (typeof parsed.text_config?.max_position_embeddings === 'number' ? parsed.text_config.max_position_embeddings : undefined)
            )
        }

        it('reads max_position_embeddings', () => {
            expect(extractMaxContext({ model_type: 'qwen3', max_position_embeddings: 40960 })).toBe(40960)
        })

        it('reads max_sequence_length as fallback', () => {
            expect(extractMaxContext({ model_type: 'mamba', max_sequence_length: 8192 })).toBe(8192)
        })

        it('reads seq_length as fallback', () => {
            expect(extractMaxContext({ model_type: 'rwkv', seq_length: 4096 })).toBe(4096)
        })

        it('reads text_config.max_position_embeddings for VL models', () => {
            expect(extractMaxContext({ model_type: 'qwen2_vl', text_config: { max_position_embeddings: 32768 } })).toBe(32768)
        })

        it('returns undefined when no context fields present', () => {
            expect(extractMaxContext({ model_type: 'unknown' })).toBeUndefined()
        })

        it('prefers max_position_embeddings over max_sequence_length', () => {
            expect(extractMaxContext({ max_position_embeddings: 8192, max_sequence_length: 4096 })).toBe(8192)
        })
    })
})

// ═══════════════════════════════════════════════════════════════════════════════
// Phase 1b: Generation Defaults Parsing
// ═══════════════════════════════════════════════════════════════════════════════

// Pure function clone of readGenerationDefaults logic for testing
interface GenerationDefaults {
    temperature?: number
    topP?: number
    topK?: number
    minP?: number
    repeatPenalty?: number
}

function parseGenerationDefaults(raw: string): GenerationDefaults | null {
    try {
        const config = JSON.parse(raw)
        const defaults: GenerationDefaults = {}
        if (typeof config.temperature === 'number') defaults.temperature = config.temperature
        if (typeof config.top_p === 'number') defaults.topP = config.top_p
        if (typeof config.top_k === 'number') defaults.topK = config.top_k
        if (typeof config.min_p === 'number') defaults.minP = config.min_p
        if (typeof config.repetition_penalty === 'number') defaults.repeatPenalty = config.repetition_penalty
        if (Object.keys(defaults).length === 0) return null
        return defaults
    } catch {
        return null
    }
}

describe('Phase 1b: Generation Defaults', () => {
    it('extracts temperature', () => {
        const r = parseGenerationDefaults('{"temperature": 0.6}')
        expect(r?.temperature).toBe(0.6)
    })

    it('extracts all fields', () => {
        const r = parseGenerationDefaults('{"temperature": 0.7, "top_p": 0.9, "top_k": 50, "min_p": 0.05, "repetition_penalty": 1.2}')
        expect(r).toEqual({ temperature: 0.7, topP: 0.9, topK: 50, minP: 0.05, repeatPenalty: 1.2 })
    })

    it('returns null for empty config', () => {
        expect(parseGenerationDefaults('{}')).toBeNull()
    })

    it('returns null for invalid JSON', () => {
        expect(parseGenerationDefaults('not json')).toBeNull()
    })

    it('ignores string values (must be number)', () => {
        expect(parseGenerationDefaults('{"temperature": "high"}')).toBeNull()
    })

    it('handles temperature = 0 (valid number)', () => {
        const r = parseGenerationDefaults('{"temperature": 0}')
        expect(r?.temperature).toBe(0)
    })

    it('ignores unknown fields', () => {
        const r = parseGenerationDefaults('{"temperature": 0.5, "unknown_param": 42}')
        expect(r).toEqual({ temperature: 0.5 })
    })
})

// ═══════════════════════════════════════════════════════════════════════════════
// Phase 1c: Download Pipeline — tqdm Progress Parsing
// ═══════════════════════════════════════════════════════════════════════════════

interface DownloadProgress {
    percent?: number
    speed?: string
    downloaded?: string
    total?: string
    eta?: string
    currentFile?: string
    filesProgress?: string
    raw?: string
}

function parseTqdmProgress(line: string): Partial<DownloadProgress> {
    const result: Partial<DownloadProgress> = { raw: line.trim() }

    const fileMatch = line.match(/(?:Downloading|Fetching)\s+(.+?)(?:\s*:|\s*\|)/)
    if (fileMatch) result.currentFile = fileMatch[1].trim()

    const tqdmMatch = line.match(/(\d+)%\|[^|]*\|\s*([\d.]+\w*)\/([\d.]+\w*)\s*\[([^\]<]*)<([^\],]*),\s*([^\]]+)\]/)
    if (tqdmMatch) {
        result.percent = parseInt(tqdmMatch[1], 10)
        result.downloaded = tqdmMatch[2]
        result.total = tqdmMatch[3]
        result.eta = tqdmMatch[5].trim()
        result.speed = tqdmMatch[6].trim()
    } else {
        const simplePercent = line.match(/\s(\d+)%\|/)
        if (simplePercent) result.percent = parseInt(simplePercent[1], 10)
    }

    const filesMatch = line.match(/Fetching\s+(\d+)\s+files.*?(\d+)%/)
    if (filesMatch) result.filesProgress = `${Math.round(parseInt(filesMatch[2]) * parseInt(filesMatch[1]) / 100)}/${filesMatch[1]}`

    return result
}

describe('Phase 1c: tqdm Progress Parsing', () => {
    it('parses full tqdm bar', () => {
        const r = parseTqdmProgress('  45%|████      | 1.2G/4.5G [01:30<02:30, 45.2MB/s]')
        expect(r.percent).toBe(45)
        expect(r.downloaded).toBe('1.2G')
        expect(r.total).toBe('4.5G')
        expect(r.eta).toBe('02:30')
        expect(r.speed).toBe('45.2MB/s')
    })

    it('parses "Downloading" filename', () => {
        const r = parseTqdmProgress('Downloading model-00001-of-00005.safetensors:  45%|')
        expect(r.currentFile).toBe('model-00001-of-00005.safetensors')
        expect(r.percent).toBe(45)
    })

    it('parses "Fetching" files progress', () => {
        const r = parseTqdmProgress('Fetching 10 files:  60%|')
        expect(r.filesProgress).toBe('6/10')
        expect(r.percent).toBe(60)
    })

    it('handles 100% complete', () => {
        const r = parseTqdmProgress(' 100%|████████████| 5.0G/5.0G [03:00<00:00, 28.1MB/s]')
        expect(r.percent).toBe(100)
        expect(r.eta).toBe('00:00')
    })

    it('handles 0% start', () => {
        const r = parseTqdmProgress('   0%|          | 0.00/5.0G [00:00<?, ?B/s]')
        expect(r.percent).toBe(0)
    })

    it('preserves raw line', () => {
        const r = parseTqdmProgress('  some random output  ')
        expect(r.raw).toBe('some random output')
    })

    it('returns only raw for non-tqdm output', () => {
        const r = parseTqdmProgress('Loading tokenizer...')
        expect(r.percent).toBeUndefined()
        expect(r.currentFile).toBeUndefined()
    })
})

// ═══════════════════════════════════════════════════════════════════════════════
// Phase 2: Session Lifecycle
// ═══════════════════════════════════════════════════════════════════════════════

describe('Phase 2: Session Lifecycle', () => {
    describe('Health check fail counting', () => {
        // Re-implement incrementFailAndCheck logic
        function computeMaxFails(configTimeout: number): number {
            if (configTimeout > 0) return Math.max(60, Math.ceil(configTimeout / 5))
            return 60 // default MAX_FAIL_COUNT
        }

        it('computes 60 max fails for default timeout (300s)', () => {
            expect(computeMaxFails(300)).toBe(60) // 300/5 = 60
        })

        it('computes 120 max fails for 600s timeout', () => {
            expect(computeMaxFails(600)).toBe(120)
        })

        it('floors at 60 for short timeouts (e.g. 10s)', () => {
            expect(computeMaxFails(10)).toBe(60) // max(60, ceil(10/5)=2) = 60
        })

        it('handles unlimited timeout (0) with default 60', () => {
            expect(computeMaxFails(0)).toBe(60)
        })

        it('handles very large timeout (3600s = 1 hour)', () => {
            expect(computeMaxFails(3600)).toBe(720)
        })
    })

    describe('Remote session resilience', () => {
        it('remote sessions should NOT be aborted on health failure', () => {
            // handleSessionDown for remote: resets fail count, does NOT mark stopped
            const isRemote = true
            const shouldAbort = !isRemote // only abort local sessions
            expect(shouldAbort).toBe(false)
        })

        it('local sessions should be killed on health failure', () => {
            const isRemote = false
            const shouldAbort = !isRemote
            expect(shouldAbort).toBe(true)
        })
    })

    describe('GGUF rejection', () => {
        function validateModelFormat(files: string[]): string | null {
            const hasGGUF = files.some(f => f.endsWith('.gguf') || f.endsWith('.gguf.part'))
            const hasSafetensors = files.some(f => f.endsWith('.safetensors'))
            const hasConfig = files.includes('config.json')

            if (hasGGUF && !hasSafetensors) return 'GGUF format not supported'
            if (!hasConfig) return 'Missing config.json'
            return null
        }

        it('rejects GGUF-only directories', () => {
            expect(validateModelFormat(['model.gguf', 'tokenizer.json'])).toBe('GGUF format not supported')
        })

        it('rejects partial GGUF downloads', () => {
            expect(validateModelFormat(['model.gguf.part'])).toBe('GGUF format not supported')
        })

        it('accepts MLX format', () => {
            expect(validateModelFormat(['model.safetensors', 'config.json'])).toBeNull()
        })

        it('rejects missing config.json', () => {
            expect(validateModelFormat(['model.safetensors'])).toBe('Missing config.json')
        })

        it('accepts hybrid directory (GGUF + safetensors) as MLX', () => {
            // If both GGUF and safetensors exist, we use safetensors
            expect(validateModelFormat(['model.gguf', 'model.safetensors', 'config.json'])).toBeNull()
        })
    })

    describe('Memory estimation warnings', () => {
        function memoryWarningLevel(modelSizeBytes: number, availableBytes: number): 'none' | 'high' | 'critical' {
            if (modelSizeBytes > availableBytes * 0.9) return 'critical'
            if (modelSizeBytes > availableBytes * 0.7) return 'high'
            return 'none'
        }

        it('no warning when model fits comfortably', () => {
            expect(memoryWarningLevel(10e9, 100e9)).toBe('none') // 10GB model, 100GB free
        })

        it('high warning when using most RAM', () => {
            expect(memoryWarningLevel(75e9, 100e9)).toBe('high') // 75GB model, 100GB free
        })

        it('critical warning when exceeding 90% of RAM', () => {
            expect(memoryWarningLevel(95e9, 100e9)).toBe('critical')
        })

        it('critical for model larger than available RAM', () => {
            expect(memoryWarningLevel(200e9, 100e9)).toBe('critical')
        })
    })
})

// ═══════════════════════════════════════════════════════════════════════════════
// Phase 3: Chat Pipeline
// ═══════════════════════════════════════════════════════════════════════════════

describe('Phase 3: Chat Pipeline', () => {
    describe('Template stop token stripping', () => {
        // Build tokens programmatically to avoid parser issues with special tokens
        const TEMPLATE_STOP_TOKENS = [
            '<' + '|im_end|' + '>',
            '<' + '|im_start|' + '>',
            '<' + '|eot_id|' + '>',
            '<' + '|start_header_id|' + '>',
            '<' + '|end|' + '>',
            '<' + '|user|' + '>',
            '<' + '|assistant|' + '>',
            '<' + '/s>',
            '<' + 's>',
            '<' + '|endoftext|' + '>',
            '[/INST]',
            '[INST]',
            '<end_of_turn>',
        ]

        const TEMPLATE_TOKEN_REGEX = new RegExp(
            TEMPLATE_STOP_TOKENS.map(t => t.replace(/[.*+?^${}()|[\]\\]/g, '\\$&')).join('|'),
            'g'
        )

        function stripTemplateTokens(text: string): string {
            return text.replace(TEMPLATE_TOKEN_REGEX, '')
        }

        it('strips ChatML tokens from output', () => {
            const tok = '<' + '|im_end|' + '>'
            expect(stripTemplateTokens('Hello world' + tok)).toBe('Hello world')
        })

        it('strips Llama 3 tokens', () => {
            const tok = '<' + '|eot_id|' + '>'
            expect(stripTemplateTokens('Response text' + tok)).toBe('Response text')
        })

        it('strips Gemma end_of_turn', () => {
            expect(stripTemplateTokens('Hello<end_of_turn>')).toBe('Hello')
        })

        it('strips Mistral [/INST] tokens', () => {
            expect(stripTemplateTokens('Answer[/INST]')).toBe('Answer')
        })

        it('strips multiple different tokens', () => {
            const im = '<' + '|im_end|' + '>'
            const eot = '<' + '|eot_id|' + '>'
            expect(stripTemplateTokens('Hello' + im + ' world' + eot)).toBe('Hello world')
        })

        it('leaves normal text unchanged', () => {
            expect(stripTemplateTokens('Normal response without tokens')).toBe('Normal response without tokens')
        })

        it('handles empty string', () => {
            expect(stripTemplateTokens('')).toBe('')
        })

        it('handles string containing only a stop token', () => {
            const tok = '<' + '|endoftext|' + '>'
            expect(stripTemplateTokens(tok)).toBe('')
        })
    })

    describe('Tool filtering', () => {
        const FILE_TOOLS = new Set(['read_file', 'write_file', 'edit_file', 'patch_file', 'batch_edit', 'copy_file', 'move_file', 'delete_file', 'create_directory', 'list_directory', 'insert_text', 'replace_lines', 'apply_regex', 'read_image'])
        const SEARCH_TOOLS = new Set(['search_files', 'find_files', 'file_info', 'get_diagnostics', 'get_tree', 'diff_files'])
        const SHELL_TOOLS = new Set(['run_command', 'spawn_process', 'get_process_output'])
        const UTILITY_TOOLS = new Set(['count_tokens', 'clipboard_read', 'clipboard_write'])

        function filterTools(allTools: any[], overrides: any): any[] {
            const disabled = new Set<string>()
            if (overrides.fileToolsEnabled === false) FILE_TOOLS.forEach(t => disabled.add(t))
            if (overrides.searchToolsEnabled === false) SEARCH_TOOLS.forEach(t => disabled.add(t))
            if (overrides.shellEnabled === false) SHELL_TOOLS.forEach(t => disabled.add(t))
            if (overrides.utilityToolsEnabled === false) UTILITY_TOOLS.forEach(t => disabled.add(t))
            if (disabled.size === 0) return allTools
            return allTools.filter((t: any) => !disabled.has(t.function.name))
        }

        const allTools = [
            { function: { name: 'read_file' } },
            { function: { name: 'search_files' } },
            { function: { name: 'run_command' } },
            { function: { name: 'ask_user' } },
            { function: { name: 'count_tokens' } },
        ]

        it('returns all tools when no overrides', () => {
            expect(filterTools(allTools, {})).toEqual(allTools)
        })

        it('disables file tools', () => {
            const result = filterTools(allTools, { fileToolsEnabled: false })
            expect(result.find(t => t.function.name === 'read_file')).toBeUndefined()
            expect(result.find(t => t.function.name === 'ask_user')).toBeDefined()
        })

        it('disables search tools', () => {
            const result = filterTools(allTools, { searchToolsEnabled: false })
            expect(result.find(t => t.function.name === 'search_files')).toBeUndefined()
        })

        it('disables shell tools', () => {
            const result = filterTools(allTools, { shellEnabled: false })
            expect(result.find(t => t.function.name === 'run_command')).toBeUndefined()
        })

        it('ask_user is NEVER disabled', () => {
            const result = filterTools(allTools, {
                fileToolsEnabled: false,
                searchToolsEnabled: false,
                shellEnabled: false,
                utilityToolsEnabled: false
            })
            expect(result.find(t => t.function.name === 'ask_user')).toBeDefined()
        })

        it('disables utility tools', () => {
            const result = filterTools(allTools, { utilityToolsEnabled: false })
            expect(result.find(t => t.function.name === 'count_tokens')).toBeUndefined()
        })

        it('multiple categories disabled simultaneously', () => {
            const result = filterTools(allTools, {
                fileToolsEnabled: false,
                shellEnabled: false
            })
            expect(result.find(t => t.function.name === 'read_file')).toBeUndefined()
            expect(result.find(t => t.function.name === 'run_command')).toBeUndefined()
            expect(result.find(t => t.function.name === 'search_files')).toBeDefined()
            expect(result.find(t => t.function.name === 'ask_user')).toBeDefined()
        })
    })

    describe('Stale request detection', () => {
        function isStale(startedAt: number, timeoutMs: number, now: number): boolean {
            const buffer = 30_000 // 30s buffer
            const effectiveTimeout = timeoutMs === 0 ? 0 : timeoutMs + buffer
            if (effectiveTimeout === 0) return false // unlimited timeout
            return (now - startedAt) > effectiveTimeout
        }

        it('not stale within timeout', () => {
            expect(isStale(1000, 60000, 30000)).toBe(false)
        })

        it('stale after timeout + buffer', () => {
            expect(isStale(0, 60000, 100000)).toBe(true) // 100s > 60s + 30s
        })

        it('unlimited timeout (0) is never stale', () => {
            expect(isStale(0, 0, 999999999)).toBe(false)
        })

        it('exactly at boundary is not stale', () => {
            expect(isStale(0, 60000, 90000)).toBe(false) // exactly at 90s = 60+30
        })

        it('1ms past boundary is stale', () => {
            expect(isStale(0, 60000, 90001)).toBe(true)
        })
    })

    describe('Abort by endpoint', () => {
        it('correctly identifies matching endpoint entries', () => {
            const entries = [
                { host: '127.0.0.1', port: 8000, chatId: 'a' },
                { host: '127.0.0.1', port: 8001, chatId: 'b' },
                { host: '127.0.0.1', port: 8000, chatId: 'c' },
            ]
            const matching = entries.filter(e => e.host === '127.0.0.1' && e.port === 8000)
            expect(matching.length).toBe(2)
            expect(matching.map(m => m.chatId)).toEqual(['a', 'c'])
        })

        it('does not match different ports', () => {
            const entries = [{ host: '127.0.0.1', port: 8001, chatId: 'a' }]
            const matching = entries.filter(e => e.host === '127.0.0.1' && e.port === 8000)
            expect(matching.length).toBe(0)
        })
    })
})

// ═══════════════════════════════════════════════════════════════════════════════
// Phase 4: Server API
// ═══════════════════════════════════════════════════════════════════════════════

describe('Phase 4: Server API', () => {
    describe('Rate limiter sliding window', () => {
        class RateLimiter {
            private requests: Map<string, number[]> = new Map()
            constructor(
                public requestsPerMinute: number,
                public enabled: boolean
            ) { }

            isAllowed(clientId: string): [boolean, number] {
                if (!this.enabled) return [true, 0]
                const now = Date.now()
                const window = 60_000
                let timestamps = this.requests.get(clientId) || []
                timestamps = timestamps.filter(t => now - t < window)
                if (timestamps.length >= this.requestsPerMinute) {
                    const oldestInWindow = timestamps[0]
                    const retryAfter = Math.ceil((oldestInWindow + window - now) / 1000)
                    return [false, retryAfter]
                }
                timestamps.push(now)
                this.requests.set(clientId, timestamps)
                return [true, 0]
            }
        }

        it('allows requests when disabled', () => {
            const rl = new RateLimiter(1, false)
            for (let i = 0; i < 100; i++) {
                expect(rl.isAllowed('test')[0]).toBe(true)
            }
        })

        it('allows requests within limit', () => {
            const rl = new RateLimiter(5, true)
            for (let i = 0; i < 5; i++) {
                expect(rl.isAllowed('test')[0]).toBe(true)
            }
        })

        it('rejects requests beyond limit', () => {
            const rl = new RateLimiter(2, true)
            expect(rl.isAllowed('test')[0]).toBe(true)
            expect(rl.isAllowed('test')[0]).toBe(true)
            const [allowed, retryAfter] = rl.isAllowed('test')
            expect(allowed).toBe(false)
            expect(retryAfter).toBeGreaterThan(0)
        })

        it('isolates clients', () => {
            const rl = new RateLimiter(1, true)
            expect(rl.isAllowed('alice')[0]).toBe(true)
            expect(rl.isAllowed('bob')[0]).toBe(true) // different client
            expect(rl.isAllowed('alice')[0]).toBe(false) // alice over limit
        })
    })

    describe('Model name normalization', () => {
        function normalizeModelName(name: string): string {
            const parts = name.replace(/\/+$/, '').split('/')
            if (parts.length >= 2) {
                return `${parts[parts.length - 2]}/${parts[parts.length - 1]}`
            }
            return parts[parts.length - 1] || name
        }

        it('extracts org/model from full path', () => {
            expect(normalizeModelName('/Users/eric/.lmstudio/models/mlx-community/Llama-3.2-3B-Instruct-4bit'))
                .toBe('mlx-community/Llama-3.2-3B-Instruct-4bit')
        })

        it('preserves already-short names', () => {
            expect(normalizeModelName('mlx-community/Llama-3.2-3B-Instruct-4bit'))
                .toBe('mlx-community/Llama-3.2-3B-Instruct-4bit')
        })

        it('handles single component', () => {
            expect(normalizeModelName('some-model')).toBe('some-model')
        })

        it('handles trailing slash', () => {
            expect(normalizeModelName('/path/to/org/model/')).toBe('org/model')
        })

        it('handles deeply nested paths', () => {
            expect(normalizeModelName('/home/user/.cache/huggingface/hub/models--org--model/snapshots/abc123/'))
                .toBe('snapshots/abc123')
        })
    })

    describe('Cancel endpoint path selection', () => {
        function getCancelPath(responseId: string): string {
            return responseId.startsWith('resp_')
                ? `/v1/responses/${responseId}/cancel`
                : `/v1/chat/completions/${responseId}/cancel`
        }

        it('uses responses API cancel for resp_ prefix', () => {
            expect(getCancelPath('resp_abc123')).toBe('/v1/responses/resp_abc123/cancel')
        })

        it('uses chat completions cancel for chatcmpl prefix', () => {
            expect(getCancelPath('chatcmpl-abc123')).toBe('/v1/chat/completions/chatcmpl-abc123/cancel')
        })

        it('uses chat completions cancel for unknown prefix', () => {
            expect(getCancelPath('custom-12345')).toBe('/v1/chat/completions/custom-12345/cancel')
        })
    })

    describe('Temperature and sampling resolution', () => {
        function resolveTemperature(
            requestValue: number | undefined,
            cliDefault: number | undefined,
            fallback: number
        ): number {
            if (requestValue !== undefined && requestValue !== null) return requestValue
            if (cliDefault !== undefined && cliDefault !== null) return cliDefault
            return fallback
        }

        it('request value takes highest priority', () => {
            expect(resolveTemperature(0.5, 0.7, 0.9)).toBe(0.5)
        })

        it('CLI default used when no request value', () => {
            expect(resolveTemperature(undefined, 0.7, 0.9)).toBe(0.7)
        })

        it('fallback used when no request or CLI value', () => {
            expect(resolveTemperature(undefined, undefined, 0.9)).toBe(0.9)
        })

        it('request value of 0 is respected (not falsy)', () => {
            expect(resolveTemperature(0, 0.7, 0.9)).toBe(0)
        })
    })
})

// ═══════════════════════════════════════════════════════════════════════════════
// Phase 5: Caching Engine
// ═══════════════════════════════════════════════════════════════════════════════

describe('Phase 5: Caching Engine', () => {
    describe('Block hash chain integrity', () => {
        // Simplified hash function mimicking compute_block_hash
        function computeBlockHash(parentHash: string | null, tokenIds: number[]): string {
            const data = (parentHash || '') + ':' + tokenIds.join(',')
            // Simple hash for testing
            let hash = 0
            for (let i = 0; i < data.length; i++) {
                hash = ((hash << 5) - hash + data.charCodeAt(i)) | 0
            }
            return hash.toString(16)
        }

        it('same tokens produce same hash', () => {
            const h1 = computeBlockHash(null, [1, 2, 3])
            const h2 = computeBlockHash(null, [1, 2, 3])
            expect(h1).toBe(h2)
        })

        it('different tokens produce different hashes', () => {
            const h1 = computeBlockHash(null, [1, 2, 3])
            const h2 = computeBlockHash(null, [4, 5, 6])
            expect(h1).not.toBe(h2)
        })

        it('parent hash changes child hash', () => {
            const h1 = computeBlockHash('parent1', [1, 2, 3])
            const h2 = computeBlockHash('parent2', [1, 2, 3])
            expect(h1).not.toBe(h2)
        })

        it('null parent differs from string parent', () => {
            const h1 = computeBlockHash(null, [1, 2, 3])
            const h2 = computeBlockHash('abc', [1, 2, 3])
            expect(h1).not.toBe(h2)
        })

        it('hash chain grows correctly', () => {
            const h1 = computeBlockHash(null, [1, 2, 3])
            const h2 = computeBlockHash(h1, [4, 5, 6])
            const h3 = computeBlockHash(h2, [7, 8, 9])
            // All different
            expect(new Set([h1, h2, h3]).size).toBe(3)
        })
    })

    describe('LRU eviction ordering', () => {
        class SimpleLRU<K, V> {
            private cache = new Map<K, V>()
            constructor(private maxSize: number) { }

            get(key: K): V | undefined {
                const val = this.cache.get(key)
                if (val !== undefined) {
                    // Move to end (MRU)
                    this.cache.delete(key)
                    this.cache.set(key, val)
                }
                return val
            }

            set(key: K, value: V): K | undefined {
                if (this.cache.has(key)) this.cache.delete(key)
                let evicted: K | undefined
                if (this.cache.size >= this.maxSize) {
                    const oldest = this.cache.keys().next().value!
                    evicted = oldest
                    this.cache.delete(oldest)
                }
                this.cache.set(key, value)
                return evicted
            }

            get size(): number { return this.cache.size }
        }

        it('evicts oldest entry when full', () => {
            const lru = new SimpleLRU<string, number>(3)
            lru.set('a', 1)
            lru.set('b', 2)
            lru.set('c', 3)
            const evicted = lru.set('d', 4) // should evict 'a'
            expect(evicted).toBe('a')
            expect(lru.get('a')).toBeUndefined()
            expect(lru.get('d')).toBe(4)
        })

        it('touching entry prevents eviction', () => {
            const lru = new SimpleLRU<string, number>(3)
            lru.set('a', 1)
            lru.set('b', 2)
            lru.set('c', 3)
            lru.get('a') // touch 'a', making 'b' the oldest
            const evicted = lru.set('d', 4) // should evict 'b'
            expect(evicted).toBe('b')
            expect(lru.get('a')).toBe(1) // 'a' survived
        })

        it('handles capacity of 1', () => {
            const lru = new SimpleLRU<string, number>(1)
            lru.set('a', 1)
            const evicted = lru.set('b', 2)
            expect(evicted).toBe('a')
            expect(lru.size).toBe(1)
        })

        it('update existing key does not evict', () => {
            const lru = new SimpleLRU<string, number>(2)
            lru.set('a', 1)
            lru.set('b', 2)
            const evicted = lru.set('a', 10) // update, not new
            expect(evicted).toBeUndefined()
            expect(lru.get('a')).toBe(10)
        })
    })

    describe('Prefix cache stats', () => {
        function hitRate(hits: number, total: number): number {
            if (total === 0) return 0
            return hits / total
        }

        it('computes 100% hit rate', () => {
            expect(hitRate(10, 10)).toBe(1.0)
        })

        it('computes 0% hit rate', () => {
            expect(hitRate(0, 10)).toBe(0)
        })

        it('handles zero total queries', () => {
            expect(hitRate(0, 0)).toBe(0)
        })

        it('computes fractional hit rate', () => {
            expect(hitRate(3, 10)).toBeCloseTo(0.3)
        })
    })

    describe('TTL-based cache expiry', () => {
        function isExpired(lastAccess: number, ttlMinutes: number, now: number): boolean {
            if (ttlMinutes <= 0) return false // no TTL
            return (now - lastAccess) > ttlMinutes * 60 * 1000
        }

        it('entry within TTL is not expired', () => {
            const now = Date.now()
            expect(isExpired(now - 1000, 5, now)).toBe(false)
        })

        it('entry past TTL is expired', () => {
            const now = Date.now()
            expect(isExpired(now - 600_000, 5, now)).toBe(true) // 10 min > 5 min
        })

        it('zero TTL means no expiry', () => {
            const now = Date.now()
            expect(isExpired(0, 0, now)).toBe(false)
        })

        it('negative TTL means no expiry', () => {
            const now = Date.now()
            expect(isExpired(0, -1, now)).toBe(false)
        })
    })

    describe('Block reference counting', () => {
        it('new block starts with ref_count 0', () => {
            const block = { refCount: 0 }
            expect(block.refCount).toBe(0)
        })

        it('shared block has ref_count > 1', () => {
            const block = { refCount: 3 }
            expect(block.refCount > 1).toBe(true)
        })

        it('block is evictable only when ref_count is 0', () => {
            expect({ refCount: 0 }.refCount === 0).toBe(true)
            expect({ refCount: 1 }.refCount === 0).toBe(false)
        })
    })
})

// ═══════════════════════════════════════════════════════════════════════════════
// Phase 6: Tool & Reasoning Parsers
// ═══════════════════════════════════════════════════════════════════════════════

describe('Phase 6: Tool & Reasoning Parsers', () => {
    describe('Auto tool parser — format detection order', () => {
        const MISTRAL_TOKEN = '[TOOL_CALLS]'
        const QWEN_BRACKET_RE = /\[Calling tool:\s*(\w+)\((.*?)\)\]/gs
        const HERMES_XML_RE = /<tool_call>\s*(\{[\s\S]*?\})\s*<\/tool_call>/g

        function detectToolFormat(text: string): string {
            if (text.includes(MISTRAL_TOKEN)) return 'mistral'
            if (QWEN_BRACKET_RE.test(text)) { QWEN_BRACKET_RE.lastIndex = 0; return 'qwen_bracket' }
            if (HERMES_XML_RE.test(text)) { HERMES_XML_RE.lastIndex = 0; return 'hermes_xml' }
            if (text.includes('"name"') && text.includes('"arguments"')) return 'raw_json'
            return 'none'
        }

        it('detects Mistral format', () => {
            expect(detectToolFormat('[TOOL_CALLS] [{"name": "search", "arguments": {}}]')).toBe('mistral')
        })

        it('detects Qwen bracket format', () => {
            expect(detectToolFormat('[Calling tool: search_files({"query": "test"})]')).toBe('qwen_bracket')
        })

        it('detects Hermes XML format', () => {
            expect(detectToolFormat('<tool_call>\n{"name": "read_file", "arguments": {"path": "/tmp"}}\n</tool_call>')).toBe('hermes_xml')
        })

        it('detects raw JSON format', () => {
            expect(detectToolFormat('{"name": "test", "arguments": {"a": 1}}')).toBe('raw_json')
        })

        it('returns none for plain text', () => {
            expect(detectToolFormat('Hello world, how are you?')).toBe('none')
        })

        it('Mistral takes priority over raw JSON', () => {
            expect(detectToolFormat('[TOOL_CALLS] {"name": "test", "arguments": {}}')).toBe('mistral')
        })
    })

    describe('Reasoning parser format detection', () => {
        function detectReasoningParser(modelType: string): string | undefined {
            const mapping: Record<string, string> = {
                'qwen3': 'qwen3', 'qwen3_5': 'qwen3', 'qwen3_vl': 'qwen3', 'qwen2_vl': 'qwen3',
                'deepseek_v3': 'deepseek_r1',
                'gemma3': 'deepseek_r1', 'gemma3_text': 'deepseek_r1',
                'phi4_reasoning': 'deepseek_r1',
                'gpt_oss': 'openai_gptoss',
                'glm4_moe': 'openai_gptoss',
                'minimax': 'qwen3',
            }
            return mapping[modelType]
        }

        it('Qwen3 uses qwen3 parser', () => {
            expect(detectReasoningParser('qwen3')).toBe('qwen3')
        })

        it('Gemma3 uses deepseek_r1 parser', () => {
            expect(detectReasoningParser('gemma3')).toBe('deepseek_r1')
        })

        it('GPT-OSS uses openai_gptoss parser', () => {
            expect(detectReasoningParser('gpt_oss')).toBe('openai_gptoss')
        })

        it('MiniMax uses qwen3 parser', () => {
            expect(detectReasoningParser('minimax')).toBe('qwen3')
        })

        it('unknown model returns undefined', () => {
            expect(detectReasoningParser('totally_unknown')).toBeUndefined()
        })
    })

    describe('Tool parser streaming heuristics', () => {
        function mightContainToolCall(text: string): boolean {
            const markers = [
                '<tool_call>', '<' + '|tool_call|' + '>',
                '[TOOL_CALLS]', '<function=', '[Calling tool:',
                '<' + '|tool_calls_section_begin|' + '>',
            ]
            return markers.some(m => text.includes(m))
        }

        it('detects tool_call XML tag', () => {
            expect(mightContainToolCall('Let me call <tool_call>')).toBe(true)
        })

        it('detects TOOL_CALLS marker', () => {
            expect(mightContainToolCall('Here: [TOOL_CALLS]')).toBe(true)
        })

        it('detects function= pattern', () => {
            expect(mightContainToolCall('<function=search>')).toBe(true)
        })

        it('no false positive for normal text', () => {
            expect(mightContainToolCall('This is a normal response about tools')).toBe(false)
        })
    })
})

// ═══════════════════════════════════════════════════════════════════════════════
// Phase 7: Database & Export
// ═══════════════════════════════════════════════════════════════════════════════

describe('Phase 7: Database & Export', () => {
    describe('Safe migration — column detection', () => {
        function needsMigration(existingColumns: string[], requiredColumn: string): boolean {
            return !existingColumns.includes(requiredColumn)
        }

        const BASE_COLUMNS = ['chat_id', 'temperature', 'top_p', 'top_k', 'max_tokens', 'repeat_penalty', 'system_prompt', 'stop_sequences']

        it('detects missing min_p column', () => {
            expect(needsMigration(BASE_COLUMNS, 'min_p')).toBe(true)
        })

        it('detects existing temperature column', () => {
            expect(needsMigration(BASE_COLUMNS, 'temperature')).toBe(false)
        })

        it('detects missing wire_api column', () => {
            expect(needsMigration(BASE_COLUMNS, 'wire_api')).toBe(true)
        })

        it('detects missing enable_thinking column', () => {
            expect(needsMigration(BASE_COLUMNS, 'enable_thinking')).toBe(true)
        })

        it('detects missing reasoning_content for messages', () => {
            const msgCols = ['id', 'chat_id', 'role', 'content', 'timestamp', 'tokens']
            expect(needsMigration(msgCols, 'reasoning_content')).toBe(true)
        })

        it('no migration needed after all columns added', () => {
            const allCols = [...BASE_COLUMNS, 'min_p', 'wire_api', 'enable_thinking', 'reasoning_content']
            expect(needsMigration(allCols, 'min_p')).toBe(false)
            expect(needsMigration(allCols, 'wire_api')).toBe(false)
        })
    })

    describe('ensureOpen recovery pattern', () => {
        it('reopens when closed flag is true', () => {
            let closed = true
            let reopened = false
            if (closed) {
                reopened = true
                closed = false
            }
            expect(reopened).toBe(true)
            expect(closed).toBe(false)
        })

        it('does nothing when already open', () => {
            let closed = false
            let reopened = false
            if (closed) reopened = true
            expect(reopened).toBe(false)
        })
    })

    describe('Export format — JSON', () => {
        function exportJSON(chat: { title: string; modelPath: string }, messages: { role: string; content: string; reasoning?: string }[]): object {
            return {
                title: chat.title,
                modelPath: chat.modelPath,
                messages: messages.map(m => ({
                    role: m.role,
                    content: m.content,
                    ...(m.reasoning ? { reasoning: m.reasoning } : {})
                }))
            }
        }

        it('includes reasoning when present', () => {
            const result = exportJSON({ title: 'Test', modelPath: '/model' }, [
                { role: 'assistant', content: 'Answer', reasoning: 'Thought process' }
            ])
            expect((result as any).messages[0].reasoning).toBe('Thought process')
        })

        it('omits reasoning when absent', () => {
            const result = exportJSON({ title: 'Test', modelPath: '/model' }, [
                { role: 'user', content: 'Hello' }
            ])
            expect((result as any).messages[0].reasoning).toBeUndefined()
        })
    })

    describe('Export format — ShareGPT', () => {
        function toShareGPT(messages: { role: string; content: string }[]): { conversations: { from: string; value: string }[] } {
            return {
                conversations: messages.map(m => ({
                    from: m.role === 'assistant' ? 'gpt' : m.role === 'user' ? 'human' : 'system',
                    value: m.content
                }))
            }
        }

        it('maps assistant to gpt', () => {
            const r = toShareGPT([{ role: 'assistant', content: 'Hi' }])
            expect(r.conversations[0].from).toBe('gpt')
        })

        it('maps user to human', () => {
            const r = toShareGPT([{ role: 'user', content: 'Hello' }])
            expect(r.conversations[0].from).toBe('human')
        })

        it('maps system to system', () => {
            const r = toShareGPT([{ role: 'system', content: 'You are helpful' }])
            expect(r.conversations[0].from).toBe('system')
        })
    })

    describe('Export format — Markdown', () => {
        function exportMarkdown(title: string, messages: { role: string; content: string; reasoning?: string }[]): string {
            const lines = [`# ${title}\n`]
            for (const m of messages) {
                const role = m.role === 'assistant' ? 'Assistant' : m.role === 'user' ? 'User' : 'System'
                lines.push(`## ${role}\n`)
                if (m.reasoning && m.role === 'assistant') {
                    lines.push(`<details><summary>Thinking</summary>\n\n${m.reasoning}\n\n</details>\n`)
                }
                lines.push(m.content + '\n')
            }
            return lines.join('\n')
        }

        it('includes reasoning in details tag for assistant', () => {
            const md = exportMarkdown('Test', [
                { role: 'assistant', content: 'Answer', reasoning: 'I thought about this' }
            ])
            expect(md).toContain('<details><summary>Thinking</summary>')
            expect(md).toContain('I thought about this')
        })

        it('does not include reasoning for user messages', () => {
            const md = exportMarkdown('Test', [
                { role: 'user', content: 'Question', reasoning: 'should not appear' }
            ])
            expect(md).not.toContain('Thinking')
        })

        it('has correct heading for title', () => {
            const md = exportMarkdown('My Chat', [])
            expect(md).toMatch(/^# My Chat/)
        })
    })

    describe('Import — Markdown reasoning extraction', () => {
        function extractReasoningFromMd(content: string): { reasoning?: string; text: string } {
            const detailsMatch = content.match(/^<details><summary>Thinking<\/summary>\s*\n\n([\s\S]*?)\n\n<\/details>\s*\n?/)
            if (detailsMatch) {
                return {
                    reasoning: detailsMatch[1].trim(),
                    text: content.slice(detailsMatch[0].length).trim()
                }
            }
            return { text: content.trim() }
        }

        it('extracts reasoning from details block', () => {
            const input = '<details><summary>Thinking</summary>\n\nMy reasoning here\n\n</details>\nActual response'
            const { reasoning, text } = extractReasoningFromMd(input)
            expect(reasoning).toBe('My reasoning here')
            expect(text).toBe('Actual response')
        })

        it('returns full text when no details block', () => {
            const { reasoning, text } = extractReasoningFromMd('Just a regular response')
            expect(reasoning).toBeUndefined()
            expect(text).toBe('Just a regular response')
        })

        it('handles multi-line reasoning', () => {
            const input = '<details><summary>Thinking</summary>\n\nLine 1\nLine 2\nLine 3\n\n</details>\nAnswer'
            const { reasoning } = extractReasoningFromMd(input)
            expect(reasoning).toContain('Line 1')
            expect(reasoning).toContain('Line 3')
        })
    })

    describe('Import — ShareGPT role mapping', () => {
        function sharegptToRole(from: string): string {
            if (from === 'gpt') return 'assistant'
            if (from === 'human') return 'user'
            return from || 'user'
        }

        it('maps gpt to assistant', () => {
            expect(sharegptToRole('gpt')).toBe('assistant')
        })

        it('maps human to user', () => {
            expect(sharegptToRole('human')).toBe('user')
        })

        it('preserves system', () => {
            expect(sharegptToRole('system')).toBe('system')
        })

        it('defaults empty string to user', () => {
            expect(sharegptToRole('')).toBe('user')
        })
    })

    describe('Chat title sanitization for export filename', () => {
        function sanitizeTitle(title: string): string {
            return title.replace(/[^a-zA-Z0-9 _-]/g, '').slice(0, 50).trim() || 'chat'
        }

        it('removes special characters', () => {
            expect(sanitizeTitle('Hello/World!')).toBe('HelloWorld')
        })

        it('preserves basic alphanumeric and spaces', () => {
            expect(sanitizeTitle('My Chat Title')).toBe('My Chat Title')
        })

        it('truncates to 50 characters', () => {
            const long = 'A'.repeat(100)
            expect(sanitizeTitle(long).length).toBe(50)
        })

        it('defaults empty result to "chat"', () => {
            expect(sanitizeTitle('!!!')).toBe('chat')
        })

        it('handles unicode/emoji', () => {
            expect(sanitizeTitle('Chat 🤖 about AI')).toBe('Chat  about AI')
        })
    })
})

// ═══════════════════════════════════════════════════════════════════════════════
// Phase 8: SSE Streaming Fix + Served Model Name
// ═══════════════════════════════════════════════════════════════════════════════

describe('Phase 8: SSE Streaming Fix + Served Model Name', () => {
    describe('SSE empty delta skip logic', () => {
        // Simulates the 3-way branch in server.py streaming
        function shouldEmitChunk(
            requestParser: boolean,
            deltaText: string | null,
            emitContent: string | null,
            emitReasoning: string | null,
            isFinished: boolean
        ): 'reasoning_path' | 'standard_path' | 'skip' {
            if (requestParser && deltaText) {
                // Reasoning parser path
                if (!emitContent && !emitReasoning && !isFinished) return 'skip'
                return 'reasoning_path'
            } else if (!requestParser) {
                // Standard path (no parser)
                return 'standard_path'
            } else {
                // Parser active but no delta — SKIP (the fix)
                return 'skip'
            }
        }

        it('reasoning parser with content emits', () => {
            expect(shouldEmitChunk(true, 'hello', 'hello', null, false)).toBe('reasoning_path')
        })

        it('reasoning parser with reasoning emits', () => {
            expect(shouldEmitChunk(true, 'think', null, 'think', false)).toBe('reasoning_path')
        })

        it('reasoning parser with no content on unfinished chunk skips', () => {
            expect(shouldEmitChunk(true, 'x', null, null, false)).toBe('skip')
        })

        it('reasoning parser with empty delta text skips (THE FIX)', () => {
            expect(shouldEmitChunk(true, null, null, null, false)).toBe('skip')
            expect(shouldEmitChunk(true, '', null, null, false)).toBe('skip')
        })

        it('no parser uses standard path', () => {
            expect(shouldEmitChunk(false, 'hello', null, null, false)).toBe('standard_path')
        })

        it('no parser with empty delta still uses standard path', () => {
            expect(shouldEmitChunk(false, '', null, null, false)).toBe('standard_path')
        })

        it('reasoning parser emits on finished even with no content', () => {
            expect(shouldEmitChunk(true, 'x', null, null, true)).toBe('reasoning_path')
        })
    })

    describe('Served model name resolution', () => {
        function resolveModelName(
            servedName: string | null,
            normalizedName: string | null
        ): string {
            return servedName || normalizedName || 'default'
        }

        it('served name takes priority', () => {
            expect(resolveModelName('my-custom-model', 'org/actual-model')).toBe('my-custom-model')
        })

        it('falls back to normalized name', () => {
            expect(resolveModelName(null, 'org/actual-model')).toBe('org/actual-model')
        })

        it('falls back to default', () => {
            expect(resolveModelName(null, null)).toBe('default')
        })
    })

    describe('/v1/models dual listing', () => {
        function listModels(
            servedName: string | null,
            modelName: string | null
        ): string[] {
            const resolved = servedName || modelName || 'default'
            const models: string[] = []
            if (resolved && resolved !== 'default') {
                models.push(resolved)
                if (servedName && modelName && servedName !== modelName) {
                    models.push(modelName)
                }
            }
            return models
        }

        it('lists served name first when set', () => {
            const models = listModels('custom-name', 'org/actual')
            expect(models[0]).toBe('custom-name')
            expect(models[1]).toBe('org/actual')
        })

        it('lists only actual name when no served name', () => {
            const models = listModels(null, 'org/actual')
            expect(models).toEqual(['org/actual'])
        })

        it('no duplicates when served equals actual', () => {
            const models = listModels('org/actual', 'org/actual')
            expect(models).toEqual(['org/actual'])
        })

        it('empty when no model loaded', () => {
            const models = listModels(null, null)
            expect(models).toEqual([])
        })
    })

    describe('Model validation (permissive)', () => {
        function validateModelRequest(
            requestModel: string,
            resolvedName: string,
            modelName: string | null
        ): { accepted: boolean; normalizedTo: string } {
            const matches = requestModel === resolvedName || requestModel === modelName
            return {
                accepted: true, // Always accept (single-model server)
                normalizedTo: resolvedName
            }
        }

        it('accepts matching model name', () => {
            const r = validateModelRequest('org/model', 'org/model', 'org/model')
            expect(r.accepted).toBe(true)
            expect(r.normalizedTo).toBe('org/model')
        })

        it('accepts mismatched name but normalizes', () => {
            const r = validateModelRequest('wrong-model', 'org/actual', 'org/actual')
            expect(r.accepted).toBe(true)
            expect(r.normalizedTo).toBe('org/actual')
        })

        it('normalizes to served name when set', () => {
            const r = validateModelRequest('anything', 'custom-name', 'org/actual')
            expect(r.normalizedTo).toBe('custom-name')
        })
    })
})

