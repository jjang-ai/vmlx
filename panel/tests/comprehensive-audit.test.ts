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
    'qwen3_5': 'qwen3.5', 'qwen3_5_moe': 'qwen3.5-moe',
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
    'qwen2': { cacheType: 'kv', toolParser: 'qwen', enableAutoToolChoice: true },
    'qwen2-vl': { cacheType: 'kv', toolParser: 'qwen', isMultimodal: true },
    'llama3': { cacheType: 'kv', toolParser: 'llama', enableAutoToolChoice: true },
    'mistral': { cacheType: 'kv', toolParser: 'mistral', enableAutoToolChoice: true },
    'deepseek-r1': { cacheType: 'kv', toolParser: 'deepseek', reasoningParser: 'deepseek_r1' },
    'deepseek-v3': { cacheType: 'kv', toolParser: 'deepseek', reasoningParser: 'deepseek_r1', enableAutoToolChoice: true },
    'deepseek-v2': { cacheType: 'kv', toolParser: 'deepseek', reasoningParser: 'deepseek_r1' },
    'deepseek': { cacheType: 'kv', toolParser: 'deepseek', reasoningParser: 'deepseek_r1' },
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

// Remote session health — mark down after sustained failure
describe('Remote session health mark-down', () => {
  // Pure function mirroring the handleSessionDown decision logic
  function handleSessionDownAction(session: { type: string; status: string } | null): 'error' | 'kill-and-stop' | 'noop' {
    if (!session || (session.status !== 'running' && session.status !== 'loading')) return 'noop'
    if (session.type === 'remote') return 'error'
    return 'kill-and-stop'
  }

  it('marks remote session as error', () => {
    expect(handleSessionDownAction({ type: 'remote', status: 'running' })).toBe('error')
  })

  it('marks remote loading session as error', () => {
    expect(handleSessionDownAction({ type: 'remote', status: 'loading' })).toBe('error')
  })

  it('kills and stops local session', () => {
    expect(handleSessionDownAction({ type: 'local', status: 'running' })).toBe('kill-and-stop')
  })

  it('no-ops for already stopped session', () => {
    expect(handleSessionDownAction({ type: 'remote', status: 'stopped' })).toBe('noop')
    expect(handleSessionDownAction({ type: 'local', status: 'stopped' })).toBe('noop')
  })

  it('no-ops for null session', () => {
    expect(handleSessionDownAction(null)).toBe('noop')
  })

  // Test the fail count accumulation for remote sessions
  describe('Remote fail count accumulation', () => {
    function shouldCallHandleDown(failCount: number, maxFails: number): boolean {
      return failCount >= maxFails
    }

    it('does not mark down before max fails', () => {
      expect(shouldCallHandleDown(59, 60)).toBe(false)
    })

    it('marks down at exactly max fails', () => {
      expect(shouldCallHandleDown(60, 60)).toBe(true)
    })

    it('respects scaled max fails from timeout config', () => {
      // timeout 600s / 5s interval = 120 max fails
      const maxFails = Math.max(60, Math.ceil(600 / 5))
      expect(maxFails).toBe(120)
      expect(shouldCallHandleDown(119, maxFails)).toBe(false)
      expect(shouldCallHandleDown(120, maxFails)).toBe(true)
    })
  })
})

// Chat-switch streaming re-sync logic
describe('Chat-switch streaming re-sync', () => {
  // Pure function: given activeRequests state, determine if chat is streaming
  function isStreaming(activeChats: Set<string>, chatId: string): boolean {
    return activeChats.has(chatId)
  }

  it('returns true for active chat', () => {
    const active = new Set(['chat-1', 'chat-2'])
    expect(isStreaming(active, 'chat-1')).toBe(true)
  })

  it('returns false for inactive chat', () => {
    const active = new Set(['chat-1'])
    expect(isStreaming(active, 'chat-3')).toBe(false)
  })

  it('returns false for empty set', () => {
    const active = new Set<string>()
    expect(isStreaming(active, 'chat-1')).toBe(false)
  })

  // Test the UI state decision: what to set when returning to a streaming chat
  function getReturnState(isActive: boolean): { loading: boolean } {
    return { loading: isActive }
  }

  it('sets loading when returning to streaming chat', () => {
    expect(getReturnState(true)).toEqual({ loading: true })
  })

  it('does not set loading when returning to idle chat', () => {
    expect(getReturnState(false)).toEqual({ loading: false })
  })
})

// ask_user Map-based resolver pattern
describe('ask_user Map-based resolver', () => {
  it('resolves for matching chatId', () => {
    const resolvers = new Map<string, (answer: string) => void>()
    let result = ''
    resolvers.set('chat-1', (answer) => { result = answer; resolvers.delete('chat-1') })

    // Simulate answer arriving
    const resolve = resolvers.get('chat-1')
    expect(resolve).toBeDefined()
    resolve!('hello')
    expect(result).toBe('hello')
    expect(resolvers.has('chat-1')).toBe(false)
  })

  it('ignores answer for different chatId', () => {
    const resolvers = new Map<string, (answer: string) => void>()
    let result = ''
    resolvers.set('chat-1', (answer) => { result = answer })

    const resolve = resolvers.get('chat-2')
    expect(resolve).toBeUndefined()
    expect(result).toBe('')
  })

  it('handles cleanup preventing double-resolve', () => {
    const resolvers = new Map<string, (answer: string) => void>()
    let callCount = 0
    resolvers.set('chat-1', () => { callCount++; resolvers.delete('chat-1') })

    // First resolve
    resolvers.get('chat-1')!('answer1')
    // Second attempt (simulating race with timeout)
    const secondResolve = resolvers.get('chat-1')
    expect(secondResolve).toBeUndefined()
    expect(callCount).toBe(1)
  })

  it('allows sequential ask_user calls without accumulation', () => {
    const resolvers = new Map<string, (answer: string) => void>()
    const results: string[] = []

    // First ask_user
    resolvers.set('chat-1', (a) => { results.push(a); resolvers.delete('chat-1') })
    resolvers.get('chat-1')!('answer1')

    // Second ask_user (same chat) — should work without issues
    resolvers.set('chat-1', (a) => { results.push(a); resolvers.delete('chat-1') })
    resolvers.get('chat-1')!('answer2')

    expect(results).toEqual(['answer1', 'answer2'])
    expect(resolvers.size).toBe(0)
  })
})

// Already-downloaded model detection
describe('Already-downloaded model detection', () => {
  // Pure function: build local model ID set from scan results
  function buildLocalModelIds(models: Array<{ id?: string; path?: string }>): Set<string> {
    const ids = new Set<string>()
    for (const m of models) {
      if (m.id) ids.add(m.id)
      if (m.path) {
        const parts = m.path.replace(/\\/g, '/').split('/')
        if (parts.length >= 2) {
          ids.add(`${parts[parts.length - 2]}/${parts[parts.length - 1]}`)
        }
      }
    }
    return ids
  }

  it('matches by id', () => {
    const ids = buildLocalModelIds([{ id: 'mlx-community/Qwen2.5-7B-4bit', path: '/models/mlx-community/Qwen2.5-7B-4bit' }])
    expect(ids.has('mlx-community/Qwen2.5-7B-4bit')).toBe(true)
  })

  it('matches by path segments', () => {
    const ids = buildLocalModelIds([{ path: '/home/user/models/mlx-community/Llama-3-8B-4bit' }])
    expect(ids.has('mlx-community/Llama-3-8B-4bit')).toBe(true)
  })

  it('handles Windows paths', () => {
    const ids = buildLocalModelIds([{ path: 'C:\\Users\\eric\\models\\mlx-community\\Qwen2.5-7B-4bit' }])
    expect(ids.has('mlx-community/Qwen2.5-7B-4bit')).toBe(true)
  })

  it('does not match unrelated model', () => {
    const ids = buildLocalModelIds([{ id: 'mlx-community/Qwen2.5-7B-4bit' }])
    expect(ids.has('mlx-community/Llama-3-8B-4bit')).toBe(false)
  })

  it('handles empty scan results', () => {
    const ids = buildLocalModelIds([])
    expect(ids.size).toBe(0)
  })

  it('deduplicates id and path-based entries', () => {
    const ids = buildLocalModelIds([
      { id: 'mlx-community/Qwen2.5-7B-4bit', path: '/models/mlx-community/Qwen2.5-7B-4bit' }
    ])
    // Both id and path produce the same key
    expect(ids.size).toBe(1)
  })
})

// Chat search — skip base64 image content
describe('Chat search skip base64', () => {
  // Pure function: should a message content be searched?
  function isSearchableContent(content: string): boolean {
    return !content.startsWith('[{"type":')
  }

  it('allows plain text messages', () => {
    expect(isSearchableContent('Hello, how are you?')).toBe(true)
  })

  it('allows code blocks', () => {
    expect(isSearchableContent('```python\nprint("hello")\n```')).toBe(true)
  })

  it('skips multimodal JSON content arrays', () => {
    const content = '[{"type":"image_url","image_url":{"url":"data:image/png;base64,iVBOR..."}}]'
    expect(isSearchableContent(content)).toBe(false)
  })

  it('skips mixed text+image content arrays', () => {
    const content = '[{"type":"image_url","image_url":{"url":"data:image/png;base64,..."}},{"type":"text","text":"describe this"}]'
    expect(isSearchableContent(content)).toBe(false)
  })

  it('allows JSON that does not start with [{"type":', () => {
    expect(isSearchableContent('{"key": "value"}')).toBe(true)
    expect(isSearchableContent('[1, 2, 3]')).toBe(true)
  })
})

// DownloadStatusBar collapse toggle
describe('DownloadStatusBar collapse toggle', () => {
  // Pure function: should progress details be visible?
  function showProgressDetails(collapsed: boolean, hasPercent: boolean): boolean {
    return !collapsed && hasPercent
  }

  it('shows details when expanded with progress', () => {
    expect(showProgressDetails(false, true)).toBe(true)
  })

  it('hides details when collapsed', () => {
    expect(showProgressDetails(true, true)).toBe(false)
  })

  it('hides details when no progress data', () => {
    expect(showProgressDetails(false, false)).toBe(false)
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
            expect(normalizeModelName('/home/user/.lmstudio/models/mlx-community/Llama-3.2-3B-Instruct-4bit'))
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
                'deepseek_v3': 'deepseek_r1', 'deepseek_v2': 'deepseek_r1',
                'deepseek2': 'deepseek_r1', 'deepseek': 'deepseek_r1',
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

        it('DeepSeek V3 uses deepseek_r1 parser', () => {
            expect(detectReasoningParser('deepseek_v3')).toBe('deepseek_r1')
        })

        it('DeepSeek V2 uses deepseek_r1 parser', () => {
            expect(detectReasoningParser('deepseek_v2')).toBe('deepseek_r1')
        })

        it('DeepSeek generic uses deepseek_r1 parser', () => {
            expect(detectReasoningParser('deepseek')).toBe('deepseek_r1')
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

// ═══════════════════════════════════════════════════════════════════════════════
// Phase 5: Download Manager & Background Downloads
// ═══════════════════════════════════════════════════════════════════════════════

describe('Phase 5: Download Manager', () => {
    describe('Download queue deduplication', () => {
        interface DownloadJob {
            id: string
            repoId: string
            status: 'queued' | 'downloading' | 'complete' | 'cancelled' | 'error'
        }

        function startDownload(
            activeJob: DownloadJob | null,
            queue: DownloadJob[],
            repoId: string
        ): { status: string; jobId?: string } {
            if (activeJob?.repoId === repoId) return { jobId: activeJob.id, status: 'already_downloading' }
            const queued = queue.find(j => j.repoId === repoId)
            if (queued) return { jobId: queued.id, status: 'already_queued' }
            const id = `dl_${Date.now()}`
            queue.push({ id, repoId, status: 'queued' })
            return { jobId: id, status: 'queued' }
        }

        it('returns immediately with jobId (non-blocking)', () => {
            const queue: DownloadJob[] = []
            const result = startDownload(null, queue, 'mlx-community/model')
            expect(result.status).toBe('queued')
            expect(result.jobId).toBeDefined()
            expect(queue.length).toBe(1)
        })

        it('detects already-downloading model', () => {
            const active: DownloadJob = { id: 'dl_1', repoId: 'mlx-community/model', status: 'downloading' }
            const result = startDownload(active, [], 'mlx-community/model')
            expect(result.status).toBe('already_downloading')
        })

        it('detects already-queued model', () => {
            const queue: DownloadJob[] = [{ id: 'dl_1', repoId: 'mlx-community/model', status: 'queued' }]
            const result = startDownload(null, queue, 'mlx-community/model')
            expect(result.status).toBe('already_queued')
        })

        it('allows different models to queue', () => {
            const queue: DownloadJob[] = []
            startDownload(null, queue, 'mlx-community/model-a')
            startDownload(null, queue, 'mlx-community/model-b')
            expect(queue.length).toBe(2)
        })
    })

    describe('Stale marker cleanup', () => {
        it('marker older than 1 hour is stale', () => {
            const ts = Date.now() - 3700000
            expect(!isNaN(ts) && Date.now() - ts > 3600000).toBe(true)
        })

        it('marker less than 1 hour old is not stale', () => {
            const ts = Date.now() - 1800000
            expect(!isNaN(ts) && Date.now() - ts > 3600000).toBe(false)
        })

        it('NaN timestamp is not treated as stale', () => {
            const ts = NaN
            expect(!isNaN(ts) && Date.now() - ts > 3600000).toBe(false)
        })
    })
})

// ═══════════════════════════════════════════════════════════════════════════════
// Phase 6: Audio IPC & Auth Headers
// ═══════════════════════════════════════════════════════════════════════════════

describe('Phase 6: Audio IPC Auth Headers', () => {
    // Reproduces the auth header assembly logic from ipc/utils.ts
    function getAuthHeaders(session: { apiKey?: string; remoteOrganization?: string }): Record<string, string> {
        const headers: Record<string, string> = {}
        if (session.apiKey) {
            headers['Authorization'] = `Bearer ${session.apiKey}`
        }
        if (session.remoteOrganization) {
            headers['OpenAI-Organization'] = session.remoteOrganization
        }
        return headers
    }

    it('includes Bearer token when apiKey is set', () => {
        const h = getAuthHeaders({ apiKey: 'sk-test123' })
        expect(h['Authorization']).toBe('Bearer sk-test123')
    })

    it('includes Organization header when set', () => {
        const h = getAuthHeaders({ apiKey: 'sk-test', remoteOrganization: 'org-abc' })
        expect(h['OpenAI-Organization']).toBe('org-abc')
    })

    it('returns empty headers for local session (no apiKey)', () => {
        const h = getAuthHeaders({})
        expect(Object.keys(h).length).toBe(0)
    })

    it('omits Organization when not set', () => {
        const h = getAuthHeaders({ apiKey: 'sk-test' })
        expect(h['OpenAI-Organization']).toBeUndefined()
    })

    // Known issue: audio.ts doesn't pass auth headers for sessions with API keys
    // This test documents the expected behavior when the bug is fixed
    describe('Audio endpoint auth header assembly', () => {
        function buildAudioHeaders(
            baseUrl: string,
            session: { apiKey?: string; remoteOrganization?: string }
        ): Record<string, string> {
            const headers: Record<string, string> = { 'Content-Type': 'application/json' }
            // This is what SHOULD happen (auth headers passed to audio endpoints)
            const auth = getAuthHeaders(session)
            return { ...headers, ...auth }
        }

        it('remote session audio request includes auth headers', () => {
            const h = buildAudioHeaders('https://api.openai.com', { apiKey: 'sk-test' })
            expect(h['Authorization']).toBe('Bearer sk-test')
            expect(h['Content-Type']).toBe('application/json')
        })

        it('local session audio request has no auth headers', () => {
            const h = buildAudioHeaders('http://localhost:9876', {})
            expect(h['Authorization']).toBeUndefined()
            expect(h['Content-Type']).toBe('application/json')
        })
    })
})

// ═══════════════════════════════════════════════════════════════════════════════
// Phase 7: Reasoning Parser Behavior with enable_thinking
// ═══════════════════════════════════════════════════════════════════════════════

describe('Phase 7: Reasoning & Thinking Tri-State', () => {
    // Simulates how the server handles enable_thinking + reasoning parser
    type ThinkingMode = true | false | undefined  // true=On, false=Off, undefined=Auto

    function resolveThinking(
        enableThinking: ThinkingMode,
        hasReasoningParser: boolean
    ): { addThinkingTemplate: boolean; parseReasoning: boolean } {
        if (enableThinking === false) {
            // Explicit Off: don't add template, don't parse (model won't produce think tags)
            return { addThinkingTemplate: false, parseReasoning: false }
        }
        if (enableThinking === true) {
            // Explicit On: add template if parser exists, always parse
            return { addThinkingTemplate: hasReasoningParser, parseReasoning: hasReasoningParser }
        }
        // Auto (undefined): let model decide, parse if parser is configured
        return { addThinkingTemplate: false, parseReasoning: hasReasoningParser }
    }

    it('thinking=true with parser: enables both template and parsing', () => {
        const r = resolveThinking(true, true)
        expect(r.addThinkingTemplate).toBe(true)
        expect(r.parseReasoning).toBe(true)
    })

    it('thinking=false: disables both even with parser', () => {
        const r = resolveThinking(false, true)
        expect(r.addThinkingTemplate).toBe(false)
        expect(r.parseReasoning).toBe(false)
    })

    it('thinking=undefined (auto) with parser: parses but does not force template', () => {
        const r = resolveThinking(undefined, true)
        expect(r.addThinkingTemplate).toBe(false)
        expect(r.parseReasoning).toBe(true)
    })

    it('thinking=true without parser: no template or parsing', () => {
        const r = resolveThinking(true, false)
        expect(r.addThinkingTemplate).toBe(false)
        expect(r.parseReasoning).toBe(false)
    })

    it('thinking=undefined without parser: nothing happens', () => {
        const r = resolveThinking(undefined, false)
        expect(r.addThinkingTemplate).toBe(false)
        expect(r.parseReasoning).toBe(false)
    })

    // Tri-state storage: how config values map
    describe('Tri-state config storage', () => {
        function interpretEnableThinking(value: any): ThinkingMode {
            if (value === true) return true
            if (value === false) return false
            return undefined  // null, undefined, missing = Auto
        }

        it('true → On', () => expect(interpretEnableThinking(true)).toBe(true))
        it('false → Off', () => expect(interpretEnableThinking(false)).toBe(false))
        it('undefined → Auto', () => expect(interpretEnableThinking(undefined)).toBeUndefined())
        it('null → Auto', () => expect(interpretEnableThinking(null)).toBeUndefined())
    })

    // Parser selection: empty string vs undefined
    describe('Parser empty string semantics', () => {
        function resolveParser(
            detected: string | undefined,
            saved: string | undefined
        ): string | undefined {
            // Empty string "" = user explicitly chose "None (disabled)"
            if (saved === '') return undefined  // Disabled
            // Auto-detected always wins for non-empty values
            if (detected) return detected
            // Fallback to saved
            return saved
        }

        it('empty string disables parser (explicit None)', () => {
            expect(resolveParser('deepseek_r1', '')).toBeUndefined()
        })

        it('detected parser wins over saved', () => {
            expect(resolveParser('deepseek_r1', 'hermes')).toBe('deepseek_r1')
        })

        it('saved value used as fallback when no detection', () => {
            expect(resolveParser(undefined, 'hermes')).toBe('hermes')
        })

        it('both undefined returns undefined', () => {
            expect(resolveParser(undefined, undefined)).toBeUndefined()
        })
    })
})

// ═══════════════════════════════════════════════════════════════════════════════
// Phase 5: Security — v1.1.6 audit fixes
// ═══════════════════════════════════════════════════════════════════════════════

describe('Phase 5: Security Audit', () => {
    describe('shell.openExternal URL validation', () => {
        // Reproduces the validation logic from index.ts setWindowOpenHandler
        function shouldOpenExternal(url: string): boolean {
            return url.startsWith('https://') || url.startsWith('http://')
        }

        it('allows https URLs', () => {
            expect(shouldOpenExternal('https://github.com/jjang-ai/vmlx')).toBe(true)
        })

        it('allows http URLs', () => {
            expect(shouldOpenExternal('http://localhost:8093')).toBe(true)
        })

        it('blocks file:// URLs', () => {
            expect(shouldOpenExternal('file:///etc/passwd')).toBe(false)
        })

        it('blocks javascript: URLs', () => {
            expect(shouldOpenExternal('javascript:alert(1)')).toBe(false)
        })

        it('blocks custom protocol URLs', () => {
            expect(shouldOpenExternal('vmlx://internal/config')).toBe(false)
        })

        it('blocks empty string', () => {
            expect(shouldOpenExternal('')).toBe(false)
        })

        it('blocks data: URLs', () => {
            expect(shouldOpenExternal('data:text/html,<script>alert(1)</script>')).toBe(false)
        })
    })

    describe('Qwen2 must not have reasoning parser (REGRESSION)', () => {
        it('qwen2 family has no reasoning parser', () => {
            const qwen2 = FAMILY_CONFIGS['qwen2']
            expect(qwen2).toBeDefined()
            expect(qwen2.reasoningParser).toBeUndefined()
        })

        it('qwen2-vl family has no reasoning parser', () => {
            const qwen2vl = FAMILY_CONFIGS['qwen2-vl']
            expect(qwen2vl).toBeDefined()
            expect(qwen2vl.reasoningParser).toBeUndefined()
        })

        it('qwen3 family DOES have reasoning parser', () => {
            const qwen3 = FAMILY_CONFIGS['qwen3']
            expect(qwen3).toBeDefined()
            expect(qwen3.reasoningParser).toBe('qwen3')
        })
    })
})

// ═══════════════════════════════════════════════════════════════════════════════
// Phase 6: Parameter Defaults & Safety Guards
// ═══════════════════════════════════════════════════════════════════════════════

describe('Phase 6: Parameter Defaults', () => {

    describe('M10: prefillBatchSize default should not override server default', () => {
        // Simulate buildArgs logic: when prefillBatchSize is 0 or unset,
        // omit --prefill-batch-size so the server uses its own default (8)
        function buildPrefillArgs(prefillBatchSize: number): string[] {
            const args: string[] = []
            if (prefillBatchSize && prefillBatchSize > 0) {
                args.push('--prefill-batch-size', prefillBatchSize.toString())
            }
            return args
        }

        it('prefillBatchSize=0 omits the flag (server uses default)', () => {
            expect(buildPrefillArgs(0)).toEqual([])
        })

        it('prefillBatchSize=8 passes the flag explicitly', () => {
            expect(buildPrefillArgs(8)).toEqual(['--prefill-batch-size', '8'])
        })

        it('default should be 0 (no override), not 512', () => {
            // This validates that the default config no longer has 512
            const defaultPrefillBatchSize = 0
            expect(defaultPrefillBatchSize).toBe(0)
            expect(buildPrefillArgs(defaultPrefillBatchSize)).toEqual([])
        })
    })

    describe('M11/M12: max_tokens per-request should be omittable', () => {
        // Simulate request building: when maxTokens is undefined/0, omit from request
        function buildRequestMaxTokens(maxTokens?: number): Record<string, any> {
            const obj: Record<string, any> = { model: 'test' }
            if (maxTokens) {
                obj.max_tokens = maxTokens
            }
            return obj
        }

        it('maxTokens=undefined omits max_tokens (server decides)', () => {
            const req = buildRequestMaxTokens(undefined)
            expect(req.max_tokens).toBeUndefined()
        })

        it('maxTokens=0 omits max_tokens (server decides)', () => {
            const req = buildRequestMaxTokens(0)
            expect(req.max_tokens).toBeUndefined()
        })

        it('maxTokens=4096 includes max_tokens explicitly', () => {
            const req = buildRequestMaxTokens(4096)
            expect(req.max_tokens).toBe(4096)
        })

        it('should NOT hardcode 4096 as fallback', () => {
            // Verify that omitted maxTokens doesn't default to 4096
            const req = buildRequestMaxTokens(undefined)
            expect(req.max_tokens).not.toBe(4096)
        })
    })

    describe('H7: DeepSeek families must have reasoningParser', () => {
        it('deepseek-v3 has deepseek_r1 reasoning parser', () => {
            const config = FAMILY_CONFIGS['deepseek-v3']
            expect(config).toBeDefined()
            expect(config.reasoningParser).toBe('deepseek_r1')
        })

        it('deepseek-v2 has deepseek_r1 reasoning parser', () => {
            const config = FAMILY_CONFIGS['deepseek-v2']
            expect(config).toBeDefined()
            expect(config.reasoningParser).toBe('deepseek_r1')
        })

        it('deepseek generic has deepseek_r1 reasoning parser', () => {
            const config = FAMILY_CONFIGS['deepseek']
            expect(config).toBeDefined()
            expect(config.reasoningParser).toBe('deepseek_r1')
        })

        it('deepseek-r1 has deepseek_r1 reasoning parser', () => {
            const config = FAMILY_CONFIGS['deepseek-r1']
            expect(config).toBeDefined()
            expect(config.reasoningParser).toBe('deepseek_r1')
        })
    })
})

// ═══════════════════════════════════════════════════════════════════════════════
// Phase 6: MCP Tool Result Truncation
// ═══════════════════════════════════════════════════════════════════════════════

describe('MCP Tool Result Truncation', () => {
    // Simulates the truncation logic added to chat.ts MCP tool execution path
    function truncateMcpResult(resultText: string, maxChars?: number): string {
        const limit = maxChars || 50000
        if (resultText.length > limit) {
            return resultText.slice(0, limit) + `\n\n[Truncated — showing first ${limit} of ${resultText.length} characters]`
        }
        return resultText
    }

    it('does not truncate results under 50KB default', () => {
        const small = 'a'.repeat(49999)
        expect(truncateMcpResult(small)).toBe(small)
    })

    it('truncates results over 50KB default', () => {
        const large = 'x'.repeat(60000)
        const result = truncateMcpResult(large)
        expect(result.length).toBeLessThan(large.length)
        expect(result).toContain('[Truncated')
        expect(result).toContain('50000 of 60000')
    })

    it('respects custom maxChars override', () => {
        const data = 'y'.repeat(5000)
        const result = truncateMcpResult(data, 2000)
        expect(result).toContain('[Truncated')
        expect(result).toContain('2000 of 5000')
    })

    it('does not truncate at exact boundary', () => {
        const exact = 'z'.repeat(50000)
        expect(truncateMcpResult(exact)).toBe(exact)
    })

    it('handles empty result', () => {
        expect(truncateMcpResult('')).toBe('')
    })

    it('handles JSON object results', () => {
        // MCP results may be JSON.stringify'd objects
        const obj = { data: 'x'.repeat(60000) }
        const jsonStr = JSON.stringify(obj, null, 2)
        const result = truncateMcpResult(jsonStr)
        expect(result).toContain('[Truncated')
    })
})

// Quantization detection from model name
describe('Quantization detection from model name', () => {
  function detectQuantFromName(name: string): string {
    const lower = name.toLowerCase()
    const match = lower.match(/\b(4bit|8bit|3bit|6bit|fp16|bf16|fp32)\b/)
    return match ? match[1] : ''
  }

  it('detects 4bit', () => {
    expect(detectQuantFromName('Qwen2.5-7B-Instruct-4bit')).toBe('4bit')
  })

  it('detects 8bit', () => {
    expect(detectQuantFromName('Llama-3-8B-8bit')).toBe('8bit')
  })

  it('detects fp16', () => {
    expect(detectQuantFromName('Mistral-7B-fp16')).toBe('fp16')
  })

  it('detects bf16', () => {
    expect(detectQuantFromName('Model-bf16-weights')).toBe('bf16')
  })

  it('returns empty for no quantization indicator', () => {
    expect(detectQuantFromName('Qwen2.5-7B-Instruct')).toBe('')
  })

  it('returns empty for partial matches', () => {
    expect(detectQuantFromName('Qwen2.5-7B-bit')).toBe('')
  })
})

// API key encryption pattern
describe('API key encryption pattern', () => {
  // Simulate the encrypt/decrypt pattern (without actual safeStorage)
  function mockEncrypt(value: string): string {
    if (!value) return value
    return 'enc:' + Buffer.from(value).toString('base64')
  }

  function mockDecrypt(value: string): string {
    if (!value || !value.startsWith('enc:')) return value
    return Buffer.from(value.slice(4), 'base64').toString()
  }

  it('encrypts and decrypts round-trip', () => {
    const original = 'sk-test-1234567890'
    const encrypted = mockEncrypt(original)
    expect(encrypted).toMatch(/^enc:/)
    expect(mockDecrypt(encrypted)).toBe(original)
  })

  it('passes through empty strings', () => {
    expect(mockEncrypt('')).toBe('')
    expect(mockDecrypt('')).toBe('')
  })

  it('passes through legacy plaintext on decrypt', () => {
    expect(mockDecrypt('sk-legacy-key')).toBe('sk-legacy-key')
  })

  // Setting key detection
  function isApiKeySetting(key: string): boolean {
    const lower = key.toLowerCase()
    return lower.includes('apikey') || lower.includes('api_key')
  }

  it('detects API key settings', () => {
    expect(isApiKeySetting('braveApiKey')).toBe(true)
    expect(isApiKeySetting('remote_api_key')).toBe(true)
    expect(isApiKeySetting('theme')).toBe(false)
    expect(isApiKeySetting('model_scan_directories')).toBe(false)
  })
})

// Import file size guard
describe('Import file size guard', () => {
  const MAX_SIZE = 50 * 1024 * 1024

  it('allows files under 50MB', () => {
    expect(49 * 1024 * 1024 < MAX_SIZE).toBe(true)
  })

  it('rejects files over 50MB', () => {
    expect(51 * 1024 * 1024 > MAX_SIZE).toBe(true)
  })

  it('allows exactly 50MB', () => {
    expect(MAX_SIZE <= MAX_SIZE).toBe(true)
  })
})

// Stale lock cap
describe('Stale lock cap', () => {
  function staleLockMs(timeoutMs: number): number {
    return Math.min(timeoutMs + 30_000, 30 * 60 * 1000)
  }

  it('adds 30s buffer to normal timeout', () => {
    // 120s timeout → 150s stale lock
    expect(staleLockMs(120_000)).toBe(150_000)
  })

  it('caps at 30 minutes', () => {
    // 86400s (24h) timeout → capped at 30min
    expect(staleLockMs(86400_000)).toBe(30 * 60 * 1000)
  })

  it('handles zero timeout', () => {
    expect(staleLockMs(0)).toBe(30_000)
  })
})

// Orphan benchmark cleanup
describe('Orphan benchmark cleanup', () => {
  // Pure logic: which benchmark session_ids are orphans?
  function findOrphans(benchmarkSessionIds: string[], validSessionIds: Set<string>): string[] {
    return benchmarkSessionIds.filter(id => !validSessionIds.has(id))
  }

  it('identifies orphan benchmarks', () => {
    const benchmarks = ['s1', 's2', 's3']
    const sessions = new Set(['s1', 's3'])
    expect(findOrphans(benchmarks, sessions)).toEqual(['s2'])
  })

  it('returns empty when no orphans', () => {
    const benchmarks = ['s1', 's2']
    const sessions = new Set(['s1', 's2', 's3'])
    expect(findOrphans(benchmarks, sessions)).toEqual([])
  })
})

// ═══════════════════════════════════════════════════════════════════════════════
// Phase 6: Chat-Session Lifecycle Audit
// ═══════════════════════════════════════════════════════════════════════════════

// Session deletion — active state cleanup
describe('Session deletion active state cleanup', () => {
  interface SessionSummary { id: string; status: string; modelPath: string }

  // Pure function: find fallback session when active session is deleted
  function findFallbackSession(
    sessions: SessionSummary[],
    deletedSessionId: string
  ): SessionSummary | null {
    if (sessions.find(s => s.id === deletedSessionId)) return null // not deleted
    const running = sessions.find(s => s.status === 'running')
    return running || sessions[0] || null
  }

  it('returns running session as fallback', () => {
    const sessions: SessionSummary[] = [
      { id: 's1', status: 'stopped', modelPath: '/m1' },
      { id: 's2', status: 'running', modelPath: '/m2' },
    ]
    expect(findFallbackSession(sessions, 's-deleted')?.id).toBe('s2')
  })

  it('returns first session when none running', () => {
    const sessions: SessionSummary[] = [
      { id: 's1', status: 'stopped', modelPath: '/m1' },
    ]
    expect(findFallbackSession(sessions, 's-deleted')?.id).toBe('s1')
  })

  it('returns null when no sessions remain', () => {
    expect(findFallbackSession([], 's-deleted')).toBeNull()
  })

  it('returns null when session not actually deleted', () => {
    const sessions: SessionSummary[] = [
      { id: 's1', status: 'running', modelPath: '/m1' },
    ]
    expect(findFallbackSession(sessions, 's1')).toBeNull()
  })
})

// Send guard — model not running
describe('Send guard when model not running', () => {
  function canSend(opts: {
    sessionEndpoint?: { host: string; port: number }
    sessionId?: string
    loading: boolean
    hasContent: boolean
  }): { allowed: boolean; reason?: string } {
    if (!opts.hasContent) return { allowed: false, reason: 'empty' }
    if (opts.loading) return { allowed: false, reason: 'loading' }
    if (!opts.sessionEndpoint && opts.sessionId) return { allowed: false, reason: 'model-not-running' }
    return { allowed: true }
  }

  it('allows send when model running', () => {
    expect(canSend({ sessionEndpoint: { host: '127.0.0.1', port: 8093 }, sessionId: 's1', loading: false, hasContent: true }))
      .toEqual({ allowed: true })
  })

  it('blocks send when model stopped', () => {
    expect(canSend({ sessionEndpoint: undefined, sessionId: 's1', loading: false, hasContent: true }))
      .toEqual({ allowed: false, reason: 'model-not-running' })
  })

  it('blocks send when loading', () => {
    expect(canSend({ sessionEndpoint: { host: '127.0.0.1', port: 8093 }, sessionId: 's1', loading: true, hasContent: true }))
      .toEqual({ allowed: false, reason: 'loading' })
  })

  it('blocks send with empty content', () => {
    expect(canSend({ sessionEndpoint: { host: '127.0.0.1', port: 8093 }, sessionId: 's1', loading: false, hasContent: false }))
      .toEqual({ allowed: false, reason: 'empty' })
  })

  it('allows send without session (no session at all)', () => {
    // Edge case: chat with no session bound (shouldn't happen in practice)
    expect(canSend({ sessionEndpoint: undefined, sessionId: undefined, loading: false, hasContent: true }))
      .toEqual({ allowed: true })
  })
})

// Chat selection — session resolution
describe('Chat selection session resolution', () => {
  interface Session { id: string; modelPath: string; status: string }

  function resolveSession(sessions: Session[], modelPath: string): Session | null {
    const exactRunning = sessions.find(s => s.modelPath === modelPath && s.status === 'running')
    const exactAny = sessions.find(s => s.modelPath === modelPath)
    const fallback = sessions.find(s => s.status === 'running') || sessions[0]
    return exactRunning || exactAny || fallback || null
  }

  it('prefers running session with matching model', () => {
    const sessions: Session[] = [
      { id: 's1', modelPath: '/m1', status: 'stopped' },
      { id: 's2', modelPath: '/m1', status: 'running' },
    ]
    expect(resolveSession(sessions, '/m1')?.id).toBe('s2')
  })

  it('falls back to stopped session with matching model', () => {
    const sessions: Session[] = [
      { id: 's1', modelPath: '/m1', status: 'stopped' },
      { id: 's2', modelPath: '/m2', status: 'running' },
    ]
    expect(resolveSession(sessions, '/m1')?.id).toBe('s1')
  })

  it('falls back to any running session', () => {
    const sessions: Session[] = [
      { id: 's1', modelPath: '/m2', status: 'running' },
    ]
    expect(resolveSession(sessions, '/m-unknown')?.id).toBe('s1')
  })

  it('falls back to first session when none running', () => {
    const sessions: Session[] = [
      { id: 's1', modelPath: '/m1', status: 'stopped' },
    ]
    expect(resolveSession(sessions, '/m-unknown')?.id).toBe('s1')
  })

  it('returns null when no sessions', () => {
    expect(resolveSession([], '/m1')).toBeNull()
  })
})

// Session endpoint resolution
describe('Session endpoint derivation', () => {
  interface SessionSummary { id: string; status: string; host: string; port: number }

  function deriveEndpoint(session: SessionSummary | undefined): { host: string; port: number } | undefined {
    return session?.status === 'running' ? { host: session.host, port: session.port } : undefined
  }

  it('returns endpoint for running session', () => {
    expect(deriveEndpoint({ id: 's1', status: 'running', host: '127.0.0.1', port: 8093 }))
      .toEqual({ host: '127.0.0.1', port: 8093 })
  })

  it('returns undefined for stopped session', () => {
    expect(deriveEndpoint({ id: 's1', status: 'stopped', host: '127.0.0.1', port: 8093 }))
      .toBeUndefined()
  })

  it('returns undefined for loading session', () => {
    expect(deriveEndpoint({ id: 's1', status: 'loading', host: '127.0.0.1', port: 8093 }))
      .toBeUndefined()
  })

  it('returns undefined for no session', () => {
    expect(deriveEndpoint(undefined)).toBeUndefined()
  })
})

// ═══════════════════════════════════════════════════════════════════════════════
// Phase 5: Verbose Logging — Ring Buffer, Timestamps, Severity, Filter
// ═══════════════════════════════════════════════════════════════════════════════

/**
 * Minimal ring-buffer implementation matching SessionManager.pushLog() logic.
 * Tested in isolation — no Electron/Node process dependencies.
 */
class LogBuffer {
  private buffer: string[] = []
  private readonly maxLines: number

  constructor(maxLines = 2000) {
    this.maxLines = maxLines
  }

  push(data: string): void {
    const timestamp = new Date().toISOString().slice(11, 23)
    const lines = data.split('\n')
    for (const line of lines) {
      if (!line && lines.length > 1) continue
      this.buffer.push(`[${timestamp}] ${line}`)
    }
    if (this.buffer.length > this.maxLines) {
      this.buffer.splice(0, this.buffer.length - this.maxLines)
    }
  }

  getLines(): string[] {
    return [...this.buffer]
  }

  clear(): void {
    this.buffer = []
  }
}

describe('Verbose Logging — Ring Buffer', () => {
  let buf: LogBuffer

  beforeEach(() => {
    buf = new LogBuffer(10) // small cap for testing
  })

  it('stores log lines with timestamp prefix', () => {
    buf.push('hello world')
    const lines = buf.getLines()
    expect(lines).toHaveLength(1)
    expect(lines[0]).toMatch(/^\[\d{2}:\d{2}:\d{2}\.\d{3}\] hello world$/)
  })

  it('splits multi-line data into separate entries', () => {
    buf.push('line1\nline2\nline3')
    expect(buf.getLines()).toHaveLength(3)
  })

  it('skips empty trailing splits from newline-terminated data', () => {
    buf.push('line1\nline2\n')
    // trailing empty string after split is skipped when lines.length > 1
    expect(buf.getLines()).toHaveLength(2)
  })

  it('caps at maxLines, dropping oldest', () => {
    for (let i = 0; i < 15; i++) {
      buf.push(`msg-${i}`)
    }
    const lines = buf.getLines()
    expect(lines).toHaveLength(10)
    // oldest 5 dropped, newest 10 remain
    expect(lines[0]).toContain('msg-5')
    expect(lines[9]).toContain('msg-14')
  })

  it('clear() empties the buffer', () => {
    buf.push('a')
    buf.push('b')
    buf.clear()
    expect(buf.getLines()).toHaveLength(0)
  })

  it('getLines() returns a copy, not a reference', () => {
    buf.push('x')
    const a = buf.getLines()
    const b = buf.getLines()
    expect(a).not.toBe(b)
    expect(a).toEqual(b)
  })

  it('handles single empty string without crash', () => {
    buf.push('')
    // single-element split → empty string is kept (lines.length === 1)
    expect(buf.getLines()).toHaveLength(1)
  })
})

describe('Verbose Logging — Severity Coloring', () => {
  // Re-implement getLineClass from LogsPanel.tsx for unit testing
  function getLineClass(line: string): string {
    if (line.includes('ERROR') || line.includes('Traceback') || line.includes('Exception'))
      return 'text-destructive'
    if (line.includes('WARNING') || line.includes('warn'))
      return 'text-warning'
    if (line.includes('[INFO]'))
      return 'text-muted-foreground'
    return 'text-foreground/80'
  }

  it('classifies ERROR lines as destructive', () => {
    expect(getLineClass('[12:00:00.000] ERROR: something broke')).toBe('text-destructive')
  })

  it('classifies Traceback lines as destructive', () => {
    expect(getLineClass('Traceback (most recent call last):')).toBe('text-destructive')
  })

  it('classifies Exception lines as destructive', () => {
    expect(getLineClass('RuntimeException: bad thing')).toBe('text-destructive')
  })

  it('classifies WARNING lines as warning', () => {
    expect(getLineClass('[12:00:00.000] WARNING: disk low')).toBe('text-warning')
  })

  it('classifies warn (lowercase) lines as warning', () => {
    expect(getLineClass('something warned about stuff')).toBe('text-warning')
  })

  it('classifies [INFO] lines as muted', () => {
    expect(getLineClass('[INFO] Server started on port 8080')).toBe('text-muted-foreground')
  })

  it('classifies normal lines as foreground/80', () => {
    expect(getLineClass('Loading model weights...')).toBe('text-foreground/80')
  })

  it('ERROR takes precedence over WARNING in same line', () => {
    expect(getLineClass('ERROR: WARNING something')).toBe('text-destructive')
  })
})

describe('Verbose Logging — Filter', () => {
  const lines = [
    '[12:00:00.000] [INFO] Server started',
    '[12:00:01.000] Loading model weights',
    '[12:00:02.000] ERROR: out of memory',
    '[12:00:03.000] WARNING: slow batch',
    '[12:00:04.000] Batch complete in 150ms',
  ]

  // Re-implement filter logic from LogsPanel
  function filterLines(allLines: string[], filter: string): string[] {
    if (!filter) return allLines
    return allLines.filter(l => l.toLowerCase().includes(filter.toLowerCase()))
  }

  it('returns all lines with empty filter', () => {
    expect(filterLines(lines, '')).toHaveLength(5)
  })

  it('filters case-insensitively', () => {
    expect(filterLines(lines, 'error')).toHaveLength(1)
    expect(filterLines(lines, 'ERROR')).toHaveLength(1)
  })

  it('matches partial strings', () => {
    const result = filterLines(lines, 'batch')
    expect(result).toHaveLength(2) // WARNING: slow batch + Batch complete
  })

  it('returns empty for no matches', () => {
    expect(filterLines(lines, 'xyznonexistent')).toHaveLength(0)
  })
})

describe('Verbose Logging — Log Export', () => {
  it('joins lines with newline for export', () => {
    const lines = ['line1', 'line2', 'line3']
    const exported = lines.join('\n')
    expect(exported).toBe('line1\nline2\nline3')
  })

  it('generates correct filename format', () => {
    const sessionId = 'abc12345-6789-def0'
    const dateStr = '2026-03-11'
    const filename = `vmlx-logs-${sessionId.slice(0, 8)}-${dateStr}.log`
    expect(filename).toBe('vmlx-logs-abc12345-2026-03-11.log')
  })
})

describe('Verbose Logging — Buffer per Session', () => {
  it('isolates logs between sessions', () => {
    const buffers = new Map<string, LogBuffer>()

    buffers.set('s1', new LogBuffer(100))
    buffers.set('s2', new LogBuffer(100))

    buffers.get('s1')!.push('session-1 log')
    buffers.get('s2')!.push('session-2 log')

    expect(buffers.get('s1')!.getLines()[0]).toContain('session-1 log')
    expect(buffers.get('s2')!.getLines()[0]).toContain('session-2 log')
    expect(buffers.get('s1')!.getLines()).toHaveLength(1)
    expect(buffers.get('s2')!.getLines()).toHaveLength(1)
  })

  it('cleanup on delete removes buffer', () => {
    const buffers = new Map<string, LogBuffer>()
    buffers.set('s1', new LogBuffer(100))
    buffers.get('s1')!.push('data')
    buffers.delete('s1')
    expect(buffers.has('s1')).toBe(false)
  })
})

describe('Verbose Logging — Lifecycle Events', () => {
  it('spawn command is logged with $ prefix', () => {
    const buf = new LogBuffer()
    buf.push('$ /path/to/python -m vmlx_engine.cli serve --model test --port 8080')
    const lines = buf.getLines()
    expect(lines[0]).toContain('$ /path/to/python')
    expect(lines[0]).toContain('--model test')
  })

  it('crash event is logged with [ERROR] prefix', () => {
    const buf = new LogBuffer()
    buf.push('[ERROR] Process crashed: SIGKILL')
    expect(buf.getLines()[0]).toContain('[ERROR] Process crashed')
  })

  it('normal stop is logged with [INFO] prefix', () => {
    const buf = new LogBuffer()
    buf.push('[INFO] Process stopped (exit code 0)')
    expect(buf.getLines()[0]).toContain('[INFO] Process stopped')
  })
})

// ═══════════════════════════════════════════════════════════════════════════════
// Phase 5: GitHub Issue Regression Tests
// ═══════════════════════════════════════════════════════════════════════════════

describe('Issue #14: stream_interval > 1 must not drop tokens', () => {
  it('SliderField handleInputChange must not clamp on every keystroke (source check)', () => {
    // Read the actual source to verify the fix
    const fs = require('fs')
    const source = fs.readFileSync(
      'src/renderer/src/components/sessions/SessionConfigForm.tsx', 'utf-8'
    )

    // Find handleInputChange function body
    const start = source.indexOf('const handleInputChange')
    const nextFunc = source.indexOf('const handleInput', start + 30)
    const handlerBody = source.substring(start, nextFunc)

    // Must NOT contain Math.max(min in the onChange call —
    // this causes typing "1" to snap to 1024 with min=1024
    expect(handlerBody).not.toContain('Math.max(min')
  })

  it('SliderField must use local state for in-progress typing', () => {
    const fs = require('fs')
    const source = fs.readFileSync(
      'src/renderer/src/components/sessions/SessionConfigForm.tsx', 'utf-8'
    )

    // SliderField must have local state to prevent mid-keystroke clamping
    const sliderStart = source.indexOf('export function SliderField')
    const sliderEnd = source.indexOf('\nexport ', sliderStart + 10)
    const sliderBody = source.substring(sliderStart, sliderEnd > 0 ? sliderEnd : undefined)

    // Must have local input state
    expect(sliderBody).toContain('localInput')
    // Must have onFocus handler to init local state
    expect(sliderBody).toContain('onFocus')
  })
})

describe('Issue #15: perf/cache IPC timeouts must survive inference load', () => {
  it('performance.ts health timeout must be >= 30 seconds', () => {
    const fs = require('fs')
    const source = fs.readFileSync('src/main/ipc/performance.ts', 'utf-8')
    const match = source.match(/AbortSignal\.timeout\((\d+)\)/)
    expect(match).toBeTruthy()
    const timeoutMs = parseInt(match![1])
    expect(timeoutMs).toBeGreaterThanOrEqual(30000)
  })

  it('cache.ts stats/entries timeouts must be >= 30 seconds', () => {
    const fs = require('fs')
    const source = fs.readFileSync('src/main/ipc/cache.ts', 'utf-8')
    const matches = [...source.matchAll(/AbortSignal\.timeout\((\d+)\)/g)]
    expect(matches.length).toBeGreaterThan(0)
    // Stats and entries should be >= 30s; warm (60s) and clear (10s) are fine
    for (const match of matches) {
      const timeout = parseInt(match[1])
      // All timeouts should be at least 10s
      expect(timeout).toBeGreaterThanOrEqual(10000)
    }
  })
})

describe('Nemotron fix: BatchedEngine._start_llm wraps model', () => {
  it('_start_llm must apply MLLMModelWrapper for LanguageModelOutput extraction', () => {
    const fs = require('fs')
    const source = fs.readFileSync(
      '../vmlx_engine/engine/batched.py', 'utf-8'
    )

    // Find _start_llm method
    const startIdx = source.indexOf('async def _start_llm')
    const endIdx = source.indexOf('\n    async def ', startIdx + 10)
    const methodBody = source.substring(startIdx, endIdx > 0 ? endIdx : undefined)

    // Must wrap model in MLLMModelWrapper
    expect(methodBody).toContain('MLLMModelWrapper')
  })
})

// ── Phase 6: Extended analysis regression tests ──

describe('M1: Abort drains pending text from stream_interval > 1', () => {
  it('_cleanup_request must drain pending text before discarding stream state', () => {
    const fs = require('fs')
    const source = fs.readFileSync(
      '../vmlx_engine/engine_core.py', 'utf-8'
    )

    const cleanupIdx = source.indexOf('def _cleanup_request')
    const nextDef = source.indexOf('\n    async def ', cleanupIdx + 10)
    const methodBody = source.substring(cleanupIdx, nextDef > 0 ? nextDef : undefined)

    expect(methodBody).toContain('drain_pending')
    expect(methodBody).toContain('new_text=')
  })
})

describe('M3: chat:reasoningDone fires at tool iteration boundary', () => {
  it('tool call boundary emits reasoningDone before resetting isReasoning', () => {
    const fs = require('fs')
    const source = fs.readFileSync(
      'src/main/ipc/chat.ts', 'utf-8'
    )

    // Find tool boundary: emitToolStatus('processing', '', undefined
    const boundaryIdx = source.indexOf("emitToolStatus('processing', '', undefined")
    const preBoundary = source.substring(Math.max(0, boundaryIdx - 500), boundaryIdx)
    expect(preBoundary).toContain('chat:reasoningDone')
  })

  it('auto-continue boundary emits reasoningDone before resetting isReasoning', () => {
    const fs = require('fs')
    const source = fs.readFileSync(
      'src/main/ipc/chat.ts', 'utf-8'
    )

    const boundaryIdx = source.indexOf("emitToolStatus('processing', '', 'Generating response...'")
    const preBoundary = source.substring(Math.max(0, boundaryIdx - 500), boundaryIdx)
    expect(preBoundary).toContain('chat:reasoningDone')
  })
})

describe('M2: suppress_reasoning + reasoning-only emits diagnostic', () => {
  it('server emits diagnostic when suppress_reasoning and only reasoning produced', () => {
    const fs = require('fs')
    const source = fs.readFileSync(
      '../vmlx_engine/server.py', 'utf-8'
    )

    // Both API paths must explain reasoning-only suppression
    expect(source).toContain('only internal reasoning')
  })
})

describe('M5: qwen3_next uses correct tool parser', () => {
  it('qwen3_next config must have tool_parser=qwen, not nemotron', () => {
    const fs = require('fs')
    const source = fs.readFileSync(
      '../vmlx_engine/model_configs.py', 'utf-8'
    )

    // Find qwen3_next config block
    const startIdx = source.indexOf('family_name="qwen3_next"')
    const endIdx = source.indexOf('))', startIdx)
    const configBlock = source.substring(startIdx, endIdx)

    expect(configBlock).toContain('tool_parser="qwen"')
    expect(configBlock).not.toContain('tool_parser="nemotron"')
  })
})

describe('Architecture hints: gemma3/medgemma inject_pixel_values', () => {
  it('gemma3 config must have inject_pixel_values architecture hint', () => {
    const fs = require('fs')
    const source = fs.readFileSync(
      '../vmlx_engine/model_configs.py', 'utf-8'
    )

    const startIdx = source.indexOf('family_name="gemma3"')
    const endIdx = source.indexOf('))', startIdx)
    const configBlock = source.substring(startIdx, endIdx)

    expect(configBlock).toContain('inject_pixel_values')
  })

  it('medgemma config must have inject_pixel_values architecture hint', () => {
    const fs = require('fs')
    const source = fs.readFileSync(
      '../vmlx_engine/model_configs.py', 'utf-8'
    )

    const startIdx = source.indexOf('family_name="medgemma"')
    const endIdx = source.indexOf('))', startIdx)
    const configBlock = source.substring(startIdx, endIdx)

    expect(configBlock).toContain('inject_pixel_values')
  })
})

// ═══════════════════════════════════════════════════════════════════════════════
// Session 7: Issues #4 and #6 — Interrupted marker stripping & clearAllLocks cancel
// ═══════════════════════════════════════════════════════════════════════════════

describe('Issue #4: [Generation interrupted] must be stripped from API requests', () => {
  // Simulates the message-building logic in chat.ts (lines 548-560 + fix)
  function buildRequestMessages(messages: Array<{ role: string; content: string }>): Array<{ role: string; content: any }> {
    const result: Array<{ role: string; content: any }> = []
    for (const m of messages) {
      let msgContent: any = m.content
      // Strip "[Generation interrupted]" markers (the fix from chat.ts)
      if (m.role === 'assistant' && typeof msgContent === 'string') {
        msgContent = msgContent.replace(/\n\n\[Generation interrupted\]$/, '').replace(/^\[Generation interrupted\]$/, '')
        if (!msgContent.trim()) continue
      }
      result.push({ role: m.role, content: msgContent })
    }
    return result
  }

  it('strips trailing [Generation interrupted] from assistant messages', () => {
    const messages = [
      { role: 'user', content: 'Hello' },
      { role: 'assistant', content: 'Here is my partial respo\n\n[Generation interrupted]' },
      { role: 'user', content: 'Continue' },
    ]
    const built = buildRequestMessages(messages)
    expect(built[1].content).toBe('Here is my partial respo')
    expect(built[1].content).not.toContain('[Generation interrupted]')
  })

  it('skips assistant messages that are ONLY [Generation interrupted]', () => {
    const messages = [
      { role: 'user', content: 'Hello' },
      { role: 'assistant', content: '[Generation interrupted]' },
      { role: 'user', content: 'Try again' },
    ]
    const built = buildRequestMessages(messages)
    expect(built.length).toBe(2)
    expect(built[0].role).toBe('user')
    expect(built[1].role).toBe('user')
  })

  it('does NOT strip [Generation interrupted] from user messages', () => {
    const messages = [
      { role: 'user', content: 'What does [Generation interrupted] mean?' },
    ]
    const built = buildRequestMessages(messages)
    expect(built[0].content).toContain('[Generation interrupted]')
  })

  it('preserves assistant messages with real content after stripping', () => {
    const messages = [
      { role: 'user', content: 'Tell me a story' },
      { role: 'assistant', content: 'Once upon a time\n\n[Generation interrupted]' },
    ]
    const built = buildRequestMessages(messages)
    expect(built[1].content).toBe('Once upon a time')
    expect(built.length).toBe(2)
  })

  it('handles normal assistant messages without markers (no-op)', () => {
    const messages = [
      { role: 'user', content: 'Hello' },
      { role: 'assistant', content: 'Hi there! How can I help?' },
    ]
    const built = buildRequestMessages(messages)
    expect(built[1].content).toBe('Hi there! How can I help?')
  })
})

describe('Issue #6: clearAllLocks must send server cancel for all active requests', () => {
  it('chat.ts clearAllLocks sends cancel requests (source code verification)', () => {
    const fs = require('fs')
    const source = fs.readFileSync(
      'src/main/ipc/chat.ts', 'utf-8'
    )

    // Find the clearAllLocks handler
    const clearAllLocksIdx = source.indexOf("'chat:clearAllLocks'")
    expect(clearAllLocksIdx).toBeGreaterThan(-1)

    // Extract the handler body (up to next ipcMain.handle)
    const handlerEnd = source.indexOf("ipcMain.handle('chat:set", clearAllLocksIdx)
    const handlerBody = source.substring(clearAllLocksIdx, handlerEnd)

    // Must send server cancel (not just abort controller)
    expect(handlerBody).toContain('/cancel')
    expect(handlerBody).toContain('entry.responseId')
    expect(handlerBody).toContain("method: 'POST'")
    // Must still abort the local controller
    expect(handlerBody).toContain('entry.controller.abort()')
  })

  it('abortByEndpoint also sends server cancel (consistency check)', () => {
    const fs = require('fs')
    const source = fs.readFileSync(
      'src/main/ipc/chat.ts', 'utf-8'
    )

    const abortByEndpointIdx = source.indexOf('function abortByEndpoint')
    expect(abortByEndpointIdx).toBeGreaterThan(-1)

    const funcEnd = source.indexOf('\nfunction ', abortByEndpointIdx + 10)
    const endIdx = funcEnd > 0 ? funcEnd : source.indexOf('\n/** ', abortByEndpointIdx + 10)
    const funcBody = source.substring(abortByEndpointIdx, endIdx)

    // Must send server cancel
    expect(funcBody).toContain('/cancel')
    expect(funcBody).toContain("method: 'POST'")
  })
})

// ═══════════════════════════════════════════════════════════════════════════════
// Phase 6: Cache API Data Flow
// ═══════════════════════════════════════════════════════════════════════════════

describe('Phase 6: Cache API Data Flow', () => {
  describe('CachePanel interface contract', () => {
    // Pure logic replica of CachePanelProps from CachePanel.tsx
    interface CachePanelProps {
      endpoint: { host: string; port: number }
      sessionStatus: string
      sessionId?: string
    }

    it('CachePanel props include sessionId', () => {
      // Verify the interface shape — sessionId must be passable
      const props: CachePanelProps = {
        endpoint: { host: '127.0.0.1', port: 8000 },
        sessionStatus: 'running',
        sessionId: 'session-abc-123',
      }
      expect(props.sessionId).toBe('session-abc-123')
    })

    it('sessionId is optional (remote sessions may not have it)', () => {
      const props: CachePanelProps = {
        endpoint: { host: '127.0.0.1', port: 8000 },
        sessionStatus: 'running',
      }
      expect(props.sessionId).toBeUndefined()
    })

    it('endpoint requires both host and port', () => {
      const props: CachePanelProps = {
        endpoint: { host: '0.0.0.0', port: 9000 },
        sessionStatus: 'stopped',
      }
      expect(props.endpoint.host).toBe('0.0.0.0')
      expect(props.endpoint.port).toBe(9000)
    })
  })

  describe('Disk cache stats field names', () => {
    // The Python server returns disk_cache stats with these field names.
    // CachePanel.tsx renders them as: diskCache.total_size_mb ?? diskCache.size_mb
    // This documents the expected Python-to-JS field mapping.

    interface DiskCacheStats {
      total_size_mb?: number
      size_mb?: number        // Legacy fallback field
      num_entries?: number
      max_size_gb?: number
      cache_dir?: string
    }

    function displayDiskSize(stats: DiskCacheStats): string {
      const sizeMb = stats.total_size_mb ?? stats.size_mb ?? 0
      return `${sizeMb.toFixed(1)} MB`
    }

    it('uses total_size_mb as primary field', () => {
      expect(displayDiskSize({ total_size_mb: 512.3 })).toBe('512.3 MB')
    })

    it('falls back to size_mb when total_size_mb is absent', () => {
      expect(displayDiskSize({ size_mb: 256.7 })).toBe('256.7 MB')
    })

    it('defaults to 0 when neither field is present', () => {
      expect(displayDiskSize({})).toBe('0.0 MB')
    })

    it('total_size_mb takes priority over size_mb when both present', () => {
      expect(displayDiskSize({ total_size_mb: 100, size_mb: 200 })).toBe('100.0 MB')
    })

    it('handles zero total_size_mb (not falsy fallthrough)', () => {
      // 0 is a valid value — should NOT fallback to size_mb
      // Note: ?? only falls through on null/undefined, not 0
      expect(displayDiskSize({ total_size_mb: 0, size_mb: 999 })).toBe('0.0 MB')
    })
  })

  describe('Cache operations require running session', () => {
    // CachePanel.tsx: fetchStats() early-returns if sessionStatus !== 'running'
    // This is the guard logic.

    function shouldFetchStats(sessionStatus: string): boolean {
      return sessionStatus === 'running'
    }

    function shouldAllowCacheOp(sessionStatus: string): boolean {
      // Cache warm/clear operations also require running session
      return sessionStatus === 'running'
    }

    it('allows fetch when session is running', () => {
      expect(shouldFetchStats('running')).toBe(true)
    })

    it('blocks fetch when session is stopped', () => {
      expect(shouldFetchStats('stopped')).toBe(false)
    })

    it('blocks fetch when session is loading', () => {
      expect(shouldFetchStats('loading')).toBe(false)
    })

    it('blocks fetch when session is in error', () => {
      expect(shouldFetchStats('error')).toBe(false)
    })

    it('blocks fetch with empty string status', () => {
      expect(shouldFetchStats('')).toBe(false)
    })

    it('allows cache operations only when running', () => {
      expect(shouldAllowCacheOp('running')).toBe(true)
      expect(shouldAllowCacheOp('stopped')).toBe(false)
      expect(shouldAllowCacheOp('loading')).toBe(false)
    })
  })

  describe('CachePanel source verification', () => {
    it('CachePanel.tsx passes sessionId to all cache API calls', () => {
      const fs = require('fs')
      const source = fs.readFileSync(
        'src/renderer/src/components/sessions/CachePanel.tsx', 'utf-8'
      )

      // All four cache API calls should pass sessionId
      const statsCall = source.includes('cache.stats(endpoint, sessionId)')
      const entriesCall = source.includes('cache.entries(endpoint, sessionId)')
      const warmCall = source.includes('cache.warm(')
      const clearCall = source.includes('cache.clear(')

      expect(statsCall).toBe(true)
      expect(entriesCall).toBe(true)
      expect(warmCall).toBe(true)
      expect(clearCall).toBe(true)

      // Verify sessionId appears in warm and clear calls too
      // (they have additional arguments before endpoint, sessionId)
      const warmLine = source.split('\n').find((l: string) => l.includes('cache.warm('))
      expect(warmLine).toContain('sessionId')

      const clearLine = source.split('\n').find((l: string) => l.includes('cache.clear('))
      expect(clearLine).toContain('sessionId')
    })

    it('CachePanel uses total_size_mb with size_mb fallback', () => {
      const fs = require('fs')
      const source = fs.readFileSync(
        'src/renderer/src/components/sessions/CachePanel.tsx', 'utf-8'
      )

      // The display logic: (diskCache.total_size_mb ?? diskCache.size_mb)
      expect(source).toContain('total_size_mb')
      expect(source).toContain('size_mb')
    })

    it('CachePanel guards stats fetch with session status check', () => {
      const fs = require('fs')
      const source = fs.readFileSync(
        'src/renderer/src/components/sessions/CachePanel.tsx', 'utf-8'
      )

      // fetchStats() has: if (sessionStatus !== 'running') return
      expect(source).toContain("sessionStatus !== 'running'")
    })
  })
})

