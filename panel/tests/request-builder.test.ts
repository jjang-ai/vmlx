/**
 * Tests for chat request body building — verifies that all sampling parameters
 * (temperature, top_p, top_k, min_p, repeat_penalty, stop, max_tokens, etc.)
 * are correctly forwarded for both Chat Completions and Responses API wire formats,
 * and that remote-only gating (chat_template_kwargs exclusion) works correctly.
 */
import { describe, it, expect } from 'vitest'

// ─── buildRequestBody logic (extracted from chat.ts) ─────────────────────────

interface ChatOverrides {
    temperature?: number
    topP?: number
    topK?: number
    minP?: number
    maxTokens?: number
    repeatPenalty?: number
    systemPrompt?: string
    stopSequences?: string
    wireApi?: 'completions' | 'responses'
    builtinToolsEnabled?: boolean
    enableThinking?: boolean
    reasoningEffort?: string
}

function buildRequestBody(
    wireApi: 'completions' | 'responses',
    modelName: string,
    requestMessages: any[],
    overrides: ChatOverrides | undefined,
    isRemote: boolean,
    sessionHasReasoningParser: boolean,
    tools?: any[]
): Record<string, any> {
    const stopSequences = overrides?.stopSequences
        ? overrides.stopSequences.split(',').map(s => s.trim()).filter(Boolean)
        : undefined

    if (wireApi === 'responses') {
        const systemMessages = requestMessages.filter(m => m.role === 'system')
        const instructions = overrides?.systemPrompt || (systemMessages.length > 0 ? systemMessages.map(m => m.content).join('\n') : undefined)
        const inputMessages = requestMessages.filter(m => m.role !== 'system')
        const obj: Record<string, any> = {
            model: modelName,
            input: inputMessages,
            instructions,
            temperature: overrides?.temperature ?? 0.7,
            top_p: overrides?.topP ?? 0.9,
            max_output_tokens: overrides?.maxTokens ?? 4096,
            stream: true,
            stream_options: { include_usage: true }
        }
        if (stopSequences) obj.stop = stopSequences
        if (overrides?.topK != null && overrides.topK > 0) obj.top_k = overrides.topK
        if (overrides?.minP != null && overrides.minP > 0) obj.min_p = overrides.minP
        if (overrides?.repeatPenalty != null && overrides.repeatPenalty !== 1.0) obj.repetition_penalty = overrides.repeatPenalty
        if (tools) {
            obj.tools = tools.map(t => ({
                type: 'function',
                name: t.function.name,
                description: t.function.description,
                parameters: t.function.parameters
            }))
        }
        if (overrides?.enableThinking !== undefined) {
            obj.enable_thinking = overrides.enableThinking
        } else if (isRemote) {
            obj.enable_thinking = sessionHasReasoningParser
        }
        if (!isRemote && obj.enable_thinking !== undefined) obj.chat_template_kwargs = { enable_thinking: obj.enable_thinking }
        if (overrides?.reasoningEffort) obj.reasoning_effort = overrides.reasoningEffort
        return obj
    } else {
        const obj: Record<string, any> = {
            model: modelName,
            messages: requestMessages,
            temperature: overrides?.temperature ?? 0.7,
            top_p: overrides?.topP ?? 0.9,
            max_tokens: overrides?.maxTokens ?? 4096,
            stream: true,
            stream_options: { include_usage: true }
        }
        if (stopSequences) obj.stop = stopSequences
        if (overrides?.topK != null && overrides.topK > 0) obj.top_k = overrides.topK
        if (overrides?.minP != null && overrides.minP > 0) obj.min_p = overrides.minP
        if (overrides?.repeatPenalty != null && overrides.repeatPenalty !== 1.0) obj.repetition_penalty = overrides.repeatPenalty
        if (tools) {
            obj.tools = tools
        }
        if (overrides?.enableThinking !== undefined) {
            obj.enable_thinking = overrides.enableThinking
        } else if (isRemote) {
            obj.enable_thinking = sessionHasReasoningParser
        }
        if (!isRemote && obj.enable_thinking !== undefined) obj.chat_template_kwargs = { enable_thinking: obj.enable_thinking }
        if (overrides?.reasoningEffort) obj.reasoning_effort = overrides.reasoningEffort
        return obj
    }
}

// ─── Tests ───────────────────────────────────────────────────────────────────

describe('buildRequestBody — Chat Completions API', () => {
    const messages = [
        { role: 'system', content: 'You are helpful.' },
        { role: 'user', content: 'Hello' }
    ]

    it('includes all basic parameters with defaults', () => {
        const body = buildRequestBody('completions', 'gpt-4', messages, undefined, false, false)
        expect(body.model).toBe('gpt-4')
        expect(body.messages).toBe(messages)
        expect(body.temperature).toBe(0.7)
        expect(body.top_p).toBe(0.9)
        expect(body.max_tokens).toBe(4096)
        expect(body.stream).toBe(true)
        expect(body.stream_options).toEqual({ include_usage: true })
    })

    it('forwards custom temperature and top_p', () => {
        const body = buildRequestBody('completions', 'gpt-4', messages, { temperature: 0.3, topP: 0.5 }, false, false)
        expect(body.temperature).toBe(0.3)
        expect(body.top_p).toBe(0.5)
    })

    it('includes top_k when > 0', () => {
        const body = buildRequestBody('completions', 'gpt-4', messages, { topK: 40 }, false, false)
        expect(body.top_k).toBe(40)
    })

    it('omits top_k when 0 or undefined', () => {
        const body = buildRequestBody('completions', 'gpt-4', messages, { topK: 0 }, false, false)
        expect(body.top_k).toBeUndefined()
    })

    it('includes min_p when > 0', () => {
        const body = buildRequestBody('completions', 'gpt-4', messages, { minP: 0.05 }, false, false)
        expect(body.min_p).toBe(0.05)
    })

    it('includes repetition_penalty when != 1.0', () => {
        const body = buildRequestBody('completions', 'gpt-4', messages, { repeatPenalty: 1.2 }, false, false)
        expect(body.repetition_penalty).toBe(1.2)
    })

    it('omits repetition_penalty when exactly 1.0', () => {
        const body = buildRequestBody('completions', 'gpt-4', messages, { repeatPenalty: 1.0 }, false, false)
        expect(body.repetition_penalty).toBeUndefined()
    })

    it('includes stop sequences when provided', () => {
        const body = buildRequestBody('completions', 'gpt-4', messages, { stopSequences: '<stop>, END' }, false, false)
        expect(body.stop).toEqual(['<stop>', 'END'])
    })

    it('omits stop when stopSequences is empty', () => {
        const body = buildRequestBody('completions', 'gpt-4', messages, { stopSequences: '' }, false, false)
        expect(body.stop).toBeUndefined()
    })

    it('forwards max_tokens override', () => {
        const body = buildRequestBody('completions', 'gpt-4', messages, { maxTokens: 8192 }, false, false)
        expect(body.max_tokens).toBe(8192)
    })

    it('forwards reasoning_effort', () => {
        const body = buildRequestBody('completions', 'gpt-4', messages, { reasoningEffort: 'high' }, false, false)
        expect(body.reasoning_effort).toBe('high')
    })
})

describe('buildRequestBody — Remote vs Local gating', () => {
    const messages = [{ role: 'user', content: 'Hello' }]

    it('omits enable_thinking and chat_template_kwargs for local Auto sessions', () => {
        const body = buildRequestBody('completions', 'model', messages, undefined, false, false)
        expect(body.enable_thinking).toBeUndefined()
        expect(body.chat_template_kwargs).toBeUndefined()
    })

    it('includes chat_template_kwargs for explicit local thinking override', () => {
        const body = buildRequestBody('completions', 'model', messages, { enableThinking: true }, false, false)
        expect(body.enable_thinking).toBe(true)
        expect(body.chat_template_kwargs).toEqual({ enable_thinking: true })
    })

    it('EXCLUDES chat_template_kwargs for remote sessions', () => {
        const body = buildRequestBody('completions', 'model', messages, undefined, true, false)
        expect(body.chat_template_kwargs).toBeUndefined()
    })

    it('enable_thinking defaults to true when session has reasoning parser', () => {
        const body = buildRequestBody('completions', 'model', messages, undefined, true, true)
        expect(body.enable_thinking).toBe(true)
    })

    it('enable_thinking can be explicitly set via overrides', () => {
        const body = buildRequestBody('completions', 'model', messages, { enableThinking: false }, true, true)
        expect(body.enable_thinking).toBe(false)
    })
})

describe('buildRequestBody — Responses API', () => {
    const messages = [
        { role: 'system', content: 'You are helpful.' },
        { role: 'user', content: 'Hello' }
    ]

    it('uses max_output_tokens instead of max_tokens', () => {
        const body = buildRequestBody('responses', 'gpt-4', messages, undefined, false, false)
        expect(body.max_output_tokens).toBe(4096)
        expect(body.max_tokens).toBeUndefined()
    })

    it('uses input instead of messages', () => {
        const body = buildRequestBody('responses', 'gpt-4', messages, undefined, false, false)
        expect(body.input).toEqual([{ role: 'user', content: 'Hello' }])
        expect(body.messages).toBeUndefined()
    })

    it('extracts system messages as instructions', () => {
        const body = buildRequestBody('responses', 'gpt-4', messages, undefined, false, false)
        expect(body.instructions).toBe('You are helpful.')
    })

    it('prefers systemPrompt override for instructions', () => {
        const body = buildRequestBody('responses', 'gpt-4', messages, { systemPrompt: 'Custom system prompt' }, false, false)
        expect(body.instructions).toBe('Custom system prompt')
    })

    it('EXCLUDES chat_template_kwargs for remote', () => {
        const body = buildRequestBody('responses', 'gpt-4', messages, undefined, true, false)
        expect(body.chat_template_kwargs).toBeUndefined()
    })
})

describe('buildRequestBody — Tool format', () => {
    const messages = [{ role: 'user', content: 'Hello' }]
    const sampleTools = [
        {
            type: 'function',
            function: {
                name: 'read_file',
                description: 'Read a file',
                parameters: { type: 'object', properties: { path: { type: 'string' } } }
            }
        }
    ]

    it('Completions API: tools use OpenAI format with function wrapper', () => {
        const body = buildRequestBody('completions', 'model', messages, undefined, false, false, sampleTools)
        expect(body.tools).toEqual(sampleTools)
        expect(body.tools[0].function).toBeDefined()
        expect(body.tools[0].function.name).toBe('read_file')
    })

    it('Responses API: tools use flat format WITHOUT function wrapper', () => {
        const body = buildRequestBody('responses', 'model', messages, undefined, false, false, sampleTools)
        expect(body.tools[0].name).toBe('read_file')
        expect(body.tools[0].function).toBeUndefined()
    })
})

// ─── filterTools logic ───────────────────────────────────────────────────────

describe('filterTools', () => {
    const FILE_TOOLS = new Set(['read_file', 'write_file', 'edit_file', 'patch_file', 'batch_edit', 'copy_file', 'move_file', 'delete_file', 'create_directory', 'list_directory', 'insert_text', 'replace_lines', 'apply_regex', 'read_image'])
    const SEARCH_TOOLS = new Set(['search_files', 'find_files', 'file_info', 'get_diagnostics', 'get_tree', 'diff_files'])
    const SHELL_TOOLS = new Set(['run_command', 'spawn_process', 'get_process_output'])

    function filterTools(allTools: any[], overrides: any): any[] {
        const disabled = new Set<string>()
        if (overrides.fileToolsEnabled === false) FILE_TOOLS.forEach(t => disabled.add(t))
        if (overrides.searchToolsEnabled === false) SEARCH_TOOLS.forEach(t => disabled.add(t))
        if (overrides.shellEnabled === false) SHELL_TOOLS.forEach(t => disabled.add(t))
        if (disabled.size === 0) return allTools
        return allTools.filter((t: any) => !disabled.has(t.function.name))
    }

    const allTools = [
        { function: { name: 'read_file' } },
        { function: { name: 'search_files' } },
        { function: { name: 'run_command' } },
        { function: { name: 'ask_user' } },
    ]

    it('returns all tools when no toggles disabled', () => {
        expect(filterTools(allTools, {})).toEqual(allTools)
    })

    it('disables file tools when fileToolsEnabled=false', () => {
        const result = filterTools(allTools, { fileToolsEnabled: false })
        expect(result.find(t => t.function.name === 'read_file')).toBeUndefined()
        expect(result.find(t => t.function.name === 'search_files')).toBeDefined()
    })

    it('disables shell tools when shellEnabled=false', () => {
        const result = filterTools(allTools, { shellEnabled: false })
        expect(result.find(t => t.function.name === 'run_command')).toBeUndefined()
        expect(result.find(t => t.function.name === 'ask_user')).toBeDefined()
    })

    it('ask_user is never disabled by any toggle', () => {
        const result = filterTools(allTools, {
            fileToolsEnabled: false,
            searchToolsEnabled: false,
            shellEnabled: false
        })
        expect(result.find(t => t.function.name === 'ask_user')).toBeDefined()
    })
})
