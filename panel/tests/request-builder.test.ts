/**
 * Tests for chat request body building — verifies that all sampling parameters
 * (temperature, top_p, top_k, min_p, repeat_penalty, stop, max_tokens, etc.)
 * are correctly forwarded for both Chat Completions and Responses API wire formats,
 * and that remote-only gating (chat_template_kwargs exclusion) works correctly.
 */
import { describe, it, expect } from 'vitest'
import { readFileSync } from 'node:fs'
import { dsv4OutputBudget } from '../src/shared/dsv4RequestBudget'

// ─── buildRequestBody logic (extracted from chat.ts) ─────────────────────────

interface ChatOverrides {
    temperature?: number
    topP?: number
    topK?: number
    minP?: number
    maxTokens?: number
    maxThinkingTokens?: number
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
    tools?: any[],
    detectedFamily?: string,
    thinkingBudgetSupported?: boolean,
): Record<string, any> {
    const stopSequences = overrides?.stopSequences
        ? overrides.stopSequences.split(',').map(s => s.trim()).filter(Boolean)
        : undefined
    const shouldForwardReasoningEffort =
        !!overrides?.reasoningEffort &&
        overrides.enableThinking !== false &&
        (detectedFamily !== 'hy3' || overrides.enableThinking === true) &&
        (sessionHasReasoningParser || detectedFamily === 'deepseek-v4')
    const outputBudget = dsv4OutputBudget(
        overrides?.maxTokens,
        overrides?.enableThinking,
        detectedFamily,
        overrides?.reasoningEffort,
    )
    const thinkingBudget = typeof overrides?.maxThinkingTokens === 'number' &&
        Number.isFinite(overrides.maxThinkingTokens) &&
        overrides.maxThinkingTokens > 0
        ? Math.floor(overrides.maxThinkingTokens)
        : undefined
    const effectiveEnableThinkingOverride =
        !isRemote &&
        !sessionHasReasoningParser &&
        detectedFamily !== 'deepseek-v4'
            ? undefined
            : overrides?.enableThinking
    const applyLocalThinkingBudget = (obj: Record<string, any>) => {
        if (isRemote || thinkingBudget == null || obj.enable_thinking === false) return
        if (thinkingBudgetSupported === false) return
        if (!sessionHasReasoningParser && detectedFamily !== 'deepseek-v4') return
        obj.max_thinking_tokens = thinkingBudget
        obj.chat_template_kwargs = {
            ...(obj.chat_template_kwargs || {}),
            thinking_budget: thinkingBudget,
        }
    }

    if (wireApi === 'responses') {
        const systemMessages = requestMessages.filter(m => m.role === 'system')
        const instructions =
            overrides?.builtinToolsEnabled && systemMessages.length > 0
                ? systemMessages.map(m => m.content).join('\n')
                : overrides?.systemPrompt || (systemMessages.length > 0 ? systemMessages.map(m => m.content).join('\n') : undefined)
        const inputMessages = requestMessages.filter(m => m.role !== 'system')
        const obj: Record<string, any> = {
            model: modelName,
            input: inputMessages,
            instructions,
            stream: true,
            stream_options: { include_usage: true }
        }
        if (overrides?.temperature != null) obj.temperature = overrides.temperature
        if (overrides?.topP != null) obj.top_p = overrides.topP
        if (outputBudget) obj.max_output_tokens = outputBudget
        if (stopSequences) obj.stop = stopSequences
        const effectiveTopK = overrides?.topK
        if (effectiveTopK != null && effectiveTopK > 0) obj.top_k = effectiveTopK
        if (overrides?.minP != null && overrides.minP > 0) obj.min_p = overrides.minP
        if (overrides?.repeatPenalty != null) obj.repetition_penalty = overrides.repeatPenalty
        if (tools) {
            obj.tools = tools.map(t => ({
                type: 'function',
                name: t.function.name,
                description: t.function.description,
                parameters: t.function.parameters
            }))
        }
        if (effectiveEnableThinkingOverride !== undefined) {
            obj.enable_thinking = effectiveEnableThinkingOverride
        } else if (isRemote) {
            obj.enable_thinking = sessionHasReasoningParser
        }
        if (!isRemote && obj.enable_thinking !== undefined) obj.chat_template_kwargs = { enable_thinking: obj.enable_thinking }
        applyLocalThinkingBudget(obj)
        if (shouldForwardReasoningEffort) obj.reasoning_effort = overrides.reasoningEffort
        return obj
    } else {
        const obj: Record<string, any> = {
            model: modelName,
            messages: requestMessages,
            stream: true,
            stream_options: { include_usage: true }
        }
        if (overrides?.temperature != null) obj.temperature = overrides.temperature
        if (overrides?.topP != null) obj.top_p = overrides.topP
        if (outputBudget) obj.max_tokens = outputBudget
        if (stopSequences) obj.stop = stopSequences
        const effectiveTopK = overrides?.topK
        if (effectiveTopK != null && effectiveTopK > 0) obj.top_k = effectiveTopK
        if (overrides?.minP != null && overrides.minP > 0) obj.min_p = overrides.minP
        if (overrides?.repeatPenalty != null) obj.repetition_penalty = overrides.repeatPenalty
        if (tools) {
            obj.tools = tools
        }
        if (effectiveEnableThinkingOverride !== undefined) {
            obj.enable_thinking = effectiveEnableThinkingOverride
        } else if (isRemote) {
            obj.enable_thinking = sessionHasReasoningParser
        }
        if (!isRemote && obj.enable_thinking !== undefined) obj.chat_template_kwargs = { enable_thinking: obj.enable_thinking }
        applyLocalThinkingBudget(obj)
        if (shouldForwardReasoningEffort) obj.reasoning_effort = overrides.reasoningEffort
        return obj
    }
}

// ─── Tests ───────────────────────────────────────────────────────────────────

describe('buildRequestBody — Chat Completions API', () => {
    const messages = [
        { role: 'system', content: 'You are helpful.' },
        { role: 'user', content: 'Hello' }
    ]

    it('omits sampling and token defaults when unset so the engine resolves bundle metadata', () => {
        const body = buildRequestBody('completions', 'gpt-4', messages, undefined, false, false)
        expect(body.model).toBe('gpt-4')
        expect(body.messages).toBe(messages)
        expect(body.temperature).toBeUndefined()
        expect(body.top_p).toBeUndefined()
        expect(body.max_tokens).toBeUndefined()
        expect(body.stream).toBe(true)
        expect(body.stream_options).toEqual({ include_usage: true })
    })

    it('does not invent sampler or output-budget values when chat overrides are absent', () => {
        const body = buildRequestBody('completions', 'qwen-mtp', messages, undefined, false, true)
        expect(body.temperature).toBeUndefined()
        expect(body.top_p).toBeUndefined()
        expect(body.max_tokens).toBeUndefined()
        expect(body.repetition_penalty).toBeUndefined()
    })

    it('forwards explicit greedy MTP-compatible sampler values including neutral repeat penalty', () => {
        const body = buildRequestBody(
            'completions',
            'qwen-mtp',
            messages,
            { temperature: 0, topP: 1, repeatPenalty: 1.0 },
            false,
            true,
        )
        expect(body.temperature).toBe(0)
        expect(body.top_p).toBe(1)
        expect(body.repetition_penalty).toBe(1.0)
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

    it('omits top_k when explicitly 0', () => {
        const body = buildRequestBody('completions', 'gpt-4', messages, { topK: 0 }, false, false)
        expect(body.top_k).toBeUndefined()
    })

    it('omits top_k when unset so the engine uses bundle metadata or disabled fallback', () => {
        const body = buildRequestBody('completions', 'local-model', messages, {}, false, false)
        expect(body.top_k).toBeUndefined()
    })

    it('does not default remote requests to top_k', () => {
        const body = buildRequestBody('completions', 'gpt-4', messages, {}, true, false)
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

    it('forwards repetition_penalty when explicitly set to neutral 1.0', () => {
        const body = buildRequestBody('completions', 'gpt-4', messages, { repeatPenalty: 1.0 }, false, false)
        expect(body.repetition_penalty).toBe(1.0)
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

    it('keeps per-chat maxTokens as output budget only, never prompt context', () => {
        const body = buildRequestBody('completions', 'gpt-4', messages, { maxTokens: 2048 }, false, false)
        expect(body.max_tokens).toBe(2048)
        expect(body.max_output_tokens).toBeUndefined()
        expect(body.max_prompt_tokens).toBeUndefined()
        expect(body.max_context_tokens).toBeUndefined()
        expect(body.max_context).toBeUndefined()
    })

    it('per-chat maxTokens below or above the server startup default remain request scoped', () => {
        const serverStartupDefaultMaxTokens = 4096
        const below = buildRequestBody('completions', 'gpt-4', messages, { maxTokens: 512 }, false, false)
        const above = buildRequestBody('completions', 'gpt-4', messages, { maxTokens: 8192 }, false, false)

        expect(below.max_tokens).toBe(512)
        expect(above.max_tokens).toBe(8192)
        expect(below.max_tokens).toBeLessThan(serverStartupDefaultMaxTokens)
        expect(above.max_tokens).toBeGreaterThan(serverStartupDefaultMaxTokens)
        for (const body of [below, above]) {
            expect(body.max_output_tokens).toBeUndefined()
            expect(body.max_prompt_tokens).toBeUndefined()
            expect(body.max_context_tokens).toBeUndefined()
            expect(body.max_context).toBeUndefined()
        }
    })

    it('omits invalid persisted maxTokens values instead of poisoning Chat Completions', () => {
        const badValues = [0, -5, Number.NaN, Number.POSITIVE_INFINITY, '1024']
        for (const value of badValues) {
            const body = buildRequestBody('completions', 'gpt-4', messages, { maxTokens: value as any }, false, false)
            expect(body.max_tokens).toBeUndefined()
        }
    })

    it('cleared persisted chat maxTokens null stays Auto for Chat Completions and Responses', () => {
        const completions = buildRequestBody('completions', 'gpt-4', messages, { maxTokens: null as any }, false, false)
        const responses = buildRequestBody('responses', 'gpt-4', messages, { maxTokens: null as any }, false, false)

        expect(completions.max_tokens).toBeUndefined()
        expect(completions.max_output_tokens).toBeUndefined()
        expect(completions.max_prompt_tokens).toBeUndefined()
        expect(responses.max_output_tokens).toBeUndefined()
        expect(responses.max_tokens).toBeUndefined()
        expect(responses.max_prompt_tokens).toBeUndefined()
    })

    it('Auto chat maxTokens omits per-request output caps so server default can apply', () => {
        const autoCompletions = buildRequestBody('completions', 'gpt-4', messages, {}, false, false)
        const disabledCompletions = buildRequestBody('completions', 'gpt-4', messages, { maxTokens: 0 }, false, false)
        const dsv4MaxThinkingAuto = buildRequestBody(
            'completions',
            'dsv4',
            messages,
            { reasoningEffort: 'max' },
            false,
            true,
            undefined,
            'deepseek-v4',
        )
        const autoResponses = buildRequestBody('responses', 'gpt-4', messages, {}, false, false)
        const disabledResponses = buildRequestBody('responses', 'gpt-4', messages, { maxTokens: 0 }, false, false)

        for (const body of [
            autoCompletions,
            disabledCompletions,
            dsv4MaxThinkingAuto,
            autoResponses,
            disabledResponses,
        ]) {
            expect(body.max_tokens).toBeUndefined()
            expect(body.max_output_tokens).toBeUndefined()
            expect(body.max_prompt_tokens).toBeUndefined()
            expect(body.max_context_tokens).toBeUndefined()
            expect(body.max_context).toBeUndefined()
            expect(body.max_thinking_tokens).toBeUndefined()
            expect(body.chat_template_kwargs?.thinking_budget).toBeUndefined()
        }
        expect(dsv4MaxThinkingAuto.reasoning_effort).toBe('max')
    })

    it('keeps per-chat maxThinkingTokens as template thinking budget only, never output or prompt context', () => {
        const body = buildRequestBody(
            'completions',
            'dsv4',
            messages,
            { enableThinking: true, reasoningEffort: 'max', maxThinkingTokens: 4096 },
            false,
            true,
            undefined,
            'deepseek-v4',
        )

        expect(body.max_thinking_tokens).toBe(4096)
        expect(body.chat_template_kwargs).toEqual({ enable_thinking: true, thinking_budget: 4096 })
        expect(body.reasoning_effort).toBe('max')
        expect(body.max_tokens).toBeUndefined()
        expect(body.max_output_tokens).toBeUndefined()
        expect(body.max_prompt_tokens).toBeUndefined()
        expect(body.max_context_tokens).toBeUndefined()
        expect(body.max_context).toBeUndefined()
    })

    it('suppresses maxThinkingTokens when local template metadata says thinking budget is unsupported', () => {
        const body = buildRequestBody(
            'completions',
            'gemma4',
            messages,
            { enableThinking: true, maxThinkingTokens: 4096 },
            false,
            true,
            undefined,
            'gemma4',
            false,
        )

        expect(body.enable_thinking).toBe(true)
        expect(body.chat_template_kwargs).toEqual({ enable_thinking: true })
        expect(body.max_thinking_tokens).toBeUndefined()
        expect(body.chat_template_kwargs.thinking_budget).toBeUndefined()
        expect(body.max_tokens).toBeUndefined()
        expect(body.max_prompt_tokens).toBeUndefined()
    })

    it('switching chats never carries a previous chat maxTokens into Auto Chat Completions', () => {
        const cappedChat = buildRequestBody('completions', 'gpt-4', messages, { maxTokens: 512 }, false, false)
        const autoChat = buildRequestBody('completions', 'gpt-4', messages, {}, false, false)
        const raisedChat = buildRequestBody('completions', 'gpt-4', messages, { maxTokens: 8192 }, false, false)

        expect(cappedChat.max_tokens).toBe(512)
        expect(autoChat.max_tokens).toBeUndefined()
        expect(autoChat.max_output_tokens).toBeUndefined()
        expect(autoChat.max_prompt_tokens).toBeUndefined()
        expect(autoChat.max_context_tokens).toBeUndefined()
        expect(autoChat.max_context).toBeUndefined()
        expect(raisedChat.max_tokens).toBe(8192)
    })

    it('profile-loaded chat maxTokens remains request-scoped and later Auto still uses the server default', () => {
        const serverStartupDefaultMaxTokens = 4096
        const profileLoadedCompletions = buildRequestBody(
            'completions',
            'gpt-4',
            messages,
            { maxTokens: 12000, builtinToolsEnabled: true },
            false,
            false,
        )
        const profileLoadedResponses = buildRequestBody(
            'responses',
            'gpt-4',
            messages,
            { maxTokens: 12000, builtinToolsEnabled: true },
            false,
            false,
        )
        const laterAutoCompletions = buildRequestBody('completions', 'gpt-4', messages, {}, false, false)
        const laterAutoResponses = buildRequestBody('responses', 'gpt-4', messages, {}, false, false)

        expect(profileLoadedCompletions.max_tokens).toBe(12000)
        expect(profileLoadedResponses.max_output_tokens).toBe(12000)
        expect(profileLoadedCompletions.max_tokens).toBeGreaterThan(serverStartupDefaultMaxTokens)
        expect(profileLoadedResponses.max_output_tokens).toBeGreaterThan(serverStartupDefaultMaxTokens)

        for (const body of [profileLoadedCompletions, profileLoadedResponses]) {
            expect(body.max_prompt_tokens).toBeUndefined()
            expect(body.max_context_tokens).toBeUndefined()
            expect(body.max_context).toBeUndefined()
        }
        expect(laterAutoCompletions.max_tokens).toBeUndefined()
        expect(laterAutoResponses.max_output_tokens).toBeUndefined()
        expect(laterAutoCompletions.max_prompt_tokens).toBeUndefined()
        expect(laterAutoResponses.max_prompt_tokens).toBeUndefined()
    })

    it('forwards reasoning_effort', () => {
        const body = buildRequestBody('completions', 'gpt-4', messages, { reasoningEffort: 'high' }, false, true)
        expect(body.reasoning_effort).toBe('high')
    })

    it('suppresses stale reasoning_effort when the model has no reasoning parser', () => {
        const body = buildRequestBody('completions', 'plain-model', messages, { reasoningEffort: 'max' }, false, false)
        expect(body.reasoning_effort).toBeUndefined()
    })

    it('suppresses reasoning_effort when thinking is explicitly off', () => {
        const body = buildRequestBody('completions', 'qwen', messages, { enableThinking: false, reasoningEffort: 'high' }, false, true)
        expect(body.enable_thinking).toBe(false)
        expect(body.reasoning_effort).toBeUndefined()
    })

    it('preserves explicit DSV4 standard thinking max_tokens', () => {
        const body = buildRequestBody('completions', 'dsv4', messages, { maxTokens: 128 }, false, true, undefined, 'deepseek-v4')
        expect(body.max_tokens).toBe(128)
        expect(body.dsv4_finalizer_tokens).toBeUndefined()
    })

    it('preserves explicit DSV4 Max thinking max_tokens', () => {
        const body = buildRequestBody('completions', 'dsv4', messages, { maxTokens: 128, reasoningEffort: 'max' }, false, true, undefined, 'deepseek-v4')
        expect(body.max_tokens).toBe(128)
        expect(body.reasoning_effort).toBe('max')
        expect(body.dsv4_finalizer_tokens).toBeUndefined()
    })

    it('does not floor DSV4 max_tokens when thinking is explicitly off', () => {
        const body = buildRequestBody('completions', 'dsv4', messages, { enableThinking: false, maxTokens: 128 }, false, true, undefined, 'deepseek-v4')
        expect(body.max_tokens).toBe(128)
        expect(body.dsv4_finalizer_tokens).toBeUndefined()
    })
})

describe('buildRequestBody — Remote vs Local gating', () => {
    const messages = [{ role: 'user', content: 'Hello' }]

    it('omits enable_thinking and chat_template_kwargs for local Auto sessions', () => {
        const body = buildRequestBody('completions', 'model', messages, undefined, false, false)
        expect(body.enable_thinking).toBeUndefined()
        expect(body.chat_template_kwargs).toBeUndefined()
    })

    it('includes chat_template_kwargs for explicit local thinking override on reasoning-capable models', () => {
        const body = buildRequestBody('completions', 'model', messages, { enableThinking: true }, false, true)
        expect(body.enable_thinking).toBe(true)
        expect(body.chat_template_kwargs).toEqual({ enable_thinking: true })
    })

    it('suppresses explicit local thinking override when a plain model has no reasoning parser', () => {
        const body = buildRequestBody('completions', 'plain-model', messages, { enableThinking: true }, false, false)
        expect(body.enable_thinking).toBeUndefined()
        expect(body.chat_template_kwargs).toBeUndefined()
    })

    it('forwards explicit ZAYA thinking override when registry detection provides qwen3 parser', () => {
        const body = buildRequestBody('completions', 'zaya-k', messages, { enableThinking: true }, false, true, undefined, 'zaya')
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
        const body = buildRequestBody('responses', 'gpt-4', messages, { maxTokens: 4096 }, false, false)
        expect(body.max_output_tokens).toBe(4096)
        expect(body.max_tokens).toBeUndefined()
    })

    it('keeps Responses maxTokens as output budget only, never prompt context', () => {
        const body = buildRequestBody('responses', 'gpt-4', messages, { maxTokens: 2048 }, false, false)
        expect(body.max_output_tokens).toBe(2048)
        expect(body.max_tokens).toBeUndefined()
        expect(body.max_prompt_tokens).toBeUndefined()
        expect(body.max_context_tokens).toBeUndefined()
        expect(body.max_context).toBeUndefined()
    })

    it('per-chat maxTokens below or above the server startup default remain request scoped for Responses', () => {
        const serverStartupDefaultMaxTokens = 4096
        const below = buildRequestBody('responses', 'gpt-4', messages, { maxTokens: 512 }, false, false)
        const above = buildRequestBody('responses', 'gpt-4', messages, { maxTokens: 8192 }, false, false)

        expect(below.max_output_tokens).toBe(512)
        expect(above.max_output_tokens).toBe(8192)
        expect(below.max_output_tokens).toBeLessThan(serverStartupDefaultMaxTokens)
        expect(above.max_output_tokens).toBeGreaterThan(serverStartupDefaultMaxTokens)
        for (const body of [below, above]) {
            expect(body.max_tokens).toBeUndefined()
            expect(body.max_prompt_tokens).toBeUndefined()
            expect(body.max_context_tokens).toBeUndefined()
            expect(body.max_context).toBeUndefined()
        }
    })

    it('omits invalid persisted maxTokens values instead of poisoning Responses', () => {
        const badValues = [0, -5, Number.NaN, Number.POSITIVE_INFINITY, '1024']
        for (const value of badValues) {
            const body = buildRequestBody('responses', 'gpt-4', messages, { maxTokens: value as any }, false, false)
            expect(body.max_output_tokens).toBeUndefined()
            expect(body.max_tokens).toBeUndefined()
        }
    })

    it('does not invent Responses sampler or output-budget values when chat overrides are absent', () => {
        const body = buildRequestBody('responses', 'qwen-mtp', messages, undefined, false, true)
        expect(body.temperature).toBeUndefined()
        expect(body.top_p).toBeUndefined()
        expect(body.max_output_tokens).toBeUndefined()
        expect(body.repetition_penalty).toBeUndefined()
    })

    it('switching chats never carries a previous chat maxTokens into Auto Responses', () => {
        const cappedChat = buildRequestBody('responses', 'gpt-4', messages, { maxTokens: 512 }, false, false)
        const autoChat = buildRequestBody('responses', 'gpt-4', messages, {}, false, false)
        const raisedChat = buildRequestBody('responses', 'gpt-4', messages, { maxTokens: 8192 }, false, false)

        expect(cappedChat.max_output_tokens).toBe(512)
        expect(autoChat.max_output_tokens).toBeUndefined()
        expect(autoChat.max_tokens).toBeUndefined()
        expect(autoChat.max_prompt_tokens).toBeUndefined()
        expect(autoChat.max_context_tokens).toBeUndefined()
        expect(autoChat.max_context).toBeUndefined()
        expect(raisedChat.max_output_tokens).toBe(8192)
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

    it('preserves DSV4 Responses max_output_tokens for Auto/On thinking', () => {
        const body = buildRequestBody('responses', 'dsv4', messages, { maxTokens: 300 }, false, true, undefined, 'deepseek-v4')
        expect(body.max_output_tokens).toBe(300)
        expect(body.dsv4_finalizer_tokens).toBeUndefined()
    })

    it('preserves DSV4 Responses max_output_tokens for Max thinking', () => {
        const body = buildRequestBody('responses', 'dsv4', messages, { maxTokens: 300, reasoningEffort: 'max' }, false, true, undefined, 'deepseek-v4')
        expect(body.max_output_tokens).toBe(300)
        expect(body.reasoning_effort).toBe('max')
        expect(body.dsv4_finalizer_tokens).toBeUndefined()
    })

    it('keeps Responses maxThinkingTokens as thinking budget only and separate from output caps', () => {
        const body = buildRequestBody(
            'responses',
            'dsv4',
            messages,
            { enableThinking: true, maxTokens: 300, reasoningEffort: 'max', maxThinkingTokens: 4096 },
            false,
            true,
            undefined,
            'deepseek-v4',
        )
        expect(body.max_output_tokens).toBe(300)
        expect(body.max_thinking_tokens).toBe(4096)
        expect(body.chat_template_kwargs).toEqual({ enable_thinking: true, thinking_budget: 4096 })
        expect(body.reasoning_effort).toBe('max')
        expect(body.max_tokens).toBeUndefined()
        expect(body.max_prompt_tokens).toBeUndefined()
        expect(body.max_context_tokens).toBeUndefined()
        expect(body.max_context).toBeUndefined()
    })

    it('suppresses stale maxThinkingTokens when thinking is explicitly off', () => {
        const body = buildRequestBody(
            'responses',
            'dsv4',
            messages,
            { enableThinking: false, maxTokens: 300, reasoningEffort: 'max', maxThinkingTokens: 4096 },
            false,
            true,
            undefined,
            'deepseek-v4',
        )
        expect(body.max_output_tokens).toBe(300)
        expect(body.max_thinking_tokens).toBeUndefined()
        expect(body.chat_template_kwargs).toEqual({ enable_thinking: false })
        expect(body.chat_template_kwargs.thinking_budget).toBeUndefined()
        expect(body.reasoning_effort).toBeUndefined()
    })

    it('suppresses stale Responses reasoning_effort when thinking is explicitly off', () => {
        const body = buildRequestBody('responses', 'dsv4', messages, { enableThinking: false, reasoningEffort: 'max', maxTokens: 300 }, false, true, undefined, 'deepseek-v4')
        expect(body.max_output_tokens).toBe(300)
        expect(body.reasoning_effort).toBeUndefined()
        expect(body.dsv4_finalizer_tokens).toBeUndefined()
    })

    it('suppresses Responses enable_thinking when a plain local model has no reasoning parser', () => {
        const body = buildRequestBody('responses', 'plain-model', messages, { enableThinking: true }, false, false)
        expect(body.enable_thinking).toBeUndefined()
        expect(body.chat_template_kwargs).toBeUndefined()
    })

    it('forwards explicit ZAYA-VL thinking override when registry detection provides qwen3 parser', () => {
        const body = buildRequestBody('responses', 'zaya-vl-k', messages, { enableThinking: true }, false, true, undefined, 'zaya1-vl')
        expect(body.enable_thinking).toBe(true)
        expect(body.chat_template_kwargs).toEqual({ enable_thinking: true })
    })

    it('Hy3 local Responses Auto omits enable_thinking and reasoning_effort', () => {
        const body = buildRequestBody('responses', 'hy3', messages, undefined, false, true, undefined, 'hy3')
        expect(body.enable_thinking).toBeUndefined()
        expect(body.chat_template_kwargs).toBeUndefined()
        expect(body.reasoning_effort).toBeUndefined()
    })

    it('Hy3 local Responses Auto suppresses stale High effort', () => {
        const body = buildRequestBody(
            'responses',
            'hy3',
            messages,
            { enableThinking: undefined, reasoningEffort: 'high' },
            false,
            true,
            undefined,
            'hy3',
        )
        expect(body.enable_thinking).toBeUndefined()
        expect(body.chat_template_kwargs).toBeUndefined()
        expect(body.reasoning_effort).toBeUndefined()
    })

    it('Hy3 local Responses Thinking Off sends explicit off and no High effort', () => {
        const body = buildRequestBody(
            'responses',
            'hy3',
            messages,
            { enableThinking: false, reasoningEffort: 'high' },
            false,
            true,
            undefined,
            'hy3',
        )
        expect(body.enable_thinking).toBe(false)
        expect(body.chat_template_kwargs).toEqual({ enable_thinking: false })
        expect(body.reasoning_effort).toBeUndefined()
        expect(body.chat_template_kwargs.reasoning_effort).toBeUndefined()
        expect(body.chat_template_kwargs.thinking_budget).toBeUndefined()
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

    it('Responses API preserves the augmented custom system prompt when built-in tools are enabled', () => {
        const augmentedSystemPrompt =
            'Custom coding instructions\n\nIMPORTANT: After using any tools, provide a final response. If the user explicitly requested exact final wording or a strict output format, follow that format exactly; otherwise provide a substantive response explaining what you found or did. Never stop after just executing tools.'
        const body = buildRequestBody(
            'responses',
            'model',
            [
                { role: 'system', content: augmentedSystemPrompt },
                { role: 'user', content: 'Create index.html' },
            ],
            { systemPrompt: 'Custom coding instructions', builtinToolsEnabled: true },
            false,
            false,
            sampleTools,
        )

        expect(body.instructions).toBe(augmentedSystemPrompt)
        expect(body.input).toEqual([{ role: 'user', content: 'Create index.html' }])
        expect(body.tools[0].name).toBe('read_file')
    })

    it('agentic tool prompts do not override explicit exact-output user formats', () => {
        const registry = readFileSync('src/main/tools/registry.ts', 'utf8')
        const chat = readFileSync('src/main/ipc/chat.ts', 'utf8')

        expect(registry).toContain('If the user explicitly requested exact final wording or a strict output format, follow that format exactly')
        expect(registry).not.toContain('MUST ALWAYS provide a substantive response')
        expect(chat).toContain('If the user explicitly requested exact final wording or a strict output format, follow that format exactly')
        expect(chat).toContain('suppressAgenticToolPromptForExactOutput')
        expect(chat).toContain('/\\breply exactly\\s*:/i.test(latestUserText)')
        expect(chat).toContain('!suppressAgenticToolPromptForExactOutput')
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
