/**
 * Reasoning Display Tests — comprehensive coverage for the reasoning box pipeline:
 *
 * 1. Client-side <think> tag extraction (fallback when server doesn't send reasoning_content)
 * 2. SSE delta parsing for reasoning_content field (server-side parser active)
 * 3. emitDelta state machine (reasoning→content transitions, reasoningDone events)
 * 4. enable_thinking derivation from session config / overrides
 * 5. Responses API reasoning events (response.reasoning.delta / response.reasoning.done)
 * 6. MessageBubble rendering conditions (dedup guard, hide when content === reasoning)
 * 7. Tool iteration state resets
 * 8. All parser types: qwen3, deepseek_r1, openai_gptoss, none
 * 9. VL model compatibility
 * 10. Wire format compatibility: completions vs responses
 */
import { describe, it, expect } from 'vitest'

// ════════════════════════════════════════════════════════════════════════════════
// 1. Client-side <think> tag extraction
// ════════════════════════════════════════════════════════════════════════════════

/**
 * Extracted from chat.ts — this is the client-side fallback logic that
 * extracts <think>...</think> tags from content when the server doesn't
 * provide a separate reasoning_content field.
 *
 * Returns an array of [text, isReasoning] tuples representing the
 * emitted deltas from processing a single content chunk.
 */
interface ThinkParserState {
    clientSideThinkParsing: boolean
}

function processContentWithThinkFallback(
    content: string,
    serverReasoningProvided: boolean,
    sessionHasReasoningParser: boolean,
    state: ThinkParserState
): Array<[string, boolean]> {
    const emitted: Array<[string, boolean]> = []
    const emit = (text: string, isReasoning: boolean) => {
        if (text) emitted.push([text, isReasoning])
    }

    if (!serverReasoningProvided && sessionHasReasoningParser) {
        if (state.clientSideThinkParsing) {
            // Inside a <think> block — check for closing tag
            const endIdx = content.indexOf('</think>')
            if (endIdx >= 0) {
                const reasoningPart = content.slice(0, endIdx)
                const contentPart = content.slice(endIdx + 8) // 8 = '</think>'.length
                state.clientSideThinkParsing = false
                if (reasoningPart) emit(reasoningPart, true)
                if (contentPart) emit(contentPart, false)
            } else {
                // Still in reasoning block
                emit(content, true)
            }
        } else if (content.includes('<think>')) {
            // Start of think block found
            const startIdx = content.indexOf('<think>')
            const preContent = content.slice(0, startIdx)
            const afterStart = content.slice(startIdx + 7) // 7 = '<think>'.length
            if (preContent) emit(preContent, false)
            // Check if closing tag is also in this delta
            const endIdx = afterStart.indexOf('</think>')
            if (endIdx >= 0) {
                const reasoningPart = afterStart.slice(0, endIdx)
                const postContent = afterStart.slice(endIdx + 8)
                if (reasoningPart) emit(reasoningPart, true)
                if (postContent) emit(postContent, false)
            } else {
                state.clientSideThinkParsing = true
                if (afterStart) emit(afterStart, true)
            }
        } else {
            emit(content, false)
        }
    } else {
        emit(content, false)
    }

    return emitted
}

describe('Client-side <think> tag extraction — basic cases', () => {
    it('passes content through when no reasoning parser configured', () => {
        const state = { clientSideThinkParsing: false }
        const result = processContentWithThinkFallback(
            'Hello world', false, false, state
        )
        expect(result).toEqual([['Hello world', false]])
        expect(state.clientSideThinkParsing).toBe(false)
    })

    it('passes content through when server already provided reasoning_content', () => {
        const state = { clientSideThinkParsing: false }
        const result = processContentWithThinkFallback(
            'Hello world', true, true, state
        )
        expect(result).toEqual([['Hello world', false]])
        expect(state.clientSideThinkParsing).toBe(false)
    })

    it('extracts complete <think> block in single delta', () => {
        const state = { clientSideThinkParsing: false }
        const result = processContentWithThinkFallback(
            '<think>Let me reason about this</think>The answer is 42.',
            false, true, state
        )
        expect(result).toEqual([
            ['Let me reason about this', true],
            ['The answer is 42.', false]
        ])
        expect(state.clientSideThinkParsing).toBe(false)
    })

    it('handles <think> open tag without close in one delta', () => {
        const state = { clientSideThinkParsing: false }
        const result = processContentWithThinkFallback(
            '<think>Starting to think...', false, true, state
        )
        expect(result).toEqual([['Starting to think...', true]])
        expect(state.clientSideThinkParsing).toBe(true)
    })

    it('plain content with parser = true but no tags → emits as content', () => {
        const state = { clientSideThinkParsing: false }
        const result = processContentWithThinkFallback(
            'Just regular content', false, true, state
        )
        expect(result).toEqual([['Just regular content', false]])
        expect(state.clientSideThinkParsing).toBe(false)
    })

    it('empty string produces no emissions', () => {
        const state = { clientSideThinkParsing: false }
        const result = processContentWithThinkFallback('', false, true, state)
        expect(result).toEqual([])
    })
})

describe('Client-side <think> tag extraction — streaming sequences', () => {
    it('multi-delta reasoning: open, middle, close', () => {
        const state = { clientSideThinkParsing: false }

        // Chunk 1: <think> opens
        const r1 = processContentWithThinkFallback(
            '<think>Step 1: ', false, true, state
        )
        expect(r1).toEqual([['Step 1: ', true]])
        expect(state.clientSideThinkParsing).toBe(true)

        // Chunk 2: middle of reasoning
        const r2 = processContentWithThinkFallback(
            'analyze the problem. ', false, true, state
        )
        expect(r2).toEqual([['analyze the problem. ', true]])
        expect(state.clientSideThinkParsing).toBe(true)

        // Chunk 3: close tag with content after
        const r3 = processContentWithThinkFallback(
            'Step 2: conclude.</think>Final answer: 42', false, true, state
        )
        expect(r3).toEqual([
            ['Step 2: conclude.', true],
            ['Final answer: 42', false]
        ])
        expect(state.clientSideThinkParsing).toBe(false)
    })

    it('reasoning completes then normal content flows', () => {
        const state = { clientSideThinkParsing: false }

        processContentWithThinkFallback('<think>Think', false, true, state)
        expect(state.clientSideThinkParsing).toBe(true)

        processContentWithThinkFallback('</think>', false, true, state)
        expect(state.clientSideThinkParsing).toBe(false)

        // Subsequent content flows normally
        const r = processContentWithThinkFallback('Hello', false, true, state)
        expect(r).toEqual([['Hello', false]])
        expect(state.clientSideThinkParsing).toBe(false)
    })

    it('only </think> in delta after reasoning (close tag is the entire delta)', () => {
        const state = { clientSideThinkParsing: true }
        const r = processContentWithThinkFallback('</think>', false, true, state)
        // '</think>' alone: endIdx=0 → reasoningPart='', contentPart=''
        expect(r).toEqual([])
        expect(state.clientSideThinkParsing).toBe(false)
    })

    it('close tag with trailing content only', () => {
        const state = { clientSideThinkParsing: true }
        const r = processContentWithThinkFallback('</think>The answer', false, true, state)
        expect(r).toEqual([['The answer', false]])
        expect(state.clientSideThinkParsing).toBe(false)
    })

    it('reasoning with leading content before <think>', () => {
        const state = { clientSideThinkParsing: false }
        const r = processContentWithThinkFallback(
            'Preamble <think>Now reasoning', false, true, state
        )
        expect(r).toEqual([
            ['Preamble ', false],
            ['Now reasoning', true]
        ])
        expect(state.clientSideThinkParsing).toBe(true)
    })
})

describe('Client-side <think> tag extraction — edge cases', () => {
    it('<think> tag only (no text after)', () => {
        const state = { clientSideThinkParsing: false }
        const r = processContentWithThinkFallback('<think>', false, true, state)
        // afterStart = '' — no emit but state changes
        expect(r).toEqual([])
        expect(state.clientSideThinkParsing).toBe(true)
    })

    it('handles multiple <think> blocks — only first is parsed', () => {
        // This is intentional: we don't support nested or repeated think blocks
        const state = { clientSideThinkParsing: false }
        const r = processContentWithThinkFallback(
            '<think>first</think>middle<think>second</think>end',
            false, true, state
        )
        // First <think> found — splits at it, afterStart includes everything after first <think>
        // afterStart = 'first</think>middle<think>second</think>end'
        // First </think> found in afterStart at idx=5
        // reasoningPart = 'first', postContent = 'middle<think>second</think>end'
        expect(r).toEqual([
            ['first', true],
            ['middle<think>second</think>end', false]
        ])
    })

    it('think tags with newlines and whitespace', () => {
        const state = { clientSideThinkParsing: false }
        const r = processContentWithThinkFallback(
            '<think>\nLet me think\nstep by step\n</think>\nThe answer is 42.',
            false, true, state
        )
        expect(r).toEqual([
            ['\nLet me think\nstep by step\n', true],
            ['\nThe answer is 42.', false]
        ])
    })

    it('unicode content inside think tags', () => {
        const state = { clientSideThinkParsing: false }
        const r = processContentWithThinkFallback(
            '<think>思考中... 🤔</think>答案是42。',
            false, true, state
        )
        expect(r).toEqual([
            ['思考中... 🤔', true],
            ['答案是42。', false]
        ])
    })

    it('malformed: </think> without opening — treated as content', () => {
        const state = { clientSideThinkParsing: false }
        const r = processContentWithThinkFallback(
            '</think>Some content', false, true, state
        )
        // No <think> in content, not in think mode → passes as content
        expect(r).toEqual([['</think>Some content', false]])
    })

    it('state reset between tool iterations', () => {
        const state = { clientSideThinkParsing: true } // left over from previous iteration

        // Simulating tool iteration reset
        state.clientSideThinkParsing = false

        const r = processContentWithThinkFallback(
            'Fresh content after reset', false, true, state
        )
        expect(r).toEqual([['Fresh content after reset', false]])
    })
})

// ════════════════════════════════════════════════════════════════════════════════
// 2. SSE delta parsing — reasoning_content field extraction
// ════════════════════════════════════════════════════════════════════════════════

/**
 * Simulates the Chat Completions SSE delta parsing from chat.ts.
 * Extracts reasoning_content and content from parsed SSE choices.
 */
function parseSseDelta(
    parsed: any,
    sessionHasReasoningParser: boolean,
    state: ThinkParserState
): Array<[string, boolean]> {
    const emitted: Array<[string, boolean]> = []
    const choice = parsed.choices?.[0]?.delta

    const reasoning = choice?.reasoning_content || choice?.reasoning
    if (reasoning) {
        emitted.push([reasoning, true])
    }

    if (choice?.content) {
        const serverReasoningProvided = !!reasoning
        const results = processContentWithThinkFallback(
            choice.content, serverReasoningProvided, sessionHasReasoningParser, state
        )
        emitted.push(...results)
    }

    return emitted
}

describe('SSE delta parsing — reasoning_content field', () => {
    it('extracts reasoning_content from delta', () => {
        const state = { clientSideThinkParsing: false }
        const result = parseSseDelta({
            choices: [{ delta: { reasoning_content: 'Let me think about this' } }]
        }, true, state)
        expect(result).toEqual([['Let me think about this', true]])
    })

    it('extracts reasoning (alternative field name) from delta', () => {
        const state = { clientSideThinkParsing: false }
        const result = parseSseDelta({
            choices: [{ delta: { reasoning: 'Analyzing...' } }]
        }, true, state)
        expect(result).toEqual([['Analyzing...', true]])
    })

    it('reasoning_content takes priority over reasoning field', () => {
        const state = { clientSideThinkParsing: false }
        const result = parseSseDelta({
            choices: [{
                delta: {
                    reasoning_content: 'From reasoning_content',
                    reasoning: 'From reasoning'
                }
            }]
        }, true, state)
        expect(result).toEqual([['From reasoning_content', true]])
    })

    it('content without reasoning_content passes through normally', () => {
        const state = { clientSideThinkParsing: false }
        const result = parseSseDelta({
            choices: [{ delta: { content: 'Hello world' } }]
        }, true, state)
        expect(result).toEqual([['Hello world', false]])
    })

    it('both reasoning_content and content in one delta', () => {
        const state = { clientSideThinkParsing: false }
        const result = parseSseDelta({
            choices: [{
                delta: {
                    reasoning_content: 'Thinking...',
                    content: 'The answer'
                }
            }]
        }, true, state)
        expect(result).toEqual([
            ['Thinking...', true],
            ['The answer', false]
        ])
    })

    it('empty delta (role only) produces no emissions', () => {
        const state = { clientSideThinkParsing: false }
        const result = parseSseDelta({
            choices: [{ delta: { role: 'assistant' } }]
        }, true, state)
        expect(result).toEqual([])
    })

    it('missing choices array produces no emissions', () => {
        const state = { clientSideThinkParsing: false }
        const result = parseSseDelta({ object: 'chat.completion.chunk' }, true, state)
        expect(result).toEqual([])
    })
})

describe('SSE delta parsing — client-side think fallback in SSE context', () => {
    it('content with <think> tags and no reasoning_content triggers fallback', () => {
        const state = { clientSideThinkParsing: false }
        const result = parseSseDelta({
            choices: [{ delta: { content: '<think>Thinking...</think>Answer.' } }]
        }, true, state)
        expect(result).toEqual([
            ['Thinking...', true],
            ['Answer.', false]
        ])
    })

    it('content with <think> tags but server provides reasoning_content — no double-extraction', () => {
        const state = { clientSideThinkParsing: false }
        const result = parseSseDelta({
            choices: [{
                delta: {
                    reasoning_content: 'Server reasoning',
                    content: '<think>This should pass through</think>as-is'
                }
            }]
        }, true, state)
        // reasoning_content already extracted → content passes through as regular content
        expect(result).toEqual([
            ['Server reasoning', true],
            ['<think>This should pass through</think>as-is', false]
        ])
    })

    it('streaming think tags across multiple SSE deltas', () => {
        const state = { clientSideThinkParsing: false }

        const r1 = parseSseDelta({
            choices: [{ delta: { content: '<think>Step 1' } }]
        }, true, state)
        expect(r1).toEqual([['Step 1', true]])
        expect(state.clientSideThinkParsing).toBe(true)

        const r2 = parseSseDelta({
            choices: [{ delta: { content: ', Step 2' } }]
        }, true, state)
        expect(r2).toEqual([[', Step 2', true]])

        const r3 = parseSseDelta({
            choices: [{ delta: { content: '</think>Final' } }]
        }, true, state)
        expect(r3).toEqual([['Final', false]])
        expect(state.clientSideThinkParsing).toBe(false)
    })
})

// ════════════════════════════════════════════════════════════════════════════════
// 3. emitDelta state machine
// ════════════════════════════════════════════════════════════════════════════════

/**
 * Simulates the emitDelta state machine from chat.ts.
 * Tracks reasoningContent, fullContent, and isReasoning transitions.
 */
interface EmitState {
    isReasoning: boolean
    reasoningContent: string
    fullContent: string
    reasoningDoneEmitted: boolean
}

function emitDelta(
    delta: string,
    isReasoningDelta: boolean,
    state: EmitState
): void {
    if (isReasoningDelta) {
        state.isReasoning = true
        state.reasoningContent += delta
    } else {
        if (state.isReasoning) {
            state.isReasoning = false
            state.reasoningDoneEmitted = true
        }
        state.fullContent += delta
    }
}

describe('emitDelta state machine — reasoning→content transitions', () => {
    it('reasoning deltas accumulate in reasoningContent', () => {
        const state: EmitState = { isReasoning: false, reasoningContent: '', fullContent: '', reasoningDoneEmitted: false }
        emitDelta('Think ', true, state)
        emitDelta('step 1 ', true, state)
        emitDelta('step 2', true, state)
        expect(state.reasoningContent).toBe('Think step 1 step 2')
        expect(state.fullContent).toBe('')
        expect(state.isReasoning).toBe(true)
        expect(state.reasoningDoneEmitted).toBe(false)
    })

    it('content deltas accumulate in fullContent', () => {
        const state: EmitState = { isReasoning: false, reasoningContent: '', fullContent: '', reasoningDoneEmitted: false }
        emitDelta('Hello ', false, state)
        emitDelta('world', false, state)
        expect(state.fullContent).toBe('Hello world')
        expect(state.reasoningContent).toBe('')
        expect(state.isReasoning).toBe(false)
    })

    it('reasoning→content transition fires reasoningDone', () => {
        const state: EmitState = { isReasoning: false, reasoningContent: '', fullContent: '', reasoningDoneEmitted: false }

        emitDelta('Thinking...', true, state)
        expect(state.isReasoning).toBe(true)
        expect(state.reasoningDoneEmitted).toBe(false)

        emitDelta('The answer', false, state)
        expect(state.isReasoning).toBe(false)
        expect(state.reasoningDoneEmitted).toBe(true)
        expect(state.reasoningContent).toBe('Thinking...')
        expect(state.fullContent).toBe('The answer')
    })

    it('multiple transitions: reasoning → content → reasoning → content', () => {
        // This simulates a model that reasons, answers, then reasons again (shouldn't happen normally
        // but tests edge case resilience)
        const state: EmitState = { isReasoning: false, reasoningContent: '', fullContent: '', reasoningDoneEmitted: false }

        emitDelta('Thought 1', true, state)
        expect(state.isReasoning).toBe(true)

        emitDelta('Answer 1', false, state)
        expect(state.isReasoning).toBe(false)
        expect(state.reasoningDoneEmitted).toBe(true)

        // Reset to test second reasoning block
        state.reasoningDoneEmitted = false
        emitDelta('Thought 2', true, state)
        expect(state.isReasoning).toBe(true)
        expect(state.reasoningContent).toBe('Thought 1Thought 2') // accumulates

        emitDelta('Answer 2', false, state)
        expect(state.reasoningDoneEmitted).toBe(true)
        expect(state.fullContent).toBe('Answer 1Answer 2')
    })

    it('content-only stream: no reasoning events emitted', () => {
        const state: EmitState = { isReasoning: false, reasoningContent: '', fullContent: '', reasoningDoneEmitted: false }
        emitDelta('Hello', false, state)
        emitDelta(' world', false, state)
        expect(state.reasoningContent).toBe('')
        expect(state.reasoningDoneEmitted).toBe(false)
    })

    it('reasoning-only stream (no content): isReasoning stays true', () => {
        const state: EmitState = { isReasoning: false, reasoningContent: '', fullContent: '', reasoningDoneEmitted: false }
        emitDelta('Only thinking', true, state)
        emitDelta(' and more thinking', true, state)
        expect(state.isReasoning).toBe(true)
        expect(state.reasoningContent).toBe('Only thinking and more thinking')
        expect(state.fullContent).toBe('')
        expect(state.reasoningDoneEmitted).toBe(false)
    })
})

// ════════════════════════════════════════════════════════════════════════════════
// 4. enable_thinking derivation
// ════════════════════════════════════════════════════════════════════════════════

/**
 * Extracted from chat.ts buildRequestBody — determines the enable_thinking value
 * sent to the server based on chat overrides and session configuration.
 */
function deriveEnableThinking(
    overridesEnableThinking: boolean | undefined,
    sessionHasReasoningParser: boolean
): boolean {
    return overridesEnableThinking ?? sessionHasReasoningParser
}

describe('enable_thinking derivation', () => {
    it('Auto mode with qwen3 parser → true', () => {
        expect(deriveEnableThinking(undefined, true)).toBe(true)
    })

    it('Auto mode without parser → false', () => {
        expect(deriveEnableThinking(undefined, false)).toBe(false)
    })

    it('Explicit On overrides no-parser → true', () => {
        expect(deriveEnableThinking(true, false)).toBe(true)
    })

    it('Explicit Off overrides parser → false', () => {
        expect(deriveEnableThinking(false, true)).toBe(false)
    })

    it('Explicit On with parser → true', () => {
        expect(deriveEnableThinking(true, true)).toBe(true)
    })

    it('Explicit Off without parser → false', () => {
        expect(deriveEnableThinking(false, false)).toBe(false)
    })
})

/**
 * Determines sessionHasReasoningParser from session config JSON.
 * Extracted from chat.ts lines 416-433.
 */
function resolveSessionHasReasoningParser(
    sessionConfig: { reasoningParser?: string },
    modelPathDetectedParser?: string
): boolean {
    if (sessionConfig.reasoningParser && sessionConfig.reasoningParser !== 'auto') {
        return true
    } else if (sessionConfig.reasoningParser === 'auto') {
        return !!modelPathDetectedParser
    }
    return false
}

describe('sessionHasReasoningParser resolution', () => {
    it('explicit qwen3 parser → true', () => {
        expect(resolveSessionHasReasoningParser({ reasoningParser: 'qwen3' })).toBe(true)
    })

    it('explicit deepseek_r1 parser → true', () => {
        expect(resolveSessionHasReasoningParser({ reasoningParser: 'deepseek_r1' })).toBe(true)
    })

    it('explicit openai_gptoss parser → true', () => {
        expect(resolveSessionHasReasoningParser({ reasoningParser: 'openai_gptoss' })).toBe(true)
    })

    it('auto with detected parser → true', () => {
        expect(resolveSessionHasReasoningParser({ reasoningParser: 'auto' }, 'qwen3')).toBe(true)
    })

    it('auto without detected parser → false', () => {
        expect(resolveSessionHasReasoningParser({ reasoningParser: 'auto' }, undefined)).toBe(false)
    })

    it('auto with empty string detected parser → false', () => {
        expect(resolveSessionHasReasoningParser({ reasoningParser: 'auto' }, '')).toBe(false)
    })

    it('no parser set → false', () => {
        expect(resolveSessionHasReasoningParser({})).toBe(false)
    })

    it('empty string parser → false', () => {
        expect(resolveSessionHasReasoningParser({ reasoningParser: '' })).toBe(false)
    })

    it('none parser → true (it is a non-auto, non-empty value)', () => {
        expect(resolveSessionHasReasoningParser({ reasoningParser: 'none' })).toBe(true)
    })
})

// ════════════════════════════════════════════════════════════════════════════════
// 5. Responses API reasoning events
// ════════════════════════════════════════════════════════════════════════════════

/**
 * Simulates Responses API SSE event parsing from chat.ts.
 * Handles response.reasoning.delta and response.reasoning.done events.
 */
function parseResponsesApiEvent(
    eventType: string,
    parsed: any,
    isReasoningState: boolean
): { emitted: Array<[string, boolean]>; isReasoning: boolean; reasoningDone: boolean } {
    const emitted: Array<[string, boolean]> = []
    let isReasoning = isReasoningState
    let reasoningDone = false

    if (eventType === 'response.reasoning.delta' && parsed.delta) {
        emitted.push([parsed.delta, true])
        isReasoning = true
    }

    if (eventType === 'response.reasoning.done') {
        if (isReasoning) {
            isReasoning = false
            reasoningDone = true
        }
    }

    if (eventType === 'response.output_text.delta' && (parsed.delta || parsed.text)) {
        emitted.push([parsed.delta || parsed.text, false])
        if (isReasoning) {
            isReasoning = false
            reasoningDone = true
        }
    }

    return { emitted, isReasoning, reasoningDone }
}

describe('Responses API — reasoning events', () => {
    it('response.reasoning.delta emits reasoning', () => {
        const { emitted, isReasoning } = parseResponsesApiEvent(
            'response.reasoning.delta',
            { delta: 'Thinking step 1' },
            false
        )
        expect(emitted).toEqual([['Thinking step 1', true]])
        expect(isReasoning).toBe(true)
    })

    it('response.reasoning.done signals end of reasoning', () => {
        const { reasoningDone, isReasoning } = parseResponsesApiEvent(
            'response.reasoning.done', {}, true
        )
        expect(reasoningDone).toBe(true)
        expect(isReasoning).toBe(false)
    })

    it('response.reasoning.done when not in reasoning → no-op', () => {
        const { reasoningDone, isReasoning } = parseResponsesApiEvent(
            'response.reasoning.done', {}, false
        )
        expect(reasoningDone).toBe(false)
        expect(isReasoning).toBe(false)
    })

    it('response.output_text.delta emits content', () => {
        const { emitted, isReasoning } = parseResponsesApiEvent(
            'response.output_text.delta',
            { delta: 'The answer is 42' },
            false
        )
        expect(emitted).toEqual([['The answer is 42', false]])
        expect(isReasoning).toBe(false)
    })

    it('response.output_text.delta also triggers reasoningDone if was reasoning', () => {
        const { emitted, isReasoning, reasoningDone } = parseResponsesApiEvent(
            'response.output_text.delta',
            { delta: 'Answer' },
            true
        )
        expect(emitted).toEqual([['Answer', false]])
        expect(isReasoning).toBe(false)
        expect(reasoningDone).toBe(true)
    })

    it('handles text field (alternative to delta)', () => {
        const { emitted } = parseResponsesApiEvent(
            'response.output_text.delta',
            { text: 'Via text field' },
            false
        )
        expect(emitted).toEqual([['Via text field', false]])
    })

    it('full reasoning→content sequence via Responses API', () => {
        let isReasoning = false

        // Reasoning phase
        const r1 = parseResponsesApiEvent('response.reasoning.delta', { delta: 'Step 1. ' }, isReasoning)
        isReasoning = r1.isReasoning
        expect(r1.emitted).toEqual([['Step 1. ', true]])
        expect(isReasoning).toBe(true)

        const r2 = parseResponsesApiEvent('response.reasoning.delta', { delta: 'Step 2.' }, isReasoning)
        isReasoning = r2.isReasoning
        expect(r2.emitted).toEqual([['Step 2.', true]])

        // Reasoning done
        const r3 = parseResponsesApiEvent('response.reasoning.done', {}, isReasoning)
        isReasoning = r3.isReasoning
        expect(r3.reasoningDone).toBe(true)
        expect(isReasoning).toBe(false)

        // Content phase
        const r4 = parseResponsesApiEvent('response.output_text.delta', { delta: 'The answer' }, isReasoning)
        expect(r4.emitted).toEqual([['The answer', false]])
    })
})

// ════════════════════════════════════════════════════════════════════════════════
// 6. MessageBubble rendering conditions
// ════════════════════════════════════════════════════════════════════════════════

/**
 * Extracted from MessageBubble.tsx — determines whether the ReasoningBox
 * should be rendered for a given message.
 */
function shouldShowReasoningBox(
    messageRole: string,
    messageContent: string | undefined,
    reasoningContent: string | undefined
): boolean {
    if (messageRole !== 'assistant') return false
    if (!reasoningContent) return false
    // Dedup guard: hide when content matches reasoning
    // (server fallback copies reasoning→content when model has no </think>)
    if (messageContent && reasoningContent.trim() === messageContent.trim()) return false
    return true
}

describe('MessageBubble — reasoning box rendering conditions', () => {
    it('shows for assistant with reasoning content', () => {
        expect(shouldShowReasoningBox('assistant', 'Answer', 'Thinking...')).toBe(true)
    })

    it('hides for user messages', () => {
        expect(shouldShowReasoningBox('user', 'Question', 'Thinking...')).toBe(false)
    })

    it('hides when reasoningContent is empty', () => {
        expect(shouldShowReasoningBox('assistant', 'Answer', '')).toBe(false)
    })

    it('hides when reasoningContent is undefined', () => {
        expect(shouldShowReasoningBox('assistant', 'Answer', undefined)).toBe(false)
    })

    it('hides when reasoning === content (dedup guard)', () => {
        expect(shouldShowReasoningBox('assistant', 'Same text', 'Same text')).toBe(false)
    })

    it('hides when reasoning === content with whitespace differences', () => {
        expect(shouldShowReasoningBox('assistant', '  Same text  ', '  Same text  ')).toBe(false)
    })

    it('shows when reasoning !== content', () => {
        expect(shouldShowReasoningBox('assistant', 'Different answer', 'Reasoning process')).toBe(true)
    })

    it('shows when content is empty but reasoning exists', () => {
        expect(shouldShowReasoningBox('assistant', '', 'Thinking but no answer yet')).toBe(true)
    })

    it('shows when content is undefined but reasoning exists', () => {
        expect(shouldShowReasoningBox('assistant', undefined, 'Thinking...')).toBe(true)
    })
})

// ════════════════════════════════════════════════════════════════════════════════
// 7. Tool iteration state resets
// ════════════════════════════════════════════════════════════════════════════════

/**
 * Simulates the tool iteration reset from chat.ts.
 * Verifies that all reasoning-related state is properly reset.
 */
interface ToolIterationState {
    fullContent: string
    rawAccumulated: string
    clientToolCallBuffering: boolean
    clientSideThinkParsing: boolean
    isReasoning: boolean
    lastFinishReason: string | undefined
}

function resetToolIterationState(state: ToolIterationState): void {
    state.fullContent = ''
    state.rawAccumulated = ''
    state.lastFinishReason = undefined
    state.clientToolCallBuffering = false
    state.clientSideThinkParsing = false
    state.isReasoning = false
}

describe('Tool iteration state resets', () => {
    it('resets all state after tool execution', () => {
        const state: ToolIterationState = {
            fullContent: 'Some content from iteration 1',
            rawAccumulated: 'raw content',
            clientToolCallBuffering: true,
            clientSideThinkParsing: true,
            isReasoning: true,
            lastFinishReason: 'stop'
        }

        resetToolIterationState(state)

        expect(state.fullContent).toBe('')
        expect(state.rawAccumulated).toBe('')
        expect(state.clientToolCallBuffering).toBe(false)
        expect(state.clientSideThinkParsing).toBe(false)
        expect(state.isReasoning).toBe(false)
        expect(state.lastFinishReason).toBeUndefined()
    })

    it('handles already-clean state (idempotent)', () => {
        const state: ToolIterationState = {
            fullContent: '',
            rawAccumulated: '',
            clientToolCallBuffering: false,
            clientSideThinkParsing: false,
            isReasoning: false,
            lastFinishReason: undefined
        }

        resetToolIterationState(state)

        expect(state.fullContent).toBe('')
        expect(state.clientSideThinkParsing).toBe(false)
    })

    it('think parsing state from prev iteration does not leak', () => {
        // Simulate: iteration 1 had in-progress think parsing
        const state: ToolIterationState = {
            fullContent: '',
            rawAccumulated: '',
            clientToolCallBuffering: false,
            clientSideThinkParsing: true, // leaked from iteration 1
            isReasoning: true,
            lastFinishReason: undefined
        }

        resetToolIterationState(state)

        // After reset, new content should be treated as regular content
        const thinkState = { clientSideThinkParsing: state.clientSideThinkParsing }
        const result = processContentWithThinkFallback(
            'Fresh content', false, true, thinkState
        )
        expect(result).toEqual([['Fresh content', false]])
    })
})

// ════════════════════════════════════════════════════════════════════════════════
// 8. Parser-type specific scenarios
// ════════════════════════════════════════════════════════════════════════════════

describe('Parser-type scenarios — qwen3', () => {
    it('standard Qwen3 <think>...</think> pattern', () => {
        const state = { clientSideThinkParsing: false }
        const result = processContentWithThinkFallback(
            '<think>Qwen3 reasoning here</think>Qwen3 answer',
            false, true, state
        )
        expect(result).toEqual([
            ['Qwen3 reasoning here', true],
            ['Qwen3 answer', false]
        ])
    })

    it('Qwen3 with enable_thinking=false → no think tags, just content', () => {
        // When thinking is off, model doesn't generate <think> tags
        const state = { clientSideThinkParsing: false }
        const result = processContentWithThinkFallback(
            'Direct answer without thinking', false, true, state
        )
        expect(result).toEqual([['Direct answer without thinking', false]])
    })

    it('Qwen3 empty thinking: <think>\\n</think>', () => {
        const state = { clientSideThinkParsing: false }
        const result = processContentWithThinkFallback(
            '<think>\n</think>Direct answer', false, true, state
        )
        expect(result).toEqual([
            ['\n', true],
            ['Direct answer', false]
        ])
    })
})

describe('Parser-type scenarios — deepseek_r1', () => {
    it('DeepSeek R1 uses same <think> tags', () => {
        const state = { clientSideThinkParsing: false }
        const result = processContentWithThinkFallback(
            '<think>DeepSeek reasoning</think>DeepSeek answer',
            false, true, state
        )
        expect(result).toEqual([
            ['DeepSeek reasoning', true],
            ['DeepSeek answer', false]
        ])
    })
})

describe('Parser-type scenarios — openai_gptoss', () => {
    // GPT-OSS models typically send reasoning_content in the SSE field,
    // not as <think> tags in content. Here we test that the fallback
    // doesn't interfere when reasoning is already provided by the server.
    it('server-provided reasoning_content → no fallback extraction', () => {
        const state = { clientSideThinkParsing: false }
        const result = parseSseDelta({
            choices: [{
                delta: {
                    reasoning_content: 'GPT-OSS thinking',
                    content: 'GPT-OSS answer'
                }
            }]
        }, true, state)
        expect(result).toEqual([
            ['GPT-OSS thinking', true],
            ['GPT-OSS answer', false]
        ])
    })
})

describe('Parser-type scenarios — no parser', () => {
    it('no parser: <think> tags pass through as content', () => {
        const state = { clientSideThinkParsing: false }
        const result = processContentWithThinkFallback(
            '<think>This should not be extracted</think>Content',
            false, false, state  // sessionHasReasoningParser = false
        )
        // Without parser, everything is content
        expect(result).toEqual([
            ['<think>This should not be extracted</think>Content', false]
        ])
    })
})

// ════════════════════════════════════════════════════════════════════════════════
// 9. VL model compatibility
// ════════════════════════════════════════════════════════════════════════════════

describe('VL model reasoning compatibility', () => {
    it('VL model with qwen3 parser gets think extraction', () => {
        // VL models are just models with isMultimodal=true — reasoning parser
        // behavior should be identical
        const state = { clientSideThinkParsing: false }
        const result = processContentWithThinkFallback(
            '<think>Analyzing the image...</think>This is a cat.',
            false, true, state
        )
        expect(result).toEqual([
            ['Analyzing the image...', true],
            ['This is a cat.', false]
        ])
    })

    it('VL model SSE reasoning_content works same as text model', () => {
        const state = { clientSideThinkParsing: false }
        const result = parseSseDelta({
            choices: [{
                delta: { reasoning_content: 'VL reasoning', content: 'VL answer' }
            }]
        }, true, state)
        expect(result).toEqual([
            ['VL reasoning', true],
            ['VL answer', false]
        ])
    })
})

// ════════════════════════════════════════════════════════════════════════════════
// 10. Wire format compatibility: completions vs responses
// ════════════════════════════════════════════════════════════════════════════════

describe('Wire format — enable_thinking in request body', () => {
    // Import buildRequestBody from request-builder pattern
    function buildRequestBody(
        wireApi: 'completions' | 'responses',
        overrides: { enableThinking?: boolean; reasoningEffort?: string } | undefined,
        isRemote: boolean,
        sessionHasReasoningParser: boolean
    ): Record<string, any> {
        const obj: Record<string, any> = {
            model: 'test-model',
            stream: true
        }
        obj.enable_thinking = overrides?.enableThinking ?? sessionHasReasoningParser
        if (!isRemote) obj.chat_template_kwargs = { enable_thinking: obj.enable_thinking }
        if (overrides?.reasoningEffort) obj.reasoning_effort = overrides.reasoningEffort
        if (wireApi === 'responses') obj.max_output_tokens = 4096
        else obj.max_tokens = 4096
        return obj
    }

    it('Completions: Auto + qwen3 parser → enable_thinking=true, chat_template_kwargs', () => {
        const body = buildRequestBody('completions', undefined, false, true)
        expect(body.enable_thinking).toBe(true)
        expect(body.chat_template_kwargs).toEqual({ enable_thinking: true })
    })

    it('Completions: Explicit Off → enable_thinking=false', () => {
        const body = buildRequestBody('completions', { enableThinking: false }, false, true)
        expect(body.enable_thinking).toBe(false)
        expect(body.chat_template_kwargs).toEqual({ enable_thinking: false })
    })

    it('Completions: remote excludes chat_template_kwargs', () => {
        const body = buildRequestBody('completions', undefined, true, true)
        expect(body.enable_thinking).toBe(true)
        expect(body.chat_template_kwargs).toBeUndefined()
    })

    it('Responses: Auto + parser → enable_thinking=true', () => {
        const body = buildRequestBody('responses', undefined, false, true)
        expect(body.enable_thinking).toBe(true)
        expect(body.max_output_tokens).toBe(4096)
    })

    it('Responses: remote excludes chat_template_kwargs', () => {
        const body = buildRequestBody('responses', undefined, true, true)
        expect(body.chat_template_kwargs).toBeUndefined()
    })

    it('reasoning_effort passed when set', () => {
        const body = buildRequestBody('completions', { reasoningEffort: 'high' }, false, true)
        expect(body.reasoning_effort).toBe('high')
    })

    it('reasoning_effort not present when not set', () => {
        const body = buildRequestBody('completions', undefined, false, true)
        expect(body.reasoning_effort).toBeUndefined()
    })
})

// ════════════════════════════════════════════════════════════════════════════════
// 11. End-to-end integration scenarios
// ════════════════════════════════════════════════════════════════════════════════

describe('End-to-end — full stream processing', () => {
    it('Qwen3.5 model: <think> in content, no server reasoning_content', () => {
        const state = { clientSideThinkParsing: false }
        const emitState: EmitState = { isReasoning: false, reasoningContent: '', fullContent: '', reasoningDoneEmitted: false }

        // Simulate SSE chunks from server (no reasoning_content field)
        const chunks = [
            { choices: [{ delta: { role: 'assistant' } }] },
            { choices: [{ delta: { content: '<think>' } }] },
            { choices: [{ delta: { content: 'Let me think step by step.\n' } }] },
            { choices: [{ delta: { content: '2 + 2 = 4\n' } }] },
            { choices: [{ delta: { content: '</think>' } }] },
            { choices: [{ delta: { content: 'The answer is 4.' } }] },
        ]

        for (const chunk of chunks) {
            const deltas = parseSseDelta(chunk, true, state)
            for (const [text, isReasoning] of deltas) {
                emitDelta(text, isReasoning, emitState)
            }
        }

        expect(emitState.reasoningContent).toBe('Let me think step by step.\n2 + 2 = 4\n')
        expect(emitState.fullContent).toBe('The answer is 4.')
        expect(emitState.reasoningDoneEmitted).toBe(true)
        expect(emitState.isReasoning).toBe(false)

        // The UI should show the reasoning box
        expect(shouldShowReasoningBox('assistant', emitState.fullContent, emitState.reasoningContent)).toBe(true)
    })

    it('Server with reasoning parser active: reasoning_content in SSE', () => {
        const state = { clientSideThinkParsing: false }
        const emitState: EmitState = { isReasoning: false, reasoningContent: '', fullContent: '', reasoningDoneEmitted: false }

        const chunks = [
            { choices: [{ delta: { role: 'assistant' } }] },
            { choices: [{ delta: { reasoning_content: 'Step 1: analyze.' } }] },
            { choices: [{ delta: { reasoning_content: ' Step 2: solve.' } }] },
            { choices: [{ delta: { content: 'The answer is 42.' } }] },
        ]

        for (const chunk of chunks) {
            const deltas = parseSseDelta(chunk, true, state)
            for (const [text, isReasoning] of deltas) {
                emitDelta(text, isReasoning, emitState)
            }
        }

        expect(emitState.reasoningContent).toBe('Step 1: analyze. Step 2: solve.')
        expect(emitState.fullContent).toBe('The answer is 42.')
        expect(emitState.reasoningDoneEmitted).toBe(true)
        expect(shouldShowReasoningBox('assistant', emitState.fullContent, emitState.reasoningContent)).toBe(true)
    })

    it('Non-reasoning model: no think tags, content only', () => {
        const state = { clientSideThinkParsing: false }
        const emitState: EmitState = { isReasoning: false, reasoningContent: '', fullContent: '', reasoningDoneEmitted: false }

        const chunks = [
            { choices: [{ delta: { content: 'Hello! ' } }] },
            { choices: [{ delta: { content: 'How can I help?' } }] },
        ]

        for (const chunk of chunks) {
            const deltas = parseSseDelta(chunk, false, state)
            for (const [text, isReasoning] of deltas) {
                emitDelta(text, isReasoning, emitState)
            }
        }

        expect(emitState.reasoningContent).toBe('')
        expect(emitState.fullContent).toBe('Hello! How can I help?')
        expect(emitState.reasoningDoneEmitted).toBe(false)
        expect(shouldShowReasoningBox('assistant', emitState.fullContent, emitState.reasoningContent)).toBe(false)
    })

    it('Reasoning model with thinking=Off: no think tags generated', () => {
        // User explicitly disabled thinking — model sends direct content
        // enable_thinking=false means no <think> tags in output
        const enableThinking = deriveEnableThinking(false, true)
        expect(enableThinking).toBe(false)

        const state = { clientSideThinkParsing: false }
        const emitState: EmitState = { isReasoning: false, reasoningContent: '', fullContent: '', reasoningDoneEmitted: false }

        const chunks = [
            { choices: [{ delta: { content: 'Direct answer: 42' } }] },
        ]

        for (const chunk of chunks) {
            const deltas = parseSseDelta(chunk, true, state)
            for (const [text, isReasoning] of deltas) {
                emitDelta(text, isReasoning, emitState)
            }
        }

        expect(emitState.reasoningContent).toBe('')
        expect(emitState.fullContent).toBe('Direct answer: 42')
        expect(shouldShowReasoningBox('assistant', emitState.fullContent, emitState.reasoningContent)).toBe(false)
    })

    it('Tool iteration: reasoning in iteration 1, tool call, clean iteration 2', () => {
        const state = { clientSideThinkParsing: false }
        const emitState: EmitState = { isReasoning: false, reasoningContent: '', fullContent: '', reasoningDoneEmitted: false }

        // Iteration 1: model thinks and produces content
        parseSseDelta(
            { choices: [{ delta: { content: '<think>Analyzing for tool use</think>' } }] },
            true, state
        ).forEach(([text, isR]) => emitDelta(text, isR, emitState))

        expect(emitState.reasoningContent).toBe('Analyzing for tool use')
        expect(state.clientSideThinkParsing).toBe(false)

        // Simulate tool iteration reset
        state.clientSideThinkParsing = false
        emitState.fullContent = ''
        emitState.isReasoning = false

        // Iteration 2: fresh content after tool result
        parseSseDelta(
            { choices: [{ delta: { content: 'Based on the tool result...' } }] },
            true, state
        ).forEach(([text, isR]) => emitDelta(text, isR, emitState))

        expect(emitState.fullContent).toBe('Based on the tool result...')
    })
})

// ════════════════════════════════════════════════════════════════════════════════
// 12. Token counting accuracy
// ════════════════════════════════════════════════════════════════════════════════

/**
 * Simulates the full SSE → emitDelta pipeline with accurate token counting.
 * This matches the chat.ts logic where:
 * - Each SSE chunk should count as ONE token (in client-side counting mode)
 * - Think-tag splitting must NOT inflate the count
 * - Server-provided reasoning_content + content in one delta counts as ONE token
 *
 * The `skipClientCount` parameter in emitDelta prevents inflation when
 * a single SSE chunk is split into multiple emitDelta calls.
 */
interface TokenCountState {
    tokenCount: number
    thinkState: ThinkParserState
    emitState: EmitState
    serverSendsUsage: boolean
}

function processChunkWithCounting(
    parsed: any,
    sessionHasReasoningParser: boolean,
    state: TokenCountState
): void {
    const choice = parsed.choices?.[0]?.delta
    if (!choice) return

    const reasoning = choice?.reasoning_content || choice?.reasoning
    let chunkCounted = false

    // Count once per SSE chunk (not per emitDelta call)
    const countOnce = () => {
        if (!state.serverSendsUsage && !chunkCounted) {
            state.tokenCount++
            chunkCounted = true
        }
    }

    if (reasoning) {
        countOnce()
        emitDelta(reasoning, true, state.emitState)
    }

    if (choice?.content) {
        if (!reasoning && sessionHasReasoningParser) {
            // Think-fallback path: may produce multiple emitDelta calls
            const content = choice.content as string
            const results = processContentWithThinkFallback(
                content, !!reasoning, sessionHasReasoningParser, state.thinkState
            )
            for (const [text, isR] of results) {
                countOnce()
                emitDelta(text, isR, state.emitState)
            }
        } else {
            countOnce()
            emitDelta(choice.content, false, state.emitState)
        }
    }

    // Handle server-sent usage (overrides client counting)
    if (parsed.usage?.completion_tokens != null) {
        state.serverSendsUsage = true
        state.tokenCount = parsed.usage.completion_tokens
    }
}

function newTokenState(): TokenCountState {
    return {
        tokenCount: 0,
        thinkState: { clientSideThinkParsing: false },
        emitState: { isReasoning: false, reasoningContent: '', fullContent: '', reasoningDoneEmitted: false },
        serverSendsUsage: false
    }
}

describe('Token counting accuracy — no inflation from think-splitting', () => {
    it('single content chunk: 1 token', () => {
        const state = newTokenState()
        processChunkWithCounting(
            { choices: [{ delta: { content: 'Hello' } }] },
            true, state
        )
        expect(state.tokenCount).toBe(1)
    })

    it('<think>X</think>Y in single chunk: still 1 token', () => {
        const state = newTokenState()
        processChunkWithCounting(
            { choices: [{ delta: { content: '<think>Reasoning</think>Answer' } }] },
            true, state
        )
        // This produces 2 emitDelta calls (reasoning + content) but should count as 1 token
        expect(state.tokenCount).toBe(1)
        expect(state.emitState.reasoningContent).toBe('Reasoning')
        expect(state.emitState.fullContent).toBe('Answer')
    })

    it('pre<think>X in single chunk: still 1 token', () => {
        const state = newTokenState()
        processChunkWithCounting(
            { choices: [{ delta: { content: 'Pre <think>Reasoning' } }] },
            true, state
        )
        // 2 emitDelta calls (pre-content + reasoning) but 1 token
        expect(state.tokenCount).toBe(1)
    })

    it('streaming 5 chunks: 5 tokens total (including think splits)', () => {
        const state = newTokenState()
        const chunks = [
            { choices: [{ delta: { content: '<think>Step 1' } }] },   // think-split: 1 token
            { choices: [{ delta: { content: ' Step 2' } }] },          // inside think: 1 token
            { choices: [{ delta: { content: '</think>Answer' } }] },   // think-split: 1 token
            { choices: [{ delta: { content: ' more' } }] },            // plain: 1 token
            { choices: [{ delta: { content: ' content' } }] },         // plain: 1 token
        ]
        for (const chunk of chunks) {
            processChunkWithCounting(chunk, true, state)
        }
        expect(state.tokenCount).toBe(5)
        expect(state.emitState.reasoningContent).toBe('Step 1 Step 2')
        expect(state.emitState.fullContent).toBe('Answer more content')
    })

    it('server-provided reasoning_content + content: 1 token', () => {
        const state = newTokenState()
        processChunkWithCounting(
            { choices: [{ delta: { reasoning_content: 'Think', content: 'Answer' } }] },
            true, state
        )
        // Both fields in one SSE chunk = 1 token
        expect(state.tokenCount).toBe(1)
    })

    it('reasoning_content only: 1 token', () => {
        const state = newTokenState()
        processChunkWithCounting(
            { choices: [{ delta: { reasoning_content: 'Thinking...' } }] },
            true, state
        )
        expect(state.tokenCount).toBe(1)
    })

    it('role-only delta: 0 tokens', () => {
        const state = newTokenState()
        processChunkWithCounting(
            { choices: [{ delta: { role: 'assistant' } }] },
            true, state
        )
        expect(state.tokenCount).toBe(0)
    })

    it('server usage overrides client counting', () => {
        const state = newTokenState()
        // First few chunks use client counting
        processChunkWithCounting(
            { choices: [{ delta: { content: 'A' } }] },
            false, state
        )
        processChunkWithCounting(
            { choices: [{ delta: { content: 'B' } }] },
            false, state
        )
        expect(state.tokenCount).toBe(2) // client counting

        // Then server starts sending usage
        processChunkWithCounting(
            { choices: [{ delta: { content: 'C' } }], usage: { completion_tokens: 10 } },
            false, state
        )
        expect(state.tokenCount).toBe(10) // server override

        // Subsequent chunks: server usage stays authoritative
        processChunkWithCounting(
            { choices: [{ delta: { content: 'D' } }], usage: { completion_tokens: 11 } },
            false, state
        )
        expect(state.tokenCount).toBe(11)
    })

    it('no parser: no inflation even with <think> in content', () => {
        const state = newTokenState()
        processChunkWithCounting(
            { choices: [{ delta: { content: '<think>X</think>Y' } }] },
            false, state  // no parser → passes through as content
        )
        expect(state.tokenCount).toBe(1)
        expect(state.emitState.fullContent).toBe('<think>X</think>Y')
    })

    it('extended stream: accurate count across 20+ chunks', () => {
        const state = newTokenState()
        // Simulate a realistic stream: 3 reasoning + 17 content chunks
        processChunkWithCounting({ choices: [{ delta: { content: '<think>' } }] }, true, state)
        for (let i = 0; i < 10; i++) {
            processChunkWithCounting(
                { choices: [{ delta: { content: `reasoning token ${i} ` } }] },
                true, state
            )
        }
        processChunkWithCounting({ choices: [{ delta: { content: '</think>' } }] }, true, state)
        for (let i = 0; i < 10; i++) {
            processChunkWithCounting(
                { choices: [{ delta: { content: `content token ${i} ` } }] },
                true, state
            )
        }
        // <think> alone and </think> alone produce empty emissions (no text),
        // so they correctly don't increment the counter.
        // 10 (reasoning tokens) + 10 (content tokens) = 20
        expect(state.tokenCount).toBe(20)
    })
})

// ════════════════════════════════════════════════════════════════════════════════
// 13. DeepSeek R1 implicit reasoning edge cases
// ════════════════════════════════════════════════════════════════════════════════

describe('DeepSeek R1 — implicit reasoning (no opening <think>)', () => {
    // DeepSeek R1 may output reasoning without an explicit <think> tag,
    // only closing with </think>. In this case, the CLIENT-SIDE fallback
    // treats everything before </think> as content because there's no
    // <think> start tag. The SERVER-SIDE DeepSeek parser handles this
    // correctly by treating everything before </think> as reasoning.
    // These tests verify both paths.

    it('client-side: </think> without opener treats pre-tag as content', () => {
        const state = { clientSideThinkParsing: false }
        const result = processContentWithThinkFallback(
            'I need to analyze this</think>Final answer',
            false, true, state
        )
        // Client-side doesn't handle implicit reasoning — passes as content
        expect(result).toEqual([['I need to analyze this</think>Final answer', false]])
    })

    it('server-side DeepSeek reasoning_content handles implicit correctly', () => {
        // When server has DeepSeek parser, it sends reasoning_content field
        const state = { clientSideThinkParsing: false }
        const result = parseSseDelta({
            choices: [{
                delta: {
                    reasoning_content: 'I need to analyze this',
                    content: 'Final answer'
                }
            }]
        }, true, state)
        expect(result).toEqual([
            ['I need to analyze this', true],
            ['Final answer', false]
        ])
    })

    it('DeepSeek streaming: reasoning_content deltas followed by content', () => {
        const state = { clientSideThinkParsing: false }
        const emitState: EmitState = {
            isReasoning: false, reasoningContent: '', fullContent: '', reasoningDoneEmitted: false
        }

        const chunks = [
            { choices: [{ delta: { reasoning_content: 'Let me think' } }] },
            { choices: [{ delta: { reasoning_content: ' step by step.' } }] },
            { choices: [{ delta: { content: 'The answer is 42.' } }] },
        ]

        for (const chunk of chunks) {
            const deltas = parseSseDelta(chunk, true, state)
            for (const [text, isR] of deltas) {
                emitDelta(text, isR, emitState)
            }
        }

        expect(emitState.reasoningContent).toBe('Let me think step by step.')
        expect(emitState.fullContent).toBe('The answer is 42.')
        expect(emitState.reasoningDoneEmitted).toBe(true)
    })
})

// ════════════════════════════════════════════════════════════════════════════════
// 14. GPT-OSS / Harmony protocol client-side behavior
// ════════════════════════════════════════════════════════════════════════════════

describe('GPT-OSS / Harmony — client-side behavior', () => {
    // GPT-OSS uses channel markers (<|channel|>analysis<|message|>, etc.)
    // NOT <think> tags. The server-side parser handles channel extraction
    // and sends reasoning_content field. The client-side <think> fallback
    // should NOT interfere with this.

    it('Harmony content without <think> passes through even with parser', () => {
        const state = { clientSideThinkParsing: false }
        const result = processContentWithThinkFallback(
            'This is content from final channel',
            false, true, state
        )
        expect(result).toEqual([['This is content from final channel', false]])
    })

    it('server-provided reasoning from Harmony parser works correctly', () => {
        const state = { clientSideThinkParsing: false }
        const emitState: EmitState = {
            isReasoning: false, reasoningContent: '', fullContent: '', reasoningDoneEmitted: false
        }

        const chunks = [
            { choices: [{ delta: { reasoning_content: 'Harmony analysis content' } }] },
            { choices: [{ delta: { content: 'Harmony final content' } }] },
        ]

        for (const chunk of chunks) {
            const deltas = parseSseDelta(chunk, true, state)
            for (const [text, isR] of deltas) {
                emitDelta(text, isR, emitState)
            }
        }

        expect(emitState.reasoningContent).toBe('Harmony analysis content')
        expect(emitState.fullContent).toBe('Harmony final content')
        expect(emitState.reasoningDoneEmitted).toBe(true)
    })
})

