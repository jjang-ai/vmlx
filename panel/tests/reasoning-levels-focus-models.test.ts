import { mkdtempSync, mkdirSync, rmSync, writeFileSync } from 'node:fs'
import { tmpdir } from 'node:os'
import { join } from 'node:path'
import { afterEach, describe, expect, it } from 'vitest'

import { detectModelConfigFromDir } from '../src/main/model-config-registry'
import { sanitizeChatOverrides } from '../src/main/chat-override-policy'
import {
  DSV41_REASONING_BUDGETS,
  applyFinalAnswerRecoveryThinkingOff,
  applyReasoningRequestFields,
} from '../src/shared/reasoningEffortPolicy'

/**
 * Reasoning effort levels for the two focus models come from the bundles' own
 * prompt code, read on the Studio 2026-11-02:
 *
 *   MiniMax-M3.1-Flash-JANGHT1.9/chat_template.jinja
 *     L84-89  thinking_mode == "enabled" | "disabled" | else adaptive
 *     L93-94  {%- if reasoning_effort is defined and reasoning_effort is not none -%}
 *             {{- '\n\n<effort>' ~ reasoning_effort ~ '</effort>' }}
 *     card README "Reasoning: adaptive by default; thinking_mode = enabled / disabled;
 *     reasoning_effort = low / medium / high / xhigh / max"
 *
 *   DeepSeek-V4.1-Flash-1017-JANG_2K/encoding/encoding.py
 *     L443-453 REASONING_EFFORT_TEMPLATE "Reasoning Effort: {budget} (range 1-100, ...)",
 *              REASONING_EFFORT_MAPPINGS {"low": 50, "high": 75, "max": 100}, DEFAULT "high"
 *     L472     rendered only when thinking_mode == "thinking" (index 0)
 *   vmlx_engine/models/deepseek_v4_1/chat_encoding.py maps minimal 25 / low 50 / medium 62 /
 *   high 75 / xhigh 88 / max 100 onto that budget on every endpoint.
 */
const dirs: string[] = []
afterEach(() => {
  for (const dir of dirs.splice(0)) rmSync(dir, { recursive: true, force: true })
})
function bundle(name: string): string {
  const dir = mkdtempSync(join(tmpdir(), `${name}-`))
  dirs.push(dir)
  return dir
}

// The relevant lines of the real M3.1 template.
const M31_TEMPLATE = `{%- if thinking_mode is defined and thinking_mode == "enabled" -%}
    {{- 'Current thinking mode: enabled.' }}
{%- elif thinking_mode is defined and thinking_mode == "disabled" -%}
    {{- 'Current thinking mode: disabled.' }}
{%- else -%}
    {{- 'Current thinking mode: adaptive.' }}
{%- endif -%}
{%- if reasoning_effort is defined and reasoning_effort is not none -%}
    {{- '\\n\\n<effort>' ~ reasoning_effort ~ '</effort>' }}
{%- elif effort is defined and effort is not none -%}
    {{- '\\n\\n<effort>' ~ effort ~ '</effort>' }}
{%- endif -%}`

describe('MiniMax-M3.1 Flash: enabled / disabled / adaptive plus five effort tags', () => {
  it('offers the five documented levels only when the bundle template renders <effort>', () => {
    const m31 = bundle('m31')
    writeFileSync(join(m31, 'config.json'), JSON.stringify({ model_type: 'minimax_m3_vl' }))
    writeFileSync(join(m31, 'chat_template.jinja'), M31_TEMPLATE)
    const detected = detectModelConfigFromDir(m31)
    expect(detected.family).toBe('minimax_m3')
    expect(detected.supportsAdaptiveThinking).toBe(true)
    expect(detected.honorsEnableThinking).toBe(true)
    expect(detected.supportedReasoningEfforts).toEqual(['low', 'medium', 'high', 'xhigh', 'max'])
    // Unset = no <effort> tag: there is no default level to advertise.
    expect(detected.defaultReasoningEffort).toBeUndefined()

    // A MiniMax-M3 template without the effort branch has no effort control, so none is offered.
    const m3 = bundle('m3')
    writeFileSync(join(m3, 'config.json'), JSON.stringify({ model_type: 'minimax_m3' }))
    writeFileSync(join(m3, 'chat_template.jinja'), M31_TEMPLATE.split('{%- if reasoning_effort')[0])
    expect(detectModelConfigFromDir(m3).supportedReasoningEfforts).toBeUndefined()
  })

  it.each(['completions', 'responses'] as const)('%s: Adaptive + a level sends thinking_mode adaptive AND reasoning_effort (the template renders both)', wireApi => {
    const saved = sanitizeChatOverrides({ chatId: 'c', thinkingMode: 'adaptive' as const, reasoningEffort: 'xhigh' })
    expect(saved).toMatchObject({ thinkingMode: 'adaptive', reasoningEffort: 'xhigh' })
    const body: Record<string, any> = {}
    applyReasoningRequestFields(body, {
      ...saved, isRemote: false, sessionHasReasoningParser: true, supportsAdaptiveThinking: true,
      detectedFamily: 'minimax_m3', supportedReasoningEfforts: ['low', 'medium', 'high', 'xhigh', 'max'], wireApi,
    })
    expect(body).toEqual({ thinking_mode: 'adaptive', chat_template_kwargs: { thinking_mode: 'adaptive' }, reasoning_effort: 'xhigh' })
  })

  it.each(['completions', 'responses'] as const)('%s: On + level and Off reach the request the way the engine maps them to enabled / disabled', wireApi => {
    const on: Record<string, any> = {}
    applyReasoningRequestFields(on, {
      enableThinking: true, reasoningEffort: 'low', isRemote: false, sessionHasReasoningParser: true,
      supportsAdaptiveThinking: true, detectedFamily: 'minimax_m3', supportedReasoningEfforts: ['low', 'medium', 'high', 'xhigh', 'max'], wireApi,
    })
    expect(on).toEqual({ enable_thinking: true, chat_template_kwargs: { enable_thinking: true }, reasoning_effort: 'low', thinking_mode: 'reasoning' })
    const off: Record<string, any> = {}
    applyReasoningRequestFields(off, {
      enableThinking: false, reasoningEffort: 'low', isRemote: false, sessionHasReasoningParser: true,
      supportsAdaptiveThinking: true, detectedFamily: 'minimax_m3', supportedReasoningEfforts: ['low', 'medium', 'high', 'xhigh', 'max'], wireApi,
    })
    expect(off).toEqual({ enable_thinking: false, chat_template_kwargs: { enable_thinking: false }, thinking_mode: 'instruct' })
  })

  it('the bounded post-tool recovery pass drops the adaptive template mode it is overriding', () => {
    // Retained record: a chat saved as Adaptive produced requests with thinking_mode instruct + enable_thinking
    // false. Those are the recovery pass; it left chat_template_kwargs.thinking_mode = adaptive behind, which the
    // engine rejects as "adaptive thinking_mode conflicts with the template mode".
    const body: Record<string, any> = {}
    applyReasoningRequestFields(body, { thinkingMode: 'adaptive', reasoningEffort: 'high', isRemote: false, sessionHasReasoningParser: true, supportsAdaptiveThinking: true })
    applyFinalAnswerRecoveryThinkingOff(body)
    expect(body).toEqual({ enable_thinking: false, thinking_mode: 'instruct', chat_template_kwargs: { enable_thinking: false } })
  })
})

describe('DeepSeek-V4.1 Flash: chat / thinking plus a 1..100 budget', () => {
  it('offers the five vMLX levels with the encoder default High when the bundle ships encoding/encoding.py', () => {
    const dir = bundle('dsv41')
    writeFileSync(join(dir, 'config.json'), JSON.stringify({ model_type: 'deepseek_v41' }))
    mkdirSync(join(dir, 'encoding'))
    writeFileSync(join(dir, 'encoding', 'encoding.py'), 'REASONING_EFFORT_MAPPINGS = {"low": 50, "high": 75, "max": 100}\n')
    const detected = detectModelConfigFromDir(dir)
    expect(detected.family).toBe('deepseek-v4')
    expect(detected.architectureHints?.deepseekVariant).toBe('v4.1')
    expect(detected.honorsEnableThinking).toBe(true)
    expect(detected.supportsInstructMode).not.toBe(false)
    expect(detected.supportedReasoningEfforts).toEqual(['low', 'medium', 'high', 'xhigh', 'max'])
    expect(detected.defaultReasoningEffort).toBe('high')
    // Each level is a distinct budget the encoder accepts (int 1..100).
    expect(Object.values(DSV41_REASONING_BUDGETS)).toEqual([50, 62, 75, 88, 100])
    expect(DSV41_REASONING_BUDGETS.high).toBe(75)
  })

  it.each(['completions', 'responses'] as const)('%s: On + level sends enable_thinking + reasoning_effort and no generic thinking_mode; Off sends the chat rail', wireApi => {
    const on: Record<string, any> = {}
    applyReasoningRequestFields(on, {
      enableThinking: true, reasoningEffort: 'medium', isRemote: false, sessionHasReasoningParser: true,
      detectedFamily: 'deepseek-v4', supportedReasoningEfforts: ['low', 'medium', 'high', 'xhigh', 'max'], defaultReasoningEffort: 'high', wireApi,
    })
    expect(on).toEqual({ enable_thinking: true, chat_template_kwargs: { enable_thinking: true }, reasoning_effort: 'medium' })
    const off: Record<string, any> = {}
    applyReasoningRequestFields(off, {
      enableThinking: false, reasoningEffort: 'medium', isRemote: false, sessionHasReasoningParser: true,
      detectedFamily: 'deepseek-v4', supportedReasoningEfforts: ['low', 'medium', 'high', 'xhigh', 'max'], wireApi,
    })
    expect(off).toEqual({ enable_thinking: false, chat_template_kwargs: { enable_thinking: false } })
  })

  it('a level the bundle does not implement is refused instead of silently falling back', () => {
    expect(() => applyReasoningRequestFields({}, {
      enableThinking: true, reasoningEffort: 'xhigh', isRemote: false, sessionHasReasoningParser: true,
      detectedFamily: 'deepseek-v4', supportedReasoningEfforts: ['low', 'high', 'max'],
    })).toThrow('not supported by the loaded bundle')
  })
})
