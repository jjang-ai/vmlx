import { describe, expect, it } from 'vitest'
import { readFileSync } from 'node:fs'
import {
  extractResponsesWarnings,
  categorizeResponsesWarning,
} from '../src/renderer/src/lib/responsesWarnings'

describe('extractResponsesWarnings', () => {
  it('returns null for null/undefined payload', () => {
    expect(extractResponsesWarnings(null)).toBeNull()
    expect(extractResponsesWarnings(undefined)).toBeNull()
  })

  it('returns null when warnings field absent', () => {
    expect(extractResponsesWarnings({})).toBeNull()
  })

  it('returns null when warnings is not an array', () => {
    expect(extractResponsesWarnings({ warnings: 'not an array' as any })).toBeNull()
    expect(extractResponsesWarnings({ warnings: 42 as any })).toBeNull()
    expect(extractResponsesWarnings({ warnings: null })).toBeNull()
  })

  it('returns null when warnings array is empty', () => {
    expect(extractResponsesWarnings({ warnings: [] })).toBeNull()
  })

  it('returns null when warnings contain only empty/whitespace strings', () => {
    expect(extractResponsesWarnings({ warnings: ['', '   ', '\n\t'] })).toBeNull()
  })

  it('returns null when warnings contain only non-strings', () => {
    expect(extractResponsesWarnings({ warnings: [1, 2, {}, null] as any })).toBeNull()
  })

  it('extracts a single valid warning', () => {
    const w = extractResponsesWarnings({ warnings: ['something happened'] })
    expect(w).toEqual(['something happened'])
  })

  it('trims whitespace from warnings', () => {
    expect(extractResponsesWarnings({ warnings: ['  hello  '] })).toEqual(['hello'])
  })

  it('drops non-string entries but keeps strings', () => {
    expect(
      extractResponsesWarnings({ warnings: ['real', 42, 'also real', null] as any }),
    ).toEqual(['real', 'also real'])
  })

  it('deduplicates identical strings', () => {
    expect(
      extractResponsesWarnings({
        warnings: ['same', 'same', '  same  ', 'different'],
      }),
    ).toEqual(['same', 'different'])
  })

  it('extracts response.warning SSE event messages', () => {
    expect(
      extractResponsesWarnings({
        type: 'response.warning',
        code: 'empty_model_response',
        message: 'The model produced no visible response.',
      } as any),
    ).toEqual(['The model produced no visible response.'])
  })

  it('matches the exact server-emitted reasoning-only warning shape', () => {
    // This is the literal string emitted by
    // vmlx_engine/server.py:_chain_warnings_for_previous_response_id.
    // Pinning it here means the frontend test file is a contract surface
    // — if the server changes wording, this fails and forces UI review.
    const SERVER_MSG =
      'previous_response_id chained a response that produced reasoning only ' +
      '(no visible message, no tool calls). Chat continuity may be impaired ' +
      'and prefix-cache reuse may be lower than expected. Consider raising ' +
      'max_output_tokens or sending enable_thinking=false on the prior turn.'
    const w = extractResponsesWarnings({ warnings: [SERVER_MSG] })
    expect(w).toEqual([SERVER_MSG])
  })
})

describe('categorizeResponsesWarning', () => {
  it('categorizes the reasoning-only chain warning', () => {
    expect(
      categorizeResponsesWarning(
        'previous_response_id chained a response that produced reasoning only',
      ),
    ).toBe('chain_reasoning_only')
  })

  it('categorizes cache/prefix warnings', () => {
    expect(categorizeResponsesWarning('prefix-cache reuse may be lower')).toBe(
      'cache_alignment',
    )
    expect(categorizeResponsesWarning('cache may be invalidated')).toBe(
      'cache_alignment',
    )
  })

  it('falls back to "other" for unrecognized warnings', () => {
    expect(categorizeResponsesWarning('something completely unrelated')).toBe('other')
  })

  it('is case-insensitive', () => {
    expect(categorizeResponsesWarning('PREVIOUS_RESPONSE_ID was bad')).toBe(
      'chain_reasoning_only',
    )
  })
})

describe('Responses warnings panel wiring', () => {
  it('main chat IPC extracts response.completed warnings and forwards them on chat:complete', () => {
    const source = readFileSync(new URL('../src/main/ipc/chat.ts', import.meta.url), 'utf8')
    expect(source).toContain('import { extractResponsesWarnings } from "../../shared/responsesWarnings"')
    expect(source).toContain('responsesEventType === "response.warning"')
    expect(source).toContain('const eventWarnings = extractResponsesWarnings(parsed)')
    expect(source).toContain('const completedWarnings = extractResponsesWarnings(')
    expect(source).toContain('const chatWarnings = extractResponsesWarnings(parsed)')
    expect(source).toContain('assistantMessage.warningsJson = JSON.stringify(finalResponseWarnings)')
    expect(source).toContain('warnings: finalResponseWarnings || undefined')
  })

  it('database persists warnings_json with a safe additive migration', () => {
    const source = readFileSync(new URL('../src/main/database.ts', import.meta.url), 'utf8')
    expect(source).toContain('warningsJson?: string')
    expect(source).toContain('warnings_json TEXT')
    expect(source).toContain('ALTER TABLE messages ADD COLUMN warnings_json TEXT')
    expect(source).toContain('message.warningsJson')
    expect(source).toContain('warningsJson: row.warnings_json')
  })

  it('renderer keeps warnings separate from assistant content', () => {
    const chatInterface = readFileSync(new URL('../src/renderer/src/components/chat/ChatInterface.tsx', import.meta.url), 'utf8')
    const messageBubble = readFileSync(new URL('../src/renderer/src/components/chat/MessageBubble.tsx', import.meta.url), 'utf8')
    expect(chatInterface).toContain('const responseWarnings = extractResponsesWarnings({ warnings: data.warnings })')
    expect(chatInterface).toContain('warnings: responseWarnings ?? m.warnings')
    expect(messageBubble).toContain('{warnings && warnings.length > 0 && (')
    expect(messageBubble).toContain('warnings.map((warning, index) => (')
    expect(messageBubble).toContain('{warning}</span>')
  })
})
