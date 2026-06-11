import { readFileSync } from 'node:fs'
import { join } from 'node:path'
import { describe, expect, it } from 'vitest'

const source = readFileSync(
  join(__dirname, '..', 'scripts', 'live-real-ui-model-proof.mjs'),
  'utf-8',
)

describe('live real UI proof harness', () => {
  it('treats exact text and JSON prompts as strict assertions', () => {
    expect(source).toContain('reply with exactly this (?:text|json) and nothing else')
  })

  it('keeps existing exact visible final wording assertions', () => {
    expect(source).toContain('send visible final text exactly')
    expect(source).toContain('output visible final text exactly')
  })

  it('counts tool-first Responses argument deltas as streaming proof', () => {
    expect(source).toContain('responsesFunctionCallArgumentStreamingSeen')
    expect(source).toContain('response.function_call_arguments.delta')
    expect(source).toContain('response.function_call_arguments.done')
    expect(source).toContain('Responses function_call_arguments.delta')
    expect(source).toContain('Responses function_call_arguments.done')
  })
})
