import { describe, expect, it } from 'vitest'
import { insertPastedText, LARGE_PASTE_CHARS } from '../src/renderer/src/components/chat/InputBox'

// A 370k-character paste took 44-46 s through Chromium's insertText path in the dev app (2026-10-17); large pastes are
// inserted through React state instead. The helper must behave like a paste at the selection.
describe('large text paste', () => {
  it('replaces the selection and puts the caret after the pasted text', () => {
    expect(insertPastedText('hello world', 6, 11, 'there')).toEqual({ value: 'hello there', caret: 11 })
    expect(insertPastedText('ab', 1, 1, 'XYZ')).toEqual({ value: 'aXYZb', caret: 4 })
    expect(insertPastedText('', 0, 0, 'doc')).toEqual({ value: 'doc', caret: 3 })
  })
  it('clamps a stale selection to the current value', () => {
    expect(insertPastedText('abc', 10, 20, 'Z')).toEqual({ value: 'abcZ', caret: 4 })
    expect(insertPastedText('abc', 2, 1, 'Z')).toEqual({ value: 'abZc', caret: 3 })
  })
  it('keeps a document-sized paste intact', () => {
    const doc = 'x'.repeat(LARGE_PASTE_CHARS * 20)
    const r = insertPastedText('Question: ', 10, 10, doc)
    expect(r.value.length).toBe(10 + doc.length)
    expect(r.caret).toBe(10 + doc.length)
  })
})
