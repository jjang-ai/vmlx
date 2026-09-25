import { readFileSync } from 'node:fs'
import { describe, expect, it } from 'vitest'

const source = readFileSync('src/main/ipc/chat.ts', 'utf8')
function deltaElapsed(now: number, startTime: number, fetchStartTime: number, firstTokenTime: number, generationMs: number) {
  const begin = source.indexOf('          const elapsed =')
  const end = source.indexOf('          // TTFT measured', begin)
  return new Function('now', 'startTime', 'fetchStartTime', 'firstTokenTime', 'generationMs', source.slice(begin, end) + '\nreturn elapsed')(now, startTime, fetchStartTime, firstTokenTime, generationMs)
}

describe('live turn elapsed', () => {
  it('includes first prefill in elapsed without changing generation duration', () => {
    expect(deltaElapsed(85000, 0, 0, 19000, 66000)).toBe(85)
  })
  it('retains turn elapsed across a follow-up HTTP pass reset', () => {
    expect(deltaElapsed(85000, 0, 80000, 81000, 66000)).toBe(85)
  })


  it('uses the same turn clock for heartbeat elapsed', () => {
    const anchor = source.indexOf('                    const _hbElapsed =')
    const begin = source.indexOf('                        elapsed:', anchor)
    const line = source.slice(begin, source.indexOf('\n', begin)).trim().replace(/^elapsed:\s*/, '').replace(/,$/, '')
    expect(new Function('now', 'startTime', 'fetchStartTime', `return ${line}`)(85000, 0, 80000)).toBe('85.0')
  })
})


describe('visible content flush before tool execution', () => {
  it('retains whole-turn elapsed when flushing visible pre-tool content', () => {
    const anchor = source.indexOf('// Flush accumulated content to renderer before blocking on tool execution')
    const begin = source.indexOf('                    elapsed:', anchor)
    expect(anchor).toBeGreaterThan(0)
    expect(begin).toBeGreaterThan(anchor)
    const line = source.slice(begin, source.indexOf('\n', begin)).trim().replace(/^elapsed:\s*/, '').replace(/,$/, '')
    const evaluate = new Function('Date', 'startTime', 'generationMs', `return ${line}`)
    expect(evaluate({now: () => 85000}, 0, 66000)).toBe('85.0')
    expect(evaluate({now: () => 90000}, 0, 66500)).toBe('90.0')
  })
})
