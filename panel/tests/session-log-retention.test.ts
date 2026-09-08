import { readFileSync } from 'node:fs'
import { describe, expect, it } from 'vitest'

// STOP-DESTROYS-SESSION-LOG-BUFFER regression: stopping a session must retain
// the log buffer (postmortems were impossible — every stop erased the server's
// stderr history). The buffer resets on the next start, and deleteSession
// still drops it entirely.
describe('session log retention across stop', () => {
  const source = readFileSync(
    new URL('../src/main/sessions.ts', import.meta.url),
    'utf8',
  )

  it('stopSession retains the log buffer with a stop marker', () => {
    expect(source).toContain(
      "this.pushLog(sessionId, '[INFO] Session stopped — log retained for postmortem until next start')",
    )
    // The stop path must NOT delete the buffer anymore.
    const stopBlock = source.slice(
      source.indexOf('async stopSession('),
      source.indexOf('async deleteSession('),
    )
    expect(stopBlock).not.toContain('this.logBuffers.delete(sessionId)')
  })

  it('a new start resets the buffer so runs do not blend', () => {
    const startBlock = source.slice(
      source.indexOf('private async _startSessionInner('),
      source.indexOf('private async _startSessionInner(') + 800,
    )
    expect(startBlock).toContain('this.logBuffers.delete(sessionId)')
  })

  it('deleteSession still drops the buffer entirely', () => {
    const delBlock = source.slice(
      source.indexOf('async deleteSession('),
      source.indexOf('async deleteSession(') + 1200,
    )
    expect(delBlock).toContain('this.logBuffers.delete(sessionId)')
  })
})

describe('session create dedupes only actual bundle paths', () => {
  const source = readFileSync(
    new URL('../src/main/sessions.ts', import.meta.url),
    'utf8',
  )

  it('create-path existing lookup uses filesystem identity, not basename', () => {
    expect(source).toContain(
      "db.getSessionByModelPath(modelPath) ||",
    )
    expect(source).toContain(
      "s => s.type !== 'remote' && sameLocalBundlePath(s.modelPath, modelPath)",
    )
  })
})
