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
    expect(delBlock).toContain('this.pendingBundlePreflightLogs.delete(sessionId)')
  })

  it('restores only the pending exact-bundle preflight after the fresh-run reset', () => {
    const start = source.slice(source.indexOf('private async _startSessionInner('),
      source.indexOf('private async _startSessionInner(') + 1400)
    expect(start).toContain('preflightLogs?.modelPath === session.modelPath')
    expect(start).toContain('this.logBuffers.set(sessionId, preflightLogs.lines)')
    expect(start.indexOf('this.logBuffers.delete(sessionId)')).toBeLessThan(
      start.indexOf('this.logBuffers.set(sessionId, preflightLogs.lines)'))
    expect(source).toContain('if (preflightLines.length) this.pendingBundlePreflightLogs.set')
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

describe('startup repair notice wiring', () => {
  it('translates the repair log event and scrolls only the log pane', () => {
    const source = readFileSync(new URL('../src/renderer/src/components/sessions/CreateSession.tsx', import.meta.url), 'utf8')
    expect(source).toContain('data.bundleRepairNotice === true')
    expect(source).toContain('data-vmlx-control="bundle-repair-notice"')
    expect(source).toContain('ref={logPanelRef}')
    expect(source).toContain('panel?.scrollTo({ top: panel.scrollHeight')
    expect(source).toContain('if (launching) launchPanelRef.current?.scrollTo({ top: 0 })')
    expect(source.slice(source.indexOf('// Launching state'))).toContain('ref={launchPanelRef}')
    expect(source).not.toContain('logEndRef')
  })
})
