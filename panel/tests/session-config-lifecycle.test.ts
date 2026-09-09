import { describe, expect, it } from 'vitest'
import { readFileSync } from 'node:fs'
import { hasLiveLocalSession, planSessionConfigSave } from '../src/shared/sessionConfigLifecycle'
const restart = new Set(['host', 'port', 'apiKey', 'logLevel', 'maxTokens', 'mcpDisabledTools'])
const effective = { host: '127.0.0.1', port: 8013, apiKey: 'old-fixture', logLevel: 'INFO', maxTokens: 4096, autoSleepEnabled: true }
const desired = { ...effective, port: 8014, apiKey: 'new-fixture', logLevel: 'DEBUG', autoSleepEnabled: false }
describe('saved next-start configuration is not a live endpoint', () => {
  it('labels the effective endpoint separately from edited future fields', () => {
    const src = readFileSync('src/renderer/src/components/sessions/SessionSettings.tsx', 'utf8')
    expect(src).toContain('data-vmlx-control="session-effective-endpoint">{session.host}:{session.port}')
  })
  it.each(['running', 'loading', 'standby'])('stages launch changes for %s without changing the active socket/key', status => {
    const x = planSessionConfigSave({ status, type: 'local' }, effective, desired, restart)
    expect(x.restartRequired).toBe(true)
    expect(x.config).toEqual({ ...effective, autoSleepEnabled: false })
    expect(x.pendingConfig).toEqual(desired)
    expect(x.changedKeys).toEqual(['port', 'apiKey', 'logLevel'])
    expect(effective.port).toBe(8013)
    expect(desired.port).toBe(8014)
  })
  it.each(['stopped', 'error'])('applies changes directly when %s', status => {
    expect(planSessionConfigSave({ status }, effective, desired, restart))
      .toMatchObject({ config: desired, pendingConfig: null, restartRequired: false })
  })
  it('does not invent a local engine restart for a remote connection', () => {
    expect(planSessionConfigSave({ status: 'running', type: 'remote' }, effective, desired, restart))
      .toMatchObject({ config: desired, pendingConfig: null, restartRequired: false })
  })
  it('keeps cleared Auto values pending without changing the running explicit cap', () => {
    const next: Record<string, unknown> = { ...effective }; delete next.maxTokens
    const x = planSessionConfigSave({ status: 'standby' }, effective, next, restart)
    expect(x.config.maxTokens).toBe(4096)
    expect(x.pendingConfig).not.toHaveProperty('maxTokens')
    expect(x.changedKeys).toEqual(['maxTokens'])
  })
  it('a second Save retains restart-required status and earlier pending edits', () => {
    const first = planSessionConfigSave({ status: 'standby' }, effective, desired, restart)
    const x = planSessionConfigSave({ status: 'standby' }, first.config, { ...first.pendingConfig, maxTokens: 8192 }, restart)
    expect(x.config.port).toBe(8013)
    expect(x.config.maxTokens).toBe(4096)
    expect(x.pendingConfig).toMatchObject({ port: 8014, maxTokens: 8192 })
    expect(x.restartRequired).toBe(true)
  })
  it('reverting future launch values clears the pending snapshot', () => {
    expect(planSessionConfigSave({ status: 'standby' }, effective, { ...effective }, restart))
      .toMatchObject({ pendingConfig: null, restartRequired: false })
  })
  it('does not stage equivalent array values or ordinary live power controls', () => {
    const old = { ...effective, mcpDisabledTools: ['a'] }
    const x = planSessionConfigSave({ status: 'running' }, old, { ...old, mcpDisabledTools: ['a'], autoSleepEnabled: false }, restart)
    expect(x.restartRequired).toBe(false)
    expect(x.config.autoSleepEnabled).toBe(false)
  })
  it.each(['running', 'loading', 'standby'])('the two settings surfaces recognize %s as a live local process', status => {
    expect(hasLiveLocalSession({ status })).toBe(true)
  })
  it('uses the shared planner and applies the persisted snapshot only after the process guard', () => {
    const main = readFileSync('src/main/sessions.ts', 'utf8')
    const start = main.slice(main.indexOf('private async _startSessionInner'), main.indexOf('async stopSession'))
    expect(start.indexOf('db.applyPendingSessionConfig(sessionId)')).toBeGreaterThan(start.indexOf("throw new Error('Session is already running')"))
    expect(main).toContain('planSessionConfigSave(session, effectiveConfig, merged, SessionManager.RESTART_REQUIRED_KEYS)')
    for (const file of ['SessionSettings', 'ServerSettingsDrawer']) {
      const src = readFileSync('src/renderer/src/components/sessions/' + file + '.tsx', 'utf8')
      expect(src).toContain('hasLiveLocalSession(')
      expect(src).toMatch(/pendingConfig \|\| (?:session|s).config/)
    }
    const db = readFileSync('src/main/database.ts', 'utf8')
    expect(db).toContain('ALTER TABLE sessions ADD COLUMN pending_config TEXT')
    expect(db).toContain('pendingConfig: row.pending_config ?? null')
    expect(db).toContain('config: session.pendingConfig, pendingConfig: null, host: config.host, port: config.port')
  })
  it('does not advertise LLM SSD caching or a hardcoded host for image servers', () => {
    const form = readFileSync('src/renderer/src/components/sessions/SessionConfigForm.tsx', 'utf8')
    expect(form).toContain("{!isImage && <InfoNote text={t('sessions.config.ramCacheTradeoffNotice')} />}")
    const image = readFileSync('src/renderer/src/components/image/ImageSettings.tsx', 'utf8')
    expect(image).not.toContain("t('image.settings.hostLocalhost')")
    expect(image).not.toContain("t('image.settings.portAutoAssigned')")
  })
})
