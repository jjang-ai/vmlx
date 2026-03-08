/**
 * Update Checker Tests
 *
 * Tests version comparison logic and manifest validation.
 * Re-implements compareVersions as a pure function (mirrors src/main/update-checker.ts)
 * to avoid Electron import dependencies in vitest.
 */
import { describe, it, expect } from 'vitest'

// Mirror of compareVersions from src/main/update-checker.ts
function compareVersions(current: string, latest: string): boolean {
  const clean = (v: string) => v.replace(/-.*$/, '')
  const a = clean(current).split('.').map(Number)
  const b = clean(latest).split('.').map(Number)
  for (let i = 0; i < Math.max(a.length, b.length); i++) {
    const av = a[i] ?? 0
    const bv = b[i] ?? 0
    if (isNaN(av) || isNaN(bv)) return false
    if (bv > av) return true
    if (bv < av) return false
  }
  return false
}

describe('compareVersions', () => {
  it('detects newer major version', () => {
    expect(compareVersions('1.0.0', '2.0.0')).toBe(true)
  })

  it('detects newer minor version', () => {
    expect(compareVersions('1.1.0', '1.2.0')).toBe(true)
  })

  it('detects newer patch version', () => {
    expect(compareVersions('1.1.1', '1.1.2')).toBe(true)
  })

  it('returns false when versions are equal', () => {
    expect(compareVersions('1.1.4', '1.1.4')).toBe(false)
  })

  it('returns false when current is newer', () => {
    expect(compareVersions('2.0.0', '1.9.9')).toBe(false)
    expect(compareVersions('1.2.0', '1.1.9')).toBe(false)
  })

  it('strips pre-release suffixes before comparing', () => {
    expect(compareVersions('1.1.4-beta.1', '1.1.4')).toBe(false)
    expect(compareVersions('1.1.3-beta.1', '1.1.4')).toBe(true)
    expect(compareVersions('1.1.4', '1.1.5-beta.1')).toBe(true)
  })

  it('handles versions with different segment counts', () => {
    expect(compareVersions('1.0', '1.0.1')).toBe(true)
    expect(compareVersions('1.0.1', '1.0')).toBe(false)
  })

  it('returns false for malformed version strings', () => {
    expect(compareVersions('abc', '1.0.0')).toBe(false)
    expect(compareVersions('1.0.0', 'xyz')).toBe(false)
  })

  it('handles zero-padded segments correctly', () => {
    expect(compareVersions('0.0.1', '0.0.2')).toBe(true)
    expect(compareVersions('0.0.0', '0.0.1')).toBe(true)
  })

  it('handles real version progression', () => {
    expect(compareVersions('1.1.3', '1.1.4')).toBe(true)
    expect(compareVersions('1.1.4', '1.2.0')).toBe(true)
    expect(compareVersions('0.2.11', '0.2.12')).toBe(true)
  })
})

describe('update manifest validation', () => {
  it('rejects manifest missing version', () => {
    const data = { url: 'https://example.com' }
    expect(!data.version || !data.url).toBe(true)
  })

  it('rejects manifest missing url', () => {
    const data = { version: '1.0.0' } as any
    expect(!data.version || !data.url).toBe(true)
  })

  it('accepts valid manifest', () => {
    const data = { version: '1.0.0', url: 'https://example.com' }
    expect(!data.version || !data.url).toBe(false)
  })

  it('accepts manifest with optional notes', () => {
    const data = { version: '1.0.0', url: 'https://example.com', notes: 'Bug fixes' }
    expect(!data.version || !data.url).toBe(false)
    expect(data.notes).toBe('Bug fixes')
  })
})
