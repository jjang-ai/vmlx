import { describe, expect, it } from 'vitest'
import { readFileSync } from 'fs'
import { join } from 'path'

describe('new chat targets a running session over a stopped pinned one', () => {
  it('App.tsx prefers the running session when the pinned one is not running', () => {
    const src = readFileSync(join(__dirname, '..', 'src', 'renderer', 'src', 'App.tsx'), 'utf8')
    expect(src).toContain("const explicitUsable = explicit && (explicit.status === 'running' || explicit.status === 'loading' || !running)")
    expect(src).toContain("const target = (explicitUsable ? explicit : null) || running || explicit || sessions[0]")
  })
})
