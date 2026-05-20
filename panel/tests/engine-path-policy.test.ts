import { readFileSync } from 'node:fs'
import { describe, expect, it } from 'vitest'

describe('engine path policy', () => {
  it('prefers the repo project venv before stale user/system vmlx-engine binaries in dev mode', () => {
    const source = readFileSync('src/main/sessions.ts', 'utf8')
    const findEnginePath = source.slice(
      source.indexOf('findEnginePath(): EnginePath | null'),
      source.indexOf('  private async findAvailablePort'),
    )

    const projectVenvIndex = findEnginePath.indexOf('Development builds must exercise the source tree')
    const systemSearchIndex = findEnginePath.indexOf('// System binary search')
    const staleSystemIndex = findEnginePath.indexOf("join(home, '.local', 'bin', 'vmlx-engine')")

    expect(projectVenvIndex).toBeGreaterThanOrEqual(0)
    expect(systemSearchIndex).toBeGreaterThanOrEqual(0)
    expect(staleSystemIndex).toBeGreaterThanOrEqual(0)
    expect(projectVenvIndex).toBeLessThan(systemSearchIndex)
    expect(projectVenvIndex).toBeLessThan(staleSystemIndex)
  })

  it('uses the published vmlx package name for PyPI installs while preserving vmlx-engine entrypoint detection', () => {
    const engineManager = readFileSync('src/main/engine-manager.ts', 'utf8')
    const createSession = readFileSync('src/renderer/src/components/sessions/CreateSession.tsx', 'utf8')

    expect(engineManager).toContain("const PYPI_PACKAGE_NAME = 'vmlx'")
    expect(engineManager).toContain("const ENTRY_POINT_NAMES = ['vmlx-engine', 'vmlx-serve', 'vmlx']")
    expect(engineManager).toContain("const pkg = bundledSource || PYPI_PACKAGE_NAME")
    expect(engineManager).toContain("['tool', 'upgrade', PYPI_PACKAGE_NAME]")
    expect(engineManager).not.toContain("const pkg = bundledSource || 'vmlx-engine'")
    expect(engineManager).not.toContain("['tool', 'upgrade', 'vmlx-engine']")

    expect(createSession).toContain('uv tool install vmlx')
    expect(createSession).toContain('pip3 install vmlx')
    expect(createSession).not.toContain('uv tool install vmlx-engine')
    expect(createSession).not.toContain('pip3 install vmlx-engine')
  })
})
