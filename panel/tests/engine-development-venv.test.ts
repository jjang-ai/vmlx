import { afterEach, describe, expect, it, vi } from 'vitest'
import { chmodSync, mkdirSync, mkdtempSync, rmSync, writeFileSync } from 'node:fs'
import { tmpdir } from 'node:os'
import { join } from 'node:path'

vi.mock('electron', () => ({
  app: {
    isPackaged: false,
    getAppPath: () => '/nonexistent/panel',
  },
}))

import { getDevelopmentProjectVenv } from '../src/main/engine-manager'

const roots: string[] = []

function makeProjectPython(script: string): { root: string; pythonPath: string } {
  const root = mkdtempSync(join(tmpdir(), 'vmlx-project-venv-'))
  roots.push(root)
  const binDir = join(root, '.venv', 'bin')
  const pythonPath = join(binDir, 'python3')
  mkdirSync(binDir, { recursive: true })
  writeFileSync(pythonPath, script)
  chmodSync(pythonPath, 0o755)
  return { root, pythonPath }
}

afterEach(() => {
  while (roots.length > 0) {
    rmSync(roots.pop()!, { recursive: true, force: true })
  }
})

describe('development project venv detection', () => {
  it('probes from the owning checkout rather than the panel or installed package directory', () => {
    const { root, pythonPath } = makeProjectPython(
      '#!/bin/sh\n[ -f .source-check ] || exit 1\nprintf "1.6.56\\n"\n',
    )
    writeFileSync(join(root, '.source-check'), 'owned checkout')
    expect(getDevelopmentProjectVenv(root)).toEqual({ pythonPath, version: '1.6.56' })
  })
  it('returns the same isolated Python and imported engine version used by session startup', () => {
    const { root, pythonPath } = makeProjectPython(
      '#!/bin/sh\nprintf "1.6.16\\n"\n',
    )

    expect(getDevelopmentProjectVenv(root)).toEqual({
      pythonPath,
      version: '1.6.16',
    })
  })

  it('rejects a project Python that cannot import a versioned engine', () => {
    const { root } = makeProjectPython('#!/bin/sh\nexit 1\n')

    expect(getDevelopmentProjectVenv(root)).toBeNull()
  })

  it('does not probe a development venv without a development source root', () => {
    expect(getDevelopmentProjectVenv(null)).toBeNull()
  })
})
