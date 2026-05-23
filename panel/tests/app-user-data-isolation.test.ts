import { readFileSync } from 'node:fs'
import { resolve } from 'node:path'
import { describe, expect, it } from 'vitest'

import {
  resolveUserDataDirOverride,
} from '../src/main/user-data-dir'

describe('app user-data isolation bootstrap', () => {
  const source = () =>
    readFileSync(resolve(process.cwd(), 'src/main/index.ts'), 'utf8')

  it('applies --vmlx-user-data-dir before taking the single-instance lock', () => {
    const main = source()
    const bootstrapIndex = main.indexOf("import './user-data-dir'")
    const databaseIndex = main.indexOf("from './database'")
    const lockIndex = main.indexOf('app.requestSingleInstanceLock()')

    expect(bootstrapIndex).toBeGreaterThanOrEqual(0)
    expect(databaseIndex).toBeGreaterThanOrEqual(0)
    expect(lockIndex).toBeGreaterThanOrEqual(0)
    expect(bootstrapIndex).toBeLessThan(databaseIndex)
    expect(bootstrapIndex).toBeLessThan(lockIndex)
  })

  it('supports environment override for non-UI packaged smoke tests', () => {
    expect(resolveUserDataDirOverride(['vMLX'], { VMLX_USER_DATA_DIR: 'build/user-data' })).toMatch(
      /build\/user-data$/,
    )
    expect(resolveUserDataDirOverride(['vMLX'], { VMLINUX_USER_DATA_DIR: 'build/legacy-user-data' })).toMatch(
      /build\/legacy-user-data$/,
    )
  })

  it('supports --vmlx-user-data-dir forms for repo-local dev app launches', () => {
    expect(resolveUserDataDirOverride(['vMLX', '--vmlx-user-data-dir=/tmp/vmlx-a'], {})).toBe(
      '/tmp/vmlx-a',
    )
    expect(resolveUserDataDirOverride(['vMLX', '--vmlx-user-data-dir', '/tmp/vmlx-b'], {})).toBe(
      '/tmp/vmlx-b',
    )
  })
})
