import { describe, expect, it } from 'vitest'
import { resolveUserDataOverride, shouldAllowSecondaryInstance } from '../src/shared/userDataOverride'

describe('local proof user-data override', () => {
  it('prefers explicit launch args over env vars', () => {
    expect(
      resolveUserDataOverride(
        ['vMLX', '--vmlx-user-data-dir=/tmp/vmlx-proof-arg'],
        {
          VMLINUX_USER_DATA_DIR: '/tmp/vmlx-proof-env',
          VMLX_USER_DATA_DIR: '/tmp/vmlx-short-env',
        },
      ),
    ).toBe('/tmp/vmlx-proof-arg')
  })

  it('supports split args and the shorter env override', () => {
    expect(resolveUserDataOverride(['vMLX', '--user-data-dir', '/tmp/split'], {})).toBe('/tmp/split')
    expect(resolveUserDataOverride(['vMLX'], { VMLX_USER_DATA_DIR: '/tmp/env' })).toBe('/tmp/env')
  })

  it('ignores blank values', () => {
    expect(resolveUserDataOverride(['vMLX', '--vmlx-user-data-dir=  '], {})).toBe('')
    expect(resolveUserDataOverride(['vMLX'], { VMLINUX_USER_DATA_DIR: ' ' })).toBe('')
  })

  it('allows secondary instances only for explicit isolated proof launches', () => {
    expect(shouldAllowSecondaryInstance(['vMLX'], {})).toBe(false)
    expect(
      shouldAllowSecondaryInstance(['vMLX', '--vmlx-user-data-dir=/tmp/proof'], {}),
    ).toBe(false)
    expect(
      shouldAllowSecondaryInstance(['vMLX', '--vmlx-allow-secondary-instance'], {}),
    ).toBe(false)
    expect(
      shouldAllowSecondaryInstance(
        ['vMLX', '--vmlx-user-data-dir=/tmp/proof', '--vmlx-allow-secondary-instance'],
        {},
      ),
    ).toBe(true)
    expect(
      shouldAllowSecondaryInstance(['vMLX'], {
        VMLX_USER_DATA_DIR: '/tmp/proof',
        VMLX_ALLOW_SECONDARY_INSTANCE: '1',
      }),
    ).toBe(true)
  })
})
