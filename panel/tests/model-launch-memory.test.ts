import { mkdirSync, writeFileSync } from 'fs'
import { join } from 'path'
import { tmpdir } from 'os'
import { describe, expect, it } from 'vitest'
import {
  estimateModelFileBytes,
  estimateModelLaunchResidentBytes,
  estimateModelMemory,
  estimateMacReclaimableMemoryBytesFromVmStat,
  isLazyMmapJangBundle,
  modelLaunchReserveWarning,
  unsafeModelLaunchOverrideEnabled,
  unsafeModelLaunchReason,
} from '../src/main/modelLaunchMemory'

function makeModelDir(name: string): string {
  const dir = join(tmpdir(), `vmlx-model-launch-${name}-${Date.now()}-${Math.random().toString(16).slice(2)}`)
  mkdirSync(dir, { recursive: true })
  return dir
}

function writeBytes(path: string, bytes: number): void {
  writeFileSync(path, Buffer.alloc(bytes, 1))
}

describe('model launch memory admission', () => {
  it('recursively counts model files for launch estimates', () => {
    const dir = makeModelDir('recursive')
    mkdirSync(join(dir, 'sub'))
    writeBytes(join(dir, 'model.safetensors'), 11)
    writeBytes(join(dir, 'sub', 'extra.safetensors'), 7)

    expect(estimateModelFileBytes(dir)).toBe(18)
    expect(estimateModelMemory(dir)).toBe(Math.round(18 * 1.3))
  })

  it('treats JANG/JANGTQ bundles as lazy-mmap launch residents', () => {
    const dir = makeModelDir('jang')
    writeFileSync(join(dir, 'jang_config.json'), '{}')
    writeBytes(join(dir, 'weights.safetensors'), 100)

    expect(isLazyMmapJangBundle(dir)).toBe(true)
    expect(estimateModelLaunchResidentBytes(dir, 100, 1_000)).toBe(70)
  })

  it('treats plain non-JANG bundles as conservative full resident loads', () => {
    const dir = makeModelDir('plain')
    writeFileSync(join(dir, 'config.json'), JSON.stringify({ model_type: 'llama' }))
    writeBytes(join(dir, 'model.safetensors'), 100)

    expect(isLazyMmapJangBundle(dir)).toBe(false)
    expect(estimateModelLaunchResidentBytes(dir, 100, 1_000)).toBe(130)
  })

  it('does not hard-block when only the 16 GiB reserve is tight', () => {
    const gib = 1024 ** 3
    const launchResident = 73.2 * gib
    const available = 88.3 * gib

    expect(unsafeModelLaunchReason(launchResident, available, {})).toBeNull()
    expect(modelLaunchReserveWarning(launchResident, available)).toContain('16.0 GB system safety headroom')
  })

  it('still hard-blocks when estimated launch resident exceeds free RAM', () => {
    const gib = 1024 ** 3

    expect(unsafeModelLaunchReason(91 * gib, 88.3 * gib, {})).toContain('exceeds currently free RAM')
  })

  it('allows lazy-mmap launches when reclaimable macOS pages cover the freemem gap', () => {
    const gib = 1024 ** 3

    expect(
      unsafeModelLaunchReason(60.2 * gib, 51.3 * gib, {}, {
        lazyMmap: true,
        reclaimableBytes: 12 * gib,
        totalBytes: 128 * gib,
      }),
    ).toBeNull()
    expect(
      unsafeModelLaunchReason(60.2 * gib, 51.3 * gib, {}, {
        lazyMmap: false,
        reclaimableBytes: 12 * gib,
        totalBytes: 128 * gib,
      }),
    ).toContain('exceeds currently free RAM')
  })

  it('parses conservative reclaimable memory from macOS vm_stat output', () => {
    const vmStat = [
      'Mach Virtual Memory Statistics: (page size of 16384 bytes)',
      'Pages inactive:                            1499093.',
      'Pages purgeable:                             19462.',
      'Pages speculative:                         1746605.',
      'File-backed pages:                         5995950.',
    ].join('\n')

    expect(estimateMacReclaimableMemoryBytesFromVmStat(vmStat)).toBe((1_499_093 + 19_462) * 16_384)
  })

  it('accepts canonical and legacy unsafe launch override env names', () => {
    const gib = 1024 ** 3

    expect(unsafeModelLaunchOverrideEnabled({ VMLX_ALLOW_UNSAFE_MODEL_LAUNCH: '1' })).toBe(true)
    expect(unsafeModelLaunchOverrideEnabled({ VMLINUX_ALLOW_UNSAFE_MODEL_LAUNCH: '1' })).toBe(true)
    expect(unsafeModelLaunchReason(91 * gib, 88.3 * gib, { VMLX_ALLOW_UNSAFE_MODEL_LAUNCH: '1' })).toBeNull()
    expect(unsafeModelLaunchReason(91 * gib, 88.3 * gib, { VMLINUX_ALLOW_UNSAFE_MODEL_LAUNCH: '1' })).toBeNull()
  })
})
