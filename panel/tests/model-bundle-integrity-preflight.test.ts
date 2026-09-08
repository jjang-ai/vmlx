import { describe, expect, it } from 'vitest'
import {
  buildBundleIntegrityInvocation,
  parseBundleIntegrityReport,
} from '../src/main/model-bundle-integrity'

const okReport = {
  schema: 'vmlx-model-bundle-integrity-v1',
  status: 'ok',
  bundle: '/Volumes/Models/example',
  shards: 2,
  tensors: 91,
  misaligned_tensors: 7,
  alignment_contract: 'atomic_realign',
  repairs: [],
  warnings: [],
  cache_hit: true,
}

describe('model bundle integrity preflight', () => {
  it('binds development Python to the authoritative source tree', () => {
    const invocation = buildBundleIntegrityInvocation(
      {
        type: 'development',
        pythonPath: '/repo/.venv/bin/python',
        sourceRoot: '/repo',
      },
      '/Volumes/Models/model with spaces',
    )

    expect(invocation.command).toBe('/repo/.venv/bin/python')
    expect(invocation.args).toEqual([
      '-B',
      '-s',
      '-m',
      'vmlx_engine.cli',
      'bundle-check',
      '/Volumes/Models/model with spaces',
      '--json',
    ])
    expect(invocation.env.PYTHONPATH).toBe('/repo')
    expect(invocation.env.PYTHONNOUSERSITE).toBe('1')
  })

  it('uses a system entry point without a shell', () => {
    const invocation = buildBundleIntegrityInvocation(
      { type: 'system', binaryPath: '/opt/homebrew/bin/vmlx-engine' },
      '/model;not-a-shell-command',
    )
    expect(invocation.command).toBe('/opt/homebrew/bin/vmlx-engine')
    expect(invocation.args).toEqual([
      'bundle-check',
      '/model;not-a-shell-command',
      '--json',
    ])
  })

  it('accepts only the complete shared checker schema', () => {
    expect(parseBundleIntegrityReport(JSON.stringify(okReport))).toEqual(okReport)
    expect(() =>
      parseBundleIntegrityReport(JSON.stringify({ ...okReport, shards: undefined })),
    ).toThrow('incomplete bundle-integrity report')
  })

  it('surfaces a checker failure without treating it as malformed output', () => {
    expect(() =>
      parseBundleIntegrityReport(JSON.stringify({
        schema: 'vmlx-model-bundle-integrity-v1',
        status: 'error',
        error: 'model-00002-of-00002.safetensors is missing',
      })),
    ).toThrow('Model bundle integrity check failed: model-00002-of-00002.safetensors is missing')
  })
})
