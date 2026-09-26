import { readFileSync } from 'node:fs'
import { describe, expect, it } from 'vitest'

const source = readFileSync('src/renderer/src/components/sessions/PerformancePanel.tsx', 'utf8')
const anchor = source.indexOf("label={t('sessions.performance.metalNa')}")
const begin = source.indexOf('value={', anchor) + 'value={'.length
const end = source.indexOf('\n                }', begin)
const render = new Function('health', 't', `return (${source.slice(begin, end)})`)

describe('Metal acceleration label', () => {
  it('describes TQ2 backends without claiming an observed NAX dispatch', () => {
    expect(render({ acceleration: { kernel_type: 'jangtq2_codebook', metal_na_capable: true,
      metal_na_active_on_host: false } }, (key: string) => key))
      .toBe('Custom decode; NAX/Steel prefill (route not observed)')
  })
  it('retains affine active and unsupported labels', () => {
    expect(render({ acceleration: { kernel_type: 'affine_quantized_matmul',
      metal_na_active_on_host: true } }, (key: string) => key))
      .toBe('sessions.performance.statusActive')
    expect(render({ acceleration: { kernel_type: 'full_precision_or_unknown',
      metal_na_capable: false } }, (key: string) => key))
      .toBe('sessions.performance.notApplicable')
  })
})
