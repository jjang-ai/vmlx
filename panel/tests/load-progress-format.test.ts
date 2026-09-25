import { describe, expect, it } from 'vitest'
import { formatResidentLoad } from '../src/renderer/src/components/sessions/loadProgressFormat'

describe('loading residency presentation', () => {
  it('uses decimal GB for measured RSS and its expected-residency denominator', () => {
    expect(formatResidentLoad({
      residentMb: 50_000_000_000 / 1048576,
      expectedResidentBytes: 100_000_000_000,
      residentPercent: 50,
    })).toBe('50.0 GB / 100.0 GB (50.0%)')
  })

})
