const JANGTQ_PROFILE_BITS_RE = /^JANGTQ([124])(?:$|[_-])/i

function numberOrUndefined(value: unknown): number | undefined {
  return typeof value === 'number' && Number.isFinite(value) ? value : undefined
}

export function jangtqBitsFromProfile(profile: unknown): number | undefined {
  if (typeof profile !== 'string') return undefined
  const match = profile.trim().match(JANGTQ_PROFILE_BITS_RE)
  return match ? Number(match[1]) : undefined
}

export function formatJangQuantizationLabel(config: {
  format?: unknown
  weight_format?: unknown
  quantization?: {
    profile?: unknown
    actual_bits?: unknown
    target_bits?: unknown
    bits?: unknown
  }
}): string | undefined {
  const format = typeof config.format === 'string' ? config.format.toLowerCase() : ''
  const weightFormat =
    typeof config.weight_format === 'string' ? config.weight_format.toLowerCase() : ''
  const quant = config.quantization ?? {}
  const profile = typeof quant.profile === 'string' ? quant.profile.trim() : undefined
  const explicitBits =
    numberOrUndefined(quant.actual_bits) ??
    numberOrUndefined(quant.target_bits) ??
    numberOrUndefined(quant.bits)

  if (weightFormat === 'mxtq' || format === 'mxtq' || format === 'jangtq') {
    const bits = explicitBits ?? jangtqBitsFromProfile(profile)
    const suffix = bits === 1 ? ', unsupported' : ''
    if (profile) return bits ? `${profile} (${bits}b${suffix})` : profile
    return bits ? `JANGTQ ${bits}-bit${suffix}` : 'JANGTQ'
  }

  if (
    format === 'jang' ||
    format === 'jjqf' ||
    format === 'mxq' ||
    weightFormat === 'jang' ||
    weightFormat === 'jjqf' ||
    weightFormat === 'mxq'
  ) {
    if (profile) return explicitBits ? `${profile} (${explicitBits}b)` : profile
    return explicitBits ? `JANG ${explicitBits}-bit` : 'JANG'
  }

  return undefined
}
