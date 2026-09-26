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
  profile?: unknown
  quantization?: {
    profile?: unknown
    actual_bits?: unknown
    target_bits?: unknown
    bits?: unknown
    routed_avg_bits?: unknown
  }
}): string | undefined {
  const format = typeof config.format === 'string' ? config.format.toLowerCase() : ''
  const weightFormat =
    typeof config.weight_format === 'string' ? config.weight_format.toLowerCase() : ''
  const quant = config.quantization ?? {}
  const profile =
    typeof config.profile === 'string'
      ? config.profile.trim()
      : typeof quant.profile === 'string'
        ? quant.profile.trim()
        : undefined
  const actualBits = numberOrUndefined(quant.actual_bits)
  const targetBits = numberOrUndefined(quant.target_bits)
  const containerBits = numberOrUndefined(quant.bits)
  const routedAverageBits = numberOrUndefined(quant.routed_avg_bits)
  const explicitBits = actualBits ?? targetBits ?? containerBits

  // Canonical component name for the existing v2 serialized ABI.
  if (format === 'jangtq2' || weightFormat === 'jangtq2') return 'JANGH'

  if (weightFormat === 'mxtq' || format === 'mxtq' || format === 'jangtq') {
    const bits = explicitBits ?? jangtqBitsFromProfile(profile)
    const suffix = bits === 1 ? ', unsupported' : ''
    if (profile) return bits ? `${profile} (${bits}b${suffix})` : profile
    return bits ? `JANGTQ ${bits}-bit${suffix}` : 'JANGTQ'
  }

  if (
    format === 'jang' ||
    format === 'affine' ||
    format === 'jang_affine' ||
    format === 'jjqf' ||
    format === 'mxq' ||
    weightFormat === 'jang' ||
    weightFormat === 'affine' ||
    weightFormat === 'jang_affine' ||
    weightFormat === 'jjqf' ||
    weightFormat === 'mxq'
  ) {
    if (profile && actualBits != null) return `${profile} (${actualBits}b)`
    if (profile && routedAverageBits != null) {
      const routedLabel = Number(routedAverageBits.toFixed(2))
      return `${profile} (${routedLabel}b routed)`
    }
    if (profile) return explicitBits ? `${profile} (${explicitBits}b)` : profile
    return explicitBits ? `JANG ${explicitBits}-bit` : 'JANG'
  }

  // Any other weight_format that still declares a PROFILE.
  //
  // MXFP bundles land here: they carry a real jang_config.json, but with
  //   weight_format: "mlx", profile: "MXFP4", quantization: { bits: 4, ... }
  // and "mlx" matches neither branch above, so the label came back undefined.
  // MEASURED via the app's own detectConfig:
  //   Nemotron-Omni-Nano-MXFP4-CRACK -> quantizationLabel: null
  //   gemma-4-E4B-it-qat-JANG_4M     -> "JANG_4M"
  // In the session list that renders as NO badge at all, which is
  // indistinguishable from "unquantized" — for a whole format family, in a
  // list where the badge is how you tell a 4-bit bundle from a ternary one.
  //
  // Gated on `profile` on purpose: a bundle that declares no profile has
  // nothing truthful to show, and inventing a label would be worse than the
  // blank it replaces.
  if (profile) {
    if (actualBits != null) return `${profile} (${actualBits}b)`
    if (routedAverageBits != null) {
      return `${profile} (${Number(routedAverageBits.toFixed(2))}b routed)`
    }
    return explicitBits ? `${profile} (${explicitBits}b)` : profile
  }

  return undefined
}
