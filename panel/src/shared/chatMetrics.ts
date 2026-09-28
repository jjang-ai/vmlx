export interface FinalDecodeTpsInput {
  cumulativeTps: number
  rollingTps: number[]
  lastRollingTps?: number
  maxCumulativeToRollingRatio?: number
}

export interface PrefillTpsInput {
  prefillUsage?: unknown
}

export interface ServerDecodePass {
  outputTokens: number
  decodeTokens: number
  decodeSeconds: number
  tokensPerSecond: number
}

export interface ServerDecodeSummary extends ServerDecodePass {}

/**
 * Parse vMLX's negotiated, server-authoritative decode timing extension.
 *
 * A stream chunk is not a token: the engine deliberately batches multiple
 * tokens behind --stream-interval, and tool-control output may not create a
 * visible delta at all. Renderer arrival timing therefore cannot be used as
 * the final model decode rate when this private local-engine receipt exists.
 */
export function parseServerDecodeUsage(usage: unknown): ServerDecodePass | undefined {
  if (!usage || typeof usage !== 'object') return undefined
  const record = usage as Record<string, unknown>
  const decode = record.vmlx_decode
  if (!decode || typeof decode !== 'object') return undefined
  const decodeRecord = decode as Record<string, unknown>

  const outputTokens = Number(record.output_tokens)
  const decodeTokens = Number(decodeRecord.tokens)
  const decodeSeconds = Number(decodeRecord.seconds)
  if (
    !Number.isFinite(outputTokens) || outputTokens <= 0 ||
    !Number.isFinite(decodeTokens) || decodeTokens <= 0 ||
    decodeTokens > outputTokens ||
    !Number.isFinite(decodeSeconds) || decodeSeconds <= 0
  ) return undefined

  return {
    outputTokens,
    decodeTokens,
    decodeSeconds,
    // Recompute from the receipt's numerator/denominator instead of trusting
    // a separately rounded rate field.
    tokensPerSecond: decodeTokens / decodeSeconds,
  }
}

/** Combine completed HTTP passes in one visible agent/tool exchange. */
export function summarizeServerDecodePasses(
  passes: Array<ServerDecodePass | undefined>,
): ServerDecodeSummary | undefined {
  const valid = passes.filter((pass): pass is ServerDecodePass => !!pass)
  if (valid.length === 0) return undefined

  const outputTokens = valid.reduce((sum, pass) => sum + pass.outputTokens, 0)
  const decodeTokens = valid.reduce((sum, pass) => sum + pass.decodeTokens, 0)
  const decodeSeconds = valid.reduce((sum, pass) => sum + pass.decodeSeconds, 0)
  if (decodeTokens <= 0 || decodeSeconds <= 0) return undefined

  return {
    outputTokens,
    decodeTokens,
    decodeSeconds,
    tokensPerSecond: decodeTokens / decodeSeconds,
  }
}

/** Engine-measured model prefill, never prompt tokens divided by TTFT. */
export function calculatePrefillTps({ prefillUsage }: PrefillTpsInput): string | undefined {
  if (!prefillUsage || typeof prefillUsage !== 'object') return undefined
  const receipt = prefillUsage as Record<string, unknown>
  if (receipt.scope !== 'model_prefill_and_prompt_state') return undefined
  const tokens = receipt.tokens
  const seconds = receipt.seconds
  if (typeof tokens !== 'number' || !Number.isSafeInteger(tokens) || tokens <= 0 ||
      typeof seconds !== 'number' || !Number.isFinite(seconds) || seconds <= 0) return undefined
  return (tokens / seconds).toFixed(1)
}

/**
 * Select a truthful final decode rate for a possibly multi-request agent turn.
 *
 * Cumulative delta timing represents every reasoning/tool iteration, but a
 * server-side buffered pass can make it look impossibly fast. The last rolling
 * rate avoids that burst but can represent only a short final-answer tail after
 * a pause. Use the median rolling rate as the representative control: retain
 * cumulative throughput when it agrees with the progressive stream, otherwise
 * reject the buffered burst.
 */
export function selectFinalDecodeTps({
  cumulativeTps,
  rollingTps,
  lastRollingTps = 0,
  maxCumulativeToRollingRatio = 1.5,
}: FinalDecodeTpsInput): number {
  const samples = rollingTps
    .filter(value => Number.isFinite(value) && value > 0)
    .sort((a, b) => a - b)

  const representativeRolling = samples.length > 0
    ? samples.length % 2 === 1
      ? samples[Math.floor(samples.length / 2)]
      : (samples[samples.length / 2 - 1] + samples[samples.length / 2]) / 2
    : Number.isFinite(lastRollingTps) && lastRollingTps > 0
      ? lastRollingTps
      : 0

  const validCumulative = Number.isFinite(cumulativeTps) && cumulativeTps > 0
    ? cumulativeTps
    : 0
  if (representativeRolling <= 0) return validCumulative
  if (validCumulative <= 0) return representativeRolling

  return validCumulative <= representativeRolling * maxCumulativeToRollingRatio
    ? validCumulative
    : representativeRolling
}
