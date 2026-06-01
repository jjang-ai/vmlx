export type ImageGenerationAbortReason = 'cancel' | 'timeout'

type ActiveGeneration = {
  controller: AbortController
  sessionId: string
  startTime: number
}

let activeGeneration: ActiveGeneration | null = null
let lastGenerationSessionId: string | null = null
const abortReasons = new WeakMap<AbortController, ImageGenerationAbortReason>()

export function beginImageGeneration(
  sessionId: string,
  controller: AbortController = new AbortController(),
): AbortController {
  activeGeneration = {
    controller,
    sessionId,
    startTime: Date.now(),
  }
  lastGenerationSessionId = sessionId
  return controller
}

export function getActiveImageGenerationController(): AbortController | null {
  return activeGeneration?.controller || null
}

export function markImageGenerationAbort(
  controller: AbortController,
  reason: ImageGenerationAbortReason,
): void {
  abortReasons.set(controller, reason)
}

export function classifyImageGenerationError(
  error: unknown,
  controller?: AbortController | null,
): string {
  const err = error as any
  const msg = String(err?.message || error)
  const code = String(err?.code || '')
  const cause = err?.cause
  const wrappedDisconnects = [
    cause,
    err?.reason,
    err?.error,
    err?.detail,
  ].filter(Boolean)
  const nestedErrors = Array.isArray(err?.errors) ? err.errors : []
  const reason = controller ? abortReasons.get(controller) : undefined

  if (reason === 'cancel') return 'Image generation cancelled.'
  if (reason === 'timeout') return 'Image generation timed out after 30 minutes.'

  const resetLike =
    code === 'ECONNRESET' ||
    code === 'EPIPE' ||
    code === 'ERR_STREAM_DESTROYED' ||
    code === 'ERR_STREAM_WRITE_AFTER_END' ||
    /EPIPE|socket hang up|ECONNRESET|write EPIPE|broken pipe|premature close|stream.*destroyed|write after end/i.test(msg) ||
    wrappedDisconnects.some((nested) => classifyImageGenerationError(nested, controller).startsWith('Image server connection lost')) ||
    nestedErrors.some((nested) => classifyImageGenerationError(nested, controller).startsWith('Image server connection lost'))
  return resetLike
    ? 'Image server connection lost. The model may have crashed, been stopped, or hit memory pressure. Check Logs and restart the image server.'
    : msg
}

export function finishImageGeneration(controller?: AbortController | null): void {
  if (!controller || activeGeneration?.controller === controller) {
    activeGeneration = null
  }
  if (controller) abortReasons.delete(controller)
}

export function clearImageGenerationAfterLocalAbort(
  controller?: AbortController | null,
): void {
  if (!controller || activeGeneration?.controller === controller) {
    activeGeneration = null
  }
}

export function clearImageGenerationSessionHistory(): void {
  lastGenerationSessionId = null
}

export function getImageGenerationStatus(): {
  generating: boolean
  startTime: number | null
  sessionId: string | null
} {
  return {
    generating: activeGeneration != null,
    startTime: activeGeneration?.startTime ?? null,
    sessionId: activeGeneration?.sessionId || lastGenerationSessionId,
  }
}

export function resetImageGenerationStateForTests(): void {
  activeGeneration = null
  lastGenerationSessionId = null
}
