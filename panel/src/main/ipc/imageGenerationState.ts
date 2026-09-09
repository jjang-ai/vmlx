import type { ImageJobProgress } from '../../shared/imageJobProgress'
export type ImageGenerationAbortReason = 'cancel' | 'timeout'

type ActiveGeneration = {
  controller: AbortController
  sessionId: string
  startTime: number
  serverSessionId?: string | null
  progress?: ImageJobProgress
  logTails?: Record<string, string>
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

export function bindImageGenerationRequest(controller: AbortController, serverSessionId: string | null, requestId: string): void {
  if (activeGeneration?.controller !== controller) return
  activeGeneration.serverSessionId = serverSessionId
  activeGeneration.progress = { requestId, phase: 'waiting' }
  activeGeneration.logTails = {}
}

export function recordImageGenerationLog(serverSessionId: string, data: string, stream: 'stdout' | 'stderr' | 'client' = 'client'): void {
  const active = activeGeneration
  if (!active?.progress || active.serverSessionId !== serverSessionId) return
  const tails = active.logTails ??= {}
  const lines = ((tails[stream] || '') + data).split('\n')
  tails[stream] = lines.pop()!.slice(-65536)
  for (const line of lines) {
    const match = line.match(/\b(IMAGEJOB|IMAGECLIENT) (\{.*\})\s*$/)
    if (!match) continue
    let event: any
    try { event = JSON.parse(match[2]) } catch { continue }
    const requestId = match[1] === 'IMAGEJOB' ? event.request_id : event.client_job_id
    if (requestId !== active.progress.requestId) continue
    if (match[1] === 'IMAGECLIENT') {
      if (event.phase === 'saving_outputs') active.progress = { requestId, phase: 'saving' }
      continue
    }
    if (typeof event.job_id !== 'string' || !event.job_id) continue
    const phases: Record<string, ImageJobProgress['phase']> = {
      model_call_started: 'preparing', before_denoise_loop: 'preparing',
      denoise_checkpoint: 'denoising', after_denoise_loop: 'rendering',
      model_call_returned: 'rendering', encoding_png: 'encoding',
    }
    const phase = phases[event.phase]
    if (!phase) continue
    const progress: ImageJobProgress = { requestId, jobId: event.job_id, phase }
    if (Number.isInteger(event.requested_steps) && event.requested_steps > 0) progress.totalSteps = event.requested_steps
    if (phase === 'denoising' && Number.isInteger(event.step_index) && event.step_index >= 0 && progress.totalSteps && event.step_index < progress.totalSteps) progress.stepIndex = event.step_index
    active.progress = progress
  }
}

export function markImageGenerationAbort(
  controller: AbortController,
  reason: ImageGenerationAbortReason,
): void {
  abortReasons.set(controller, reason)
}

export function clearImageGenerationAbortReason(controller: AbortController): void {
  abortReasons.delete(controller)
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
  cancelling: boolean
  startTime: number | null
  sessionId: string | null
  progress: ImageJobProgress | null
} {
  return {
    generating: activeGeneration != null,
    cancelling: !!activeGeneration && abortReasons.get(activeGeneration.controller) === 'cancel',
    startTime: activeGeneration?.startTime ?? null,
    sessionId: activeGeneration?.sessionId || lastGenerationSessionId,
    progress: activeGeneration?.progress ? { ...activeGeneration.progress } : null,
  }
}

export function resetImageGenerationStateForTests(): void {
  activeGeneration = null
  lastGenerationSessionId = null
}
