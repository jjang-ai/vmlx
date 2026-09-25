/**
 * Shared formatters for the model-load progress readouts.
 *
 * SessionCard (the server tab) and SessionView (the chat header) render the
 * SAME load progress from the SAME `useSessionsContext().loadProgress` entry,
 * and each used to carry a byte-identical private copy of these two functions.
 * That is how they drift: a fix applied to one surface silently leaves the
 * other showing something different for the same load. This module exists so
 * there is exactly one definition to change.
 */

export interface LoadProgressLike {
  residentMb?: number
  modelBytes?: number
  expectedResidentBytes?: number
  residentPercent?: number
}

export function formatModelBytes(bytes?: number): string | null {
  if (!bytes || bytes <= 0) return null
  return `${(bytes / 1e9).toFixed(1)} GB`
}

export function formatResidentLoad(progress?: LoadProgressLike): string | null {
  if (!progress?.residentMb || progress.residentMb <= 0) return null
  // The monitor emits MiB; convert back to bytes before formatting decimal GB.
  const resident = formatModelBytes(progress.residentMb * 1048576)!
  // residentPercent is normalized against the family's EXPECTED resident
  // bytes, so the denominator shown must be the same quantity — dividing the
  // displayed pair by bundle size would compare the measurement against a
  // different denominator from the estimated residency used for the percent.
  const total = formatModelBytes(progress.expectedResidentBytes ?? progress.modelBytes)
  const pct = progress.residentPercent != null ? ` (${progress.residentPercent.toFixed(1)}%)` : ''
  return total ? `${resident} / ${total}${pct}` : `${resident}${pct}`
}
