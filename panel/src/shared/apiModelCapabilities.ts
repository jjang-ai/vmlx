/** Bind display metadata to one session incarnation, not a reused port. */
export function apiCapabilityKey(s: {
  id: string; host: string; port: number; modelPath: string;
  pid?: number | null; status: string;
}): string {
  return JSON.stringify([s.id, s.host, s.port, s.modelPath, s.pid ?? null, s.status])
}

/** Runtime-advertised modalities only; MLLM lane selection is not a modality. */
export function apiCapabilityLabel(value: unknown): string | null {
  if (!value || typeof value !== 'object') return null
  const caps = value as { modalities?: unknown; media?: { runtime_modalities?: unknown } }
  const modes = caps.media?.runtime_modalities ?? caps.modalities
  if (!Array.isArray(modes)) return null
  const names = [...new Set(modes.filter((v): v is string => typeof v === 'string' && v.length > 0))]
  return names.length ? names.join(' · ') : null
}
