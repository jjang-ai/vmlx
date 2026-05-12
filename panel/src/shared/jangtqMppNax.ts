export type JangtqMppNaxMode = 'auto' | 'off' | 'on'

export const JANGTQ_MPP_NAX_SETTING_KEY = 'jangtq_mpp_nax_mode'

export function normalizeJangtqMppNaxMode(mode: unknown): JangtqMppNaxMode {
  const value = String(mode ?? 'auto').trim().toLowerCase()
  if (value === 'off' || value === '0' || value === 'false' || value === 'no') return 'off'
  if (value === 'on' || value === '1' || value === 'true' || value === 'yes') return 'on'
  return 'auto'
}
