/** Returns true if the session config has modelType === 'image' */
export function isImageSession(s: { config?: string }): boolean {
  if (!s.config) return false
  try { return JSON.parse(s.config).modelType === 'image' } catch { return false }
}
