export interface SsdPoolSnapshot {
  root: string
  used: number
  cap: number
  evicted: number
}

export function readSsdPoolSnapshot(stats: any): SsdPoolSnapshot | null {
  const b = stats?.block_disk_cache?.global_budget
  const finite = (v: unknown): v is number => typeof v === 'number' && Number.isFinite(v) && v >= 0
  if (!b || b.accounted !== true || typeof b.root !== 'string' || !b.root
    || !finite(b.bytes_after) || !finite(b.max_size_bytes) || !finite(b.evicted_entries_total)) return null
  return { root: b.root, used: b.bytes_after, cap: b.max_size_bytes, evicted: b.evicted_entries_total }
}

export function ssdPoolNoticeKind(current: SsdPoolSnapshot, previous: SsdPoolSnapshot | null): 'capacity' | 'evicted' | null {
  if (current.cap > 0 && current.used >= current.cap * 0.95) return 'capacity'
  if (previous?.root === current.root && current.evicted > previous.evicted) return 'evicted'
  return null
}
