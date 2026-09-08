export interface SsdPoolSnapshot {
  root: string
  used: number
  cap: number
  capacityEvicted: number
}

export function readSsdPoolSnapshot(stats: any): SsdPoolSnapshot | null {
  const b = stats?.block_disk_cache?.global_budget
  const finite = (v: unknown): v is number => typeof v === 'number' && Number.isFinite(v) && v >= 0
  if (!b || b.accounted !== true || typeof b.root !== 'string' || !b.root
    || !finite(b.bytes_after) || !finite(b.max_size_bytes)) return null
  // Older engines have only a combined removal counter. Never reinterpret it
  // as capacity pressure: it includes manual clear and ordinary janitor work.
  const capacityEvicted = b.capacity_evicted_entries_total ?? 0
  if (!finite(capacityEvicted)) return null
  return { root: b.root, used: b.bytes_after, cap: b.max_size_bytes, capacityEvicted }
}

export function ssdPoolNoticeKind(current: SsdPoolSnapshot, previous: SsdPoolSnapshot | null): 'capacity' | 'evicted' | null {
  if (current.cap <= 0) return null
  if (current.used >= current.cap) return 'capacity'
  if (previous?.root === current.root && previous.cap === current.cap
    && current.capacityEvicted > previous.capacityEvicted) return 'evicted'
  return null
}
