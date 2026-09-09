/** Discover an image engine even when its Server-tab launch finishes after mount.
 * Subscribe before the snapshot and coalesce readiness events during that read.
 * Stop after adopting one owner; later manual selections retain their authority.
 */
export function observeImageServerDiscovery<T>(options: {
  read: () => Promise<T | null | undefined>
  onReady: (callback: () => void) => () => void
  isCurrent: () => boolean
  accept: (server: T) => void | Promise<void>
  onError: (error: unknown) => void
}): () => void {
  let disposed = false
  let found = false
  let busy = false
  let retry = false
  const current = () => !disposed && options.isCurrent()
  const discover = () => {
    if (!current() || found) return
    retry = true
    if (busy) return
    busy = true
    void (async () => {
      try {
        do {
          retry = false
          const server = await options.read()
          if (!current()) return
          if (server) {
            found = true
            await options.accept(server)
          }
        } while (retry && !found && current())
      } catch (error) {
        if (current()) options.onError(error)
      } finally {
        busy = false
      }
    })()
  }
  const unsubscribe = options.onReady(discover)
  discover()
  return () => { disposed = true; unsubscribe() }
}
