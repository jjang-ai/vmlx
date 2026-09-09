export interface ImageServerStartProgress {
  requestId: string
  label: string
  labelKey: string
  labelParams?: Record<string, string | number>
  notice: boolean
}

/** Subscribe before invoke; never let another queued window/request own this UI. */
export async function withImageServerStartProgress<T>(
  requestId: string,
  subscribe: (accept: (event: ImageServerStartProgress) => void) => () => void,
  invoke: () => Promise<T>,
  accept?: (event: ImageServerStartProgress) => void,
): Promise<T> {
  const unsubscribe = subscribe(event => {
    if (event?.requestId === requestId) accept?.(event)
  })
  try { return await invoke() }
  finally { unsubscribe() }
}
