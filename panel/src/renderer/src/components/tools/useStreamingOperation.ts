import { useState, useEffect, useRef, useCallback } from 'react'

const MAX_LOG_LINES = 2000

interface StreamingOperationResult {
  running: boolean
  logLines: string[]
  wasCancelled: boolean
  /** Start a streaming operation. Returns the IPC result and accumulated log lines. */
  start: (ipcCall: () => Promise<any>) => Promise<{ ipcResult: any; allLines: string[] }>
  cancel: () => Promise<void>
}

/**
 * Shared hook for ModelDoctor and ModelConverter streaming operations.
 * Handles mounted guards, event listener lifecycle, log accumulation, and cancel.
 *
 * @param onLogUpdate Optional callback called with accumulated lines on each log event (e.g., for parsing).
 */
export function useStreamingOperation(onLogUpdate?: (lines: string[]) => void): StreamingOperationResult {
  const [running, setRunning] = useState(false)
  const [logLines, setLogLines] = useState<string[]>([])
  const [wasCancelled, setWasCancelled] = useState(false)
  const mountedRef = useRef(true)
  const cleanupRef = useRef<(() => void) | null>(null)

  useEffect(() => {
    mountedRef.current = true
    return () => {
      mountedRef.current = false
      cleanupRef.current?.()
    }
  }, [])

  const start = useCallback(async (ipcCall: () => Promise<any>): Promise<{ ipcResult: any; allLines: string[] }> => {
    setRunning(true)
    setLogLines([])
    setWasCancelled(false)

    const allLines: string[] = []

    const unsubLog = window.api.developer.onLog((data: any) => {
      if (!mountedRef.current) return
      const lines = data.data.split('\n').filter((l: string) => l.length > 0)
      allLines.push(...lines)
      setLogLines(prev => {
        const updated = [...prev, ...lines].slice(-MAX_LOG_LINES)
        onLogUpdate?.(updated)
        return updated
      })
    })

    cleanupRef.current = () => { unsubLog() }

    let ipcResult: any
    try {
      ipcResult = await ipcCall()
    } catch {
      // IPC call itself threw
    } finally {
      unsubLog()
      cleanupRef.current = null
      if (mountedRef.current) {
        setRunning(false)
        if (ipcResult?.cancelled) setWasCancelled(true)
      }
    }
    return { ipcResult, allLines }
  }, [onLogUpdate])

  const cancel = useCallback(async () => {
    try {
      await window.api.developer.cancelOp()
    } catch { /* main process unavailable */ }
  }, [])

  return { running, logLines, wasCancelled, start, cancel }
}
