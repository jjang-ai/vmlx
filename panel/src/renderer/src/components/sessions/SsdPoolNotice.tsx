import { useEffect, useRef, useState } from 'react'
import { AlertTriangle, X } from 'lucide-react'
import { useTranslation } from '../../i18n'
import { Modal } from '../ui/Modal'
import { formatCacheStorageBytes } from './CachePanel'
import { readSsdPoolSnapshot, ssdPoolNoticeKind, type SsdPoolSnapshot } from './ssdPoolNoticeState'

/** Mount with a session/PID key: no dismissals or late replies cross engines. */
export function SsdPoolNotice({ sessionId, host, port, pid }: { sessionId: string; host: string; port: number; pid?: number }) {
  const { t } = useTranslation()
  const [snapshot, setSnapshot] = useState<SsdPoolSnapshot | null>(null)
  const [notice, setNotice] = useState<{ key: string; kind: 'capacity' | 'evicted' } | null>(null)
  const [dismissed, setDismissed] = useState('')
  const [confirm, setConfirm] = useState(false)
  const [confirmedRoot, setConfirmedRoot] = useState('')
  const [busy, setBusy] = useState(false)
  const [message, setMessage] = useState('')
  const alive = useRef(false)
  const action = useRef(false)
  const previous = useRef<SsdPoolSnapshot | null>(null)

  useEffect(() => {
    alive.current = true
    const unsubscribe = window.api.sessions.onHealth((data: any) => {
        if (data.sessionId === sessionId && data.port === port && data.enginePid === pid && !action.current) {
          const value = readSsdPoolSnapshot({ block_disk_cache: { global_budget: data.ssdPool } })
          setSnapshot(value)
          if (value) {
            const kind = ssdPoolNoticeKind(value, previous.current)
            if (kind) setNotice({ kind, key: `${value.root}:${value.cap}:${value.evicted}:${kind}` })
            else if (previous.current?.root !== value.root) setNotice(null)
            previous.current = value
          } else setNotice(null)
        }
    })
    return () => { alive.current = false; unsubscribe() }
  }, [sessionId, host, port, pid])

  const clear = async () => {
    if (action.current || !snapshot) return
    action.current = true; setBusy(true); setMessage('')
    const root = confirmedRoot
    try {
      if (!root || root !== snapshot.root) throw new Error(t('ssdPool.unknownResult'))
      const result = await window.api.cache.clear('ssd_pool', { host, port }, sessionId, { root, pid })
      if (!alive.current) return
      if (result?.status !== 'eligible_cleared' || result.root !== root
          || !Number.isFinite(result.freed_bytes) || !Number.isFinite(result.remaining_bytes)) {
        throw new Error(t('ssdPool.unknownResult'))
      }
      setMessage(t('ssdPool.result', {
        freed: formatCacheStorageBytes(result.freed_bytes),
        remaining: formatCacheStorageBytes(result.remaining_bytes),
      }))
      setConfirm(false)
      setSnapshot(s => s ? { ...s, used: result.remaining_bytes } : s)
    } catch (error) {
      if (alive.current) setMessage(t('ssdPool.failed', { detail: String(error) }))
    } finally {
      action.current = false
      if (alive.current) setBusy(false)
    }
  }

  if (!snapshot || !notice || (dismissed === notice.key && !confirm)) return null
  return <>
    <div data-vmlx-control="ssd-pool-notice" role="status" className="flex items-start gap-2 px-4 py-2 bg-amber-500/10 border-b border-amber-500/20 text-xs flex-shrink-0">
      <AlertTriangle className="h-4 w-4 text-amber-400 flex-shrink-0" />
      <div className="flex-1 min-w-0">
        <p>{t(notice.kind === 'capacity' ? 'ssdPool.capacity' : 'ssdPool.evicted', {
          used: formatCacheStorageBytes(snapshot.used),
          cap: snapshot.cap > 0 ? formatCacheStorageBytes(snapshot.cap) : t('sessions.cache.unlimited'),
        })}</p>
        <p className="text-muted-foreground break-all">{snapshot.root}</p>
        {message && <p data-vmlx-control="ssd-pool-result">{message}</p>}
      </div>
      <button data-vmlx-control="ssd-pool-clear" className="underline whitespace-nowrap" disabled={busy}
        onClick={() => { setMessage(''); setConfirmedRoot(snapshot.root); setConfirm(true) }}>{t('ssdPool.clear')}</button>
      <button data-vmlx-control="ssd-pool-dismiss" disabled={busy} aria-label={t('common.dismiss')}
        onClick={() => setDismissed(notice.key)}><X className="h-4 w-4" /></button>
    </div>
    {confirm && <Modal title={t('ssdPool.clear')} onClose={() => { if (!busy) setConfirm(false) }} className="max-w-lg">
      <p className="text-sm mb-3">{t('ssdPool.confirm')}</p>
      <p className="text-xs break-all mb-3">{confirmedRoot}</p>
      {message && <p role="alert" className="text-sm mb-3">{message}</p>}
      <div className="flex gap-3 justify-end">
        <button data-vmlx-control="ssd-pool-cancel" disabled={busy} onClick={() => setConfirm(false)}>{t('common.cancel')}</button>
        <button data-vmlx-control="ssd-pool-confirm" disabled={busy} onClick={() => void clear()}>
          {t(busy ? 'ssdPool.clearing' : 'ssdPool.clear')}
        </button>
      </div>
    </Modal>}
  </>
}
