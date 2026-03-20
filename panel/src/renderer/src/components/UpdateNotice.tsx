import { useState, useEffect } from 'react'
import { X } from 'lucide-react'

const CURRENT_NOTICE_VERSION = '1.2.7'

const NOTICE_CONTENT = {
  title: "What's New in MLX Studio",
  sections: [
    {
      heading: "JIT Sleep & Auto-Wake",
      lines: [
        "Idle models now automatically sleep to free GPU memory — light sleep clears caches instantly, deep sleep fully unloads the model.",
        "Models auto-wake on the next request (JIT) in 2-15 seconds. Configure per-session timeouts in Power Management settings.",
      ],
    },
    {
      heading: "Disk Cache & Stability",
      lines: [
        "Fixed disk cache writes that silently failed (ENOENT on rename). Orphaned temp files are now cleaned up on startup.",
        "Smoother token streaming (8ms throttle), fixed memory graph visibility, and 38+ bug fixes across the production audit.",
      ],
    },
    {
      heading: "JANG v2 Quantization",
      lines: [
        "JANG now supports Vision-Language (VL) models — load images in chat with JANG MoE models like Qwen3.5-35B and 122B.",
        "Re-download JANG models from HuggingFace (JANGQ-AI) to get the latest v2 format with VL support and mixed group-size fixes.",
      ],
    },
  ],
  footer: "Thank you for using MLX Studio and JANG. — Jinho Jang",
}

export function UpdateNotice() {
  const [visible, setVisible] = useState(false)

  useEffect(() => {
    const dismissed = window.api.settings?.get('notice_dismissed_version')
    if (dismissed instanceof Promise) {
      dismissed.then((v: any) => {
        if (v !== CURRENT_NOTICE_VERSION) setVisible(true)
      }).catch(() => setVisible(true))
    } else {
      if (dismissed !== CURRENT_NOTICE_VERSION) setVisible(true)
    }
  }, [])

  const dismiss = () => {
    setVisible(false)
    try {
      window.api.settings?.set('notice_dismissed_version', CURRENT_NOTICE_VERSION)
    } catch {}
  }

  if (!visible) return null

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/40 backdrop-blur-sm">
      <div className="bg-card border border-border rounded-xl shadow-2xl max-w-md w-full mx-4 p-5">
        <div className="flex items-start justify-between mb-3">
          <h2 className="text-sm font-bold">{NOTICE_CONTENT.title}</h2>
          <button onClick={dismiss} className="p-0.5 text-muted-foreground hover:text-foreground">
            <X className="h-4 w-4" />
          </button>
        </div>
        <div className="space-y-3 text-xs text-muted-foreground leading-relaxed max-h-[60vh] overflow-y-auto">
          {NOTICE_CONTENT.sections.map((section, si) => (
            <div key={si}>
              <p className="text-[11px] font-semibold text-foreground/80 mb-1">{section.heading}</p>
              {section.lines.map((line, li) => (
                <p key={li} className="ml-2">{line}</p>
              ))}
            </div>
          ))}
        </div>
        <p className="text-[10px] text-muted-foreground/60 mt-3 italic">{NOTICE_CONTENT.footer}</p>
        <button
          onClick={dismiss}
          className="mt-3 w-full py-1.5 bg-primary text-primary-foreground text-xs font-medium rounded-lg hover:bg-primary/90"
        >
          Got it — don't show until next update
        </button>
      </div>
    </div>
  )
}
