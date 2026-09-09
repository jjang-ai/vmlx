import React from 'react'
import { renderToStaticMarkup } from 'react-dom/server'
import { describe, expect, it, vi } from 'vitest'
vi.mock('../src/renderer/src/i18n', () => ({ useTranslation: () => ({ t: (key: string) => key }) }))
import { ImageTopBar } from '../src/renderer/src/components/image/ImageTopBar'
import type { ImageServerStatus } from '../src/shared/imageCapabilities'
const noop = () => {}
function render(status: ImageServerStatus) {
  return renderToStaticMarkup(React.createElement(ImageTopBar, {
    model: 'qwen-image-edit', quantize: 8, status, port: 8000, mode: 'edit',
    onSettings: noop, onLogs: noop, onStop: noop, onWake: noop, onChangeModel: noop,
    sidebarCollapsed: false, onToggleSidebar: noop,
  }))
}
describe('image standby controls', () => {
  it('renders Sleeping with Wake and Stop, not Running', () => {
    const html = render('standby')
    expect(html).toContain('data-vmlx-state="standby"')
    expect(html).toContain('status.sleeping')
    expect(html).toContain('data-vmlx-control="image-wake"')
    expect(html).toContain('data-vmlx-control="image-stop"')
    expect(html).not.toContain('image.topbar.runningOnPort')
  })
  it('does not offer Wake for a running, loading or stopped model', () => {
    for (const status of ['running', 'starting', 'stopped', 'error'] as const) {
      expect(render(status)).not.toContain('data-vmlx-control="image-wake"')
    }
  })
})

