import { describe, expect, it } from 'vitest'

import { buildContentSecurityPolicy } from '../src/main/csp'

describe('main-process content security policy', () => {
  it('keeps production renderer script execution strict', () => {
    const policy = buildContentSecurityPolicy({ devRenderer: false })

    expect(policy).toContain("script-src 'self';")
    expect(policy).not.toContain("script-src 'self' 'unsafe-inline';")
  })

  it('allows the Vite React dev preamble only for dev renderer sessions', () => {
    const policy = buildContentSecurityPolicy({ devRenderer: true })

    expect(policy).toContain("script-src 'self' 'unsafe-inline';")
    expect(policy).toContain("connect-src 'self' http://127.0.0.1:* http://localhost:*")
    expect(policy).toContain("https://hf-mirror.com https://*.hf-mirror.com")
    expect(policy).toContain("https://modelscope.cn https://*.modelscope.cn")
  })
})
