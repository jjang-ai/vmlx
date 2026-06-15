#!/usr/bin/env node
/* Live-app UI automation driver for vMLX dev build (CDP).
 * Connects to the running Electron renderer over CDP and runs a sub-command.
 * Usage: node scripts/uidrv.cjs <cmd> [args...]
 *   inspect                 - screenshot + dump interactive elements & headings
 *   shot [path]             - screenshot only
 *   click "<text>"          - click first visible element whose text matches
 *   type "<sel>" "<text>"   - fill an input/textarea by selector
 *   eval "<js>"             - run JS in page, print JSON result
 */
const { chromium } = require('playwright-core')
const CDP = process.env.VMLX_CDP || 'http://127.0.0.1:9333'

async function getPage(browser) {
  // Electron exposes contexts; find the vMLX renderer page.
  for (const ctx of browser.contexts()) {
    for (const p of ctx.pages()) {
      const u = p.url()
      if (u.includes('5173') || u.includes('index.html') || u.startsWith('file:')) return p
    }
  }
  const ctx = browser.contexts()[0]
  return ctx.pages()[0]
}

async function main() {
  const [cmd, a1, a2] = process.argv.slice(2)
  const browser = await chromium.connectOverCDP(CDP)
  const page = await getPage(browser)
  if (!page) { console.error('NO_PAGE'); process.exit(2) }
  await page.waitForLoadState('domcontentloaded').catch(() => {})

  if (cmd === 'shot' || cmd === 'inspect') {
    const out = a1 || '/tmp/vmlx-ui.png'
    await page.screenshot({ path: out, fullPage: false }).catch(e => console.error('shot err', e.message))
    console.log('SHOT', out)
  }
  if (cmd === 'inspect') {
    const info = await page.evaluate(() => {
      const vis = (el) => {
        const r = el.getBoundingClientRect()
        const s = getComputedStyle(el)
        return r.width > 0 && r.height > 0 && s.visibility !== 'hidden' && s.display !== 'none'
      }
      const txt = (el) => (el.innerText || el.value || el.getAttribute('aria-label') || el.title || '').trim().replace(/\s+/g, ' ').slice(0, 60)
      const clickable = [...document.querySelectorAll('button,[role=button],a,[role=tab],input[type=checkbox],input[type=radio],select,label')]
        .filter(vis).map(el => ({ tag: el.tagName.toLowerCase(), type: el.getAttribute('type') || '', t: txt(el), checked: el.checked ?? null }))
        .filter(x => x.t || x.type)
      const inputs = [...document.querySelectorAll('input,textarea')].filter(vis).map(el => ({
        tag: el.tagName.toLowerCase(), type: el.type || '', ph: el.placeholder || '', name: el.name || '', val: (el.value || '').slice(0, 30)
      }))
      const headings = [...document.querySelectorAll('h1,h2,h3,[class*=title],[class*=header]')].filter(vis).map(txt).filter(Boolean).slice(0, 25)
      return { url: location.href, title: document.title, headings: [...new Set(headings)], clickable, inputs }
    })
    console.log(JSON.stringify(info, null, 2))
  }
  if (cmd === 'click') {
    const loc = page.getByText(a1, { exact: false }).first()
    await loc.click({ timeout: 5000 })
    console.log('CLICKED', a1)
  }
  if (cmd === 'type') {
    await page.fill(a1, a2, { timeout: 5000 })
    console.log('TYPED into', a1)
  }
  if (cmd === 'press') {
    await page.press(a1, a2, { timeout: 5000 })
    console.log('PRESSED', a2, 'in', a1)
  }
  if (cmd === 'send') {
    // fill the message textarea and submit with Enter
    await page.fill(a1 || 'textarea', a2)
    await page.press(a1 || 'textarea', 'Enter')
    console.log('SENT')
  }
  if (cmd === 'setfile') {
    // set a file input (a1=selector or 'auto' to pick the first file input) to path a2
    const sel = (a1 && a1 !== 'auto') ? a1 : 'input[type=file]'
    const inp = page.locator(sel).first()
    await inp.setInputFiles(a2)
    console.log('SETFILE', a2, '->', sel)
  }
  if (cmd === 'text') {
    const r = await page.evaluate(() => document.body.innerText.replace(/\n{2,}/g, '\n').trim())
    console.log(r)
  }
  if (cmd === 'eval') {
    const r = await page.evaluate(a1)
    console.log(JSON.stringify(r, null, 2))
  }
  await browser.close().catch(() => {})
}
main().catch(e => { console.error('DRV_ERR', e.message); process.exit(1) })
