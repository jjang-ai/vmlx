import { describe, expect, it } from 'vitest'
import { readFileSync, readdirSync, statSync, writeFileSync, existsSync } from 'fs'
import { join } from 'path'
import ts from 'typescript'

/**
 * Hard-coded JSX text is what a translator never sees. This scan lists every
 * JsxText node (and literal placeholder/title/aria-label/alt attributes) with
 * at least two letters, outside an allowlist of units/brands/symbols, and
 * compares it with a pinned baseline. New entries fail; removing entries is
 * fine (regenerate the baseline with I18N_JSX_BASELINE=write).
 */
const ROOT = join(__dirname, '..', 'src', 'renderer', 'src')
const BASELINE = join(__dirname, 'fixtures', 'i18n-jsx-text-baseline.json')
const ALLOW = /^(?:[\s\d.,:;()%×x\-–—+\/|·•→←…'"“”‘’&#!?]*|GB|MB|KB|TB|ms|s|tok\/s|fps|px|bit|-bit|\d+-bit|mlx\.studio|GitHub|JANG|MLX|API|CDP|URL|OpenAI|Anthropic|Ollama|Claude|Codex|Hermes|TTFT|TPS|PP\/s|PID|JSON|HF|vMLX|mflux|MTP|KV|SSD|RAM|CPU|GPU|Q\d|D\d|AR|N\/A|OK|curl|Python|JavaScript|npm|npx|pip|uv|Electron|macOS|Metal|Tailscale|ID|UI|CLI|HTTP|HTTPS|ws|LAN|IP|Ctrl|Cmd|Shift|Enter|Esc)$/i

function walk(dir: string, out: string[] = []): string[] {
  for (const name of readdirSync(dir)) {
    const p = join(dir, name)
    if (statSync(p).isDirectory()) walk(p, out)
    else if (name.endsWith('.tsx') && !name.includes('.test.')) out.push(p)
  }
  return out
}
function scan(): string[] {
  const found = new Set<string>()
  for (const file of walk(ROOT)) {
    const src = readFileSync(file, 'utf8')
    const sf = ts.createSourceFile(file, src, ts.ScriptTarget.Latest, true, ts.ScriptKind.TSX)
    const rel = file.replace(ROOT + '/', '')
    const visit = (n: ts.Node) => {
      if (ts.isJsxText(n)) {
        const text = n.getText().replace(/\s+/g, ' ').trim()
        if (/[A-Za-z]{2,}/.test(text) && !ALLOW.test(text)) found.add(`${rel}: ${text}`)
      }
      if (ts.isJsxAttribute(n) && n.initializer && ts.isStringLiteral(n.initializer)) {
        const name = n.name.getText()
        if (['placeholder', 'title', 'aria-label', 'alt', 'label'].includes(name)) {
          const text = n.initializer.text.trim()
          if (/[A-Za-z]{2,}/.test(text) && !ALLOW.test(text)) found.add(`${rel}: [${name}] ${text}`)
        }
      }
      ts.forEachChild(n, visit)
    }
    visit(sf)
  }
  return [...found].sort()
}

describe('hard-coded JSX text does not grow', () => {
  const current = scan()
  if (process.env.I18N_JSX_BASELINE === 'write' || !existsSync(BASELINE)) {
    writeFileSync(BASELINE, JSON.stringify(current, null, 2) + '\n')
  }
  const baseline: string[] = JSON.parse(readFileSync(BASELINE, 'utf8'))
  it('has no new untranslated JSX text beyond the pinned baseline', () => {
    const added = current.filter(x => !baseline.includes(x))
    expect(added, 'new hard-coded JSX text (translate it, or add to the allowlist/baseline deliberately):\n' + added.join('\n')).toEqual([])
  })
  it('reports the baseline size so shrinking it is visible', () => {
    expect(baseline.length).toBeLessThanOrEqual(200)
  })
})
