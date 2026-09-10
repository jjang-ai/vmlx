import { readFileSync } from 'fs'
import { join } from 'path'
import { describe, expect, it } from 'vitest'

const repo = join(__dirname, '..')

function read(rel: string): string {
  return readFileSync(join(repo, rel), 'utf8')
}

describe('load progress honesty', () => {
  it('does not label phase progress as loading into GPU memory', () => {
    const source = read('src/main/sessions.ts')

    expect(source).not.toContain('Loading model into GPU')
    // Residency is a DIAGNOSTIC readout under the bar; the percentage comes
    // from the engine's lifecycle contract, never from RSS.
    expect(source).toContain('never the percentage oracle')
    expect(source).toContain('residentMb')
    expect(source).toContain('modelBytes')
  })

  it('scans nested model files instead of only top-level safetensors shards', () => {
    // The recursive counter now lives in the shared modelLaunchMemory module
    // (sessions.ts imports it — the byte-identical private copy was removed to
    // stop the admission preflight and the load bar drifting apart).
    const source = read('src/main/modelLaunchMemory.ts')

    expect(source).toContain('withFileTypes: true')
    expect(source).toContain('entry.isDirectory()')
    expect(source).toContain('estimateModelFileBytes(fullPath)')
    // sessions.ts must consume it, not redefine it.
    const sessions = read('src/main/sessions.ts')
    expect(sessions).toContain("estimateModelFileBytes,\n")
    expect(sessions).not.toContain('function estimateModelFileBytes(')
  })

  it('preserves model-size metadata when later phase updates arrive', () => {
    const source = read('src/renderer/src/contexts/SessionsContext.tsx')

    expect(source).toContain('modelBytes?: number')
    expect(source).toContain('...(next.get(data.sessionId) || {})')
  })

  it('shows model file size separately from phase percent on every load surface', () => {
    const card = read('src/renderer/src/components/sessions/SessionCard.tsx')
    const view = read('src/renderer/src/components/sessions/SessionView.tsx')
    const create = read('src/renderer/src/components/sessions/CreateSession.tsx')

    // The label moved behind t() in the i18n pass. The invariant is unchanged —
    // both surfaces must render the model-files line — so pin the key in both
    // components AND the English copy that key resolves to.
    const en = read('src/renderer/src/i18n/locales/en.json')
    expect(card).toContain("t('sessions.card.modelFiles')")
    expect(view).toContain("t('sessions.card.modelFiles')")
    expect(create).toContain("t('sessions.card.modelFiles')")
    expect(en).toContain('"modelFiles": "Model files:"')
  })

  it('polls process RSS during loading and renders resident RAM separately', () => {
    const source = read('src/main/sessions.ts')
    const context = read('src/renderer/src/contexts/SessionsContext.tsx')
    const card = read('src/renderer/src/components/sessions/SessionCard.tsx')
    const view = read('src/renderer/src/components/sessions/SessionView.tsx')
    const create = read('src/renderer/src/components/sessions/CreateSession.tsx')

    expect(source).toContain('readProcessGroupResidentBytes')
    expect(source).toContain("execFileSync('ps', ['-o', 'rss=', '-g'")
    expect(source).toContain('residentPercent')
    expect(context).toContain('residentPercent?: number')
    // Same i18n retarget as the model-files label above.
    const en = read('src/renderer/src/i18n/locales/en.json')
    expect(card).toContain("t('sessions.card.residentRam')")
    expect(view).toContain("t('sessions.card.residentRam')")
    expect(create).toContain("t('sessions.card.residentRam')")
    expect(en).toContain('"residentRam": "Resident RAM:"')
  })

  it('the blocking create-session loader consumes the shared live progress entry', () => {
    const create = read('src/renderer/src/components/sessions/CreateSession.tsx')

    expect(create).toContain('const { loadProgress } = useSessionsContext()')
    expect(create).toContain('loadProgress.get(launchSessionId)')
    expect(create).toContain('role="progressbar"')
    expect(create).toContain('aria-valuenow={progress.indeterminate === false ? progress.progress : undefined}')
    expect(create).toContain("progress.indeterminate === false ? `(${progress.progress}%)` : ''")
    expect(create).toContain('data-vmlx-create-load-session-id={launchSessionId')
  })
})

describe('load-progress labels can reach the locale catalogs', () => {
  const sessions = readFileSync(
    join(repo, 'src/main/sessions.ts'),
    'utf8',
  )
  const card = readFileSync(
    join(repo, 'src/renderer/src/components/sessions/SessionCard.tsx'),
    'utf8',
  )

  it('every progress pattern ships an i18n key beside its English text', () => {
    // These 32 labels live in the MAIN process, which has no locale catalog:
    // they are derived from engine log lines and delivered to the renderer as
    // DATA. SessionCard rendered `progress.label` verbatim, so the renderer's
    // i18n could never reach them and they stayed English in all five locales.
    //
    // The namespace is asserted because it is the half that actually broke.
    // These keys were emitted as `sessions.loadProgress.*` while every catalog
    // defined `main.loadProgress.*`, so all 31 resolved nowhere and the
    // defaultValue quietly served English to all five locales. Pinning the
    // prefix alone cannot prove they resolve — see
    // i18n-load-progress-keys.test.ts, which checks each key against every
    // catalog.
    const start = sessions.indexOf('LOAD_PROGRESS_PATTERNS')
    const block = sessions.slice(start, sessions.indexOf('\n  ]', start))
    const labels = block.match(/label: '/g) || []
    const keys = block.match(/labelKey: 'main\.loadProgress\./g) || []
    expect(labels.length).toBeGreaterThan(20)
    expect(keys.length).toBe(labels.length)
  })

  it('the renderer resolves the key and falls back to the English text', () => {
    expect(card).toContain('progress.labelKey')
    expect(card).toContain('defaultValue: progress.label')
    // The fallback keeps an older main process — one that sends no labelKey —
    // rendering correctly. It is NOT licence to ship keys the catalogs lack:
    // it makes a missing entry invisible rather than loud, which is how the
    // whole namespace stayed unresolved without anyone noticing.
    expect(card).not.toMatch(/\{progress\.label\}\s*\(\{progress\.progress\}%\)/)
  })

  it('labelKey is optional on the wire so an older main process still renders', () => {
    const context = readFileSync(
      join(repo, 'src/renderer/src/contexts/SessionsContext.tsx'),
      'utf8',
    )
    expect(context).toContain('labelKey?: string')
  })

  it('the loadProgress handler actually forwards labelKey into state', () => {
    // Declaring `labelKey?: string` on the interface proves nothing about the
    // wire. The handler copied label, progress, modelBytes, residentMb, peakMb
    // and cacheMb field by field and silently omitted labelKey, so every card
    // saw `undefined`, took the English branch, and stayed English in all five
    // locales while its own sibling lines rendered translated. The type said
    // the field existed the entire time.
    const context = readFileSync(
      join(repo, 'src/renderer/src/contexts/SessionsContext.tsx'),
      'utf8',
    )
    const start = context.indexOf('onLoadProgress')
    expect(start).toBeGreaterThan(-1)
    const handler = context.slice(start, start + 1600)
    expect(handler).toMatch(/labelKey:\s*data\.labelKey/)
    // Unconditional, because the previous entry is spread in first: a
    // conditional copy would pair a stale key with a fresh English label.
    expect(handler).not.toMatch(/\.\.\.\(data\.labelKey[^)]*\)/)
  })
})
