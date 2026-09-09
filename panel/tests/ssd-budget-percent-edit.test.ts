import { readFileSync } from 'node:fs'
import ts from 'typescript'
import { describe, expect, it } from 'vitest'
import { buildCacheLaunchArgs } from '../src/shared/cacheLaunchArgs'

// Execute the actual shared form callback, not a separately rewritten reducer.
// This is callback/argv coverage, not native GUI acceptance.
function percentHandler(onChange: (key: string, value: unknown) => void) {
  const file = ts.createSourceFile('form.tsx', readFileSync(
    'src/renderer/src/components/sessions/SessionConfigForm.tsx', 'utf8'),
    ts.ScriptTarget.Latest, true, ts.ScriptKind.TSX)
  let expression: ts.Expression | undefined
  function visit(node: ts.Node) {
    if (ts.isJsxSelfClosingElement(node) && node.tagName.getText(file) === 'SliderField') {
      const attrs = node.attributes.properties.filter(ts.isJsxAttribute)
      const setting = attrs.find(a => a.name.getText(file) === 'settingKey')?.initializer
      if (setting && ts.isStringLiteral(setting) && setting.text === 'blockDiskCacheMaxPercent') {
        const callback = attrs.find(a => a.name.getText(file) === 'onChange')?.initializer
        if (callback && ts.isJsxExpression(callback)) expression = callback.expression
      }
    }
    ts.forEachChild(node, visit)
  }
  visit(file)
  if (!expression) throw new Error('Visible SSD percent callback not found')
  const js = ts.transpileModule(`const handler = ${expression.getText(file)};`, {
    compilerOptions: { target: ts.ScriptTarget.ES2022 },
  }).outputText
  return new Function('onChange', `${js}\nreturn handler;`)(onChange) as (v: number) => void
}

describe('visible SSD budget edit with migrated sessions', () => {
  it('explains retained GB precedence in every supported locale', () => {
    for (const locale of ['en', 'es', 'ja', 'ko', 'zh']) {
      const catalog = JSON.parse(readFileSync(`src/renderer/src/i18n/locales/${locale}.json`, 'utf8'))
      expect(catalog.sessions.config.blockCacheSavedGbOverride).toContain('{gb}')
    }
  })
  for (const percent of [0, 3, 90]) {
    for (const gb of [undefined, 1, 7.5]) {
      it(`percent ${percent} replaces only the explicit GB override ${gb}`, () => {
        const config: any = {
          continuousBatching: true, enablePrefixCache: true, usePagedCache: false,
          enableDiskCache: false, enableBlockDiskCache: true,
          blockDiskCacheMaxGb: gb, blockDiskCacheMaxPercent: 10,
          blockDiskCacheDir: '/isolated/ssd', maxTokens: 4096,
        }
        const before = { ...config }
        const edit = percentHandler((key, value) => { config[key] = value })
        expect(config).toEqual(before) // Opening/re-rendering must not migrate user intent.
        edit(percent)
        expect(config.blockDiskCacheMaxGb).toBeUndefined()
        expect(Object.hasOwn(config, 'blockDiskCacheMaxGb')).toBe(true) // IPC deletion, not omission.
        expect(config.blockDiskCacheMaxPercent).toBe(percent)
        expect(config.blockDiskCacheDir).toBe(before.blockDiskCacheDir)
        expect(config.maxTokens).toBe(4096)
        const { args } = buildCacheLaunchArgs(config)
        expect(args).not.toContain('--block-disk-cache-max-gb')
        expect(args[args.indexOf('--block-disk-cache-max-percent') + 1]).toBe(String(percent))
        expect(args).toContain('--no-paged-cache')
      })
    }
  }
})
