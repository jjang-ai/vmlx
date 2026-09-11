import { readFileSync } from 'node:fs'
import ts from 'typescript'
import { describe, expect, it } from 'vitest'
import { buildNativeMtpLaunchArgs, resolveFixedNativeMtpDepth } from '../src/shared/nativeMtpLaunchArgs'

describe('fixed MTP input and effective ceiling parity', () => {
  it('explains dynamic fallback and bounded recovery instead of an exact draft depth', () => {
    const note = JSON.parse(readFileSync('src/renderer/src/i18n/locales/en.json', 'utf8'))
      .sessions.config.nativeMtpDepthFixedNote
    expect(note).toContain('D{depth} is the maximum draft depth')
    expect(note).toContain('not a speed guarantee')
    expect(note).toContain('autoregressive decoding (AR)')
    expect(note).toContain('retry higher depths without exceeding this limit')
    expect(note).not.toContain('D1-D2 draft that depth')
  })
  it.each([1, 2, 3, 23, 0, -1, NaN, Infinity, 2.9, undefined])(
    'displays the same fixed ceiling that launch emits for %s', (configuredDepth) => {
      const args = buildNativeMtpLaunchArgs({
        supported: true, depthOverride: true, configuredDepth, detectedDepth: 3,
      })
      const displayed = resolveFixedNativeMtpDepth(configuredDepth, 3)
      expect(displayed).toBe(Number(args[args.indexOf('--native-mtp-depth') + 1]))
      expect(displayed).toBeGreaterThanOrEqual(1)
      expect(displayed).toBeLessThanOrEqual(3)
    },
  )
  it('bounds typed depth as well as its range slider without changing other fields', () => {
    const source = readFileSync('src/renderer/src/components/sessions/SessionConfigForm.tsx', 'utf8')
    const file = ts.createSourceFile('form.tsx', source, ts.ScriptTarget.Latest, true, ts.ScriptKind.TSX)
    const matches: ts.JsxSelfClosingElement[] = []
    function visit(node: ts.Node) {
      if (ts.isJsxSelfClosingElement(node) && node.tagName.getText(file) === 'SliderField') {
        const key = node.attributes.properties.filter(ts.isJsxAttribute)
          .find(a => a.name.getText(file) === 'settingKey')?.initializer
        if (key && ts.isStringLiteral(key) && key.text === 'nativeMtpDepth') matches.push(node)
      }
      ts.forEachChild(node, visit)
    }
    visit(file)
    expect(matches).toHaveLength(1)
    const attrs = matches[0].attributes.properties.filter(ts.isJsxAttribute)
    for (const key of ['max', 'maxInput']) {
      const value = attrs.find(a => a.name.getText(file) === key)?.initializer
      expect(value && ts.isJsxExpression(value) && value.expression?.getText(file)).toBe('3')
    }
    expect(source).toContain('resolveFixedNativeMtpDepth(config.nativeMtpDepth, detectedNativeMtp?.depth)')
  })
})
