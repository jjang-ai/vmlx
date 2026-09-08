import { readFileSync } from 'node:fs'
import React from 'react'
import ts from 'typescript'
import { describe, expect, it } from 'vitest'

const src = readFileSync('src/renderer/src/components/UpdateBanner.tsx', 'utf8')
const ast = ts.createSourceFile('banner.tsx', src, ts.ScriptTarget.Latest, true, ts.ScriptKind.TSX)
const declaration = ast.statements.find(n => ts.isFunctionDeclaration(n) && n.name?.text === 'UpdateBanner')!
const js = ts.transpileModule(declaration.getText(ast).replace(/^export /, ''), {
  compilerOptions: { target: ts.ScriptTarget.ES2022, jsx: ts.JsxEmit.React },
}).outputText

function mount() {
  const slots: unknown[] = [], storage = new Map<string, string>(), opened: string[] = []
  let index = 0, subscribed = false, listener!: (data: unknown) => void
  const env = { React, X: 'icon',
    useState: (initial: unknown) => {
      const slot = index++
      if (!(slot in slots)) slots[slot] = initial
      return [slots[slot], (value: unknown) => { slots[slot] = value }]
    },
    useEffect: (effect: () => unknown) => { if (!subscribed) { subscribed = true; effect() } },
    useTranslation: () => ({ t: (key: string, args?: { version: string }) => args?.version || key }),
    localStorage: { getItem: (key: string) => storage.get(key), setItem: (key: string, value: string) => storage.set(key, value) },
    window: { api: { app: { onUpdateAvailable: (fn: typeof listener) => { listener = fn; return () => {} } } }, open: (url: string) => opened.push(url) },
  }
  const Component = new Function(...Object.keys(env), `${js}; return UpdateBanner`)(...Object.values(env))
  const render = () => { index = 0; return Component() }
  const emit = (version: string) => listener({ currentVersion: '1.6.55', latestVersion: version, url: 'https://github.com/jjang-ai/mlxstudio/releases' })
  render()
  return { render, emit, storage, opened }
}
function find(node: any, type: string): any {
  if (!node) return undefined
  if (node.type === type) return node
  return React.Children.toArray(node.props?.children).map(child => find(child, type)).find(Boolean)
}

describe('release notification dismissal', () => {
  it('dismisses this release but shows a subsequent release in the same app lifetime', () => {
    const f = mount()
    f.emit('1.6.56')
    find(f.render(), 'button').props.onClick()
    expect(f.render()).toBeNull()
    f.emit('1.6.56')
    expect(f.render()).toBeNull()
    f.emit('1.6.57')
    expect(f.render()).not.toBeNull()
  })
  it('opens the provided download URL through the existing external-link path', () => {
    const f = mount()
    f.emit('1.6.56')
    find(f.render(), 'a').props.onClick({ preventDefault() {} })
    expect(f.opened).toEqual(['https://github.com/jjang-ai/mlxstudio/releases'])
  })
})
