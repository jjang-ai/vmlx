import { readFileSync } from 'node:fs'
import ts from 'typescript'
import { describe, expect, it, vi } from 'vitest'

// Execute the production New Session callback, including its registry fallback.
function createSession(capabilities: unknown, category = 'generate') {
  const source = readFileSync('src/renderer/src/components/image/ImageTab.tsx', 'utf8')
  const start = source.indexOf('const handleNewSession = useCallback(') + 'const handleNewSession = useCallback('.length
  const end = source.indexOf('}, [selectedModel, capabilities])', start) + 1
  const context = {
    capabilities, selectedModel: '/models/flux2-klein-9b',
    imageDrafts: {newConversation: vi.fn()}, setGenerations: vi.fn(), setError: vi.fn(), setSessionMode: vi.fn(),
    getImageModel: () => ({category}), resolveImageModelFromDirectoryName: () => ({category}),
  }
  const js = ts.transpile('return (' + source.slice(start,end) + ')', {target: ts.ScriptTarget.ES2022})
  Function(...Object.keys(context), js)(...Object.values(context))()
  return context
}

describe('new image conversation adapter mode', () => {
  it('keeps loaded KleinEdit in edit mode even when shared bundle name resolves to generator', () => {
    const ctx = createSession({loaded:true,mode:'edit',mflux_class:'Flux2KleinEdit'})
    expect(ctx.setSessionMode).toHaveBeenCalledWith('edit')
    expect(ctx.imageDrafts.newConversation).toHaveBeenCalledOnce()
  })
  it('keeps the selected operation for a legitimate dual adapter', () => {
    expect(createSession({loaded:true,mode:'dual'}).setSessionMode).not.toHaveBeenCalled()
  })
  it('uses generation for a loaded generation adapter, and registry only before capabilities load', () => {
    expect(createSession({loaded:true,mode:'generate'},'edit').setSessionMode).toHaveBeenCalledWith('generate')
    expect(createSession(null,'edit').setSessionMode).toHaveBeenCalledWith('edit')
    expect(createSession({loaded:false,mode:'generate'},'edit').setSessionMode).toHaveBeenCalledWith('edit')
  })
})
