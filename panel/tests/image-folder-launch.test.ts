import { describe, expect, it } from 'vitest'
import { canLaunchInspectedImageFolder } from '../src/shared/imageFolderLaunch'
import { readFileSync } from 'fs'
import { join } from 'path'

const valid = { input: '/models/q8', inspectedInput: '/models/q8', inspecting: false, success: true, detected: true, explicitClass: '', explicitTask: '' }
describe('folder-first launch contract', () => {
  it('allows a current detected folder only after inspection', () => {
    expect(canLaunchInspectedImageFolder(valid)).toBe(true)
    expect(canLaunchInspectedImageFolder({ ...valid, inspecting: true })).toBe(false)
    expect(canLaunchInspectedImageFolder({ ...valid, success: false })).toBe(false)
  })
  it('rejects the prior folder preview after changing the path', () => {
    expect(canLaunchInspectedImageFolder({ ...valid, input: '/models/other' })).toBe(false)
    expect(canLaunchInspectedImageFolder({ ...valid, input: '' })).toBe(false)
  })
  it('never defaults an unknown architecture to Flux or generation', () => {
    const unknown = { ...valid, detected: false }
    expect(canLaunchInspectedImageFolder(unknown)).toBe(false)
    expect(canLaunchInspectedImageFolder({ ...unknown, explicitClass: 'QwenImageEdit' })).toBe(false)
    expect(canLaunchInspectedImageFolder({ ...unknown, explicitClass: 'QwenImageEdit', explicitTask: 'edit' })).toBe(true)
  })
  it('keeps selection separate from launch and clears per-folder overrides', () => {
    const source = readFileSync(join(__dirname, '../src/renderer/src/components/image/ImageModelPicker.tsx'), 'utf8')
    expect(source).not.toContain('downloadImageModel(')
    expect(source).not.toContain('checkImageModel(')
    const select = source.slice(source.indexOf('const chooseCustomPath'), source.indexOf('const canLoadFolder'))
    expect(select).toContain("setCustomCategory('')")
    expect(select).toContain("setCustomMfluxClass('')")
    expect(select).not.toContain('onSelect(')
    const browse = source.slice(source.indexOf('const handleBrowse'), source.indexOf('  return (', source.indexOf('const handleBrowse')))
    expect(browse).not.toContain('onSelect(')
    expect(source).toContain('if (!canLoadFolder || !localPreview?.success) return')
  })
  it('routes the running-model switch to inspection instead of preset launch', () => {
    const top = readFileSync(join(__dirname, '../src/renderer/src/components/image/ImageTopBar.tsx'), 'utf8')
    expect(top).toContain('onClick={onChangeModel}')
    expect(top).not.toContain('onSelectModel')
    expect(top).not.toContain('checkImageModel(')
    const tab = readFileSync(join(__dirname, '../src/renderer/src/components/image/ImageTab.tsx'), 'utf8')
    expect(tab).not.toContain('onSelectModel=')
  })
})
