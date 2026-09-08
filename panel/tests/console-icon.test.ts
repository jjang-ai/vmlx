import { describe, expect, it } from 'vitest'
import { readFileSync } from 'node:fs'
import { resolve } from 'node:path'
import sharp from 'sharp'

describe('Console app icon assets', () => {
  it('ships the 1024px packaged icon and local renderer derivatives', async () => {
    const root = resolve(__dirname, '..')
    for (const [file, size] of [
      ['resources/icon.png', 1024],
      ['src/renderer/public/app-icon-64.png', 64],
      ['src/renderer/public/favicon-32.png', 32],
      ['src/renderer/public/favicon-16.png', 16],
    ] as const) {
      const metadata = await sharp(resolve(root, file)).metadata()
      expect([metadata.width, metadata.height]).toEqual([size, size])
    }
    const pkg = JSON.parse(readFileSync(resolve(root, 'package.json'), 'utf8'))
    expect(pkg.build.mac.icon).toBe('resources/icon.png')
    const generator = readFileSync(resolve(root, 'generate-icons.js'), 'utf8')
    expect(generator).not.toContain('resources/logo.svg')
    expect(generator).toContain('src/renderer/public')
    const sidebar = readFileSync(resolve(root, 'src/renderer/src/components/layout/ConsoleSidebar.tsx'), 'utf8')
    expect(sidebar).toContain('src="./app-icon-64.png"')
    expect(readFileSync(resolve(root, 'src/renderer/index.html'), 'utf8')).not.toContain('favicon.svg')
  })
})
