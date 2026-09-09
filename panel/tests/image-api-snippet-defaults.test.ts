import { describe, expect, it, vi } from 'vitest'
import { spawnSync } from 'node:child_process'
import { mkdtempSync, writeFileSync, rmSync } from 'node:fs'
import { tmpdir } from 'node:os'
import { join } from 'node:path'
vi.mock('../src/renderer/src/i18n', () => ({ useTranslation: () => ({ t: (key: string) => key }) }))
import { buildImageCurl, buildImagePython, buildImageJavaScript } from '../src/renderer/src/components/api/CodeSnippets'

describe('image generation snippets preserve engine defaults', () => {
  for (const model of ['dev', 'schnell', 'z-image-turbo', 'my-local-image-alias']) {
    for (const builder of [buildImageCurl, buildImagePython, buildImageJavaScript]) {
      it(`${builder.name} does not force Schnell settings on ${model}`, () => {
        const text = builder('http://localhost:8115', null, model, false)
        expect(text).toContain('/v1/images/generations')
        expect(text).toContain(model)
        expect(text).toContain('1024x1024')
        expect(text).not.toMatch(/\bsteps\b|\bguidance\b/)
      })
    }
  }
  it('curl emits valid image request JSON', () => {
    const text = buildImageCurl('http://localhost:8115', null, 'dev', false)
    const body = JSON.parse(text.match(/-d '([\s\S]+)'$/)![1])
    expect(body).toMatchObject({model: 'dev', size: '1024x1024', response_format: 'b64_json'})
    expect(body).not.toHaveProperty('steps')
    expect(body).not.toHaveProperty('guidance')
  })
})

describe('image edit snippets preserve wire JSON and adapter defaults', () => {
  for (const builder of [buildImageCurl, buildImagePython, buildImageJavaScript]) {
    it(`${builder.name} leaves edit steps and guidance to the loaded adapter`, () => {
      const text = builder('http://localhost:8115', null, 'dev-fill', true)
      expect(text).toContain('/v1/images/edits')
      expect(text).not.toMatch(/\bstep[s]\b|\bguidance\b/)
    })
  }
  for (const mask of [false, true]) {
    it(`curl actually sends parseable JSON (mask=${mask})`, () => {
      const dir = mkdtempSync(join(tmpdir(), 'image-edit-snippet-'))
      try {
        if (mask) writeFileSync(join(dir, 'mask.png'), 'mask fixture')
        const script = [
          "base64() { printf 'aW1hZ2U='; }",
          "curl() { printf '%s\\0' \"$@\"; }",
          buildImageCurl('http://localhost:8115', 'test-only-key', 'dev-fill', true),
        ].join('\n')
        const run = spawnSync('/bin/bash', ['-c', script], { cwd: dir, encoding: 'utf8' })
        expect(run.status, run.stderr).toBe(0)
        const args = run.stdout.split('\0')
        expect(args).toContain('Authorization: Bearer test-only-key')
        const body = JSON.parse(args[args.indexOf('-d') + 1])
        expect(body).toMatchObject({ model: 'dev-fill', image: 'aW1hZ2U=', mask: mask ? 'aW1hZ2U=' : '' })
      } finally {
        rmSync(dir, { recursive: true, force: true })
      }
    })
  }
})
