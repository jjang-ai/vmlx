/**
 * mlxstudio#82 (reported by @LewnWorx) — display consistency guard.
 *
 * Mark's complaint was that a relocated model showed two different names
 * in different UI surfaces: "FLUX2" (dots stripped — the `cfg.servedModelName`
 * canonical mflux form) in one place, "FLUX.2" (the on-disk directory
 * basename) in another. Root cause was the priority inversion in
 * `image.ts:getRunningServer::buildResult`: it preferred `cfg.servedModelName`
 * FIRST, falling back to `modelPath.split('/').pop()` only when the config
 * lacked a served name.
 *
 * Rule (locked by this file):
 *   - `servedModelName` is the API ROUTING key (used in api-gateway.ts to
 *     resolve `model` in chat/completions requests → session).
 *   - UI DISPLAYS should always show the directory basename exactly as on
 *     disk. That's the identity the user filed on a TB4 drive, what their
 *     OS Finder shows, and what matches between the Server tab list, the
 *     Session config form, and the Image tab picker.
 */
import { describe, it, expect } from 'vitest'
import { readFileSync } from 'fs'
import { join } from 'path'

// Replica of buildResult's modelName-picking logic, extracted for direct test.
// Must stay in sync with image.ts:getRunningServer::buildResult.
function pickDisplayModelName(s: {
    modelName?: string
    modelPath?: string
}, cfg: { servedModelName?: string }): string {
    const pathBase = s.modelPath?.includes('/') ? s.modelPath.split('/').pop()! : s.modelPath
    return (
        pathBase ||
        (s.modelName?.includes('/') ? s.modelName.split('/').pop()! : s.modelName) ||
        cfg.servedModelName ||
        ''
    )
}

describe('mlxstudio#82 display consistency: buildResult modelName priority', () => {
    it("shows directory basename for Mark's external FLUX.2-klein-9B (NOT stripped FLUX2)", () => {
        const s = {
            modelPath: '/Volumes/SSD_4TB_01/AI_MODELS/Image/FLUX.2-klein-9B',
            modelName: 'flux2-klein-9b',
        }
        const cfg = { servedModelName: 'flux2-klein-9b' }
        expect(pickDisplayModelName(s, cfg)).toBe('FLUX.2-klein-9B')
        // Must NOT accidentally return the canonical dots-stripped form
        expect(pickDisplayModelName(s, cfg)).not.toBe('flux2-klein-9b')
        expect(pickDisplayModelName(s, cfg)).not.toBe('FLUX2')
    })

    it('prefers modelPath basename over servedModelName when both present', () => {
        const s = { modelPath: '/ext/img/FLUX.1-dev-mflux-8bit', modelName: 'dev' }
        const cfg = { servedModelName: 'dev' }
        expect(pickDisplayModelName(s, cfg)).toBe('FLUX.1-dev-mflux-8bit')
    })

    it('falls back to servedModelName only when modelPath is empty', () => {
        const s = { modelPath: '', modelName: '' }
        const cfg = { servedModelName: 'flux2-klein-9b' }
        expect(pickDisplayModelName(s, cfg)).toBe('flux2-klein-9b')
    })

    it('strips HF org prefix from modelName fallback', () => {
        const s = { modelPath: '', modelName: 'black-forest-labs/FLUX.2-klein-9B' }
        const cfg = { servedModelName: '' }
        expect(pickDisplayModelName(s, cfg)).toBe('FLUX.2-klein-9B')
    })

    it('returns empty string when all sources missing (defensive)', () => {
        expect(pickDisplayModelName({}, {})).toBe('')
    })
})

describe('mlxstudio#82 display consistency: image.ts source regression guard', () => {
    const IMAGE_TS = join(__dirname, '..', 'src', 'main', 'ipc', 'image.ts')

    it('buildResult priority chain puts modelPath basename before cfg.servedModelName in the ASSIGNMENT (not just comments)', () => {
        // Guard the specific ordering Mark's bug was caused by so that a
        // future refactor that accidentally re-inverts priority fails CI.
        const src = readFileSync(IMAGE_TS, 'utf-8')
        // Strip single-line `// ...` comments before searching — comments
        // legitimately mention both names in explanatory order.
        const code = src.replace(/\/\/[^\n]*\n/g, '\n')
        const idx = code.indexOf('const buildResult = (s: any) =>')
        expect(idx, 'buildResult closure must exist in image.ts').toBeGreaterThan(0)
        const block = code.slice(idx, idx + 2000)
        const pathBaseIdx = block.indexOf('s.modelPath')
        const servedIdx = block.indexOf('cfg.servedModelName')
        expect(pathBaseIdx).toBeGreaterThan(0)
        expect(servedIdx).toBeGreaterThan(0)
        expect(
            pathBaseIdx,
            'regression: cfg.servedModelName assignment comes before s.modelPath; Mark\'s FLUX2/FLUX.2 display bug is back',
        ).toBeLessThan(servedIdx)
    })

    it('buildResult has the mlxstudio#82 explanatory comment', () => {
        const src = readFileSync(IMAGE_TS, 'utf-8')
        expect(src).toContain('mlxstudio#82')
        // Phrase may span a wrapped comment line — match across whitespace.
        expect(src).toMatch(/routing key,\s*(\/\/\s*)?not a display label/)
    })
})
