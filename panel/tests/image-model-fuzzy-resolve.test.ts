/**
 * mlxstudio#82 (reported by @LewnWorx) — "Moved image models still not opening
 * REDUX / Deeper Dive".
 *
 * Locks the fuzzy-match resolver `resolveImageModelFromDirectoryName` in
 * `src/shared/imageModels.ts` against regression. Mark renamed his models
 * with INT-/EXT- prefixes and moved them to an external TB4 drive; the Server
 * tab's launch flow uses `basename(modelPath)` as the model lookup key and
 * the old `getImageModel()` exact-id match never hit for scanned directory
 * names like `FLUX.2-klein-9B` or `FLUX.1-dev-mflux-8bit`.
 *
 * Fuzzy resolver rules (in order, first match wins):
 *   1. exact id
 *   2. exact mfluxName
 *   3. normalize (lowercase + strip hf-org + strip `-mflux[-Nbit]`, `-Nbit`,
 *      `_Nbit`, `int-/ext-` prefix) -> retry id + mfluxName
 *   4. same as 3 + drop dots -> retry id + mfluxName
 *   5. match basename(repoMap[q]) for any quantize, then again against
 *      normalized input vs normalized basename.
 */
import { describe, it, expect } from 'vitest'
import {
    resolveImageModelFromDirectoryName,
    getImageModel,
} from '../src/shared/imageModels'

describe('mlxstudio#82: resolveImageModelFromDirectoryName', () => {
    describe('Rule 1: exact id match (superset of getImageModel)', () => {
        it('resolves canonical ids', () => {
            expect(resolveImageModelFromDirectoryName('schnell')?.id).toBe('schnell')
            expect(resolveImageModelFromDirectoryName('klein-9b')?.id).toBe('klein-9b')
            expect(resolveImageModelFromDirectoryName('qwen-image-edit')?.id).toBe('qwen-image-edit')
        })

        it('is a strict superset of getImageModel for canonical ids', () => {
            for (const id of ['schnell', 'dev', 'klein-4b', 'klein-9b', 'qwen-image-edit', 'kontext']) {
                expect(resolveImageModelFromDirectoryName(id)?.id).toBe(getImageModel(id)?.id)
            }
        })
    })

    describe('Rule 2: exact mfluxName match', () => {
        it('resolves mflux canonical names', () => {
            expect(resolveImageModelFromDirectoryName('flux2-klein-9b')?.id).toBe('klein-9b')
            expect(resolveImageModelFromDirectoryName('z-image-turbo')?.id).toBe('z-image-turbo')
            expect(resolveImageModelFromDirectoryName('dev-kontext')?.id).toBe('kontext')
        })
    })

    describe('Rule 3: normalize (lowercase + decoration strip)', () => {
        it("resolves Mark's external FLUX.2-klein-9B (the exact log failure)", () => {
            expect(resolveImageModelFromDirectoryName('FLUX.2-klein-9B')?.id).toBe('klein-9b')
        })

        it('resolves FLUX.1-dev (dotted HF form)', () => {
            expect(resolveImageModelFromDirectoryName('FLUX.1-dev')?.id).toBe('dev')
        })

        it('strips -mflux-Nbit quant decoration', () => {
            expect(resolveImageModelFromDirectoryName('FLUX.1-dev-mflux-8bit')?.id).toBe('dev')
            expect(resolveImageModelFromDirectoryName('FLUX.1-schnell-mflux-4bit')?.id).toBe('schnell')
            expect(resolveImageModelFromDirectoryName('Z-Image-Turbo-mflux-8bit')?.id).toBe('z-image-turbo')
        })

        it('strips -mflux only', () => {
            expect(resolveImageModelFromDirectoryName('FLUX.1-dev-mflux')?.id).toBe('dev')
        })

        it('strips bare -Nbit and _Nbit', () => {
            expect(resolveImageModelFromDirectoryName('FLUX.1-dev-8bit')?.id).toBe('dev')
            expect(resolveImageModelFromDirectoryName('FLUX.1-dev_4bit')?.id).toBe('dev')
        })

        it("strips Mark's INT-/EXT- prefix convention", () => {
            expect(resolveImageModelFromDirectoryName('INT-Qwen-Image-Edit')?.id).toBe('qwen-image-edit')
            expect(resolveImageModelFromDirectoryName('EXT-FLUX.2-klein-9B')?.id).toBe('klein-9b')
            expect(resolveImageModelFromDirectoryName('INT_Qwen-Image-Edit')?.id).toBe('qwen-image-edit')
        })

        it('strips HuggingFace org prefix', () => {
            expect(resolveImageModelFromDirectoryName('black-forest-labs/FLUX.2-klein-9B')?.id).toBe('klein-9b')
            expect(resolveImageModelFromDirectoryName('Qwen/Qwen-Image-Edit')?.id).toBe('qwen-image-edit')
        })
    })

    describe('Rule 4: drop dots fallback', () => {
        it('resolves when only the dot is the difference (flux.2 vs flux2)', () => {
            expect(resolveImageModelFromDirectoryName('flux.2-klein-4b')?.id).toBe('klein-4b')
        })
    })

    describe('Rule 5: repoMap basename match', () => {
        it('resolves downloaded HF-repo-basename directory names', () => {
            // Model downloaded as `dhairyashil/FLUX.1-dev-mflux-8bit` -> dir
            // basename is `FLUX.1-dev-mflux-8bit` -> matches repoMap[8].
            expect(resolveImageModelFromDirectoryName('FLUX.1-dev-mflux-8bit')?.id).toBe('dev')
            expect(resolveImageModelFromDirectoryName('RunPod/FLUX.2-klein-4B-mflux-4bit')?.id).toBe('klein-4b')
        })
    })

    describe('Negative cases', () => {
        it('returns undefined for empty string', () => {
            expect(resolveImageModelFromDirectoryName('')).toBeUndefined()
        })

        it('returns undefined for completely unknown names', () => {
            expect(resolveImageModelFromDirectoryName('completely-unknown-model')).toBeUndefined()
            expect(resolveImageModelFromDirectoryName('stable-diffusion-xl')).toBeUndefined()
        })

        it('does NOT false-positive on partial substrings', () => {
            // "dev" is a canonical id — but "development-notes" should not
            // accidentally resolve to it. Fuzzy resolver rules are all
            // exact-match-after-normalize, not substring.
            expect(resolveImageModelFromDirectoryName('development-notes')).toBeUndefined()
        })
    })

    describe("Full Mark's-report permutations table", () => {
        const cases: Array<[string, string | undefined]> = [
            ['FLUX.2-klein-9B', 'klein-9b'],
            ['FLUX.2-klein-4B', 'klein-4b'],
            ['FLUX.1-dev', 'dev'],
            ['FLUX.1-dev-mflux-8bit', 'dev'],
            ['FLUX.1-dev-mflux', 'dev'],
            ['FLUX.1-dev-8bit', 'dev'],
            ['Qwen-Image-Edit', 'qwen-image-edit'],
            ['INT-Qwen-Image-Edit', 'qwen-image-edit'],
            // The two cases that need model_index.json fallback (engine-side,
            // tested in pytest) — fuzzy-resolver correctly returns undefined:
            ['FLUX.2-klien-blah-blah', undefined],
            ['INT_QIE', undefined],
        ]

        for (const [input, expected] of cases) {
            it(`${input} -> ${expected ?? 'undefined (model_index.json fallback domain)'}`, () => {
                expect(resolveImageModelFromDirectoryName(input)?.id).toBe(expected)
            })
        }
    })
})
