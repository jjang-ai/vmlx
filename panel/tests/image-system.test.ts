/**
 * Image System Tests
 *
 * Tests for image-related pure logic across the panel codebase:
 *   1. getDefaultSteps -- per-model step defaults
 *   2. NAMED_MODELS / IMAGE_MODEL_REPOS consistency
 *   3. connectHost -- bind address normalization
 *   4. getSizeEstimate -- model size estimates
 *   5. Gated error detection regex
 *   6. Data URL prefix stripping
 *
 * Functions are re-implemented here as pure copies (project testing pattern) to avoid
 * importing from Electron/React modules that require DOM or Node environments.
 *
 * NOTE: Edit vs generation model detection is done via config.imageMode (set at server
 * start time and stored in session config). There is NO regex-based model name detection.
 */
import { describe, it, expect } from 'vitest'

// =============================================================================
// 1. getDefaultSteps (from src/renderer/src/components/image/ImageTab.tsx)
// =============================================================================

function getDefaultSteps(modelId: string): number {
  const defaults: Record<string, number> = {
    'schnell': 4,
    'dev': 20,
    'z-image-turbo': 4,
    'flux2-klein-4b': 20,
    'flux2-klein-9b': 20,
    'qwen-image-edit': 28,
  }
  return defaults[modelId] || 4
}

describe('getDefaultSteps', () => {
  describe('returns correct defaults for every known model', () => {
    it('schnell -> 4', () => expect(getDefaultSteps('schnell')).toBe(4))
    it('dev -> 20', () => expect(getDefaultSteps('dev')).toBe(20))
    it('z-image-turbo -> 4', () => expect(getDefaultSteps('z-image-turbo')).toBe(4))
    it('flux2-klein-4b -> 20', () => expect(getDefaultSteps('flux2-klein-4b')).toBe(20))
    it('flux2-klein-9b -> 20', () => expect(getDefaultSteps('flux2-klein-9b')).toBe(20))
    it('qwen-image-edit -> 28', () => expect(getDefaultSteps('qwen-image-edit')).toBe(28))
  })

  describe('unknown model returns default of 4', () => {
    it('custom-model -> 4', () => expect(getDefaultSteps('custom-model')).toBe(4))
    it('empty string -> 4', () => expect(getDefaultSteps('')).toBe(4))
    it('random-name -> 4', () => expect(getDefaultSteps('random-name')).toBe(4))
    it('SCHNELL (wrong case) -> 4', () => expect(getDefaultSteps('SCHNELL')).toBe(4))
    it('/path/to/schnell -> 4 (not a substring match)', () => {
      expect(getDefaultSteps('/path/to/schnell')).toBe(4)
    })
  })

  describe('step values match NAMED_MODELS definitions', () => {
    // The NAMED_MODELS array defines steps per model; getDefaultSteps must agree
    const NAMED_MODELS_STEPS: Record<string, number> = {
      'schnell': 4,
      'dev': 20,
      'z-image-turbo': 4,
      'flux2-klein-4b': 20,
      'flux2-klein-9b': 20,
      'qwen-image-edit': 28,
    }
    for (const [id, expectedSteps] of Object.entries(NAMED_MODELS_STEPS)) {
      it(`${id}: getDefaultSteps matches NAMED_MODELS.steps (${expectedSteps})`, () => {
        expect(getDefaultSteps(id)).toBe(expectedSteps)
      })
    }
  })
})

// =============================================================================
// 2. NAMED_MODELS / IMAGE_MODEL_REPOS consistency
// =============================================================================

// Re-implement from ImageModelPicker.tsx
const NAMED_MODELS = [
  { id: 'schnell', name: 'Flux Schnell', steps: 4, category: 'generate' as const, quantizeOptions: [4, 8, 0] },
  { id: 'dev', name: 'Flux Dev', steps: 20, category: 'generate' as const, quantizeOptions: [4, 8, 0] },
  { id: 'z-image-turbo', name: 'Z-Image Turbo', steps: 4, category: 'generate' as const, quantizeOptions: [4, 8, 0] },
  { id: 'flux2-klein-4b', name: 'Flux Klein 4B', steps: 20, category: 'generate' as const, quantizeOptions: [8, 0] },
  { id: 'flux2-klein-9b', name: 'Flux Klein 9B', steps: 20, category: 'generate' as const, quantizeOptions: [0] },
  { id: 'qwen-image-edit', name: 'Qwen Image Edit', steps: 28, category: 'edit' as const, quantizeOptions: [0] },
]

// Re-implement from models.ts
const IMAGE_MODEL_REPOS: Record<string, Record<number, string>> = {
  'schnell': {
    4: 'dhairyashil/FLUX.1-schnell-mflux-4bit',
    8: 'dhairyashil/FLUX.1-schnell-mflux-8bit',
    0: 'black-forest-labs/FLUX.1-schnell',
  },
  'dev': {
    4: 'dhairyashil/FLUX.1-dev-mflux-4bit',
    8: 'dhairyashil/FLUX.1-dev-mflux-8bit',
    0: 'black-forest-labs/FLUX.1-dev',
  },
  'z-image-turbo': {
    4: 'filipstrand/Z-Image-Turbo-mflux-4bit',
    8: 'carsenk/z-image-turbo-mflux-8bit',
    0: 'Tongyi-MAI/Z-Image-Turbo',
  },
  'flux2-klein-4b': {
    8: 'AITRADER/FLUX2-klein-4B-mlx-8bit',
    0: 'black-forest-labs/FLUX.2-klein-4B',
  },
  'flux2-klein-9b': {
    0: 'black-forest-labs/FLUX.2-klein-9B',
  },
  'qwen-image-edit': {
    0: 'Qwen/Qwen-Image-Edit',
  },
}

const QUANTIZE_OPTIONS = [
  { value: 4, label: '4-bit' },
  { value: 8, label: '8-bit' },
  { value: 0, label: 'Full' },
]

describe('NAMED_MODELS / IMAGE_MODEL_REPOS consistency', () => {
  describe('every NAMED_MODEL quantize option has a repo entry', () => {
    for (const model of NAMED_MODELS) {
      for (const q of model.quantizeOptions) {
        it(`${model.id} quantize=${q} has IMAGE_MODEL_REPOS entry`, () => {
          const repos = IMAGE_MODEL_REPOS[model.id]
          expect(repos).toBeDefined()
          expect(repos[q]).toBeDefined()
          expect(typeof repos[q]).toBe('string')
          expect(repos[q].length).toBeGreaterThan(0)
        })
      }
    }
  })

  describe('no orphan repos (every repo key maps to a NAMED_MODEL)', () => {
    for (const repoId of Object.keys(IMAGE_MODEL_REPOS)) {
      it(`${repoId} exists in NAMED_MODELS`, () => {
        const found = NAMED_MODELS.find(m => m.id === repoId)
        expect(found).toBeDefined()
      })
    }
  })

  describe('NAMED_MODELS IDs are unique', () => {
    it('no duplicate IDs', () => {
      const ids = NAMED_MODELS.map(m => m.id)
      expect(new Set(ids).size).toBe(ids.length)
    })
  })

  describe('NAMED_MODELS categories are valid', () => {
    for (const model of NAMED_MODELS) {
      it(`${model.id} has valid category`, () => {
        expect(['generate', 'edit']).toContain(model.category)
      })
    }
  })

  describe('NAMED_MODELS quantize options are all valid QUANTIZE_OPTIONS values', () => {
    const validValues = QUANTIZE_OPTIONS.map(o => o.value)
    for (const model of NAMED_MODELS) {
      for (const q of model.quantizeOptions) {
        it(`${model.id} quantize=${q} is a valid QUANTIZE_OPTIONS value`, () => {
          expect(validValues).toContain(q)
        })
      }
    }
  })

  describe('every NAMED_MODEL has at least one quantize option', () => {
    for (const model of NAMED_MODELS) {
      it(`${model.id} has >= 1 quantize option`, () => {
        expect(model.quantizeOptions.length).toBeGreaterThanOrEqual(1)
      })
    }
  })

  describe('generation models have expected characteristics', () => {
    const genModels = NAMED_MODELS.filter(m => m.category === 'generate')

    it('there are at least 3 generation models', () => {
      expect(genModels.length).toBeGreaterThanOrEqual(3)
    })
  })

  describe('edit models have expected characteristics', () => {
    const editModels = NAMED_MODELS.filter(m => m.category === 'edit')

    it('there are at least 1 edit model', () => {
      expect(editModels.length).toBeGreaterThanOrEqual(1)
    })
  })

  describe('repo URLs look valid (org/repo format)', () => {
    for (const [modelId, repos] of Object.entries(IMAGE_MODEL_REPOS)) {
      for (const [q, url] of Object.entries(repos)) {
        it(`${modelId} q=${q}: "${url}" matches org/repo format`, () => {
          expect(url).toMatch(/^[a-zA-Z0-9_-]+\/[a-zA-Z0-9._-]+$/)
        })
      }
    }
  })
})

// =============================================================================
// 3. connectHost (from src/main/sessions.ts and src/renderer/.../ApiDashboard.tsx)
// =============================================================================

function connectHost(host: string): string {
  return host === '0.0.0.0' ? '127.0.0.1' : host
}

describe('connectHost', () => {
  it('converts 0.0.0.0 to 127.0.0.1', () => {
    expect(connectHost('0.0.0.0')).toBe('127.0.0.1')
  })

  it('preserves 127.0.0.1', () => {
    expect(connectHost('127.0.0.1')).toBe('127.0.0.1')
  })

  it('preserves localhost', () => {
    expect(connectHost('localhost')).toBe('localhost')
  })

  it('preserves specific IP addresses', () => {
    expect(connectHost('192.168.1.100')).toBe('192.168.1.100')
    expect(connectHost('10.0.0.1')).toBe('10.0.0.1')
  })

  it('preserves ::1 (IPv6 loopback)', () => {
    expect(connectHost('::1')).toBe('::1')
  })

  it('preserves :: (IPv6 any)', () => {
    expect(connectHost('::')).toBe('::')
  })

  it('preserves empty string', () => {
    expect(connectHost('')).toBe('')
  })

  it('preserves hostname strings', () => {
    expect(connectHost('my-server.local')).toBe('my-server.local')
  })
})

// =============================================================================
// 4. getSizeEstimate (from src/main/ipc/image.ts)
// =============================================================================

function getSizeEstimate(modelName: string): string {
  const sizes: Record<string, string> = {
    'schnell': '~12 GB',
    'dev': '~24 GB',
    'z-image-turbo': '~12 GB',
    'flux2-klein-4b': '~8 GB',
    'flux2-klein-9b': '~16 GB',
  }
  return sizes[modelName] || '~12 GB'
}

describe('getSizeEstimate', () => {
  it('schnell -> ~12 GB', () => expect(getSizeEstimate('schnell')).toBe('~12 GB'))
  it('dev -> ~24 GB', () => expect(getSizeEstimate('dev')).toBe('~24 GB'))
  it('z-image-turbo -> ~12 GB', () => expect(getSizeEstimate('z-image-turbo')).toBe('~12 GB'))
  it('flux2-klein-4b -> ~8 GB', () => expect(getSizeEstimate('flux2-klein-4b')).toBe('~8 GB'))
  it('flux2-klein-9b -> ~16 GB', () => expect(getSizeEstimate('flux2-klein-9b')).toBe('~16 GB'))
  it('unknown model -> ~12 GB (default)', () => expect(getSizeEstimate('custom')).toBe('~12 GB'))
  it('edit model (not in sizes) -> ~12 GB', () => expect(getSizeEstimate('qwen-image-edit')).toBe('~12 GB'))
})

// =============================================================================
// 5. Gated error detection regex (from ImageTab.tsx)
// =============================================================================

describe('gated error detection regex', () => {
  const gatedRegex = /40[13]|gated|access.*denied|authentication|authorized|forbidden/i

  it('matches HTTP 401', () => expect(gatedRegex.test('Server returned 401')).toBe(true))
  it('matches HTTP 403', () => expect(gatedRegex.test('Server returned 403')).toBe(true))
  it('does not match HTTP 404', () => expect(gatedRegex.test('Server returned 404')).toBe(false))
  it('matches "gated"', () => expect(gatedRegex.test('This model is gated')).toBe(true))
  it('matches "access denied"', () => expect(gatedRegex.test('Access denied for this resource')).toBe(true))
  it('matches "Access Denied" (case insensitive)', () => expect(gatedRegex.test('Access Denied')).toBe(true))
  it('matches "authentication"', () => expect(gatedRegex.test('Authentication required')).toBe(true))
  it('matches "authorized"', () => expect(gatedRegex.test('Not authorized')).toBe(true))
  it('matches "forbidden"', () => expect(gatedRegex.test('Forbidden')).toBe(true))
  it('does not match normal errors', () => expect(gatedRegex.test('Connection timeout')).toBe(false))
  it('does not match "200 OK"', () => expect(gatedRegex.test('200 OK')).toBe(false))
})

// =============================================================================
// 6. Data URL prefix stripping (from image.ts edit handler)
// =============================================================================

describe('data URL prefix stripping', () => {
  function stripDataUrlPrefix(imageBase64: string): string {
    return imageBase64.replace(/^data:image\/\w+;base64,/, '')
  }

  it('strips PNG prefix', () => {
    expect(stripDataUrlPrefix('data:image/png;base64,abc123')).toBe('abc123')
  })
  it('strips JPEG prefix', () => {
    expect(stripDataUrlPrefix('data:image/jpeg;base64,xyz')).toBe('xyz')
  })
  it('strips WEBP prefix', () => {
    expect(stripDataUrlPrefix('data:image/webp;base64,data')).toBe('data')
  })
  it('strips GIF prefix', () => {
    expect(stripDataUrlPrefix('data:image/gif;base64,gifdata')).toBe('gifdata')
  })
  it('preserves raw base64 (no prefix)', () => {
    expect(stripDataUrlPrefix('abc123def456')).toBe('abc123def456')
  })
  it('preserves empty string', () => {
    expect(stripDataUrlPrefix('')).toBe('')
  })
  it('does not strip non-image data URLs', () => {
    // \w+ won't match "svg+xml" because of the +, so this is preserved correctly
    expect(stripDataUrlPrefix('data:image/svg+xml;base64,abc')).toBe('data:image/svg+xml;base64,abc')
  })
})
