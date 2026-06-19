import { describe, expect, it } from 'vitest'

import {
  appendMetalWiredLimitGuidance,
  metalWiredLimitCommand,
  metalWiredLimitHelpText,
} from '../src/shared/metalWiredLimit'

describe('Metal wired-memory limit guidance', () => {
  it('includes the sudo sysctl command in user-facing help text', () => {
    expect(metalWiredLimitHelpText).toContain(metalWiredLimitCommand)
    expect(metalWiredLimitHelpText).toContain('115000-120000 MB')
    expect(metalWiredLimitHelpText).toContain('Do not set it equal to physical RAM')
    expect(metalWiredLimitHelpText).toContain('admin password')
    expect(metalWiredLimitHelpText).toContain('resets after reboot')
  })

  it('annotates the exact Metal command-buffer OOM startup error', () => {
    const message =
      'Process exited before becoming ready: libc++abi: terminating due to uncaught exception of type std::runtime_error: [METAL] Command buffer execution failed: Insufficient Memory (00000008:kIOGPUCommandBufferCallbackErrorOutOfMemory)'

    const annotated = appendMetalWiredLimitGuidance(message)

    expect(annotated).toContain(message)
    expect(annotated).toContain('Metal wired-memory limit help')
    expect(annotated).toContain(metalWiredLimitCommand)
  })

  it('does not annotate unrelated backend errors', () => {
    expect(appendMetalWiredLimitGuidance('Process exited before becoming ready: ImportError: mlx missing')).toBe(
      'Process exited before becoming ready: ImportError: mlx missing',
    )
  })

  it('annotates the SIGKILL load failure users see when macOS kills the engine', () => {
    const message =
      'Process was killed (SIGKILL) — likely out of memory. Try a smaller/more quantized model, reduce cache size, or close other apps.'

    const annotated = appendMetalWiredLimitGuidance(message)

    expect(annotated).toContain('Metal wired-memory limit help')
    expect(annotated).toContain('115000-120000 MB')
  })
})
