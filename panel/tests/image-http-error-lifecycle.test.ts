import { afterAll, beforeAll, describe, expect, it, vi } from 'vitest'
import { createServer, type Server } from 'node:http'
import { mkdtempSync, rmSync } from 'node:fs'
import { tmpdir } from 'node:os'
import { join } from 'node:path'

const state = vi.hoisted(() => ({ handlers: new Map<string, Function>(), root: '' }))
vi.mock('electron', () => ({ ipcMain: { handle: (name: string, fn: Function) => state.handlers.set(name, fn) } }))
vi.mock('../src/main/sessions', () => ({ sessionManager: {} }))
vi.mock('../src/main/database', () => ({ db: {} }))
vi.mock('os', async (original) => ({ ...await original<object>(), homedir: () => state.root }))

import { registerImageHandlers } from '../src/main/ipc/image'
import { getImageGenerationStatus, resetImageGenerationStateForTests } from '../src/main/ipc/imageGenerationState'

describe('image HTTP failure releases the actual IPC job', () => {
  let server: Server
  let port: number
  let responseStatus = 500
  let responseBody = '{"detail":"test backend rejection"}'
  beforeAll(async () => {
    state.root = mkdtempSync(join(tmpdir(), 'vmlx-image-http-test-'))
    registerImageHandlers()
    server = createServer((req, res) => {
      req.resume()
      req.on('end', () => {
        res.writeHead(responseStatus, { 'Content-Type': 'application/json' })
        res.end(responseBody)
      })
    })
    await new Promise<void>(resolve => server.listen(0, '127.0.0.1', resolve))
    port = (server.address() as { port: number }).port
  })
  afterAll(async () => {
    await new Promise<void>(resolve => server.close(() => resolve()))
    rmSync(state.root, { recursive: true, force: true })
  })
  for (const lane of ['generate', 'edit']) {
    for (const status of [400, 500, 200]) {
      it(lane + ' clears busy state after ' + status + ' failure and permits a retry', async () => {
        resetImageGenerationStateForTests()
        responseStatus = status
        responseBody = status === 200 ? 'not json' : '{"detail":"test backend rejection"}'
        const handler = state.handlers.get('image:' + lane)!
        const params = {
          sessionId: 'owned-test', model: 'qwen-image-edit', prompt: 'test',
          imageBase64: 'dGVzdA==', width: 512, height: 512, steps: 1,
          guidance: 4, count: 1, serverPort: port,
        }
        for (let attempt = 0; attempt < 2; attempt++) {
          const result = await handler({}, params)
          expect(result.success).toBe(false)
          expect(result.error).toBeTruthy()
          expect(getImageGenerationStatus()).toMatchObject({
            generating: false, startTime: null, sessionId: 'owned-test',
          })
        }
      })
    }
  }
})
