import { afterAll, beforeAll, describe, expect, it, vi } from 'vitest'
import { createServer, type Server } from 'node:http'
import { mkdtempSync, rmSync, readdirSync } from 'node:fs'
import { tmpdir } from 'node:os'
import { join } from 'node:path'

const state = vi.hoisted(() => ({ handlers: new Map<string, Function>(), root: '' }))
vi.mock('electron', () => ({ ipcMain: { handle: (name: string, fn: Function) => state.handlers.set(name, fn) } }))
vi.mock('../src/main/sessions', () => ({ sessionManager: {} }))
vi.mock('../src/main/database', () => ({ db: {
  addImageGeneration: () => { throw new Error('test history insert failure') },
  addImageGenerations: () => { throw new Error('test history insert failure') },
} }))
vi.mock('os', async (original) => ({ ...await original<object>(), homedir: () => state.root }))

import { registerImageHandlers } from '../src/main/ipc/image'
import { getImageGenerationStatus, resetImageGenerationStateForTests } from '../src/main/ipc/imageGenerationState'

describe('image HTTP failure releases the actual IPC job', () => {
  let server: Server
  let port: number
  let responseStatus = 500
  let responseBody = '{"detail":"test backend rejection"}'
  let cancellation: 'match' | 'foreign' | null = null
  beforeAll(async () => {
    state.root = mkdtempSync(join(tmpdir(), 'vmlx-image-http-test-'))
    registerImageHandlers()
    server = createServer((req, res) => {
      let body = ''
      req.on('data', chunk => { body += chunk })
      req.on('end', () => {
        res.writeHead(responseStatus, { 'Content-Type': 'application/json' })
        res.end(cancellation ? JSON.stringify({detail:{code:'image_generation_cancelled',request_id:cancellation === 'match' ? JSON.parse(body).request_id : 'foreign'}}) : responseBody)
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
    it(lane + ' treats only its matching cancellation as an expected outcome', async () => {
      responseStatus = 409
      try {
        for (const kind of ['match','foreign'] as const) {
          resetImageGenerationStateForTests()
          cancellation = kind
          const result = await state.handlers.get('image:' + lane)!({}, {
            sessionId:'cancelled-' + lane, model:'qwen-image-edit',prompt:'test',imageBase64:'dGVzdA==',
            width:512,height:512,steps:1,guidance:4,count:1,serverPort:port,
          })
          expect(result.success).toBe(false)
          if (kind === 'match') { expect(result.cancelled).toBe(true); expect(result.error).toBeUndefined() }
          else { expect(result.cancelled).toBeUndefined(); expect(result.error).toContain('409') }
          expect(getImageGenerationStatus().generating).toBe(false)
        }
      } finally { cancellation = null }
    })
    it(lane + ' removes its own files when history publication fails', async () => {
      resetImageGenerationStateForTests()
      responseStatus = 200
      responseBody = JSON.stringify({ data: [{ b64_json: Buffer.from('fixture bytes').toString('base64'), seed: 1 }] })
      const result = await state.handlers.get('image:' + lane)!({}, {
        sessionId: 'write-failure-' + lane, model: 'qwen-image-edit', prompt: 'fixture',
        imageBase64: 'dGVzdA==', strength: .75, width: 512, height: 512, steps: 1,
        guidance: 4, count: 1, serverPort: port,
      })
      expect(result.success).toBe(false)
      expect(result.error).toContain('test history insert failure')
      expect(readdirSync(join(state.root, '.mlxstudio', 'generated', 'write-failure-' + lane))).toEqual([])
      expect(getImageGenerationStatus().generating).toBe(false)
    })
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
