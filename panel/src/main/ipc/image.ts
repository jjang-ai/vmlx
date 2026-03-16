import { ipcMain, BrowserWindow } from 'electron'
import { v4 as uuidv4 } from 'uuid'
import { join } from 'path'
import { homedir } from 'os'
import { mkdirSync, writeFileSync, existsSync, unlinkSync, readdirSync, rmdirSync } from 'fs'
import { sessionManager, connectHost } from '../sessions'
import { db } from '../database'
import type { ServerConfig } from '../server'
import type { ImageSession, ImageGeneration } from '../database'

let handlersRegistered = false

// Track the current image server session ID (only one at a time)
let activeImageSessionId: string | null = null
let activeGenerationController: AbortController | null = null

export function registerImageHandlers(getWindow: () => BrowserWindow | null): void {
  if (handlersRegistered) return
  handlersRegistered = true

  // ─── Image Session CRUD ──────────────────────────────────────────────

  ipcMain.handle('image:createSession', async (_, modelName: string) => {
    try {
      const now = Date.now()
      const session: ImageSession = {
        id: uuidv4(),
        modelName,
        createdAt: now,
        updatedAt: now
      }
      db.createImageSession(session)
      return { success: true, session }
    } catch (error) {
      return { success: false, error: (error as Error).message }
    }
  })

  ipcMain.handle('image:getSessions', async () => {
    try {
      return db.getImageSessions()
    } catch (error) {
      console.error('[IMAGE] Failed to get sessions:', error)
      return []
    }
  })

  ipcMain.handle('image:getSession', async (_, id: string) => {
    try {
      return db.getImageSession(id) || null
    } catch (error) {
      return null
    }
  })

  ipcMain.handle('image:deleteSession', async (_, id: string) => {
    try {
      // Clean up generated image files
      const outputDir = join(homedir(), '.mlxstudio', 'generated', id)
      if (existsSync(outputDir)) {
        try {
          const files = readdirSync(outputDir)
          for (const f of files) unlinkSync(join(outputDir, f))
          rmdirSync(outputDir)
        } catch (e) {
          console.error('[IMAGE] Failed to clean up image files:', e)
        }
      }
      db.deleteImageSession(id)
      return { success: true }
    } catch (error) {
      return { success: false, error: (error as Error).message }
    }
  })

  ipcMain.handle('image:getGenerations', async (_, sessionId: string) => {
    try {
      return db.getImageGenerations(sessionId)
    } catch (error) {
      console.error('[IMAGE] Failed to get generations:', error)
      return []
    }
  })

  // ─── Image Generation ────────────────────────────────────────────────

  ipcMain.handle('image:generate', async (_, params: {
    sessionId: string
    prompt: string
    negativePrompt?: string
    model: string
    width: number
    height: number
    steps: number
    guidance: number
    seed?: number
    count: number
    quantize?: number
    serverPort: number
  }) => {
    try {
      const { sessionId, prompt, negativePrompt, model, width, height, steps, guidance, seed, count, serverPort } = params
      const baseUrl = `http://127.0.0.1:${serverPort}`

      // Ensure output directory exists
      const outputDir = join(homedir(), '.mlxstudio', 'generated', sessionId)
      mkdirSync(outputDir, { recursive: true })

      const startTime = Date.now()

      // Call the image generation endpoint
      const body: Record<string, any> = {
        prompt,
        model,
        size: `${width}x${height}`,
        steps,
        guidance,
        n: count,
        response_format: 'b64_json'
      }
      if (negativePrompt) body.negative_prompt = negativePrompt
      if (seed != null) body.seed = seed
      if (params.quantize != null) body.quantize = params.quantize

      activeGenerationController = new AbortController()
      const resp = await fetch(`${baseUrl}/v1/images/generations`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(body),
        signal: activeGenerationController.signal,
      })

      if (!resp.ok) {
        const errText = await resp.text().catch(() => resp.statusText)
        return { success: false, error: `Server returned ${resp.status}: ${errText}` }
      }

      const result = await resp.json() as { data: Array<{ b64_json: string; revised_prompt?: string }> }
      const elapsed = (Date.now() - startTime) / 1000

      // Save each image to disk and database
      const generations: ImageGeneration[] = []
      for (const item of result.data) {
        const genId = uuidv4()
        const imagePath = join(outputDir, `${genId}.png`)

        // Decode base64 and save as PNG
        const buffer = Buffer.from(item.b64_json, 'base64')
        writeFileSync(imagePath, buffer)

        const gen: ImageGeneration = {
          id: genId,
          sessionId,
          prompt,
          negativePrompt: negativePrompt || undefined,
          modelName: model,
          width,
          height,
          steps,
          guidance,
          seed: seed,
          elapsedSeconds: elapsed,
          imagePath,
          createdAt: Date.now()
        }
        db.addImageGeneration(gen)
        generations.push(gen)
      }

      return { success: true, generations }
    } catch (error) {
      console.error('[IMAGE] Generation failed:', error)
      return { success: false, error: (error as Error).message }
    }
  })

  // ─── Server Lifecycle ────────────────────────────────────────────────

  ipcMain.handle('image:startServer', async (_, modelName: string, quantize?: number) => {
    try {
      // Stop any existing image server first
      if (activeImageSessionId) {
        try {
          await sessionManager.stopSession(activeImageSessionId)
        } catch (e) {
          console.error('[IMAGE] Failed to stop previous image server:', e)
        }
        activeImageSessionId = null
      }

      // For named mflux models, use the name directly (mflux resolves them internally)
      const modelPath = modelName

      // Create a session config for image serving
      // Pass quantize via additionalArgs since it's image-specific
      const quantizeArgs = quantize && quantize > 0 ? `--image-quantize ${quantize}` : ''
      const config: Partial<ServerConfig> = {
        host: '0.0.0.0',
        port: 0,  // auto-assign
        timeout: 600,
        modelType: 'image',
        additionalArgs: quantizeArgs,
      }

      const session = await sessionManager.createSession(modelPath, config)

      // Start the session
      await sessionManager.startSession(session.id)
      activeImageSessionId = session.id

      // Notify renderer
      const win = getWindow()
      if (win && !win.isDestroyed()) {
        win.webContents.send('image:serverStarting', { sessionId: session.id, modelName })
      }

      return { success: true, sessionId: session.id, port: session.port }
    } catch (error) {
      console.error('[IMAGE] Failed to start server:', error)
      return { success: false, error: (error as Error).message }
    }
  })

  ipcMain.handle('image:stopServer', async () => {
    try {
      if (activeImageSessionId) {
        await sessionManager.stopSession(activeImageSessionId)
        activeImageSessionId = null
      }
      return { success: true }
    } catch (error) {
      return { success: false, error: (error as Error).message }
    }
  })

  ipcMain.handle('image:cancelGeneration', async () => {
    if (activeGenerationController) {
      activeGenerationController.abort()
      activeGenerationController = null
      return { success: true }
    }
    return { success: false, error: 'No active generation' }
  })

  ipcMain.handle('image:getRunningServer', async () => {
    try {
      // First check the tracked active image session
      if (activeImageSessionId) {
        const session = sessionManager.getSession(activeImageSessionId)
        if (session && session.status === 'running') {
          return {
            sessionId: session.id,
            modelName: session.modelName || session.modelPath,
            host: session.host,
            port: session.port,
            status: session.status
          }
        }
        activeImageSessionId = null
      }

      // Also scan all running sessions for any image model (e.g., started from Server tab)
      const allSessions = db.getSessions()
      for (const s of allSessions) {
        if (s.status !== 'running') continue
        try {
          const cfg = JSON.parse(s.config || '{}')
          if (cfg.modelType === 'image') {
            activeImageSessionId = s.id  // Adopt it
            return {
              sessionId: s.id,
              modelName: s.modelName || s.modelPath,
              host: s.host,
              port: s.port,
              status: s.status
            }
          }
        } catch {}
      }

      return null
    } catch (error) {
      return null
    }
  })

  ipcMain.handle('image:getModelStatus', async (_, modelName: string) => {
    // Basic status — the ImageModelPicker now uses models:checkImageModel for
    // proper local availability checking with download-before-start flow
    return {
      downloaded: false,
      sizeEstimate: getSizeEstimate(modelName),
      modelName
    }
  })

  // ─── Image file reading ──────────────────────────────────────────────

  ipcMain.handle('image:readFile', async (_, imagePath: string) => {
    try {
      if (!existsSync(imagePath)) return null
      const { readFileSync } = require('fs')
      const data = readFileSync(imagePath)
      return `data:image/png;base64,${data.toString('base64')}`
    } catch (error) {
      console.error('[IMAGE] Failed to read image file:', error)
      return null
    }
  })

  // ─── Save image to user-chosen location ──────────────────────────────

  ipcMain.handle('image:saveFile', async (_, imagePath: string) => {
    try {
      const { dialog } = require('electron')
      const { copyFileSync } = require('fs')
      const fileName = imagePath.split('/').pop() || 'image.png'
      const result = await dialog.showSaveDialog({
        defaultPath: fileName,
        filters: [{ name: 'PNG Image', extensions: ['png'] }]
      })
      if (!result.canceled && result.filePath) {
        copyFileSync(imagePath, result.filePath)
        return { success: true, path: result.filePath }
      }
      return { success: false }
    } catch (error) {
      return { success: false, error: (error as Error).message }
    }
  })
}

function getSizeEstimate(modelName: string): string {
  const sizes: Record<string, string> = {
    'schnell': '~12 GB',
    'dev': '~24 GB',
    'z-image-turbo': '~12 GB',
    'flux2-klein-4b': '~8 GB',
    'flux2-klein-9b': '~16 GB'
  }
  return sizes[modelName] || '~12 GB'
}
