import { ipcMain } from 'electron'
import { v4 as uuidv4 } from 'uuid'
import { join, resolve } from 'path'
import { homedir } from 'os'
import { mkdirSync, writeFileSync, existsSync, unlinkSync, readdirSync, rmdirSync, readFileSync } from 'fs'
import { sessionManager } from '../sessions'
import { db } from '../database'
// startServer uses DB-backed path lookup via db.getImageModelPath()
import type { ServerConfig } from '../server'
import type { ImageSession, ImageGeneration } from '../database'

let handlersRegistered = false

// Track the current image server session ID (only one at a time)
let activeImageSessionId: string | null = null
let activeGenerationController: AbortController | null = null

// Serialize startServer calls to prevent race conditions when the user
// rapidly switches models (e.g., clicks model A then immediately model B).
// Without this lock, both calls can read the same activeImageSessionId,
// double-stop the same session, and both create new servers — leaving an
// orphaned server process that nobody tracks.
let startServerChain: Promise<any> = Promise.resolve()

/** Build fetch headers for image server requests, including auth if API key is configured. */
function getImageFetchHeaders(): Record<string, string> {
  const headers: Record<string, string> = { 'Content-Type': 'application/json' }
  if (activeImageSessionId) {
    try {
      const session = db.getSession(activeImageSessionId)
      if (session?.config) {
        const cfg = JSON.parse(session.config)
        if (cfg.apiKey) {
          headers['Authorization'] = `Bearer ${cfg.apiKey}`
        }
      }
    } catch { /* ignore parse errors */ }
  }
  return headers
}

export function registerImageHandlers(): void {
  if (handlersRegistered) return
  handlersRegistered = true

  // ─── Image Session CRUD ──────────────────────────────────────────────

  ipcMain.handle('image:createSession', async (_, modelName: string, sessionType?: 'generate' | 'edit') => {
    try {
      const now = Date.now()
      const session: ImageSession = {
        id: uuidv4(),
        modelName,
        sessionType: sessionType || 'generate',
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
      // 30-minute timeout for image generation
      const timeoutId = setTimeout(() => activeGenerationController?.abort(), 30 * 60 * 1000)
      const resp = await fetch(`${baseUrl}/v1/images/generations`, {
        method: 'POST',
        headers: getImageFetchHeaders(),
        body: JSON.stringify(body),
        signal: activeGenerationController.signal,
      })
      clearTimeout(timeoutId)

      if (!resp.ok) {
        const errText = await resp.text().catch(() => resp.statusText)
        return { success: false, error: `Server returned ${resp.status}: ${errText}` }
      }

      const result = await resp.json() as { data: Array<{ b64_json: string; revised_prompt?: string; seed?: number }> }
      const elapsed = (Date.now() - startTime) / 1000

      // Save each image to disk and database
      const generations: ImageGeneration[] = []
      for (const item of result.data) {
        const genId = uuidv4()
        const imagePath = join(outputDir, `${genId}.png`)

        // Decode base64 and save as PNG
        const buffer = Buffer.from(item.b64_json, 'base64')
        writeFileSync(imagePath, buffer)

        // Use the actual seed from the server response (engine resolves random seeds)
        // so the user can reproduce the same image by entering the seed later
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
          seed: item.seed ?? seed,
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

  // ─── Image Editing ──────────────────────────────────────────────────

  ipcMain.handle('image:edit', async (_, params: {
    sessionId: string
    prompt: string
    negativePrompt?: string
    model: string
    imageBase64: string       // Base64-encoded source image
    maskBase64?: string       // Base64-encoded mask (for inpainting)
    width: number
    height: number
    steps: number
    guidance: number
    strength: number
    seed?: number
    serverPort: number
  }) => {
    try {
      const { sessionId, prompt, model, imageBase64, maskBase64, width, height, steps, guidance, strength, seed, serverPort } = params
      const baseUrl = `http://127.0.0.1:${serverPort}`

      // Ensure output directory exists
      const outputDir = join(homedir(), '.mlxstudio', 'generated', sessionId)
      mkdirSync(outputDir, { recursive: true })

      const startTime = Date.now()

      // Call the image editing endpoint
      // Strip data URL prefix if present (FileReader adds it)
      const cleanImageB64 = imageBase64.replace(/^data:image\/[\w+.-]+;base64,/, '')
      const body: Record<string, any> = {
        prompt,
        model,
        image: cleanImageB64,
        size: `${width}x${height}`,
        steps,
        guidance,
        strength,
        n: 1,
        response_format: 'b64_json'
      }
      if (params.negativePrompt) body.negative_prompt = params.negativePrompt
      if (seed != null) body.seed = seed
      if (maskBase64) body.mask = maskBase64

      activeGenerationController = new AbortController()
      // 30-minute timeout for image edits (Qwen full precision can take 7+ minutes)
      const timeoutId = setTimeout(() => activeGenerationController?.abort(), 30 * 60 * 1000)
      const resp = await fetch(`${baseUrl}/v1/images/edits`, {
        method: 'POST',
        headers: getImageFetchHeaders(),
        body: JSON.stringify(body),
        signal: activeGenerationController.signal,
        // @ts-ignore — Node.js fetch keepalive prevents socket timeout
        keepalive: true,
      })
      clearTimeout(timeoutId)

      if (!resp.ok) {
        const errText = await resp.text().catch(() => resp.statusText)
        return { success: false, error: `Server returned ${resp.status}: ${errText}` }
      }

      const result = await resp.json() as { data: Array<{ b64_json: string; revised_prompt?: string; seed?: number }> }
      const elapsed = (Date.now() - startTime) / 1000

      // Save source image to disk for gallery display
      const srcGenId = uuidv4()
      const sourceImagePath = join(outputDir, `src_${srcGenId}.png`)
      const rawB64 = imageBase64.replace(/^data:image\/[\w+.-]+;base64,/, '')
      const srcBuffer = Buffer.from(rawB64, 'base64')
      writeFileSync(sourceImagePath, srcBuffer)

      // Save edited image to disk and database
      const generations: ImageGeneration[] = []
      for (const item of result.data) {
        const genId = uuidv4()
        const imagePath = join(outputDir, `${genId}.png`)

        const buffer = Buffer.from(item.b64_json, 'base64')
        writeFileSync(imagePath, buffer)

        // Use the actual seed from the server response (engine resolves random seeds)
        const gen: ImageGeneration = {
          id: genId,
          sessionId,
          prompt,
          negativePrompt: params.negativePrompt || undefined,
          modelName: model,
          width,
          height,
          steps,
          guidance,
          seed: item.seed ?? seed,
          strength,
          elapsedSeconds: elapsed,
          imagePath,
          sourceImagePath,
          createdAt: Date.now()
        }
        db.addImageGeneration(gen)
        generations.push(gen)
      }

      return { success: true, generations }
    } catch (error) {
      console.error('[IMAGE] Edit failed:', error)
      return { success: false, error: (error as Error).message }
    }
  })

  // ─── Server Lifecycle ────────────────────────────────────────────────

  ipcMain.handle('image:startServer', async (_, modelName: string, quantize?: number, imageMode?: 'generate' | 'edit', serverSettings?: { host?: string; port?: number; apiKey?: string; logLevel?: string }) => {
    // Serialize concurrent startServer calls to prevent race conditions.
    // Each call chains onto the previous one so only one stop+create+start
    // sequence runs at a time.
    const result = startServerChain = startServerChain
      .catch(() => {})  // Don't let a previous failure block the next call
      .then(async () => {
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

          // Look up model path from DB — no directory scanning needed.
          let modelPath = modelName
          const storedPath = db.getImageModelPath(modelName, quantize || 0)
          if (storedPath && existsSync(storedPath.localPath)) {
            modelPath = storedPath.localPath
            console.log(`[IMAGE] Using stored model path: ${modelPath}`)
          } else {
            // Clean up stale DB entry (file was deleted from disk)
            if (storedPath) {
              console.log(`[IMAGE] Stale DB entry for ${modelName} (quantize=${quantize}): path no longer exists, removing.`)
              db.deleteImageModelPath(modelName, quantize || 0)
            }
            console.log(`[IMAGE] No local model found for ${modelName} (quantize=${quantize}). User must download first.`)
            return { success: false, error: `Model "${modelName}" not downloaded. Use the Download button first.` }
          }

          // Create a session config for image serving
          // imageMode, imageQuantize, and servedModelName are stored in config fields
          // and passed as CLI flags by buildArgs() — NOT via additionalArgs (avoids duplication)
          const mode = imageMode || 'generate'
          const config: Partial<ServerConfig> = {
            host: serverSettings?.host || '0.0.0.0',
            port: serverSettings?.port || 0,  // 0 = auto-assign
            apiKey: serverSettings?.apiKey || '',
            logLevel: (serverSettings?.logLevel || 'INFO') as 'DEBUG' | 'INFO' | 'WARNING' | 'ERROR',
            timeout: 1800,  // 30 minutes — image edits can take 10+ minutes for large models
            modelType: 'image',
            imageMode: mode,
            imageQuantize: quantize || 0,
            // Pass original model ID so engine uses it for loading (not directory name)
            servedModelName: modelPath !== modelName ? modelName : undefined,
          }

          const session = await sessionManager.createSession(modelPath, config)

          // Start the session
          await sessionManager.startSession(session.id)
          activeImageSessionId = session.id

          return { success: true, sessionId: session.id, port: session.port }
        } catch (error) {
          console.error('[IMAGE] Failed to start server:', error)
          return { success: false, error: (error as Error).message }
        }
      })
    return result
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
      const buildResult = (s: any) => {
        const cfg = (() => { try { return JSON.parse(s.config || '{}') } catch { return {} } })()
        // Read imageMode directly from config — no regex guessing
        const imageMode: 'generate' | 'edit' = cfg.imageMode || 'generate'
        // Read quantize from config
        const quantize = cfg.imageQuantize ?? 0
        // Model name: prefer servedModelName (canonical ID like "schnell"),
        // then session modelName, then last path component as fallback
        const modelName = cfg.servedModelName
          || (s.modelName?.includes('/') ? s.modelName.split('/').pop()! : s.modelName)
          || (s.modelPath?.includes('/') ? s.modelPath.split('/').pop()! : s.modelPath)
          || ''
        return {
          sessionId: s.id,
          modelName,
          modelPath: s.modelPath,
          host: s.host,
          port: s.port,
          status: s.status,
          quantize,
          imageMode,
        }
      }

      // First check the tracked active image session
      if (activeImageSessionId) {
        const session = sessionManager.getSession(activeImageSessionId)
        if (session && (session.status === 'running' || session.status === 'loading')) {
          return buildResult(session)
        }
        activeImageSessionId = null
      }

      // Also scan all sessions for any image model (e.g., started from Server tab)
      const allSessions = db.getSessions()
      for (const s of allSessions) {
        if (s.status !== 'running' && s.status !== 'loading') continue
        try {
          const cfg = JSON.parse(s.config || '{}')
          if (cfg.modelType === 'image') {
            activeImageSessionId = s.id  // Adopt it
            return buildResult(s)
          }
        } catch {}
      }

      return null
    } catch (error) {
      return null
    }
  })

  // List ALL running image sessions (gen + edit) so the Image tab can show a selector
  ipcMain.handle('image:getRunningServers', async () => {
    try {
      const results: any[] = []
      const allSessions = db.getSessions()
      for (const s of allSessions) {
        if (s.status !== 'running' && s.status !== 'loading') continue
        try {
          const cfg = JSON.parse(s.config || '{}')
          if (cfg.modelType === 'image') {
            const rawName = s.modelName || s.modelPath || ''
            const modelName = rawName.includes('/') ? rawName.split('/').pop()! : rawName
            results.push({
              sessionId: s.id,
              modelName,
              modelPath: s.modelPath,
              host: s.host,
              port: s.port,
              status: s.status,
              quantize: cfg.imageQuantize ?? 0,
              imageMode: cfg.imageMode || 'generate',
            })
          }
        } catch {}
      }
      return results
    } catch {
      return []
    }
  })

  // ─── Image file reading ──────────────────────────────────────────────

  ipcMain.handle('image:readFile', async (_, imagePath: string) => {
    try {
      // Restrict to ~/.mlxstudio/ for security (prevent arbitrary file reads)
      const allowedDir = resolve(homedir(), '.mlxstudio')
      const resolved = resolve(imagePath)
      if (!resolved.startsWith(allowedDir)) {
        console.warn('[IMAGE] Blocked readFile outside ~/.mlxstudio/:', resolved)
        return null
      }
      if (!existsSync(resolved)) return null
      const data = readFileSync(resolved)
      return `data:image/png;base64,${data.toString('base64')}`
    } catch (error) {
      console.error('[IMAGE] Failed to read image file:', error)
      return null
    }
  })

  // ─── Save image to user-chosen location ──────────────────────────────

  ipcMain.handle('image:saveFile', async (_, imagePath: string) => {
    try {
      // Restrict source to ~/.mlxstudio/ for security (prevent arbitrary file reads)
      const allowedDir = resolve(homedir(), '.mlxstudio')
      const resolved = resolve(imagePath)
      if (!resolved.startsWith(allowedDir)) {
        console.warn('[IMAGE] Blocked saveFile outside ~/.mlxstudio/:', resolved)
        return { success: false, error: 'Source path not allowed' }
      }
      const { dialog } = require('electron')
      const { copyFileSync } = require('fs')
      const fileName = resolved.split('/').pop() || 'image.png'
      const result = await dialog.showSaveDialog({
        defaultPath: fileName,
        filters: [{ name: 'PNG Image', extensions: ['png'] }]
      })
      if (!result.canceled && result.filePath) {
        copyFileSync(resolved, result.filePath)
        return { success: true, path: result.filePath }
      }
      return { success: false }
    } catch (error) {
      return { success: false, error: (error as Error).message }
    }
  })
}
