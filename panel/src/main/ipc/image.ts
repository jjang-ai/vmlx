// MLX Studio Image System — mlx.studio — Jinho Jang
import { ipcMain } from 'electron'
import { v4 as uuidv4 } from 'uuid'
import { join, resolve } from 'path'
import { homedir } from 'os'
import { mkdirSync, existsSync, unlinkSync, readdirSync, rmdirSync, readFileSync } from 'fs'
import { publishImageOutputs, type ImageOutputFile } from '../imageOutputPublication'
import { sessionManager } from '../sessions'
import { sameLocalBundlePath } from '../local-bundle-identity'
import { db } from '../database'
import { getImageModel, resolveImageModelArtifact, resolveImageModelFromDirectoryName } from '../../shared/imageModels'
import { resolveLocalImageModelDirectory, localImageModelError, unmountedVolume, editPrecisionAlternative, resolveImageModelForLocalDirectory, inspectLocalImageModel } from '../../shared/imageLocalModel'
import {
  beginImageGeneration,
  bindImageGenerationRequest,
  classifyImageGenerationError,
  clearImageGenerationAfterLocalAbort,
  clearImageGenerationSessionHistory,
  finishImageGeneration,
  getActiveImageGenerationController,
  getImageGenerationStatus,
  markImageGenerationAbort,
  clearImageGenerationAbortReason,
} from './imageGenerationState'
import type { ServerConfig } from '../server'
import type { ImageSession, ImageGeneration } from '../database'
import { loadImageRuntimeSettings, saveImageRuntimeSettings } from '../../shared/imageRuntimeSettings'

function imageSettingsOwner(sessionId: string) {
  const session = sessionManager.getSession(sessionId)
  if (!session || session.type === 'remote') throw new Error('Local image session not found')
  const config = JSON.parse(session.config || '{}')
  if (config.modelType !== 'image') throw new Error('Session is not an image model')
  const model = resolveImageModelForLocalDirectory(session.modelPath)
    || getImageModel(config.servedModelName) || resolveImageModelFromDirectoryName(config.servedModelName || '')
  return { sessionId, modelId: model?.id || config.servedModelName || '', quantize: config.imageQuantize ?? 0 }
}

let handlersRegistered = false

// Track the current image server session ID (only one at a time)
let activeImageSessionId: string | null = null
// Freeze request ownership at submission; later session switches cannot retarget Cancel.
const imageRequestOwners = new WeakMap<AbortController, { port: number; requestId: string; headers: Record<string, string> }>()

// Serialize startServer calls to prevent race conditions when the user
// rapidly switches models (e.g., clicks model A then immediately model B).
// Without this lock, both calls can read the same activeImageSessionId,
// double-stop the same session, and both create new servers — leaving an
// orphaned server process that nobody tracks.
let startServerChain: Promise<any> = Promise.resolve()

function logImageClientJob(serverSessionId: string | null, fields: Record<string, unknown>): void {
  if (!serverSessionId) return
  const data = 'IMAGECLIENT ' + JSON.stringify(fields) + '\n'
  try {
    sessionManager.pushLog(serverSessionId, data)
    sessionManager.emit('session:log', { sessionId: serverSessionId, data })
  } catch (error) {
    console.warn('[IMAGE] Could not publish job log:', error)
  }
}

function findDownloadedImageModelPath(modelName: string, quantize: number): { localPath: string; modelId: string; repoId?: string } | null {
  const modelDef = resolveImageModelFromDirectoryName(modelName) || getImageModel(modelName)
  const modelId = modelDef?.id || modelName
  const artifact = resolveImageModelArtifact(modelId, quantize)
  const repoId = artifact?.repoId
  const repoName = repoId?.split('/').pop()
  if (!repoName) return null

  for (const base of [join(homedir(), '.mlxstudio', 'models', 'image'), join(homedir(), '.mlxstudio', 'models', 'xcreates')]) {
    const candidate = artifact?.subfolder
      ? join(base, repoName, artifact.subfolder)
      : join(base, repoName)
    if (existsSync(candidate)) {
      return { localPath: candidate, modelId, repoId: repoId || undefined }
    }
  }
  return null
}

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

function isExpectedImageServerDisconnectError(error: unknown): boolean {
  const err = error as NodeJS.ErrnoException | undefined
  const code = String(err?.code || '')
  const message = String(err?.message || error || '')
  const cause = (err as any)?.cause
  const wrappedDisconnects = [
    cause,
    (err as any)?.reason,
    (err as any)?.error,
    (err as any)?.detail,
  ].filter(Boolean)
  const nestedErrors = Array.isArray((err as any)?.errors) ? (err as any).errors : []
  return (
    code === 'EPIPE' ||
    code === 'ECONNRESET' ||
    code === 'ERR_STREAM_DESTROYED' ||
    code === 'ERR_STREAM_WRITE_AFTER_END' ||
    /EPIPE|write EPIPE|broken pipe|socket hang up|connection reset|premature close|stream.*destroyed|write after end/i.test(message) ||
    wrappedDisconnects.some((nested) => isExpectedImageServerDisconnectError(nested)) ||
    nestedErrors.some((nested) => isExpectedImageServerDisconnectError(nested))
  )
}

function imageServerRequestWritable(req: any): boolean {
  return (
    !req.closed &&
    !req.destroyed &&
    !req.writableEnded &&
    !req.writableDestroyed &&
    !req.socket?.destroyed
  )
}

function writeImageServerRequestBody(req: any, bodyStr: string): boolean {
  if (!imageServerRequestWritable(req)) return false
  try {
    req.write(bodyStr)
    return true
  } catch (error) {
    if (isExpectedImageServerDisconnectError(error)) return false
    throw error
  }
}

function endImageServerRequest(req: any): boolean {
  if (!imageServerRequestWritable(req)) return false
  try {
    req.end()
    return true
  } catch (error) {
    if (isExpectedImageServerDisconnectError(error)) return false
    throw error
  }
}

function writeAndEndImageServerRequest(req: any, bodyStr: string): boolean {
  return writeImageServerRequestBody(req, bodyStr) && endImageServerRequest(req)
}

function imageServerDisconnectedError(): Error {
  const error = new Error('Image server connection lost before request completed.') as NodeJS.ErrnoException
  error.code = 'EPIPE'
  return error
}

function requestImageServerCancel(controller: AbortController): Promise<void> {
  const owner = imageRequestOwners.get(controller)
  if (!owner) return Promise.resolve()
  return new Promise((resolve, reject) => {
    const http = require('http')
    const bodyStr = JSON.stringify({ request_id: owner.requestId })
    const headers = { ...owner.headers, 'Content-Length': Buffer.byteLength(bodyStr) }
    const req = http.request(`http://127.0.0.1:${owner.port}/v1/images/cancel`, {
      method: 'POST',
      headers,
      agent: false,
      timeout: 5000,
    }, (res: any) => {
      res.resume()
      res.on('end', () => res.statusCode === 200 ? resolve() : reject(new Error(`Image cancel HTTP ${res.statusCode}`)))
    })
    req.on('error', reject)
    req.on('timeout', () => req.destroy(new Error('Image cancel request timed out')))
    writeAndEndImageServerRequest(req, bodyStr)
  })
}

export function registerImageHandlers(): void {
  if (handlersRegistered) return
  handlersRegistered = true

  ipcMain.handle('image:getRuntimeSettings', (_, sessionId: string, adoptLegacy = false) =>
    loadImageRuntimeSettings(db, imageSettingsOwner(sessionId), adoptLegacy === true))
  ipcMain.handle('image:saveRuntimeSettings', (_, sessionId: string, settings: unknown) =>
    saveImageRuntimeSettings(db, imageSettingsOwner(sessionId), settings))

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

  // ms#61: delete a single image generation from the gallery.
  // The gallery can grow to unmanageable size without pruning (reporter's
  // words); we expose per-row delete so users don't need to purge the
  // whole session to clean up.
  // Unlinks the image file (output) and, if present, the source image —
  // BUT only if the source path is under ~/.mlxstudio (don't rm user's
  // home-folder pictures that were only referenced, never copied).
  ipcMain.handle('image:deleteGeneration', async (_, generationId: string) => {
    try {
      const gen = db.getImageGeneration(generationId)
      if (!gen) return { success: false, error: 'generation not found' }
      const mlxstudioRoot = resolve(join(homedir(), '.mlxstudio'))
      const tryUnlink = (p?: string | null): void => {
        if (!p) return
        // Only unlink paths inside ~/.mlxstudio (defensive — never rm a
        // file the user originally chose from their Pictures / Desktop).
        try {
          const abs = resolve(p)
          if (!abs.startsWith(mlxstudioRoot + '/') && abs !== mlxstudioRoot) return
          if (existsSync(abs)) unlinkSync(abs)
        } catch (e) {
          console.error('[IMAGE] Failed to unlink', p, e)
        }
      }
      tryUnlink(gen.imagePath)
      tryUnlink(gen.sourceImagePath)
      db.deleteImageGeneration(generationId)
      return { success: true }
    } catch (error) {
      return { success: false, error: (error as Error).message }
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
    imageBase64?: string    // Source image for img2img (optional)
    strength?: number       // img2img strength (0-1, optional)
  }) => {
    let generationController: AbortController | null = null
    const logOwner = activeImageSessionId
    const clientJobId = uuidv4()
    try {
      const { sessionId, prompt, negativePrompt, model, width, height, steps, guidance, seed, count, serverPort } = params
      const baseUrl = `http://127.0.0.1:${serverPort}`

      // Touch session to reset idle timer — prevents sleep during image generation
      if (activeImageSessionId) sessionManager.touchSession(activeImageSessionId)

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
      // img2img: pass source image + strength to generation endpoint
      if (params.imageBase64 && params.strength != null) {
        const cleanB64 = params.imageBase64.replace(/^data:image\/[\w+.-]+;base64,/, '')
        body.image = cleanB64
        body.strength = params.strength
      }

      const controller = beginImageGeneration(sessionId)
      generationController = controller
      bindImageGenerationRequest(controller, logOwner, clientJobId)
      body.request_id = clientJobId
      imageRequestOwners.set(controller, { port: serverPort, requestId: clientJobId, headers: getImageFetchHeaders() })
      logImageClientJob(logOwner, { client_job_id: clientJobId, phase: 'request_submitted', endpoint: 'generations', model, width, height, steps, guidance, seed, history_session_id: sessionId })
      // 30-minute timeout — use Node.js http.request instead of Electron fetch
      // (Chromium's net stack has its own ~5 min socket timeout that ignores keepalive)
      const timeoutId = setTimeout(() => {
        markImageGenerationAbort(controller, 'timeout')
        void requestImageServerCancel(controller).catch(error => console.warn('[IMAGE] Cancel request failed:', error))
        controller.abort()
      }, 30 * 60 * 1000)
      // Periodically touch session during long image generations (Qwen edits can take 10+ min)
      // to prevent idle timer from triggering sleep mid-generation
      const touchInterval = setInterval(() => {
        if (activeImageSessionId) sessionManager.touchSession(activeImageSessionId)
      }, 60_000) // Every 60 seconds
      let resp: any
      try {
        resp = await new Promise<any>((resolve, reject) => {
          const http = require('http')
          const bodyStr = JSON.stringify(body)
          const headers = { ...getImageFetchHeaders(), 'Content-Length': Buffer.byteLength(bodyStr) }
          const req = http.request(`${baseUrl}/v1/images/generations`, {
            method: 'POST',
            headers,
            agent: false,
            timeout: 30 * 60 * 1000,
          }, (res: any) => {
            let data = ''
            res.on('data', (chunk: any) => { data += chunk })
            res.on('end', () => {
              resolve({ ok: res.statusCode >= 200 && res.statusCode < 300, status: res.statusCode, statusText: res.statusMessage, data })
            })
          })
          req.on('error', reject)
          controller.signal.addEventListener('abort', () => {
            req.destroy()
            reject(new Error('ImageGenerationAborted'))
          })
          try {
            if (!writeAndEndImageServerRequest(req, bodyStr)) {
              reject(imageServerDisconnectedError())
            }
          } catch (error) {
            reject(error)
          }
        })
      } finally {
        clearTimeout(timeoutId)
        clearInterval(touchInterval)
      }

      if (!resp.ok) {
        logImageClientJob(logOwner, { client_job_id: clientJobId, phase: 'http_rejected', status: resp.status })
        return { success: false, error: `Server returned ${resp.status}: ${resp.data?.slice(0, 500) || resp.statusText}` }
      }

      const result = JSON.parse(resp.data) as { data: Array<{ b64_json: string; revised_prompt?: string; seed?: number; image_job_id?: string }> }
      logImageClientJob(logOwner, { client_job_id: clientJobId, phase: 'saving_outputs', image_job_ids: result.data.map(item => item.image_job_id).filter(Boolean) })
      const elapsed = (Date.now() - startTime) / 1000

      // If img2img, save source image to disk for gallery display
      const outputFiles: ImageOutputFile[] = []
      let sourceImagePath: string | undefined
      if (params.imageBase64 && params.strength != null) {
        const srcId = uuidv4()
        sourceImagePath = join(outputDir, `src_${srcId}.png`)
        const rawB64 = params.imageBase64.replace(/^data:image\/[\w+.-]+;base64,/, '')
        outputFiles.push({ path: sourceImagePath, data: Buffer.from(rawB64, 'base64') })
      }

      // Save each image to disk and database
      const generations: ImageGeneration[] = []
      for (const item of result.data) {
        const genId = uuidv4()
        const imagePath = join(outputDir, `${genId}.png`)

        // Decode base64 and save as PNG
        const buffer = Buffer.from(item.b64_json, 'base64')
        outputFiles.push({ path: imagePath, data: buffer })

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
          strength: params.strength,
          elapsedSeconds: elapsed,
          imagePath,
          sourceImagePath,
          createdAt: Date.now()
        }
        generations.push(gen)
      }

      publishImageOutputs(outputFiles, () => db.addImageGenerations(generations))

      logImageClientJob(logOwner, { client_job_id: clientJobId, phase: 'outputs_saved', count: generations.length, history_session_id: sessionId })
      return { success: true, generations }
    } catch (error) {
      console.error('[IMAGE] Generation failed:', error)
      const errorMessage = classifyImageGenerationError(error, generationController)
      logImageClientJob(logOwner, { client_job_id: clientJobId, phase: 'failed', error: errorMessage })
      return {
        success: false,
        error: errorMessage
      }
    } finally {
      // Includes early HTTP rejection returns, not just success/throw paths.
      // A failure before begin must not clear another request's active job.
      if (generationController) finishImageGeneration(generationController)
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
    let generationController: AbortController | null = null
    const logOwner = activeImageSessionId
    const clientJobId = uuidv4()
    try {
      const { sessionId, prompt, model, imageBase64, maskBase64, width, height, steps, guidance, strength, seed, serverPort } = params
      const baseUrl = `http://127.0.0.1:${serverPort}`

      // Touch session to reset idle timer
      if (activeImageSessionId) sessionManager.touchSession(activeImageSessionId)

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
      if (maskBase64) {
        body.mask = maskBase64.replace(/^data:image\/[\w+.-]+;base64,/, '')
      }

      const controller = beginImageGeneration(sessionId)
      generationController = controller
      bindImageGenerationRequest(controller, logOwner, clientJobId)
      body.request_id = clientJobId
      imageRequestOwners.set(controller, { port: serverPort, requestId: clientJobId, headers: getImageFetchHeaders() })
      logImageClientJob(logOwner, { client_job_id: clientJobId, phase: 'request_submitted', endpoint: 'edits', model, width, height, steps, guidance, seed, history_session_id: sessionId })
      // 30-minute timeout for image edits (Qwen full precision can take 10+ minutes)
      // Use Node.js http.request instead of Electron fetch — Chromium's net stack
      // has its own socket timeout (~5 min) that ignores keepalive, causing
      // "fetch failed" errors on long image edits.
      const timeoutId = setTimeout(() => {
        markImageGenerationAbort(controller, 'timeout')
        void requestImageServerCancel(controller).catch(error => console.warn('[IMAGE] Cancel request failed:', error))
        controller.abort()
      }, 30 * 60 * 1000)
      // Periodically touch session during long image edits (Qwen can take 10+ min)
      const touchInterval = setInterval(() => {
        if (activeImageSessionId) sessionManager.touchSession(activeImageSessionId)
      }, 60_000)
      let resp: any
      try {
        resp = await new Promise<any>((resolve, reject) => {
          const http = require('http')
          const bodyStr = JSON.stringify(body)
          const headers = { ...getImageFetchHeaders(), 'Content-Length': Buffer.byteLength(bodyStr) }
          const req = http.request(`${baseUrl}/v1/images/edits`, {
            method: 'POST',
            headers,
            agent: false,
            timeout: 30 * 60 * 1000,
          }, (res: any) => {
            let data = ''
            res.on('data', (chunk: any) => { data += chunk })
            res.on('end', () => {
              resolve({ ok: res.statusCode >= 200 && res.statusCode < 300, status: res.statusCode, statusText: res.statusMessage, data })
            })
          })
          req.on('error', reject)
          controller.signal.addEventListener('abort', () => {
            req.destroy()
            reject(new Error('ImageGenerationAborted'))
          })
          try {
            if (!writeAndEndImageServerRequest(req, bodyStr)) {
              reject(imageServerDisconnectedError())
            }
          } catch (error) {
            reject(error)
          }
        })
      } finally {
        clearTimeout(timeoutId)
        clearInterval(touchInterval)
      }

      if (!resp.ok) {
        logImageClientJob(logOwner, { client_job_id: clientJobId, phase: 'http_rejected', status: resp.status })
        return { success: false, error: `Server returned ${resp.status}: ${resp.data?.slice(0, 500) || resp.statusText}` }
      }

      const result = JSON.parse(resp.data) as { data: Array<{ b64_json: string; revised_prompt?: string; seed?: number; image_job_id?: string }> }
      logImageClientJob(logOwner, { client_job_id: clientJobId, phase: 'saving_outputs', image_job_ids: result.data.map(item => item.image_job_id).filter(Boolean) })
      const elapsed = (Date.now() - startTime) / 1000

      // Save source image to disk for gallery display
      const outputFiles: ImageOutputFile[] = []
      const srcGenId = uuidv4()
      const sourceImagePath = join(outputDir, `src_${srcGenId}.png`)
      const rawB64 = imageBase64.replace(/^data:image\/[\w+.-]+;base64,/, '')
      const srcBuffer = Buffer.from(rawB64, 'base64')
      outputFiles.push({ path: sourceImagePath, data: srcBuffer })

      // Save edited image to disk and database
      const generations: ImageGeneration[] = []
      for (const item of result.data) {
        const genId = uuidv4()
        const imagePath = join(outputDir, `${genId}.png`)

        const buffer = Buffer.from(item.b64_json, 'base64')
        outputFiles.push({ path: imagePath, data: buffer })

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
        generations.push(gen)
      }

      publishImageOutputs(outputFiles, () => db.addImageGenerations(generations))

      logImageClientJob(logOwner, { client_job_id: clientJobId, phase: 'outputs_saved', count: generations.length, history_session_id: sessionId })
      return { success: true, generations }
    } catch (error) {
      console.error('[IMAGE] Edit failed:', error)
      const errorMessage = classifyImageGenerationError(error, generationController)
      logImageClientJob(logOwner, { client_job_id: clientJobId, phase: 'failed', error: errorMessage })
      return {
        success: false,
        error: errorMessage
      }
    } finally {
      if (generationController) finishImageGeneration(generationController)
    }
  })

  // ─── Generation Status (persists across tab switches) ─────────────

  ipcMain.handle('image:isGenerating', async () => {
    return getImageGenerationStatus()
  })

  // ─── Server Lifecycle ────────────────────────────────────────────────

  ipcMain.handle('image:inspectLocalModel', async (_, path: string) => inspectLocalImageModel(path))

  ipcMain.handle('image:startServer', async (_, modelName: string, quantize?: number, imageMode?: 'generate' | 'edit', serverSettings?: { host?: string; port?: number; apiKey?: string; logLevel?: string; mfluxClass?: string }) => {
    // Serialize concurrent startServer calls to prevent race conditions.
    // Each call chains onto the previous one so only one stop+create+start
    // sequence runs at a time.
    const result = startServerChain = startServerChain
      .catch(() => {})  // Don't let a previous failure block the next call
      .then(async () => {
        try {
          // Resolve the requested model BEFORE touching the running server: a
          // rejected folder (unmounted drive, typo, folder of variants) must
          // leave the working server and its session untouched.
          //
          // A typed or browsed LOCAL DIRECTORY is the model itself: use it
          // as-is at the precision the bundle declares, and only fall back to
          // the downloaded-model registry for registry ids / Hugging Face
          // names. Before this, an external-drive folder fell through the
          // registry and the user was told to "download" a model already on
          // disk.
          let modelPath = modelName
          let effectiveQuantize = quantize || 0
          let discoveredRepoId: string | undefined
          let localDir = resolveLocalImageModelDirectory(modelName, effectiveQuantize)
          if (localDir && localDir.kind !== 'model') {
            const err = localImageModelError(localDir)
            console.log(`[IMAGE] Rejected local model path (${err.code}): ${modelName}`)
            return { success: false, error: err.message, errorCode: err.code, errorParams: err.params, serverKept: true }
          }
          if (localDir?.kind === 'model') {
            modelPath = localDir.path
            if (localDir.quantize !== null) effectiveQuantize = localDir.quantize
            console.log(`[IMAGE] Using local model directory: ${modelPath} (quantize=${effectiveQuantize || 'full'}, source=${localDir.quantizeSource || 'none'}${localDir.mfluxVersion ? `, mflux ${localDir.mfluxVersion}` : ''})`)
          }
          // Look up model path from DB — no directory scanning needed.
          const storedPath = localDir?.kind === 'model' ? null : db.getImageModelPath(modelName, effectiveQuantize)
          if (localDir?.kind === 'model') {
            // already resolved from the filesystem above
          } else if (storedPath && existsSync(storedPath.localPath)) {
            modelPath = storedPath.localPath
            console.log(`[IMAGE] Using stored model path: ${modelPath}`)
          } else {
            // The registration's folder is not reachable right now. A drive
            // that is unplugged is not a deleted model: keep the registration
            // either way and only report what is actually the case.
            const storedVolume = storedPath ? unmountedVolume(storedPath.localPath) : null
            if (storedPath) {
              console.log(`[IMAGE] Registered path for ${modelName} (quantize=${quantize}) is not reachable: ${storedPath.localPath}${storedVolume ? ` (volume "${storedVolume}" not mounted)` : ''}; registration kept.`)
            }
            const discovered = findDownloadedImageModelPath(modelName, effectiveQuantize)
            if (discovered) {
              modelPath = discovered.localPath
              discoveredRepoId = discovered.repoId
            } else if (storedPath && storedVolume) {
              return {
                success: false,
                error: `Model "${modelName}" is registered at "${storedPath.localPath}" on the volume "${storedVolume}", which is not mounted. Connect the drive and try again.`,
                errorCode: 'storedVolumeUnavailable',
                errorParams: { model: modelName, path: storedPath.localPath, volume: storedVolume },
                serverKept: true,
              }
            } else {
              console.log(`[IMAGE] No local model found for ${modelName} (quantize=${quantize}). User must download first.`)
              return { success: false, error: `Model "${modelName}" not downloaded. Use the Download button first.`, errorCode: 'notDownloaded', errorParams: { model: modelName }, serverKept: true }
            }
          }

          // A registry entry is only a location hint, not authority for the
          // contents currently at that location. Re-inspect the resolved
          // folder before choosing an adapter, precision or replacement.
          if (localDir?.kind !== 'model') {
            localDir = resolveLocalImageModelDirectory(modelPath, effectiveQuantize)
            if (!localDir) {
              return { success: false, error: 'Registered image model did not resolve to a local folder.', serverKept: true }
            }
            if (localDir.kind !== 'model') {
              const err = localImageModelError(localDir)
              console.warn(`[IMAGE] Rejected registered model path (${err.code}): ${modelPath}; current server kept`)
              return { success: false, error: err.message, errorCode: err.code, errorParams: err.params, serverKept: true }
            }
            modelPath = localDir.path
            if (localDir.quantize !== null) effectiveQuantize = localDir.quantize
            console.log(`[IMAGE] Validated registered model directory: ${modelPath} (quantize=${effectiveQuantize || 'full'}, source=${localDir.quantizeSource || 'none'})`)
          }

          // Resolve the adapter and its defaults BEFORE stopping a working
          // engine. Auto selection must not send a stale Flux1/generate pair.
          // A local inspection's unresolved result is meaningful. Do not
          // resurrect a rejected/unknown declaration from its folder name.
          const modelDef = resolveImageModelForLocalDirectory(modelPath)
          const mfluxName = modelDef?.mfluxName || modelName
          const mfluxClass = serverSettings?.mfluxClass || modelDef?.mfluxClass || ''
          const mode = imageMode || modelDef?.category || 'generate'
          if (!mfluxClass) {
            console.warn(`[IMAGE] Adapter unresolved for ${modelPath}; current server kept`)
            return { success: false, error: 'Could not identify this image model. Select its architecture explicitly or use a folder with supported model metadata.', serverKept: true }
          }
          if (modelDef && (mfluxClass !== modelDef.mfluxClass || mode !== modelDef.category)) {
            return { success: false, error: `The selected folder resolves to ${modelDef.name} (${modelDef.mfluxClass}, ${modelDef.category}), but the selected architecture or mode conflicts. Use automatic detection or select a matching folder.`, serverKept: true }
          }

          if (localDir?.kind === 'model' && modelDef) {
            try { db.setImageModelPath(modelDef.id, effectiveQuantize, modelPath, discoveredRepoId) } catch (e) { console.warn('[IMAGE] Could not register validated local model directory:', e) }
          }

          // Discovery intentionally presents running/loading image engines.
          // Its active pointer can therefore be empty after deep standby or
          // renderer navigation. Explicit Load still owns replacement of the
          // same actual folder: stop that session before createSession's
          // active-session guard, never bypass or weaken the guard itself.
          // Only exact/canonical directory identity is eligible, not basename.
          const replacingSessionIds = new Set<string>()
          if (activeImageSessionId) replacingSessionIds.add(activeImageSessionId)
          for (const existing of db.getSessions()) {
            if (existing.type === 'remote' || !sameLocalBundlePath(existing.modelPath, modelPath)) continue
            try {
              if (JSON.parse(existing.config || '{}').modelType === 'image') {
                replacingSessionIds.add(existing.id)
              }
            } catch { /* Malformed unrelated config is not authority to stop. */ }
          }
          // All replacement validation above must finish before this boundary.
          if (replacingSessionIds.size > 0) {
            const controller = getActiveImageGenerationController()
            if (controller) {
              markImageGenerationAbort(controller, "cancel")
              void requestImageServerCancel(controller).catch(error => console.warn('[IMAGE] Cancel request failed:', error))
              controller.abort()
            }
            clearImageGenerationAfterLocalAbort(controller)
            clearImageGenerationSessionHistory()
            for (const sessionId of replacingSessionIds) {
              // A failed stop must not proceed to create/start or overwrite
              // advertised endpoint/config while its process still owns them.
              await sessionManager.stopSession(sessionId)
            }
            activeImageSessionId = null
          }

          // Create a session config for image serving
          // imageMode, imageQuantize, and servedModelName are stored in config fields
          // and passed as CLI flags by buildArgs() — NOT via additionalArgs (avoids duplication)
          // Look up model definition.
          // mlxstudio#82: use fuzzy resolver (directory basenames like
          // "FLUX.2-klein-9B" or "FLUX.1-dev-mflux-8bit" need more than
          // exact-id match). Fuzzy resolver rule #1 is exact-id so this
          // is a strict superset of getImageModel().
          // A local folder resolves through its own name and, for a precision
          // variant ("q8"), its bundle root: the mflux class must not depend
          // on the caller passing it (the warning's "Use q8" action did not).
          if (modelDef && modelDef.id !== modelName) {
            console.log(`[IMAGE] mlxstudio#82: resolved '${modelName}' -> modelDef id=${modelDef.id}, mfluxClass=${modelDef.mfluxClass}, mfluxName=${modelDef.mfluxName}`)
          }
          // Low precision on an EDIT class. The measured claim is scoped to the
          // artifact it was measured on (Qwen-Image-Edit-mflux through mflux
          // directly: q4 noise + instruction ignored, q8 correct); any other
          // edit class at <= 4-bit gets a neutral "not measured" note. Start it
          // either way (the user chose it) and offer the higher-precision
          // sibling when the folder has one; never switch silently.
          let warningCode: string | undefined
          let warningParams: Record<string, string> | undefined
          if (mode === 'edit' && localDir?.kind === 'model' && localDir.quantize !== null && localDir.quantize <= 4) {
            const alt = editPrecisionAlternative(localDir)
            warningCode = mfluxClass === 'QwenImageEdit' ? 'editLowPrecision' : 'editLowPrecisionUntested'
            warningParams = { bits: String(localDir.quantize), alternative: alt ? alt.name : '', alternativePath: alt ? alt.path : '', alternativeBits: alt && alt.quantize !== null ? String(alt.quantize) : '' }
            console.log(`[IMAGE] Edit class ${mfluxClass || '(unknown)'} at ${localDir.quantize}-bit: ${warningCode}${alt ? `, alternative ${alt.path}` : ''}`)
          }

          const config: Partial<ServerConfig> = {
            host: serverSettings?.host || '127.0.0.1',
            port: serverSettings?.port || 0,  // 0 = auto-assign
            apiKey: serverSettings?.apiKey || '',
            logLevel: (serverSettings?.logLevel || 'INFO') as 'DEBUG' | 'INFO' | 'WARNING' | 'ERROR',
            timeout: 1800,  // 30 minutes — image edits can take 10+ minutes for large models
            modelType: 'image',
            imageMode: mode,
            imageQuantize: effectiveQuantize,
            // Pass mflux canonical name so engine uses the correct class (not directory name)
            servedModelName: mfluxName,
            // Explicit mflux class — buildArgs passes --mflux-class flag
            mfluxClass: mfluxClass || undefined,
          }

          const session = await sessionManager.createSession(modelPath, config)

          // Start the session
          await sessionManager.startSession(session.id)
          activeImageSessionId = session.id

          // The precision actually configured (the bundle's own level for a local
          // folder), so the renderer shows what runs rather than what was picked.
          return { success: true, sessionId: session.id, port: session.port, quantize: effectiveQuantize, modelId: modelDef?.id, imageMode: mode, warningCode, warningParams }
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
        // Cancel any in-flight generation before stopping
        const controller = getActiveImageGenerationController()
        if (controller) {
          markImageGenerationAbort(controller, "cancel")
          void requestImageServerCancel(controller).catch(error => console.warn('[IMAGE] Cancel request failed:', error))
          controller.abort()
        }
        clearImageGenerationAfterLocalAbort(controller)
        clearImageGenerationSessionHistory()
        await sessionManager.stopSession(activeImageSessionId)
        activeImageSessionId = null
      }
      return { success: true }
    } catch (error) {
      return { success: false, error: (error as Error).message }
    }
  })

  ipcMain.handle('image:cancelGeneration', async () => {
    const controller = getActiveImageGenerationController()
    if (controller) {
      markImageGenerationAbort(controller, "cancel")
      // Keep the original request and UI busy until the server releases its
      // worker. A cancellation acknowledgement is not GPU completion.
      try {
        await requestImageServerCancel(controller)
      } catch (error) {
        clearImageGenerationAbortReason(controller)
        throw error
      }
      return { success: true }
    }
    return { success: false, error: 'No active generation' }
  })

  ipcMain.handle('image:getRunningServer', async () => {
    try {
      const buildResult = (s: any) => {
        const cfg = (() => { try { return JSON.parse(s.config || '{}') } catch { return {} } })()
        // Read imageMode from config — always set explicitly by startServer, no name guessing
        const imageMode: 'generate' | 'edit' = cfg.imageMode || 'generate'
        // Read quantize from config
        const quantize = cfg.imageQuantize ?? 0
        // mlxstudio#82: display the directory basename exactly as on disk.
        // Prior flip prefers cfg.servedModelName (canonical mflux form like
        // "flux2-klein-9b") which stripped dots and diverged from the
        // session list / config form — Mark reported seeing "FLUX2" here
        // and "FLUX.2" elsewhere. `servedModelName` is an API routing key,
        // not a display label. Use it only when modelPath is missing.
        const pathBase = s.modelPath?.includes('/') ? s.modelPath.split('/').pop()! : s.modelPath
        const modelName = pathBase
          || (s.modelName?.includes('/') ? s.modelName.split('/').pop()! : s.modelName)
          || cfg.servedModelName
          || ''
        return {
          sessionId: s.id,
          modelName,
          displayModelName: modelName,
          canonicalModelId: cfg.servedModelName || undefined,
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
        if (session && (session.status === 'running' || session.status === 'loading' || session.status === 'standby')) {
          return buildResult(session)
        }
        activeImageSessionId = null
      }

      // Also scan all sessions for any image model (e.g., started from Server tab)
      // Prefer a live model over an unrelated sleeper when this tab has no
      // explicit current owner; do not mutate the database's returned array.
      const allSessions = [...db.getSessions()].sort((a, b) => Number(a.status === 'standby') - Number(b.status === 'standby'))
      for (const s of allSessions) {
        if (s.type === 'remote' || (s.status !== 'running' && s.status !== 'loading' && s.status !== 'standby')) continue
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
            // Prefer servedModelName from config (canonical mflux name, no path guessing)
            const modelName = cfg.servedModelName
              || (s.modelName?.includes('/') ? s.modelName.split('/').pop()! : s.modelName)
              || ''
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
