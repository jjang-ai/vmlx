import { ipcMain } from 'electron'
import { resolveBaseUrl, getAuthHeaders } from './utils'

/**
 * Audio IPC handlers for STT (transcription) and TTS (speech synthesis).
 * These proxy requests to the vllm-mlx server's /v1/audio/* endpoints.
 */

export function registerAudioHandlers(): void {
  // STT: Transcribe audio to text
  // Accepts: { audioData: ArrayBuffer (base64-encoded), model?: string, language?: string, endpoint?: {host, port} }
  // Returns: { text: string, language?: string, duration?: number }
  ipcMain.handle('audio:transcribe', async (_, opts: {
    audioBase64: string
    model?: string
    language?: string
    endpoint?: { host: string; port: number }
    sessionId?: string
  }) => {
    const baseUrl = await resolveBaseUrl(opts.endpoint)
    const authHeaders = getAuthHeaders(opts.sessionId)

    // Build multipart/form-data with the audio file
    const boundary = `----AudioBoundary${Date.now()}`
    const audioBuffer = Buffer.from(opts.audioBase64, 'base64')

    let body = ''
    // File field
    body += `--${boundary}\r\n`
    body += `Content-Disposition: form-data; name="file"; filename="audio.webm"\r\n`
    body += `Content-Type: audio/webm\r\n\r\n`

    // Model field
    let modelField = ''
    modelField += `--${boundary}\r\n`
    modelField += `Content-Disposition: form-data; name="model"\r\n\r\n`
    modelField += `${opts.model || 'whisper-large-v3'}\r\n`

    // Language field (optional)
    let langField = ''
    if (opts.language) {
      langField += `--${boundary}\r\n`
      langField += `Content-Disposition: form-data; name="language"\r\n\r\n`
      langField += `${opts.language}\r\n`
    }

    const closing = `--${boundary}--\r\n`

    // Build the complete body as a Buffer (binary audio data can't be a string)
    const parts: Buffer[] = [
      Buffer.from(body),
      audioBuffer,
      Buffer.from(`\r\n${modelField}${langField}${closing}`)
    ]
    const bodyBuffer = Buffer.concat(parts)

    const response = await fetch(`${baseUrl}/v1/audio/transcriptions`, {
      method: 'POST',
      headers: {
        'Content-Type': `multipart/form-data; boundary=${boundary}`,
        ...authHeaders
      },
      body: bodyBuffer,
      signal: AbortSignal.timeout(60000) // 60s timeout for transcription
    })

    if (!response.ok) {
      const errText = await response.text()
      throw new Error(`Transcription failed: ${response.status} ${errText}`)
    }

    return await response.json()
  })

  // TTS: Generate speech from text
  // Returns: base64-encoded WAV audio
  ipcMain.handle('audio:speak', async (_, opts: {
    text: string
    model?: string
    voice?: string
    speed?: number
    endpoint?: { host: string; port: number }
    sessionId?: string
  }) => {
    const baseUrl = await resolveBaseUrl(opts.endpoint)
    const authHeaders = getAuthHeaders(opts.sessionId)

    // Build form-data (the endpoint expects form fields, not JSON)
    const params = new URLSearchParams()
    params.set('model', opts.model || 'kokoro')
    params.set('input', opts.text)
    params.set('voice', opts.voice || 'af_heart')
    params.set('speed', String(opts.speed || 1.0))
    params.set('response_format', 'wav')

    const response = await fetch(`${baseUrl}/v1/audio/speech`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/x-www-form-urlencoded', ...authHeaders },
      body: params.toString(),
      signal: AbortSignal.timeout(120000) // 120s timeout for TTS
    })

    if (!response.ok) {
      const errText = await response.text()
      throw new Error(`TTS failed: ${response.status} ${errText}`)
    }

    const arrayBuffer = await response.arrayBuffer()
    return Buffer.from(arrayBuffer).toString('base64')
  })

  // List available TTS voices for a model
  ipcMain.handle('audio:voices', async (_, opts: {
    model?: string
    endpoint?: { host: string; port: number }
    sessionId?: string
  }) => {
    const baseUrl = await resolveBaseUrl(opts.endpoint)
    const authHeaders = getAuthHeaders(opts.sessionId)
    const model = opts.model || 'kokoro'

    const response = await fetch(`${baseUrl}/v1/audio/voices?model=${encodeURIComponent(model)}`, {
      headers: { ...authHeaders },
      signal: AbortSignal.timeout(10000)
    })

    if (!response.ok) {
      const errText = await response.text()
      throw new Error(`Failed to list voices: ${response.status} ${errText}`)
    }

    return await response.json()
  })
}
