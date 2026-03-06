import { useState, useRef, useCallback, useEffect } from 'react'

interface VoiceChatProps {
  onTranscription: (text: string) => void
  endpoint?: { host: string; port: number }
  sessionId?: string
  sttModel?: string
  disabled?: boolean
}

type RecordingState = 'idle' | 'recording' | 'transcribing'

/**
 * Load audio settings from the app settings store.
 * Settings keys: sttModel, ttsModel, ttsVoice, ttsSpeed
 */
function useAudioSettings() {
  const [settings, setSettings] = useState<{
    sttModel?: string; ttsModel?: string; ttsVoice?: string; ttsSpeed?: number
  }>({})

  useEffect(() => {
    let cancelled = false
    async function load() {
      try {
        const [stt, tts, voice, speed] = await Promise.all([
          window.api.settings.get('sttModel'),
          window.api.settings.get('ttsModel'),
          window.api.settings.get('ttsVoice'),
          window.api.settings.get('ttsSpeed'),
        ])
        if (!cancelled) {
          setSettings({
            sttModel: stt || undefined,
            ttsModel: tts || undefined,
            ttsVoice: voice || undefined,
            ttsSpeed: speed ? parseFloat(speed) : undefined,
          })
        }
      } catch {
        // Settings not available — use hardcoded defaults
      }
    }
    load()
    return () => { cancelled = true }
  }, [])

  return settings
}

/**
 * Microphone button component that records audio and transcribes via STT.
 * Uses MediaRecorder API for recording, sends to /v1/audio/transcriptions.
 *
 * STT model resolution: explicit prop > settings('sttModel') > 'whisper-large-v3'
 */
export function VoiceChat({ onTranscription, endpoint, sessionId, sttModel, disabled }: VoiceChatProps) {
  const [state, setState] = useState<RecordingState>('idle')
  const [error, setError] = useState<string | null>(null)
  const mediaRecorderRef = useRef<MediaRecorder | null>(null)
  const chunksRef = useRef<Blob[]>([])
  const streamRef = useRef<MediaStream | null>(null)
  const audioSettings = useAudioSettings()

  const effectiveSttModel = sttModel || audioSettings.sttModel || 'whisper-large-v3'

  // Cleanup on unmount
  useEffect(() => {
    return () => {
      if (streamRef.current) {
        streamRef.current.getTracks().forEach(t => t.stop())
      }
    }
  }, [])

  const startRecording = useCallback(async () => {
    setError(null)
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true })
      streamRef.current = stream

      // Prefer webm/opus (well-supported in Chromium) with WAV as fallback
      const mimeType = MediaRecorder.isTypeSupported('audio/webm;codecs=opus')
        ? 'audio/webm;codecs=opus'
        : MediaRecorder.isTypeSupported('audio/webm')
          ? 'audio/webm'
          : 'audio/mp4'

      const recorder = new MediaRecorder(stream, { mimeType })
      mediaRecorderRef.current = recorder
      chunksRef.current = []

      recorder.ondataavailable = (e) => {
        if (e.data.size > 0) chunksRef.current.push(e.data)
      }

      recorder.onstop = async () => {
        // Stop all tracks to release the microphone
        stream.getTracks().forEach(t => t.stop())
        streamRef.current = null

        if (chunksRef.current.length === 0) {
          setState('idle')
          return
        }

        setState('transcribing')
        try {
          const blob = new Blob(chunksRef.current, { type: mimeType })
          // Convert to base64 for IPC transport
          const arrayBuffer = await blob.arrayBuffer()
          const base64 = btoa(
            new Uint8Array(arrayBuffer).reduce((data, byte) => data + String.fromCharCode(byte), '')
          )

          const result = await window.api.audio.transcribe({
            audioBase64: base64,
            model: effectiveSttModel,
            endpoint,
            sessionId
          })

          if (result.text && result.text.trim()) {
            onTranscription(result.text.trim())
          }
        } catch (err: any) {
          console.error('Transcription failed:', err)
          const msg = err.message || 'Transcription failed'
          if (msg.includes('mlx-audio') || msg.includes('mlx_audio')) {
            setError('STT not available — mlx-audio not installed on server')
          } else {
            setError(msg.length > 80 ? msg.slice(0, 80) + '…' : msg)
          }
        } finally {
          setState('idle')
        }
      }

      recorder.start(100) // Collect data every 100ms
      setState('recording')
    } catch (err: any) {
      console.error('Failed to start recording:', err)
      if (err.name === 'NotAllowedError') {
        setError('Microphone access denied')
      } else {
        setError(err.message || 'Failed to start recording')
      }
    }
  }, [endpoint, effectiveSttModel, onTranscription])

  const stopRecording = useCallback(() => {
    if (mediaRecorderRef.current && mediaRecorderRef.current.state === 'recording') {
      mediaRecorderRef.current.stop()
    }
  }, [])

  const handleClick = () => {
    if (state === 'recording') {
      stopRecording()
    } else if (state === 'idle') {
      startRecording()
    }
  }

  return (
    <div className="relative">
      <button
        onClick={handleClick}
        disabled={disabled || state === 'transcribing'}
        className={`px-3 py-3 rounded-lg border transition-colors ${
          state === 'recording'
            ? 'bg-destructive/20 border-destructive text-destructive animate-pulse'
            : state === 'transcribing'
              ? 'bg-primary/10 border-primary/30 text-primary'
              : 'bg-background border-input text-muted-foreground hover:text-foreground hover:bg-accent'
        } disabled:opacity-50 disabled:cursor-not-allowed`}
        title={
          state === 'recording' ? 'Click to stop recording'
            : state === 'transcribing' ? 'Transcribing...'
            : 'Click to start voice input'
        }
      >
        {state === 'transcribing' ? (
          // Loading spinner
          <svg xmlns="http://www.w3.org/2000/svg" width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" className="animate-spin">
            <path d="M21 12a9 9 0 1 1-6.219-8.56" />
          </svg>
        ) : (
          // Microphone icon
          <svg xmlns="http://www.w3.org/2000/svg" width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
            <path d="M12 2a3 3 0 0 0-3 3v7a3 3 0 0 0 6 0V5a3 3 0 0 0-3-3Z" />
            <path d="M19 10v2a7 7 0 0 1-14 0v-2" />
            <line x1="12" x2="12" y1="19" y2="22" />
          </svg>
        )}
      </button>
      {error && (
        <div className="absolute bottom-full mb-1 left-1/2 -translate-x-1/2 whitespace-nowrap bg-destructive text-destructive-foreground text-xs px-2 py-1 rounded">
          {error}
        </div>
      )}
    </div>
  )
}

interface TTSPlayerProps {
  text: string
  endpoint?: { host: string; port: number }
  sessionId?: string
  ttsModel?: string
  voice?: string
  speed?: number
  autoPlay?: boolean
}

/**
 * TTS playback: converts text to speech and plays audio.
 * Can auto-play when text changes (for voice mode).
 *
 * TTS settings resolution: explicit prop > settings('ttsModel'/'ttsVoice'/'ttsSpeed') > defaults
 */
export function TTSPlayer({ text, endpoint, sessionId, ttsModel, voice, speed, autoPlay }: TTSPlayerProps) {
  const [playing, setPlaying] = useState(false)
  const [loading, setLoading] = useState(false)
  const audioRef = useRef<HTMLAudioElement | null>(null)
  const audioSettings = useAudioSettings()

  const effectiveModel = ttsModel || audioSettings.ttsModel || 'kokoro'
  const effectiveVoice = voice || audioSettings.ttsVoice || 'af_heart'
  const effectiveSpeed = speed || audioSettings.ttsSpeed || 1.0

  const play = useCallback(async () => {
    if (!text.trim()) return
    setLoading(true)
    try {
      const audioBase64 = await window.api.audio.speak({
        text,
        model: effectiveModel,
        voice: effectiveVoice,
        speed: effectiveSpeed,
        endpoint,
        sessionId
      })

      // Create audio element and play
      const audioUrl = `data:audio/wav;base64,${audioBase64}`
      const audio = new Audio(audioUrl)
      audioRef.current = audio

      audio.onended = () => {
        setPlaying(false)
        audioRef.current = null
      }
      audio.onerror = () => {
        setPlaying(false)
        audioRef.current = null
      }

      await audio.play()
      setPlaying(true)
    } catch (err) {
      console.error('TTS playback failed:', err)
    } finally {
      setLoading(false)
    }
  }, [text, endpoint, effectiveModel, effectiveVoice, effectiveSpeed])

  const stop = useCallback(() => {
    if (audioRef.current) {
      audioRef.current.pause()
      audioRef.current = null
      setPlaying(false)
    }
  }, [])

  // Auto-play when text changes in voice mode
  useEffect(() => {
    if (autoPlay && text.trim()) {
      play()
    }
    return () => { stop() }
  }, [text, autoPlay, play, stop])

  if (!text.trim()) return null

  return (
    <button
      onClick={playing ? stop : play}
      disabled={loading}
      className="inline-flex items-center gap-1 text-xs text-muted-foreground hover:text-foreground transition-colors ml-2"
      title={playing ? 'Stop playback' : 'Read aloud'}
    >
      {loading ? (
        <svg xmlns="http://www.w3.org/2000/svg" width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" className="animate-spin">
          <path d="M21 12a9 9 0 1 1-6.219-8.56" />
        </svg>
      ) : playing ? (
        <svg xmlns="http://www.w3.org/2000/svg" width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
          <rect width="4" height="14" x="6" y="5" />
          <rect width="4" height="14" x="14" y="5" />
        </svg>
      ) : (
        <svg xmlns="http://www.w3.org/2000/svg" width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
          <polygon points="11 5 6 9 2 9 2 15 6 15 11 19 11 5" />
          <path d="M15.54 8.46a5 5 0 0 1 0 7.07" />
          <path d="M19.07 4.93a10 10 0 0 1 0 14.14" />
        </svg>
      )}
      {playing ? 'stop' : 'speak'}
    </button>
  )
}
