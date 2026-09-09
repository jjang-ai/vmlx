export interface ImageJobProgress {
  requestId: string
  jobId?: string
  phase: 'waiting' | 'preparing' | 'denoising' | 'rendering' | 'encoding' | 'saving'
  stepIndex?: number
  totalSteps?: number
}
