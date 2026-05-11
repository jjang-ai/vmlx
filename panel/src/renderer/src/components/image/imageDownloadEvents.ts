export type ImageDownloadState = 'idle' | 'checking' | 'downloading' | 'ready' | 'error'

export type ActiveImageDownload = {
  jobId: string
  model: string
  quantize: number
}

export function isImageDownloadEventForActive(
  data: any,
  activeDownload: ActiveImageDownload | null,
  downloadState: ImageDownloadState,
): boolean {
  if (!activeDownload) return downloadState === 'downloading'
  if (data?.jobId && data.jobId !== activeDownload.jobId) return false
  if (data?.imageModelName != null && data.imageModelName !== activeDownload.model) return false
  if (data?.imageQuantize != null && Number(data.imageQuantize) !== activeDownload.quantize) return false
  return true
}
