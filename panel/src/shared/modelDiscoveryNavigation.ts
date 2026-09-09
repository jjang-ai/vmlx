export type DownloadModelType = 'text' | 'image'

// Navigation only: discovering a model never downloads or starts it.
export const IMAGE_MODEL_DISCOVERY_NAVIGATION = {
  mode: 'models',
  downloadModelType: 'image',
} as const

export function resolveDownloadModelType(value: unknown): DownloadModelType {
  return value === 'image' ? 'image' : 'text'
}
