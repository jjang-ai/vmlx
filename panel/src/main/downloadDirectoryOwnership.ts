import { mkdir, rm } from 'fs/promises'

export interface DownloadDirectoryOwner {
  modelDir: string
  ownsModelDir?: boolean
}

export async function prepareDownloadDirectory(job: DownloadDirectoryOwner): Promise<void> {
  // mkdir reports the first directory actually created, avoiding an exists/create race.
  const created = await mkdir(job.modelDir, { recursive: true })
  job.ownsModelDir ??= created === job.modelDir
}

export async function removeOwnedDownloadDirectory(job: DownloadDirectoryOwner): Promise<void> {
  // Restored jobs deliberately lack ownership proof. Never infer it from a marker,
  // an incomplete download, or the fact that this job wrote into the directory.
  if (job.ownsModelDir === true) {
    await rm(job.modelDir, { recursive: true, force: true })
  }
}
