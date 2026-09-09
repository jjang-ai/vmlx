import { closeSync, fsyncSync, linkSync, openSync, unlinkSync, writeFileSync } from 'fs'
import { dirname } from 'path'
import { randomUUID } from 'crypto'

export interface ImageOutputFile { path: string; data: Buffer }

/** Publish only request-owned new files, then commit history as one transaction.
 * A crash before history commit may leave orphan files, never a history row
 * pointing at an incompletely written output. Existing files are never replaced.
 */
export function publishImageOutputs(files: ImageOutputFile[], commitHistory: () => void): void {
  const temporary = new Set<string>()
  const published: string[] = []
  try {
    for (const file of files) {
      const staging = file.path + '.' + randomUUID() + '.tmp'
      const fd = openSync(staging, 'wx', 0o600)
      temporary.add(staging)
      try {
        writeFileSync(fd, file.data)
        fsyncSync(fd)
      } finally {
        closeSync(fd)
      }
      // Same-directory link publishes complete bytes without clobbering a path
      // created by another actor between validation and publication.
      linkSync(staging, file.path)
      published.push(file.path)
      unlinkSync(staging)
      temporary.delete(staging)
    }
    for (const directory of new Set(files.map(file => dirname(file.path)))) {
      const fd = openSync(directory, 'r')
      try { fsyncSync(fd) } finally { closeSync(fd) }
    }
    commitHistory()
  } catch (error) {
    for (const path of [...temporary, ...published]) {
      try { unlinkSync(path) } catch (cleanupError) {
        console.warn('[IMAGE] Could not clean up unpublished output:', path, cleanupError)
      }
    }
    throw error
  }
}
