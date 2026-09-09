/** A local submission owns busy state before the backend has accepted its job. */
export class ImageSubmissionGuard {
  private revision = 0
  private owner: number | null = null

  begin(): number | null {
    if (this.owner !== null) return null
    this.owner = ++this.revision
    return this.owner
  }

  finish(owner: number): boolean {
    if (this.owner !== owner) return false
    this.owner = null
    ++this.revision
    return true
  }

  snapshot(): number { return this.revision }

  canApply(snapshot: number): boolean {
    return this.owner === null && snapshot === this.revision
  }
}
