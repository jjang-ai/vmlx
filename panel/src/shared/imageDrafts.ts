export interface ImageDraft {
  prompt: string
  sourceImage: { dataUrl: string; name: string } | null
  maskBase64: string | null
  iteratePrompt: string | null
  iterateCounter: number
}

export interface ImageDraftSnapshot {
  owner: string | null
  currentSessionId: string | null
  epoch: number
  draft: ImageDraft
}

const emptyDraft = (): ImageDraft => ({
  prompt: '', sourceImage: null, maskBase64: null, iteratePrompt: null, iterateCounter: 0,
})

/** Renderer-lifetime only. No base64 in preferences, no silent eviction of unsent work.
 * Model server identity and image conversation identity are deliberately separate.
 */
export class ImageDraftStore {
  private owners = new Map<string, { current: string | null; drafts: Map<string | null, ImageDraft> }>()
  private listeners = new Set<() => void>()
  private snapshot: ImageDraftSnapshot = { owner: null, currentSessionId: null, epoch: 0, draft: emptyDraft() }

  getSnapshot = (): ImageDraftSnapshot => this.snapshot
  subscribe = (listener: () => void): (() => void) => {
    this.listeners.add(listener)
    return () => { this.listeners.delete(listener) }
  }
  private emit(owner: string, currentSessionId: string | null, draft: ImageDraft, selection = false): void {
    this.snapshot = { owner, currentSessionId, draft, epoch: this.snapshot.epoch + Number(selection) }
    for (const listener of this.listeners) listener()
  }
  activate(owner: string): void {
    if (this.snapshot.owner === owner) return
    let state = this.owners.get(owner)
    if (!state) {
      state = { current: null, drafts: new Map([[null, emptyDraft()]]) }
      this.owners.set(owner, state)
    }
    this.emit(owner, state.current, state.drafts.get(state.current)!, true)
  }
  select(id: string | null): void {
    const owner = this.snapshot.owner
    if (!owner) return
    const state = this.owners.get(owner)!
    state.current = id
    if (!state.drafts.has(id)) state.drafts.set(id, emptyDraft())
    this.emit(owner, id, state.drafts.get(id)!, true)
  }
  isCurrent(token: ImageDraftSnapshot): boolean {
    return token.owner === this.snapshot.owner &&
      token.currentSessionId === this.snapshot.currentSessionId && token.epoch === this.snapshot.epoch
  }
  update(token: ImageDraftSnapshot, patch: Partial<ImageDraft>): boolean {
    if (!token.owner || !this.isCurrent(token)) return false
    const draft = { ...this.snapshot.draft, ...patch }
    this.owners.get(token.owner)!.drafts.set(token.currentSessionId, draft)
    this.emit(token.owner, token.currentSessionId, draft)
    return true
  }
  newConversation(): void {
    const owner = this.snapshot.owner
    if (!owner) return
    this.owners.get(owner)!.drafts.set(null, emptyDraft())
    this.select(null)
  }
  /** A delayed create result belongs to its submitted draft, never the new selection. */
  promote(token: ImageDraftSnapshot, id: string): void {
    if (!token.owner || token.currentSessionId !== null) return
    const state = this.owners.get(token.owner)!
    const draft = this.isCurrent(token) ? this.snapshot.draft : token.draft
    state.drafts.set(id, draft)
    if (this.isCurrent(token)) {
      state.drafts.delete(null)
      this.select(id)
    }
  }
  remove(id: string): void {
    for (const [owner, state] of this.owners) {
      state.drafts.delete(id)
      if (state.current !== id) continue
      state.current = null
      if (!state.drafts.has(null)) state.drafts.set(null, emptyDraft())
      if (this.snapshot.owner === owner) this.emit(owner, null, state.drafts.get(null)!, true)
    }
  }
}
