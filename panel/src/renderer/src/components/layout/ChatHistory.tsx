import { useState, useEffect, useRef } from 'react'
import { Pencil, Trash2, X, Check, MessageSquare, Search } from 'lucide-react'

interface ChatSummary {
  id: string
  title: string
  modelId: string
  modelPath: string
  createdAt: number
  updatedAt: number
  messageCount: number
}

interface ChatHistoryProps {
  currentChatId: string | null
  onChatSelect: (chatId: string, modelPath: string) => void
  searchQuery: string
}

interface DateGroup {
  label: string
  chats: ChatSummary[]
}

function groupByDate(chats: ChatSummary[]): DateGroup[] {
  const today = new Date(); today.setHours(0, 0, 0, 0)
  const yesterday = new Date(today); yesterday.setDate(yesterday.getDate() - 1)
  const weekAgo = new Date(today); weekAgo.setDate(weekAgo.getDate() - 7)
  const monthAgo = new Date(today); monthAgo.setDate(monthAgo.getDate() - 30)

  const groups: Record<string, ChatSummary[]> = {
    'Today': [],
    'Yesterday': [],
    'This Week': [],
    'This Month': [],
    'Older': [],
  }

  for (const chat of chats) {
    const ts = chat.updatedAt
    if (ts >= today.getTime()) groups['Today'].push(chat)
    else if (ts >= yesterday.getTime()) groups['Yesterday'].push(chat)
    else if (ts >= weekAgo.getTime()) groups['This Week'].push(chat)
    else if (ts >= monthAgo.getTime()) groups['This Month'].push(chat)
    else groups['Older'].push(chat)
  }

  return Object.entries(groups)
    .filter(([_, chats]) => chats.length > 0)
    .map(([label, chats]) => ({ label, chats }))
}

export function ChatHistory({ currentChatId, onChatSelect, searchQuery }: ChatHistoryProps) {
  const [chats, setChats] = useState<ChatSummary[]>([])
  const [loading, setLoading] = useState(true)
  const [renamingId, setRenamingId] = useState<string | null>(null)
  const [renameValue, setRenameValue] = useState('')
  const renameRef = useRef<HTMLInputElement>(null)
  const chatsRef = useRef(chats)
  chatsRef.current = chats

  const loadChats = async () => {
    try {
      const recent = await window.api.chat.getRecent(200)
      setChats(recent)
    } catch (e) {
      console.error('Failed to load chats:', e)
    } finally {
      setLoading(false)
    }
  }

  useEffect(() => { loadChats() }, [])

  // Refresh when a new chat is opened that we don't know about
  useEffect(() => {
    if (currentChatId && !chatsRef.current.find(c => c.id === currentChatId)) {
      loadChats()
    }
  }, [currentChatId])

  useEffect(() => {
    if (renamingId && renameRef.current) {
      renameRef.current.focus()
      renameRef.current.select()
    }
  }, [renamingId])

  const handleRename = async (id: string) => {
    const trimmed = renameValue.trim()
    if (trimmed && trimmed !== chats.find(c => c.id === id)?.title) {
      await window.api.chat.update(id, { title: trimmed })
      setChats(prev => prev.map(c => c.id === id ? { ...c, title: trimmed } : c))
    }
    setRenamingId(null)
  }

  const handleDelete = async (id: string, e: React.MouseEvent) => {
    e.stopPropagation()
    if (!confirm('Delete this chat?')) return
    await window.api.chat.delete(id)
    // Deselect if the deleted chat was the active one
    if (id === currentChatId) {
      onChatSelect('', '')
    }
    // Reload from DB to get a consistent view (avoids stale backfill on next refresh)
    loadChats()
  }

  // Filter by search
  const filtered = searchQuery
    ? chats.filter(c => c.title.toLowerCase().includes(searchQuery.toLowerCase()))
    : chats

  const groups = groupByDate(filtered)

  if (loading) {
    return <div className="px-3 py-4 text-xs text-muted-foreground">Loading chats...</div>
  }

  if (filtered.length === 0) {
    return (
      <div className="px-3 py-8 flex flex-col items-center text-center">
        {searchQuery ? (
          <>
            <Search className="h-5 w-5 text-muted-foreground/40 mb-2" />
            <span className="text-xs text-muted-foreground">No chats match your search</span>
          </>
        ) : (
          <>
            <MessageSquare className="h-5 w-5 text-muted-foreground/40 mb-2" />
            <span className="text-xs text-muted-foreground">No chats yet</span>
            <span className="text-[10px] text-muted-foreground/50 mt-0.5">Start a new chat to begin</span>
          </>
        )}
      </div>
    )
  }

  return (
    <div className="flex-1 overflow-y-auto">
      {groups.map(group => (
        <div key={group.label}>
          <div className="px-3 pt-3 pb-1 text-[10px] font-semibold uppercase tracking-wider text-muted-foreground">
            {group.label}
          </div>
          {group.chats.map(chat => (
            <div
              key={chat.id}
              onClick={() => onChatSelect(chat.id, chat.modelPath)}
              className={`group flex items-center gap-2 px-3 py-2 mx-1 rounded-md cursor-pointer transition-colors ${
                chat.id === currentChatId
                  ? 'bg-accent text-foreground'
                  : 'text-foreground/80 hover:bg-accent/50'
              }`}
            >
              {renamingId === chat.id ? (
                <div className="flex-1 flex items-center gap-1" onClick={e => e.stopPropagation()}>
                  <input
                    ref={renameRef}
                    value={renameValue}
                    onChange={e => setRenameValue(e.target.value)}
                    onKeyDown={e => {
                      if (e.key === 'Enter') handleRename(chat.id)
                      if (e.key === 'Escape') { setRenamingId(null); setRenameValue('') }
                    }}
                    className="flex-1 bg-background border border-input rounded px-1.5 py-0.5 text-xs"
                  />
                  <button onClick={() => handleRename(chat.id)} className="text-primary">
                    <Check className="h-3 w-3" />
                  </button>
                  <button onClick={() => { setRenamingId(null); setRenameValue('') }} className="text-muted-foreground">
                    <X className="h-3 w-3" />
                  </button>
                </div>
              ) : (
                <>
                  <span className="flex-1 text-xs truncate">{chat.title}</span>
                  <div className="hidden group-hover:flex items-center gap-0.5 flex-shrink-0">
                    <button
                      onClick={(e) => {
                        e.stopPropagation()
                        setRenamingId(chat.id)
                        setRenameValue(chat.title)
                      }}
                      className="p-0.5 text-muted-foreground hover:text-foreground rounded"
                    >
                      <Pencil className="h-3 w-3" />
                    </button>
                    <button
                      onClick={(e) => handleDelete(chat.id, e)}
                      className="p-0.5 text-muted-foreground hover:text-destructive rounded"
                    >
                      <Trash2 className="h-3 w-3" />
                    </button>
                  </div>
                </>
              )}
            </div>
          ))}
        </div>
      ))}
    </div>
  )
}
