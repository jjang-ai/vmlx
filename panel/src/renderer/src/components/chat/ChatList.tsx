import { useEffect, useState, useRef } from 'react'
import { Pencil, Trash2, Folder, X, MessageSquare, Search } from 'lucide-react'

interface Chat {
  id: string
  title: string
  folderId?: string
  createdAt: number
  updatedAt: number
  modelId: string
}

interface Folder {
  id: string
  name: string
  parentId?: string
  createdAt: number
}

interface ChatListProps {
  currentChatId: string | null
  onChatSelect: (chatId: string) => void
  onNewChat: () => void
  modelPath?: string
}

export function ChatList({ currentChatId, onChatSelect, onNewChat, modelPath }: ChatListProps) {
  const [chats, setChats] = useState<Chat[]>([])
  const [folders, setFolders] = useState<Folder[]>([])
  const [loading, setLoading] = useState(true)
  const [renamingId, setRenamingId] = useState<string | null>(null)
  const [renameValue, setRenameValue] = useState('')
  const [newFolderName, setNewFolderName] = useState('')
  const [showNewFolder, setShowNewFolder] = useState(false)
  const [searchQuery, setSearchQuery] = useState('')
  const [searchResults, setSearchResults] = useState<Chat[] | null>(null)
  const [exportingId, setExportingId] = useState<string | null>(null)
  const renameInputRef = useRef<HTMLInputElement>(null)

  const loadChats = async () => {
    try {
      const allChats = modelPath
        ? await window.api.chat.getByModel(modelPath)
        : await window.api.chat.getAll()
      setChats(allChats)

      const allFolders = await window.api.chat.getFolders()
      setFolders(allFolders)
    } catch (error) {
      console.error('Failed to load chats:', error)
    } finally {
      setLoading(false)
    }
  }

  useEffect(() => {
    loadChats()
  }, [modelPath])

  // Reload when current chat changes (new chat created, chat deleted, etc.)
  useEffect(() => {
    loadChats()
  }, [currentChatId])

  useEffect(() => {
    if (renamingId && renameInputRef.current) {
      renameInputRef.current.focus()
      renameInputRef.current.select()
    }
  }, [renamingId])

  const handleDelete = async (id: string, e: React.MouseEvent) => {
    e.stopPropagation()
    try {
      // Abort any active stream before deleting (prevents orphaned generation)
      try { await window.api.chat.abort(id) } catch {}
      await window.api.chat.delete(id)
      setChats(prev => prev.filter(c => c.id !== id))
      if (id === currentChatId) {
        onChatSelect('')
      }
      // Force reload to sync with DB
      await loadChats()
    } catch (err) {
      console.error('[ChatList] Failed to delete chat:', err)
    }
  }

  const handleRenameStart = (chat: Chat, e: React.MouseEvent) => {
    e.stopPropagation()
    setRenamingId(chat.id)
    setRenameValue(chat.title)
  }

  const handleRenameSubmit = async (id: string) => {
    const trimmed = renameValue.trim()
    if (trimmed && trimmed !== chats.find(c => c.id === id)?.title) {
      await window.api.chat.update(id, { title: trimmed, updatedAt: Date.now() })
      setChats(prev => prev.map(c => c.id === id ? { ...c, title: trimmed, updatedAt: Date.now() } : c))
    }
    setRenamingId(null)
  }

  const handleRenameKeyDown = (e: React.KeyboardEvent, id: string) => {
    if (e.key === 'Enter') {
      e.preventDefault()
      handleRenameSubmit(id)
    } else if (e.key === 'Escape') {
      setRenamingId(null)
    }
  }

  const handleCreateFolder = async () => {
    const name = newFolderName.trim()
    if (!name) return
    const folder = await window.api.chat.createFolder(name)
    setFolders(prev => [...prev, folder])
    setNewFolderName('')
    setShowNewFolder(false)
  }

  const handleDeleteFolder = async (id: string, e: React.MouseEvent) => {
    e.stopPropagation()
    try {
      await window.api.chat.deleteFolder(id)
      setFolders(prev => prev.filter(f => f.id !== id))
      setChats(prev => prev.map(c => c.folderId === id ? { ...c, folderId: undefined } : c))
    } catch (err) {
      console.error('[ChatList] Failed to delete folder:', err)
    }
  }

  const handleMoveToFolder = async (chatId: string, folderId: string | undefined) => {
    await window.api.chat.update(chatId, { folderId: folderId ?? null, updatedAt: Date.now() })
    setChats(prev => prev.map(c => c.id === chatId ? { ...c, folderId, updatedAt: Date.now() } : c))
  }

  const handleExport = async (chatId: string, format: 'json' | 'markdown' | 'sharegpt', e: React.MouseEvent) => {
    e.stopPropagation()
    setExportingId(null)
    try {
      await window.api.chat.export(chatId, format)
    } catch (err) {
      console.error('Export failed:', err)
    }
  }

  const handleImport = async () => {
    try {
      const result = await window.api.chat.import(modelPath)
      if (result.success && result.chatId) {
        await loadChats()
        onChatSelect(result.chatId)
      }
    } catch (err) {
      console.error('Import failed:', err)
    }
  }

  const handleSearch = async (query: string) => {
    setSearchQuery(query)
    if (!query.trim()) {
      setSearchResults(null)
      return
    }
    try {
      const results = await window.api.chat.search(query)
      setSearchResults(results)
    } catch {
      setSearchResults([])
    }
  }

  const formatDate = (timestamp: number) => {
    const date = new Date(timestamp)
    const now = new Date()
    const diff = now.getTime() - date.getTime()
    const days = Math.floor(diff / (1000 * 60 * 60 * 24))

    if (days === 0) return 'Today'
    if (days === 1) return 'Yesterday'
    if (days < 7) return `${days}d ago`
    return date.toLocaleDateString()
  }

  if (loading) {
    return <div className="p-4 text-center text-muted-foreground">Loading...</div>
  }

  const displayChats = searchResults !== null ? searchResults : chats
  const unfiled = displayChats.filter(c => !c.folderId)
  const byFolder = folders.map(f => ({
    folder: f,
    chats: displayChats.filter(c => c.folderId === f.id)
  }))

  const renderChatItem = (chat: Chat) => (
    <div
      key={chat.id}
      onClick={() => onChatSelect(chat.id)}
      className={`group relative px-3 py-2 rounded cursor-pointer ${
        chat.id === currentChatId
          ? 'bg-primary text-primary-foreground'
          : 'hover:bg-accent'
      }`}
    >
      <div className="flex items-start justify-between gap-2">
        <div className="flex-1 min-w-0">
          {renamingId === chat.id ? (
            <input
              ref={renameInputRef}
              type="text"
              value={renameValue}
              onChange={e => setRenameValue(e.target.value)}
              onBlur={() => handleRenameSubmit(chat.id)}
              onKeyDown={e => handleRenameKeyDown(e, chat.id)}
              onClick={e => e.stopPropagation()}
              className="w-full text-sm font-medium bg-background border border-input rounded px-1 py-0.5 focus:outline-none focus:ring-1 focus:ring-ring"
            />
          ) : (
            <p className="text-sm font-medium truncate">{chat.title}</p>
          )}
          <p className="text-xs opacity-70">{formatDate(chat.updatedAt)}</p>
        </div>
        {renamingId !== chat.id && (
          <div className="flex items-center gap-1 opacity-0 group-hover:opacity-100 flex-shrink-0">
            <button
              onClick={(e) => handleRenameStart(chat, e)}
              className="text-xs hover:text-foreground"
              title="Rename"
            >
              <Pencil className="h-3 w-3" />
            </button>
            <div className="relative">
              <button
                onClick={(e) => { e.stopPropagation(); setExportingId(exportingId === chat.id ? null : chat.id) }}
                className="text-xs hover:text-foreground"
                title="Export"
              >
                ↗
              </button>
              {exportingId === chat.id && (
                <div className="absolute right-0 top-6 z-20 bg-popover border border-border rounded shadow-lg py-1 min-w-[100px]">
                  <button onClick={e => handleExport(chat.id, 'json', e)} className="w-full text-left px-3 py-1 text-xs hover:bg-accent">JSON</button>
                  <button onClick={e => handleExport(chat.id, 'markdown', e)} className="w-full text-left px-3 py-1 text-xs hover:bg-accent">Markdown</button>
                  <button onClick={e => handleExport(chat.id, 'sharegpt', e)} className="w-full text-left px-3 py-1 text-xs hover:bg-accent">ShareGPT</button>
                </div>
              )}
            </div>
            {folders.length > 0 && (
              <select
                value={chat.folderId || ''}
                onClick={e => e.stopPropagation()}
                onChange={e => handleMoveToFolder(chat.id, e.target.value || undefined)}
                className="text-xs bg-transparent border-none cursor-pointer w-5 opacity-60 hover:opacity-100"
                title="Move to folder"
              >
                <option value="">📁</option>
                {folders.map(f => (
                  <option key={f.id} value={f.id}>{f.name}</option>
                ))}
              </select>
            )}
            <button
              onClick={(e) => handleDelete(chat.id, e)}
              className="text-xs hover:text-destructive"
              title="Delete"
            >
              <Trash2 className="h-3 w-3" />
            </button>
          </div>
        )}
      </div>
    </div>
  )

  return (
    <div className="flex flex-col h-full">
      <div className="p-2 space-y-2">
        <div className="flex gap-2">
          <button
            onClick={onNewChat}
            className="flex-1 px-4 py-2 bg-primary text-primary-foreground rounded hover:bg-primary/90 font-medium"
          >
            + New Chat
          </button>
          <button
            onClick={handleImport}
            className="px-3 py-2 border border-border rounded hover:bg-accent text-sm"
            title="Import chat from file"
          >
            Import
          </button>
        </div>

        {/* Search */}
        <input
          type="text"
          value={searchQuery}
          onChange={e => handleSearch(e.target.value)}
          placeholder="Search chats..."
          className="w-full px-3 py-1.5 bg-background border border-input rounded text-sm focus:outline-none focus:ring-1 focus:ring-ring"
        />

        {/* Folder controls */}
        <div className="flex items-center gap-1">
          {showNewFolder ? (
            <div className="flex items-center gap-1 flex-1">
              <input
                type="text"
                value={newFolderName}
                onChange={e => setNewFolderName(e.target.value)}
                onKeyDown={e => { if (e.key === 'Enter') handleCreateFolder(); if (e.key === 'Escape') setShowNewFolder(false) }}
                placeholder="Folder name"
                className="flex-1 px-2 py-1 bg-background border border-input rounded text-xs focus:outline-none focus:ring-1 focus:ring-ring"
                autoFocus
              />
              <button onClick={handleCreateFolder} className="text-xs px-2 py-1 bg-primary text-primary-foreground rounded">OK</button>
              <button onClick={() => setShowNewFolder(false)} className="text-xs px-1 py-1 text-muted-foreground"><X className="h-3 w-3" /></button>
            </div>
          ) : (
            <button
              onClick={() => setShowNewFolder(true)}
              className="text-xs text-muted-foreground hover:text-foreground px-2 py-1"
            >
              + Folder
            </button>
          )}
        </div>
      </div>

      <div className="flex-1 overflow-y-auto p-2 space-y-1">
        {displayChats.length === 0 ? (
          <div className="flex flex-col items-center text-center p-6">
            {searchResults !== null ? (
              <>
                <Search className="h-5 w-5 text-muted-foreground/40 mb-2" />
                <span className="text-sm text-muted-foreground">No results</span>
              </>
            ) : (
              <>
                <MessageSquare className="h-5 w-5 text-muted-foreground/40 mb-2" />
                <span className="text-sm text-muted-foreground">No chats yet</span>
              </>
            )}
          </div>
        ) : (
          <>
            {/* Folders with their chats */}
            {byFolder.filter(g => g.chats.length > 0).map(({ folder, chats: folderChats }) => (
              <div key={folder.id} className="mb-2">
                <div className="group flex items-center gap-1 px-2 py-1 text-xs font-medium text-muted-foreground">
                  <span className="flex items-center gap-1"><Folder className="h-3 w-3" /> {folder.name}</span>
                  <span className="text-[10px] opacity-60">({folderChats.length})</span>
                  <button
                    onClick={(e) => handleDeleteFolder(folder.id, e)}
                    className="ml-auto opacity-0 group-hover:opacity-100 text-[10px] hover:text-destructive"
                    title="Delete folder"
                  >
                    <X className="h-3 w-3" />
                  </button>
                </div>
                <div className="pl-2">
                  {folderChats.map(renderChatItem)}
                </div>
              </div>
            ))}

            {/* Unfiled chats */}
            {unfiled.map(renderChatItem)}
          </>
        )}
      </div>
    </div>
  )
}
