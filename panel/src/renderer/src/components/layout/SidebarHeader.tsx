import { useState } from 'react'
import { Plus, Search, X } from 'lucide-react'
import { useTranslation } from '../../i18n'

interface SidebarHeaderProps {
  onNewChat: () => void
  onSearch: (query: string) => void
}

export function SidebarHeader({ onNewChat, onSearch }: SidebarHeaderProps) {
  const { t } = useTranslation()
  const [searchOpen, setSearchOpen] = useState(false)
  const [query, setQuery] = useState('')

  const handleSearchChange = (value: string) => {
    setQuery(value)
    onSearch(value)
  }

  const closeSearch = () => {
    setQuery('')
    setSearchOpen(false)
    onSearch('')
  }

  return (
    <div className="flex flex-col gap-2 px-3 pt-3 pb-2 border-b border-sidebar-border">
      <div className="flex items-center gap-2">
        <span className="text-xs font-semibold text-foreground tracking-wide flex-1">{t('layout.sidebarHeader.chats')}</span>
        <button
          onClick={() => setSearchOpen(!searchOpen)}
          className="p-1 text-muted-foreground hover:text-foreground rounded hover:bg-accent transition-colors"
          title={t('layout.sidebarHeader.searchTitle')}
        >
          <Search className="h-3.5 w-3.5" />
        </button>
        <button
          onClick={onNewChat}
          className="p-1 text-muted-foreground hover:text-foreground rounded hover:bg-accent transition-colors"
          title={t('layout.sidebarHeader.newTitle')}
        >
          <Plus className="h-3.5 w-3.5" />
        </button>
      </div>

      {searchOpen && (
        <div className="relative">
          <Search className="absolute left-2 top-1/2 -translate-y-1/2 h-3 w-3 text-muted-foreground" />
          <input
            autoFocus
            value={query}
            onChange={e => handleSearchChange(e.target.value)}
            onKeyDown={e => e.key === 'Escape' && closeSearch()}
            placeholder={t('layout.sidebarHeader.searchPlaceholder')}
            className="w-full pl-7 pr-7 py-1.5 bg-background border border-input rounded text-xs focus:outline-none focus:ring-1 focus:ring-ring"
          />
          {query && (
            <button
              onClick={closeSearch}
              className="absolute right-2 top-1/2 -translate-y-1/2 text-muted-foreground hover:text-foreground"
            >
              <X className="h-3 w-3" />
            </button>
          )}
        </div>
      )}
    </div>
  )
}
