import { useState } from 'react'
import { SidebarHeader } from './SidebarHeader'
import { ChatHistory } from './ChatHistory'

interface SidebarProps {
  collapsed: boolean
  currentChatId: string | null
  onChatSelect: (chatId: string, modelPath: string) => void
  onNewChat: () => void
}

export function Sidebar({ collapsed, currentChatId, onChatSelect, onNewChat }: SidebarProps) {
  const [searchQuery, setSearchQuery] = useState('')

  return (
    <div
      className={`h-full bg-sidebar border-r border-sidebar-border flex flex-col transition-all duration-200 overflow-hidden ${
        collapsed ? 'w-0' : 'w-[260px]'
      }`}
    >
      <SidebarHeader
        onNewChat={onNewChat}
        onSearch={setSearchQuery}
      />
      <ChatHistory
        currentChatId={currentChatId}
        onChatSelect={onChatSelect}
        searchQuery={searchQuery}
      />
    </div>
  )
}
