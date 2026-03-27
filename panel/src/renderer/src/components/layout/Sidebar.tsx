import { useState } from 'react'
import { SidebarHeader } from './SidebarHeader'
import { ChatHistory } from './ChatHistory'
import { InferenceModeToggle, useInferenceMode } from './InferenceMode'

interface SidebarProps {
  collapsed: boolean
  currentChatId: string | null
  onChatSelect: (chatId: string, modelPath: string) => void
  onNewChat: () => void
}

export function Sidebar({ collapsed, currentChatId, onChatSelect, onNewChat }: SidebarProps) {
  const [searchQuery, setSearchQuery] = useState('')
  const { mode, setMode } = useInferenceMode()

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
      <div className="px-3 py-2 border-t border-sidebar-border">
        <InferenceModeToggle mode={mode} onToggle={setMode} />
      </div>
    </div>
  )
}
