import { ipcMain, dialog } from 'electron'
import { writeFileSync, readFileSync } from 'fs'
import { db } from '../database'
import { randomUUID } from 'crypto'

/**
 * Chat export/import IPC handlers.
 * Supports JSON, Markdown, and ShareGPT formats.
 */

export function registerExportHandlers(): void {
  // Export a single chat
  ipcMain.handle('chat:export', async (_, chatId: string, format: 'json' | 'markdown' | 'sharegpt') => {
    const chat = db.getChat(chatId)
    if (!chat) throw new Error('Chat not found')

    const messages = db.getMessages(chatId)
    let content: string
    let ext: string

    switch (format) {
      case 'markdown': {
        const lines = [`# ${chat.title}\n`, `*Exported ${new Date().toLocaleString()}*\n`]
        for (const m of messages) {
          const role = m.role === 'assistant' ? 'Assistant' : m.role === 'user' ? 'User' : 'System'
          lines.push(`## ${role}\n`)
          if (m.reasoningContent && m.role === 'assistant') {
            lines.push(`<details><summary>Thinking</summary>\n\n${m.reasoningContent}\n\n</details>\n`)
          }
          lines.push(m.content + '\n')
        }
        content = lines.join('\n')
        ext = 'md'
        break
      }
      case 'sharegpt': {
        const conversations = messages.map(m => ({
          from: m.role === 'assistant' ? 'gpt' : m.role === 'user' ? 'human' : 'system',
          value: m.content
        }))
        content = JSON.stringify({ conversations }, null, 2)
        ext = 'json'
        break
      }
      default: {
        content = JSON.stringify({
          title: chat.title,
          modelPath: chat.modelPath,
          createdAt: chat.createdAt,
          messages: messages.map(m => ({
            role: m.role,
            content: m.content,
            timestamp: m.timestamp,
            ...(m.reasoningContent ? { reasoning: m.reasoningContent } : {})
          }))
        }, null, 2)
        ext = 'json'
      }
    }

    const safeName = chat.title.replace(/[^a-zA-Z0-9 _-]/g, '').slice(0, 50).trim() || 'chat'
    const result = await dialog.showSaveDialog({
      title: 'Export Chat',
      defaultPath: `${safeName}.${ext}`,
      filters: ext === 'md'
        ? [{ name: 'Markdown', extensions: ['md'] }]
        : [{ name: 'JSON', extensions: ['json'] }]
    })

    if (result.canceled || !result.filePath) return { success: false }

    writeFileSync(result.filePath, content, 'utf-8')
    return { success: true, path: result.filePath }
  })

  // Import a chat from file
  ipcMain.handle('chat:import', async (_, modelPath?: string) => {
    const result = await dialog.showOpenDialog({
      title: 'Import Chat',
      filters: [{ name: 'Chat files', extensions: ['json', 'md'] }],
      properties: ['openFile']
    })

    if (result.canceled || result.filePaths.length === 0) return { success: false }

    const filePath = result.filePaths[0]
    const raw = readFileSync(filePath, 'utf-8')

    let title = 'Imported Chat'
    let messages: Array<{ role: string; content: string; reasoning?: string }> = []

    if (filePath.endsWith('.json')) {
      const parsed = JSON.parse(raw)

      // ShareGPT format
      if (parsed.conversations && Array.isArray(parsed.conversations)) {
        title = 'Imported (ShareGPT)'
        messages = parsed.conversations.map((c: any) => ({
          role: c.from === 'gpt' ? 'assistant' : c.from === 'human' ? 'user' : c.from || 'user',
          content: c.value || ''
        }))
      }
      // vMLX native format
      else if (parsed.messages && Array.isArray(parsed.messages)) {
        title = parsed.title || 'Imported Chat'
        messages = parsed.messages.map((m: any) => ({
          role: m.role || 'user',
          content: m.content || '',
          ...(m.reasoning ? { reasoning: m.reasoning } : {})
        }))
      }
    } else if (filePath.endsWith('.md')) {
      // Parse markdown: ## User / ## Assistant / ## System sections
      title = 'Imported (Markdown)'
      const sections = raw.split(/^## /m).slice(1)
      for (const section of sections) {
        const firstLine = section.split('\n')[0].trim().toLowerCase()
        let content = section.split('\n').slice(1).join('\n').trim()
        const role = firstLine.includes('assistant') || firstLine.includes('gpt') ? 'assistant'
          : firstLine.includes('system') ? 'system' : 'user'
        // Extract reasoning from <details><summary>Thinking</summary>...</details> blocks
        let reasoning: string | undefined
        const detailsMatch = content.match(/^<details><summary>Thinking<\/summary>\s*\n\n([\s\S]*?)\n\n<\/details>\s*\n?/)
        if (detailsMatch) {
          reasoning = detailsMatch[1].trim()
          content = content.slice(detailsMatch[0].length).trim()
        }
        if (content) messages.push({ role, content, ...(reasoning ? { reasoning } : {}) })
      }
    }

    if (messages.length === 0) {
      throw new Error('No messages found in file')
    }

    // Create chat and add messages
    const chatId = randomUUID()
    const now = Date.now()
    db.createChat({
      id: chatId,
      title,
      modelId: 'default',
      modelPath: modelPath || '',
      folderId: undefined,
      createdAt: now,
      updatedAt: now
    })

    for (const msg of messages) {
      db.addMessage({
        id: randomUUID(),
        chatId,
        role: msg.role,
        content: msg.content,
        timestamp: now,
        ...(msg.reasoning ? { reasoningContent: msg.reasoning } : {})
      })
    }

    return { success: true, chatId, title, messageCount: messages.length }
  })
}
