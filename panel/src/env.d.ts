import { ElectronAPI } from '@electron-toolkit/preload'

declare global {
  interface Window {
    electron: ElectronAPI
    api: {
      models: {
        scan: () => Promise<any[]>
        info: (modelPath: string) => Promise<any>
        getDirectories: () => Promise<{ directories: string[]; userDirectories: string[]; builtinDirectories: string[] }>
        addDirectory: (dirPath: string) => Promise<{ success: boolean; error?: string }>
        removeDirectory: (dirPath: string) => Promise<{ success: boolean }>
        browseDirectory: () => Promise<{ canceled: boolean; path?: string }>
        detectConfig: (modelPath: string) => Promise<{ family: string; toolParser?: string; reasoningParser?: string; cacheType: string; usePagedCache: boolean; enableAutoToolChoice: boolean; isMultimodal: boolean; description: string }>
        getGenerationDefaults: (modelPath: string) => Promise<{ temperature?: number; topP?: number; topK?: number; repeatPenalty?: number } | null>
        searchHF: (query: string) => Promise<Array<{ id: string; author: string; downloads: number; likes: number; lastModified: string; tags: string[]; pipelineTag?: string }>>
        getRecommendedModels: () => Promise<Array<{ id: string; author: string; downloads: number; likes: number; lastModified: string; tags: string[]; pipelineTag?: string }>>
        downloadModel: (repoId: string) => Promise<{ status: string; path?: string; error?: string }>
        cancelDownload: () => Promise<{ success: boolean; error?: string }>
        getDownloadDir: () => Promise<string>
        setDownloadDir: (dir: string) => Promise<{ success: boolean }>
        browseDownloadDir: () => Promise<{ canceled: boolean; path?: string }>
        onDownloadProgress: (callback: (data: { repoId: string; progress: string }) => void) => () => void
      }
      chat: {
        createFolder: (name: string, parentId?: string) => Promise<any>
        getFolders: () => Promise<any[]>
        deleteFolder: (id: string) => Promise<{ success: boolean }>
        create: (title: string, modelId: string, folderId?: string, modelPath?: string) => Promise<any>
        getAll: (folderId?: string) => Promise<any[]>
        getByModel: (modelPath: string) => Promise<any[]>
        get: (id: string) => Promise<any>
        update: (id: string, updates: any) => Promise<{ success: boolean }>
        delete: (id: string) => Promise<{ success: boolean }>
        search: (query: string) => Promise<any[]>
        getMessages: (chatId: string) => Promise<any[]>
        addMessage: (chatId: string, role: string, content: string) => Promise<any>
        sendMessage: (chatId: string, content: string, endpoint?: { host: string; port: number }, attachments?: Array<{ dataUrl: string; name: string }>) => Promise<any>
        onStream: (callback: (data: any) => void) => () => void
        onComplete: (callback: (data: any) => void) => () => void
        onReasoningDone: (callback: (data: any) => void) => () => void
        onTyping: (callback: (data: any) => void) => () => void
        onToolStatus: (callback: (data: any) => void) => () => void
        abort: (chatId: string) => Promise<void>
        clearAllLocks: () => Promise<{ cleared: number }>
        onAskUser: (callback: (data: any) => void) => () => void
        answerUser: (chatId: string, answer: string) => void
        setOverrides: (chatId: string, overrides: any) => Promise<{ success: boolean }>
        getOverrides: (chatId: string) => Promise<any>
        clearOverrides: (chatId: string) => Promise<{ success: boolean }>
        pickImages: () => Promise<Array<{ dataUrl: string; name: string }>>
        openDirectory: () => Promise<{ canceled: boolean; filePaths: string[] }>
        export: (chatId: string, format: 'json' | 'markdown' | 'sharegpt') => Promise<{ success: boolean; path?: string }>
        import: (modelPath?: string) => Promise<{ success: boolean; chatId?: string; title?: string; messageCount?: number }>
      }
      cache: {
        stats: (endpoint?: { host: string; port: number }) => Promise<any>
        entries: (endpoint?: { host: string; port: number }) => Promise<any>
        warm: (prompts: string[], endpoint?: { host: string; port: number }) => Promise<any>
        clear: (cacheType: string, endpoint?: { host: string; port: number }) => Promise<any>
      }
      audio: {
        transcribe: (opts: { audioBase64: string; model?: string; language?: string; endpoint?: { host: string; port: number } }) => Promise<{ text: string; language?: string; duration?: number }>
        speak: (opts: { text: string; model?: string; voice?: string; speed?: number; endpoint?: { host: string; port: number } }) => Promise<string>
        voices: (opts: { model?: string; endpoint?: { host: string; port: number } }) => Promise<{ voices: string[] }>
      }
      benchmark: {
        run: (sessionId: string, endpoint: { host: string; port: number }, modelPath: string, modelName?: string, options?: { flushCache?: boolean }) => Promise<any>
        history: (modelPath?: string) => Promise<any[]>
        delete: (id: string) => Promise<{ success: boolean }>
        onProgress: (callback: (data: any) => void) => () => void
      }
      embeddings: {
        embed: (texts: string[], endpoint: { host: string; port: number }, model?: string, sessionId?: string) => Promise<any>
      }
      performance: {
        health: (endpoint: { host: string; port: number }) => Promise<any>
      }
      vllm: {
        checkInstallation: () => Promise<{ installed: boolean; path?: string; version?: string; method?: string; bundled?: boolean }>
        detectInstallers: () => Promise<any[]>
        checkEngineVersion: () => Promise<{ current: string; bundled: string; needsUpdate: boolean }>
        installStreaming: (method: 'uv' | 'pip' | 'bundled-update', action: 'install' | 'upgrade', installerPath?: string) => Promise<{ success: boolean; error?: string }>
        cancelInstall: () => Promise<void>
        onInstallLog: (callback: (data: any) => void) => () => void
        onInstallComplete: (callback: (data: any) => void) => () => void
      }
      templates: {
        list: () => Promise<Array<{ id: string; name: string; content: string; category: string; isBuiltin: boolean; createdAt: number }>>
        save: (template: { id: string; name: string; content: string; category: string }) => Promise<{ success: boolean }>
        delete: (id: string) => Promise<{ success: boolean }>
      }
      settings: {
        get: (key: string) => Promise<string | null>
        set: (key: string, value: string) => Promise<{ success: boolean }>
        delete: (key: string) => Promise<{ success: boolean }>
      }
      sessions: {
        list: () => Promise<any[]>
        get: (id: string) => Promise<any>
        create: (modelPath: string, config: any) => Promise<any>
        createRemote: (params: { remoteUrl: string; remoteApiKey?: string; remoteModel: string; remoteOrganization?: string }) => Promise<any>
        start: (sessionId: string) => Promise<{ success: boolean; error?: string }>
        stop: (sessionId: string) => Promise<{ success: boolean; error?: string }>
        delete: (sessionId: string) => Promise<{ success: boolean; error?: string }>
        detect: () => Promise<any[]>
        update: (sessionId: string, config: any) => Promise<{ success: boolean; error?: string }>
        onStarting: (callback: (data: any) => void) => () => void
        onReady: (callback: (data: any) => void) => () => void
        onStopped: (callback: (data: any) => void) => () => void
        onError: (callback: (data: any) => void) => () => void
        onHealth: (callback: (data: any) => void) => () => void
        onLog: (callback: (data: any) => void) => () => void
        onCreated: (callback: (data: any) => void) => () => void
        onDeleted: (callback: (data: any) => void) => () => void
      }
    }
  }
}
