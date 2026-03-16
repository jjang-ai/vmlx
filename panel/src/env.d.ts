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
        getGenerationDefaults: (modelPath: string) => Promise<{ temperature?: number; topP?: number; topK?: number; minP?: number; repeatPenalty?: number } | null>
        searchHF: (query: string, sortBy?: string, sortDir?: string) => Promise<Array<{ id: string; author: string; downloads: number; likes: number; lastModified: string; tags: string[]; pipelineTag?: string; size?: string }>>
        getRecommendedModels: () => Promise<Array<{ id: string; author: string; downloads: number; likes: number; lastModified: string; tags: string[]; pipelineTag?: string }>>
        downloadModel: (repoId: string) => Promise<{ status: string; path?: string; error?: string }>
        cancelDownload: (jobId?: string) => Promise<{ success: boolean; error?: string }>
        getDownloadDir: () => Promise<string>
        setDownloadDir: (dir: string) => Promise<{ success: boolean }>
        browseDownloadDir: () => Promise<{ canceled: boolean; path?: string }>
        onDownloadProgress: (callback: (data: { repoId: string; progress: string }) => void) => () => void
        getDownloadStatus: () => Promise<any>
        onDownloadStarted: (callback: (data: any) => void) => () => void
        onDownloadComplete: (callback: (data: any) => void) => () => void
        onDownloadError: (callback: (data: any) => void) => () => void
        startDownload: (repoId: string) => Promise<{ status: string; path?: string; error?: string }>
        checkImageModel: (modelName: string, quantize?: number) => Promise<{ available: boolean; localPath?: string; repoId?: string }>
        downloadImageModel: (modelName: string, quantize?: number) => Promise<{ jobId?: string; status: string; localPath?: string; repoId?: string; queuePosition?: number }>
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
        isStreaming: (chatId: string) => Promise<boolean>
        clearAllLocks: () => Promise<{ cleared: number }>
        getRecent: (limit?: number) => Promise<any[]>
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
        stats: (endpoint?: { host: string; port: number }, sessionId?: string) => Promise<any>
        entries: (endpoint?: { host: string; port: number }, sessionId?: string) => Promise<any>
        warm: (prompts: string[], endpoint?: { host: string; port: number }, sessionId?: string) => Promise<any>
        clear: (cacheType: string, endpoint?: { host: string; port: number }, sessionId?: string) => Promise<any>
      }
      audio: {
        transcribe: (opts: { audioBase64: string; model?: string; language?: string; endpoint?: { host: string; port: number }; sessionId?: string }) => Promise<{ text: string; language?: string; duration?: number }>
        speak: (opts: { text: string; model?: string; voice?: string; speed?: number; endpoint?: { host: string; port: number }; sessionId?: string }) => Promise<string>
        voices: (opts: { model?: string; endpoint?: { host: string; port: number }; sessionId?: string }) => Promise<{ voices: string[] }>
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
      engine: {
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
      app: {
        getVersion: () => Promise<string>
        onUpdateAvailable: (callback: (data: { currentVersion: string; latestVersion: string; url: string; notes?: string }) => void) => () => void
      }
      developer: {
        info: (modelPath: string) => Promise<{ success: boolean; output: string; error?: string }>
        doctor: (modelPath: string, options?: { noInference?: boolean }) => Promise<{ success: boolean; error?: string }>
        convert: (args: { model: string; output?: string; bits: number; groupSize: number; mode?: string; dtype?: string; force?: boolean; skipVerify?: boolean; trustRemoteCode?: boolean; jangProfile?: string; jangMethod?: string }) => Promise<{ success: boolean; error?: string }>
        cancelOp: () => Promise<{ success: boolean; error?: string }>
        browseOutputDir: () => Promise<string | null>
        onLog: (callback: (data: { data: string }) => void) => () => void
        onComplete: (callback: (data: { success: boolean; cancelled?: boolean; error?: string }) => void) => () => void
      }
      image: {
        createSession: (modelName: string) => Promise<{ success: boolean; session?: any; error?: string }>
        getSessions: () => Promise<any[]>
        getSession: (id: string) => Promise<any>
        deleteSession: (id: string) => Promise<{ success: boolean; error?: string }>
        getGenerations: (sessionId: string) => Promise<any[]>
        generate: (params: {
          sessionId: string; prompt: string; negativePrompt?: string; model: string;
          width: number; height: number; steps: number; guidance: number;
          seed?: number; count: number; quantize?: number; serverPort: number
        }) => Promise<{ success: boolean; generations?: any[]; error?: string }>
        startServer: (modelName: string, quantize?: number) => Promise<{ success: boolean; sessionId?: string; port?: number; error?: string }>
        stopServer: () => Promise<{ success: boolean; error?: string }>
        getRunningServer: () => Promise<{ sessionId: string; modelName: string; host: string; port: number; status: string } | null>
        getModelStatus: (modelName: string) => Promise<{ downloaded: boolean; sizeEstimate: string; modelName: string }>
        readFile: (imagePath: string) => Promise<string | null>
        saveFile: (imagePath: string) => Promise<{ success: boolean; path?: string; error?: string }>
        onServerStarting: (callback: (data: any) => void) => () => void
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
        update: (sessionId: string, config: any) => Promise<{ success: boolean; error?: string; restartRequired?: boolean; changedKeys?: string[] }>
        getLogs: (sessionId: string) => Promise<string[]>
        clearLogs: (sessionId: string) => Promise<{ success: boolean }>
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
