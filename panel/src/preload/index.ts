import { contextBridge, ipcRenderer } from 'electron'
import { electronAPI } from '@electron-toolkit/preload'

// Custom APIs for renderer
const api = {
  // Model management
  models: {
    scan: (modelType?: string) => ipcRenderer.invoke('models:scan', modelType),
    info: (modelPath: string) => ipcRenderer.invoke('models:info', modelPath),
    getDirectories: (modelType?: string) => ipcRenderer.invoke('models:getDirectories', modelType),
    addDirectory: (dirPath: string, modelType?: string) => ipcRenderer.invoke('models:addDirectory', dirPath, modelType),
    removeDirectory: (dirPath: string, modelType?: string) => ipcRenderer.invoke('models:removeDirectory', dirPath, modelType),
    browseDirectory: () => ipcRenderer.invoke('models:browseDirectory'),
    detectConfig: (modelPath: string) => ipcRenderer.invoke('models:detect-config', modelPath),
    detectTypes: (modelPaths: string[]) => ipcRenderer.invoke('models:detectTypes', modelPaths),
    getGenerationDefaults: (modelPath: string) => ipcRenderer.invoke('models:getGenerationDefaults', modelPath),
    // HuggingFace search & download
    searchHF: (query: string, sortBy?: string, sortDir?: string, modelType?: string) => ipcRenderer.invoke('models:searchHF', query, sortBy, sortDir, modelType),
    getRecommendedModels: () => ipcRenderer.invoke('models:getRecommendedModels'),
    downloadModel: (repoId: string) => ipcRenderer.invoke('models:downloadModel', repoId),
    startDownload: (repoId: string) => ipcRenderer.invoke('models:startDownload', repoId),
    cancelDownload: (jobId?: string) => ipcRenderer.invoke('models:cancelDownload', jobId),
    getDownloadStatus: () => ipcRenderer.invoke('models:getDownloadStatus'),
    getDownloadDir: () => ipcRenderer.invoke('models:getDownloadDir'),
    setDownloadDir: (dir: string) => ipcRenderer.invoke('models:setDownloadDir', dir),
    browseDownloadDir: () => ipcRenderer.invoke('models:browseDownloadDir'),
    onDownloadProgress: (callback: (data: any) => void) => {
      const handler = (_: any, data: any) => callback(data)
      ipcRenderer.on('models:downloadProgress', handler)
      return () => { ipcRenderer.removeListener('models:downloadProgress', handler) }
    },
    onDownloadComplete: (callback: (data: any) => void) => {
      const handler = (_: any, data: any) => callback(data)
      ipcRenderer.on('models:downloadComplete', handler)
      return () => { ipcRenderer.removeListener('models:downloadComplete', handler) }
    },
    onDownloadError: (callback: (data: any) => void) => {
      const handler = (_: any, data: any) => callback(data)
      ipcRenderer.on('models:downloadError', handler)
      return () => { ipcRenderer.removeListener('models:downloadError', handler) }
    },
    onDownloadStarted: (callback: (data: any) => void) => {
      const handler = (_: any, data: any) => callback(data)
      ipcRenderer.on('models:downloadStarted', handler)
      return () => { ipcRenderer.removeListener('models:downloadStarted', handler) }
    },
    onDownloadQueued: (callback: (data: any) => void) => {
      const handler = (_: any, data: any) => callback(data)
      ipcRenderer.on('models:downloadQueued', handler)
      return () => { ipcRenderer.removeListener('models:downloadQueued', handler) }
    },
    // Image model download support
    checkImageModel: (modelName: string, quantize?: number) =>
      ipcRenderer.invoke('models:checkImageModel', modelName, quantize ?? 4),
    downloadImageModel: (modelName: string, quantize?: number) =>
      ipcRenderer.invoke('models:downloadImageModel', modelName, quantize ?? 4),
    openDownloadWindow: () => ipcRenderer.invoke('downloads:openWindow'),
  },

  // Chat management
  chat: {
    // Folders
    createFolder: (name: string, parentId?: string) =>
      ipcRenderer.invoke('chat:createFolder', name, parentId),
    getFolders: () => ipcRenderer.invoke('chat:getFolders'),
    deleteFolder: (id: string) => ipcRenderer.invoke('chat:deleteFolder', id),

    // Chats
    create: (title: string, modelId: string, folderId?: string, modelPath?: string) =>
      ipcRenderer.invoke('chat:create', title, modelId, folderId, modelPath),
    getRecent: (limit: number = 100) => ipcRenderer.invoke('chat:getRecent', limit),
    getByModel: (modelPath: string) => ipcRenderer.invoke('chat:getByModel', modelPath),
    getAll: (folderId?: string) => ipcRenderer.invoke('chat:getAll', folderId),
    get: (id: string) => ipcRenderer.invoke('chat:get', id),
    update: (id: string, updates: any) => ipcRenderer.invoke('chat:update', id, updates),
    delete: (id: string) => ipcRenderer.invoke('chat:delete', id),
    search: (query: string) => ipcRenderer.invoke('chat:search', query),

    // Messages
    getMessages: (chatId: string) => ipcRenderer.invoke('chat:getMessages', chatId),
    addMessage: (chatId: string, role: string, content: string) =>
      ipcRenderer.invoke('chat:addMessage', chatId, role, content),
    sendMessage: (chatId: string, content: string, endpoint?: { host: string; port: number }, attachments?: Array<{ dataUrl: string; name: string }>) =>
      ipcRenderer.invoke('chat:sendMessage', chatId, content, endpoint, attachments),
    onStream: (callback: (data: any) => void) => {
      const handler = (_: any, data: any) => callback(data)
      ipcRenderer.on('chat:stream', handler)
      return () => { ipcRenderer.removeListener('chat:stream', handler) }
    },
    onComplete: (callback: (data: any) => void) => {
      const handler = (_: any, data: any) => callback(data)
      ipcRenderer.on('chat:complete', handler)
      return () => { ipcRenderer.removeListener('chat:complete', handler) }
    },
    onReasoningDone: (callback: (data: any) => void) => {
      const handler = (_: any, data: any) => callback(data)
      ipcRenderer.on('chat:reasoningDone', handler)
      return () => { ipcRenderer.removeListener('chat:reasoningDone', handler) }
    },
    onTyping: (callback: (data: any) => void) => {
      const handler = (_: any, data: any) => callback(data)
      ipcRenderer.on('chat:typing', handler)
      return () => { ipcRenderer.removeListener('chat:typing', handler) }
    },
    onToolStatus: (callback: (data: any) => void) => {
      const handler = (_: any, data: any) => callback(data)
      ipcRenderer.on('chat:toolStatus', handler)
      return () => { ipcRenderer.removeListener('chat:toolStatus', handler) }
    },
    abort: (chatId: string) => ipcRenderer.invoke('chat:abort', chatId),
    isStreaming: (chatId: string) => ipcRenderer.invoke('chat:isStreaming', chatId),
    clearAllLocks: () => ipcRenderer.invoke('chat:clearAllLocks'),

    // ask_user tool: model asks user a question mid-tool-loop
    onAskUser: (callback: (data: any) => void) => {
      const handler = (_: any, data: any) => callback(data)
      ipcRenderer.on('chat:askUser', handler)
      return () => { ipcRenderer.removeListener('chat:askUser', handler) }
    },
    answerUser: (chatId: string, answer: string) =>
      ipcRenderer.send('chat:answerUser', chatId, answer),

    // Image picker for vision/multimodal input
    pickImages: () => ipcRenderer.invoke('dialog:pickImages') as Promise<Array<{ dataUrl: string; name: string }>>,
    openDirectory: () => ipcRenderer.invoke('dialog:openDirectory') as Promise<{ canceled: boolean; filePaths: string[] }>,

    // Overrides
    setOverrides: (chatId: string, overrides: any) =>
      ipcRenderer.invoke('chat:setOverrides', chatId, overrides),
    getOverrides: (chatId: string) => ipcRenderer.invoke('chat:getOverrides', chatId),
    clearOverrides: (chatId: string) => ipcRenderer.invoke('chat:clearOverrides', chatId),

    // Export/Import
    export: (chatId: string, format: 'json' | 'markdown' | 'sharegpt') =>
      ipcRenderer.invoke('chat:export', chatId, format),
    import: (modelPath?: string) =>
      ipcRenderer.invoke('chat:import', modelPath)
  },

  // vMLX Engine management
  engine: {
    checkInstallation: () => ipcRenderer.invoke('engine:check-installation'),
    detectInstallers: () => ipcRenderer.invoke('engine:detect-installers'),
    checkEngineVersion: () => ipcRenderer.invoke('engine:check-engine-version'),
    installStreaming: (method: 'uv' | 'pip' | 'bundled-update', action: 'install' | 'upgrade', installerPath?: string) =>
      ipcRenderer.invoke('engine:install-streaming', method, action, installerPath),
    cancelInstall: () => ipcRenderer.invoke('engine:cancel-install'),
    onInstallLog: (callback: (data: any) => void) => {
      const handler = (_: any, data: any) => callback(data)
      ipcRenderer.on('engine:install-log', handler)
      return () => { ipcRenderer.removeListener('engine:install-log', handler) }
    },
    onInstallComplete: (callback: (data: any) => void) => {
      const handler = (_: any, data: any) => callback(data)
      ipcRenderer.on('engine:install-complete', handler)
      return () => { ipcRenderer.removeListener('engine:install-complete', handler) }
    }
  },

  // Cache management
  cache: {
    stats: (endpoint?: { host: string; port: number }, sessionId?: string) =>
      ipcRenderer.invoke('cache:stats', endpoint, sessionId),
    entries: (endpoint?: { host: string; port: number }, sessionId?: string) =>
      ipcRenderer.invoke('cache:entries', endpoint, sessionId),
    warm: (prompts: string[], endpoint?: { host: string; port: number }, sessionId?: string) =>
      ipcRenderer.invoke('cache:warm', prompts, endpoint, sessionId),
    clear: (cacheType: string, endpoint?: { host: string; port: number }, sessionId?: string) =>
      ipcRenderer.invoke('cache:clear', cacheType, endpoint, sessionId)
  },

  // Audio: STT and TTS
  audio: {
    transcribe: (opts: { audioBase64: string; model?: string; language?: string; endpoint?: { host: string; port: number }; sessionId?: string }) =>
      ipcRenderer.invoke('audio:transcribe', opts),
    speak: (opts: { text: string; model?: string; voice?: string; speed?: number; endpoint?: { host: string; port: number }; sessionId?: string }) =>
      ipcRenderer.invoke('audio:speak', opts) as Promise<string>,
    voices: (opts: { model?: string; endpoint?: { host: string; port: number }; sessionId?: string }) =>
      ipcRenderer.invoke('audio:voices', opts) as Promise<{ voices: string[] }>
  },

  // Benchmark
  benchmark: {
    run: (sessionId: string, endpoint: { host: string; port: number }, modelPath: string, modelName?: string, options?: { flushCache?: boolean }) =>
      ipcRenderer.invoke('benchmark:run', sessionId, endpoint, modelPath, modelName, options),
    history: (modelPath?: string) =>
      ipcRenderer.invoke('benchmark:history', modelPath),
    delete: (id: string) =>
      ipcRenderer.invoke('benchmark:delete', id),
    onProgress: (callback: (data: any) => void) => {
      const handler = (_: any, data: any) => callback(data)
      ipcRenderer.on('benchmark:progress', handler)
      return () => { ipcRenderer.removeListener('benchmark:progress', handler) }
    }
  },

  // Embeddings
  embeddings: {
    embed: (texts: string[], endpoint: { host: string; port: number }, model?: string, sessionId?: string) =>
      ipcRenderer.invoke('embeddings:embed', texts, endpoint, model, sessionId)
  },

  // Performance monitoring
  performance: {
    health: (endpoint: { host: string; port: number }) =>
      ipcRenderer.invoke('performance:health', endpoint)
  },

  // Prompt templates
  templates: {
    list: () => ipcRenderer.invoke('templates:list'),
    save: (template: { id: string; name: string; content: string; category: string }) =>
      ipcRenderer.invoke('templates:save', template),
    delete: (id: string) => ipcRenderer.invoke('templates:delete', id)
  },

  // Developer tools (convert, doctor, info)
  developer: {
    info: (modelPath: string) => ipcRenderer.invoke('developer:info', modelPath),
    doctor: (modelPath: string, options?: { noInference?: boolean }) =>
      ipcRenderer.invoke('developer:doctor', modelPath, options || {}),
    convert: (args: {
      model: string; output?: string; bits: number; groupSize: number;
      mode?: string; dtype?: string; force?: boolean; skipVerify?: boolean; trustRemoteCode?: boolean;
      jangProfile?: string; jangMethod?: string
    }) => ipcRenderer.invoke('developer:convert', args),
    cancelOp: () => ipcRenderer.invoke('developer:cancelOp'),
    browseOutputDir: () => ipcRenderer.invoke('developer:browseOutputDir') as Promise<string | null>,
    onLog: (callback: (data: any) => void) => {
      const handler = (_: any, data: any) => callback(data)
      ipcRenderer.on('developer:log', handler)
      return () => { ipcRenderer.removeListener('developer:log', handler) }
    },
    onComplete: (callback: (data: any) => void) => {
      const handler = (_: any, data: any) => callback(data)
      ipcRenderer.on('developer:complete', handler)
      return () => { ipcRenderer.removeListener('developer:complete', handler) }
    }
  },

  // App-level events
  app: {
    getVersion: () => ipcRenderer.invoke('app:getVersion'),
    onUpdateAvailable: (callback: (data: any) => void) => {
      const handler = (_: any, data: any) => callback(data)
      ipcRenderer.on('app:updateAvailable', handler)
      return () => { ipcRenderer.removeListener('app:updateAvailable', handler) }
    }
  },

  // App-level settings (API keys, preferences)
  settings: {
    get: (key: string) => ipcRenderer.invoke('settings:get', key),
    set: (key: string, value: string) => ipcRenderer.invoke('settings:set', key, value),
    delete: (key: string) => ipcRenderer.invoke('settings:delete', key)
  },

  // Per-model settings
  modelSettings: {
    get: (modelPath: string) => ipcRenderer.invoke('model-settings:get', modelPath),
    getAll: () => ipcRenderer.invoke('model-settings:getAll'),
    save: (modelPath: string, settings: any) => ipcRenderer.invoke('model-settings:save', modelPath, settings),
    delete: (modelPath: string) => ipcRenderer.invoke('model-settings:delete', modelPath),
  },

  // Image generation
  image: {
    createSession: (modelName: string, sessionType?: 'generate' | 'edit') => ipcRenderer.invoke('image:createSession', modelName, sessionType),
    getSessions: () => ipcRenderer.invoke('image:getSessions'),
    getSession: (id: string) => ipcRenderer.invoke('image:getSession', id),
    deleteSession: (id: string) => ipcRenderer.invoke('image:deleteSession', id),
    getGenerations: (sessionId: string) => ipcRenderer.invoke('image:getGenerations', sessionId),
    generate: (params: any) => ipcRenderer.invoke('image:generate', params),
    edit: (params: any) => ipcRenderer.invoke('image:edit', params),
    startServer: (modelName: string, quantize?: number, imageMode?: 'generate' | 'edit', serverSettings?: any) => ipcRenderer.invoke('image:startServer', modelName, quantize, imageMode, serverSettings),
    stopServer: () => ipcRenderer.invoke('image:stopServer'),
    cancelGeneration: () => ipcRenderer.invoke('image:cancelGeneration'),
    getRunningServer: () => ipcRenderer.invoke('image:getRunningServer'),
    getRunningServers: () => ipcRenderer.invoke('image:getRunningServers'),
    readFile: (imagePath: string) => ipcRenderer.invoke('image:readFile', imagePath),
    saveFile: (imagePath: string) => ipcRenderer.invoke('image:saveFile', imagePath),
    getModelPaths: () => ipcRenderer.invoke('image:getModelPaths'),
  },

  // Session management
  sessions: {
    list: () => ipcRenderer.invoke('sessions:list'),
    get: (id: string) => ipcRenderer.invoke('sessions:get', id),
    create: (modelPath: string, config: any) => ipcRenderer.invoke('sessions:create', modelPath, config),
    createRemote: (params: { remoteUrl: string; remoteApiKey?: string; remoteModel: string; remoteOrganization?: string }) =>
      ipcRenderer.invoke('sessions:createRemote', params),
    start: (sessionId: string) => ipcRenderer.invoke('sessions:start', sessionId),
    stop: (sessionId: string) => ipcRenderer.invoke('sessions:stop', sessionId),
    delete: (sessionId: string) => ipcRenderer.invoke('sessions:delete', sessionId),
    detect: () => ipcRenderer.invoke('sessions:detect'),
    update: (sessionId: string, config: any) => ipcRenderer.invoke('sessions:update', sessionId, config),
    getLogs: (sessionId: string) => ipcRenderer.invoke('sessions:getLogs', sessionId),
    clearLogs: (sessionId: string) => ipcRenderer.invoke('sessions:clearLogs', sessionId),

    // Events — each returns an unsubscribe function for targeted cleanup
    onStarting: (callback: (data: any) => void) => {
      const handler = (_: any, data: any) => callback(data)
      ipcRenderer.on('session:starting', handler)
      return () => { ipcRenderer.removeListener('session:starting', handler) }
    },
    onReady: (callback: (data: any) => void) => {
      const handler = (_: any, data: any) => callback(data)
      ipcRenderer.on('session:ready', handler)
      return () => { ipcRenderer.removeListener('session:ready', handler) }
    },
    onStopped: (callback: (data: any) => void) => {
      const handler = (_: any, data: any) => callback(data)
      ipcRenderer.on('session:stopped', handler)
      return () => { ipcRenderer.removeListener('session:stopped', handler) }
    },
    onError: (callback: (data: any) => void) => {
      const handler = (_: any, data: any) => callback(data)
      ipcRenderer.on('session:error', handler)
      return () => { ipcRenderer.removeListener('session:error', handler) }
    },
    onHealth: (callback: (data: any) => void) => {
      const handler = (_: any, data: any) => callback(data)
      ipcRenderer.on('session:health', handler)
      return () => { ipcRenderer.removeListener('session:health', handler) }
    },
    onLog: (callback: (data: any) => void) => {
      const handler = (_: any, data: any) => callback(data)
      ipcRenderer.on('session:log', handler)
      return () => { ipcRenderer.removeListener('session:log', handler) }
    },
    onCreated: (callback: (data: any) => void) => {
      const handler = (_: any, data: any) => callback(data)
      ipcRenderer.on('session:created', handler)
      return () => { ipcRenderer.removeListener('session:created', handler) }
    },
    onDeleted: (callback: (data: any) => void) => {
      const handler = (_: any, data: any) => callback(data)
      ipcRenderer.on('session:deleted', handler)
      return () => { ipcRenderer.removeListener('session:deleted', handler) }
    },
  }
}

// Use `contextBridge` APIs to expose Electron APIs to renderer only if context isolation is enabled
if (process.contextIsolated) {
  try {
    contextBridge.exposeInMainWorld('electron', electronAPI)
    contextBridge.exposeInMainWorld('api', api)
  } catch (error) {
    console.error(error)
  }
} else {
  // @ts-ignore (define in dts)
  window.electron = electronAPI
  // @ts-ignore (define in dts)
  window.api = api
}
