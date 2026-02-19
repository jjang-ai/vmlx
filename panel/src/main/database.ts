import Database from 'better-sqlite3'
import { app } from 'electron'
import { join } from 'path'
import { existsSync, unlinkSync, renameSync } from 'fs'

export interface Chat {
  id: string
  title: string
  folderId?: string
  createdAt: number
  updatedAt: number
  modelId: string
  modelPath?: string
}

export interface Session {
  id: string
  modelPath: string
  modelName?: string
  host: string
  port: number
  pid?: number
  status: 'running' | 'stopped' | 'error' | 'loading'
  config: string // JSON blob of ServerConfig
  createdAt: number
  updatedAt: number
  lastStartedAt?: number
  lastStoppedAt?: number
  // Remote endpoint support
  type: 'local' | 'remote'
  remoteUrl?: string
  remoteApiKey?: string
  remoteModel?: string
  remoteOrganization?: string
}

export interface MessageMetrics {
  tokenCount: number
  promptTokens?: number
  cachedTokens?: number
  tokensPerSecond: string
  ppSpeed?: string
  ttft: string
  totalTime?: string
}

export interface Message {
  id: string
  chatId: string
  role: 'system' | 'user' | 'assistant'
  content: string
  timestamp: number
  tokens?: number
  metricsJson?: string  // JSON-serialized MessageMetrics
  toolCallsJson?: string  // JSON-serialized tool call statuses for inline display
  reasoningContent?: string  // Reasoning/thinking content (from <think> tags or similar)
}

export interface Folder {
  id: string
  name: string
  parentId?: string
  color?: string
  icon?: string
  createdAt: number
}

export interface BenchmarkResult {
  id: string
  sessionId: string
  modelPath: string
  modelName?: string
  resultsJson: string // JSON array of per-prompt results
  createdAt: number
}

export interface ChatOverrides {
  chatId: string
  temperature?: number
  topP?: number
  topK?: number
  minP?: number
  maxTokens?: number
  repeatPenalty?: number
  systemPrompt?: string
  stopSequences?: string
  wireApi?: string
  maxToolIterations?: number
  builtinToolsEnabled?: boolean
  workingDirectory?: string
  enableThinking?: boolean  // tri-state: undefined=Auto, true=On, false=Off
  reasoningEffort?: string  // 'low' | 'medium' | 'high' | undefined=Auto
  hideToolStatus?: boolean
  webSearchEnabled?: boolean
  braveSearchEnabled?: boolean
  fetchUrlEnabled?: boolean
  fileToolsEnabled?: boolean
  searchToolsEnabled?: boolean
  shellEnabled?: boolean
  toolResultMaxChars?: number
  gitEnabled?: boolean
  utilityToolsEnabled?: boolean
}

class DatabaseManager {
  private db: Database.Database
  /** Set when DB was recovered from corruption — main process shows dialog */
  recoveryBackupPath: string | null = null

  constructor() {
    const dbPath = join(app.getPath('userData'), 'chats.db')
    try {
      this.db = new Database(dbPath)
      this.initialize()
    } catch (err) {
      // DB is corrupt — back up the bad file and start fresh
      console.error('[DB] Database corrupt, recreating:', err)
      const backupPath = `${dbPath}.corrupt.${Date.now()}`
      try {
        if (existsSync(dbPath)) renameSync(dbPath, backupPath)
        // Also clean up WAL/SHM files
        if (existsSync(`${dbPath}-wal`)) unlinkSync(`${dbPath}-wal`)
        if (existsSync(`${dbPath}-shm`)) unlinkSync(`${dbPath}-shm`)
      } catch (_) {}
      this.db = new Database(dbPath)
      this.initialize()
      this.recoveryBackupPath = backupPath
      console.log(`[DB] Fresh database created. Corrupt file saved to: ${backupPath}`)
    }
  }

  private initialize(): void {
    // Enable WAL mode for better concurrent read performance
    this.db.pragma('journal_mode = WAL')
    // Enable foreign keys
    this.db.pragma('foreign_keys = ON')

    // Create tables
    this.db.exec(`
      CREATE TABLE IF NOT EXISTS folders (
        id TEXT PRIMARY KEY,
        name TEXT NOT NULL,
        parent_id TEXT,
        color TEXT,
        icon TEXT,
        created_at INTEGER NOT NULL,
        FOREIGN KEY (parent_id) REFERENCES folders(id) ON DELETE CASCADE
      );

      CREATE TABLE IF NOT EXISTS chats (
        id TEXT PRIMARY KEY,
        title TEXT NOT NULL,
        folder_id TEXT,
        created_at INTEGER NOT NULL,
        updated_at INTEGER NOT NULL,
        model_id TEXT NOT NULL,
        FOREIGN KEY (folder_id) REFERENCES folders(id) ON DELETE SET NULL
      );

      CREATE TABLE IF NOT EXISTS messages (
        id TEXT PRIMARY KEY,
        chat_id TEXT NOT NULL,
        role TEXT NOT NULL CHECK(role IN ('system', 'user', 'assistant')),
        content TEXT NOT NULL,
        timestamp INTEGER NOT NULL,
        tokens INTEGER,
        FOREIGN KEY (chat_id) REFERENCES chats(id) ON DELETE CASCADE
      );

      CREATE TABLE IF NOT EXISTS chat_overrides (
        chat_id TEXT PRIMARY KEY,
        temperature REAL,
        top_p REAL,
        top_k INTEGER,
        max_tokens INTEGER,
        repeat_penalty REAL,
        system_prompt TEXT,
        stop_sequences TEXT,
        FOREIGN KEY (chat_id) REFERENCES chats(id) ON DELETE CASCADE
      );

      CREATE INDEX IF NOT EXISTS idx_messages_chat ON messages(chat_id);
      CREATE INDEX IF NOT EXISTS idx_chats_folder ON chats(folder_id);
      CREATE INDEX IF NOT EXISTS idx_folders_parent ON folders(parent_id);
      CREATE INDEX IF NOT EXISTS idx_messages_timestamp ON messages(timestamp);

      CREATE TABLE IF NOT EXISTS settings (
        key TEXT PRIMARY KEY,
        value TEXT NOT NULL
      );

      CREATE TABLE IF NOT EXISTS sessions (
        id TEXT PRIMARY KEY,
        model_path TEXT NOT NULL UNIQUE,
        model_name TEXT,
        host TEXT NOT NULL DEFAULT '127.0.0.1',
        port INTEGER NOT NULL,
        pid INTEGER,
        status TEXT NOT NULL DEFAULT 'stopped'
          CHECK(status IN ('running','stopped','error','loading')),
        config TEXT NOT NULL,
        created_at INTEGER NOT NULL,
        updated_at INTEGER NOT NULL,
        last_started_at INTEGER,
        last_stopped_at INTEGER
      );
      CREATE INDEX IF NOT EXISTS idx_sessions_status ON sessions(status);

      CREATE TABLE IF NOT EXISTS benchmarks (
        id TEXT PRIMARY KEY,
        session_id TEXT NOT NULL,
        model_path TEXT NOT NULL,
        model_name TEXT,
        results_json TEXT NOT NULL,
        created_at INTEGER NOT NULL
      );
      CREATE INDEX IF NOT EXISTS idx_benchmarks_session ON benchmarks(session_id);
      CREATE INDEX IF NOT EXISTS idx_benchmarks_model ON benchmarks(model_path);
    `)

    // Safe migration: add model_path column to chats if missing
    const chatColumns = this.db.pragma('table_info(chats)') as { name: string }[]
    if (!chatColumns.find(c => c.name === 'model_path')) {
      this.db.exec('ALTER TABLE chats ADD COLUMN model_path TEXT')
      this.db.exec('CREATE INDEX IF NOT EXISTS idx_chats_model_path ON chats(model_path)')
    }

    // Safe migration: add metrics_json column to messages if missing
    const msgColumns = this.db.pragma('table_info(messages)') as { name: string }[]
    if (!msgColumns.find(c => c.name === 'metrics_json')) {
      this.db.exec('ALTER TABLE messages ADD COLUMN metrics_json TEXT')
    }
    if (!msgColumns.find(c => c.name === 'tool_calls_json')) {
      this.db.exec('ALTER TABLE messages ADD COLUMN tool_calls_json TEXT')
    }
    if (!msgColumns.find(c => c.name === 'reasoning_content')) {
      this.db.exec('ALTER TABLE messages ADD COLUMN reasoning_content TEXT')
    }

    // Safe migration: add new chat_overrides columns if missing
    // (top_k and repeat_penalty already exist in CREATE TABLE)
    const overrideColumns = this.db.pragma('table_info(chat_overrides)') as { name: string }[]
    if (!overrideColumns.find(c => c.name === 'min_p')) {
      this.db.exec('ALTER TABLE chat_overrides ADD COLUMN min_p REAL')
    }
    if (!overrideColumns.find(c => c.name === 'wire_api')) {
      this.db.exec('ALTER TABLE chat_overrides ADD COLUMN wire_api TEXT')
    }
    if (!overrideColumns.find(c => c.name === 'max_tool_iterations')) {
      this.db.exec('ALTER TABLE chat_overrides ADD COLUMN max_tool_iterations INTEGER')
    }
    if (!overrideColumns.find(c => c.name === 'builtin_tools_enabled')) {
      this.db.exec('ALTER TABLE chat_overrides ADD COLUMN builtin_tools_enabled INTEGER DEFAULT 0')
    }
    if (!overrideColumns.find(c => c.name === 'working_directory')) {
      this.db.exec('ALTER TABLE chat_overrides ADD COLUMN working_directory TEXT')
    }
    if (!overrideColumns.find(c => c.name === 'enable_thinking')) {
      this.db.exec('ALTER TABLE chat_overrides ADD COLUMN enable_thinking INTEGER DEFAULT NULL')
    }
    if (!overrideColumns.find(c => c.name === 'hide_tool_status')) {
      this.db.exec('ALTER TABLE chat_overrides ADD COLUMN hide_tool_status INTEGER DEFAULT 0')
    }
    if (!overrideColumns.find(c => c.name === 'web_search_enabled')) {
      this.db.exec('ALTER TABLE chat_overrides ADD COLUMN web_search_enabled INTEGER DEFAULT 1')
    }
    if (!overrideColumns.find(c => c.name === 'fetch_url_enabled')) {
      this.db.exec('ALTER TABLE chat_overrides ADD COLUMN fetch_url_enabled INTEGER DEFAULT 1')
    }
    if (!overrideColumns.find(c => c.name === 'file_tools_enabled')) {
      this.db.exec('ALTER TABLE chat_overrides ADD COLUMN file_tools_enabled INTEGER DEFAULT 1')
    }
    if (!overrideColumns.find(c => c.name === 'search_tools_enabled')) {
      this.db.exec('ALTER TABLE chat_overrides ADD COLUMN search_tools_enabled INTEGER DEFAULT 1')
    }
    if (!overrideColumns.find(c => c.name === 'shell_enabled')) {
      this.db.exec('ALTER TABLE chat_overrides ADD COLUMN shell_enabled INTEGER DEFAULT 1')
    }
    if (!overrideColumns.find(c => c.name === 'reasoning_effort')) {
      this.db.exec('ALTER TABLE chat_overrides ADD COLUMN reasoning_effort TEXT')
    }
    if (!overrideColumns.find(c => c.name === 'brave_search_enabled')) {
      this.db.exec('ALTER TABLE chat_overrides ADD COLUMN brave_search_enabled INTEGER DEFAULT 0')
    }
    if (!overrideColumns.find(c => c.name === 'tool_result_max_chars')) {
      this.db.exec('ALTER TABLE chat_overrides ADD COLUMN tool_result_max_chars INTEGER')
    }
    if (!overrideColumns.find(c => c.name === 'git_enabled')) {
      this.db.exec('ALTER TABLE chat_overrides ADD COLUMN git_enabled INTEGER')
    }
    if (!overrideColumns.find(c => c.name === 'utility_tools_enabled')) {
      this.db.exec('ALTER TABLE chat_overrides ADD COLUMN utility_tools_enabled INTEGER')
    }

    // Fix: older migration used DEFAULT 1 for enable_thinking, corrupting existing
    // chat overrides to thinking=ON instead of Auto (NULL). Check the column default
    // in the schema and reset if it's still 1.
    const tableInfo = this.db.pragma('table_info(chat_overrides)') as { name: string; dflt_value: string | null }[]
    const etCol = tableInfo.find(c => c.name === 'enable_thinking')
    if (etCol && etCol.dflt_value === '1') {
      // Reset all rows that have the DEFAULT value (1) to NULL (Auto)
      this.db.exec('UPDATE chat_overrides SET enable_thinking = NULL WHERE enable_thinking = 1')
      // Recreate the column with correct default by rebuilding the table
      // SQLite doesn't support ALTER COLUMN, but we can work around it:
      // The DEFAULT in schema only affects future INSERTs without explicit value,
      // and our INSERT OR REPLACE always specifies the value explicitly.
      // So the stale DEFAULT 1 won't cause further issues — the UPDATE above
      // fixes existing rows. Log for visibility.
      console.log('[DB] Fixed enable_thinking DEFAULT 1 → reset affected rows to NULL (Auto)')
    }

    // Safe migration: add remote session columns to sessions table
    const sessionColumns = this.db.pragma('table_info(sessions)') as { name: string }[]
    if (!sessionColumns.find(c => c.name === 'type')) {
      this.db.exec("ALTER TABLE sessions ADD COLUMN type TEXT NOT NULL DEFAULT 'local'")
    }
    if (!sessionColumns.find(c => c.name === 'remote_url')) {
      this.db.exec('ALTER TABLE sessions ADD COLUMN remote_url TEXT')
    }
    if (!sessionColumns.find(c => c.name === 'remote_api_key')) {
      this.db.exec('ALTER TABLE sessions ADD COLUMN remote_api_key TEXT')
    }
    if (!sessionColumns.find(c => c.name === 'remote_model')) {
      this.db.exec('ALTER TABLE sessions ADD COLUMN remote_model TEXT')
    }
    if (!sessionColumns.find(c => c.name === 'remote_organization')) {
      this.db.exec('ALTER TABLE sessions ADD COLUMN remote_organization TEXT')
    }

    // Prompt templates table
    this.db.exec(`
      CREATE TABLE IF NOT EXISTS prompt_templates (
        id TEXT PRIMARY KEY,
        name TEXT NOT NULL,
        content TEXT NOT NULL,
        category TEXT NOT NULL DEFAULT 'custom',
        is_builtin INTEGER NOT NULL DEFAULT 0,
        created_at INTEGER NOT NULL
      )
    `)

    // Seed built-in templates (only if table is empty)
    const templateCount = (this.db.prepare('SELECT COUNT(*) as cnt FROM prompt_templates WHERE is_builtin = 1').get() as any).cnt
    if (templateCount === 0) {
      const builtins = [
        { id: 'builtin-coder', name: 'Coding Assistant', category: 'development', content: 'You are an expert software engineer. Write clean, efficient, well-documented code. Explain your reasoning. When debugging, think step-by-step. Prefer simple solutions.' },
        { id: 'builtin-writer', name: 'Creative Writer', category: 'creative', content: 'You are a skilled creative writer. Write vivid, engaging prose. Vary sentence structure and length. Show, don\'t tell. Use sensory details and strong verbs.' },
        { id: 'builtin-analyst', name: 'Data Analyst', category: 'analysis', content: 'You are a data analyst. Provide clear, data-driven insights. Use precise numbers. Structure your analysis with sections. Highlight key findings and actionable recommendations.' },
        { id: 'builtin-tutor', name: 'Patient Tutor', category: 'education', content: 'You are a patient, encouraging tutor. Explain concepts from first principles. Use analogies and examples. Check understanding. Adapt your explanations to the student\'s level.' },
        { id: 'builtin-concise', name: 'Concise Responder', category: 'general', content: 'Be extremely concise. Answer in as few words as possible while remaining accurate and helpful. No unnecessary elaboration. Bullet points preferred.' },
        { id: 'builtin-socratic', name: 'Socratic Guide', category: 'education', content: 'Guide through questions rather than direct answers. Ask thought-provoking questions that lead to understanding. Help the user discover the answer themselves.' },
        { id: 'builtin-reviewer', name: 'Code Reviewer', category: 'development', content: 'You are a thorough code reviewer. Check for bugs, security issues, performance problems, and style. Be specific about line numbers. Suggest concrete improvements. Rate severity.' },
      ]
      const ins = this.db.prepare('INSERT INTO prompt_templates (id, name, content, category, is_builtin, created_at) VALUES (?, ?, ?, ?, 1, ?)')
      const now = Date.now()
      for (const t of builtins) {
        ins.run(t.id, t.name, t.content, t.category, now)
      }
    }
  }

  // Folders
  createFolder(folder: Folder): void {
    const stmt = this.db.prepare(`
      INSERT INTO folders (id, name, parent_id, color, icon, created_at)
      VALUES (?, ?, ?, ?, ?, ?)
    `)
    stmt.run(folder.id, folder.name, folder.parentId, folder.color, folder.icon, folder.createdAt)
  }

  getFolders(): Folder[] {
    const stmt = this.db.prepare('SELECT * FROM folders ORDER BY name')
    return stmt.all().map((row: any) => ({
      id: row.id,
      name: row.name,
      parentId: row.parent_id,
      color: row.color,
      icon: row.icon,
      createdAt: row.created_at
    }))
  }

  deleteFolder(id: string): void {
    const stmt = this.db.prepare('DELETE FROM folders WHERE id = ?')
    stmt.run(id)
  }

  // Chats
  createChat(chat: Chat): void {
    const stmt = this.db.prepare(`
      INSERT INTO chats (id, title, folder_id, created_at, updated_at, model_id, model_path)
      VALUES (?, ?, ?, ?, ?, ?, ?)
    `)
    stmt.run(chat.id, chat.title, chat.folderId ?? null, chat.createdAt, chat.updatedAt, chat.modelId || 'default', chat.modelPath ?? null)
  }

  getChats(folderId?: string): Chat[] {
    const query = folderId
      ? 'SELECT * FROM chats WHERE folder_id = ? ORDER BY updated_at DESC'
      : 'SELECT * FROM chats ORDER BY updated_at DESC'

    const stmt = this.db.prepare(query)
    const rows = folderId ? stmt.all(folderId) : stmt.all()

    return rows.map((row: any) => ({
      id: row.id,
      title: row.title,
      folderId: row.folder_id,
      createdAt: row.created_at,
      updatedAt: row.updated_at,
      modelId: row.model_id,
      modelPath: row.model_path
    }))
  }

  getChat(id: string): Chat | undefined {
    const stmt = this.db.prepare('SELECT * FROM chats WHERE id = ?')
    const row = stmt.get(id) as any
    if (!row) return undefined

    return {
      id: row.id,
      title: row.title,
      folderId: row.folder_id,
      createdAt: row.created_at,
      updatedAt: row.updated_at,
      modelId: row.model_id,
      modelPath: row.model_path
    }
  }

  getChatsByModelPath(modelPath: string): Chat[] {
    const stmt = this.db.prepare('SELECT * FROM chats WHERE model_path = ? ORDER BY updated_at DESC')
    return stmt.all(modelPath).map((row: any) => ({
      id: row.id,
      title: row.title,
      folderId: row.folder_id,
      createdAt: row.created_at,
      updatedAt: row.updated_at,
      modelId: row.model_id,
      modelPath: row.model_path
    }))
  }

  updateChat(id: string, updates: Partial<Chat>): void {
    const fields: string[] = []
    const values: any[] = []

    if (updates.title !== undefined) {
      fields.push('title = ?')
      values.push(updates.title)
    }
    if (updates.folderId !== undefined) {
      fields.push('folder_id = ?')
      values.push(updates.folderId)
    }
    if (updates.modelId !== undefined) {
      fields.push('model_id = ?')
      values.push(updates.modelId)
    }
    if (updates.modelPath !== undefined) {
      fields.push('model_path = ?')
      values.push(updates.modelPath)
    }
    if (updates.updatedAt !== undefined) {
      fields.push('updated_at = ?')
      values.push(updates.updatedAt)
    }

    if (fields.length === 0) return

    values.push(id)
    const stmt = this.db.prepare(`UPDATE chats SET ${fields.join(', ')} WHERE id = ?`)
    stmt.run(...values)
  }

  deleteChat(id: string): void {
    const stmt = this.db.prepare('DELETE FROM chats WHERE id = ?')
    stmt.run(id)
  }

  // Messages
  addMessage(message: Message): void {
    const stmt = this.db.prepare(`
      INSERT OR REPLACE INTO messages (id, chat_id, role, content, timestamp, tokens, metrics_json, tool_calls_json, reasoning_content)
      VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
    `)
    stmt.run(message.id, message.chatId, message.role, message.content, message.timestamp, message.tokens, message.metricsJson, message.toolCallsJson, message.reasoningContent)

    // Update chat's updatedAt
    this.updateChat(message.chatId, { updatedAt: message.timestamp })
  }

  /** Update an existing message's content in-place (for incremental persistence during streaming) */
  updateMessageContent(messageId: string, content: string, reasoningContent?: string): void {
    const stmt = this.db.prepare(`
      UPDATE messages SET content = ?, reasoning_content = ? WHERE id = ?
    `)
    stmt.run(content, reasoningContent || null, messageId)
  }

  /** Delete a message by ID (used to clean up empty pre-inserted placeholders on error) */
  deleteMessage(messageId: string): void {
    this.db.prepare('DELETE FROM messages WHERE id = ?').run(messageId)
  }

  getMessages(chatId: string): Message[] {
    const stmt = this.db.prepare('SELECT * FROM messages WHERE chat_id = ? ORDER BY timestamp ASC')
    return stmt.all(chatId).map((row: any) => ({
      id: row.id,
      chatId: row.chat_id,
      role: row.role,
      content: row.content,
      timestamp: row.timestamp,
      tokens: row.tokens,
      metricsJson: row.metrics_json,
      toolCallsJson: row.tool_calls_json,
      reasoningContent: row.reasoning_content
    }))
  }

  // Chat Overrides
  setChatOverrides(overrides: ChatOverrides): void {
    const stmt = this.db.prepare(`
      INSERT OR REPLACE INTO chat_overrides
      (chat_id, temperature, top_p, top_k, min_p, max_tokens, repeat_penalty,
       system_prompt, stop_sequences, wire_api, max_tool_iterations,
       builtin_tools_enabled, working_directory, enable_thinking, reasoning_effort,
       hide_tool_status,
       web_search_enabled, brave_search_enabled, fetch_url_enabled, file_tools_enabled,
       search_tools_enabled, shell_enabled, tool_result_max_chars,
       git_enabled, utility_tools_enabled)
      VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    `)
    // enable_thinking tri-state: undefined/null → NULL (Auto), true → 1, false → 0
    const enableThinkingVal = overrides.enableThinking === true ? 1
      : overrides.enableThinking === false ? 0
      : null  // Auto
    stmt.run(
      overrides.chatId,
      overrides.temperature,
      overrides.topP,
      overrides.topK,
      overrides.minP,
      overrides.maxTokens,
      overrides.repeatPenalty,
      overrides.systemPrompt,
      overrides.stopSequences,
      overrides.wireApi,
      overrides.maxToolIterations,
      overrides.builtinToolsEnabled ? 1 : 0,
      overrides.workingDirectory,
      enableThinkingVal,
      overrides.reasoningEffort || null,
      overrides.hideToolStatus ? 1 : 0,
      overrides.webSearchEnabled === false ? 0 : 1,
      overrides.braveSearchEnabled ? 1 : 0,
      overrides.fetchUrlEnabled === false ? 0 : 1,
      overrides.fileToolsEnabled === false ? 0 : 1,
      overrides.searchToolsEnabled === false ? 0 : 1,
      overrides.shellEnabled === false ? 0 : 1,
      overrides.toolResultMaxChars || null,
      overrides.gitEnabled === false ? 0 : 1,
      overrides.utilityToolsEnabled === false ? 0 : 1
    )
  }

  getChatOverrides(chatId: string): ChatOverrides | undefined {
    const stmt = this.db.prepare('SELECT * FROM chat_overrides WHERE chat_id = ?')
    const row = stmt.get(chatId) as any
    if (!row) return undefined

    // enable_thinking tri-state: NULL → undefined (Auto), 0 → false (Off), 1 → true (On)
    const enableThinking = row.enable_thinking === null || row.enable_thinking === undefined
      ? undefined  // Auto
      : row.enable_thinking !== 0  // 0 → false (Off), 1 → true (On)

    return {
      chatId: row.chat_id,
      temperature: row.temperature,
      topP: row.top_p,
      topK: row.top_k,
      minP: row.min_p,
      maxTokens: row.max_tokens,
      repeatPenalty: row.repeat_penalty,
      systemPrompt: row.system_prompt,
      stopSequences: row.stop_sequences,
      wireApi: row.wire_api,
      maxToolIterations: row.max_tool_iterations,
      builtinToolsEnabled: row.builtin_tools_enabled === 1,
      workingDirectory: row.working_directory,
      enableThinking,
      reasoningEffort: row.reasoning_effort || undefined,
      hideToolStatus: row.hide_tool_status === 1,
      webSearchEnabled: row.web_search_enabled !== 0,
      braveSearchEnabled: row.brave_search_enabled === 1,
      fetchUrlEnabled: row.fetch_url_enabled !== 0,
      fileToolsEnabled: row.file_tools_enabled !== 0,
      searchToolsEnabled: row.search_tools_enabled !== 0,
      shellEnabled: row.shell_enabled !== 0,
      toolResultMaxChars: row.tool_result_max_chars || undefined,
      gitEnabled: row.git_enabled !== 0,
      utilityToolsEnabled: row.utility_tools_enabled !== 0
    }
  }

  clearChatOverrides(chatId: string): void {
    const stmt = this.db.prepare('DELETE FROM chat_overrides WHERE chat_id = ?')
    stmt.run(chatId)
  }

  // Search
  searchChats(query: string): Chat[] {
    const stmt = this.db.prepare(`
      SELECT DISTINCT c.* FROM chats c
      LEFT JOIN messages m ON c.id = m.chat_id
      WHERE c.title LIKE ? OR m.content LIKE ?
      ORDER BY c.updated_at DESC
      LIMIT 50
    `)
    const searchTerm = `%${query}%`
    return stmt.all(searchTerm, searchTerm).map((row: any) => ({
      id: row.id,
      title: row.title,
      folderId: row.folder_id,
      createdAt: row.created_at,
      updatedAt: row.updated_at,
      modelId: row.model_id,
      modelPath: row.model_path
    }))
  }

  // Sessions
  createSession(session: Session): void {
    const stmt = this.db.prepare(`
      INSERT INTO sessions (id, model_path, model_name, host, port, pid, status, config, created_at, updated_at, last_started_at, last_stopped_at, type, remote_url, remote_api_key, remote_model, remote_organization)
      VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    `)
    stmt.run(
      session.id, session.modelPath, session.modelName, session.host, session.port,
      session.pid, session.status, session.config, session.createdAt, session.updatedAt,
      session.lastStartedAt, session.lastStoppedAt,
      session.type || 'local', session.remoteUrl, session.remoteApiKey, session.remoteModel, session.remoteOrganization
    )
  }

  getSession(id: string): Session | undefined {
    const stmt = this.db.prepare('SELECT * FROM sessions WHERE id = ?')
    const row = stmt.get(id) as any
    if (!row) return undefined
    return this.mapSessionRow(row)
  }

  getSessions(): Session[] {
    const stmt = this.db.prepare('SELECT * FROM sessions ORDER BY updated_at DESC')
    return stmt.all().map((row: any) => this.mapSessionRow(row))
  }

  getSessionByModelPath(modelPath: string): Session | undefined {
    // Normalize trailing slashes: try both with and without to handle legacy data
    const normalized = modelPath.replace(/\/+$/, '')
    const stmt = this.db.prepare('SELECT * FROM sessions WHERE model_path = ? OR model_path = ?')
    const row = stmt.get(normalized, normalized + '/') as any
    if (!row) return undefined
    return this.mapSessionRow(row)
  }

  updateSession(id: string, updates: Partial<Session>): void {
    const fields: string[] = []
    const values: any[] = []

    if (updates.modelPath !== undefined) { fields.push('model_path = ?'); values.push(updates.modelPath) }
    if (updates.modelName !== undefined) { fields.push('model_name = ?'); values.push(updates.modelName) }
    if (updates.host !== undefined) { fields.push('host = ?'); values.push(updates.host) }
    if (updates.port !== undefined) { fields.push('port = ?'); values.push(updates.port) }
    if ('pid' in updates) { fields.push('pid = ?'); values.push(updates.pid ?? null) }
    if (updates.status !== undefined) { fields.push('status = ?'); values.push(updates.status) }
    if (updates.config !== undefined) { fields.push('config = ?'); values.push(updates.config) }
    if ('lastStartedAt' in updates) { fields.push('last_started_at = ?'); values.push(updates.lastStartedAt ?? null) }
    if ('lastStoppedAt' in updates) { fields.push('last_stopped_at = ?'); values.push(updates.lastStoppedAt ?? null) }
    if (updates.type !== undefined) { fields.push('type = ?'); values.push(updates.type) }
    if ('remoteUrl' in updates) { fields.push('remote_url = ?'); values.push(updates.remoteUrl ?? null) }
    if ('remoteApiKey' in updates) { fields.push('remote_api_key = ?'); values.push(updates.remoteApiKey ?? null) }
    if ('remoteModel' in updates) { fields.push('remote_model = ?'); values.push(updates.remoteModel ?? null) }
    if ('remoteOrganization' in updates) { fields.push('remote_organization = ?'); values.push(updates.remoteOrganization ?? null) }

    if (fields.length === 0) return

    // Always update updated_at
    fields.push('updated_at = ?')
    values.push(Date.now())

    values.push(id)
    const stmt = this.db.prepare(`UPDATE sessions SET ${fields.join(', ')} WHERE id = ?`)
    stmt.run(...values)
  }

  deleteSession(id: string): void {
    const stmt = this.db.prepare('DELETE FROM sessions WHERE id = ?')
    stmt.run(id)
  }

  private mapSessionRow(row: any): Session {
    return {
      id: row.id,
      modelPath: row.model_path,
      modelName: row.model_name,
      host: row.host,
      port: row.port,
      pid: row.pid,
      status: row.status,
      config: row.config,
      createdAt: row.created_at,
      updatedAt: row.updated_at,
      lastStartedAt: row.last_started_at,
      lastStoppedAt: row.last_stopped_at,
      type: row.type || 'local',
      remoteUrl: row.remote_url,
      remoteApiKey: row.remote_api_key,
      remoteModel: row.remote_model,
      remoteOrganization: row.remote_organization
    }
  }

  // Settings (key-value store)
  getSetting(key: string): string | undefined {
    const stmt = this.db.prepare('SELECT value FROM settings WHERE key = ?')
    const row = stmt.get(key) as { value: string } | undefined
    return row?.value
  }

  setSetting(key: string, value: string): void {
    const stmt = this.db.prepare('INSERT OR REPLACE INTO settings (key, value) VALUES (?, ?)')
    stmt.run(key, value)
  }

  deleteSetting(key: string): void {
    const stmt = this.db.prepare('DELETE FROM settings WHERE key = ?')
    stmt.run(key)
  }

  // Benchmarks
  saveBenchmark(b: BenchmarkResult): void {
    const stmt = this.db.prepare(
      'INSERT INTO benchmarks (id, session_id, model_path, model_name, results_json, created_at) VALUES (?, ?, ?, ?, ?, ?)'
    )
    stmt.run(b.id, b.sessionId, b.modelPath, b.modelName || null, b.resultsJson, b.createdAt)
  }

  getBenchmarks(modelPath?: string): BenchmarkResult[] {
    const query = modelPath
      ? 'SELECT * FROM benchmarks WHERE model_path = ? ORDER BY created_at DESC LIMIT 50'
      : 'SELECT * FROM benchmarks ORDER BY created_at DESC LIMIT 50'
    const stmt = this.db.prepare(query)
    const rows = (modelPath ? stmt.all(modelPath) : stmt.all()) as any[]
    return rows.map(r => ({
      id: r.id,
      sessionId: r.session_id,
      modelPath: r.model_path,
      modelName: r.model_name,
      resultsJson: r.results_json,
      createdAt: r.created_at
    }))
  }

  deleteBenchmark(id: string): void {
    const stmt = this.db.prepare('DELETE FROM benchmarks WHERE id = ?')
    stmt.run(id)
  }

  // Prompt Templates
  getPromptTemplates(): Array<{ id: string; name: string; content: string; category: string; isBuiltin: boolean; createdAt: number }> {
    const rows = this.db.prepare('SELECT * FROM prompt_templates ORDER BY is_builtin DESC, name ASC').all() as any[]
    return rows.map(r => ({
      id: r.id,
      name: r.name,
      content: r.content,
      category: r.category,
      isBuiltin: !!r.is_builtin,
      createdAt: r.created_at
    }))
  }

  savePromptTemplate(t: { id: string; name: string; content: string; category: string }): void {
    this.db.prepare(
      'INSERT OR REPLACE INTO prompt_templates (id, name, content, category, is_builtin, created_at) VALUES (?, ?, ?, ?, 0, ?)'
    ).run(t.id, t.name, t.content, t.category, Date.now())
  }

  deletePromptTemplate(id: string): void {
    this.db.prepare('DELETE FROM prompt_templates WHERE id = ? AND is_builtin = 0').run(id)
  }

  close(): void {
    this.db.close()
  }
}

export const db = new DatabaseManager()
