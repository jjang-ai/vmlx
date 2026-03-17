import Database from 'better-sqlite3'
import { app, safeStorage } from 'electron'
import { join } from 'path'
import { existsSync, unlinkSync, renameSync } from 'fs'

function encryptValue(value: string): string {
  if (!value || !safeStorage.isEncryptionAvailable()) return value
  return 'enc:' + safeStorage.encryptString(value).toString('base64')
}

function decryptValue(value: string): string {
  if (!value || !value.startsWith('enc:')) return value  // legacy plaintext
  if (!safeStorage.isEncryptionAvailable()) return ''
  return safeStorage.decryptString(Buffer.from(value.slice(4), 'base64'))
}

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

export interface ImageSession {
  id: string
  modelName: string
  sessionType?: 'generate' | 'edit'
  createdAt: number
  updatedAt: number
}

export interface ImageGeneration {
  id: string
  sessionId: string
  prompt: string
  negativePrompt?: string
  modelName: string
  width: number
  height: number
  steps: number
  guidance: number
  seed?: number
  strength?: number
  elapsedSeconds?: number
  imagePath: string
  sourceImagePath?: string
  createdAt: number
}

export interface ImageModelPath {
  modelId: string
  quantize: number
  localPath: string
  repoId?: string
  downloadedAt: number
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
  private closed = false
  private dbPath: string
  /** Set when DB was recovered from corruption — main process shows dialog */
  recoveryBackupPath: string | null = null

  constructor() {
    this.dbPath = join(app.getPath('userData'), 'chats.db')
    const dbPath = this.dbPath
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
      } catch (_) { }
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

      CREATE TABLE IF NOT EXISTS bookmarks (
        path TEXT PRIMARY KEY,
        bookmark TEXT NOT NULL
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
        host TEXT NOT NULL DEFAULT '0.0.0.0',
        port INTEGER NOT NULL UNIQUE,
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

      CREATE TABLE IF NOT EXISTS image_sessions (
        id TEXT PRIMARY KEY,
        model_name TEXT NOT NULL,
        session_type TEXT DEFAULT 'generate',
        created_at INTEGER NOT NULL,
        updated_at INTEGER NOT NULL
      );
      CREATE INDEX IF NOT EXISTS idx_image_sessions_updated ON image_sessions(updated_at);

      CREATE TABLE IF NOT EXISTS image_generations (
        id TEXT PRIMARY KEY,
        session_id TEXT NOT NULL,
        prompt TEXT NOT NULL,
        negative_prompt TEXT,
        model_name TEXT NOT NULL,
        width INTEGER NOT NULL,
        height INTEGER NOT NULL,
        steps INTEGER NOT NULL,
        guidance REAL NOT NULL,
        seed INTEGER,
        strength REAL,
        elapsed_seconds REAL,
        image_path TEXT NOT NULL,
        source_image_path TEXT,
        created_at INTEGER NOT NULL,
        FOREIGN KEY (session_id) REFERENCES image_sessions(id) ON DELETE CASCADE
      );
      CREATE INDEX IF NOT EXISTS idx_image_gens_session ON image_generations(session_id);
    `)

    // Run all migrations inside a transaction for atomicity — if the app crashes
    // mid-migration, the transaction rolls back so we don't end up in a partial state.
    const runMigrations = this.db.transaction(() => {

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
      this.db.exec('ALTER TABLE chat_overrides ADD COLUMN git_enabled INTEGER DEFAULT 1')
    }
    if (!overrideColumns.find(c => c.name === 'utility_tools_enabled')) {
      this.db.exec('ALTER TABLE chat_overrides ADD COLUMN utility_tools_enabled INTEGER DEFAULT 1')
    }

    // Fix: older migration used DEFAULT 1 for enable_thinking, corrupting existing
    // chat overrides to thinking=ON instead of Auto (NULL). Check the column default
    // in the schema and reset if it's still 1.
    const tableInfo = this.db.pragma('table_info(chat_overrides)') as { name: string; dflt_value: string | null }[]
    const etCol = tableInfo.find(c => c.name === 'enable_thinking')
    if (etCol && etCol.dflt_value === '1') {
      // Only run UPDATE if there are actually rows with enable_thinking = 1
      const affected = this.db.prepare('SELECT COUNT(*) as cnt FROM chat_overrides WHERE enable_thinking = 1').get() as { cnt: number }
      if (affected.cnt > 0) {
        this.db.exec('UPDATE chat_overrides SET enable_thinking = NULL WHERE enable_thinking = 1')
        console.log(`[DB] Fixed enable_thinking DEFAULT 1 → reset ${affected.cnt} affected rows to NULL (Auto)`)
      }
    }

    // Fix: older migration added git_enabled/utility_tools_enabled without DEFAULT 1.
    // Existing rows with NULL values should be treated as enabled (1).
    const gitCol = tableInfo.find(c => c.name === 'git_enabled')
    if (gitCol && gitCol.dflt_value === null) {
      this.db.exec('UPDATE chat_overrides SET git_enabled = 1 WHERE git_enabled IS NULL')
    }
    const utilCol = tableInfo.find(c => c.name === 'utility_tools_enabled')
    if (utilCol && utilCol.dflt_value === null) {
      this.db.exec('UPDATE chat_overrides SET utility_tools_enabled = 1 WHERE utility_tools_enabled IS NULL')
    }

    // Clean up orphan benchmarks from deleted sessions
    this.db.exec('DELETE FROM benchmarks WHERE session_id NOT IN (SELECT id FROM sessions)')

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

    // Safe migration: add UNIQUE constraint on sessions.port for existing databases.
    // Deduplicate any existing port conflicts first (keep most recent session per port).
    try {
      const indexInfo = this.db.pragma('index_list(sessions)') as { name: string; unique: number }[]
      const hasUniquePort = indexInfo.some(idx =>
        idx.unique === 1 && (this.db.pragma(`index_info(${idx.name})`) as { name: string }[]).some(col => col.name === 'port')
      )
      if (!hasUniquePort) {
        // Remove duplicates: keep the row with the latest updated_at for each port
        this.db.exec(`
          DELETE FROM sessions WHERE rowid NOT IN (
            SELECT MAX(rowid) FROM sessions GROUP BY port
          )
        `)
        this.db.exec('CREATE UNIQUE INDEX IF NOT EXISTS idx_sessions_port_unique ON sessions(port)')
        console.log('[DB] Added UNIQUE index on sessions.port')
      }
    } catch (e) {
      console.warn('[DB] Could not add UNIQUE index on sessions.port:', e)
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

    // Safe migration: add model_settings table for per-model configuration
    this.db.exec(`
      CREATE TABLE IF NOT EXISTS model_settings (
        model_path TEXT PRIMARY KEY,
        alias TEXT,
        temperature REAL,
        top_p REAL,
        max_tokens INTEGER,
        ttl_minutes INTEGER,
        pinned INTEGER DEFAULT 0,
        port INTEGER,
        cache_quant TEXT,
        disk_cache_enabled INTEGER DEFAULT 0,
        reasoning_mode TEXT DEFAULT 'auto'
      )
    `)

    // Image editing schema migrations
    const imgGenCols = this.db.pragma('table_info(image_generations)') as { name: string }[]
    if (!imgGenCols.find(c => c.name === 'source_image_path')) {
      this.db.exec('ALTER TABLE image_generations ADD COLUMN source_image_path TEXT')
    }
    if (!imgGenCols.find(c => c.name === 'strength')) {
      this.db.exec('ALTER TABLE image_generations ADD COLUMN strength REAL')
    }
    const imgSessCols = this.db.pragma('table_info(image_sessions)') as { name: string }[]
    if (!imgSessCols.find(c => c.name === 'session_type')) {
      this.db.exec("ALTER TABLE image_sessions ADD COLUMN session_type TEXT DEFAULT 'generate'")
    }

    // Image model paths table — tracks where downloaded image models live on disk
    this.db.exec(`
      CREATE TABLE IF NOT EXISTS image_model_paths (
        model_id TEXT NOT NULL,
        quantize INTEGER NOT NULL,
        local_path TEXT NOT NULL,
        repo_id TEXT,
        downloaded_at INTEGER NOT NULL,
        PRIMARY KEY (model_id, quantize)
      )
    `)

    }) // end runMigrations transaction
    runMigrations()
  }

  // Folders
  createFolder(folder: Folder): void {
    this.ensureOpen()
    const stmt = this.db.prepare(`
      INSERT INTO folders (id, name, parent_id, color, icon, created_at)
      VALUES (?, ?, ?, ?, ?, ?)
    `)
    stmt.run(folder.id, folder.name, folder.parentId, folder.color, folder.icon, folder.createdAt)
  }

  getFolders(): Folder[] {
    this.ensureOpen()
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
    this.ensureOpen()
    const stmt = this.db.prepare('DELETE FROM folders WHERE id = ?')
    stmt.run(id)
  }

  // Chats
  createChat(chat: Chat): void {
    this.ensureOpen()
    const stmt = this.db.prepare(`
      INSERT INTO chats (id, title, folder_id, created_at, updated_at, model_id, model_path)
      VALUES (?, ?, ?, ?, ?, ?, ?)
    `)
    stmt.run(chat.id, chat.title, chat.folderId ?? null, chat.createdAt, chat.updatedAt, chat.modelId || 'default', chat.modelPath ?? null)
  }

  getChats(folderId?: string): Chat[] {
    this.ensureOpen()
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
    this.ensureOpen()
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
    this.ensureOpen()
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

  getRecentChats(limit: number = 100) {
    this.ensureOpen()
    return this.db.prepare(`
      SELECT c.id, c.title, c.model_id, c.model_path, c.created_at, c.updated_at,
             (SELECT COUNT(*) FROM messages WHERE chat_id = c.id) as message_count
      FROM chats c
      ORDER BY c.updated_at DESC
      LIMIT ?
    `).all(limit).map((row: any) => ({
      id: row.id,
      title: row.title,
      modelId: row.model_id,
      modelPath: row.model_path,
      createdAt: row.created_at,
      updatedAt: row.updated_at,
      messageCount: row.message_count
    }))
  }

  updateChat(id: string, updates: Partial<Chat>): void {
    this.ensureOpen()
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
    this.ensureOpen()
    const stmt = this.db.prepare('DELETE FROM chats WHERE id = ?')
    stmt.run(id)
  }

  // Messages
  addMessage(message: Message): void {
    this.ensureOpen()
    const insertAndUpdate = this.db.transaction(() => {
      const stmt = this.db.prepare(`
        INSERT OR REPLACE INTO messages (id, chat_id, role, content, timestamp, tokens, metrics_json, tool_calls_json, reasoning_content)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
      `)
      stmt.run(message.id, message.chatId, message.role, message.content, message.timestamp, message.tokens, message.metricsJson, message.toolCallsJson, message.reasoningContent)

      // Update chat's updatedAt atomically with the message insert
      this.updateChat(message.chatId, { updatedAt: message.timestamp })
    })
    insertAndUpdate()
  }

  /** Update an existing message's content in-place (for incremental persistence during streaming) */
  updateMessageContent(messageId: string, content: string, reasoningContent?: string): void {
    this.ensureOpen()
    const stmt = this.db.prepare(`
      UPDATE messages SET content = ?, reasoning_content = ? WHERE id = ?
    `)
    stmt.run(content, reasoningContent || null, messageId)
  }

  /** Delete a message by ID (used to clean up empty pre-inserted placeholders on error) */
  deleteMessage(messageId: string): void {
    this.ensureOpen()
    this.db.prepare('DELETE FROM messages WHERE id = ?').run(messageId)
  }

  getMessages(chatId: string): Message[] {
    this.ensureOpen()
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
    this.ensureOpen()
    const upsert = this.db.transaction(() => {
    // Ensure chat row exists (FK constraint) — create stub if needed
    const exists = this.db.prepare('SELECT 1 FROM chats WHERE id = ?').get(overrides.chatId)
    if (!exists) {
      const now = Date.now()
      this.db.prepare(
        'INSERT OR IGNORE INTO chats (id, title, folder_id, created_at, updated_at, model_id) VALUES (?, ?, NULL, ?, ?, ?)'
      ).run(overrides.chatId, 'New Chat', now, now, 'default')
    }
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
    }) // end upsert transaction
    upsert()
  }

  getChatOverrides(chatId: string): ChatOverrides | undefined {
    this.ensureOpen()
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
    this.ensureOpen()
    const stmt = this.db.prepare('DELETE FROM chat_overrides WHERE chat_id = ?')
    stmt.run(chatId)
  }

  // Search
  searchChats(query: string): Chat[] {
    this.ensureOpen()
    const stmt = this.db.prepare(`
      SELECT DISTINCT c.* FROM chats c
      LEFT JOIN messages m ON c.id = m.chat_id
      WHERE c.title LIKE ?
         OR (m.content LIKE ? AND m.content NOT LIKE '[{"type":%')
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
    this.ensureOpen()
    const stmt = this.db.prepare(`
      INSERT INTO sessions (id, model_path, model_name, host, port, pid, status, config, created_at, updated_at, last_started_at, last_stopped_at, type, remote_url, remote_api_key, remote_model, remote_organization)
      VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    `)
    stmt.run(
      session.id, session.modelPath, session.modelName, session.host, session.port,
      session.pid, session.status, session.config, session.createdAt, session.updatedAt,
      session.lastStartedAt, session.lastStoppedAt,
      session.type || 'local', session.remoteUrl, session.remoteApiKey ? encryptValue(session.remoteApiKey) : null, session.remoteModel, session.remoteOrganization
    )
  }

  getSession(id: string): Session | undefined {
    this.ensureOpen()
    const stmt = this.db.prepare('SELECT * FROM sessions WHERE id = ?')
    const row = stmt.get(id) as any
    if (!row) return undefined
    return this.mapSessionRow(row)
  }

  getSessions(): Session[] {
    this.ensureOpen()
    const stmt = this.db.prepare('SELECT * FROM sessions ORDER BY updated_at DESC')
    return stmt.all().map((row: any) => this.mapSessionRow(row))
  }

  getSessionByModelPath(modelPath: string): Session | undefined {
    this.ensureOpen()
    // Normalize trailing slashes: try both with and without to handle legacy data
    const normalized = modelPath.replace(/\/+$/, '')
    const stmt = this.db.prepare('SELECT * FROM sessions WHERE model_path = ? OR model_path = ?')
    const row = stmt.get(normalized, normalized + '/') as any
    if (!row) return undefined
    return this.mapSessionRow(row)
  }

  updateSession(id: string, updates: Partial<Session>): void {
    this.ensureOpen()
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
    if ('remoteApiKey' in updates) { fields.push('remote_api_key = ?'); values.push(updates.remoteApiKey ? encryptValue(updates.remoteApiKey) : null) }
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
    this.ensureOpen()
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
      remoteApiKey: row.remote_api_key ? decryptValue(row.remote_api_key) : undefined,
      remoteModel: row.remote_model,
      remoteOrganization: row.remote_organization
    }
  }

  // Settings (key-value store)
  getSetting(key: string): string | undefined {
    this.ensureOpen()
    const stmt = this.db.prepare('SELECT value FROM settings WHERE key = ?')
    const row = stmt.get(key) as { value: string } | undefined
    if (!row?.value) return row?.value
    return (key.toLowerCase().includes('apikey') || key.toLowerCase().includes('api_key'))
      ? decryptValue(row.value) : row.value
  }

  setSetting(key: string, value: string): void {
    this.ensureOpen()
    const encValue = (key.toLowerCase().includes('apikey') || key.toLowerCase().includes('api_key'))
      ? encryptValue(value) : value
    const stmt = this.db.prepare('INSERT OR REPLACE INTO settings (key, value) VALUES (?, ?)')
    stmt.run(key, encValue)
  }

  deleteSetting(key: string): void {
    this.ensureOpen()
    const stmt = this.db.prepare('DELETE FROM settings WHERE key = ?')
    stmt.run(key)
  }

  // ─── Per-Model Settings ──────────────────────────────────────────────────────────

  getModelSettings(modelPath: string): Record<string, any> | undefined {
    this.ensureOpen()
    const stmt = this.db.prepare('SELECT * FROM model_settings WHERE model_path = ?')
    return stmt.get(modelPath) as Record<string, any> | undefined
  }

  getAllModelSettings(): Record<string, any>[] {
    this.ensureOpen()
    const stmt = this.db.prepare('SELECT * FROM model_settings ORDER BY model_path')
    return stmt.all() as Record<string, any>[]
  }

  saveModelSettings(modelPath: string, settings: Record<string, any>): void {
    this.ensureOpen()
    const stmt = this.db.prepare(`
      INSERT INTO model_settings
        (model_path, alias, temperature, top_p, max_tokens, ttl_minutes, pinned, port, cache_quant, disk_cache_enabled, reasoning_mode)
      VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
      ON CONFLICT(model_path) DO UPDATE SET
        alias = excluded.alias,
        temperature = excluded.temperature,
        top_p = excluded.top_p,
        max_tokens = excluded.max_tokens,
        ttl_minutes = excluded.ttl_minutes,
        pinned = excluded.pinned,
        port = excluded.port,
        cache_quant = excluded.cache_quant,
        disk_cache_enabled = excluded.disk_cache_enabled,
        reasoning_mode = excluded.reasoning_mode
    `)
    stmt.run(
      modelPath,
      settings.alias ?? null,
      settings.temperature ?? null,
      settings.top_p ?? null,
      settings.max_tokens ?? null,
      settings.ttl_minutes ?? null,
      settings.pinned ? 1 : 0,
      settings.port ?? null,
      settings.cache_quant ?? null,
      settings.disk_cache_enabled ? 1 : 0,
      settings.reasoning_mode ?? 'auto',
    )
  }

  deleteModelSettings(modelPath: string): void {
    this.ensureOpen()
    const stmt = this.db.prepare('DELETE FROM model_settings WHERE model_path = ?')
    stmt.run(modelPath)
  }

  // ─── Sandboxed Bookmarks ────────────────────────────────────────────────────────

  saveBookmark(path: string, bookmark: string): void {
    this.ensureOpen()
    const stmt = this.db.prepare('INSERT OR REPLACE INTO bookmarks (path, bookmark) VALUES (?, ?)')
    stmt.run(path, bookmark)
  }

  getBookmark(path: string): string | null {
    this.ensureOpen()
    const stmt = this.db.prepare('SELECT bookmark FROM bookmarks WHERE path = ?')
    const result = stmt.get(path) as { bookmark: string } | undefined
    return result ? result.bookmark : null
  }

  getAllBookmarks(): { path: string; bookmark: string }[] {
    this.ensureOpen()
    const stmt = this.db.prepare('SELECT path, bookmark FROM bookmarks')
    return stmt.all() as { path: string; bookmark: string }[]
  }

  // Benchmarks
  saveBenchmark(b: BenchmarkResult): void {
    this.ensureOpen()
    const stmt = this.db.prepare(
      'INSERT INTO benchmarks (id, session_id, model_path, model_name, results_json, created_at) VALUES (?, ?, ?, ?, ?, ?)'
    )
    stmt.run(b.id, b.sessionId, b.modelPath, b.modelName || null, b.resultsJson, b.createdAt)
  }

  getBenchmarks(modelPath?: string): BenchmarkResult[] {
    this.ensureOpen()
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
    this.ensureOpen()
    const stmt = this.db.prepare('DELETE FROM benchmarks WHERE id = ?')
    stmt.run(id)
  }

  // Prompt Templates
  getPromptTemplates(): Array<{ id: string; name: string; content: string; category: string; isBuiltin: boolean; createdAt: number }> {
    this.ensureOpen()
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
    this.ensureOpen()
    this.db.prepare(
      'INSERT OR REPLACE INTO prompt_templates (id, name, content, category, is_builtin, created_at) VALUES (?, ?, ?, ?, 0, ?)'
    ).run(t.id, t.name, t.content, t.category, Date.now())
  }

  deletePromptTemplate(id: string): void {
    this.ensureOpen()
    this.db.prepare('DELETE FROM prompt_templates WHERE id = ? AND is_builtin = 0').run(id)
  }

  // ─── Image Sessions & Generations ─────────────────────────────────────────

  createImageSession(session: ImageSession): void {
    this.ensureOpen()
    this.db.prepare(
      'INSERT INTO image_sessions (id, model_name, session_type, created_at, updated_at) VALUES (?, ?, ?, ?, ?)'
    ).run(session.id, session.modelName, session.sessionType || 'generate', session.createdAt, session.updatedAt)
  }

  getImageSessions(): ImageSession[] {
    this.ensureOpen()
    return this.db.prepare('SELECT * FROM image_sessions ORDER BY updated_at DESC').all().map((r: any) => ({
      id: r.id, modelName: r.model_name, sessionType: r.session_type || 'generate', createdAt: r.created_at, updatedAt: r.updated_at
    }))
  }

  getImageSession(id: string): ImageSession | undefined {
    this.ensureOpen()
    const r = this.db.prepare('SELECT * FROM image_sessions WHERE id = ?').get(id) as any
    if (!r) return undefined
    return { id: r.id, modelName: r.model_name, sessionType: r.session_type || 'generate', createdAt: r.created_at, updatedAt: r.updated_at }
  }

  updateImageSession(id: string, updates: Partial<ImageSession>): void {
    this.ensureOpen()
    const fields: string[] = []
    const values: any[] = []
    if (updates.modelName !== undefined) { fields.push('model_name = ?'); values.push(updates.modelName) }
    if (updates.updatedAt !== undefined) { fields.push('updated_at = ?'); values.push(updates.updatedAt) }
    if (fields.length === 0) return
    values.push(id)
    this.db.prepare(`UPDATE image_sessions SET ${fields.join(', ')} WHERE id = ?`).run(...values)
  }

  deleteImageSession(id: string): void {
    this.ensureOpen()
    this.db.prepare('DELETE FROM image_sessions WHERE id = ?').run(id)
  }

  addImageGeneration(gen: ImageGeneration): void {
    this.ensureOpen()
    this.db.prepare(
      `INSERT INTO image_generations (id, session_id, prompt, negative_prompt, model_name, width, height, steps, guidance, seed, strength, elapsed_seconds, image_path, source_image_path, created_at)
       VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)`
    ).run(gen.id, gen.sessionId, gen.prompt, gen.negativePrompt || null, gen.modelName,
      gen.width, gen.height, gen.steps, gen.guidance, gen.seed ?? null, gen.strength ?? null,
      gen.elapsedSeconds ?? null, gen.imagePath, gen.sourceImagePath || null, gen.createdAt)
    // Update session's updatedAt
    this.updateImageSession(gen.sessionId, { updatedAt: gen.createdAt })
  }

  getImageGenerations(sessionId: string): ImageGeneration[] {
    this.ensureOpen()
    return this.db.prepare('SELECT * FROM image_generations WHERE session_id = ? ORDER BY created_at ASC').all(sessionId).map((r: any) => ({
      id: r.id, sessionId: r.session_id, prompt: r.prompt, negativePrompt: r.negative_prompt,
      modelName: r.model_name, width: r.width, height: r.height, steps: r.steps, guidance: r.guidance,
      seed: r.seed, strength: r.strength, elapsedSeconds: r.elapsed_seconds,
      imagePath: r.image_path, sourceImagePath: r.source_image_path, createdAt: r.created_at
    }))
  }

  deleteImageGeneration(id: string): void {
    this.ensureOpen()
    this.db.prepare('DELETE FROM image_generations WHERE id = ?').run(id)
  }

  // ─── Image Model Paths ──────────────────────────────────────────────────────

  getImageModelPath(modelId: string, quantize: number): ImageModelPath | undefined {
    this.ensureOpen()
    const row = this.db.prepare(
      'SELECT * FROM image_model_paths WHERE model_id = ? AND quantize = ?'
    ).get(modelId, quantize) as any
    if (!row) return undefined
    return {
      modelId: row.model_id,
      quantize: row.quantize,
      localPath: row.local_path,
      repoId: row.repo_id || undefined,
      downloadedAt: row.downloaded_at,
    }
  }

  setImageModelPath(modelId: string, quantize: number, localPath: string, repoId?: string): void {
    this.ensureOpen()
    this.db.prepare(
      `INSERT OR REPLACE INTO image_model_paths (model_id, quantize, local_path, repo_id, downloaded_at)
       VALUES (?, ?, ?, ?, ?)`
    ).run(modelId, quantize, localPath, repoId || null, Date.now())
  }

  deleteImageModelPath(modelId: string, quantize: number): void {
    this.ensureOpen()
    this.db.prepare(
      'DELETE FROM image_model_paths WHERE model_id = ? AND quantize = ?'
    ).run(modelId, quantize)
  }

  getAllImageModelPaths(): ImageModelPath[] {
    this.ensureOpen()
    return this.db.prepare('SELECT * FROM image_model_paths ORDER BY model_id, quantize').all().map((row: any) => ({
      modelId: row.model_id,
      quantize: row.quantize,
      localPath: row.local_path,
      repoId: row.repo_id || undefined,
      downloadedAt: row.downloaded_at,
    }))
  }

  close(): void {
    if (!this.closed) {
      this.closed = true
      try {
        this.db.close()
      } catch (_) {
        // Already closed or GC'd — ignore
      }
    }
  }

  /**
   * Ensure DB is open. If it was closed (e.g., during quit) but something
   * still needs it (IPC call in flight, crash handler writing status),
   * reopen it to prevent cascading "connection is not open" errors.
   */
  private ensureOpen(): void {
    if (this.closed) {
      console.warn('[DB] Database was closed but is being accessed — reopening')
      try {
        this.db = new Database(this.dbPath)
        this.db.pragma('journal_mode = WAL')
        this.db.pragma('foreign_keys = ON')
        this.closed = false
      } catch (err) {
        console.error('[DB] Failed to reopen database:', err)
        throw err
      }
    }
  }
}

export const db = new DatabaseManager()
