import { useState, useRef } from 'react'
import { Modal } from '../ui/Modal'
export interface SessionConfig {
  host: string
  port: number
  apiKey: string
  rateLimit: number
  timeout: number
  maxNumSeqs: number
  prefillBatchSize: number
  completionBatchSize: number
  continuousBatching: boolean
  enablePrefixCache: boolean
  prefixCacheSize: number
  cacheMemoryMb: number
  cacheMemoryPercent: number
  cacheTtlMinutes: number
  noMemoryAwareCache: boolean
  usePagedCache: boolean
  pagedCacheBlockSize: number
  maxCacheBlocks: number
  kvCacheQuantization: string
  kvCacheGroupSize: number
  enableDiskCache: boolean
  diskCacheMaxGb: number
  diskCacheDir: string
  enableBlockDiskCache: boolean
  blockDiskCacheMaxGb: number
  blockDiskCacheDir: string
  streamInterval: number
  maxTokens: number
  mcpConfig: string
  enableAutoToolChoice?: boolean
  toolCallParser: string
  reasoningParser: string
  isMultimodal?: boolean
  servedModelName: string
  speculativeModel: string
  numDraftTokens: number
  defaultTemperature: number
  defaultTopP: number
  embeddingModel: string
  additionalArgs: string
  streamFromDisk: boolean
  streamMemoryPercent: number
  enableJit: boolean
  idleTimeoutSoftMin?: number
  idleTimeoutHardMin?: number
  autoSleepEnabled?: boolean
  logLevel: string
  corsOrigins: string
  maxContextLength: number
  imageMode?: string
  imageQuantize?: number
}

export const DEFAULT_CONFIG: SessionConfig = {
  host: '0.0.0.0',
  port: 8000,
  apiKey: '',
  rateLimit: 0,
  timeout: 300,
  maxNumSeqs: 256,
  prefillBatchSize: 0,
  completionBatchSize: 0,
  continuousBatching: true,
  enablePrefixCache: true,
  prefixCacheSize: 100,
  cacheMemoryMb: 0,
  cacheMemoryPercent: 30,
  cacheTtlMinutes: 0,
  noMemoryAwareCache: false,
  usePagedCache: true,
  pagedCacheBlockSize: 64,
  maxCacheBlocks: 1000,
  kvCacheQuantization: 'q8',
  kvCacheGroupSize: 64,
  enableDiskCache: false,
  diskCacheMaxGb: 10,
  diskCacheDir: '',
  enableBlockDiskCache: true,
  blockDiskCacheMaxGb: 10,
  blockDiskCacheDir: '',
  streamInterval: 1,
  maxTokens: 32768,
  mcpConfig: '',
  // enableAutoToolChoice intentionally omitted (undefined = auto-detect from model config).
  // false blocks auto-detection because ?? doesn't fall through on false.
  toolCallParser: 'auto',
  reasoningParser: 'auto',
  isMultimodal: undefined,
  servedModelName: '',
  speculativeModel: '',
  numDraftTokens: 3,
  defaultTemperature: 0,
  defaultTopP: 0,
  embeddingModel: '',
  additionalArgs: '',
  streamFromDisk: false,
  streamMemoryPercent: 90,
  enableJit: false,
  logLevel: 'INFO',
  corsOrigins: '*',
  maxContextLength: 0,
  imageMode: undefined,
  imageQuantize: undefined
}

interface SessionConfigFormProps {
  config: SessionConfig
  onChange: <K extends keyof SessionConfig>(key: K, value: SessionConfig[K]) => void
  onReset?: () => void
  /** Detected model cache type ('kv', 'mamba', etc.) for feature gating */
  detectedCacheType?: string
  /** Detected model max context length from config.json (max_position_embeddings) */
  detectedMaxContext?: number
  /** Model type — image models show minimal settings */
  modelType?: 'text' | 'image'
  /** Image mode — 'edit' or 'generate' (only relevant when modelType is 'image') */
  imageMode?: string
}

export function SessionConfigForm({ config, onChange, onReset, detectedCacheType, detectedMaxContext, modelType, imageMode }: SessionConfigFormProps) {
  const isImage = modelType === 'image'
  const isImageEdit = isImage && (imageMode === 'edit' || config.imageMode === 'edit')
  const [expandedSections, setExpandedSections] = useState({
    server: true,
    concurrent: false,
    prefixCache: false,
    pagedCache: false,
    kvCacheQuant: false,
    diskCache: false,
    diskStreaming: false,
    power: false,
    performance: false,
    tools: false,
    specDecode: false
  })

  const [showCachingHelp, setShowCachingHelp] = useState(false)

  const batchingOff = !config.continuousBatching
  const effectivelyNoBatching = batchingOff
  const prefixOff = !config.enablePrefixCache
  const isMambaCache = detectedCacheType === 'mamba' || detectedCacheType === 'hybrid'

  const toggleSection = (section: keyof typeof expandedSections) => {
    setExpandedSections(prev => ({ ...prev, [section]: !prev[section] }))
  }

  return (
    <div className="space-y-0">
      {/* Server Settings */}
      <Section title="Server Settings" expanded={expandedSections.server} onToggle={() => toggleSection('server')}>
        <Field label="Host" tooltip="The network interface to bind to. Default 0.0.0.0 accepts connections from any machine on your network. Change to 127.0.0.1 (localhost) to restrict to local-only access. If you don't need remote access, consider changing this to 127.0.0.1 for security.">
          <input type="text" value={config.host} onChange={e => onChange('host', e.target.value)} className="cfg-input" />
        </Field>
        <SliderField
          label="Port"
          tooltip="The TCP port the server listens on. Each running model instance needs a unique port. Ports are auto-assigned starting from 8000. You can manually set any port between 1024-65535 that isn't already in use."
          value={config.port}
          onChange={v => onChange('port', v)}
          min={1024}
          max={65535}
          step={1}
          defaultValue={DEFAULT_CONFIG.port}
        />
        <Field label="API Key" tooltip="Optional authentication key for the OpenAI-compatible API. When set, all API requests must include this key in the Authorization header. Leave empty to allow unauthenticated access (fine for local-only servers).">
          <input type="password" value={config.apiKey} onChange={e => onChange('apiKey', e.target.value)} placeholder="Leave empty for no auth" className="cfg-input" />
        </Field>
        <Field label="Served Model Name" tooltip="Custom name to expose via the /v1/models API and in response objects. When set, API clients can use this name instead of the full model path. Both the custom name and the actual model name are listed in /v1/models. Leave empty to auto-derive from model path (e.g. 'mlx-community/Llama-3.2-3B').">
          <input type="text" value={config.servedModelName} onChange={e => onChange('servedModelName', e.target.value)} placeholder="Auto (from model path)" className="cfg-input" />
        </Field>
        <SliderField
          label="Rate Limit (req/min)"
          tooltip="Maximum number of API requests allowed per minute. Set to 0 to disable rate limiting. Useful when exposing the server to multiple users or external applications to prevent overloading."
          value={config.rateLimit}
          onChange={v => onChange('rateLimit', v)}
          min={1}
          max={1000}
          step={10}
          defaultValue={60}
          allowUnlimited
          unlimitedValue={0}
          unlimitedLabel="Disabled"
        />
        <SliderField
          label="Timeout (seconds)"
          tooltip="Maximum time in seconds to wait for a single inference request to complete before timing out. Increase this for very long generations or slow models. Default 300s (5 minutes) should be sufficient for most use cases."
          value={config.timeout}
          onChange={v => onChange('timeout', v)}
          min={10}
          max={3600}
          step={10}
          defaultValue={DEFAULT_CONFIG.timeout}
          allowUnlimited
          unlimitedValue={0}
          unlimitedLabel="No limit"
        />
        <Field label="Log Level" tooltip="Controls how much detail the server logs. DEBUG shows everything (very verbose). INFO is the default. WARNING and ERROR reduce noise to only important messages.">
          <select value={config.logLevel || 'INFO'} onChange={e => onChange('logLevel', e.target.value)} className="cfg-input">
            <option value="DEBUG">DEBUG (verbose)</option>
            <option value="INFO">INFO (default)</option>
            <option value="WARNING">WARNING</option>
            <option value="ERROR">ERROR (minimal)</option>
          </select>
        </Field>
        <Field label="CORS Origins" tooltip="Allowed origins for cross-origin API requests (from web browsers). Use * to allow all origins, or a comma-separated list of specific origins (e.g. http://localhost:3000,https://myapp.com). Only matters when external web apps call your API.">
          <input type="text" value={config.corsOrigins || '*'} onChange={e => onChange('corsOrigins', e.target.value)} placeholder="* (allow all)" className="cfg-input" />
        </Field>
      </Section>

      {/* Concurrent Processing */}
      {isImage && (
        <div className="px-4 py-3 text-xs text-muted-foreground border-b border-border">
          {isImageEdit
            ? <>This is an image editing server. Only server settings (host, port, timeout) apply. Use the Image tab to edit images or call <code className="bg-muted px-1 rounded">/v1/images/edits</code>.</>
            : <>This is an image generation server. Only server settings (host, port, timeout) apply. Use the Image tab to generate images or call <code className="bg-muted px-1 rounded">/v1/images/generations</code>.</>
          }
        </div>
      )}

      <Section title="Concurrent Processing" expanded={expandedSections.concurrent} onToggle={() => toggleSection('concurrent')} hidden={isImage}>
        <div className="flex items-center gap-2 mb-2">
          <PerformanceHint text="Controls how many requests your server handles at once. Keep Continuous Batching ON to enable the caching engine." />
          <button
            onClick={(e) => { e.preventDefault(); e.stopPropagation(); setShowCachingHelp(true) }}
            className="w-6 h-6 flex items-center justify-center rounded-full bg-accent/50 text-accent-foreground hover:bg-accent hover:text-white transition-colors text-xs font-bold"
            title="Caching & Compatibility Reference"
          >
            ?
          </button>
        </div>
        <SliderField
          label="Max Concurrent Sequences"
          tooltip="Maximum number of sequences (requests) that can be processed simultaneously. Higher values allow more parallel users but consume more memory. For single-user local use, 1-4 is sufficient. For multi-user servers, 16-256 depending on available RAM."
          value={config.maxNumSeqs}
          onChange={v => onChange('maxNumSeqs', v)}
          min={1}
          max={1024}
          step={1}
          defaultValue={DEFAULT_CONFIG.maxNumSeqs}
          allowUnlimited
          unlimitedValue={0}
          unlimitedLabel="Default (256)"
        />
        <SliderField
          label="Prefill Batch Size"
          tooltip="Maximum number of tokens to process in a single prefill (prompt processing) step. Larger batches use more memory but process prompts faster. Reduce if you're running out of memory during prompt processing."
          value={config.prefillBatchSize}
          onChange={v => onChange('prefillBatchSize', v)}
          min={1}
          max={4096}
          step={64}
          defaultValue={512}
          allowUnlimited
          unlimitedValue={0}
          unlimitedLabel="Default (8)"
        />
        <SliderField
          label="Completion Batch Size"
          tooltip="Maximum number of tokens to generate in a single completion (token generation) step. Similar to prefill batch size but for the generation phase. Larger values can improve throughput for multi-user scenarios."
          value={config.completionBatchSize}
          onChange={v => onChange('completionBatchSize', v)}
          min={1}
          max={4096}
          step={64}
          defaultValue={512}
          allowUnlimited
          unlimitedValue={0}
          unlimitedLabel="Default (32)"
        />
        <CheckField label="Continuous Batching" tooltip="Processes multiple user requests simultaneously by continuously updating the batch. Crucial for serving multiple users efficiently. If disabled, requests are processed one by one (ideal for single-user peak throughput)." checked={config.continuousBatching} onChange={v => onChange('continuousBatching', v)} />
        <PerformanceHint text="Keep ON for best performance. This is the master switch — turning it off disables all caching features below." />
        {!config.continuousBatching && config.enablePrefixCache && (
          <InfoNote text="Continuous batching will be auto-enabled at launch because prefix cache requires it." />
        )}
        {!config.continuousBatching && (
          <InfoNote text="Turning this off disables: prefix caching, paged KV cache, KV cache quantization, and disk caching. Enable it to unlock these features." />
        )}
      </Section>

      {/* Disk-Streaming Mode */}
      <Section title="Disk-Streaming Mode" expanded={expandedSections.diskStreaming} onToggle={() => toggleSection('diskStreaming')} hidden={isImage}>
        <CheckField
          label="Stream from Disk (SSD)"
          tooltip="Enable for models that exceed available RAM. Weights stay on SSD and are paged in on demand by macOS. Automatically disables ALL caching, limits to 1 sequence. Expect 2-5x slower inference but the model RUNS instead of crashing."
          checked={config.streamFromDisk}
          onChange={v => onChange('streamFromDisk', v)}
        />
        {config.streamFromDisk && (
          <>
            <SliderField
              label="Virtual Memory Budget (%)"
              tooltip="Controls how much virtual memory Metal can allocate (as a multiplier of physical RAM). Higher values allow larger models by letting macOS page weights to SSD. At 90% (default), Metal gets ~3.5x your RAM in virtual space. Lower if you experience system instability."
              value={config.streamMemoryPercent}
              onChange={v => onChange('streamMemoryPercent', v)}
              min={50}
              max={95}
              step={5}
              defaultValue={90}
            />
            <div className="rounded-md border border-yellow-600/40 bg-yellow-900/20 px-3 py-2 text-xs text-yellow-200">
              Disk-streaming mode: All caching features will be automatically disabled.
              Performance will be 2-5x slower than normal. Only use this when the model
              exceeds your available RAM. Lower the memory % if inference crashes.
            </div>
          </>
        )}
      </Section>

      {/* Prefix Cache */}
      <Section title="Prefix Cache" expanded={expandedSections.prefixCache} onToggle={() => toggleSection('prefixCache')} hidden={isImage}>
        {!effectivelyNoBatching && <PerformanceHint text="Speeds up repeated conversations by remembering previous prompts. Makes follow-up messages much faster (lower time-to-first-token)." />}
        {batchingOff && <IncompatWarning text="Prefix cache requires continuous batching. Turn on 'Continuous Batching' in the Concurrent Processing section above to enable prefix caching." />}
        <CheckField label="Enable Prefix Cache" tooltip="Caches prompt prefixes in memory. If you send the same system prompt or document multiple times, the server reuses the cached internal states instead of recomputing them, drastically reducing Time-To-First-Token (TTFT) and saving GPU compute. Highly recommended for agents and tool calling." checked={config.enablePrefixCache} onChange={v => onChange('enablePrefixCache', v)} />
        {config.enablePrefixCache && (
          <>
            <CheckField label="Legacy Entry-Count Cache" tooltip="Switches from memory-aware cache (which uses Cache Memory %, Cache Memory Limit, and Cache TTL controls) to a simpler entry-count cache. When ON: you control cache by max entries only. When OFF: you get fine-grained memory budget controls (% of RAM, MB limit, TTL expiration). Memory-aware mode is recommended for most users." checked={config.noMemoryAwareCache} onChange={v => onChange('noMemoryAwareCache', v)} />
            {config.noMemoryAwareCache ? (
              <>
                <InfoNote text="Legacy mode active — Cache Memory %, Cache Memory Limit, and Cache TTL are hidden. Turn off 'Legacy Entry-Count Cache' above to use memory-aware caching with those controls." />
                <SliderField
                  label="Max Cache Entries"
                  tooltip="Maximum number of prefix cache entries to store when using legacy entry-count mode. Each entry stores the KV cache for one unique prefix. Higher values cache more prefixes but use more memory. For finer control over memory usage, switch to memory-aware mode by unchecking 'Legacy Entry-Count Cache' above."
                  value={config.prefixCacheSize}
                  onChange={v => onChange('prefixCacheSize', v)}
                  min={1}
                  max={10000}
                  step={10}
                  defaultValue={DEFAULT_CONFIG.prefixCacheSize}
                  allowUnlimited
                  unlimitedValue={0}
                  unlimitedLabel="Default (100)"
                />
              </>
            ) : (
              <>
                <SliderField
                  label="Cache Memory Limit (MB)"
                  tooltip="Hard limit on memory used by the prefix cache in megabytes. Set to 'Auto-detect' to let the system auto-detect based on available RAM and the percentage setting below. Set an explicit value if you need to reserve memory for other applications."
                  value={config.cacheMemoryMb}
                  onChange={v => onChange('cacheMemoryMb', v)}
                  min={256}
                  max={65536}
                  step={256}
                  defaultValue={4096}
                  allowUnlimited
                  unlimitedValue={0}
                  unlimitedLabel="Auto-detect"
                />
                <SliderField
                  label="Cache Memory %"
                  tooltip="Percentage of available system RAM to allocate for the prefix cache. Only used when Cache Memory Limit is set to 'Auto-detect'. Default 30% is a good balance — lower this for large models that leave little headroom (e.g. 10-15% for 120GB+ models on 256GB systems). Higher values cache more prefixes but risk memory pressure during long generations."
                  value={config.cacheMemoryPercent}
                  onChange={v => onChange('cacheMemoryPercent', v)}
                  min={1}
                  max={100}
                  step={1}
                  defaultValue={DEFAULT_CONFIG.cacheMemoryPercent}
                  maxInput={100}
                />
                {config.usePagedCache && <IncompatWarning text="Cache TTL has no effect when paged cache is enabled — paged cache uses block-count LRU eviction instead. To control paged cache size, adjust 'Max Cache Blocks' in the Paged KV Cache section below. To use time-based TTL, disable 'Use Paged KV Cache' in the Paged KV Cache section." />}
                <SliderField
                  label="Cache TTL (minutes)"
                  tooltip="Time-to-live for memory-aware cache entries. Entries not accessed within this window are evicted to free memory. 'No expiration' means entries are only evicted by memory pressure. Note: this setting has no effect when Paged KV Cache is enabled (paged cache uses its own LRU eviction based on Max Cache Blocks)."
                  value={config.cacheTtlMinutes}
                  onChange={v => onChange('cacheTtlMinutes', v)}
                  min={1}
                  max={120}
                  step={5}
                  defaultValue={30}
                  allowUnlimited
                  unlimitedValue={0}
                  unlimitedLabel="No expiration"
                  disabled={config.usePagedCache}
                />
              </>
            )}

            {/* Caching Help Modal */}
            {showCachingHelp && (
              <Modal title="Caching & Compatibility Engine" onClose={() => setShowCachingHelp(false)} className="max-w-2xl max-h-[85vh] overflow-y-auto">
                <div className="space-y-6 text-sm">
                  <div>
                    <h3 className="text-base font-semibold text-foreground mb-2">The Continuous Batching Engine</h3>
                    <p className="text-muted-foreground leading-relaxed">
                      <strong>Continuous Batching</strong> is the heart of vMLX's server performance. Unlike simple mode (which processes exactly one request at a time), continuous batching allows multiple requests to be processed simultaneously. More importantly, <strong>it is required to enable all advanced caching features</strong> (Prefix Cache, Paged Cache, KV Quantization, and Disk Cache).
                    </p>
                  </div>

                  <div>
                    <h3 className="text-base font-semibold text-foreground mb-2">Prefix Caching (Memory-Aware vs Legacy)</h3>
                    <p className="text-muted-foreground leading-relaxed mb-2">
                      Prefix caching drastically speeds up interactions by remembering previous prompts (like a system prompt or a long document), skipping the expensive prefill phase.
                    </p>
                    <ul className="list-disc pl-5 space-y-2 text-muted-foreground">
                      <li><strong>Memory-Aware (Default):</strong> Intelligently manages the cache based on explicit memory boundaries (MB) or a percentage of total system RAM. It automatically evicts the oldest items when crossing these limits.</li>
                      <li><strong>Legacy Entry-Count:</strong> A simpler system that just stores a fixed number of complete prompt states regardless of their size. Useful if you want strict deterministic eviction.</li>
                    </ul>
                  </div>

                  <div>
                    <h3 className="text-base font-semibold text-foreground mb-2">Mamba & Hybrid Compatibility</h3>
                    <p className="text-muted-foreground leading-relaxed mb-2">
                      Newer models like Qwen 2.5/3, Falcon Mamba, and Jamba mix standard Attention (KV cache) with SSM blocks (Mamba/Arrays cache).
                    </p>
                    <ul className="list-disc pl-5 space-y-2 text-muted-foreground">
                      <li><strong>KV Quantization:</strong> vMLX securely isolates Mamba layers. If you turn on KV Quantization (e.g. q8), it will safely compress the Attention layers while leaving the internal Mamba/SSM memory at full precision, ensuring no corruption or quality loss.</li>
                      <li><strong>Paged Cache Requirement:</strong> Since cumulative SSM states cannot be safely stored as continuous memory-aware blocks, the engine automatically forces <code>--use-paged-cache</code> internally for these models.</li>
                    </ul>
                  </div>

                  <div>
                    <h3 className="text-base font-semibold text-foreground mb-2">KV Cache Quantization</h3>
                    <p className="text-muted-foreground leading-relaxed">
                      By converting stored prompts to q8 or q4 precision, you can reduce the cache's RAM footprint by 2-4x. <strong>This only safely compresses saved prefixes</strong>. The actual text generation continues to run at standard full precision natively in MLX.
                    </p>
                  </div>

                  <div>
                    <h3 className="text-base font-semibold tracking-tight text-foreground mb-2">Vision-Language (VL) Models</h3>
                    <p className="text-muted-foreground leading-relaxed">
                      The core engine handles Vision models automatically. <strong>Prefix caching works for images too!</strong> If you repeatedly ask questions about the exact same image (like in a tool-calling flow analyzing a dashboard), the massive vision embedding prefill is cached and reused instantly.
                    </p>
                  </div>
                </div>
              </Modal>
            )}
          </>
        )}
      </Section>

      {/* Paged Cache */}
      <Section title="Paged KV Cache" expanded={expandedSections.pagedCache} onToggle={() => toggleSection('pagedCache')} hidden={isImage}>
        {!effectivelyNoBatching && <PerformanceHint text="Reduces memory waste by splitting the KV cache into small blocks instead of one big chunk. Lets the server handle longer conversations without running out of RAM." />}
        {batchingOff && <IncompatWarning text="Paged cache requires continuous batching. Turn on 'Continuous Batching' in the Concurrent Processing section above to enable paged cache." />}
        {config.enableDiskCache && <IncompatWarning text="Paged cache and legacy Disk Cache cannot run simultaneously. Enabling paged cache will auto-disable legacy Disk Cache. For persistent caching with paged cache, use 'Block Disk Cache (L2)' below instead." />}
        {!batchingOff && prefixOff && <IncompatWarning text="Paged cache requires prefix cache. Enable 'Prefix Cache' above to use paged KV cache." />}
        <CheckField label="Use Paged KV Cache" tooltip="Manages the KV cache in fixed-size pages instead of contiguous memory. Greatly reduces memory fragmentation and allows serving larger batches or larger contexts on limited GPU RAM. Extremely recommended for long conversations." checked={config.usePagedCache} onChange={v => { onChange('usePagedCache', v); if (v && config.enableDiskCache) onChange('enableDiskCache', false) }} disabled={batchingOff || prefixOff} />
        {config.usePagedCache && (
          <>
            <SliderField
              label="Block Size (tokens)"
              tooltip="Number of tokens per paged KV cache block. Smaller blocks reduce memory waste per sequence but increase overhead from managing more blocks. Default 64 is optimal for most models. Increase to 128-256 for very long context scenarios."
              value={config.pagedCacheBlockSize}
              onChange={v => onChange('pagedCacheBlockSize', v)}
              min={1}
              max={1024}
              step={16}
              defaultValue={DEFAULT_CONFIG.pagedCacheBlockSize}
            />
            <SliderField
              label="Max Cache Blocks"
              tooltip="Maximum total number of KV cache blocks allocated. Total cache capacity = block_size x max_blocks tokens. Default 1000 blocks x 64 tokens = 64K tokens capacity. Increase for longer contexts, decrease to save memory."
              value={config.maxCacheBlocks}
              onChange={v => onChange('maxCacheBlocks', v)}
              min={1}
              max={100000}
              step={100}
              defaultValue={DEFAULT_CONFIG.maxCacheBlocks}
              allowUnlimited
              unlimitedValue={0}
              unlimitedLabel="Default (1000)"
            />
            <CheckField label="Block Disk Cache (L2)" tooltip="Persist individual paged cache blocks to SSD. When a block is evicted from RAM, it's saved to disk and can be reloaded later without recomputation. Dramatically speeds up cache warm-up for repeated system prompts and common prefixes. Uses content-addressable storage with background writes so disk I/O doesn't block inference." checked={config.enableBlockDiskCache} onChange={v => onChange('enableBlockDiskCache', v)} />
            {config.enableBlockDiskCache && (
              <>
                <SliderField
                  label="Block Cache Max (GB)"
                  tooltip="Maximum disk space for cached blocks. Oldest blocks are evicted when exceeded. Each block is small (~100KB-1MB), so 10GB can hold tens of thousands of blocks. Set to 0 for unlimited."
                  value={config.blockDiskCacheMaxGb}
                  onChange={v => onChange('blockDiskCacheMaxGb', v)}
                  min={0}
                  max={100}
                  step={1}
                  defaultValue={10}
                  allowUnlimited
                  unlimitedValue={0}
                  unlimitedLabel="Unlimited"
                />
                <div className="block">
                  <span className="text-xs font-medium text-muted-foreground">
                    Block Cache Directory
                    <Tooltip text="Directory for block-level disk cache files. A model-specific subdirectory is created automatically. Leave empty for default (~/.cache/vmlx-engine/block-cache/<model_hash>/)." />
                  </span>
                  <input
                    type="text"
                    value={config.blockDiskCacheDir || ''}
                    onChange={e => onChange('blockDiskCacheDir', e.target.value)}
                    placeholder="~/.cache/vmlx-engine/block-cache"
                    className="cfg-input text-xs"
                  />
                </div>
              </>
            )}
          </>
        )}
      </Section>

      {/* KV Cache Quantization */}
      <Section title="KV Cache Quantization" expanded={expandedSections.kvCacheQuant} onToggle={() => toggleSection('kvCacheQuant')} hidden={isImage}>
        {!effectivelyNoBatching && <PerformanceHint text="Compresses cached prompts to use less RAM. Only affects saved cache entries — your model's actual output quality stays the same. q8 is a safe default." />}
        {batchingOff && <IncompatWarning text="KV cache quantization requires continuous batching. Turn on 'Continuous Batching' in the Concurrent Processing section above." />}
        {!batchingOff && prefixOff && <IncompatWarning text="KV cache quantization requires prefix cache. Enable 'Prefix Cache' above to use KV cache quantization." />}
        {!effectivelyNoBatching && !prefixOff && isMambaCache && <PerformanceHint text="Hybrid model detected — KV cache quantization will only compress the attention layers. Non-attention layers (Mamba/GatedDeltaNet) are stored at full precision." />}
        <InfoNote text="KV cache quantization compresses entries stored in the prefix cache (completed prompts). It does NOT affect model weights or live generation KV cache, which always run at full precision. RAM savings apply only to cached prompt states." />
        <div className="block">
          <span className="text-xs font-medium text-muted-foreground">
            Quantization
            <Tooltip text="Compress KV states stored in the prefix cache to reduce cache memory by 2-4x. Only affects cached entries — generation always runs at full precision (no quality loss during inference). Requires prefix cache to be enabled. q8 (8-bit) is recommended. q4 (4-bit) saves more cache memory but may reduce reuse accuracy. Works with both LLMs and VLMs." />
          </span>
          <select value={config.kvCacheQuantization} onChange={e => onChange('kvCacheQuantization', e.target.value)} className="cfg-input" disabled={effectivelyNoBatching || prefixOff}>
            <option value="none">None (full precision cache)</option>
            <option value="q8">q8 (8-bit, ~2x cache savings)</option>
            <option value="q4">q4 (4-bit, ~4x cache savings)</option>
          </select>
        </div>
        {config.kvCacheQuantization !== 'none' && (
          <SliderField
            label="Group Size"
            tooltip="Number of elements quantized together. Smaller groups preserve more precision but use slightly more memory for scale/zero-point metadata. Default 64 is optimal for most models."
            value={config.kvCacheGroupSize}
            onChange={v => onChange('kvCacheGroupSize', v)}
            min={32}
            max={128}
            step={32}
            defaultValue={DEFAULT_CONFIG.kvCacheGroupSize}
          />
        )}
      </Section>

      {/* Disk Cache (L2 Persistent) */}
      <Section title="Disk Cache (Persistent)" expanded={expandedSections.diskCache} onToggle={() => toggleSection('diskCache')} hidden={isImage}>
        {!effectivelyNoBatching && <PerformanceHint text="Saves cached prompts to your SSD so they survive server restarts. Next time you load the same model, previous conversations warm up instantly." />}
        <InfoNote text="Legacy disk cache works with memory-aware prefix cache. Block disk cache (in the Paged KV Cache section) works with paged cache. Only one can be active at a time." />
        {batchingOff && <IncompatWarning text="Disk cache requires continuous batching. Turn on 'Continuous Batching' in the Concurrent Processing section above." />}
        {!effectivelyNoBatching && config.usePagedCache && <IncompatWarning text="Legacy disk cache is not compatible with paged cache. To use disk-based persistence with paged cache, use 'Block Disk Cache (L2)' in the Paged KV Cache section instead. To use this legacy disk cache, disable 'Use Paged KV Cache' first." />}
        {!batchingOff && prefixOff && <IncompatWarning text="Disk cache requires prefix cache. Enable 'Prefix Cache' above to use disk caching." />}
        <CheckField label="Enable Disk Cache" tooltip="Persist prompt caches to disk for reuse across server restarts. Acts as L2 cache behind the in-memory prefix cache — when a prompt isn't found in memory, it's loaded from disk instead of recomputing. Dramatically speeds up repeated prompts (system prompts, common prefixes). Requires prefix cache to be enabled. Note: not compatible with paged cache (uses different storage format)." checked={config.enableDiskCache} onChange={v => onChange('enableDiskCache', v)} disabled={effectivelyNoBatching || prefixOff || config.usePagedCache} />
        {config.enableDiskCache && (
          <>
            <SliderField
              label="Max Cache Size (GB)"
              tooltip="Maximum disk space for cached prompt states. Oldest entries are evicted when this limit is exceeded. Set to 0 for unlimited. Each cached prompt typically uses 50-500MB depending on model size and prompt length."
              value={config.diskCacheMaxGb}
              onChange={v => onChange('diskCacheMaxGb', v)}
              min={0}
              max={100}
              step={1}
              defaultValue={10}
              allowUnlimited
              unlimitedValue={0}
              unlimitedLabel="Unlimited"
            />
            <div className="block">
              <span className="text-xs font-medium text-muted-foreground">
                Cache Directory
                <Tooltip text="Base directory for disk cache files (.safetensors). A model-specific subdirectory is created automatically. Leave empty for the default location (~/.cache/vmlx-engine/prompt-cache/<model>/). Set a custom path if you want to use a specific drive." />
              </span>
              <input
                type="text"
                value={config.diskCacheDir || ''}
                onChange={e => onChange('diskCacheDir', e.target.value)}
                placeholder="~/.cache/vmlx-engine/prompt-cache"
                className="cfg-input text-xs"
              />
            </div>
          </>
        )}
      </Section>

      {/* Power Management — visible for ALL model types (text + image) */}
      <Section title="Power Management" expanded={expandedSections.power} onToggle={() => toggleSection('power')}>
        <PerformanceHint text="Control when idle models automatically sleep to free GPU memory. Sleeping models auto-wake when a new request arrives." />
        <Field label="Auto-Sleep" tooltip="Automatically put the model to sleep after a period of inactivity to free memory. Light sleep clears caches but keeps the model loaded (instant wake). Deep sleep unloads the model entirely (2-15s wake). Models auto-wake when a new request arrives.">
          <label className="flex items-center gap-2 cursor-pointer">
            <input
              type="checkbox"
              checked={config.autoSleepEnabled !== false}
              onChange={e => onChange('autoSleepEnabled', e.target.checked)}
              className="rounded border-input"
            />
            <span className="text-xs text-muted-foreground">
              Put model to sleep when idle (frees GPU memory)
            </span>
          </label>
        </Field>
        {config.autoSleepEnabled !== false && (
          <>
            <SliderField
              label="Light Sleep After"
              tooltip="Minutes of inactivity before entering light sleep. Light sleep clears KV/prefix caches to free memory but keeps the model loaded in GPU. Wake is instant — no reload needed. Set to 0 to disable light sleep."
              value={config.idleTimeoutSoftMin ?? (isImage ? 5 : 10)}
              onChange={v => onChange('idleTimeoutSoftMin', v)}
              min={0}
              max={120}
              step={1}
              defaultValue={isImage ? 5 : 10}
              allowUnlimited
              unlimitedValue={0}
              unlimitedLabel="Disabled"
            />
            <SliderField
              label="Deep Sleep After"
              tooltip="Minutes of inactivity before entering deep sleep. Deep sleep unloads the model entirely from GPU memory. The server process stays alive and the model auto-reloads when a new request arrives (2-15 seconds for most models). Set to 0 to disable deep sleep."
              value={config.idleTimeoutHardMin ?? (isImage ? 15 : 30)}
              onChange={v => onChange('idleTimeoutHardMin', v)}
              min={0}
              max={240}
              step={1}
              defaultValue={isImage ? 15 : 30}
              allowUnlimited
              unlimitedValue={0}
              unlimitedLabel="Disabled"
            />
          </>
        )}
      </Section>

      {/* Performance */}
      <Section title="Performance & Generation" expanded={expandedSections.performance} onToggle={() => toggleSection('performance')} hidden={isImage}>
        <PerformanceHint text="Controls how tokens stream to you and the max response length. For chat, keep stream interval at 1. Max tokens limits how long a single reply can be." />
        {/* JIT is not available for image models (mflux uses its own GPU pipeline). */}
        <Field label="JIT Compile (mx.compile)" tooltip="Enable Metal kernel fusion via mx.compile on the model forward pass. This optimizes GPU operations for faster inference after a one-time warmup on the first request. May not work with all models — falls back gracefully if compilation fails. Requires restart.">
          <label className="flex items-center gap-2 cursor-pointer">
            <input
              type="checkbox"
              checked={!!config.enableJit}
              onChange={e => onChange('enableJit', e.target.checked)}
              className="rounded border-input"
            />
            <span className="text-xs text-muted-foreground">
              Fuse Metal operations for faster inference (experimental)
            </span>
          </label>
        </Field>

        <SliderField
          label="Stream Interval"
          tooltip="Controls how often streaming tokens are sent to the client. A value of 1 sends each token immediately (smoothest streaming). Higher values batch multiple tokens together, which improves throughput but makes streaming feel chunkier. Set to 1 for chat use, higher for batch processing."
          value={config.streamInterval}
          onChange={v => onChange('streamInterval', v)}
          min={1}
          max={100}
          step={1}
          defaultValue={DEFAULT_CONFIG.streamInterval}
        />
        <SliderField
          label="Default Max Tokens"
          tooltip={`Maximum number of tokens the model can generate per request. This is the server default - individual API requests can override this.${detectedMaxContext ? ` Model context window: ${detectedMaxContext.toLocaleString()} tokens.` : ' Set based on your model\'s context window.'}`}
          value={config.maxTokens}
          onChange={v => onChange('maxTokens', v)}
          min={1}
          max={detectedMaxContext || 262144}
          step={1024}
          defaultValue={Math.min(DEFAULT_CONFIG.maxTokens, detectedMaxContext || DEFAULT_CONFIG.maxTokens)}
          maxInput={1000000}
          allowUnlimited
          unlimitedValue={0}
          unlimitedLabel="No limit"
        />
        <SliderField
          label="Default Temperature"
          tooltip="Server-wide default temperature for generation. Controls randomness: 0.0 = deterministic, 1.0 = creative. Overridden by per-request 'temperature' parameter. Set to 'Server default' to use vmlx-engine's built-in default (0.7)."
          value={config.defaultTemperature}
          onChange={v => onChange('defaultTemperature', v)}
          min={0}
          max={200}
          step={5}
          defaultValue={70}
          maxInput={200}
          allowUnlimited
          unlimitedValue={0}
          unlimitedLabel="Server default"
        />
        {config.defaultTemperature > 0 && (
          <InfoNote text={`Temperature: ${(config.defaultTemperature / 100).toFixed(2)} — ${config.defaultTemperature < 30 ? 'very focused' : config.defaultTemperature < 70 ? 'balanced' : config.defaultTemperature < 120 ? 'creative' : 'very random'}`} />
        )}
        <SliderField
          label="Default Top-P"
          tooltip="Server-wide default nucleus sampling threshold. Only considers tokens whose cumulative probability ≤ this value. 0.9 = use top 90% of probability mass. Lower = more focused, higher = more diverse. Overridden by per-request 'top_p'. Set to 'Server default' to use vmlx-engine's built-in default."
          value={config.defaultTopP}
          onChange={v => onChange('defaultTopP', v)}
          min={1}
          max={100}
          step={1}
          defaultValue={90}
          maxInput={100}
          allowUnlimited
          unlimitedValue={0}
          unlimitedLabel="Server default"
        />
        {config.defaultTopP > 0 && (
          <InfoNote text={`Top-P: ${(config.defaultTopP / 100).toFixed(2)}`} />
        )}
      </Section>

      {/* Tool Integration */}
      <Section title="Tool Integration (MCP)" expanded={expandedSections.tools} onToggle={() => toggleSection('tools')} hidden={isImage}>
        <PerformanceHint text="Lets the model call external tools (web search, code execution, etc.) during conversations. Requires a model that supports tool calling." />
        <Field label="MCP Config File" tooltip="Path to a JSON config file defining MCP (Model Context Protocol) tool servers. When configured, the model can call external tools during generation. The config file defines tool server endpoints, authentication, and available capabilities.">
          <input type="text" value={config.mcpConfig} onChange={e => onChange('mcpConfig', e.target.value)} placeholder="/path/to/mcp-config.json" className="cfg-input" />
        </Field>
        <CheckField label="Enable Auto Tool Choice" tooltip="When enabled, the model automatically decides when to call tools based on the conversation context. Requires a model that supports tool calling (Qwen, Llama 3+, Mistral, Gemma 3, Phi-4, Hermes, DeepSeek, GLM, Granite, Kimi, xLAM, Functionary, MiniMax, StepFun). The model will format tool calls according to the selected parser. Leave unchecked for auto-detection (recommended)." checked={config.enableAutoToolChoice ?? false} onChange={v => onChange('enableAutoToolChoice', v || undefined)} />
        {config.enableAutoToolChoice === undefined && (
          <InfoNote text="Auto-detect: most models enable this automatically when a tool parser is detected." />
        )}
        <ParserField
          label="Tool Call Parser"
          tooltip="Specifies how to parse the model's tool call output. Each model family uses a different format (Qwen, Llama, Mistral, Hermes, DeepSeek, GLM, etc). 'Auto-detect' reads config.json to pick the right one. If auto-detection fails (e.g. GGUF, renamed fine-tunes), select the parser matching your model's base architecture. Click '?' to see format examples and supported models for each parser."
          value={config.toolCallParser}
          onChange={v => onChange('toolCallParser', v)}
          options={TOOL_PARSER_OPTIONS}
        />
        <ParserField
          label="Reasoning Parser"
          tooltip="Separates reasoning/thinking from final content. Use Auto-detect unless it picks wrong. Qwen3 parser: Qwen, QwQ, MiniMax, StepFun (strict <think> tags). DeepSeek R1 parser: DeepSeek-R1, Gemma 3, GLM-4.7, Phi-4, Nemotron (lenient <think> tags). GPT-OSS parser: GLM-4.7 Flash, GPT-OSS only (Harmony protocol). Click '?' for full model list."
          value={config.reasoningParser}
          onChange={v => onChange('reasoningParser', v)}
          options={REASONING_PARSER_OPTIONS}
        />
        <SelectField
          label="Multimodal Support (VLM)"
          tooltip="Vision-Language Model mode for models like Qwen2-VL, Qwen3-VL, Pixtral, InternVL, or LLaVA. Auto: detected from config.json (vision_config presence). Force On: always use MLLM scheduler. Force Off: never use MLLM scheduler even if auto-detected."
          value={config.isMultimodal === true ? 'on' : config.isMultimodal === false ? 'off' : 'auto'}
          onChange={v => onChange('isMultimodal', v === 'on' ? true : v === 'off' ? false : undefined)}
          options={[
            { value: 'auto', label: 'Auto (detect from model)' },
            { value: 'on', label: 'Force On' },
            { value: 'off', label: 'Force Off' },
          ]}
        />
        {config.isMultimodal === true && (
          <InfoNote text="VLM mode forced ON — the MLLM scheduler handles image/video processing with full prefix cache, paged KV cache, and KV quantization support." />
        )}
        {config.isMultimodal === false && (
          <InfoNote text="VLM mode forced OFF — auto-detection is bypassed. Use this only if the model is incorrectly detected as multimodal." />
        )}
      </Section>

      {/* Speculative Decoding */}
      <Section title="Speculative Decoding" expanded={expandedSections.specDecode} onToggle={() => toggleSection('specDecode')} hidden={isImage}>
        <PerformanceHint text="Use a small draft model to propose tokens, then verify them in a single target model pass. Can give 20-90% speedup with zero quality loss." />
        {config.continuousBatching && <IncompatWarning text="Speculative decoding is incompatible with continuous batching. The draft model will only be used in SimpleEngine (non-batched) mode. Batched requests will use standard generation." />}
        {config.isMultimodal === true && <IncompatWarning text="Speculative decoding is incompatible with multimodal (VLM) models. The draft model will be ignored for VLM requests." />}
        <Field label="Draft Model" tooltip="Path or HuggingFace name of a small draft model. Must use the same tokenizer as the main model. Example: mlx-community/Llama-3.2-1B-Instruct-4bit for a Llama 3 target model. Leave empty to disable speculative decoding.">
          <input type="text" value={config.speculativeModel} onChange={e => onChange('speculativeModel', e.target.value)} placeholder="mlx-community/small-draft-model" className="cfg-input" />
        </Field>
        {config.speculativeModel && (
          <SliderField
            label="Draft Tokens per Step"
            tooltip="Number of tokens the draft model proposes per speculative decoding step. Higher values = more potential speedup but lower acceptance rate. Sweet spot is typically 2-5."
            value={config.numDraftTokens}
            onChange={v => onChange('numDraftTokens', v)}
            min={1}
            max={20}
            step={1}
            defaultValue={DEFAULT_CONFIG.numDraftTokens}
          />
        )}
      </Section>

      {/* Embedding Model */}
      {!isImage && (
      <div className="mb-2">
        <Field label="Embedding Model" tooltip="Pre-load a separate embedding model at startup for the /v1/embeddings endpoint. Runs alongside the main chat model. Example: mlx-community/embeddinggemma-300m-6bit. Leave empty to disable embeddings endpoint.">
          <input type="text" value={config.embeddingModel} onChange={e => onChange('embeddingModel', e.target.value)} placeholder="Optional — mlx-community/embedding-model" className="cfg-input" />
        </Field>
      </div>
      )}

      {/* Additional */}
      <div className="mb-4">
        <Field label="Additional Arguments" tooltip="Raw command-line arguments appended to the serve command. Use this for flags not exposed in the UI above. Example: --log-level DEBUG. Arguments are split by whitespace and passed directly to the CLI.">
          <input type="text" value={config.additionalArgs} onChange={e => onChange('additionalArgs', e.target.value)} placeholder="--custom-flag value" className="cfg-input" />
        </Field>
      </div>

      {/* Reset to Defaults */}
      {onReset && (
        <div className="pt-2 pb-1 border-t border-border">
          <button
            onClick={onReset}
            className="text-xs text-muted-foreground hover:text-foreground transition-colors"
          >
            Reset all parameters to defaults
          </button>
        </div>
      )}
    </div>
  )
}

// ─── Shared Helper Components ─────────────────────────────────────────────────

export function Tooltip({ text }: { text: string }) {
  const [show, setShow] = useState(false)
  const [pinned, setPinned] = useState(false)
  const [above, setAbove] = useState(true)
  const triggerRef = useRef<HTMLSpanElement>(null)

  const updatePosition = () => {
    if (triggerRef.current) {
      const rect = triggerRef.current.getBoundingClientRect()
      setAbove(rect.top > 130)
    }
  }

  const handleClick = (e: React.MouseEvent) => {
    e.stopPropagation()
    updatePosition()
    const willPin = !pinned
    setPinned(willPin)
    setShow(willPin)
  }

  const handleEnter = () => {
    if (!pinned) {
      updatePosition()
      setShow(true)
    }
  }

  const handleLeave = () => {
    if (!pinned) setShow(false)
  }

  return (
    <span className="relative inline-flex ml-1">
      <span
        ref={triggerRef}
        className={`inline-flex items-center justify-center w-3.5 h-3.5 rounded-full text-[10px] font-bold cursor-help select-none ${pinned ? 'bg-primary text-primary-foreground' : 'bg-muted text-muted-foreground'}`}
        onClick={handleClick}
        onMouseEnter={handleEnter}
        onMouseLeave={handleLeave}
      >
        ?
      </span>
      {show && (
        <div
          className={`absolute left-1/2 -translate-x-1/2 w-72 p-2.5 bg-popover text-popover-foreground text-xs rounded-lg shadow-lg border border-border z-50 leading-relaxed ${above ? 'bottom-full mb-2' : 'top-full mt-2'
            }`}
        >
          {text}
          <div className={`absolute left-1/2 -translate-x-1/2 border-4 border-transparent ${above ? 'top-full -mt-px border-t-border' : 'bottom-full -mb-px border-b-border'
            }`} />
        </div>
      )}
    </span>
  )
}

// ─── Parser Options with Format Examples ──────────────────────────────────────

interface ParserOption {
  value: string
  label: string
  format?: string  // Example of the format for tooltip
  models?: string[]  // Specific models that use this parser (shown in help panel)
}

const TOOL_PARSER_OPTIONS: ParserOption[] = [
  { value: 'auto', label: 'Auto-detect (recommended)' },
  { value: '', label: 'None (disable tool parsing)' },
  {
    value: 'qwen', label: 'Qwen — Qwen3.5 / Qwen3 / Qwen2.5 / QwQ', format: '<tool_call>{"name":"fn","arguments":{...}}</tool_call>', models: [
      'Qwen3.5-VL (0.8B\u2013122B MoE, native vision)', 'Qwen3 (0.6B\u2013235B)', 'Qwen3-Coder',
      'Qwen3-MoE (22B/57B)', 'Qwen3-VL (2B/32B/72B)', 'QwQ-32B',
      'Qwen2.5 (0.5B\u201372B)', 'Qwen2.5-Coder (0.5B\u201332B)',
      'Qwen2.5-VL (3B\u201372B)', 'Qwen2 (0.5B\u201372B)', 'Qwen2-VL (2B\u201372B)',
    ]
  },
  {
    value: 'llama', label: 'Llama — Llama 4 / 3.x / Yi', format: '<function=name>{"arg":"val"}</function>', models: [
      'Llama 4 Scout (17Bx16E MoE)', 'Llama 4 Maverick (17Bx128E MoE)',
      'Llama 3.3 (70B)', 'Llama 3.2 (1B/3B/11B/90B)', 'Llama 3.1 (8B/70B/405B)', 'Llama 3 (8B/70B)',
      'Yi / Yi-1.5 (Llama architecture)',
    ]
  },
  {
    value: 'mistral', label: 'Mistral — Mistral / Mixtral / Pixtral / Codestral', format: '[TOOL_CALLS][{"name":"fn","arguments":{...}}]', models: [
      'Mistral Large (123B)', 'Mistral Small 3.1 (24B)', 'Mistral Nemo (12B)', 'Mistral 7B v0.3',
      'Mixtral 8x7B / 8x22B', 'Pixtral 12B / Pixtral Large', 'Codestral (22B)', 'Devstral Small (24B)',
    ]
  },
  {
    value: 'hermes', label: 'Hermes — Gemma 3 / Phi-4 / Hermes fine-tunes', format: '<tool_call>{"name":"fn","arguments":{...}}</tool_call>', models: [
      'Gemma 3 (1B/4B/12B/27B)', 'Gemma 3n (E2B/E4B)', 'Phi-4 Mini (3.8B)', 'Phi-4 Medium (14B)',
      'Phi-4 Reasoning (14B)', 'Hermes 2 / 3 / 4', 'Any Hermes-format fine-tune',
    ]
  },
  {
    value: 'deepseek', label: 'DeepSeek — V2/V2.5/V3/R1 (native arch only)', format: '\u{ff5c}<tool_call>name\n{"arg":"val"}</tool_call>\u{ff5c}', models: [
      'DeepSeek-V3 (671B MoE)', 'DeepSeek-V2.5 (236B MoE)', 'DeepSeek-V2 (236B MoE)',
      'DeepSeek-R1 (671B native)', 'DeepSeek-Coder-V2 (236B)',
      '\u26A0 R1-Distill-Qwen/Llama use qwen/llama parsers',
    ]
  },
  {
    value: 'nemotron', label: 'Nemotron — Nemotron / Qwen3-Next', format: '<tool_call><function=fn><parameter=p>val</parameter></function></tool_call>', models: [
      'Nemotron-H (8B/47B/56B)', 'Nemotron-4 Nano/Super/Ultra',
      'Qwen3-Next / Qwen3-Coder-Next (hybrid Mamba)',
      '\u26A0 Llama/Qwen fine-tunes named "Nemotron" use their base parser',
    ]
  },
  {
    value: 'glm47', label: 'GLM / GPT-OSS — GLM-4 / GLM-4.7 / GLM-Z1', format: '<tool_call>name\n<arg_key>k</arg_key><arg_value>v</arg_value></tool_call>', models: [
      'GLM-4 (9B)', 'GLM-4.7 (9B)', 'GLM-4.7 Flash (9B MoE)', 'GLM-Z1 (32B)', 'GPT-OSS-20B/120B',
    ]
  },
  {
    value: 'granite', label: 'Granite — IBM Granite 3.x / Granite-Code', format: '<|tool_call|>[{"name":"fn","arguments":{...}}]', models: [
      'Granite 3.0/3.1/3.2/3.3 (2B/8B)', 'Granite-Code (3B/8B/20B/34B)',
    ]
  },
  {
    value: 'functionary', label: 'Functionary — MeetKai Functionary v2/v3/v4r', format: '<|from|>assistant\n<|recipient|>fn\n<|content|>{"arg":"val"}', models: [
      'Functionary v2 (7B)', 'Functionary v3 (8B/70B)', 'Functionary v4r (8B)',
    ]
  },
  {
    value: 'minimax', label: 'MiniMax — MiniMax-M1 / M2 / M2.5', format: '<minimax:tool_call><invoke name="fn"><parameter name="arg">val</parameter></invoke></minimax:tool_call>', models: [
      'MiniMax-M1 (40B MoE)', 'MiniMax-M2 (230B MoE)', 'MiniMax-M2.5 (230B MoE)',
    ]
  },
  {
    value: 'xlam', label: 'xLAM — Salesforce xLAM-v2 series', format: '[{"name":"fn","arguments":{...}}]', models: [
      'xLAM-1B', 'xLAM-7B', 'xLAM-v2 (8x7B/8x22B)',
    ]
  },
  {
    value: 'kimi', label: 'Kimi — Kimi-K2 / Moonshot', format: '<|tool_calls_section_begin|><|tool_call_begin|>fn<|tool_call_argument_begin|>{...}<|tool_call_end|>', models: [
      'Kimi-K2 (1T MoE)', 'Kimi-K2.5', 'Moonshot-v1',
    ]
  },
  {
    value: 'step3p5', label: 'StepFun — Step-3.5 Flash / Step-3.5', format: '<tool_call><function=fn><parameter=arg>val</parameter></function></tool_call>', models: [
      'Step-3.5 Flash (8B MoE)', 'Step-3.5',
    ]
  },
]

const REASONING_PARSER_OPTIONS: ParserOption[] = [
  { value: 'auto', label: 'Auto-detect (recommended)' },
  { value: '', label: 'None (disable reasoning extraction)' },
  {
    value: 'qwen3', label: 'Qwen3 — Qwen / QwQ / MiniMax / StepFun', format: '<think>...reasoning...</think>content  (strict: both tags required)', models: [
      'Qwen3.5-VL (0.8B\u2013122B MoE, vision+reasoning)', 'Qwen3 (0.6B\u2013235B, all sizes)',
      'Qwen3-Coder (all sizes)', 'Qwen3-MoE (22B/57B)', 'QwQ-32B',
      'MiniMax-M2 (46B)', 'MiniMax-M2.5 (172B MoE)',
      'MiniMax Prism Pro (80B)', 'StepFun Step-3.5 Flash (8B MoE)',
      'StepFun Step-3.5', 'StepFun Step-1V (vision)',
    ]
  },
  {
    value: 'deepseek_r1', label: 'DeepSeek R1 — DeepSeek / Gemma / GLM / Phi / Nemotron', format: '<think>...reasoning...</think>content  (lenient: handles missing <think>)', models: [
      'DeepSeek-R1 (671B native)', 'DeepSeek-R1-0528',
      'Gemma 3 (1B/4B/12B/27B, thinking mode)',
      'GLM-4.7 (9B) \u2014 NOT GLM-4.7 Flash', 'GLM-Z1 (32B)',
      'Phi-4 Reasoning / Reasoning Plus (14B)',
      'Nemotron (hybrid Mamba+attention)',
      '\u26A0 R1-Distill-Qwen/Llama: must select manually (auto-detect has no reasoning)',
    ]
  },
  {
    value: 'openai_gptoss', label: 'GPT-OSS / Harmony — GLM-4.7 Flash / GPT-OSS', format: '<|channel|>analysis<|message|>reasoning...<|channel|>final<|message|>content', models: [
      'GLM-4.7 Flash (9B MoE) \u2014 uses Harmony, NOT deepseek_r1',
      'GPT-OSS-20B', 'GPT-OSS-120B',
    ]
  },
]

function ParserField({ label, tooltip, value, onChange, options }: {
  label: string; tooltip: string; value: string; onChange: (v: string) => void; options: ParserOption[]
}) {
  const [showHelp, setShowHelp] = useState(false)
  const selected = options.find(o => o.value === value)
  // Show help panel when explicitly toggled OR when a non-auto parser is manually selected
  const helpVisible = showHelp || (value !== 'auto' && value !== '')

  return (
    <div className="block">
      <span className="text-xs font-medium text-muted-foreground">
        {label}
        <Tooltip text={tooltip} />
        <button
          type="button"
          onClick={() => setShowHelp(!showHelp)}
          className="ml-1 inline-flex items-center justify-center w-3.5 h-3.5 rounded-full bg-muted text-muted-foreground text-[10px] font-bold cursor-help select-none hover:bg-accent"
          title="Show model compatibility reference"
        >
          ?
        </button>
      </span>
      <select value={value} onChange={e => onChange(e.target.value)} className="cfg-input">
        {options.map(o => (
          <option key={o.value} value={o.value}>{o.label}</option>
        ))}
      </select>
      {helpVisible && (
        <div className="mt-1.5 bg-background border border-border rounded p-2 text-xs max-h-48 overflow-auto space-y-2">
          {options.filter(o => o.format).map(o => {
            const isSelected = o.value === value
            return (
              <div key={o.value} className={`pl-1.5 border-l-2 ${isSelected ? 'border-primary' : 'border-transparent'}`}>
                <div className={`font-medium leading-snug ${isSelected ? 'text-primary' : 'text-foreground'}`}>
                  {o.label}
                </div>
                <code className="block mt-0.5 text-[10px] bg-muted text-muted-foreground px-1.5 py-0.5 rounded break-all leading-snug">
                  {o.format}
                </code>
                {o.models && o.models.length > 0 && (
                  <div className="mt-1 flex flex-wrap gap-1">
                    {o.models.map((m, i) => (
                      <span key={i} className={`inline-block text-[10px] px-1.5 py-px rounded-sm leading-tight ${m.startsWith('\u26A0') ? 'bg-warning/15 text-warning border border-warning/30' : 'bg-muted text-muted-foreground'
                        }`}>{m}</span>
                    ))}
                  </div>
                )}
              </div>
            )
          })}
          <div className="pt-1 border-t border-border text-[10px] text-muted-foreground/70 italic leading-snug">
            Fine-tunes inherit the base model&apos;s parser. A Llama fine-tune uses llama, a Qwen fine-tune uses qwen, regardless of its marketing name. When auto-detect fails, select the parser matching the base architecture.
          </div>
        </div>
      )}
      {selected?.format && !helpVisible && (
        <p className="text-[10px] text-muted-foreground mt-0.5 font-mono truncate" title={selected.format}>
          {selected.format}
        </p>
      )}
    </div>
  )
}

function IncompatWarning({ text }: { text: string }) {
  return (
    <div className="px-2 py-1.5 mb-1 rounded text-[11px] bg-warning/10 border border-warning/30 text-warning leading-tight">
      {text}
    </div>
  )
}

function InfoNote({ text }: { text: string }) {
  return (
    <div className="px-2 py-1.5 mb-1 rounded text-[11px] bg-primary/10 border border-primary/30 text-primary leading-tight">
      {text}
    </div>
  )
}

function PerformanceHint({ text }: { text: string }) {
  return (
    <div className="px-2 py-1.5 mb-1 rounded text-[11px] text-muted-foreground/70 italic leading-tight">
      {text}
    </div>
  )
}

export function Section({ title, expanded, onToggle, children, hidden }: {
  title: string; expanded: boolean; onToggle: () => void; children: React.ReactNode; hidden?: boolean
}) {
  if (hidden) return null
  return (
    <div className="mb-3 border border-border rounded">
      <button onClick={onToggle} className="w-full flex items-center gap-2 px-3 py-2 text-sm font-medium hover:bg-accent rounded-t">
        <span className={`transition-transform ${expanded ? 'rotate-90' : ''}`}>&#9654;</span>
        {title}
      </button>
      {expanded && <div className="px-3 pb-3 space-y-3">{children}</div>}
    </div>
  )
}

export function Field({ label, tooltip, children }: { label: string; tooltip?: string; children: React.ReactNode }) {
  return (
    <label className="block">
      <span className="text-xs font-medium text-muted-foreground">
        {label}
        {tooltip && <Tooltip text={tooltip} />}
      </span>
      {children}
    </label>
  )
}

export function CheckField({ label, tooltip, checked, onChange, disabled }: {
  label: string; tooltip?: string; checked: boolean; onChange: (v: boolean) => void; disabled?: boolean
}) {
  return (
    <label className={`flex items-center gap-2 ${disabled ? 'opacity-50 cursor-not-allowed' : 'cursor-pointer'}`}>
      <input type="checkbox" checked={checked} onChange={e => onChange(e.target.checked)} disabled={disabled} />
      <span className="text-sm">{label}</span>
      {tooltip && <Tooltip text={tooltip} />}
    </label>
  )
}

export function SelectField({ label, tooltip, value, onChange, options, disabled }: {
  label: string; tooltip?: string; value: string; onChange: (v: string) => void
  options: { value: string; label: string }[]; disabled?: boolean
}) {
  return (
    <Field label={label} tooltip={tooltip}>
      <select value={value} onChange={e => onChange(e.target.value)} disabled={disabled} className="cfg-input">
        {options.map(o => <option key={o.value} value={o.value}>{o.label}</option>)}
      </select>
    </Field>
  )
}

interface SliderFieldProps {
  label: string
  tooltip?: string
  value: number
  onChange: (v: number) => void
  min: number
  max: number
  step: number
  defaultValue: number
  allowUnlimited?: boolean
  unlimitedValue?: number
  unlimitedLabel?: string
  disabled?: boolean
  /** Hard upper limit for number input (prevents server crash from out-of-range values) */
  maxInput?: number
}

export function SliderField({
  label, tooltip, value, onChange, min, max, step, defaultValue,
  allowUnlimited = false, unlimitedValue = 0, unlimitedLabel = 'Unlimited',
  disabled = false, maxInput
}: SliderFieldProps) {
  const isUnlimited = allowUnlimited && value === unlimitedValue
  // Local string state for the number input so typing isn't clamped mid-keystroke.
  // Without this, min=1024 causes typing "1" to immediately snap to 1024.
  const [localInput, setLocalInput] = useState<string | null>(null)

  const handleSliderChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    onChange(Number(e.target.value))
  }

  const handleInputChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    // Store raw typed value locally — no clamping until blur
    setLocalInput(e.target.value)
  }

  const handleInputFocus = () => {
    // Initialize local state with current value when focus starts
    setLocalInput(isUnlimited ? '' : String(value))
  }

  const handleInputBlur = () => {
    const raw = localInput ?? ''
    setLocalInput(null)
    if (raw === '') {
      onChange(defaultValue)
      return
    }
    const num = Math.round(Number(raw))
    if (isNaN(num)) {
      onChange(defaultValue)
    } else {
      // Clamp to valid range — maxInput enforces hard server-side limits
      const clamped = maxInput != null ? Math.min(maxInput, Math.max(min, num)) : Math.max(min, num)
      onChange(clamped)
    }
  }

  const toggleUnlimited = () => {
    if (isUnlimited) {
      onChange(defaultValue)
    } else {
      onChange(unlimitedValue)
    }
  }

  // Clamp slider display value to range (for when input allows beyond max)
  const sliderValue = isUnlimited ? min : Math.min(Math.max(value, min), max)
  // Show local input while editing, parent value otherwise
  const displayValue = localInput !== null ? localInput : (isUnlimited ? '' : value)

  return (
    <div className={`block ${disabled ? 'opacity-50 pointer-events-none' : ''}`}>
      <div className="flex items-center justify-between">
        <span className="text-xs font-medium text-muted-foreground">
          {label}
          {tooltip && <Tooltip text={tooltip} />}
        </span>
        {allowUnlimited && (
          <button
            type="button"
            onClick={toggleUnlimited}
            disabled={disabled}
            className={`text-[10px] px-1.5 py-0.5 rounded border transition-colors ${isUnlimited
              ? 'bg-primary/15 border-primary/40 text-primary'
              : 'border-border text-muted-foreground hover:text-foreground hover:border-foreground/30'
              }`}
          >
            {unlimitedLabel}
          </button>
        )}
      </div>
      <div className="flex items-center gap-2 mt-1">
        <input
          type="range"
          className="cfg-slider flex-1"
          min={min}
          max={max}
          step={step}
          value={sliderValue}
          onChange={handleSliderChange}
          disabled={disabled || isUnlimited}
        />
        <input
          type="number"
          className="w-20 px-2 py-1 bg-background border border-input rounded text-sm text-right tabular-nums"
          value={displayValue}
          onChange={handleInputChange}
          onFocus={handleInputFocus}
          onBlur={handleInputBlur}
          placeholder={isUnlimited ? unlimitedLabel : undefined}
          disabled={disabled || isUnlimited}
          min={min}
          step={step}
        />
      </div>
    </div>
  )
}
