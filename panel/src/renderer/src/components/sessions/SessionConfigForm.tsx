import { useState, useRef } from 'react'

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
  enableAutoToolChoice: boolean
  toolCallParser: string
  reasoningParser: string
  additionalArgs: string
}

export const DEFAULT_CONFIG: SessionConfig = {
  host: '127.0.0.1',
  port: 8000,
  apiKey: '',
  rateLimit: 0,
  timeout: 300,
  maxNumSeqs: 256,
  prefillBatchSize: 512,
  completionBatchSize: 512,
  continuousBatching: true,
  enablePrefixCache: true,
  prefixCacheSize: 100,
  cacheMemoryMb: 0,
  cacheMemoryPercent: 20,
  noMemoryAwareCache: false,
  usePagedCache: true,
  pagedCacheBlockSize: 64,
  maxCacheBlocks: 1000,
  kvCacheQuantization: 'none',
  kvCacheGroupSize: 64,
  enableDiskCache: false,
  diskCacheMaxGb: 10,
  diskCacheDir: '',
  enableBlockDiskCache: false,
  blockDiskCacheMaxGb: 10,
  blockDiskCacheDir: '',
  streamInterval: 1,
  maxTokens: 32768,
  mcpConfig: '',
  enableAutoToolChoice: false,
  toolCallParser: 'auto',
  reasoningParser: 'auto',
  additionalArgs: ''
}

interface SessionConfigFormProps {
  config: SessionConfig
  onChange: <K extends keyof SessionConfig>(key: K, value: SessionConfig[K]) => void
  onReset?: () => void
  /** Detected model cache type ('kv', 'mamba', etc.) for feature gating */
  detectedCacheType?: string
}

export function SessionConfigForm({ config, onChange, onReset, detectedCacheType }: SessionConfigFormProps) {
  const [expandedSections, setExpandedSections] = useState({
    server: true,
    concurrent: false,
    prefixCache: false,
    pagedCache: false,
    kvCacheQuant: false,
    diskCache: false,
    performance: false,
    tools: false
  })

  const toggleSection = (section: keyof typeof expandedSections) => {
    setExpandedSections(prev => ({ ...prev, [section]: !prev[section] }))
  }

  // Feature gating: cache features require continuous batching
  const batchingOff = !config.continuousBatching
  // Feature gating: KV cache quantization doesn't work with Mamba/SSM cache types
  const isMambaCache = detectedCacheType === 'mamba'

  return (
    <div className="space-y-0">
      {/* Server Settings */}
      <Section title="Server Settings" expanded={expandedSections.server} onToggle={() => toggleSection('server')}>
        <Field label="Host" tooltip="The network interface to bind to. Use 127.0.0.1 (localhost) to only accept local connections, or 0.0.0.0 to accept connections from other machines on your network. For most local use, 127.0.0.1 is recommended.">
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
        <SliderField
          label="Rate Limit (req/min)"
          tooltip="Maximum number of API requests allowed per minute. Set to 0 to disable rate limiting. Useful when exposing the server to multiple users or external applications to prevent overloading."
          value={config.rateLimit}
          onChange={v => onChange('rateLimit', v)}
          min={0}
          max={1000}
          step={10}
          defaultValue={DEFAULT_CONFIG.rateLimit}
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
      </Section>

      {/* Concurrent Processing */}
      <Section title="Concurrent Processing" expanded={expandedSections.concurrent} onToggle={() => toggleSection('concurrent')}>
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
          unlimitedLabel="No limit"
        />
        <SliderField
          label="Prefill Batch Size"
          tooltip="Maximum number of tokens to process in a single prefill (prompt processing) step. Larger batches use more memory but process prompts faster. Reduce if you're running out of memory during prompt processing. Default 512 works well for most models."
          value={config.prefillBatchSize}
          onChange={v => onChange('prefillBatchSize', v)}
          min={1}
          max={4096}
          step={64}
          defaultValue={DEFAULT_CONFIG.prefillBatchSize}
          allowUnlimited
          unlimitedValue={0}
          unlimitedLabel="No limit"
        />
        <SliderField
          label="Completion Batch Size"
          tooltip="Maximum number of tokens to generate in a single completion (token generation) step. Similar to prefill batch size but for the generation phase. Larger values can improve throughput for multi-user scenarios."
          value={config.completionBatchSize}
          onChange={v => onChange('completionBatchSize', v)}
          min={1}
          max={4096}
          step={64}
          defaultValue={DEFAULT_CONFIG.completionBatchSize}
          allowUnlimited
          unlimitedValue={0}
          unlimitedLabel="No limit"
        />
        <CheckField label="Continuous Batching" tooltip="When enabled, new requests can join an ongoing batch without waiting for the current batch to complete. This significantly improves throughput when serving multiple concurrent users. Has minimal overhead for single-user use." checked={config.continuousBatching} onChange={v => onChange('continuousBatching', v)} />
        {!config.continuousBatching && config.enablePrefixCache && (
          <InfoNote text="Continuous batching will be auto-enabled at launch because prefix cache is on." />
        )}
      </Section>

      {/* Prefix Cache */}
      <Section title="Prefix Cache" expanded={expandedSections.prefixCache} onToggle={() => toggleSection('prefixCache')}>
        {batchingOff && <IncompatWarning text="Prefix cache requires continuous batching to be enabled." />}
        <CheckField label="Enable Prefix Caching" tooltip="Caches computed attention states for common prompt prefixes (like system prompts). When multiple requests share the same prefix, the cached state is reused, dramatically reducing time-to-first-token. Recommended to keep enabled." checked={config.enablePrefixCache} onChange={v => onChange('enablePrefixCache', v)} disabled={batchingOff} />
        {config.enablePrefixCache && (
          <>
            <CheckField label="Legacy Entry-Count Cache" tooltip="Switches from the default memory-aware cache to a simpler cache that limits by number of entries rather than memory usage. Only use this if you're experiencing issues with the memory-aware cache or need deterministic cache eviction behavior." checked={config.noMemoryAwareCache} onChange={v => onChange('noMemoryAwareCache', v)} />
            {config.noMemoryAwareCache ? (
              <SliderField
                label="Max Cache Entries"
                tooltip="Maximum number of prefix cache entries to store when using legacy entry-count mode. Each entry stores the KV cache for one unique prefix. Higher values cache more prefixes but use more memory."
                value={config.prefixCacheSize}
                onChange={v => onChange('prefixCacheSize', v)}
                min={1}
                max={10000}
                step={10}
                defaultValue={DEFAULT_CONFIG.prefixCacheSize}
                allowUnlimited
                unlimitedValue={0}
                unlimitedLabel="No limit"
              />
            ) : (
              <>
                <SliderField
                  label="Cache Memory Limit (MB)"
                  tooltip="Hard limit on memory used by the prefix cache in megabytes. Set to 0 to let the system auto-detect based on available RAM and the percentage setting below. Set an explicit value if you need to reserve memory for other applications."
                  value={config.cacheMemoryMb}
                  onChange={v => onChange('cacheMemoryMb', v)}
                  min={0}
                  max={65536}
                  step={256}
                  defaultValue={DEFAULT_CONFIG.cacheMemoryMb}
                  allowUnlimited
                  unlimitedValue={0}
                  unlimitedLabel="Auto-detect"
                />
                <SliderField
                  label="Cache Memory %"
                  tooltip="Percentage of available system RAM to allocate for the prefix cache. Only used when Cache Memory Limit is 0 (auto-detect). Default 20% is a good balance. Increase for workloads with many shared prefixes, decrease if running other memory-intensive apps."
                  value={config.cacheMemoryPercent}
                  onChange={v => onChange('cacheMemoryPercent', v)}
                  min={1}
                  max={100}
                  step={1}
                  defaultValue={DEFAULT_CONFIG.cacheMemoryPercent}
                />
              </>
            )}
          </>
        )}
      </Section>

      {/* Paged Cache */}
      <Section title="Paged KV Cache" expanded={expandedSections.pagedCache} onToggle={() => toggleSection('pagedCache')}>
        {batchingOff && <IncompatWarning text="Paged cache requires continuous batching to be enabled." />}
        {config.enableDiskCache && <IncompatWarning text="Paged cache is not compatible with disk cache. Enabling paged cache will disable disk cache." />}
        <CheckField label="Use Paged KV Cache" tooltip="Enables paged attention, which allocates KV cache memory in fixed-size blocks instead of one large contiguous allocation. This reduces memory fragmentation and allows the model to handle longer contexts with less total memory. Recommended for models with long context windows." checked={config.usePagedCache} onChange={v => { onChange('usePagedCache', v); if (v && config.enableDiskCache) onChange('enableDiskCache', false) }} disabled={batchingOff} />
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
              unlimitedLabel="No limit"
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
                    <Tooltip text="Directory for block-level disk cache files. A model-specific subdirectory is created automatically. Leave empty for default (~/.cache/vllm-mlx/block-cache/<model_hash>/)." />
                  </span>
                  <input
                    type="text"
                    value={config.blockDiskCacheDir || ''}
                    onChange={e => onChange('blockDiskCacheDir', e.target.value)}
                    placeholder="~/.cache/vllm-mlx/block-cache"
                    className="cfg-input text-xs"
                  />
                </div>
              </>
            )}
          </>
        )}
      </Section>

      {/* KV Cache Quantization */}
      <Section title="KV Cache Quantization" expanded={expandedSections.kvCacheQuant} onToggle={() => toggleSection('kvCacheQuant')}>
        {batchingOff && <IncompatWarning text="KV cache quantization requires continuous batching to be enabled." />}
        {isMambaCache && !batchingOff && <IncompatWarning text="KV cache quantization is not supported for Mamba/SSM models." />}
        <div className="block">
          <span className="text-xs font-medium text-muted-foreground">
            Quantization
            <Tooltip text="Compress KV states stored in the prefix cache to reduce cache memory by 2-4x. Only affects cached entries — generation always runs at full precision (no quality loss during inference). Requires prefix cache to be enabled. q8 (8-bit) is recommended. q4 (4-bit) saves more cache memory but may reduce reuse accuracy." />
          </span>
          <select value={config.kvCacheQuantization} onChange={e => onChange('kvCacheQuantization', e.target.value)} className="cfg-input" disabled={batchingOff || isMambaCache}>
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
      <Section title="Disk Cache (Persistent)" expanded={expandedSections.diskCache} onToggle={() => toggleSection('diskCache')}>
        {batchingOff && <IncompatWarning text="Disk cache requires continuous batching to be enabled." />}
        {config.usePagedCache && <IncompatWarning text="Disk cache is not compatible with paged cache. Disable paged cache first, or disk cache will be ignored." />}
        <CheckField label="Enable Disk Cache" tooltip="Persist prompt caches to disk for reuse across server restarts. Acts as L2 cache behind the in-memory prefix cache — when a prompt isn't found in memory, it's loaded from disk instead of recomputing. Dramatically speeds up repeated prompts (system prompts, common prefixes). Requires prefix cache to be enabled. Note: not compatible with paged cache (uses different storage format)." checked={config.enableDiskCache} onChange={v => onChange('enableDiskCache', v)} disabled={batchingOff || config.usePagedCache} />
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
                <Tooltip text="Base directory for disk cache files (.safetensors). A model-specific subdirectory is created automatically. Leave empty for the default location (~/.cache/vllm-mlx/prompt-cache/<model>/). Set a custom path if you want to use a specific drive." />
              </span>
              <input
                type="text"
                value={config.diskCacheDir || ''}
                onChange={e => onChange('diskCacheDir', e.target.value)}
                placeholder="~/.cache/vllm-mlx/prompt-cache"
                className="cfg-input text-xs"
              />
            </div>
          </>
        )}
      </Section>

      {/* Performance */}
      <Section title="Performance & Generation" expanded={expandedSections.performance} onToggle={() => toggleSection('performance')}>
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
          tooltip="Maximum number of tokens the model can generate per request. This is the server default - individual API requests can override this. Set based on your model's context window. For a 128K context model, 32768 is a reasonable default."
          value={config.maxTokens}
          onChange={v => onChange('maxTokens', v)}
          min={1}
          max={262144}
          step={1024}
          defaultValue={DEFAULT_CONFIG.maxTokens}
          allowUnlimited
          unlimitedValue={0}
          unlimitedLabel="No limit"
        />
      </Section>

      {/* Tool Integration */}
      <Section title="Tool Integration (MCP)" expanded={expandedSections.tools} onToggle={() => toggleSection('tools')}>
        <Field label="MCP Config File" tooltip="Path to a JSON config file defining MCP (Model Context Protocol) tool servers. When configured, the model can call external tools during generation. The config file defines tool server endpoints, authentication, and available capabilities.">
          <input type="text" value={config.mcpConfig} onChange={e => onChange('mcpConfig', e.target.value)} placeholder="/path/to/mcp-config.json" className="cfg-input" />
        </Field>
        <CheckField label="Enable Auto Tool Choice" tooltip="When enabled, the model automatically decides when to call tools based on the conversation context. Requires a model that supports tool calling (Qwen, Llama 3+, Mistral, Gemma 3, Phi-4, Hermes, DeepSeek, GLM, Granite, Kimi, xLAM, Functionary, MiniMax, StepFun). The model will format tool calls according to the selected parser." checked={config.enableAutoToolChoice} onChange={v => onChange('enableAutoToolChoice', v)} />
        <ParserField
          label="Tool Call Parser"
          tooltip="Specifies how to parse the model's tool call output. Each model family uses a different format (Qwen, Llama, Mistral, Hermes, DeepSeek, GLM, etc). 'Auto-detect' reads config.json to pick the right one. If auto-detection fails (e.g. GGUF, renamed fine-tunes), select the parser matching your model's base architecture. Click '?' to see format examples and supported models for each parser."
          value={config.toolCallParser}
          onChange={v => onChange('toolCallParser', v)}
          options={TOOL_PARSER_OPTIONS}
        />
        <ParserField
          label="Reasoning Parser"
          tooltip="Extract reasoning/thinking content from models that support it (Qwen3, DeepSeek-R1, Gemma 3, GLM-4.7, Phi-4 Reasoning, etc). 'Auto-detect' reads config.json. Qwen3 uses strict <think> tags, DeepSeek R1 uses lenient <think> (also works for Gemma 3, GLM-4.7, Phi-4), GPT-OSS/Harmony uses <|channel|> protocol. Click '?' for format examples."
          value={config.reasoningParser}
          onChange={v => onChange('reasoningParser', v)}
          options={REASONING_PARSER_OPTIONS}
        />
      </Section>

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
  const [above, setAbove] = useState(true)
  const triggerRef = useRef<HTMLSpanElement>(null)

  const handleEnter = () => {
    if (triggerRef.current) {
      const rect = triggerRef.current.getBoundingClientRect()
      setAbove(rect.top > 130)
    }
    setShow(true)
  }

  return (
    <span className="relative inline-flex ml-1">
      <span
        ref={triggerRef}
        className="inline-flex items-center justify-center w-3.5 h-3.5 rounded-full bg-muted text-muted-foreground text-[10px] font-bold cursor-help select-none"
        onMouseEnter={handleEnter}
        onMouseLeave={() => setShow(false)}
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
}

const TOOL_PARSER_OPTIONS: ParserOption[] = [
  { value: 'auto', label: 'Auto-detect (from config.json)' },
  { value: '', label: 'None (disable tool parsing)' },
  { value: 'qwen', label: 'Qwen — Qwen3, Qwen3-Coder, Qwen3-Next, Qwen2.5, Qwen2, QwQ-32B, Qwen3-MoE, Qwen3-VL', format: '<tool_call>{"name":"fn","arguments":{...}}</tool_call>' },
  { value: 'llama', label: 'Llama — Llama 4, Llama 3.3 70B, Llama 3.2 1B/3B/11B/90B, Llama 3.1 8B/70B/405B, Yi (Llama-arch)', format: '<function=name>{"arg":"val"}</function>' },
  { value: 'mistral', label: 'Mistral — Mistral Large/Small/Nemo/7B, Mixtral 8x7B/8x22B, Pixtral 12B/Large, Codestral 22B, Devstral Small', format: '[TOOL_CALLS][{"name":"fn","arguments":{...}}]' },
  { value: 'hermes', label: 'Hermes — Gemma 3 (1B–27B), Phi-4 Mini/Medium, Hermes 2/3/4/4.3, Qwen2.5-Hermes, other Hermes fine-tunes', format: '<tool_call>{"name":"fn","arguments":{...}}</tool_call>' },
  { value: 'deepseek', label: 'DeepSeek — DeepSeek-V2/V2.5/V3, DeepSeek-R1, Coder-V2 (native 671B only, NOT R1-Distill-Qwen/Llama)', format: '\u{ff5c}<tool_call>name\n{"arg":"val"}</tool_call>\u{ff5c}' },
  { value: 'nemotron', label: 'Nemotron — NVIDIA Nemotron-3/4 Nano/Super/Ultra (native Nemotron arch only, NOT Llama/Qwen fine-tunes)', format: '<toolcall>{"name":"fn","arguments":{...}}</toolcall>' },
  { value: 'glm47', label: 'GLM / GPT-OSS — GLM-4, GLM-4.7 9B, GLM-4.7 Flash, GLM-Z1, GPT-OSS-20B/120B', format: '<tool_call>name\n<arg_key>key</arg_key><arg_value>val</arg_value>\n</tool_call>' },
  { value: 'granite', label: 'Granite — IBM Granite 3.0/3.1/3.2/3.3, Granite-Code 3B/8B/20B/34B', format: '<|tool_call|>[{"name":"fn","arguments":{...}}]' },
  { value: 'functionary', label: 'Functionary — MeetKai Functionary v2/v3/v4r (7B/8B/70B)', format: '<|from|>assistant\n<|recipient|>fn\n<|content|>{"arg":"val"}' },
  { value: 'minimax', label: 'MiniMax — MiniMax-M1 40B, M2, M2.5 230B MoE, Prism Pro', format: '<minimax:tool_call><invoke name="fn"><parameter name="arg">val</parameter></invoke></minimax:tool_call>' },
  { value: 'xlam', label: 'xLAM — Salesforce xLAM-v2, xLAM-1B/7B/8x7B/8x22B', format: '[{"name":"fn","arguments":{...}}]' },
  { value: 'kimi', label: 'Kimi — Kimi-K2 1T MoE, K2.5, Moonshot-v1', format: '<|tool_calls_section_begin|><|tool_call_begin|>fn<|tool_call_argument_begin|>{...}<|tool_call_end|>' },
  { value: 'step3p5', label: 'StepFun — Step-3.5 Flash 8B MoE, Step-3.5', format: '<tool_call><function=fn><parameter=arg>val</parameter></function></tool_call>' },
]

const REASONING_PARSER_OPTIONS: ParserOption[] = [
  { value: 'auto', label: 'Auto-detect (from config.json)' },
  { value: '', label: 'None (disable reasoning extraction)' },
  { value: 'qwen3', label: 'Qwen3 — Qwen3, Qwen3-Coder, Qwen3-Next, QwQ-32B, StepFun Step-3.5, MiniMax-M2/M2.5', format: '<think>...reasoning...</think>content  (strict: requires both tags)' },
  { value: 'deepseek_r1', label: 'DeepSeek R1 — DeepSeek-R1/R1-Distill/R1-0528, Gemma 3, GLM-4.7, GLM-Z1, Phi-4 Reasoning', format: '<think>...reasoning...</think>content  (lenient: handles missing <think> tag)' },
  { value: 'openai_gptoss', label: 'GPT-OSS / Harmony — GLM-4.7 Flash, GPT-OSS-20B/120B', format: '<|channel|>analysis<|message|>reasoning...<|channel|>final<|message|>content' },
]

function ParserField({ label, tooltip, value, onChange, options }: {
  label: string; tooltip: string; value: string; onChange: (v: string) => void; options: ParserOption[]
}) {
  const [showHelp, setShowHelp] = useState(false)
  const selected = options.find(o => o.value === value)

  return (
    <div className="block">
      <span className="text-xs font-medium text-muted-foreground">
        {label}
        <Tooltip text={tooltip} />
        <button
          type="button"
          onClick={() => setShowHelp(!showHelp)}
          className="ml-1 inline-flex items-center justify-center w-3.5 h-3.5 rounded-full bg-muted text-muted-foreground text-[10px] font-bold cursor-help select-none hover:bg-accent"
          title="Show format examples"
        >
          ?
        </button>
      </span>
      <select value={value} onChange={e => onChange(e.target.value)} className="cfg-input">
        {options.map(o => (
          <option key={o.value} value={o.value}>{o.label}</option>
        ))}
      </select>
      {showHelp && (
        <div className="mt-1.5 bg-background border border-border rounded p-2 text-xs space-y-1.5 max-h-48 overflow-auto">
          {options.filter(o => o.format).map(o => (
            <div key={o.value} className={`${o.value === value ? 'text-primary' : 'text-muted-foreground'}`}>
              <span className="font-medium">{o.label}:</span>
              <code className="ml-1 text-[10px] bg-muted px-1 py-0.5 rounded break-all">{o.format}</code>
            </div>
          ))}
        </div>
      )}
      {selected?.format && !showHelp && (
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

export function Section({ title, expanded, onToggle, children }: {
  title: string; expanded: boolean; onToggle: () => void; children: React.ReactNode
}) {
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
}

export function SliderField({
  label, tooltip, value, onChange, min, max, step, defaultValue,
  allowUnlimited = false, unlimitedValue = 0, unlimitedLabel = 'Unlimited'
}: SliderFieldProps) {
  const isUnlimited = allowUnlimited && value === unlimitedValue

  const handleSliderChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    onChange(Number(e.target.value))
  }

  const handleInputChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const raw = e.target.value
    if (raw === '') return
    const num = Math.round(Number(raw))
    if (!isNaN(num)) {
      // Allow values beyond slider max via direct input (no upper clamp)
      onChange(Math.max(min, num))
    }
  }

  const handleInputBlur = (e: React.FocusEvent<HTMLInputElement>) => {
    const raw = e.target.value
    if (raw === '') {
      onChange(defaultValue)
      return
    }
    const num = Math.round(Number(raw))
    if (isNaN(num)) {
      onChange(defaultValue)
    } else {
      // Allow values beyond slider max via direct input (no upper clamp)
      onChange(Math.max(min, num))
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

  return (
    <div className="block">
      <div className="flex items-center justify-between">
        <span className="text-xs font-medium text-muted-foreground">
          {label}
          {tooltip && <Tooltip text={tooltip} />}
        </span>
        {allowUnlimited && (
          <button
            type="button"
            onClick={toggleUnlimited}
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
          disabled={isUnlimited}
        />
        <input
          type="number"
          className="w-20 px-2 py-1 bg-background border border-input rounded text-sm text-right tabular-nums"
          value={isUnlimited ? '' : value}
          onChange={handleInputChange}
          onBlur={handleInputBlur}
          placeholder={isUnlimited ? unlimitedLabel : undefined}
          disabled={isUnlimited}
          min={min}
          step={step}
        />
      </div>
    </div>
  )
}
