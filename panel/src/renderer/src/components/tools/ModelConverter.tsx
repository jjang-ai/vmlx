import { useState, useEffect } from 'react'
import { ArrowLeft, CheckCircle2, XCircle, Play, Settings2, FolderOpen } from 'lucide-react'
import { useStreamingOperation } from './useStreamingOperation'
import { LogViewer } from './LogViewer'

type QuantMode = 'mlx' | 'jang'
type Preset = 'balanced' | 'quality' | 'compact' | 'custom' | string

const MLX_PRESETS: Record<string, { bits: number; groupSize: number; label: string; desc: string }> = {
  balanced: { bits: 4, groupSize: 64, label: 'Balanced (4-bit)', desc: 'Good quality/size tradeoff — recommended' },
  quality: { bits: 8, groupSize: 64, label: 'Quality (8-bit)', desc: 'Larger but better quality' },
  compact: { bits: 3, groupSize: 64, label: 'Compact (3-bit)', desc: 'Smallest, some quality loss' },
}

const JANG_PRESETS: Record<string, { profile: string; method: string; label: string; desc: string; avgBits: string }> = {
  // 2-bit tier — ultra compact (attention MUST be protected at 2-bit)
  jang_2s: { profile: 'JANG_2S', method: 'mse', label: '2S — Tight', desc: '6-bit attn, 4-bit embed, 2-bit MLP', avgBits: '~2.3' },
  jang_2m: { profile: 'JANG_2M', method: 'mse', label: '2M — Balanced', desc: '8-bit attn, 4-bit embed, 2-bit MLP', avgBits: '~2.5' },
  jang_2l: { profile: 'JANG_2L', method: 'mse', label: '2L — Quality', desc: '8-bit attn, 6-bit embed, 2-bit MLP (proven 73% MMLU on 122B)', avgBits: '~2.7' },
  jang_1l: { profile: 'JANG_1L', method: 'mse', label: '1L — Max Quality 2-bit', desc: '8-bit attn+embed, 2-bit MLP (proven 6/6 free-form)', avgBits: '~2.4' },
  // 3-bit tier — only attention needs protection at 3-bit+
  jang_3s: { profile: 'JANG_3S', method: 'mse', label: '3S — Compact', desc: '6-bit attn, 3-bit everything else', avgBits: '~3.1' },
  jang_3m: { profile: 'JANG_3M', method: 'mse', label: '3M — Recommended', desc: '8-bit attn, 3-bit everything else', avgBits: '~3.2' },
  jang_3l: { profile: 'JANG_3L', method: 'mse', label: '3L — Protected', desc: '8-bit attn, 4-bit embed, 3-bit MLP', avgBits: '~3.4' },
  // 4-bit tier — standard quality, ~2% overhead on MoE for 8-bit attention
  jang_4s: { profile: 'JANG_4S', method: 'mse', label: '4S — Light', desc: '6-bit attn, 4-bit everything else', avgBits: '~4.1' },
  jang_4m: { profile: 'JANG_4M', method: 'mse', label: '4M — Standard', desc: '8-bit attn, 4-bit everything else (~2% overhead on MoE)', avgBits: '~4.2' },
  jang_4l: { profile: 'JANG_4L', method: 'mse', label: '4L — Premium', desc: '8-bit attn, 6-bit embed, 4-bit MLP', avgBits: '~4.5' },
  // 6-bit tier — near lossless
  jang_6m: { profile: 'JANG_6M', method: 'mse', label: '6M — Near Lossless', desc: '8-bit attn, 6-bit everything else', avgBits: '~6.2' },
}

interface ModelConverterProps {
  initialModelPath?: string | null
  onBack: () => void
  onServe?: (modelPath: string) => void
  models?: Array<{ name: string; path: string }>
}

export function ModelConverter({ initialModelPath, onBack, onServe, models = [] }: ModelConverterProps) {
  const [modelPath, setModelPath] = useState(initialModelPath || '')
  const [quantMode, setQuantMode] = useState<QuantMode>('jang')
  const [preset, setPreset] = useState<Preset>('jang_3m')
  const [bits, setBits] = useState(4)
  const [groupSize, setGroupSize] = useState(64)
  const [mode, setMode] = useState('default')
  const [dtype, setDtype] = useState('')
  const [jangMethod, setJangMethod] = useState('mse')
  // Custom JANG mixed-precision settings
  const [customCritical, setCustomCritical] = useState(8)
  const [customImportant, setCustomImportant] = useState(4)
  const [customCompress, setCustomCompress] = useState(3)
  const [outputDir, setOutputDir] = useState('')
  const [force, setForce] = useState(false)
  const [skipVerify, setSkipVerify] = useState(false)
  const [trustRemoteCode, setTrustRemoteCode] = useState(false)

  const [success, setSuccess] = useState<boolean | null>(null)
  const [outputPath, setOutputPath] = useState<string | undefined>()

  const { running, logLines, wasCancelled, start, cancel } = useStreamingOperation()

  useEffect(() => {
    if (initialModelPath) setModelPath(initialModelPath)
  }, [initialModelPath])

  useEffect(() => {
    if (preset in MLX_PRESETS) {
      const p = MLX_PRESETS[preset]
      setBits(p.bits)
      setGroupSize(p.groupSize)
    } else if (preset in JANG_PRESETS) {
      setJangMethod(JANG_PRESETS[preset].method)
    }
  }, [preset])

  const runConvert = async () => {
    if (!modelPath.trim() || running) return
    setSuccess(null)
    setOutputPath(undefined)

    const isJang = preset in JANG_PRESETS || preset === 'jang_custom'
    // For custom mix, create a dynamic profile name that jang-tools will interpret
    const jangProfile = preset === 'jang_custom'
      ? `CUSTOM_${customCritical}_${customImportant}_${customCompress}`
      : (isJang ? JANG_PRESETS[preset].profile : undefined)
    const { ipcResult, allLines } = await start(() =>
      window.api.developer.convert({
        model: modelPath.trim(),
        output: outputDir || undefined,
        bits,
        groupSize,
        mode: mode !== 'default' ? mode : undefined,
        dtype: dtype || undefined,
        force,
        skipVerify,
        trustRemoteCode,
        jangProfile,
        jangMethod: isJang ? jangMethod : undefined,
      })
    )

    if (ipcResult?.cancelled) {
      setSuccess(false)
    } else {
      const ok = ipcResult?.success ?? false
      setSuccess(ok)
      if (ok) {
        const pathLine = allLines?.find((l: string) => l.startsWith('Output path:'))
        if (pathLine) setOutputPath(pathLine.replace('Output path:', '').trim())
      }
    }
  }

  const handleBrowseOutput = async () => {
    try {
      const dir = await window.api.developer.browseOutputDir()
      if (dir) setOutputDir(dir)
    } catch { /* dialog cancelled or unavailable */ }
  }

  return (
    <div className="p-6 overflow-auto h-full">
      <div className="max-w-3xl mx-auto space-y-6">
        <button
          onClick={onBack}
          className="text-xs text-muted-foreground hover:text-foreground inline-flex items-center gap-1"
        >
          <ArrowLeft className="h-3 w-3" />
          Back
        </button>

        <h2 className="text-2xl font-bold">Model Converter</h2>
        <p className="text-sm text-muted-foreground">
          Convert HuggingFace models to quantized format for Apple Silicon
        </p>

        {/* Quantization Mode Toggle */}
        <div className="space-y-2">
          <label className="text-sm font-medium">Format</label>
          <div className="flex gap-2">
            <button
              onClick={() => { setQuantMode('jang'); setPreset('jang_3m') }}
              disabled={running}
              className={`flex-1 px-4 py-2.5 text-sm rounded-lg border transition-colors ${
                quantMode === 'jang'
                  ? 'border-violet-500 bg-violet-500/10 text-violet-400'
                  : 'border-border hover:bg-accent text-muted-foreground'
              }`}
            >
              <span className="font-medium">JANG</span>
              <span className="text-xs block mt-0.5 opacity-75">Mixed-precision adaptive — best quality/size</span>
            </button>
            <button
              onClick={() => { setQuantMode('mlx'); setPreset('balanced') }}
              disabled={running}
              className={`flex-1 px-4 py-2.5 text-sm rounded-lg border transition-colors ${
                quantMode === 'mlx'
                  ? 'border-primary bg-primary/10 text-primary'
                  : 'border-border hover:bg-accent text-muted-foreground'
              }`}
            >
              <span className="font-medium">MLX Uniform</span>
              <span className="text-xs block mt-0.5 opacity-75">Standard uniform bit-width quantization</span>
            </button>
          </div>
        </div>

        {/* Model input */}
        <div className="space-y-2">
          <label className="text-sm font-medium">Source Model</label>
          <p className="text-xs text-muted-foreground">
            Local path or HuggingFace repo ID (e.g., nvidia/Nemotron-H-47B-BF16)
          </p>
          <input
            type="text"
            value={modelPath}
            onChange={e => setModelPath(e.target.value)}
            onKeyDown={e => e.key === 'Enter' && runConvert()}
            placeholder="/path/to/model or org/model-name"
            className="w-full px-3 py-2 bg-background border border-input rounded text-sm focus:outline-none focus:ring-1 focus:ring-ring"
            list="convert-model-paths"
            disabled={running}
          />
          <datalist id="convert-model-paths">
            {models.map(m => (
              <option key={m.path} value={m.path}>{m.name}</option>
            ))}
          </datalist>
        </div>

        {/* Presets */}
        <div className="space-y-3">
          <label className="text-sm font-medium">Profile</label>
          <div className="grid grid-cols-2 gap-2">
            {quantMode === 'jang' ? (
              <>
                {['2-bit', '3-bit', '4-bit', '6-bit'].map(tier => {
                  const tierKeys = Object.entries(JANG_PRESETS).filter(([, p]) => {
                    if (tier === '2-bit') return p.profile.startsWith('JANG_2') || p.profile === 'JANG_1L'
                    if (tier === '3-bit') return p.profile.startsWith('JANG_3')
                    if (tier === '4-bit') return p.profile.startsWith('JANG_4')
                    if (tier === '6-bit') return p.profile.startsWith('JANG_6')
                    return false
                  })
                  if (tierKeys.length === 0) return null
                  return (
                    <div key={tier} className="col-span-2">
                      <p className="text-[10px] text-muted-foreground uppercase tracking-wider mb-1">{tier} tier</p>
                      <div className="grid grid-cols-2 sm:grid-cols-4 gap-1.5">
                        {tierKeys.map(([key, p]) => (
                          <button
                            key={key}
                            onClick={() => setPreset(key)}
                            disabled={running}
                            className={`p-2 text-left border rounded transition-colors ${
                              preset === key
                                ? 'border-violet-500 bg-violet-500/5'
                                : 'border-border hover:bg-accent'
                            }`}
                          >
                            <div className="flex items-center gap-1">
                              <p className="text-xs font-medium">{p.label}</p>
                              <span className="text-[9px] px-1 py-0.5 rounded bg-violet-500/15 text-violet-400">{p.avgBits}</span>
                            </div>
                            <p className="text-[10px] text-muted-foreground mt-0.5 leading-tight">{p.desc}</p>
                          </button>
                        ))}
                      </div>
                    </div>
                  )
                })}
                {/* Custom mix option */}
                <div className="col-span-2">
                  <p className="text-[10px] text-muted-foreground uppercase tracking-wider mb-1">Custom Mix</p>
                  <button
                    onClick={() => setPreset('jang_custom')}
                    disabled={running}
                    className={`w-full p-2 text-left border rounded transition-colors ${
                      preset === 'jang_custom'
                        ? 'border-violet-500 bg-violet-500/5'
                        : 'border-border hover:bg-accent'
                    }`}
                  >
                    <p className="text-xs font-medium">Custom — Choose your own bit widths</p>
                    <p className="text-[10px] text-muted-foreground mt-0.5">Set Critical (attention), Important (embeddings), Compress (MLP) bits independently</p>
                  </button>
                  {preset === 'jang_custom' && (
                    <div className="grid grid-cols-3 gap-2 mt-2">
                      <div>
                        <label className="text-[10px] text-muted-foreground">Critical (attention)</label>
                        <select value={customCritical} onChange={e => setCustomCritical(Number(e.target.value))} disabled={running} className="w-full px-2 py-1 bg-background border border-input rounded text-xs">
                          {[2,3,4,5,6,8].map(b => <option key={b} value={b}>{b}-bit</option>)}
                        </select>
                      </div>
                      <div>
                        <label className="text-[10px] text-muted-foreground">Important (embeddings)</label>
                        <select value={customImportant} onChange={e => setCustomImportant(Number(e.target.value))} disabled={running} className="w-full px-2 py-1 bg-background border border-input rounded text-xs">
                          {[2,3,4,5,6,8].map(b => <option key={b} value={b}>{b}-bit</option>)}
                        </select>
                      </div>
                      <div>
                        <label className="text-[10px] text-muted-foreground">Compress (MLP/FFN)</label>
                        <select value={customCompress} onChange={e => setCustomCompress(Number(e.target.value))} disabled={running} className="w-full px-2 py-1 bg-background border border-input rounded text-xs">
                          {[2,3,4,5,6,8].map(b => <option key={b} value={b}>{b}-bit</option>)}
                        </select>
                      </div>
                    </div>
                  )}
                </div>
              </>
            ) : (
              <>
                {Object.entries(MLX_PRESETS).map(([key, p]) => (
                  <button
                    key={key}
                    onClick={() => setPreset(key as Preset)}
                    disabled={running}
                    className={`p-3 text-left border rounded-lg transition-colors ${
                      preset === key
                        ? 'border-primary bg-primary/5'
                        : 'border-border hover:bg-accent'
                    }`}
                  >
                    <p className="text-sm font-medium">{p.label}</p>
                    <p className="text-xs text-muted-foreground mt-0.5">{p.desc}</p>
                  </button>
                ))}
                <button
                  onClick={() => setPreset('custom')}
                  disabled={running}
                  className={`p-3 text-left border rounded-lg transition-colors ${
                    preset === 'custom'
                      ? 'border-primary bg-primary/5'
                      : 'border-border hover:bg-accent'
                  }`}
                >
                  <div className="flex items-center gap-1.5">
                    <Settings2 className="h-3.5 w-3.5" />
                    <p className="text-sm font-medium">Custom</p>
                  </div>
                  <p className="text-xs text-muted-foreground mt-0.5">Full control over all settings</p>
                </button>
              </>
            )}
          </div>
        </div>

        {/* JANG method selector */}
        {quantMode === 'jang' && (
          <div className="space-y-2">
            <label className="text-sm font-medium">Quantization Method</label>
            <select
              value={jangMethod}
              onChange={e => setJangMethod(e.target.value)}
              disabled={running}
              className="w-full px-3 py-2 bg-background border border-input rounded text-sm"
            >
              <option value="mse">MSE-optimal (recommended — best quality)</option>
              <option value="rtn">RTN (fast — lower quality)</option>
              <option value="mse-all">MSE everywhere (slow — maximum quality)</option>
            </select>
          </div>
        )}

        {/* Advanced options (MLX custom preset only) */}
        {quantMode === 'mlx' && preset === 'custom' && (
          <div className="space-y-4 p-4 border border-border rounded-lg">
            <h3 className="text-sm font-medium">Advanced Settings</h3>

            <div className="grid grid-cols-2 gap-4">
              <div className="space-y-1">
                <label className="text-xs font-medium">Bits</label>
                <select
                  value={bits}
                  onChange={e => setBits(Number(e.target.value))}
                  disabled={running}
                  className="w-full px-3 py-2 bg-background border border-input rounded text-sm"
                >
                  {[2, 3, 4, 6, 8].map(b => (
                    <option key={b} value={b}>{b}-bit</option>
                  ))}
                </select>
              </div>

              <div className="space-y-1">
                <label className="text-xs font-medium">Group Size</label>
                <select
                  value={groupSize}
                  onChange={e => setGroupSize(Number(e.target.value))}
                  disabled={running}
                  className="w-full px-3 py-2 bg-background border border-input rounded text-sm"
                >
                  {[32, 64, 128].map(g => (
                    <option key={g} value={g}>{g}</option>
                  ))}
                </select>
              </div>

              <div className="space-y-1">
                <label className="text-xs font-medium">Mode</label>
                <select
                  value={mode}
                  onChange={e => setMode(e.target.value)}
                  disabled={running}
                  className="w-full px-3 py-2 bg-background border border-input rounded text-sm"
                >
                  <option value="default">Default</option>
                  <option value="NF4">NF4</option>
                </select>
              </div>

              <div className="space-y-1">
                <label className="text-xs font-medium">Non-quantized dtype</label>
                <select
                  value={dtype}
                  onChange={e => setDtype(e.target.value)}
                  disabled={running}
                  className="w-full px-3 py-2 bg-background border border-input rounded text-sm"
                >
                  <option value="">Auto</option>
                  <option value="float16">float16</option>
                  <option value="bfloat16">bfloat16</option>
                </select>
              </div>
            </div>

          </div>
        )}

        {/* Options (shared between MLX and JANG) */}
        <div className="space-y-2">
          <label className="flex items-center gap-2 text-xs">
            <input type="checkbox" checked={force} onChange={e => setForce(e.target.checked)} disabled={running} className="rounded border-input" />
            <span className="text-muted-foreground">Force overwrite if output exists</span>
          </label>
          {quantMode === 'mlx' && (
            <label className="flex items-center gap-2 text-xs">
              <input type="checkbox" checked={skipVerify} onChange={e => setSkipVerify(e.target.checked)} disabled={running} className="rounded border-input" />
              <span className="text-muted-foreground">Skip post-conversion verification</span>
            </label>
          )}
          <label className="flex items-center gap-2 text-xs">
            <input type="checkbox" checked={trustRemoteCode} onChange={e => setTrustRemoteCode(e.target.checked)} disabled={running} className="rounded border-input" />
            <span className="text-muted-foreground">Trust remote code from HuggingFace</span>
          </label>
        </div>

        {/* Output directory */}
        <div className="space-y-2">
          <label className="text-sm font-medium">Output Directory</label>
          <p className="text-xs text-muted-foreground">Leave empty for auto-generated name</p>
          <div className="flex gap-2">
            <input
              type="text"
              value={outputDir}
              onChange={e => setOutputDir(e.target.value)}
              placeholder={quantMode === 'jang' ? 'Auto-generated (e.g., Model-Name-JANG_3M)' : 'Auto-generated (e.g., Model-Name-vmlx-4bit)'}
              className="flex-1 px-3 py-2 bg-background border border-input rounded text-sm focus:outline-none focus:ring-1 focus:ring-ring"
              disabled={running}
            />
            <button
              onClick={handleBrowseOutput}
              disabled={running}
              className="px-3 py-2 text-sm border border-input rounded hover:bg-accent transition-colors disabled:opacity-50"
              title="Browse"
            >
              <FolderOpen className="h-4 w-4" />
            </button>
          </div>
        </div>

        {/* Action buttons */}
        <div className="flex gap-3">
          {running ? (
            <button
              onClick={cancel}
              className="px-6 py-2.5 text-sm bg-destructive text-destructive-foreground rounded-md hover:bg-destructive/90 flex items-center gap-2"
            >
              Cancel Conversion
            </button>
          ) : (
            <button
              onClick={runConvert}
              disabled={!modelPath.trim()}
              className="px-6 py-2.5 text-sm bg-primary text-primary-foreground rounded-md hover:bg-primary/90 disabled:opacity-50 flex items-center gap-2"
            >
              <Play className="h-4 w-4" />
              {quantMode === 'jang' ? 'Convert to JANG' : 'Convert to MLX'}
            </button>
          )}
        </div>

        {/* Status */}
        {success !== null && (
          <div className={`p-4 rounded-lg border flex items-center gap-3 ${
            success ? 'bg-green-500/10 border-green-500/20' : 'bg-destructive/10 border-destructive/20'
          }`}>
            {success ? (
              <>
                <CheckCircle2 className="h-5 w-5 text-green-500" />
                <div className="flex-1">
                  <p className="text-sm font-medium">Conversion complete</p>
                  {outputPath && <p className="text-xs text-muted-foreground mt-0.5">{outputPath}</p>}
                </div>
                {outputPath && onServe && (
                  <button
                    onClick={() => onServe(outputPath)}
                    className="px-3 py-1.5 text-xs bg-primary text-primary-foreground rounded hover:bg-primary/90 flex items-center gap-1.5"
                  >
                    <Play className="h-3 w-3" />
                    Serve Model
                  </button>
                )}
              </>
            ) : (
              <>
                <XCircle className="h-5 w-5 text-destructive" />
                <p className="text-sm font-medium">{wasCancelled ? 'Conversion cancelled' : 'Conversion failed'}</p>
              </>
            )}
          </div>
        )}

        <LogViewer logLines={logLines} running={running} defaultOpen />
      </div>
    </div>
  )
}
