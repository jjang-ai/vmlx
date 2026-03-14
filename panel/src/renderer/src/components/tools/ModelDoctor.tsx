import { useState, useEffect, useCallback } from 'react'
import { ArrowLeft, Loader2, CheckCircle2, XCircle, AlertTriangle } from 'lucide-react'
import { useStreamingOperation } from './useStreamingOperation'
import { LogViewer } from './LogViewer'

type CheckStatus = 'pass' | 'fail' | 'warn' | 'pending' | 'running'

interface CheckResult {
  name: string
  status: CheckStatus
  details: string[]
}

interface ModelDoctorProps {
  initialModelPath?: string | null
  onBack: () => void
  models?: Array<{ name: string; path: string }>
}

function parseOutput(lines: string[]): CheckResult[] {
  const checks: CheckResult[] = [
    { name: 'Config', status: 'pending', details: [] },
    { name: 'Weights', status: 'pending', details: [] },
    { name: 'Architecture', status: 'pending', details: [] },
    { name: 'Inference', status: 'pending', details: [] },
  ]

  let currentIdx = -1

  for (const line of lines) {
    const trimmed = line.trim()

    // Section headers — set current check to running
    if (trimmed.startsWith('Checking config')) {
      currentIdx = 0
      checks[0].status = 'running'
    } else if (trimmed.startsWith('Checking weights')) {
      currentIdx = 1
      checks[1].status = 'running'
    } else if (trimmed.startsWith('Checking architecture')) {
      currentIdx = 2
      checks[2].status = 'running'
    } else if (trimmed.startsWith('Running inference')) {
      currentIdx = 3
      checks[3].status = 'running'
    } else if (trimmed.includes('skipped (--no-inference)')) {
      checks[3].status = 'pass'
      checks[3].details.push('Skipped')
      currentIdx = -1
    } else if (currentIdx >= 0) {
      const current = checks[currentIdx]
      if (trimmed.includes(': OK')) {
        current.status = 'pass'
        currentIdx = -1
      } else if (trimmed.includes(': PASS')) {
        current.status = 'pass'
        const passIdx = trimmed.indexOf('PASS')
        const detail = trimmed.substring(passIdx + 4).replace(/^\s*-\s*/, '').trim()
        if (detail) current.details.push(detail)
        currentIdx = -1
      } else if (trimmed.startsWith('Found ') || trimmed.startsWith('LatentMoE')) {
        current.details.push(trimmed)
      }
    }

    // Summary section — apply FAIL/WARN from summary to correct check
    if (trimmed.startsWith('FAIL:')) {
      const detail = trimmed.replace(/^FAIL:\s*/, '')
      for (const check of checks) {
        if (detail.startsWith(check.name + ':')) {
          check.status = 'fail'
          check.details.push(detail.replace(check.name + ': ', ''))
          break
        }
      }
    }
    if (trimmed.startsWith('WARN:')) {
      const detail = trimmed.replace(/^WARN:\s*/, '')
      for (const check of checks) {
        if (detail.startsWith(check.name + ':')) {
          if (check.status !== 'fail') check.status = 'warn'
          check.details.push(detail.replace(check.name + ': ', ''))
          break
        }
      }
    }
    if (trimmed.startsWith('ALL CHECKS PASSED')) {
      checks.forEach(c => { if (c.status === 'pending' || c.status === 'running') c.status = 'pass' })
    }
  }

  return checks
}

export function ModelDoctor({ initialModelPath, onBack, models = [] }: ModelDoctorProps) {
  const [modelPath, setModelPath] = useState(initialModelPath || '')
  const [noInference, setNoInference] = useState(true)
  const [checks, setChecks] = useState<CheckResult[]>([])
  const [done, setDone] = useState(false)

  const onLogUpdate = useCallback((lines: string[]) => {
    setChecks(parseOutput(lines))
  }, [])

  const { running, logLines, wasCancelled, start, cancel } = useStreamingOperation(onLogUpdate)

  useEffect(() => {
    if (initialModelPath) setModelPath(initialModelPath)
  }, [initialModelPath])

  const runDoctor = async () => {
    if (!modelPath.trim() || running) return
    setChecks([])
    setDone(false)
    await start(() => window.api.developer.doctor(modelPath.trim(), { noInference }))
    setDone(true)
  }

  const passCount = checks.filter(c => c.status === 'pass').length
  const failCount = checks.filter(c => c.status === 'fail').length
  const warnCount = checks.filter(c => c.status === 'warn').length
  const totalChecks = checks.filter(c => c.status !== 'pending').length

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

        <h2 className="text-2xl font-bold">Model Doctor</h2>
        <p className="text-sm text-muted-foreground">
          Run diagnostics on model config, weights, architecture, and inference
        </p>

        {/* Model input */}
        <div className="space-y-3">
          <div className="space-y-2">
            <label className="text-sm font-medium">Model Path or HuggingFace ID</label>
            <div className="flex gap-2">
              <input
                type="text"
                value={modelPath}
                onChange={e => setModelPath(e.target.value)}
                onKeyDown={e => e.key === 'Enter' && runDoctor()}
                placeholder="/path/to/model or org/model-name"
                className="flex-1 px-3 py-2 bg-background border border-input rounded text-sm focus:outline-none focus:ring-1 focus:ring-ring"
                list="doctor-model-paths"
                disabled={running}
              />
              <datalist id="doctor-model-paths">
                {models.map(m => (
                  <option key={m.path} value={m.path}>{m.name}</option>
                ))}
              </datalist>
              {running ? (
                <button
                  onClick={cancel}
                  className="px-4 py-2 text-sm bg-destructive text-destructive-foreground rounded hover:bg-destructive/90"
                >
                  Cancel
                </button>
              ) : (
                <button
                  onClick={runDoctor}
                  disabled={!modelPath.trim()}
                  className="px-4 py-2 text-sm bg-primary text-primary-foreground rounded hover:bg-primary/90 disabled:opacity-50 flex items-center gap-2"
                >
                  Run Diagnostics
                </button>
              )}
            </div>
          </div>

          <label className="flex items-center gap-2 text-sm">
            <input
              type="checkbox"
              checked={!noInference}
              onChange={e => setNoInference(!e.target.checked)}
              disabled={running}
              className="rounded border-input"
            />
            <span className="text-muted-foreground">
              Include inference test
              <span className="text-xs ml-1">(loads full model, uses significant memory)</span>
            </span>
          </label>
        </div>

        {/* Summary bar */}
        {totalChecks > 0 && (
          <div className={`p-4 rounded-lg border ${
            wasCancelled ? 'bg-muted border-border' :
            failCount > 0 ? 'bg-destructive/10 border-destructive/20' :
            warnCount > 0 ? 'bg-yellow-500/10 border-yellow-500/20' :
            done ? 'bg-green-500/10 border-green-500/20' :
            'bg-muted border-border'
          }`}>
            <div className="flex items-center gap-3">
              {running ? (
                <Loader2 className="h-5 w-5 animate-spin text-muted-foreground" />
              ) : wasCancelled ? (
                <XCircle className="h-5 w-5 text-muted-foreground" />
              ) : failCount > 0 ? (
                <XCircle className="h-5 w-5 text-destructive" />
              ) : warnCount > 0 ? (
                <AlertTriangle className="h-5 w-5 text-yellow-500" />
              ) : (
                <CheckCircle2 className="h-5 w-5 text-green-500" />
              )}
              <div>
                <p className="text-sm font-medium">
                  {running ? 'Running diagnostics...' :
                   wasCancelled ? 'Diagnostics cancelled' :
                   failCount > 0 ? `${failCount} issue${failCount > 1 ? 's' : ''} found` :
                   warnCount > 0 ? `${passCount} passed, ${warnCount} warning${warnCount > 1 ? 's' : ''}` :
                   `All ${passCount} checks passed`}
                </p>
                {!running && !wasCancelled && (
                  <p className="text-xs text-muted-foreground">
                    {passCount} passed, {failCount} failed, {warnCount} warnings
                  </p>
                )}
              </div>
            </div>
          </div>
        )}

        {/* Check cards */}
        {checks.filter(c => c.status !== 'pending').map(check => (
          <CheckCard key={check.name} check={check} />
        ))}

        <LogViewer logLines={logLines} running={running} />
      </div>
    </div>
  )
}

function CheckCard({ check }: { check: CheckResult }) {
  const statusIcon = {
    pass: <CheckCircle2 className="h-4 w-4 text-green-500" />,
    fail: <XCircle className="h-4 w-4 text-destructive" />,
    warn: <AlertTriangle className="h-4 w-4 text-yellow-500" />,
    running: <Loader2 className="h-4 w-4 animate-spin text-muted-foreground" />,
    pending: <div className="h-4 w-4 rounded-full border-2 border-muted-foreground/30" />,
  }

  const statusBg = {
    pass: 'border-green-500/20',
    fail: 'border-destructive/20',
    warn: 'border-yellow-500/20',
    running: 'border-border',
    pending: 'border-border',
  }

  return (
    <div className={`p-4 border rounded-lg ${statusBg[check.status]}`}>
      <div className="flex items-center gap-2.5">
        {statusIcon[check.status]}
        <span className="text-sm font-medium">{check.name}</span>
        <span className="text-xs text-muted-foreground uppercase">
          {check.status === 'running' ? 'checking...' : check.status}
        </span>
      </div>
      {check.details.length > 0 && (
        <div className="mt-2 ml-7 space-y-1">
          {check.details.map((detail, i) => (
            <p key={i} className="text-xs text-muted-foreground font-mono">{detail}</p>
          ))}
        </div>
      )}
    </div>
  )
}
