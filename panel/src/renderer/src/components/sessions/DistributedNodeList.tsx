import { useState, useEffect, useCallback } from 'react'

interface NodeInfo {
  node_id: string
  hostname: string
  address: string
  port: number
  chip: string
  ram_gb: number
  gpu_cores: number
  available_gb: number
  capability_score: number
  state: string
  is_coordinator: boolean
  assigned_layers?: [number, number] | null
  is_alive: boolean
  link_type?: string
  bandwidth_mbps?: number
  latency_ms?: number
}

interface DistributedNodeListProps {
  sessionId?: string
  enabled: boolean
}

const STATUS_COLORS: Record<string, string> = {
  active: 'bg-green-500',
  ready: 'bg-green-400',
  connected: 'bg-blue-400',
  discovered: 'bg-yellow-400',
  error: 'bg-red-500',
  dead: 'bg-red-700',
}

const LINK_LABELS: Record<string, string> = {
  thunderbolt: 'TB5',
  ethernet_10g: '10GbE',
  ethernet_1g: '1GbE',
  wifi: 'WiFi',
  tailscale: 'Tailscale',
  internet: 'Internet',
  unknown: '?',
}

export function DistributedNodeList({ sessionId, enabled }: DistributedNodeListProps): JSX.Element | null {
  const [nodes, setNodes] = useState<NodeInfo[]>([])
  const [scanning, setScanning] = useState(false)
  const [showManual, setShowManual] = useState(false)
  const [manualAddress, setManualAddress] = useState('')
  const [manualPort, setManualPort] = useState('9100')
  const [error, setError] = useState<string | null>(null)

  const api = (window as any).api?.distributed

  const fetchNodes = useCallback(async () => {
    if (!api || !enabled) return
    try {
      const result = await api.nodes(sessionId)
      if (result.success && result.nodes) {
        setNodes(result.nodes)
        setError(null)
      }
    } catch (e) {
      // Silent — session may not be running yet
    }
  }, [api, sessionId, enabled])

  // Poll for node updates every 5s when enabled
  useEffect(() => {
    if (!enabled) return
    fetchNodes()
    const interval = setInterval(fetchNodes, 5000)
    return () => clearInterval(interval)
  }, [enabled, fetchNodes])

  const handleScan = async () => {
    if (!api) return
    setScanning(true)
    setError(null)
    try {
      const result = await api.discover(sessionId)
      if (result.success) {
        await fetchNodes() // Refresh after scan
      } else {
        setError(result.error || 'Scan failed')
      }
    } catch (e) {
      setError((e as Error).message)
    }
    setScanning(false)
  }

  const handleAddManual = async () => {
    if (!api || !manualAddress.trim()) return
    setError(null)
    try {
      const result = await api.addNode(manualAddress.trim(), parseInt(manualPort) || 9100, sessionId)
      if (result.success) {
        setManualAddress('')
        setShowManual(false)
        await fetchNodes()
      } else {
        setError(result.error || 'Failed to add node')
      }
    } catch (e) {
      setError((e as Error).message)
    }
  }

  const handleRemove = async (nodeId: string) => {
    if (!api) return
    try {
      await api.removeNode(nodeId, sessionId)
      await fetchNodes()
    } catch (e) {
      setError((e as Error).message)
    }
  }

  if (!enabled) return null

  return (
    <div className="px-4 py-3 space-y-3">
      <div className="flex items-center justify-between">
        <div className="text-xs font-medium text-foreground">Cluster Nodes</div>
        <div className="flex gap-2">
          <button
            onClick={handleScan}
            disabled={scanning}
            className="text-xs px-2 py-1 rounded bg-primary/10 hover:bg-primary/20 text-primary transition-colors disabled:opacity-50"
          >
            {scanning ? 'Scanning...' : 'Scan for Nodes'}
          </button>
          <button
            onClick={() => setShowManual(!showManual)}
            className="text-xs px-2 py-1 rounded bg-muted hover:bg-muted/80 text-foreground transition-colors"
          >
            Add Manual
          </button>
        </div>
      </div>

      {error && (
        <div className="text-xs text-red-500 bg-red-500/10 px-2 py-1 rounded">{error}</div>
      )}

      {showManual && (
        <div className="flex gap-2 items-center">
          <input
            type="text"
            value={manualAddress}
            onChange={e => setManualAddress(e.target.value)}
            placeholder="IP address (e.g. 192.168.1.50)"
            className="cfg-input flex-1 text-xs"
            onKeyDown={e => e.key === 'Enter' && handleAddManual()}
          />
          <input
            type="text"
            value={manualPort}
            onChange={e => setManualPort(e.target.value)}
            placeholder="Port"
            className="cfg-input w-16 text-xs"
            onKeyDown={e => e.key === 'Enter' && handleAddManual()}
          />
          <button
            onClick={handleAddManual}
            className="text-xs px-2 py-1 rounded bg-primary text-primary-foreground hover:bg-primary/90 transition-colors"
          >
            Add
          </button>
        </div>
      )}

      {nodes.length === 0 ? (
        <div className="text-xs text-muted-foreground italic py-2">
          No nodes discovered yet. Start workers on other Macs, then click "Scan for Nodes".
        </div>
      ) : (
        <div className="space-y-2">
          {nodes.map(node => (
            <div
              key={node.node_id}
              className="flex items-center gap-3 p-2 rounded-md bg-muted/30 border border-border/50"
            >
              {/* Status indicator */}
              <div className={`w-2 h-2 rounded-full flex-shrink-0 ${
                node.is_alive ? (node.is_coordinator ? 'bg-blue-500' : STATUS_COLORS[node.state] || 'bg-gray-400') : 'bg-red-700'
              }`} />

              {/* Node info */}
              <div className="flex-1 min-w-0">
                <div className="flex items-center gap-2">
                  <span className="text-xs font-medium text-foreground truncate">
                    {node.hostname}
                  </span>
                  {node.is_coordinator && (
                    <span className="text-[10px] px-1 py-0.5 rounded bg-blue-500/20 text-blue-400 font-medium">
                      Coordinator
                    </span>
                  )}
                  {node.link_type && (
                    <span className="text-[10px] px-1 py-0.5 rounded bg-muted text-muted-foreground">
                      {LINK_LABELS[node.link_type] || node.link_type}
                    </span>
                  )}
                </div>
                <div className="text-[10px] text-muted-foreground flex gap-3 mt-0.5">
                  <span>{node.chip}</span>
                  <span>{node.ram_gb}GB RAM</span>
                  {node.assigned_layers && (
                    <span>Layers {node.assigned_layers[0]}-{node.assigned_layers[1] - 1}</span>
                  )}
                  {node.latency_ms !== undefined && node.latency_ms < 999 && (
                    <span>{node.latency_ms.toFixed(1)}ms</span>
                  )}
                  {node.bandwidth_mbps !== undefined && node.bandwidth_mbps > 0 && (
                    <span>{node.bandwidth_mbps >= 1000 ? `${(node.bandwidth_mbps / 1000).toFixed(0)}Gbps` : `${node.bandwidth_mbps.toFixed(0)}Mbps`}</span>
                  )}
                </div>
              </div>

              {/* Remove button (not for coordinator) */}
              {!node.is_coordinator && (
                <button
                  onClick={() => handleRemove(node.node_id)}
                  className="text-[10px] text-muted-foreground hover:text-red-400 transition-colors px-1"
                  title="Remove from cluster"
                >
                  x
                </button>
              )}
            </div>
          ))}
        </div>
      )}
    </div>
  )
}
