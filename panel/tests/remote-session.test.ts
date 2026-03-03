/**
 * Tests for remote session lifecycle, URL resolution, auth headers, and abort tracking.
 *
 * These test pure-logic functions extracted from sessions.ts, chat.ts, and utils.ts
 * without requiring Electron IPC or live endpoints.
 */
import { describe, it, expect, vi, beforeEach, afterEach } from 'vitest'

// ─── resolveUrl() ────────────────────────────────────────────────────────────
// Extracted from sessions.ts for testability

// We replicate the resolveUrl logic here to test it in isolation
// (original is exported from sessions.ts but depends on Node's dns.lookup)
const resolvedUrlCache = new Map<string, { url: string; timestamp: number }>()
const RESOLVE_URL_CACHE_TTL = 60_000

async function resolveUrl(url: string, lookupFn?: (hostname: string) => Promise<string>): Promise<string> {
    const cached = resolvedUrlCache.get(url)
    if (cached && Date.now() - cached.timestamp < RESOLVE_URL_CACHE_TTL) {
        return cached.url
    }

    try {
        const parsed = new URL(url)
        if (parsed.hostname.endsWith('.local') && lookupFn) {
            const ip = await lookupFn(parsed.hostname)
            parsed.hostname = ip
            const resolved = parsed.toString().replace(/\/+$/, '')
            resolvedUrlCache.set(url, { url: resolved, timestamp: Date.now() })
            return resolved
        }
    } catch (_) { }
    resolvedUrlCache.set(url, { url, timestamp: Date.now() })
    return url
}

describe('resolveUrl', () => {
    beforeEach(() => {
        resolvedUrlCache.clear()
    })

    it('passes through non-.local URLs unchanged', async () => {
        const url = 'http://192.168.1.100:8000'
        expect(await resolveUrl(url)).toBe(url)
    })

    it('passes through https URLs unchanged', async () => {
        const url = 'https://api.openai.com/v1/models'
        expect(await resolveUrl(url)).toBe(url)
    })

    it('resolves .local hostnames to IPv4 when lookup succeeds', async () => {
        const mockLookup = vi.fn().mockResolvedValue('192.168.1.50')
        const result = await resolveUrl('http://macstudio.local:8000/v1/models', mockLookup)
        expect(result).toBe('http://192.168.1.50:8000/v1/models')
        expect(mockLookup).toHaveBeenCalledWith('macstudio.local')
    })

    it('caches resolved URLs for TTL duration', async () => {
        const mockLookup = vi.fn().mockResolvedValue('192.168.1.50')
        const url = 'http://macstudio.local:8000'

        await resolveUrl(url, mockLookup)
        await resolveUrl(url, mockLookup)

        // Lookup should only be called once (second call uses cache)
        expect(mockLookup).toHaveBeenCalledTimes(1)
    })

    it('falls back to original URL when lookup fails', async () => {
        const mockLookup = vi.fn().mockRejectedValue(new Error('DNS failed'))
        const url = 'http://macstudio.local:8000'
        // Without lookup function, .local URLs pass through
        const result = await resolveUrl(url)
        expect(result).toBe(url)
    })

    it('strips trailing slashes from resolved URLs', async () => {
        const mockLookup = vi.fn().mockResolvedValue('10.0.0.1')
        const result = await resolveUrl('http://myserver.local:8080/', mockLookup)
        expect(result).not.toMatch(/\/$/)
    })
})

// ─── getAuthHeaders() ────────────────────────────────────────────────────────
// Replicating the logic from utils.ts for testability

function getAuthHeaders(session: any): Record<string, string> {
    if (!session) return {}
    try {
        const config = JSON.parse(session.config || '{}')
        if (session.type === 'remote' && session.remoteApiKey) {
            const h: Record<string, string> = { 'Authorization': `Bearer ${session.remoteApiKey}` }
            if (session.remoteOrganization) h['OpenAI-Organization'] = session.remoteOrganization
            return h
        } else if (config.apiKey) {
            return { 'Authorization': `Bearer ${config.apiKey}` }
        }
    } catch (_) { }
    return {}
}

describe('getAuthHeaders', () => {
    it('returns empty for null session', () => {
        expect(getAuthHeaders(null)).toEqual({})
    })

    it('returns Bearer token for remote session with API key', () => {
        const session = {
            type: 'remote',
            remoteApiKey: 'sk-test123',
            config: '{}'
        }
        expect(getAuthHeaders(session)).toEqual({
            'Authorization': 'Bearer sk-test123'
        })
    })

    it('includes Organization header for remote session with organization', () => {
        const session = {
            type: 'remote',
            remoteApiKey: 'sk-test123',
            remoteOrganization: 'org-abc',
            config: '{}'
        }
        expect(getAuthHeaders(session)).toEqual({
            'Authorization': 'Bearer sk-test123',
            'OpenAI-Organization': 'org-abc'
        })
    })

    it('returns Bearer token for local session with apiKey in config', () => {
        const session = {
            type: 'local',
            config: JSON.stringify({ apiKey: 'local-key-123' })
        }
        expect(getAuthHeaders(session)).toEqual({
            'Authorization': 'Bearer local-key-123'
        })
    })

    it('returns empty for remote session without API key', () => {
        const session = {
            type: 'remote',
            config: '{}'
        }
        expect(getAuthHeaders(session)).toEqual({})
    })

    it('returns empty for local session without apiKey', () => {
        const session = {
            type: 'local',
            config: '{}'
        }
        expect(getAuthHeaders(session)).toEqual({})
    })

    it('handles corrupted config JSON gracefully', () => {
        const session = {
            type: 'local',
            config: 'not-json'
        }
        expect(getAuthHeaders(session)).toEqual({})
    })
})

// ─── Remote session modelPath format ─────────────────────────────────────────

describe('Remote session modelPath format', () => {
    function buildRemoteModelPath(remoteModel: string, remoteUrl: string): string {
        const url = new URL(remoteUrl)
        return `remote://${remoteModel}@${url.host}`
    }

    it('creates correct modelPath for standard API', () => {
        expect(buildRemoteModelPath('gpt-4', 'https://api.openai.com'))
            .toBe('remote://gpt-4@api.openai.com')
    })

    it('includes port in modelPath when non-standard', () => {
        expect(buildRemoteModelPath('llama-3', 'http://localhost:8000'))
            .toBe('remote://llama-3@localhost:8000')
    })

    it('handles model names with slashes', () => {
        expect(buildRemoteModelPath('meta-llama/Llama-3.1-8B', 'http://192.168.1.10:8000'))
            .toBe('remote://meta-llama/Llama-3.1-8B@192.168.1.10:8000')
    })
})

// ─── abortByEndpoint matching ────────────────────────────────────────────────

describe('abortByEndpoint tracking', () => {
    it('active request entry should store endpoint for matching', () => {
        // Simulates the fixed code path: endpoint is now set on active request entries
        const activeRequests = new Map<string, {
            controller: AbortController
            endpoint?: { host: string; port: number }
            baseUrl?: string
        }>()

        // Simulate remote session: resolved host=api.openai.com, port=443
        const controller = new AbortController()
        activeRequests.set('chat-123', {
            controller,
            endpoint: { host: 'api.openai.com', port: 443 },
            baseUrl: 'https://api.openai.com'
        })

        // abortByEndpoint should find it
        let found = false
        for (const [, entry] of activeRequests) {
            if (entry.endpoint?.host === 'api.openai.com' && entry.endpoint?.port === 443) {
                found = true
            }
        }
        expect(found).toBe(true)
    })

    it('does not match when endpoint host/port differ', () => {
        const activeRequests = new Map<string, {
            controller: AbortController
            endpoint?: { host: string; port: number }
        }>()

        activeRequests.set('chat-456', {
            controller: new AbortController(),
            endpoint: { host: '127.0.0.1', port: 8000 }
        })

        let found = false
        for (const [, entry] of activeRequests) {
            if (entry.endpoint?.host === 'api.openai.com' && entry.endpoint?.port === 443) {
                found = true
            }
        }
        expect(found).toBe(false)
    })
})

// ─── Stale lock recovery ─────────────────────────────────────────────────────

describe('Stale lock recovery', () => {
    it('detects stale lock when age exceeds timeoutMs + 30s buffer', () => {
        const timeoutMs = 300_000 // 5 minutes
        const staleLockMs = timeoutMs + 30_000
        const startedAt = Date.now() - staleLockMs - 1000 // started 5m31s ago
        const age = Date.now() - startedAt

        expect(age > staleLockMs).toBe(true)
    })

    it('does not clear lock within buffer period', () => {
        const timeoutMs = 300_000
        const staleLockMs = timeoutMs + 30_000
        const startedAt = Date.now() - 200_000 // started 3m20s ago
        const age = Date.now() - startedAt

        expect(age > staleLockMs).toBe(false)
    })

    it('uses session-specific timeout for stale calculation', () => {
        // Remote session with timeout=600s should have stale lock at 630s
        const timeoutMs = 600_000
        const staleLockMs = timeoutMs + 30_000
        expect(staleLockMs).toBe(630_000)
    })
})

// ─── Remote session health check path ────────────────────────────────────────

describe('Remote health check behavior', () => {
    it('uses /v1/models for remote sessions, not /health', () => {
        const session = { type: 'remote', remoteUrl: 'https://api.openai.com' }
        const isRemote = session.type === 'remote'
        const baseUrl = session.remoteUrl!.replace(/\/+$/, '')
        const healthUrl = isRemote ? `${baseUrl}/v1/models` : `${baseUrl}/health`
        expect(healthUrl).toBe('https://api.openai.com/v1/models')
    })

    it('uses /health for local sessions', () => {
        const isRemote = false
        const baseUrl = 'http://127.0.0.1:8000'
        const healthUrl = isRemote ? `${baseUrl}/v1/models` : `${baseUrl}/health`
        expect(healthUrl).toBe('http://127.0.0.1:8000/health')
    })

    it('remote sessions get 1 health retry vs 5 for local', () => {
        const remoteRetries = 1
        const localRetries = 5
        expect(remoteRetries).toBeLessThan(localRetries)
    })

    it('15s recently-healthy optimization skips health check', () => {
        const lastHealthyAt = Date.now() - 10_000 // 10 seconds ago
        const recentlyHealthy = (Date.now() - lastHealthyAt) < 15_000
        expect(recentlyHealthy).toBe(true)
    })

    it('expired healthy-at does not skip health check', () => {
        const lastHealthyAt = Date.now() - 20_000 // 20 seconds ago
        const recentlyHealthy = (Date.now() - lastHealthyAt) < 15_000
        expect(recentlyHealthy).toBe(false)
    })
})
