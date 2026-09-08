// SPDX-License-Identifier: Apache-2.0
// API Gateway — single-port reverse proxy for all model sessions.
// Routes OpenAI, Anthropic, and Ollama requests to the correct backend
// by model name. Supports JIT auto-load for stopped models.
//
// Architecture:
//   Client → [Gateway :8080] → route by model field → [Session A :52431]
//                                                    → [Session B :52432]
//   Ollama /api/* endpoints translated to OpenAI format before forwarding.

import { GATEWAY_SINGLE_MODEL_MODE_KEY, isGatewaySettingEnabled } from '../shared/gatewaySettingsKeys'
import {
  ClientRequest,
  createServer,
  IncomingMessage,
  OutgoingHttpHeaders,
  ServerResponse,
  request as httpRequest,
  Server,
} from "http";
import { db } from "./database";
import { sessionManager } from "./sessions";
import { detectModelConfigFromDir } from "./model-config-registry";
import { EventEmitter } from "events";
import { extractGatewayModelFromBody } from "./gateway-body";
import {
  reasoningParserIsEnabled,
  resolveEffectiveReasoningParser,
} from "../shared/reasoningParserAliases";
import {
  resolveEffectiveToolParser,
  toolParserIsEnabled,
} from "../shared/toolParserAliases";
import { SLOW_FAMILY_TIMEOUTS, resolveSlowFamilyTimeoutSeconds } from '../shared/slowFamilyTimeouts';

const DEFAULT_PORT = 8080;
const JIT_TIMEOUT_MS = 120_000;
const HEALTH_POLL_MS = 2_000;
const GENERIC_DEFAULT_TIMEOUT_SECONDS = 300;
const PROXY_TIMEOUT_MS = GENERIC_DEFAULT_TIMEOUT_SECONDS * 1000; // generic 5 min max for a single proxied request
const SINGLE_MODEL_MODE_KEY = GATEWAY_SINGLE_MODEL_MODE_KEY;

interface ResolvedSession {
  id: string;
  host: string;
  port: number;
  status: string;
  modelName: string;
  modelPath: string;
  config?: string;
  servedModelName?: string;
  embeddingModel?: string;
}

type PreparedSession =
  | { status: "ready"; session: ResolvedSession }
  | { status: "unload_failed"; session: ResolvedSession }
  | {
      status: "load_failed";
      session: ResolvedSession;
      code:
        | "model_preflight_failed"
        | "model_load_failed"
        | "model_load_timeout";
      message: string;
    };

type JitLoadResult =
  | { status: "ready" }
  | { status: "load_failed"; detail: string }
  | { status: "timeout" };

type ProxyStreamProtocol = "openai" | "responses" | "anthropic" | "ollama";

function parsePositiveTimeoutSeconds(value: unknown): number | undefined {
  if (typeof value === "number" && Number.isFinite(value)) return value;
  if (typeof value === "string" && value.trim()) {
    const parsed = Number(value);
    if (Number.isFinite(parsed)) return parsed;
  }
  return undefined;
}

function timeoutSecondsToMs(timeoutSeconds: number): number {
  return timeoutSeconds <= 0 ? 86_400_000 : timeoutSeconds * 1000;
}

export class ApiGateway extends EventEmitter {
  private server: Server | null = null;
  private port: number = DEFAULT_PORT;
  private host: string = "127.0.0.1";
  private _running = false;
  /** Track in-flight JIT loads to prevent duplicate starts */
  private jitPending = new Map<string, Promise<JitLoadResult>>();
  /** Serialize cross-model transitions when single-model mode is enabled. */
  private singleModelTransitionPending: Promise<void> = Promise.resolve();
  /** Responses for model/API requests that have not reached finish/close yet. */
  private inFlightResponses = new Set<ServerResponse>();

  get running(): boolean {
    return this._running;
  }
  get activePort(): number {
    return this.port;
  }
  get activeHost(): string {
    return this.host;
  }
  get activeRequestCount(): number {
    return this.inFlightResponses.size;
  }
  get singleModelMode(): boolean {
    return isGatewaySettingEnabled(db.getSetting(SINGLE_MODEL_MODE_KEY));
  }

  setSingleModelMode(enabled: boolean): void {
    db.setSetting(SINGLE_MODEL_MODE_KEY, enabled ? "true" : "false");
    this.emit("singleModelModeChanged", enabled);
  }

  private effectiveGatewayProxyTimeoutMs(
    session: ResolvedSession,
    body?: Buffer | Record<string, any>,
  ): number {
    let sessionTimeout: number | undefined;
    try {
      const cfg = session.config ? JSON.parse(session.config) : {};
      sessionTimeout = parsePositiveTimeoutSeconds(cfg.timeout);
    } catch (_) {
      sessionTimeout = undefined;
    }

    let requestTimeout: number | undefined;
    try {
      const parsed =
        Buffer.isBuffer(body)
          ? JSON.parse(body.toString("utf8") || "{}")
          : body;
      requestTimeout = parsePositiveTimeoutSeconds(
        parsed?.timeout ?? parsed?.options?.timeout,
      );
    } catch (_) {
      requestTimeout = undefined;
    }
    if (requestTimeout != null) return timeoutSecondsToMs(requestTimeout);

    let detectedFamily: string | undefined;
    try {
      detectedFamily = session.modelPath
        ? detectModelConfigFromDir(session.modelPath)?.family
        : undefined;
    } catch (_) {
      detectedFamily = undefined;
    }

    const slowFamilySeconds = detectedFamily
      ? SLOW_FAMILY_TIMEOUTS[detectedFamily]
      : undefined;

    if (sessionTimeout == null && slowFamilySeconds == null) {
      return PROXY_TIMEOUT_MS;
    }

    // Shared table — see panel/src/shared/slowFamilyTimeouts.ts.
    return timeoutSecondsToMs(
      resolveSlowFamilyTimeoutSeconds(sessionTimeout, detectedFamily),
    );
  }

  private sessionApiKey(session: Pick<ResolvedSession, "config">): string | undefined {
    try {
      const cfg = session.config ? JSON.parse(session.config) : {};
      return typeof cfg.apiKey === "string" && cfg.apiKey
        ? cfg.apiKey
        : undefined;
    } catch (_) {
      return undefined;
    }
  }

  private gatewayApiKeys(): Set<string> {
    const keys = new Set<string>();
    for (const s of db.getSessions()) {
      if (s.type === "remote") continue;
      const key = this.sessionApiKey({ config: s.config });
      if (key) keys.add(key);
    }
    return keys;
  }

  private bearerCredential(req: IncomingMessage): string | undefined {
    const value = req.headers.authorization;
    if (!value) return undefined;
    const match = value.match(/^Bearer\s+(.+)$/i);
    return match?.[1]?.trim() || undefined;
  }

  private sendUnauthorized(res: ServerResponse): void {
    this.sendJson(res, 401, {
      error: {
        message: "Invalid or missing API key",
        type: "authentication_error",
        code: "invalid_api_key",
      },
    });
  }

  private requireAnyGatewayAuth(
    req: IncomingMessage,
    res: ServerResponse,
  ): boolean {
    const keys = this.gatewayApiKeys();
    if (keys.size === 0) return true;
    const token = this.bearerCredential(req);
    if (token && keys.has(token)) return true;
    this.sendUnauthorized(res);
    return false;
  }

  private requireSessionAuth(
    req: IncomingMessage,
    res: ServerResponse,
    session: ResolvedSession,
  ): boolean {
    const key = this.sessionApiKey(session);
    if (!key) return true;
    if (this.bearerCredential(req) === key) return true;
    this.sendUnauthorized(res);
    return false;
  }

  // ═══════════════════════════════════════════════════════════════
  // Lifecycle
  // ═══════════════════════════════════════════════════════════════

  async start(port?: number, host?: string): Promise<void> {
    if (this._running) return;
    this.port =
      port ??
      parseInt(db.getSetting("gateway_port") || String(DEFAULT_PORT), 10);
    this.host = host ?? db.getSetting("gateway_host") ?? "127.0.0.1";

    // Reject only if an active local session is already binding this port (#44).
    // Stopped/error rows and remote sessions can retain stale DB ports across
    // app restarts or sleep/wake transitions; they must not block gateway
    // startup because they do not own a local listener.
    const sessions = db.getSessions();
    const sessionPorts = new Set(
      sessions
        .filter(
          (s: any) =>
            s.type !== "remote" &&
            ["running", "loading", "standby"].includes(String(s.status || "")),
        )
        .map((s: any) => s.port),
    );
    if (sessionPorts.has(this.port)) {
      throw new Error(
        `Gateway port ${this.port} conflicts with an existing session. ` +
          `Choose a different port to avoid crashes.`,
      );
    }

    const maxRetries = 10;
    for (let attempt = 0; attempt < maxRetries; attempt++) {
      try {
        await this._tryListen(this.port);
        return;
      } catch (err: any) {
        if (err.code === "EADDRINUSE" && attempt < maxRetries - 1) {
          const nextPort = this.port + 1;
          console.warn(
            `[gateway] Port ${this.port} in use, trying ${nextPort}`,
          );
          this.port = nextPort;
        } else {
          throw err;
        }
      }
    }
  }

  private _tryListen(port: number): Promise<void> {
    return new Promise((resolve, reject) => {
      this.server = createServer((req, res) => {
        this.attachResponseErrorGuard(res);
        this.trackInFlightRequest(req, res);
        this.handleRequest(req, res).catch((err) => {
          this.handleRequestError(err, res);
        });
      });

      this.server.on("error", (err: NodeJS.ErrnoException) => {
        reject(err);
      });

      this.server.listen(port, this.host, () => {
        this._running = true;
        db.setSetting("gateway_port", String(port));
        db.setSetting("gateway_host", this.host);
        console.log(`[gateway] Listening on ${this.host}:${port}`);
        this.emit("started", port);
        resolve();
      });
    });
  }

  private isClientDisconnectError(err: unknown): boolean {
    const anyErr = err as NodeJS.ErrnoException | undefined;
    const code = String(anyErr?.code || "");
    const message = String(anyErr?.message || "");
    const cause = (anyErr as any)?.cause;
    const wrappedDisconnects = [
      cause,
      (anyErr as any)?.reason,
      (anyErr as any)?.error,
      (anyErr as any)?.detail,
    ].filter(Boolean);
    const nestedErrors = Array.isArray((anyErr as any)?.errors)
      ? (anyErr as any).errors
      : [];
    return (
      code === "EPIPE" ||
      code === "ECONNRESET" ||
      code === "ERR_STREAM_DESTROYED" ||
      code === "ERR_STREAM_WRITE_AFTER_END" ||
      /EPIPE|write EPIPE|broken pipe|socket hang up|connection reset|premature close|stream.*destroyed|write after end/i.test(message) ||
      wrappedDisconnects.some((nested) => this.isClientDisconnectError(nested)) ||
      nestedErrors.some((nested) => this.isClientDisconnectError(nested))
    );
  }

  private attachResponseErrorGuard(res: ServerResponse): void {
    res.on("error", (err) => {
      if (this.isClientDisconnectError(err)) {
        return;
      }
      console.error("[gateway] Response stream error:", err);
    });
  }

  private trackInFlightRequest(
    req: IncomingMessage,
    res: ServerResponse,
  ): void {
    const path = (req.url || "/").split("?", 1)[0];
    // Liveness probes must remain available while a model request is active,
    // and should not themselves block an otherwise-idle listener change.
    if (req.method === "GET" && (path === "/" || path === "/health")) return;

    this.inFlightResponses.add(res);
    let released = false;
    const release = () => {
      if (released) return;
      released = true;
      this.inFlightResponses.delete(res);
    };
    res.once("finish", release);
    res.once("close", release);
  }

  private handleRequestError(err: unknown, res: ServerResponse): boolean {
    if (this.isClientDisconnectError(err)) {
      return false;
    }
    console.error("[gateway] Unhandled request error:", err);
    if (!res.headersSent) {
      this.sendJson(res, 500, {
        error: {
          message: "Internal gateway error",
          type: "server_error",
        },
      });
    }
    return true;
  }

  private responseWritable(res: ServerResponse): boolean {
    const anyRes = res as ServerResponse & {
      closed?: boolean;
      writableDestroyed?: boolean;
      socket?: { destroyed?: boolean } | null;
    };
    return (
      !anyRes.closed &&
      !anyRes.destroyed &&
      !anyRes.writableEnded &&
      !anyRes.writableDestroyed &&
      !anyRes.socket?.destroyed
    );
  }

  private requestWritable(req: ClientRequest): boolean {
    const anyReq = req as ClientRequest & {
      closed?: boolean;
      writableDestroyed?: boolean;
      socket?: { destroyed?: boolean } | null;
    };
    return (
      !anyReq.closed &&
      !anyReq.destroyed &&
      !anyReq.writableEnded &&
      !anyReq.writableDestroyed &&
      !anyReq.socket?.destroyed
    );
  }

  private writeProxyBody(req: ClientRequest, chunk: string | Buffer): boolean {
    if (!this.requestWritable(req)) return false;
    try {
      req.write(chunk);
      return true;
    } catch (err) {
      if (this.isClientDisconnectError(err)) return false;
      throw err;
    }
  }

  private endProxyRequest(req: ClientRequest): boolean {
    if (!this.requestWritable(req)) return false;
    try {
      req.end();
      return true;
    } catch (err) {
      if (this.isClientDisconnectError(err)) return false;
      throw err;
    }
  }

  private writeResponse(res: ServerResponse, chunk: string | Buffer): boolean {
    if (!this.responseWritable(res)) return false;
    try {
      res.write(chunk);
      return true;
    } catch (err) {
      if (this.isClientDisconnectError(err)) return false;
      throw err;
    }
  }

  private writeHeadResponse(
    res: ServerResponse,
    status: number,
    headers?: OutgoingHttpHeaders,
  ): boolean {
    if (!this.responseWritable(res)) return false;
    try {
      res.writeHead(status, headers);
      return true;
    } catch (err) {
      if (this.isClientDisconnectError(err)) return false;
      throw err;
    }
  }

  private endResponse(res: ServerResponse, chunk?: string | Buffer): void {
    if (!this.responseWritable(res)) return;
    try {
      if (chunk !== undefined) res.end(chunk);
      else res.end();
    } catch (err) {
      if (!this.isClientDisconnectError(err)) throw err;
    }
  }

  private writeJsonLine(res: ServerResponse, payload: any): boolean {
    return this.writeResponse(res, JSON.stringify(payload) + "\n");
  }

  private abortProxyResponseOnClientClose(
    res: ServerResponse,
    proxyRes: IncomingMessage,
  ): void {
    res.on("close", () => {
      if (!proxyRes.destroyed) proxyRes.destroy();
    });
  }

  private abortProxyRequestOnClientClose(
    res: ServerResponse,
    proxyReq: ClientRequest,
  ): void {
    // Install this before backend response headers exist. Non-streaming model
    // routes do not produce a proxyRes until generation has finished, so only
    // watching proxyRes (or the already-consumed client request body) strands
    // backend inference when the downstream client disconnects mid-generation.
    res.once("close", () => {
      if (!res.writableEnded && !proxyReq.destroyed) proxyReq.destroy();
    });
  }

  private proxyStreamProtocol(path: string | undefined): ProxyStreamProtocol {
    if (path?.startsWith("/v1/responses")) return "responses";
    if (path?.startsWith("/v1/messages")) return "anthropic";
    return "openai";
  }

  private proxyRequestStreams(body: Buffer): boolean {
    try {
      return JSON.parse(body.toString("utf8") || "{}").stream === true;
    } catch (_) {
      return false;
    }
  }

  private emitProxyStreamFailure(
    res: ServerResponse,
    protocol: ProxyStreamProtocol,
    message: string,
  ): void {
    if (!this.responseWritable(res)) return;
    if (protocol === "ollama") {
      if (!this.writeJsonLine(res, { error: message })) return;
      this.endResponse(res);
      return;
    }
    if (protocol === "responses") {
      const error = {
        type: "server_error",
        message,
        code: "backend_connection_closed",
      };
      if (
        !this.writeResponse(
          res,
          `event: error\ndata: ${JSON.stringify({ type: "error", error })}\n\n`,
        )
      )
        return;
      if (
        !this.writeResponse(
          res,
          `event: response.failed\ndata: ${JSON.stringify({
            type: "response.failed",
            response: {
              id: "resp_gateway_backend_closed",
              object: "response",
              status: "failed",
              error,
            },
          })}\n\n`,
        )
      )
        return;
      this.endResponse(res);
      return;
    }
    if (protocol === "anthropic") {
      if (
        !this.writeResponse(
          res,
          `event: error\ndata: ${JSON.stringify({
            type: "error",
            error: { type: "api_error", message },
          })}\n\n`,
        )
      )
        return;
      this.endResponse(res);
      return;
    }
    if (
      !this.writeResponse(
        res,
        `data: ${JSON.stringify({
          error: {
            message,
            type: "server_error",
            code: "backend_connection_closed",
          },
        })}\n\n`,
      )
    )
      return;
    this.endResponse(res);
  }

  private guardProxyResponseLifecycle(
    res: ServerResponse,
    proxyRes: IncomingMessage,
    options: {
      context: string;
      streaming: boolean;
      protocol: ProxyStreamProtocol;
    },
  ): void {
    let settled = false;
    const fail = (cause?: Error): void => {
      if (settled || proxyRes.complete) return;
      settled = true;
      // A downstream disconnect deliberately destroys proxyRes. There is no
      // client left to notify, so do not log or manufacture a backend failure.
      if (!this.responseWritable(res)) return;
      const message = "Backend connection closed before response completed";
      console.error(
        `[gateway] ${options.context} ended prematurely: ${cause?.message || message}`,
      );
      if (options.streaming) {
        this.emitProxyStreamFailure(res, options.protocol, message);
      } else if (!res.headersSent) {
        this.sendJson(res, 502, {
          error:
            options.protocol === "ollama"
              ? message
              : {
                  message,
                  type: "server_error",
                  code: "backend_connection_closed",
                },
        });
      } else {
        // Headers or a partial JSON body already escaped. A second response
        // would be invalid; terminate the HTTP message so clients get a prompt
        // truncated-response error instead of waiting forever.
        res.destroy(new Error(message));
      }
    };

    proxyRes.once("end", () => {
      settled = true;
    });
    proxyRes.once("aborted", () => fail(new Error("backend response aborted")));
    proxyRes.once("error", (error) => fail(error));
    proxyRes.once("close", () => fail(new Error("backend response closed")));
  }

  async stop(): Promise<void> {
    if (!this.server) return;
    return new Promise((resolve) => {
      this.server!.close(() => {
        this._running = false;
        this.server = null;
        this.jitPending.clear();
        console.log("[gateway] Stopped");
        this.emit("stopped");
        resolve();
      });
    });
  }

  async restart(port: number, host?: string): Promise<void> {
    const activeRequests = this.inFlightResponses.size;
    if (activeRequests > 0) {
      throw new Error(
        `Gateway cannot change host or port while ${activeRequests} model ` +
          `request${activeRequests === 1 ? " is" : "s are"} active. ` +
          "Wait for the request to finish or cancel it first.",
      );
    }
    const wasRunning = this._running;
    const previousPort = this.port;
    const previousHost = this.host;
    await this.stop();
    try {
      await this.start(port, host);
    } catch (err) {
      // A rejected settings change must not take down a gateway that was
      // already serving. Restore the last known-good listener and persisted
      // host/port, then surface the original validation/bind error to the UI.
      if (wasRunning) {
        try {
          await this.start(previousPort, previousHost);
        } catch (restoreErr) {
          console.error(
            `[gateway] Failed to restore ${previousHost}:${previousPort} after rejected restart:`,
            restoreErr,
          );
        }
      }
      throw err;
    }
  }

  // ═══════════════════════════════════════════════════════════════
  // Request Router
  // ═══════════════════════════════════════════════════════════════

  private isMcpGatewayRoute(url: string): boolean {
    const path = url.split("?")[0];
    return (
      path === "/v1/mcp/tools" ||
      path === "/v1/mcp/servers" ||
      path === "/v1/mcp/execute"
    );
  }

  private mcpRequiresExplicitModel(modelName?: string): boolean {
    if (modelName) return false;
    const localSessions = db.getSessions().filter((s) => s.type !== "remote");
    return localSessions.length > 1;
  }

  private async handleRequest(
    req: IncomingMessage,
    res: ServerResponse,
  ): Promise<void> {
    const url = req.url || "/";
    const method = req.method || "GET";

    // ── CORS preflight (Open WebUI, browser clients) ──
    if (method === "OPTIONS") {
      if (
        !this.writeHeadResponse(res, 204, {
          "Access-Control-Allow-Origin": "*",
          "Access-Control-Allow-Methods": "GET, POST, PUT, DELETE, OPTIONS, HEAD",
          "Access-Control-Allow-Headers":
            "Content-Type, Authorization, X-Requested-With",
          "Access-Control-Max-Age": "86400",
        })
      )
        return;
      this.endResponse(res);
      return;
    }

    // ── Ollama liveness check (HEAD / or GET /) ──
    if (url === "/" && (method === "HEAD" || method === "GET")) {
      if (!this.writeHeadResponse(res, 200, { "Content-Type": "text/plain" }))
        return;
      if (
        method === "GET" &&
        !this.writeResponse(res, "vMLX Gateway is running")
      )
        return;
      this.endResponse(res);
      return;
    }

    // ── Gateway meta endpoints (no proxy needed) ──
    if (url === "/health" && method === "GET") return this.handleHealth(res);
    if (url === "/v1/models" && method === "GET") {
      if (!this.requireAnyGatewayAuth(req, res)) return;
      return this.handleListModels(res);
    }

    // ── Ollama endpoints ──
    if (url.startsWith("/api/"))
      return this.handleOllamaRoute(req, res, url, method);

    // ── All other routes: read body, resolve model, proxy ──
    const body = await this.readBody(req);
    let modelName: string | undefined;

    // Extract model from body (POST) or query param (GET/DELETE)
    if (method === "POST" && body.length > 0) {
      modelName = extractGatewayModelFromBody(body, req.headers["content-type"]);
    }
    if (!modelName) {
      const capMatch = url.match(/^\/v1\/models\/(.+)\/capabilities(?:\?|$)/);
      if (capMatch?.[1]) {
        try {
          modelName = decodeURIComponent(capMatch[1]);
        } catch (_) {
          modelName = capMatch[1];
        }
      }
    }
    if (!modelName) {
      // Support ?model=X query parameter for GET/DELETE endpoints (cache, MCP, audio voices)
      const qIdx = url.indexOf("?");
      if (qIdx >= 0) {
        const params = new URLSearchParams(url.slice(qIdx));
        modelName = params.get("model") || undefined;
      }
    }

    if (this.isMcpGatewayRoute(url) && this.mcpRequiresExplicitModel(modelName)) {
      return this.sendJson(res, 400, {
        error: {
          message:
            "MCP gateway requests require a model when multiple local sessions exist. Pass ?model=served-name for /v1/mcp/tools and /v1/mcp/servers, or include model in the /v1/mcp/execute body.",
          type: "invalid_request_error",
          code: "model_required",
        },
      });
    }

    // Cancel requests without model field: broadcast to all running backends.
    // Only the backend holding that request ID will actually cancel.
    const isCancel = method === "POST" && /\/cancel\/?$/.test(url);
    if (isCancel && !modelName) {
      if (!this.requireAnyGatewayAuth(req, res)) return;
      const sessions = db
        .getSessions()
        .filter((s: any) => s.status === "running");
      if (sessions.length === 0)
        return this.sendJson(res, 404, { error: "No running models" });
      let accepted = false;
      for (const s of sessions) {
        const host = s.host === "0.0.0.0" ? "127.0.0.1" : s.host;
        try {
          const cancelRes = await new Promise<number>((resolve) => {
            const cancelReq = httpRequest(
              {
                hostname: host,
                port: s.port,
                path: url,
                method: "POST",
                headers: this.jsonProxyHeadersWithAuth(req),
                timeout: 5000,
              },
              (r) => {
                r.resume();
                resolve(r.statusCode || 500);
              },
            );
            cancelReq.on("error", () => resolve(500));
            cancelReq.on("timeout", () => {
              cancelReq.destroy();
              resolve(500);
            });
            if (body.length > 0 && !this.writeProxyBody(cancelReq, body)) {
              resolve(500);
              return;
            }
            if (!this.endProxyRequest(cancelReq)) {
              resolve(500);
            }
          });
          if (cancelRes >= 200 && cancelRes < 300) accepted = true;
        } catch {}
      }
      return this.sendJson(
        res,
        accepted ? 200 : 404,
        accepted
          ? { status: "cancelled" }
          : { error: "Request ID not found on any backend" },
      );
    }

    const session = this.resolveSession(modelName);
    if (!session) {
      const available = this.getAvailableModelNames();
      return this.sendJson(res, 404, {
        error: {
          message: `Model '${modelName || "unknown"}' not found. Available: [${available.join(", ")}]`,
          type: "invalid_request_error",
          code: "model_not_found",
        },
      });
    }

    if (!this.requireSessionAuth(req, res, session)) return;

    const prepared = await this.prepareSessionForRouting(session);
    if (prepared.status === "unload_failed") {
      return this.sendJson(res, 503, {
        error: {
          message: `Model '${session.modelName}' cannot be routed because another local model could not be unloaded`,
          type: "server_error",
          code: "single_model_unload_failed",
        },
        retry_after: 30,
      });
    }
    if (prepared.status === "load_failed") {
      return this.sendJson(res, 503, {
        error: {
          message: prepared.message,
          type: "server_error",
          code: prepared.code,
        },
        retry_after: 30,
      });
    }
    const routedSession = prepared.session;

    // Touch session to prevent idle sleep
    sessionManager.touchSession(routedSession.id);

    return this.proxyRequest(req, res, routedSession, body);
  }

  // ═══════════════════════════════════════════════════════════════
  // Model Resolution
  // ═══════════════════════════════════════════════════════════════

  private resolveSession(modelName?: string): ResolvedSession | undefined {
    const sessions = db.getSessions();
    if (!sessions.length) return undefined;

    // Filter out remote sessions — they proxy through renderer, not this gateway
    const localSessions = sessions.filter((s) => s.type !== "remote");
    if (!localSessions.length) return undefined;

    const candidates: ResolvedSession[] = localSessions.map((s) => {
      let config: any = {};
      try {
        config = JSON.parse(s.config || "{}");
      } catch (_) {}
      return {
        id: s.id,
        host: s.host === "0.0.0.0" ? "127.0.0.1" : s.host,
        port: s.port,
        status: s.status,
        modelName: s.modelName || s.modelPath.split("/").pop() || "",
        modelPath: s.modelPath,
        config: s.config,
        servedModelName: config.servedModelName || undefined,
        embeddingModel: config.embeddingModel || undefined,
      };
    });

    // No model specified — prefer running session, then first available
    if (!modelName) {
      return (
        candidates.find((c) => c.status === "running") ||
        candidates.find((c) => c.status === "standby") ||
        candidates[0]
      );
    }

    const lower = modelName.toLowerCase();
    // Strip Ollama :tag suffix (e.g., "qwen3.5:latest" → "qwen3.5")
    const baseName = lower.split(":")[0];

    // 1. Exact match on servedModelName (user alias — highest priority)
    const byAlias = candidates.find((c) => c.servedModelName === modelName);
    if (byAlias) return byAlias;

    // 2. Exact match on servedModelName (case-insensitive)
    const byAliasCI = candidates.find(
      (c) => c.servedModelName?.toLowerCase() === lower,
    );
    if (byAliasCI) return byAliasCI;

    // 3. Exact match on modelName (basename of path)
    const byName = candidates.find((c) => c.modelName === modelName);
    if (byName) return byName;

    // 4. Exact match on full modelPath
    const byPath = candidates.find((c) => c.modelPath === modelName);
    if (byPath) return byPath;

    // 4b. Exact match on embeddingModel (for --embedding-model sessions)
    const byEmbed = candidates.find((c) => c.embeddingModel === modelName);
    if (byEmbed) return byEmbed;

    // 4c. Case-insensitive embeddingModel match
    const byEmbedCI = candidates.find(
      (c) => c.embeddingModel?.toLowerCase() === lower,
    );
    if (byEmbedCI) return byEmbedCI;

    // 5. Case-insensitive modelName match
    const byNameCI = candidates.find(
      (c) => c.modelName.toLowerCase() === lower,
    );
    if (byNameCI) return byNameCI;

    // 6. Partial — model name contains query or vice versa
    const byPartial = candidates.find(
      (c) =>
        c.modelName.toLowerCase().includes(baseName) ||
        baseName.includes(c.modelName.toLowerCase()),
    );
    if (byPartial) return byPartial;

    // 7. Partial on servedModelName
    const byAliasPartial = candidates.find(
      (c) =>
        c.servedModelName &&
        (c.servedModelName.toLowerCase().includes(baseName) ||
          baseName.includes(c.servedModelName.toLowerCase())),
    );
    if (byAliasPartial) return byAliasPartial;

    // 8. Single-model fallback — if only one session exists, route to it
    if (candidates.length === 1) return candidates[0];

    return undefined;
  }

  private getResolvedSessionById(sessionId: string): ResolvedSession | undefined {
    const s = db.getSession(sessionId);
    if (!s) return undefined;
    let config: any = {};
    try {
      config = JSON.parse(s.config || "{}");
    } catch (_) {}
    return {
      id: s.id,
      host: s.host === "0.0.0.0" ? "127.0.0.1" : s.host,
      port: s.port,
      status: s.status,
      modelName: s.modelName || s.modelPath.split("/").pop() || "",
      modelPath: s.modelPath,
      config: s.config,
      servedModelName: config.servedModelName || undefined,
      embeddingModel: config.embeddingModel || undefined,
    };
  }

  private getAvailableModelNames(): string[] {
    const sessions = db.getSessions();
    const names: string[] = [];
    for (const s of sessions) {
      let config: any = {};
      try {
        config = JSON.parse(s.config || "{}");
      } catch (_) {}
      names.push(
        config.servedModelName ||
          s.modelName ||
          s.modelPath.split("/").pop() ||
          "unknown",
      );
      if (config.embeddingModel && !names.includes(config.embeddingModel)) {
        names.push(config.embeddingModel);
      }
    }
    return names;
  }

  private async enforceSingleModelMode(targetSessionId: string): Promise<boolean> {
    if (!this.singleModelMode) return true;
    const sessions = db.getSessions().filter((s: any) => {
      if (s.id === targetSessionId) return false;
      if (s.type === "remote") return false;
      return ["running", "loading", "standby"].includes(s.status);
    });
    let ok = true;
    for (const s of sessions) {
      try {
        console.log(
          `[gateway] single-model mode: stopping session ${s.id} before routing to ${targetSessionId}`,
        );
        await sessionManager.stopSession(s.id);
      } catch (err) {
        ok = false;
        console.warn(
          `[gateway] single-model mode: failed to stop session ${s.id}: ${err}`,
        );
      }
    }
    return ok;
  }

  private async withSingleModelTransition<T>(fn: () => Promise<T>): Promise<T> {
    if (!this.singleModelMode) return fn();

    const previous = this.singleModelTransitionPending.catch(() => {});
    let release: () => void = () => {};
    const current = new Promise<void>((resolve) => {
      release = resolve;
    });
    this.singleModelTransitionPending = previous.then(() => current);

    await previous;
    try {
      return await fn();
    } finally {
      release();
    }
  }

  private getSingleModelRollbackCandidate(
    targetSessionId: string,
  ): ResolvedSession | undefined {
    if (!this.singleModelMode) return undefined;
    const candidates = db
      .getSessions()
      .filter(
        (candidate: any) =>
          candidate.id !== targetSessionId &&
          candidate.type !== "remote" &&
          ["running", "standby"].includes(candidate.status),
      )
      .map((candidate: any) => this.getResolvedSessionById(candidate.id))
      .filter((candidate): candidate is ResolvedSession => Boolean(candidate));
    return (
      candidates.find((candidate) => candidate.status === "running") ||
      candidates.find((candidate) => candidate.status === "standby")
    );
  }

  private async restoreDisplacedSingleModel(
    candidate: ResolvedSession | undefined,
    failedTargetId: string,
  ): Promise<void> {
    if (!candidate) return;
    try {
      console.warn(
        `[gateway] target ${failedTargetId} failed to load; restoring displaced session ${candidate.id}`,
      );
      await sessionManager.startSession(candidate.id, {
        launchOrigin: "gateway",
      });
      console.log(
        `[gateway] restored displaced session ${candidate.id} after target ${failedTargetId} failed`,
      );
    } catch (error) {
      console.error(
        `[gateway] failed to restore displaced session ${candidate.id} after target ${failedTargetId}: ${error}`,
      );
    }
  }

  private async prepareSessionForRouting(
    session: ResolvedSession,
  ): Promise<PreparedSession> {
    return this.withSingleModelTransition(async () => {
      const target = this.getResolvedSessionById(session.id) || session;
      if (target.status !== "running") {
        try {
          // A stale/malformed target must fail before Single Model mode unloads
          // the currently healthy engine. The session manager repeats this same
          // shared validation at spawn time to cover filesystem races.
          //
          // Do not repeat this launch preflight for an already-running target.
          // In development it resolves the project engine by importing
          // vmlx_engine in a fresh Python process, which otherwise adds seconds
          // of fixed latency to every gateway request even though no process is
          // being launched. Single Model enforcement still runs below.
          await sessionManager.preflightSessionStart(target.id);
        } catch (error) {
          const detail = error instanceof Error ? error.message : String(error);
          console.error(
            `[gateway] target session ${target.id} failed preflight: ${error}`,
          );
          return {
            status: "load_failed",
            session: target,
            code: "model_preflight_failed",
            message: `Model '${target.modelName}' cannot be loaded: ${detail}`,
          };
        }
      }

      const rollbackCandidate = this.getSingleModelRollbackCandidate(target.id);
      const singleModelReady = await this.enforceSingleModelMode(target.id);
      if (!singleModelReady) return { status: "unload_failed", session: target };

      const latest = this.getResolvedSessionById(target.id) || target;
      if (latest.status === "running") {
        return { status: "ready", session: latest };
      }

      const loadResult = await this.jitLoad(latest.id);
      if (loadResult.status !== "ready") {
        await this.restoreDisplacedSingleModel(rollbackCandidate, latest.id);
        const timedOut = loadResult.status === "timeout";
        return {
          status: "load_failed",
          session: latest,
          code: timedOut ? "model_load_timeout" : "model_load_failed",
          message: timedOut
            ? `Model '${latest.modelName}' failed to load within ${JIT_TIMEOUT_MS / 1000}s`
            : `Model '${latest.modelName}' failed to load: ${loadResult.detail}`,
        };
      }

      return {
        status: "ready",
        session: this.getResolvedSessionById(latest.id) || latest,
      };
    });
  }

  // ═══════════════════════════════════════════════════════════════
  // JIT Auto-Load
  // ═══════════════════════════════════════════════════════════════

  private async jitLoad(sessionId: string): Promise<JitLoadResult> {
    // Deduplicate concurrent JIT loads for the same session
    const existing = this.jitPending.get(sessionId);
    if (existing) return existing;

    const promise = this._doJitLoad(sessionId);
    this.jitPending.set(sessionId, promise);
    try {
      return await promise;
    } finally {
      this.jitPending.delete(sessionId);
    }
  }

  private async _doJitLoad(sessionId: string): Promise<JitLoadResult> {
    const session = db.getSession(sessionId);
    const isStandby = session?.status === "standby";
    console.log(
      `[gateway] JIT ${isStandby ? "waking" : "loading"} session ${sessionId}`,
    );
    try {
      if (isStandby) {
        // Session process is alive but model is sleeping — wake it via admin endpoint
        const wakeResult = await sessionManager.wakeSession(sessionId);
        if (!wakeResult.success) {
          throw new Error(wakeResult.error || "wake failed");
        }
      } else {
        // Session process not running — start it
        await sessionManager.startSession(sessionId, {
          launchOrigin: "gateway",
        });
      }
    } catch (err) {
      const detail = err instanceof Error ? err.message : String(err);
      console.error(
        `[gateway] Failed to ${isStandby ? "wake" : "start"} session ${sessionId}: ${err}`,
      );
      return { status: "load_failed", detail };
    }

    const deadline = Date.now() + JIT_TIMEOUT_MS;
    while (Date.now() < deadline) {
      const s = db.getSession(sessionId);
      if (s?.status === "running") return { status: "ready" };
      if (s?.status === "error") {
        return {
          status: "load_failed",
          detail: "engine process entered error state before becoming ready",
        };
      }
      await new Promise((r) => setTimeout(r, HEALTH_POLL_MS));
    }
    console.error(`[gateway] JIT load timeout for session ${sessionId}`);
    return { status: "timeout" };
  }

  // ═══════════════════════════════════════════════════════════════
  // Reverse Proxy
  // ═══════════════════════════════════════════════════════════════

  private jsonProxyHeadersWithAuth(clientReq: IncomingMessage): OutgoingHttpHeaders {
    const headers: OutgoingHttpHeaders = { "Content-Type": "application/json" };
    const authorization = clientReq.headers.authorization;
    if (authorization) headers.Authorization = authorization;
    return headers;
  }

  private proxyRequest(
    clientReq: IncomingMessage,
    clientRes: ServerResponse,
    session: ResolvedSession,
    body: Buffer,
  ): void {
    const options = {
      hostname: session.host,
      port: session.port,
      path: clientReq.url,
      method: clientReq.method,
      headers: {
        ...clientReq.headers,
        host: `${session.host}:${session.port}`,
      },
      timeout: this.effectiveGatewayProxyTimeoutMs(session, body),
    };

    const proxyReq = httpRequest(options, (proxyRes) => {
      const streaming = this.proxyRequestStreams(body);
      this.abortProxyResponseOnClientClose(clientRes, proxyRes);
      this.guardProxyResponseLifecycle(clientRes, proxyRes, {
        context: `Proxy response ${session.host}:${session.port}${clientReq.url}`,
        streaming,
        protocol: this.proxyStreamProtocol(clientReq.url),
      });

      if (!streaming) {
        // Non-stream responses are one JSON document. Do not commit the
        // backend's status/headers or leak body fragments until the upstream
        // message completes; otherwise a reset after `{"partial":` leaves
        // clients with HTTP 200 and truncated JSON. The lifecycle guard can
        // now return one atomic 502 while headers are still uncommitted.
        const chunks: Buffer[] = [];
        proxyRes.on("data", (chunk: Buffer) => {
          chunks.push(Buffer.from(chunk));
        });
        proxyRes.on("end", () => {
          if (
            !this.writeHeadResponse(
              clientRes,
              proxyRes.statusCode || 502,
              proxyRes.headers,
            )
          )
            return;
          this.endResponse(clientRes, Buffer.concat(chunks));
        });
        return;
      }

      // Streaming routes retain byte-for-byte forwarding and native SSE
      // failure terminals.
      if (
        !this.writeHeadResponse(
          clientRes,
          proxyRes.statusCode || 502,
          proxyRes.headers,
        )
      ) {
        proxyRes.destroy();
        return;
      }
      // Forward bytes manually so client disconnects go through the guarded writer.
      // This preserves SSE Content-Type while avoiding noisy write EPIPE crashes.
      proxyRes.on("data", (chunk: Buffer) => {
        if (!this.writeResponse(clientRes, chunk)) {
          proxyRes.destroy();
        }
      });
      proxyRes.on("end", () => {
        this.endResponse(clientRes);
      });
    });

    this.abortProxyRequestOnClientClose(clientRes, proxyReq);

    proxyReq.on("error", (err) => {
      if (this.isClientDisconnectError(err)) {
        if (!this.responseWritable(clientRes)) return;
        if (!clientRes.headersSent) {
          this.sendJson(clientRes, 502, {
            error: {
              message: "Backend connection closed while receiving request",
              type: "server_error",
            },
          });
        }
        return;
      }
      console.error(
        `[gateway] Proxy error → ${session.host}:${session.port}${clientReq.url}: ${err.message}`,
      );
      if (!clientRes.headersSent) {
        this.sendJson(clientRes, 502, {
          error: {
            message: `Backend unavailable: ${err.message}`,
            type: "server_error",
          },
        });
      }
    });

    proxyReq.on("timeout", () => {
      proxyReq.destroy();
      if (!clientRes.headersSent) {
        this.sendJson(clientRes, 504, {
          error: { message: "Backend request timed out", type: "server_error" },
        });
      }
    });

    if (body.length > 0 && !this.writeProxyBody(proxyReq, body)) return;
    if (!this.endProxyRequest(proxyReq)) return;
  }

  // ═══════════════════════════════════════════════════════════════
  // /v1/models — Aggregate all sessions
  // ═══════════════════════════════════════════════════════════════

  private handleListModels(res: ServerResponse): void {
    const sessions = db.getSessions();
    const models: any[] = [];
    const seen = new Set<string>();

    for (const s of sessions) {
      let config: any = {};
      try {
        config = JSON.parse(s.config || "{}");
      } catch (_) {}

      // Primary name: alias if set, otherwise basename
      const primaryName =
        config.servedModelName ||
        s.modelName ||
        s.modelPath.split("/").pop() ||
        "unknown";
      if (!seen.has(primaryName)) {
        seen.add(primaryName);
        models.push({
          id: primaryName,
          object: "model",
          created: Math.floor((s.createdAt || Date.now()) / 1000),
          owned_by: "vmlx-engine",
        });
      }

      // Also list actual model name if alias differs
      const actualName = s.modelName || s.modelPath.split("/").pop() || "";
      if (
        config.servedModelName &&
        actualName &&
        config.servedModelName !== actualName &&
        !seen.has(actualName)
      ) {
        seen.add(actualName);
        models.push({
          id: actualName,
          object: "model",
          created: Math.floor((s.createdAt || Date.now()) / 1000),
          owned_by: "vmlx-engine",
        });
      }

      // List embedding model if configured
      if (config.embeddingModel && !seen.has(config.embeddingModel)) {
        seen.add(config.embeddingModel);
        models.push({
          id: config.embeddingModel,
          object: "model",
          created: Math.floor((s.createdAt || Date.now()) / 1000),
          owned_by: "vmlx-engine",
        });
      }
    }

    // `data` is the OpenAI-standard field every client reads. `models` is an
    // additive mirror: codex 0.150's model-manager deserializes the list
    // expecting a top-level `models` array and logs a refresh error against a
    // pure `data` body ("missing field `models`"). The extra key is ignored by
    // standard OpenAI clients, so this satisfies codex without changing the
    // contract anyone else depends on.
    this.sendJson(res, 200, { object: "list", data: models, models });
  }

  // ═══════════════════════════════════════════════════════════════
  // /health — Gateway + backend status
  // ═══════════════════════════════════════════════════════════════

  private handleHealth(res: ServerResponse): void {
    const sessions = db.getSessions();
    this.sendJson(res, 200, {
      status: "ok",
      gateway_port: this.port,
      active_requests: this.inFlightResponses.size,
      single_model_mode: this.singleModelMode,
      backends: sessions.map((s) => {
        let config: any = {};
        try {
          config = JSON.parse(s.config || "{}");
        } catch (_) {}
        return {
          id: s.id,
          model: config.servedModelName || s.modelName,
          status: s.status,
          port: s.port,
        };
      }),
    });
  }

  // ═══════════════════════════════════════════════════════════════
  // Ollama API Compatibility
  // ═══════════════════════════════════════════════════════════════

  private async handleOllamaRoute(
    req: IncomingMessage,
    res: ServerResponse,
    url: string,
    method: string,
  ): Promise<void> {
    // GET endpoints (no body)
    if (method === "GET") {
      if (url === "/") {
        if (!this.writeHeadResponse(res, 200, { "Content-Type": "text/plain" }))
          return;
        this.endResponse(res, "Ollama is running\n");
        return;
      }
      if (url === "/api/tags") {
        if (!this.requireAnyGatewayAuth(req, res)) return;
        return this.handleOllamaTags(res);
      }
      if (url === "/api/ps") {
        if (!this.requireAnyGatewayAuth(req, res)) return;
        return this.handleOllamaPs(res);
      }
      if (url === "/api/version")
        // mlxstudio#72: Copilot in VS Code gates on version >= 0.6.4. Real
        // Ollama is on 0.12.x; report a plausible recent real version so
        // other version-gated clients don't refuse to connect. Kept in sync
        // with vmlx_engine/server.py:ollama_version.
        return this.sendJson(res, 200, { version: "0.12.6" });
      return this.sendJson(res, 404, { error: "Unknown endpoint" });
    }

    // POST endpoints
    if (method === "POST") {
      if (url === "/api/chat") return this.handleOllamaChat(req, res);
      if (url === "/api/generate") return this.handleOllamaGenerate(req, res);
      if (url === "/api/show") return this.handleOllamaShow(req, res);
      if (url === "/api/embeddings" || url === "/api/embed")
        return this.handleOllamaEmbed(req, res);
      // Unsupported but don't error — return empty success for compat
      if (url === "/api/pull") {
        if (!this.requireAnyGatewayAuth(req, res)) return;
        return this.sendJson(res, 200, { status: "success" });
      }
      if (url === "/api/delete") {
        if (!this.requireAnyGatewayAuth(req, res)) return;
        return this.sendJson(res, 200, { status: "success" });
      }
      if (url === "/api/copy") {
        if (!this.requireAnyGatewayAuth(req, res)) return;
        return this.sendJson(res, 200, { status: "success" });
      }
      if (url === "/api/create") {
        if (!this.requireAnyGatewayAuth(req, res)) return;
        return this.sendJson(res, 200, { status: "success" });
      }
      return this.sendJson(res, 404, { error: "Unknown endpoint" });
    }

    // HEAD for health/version checks from Ollama-compatible clients.
    if (method === "HEAD" && (url === "/" || url === "/api/version")) {
      if (!this.writeHeadResponse(res, 200)) return;
      this.endResponse(res);
      return;
    }

    this.sendJson(res, 405, { error: "Method not allowed" });
  }

  // ── /api/tags ──

  private handleOllamaTags(res: ServerResponse): void {
    const sessions = db.getSessions();
    const models = sessions.map((s) => {
      let config: any = {};
      try {
        config = JSON.parse(s.config || "{}");
      } catch (_) {}
      return {
        name: config.servedModelName || s.modelName || "unknown",
        model: s.modelPath,
        modified_at: new Date(s.updatedAt || Date.now()).toISOString(),
        size: 0,
        digest: "",
        details: {
          format: "mlx",
          family: "",
          parameter_size: "",
          quantization_level: "",
        },
      };
    });
    this.sendJson(res, 200, { models });
  }

  // ── /api/ps ──

  private handleOllamaPs(res: ServerResponse): void {
    const sessions = db.getSessions().filter((s) => s.status === "running");
    const models = sessions.map((s) => {
      let config: any = {};
      try {
        config = JSON.parse(s.config || "{}");
      } catch (_) {}
      return {
        name: config.servedModelName || s.modelName || "unknown",
        model: s.modelPath,
        size: 0,
        digest: "",
        expires_at: new Date(Date.now() + 300_000).toISOString(),
      };
    });
    this.sendJson(res, 200, { models });
  }

  private translateOllamaMessages(messages: any): any[] {
    if (!Array.isArray(messages)) return [];
    return messages.map((msg: any) => {
      if (!msg || typeof msg !== "object" || Array.isArray(msg)) return msg;

      // Ollama names prior assistant private reasoning `thinking`; the shared
      // Chat/Responses history contract names it `reasoning_content`. Normalize
      // before the text-only/media split so the Electron gateway and direct
      // Python Ollama route render identical follow-up prompts.
      const normalized = { ...msg };
      if (
        normalized.role === "assistant" &&
        Object.prototype.hasOwnProperty.call(normalized, "thinking")
      ) {
        const thinking = normalized.thinking;
        delete normalized.thinking;
        if (
          typeof thinking === "string" &&
          thinking.length > 0 &&
          !(typeof normalized.reasoning_content === "string" &&
            normalized.reasoning_content.length > 0)
        ) {
          normalized.reasoning_content = thinking;
        }
      }

      const images = Array.isArray(normalized.images)
        ? normalized.images
        : typeof normalized.images === "string"
          ? [normalized.images]
          : [];
      const videos = Array.isArray(normalized.videos)
        ? normalized.videos
        : typeof normalized.videos === "string"
          ? [normalized.videos]
          : [];
      const rawAudio = normalized.audio ?? normalized.audios;
      const audio = Array.isArray(rawAudio)
        ? rawAudio
        : typeof rawAudio === "string"
          ? [rawAudio]
          : [];
      if (images.length === 0 && videos.length === 0 && audio.length === 0) {
        return normalized;
      }
      const parts: any[] = [];
      const text = normalized.content || "";
      if (text) parts.push({ type: "text", text });
      for (const img of images) {
        if (typeof img !== "string") continue;
        const url = img.startsWith("data:")
          ? img
          : `data:image/png;base64,${img}`;
        parts.push({ type: "image_url", image_url: { url } });
      }
      for (const video of videos) {
        if (typeof video !== "string") continue;
        const url = video.startsWith("data:")
          ? video
          : `data:video/mp4;base64,${video}`;
        parts.push({ type: "video_url", video_url: { url } });
      }
      for (const audioItem of audio) {
        if (typeof audioItem !== "string") continue;
        const url = audioItem.startsWith("data:")
          ? audioItem
          : `data:audio/wav;base64,${audioItem}`;
        parts.push({ type: "audio_url", audio_url: { url } });
      }
      const {
        images: _images,
        videos: _videos,
        audio: _audio,
        audios: _audios,
        content: _content,
        ...rest
      } = normalized;
      return {
        ...rest,
        role: normalized.role || "user",
        content: parts,
      };
    });
  }

  private ollamaResponseFormat(format: any): any | undefined {
    if (format === "json") return { type: "json_object" };
    if (format && typeof format === "object" && !Array.isArray(format)) {
      return {
        type: "json_schema",
        json_schema: {
          name: "ollama_schema",
          strict: false,
          schema: format,
        },
      };
    }
    return undefined;
  }

  private shouldForwardOllamaReasoningEffort(parsed: any, openaiBody: any): boolean {
    if (parsed?.reasoning_effort == null) return false;
    if (openaiBody?.enable_thinking === false) return false;
    const ct = parsed?.chat_template_kwargs;
    if (
      ct &&
      typeof ct === "object" &&
      !Array.isArray(ct) &&
      ct.enable_thinking === false
    ) {
      return false;
    }
    return true;
  }

  private normalizeOllamaBoolean(value: any): boolean | undefined {
    if (typeof value === "boolean") return value;
    if (typeof value === "string") {
      const normalized = value.trim().toLowerCase();
      if (["true", "1", "yes", "on"].includes(normalized)) return true;
      if (["false", "0", "no", "off"].includes(normalized)) return false;
      return undefined;
    }
    if (typeof value === "number" && Number.isFinite(value)) {
      if (value === 1) return true;
      if (value === 0) return false;
    }
    return undefined;
  }

  private applyOllamaThinking(parsed: any, openaiBody: any): void {
    // Omitted thinking controls stay omitted so the model's native
    // tokenizer/template/runtime default decides. Native Ollama `think:false`
    // is an explicit opt-out and must not be overwritten.
    const think = this.normalizeOllamaBoolean(parsed?.think);
    const enableThinking = this.normalizeOllamaBoolean(parsed?.enable_thinking);
    if (think !== undefined) {
      openaiBody.enable_thinking = think;
    } else if (enableThinking !== undefined) {
      openaiBody.enable_thinking = enableThinking;
    } else {
      const ct = parsed?.chat_template_kwargs;
      if (
        ct &&
        typeof ct === "object" &&
        !Array.isArray(ct) &&
        ct.enable_thinking === false
      ) {
        return;
      }
    }
  }

  private applyOllamaPromptContextLimit(parsed: any, opts: any, openaiBody: any): void {
    const value =
      opts?.num_ctx ??
      opts?.num_context ??
      opts?.max_prompt_tokens ??
      opts?.max_context_tokens ??
      opts?.max_context ??
      parsed?.max_prompt_tokens ??
      parsed?.max_context_tokens ??
      parsed?.max_context;
    if (value !== undefined && value !== null) {
      const parsedValue = Number(value);
      if (Number.isFinite(parsedValue) && parsedValue > 0) {
        openaiBody.max_prompt_tokens = Math.floor(parsedValue);
      }
    }
  }

  private applyOllamaNumPredict(opts: any, openaiBody: any): void {
    const value = opts?.num_predict;
    if (value === undefined || value === null) return;
    const parsed = Number(value);
    if (Number.isFinite(parsed) && parsed > 0) {
      openaiBody.max_tokens = Math.floor(parsed);
    }
  }

  private openAIToolCallsToOllama(
    toolCalls: any[] | undefined | null,
  ): any[] | undefined {
    if (!toolCalls) return undefined;
    const converted = toolCalls
      .filter((tc: any) => tc?.function?.name)
      .map((tc: any) => {
        let args: any = tc.function.arguments;
        if (typeof args === "string") {
          try {
            args = args.length > 0 ? JSON.parse(args) : {};
          } catch {
            args = { _raw: args };
          }
        } else if (args == null) {
          args = {};
        }
        return {
          function: { name: tc.function.name, arguments: args },
        };
      });
    return converted.length > 0 ? converted : undefined;
  }

  private openAIStreamErrorToOllama(payload: any): string | undefined {
    const error = payload?.error;
    const message =
      typeof error === "string"
        ? error
        : error?.message || error?.detail || error?.type;
    if (message == null) return undefined;
    const text = String(message).trim();
    return text || "the model failed to generate a response";
  }

  // ── /api/chat ──

  private async handleOllamaChat(
    req: IncomingMessage,
    res: ServerResponse,
  ): Promise<void> {
    const body = await this.readBody(req);
    if (body.length === 0) return this.sendJson(res, 400, { error: "Empty request body" });
    const bodyText = body.toString("utf8");

    let parsed: any;
    try {
      parsed = JSON.parse(bodyText);
    } catch (_) {
      return this.sendJson(res, 400, { error: "Invalid JSON" });
    }

    const session = this.resolveSession(parsed.model);
    if (!session) {
      return this.sendJson(res, 404, {
        error: `model '${parsed.model || "unknown"}' not found`,
      });
    }
    if (!this.requireSessionAuth(req, res, session)) return;

    const prepared = await this.prepareSessionForRouting(session);
    if (prepared.status === "unload_failed") {
      return this.sendJson(res, 503, {
        error: "Another local model could not be unloaded",
        code: "single_model_unload_failed",
      });
    }
    if (prepared.status === "load_failed") {
      return this.sendJson(res, 503, {
        error: prepared.message,
        code: prepared.code,
      });
    }
    const routedSession = prepared.session;

    sessionManager.touchSession(routedSession.id);

    // Translate Ollama → OpenAI
    const opts = parsed.options || {};
    const openaiBody: any = {
      model: parsed.model || routedSession.modelName,
      messages: this.translateOllamaMessages(parsed.messages || []),
      stream: parsed.stream !== false,
      stream_options: { include_usage: true },
    };
    this.applyOllamaNumPredict(opts, openaiBody);
    if (opts.temperature != null) openaiBody.temperature = opts.temperature;
    if (opts.top_p != null) openaiBody.top_p = opts.top_p;
    // Ollama top_k=0 explicitly disables top-k. Forward it so the local engine
    // does not silently re-inherit a non-zero bundle default; negative Ollama
    // sentinels remain omitted because the OpenAI-compatible engine rejects them.
    if (opts.top_k != null && Number(opts.top_k) >= 0) openaiBody.top_k = opts.top_k;
    if (opts.min_p != null) openaiBody.min_p = opts.min_p;
    if (opts.stop) openaiBody.stop = opts.stop;
    if (opts.repeat_penalty != null)
      openaiBody.repetition_penalty = opts.repeat_penalty;
    // Ollama accepts the seed in options or at the top level; the Python route
    // honours both. Dropping it here made the identical request reproducible
    // against the engine port and silently non-deterministic through the
    // gateway.
    const seedValue = opts.seed ?? parsed.seed;
    if (seedValue != null) openaiBody.seed = seedValue;
    this.applyOllamaPromptContextLimit(parsed, opts, openaiBody);
    if (parsed.tools) openaiBody.tools = parsed.tools;
    if (parsed.cache_salt != null) openaiBody.cache_salt = parsed.cache_salt;
    if (parsed.skip_prefix_cache != null)
      openaiBody.skip_prefix_cache = parsed.skip_prefix_cache;
    this.applyOllamaThinking(parsed, openaiBody);
    if (this.shouldForwardOllamaReasoningEffort(parsed, openaiBody))
      openaiBody.reasoning_effort = parsed.reasoning_effort;
    if (
      parsed.chat_template_kwargs &&
      typeof parsed.chat_template_kwargs === "object" &&
      !Array.isArray(parsed.chat_template_kwargs)
    ) {
      openaiBody.chat_template_kwargs = parsed.chat_template_kwargs;
    }
    const responseFormat = this.ollamaResponseFormat(parsed.format);
    if (responseFormat) openaiBody.response_format = responseFormat;

    const isStreaming = parsed.stream !== false;
    const modelForResponse = parsed.model || routedSession.modelName;

    const proxyOpts = {
      hostname: routedSession.host,
      port: routedSession.port,
      path: "/v1/chat/completions",
      method: "POST",
      headers: this.jsonProxyHeadersWithAuth(req),
      timeout: this.effectiveGatewayProxyTimeoutMs(routedSession, parsed),
    };

    const proxyReq = httpRequest(proxyOpts, (proxyRes) => {
      this.abortProxyResponseOnClientClose(res, proxyRes);
      this.guardProxyResponseLifecycle(res, proxyRes, {
        context: `Ollama chat proxy response ${routedSession.host}:${routedSession.port}`,
        streaming: isStreaming,
        protocol: "ollama",
      });
      if (!isStreaming) {
        // Non-streaming: buffer, translate, send
        let data = "";
        proxyRes.on("data", (chunk: Buffer) => {
          data += chunk.toString();
        });
        proxyRes.on("end", () => {
          try {
            const openai = JSON.parse(data);
            if ((proxyRes.statusCode || 200) >= 400) {
              return this.sendOllamaBackendError(res, proxyRes.statusCode || 502, openai);
            }
            const choice = openai.choices?.[0];
            const response: any = {
              model: modelForResponse,
              created_at: new Date().toISOString(),
              message: {
                role: "assistant",
                content: choice?.message?.content || "",
              },
              done: true,
              done_reason: choice?.finish_reason || "stop",
              total_duration: 0,
              eval_count: openai.usage?.completion_tokens || 0,
              prompt_eval_count: openai.usage?.prompt_tokens || 0,
            };
            const reasoning =
              choice?.message?.reasoning_content || choice?.message?.reasoning;
            if (reasoning) response.message.thinking = reasoning;
            const convertedToolCalls = this.openAIToolCallsToOllama(
              choice?.message?.tool_calls,
            );
            if (convertedToolCalls)
              response.message.tool_calls = convertedToolCalls;
            this.sendJson(res, 200, response);
          } catch (_) {
            this.sendJson(res, 502, {
              error: "Failed to parse backend response",
            });
          }
        });
      } else {
        if ((proxyRes.statusCode || 200) >= 400) {
          let data = "";
          proxyRes.on("data", (chunk: Buffer) => {
            data += chunk.toString();
          });
          proxyRes.on("end", () => {
            this.sendOllamaBackendError(res, proxyRes.statusCode || 502, data);
          });
          return;
        }
        // Streaming: SSE → NDJSON
        if (
          !this.writeHeadResponse(res, 200, {
            "Content-Type": "application/x-ndjson",
            "Transfer-Encoding": "chunked",
          })
        ) {
          proxyRes.destroy();
          return;
        }

        let buffer = "";
        let toolCalls: any[] | null = null;
        let doneReason: string | null = null;
        let done = false;
        let usage: {
          completion_tokens?: number;
          prompt_tokens?: number;
        } | null = null;

        const emitTerminal = (): void => {
          if (done || !this.responseWritable(res)) return;
          done = true;
          const ollamaMsg: any = {
            model: modelForResponse,
            created_at: new Date().toISOString(),
            message: { role: "assistant", content: "" },
            done: true,
            done_reason:
              doneReason === "tool_calls" ? "tool_calls" : doneReason || "stop",
          };
          // Ollama streams content and thinking incrementally. Its terminal
          // chat object carries an empty message plus statistics; repeating
          // either accumulated rail makes conforming clients concatenate it.
          if (toolCalls) {
            const convertedToolCalls = this.openAIToolCallsToOllama(toolCalls);
            if (convertedToolCalls) ollamaMsg.message.tool_calls = convertedToolCalls;
          }
          if (usage) {
            ollamaMsg.eval_count = usage.completion_tokens;
            ollamaMsg.prompt_eval_count = usage.prompt_tokens;
          }
          if (!this.writeJsonLine(res, ollamaMsg)) {
            proxyRes.destroy();
            return;
          }
          this.endResponse(res);
        };

        proxyRes.on("data", (chunk: Buffer) => {
          buffer += chunk.toString();
          const lines = buffer.split("\n");
          buffer = lines.pop() || "";

          for (const line of lines) {
            const trimmed = line.trim();
            if (!trimmed.startsWith("data: ")) continue;
            const payload = trimmed.slice(6);

            if (payload === "[DONE]") {
              emitTerminal();
              continue;
            }

            try {
              const parsed = JSON.parse(payload);
              const streamError = this.openAIStreamErrorToOllama(parsed);
              if (streamError) {
                // The upstream HTTP status is already 200 once an SSE stream
                // begins. Preserve its structured terminal failure using
                // Ollama's native NDJSON error row and suppress the later
                // [DONE] marker; emitting done:true/stop here would falsely
                // report a failed reasoning-only turn as successful.
                done = true;
                if (!this.writeJsonLine(res, { error: streamError })) {
                  proxyRes.destroy();
                  return;
                }
                this.endResponse(res);
                return;
              }
              const delta = parsed.choices?.[0]?.delta;
              const finishReason = parsed.choices?.[0]?.finish_reason;
              const reasoningDelta =
                delta?.reasoning_content || delta?.reasoning;
              if (parsed.usage) {
                usage = {
                  completion_tokens: parsed.usage.completion_tokens || 0,
                  prompt_tokens: parsed.usage.prompt_tokens || 0,
                };
              }

              // tool_calls may arrive in one delta chunk or be fragmented
              // across several (OpenAI streams function name in the first
              // chunk, arguments incrementally). We accumulate per-index
              // and collapse to Ollama format at the finish boundary.
              // mlxstudio#72: Ollama expects `arguments` as an object, not
              // a JSON-encoded string — we parse on finish.
              if (delta?.tool_calls) {
                if (!toolCalls) toolCalls = [];
                for (const tc of delta.tool_calls) {
                  // Resolve target slot. Most vMLX-emitted chunks carry an
                  // explicit `index`, but some providers (and the CLI tools
                  // path) omit it — in that case fall back to "continue the
                  // last slot we opened", defaulting to slot 0 on the very
                  // first fragment. NEVER use `toolCalls.length - 1` as a
                  // default when the array is empty — that's -1, and JS
                  // array[-1] silently creates a string-keyed prop.
                  let idx: number;
                  if (typeof tc.index === "number" && tc.index >= 0) {
                    idx = tc.index;
                  } else if (toolCalls.length > 0) {
                    idx = toolCalls.length - 1;
                  } else {
                    idx = 0;
                  }
                  const slot = toolCalls[idx] || {
                    function: { name: "", arguments: "" },
                  };
                  if (tc.function?.name) slot.function.name = tc.function.name;
                  if (tc.function?.arguments != null) {
                    slot.function.arguments =
                      (slot.function.arguments || "") + tc.function.arguments;
                  }
                  toolCalls[idx] = slot;
                }
              }

              if (finishReason != null) {
                doneReason = finishReason || "stop";
              }
              if (delta?.content || reasoningDelta) {
                const ollamaMsg: any = {
                  model: modelForResponse,
                  created_at: new Date().toISOString(),
                  message: {
                    role: "assistant",
                    content: delta?.content || "",
                  },
                  done: false,
                };
                if (reasoningDelta) ollamaMsg.message.thinking = reasoningDelta;
                if (!this.writeJsonLine(res, ollamaMsg)) {
                  proxyRes.destroy();
                  return;
                }
              }
            } catch (_) {
              /* skip malformed chunks */
            }
          }
        });

        proxyRes.on("end", () => {
          emitTerminal();
        });
      }
    });

    this.abortProxyRequestOnClientClose(res, proxyReq);

    proxyReq.on("error", (err) => {
      if (this.isClientDisconnectError(err)) {
        if (!this.responseWritable(res)) return;
        if (!res.headersSent)
          this.sendJson(res, 502, {
            error: "Backend connection closed while receiving request",
          });
        return;
      }
      if (!res.headersSent)
        this.sendJson(res, 502, {
          error: `Backend unavailable: ${err.message}`,
        });
    });
    proxyReq.on("timeout", () => {
      proxyReq.destroy();
      if (!res.headersSent)
        this.sendJson(res, 504, { error: "Request timed out" });
    });
    if (!this.writeProxyBody(proxyReq, JSON.stringify(openaiBody))) return;
    if (!this.endProxyRequest(proxyReq)) return;
  }

  // ── /api/generate ──

  private async handleOllamaGenerate(
    req: IncomingMessage,
    res: ServerResponse,
  ): Promise<void> {
    const body = await this.readBody(req);
    if (body.length === 0) return this.sendJson(res, 400, { error: "Empty request body" });
    const bodyText = body.toString("utf8");

    let parsed: any;
    try {
      parsed = JSON.parse(bodyText);
    } catch (_) {
      return this.sendJson(res, 400, { error: "Invalid JSON" });
    }

    const session = this.resolveSession(parsed.model);
    if (!session)
      return this.sendJson(res, 404, {
        error: `model '${parsed.model || "unknown"}' not found`,
      });
    if (!this.requireSessionAuth(req, res, session)) return;

    const prepared = await this.prepareSessionForRouting(session);
    if (prepared.status === "unload_failed") {
      return this.sendJson(res, 503, {
        error: "Another local model could not be unloaded",
        code: "single_model_unload_failed",
      });
    }
    if (prepared.status === "load_failed") {
      return this.sendJson(res, 503, {
        error: prepared.message,
        code: prepared.code,
      });
    }
    const routedSession = prepared.session;

    sessionManager.touchSession(routedSession.id);

    const opts = parsed.options || {};
    const isStreaming = parsed.stream !== false;
    const useRawCompletion = parsed.raw === true;
    const openaiBody: any = useRawCompletion
      ? {
          model: parsed.model || routedSession.modelName,
          prompt: parsed.prompt || "",
          stream: isStreaming,
        }
      : {
          model: parsed.model || routedSession.modelName,
          messages: [
            ...(typeof parsed.system === "string" && parsed.system
              ? [{ role: "system", content: parsed.system }]
              : []),
            // Ollama's /api/generate carries media at the TOP LEVEL of the
            // body, not inside a message. Building a plain string here dropped
            // it silently: a vision request became text-only and the model
            // answered about nothing, with no error — the same regression the
            // Python route already fixed. Route the synthetic user message
            // through the shared translator so the two paths cannot drift
            // apart again. Raw mode is text-only by definition and is
            // unaffected (it targets /v1/completions above).
            ...this.translateOllamaMessages([
              {
                role: "user",
                content: parsed.prompt || "",
                images: parsed.images,
                videos: parsed.videos,
                audio: parsed.audio ?? parsed.audios,
              },
            ]),
          ],
          stream: isStreaming,
          stream_options: { include_usage: true },
        };
    this.applyOllamaNumPredict(opts, openaiBody);
    if (opts.temperature != null) openaiBody.temperature = opts.temperature;
    if (opts.top_p != null) openaiBody.top_p = opts.top_p;
    if (opts.top_k != null && Number(opts.top_k) >= 0) openaiBody.top_k = opts.top_k;
    if (opts.min_p != null) openaiBody.min_p = opts.min_p;
    if (opts.stop) openaiBody.stop = opts.stop;
    if (opts.repeat_penalty != null)
      openaiBody.repetition_penalty = opts.repeat_penalty;
    // Ollama accepts the seed in options or at the top level; the Python route
    // honours both. Dropping it here made the identical request reproducible
    // against the engine port and silently non-deterministic through the
    // gateway.
    const seedValue = opts.seed ?? parsed.seed;
    if (seedValue != null) openaiBody.seed = seedValue;
    this.applyOllamaPromptContextLimit(parsed, opts, openaiBody);
    if (parsed.cache_salt != null) openaiBody.cache_salt = parsed.cache_salt;
    if (parsed.skip_prefix_cache != null)
      openaiBody.skip_prefix_cache = parsed.skip_prefix_cache;
    if (!useRawCompletion) this.applyOllamaThinking(parsed, openaiBody);
    if (this.shouldForwardOllamaReasoningEffort(parsed, openaiBody))
      openaiBody.reasoning_effort = parsed.reasoning_effort;
    if (
      parsed.chat_template_kwargs &&
      typeof parsed.chat_template_kwargs === "object" &&
      !Array.isArray(parsed.chat_template_kwargs)
    ) {
      openaiBody.chat_template_kwargs = parsed.chat_template_kwargs;
    }
    const responseFormat = this.ollamaResponseFormat(parsed.format);
    if (responseFormat) openaiBody.response_format = responseFormat;

    const modelForResponse = parsed.model || routedSession.modelName;
    const backendPath = useRawCompletion
      ? "/v1/completions"
      : "/v1/chat/completions";

    const proxyOpts = {
      hostname: routedSession.host,
      port: routedSession.port,
      path: backendPath,
      method: "POST",
      headers: this.jsonProxyHeadersWithAuth(req),
      timeout: this.effectiveGatewayProxyTimeoutMs(routedSession, parsed),
    };

    const proxyReq = httpRequest(proxyOpts, (proxyRes) => {
      this.abortProxyResponseOnClientClose(res, proxyRes);
      this.guardProxyResponseLifecycle(res, proxyRes, {
        context: `Ollama generate proxy response ${routedSession.host}:${routedSession.port}`,
        streaming: isStreaming,
        protocol: "ollama",
      });
      if (!isStreaming) {
        // Non-streaming: buffer, translate, send
        let data = "";
        proxyRes.on("data", (chunk: Buffer) => {
          data += chunk.toString();
        });
        proxyRes.on("end", () => {
          try {
            const openai = JSON.parse(data);
            if ((proxyRes.statusCode || 200) >= 400) {
              return this.sendOllamaBackendError(res, proxyRes.statusCode || 502, openai);
            }
            const choice = openai.choices?.[0];
            const text = useRawCompletion
              ? choice?.text || ""
              : choice?.message?.content || "";
            const thinking = useRawCompletion
              ? undefined
              : choice?.message?.reasoning_content || choice?.message?.reasoning;
            this.sendJson(res, 200, {
              model: modelForResponse,
              created_at: new Date().toISOString(),
              response: text,
              ...(thinking ? { thinking } : {}),
              done: true,
              done_reason: choice?.finish_reason || "stop",
              eval_count: openai.usage?.completion_tokens || 0,
              prompt_eval_count: openai.usage?.prompt_tokens || 0,
            });
          } catch (_) {
            this.sendJson(res, 502, {
              error: "Failed to parse backend response",
            });
          }
        });
      } else {
        if ((proxyRes.statusCode || 200) >= 400) {
          let data = "";
          proxyRes.on("data", (chunk: Buffer) => {
            data += chunk.toString();
          });
          proxyRes.on("end", () => {
            this.sendOllamaBackendError(res, proxyRes.statusCode || 502, data);
          });
          return;
        }
        // Streaming: SSE → NDJSON (same pattern as handleOllamaChat)
        if (
          !this.writeHeadResponse(res, 200, {
            "Content-Type": "application/x-ndjson",
            "Transfer-Encoding": "chunked",
          })
        ) {
          proxyRes.destroy();
          return;
        }

        let buffer = "";
        let streamDone = false;
        let doneReason: string | null = null;
        let usage: { completion_tokens?: number; prompt_tokens?: number } | null = null;
        // finish_reason may precede the choices-empty usage event. Finish the
        // downstream stream only after consuming the upstream terminal marker.
        const emitTerminal = (): void => {
          if (streamDone || !this.responseWritable(res)) return;
          streamDone = true;
          const terminal: any = {
            model: modelForResponse,
            created_at: new Date().toISOString(),
            response: "",
            done: true,
            done_reason: doneReason || "stop",
          };
          if (usage) {
            terminal.eval_count = usage.completion_tokens;
            terminal.prompt_eval_count = usage.prompt_tokens;
          }
          if (!this.writeJsonLine(res, terminal)) {
            proxyRes.destroy();
            return;
          }
          this.endResponse(res);
        };
        proxyRes.on("data", (chunk: Buffer) => {
          if (streamDone) return;
          buffer += chunk.toString();
          const lines = buffer.split("\n");
          buffer = lines.pop() || "";

          for (const line of lines) {
            const trimmed = line.trim();
            if (!trimmed.startsWith("data: ")) continue;
            const payload = trimmed.slice(6);

            if (payload === "[DONE]") {
              emitTerminal();
              return;
            }

            try {
              const chunk = JSON.parse(payload);
              const streamError = this.openAIStreamErrorToOllama(chunk);
              if (streamError) {
                streamDone = true;
                if (!this.writeJsonLine(res, { error: streamError })) {
                  proxyRes.destroy();
                  return;
                }
                this.endResponse(res);
                return;
              }
              const choice = chunk.choices?.[0];
              const text = useRawCompletion
                ? choice?.text || ""
                : choice?.delta?.content || "";
              const thinking = useRawCompletion
                ? ""
                : choice?.delta?.reasoning_content || choice?.delta?.reasoning || "";
              const finishReason = choice?.finish_reason;
              if (finishReason != null) doneReason = finishReason;
              if (chunk.usage) {
                usage = {
                  completion_tokens: chunk.usage.completion_tokens,
                  prompt_tokens: chunk.usage.prompt_tokens,
                };
              }
              if (!text && !thinking) continue;

              const ollamaChunk: any = {
                model: modelForResponse,
                created_at: new Date().toISOString(),
                response: text,
                ...(thinking ? { thinking } : {}),
                done: false,
              };
              if (!this.writeJsonLine(res, ollamaChunk)) {
                proxyRes.destroy();
                return;
              }
            } catch (_) {
              /* skip malformed chunks */
            }
          }
        });

        proxyRes.on("end", emitTerminal);
      }
    });

    this.abortProxyRequestOnClientClose(res, proxyReq);

    proxyReq.on("error", (err) => {
      if (this.isClientDisconnectError(err)) {
        if (!this.responseWritable(res)) return;
        if (!res.headersSent)
          this.sendJson(res, 502, {
            error: "Backend connection closed while receiving request",
          });
        return;
      }
      if (!res.headersSent)
        this.sendJson(res, 502, {
          error: `Backend unavailable: ${err.message}`,
        });
    });
    proxyReq.on("timeout", () => {
      proxyReq.destroy();
      if (!res.headersSent) this.sendJson(res, 504, { error: "Timed out" });
    });
    if (!this.writeProxyBody(proxyReq, JSON.stringify(openaiBody))) return;
    if (!this.endProxyRequest(proxyReq)) return;
  }

  // ── /api/show ──

  private async handleOllamaShow(
    req: IncomingMessage,
    res: ServerResponse,
  ): Promise<void> {
    const body = await this.readBody(req);
    const bodyText = body.toString("utf8");
    let parsed: any;
    try {
      parsed = JSON.parse(bodyText || "{}");
    } catch (_) {
      return this.sendJson(res, 400, { error: "Invalid JSON" });
    }

    const session = this.resolveSession(parsed.name || parsed.model);
    if (!session) return this.sendJson(res, 404, { error: "model not found" });
    if (!this.requireSessionAuth(req, res, session)) return;

    // mlxstudio#72 — Copilot (Ollama spec v0.20.x) gates on `capabilities`.
    // Compute from the saved session config; fall back to permissive defaults.
    let cfg: any = {};
    try {
      cfg = session.config ? JSON.parse(session.config) : {};
    } catch (_) {
      cfg = {};
    }
    let detectedConfig: ReturnType<typeof detectModelConfigFromDir> | null = null;
    try {
      detectedConfig = session.modelPath
        ? detectModelConfigFromDir(session.modelPath)
        : null;
    } catch (_) {
      detectedConfig = null;
    }
    const capabilities: string[] = ["completion"];
    const effectiveToolParser = resolveEffectiveToolParser({
      configuredParser: cfg.toolCallParser ?? cfg.toolParser,
      detectedParser: detectedConfig?.toolParser,
    });
    if (toolParserIsEnabled(effectiveToolParser)) capabilities.push("tools");
    const detectedForceTextOnly = detectedConfig?.forceTextOnly === true;
    if (!detectedForceTextOnly && (cfg.isMultimodal === true || cfg.modelType === "vlm"))
      capabilities.push("vision");
    const effectiveReasoningParser = resolveEffectiveReasoningParser({
      configuredParser: cfg.reasoningParser,
      detectedParser: detectedConfig?.reasoningParser,
      supportsThinking: detectedConfig?.supportsThinking,
    });
    if (reasoningParserIsEnabled(effectiveReasoningParser))
      capabilities.push("thinking");
    capabilities.push("insert");

    const modelName = session.servedModelName || session.modelName;
    this.sendJson(res, 200, {
      modelfile: "",
      parameters: "",
      template: "",
      details: {
        parent_model: "",
        format: "mlx",
        family: (cfg.family || cfg.modelFamily || "mlx") as string,
        families: [(cfg.family || cfg.modelFamily || "mlx") as string],
        parameter_size: cfg.parameterSize || "",
        quantization_level:
          cfg.quantization || cfg.bits
            ? String(cfg.quantization || `${cfg.bits}bit`)
            : "",
      },
      model_info: { name: modelName },
      capabilities,
    });
  }

  // ── /api/embeddings ──

  private async handleOllamaEmbed(
    req: IncomingMessage,
    res: ServerResponse,
  ): Promise<void> {
    const body = await this.readBody(req);
    const bodyText = body.toString("utf8");
    let parsed: any;
    try {
      parsed = JSON.parse(bodyText || "{}");
    } catch (_) {
      return this.sendJson(res, 400, { error: "Invalid JSON" });
    }

    const session = this.resolveSession(parsed.model);
    if (!session) return this.sendJson(res, 404, { error: "model not found" });
    if (!this.requireSessionAuth(req, res, session)) return;

    const prepared = await this.prepareSessionForRouting(session);
    if (prepared.status === "unload_failed") {
      return this.sendJson(res, 503, {
        error: "Another local model could not be unloaded",
        code: "single_model_unload_failed",
      });
    }
    if (prepared.status === "load_failed") {
      return this.sendJson(res, 503, {
        error: prepared.message,
        code: prepared.code,
      });
    }
    const routedSession = prepared.session;

    sessionManager.touchSession(routedSession.id);

    // Translate Ollama embeddings → OpenAI
    const openaiBody = JSON.stringify({
      model: parsed.model,
      input: parsed.input || parsed.prompt || "",
    });

    const proxyOpts = {
      hostname: routedSession.host,
      port: routedSession.port,
      path: "/v1/embeddings",
      method: "POST",
      headers: this.jsonProxyHeadersWithAuth(req),
      timeout: this.effectiveGatewayProxyTimeoutMs(routedSession, parsed),
    };

    const proxyReq = httpRequest(proxyOpts, (proxyRes) => {
      this.abortProxyResponseOnClientClose(res, proxyRes);
      this.guardProxyResponseLifecycle(res, proxyRes, {
        context: `Ollama embeddings proxy response ${routedSession.host}:${routedSession.port}`,
        streaming: false,
        protocol: "ollama",
      });
      let data = "";
      proxyRes.on("data", (chunk: Buffer) => {
        data += chunk.toString();
      });
      proxyRes.on("end", () => {
        try {
          const openai = JSON.parse(data);
          // Ollama format: { embeddings: [[...]], model: "...", total_duration: ... }
          const embeddings = openai.data?.map((d: any) => d.embedding) || [];
          this.sendJson(res, 200, {
            model: parsed.model,
            embeddings,
            total_duration: 0,
          });
        } catch (_) {
          this.sendJson(res, 502, {
            error: "Failed to parse backend response",
          });
        }
      });
    });

    this.abortProxyRequestOnClientClose(res, proxyReq);

    proxyReq.on("error", (err) => {
      if (this.isClientDisconnectError(err)) {
        if (!this.responseWritable(res)) return;
        if (!res.headersSent)
          this.sendJson(res, 502, {
            error: "Backend connection closed while receiving request",
          });
        return;
      }
      if (!res.headersSent)
        this.sendJson(res, 502, { error: "Backend unavailable" });
    });
    proxyReq.on("timeout", () => {
      proxyReq.destroy();
      if (!res.headersSent) this.sendJson(res, 504, { error: "Timed out" });
    });
    if (!this.writeProxyBody(proxyReq, openaiBody)) return;
    if (!this.endProxyRequest(proxyReq)) return;
  }

  // ═══════════════════════════════════════════════════════════════
  // Utilities
  // ═══════════════════════════════════════════════════════════════

  private readBody(req: IncomingMessage): Promise<Buffer> {
    return new Promise((resolve) => {
      const chunks: Buffer[] = [];
      req.on("data", (chunk: Buffer) => chunks.push(chunk));
      req.on("end", () => resolve(Buffer.concat(chunks)));
      req.on("error", () => resolve(Buffer.alloc(0)));
    });
  }

  private sendJson(res: ServerResponse, status: number, data: any): void {
    const json = JSON.stringify(data);
    if (!this.responseWritable(res)) return;
    try {
      if (
        !this.writeHeadResponse(res, status, {
          "Content-Type": "application/json",
          "Content-Length": Buffer.byteLength(json),
          "Access-Control-Allow-Origin": "*",
        })
      )
        return;
      this.endResponse(res, json);
    } catch (err) {
      if (!this.isClientDisconnectError(err)) throw err;
    }
  }

  private sendOllamaBackendError(
    res: ServerResponse,
    status: number,
    backendPayload: unknown,
  ): void {
    let payload = backendPayload;
    if (typeof backendPayload === "string") {
      try {
        payload = JSON.parse(backendPayload);
      } catch (_) {
        payload = { error: backendPayload || "Backend request failed" };
      }
    }
    const error = (payload as any)?.error;
    const detail = (payload as any)?.detail;
    const detailMessage =
      typeof detail === "string"
        ? detail
        : typeof detail?.message === "string"
          ? detail.message
          : undefined;
    const message =
      typeof error === "string"
        ? error
        : error?.message ||
          detailMessage ||
          (payload as any)?.message ||
          "Backend request failed";
    const code = error?.code || detail?.code || (payload as any)?.code;
    this.sendJson(res, status, {
      error: message,
      ...(code ? { code } : {}),
    });
  }
}

// Singleton
export const apiGateway = new ApiGateway();
