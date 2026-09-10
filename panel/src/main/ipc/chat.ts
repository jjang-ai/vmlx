// vMLX panel chat IPC — authored by Jinho Jang
import { ipcMain, BrowserWindow, net } from "electron";
import { v4 as uuidv4 } from "uuid";
import { request as httpsRequest } from "node:https";
import { request as httpRequest } from "node:http";
import type { ClientRequest } from "node:http";
import { readFileSync } from "node:fs";
import { join } from "node:path";
import { db, Chat, Message, Folder } from "../database";
import { sessionManager, resolveUrl, connectHost } from "../sessions";
import {
  readDetectedModelConfig,
  readGenerationDefaults,
} from "./models";
import type { RemoteDetectedConfig } from "../../shared/remoteModelCapabilities";
import {
  BUILTIN_TOOLS,
  isBuiltinTool,
  AGENTIC_SYSTEM_PROMPT,
} from "../tools/registry";
import {
  executeBuiltinTool,
  WORKING_DIR_INDEPENDENT_TOOLS,
} from "../tools/executor";
import { detectModelConfigFromDir } from "../model-config-registry";
import { localEngineReadyFromHealthBody } from "../engineReadiness";
import { getAuthHeaders } from "./utils";
import {
  appendOutputTruncationWarning,
  contextExhaustionNotice,
  dropSupersededRecoveryWarnings,
  effortSubstitutionNotice,
  extractResponsesWarnings,
  responsesTerminalFinishReason,
} from "../../shared/responsesWarnings";
import {
  appendReasoningDelta,
  joinReasoningSegments,
  markReasoningToolBoundary,
  reconcileReasoningSummaryDone,
  visibleReasoningSegments,
} from "../../shared/interleavedReasoning";
import {
  applyPostToolRequestFields,
  captureToolRequestFields,
  isToolAuthorizedForCurrentTurn,
  requiredToolChoiceNamesForCurrentTurn,
  requestedExactFinalToolNames,
  requestedOnceToolNames,
  requestedScopedToolNames,
  repeatsOnlyCompletedExactlyOnceTools,
  replaySafeToolCallKey,
  requestsExactTextOnlyWithoutToolUse,
  requestsNoToolCalls,
  requestsPrivateReasoningWithoutToolUse,
  scopeToolDefinitionsByName,
  shouldAutoContinueAfterToolUse,
  shouldFinishZayaAppleScriptToolRound,
  shouldPlanCompletedToolAnswerPass,
  toolChoiceForCurrentTurn,
  type ToolRequestFields,
  unavailableRequestedToolNames,
} from "../../shared/toolAutoContinue";
import { buildToolMediaFollowupContent } from "../../shared/toolMediaFollowup";
import { dsv4OutputBudget } from "../../shared/dsv4RequestBudget";
import {
  applyReasoningRequestFields,
  type ReasoningEffort,
} from "../../shared/reasoningEffortPolicy";
import {
  isMetalHeadroomBubbleContent,
  isPromptTooLongBubbleContent,
  projectedMetalHeadroomChatErrorContent,
  promptTooLongChatErrorContent,
} from "../../shared/chatErrorDisplay";
import {
  reconcileResponsesToolBufferAtStreamEnd,
  resolveNeverEmptyAssistantAnswer,
  TOOL_CALL_MARKER_LINE_START,
} from "../../shared/responsesStreamRecovery";
import {
  stripLeakedToolMarkup,
  stripStreamingToolTags,
  toolMarkupHoldbackLength,
} from "../../shared/toolMarkupSanitizer";
import { mergeCacheDetails } from "../../shared/cacheMetrics";
import { replayPersistedUserContentParts } from "../../shared/mediaHistoryReplay";
import {
  calculatePrefillTps,
  parseServerDecodeUsage,
  selectFinalDecodeTps,
  type ServerDecodePass,
  summarizeServerDecodePasses,
} from "../../shared/chatMetrics";
import { stripRedundantNamespacedToolPreview } from "../../shared/namespacedToolScaffold";
import { replayPersistedAssistantHistory } from "../../shared/toolHistoryReplay";
import { orderComposerContentParts } from "../../shared/composerContentOrder";
import { splitResponsesSystemMessages } from "../../shared/responsesSystemMessages";
import {
  ChatStreamServerEventError,
  chatStreamServerEventErrorDetail,
  shouldRethrowChatStreamLineError,
} from "../../shared/chatStreamErrors";
import {
  buildNewChatInheritedOverrides,
  sanitizeChatOverrides,
  sanitizeChatProfileOverrides,
} from "../chat-override-policy";
import {
  reasoningParserIsEnabled,
  resolveEffectiveReasoningParser,
} from "../../shared/reasoningParserAliases";
import {
  historicalUnavailableToolNames,
  latestUserMessageText,
  toolCapabilityFingerprint,
  toolCapabilityEpochInstruction,
  toolCapabilityNames,
} from "../tool-capability-epoch";
import { resolveSlowFamilyTimeoutSeconds } from '../../shared/slowFamilyTimeouts';
import { applyMtpSamplerOverrides } from '../../shared/effectiveGenerationDefaults';

// Default connection config (fallback values)
const DEFAULT_PORT = 8000;
const configuredToolStreamStallTimeoutMs = Number(
  process.env.VMLX_TOOL_STREAM_STALL_TIMEOUT_MS,
);
const TOOL_STREAM_STALL_TIMEOUT_MS = Number.isFinite(
  configuredToolStreamStallTimeoutMs,
)
  ? Math.max(5_000, configuredToolStreamStallTimeoutMs)
  : 30_000;


function effectiveFamilyRequestTimeoutSeconds(
  timeoutSeconds: number,
  detectedFamily?: string,
): number {
  // Shared table — see panel/src/shared/slowFamilyTimeouts.ts. This is the
  // timer that actually fails an in-app chat, so a divergence here is
  // immediately user-visible.
  return resolveSlowFamilyTimeoutSeconds(timeoutSeconds, detectedFamily);
}

function shouldSuppressGenericAgenticPromptForNativeTools(
  detectedFamily?: string,
  modelNameOrPath?: string,
): boolean {
  const modelName = String(modelNameOrPath || "").toLowerCase();
  return (
    detectedFamily === "deepseek-v4" ||
    detectedFamily === "lfm2" ||
    detectedFamily === "zaya" ||
    detectedFamily === "zaya1-vl" ||
    detectedFamily === "zaya1_vl" ||
    modelName.includes("zaya") ||
    modelName.includes("lfm2")
  );
}

function isZayaAppleScriptToolBundle(
  detectedFamily?: string,
  modelPath?: string,
): boolean {
  if (detectedFamily !== "zaya" || !modelPath) return false;
  try {
    // The AppleScript fine-tune intentionally keeps the generic ZAYA model
    // type/path. Its local model card is the only stable bundle-owned signal:
    // it names run_applescript and documents raw AppleScript when tools are
    // omitted. Restrict the schema only for that explicit model contract.
    const readme = readFileSync(join(modelPath, "README.md"), "utf8").slice(0, 32_768);
    return /\brun_applescript\b/i.test(readme) && /\bAppleScript\b/i.test(readme);
  } catch {
    return false;
  }
}

function isExpectedChatBackendDisconnectError(error: unknown): boolean {
  const err = error as NodeJS.ErrnoException | undefined;
  const code = String(err?.code || "");
  const message = String(err?.message || error || "");
  const cause = (err as any)?.cause;
  const wrappedDisconnects = [
    cause,
    (err as any)?.reason,
    (err as any)?.error,
    (err as any)?.detail,
  ].filter(Boolean);
  const nestedErrors = Array.isArray((err as any)?.errors)
    ? (err as any).errors
    : [];
  return (
    code === "EPIPE" ||
    code === "ECONNRESET" ||
    code === "ERR_STREAM_DESTROYED" ||
    code === "ERR_STREAM_WRITE_AFTER_END" ||
    /EPIPE|write EPIPE|broken pipe|socket hang up|connection reset|premature close|stream.*destroyed|write after end/i.test(message) ||
    wrappedDisconnects.some((nested) => isExpectedChatBackendDisconnectError(nested)) ||
    nestedErrors.some((nested) => isExpectedChatBackendDisconnectError(nested))
  );
}

function expectedChatBackendDisconnectError(): NodeJS.ErrnoException {
  const error = new Error(
    "Backend connection closed while streaming response.",
  ) as NodeJS.ErrnoException;
  error.code = "EPIPE";
  return error;
}

function chatBackendRequestWritable(req: ClientRequest): boolean {
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

function endChatBackendRequest(
  req: ClientRequest,
  bodyBuf: Buffer,
  reject: (reason?: unknown) => void,
): void {
  if (!chatBackendRequestWritable(req)) {
    const error = new Error("Backend connection closed before request completed.") as NodeJS.ErrnoException;
    error.code = "EPIPE";
    reject(error);
    return;
  }
  try {
    req.end(bodyBuf);
  } catch (error) {
    if (isExpectedChatBackendDisconnectError(error)) {
      reject(error);
      return;
    }
    throw error;
  }
}

type ComposerAttachment = {
  dataUrl: string;
  name: string;
  kind?: "image" | "video" | "audio" | "text";
  type?: string;
  size?: number;
  text?: string;
};

function inferKind(a: ComposerAttachment): "image" | "video" | "audio" | "text" {
  if (a.kind) return a.kind;
  if (a.text !== undefined) return "text";
  if (a.dataUrl.startsWith("data:audio/")) return "audio";
  if (a.dataUrl.startsWith("data:video/")) return "video";
  if (a.dataUrl.startsWith("data:text/")) return "text";
  return "image";
}

function mimeFromDataUrl(dataUrl?: string): string | undefined {
  return dataUrl?.match(/^data:([^;,]+)[;,]/)?.[1]?.toLowerCase();
}

function redactContentForLog(content: any): any {
  if (Array.isArray(content)) {
    return content.map((part: any) => {
      if (!part || typeof part !== "object") return { type: typeof part };
      if (part.type === "text") {
        return { type: "text", chars: String(part.text || "").length };
      }
      if (part.type === "image_url") {
        const url = part.image_url?.url || part.image_url;
        return {
          type: "image_url",
          mime: mimeFromDataUrl(typeof url === "string" ? url : undefined),
          data_url_chars: typeof url === "string" && url.startsWith("data:") ? url.length : undefined,
          url: "<redacted>",
        };
      }
      if (part.type === "video_url") {
        const url = part.video_url?.url || part.video_url;
        return {
          type: "video_url",
          mime: mimeFromDataUrl(typeof url === "string" ? url : undefined),
          data_url_chars: typeof url === "string" && url.startsWith("data:") ? url.length : undefined,
          url: "<redacted>",
        };
      }
      if (part.type === "input_audio") {
        return {
          type: "input_audio",
          format: part.input_audio?.format,
          data_chars:
            typeof part.input_audio?.data === "string"
              ? part.input_audio.data.length
              : undefined,
        };
      }
      return { type: part.type || "unknown", keys: Object.keys(part).sort() };
    });
  }
  if (typeof content === "string") return { type: "text", chars: content.length };
  if (content == null) return null;
  return { type: typeof content };
}

function summarizeRequestForLog(bodyJson: string, useResponsesApi: boolean): Record<string, any> {
  try {
    const body = JSON.parse(bodyJson);
    const items = useResponsesApi ? body.input : body.messages;
    return {
      route: useResponsesApi ? "/v1/responses" : "/v1/chat/completions",
      model: body.model,
      stream: body.stream === true,
      max_tokens: body.max_output_tokens ?? body.max_tokens,
      temperature: body.temperature,
      top_p: body.top_p,
      top_k: body.top_k,
      min_p: body.min_p,
      repetition_penalty: body.repetition_penalty,
      enable_thinking: body.enable_thinking,
      thinking_mode: body.thinking_mode,
      reasoning_effort: body.reasoning_effort,
      reasoning_strength: body.chat_template_kwargs?.reasoning_strength,
      max_thinking_tokens: body.max_thinking_tokens,
      image_token_budget: body.image_token_budget,
      thinking_budget: body.chat_template_kwargs?.thinking_budget,
      previous_response_id: body.previous_response_id ? "<present>" : undefined,
      has_tools: Array.isArray(body.tools) && body.tools.length > 0,
      tool_choice: body.tool_choice,
      tool_names: Array.isArray(body.tools)
        ? body.tools
            .map((tool: any) => tool?.function?.name ?? tool?.name)
            .filter((name: unknown): name is string => typeof name === "string")
        : [],
      messages: Array.isArray(items)
        ? items.slice(-8).map((m: any) => ({
            role: m.role || m.type || "item",
            content: redactContentForLog(m.content ?? m.input ?? m.output ?? m.text),
          }))
        : redactContentForLog(items),
    };
  } catch (e: any) {
    return { error: `request summary failed: ${e?.message || String(e)}` };
  }
}

function summarizeAttachmentsForLog(attachments?: ComposerAttachment[]): Record<string, any> {
  const counts = { image: 0, video: 0, audio: 0, text: 0 };
  const files: Array<Record<string, any>> = [];
  for (const attachment of attachments || []) {
    const kind = inferKind(attachment);
    counts[kind] += 1;
    files.push({
      kind,
      name: attachment.name,
      mime: attachment.type || mimeFromDataUrl(attachment.dataUrl),
      size: attachment.size,
      data_url_chars: attachment.dataUrl?.startsWith("data:")
        ? attachment.dataUrl.length
        : undefined,
      text_chars: attachment.text?.length,
    });
  }
  return { total: attachments?.length || 0, counts, files };
}

/**
 * SSE-streaming fetch using Node.js http/https directly.
 * Electron 28's global fetch() uses Chromium's net module which buffers
 * SSE chunks instead of delivering them immediately. Node.js http/https
 * streams data as it arrives from the socket.
 */
async function streamingFetch(
  url: string,
  init: {
    method: string;
    headers: Record<string, string>;
    body: string;
    signal?: AbortSignal;
  },
): Promise<{
  ok: boolean;
  status: number;
  statusText: string;
  body: ReadableStream<Uint8Array> | null;
  text: () => Promise<string>;
}> {
  const parsed = new URL(url);
  const isHttps = parsed.protocol === "https:";
  const reqFn = isHttps ? httpsRequest : httpRequest;
  const bodyBuf = Buffer.from(init.body, "utf-8");

  return new Promise((resolve, reject) => {
    if (init.signal?.aborted) {
      reject(
        Object.assign(new Error("The operation was aborted."), {
          name: "AbortError",
        }),
      );
      return;
    }

    let settled = false;
    const settle = (fn: () => void) => {
      if (!settled) {
        settled = true;
        fn();
      }
    };

    const req = reqFn(
      {
        hostname: parsed.hostname,
        port: parsed.port || (isHttps ? 443 : 80),
        path: parsed.pathname + parsed.search,
        method: init.method,
        // Disable connection pooling — each SSE stream gets a fresh TCP connection.
        // Prevents stale keep-alive connections from causing ECONNRESET/"aborted" errors.
        agent: false,
        headers: {
          ...init.headers,
          "Content-Length": bodyBuf.length.toString(),
        },
      },
      (res) => {
        const ok = (res.statusCode ?? 0) >= 200 && (res.statusCode ?? 0) < 300;

        if (!ok) {
          let data = "";
          res.on("data", (chunk) => {
            data += chunk.toString();
          });
          res.on("end", () => {
            settle(() =>
              resolve({
                ok,
                status: res.statusCode ?? 0,
                statusText: res.statusMessage ?? "",
                body: null,
                text: () => Promise.resolve(data),
              }),
            );
          });
          res.on("error", () => {
            settle(() =>
              resolve({
                ok,
                status: res.statusCode ?? 0,
                statusText: res.statusMessage ?? "",
                body: null,
                text: () => Promise.resolve(data),
              }),
            );
          });
          return;
        }

        // Wrap Node.js stream in Web ReadableStream for compatibility with streamSSE
        const stream = new ReadableStream<Uint8Array>({
          start(controller) {
            res.on("data", (chunk: Buffer) => {
              controller.enqueue(new Uint8Array(chunk));
            });
            res.on("end", () => {
              try {
                controller.close();
              } catch (_) {}
            });
            res.on("error", (err) => {
              if (isExpectedChatBackendDisconnectError(err)) {
                try {
                  controller.error(err);
                } catch (_) {}
                return;
              }
              console.error(
                `[streamingFetch] stream error: message="${(err as any)?.message}" code="${(err as any)?.code}"`,
              );
              try {
                controller.error(err);
              } catch (_) {}
            });
            // Handle premature close (server drops connection before response completes)
            res.on("close", () => {
              if (!res.complete) {
                try {
                  controller.error(
                    new Error("Connection closed before response completed"),
                  );
                } catch (_) {}
              }
            });
          },
          cancel() {
            res.destroy();
          },
        });

        settle(() =>
          resolve({
            ok: true,
            status: res.statusCode ?? 200,
            statusText: res.statusMessage ?? "OK",
            body: stream,
            text: () =>
              Promise.reject(
                new Error("Cannot read text from streaming response"),
              ),
          }),
        );
      },
    );

    req.on("error", (err) => {
      settle(() => reject(err));
    });

    if (init.signal) {
      const onAbort = () => {
        req.destroy();
        settle(() =>
          reject(
            Object.assign(new Error("The operation was aborted."), {
              name: "AbortError",
            }),
          ),
        );
      };
      init.signal.addEventListener("abort", onAbort, { once: true });
    }

    endChatBackendRequest(req, bodyBuf, reject);
  });
}

// Common chat template stop tokens that models may generate
const TEMPLATE_STOP_TOKENS = [
  "<|im_end|>",
  "<|im_start|>", // ChatML (Qwen, etc.)
  "<|eot_id|>",
  "<|start_header_id|>", // Llama 3
  "<|end|>",
  "<|user|>",
  "<|assistant|>", // Phi-3
  "</s>",
  "<s>", // Llama 2, Mistral
  "<|endoftext|>", // GPT-NeoX, StableLM
  "[/INST]",
  "[INST]", // Mistral instruct
  "<end_of_turn>", // Gemma
  "<minimax:tool_call>", // MiniMax tool call open tag
  "</minimax:tool_call>", // MiniMax tool call close tag
  "<|start|>",
  "<|channel|>",
  "<|message|>", // Harmony/GPT-OSS protocol (GLM-4.7, GPT-OSS)
];

// Regex to strip any leaked template tokens from output
const TEMPLATE_TOKEN_REGEX = new RegExp(
  TEMPLATE_STOP_TOKENS.map((t) =>
    t.replace(/[.*+?^${}()|[\]\\]/g, "\\$&"),
  ).join("|"),
  "g",
);

/**
 * Use Electron's net.fetch for remote sessions — Chromium's network stack handles
 * HTTPS certificates, system proxies, and SSE streaming properly.
 * Local sessions use streamingFetch (Node.js http/https) to avoid Electron 28's
 * global fetch buffering SSE chunks.
 */
const remoteFetch: typeof globalThis.fetch = (input, init?) =>
  net.fetch(input as any, init as any);

function isLoopbackUrl(url: string): boolean {
  try {
    const host = new URL(url).hostname.toLowerCase();
    return host === "127.0.0.1" || host === "localhost" || host === "::1";
  } catch (_) {
    return false;
  }
}

const DIRECT_MEDIA_ATTACHMENT_TOOL_RULE =
  "\n\nIMPORTANT: When a user message includes media attachments as chat content, inspect those images, video, or audio directly through the model's multimodal input. Do not call read_image, read_video, or list_directory to find attached media unless the user explicitly gives a local filesystem path.";

// Tool category definitions for per-category filtering
const FILE_TOOLS = new Set([
  "read_file",
  "write_file",
  "edit_file",
  "patch_file",
  "batch_edit",
  "copy_file",
  "move_file",
  "delete_file",
  "create_directory",
  "list_directory",
  "insert_text",
  "replace_lines",
  "apply_regex",
  "read_image",
  "read_video",
]);
const SEARCH_TOOLS = new Set([
  "search_files",
  "find_files",
  "file_info",
  "get_diagnostics",
  "get_tree",
  "diff_files",
]);
const SHELL_TOOLS = new Set([
  "run_command",
  "run_applescript",
  "spawn_process",
  "get_process_output",
]);
const DDG_SEARCH_TOOLS = new Set(["ddg_search"]);
const FETCH_TOOLS = new Set(["fetch_url"]);
const GIT_TOOLS = new Set(["git"]);
const UTILITY_TOOLS = new Set([
  "count_tokens",
  "clipboard_read",
  "clipboard_write",
  "get_current_datetime",
]);
// ask_user is intentionally excluded from UTILITY_TOOLS — it's a core IPC tool that should
// always be available regardless of the utilityToolsEnabled toggle.

/** Build set of disabled tool names based on per-category toggle overrides */
function getDisabledTools(overrides: any): Set<string> {
  const disabled = new Set<string>();
  if (overrides.fileToolsEnabled === false)
    FILE_TOOLS.forEach((t) => disabled.add(t));
  if (overrides.searchToolsEnabled === false)
    SEARCH_TOOLS.forEach((t) => disabled.add(t));
  if (overrides.shellEnabled === false)
    SHELL_TOOLS.forEach((t) => disabled.add(t));
  if (overrides.webSearchEnabled === false)
    DDG_SEARCH_TOOLS.forEach((t) => disabled.add(t));
  if (overrides.fetchUrlEnabled === false)
    FETCH_TOOLS.forEach((t) => disabled.add(t));
  if (overrides.gitEnabled === false) GIT_TOOLS.forEach((t) => disabled.add(t));
  if (overrides.utilityToolsEnabled === false)
    UTILITY_TOOLS.forEach((t) => disabled.add(t));
  // Brave web_search requires API key — always disable if no key configured
  // (user must explicitly enable Brave search via braveSearchEnabled toggle)
  if (overrides.braveSearchEnabled === false) {
    disabled.add("web_search");
  } else {
    const braveKey = db.getSetting("braveApiKey");
    if (!braveKey && !process.env.BRAVE_API_KEY) {
      disabled.add("web_search");
    }
  }
  return disabled;
}

/** Filter BUILTIN_TOOLS based on per-category toggle overrides */
function filterTools(
  overrides: any,
  context: {
    zayaAppleScriptToolBundle?: boolean;
  } = {},
): any[] {
  const availableTools = context.zayaAppleScriptToolBundle
    ? BUILTIN_TOOLS.filter((t: any) => t.function.name === "run_applescript")
    : BUILTIN_TOOLS;
  const disabled = getDisabledTools(overrides);
  if (disabled.size === 0) return availableTools;
  return availableTools.filter((t: any) => !disabled.has(t.function.name));
}

// Track active requests per chat for abort/concurrency (B5/B6)
const activeRequests = new Map<
  string,
  {
    controller: AbortController;
    startedAt: number;
    timeoutMs: number;
    responseId?: string;
    endpoint?: { host: string; port: number };
    baseUrl?: string;
    authHeaders?: Record<string, string>;
  }
>();
// Stale lock: each request stores its timeoutMs; stale check uses timeoutMs + 30s buffer

// ask_user: single global listener with Map-based resolver (prevents listener accumulation)
const askUserResolvers = new Map<string, (answer: string) => void>();
ipcMain.on("chat:answerUser", (_, chatId: string, answer: string) => {
  const resolve = askUserResolvers.get(chatId);
  if (resolve) {
    askUserResolvers.delete(chatId);
    resolve(answer);
  }
});

/** Abort all active chat requests targeting a specific endpoint (called when session stops) */
export function abortByEndpoint(host: string, port: number): number {
  let count = 0;
  for (const [chatId, entry] of activeRequests) {
    if (entry.endpoint?.host === host && entry.endpoint?.port === port) {
      console.log(
        `[CHAT] Aborting chat ${chatId} — session endpoint ${host}:${port} stopped`,
      );
      // Send server cancel if we have a response ID (fire-and-forget)
      if (entry.responseId && (entry.baseUrl || entry.endpoint)) {
        const cancelPath = entry.responseId.startsWith("resp_")
          ? `/v1/responses/${entry.responseId}/cancel`
          : `/v1/chat/completions/${entry.responseId}/cancel`;
        const cancelBase = entry.baseUrl || `http://${host}:${port}`;
        fetch(`${cancelBase}${cancelPath}`, {
          method: "POST",
          headers: entry.authHeaders || {},
          signal: AbortSignal.timeout(1000),
        }).catch(() => {
          /* server may already be stopped */
        });
      }
      try {
        entry.controller.abort();
      } catch (_) {}
      activeRequests.delete(chatId);
      count++;
    }
  }
  return count;
}

/** Resolved endpoint info including optional session reference */
interface ResolvedEndpoint {
  host: string;
  port: number;
  session?: import("../database").Session;
}

/** Resolve endpoint for a chat: use modelPath to find session, fallback to detection */
async function resolveServerEndpoint(
  modelPath?: string,
): Promise<ResolvedEndpoint> {
  // 1. If chat has modelPath, find its session (normalize to handle trailing slash)
  if (modelPath) {
    const session = sessionManager.getSessionByModelPath(
      modelPath.replace(/\/+$/, ""),
    );
    // loading/standby resolve too: a message sent mid-load queues against
    // this session (exactly once, Stop cancels) and a sleeping one wakes it.
    if (
      session &&
      (session.status === "running" ||
        session.status === "loading" ||
        session.status === "standby")
    ) {
      return { host: session.host, port: session.port, session };
    }
  }

  // 2. Detect any running processes
  const processes = await sessionManager.detect();
  const healthy = processes.find((p) => p.healthy);
  if (healthy) {
    // Use 127.0.0.1 for connection (0.0.0.0 is a bind address, not connectable)
    return { host: "127.0.0.1", port: healthy.port };
  }

  return { host: "127.0.0.1", port: DEFAULT_PORT };
}

export function registerChatHandlers(
  getWindow: () => BrowserWindow | null,
): void {
  const pushChatSessionLog = (sessionId: string | undefined, line: string) => {
    if (!sessionId) return;
    const data = line.endsWith("\n") ? line : `${line}\n`;
    sessionManager.pushLog(sessionId, data);
    sessionManager.emit("session:log", { sessionId, data });
  };

  // Folders
  ipcMain.handle(
    "chat:createFolder",
    async (_, name: string, parentId?: string) => {
      const folder: Folder = {
        id: uuidv4(),
        name,
        parentId,
        createdAt: Date.now(),
      };
      db.createFolder(folder);
      return folder;
    },
  );

  ipcMain.handle("chat:getFolders", async () => {
    return db.getFolders();
  });

  ipcMain.handle("chat:deleteFolder", async (_, id: string) => {
    db.deleteFolder(id);
    return { success: true };
  });

  // Chats
  const createChatRecord = (
    title: string,
    modelId: string,
    folderId?: string,
    modelPath?: string,
  ): Chat => {
    const chat: Chat = {
      id: uuidv4(),
      title,
      folderId,
      createdAt: Date.now(),
      updatedAt: Date.now(),
      modelId,
      modelPath,
    };
    db.createChat(chat);

      // Do not seed new chats from per-model settings or sibling chats. A
      // clean chat starts with no overrides; the engine resolves bundle
      // defaults and explicit chat/API settings are stored only per chat.
      // Default profiles / sibling chats may carry tool and workspace
      // ergonomics into a clean chat, but sampling/reasoning/system prompts are
      // intentionally excluded so bundle generation defaults stay authoritative.
      try {
        const defaultProfile = db.getDefaultChatProfile();
        if (defaultProfile) {
          const existing = db.getChatOverrides(chat.id) || { chatId: chat.id };
          // Default profiles are auto-applied at chat creation time, so treat
          // them as coding/tool presets. Manual profile loads can still apply
          // full sampler/reasoning/prompt settings after the user chooses them.
          const merged = buildNewChatInheritedOverrides(
            existing as any,
            sanitizeChatProfileOverrides(defaultProfile) as any,
          );
          db.setChatOverrides(merged as any);
          console.log(
            `[CHAT] Applied default profile coding/tool settings to new chat ${chat.id}; generation/reasoning defaults stayed model-derived`,
          );
        } else {
          // No starred profile — honor the visible "last chat" contract across
          // model switches. Skip this newly inserted row and any older chat
          // without an override record instead of stopping at an empty sibling.
          // The inheritance policy still copies only tool/workspace ergonomics;
          // sampling, prompts, and reasoning remain model-derived.
          const lastInherited = db
            .getRecentChats(100)
            .filter((candidate) => candidate.id !== chat.id)
            .map((candidate) => ({
              chat: candidate,
              overrides: db.getChatOverrides(candidate.id),
            }))
            .find((candidate) => candidate.overrides !== undefined);
          if (lastInherited?.overrides) {
            const existing = db.getChatOverrides(chat.id) || {
              chatId: chat.id,
            };
            const merged = buildNewChatInheritedOverrides(
              existing as any,
              lastInherited.overrides as any,
            );
            db.setChatOverrides(merged);
            console.log(
              `[CHAT] Inherited coding/tool settings from last chat ${lastInherited.chat.id.slice(0, 8)}; generation/reasoning defaults stayed model-derived`,
            );
          }
        }
      } catch (e) {
        console.error("[CHAT] Failed to apply default profile:", e);
      }

    return chat;
  };

  ipcMain.handle(
    "chat:create",
    async (
      _,
      title: string,
      modelId: string,
      folderId?: string,
      modelPath?: string,
    ) => createChatRecord(title, modelId, folderId, modelPath),
  );

  // SessionView effects can be replayed in development. Keep the initial
  // chat lookup and insert in one main-process turn so replay cannot create
  // two empty chats for the same newly launched model.
  ipcMain.handle(
    "chat:ensureForModel",
    async (_, title: string, modelPath: string) => {
      const existing = db.getChatsByModelPath(modelPath)[0];
      return existing ?? createChatRecord(title, "default", undefined, modelPath);
    },
  );

  ipcMain.handle("chat:getByModel", async (_, modelPath: string) => {
    return db.getChatsByModelPath(modelPath);
  });

  ipcMain.handle("chat:getRecent", async (_, limit: number) => {
    return db.getRecentChats(limit);
  });

  ipcMain.handle("chat:getAll", async (_, folderId?: string) => {
    return db.getChats(folderId);
  });

  ipcMain.handle("chat:get", async (_, id: string) => {
    return db.getChat(id);
  });

  ipcMain.handle(
    "chat:update",
    async (_, id: string, updates: Partial<Chat>) => {
      db.updateChat(id, updates);
      return { success: true };
    },
  );

  ipcMain.handle("chat:delete", async (_, id: string) => {
    db.deleteChat(id);
    return { success: true };
  });

  // vmlx#70: bulk-delete. Reporter wanted "mass delete or wipe chat
  // history" instead of one-by-one. This single handler covers:
  //   - wipe everything:               {}
  //   - wipe unfiled:                  { folderId: "unfiled" }
  //   - wipe one folder:               { folderId: "<id>" }
  //   - wipe all chats for a model:    { modelPath: "<path>" }
  // Returns the count so the UI can confirm "N chats deleted".
  ipcMain.handle(
    "chat:deleteAll",
    async (_, scope?: { folderId?: string; modelPath?: string }) => {
      try {
        const deleted = db.deleteAllChats(scope);
        return { success: true, deleted };
      } catch (error) {
        return { success: false, error: (error as Error).message };
      }
    },
  );

  ipcMain.handle("chat:deleteMessage", async (_, messageId: string) => {
    db.deleteMessage(messageId);
    return { success: true };
  });

  ipcMain.handle("chat:deleteMessagesFrom", async (_, chatId: string, fromTimestamp: number) => {
    db.deleteMessagesFrom(chatId, fromTimestamp);
    return { success: true };
  });

  ipcMain.handle("chat:search", async (_, query: string) => {
    return db.searchChats(query);
  });

  // Messages
  ipcMain.handle("chat:getMessages", async (_, chatId: string) => {
    return db.getMessages(chatId);
  });

  ipcMain.handle(
    "chat:addMessage",
    async (_, chatId: string, role: string, content: string) => {
      // Ensure chat exists (FK constraint on messages.chat_id)
      const chat = db.getChat(chatId);
      if (!chat) {
        throw new Error(`Cannot add message: chat ${chatId} not found`);
      }
      const message: Message = {
        id: uuidv4(),
        chatId,
        role: role as "system" | "user" | "assistant",
        content,
        timestamp: Date.now(),
      };
      db.addMessage(message);
      return message;
    },
  );

  // Send message and get streaming response
  // Optional 4th arg: endpoint override { host, port } for multi-server support
  // Optional 5th arg: attachments from the chat composer. Media attachments
  // force multimodal routing; text files are inlined as text context.
  // `kind` distinguishes image, video, audio, and text so we emit the right
  // OpenAI content part. Back-compat: undefined `kind` falls back to detection
  // via the data URL mime prefix below.
  ipcMain.handle(
    "chat:sendMessage",
    async (
      _,
      chatId: string,
      content: string,
      endpoint?: { host: string; port: number },
      attachments?: Array<{
        dataUrl: string;
        name: string;
        kind?: "image" | "video" | "audio" | "text";
        type?: string;
        size?: number;
        text?: string;
      }>,
    ) => {
      // B6: Concurrency guard — reject if a request is already active for this chat
      // B6: Concurrency guard with stale lock recovery
      const existing = activeRequests.get(chatId);
      if (existing) {
        const age = Date.now() - existing.startedAt;
        // Use the timeout configured when that request started, plus 30s buffer
        // Cap at 30 minutes to prevent indefinite lock (e.g. serverTimeout=86400s)
        const staleLockMs = Math.min(
          existing.timeoutMs + 30_000,
          30 * 60 * 1000,
        );
        // An ALREADY-ABORTED controller can never produce a response, so it
        // must never hold the chat hostage for the full stale window. Every
        // abort path (session down, explicit Stop, window reload) deletes the
        // entry itself; this is the backstop for the one that forgets. Without
        // it a single missed delete makes the chat silently unusable for up to
        // 30 minutes — the user sends, sees their message appear, and gets a
        // toast that vanishes.
        const abandoned = existing.controller.signal.aborted;
        if (abandoned || age > staleLockMs) {
          // Lock is stale or abandoned — abort and clear it
          console.log(
            `[CHAT] Clearing ${abandoned ? "abandoned" : "stale"} lock for ${chatId} ` +
              `(${Math.round(age / 1000)}s old, limit ${Math.round(staleLockMs / 1000)}s)`,
          );
          try {
            existing.controller.abort();
          } catch (_) {}
          activeRequests.delete(chatId);
        } else {
          throw new Error("A message is already being generated for this chat");
        }
      }

      // B5: Create AbortController for this request
      const abortController = new AbortController();
      let timedOut = false;

      const chat = db.getChat(chatId);
      if (!chat) {
        throw new Error("Chat not found");
      }

      // Look up session for this chat — needed for timeout, reasoning parser,
      // AND for endpoint resolution (remote sessions need remoteUrl/apiKey/type)
      let timeoutSeconds = 300;
      let sessionHasReasoningParser = false;
      let isHarmonyModel = false;
      let chatIsMultimodal = false;
      let chatDetectedFamily: string | undefined;
      let chatUsesZayaAppleScriptToolBundle = false;
      let thinkingBudgetSupported: boolean | undefined;
      let supportsThinkingBudget: boolean | undefined;
      let supportsInstructMode: boolean | undefined;
      let supportedReasoningEfforts: ReasoningEffort[] | undefined;
      let chatNativeMtp: RemoteDetectedConfig['nativeMtp'];
      let chatSessionConfig: Record<string, unknown> = {};
      let sessionImageTokenBudget: number | undefined;
      // VLM video sampling (Qwen 3.6, Qwen3.5-VL, etc.) — forwarded as
      // video_fps / video_max_frames on the request body when present.
      // Default undefined = engine default (2.0 fps, 8 max frames).
      let sessionVideoFps: number | undefined;
      let sessionVideoMaxFrames: number | undefined;
      let sessionVideoMaxPixels: number | undefined;
      let sessionVideoTokenBudget: number | undefined;
      let chatSession: import("../database").Session | undefined;
      if (chat.modelPath) {
        chatSession = sessionManager.getSessionByModelPath(
          chat.modelPath.replace(/\/+$/, ""),
        );
        if (!chatSession) {
          // Path mismatch fallback: HF repo ID vs resolved local path can differ
          // (e.g., "org/Model-CRACKED-MLX" resolves to directory "org/Model-CRACK").
          // Try matching by basename as a last resort.
          const basename = chat.modelPath.replace(/\/+$/, "").split("/").pop();
          if (basename) {
            const allSessions = sessionManager.getSessions();
            chatSession = allSessions.find(
              (s) =>
                (s.status === "running" ||
                  s.status === "loading" ||
                  s.status === "standby") &&
                s.modelPath.split("/").pop()?.replace(/\/+$/, "") === basename,
            );
            if (chatSession) {
              console.log(
                `[CHAT] Session found by basename fallback: ${basename} → ${chatSession.id.slice(0, 8)}`,
              );
            } else {
              console.warn(
                `[CHAT] No session found for modelPath=${chat.modelPath} — timeout will use default 300s`,
              );
            }
          }
        }
        if (chatSession) {
          // Touch session to reset idle timer — prevents premature sleep during active chat
          sessionManager.touchSession(chatSession.id);
          try {
            const sessionConfig = JSON.parse(chatSession.config);
            chatSessionConfig = sessionConfig;
            if (sessionConfig.timeout === 0) {
              // "No limit" — match the 86400s (24h) that sessions.ts sends via --timeout
              timeoutSeconds = 86400;
            } else if (sessionConfig.timeout && sessionConfig.timeout > 0) {
              timeoutSeconds = sessionConfig.timeout;
            }
            // Check if model has a reasoning parser (for enable_thinking default)
            // Remote sessions have no readable local bundle directory. Hydrate
            // the exact live family/parser/cache contract from the model's
            // capability endpoint instead of guessing from its display alias.
            const detected =
              (await readDetectedModelConfig(chat.modelPath)) ??
              ({} as RemoteDetectedConfig);
            try {
              const generationDefaults = await readGenerationDefaults(chat.modelPath);
              thinkingBudgetSupported = generationDefaults?.thinkingBudgetSupported;
            } catch {
              thinkingBudgetSupported = undefined;
            }
            chatDetectedFamily = detected.family;
            chatUsesZayaAppleScriptToolBundle = isZayaAppleScriptToolBundle(
              chatDetectedFamily,
              chat.modelPath,
            );
            supportsThinkingBudget = detected.supportsThinkingBudget;
            supportsInstructMode = detected.supportsInstructMode;
            supportedReasoningEfforts = detected.supportedReasoningEfforts;
            chatNativeMtp = detected.nativeMtp;
            timeoutSeconds = effectiveFamilyRequestTimeoutSeconds(
              timeoutSeconds,
              chatDetectedFamily,
            );
            const smeltActive = !!sessionConfig.smelt;
            chatIsMultimodal =
              smeltActive || detected.forceTextOnly
                ? false
                : detected.isMultimodal === true
                  ? true
                  : sessionConfig.isMultimodal === true
                    ? true
                    : sessionConfig.isMultimodal === false
                      ? false
                      : false;

            const effectiveReasoningParser = resolveEffectiveReasoningParser({
              configuredParser: sessionConfig.reasoningParser,
              detectedParser: detected.reasoningParser,
              supportsThinking: detected.supportsThinking,
            });
            sessionHasReasoningParser = reasoningParserIsEnabled(
              effectiveReasoningParser,
            );
            isHarmonyModel = effectiveReasoningParser === "openai_gptoss";
            // VLM video sampling knobs (undefined → engine default)
            if (
              chatDetectedFamily === "gemma4" &&
              typeof sessionConfig.imageTokenBudget === "number" &&
              sessionConfig.imageTokenBudget > 0
            )
              sessionImageTokenBudget = sessionConfig.imageTokenBudget;
            if (typeof sessionConfig.videoFps === "number" && sessionConfig.videoFps > 0)
              sessionVideoFps = sessionConfig.videoFps;
            if (typeof sessionConfig.videoMaxFrames === "number" && sessionConfig.videoMaxFrames > 0)
              sessionVideoMaxFrames = sessionConfig.videoMaxFrames;
            if (typeof sessionConfig.videoMaxPixels === "number" && sessionConfig.videoMaxPixels > 0)
              sessionVideoMaxPixels = sessionConfig.videoMaxPixels;
            if (typeof sessionConfig.videoTokenBudget === "number" && sessionConfig.videoTokenBudget > 0)
              sessionVideoTokenBudget = sessionConfig.videoTokenBudget;
          } catch (_) {}
        }
      }
      // Session timeout is an INACTIVITY guard for the app. The local engine
      // sends SSE keep-alive comments while a long prefill is advancing, so
      // every received byte re-arms this timer below. A single wall-clock
      // timer discarded valid near-context work at exactly 900s even though
      // the engine was still sending keep-alives and processing prompt chunks.
      //
      // The engine's SSE contract emits a keep-alive every 15 seconds. A saved
      // timeout can be as low as 10 seconds, so using it directly here races
      // the first keep-alive and aborts a healthy prefill before the server can
      // report liveness. Give local SSE two keep-alive intervals; the engine
      // still enforces the configured progress-aware timeout itself. Remote
      // endpoints retain their configured client-side inactivity budget.
      // chatSession is the same known session later returned by endpoint
      // resolution (non-local renderer endpoints are rejected unless they
      // match it), so establish this before the watchdog is armed.
      const isRemote = chatSession?.type === "remote";
      const fetchInactivitySeconds = isRemote
        ? timeoutSeconds
        : Math.max(timeoutSeconds, 30);
      let fetchTimeout: ReturnType<typeof setTimeout> | undefined;
      const armFetchInactivityTimeout = () => {
        if (fetchTimeout) clearTimeout(fetchTimeout);
        if (abortController.signal.aborted) return;
        // The concurrency guard uses startedAt to decide whether an existing
        // lock is abandoned. Keep that liveness clock on the same definition
        // as the fetch watchdog, or a second send can abort a healthy request
        // merely because its total wall time exceeded the stale-lock cap.
        const active = activeRequests.get(chatId);
        if (active && active.controller === abortController) {
          active.startedAt = Date.now();
        }
        fetchTimeout = setTimeout(() => {
          timedOut = true;
          abortController.abort();
        }, fetchInactivitySeconds * 1000);
      };
      // Deliberately NOT armed here: the watchdog is an INFERENCE inactivity
      // guard. Arming it at entry made load/wake queue waiting consume the
      // generation timeout — exactly the premature timeout the user hit. It
      // is armed immediately before the inference fetch below.
      activeRequests.set(chatId, {
        controller: abortController,
        startedAt: Date.now(),
        timeoutMs: fetchInactivitySeconds * 1000,
        endpoint: undefined,
        responseId: undefined,
      });
      // The entry above must be removed on EVERY exit path. Setup below can
      // throw (endpoint validation/resolution, health-check exhaustion while
      // an engine is still loading); without this try/finally those throws
      // leaked the entry, leaving the chat locked in "streaming" state until
      // the stale-lock window expired.
      try {

      // Resolve actual server endpoint: explicit endpoint > session by modelPath > detect > default
      // CRITICAL: When endpoint is passed from the renderer, attach the chatSession
      // so remote sessions get proper remoteUrl, auth headers, and health check path.
      // SECURITY: Validate renderer-provided endpoint is localhost or matches a known session
      if (endpoint) {
        const isLocalhost =
          endpoint.host === "127.0.0.1" ||
          endpoint.host === "localhost" ||
          endpoint.host === "::1" ||
          endpoint.host === "0.0.0.0";
        const isKnownSession =
          chatSession &&
          chatSession.host === endpoint.host &&
          chatSession.port === endpoint.port;
        if (!isLocalhost && !isKnownSession) {
          throw new Error(
            `Endpoint ${endpoint.host}:${endpoint.port} not allowed — must be localhost or match a configured session`,
          );
        }
      }
      const resolved = endpoint
        ? ({
            host: endpoint.host,
            port: endpoint.port,
            session: chatSession,
          } as ResolvedEndpoint)
        : await resolveServerEndpoint(chat.modelPath);

      // Detect remote session and compute base URL + auth headers
      const resolvedSession = resolved.session;

      // Register the endpoint BEFORE the wake/queue waits below: Stop cancels
      // chats via abortByEndpoint(), which can only match entries whose
      // endpoint is populated. Without this, a queued message survived Stop.
      {
        const queuedEntry = activeRequests.get(chatId);
        if (queuedEntry) {
          queuedEntry.endpoint = { host: resolved.host, port: resolved.port };
        }
      }

      // A sleeping local engine answers /health immediately, so the generic
      // health check below previously declared it ready and let Python perform
      // a hidden JIT wake inside the inference request.  Electron remained
      // labelled Deep Sleep for the whole reload and could not show resident
      // progress.  Wake through SessionManager first: it owns the visible
      // loading state, progress monitor, failure rollback, and idle clock.
      const settleTimeoutMsFor = (session?: { config?: string }) => {
        let configured = 300;
        try {
          configured = Number(JSON.parse(session?.config || "{}").timeout) || 300;
        } catch {}
        return Math.max(configured * 1000, 120_000) + 60_000;
      };

      if (!isRemote && resolvedSession?.status === "standby") {
        console.log(`[CHAT] Waking standby session ${resolvedSession.id} before inference`);
        const wake = await sessionManager.wakeSession(resolvedSession.id);
        if (!wake.success) {
          const currentStatus = db.getSession(resolvedSession.id)?.status;
          if (currentStatus !== "loading" && currentStatus !== "running") {
            throw new Error(wake.error || "Failed to wake sleeping model");
          }
        }
      }

      // A message sent while the model is LOADING (cold start, Save & Restart,
      // or the wake above still finishing) queues exactly once until the
      // lifecycle settles. The blind 30s health poll below used to fail these
      // messages while a legitimate multi-minute load was in progress; the
      // user watches live load progress instead, and Stop/cancel aborts the
      // queued message immediately.
      if (!isRemote && resolvedSession?.id) {
        const liveStatus = db.getSession(resolvedSession.id)?.status;
        if (liveStatus === "loading") {
          const settleTimeoutMs = settleTimeoutMsFor(resolvedSession);
          console.log(
            `[CHAT] Session ${resolvedSession.id.slice(0, 8)} is loading — queueing message until ready (up to ${Math.round(settleTimeoutMs / 1000)}s)`,
          );
          const settled = await sessionManager.waitForSessionLifecycleSettled(
            resolvedSession.id,
            { timeoutMs: settleTimeoutMs, signal: abortController.signal },
          );
          if (settled !== "running") {
            throw new Error(
              settled === "loading"
                ? "Model is still loading — try again once the status indicator turns green"
                : `Model did not become ready (status: ${settled})`,
            );
          }
        }
      }

      const rawBaseUrl =
        isRemote && resolvedSession?.remoteUrl
          ? resolvedSession.remoteUrl.replace(/\/+$/, "")
          : `http://${connectHost(resolved.host)}:${resolved.port}`;
      // Resolve .local mDNS hostnames to IPv4 — Node.js fetch resolves them to
      // unreachable IPv6 link-local addresses (fe80::...) causing "fetch failed"
      const baseUrl = await resolveUrl(rawBaseUrl);
      console.log(
        `[CHAT] Endpoint resolution: isRemote=${isRemote}, rawBaseUrl=${rawBaseUrl}, baseUrl=${baseUrl}, session=${resolvedSession?.id ?? "none"}, type=${resolvedSession?.type ?? "none"}`,
      );
      const authHeaders: Record<string, string> = resolvedSession?.id
        ? getAuthHeaders(resolvedSession.id)
        : {};
      // Update active request entry with resolved baseUrl and auth for cancel support
      const activeEntry = activeRequests.get(chatId);
      if (activeEntry) {
        activeEntry.endpoint = { host: resolved.host, port: resolved.port };
        activeEntry.baseUrl = baseUrl;
        if (Object.keys(authHeaders).length > 0)
          activeEntry.authHeaders = authHeaders;
      }

      // Health check with retry — wait for server to become ready instead of
      // failing immediately. This prevents orphaned user messages and allows
      // chatting as soon as the server finishes loading.
      //
      // OPTIMIZATION: If the global health monitor confirmed this session healthy
      // within the last 15 seconds, skip the per-message health check entirely.
      // The global monitor runs every 5s, so 15s gives a generous window.
      // This avoids adding 100-500ms+ RTT on every single message for remote sessions.
      const recentlyHealthy = resolvedSession?.id
        ? Date.now() - sessionManager.getLastHealthyAt(resolvedSession.id) <
          15_000
        : false;

      // Remote sessions: 1 quick attempt then proceed (the request itself has a timeout).
      // Local sessions: 15 retries with 2s delays (30s total — JANG models need longer to dequantize).
      const maxHealthRetries = isRemote ? 1 : 15;
      const healthRetryDelay = isRemote ? 500 : 2000;
      let healthOk = recentlyHealthy;
      if (recentlyHealthy) {
        console.log(
          `[CHAT] Skipping health check — global monitor confirmed healthy within 15s`,
        );
      } else {
        const healthUrl = isRemote
          ? `${baseUrl}/v1/models`
          : `${baseUrl}/health`;
        console.log(
          `[CHAT] Health check URL: ${healthUrl} (${isRemote ? "remote" : "local"}, max ${maxHealthRetries} attempts)`,
        );
        for (let attempt = 0; attempt < maxHealthRetries; attempt++) {
          try {
            const healthRes = await fetch(healthUrl, {
              headers: authHeaders,
              signal: AbortSignal.timeout(isRemote ? 3000 : 5000),
            });
            if (healthRes.ok) {
              // Local /health answers 200 in EVERY state (no_model, standby,
              // waking). HTTP 200 alone is not inference readiness — the
              // body must say healthy + model_loaded + contract-ready.
              let bodyReady = isRemote;
              if (!isRemote) {
                try {
                  bodyReady = localEngineReadyFromHealthBody(await healthRes.json());
                } catch {
                  bodyReady = false;
                }
              }
              if (bodyReady) {
                healthOk = true;
                console.log(
                  `[CHAT] Health check passed on attempt ${attempt + 1}`,
                );
                break;
              }
              console.log(
                `[CHAT] Server up but not inference-ready (attempt ${attempt + 1}/${maxHealthRetries})`,
              );
              const live = resolvedSession?.id
                ? db.getSession(resolvedSession.id)
                : undefined;
              if (live?.status === "standby") {
                // The engine was slept externally and the monitor synced the
                // row after this send began. Wake through SessionManager (the
                // visible progress owner) and queue until readiness.
                console.log(
                  `[CHAT] Engine asleep mid-poll — waking session ${live.id.slice(0, 8)} and queueing`,
                );
                const wake = await sessionManager.wakeSession(live.id);
                if (wake.success) {
                  const settled = await sessionManager.waitForSessionLifecycleSettled(
                    live.id,
                    { timeoutMs: settleTimeoutMsFor(live), signal: abortController.signal },
                  );
                  if (settled === "running") continue;
                }
              }
              if (attempt < maxHealthRetries - 1) {
                await new Promise((r) => setTimeout(r, healthRetryDelay));
              }
              continue;
            }
            if (attempt < maxHealthRetries - 1) {
              console.log(
                `[CHAT] Server not ready (HTTP ${healthRes.status}), retrying in ${healthRetryDelay}ms...`,
              );
              await new Promise((r) => setTimeout(r, healthRetryDelay));
            }
          } catch (healthErr: any) {
            console.log(
              `[CHAT] Health check failed (attempt ${attempt + 1}/${maxHealthRetries}): ${healthErr.message || healthErr.cause?.message || healthErr}`,
            );
            if (attempt < maxHealthRetries - 1) {
              await new Promise((r) => setTimeout(r, healthRetryDelay));
            }
          }
        }
      }
      // For remote sessions: proceed even if health check failed — the request has
      // its own timeout and the server may just be busy with another generation.
      if (!healthOk && isRemote) {
        console.log(
          `[CHAT] Remote health check failed but proceeding anyway — request will use its own timeout`,
        );
        healthOk = true;
      }
      if (!healthOk) {
        // Entry + timer cleanup handled by the enclosing finally.
        throw new Error(
          `Cannot reach server on port ${resolved.port} after ${maxHealthRetries} attempts (${(maxHealthRetries * healthRetryDelay) / 1000}s). The model may still be loading — wait for the status indicator to turn green, then try again.`,
        );
      }

      // Add user message AFTER health check passes — this prevents orphaned
      // user messages when the server isn't ready yet.
      // When attachments are present, store content as JSON array of content parts.
      const hasAttachments = attachments && attachments.length > 0;
      // mlxstudio#69: explicit media attachments override chatIsMultimodal
      // detection. The user clicked "attach image" — that intent must be
      // honored even when (a) the session lookup failed, (b) the session
      // config has isMultimodal=false from an older save, or (c) the model
      // dir's config.json doesn't expose vision_config. The downstream
      // server will reject the request properly if the model truly cannot
      // handle media, which is far better than silently dropping it. Text-file
      // attachments are plain text context and do not need multimodal routing.
      const hasMediaAttachments =
        hasAttachments && attachments!.some((a) => inferKind(a) !== "text");
      const modelForceTextOnly = (() => {
        try {
          return !!chat.modelPath &&
            detectModelConfigFromDir(chat.modelPath).forceTextOnly === true;
        } catch (_) {
          return false;
        }
      })();
      if (hasMediaAttachments && modelForceTextOnly) {
        const imgs = attachments!.filter((a) => inferKind(a) === "image").length;
        const vids = attachments!.filter((a) => inferKind(a) === "video").length;
        const auds = attachments!.filter((a) => inferKind(a) === "audio").length;
        console.log(
          `[CHAT] Keeping multimodal=false for ${chatId} — model is forceTextOnly and user attached ${imgs} image(s), ${vids} video(s), ${auds} audio file(s)`,
        );
      } else if (hasMediaAttachments && !chatIsMultimodal) {
        const imgs = attachments!.filter((a) => inferKind(a) === "image").length;
        const vids = attachments!.filter((a) => inferKind(a) === "video").length;
        const auds = attachments!.filter((a) => inferKind(a) === "audio").length;
        console.log(
          `[CHAT] Forcing multimodal=true for ${chatId} — user attached ${imgs} image(s), ${vids} video(s), ${auds} audio file(s)`,
        );
        chatIsMultimodal = true;
      }
      if (hasAttachments || chatIsMultimodal) {
        pushChatSessionLog(
          chatSession?.id || resolvedSession?.id,
          `[CHAT_DIAG] attachment_route=${JSON.stringify({
            chatId: chatId.slice(0, 8),
            modelPath: chat.modelPath,
            detectedFamily: chatDetectedFamily,
            modelForceTextOnly,
            chatIsMultimodal,
            attachments: summarizeAttachmentsForLog(attachments),
          })}`,
        );
      }
      const audioFormatFromDataUrl = (dataUrl: string): string => {
        const mime = dataUrl.match(/^data:([^;,]+)[;,]/)?.[1]?.toLowerCase() || "";
        if (mime === "audio/mpeg" || mime === "audio/mp3") return "mp3";
        if (mime === "audio/wave" || mime === "audio/x-wav" || mime === "audio/wav") return "wav";
        if (mime === "audio/mp4" || mime === "audio/x-m4a") return "m4a";
        if (mime.startsWith("audio/")) return mime.slice("audio/".length);
        return "wav";
      };
      const audioDataFromDataUrl = (dataUrl: string): string =>
        dataUrl.includes(",") ? dataUrl.split(",", 2)[1] : dataUrl;
      const userContentForDb = hasAttachments
        ? JSON.stringify(
            orderComposerContentParts(
              content,
              attachments.map((a) => {
                const kind = inferKind(a);
                if (kind === "audio") {
                  return {
                    type: "input_audio",
                    input_audio: {
                      data: audioDataFromDataUrl(a.dataUrl),
                      format: audioFormatFromDataUrl(a.dataUrl),
                    },
                  };
                }
                if (kind === "text") {
                  return {
                    type: "text",
                    text: `[Attached file: ${a.name}]\n${a.text || ""}`.trim(),
                  };
                }
                return kind === "video"
                  ? { type: "video_url", video_url: { url: a.dataUrl } }
                  : { type: "image_url", image_url: { url: a.dataUrl } };
              }),
              chatDetectedFamily,
            ),
          )
        : content;
      const userMessage: Message = {
        id: uuidv4(),
        chatId,
        role: "user",
        content: userContentForDb,
        timestamp: Date.now(),
      };
      db.addMessage(userMessage);

      // Generate assistant message ID upfront so typing indicator can reference it
      const assistantMessageId = uuidv4();
      // Local proof correlation is carried out-of-band so it cannot perturb the
      // model request body or remote-provider contracts. One visible user turn
      // owns one proof/message identity; every initial/tool-continuation HTTP
      // request gets its own request ID.
      const proofRequestId = userMessage.id;
      const wireRequestIds: string[] = [];
      const nextLocalRequestCorrelationHeaders = (): Record<string, string> => {
        if (isRemote) return {};
        const requestId = uuidv4();
        wireRequestIds.push(requestId);
        return {
          "X-vMLX-Proof-Request-ID": proofRequestId,
          "X-vMLX-Request-ID": requestId,
          "X-vMLX-Message-ID": assistantMessageId,
        };
      };

      // Signal to renderer that the model is processing (typing indicator during TTFT)
      try {
        const win = getWindow();
        if (win && !win.isDestroyed()) {
          win.webContents.send("chat:typing", {
            chatId,
            messageId: assistantMessageId,
          });
        }
      } catch (_) {}

      // Get messages for context
      const messages = db.getMessages(chatId);

      // Re-validate persisted values at the request boundary. Older database
      // rows and imported profiles can predate the current Chat/Responses
      // schema domains; forwarding one of those rows verbatim turns a valid UI
      // action into a backend 422. Invalid legacy fields revert to inheritance,
      // exactly like an unset override, rather than being silently clamped to a
      // different sampler value.
      const overrides = sanitizeChatOverrides({
        ...(db.getChatOverrides(chatId) || {}),
        chatId,
      });
      // Chat Settings displays Native-MTP's effective greedy tuple, but those
      // controls are intentionally disabled and therefore are not persisted as
      // synthetic per-chat overrides. Apply the same shared session policy at
      // the actual request boundary so the visible contract and the Chat /
      // Responses wire bodies cannot diverge. Remote providers retain their
      // own sampling contract.
      const effectiveSamplerOverrides = isRemote
        ? overrides
        : applyMtpSamplerOverrides(
            overrides,
            chatSessionConfig,
            chatNativeMtp,
          );
      if (!isRemote && supportsInstructMode === false && overrides?.enableThinking === false) {
        throw new Error(
          "This model has no native Thinking Off/Instruct mode. Open Chat Settings and choose Auto or On.",
        );
      }

      // Build request messages with system prompt if set
      // Using any[] to support tool_calls and tool_call_id fields
      const requestMessages: any[] = [];
      // Boundary marker: index into requestMessages where THIS turn's fresh tool
      // exchange begins (set right after history replay is assembled). The
      // tool-call/result harvest that persists into this assistant row must only
      // scan from here — scanning the whole array re-harvests every prior turn's
      // replayed tool calls/results into this row, so each turn stores the union
      // of all prior tool exchanges and the payload grows super-linearly across
      // turns (measured 3.3k→12.5k→21.3k tokens), starving the answer budget.
      let currentTurnToolStart = 0;

      // Add system prompt from overrides if available, or agentic prompt when built-in tools enabled
      const hasSystemPrompt = !!overrides?.systemPrompt;
      // The persisted content of a media turn is an array containing its text
      // and attachment parts. Always bind authorization to the newest user
      // message; falling back to an older string turn can silently re-enable a
      // tool that the current image/video/audio prompt explicitly prohibited.
      const latestUserText = latestUserMessageText(messages);
      const suppressAgenticToolPromptForExactOutput =
        overrides?.builtinToolsEnabled === true &&
        // Exact-output probes usually say "reply exactly MARKER" without a
        // colon. Injecting the broad coding-agent prompt in that case repeats
        // the final-response instruction and made Nemotron emit the requested
        // marker twice after a correct tool result.
        /\breply exactly\b/i.test(latestUserText);
      // These are current-user authorization constraints, not built-in-tool
      // preferences. The local engine auto-merges MCP schemas even when the
      // Electron built-in catalog is disabled, so recognize a no-tools turn
      // independently of the persistent built-in-tools toggle.
      const userForbidsToolCalls = requestsNoToolCalls(latestUserText);
      const exactTextOnlyNoToolTurn =
        requestsExactTextOnlyWithoutToolUse(latestUserText);
      const privateReasoningNoToolTurn =
        requestsPrivateReasoningWithoutToolUse(latestUserText);
      const attachBuiltinToolsForCurrentTurn =
        overrides?.builtinToolsEnabled === true &&
        !userForbidsToolCalls &&
        !exactTextOnlyNoToolTurn &&
        !privateReasoningNoToolTurn;
      // Parse explicit built-in contracts even when the catalog or one of its
      // categories is disabled. Otherwise the request can silently ignore the
      // user's tool requirement, or (with a category disabled) send a required
      // tool_choice without the matching schema.
      const exactlyOnceToolNames = requestedOnceToolNames(latestUserText);
      const exactlyOnceBuiltinToolNames =
        exactlyOnceToolNames.filter((name) => isBuiltinTool(name));
      const completedExactFinalTools = new Set<string>();
      const completedExactlyOnceTools = new Set<string>();
      let lastReplaySafeToolResultKey: string | null = null;
      const suppressGenericAgenticToolPromptForNativeTools =
        overrides?.builtinToolsEnabled === true &&
        shouldSuppressGenericAgenticPromptForNativeTools(
          chatDetectedFamily,
          chat.modelPath || resolvedSession?.remoteModel || chat.modelId,
        );
      const directMediaAttachmentRule =
        // Attachment presence is not a tool-capability change. Keep both the
        // catalog and this conditional instruction stable across media/text
        // turns so historical media prefixes remain reusable. Explicit tool
        // restrictions and category toggles still control the catalog below.
        (chatIsMultimodal || isRemote) && attachBuiltinToolsForCurrentTurn
          ? DIRECT_MEDIA_ATTACHMENT_TOOL_RULE
          : "";
      const unscopedCurrentTurnToolDefinitions =
        attachBuiltinToolsForCurrentTurn
          ? filterTools(overrides || {}, {
              zayaAppleScriptToolBundle: chatUsesZayaAppleScriptToolBundle,
            })
          : [];
      const exactFinalToolNames =
        overrides?.builtinToolsEnabled === true
          ? requestedExactFinalToolNames(
              latestUserText,
              toolCapabilityNames(unscopedCurrentTurnToolDefinitions),
            )
          : [];
      const exactFinalBuiltinToolNames =
        exactFinalToolNames.filter((name) => isBuiltinTool(name));
      const scopedCurrentTurnToolNames =
        overrides?.builtinToolsEnabled === true
          ? requestedScopedToolNames(
              latestUserText,
              toolCapabilityNames(unscopedCurrentTurnToolDefinitions),
            )
          : [];
      const scopedCurrentTurnBuiltinToolNames =
        scopedCurrentTurnToolNames.filter((name) => isBuiltinTool(name));
      const restrictedToolNamesForCurrentTurn =
        exactFinalBuiltinToolNames.length > 0
          ? exactFinalBuiltinToolNames
          : scopedCurrentTurnBuiltinToolNames;
      const directAnswerAfterSingleTool =
        exactFinalBuiltinToolNames.length === 1;
      const currentTurnToolDefinitions = attachBuiltinToolsForCurrentTurn
        ? scopeToolDefinitionsByName(
            unscopedCurrentTurnToolDefinitions,
            exactFinalBuiltinToolNames.length > 0
              ? exactFinalBuiltinToolNames
              : scopedCurrentTurnBuiltinToolNames,
          )
        : [];
      const currentToolCapabilityFingerprint = toolCapabilityFingerprint(
        currentTurnToolDefinitions,
      );
      const priorAssistantToolFingerprints = messages
        .filter((message) => message.role === "assistant")
        .map((message) => message.toolCapabilityFingerprint);
      const currentToolNames = toolCapabilityNames(currentTurnToolDefinitions);
      const unavailableExactlyOnceBuiltinToolNames =
        unavailableRequestedToolNames(
          exactlyOnceBuiltinToolNames,
          currentToolNames,
        );
      const historicallyUnavailableTools = historicalUnavailableToolNames(
        messages,
        currentToolCapabilityFingerprint,
        currentToolNames,
      );
      // Only an explicit current-user prohibition is a hard authorization
      // boundary. Exact-output and private-reasoning heuristics may omit the
      // Electron built-in catalog to avoid prompt contamination, but the local
      // engine can still expose MCP tools that the request genuinely needs
      // (for example, "check weather and reply exactly RAIN or DRY").
      const currentPromptAlreadyForbidsTools = userForbidsToolCalls;
      const toolCapabilityEpoch = toolCapabilityEpochInstruction(
        priorAssistantToolFingerprints,
        currentToolCapabilityFingerprint,
        currentToolNames,
        historicallyUnavailableTools,
        currentPromptAlreadyForbidsTools,
      );
      if (hasSystemPrompt && overrides?.builtinToolsEnabled) {
        const toolRule =
          "\n\nIMPORTANT: After using any tools, provide a final response. If the user explicitly requested exact final wording or a strict output format, follow that format exactly; otherwise provide a substantive response explaining what you found or did. Never stop after just executing tools.";
        requestMessages.push({
          role: "system",
          content:
            overrides!.systemPrompt! +
            (userForbidsToolCalls ||
            suppressAgenticToolPromptForExactOutput ||
            suppressGenericAgenticToolPromptForNativeTools
              ? ""
              : toolRule) +
            directMediaAttachmentRule,
        });
      } else if (hasSystemPrompt) {
        requestMessages.push({
          role: "system",
          content: overrides!.systemPrompt!,
        });
      } else if (
        attachBuiltinToolsForCurrentTurn &&
        !userForbidsToolCalls &&
        !suppressAgenticToolPromptForExactOutput &&
        !suppressGenericAgenticToolPromptForNativeTools
      ) {
        requestMessages.push({
          role: "system",
          content: AGENTIC_SYSTEM_PROMPT + directMediaAttachmentRule,
        });
      } else if (directMediaAttachmentRule) {
        requestMessages.push({
          role: "system",
          content: directMediaAttachmentRule.trim(),
        });
      }
      if (toolCapabilityEpoch) {
        requestMessages.push({
          role: "system",
          content: toolCapabilityEpoch,
        });
      }
      // No default system prompt injected — let the model's native template handle defaults.
      // Injecting "You are a helpful assistant." reinforces safety behavior in abliterated/CRACK models.

      // Determine wire format before rebuilding history because persisted
      // tool-call context must be reconstructed differently for Responses
      // (`function_call` / `function_call_output` items) versus Chat
      // Completions (`assistant.tool_calls` / `role:"tool"` messages).
      const wireApi =
        overrides?.wireApi || (isRemote ? "completions" : "responses");
      const useResponsesApi = wireApi === "responses";
      // Local vmlx-engine merges MCP schemas behind the public request. An
      // explicit no-tools turn therefore needs tool_choice=none, while a
      // singular exact built-in contract needs a specific function choice so
      // the engine filters its MCP catalog to that same name. Open-ended turns
      // intentionally leave tool_choice omitted so legitimate MCP use remains
      // available. Remote providers only see the already-scoped request.tools
      // catalog and do not participate in the local MCP merge. They still need
      // a specific tool_choice for an explicit singular contract; one schema
      // by itself is only auto choice and can terminate without calling it.
      const currentTurnToolChoice = () => {
        const remainingExactBuiltinTools = exactFinalBuiltinToolNames.filter(
          (name) => !completedExactFinalTools.has(name),
        );
        const remainingExactlyOnceBuiltinTools =
          exactlyOnceBuiltinToolNames.filter(
            (name) => !completedExactlyOnceTools.has(name),
          );
        const requiredToolNames = requiredToolChoiceNamesForCurrentTurn(
          remainingExactBuiltinTools,
          remainingExactlyOnceBuiltinTools,
        );
        return toolChoiceForCurrentTurn(
          currentPromptAlreadyForbidsTools,
          requiredToolNames,
          useResponsesApi ? "responses" : "chat",
          isRemote,
        );
      };

      // Add conversation messages (skip any existing system messages to avoid duplicates)
      // Messages with JSON content arrays (multimodal) are parsed back to content parts for the API
      for (const m of messages) {
        if (
          m.role === "system" &&
          (hasSystemPrompt || overrides?.builtinToolsEnabled)
        )
          continue;
        let msgContent: any = m.content;
        // Panel-synthesized error bubbles ("Message not sent — …" /
        // "Generation blocked: …") are UI-only annotations — never replay them
        // as assistant history. For prompt-too-long, ALSO drop the user message
        // that produced the bubble: that message IS the oversized payload, and
        // replaying it re-413s every later turn in the chat.
        if (m.role === "assistant" && isPromptTooLongBubbleContent(m.content)) {
          const prev = requestMessages[requestMessages.length - 1];
          if (prev && prev.role === "user") requestMessages.pop();
          continue;
        }
        if (m.role === "assistant" && isMetalHeadroomBubbleContent(m.content)) {
          continue;
        }
        // Strip "[Generation interrupted]" markers from previous assistant messages —
        // these are UI-only annotations saved to DB on abort, not meant for the model
        if (m.role === "assistant" && typeof msgContent === "string") {
          msgContent = msgContent
            .replace(/\n\n\[Generation interrupted\]$/, "")
            .replace(/^\[Generation interrupted\]$/, "");
          // Strip any leaked <think> blocks from prior messages when thinking is OFF.
          // These can leak if server didn't catch them or model was mid-think on abort.
          // Without stripping, the model sees prior thinking in context and mimics it.
          if (overrides?.enableThinking === false) {
            msgContent = msgContent.replace(/<think>[\s\S]*?<\/think>\s*/g, "");
            msgContent = msgContent.replace(/<mm:think>[\s\S]*?<\/mm:think>\s*/g, "");
            msgContent = msgContent.replace(
              /\[THINK\][\s\S]*?\[\/THINK\]\s*/g,
              "",
            );
          }
          // 2026-05-03: keep empty-content assistant turns when they
          // carry tool_calls. An assistant turn that ONLY emits tool
          // calls has no visible content — but the chat template still
          // needs the message in history with `tool_calls` set so the
          // model sees prior calls + their results. Skipping it
          // collapsed the conversation into "user → user" gibberish
          // on continuation.
          const _hasOaiToolCalls =
            typeof (m as any).toolCallsOaiJson === "string" &&
            (m as any).toolCallsOaiJson.length > 0;
          // Codex 2026-05-06 B2: don't drop reasoning-only assistant
          // rows. A DSV4 turn that hits max_tokens during <think> emits
          // empty output_text (correct — see server.py B1 fix) but
          // populated reasoning_content. Dropping the row here would
          // collapse the conversation into "user → user" pairs and
          // cause the next turn to replay/merge wrong-prompt framing.
          // Treat reasoning_content as substantive content for the
          // skip-empty check.
          const _hasReasoningContent =
            typeof (m as any).reasoningContent === "string" &&
            (m as any).reasoningContent.trim().length > 0;
          if (
            !msgContent.trim()
            && !_hasOaiToolCalls
            && !_hasReasoningContent
          ) continue; // Skip entirely empty aborted messages
        }
        // Detect JSON content arrays (multimodal messages with images)
        if (
          m.role === "user" &&
          typeof msgContent === "string" &&
          msgContent.startsWith("[")
        ) {
          try {
            const parsed = JSON.parse(msgContent);
            if (Array.isArray(parsed) && parsed.length > 0 && parsed[0].type) {
              // Preserve every historical media item for a real multimodal
              // route. Replacing prior bytes with explanatory text rewrites
              // both the token stream and media identity, defeating causal SSD
              // prefix reuse and cold/restart reconstruction. Text-only routes
              // still strip media defensively.
              msgContent = replayPersistedUserContentParts(
                parsed,
                chatIsMultimodal || isRemote,
              );
            }
          } catch {
            /* not JSON, use as plain string */
          }
        }
        // Rebuild each persisted assistant row into its original model-visible
        // order. A tool loop is stored as one UI row, but native history must be
        // reasoning -> function_call -> result -> reasoning -> final answer.
        // This also preserves reasoning-only assistant turns instead of keeping
        // them merely as a UI display field.
        if (m.role === "assistant") {
          const replay = replayPersistedAssistantHistory(
            { ...(m as any), content: msgContent },
            useResponsesApi,
            {
              // A current explicit no-tool instruction owns authorization for
              // this turn. Preserve visible history and real call/result
              // records, but do not feed the model older private reasoning
              // that may discuss a now-forbidden tool contract as if it were
              // still current system authority.
              includeReasoning: !currentPromptAlreadyForbidsTools,
            },
          );
          if (replay.length > 0) {
            requestMessages.push(...replay);
            continue;
          }
        }
        const reqMsg: any = { role: m.role, content: msgContent };
        requestMessages.push(reqMsg);
      }

      // Fix role alternation: merge consecutive same-role messages to prevent
      // template errors (e.g., Mistral enforces strict user/assistant alternation).
      // This can happen when an aborted assistant message is stripped empty, leaving
      // two consecutive user messages in history.
      const mergedMessages: typeof requestMessages = [];
      for (const msg of requestMessages) {
        const prev = mergedMessages[mergedMessages.length - 1];
        if (
          prev &&
          prev.role === msg.role &&
          typeof prev.content === "string" &&
          typeof msg.content === "string"
        ) {
          prev.content = prev.content + "\n\n" + msg.content;
        } else {
          mergedMessages.push({ ...msg });
        }
      }
      requestMessages.length = 0;
      requestMessages.push(...mergedMessages);
      // History replay is now fully assembled (system + all prior turns + current
      // user message). Everything pushed after this point is THIS turn's fresh
      // tool exchange (the agentic tool loop). Harvest only from here.
      currentTurnToolStart = requestMessages.length;

      // Prepare assistant message placeholder
      const assistantMessage: Message = {
        id: assistantMessageId,
        chatId,
        role: "assistant",
        content: "",
        timestamp: Date.now(),
        toolCapabilityFingerprint: currentToolCapabilityFingerprint,
      };

      // Metrics tracking
      const startTime = Date.now();
      let fetchStartTime = startTime; // Updated just before the API fetch (for accurate TTFT)
      let tokenCount = 0;
      let promptTokens = 0;
      let cachedTokens = 0;
      // Exchange totals across tool-loop streams. promptTokens/cachedTokens
      // are per-stream (usage chunks are cumulative within one stream, so
      // recordCacheUsage takes a max); pairing the LAST stream's prompt with
      // the max cached across ALL streams displayed impossible stats like
      // "481 prompt (3904 cached)". Streams fold into these totals at each
      // follow-up boundary and the final metrics report coherent sums.
      let exchangePromptTokens = 0;
      let exchangeCachedTokens = 0;
      // Per-HTTP-pass provenance for prompt/cache counts. This remains in the
      // outer scope so interrupted responses can refuse client-estimated pp/s.
      let serverSendsUsage = false;
      let cacheDetail = "";
      const recordCacheUsage = (details: any) => {
        const nextCachedTokens = Number(details?.cached_tokens);
        if (Number.isFinite(nextCachedTokens) && nextCachedTokens > 0) {
          cachedTokens = Math.max(cachedTokens, nextCachedTokens);
        }
        if (typeof details?.cache_detail === "string") {
          cacheDetail = mergeCacheDetails(cacheDetail, details.cache_detail);
        }
      };
      let firstTokenTime: number | null = null;
      // Track actual generation time (excludes PP and tool execution pauses)
      let generationMs = 0;
      let lastTokenTime: number | null = null;
      // Rolling window for live TPS: circular buffer of (timestamp, tokenCount) snapshots.
      // Uses actual token count deltas for accurate throughput — handles multi-token SSE chunks
      // correctly (e.g., reasoning batches where each chunk may contain 2+ tokens).
      const TPS_BUFFER_SIZE = 30;
      const tpsSnapshots: Array<[number, number]> = []; // [timestamp, relative tokenCount]
      let liveTps = 0;
      // Keep a bounded history across tool iterations. The last rolling value
      // can represent only a tiny final-answer tail after a long pause, while
      // the displayed token count is cumulative across every reasoning/tool
      // round. Final metrics need a representative stream rate, not merely the
      // last ten tokens of the last HTTP request.
      const liveTpsHistory: number[] = [];
      const MAX_LIVE_TPS_SAMPLES = 512;
      let tpsTokenBase = 0; // re-anchor point for tpsSnapshots after iteration reset
      // Local vMLX Responses streams negotiate exact engine-side decode
      // windows. Keep one latest cumulative snapshot per HTTP pass, then sum
      // passes across the visible agent/tool exchange. SSE arrival timing is
      // only a fallback: --stream-interval batches tokens, and tool-control
      // output can advance usage without producing a visible delta.
      const completedServerDecodePasses: ServerDecodePass[] = [];
      let currentServerDecodePass: ServerDecodePass | undefined;
      const recordServerDecodeUsage = (usage: unknown) => {
        const decoded = parseServerDecodeUsage(usage);
        if (!decoded) return;
        currentServerDecodePass = decoded;
        liveTps = decoded.tokensPerSecond;
      };
      const finishServerDecodePass = () => {
        if (currentServerDecodePass) {
          completedServerDecodePasses.push(currentServerDecodePass);
          currentServerDecodePass = undefined;
        }
      };
      // No streaming throttle — emit every token. Renderer-side useTypewriter
      // in MessageBubble.tsx handles smooth character reveal via rAF.
      let reader: ReadableStreamDefaultReader<Uint8Array> | undefined;
      // (thinkingTimer removed — "Thinking silently" indicator disabled)
      let fullContent = "";
      let reasoningContent = "";
      let reasoningSegments: string[] = [];
      const currentReasoningContent = () =>
        joinReasoningSegments(reasoningSegments) || reasoningContent;
      const currentReasoningSegments = () =>
        visibleReasoningSegments(reasoningSegments);
      const currentReasoningSegment = () => {
        const segments = currentReasoningSegments();
        return segments[segments.length - 1] || reasoningContent;
      };
      const responsesReasoningItem = (text: string) => ({
        type: "reasoning",
        content: [{ type: "reasoning", text }],
      });
      let responseWarnings: string[] | null = null;
      // A model can complete the first post-tool stream with reasoning but no
      // visible answer. The one recovery request is intentionally answer-only:
      // it preserves the completed tool result in history while removing tool
      // schemas and putting local models on their direct (thinking-off) rail.
      // This is scoped to the recovery request; the normal tool loop keeps the
      // user's reasoning setting and supports additional tool calls.
      let finalAnswerRecovery = false;
      // An explicit "exactly once; after the tool result, reply exactly ..."
      // contract has no valid second execution. Keep its rendered schema prefix
      // stable for SSD reuse; the local fulfilled header plus the execution
      // dedupe below prevent a second call without rewriting that prefix.
      let plannedDirectAnswerPass = false;
      let previousToolRequestFields: ToolRequestFields | undefined;
      // Accumulates content across tool iterations so abort during tool execution can recover
      // earlier content that would otherwise be lost when fullContent is reset between iterations
      let allGeneratedContent = "";
      // Per-iteration token count for auto-continue threshold (tokenCount is cumulative)
      let iterationTokenCount = 0;
      let iterationTokenBase = 0; // tokenCount at start of iteration (for server-usage delta)
      // Cumulative token offset: tracks total tokens from completed iterations.
      // Server restarts completion_tokens from 0 on each new HTTP request, so
      // raw tokenCount only reflects the current iteration. This offset + iterationTokenCount
      // gives the true total across all tool iterations.
      let cumulativeTokenOffset = 0;
      // Collect tool statuses for DB persistence (mirrors what's emitted to renderer)
      const collectedToolStatuses: Array<{
        phase: string;
        toolName: string;
        toolCallId?: string;
        detail?: string;
        iteration?: number;
        contentOffset?: number;
      }> = [];
      let toolStatusNeedsFlush = false;
      // A single ReadableStream read can contain dozens of SSE events. Without
      // yielding between visible deltas, Electron queues every chat:stream IPC
      // message in one main-process turn and React coalesces the whole answer
      // into the terminal render. Reasoning appears live, then content appears
      // all at once even though the server emitted token-level deltas.
      let rendererStreamNeedsFlush = false;
      // Declared outside try so catch block can access them for error recovery
      let isReasoning = false;
      let lastFinishReason: string | undefined;
      // Periodic DB save interval — saves content every 5s so it survives navigation/crashes
      let periodicSaveInterval: ReturnType<typeof setInterval> | null = null;

      // Pre-insert assistant message to DB immediately so periodic updates have a row to update.
      // Uses INSERT OR REPLACE so the final addMessage at completion overwrites cleanly.
      db.addMessage(assistantMessage);

      const startPeriodicSave = () => {
        if (periodicSaveInterval) return;
        periodicSaveInterval = setInterval(() => {
          const saveContent = allGeneratedContent
            ? fullContent.trim()
              ? allGeneratedContent + "\n\n" + fullContent.trim()
              : allGeneratedContent
            : fullContent;
          const saveReasoning = currentReasoningContent();
          if (saveContent || saveReasoning) {
            try {
              db.updateMessageContent(
                assistantMessage.id,
                saveContent,
                saveReasoning || undefined,
                currentReasoningSegments().length > 0
                  ? JSON.stringify(currentReasoningSegments())
                  : undefined,
              );
            } catch (_) {}
          }
        }, 5000);
      };
      const stopPeriodicSave = () => {
        if (periodicSaveInterval) {
          clearInterval(periodicSaveInterval);
          periodicSaveInterval = null;
        }
      };

      try {
        if (unavailableExactlyOnceBuiltinToolNames.length > 0) {
          const quotedNames = unavailableExactlyOnceBuiltinToolNames
            .map((name) => `"${name}"`)
            .join(", ");
          const subject =
            unavailableExactlyOnceBuiltinToolNames.length === 1
              ? `tool ${quotedNames} is`
              : `tools ${quotedNames} are`;
          throw new Error(
            `Requested built-in ${subject} disabled or unavailable for this turn. Enable the relevant Chat Settings category and retry.`,
          );
        }
        // Call API (local vMLX Engine or remote OpenAI-compatible endpoint)
        const apiUrl = useResponsesApi
          ? `${baseUrl}/v1/responses`
          : `${baseUrl}/v1/chat/completions`;
        console.log(
          `[CHAT] Sending to: ${apiUrl} (wire: ${wireApi}, remote: ${isRemote})`,
        );

        // Get model name: remote uses configured model, local reads from health endpoint
        let modelName = isRemote
          ? resolvedSession?.remoteModel || chat.modelId || "default"
          : chat.modelId || "default";
        if (!isRemote) {
          try {
            const healthRes = await fetch(`${baseUrl}/health`, {
              signal: AbortSignal.timeout(1000),
            });
            if (healthRes.ok) {
              const health = await healthRes.json();
              if (health.model_name) modelName = health.model_name;
            }
          } catch (_) {
            /* use fallback */
          }
        }

        // Only send stop sequences when the user explicitly sets them in chat settings.
        // The server already handles stop tokens via the model's chat template — sending
        // all template tokens for every model risks false-positive stops (e.g. Qwen hitting </s>).
        const stopSequences = overrides?.stopSequences
          ? overrides.stopSequences
              .split(",")
              .map((s: string) => s.trim())
              .filter(Boolean)
          : undefined;

        const availableToolDefinitions = () =>
          currentTurnToolDefinitions.filter(
            (tool) => {
              const name = tool.function.name.toLowerCase();
              return (
                !completedExactFinalTools.has(name) &&
                !completedExactlyOnceTools.has(name)
              );
            },
          );

        // Build request body — shared between initial request and tool follow-ups
        const buildRequestBody = (): Record<string, any> => {
          const resolvedOutputBudget = dsv4OutputBudget(
            overrides?.maxTokens,
            overrides?.enableThinking,
            chatDetectedFamily,
            overrides?.reasoningEffort,
          );
          const resolvedThinkingBudget =
            typeof overrides?.maxThinkingTokens === "number" &&
              Number.isFinite(overrides.maxThinkingTokens) &&
              overrides.maxThinkingTokens > 0
              ? Math.floor(overrides.maxThinkingTokens)
              : undefined;
          const effectiveEnableThinkingOverride =
            !isRemote &&
            !sessionHasReasoningParser &&
            chatDetectedFamily !== "deepseek-v4"
              ? undefined
              : overrides?.enableThinking;
          const applyThinkingBudget = (obj: Record<string, any>) => {
            // This used to be named for the LOCAL path and returned early on
            // isRemote. That was right while the capability was unknowable over
            // the wire -- but the engine now emits supports_thinking_budget in
            // /v1/capabilities, so a remote session knows it too. Without this,
            // making the control RENDER remotely just made it decorative: the
            // user set a budget and neither max_thinking_tokens nor
            // chat_template_kwargs.thinking_budget ever left the app.
            //
            // Safe against an older remote engine that does not emit the key:
            // supportsThinkingBudget stays undefined and the capability gate
            // below returns without sending anything.
            if (resolvedThinkingBudget == null || obj.enable_thinking === false) {
              return;
            }
            // Engine-level reasoning-phase cap: the server honors a top-level
            // max_thinking_tokens ONLY for the families whose answer-pass machinery
            // caps the thinking pass at the budget (registry supportsThinkingBudget),
            // or when the chat TEMPLATE itself declares a budget marker
            // (thinkingBudgetSupported). Families with a reasoning parser but NO
            // engine-side thinking-budget behavior (e.g. deepseek-v4, step-3.7-flash)
            // ignore the field — dsv4 uses reasoning_effort instead — so sending it
            // would be untruthful. Gate the whole block on either capability.
            if (!(supportsThinkingBudget === true || thinkingBudgetSupported === true)) {
              return;
            }
            obj.max_thinking_tokens = resolvedThinkingBudget;
            if (thinkingBudgetSupported !== false) {
              obj.chat_template_kwargs = {
                ...(obj.chat_template_kwargs || {}),
                thinking_budget: resolvedThinkingBudget,
              };
            }
          };
          const applyPostToolAnswerPolicy = (obj: Record<string, any>) => {
            applyPostToolRequestFields(obj, {
              finalAnswerRecovery,
              plannedDirectAnswerPass,
              isRemote,
              previous: previousToolRequestFields,
            });
            // A normal exact-final follow-up is still part of the user's
            // requested reasoning mode. It must not silently turn an explicit
            // On (or model-owned Auto) into Thinking Off. Only the bounded
            // recovery after an actually empty/incomplete post-tool pass may
            // remove schemas and request instruct mode.
            if (!finalAnswerRecovery) return;
            if (isRemote) return;
            // Some native templates (currently Step-3.7) have no truthful
            // thinking-off rail. Keep their bounded follow-up answer-only by
            // removing tools, but do not coerce an empty <think> sentinel.
            if (supportsInstructMode === false) return;
            obj.enable_thinking = false;
            obj.thinking_mode = "instruct";
            obj.chat_template_kwargs = {
              ...(obj.chat_template_kwargs || {}),
              enable_thinking: false,
            };
            delete obj.reasoning_effort;
            delete obj.max_thinking_tokens;
          };
          const finalizeRequestBody = (obj: Record<string, any>) => {
            applyPostToolAnswerPolicy(obj);
            previousToolRequestFields = captureToolRequestFields(obj);
            return obj;
          };
          if (useResponsesApi) {
            const { systemMessages, inputMessages } =
              splitResponsesSystemMessages(
                requestMessages,
                chatDetectedFamily === "deepseek-v4",
              );
            const instructions =
              overrides?.builtinToolsEnabled && systemMessages.length > 0
                ? systemMessages.map((m: any) => m.content).join("\n")
                : overrides?.systemPrompt ||
              (systemMessages.length > 0
                ? systemMessages.map((m: any) => m.content).join("\n")
                : undefined);
            const obj: Record<string, any> = {
              model: modelName,
              input: inputMessages,
              instructions,
              // Native MTP contributes the product-enforced greedy tuple;
              // otherwise only explicit chat overrides are sent and the
              // server remains authoritative for bundle defaults.
              ...(effectiveSamplerOverrides?.temperature != null
                ? { temperature: effectiveSamplerOverrides.temperature }
                : {}),
              ...(effectiveSamplerOverrides?.topP != null
                ? { top_p: effectiveSamplerOverrides.topP }
                : {}),
              ...(resolvedOutputBudget
                ? { max_output_tokens: resolvedOutputBudget }
                : {}),
              stream: true,
            };
            if (stopSequences) obj.stop = stopSequences;
            const effectiveTopK = effectiveSamplerOverrides?.topK;
            // Zero is an explicit disable override. Omitting it would make the
            // engine re-inherit a non-zero bundle/default top_k.
            if (effectiveTopK != null) obj.top_k = effectiveTopK;
            // Explicit zero disables a non-zero bundle min_p default. Only an
            // absent override means inherit the model/server default.
            if (effectiveSamplerOverrides?.minP != null)
              obj.min_p = effectiveSamplerOverrides.minP;
            // Always send when explicitly set; 1.0 can be an intentional
            // per-chat override of a bundle repetition penalty.
            if (overrides?.repeatPenalty != null)
              obj.repetition_penalty = overrides.repeatPenalty;
            if (overrides?.frequencyPenalty != null)
              obj.frequency_penalty = overrides.frequencyPenalty;
            if (overrides?.presencePenalty != null)
              obj.presence_penalty = overrides.presencePenalty;
            // An explicit current-turn no-tool directive is stronger than the
            // chat's persistent built-in-tools toggle. Omitting the schemas is
            // API-equivalent to tool_choice="none", avoids paying for an
            // unusable catalog, and keeps the rendered prefix stable across
            // Responses history replay (some native templates otherwise add a
            // fallback schema block only on the follow-up turn).
            if (attachBuiltinToolsForCurrentTurn) {
              obj.tools = availableToolDefinitions().map((t) => ({
                type: "function",
                name: t.function.name,
                description: t.function.description,
                parameters: t.function.parameters,
              }));
            }
            const requestToolChoice = currentTurnToolChoice();
            if (requestToolChoice !== undefined) {
              obj.tool_choice = requestToolChoice;
            }
            // Only explicit On/Off is serialized. Auto stays omitted so the
            // concrete bundle/family owns its native default (for example,
            // Laguna S2.1's stamped default-on, MiniMax M3 adaptive, and DSV4's
            // reasoning.default_mode). Parser seeding is resolved by the engine
            // from that same effective policy; the presence of a parser alone is
            // not permission for the panel to force thinking on.
            applyReasoningRequestFields(obj, {
              enableThinking: effectiveEnableThinkingOverride,
              reasoningEffort: overrides?.reasoningEffort,
              isRemote,
              sessionHasReasoningParser,
              detectedFamily: chatDetectedFamily,
              supportedReasoningEfforts,
            });
            applyThinkingBudget(obj);
            // VLM video sampling — forward to engine only when session
            // config has non-default values. Remote OpenAI-compatible
            // providers don't support these fields, so skip there.
            if (!isRemote && sessionImageTokenBudget !== undefined)
              obj.image_token_budget = sessionImageTokenBudget;
            if (!isRemote && sessionVideoFps !== undefined)
              obj.video_fps = sessionVideoFps;
            if (!isRemote && sessionVideoMaxFrames !== undefined)
              obj.video_max_frames = sessionVideoMaxFrames;
            if (!isRemote && sessionVideoMaxPixels !== undefined)
              obj.video_max_pixels = sessionVideoMaxPixels;
            if (!isRemote && sessionVideoTokenBudget !== undefined)
              obj.video_token_budget = sessionVideoTokenBudget;
            // Do not serialize the session timeout as the API's explicit
            // per-request `timeout`. The engine deliberately defines an
            // explicit request timeout as a hard wall-clock budget, while its
            // server default is progress-aware. The app owns the saved value
            // as the inactivity watchdog armed above.
            return finalizeRequestBody(obj);
          } else {
            const obj: Record<string, any> = {
              model: modelName,
              messages: requestMessages,
              // Same effective sampler contract as the Responses branch.
              ...(effectiveSamplerOverrides?.temperature != null
                ? { temperature: effectiveSamplerOverrides.temperature }
                : {}),
              ...(effectiveSamplerOverrides?.topP != null
                ? { top_p: effectiveSamplerOverrides.topP }
                : {}),
              ...(resolvedOutputBudget ? { max_tokens: resolvedOutputBudget } : {}),
              stream: true,
              stream_options: { include_usage: true },
            };
            if (stopSequences) obj.stop = stopSequences;
            const effectiveTopK = effectiveSamplerOverrides?.topK;
            // Preserve explicit Off across both wire APIs; only undefined means
            // inherit the current model/server default.
            if (effectiveTopK != null) obj.top_k = effectiveTopK;
            if (effectiveSamplerOverrides?.minP != null)
              obj.min_p = effectiveSamplerOverrides.minP;
            // Always send when explicitly set; 1.0 can be an intentional
            // per-chat override of a bundle repetition penalty.
            if (overrides?.repeatPenalty != null)
              obj.repetition_penalty = overrides.repeatPenalty;
            if (overrides?.frequencyPenalty != null)
              obj.frequency_penalty = overrides.frequencyPenalty;
            if (overrides?.presencePenalty != null)
              obj.presence_penalty = overrides.presencePenalty;
            if (attachBuiltinToolsForCurrentTurn) {
              // Chat Completions API: tools must be in OpenAI format with "function" wrapper
              // e.g. {"type": "function", "function": {"name": ..., "parameters": ...}}
              obj.tools = availableToolDefinitions();
            }
            const requestToolChoice = currentTurnToolChoice();
            if (requestToolChoice !== undefined) {
              obj.tool_choice = requestToolChoice;
            }
            // Only explicit On/Off is serialized. Auto stays omitted so the
            // provider or local engine can apply the model's native policy.
            // STRICT ENV: Filter out enable_thinking for strict generic 3rd-party API hosts that throw 400 Bad Request.
            const isStrictApi =
              isRemote &&
              apiUrl &&
              (apiUrl.includes("api.openai.com") ||
                apiUrl.includes("api.groq.com") ||
                apiUrl.includes("api.together.xyz") ||
                apiUrl.includes("api.anthropic.com") ||
                apiUrl.includes("openrouter.ai") ||
                apiUrl.includes("api.deepseek.com"));

            applyReasoningRequestFields(obj, {
              enableThinking: effectiveEnableThinkingOverride,
              reasoningEffort: overrides?.reasoningEffort,
              isRemote,
              sessionHasReasoningParser,
              detectedFamily: chatDetectedFamily,
              supportedReasoningEfforts,
              allowRequestControls: !isStrictApi,
            });
            applyThinkingBudget(obj);
            // VLM video sampling — local engine only (strict 3rd-party APIs
            // reject unknown fields, remote OpenAI-compat doesn't support it).
            if (!isRemote && sessionImageTokenBudget !== undefined)
              obj.image_token_budget = sessionImageTokenBudget;
            if (!isRemote && sessionVideoFps !== undefined)
              obj.video_fps = sessionVideoFps;
            if (!isRemote && sessionVideoMaxFrames !== undefined)
              obj.video_max_frames = sessionVideoMaxFrames;
            if (!isRemote && sessionVideoMaxPixels !== undefined)
              obj.video_max_pixels = sessionVideoMaxPixels;
            if (!isRemote && sessionVideoTokenBudget !== undefined)
              obj.video_token_budget = sessionVideoTokenBudget;
            // Keep the local engine on its progress-aware server default. An
            // API caller that explicitly sends `timeout` still gets the
            // documented hard wall-clock behavior.
            return finalizeRequestBody(obj);
          }
        };
        const requestDiagSessionId = chatSession?.id || resolvedSession?.id;
        const logRequestShape = (
          bodyJson: string,
          phase: "initial" | "follow_up",
        ) => {
          if (!requestDiagSessionId) return;
          pushChatSessionLog(
            requestDiagSessionId,
            `[CHAT_DIAG] request_shape=${JSON.stringify({
              phase,
              chatId: chatId.slice(0, 8),
              wireApi,
              isRemote,
              baseUrl,
              chatIsMultimodal,
              historicalMediaReplay:
                chatIsMultimodal || isRemote ? "full" : "text_only",
              detectedFamily: chatDetectedFamily,
              sessionHasReasoningParser,
              plannedDirectAnswerPass,
              finalAnswerRecovery,
              body: summarizeRequestForLog(bodyJson, useResponsesApi),
            })}`,
          );
        };
        const requestBody = JSON.stringify(buildRequestBody());
        logRequestShape(requestBody, "initial");

        fetchStartTime = Date.now(); // Capture just before fetch for accurate TTFT
        // Remote internet providers use Electron's net.fetch for certificates
        // and proxies; loopback model servers use Node streaming for SSE.
        const useNodeStreamingFetch = !isRemote || isLoopbackUrl(apiUrl);
        // `response.usage` is a vMLX-only incremental telemetry extension.
        // Negotiate it out-of-band only with a local engine.  The public
        // Responses request body must not send Chat's non-standard
        // stream_options.include_usage to OpenAI-compatible remote providers.
        const vmlxResponsesUsageHeaders: Record<string, string> =
          useResponsesApi && !isRemote
            ? { "X-vMLX-Stream-Usage": "incremental" }
            : {};
        // Inference begins HERE — arm the inactivity watchdog now, not at
        // entry, so queued load/wake waiting never counts against it.
        armFetchInactivityTimeout();
        const response = useNodeStreamingFetch
          ? await streamingFetch(apiUrl, {
              method: "POST",
              headers: {
                "Content-Type": "application/json",
                ...authHeaders,
                ...vmlxResponsesUsageHeaders,
                ...nextLocalRequestCorrelationHeaders(),
              },
              body: requestBody,
              signal: abortController.signal,
            })
          : await remoteFetch(apiUrl, {
              method: "POST",
              headers: {
                "Content-Type": "application/json",
                ...authHeaders,
                ...vmlxResponsesUsageHeaders,
                ...nextLocalRequestCorrelationHeaders(),
              },
              body: requestBody,
              signal: abortController.signal,
            });

        if (!response.ok) {
          const errorText = await response.text();
          // Try to extract structured error detail from JSON responses
          let errorDetail = errorText;
          try {
            const parsed = JSON.parse(errorText);
            if (parsed.detail) {
              errorDetail =
                typeof parsed.detail === "string"
                  ? parsed.detail
                  : Array.isArray(parsed.detail)
                    ? parsed.detail
                        .map((d: any) => d.msg || JSON.stringify(d))
                        .join("; ")
                    : JSON.stringify(parsed.detail);
            } else if (
              typeof parsed.error?.message === "string" &&
              parsed.error.message
            ) {
              // OpenAI-style envelope — what the engine actually sends for
              // 4xx rejections like prompt_too_long
              errorDetail = parsed.error.message;
            }
          } catch {
            /* use raw text */
          }
          throw new Error(`API error: ${response.status} - ${errorDetail}`);
        }

        // Stream response
        reader = response.body?.getReader();
        if (!reader) throw new Error("Response body is null");

        fullContent = "";
        reasoningContent = "";
        isReasoning = false;
        let currentEventType = ""; // Track SSE event type for Responses API
        // The server deliberately emits the error signal before its terminal
        // usage event (Chat: usage chunk + [DONE], Responses: response.failed).
        // Keep reading long enough to consume authoritative partial usage, then
        // throw through the normal partial-response persistence path.
        let pendingStreamServerError: ChatStreamServerEventError | null = null;
        const seenResponsesApiEvents = new Set<string>();
        const responsesFunctionCallArgsByKey = new Map<
          string,
          { value: string }
        >();
        const responsesFunctionCallItemKey = (
          itemId: unknown,
          outputIndex: unknown,
        ): string => {
          if (typeof itemId === "string" && itemId.length > 0) {
            return `item:${itemId}`;
          }
          if (typeof outputIndex === "number") {
            return `output:${outputIndex}`;
          }
          return "";
        };
        const responsesFunctionCallArgsBuffer = (
          itemId: unknown,
          outputIndex: unknown,
        ): { value: string } | undefined => {
          const itemKey = responsesFunctionCallItemKey(itemId, undefined);
          const outputKey = responsesFunctionCallItemKey(undefined, outputIndex);
          const key = itemKey || outputKey;
          if (!key) return undefined;
          let argsBuffer =
            responsesFunctionCallArgsByKey.get(itemKey) ||
            responsesFunctionCallArgsByKey.get(outputKey);
          if (!argsBuffer) {
            argsBuffer = { value: "" };
          }
          if (itemKey) responsesFunctionCallArgsByKey.set(itemKey, argsBuffer);
          if (outputKey) {
            responsesFunctionCallArgsByKey.set(outputKey, argsBuffer);
          }
          return argsBuffer;
        };

        // Codex 2026-05-06 #2: track if Responses-API stream emitted any
        // text-delta. When the server sends final text via .done events
        // only (no deltas), we fall back to consuming those — otherwise
        // assistant message is blank and history rebuilds skip it,
        // causing wrong-prompt replay on next turn.
        let _sawResponsesTextDelta = false;
        let responsesFinalText = "";

        // Track tool calls received during streaming for MCP auto-execution
        let receivedToolCalls: Array<{
          id: string;
          function: { name: string; arguments: string };
        }> = [];
        // Track finish_reason from server to detect truncation (length), content filter, etc.
        // (declared outside try block so catch can access it for abort recovery)
        // Track tool iteration count (declared here so processLine closure can access it)
        const MAX_TOOL_ITERATIONS = overrides?.maxToolIterations ?? 10;
        let toolIteration = 0;

        // Track the length of content last emitted to renderer (for inline tool call positioning)
        let lastEmittedContentLength = 0;

        // Helper: emit tool call status to renderer (separate from content stream)
        const emitToolStatus = (
          phase: string,
          toolName: string,
          detail?: string,
          iteration?: number,
          toolCallId?: string,
        ) => {
          const contentOffset =
            phase === "calling" ? lastEmittedContentLength : undefined;
          // Collect for persistence — include detail for calling, result, and error phases
          // so tool results are visible after reload (truncate large results to 4KB)
          const persistDetail =
            phase === "calling" || phase === "result" || phase === "error"
              ? detail && detail.length > 4096
                ? detail.slice(0, 4096) + "..."
                : detail
              : undefined;
          collectedToolStatuses.push({
            phase,
            toolName,
            toolCallId,
            iteration,
            contentOffset,
            detail: persistDetail,
          });
          toolStatusNeedsFlush = true;
          try {
            const win = getWindow();
            if (win && !win.isDestroyed()) {
              win.webContents.send("chat:toolStatus", {
                chatId,
                messageId: assistantMessage.id,
                phase,
                toolName,
                toolCallId,
                detail,
                iteration,
                contentOffset,
              });
            }
          } catch (_) {}
        };
        const flushToolStatusToRenderer = async () => {
          if (!toolStatusNeedsFlush) return;
          toolStatusNeedsFlush = false;
          await new Promise<void>((resolve) => setImmediate(resolve));
        };
        const flushStreamDeltaToRenderer = async () => {
          if (!rendererStreamNeedsFlush) return;
          rendererStreamNeedsFlush = false;
          await new Promise<void>((resolve) => setImmediate(resolve));
        };

        const beginAnswerPass = () => {
          // The bounded answer pass is a second model generation. Its prompt
          // prefill is real end-to-end latency, but it is not decode time. Keep
          // totalTime honest while starting a fresh physical decode window so
          // the UI does not report a healthy ~50 t/s model as ~3 t/s during
          // the silent second prefill.
          lastTokenTime = null;
          tpsSnapshots.length = 0;
          tpsTokenBase = tokenCount;
          liveTps = 0;
          try {
            const win = getWindow();
            if (win && !win.isDestroyed()) {
              win.webContents.send("chat:answerPass", {
                chatId,
                messageId: assistantMessage.id,
              });
            }
          } catch (_) {}
        };

        // Client-side tool call buffering: suppress content when leaked tool call XML detected.
        // Must check RAW content before template token stripping, since markers like
        // <minimax:tool_call> get stripped by TEMPLATE_TOKEN_REGEX and never reach fullContent.
        let clientToolCallBuffering = false;
        // Set when the Responses reconciler rejects an all-markup terminal
        // payload. Never rendered; drives the never-empty notice at finalize.
        let rejectedControlMarkupText = "";
        let rawAccumulated = ""; // Tracks unstripped content for tool call detection
        // Client-side <think> tag extraction: tracks whether we're inside a <think> block
        // when the server doesn't provide reasoning_content (fallback for all parser types)
        let clientSideThinkParsing = false;
        let clientSideThinkHoldback = "";
        // Trailing characters withheld from the renderer because they could
        // still grow into a tool tag (models tokenize "</parameter>" as several
        // tokens). Flushed at every iteration boundary and before finalize, so
        // ordinary prose that merely ends in "<" is never lost.
        let toolTagHoldback = "";

        const normalizeThinkingMarkers = (content: string) =>
          content
            .replace(/<mm:think>/g, "<think>")
            .replace(/<\/mm:think>/g, "</think>")
            .replace(/\[THINK\]/g, "<think>")
            .replace(/\[\/THINK\]/g, "</think>");

        const thinkMarkerHoldbackLength = (content: string) => {
          const markers = ["<think>", "</think>"];
          let hold = 0;
          for (const marker of markers) {
            const max = Math.min(marker.length - 1, content.length);
            for (let len = 1; len <= max; len++) {
              if (content.endsWith(marker.slice(0, len))) {
                hold = Math.max(hold, len);
              }
            }
          }
          return hold;
        };

        // Helper: emit streaming delta to renderer
        // skipClientCount: when true, skip client-side token counting/TPS (used when
        // a single SSE chunk is split into multiple emitDelta calls by think-tag extraction,
        // so we only count once per SSE chunk, not once per emitDelta call)
        const emitDelta = (
          delta: string,
          isReasoningDelta: boolean,
          skipClientCount = false,
          bypassToolMarkerDetection = false,
        ) => {
          // Skip emission if abort already fired — prevents stale tokens from reaching renderer
          if (abortController.signal.aborted) return;
          // Track raw content BEFORE stripping for tool call marker detection
          let suppressVisibleToolDelta = false;
          if (!isReasoningDelta && !bypassToolMarkerDetection) {
            rawAccumulated += delta;
            // Only activate buffering when tool call markers appear at the start of a line,
            // not when the model is explaining tool syntax in prose (e.g., "I'll use <tool_call>...")
            if (!clientToolCallBuffering) {
              // Catch real tool call formats and common hallucinated tool-call tags.
              // Use trailing window (last 200 chars) to avoid O(n) regex on full response
              const lineStartPattern = TOOL_CALL_MARKER_LINE_START;
              const searchWindow =
                rawAccumulated.length > 200
                  ? rawAccumulated.slice(-200)
                  : rawAccumulated;
              if (lineStartPattern.test(searchWindow)) {
                clientToolCallBuffering = true;
                console.log(`[CHAT] Client-side tool call buffering activated`);
                emitToolStatus(
                  "generating",
                  "",
                  "Generating tool call...",
                  toolIteration,
                );
              }
            }
            suppressVisibleToolDelta = clientToolCallBuffering;
          }

          // Strip any leaked chat template tokens from the delta
          delta = delta.replace(TEMPLATE_TOKEN_REGEX, "");
          if (!delta) return;
          // Strip Harmony protocol residue — only for GLM/GPT-OSS models that use the
          // Harmony <|start|><|channel|><|message|> protocol. Without this guard, these
          // regexes would strip legitimate prose like "assistant analysis" from all models.
          if (isHarmonyModel) {
            delta = delta.replace(/<\/?(?:assistant|analysis|final)+/gi, "");
            delta = delta.replace(
              /(?:assistant\s*){1,3}(?:analysis|final)/gi,
              "",
            );
            delta = delta.replace(
              /(?:analysis|final)\s*(?:assistant\s*){1,3}/gi,
              "",
            );
            if (!delta) return;
          }
          // Strip U+FFFD replacement characters
          delta = delta.replace(/\uFFFD/g, "");
          if (!delta) return;

          // Complete orphan tool tags must never reach the renderer, or the
          // user watches </parameter> stream into the answer and the live view
          // disagrees with what gets persisted. Withhold any trailing fragment
          // that could still become a tag; bypassed emissions are our own
          // notices and are already clean.
          if (!isReasoningDelta && !bypassToolMarkerDetection) {
            const merged = toolTagHoldback + delta;
            const hold = toolMarkupHoldbackLength(merged);
            toolTagHoldback = hold > 0 ? merged.slice(merged.length - hold) : "";
            const emittable = hold > 0
              ? merged.slice(0, merged.length - hold)
              : merged;
            delta = stripStreamingToolTags(emittable);
            if (!delta) return;
          }

          // === State updates (always, no throttle) ===
          const now = Date.now();
          if (firstTokenTime === null) {
            firstTokenTime = now;
            startPeriodicSave();
          }
          // Track generation time between consecutive streamed deltas. Tool
          // execution and follow-up prefill are excluded by resetting
          // lastTokenTime at each HTTP stream boundary. Do not discard long
          // intra-stream gaps: buffered answer passes and slow models still
          // spent that time generating, and dropping it inflates final TPS.
          if (lastTokenTime !== null) {
            const gap = now - lastTokenTime;
            if (gap > 0) generationMs += gap;
          }
          lastTokenTime = now;

          if (isReasoningDelta) {
            isReasoning = true;
            reasoningSegments = appendReasoningDelta(reasoningSegments, delta);
            const visibleSegments = currentReasoningSegments();
            reasoningContent =
              visibleSegments.length > 0
                ? visibleSegments[visibleSegments.length - 1]
                : "";
          } else {
            if (isReasoning) {
              isReasoning = false;
              try {
                const win = getWindow();
                if (win && !win.isDestroyed()) {
                  win.webContents.send("chat:reasoningDone", {
                    chatId,
                    messageId: assistantMessage.id,
                    reasoningContent: currentReasoningContent(),
                    reasoningSegments: currentReasoningSegments(),
                  });
                }
              } catch (_) {}
            }
            if (!suppressVisibleToolDelta) {
              fullContent += delta;
              // Update content offset immediately (not throttled) for accurate tool call positioning
              lastEmittedContentLength = allGeneratedContent
                ? allGeneratedContent.length + 2 + fullContent.length
                : fullContent.length;
            }
          }
          // Client-side counting (fallback when server doesn't send usage in each chunk).
          // Must happen BEFORE TPS snapshot so the rolling window uses accurate counts.
          // skipClientCount prevents inflation when think-tag splitting calls emitDelta
          // multiple times for a single SSE chunk.
          if (!serverSendsUsage && !skipClientCount) {
            tokenCount++;
            iterationTokenCount++;
          }

          // Rolling TPS: snapshot (timestamp, relative tokenCount) for accurate throughput.
          // Uses tpsTokenBase-relative count to avoid negative deltas at iteration boundaries
          // (server restarts completion_tokens from 0 on each new HTTP request).
          tpsSnapshots.push([now, tokenCount - tpsTokenBase]);
          if (tpsSnapshots.length > TPS_BUFFER_SIZE) tpsSnapshots.shift();
          if (tpsSnapshots.length >= 2) {
            const [oldT, oldN] = tpsSnapshots[0];
            const [newT, newN] = tpsSnapshots[tpsSnapshots.length - 1];
            const span = (newT - oldT) / 1000;
            const tpsDelta = newN - oldN;
            liveTps =
              span > 0.01 && tpsDelta > 0
                ? tpsDelta / span
                : tpsDelta <= 0
                  ? 0
                  : liveTps;
            if (Number.isFinite(liveTps) && liveTps > 0) {
              liveTpsHistory.push(liveTps);
              if (liveTpsHistory.length > MAX_LIVE_TPS_SAMPLES) {
                liveTpsHistory.shift();
              }
            }
          }

          // Suppress rendering (but not counting/TPS) when tool call content is detected
          if (!isReasoningDelta && suppressVisibleToolDelta) return;

          // === IPC emission — every token emitted immediately ===
          // Renderer-side useTypewriter handles smooth character reveal via rAF.

          // Live generation TPS from rolling window (real-time speed of incoming tokens).
          // Cumulative TPS (tokenCount / generationMs) is used for final saved metrics only.
          const streamTps = liveTps;
          // Cumulative generation time for elapsed display
          const genSec = generationMs / 1000;
          const wallSec = (now - (firstTokenTime || fetchStartTime)) / 1000;
          const elapsed = genSec > 0.05 ? genSec : wallSec;
          // TTFT measured from fetchStartTime (excludes health check and message building overhead)
          const ttft = Math.max(
            0,
            firstTokenTime ? (firstTokenTime - fetchStartTime) / 1000 : 0,
          );
          const ppSpeed = calculatePrefillTps({
            promptTokens,
            cachedTokens,
            ttftSeconds: ttft,
            serverUsageKnown: serverSendsUsage,
          });

          try {
            const win = getWindow();
            if (win && !win.isDestroyed()) {
              // Include pre-tool content so UI doesn't lose earlier text when fullContent resets
              const displayContent =
                !isReasoningDelta && allGeneratedContent
                  ? allGeneratedContent + "\n\n" + fullContent
                  : isReasoningDelta
                    ? currentReasoningContent()
                    : fullContent;
              win.webContents.send("chat:stream", {
                chatId,
                messageId: assistantMessage.id,
                fullContent: displayContent,
                isReasoning: isReasoningDelta,
                reasoningSegments: isReasoningDelta
                  ? currentReasoningSegments()
                  : undefined,
                metrics: {
                  tokenCount: cumulativeTokenOffset + iterationTokenCount,
                  promptTokens,
                  cachedTokens,
                  cacheDetail,
                  tokensPerSecond: streamTps.toFixed(1),
                  ppSpeed,
                  ttft: ttft.toFixed(2),
                  elapsed: elapsed.toFixed(1),
                },
              });
              rendererStreamNeedsFlush = true;
            }
          } catch (_) {}
        };

        // Process a single SSE data line (with event type context)
        const processLine = (trimmed: string) => {
          // Standards-safe vMLX phase signal. The server emits this as an SSE
          // comment, so generic OpenAI clients ignore it. The local panel uses
          // it to distinguish answer synthesis from a stalled stream.
          if (trimmed === ": vmlx-answer-pass-start") {
            beginAnswerPass();
            return;
          }
          // Track SSE event type (Responses API uses "event:" lines)
          if (trimmed.startsWith("event: ")) {
            currentEventType = trimmed.slice(7);
            return;
          }
          if (!trimmed) {
            currentEventType = "";
            return;
          } // Blank line = SSE event boundary, reset type
          if (!trimmed.startsWith("data: ")) return;
          const data = trimmed.slice(6);
          if (data === "[DONE]") {
            currentEventType = "";
            return;
          }

          try {
            const parsed = JSON.parse(data);
            const responsesEventType =
              typeof parsed.type === "string" ? parsed.type : currentEventType;
            const isResponsesTerminalEvent =
              responsesEventType === "response.completed" ||
              responsesEventType === "response.incomplete" ||
              responsesEventType === "response.failed";

            if (useResponsesApi && responsesEventType) {
              const seq = parsed.sequence_number;
              if (typeof seq === "number") {
                const key = `${responsesEventType}:${seq}`;
                if (seenResponsesApiEvents.has(key)) return;
                seenResponsesApiEvents.add(key);
              }
            }

            if (useResponsesApi) {
              // ── Responses API SSE parsing ──
              // Track response ID from response.created event
              // Server wraps in { response: { id: "resp_..." } }
              const respId = parsed.response?.id || parsed.id;
              if (responsesEventType === "response.created" && respId) {
                const entry = activeRequests.get(chatId);
                if (entry) {
                  // A single visible UI turn can issue more than one HTTP
                  // response (for example, after a tool result). Always retain
                  // the newest response identity so cancellation and terminal
                  // telemetry bind to the request whose final metrics are
                  // displayed.
                  entry.responseId = respId;
                  entry.endpoint = { host: resolved.host, port: resolved.port };
                }
              }

              if (
                responsesEventType === "response.heartbeat" &&
                parsed.tool_call_generating
              ) {
                if (!clientToolCallBuffering) {
                  clientToolCallBuffering = true;
                  emitToolStatus(
                    "generating",
                    "",
                    "Generating tool call...",
                    toolIteration,
                  );
                }
              }

              // Reasoning delta from OpenAI Responses reasoning-summary events.
              // Keep the legacy vMLX event accepted so older installed engines
              // still display reasoning in the panel.
              if (
                (responsesEventType === "response.reasoning_summary_text.delta" ||
                  responsesEventType === "response.reasoning.delta") &&
                parsed.delta
              ) {
                emitDelta(parsed.delta, true);
              }

              // Reasoning done — triggers reasoningDone event in emitDelta (isReasoning=true→false transition)
              if (
                responsesEventType === "response.reasoning_summary_text.done" ||
                responsesEventType === "response.reasoning.done"
              ) {
                reasoningSegments = reconcileReasoningSummaryDone(
                  reasoningSegments,
                  parsed.text,
                );
                const visibleSegments = currentReasoningSegments();
                reasoningContent =
                  visibleSegments.length > 0
                    ? visibleSegments[visibleSegments.length - 1]
                    : "";
                // Force the reasoning→content transition so reasoningDone fires
                if (isReasoning) {
                  isReasoning = false;
                  try {
                    const win = getWindow();
                    if (win && !win.isDestroyed()) {
                      win.webContents.send("chat:reasoningDone", {
                        chatId,
                        messageId: assistantMessage.id,
                        reasoningContent: currentReasoningContent(),
                        reasoningSegments: currentReasoningSegments(),
                      });
                    }
                  } catch (_) {}
                }
              }

              // Delta text from response.output_text.delta
              // Server sends { delta: "text" }, not { text: "..." }
              if (
                responsesEventType === "response.output_text.delta" &&
                (parsed.delta || parsed.text)
              ) {
                emitDelta(parsed.delta || parsed.text, false);
                _sawResponsesTextDelta = true;
              }

              // Codex 2026-05-06 #2: SSE final-text fallback. If a stream
              // never emitted any response.output_text.delta events but
              // the server sends final text via response.output_text.done
              // or response.content_part.done or a terminal Responses event
              // wrapping output[*].content[*].text, we MUST consume that
              // final text — otherwise the assistant message is blank,
              // gets skipped on next-turn rebuild (empty skip in
              // buildMessagesForApi), consecutive user messages merge
              // → new user prompt becomes invisible to the model and
              // it replays the LAST coherent user turn (e.g. "testing").
              if (
                responsesEventType === "response.output_text.done" &&
                typeof parsed.text === "string"
              ) {
                responsesFinalText = parsed.text;
                if (parsed.text.length > 0 && !_sawResponsesTextDelta) {
                  emitDelta(parsed.text, false);
                  _sawResponsesTextDelta = true;
                }
              }
              if (
                responsesEventType === "response.content_part.done" &&
                parsed.part?.type === "output_text" &&
                typeof parsed.part?.text === "string"
              ) {
                responsesFinalText = parsed.part.text;
                if (parsed.part.text.length > 0 && !_sawResponsesTextDelta) {
                  emitDelta(parsed.part.text, false);
                  _sawResponsesTextDelta = true;
                }
              }
              if (
                isResponsesTerminalEvent && !_sawResponsesTextDelta
              ) {
                // Walk parsed.response.output[*].content[*].text and
                // emit any output_text we find.
                const outputs = parsed.response?.output || [];
                const completedTextParts: string[] = [];
                for (const item of outputs) {
                  if (item?.type !== "message") continue;
                  for (const part of item?.content || []) {
                    if (
                      part?.type === "output_text" &&
                      typeof part.text === "string" &&
                      part.text.length > 0
                    ) {
                      completedTextParts.push(part.text);
                      emitDelta(part.text, false);
                      _sawResponsesTextDelta = true;
                    }
                  }
                }
                if (completedTextParts.length > 0) {
                  responsesFinalText = completedTextParts.join("");
                }
              }

              // Handle function_call items (tool calls) from Responses API
              // response.output_item.done carries the complete tool call: { item: { type, call_id, name, arguments } }
              if (
                responsesEventType === "response.function_call_arguments.delta" &&
                typeof parsed.delta === "string"
              ) {
                const argsBuffer = responsesFunctionCallArgsBuffer(
                  parsed.item_id,
                  parsed.output_index,
                );
                if (argsBuffer) argsBuffer.value += parsed.delta;
              }
              if (
                responsesEventType === "response.function_call_arguments.done" &&
                typeof parsed.arguments === "string"
              ) {
                const argsBuffer = responsesFunctionCallArgsBuffer(
                  parsed.item_id,
                  parsed.output_index,
                );
                if (argsBuffer) argsBuffer.value = parsed.arguments;
              }
              if (
                responsesEventType === "response.output_item.done" &&
                parsed.item?.type === "function_call"
              ) {
                const item = parsed.item;
                const argsBuffer = responsesFunctionCallArgsBuffer(
                  item.id,
                  parsed.output_index,
                );
                const finalArguments = item.arguments || argsBuffer?.value || "{}";
                const toolCallId =
                  item.call_id ||
                  `call_${uuidv4().replace(/-/g, "").slice(0, 16)}`;
                receivedToolCalls.push({
                  id: toolCallId,
                  function: {
                    name: item.name,
                    arguments: finalArguments,
                  },
                });
              }

              // Real-time usage from the explicitly negotiated local vMLX
              // response.usage extension (never part of a standard remote stream).
              if (responsesEventType === "response.usage" && parsed.usage) {
                recordServerDecodeUsage(parsed.usage);
                if (parsed.usage.output_tokens != null) {
                  tokenCount = parsed.usage.output_tokens;
                  // Detect server token count restart (new HTTP request resets completion_tokens to 0)
                  if (tokenCount < iterationTokenBase) iterationTokenBase = 0;
                  iterationTokenCount = Math.max(
                    0,
                    tokenCount - iterationTokenBase,
                  );
                  // Clear contaminated client-counted entries when transitioning to server usage
                  if (!serverSendsUsage) {
                    tpsSnapshots.length = 0;
                    tpsTokenBase = tokenCount;
                  }
                  serverSendsUsage = true;
                }
                if (parsed.usage.input_tokens != null)
                  promptTokens = parsed.usage.input_tokens;
                recordCacheUsage(parsed.usage.input_tokens_details);
              }

              // Defer Responses failures until the response.failed terminal has
              // supplied partial usage. Throwing on the preceding error event
              // cancels the reader and loses the authoritative token counts.
              const responsesErrorDetail = chatStreamServerEventErrorDetail(
                parsed,
                responsesEventType,
              );
              if (responsesErrorDetail) {
                const errDetail = responsesErrorDetail;
                if (isExpectedChatBackendDisconnectError(errDetail)) {
                  throw expectedChatBackendDisconnectError();
                }
                console.error(`[CHAT] Responses API error event: ${errDetail}`);
                pendingStreamServerError = new ChatStreamServerEventError(
                  `Server error: ${errDetail}`,
                );
              }

              if (responsesEventType === "response.warning") {
                const eventWarnings = extractResponsesWarnings(parsed);
                if (eventWarnings) {
                  responseWarnings = Array.from(
                    new Set([...(responseWarnings || []), ...eventWarnings]),
                  );
                }
              }

              // Final usage from response.completed / response.incomplete.
              // Server wraps in { response: { usage: { input_tokens, output_tokens } } }
              const respUsage = parsed.response?.usage || parsed.usage;
              if (isResponsesTerminalEvent) {
                const completedWarnings = extractResponsesWarnings(
                  parsed.response || parsed,
                );
                if (completedWarnings) {
                  responseWarnings = Array.from(
                    new Set([...(responseWarnings || []), ...completedWarnings]),
                  );
                }
                // Structured engine notices (#175): effort substitution rides
                // the terminal snapshot; context exhaustion rides
                // incomplete_details on a length terminal. Rendered through
                // the same persisted warnings rail.
                const structuredNotices = [
                  effortSubstitutionNotice(parsed.response?.effort_substitution),
                  contextExhaustionNotice(
                    parsed.response?.incomplete_details?.context_exhaustion,
                  ),
                ].filter((n): n is string => !!n);
                if (structuredNotices.length > 0) {
                  responseWarnings = Array.from(
                    new Set([...(responseWarnings || []), ...structuredNotices]),
                  );
                }
                // Track status for truncation detection
                const respStatus = parsed.response?.status;
                lastFinishReason =
                  responsesTerminalFinishReason(
                    respStatus,
                    parsed.response?.incomplete_details,
                  ) ?? lastFinishReason;
              }
              if (isResponsesTerminalEvent && respUsage) {
                recordServerDecodeUsage(respUsage);
                if (respUsage.output_tokens != null) {
                  tokenCount = respUsage.output_tokens;
                  if (tokenCount < iterationTokenBase) iterationTokenBase = 0;
                  iterationTokenCount = Math.max(
                    0,
                    tokenCount - iterationTokenBase,
                  );
                  if (!serverSendsUsage) {
                    tpsSnapshots.length = 0;
                    tpsTokenBase = tokenCount;
                  }
                  serverSendsUsage = true;
                }
                if (respUsage.input_tokens != null)
                  promptTokens = respUsage.input_tokens;
                recordCacheUsage(respUsage.input_tokens_details);
              }
            } else {
              // ── Chat Completions SSE parsing ──
              const chatWarnings = extractResponsesWarnings(parsed);
              if (chatWarnings) {
                responseWarnings = Array.from(
                  new Set([...(responseWarnings || []), ...chatWarnings]),
                );
              }
              const choice = parsed.choices?.[0]?.delta;

              // Track response ID for server-side cancel
              if (parsed.id) {
                const entry = activeRequests.get(chatId);
                if (entry) {
                  // Tool/result continuation requests receive a new response
                  // ID. Keeping only the first ID makes Stop target a completed
                  // request and misattributes final cache/usage telemetry.
                  entry.responseId = parsed.id;
                  entry.endpoint = { host: resolved.host, port: resolved.port };
                }
              }

              // Update usage BEFORE emitting delta so metrics use real server counts
              if (parsed.usage) {
                if (parsed.usage.completion_tokens != null) {
                  tokenCount = parsed.usage.completion_tokens;
                  // Detect server token count restart (new HTTP request resets completion_tokens to 0)
                  if (tokenCount < iterationTokenBase) iterationTokenBase = 0;
                  iterationTokenCount = Math.max(
                    0,
                    tokenCount - iterationTokenBase,
                  );
                  // Clear contaminated client-counted entries when transitioning to server usage
                  if (!serverSendsUsage) {
                    tpsSnapshots.length = 0;
                    tpsTokenBase = tokenCount;
                  }
                  serverSendsUsage = true;
                }
                if (parsed.usage.prompt_tokens != null)
                  promptTokens = parsed.usage.prompt_tokens;
                recordCacheUsage(parsed.usage.prompt_tokens_details);
              }

              // Structured engine notices (#175): the chat-dialect terminal
              // chunk carries effort_substitution / context_exhaustion
              // top-level (popped at _dump_chat_chunk terminal_usage).
              const chatStructuredNotices = [
                effortSubstitutionNotice(parsed.effort_substitution),
                contextExhaustionNotice(parsed.context_exhaustion),
              ].filter((n): n is string => !!n);
              if (chatStructuredNotices.length > 0) {
                responseWarnings = Array.from(
                  new Set([...(responseWarnings || []), ...chatStructuredNotices]),
                );
              }

              // Track finish_reason (length = truncated, content_filter = filtered)
              const finishReason = parsed.choices?.[0]?.finish_reason;
              if (finishReason) lastFinishReason = finishReason;

              // Chat Completions sends partial usage after its error chunk.
              // Preserve the failure now and throw only after the stream closes.
              const chatErrorDetail = chatStreamServerEventErrorDetail(parsed);
              if (chatErrorDetail) {
                const errDetail = chatErrorDetail;
                if (isExpectedChatBackendDisconnectError(errDetail)) {
                  throw expectedChatBackendDisconnectError();
                }
                console.error(
                  `[CHAT] Chat completions error chunk: ${errDetail}`,
                );
                pendingStreamServerError = new ChatStreamServerEventError(
                  `Server error: ${errDetail}`,
                );
              }

              // Handle reasoning_content from reasoning parser
              const reasoning = choice?.reasoning_content || choice?.reasoning;
              if (reasoning) {
                emitDelta(reasoning, true);
              }

              // Suppressed-reasoning heartbeat: server emits a chunk with
              // usage but no content and no reasoning when the model's
              // template ignored enable_thinking=false and is generating
              // reasoning tokens internally (e.g., MiniMax M2.5). Without
              // this branch the UI sees no stream updates for many seconds
              // and appears hung. Update the stale message metrics so the
              // user at least sees a live token counter while reasoning
              // finishes internally.
              const _hasContent = !!choice?.content;
              const _hasFinish = !!finishReason;
              if (
                !_hasContent &&
                !reasoning &&
                !_hasFinish &&
                parsed.usage &&
                fullContent === "" &&
                reasoningContent === ""
              ) {
                try {
                  const win = getWindow();
                  if (win && !win.isDestroyed()) {
                    const now = Date.now();
                    if (firstTokenTime === null) firstTokenTime = now;
                    const ttft = Math.max(
                      0,
                      firstTokenTime
                        ? (firstTokenTime - fetchStartTime) / 1000
                        : 0,
                    );
                    // 2026-05-02: derive tps from server's usage instead of
                    // hard-coded "0.0". Suppressed-reasoning heartbeats fire
                    // throughout the internal thinking phase on models like
                    // MiniMax M2.5; "0.0 t/s" was misleading users to think
                    // generation had stalled. server's usage.completion_tokens
                    // increments correctly per the SSE stream — use that.
                    const _hbToks =
                      parsed.usage.completion_tokens ||
                      parsed.usage.output_tokens ||
                      0;
                    const _hbElapsed =
                      firstTokenTime !== null
                        ? (now - firstTokenTime) / 1000
                        : 0;
                    const _hbTps =
                      _hbElapsed > 0.1 && _hbToks > 0
                        ? (_hbToks / _hbElapsed).toFixed(1)
                        : "0.0";
                    win.webContents.send("chat:stream", {
                      chatId,
                      messageId: assistantMessage.id,
                      fullContent: "",
                      isReasoning: false,
                      metrics: {
                        tokenCount: _hbToks || tokenCount,
                        promptTokens,
                        cachedTokens,
                        cacheDetail,
                        tokensPerSecond: _hbTps,
                        ttft: ttft.toFixed(2),
                        elapsed: ((now - fetchStartTime) / 1000).toFixed(1),
                      },
                    });
                  }
                } catch (_) {}
              }

              if (choice?.content) {
                // Client-side fallback: if server didn't provide reasoning_content
                // but content contains <think> tags, extract them client-side.
                // This handles servers without a reasoning parser, remote endpoints,
                // and older server versions.
                // chunkCounted tracks whether we've already counted this SSE chunk's token
                // to prevent inflation from think-tag splitting into multiple emitDelta calls.
                if (!reasoning) {
                  let content =
                    clientSideThinkHoldback + (choice.content as string);
                  clientSideThinkHoldback = "";
                  let chunkCounted = !!reasoning; // if reasoning was emitted above, counting already happened
                  const emitWithCount = (text: string, isR: boolean) => {
                    emitDelta(text, isR, chunkCounted);
                    chunkCounted = true; // subsequent calls skip counting
                  };
                  // Normalize [THINK]/[/THINK] (Mistral 4) and
                  // <mm:think>...</mm:think> (MiniMax-M3) to
                  // <think>...</think> for unified fallback parsing when an
                  // older/misconfigured server streams reasoning tags in
                  // content instead of reasoning_content.
                  content = normalizeThinkingMarkers(content);
                  const holdback = thinkMarkerHoldbackLength(content);
                  if (holdback > 0) {
                    clientSideThinkHoldback = content.slice(-holdback);
                    content = content.slice(0, -holdback);
                  }

                  while (content) {
                    if (clientSideThinkParsing) {
                      const endIdx = content.indexOf("</think>");
                      if (endIdx >= 0) {
                        const reasoningPart = content.slice(0, endIdx);
                        content = content.slice(endIdx + 8); // 8 = '</think>'.length
                        clientSideThinkParsing = false;
                        if (reasoningPart) emitWithCount(reasoningPart, true);
                      } else {
                        emitWithCount(content, true);
                        content = "";
                      }
                    } else {
                      const startIdx = content.indexOf("<think>");
                      if (startIdx >= 0) {
                        const preContent = content.slice(0, startIdx);
                        content = content.slice(startIdx + 7); // 7 = '<think>'.length
                        if (preContent) emitWithCount(preContent, false);
                        clientSideThinkParsing = true;
                      } else {
                        emitWithCount(content, false);
                        content = "";
                      }
                    }
                  }
                } else {
                  emitDelta(choice.content, false, !!reasoning);
                }
              }

              // Detect server-side tool call buffering signal (TPS keeps counting, show status)
              if (!useResponsesApi && parsed.tool_call_generating) {
                if (!clientToolCallBuffering) {
                  clientToolCallBuffering = true;
                  console.log(
                    `[CHAT] Server signaled tool call generation in progress`,
                  );
                  emitToolStatus(
                    "generating",
                    "",
                    "Generating tool call...",
                    toolIteration,
                  );
                }
              }

              // Handle tool_calls from streaming response
              // Supports both complete tool calls (vmlx-engine default) and incremental argument
              // streaming (OpenAI-style: first chunk has name, subsequent chunks append arguments)
              if (choice?.tool_calls && Array.isArray(choice.tool_calls)) {
                for (const tc of choice.tool_calls) {
                  const fn = tc.function;
                  const idx = tc.index ?? -1;
                  if (fn?.name) {
                    // New tool call: initialize (use index for positional tracking)
                    const toolCall = {
                      id:
                        tc.id ||
                        `call_${uuidv4().replace(/-/g, "").slice(0, 16)}`,
                      function: {
                        name: fn.name,
                        arguments: fn.arguments || "",
                      },
                    };
                    if (idx >= 0) {
                      receivedToolCalls[idx] = toolCall;
                    } else {
                      receivedToolCalls.push(toolCall);
                    }
                    console.log(
                      `[CHAT] Tool call detected: ${fn.name}(${(fn.arguments || "").slice(0, 100)})`,
                    );
                  } else if (fn?.arguments && idx >= 0) {
                    // Incremental argument chunk: accumulate arguments for existing tool call
                    if (receivedToolCalls[idx]) {
                      receivedToolCalls[idx].function.arguments += fn.arguments;
                    } else {
                      // Out-of-order index: initialize a placeholder to prevent sparse array crash
                      receivedToolCalls[idx] = {
                        id:
                          tc.id ||
                          `call_${uuidv4().replace(/-/g, "").slice(0, 16)}`,
                        function: { name: "", arguments: fn.arguments },
                      };
                    }
                  }
                }
              }
            }
          } catch (e) {
            // Error events are valid SSE JSON, not malformed lines. Propagate
            // them to the outer request catch so the empty assistant placeholder
            // is removed and the renderer receives the real server failure.
            // The previous catch logged and swallowed these intentional throws,
            // turning prefill failures into a false zero-token completion.
            if (
              shouldRethrowChatStreamLineError(
                e,
                isExpectedChatBackendDisconnectError(e),
              )
            ) {
              throw e;
            }
            // Skip malformed JSON lines — log at debug level for troubleshooting
            if (e instanceof SyntaxError) {
              // Expected: malformed SSE data line
            } else {
              console.warn(
                "[CHAT] Error processing SSE line:",
                (e as Error).message,
              );
            }
          }
        };

        // ─── Helper: stream SSE response through processLine ──────────────
        const streamSSE = async (
          rdr: ReadableStreamDefaultReader<Uint8Array>,
        ) => {
          const dec = new TextDecoder();
          let buf = "";
          const readNext = async (): Promise<
            ReadableStreamReadResult<Uint8Array>
          > => {
            if (!clientToolCallBuffering) return rdr.read();
            let timer: ReturnType<typeof setTimeout> | undefined;
            try {
              return await Promise.race([
                rdr.read(),
                new Promise<ReadableStreamReadResult<Uint8Array>>(
                  (resolve) => {
                    timer = setTimeout(async () => {
                      console.warn(
                        `[CHAT] Tool call generation stalled for ${TOOL_STREAM_STALL_TIMEOUT_MS}ms; cancelling stalled stream`,
                      );
                      emitToolStatus(
                        "error",
                        "",
                        "Tool call generation stalled; cancelled stalled stream.",
                        toolIteration,
                      );
                      clientToolCallBuffering = false;
                      try {
                        await rdr.cancel();
                      } catch (_) {}
                      resolve({ value: undefined, done: true });
                    }, TOOL_STREAM_STALL_TIMEOUT_MS);
                  },
                ),
              ]);
            } finally {
              if (timer) clearTimeout(timer);
            }
          };
          while (true) {
            // Check abort before each read — fast models can buffer many chunks
            if (abortController.signal.aborted) break;
            const { value, done } = await readNext();
            if (done) break;
            // Includes SSE comments (`: keep-alive`) emitted during prefill.
            // Receipt of bytes proves that the request path is alive even
            // before the model can produce its first token.
            if (value && value.byteLength > 0) armFetchInactivityTimeout();
            buf += dec.decode(value, { stream: true });
            const lines = buf.split("\n");
            buf = lines.pop() || "";
            for (let li = 0; li < lines.length; li++) {
              if (abortController.signal.aborted) break;
              processLine(lines[li].trim());
              if (toolStatusNeedsFlush) {
                await flushToolStatusToRenderer();
              }
              if (rendererStreamNeedsFlush) {
                await flushStreamDeltaToRenderer();
              }
            }
          }
          if (abortController.signal.aborted) return;
          const rem = dec.decode(); // flush TextDecoder streaming buffer
          if (rem) buf += rem;
          // Process remaining lines (may contain multiple newline-separated events)
          if (buf.trim()) {
            for (const line of buf.split("\n")) {
              if (abortController.signal.aborted) break;
              if (line.trim()) {
                processLine(line.trim());
                if (toolStatusNeedsFlush) {
                  await flushToolStatusToRenderer();
                }
                if (rendererStreamNeedsFlush) {
                  await flushStreamDeltaToRenderer();
                }
              }
            }
          }
        };

        // Release whatever is still withheld. A fragment that never grew into a
        // tag is ordinary prose (an answer ending in "<" holds back one char),
        // so it must be emitted, not dropped - and it has to reach the renderer
        // as well as fullContent or the live view and the saved record diverge.
        const flushToolTagHoldback = () => {
          if (!toolTagHoldback) return;
          const pending = toolTagHoldback;
          toolTagHoldback = "";
          emitDelta(pending, false, true, true);
        };

        const reconcileResponsesToolBuffer = () => {
          const reconciliation = reconcileResponsesToolBufferAtStreamEnd({
            useResponsesApi,
            clientToolCallBuffering,
            receivedToolCallCount: receivedToolCalls.filter(Boolean).length,
            finalText: responsesFinalText,
          });
          if (!reconciliation.clearSpeculativeBuffering) return;

          clientToolCallBuffering = false;
          if (reconciliation.rejectedControlMarkup) {
            // Retained (never rendered as prose) so the finalizer can tell the
            // user why this turn has no answer instead of leaving it blank.
            rejectedControlMarkupText = reconciliation.rejectedText || "";
            const warning =
              "The model emitted parser-rejected tool control markup. It was hidden instead of being shown as assistant content.";
            console.warn(`[CHAT] ${warning}`);
            responseWarnings = Array.from(
              new Set([...(responseWarnings || []), warning]),
            );
            emitToolStatus(
              "error",
              "",
              warning,
              toolIteration,
            );
            return;
          }
          if (reconciliation.authoritativeText === null) return;

          console.log(
            `[CHAT] Restoring ${reconciliation.authoritativeText.length} authoritative Responses text chars after zero-tool speculative buffering`,
          );
          // Replace only the current iteration. allGeneratedContent owns text
          // from completed tool iterations and emitDelta will prepend it for UI.
          fullContent = "";
          rawAccumulated = "";
          // The shared reconciliation helper has already rejected tool
          // control markup. Bypass speculative marker detection only for the
          // remaining authoritative prose so a false-positive heartbeat
          // cannot swallow the completed answer.
          emitDelta(reconciliation.authoritativeText, false, false, true);
          _sawResponsesTextDelta = true;
        };

        await streamSSE(reader);
        if (pendingStreamServerError) throw pendingStreamServerError;
        reconcileResponsesToolBuffer();

        // ─── Helper: send follow-up request and stream response ────────────
        const sendFollowUp = async (): Promise<boolean> => {
          finishServerDecodePass();
          // Fold the finished stream's prompt/cached counts into the exchange
          // totals so the final metrics pair coherently (cached <= prompt).
          exchangePromptTokens += promptTokens;
          exchangeCachedTokens += Math.min(cachedTokens, promptTokens);
          promptTokens = 0;
          cachedTokens = 0;
          // Reset SSE parser state from previous stream
          currentEventType = "";
          pendingStreamServerError = null;
          seenResponsesApiEvents.clear();
          _sawResponsesTextDelta = false;
          responsesFinalText = "";
          // Reset fetchStartTime so TTFT for follow-up is measured correctly
          fetchStartTime = Date.now();
          firstTokenTime = null;
          lastTokenTime = null;
          // Use the same wire API format as the initial request
          const url = useResponsesApi
            ? `${baseUrl}/v1/responses`
            : `${baseUrl}/v1/chat/completions`;
          // Remote internet providers use Electron net.fetch; loopback model
          // servers use Node streaming so SSE tool events are not buffered.
          const followUpBody = JSON.stringify(buildRequestBody());
          logRequestShape(followUpBody, "follow_up");
          const followUpInit = {
            method: "POST",
            headers: {
              "Content-Type": "application/json",
              ...authHeaders,
              // Every local Responses HTTP pass needs the same private usage
              // negotiation. Omitting it only on tool follow-ups left the
              // final footer with first-pass token/decode counts paired to
              // final-pass TTFT/prefill throughput.
              ...vmlxResponsesUsageHeaders,
              ...nextLocalRequestCorrelationHeaders(),
              ...(!isRemote && plannedDirectAnswerPass && !finalAnswerRecovery
                ? { "X-vMLX-Tool-Choice-Fulfilled": "1" }
                : {}),
            },
            body: followUpBody,
            signal: abortController.signal,
          };
          const useNodeStreamingFetch = !isRemote || isLoopbackUrl(url);
          const res = useNodeStreamingFetch
            ? await streamingFetch(url, followUpInit as any)
            : await remoteFetch(url, followUpInit);
          if (!res.ok) {
            const errText = await res.text();
            console.log(`[CHAT] Follow-up failed: ${res.status} ${errText}`);
            emitToolStatus(
              "error",
              "",
              `Follow-up error: ${res.status} ${errText}`,
              toolIteration,
            );
            return false;
          }
          const followUpReader = res.body?.getReader();
          if (!followUpReader) return false;
          await streamSSE(followUpReader);
          if (pendingStreamServerError) throw pendingStreamServerError;
          reconcileResponsesToolBuffer();
          return true;
        };

        // ─── Helper: execute tool calls and push results to messages ───────
        const executeToolCalls = async () => {
          const toolReasoning = currentReasoningSegment();
          if (useResponsesApi) {
            // Responses API: push individual output items (not Chat Completions format)
            if (toolReasoning.trim()) {
              requestMessages.push(responsesReasoningItem(toolReasoning));
            }
            if (fullContent) {
              requestMessages.push({ type: "output_text", text: fullContent });
            }
            for (const tc of receivedToolCalls) {
              requestMessages.push({
                type: "function_call",
                call_id: tc.id,
                name: tc.function.name,
                arguments: tc.function.arguments,
              });
            }
          } else {
            // Chat Completions: push assistant message with tool_calls array
            const assistantToolTurn: any = {
              role: "assistant",
              content: fullContent || null,
              tool_calls: receivedToolCalls.map((tc) => ({
                id: tc.id,
                type: "function" as const,
                function: {
                  name: tc.function.name,
                  arguments: tc.function.arguments,
                },
              })),
            };
            if (toolReasoning.trim()) {
              assistantToolTurn.reasoning_content = toolReasoning;
            }
            requestMessages.push(assistantToolTurn);
          }

          const pendingImageDataUrls: string[] = [];
          const pendingVideoDataUrls: string[] = [];
          for (const tc of receivedToolCalls) {
            // Check abort between each tool — don't make user wait for all tools to finish
            if (abortController.signal.aborted)
              throw Object.assign(new Error("AbortError"), {
                name: "AbortError",
              });
            let resultText = "";
            const normalizedToolName = tc.function.name.toLowerCase();
            const isExactlyOnceTool =
              exactlyOnceToolNames.includes(normalizedToolName);
            if (
              isExactlyOnceTool &&
              completedExactlyOnceTools.has(normalizedToolName)
            ) {
              // A rejected tool call is still an intervening action in the
              // model transcript, so it ends any replay-safe read chain.
              lastReplaySafeToolResultKey = null;
              resultText = `Duplicate ${tc.function.name} call was not executed because the user requested it exactly once.`;
              emitToolStatus(
                "calling",
                tc.function.name,
                tc.function.arguments,
                toolIteration,
                tc.id,
              );
              emitToolStatus(
                "error",
                tc.function.name,
                resultText,
                toolIteration,
                tc.id,
              );
              requestMessages.push(
                useResponsesApi
                  ? {
                      type: "function_call_output",
                      call_id: tc.id,
                      output: resultText,
                    }
                  : {
                      role: "tool",
                      tool_call_id: tc.id,
                      content: resultText,
                    },
              );
              continue;
            }
            // A parsed stream item is still speculative until the complete
            // pass reaches loop control. Emit the concrete card only when this
            // call is actually handed to execution/validation. That lets a
            // replay-only exactly-once pass be discarded without leaving an
            // orphaned "interrupted" card in the live UI or persisted history.
            emitToolStatus(
              "calling",
              tc.function.name,
              tc.function.arguments,
              toolIteration,
              tc.id,
            );
            try {
              let toolArgs: Record<string, any>;
              try {
                toolArgs = JSON.parse(tc.function.arguments || "{}");
              } catch (parseErr) {
                // Invalid arguments cannot be compared safely and therefore
                // break consecutiveness before the model is allowed to retry.
                lastReplaySafeToolResultKey = null;
                resultText = `Invalid tool arguments: ${(parseErr as Error).message}`;
                emitToolStatus(
                  "error",
                  tc.function.name,
                  resultText,
                  toolIteration,
                  tc.id,
                );
                requestMessages.push(
                  useResponsesApi
                    ? {
                        type: "function_call_output",
                        call_id: tc.id,
                        output: resultText,
                      }
                    : {
                        role: "tool",
                        tool_call_id: tc.id,
                        content: resultText,
                      },
                );
                continue;
              }
              const toolAuthorized = isToolAuthorizedForCurrentTurn(
                normalizedToolName,
                // Authorize against the immutable catalog provided for this
                // user turn. The filtered follow-up catalog intentionally
                // drops a just-completed exactly-once tool; consulting it
                // here would reject that tool's first real execution.
                currentToolNames,
                currentPromptAlreadyForbidsTools,
                restrictedToolNamesForCurrentTurn,
              );
              const replaySafeKey = toolAuthorized
                ? replaySafeToolCallKey(normalizedToolName, toolArgs)
                : undefined;
              if (
                replaySafeKey &&
                lastReplaySafeToolResultKey === replaySafeKey
              ) {
                resultText =
                  `Duplicate ${tc.function.name} call was deduplicated: ` +
                  `the immediately preceding call used identical arguments and ` +
                  `completed successfully. Use its real result above.`;
                console.log(
                  `[CHAT] Deduplicated consecutive replay-safe tool call: ${tc.function.name}`,
                );
                emitToolStatus(
                  "result",
                  tc.function.name,
                  resultText,
                  toolIteration,
                  tc.id,
                );
                await flushToolStatusToRenderer();
                requestMessages.push(
                  useResponsesApi
                    ? {
                        type: "function_call_output",
                        call_id: tc.id,
                        output: resultText,
                      }
                    : {
                        role: "tool",
                        tool_call_id: tc.id,
                        content: resultText,
                      },
                );
                continue;
              }
              // Only consecutive successful replay-safe reads qualify. Any
              // different, rejected, failed, mutating, command, or MCP call
              // breaks the chain before execution.
              lastReplaySafeToolResultKey = null;
              if (toolAuthorized) {
                if (isExactlyOnceTool) {
                  // Mark only after schema-valid arguments and current-turn
                  // authorization. A malformed first emission may retry on the
                  // follow-up; once one valid call reaches execution, any
                  // second call in the same model pass is rejected at the loop
                  // head.
                  completedExactlyOnceTools.add(normalizedToolName);
                }
                // Do not display or persist an "executing" phase for a call
                // rejected by current-turn authorization.
                emitToolStatus(
                  "executing",
                  tc.function.name,
                  undefined,
                  toolIteration,
                  tc.id,
                );
                await flushToolStatusToRenderer();
              }

              if (!toolAuthorized) {
                // A model may emit a built-in or MCP name from training or
                // history even when that schema was not authorized for this
                // request. A scoped "file_info exactly once" turn must not
                // execute write_file or an unrelated MCP function, and a
                // no-tools turn must execute nothing. Preserve the rejected
                // call/result in the transcript, but never perform the side
                // effect.
                resultText = `Tool "${tc.function.name}" was not provided for this request and was not executed.`;
                emitToolStatus(
                  "error",
                  tc.function.name,
                  resultText,
                  toolIteration,
                  tc.id,
                );
              } else if (tc.function.name === "ask_user") {
                // Special handling: ask_user needs IPC to renderer, not executor
                const question =
                  toolArgs.question || "What would you like to do?";
                emitToolStatus("asking", "ask_user", question, toolIteration, tc.id);
                resultText = await new Promise<string>((resolve) => {
                  const win = getWindow();
                  if (!win || win.isDestroyed()) {
                    resolve("(User interface not available)");
                    return;
                  }
                  if (abortController.signal.aborted) {
                    resolve("(Generation was stopped)");
                    return;
                  }
                  win.webContents.send("chat:askUser", { chatId, question });
                  // Use Map-based resolver (single global listener, no per-call listener accumulation)
                  const cleanup = () => {
                    askUserResolvers.delete(chatId);
                    clearTimeout(askTimeout);
                    abortController.signal.removeEventListener(
                      "abort",
                      onAbort,
                    );
                  };
                  askUserResolvers.set(chatId, (answer: string) => {
                    cleanup();
                    resolve(answer);
                  });
                  const onAbort = () => {
                    cleanup();
                    resolve("(Generation was stopped)");
                  };
                  abortController.signal.addEventListener("abort", onAbort, {
                    once: true,
                  });
                  const askTimeout = setTimeout(() => {
                    cleanup();
                    resolve("(User did not respond within 5 minutes)");
                  }, 300000);
                });
                emitToolStatus("result", "ask_user", resultText, toolIteration, tc.id);
              } else if (isBuiltinTool(tc.function.name)) {
                // Enforce tool category toggles at execution time (defense-in-depth:
                // filterTools removes disabled tools from definitions sent to model,
                // but models can hallucinate tool calls not in the provided list)
                const disabledSet = getDisabledTools(overrides || {});
                if (disabledSet.has(tc.function.name)) {
                  resultText = `Tool "${tc.function.name}" is disabled in chat settings.`;
                  emitToolStatus(
                    "error",
                    tc.function.name,
                    resultText,
                    toolIteration,
                    tc.id,
                  );
                } else if (
                  !overrides?.workingDirectory &&
                  !WORKING_DIR_INDEPENDENT_TOOLS.has(tc.function.name)
                ) {
                  // The SAME exemption the executor applies, from the same set.
                  // Without it here this gate rejected EVERY tool before
                  // executeBuiltinTool ever ran, so the executor's exemption was
                  // unreachable whenever the directory was UNSET -- it only
                  // helped a configured-but-broken one. Searching the web,
                  // fetching a URL, reading the clipboard or the clock need no
                  // folder.
                  resultText =
                    "Error: Working directory not set. Configure it in Chat Settings.";
                  emitToolStatus(
                    "error",
                    tc.function.name,
                    resultText,
                    toolIteration,
                    tc.id,
                  );
                } else {
                  // Undefined only for the working-dir-independent tools the
                  // guard above lets through; executeBuiltinTool never reads it
                  // for those, and for every other tool the guard proves it set.
                  const workDir = overrides?.workingDirectory ?? "";
                  console.log(`[CHAT] Builtin tool: ${tc.function.name}`);
                  const result = await executeBuiltinTool(
                    tc.function.name,
                    toolArgs,
                    workDir,
                    overrides?.toolResultMaxChars,
                  );
                  resultText = result.content;
                  if (replaySafeKey && !result.is_error) {
                    lastReplaySafeToolResultKey = replaySafeKey;
                  }
                  // For read_image/read_video: inject media as multimodal
                  // content for VLM follow-ups. Tool result text is not enough
                  // for vision models to actually inspect local media bytes.
                  // Keep local text-only models on text; vmlx-engine will
                  // reject image_url/video_url content when the loaded runtime
                  // is not multimodal. Remote endpoints may be multimodal even
                  // when the local session registry cannot infer it.
                  if ((result.imageDataUrl || result.videoDataUrl) && !(chatIsMultimodal || isRemote)) {
                    resultText += "\n\n[Media bytes were not sent because this local session is text-only. Use a VL-compatible model/session to inspect the file visually.]";
                    console.log(
                      `[CHAT] Skipping tool media bytes for text-only local session (${tc.function.name})`,
                    );
                  } else {
                    if (result.imageDataUrl) {
                      pendingImageDataUrls.push(result.imageDataUrl);
                    }
                    if (result.videoDataUrl) {
                      pendingVideoDataUrls.push(result.videoDataUrl);
                    }
                  }
                  emitToolStatus(
                    result.is_error ? "error" : "result",
                    tc.function.name,
                    resultText,
                    toolIteration,
                    tc.id,
                  );
                }
              } else if (isRemote) {
                // MCP tool passthrough is only available on local vmlx-engine servers
                resultText = `MCP tool "${tc.function.name}" is only available with local vmlx-engine sessions.`;
                emitToolStatus(
                  "error",
                  tc.function.name,
                  resultText,
                  toolIteration,
                  tc.id,
                );
              } else {
                const execRes = await fetch(`${baseUrl}/v1/mcp/execute`, {
                  method: "POST",
                  headers: {
                    "Content-Type": "application/json",
                    ...authHeaders,
                  },
                  body: JSON.stringify({
                    tool_name: tc.function.name,
                    arguments: toolArgs,
                  }),
                  signal: abortController.signal,
                });
                if (!execRes.ok) {
                  const errText = await execRes.text();
                  resultText = `Error (${execRes.status}): ${errText}`;
                  emitToolStatus(
                    "error",
                    tc.function.name,
                    resultText,
                    toolIteration,
                    tc.id,
                  );
                } else {
                  const result = await execRes.json();
                  if (result.is_error) {
                    resultText = `Error: ${result.error_message || "Unknown error"}`;
                    emitToolStatus(
                      "error",
                      tc.function.name,
                      resultText,
                      toolIteration,
                      tc.id,
                    );
                  } else {
                    resultText =
                      typeof result.content === "string"
                        ? result.content
                        : JSON.stringify(result.content, null, 2);
                    // Apply same truncation as built-in tools to prevent context overflow
                    const mcpMaxChars = overrides?.toolResultMaxChars || 50000;
                    if (resultText.length > mcpMaxChars) {
                      resultText =
                        resultText.slice(0, mcpMaxChars) +
                        `\n\n[Truncated — showing first ${mcpMaxChars} of ${resultText.length} characters]`;
                    }
                    emitToolStatus(
                      "result",
                      tc.function.name,
                      resultText,
                      toolIteration,
                      tc.id,
                    );
                  }
                }
              }
            } catch (err: any) {
              if (err?.name === "AbortError") throw err;
              resultText = `Tool execution error: ${err.message}`;
              emitToolStatus(
                "error",
                tc.function.name,
                err.message,
                toolIteration,
                tc.id,
              );
            }
            await flushToolStatusToRenderer();

            requestMessages.push(
              useResponsesApi
                ? {
                    type: "function_call_output",
                    call_id: tc.id,
                    output: resultText,
                  }
                : { role: "tool", tool_call_id: tc.id, content: resultText },
            );
            if (exactFinalToolNames.includes(normalizedToolName)) {
              completedExactFinalTools.add(normalizedToolName);
            }
          }

          // Inject media from read_image/read_video tool results as multimodal
          // content parts. VL models can only process media in content arrays,
          // not in tool result strings. Text FIRST, then media — Qwen VL
          // expects this order.
          const contentParts = buildToolMediaFollowupContent(
            pendingImageDataUrls,
            pendingVideoDataUrls,
          );
          if (contentParts) {
            requestMessages.push({ role: "user", content: contentParts });
            console.log(
              `[CHAT] Injected ${pendingImageDataUrls.length} image(s), ${pendingVideoDataUrls.length} video(s) as multimodal content for VLM`,
            );
          }
        };

        console.log(
          `[CHAT] Stream ended — content: ${fullContent.length} chars, reasoning: ${reasoningContent.length} chars, tool calls: ${receivedToolCalls.length}, buffered: ${clientToolCallBuffering}`,
        );

        // ─── Unified Tool Execution + Auto-Continue Loop ───────────────────
        // Handles both tool call execution and auto-continuation for models
        // that stop after tool use without providing a response.
        // A reasoning-only/empty post-tool completion gets one direct-answer
        // recovery. Repeating the same generic prompt three times created
        // duplicate partial reasoning cards and misleading cumulative metrics.
        const AUTO_CONTINUE_TOKEN_THRESHOLD = 100;
        const MAX_AUTO_CONTINUES = 1;
        let autoContinueCount = 0;
        while (toolIteration < MAX_TOOL_ITERATIONS) {
          // Compact sparse array: parallel tool calls at non-contiguous indices create holes
          // that for...of silently skips. Filter to only real entries.
          if (receivedToolCalls.length > 0) {
            receivedToolCalls = receivedToolCalls.filter(Boolean);
          }
          if (
            repeatsOnlyCompletedExactlyOnceTools(
              receivedToolCalls.map((tc) => tc.function.name),
              exactlyOnceToolNames,
              completedExactlyOnceTools,
            )
          ) {
            // The prior pass already executed the user's exactly-once contract.
            // Treat a replay-only model pass as a failed answer pass, not as
            // progress through another tool iteration. Counting it as progress
            // resets autoContinueCount and lets a stubborn model repeat the
            // same rejected call until MAX_TOOL_ITERATIONS. Drop the replayed
            // control output and enter the existing one-shot answer-only
            // recovery below; the real tool result is already in history.
            console.warn(
              "[CHAT] Completed exactly-once tool was replayed; using bounded answer-only recovery",
            );
            receivedToolCalls = [];
            fullContent = "";
            rawAccumulated = "";
            lastFinishReason = "stop";
          }
          if (receivedToolCalls.length > 0) {
            // ── Model made tool calls: execute and send follow-up ──
            toolIteration++;
            autoContinueCount = 0; // reset — model is making progress
            console.log(
              `[CHAT] Tool execution iteration ${toolIteration} (${receivedToolCalls.length} tool calls)`,
            );
            const contentBeforeToolPreviewCleanup = fullContent;
            fullContent = stripRedundantNamespacedToolPreview(
              fullContent,
              receivedToolCalls,
            );
            const clearedRedundantToolPreview =
              contentBeforeToolPreviewCleanup !== fullContent;
            if (clearedRedundantToolPreview) {
              console.log(
                `[CHAT] Removed redundant namespaced tool preview before Responses continuation`,
              );
            }
            // Preserve content before tool execution so abort can recover it
            flushToolTagHoldback();
            if (fullContent.trim()) {
              allGeneratedContent +=
                (allGeneratedContent ? "\n\n" : "") + fullContent.trim();
            }
            // Flush accumulated content to renderer before blocking on tool execution
            try {
              const win = getWindow();
              if (
                win &&
                !win.isDestroyed() &&
                (allGeneratedContent.trim() || clearedRedundantToolPreview)
              ) {
                win.webContents.send("chat:stream", {
                  chatId,
                  messageId: assistantMessage.id,
                  fullContent: allGeneratedContent,
                  isReasoning: false,
                  metrics: {
                    tokenCount: cumulativeTokenOffset + iterationTokenCount,
                    promptTokens,
                    cachedTokens,
                    cacheDetail,
                    tokensPerSecond: liveTps.toFixed(1),
                    ttft: firstTokenTime
                      ? ((firstTokenTime - fetchStartTime) / 1000).toFixed(2)
                      : "0",
                    elapsed: (generationMs / 1000).toFixed(1),
                  },
                });
              }
            } catch (_) {}
            // Reset idle timer before tool execution — builtin tools run locally
            // without server contact, so the model could sleep during long tool runs
            if (chatSession) sessionManager.touchSession(chatSession.id);
            const touchInterval = chatSession
              ? setInterval(() => {
                  sessionManager.touchSession(chatSession.id);
                }, 30000)
              : null; // Ping every 30s during tool execution

            // AppleScript-8B is a single native-action specialist: its model
            // card defines one run_applescript call and explicitly says the
            // agent executes that result. Live Electron follow-ups showed the
            // fine-tune re-emits the same call indefinitely when the tool
            // schema remains available after a successful result. Treat that
            // bundle-owned one-call contract as terminal while leaving every
            // general ZAYA/tool bundle on the normal multi-tool loop.
            const finishAfterNativeToolResult =
              shouldFinishZayaAppleScriptToolRound(
                chatUsesZayaAppleScriptToolBundle,
                receivedToolCalls.map((tc) => tc.function.name),
              );

            try {
              await executeToolCalls();
            } finally {
              if (touchInterval) clearInterval(touchInterval);
            }
            receivedToolCalls = [];
            fullContent = "";
            rawAccumulated = "";
            currentEventType = "";
            seenResponsesApiEvents.clear();
            lastFinishReason = undefined; // Reset for next iteration
            // Reset content offset tracker to match the accumulated content position
            lastEmittedContentLength = allGeneratedContent.length
              ? allGeneratedContent.length + 2
              : 0;
            clientToolCallBuffering = false;
            clientSideThinkParsing = false;
            clientSideThinkHoldback = "";
            cumulativeTokenOffset += iterationTokenCount; // Save completed iteration tokens for cumulative total
            iterationTokenBase = tokenCount; // Save cumulative base for server-usage delta
            iterationTokenCount = 0;
            tpsSnapshots.length = 0;
            liveTps = 0;
            tpsTokenBase = tokenCount; // Reset rolling TPS for fresh generation phase
            serverSendsUsage = false; // Re-detect for new HTTP request (server restarts completion_tokens from 1)
            // Fire reasoningDone if model was still in reasoning mode when tool calls appeared
            if (isReasoning && reasoningContent) {
              try {
                const win = getWindow();
                if (win && !win.isDestroyed()) {
                  win.webContents.send("chat:reasoningDone", {
                    chatId,
                    messageId: assistantMessage.id,
                    reasoningContent: currentReasoningContent(),
                    reasoningSegments: currentReasoningSegments(),
                  });
                }
              } catch (_) {}
            }
            // chat:reasoningDone emitted above before resetting for the next segment.
            isReasoning = false; // Reset reasoning state for new iteration
            reasoningSegments = markReasoningToolBoundary(reasoningSegments);
            reasoningContent = ""; // Start a fresh reasoning segment for the next iteration
            // (thinking indicator removed)
            if (finishAfterNativeToolResult) {
              console.log(
                "[CHAT] ZAYA AppleScript native tool result completed the one-call bundle contract",
              );
              break;
            }
            emitToolStatus("processing", "", undefined, toolIteration);
            const exactFinalToolsComplete =
              exactFinalToolNames.length > 0 &&
              exactFinalToolNames.every((name) =>
                completedExactFinalTools.has(name),
              );
            const exactlyOnceToolsComplete =
              exactlyOnceToolNames.length > 0 &&
              exactlyOnceToolNames.every((name) =>
                completedExactlyOnceTools.has(name),
              );
            plannedDirectAnswerPass = shouldPlanCompletedToolAnswerPass({
              directAnswerAfterSingleTool,
              exactFinalToolCount: exactFinalToolNames.length,
              exactFinalToolsComplete,
              exactlyOnceToolCount: exactlyOnceToolNames.length,
              exactlyOnceToolsComplete,
            });
            // Reset idle timer before follow-up — tools may have consumed minutes
            if (chatSession) sessionManager.touchSession(chatSession.id);
            if (!(await sendFollowUp())) break;
          } else if (
            toolIteration > 0 &&
            autoContinueCount < MAX_AUTO_CONTINUES &&
            shouldAutoContinueAfterToolUse({
              content: fullContent,
              iterationTokenCount,
              finishReason: lastFinishReason,
              thresholdTokens: AUTO_CONTINUE_TOKEN_THRESHOLD,
            })
          ) {
            // ── Auto-continue: model stopped without a substantive response after tool use ──
            // This handles two cases:
            // 1. Model generated ZERO content after tool results (just stopped)
            // 2. Model hit the length limit with a brief/incomplete response
            autoContinueCount++;
            finalAnswerRecovery = true;
            flushToolTagHoldback();
            const hasContent = fullContent.trim().length > 0;
            console.log(
              `[CHAT] Auto-continue ${autoContinueCount}/${MAX_AUTO_CONTINUES}: model stopped with ${iterationTokenCount} tokens (iteration), content=${hasContent}`,
            );
            if (hasContent) {
              allGeneratedContent +=
                (allGeneratedContent ? "\n\n" : "") + fullContent.trim();
              if (useResponsesApi) {
                requestMessages.push({
                  type: "output_text",
                  text: fullContent,
                });
              } else {
                requestMessages.push({
                  role: "assistant",
                  content: fullContent,
                });
              }
            }
            const continuePrompt =
              "The tool already ran and its real result is above. Complete the original request now. Return only the requested final answer; do not call another tool or summarize these instructions.";
            if (useResponsesApi) {
              requestMessages.push({
                type: "message",
                role: "user",
                content: continuePrompt,
              });
            } else {
              requestMessages.push({ role: "user", content: continuePrompt });
            }
            fullContent = "";
            rawAccumulated = "";
            currentEventType = "";
            seenResponsesApiEvents.clear();
            lastFinishReason = undefined; // Reset for next iteration
            clientToolCallBuffering = false;
            clientSideThinkParsing = false;
            clientSideThinkHoldback = "";
            receivedToolCalls = [];
            // Reset content offset tracker to match the accumulated content position
            lastEmittedContentLength = allGeneratedContent.length
              ? allGeneratedContent.length + 2
              : 0;
            cumulativeTokenOffset += iterationTokenCount; // Save completed iteration tokens for cumulative total
            iterationTokenBase = tokenCount; // Save cumulative base for server-usage delta
            iterationTokenCount = 0;
            tpsSnapshots.length = 0;
            liveTps = 0;
            tpsTokenBase = tokenCount; // Reset rolling TPS for fresh generation phase
            serverSendsUsage = false; // Re-detect for new HTTP request (server restarts completion_tokens from 1)
            // Fire reasoningDone if model was still in reasoning mode at auto-continue boundary
            if (isReasoning && reasoningContent) {
              try {
                const win = getWindow();
                if (win && !win.isDestroyed()) {
                  win.webContents.send("chat:reasoningDone", {
                    chatId,
                    messageId: assistantMessage.id,
                    reasoningContent: currentReasoningContent(),
                    reasoningSegments: currentReasoningSegments(),
                  });
                }
              } catch (_) {}
            }
            // chat:reasoningDone emitted above before resetting for the next segment.
            isReasoning = false; // Reset reasoning state for new iteration
            reasoningSegments = markReasoningToolBoundary(reasoningSegments);
            reasoningContent = ""; // Start a fresh reasoning segment for the next iteration
            // (thinking indicator removed)
            emitToolStatus(
              "processing",
              "",
              "Generating response...",
              toolIteration,
            );
            // Reset idle timer before auto-continue follow-up
            if (chatSession) sessionManager.touchSession(chatSession.id);
            if (!(await sendFollowUp())) break;
          } else {
            break;
          }
        }

        if (
          finalAnswerRecovery &&
          (allGeneratedContent.trim() || fullContent.trim())
        ) {
          // A successful bounded recovery supersedes only the intermediate
          // empty/reasoning-only diagnostic. Preserve parser, schema, cache,
          // tool-drop, and previous_response_id warnings.
          responseWarnings = dropSupersededRecoveryWarnings(responseWarnings);
        }

        if (
          toolIteration > 0 &&
          !allGeneratedContent.trim() &&
          !fullContent.trim()
        ) {
          const noVisibleAnswerWarning =
            "The tool completed, but the model produced no visible answer after one direct-answer recovery.";
          responseWarnings = Array.from(
            new Set([...(responseWarnings || []), noVisibleAnswerWarning]),
          );
          emitToolStatus(
            "error",
            "",
            noVisibleAnswerWarning,
            toolIteration,
          );
        }

        if (toolIteration > 0 || collectedToolStatuses.length > 0) {
          if (toolIteration > 0) {
            console.log(
              `[CHAT] Tool loop completed after ${toolIteration} iteration(s)`,
            );
          }
          emitToolStatus("done", "", undefined, toolIteration);
        }

        // Fire reasoningDone if stream ended while still in reasoning mode
        // (e.g., model only produced analysis channel, never transitioned to final)
        if (isReasoning) {
          isReasoning = false;
          try {
            const win = getWindow();
            if (win && !win.isDestroyed()) {
              win.webContents.send("chat:reasoningDone", {
                chatId,
                messageId: assistantMessage.id,
                reasoningContent: currentReasoningContent(),
                reasoningSegments: currentReasoningSegments(),
              });
            }
          } catch (_) {}
        }

        // Calculate final metrics — use generation-only time for t/s, fallback to wall clock
        const totalTime = (Date.now() - startTime) / 1000;
        const genTimeSec = generationMs > 0 ? generationMs / 1000 : 0;
        const wallTimeSec =
          firstTokenTime && lastTokenTime && lastTokenTime > firstTokenTime
            ? (lastTokenTime - firstTokenTime) / 1000
            : firstTokenTime
              ? (Date.now() - firstTokenTime) / 1000
              : totalTime;
        const finalGenSec = genTimeSec > 0.05 ? genTimeSec : wallTimeSec;
        // Use cumulative total across all tool iterations (server restarts completion_tokens per request)
        const serverDecodeSummary = summarizeServerDecodePasses([
          ...completedServerDecodePasses,
          currentServerDecodePass,
        ]);
        const totalTokenCount =
          serverDecodeSummary?.outputTokens ??
          (cumulativeTokenOffset + iterationTokenCount);
        const cumulativeDecodeTps =
          finalGenSec > 0 ? totalTokenCount / finalGenSec : 0;
        // Progressive streams have credible cumulative delta timing even when
        // the last rolling window covers only a short post-tool answer tail.
        // Buffered answer passes are the opposite: their cumulative rate is an
        // impossible burst, so selectFinalDecodeTps falls back to the median
        // observed rolling stream rate.
        const finalTps =
          serverDecodeSummary?.tokensPerSecond ??
          selectFinalDecodeTps({
            cumulativeTps: cumulativeDecodeTps,
            rollingTps: liveTpsHistory,
            lastRollingTps: liveTps,
          });
        const decodeMetricSource = serverDecodeSummary ? "server" : "client";
        // TTFT measured from fetchStartTime (excludes health check and message building overhead)
        const ttft = Math.max(
          0,
          firstTokenTime ? (firstTokenTime - fetchStartTime) / 1000 : 0,
        );
        // TTFT belongs to the final HTTP pass. Keep its prefill rate paired
        // with that pass's authoritative server usage rather than combining
        // one pass's TTFT with exchange-wide tool-loop prompt totals.
        const finalStreamPromptTokens = promptTokens;
        const finalStreamCachedTokens = Math.min(
          cachedTokens,
          finalStreamPromptTokens,
        );
        const finalPpSpeed = calculatePrefillTps({
          promptTokens: finalStreamPromptTokens,
          cachedTokens: finalStreamCachedTokens,
          ttftSeconds: ttft,
          serverUsageKnown: serverSendsUsage,
        });

        // Release any withheld tail before the final content is assembled.
        flushToolTagHoldback();
        // Combine content from all tool iterations into the final message
        if (allGeneratedContent && fullContent.trim()) {
          fullContent = allGeneratedContent + "\n\n" + fullContent;
        } else if (allGeneratedContent && !fullContent.trim()) {
          fullContent = allGeneratedContent;
        }

        // Strip any remaining template tokens and leaked tool call XML.
        // preSanitizeContent backs the never-empty guard below: sanitizing a
        // non-empty answer down to nothing hides the model's real output
        // (e.g. an unparsed textual tool-call dialect) behind a blank turn.
        const preSanitizeContent = fullContent.trim();
        fullContent = fullContent.replace(TEMPLATE_TOKEN_REGEX, "");
        // Strip Harmony protocol residue (concatenated protocol words after template token removal)
        fullContent = fullContent.replace(
          /<\/?(?:assistant|analysis|final)+/gi,
          "",
        );
        fullContent = fullContent.replace(
          /(?:assistant\s*){1,3}(?:analysis|final)/gi,
          "",
        );
        fullContent = fullContent.replace(
          /(?:analysis|final)\s*(?:assistant\s*){1,3}/gi,
          "",
        );
        // Strip leaked tool call blocks that server didn't parse (various model
        // formats), including orphan closing tags with no opening tag — those
        // were reaching the screen as visible prose.
        fullContent = stripLeakedToolMarkup(fullContent);
        // Strip leaked Harmony protocol channel markers (GLM, GPT-OSS)
        fullContent = fullContent.replace(/<\|start\|>assistant/g, "");
        fullContent = fullContent.replace(
          /<\|channel\|>(?:analysis|final)<\|message\|>/g,
          "",
        );
        stopPeriodicSave(); // Stop periodic saves — final save below overwrites with complete content
        fullContent = fullContent.trim();
        // Never-empty sanitize guard: if stripping leaked tool-call markup
        // emptied an answer that had real text (and no tool actually ran this
        // turn to justify hiding it), surface the original verbatim in a fence
        // instead of persisting a blank assistant turn. Fencing renders the
        // raw dialect visibly instead of as invisible HTML tags.
        const visibleAfterSanitize = fullContent
          .replace(/```[^\n`]*\n?/g, "")
          .trim();
        // "No executed tool" must key on REAL tool calls / completed tool
        // iterations — the speculative buffering "Generating tool call..."
        // status also lands in collectedToolStatuses and would mask the guard.
        const neverEmptyAnswer = resolveNeverEmptyAssistantAnswer({
          visibleAfterSanitize,
          preSanitizeContent,
          rejectedControlMarkupText,
          executedToolCallCount: receivedToolCalls.filter(Boolean).length,
          priorIterationContent: allGeneratedContent,
          toolIterations: toolIteration,
          // A reasoning-only turn (reasoning rail fills, answer never starts)
          // otherwise falls through every case above and persists blank.
          reasoningContent: currentReasoningContent(),
        });
        if (neverEmptyAnswer) {
          console.log(
            `[CHAT] Never-empty guard (${neverEmptyAnswer.reason}): turn would have persisted blank; ` +
              `preSanitize=${preSanitizeContent.length}ch rejectedMarkup=${rejectedControlMarkupText.length}ch ` +
              `toolIterations=${toolIteration}`,
          );
          // Emit it, don't just persist it: a notice that only appears after the
          // final save leaves the live turn blank and makes the wire trace
          // disagree with the persisted record. skipClientCount keeps it out of
          // the token count (it is not model output); bypass keeps the marker
          // detector from re-suppressing it.
          fullContent = "";
          rawAccumulated = "";
          emitDelta(neverEmptyAnswer.content, false, true, true);
          // emitDelta restarts periodic saving when nothing had streamed yet.
          stopPeriodicSave();
        }
        // If no main content but reasoning was produced, keep them separate.
        // Reasoning stays in reasoningContent for the reasoning box; content stays empty.
        // (Previously this did fullContent = reasoningContent which triggered the anti-dup
        // check in MessageBubble, hiding the reasoning box.)
        const rawFinalReasoningContent = currentReasoningContent();
        const rawFinalReasoningSegments = currentReasoningSegments();
        const normalizeRailForPersistence = (value: string) =>
          value.replace(/\r\n/g, "\n").trim();
        const reasoningDuplicatesVisibleContent =
          !!fullContent &&
          !!rawFinalReasoningContent &&
          normalizeRailForPersistence(fullContent) ===
            normalizeRailForPersistence(rawFinalReasoningContent);
        if (reasoningDuplicatesVisibleContent) {
          console.warn(
            "[CHAT] Dropping content-identical reasoning rail before persistence",
          );
        }
        const finalReasoningContent = reasoningDuplicatesVisibleContent
          ? ""
          : rawFinalReasoningContent;
        const finalReasoningSegments = reasoningDuplicatesVisibleContent
          ? []
          : rawFinalReasoningSegments;
        // Preserve empty tool-boundary slots in SQLite for exact model-history
        // replay. The renderer receives only visible segments, but an empty
        // slot records that a tool iteration produced no reasoning and prevents
        // later segments from being attached to the wrong assistant turn.
        const replayReasoningSegments = reasoningDuplicatesVisibleContent
          ? []
          : [...reasoningSegments];
        if (!fullContent && finalReasoningContent) {
          console.log(
            `[CHAT] No main content — reasoning only (${finalReasoningContent.length} chars)`,
          );
        }
        assistantMessage.content = fullContent;
        assistantMessage.tokens = totalTokenCount;
        // Fold the final stream into the exchange totals so persisted metrics
        // and the completion log report coherent prompt/cached pairs across
        // tool-loop streams (cached can never exceed prompt).
        promptTokens = exchangePromptTokens + finalStreamPromptTokens;
        cachedTokens =
          exchangeCachedTokens +
          finalStreamCachedTokens;
        assistantMessage.metricsJson = JSON.stringify({
          tokenCount: totalTokenCount,
          promptTokens: promptTokens || undefined,
          cachedTokens: cachedTokens || undefined,
          cacheDetail: cacheDetail || undefined,
          tokensPerSecond: finalTps.toFixed(1),
          decodeMetricSource,
          ppSpeed: finalPpSpeed,
          ttft: ttft.toFixed(2),
          totalTime: totalTime.toFixed(1),
        });
        if (collectedToolStatuses.length > 0) {
          assistantMessage.toolCallsJson = JSON.stringify(
            collectedToolStatuses,
          );
        }
        if (finalReasoningContent) {
          assistantMessage.reasoningContent = finalReasoningContent;
        }
        if (replayReasoningSegments.length > 0) {
          assistantMessage.reasoningSegmentsJson = JSON.stringify(
            replayReasoningSegments,
          );
        }
        const finalResponseWarnings = appendOutputTruncationWarning(
          responseWarnings as string[] | null,
          lastFinishReason,
        );
        if (finalResponseWarnings && finalResponseWarnings.length > 0) {
          assistantMessage.warningsJson = JSON.stringify(finalResponseWarnings);
          console.warn(
            `[CHAT] Responses warning(s): ${finalResponseWarnings.join(" | ")}`,
          );
        }
        const mediaWarningWithoutVisibleActivity =
          hasMediaAttachments &&
          !fullContent &&
          !finalReasoningContent &&
          finalReasoningSegments.length === 0 &&
          collectedToolStatuses.length === 0 &&
          finalResponseWarnings &&
          finalResponseWarnings.length > 0;
        if (mediaWarningWithoutVisibleActivity) {
          // Responses can complete with an empty-warning payload instead of
          // throwing when a VLM image prefill guard rejects before generation.
          // Do not persist the failed media user turn; otherwise the next
          // text-only prompt replays the same image and repeats the guard until
          // the user manually rolls back.
          try {
            db.deleteMessage(userMessage.id);
            pushChatSessionLog(
              chatSession?.id || resolvedSession?.id,
              `[CHAT_DIAG] rolled_back_empty_warning_media_user_message=${JSON.stringify({
                chatId: chatId.slice(0, 8),
                messageId: userMessage.id.slice(0, 8),
                warnings: finalResponseWarnings,
              })}`,
            );
          } catch (_) {}
          throw new Error(
            `Media request failed before visible output: ${finalResponseWarnings.join(" | ")}`,
          );
        }

        // 2026-05-03: persist model-visible tool context separately from
        // the UI-display tool_calls_json. Without this, history replay drops
        // tool_calls and the model's chat template can't render `<tool_call>`
        // XML on continuation — Qwen3 was observed to improvise
        // `{"tool_name": ..., "arguments": ...}` JSON after an interrupted
        // tool round.
        //
        // Keep the data on the visible assistant row as JSON context, not as
        // hidden role="tool" DB rows. Older SQLite schemas CHECK message.role
        // to system/user/assistant, and renderer message lists would display
        // hidden tool rows. Request reconstruction expands this JSON back into
        // Responses or Chat-Completions-native history.
        try {
          const harvestedCalls: Array<{
            id: string;
            type: "function";
            function: { name: string; arguments: string };
          }> = [];
          const harvestedResults: Array<{
            tool_call_id: string;
            content: string;
          }> = [];
          // Only THIS turn's fresh tool exchange (from currentTurnToolStart) —
          // never the replayed prior-turn tool calls/results, which would
          // otherwise be re-persisted into this row and multiply on each replay.
          for (const m of requestMessages.slice(currentTurnToolStart)) {
            if (
              m &&
              m.role === "assistant" &&
              Array.isArray(m.tool_calls) && m.tool_calls.length > 0
            ) {
              for (const tc of m.tool_calls) {
                if (
                  tc &&
                  tc.id &&
                  tc.function &&
                  typeof tc.function.name === "string"
                ) {
                  harvestedCalls.push({
                    id: tc.id,
                    type: "function",
                    function: {
                      name: tc.function.name,
                      arguments:
                        typeof tc.function.arguments === "string"
                          ? tc.function.arguments
                          : JSON.stringify(tc.function.arguments ?? {}),
                    },
                  });
                }
              }
            } else if (
              m &&
              m.type === "function_call" &&
              typeof m.call_id === "string"
            ) {
              harvestedCalls.push({
                id: m.call_id,
                type: "function",
                function: {
                  name: typeof m.name === "string" ? m.name : "",
                  arguments:
                    typeof m.arguments === "string"
                      ? m.arguments
                      : JSON.stringify(m.arguments ?? {}),
                },
              });
            } else if (
              m &&
              m.role === "tool" &&
              typeof m.tool_call_id === "string"
            ) {
              harvestedResults.push({
                tool_call_id: m.tool_call_id,
                content:
                  typeof m.content === "string"
                    ? m.content
                    : JSON.stringify(m.content),
              });
            } else if (
              m &&
              m.type === "function_call_output" &&
              typeof m.call_id === "string"
            ) {
              harvestedResults.push({
                tool_call_id: m.call_id,
                content:
                  typeof m.output === "string"
                    ? m.output
                    : JSON.stringify(m.output),
              });
            }
          }
          if (harvestedCalls.length > 0) {
            assistantMessage.toolCallsOaiJson = JSON.stringify(
              harvestedCalls,
            );
          }
          if (harvestedResults.length > 0) {
            assistantMessage.toolResultsOaiJson = JSON.stringify(
              harvestedResults,
            );
          }
        } catch (_persistErr) {
          /* Non-fatal: persistence is best-effort; UI still works without it. */
        }
        db.addMessage(assistantMessage);

        // Send final metrics
        try {
          const win = getWindow();
          if (win && !win.isDestroyed()) {
            win.webContents.send("chat:complete", {
              chatId,
              messageId: assistantMessage.id,
              proofRequestId,
              requestIds: [...wireRequestIds],
              responseId: activeRequests.get(chatId)?.responseId,
              content: fullContent,
              reasoningContent: finalReasoningContent || undefined,
              reasoningSegments:
                finalReasoningSegments.length > 0
                  ? finalReasoningSegments
                  : undefined,
              warnings: finalResponseWarnings || undefined,
              finishReason: lastFinishReason,
              metrics: {
                tokenCount: totalTokenCount,
                promptTokens,
                cachedTokens,
                cacheDetail,
                tokensPerSecond: finalTps.toFixed(1),
                ppSpeed: finalPpSpeed,
                ttft: ttft.toFixed(2),
                totalTime: totalTime.toFixed(1),
              },
            });
          }
        } catch (_) {}

        console.log(
          `[CHAT] Response complete: ${totalTokenCount} tokens in ${totalTime.toFixed(1)}s (${finalTps.toFixed(1)} t/s, decode=${decodeMetricSource}${serverDecodeSummary ? `:${serverDecodeSummary.decodeTokens}/${serverDecodeSummary.decodeSeconds.toFixed(3)}s` : ""}, live=${liveTps.toFixed(1)} t/s, TTFT: ${ttft.toFixed(2)}s${finalStreamPromptTokens ? `, final-pass pp: ${finalStreamPromptTokens} tokens${finalStreamCachedTokens ? ` (${finalStreamCachedTokens} cached)` : ""}${finalPpSpeed ? `, ${finalPpSpeed} pp/s` : ", rate unavailable"}` : ""}, exchange prompt: ${promptTokens} tokens${cachedTokens ? ` (${cachedTokens} cached)` : ""}, usage=${serverSendsUsage ? "server" : "client"})`,
        );

        return assistantMessage;
      } catch (error) {
        stopPeriodicSave();
        // Release the SSE reader if it was acquired
        try {
          reader?.cancel();
        } catch (_) {}

        const _err = error as any;
        const errMsg = (error as Error).message || "";
        const projectedMetalHeadroomErrorContent =
          projectedMetalHeadroomChatErrorContent(errMsg);
        // GH #253: honest engine 413 (prompt_too_long) must render as a
        // graceful in-chat message, not a raw IPC error dialog.
        const promptTooLongErrorContent = promptTooLongChatErrorContent(errMsg);
        if (
          !projectedMetalHeadroomErrorContent &&
          !promptTooLongErrorContent &&
          !isExpectedChatBackendDisconnectError(error)
        ) {
          console.error("[CHAT] Error caught:", {
            message: _err?.message,
            name: _err?.name,
            code: _err?.code,
            type: _err?.constructor?.name,
            stack: _err?.stack?.split("\n").slice(0, 5).join("\n"),
            abortSignal: abortController.signal.aborted,
            timedOut,
            fullContentLen: fullContent?.length,
            readerAcquired: !!reader,
          });
        }
        pushChatSessionLog(
          chatSession?.id || resolvedSession?.id,
          `[CHAT_DIAG] request_error=${JSON.stringify({
            chatId: chatId.slice(0, 8),
            message: _err?.message,
            name: _err?.name,
            code: _err?.code,
            timedOut,
            fullContentLen: fullContent?.length || 0,
            readerAcquired: !!reader,
          })}`,
        );

        // Fire reasoningDone if interrupted during reasoning mode
        if (isReasoning) {
          isReasoning = false;
          try {
            const win = getWindow();
            if (win && !win.isDestroyed()) {
              win.webContents.send("chat:reasoningDone", {
                chatId,
                messageId: assistantMessage.id,
                reasoningContent: currentReasoningContent(),
                reasoningSegments: currentReasoningSegments(),
              });
            }
          } catch (_) {}
        }

        // Save partial response: combine all content from previous tool iterations + current.
        // allGeneratedContent holds text from completed iterations; fullContent has current iteration.
        const abortFinishReason = lastFinishReason ?? null;
        let partialContent = "";
        if (allGeneratedContent.trim() && fullContent.trim()) {
          partialContent =
            allGeneratedContent.trim() + "\n\n" + fullContent.trim();
        } else if (allGeneratedContent.trim()) {
          partialContent = allGeneratedContent.trim();
        } else {
          partialContent = fullContent.trim();
        }
        if (partialContent) {
          partialContent = partialContent.replace(TEMPLATE_TOKEN_REGEX, "");
          partialContent = stripLeakedToolMarkup(partialContent);
          partialContent = partialContent.replace(/<\|start\|>assistant/g, "");
          partialContent = partialContent.replace(
            /<\|channel\|>(?:analysis|final)<\|message\|>/g,
            "",
          );
          partialContent = partialContent.trim();
        }
        // Check abort status BEFORE save/delete decision — needed to preserve
        // tool call displays that the user already saw on screen.
        const wasAborted = abortController.signal.aborted;
        const abortTotalTokens = cumulativeTokenOffset + iterationTokenCount;
        const abortReasoningContent = currentReasoningContent();
        const abortReasoningSegments = currentReasoningSegments();
        const hadVisibleActivity =
          partialContent ||
          abortReasoningContent.trim() ||
          projectedMetalHeadroomErrorContent ||
          promptTooLongErrorContent ||
          collectedToolStatuses.length > 0 ||
          abortTotalTokens > 0;
        // True when the ONLY thing to show is the prompt-too-long bubble — no
        // generated content existed. Used below to keep the failed-media
        // rollback intact (the oversized payload must not replay next turn).
        const promptTooLongOnlyBubble =
          Boolean(promptTooLongErrorContent) &&
          !partialContent &&
          !abortReasoningContent.trim() &&
          collectedToolStatuses.length === 0 &&
          abortTotalTokens === 0;

        // Save message if we have any content, reasoning, or visible tool activity
        if (hadVisibleActivity) {
          assistantMessage.content = partialContent
            ? partialContent + "\n\n[Generation interrupted]"
            : projectedMetalHeadroomErrorContent ||
              promptTooLongErrorContent ||
              "[Generation interrupted]";
          assistantMessage.tokens = abortTotalTokens;

          // Calculate real metrics for the partial generation (not hardcoded zeros)
          const abortTotalTime = (Date.now() - startTime) / 1000;
          const abortGenSec =
            generationMs > 50
              ? generationMs / 1000
              : firstTokenTime
                ? (Date.now() - firstTokenTime) / 1000
                : abortTotalTime;
          const abortTps =
            abortGenSec > 0 && abortTotalTokens > 0
              ? abortTotalTokens / abortGenSec
              : 0;
          // Use fetchStartTime for TTFT (consistent with non-abort path)
          const abortTtft = firstTokenTime
            ? (firstTokenTime - fetchStartTime) / 1000
            : 0;
          const abortPpSpeed = calculatePrefillTps({
            promptTokens,
            cachedTokens,
            ttftSeconds: abortTtft,
            serverUsageKnown: serverSendsUsage,
          });

          const abortMetrics = {
            tokenCount: abortTotalTokens,
            promptTokens: promptTokens || undefined,
            cachedTokens: cachedTokens || undefined,
            cacheDetail: cacheDetail || undefined,
            tokensPerSecond: abortTps.toFixed(1),
            ppSpeed: abortPpSpeed,
            ttft: abortTtft.toFixed(2),
            totalTime: abortTotalTime.toFixed(1),
          };

          // Persist metricsJson to DB so reloading the chat shows real stats
          assistantMessage.metricsJson = JSON.stringify(abortMetrics);
          if (collectedToolStatuses.length > 0) {
            assistantMessage.toolCallsJson = JSON.stringify(
              collectedToolStatuses,
            );
          }
          if (abortReasoningContent) {
            assistantMessage.reasoningContent = abortReasoningContent;
          }
          if (abortReasoningSegments.length > 0) {
            assistantMessage.reasoningSegmentsJson = JSON.stringify(
              abortReasoningSegments,
            );
          }
          db.addMessage(assistantMessage);

          try {
            const win = getWindow();
            if (win && !win.isDestroyed()) {
              win.webContents.send("chat:complete", {
                chatId,
                messageId: assistantMessage.id,
                proofRequestId,
                requestIds: [...wireRequestIds],
                responseId: activeRequests.get(chatId)?.responseId,
                content: assistantMessage.content,
                reasoningContent: abortReasoningContent || undefined,
                reasoningSegments:
                  abortReasoningSegments.length > 0
                    ? abortReasoningSegments
                    : undefined,
                finishReason: abortFinishReason,
                metrics: abortMetrics,
              });
            }
          } catch (_) {}
        } else {
          // No content generated — remove the pre-inserted empty placeholder row
          try {
            db.deleteMessage(assistantMessage.id);
          } catch (_) {}
        }

        if (
          hasMediaAttachments &&
          (!hadVisibleActivity || promptTooLongOnlyBubble) &&
          !wasAborted
        ) {
          // A failed oversized media turn must not remain in local history.
          // Otherwise the next text-only prompt replays the same image payload and
          // hits the same VLM image prefill guard until the user manually rolls back.
          try {
            db.deleteMessage(userMessage.id);
            pushChatSessionLog(
              chatSession?.id || resolvedSession?.id,
              `[CHAT_DIAG] rolled_back_failed_media_user_message=${JSON.stringify({
                chatId: chatId.slice(0, 8),
                messageId: userMessage.id.slice(0, 8),
              })}`,
            );
          } catch (_) {}
        }

        // Distinguish timeout from user-initiated abort for better error messages.
        // CRITICAL: Check abortController.signal.aborted FIRST — when abort fires during
        // reader.read(), the error message can be 'terminated' instead of 'AbortError',
        // which would be misclassified as "server connection lost".
        if (timedOut) {
          throw new Error(
            `No response bytes received for ${fetchInactivitySeconds}s. The local engine may be stalled or the remote endpoint may be unavailable.`,
          );
        }
        if (wasAborted) {
          // User-initiated abort: return normally so the renderer's success path handles it.
          // Content (if any) was already saved to DB and chat:complete event sent above.
          console.log(
            `[CHAT] Abort complete — saved ${partialContent ? partialContent.length : 0} chars, ${collectedToolStatuses.length} tool statuses`,
          );
          return hadVisibleActivity ? assistantMessage : null;
        }
        if (projectedMetalHeadroomErrorContent) {
          return assistantMessage;
        }
        if (promptTooLongErrorContent) {
          // Graceful in-chat bubble already persisted + chat:complete sent —
          // return like a normal turn so no raw IPC error dialog appears.
          return assistantMessage;
        }
        // Check both error message AND error code — Node.js ConnResetException has
        // message "aborted" but code "ECONNRESET", which the message-only check missed.
        const errCode = (error as any)?.code || "";
        if (
          errMsg === "terminated" ||
          errMsg === "aborted" ||
          errMsg.includes("ECONNREFUSED") ||
          errMsg.includes("ECONNRESET") ||
          errCode === "ECONNRESET" ||
          errCode === "ECONNREFUSED" ||
          errCode === "EPIPE" ||
          errCode === "ERR_STREAM_DESTROYED" ||
          errMsg.includes("write EPIPE") ||
          errMsg.includes("Connection closed before response completed") ||
          errMsg.includes("socket hang up") ||
          isExpectedChatBackendDisconnectError(error)
        ) {
          throw new Error(
            `Server connection lost. The model server may have crashed or stopped. Try restarting the session.`,
          );
        }
        throw new Error(`Failed to send message: ${errMsg}`);
      } finally {
        // stopPeriodicSave is scoped to this block; entry/timer cleanup lives
        // in the outer finally so setup-phase throws are covered too.
        stopPeriodicSave();
      }

      } finally {
        if (fetchTimeout) clearTimeout(fetchTimeout);
        // Only delete our own entry — after an abort or stale-lock recovery,
        // a newer request may have already registered a replacement.
        const current = activeRequests.get(chatId);
        if (current && current.controller === abortController) {
          activeRequests.delete(chatId);
        }
      }
    },
  );

  // B5: Abort active generation for a chat
  ipcMain.handle("chat:abort", async (_, chatId: string) => {
    const entry = activeRequests.get(chatId);
    if (entry) {
      console.log(`[CHAT] Aborting generation for chat ${chatId}`);
      // Tell the server to cancel inference before closing the SSE stream.
      // Closing the stream first lets the disconnect cleanup remove engine
      // bookkeeping before this explicit cancellation arrives, producing a
      // misleading 404 and leaving no deterministic cancellation receipt.
      if (entry.responseId && (entry.endpoint || entry.baseUrl)) {
        try {
          // Route to correct cancel endpoint based on response ID prefix
          const cancelPath = entry.responseId.startsWith("resp_")
            ? `/v1/responses/${entry.responseId}/cancel`
            : `/v1/chat/completions/${entry.responseId}/cancel`;
          const cancelBase =
            entry.baseUrl ||
            `http://${connectHost(entry.endpoint!.host)}:${entry.endpoint!.port}`;
          const cancelRes = await fetch(`${cancelBase}${cancelPath}`, {
            method: "POST",
            headers: entry.authHeaders || {},
            signal: AbortSignal.timeout(2000),
          });
          console.log(
            `[CHAT] Server cancel sent for ${entry.responseId} — status ${cancelRes.status}`,
          );
        } catch (cancelErr: any) {
          console.log(
            `[CHAT] Server cancel failed for ${entry.responseId}: ${cancelErr.message || cancelErr}`,
          );
        }
      } else if (!entry.responseId) {
        // Abort during prefill: responseId not assigned yet. The fetch abort
        // closes the connection; the server will detect disconnect via is_disconnected()
        // on the next token yield. No explicit cancel needed — prefill is typically <2s.
        console.log(
          `[CHAT] Abort during prefill (no responseId yet) — connection closed, server will detect disconnect`,
        );
      }

      try {
        entry.controller.abort();
      } catch (_) {}

      activeRequests.delete(chatId);
      return { success: true };
    }
    return { success: false, error: "No active request for this chat" };
  });

  // Check if a chat has an active streaming generation (used for re-sync on tab switch)
  ipcMain.handle("chat:isStreaming", (_, chatId: string) => {
    return activeRequests.has(chatId);
  });

  // Clear all active locks (called on window reload/close)
  ipcMain.handle("chat:clearAllLocks", async () => {
    const count = activeRequests.size;
    for (const [chatId, entry] of activeRequests) {
      // Start server-side cancellation before closing the local stream, for
      // the same lifecycle ordering used by chat:abort.
      if (entry.responseId && (entry.endpoint || entry.baseUrl)) {
        try {
          const cancelPath = entry.responseId.startsWith("resp_")
            ? `/v1/responses/${entry.responseId}/cancel`
            : `/v1/chat/completions/${entry.responseId}/cancel`;
          const cancelBase =
            entry.baseUrl ||
            `http://${connectHost(entry.endpoint!.host)}:${entry.endpoint!.port}`;
          fetch(`${cancelBase}${cancelPath}`, {
            method: "POST",
            headers: entry.authHeaders || {},
            signal: AbortSignal.timeout(2000),
          }).catch(() => {}); // Fire-and-forget, don't block window close
          console.log(
            `[CHAT] clearAllLocks: cancel sent for ${chatId} (${entry.responseId})`,
          );
        } catch (_) {}
      }
      try {
        entry.controller.abort();
      } catch (_) {}
    }
    activeRequests.clear();
    return { cleared: count };
  });

  // Overrides — validate numeric bounds to prevent garbage values from reaching the engine
  ipcMain.handle(
    "chat:setOverrides",
    async (_, chatId: string, overrides: any) => {
      const sanitized = sanitizeChatOverrides({ ...overrides, chatId });
      db.setChatOverrides({ ...sanitized, chatId });

      return { success: true };
    },
  );

  ipcMain.handle("chat:getOverrides", async (_, chatId: string) => {
    return sanitizeChatOverrides({
      ...(db.getChatOverrides(chatId) || {}),
      chatId,
    });
  });

  ipcMain.handle("chat:clearOverrides", async (_, chatId: string) => {
    db.clearChatOverrides(chatId);
    return { success: true };
  });

  // Chat Profiles (named presets for chat settings)
  ipcMain.handle(
    "chat:saveProfile",
    async (_, name: string, overrides: any, isDefault?: boolean) => {
      const id = db.saveChatProfile(
        name,
        sanitizeChatProfileOverrides(overrides),
        isDefault,
      );
      return { id };
    },
  );

  ipcMain.handle(
    "chat:updateProfile",
    async (
      _,
      id: string,
      name: string,
      overrides: any,
      isDefault?: boolean,
    ) => {
      db.updateChatProfile(
        id,
        name,
        sanitizeChatProfileOverrides(overrides),
        isDefault,
      );
      return { success: true };
    },
  );

  ipcMain.handle("chat:getProfiles", async () => {
    return db.getChatProfiles().map((profile) => ({
      ...profile,
      overrides: sanitizeChatProfileOverrides(profile.overrides),
    }));
  });

  ipcMain.handle("chat:getDefaultProfile", async () => {
    const profile = db.getDefaultChatProfile();
    return profile ? sanitizeChatProfileOverrides(profile) : undefined;
  });

  ipcMain.handle("chat:deleteProfile", async (_, id: string) => {
    db.deleteChatProfile(id);
    return { success: true };
  });
}
