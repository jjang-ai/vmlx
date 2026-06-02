import { describe, expect, it } from "vitest";
import { readFileSync } from "node:fs";
import { resolve } from "node:path";

const source = readFileSync(
  resolve(process.cwd(), "src/main/api-gateway.ts"),
  "utf8",
);
const bodySource = readFileSync(
  resolve(process.cwd(), "src/main/gateway-body.ts"),
  "utf8",
);

describe("Ollama gateway parity contracts", () => {
  it("translates Ollama image messages into OpenAI content parts", () => {
    expect(source).toContain("private translateOllamaMessages");
    expect(source).toContain("data:image/png;base64,");
    expect(source).toContain('type: "image_url"');
    expect(source).toContain("messages: this.translateOllamaMessages");
  });

  it("maps Ollama json and schema formats to OpenAI response_format", () => {
    expect(source).toContain("private ollamaResponseFormat");
    expect(source).toContain('format === "json"');
    expect(source).toContain('type: "json_object"');
    expect(source).toContain('type: "json_schema"');
    expect(source).toContain('name: "ollama_schema"');
  });

  it("preserves reasoning deltas as Ollama thinking output", () => {
    expect(source).toContain("delta?.reasoning_content || delta?.reasoning");
    expect(source).toContain(
      "choice?.message?.reasoning_content || choice?.message?.reasoning",
    );
    expect(source).toContain("message.thinking");
  });

  it("forwards thinking kwargs through app gateway Ollama routes", () => {
    expect(source).toContain("parsed?.enable_thinking");
    expect(source).toContain("private applyOllamaThinking");
    expect(source).toContain("private normalizeOllamaBoolean");
    expect(source).toContain("const think = this.normalizeOllamaBoolean(parsed?.think)");
    expect(source).toContain("const enableThinking = this.normalizeOllamaBoolean(parsed?.enable_thinking)");
    expect(source).toContain("openaiBody.enable_thinking = think");
    expect(source).toContain("private shouldForwardOllamaReasoningEffort");
    expect(source).toContain("openaiBody?.enable_thinking === false");
    expect(source).toContain("openaiBody.reasoning_effort = parsed.reasoning_effort");
    expect(source).toContain("openaiBody.chat_template_kwargs = parsed.chat_template_kwargs");
  });

  it("does not force enable_thinking when Ollama thinking controls are omitted", () => {
    expect(source).toContain("private applyOllamaThinking");
    expect(source).not.toContain("Omitted thinking controls default on");
    expect(source).not.toContain("openaiBody.enable_thinking = true;");
  });

  it("routes Ollama generate through chat templates unless raw=true", () => {
    expect(source).toContain("const useRawCompletion = parsed.raw === true");
    expect(source).toContain("prompt: parsed.prompt || \"\"");
    expect(source).toContain("messages: [");
    expect(source).toContain('path: backendPath');
    expect(source).toContain('choice?.message?.content || ""');
  });

  it("forwards cache bypass controls through chat and generate routes", () => {
    const cacheSaltForwards =
      source.match(/openaiBody\.cache_salt = parsed\.cache_salt/g) || [];
    const skipForwards =
      source.match(
        /openaiBody\.skip_prefix_cache = parsed\.skip_prefix_cache/g,
      ) || [];
    expect(cacheSaltForwards.length).toBeGreaterThanOrEqual(2);
    expect(skipForwards.length).toBeGreaterThanOrEqual(2);
  });

  it("forwards Ollama min_p sampling through chat and generate routes", () => {
    const forwards = source.match(/openaiBody\.min_p = opts\.min_p/g) || [];
    expect(forwards.length).toBeGreaterThanOrEqual(2);
  });

  it("forwards Ollama num_ctx/max context controls as max_prompt_tokens", () => {
    expect(source).toContain("private applyOllamaPromptContextLimit");
    expect(source).toContain("opts?.num_ctx");
    expect(source).toContain("opts?.max_prompt_tokens");
    expect(source).toContain("parsed?.max_context_tokens");
    const forwards = source.match(/openaiBody\.max_prompt_tokens = Math\.floor\(parsedValue\)/g) || [];
    expect(forwards.length).toBeGreaterThanOrEqual(1);
    const calls = source.match(/this\.applyOllamaPromptContextLimit\(parsed, opts, openaiBody\)/g) || [];
    expect(calls.length).toBeGreaterThanOrEqual(2);
  });

  it("preserves backend prompt_too_long status for Ollama gateway routes", () => {
    expect(source).toContain("private sendOllamaBackendError");
    const statusChecks = source.match(/proxyRes\.statusCode \|\| 200\) >= 400/g) || [];
    expect(statusChecks.length).toBeGreaterThanOrEqual(4);
    const errorForwards = source.match(/this\.sendOllamaBackendError\(res, proxyRes\.statusCode \|\| 502/g) || [];
    expect(errorForwards.length).toBeGreaterThanOrEqual(4);
    expect(source).toContain("error?.message");
    expect(source).toContain("error?.code");
  });

  it("applies gateway timeout handling to Ollama embeddings proxy requests", () => {
    const start = source.indexOf("private async handleOllamaEmbed");
    const end = source.indexOf("// ═══════════════════════════════════════════════════════════════", start);
    const embedSource = source.slice(start, end);

    expect(embedSource).toContain('path: "/v1/embeddings"');
    expect(embedSource).toContain("timeout: this.effectiveGatewayProxyTimeoutMs(routedSession, parsed)");
    expect(embedSource).toContain('proxyReq.on("timeout"');
    expect(embedSource).toContain('this.sendJson(res, 504, { error: "Timed out" })');
  });

  it("implements Ollama HEAD/root and version probes for strict clients", () => {
    expect(source).toContain('this.endResponse(res, "Ollama is running\\n")');
    expect(source).toContain('url === "/api/version"');
    expect(source).toContain('version: "0.12.6"');
    expect(source).toContain('method === "HEAD"');
    expect(source).toContain('url === "/" || url === "/api/version"');
  });

  it("returns the actual active gateway port after restart", () => {
    const mainSource = readFileSync(
      resolve(process.cwd(), "src/main/index.ts"),
      "utf8",
    );
    expect(mainSource).toContain("function gatewayStatusPayload()");
    expect(mainSource).toContain(
      "return gatewayStatusPayload()",
    );
    expect(mainSource).not.toContain(
      "return { running: true, port, host: apiGateway.activeHost }",
    );
  });

  it("does not turn client disconnect EPIPE into the unexpected-error crash dialog", () => {
    const mainSource = readFileSync(
      resolve(process.cwd(), "src/main/index.ts"),
      "utf8",
    );
    expect(mainSource).toContain("function isExpectedClientDisconnectError");
    expect(mainSource).toContain("code === 'EPIPE'");
    expect(mainSource).toContain("code === 'ERR_STREAM_WRITE_AFTER_END'");
    expect(mainSource).toContain("write EPIPE");
    expect(mainSource).toContain("broken pipe");
    expect(mainSource).toContain("const cause = (err as any)?.cause");
    expect(mainSource).toContain("const wrappedDisconnects = [");
    expect(mainSource).toContain("(err as any)?.reason");
    expect(mainSource).toContain("(err as any)?.error");
    expect(mainSource).toContain("(err as any)?.detail");
    expect(mainSource).toContain("wrappedDisconnects.some((nested) => isExpectedClientDisconnectError(nested))");
    expect(mainSource).toContain("const nestedErrors = Array.isArray((err as any)?.errors)");
    expect(mainSource).toContain("nestedErrors.some((nested) => isExpectedClientDisconnectError(nested))");
    expect(mainSource).toContain("if (isExpectedClientDisconnectError(error))");
    expect(mainSource).toContain("if (isExpectedClientDisconnectError(reason))");
  });

  it("guards child process stdio stream EPIPE across app-managed process lanes", () => {
    const processManagerSource = readFileSync(
      resolve(process.cwd(), "src/main/process-manager.ts"),
      "utf8",
    );
    const engineManagerSource = readFileSync(
      resolve(process.cwd(), "src/main/engine-manager.ts"),
      "utf8",
    );
    const developerSource = readFileSync(
      resolve(process.cwd(), "src/main/ipc/developer.ts"),
      "utf8",
    );
    const modelsSource = readFileSync(
      resolve(process.cwd(), "src/main/ipc/models.ts"),
      "utf8",
    );
    const toolsExecutorSource = readFileSync(
      resolve(process.cwd(), "src/main/tools/executor.ts"),
      "utf8",
    );

    for (const sourceText of [
      processManagerSource,
      engineManagerSource,
      developerSource,
      modelsSource,
      toolsExecutorSource,
    ]) {
      expect(sourceText).toContain("isExpectedChildProcessStreamDisconnectError");
      expect(sourceText).toContain('code === "EPIPE"');
      expect(sourceText).toContain('code === "ECONNRESET"');
      expect(sourceText).toContain('code === "ERR_STREAM_DESTROYED"');
      expect(sourceText).toContain('code === "ERR_STREAM_WRITE_AFTER_END"');
      expect(sourceText).toContain("write EPIPE");
      expect(sourceText).toContain("broken pipe");
      expect(sourceText).toContain("const cause = (err as any)?.cause");
      expect(sourceText).toContain("const wrappedDisconnects = [");
      expect(sourceText).toContain("(err as any)?.reason");
      expect(sourceText).toContain("(err as any)?.error");
      expect(sourceText).toContain("(err as any)?.detail");
      expect(sourceText).toContain("wrappedDisconnects.some((nested) => isExpectedChildProcessStreamDisconnectError(nested))");
      expect(sourceText).toContain("const nestedErrors = Array.isArray((err as any)?.errors)");
      expect(sourceText).toContain("nestedErrors.some((nested) => isExpectedChildProcessStreamDisconnectError(nested))");
      expect(sourceText).toContain("attachChildProcessStreamErrorGuard");
      expect(sourceText).toContain(".stdout,");
      expect(sourceText).toContain(".stderr,");
    }
  });

  it("guards live proof script child stdio EPIPE while collecting e2e evidence", () => {
    const liveRealUiProof = readFileSync(
      resolve(process.cwd(), "scripts/live-real-ui-model-proof.mjs"),
      "utf8",
    );
    const liveChatToolsProof = readFileSync(
      resolve(process.cwd(), "scripts/live-chat-tools-reasoning-proof.mjs"),
      "utf8",
    );

    for (const sourceText of [liveRealUiProof, liveChatToolsProof]) {
      expect(sourceText).toContain("function attachChildProcessStreamErrorGuard");
      expect(sourceText).toContain("isSocketDisconnectError(error)");
      expect(sourceText).toContain("stream?.on('error'");
      expect(sourceText).toContain(".stdout,");
      expect(sourceText).toContain(".stderr,");
    }
  });

  it("does not relay backend stderr EPIPE lines as user-visible session errors", () => {
    const sessionsSource = readFileSync(
      resolve(process.cwd(), "src/main/sessions.ts"),
      "utf8",
    );
    const backendStderrSource = readFileSync(
      resolve(process.cwd(), "src/main/backend-stderr.ts"),
      "utf8",
    );
    expect(backendStderrSource).toContain("function isExpectedBackendStderrDisconnectLine");
    expect(backendStderrSource).toContain("normalizeBackendStderrChunk");
    expect(backendStderrSource).toContain("write EPIPE");
    expect(backendStderrSource).toContain("broken pipe");
    expect(sessionsSource).toContain("normalizeBackendStderrChunk(");
    expect(sessionsSource).toContain(
      "data: BACKEND_STDERR_DISCONNECT_NORMALIZED_LINE",
    );
    expect(sessionsSource).toContain("return");
    const stderrHandler = sessionsSource.indexOf("proc.stderr?.on('data'");
    const disconnectNormalizer = sessionsSource.indexOf(
      "normalizeBackendStderrChunk(",
      stderrHandler,
    );
    const rawErrorLog = sessionsSource.indexOf(
      "console.error(`[SERVER] ${stderrText.trimEnd()}`)",
      stderrHandler,
    );
    expect(disconnectNormalizer).toBeGreaterThan(stderrHandler);
    expect(rawErrorLog).toBeGreaterThan(disconnectNormalizer);
  });

  it("does not leave raw chat IPC backend request finalization unguarded", () => {
    const chatSource = readFileSync(
      resolve(process.cwd(), "src/main/ipc/chat.ts"),
      "utf8",
    );
    expect(chatSource).toContain("function endChatBackendRequest");
    expect(chatSource).toContain("closed?: boolean");
    expect(chatSource).toContain("!anyReq.closed");
    expect(chatSource).toContain('code === "EPIPE"');
    expect(chatSource).toContain('code === "ERR_STREAM_DESTROYED"');
    expect(chatSource).toContain('code === "ERR_STREAM_WRITE_AFTER_END"');
    expect(chatSource).toContain("broken pipe");
    expect(chatSource).toContain("const cause = (err as any)?.cause");
    expect(chatSource).toContain("const wrappedDisconnects = [");
    expect(chatSource).toContain("(err as any)?.reason");
    expect(chatSource).toContain("(err as any)?.error");
    expect(chatSource).toContain("(err as any)?.detail");
    expect(chatSource).toContain("wrappedDisconnects.some((nested) => isExpectedChatBackendDisconnectError(nested))");
    expect(chatSource).toContain("const nestedErrors = Array.isArray((err as any)?.errors)");
    expect(chatSource).toContain("nestedErrors.some((nested) => isExpectedChatBackendDisconnectError(nested))");
    const rawEnds = chatSource.match(/req\.end\(bodyBuf\);/g) || [];
    expect(rawEnds.length).toBe(1);
    expect(chatSource).toContain("endChatBackendRequest(req, bodyBuf, reject);");
    expect(chatSource).toContain("isExpectedChatBackendDisconnectError(error)");
    expect(chatSource).toContain('errCode === "EPIPE"');
    expect(chatSource).toContain('errCode === "ERR_STREAM_DESTROYED"');
    expect(chatSource).toContain('errMsg.includes("write EPIPE")');
    expect(chatSource).toContain(
      'if (isExpectedChatBackendDisconnectError(err)) {',
    );
    expect(chatSource).toContain(
      'if (!isExpectedChatBackendDisconnectError(error)) {\n          console.error("[CHAT] Error caught:",',
    );
    expect(chatSource).not.toContain(
      'console.error("[CHAT] Error caught:", {\n          message: _err?.message,',
    );
  });

  it("guards disconnect-shaped chat stream error chunks before raw logging", () => {
    const chatSource = readFileSync(
      resolve(process.cwd(), "src/main/ipc/chat.ts"),
      "utf8",
    );
    expect(chatSource).toContain("function expectedChatBackendDisconnectError");
    expect(chatSource).toContain(
      "if (isExpectedChatBackendDisconnectError(errDetail)) {",
    );
    expect(chatSource).toContain("throw expectedChatBackendDisconnectError();");
    expect(chatSource).toContain(
      "console.error(`[CHAT] Responses API error event: ${errDetail}`);",
    );
    expect(chatSource).toContain(
      "console.error(\n                  `[CHAT] Chat completions error chunk: ${errDetail}`,",
    );
    const responseErrorStart = chatSource.indexOf(
      'currentEventType === "error"',
    );
    const responseErrorLog = chatSource.indexOf(
      "console.error(`[CHAT] Responses API error event: ${errDetail}`);",
      responseErrorStart,
    );
    const chatErrorStart = chatSource.indexOf("// Handle error chunks from Chat Completions");
    const chatErrorLog = chatSource.indexOf(
      "`[CHAT] Chat completions error chunk: ${errDetail}`,",
      chatErrorStart,
    );
    expect(
      chatSource.indexOf("throw expectedChatBackendDisconnectError();", responseErrorStart),
    ).toBeLessThan(responseErrorLog);
    expect(
      chatSource.indexOf("throw expectedChatBackendDisconnectError();", chatErrorStart),
    ).toBeLessThan(chatErrorLog);
  });

  it("normalizes cache IPC endpoint EPIPE disconnects instead of surfacing raw unexpected errors", () => {
    const cacheSource = readFileSync(
      resolve(process.cwd(), "src/main/ipc/cache.ts"),
      "utf8",
    );
    expect(cacheSource).toContain("function isExpectedCacheEndpointDisconnectError");
    expect(cacheSource).toContain('code === "EPIPE"');
    expect(cacheSource).toContain('code === "ECONNRESET"');
    expect(cacheSource).toContain('code === "ERR_STREAM_DESTROYED"');
    expect(cacheSource).toContain('code === "ERR_STREAM_WRITE_AFTER_END"');
    expect(cacheSource).toContain("write EPIPE");
    expect(cacheSource).toContain("broken pipe");
    expect(cacheSource).toContain("const cause = (err as any)?.cause");
    expect(cacheSource).toContain("const wrappedDisconnects = [");
    expect(cacheSource).toContain("(err as any)?.reason");
    expect(cacheSource).toContain("(err as any)?.error");
    expect(cacheSource).toContain("(err as any)?.detail");
    expect(cacheSource).toContain("wrappedDisconnects.some((nested) => isExpectedCacheEndpointDisconnectError(nested))");
    expect(cacheSource).toContain("const nestedErrors = Array.isArray((err as any)?.errors)");
    expect(cacheSource).toContain("nestedErrors.some((nested) => isExpectedCacheEndpointDisconnectError(nested))");
    expect(cacheSource).toContain("async function fetchCacheJson");
    const rawFetches = cacheSource.match(/await fetch\(/g) || [];
    expect(rawFetches.length).toBe(1);
    const guardedFetches = cacheSource.match(/await fetchCacheJson\(/g) || [];
    expect(guardedFetches.length).toBeGreaterThanOrEqual(4);
  });

  it("aborts Ollama backend response streams when the client response closes", () => {
    expect(source).toContain("private abortProxyResponseOnClientClose");
    expect(source).toContain('res.on("close", () => {');
    expect(source).toContain("proxyRes.destroy();");
    const ollamaAbortCalls =
      source.match(/this\.abortProxyResponseOnClientClose\(res, proxyRes\);/g) || [];
    expect(ollamaAbortCalls.length).toBeGreaterThanOrEqual(3);
    expect(source).toContain(
      "this.abortProxyResponseOnClientClose(clientRes, proxyRes);",
    );
  });

  it("surfaces a usable LAN gateway URL instead of showing 0.0.0.0 to users", () => {
    const mainSource = readFileSync(
      resolve(process.cwd(), "src/main/index.ts"),
      "utf8",
    );
    const dashboardSource = readFileSync(
      resolve(process.cwd(), "src/renderer/src/components/api/ApiDashboard.tsx"),
      "utf8",
    );
    expect(mainSource).toContain("host === '0.0.0.0' ? getLanAddress()");
    expect(mainSource).toContain("displayHost");
    expect(dashboardSource).toContain("const gatewayDisplayHost = lanEnabled ? (gwLanHost || gwHost) : \"localhost\"");
    expect(dashboardSource).toContain("const gatewayUrl = `http://${gatewayDisplayHost}:${gwPort}`");
  });

  it("exposes a synchronized single-loaded-model gateway mode in main, preload, API dashboard, server settings, and tray", () => {
    const mainSource = readFileSync(
      resolve(process.cwd(), "src/main/index.ts"),
      "utf8",
    );
    const preloadSource = readFileSync(
      resolve(process.cwd(), "src/preload/index.ts"),
      "utf8",
    );
    const envSource = readFileSync(
      resolve(process.cwd(), "src/env.d.ts"),
      "utf8",
    );
    const dashboardSource = readFileSync(
      resolve(process.cwd(), "src/renderer/src/components/api/ApiDashboard.tsx"),
      "utf8",
    );
    const drawerSource = readFileSync(
      resolve(process.cwd(), "src/renderer/src/components/sessions/ServerSettingsDrawer.tsx"),
      "utf8",
    );
    const traySource = readFileSync(
      resolve(process.cwd(), "src/main/tray.ts"),
      "utf8",
    );
    expect(source).toContain("gateway_single_model_mode");
    expect(source).toContain("enforceSingleModelMode");
    expect(source).toContain("sessionManager.stopSession(s.id)");
    expect(source).toContain("single_model_mode: this.singleModelMode");
    expect(mainSource).toContain("singleModelMode: apiGateway.singleModelMode");
    expect(mainSource).toContain("gateway:setSingleModelMode");
    expect(mainSource).toContain("gateway:singleModelModeChanged");
    expect(preloadSource).toContain("setSingleModelMode");
    expect(preloadSource).toContain("onSingleModelModeChanged");
    expect(envSource).toContain("singleModelMode: boolean");
    expect(envSource).toContain("onSingleModelModeChanged");
    expect(dashboardSource).toContain("singleModelMode");
    expect(dashboardSource).toContain("handleSingleModelModeToggle");
    expect(dashboardSource).toContain("window.api.gateway?.onSingleModelModeChanged");
    expect(drawerSource).toContain("handleGatewaySingleModelModeToggle");
    expect(drawerSource).toContain("window.api.gateway?.onSingleModelModeChanged");
    expect(traySource).toContain("main.tray.singleModelMode");
    expect(traySource).toContain("gateway:singleModelModeChanged");
  });

  it("keeps single-model gateway controls localized and tailwind-safe", () => {
    const dashboardSource = readFileSync(
      resolve(process.cwd(), "src/renderer/src/components/api/ApiDashboard.tsx"),
      "utf8",
    );
    const drawerSource = readFileSync(
      resolve(process.cwd(), "src/renderer/src/components/sessions/ServerSettingsDrawer.tsx"),
      "utf8",
    );
    expect(dashboardSource + drawerSource).not.toContain("translate-x-4.5");
    expect(drawerSource).toContain("useTranslation");
    expect(drawerSource).toContain("t('main.tray.singleModelMode')");
    expect(drawerSource).toContain("t('api.singleModelModeOn')");
    expect(drawerSource).toContain("t('api.singleModelModeOff')");
    expect(drawerSource).not.toContain("API Gateway Single Model");
    expect(drawerSource).not.toContain("Gateway requests unload other local models");
  });

  it("collapses OpenAI tool arguments back to Ollama object arguments", () => {
    expect(source).toContain("private openAIToolCallsToOllama");
    expect(source).toContain("JSON.parse(args)");
    expect(source).toContain(
      "function: { name: tc.function.name, arguments: args }",
    );
  });
});

describe("Gateway passthrough contracts for non-Ollama APIs", () => {
  it("proxies OpenAI, Anthropic, Responses, cache, audio, and MCP paths verbatim", () => {
    expect(source).toMatch(
      /return\s+this\.proxyRequest\(req,\s*res,\s*routedSession,\s*body\);?/,
    );
    expect(source).toContain("path: clientReq.url");
    expect(source).toContain("method: clientReq.method");
    expect(source).toContain("...clientReq.headers");
    expect(source).toContain("if (body.length > 0 && !this.writeProxyBody(proxyReq, body))");
  });

  it("preserves backend status, headers, and streaming response bytes", () => {
    expect(source).toContain("this.writeHeadResponse(");
    expect(source).toContain("proxyRes.statusCode || 502");
    expect(source).toContain("proxyRes.headers");
    expect(source).toContain("proxyRes.on(\"data\"");
    expect(source).toContain("this.writeResponse(clientRes");
    expect(source).not.toContain("proxyRes.pipe(clientRes)");
    expect(source).toContain("preserves SSE Content-Type");
  });

  it("guards gateway streaming writes against client disconnect EPIPE errors", () => {
    expect(source).toContain("private attachResponseErrorGuard");
    expect(source).toContain('code === "EPIPE"');
    expect(source).toContain("write EPIPE");
    expect(source).toContain("const wrappedDisconnects = [");
    expect(source).toContain("(anyErr as any)?.reason");
    expect(source).toContain("(anyErr as any)?.error");
    expect(source).toContain("(anyErr as any)?.detail");
    expect(source).toContain("wrappedDisconnects.some((nested) => this.isClientDisconnectError(nested))");
    expect(source).toContain("this.attachResponseErrorGuard(res)");
    expect(source).toContain("private writeHeadResponse");
    expect(source).toContain("private writeProxyBody");
    expect(source).toContain("private endProxyRequest");
    expect(source).toContain("private writeJsonLine");
    expect(source).toContain("if (!this.writeJsonLine(res, ollamaMsg))");
    expect(source).toContain("if (this.isClientDisconnectError(err)) return;");
    expect(source).toContain("closed?: boolean");
    expect(source).toContain("!anyRes.closed");
    expect(source).toContain("!anyReq.closed");
    expect(source).toContain("if (body.length > 0 && !this.writeProxyBody(proxyReq, body))");
    expect(source).toContain("if (!this.endProxyRequest(proxyReq)) return;");
    expect(source).toContain("if (!this.writeProxyBody(proxyReq, JSON.stringify(openaiBody)))");
    expect(source).toContain("proxyRes.destroy()");
  });

  it("does not leave raw backend request end calls unguarded after disconnect", () => {
    expect(source).not.toContain("proxyReq.end()");
    expect(source).not.toContain("cancelReq.end()");
    expect(source).toContain("this.endProxyRequest(cancelReq)");
    expect(source).toContain("this.endProxyRequest(proxyReq)");
  });

  it("does not leave unguarded Ollama streaming json writes alive after disconnect", () => {
    const rawJsonLineWrites = source.match(/this\.writeJsonLine\(\s*res,/g) || [];
    const guardedJsonLineWrites =
      source.match(/if\s*\(\s*!this\.writeJsonLine\(\s*res,/g) || [];

    expect(rawJsonLineWrites.length).toBeGreaterThan(0);
    expect(guardedJsonLineWrites.length).toBe(rawJsonLineWrites.length);
  });

  it("guards every Ollama backend response stream error as a disconnect boundary", () => {
    const ollamaProxyHandlers =
      source.match(
        /const proxyReq = httpRequest\(proxyOpts, \(proxyRes\) => \{[\s\S]*?proxyReq\.on\("error"/g,
      ) || [];
    const guardedOllamaProxyHandlers = ollamaProxyHandlers.filter((handler) =>
      handler.includes('proxyRes.on("error"'),
    );

    expect(ollamaProxyHandlers.length).toBeGreaterThanOrEqual(3);
    expect(guardedOllamaProxyHandlers.length).toBe(ollamaProxyHandlers.length);
  });

  it("resolves target sessions from POST body, capabilities URL, or query model", () => {
    expect(source).toContain("modelName = extractGatewayModelFromBody(body, req.headers[\"content-type\"]");
    expect(bodySource).toContain("return typeof parsed?.model === \"string\" ? parsed.model : undefined");
    expect(source).toContain('url.match(/^\\/v1\\/models\\/(.+)\\/capabilities');
    expect(source).toContain("new URLSearchParams(url.slice(qIdx))");
    expect(source).toContain('params.get("model")');
  });

  it("keeps multipart image edits binary-safe while still resolving model fields", () => {
    expect(source).toContain("private readBody(req: IncomingMessage): Promise<Buffer>");
    expect(source).toContain("resolve(Buffer.concat(chunks))");
    expect(bodySource).toContain("extractGatewayModelFromBody");
    expect(bodySource).toContain("extractMultipartFormField");
    expect(source).not.toContain("resolve(Buffer.concat(chunks).toString())");
    expect(source).not.toContain("name=\"model\"(?:\\r?\\n");
  });

  it("broadcasts cancel requests without requiring a model field", () => {
    expect(source).toContain('const isCancel = method === "POST" && /\\/cancel\\/?$/.test(url)');
    expect(source).toContain('accepted ? 200 : 404');
    expect(source).toContain('status: "cancelled"');
    expect(source).toContain('{ error: "Request ID not found on any backend" }');
  });
});
