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
    expect(source).toContain("parsed.enable_thinking");
    expect(source).toContain("private applyOllamaThinking");
    expect(source).toContain('typeof parsed?.think === "boolean"');
    expect(source).toContain("openaiBody.enable_thinking = parsed.think");
    expect(source).toContain("private shouldForwardOllamaReasoningEffort");
    expect(source).toContain("openaiBody?.enable_thinking === false");
    expect(source).toContain("openaiBody.reasoning_effort = parsed.reasoning_effort");
    expect(source).toContain("openaiBody.chat_template_kwargs = parsed.chat_template_kwargs");
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
    const forwards = source.match(/openaiBody\.max_prompt_tokens = value/g) || [];
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

  it("implements Ollama HEAD/root and version probes for strict clients", () => {
    expect(source).toContain('res.end("Ollama is running\\n")');
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
    expect(source).toContain("return this.proxyRequest(req, res, session, body)");
    expect(source).toContain("path: clientReq.url");
    expect(source).toContain("method: clientReq.method");
    expect(source).toContain("...clientReq.headers");
    expect(source).toContain("if (body.length > 0) proxyReq.write(body)");
  });

  it("preserves backend status, headers, and streaming response bytes", () => {
    expect(source).toContain("clientRes.writeHead(proxyRes.statusCode || 502, proxyRes.headers)");
    expect(source).toContain("proxyRes.pipe(clientRes)");
    expect(source).toContain("preserves SSE Content-Type");
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
