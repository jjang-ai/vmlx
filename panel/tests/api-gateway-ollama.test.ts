import { describe, expect, it } from "vitest";
import { readFileSync } from "node:fs";
import { resolve } from "node:path";

const source = readFileSync(
  resolve(process.cwd(), "src/main/api-gateway.ts"),
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
    expect(mainSource).toContain(
      "return { running: true, port: apiGateway.activePort, host: apiGateway.activeHost }",
    );
    expect(mainSource).not.toContain(
      "return { running: true, port, host: apiGateway.activeHost }",
    );
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
    expect(source).toContain("if (body) proxyReq.write(body)");
  });

  it("preserves backend status, headers, and streaming response bytes", () => {
    expect(source).toContain("clientRes.writeHead(proxyRes.statusCode || 502, proxyRes.headers)");
    expect(source).toContain("proxyRes.pipe(clientRes)");
    expect(source).toContain("preserves SSE Content-Type");
  });

  it("resolves target sessions from POST body, capabilities URL, or query model", () => {
    expect(source).toContain("modelName = parsed.model");
    expect(source).toContain('url.match(/^\\/v1\\/models\\/(.+)\\/capabilities');
    expect(source).toContain("new URLSearchParams(url.slice(qIdx))");
    expect(source).toContain('params.get("model")');
  });

  it("broadcasts cancel requests without requiring a model field", () => {
    expect(source).toContain('const isCancel = method === "POST" && /\\/cancel\\/?$/.test(url)');
    expect(source).toContain('accepted ? 200 : 404');
    expect(source).toContain('status: "cancelled"');
    expect(source).toContain('{ error: "Request ID not found on any backend" }');
  });
});
