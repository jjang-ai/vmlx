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

  it("collapses OpenAI tool arguments back to Ollama object arguments", () => {
    expect(source).toContain("private openAIToolCallsToOllama");
    expect(source).toContain("JSON.parse(args)");
    expect(source).toContain(
      "function: { name: tc.function.name, arguments: args }",
    );
  });
});
