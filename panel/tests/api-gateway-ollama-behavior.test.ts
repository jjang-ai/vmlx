import { createServer, Server } from "node:http";
import { AddressInfo } from "node:net";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

const dbMock = vi.hoisted(() => ({
  getSetting: vi.fn(),
  setSetting: vi.fn(),
  getSessions: vi.fn(),
  getSession: vi.fn(),
}));

const sessionManagerMock = vi.hoisted(() => ({
  touchSession: vi.fn(),
  startSession: vi.fn(),
  stopSession: vi.fn(),
  wakeSession: vi.fn(),
  preflightSessionStart: vi.fn(),
}));

vi.mock("../src/main/database", () => ({ db: dbMock }));
vi.mock("../src/main/sessions", () => ({ sessionManager: sessionManagerMock }));
vi.mock("../src/main/model-config-registry", () => ({
  detectModelConfigFromDir: vi.fn(() => ({ family: "hy-v3" })),
}));

interface BackendHandle {
  server: Server;
  port: number;
  bodies: any[];
  paths: string[];
}

interface AuthCaptureBackendHandle extends BackendHandle {
  authHeaders: Array<string | undefined>;
}

function listen(server: Server, port = 0): Promise<number> {
  return new Promise((resolve) => {
    server.listen(port, "127.0.0.1", () => {
      resolve((server.address() as AddressInfo).port);
    });
  });
}

function close(server: Server): Promise<void> {
  return new Promise((resolve) => server.close(() => resolve()));
}

async function freePort(): Promise<number> {
  const server = createServer();
  const port = await listen(server);
  await close(server);
  return port;
}

async function startCaptureBackend(): Promise<BackendHandle> {
  const bodies: any[] = [];
  const paths: string[] = [];
  const server = createServer((req, res) => {
    const chunks: Buffer[] = [];
    req.on("data", (chunk) => chunks.push(Buffer.from(chunk)));
    req.on("end", () => {
      paths.push(req.url || "");
      const raw = Buffer.concat(chunks).toString("utf8");
      bodies.push(raw ? JSON.parse(raw) : {});
      res.setHeader("Content-Type", "application/json");
      res.end(
        JSON.stringify({
          id: "chatcmpl-gateway-test",
          object: "chat.completion",
          choices: [
            {
              index: 0,
              message: { role: "assistant", content: "ok" },
              finish_reason: "stop",
            },
          ],
          usage: { prompt_tokens: 1, completion_tokens: 1, total_tokens: 2 },
        }),
      );
    });
  });
  return { server, port: await listen(server), bodies, paths };
}

async function startDetailErrorBackend(): Promise<BackendHandle> {
  const bodies: any[] = [];
  const paths: string[] = [];
  const server = createServer((req, res) => {
    const chunks: Buffer[] = [];
    req.on("data", (chunk) => chunks.push(Buffer.from(chunk)));
    req.on("end", () => {
      paths.push(req.url || "");
      const raw = Buffer.concat(chunks).toString("utf8");
      bodies.push(raw ? JSON.parse(raw) : {});
      res.writeHead(400, { "Content-Type": "application/json" });
      res.end(
        JSON.stringify({
          detail: "lfm2 does not expose a native thinking-off/instruct mode",
        }),
      );
    });
  });
  return { server, port: await listen(server), bodies, paths };
}

async function startStreamingChatBackend(): Promise<BackendHandle> {
  const bodies: any[] = [];
  const paths: string[] = [];
  const server = createServer((req, res) => {
    const chunks: Buffer[] = [];
    req.on("data", (chunk) => chunks.push(Buffer.from(chunk)));
    req.on("end", () => {
      paths.push(req.url || "");
      const raw = Buffer.concat(chunks).toString("utf8");
      bodies.push(raw ? JSON.parse(raw) : {});
      res.writeHead(200, {
        "Content-Type": "text/event-stream",
        "Cache-Control": "no-cache",
      });
      res.write(
        'data: {"choices":[{"delta":{"content":"hel"},"finish_reason":null}]}\n\n',
      );
      res.write(
        'data: {"choices":[{"delta":{"content":"lo"},"finish_reason":null}]}\n\n',
      );
      res.write(
        'data: {"choices":[{"delta":{},"finish_reason":"stop"}],"usage":{"prompt_tokens":2,"completion_tokens":2}}\n\n',
      );
      res.write("data: [DONE]\n\n");
      res.end();
    });
  });
  return { server, port: await listen(server), bodies, paths };
}

async function startStreamingThinkingBackend(rawCompletion = false): Promise<BackendHandle> {
  const bodies: any[] = [];
  const paths: string[] = [];
  const server = createServer((req, res) => {
    const chunks: Buffer[] = [];
    req.on("data", (chunk) => chunks.push(Buffer.from(chunk)));
    req.on("end", () => {
      paths.push(req.url || "");
      const raw = Buffer.concat(chunks).toString("utf8");
      bodies.push(raw ? JSON.parse(raw) : {});
      res.writeHead(200, {
        "Content-Type": "text/event-stream",
        "Cache-Control": "no-cache",
      });
      if (!rawCompletion) {
        res.write(
          'data: {"choices":[{"delta":{"reasoning_content":"plan "},"finish_reason":null}]}\n\n',
        );
        res.write(
          'data: {"choices":[{"delta":{"reasoning_content":"then act"},"finish_reason":null}]}\n\n',
        );
      }
      res.write(rawCompletion
        ? 'data: {"choices":[{"text":"answer","finish_reason":null}]}\n\n'
        : 'data: {"choices":[{"delta":{"content":"answer"},"finish_reason":null}]}\n\n',
      );
      res.write(
        'data: {"choices":[{"delta":{},"finish_reason":"stop"}]}\n\n',
      );
      res.write(
        'data: {"choices":[],"usage":{"prompt_tokens":3,"completion_tokens":4}}\n\n',
      );
      res.write("data: [DONE]\n\n");
      res.end();
    });
  });
  return { server, port: await listen(server), bodies, paths };
}

async function startReasoningOnlyErrorBackend(): Promise<BackendHandle> {
  const bodies: any[] = [];
  const paths: string[] = [];
  const server = createServer((req, res) => {
    const chunks: Buffer[] = [];
    req.on("data", (chunk) => chunks.push(Buffer.from(chunk)));
    req.on("end", () => {
      paths.push(req.url || "");
      const raw = Buffer.concat(chunks).toString("utf8");
      bodies.push(raw ? JSON.parse(raw) : {});
      res.writeHead(200, {
        "Content-Type": "text/event-stream",
        "Cache-Control": "no-cache",
      });
      res.write(
        'data: {"choices":[{"delta":{"reasoning_content":"private plan"},"finish_reason":null}]}\n\n',
      );
      res.write(
        'data: {"choices":[],"warnings":["The model ended normally while still in its reasoning phase."]}\n\n',
      );
      res.write(
        'data: {"error":{"message":"The model produced reasoning_content but no visible answer and no tool call.","type":"invalid_response_error","code":"reasoning_only_no_content"}}\n\n',
      );
      res.write("data: [DONE]\n\n");
      res.end();
    });
  });
  return { server, port: await listen(server), bodies, paths };
}

async function startEmbeddingBackend(): Promise<BackendHandle> {
  const bodies: any[] = [];
  const paths: string[] = [];
  const server = createServer((req, res) => {
    const chunks: Buffer[] = [];
    req.on("data", (chunk) => chunks.push(Buffer.from(chunk)));
    req.on("end", () => {
      paths.push(req.url || "");
      const raw = Buffer.concat(chunks).toString("utf8");
      bodies.push(raw ? JSON.parse(raw) : {});
      res.setHeader("Content-Type", "application/json");
      res.end(
        JSON.stringify({
          object: "list",
          data: [{ object: "embedding", index: 0, embedding: [0.1, 0.2, 0.3] }],
          model: "target-alias",
          usage: { prompt_tokens: 2, total_tokens: 2 },
        }),
      );
    });
  });
  return { server, port: await listen(server), bodies, paths };
}

async function startAuthCaptureBackend(): Promise<AuthCaptureBackendHandle> {
  const bodies: any[] = [];
  const paths: string[] = [];
  const authHeaders: Array<string | undefined> = [];
  const server = createServer((req, res) => {
    const chunks: Buffer[] = [];
    req.on("data", (chunk) => chunks.push(Buffer.from(chunk)));
    req.on("end", () => {
      paths.push(req.url || "");
      authHeaders.push(req.headers.authorization);
      const raw = Buffer.concat(chunks).toString("utf8");
      bodies.push(raw ? JSON.parse(raw) : {});
      res.setHeader("Content-Type", "application/json");
      if (req.url === "/v1/embeddings") {
        res.end(
          JSON.stringify({
            object: "list",
            data: [{ object: "embedding", index: 0, embedding: [0.1, 0.2] }],
            model: "hy3-model",
            usage: { prompt_tokens: 1, total_tokens: 1 },
          }),
        );
        return;
      }
      res.end(
        JSON.stringify({
          id: "chatcmpl-auth-capture",
          object: "chat.completion",
          choices: [
            {
              index: 0,
              message: { role: "assistant", content: "ok" },
              finish_reason: "stop",
            },
          ],
          usage: { prompt_tokens: 1, completion_tokens: 1, total_tokens: 2 },
        }),
      );
    });
  });
  return { server, port: await listen(server), bodies, paths, authHeaders };
}

async function startSlowStreamingChatBackend(): Promise<
  BackendHandle & { responseClosed: Promise<void> }
> {
  const bodies: any[] = [];
  const paths: string[] = [];
  let resolveClosed!: () => void;
  const responseClosed = new Promise<void>((resolve) => {
    resolveClosed = resolve;
  });
  const server = createServer((req, res) => {
    const chunks: Buffer[] = [];
    req.on("data", (chunk) => chunks.push(Buffer.from(chunk)));
    req.on("end", () => {
      paths.push(req.url || "");
      const raw = Buffer.concat(chunks).toString("utf8");
      bodies.push(raw ? JSON.parse(raw) : {});
      res.on("close", resolveClosed);
      res.writeHead(200, {
        "Content-Type": "text/event-stream",
        "Cache-Control": "no-cache",
      });
      res.write(
        'data: {"choices":[{"delta":{"content":"first"},"finish_reason":null}]}\n\n',
      );
      const interval = setInterval(() => {
        res.write(
          'data: {"choices":[{"delta":{"content":"later"},"finish_reason":null}]}\n\n',
        );
      }, 25);
      res.on("close", () => clearInterval(interval));
    });
  });
  return { server, port: await listen(server), bodies, paths, responseClosed };
}

async function startSlowNonStreamingChatBackend(): Promise<
  BackendHandle & { requestReceived: Promise<void>; responseClosed: Promise<void> }
> {
  const bodies: any[] = [];
  const paths: string[] = [];
  let resolveReceived!: () => void;
  let resolveClosed!: () => void;
  const requestReceived = new Promise<void>((resolve) => {
    resolveReceived = resolve;
  });
  const responseClosed = new Promise<void>((resolve) => {
    resolveClosed = resolve;
  });
  const server = createServer((req, res) => {
    const chunks: Buffer[] = [];
    req.on("data", (chunk) => chunks.push(Buffer.from(chunk)));
    req.on("end", () => {
      paths.push(req.url || "");
      const raw = Buffer.concat(chunks).toString("utf8");
      bodies.push(raw ? JSON.parse(raw) : {});
      res.on("close", resolveClosed);
      resolveReceived();
      // Deliberately do not send response headers. This models non-streaming
      // inference, where the backend response callback does not run until the
      // entire generation has completed.
    });
  });
  return {
    server,
    port: await listen(server),
    bodies,
    paths,
    requestReceived,
    responseClosed,
  };
}

async function startPrematureStreamingBackend(): Promise<BackendHandle> {
  const bodies: any[] = [];
  const paths: string[] = [];
  const server = createServer((req, res) => {
    const chunks: Buffer[] = [];
    req.on("data", (chunk) => chunks.push(Buffer.from(chunk)));
    req.on("end", () => {
      const path = req.url || "";
      paths.push(path);
      const raw = Buffer.concat(chunks).toString("utf8");
      bodies.push(raw ? JSON.parse(raw) : {});
      res.writeHead(200, {
        "Content-Type": "text/event-stream",
        "Cache-Control": "no-cache",
      });
      if (path === "/v1/responses") {
        res.write(
          'event: response.output_text.delta\ndata: {"type":"response.output_text.delta","delta":"partial"}\n\n',
        );
      } else if (path === "/v1/messages") {
        res.write(
          'event: content_block_delta\ndata: {"type":"content_block_delta","delta":{"type":"text_delta","text":"partial"}}\n\n',
        );
      } else {
        res.write(
          'data: {"choices":[{"delta":{"content":"partial"},"finish_reason":null}]}\n\n',
        );
      }
      setTimeout(() => res.socket?.destroy(), 10);
    });
  });
  return { server, port: await listen(server), bodies, paths };
}

async function startPrematureNonStreamingBackend(): Promise<BackendHandle> {
  const bodies: any[] = [];
  const paths: string[] = [];
  const attempts = new Map<string, number>();
  const server = createServer((req, res) => {
    const chunks: Buffer[] = [];
    req.on("data", (chunk) => chunks.push(Buffer.from(chunk)));
    req.on("end", () => {
      const path = req.url || "";
      paths.push(path);
      const raw = Buffer.concat(chunks).toString("utf8");
      bodies.push(raw ? JSON.parse(raw) : {});
      const attempt = (attempts.get(path) || 0) + 1;
      attempts.set(path, attempt);
      res.writeHead(200, { "Content-Type": "application/json" });
      if (attempt === 1) {
        res.write('{"partial":');
        setTimeout(() => res.socket?.destroy(), 10);
        return;
      }
      res.end(JSON.stringify({ object: "gateway.recovery", path, ok: true }));
    });
  });
  return { server, port: await listen(server), bodies, paths };
}

async function startGateway(sessionPort: number): Promise<{ gateway: any; port: number }> {
  const sessions = [
    {
      id: "hy3",
      modelPath: "/models/Hy3-preview-JANGTQ2",
      modelName: "hy3-model",
      host: "127.0.0.1",
      port: sessionPort,
      status: "running",
      type: "local",
      config: JSON.stringify({ servedModelName: "hy3-model" }),
      createdAt: Date.now(),
      updatedAt: Date.now(),
    },
  ];
  dbMock.getSetting.mockImplementation((key: string) =>
    key === "gateway_single_model_mode" ? "false" : undefined,
  );
  dbMock.getSessions.mockReturnValue(sessions);
  dbMock.getSession.mockImplementation((id: string) =>
    sessions.find((session) => session.id === id),
  );

  const { ApiGateway } = await import("../src/main/api-gateway");
  const gateway = new ApiGateway();
  const port = await freePort();
  await gateway.start(port, "127.0.0.1");
  return { gateway, port };
}

async function postJson(
  url: string,
  body: any,
  headers: Record<string, string> = { "Content-Type": "application/json" },
): Promise<any> {
  const response = await fetch(url, {
    method: "POST",
    headers,
    body: JSON.stringify(body),
  });
  expect(response.status).toBe(200);
  return response.json();
}

describe("Ollama gateway request translation behavior", () => {
  let backend: BackendHandle | undefined;
  let gateway: any | undefined;

  beforeEach(() => {
    vi.clearAllMocks();
    backend = undefined;
    gateway = undefined;
  });

  afterEach(async () => {
    if (gateway) await gateway.stop();
    if (backend) await close(backend.server);
  });

  it("preserves FastAPI detail messages and status for Ollama chat and generate", async () => {
    backend = await startDetailErrorBackend();
    const started = await startGateway(backend.port);
    gateway = started.gateway;

    const requests = [
      ["/api/chat", { model: "hy3-model", stream: false, messages: [] }],
      ["/api/chat", { model: "hy3-model", stream: true, messages: [] }],
      ["/api/generate", { model: "hy3-model", stream: false, prompt: "hi" }],
      ["/api/generate", { model: "hy3-model", stream: true, prompt: "hi" }],
    ] as const;

    for (const [route, body] of requests) {
      const response = await fetch(`http://127.0.0.1:${started.port}${route}`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(body),
      });
      expect(response.status).toBe(400);
      expect(await response.json()).toEqual({
        error: "lfm2 does not expose a native thinking-off/instruct mode",
      });
    }

    expect(backend.paths).toEqual([
      "/v1/chat/completions",
      "/v1/chat/completions",
      "/v1/chat/completions",
      "/v1/chat/completions",
    ]);
  });

  it("omits unset and negative sentinels while forwarding explicit neutral sampling overrides", async () => {
    backend = await startCaptureBackend();
    const started = await startGateway(backend.port);
    gateway = started.gateway;

    await postJson(`http://127.0.0.1:${started.port}/api/chat`, {
      model: "hy3-model",
      stream: false,
      messages: [{ role: "user", content: "hi" }],
    });
    await postJson(`http://127.0.0.1:${started.port}/api/chat`, {
      model: "hy3-model",
      stream: false,
      messages: [{ role: "user", content: "hi" }],
      options: { num_predict: -1, top_k: -1 },
    });
    await postJson(`http://127.0.0.1:${started.port}/api/generate`, {
      model: "hy3-model",
      stream: false,
      prompt: "hi",
      options: { num_predict: 0, top_k: 0 },
    });
    await postJson(`http://127.0.0.1:${started.port}/api/chat`, {
      model: "hy3-model",
      stream: false,
      messages: [{ role: "user", content: "hi" }],
      options: {
        num_predict: 12,
        temperature: 0.4,
        top_p: 0.82,
        top_k: 20,
        min_p: 0.03,
        repeat_penalty: 1.08,
      },
    });

    expect(backend.bodies[0]).not.toHaveProperty("max_tokens");
    expect(backend.bodies[0]).not.toHaveProperty("top_k");
    expect(backend.bodies[0]).not.toHaveProperty("temperature");
    expect(backend.bodies[0]).not.toHaveProperty("top_p");
    expect(backend.bodies[0]).not.toHaveProperty("min_p");
    expect(backend.bodies[0]).not.toHaveProperty("repetition_penalty");
    expect(backend.bodies[1]).not.toHaveProperty("max_tokens");
    expect(backend.bodies[1]).not.toHaveProperty("top_k");
    expect(backend.bodies[2]).not.toHaveProperty("max_tokens");
    expect(backend.bodies[2].top_k).toBe(0);
    expect(backend.bodies[3].max_tokens).toBe(12);
    expect(backend.bodies[3].temperature).toBe(0.4);
    expect(backend.bodies[3].top_p).toBe(0.82);
    expect(backend.bodies[3].top_k).toBe(20);
    expect(backend.bodies[3].min_p).toBe(0.03);
    expect(backend.bodies[3].repetition_penalty).toBe(1.08);
    expect(backend.paths).toEqual([
      "/v1/chat/completions",
      "/v1/chat/completions",
      "/v1/chat/completions",
      "/v1/chat/completions",
    ]);
  });

  it("forwards Ollama video and audio extensions as typed media content parts", async () => {
    backend = await startCaptureBackend();
    const started = await startGateway(backend.port);
    gateway = started.gateway;

    await postJson(`http://127.0.0.1:${started.port}/api/chat`, {
      model: "hy3-model",
      stream: false,
      messages: [{
        role: "user",
        content: "Inspect both.",
        videos: ["AAAAIGZ0eXA="],
        audio: "SUQz",
      }],
    });

    expect(backend.paths).toEqual(["/v1/chat/completions"]);
    expect(backend.bodies[0].messages).toEqual([{
      role: "user",
      content: [
        { type: "text", text: "Inspect both." },
        { type: "video_url", video_url: {
          url: "data:video/mp4;base64,AAAAIGZ0eXA=",
        } },
        { type: "audio_url", audio_url: {
          url: "data:audio/wav;base64,SUQz",
        } },
      ],
    }]);
  });

  it("forwards the top-level images array on /api/generate as media parts", async () => {
    // Ollama puts media at the TOP LEVEL of an /api/generate body, not inside
    // a message. The gateway built a plain string here, so a vision request
    // became text-only and the model answered about nothing, with no error —
    // the exact regression the Python route already fixed. /api/chat was
    // never affected.
    backend = await startCaptureBackend();
    const started = await startGateway(backend.port);
    gateway = started.gateway;

    await postJson(`http://127.0.0.1:${started.port}/api/generate`, {
      model: "vl-model",
      stream: false,
      prompt: "What colour is the left half?",
      images: ["iVBORw0KGgo="],
    });

    expect(backend.paths).toEqual(["/v1/chat/completions"]);
    expect(backend.bodies[0].messages).toEqual([{
      role: "user",
      content: [
        { type: "text", text: "What colour is the left half?" },
        { type: "image_url", image_url: {
          url: "data:image/png;base64,iVBORw0KGgo=",
        } },
      ],
    }]);
  });

  it("keeps /api/generate text-only prompts as plain strings", async () => {
    backend = await startCaptureBackend();
    const started = await startGateway(backend.port);
    gateway = started.gateway;

    await postJson(`http://127.0.0.1:${started.port}/api/generate`, {
      model: "text-model",
      stream: false,
      prompt: "Say OK.",
    });

    expect(backend.bodies[0].messages).toEqual([
      { role: "user", content: "Say OK." },
    ]);
  });

  it("forwards the Ollama seed on both chat and generate", async () => {
    // The Python route honours options.seed and top-level seed; the gateway
    // dropped both, so the identical request was reproducible against the
    // engine port and silently non-deterministic through the gateway.
    backend = await startCaptureBackend();
    const started = await startGateway(backend.port);
    gateway = started.gateway;

    await postJson(`http://127.0.0.1:${started.port}/api/chat`, {
      model: "text-model",
      stream: false,
      messages: [{ role: "user", content: "Say OK." }],
      options: { seed: 4242 },
    });
    await postJson(`http://127.0.0.1:${started.port}/api/generate`, {
      model: "text-model",
      stream: false,
      prompt: "Say OK.",
      seed: 99,
    });

    expect(backend.bodies[0].seed).toBe(4242);
    expect(backend.bodies[1].seed).toBe(99);
  });

  it("omits the seed entirely when the client did not send one", async () => {
    backend = await startCaptureBackend();
    const started = await startGateway(backend.port);
    gateway = started.gateway;

    await postJson(`http://127.0.0.1:${started.port}/api/chat`, {
      model: "text-model",
      stream: false,
      messages: [{ role: "user", content: "Say OK." }],
    });

    expect("seed" in backend.bodies[0]).toBe(false);
  });

  it("normalizes prior Ollama assistant thinking before text and media history forwarding", async () => {
    backend = await startCaptureBackend();
    const started = await startGateway(backend.port);
    gateway = started.gateway;

    await postJson(`http://127.0.0.1:${started.port}/api/chat`, {
      model: "hy3-model",
      stream: false,
      messages: [
        { role: "user", content: "first" },
        {
          role: "assistant",
          thinking: "PRIVATE-PLAN-TEXT",
          content: "visible text",
        },
        {
          role: "assistant",
          thinking: "PRIVATE-PLAN-MEDIA",
          content: "visible media",
          images: ["aW1hZ2U="],
        },
        {
          role: "assistant",
          thinking: "",
          content: "empty private rail",
        },
        {
          role: "assistant",
          thinking: "ALIAS-MUST-NOT-WIN",
          reasoning_content: "CANONICAL-PRIVATE-PLAN",
          content: "canonical wins",
        },
        { role: "user", content: "second" },
      ],
    });

    expect(backend.paths).toEqual(["/v1/chat/completions"]);
    expect(backend.bodies[0].messages).toEqual([
      { role: "user", content: "first" },
      {
        role: "assistant",
        reasoning_content: "PRIVATE-PLAN-TEXT",
        content: "visible text",
      },
      {
        role: "assistant",
        reasoning_content: "PRIVATE-PLAN-MEDIA",
        content: [
          { type: "text", text: "visible media" },
          {
            type: "image_url",
            image_url: { url: "data:image/png;base64,aW1hZ2U=" },
          },
        ],
      },
      {
        role: "assistant",
        content: "empty private rail",
      },
      {
        role: "assistant",
        reasoning_content: "CANONICAL-PRIVATE-PLAN",
        content: "canonical wins",
      },
      { role: "user", content: "second" },
    ]);
    expect(JSON.stringify(backend.bodies[0].messages)).not.toContain('"thinking"');
  });

  it("omits malformed Ollama num_predict values instead of poisoning max_tokens", async () => {
    backend = await startCaptureBackend();
    const started = await startGateway(backend.port);
    gateway = started.gateway;

    await postJson(`http://127.0.0.1:${started.port}/api/chat`, {
      model: "hy3-model",
      stream: false,
      messages: [{ role: "user", content: "bad" }],
      options: { num_predict: "not-a-number" },
    });
    await postJson(`http://127.0.0.1:${started.port}/api/generate`, {
      model: "hy3-model",
      stream: false,
      prompt: "bad",
      options: { num_predict: "Infinity" },
    });
    await postJson(`http://127.0.0.1:${started.port}/api/chat`, {
      model: "hy3-model",
      stream: false,
      messages: [{ role: "user", content: "decimal" }],
      options: { num_predict: 12.9 },
    });

    expect(backend.bodies[0]).not.toHaveProperty("max_tokens");
    expect(backend.bodies[1]).not.toHaveProperty("max_tokens");
    expect(backend.bodies[2].max_tokens).toBe(12);
  });

  it("omits malformed Ollama context values instead of poisoning max_prompt_tokens", async () => {
    backend = await startCaptureBackend();
    const started = await startGateway(backend.port);
    gateway = started.gateway;

    await postJson(`http://127.0.0.1:${started.port}/api/chat`, {
      model: "hy3-model",
      stream: false,
      messages: [{ role: "user", content: "bad context" }],
      options: { num_ctx: "not-a-number" },
    });
    await postJson(`http://127.0.0.1:${started.port}/api/generate`, {
      model: "hy3-model",
      stream: false,
      prompt: "bad context",
      options: { max_context_tokens: "Infinity" },
    });
    await postJson(`http://127.0.0.1:${started.port}/api/chat`, {
      model: "hy3-model",
      stream: false,
      messages: [{ role: "user", content: "decimal context" }],
      options: { num_ctx: 4096.9 },
    });

    expect(backend.bodies[0]).not.toHaveProperty("max_prompt_tokens");
    expect(backend.bodies[1]).not.toHaveProperty("max_prompt_tokens");
    expect(backend.bodies[2].max_prompt_tokens).toBe(4096);
  });

  it("does not coerce string false enable_thinking into reasoning on", async () => {
    backend = await startCaptureBackend();
    const started = await startGateway(backend.port);
    gateway = started.gateway;

    await postJson(`http://127.0.0.1:${started.port}/api/chat`, {
      model: "hy3-model",
      stream: false,
      messages: [{ role: "user", content: "off" }],
      enable_thinking: "false",
      reasoning_effort: "high",
    });
    await postJson(`http://127.0.0.1:${started.port}/api/generate`, {
      model: "hy3-model",
      stream: false,
      prompt: "off",
      enable_thinking: "false",
      reasoning_effort: "high",
    });

    expect(backend.bodies[0].enable_thinking).toBe(false);
    expect(backend.bodies[0]).not.toHaveProperty("reasoning_effort");
    expect(backend.bodies[1].enable_thinking).toBe(false);
    expect(backend.bodies[1]).not.toHaveProperty("reasoning_effort");
  });

  it("forwards bearer auth through translated Ollama chat, generate, and embeddings routes", async () => {
    const authBackend = await startAuthCaptureBackend();
    backend = authBackend;
    const started = await startGateway(backend.port);
    gateway = started.gateway;
    const headers = {
      "Content-Type": "application/json",
      Authorization: "Bearer gateway-session-key",
    };

    await postJson(`http://127.0.0.1:${started.port}/api/chat`, {
      model: "hy3-model",
      stream: false,
      messages: [{ role: "user", content: "hi" }],
    }, headers);
    await postJson(`http://127.0.0.1:${started.port}/api/generate`, {
      model: "hy3-model",
      stream: false,
      prompt: "hi",
    }, headers);
    await postJson(`http://127.0.0.1:${started.port}/api/embeddings`, {
      model: "hy3-model",
      input: "hi",
    }, headers);

    expect(authBackend.paths).toEqual([
      "/v1/chat/completions",
      "/v1/chat/completions",
      "/v1/embeddings",
    ]);
    expect(authBackend.authHeaders).toEqual([
      "Bearer gateway-session-key",
      "Bearer gateway-session-key",
      "Bearer gateway-session-key",
    ]);
  });

  it("requires configured session bearer before gateway-owned model lists and Ollama routing", async () => {
    const authBackend = await startAuthCaptureBackend();
    backend = authBackend;
    const sessions = [
      {
        id: "secure",
        modelPath: "/models/Secure-JANG",
        modelName: "secure-model",
        host: "127.0.0.1",
        port: backend.port,
        status: "running",
        type: "local",
        config: JSON.stringify({
          servedModelName: "secure-model",
          apiKey: "gateway-session-key",
        }),
        createdAt: Date.now(),
        updatedAt: Date.now(),
      },
    ];
    dbMock.getSetting.mockImplementation((key: string) =>
      key === "gateway_single_model_mode" ? "false" : undefined,
    );
    dbMock.getSessions.mockReturnValue(sessions);
    dbMock.getSession.mockImplementation((id: string) =>
      sessions.find((session) => session.id === id),
    );

    const { ApiGateway } = await import("../src/main/api-gateway");
    gateway = new ApiGateway();
    const port = await freePort();
    await gateway.start(port, "127.0.0.1");

    const getStatus = async (path: string, authorization?: string) => {
      const response = await fetch(`http://127.0.0.1:${port}${path}`, {
        headers: authorization ? { Authorization: authorization } : {},
      });
      return response.status;
    };

    expect(await getStatus("/v1/models")).toBe(401);
    expect(await getStatus("/v1/models", "Bearer wrong-key")).toBe(401);
    expect(await getStatus("/v1/models", "Bearer gateway-session-key")).toBe(200);
    expect(await getStatus("/api/tags")).toBe(401);
    expect(await getStatus("/api/tags", "Bearer wrong-key")).toBe(401);
    expect(await getStatus("/api/tags", "Bearer gateway-session-key")).toBe(200);
    expect(await getStatus("/api/ps", "Bearer wrong-key")).toBe(401);
    expect(await getStatus("/api/ps", "Bearer gateway-session-key")).toBe(200);

    const wrongChat = await fetch(`http://127.0.0.1:${port}/api/chat`, {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
        Authorization: "Bearer wrong-key",
      },
      body: JSON.stringify({
        model: "secure-model",
        stream: false,
        messages: [{ role: "user", content: "hi" }],
      }),
    });
    expect(wrongChat.status).toBe(401);
    expect(authBackend.paths).toEqual([]);

    const okChat = await fetch(`http://127.0.0.1:${port}/api/chat`, {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
        Authorization: "Bearer gateway-session-key",
      },
      body: JSON.stringify({
        model: "secure-model",
        stream: false,
        messages: [{ role: "user", content: "hi" }],
      }),
    });
    expect(okChat.status).toBe(200);
    expect(authBackend.paths).toEqual(["/v1/chat/completions"]);
    expect(authBackend.authHeaders).toEqual(["Bearer gateway-session-key"]);
  });

  it("auto-switches by model id in single-model mode before preserving streaming deltas", async () => {
    backend = await startStreamingChatBackend();
    const sessions = [
      {
        id: "target",
        modelPath: "/models/Target-JANG",
        modelName: "target-model",
        host: "127.0.0.1",
        port: backend.port,
        status: "stopped",
        type: "local",
        config: JSON.stringify({ servedModelName: "target-alias" }),
        createdAt: Date.now(),
        updatedAt: Date.now(),
      },
      {
        id: "other",
        modelPath: "/models/Other-JANG",
        modelName: "other-model",
        host: "127.0.0.1",
        port: await freePort(),
        status: "running",
        type: "local",
        config: JSON.stringify({ servedModelName: "other-alias" }),
        createdAt: Date.now(),
        updatedAt: Date.now(),
      },
    ];
    dbMock.getSetting.mockImplementation((key: string) =>
      key === "gateway_single_model_mode" ? "true" : undefined,
    );
    dbMock.getSessions.mockReturnValue(sessions);
    dbMock.getSession.mockImplementation((id: string) =>
      sessions.find((session) => session.id === id),
    );
    sessionManagerMock.stopSession.mockImplementation(async (id: string) => {
      const session = sessions.find((item) => item.id === id);
      if (session) session.status = "stopped";
    });
    sessionManagerMock.startSession.mockImplementation(async (id: string) => {
      const session = sessions.find((item) => item.id === id);
      if (session) session.status = "running";
    });

    const { ApiGateway } = await import("../src/main/api-gateway");
    gateway = new ApiGateway();
    const port = await freePort();
    await gateway.start(port, "127.0.0.1");

    const response = await fetch(`http://127.0.0.1:${port}/api/chat`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        model: "target-alias",
        stream: true,
        messages: [{ role: "user", content: "hi" }],
      }),
    });
    const text = await response.text();

    expect(response.status).toBe(200);
    expect(sessionManagerMock.stopSession).toHaveBeenCalledWith("other");
    expect(sessionManagerMock.startSession).toHaveBeenCalledWith("target", {
      launchOrigin: "gateway",
    });
    expect(sessionManagerMock.touchSession).toHaveBeenCalledWith("target");
    expect(backend.paths).toEqual(["/v1/chat/completions"]);
    expect(backend.bodies[0].model).toBe("target-alias");
    expect(text).toContain('"content":"hel"');
    expect(text).toContain('"content":"lo"');
    expect(text).toContain('"done":true');
  });

  it("auto-switches single-model Ollama chat while emitting incremental content chunks", async () => {
    backend = await startStreamingChatBackend();
    const sessions = [
      {
        id: "target",
        modelPath: "/models/Target-JANG",
        modelName: "target-model",
        host: "127.0.0.1",
        port: backend.port,
        status: "stopped",
        type: "local",
        config: JSON.stringify({ servedModelName: "target-alias" }),
        createdAt: Date.now(),
        updatedAt: Date.now(),
      },
      {
        id: "other",
        modelPath: "/models/Other-JANG",
        modelName: "other-model",
        host: "127.0.0.1",
        port: await freePort(),
        status: "running",
        type: "local",
        config: JSON.stringify({ servedModelName: "other-alias" }),
        createdAt: Date.now(),
        updatedAt: Date.now(),
      },
    ];
    dbMock.getSetting.mockImplementation((key: string) =>
      key === "gateway_single_model_mode" ? "true" : undefined,
    );
    dbMock.getSessions.mockReturnValue(sessions);
    dbMock.getSession.mockImplementation((id: string) =>
      sessions.find((session) => session.id === id),
    );
    sessionManagerMock.stopSession.mockImplementation(async (id: string) => {
      const session = sessions.find((item) => item.id === id);
      if (session) session.status = "stopped";
    });
    sessionManagerMock.startSession.mockImplementation(async (id: string) => {
      const session = sessions.find((item) => item.id === id);
      if (session) session.status = "running";
    });

    const { ApiGateway } = await import("../src/main/api-gateway");
    gateway = new ApiGateway();
    const port = await freePort();
    await gateway.start(port, "127.0.0.1");

    const response = await fetch(`http://127.0.0.1:${port}/api/chat`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        model: "target-alias",
        stream: true,
        messages: [{ role: "user", content: "hi" }],
      }),
    });
    const text = await response.text();
    const chunks = text
      .trim()
      .split("\n")
      .filter(Boolean)
      .map((line) => JSON.parse(line));

    expect(response.status).toBe(200);
    expect(sessionManagerMock.stopSession).toHaveBeenCalledWith("other");
    expect(sessionManagerMock.startSession).toHaveBeenCalledWith("target", {
      launchOrigin: "gateway",
    });
    expect(backend.paths).toEqual(["/v1/chat/completions"]);
    expect(chunks.map((chunk) => chunk.message.content)).toEqual([
      "hel",
      "lo",
      "",
    ]);
    expect(chunks.map((chunk) => chunk.done)).toEqual([false, false, true]);
    expect(chunks[2].done_reason).toBe("stop");
    expect(chunks[2].eval_count).toBe(2);
    expect(chunks[2].prompt_eval_count).toBe(2);
  });

  it.each([false, true])("generate waits for usage after finish_reason (raw=%s)", async (raw) => {
    backend = await startStreamingThinkingBackend(raw);
    const started = await startGateway(backend.port);
    gateway = started.gateway;
    const response = await fetch(`http://127.0.0.1:${started.port}/api/generate`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ model: "hy3-model", stream: true, raw, prompt: "hi" }),
    });
    const chunks = (await response.text()).trim().split("\n").map(line => JSON.parse(line));
    expect(response.status).toBe(200);
    expect(backend.paths).toEqual([raw ? "/v1/completions" : "/v1/chat/completions"]);
    expect(chunks.map(chunk => chunk.response || "").join("")).toBe("answer");
    expect(chunks.map(chunk => chunk.thinking || "").join("")).toBe(raw ? "" : "plan then act");
    expect(chunks.filter(chunk => chunk.done)).toHaveLength(1);
    expect(chunks.at(-1)).toMatchObject({
      response: "", done: true, done_reason: "stop",
      eval_count: 4, prompt_eval_count: 3,
    });
    expect(chunks.at(-1)).not.toHaveProperty("thinking");
  });

  it("streams thinking exactly once and leaves the terminal message empty", async () => {
    backend = await startStreamingThinkingBackend();
    const started = await startGateway(backend.port);
    gateway = started.gateway;

    const response = await fetch(`http://127.0.0.1:${started.port}/api/chat`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        model: "hy3-model",
        stream: true,
        think: true,
        messages: [{ role: "user", content: "hi" }],
      }),
    });
    const chunks = (await response.text())
      .trim()
      .split("\n")
      .filter(Boolean)
      .map((line) => JSON.parse(line));

    expect(response.status).toBe(200);
    expect(backend.paths).toEqual(["/v1/chat/completions"]);
    expect(backend.bodies[0].enable_thinking).toBe(true);
    expect(chunks.map((chunk) => chunk.message.thinking || "").join(""))
      .toBe("plan then act");
    expect(chunks.map((chunk) => chunk.message.content || "").join(""))
      .toBe("answer");
    expect(chunks.at(-1)).toMatchObject({
      message: { role: "assistant", content: "" },
      done: true,
      done_reason: "stop",
      eval_count: 4,
      prompt_eval_count: 3,
    });
    expect(chunks.at(-1).message).not.toHaveProperty("thinking");
  });

  it("maps a reasoning-only Chat stream failure to one native Ollama error row", async () => {
    backend = await startReasoningOnlyErrorBackend();
    const started = await startGateway(backend.port);
    gateway = started.gateway;

    const response = await fetch(`http://127.0.0.1:${started.port}/api/chat`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        model: "hy3-model",
        stream: true,
        think: true,
        messages: [{ role: "user", content: "reason without an answer" }],
      }),
    });
    const chunks = (await response.text())
      .trim()
      .split("\n")
      .filter(Boolean)
      .map((line) => JSON.parse(line));

    expect(response.status).toBe(200);
    expect(chunks).toEqual([
      {
        model: "hy3-model",
        created_at: expect.any(String),
        message: {
          role: "assistant",
          content: "",
          thinking: "private plan",
        },
        done: false,
      },
      {
        error:
          "The model produced reasoning_content but no visible answer and no tool call.",
      },
    ]);
    expect(chunks.some((chunk) => chunk.done === true)).toBe(false);
  });

  it("maps a reasoning-only Generate stream failure to one native Ollama error row", async () => {
    backend = await startReasoningOnlyErrorBackend();
    const started = await startGateway(backend.port);
    gateway = started.gateway;

    const response = await fetch(
      `http://127.0.0.1:${started.port}/api/generate`,
      {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          model: "hy3-model",
          stream: true,
          think: true,
          prompt: "reason without an answer",
        }),
      },
    );
    const chunks = (await response.text())
      .trim()
      .split("\n")
      .filter(Boolean)
      .map((line) => JSON.parse(line));

    expect(response.status).toBe(200);
    expect(chunks).toEqual([
      {
        model: "hy3-model",
        created_at: expect.any(String),
        response: "",
        thinking: "private plan",
        done: false,
      },
      {
        error:
          "The model produced reasoning_content but no visible answer and no tool call.",
      },
    ]);
    expect(chunks.some((chunk) => chunk.done === true)).toBe(false);
  });

  it("auto-switches single-model Ollama generate while emitting incremental response chunks", async () => {
    backend = await startStreamingChatBackend();
    const sessions = [
      {
        id: "target",
        modelPath: "/models/Target-JANG",
        modelName: "target-model",
        host: "127.0.0.1",
        port: backend.port,
        status: "stopped",
        type: "local",
        config: JSON.stringify({ servedModelName: "target-alias" }),
        createdAt: Date.now(),
        updatedAt: Date.now(),
      },
      {
        id: "other",
        modelPath: "/models/Other-JANG",
        modelName: "other-model",
        host: "127.0.0.1",
        port: await freePort(),
        status: "running",
        type: "local",
        config: JSON.stringify({ servedModelName: "other-alias" }),
        createdAt: Date.now(),
        updatedAt: Date.now(),
      },
    ];
    dbMock.getSetting.mockImplementation((key: string) =>
      key === "gateway_single_model_mode" ? "true" : undefined,
    );
    dbMock.getSessions.mockReturnValue(sessions);
    dbMock.getSession.mockImplementation((id: string) =>
      sessions.find((session) => session.id === id),
    );
    sessionManagerMock.stopSession.mockImplementation(async (id: string) => {
      const session = sessions.find((item) => item.id === id);
      if (session) session.status = "stopped";
    });
    sessionManagerMock.startSession.mockImplementation(async (id: string) => {
      const session = sessions.find((item) => item.id === id);
      if (session) session.status = "running";
    });

    const { ApiGateway } = await import("../src/main/api-gateway");
    gateway = new ApiGateway();
    const port = await freePort();
    await gateway.start(port, "127.0.0.1");

    const response = await fetch(`http://127.0.0.1:${port}/api/generate`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        model: "target-alias",
        stream: true,
        prompt: "hi",
      }),
    });
    const text = await response.text();
    const chunks = text
      .trim()
      .split("\n")
      .filter(Boolean)
      .map((line) => JSON.parse(line));

    expect(response.status).toBe(200);
    expect(sessionManagerMock.stopSession).toHaveBeenCalledWith("other");
    expect(sessionManagerMock.startSession).toHaveBeenCalledWith("target", {
      launchOrigin: "gateway",
    });
    expect(sessionManagerMock.touchSession).toHaveBeenCalledWith("target");
    expect(backend.paths).toEqual(["/v1/chat/completions"]);
    expect(backend.bodies[0].model).toBe("target-alias");
    expect(backend.bodies[0].messages).toEqual([
      { role: "user", content: "hi" },
    ]);
    expect(chunks.map((chunk) => chunk.response)).toEqual(["hel", "lo", ""]);
    expect(chunks.map((chunk) => chunk.done)).toEqual([false, false, true]);
    expect(chunks[2].done_reason).toBe("stop");
    expect(chunks[2].eval_count).toBe(2);
    expect(chunks[2].prompt_eval_count).toBe(2);
  });

  it("auto-switches single-model Ollama embeddings before proxying embedding data", async () => {
    backend = await startEmbeddingBackend();
    const sessions = [
      {
        id: "target",
        modelPath: "/models/Target-JANG-Embedding",
        modelName: "target-model",
        host: "127.0.0.1",
        port: backend.port,
        status: "standby",
        type: "local",
        config: JSON.stringify({
          servedModelName: "target-alias",
          embeddingModel: "target-embed",
        }),
        createdAt: Date.now(),
        updatedAt: Date.now(),
      },
      {
        id: "other",
        modelPath: "/models/Other-JANG",
        modelName: "other-model",
        host: "127.0.0.1",
        port: await freePort(),
        status: "running",
        type: "local",
        config: JSON.stringify({ servedModelName: "other-alias" }),
        createdAt: Date.now(),
        updatedAt: Date.now(),
      },
    ];
    dbMock.getSetting.mockImplementation((key: string) =>
      key === "gateway_single_model_mode" ? "true" : undefined,
    );
    dbMock.getSessions.mockReturnValue(sessions);
    dbMock.getSession.mockImplementation((id: string) =>
      sessions.find((session) => session.id === id),
    );
    sessionManagerMock.stopSession.mockImplementation(async (id: string) => {
      const session = sessions.find((item) => item.id === id);
      if (session) session.status = "stopped";
    });
    sessionManagerMock.wakeSession.mockImplementation(async (id: string) => {
      const session = sessions.find((item) => item.id === id);
      if (session) session.status = "running";
      return { success: true };
    });

    const { ApiGateway } = await import("../src/main/api-gateway");
    gateway = new ApiGateway();
    const port = await freePort();
    await gateway.start(port, "127.0.0.1");

    const response = await fetch(`http://127.0.0.1:${port}/api/embeddings`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        model: "target-embed",
        input: "hello",
      }),
    });
    const body = await response.json();

    expect(response.status).toBe(200);
    expect(sessionManagerMock.stopSession).toHaveBeenCalledWith("other");
    expect(sessionManagerMock.wakeSession).toHaveBeenCalledWith("target");
    expect(sessionManagerMock.startSession).not.toHaveBeenCalled();
    expect(sessionManagerMock.touchSession).toHaveBeenCalledWith("target");
    expect(backend.paths).toEqual(["/v1/embeddings"]);
    expect(backend.bodies[0]).toEqual({
      model: "target-embed",
      input: "hello",
    });
    expect(body).toEqual({
      model: "target-embed",
      embeddings: [[0.1, 0.2, 0.3]],
      total_duration: 0,
    });
  });

  it("aborts Ollama backend streaming when the client disconnects mid-response", async () => {
    backend = await startSlowStreamingChatBackend();
    const started = await startGateway(backend.port);
    gateway = started.gateway;

    const controller = new AbortController();
    const response = await fetch(`http://127.0.0.1:${started.port}/api/chat`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        model: "hy3-model",
        stream: true,
        messages: [{ role: "user", content: "hi" }],
      }),
      signal: controller.signal,
    });
    const reader = response.body!.getReader();
    const first = await reader.read();
    expect(Buffer.from(first.value || []).toString("utf8")).toContain("first");

    controller.abort();

    await Promise.race([
      backend.responseClosed,
      new Promise((_, reject) =>
        setTimeout(() => reject(new Error("backend stream stayed open")), 250),
      ),
    ]);
  });

  it("aborts a generic non-stream backend before response headers when the client disconnects", async () => {
    backend = await startSlowNonStreamingChatBackend();
    const started = await startGateway(backend.port);
    gateway = started.gateway;

    const controller = new AbortController();
    const responsePromise = fetch(
      `http://127.0.0.1:${started.port}/v1/chat/completions`,
      {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          model: "hy3-model",
          stream: false,
          messages: [{ role: "user", content: "keep generating" }],
        }),
        signal: controller.signal,
      },
    );

    await backend.requestReceived;
    controller.abort();
    await expect(responsePromise).rejects.toMatchObject({ name: "AbortError" });

    await Promise.race([
      backend.responseClosed,
      new Promise((_, reject) =>
        setTimeout(() => reject(new Error("backend request stayed open")), 250),
      ),
    ]);
  });

  it("terminates premature backend streams with protocol-native failures and no false success", async () => {
    backend = await startPrematureStreamingBackend();
    const started = await startGateway(backend.port);
    gateway = started.gateway;

    const requests = [
      {
        path: "/v1/chat/completions",
        body: {
          model: "hy3-model",
          stream: true,
          messages: [{ role: "user", content: "chat" }],
        },
      },
      {
        path: "/v1/responses",
        body: { model: "hy3-model", stream: true, input: "responses" },
      },
      {
        path: "/v1/messages",
        body: {
          model: "hy3-model",
          stream: true,
          max_tokens: 32,
          messages: [{ role: "user", content: "anthropic" }],
        },
      },
      {
        path: "/api/chat",
        body: {
          model: "hy3-model",
          stream: true,
          messages: [{ role: "user", content: "ollama" }],
        },
      },
    ];

    const outputs: string[] = [];
    for (const request of requests) {
      const response = await fetch(`http://127.0.0.1:${started.port}${request.path}`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(request.body),
      });
      expect(response.status).toBe(200);
      outputs.push(await response.text());
    }

    expect(outputs[0]).toContain("partial");
    expect(outputs[0]).toContain('"code":"backend_connection_closed"');
    expect(outputs[0]).not.toContain("[DONE]");

    expect(outputs[1]).toContain("event: response.output_text.delta");
    expect(outputs[1]).toContain("event: response.failed");
    expect(outputs[1]).toContain('"status":"failed"');
    expect(outputs[1]).not.toContain("event: response.completed");

    expect(outputs[2]).toContain("event: content_block_delta");
    expect(outputs[2]).toContain("event: error");
    expect(outputs[2]).toContain('"type":"api_error"');
    expect(outputs[2]).not.toContain("event: message_stop");

    const ollamaLines = outputs[3]
      .trim()
      .split("\n")
      .filter(Boolean)
      .map((line) => JSON.parse(line));
    expect(ollamaLines[0]).toMatchObject({
      message: { content: "partial" },
      done: false,
    });
    expect(ollamaLines.at(-1)).toEqual({
      error: "Backend connection closed before response completed",
    });
    expect(ollamaLines.some((line) => line.done === true)).toBe(false);

    expect(backend.paths).toEqual([
      "/v1/chat/completions",
      "/v1/responses",
      "/v1/messages",
      "/v1/chat/completions",
    ]);
  });

  it("keeps non-stream JSON atomic across premature backend loss and immediate recovery", async () => {
    backend = await startPrematureNonStreamingBackend();
    const started = await startGateway(backend.port);
    gateway = started.gateway;

    const requests = [
      {
        path: "/v1/chat/completions",
        body: {
          model: "hy3-model",
          stream: false,
          messages: [{ role: "user", content: "chat" }],
        },
      },
      {
        path: "/v1/responses",
        body: { model: "hy3-model", stream: false, input: "responses" },
      },
      {
        path: "/v1/messages",
        body: {
          model: "hy3-model",
          stream: false,
          max_tokens: 32,
          messages: [{ role: "user", content: "anthropic" }],
        },
      },
    ];

    for (const request of requests) {
      const failed = await fetch(`http://127.0.0.1:${started.port}${request.path}`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(request.body),
      });
      expect(failed.status).toBe(502);
      expect(await failed.json()).toEqual({
        error: {
          message: "Backend connection closed before response completed",
          type: "server_error",
          code: "backend_connection_closed",
        },
      });

      const recovered = await fetch(
        `http://127.0.0.1:${started.port}${request.path}`,
        {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify(request.body),
        },
      );
      expect(recovered.status).toBe(200);
      expect(await recovered.json()).toEqual({
        object: "gateway.recovery",
        path: request.path,
        ok: true,
      });
    }
  });

  it("refuses single-model Ollama routes when previous local model cannot unload", async () => {
    backend = await startCaptureBackend();
    const sessions = [
      {
        id: "target",
        modelPath: "/models/Target-JANG",
        modelName: "target-model",
        host: "127.0.0.1",
        port: backend.port,
        status: "stopped",
        type: "local",
        config: JSON.stringify({ servedModelName: "target-alias" }),
        createdAt: Date.now(),
        updatedAt: Date.now(),
      },
      {
        id: "other",
        modelPath: "/models/Other-JANG",
        modelName: "other-model",
        host: "127.0.0.1",
        port: await freePort(),
        status: "running",
        type: "local",
        config: JSON.stringify({ servedModelName: "other-alias" }),
        createdAt: Date.now(),
        updatedAt: Date.now(),
      },
    ];
    dbMock.getSetting.mockImplementation((key: string) =>
      key === "gateway_single_model_mode" ? "true" : undefined,
    );
    dbMock.getSessions.mockReturnValue(sessions);
    dbMock.getSession.mockImplementation((id: string) =>
      sessions.find((session) => session.id === id),
    );
    sessionManagerMock.stopSession.mockRejectedValue(new Error("still running"));

    const { ApiGateway } = await import("../src/main/api-gateway");
    gateway = new ApiGateway();
    const port = await freePort();
    await gateway.start(port, "127.0.0.1");

    const routeBodies = [
      ["/api/chat", { model: "target-alias", stream: false, messages: [] }],
      ["/api/generate", { model: "target-alias", stream: false, prompt: "hi" }],
      ["/api/embeddings", { model: "target-alias", input: "hi" }],
    ] as const;

    for (const [route, requestBody] of routeBodies) {
      const response = await fetch(`http://127.0.0.1:${port}${route}`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(requestBody),
      });
      const body = await response.json();

      expect(response.status).toBe(503);
      expect(body.code).toBe("single_model_unload_failed");
    }

    expect(sessionManagerMock.stopSession).toHaveBeenCalledTimes(routeBodies.length);
    expect(sessionManagerMock.stopSession).toHaveBeenCalledWith("other");
    expect(sessionManagerMock.startSession).not.toHaveBeenCalled();
    expect(sessionManagerMock.wakeSession).not.toHaveBeenCalled();
    expect(sessionManagerMock.touchSession).not.toHaveBeenCalled();
    expect(backend.paths).toEqual([]);
    expect(backend.bodies).toEqual([]);
  });
});
