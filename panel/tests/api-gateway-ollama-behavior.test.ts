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

async function postJson(url: string, body: any): Promise<any> {
  const response = await fetch(url, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
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

  it("omits unset and disabled sampling sentinels without dropping explicit overrides", async () => {
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
    expect(backend.bodies[2]).not.toHaveProperty("top_k");
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
    expect(sessionManagerMock.startSession).toHaveBeenCalledWith("target");
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
    expect(sessionManagerMock.startSession).toHaveBeenCalledWith("target");
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
    expect(sessionManagerMock.startSession).toHaveBeenCalledWith("target");
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
