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
});
