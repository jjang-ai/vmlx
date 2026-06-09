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
  stopSession: vi.fn(),
  startSession: vi.fn(),
  wakeSession: vi.fn(),
  touchSession: vi.fn(),
}));

const gatewayBodyMock = vi.hoisted(() => ({
  extractGatewayModelFromBody: vi.fn(),
}));

vi.mock("../src/main/database", () => ({ db: dbMock }));
vi.mock("../src/main/sessions", () => ({ sessionManager: sessionManagerMock }));
vi.mock("../src/main/model-config-registry", () => ({
  detectModelConfigFromDir: vi.fn(),
}));
vi.mock("../src/main/gateway-body", () => gatewayBodyMock);

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

function deferred<T = void>(): {
  promise: Promise<T>;
  resolve: (value: T | PromiseLike<T>) => void;
  reject: (reason?: unknown) => void;
} {
  let resolve!: (value: T | PromiseLike<T>) => void;
  let reject!: (reason?: unknown) => void;
  const promise = new Promise<T>((res, rej) => {
    resolve = res;
    reject = rej;
  });
  return { promise, resolve, reject };
}

async function startOpenAiStreamingBackend(): Promise<BackendHandle> {
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
        'data: {"choices":[{"delta":{},"finish_reason":"stop"}]}\n\n',
      );
      res.write("data: [DONE]\n\n");
      res.end();
    });
  });
  return { server, port: await listen(server), bodies, paths };
}

async function startResponsesStreamingBackend(): Promise<BackendHandle> {
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
        'event: response.output_text.delta\ndata: {"type":"response.output_text.delta","delta":"hel"}\n\n',
      );
      res.write(
        'event: response.output_text.delta\ndata: {"type":"response.output_text.delta","delta":"lo"}\n\n',
      );
      res.write(
        'event: response.completed\ndata: {"type":"response.completed","response":{"status":"completed"}}\n\n',
      );
      res.end();
    });
  });
  return { server, port: await listen(server), bodies, paths };
}

async function startResponsesToolStreamingBackend(): Promise<BackendHandle> {
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
        `event: response.heartbeat
data: {"type":"response.heartbeat","tool_call_generating":true}

`,
      );
      res.write(
        `event: response.output_item.added
data: {"type":"response.output_item.added","output_index":1,"item":{"id":"fc_1","type":"function_call","status":"in_progress","call_id":"call_1","name":"lookup","arguments":""}}

`,
      );
      res.write(
        `event: response.function_call_arguments.delta
data: {"type":"response.function_call_arguments.delta","item_id":"fc_1","output_index":1,"delta":"{\\"query\\":\\"alpha\\""}

`,
      );
      res.write(
        `event: response.function_call_arguments.delta
data: {"type":"response.function_call_arguments.delta","item_id":"fc_1","output_index":1,"delta":",\\"limit\\":2}"}

`,
      );
      res.write(
        `event: response.function_call_arguments.done
data: {"type":"response.function_call_arguments.done","item_id":"fc_1","output_index":1,"arguments":"{\\"query\\":\\"alpha\\",\\"limit\\":2}"}

`,
      );
      res.write(
        `event: response.output_item.done
data: {"type":"response.output_item.done","output_index":1,"item":{"id":"fc_1","type":"function_call","status":"completed","call_id":"call_1","name":"lookup","arguments":"{\\"query\\":\\"alpha\\",\\"limit\\":2}"}}

`,
      );
      res.write(
        `event: response.completed
data: {"type":"response.completed","response":{"status":"completed","output":[]}}

`,
      );
      res.end();
    });
  });
  return { server, port: await listen(server), bodies, paths };
}

describe("ApiGateway single-model mode behavior", () => {
  let gateway: any | undefined;
  let backend: BackendHandle | undefined;

  beforeEach(() => {
    vi.clearAllMocks();
    gateway = undefined;
    backend = undefined;
    gatewayBodyMock.extractGatewayModelFromBody.mockImplementation((body: Buffer) => {
      try {
        return JSON.parse(body.toString("utf8") || "{}")?.model;
      } catch {
        return undefined;
      }
    });
  });

  afterEach(async () => {
    if (gateway) await gateway.stop();
    if (backend) await close(backend.server);
  });

  it("stops other active local sessions before routing to the requested model", async () => {
    dbMock.getSetting.mockImplementation((key: string) =>
      key === "gateway_single_model_mode" ? "true" : undefined,
    );
    dbMock.getSessions.mockReturnValue([
      { id: "target", status: "running", type: "local" },
      { id: "other-running", status: "running", type: "local" },
      { id: "other-loading", status: "loading", type: "local" },
      { id: "other-standby", status: "standby", type: "local" },
      { id: "remote-running", status: "running", type: "remote" },
      { id: "stopped-local", status: "stopped", type: "local" },
      { id: "error-local", status: "error", type: "local" },
    ]);

    const { ApiGateway } = await import("../src/main/api-gateway");
    const gateway = new ApiGateway();
    await (gateway as any).enforceSingleModelMode("target");

    expect(sessionManagerMock.stopSession.mock.calls.map((call) => call[0])).toEqual([
      "other-running",
      "other-loading",
      "other-standby",
    ]);
  });

  it("does nothing when single-model mode is disabled", async () => {
    dbMock.getSetting.mockReturnValue("false");
    dbMock.getSessions.mockReturnValue([
      { id: "other-running", status: "running", type: "local" },
    ]);

    const { ApiGateway } = await import("../src/main/api-gateway");
    const gateway = new ApiGateway();
    await (gateway as any).enforceSingleModelMode("target");

    expect(sessionManagerMock.stopSession).not.toHaveBeenCalled();
  });

  it("allows gateway startup on ports used only by stopped or remote saved sessions", async () => {
    const port = await freePort();
    dbMock.getSetting.mockReturnValue(undefined);
    dbMock.getSessions.mockReturnValue([
      {
        id: "stopped-local",
        modelPath: "/models/Stopped-JANG",
        modelName: "stopped-model",
        host: "127.0.0.1",
        port,
        status: "stopped",
        type: "local",
        config: "{}",
      },
      {
        id: "error-local",
        modelPath: "/models/Error-JANG",
        modelName: "error-model",
        host: "127.0.0.1",
        port,
        status: "error",
        type: "local",
        config: "{}",
      },
      {
        id: "remote",
        modelPath: "/remote/model",
        modelName: "remote-model",
        host: "api.example.com",
        port,
        status: "running",
        type: "remote",
        config: "{}",
      },
    ]);

    const { ApiGateway } = await import("../src/main/api-gateway");
    gateway = new ApiGateway();

    await expect(gateway.start(port, "127.0.0.1")).resolves.toBeUndefined();
    expect(dbMock.setSetting).toHaveBeenCalledWith("gateway_port", String(port));
  });

  it("reports failure when another active local session cannot be unloaded", async () => {
    dbMock.getSetting.mockImplementation((key: string) =>
      key === "gateway_single_model_mode" ? "true" : undefined,
    );
    dbMock.getSessions.mockReturnValue([
      { id: "target", status: "running", type: "local" },
      { id: "other-running", status: "running", type: "local" },
    ]);
    sessionManagerMock.stopSession.mockRejectedValueOnce(
      new Error("process refused to exit"),
    );

    const { ApiGateway } = await import("../src/main/api-gateway");
    const gateway = new ApiGateway();

    await expect((gateway as any).enforceSingleModelMode("target")).resolves.toBe(
      false,
    );
  });

  it("writes each streamed gateway response chunk once and treats EPIPE as disconnect", async () => {
    const { ApiGateway } = await import("../src/main/api-gateway");
    const gateway = new ApiGateway();
    const res = {
      destroyed: false,
      writableEnded: false,
      write: vi.fn(),
    };

    expect((gateway as any).writeResponse(res, "data: hello\n\n")).toBe(true);
    expect(res.write).toHaveBeenCalledTimes(1);
    expect(res.write).toHaveBeenCalledWith("data: hello\n\n");

    const disconnected = {
      destroyed: false,
      writableEnded: false,
      write: vi.fn(() => {
        const err = new Error("write EPIPE") as NodeJS.ErrnoException;
        err.code = "EPIPE";
        throw err;
      }),
    };
    expect((gateway as any).writeResponse(disconnected, "late")).toBe(false);

    const unexpected = {
      destroyed: false,
      writableEnded: false,
      write: vi.fn(() => {
        const err = new Error("disk full") as NodeJS.ErrnoException;
        err.code = "ENOSPC";
        throw err;
      }),
    };
    expect(() => (gateway as any).writeResponse(unexpected, "late")).toThrow(
      "disk full",
    );
  });

  it("does not write gateway response chunks after the client socket is destroyed", async () => {
    const { ApiGateway } = await import("../src/main/api-gateway");
    const gateway = new ApiGateway();
    const closedSocket = {
      destroyed: false,
      writableEnded: false,
      socket: { destroyed: true },
      write: vi.fn(() => {
        const err = new Error("write EPIPE") as NodeJS.ErrnoException;
        err.code = "EPIPE";
        throw err;
      }),
    };

    expect((gateway as any).writeResponse(closedSocket, "late")).toBe(false);
    expect(closedSocket.write).not.toHaveBeenCalled();
  });

  it("does not write gateway response chunks after Node marks the response closed", async () => {
    const { ApiGateway } = await import("../src/main/api-gateway");
    const gateway = new ApiGateway();
    const closedResponse = {
      closed: true,
      destroyed: false,
      writableEnded: false,
      writableDestroyed: false,
      socket: { destroyed: false },
      write: vi.fn(() => {
        const err = new Error("write EPIPE") as NodeJS.ErrnoException;
        err.code = "EPIPE";
        throw err;
      }),
    };

    expect((gateway as any).writeResponse(closedResponse, "late")).toBe(false);
    expect(closedResponse.write).not.toHaveBeenCalled();
  });

  it("does not write proxied request bodies after the backend socket is destroyed", async () => {
    const { ApiGateway } = await import("../src/main/api-gateway");
    const gateway = new ApiGateway();
    const closedBackend = {
      destroyed: false,
      writableEnded: false,
      socket: { destroyed: true },
      write: vi.fn(() => {
        const err = new Error("write EPIPE") as NodeJS.ErrnoException;
        err.code = "EPIPE";
        throw err;
      }),
    };

    expect((gateway as any).writeProxyBody(closedBackend, "late")).toBe(false);
    expect(closedBackend.write).not.toHaveBeenCalled();
  });

  it("does not write proxied request bodies after Node marks the request closed", async () => {
    const { ApiGateway } = await import("../src/main/api-gateway");
    const gateway = new ApiGateway();
    const closedBackend = {
      closed: true,
      destroyed: false,
      writableEnded: false,
      writableDestroyed: false,
      socket: { destroyed: false },
      write: vi.fn(() => {
        const err = new Error("write EPIPE") as NodeJS.ErrnoException;
        err.code = "EPIPE";
        throw err;
      }),
    };

    expect((gateway as any).writeProxyBody(closedBackend, "late")).toBe(false);
    expect(closedBackend.write).not.toHaveBeenCalled();
  });

  it("treats top-level request handler EPIPE failures as client disconnects", async () => {
    const { ApiGateway } = await import("../src/main/api-gateway");
    const gateway = new ApiGateway();
    const errorSpy = vi.spyOn(console, "error").mockImplementation(() => {});
    const res = {
      headersSent: false,
      destroyed: true,
      writableEnded: false,
    };
    const err = new Error("write EPIPE") as NodeJS.ErrnoException;
    err.code = "EPIPE";

    expect((gateway as any).handleRequestError(err, res)).toBe(false);
    expect(errorSpy).not.toHaveBeenCalled();

    errorSpy.mockRestore();
  });

  it("treats nested broken-pipe stream errors as client disconnects", async () => {
    const { ApiGateway } = await import("../src/main/api-gateway");
    const gateway = new ApiGateway();

    const brokenPipe = new Error("Broken pipe") as NodeJS.ErrnoException;
    brokenPipe.code = "ERR_STREAM_WRITE_AFTER_END";
    const wrapped = Object.assign(new Error("request failed"), {
      cause: brokenPipe,
    });
    const prematureClose = new Error("Premature close");
    const aggregate = new AggregateError(
      [Object.assign(new Error("write EPIPE"), { code: "EPIPE" })],
      "all connection attempts failed",
    );

    expect((gateway as any).isClientDisconnectError(wrapped)).toBe(true);
    expect((gateway as any).isClientDisconnectError(prematureClose)).toBe(true);
    expect((gateway as any).isClientDisconnectError(aggregate)).toBe(true);
  });

  it("does not end proxied requests after the backend socket is destroyed", async () => {
    const { ApiGateway } = await import("../src/main/api-gateway");
    const gateway = new ApiGateway();
    const closedBackend = {
      destroyed: false,
      writableEnded: false,
      socket: { destroyed: true },
      end: vi.fn(() => {
        const err = new Error("write EPIPE") as NodeJS.ErrnoException;
        err.code = "EPIPE";
        throw err;
      }),
    };

    expect((gateway as any).endProxyRequest(closedBackend)).toBe(false);
    expect(closedBackend.end).not.toHaveBeenCalled();
  });

  it("does not end proxied requests after Node marks the request closed", async () => {
    const { ApiGateway } = await import("../src/main/api-gateway");
    const gateway = new ApiGateway();
    const closedBackend = {
      closed: true,
      destroyed: false,
      writableEnded: false,
      writableDestroyed: false,
      socket: { destroyed: false },
      end: vi.fn(() => {
        const err = new Error("write EPIPE") as NodeJS.ErrnoException;
        err.code = "EPIPE";
        throw err;
      }),
    };

    expect((gateway as any).endProxyRequest(closedBackend)).toBe(false);
    expect(closedBackend.end).not.toHaveBeenCalled();
  });

  it("refuses auto-switch when previous local model cannot unload before starting target", async () => {
    backend = await startOpenAiStreamingBackend();
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
    sessionManagerMock.stopSession.mockRejectedValueOnce(
      new Error("process refused to exit"),
    );

    const { ApiGateway } = await import("../src/main/api-gateway");
    gateway = new ApiGateway();
    const port = await freePort();
    await gateway.start(port, "127.0.0.1");

    const response = await fetch(`http://127.0.0.1:${port}/v1/chat/completions`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        model: "target-alias",
        stream: true,
        messages: [{ role: "user", content: "hi" }],
      }),
    });
    const body = await response.json();

    expect(response.status).toBe(503);
    expect(body.error.code).toBe("single_model_unload_failed");
    expect(sessionManagerMock.stopSession).toHaveBeenCalledWith("other");
    expect(sessionManagerMock.startSession).not.toHaveBeenCalled();
    expect(sessionManagerMock.wakeSession).not.toHaveBeenCalled();
    expect(sessionManagerMock.touchSession).not.toHaveBeenCalled();
    expect(backend.paths).toEqual([]);
    expect(backend.bodies).toEqual([]);
  });

  it("auto-switches direct OpenAI streaming by model id without mutating payload or deltas", async () => {
    backend = await startOpenAiStreamingBackend();
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

    const requestBody = {
      model: "target-alias",
      stream: true,
      messages: [{ role: "user", content: "hi" }],
      temperature: 0.4,
      top_p: 0.82,
      repetition_penalty: 1.08,
      max_tokens: 96,
    };
    const response = await fetch(`http://127.0.0.1:${port}/v1/chat/completions`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(requestBody),
    });
    const text = await response.text();

    expect(response.status).toBe(200);
    expect(sessionManagerMock.stopSession).toHaveBeenCalledWith("other");
    expect(sessionManagerMock.startSession).toHaveBeenCalledWith("target");
    expect(sessionManagerMock.touchSession).toHaveBeenCalledWith("target");
    expect(backend.paths).toEqual(["/v1/chat/completions"]);
    expect(backend.bodies[0]).toEqual(requestBody);
    expect(text).toContain('"content":"hel"');
    expect(text).toContain('"content":"lo"');
    expect(text).toContain("data: [DONE]");
  });

  it("serializes concurrent single-model switches before starting a second target", async () => {
    backend = await startOpenAiStreamingBackend();
    const sessions = [
      {
        id: "target-a",
        modelPath: "/models/Target-A-JANG",
        modelName: "target-a-model",
        host: "127.0.0.1",
        port: backend.port,
        status: "stopped",
        type: "local",
        config: JSON.stringify({ servedModelName: "target-a" }),
        createdAt: Date.now(),
        updatedAt: Date.now(),
      },
      {
        id: "target-b",
        modelPath: "/models/Target-B-JANG",
        modelName: "target-b-model",
        host: "127.0.0.1",
        port: backend.port,
        status: "stopped",
        type: "local",
        config: JSON.stringify({ servedModelName: "target-b" }),
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

    const events: string[] = [];
    const targetAStartEntered = deferred();
    const releaseTargetAStart = deferred();
    sessionManagerMock.stopSession.mockImplementation(async (id: string) => {
      events.push(`stop:${id}`);
      const session = sessions.find((item) => item.id === id);
      if (session) session.status = "stopped";
    });
    sessionManagerMock.startSession.mockImplementation(async (id: string) => {
      events.push(`start:${id}`);
      if (id === "target-a") {
        targetAStartEntered.resolve();
        await releaseTargetAStart.promise;
      }
      const session = sessions.find((item) => item.id === id);
      if (session) session.status = "running";
      events.push(`running:${id}`);
    });

    const { ApiGateway } = await import("../src/main/api-gateway");
    gateway = new ApiGateway();
    const port = await freePort();
    await gateway.start(port, "127.0.0.1");

    const first = fetch(`http://127.0.0.1:${port}/v1/chat/completions`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        model: "target-a",
        stream: true,
        messages: [{ role: "user", content: "first" }],
      }),
    });
    await targetAStartEntered.promise;

    const second = fetch(`http://127.0.0.1:${port}/v1/chat/completions`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        model: "target-b",
        stream: true,
        messages: [{ role: "user", content: "second" }],
      }),
    });

    await new Promise((resolve) => setTimeout(resolve, 25));
    expect(events).not.toContain("start:target-b");

    releaseTargetAStart.resolve();
    const [firstResponse, secondResponse] = await Promise.all([first, second]);
    expect(firstResponse.status).toBe(200);
    expect(secondResponse.status).toBe(200);

    expect(events.indexOf("running:target-a")).toBeLessThan(
      events.indexOf("start:target-b"),
    );
    expect(events).toContain("stop:target-a");
    expect(sessionManagerMock.touchSession).toHaveBeenCalledWith("target-a");
    expect(sessionManagerMock.touchSession).toHaveBeenCalledWith("target-b");
  });

  it("auto-switches to a standby model by waking it before direct OpenAI streaming", async () => {
    backend = await startOpenAiStreamingBackend();
    const sessions = [
      {
        id: "target",
        modelPath: "/models/Target-JANG",
        modelName: "target-model",
        host: "127.0.0.1",
        port: backend.port,
        status: "standby",
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
    sessionManagerMock.wakeSession.mockImplementation(async (id: string) => {
      const session = sessions.find((item) => item.id === id);
      if (session) session.status = "running";
      return { success: true };
    });

    const { ApiGateway } = await import("../src/main/api-gateway");
    gateway = new ApiGateway();
    const port = await freePort();
    await gateway.start(port, "127.0.0.1");

    const response = await fetch(`http://127.0.0.1:${port}/v1/chat/completions`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        model: "target-alias",
        stream: true,
        messages: [{ role: "user", content: "wake" }],
      }),
    });
    const text = await response.text();

    expect(response.status).toBe(200);
    expect(sessionManagerMock.stopSession).toHaveBeenCalledWith("other");
    expect(sessionManagerMock.wakeSession).toHaveBeenCalledWith("target");
    expect(sessionManagerMock.startSession).not.toHaveBeenCalledWith("target");
    expect(sessionManagerMock.touchSession).toHaveBeenCalledWith("target");
    expect(backend.paths).toEqual(["/v1/chat/completions"]);
    expect(backend.bodies[0].model).toBe("target-alias");
    expect(text).toContain('"content":"hel"');
    expect(text).toContain('"content":"lo"');
    expect(text).toContain("data: [DONE]");
  });

  it("auto-switches Responses API streaming by model id while preserving output text deltas", async () => {
    backend = await startResponsesStreamingBackend();
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

    const requestBody = {
      model: "target-alias",
      stream: true,
      input: [{ role: "user", content: "hi" }],
      max_output_tokens: 96,
    };
    const response = await fetch(`http://127.0.0.1:${port}/v1/responses`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(requestBody),
    });
    const text = await response.text();

    expect(response.status).toBe(200);
    expect(sessionManagerMock.stopSession).toHaveBeenCalledWith("other");
    expect(sessionManagerMock.startSession).toHaveBeenCalledWith("target");
    expect(sessionManagerMock.touchSession).toHaveBeenCalledWith("target");
    expect(backend.paths).toEqual(["/v1/responses"]);
    expect(backend.bodies[0]).toEqual(requestBody);
    expect(text).toContain("response.output_text.delta");
    expect(text).toContain('"delta":"hel"');
    expect(text).toContain('"delta":"lo"');
    expect(text).toContain("response.completed");
  });

  it("passes Responses function-call argument SSE through unchanged", async () => {
    backend = await startResponsesToolStreamingBackend();
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

    const requestBody = {
      model: "target-alias",
      stream: true,
      input: [{ role: "user", content: "use lookup" }],
      tools: [
        {
          type: "function",
          name: "lookup",
          parameters: {
            type: "object",
            properties: {
              query: { type: "string" },
              limit: { type: "integer" },
            },
            required: ["query", "limit"],
          },
        },
      ],
    };
    const response = await fetch(`http://127.0.0.1:${port}/v1/responses`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(requestBody),
    });
    const text = await response.text();

    expect(response.status).toBe(200);
    expect(response.headers.get("content-type")).toContain("text/event-stream");
    expect(sessionManagerMock.stopSession).toHaveBeenCalledWith("other");
    expect(sessionManagerMock.startSession).toHaveBeenCalledWith("target");
    expect(sessionManagerMock.touchSession).toHaveBeenCalledWith("target");
    expect(backend.paths).toEqual(["/v1/responses"]);
    expect(backend.bodies[0]).toEqual(requestBody);
    expect(text).toContain("event: response.heartbeat");
    expect(text).toContain('"tool_call_generating":true');
    expect(text).toContain("event: response.function_call_arguments.delta");
    expect(text).toContain('"{\\"query\\":\\"alpha\\""');
    expect(text).toContain('",\\"limit\\":2}"');
    expect(text).toContain("event: response.function_call_arguments.done");
    expect(text).toContain("event: response.output_item.done");
    expect(text).toContain('"{\\"query\\":\\"alpha\\",\\"limit\\":2}"');
  });

  it("auto-switches model capability requests by path model before proxying", async () => {
    const paths: string[] = [];
    backend = {
      bodies: [],
      paths,
      server: createServer((req, res) => {
        paths.push(req.url || "");
        res.setHeader("Content-Type", "application/json");
        res.end(JSON.stringify({
          model: "target-alias",
          capabilities: ["completion", "tools", "thinking"],
        }));
      }),
      port: 0,
    };
    backend.port = await listen(backend.server);
    const sessions = [
      {
        id: "target",
        modelPath: "/models/Target-JANG",
        modelName: "target-model",
        host: "127.0.0.1",
        port: backend.port,
        status: "standby",
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
    sessionManagerMock.wakeSession.mockImplementation(async (id: string) => {
      const session = sessions.find((item) => item.id === id);
      if (session) session.status = "running";
      return { success: true };
    });

    const { ApiGateway } = await import("../src/main/api-gateway");
    gateway = new ApiGateway();
    const port = await freePort();
    await gateway.start(port, "127.0.0.1");

    const response = await fetch(
      `http://127.0.0.1:${port}/v1/models/target-alias/capabilities`,
    );
    const body = await response.json();

    expect(response.status).toBe(200);
    expect(body.capabilities).toContain("tools");
    expect(sessionManagerMock.stopSession).toHaveBeenCalledWith("other");
    expect(sessionManagerMock.wakeSession).toHaveBeenCalledWith("target");
    expect(sessionManagerMock.startSession).not.toHaveBeenCalledWith("target");
    expect(sessionManagerMock.touchSession).toHaveBeenCalledWith("target");
    expect(paths).toEqual(["/v1/models/target-alias/capabilities"]);
  });

  it("auto-switches cache endpoints by query model before proxying cache stats", async () => {
    const bodies: any[] = [];
    const paths: string[] = [];
    backend = {
      bodies,
      paths,
      server: createServer((req, res) => {
        paths.push(req.url || "");
        const chunks: Buffer[] = [];
        req.on("data", (chunk) => chunks.push(Buffer.from(chunk)));
        req.on("end", () => {
          const raw = Buffer.concat(chunks).toString("utf8");
          bodies.push(raw ? JSON.parse(raw) : {});
          res.setHeader("Content-Type", "application/json");
          res.end(JSON.stringify({ cache_type: "paged", hit_rate: 0.75 }));
        });
      }),
      port: 0,
    };
    backend.port = await listen(backend.server);
    const sessions = [
      {
        id: "target",
        modelPath: "/models/Target-JANG",
        modelName: "target-model",
        host: "127.0.0.1",
        port: backend.port,
        status: "standby",
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
    sessionManagerMock.wakeSession.mockImplementation(async (id: string) => {
      const session = sessions.find((item) => item.id === id);
      if (session) session.status = "running";
      return { success: true };
    });

    const { ApiGateway } = await import("../src/main/api-gateway");
    gateway = new ApiGateway();
    const port = await freePort();
    await gateway.start(port, "127.0.0.1");

    const response = await fetch(
      `http://127.0.0.1:${port}/v1/cache/stats?model=target-alias`,
    );
    const json = await response.json();

    expect(response.status).toBe(200);
    expect(json).toEqual({ cache_type: "paged", hit_rate: 0.75 });
    expect(sessionManagerMock.stopSession).toHaveBeenCalledWith("other");
    expect(sessionManagerMock.wakeSession).toHaveBeenCalledWith("target");
    expect(sessionManagerMock.startSession).not.toHaveBeenCalledWith("target");
    expect(sessionManagerMock.touchSession).toHaveBeenCalledWith("target");
    expect(backend.paths).toEqual(["/v1/cache/stats?model=target-alias"]);
    expect(backend.bodies).toEqual([{}]);
  });

  it("auto-switches cache entries and clear endpoints by query model before proxying cache endpoints", async () => {
    const bodies: any[] = [];
    const paths: string[] = [];
    const methods: string[] = [];
    backend = {
      bodies,
      paths,
      server: createServer((req, res) => {
        methods.push(req.method || "");
        paths.push(req.url || "");
        const chunks: Buffer[] = [];
        req.on("data", (chunk) => chunks.push(Buffer.from(chunk)));
        req.on("end", () => {
          const raw = Buffer.concat(chunks).toString("utf8");
          bodies.push(raw ? JSON.parse(raw) : {});
          res.setHeader("Content-Type", "application/json");
          if (req.method === "DELETE") {
            res.end(JSON.stringify({ status: "cleared", caches: ["prefix"] }));
            return;
          }
          res.end(JSON.stringify({ entries: [{ key: "prefix-a" }] }));
        });
      }),
      port: 0,
    };
    backend.port = await listen(backend.server);
    const sessions = [
      {
        id: "target",
        modelPath: "/models/Target-JANG",
        modelName: "target-model",
        host: "127.0.0.1",
        port: backend.port,
        status: "standby",
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
    sessionManagerMock.wakeSession.mockImplementation(async (id: string) => {
      const session = sessions.find((item) => item.id === id);
      if (session) session.status = "running";
      return { success: true };
    });

    const { ApiGateway } = await import("../src/main/api-gateway");
    gateway = new ApiGateway();
    const port = await freePort();
    await gateway.start(port, "127.0.0.1");

    const entries = await fetch(
      `http://127.0.0.1:${port}/v1/cache/entries?model=target-alias`,
    );
    const clear = await fetch(
      `http://127.0.0.1:${port}/v1/cache?model=target-alias&type=prefix`,
      { method: "DELETE" },
    );

    expect(entries.status).toBe(200);
    expect(await entries.json()).toEqual({ entries: [{ key: "prefix-a" }] });
    expect(clear.status).toBe(200);
    expect(await clear.json()).toEqual({ status: "cleared", caches: ["prefix"] });
    expect(sessionManagerMock.stopSession).toHaveBeenCalledWith("other");
    expect(sessionManagerMock.wakeSession).toHaveBeenCalledWith("target");
    expect(sessionManagerMock.startSession).not.toHaveBeenCalledWith("target");
    expect(sessionManagerMock.touchSession).toHaveBeenCalledWith("target");
    expect(methods).toEqual(["GET", "DELETE"]);
    expect(backend.paths).toEqual([
      "/v1/cache/entries?model=target-alias",
      "/v1/cache?model=target-alias&type=prefix",
    ]);
    expect(backend.bodies).toEqual([{}, {}]);
  });

  it("auto-switches cache warm by body model before proxying warm prompts", async () => {
    const bodies: any[] = [];
    const paths: string[] = [];
    const methods: string[] = [];
    backend = {
      bodies,
      paths,
      server: createServer((req, res) => {
        methods.push(req.method || "");
        paths.push(req.url || "");
        const chunks: Buffer[] = [];
        req.on("data", (chunk) => chunks.push(Buffer.from(chunk)));
        req.on("end", () => {
          const raw = Buffer.concat(chunks).toString("utf8");
          bodies.push(raw ? JSON.parse(raw) : {});
          res.setHeader("Content-Type", "application/json");
          res.end(JSON.stringify({ warmed: 1, token_counts: [4], errors: null }));
        });
      }),
      port: 0,
    };
    backend.port = await listen(backend.server);
    const sessions = [
      {
        id: "target",
        modelPath: "/models/Target-JANG",
        modelName: "target-model",
        host: "127.0.0.1",
        port: backend.port,
        status: "standby",
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
    sessionManagerMock.wakeSession.mockImplementation(async (id: string) => {
      const session = sessions.find((item) => item.id === id);
      if (session) session.status = "running";
      return { success: true };
    });

    const { ApiGateway } = await import("../src/main/api-gateway");
    gateway = new ApiGateway();
    const port = await freePort();
    await gateway.start(port, "127.0.0.1");

    const body = {
      model: "target-alias",
      prompts: ["system prompt"],
    };
    const response = await fetch(`http://127.0.0.1:${port}/v1/cache/warm`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body),
    });
    const json = await response.json();

    expect(response.status).toBe(200);
    expect(json).toEqual({ warmed: 1, token_counts: [4], errors: null });
    expect(sessionManagerMock.stopSession).toHaveBeenCalledWith("other");
    expect(sessionManagerMock.wakeSession).toHaveBeenCalledWith("target");
    expect(sessionManagerMock.startSession).not.toHaveBeenCalledWith("target");
    expect(sessionManagerMock.touchSession).toHaveBeenCalledWith("target");
    expect(methods).toEqual(["POST"]);
    expect(backend.paths).toEqual(["/v1/cache/warm"]);
    expect(backend.bodies).toEqual([body]);
  });
});
