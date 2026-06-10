import { mkdirSync, writeFileSync } from "node:fs";
import { dirname } from "node:path";
import { createServer } from "node:http";
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
  detectModelConfigFromDir: vi.fn(() => ({ family: "qwen3" })),
}));
vi.mock("../src/main/gateway-body", () => gatewayBodyMock);

function requiredEnv(name: string): string {
  const value = process.env[name];
  if (!value) throw new Error(`${name} is required`);
  return value;
}

function freePort(): Promise<number> {
  return new Promise((resolve, reject) => {
    const server = createServer();
    server.listen(0, "127.0.0.1", () => {
      const port = (server.address() as AddressInfo).port;
      server.close(() => resolve(port));
    });
    server.on("error", reject);
  });
}

const maybeIt =
  process.env.VMLINUX_QWEN35_GATEWAY_LIVE_CAPTURE === "1" ? it : it.skip;

describe("Qwen35 live Responses capture through ApiGateway", () => {
  let gateway: any | undefined;

  beforeEach(() => {
    vi.clearAllMocks();
    gateway = undefined;
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
  });

  maybeIt("captures Qwen35 Responses tool SSE through the real gateway proxy", async () => {
    const backendPort = Number(requiredEnv("VMLINUX_QWEN35_GATEWAY_BACKEND_PORT"));
    const servedModel = requiredEnv("VMLINUX_QWEN35_GATEWAY_SERVED_MODEL");
    const modelPath = requiredEnv("VMLINUX_QWEN35_GATEWAY_MODEL_PATH");
    const outPath = requiredEnv("VMLINUX_QWEN35_GATEWAY_OUT");
    const logPath = requiredEnv("VMLINUX_QWEN35_GATEWAY_LOG");
    const requestBody = JSON.parse(
      requiredEnv("VMLINUX_QWEN35_GATEWAY_PAYLOAD_JSON"),
    );
    const sessions = [
      {
        id: "qwen35-live",
        modelPath,
        modelName: "Qwen3.6-35B-A3B-MXFP8-MTP",
        host: "127.0.0.1",
        port: backendPort,
        status: "running",
        type: "local",
        config: JSON.stringify({ servedModelName: servedModel }),
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
    sessionManagerMock.touchSession.mockResolvedValue(undefined);

    const { ApiGateway } = await import("../src/main/api-gateway");
    gateway = new ApiGateway();
    const gatewayPort = await freePort();
    await gateway.start(gatewayPort, "127.0.0.1");

    const response = await fetch(`http://127.0.0.1:${gatewayPort}/v1/responses`, {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
        Accept: "text/event-stream",
      },
      body: JSON.stringify(requestBody),
    });
    const raw = await response.text();

    mkdirSync(dirname(outPath), { recursive: true });
    mkdirSync(dirname(logPath), { recursive: true });
    writeFileSync(outPath, raw, "utf8");
    writeFileSync(
      logPath,
      JSON.stringify(
        {
          status: response.status,
          contentType: response.headers.get("content-type"),
          gatewayPort,
          backendPort,
          servedModel,
          modelPath,
          requestModel: requestBody.model,
          touchedSessions: sessionManagerMock.touchSession.mock.calls,
          responseBytes: raw.length,
          containsReasoning: raw.includes("response.reasoning_summary_text"),
          containsFunctionDelta: raw.includes(
            "response.function_call_arguments.delta",
          ),
          containsFunctionDone: raw.includes(
            "response.function_call_arguments.done",
          ),
        },
        null,
        2,
      ) + "\n",
      "utf8",
    );

    expect(response.status).toBe(200);
    expect(response.headers.get("content-type")).toContain("text/event-stream");
    expect(sessionManagerMock.touchSession).toHaveBeenCalledWith("qwen35-live");
    expect(raw).toContain("response.reasoning_summary_text");
    expect(raw).toContain("response.function_call_arguments.delta");
    expect(raw).toContain("response.function_call_arguments.done");
    expect(raw).toContain("blue-cat");
  });
});
