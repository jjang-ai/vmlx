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

function optionalJsonListEnv(name: string, fallback: string[]): string[] {
  const raw = process.env[name];
  if (!raw) return fallback;
  const parsed = JSON.parse(raw);
  if (!Array.isArray(parsed) || !parsed.every((item) => typeof item === "string")) {
    throw new Error(`${name} must be a JSON string array`);
  }
  return parsed;
}

function optionalJsonEnv(name: string): any | undefined {
  const raw = process.env[name];
  if (!raw) return undefined;
  return JSON.parse(raw);
}

function optionalEnv(name: string): string | undefined {
  const value = process.env[name];
  return value && value.length > 0 ? value : undefined;
}

function parseSsePayloads(raw: string): any[] {
  return raw
    .replace(/\r\n/g, "\n")
    .split("\n\n")
    .flatMap((block) => {
      const data = block
        .split("\n")
        .filter((line) => line.startsWith("data:"))
        .map((line) => line.replace(/^data:\s?/, ""))
        .join("\n");
      if (!data || data === "[DONE]") return [];
      try {
        return [JSON.parse(data)];
      } catch {
        return [];
      }
    });
}

function firstResponseId(raw: string): string | undefined {
  for (const payload of parseSsePayloads(raw)) {
    const responseId = payload?.response?.id ?? payload?.id;
    if (typeof responseId === "string" && responseId.length > 0) {
      return responseId;
    }
  }
  return undefined;
}

function firstFunctionCallId(raw: string): string | undefined {
  for (const payload of parseSsePayloads(raw)) {
    const item = payload?.item;
    if (item?.type === "function_call" && typeof item.call_id === "string") {
      return item.call_id;
    }
  }
  return undefined;
}

function substituteContinuationIds(body: any, raw: string): any {
  const encoded = JSON.stringify(body);
  const responseId = firstResponseId(raw);
  const callId = firstFunctionCallId(raw);
  return JSON.parse(
    encoded
      .replaceAll("$VMLINUX_RESPONSE_ID", responseId ?? "")
      .replaceAll("$VMLINUX_CALL_ID", callId ?? ""),
  );
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
    const continuationOutPath = optionalEnv(
      "VMLINUX_QWEN35_GATEWAY_CONTINUATION_OUT",
    );
    const continuationLogPath = optionalEnv(
      "VMLINUX_QWEN35_GATEWAY_CONTINUATION_LOG",
    );
    const requestBody = JSON.parse(
      requiredEnv("VMLINUX_QWEN35_GATEWAY_PAYLOAD_JSON"),
    );
    const continuationRequestBody = optionalJsonEnv(
      "VMLINUX_QWEN35_GATEWAY_CONTINUATION_PAYLOAD_JSON",
    );
    const expectedSubstrings = optionalJsonListEnv(
      "VMLINUX_QWEN35_GATEWAY_EXPECT_CONTAINS_JSON",
      [
        "response.reasoning_summary_text",
        "response.function_call_arguments.delta",
        "response.function_call_arguments.done",
        "blue-cat",
      ],
    );
    const sessions = [
      {
        id: "qwen35-live",
        modelPath,
        modelName:
          process.env.VMLINUX_QWEN35_GATEWAY_MODEL_NAME ||
          "Qwen3.6-35B-A3B-MXFP8-MTP",
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
    let continuationStatus: number | undefined;
    let continuationContentType: string | null | undefined;
    let continuationRaw = "";

    if (continuationRequestBody) {
      const resolvedContinuationRequestBody = substituteContinuationIds(
        continuationRequestBody,
        raw,
      );
      const continuationResponse = await fetch(
        `http://127.0.0.1:${gatewayPort}/v1/responses`,
        {
          method: "POST",
          headers: {
            "Content-Type": "application/json",
            Accept: "text/event-stream",
          },
          body: JSON.stringify(resolvedContinuationRequestBody),
        },
      );
      continuationStatus = continuationResponse.status;
      continuationContentType = continuationResponse.headers.get("content-type");
      continuationRaw = await continuationResponse.text();
      if (continuationOutPath) {
        mkdirSync(dirname(continuationOutPath), { recursive: true });
        writeFileSync(continuationOutPath, continuationRaw, "utf8");
      }
      if (continuationLogPath) {
        mkdirSync(dirname(continuationLogPath), { recursive: true });
        writeFileSync(
          continuationLogPath,
          JSON.stringify(
            {
              status: continuationStatus,
              contentType: continuationContentType,
              gatewayPort,
              backendPort,
              servedModel,
              modelPath,
              requestModel: resolvedContinuationRequestBody.model,
              requestKwargs: {
                stream: resolvedContinuationRequestBody.stream,
                max_output_tokens:
                  resolvedContinuationRequestBody.max_output_tokens,
                temperature: resolvedContinuationRequestBody.temperature,
                top_p: resolvedContinuationRequestBody.top_p,
                top_k: resolvedContinuationRequestBody.top_k,
                enable_thinking: resolvedContinuationRequestBody.enable_thinking,
                reasoning: resolvedContinuationRequestBody.reasoning,
                previous_response_id:
                  resolvedContinuationRequestBody.previous_response_id,
                input_count: Array.isArray(resolvedContinuationRequestBody.input)
                  ? resolvedContinuationRequestBody.input.length
                  : undefined,
                first_input_type: Array.isArray(resolvedContinuationRequestBody.input)
                  ? resolvedContinuationRequestBody.input[0]?.type
                  : undefined,
                first_call_id: Array.isArray(resolvedContinuationRequestBody.input)
                  ? resolvedContinuationRequestBody.input[0]?.call_id
                  : undefined,
              },
              touchedSessions: sessionManagerMock.touchSession.mock.calls,
              responseBytes: continuationRaw.length,
              containsOutputText: continuationRaw.includes(
                "response.output_text.delta",
              ),
              containsFunctionCall: continuationRaw.includes(
                "\"type\":\"function_call\"",
              ),
            },
            null,
            2,
          ) + "\n",
          "utf8",
        );
      }
    }

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
          requestKwargs: {
            stream: requestBody.stream,
            max_output_tokens: requestBody.max_output_tokens,
            temperature: requestBody.temperature,
            top_p: requestBody.top_p,
            top_k: requestBody.top_k,
            enable_thinking: requestBody.enable_thinking,
            reasoning: requestBody.reasoning,
            tool_choice: requestBody.tool_choice,
            tool_count: Array.isArray(requestBody.tools)
              ? requestBody.tools.length
              : 0,
            first_tool_name: Array.isArray(requestBody.tools)
              ? requestBody.tools[0]?.name
              : undefined,
          },
          touchedSessions: sessionManagerMock.touchSession.mock.calls,
          responseBytes: raw.length,
          containsReasoning: raw.includes("response.reasoning_summary_text"),
          containsFunctionDelta: raw.includes(
            "response.function_call_arguments.delta",
          ),
          containsFunctionDone: raw.includes(
            "response.function_call_arguments.done",
          ),
          continuation: continuationRequestBody
            ? {
                status: continuationStatus,
                contentType: continuationContentType,
                outPath: continuationOutPath,
                logPath: continuationLogPath,
                responseBytes: continuationRaw.length,
                containsOutputText: continuationRaw.includes(
                  "response.output_text.delta",
                ),
                containsFunctionCall: continuationRaw.includes(
                  "\"type\":\"function_call\"",
                ),
              }
            : undefined,
        },
        null,
        2,
      ) + "\n",
      "utf8",
    );

    expect(response.status).toBe(200);
    expect(response.headers.get("content-type")).toContain("text/event-stream");
    expect(sessionManagerMock.touchSession).toHaveBeenCalledWith("qwen35-live");
    for (const expected of expectedSubstrings) {
      expect(raw).toContain(expected);
    }
    if (continuationRequestBody) {
      expect(continuationStatus).toBe(200);
      expect(continuationContentType).toContain("text/event-stream");
      expect(continuationRaw).toContain("response.output_text.delta");
    }
  }, 300_000);
});
