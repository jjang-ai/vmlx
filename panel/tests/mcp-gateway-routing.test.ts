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
  detectModelConfigFromDir: vi.fn(),
}));

interface BackendHandle {
  server: Server;
  port: number;
  requests: string[];
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

async function startMcpBackend(model: string, toolName: string): Promise<BackendHandle> {
  const requests: string[] = [];
  const server = createServer((req, res) => {
    const chunks: Buffer[] = [];
    req.on("data", (chunk) => chunks.push(Buffer.from(chunk)));
    req.on("end", () => {
      requests.push(`${req.method} ${req.url}`);
      res.setHeader("Content-Type", "application/json");
      if (req.url?.startsWith("/v1/mcp/tools")) {
        res.end(JSON.stringify({
          model,
          tools: [{
            name: toolName,
            server: toolName.split("__")[0],
            description: `${model} echo`,
            parameters: { type: "object", properties: { text: { type: "string" } } },
            enabled: true,
            effective: true,
            source: "mcp",
            transport: "stdio",
            server_state: "connected",
            error: null,
          }],
          count: 1,
        }));
        return;
      }
      if (req.url?.startsWith("/v1/mcp/servers")) {
        res.end(JSON.stringify({
          model,
          servers: [{
            name: toolName.split("__")[0],
            state: "connected",
            transport: "stdio",
            tools_count: 1,
            enabled: true,
            configured: true,
            command_redacted: ".venv/bin/python",
            url_redacted: null,
            env_keys: [],
            header_keys: [],
            error: null,
          }],
        }));
        return;
      }
      if (req.url?.startsWith("/v1/mcp/execute")) {
        const parsed = JSON.parse(Buffer.concat(chunks).toString("utf8") || "{}");
        res.end(JSON.stringify({
          model,
          tool_name: parsed.tool_name,
          content: `${model}:${parsed.arguments?.text || ""}`,
          is_error: false,
          error_message: null,
        }));
        return;
      }
      res.statusCode = 404;
      res.end(JSON.stringify({ error: "not found" }));
    });
  });
  return { server, port: await listen(server), requests };
}

async function startGateway(sessions: any[]): Promise<{ gateway: any; port: number }> {
  dbMock.getSetting.mockReturnValue(undefined);
  dbMock.getSessions.mockReturnValue(sessions);
  dbMock.getSession.mockImplementation((id: string) => sessions.find((s) => s.id === id));

  const { ApiGateway } = await import("../src/main/api-gateway");
  const gateway = new ApiGateway();
  const port = await freePort();
  await gateway.start(port, "127.0.0.1");
  return { gateway, port };
}

function session(id: string, port: number, alias: string) {
  return {
    id,
    modelPath: `/models/${alias}`,
    modelName: alias,
    host: "127.0.0.1",
    port,
    status: "running",
    type: "local",
    config: JSON.stringify({ servedModelName: alias }),
    createdAt: Date.now(),
    updatedAt: Date.now(),
  };
}

describe("MCP gateway routing", () => {
  let backends: BackendHandle[] = [];
  let gateway: any | undefined;

  beforeEach(() => {
    vi.clearAllMocks();
    backends = [];
    gateway = undefined;
  });

  afterEach(async () => {
    if (gateway) await gateway.stop();
    await Promise.all(backends.map((backend) => close(backend.server)));
  });

  it("routes MCP list and execute calls by explicit model alias", async () => {
    const a = await startMcpBackend("alpha", "alpha_smoke__echo");
    const b = await startMcpBackend("beta", "beta_smoke__echo");
    backends.push(a, b);
    const started = await startGateway([
      session("a", a.port, "alpha-model"),
      session("b", b.port, "beta-model"),
    ]);
    gateway = started.gateway;

    const alphaTools = await fetch(`http://127.0.0.1:${started.port}/v1/mcp/tools?model=alpha-model`).then((r) => r.json());
    const betaTools = await fetch(`http://127.0.0.1:${started.port}/v1/mcp/tools?model=beta-model`).then((r) => r.json());
    const alphaServers = await fetch(`http://127.0.0.1:${started.port}/v1/mcp/servers?model=alpha-model`).then((r) => r.json());
    const betaServers = await fetch(`http://127.0.0.1:${started.port}/v1/mcp/servers?model=beta-model`).then((r) => r.json());
    const betaExecute = await fetch(`http://127.0.0.1:${started.port}/v1/mcp/execute`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        model: "beta-model",
        tool_name: "beta_smoke__echo",
        arguments: { text: "gateway-ok" },
      }),
    }).then((r) => r.json());

    expect(alphaTools.tools[0].name).toBe("alpha_smoke__echo");
    expect(betaTools.tools[0].name).toBe("beta_smoke__echo");
    expect(alphaServers.servers[0].name).toBe("alpha_smoke");
    expect(betaServers.servers[0].name).toBe("beta_smoke");
    expect(betaExecute.content).toBe("beta:gateway-ok");
    expect(a.requests).toContain("GET /v1/mcp/tools?model=alpha-model");
    expect(a.requests).toContain("GET /v1/mcp/servers?model=alpha-model");
    expect(b.requests).toContain("GET /v1/mcp/tools?model=beta-model");
    expect(b.requests).toContain("GET /v1/mcp/servers?model=beta-model");
    expect(b.requests).toContain("POST /v1/mcp/execute");
    expect(sessionManagerMock.touchSession).toHaveBeenCalledWith("a");
    expect(sessionManagerMock.touchSession).toHaveBeenCalledWith("b");
  });

  it("rejects ambiguous multi-session MCP requests without a model", async () => {
    const a = await startMcpBackend("alpha", "alpha_smoke__echo");
    const b = await startMcpBackend("beta", "beta_smoke__echo");
    backends.push(a, b);
    const started = await startGateway([
      session("a", a.port, "alpha-model"),
      session("b", b.port, "beta-model"),
    ]);
    gateway = started.gateway;

    const response = await fetch(`http://127.0.0.1:${started.port}/v1/mcp/tools`);
    const body = await response.json();
    const executeResponse = await fetch(`http://127.0.0.1:${started.port}/v1/mcp/execute`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        tool_name: "alpha_smoke__echo",
        arguments: { text: "must-not-route" },
      }),
    });
    const executeBody = await executeResponse.json();

    expect(response.status).toBe(400);
    expect(body.error.code).toBe("model_required");
    expect(executeResponse.status).toBe(400);
    expect(executeBody.error.code).toBe("model_required");
    expect(a.requests).toEqual([]);
    expect(b.requests).toEqual([]);
  });

  it("keeps direct single-session MCP gateway fallback compatible", async () => {
    const a = await startMcpBackend("alpha", "alpha_smoke__echo");
    backends.push(a);
    const started = await startGateway([session("a", a.port, "alpha-model")]);
    gateway = started.gateway;

    const tools = await fetch(`http://127.0.0.1:${started.port}/v1/mcp/tools`).then((r) => r.json());

    expect(tools.tools[0].name).toBe("alpha_smoke__echo");
    expect(a.requests).toContain("GET /v1/mcp/tools");
  });

  it.runIf(process.env.VMLX_MCP_GATEWAY_LIVE === "1")(
    "routes real MCP model sessions supplied by env",
    async () => {
      const zayaPort = Number(process.env.VMLX_MCP_ZAYA_PORT);
      const qwenPort = Number(process.env.VMLX_MCP_QWEN_PORT);
      expect(zayaPort).toBeGreaterThan(0);
      expect(qwenPort).toBeGreaterThan(0);

      const started = await startGateway([
        session("zaya", zayaPort, "zaya-mcp-smoke"),
        session("qwen", qwenPort, "qwen36-mcp-smoke"),
      ]);
      gateway = started.gateway;

      const zayaTools = await fetch(`http://127.0.0.1:${started.port}/v1/mcp/tools?model=zaya-mcp-smoke`).then((r) => r.json());
      const qwenTools = await fetch(`http://127.0.0.1:${started.port}/v1/mcp/tools?model=qwen36-mcp-smoke`).then((r) => r.json());
      const zayaTool = process.env.VMLX_MCP_ZAYA_TOOL || "smoke__echo";
      const qwenTool = process.env.VMLX_MCP_QWEN_TOOL || "smoke__echo";
      const qwenExec = await fetch(`http://127.0.0.1:${started.port}/v1/mcp/execute`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          model: "qwen36-mcp-smoke",
          tool_name: qwenTool,
          arguments: { text: "gateway-live-ok" },
        }),
      }).then((r) => r.json());

      expect(zayaTools.count).toBeGreaterThan(0);
      expect(qwenTools.count).toBeGreaterThan(0);
      expect(zayaTools.tools.map((tool: any) => tool.name)).toContain(zayaTool);
      expect(qwenTools.tools.map((tool: any) => tool.name)).toContain(qwenTool);
      expect(qwenExec.is_error).toBe(false);
      expect(String(qwenExec.content)).toContain("gateway-live-ok");
    },
  );

  it.runIf(process.env.VMLX_MCP_GATEWAY_DSV4 === "1")(
    "routes a real DSV4 MCP session supplied by env",
    async () => {
      const dsv4Port = Number(process.env.VMLX_MCP_DSV4_PORT);
      const dsv4Tool = process.env.VMLX_MCP_DSV4_TOOL || "smoke__echo";
      expect(dsv4Port).toBeGreaterThan(0);

      const started = await startGateway([
        session("dsv4", dsv4Port, "dsv4-mcp-smoke"),
      ]);
      gateway = started.gateway;

      const tools = await fetch(`http://127.0.0.1:${started.port}/v1/mcp/tools?model=dsv4-mcp-smoke`).then((r) => r.json());
      const executed = await fetch(`http://127.0.0.1:${started.port}/v1/mcp/execute`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          model: "dsv4-mcp-smoke",
          tool_name: dsv4Tool,
          arguments: { text: "dsv4-gateway-live-ok" },
        }),
      }).then((r) => r.json());

      expect(tools.tools.map((tool: any) => tool.name)).toContain(dsv4Tool);
      expect(executed.is_error).toBe(false);
      expect(String(executed.content)).toContain("dsv4-gateway-live-ok");
    },
  );

  it.runIf(process.env.VMLX_MCP_GATEWAY_HY3 === "1")(
    "routes a real Hy3 MCP session supplied by env",
    async () => {
      const hy3Port = Number(process.env.VMLX_MCP_HY3_PORT);
      const hy3Tool = process.env.VMLX_MCP_HY3_TOOL || "smoke__echo";
      expect(hy3Port).toBeGreaterThan(0);

      const started = await startGateway([
        session("hy3", hy3Port, "hy3-mcp-smoke"),
      ]);
      gateway = started.gateway;

      const tools = await fetch(`http://127.0.0.1:${started.port}/v1/mcp/tools?model=hy3-mcp-smoke`).then((r) => r.json());
      const executed = await fetch(`http://127.0.0.1:${started.port}/v1/mcp/execute`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          model: "hy3-mcp-smoke",
          tool_name: hy3Tool,
          arguments: { text: "hy3-gateway-live-ok" },
        }),
      }).then((r) => r.json());

      expect(tools.tools.map((tool: any) => tool.name)).toContain(hy3Tool);
      expect(executed.is_error).toBe(false);
      expect(String(executed.content)).toContain("hy3-gateway-live-ok");
    },
  );
});
