import { beforeEach, describe, expect, it, vi } from "vitest";

const dbMock = vi.hoisted(() => ({
  getSetting: vi.fn(),
  setSetting: vi.fn(),
  getSessions: vi.fn(),
}));

const sessionManagerMock = vi.hoisted(() => ({
  stopSession: vi.fn(),
}));

vi.mock("../src/main/database", () => ({ db: dbMock }));
vi.mock("../src/main/sessions", () => ({ sessionManager: sessionManagerMock }));
vi.mock("../src/main/model-config-registry", () => ({
  detectModelConfigFromDir: vi.fn(),
}));
vi.mock("../src/main/gateway-body", () => ({
  extractGatewayModelFromBody: vi.fn(),
}));

describe("ApiGateway single-model mode behavior", () => {
  beforeEach(() => {
    vi.clearAllMocks();
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
});
