import { describe, expect, it } from "vitest";
import { readFileSync } from "fs";
import { join } from "path";

const IMAGE_TS = join(__dirname, "..", "src", "main", "ipc", "image.ts");
const IMAGE_TAB_TSX = join(
  __dirname,
  "..",
  "src",
  "renderer",
  "src",
  "components",
  "image",
  "ImageTab.tsx",
);
const PRELOAD_TS = join(__dirname, "..", "src", "preload", "index.ts");
const ENV_D_TS = join(__dirname, "..", "src", "env.d.ts");

describe("image generation in-flight state survives tab switches", () => {
  it("main process status includes the active or last generation session id", () => {
    const src = readFileSync(IMAGE_TS, "utf-8");
    expect(src).toContain("let activeGenerationSessionId: string | null = null");
    expect(src).toContain("let lastGenerationSessionId: string | null = null");
    expect(src).toMatch(/activeGenerationSessionId\s*=\s*sessionId/);
    expect(src).toMatch(/lastGenerationSessionId\s*=\s*sessionId/);
    expect(src).toMatch(/sessionId:\s*activeGenerationSessionId\s*\|\|\s*lastGenerationSessionId/);
  });

  it("renderer polls in-flight image status until the detached generation finishes", () => {
    const src = readFileSync(IMAGE_TAB_TSX, "utf-8");
    expect(src).toContain("syncGenerationStatus");
    expect(src).toContain("window.api.image.isGenerating()");
    expect(src).toMatch(/setInterval\(\s*syncGenerationStatus,\s*1500\s*\)/);
    expect(src).toContain("loadGenerations(sessionIdToRefresh)");
    expect(src).toContain("loadSessions()");
  });

  it("preload and renderer types expose image generation session ids", () => {
    const preload = readFileSync(PRELOAD_TS, "utf-8");
    const env = readFileSync(ENV_D_TS, "utf-8");
    expect(preload).toContain("sessionId: string | null");
    expect(env).toContain("sessionId: string | null");
  });
});
