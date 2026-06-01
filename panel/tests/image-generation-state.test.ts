import { describe, expect, it } from "vitest";
import { readFileSync } from "fs";
import { join } from "path";
import {
  beginImageGeneration,
  classifyImageGenerationError,
  clearImageGenerationAfterLocalAbort,
  finishImageGeneration,
  getImageGenerationStatus,
  markImageGenerationAbort,
  resetImageGenerationStateForTests,
} from "../src/main/ipc/imageGenerationState";
import { maskHasPaintedPixels } from "../src/renderer/src/components/image/MaskPainter";

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
const IMAGE_GENERATION_STATE_TS = join(
  __dirname,
  "..",
  "src",
  "main",
  "ipc",
  "imageGenerationState.ts",
);
const MASK_PAINTER_TSX = join(
  __dirname,
  "..",
  "src",
  "renderer",
  "src",
  "components",
  "image",
  "MaskPainter.tsx",
);
const PRELOAD_TS = join(__dirname, "..", "src", "preload", "index.ts");
const ENV_D_TS = join(__dirname, "..", "src", "env.d.ts");

describe("image generation in-flight state survives tab switches", () => {
  it("classifies cancel per request without clearing a newer generation", () => {
    resetImageGenerationStateForTests();

    const controllerA = new AbortController();
    beginImageGeneration("session-a", controllerA);
    markImageGenerationAbort(controllerA, "cancel");
    clearImageGenerationAfterLocalAbort(controllerA);

    const controllerB = new AbortController();
    beginImageGeneration("session-b", controllerB);

    expect(
      classifyImageGenerationError(
        new Error("ImageGenerationAborted"),
        controllerA,
      ),
    ).toBe("Image generation cancelled.");

    finishImageGeneration(controllerA);
    expect(getImageGenerationStatus()).toMatchObject({
      generating: true,
      sessionId: "session-b",
    });

    finishImageGeneration(controllerB);
    expect(getImageGenerationStatus()).toMatchObject({
      generating: false,
      sessionId: "session-b",
    });
  });

  it("main process status includes the active or last generation session id", () => {
    const src = readFileSync(IMAGE_TS, "utf-8");
    expect(src).toContain("beginImageGeneration(sessionId)");
    expect(src).toContain("getImageGenerationStatus()");
  });

  it("classifies wrapped image-server EPIPE disconnects", () => {
    resetImageGenerationStateForTests();

    const wrapped = Object.assign(new Error("request failed"), {
      reason: Object.assign(new Error("write EPIPE"), { code: "EPIPE" }),
    });

    expect(classifyImageGenerationError(wrapped)).toBe(
      "Image server connection lost. The model may have crashed, been stopped, or hit memory pressure. Check Logs and restart the image server.",
    );

    const aggregate = Object.assign(new Error("aggregate failed"), {
      errors: [Object.assign(new Error("write EPIPE"), { code: "EPIPE" })],
    });

    expect(classifyImageGenerationError(aggregate)).toBe(
      "Image server connection lost. The model may have crashed, been stopped, or hit memory pressure. Check Logs and restart the image server.",
    );
  });

  it("renderer polls in-flight image status until the detached generation finishes", () => {
    const src = readFileSync(IMAGE_TAB_TSX, "utf-8");
    expect(src).toContain("syncGenerationStatus");
    expect(src).toContain("window.api.image.isGenerating()");
    expect(src).toMatch(/setInterval\(\s*syncGenerationStatus,\s*1500\s*\)/);
    expect(src).toContain("loadGenerations(sessionIdToRefresh)");
    expect(src).toContain("loadSessions()");
  });

  it("renderer keeps canonical image model id separate from display basename after tab return", () => {
    const src = readFileSync(IMAGE_TAB_TSX, "utf-8");
    const topbar = readFileSync(
      join(
        __dirname,
        "..",
        "src",
        "renderer",
        "src",
        "components",
        "image",
        "ImageTopBar.tsx",
      ),
      "utf-8",
    );
    expect(src).toContain("resolveImageModelFromDirectoryName");
    expect(src).toContain("canonicalModelId");
    expect(src).toContain("selectedModelDisplayName");
    expect(topbar).toContain("displayModelName");
  });

  it("image requests disable connection reuse and normalize reset-like socket errors", () => {
    const src = readFileSync(IMAGE_TS, "utf-8");
    const stateSrc = readFileSync(IMAGE_GENERATION_STATE_TS, "utf-8");
    expect(src.match(/agent:\s*false/g)?.length).toBeGreaterThanOrEqual(2);
    expect(stateSrc).toContain("Image server connection lost");
    expect(stateSrc).toContain("socket hang up");
    expect(stateSrc).toContain("ECONNRESET");
  });

  it("image cancel is distinguished from server reset and sends backend cancel", () => {
    const src = readFileSync(IMAGE_TS, "utf-8");
    const stateSrc = readFileSync(IMAGE_GENERATION_STATE_TS, "utf-8");
    expect(src).toContain('markImageGenerationAbort(controller, "cancel")');
    expect(src).toContain("requestImageServerCancel()");
    expect(src).toContain("/v1/images/cancel");
    expect(stateSrc).toContain("Image generation cancelled.");
    expect(src).toContain("clearImageGenerationAfterLocalAbort");
    expect(stateSrc).not.toContain("aborted/i.test(msg)");
  });

  it("preload and renderer types expose image generation session ids", () => {
    const preload = readFileSync(PRELOAD_TS, "utf-8");
    const env = readFileSync(ENV_D_TS, "utf-8");
    expect(preload).toContain("sessionId: string | null");
    expect(env).toContain("sessionId: string | null");
  });

  it("fill/edit mask state is part of submit wiring and clears when source changes", () => {
    const src = readFileSync(IMAGE_TAB_TSX, "utf-8");
    expect(src).toContain("const handleSourceImageChange = useCallback");
    expect(src).toContain("setMaskBase64(null)");
    expect(src).toContain("onSourceImageChange={handleSourceImageChange}");
    expect(src).toMatch(
      /}\s*,\s*\[[^\]]*maskBase64[^\]]*\]\s*\)/s,
    );
  });

  it("painted mask detection rejects empty masks and accepts edited pixels", () => {
    const src = readFileSync(MASK_PAINTER_TSX, "utf-8");
    const empty = new Uint8ClampedArray([
      0, 0, 0, 255,
      0, 0, 0, 255,
    ]);
    const painted = new Uint8ClampedArray([
      0, 0, 0, 255,
      255, 255, 255, 255,
    ]);

    expect(maskHasPaintedPixels(empty)).toBe(false);
    expect(maskHasPaintedPixels(painted)).toBe(true);
    expect(src).toContain("maskHasPaintedPixels(maskData.data)");
    expect(src).toContain('t("image.mask.emptyError")');
    expect(src).toContain("}, [onConfirm, t])");
  });
});
