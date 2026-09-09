import { describe, expect, it, vi } from "vitest";
import { readFileSync } from "fs";
import { join } from "path";
import {
  beginImageGeneration,
  bindImageGenerationRequest,
  markImageGenerationServerStopping,
  wasImageGenerationCancelled,
  requestImageGenerationServerStop,
  classifyImageGenerationError,
  isImageRequestCancellationResponse,
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
  it('cancels only the frozen server request once and leaves cleanup to its HTTP completion', async () => {
    resetImageGenerationStateForTests()
    const controller = beginImageGeneration('history')
    const cancel = vi.fn(async () => {})
    bindImageGenerationRequest(controller, 'server', 'request', cancel)
    await requestImageGenerationServerStop('other-server')
    expect(cancel).not.toHaveBeenCalled()
    await Promise.all([requestImageGenerationServerStop('server'), requestImageGenerationServerStop('server')])
    expect(cancel).toHaveBeenCalledTimes(1)
    expect(getImageGenerationStatus()).toMatchObject({ generating: true, cancelling: true })
    finishImageGeneration(controller)
  })
  it('marks only the stopped server and retains busy state until its request settles', () => {
    resetImageGenerationStateForTests()
    expect(markImageGenerationServerStopping('old-server')).toBe(false)
    const old = beginImageGeneration('old-history')
    bindImageGenerationRequest(old, 'old-server', 'old-request')
    expect(markImageGenerationServerStopping('other-server')).toBe(false)
    expect(wasImageGenerationCancelled(old)).toBe(false)
    expect(markImageGenerationServerStopping('old-server')).toBe(true)
    expect(wasImageGenerationCancelled(old)).toBe(true)
    expect(old.signal.aborted).toBe(false)
    expect(getImageGenerationStatus()).toMatchObject({ generating: true, cancelling: true })
    const current = beginImageGeneration('new-history')
    bindImageGenerationRequest(current, 'new-server', 'new-request')
    finishImageGeneration(old)
    expect(markImageGenerationServerStopping('old-server')).toBe(false)
    expect(wasImageGenerationCancelled(current)).toBe(false)
    expect(getImageGenerationStatus()).toMatchObject({ generating: true, cancelling: false })
    finishImageGeneration(current)
  })
  it('recognizes only the typed cancellation for this exact request', () => {
    const body = JSON.stringify({ detail: { code: 'image_generation_cancelled', request_id: 'owned' } })
    expect(isImageRequestCancellationResponse(409, body, 'owned')).toBe(true)
    expect(isImageRequestCancellationResponse(409, body, 'other')).toBe(false)
    expect(isImageRequestCancellationResponse(500, body, 'owned')).toBe(false)
    expect(isImageRequestCancellationResponse(409, 'broken JSON', 'owned')).toBe(false)
    expect(isImageRequestCancellationResponse(409, JSON.stringify({detail:{code:'image_request_id_in_use',request_id:'owned'}}), 'owned')).toBe(false)
  })
  it("keeps cancellation busy until the original request finishes", () => {
    resetImageGenerationStateForTests();
    const controller = beginImageGeneration("cancel-owner");
    markImageGenerationAbort(controller, "cancel");
    expect(getImageGenerationStatus()).toMatchObject({ generating: true, cancelling: true });
    expect(controller.signal.aborted).toBe(false);
    finishImageGeneration(controller);
    expect(getImageGenerationStatus()).toMatchObject({ generating: false, cancelling: false });
  });
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
    expect(src).toContain("await requestImageServerCancel(controller)");
    const cancelHandler = src.slice(src.indexOf("ipcMain.handle('image:cancelGeneration'"), src.indexOf("ipcMain.handle('image:getRunningServer'"));
    expect(cancelHandler).not.toContain("controller.abort()");
    expect(cancelHandler).not.toContain("clearImageGenerationAfterLocalAbort(controller)");
    expect(src).toContain("body.request_id = clientJobId");
    expect(src).toContain("request_id: owner.requestId");
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
    expect(src).toContain("imageDrafts.update(draftSnapshot, { sourceImage: img, maskBase64: null })");
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
    expect(src).toContain('setMaskError("image.mask.emptyError")');
    expect(src).toContain("}, [onConfirm, imageLoaded])");
  });
});
