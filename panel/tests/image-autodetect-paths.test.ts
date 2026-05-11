import { describe, expect, it } from "vitest";
import { readFileSync } from "fs";
import { join } from "path";
import { isImageDownloadEventForActive } from "../src/renderer/src/components/image/imageDownloadEvents";

const MODELS_TS = join(__dirname, "..", "src", "main", "ipc", "models.ts");
const IMAGE_TS = join(__dirname, "..", "src", "main", "ipc", "image.ts");
const IMAGE_DOWNLOAD_EVENTS_TS = join(
  __dirname,
  "..",
  "src",
  "renderer",
  "src",
  "components",
  "image",
  "imageDownloadEvents.ts",
);
const CODE_SNIPPETS_TSX = join(
  __dirname,
  "..",
  "src",
  "renderer",
  "src",
  "components",
  "api",
  "CodeSnippets.tsx",
);
const API_DASHBOARD_TSX = join(
  __dirname,
  "..",
  "src",
  "renderer",
  "src",
  "components",
  "api",
  "ApiDashboard.tsx",
);

describe("image model autodetection path", () => {
  it("download availability check validates mflux dirs with the model encoder topology", () => {
    const src = readFileSync(MODELS_TS, "utf-8");
    expect(src).toContain("getImageModelEncoderType");
    expect(src).toMatch(/const encoderType\s*=\s*getImageModelEncoderType\(modelName\)/);
    expect(src).toContain("validateImageModelCompleteness(localPath, encoderType)");
    expect(src).toContain('key === "text_encoder_2" && encoderType === "single"');
  });

  it("completed image downloads are validated before being marked ready", () => {
    const src = readFileSync(MODELS_TS, "utf-8");
    const successBranch = src.slice(src.indexOf("} else if (code === 0) {"));
    expect(src).toContain("Download completed but model is incomplete");
    expect(src).toContain("validateImageModelCompleteness(");
    expect(src).toContain('emitToRenderer("models:downloadError"');
    expect(successBranch.indexOf("Download completed but model is incomplete")).toBeLessThan(
      successBranch.indexOf('emitToRenderer("models:downloadComplete"'),
    );
  });

  it("image download completion events carry canonical model identity", () => {
    const src = readFileSync(MODELS_TS, "utf-8");
    expect(src).toContain("imageModelName: job.imageModelName");
    expect(src).toContain("imageQuantize: job.imageQuantize");
    expect(src).toContain("imageModelName: modelName");
    expect(src).toContain("imageQuantize: quantize");
  });

  it("image picker ignores completion events for a different download job", () => {
    const src = readFileSync(
      join(
        __dirname,
        "..",
        "src",
        "renderer",
        "src",
        "components",
        "image",
        "ImageModelPicker.tsx",
      ),
      "utf-8",
    );
    const eventSrc = readFileSync(IMAGE_DOWNLOAD_EVENTS_TS, "utf-8");
    expect(src).toContain("activeDownload");
    expect(src).toContain("isActiveDownloadEvent");
    expect(eventSrc).toContain("data.jobId !== activeDownload.jobId");
    expect(eventSrc).toContain("data.imageModelName !== activeDownload.model");
    expect(eventSrc).toContain("Number(data.imageQuantize) !== activeDownload.quantize");
    expect(src).toContain("setActiveDownload(null)");
  });

  it("image download event matching is scoped to the active job identity", () => {
    const active = { jobId: "job-a", model: "schnell", quantize: 4 };

    expect(
      isImageDownloadEventForActive(
        { jobId: "job-a", imageModelName: "schnell", imageQuantize: 4 },
        active,
        "downloading",
      ),
    ).toBe(true);
    expect(
      isImageDownloadEventForActive(
        { jobId: "job-b", imageModelName: "schnell", imageQuantize: 4 },
        active,
        "downloading",
      ),
    ).toBe(false);
    expect(
      isImageDownloadEventForActive(
        { jobId: "job-a", imageModelName: "z-image-turbo", imageQuantize: 4 },
        active,
        "downloading",
      ),
    ).toBe(false);
    expect(isImageDownloadEventForActive({ jobId: "job-a" }, null, "idle")).toBe(false);
  });

  it("download availability check registers manually downloaded registry repos from disk", () => {
    const src = readFileSync(MODELS_TS, "utf-8");
    expect(src).toContain("already have HF repos under ~/.mlxstudio/models/image");
    expect(src).toContain("const repoName = repoId?.split(\"/\").pop()");
    expect(src).toContain("db.setImageModelPath(modelName, quantize, candidate, repoId || undefined)");
  });

  it("image startServer falls back to existing downloaded repo directories before failing", () => {
    const src = readFileSync(IMAGE_TS, "utf-8");
    expect(src).toContain("findDownloadedImageModelPath");
    expect(src).toContain("Registered existing downloaded image model");
    expect(src).toContain("resolveImageModelRepo(modelId, quantize)");
    expect(src).toContain("db.setImageModelPath(discovered.modelId, quantize || 0, discovered.localPath, discovered.repoId)");
  });

  it("image edit requests normalize painted mask data URLs before proxying", () => {
    const src = readFileSync(IMAGE_TS, "utf-8");
    expect(src).toContain("body.mask = maskBase64.replace");
    expect(src).toContain("data:image");
  });

  it("API quick-start snippets switch to image generation/edit endpoints for image sessions", () => {
    const snippets = readFileSync(CODE_SNIPPETS_TSX, "utf-8");
    const dashboard = readFileSync(API_DASHBOARD_TSX, "utf-8");

    expect(snippets).toContain("IMAGE_LANGS");
    expect(snippets).toContain("/v1/images/generations");
    expect(snippets).toContain("/v1/images/edits");
    expect(snippets).toContain("mask.png");
    expect(dashboard).toContain("firstModelIsImage");
    expect(dashboard).toContain("isEdit={firstModelType === \"image-edit\"}");
  });
});
