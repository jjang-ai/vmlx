import { describe, expect, it } from "vitest";
import { readFileSync } from "fs";
import { join } from "path";

const MODELS_TS = join(__dirname, "..", "src", "main", "ipc", "models.ts");
const IMAGE_TS = join(__dirname, "..", "src", "main", "ipc", "image.ts");
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

  it("image picker delegates download jobs to Models instead of maintaining a second download lifecycle", () => {
    const src = readFileSync(join(__dirname, "../src/renderer/src/components/image/ImageModelPicker.tsx"), "utf-8");
    expect(src).toContain("IMAGE_MODEL_DISCOVERY_NAVIGATION");
    expect(src).not.toContain("onDownloadComplete(");
    expect(src).not.toContain("onDownloadProgress(");
    expect(src).not.toContain("downloadImageModel(");
  });

  it("download availability check registers manually downloaded registry repos from disk", () => {
    const src = readFileSync(MODELS_TS, "utf-8");
    expect(src).toContain("already have HF repos under ~/.mlxstudio/models/image");
    expect(src).toContain("const repoName = repoId?.split(\"/\").pop()");
    expect(src).toContain("db.setImageModelPath(modelName, quantize, candidate, repoId || undefined)");
  });

  it("image model downloader cannot write bytecode into the signed app bundle", () => {
    const src = readFileSync(MODELS_TS, "utf-8");
    expect(src).toContain('PYTHONDONTWRITEBYTECODE: "1"');
    expect(src).toContain('PYTHONNOUSERSITE: "1"');
    expect(src).toContain("PYTHONPATH: undefined");
    expect(src).toContain(
      '["-B", "-s", "-u", "-c", script, job.repoId, downloadDir, hfEndpoint, repoSubfolder]',
    );
    expect(src).toContain('files = [f for f in files if f.rfilename.startswith(prefix)]');
    expect(src).toContain('const markerFile = join(job.modelDir, ".vmlx-downloading")');
  });

  it("image startServer falls back to existing downloaded repo directories before failing", () => {
    const src = readFileSync(IMAGE_TS, "utf-8");
    expect(src).toContain("findDownloadedImageModelPath");
    expect(src).toContain("Validated registered model directory");
    expect(src).toContain("resolveLocalImageModelDirectory(modelPath, effectiveQuantize)");
    expect(src).toContain("modelDef.id, effectiveQuantize, modelPath, discoveredRepoId");
    expect(src).toContain("resolveImageModelArtifact(modelId, quantize)");
    expect(src).toContain("join(base, repoName, artifact.subfolder)");
    expect(src).not.toContain("db.setImageModelPath(discovered.modelId, quantize || 0, discovered.localPath, discovered.repoId)");
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
