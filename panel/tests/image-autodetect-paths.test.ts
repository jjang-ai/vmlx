import { describe, expect, it } from "vitest";
import { readFileSync } from "fs";
import { join } from "path";

const MODELS_TS = join(__dirname, "..", "src", "main", "ipc", "models.ts");
const IMAGE_TS = join(__dirname, "..", "src", "main", "ipc", "image.ts");

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
});
