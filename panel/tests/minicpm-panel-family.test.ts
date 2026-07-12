import { afterEach, describe, expect, it } from "vitest";
import { mkdtempSync, readFileSync, rmSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join, resolve } from "node:path";

import { detectModelConfigFromDir } from "../src/main/model-config-registry";

const createdDirs: string[] = [];

function makeModelDir(
  config: Record<string, unknown>,
  jangConfig?: Record<string, unknown>,
): string {
  const dir = mkdtempSync(join(tmpdir(), "vmlx-minicpm-panel-"));
  createdDirs.push(dir);
  writeFileSync(join(dir, "config.json"), JSON.stringify(config, null, 2));
  if (jangConfig !== undefined) {
    writeFileSync(
      join(dir, "jang_config.json"),
      JSON.stringify(jangConfig, null, 2),
    );
  }
  return dir;
}

afterEach(() => {
  while (createdDirs.length > 0) {
    const dir = createdDirs.pop();
    if (dir) rmSync(dir, { recursive: true, force: true });
  }
});

describe("MiniCPM plain-text panel family detection", () => {
  it("detects explicit text model_type without media, parser, or paged claims", () => {
    const detected = detectModelConfigFromDir(
      makeModelDir({ model_type: "minicpm" }),
    );

    expect(detected).toMatchObject({
      family: "minicpm",
      cacheType: "kv",
      usePagedCache: false,
      enableAutoToolChoice: false,
      isMultimodal: false,
    });
    expect(detected.toolParser).toBeUndefined();
    expect(detected.reasoningParser).toBeUndefined();
    expect(detected.nativeMtp).toBeUndefined();
  });

  it("detects the official legacy source signature when model_type is omitted", () => {
    const detected = detectModelConfigFromDir(
      makeModelDir({
        architectures: ["MiniCPMForCausalLM"],
        scale_emb: 12,
        scale_depth: 1.4,
        dim_model_base: 256,
      }),
    );

    expect(detected.family).toBe("minicpm");
    expect(detected.isMultimodal).toBe(false);
  });

  it("accepts a JANG capabilities family stamp for text MiniCPM", () => {
    const detected = detectModelConfigFromDir(
      makeModelDir(
        { model_type: "unregistered_wrapper" },
        {
          format: "jang",
          capabilities: {
            family: "minicpm",
            modality: "text",
            cache_type: "kv",
          },
        },
      ),
    );

    expect(detected.family).toBe("minicpm");
    expect(detected.isMultimodal).toBe(false);
  });

  it("preserves JANG family precedence over a legacy-looking source signature", () => {
    const detected = detectModelConfigFromDir(
      makeModelDir(
        {
          architectures: ["MiniCPMForCausalLM"],
          scale_emb: 12,
          scale_depth: 1.4,
          dim_model_base: 256,
        },
        {
          format: "jang",
          capabilities: { family: "qwen3", modality: "text" },
        },
      ),
    );

    expect(detected.family).toBe("qwen3");
  });

  it("does not classify incomplete or explicitly unrelated legacy signatures", () => {
    const incomplete = detectModelConfigFromDir(
      makeModelDir({
        architectures: ["MiniCPMForCausalLM"],
        scale_emb: 12,
        scale_depth: 1.4,
      }),
    );
    const explicitOther = detectModelConfigFromDir(
      makeModelDir({
        model_type: "llama",
        architectures: ["MiniCPMForCausalLM"],
        scale_emb: 12,
        scale_depth: 1.4,
        dim_model_base: 256,
      }),
    );
    const unregisteredExplicitOther = detectModelConfigFromDir(
      makeModelDir({
        model_type: "custom_minicpm",
        architectures: ["MiniCPMForCausalLM"],
        scale_emb: 12,
        scale_depth: 1.4,
        dim_model_base: 256,
      }),
    );
    const whitespaceWrappedMiniCpm = detectModelConfigFromDir(
      makeModelDir({
        model_type: " minicpm ",
        architectures: ["MiniCPMForCausalLM"],
        scale_emb: 12,
        scale_depth: 1.4,
        dim_model_base: 256,
      }),
    );

    expect(incomplete.family).toBe("unknown");
    expect(explicitOther.family).toBe("llama3");
    expect(unregisteredExplicitOther.family).toBe("unknown");
    expect(whitespaceWrappedMiniCpm.family).toBe("unknown");
  });

  it("rejects conflicting media metadata on a legacy text signature", () => {
    const detected = detectModelConfigFromDir(
      makeModelDir({
        architectures: ["MiniCPMForCausalLM"],
        scale_emb: 12,
        scale_depth: 1.4,
        dim_model_base: 256,
        vision_config: {},
      }),
    );

    expect(detected.family).toBe("unknown");
    expect(detected.isMultimodal).toBe(true);
  });

  it("keeps MiniCPM-V separate and multimodal", () => {
    const detected = detectModelConfigFromDir(
      makeModelDir({
        model_type: "minicpmv",
        vision_config: { model_type: "siglip_vision_model" },
      }),
    );
    const omittedTypeVisionArchitecture = detectModelConfigFromDir(
      makeModelDir({
        architectures: ["MiniCPMVForCausalLM"],
        scale_emb: 12,
        scale_depth: 1.4,
        dim_model_base: 256,
        vision_config: { model_type: "siglip_vision_model" },
      }),
    );

    expect(detected.family).toBe("minicpm-v");
    expect(detected.isMultimodal).toBe(true);
    expect(omittedTypeVisionArchitecture.family).toBe("unknown");
  });

  it("offers the backend text family as a manual override", () => {
    const formSource = readFileSync(
      resolve(
        __dirname,
        "../src/renderer/src/components/sessions/SessionConfigForm.tsx",
      ),
      "utf8",
    );

    expect(formSource).toMatch(
      /MODEL_FAMILY_OVERRIDE_NAMES[\s\S]*['"]minicpm['"]/,
    );
  });
});
