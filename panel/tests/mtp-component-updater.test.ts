// One-time dated MTP redownload warning: shown once per model per fix date,
// dismissal persisted, zero network / local file inspection, and a failure
// can never affect a model load.

import { beforeEach, describe, expect, it, vi } from "vitest";

const { settings, dbMock, detectMock } = vi.hoisted(() => {
  const settings = new Map<string, string>();
  return {
    settings,
    dbMock: {
      getSetting: (key: string) => settings.get(key),
      setSetting: (key: string, value: string) => {
        settings.set(key, value);
      },
      deleteSetting: (key: string) => {
        settings.delete(key);
      },
    },
    detectMock: vi.fn(() => ({ family: "qwen4-exp" })),
  };
});
vi.mock("../src/main/database", () => ({ db: dbMock }));
vi.mock("../src/main/model-config-registry", () => ({
  detectModelConfigFromDir: (...args: unknown[]) => detectMock(...args),
}));

import {
  __test,
  MTP_FIX_DATE,
  checkMtpComponentUpdateOnLoad,
  dismissMtpComponentUpdate,
} from "../src/main/mtp-component-updater";

const MODEL_A = "/Users/x/models/JANGQ-AI/Qwen3.8-Flash-Next-JANG_4M";
const MODEL_B = "/Users/x/models/dealignai/Qwen3.8-Flash-Next-CRACK-JANG2L";

function fakeWindow() {
  const sent: unknown[][] = [];
  return {
    sent,
    win: {
      isDestroyed: () => false,
      webContents: {
        send: (...args: unknown[]) => {
          sent.push(args);
        },
      },
    },
  };
}

async function flush() {
  await new Promise((resolve) => setTimeout(resolve, 10));
}

describe("mtp-component-updater (one-time dated warning)", () => {
  beforeEach(() => {
    settings.clear();
    vi.clearAllMocks();
    detectMock.mockReturnValue({ family: "qwen4-exp" });
  });

  it("parses repo id from the bundle path", () => {
    expect(__test.repoIdFromModelPath(MODEL_B)).toBe(
      "dealignai/Qwen3.8-Flash-Next-CRACK-JANG2L",
    );
    expect(__test.repoIdFromModelPath("nope")).toBe(null);
  });

  it("warns once per model per run when loading a Flash-Next model", async () => {
    const { win, sent } = fakeWindow();
    checkMtpComponentUpdateOnLoad(() => win as never, MODEL_A);
    await flush();
    expect(sent).toHaveLength(1);
    const [channel, payload] = sent[0] as [
      string,
      { repoId: string; remoteFingerprint: string },
    ];
    expect(channel).toBe("models:mtpComponentUpdateAvailable");
    expect(payload.repoId).toBe("JANGQ-AI/Qwen3.8-Flash-Next-JANG_4M");
    expect(payload.remoteFingerprint).toBe(MTP_FIX_DATE);

    // Loading the same model again this run: no second popup.
    checkMtpComponentUpdateOnLoad(() => win as never, MODEL_A);
    await flush();
    expect(sent).toHaveLength(1);

    // A DIFFERENT Flash-Next model (CRACK) gets its own one warning.
    checkMtpComponentUpdateOnLoad(() => win as never, MODEL_B);
    await flush();
    expect(sent).toHaveLength(2);
  });

  it("stays quiet after a persisted dismissal for this fix date", async () => {
    dismissMtpComponentUpdate(
      "JANGQ-AI/Qwen3.8-Flash-Next-JANG_4M",
      MTP_FIX_DATE,
    );
    const { win, sent } = fakeWindow();
    checkMtpComponentUpdateOnLoad(() => win as never, MODEL_A);
    await flush();
    expect(sent).toHaveLength(0);
  });

  it("ignores non-flash-next models and detection failures", async () => {
    const { win, sent } = fakeWindow();
    detectMock.mockReturnValue({ family: "glm5-next" });
    checkMtpComponentUpdateOnLoad(() => win as never, MODEL_A);
    await flush();
    expect(sent).toHaveLength(0);

    detectMock.mockImplementation(() => {
      throw new Error("unreadable config");
    });
    checkMtpComponentUpdateOnLoad(() => win as never, MODEL_A);
    await flush();
    expect(sent).toHaveLength(0);
  });
});

describe("bundleHasFixedMtpComponent (conformance gate)", () => {
  const { mkdtempSync, writeFileSync, mkdirSync } = require("node:fs");
  const { tmpdir } = require("node:os");
  const { join: j } = require("node:path");

  beforeEach(() => {
    settings.clear();
    vi.clearAllMocks();
    detectMock.mockReturnValue({ family: "qwen4-exp" });
  });

  function bundle(opts: { mtpKeys?: boolean; sidecar?: boolean; stamp?: boolean }) {
    const dir = mkdtempSync(j(tmpdir(), "JANGQ-AI-"));
    writeFileSync(
      j(dir, "model.safetensors.index.json"),
      JSON.stringify({
        weight_map: opts.mtpKeys
          ? { "mtp.fc.weight": "model-00023.safetensors", "lm_head.weight": "model-00023.safetensors" }
          : { "lm_head.weight": "model-00023.safetensors" },
      }),
    );
    if (opts.sidecar) {
      mkdirSync(j(dir, "mtp_draft"));
      writeFileSync(j(dir, "mtp_draft", "vmlx_mtp_proposal_head.safetensors"), "x");
    }
    if (opts.stamp) {
      writeFileSync(
        j(dir, "vmlx_mtp_proposal_head.json"),
        JSON.stringify({
          draft_artifact: { file: "mtp_draft/vmlx_mtp_proposal_head.safetensors" },
        }),
      );
    }
    return dir;
  }

  it("conformant bundle (fix present) never warns", async () => {
    const dir = bundle({ mtpKeys: true, sidecar: true, stamp: true });
    expect(__test.bundleHasFixedMtpComponent(dir)).toBe(true);
    const { win, sent } = fakeWindow();
    checkMtpComponentUpdateOnLoad(() => win as never, dir);
    await flush();
    expect(sent).toHaveLength(0);
  });

  it("does not ask an explicitly no-MTP artifact to redownload for MTP", async () => {
    const dir = bundle({ mtpKeys: false });
    writeFileSync(j(dir, "config.json"), JSON.stringify({ jang_config: { mtp: { mtp_mode: "none" } } }));
    expect(__test.bundleExplicitlyOmitsMtp(dir)).toBe(true);
    const { win, sent } = fakeWindow();
    checkMtpComponentUpdateOnLoad(() => win as never, dir);
    await flush();
    expect(sent).toHaveLength(0);
  });

  it("does not suppress broken indexed MTP from a conflicting none stamp", async () => {
    const dir = bundle({ mtpKeys: true });
    writeFileSync(j(dir, "config.json"), JSON.stringify({ jang_config: { mtp: { mtp_mode: "none" } } }));
    expect(__test.bundleExplicitlyOmitsMtp(dir)).toBe(false);
    const { win, sent } = fakeWindow();
    checkMtpComponentUpdateOnLoad(() => win as never, dir);
    await flush();
    expect(sent).toHaveLength(1);
  });

  it("older/broken layouts warn: missing sidecar, missing stamp, no indexed mtp keys", async () => {
    for (const opts of [
      { mtpKeys: true, sidecar: false, stamp: true },
      { mtpKeys: true, sidecar: true, stamp: false },
      { mtpKeys: false, sidecar: true, stamp: true },
    ]) {
      const dir = bundle(opts);
      expect(__test.bundleHasFixedMtpComponent(dir)).toBe(false);
      const { win, sent } = fakeWindow();
      checkMtpComponentUpdateOnLoad(() => win as never, dir);
      await flush();
      expect(sent).toHaveLength(1);
    }
  });
});
