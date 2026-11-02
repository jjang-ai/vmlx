import crypto from "node:crypto";
import vm from "node:vm";
// @ts-expect-error Plain Node proof harness exports.
import * as proofHarness from "../scripts/live-real-ui-model-proof.mjs";
import { createServer } from "node:http";
import {
  chmodSync,
  linkSync,
  mkdtempSync,
  mkdirSync,
  readFileSync,
  realpathSync,
  rmSync,
  statSync,
  symlinkSync,
  writeFileSync,
} from "node:fs";
import { tmpdir } from "node:os";
import path from "node:path";
import { describe, expect, it } from "vitest";

import { REASONING_WITHOUT_ANSWER_NOTICE } from "../src/shared/responsesStreamRecovery";

// @ts-expect-error The production proof harness is deliberately plain Node ESM.
import {
  applyAssertionFailureStatus,
  applyTopLevelCorrelationStatus,
  assertCdpExpressionSyntax,
  captureRequiredScreenshot,
  captureBundleGenerationContract,
  createChatThroughVisibleControl,
  expandCompletedReasoningRails,
  resolveDevElectronExecutable,
  visiblePagedCachePolicy,
  collectOllamaStream,
  deriveProvenSurfaces,
  correlateTerminalResponseToCacheExecution,
  cssEscapeIdentifier,
  expectedUiToolCallCount,
  explicitRangeInputSequence,
  expectedRawMatrixRoutes,
  extractPersistedReasoningMathLinkage,
  isCacheRequestCorrelationVerified,
  isServerRequestCorrelationVerified,
  localRendererModuleEvidence,
  mergeRequiredScreenshotOutcome,
  ownedUiProducerPid,
  parseExplicitPidList,
  parseOptionalPort,
  parseResolvedSamplingKwargs,
  privateCacheAttestationSessionArgs,
  readPrivateExternalJson,
  requestJson,
  releasePrimarySharedPrefix,
  reattestRequiredScreenshot,
  runPostSentinelWorkWithCleanup,
  resolveIndependentBundleGenerationDefaults,
  runtimeBindingFromHealth,
  upsertBoundedDomSample,
  uniqueProofBasename,
  validateExactToolLoopEvidence,
  validateFinalPhaseStopEvidence,
  validateMatrixFlow,
  validateGatewaySingleModelEvidence,
  validateAttachOnlyLifecycle,
  validateGenerationDefaultsEvidence,
  validateFrozenChatParity,
  validateModelBundleBinding,
  validateOwnedRunIntent,
  validatePairedApiEvidence,
  validateRawChannelOrder,
  validateOwnedReuseSessionAttestation,
  validateOwnedUiReleaseSentinel,
  validateReasoningEvidence,
  validateRenderedDomEvidence,
  validateRequestCorrelatedCacheEvidence,
  reasoningNumericRunIsSpew,
  validateNativeMtpSurfaceParity,
  validateServerCacheEvidence,
  validateUiRuntimeProvenance,
  uiProfileRequiresPositiveCacheReuse,
  viteRendererSourceSeen,
  viteRawRendererModulePath,
  waitForCurrentSessionStart,
  waitForOwnedUiReleaseSentinel,
  writePrivateArtifactFile,
} from "../scripts/live-real-ui-model-proof.mjs";

describe("visible New Chat creation", () => {
  it("clicks the actual control and ignores existing rows while waiting for the new row", async () => {
    const old = { id: "old", title: "same title", modelPath: "/model" };
    const fresh = { id: "fresh", title: "same title", modelPath: "/model" };
    let clicks = 0;
    let reads = 0;
    const result = await createChatThroughVisibleControl({
      chats: { getByModel: async () => ++reads < 3 ? [old] : [fresh, old] },
      modelPath: "/model", click: () => { clicks++; }, pollMs: 1,
    });
    expect(result).toEqual(fresh);
    expect(clicks).toBe(1);
    expect(reads).toBe(3);
  });

  it("does not accept a stale row when the click creates nothing", async () => {
    await expect(createChatThroughVisibleControl({
      chats: { getByModel: async () => [{ id: "old", modelPath: "/model" }] },
      modelPath: "/model", click: () => {}, timeoutMs: 5, pollMs: 1,
    })).rejects.toThrow("did not create a new model-bound chat row");
  });

  it.each([
    [[{ id: "wrong", modelPath: "/other" }], "wrong model"],
    [[{ id: "a", modelPath: "/model" }, { id: "b", modelPath: "/model" }], "ambiguous"],
  ])("rejects incorrect or ambiguous new rows", async (created, message) => {
    let clicked = false;
    await expect(createChatThroughVisibleControl({
      chats: { getByModel: async () => clicked ? created : [] },
      modelPath: "/model", click: () => { clicked = true; },
    })).rejects.toThrow(message);
  });
});

const sha = "a".repeat(64);
const otherSha = "b".repeat(64);
const testExecutablePath = realpathSync(process.execPath);
const testExecutableBytes = readFileSync(testExecutablePath);
const testExecutableSha = crypto
  .createHash("sha256")
  .update(testExecutableBytes)
  .digest("hex");
const testExecutablePathSha = crypto
  .createHash("sha256")
  .update(testExecutablePath)
  .digest("hex");
const testExecutablePrefixPath = path.dirname(testExecutablePath);
const testExecutablePrefixPathSha = crypto
  .createHash("sha256")
  .update(testExecutablePrefixPath)
  .digest("hex");
const assistantIds = ["assistant-1", "assistant-2", "assistant-3"];
const prompts = [
  "First bound prompt",
  "Second bound prompt",
  "Preserve $43 and render $47 \\times 19 = 893 < 920 = 46 \\times 20$.",
];
const persistedContents = [
  "REAL_UI_LIVE_TOOL_ONE Answer complete",
  "REAL_UI_LIVE_TOOL_TWO Answer complete",
  "$43 and $47 \\times 19 = 893 < 920 = 46 \\times 20$",
];

function startLifecycleHarness(initial: any) {
  let current = { ...initial };
  const listeners = {
    starting: new Set<(data: any) => void>(),
    ready: new Set<(data: any) => void>(),
    error: new Set<(data: any) => void>(),
  };
  const subscribe = (name: keyof typeof listeners) => (callback: (data: any) => void) => {
    listeners[name].add(callback);
    return () => listeners[name].delete(callback);
  };
  return {
    sessions: {
      get: async () => ({ ...current }),
      getLogs: async () => ["fresh-current-attempt-log"],
      onStarting: subscribe("starting"),
      onReady: subscribe("ready"),
      onError: subscribe("error"),
    },
    update(value: any) {
      current = { ...current, ...value };
    },
    emit(name: keyof typeof listeners, data: any) {
      for (const callback of [...listeners[name]]) callback(data);
    },
  };
}
const renderedContents = [
  persistedContents[0],
  persistedContents[1],
  "$43 and 47 × 19 = 893 < 920 = 46 × 20",
];

describe("current visible Start lifecycle", () => {
  it("serializes leading-digit session IDs exactly like browser CSS.escape", () => {
    expect(cssEscapeIdentifier("336d4a0e-22b1-4f6c-a89a-107aefd619ef"))
      .toBe("\\33 36d4a0e-22b1-4f6c-a89a-107aefd619ef");
    expect(cssEscapeIdentifier("owned-proof-session"))
      .toBe("owned-proof-session");
  });

  it("ignores a reused stale error row and other-session events", async () => {
    const harness = startLifecycleHarness({
      id: "target",
      status: "error",
      lastStartedAt: 100,
    });
    const result = waitForCurrentSessionStart({
      sessions: harness.sessions,
      sessionId: "target",
      baselineLastStartedAt: 100,
      timeoutMs: 500,
      pollMs: 1,
      click: () => {
        setTimeout(() => {
          harness.emit("error", { sessionId: "other", error: "ignore me" });
        }, 1);
        setTimeout(() => {
          harness.emit("starting", { sessionId: "target" });
        }, 2);
        setTimeout(() => {
          harness.update({ status: "running", lastStartedAt: 101 });
          harness.emit("ready", { sessionId: "target", pid: 42 });
        }, 6);
      },
    });

    await expect(result).resolves.toMatchObject({
      id: "target",
      status: "running",
      lastStartedAt: 101,
    });
  });

  it("rejects an attributable current-attempt error with current logs", async () => {
    const harness = startLifecycleHarness({
      id: "target",
      status: "error",
      lastStartedAt: 100,
    });
    const result = waitForCurrentSessionStart({
      sessions: harness.sessions,
      sessionId: "target",
      baselineLastStartedAt: 100,
      timeoutMs: 500,
      pollMs: 1,
      click: () => {
        setTimeout(() => {
          harness.update({ status: "loading", lastStartedAt: 101 });
          harness.emit("starting", { sessionId: "target" });
        }, 1);
        setTimeout(() => {
          harness.update({ status: "error", lastStartedAt: 101 });
          harness.emit("error", {
            sessionId: "target",
            error: "fresh-current-attempt-error",
          });
        }, 2);
      },
    });

    await expect(result).rejects.toThrow(/fresh-current-attempt-log/);
    await expect(result).rejects.toThrow(/fresh-current-attempt-error/);
  });

  it("does not relabel a stale persisted error as a new Start failure", async () => {
    const harness = startLifecycleHarness({
      id: "target",
      status: "error",
      lastStartedAt: 100,
    });
    await expect(waitForCurrentSessionStart({
      sessions: harness.sessions,
      sessionId: "target",
      baselineLastStartedAt: 100,
      timeoutMs: 20,
      pollMs: 1,
      click: () => {},
    })).rejects.toThrow(/current UI Start lifecycle/);
  });

  it("accepts a newer persisted Start identity as a polling fallback", async () => {
    const harness = startLifecycleHarness({
      id: "target",
      status: "error",
      lastStartedAt: 100,
    });
    const result = waitForCurrentSessionStart({
      sessions: harness.sessions,
      sessionId: "target",
      baselineLastStartedAt: 100,
      timeoutMs: 500,
      pollMs: 1,
      click: () => {
        setTimeout(() => {
          harness.update({ status: "running", lastStartedAt: 101 });
        }, 2);
      },
    });
    await expect(result).resolves.toMatchObject({ status: "running" });
  });
});

describe("owned UI producer identity", () => {
  it("binds orchestrated attestations to the directly observed parent worker", () => {
    expect(ownedUiProducerPid({
      orchestrated: true,
      harnessPid: 65680,
      parentPid: 65667,
    })).toBe(65667);
    expect(ownedUiProducerPid({
      orchestrated: false,
      harnessPid: 65680,
      parentPid: 65667,
    })).toBe(65680);
  });

  it("fails closed when the orchestrated parent PID is unavailable", () => {
    expect(() => ownedUiProducerPid({
      orchestrated: true,
      harnessPid: 65680,
      parentPid: 1,
    })).toThrow(/producer PID is invalid/);
  });
});

describe("proof-owned Electron launch isolation", () => {
  it("validates explicit backend/CDP/gateway ports", () => {
    expect(parseOptionalPort(undefined, "CDP")).toBeUndefined();
    expect(parseOptionalPort("9356", "CDP")).toBe(9356);
    expect(() => parseOptionalPort("0", "CDP")).toThrow(/integer from 1 to 65535/);
    expect(() => parseOptionalPort("8080.5", "gateway")).toThrow(
      /integer from 1 to 65535/,
    );
  });

  it("bounds a CDP peer that accepts a socket but never returns headers", async () => {
    const server = createServer(() => {});
    await new Promise<void>((resolve) => server.listen(0, "127.0.0.1", resolve));
    const address = server.address();
    if (!address || typeof address === "string") {
      server.close();
      throw new Error("test server did not expose a TCP port");
    }
    const started = Date.now();
    try {
      await expect(
        requestJson(`http://127.0.0.1:${address.port}/json/list`, 40),
      ).rejects.toThrow(/Timed out requesting/);
      expect(Date.now() - started).toBeLessThan(1_000);
    } finally {
      await new Promise<void>((resolve, reject) => {
        server.close((error) => (error ? reject(error) : resolve()));
      });
    }
  });

  it("starts both app variants with explicit isolated ownership", () => {
    const source = readFileSync(
      path.resolve("scripts/live-real-ui-model-proof.mjs"),
      "utf8",
    );
    const start = source.indexOf("function startUiApp(");
    const end = source.indexOf("async function childProcessTree", start);
    const block = source.slice(start, end);

    expect(block).toContain("VMLX_ALLOW_SECONDARY_INSTANCE: '1'");
    expect(block).toContain("VMLX_PROOF_OWNED_ENGINE_LIFECYCLE: '1'");
    expect(block).toContain("VMLX_PROOF_GATEWAY_PORT: String(gatewayPort)");
    expect(block).toContain("VMLINUX_PROOF_OWNED_ENGINE_LIFECYCLE: '1'");
    expect(block).toContain("VMLINUX_PROOF_GATEWAY_PORT: String(gatewayPort)");
    expect(block).toContain("`--vmlx-user-data-dir=${userDataDir}`");
    expect(block.match(/'--vmlx-allow-secondary-instance'/g)).toHaveLength(2);
    expect(block.match(/env: proofEnv/g)).toHaveLength(2);
    expect(source).toContain(
      "throw new Error('Real UI backend, CDP, and gateway ports must be distinct')",
    );
    expect(source).toContain(
      "'Proof-owned gateway shifted from requested port '",
    );
    expect(source).toContain("kind: 'electron-gateway'");
    expect(source).toContain("gateway_process_binding: gatewayProcessBinding");
  });
});
const defaults = {
  temperature: 0.7,
  topP: 0.9,
  topK: 40,
  minP: 0.05,
  repeatPenalty: 1.1,
  maxNewTokens: 2048,
  source: "jang_config",
};
const modelBundleAttestation = {
  schema: "vmlx-bundle-config-v1",
  directory_state: "available",
  files: {
    "config.json": { state: "present", size_bytes: 2, sha256: sha },
    "generation_config.json": {
      state: "present",
      size_bytes: 2,
      sha256: sha,
    },
    "jang_config.json": { state: "present", size_bytes: 2, sha256: sha },
    "tokenizer_config.json": {
      state: "present",
      size_bytes: 2,
      sha256: sha,
    },
    "chat_template.jinja": {
      state: "present",
      size_bytes: 2,
      sha256: sha,
    },
  },
  aggregate_sha256: sha,
  fingerprint_sha256: sha,
};

describe("generated CDP expression syntax", () => {
  it("accepts the renderer navigation expression used by the live UI worker", () => {
    const expression = `
      (async () => {
        const created = { session: { id: "session-1" } };
        window.dispatchEvent(new CustomEvent("vmlx:navigate", {
          detail: {
            mode: "server",
            panel: "session",
            sessionId: created.session.id,
          },
        }));
      })()
    `;

    expect(assertCdpExpressionSyntax(expression, "renderer-real-ui-chat"))
      .toBe(expression);
  });

  it("rejects the missing dispatchEvent close that broke the release gate", () => {
    const malformed = `
      window.dispatchEvent(new CustomEvent("vmlx:navigate", {
        detail: { mode: "server" },
      });
    `;

    expect(() => assertCdpExpressionSyntax(malformed, "renderer-real-ui-chat"))
      .toThrow(/generated an invalid CDP expression/);
  });

  it("keeps private cache attestation cleanup state in finally scope", async () => {
    const cleanupValues: string[] = [];
    const expression = `
      (async () => {
        let proofSessionId = "session-1";
        let privateConfigRestoreAdditionalArgs = "--existing";
        const privateCacheAttestationArgs = "--private-proof";
        try {
          throw new Error("body failed");
        } finally {
          if (
            proofSessionId
            && privateConfigRestoreAdditionalArgs !== null
            && privateCacheAttestationArgs
          ) {
            cleanupValues.push(privateCacheAttestationArgs);
          }
        }
      })()
    `;

    assertCdpExpressionSyntax(expression, "renderer-real-ui-chat");
    await expect(eval(expression)).rejects.toThrow("body failed");
    expect(cleanupValues).toEqual(["--private-proof"]);
  });

  it("declares the production worker attestation value before its try/finally", () => {
    const harnessSource = readFileSync(
      path.resolve("scripts/live-real-ui-model-proof.mjs"),
      "utf8",
    );
    const workerScopeStart = harnessSource.indexOf(
      "        let proofSessionId = null;",
    );
    const declarationIndex = harnessSource.indexOf(
      "        const privateCacheAttestationArgs = ${JSON.stringify(privateCacheAttestationArgs)};",
      workerScopeStart,
    );
    const tryIndex = harnessSource.indexOf("        try {", workerScopeStart);
    const finallyIndex = harnessSource.indexOf(
      "        } finally {",
      workerScopeStart,
    );
    const cleanupReadIndex = harnessSource.indexOf(
      "            && privateCacheAttestationArgs",
      finallyIndex,
    );

    expect(workerScopeStart).toBeGreaterThan(-1);
    expect(declarationIndex).toBeGreaterThan(workerScopeStart);
    expect(declarationIndex).toBeLessThan(tryIndex);
    expect(tryIndex).toBeLessThan(finallyIndex);
    expect(cleanupReadIndex).toBeGreaterThan(finallyIndex);
  });

  it("double-escapes every backslash inside the generated worker template", () => {
    const harnessSource = readFileSync(
      path.resolve("scripts/live-real-ui-model-proof.mjs"),
      "utf8",
    );
    const workerStart = harnessSource.indexOf(
      "      rendererResult = await evaluate(cdp, `",
    );
    const workerEnd = harnessSource.indexOf(
      "\n    `, 1_200_000)",
      workerStart,
    );
    const workerTemplate = harnessSource.slice(workerStart, workerEnd);
    const hazards: string[] = [];
    for (let index = 0; index < workerTemplate.length; index += 1) {
      if (workerTemplate[index] !== "\\") continue;
      let runLength = 0;
      while (workerTemplate[index + runLength] === "\\") {
        runLength += 1;
      }
      if (runLength % 2 === 1) {
        hazards.push(workerTemplate.slice(index, index + runLength + 1));
      }
      index += runLength - 1;
    }

    expect(workerStart).toBeGreaterThan(-1);
    expect(workerEnd).toBeGreaterThan(workerStart);
    expect(hazards).toEqual([]);
  });

  it("binds the live proof to the product-owned Chat Settings controls", () => {
    const harnessSource = readFileSync(
      path.resolve("scripts/live-real-ui-model-proof.mjs"),
      "utf8",
    );
    const toolbarSource = readFileSync(
      path.resolve("src/renderer/src/components/layout/ChatModeToolbar.tsx"),
      "utf8",
    );
    const sessionViewSource = readFileSync(
      path.resolve("src/renderer/src/components/sessions/SessionView.tsx"),
      "utf8",
    );
    const chatSettingsSource = readFileSync(
      path.resolve("src/renderer/src/components/chat/ChatSettings.tsx"),
      "utf8",
    );

    expect(toolbarSource).toContain(
      'data-vmlx-control="chat-settings"',
    );
    expect(sessionViewSource).toContain(
      'data-vmlx-control="chat-settings"',
    );
    expect(chatSettingsSource).toContain(
      'data-vmlx-surface="chat-settings"',
    );
    expect(chatSettingsSource).toContain(
      'data-vmlx-control="chat-settings-save"',
    );
    expect(chatSettingsSource).toContain(
      "data-vmlx-state={saving ? 'saving' : dirty ? 'dirty' : 'saved'}",
    );
    expect(harnessSource).toContain(
      'document.querySelectorAll(\'[data-vmlx-control="chat-settings"]\')',
    );
    expect(harnessSource).toContain(
      'document.querySelector(\'[data-vmlx-surface="chat-settings"]\')',
    );
    expect(harnessSource).toContain(
      '\'[data-vmlx-control="chat-settings-save"]\'',
    );
    expect(harnessSource).toContain(
      "const button = buttons.length === 1 ? buttons[0] : null;",
    );
    expect(harnessSource).toContain(
      "button.getAttribute('data-vmlx-state') === 'dirty'",
    );
    expect(harnessSource).toContain(
      "current.getAttribute('data-vmlx-state') === 'saved'",
    );
    // Adaptive MTP never pins the sampler; the harness must not expect greedy
    // UI values from the MTP mode any more.
    expect(harnessSource).toContain('const nativeMtpGreedyUi = false;');
    expect(harnessSource).not.toContain("['auto', 'deterministic'].includes(");
    expect(harnessSource).toContain("'Top P': explicitUiSampling.topP");
    expect(harnessSource).toContain("'Top K': explicitUiSampling.topK");
    expect(harnessSource).toContain("'Min P': explicitUiSampling.minP");
    expect(harnessSource).not.toContain(
      "(button.textContent || '').replace(/\\\\s+/g, ' ').trim() === 'Save'",
    );
    expect(harnessSource).not.toContain(
      ".trim() === 'Chat'\n            ) || null,",
    );
    expect(harnessSource).toContain("builtinInput.click();");
    expect(harnessSource).toContain("input.click();");
    expect(harnessSource).toContain("'visible Working Directory input'");
    expect(harnessSource).not.toContain("checkedSetter.call(");
  });

  it("keeps streaming proof capture linear and bounds full DOM samples", () => {
    const harnessSource = readFileSync(
      path.resolve("scripts/live-real-ui-model-proof.mjs"),
      "utf8",
    );

    expect(harnessSource).toContain("const maxDomSamplesPerMessage = 24;");
    expect(harnessSource).toContain("const streamTraceState = new Map();");
    expect(harnessSource).toContain(
      "const upsertBoundedDomSample = ${upsertBoundedDomSample.toString()}",
    );
    expect(harnessSource).toContain(
      "fullContentLength: data.fullContent.length",
    );
    expect(harnessSource).toContain(
      "scheduleDomSample(messageId, event + ':' + channel, event !== 'stream')",
    );
    expect(harnessSource).not.toContain("payload: data,\n          });");
  });

  it("reserves bounded DOM capacity for both reasoning and visible content", () => {
    const samples: Array<Record<string, unknown>> = [];
    const state = {
      count: 0,
      lastSignature: "",
      lastStoredIndex: -1,
    };
    for (let index = 1; index <= 30; index += 1) {
      upsertBoundedDomSample(
        samples,
        state,
        {
          messageId: "assistant-1",
          answerText: "",
          reasoningText: "r".repeat(index),
          katexCount: 0,
          toolCards: [],
        },
        24,
      );
    }
    expect(samples).toHaveLength(11);
    expect(state.channelCounts.reasoning).toBe(11);
    expect(state.channelCounts.content).toBe(0);

    for (let index = 1; index <= 30; index += 1) {
      upsertBoundedDomSample(
        samples,
        state,
        {
          messageId: "assistant-1",
          answerText: "x".repeat(index),
          reasoningText: "r".repeat(30),
          katexCount: 0,
          toolCards: [],
        },
        24,
      );
    }
    expect(samples).toHaveLength(22);
    expect(state.count).toBe(22);
    expect(state.channelCounts.content).toBe(11);

    upsertBoundedDomSample(
      samples,
      state,
      {
        messageId: "assistant-1",
        answerText: "terminal answer",
        reasoningText: "terminal reasoning",
        katexCount: 1,
        toolCards: [{ phase: "result" }],
      },
      24,
      true,
    );
    expect(samples).toHaveLength(22);
    expect(samples.some((sample) =>
      sample.answerText === "terminal answer"
    )).toBe(true);
  });

  it("stages and attests each V5 cache phase before the real Start click", () => {
    const harnessSource = readFileSync(
      path.resolve("scripts/live-real-ui-model-proof.mjs"),
      "utf8",
    );
    const preflightSource = readFileSync(
      path.resolve("scripts/scoped-release-preflight-20.py"),
      "utf8",
    );

    expect(harnessSource).toContain(
      "usePagedCache: Boolean(activeReleasePhase.paged_ram)",
    );
    expect(harnessSource).toContain(
      "kvCacheQuantization: String(activeReleasePhase.kv_cache_quantization || 'none')",
    );
    expect(harnessSource).toContain(
      "const releasePagedCacheMemoryMb = activeReleasePhase?.paged_ram",
    );
    expect(harnessSource).toMatch(
      /const releasePagedCacheMemoryMb = activeReleasePhase\?\.paged_ram\s+\? 4096\s+: null/,
    );
    expect(harnessSource).toContain(
      "const releaseBlockDiskCacheMaxGb = activeReleasePhase ? 10 : null",
    );
    expect(harnessSource).toContain(
      "{ cacheMemoryMb: releasePagedCacheMemoryMb }",
    );
    expect(
      harnessSource.match(
        /cacheMemoryMb: releasePagedCacheMemoryMb/g,
      ),
    ).toHaveLength(2);
    expect(
      harnessSource.match(
        /blockDiskCacheMaxGb: releaseBlockDiskCacheMaxGb/g,
      ),
    ).toHaveLength(2);
    expect(
      harnessSource.match(/cacheMemoryPercent: 0/g),
    ).toHaveLength(2);
    expect(harnessSource).not.toContain(
      "const releasePagedCacheMemoryMb =\n          ${",
    );
    expect(harnessSource).toContain("const parityIndexes = [0]");
    const cacheDrawerProbeStart = harnessSource.indexOf(
      "let serverCacheControls = { requested: false, verified: false }",
    );
    const cacheDrawerProbeEnd = harnessSource.indexOf(
      "serverCacheControls = { requested: true, verified: false, error:",
      cacheDrawerProbeStart,
    );
    const cacheDrawerProbe = harnessSource.slice(
      cacheDrawerProbeStart,
      cacheDrawerProbeEnd,
    );
    expect(cacheDrawerProbe).toContain("const isVisible = (element) => {");
    expect(cacheDrawerProbe).toContain(
      "'[data-vmlx-control=\"server-settings\"]'",
    );
    expect(preflightSource).toContain(
      '"VMLINUX_REAL_UI_EXPECT_PAGED_CACHE": (',
    );
    expect(preflightSource).toContain(
      'if int(phase["index"]) != 0:',
    );
    expect(preflightSource).toContain(
      'environment["VMLINUX_REAL_UI_MAX_TOKENS"] = "2048"',
    );
  });
});

describe("required real UI screenshot failure handling", () => {
  const onePixelPng = Buffer.from(
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mNk+A8AAQUBAScY42YAAAAASUVORK5CYII=",
    "base64",
  );

  it("preserves the primary renderer failure and records a secondary capture timeout", async () => {
    const primarySendErrors = [{
      turn: 1,
      stage: "first_visible_ui_send",
      message: "timeout waiting for terminal event",
    }];
    const capture = await captureRequiredScreenshot(
      async () => {
        throw new Error("CDP timeout: Page.captureScreenshot");
      },
      "/private/proof/chat.png",
    );
    const result = mergeRequiredScreenshotOutcome({
      rendererFailureStage: "first_visible_ui_send",
      sendErrors: primarySendErrors,
    }, capture);

    expect(result.rendererFailureStage).toBe("first_visible_ui_send");
    expect(result.sendErrors).toBe(primarySendErrors);
    expect(result.screenshotCapture).toMatchObject({
      status: "failed",
      path: null,
      error: {
        stage: "chat_screenshot_capture",
        message: "CDP timeout: Page.captureScreenshot",
      },
    });

    const proofDir = mkdtempSync(path.join(tmpdir(), "vmlx-screenshot-failure-"));
    try {
      const proofPath = path.join(proofDir, "proof.json");
      writePrivateArtifactFile(proofPath, JSON.stringify({
        rendererFailureStage: result.rendererFailureStage,
        sendErrors: result.sendErrors,
        screenshots: { chat: capture.path },
        screenshotCapture: result.screenshotCapture,
      }));
      expect(JSON.parse(readFileSync(proofPath, "utf8"))).toMatchObject({
        rendererFailureStage: "first_visible_ui_send",
        screenshots: { chat: null },
        screenshotCapture: {
          status: "failed",
          error: { stage: "chat_screenshot_capture" },
        },
      });
    } finally {
      rmSync(proofDir, { recursive: true, force: true });
    }
  });

  it("fails closed on screenshot capture when no earlier renderer failure exists", async () => {
    const capture = await captureRequiredScreenshot(
      async () => {
        throw new Error("capture rejected");
      },
      "/private/proof/chat.png",
    );
    const result = mergeRequiredScreenshotOutcome({
      rendererFailureStage: null,
      sendErrors: [],
    }, capture);

    expect(result.rendererFailureStage).toBe("chat_screenshot_capture");
    expect(result.screenshotCapture.path).toBeNull();
  });

  it("attests a real private PNG before retaining it as visual evidence", async () => {
    const proofDir = mkdtempSync(path.join(tmpdir(), "vmlx-screenshot-valid-"));
    try {
      const screenshotPath = path.join(proofDir, "chat.png");
      const capture = await captureRequiredScreenshot(
        async (filePath) => writePrivateArtifactFile(filePath, onePixelPng),
        screenshotPath,
      );
      const result = mergeRequiredScreenshotOutcome({
        rendererFailureStage: null,
        sendErrors: [],
      }, capture);
      const finalCapture = reattestRequiredScreenshot(
        result.screenshotCapture,
      );

      expect(result.rendererFailureStage).toBeNull();
      expect(finalCapture).toMatchObject({
        status: "captured",
        path: screenshotPath,
        attestation: {
          path: screenshotPath,
          byteSize: onePixelPng.length,
          sha256: crypto.createHash("sha256").update(onePixelPng).digest("hex"),
          openedNoFollow: true,
          regularFile: true,
          nonSymlink: true,
          privatePermissions: true,
          exactRequestedPath: true,
          pngSignatureValid: true,
        },
        finalAttestation: {
          path: screenshotPath,
          byteSize: onePixelPng.length,
          sha256: crypto.createHash("sha256").update(onePixelPng).digest("hex"),
        },
        error: null,
      });
    } finally {
      rmSync(proofDir, { recursive: true, force: true });
    }
  });

  it("rejects a missing screenshot artifact", async () => {
    const proofDir = mkdtempSync(path.join(tmpdir(), "vmlx-screenshot-missing-"));
    try {
      const capture = await captureRequiredScreenshot(
        async (filePath) => filePath,
        path.join(proofDir, "missing.png"),
      );
      expect(capture).toMatchObject({
        status: "failed",
        path: null,
        attestation: null,
        error: { stage: "chat_screenshot_capture" },
      });
    } finally {
      rmSync(proofDir, { recursive: true, force: true });
    }
  });

  it("rejects an empty screenshot artifact", async () => {
    const proofDir = mkdtempSync(path.join(tmpdir(), "vmlx-screenshot-empty-"));
    try {
      const capture = await captureRequiredScreenshot(
        async (filePath) => writePrivateArtifactFile(filePath, Buffer.alloc(0)),
        path.join(proofDir, "empty.png"),
      );
      expect(capture.error?.message).toMatch(/artifact is empty/);
      expect(capture.path).toBeNull();
    } finally {
      rmSync(proofDir, { recursive: true, force: true });
    }
  });

  it("rejects a non-PNG screenshot artifact", async () => {
    const proofDir = mkdtempSync(path.join(tmpdir(), "vmlx-screenshot-format-"));
    try {
      const capture = await captureRequiredScreenshot(
        async (filePath) => writePrivateArtifactFile(filePath, Buffer.from("not a png")),
        path.join(proofDir, "chat.png"),
      );
      expect(capture.error?.message).toMatch(/valid PNG signature/);
      expect(capture.path).toBeNull();
    } finally {
      rmSync(proofDir, { recursive: true, force: true });
    }
  });

  it("rejects a substituted screenshot path", async () => {
    const proofDir = mkdtempSync(path.join(tmpdir(), "vmlx-screenshot-substitute-"));
    try {
      const requestedPath = path.join(proofDir, "requested.png");
      const substitutedPath = path.join(proofDir, "substituted.png");
      const capture = await captureRequiredScreenshot(
        async () => writePrivateArtifactFile(substitutedPath, onePixelPng),
        requestedPath,
      );
      expect(capture.error?.message).toMatch(/substituted a different artifact path/);
      expect(capture.path).toBeNull();
    } finally {
      rmSync(proofDir, { recursive: true, force: true });
    }
  });

  it("rejects a symlink screenshot artifact", async () => {
    const proofDir = mkdtempSync(path.join(tmpdir(), "vmlx-screenshot-symlink-"));
    try {
      const targetPath = path.join(proofDir, "target.png");
      const screenshotPath = path.join(proofDir, "chat.png");
      writePrivateArtifactFile(targetPath, onePixelPng);
      symlinkSync(targetPath, screenshotPath);
      const capture = await captureRequiredScreenshot(
        async (filePath) => filePath,
        screenshotPath,
      );
      expect(capture.error?.message).toMatch(/regular, non-symlink file/);
      expect(capture.path).toBeNull();
    } finally {
      rmSync(proofDir, { recursive: true, force: true });
    }
  });

  it("rejects a non-regular screenshot artifact", async () => {
    const proofDir = mkdtempSync(path.join(tmpdir(), "vmlx-screenshot-directory-"));
    try {
      const screenshotPath = path.join(proofDir, "chat.png");
      const capture = await captureRequiredScreenshot(
        async (filePath) => {
          mkdirSync(filePath, { mode: 0o700 });
          return filePath;
        },
        screenshotPath,
      );
      expect(capture.error?.message).toMatch(/regular, non-symlink file/);
      expect(capture.path).toBeNull();
    } finally {
      rmSync(proofDir, { recursive: true, force: true });
    }
  });

  it("rejects a group-readable screenshot artifact", async () => {
    const proofDir = mkdtempSync(path.join(tmpdir(), "vmlx-screenshot-mode-"));
    try {
      const screenshotPath = path.join(proofDir, "chat.png");
      const capture = await captureRequiredScreenshot(
        async (filePath) => {
          writePrivateArtifactFile(filePath, onePixelPng);
          chmodSync(filePath, 0o640);
          return filePath;
        },
        screenshotPath,
      );
      expect(capture.error?.message).toMatch(/group\/world accessible/);
      expect(capture.path).toBeNull();
    } finally {
      rmSync(proofDir, { recursive: true, force: true });
    }
  });

  it("rejects deletion after capture during final re-attestation", async () => {
    const proofDir = mkdtempSync(path.join(tmpdir(), "vmlx-screenshot-delete-"));
    try {
      const screenshotPath = path.join(proofDir, "chat.png");
      const capture = await captureRequiredScreenshot(
        async (filePath) => writePrivateArtifactFile(filePath, onePixelPng),
        screenshotPath,
      );
      rmSync(screenshotPath);
      const finalCapture = reattestRequiredScreenshot(capture);
      const result = mergeRequiredScreenshotOutcome({
        rendererFailureStage: "first_visible_ui_send",
        sendErrors: [{
          turn: 1,
          stage: "first_visible_ui_send",
          message: "timeout waiting for terminal event",
        }],
      }, finalCapture);

      expect(finalCapture).toMatchObject({
        status: "failed",
        path: null,
        finalAttestation: null,
        error: { stage: "chat_screenshot_reattestation" },
      });
      expect(result.rendererFailureStage).toBe("first_visible_ui_send");
      const proofPath = path.join(proofDir, "failed-proof.json");
      writePrivateArtifactFile(proofPath, JSON.stringify({
        status: "fail",
        failureStage: result.rendererFailureStage,
        screenshots: { chat: result.screenshotCapture.path },
        screenshotCapture: result.screenshotCapture,
      }));
      expect(JSON.parse(readFileSync(proofPath, "utf8"))).toMatchObject({
        status: "fail",
        failureStage: "first_visible_ui_send",
        screenshots: { chat: null },
        screenshotCapture: {
          status: "failed",
          error: { stage: "chat_screenshot_reattestation" },
        },
      });
    } finally {
      rmSync(proofDir, { recursive: true, force: true });
    }
  });

  it("rejects valid-PNG replacement after capture by identity and digest", async () => {
    const proofDir = mkdtempSync(path.join(tmpdir(), "vmlx-screenshot-replace-"));
    try {
      const screenshotPath = path.join(proofDir, "chat.png");
      const capture = await captureRequiredScreenshot(
        async (filePath) => writePrivateArtifactFile(filePath, onePixelPng),
        screenshotPath,
      );
      rmSync(screenshotPath);
      writePrivateArtifactFile(
        screenshotPath,
        Buffer.concat([onePixelPng, Buffer.from("replacement")]),
      );
      const finalCapture = reattestRequiredScreenshot(capture);

      expect(finalCapture.error?.stage).toBe("chat_screenshot_reattestation");
      expect(finalCapture.error?.message).toMatch(/identity changed after capture/);
      expect(finalCapture.path).toBeNull();
    } finally {
      rmSync(proofDir, { recursive: true, force: true });
    }
  });

  it("rejects a symlink swap after capture during final re-attestation", async () => {
    const proofDir = mkdtempSync(path.join(tmpdir(), "vmlx-screenshot-swap-"));
    try {
      const screenshotPath = path.join(proofDir, "chat.png");
      const replacementPath = path.join(proofDir, "replacement.png");
      const capture = await captureRequiredScreenshot(
        async (filePath) => writePrivateArtifactFile(filePath, onePixelPng),
        screenshotPath,
      );
      rmSync(screenshotPath);
      writePrivateArtifactFile(replacementPath, onePixelPng);
      symlinkSync(replacementPath, screenshotPath);
      const finalCapture = reattestRequiredScreenshot(capture);

      expect(finalCapture.error?.stage).toBe("chat_screenshot_reattestation");
      expect(finalCapture.error?.message).toMatch(/regular, non-symlink file/);
      expect(finalCapture.path).toBeNull();
    } finally {
      rmSync(proofDir, { recursive: true, force: true });
    }
  });

  it("keeps the production PASS assertions fail-closed on screenshot evidence", () => {
    const harnessSource = readFileSync(
      path.resolve("scripts/live-real-ui-model-proof.mjs"),
      "utf8",
    );

    expect(harnessSource).toContain(
      "required real UI screenshot was not successfully captured",
    );
    expect(harnessSource).toContain(
      "rendererResult = mergeRequiredScreenshotOutcome(",
    );
    expect(harnessSource).toContain(
      "reattestRequiredScreenshot(rendererResult.screenshotCapture)",
    );
    expect(harnessSource).toContain(
      "fd = openSync(exactPath, fsConstants.O_RDONLY | noFollow)",
    );
    expect(harnessSource).toContain("screenshot: chatScreenshot");
  });
});

function canonicalJson(value: unknown): string {
  const normalize = (node: unknown): unknown => {
    if (Array.isArray(node)) return node.map(normalize);
    if (!node || typeof node !== "object") return node;
    return Object.fromEntries(
      Object.keys(node as Record<string, unknown>)
        .sort()
        .map((key) => [
          key,
          normalize((node as Record<string, unknown>)[key]),
        ]),
    );
  };
  return JSON.stringify(normalize(value)).replace(
    /[^\x20-\x7e]/g,
    (character) =>
      `\\u${character.charCodeAt(0).toString(16).padStart(4, "0")}`,
  );
}

function canonicalHash(value: unknown): string {
  return crypto.createHash("sha256").update(canonicalJson(value)).digest("hex");
}

function ownedRunIntent(
  primaryBundlePath: string,
  nativeBundlePath: string,
  harnessRoot: string,
): Record<string, unknown> {
  const harnessPaths = {
    ui: "panel/scripts/live-real-ui-model-proof.mjs",
    api: "tests/cross_matrix/run_agentic_protocol_matrix.py",
    cache: "tests/cross_matrix/run_cache_hierarchy_live_gate.py",
    semantic: "panel/scripts/scoped-release-preflight-20.py",
  };
  const harnesses = Object.fromEntries(
    Object.entries(harnessPaths).map(([name, relativePath]) => {
      const absolutePath = path.join(harnessRoot, relativePath);
      mkdirSync(path.dirname(absolutePath), { recursive: true });
      writeFileSync(absolutePath, `${name}-harness`);
      return [
        name,
        {
          relative_path: relativePath,
          sha256: crypto
            .createHash("sha256")
            .update(readFileSync(absolutePath))
            .digest("hex"),
        },
      ];
    }),
  );
  const primary = {
    representative_id: "primary_tq_supported",
    bundle_role: "primary",
    session_policy: "primary_stable_session",
    model: "primary-model",
    model_bundle_path: primaryBundlePath,
    bundle_fingerprint_sha256: "1".repeat(64),
    native_cache_policy: "standard_kv",
  };
  const phasePlan = [
    {
      phase_index: 0,
      phase_name: "primary_ssd_only_store",
      cache_policy: "q4",
      paged_ram: false,
      operation: "store",
      restart_required: false,
      kv_cache_quantization: "auto",
      tq_policy: "auto-model-safe-required",
      ui_action_profile: "primary-reasoning-render-store",
      ui_turn_count: 1,
      api_action_profile: "full-agentic-plus-cache-store",
    },
    {
      phase_index: 1,
      phase_name: "primary_ssd_only_restart_probe",
      cache_policy: "q4",
      paged_ram: false,
      operation: "probe",
      restart_required: true,
      kv_cache_quantization: "auto",
      tq_policy: "auto-model-safe-required",
      ui_action_profile: "primary-tool-restart-probe",
      ui_turn_count: 1,
      api_action_profile: "cache-probe",
    },
    {
      phase_index: 2,
      phase_name: "primary_paged_on_store",
      cache_policy: "q4",
      paged_ram: true,
      operation: "store-evict-refault",
      restart_required: true,
      kv_cache_quantization: "auto",
      tq_policy: "auto-model-safe-required",
      ui_action_profile: "primary-history-paged-evict-refault",
      ui_turn_count: 1,
      api_action_profile: "cache-evict-refault",
    },
    {
      phase_index: 3,
      phase_name: "primary_paged_on_restart_probe",
      cache_policy: "q4",
      paged_ram: true,
      operation: "probe",
      restart_required: true,
      kv_cache_quantization: "auto",
      tq_policy: "auto-model-safe-required",
      ui_action_profile: "primary-restart-followup",
      ui_turn_count: 1,
      api_action_profile: "cache-restart-probe",
    },
    {
      phase_index: 4,
      phase_name: "primary_tq_off",
      cache_policy: "ssd-only",
      paged_ram: false,
      operation: "store-probe",
      restart_required: true,
      kv_cache_quantization: "none",
      tq_policy: "explicit-off",
      ui_action_profile: "primary-tq-off-probe",
      ui_turn_count: 1,
      api_action_profile: "cache-tq-off-store-probe",
    },
  ].map((phase) => ({ ...primary, ...phase }));
  phasePlan.push({
    phase_index: 5,
    phase_name: "native_exception",
    representative_id: "secondary_native_exception",
    bundle_role: "native",
    cache_policy: "native",
    paged_ram: false,
    operation: "switch-validate",
    restart_required: true,
    session_policy: "distinct_native_session",
    kv_cache_quantization: "none",
    tq_policy: "native-suppressed",
    ui_action_profile: "native-three-turn-switch",
    ui_turn_count: 3,
    api_action_profile: "full-agentic-native-cache",
    native_cache_policy: "typed-composite",
    model: "native-model",
    model_bundle_path: nativeBundlePath,
    bundle_fingerprint_sha256: "2".repeat(64),
  });
  const intent: Record<string, unknown> = {
    schema: "vmlx-r20-owned-run-intent-v5",
    run_id: "run",
    nonce: "nonce",
    source_commit: "3".repeat(40),
    source_tree: "4".repeat(40),
    harnesses,
    direct_base_url: "http://127.0.0.1:8001",
    native_direct_base_url: "http://127.0.0.1:8002",
    gateway_base_url: "http://127.0.0.1:8088",
    direct_health_url: "http://127.0.0.1:8001/health",
    native_direct_health_url: "http://127.0.0.1:8002/health",
    gateway_health_url: "http://127.0.0.1:8088/health",
    l2_size_eviction_requirements: {
      disk_bytes_within_saved_limit: true,
      older_unused_prefix_eviction_required: true,
      recent_target_survival_required: true,
      restart_restore_required: true,
      counter_only_evidence_allowed: false,
    },
    created_at: new Date().toISOString(),
    phase_plan: phasePlan,
  };
  intent.canonical_sha256 = canonicalHash(intent);
  return intent;
}

function streamEvent(
  sequence: number,
  channel: "reasoning" | "content",
  fullContent: string,
  delta: string,
) {
  return {
    sequence,
    event: "stream",
    channel,
    delta,
    cumulativeReset: false,
    payload: {
      fullContent,
      fullContentLength: fullContent.length,
      isReasoning: channel === "reasoning",
      ...(channel === "reasoning"
        ? {
            reasoningSegments: [fullContent],
            reasoningSegmentLength: fullContent.length,
          }
        : {}),
    },
  };
}

function primaryTrace(messageId: string, finalContent: string, toolIndex?: number) {
  const events: Array<Record<string, unknown>> = [
    streamEvent(1, "reasoning", "Reason", "Reason"),
    streamEvent(2, "reasoning", "Reason carefully", " carefully"),
  ];
  let sequence = 3;
  if (toolIndex != null) {
    const callId = `call-${toolIndex + 1}`;
    events.push(
      {
        sequence: sequence++,
        event: "tool",
        channel: "tool",
        payload: { toolCallId: callId, phase: "calling" },
      },
      {
        sequence: sequence++,
        event: "tool",
        channel: "tool",
        payload: { toolCallId: callId, phase: "result" },
      },
    );
  }
  const prefix = finalContent.slice(0, Math.max(1, Math.floor(finalContent.length / 2)));
  events.push(
    streamEvent(sequence++, "content", prefix, prefix),
    streamEvent(
      sequence++,
      "content",
      finalContent,
      finalContent.slice(prefix.length),
    ),
    {
      sequence,
      event: "terminal",
      channel: "terminal",
      payload: { messageId },
    },
  );
  return { messageId, events };
}

function cacheStats({
  processed,
  hitRequests,
  hitTokens,
  partialTokens,
  skippedTokens,
}: {
  processed: number;
  hitRequests: number;
  hitTokens: number;
  partialTokens: number;
  skippedTokens: number;
}) {
  return {
    scheduler_stats: {
      num_requests_processed: processed,
      cache_hit_requests: hitRequests,
      cache_hit_tokens: hitTokens,
      cache_reuse_partial_tokens: partialTokens,
      cache_reuse_skip_tokens: skippedTokens,
    },
  };
}

function goodResult(): Record<string, any> {
  const sourceCommit = "c".repeat(40);
  const modules = localRendererModuleEvidence();
  const records = assistantIds.map((id, index) => ({
    id,
    content: persistedContents[index],
  }));
  const messages = records.map((record, index) => ({
    messageId: record.id,
    role: "assistant",
    visible: true,
    answerText: renderedContents[index],
    answerState: "complete",
    answerFullLength: record.content.length,
    answerRenderedLength: record.content.length,
    reasoningText: "Reason carefully",
    reasoningSegments: ["Reason carefully"],
    html:
      index === 2
        ? '$43 and <span class="katex">47 × 19 = 893 &lt; 920 = 46 × 20</span>'
        : record.content,
    katexCount: index === 2 ? 1 : 0,
    katexErrorCount: 0,
    // mathMarkdown uses KaTeX output:'html', which intentionally has no
    // MathML annotation. The persisted answer remains the source attestation.
    katexAnnotations: [],
    currencyOccurrences:
      index === 2 ? [{ text: "$43", insideKatex: false }] : [],
    toolCards:
      index < 2
        ? [
            {
              callId: `call-${index + 1}`,
              name: "run_command",
              phase: "result",
              visible: true,
              text: `Used run_command REAL_UI_LIVE_TOOL_${index + 1}`,
            },
          ]
        : [],
  }));
  const samples = assistantIds.flatMap((messageId, index) => [
    {
      messageId,
      answerText: renderedContents[index].slice(
        0,
        Math.max(1, Math.floor(renderedContents[index].length / 2)),
      ),
      reasoningText: "Reason",
    },
    {
      messageId,
      answerText: renderedContents[index],
      reasoningText: "Reason carefully",
    },
  ]);
  const healthRaw = {
    status: "healthy",
    model_loaded: true,
    model_name: "test-model",
    model_bundle_provenance: modelBundleAttestation,
    runtime_provenance: {
      model_bundle_provenance: modelBundleAttestation,
    },
  };
  const runtimeBinding = {
    backend_pid: 4321,
    pid: 4321,
    fingerprint_sha256: sha,
    runtime_source_hashes: {
      server_module_sha256: sha,
      package_init_sha256: sha,
      python_source_tree_sha256: sha,
      python_executable_fingerprint_sha256: testExecutablePathSha,
    },
    python_source_file_count: 10,
    python_source_read_error_count: 0,
    model_name: "test-model",
    model_bundle_fingerprint_sha256: sha,
    model_bundle_files: modelBundleAttestation.files,
    cache_topology_fingerprint_sha256: sha,
  };
  const result: Record<string, any> = {
    format: "vmlx-electron-ui-proof-v2",
    run_id: "ui-run-1",
    status: "pass",
    surfaceStatus: "partial_ui_only",
    modelPath: "/private/models/test-model",
    servedModel: "test-model",
    baseUrl: "http://127.0.0.1:8000",
    requestedWireApi: "chat",
    requestedEnableThinking: undefined,
    requestedBuiltinTools: true,
    requestedServerCacheControls: true,
    assistantMessageIds: assistantIds,
    assistantRecords: records,
    persistedReasoningByMessage: assistantIds.map(() => [
      "Reason carefully",
    ]),
    messageEventTrace: assistantIds.map((messageId, index) =>
      primaryTrace(
        messageId,
        persistedContents[index],
        index < 2 ? index : undefined,
      ),
    ),
    renderedDom: {
      messages,
      userMessages: [
        {
          messageId: "user-3",
          role: "user",
          visible: true,
          text: "Preserve $43 and render 47 × 19 = 893 < 920 = 46 × 20.",
          html: '$43 and <span class="katex">47 × 19 = 893 &lt; 920 = 46 × 20</span>',
          katexCount: 1,
          katexErrorCount: 0,
          currencyOccurrences: [{ text: "$43", insideKatex: false }],
        },
      ],
      samples,
      rawI18nKeys: [],
      visibleErrors: [],
      transientAlerts: [],
    },
    requestContract: {
      uiActionProfile: "legacy-three-turn",
      promptOne: prompts[0],
      promptTwo: prompts[1],
      promptThree: prompts[2],
      requestMaxTokens: undefined,
      maxToolIterations: 4,
      toolResultMaxChars: 12500,
      samplingOverrides: {},
    },
    persistedToolsByMessage: [
      [
        {
          phase: "calling",
          toolCallId: "call-1",
          toolName: "run_command",
        },
        {
          phase: "result",
          toolCallId: "call-1",
          toolName: "run_command",
          detail: "REAL_UI_LIVE_TOOL_ONE",
        },
      ],
      [
        {
          phase: "calling",
          toolCallId: "call-2",
          toolName: "run_command",
        },
        {
          phase: "result",
          toolCallId: "call-2",
          toolName: "run_command",
          detail: "REAL_UI_LIVE_TOOL_TWO",
        },
      ],
      [],
    ],
    persistedOaiCallsByMessage: [
      [
        {
          id: "call-1",
          function: {
            name: "run_command",
            arguments: JSON.stringify({
              command:
                "printf REAL_UI_LIVE_TOOL_ONE > real_ui_tool_probe_1.txt",
            }),
          },
        },
      ],
      [
        {
          id: "call-2",
          function: {
            name: "run_command",
            arguments: JSON.stringify({
              command:
                "test -f real_ui_tool_probe_1.txt && printf REAL_UI_LIVE_TOOL_TWO > real_ui_tool_probe_2.txt",
            }),
          },
        },
      ],
      [],
    ],
    persistedOaiResultsByMessage: [
      [{ tool_call_id: "call-1", output: "REAL_UI_LIVE_TOOL_ONE" }],
      [{ tool_call_id: "call-2", output: "REAL_UI_LIVE_TOOL_TWO" }],
      [],
    ],
    toolProbeFiles: {
      "real_ui_tool_probe_1.txt": "REAL_UI_LIVE_TOOL_ONE",
      "real_ui_tool_probe_2.txt": "REAL_UI_LIVE_TOOL_TWO",
    },
    toolProbeCleanup: {
      removed: true,
      paths: [
        "/private/work/real_ui_tool_probe_1.txt",
        "/private/work/real_ui_tool_probe_2.txt",
      ],
    },
    bundleGenerationContract: {
      bundle_path: "/private/models/test-model",
      defaults,
      health_attestation: modelBundleAttestation,
    },
    rendererGenerationDefaults: defaults,
    chatSettingsDom: {
      values: {
        temperature: 0.7,
        topP: 0.9,
        topK: 40,
        minP: 0.05,
        repeatPenalty: 1.1,
      },
      maxTokens: { value: "", placeholder: "2048 (model default)" },
      wireApi: "completions",
      reasoningMode: "Auto",
      builtinToolsEnabled: true,
      workingDirectory: "/private/work",
      maxToolIterations: "4",
      toolResultMaxChars: "12500",
      toolCategories: {
        file: true,
        search: false,
        shell: true,
        webSearch: false,
        urlFetch: false,
        git: false,
        utilities: true,
      },
    },
    chatSettingsInteraction: {
      openedVisibly: true,
      controlsChanged: ["API Wire Format", "Tool Result Limit"],
      savedViaVisibleControl: true,
      reopenedAfterSave: true,
      persistedAfterReopen: true,
    },
    chatOverrides: {
      wireApi: "completions",
      builtinToolsEnabled: true,
      workingDirectory: "/private/work",
      maxToolIterations: 4,
      toolResultMaxChars: 12500,
    },
    workingDirectory: "/private/work",
    uiTurnEvidence: assistantIds.map((assistantMessageId, index) => ({
      turn: index + 1,
      prompt: prompts[index],
      proofRequestId: `user-${index + 1}`,
      terminalProofRequestId: `user-${index + 1}`,
      requestIds: [`wire-request-${index + 1}`],
      userMessageId: `user-${index + 1}`,
      assistantMessageId,
      terminalMessageId: assistantMessageId,
      terminalResponseId: `server-request-${index + 1}`,
      logMatchMode: "exact_identity_ring_safe",
    })),
    resolvedSamplingKwargs: {
      temperature: 0.7,
      top_p: 0.9,
      top_k: 40,
      min_p: 0.05,
      repetition_penalty: 1.1,
      max_tokens: 2048,
    },
    resolvedSamplingRecords: assistantIds.map((assistantMessageId, index) => ({
      route_model:
        "Resolved sampling kwargs route=/v1/chat/completions model=test-model",
      route: "/v1/chat/completions",
      model: "test-model",
      turn: index + 1,
      proof_request_id: `user-${index + 1}`,
      request_id: `wire-request-${index + 1}`,
      message_id: assistantMessageId,
      correlation_source: "server_emitted",
      user_message_id: `user-${index + 1}`,
      assistant_message_id: assistantMessageId,
      values: {
        temperature: 0.7,
        top_p: 0.9,
        top_k: 40,
        min_p: 0.05,
        repetition_penalty: 1.1,
        max_tokens: 2048,
      },
    })),
    requestCorrelation: {
      status: "verified",
      turns: assistantIds.map((assistantMessageId, index) => ({
        turn: index + 1,
        proofRequestId: `user-${index + 1}`,
        userMessageId: `user-${index + 1}`,
        assistantMessageId,
        serverProofRequestId: `user-${index + 1}`,
        serverRequestIds: [`wire-request-${index + 1}`],
        serverMessageId: assistantMessageId,
        resolvedLogCorrelated: true,
        cacheObservationCorrelated: true,
      })),
    },
    serverCacheControls: {
      runningSessionDrawer: true,
      controlScope: "running-session-toolbar",
      visibleBlockDiskChecked: true,
      initialCacheControls: {
        enablePrefixCache: true,
        usePagedCache: true,
        enableBlockDiskCache: true,
      },
      argv: ["--use-paged-cache", "--enable-block-disk-cache"],
      // Base fixture is a bundle WITHOUT MTP weights, so the Native MTP
      // control must be absent — the dead-toggle arm of the parity rule.
      nativeMtpControl: {
        labelVisible: false,
        modeSelectPresent: false,
        selectedMode: null,
        modeOptions: null,
        blockedFallbackNoticeShown: false,
        mentionedInDrawer: false,
      },
    },
    session: {
      effective_config: {
        enablePrefixCache: true,
        usePagedCache: true,
        enableBlockDiskCache: true,
      },
    },
    cacheRequestEvidence: [
      {
        turn: 1,
        proofRequestId: "user-1",
        terminalResponseId: "server-request-1",
        serverRequestId: "server-request-1",
        executionRequestId: "server-request-1",
        correlationStatus: "verified",
        userMessageId: "user-1",
        assistantMessageId: assistantIds[0],
        serverObservation: {
          proof_request_id: "user-1",
          request_id: "server-request-1",
          terminal_response_id: "server-request-1",
          message_id: assistantIds[0],
          correlation_source:
            "chat_complete_response_id_to_scheduler_last_cache_execution",
          prompt_tokens: 128,
          cached_tokens: 0,
          prefill_tokens: 128,
          cache_hit_request: 0,
          cache_hit_tokens: 0,
          cache_reuse_partial_tokens: 0,
          cache_reuse_skip_tokens: 0,
        },
        before: cacheStats({
          processed: 0,
          hitRequests: 0,
          hitTokens: 0,
          partialTokens: 0,
          skippedTokens: 0,
        }),
        after: cacheStats({
          processed: 1,
          hitRequests: 0,
          hitTokens: 0,
          partialTokens: 0,
          skippedTokens: 0,
        }),
      },
      {
        turn: 2,
        proofRequestId: "user-2",
        terminalResponseId: "server-request-2",
        serverRequestId: "server-request-2",
        executionRequestId: "server-request-2",
        correlationStatus: "verified",
        userMessageId: "user-2",
        assistantMessageId: assistantIds[1],
        serverObservation: {
          proof_request_id: "user-2",
          request_id: "server-request-2",
          terminal_response_id: "server-request-2",
          message_id: assistantIds[1],
          correlation_source:
            "chat_complete_response_id_to_scheduler_last_cache_execution",
          cache_reuse_applied: true,
          prompt_tokens: 192,
          cached_tokens: 128,
          prefill_tokens: 64,
          cache_hit_request: 1,
          cache_hit_tokens: 128,
          cache_reuse_partial_tokens: 64,
          cache_reuse_skip_tokens: 64,
        },
        before: cacheStats({
          processed: 1,
          hitRequests: 0,
          hitTokens: 0,
          partialTokens: 0,
          skippedTokens: 0,
        }),
        after: cacheStats({
          processed: 2,
          hitRequests: 1,
          hitTokens: 128,
          partialTokens: 64,
          skippedTokens: 64,
        }),
      },
      {
        turn: 3,
        proofRequestId: "user-3",
        terminalResponseId: "server-request-3",
        serverRequestId: "server-request-3",
        executionRequestId: "server-request-3",
        correlationStatus: "verified",
        userMessageId: "user-3",
        assistantMessageId: assistantIds[2],
        serverObservation: {
          proof_request_id: "user-3",
          request_id: "server-request-3",
          terminal_response_id: "server-request-3",
          message_id: assistantIds[2],
          correlation_source:
            "chat_complete_response_id_to_scheduler_last_cache_execution",
          cache_reuse_applied: true,
          prompt_tokens: 192,
          cached_tokens: 128,
          prefill_tokens: 64,
          cache_hit_request: 1,
          cache_hit_tokens: 128,
          cache_reuse_partial_tokens: 64,
          cache_reuse_skip_tokens: 64,
        },
        before: cacheStats({
          processed: 2,
          hitRequests: 1,
          hitTokens: 128,
          partialTokens: 64,
          skippedTokens: 64,
        }),
        after: cacheStats({
          processed: 3,
          hitRequests: 2,
          hitTokens: 256,
          partialTokens: 128,
          skippedTokens: 128,
        }),
      },
    ],
    server: {
      models: { data: [{ id: "test-model" }] },
      health: {
        ...healthRaw,
        effective_defaults: {
          temperature: 0.7,
          top_p: 0.9,
          top_k: 40,
          min_p: 0.05,
          repetition_penalty: 1.1,
          max_output_tokens: 2048,
        },
        native_cache: {
          prefix: true,
          paged: true,
          block_disk_only: false,
          block_disk_l2: true,
        },
        mtp: {
          index_has_mtp_tensors: false,
          mtp_tensor_count: 0,
          runtime_active: false,
          status: "not_configured",
        },
      },
    },
    uiRuntimeProvenance: {
      mode: "electron-dev",
      renderer_source_tree_sha256: sha,
      renderer_build_source_commit: sourceCommit,
      electron_executable: testExecutablePath,
      electron_executable_sha256: testExecutableSha,
      source_commit: sourceCommit,
      source_tree: "tree",
      vite_renderer_source_seen: true,
      vite_client_seen: true,
      cdp_process_binding: {
        launched_root_pid: 1200,
        process_tree_pids: [1200, 1234],
        listener_pid: 1234,
        belongs_to_launched_process_tree: true,
        executable_path: testExecutablePath,
        executable_sha256: testExecutableSha,
        executable_path_fingerprint_sha256: testExecutablePathSha,
      },
      backend_python_process_binding: {
        listener_pid: 4321,
        health_pid: 4321,
        invoked_executable_path: testExecutablePath,
        invoked_executable_path_fingerprint_sha256: testExecutablePathSha,
        executable_path: testExecutablePath,
        executable_sha256: testExecutableSha,
        executable_path_fingerprint_sha256: testExecutablePathSha,
      },
      served_renderer_modules: modules,
      served_renderer_source_sha256: canonicalHash(modules),
    },
    gitProvenance: {
      before: {
        renderer_source_tree_sha256: sha,
        commit: sourceCommit,
        tree: "tree",
        server_module_sha256: sha,
        package_init_sha256: sha,
        python_source_tree_sha256: sha,
        python_source_file_count: 10,
        python_source_read_error_count: 0,
      },
      after: {
        renderer_source_tree_sha256: sha,
        commit: sourceCommit,
        tree: "tree",
        server_module_sha256: sha,
        package_init_sha256: sha,
        python_source_tree_sha256: sha,
        python_source_file_count: 10,
        python_source_read_error_count: 0,
      },
    },
    healthProvenance: {
      before: { raw: structuredClone(healthRaw), binding: runtimeBinding },
      after: { raw: structuredClone(healthRaw), binding: runtimeBinding },
    },
  };
  return result;
}

function dsv4EnabledResult(): Record<string, any> {
  const result = goodResult();
  result.expectedDsv4PoolQuant = true;
  result.serverCacheControls = {
    runningSessionDrawer: true,
    controlScope: "running-session-toolbar",
    verified: true,
    initialCacheControls: {
      enablePrefixCache: true,
      usePagedCache: false,
      enableBlockDiskCache: true,
    },
    argv: ["--no-paged-cache", "--enable-block-disk-cache"],
    nativeMtpControl: {
      labelVisible: false,
      modeSelectPresent: false,
      selectedMode: null,
      modeOptions: null,
      blockedFallbackNoticeShown: false,
      mentionedInDrawer: false,
    },
  };
  result.session.effective_config = {
    ...result.session.effective_config,
    enablePrefixCache: true,
    usePagedCache: false,
    enableBlockDiskCache: true,
  };
  result.server.health.native_cache = {
    prefix: true,
    paged: false,
    block_disk_only: true,
    block_disk_l2: true,
    pool_quant: {
      requested: true,
      enabled: true,
      observed: true,
      matches_request: true,
      error: null,
    },
  };
  return result;
}

function sse(...objects: Array<Record<string, unknown> | "[DONE]">) {
  return objects
    .map((object) =>
      object === "[DONE]"
        ? "data: [DONE]\n\n"
        : `data: ${JSON.stringify(object)}\n\n`,
    )
    .join("");
}

const pairedToolParameters: Record<string, Record<string, unknown>> = {
  file_info: {
    type: "object",
    properties: { path: { type: "string" } },
    required: ["path"],
    additionalProperties: false,
  },
  run_command: {
    type: "object",
    properties: { command: { type: "string" } },
    required: ["command"],
    additionalProperties: false,
  },
};
const pairedCaptureSemantics = [
  "Exact decompressed response-body bytes delivered to protocol parsers: ",
  "streaming bytes before requests.iter_lines line splitting or Unicode ",
  "decoding, and nonstream response bytes before JSON decoding; excludes ",
  "HTTP transfer framing and compressed transport octets.",
].join("");

function expectedTerminals(protocol: string, mode: string, round: number) {
  const toolRound = round < 3;
  if (protocol === "chat") {
    return [toolRound ? "tool_calls" : "stop", ...(mode === "stream" ? ["DONE"] : [])];
  }
  if (protocol === "responses") return ["response.completed"];
  if (protocol === "anthropic") {
    return [toolRound ? "tool_use" : "end_turn", ...(mode === "stream" ? ["message_stop"] : [])];
  }
  return [toolRound ? "tool_calls" : "stop"];
}

function toolContracts(protocol: string, stage: number) {
  const names = protocol === "ollama"
    ? stage === 1
      ? ["file_info"]
      : stage === 2
        ? ["run_command"]
        : []
    : ["file_info", "run_command"];
  return names.map((name) => ({
    name,
    parameters: structuredClone(pairedToolParameters[name]),
  }));
}

function fixtureToolChoice(protocol: string, mode: string, stage: number) {
  if (protocol === "ollama") return null;
  if (stage === 3) {
    return protocol === "anthropic" ? { type: "none" } : "none";
  }
  const name = stage === 1 ? "file_info" : "run_command";
  if (stage === 1 && mode !== "stream") {
    return protocol === "anthropic" ? { type: "any" } : "required";
  }
  if (protocol === "chat") {
    return { type: "function", function: { name } };
  }
  if (protocol === "responses") {
    return { type: "function", name };
  }
  return { type: "tool", name };
}

function rawRound(
  protocol: string,
  round: number,
  callId: string,
  expectedFinal: string,
) {
  const reasoning = round < 3 ? `Reason ${protocol} ${round} A.B.` : "";
  const content = round === 3 ? expectedFinal : "";
  const toolName = round === 1 ? "file_info" : "run_command";
  const args = round === 1
    ? { path: "panel/package.json" }
    : { command: "pwd" };
  if (protocol === "chat") {
    return sse(
      ...(round < 3
        ? [
            { choices: [{ delta: { reasoning_content: `Reason ${protocol} ${round} A.` } }] },
            { choices: [{ delta: { reasoning_content: "B." } }] },
            {
              choices: [{
                delta: {
                  tool_calls: [{
                    index: 0,
                    id: callId,
                    function: { name: toolName, arguments: JSON.stringify(args) },
                  }],
                },
              }],
            },
          ]
        : [
            { choices: [{ delta: { content: content.slice(0, 10) } }] },
            { choices: [{ delta: { content: content.slice(10) } }] },
          ]),
      { choices: [{ delta: {}, finish_reason: round < 3 ? "tool_calls" : "stop" }] },
      "[DONE]",
    );
  }
  if (protocol === "responses") {
    return sse(
      ...(round < 3
        ? [
            { type: "response.reasoning_summary_text.delta", delta: `Reason ${protocol} ${round} A.` },
            { type: "response.reasoning_summary_text.delta", delta: "B." },
            {
              type: "response.output_item.added",
              output_index: 0,
              item: {
                type: "function_call",
                id: `item-${callId}`,
                call_id: callId,
                name: toolName,
                arguments: "",
              },
            },
            {
              type: "response.function_call_arguments.delta",
              item_id: `item-${callId}`,
              call_id: callId,
              delta: JSON.stringify(args),
            },
            {
              type: "response.function_call_arguments.done",
              item_id: `item-${callId}`,
              call_id: callId,
              name: toolName,
              arguments: JSON.stringify(args),
            },
          ]
        : [
            { type: "response.output_text.delta", delta: content.slice(0, 10) },
            { type: "response.output_text.delta", delta: content.slice(10) },
          ]),
      { type: "response.completed", response: { status: "completed" } },
    );
  }
  if (protocol === "anthropic") {
    return sse(
      ...(round < 3
        ? [
            { type: "content_block_delta", delta: { type: "thinking_delta", thinking: `Reason ${protocol} ${round} A.` } },
            { type: "content_block_delta", delta: { type: "thinking_delta", thinking: "B." } },
            {
              type: "content_block_start",
              index: 0,
              content_block: {
                type: "tool_use",
                id: callId,
                name: toolName,
                input: args,
              },
            },
          ]
        : [
            { type: "content_block_delta", delta: { type: "text_delta", text: content.slice(0, 10) } },
            { type: "content_block_delta", delta: { type: "text_delta", text: content.slice(10) } },
          ]),
      { type: "message_delta", delta: { stop_reason: round < 3 ? "tool_use" : "end_turn" } },
      { type: "message_stop" },
    );
  }
  const rows = round < 3
    ? [
        { message: { thinking: `Reason ${protocol} ${round} A.` } },
        { message: { thinking: "B." } },
        {
          message: {
            tool_calls: [{
              id: callId,
              function: { name: toolName, arguments: args },
            }],
          },
        },
      ]
    : [
        { message: { content: content.slice(0, 10) } },
        { message: { content: content.slice(10) } },
      ];
  rows.push({ done: true, done_reason: round < 3 ? "tool_calls" : "stop" } as any);
  return `${rows.map((row) => JSON.stringify(row)).join("\n")}\n`;
}

function rawResponsesNonstream(
  round: number,
  callId: string,
  expectedFinal: string,
) {
  const output = round < 3
    ? [
        {
          type: "reasoning",
          summary: [
            { type: "summary_text", text: `Reason responses ${round} A.` },
            { type: "summary_text", text: "B." },
          ],
        },
        {
          type: "function_call",
          id: `item-${callId}`,
          call_id: callId,
          name: round === 1 ? "file_info" : "run_command",
          arguments: JSON.stringify(
            round === 1
              ? { path: "panel/package.json" }
              : { command: "pwd" },
          ),
        },
      ]
    : [
        {
          type: "message",
          content: [{ type: "output_text", text: expectedFinal }],
        },
      ];
  return JSON.stringify({
    id: `response-nonstream-${round}`,
    status: "completed",
    output,
  });
}

function rawProtocolNonstream(
  protocol: string,
  round: number,
  callId: string,
  expectedFinal: string,
) {
  if (protocol === "responses") {
    return rawResponsesNonstream(round, callId, expectedFinal);
  }
  const toolName = round === 1 ? "file_info" : "run_command";
  const args = round === 1
    ? { path: "panel/package.json" }
    : { command: "pwd" };
  const reasoning = `Reason ${protocol} ${round} A.B.`;
  if (protocol === "chat") {
    return JSON.stringify({
      id: `chat-nonstream-${round}`,
      choices: [{
        message: round < 3
          ? {
              reasoning_content: reasoning,
              tool_calls: [{
                id: callId,
                type: "function",
                function: {
                  name: toolName,
                  arguments: JSON.stringify(args),
                },
              }],
            }
          : { content: expectedFinal },
        finish_reason: round < 3 ? "tool_calls" : "stop",
      }],
    });
  }
  if (protocol === "anthropic") {
    return JSON.stringify({
      id: `anthropic-nonstream-${round}`,
      content: round < 3
        ? [
            { type: "thinking", thinking: reasoning },
            {
              type: "tool_use",
              id: callId,
              name: toolName,
              input: args,
            },
          ]
        : [{ type: "text", text: expectedFinal }],
      stop_reason: round < 3 ? "tool_use" : "end_turn",
    });
  }
  if (protocol === "ollama") {
    return JSON.stringify({
      message: round < 3
        ? {
            thinking: reasoning,
            tool_calls: [{
              id: callId,
              function: { name: toolName, arguments: args },
            }],
          }
        : { content: expectedFinal },
      done: true,
      done_reason: round < 3 ? "tool_calls" : "stop",
    });
  }
  throw new Error(`unsupported nonstream fixture protocol ${protocol}`);
}

function createValidPairedArtifact(result: Record<string, any>) {
  const binding = result.healthProvenance.after.binding;
  const directory = mkdtempSync(path.join(tmpdir(), "vmlx-paired-api-v2-"));
  const repoRoot = realpathSync(path.resolve(process.cwd(), ".."));
  const runDirectory = path.join(directory, result.run_id);
  mkdirSync(runDirectory, { mode: 0o700 });
  const protocolNames = ["chat", "responses", "anthropic", "ollama"];
  const modeNames = ["stream", "nonstream"];
  const flows: Record<string, any> = { direct: {}, gateway: {} };
  const manifestRoutes: Array<Record<string, any>> = [];
  let sequence = 0;
  for (const baseLabel of ["direct", "gateway"]) {
    for (const protocol of protocolNames) {
      flows[baseLabel][protocol] = {};
      for (const mode of modeNames) {
        const executionOutputs = [
          JSON.stringify({
            path: "panel/package.json",
            size_human: "5.2 KB",
          }),
          JSON.stringify({ stdout: repoRoot }),
        ];
        const executions = [
          {
            name: "file_info",
            call_id: `${baseLabel}-${protocol}-${mode}-file`,
            arguments: { path: "panel/package.json" },
            result: { path: "panel/package.json", size_human: "5.2 KB" },
            output_chars: executionOutputs[0].length,
            output_sha256: crypto.createHash("sha256").update(executionOutputs[0]).digest("hex"),
          },
          {
            name: "run_command",
            call_id: `${baseLabel}-${protocol}-${mode}-pwd`,
            arguments: { command: "pwd" },
            result: { stdout: repoRoot },
            output_chars: executionOutputs[1].length,
            output_sha256: crypto.createHash("sha256").update(executionOutputs[1]).digest("hex"),
          },
        ];
        const expectedFinal =
          `AGENTIC-${protocol.toUpperCase()}-${mode.toUpperCase()}-DONE SIZE=5.2 KB PWD=${repoRoot}`;
        const requests = [1, 2, 3].map((stage) => {
          const bodyIdentity = { baseLabel, protocol, mode, stage };
          const canonicalBodyIdentity = { protocol, mode, stage };
          const request: Record<string, any> = {
            stage,
            body_chars: canonicalJson(bodyIdentity).length,
            body_sha256: canonicalHash(bodyIdentity),
            canonical_body_sha256: canonicalHash(canonicalBodyIdentity),
            tool_choice: fixtureToolChoice(protocol, mode, stage),
            stream: mode === "stream",
            enable_thinking: stage === 1 ? null : stage === 2,
            previous_response_id: (
              protocol === "responses" && stage > 1
                ? `${baseLabel}-${protocol}-${mode}-${stage - 1}`
                : null
            ),
            max_output_tokens: 512,
            tool_contracts: toolContracts(protocol, stage),
            tool_history_linkage: [],
          };
          if (stage >= 2 && protocol === "responses") {
            const prior = executions[stage - 2];
            request.tool_history_linkage.push({
              kind: "tool_result",
              role: "function_call_output",
              call_id: prior.call_id,
              output_chars: prior.output_chars,
              output_sha256: prior.output_sha256,
            });
          } else if (stage >= 2) {
            for (const prior of executions.slice(0, stage - 1)) {
              request.tool_history_linkage.push(
                {
                  kind: "assistant_tool_call",
                  role: "assistant",
                  call_id: protocol === "ollama" ? "" : prior.call_id,
                  name: prior.name,
                },
                {
                  kind: "tool_result",
                  role: protocol === "anthropic" ? "user" : "tool",
                  call_id: protocol === "ollama" ? "" : prior.call_id,
                  ...(protocol === "ollama" ? { name: prior.name } : {}),
                  output_chars: prior.output_chars,
                  output_sha256: prior.output_sha256,
                },
              );
            }
          }
          return request;
        });
        const rounds = [1, 2, 3].map((roundNumber) => {
          const reasoning = roundNumber < 3 ? `Reason ${protocol} ${roundNumber} A.B.` : "";
          const content = roundNumber === 3 ? expectedFinal : "";
          const terminals = expectedTerminals(protocol, mode, roundNumber);
          const events: Array<Record<string, any>> = [];
          let at = 1;
          if (roundNumber < 3) {
            for (const part of [`Reason ${protocol} ${roundNumber} A.`, "B."]) {
              events.push({
                at_ms: at++,
                channel: "reasoning",
                kind: `${protocol}.reasoning.delta`,
                chars: part.length,
                sha256: crypto.createHash("sha256").update(part).digest("hex"),
              });
            }
            events.push({
              at_ms: at++,
              channel: "tool",
              kind: `${protocol}.tool`,
              call_id: executions[roundNumber - 1].call_id,
            });
          } else {
            for (const part of [content.slice(0, 10), content.slice(10)]) {
              events.push({
                at_ms: at++,
                channel: "content",
                kind: `${protocol}.content.delta`,
                chars: part.length,
                sha256: crypto.createHash("sha256").update(part).digest("hex"),
              });
            }
          }
          for (const terminal of terminals) {
            events.push({ at_ms: at++, channel: "terminal", kind: terminal });
          }
          return {
            status_code: 200,
            elapsed_ms: 10,
            response_id: `${baseLabel}-${protocol}-${mode}-${roundNumber}`,
            reasoning_chars: reasoning.length,
            reasoning_sha256: crypto.createHash("sha256").update(reasoning).digest("hex"),
            reasoning_delta_count: roundNumber < 3 ? (mode === "stream" ? 2 : 1) : 0,
            content,
            content_chars: content.length,
            content_sha256: crypto.createHash("sha256").update(content).digest("hex"),
            content_delta_count: roundNumber === 3 ? (mode === "stream" ? 2 : 1) : 0,
            tool_calls: roundNumber < 3
              ? [{
                  index: 0,
                  id: executions[roundNumber - 1].call_id,
                  name: executions[roundNumber - 1].name,
                  arguments: executions[roundNumber - 1].arguments,
                }]
              : [],
            terminals,
            errors: [],
            events,
          };
        });
        flows[baseLabel][protocol][mode] = {
          pass: true,
          checks: { exact: true },
          expected_final: expectedFinal,
          requests,
          rounds,
          executions,
          terminal_classification: rounds.map((round) => ({
            pass: true,
            values: round.terminals,
            semantic: round.terminals.filter((value: string) =>
              !["DONE", "message_stop"].includes(value)),
          })),
        };
        for (let roundNumber = 1; roundNumber <= 3; roundNumber += 1) {
          sequence += 1;
          const captureLabel = `${mode}-flow-round${roundNumber}`;
          const body = mode === "stream"
            ? rawRound(
                protocol,
                roundNumber,
                executions[roundNumber - 1]?.call_id || "",
                expectedFinal,
              )
            : rawProtocolNonstream(
                protocol,
                roundNumber,
                executions[roundNumber - 1]?.call_id || "",
                expectedFinal,
              );
          const stem = `${String(sequence).padStart(4, "0")}-${baseLabel}-${protocol}-${captureLabel}`;
          const bodyFile = `${stem}.decompressed-parser-input.bin`;
          const metadataFile = `${stem}.metadata.json`;
          const bodyPath = path.join(runDirectory, bodyFile);
          const metadataPath = path.join(runDirectory, metadataFile);
          writePrivateArtifactFile(bodyPath, body);
          const bodySha = crypto.createHash("sha256").update(body).digest("hex");
          const requestPayload = { ...requests[roundNumber - 1] };
          delete requestPayload.stage;
          const requestBodySha = canonicalHash({
            baseLabel,
            protocol,
            roundNumber,
          });
          const metadata = {
            schema_version: 1,
            capture_layer: "requests.decompressed_response_parser_input",
            capture_semantics: pairedCaptureSemantics,
            base_label: baseLabel,
            protocol,
            capture_label: captureLabel,
            request: {
              method: "POST",
              url: `${baseLabel === "direct" ? "http://127.0.0.1:8000" : "http://127.0.0.1:8080"}${{
                chat: "/v1/chat/completions",
                responses: "/v1/responses",
                anthropic: "/v1/messages",
                ollama: "/api/chat",
              }[protocol]}`,
              body_bytes: 1,
              body_sha256: requestBodySha,
              prepared_payload_body_sha256: requestPayload.body_sha256,
              prepared_payload_canonical_body_sha256:
                requestPayload.canonical_body_sha256,
              headers: [],
              payload: {
                ...requestPayload,
                model: result.servedModel,
                top_level_fields: ["model"],
              },
            },
            response: {
              status_code: 200,
              headers: [],
              body_file: bodyFile,
              body_bytes: Buffer.byteLength(body),
              body_sha256: bodySha,
            },
          };
          writePrivateArtifactFile(metadataPath, `${JSON.stringify(metadata, null, 2)}\n`);
          const metadataBytes = readFileSync(metadataPath);
          manifestRoutes.push({
            base_label: baseLabel,
            protocol,
            capture_label: captureLabel,
            expected: 1,
            started: 1,
            finished: 1,
            errors: [],
            artifacts: [{
              sequence,
              body_file: bodyFile,
              metadata_file: metadataFile,
              verified: true,
              body_bytes: Buffer.byteLength(body),
              body_sha256: bodySha,
              metadata_sha256: crypto.createHash("sha256").update(metadataBytes).digest("hex"),
              request_body_sha256: requestBodySha,
              prepared_payload_body_sha256: requestPayload.body_sha256,
              prepared_payload_canonical_body_sha256:
                requestPayload.canonical_body_sha256,
            }],
          });
        }
      }
    }
  }
  const manifest = {
    schema_version: 1,
    enabled: true,
    capture_layer: "requests.decompressed_response_parser_input",
    capture_semantics: pairedCaptureSemantics,
    run_id: result.run_id,
    expected: manifestRoutes.length,
    started: manifestRoutes.length,
    finished: manifestRoutes.length,
    errors: 0,
    complete: true,
    routes: manifestRoutes,
  };
  const manifestPath = path.join(runDirectory, "manifest.json");
  const manifestText = `${JSON.stringify(manifest, null, 2)}\n`;
  writePrivateArtifactFile(manifestPath, manifestText);
  const producerSource = realpathSync(
    path.join(repoRoot, "tests/cross_matrix/run_agentic_protocol_matrix.py"),
  );
  const producerExecutable = realpathSync(process.execPath);
  const producerExecutableBytes = readFileSync(producerExecutable);
  const producerHarnessBytes = readFileSync(producerSource);
  const source = {
    git_root: repoRoot,
    head: result.gitProvenance.after.commit,
    tree: result.gitProvenance.after.tree,
    clean: true,
    status_sha256: sha,
    server_module_sha256: result.gitProvenance.after.server_module_sha256,
    package_init_sha256: result.gitProvenance.after.package_init_sha256,
    python_source_tree_sha256: result.gitProvenance.after.python_source_tree_sha256,
    python_source_file_count: result.gitProvenance.after.python_source_file_count,
    python_source_read_error_count: result.gitProvenance.after.python_source_read_error_count,
  };
  const runner = {
    execution_mode: "source-checkout-venv",
    repo_venv: true,
    repo_python: true,
    python_executable_path: producerExecutable,
    python_executable_fingerprint_sha256:
      binding.runtime_source_hashes.python_executable_fingerprint_sha256,
    checkout_python_invocation_fingerprints_sha256: [
      binding.runtime_source_hashes.python_executable_fingerprint_sha256,
    ],
    installed_python_invocation_fingerprints_sha256: [],
    accepted_python_invocation_fingerprints_sha256: [
      binding.runtime_source_hashes.python_executable_fingerprint_sha256,
    ],
    python_prefix_path: testExecutablePrefixPath,
    python_prefix_fingerprint_sha256: testExecutablePrefixPathSha,
    producer_pid: 987654,
    producer_executable_path: producerExecutable,
    producer_executable_sha256: crypto.createHash("sha256").update(producerExecutableBytes).digest("hex"),
    producer_executable_size_bytes: producerExecutableBytes.length,
    producer_harness_relative_path: "tests/cross_matrix/run_agentic_protocol_matrix.py",
    producer_harness_path: producerSource,
    producer_harness_sha256: crypto.createHash("sha256").update(producerHarnessBytes).digest("hex"),
    producer_harness_size_bytes: producerHarnessBytes.length,
    installed_runtime: null,
  };
  const bundle = {
    ...result.bundleGenerationContract.health_attestation,
    model_name: result.servedModel,
  };
  const healthFull = result.healthProvenance.after.raw;
  const healthRow = {
    url: "http://127.0.0.1:8000/health",
    full: healthFull,
    full_canonical_json: canonicalJson(healthFull),
    full_sha256: canonicalHash(healthFull),
    identity: binding,
  };
  const rawCapture = {
    ...manifest,
    manifest_file: "manifest.json",
    manifest_path: manifestPath,
    run_directory: runDirectory,
    manifest_sha256: crypto.createHash("sha256").update(manifestText).digest("hex"),
  };
  const pairedReplays = Object.fromEntries(
    [2, 3].map((stage) => {
      const flowRequest = flows.direct.chat.nonstream.requests[stage - 1];
      const preparedBodySha = flowRequest.body_sha256;
      return [
        `chat_nonstream_round${stage}`,
        {
          schema: "vmlx-agentic-protocol-paired-replay-v1",
          target: { protocol: "chat", mode: "nonstream", stage },
          request: {
            body_sha256: preparedBodySha,
            enable_thinking: stage === 2,
            prepared_body_sha256: preparedBodySha,
            leg_body_sha256: {
              a1: preparedBodySha,
              b: preparedBodySha,
              a2: preparedBodySha,
            },
          },
          checks: {
            exact_body_sha_equal: true,
            gateway_backend_lifecycle_pass: true,
          },
          pass: true,
        },
      ];
    }),
  );
  const value = {
    schema: "vmlx-agentic-protocol-matrix-v2",
    schema_version: 2,
    run_id: result.run_id,
    requested_model: result.servedModel,
    repo_root: repoRoot,
    bases: {
      direct: "http://127.0.0.1:8000",
      gateway: "http://127.0.0.1:8080",
    },
    protocols: protocolNames,
    modes: modeNames,
    second_tool_choice: "explicit",
    backend_identity_fingerprint_sha256: binding.fingerprint_sha256,
    identity: {
      source: {
        declared_head: source.head,
        before: source,
        after: structuredClone(source),
      },
      runner: {
        before: runner,
        after: structuredClone(runner),
      },
      bundle: {
        before: bundle,
        after: structuredClone(bundle),
      },
      health: {
        direct: {
          before: healthRow,
          after: structuredClone(healthRow),
        },
        gateway: {
          before: {
            ...structuredClone(healthRow),
            url: "http://127.0.0.1:8080/health",
          },
          after: {
            ...structuredClone(healthRow),
            url: "http://127.0.0.1:8080/health",
          },
        },
      },
      failures: [],
    },
    raw_capture: rawCapture,
    flows,
    paired_replays: pairedReplays,
    abort_recovery: {},
    checks: {
      identity_provenance_pass: true,
      all_requested_flows_present: true,
      all_flows_pass: true,
      abort_recovery_skipped: true,
      all_abort_recovery_pass: true,
      paired_replay_chat_nonstream_rounds_pass: true,
      raw_capture_complete: true,
    },
    pass: true,
  };
  const artifactPath = path.join(directory, "paired-api-proof.json");
  writePrivateArtifactFile(artifactPath, JSON.stringify(value));
  return {
    directory,
    artifact: readPrivateExternalJson(
      artifactPath,
      "Paired raw API proof artifact fixture",
    ),
  };
}

function writePairedArtifactValue(
  directory: string,
  name: string,
  value: Record<string, unknown>,
) {
  const artifactPath = path.join(directory, name);
  writePrivateArtifactFile(artifactPath, JSON.stringify(value));
  return readPrivateExternalJson(artifactPath, `Paired fixture ${name}`);
}

function createInstalledPairedArtifact(result: Record<string, any>) {
  const fixture = createValidPairedArtifact(result);
  const value = structuredClone(fixture.artifact.value);
  const appPath = path.join(fixture.directory, "installed", "vMLX.app");
  const resources = path.join(appPath, "Contents", "Resources");
  const pythonPrefix = path.join(resources, "bundled-python", "python");
  const pythonPath = path.join(pythonPrefix, "bin", "python3");
  const pythonTargetPath = path.join(pythonPrefix, "bin", "python3.12");
  const electronPath = path.join(appPath, "Contents", "MacOS", "vMLX");
  const appAsarPath = path.join(resources, "app.asar");
  const bundledProvenancePath = path.join(
    resources,
    "bundled-python",
    "vmlx-bundle-provenance.json",
  );
  mkdirSync(path.dirname(pythonPath), { recursive: true });
  mkdirSync(path.dirname(electronPath), { recursive: true });
  writeFileSync(pythonTargetPath, testExecutableBytes);
  chmodSync(pythonTargetPath, 0o755);
  symlinkSync("python3.12", pythonPath);
  writeFileSync(electronPath, testExecutableBytes);
  chmodSync(electronPath, 0o755);
  writeFileSync(appAsarPath, "installed-renderer-asar\n");
  const bundledProvenance = {
    schema_version: 1,
    vmlx: {
      commit: result.gitProvenance.after.commit,
      version: "1.6.19",
    },
  };
  writeFileSync(
    bundledProvenancePath,
    `${JSON.stringify(bundledProvenance)}\n`,
  );
  const fileIdentity = (filePath: string) => {
    const bytes = readFileSync(filePath);
    return {
      path: realpathSync(filePath),
      requested_path: filePath,
      sha256: crypto.createHash("sha256").update(bytes).digest("hex"),
      size_bytes: bytes.length,
      opened_nofollow: true,
    };
  };
  const pythonIdentity = fileIdentity(pythonPath);
  const canonicalPythonPath = pythonIdentity.path;
  const canonicalPythonPrefix = realpathSync(pythonPrefix);
  const appAsarIdentity = fileIdentity(appAsarPath);
  const electronIdentity = fileIdentity(electronPath);
  const provenanceIdentity = fileIdentity(bundledProvenancePath);
  const pythonPathSha = crypto
    .createHash("sha256")
    .update(canonicalPythonPath)
    .digest("hex");
  const installedPythonFingerprints = [...new Set([
    pythonPathSha,
    crypto.createHash("sha256").update(pythonPath).digest("hex"),
  ])].sort();
  const pythonPrefixSha = crypto
    .createHash("sha256")
    .update(canonicalPythonPrefix)
    .digest("hex");
  const manifest = {
    schema: "vmlx-installed-release-manifest-v1",
    source_commit: result.gitProvenance.after.commit,
    source_tree: result.gitProvenance.after.tree,
    app_asar_sha256: appAsarIdentity.sha256,
    electron_executable_sha256: electronIdentity.sha256,
    bundled_provenance_sha256: provenanceIdentity.sha256,
    bundled_python_executable_sha256: pythonIdentity.sha256,
    bundled_python_executable_fingerprint_sha256: pythonPathSha,
  };
  const manifestPath = path.join(
    fixture.directory,
    "installed-release-manifest.json",
  );
  writePrivateArtifactFile(manifestPath, `${JSON.stringify(manifest)}\n`);
  const openedManifest = readPrivateExternalJson(
    manifestPath,
    "Installed release manifest fixture",
  );
  const source = value.identity.source.after;
  const sourceBinding = {
    head: source.head,
    tree: source.tree,
    server_module_sha256: source.server_module_sha256,
    package_init_sha256: source.package_init_sha256,
    python_source_tree_sha256: source.python_source_tree_sha256,
    python_source_file_count: source.python_source_file_count,
    python_source_read_error_count: source.python_source_read_error_count,
  };
  const bundledSource = {
    server_module_sha256: source.server_module_sha256,
    package_init_sha256: source.package_init_sha256,
    python_source_tree_sha256: source.python_source_tree_sha256,
    python_source_file_count: source.python_source_file_count,
    python_source_read_error_count: source.python_source_read_error_count,
  };
  const runner = {
    ...value.identity.runner.before,
    execution_mode: "installed-runtime",
    repo_venv: false,
    repo_python: false,
    python_executable_path: canonicalPythonPath,
    python_executable_fingerprint_sha256: pythonPathSha,
    checkout_python_invocation_fingerprints_sha256: [],
    installed_python_invocation_fingerprints_sha256: installedPythonFingerprints,
    accepted_python_invocation_fingerprints_sha256: installedPythonFingerprints,
    python_prefix_path: canonicalPythonPrefix,
    python_prefix_fingerprint_sha256: pythonPrefixSha,
    producer_executable_path: pythonIdentity.path,
    producer_executable_sha256: pythonIdentity.sha256,
    producer_executable_size_bytes: pythonIdentity.size_bytes,
    installed_runtime: {
      schema: "vmlx-agentic-installed-runtime-v1",
      manifest,
      manifest_path: openedManifest.path,
      manifest_sha256: openedManifest.sha256,
      manifest_size_bytes: openedManifest.bytes,
      manifest_nlink: openedManifest.nlink,
      manifest_opened_nofollow: true,
      app_path: appPath,
      invoked_python_path: canonicalPythonPath,
      invoked_python_fingerprint_sha256: pythonPathSha,
      python_prefix_path: canonicalPythonPrefix,
      bundled_python: {
        path: pythonIdentity.path,
        sha256: pythonIdentity.sha256,
        size_bytes: pythonIdentity.size_bytes,
      },
      artifacts: {
        app_asar: appAsarIdentity,
        electron_executable: electronIdentity,
        bundled_provenance: provenanceIdentity,
      },
      bundled_provenance: bundledProvenance,
      bundled_source: bundledSource,
      source_binding: sourceBinding,
    },
  };
  value.identity.runner.before = runner;
  value.identity.runner.after = structuredClone(runner);

  const setHealthPythonFingerprint = (row: Record<string, any>) => {
    row.identity.runtime_source_hashes.python_executable_fingerprint_sha256 =
      pythonPathSha;
  };
  for (const baseLabel of ["direct", "gateway"]) {
    for (const phase of ["before", "after"]) {
      setHealthPythonFingerprint(value.identity.health[baseLabel][phase]);
    }
  }
  for (const phase of ["before", "after"]) {
    result.healthProvenance[phase].binding.runtime_source_hashes
      .python_executable_fingerprint_sha256 = pythonPathSha;
  }
  result.installedAppPath = appPath;
  result.uiRuntimeProvenance = {
    ...result.uiRuntimeProvenance,
    mode: "installed-app",
    electron_executable: electronPath,
    electron_executable_sha256: electronIdentity.sha256,
    cdp_process_binding: {
      ...result.uiRuntimeProvenance.cdp_process_binding,
      executable_path: electronIdentity.path,
      executable_sha256: electronIdentity.sha256,
      executable_path_fingerprint_sha256: crypto
        .createHash("sha256")
        .update(electronIdentity.path)
        .digest("hex"),
    },
    backend_python_process_binding: {
      ...result.uiRuntimeProvenance.backend_python_process_binding,
      invoked_executable_path: pythonPath,
      invoked_executable_path_fingerprint_sha256: crypto
        .createHash("sha256")
        .update(pythonPath)
        .digest("hex"),
      executable_path: pythonIdentity.path,
      executable_sha256: pythonIdentity.sha256,
      executable_path_fingerprint_sha256: crypto
        .createHash("sha256")
        .update(pythonIdentity.path)
        .digest("hex"),
    },
    app_asar: appAsarPath,
    app_asar_sha256: appAsarIdentity.sha256,
    external_release_manifest_path: openedManifest.path,
    external_release_manifest_sha256: openedManifest.sha256,
    external_release_manifest: manifest,
    bundled_provenance_path: bundledProvenancePath,
    bundled_provenance_sha256: provenanceIdentity.sha256,
    bundled_provenance: bundledProvenance,
    bundled_provenance_error: null,
    bundled_source_root: path.join(
      resources,
      "vmlx-engine-source",
      "vmlx_engine",
    ),
    bundled_source: bundledSource,
  };
  return {
    ...fixture,
    appPath,
    manifestPath,
    pythonPath,
    pythonTargetPath,
    artifact: writePairedArtifactValue(
      fixture.directory,
      "installed-paired-api-proof.json",
      value,
    ),
  };
}

function refreshRawCaptureManifest(value: Record<string, any>) {
  const rawCapture = value.raw_capture;
  const manifest = structuredClone(rawCapture);
  for (const field of [
    "manifest_file",
    "manifest_path",
    "manifest_sha256",
    "run_directory",
  ]) delete manifest[field];
  const manifestText = `${JSON.stringify(manifest, null, 2)}\n`;
  writeFileSync(rawCapture.manifest_path, manifestText);
  rawCapture.manifest_sha256 = crypto
    .createHash("sha256")
    .update(manifestText)
    .digest("hex");
}

function restrictPairedArtifactToContract(
  value: Record<string, any>,
  protocols: string[],
  modes: string[],
) {
  value.protocols = [...protocols];
  value.modes = [...modes];
  for (const baseLabel of ["direct", "gateway"]) {
    for (const protocol of Object.keys(value.flows[baseLabel])) {
      if (!protocols.includes(protocol)) {
        delete value.flows[baseLabel][protocol];
        continue;
      }
      for (const mode of Object.keys(value.flows[baseLabel][protocol])) {
        if (!modes.includes(mode)) {
          delete value.flows[baseLabel][protocol][mode];
        }
      }
    }
  }
  value.paired_replays = {};
  value.raw_capture.routes = value.raw_capture.routes.filter(
    (route: Record<string, any>) => {
      const mode = String(route.capture_label || "").startsWith("nonstream-")
        ? "nonstream"
        : "stream";
      return protocols.includes(route.protocol) && modes.includes(mode);
    },
  );
  for (const field of ["expected", "started", "finished"]) {
    value.raw_capture[field] = value.raw_capture.routes.length;
  }
  refreshRawCaptureManifest(value);
}

function rewriteRawCaptureBody(
  value: Record<string, any>,
  {
    baseLabel,
    protocol,
    captureLabel,
    body,
  }: {
    baseLabel: string;
    protocol: string;
    captureLabel: string;
    body: string;
  },
) {
  const rawCapture = value.raw_capture;
  const route = rawCapture.routes.find(
    (row: Record<string, any>) =>
      row.base_label === baseLabel
      && row.protocol === protocol
      && row.capture_label === captureLabel,
  );
  if (!route) throw new Error("raw capture fixture route not found");
  const artifact = route.artifacts[0];
  const bodyPath = path.join(rawCapture.run_directory, artifact.body_file);
  const metadataPath = path.join(
    rawCapture.run_directory,
    artifact.metadata_file,
  );
  writeFileSync(bodyPath, body);
  artifact.body_bytes = Buffer.byteLength(body);
  artifact.body_sha256 = crypto
    .createHash("sha256")
    .update(body)
    .digest("hex");
  const metadata = JSON.parse(readFileSync(metadataPath, "utf8"));
  metadata.response.body_bytes = artifact.body_bytes;
  metadata.response.body_sha256 = artifact.body_sha256;
  const metadataText = `${JSON.stringify(metadata, null, 2)}\n`;
  writeFileSync(metadataPath, metadataText);
  artifact.metadata_sha256 = crypto
    .createHash("sha256")
    .update(metadataText)
    .digest("hex");

  const manifest = structuredClone(rawCapture);
  for (const field of [
    "manifest_file",
    "manifest_path",
    "manifest_sha256",
    "run_directory",
  ]) delete manifest[field];
  const manifestText = `${JSON.stringify(manifest, null, 2)}\n`;
  writeFileSync(rawCapture.manifest_path, manifestText);
  rawCapture.manifest_sha256 = crypto
    .createHash("sha256")
    .update(manifestText)
    .digest("hex");
}

function rewriteRawCaptureMetadata(
  value: Record<string, any>,
  {
    baseLabel,
    protocol,
    captureLabel,
    mutate,
  }: {
    baseLabel: string;
    protocol: string;
    captureLabel: string;
    mutate: (metadata: Record<string, any>, artifact: Record<string, any>) => void;
  },
) {
  const rawCapture = value.raw_capture;
  const route = rawCapture.routes.find(
    (row: Record<string, any>) =>
      row.base_label === baseLabel
      && row.protocol === protocol
      && row.capture_label === captureLabel,
  );
  if (!route) throw new Error("raw capture fixture route not found");
  const artifact = route.artifacts[0];
  const metadataPath = path.join(
    rawCapture.run_directory,
    artifact.metadata_file,
  );
  const metadata = JSON.parse(readFileSync(metadataPath, "utf8"));
  mutate(metadata, artifact);
  const metadataText = `${JSON.stringify(metadata, null, 2)}\n`;
  writeFileSync(metadataPath, metadataText);
  artifact.metadata_sha256 = crypto
    .createHash("sha256")
    .update(metadataText)
    .digest("hex");

  const manifest = structuredClone(rawCapture);
  for (const field of [
    "manifest_file",
    "manifest_path",
    "manifest_sha256",
    "run_directory",
  ]) delete manifest[field];
  const manifestText = `${JSON.stringify(manifest, null, 2)}\n`;
  writeFileSync(rawCapture.manifest_path, manifestText);
  rawCapture.manifest_sha256 = crypto
    .createHash("sha256")
    .update(manifestText)
    .digest("hex");
}

describe("real UI model proof harness", () => {
  it("keeps raw capture semantics byte-exact with the Python producer", () => {
    const exactFragment =
      "decoding, and nonstream response bytes before JSON decoding; excludes ";
    const attestorSource = readFileSync(
      path.resolve(process.cwd(), "scripts/live-real-ui-model-proof.mjs"),
      "utf8",
    );
    const producerSource = readFileSync(
      path.resolve(
        process.cwd(),
        "../tests/cross_matrix/run_agentic_protocol_matrix.py",
      ),
      "utf8",
    );
    expect(pairedCaptureSemantics).toContain(exactFragment);
    expect(attestorSource).toContain(`'${exactFragment}'`);
    expect(producerSource).toContain(`"${exactFragment}"`);
    expect(attestorSource).not.toContain("Responses nonstream bytes");
  });

  it("maps renderer paths to distinct raw Vite filesystem module identities", () => {
    const panelRoot = path.resolve(new URL("..", import.meta.url).pathname);
    expect(
      viteRawRendererModulePath(
        "src/renderer/src/main.tsx",
        "r19 proof/one",
      ),
    ).toBe(
      `/@fs${panelRoot}/src/renderer/src/main.tsx?raw&vmlx_proof=r19%20proof%2Fone`,
    );
    expect(
      viteRawRendererModulePath(
        "src/renderer/src/components/chat/ReasoningBox.tsx",
        "r19-two",
      ),
    ).toBe(
      `/@fs${panelRoot}/src/renderer/src/components/chat/ReasoningBox.tsx?raw&vmlx_proof=r19-two`,
    );
  });

  it("recognizes the renderer entry at the Vite renderer-root URL", () => {
    expect(
      viteRendererSourceSeen([
        "http://127.0.0.1:5173/@vite/client",
        "http://127.0.0.1:5173/src/main.tsx",
      ]),
    ).toBe(true);
    expect(
      viteRendererSourceSeen([
        "http://127.0.0.1:5173/@vite/client",
        "http://127.0.0.1:5173/src/main.css",
      ]),
    ).toBe(false);
  });

  it("opens Server settings from the active session toolbar, not app mode", () => {
    const source = readFileSync(
      path.resolve("scripts/live-real-ui-model-proof.mjs"),
      "utf8",
    );
    expect(source).toContain(
      "'[data-vmlx-control=\"server-settings\"]'",
    );
    expect(source).toContain(
      "'[data-vmlx-surface=\"server-settings\"]'",
    );
    expect(source).toContain(
      "controlScope: 'running-session-toolbar'",
    );
    expect(source).not.toContain("visibleChatSettings?.parentElement");
    expect(source).not.toContain(
      "[...document.querySelectorAll('button')].find((button) =>\n"
        + "              (button.textContent || '').replace(/\\\\s+/g, ' ').trim() === 'Server'",
    );
  });

  it("rejects renderer proof paths outside the served renderer root", () => {
    expect(() => viteRawRendererModulePath("src/main.tsx", "r19")).toThrow(
      /outside the Vite renderer root/,
    );
    expect(() =>
      viteRawRendererModulePath("src/renderer/../main.tsx", "r19"),
    ).toThrow(/unsafe served path/);
  });

  it("uses only a validated private token-file path in session launch args", () => {
    const root = mkdtempSync(path.join(tmpdir(), "vmlx-private-proof-"));
    try {
      const token = `private_${"q".repeat(48)}`;
      const tokenPath = path.join(root, "cache-attestation.token");
      writeFileSync(tokenPath, token, { mode: 0o600 });
      chmodSync(tokenPath, 0o600);
      const args = privateCacheAttestationSessionArgs(tokenPath);
      expect(args).toBe(
        "--enable-private-cache-attestation "
          + `--private-cache-attestation-token-file=${realpathSync(tokenPath)}`,
      );
      expect(args).not.toContain(token);

      chmodSync(tokenPath, 0o640);
      expect(() => privateCacheAttestationSessionArgs(tokenPath)).toThrow(
        /0600|group\/world/,
      );
      chmodSync(tokenPath, 0o600);
      const linkPath = path.join(root, "cache-attestation-link.token");
      symlinkSync(tokenPath, linkPath);
      expect(() => privateCacheAttestationSessionArgs(linkPath)).toThrow(
        /regular, non-symlink/,
      );
    } finally {
      rmSync(root, { recursive: true, force: true });
    }
  });

  it("uses model/run-specific exclusive private artifacts and bundle-owned defaults", () => {
    expect(
      uniqueProofBasename({
        requested: "ui-proof",
        model: "Gemma 4/MoE",
        run: "run 42",
      }),
    ).toBe(
      `ui-proof-Gemma-4-MoE-run-42-${canonicalHash({
        requested: "ui-proof",
        model: "Gemma 4/MoE",
        run: "run 42",
      }).slice(0, 20)}`,
    );
    expect(
      resolveIndependentBundleGenerationDefaults(
        {
          temperature: 0.2,
          top_p: 0.8,
          top_k: 20,
          repetition_penalty: 1.02,
          max_new_tokens: 1024,
        },
        {
          chat: {
            reasoning: { default_mode: "thinking" },
            sampling_defaults: {
              temperature: 0.7,
              top_p: 0.9,
              top_k: 40,
              min_p: 0.05,
              repetition_penalty_thinking: 1.1,
              max_new_tokens: 2048,
            },
          },
        },
        { model_type: "laguna" },
      ),
    ).toEqual(defaults);

    const directory = mkdtempSync(path.join(tmpdir(), "vmlx-private-proof-"));
    try {
      const artifact = path.join(directory, "proof.json");
      writePrivateArtifactFile(artifact, "{}");
      expect(statSync(artifact).mode & 0o777).toBe(0o600);
      expect(readFileSync(artifact, "utf8")).toBe("{}");
      expect(() => writePrivateArtifactFile(artifact, "forged")).toThrow();
    } finally {
      rmSync(directory, { recursive: true, force: true });
    }
  });

  it("retains bundle config hashes and accepts only a recognized native encoder without a template", () => {
    const bundle = mkdtempSync(path.join(tmpdir(), "vmlx-ui-proof-bundle-"));
    try {
      writeFileSync(
        path.join(bundle, "config.json"),
        JSON.stringify({ model_type: "laguna" }),
      );
      writeFileSync(
        path.join(bundle, "generation_config.json"),
        JSON.stringify({ temperature: 0.7, max_new_tokens: 2048 }),
      );
      writeFileSync(
        path.join(bundle, "tokenizer_config.json"),
        JSON.stringify({
          chat_template: "{% include 'chat_template.jinja' %}",
        }),
      );
      const missing = captureBundleGenerationContract(bundle);
      expect(missing.template.usable).toBe(false);
      expect(missing.template.warning).toMatch(/neither a usable chat template nor a recognized native encoder/i);
      expect(missing.files["config.json"].sha256).toMatch(/^[0-9a-f]{64}$/);
      expect(missing.health_attestation.schema).toBe("vmlx-bundle-config-v1");

      writeFileSync(
        path.join(bundle, "config.json"),
        JSON.stringify({ model_type: "deepseek_v4" }),
      );
      writeFileSync(
        path.join(bundle, "jang_config.json"),
        JSON.stringify({
          chat: {
            encoder: "encoding_dsv4",
            encoder_fn: "encode_messages",
            chat_template_source: "official_python_encoder",
          },
        }),
      );
      const native = captureBundleGenerationContract(bundle);
      expect(native.template.usable).toBe(true);
      expect(native.template.mode).toBe("native_encoder");
      expect(native.template.native_encoder).toEqual({
        encoder: "encoding_dsv4",
        encoder_fn: "encode_messages",
        source: "official_python_encoder",
      });
      expect(native.template.warning).toBeNull();

      writeFileSync(path.join(bundle, "chat_template.jinja"), "{{ messages }}");
      const usable = captureBundleGenerationContract(bundle);
      expect(usable.template.usable).toBe(true);
      expect(usable.template.mode).toBe("jinja");
      expect(usable.template.warning).toBeNull();
    } finally {
      rmSync(bundle, { recursive: true, force: true });
    }
  });

  it("accepts fully linked UI evidence across all nine hardened contracts", () => {
    const result = goodResult();
    expect(validateRenderedDomEvidence(result)).toEqual([]);
    expect(validateReasoningEvidence(result, "required")).toEqual([]);
    expect(validateExactToolLoopEvidence(result)).toEqual([]);
    expect(validateGenerationDefaultsEvidence(result)).toEqual([]);
    expect(validateRequestCorrelatedCacheEvidence(result)).toEqual([]);
    expect(validateServerCacheEvidence(result)).toEqual([]);
    expect(validateUiRuntimeProvenance(result)).toEqual([]);
    expect(validateModelBundleBinding(result)).toEqual([]);
    expect(validatePairedApiEvidence(result)).toEqual([]);
  });

  it("binds the terminal response ID to the exact health cache execution", () => {
    const correlated = correlateTerminalResponseToCacheExecution({
      terminal: {
        messageId: assistantIds[0],
        responseId: "chatcmpl-current",
      },
      cacheSnapshot: {
        scheduler: {
          last_cache_execution: {
            request_id: "chatcmpl-current",
            cache_reuse_applied: true,
            prompt_tokens: 192,
            cached_tokens: 128,
            prefill_tokens: 64,
          },
        },
      },
      turn: 1,
      proofRequestId: "ui-run-1:ui:1",
      userMessageId: "user-1",
      assistantMessageId: assistantIds[0],
    });
    expect(correlated.correlationStatus).toBe("verified");
    expect(correlated.serverRequestId).toBe("chatcmpl-current");
    expect(correlated.executionRequestId).toBe("chatcmpl-current");

    const overwritten = correlateTerminalResponseToCacheExecution({
      terminal: {
        messageId: assistantIds[0],
        responseId: "chatcmpl-current",
      },
      cacheSnapshot: {
        scheduler: {
          last_cache_execution: {
            request_id: "chatcmpl-unrelated-later-request",
          },
        },
      },
      turn: 1,
      proofRequestId: "ui-run-1:ui:1",
      userMessageId: "user-1",
      assistantMessageId: assistantIds[0],
    });
    expect(overwritten.correlationStatus).toBe(
      "partial_request_identity_mismatch",
    );

    const missing = correlateTerminalResponseToCacheExecution({
      terminal: { messageId: assistantIds[0] },
      cacheSnapshot: { scheduler: {} },
      turn: 1,
      proofRequestId: "ui-run-1:ui:1",
      userMessageId: "user-1",
      assistantMessageId: assistantIds[0],
    });
    expect(missing.correlationStatus).toBe(
      "partial_product_support_missing",
    );
  });

  it("rejects missing or overwritten per-turn cache request identity", () => {
    const result = goodResult();
    expect(isCacheRequestCorrelationVerified(result)).toBe(true);

    result.cacheRequestEvidence[1].executionRequestId =
      "chatcmpl-unrelated-later-request";
    result.cacheRequestEvidence[1].serverObservation.request_id =
      "chatcmpl-unrelated-later-request";
    expect(isCacheRequestCorrelationVerified(result)).toBe(false);
    expect(validateRequestCorrelatedCacheEvidence(result).join("\n")).toMatch(
      /do not exactly match|lacks exact server-emitted request correlation/,
    );

    result.cacheRequestEvidence[1].terminalResponseId = null;
    result.cacheRequestEvidence[1].serverRequestId = null;
    result.cacheRequestEvidence[1].correlationStatus =
      "partial_product_support_missing";
    expect(isCacheRequestCorrelationVerified(result)).toBe(false);
  });

  it("retains the newest HTTP response ID for continuation cancel and complete", () => {
    const source = readFileSync("src/main/ipc/chat.ts", "utf8");
    const responseCreated = source.slice(
      source.indexOf('responsesEventType === "response.created"'),
      source.indexOf("response.reasoning_summary_text.delta"),
    );
    const chatResponse = source.slice(
      source.indexOf("// Track response ID for server-side cancel"),
      source.indexOf("// Update usage BEFORE emitting delta"),
    );
    expect(responseCreated).toContain("entry.responseId = respId");
    expect(chatResponse).toContain("entry.responseId = parsed.id");
    expect(responseCreated).not.toContain("!entry.responseId");
    expect(chatResponse).not.toContain("!entry.responseId");
    expect(source.match(/responseId: activeRequests\.get\(chatId\)\?\.responseId/g))
      .toHaveLength(2);

    const active = { responseId: undefined as string | undefined };
    for (const responseId of ["chatcmpl-first", "chatcmpl-second"]) {
      active.responseId = responseId;
    }
    expect(active.responseId).toBe("chatcmpl-second");
  });

  it("accepts a one-turn tool restart profile without borrowing later-turn evidence", () => {
    const result = structuredClone(goodResult());
    result.requestContract = {
      ...result.requestContract,
      promptTwo: undefined,
      promptThree: undefined,
      uiActionProfile: "primary-tool-restart-probe",
      uiTurnCount: 1,
      apiActionProfile: "cache-probe",
    };
    result.assistantMessageIds = result.assistantMessageIds.slice(0, 1);
    result.assistantRecords = result.assistantRecords.slice(0, 1);
    result.persistedReasoningByMessage =
      result.persistedReasoningByMessage.slice(0, 1);
    result.persistedToolsByMessage = result.persistedToolsByMessage.slice(0, 1);
    result.persistedOaiCallsByMessage =
      result.persistedOaiCallsByMessage.slice(0, 1);
    result.persistedOaiResultsByMessage =
      result.persistedOaiResultsByMessage.slice(0, 1);
    result.messageEventTrace = result.messageEventTrace.slice(0, 1);
    result.renderedDom.messages = result.renderedDom.messages.slice(0, 1);
    result.renderedDom.samples = result.renderedDom.samples.filter(
      (sample: Record<string, unknown>) =>
        sample.messageId === result.assistantMessageIds[0],
    );
    result.uiTurnEvidence = result.uiTurnEvidence.slice(0, 1);
    result.resolvedSamplingRecords = result.resolvedSamplingRecords.slice(0, 1);
    result.requestCorrelation.turns = result.requestCorrelation.turns.slice(0, 1);
    result.cacheRequestEvidence = result.cacheRequestEvidence.slice(0, 1);
    result.cacheRequestEvidence[0].serverObservation = {
      ...result.cacheRequestEvidence[0].serverObservation,
      cache_reuse_applied: true,
      prompt_tokens: 192,
      cached_tokens: 128,
      prefill_tokens: 64,
      cache_hit_request: 1,
      cache_hit_tokens: 128,
      cache_reuse_partial_tokens: 64,
      cache_reuse_skip_tokens: 64,
    };
    result.cacheRequestEvidence[0].after = cacheStats({
      processed: 1,
      hitRequests: 1,
      hitTokens: 128,
      partialTokens: 64,
      skippedTokens: 64,
    });
    result.toolProbeFiles = {
      "real_ui_tool_probe_1.txt": "REAL_UI_LIVE_TOOL_ONE",
    };
    result.serverCacheControls.initialCacheControls.usePagedCache = false;
    result.serverCacheControls.argv = [
      "--no-paged-cache",
      "--enable-block-disk-cache",
    ];
    result.session.effective_config.usePagedCache = false;
    result.server.health.native_cache.paged = false;
    result.server.health.native_cache.block_disk_only = true;

    expect(validateRenderedDomEvidence(result)).toEqual([]);
    expect(validateReasoningEvidence(result, "required")).toEqual([]);
    expect(validateExactToolLoopEvidence(result)).toEqual([]);
    expect(validateGenerationDefaultsEvidence(result)).toEqual([]);
    expect(validateRequestCorrelatedCacheEvidence(result)).toEqual([]);
    expect(validateServerCacheEvidence(result)).toEqual([]);
    expect(isServerRequestCorrelationVerified(result)).toBe(true);
  });

  it("rejects forged CDP process bytes and served renderer source", () => {
    const result = structuredClone(goodResult());
    result.uiRuntimeProvenance.cdp_process_binding.listener_pid = 0;
    result.uiRuntimeProvenance.served_renderer_modules[0].sha256 = otherSha;
    expect(validateUiRuntimeProvenance(result).join("\n")).toMatch(
      /CDP listener PID|served through CDP/,
    );
  });

  it("rejects a renderer build commit that does not match the frozen proof HEAD", () => {
    const result = structuredClone(goodResult());
    result.uiRuntimeProvenance.renderer_build_source_commit = "d".repeat(40);
    expect(validateUiRuntimeProvenance(result).join("\n")).toMatch(
      /build-injected renderer source commit/,
    );
  });

  it("accepts exact dev provenance when Vite has no build-injected commit", () => {
    const result = structuredClone(goodResult());
    result.uiRuntimeProvenance.renderer_build_source_commit = null;
    expect(validateUiRuntimeProvenance(result)).toEqual([]);
  });

  it("canonicalizes a normal venv-style Python executable symlink", () => {
    const directory = mkdtempSync(path.join(tmpdir(), "vmlx-python-alias-"));
    try {
      const alias = path.join(directory, "python3");
      symlinkSync(testExecutablePath, alias);
      const aliasFingerprint = crypto
        .createHash("sha256")
        .update(alias)
        .digest("hex");
      const result = structuredClone(goodResult());
      result.uiRuntimeProvenance.backend_python_process_binding = {
        ...result.uiRuntimeProvenance.backend_python_process_binding,
        invoked_executable_path: alias,
        invoked_executable_path_fingerprint_sha256: aliasFingerprint,
        executable_path: alias,
        executable_path_fingerprint_sha256: testExecutablePathSha,
      };
      // /health fingerprints sys.executable's invoked alias spelling. The
      // listener retains and binds that path while canonical executable bytes
      // permit python/python3/versioned aliases of the same venv target.
      result.healthProvenance.after.binding.runtime_source_hashes
        .python_executable_fingerprint_sha256 = aliasFingerprint;
      expect(validateUiRuntimeProvenance(result)).toEqual([]);

      result.healthProvenance.after.binding.runtime_source_hashes
        .python_executable_fingerprint_sha256 = testExecutablePathSha;
      expect(validateUiRuntimeProvenance(result).join("\n")).toMatch(
        /backend TCP listener is not bound/,
      );
    } finally {
      rmSync(directory, { recursive: true, force: true });
    }
  });

  it("derives renderer provenance from Git and wires local request identity through every wire request", () => {
    const viteConfig = readFileSync(
      path.join(process.cwd(), "electron.vite.config.ts"),
      "utf8",
    );
    const rendererMain = readFileSync(
      path.join(process.cwd(), "src/renderer/src/main.tsx"),
      "utf8",
    );
    const chatSource = readFileSync(
      path.join(process.cwd(), "src/main/ipc/chat.ts"),
      "utf8",
    );
    const serverSource = readFileSync(
      path.join(process.cwd(), "../vmlx_engine/server.py"),
      "utf8",
    );
    expect(viteConfig).toContain("['-C', __dirname, 'rev-parse', 'HEAD']");
    expect(viteConfig).toContain("__VMLINUX_BUILD_SOURCE_COMMIT__");
    expect(rendererMain).toContain(
      "globalThis.__VMLINUX_SOURCE_COMMIT__ = __VMLINUX_BUILD_SOURCE_COMMIT__",
    );
    expect(rendererMain).toContain(
      "document.documentElement.dataset.sourceCommit = __VMLINUX_BUILD_SOURCE_COMMIT__",
    );
    expect(chatSource).toContain("const proofRequestId = userMessage.id;");
    expect(chatSource).toContain('"X-vMLX-Proof-Request-ID": proofRequestId');
    expect(chatSource).toContain('"X-vMLX-Request-ID": requestId');
    expect(chatSource).toContain('"X-vMLX-Message-ID": assistantMessageId');
    expect(chatSource.match(/nextLocalRequestCorrelationHeaders\(\)/g)).toHaveLength(3);
    expect(chatSource.match(/requestIds: \[\.\.\.wireRequestIds\]/g)).toHaveLength(2);
    expect(serverSource).toContain("def _request_header_value(request: Any, name: str)");
    // Three wire routes thread the proof request identity: chat, responses,
    // and (since the resolved-sampling-kwargs logging landed on it) the
    // Anthropic-shaped route.
    expect(
      serverSource.match(/proof_request_id=_request_header_value\(/g),
    ).toHaveLength(3);
  });

  it("binds installed ASAR, release manifest, bundled Python, runtime, and proof HEAD", () => {
    const result = structuredClone(goodResult());
    result.uiRuntimeProvenance = {
      ...result.uiRuntimeProvenance,
      mode: "installed-app",
      app_asar_sha256: sha,
      external_release_manifest_sha256: sha,
      external_release_manifest: {
        schema: "vmlx-installed-release-manifest-v1",
        source_commit: result.gitProvenance.after.commit,
        source_tree: "tree",
        app_asar_sha256: sha,
        electron_executable_sha256: testExecutableSha,
        bundled_provenance_sha256: sha,
        bundled_python_executable_sha256: testExecutableSha,
        bundled_python_executable_fingerprint_sha256: testExecutablePathSha,
      },
      bundled_provenance_sha256: sha,
      bundled_provenance: {
        vmlx: { commit: result.gitProvenance.after.commit, version: "1.6.19" },
      },
      bundled_source: {
        server_module_sha256: sha,
        package_init_sha256: sha,
        python_source_tree_sha256: sha,
        python_source_file_count: 10,
        python_source_read_error_count: 0,
      },
    };
    expect(validateUiRuntimeProvenance(result)).toEqual([]);

    result.uiRuntimeProvenance.external_release_manifest.source_commit = "stale";
    result.uiRuntimeProvenance.bundled_source.server_module_sha256 = otherSha;
    expect(validateUiRuntimeProvenance(result).join("\n")).toMatch(
      /release manifest|bundled\/runtime\/source identity/,
    );
  });

  it("rejects a different served model or bundle attestation", () => {
    const result = structuredClone(goodResult());
    result.server.models.data = [{ id: "different-model" }];
    result.healthProvenance.after.raw.model_bundle_provenance = {
      ...modelBundleAttestation,
      aggregate_sha256: otherSha,
    };
    expect(validateModelBundleBinding(result).join("\n")).toMatch(
      /exactly bound|bundle attestation/,
    );
  });

  it("keeps served model identity separate from bundle filesystem identity", () => {
    const result = structuredClone(goodResult());
    expect(result.servedModel).not.toBe(result.modelPath);
    expect(validateModelBundleBinding(result)).toEqual([]);
    expect(validateGenerationDefaultsEvidence(result)).toEqual([]);

    result.healthProvenance.before.raw.model_name = result.modelPath;
    result.healthProvenance.after.raw.model_name = result.modelPath;
    expect(validateModelBundleBinding(result).join("\n")).toMatch(
      /does not match requested served model/,
    );
  });

  it("accepts the requested served name plus the exact filesystem bundle alias", () => {
    const result = structuredClone(goodResult());
    result.server.models.data.push({ id: "models/test-model" });
    expect(validateModelBundleBinding(result)).toEqual([]);
  });

  it("uses the same bundle-file-bound backend fingerprint as the API matrix", () => {
    const health = {
      model_name: "test-model",
      loaded_model_name: "/private/models/test-model",
      runtime_provenance: {
        pid: 4321,
        server_module_sha256: sha,
        package_init_sha256: sha,
        python_source_tree_sha256: sha,
        python_executable_fingerprint_sha256: testExecutablePathSha,
        python_source_file_count: 10,
        python_source_read_error_count: 0,
      },
      model_bundle_provenance: modelBundleAttestation,
      cache_topology_provenance: {
        fingerprint_sha256: sha,
      },
    };
    const binding = runtimeBindingFromHealth(health);
    const canonicalIdentity = {
      backend_pid: 4321,
      runtime_source_hashes: {
        server_module_sha256: sha,
        package_init_sha256: sha,
        python_source_tree_sha256: sha,
        python_executable_fingerprint_sha256: testExecutablePathSha,
      },
      python_source_file_count: 10,
      python_source_read_error_count: 0,
      model_name: "test-model",
      loaded_model_name: "/private/models/test-model",
      model_bundle_fingerprint_sha256: sha,
      model_bundle_files: modelBundleAttestation.files,
      cache_topology_fingerprint_sha256: sha,
    };
    expect(binding.fingerprint_sha256).toBe(canonicalHash(canonicalIdentity));

    const changedHealth = structuredClone(health);
    changedHealth.model_bundle_provenance.files["config.json"].sha256 =
      otherSha;
    expect(runtimeBindingFromHealth(changedHealth).fingerprint_sha256)
      .not.toBe(binding.fingerprint_sha256);

    const changedLoadedModel = structuredClone(health);
    changedLoadedModel.loaded_model_name = "/private/models/other-model";
    expect(runtimeBindingFromHealth(changedLoadedModel).fingerprint_sha256)
      .not.toBe(binding.fingerprint_sha256);
  });

  it("treats an effort-only visible reasoning selection as persisted On", () => {
    const result = structuredClone(goodResult());
    result.requestedReasoningEffort = "xhigh";
    result.chatOverrides.enableThinking = true;
    result.chatOverrides.reasoningEffort = "xhigh";
    result.chatSettingsDom.reasoningMode = "On";
    result.chatSettingsDom.reasoningEffort = "xhigh";

    expect(validateGenerationDefaultsEvidence(result)).toEqual([]);
  });

  it("reads the effort-only persisted mode with the persisted label set", () => {
    const source = readFileSync(
      path.resolve("scripts/live-real-ui-model-proof.mjs"),
      "utf8",
    );
    expect(source).toMatch(
      /const reopenedThinkingButton =[\s\S]*?acceptablePersistedThinkingLabels\.includes\(/,
    );
  });

  it("includes nonstream recovery in the cancellation raw-capture contract", () => {
    const contract = {
      protocols: ["chat", "responses", "anthropic", "ollama"],
      modes: ["stream", "nonstream"],
    };
    const routes = expectedRawMatrixRoutes(contract, true);

    expect(routes).toHaveLength(72);
    expect(routes).toContain("direct\0chat\0nonstream-recovery");
    expect(routes).toContain("gateway\0ollama\0nonstream-recovery");
    expect(new Set(routes).size).toBe(routes.length);
  });

  it("rejects settings not visibly persisted and stale route/model log records", () => {
    const result = structuredClone(goodResult());
    result.chatSettingsInteraction.reopenedAfterSave = false;
    result.resolvedSamplingRecords[1].model = "stale-model";
    result.resolvedSamplingRecords[2].message_id = "wrong-message";
    expect(validateGenerationDefaultsEvidence(result).join("\n")).toMatch(
      /visibly changed, saved, and reopened|route\/model|server-emitted request\/message correlation/,
    );
  });

  it("uses concrete reopened and request evidence instead of the aggregate persistence bit", () => {
    const result = structuredClone(goodResult());
    result.chatSettingsInteraction.persistedAfterReopen = false;

    expect(validateGenerationDefaultsEvidence(result)).toEqual([]);
  });

  it("distinguishes inherited defaults from explicit visible per-chat overrides", () => {
    const result = structuredClone(goodResult());
    result.requestContract.samplingOverrides = { temperature: 0.25 };
    result.requestContract.requestMaxTokens = 512;
    result.chatOverrides.temperature = 0.25;
    result.chatOverrides.maxTokens = 512;
    result.chatSettingsDom.values.temperature = 0.25;
    result.chatSettingsDom.maxTokens.value = "512";
    result.resolvedSamplingKwargs.temperature = 0.25;
    result.resolvedSamplingKwargs.max_tokens = 512;
    for (const record of result.resolvedSamplingRecords) {
      record.values.temperature = 0.25;
      record.values.max_tokens = 512;
    }
    expect(validateGenerationDefaultsEvidence(result)).toEqual([]);

    result.chatOverrides.temperature = 0.7;
    expect(validateGenerationDefaultsEvidence(result).join("\n")).toMatch(
      /explicit temperature override=0.25 was not persisted exactly/,
    );
  });

  it("accepts a redundant explicit sampler value through visible and resolved persistence", () => {
    const result = structuredClone(goodResult());
    result.requestContract.samplingOverrides = {
      temperature: result.bundleGenerationContract.defaults.temperature,
      topP: result.bundleGenerationContract.defaults.topP,
      topK: result.bundleGenerationContract.defaults.topK,
    };
    delete result.chatOverrides.temperature;
    delete result.chatOverrides.topP;
    delete result.chatOverrides.topK;

    expect(validateGenerationDefaultsEvidence(result)).toEqual([]);
  });

  it("requires an active Native MTP runtime to report the compatible-only sampling policy", () => {
    const result = structuredClone(goodResult());
    result.server.health.mtp.runtime_active = true;
    result.nativeMtpSelection = { persistedMode: "adaptive" };
    result.effectiveSessionConfig = { nativeMtpMode: "adaptive" };
    result.server.health.mtp.request_policy = "compatible-only";
    expect(validateGenerationDefaultsEvidence(result)).toEqual([]);
    for (const retired of ["deterministic-defaults", "greedy-only"]) {
      result.server.health.mtp.request_policy = retired;
      expect(validateGenerationDefaultsEvidence(result).join("\n")).toMatch(/compatible-only/);
    }
  });

  it("never expects greedy UI/engine defaults from the MTP mode: bundle sampling stays authoritative", () => {
    const result = structuredClone(goodResult());
    result.server.health.mtp.runtime_active = true;
    result.server.health.mtp.request_policy = "compatible-only";
    result.nativeMtpSelection = { persistedMode: "adaptive" };
    result.effectiveSessionConfig = { nativeMtpMode: "adaptive" };
    // goodResult() already carries bundle-default sampling on every surface.
    expect(validateGenerationDefaultsEvidence(result)).toEqual([]);
    const explicit = { temperature: 0.25, topP: 0.8, topK: 10, minP: 0.02 };
    result.requestContract.samplingOverrides = explicit;
    Object.assign(result.chatOverrides, explicit);
    Object.assign(result.chatSettingsDom.values, explicit);
    const wire = { temperature: 0.25, top_p: 0.8, top_k: 10, min_p: 0.02 };
    Object.assign(result.resolvedSamplingKwargs, wire);
    for (const record of result.resolvedSamplingRecords) Object.assign(record.values, wire);
    expect(validateGenerationDefaultsEvidence(result)).toEqual([]);
  });

  it("allows an exact one-token reasoning segment to arrive in one delta", () => {
    const source = readFileSync(
      path.resolve("scripts/live-real-ui-model-proof.mjs"),
      "utf8",
    );
    expect(source).toContain("const singleTokenReasoning = /^\\S+$/.test(completeReasoningText)");
    expect(source).toContain("progressiveReasoningDeltaCount < 2 && !singleTokenReasoning");
  });

  it("binds model-native reasoning effort before and during the conversation", () => {
    const result = structuredClone(goodResult());
    result.requestedReasoningEffort = "xhigh";
    result.requestedMidConvReasoningEffort = "low";
    result.chatSettingsDom.reasoningMode = null;
    result.chatSettingsDom.thinkingNotice = true;
    result.chatSettingsDom.reasoningEffort = "xhigh";
    result.chatOverrides.enableThinking = true;
    result.chatOverrides.reasoningEffort = "xhigh";
    result.midConvReasoningFlip = {
      requested: true,
      buttonFound: true,
      saved: true,
      persistedReasoningEffortAfter: "low",
    };
    expect(validateGenerationDefaultsEvidence(result)).toEqual([]);

    result.midConvReasoningFlip.persistedReasoningEffortAfter = "medium";
    expect(validateGenerationDefaultsEvidence(result).join("\n")).toMatch(
      /mid-conversation reasoning effort was not visibly changed and persisted/,
    );
  });

  it("rejects missing visible cache state, contradictory argv, and capacity-only telemetry", () => {
    const result = structuredClone(goodResult());
    result.serverCacheControls.controlScope = "app-mode-navigation";
    delete result.serverCacheControls.initialCacheControls.enablePrefixCache;
    result.serverCacheControls.argv = [
      "--disable-prefix-cache",
      "--no-paged-cache",
      "--enable-block-disk-cache",
    ];
    result.cacheRequestEvidence = result.cacheRequestEvidence.map(
      (row: Record<string, any>, index: number) => ({
        ...row,
        before: cacheStats({
          processed: index,
          hitRequests: 0,
          hitTokens: 0,
          partialTokens: 0,
          skippedTokens: 0,
        }),
        after: {
          ...cacheStats({
            processed: index + 1,
            hitRequests: 0,
            hitTokens: 0,
            partialTokens: 0,
            skippedTokens: 0,
          }),
          max_cache_blocks: 8192,
        },
      }),
    );
    expect(validateServerCacheEvidence(result).join("\n")).toMatch(
      /running-session toolbar|visible cache control|contradicted enabled prefix cache|contradictory --no-paged-cache|no UI turn had request-correlated/,
    );
  });

  it("keeps cache-positive evidence mandatory for non-DSV4 sessions", () => {
    const result = goodResult();
    expect(validateServerCacheEvidence(result)).toEqual([]);

    result.serverCacheControls.initialCacheControls.enableBlockDiskCache = false;
    expect(validateServerCacheEvidence(result).join("\n")).toMatch(
      /SSD\/L2 control was not visibly enabled/,
    );
  });

  it("accepts a source-matched exact typed prompt-disk lane without generic block L2", () => {
    const result = structuredClone(goodResult());
    result.serverCacheControls.initialCacheControls = {
      ...result.serverCacheControls.initialCacheControls,
      usePagedCache: false,
      enableDiskCache: true,
      enableBlockDiskCache: false,
      diskCachePresent: true,
      blockDiskCachePresent: true,
    };
    result.serverCacheControls.argv = [
      "--no-paged-cache",
      "--enable-disk-cache",
    ];
    result.session.effective_config = {
      ...result.session.effective_config,
      usePagedCache: false,
      enableDiskCache: true,
      enableBlockDiskCache: false,
    };
    result.server.health.native_cache = {
      family: "glm5_next",
      schema: "glm5_next_native_v1",
      schema_implemented: true,
      prefix: true,
      paged: false,
      prompt_disk_l2_configured: true,
      block_disk_l2_configured: false,
      prompt_disk_l2: true,
      block_disk_l2: false,
      cache_store_policy: {
        prompt_boundary: "exact_n_minus_one",
        prompt_disk_l2: "typed_full_state",
      },
    };

    expect(validateServerCacheEvidence(result)).toEqual([]);

    result.serverCacheControls.argv.push("--enable-block-disk-cache");
    expect(validateServerCacheEvidence(result).join("\n")).toMatch(
      /generic block disk cache for an exact prompt-disk lane/,
    );
  });

  it("binds the visible aggregate SSD percentage to config, argv, and health", () => {
    const result = structuredClone(goodResult());
    result.requestedBlockDiskCacheMaxPercent = 5;
    result.serverCacheControls.initialCacheControls.blockDiskCacheMaxPercent = 5;
    result.session.effective_config.blockDiskCacheMaxPercent = 5;
    result.serverCacheControls.argv = [
      "--use-paged-cache",
      "--enable-block-disk-cache",
      "--block-disk-cache-max-percent",
      "5",
    ];
    result.server.health.block_disk_cache = { max_size_gb: 92.5 };
    expect(validateServerCacheEvidence(result)).toEqual([]);

    result.serverCacheControls.argv.push("--block-disk-cache-max-gb", "10");
    expect(validateServerCacheEvidence(result).join("\n")).toMatch(
      /flat GB cap that overrides the requested SSD percentage/,
    );
  });

  it("accepts the live command-log and global-budget cache telemetry shapes", () => {
    const result = structuredClone(goodResult());
    result.requestedBlockDiskCacheMaxPercent = 10;
    result.serverCacheControls.initialCacheControls.blockDiskCacheMaxPercent = 10;
    result.session.effective_config.blockDiskCacheMaxPercent = 10;
    result.serverCacheControls.argv = [
      "--use-paged-cache",
      "--enable-block-disk-cache",
      "--block-disk-cache-max-percent",
      "--native-mtp-depth-policy",
    ];
    result.serverCacheControls.commandLine = [
      "$ vmlx-serve model",
      "--enable-block-disk-cache",
      "--block-disk-cache-max-percent 10",
    ].join(" ");
    result.server.health.block_disk_cache = {
      global_budget: { max_size_bytes: 399_625_232_056 },
    };

    expect(validateServerCacheEvidence(result)).toEqual([]);
  });

  it("accepts DSV4 only with visible SSD-only native cache and observed pool quant", () => {
    expect(validateServerCacheEvidence(dsv4EnabledResult())).toEqual([]);
  });

  it.each([
    [
      "visible prefix disabled",
      (result: Record<string, any>) => {
        result.serverCacheControls.initialCacheControls.enablePrefixCache = false;
      },
      /visible prefix-cache state does not match persisted session config/,
    ],
    [
      "visible SSD disabled",
      (result: Record<string, any>) => {
        result.serverCacheControls.initialCacheControls.enableBlockDiskCache = false;
      },
      /SSD\/L2 control was not visibly enabled/,
    ],
    [
      "missing --no-paged-cache",
      (result: Record<string, any>) => {
        result.serverCacheControls.argv = ["--enable-block-disk-cache"];
      },
      /argv omitted --no-paged-cache/,
    ],
    [
      "missing SSD argv",
      (result: Record<string, any>) => {
        result.serverCacheControls.argv = ["--no-paged-cache"];
      },
      /argv omitted --enable-block-disk-cache/,
    ],
    [
      "health prefix mismatch",
      (result: Record<string, any>) => {
        result.server.health.native_cache.prefix = false;
      },
      /native prefix state does not match/,
    ],
    [
      "pool cache class mismatch",
      (result: Record<string, any>) => {
        result.server.health.native_cache.pool_quant.observed = false;
      },
      /observed cache class does not match/,
    ],
    [
      "pool attestation error",
      (result: Record<string, any>) => {
        result.server.health.native_cache.pool_quant.error = "cache construction failed";
      },
      /attestation reported an error/,
    ],
  ] as Array<[
    string,
    (result: Record<string, any>) => void,
    RegExp,
  ]>)("rejects an unproven DSV4 cache surface: %s", (_name, mutate, expected) => {
    const result = dsv4EnabledResult();
    mutate(result);
    expect(validateServerCacheEvidence(result).join("\n")).toMatch(expected);
  });

  it("rejects cumulative resets, transient parser markers, and final/persisted mismatch", () => {
    const result = structuredClone(goodResult());
    result.messageEventTrace[0].events[1].cumulativeReset = true;
    result.messageEventTrace[1].events[0].delta = "<thi";
    result.messageEventTrace[2].events.at(-2).delta += " different final";
    result.messageEventTrace[2].events.at(-2).payload.fullContentLength =
      result.messageEventTrace[2].events.at(-2).payload.fullContent.length
      + " different final".length;
    expect(validateReasoningEvidence(result, "required").join("\n")).toMatch(
      /stream reset or shrank|transient reasoning stream|does not equal persisted final/,
    );
  });

  it.each([
    ["<t", "hink>"],
    ["[T", "HINK]"],
    ["<m", "m:think>"],
  ])("rejects parser markers split across stream boundaries", (first, second) => {
    const result = structuredClone(goodResult());
    const reasoningEvents = result.messageEventTrace[0].events.filter(
      (event: Record<string, unknown>) => event.event === "stream" && event.channel === "reasoning",
    );
    reasoningEvents[0].delta = first;
    reasoningEvents[0].payload.fullContentLength = first.length;
    reasoningEvents[0].payload.reasoningSegments = [first];
    reasoningEvents[0].payload.reasoningSegmentLength = first.length;
    delete reasoningEvents[0].payload.fullContent;
    reasoningEvents[1].delta = second;
    reasoningEvents[1].payload.fullContentLength = first.length + second.length;
    reasoningEvents[1].payload.reasoningSegments = [first + second];
    reasoningEvents[1].payload.reasoningSegmentLength =
      first.length + second.length;
    delete reasoningEvents[1].payload.fullContent;
    result.persistedReasoningByMessage[0] = [first + second];

    expect(validateReasoningEvidence(result, "required").join("\n")).toMatch(
      /parser markers across reasoning stream boundaries/,
    );
  });

  it("accepts compact stream events without retaining cumulative payload text", () => {
    const result = structuredClone(goodResult());
    for (const trace of result.messageEventTrace) {
      for (const event of trace.events) {
        if (event.event !== "stream") continue;
        event.payload.fullContentLength = event.payload.fullContent.length;
        delete event.payload.fullContent;
      }
    }

    expect(validateRenderedDomEvidence(result)).toEqual([]);
    expect(validateReasoningEvidence(result, "required")).toEqual([]);
  });

  it("compares interleaved reasoning passes as ordered segments", () => {
    const result = structuredClone(goodResult());
    const trace = result.messageEventTrace[0];
    const contentEvents = trace.events.filter(
      (event: Record<string, unknown>) => event.channel === "content",
    );
    const terminal = trace.events.find(
      (event: Record<string, unknown>) => event.event === "terminal",
    );
    trace.events = [
      {
        ...streamEvent(1, "reasoning", "Reason", "Reason"),
        segmentIndex: 0,
      },
      {
        ...streamEvent(2, "reasoning", "Reason carefully", " carefully"),
        segmentIndex: 0,
      },
      {
        sequence: 3,
        event: "reasoning_terminal",
        channel: "reasoning",
        segmentIndex: 0,
        payload: { reasoningSegments: ["Reason carefully"] },
      },
      {
        ...streamEvent(4, "reasoning", "Plan", "Plan"),
        segmentIndex: 1,
        payload: {
          fullContent: "Reason carefully\n\nPlan",
          fullContentLength: "Reason carefully\n\nPlan".length,
          isReasoning: true,
          reasoningSegments: ["Reason carefully", "Plan"],
          reasoningSegmentLength: "Plan".length,
        },
      },
      {
        ...streamEvent(5, "reasoning", "Plan again", " again"),
        segmentIndex: 1,
        payload: {
          fullContent: "Reason carefully\n\nPlan again",
          fullContentLength: "Reason carefully\n\nPlan again".length,
          isReasoning: true,
          reasoningSegments: ["Reason carefully", "Plan again"],
          reasoningSegmentLength: "Plan again".length,
        },
      },
      ...contentEvents.map((event: Record<string, unknown>, index: number) => ({
        ...event,
        sequence: 6 + index,
      })),
      { ...terminal, sequence: 8 },
    ];
    result.persistedReasoningByMessage[0] = [
      "Reason carefully",
      "Plan again",
    ];
    result.renderedDom.messages[0].reasoningSegments = [
      "Reason carefully",
      "Plan again",
    ];
    result.renderedDom.messages[0].reasoningText =
      "Reason carefully\nPlan again";
    result.renderedDom.samples = result.renderedDom.samples.map(
      (sample: Record<string, any>) =>
        sample.messageId === assistantIds[0]
          ? {
              ...sample,
              reasoningText:
                sample.answerText === renderedContents[0]
                  ? "Reason carefully\nPlan again"
                  : "Reason carefully",
            }
          : sample,
    );

    expect(validateReasoningEvidence(result, "required")).toEqual([]);

    result.persistedReasoningByMessage[0] = [
      "Plan again",
      "Reason carefully",
    ];
    expect(validateReasoningEvidence(result, "required").join("\n")).toMatch(
      /stream segments do not equal persisted reasoning segments|rail segments are not linked/,
    );
  });

  it("binds KaTeX reasoning to persisted canonical TeX identity without comparing presentation text", () => {
    const result = structuredClone(goodResult());
    const persistedReasoning =
      "Keep literal $43, then inspect $47 \\times 19 = 893 < 920 = 46 \\× 20$ exactly.";
    const renderedReasoning =
      "Keep literal $43, then inspect 47×19=893<920=46×20 exactly.";
    const linkage = extractPersistedReasoningMathLinkage(persistedReasoning);
    expect(linkage.mathSources).toEqual([
      {
        source: "47 \\times 19 = 893 < 920 = 46 \\× 20",
        delimiter: "single-dollar",
        displayMode: "inline",
      },
    ]);
    expect(linkage.linkedText).toBe(
      "Keep literal $43, then inspect VMLXPROOFMATH0 exactly.",
    );

    const trace = result.messageEventTrace[0];
    const prefix = persistedReasoning.slice(
      0,
      Math.max(1, Math.floor(persistedReasoning.length / 2)),
    );
    trace.events[0] = streamEvent(1, "reasoning", prefix, prefix);
    trace.events[1] = streamEvent(
      2,
      "reasoning",
      persistedReasoning,
      persistedReasoning.slice(prefix.length),
    );
    result.persistedReasoningByMessage[0] = [persistedReasoning];
    result.renderedDom.messages[0].reasoningText = renderedReasoning;
    result.renderedDom.messages[0].reasoningSegments = [renderedReasoning];
    result.renderedDom.messages[0].reasoningLinkedSegments = [
      linkage.linkedText,
    ];
    result.renderedDom.messages[0].reasoningMathSources = [
      linkage.mathSources,
    ];
    result.renderedDom.messages[0].reasoningMathDisplayModes = [[
      { wrapperDisplayMode: "inline", katexDisplayMode: "inline" },
    ]];
    result.renderedDom.messages[0].reasoningKatexCounts = [1];
    result.renderedDom.messages[0].reasoningKatexErrorCounts = [0];

    expect(validateReasoningEvidence(result, "required")).toEqual([]);

    const changedSource = structuredClone(result);
    changedSource.renderedDom.messages[0].reasoningMathSources[0][0].source =
      "48 \\times 19 = 912";
    expect(validateReasoningEvidence(changedSource, "required").join("\n")).toMatch(
      /math identity is not exactly linked/,
    );

    const changedMode = structuredClone(result);
    changedMode.renderedDom.messages[0].reasoningMathDisplayModes[0][0] = {
      wrapperDisplayMode: "display",
      katexDisplayMode: "display",
    };
    expect(validateReasoningEvidence(changedMode, "required").join("\n")).toMatch(
      /inline\/display mode does not match/,
    );

    const changedProse = structuredClone(result);
    changedProse.renderedDom.messages[0].reasoningLinkedSegments[0] =
      "Different prose VMLXPROOFMATH0 exactly.";
    expect(validateReasoningEvidence(changedProse, "required").join("\n")).toMatch(
      /reasoning rail segments are not linked/,
    );

    const missingKatex = structuredClone(result);
    missingKatex.renderedDom.messages[0].reasoningKatexCounts = [0];
    expect(validateReasoningEvidence(missingKatex, "required").join("\n")).toMatch(
      /did not render cleanly through KaTeX/,
    );
  });

  it("does not classify literal currency or protected code as reasoning math", () => {
    const linkage = extractPersistedReasoningMathLinkage(
      "Prices stay $5 < $10 and `$47 \\times 19$`; render $6 \\times 7 = 42$.",
    );
    expect(linkage.mathSources).toEqual([{
      source: "6 \\times 7 = 42",
      delimiter: "single-dollar",
      displayMode: "inline",
    }]);
    expect(linkage.linkedText).toBe(
      "Prices stay $5 < $10 and `$47 \\times 19$`; render VMLXPROOFMATH0.",
    );
  });

  it("mirrors repeated-delimiter normalization before binding persisted TeX", () => {
    const linkage = extractPersistedReasoningMathLinkage(
      "Draft: \\(\\(47 \\times 19 = 893 < 920\\)",
    );
    expect(linkage).toEqual({
      linkedText: "Draft: VMLXPROOFMATH0",
      mathSources: [{
        source: "47 \\times 19 = 893 < 920",
        delimiter: "paren",
        displayMode: "inline",
      }],
    });
  });

  it("mirrors the renderer when whitespace-only math produces no KaTeX node", () => {
    expect(extractPersistedReasoningMathLinkage(
      "Before \\(   \\) after.",
    )).toEqual({
      linkedText: "Before  after.",
      mathSources: [],
    });
  });

  it("rejects unexpected rendered math when persisted reasoning has no math", () => {
    const result = structuredClone(goodResult());
    result.renderedDom.messages[0].reasoningMathSources = [[{
      source: "2 + 2 = 4",
      delimiter: "paren",
      displayMode: "inline",
    }]];
    result.renderedDom.messages[0].reasoningMathDisplayModes = [[{
      wrapperDisplayMode: "inline",
      katexDisplayMode: "inline",
    }]];
    result.renderedDom.messages[0].reasoningKatexCounts = [1];
    result.renderedDom.messages[0].reasoningKatexErrorCounts = [0];

    expect(validateReasoningEvidence(result, "required").join("\n")).toMatch(
      /unexpected reasoning math absent from persisted reasoning/,
    );
  });

  it("does not erase ordinary prose punctuation during reasoning linkage", () => {
    const result = structuredClone(goodResult());
    const persisted = "Check (scope) before $2 + 2 = 4$.";
    const linkage = extractPersistedReasoningMathLinkage(persisted);
    result.persistedReasoningByMessage[0] = [persisted];
    const trace = result.messageEventTrace[0];
    const prefix = persisted.slice(0, Math.floor(persisted.length / 2));
    trace.events[0] = streamEvent(1, "reasoning", prefix, prefix);
    trace.events[1] = streamEvent(
      2,
      "reasoning",
      persisted,
      persisted.slice(prefix.length),
    );
    result.renderedDom.messages[0].reasoningText = "Check scope before 2+2=4.";
    result.renderedDom.messages[0].reasoningSegments = ["Check scope before 2+2=4."];
    result.renderedDom.messages[0].reasoningLinkedSegments = [
      "Check scope before VMLXPROOFMATH0.",
    ];
    result.renderedDom.messages[0].reasoningMathSources = [linkage.mathSources];
    result.renderedDom.messages[0].reasoningMathDisplayModes = [[{
      wrapperDisplayMode: "inline",
      katexDisplayMode: "inline",
    }]];
    result.renderedDom.messages[0].reasoningKatexCounts = [1];
    result.renderedDom.messages[0].reasoningKatexErrorCounts = [0];

    expect(validateReasoningEvidence(result, "required").join("\n")).toMatch(
      /reasoning rail segments are not linked/,
    );
  });

  it("rejects duplicate/error tool records and missing probe cleanup", () => {
    const result = structuredClone(goodResult());
    result.persistedOaiCallsByMessage[1][0].id = "call-1";
    result.persistedToolsByMessage[1].push({
      phase: "error",
      toolCallId: "call-1",
      toolName: "run_command",
    });
    result.toolProbeCleanup.removed = false;
    expect(validateExactToolLoopEvidence(result).join("\n")).toMatch(
      /IDs are not unique|error status|not cleaned/,
    );
  });

  it("requires both built-in calls for the legacy three-turn profile", () => {
    const result = structuredClone(goodResult());
    expect(result.requestContract.uiActionProfile).toBe("legacy-three-turn");
    expect(validateExactToolLoopEvidence(result)).toEqual([]);

    result.persistedOaiCallsByMessage[1] = [];
    result.persistedOaiResultsByMessage[1] = [];
    result.persistedToolsByMessage[1] = [];
    result.renderedDom.messages[1].toolCards = [];
    expect(validateExactToolLoopEvidence(result).join("\n")).toMatch(
      // Surplus verification calls are tolerated now, so the count rule is a
      // FLOOR — but a run that made only one call still has to fail, and the
      // second protocol step still has to be missing from its own turn.
      /expected at least 2 tool calls|tool call 2 was not persisted on assistant turn 2/,
    );
  });

  it("rejects unlinked DOM content, malformed KaTeX/currency, and raw i18n chrome", () => {
    const linkResult = structuredClone(goodResult());
    linkResult.renderedDom.messages[0].answerText = "unrelated";
    expect(validateRenderedDomEvidence(linkResult).join("\n")).toMatch(
      /normalized visible answer is not linked/,
    );

    const renderResult = structuredClone(goodResult());
    renderResult.renderedDom.messages[2].currencyOccurrences[0].insideKatex =
      true;
    renderResult.renderedDom.messages[2].katexAnnotations = [
      "47 \\times 19 = wrong",
    ];
    renderResult.renderedDom.rawI18nKeys = ["layout.chatHistory.groupToday"];
    expect(validateRenderedDomEvidence(renderResult).join("\n")).toMatch(
      /outside KaTeX|does not exactly match|raw i18n keys/,
    );
  });

  it("attests answer prose without folding renderer chrome into it", () => {
    const source = readFileSync(
      path.resolve("scripts/live-real-ui-model-proof.mjs"),
      "utf8",
    );
    expect(source).toContain("const proseAnswer = answer?.cloneNode(true);");
    expect(source).toContain(
      "'[data-vmlx-proof-tool-card], [data-vmlx-proof-tool-container], .code-header'",
    );
    expect(source).toContain("const readMountedInnerText = (element) =>");
    expect(source).toContain("probe.appendChild(element);");
    expect(source).toContain("document.body.appendChild(probe);");
    expect(source).toContain("return element.innerText.trim();");
    expect(source).not.toContain(
      "proseAnswer.innerText || proseAnswer.textContent",
    );
    expect(source).toContain("probe.remove();");

    const valid = structuredClone(goodResult());
    valid.renderedDom.messages[0].toolCards[0].text =
      "Used run_command REAL_UI_LIVE_TOOL_ONE";
    expect(validateRenderedDomEvidence(valid)).toEqual([]);

    valid.renderedDom.messages[0].answerText +=
      " Used run_command REAL_UI_LIVE_TOOL_ONE";
    expect(validateRenderedDomEvidence(valid).join("\n")).toMatch(
      /normalized visible answer is not linked/,
    );
  });

  it("links rendered line breaks and list glyphs to persisted Markdown", () => {
    const valid = structuredClone(goodResult());
    const persisted = "R19-DONE\n- first item\n- second item";
    valid.assistantRecords[0].content = persisted;
    valid.messageEventTrace[0] = primaryTrace(assistantIds[0], persisted, 0);
    valid.renderedDom.messages[0].answerText =
      "R19-DONE\nfirst item\nsecond item";
    valid.renderedDom.messages[0].answerFullLength = persisted.length;
    valid.renderedDom.messages[0].answerRenderedLength = persisted.length;
    expect(validateRenderedDomEvidence(valid)).toEqual([]);
  });

  it("rejects a terminal DOM snapshot before the typewriter fully drains", () => {
    const invalid = structuredClone(goodResult());
    invalid.renderedDom.messages[0].answerRenderedLength -= 1;
    expect(validateRenderedDomEvidence(invalid).join("\n")).toMatch(
      /visible typewriter did not drain/,
    );
  });

  it("binds output-html KaTeX to exact persisted TeX without requiring MathML", () => {
    const valid = structuredClone(goodResult());
    valid.renderedDom.messages[2].katexAnnotations = [];
    expect(validateRenderedDomEvidence(valid)).toEqual([]);

    valid.renderedDom.messages[2].katexCount = 0;
    expect(validateRenderedDomEvidence(valid).join("\n")).toMatch(
      /persisted TeX source did not produce a KaTeX-rendered expression/,
    );
  });

  it("accepts truthful Unicode math when the model does not emit TeX delimiters", () => {
    const valid = structuredClone(goodResult());
    const unicodeAnswer = "$43 and 47 × 19 = 893 < 920 = 46 × 20";
    valid.assistantRecords[2].content = unicodeAnswer;
    valid.messageEventTrace[2] = primaryTrace(
      assistantIds[2],
      unicodeAnswer,
    );
    valid.renderedDom.messages[2].answerText = unicodeAnswer;
    valid.renderedDom.messages[2].html = `<p>${unicodeAnswer}</p>`;
    valid.renderedDom.messages[2].answerFullLength = unicodeAnswer.length;
    valid.renderedDom.messages[2].answerRenderedLength = unicodeAnswer.length;
    valid.renderedDom.messages[2].katexCount = 0;
    valid.renderedDom.messages[2].katexAnnotations = [];
    expect(validateRenderedDomEvidence(valid)).toEqual([]);
  });

  it("requires a separate, exactly bound raw API artifact before dual-surface claims", () => {
    const missing = structuredClone(goodResult());
    missing.surfaceStatus = "dual_surface_attested";
    expect(validatePairedApiEvidence(missing).join("\n")).toMatch(
      /without a separate raw API artifact/,
    );

    const valid = structuredClone(goodResult());
    valid.surfaceStatus = "dual_surface_attested";
    const fixture = createValidPairedArtifact(valid);
    try {
      expect(fixture.artifact.value.raw_capture.routes).toHaveLength(48);
      valid.pairedApiArtifact = fixture.artifact;
      expect(validatePairedApiEvidence(valid)).toEqual([]);

      valid.pairedApiArtifact.value.backend_pid = 9999;
      expect(validatePairedApiEvidence(valid).join("\n")).toMatch(
        /metadata does not match|backend identity/,
      );
    } finally {
      rmSync(fixture.directory, { recursive: true, force: true });
    }
  });

  it("enforces the exact phase-scoped paired API contract without weakening full phases", () => {
    const result = structuredClone(goodResult());
    result.surfaceStatus = "dual_surface_attested";
    result.apiActionProfile = "cache-probe";
    result.requestContract.apiActionProfile = "cache-probe";
    const fixture = createValidPairedArtifact(result);
    try {
      result.apiActionProfile = "full-agentic";
      result.requestContract.apiActionProfile = "full-agentic";
      result.pairedApiArtifact = fixture.artifact;
      expect(validatePairedApiEvidence(result)).toEqual([]);

      delete result.apiActionProfile;
      expect(validatePairedApiEvidence(result).join("\n")).toMatch(
        /action profile is missing, inconsistent, or unsupported/,
      );
      result.apiActionProfile = "full-agentic";
      delete result.requestContract.apiActionProfile;
      expect(validatePairedApiEvidence(result).join("\n")).toMatch(
        /action profile is missing, inconsistent, or unsupported/,
      );

      result.apiActionProfile = "cache-probe";
      result.requestContract.apiActionProfile = "cache-probe";
      const scoped = structuredClone(fixture.artifact.value);
      restrictPairedArtifactToContract(scoped, ["chat"], ["stream"]);
      result.pairedApiArtifact = writePairedArtifactValue(
        fixture.directory,
        "scoped-chat-stream.json",
        scoped,
      );
      expect(validatePairedApiEvidence(result)).toEqual([]);

      const extraProtocol = structuredClone(scoped);
      extraProtocol.protocols.push("responses");
      result.pairedApiArtifact = writePairedArtifactValue(
        fixture.directory,
        "scoped-extra-protocol.json",
        extraProtocol,
      );
      expect(validatePairedApiEvidence(result).join("\n")).toMatch(
        /exact passing vmlx-agentic-protocol-matrix-v2 run/,
      );

      result.apiActionProfile = "full-agentic-plus-cache-store";
      result.requestContract.apiActionProfile =
        "full-agentic-plus-cache-store";
      result.pairedApiArtifact = writePairedArtifactValue(
        fixture.directory,
        "subset-on-full-profile.json",
        scoped,
      );
      expect(validatePairedApiEvidence(result).join("\n")).toMatch(
        /exact passing vmlx-agentic-protocol-matrix-v2 run|matrix protocols are incomplete/,
      );

      result.apiActionProfile = "cache-probe";
      result.requestContract.apiActionProfile = "cache-restart-probe";
      expect(validatePairedApiEvidence(result).join("\n")).toMatch(
        /action profile is missing, inconsistent, or unsupported/,
      );
    } finally {
      rmSync(fixture.directory, { recursive: true, force: true });
    }
  });

  it("binds an installed paired producer to the same private manifest and packaged bytes", () => {
    const result = goodResult();
    result.surfaceStatus = "dual_surface_attested";
    const fixture = createInstalledPairedArtifact(result);
    try {
      result.pairedApiArtifact = fixture.artifact;
      const installed = fixture.artifact.value.identity.runner.before
        .installed_runtime;
      expect(installed.app_path).toBe(fixture.appPath);
      expect(installed.bundled_python.path).toBe(
        realpathSync(fixture.pythonPath),
      );
      expect(path.basename(installed.invoked_python_path)).toBe("python3.12");
      expect(
        fixture.artifact.value.identity.runner.before.python_executable_path,
      ).toBe(realpathSync(fixture.pythonPath));
      expect(path.relative(realpathSync(installed.app_path), installed.bundled_python.path))
        .not.toMatch(/^\.\.(?:\/|$)/);
      expect(validateUiRuntimeProvenance(result)).toEqual([]);
      expect(validatePairedApiEvidence(result)).toEqual([]);

      writeFileSync(fixture.pythonPath, "mutated-after-attestation\n");
      expect(validateUiRuntimeProvenance(result).join("\n")).toMatch(
        /Python backend executable path\/bytes|backend Python bytes/,
      );
      expect(validatePairedApiEvidence(result).join("\n")).toMatch(
        /producer executable\/harness bytes|bundled Python path\/bytes|canonical executable identity/,
      );
    } finally {
      rmSync(fixture.directory, { recursive: true, force: true });
    }
  });

  it("binds the live backend invocation alias to the same manifest-attested target", () => {
    const result = goodResult();
    const fixture = createInstalledPairedArtifact(result);
    try {
      const binding = result.uiRuntimeProvenance.backend_python_process_binding;
      result.healthProvenance.after.binding.runtime_source_hashes.python_executable_fingerprint_sha256 =
        binding.invoked_executable_path_fingerprint_sha256;
      expect(validateUiRuntimeProvenance(result)).toEqual([]);
      result.healthProvenance.after.binding.runtime_source_hashes.python_executable_fingerprint_sha256 = "f".repeat(64);
      expect(validateUiRuntimeProvenance(result).join("\n")).toMatch(/manifest-attested bundled Python/);
      result.healthProvenance.after.binding.runtime_source_hashes.python_executable_fingerprint_sha256 =
        binding.invoked_executable_path_fingerprint_sha256;
      rmSync(fixture.pythonPath);
      symlinkSync(testExecutablePath, fixture.pythonPath);
      expect(validateUiRuntimeProvenance(result).join("\n")).toMatch(/independently bound/);
    } finally { rmSync(fixture.directory, { recursive: true, force: true }); }
  });

  it("rejects extra or omitted installed Python invocation fingerprints", () => {
    for (const extra of [true, false]) {
      const result = goodResult();
      const fixture = createInstalledPairedArtifact(result);
      try {
        const value = structuredClone(fixture.artifact.value);
        for (const phase of ["before", "after"]) {
          const runner = value.identity.runner[phase];
          const hashes = runner.installed_python_invocation_fingerprints_sha256;
          runner.installed_python_invocation_fingerprints_sha256 = extra ? [...hashes, "f".repeat(64)].sort() : hashes.slice(0, 1);
          runner.accepted_python_invocation_fingerprints_sha256 = runner.installed_python_invocation_fingerprints_sha256;
        }
        result.pairedApiArtifact = writePairedArtifactValue(fixture.directory, "bad-aliases.json", value);
        expect(validatePairedApiEvidence(result).join("\n")).toMatch(/installed-runtime runner identity/);
      } finally { rmSync(fixture.directory, { recursive: true, force: true }); }
    }
  });

  it("rejects an in-app versioned Python target that differs from exact python3", () => {
    const result = goodResult();
    result.surfaceStatus = "dual_surface_attested";
    const fixture = createInstalledPairedArtifact(result);
    try {
      result.pairedApiArtifact = fixture.artifact;
      expect(validatePairedApiEvidence(result)).toEqual([]);
      const otherVersion = path.join(path.dirname(fixture.pythonPath), "python3.11");
      writeFileSync(otherVersion, testExecutableBytes);
      chmodSync(otherVersion, 0o755);
      rmSync(fixture.pythonPath);
      symlinkSync("python3.11", fixture.pythonPath);
      expect(validatePairedApiEvidence(result).join("\n")).toMatch(
        /installed app\/Python paths|canonical path escapes or differs/,
      );
    } finally {
      rmSync(fixture.directory, { recursive: true, force: true });
    }
  });

  it.each([
    {
      name: "oversized private manifest",
      mutate: (fixture: Record<string, any>) => {
        writeFileSync(
          fixture.manifestPath,
          Buffer.concat([
            readFileSync(fixture.manifestPath),
            Buffer.alloc(1024 * 1024, 0x20),
          ]),
        );
      },
      expected: /safety limit/,
    },
    {
      name: "hard-linked private manifest",
      mutate: (fixture: Record<string, any>) => {
        linkSync(
          fixture.manifestPath,
          path.join(fixture.directory, "installed-release-hardlink.json"),
        );
      },
      expected: /exactly one filesystem link/,
    },
    {
      name: "bundled Python symlink escaping the app",
      mutate: (fixture: Record<string, any>) => {
        const escapedPython = path.join(fixture.directory, "escaped-python3");
        writeFileSync(escapedPython, readFileSync(fixture.pythonPath));
        chmodSync(escapedPython, 0o755);
        rmSync(fixture.pythonPath);
        symlinkSync(escapedPython, fixture.pythonPath);
      },
      expected: /canonical path escapes or differs from the UI app/,
    },
  ])("rejects installed paired evidence with $name", ({ mutate, expected }) => {
    const result = goodResult();
    result.surfaceStatus = "dual_surface_attested";
    const fixture = createInstalledPairedArtifact(result);
    try {
      result.pairedApiArtifact = fixture.artifact;
      expect(validatePairedApiEvidence(result)).toEqual([]);
      mutate(fixture);
      expect(validatePairedApiEvidence(result).join("\n")).toMatch(expected);
    } finally {
      rmSync(fixture.directory, { recursive: true, force: true });
    }
  });

  it("requires frozen Chat stage-2 and stage-3 parity attestations", () => {
    const result = goodResult();
    const fixture = createValidPairedArtifact(result);
    try {
      const value = structuredClone(fixture.artifact.value);
      expect(validateFrozenChatParity(value)).toEqual([]);

      delete value.paired_replays.chat_nonstream_round3;
      expect(validateFrozenChatParity(value).join("\n")).toMatch(
        /round 3 stochastic-history parity|round 3 frozen paired replay/,
      );

      const wrongMode = structuredClone(fixture.artifact.value);
      wrongMode.paired_replays.chat_nonstream_round3.request.enable_thinking = true;
      expect(validateFrozenChatParity(wrongMode).join("\n")).toMatch(
        /round 3 frozen paired replay is not bound to the transmitted Off flow body/,
      );

      const wrongBody = structuredClone(fixture.artifact.value);
      wrongBody.paired_replays.chat_nonstream_round3.request.body_sha256 =
        canonicalHash({ wrong: "body" });
      expect(validateFrozenChatParity(wrongBody).join("\n")).toMatch(
        /round 3 frozen paired replay is not bound to the transmitted Off flow body/,
      );

      const thinkingOff = structuredClone(fixture.artifact.value);
      thinkingOff.flows.direct.chat.nonstream.requests[1].enable_thinking = false;
      thinkingOff.paired_replays.chat_nonstream_round2.request.enable_thinking = false;
      expect(validateFrozenChatParity(thinkingOff)).toEqual([]);
    } finally {
      rmSync(fixture.directory, { recursive: true, force: true });
    }
  });

  it("grades reasoning against the paired request instead of forcing it on", () => {
    const result = goodResult();
    const fixture = createValidPairedArtifact(result);
    try {
      const flow = structuredClone(
        fixture.artifact.value.flows.direct.chat.stream,
      );
      for (let index = 0; index < 2; index += 1) {
        flow.requests[index].enable_thinking = false;
        flow.rounds[index].reasoning = "";
        flow.rounds[index].reasoning_chars = 0;
        flow.rounds[index].reasoning_sha256 = crypto
          .createHash("sha256")
          .update("")
          .digest("hex");
        flow.rounds[index].reasoning_delta_count = 0;
        flow.rounds[index].events = flow.rounds[index].events.filter(
          (event: Record<string, any>) => event.channel !== "reasoning",
        );
      }
      expect(validateMatrixFlow(
        flow,
        "chat",
        "stream",
        "direct/chat/stream",
        fixture.artifact.value.repo_root,
      )).toEqual([]);

      flow.requests[0].enable_thinking = true;
      expect(validateMatrixFlow(
        flow,
        "chat",
        "stream",
        "direct/chat/stream",
        fixture.artifact.value.repo_root,
      ).join("\n")).toMatch(/requested reasoning but no tool round emitted it/);
    } finally {
      rmSync(fixture.directory, { recursive: true, force: true });
    }
  });

  it("grades raw tool-stream reasoning order against the request mode", () => {
    const thinkingOff = {
      orderedChannels: ["tool", "terminal:tool_calls"],
    };
    expect(validateRawChannelOrder(
      thinkingOff,
      0,
      "direct/chat/stream-flow-round1",
      false,
    )).toEqual([]);
    expect(validateRawChannelOrder(
      thinkingOff,
      0,
      "direct/chat/stream-flow-round1",
      true,
    ).join("\n")).toMatch(/progressive reasoning/);

    const thinkingOn = {
      orderedChannels: [
        "reasoning",
        "reasoning",
        "tool",
        "terminal:tool_calls",
      ],
    };
    expect(validateRawChannelOrder(
      thinkingOn,
      0,
      "direct/chat/stream-flow-round1",
      true,
    )).toEqual([]);
    expect(validateRawChannelOrder(
      thinkingOn,
      0,
      "direct/chat/stream-flow-round1",
      false,
    )).toEqual([]);
  });

  it("does not promote a phase-0 store profile to cache-hit or tool-loop proof", () => {
    const result = goodResult();
    result.requestContract.uiActionProfile = "primary-reasoning-render-store";
    result.requestContract.uiTurnCount = 1;
    result.requestedBuiltinTools = true;

    expect(expectedUiToolCallCount(result)).toBe(0);
    expect(uiProfileRequiresPositiveCacheReuse(result)).toBe(false);
    const surfaces = deriveProvenSurfaces(result);
    expect(surfaces).not.toContain("cache_hit_telemetry");
    expect(surfaces).not.toContain("tool_loop");
    expect(surfaces).not.toContain("long_tool_loop");

    const orphanResult = {
      requestContract: {
        uiActionProfile: "primary-reasoning-render-store",
      },
      persistedToolsByMessage: [[]],
      persistedOaiCallsByMessage: [[]],
      persistedOaiResultsByMessage: [[{ tool_call_id: "orphan" }]],
      renderedDom: { messages: [] },
    };
    expect(validateExactToolLoopEvidence(orphanResult).join("\n")).toMatch(
      /tool call\/result\/status residue/,
    );

    const errorStatus = {
      ...orphanResult,
      persistedToolsByMessage: [[{ phase: "error", toolCallId: "broken" }]],
      persistedOaiResultsByMessage: [[]],
    };
    expect(validateExactToolLoopEvidence(errorStatus).join("\n")).toMatch(
      /tool call\/result\/status residue/,
    );
  });

  it("requires positive cache reuse only on probes with a complete shared block", () => {
    const store = goodResult();
    store.requestContract.uiActionProfile =
      "primary-history-paged-evict-refault";
    expect(uiProfileRequiresPositiveCacheReuse(store)).toBe(false);

    const restart = goodResult();
    restart.requestContract.uiActionProfile = "primary-restart-followup";
    expect(uiProfileRequiresPositiveCacheReuse(restart)).toBe(true);

    const sharedAnchors = releasePrimarySharedPrefix.match(
      /cache-token-\d{3}/g,
    );
    expect(sharedAnchors).toHaveLength(96);
    expect(new Set(sharedAnchors).size).toBe(96);
  });

  it("uses the canonical Ollama fallback ID when the backend omits one", () => {
    const parsed = collectOllamaStream(
      Buffer.from([
        JSON.stringify({
          message: {
            tool_calls: [{
              function: {
                name: "file_info",
                arguments: { path: "panel/package.json" },
              },
            }],
          },
          done: false,
        }),
        JSON.stringify({ message: {}, done: true, done_reason: "stop" }),
      ].join("\n")),
      "ollama-fallback-id",
    );

    expect(parsed.toolCalls).toEqual([{
      id: "ollama_call_0",
      name: "file_info",
      arguments: JSON.stringify({ path: "panel/package.json" }),
    }]);
  });

  it.each([
    {
      name: "missing",
      mutate: (value: Record<string, any>) => {
        value.raw_capture.routes.pop();
        for (const field of ["expected", "started", "finished"]) {
          value.raw_capture[field] -= 1;
        }
      },
    },
    {
      name: "extra",
      mutate: (value: Record<string, any>) => {
        const extra = structuredClone(value.raw_capture.routes[0]);
        extra.capture_label = "nonstream-flow-round4";
        value.raw_capture.routes.push(extra);
        for (const field of ["expected", "started", "finished"]) {
          value.raw_capture[field] += 1;
        }
      },
    },
    {
      name: "duplicate",
      mutate: (value: Record<string, any>) => {
        value.raw_capture.routes.push(
          structuredClone(value.raw_capture.routes[0]),
        );
        for (const field of ["expected", "started", "finished"]) {
          value.raw_capture[field] += 1;
        }
      },
    },
    {
      name: "mismatched",
      mutate: (value: Record<string, any>) => {
        value.raw_capture.routes[0].protocol = "responses";
      },
    },
  ])("rejects a $name raw route set", ({ name, mutate }) => {
    const result = goodResult();
    result.surfaceStatus = "dual_surface_attested";
    const fixture = createValidPairedArtifact(result);
    try {
      const value = structuredClone(fixture.artifact.value);
      mutate(value);
      refreshRawCaptureManifest(value);
      result.pairedApiArtifact = writePairedArtifactValue(
        fixture.directory,
        `bad-raw-routes-${name}.json`,
        value,
      );
      expect(validatePairedApiEvidence(result).join("\n")).toMatch(
        /raw capture manifest contract\/totals are not exact|raw capture routes are missing, duplicated, or unexpected/,
      );
    } finally {
      rmSync(fixture.directory, { recursive: true, force: true });
    }
  });

  it("rejects substituted producer, direct/gateway, source, and bundle identities", () => {
    const result = goodResult();
    result.surfaceStatus = "dual_surface_attested";
    const fixture = createValidPairedArtifact(result);
    try {
      const value = structuredClone(fixture.artifact.value);
      value.bases.gateway = value.bases.direct;
      value.identity.source.before.head = "stale";
      value.identity.source.after.head = "stale";
      value.identity.source.declared_head = "stale";
      value.identity.bundle.before.fingerprint_sha256 = otherSha;
      value.identity.bundle.after.fingerprint_sha256 = otherSha;
      for (const phase of ["before", "after"]) {
        value.identity.runner[phase].producer_pid =
          result.healthProvenance.after.binding.backend_pid;
        value.identity.runner[phase].producer_harness_sha256 = otherSha;
      }
      result.pairedApiArtifact = writePairedArtifactValue(
        fixture.directory,
        "substituted-identities.json",
        value,
      );
      expect(validatePairedApiEvidence(result).join("\n")).toMatch(
        /origins are not bound|source identity mismatch|bundle identity|producer executable\/harness bytes|aliases Electron\/backend/,
      );
    } finally {
      rmSync(fixture.directory, { recursive: true, force: true });
    }
  });

  it("matches paired producer/backend Python by canonical executable identity", () => {
    const result = goodResult();
    result.surfaceStatus = "dual_surface_attested";
    const fixture = createValidPairedArtifact(result);
    try {
      const alias = path.join(fixture.directory, "python3");
      symlinkSync(testExecutablePath, alias);
      const value = structuredClone(fixture.artifact.value);
      const aliasFingerprint = crypto
        .createHash("sha256")
        .update(alias)
        .digest("hex");
      const prefixFingerprint = crypto
        .createHash("sha256")
        .update(fixture.directory)
        .digest("hex");
      for (const phase of ["before", "after"]) {
        value.identity.runner[phase].python_executable_path = alias;
        value.identity.runner[phase].python_executable_fingerprint_sha256 =
          aliasFingerprint;
        value.identity.runner[phase]
          .checkout_python_invocation_fingerprints_sha256 = [aliasFingerprint];
        value.identity.runner[phase]
          .accepted_python_invocation_fingerprints_sha256 = [aliasFingerprint];
        value.identity.runner[phase].python_prefix_path = fixture.directory;
        value.identity.runner[phase].python_prefix_fingerprint_sha256 =
          prefixFingerprint;
      }
      result.pairedApiArtifact = writePairedArtifactValue(
        fixture.directory,
        "python-alias-identity.json",
        value,
      );
      expect(validatePairedApiEvidence(result)).toEqual([]);
    } finally {
      rmSync(fixture.directory, { recursive: true, force: true });
    }
  });

  it("rejects duplicate, out-of-order, and post-terminal protocol events", () => {
    const result = goodResult();
    result.surfaceStatus = "dual_surface_attested";
    const fixture = createValidPairedArtifact(result);
    try {
      const value = structuredClone(fixture.artifact.value);
      const round = value.flows.direct.chat.stream.rounds[0];
      round.terminals.push("DONE");
      round.events.push({
        at_ms: 0,
        channel: "content",
        kind: "chat.content.delta",
      });
      value.flows.direct.chat.stream.terminal_classification[0].values =
        round.terminals;
      result.pairedApiArtifact = writePairedArtifactValue(
        fixture.directory,
        "bad-terminals.json",
        value,
      );
      expect(validatePairedApiEvidence(result).join("\n")).toMatch(
        /terminal status\/count\/order|non-terminal output after terminalization|timestamps are not monotonic/,
      );
    } finally {
      rmSync(fixture.directory, { recursive: true, force: true });
    }
  });

  it("rejects reversed Chat and Anthropic native terminal arrays", () => {
    const result = goodResult();
    result.surfaceStatus = "dual_surface_attested";
    const fixture = createValidPairedArtifact(result);
    try {
      const value = structuredClone(fixture.artifact.value);
      for (const [baseLabel, protocol] of [
        ["direct", "chat"],
        ["gateway", "anthropic"],
      ]) {
        const flow = value.flows[baseLabel][protocol].stream;
        flow.rounds[2].terminals.reverse();
        const terminalEvents = flow.rounds[2].events.filter(
          (event: Record<string, any>) => event.channel === "terminal",
        );
        terminalEvents.reverse().forEach(
          (event: Record<string, any>, index: number) => {
            const target = flow.rounds[2].events.findIndex(
              (candidate: Record<string, any>) =>
                candidate.channel === "terminal"
                && candidate.kind === event.kind,
            );
            flow.rounds[2].events[target] = {
              ...event,
              at_ms: 100 + index,
            };
          },
        );
        flow.terminal_classification[2].values =
          structuredClone(flow.rounds[2].terminals);
      }
      result.pairedApiArtifact = writePairedArtifactValue(
        fixture.directory,
        "reversed-native-terminals.json",
        value,
      );
      expect(validatePairedApiEvidence(result).join("\n")).toMatch(
        /terminal status\/count\/order|terminal classifications are not exact/,
      );
    } finally {
      rmSync(fixture.directory, { recursive: true, force: true });
    }
  });

  it("rejects malformed tool schemas, arguments, result linkage, and missing reasoning chain", () => {
    const result = goodResult();
    result.surfaceStatus = "dual_surface_attested";
    const fixture = createValidPairedArtifact(result);
    try {
      const value = structuredClone(fixture.artifact.value);
      const flow = value.flows.gateway.responses.stream;
      flow.requests[0].tool_contracts[0].parameters.required = [];
      flow.rounds[0].tool_calls[0].arguments = { path: "wrong.json" };
      flow.executions[0].output_sha256 = otherSha;
      flow.rounds[1].reasoning_chars = 0;
      flow.rounds[1].reasoning_sha256 = crypto
        .createHash("sha256")
        .update("")
        .digest("hex");
      result.pairedApiArtifact = writePairedArtifactValue(
        fixture.directory,
        "bad-tools.json",
        value,
      );
      expect(validatePairedApiEvidence(result).join("\n")).toMatch(
        /tool schema is not exact|exact tool name\/arguments|not exactly linked|lacks a separate reasoning payload|does not prove reasoning/,
      );
    } finally {
      rmSync(fixture.directory, { recursive: true, force: true });
    }
  });

  it("rejects tool-choice, continuation, history-order, and direct/gateway request drift", () => {
    const result = goodResult();
    result.surfaceStatus = "dual_surface_attested";
    const fixture = createValidPairedArtifact(result);
    try {
      const value = structuredClone(fixture.artifact.value);
      value.second_tool_choice = "auto";
      value.flows.gateway.chat.nonstream.requests[1].tool_choice = "auto";
      value.flows.direct.responses.stream.requests[1].previous_response_id =
        "wrong-response";
      value.flows.direct.anthropic.nonstream.requests[1].tool_history_linkage.reverse();
      value.flows.gateway.ollama.stream.requests[0].canonical_body_sha256 =
        otherSha;
      result.pairedApiArtifact = writePairedArtifactValue(
        fixture.directory,
        "bad-request-chain.json",
        value,
      );
      expect(validatePairedApiEvidence(result).join("\n")).toMatch(
        /exact passing vmlx-agentic|tool_choice is not exact|continuation ID is not exact|tool history order is not exact|canonical request bodies are not byte-parity equivalent/,
      );
    } finally {
      rmSync(fixture.directory, { recursive: true, force: true });
    }
  });

  it("rejects a forged raw manifest and artifact path traversal", () => {
    const result = goodResult();
    result.surfaceStatus = "dual_surface_attested";
    const fixture = createValidPairedArtifact(result);
    try {
      const value = structuredClone(fixture.artifact.value);
      const forgedManifest = structuredClone(value.raw_capture);
      for (const field of [
        "manifest_file",
        "manifest_path",
        "manifest_sha256",
        "run_directory",
      ]) delete forgedManifest[field];
      forgedManifest.routes[0].artifacts[0].body_file = "../escape.bin";
      const forgedManifestPath = path.join(
        value.raw_capture.run_directory,
        "forged-manifest.json",
      );
      const forgedManifestText = `${JSON.stringify(forgedManifest, null, 2)}\n`;
      writePrivateArtifactFile(forgedManifestPath, forgedManifestText);
      value.raw_capture = {
        ...forgedManifest,
        manifest_file: "forged-manifest.json",
        manifest_path: forgedManifestPath,
        run_directory: value.raw_capture.run_directory,
        manifest_sha256: crypto
          .createHash("sha256")
          .update(forgedManifestText)
          .digest("hex"),
      };
      result.pairedApiArtifact = writePairedArtifactValue(
        fixture.directory,
        "forged-manifest-proof.json",
        value,
      );
      expect(validatePairedApiEvidence(result).join("\n")).toMatch(
        /filename is unsafe|raw capture/,
      );
    } finally {
      rmSync(fixture.directory, { recursive: true, force: true });
    }
  });

  it("rejects reordered raw reasoning/tool bytes and incomplete Responses terminal status", () => {
    const result = goodResult();
    result.surfaceStatus = "dual_surface_attested";
    const fixture = createValidPairedArtifact(result);
    try {
      const value = structuredClone(fixture.artifact.value);
      const chatCall =
        value.flows.direct.chat.stream.rounds[0].tool_calls[0];
      rewriteRawCaptureBody(value, {
        baseLabel: "direct",
        protocol: "chat",
        captureLabel: "stream-flow-round1",
        body: sse(
          {
            choices: [{
              delta: {
                tool_calls: [{
                  index: 0,
                  id: chatCall.id,
                  function: {
                    name: chatCall.name,
                    arguments: JSON.stringify(chatCall.arguments),
                  },
                }],
              },
            }],
          },
          { choices: [{ delta: { reasoning_content: "Reason chat 1 A." } }] },
          { choices: [{ delta: { reasoning_content: "B." } }] },
          { choices: [{ delta: {}, finish_reason: "tool_calls" }] },
          "[DONE]",
        ),
      });

      const final =
        value.flows.gateway.responses.stream.expected_final;
      rewriteRawCaptureBody(value, {
        baseLabel: "gateway",
        protocol: "responses",
        captureLabel: "stream-flow-round3",
        body: sse(
          { type: "response.output_text.delta", delta: final.slice(0, 10) },
          { type: "response.output_text.delta", delta: final.slice(10) },
          {
            type: "response.completed",
            response: { status: "incomplete" },
          },
        ),
      });
      result.pairedApiArtifact = writePairedArtifactValue(
        fixture.directory,
        "bad-raw-order-terminal.json",
        value,
      );
      expect(validatePairedApiEvidence(result).join("\n")).toMatch(
        /raw stream does not preserve progressive reasoning before tool output/,
      );
      expect(validatePairedApiEvidence(result).join("\n")).toMatch(
        /raw bytes do not reproduce the public flow\/request evidence/,
      );
    } finally {
      rmSync(fixture.directory, { recursive: true, force: true });
    }
  });

  it("rejects Responses nonstream raw bytes without explicit completed status", () => {
    const result = goodResult();
    result.surfaceStatus = "dual_surface_attested";
    const fixture = createValidPairedArtifact(result);
    try {
      const value = structuredClone(fixture.artifact.value);
      const expectedFinal =
        value.flows.gateway.responses.nonstream.expected_final;
      rewriteRawCaptureBody(value, {
        baseLabel: "gateway",
        protocol: "responses",
        captureLabel: "nonstream-flow-round3",
        body: JSON.stringify({
          id: "missing-status",
          output: [{
            type: "message",
            content: [{ type: "output_text", text: expectedFinal }],
          }],
        }),
      });
      result.pairedApiArtifact = writePairedArtifactValue(
        fixture.directory,
        "missing-nonstream-status.json",
        value,
      );
      expect(validatePairedApiEvidence(result).join("\n")).toMatch(
        /raw bytes do not reproduce the public flow\/request evidence/,
      );
    } finally {
      rmSync(fixture.directory, { recursive: true, force: true });
    }
  });

  it.each([
    {
      name: "Chat missing finish_reason",
      protocol: "chat",
      body: (expectedFinal: string) => JSON.stringify({
        id: "chat-missing-terminal",
        choices: [{
          message: { content: expectedFinal },
        }],
      }),
    },
    {
      name: "Anthropic incorrect stop_reason",
      protocol: "anthropic",
      body: (expectedFinal: string) => JSON.stringify({
        id: "anthropic-wrong-terminal",
        content: [{ type: "text", text: expectedFinal }],
        stop_reason: "max_tokens",
      }),
    },
    {
      name: "Ollama missing done terminal",
      protocol: "ollama",
      body: (expectedFinal: string) => JSON.stringify({
        message: { content: expectedFinal },
        done: false,
        done_reason: "stop",
      }),
    },
  ])("rejects $name in self-consistent nonstream raw bytes", ({
    protocol,
    body,
  }) => {
    const result = goodResult();
    result.surfaceStatus = "dual_surface_attested";
    const fixture = createValidPairedArtifact(result);
    try {
      const value = structuredClone(fixture.artifact.value);
      const expectedFinal =
        value.flows.direct[protocol].nonstream.expected_final;
      rewriteRawCaptureBody(value, {
        baseLabel: "direct",
        protocol,
        captureLabel: "nonstream-flow-round3",
        body: body(expectedFinal),
      });
      result.pairedApiArtifact = writePairedArtifactValue(
        fixture.directory,
        `bad-${protocol}-nonstream-terminal.json`,
        value,
      );
      expect(validatePairedApiEvidence(result).join("\n")).toMatch(
        /raw bytes do not reproduce the public flow\/request evidence/,
      );
    } finally {
      rmSync(fixture.directory, { recursive: true, force: true });
    }
  });

  it("rejects cross-origin health and forged prepared-body parity", () => {
    const result = goodResult();
    result.surfaceStatus = "dual_surface_attested";
    const fixture = createValidPairedArtifact(result);
    try {
      const value = structuredClone(fixture.artifact.value);
      value.identity.health.gateway.before.url =
        "http://127.0.0.1:8000/health";
      rewriteRawCaptureMetadata(value, {
        baseLabel: "direct",
        protocol: "chat",
        captureLabel: "stream-flow-round1",
        mutate: (metadata, artifact) => {
          metadata.request.prepared_payload_canonical_body_sha256 = otherSha;
          artifact.prepared_payload_canonical_body_sha256 = otherSha;
        },
      });
      result.pairedApiArtifact = writePairedArtifactValue(
        fixture.directory,
        "unbound-health-prepared-body.json",
        value,
      );
      const failures = validatePairedApiEvidence(result).join("\n");
      expect(failures).toMatch(/health URL origin is not bound/);
      expect(failures).toMatch(/raw body\/metadata\/request\/response binding is invalid/);
    } finally {
      rmSync(fixture.directory, { recursive: true, force: true });
    }
  });

  it("retains only stable DOM proof identifiers, not answer, reasoning, argument, or result payloads", () => {
    for (const relative of [
      "src/renderer/src/components/chat/MessageBubble.tsx",
      "src/renderer/src/components/chat/ReasoningBox.tsx",
      "src/renderer/src/components/chat/InlineToolCall.tsx",
      "src/renderer/src/components/chat/ToolCallStatus.tsx",
    ]) {
      const source = readFileSync(path.join(process.cwd(), relative), "utf8");
      expect(source).not.toMatch(
        /data-vmlx-proof-(?:answer-source|reasoning-source|tool-arguments|tool-result)/,
      );
    }
  });

  it("keeps a valid raw API matrix separate but not dual when UI request correlation is partial", () => {
    const result = goodResult();
    result.requestCorrelation = {
      status: "partial_product_support_missing",
      turns: [],
    };
    result.surfaceStatus = "partial_dual_surface_uncorrelated";
    const fixture = createValidPairedArtifact(result);
    try {
      result.pairedApiArtifact = fixture.artifact;
      expect(validatePairedApiEvidence(result)).toEqual([]);
      const surfaces = deriveProvenSurfaces(result);
      expect(surfaces).toContain("separate_raw_api");
      expect(surfaces).not.toContain("dual_surface");
    } finally {
      rmSync(fixture.directory, { recursive: true, force: true });
    }
  });

  it("keeps top-level status PARTIAL until exact UI request correlation is verified", () => {
    const verified = goodResult();
    const fixture = createValidPairedArtifact(verified);
    try {
      verified.pairedApiArtifact = fixture.artifact;
      expect(validatePairedApiEvidence(verified)).toEqual([]);
      expect(applyTopLevelCorrelationStatus(verified)).toMatchObject({
        status: "pass",
        pass: true,
        surfaceStatus: "dual_surface_attested",
      });

      const uiOnly = goodResult();
      expect(applyTopLevelCorrelationStatus(uiOnly)).toMatchObject({
        status: "partial",
        pass: false,
        surfaceStatus: "partial_ui_only",
      });

      const partial = structuredClone(verified);
      partial.requestCorrelation = {
        status: "partial_product_support_missing",
        turns: [],
      };
      expect(applyTopLevelCorrelationStatus(partial)).toMatchObject({
        status: "partial",
        pass: false,
        surfaceStatus: "partial_dual_surface_uncorrelated",
      });

      const rendererFailed = structuredClone(verified);
      expect(
        applyTopLevelCorrelationStatus(rendererFailed, {
          rendererFailed: true,
        }),
      ).toMatchObject({
        status: "fail",
        pass: false,
        surfaceStatus: "partial_ui_only",
      });

      expect(
        applyAssertionFailureStatus(
          structuredClone(verified),
          Object.assign(new Error("bound assertion failed"), {
            failures: ["bound assertion failed"],
          }),
        ),
      ).toMatchObject({
        status: "fail",
        pass: false,
        failureStage: "release_assertions",
        assertionFailures: ["bound assertion failed"],
      });
    } finally {
      rmSync(fixture.directory, { recursive: true, force: true });
    }
  });

  it("requires an exact parent-owned attach-only Electron lifecycle", () => {
    expect(validateAttachOnlyLifecycle({
      cdpUrl: "http://127.0.0.1:9335",
      electronPid: 4242,
      owner: "parent",
      teardownAllowed: false,
    })).toEqual([]);
    expect(validateAttachOnlyLifecycle({
      cdpUrl: "http://example.com:9335",
      electronPid: 0,
      owner: "child",
      teardownAllowed: true,
    }).join("\n")).toMatch(
      /not an exact loopback origin|PID is invalid|not parent-owned|allowed to tear down/,
    );
  });

  it("binds V5 Single model proof to visible DOM controls and live gateway status", () => {
    const result = {
      ownedRunIntent: { phase_index: 0 },
      gatewaySingleModelMode: {
        serverSettingsControl: {
          selector: '[data-vmlx-control="server-settings"]',
          visible: true,
          ariaPressedAfterOpen: "true",
          ariaPressedAfterClose: "false",
        },
        toggleControl: {
          selector: '[data-vmlx-control="gateway-single-model-mode"]',
          visible: true,
          ariaPressedBefore: "true",
          ariaPressedAfter: "true",
          observedAlreadyOn: true,
          clickedToEnable: false,
        },
        gatewayStatusAfterToggle: {
          running: true,
          singleModelMode: true,
        },
        gatewayStatusImmediatelyBeforeStart: {
          running: true,
          singleModelMode: true,
        },
        persistedSettingImmediatelyBeforeStart: {
          key: "gateway_single_model_mode",
          value: "true",
          source: "window.api.settings.get",
        },
      },
    };
    expect(validateGatewaySingleModelEvidence(result)).toEqual([]);

    const enabledByClick = structuredClone(result);
    enabledByClick.gatewaySingleModelMode.toggleControl.ariaPressedBefore =
      "false";
    enabledByClick.gatewaySingleModelMode.toggleControl.observedAlreadyOn =
      false;
    enabledByClick.gatewaySingleModelMode.toggleControl.clickedToEnable = true;
    expect(validateGatewaySingleModelEvidence(enabledByClick)).toEqual([]);

    const staleDom = structuredClone(result);
    staleDom.gatewaySingleModelMode.toggleControl.ariaPressedAfter = "false";
    expect(validateGatewaySingleModelEvidence(staleDom).join("\n")).toMatch(
      /visible Server settings toggle/,
    );

    const staleGateway = structuredClone(result);
    staleGateway.gatewaySingleModelMode.gatewayStatusImmediatelyBeforeStart
      .singleModelMode = false;
    expect(validateGatewaySingleModelEvidence(staleGateway).join("\n")).toMatch(
      /live gateway status before Start/,
    );

    const selfDerivedOnly = structuredClone(result);
    delete (selfDerivedOnly.gatewaySingleModelMode as any)
      .persistedSettingImmediatelyBeforeStart;
    expect(validateGatewaySingleModelEvidence(selfDerivedOnly).join("\n"))
      .toMatch(/independently read the persisted gateway setting/);
  });

  it("requires exact phase-5 visible Stop and proof-backend teardown evidence", () => {
    const result = {
      baseUrl: "http://127.0.0.1:8017",
      backend: { pid: 4201 },
      ownedRunIntent: { phase_index: 5 },
      releaseEvidence: { phase_index: 5 },
      requestedRetainedPids: [7101, 7102],
      uiBackendBinding: { electron_pid: 4301 },
      uiSessionAttestation: { value: { gateway_pid: 4301 } },
      session: { id: "owned-proof-session" },
      finalPhaseStopEvidence: {
        releaseSentinelConsumed: true,
        phaseIndex: 5,
        visibleControl: {
          selector: '[data-vmlx-control="session-stop"]',
          exactSelector:
            '[data-vmlx-control="session-stop"]'
            + '[data-vmlx-session-id="owned-proof-session"]',
          clicked: true,
          label: "Stop",
          sessionId: "owned-proof-session",
        },
        session: {
          id: "owned-proof-session",
          before: { status: "running", pid: 4201, port: 8017 },
          after: { status: "stopped", pid: null, port: 8017 },
          pidClearSemantics: "nullable_pid_cleared",
          portClearSemantics: "non_nullable_endpoint_retained",
        },
        backend: {
          backend_pid: 4201,
          port: 8017,
          backend_process_gone: true,
          listener_gone: true,
          observed_listener_pids: [],
        },
        survivors: {
          before: {
            stage: "before_visible_stop",
            expected_retained_pids: [7101, 7102],
            processes: [
              { role: "parent_electron", pid: 4301, alive: true },
              { role: "parent_gateway", pid: 4301, alive: true },
              { role: "explicit_retained_1", pid: 7101, alive: true },
              { role: "explicit_retained_2", pid: 7102, alive: true },
            ],
          },
          after: {
            stage: "after_backend_teardown",
            expected_retained_pids: [7101, 7102],
            processes: [
              { role: "parent_electron", pid: 4301, alive: true },
              { role: "parent_gateway", pid: 4301, alive: true },
              { role: "explicit_retained_1", pid: 7101, alive: true },
              { role: "explicit_retained_2", pid: 7102, alive: true },
            ],
          },
        },
      },
    };
    expect(validateFinalPhaseStopEvidence(result)).toEqual([]);
    expect(validateFinalPhaseStopEvidence({
      ...result,
      ownedRunIntent: { phase_index: 4 },
      releaseEvidence: null,
      finalPhaseStopEvidence: null,
    })).toEqual([]);

    const retained = structuredClone(result);
    retained.finalPhaseStopEvidence.backend.backend_process_gone = false;
    expect(validateFinalPhaseStopEvidence(retained).join("\n")).toMatch(
      /backend process\/listener teardown/,
    );

    const wrongDbBinding = structuredClone(result);
    wrongDbBinding.finalPhaseStopEvidence.session.before.pid = 9999;
    expect(validateFinalPhaseStopEvidence(wrongDbBinding).join("\n")).toMatch(
      /bound to the backend PID\/port/,
    );

    const parentKilled = structuredClone(result);
    parentKilled.finalPhaseStopEvidence.survivors.after.processes[0].alive =
      false;
    expect(validateFinalPhaseStopEvidence(parentKilled).join("\n")).toMatch(
      /parent Electron, gateway, and explicit retained PIDs/,
    );

    const distinctParentPids = structuredClone(result);
    distinctParentPids.uiSessionAttestation.value.gateway_pid = 4401;
    for (const snapshot of [
      distinctParentPids.finalPhaseStopEvidence.survivors.before,
      distinctParentPids.finalPhaseStopEvidence.survivors.after,
    ]) {
      snapshot.processes.find(
        (process) => process.role === "parent_gateway",
      )!.pid = 4401;
    }
    expect(validateFinalPhaseStopEvidence(distinctParentPids).join("\n"))
      .toMatch(/parent Electron, gateway, and explicit retained PIDs/);

    const retainedAliasesParent = structuredClone(result);
    retainedAliasesParent.requestedRetainedPids[0] = 4301;
    for (const snapshot of [
      retainedAliasesParent.finalPhaseStopEvidence.survivors.before,
      retainedAliasesParent.finalPhaseStopEvidence.survivors.after,
    ]) {
      snapshot.expected_retained_pids[0] = 4301;
      snapshot.processes.find(
        (process) => process.role === "explicit_retained_1",
      )!.pid = 4301;
    }
    expect(validateFinalPhaseStopEvidence(retainedAliasesParent).join("\n"))
      .toMatch(/parent Electron, gateway, and explicit retained PIDs/);

    const retainedAliasesBackend = structuredClone(result);
    retainedAliasesBackend.requestedRetainedPids[0] = 4201;
    for (const snapshot of [
      retainedAliasesBackend.finalPhaseStopEvidence.survivors.before,
      retainedAliasesBackend.finalPhaseStopEvidence.survivors.after,
    ]) {
      snapshot.expected_retained_pids[0] = 4201;
      snapshot.processes.find(
        (process) => process.role === "explicit_retained_1",
      )!.pid = 4201;
    }
    expect(validateFinalPhaseStopEvidence(retainedAliasesBackend).join("\n"))
      .toMatch(/parent Electron, gateway, and explicit retained PIDs/);
  });

  it("always runs post-sentinel cleanup and preserves the original work error", async () => {
    const original = new Error("paired artifact read failed");
    let cleanupCalls = 0;
    await expect(runPostSentinelWorkWithCleanup({
      work: async () => {
        throw original;
      },
      cleanup: async () => {
        cleanupCalls += 1;
        throw new Error("cleanup also failed");
      },
    })).rejects.toBe(original);
    expect(cleanupCalls).toBe(1);
    expect((original as any).cleanupError?.message).toBe(
      "cleanup also failed",
    );
  });

  it("attempts visible Stop cleanup after pre-Stop survivor attestation fails", async () => {
    const preAttestationError = new Error(
      "pre-Stop survivor attestation failed",
    );
    const events: string[] = [];
    await expect(runPostSentinelWorkWithCleanup({
      work: async () => {
        events.push("pre-survivor-attestation");
        throw preAttestationError;
      },
      cleanup: async () => {
        events.push("visible-stop");
        events.push("exact-backend-teardown");
      },
    })).rejects.toBe(preAttestationError);
    expect(events).toEqual([
      "pre-survivor-attestation",
      "visible-stop",
      "exact-backend-teardown",
    ]);
  });

  it("parses only explicit unique retained process PIDs", () => {
    expect(parseExplicitPidList("7101, 7102 7103")).toEqual([
      7101,
      7102,
      7103,
    ]);
    expect(() => parseExplicitPidList("7101,7101")).toThrow(/duplicate/);
    expect(() => parseExplicitPidList("7101,abc")).toThrow(
      /integer PIDs greater than 1/,
    );
  });

  it("uses the real visible settings and Stop controls in the V5 harness", () => {
    const sessionView = readFileSync(
      path.resolve("src/renderer/src/components/sessions/SessionView.tsx"),
      "utf8",
    );
    const chatToolbar = readFileSync(
      path.resolve("src/renderer/src/components/layout/ChatModeToolbar.tsx"),
      "utf8",
    );
    const serverDrawer = readFileSync(
      path.resolve(
        "src/renderer/src/components/sessions/ServerSettingsDrawer.tsx",
      ),
      "utf8",
    );
    const harness = readFileSync(
      path.resolve("scripts/live-real-ui-model-proof.mjs"),
      "utf8",
    );
    expect(sessionView).toContain('data-vmlx-control="server-settings"');
    expect(sessionView).toContain("aria-pressed={showServerSettings}");
    expect(chatToolbar).toContain('data-vmlx-control="server-settings"');
    expect(serverDrawer).toContain(
      'data-vmlx-control="gateway-single-model-mode"',
    );
    expect(serverDrawer).toContain("aria-pressed={singleModelMode}");
    expect(sessionView).toContain('data-vmlx-control="session-start"');
    expect(sessionView).toContain('data-vmlx-control="session-stop"');
    expect(sessionView).toContain("data-vmlx-session-id={session.id}");

    const enableStart = harness.indexOf(
      "const serverSettingsControl = await waitFor",
    );
    const startControl = harness.indexOf(
      "const startButton = await new Promise",
      enableStart,
    );
    expect(enableStart).toBeGreaterThan(0);
    expect(startControl).toBeGreaterThan(enableStart);
    const enableBlock = harness.slice(enableStart, startControl);
    expect(enableBlock).toContain(
      '[data-vmlx-control="gateway-single-model-mode"]',
    );
    expect(enableBlock).toContain("singleModelToggle.click()");
    expect(enableBlock).toContain("window.api.gateway.getStatus()");
    expect(enableBlock).toContain(
      "window.api.settings.get('gateway_single_model_mode')",
    );
    expect(enableBlock).toContain(
      "persistedSettingImmediatelyBeforeStart",
    );
    expect(harness).toContain(
      '[data-vmlx-control="session-start"][data-vmlx-session-id="',
    );
    expect(harness).toContain(
      '[data-vmlx-control="session-stop"][data-vmlx-session-id="',
    );
    expect(harness).not.toContain(
      "return label === 'Start'",
    );
    expect(harness).toContain(
      "releaseGatewayPid !== expectedElectronPid",
    );
    expect(harness).toContain(
      "releaseRetainedPids.includes(expectedElectronPid)",
    );
    expect(harness).toContain(
      "cdpProcessBinding.listener_pid !== expectedElectronPid",
    );

    const finalStopStart = harness.indexOf("let finalPhaseStopEvidence = null");
    const finalStopEnd = harness.indexOf("const result = {", finalStopStart);
    const finalStopBlock = harness.slice(finalStopStart, finalStopEnd);
    expect(finalStopBlock).toContain(
      "activeReleasePhase?.phase_index === 5",
    );
    expect(finalStopBlock).toContain("releaseEvidence");
    expect(finalStopBlock).toContain(
      '[data-vmlx-control="session-stop"]',
    );
    expect(finalStopBlock).toContain("control.click()");
    expect(finalStopBlock).toContain("window.api.sessions.get(sessionId)");
    expect(finalStopBlock).not.toContain("window.api.sessions.stop");
    expect(finalStopBlock).toContain("waitForExactProofBackendTeardown");
    expect(finalStopBlock).toContain("runPostSentinelWorkWithCleanup");
    expect(
      finalStopBlock.match(/runPostSentinelWorkWithCleanup/g),
    ).toHaveLength(2);
    expect(finalStopBlock).toContain("attestExactSurvivorPids");
    expect(finalStopBlock).toContain("nullable_pid_cleared");
    expect(finalStopBlock).toContain("non_nullable_endpoint_retained");
  });

  it("requires one canonical parent-owned six-phase run intent", () => {
    const directory = mkdtempSync(path.join(tmpdir(), "vmlx-owned-intent-"));
    try {
      const primaryBundle = path.join(directory, "primary");
      const nativeBundle = path.join(directory, "native");
      mkdirSync(primaryBundle);
      mkdirSync(nativeBundle);
      const value = ownedRunIntent(primaryBundle, nativeBundle, directory);
      const raw = JSON.stringify(value);
      const rawSha = crypto.createHash("sha256").update(raw).digest("hex");
      const opened = {
        opened_nofollow: true,
        nlink: 1,
        mode: 0o600,
        sha256: rawSha,
        value,
      };
      const options = {
        runId: "run",
        nonce: "nonce",
        expectedSha256: rawSha,
        expectedSourceCommit: "3".repeat(40),
        expectedSourceTree: "4".repeat(40),
        expectedUiHarnessSha256: (
          (value.harnesses as Record<string, Record<string, string>>).ui.sha256
        ),
        harnessRoot: directory,
        activePhaseIndex: 0,
        activeModel: "primary-model",
        activeModelBundlePath: primaryBundle,
        expectedDirectBaseUrl: "http://127.0.0.1:8001",
        expectedGatewayBaseUrl: "http://127.0.0.1:8088",
      };
      expect(validateOwnedRunIntent(opened, options)).toEqual([]);
      expect(validateOwnedRunIntent(opened, {
        ...options,
        activePhaseIndex: 5,
        activeModel: "native-model",
        activeModelBundlePath: nativeBundle,
        expectedDirectBaseUrl: "http://127.0.0.1:8002",
      })).toEqual([]);

      const staleReleaseSchema = structuredClone(value);
      staleReleaseSchema.schema = "vmlx-r19-owned-run-intent-v5";
      staleReleaseSchema.canonical_sha256 = canonicalHash(
        Object.fromEntries(
          Object.entries(staleReleaseSchema).filter(
            ([key]) => key !== "canonical_sha256",
          ),
        ),
      );
      expect(validateOwnedRunIntent(
        { ...opened, value: staleReleaseSchema },
        options,
      ).join("\n")).toMatch(/schema\/run\/nonce does not match/);

      const reordered = structuredClone(value);
      [
        (reordered.phase_plan as Array<Record<string, unknown>>)[0],
        (reordered.phase_plan as Array<Record<string, unknown>>)[1],
      ] = [
        (reordered.phase_plan as Array<Record<string, unknown>>)[1],
        (reordered.phase_plan as Array<Record<string, unknown>>)[0],
      ];
      reordered.canonical_sha256 = canonicalHash(
        Object.fromEntries(
          Object.entries(reordered).filter(([key]) => key !== "canonical_sha256"),
        ),
      );
      expect(validateOwnedRunIntent(
        { ...opened, value: reordered },
        options,
      ).join("\n")).toMatch(/phase 0 policy\/order/);

      const forged = structuredClone(value);
      (forged.harnesses as Record<string, unknown>).extra = {
        relative_path: "unexpected",
        sha256: sha,
      };
      (forged.phase_plan as Array<Record<string, unknown>>)[0].unexpected = true;
      (forged as Record<string, unknown>).unknown = true;
      expect(validateOwnedRunIntent(
        { ...opened, value: forged },
        options,
      ).join("\n")).toMatch(/top-level fields are missing or unexpected/);

      const crossOrigin = structuredClone(value);
      crossOrigin.direct_health_url = "http://127.0.0.1:9999/health";
      crossOrigin.canonical_sha256 = canonicalHash(
        Object.fromEntries(
          Object.entries(crossOrigin).filter(([key]) => key !== "canonical_sha256"),
        ),
      );
      expect(validateOwnedRunIntent(
        { ...opened, value: crossOrigin },
        options,
      ).join("\n")).toMatch(/primary\/native\/gateway endpoint binding is invalid/);

      const sharedDirect = structuredClone(value);
      sharedDirect.native_direct_base_url = sharedDirect.direct_base_url;
      sharedDirect.native_direct_health_url = sharedDirect.direct_health_url;
      sharedDirect.canonical_sha256 = canonicalHash(
        Object.fromEntries(
          Object.entries(sharedDirect).filter(([key]) => key !== "canonical_sha256"),
        ),
      );
      expect(validateOwnedRunIntent(
        { ...opened, value: sharedDirect },
        options,
      ).join("\n")).toMatch(/primary\/native\/gateway endpoint binding is invalid/);

      const nativePath = structuredClone(value);
      nativePath.native_direct_base_url = "http://127.0.0.1:8002/extra";
      nativePath.canonical_sha256 = canonicalHash(
        Object.fromEntries(
          Object.entries(nativePath).filter(([key]) => key !== "canonical_sha256"),
        ),
      );
      expect(validateOwnedRunIntent(
        { ...opened, value: nativePath },
        options,
      ).join("\n")).toMatch(/primary\/native\/gateway endpoint binding is invalid/);
    } finally {
      rmSync(directory, { recursive: true, force: true });
    }
  });

  it("accepts only the exact parent-owned prior phase session attestation", () => {
    const directory = mkdtempSync(
      path.join(tmpdir(), "vmlx-reuse-session-attestation-"),
    );
    try {
      const activePhase = {
        phase_index: 2,
        representative_id: "primary_tq_supported",
        bundle_fingerprint_sha256: "1".repeat(64),
      };
      const options = {
        runId: "run",
        nonce: "nonce",
        runIntentSha256: "2".repeat(64),
        sessionId: "session-primary",
        activePhase,
        model: "primary-model",
        modelBundlePath: directory,
        electronPid: 4103,
        cdpOrigin: "http://127.0.0.1:9335",
        gatewayPid: 4102,
        gatewayBaseUrl: "http://127.0.0.1:8080",
        sourceCommit: "3".repeat(40),
        sourceTree: "4".repeat(40),
      };
      const opened = {
        opened_nofollow: true,
        nlink: 1,
        mode: 0o600,
        value: {
          schema: "vmlx-r20-owned-ui-session-attestation-v5",
          run_id: options.runId,
          nonce: options.nonce,
          run_intent_sha256: options.runIntentSha256,
          phase_index: 1,
          representative_id: activePhase.representative_id,
          session_id: options.sessionId,
          model: options.model,
          model_bundle_path: directory,
          bundle_fingerprint_sha256:
            activePhase.bundle_fingerprint_sha256,
          electron_pid: options.electronPid,
          cdp_origin: options.cdpOrigin,
          gateway_pid: options.gatewayPid,
          gateway_base_url: options.gatewayBaseUrl,
          lifecycle_owner: "parent",
          source_commit: options.sourceCommit,
          source_tree: options.sourceTree,
          created_at: new Date().toISOString(),
        },
      };
      expect(validateOwnedReuseSessionAttestation(opened, options)).toEqual([]);

      const staleSchema = structuredClone(opened);
      staleSchema.value.schema = "vmlx-r19-owned-ui-session-attestation-v5";
      expect(
        validateOwnedReuseSessionAttestation(staleSchema, options).join("\n"),
      ).toMatch(/stale, wrong-phase, wrong-model|not owned/);

      for (const [field, value] of [
        ["phase_index", 0],
        ["model", "wrong-model"],
        ["lifecycle_owner", "ui-proof-child"],
        ["electron_pid", 9999],
      ] as const) {
        const tampered = structuredClone(opened);
        (tampered.value as Record<string, unknown>)[field] = value;
        expect(
          validateOwnedReuseSessionAttestation(tampered, options).join("\n"),
        ).toMatch(/stale, wrong-phase, wrong-model|not owned/);
      }
    } finally {
      rmSync(directory, { recursive: true, force: true });
    }
  });

  it("rejects stale, mismatched, preexisting, and symlink release sentinels", async () => {
    const now = Date.now();
    const runIntentSha = "d".repeat(64);
    const sessionAttestationSha = "e".repeat(64);
    const activePhase = {
      phase_index: 0,
      phase_name: "primary_ssd_only_store",
      representative_id: "primary_tq_supported",
      bundle_role: "primary",
      cache_policy: "q4",
      paged_ram: false,
      ui_action_profile: "primary-reasoning-render-store",
      ui_turn_count: 1,
      api_action_profile: "full-agentic-plus-cache-store",
      model: "primary-model",
      bundle_fingerprint_sha256: "1".repeat(64),
    };
    const validOpened = {
      opened_nofollow: true,
      nlink: 1,
      mode: 0o600,
      value: {
        schema: "vmlx-r20-owned-ui-release-v5",
        run_id: "run",
        nonce: "nonce",
        session_id: "session",
        ...activePhase,
        run_intent_sha256: runIntentSha,
        ui_session_attestation_sha256: sessionAttestationSha,
        api_capture_sha256: otherSha,
        cache_capture_sha256: "c".repeat(64),
        released_at: new Date(now + 1000).toISOString(),
      },
    };
    expect(validateOwnedUiReleaseSentinel(validOpened, {
      runId: "run",
      nonce: "nonce",
      sessionId: "session",
      orchestrated: true,
      runIntentSha256: runIntentSha,
      uiSessionAttestationSha256: sessionAttestationSha,
      activePhase,
      notBeforeMs: now,
    })).toEqual([]);

    const staleSchema = structuredClone(validOpened);
    staleSchema.value.schema = "vmlx-r19-owned-ui-release-v5";
    expect(validateOwnedUiReleaseSentinel(staleSchema, {
      runId: "run",
      nonce: "nonce",
      sessionId: "session",
      orchestrated: true,
      runIntentSha256: runIntentSha,
      uiSessionAttestationSha256: sessionAttestationSha,
      activePhase,
      notBeforeMs: now,
    }).join("\n")).toMatch(/fields\/run\/nonce\/session do not match/);

    const mismatched = structuredClone(validOpened);
    mismatched.value.nonce = "wrong";
    mismatched.value.released_at = new Date(now - 1000).toISOString();
    expect(validateOwnedUiReleaseSentinel(mismatched, {
      runId: "run",
      nonce: "nonce",
      sessionId: "session",
      orchestrated: true,
      runIntentSha256: runIntentSha,
      uiSessionAttestationSha256: sessionAttestationSha,
      activePhase,
      notBeforeMs: now,
    }).join("\n")).toMatch(/does not match|stale/);

    const missingRunIntent = structuredClone(validOpened);
    delete missingRunIntent.value.run_intent_sha256;
    expect(validateOwnedUiReleaseSentinel(missingRunIntent, {
      runId: "run",
      nonce: "nonce",
      sessionId: "session",
      orchestrated: true,
      runIntentSha256: runIntentSha,
      uiSessionAttestationSha256: sessionAttestationSha,
      activePhase,
      notBeforeMs: now,
    }).join("\n")).toMatch(/run_intent_sha256 is missing or unbound/);

    const directory = mkdtempSync(path.join(tmpdir(), "vmlx-owned-release-"));
    try {
      const runIntentArtifact = path.join(directory, "run-intent.json");
      const sessionAttestationArtifact = path.join(directory, "ui-session.json");
      const apiArtifact = path.join(directory, "api.json");
      const cacheArtifact = path.join(directory, "cache.json");
      writePrivateArtifactFile(runIntentArtifact, '{"intent":"exact"}');
      writePrivateArtifactFile(
        sessionAttestationArtifact,
        '{"ui_session":"exact"}',
      );
      writePrivateArtifactFile(apiArtifact, '{"api":"exact"}');
      writePrivateArtifactFile(cacheArtifact, '{"cache":"exact"}');
      const exactRunIntentSha = crypto
        .createHash("sha256")
        .update(readFileSync(runIntentArtifact))
        .digest("hex");
      const exactSessionAttestationSha = crypto
        .createHash("sha256")
        .update(readFileSync(sessionAttestationArtifact))
        .digest("hex");
      const apiSha = crypto
        .createHash("sha256")
        .update(readFileSync(apiArtifact))
        .digest("hex");
      const cacheSha = crypto
        .createHash("sha256")
        .update(readFileSync(cacheArtifact))
        .digest("hex");
      const boundValue = {
        ...structuredClone(validOpened.value),
        run_intent_sha256: exactRunIntentSha,
        ui_session_attestation_sha256: exactSessionAttestationSha,
        api_capture_sha256: apiSha,
        cache_capture_sha256: cacheSha,
      };
      const artifactBindings = {
        runIntentPath: runIntentArtifact,
        runIntentSha256: exactRunIntentSha,
        uiSessionAttestationPath: sessionAttestationArtifact,
        uiSessionAttestationSha256: exactSessionAttestationSha,
        activePhase,
      };
      const preexisting = path.join(directory, "preexisting.json");
      writePrivateArtifactFile(preexisting, JSON.stringify(boundValue));
      await expect(waitForOwnedUiReleaseSentinel({
        filePath: preexisting,
        runId: "run",
        nonce: "nonce",
        sessionId: "session",
        ...artifactBindings,
        apiArtifactPath: apiArtifact,
        cacheArtifactPath: cacheArtifact,
        notBeforeMs: now,
        timeoutMs: 50,
        pollMs: 1,
      })).rejects.toThrow(/existed before/);

      const target = path.join(directory, "target.json");
      const link = path.join(directory, "release-link.json");
      writePrivateArtifactFile(target, JSON.stringify(validOpened.value));
      const waiter = waitForOwnedUiReleaseSentinel({
        filePath: link,
        runId: "run",
        nonce: "nonce",
        sessionId: "session",
        ...artifactBindings,
        apiArtifactPath: apiArtifact,
        cacheArtifactPath: cacheArtifact,
        notBeforeMs: now,
        timeoutMs: 250,
        pollMs: 2,
      });
      setTimeout(() => symlinkSync(target, link), 10);
      await expect(waiter).rejects.toThrow(/regular, non-symlink/);

      const released = path.join(directory, "released.json");
      const releasedWaiter = waitForOwnedUiReleaseSentinel({
        filePath: released,
        runId: "run",
        nonce: "nonce",
        sessionId: "session",
        ...artifactBindings,
        apiArtifactPath: apiArtifact,
        cacheArtifactPath: cacheArtifact,
        notBeforeMs: now,
        timeoutMs: 250,
        pollMs: 2,
      });
      setTimeout(
        () => writePrivateArtifactFile(released, JSON.stringify(boundValue)),
        10,
      );
      await expect(releasedWaiter).resolves.toMatchObject({
        run_intent_sha256: exactRunIntentSha,
        ui_session_attestation_sha256: exactSessionAttestationSha,
        api_capture_sha256: apiSha,
        cache_capture_sha256: cacheSha,
        run_intent_path: realpathSync(runIntentArtifact),
        ui_session_attestation_path:
          realpathSync(sessionAttestationArtifact),
        api_capture_path: realpathSync(apiArtifact),
        cache_capture_path: realpathSync(cacheArtifact),
      });

      const unbound = path.join(directory, "unbound.json");
      const unboundWaiter = waitForOwnedUiReleaseSentinel({
        filePath: unbound,
        runId: "run",
        nonce: "nonce",
        sessionId: "session",
        ...artifactBindings,
        apiArtifactPath: apiArtifact,
        cacheArtifactPath: cacheArtifact,
        notBeforeMs: now,
        timeoutMs: 250,
        pollMs: 2,
      });
      setTimeout(
        () => writePrivateArtifactFile(
          unbound,
          JSON.stringify({
            ...boundValue,
            cache_capture_sha256: otherSha,
          }),
        ),
        10,
      );
      await expect(unboundWaiter).rejects.toThrow(
        /hashes do not match safely reopened exact artifacts/,
      );
    } finally {
      rmSync(directory, { recursive: true, force: true });
    }
  });

  it("parses route, model, and only named resolved sampling kwargs", () => {
    expect(
      parseResolvedSamplingKwargs([
        "INFO Resolved sampling kwargs route=/v1/chat/completions model=/models/laguna kwargs={'temperature': 0.7, 'top_p': 0.9, 'top_k': 40, 'enable_thinking': True}",
      ]),
    ).toEqual([
      {
        route_model:
          "Resolved sampling kwargs route=/v1/chat/completions model=/models/laguna",
        route: "/v1/chat/completions",
        model: "/models/laguna",
        raw: "{'temperature': 0.7, 'top_p': 0.9, 'top_k': 40, 'enable_thinking': True}",
        values: {
          temperature: 0.7,
          top_p: 0.9,
          top_k: 40,
          enable_thinking: true,
        },
      },
    ]);
  });

  it("parses the complete server-emitted proof, request, and message identity tuple", () => {
    expect(
      parseResolvedSamplingKwargs([
        "INFO Resolved sampling kwargs route=/v1/responses model=/models/gemma proof_request_id=user-1 request_id=wire-1 message_id=assistant-1 kwargs={'temperature': 0.8}",
      ])[0],
    ).toMatchObject({
      route: "/v1/responses",
      model: "/models/gemma",
      proof_request_id: "user-1",
      request_id: "wire-1",
      message_id: "assistant-1",
      correlation_source: "server_emitted",
      values: { temperature: 0.8 },
    });
  });

  it("accepts multiple unique tool-continuation request IDs under one UI turn and rejects reuse", () => {
    const result = structuredClone(goodResult());
    const followupId = "wire-request-1-followup";
    result.uiTurnEvidence[0].requestIds.push(followupId);
    result.requestCorrelation.turns[0].serverRequestIds.push(followupId);
    result.resolvedSamplingRecords.splice(1, 0, {
      ...structuredClone(result.resolvedSamplingRecords[0]),
      request_id: followupId,
    });
    expect(isServerRequestCorrelationVerified(result)).toBe(true);

    result.uiTurnEvidence[1].requestIds = [followupId];
    result.requestCorrelation.turns[1].serverRequestIds = [followupId];
    result.resolvedSamplingRecords.find(
      (record: Record<string, unknown>) => record.proof_request_id === "user-2",
    ).request_id = followupId;
    expect(isServerRequestCorrelationVerified(result)).toBe(false);
  });

  it("keeps a fixed full-tuple hash suffix when long sanitized prefixes collide", () => {
    const common = "x".repeat(400);
    const first = uniqueProofBasename({
      requested: common,
      model: `${common}-model-a`,
      run: `${common}-run-a`,
    });
    const second = uniqueProofBasename({
      requested: common,
      model: `${common}-model-b`,
      run: `${common}-run-b`,
    });
    expect(first).not.toBe(second);
    expect(first).toMatch(/-[0-9a-f]{20}$/);
    expect(second).toMatch(/-[0-9a-f]{20}$/);
    expect(first.length).toBeLessThanOrEqual(240);
    expect(second.length).toBeLessThanOrEqual(240);
  });

  it("rejects metadata-only or self-referential API evidence instead of treating JSON existence as dual proof", () => {
    const result = goodResult();
    result.surfaceStatus = "dual_surface_attested";
    const fixture = createValidPairedArtifact(result);
    try {
      const metadataOnly = structuredClone(fixture.artifact);
      metadataOnly.value.protocols = {};
      result.pairedApiArtifact = metadataOnly;
      expect(validatePairedApiEvidence(result).join("\n")).toMatch(
        /metadata does not match|exact Chat\/Responses\/Anthropic\/Ollama/,
      );

      const selfPath = path.join(fixture.directory, "self-referential.json");
      const selfValue = structuredClone(fixture.artifact.value);
      selfValue.raw_capture.manifest_path = selfPath;
      selfValue.raw_capture.manifest_sha256 = "0".repeat(64);
      writePrivateArtifactFile(selfPath, JSON.stringify(selfValue));
      result.pairedApiArtifact = readPrivateExternalJson(
        selfPath,
        "Self-referential API fixture",
      );
      expect(validatePairedApiEvidence(result).join("\n")).toMatch(
        /raw capture manifest|manifest bytes\/path\/summary/,
      );
    } finally {
      rmSync(fixture.directory, { recursive: true, force: true });
    }
  });

  it("keeps time-window logs and global cache deltas PARTIAL without server-emitted request IDs", () => {
    const result = goodResult();
    result.requestCorrelation = {
      status: "partial_product_support_missing",
      reason: "server does not emit proof request IDs",
      turns: result.uiTurnEvidence.map((turn: Record<string, any>) => ({
        turn: turn.turn,
        proofRequestId: turn.proofRequestId,
        userMessageId: turn.userMessageId,
        assistantMessageId: turn.assistantMessageId,
        serverProofRequestId: null,
        serverRequestId: null,
        serverMessageId: null,
      })),
    };
    result.resolvedSamplingRecords = result.resolvedSamplingRecords.map(
      (record: Record<string, any>, index: number) => {
        const copy = { ...record, observed_window_turn: index + 1 };
        delete copy.proof_request_id;
        delete copy.request_id;
        delete copy.message_id;
        delete copy.correlation_source;
        return copy;
      },
    );
    result.cacheRequestEvidence = result.cacheRequestEvidence.map(
      (row: Record<string, any>) => ({
        ...row,
        correlationStatus: "partial_product_support_missing",
        serverRequestId: null,
        serverObservation: null,
      }),
    );

    expect(isServerRequestCorrelationVerified(result)).toBe(false);
    expect(validateGenerationDefaultsEvidence(result).join("\n")).toMatch(
      /lack exact proof\/request\/message correlation/,
    );
    expect(validateRequestCorrelatedCacheEvidence(result).join("\n")).toMatch(
      /do not exactly match|lacks exact server-emitted request correlation/,
    );
    const surfaces = deriveProvenSurfaces(result);
    expect(surfaces).not.toContain("generation_defaults_visible_ui");
    expect(surfaces).not.toContain("generation_defaults_applied");
    expect(surfaces).not.toContain("cache_hit_telemetry");
  });
});

describe("native MTP surface / engine parity", () => {
  // Both shapes below are verbatim from live Electron CDP runs on 2026-08-17,
  // which is why this rule could be pinned at all: before observing the
  // non-MTP arm, "the control renders for MTP bundles" was an inference.
  const mtpBundleResult = () => ({
    requestedServerCacheControls: true,
    serverCacheControls: {
      nativeMtpControl: {
        labelVisible: true,
        labelText:
          "Native MTP ? Adaptive MTP AR (MTP off)",
        modeSelectPresent: true,
        selectedMode: "adaptive",
        modeOptions: ["adaptive", "off"],
        blockedFallbackNoticeShown: false,
        mentionedInDrawer: true,
      },
    },
    // Qwen3.6-35B-A3B-MXFP8-CRACK-MTP
    server: {
      health: {
        mtp: {
          index_has_mtp_tensors: true,
          mtp_tensor_count: 31,
          runtime_active: true,
          effective_depth: 2,
          status: "native_runtime_active",
        },
      },
    },
  });

  // Nemotron-Omni-Nano-JANGTQ-CRACK
  const nonMtpBundleResult = () => ({
    requestedServerCacheControls: true,
    serverCacheControls: {
      nativeMtpControl: {
        labelVisible: false,
        labelText: "",
        modeSelectPresent: false,
        selectedMode: null,
        modeOptions: null,
        blockedFallbackNoticeShown: false,
        mentionedInDrawer: false,
      },
    },
    server: {
      health: {
        mtp: {
          index_has_mtp_tensors: false,
          mtp_tensor_count: 0,
          runtime_active: false,
          status: "not_configured",
        },
      },
    },
  });

  it("accepts an MTP bundle that renders the control", () => {
    expect(validateNativeMtpSurfaceParity(mtpBundleResult())).toEqual([]);
  });

  it("accepts hidden controls for witnessed weights-present runtime-unwired capability", () => {
    const result: any = nonMtpBundleResult();
    Object.assign(result.server.health.mtp, {
      index_has_mtp_tensors: true, mtp_tensor_count: 72,
      runtime_supported: false, runtime_available: false,
      status: "weights_present_runtime_unwired",
    });
    expect(validateNativeMtpSurfaceParity(result)).toEqual([]);
    result.serverCacheControls.nativeMtpControl.labelVisible = true;
    expect(validateNativeMtpSurfaceParity(result).join("\n")).toMatch(/runtime unwired/);
  });

  it("accepts hidden MTP controls only with a complete inconsistent-artifact verdict", () => {
    const result: any = nonMtpBundleResult();
    Object.assign(result.server.health.mtp, {
      index_has_mtp_tensors: true, mtp_tensor_count: 42,
      artifact_available: false, runtime_supported: false,
      runtime_available: false, runtime_active: false,
      status: "metadata_inconsistent", runtime_reason: "metadata_inconsistent",
      issues: ["config declares 1 layer but index has 2"],
    });
    expect(validateNativeMtpSurfaceParity(result)).toEqual([]);
    for (const field of ["artifact_available", "runtime_supported", "runtime_available", "runtime_active"]) {
      for (const value of [true, undefined]) {
        const contradictory = structuredClone(result);
        contradictory.server.health.mtp[field] = value;
        expect(validateNativeMtpSurfaceParity(contradictory).length).toBeGreaterThan(0);
      }
    }
    for (const issues of [[], undefined]) {
      const missing = structuredClone(result);
      missing.server.health.mtp.issues = issues;
      expect(validateNativeMtpSurfaceParity(missing).length).toBeGreaterThan(0);
    }
    result.serverCacheControls.nativeMtpControl.modeSelectPresent = true;
    expect(validateNativeMtpSurfaceParity(result).join("\n")).toMatch(/inconsistent/);
  });

  it.each(["runtime_supported", "runtime_available", "runtime_active"])(
    "does not hide supported controls when the unwired %s verdict is contradictory or absent",
    (field) => {
      for (const value of [true, undefined]) {
        const result: any = nonMtpBundleResult();
        Object.assign(result.server.health.mtp, {
          index_has_mtp_tensors: true, runtime_supported: false,
          runtime_available: false, runtime_active: false,
          status: "weights_present_runtime_unwired", [field]: value,
        });
        expect(validateNativeMtpSurfaceParity(result).join("\n")).toMatch(/rendered no Native MTP control/);
      }
    },
  );

  it("accepts a visibly selected and persisted adaptive runtime", () => {
    const result: any = mtpBundleResult();
    result.requestContract = { nativeMtpMode: "adaptive" };
    result.nativeMtpSelection = {
      requested: true,
      requestedMode: "adaptive",
      selectedMode: "adaptive",
      persistedMode: "adaptive",
    };
    result.serverCacheControls.nativeMtpControl.selectedMode = "adaptive";
    result.effectiveSessionConfig = { nativeMtpMode: "adaptive" };
    expect(validateNativeMtpSurfaceParity(result)).toEqual([]);
  });

  it("accepts a visibly selected and persisted AR (off) runtime", () => {
    const result: any = mtpBundleResult();
    result.requestContract = { nativeMtpMode: "off" };
    result.nativeMtpSelection = {
      requested: true,
      requestedMode: "off",
      selectedMode: "off",
      persistedMode: "off",
    };
    result.serverCacheControls.nativeMtpControl.selectedMode = "off";
    result.effectiveSessionConfig = { nativeMtpMode: "off" };
    expect(validateNativeMtpSurfaceParity(result)).toEqual([]);
  });

  it("rejects a requested mode that the UI or the persisted session silently changed", () => {
    const result: any = mtpBundleResult();
    result.requestContract = { nativeMtpMode: "off" };
    result.nativeMtpSelection = {
      requested: true,
      selectedMode: "adaptive",
      persistedMode: "adaptive",
    };
    result.serverCacheControls.nativeMtpControl.selectedMode = "adaptive";
    result.effectiveSessionConfig = { nativeMtpMode: "adaptive" };
    expect(validateNativeMtpSurfaceParity(result).join("\n")).toMatch(
      /did not match requested off/,
    );
  });

  it("accepts a non-MTP bundle that renders no control", () => {
    expect(validateNativeMtpSurfaceParity(nonMtpBundleResult())).toEqual([]);
  });

  it("rejects an MTP bundle whose control never rendered", () => {
    const result = mtpBundleResult();
    result.serverCacheControls.nativeMtpControl.labelVisible = false;
    result.serverCacheControls.nativeMtpControl.modeSelectPresent = false;
    expect(validateNativeMtpSurfaceParity(result).join("\n")).toMatch(
      /engine reports MTP weights in the bundle but the UI rendered no Native MTP control/,
    );
  });

  it("rejects a DEAD control on a bundle with no MTP weights", () => {
    // The failure mode that made four families render a dead Thinking toggle.
    const result = nonMtpBundleResult();
    result.serverCacheControls.nativeMtpControl.labelVisible = true;
    result.serverCacheControls.nativeMtpControl.modeSelectPresent = true;
    expect(validateNativeMtpSurfaceParity(result).join("\n")).toMatch(
      /engine reports no MTP weights in the bundle but the UI rendered a Native MTP control/,
    );
  });

  it("rejects a mode selector missing one of the two modes or still offering a retired one", () => {
    const result = mtpBundleResult();
    result.serverCacheControls.nativeMtpControl.modeOptions = ["adaptive"];
    expect(validateNativeMtpSurfaceParity(result).join("\n")).toMatch(
      /missing the off option/,
    );
    result.serverCacheControls.nativeMtpControl.modeOptions = ["adaptive", "deterministic", "off"];
    expect(validateNativeMtpSurfaceParity(result).join("\n")).toMatch(
      /retired deterministic option/,
    );
  });

  it("rejects a blocked-fallback notice while the engine says the runtime is ACTIVE", () => {
    const result = mtpBundleResult();
    result.serverCacheControls.nativeMtpControl.blockedFallbackNoticeShown = true;
    expect(validateNativeMtpSurfaceParity(result).join("\n")).toMatch(
      /runtime ACTIVE but the UI showed the blocked-fallback notice/,
    );
  });

  it("fails loudly when the surface or the engine report is missing", () => {
    // A silently-skipped check is indistinguishable from a passing one.
    expect(
      validateNativeMtpSurfaceParity({
        requestedServerCacheControls: true,
        serverCacheControls: {},
        server: { health: { mtp: { index_has_mtp_tensors: false } } },
      }).join("\n"),
    ).toMatch(/Native MTP surface was not captured/);
    const noHealth = mtpBundleResult();
    (noHealth.server as any).health = {};
    expect(validateNativeMtpSurfaceParity(noHealth).join("\n")).toMatch(
      /omitted the mtp block/,
    );
    const notBoolean = mtpBundleResult();
    (notBoolean.server.health.mtp as any).index_has_mtp_tensors = "yes";
    expect(validateNativeMtpSurfaceParity(notBoolean).join("\n")).toMatch(
      /is not a boolean/,
    );
  });

  it("stays inert when the run never opened the cache drawer", () => {
    expect(
      validateNativeMtpSurfaceParity({ requestedServerCacheControls: false }),
    ).toEqual([]);
  });
});

describe("a run made entirely of never-empty notices is not a clean run", () => {
  it("fails when every visible answer is the notice", () => {
    // dots3-note tools-off did this across three budgets: reasoning of 11.4k
    // then 23.9k characters with every answer exactly the 98-character notice.
    // The notice itself is correct product behaviour - it replaces a blank
    // bubble - but a run made entirely of them proves nothing the model said,
    // and it read as healthy enough that I called it clean.
    const result = structuredClone(goodResult());
    result.assistantRecords = result.assistantMessageIds.map(() => ({
      content: REASONING_WITHOUT_ANSWER_NOTICE,
    }));
    (result.renderedDom.messages || []).forEach((message: any) => {
      message.answerText = REASONING_WITHOUT_ANSWER_NOTICE;
    });
    const failures = validateReasoningEvidence(result, "optional").join("\n");
    expect(failures).toMatch(/every visible answer .* was a never-empty notice/);
  });

  it("stays quiet when the model actually answered", () => {
    const result = structuredClone(goodResult());
    const failures = validateReasoningEvidence(result, "optional").join("\n");
    expect(failures).not.toMatch(/never-empty notice, so the model produced no answer/);
  });
});

describe("reasoning numeric runs: spew vs deliberate counting", () => {
  it("does not flag one counting run inside long coherent reasoning", () => {
    // Verbatim proportions from a step37 run: the model wrote
    // "1 2 3 ... 21" to check `wc -c` against REAL_UI_LIVE_TOOL_ONE,
    // a 55-character run inside 8,536 characters of English.
    expect(
      reasoningNumericRunIsSpew({
        reasoningNumericRunCount: 1,
        reasoningNumericRunChars: 55,
        reasoningTextLength: 8536,
      }),
    ).toBe(false);
  });

  it("flags runs that dominate the reasoning text", () => {
    expect(
      reasoningNumericRunIsSpew({
        reasoningNumericRunCount: 12,
        reasoningNumericRunChars: 900,
        reasoningTextLength: 1000,
      }),
    ).toBe(true);
  });

  it("keeps the count-only rule when an older artifact lacks the measurements", () => {
    // Silently passing something unmeasured would be worse than a false alarm.
    expect(reasoningNumericRunIsSpew({ reasoningNumericRunCount: 3 })).toBe(true);
  });

  it("passes the measurements through to the object the validator reads", () => {
    // The in-page capture and the chat object handed to the validator are two
    // different places. Recording chars/length in only the first left the
    // refinement inert and a step37 run kept failing with the numbers that
    // exonerate it sitting in the artifact.
    const source = readFileSync(
      new URL("../scripts/live-real-ui-model-proof.mjs", import.meta.url),
      "utf8",
    );
    expect(source).toContain(
      "reasoningNumericRunChars: rendererResult.reasoningNumericRunChars",
    );
    expect(source).toContain("reasoningTextLength: rendererResult.reasoningTextLength");
  });

  it("stays quiet when there is no numeric run at all", () => {
    expect(
      reasoningNumericRunIsSpew({
        reasoningNumericRunCount: 0,
        reasoningNumericRunChars: 0,
        reasoningTextLength: 4000,
      }),
    ).toBe(false);
  });
});

describe("reasoning rail: dangling TeX delimiter", () => {
  // Three consecutive LFM2.5 runs failed on nothing but the rail-linkage check.
  // The model's reasoning wrote an unpaired escaped paren; markdown rendered it
  // as a literal "(" while the persisted record kept the backslash, so the whole
  // diff was five backslashes. Comparing against a baseline where both sides
  // already agree isolates the delimiter handling from the rest of the fixture.
  const failuresFor = (persistedText: string, renderedText: string) => {
    const result = structuredClone(goodResult());
    const messageId = result.assistantMessageIds[0];
    result.persistedReasoningByMessage = [[persistedText], [], []];
    const domMessage = result.renderedDom.messages[0];
    domMessage.messageId = messageId;
    domMessage.reasoningSegments = [renderedText];
    domMessage.reasoningLinkedSegments = [renderedText];
    domMessage.reasoningText = renderedText;
    return validateReasoningEvidence(result, "optional");
  };

  it("treats an unpaired escaped paren as the paren on both sides", () => {
    const escaped = failuresFor(
      "Receipt line. Math: \\(2 + 2 = 4 and then done.",
      "Receipt line. Math: (2 + 2 = 4 and then done.",
    );
    const baseline = failuresFor(
      "Receipt line. Math: (2 + 2 = 4 and then done.",
      "Receipt line. Math: (2 + 2 = 4 and then done.",
    );
    // The delimiter difference must change NOTHING about the outcome. (This
    // fixture does not satisfy every other rail invariant, which is precisely
    // why the assertion is a comparison against the same fixture rather than an
    // empty-failure claim — that would be testing the fixture, not the change.)
    expect(escaped).toEqual(baseline);
  });

  it("still reports a rail segment whose text genuinely differs", () => {
    const failures = failuresFor(
      "Receipt line. Math: \\(2 + 2 = 4 and then done.",
      "Something else entirely.",
    ).join("\n");
    expect(failures).toMatch(
      /normalized visible reasoning rail segments are not linked/,
    );
  });
});

describe("tool loop: per-turn protocol resolution", () => {
  it("rejects the qwen36 batched case where turn 1 made NO call at all", () => {
    // Verbatim shape from a stored artifact: message 0 has ZERO calls, both
    // calls landed on message 1, and the successful one is a single command
    // doing BOTH steps. The per-turn chain this surface asserts never happened,
    // so it must fail — and it must say WHICH step was missing rather than
    // cascading ten positional failures.
    const result = structuredClone(goodResult());
    result.persistedOaiCallsByMessage[0] = [];
    result.persistedOaiResultsByMessage[0] = [];
    result.persistedToolsByMessage[0] = [];
    result.renderedDom.messages[0].toolCards = [];
    result.persistedOaiCallsByMessage[1] = [
      {
        id: "call_batched",
        function: {
          name: "run_command",
          arguments: JSON.stringify({
            command:
              "touch real_ui_tool_probe_1.txt && echo -n REAL_UI_LIVE_TOOL_ONE > real_ui_tool_probe_1.txt && printf %s REAL_UI_LIVE_TOOL_TWO > real_ui_tool_probe_2.txt && cat real_ui_tool_probe_2.txt",
          }),
        },
      },
    ];
    expect(validateExactToolLoopEvidence(result).join("\n")).toMatch(
      /tool call 1 was not persisted on assistant turn 1/,
    );
  });

  it("accepts LFM2.5 splitting the prescribed step-1 command into two calls", () => {
    // Verbatim from a live default-prompt run: the prompt asks for ONE call
    // running `printf ... > probe_1 && cat probe_1`; lfm25 issued the printf and
    // the cat as separate calls. Same work, chain intact, probe files exact.
    const result = structuredClone(goodResult());
    const extra = {
      id: "call_split_cat",
      function: {
        name: "run_command",
        arguments: JSON.stringify({ command: "cat real_ui_tool_probe_1.txt" }),
      },
    };
    result.persistedOaiCallsByMessage[0] = [
      ...result.persistedOaiCallsByMessage[0],
      extra,
    ];
    result.persistedOaiResultsByMessage[0] = [
      ...result.persistedOaiResultsByMessage[0],
      { tool_call_id: "call_split_cat", content: "REAL_UI_LIVE_TOOL_ONE" },
    ];
    result.persistedToolsByMessage[0] = [
      ...result.persistedToolsByMessage[0],
      { phase: "calling", toolName: "run_command", toolCallId: "call_split_cat" },
      { phase: "result", toolName: "run_command", toolCallId: "call_split_cat" },
    ];
    result.renderedDom.messages[0].toolCards = [
      ...result.renderedDom.messages[0].toolCards,
      { callId: "call_split_cat", name: "run_command", phase: "result", visible: true },
    ];
    const trace = result.messageEventTrace.find(
      (row: any) => row.messageId === result.assistantMessageIds[0],
    );
    trace.events = [
      ...trace.events,
      { event: "tool", payload: { phase: "calling", toolCallId: "call_split_cat" } },
      { event: "tool", payload: { phase: "result", toolCallId: "call_split_cat" } },
    ];
    expect(validateExactToolLoopEvidence(result)).toEqual([]);
  });

  it("still requires every extra call to be visible to the user", () => {
    const result = structuredClone(goodResult());
    result.persistedOaiCallsByMessage[0] = [
      ...result.persistedOaiCallsByMessage[0],
      {
        id: "call_hidden",
        function: {
          name: "run_command",
          arguments: JSON.stringify({ command: "cat real_ui_tool_probe_1.txt" }),
        },
      },
    ];
    expect(validateExactToolLoopEvidence(result).join("\n")).toMatch(
      /call_hidden has no matching rendered tool card|one rendered tool card per call/,
    );
  });

  it("surfaces a VISIBLE tool status whose call was never persisted", () => {
    // Observed live on a 30-call lfm25 churn run: calling status call_ee7e26d2
    // had no persisted call, meaning the user saw a tool call absent from the
    // saved conversation. Positional pairing buried this under a cascade of
    // "call ID/order does not match" noise.
    const result = structuredClone(goodResult());
    result.persistedToolsByMessage[0] = [
      ...result.persistedToolsByMessage[0],
      { phase: "calling", toolName: "run_command", toolCallId: "call_orphan" },
      { phase: "result", toolName: "run_command", toolCallId: "call_orphan" },
    ];
    expect(validateExactToolLoopEvidence(result).join("\n")).toMatch(
      /visible tool status call_orphan has no persisted tool call/,
    );
  });

  it("rejects reaching ahead to the second probe on ANY turn-one call", () => {
    // Strengthened: the ordering rule used to be checked only on the call that
    // was positionally first, so a model could reach ahead in a later turn-one
    // call unnoticed.
    const result = structuredClone(goodResult());
    result.persistedOaiCallsByMessage[0][0].function.arguments = JSON.stringify({
      command:
        "printf %s REAL_UI_LIVE_TOOL_ONE > real_ui_tool_probe_1.txt && cat real_ui_tool_probe_2.txt",
    });
    expect(validateExactToolLoopEvidence(result).join("\n")).toMatch(
      /referenced the second-turn probe prematurely/,
    );
  });
});


describe("dev Electron launch identity", () => {
  it("records the same explicit binary passed to electron-vite", () => {
    expect(resolveDevElectronExecutable("/panel", { ELECTRON_EXEC_PATH: "/private/Electron" })).toBe("/private/Electron");
    expect(resolveDevElectronExecutable("/panel", {})).toBe("/panel/node_modules/electron/dist/Electron.app/Contents/MacOS/Electron");
  });
});

describe("visible SSD-only cache policy", () => {
  it("requires visible policy evidence when the RAM checkbox is omitted", () => {
    expect(visiblePagedCachePolicy({present: false, checked: false, disabled: false, policyVisible: true})).toEqual({established: true, usePagedCache: false, lockedOff: true});
    expect(visiblePagedCachePolicy({present: false, checked: false, disabled: false, policyVisible: false}).established).toBe(false);
  });
  it("does not let policy prose override a conflicting real RAM control", () => {
    expect(visiblePagedCachePolicy({present: true, checked: true, disabled: false, policyVisible: true})).toEqual({established: true, usePagedCache: true, lockedOff: false});
  });
});

describe("server cache controls are driven through the two-choice Prefix Cache form", () => {
  const source = readFileSync(
    path.resolve("scripts/live-real-ui-model-proof.mjs"),
    "utf8",
  );

  it("reads and drives the SSD cache / RAM cache hooks, never the removed labels", () => {
    // Both the pre-Start lane selection and the running-session capture.
    expect(
      source.match(/'\[data-vmlx-control="' \+ name \+ '"\] input\[type="checkbox"\]'/g),
    ).toHaveLength(1);
    expect(source).toContain("'[data-vmlx-control=\"' + name + '\"]'");
    expect(source.match(/preCacheInput\('ssd-cache'\)/g)).toHaveLength(1);
    expect(source.match(/preCacheInput\('ram-cache'\)/g)).toHaveLength(1);
    expect(source).toContain("cacheInput('ssd-cache')");
    expect(source).toContain("cacheInput('ram-cache')");
    for (const removed of [
      "preInputFor('In-Memory Paged Cache (RAM)')",
      "preInputFor('Block Disk Cache (SSD / L2)')",
      "inputFor('Enable Prefix Cache')",
      "inputFor('Enable Disk Cache')",
      "inputFor('Block Disk Cache (SSD / L2)')",
      "inputFor('In-Memory Paged Cache (RAM)')",
      "clickSection('In-Memory Paged Cache')",
      "clickSection('Disk Cache')",
      "expandSection('In-Memory Paged Cache')",
      "'Stored Cache Quantization',",
    ]) {
      expect(source).not.toContain(removed);
    }
  });

  it("reports the visible tier and its saved-key view, SSD first for the SSD-only lane", () => {
    expect(source).toContain(
      "tier: !ssd ? null : ssdOn ? (ramOn ? 'ram-ssd' : 'ssd') : ramOn ? 'ram' : 'off',",
    );
    expect(source).toContain("enablePrefixCache: ssdOn || ramOn,");
    expect(source).toContain("usePagedCache: ssdOn && ramOn,");
    expect(source).toContain("enableDiskCache: ssdOn && exactTypedPromptDiskLane,");
    expect(source).toContain("enableBlockDiskCache: ssdOn && !exactTypedPromptDiskLane,");
    expect(source).toContain("policyVisible: !!ssdInput && isVisible(ssdControl),");
    expect(source).toContain("visibleCacheTier: initialCacheControls.tier,");
    const ssdClick = source.indexOf("if (ssd && !ssd.checked && !ssd.disabled) {");
    const ramClick = source.indexOf("if (ram && ram.checked && !ram.disabled) {");
    expect(ssdClick).toBeGreaterThan(-1);
    expect(ramClick).toBeGreaterThan(ssdClick);
    expect(source).toContain("'SSD Cache Size (% of disk)',");
    expect(source).toContain("'RAM Cache Size (% of RAM)',");
  });
});


describe("completed reasoning rail expansion", () => {
  it("reopens a rail collapsed by the pending completion timer before taking proof", async () => {
    let time = 0;
    let expanded = true;
    let collapsePending = true;
    let clicks = 0;
    const result = await expandCompletedReasoningRails({
      now: () => time,
      sleep: async (ms: number) => {
        time += ms;
        if (collapsePending && time >= 200) {
          expanded = false;
          collapsePending = false;
        }
      },
      readRows: () => [{ expectsReasoning: true, rails: [{
        complete: true, visible: true, expanded,
        expand: () => { clicks++; expanded = true; },
      }] }],
    });
    expect(clicks).toBe(1);
    expect(expanded).toBe(true);
    expect(result.elapsedMs).toBeGreaterThanOrEqual(1450);
  });

  it.each(["missing", "hidden", "streaming", "unresponsive"])("rejects %s reasoning instead of weakening the visible-content gate", async (kind) => {
    let time = 0;
    await expect(expandCompletedReasoningRails({
      timeoutMs: 100, stableMs: 50, pollMs: 10,
      now: () => time, sleep: async (ms: number) => { time += ms; },
      readRows: () => [{ expectsReasoning: true, rails: kind === "missing" ? [] : [{
        complete: kind !== "streaming", visible: kind !== "hidden",
        expanded: kind !== "unresponsive", expand: () => {},
      }] }],
    })).rejects.toThrow(/did not remain visibly expanded/);
  });

  it("accepts a no-reasoning turn without inventing a rail", async () => {
    let time = 0;
    const result = await expandCompletedReasoningRails({
      now: () => time, sleep: async (ms: number) => { time += ms; },
      readRows: () => [{ expectsReasoning: false, rails: [] }],
    });
    expect(result.clicks).toBe(0);
  });
});


describe("explicit sampler controls and verbatim receipts", () => {
  it("produces a real transition when explicitly selecting an already displayed default", () => {
    expect(explicitRangeInputSequence(0, 0, 0, 100)).toEqual([100, 0]);
    expect(explicitRangeInputSequence(1, 1, 0, 1)).toEqual([0, 1]);
    expect(explicitRangeInputSequence(0.9, 0, 0, 2)).toEqual([0]);
    expect(() => explicitRangeInputSequence(0, -1, 0, 100)).toThrow();
    expect(() => explicitRangeInputSequence(1, 1, 1, 1)).toThrow();
    expect(() => explicitRangeInputSequence(0, NaN, 0, 100)).toThrow();
  });
  it("rejects a corrupted copy-only receipt even when currency and math still render", () => {
    const result = goodResult();
    const receipt = "Third UI turn: REAL_UI_LIVE_TOOL_ONE REAL_UI_LIVE_TOOL_TWO\nCurrency: $43\nMath: \\(2 + 2 = 4\\)";
    result.requestContract.promptThree = "Return this exact three-line rendering receipt and nothing else.\n" + receipt;
    const record = result.assistantRecords.find(row => row.id === result.assistantMessageIds[2]);
    record.content = receipt.replace("TOOL_ONE", "TO<NL>ONE");
    expect(validateRenderedDomEvidence(result)).toContain("final assistant answer did not preserve the requested verbatim receipt");
    record.content = receipt;
    expect(validateRenderedDomEvidence(result)).not.toContain("final assistant answer did not preserve the requested verbatim receipt");
  });
});


describe("media semantics are bound to the submitted attachment turn", () => {
  function score(answer: string, earlier: string, kind = "video", includeAttachment = true) {
    const source = readFileSync(path.resolve("scripts/live-real-ui-model-proof.mjs"), "utf8");
    const start = source.indexOf("          const imageSemanticVerified =");
    const end = source.indexOf("          const mediaEvidence =", start);
    expect(start).toBeGreaterThan(0);
    const content = includeAttachment
      ? JSON.stringify([{ type: `${kind}_url`, [`${kind}_url`]: { url: "data:fixture" } }])
      : "no attachment";
    const messages = [
      { id: "old-user", role: "user", content: "old" },
      { id: "old-answer", role: "assistant", content: earlier },
      { id: "media-user", role: "user", content },
      { id: "media-answer", role: "assistant", content: answer },
    ];
    const binding = { userMessageId: "media-user" };
    return vm.runInNewContext(source.slice(start, end) + "\n({imageSemanticVerified, videoSemanticVerified, audioSemanticVerified})", {
      checkMedia: kind === "image", checkVideo: kind === "video", checkAudio: kind === "audio",
      imageExpectRegex: "^red$", videoExpectRegex: "^right$", audioExpectRegex: "green bicycle",
      allAssistantText: earlier + "\n" + answer, messages,
      mediaTurnBindings: { 4: binding, 5: binding, 6: binding },
      validateMediaTurnAnswer: (proofHarness as any).validateMediaTurnAnswer,
    });
  }
  it("does not accept a word from another turn as the audio transcript", () => {
    expect(score("I cannot hear it", "green bicycle", "audio").audioSemanticVerified).toBe(false);
  });
  it("anchors a direction against its own answer, not the whole conversation", () => {
    expect(score("right", "Earlier unrelated answer").videoSemanticVerified).toBe(true);
  });
  it("requires the attachment on the bound user message", () => {
    expect(score("green bicycle", "", "audio", false).audioSemanticVerified).toBe(false);
  });
});


describe("packaged media fixture decoding without data fetch", () => {
  it("preserves binary bytes without a network or renderer fetch", () => {
    expect(Array.from((proofHarness as any).decodeProofAttachment({
      dataUrl: "data:image/png;base64,AAEC/w==", type: "image/png",
    }))).toEqual([0, 1, 2, 255]);
  });
  it("rejects a MIME mismatch instead of injecting a mislabeled attachment", () => {
    expect(() => (proofHarness as any).decodeProofAttachment({
      dataUrl: "data:image/png;base64,AAEC/w==", type: "audio/wav",
    })).toThrow(/MIME/);
  });
  it("rejects non-fixture URLs", () => {
    expect(() => (proofHarness as any).decodeProofAttachment({
      dataUrl: "https://example.invalid/file", type: "image/png",
    })).toThrow(/data URL/);
  });
});
