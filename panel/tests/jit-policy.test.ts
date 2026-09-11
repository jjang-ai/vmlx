import { readFileSync } from "node:fs"
import { resolve } from "node:path"
import { describe, expect, it } from "vitest"
import {
  computeEffectiveJit,
  isJitSuppressedByRuntime,
  jitSuppressionReason,
  resolveRequestedJit,
} from "../src/shared/jitPolicy"

const ENABLED = {
  enableJitRequested: true,
  isMultimodal: false,
  flashMoeActive: false,
  distributedActive: false,
  dsv4Active: false,
  m3Active: false,
  zayaCcaActive: false,
  turboQuantActive: false,
  lagunaMixedSwaTurboQuantActive: false,
  hybridCacheActive: false,
}

/** Every consumer that must not re-inline the rule. */
const CONSUMERS = [
  "src/main/sessions.ts",
  "src/renderer/src/components/sessions/SessionSettings.tsx",
  "src/renderer/src/components/sessions/SessionConfigForm.tsx",
]

describe("JIT suppression policy", () => {
  it("defaults missing persisted JIT on without overriding an explicit off", () => {
    for (const saved of [undefined, true, false]) {
      const requested = resolveRequestedJit(saved)
      expect(requested).toBe(saved !== false)
      expect(computeEffectiveJit({ ...ENABLED, enableJitRequested: requested })).toBe(saved !== false)
      expect(computeEffectiveJit({ ...ENABLED, enableJitRequested: requested, isMultimodal: true })).toBe(false)
    }
  })

  it("preserves requested JIT through the drawer default merge", () => {
    for (const stored of [{}, { enableJit: true }, { enableJit: false }]) {
      const reopened = { enableJit: true, ...stored }
      expect(resolveRequestedJit(stored.enableJit)).toBe(resolveRequestedJit(reopened.enableJit))
    }
    for (const rel of CONSUMERS) {
      const src = readFileSync(resolve(__dirname, "..", rel), "utf8")
      expect(src).not.toContain("enableJitRequested: !!config.enableJit")
      expect(src).toContain("enableJitRequested: resolveRequestedJit(config.enableJit)")
    }
  })
  it("enables JIT only when requested and no runtime owns its own kernels", () => {
    expect(computeEffectiveJit(ENABLED)).toBe(true)
    expect(jitSuppressionReason(ENABLED)).toBeNull()
  })

  it("reports a reason for every suppressing condition", () => {
    const cases: Array<[keyof typeof ENABLED, string]> = [
      ["isMultimodal", "multimodal"],
      ["flashMoeActive", "flash_moe"],
      ["distributedActive", "distributed"],
      ["dsv4Active", "dsv4"],
      ["m3Active", "m3"],
      ["zayaCcaActive", "zaya_cca"],
      ["turboQuantActive", "turboquant"],
      ["lagunaMixedSwaTurboQuantActive", "laguna_mixed_swa_turboquant"],
      ["hybridCacheActive", "hybrid_cache"],
    ]
    for (const [field, reason] of cases) {
      const input = { ...ENABLED, [field]: true }
      expect(jitSuppressionReason(input), field).toBe(reason)
      expect(computeEffectiveJit(input), field).toBe(false)
    }
  })

  it("treats an explicit off as disabled regardless of runtime", () => {
    expect(
      jitSuppressionReason({ ...ENABLED, enableJitRequested: false }),
    ).toBe("disabled")
  })

  it("keeps all consumers on the shared policy instead of re-inlining it", () => {
    // The rule previously lived as three hand-copied boolean expressions that
    // spelled the multimodal condition differently (isVLM vs multimodalActive),
    // so the checkbox and the launched flag could disagree. If this fails,
    // someone re-inlined it -- route the call site through computeEffectiveJit.
    for (const rel of CONSUMERS) {
      const src = readFileSync(resolve(__dirname, "..", rel), "utf8")
      expect(src, `${rel} must import the shared policy`).toContain(
        "shared/jitPolicy",
      )
      expect(src, `${rel} must call computeEffectiveJit`).toContain(
        "computeEffectiveJit(",
      )
      expect(
        /!!config\.enableJit\s*&&\s*!/.test(src),
        `${rel} re-inlined the JIT suppression chain`,
      ).toBe(false)
    }
  })

  it("renders the same explicit JIT polarity that the main process launches", () => {
    // The engine can auto-enable affine JANG JIT when neither flag is present.
    // Therefore a suppressed/unchecked setting must be visible as --no-jit;
    // merely omitting --enable-jit makes the preview disagree with the process.
    for (const rel of CONSUMERS.slice(0, 2)) {
      const src = readFileSync(resolve(__dirname, "..", rel), "utf8")
      const start = src.indexOf("if (effectiveEnableJit)")
      expect(start, `${rel} must emit an explicit JIT choice`).toBeGreaterThanOrEqual(0)
      const polarity = src.slice(start, start + 180)
      expect(polarity, `${rel} must show/launch JIT on`).toContain("--enable-jit")
      expect(polarity, `${rel} must show/launch JIT off`).toContain("--no-jit")
    }
  })

  it("derives runtime suppression independently of the saved toggle", () => {
    const runtime = {
      isMultimodal: false,
      flashMoeActive: false,
      distributedActive: false,
      dsv4Active: false,
      m3Active: false,
      zayaCcaActive: false,
      turboQuantActive: false,
      lagunaMixedSwaTurboQuantActive: false,
      hybridCacheActive: false,
    }
    expect(isJitSuppressedByRuntime(runtime)).toBe(false)
    expect(isJitSuppressedByRuntime({ ...runtime, dsv4Active: true })).toBe(true)
    expect(isJitSuppressedByRuntime({ ...runtime, isMultimodal: true })).toBe(true)
  })

  it("keeps the checkbox disabled state on the same condition list", () => {
    // `checked` and `disabled` were separately inlined; if they drift the box
    // can render enabled while the launcher suppresses JIT.
    const form = readFileSync(resolve(__dirname, "..", CONSUMERS[2]), "utf8")
    expect(
      /flashMoeActive \|\| distributedActive \|\| dsv4Active/.test(form),
      "SessionConfigForm re-inlined the runtime suppression chain",
    ).toBe(false)
    expect(form).toContain("isJitSuppressedByRuntime")
  })
})
