/**
 * ONE definition of "which families need longer than the generic timeout".
 *
 * This rule was previously written out SEVEN times — sessions.ts (the engine
 * --timeout arg), ipc/chat.ts (the timer that actually fails an in-app chat),
 * api-gateway.ts (external OpenAI-compat/Ollama clients), SessionSettings.tsx
 * (the CLI preview), CreateSession.tsx, the persisted-config writers, and the
 * engine's own dict — and four of them had diverged. The consequences were all
 * user-visible and each was found only by live testing:
 *
 *  - a 101,502-token prompt to Qwen3.6 rendered "Message failed - Request timed
 *    out after 300s" while the engine served it in ~230s, because chat.ts still
 *    said 300;
 *  - external gateway clients were aborted at 300s for openpangu_v2 and every
 *    hybrid SSM family long after the in-app path was fixed;
 *  - the Settings CLI preview displayed --timeout 300 for six of seven families
 *    whose launch actually passes 900.
 *
 * Two fixes before this one were SILENT NO-OPS because they were written
 * against the wrong namespace, so read this carefully:
 *
 * KEYS ARE PANEL REGISTRY NAMES, not engine family_name. The registry maps
 * engine qwen3_5 -> "qwen3.5", qwen3_next -> "qwen3-next", nemotron_h ->
 * "nemotron-h" (model-config-registry.ts MODEL_TYPE_TO_FAMILY), while
 * minimax_m3 and openpangu_v2 keep their underscores because those ARE their
 * registry names. The registry is not internally consistent about this, so
 * every key here was taken from it individually rather than transliterated.
 *
 * The engine keeps its own copy (separate process, cli.py _SLOW_FAMILY_TIMEOUTS
 * in engine family_name spelling); tests/test_cli_panel_settings_parity.py
 * asserts the two stay in step.
 */

/** The generic request timeout, in seconds. */
export const GENERIC_DEFAULT_TIMEOUT_SECONDS = 300

/** What a slow family gets instead, in seconds. */
export const SLOW_FAMILY_TIMEOUT_SECONDS = 900

/**
 * Registry-name -> timeout seconds.
 *
 * DSV4, MiniMax-M3 and openPangu are slow by weight/architecture. The hybrid
 * SSM+attention families (qwen3.5*, qwen3-next, nemotron-h) chunk their prefill,
 * so a long prompt spends minutes before the first token.
 */
export const SLOW_FAMILY_TIMEOUTS: Readonly<Record<string, number>> = Object.freeze({
  'deepseek-v4': SLOW_FAMILY_TIMEOUT_SECONDS,
  minimax_m3: SLOW_FAMILY_TIMEOUT_SECONDS,
  openpangu_v2: SLOW_FAMILY_TIMEOUT_SECONDS,
  ernie5_5: SLOW_FAMILY_TIMEOUT_SECONDS,
  'hy4-flash': SLOW_FAMILY_TIMEOUT_SECONDS,
  'qwen3.5': SLOW_FAMILY_TIMEOUT_SECONDS,
  'qwen3.5-moe': SLOW_FAMILY_TIMEOUT_SECONDS,
  'qwen3-next': SLOW_FAMILY_TIMEOUT_SECONDS,
  'nemotron-h': SLOW_FAMILY_TIMEOUT_SECONDS,
})

/**
 * Resolve the timeout for a session.
 *
 * A family default applies only while the session is still on the generic
 * value; anything the user explicitly chose is always honoured. A configured
 * value of 0 or less means "no limit" and is normalised by the caller.
 */
export function resolveSlowFamilyTimeoutSeconds(
  configuredSeconds: number | null | undefined,
  registryFamily: string | null | undefined,
): number {
  const familyDefault = registryFamily
    ? SLOW_FAMILY_TIMEOUTS[registryFamily]
    : undefined
  if (
    familyDefault != null &&
    (configuredSeconds == null ||
      configuredSeconds === GENERIC_DEFAULT_TIMEOUT_SECONDS)
  ) {
    return familyDefault
  }
  return configuredSeconds != null && configuredSeconds > 0
    ? configuredSeconds
    : GENERIC_DEFAULT_TIMEOUT_SECONDS
}
