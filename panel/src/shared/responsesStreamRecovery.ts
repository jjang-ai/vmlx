export interface ResponsesToolBufferReconciliation {
  clearSpeculativeBuffering: boolean;
  authoritativeText: string | null;
  rejectedControlMarkup?: boolean;
  /**
   * The rejected text itself. It must never be rendered as prose, but the
   * finalizer needs to know rejection happened with a non-empty payload so the
   * turn can carry an explicit notice instead of persisting visibly empty.
   */
  rejectedText?: string;
}

/**
 * Line-start tool-call markers that activate client-side speculative
 * buffering. Includes real native dialects and common hallucinated tags.
 *
 * Contract: line-start tool control markup is never visible assistant prose.
 * A Responses terminal can carry parser-rejected or incomplete tool markup in
 * output_text even though no concrete function_call item was emitted. That
 * control text must be rejected, not fenced and shown to the user. Ordinary
 * authoritative prose is still restored after a false-positive heartbeat.
 */
export const TOOL_CALL_MARKER_LINE_START =
  /(?:^|\n)\s*(?:<zyphra_tool_call\b|<function(?:=|\b)|<minimax:tool_call|<tool_call\b|\[Calling tool:|<invoke name=|<read_file\b|<write_file\b|<run_command\b|<search_files\b|<edit_file\b|<list_directory\b|<execute_command\b|<bash\b)/;

/**
 * Reconcile a Responses stream that advertised speculative tool generation but
 * completed without a concrete function_call item.
 *
 * The server's output_text.done value is authoritative after native tool
 * parsing. A heartbeat or partial XML prefix may have caused the Electron
 * client to suppress earlier text deltas, so the final text must replace the
 * current-iteration buffer instead of being skipped merely because a delta was
 * observed on the wire.
 */
export function reconcileResponsesToolBufferAtStreamEnd(args: {
  useResponsesApi: boolean;
  clientToolCallBuffering: boolean;
  receivedToolCallCount: number;
  finalText: string;
}): ResponsesToolBufferReconciliation {
  if (
    !args.useResponsesApi ||
    !args.clientToolCallBuffering ||
    args.receivedToolCallCount > 0
  ) {
    return {
      clearSpeculativeBuffering: false,
      authoritativeText: null,
    };
  }

  if (
    args.finalText.length > 0 &&
    TOOL_CALL_MARKER_LINE_START.test(args.finalText)
  ) {
    return {
      clearSpeculativeBuffering: true,
      authoritativeText: null,
      rejectedControlMarkup: true,
      rejectedText: args.finalText,
    };
  }

  return {
    clearSpeculativeBuffering: true,
    authoritativeText: args.finalText.length > 0 ? args.finalText : null,
  };
}

export type NeverEmptyAnswerReason =
  | "sanitized_to_empty"
  | "rejected_control_markup"
  | "tool_without_answer"
  | "reasoning_without_answer";

export interface NeverEmptyAnswerResolution {
  content: string;
  reason: NeverEmptyAnswerReason;
}

/**
 * Notices are deliberately prose, not the model's raw payload. Rejected control
 * markup can contain `<tool_call>`/`<function>`/`<invoke>`, which the release
 * proof counts as parser leakage — echoing it back to satisfy never-empty would
 * trade a blank turn for a leaking one.
 */
export const REJECTED_CONTROL_MARKUP_NOTICE =
  "_The model emitted tool-call markup that could not be parsed or executed, so it produced no answer for this turn._";

export const TOOL_WITHOUT_ANSWER_NOTICE =
  "_The tool call above completed, but the model produced no visible answer for this turn._";

export const REASONING_WITHOUT_ANSWER_NOTICE =
  "_Reasoning was received, but no visible answer was returned for this turn. See the response diagnostics for details._";

/**
 * Decide what an assistant turn shows when nothing renderable survived.
 *
 * A blank assistant bubble is the worst available outcome: the user cannot tell
 * whether the app broke or the model misbehaved. Four distinct ways a turn can
 * arrive here, all of which previously ended empty:
 *  - the leaked-markup sanitizer stripped an answer down to nothing
 *  - the Responses reconciler rejected an all-markup terminal payload
 *  - a tool ran and the model never followed up with an answer
 *  - the model emitted ONLY reasoning and never started an answer
 *
 * The reasoning-only case was found live on qwen36 (reasoning required + tools,
 * Responses door): 3 of 4 runs produced a first turn with zero content, tool
 * phases of only "generating"/"done" (so no tool call to blame) and 7-8
 * persisted reasoning segments. Every earlier case declined it — nothing to
 * fence, no rejected markup, no tool iterations — so the bubble stayed blank
 * while the reasoning rail filled up.
 *
 * One resolver so a fix in one path cannot be inert in the others.
 */
export function resolveNeverEmptyAssistantAnswer(args: {
  visibleAfterSanitize: string;
  preSanitizeContent: string;
  rejectedControlMarkupText: string;
  executedToolCallCount: number;
  priorIterationContent: string;
  toolIterations: number;
  reasoningContent?: string;
}): NeverEmptyAnswerResolution | null {
  if (args.visibleAfterSanitize.trim()) return null;
  if (args.priorIterationContent.trim()) return null;

  // Existing precedent: real prose over-stripped by the leak sanitizer is
  // preserved verbatim in a fence, because that text IS the model's answer.
  if (args.preSanitizeContent.trim() && args.executedToolCallCount === 0) {
    const preserved = args.preSanitizeContent
      .replace(/```[^\n`]*\n?/g, "")
      .trim();
    return {
      content: "```text\n" + (preserved || args.preSanitizeContent) + "\n```",
      reason: "sanitized_to_empty",
    };
  }

  if (args.rejectedControlMarkupText.trim()) {
    return {
      content: REJECTED_CONTROL_MARKUP_NOTICE,
      reason: "rejected_control_markup",
    };
  }

  if (args.toolIterations > 0 || args.executedToolCallCount > 0) {
    return {
      content: TOOL_WITHOUT_ANSWER_NOTICE,
      reason: "tool_without_answer",
    };
  }

  // Last: reasoning happened but no answer ever started. Checked after the tool
  // case because a tool card is the more useful thing to point at when both are
  // true.
  if ((args.reasoningContent || "").trim()) {
    return {
      content: REASONING_WITHOUT_ANSWER_NOTICE,
      reason: "reasoning_without_answer",
    };
  }

  return null;
}
