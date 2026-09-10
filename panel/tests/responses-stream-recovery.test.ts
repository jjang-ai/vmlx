import { readFileSync } from "node:fs";
import { describe, expect, it } from "vitest";
import {
  reconcileResponsesToolBufferAtStreamEnd,
  REASONING_WITHOUT_ANSWER_NOTICE,
  REJECTED_CONTROL_MARKUP_NOTICE,
  resolveNeverEmptyAssistantAnswer,
  TOOL_CALL_MARKER_LINE_START,
  TOOL_WITHOUT_ANSWER_NOTICE,
} from "../src/shared/responsesStreamRecovery";

describe("Responses speculative tool-buffer reconciliation", () => {
  it("does not guess a token-budget cause from reasoning without an answer", () => {
    expect(REASONING_WITHOUT_ANSWER_NOTICE).toContain("response diagnostics");
    expect(REASONING_WITHOUT_ANSWER_NOTICE).not.toMatch(/budget|truncat|limit/i);
  });
  it("restores authoritative final text when a heartbeat produced no function call", () => {
    expect(
      reconcileResponsesToolBufferAtStreamEnd({
        useResponsesApi: true,
        clientToolCallBuffering: true,
        receivedToolCallCount: 0,
        finalText: "B1-REASON-SOAK14-PASS",
      }),
    ).toEqual({
      clearSpeculativeBuffering: true,
      authoritativeText: "B1-REASON-SOAK14-PASS",
    });
  });

  it("clears a zero-tool speculative buffer even when the server final is empty", () => {
    expect(
      reconcileResponsesToolBufferAtStreamEnd({
        useResponsesApi: true,
        clientToolCallBuffering: true,
        receivedToolCallCount: 0,
        finalText: "",
      }),
    ).toEqual({ clearSpeculativeBuffering: true, authoritativeText: null });
  });

  it("does not expose buffered markup when a concrete function call arrived", () => {
    expect(
      reconcileResponsesToolBufferAtStreamEnd({
        useResponsesApi: true,
        clientToolCallBuffering: true,
        receivedToolCallCount: 1,
        finalText: "pre-tool prose",
      }),
    ).toEqual({ clearSpeculativeBuffering: false, authoritativeText: null });
  });

  it("rejects parser-missed tool control markup instead of restoring it as prose", () => {
    expect(
      reconcileResponsesToolBufferAtStreamEnd({
        useResponsesApi: true,
        clientToolCallBuffering: true,
        receivedToolCallCount: 0,
        finalText:
          "<tool_calls>\n<tool_call>file_info<arg_key>\n" +
          "<arg_key>path</arg_key>\n<arg_value>panel/package.json",
      }),
    ).toEqual({
      clearSpeculativeBuffering: true,
      authoritativeText: null,
      rejectedControlMarkup: true,
      rejectedText:
        "<tool_calls>\n<tool_call>file_info<arg_key>\n" +
        "<arg_key>path</arg_key>\n<arg_value>panel/package.json",
    });
  });

  it("does nothing for non-Responses or non-buffered streams", () => {
    for (const [useResponsesApi, clientToolCallBuffering] of [
      [false, true],
      [true, false],
    ] as const) {
      expect(
        reconcileResponsesToolBufferAtStreamEnd({
          useResponsesApi,
          clientToolCallBuffering,
          receivedToolCallCount: 0,
          finalText: "visible",
        }),
      ).toEqual({ clearSpeculativeBuffering: false, authoritativeText: null });
    }
  });

  it("is wired after both initial and follow-up SSE streams", () => {
    const source = readFileSync(
      new URL("../src/main/ipc/chat.ts", import.meta.url),
      "utf8",
    );

    expect(source).toContain("responsesFinalText = parsed.text");
    expect(source).toContain("const reconcileResponsesToolBuffer = () =>");
    expect(source.match(/reconcileResponsesToolBuffer\(\);/g)).toHaveLength(2);
    expect(source).toContain("_sawResponsesTextDelta = false;");
    expect(source).toContain('responsesFinalText = "";');
  });

  it("marker pattern catches hallucinated line-start tool dialects that get buffered", () => {
    // The exact Q36MTP-TOOL-CALL-NOT-EMITTED shape: a textual <run_command>
    // dialect the server's native parser did not convert to a function_call.
    expect(
      TOOL_CALL_MARKER_LINE_START.test('<run_command>\n{"command": "pwd"}'),
    ).toBe(true);
    expect(
      TOOL_CALL_MARKER_LINE_START.test("prose then\n  <tool_call>{}"),
    ).toBe(true);
    // Prose mentions mid-line must NOT activate buffering.
    expect(
      TOOL_CALL_MARKER_LINE_START.test("I'll use the <run_command> tool now"),
    ).toBe(false);
  });

  it("ordinary restore bypasses marker re-detection after control markup was rejected", () => {
    const source = readFileSync(
      new URL("../src/main/ipc/chat.ts", import.meta.url),
      "utf8",
    );

    // emitDelta must accept the bypass flag and gate detection on it: the
    // restored authoritative text often still contains the marker that
    // activated buffering; re-scanning it would re-suppress it forever.
    expect(source).toContain("bypassToolMarkerDetection = false,");
    expect(source).toContain(
      "if (!isReasoningDelta && !bypassToolMarkerDetection) {",
    );
    // The zero-tool prose restore call must pass bypass=true.
    expect(source).toContain(
      "emitDelta(reconciliation.authoritativeText, false, false, true);",
    );
    // Detection uses the shared exported pattern (kept in sync with tests).
    expect(source).toContain("TOOL_CALL_MARKER_LINE_START");
  });

  it("tool-dialect text is rejected and the sanitizer cannot silently empty prose", () => {
    const source = readFileSync(
      new URL("../src/main/ipc/chat.ts", import.meta.url),
      "utf8",
    );

    expect(source).toContain("if (reconciliation.rejectedControlMarkup)");
    expect(source).toContain(
      "The model emitted parser-rejected tool control markup.",
    );
    expect(source).not.toContain(
      '"```text\\n" + reconciliation.authoritativeText.trim() + "\\n```"',
    );
    // Final persistence guard: sanitizing a non-empty answer down to nothing
    // (with no executed tool) must preserve the original in a fence rather
    // than persisting a blank assistant turn.
    expect(source).toContain("const preSanitizeContent = fullContent.trim();");
    expect(source).toContain("const visibleAfterSanitize = fullContent");
    // The guard is one shared resolver, not an inline condition, so a fix to
    // one blank-turn path cannot be inert in the others.
    expect(source).toContain("resolveNeverEmptyAssistantAnswer({");
    expect(source).toContain(
      "executedToolCallCount: receivedToolCalls.filter(Boolean).length,",
    );
    expect(source).toContain("[CHAT] Never-empty guard (${neverEmptyAnswer.reason})");
    // The notice must be STREAMED, not only persisted: a notice that appears
    // solely in the final save leaves the live turn blank and makes the wire
    // trace disagree with the persisted record.
    expect(source).toContain(
      "emitDelta(neverEmptyAnswer.content, false, true, true);",
    );
    // Speculative "generating" statuses must NOT mask the guard.
    expect(source).not.toContain(
      "collectedToolStatuses.length === 0 &&\n          preSanitizeContent",
    );
  });

  it("retains rejected markup text so the finalizer can explain the blank turn", () => {
    const source = readFileSync(
      new URL("../src/main/ipc/chat.ts", import.meta.url),
      "utf8",
    );

    expect(source).toContain("let rejectedControlMarkupText = \"\";");
    expect(source).toContain(
      "rejectedControlMarkupText = reconciliation.rejectedText || \"\";",
    );
    expect(source).toContain("rejectedControlMarkupText,");
  });
});

describe("Never-empty assistant answer resolution", () => {
  const base = {
    visibleAfterSanitize: "",
    preSanitizeContent: "",
    rejectedControlMarkupText: "",
    executedToolCallCount: 0,
    priorIterationContent: "",
    toolIterations: 0,
  };

  it("leaves a turn alone when it has visible content", () => {
    expect(
      resolveNeverEmptyAssistantAnswer({
        ...base,
        visibleAfterSanitize: "the real answer",
        rejectedControlMarkupText: "<tool_call>x",
      }),
    ).toBeNull();
  });

  it("leaves a turn alone when an earlier tool iteration produced content", () => {
    expect(
      resolveNeverEmptyAssistantAnswer({
        ...base,
        priorIterationContent: "answer from iteration 1",
        toolIterations: 2,
      }),
    ).toBeNull();
  });

  it("preserves prose the leak sanitizer stripped to nothing", () => {
    const resolved = resolveNeverEmptyAssistantAnswer({
      ...base,
      preSanitizeContent: "<read_file path=\"a.txt\" />",
    });
    expect(resolved?.reason).toBe("sanitized_to_empty");
    expect(resolved?.content).toBe(
      "```text\n<read_file path=\"a.txt\" />\n```",
    );
  });

  it("explains an all-markup Responses payload instead of rendering blank", () => {
    // The exact zaya_text release-blocker shape: 1024 generated tokens, every
    // one of them rejected control markup, previously a blank bubble.
    const resolved = resolveNeverEmptyAssistantAnswer({
      ...base,
      rejectedControlMarkupText: "<tool_call>\n{\"a\":1}\n</tool_call>",
    });
    expect(resolved?.reason).toBe("rejected_control_markup");
    expect(resolved?.content).toBe(REJECTED_CONTROL_MARKUP_NOTICE);
  });

  it("never echoes rejected markup back into the answer", () => {
    // Echoing it would satisfy never-empty by creating parser leakage, which
    // the release proof counts as a failure.
    const leaky = "<tool_call>{}</tool_call>\n<invoke name=\"x\">\n<function=y>";
    const resolved = resolveNeverEmptyAssistantAnswer({
      ...base,
      rejectedControlMarkupText: leaky,
    });
    const leakRegex =
      /<think>|<\/think>|<tool_call>|<\/tool_call>|<function>|<invoke>|<minimax:tool_call>|<zyphra_tool_call>/i;
    expect(leakRegex.test(resolved?.content || "")).toBe(false);
  });

  it("explains a tool loop that produced no answer", () => {
    const resolved = resolveNeverEmptyAssistantAnswer({
      ...base,
      executedToolCallCount: 1,
      toolIterations: 1,
    });
    expect(resolved?.reason).toBe("tool_without_answer");
    expect(resolved?.content).toBe(TOOL_WITHOUT_ANSWER_NOTICE);
  });

  it("prefers the sanitizer fence over a notice when real prose exists", () => {
    const resolved = resolveNeverEmptyAssistantAnswer({
      ...base,
      preSanitizeContent: "real prose the sanitizer ate",
      rejectedControlMarkupText: "<tool_call>x",
    });
    expect(resolved?.reason).toBe("sanitized_to_empty");
  });

  it("explains a reasoning-only turn that never started an answer", () => {
    // Found live on qwen36 (reasoning required + tools, Responses door): 3 of 4
    // runs gave a first turn with zero content, tool phases of only
    // generating/done, and 7-8 persisted reasoning segments. Every other case
    // declined it, so the bubble stayed blank while the reasoning rail filled.
    const resolved = resolveNeverEmptyAssistantAnswer({
      ...base,
      reasoningContent: "Let me work through the probe steps...",
    });
    expect(resolved?.reason).toBe("reasoning_without_answer");
    expect(resolved?.content).toBe(REASONING_WITHOUT_ANSWER_NOTICE);
  });

  it("prefers the tool notice over the reasoning notice when both apply", () => {
    const resolved = resolveNeverEmptyAssistantAnswer({
      ...base,
      toolIterations: 1,
      executedToolCallCount: 1,
      reasoningContent: "thinking...",
    });
    expect(resolved?.reason).toBe("tool_without_answer");
  });

  it("does not fire the reasoning notice when the turn has content", () => {
    expect(
      resolveNeverEmptyAssistantAnswer({
        ...base,
        visibleAfterSanitize: "a real answer",
        reasoningContent: "thinking...",
      }),
    ).toBeNull();
  });

  it("ignores whitespace-only reasoning", () => {
    expect(
      resolveNeverEmptyAssistantAnswer({ ...base, reasoningContent: "   \n " }),
    ).toBeNull();
  });

  it("returns null when there is genuinely nothing to report", () => {
    expect(resolveNeverEmptyAssistantAnswer(base)).toBeNull();
  });
});
