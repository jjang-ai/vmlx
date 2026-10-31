import { readFileSync } from "node:fs";
import { describe, expect, it } from "vitest";
import {
  observeVisibleToolMarkup,
  type StreamToolBufferingState,
} from "../src/shared/responsesStreamRecovery";

// Live MLX Studio proof (DSV4.1, Responses wire, enable_thinking=false,
// --stream-interval 8): the server buffered " for each result.\n\n<｜DSML｜ calls>",
// sent `response.heartbeat {tool_call_generating: true}`, then released the
// prose as ordinary output_text when the call closed. The panel treated the
// heartbeat as leaked markup and suppressed every later visible delta, so the
// bubble showed "...waiting" and the next request replayed the short text.

type SseEvent =
  | { type: "response.output_text.delta"; delta: string }
  | { type: "response.heartbeat"; tool_call_generating: true }
  | { type: "response.output_text.done"; text: string }
  | { type: "response.output_item.done"; item: { type: "function_call"; name: string } };

/** Mirrors chat.ts: heartbeats mark generation; visible deltas go through emitDelta. */
function renderVisible(events: SseEvent[]): { visible: string; generating: boolean; calls: string[] } {
  let state: StreamToolBufferingState = { generating: false, markupDetected: false };
  let raw = "";
  let visible = "";
  const calls: string[] = [];
  for (const event of events) {
    if (event.type === "response.heartbeat" && event.tool_call_generating) {
      // chat.ts: `if (!clientToolCallBuffering) { clientToolCallBuffering = true; emitToolStatus(...) }`
      state = { ...state, generating: true };
    } else if (event.type === "response.output_text.delta") {
      raw += event.delta;
      const observed = observeVisibleToolMarkup(state, raw);
      state = observed.state;
      if (!observed.suppress) visible += event.delta;
    } else if (event.type === "response.output_item.done") {
      calls.push(event.item.name);
    }
  }
  return { visible, generating: state.generating, calls };
}

const LIVE_STEP_2: SseEvent[] = [
  { type: "response.output_text.delta", delta: "I'll run these four commands " },
  { type: "response.output_text.delta", delta: "sequentially, waiting" },
  { type: "response.heartbeat", tool_call_generating: true },
  { type: "response.heartbeat", tool_call_generating: true },
  { type: "response.output_text.delta", delta: " for each result." },
  {
    type: "response.output_text.done",
    text: "I'll run these four commands sequentially, waiting for each result.",
  },
  { type: "response.output_item.done", item: { type: "function_call", name: "run_command" } },
];

describe("Responses prose that precedes a tool call", () => {
  it("keeps the prose the server releases after its tool-generation heartbeat", () => {
    const out = renderVisible(LIVE_STEP_2);
    expect(out.visible).toBe(
      "I'll run these four commands sequentially, waiting for each result.",
    );
    expect(out.calls).toEqual(["run_command"]);
    expect(out.generating).toBe(true);
  });

  it("keeps a follow-up step's prose ('First' + ' done. Now the second.')", () => {
    const out = renderVisible([
      { type: "response.output_text.delta", delta: "First" },
      { type: "response.heartbeat", tool_call_generating: true },
      { type: "response.output_text.delta", delta: " done. Now the second." },
      { type: "response.output_item.done", item: { type: "function_call", name: "run_command" } },
    ]);
    expect(out.visible).toBe("First done. Now the second.");
  });

  it("a tool-only turn stays empty", () => {
    const out = renderVisible([
      { type: "response.heartbeat", tool_call_generating: true },
      { type: "response.output_item.done", item: { type: "function_call", name: "run_command" } },
    ]);
    expect(out.visible).toBe("");
  });

  it("still suppresses leaked line-start tool markup, with or without a heartbeat", () => {
    for (const heartbeatFirst of [false, true]) {
      const events: SseEvent[] = [
        { type: "response.output_text.delta", delta: "Checking.\n" },
        ...(heartbeatFirst
          ? [{ type: "response.heartbeat", tool_call_generating: true } as const]
          : []),
        { type: "response.output_text.delta", delta: "<tool_call>\n<function=run" },
        { type: "response.output_text.delta", delta: "_command>ls</function>" },
      ];
      const out = renderVisible(events);
      expect(out.visible).toBe("Checking.\n");
      expect(out.generating).toBe(true);
    }
  });

  it("reports the start of generation once, only when no heartbeat already did", () => {
    const fresh = observeVisibleToolMarkup(
      { generating: false, markupDetected: false },
      "x\n<tool_call>",
    );
    expect(fresh).toMatchObject({ suppress: true, startedGenerating: true });
    const afterHeartbeat = observeVisibleToolMarkup(
      { generating: true, markupDetected: false },
      "x\n<tool_call>",
    );
    expect(afterHeartbeat).toMatchObject({ suppress: true, startedGenerating: false });
    const prose = observeVisibleToolMarkup(
      { generating: true, markupDetected: false },
      "I'll use <tool_call> syntax in prose",
    );
    expect(prose).toMatchObject({ suppress: false, startedGenerating: false });
  });

  it("chat.ts derives visible suppression from markup only, not from the heartbeat flag", () => {
    const source = readFileSync(new URL("../src/main/ipc/chat.ts", import.meta.url), "utf8");
    const emitDelta = source.slice(
      source.indexOf("const emitDelta = ("),
      source.indexOf("// Strip any leaked chat template tokens from the delta"),
    );
    expect(emitDelta).toContain("observeVisibleToolMarkup(");
    expect(emitDelta).toContain("suppressVisibleToolDelta = observed.suppress;");
    expect(source).not.toContain("suppressVisibleToolDelta = clientToolCallBuffering;");
    const heartbeat = source.slice(
      source.indexOf('responsesEventType === "response.heartbeat"'),
      source.indexOf("// Reasoning delta from OpenAI Responses"),
    );
    expect(heartbeat).toContain("clientToolCallBuffering = true;");
    expect(heartbeat).not.toContain("clientToolMarkupBuffering = true");
    // Every iteration/stall/reconcile reset clears both flags.
    expect(source.split("clientToolCallBuffering = false;").length).toBe(
      source.split("clientToolMarkupBuffering = false;").length,
    );
  });
});
