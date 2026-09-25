import { createHash } from "node:crypto";

const NO_TOOLS_FINGERPRINT = "tools:none";

export interface ToolCapabilityDefinition {
  type?: unknown;
  function?: {
    name?: unknown;
    description?: unknown;
    parameters?: unknown;
  };
}

export interface ToolCapabilityHistoryEntry {
  role?: unknown;
  content?: unknown;
  toolCallsOaiJson?: string;
  toolCapabilityFingerprint?: string;
}

function canonicalize(value: unknown): unknown {
  if (Array.isArray(value)) return value.map(canonicalize);
  if (!value || typeof value !== "object") return value;
  return Object.fromEntries(
    Object.entries(value as Record<string, unknown>)
      .sort(([left], [right]) => left.localeCompare(right))
      .map(([key, nested]) => [key, canonicalize(nested)]),
  );
}

export function toolCapabilityFingerprint(
  tools: ToolCapabilityDefinition[],
): string {
  if (tools.length === 0) return NO_TOOLS_FINGERPRINT;
  const canonicalTools = tools
    .map((tool) => ({
      type: tool.type,
      function: canonicalize(tool.function),
    }))
    .sort((left, right) =>
      String(left.function && (left.function as any).name).localeCompare(
        String(right.function && (right.function as any).name),
      ),
    );
  return `tools:${createHash("sha256")
    .update(JSON.stringify(canonicalTools))
    .digest("hex")
    .slice(0, 20)}`;
}

export function toolCapabilityNames(
  tools: ToolCapabilityDefinition[],
): string[] {
  return tools
    .map((tool) =>
      typeof tool.function?.name === "string" ? tool.function.name : "",
    )
    .filter(Boolean)
    .sort();
}

function messageText(content: unknown): string {
  if (typeof content === "string") {
    // Database media rows store content parts as JSON. Decode only the known
    // content-array shape; quoted policy and nested attachment data are not
    // current-user instructions.
    try {
      const parts: unknown = JSON.parse(content);
      const contentTypes = new Set([
        "text", "input_text", "image_url", "input_image", "video_url",
        "input_video", "input_audio", "audio_url",
      ]);
      if (
        Array.isArray(parts) && parts.length > 0 &&
        parts.every((part) => part && typeof part === "object" &&
          contentTypes.has(part.type))
      ) {
        return parts
          .filter((part) => part.type === "text" || part.type === "input_text")
          .map((part) => typeof part.text === "string" ? part.text : "")
          .filter(Boolean)
          .join("\n");
      }
    } catch {
      // Plain text and malformed JSON retain their original meaning.
    }
    return content;
  }
  if (!Array.isArray(content)) return "";
  return content
    .map((part) => {
      if (!part || typeof part !== "object") return "";
      const record = part as Record<string, unknown>;
      if (typeof record.text === "string") return record.text;
      if (typeof record.content === "string") return record.content;
      return "";
    })
    .filter(Boolean)
    .join("\n");
}

export function latestUserMessageText(
  messages: ToolCapabilityHistoryEntry[],
): string {
  const latestUserMessage = [...messages]
    .reverse()
    .find((message) => message.role === "user");
  return messageText(latestUserMessage?.content);
}

function requestsTool(text: string, toolName: string): boolean {
  const escaped = toolName.replace(/[.*+?^${}()|[\]\\]/g, "\\$&");
  const noToolDirective = new RegExp(
    `\\b(?:do\\s+not|do(?:n't|n’t|nt)|never|without|must\\s+not|` +
      `should\\s+not|need\\s+not|no\\s+need\\s+to|` +
      `can\\s+you\\s+not|could\\s+you\\s+not)\\b[\\s\\S]{0,32}?` +
      `\\b(?:call(?:ing)?|use|using|invoke|invoking|run|running|execute|executing)\\b` +
      `[\\s\\S]{0,24}?\\b${escaped}\\b`,
    "i",
  );
  if (noToolDirective.test(text)) return false;
  return new RegExp(
    `(?:^|[.!?]\\s+|\\]\\s+)(?:(?:please|kindly)\\s+|` +
      `(?:can|could|would|will)\\s+you\\s+(?:(?:please|kindly)\\s+)?|` +
      `i\\s+(?:need|want|would\\s+like)\\s+you\\s+to\\s+|` +
      `you\\s+(?:must|should|need\\s+to|have\\s+to)\\s+)?` +
      `(?:call|use|invoke|run|execute)\\b[\\s\\S]{0,24}?\\b${escaped}\\b`,
    "i",
  ).test(text);
}

function assistantDeniesTool(text: string, toolName: string): boolean {
  const escaped = toolName.replace(/[.*+?^${}()|[\]\\]/g, "\\$&");
  const namedTool =
    `(?:the\\s+)?(?:\`${escaped}\`|` +
    `(?<![A-Za-z0-9_])${escaped}(?![A-Za-z0-9_]))` +
    `(?:\\s+(?:tool|function))?`;
  const unavailableState =
    "(?:not\\s+(?:available|enabled|provided|attached|accessible|callable)|unavailable|disabled)";
  return (
    new RegExp(
      `${namedTool}\\s+(?:is|was|remains?)\\s+${unavailableState}\\b`,
      "i",
    ).test(text) ||
    new RegExp(
      `${namedTool}\\s+(?:isn't|isn’t|isnt|wasn't|wasn’t|wasnt)\\s+` +
        "(?:available|enabled|provided|attached|accessible|callable)\\b",
      "i",
    ).test(text) ||
    new RegExp(
      `${namedTool}\\s+(?:cannot|can't|can’t)\\s+be\\s+` +
        "(?:used|called|invoked)\\b",
      "i",
    ).test(text) ||
    new RegExp(
      `\\b(?:cannot|can't|can’t|do\\s+not|do(?:n't|n’t|nt))\\s+` +
        "(?:currently\\s+)?have" +
        `(?:\\s+access\\s+to)?\\s+${namedTool}`,
      "i",
    ).test(text) ||
    new RegExp(
      `\\b(?:cannot|can't|can’t)\\s+` +
        `(?:call|use|invoke|run|execute)\\s+${namedTool}`,
      "i",
    ).test(text)
  );
}

function emittedToolNames(message: ToolCapabilityHistoryEntry): Set<string> {
  if (!message.toolCallsOaiJson) return new Set();
  try {
    const calls = JSON.parse(message.toolCallsOaiJson);
    if (!Array.isArray(calls)) return new Set();
    return new Set(
      calls
        .map((call) => call?.function?.name ?? call?.name)
        .filter((name): name is string => typeof name === "string"),
    );
  } catch {
    return new Set();
  }
}

export function historicalUnavailableToolNames(
  messages: ToolCapabilityHistoryEntry[],
  currentFingerprint: string,
  currentToolNames: string[],
): string[] {
  if (currentToolNames.length === 0) return [];

  const unavailableNames = new Set<string>();
  let latestUserText = "";
  for (const message of messages) {
    if (message.role === "user") {
      latestUserText = messageText(message.content);
      continue;
    }
    if (message.role !== "assistant") continue;

    if (message.toolCapabilityFingerprint !== currentFingerprint) {
      const assistantText = messageText(message.content);
      const calledTools = emittedToolNames(message);
      for (const name of currentToolNames) {
        if (
          !calledTools.has(name) &&
          requestsTool(latestUserText, name) &&
          assistantDeniesTool(assistantText, name)
        ) {
          unavailableNames.add(name);
        }
      }
    }
    latestUserText = "";
  }
  return [...unavailableNames].sort();
}

export function toolCapabilityEpochInstruction(
  previousFingerprints: Array<string | undefined>,
  currentFingerprint: string,
  currentToolNames: string[],
  historicallyUnavailableToolNames: string[],
  currentPromptAlreadyForbidsTools = false,
): string | undefined {
  // Keep the current capability instruction stable after a chat has crossed
  // tool epochs. Emitting it only on the first changed turn shifts the entire
  // rendered prompt on the next turn (Responses `instructions` are a leading
  // system message), which destroys otherwise reusable RAM/L2 prefix blocks.
  //
  // Legacy assistant rows have no fingerprint. Avoid injecting anything into
  // ordinary no-tools chats, while still repairing an unknown -> tools-ON
  // transition whenever the current catalog is attached.
  const crossedKnownEpoch = previousFingerprints.some(
    (fingerprint) =>
      fingerprint !== undefined && fingerprint !== currentFingerprint,
  );
  const hasUnknownLegacyEpoch =
    currentToolNames.length > 0 &&
    previousFingerprints.some((fingerprint) => fingerprint === undefined);
  if (!crossedKnownEpoch && !hasUnknownLegacyEpoch) return undefined;

  if (currentToolNames.length === 0) {
    // A current user instruction such as "do not call any tool" is already
    // the strongest no-tools authority. Repeating the generic capability-epoch
    // history warning is redundant and can condition small native-tool models
    // to copy content from the prior tool turn. Raw same-history LFM2 A/B
    // reproduced that stale visible suffix only when this extra prose was
    // present. Keep the warning for a silent UI Tools-Off transition.
    if (currentPromptAlreadyForbidsTools) return undefined;
    return [
      "Tool capability epoch changed for this request.",
      "No callable function schemas are attached to the current turn.",
      "Earlier assistant statements or tool calls belong to an older capability epoch and do not grant tools now.",
    ].join(" ");
  }

  // Attached schemas are already authoritative. Add the tools-ON repair only
  // when an earlier differently-versioned assistant turn followed a user
  // request for one of the current functions. That is the stale-capability
  // case this instruction fixes (for example, "file_info is unavailable").
  // Injecting it after an unrelated no-tool turn competes with native
  // small-model tool scaffolds and can make the model describe or simulate a
  // call instead of emitting one.
  if (historicallyUnavailableToolNames.length === 0) return undefined;

  if (historicallyUnavailableToolNames.length === 1) {
    const name = historicallyUnavailableToolNames[0];
    return `The earlier statement that ${name} was unavailable is outdated; the attached ${name} function is available for this request.`;
  }
  return `Earlier statements that ${historicallyUnavailableToolNames.join(
    ", ",
  )} were unavailable are outdated; those attached functions are available for this request.`;
}
