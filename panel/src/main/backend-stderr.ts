export const BACKEND_STDERR_DISCONNECT_NORMALIZED_LINE =
  "[SERVER] Client disconnected during stream; backend pipe closed cleanly.\n";

export type BackendStderrEvent =
  | { type: "disconnect"; text: string }
  | { type: "stderr"; text: string };

export function isExpectedBackendStderrDisconnectLine(text: string): boolean {
  return /(?:EPIPE|write EPIPE|broken pipe|socket hang up|connection reset|ECONNRESET|ERR_STREAM_DESTROYED|ERR_STREAM_WRITE_AFTER_END|premature close|stream.*destroyed|write after end)/i.test(
    text,
  );
}

export function normalizeBackendStderrChunk(
  pending: string,
  chunk: string,
): { pending: string; events: BackendStderrEvent[] } {
  let remaining = pending + chunk;
  const events: BackendStderrEvent[] = [];

  while (true) {
    // Terminal progress writers use carriage returns without newlines.
    // Preserve those records instead of buffering an entire denoising run.
    const newlineIndex = remaining.search(/[\r\n]/);
    if (newlineIndex < 0) break;
    const line = remaining.slice(0, newlineIndex) + "\n";
    const separatorLength = remaining.slice(newlineIndex, newlineIndex + 2) === "\r\n" ? 2 : 1;
    remaining = remaining.slice(newlineIndex + separatorLength);
    if (line === "\n") continue;
    if (isExpectedBackendStderrDisconnectLine(line)) {
      events.push({
        type: "disconnect",
        text: BACKEND_STDERR_DISCONNECT_NORMALIZED_LINE,
      });
    } else {
      events.push({ type: "stderr", text: line });
    }
  }

  if (remaining && isExpectedBackendStderrDisconnectLine(remaining)) {
    events.push({
      type: "disconnect",
      text: BACKEND_STDERR_DISCONNECT_NORMALIZED_LINE,
    });
    remaining = "";
  }

  return { pending: remaining, events };
}
