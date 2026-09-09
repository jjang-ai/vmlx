import { describe, expect, it } from "vitest";
import {
  BACKEND_STDERR_DISCONNECT_NORMALIZED_LINE,
  normalizeBackendStderrChunk,
} from "../src/main/backend-stderr";

describe("backend stderr disconnect normalization", () => {
  it("publishes carriage-return progress before the process finishes", () => {
    const first = normalizeBackendStderrChunk("", "\r  0%|          | 0/28");
    const second = normalizeBackendStderrChunk(first.pending, "\r  4%|x         | 1/28");
    expect(second.events).toEqual([{ type: "stderr", text: "  0%|          | 0/28\n" }]);
    const third = normalizeBackendStderrChunk(second.pending, "\r  7%|xx        | 2/28\n");
    expect(third.events.map(event => event.text)).toEqual([
      "  4%|x         | 1/28\n", "  7%|xx        | 2/28\n",
    ]);
    expect(third.pending).toBe("");
  });

  it("handles split CRLF without duplicate empty lines or losing traceback fragments", () => {
    const first = normalizeBackendStderrChunk("", "Traceback\r");
    const second = normalizeBackendStderrChunk(first.pending, "\nValueError: bad image\r\n");
    expect([...first.events, ...second.events].map(event => event.text)).toEqual([
      "Traceback\n", "ValueError: bad image\n",
    ]);
    expect(second.pending).toBe("");
  });
  it("normalizes split write EPIPE chunks before raw stderr reaches the UI", () => {
    const first = normalizeBackendStderrChunk("", "Traceback line\nError: write ");

    expect(first).toEqual({
      pending: "Error: write ",
      events: [{ type: "stderr", text: "Traceback line\n" }],
    });

    const second = normalizeBackendStderrChunk(
      first.pending,
      "EPIPE\nValueError: real failure\n",
    );

    expect(second).toEqual({
      pending: "",
      events: [
        {
          type: "disconnect",
          text: BACKEND_STDERR_DISCONNECT_NORMALIZED_LINE,
        },
        { type: "stderr", text: "ValueError: real failure\n" },
      ],
    });
  });

  it("normalizes a no-newline disconnect chunk without waiting for process exit", () => {
    expect(normalizeBackendStderrChunk("", "Error: write EPIPE")).toEqual({
      pending: "",
      events: [
        {
          type: "disconnect",
          text: BACKEND_STDERR_DISCONNECT_NORMALIZED_LINE,
        },
      ],
    });
  });
});
