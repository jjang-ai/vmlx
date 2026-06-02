import { describe, expect, it } from "vitest";
import {
  BACKEND_STDERR_DISCONNECT_NORMALIZED_LINE,
  normalizeBackendStderrChunk,
} from "../src/main/backend-stderr";

describe("backend stderr disconnect normalization", () => {
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
