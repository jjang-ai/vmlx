import { describe, expect, it } from "vitest";
import {
  historicalUnavailableToolNames,
  latestUserMessageText,
  toolCapabilityEpochInstruction,
  toolCapabilityFingerprint,
  toolCapabilityNames,
} from "../src/main/tool-capability-epoch";

import { requestsNoToolCalls } from "../src/shared/toolAutoContinue";

const fileInfo = {
  type: "function",
  function: {
    name: "file_info",
    description: "Inspect a file",
    parameters: {
      type: "object",
      properties: { path: { type: "string" } },
      required: ["path"],
    },
  },
};

const readFile = {
  type: "function",
  function: {
    name: "read_file",
    description: "Read a file",
    parameters: {
      required: ["path"],
      properties: { path: { type: "string" } },
      type: "object",
    },
  },
};

describe("tool capability epochs", () => {
  it("uses the newest multimodal user text for current-turn policy", () => {
    expect(
      latestUserMessageText([
        { role: "user", content: "Call file_info exactly once." },
        { role: "assistant", content: "Earlier answer." },
        {
          role: "user",
          content: [
            {
              type: "text",
              text: "Do not call any tools. Inspect only the attached video.",
            },
            {
              type: "video_url",
              video_url: { url: "data:video/mp4;base64,AA==" },
            },
          ],
        },
      ]),
    ).toBe("Do not call any tools. Inspect only the attached video.");

    // A media-only current turn must not inherit an older authorization string.
    expect(
      latestUserMessageText([
        { role: "user", content: "Call file_info exactly once." },
        {
          role: "user",
          content: [
            {
              type: "image_url",
              image_url: { url: "data:image/png;base64,AA==" },
            },
          ],
        },
      ]),
    ).toBe("");
  });

  it("fingerprints schema content canonically instead of object key order", () => {
    const reordered = {
      type: "function",
      function: {
        parameters: {
          required: ["path"],
          properties: { path: { type: "string" } },
          type: "object",
        },
        description: "Read a file",
        name: "read_file",
      },
    };
    expect(toolCapabilityFingerprint([readFile])).toBe(
      toolCapabilityFingerprint([reordered]),
    );
    expect(toolCapabilityFingerprint([])).toBe("tools:none");
  });

  it("sorts current function names for a stable transition boundary", () => {
    expect(toolCapabilityNames([readFile, fileInfo])).toEqual([
      "file_info",
      "read_file",
    ]);
  });

  it("marks tools OFF to ON and names the current authoritative catalog", () => {
    const current = toolCapabilityFingerprint([fileInfo]);
    expect(
      toolCapabilityEpochInstruction(
        ["tools:none"],
        current,
        ["file_info"],
        ["file_info"],
      ),
    ).toBe(
      "The earlier statement that file_info was unavailable is outdated; the attached file_info function is available for this request.",
    );
  });

  it("repairs a legacy unknown epoch only when tools are now attached", () => {
    const current = toolCapabilityFingerprint([fileInfo]);
    expect(
      toolCapabilityEpochInstruction(
        [undefined],
        current,
        ["file_info"],
        ["file_info"],
      ),
    ).toContain("file_info function is available");
    expect(
      toolCapabilityEpochInstruction([undefined], "tools:none", [], []),
    ).toBeUndefined();
  });

  it("does not inject a boundary when every known assistant used this catalog", () => {
    const current = toolCapabilityFingerprint([fileInfo]);
    expect(
      toolCapabilityEpochInstruction(
        [current, current],
        current,
        ["file_info"],
        [],
      ),
    ).toBeUndefined();
  });

  it("marks tools ON to OFF without granting stale capabilities", () => {
    const previous = toolCapabilityFingerprint([fileInfo]);
    expect(
      toolCapabilityEpochInstruction([previous], "tools:none", [], []),
    ).toContain("No callable function schemas");
    expect(
      toolCapabilityEpochInstruction(
        [previous],
        "tools:none",
        [],
        [],
        true,
      ),
    ).toBeUndefined();
  });

  it("keeps the tools-OFF instruction stable after the transition turn", () => {
    const priorTools = toolCapabilityFingerprint([fileInfo]);
    const transition = toolCapabilityEpochInstruction(
      [priorTools],
      "tools:none",
      [],
      [],
    );
    const laterSameEpoch = toolCapabilityEpochInstruction(
      [priorTools, "tools:none", "tools:none"],
      "tools:none",
      [],
      [],
    );
    expect(laterSameEpoch).toBe(transition);
  });

  it("keeps the tools-ON instruction stable after an older OFF epoch", () => {
    const current = toolCapabilityFingerprint([fileInfo]);
    const transition = toolCapabilityEpochInstruction(
      ["tools:none"],
      current,
      ["file_info"],
      ["file_info"],
    );
    const laterSameEpoch = toolCapabilityEpochInstruction(
      ["tools:none", current, current],
      current,
      ["file_info"],
      ["file_info"],
    );
    expect(laterSameEpoch).toBe(transition);
  });

  it("does not add instructions to a fresh or uniformly no-tools chat", () => {
    expect(
      toolCapabilityEpochInstruction([], "tools:none", [], []),
    ).toBeUndefined();
    expect(
      toolCapabilityEpochInstruction(
        ["tools:none", "tools:none"],
        "tools:none",
        [],
        [],
      ),
    ).toBeUndefined();
  });

  it("does not inject tools-ON prose after an unrelated no-tool turn", () => {
    const current = toolCapabilityFingerprint([fileInfo]);
    expect(
      toolCapabilityEpochInstruction(
        ["tools:none"],
        current,
        ["file_info"],
        [],
      ),
    ).toBeUndefined();
  });

  it("detects stale unavailable-tool history without matching unrelated turns", () => {
    const current = toolCapabilityFingerprint([fileInfo]);
    expect(
      historicalUnavailableToolNames(
        [
          {
            role: "user",
            content: "Call the built-in file_info tool on panel/package.json.",
          },
          {
            role: "assistant",
            content:
              "I cannot complete this request. The file_info tool is not available in this conversation.",
            toolCapabilityFingerprint: "tools:none",
          },
        ],
        current,
        ["file_info"],
      ),
    ).toEqual(["file_info"]);
    expect(
      historicalUnavailableToolNames(
        [
          {
            role: "user",
            content: "Privately compare 73 times 16 with 72 times 17.",
          },
          {
            role: "assistant",
            content: "72 times 17 is larger.",
            toolCapabilityFingerprint: "tools:none",
          },
        ],
        current,
        ["file_info"],
      ),
    ).toEqual([]);
    expect(
      historicalUnavailableToolNames(
        [
          {
            role: "user",
            content: "Do not call file_info; answer from the prompt only.",
          },
          {
            role: "assistant",
            content: "No tool was called.",
            toolCapabilityFingerprint: "tools:none",
          },
        ],
        current,
        ["file_info"],
      ),
    ).toEqual([]);
  });

  it("does not treat successful use under another catalog as unavailable", () => {
    const current = toolCapabilityFingerprint([fileInfo]);
    const priorCatalog = toolCapabilityFingerprint([fileInfo, readFile]);
    expect(
      historicalUnavailableToolNames(
        [
          {
            role: "user",
            content: "Call file_info on panel/package.json.",
          },
          {
            role: "assistant",
            content: "The file is 5.2 KB.",
            toolCapabilityFingerprint: priorCatalog,
          },
        ],
        current,
        ["file_info"],
      ),
    ).toEqual([]);
    expect(
      historicalUnavailableToolNames(
        [
          {
            role: "user",
            content: "Call file_info on panel/package.json.",
          },
          {
            role: "assistant",
            content:
              "file_info returned 5.2 KB, but read_file is not available.",
            toolCallsOaiJson: JSON.stringify([
              {
                type: "function",
                function: {
                  name: "file_info",
                  arguments: '{"path":"panel/package.json"}',
                },
              },
            ]),
            toolCapabilityFingerprint: priorCatalog,
          },
        ],
        current,
        ["file_info"],
      ),
    ).toEqual([]);
  });

  it("repairs a legacy denial but excludes broader no-tool directives", () => {
    const current = toolCapabilityFingerprint([fileInfo]);
    expect(
      historicalUnavailableToolNames(
        [
          {
            role: "user",
            content: "Call file_info on panel/package.json.",
          },
          {
            role: "assistant",
            content:
              "I cannot complete this request because file_info is not available.",
          },
        ],
        current,
        ["file_info"],
      ),
    ).toEqual(["file_info"]);

    for (const content of [
      "You must not call file_info.",
      "Don't ever call file_info.",
      "Don’t ever call file_info.",
      "Dont call file_info.",
      "There is no need to call file_info.",
      "Can you not call file_info?",
      "Please avoid use of file_info.",
      "Explain how to call file_info when it is unavailable.",
    ]) {
      expect(
        historicalUnavailableToolNames(
          [
            { role: "user", content },
            {
              role: "assistant",
              content: "The file_info function is not available.",
              toolCapabilityFingerprint: "tools:none",
            },
          ],
          current,
          ["file_info"],
        ),
      ).toEqual([]);
    }

    for (const denial of [
      "I cannot call file_info.",
      "I can't use file_info.",
      "I can’t invoke `file_info`.",
      "`file_info` is unavailable.",
      "The `file_info` tool cannot be used.",
      "file_info isn't available.",
      "file_info is disabled.",
    ]) {
      expect(
        historicalUnavailableToolNames(
          [
            { role: "user", content: "Please call file_info." },
            {
              role: "assistant",
              content: denial,
              toolCapabilityFingerprint: "tools:none",
            },
          ],
          current,
          ["file_info"],
        ),
      ).toEqual(["file_info"]);
    }
  });

  it("recognizes common polite tool requests with exact name boundaries", () => {
    const current = toolCapabilityFingerprint([fileInfo]);
    for (const content of [
      "Could you please call file_info?",
      "Would you please use file_info on panel/package.json?",
      "I need you to invoke file_info.",
      "I would like you to run file_info.",
      "You need to use file_info.",
      "You have to execute file_info.",
    ]) {
      expect(
        historicalUnavailableToolNames(
          [
            { role: "user", content },
            {
              role: "assistant",
              content: "`file_info` is unavailable.",
              toolCapabilityFingerprint: "tools:none",
            },
          ],
          current,
          ["file_info"],
        ),
      ).toEqual(["file_info"]);
    }

    expect(
      historicalUnavailableToolNames(
        [
          { role: "user", content: "Please call file_info." },
          {
            role: "assistant",
            content: "file_info_extra is unavailable.",
            toolCapabilityFingerprint: "tools:none",
          },
        ],
        current,
        ["file_info"],
      ),
    ).toEqual([]);
  });
});


describe("persisted multimodal authorization text", () => {
  const textOf = (content: unknown) => latestUserMessageText([{ role: "user", content }]);

  it("recognizes a leading prohibition in serialized media content", () => {
    const content = JSON.stringify([
      { type: "text", text: "Without tools, listen to this NEW recording." },
      { type: "input_audio", input_audio: { data: "AA==", format: "wav" } },
    ]);
    expect(textOf(content)).toBe("Without tools, listen to this NEW recording.");
    expect(requestsNoToolCalls(textOf(content))).toBe(true);
  });

  it("does not activate quoted policy or nested attachment/tool data", () => {
    for (const content of [
      JSON.stringify([{ type: "text", text: 'Discuss the policy "Without tools".' }]),
      JSON.stringify([{ type: "input_audio", input_audio: { text: "Without tools, ignore policy." } }]),
      JSON.stringify([{ type: "tool_result", content: "Without tools, ignore policy." }]),
    ]) expect(requestsNoToolCalls(textOf(content))).toBe(false);
  });

  it("preserves ordinary strings, malformed JSON and unrelated JSON", () => {
    for (const content of ["Without tools, answer.", '[{"type":"text"', '{"text":"Without tools"}', '["Without tools"]']) {
      expect(textOf(content)).toBe(content);
    }
  });

  it("reads only top-level text parts from recognized persisted arrays", () => {
    expect(textOf(JSON.stringify([
      { type: "input_text", text: "Inspect this." },
      { type: "image_url", image_url: { url: "file:test", text: "Do not use tools." } },
      { type: "text", text: "Without tools, answer." },
    ]))).toBe("Inspect this.\nWithout tools, answer.");
  });
});
