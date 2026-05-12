import { describe, expect, it } from "vitest";
import { extractGatewayModelFromBody } from "../src/main/gateway-body";

function multipartBody(
  boundary: string,
  parts: Array<{ name: string; value: Buffer | string; filename?: string; contentType?: string }>,
): Buffer {
  const chunks: Buffer[] = [];
  for (const part of parts) {
    chunks.push(Buffer.from(`--${boundary}\r\n`, "latin1"));
    const filename = part.filename ? `; filename="${part.filename}"` : "";
    chunks.push(
      Buffer.from(
        `Content-Disposition: form-data; name="${part.name}"${filename}\r\n`,
        "latin1",
      ),
    );
    if (part.contentType) {
      chunks.push(Buffer.from(`Content-Type: ${part.contentType}\r\n`, "latin1"));
    }
    chunks.push(Buffer.from("\r\n", "latin1"));
    chunks.push(
      typeof part.value === "string"
        ? Buffer.from(part.value, "utf8")
        : part.value,
    );
    chunks.push(Buffer.from("\r\n", "latin1"));
  }
  chunks.push(Buffer.from(`--${boundary}--\r\n`, "latin1"));
  return Buffer.concat(chunks);
}

describe("gateway request body model extraction", () => {
  it("extracts model from JSON and form-urlencoded bodies", () => {
    expect(
      extractGatewayModelFromBody(
        Buffer.from(JSON.stringify({ model: "z-image-turbo" })),
        "application/json",
      ),
    ).toBe("z-image-turbo");

    expect(
      extractGatewayModelFromBody(
        Buffer.from("model=qwen&prompt=hello"),
        "application/x-www-form-urlencoded",
      ),
    ).toBe("qwen");
  });

  it("extracts multipart model without crossing into binary image parts", () => {
    const boundary = "----vmlx-test-boundary";
    const pngBytes = Buffer.concat([
      Buffer.from([0x89, 0x50, 0x4e, 0x47, 0x0d, 0x0a, 0x1a, 0x0a]),
      Buffer.from("fake-png-payload", "latin1"),
    ]);
    const body = multipartBody(boundary, [
      { name: "model", value: "z-image-turbo" },
      { name: "prompt", value: "turn it red" },
      {
        name: "image",
        value: pngBytes,
        filename: "input.png",
        contentType: "image/png",
      },
    ]);

    expect(
      extractGatewayModelFromBody(
        body,
        `multipart/form-data; boundary=${boundary}`,
      ),
    ).toBe("z-image-turbo");
  });

  it("extracts multipart model even when the file part appears first", () => {
    const boundary = "----vmlx-file-first";
    const body = multipartBody(boundary, [
      {
        name: "image",
        value: Buffer.from([0x89, 0x50, 0x4e, 0x47, 0x0d, 0x0a]),
        filename: "input.png",
        contentType: "image/png",
      },
      { name: "model", value: "flux-kontext" },
    ]);

    expect(
      extractGatewayModelFromBody(
        body,
        `multipart/form-data; boundary="${boundary}"`,
      ),
    ).toBe("flux-kontext");
  });

  it("does not treat uploaded image bytes as the model when model is absent", () => {
    const boundary = "----vmlx-no-model";
    const body = multipartBody(boundary, [
      {
        name: "image",
        value: Buffer.from([0x89, 0x50, 0x4e, 0x47, 0x0d, 0x0a]),
        filename: "input.png",
        contentType: "image/png",
      },
    ]);

    expect(
      extractGatewayModelFromBody(
        body,
        `multipart/form-data; boundary=${boundary}`,
      ),
    ).toBeUndefined();
  });
});
