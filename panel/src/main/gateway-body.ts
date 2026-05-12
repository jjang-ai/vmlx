// SPDX-License-Identifier: Apache-2.0
// Binary-safe request body helpers for the API gateway.

function normalizeContentType(contentType: string | string[] | undefined): string {
  return Array.isArray(contentType) ? contentType.join(";") : contentType || "";
}

function escapeRegExp(value: string): string {
  return value.replace(/[.*+?^${}()|[\]\\]/g, "\\$&");
}

function multipartBoundary(contentType: string): string | undefined {
  const match = contentType.match(/(?:^|;)\s*boundary=(?:"([^"]+)"|([^;]+))/i);
  return (match?.[1] || match?.[2])?.trim() || undefined;
}

function stripOneTrailingLineBreak(value: string): string {
  if (value.endsWith("\r\n")) return value.slice(0, -2);
  if (value.endsWith("\n")) return value.slice(0, -1);
  return value;
}

export function extractMultipartFormField(
  body: Buffer,
  contentType: string | string[] | undefined,
  fieldName: string,
): string | undefined {
  const normalized = normalizeContentType(contentType);
  const boundary = multipartBoundary(normalized);
  if (!boundary) return undefined;

  const namePattern = new RegExp(
    `(?:^|;)\\s*name="${escapeRegExp(fieldName)}"(?:\\s*;|\\s*$)`,
    "i",
  );
  const delimiter = `--${boundary}`;
  const parts = body.toString("latin1").split(delimiter);

  for (let part of parts) {
    if (!part || part.startsWith("--")) continue;
    if (part.startsWith("\r\n")) part = part.slice(2);
    else if (part.startsWith("\n")) part = part.slice(1);

    let headerEnd = part.indexOf("\r\n\r\n");
    let separatorLength = 4;
    if (headerEnd < 0) {
      headerEnd = part.indexOf("\n\n");
      separatorLength = 2;
    }
    if (headerEnd < 0) continue;

    const headerText = part.slice(0, headerEnd);
    const contentDisposition = headerText
      .split(/\r?\n/)
      .find((line) => /^content-disposition\s*:/i.test(line));
    if (!contentDisposition || !namePattern.test(contentDisposition)) continue;

    const rawValue = part.slice(headerEnd + separatorLength);
    return stripOneTrailingLineBreak(rawValue).trim();
  }

  return undefined;
}

export function extractGatewayModelFromBody(
  body: Buffer,
  contentType: string | string[] | undefined,
): string | undefined {
  const normalized = normalizeContentType(contentType);

  if (/application\/json/i.test(normalized) || !normalized) {
    try {
      const parsed = JSON.parse(body.toString("utf8"));
      return typeof parsed?.model === "string" ? parsed.model : undefined;
    } catch (_) {
      // Fall through: clients sometimes omit Content-Type but send raw bytes.
    }
  }

  if (/application\/x-www-form-urlencoded/i.test(normalized)) {
    const params = new URLSearchParams(body.toString("utf8"));
    return params.get("model") || undefined;
  }

  if (/multipart\/form-data/i.test(normalized)) {
    return extractMultipartFormField(body, normalized, "model");
  }

  return undefined;
}
