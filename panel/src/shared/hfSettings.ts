export const HF_CANONICAL_ENDPOINT = "https://huggingface.co";

export function normalizeHfTokenSetting(raw: string | null | undefined): string | null {
  const token = raw?.trim();
  return token ? token : null;
}

export function normalizeHfEndpointSetting(raw: string | null | undefined): string | null {
  const value = raw?.trim();
  if (!value) return null;

  let url: URL;
  try {
    url = new URL(value);
  } catch {
    return null;
  }

  if (url.protocol !== "https:" && url.protocol !== "http:") return null;

  url.hash = "";
  url.search = "";
  url.pathname = url.pathname.replace(/\/+$/, "");

  return url.toString().replace(/\/+$/, "");
}

export function buildHfAuthHeaders(rawToken: string | null | undefined): Record<string, string> {
  const token = normalizeHfTokenSetting(rawToken);
  return token ? { Authorization: `Bearer ${token}` } : {};
}
