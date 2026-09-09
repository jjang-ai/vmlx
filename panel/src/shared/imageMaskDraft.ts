/** Decode a source and its existing mask together; late loads must not reset a newer draft. */
export function loadMaskPainterImages(
  sourceUrl: string,
  maskUrl: string | null | undefined,
  ready: (source: HTMLImageElement, mask: HTMLImageElement | null) => void,
  failed: () => void,
  createImage: () => HTMLImageElement = () => new Image(),
): () => void {
  let cancelled = false;
  const source = createImage();
  let mask: HTMLImageElement | null = null;
  const fail = () => { if (!cancelled) failed(); };
  source.onerror = fail;
  source.onload = () => {
    if (cancelled) return;
    if (!source.width || !source.height) { fail(); return; }
    if (!maskUrl) { ready(source, null); return; }
    mask = createImage();
    mask.onerror = fail;
    mask.onload = () => {
      if (cancelled || !mask) return;
      // A draft belongs to this exact source geometry. Never silently stretch it.
      if (mask.width !== source.width || mask.height !== source.height) { fail(); return; }
      ready(source, mask);
    };
    mask.src = maskUrl;
  };
  source.src = sourceUrl;
  return () => {
    cancelled = true;
    source.onload = null;
    source.onerror = null;
    if (mask) { mask.onload = null; mask.onerror = null; }
  };
}
