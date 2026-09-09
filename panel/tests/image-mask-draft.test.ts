import { describe, expect, it, vi } from 'vitest';
import { readFileSync } from 'node:fs';
import { loadMaskPainterImages } from '../src/shared/imageMaskDraft';

function setup(mask: string | null = 'mask') {
  const images: HTMLImageElement[] = [];
  const create = () => {
    const image = { width: 1024, height: 768, onload: null, onerror: null, src: '' } as unknown as HTMLImageElement;
    images.push(image);
    return image;
  };
  const ready = vi.fn(), failed = vi.fn();
  const cancel = loadMaskPainterImages('source', mask, ready, failed, create);
  const fire = (i: number, type: 'onload' | 'onerror' = 'onload') => images[i][type]?.call(images[i], {} as Event);
  return { images, ready, failed, cancel, fire };
}
describe('mask draft decode lifecycle', () => {
  it('waits for the existing mask and returns it without replacing it', () => {
    const x = setup(); expect(x.images[0].src).toBe('source');
    x.fire(0); expect(x.ready).not.toHaveBeenCalled(); expect(x.images[1].src).toBe('mask');
    x.fire(1); expect(x.ready).toHaveBeenCalledWith(x.images[0], x.images[1]);
  });
  it('starts an unmasked new source without loading an old selection', () => {
    const x = setup(null); x.fire(0);
    expect(x.images).toHaveLength(1); expect(x.ready).toHaveBeenCalledWith(x.images[0], null);
  });
  it.each([0, 1])('reports decode failure at boundary %s, never blank success', (which) => {
    const x = setup(); if (which) x.fire(0); x.fire(which, 'onerror');
    expect(x.failed).toHaveBeenCalledOnce(); expect(x.ready).not.toHaveBeenCalled();
  });
  it('rejects mismatched source geometry instead of stretching the saved mask', () => {
    const x = setup(); x.fire(0); x.images[1].width = 512; x.fire(1);
    expect(x.failed).toHaveBeenCalledOnce(); expect(x.ready).not.toHaveBeenCalled();
  });
  it('does not initialize an invalid empty source', () => {
    const x = setup(); x.images[0].width = 0; x.fire(0);
    expect(x.failed).toHaveBeenCalledOnce(); expect(x.images).toHaveLength(1);
  });
  it.each([0, 1])('cancels late callbacks at boundary %s', (which) => {
    const x = setup(); if (which) x.fire(0);
    const stale = x.images[which].onload; x.cancel();
    stale?.call(x.images[which], {} as Event);
    expect(x.ready).not.toHaveBeenCalled(); expect(x.failed).not.toHaveBeenCalled();
    expect(x.images).toHaveLength(which + 1);
  });
  it('wires current mask into the editor and gates Apply on completed initialization', () => {
    const root = new URL('../src/renderer/src/components/image/', import.meta.url);
    const bar = readFileSync(new URL('ImagePromptBar.tsx', root), 'utf8');
    const painter = readFileSync(new URL('MaskPainter.tsx', root), 'utf8');
    expect(bar).toContain('initialMaskDataUrl={maskBase64}');
    expect(painter).toContain('if (mask) ctx.drawImage(mask, 0, 0)');
    expect(painter).toContain('disabled={!imageLoaded}');
    expect(painter).toContain('[imageDataUrl, initialMaskDataUrl, redraw]');
    expect(painter).toContain('onClick={onCancel}');
  });
});
