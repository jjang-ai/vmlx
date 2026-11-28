# SPDX-License-Identifier: Apache-2.0
"""Native Vision Tower (`vit.*`) and VL preprocessing for Hy4-flash (`model_type='hy_v4_flash'`).

Architecture verified directly from shipped `Hy4-flash` checkpoint tensors & `vision_config`:
- `vit.embeddings.patch_embedding`: Conv2d(3 -> 1152, k=16, s=16, bias=True)
- `vit.embeddings.position_embedding`: Embedding(16385, 1152) (index 0 = CLS, 1..16384 = 128x128 grid)
- `vit.layers.0..26`: LayerNorm -> 16-head self-attention (`head_dim=72`, 2D RoPE `theta=10000.0`)
  -> LayerNorm -> GELU MLP (`1152 -> 4304 -> 1152`)
- `vit.perceive` (`projector_type='rmsnorm_mlp_adapter'`, `perceive_pre_norm=True`):
  RMSNorm(1152) -> Conv2d(1152 -> 4608, k=2, s=2) -> GELU -> Conv2d(4608 -> 3072, k=1, s=1)
  + row-ending `image_newline [3072]` -> `merged_h * (merged_w + 1)` tokens of dim `3072` per image.
"""

from __future__ import annotations

import base64
import io
import json
import logging
import math
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import mlx.core as mx
import mlx.nn as nn
import numpy as np
from PIL import Image

logger = logging.getLogger(__name__)

_FALSE = {"0", "false", "off", "no"}


def hy4_vl_enabled() -> bool:
    """True unless explicitly disabled via ``VMLX_HY4_VL=0``."""
    return os.environ.get("VMLX_HY4_VL", "1").strip().lower() not in _FALSE


def is_hy4_vl_model(model: Any) -> bool:
    """Structural check: True iff ``model`` is ``hy_v4_flash`` carrying a loaded ``vision_tower``."""
    try:
        mt = str(getattr(model, "model_type", "")).lower()
        if mt != "hy_v4_flash":
            inner = getattr(model, "model", None)
            mt = str(getattr(inner, "model_type", "")).lower()
        if mt != "hy_v4_flash":
            return False
        return getattr(model, "vision_tower", None) is not None
    except Exception:
        return False


def _rotate_half(x: mx.array) -> mx.array:
    half = x.shape[-1] // 2
    return mx.concatenate([-x[..., half:], x[..., :half]], axis=-1)


def _apply_2d_rope(
    q: mx.array,
    k: mx.array,
    grid_h: int,
    grid_w: int,
    *,
    theta: float = 10000.0,
    rope_percent: float = 1.0,
) -> Tuple[mx.array, mx.array]:
    """Apply 2D rotary embeddings to `[1, H, N, D]` visual Q/K for a `(grid_h, grid_w)` patch grid."""
    if rope_percent <= 0.0:
        return q, k
    D = int(q.shape[-1])
    rot_dim = int(D * rope_percent)
    rot_dim = (rot_dim // 4) * 4
    if rot_dim <= 0:
        return q, k
    half_d = rot_dim // 2  # split evenly between H and W axes
    inv_freq = 1.0 / (float(theta) ** (mx.arange(0, half_d, 2, dtype=mx.float32) / float(half_d)))

    h_pos = mx.repeat(mx.arange(grid_h, dtype=mx.float32), grid_w)  # [N]
    w_pos = mx.tile(mx.arange(grid_w, dtype=mx.float32), grid_h)  # [N]
    ang_h = h_pos[:, None] * inv_freq[None, :]  # [N, half_d/2]
    ang_w = w_pos[:, None] * inv_freq[None, :]  # [N, half_d/2]
    # Standard split-half 2D RoPE: [ang_h, ang_w, ang_h, ang_w] so _rotate_half pairs
    # H-dims with H-dims and W-dims with W-dims without cross-axis mixing.
    emb_half = mx.concatenate([ang_h, ang_w], axis=-1)  # [N, half_d]
    emb = mx.concatenate([emb_half, emb_half], axis=-1)  # [N, rot_dim]
    cos = mx.cos(emb)[None, None, :, :]
    sin = mx.sin(emb)[None, None, :, :]

    def _rot(t: mx.array) -> mx.array:
        if rot_dim == D:
            tf = t.astype(mx.float32)
            return (tf * cos + _rotate_half(tf) * sin).astype(t.dtype)
        t_rot = t[..., :rot_dim].astype(mx.float32)
        t_pass = t[..., rot_dim:]
        out_rot = (t_rot * cos + _rotate_half(t_rot) * sin).astype(t.dtype)
        return mx.concatenate([out_rot, t_pass], axis=-1)

    return _rot(q), _rot(k)


class HyV4VisionMLP(nn.Module):
    def __init__(self, hidden_size: int, intermediate_size: int) -> None:
        super().__init__()
        self.dense_h_to_4h = nn.Linear(hidden_size, intermediate_size, bias=True)
        self.dense_4h_to_h = nn.Linear(intermediate_size, hidden_size, bias=True)

    def __call__(self, x: mx.array) -> mx.array:
        return self.dense_4h_to_h(nn.gelu(self.dense_h_to_4h(x)))


class HyV4VisionAttention(nn.Module):
    def __init__(self, hidden_size: int, num_heads: int, rope_theta: float = 10000.0, rope_percent: float = 1.0) -> None:
        super().__init__()
        self.hidden_size = int(hidden_size)
        self.num_heads = int(num_heads)
        self.head_dim = self.hidden_size // self.num_heads
        self.scale = float(self.head_dim ** -0.5)
        self.rope_theta = float(rope_theta)
        self.rope_percent = float(rope_percent)
        self.q_proj = nn.Linear(hidden_size, hidden_size, bias=True)
        self.k_proj = nn.Linear(hidden_size, hidden_size, bias=True)
        self.v_proj = nn.Linear(hidden_size, hidden_size, bias=True)
        self.o_proj = nn.Linear(hidden_size, hidden_size, bias=True)

    def __call__(self, x: mx.array, grid_h: int, grid_w: int) -> mx.array:
        B, L, _ = x.shape
        q = self.q_proj(x).reshape(B, L, self.num_heads, self.head_dim).transpose(0, 2, 1, 3)
        k = self.k_proj(x).reshape(B, L, self.num_heads, self.head_dim).transpose(0, 2, 1, 3)
        v = self.v_proj(x).reshape(B, L, self.num_heads, self.head_dim).transpose(0, 2, 1, 3)
        q, k = _apply_2d_rope(q, k, grid_h, grid_w, theta=self.rope_theta, rope_percent=self.rope_percent)
        out = mx.fast.scaled_dot_product_attention(q, k, v, scale=self.scale)
        return self.o_proj(out.transpose(0, 2, 1, 3).reshape(B, L, self.hidden_size))


class HyV4VisionBlock(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        num_heads: int,
        eps: float = 1e-5,
        rope_theta: float = 10000.0,
        rope_percent: float = 1.0,
    ) -> None:
        super().__init__()
        self.input_layernorm = nn.LayerNorm(hidden_size, eps=eps)
        self.post_attention_layernorm = nn.LayerNorm(hidden_size, eps=eps)
        self.self_attn = HyV4VisionAttention(hidden_size, num_heads, rope_theta=rope_theta, rope_percent=rope_percent)
        self.mlp = HyV4VisionMLP(hidden_size, intermediate_size)

    def __call__(self, x: mx.array, grid_h: int, grid_w: int) -> mx.array:
        h = x + self.self_attn(self.input_layernorm(x), grid_h, grid_w)
        return h + self.mlp(self.post_attention_layernorm(h))


class HyV4VisionPatchEmbed(nn.Module):
    def __init__(self, vcfg: Dict[str, Any]) -> None:
        super().__init__()
        self.embed_dim = int(vcfg.get("hidden_size", 1152))
        self.patch_size = int(vcfg.get("patch_size", 16))
        self.num_channels = int(vcfg.get("num_channels", 3))
        max_img = int(vcfg.get("max_image_size", 2048))
        self.position_edge = max_img // self.patch_size  # 128
        self.num_positions = self.position_edge * self.position_edge + 1  # 16385
        self.patch_embedding = nn.Conv2d(
            self.num_channels,
            self.embed_dim,
            kernel_size=self.patch_size,
            stride=self.patch_size,
            bias=True,
        )
        self.position_embedding = nn.Embedding(self.num_positions, self.embed_dim)

    def _interpolate_pos_embed(self, target_h: int, target_w: int) -> mx.array:
        base = self.position_embedding.weight[1:, :].reshape(
            1, self.position_edge, self.position_edge, self.embed_dim
        )
        src_h = src_w = self.position_edge
        if src_h == target_h and src_w == target_w:
            return base.reshape(1, target_h * target_w, self.embed_dim)
        # align_corners=False bilinear interpolation matching PyTorch F.interpolate
        h_coords = (mx.arange(target_h, dtype=mx.float32) + 0.5) * (src_h / float(target_h)) - 0.5
        w_coords = (mx.arange(target_w, dtype=mx.float32) + 0.5) * (src_w / float(target_w)) - 0.5
        h_coords = mx.clip(h_coords, 0.0, float(src_h - 1))
        w_coords = mx.clip(w_coords, 0.0, float(src_w - 1))
        i0 = mx.floor(h_coords).astype(mx.int32)
        j0 = mx.floor(w_coords).astype(mx.int32)
        i1 = mx.minimum(i0 + 1, src_h - 1)
        j1 = mx.minimum(j0 + 1, src_w - 1)
        di = (h_coords - i0.astype(mx.float32))[:, None, None]
        dj = (w_coords - j0.astype(mx.float32))[None, :, None]
        p00 = base[0, i0][:, j0].astype(mx.float32)
        p01 = base[0, i0][:, j1].astype(mx.float32)
        p10 = base[0, i1][:, j0].astype(mx.float32)
        p11 = base[0, i1][:, j1].astype(mx.float32)
        out = (
            (1.0 - di) * (1.0 - dj) * p00
            + (1.0 - di) * dj * p01
            + di * (1.0 - dj) * p10
            + di * dj * p11
        )
        return out.reshape(1, target_h * target_w, self.embed_dim).astype(base.dtype)

    def __call__(self, patches_nhwc: mx.array, grid_h: int, grid_w: int) -> mx.array:
        # patches_nhwc: [N, P, P, 3]
        dtype = self.patch_embedding.weight.dtype
        emb = self.patch_embedding(patches_nhwc.astype(dtype)).reshape(1, grid_h * grid_w, self.embed_dim)
        return emb + self._interpolate_pos_embed(grid_h, grid_w)


class HyV4VisionPerceive(nn.Module):
    """RMSNorm + 2x2 stride-2 Conv2d -> GELU -> 1x1 Conv2d + row-ending `image_newline`."""

    def __init__(self, vcfg: Dict[str, Any]) -> None:
        super().__init__()
        self.hidden_size = int(vcfg.get("hidden_size", 1152))
        self.out_hidden_size = int(vcfg.get("out_hidden_size", vcfg.get("text_hidden_size", 3072)))
        self.merge_size = int(vcfg.get("spatial_merge_size", 2))
        self.pre_norm = bool(vcfg.get("perceive_pre_norm", True))
        eps = float(vcfg.get("rms_norm_eps", 1e-5))
        mid_dim = self.hidden_size * (self.merge_size * self.merge_size)  # 4608
        self.norm = nn.RMSNorm(self.hidden_size, eps=eps)
        self.proj_0 = nn.Conv2d(self.hidden_size, mid_dim, kernel_size=self.merge_size, stride=self.merge_size, bias=True)
        self.proj_2 = nn.Conv2d(mid_dim, self.out_hidden_size, kernel_size=1, stride=1, bias=True)
        self.image_newline = mx.zeros((self.out_hidden_size,), dtype=mx.bfloat16)

    def __call__(self, x: mx.array, grid_h: int, grid_w: int) -> mx.array:
        # x: [1, grid_h * grid_w, 1152]
        if self.pre_norm:
            x = self.norm(x)
        x = x.reshape(1, grid_h, grid_w, self.hidden_size)
        x = self.proj_2(nn.gelu(self.proj_0(x)))  # [1, merged_h, merged_w, out_hidden_size]
        mh, mw = int(x.shape[1]), int(x.shape[2])
        nl = mx.broadcast_to(self.image_newline[None, None, None, :].astype(x.dtype), (1, mh, 1, self.out_hidden_size))
        x = mx.concatenate([x, nl], axis=2)  # [1, mh, mw + 1, out_hidden_size]
        return x.reshape(1, mh * (mw + 1), self.out_hidden_size)


class HyV4VisionModel(nn.Module):
    """Full 27-layer BF16 Vision Tower (`vit.*`) for Hy4-flash."""

    def __init__(self, vcfg: Dict[str, Any]) -> None:
        super().__init__()
        self.config = dict(vcfg)
        h_dim = int(vcfg.get("hidden_size", 1152))
        inter_dim = int(vcfg.get("intermediate_size", 4304))
        n_heads = int(vcfg.get("num_attention_heads", 16))
        n_layers = int(vcfg.get("num_hidden_layers", 27))
        eps = float(vcfg.get("rms_norm_eps", 1e-5))
        rope_theta = float(vcfg.get("rope_2d_theta", 10000.0))
        rope_pct = float(vcfg.get("rope_2d_percent", 1.0))

        self.patch_size = int(vcfg.get("patch_size", 16))
        self.merge_size = int(vcfg.get("spatial_merge_size", 2))
        self.embeddings = HyV4VisionPatchEmbed(vcfg)
        self.layers = [
            HyV4VisionBlock(h_dim, inter_dim, n_heads, eps=eps, rope_theta=rope_theta, rope_percent=rope_pct)
            for _ in range(n_layers)
        ]
        self.perceive = HyV4VisionPerceive(vcfg)

    def encode_single(self, patches_nhwc: mx.array, grid_h: int, grid_w: int) -> mx.array:
        h = self.embeddings(patches_nhwc, grid_h, grid_w)
        for layer in self.layers:
            h = layer(h, grid_h, grid_w)
        return self.perceive(h, grid_h, grid_w)

    def __call__(self, pixel_values: mx.array, image_grid_thw: mx.array | Sequence[Sequence[int]]) -> mx.array:
        """Encode flattened patch batch `[total_patches, P, P, 3]` (or `[total_patches, 3*P*P]`) -> `[1, vis_tokens, D]`."""
        P = self.patch_size
        if pixel_values.ndim == 2:
            pv = pixel_values.reshape(-1, 3, P, P).transpose(0, 2, 3, 1)
        elif pixel_values.ndim == 4 and int(pixel_values.shape[1]) == 3 and int(pixel_values.shape[-1]) != 3:
            pv = pixel_values.transpose(0, 2, 3, 1)
        else:
            pv = pixel_values

        grids = image_grid_thw.tolist() if isinstance(image_grid_thw, mx.array) else list(image_grid_thw)
        feats: List[mx.array] = []
        cursor = 0
        for g in grids:
            t, gh, gw = int(g[0]), int(g[1]), int(g[2])
            for _ in range(t):
                n_p = gh * gw
                feats.append(self.encode_single(pv[cursor : cursor + n_p], gh, gw))
                cursor += n_p
        return mx.concatenate(feats, axis=1) if len(feats) > 1 else feats[0]


# ---------------------------------------------------------------------------
# Image preprocessing & 3D M-RoPE position construction
# ---------------------------------------------------------------------------
def smart_resize(
    height: int,
    width: int,
    *,
    factor: int = 32,
    min_pixels: int = 4096,
    max_pixels: int = 1024 * 1024,
) -> Tuple[int, int]:
    h = max(1, int(height))
    w = max(1, int(width))
    h_bar = max(factor, round(h / factor) * factor)
    w_bar = max(factor, round(w / factor) * factor)
    if h_bar * w_bar > max_pixels:
        beta = math.sqrt((h * w) / float(max_pixels))
        h_bar = max(factor, math.floor(h / beta / factor) * factor)
        w_bar = max(factor, math.floor(w / beta / factor) * factor)
    elif h_bar * w_bar < min_pixels:
        beta = math.sqrt(float(min_pixels) / (h * w))
        h_bar = max(factor, math.ceil(h * beta / factor) * factor)
        w_bar = max(factor, math.ceil(w * beta / factor) * factor)
    return int(h_bar), int(w_bar)


def _load_pil_image(source: Any) -> Image.Image:
    if isinstance(source, Image.Image):
        return source.convert("RGB")
    if isinstance(source, dict):
        source = source.get("url") or source.get("image") or source.get("path") or source
    if isinstance(source, (bytes, bytearray)):
        return Image.open(io.BytesIO(source)).convert("RGB")
    s = str(source or "").strip()
    if s.startswith("data:"):
        _, b64 = s.split(",", 1)
        return Image.open(io.BytesIO(base64.b64decode(b64))).convert("RGB")
    if s.startswith("file://"):
        s = s[7:]
    if s.startswith("http://") or s.startswith("https://"):
        import urllib.request

        with urllib.request.urlopen(s, timeout=30) as resp:
            return Image.open(io.BytesIO(resp.read())).convert("RGB")
    return Image.open(s).convert("RGB")


def _extract_image_patches(
    img: Image.Image,
    *,
    patch_size: int = 16,
    merge_size: int = 2,
    min_pixels: int = 4096,
    max_pixels: int = 1024 * 1024,
    mean: Tuple[float, float, float] = (0.5, 0.5, 0.5),
    std: Tuple[float, float, float] = (0.5, 0.5, 0.5),
) -> Tuple[np.ndarray, Tuple[int, int, int]]:
    factor = patch_size * merge_size
    rh, rw = smart_resize(img.height, img.width, factor=factor, min_pixels=min_pixels, max_pixels=max_pixels)
    if img.size != (rw, rh):
        img = img.resize((rw, rh), Image.BILINEAR)
    arr = np.asarray(img, dtype=np.float32) / 255.0
    arr = (arr - np.asarray(mean, dtype=np.float32)) / np.asarray(std, dtype=np.float32)  # [rh, rw, 3]
    gh = rh // patch_size
    gw = rw // patch_size
    # Split into [gh, gw, patch_size, patch_size, 3] -> [gh * gw, patch_size, patch_size, 3]
    patches = (
        arr.reshape(gh, patch_size, gw, patch_size, 3)
        .transpose(0, 2, 1, 3, 4)
        .reshape(gh * gw, patch_size, patch_size, 3)
    )
    return patches, (1, gh, gw)


def _normalize_messages_for_hy4_template(messages: Sequence[dict]) -> List[dict]:
    """Ensure image/video content parts match `chat_template.jinja`'s `item.type == 'image_url'` checks."""
    out: List[dict] = []
    for msg in messages:
        if not isinstance(msg, dict):
            out.append(msg)
            continue
        content = msg.get("content")
        if not isinstance(content, list):
            out.append(msg)
            continue
        parts = []
        for item in content:
            if not isinstance(item, dict):
                parts.append(item)
                continue
            itype = str(item.get("type", "") or "").lower()
            if itype in ("image", "image_url", "input_image") or "image_url" in item or "image" in item:
                parts.append({"type": "image_url", "image_url": {"url": ""}})
            elif itype in ("video", "video_url", "input_video") or "video_url" in item or "video" in item:
                parts.append({"type": "video_url", "video_url": {"url": ""}})
            elif itype in ("text", "input_text"):
                parts.append({"type": "text", "text": str(item.get("text", ""))})
            else:
                parts.append(item)
        out.append({**msg, "content": parts})
    return out


def preprocess_hy4_vl(
    model_path: str,
    tokenizer: Any,
    messages: Sequence[dict],
    images: Sequence[Any],
    *,
    enable_thinking: Optional[bool] = None,
    reasoning_effort: Any = None,
    tools: Optional[list] = None,
    chat_template_kwargs: Optional[dict] = None,
) -> Optional[Tuple[List[int], mx.array, mx.array]]:
    """Build `(expanded_token_ids, pixel_values [N, P, P, 3], image_grid_thw [M, 3])` for Hy4-flash VL."""
    if not images:
        return None
    root = Path(model_path)
    cfg = json.loads((root / "config.json").read_text()) if (root / "config.json").is_file() else {}
    vcfg = cfg.get("vision_config") or {}
    pcfg_path = root / "processor_config.json"
    ip_cfg = (json.loads(pcfg_path.read_text()).get("image_processor") or {}) if pcfg_path.is_file() else {}

    patch_size = int(ip_cfg.get("patch_size", vcfg.get("patch_size", 16)))
    merge_size = int(ip_cfg.get("merge_size", vcfg.get("spatial_merge_size", 2)))
    min_pixels = int(ip_cfg.get("min_pixels", 4096))
    # Cap interactive max_pixels to 1024x1024 (1056 visual tokens) unless overridden, preserving detail + speed
    env_max = int(os.environ.get("VMLX_HY4_MAX_PIXELS", str(1024 * 1024)) or (1024 * 1024))
    max_pixels = min(int(ip_cfg.get("max_pixels", env_max)), env_max)
    mean = tuple(float(x) for x in (ip_cfg.get("image_mean") or (0.5, 0.5, 0.5)))
    std = tuple(float(x) for x in (ip_cfg.get("image_std") or (0.5, 0.5, 0.5)))
    image_token_id = int(cfg.get("image_token_id", 120120))

    # 1. Extract patches + grid_thw for every image
    patch_chunks: List[np.ndarray] = []
    grids: List[Tuple[int, int, int]] = []
    for img_src in images:
        pil = _load_pil_image(img_src)
        patches, grid = _extract_image_patches(
            pil,
            patch_size=patch_size,
            merge_size=merge_size,
            min_pixels=min_pixels,
            max_pixels=max_pixels,
            mean=mean,
            std=std,
        )
        patch_chunks.append(patches)
        grids.append(grid)

    # 2. Render chat template with Hy4-flash kwarg contract
    from vmlx_engine.utils.chat_template_kwargs import build_chat_template_kwargs

    extra = dict(chat_template_kwargs or {})
    if reasoning_effort is not None and "reasoning_effort" not in extra:
        extra["reasoning_effort"] = reasoning_effort
    if tools:
        extra["tools"] = tools
    tpl_kwargs = build_chat_template_kwargs(
        enable_thinking=enable_thinking,
        extra=extra,
        tokenize=False,
        add_generation_prompt=True,
        model_type="hy_v4_flash",
    )
    norm_msgs = _normalize_messages_for_hy4_template(messages)
    # If images were supplied out-of-band (not inside message content parts), attach placeholders to last user msg
    has_img_part = any(
        isinstance(m.get("content"), list)
        and any(isinstance(p, dict) and p.get("type") == "image_url" for p in m["content"])
        for m in norm_msgs
        if isinstance(m, dict)
    )
    if not has_img_part and norm_msgs:
        for idx in range(len(norm_msgs) - 1, -1, -1):
            if isinstance(norm_msgs[idx], dict) and norm_msgs[idx].get("role") == "user":
                text_val = norm_msgs[idx].get("content", "")
                parts = [{"type": "image_url", "image_url": {"url": ""}} for _ in images]
                if isinstance(text_val, str) and text_val:
                    parts.append({"type": "text", "text": text_val})
                norm_msgs[idx] = {**norm_msgs[idx], "content": parts}
                break

    rendered = tokenizer.apply_chat_template(norm_msgs, **tpl_kwargs)
    tok = getattr(tokenizer, "_tokenizer", tokenizer)
    raw_ids = list(tok.encode(rendered, add_special_tokens=False))

    # 3. Expand each single `image_token_id` placeholder into `merged_h * (merged_w + 1)` tokens
    expanded: List[int] = []
    img_idx = 0
    for tid in raw_ids:
        if int(tid) == image_token_id and img_idx < len(grids):
            _, gh, gw = grids[img_idx]
            mh = gh // merge_size
            mw = gw // merge_size
            n_vis = mh * (mw + 1)
            expanded.extend([image_token_id] * n_vis)
            img_idx += 1
        else:
            expanded.append(int(tid))

    if img_idx != len(grids):
        logger.warning(
            "Hy4-flash VL: expanded %d image placeholders for %d image(s); skipping VL override",
            img_idx,
            len(grids),
        )
        return None

    pv = mx.array(np.concatenate(patch_chunks, axis=0), dtype=mx.bfloat16)
    grid_arr = mx.array(grids, dtype=mx.int32)
    return expanded, pv, grid_arr


def compute_hy4_mrope_positions(
    input_ids: Sequence[int],
    image_grid_thw: Sequence[Sequence[int]],
    *,
    image_token_id: int = 120120,
    merge_size: int = 2,
) -> Tuple[mx.array, int]:
    """Build exact 3D M-RoPE `position_ids_3d [3, 1, L]` and `rope_delta = max_pos + 1 - L`."""
    L = len(input_ids)
    pos_t = np.zeros(L, dtype=np.int32)
    pos_h = np.zeros(L, dtype=np.int32)
    pos_w = np.zeros(L, dtype=np.int32)

    cur_pos = 0
    i = 0
    img_idx = 0
    grids = [tuple(int(x) for x in g) for g in image_grid_thw]

    while i < L:
        if int(input_ids[i]) == image_token_id and img_idx < len(grids):
            _, gh, gw = grids[img_idx]
            mh = gh // merge_size
            mw = gw // merge_size
            row_w = mw + 1  # merged_w spatial tokens + 1 image_newline per row
            n_vis = mh * row_w
            end = min(L, i + n_vis)
            for k in range(end - i):
                r = k // row_w
                c = k % row_w
                pos_t[i + k] = cur_pos
                pos_h[i + k] = cur_pos + r
                pos_w[i + k] = cur_pos + c
            cur_pos += max(mh, row_w)
            i = end
            img_idx += 1
        else:
            pos_t[i] = cur_pos
            pos_h[i] = cur_pos
            pos_w[i] = cur_pos
            cur_pos += 1
            i += 1

    pos_3d = mx.array(np.stack([pos_t, pos_h, pos_w], axis=0)[:, None, :], dtype=mx.int32)  # [3, 1, L]
    rope_delta = cur_pos - L
    return pos_3d, int(rope_delta)
