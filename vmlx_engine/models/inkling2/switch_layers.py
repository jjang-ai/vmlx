"""Inkling block verification with the supported MLX-LM switch layers."""

import mlx.core as mx
from mlx_lm.models.switch_layers import SwitchGLU as _SwitchGLU

from .masked_quantized import DECODE_BLOCK_SIZE


class SwitchGLU(_SwitchGLU):
    def __call__(self, x, indices):
        if not self.training and x.ndim == 3 and 1 < x.shape[1] <= DECODE_BLOCK_SIZE:
            rows = [
                super(SwitchGLU, self).__call__(x[:, i:i + 1], indices[:, i:i + 1])
                for i in range(x.shape[1])
            ]
            return mx.concatenate(rows, axis=1)
        return super().__call__(x, indices)

    def masked(self, x, indices):
        """Skip unowned quantized routes when verifying a short decode block.

        Negative indices denote routes owned by another rank. Preserve each
        projection's native reduction, including the down projection's width.
        """
        from .masked_quantized import (
            grouped_masked_qmv,
            masked_qmv,
            paired_masked_mxfp8_qmv,
            paired_masked_qmv,
            supports_masked_qmv,
            supports_paired_mxfp8_qmv,
        )

        if (
            not self.training
            and x.ndim == 3
            and 1 <= x.shape[0] <= 4
            and 1 <= x.shape[1] <= DECODE_BLOCK_SIZE
            and x.shape[0] * indices.shape[-1] < 64
            and indices.size % 2 == 0
            and x.dtype in (mx.bfloat16, mx.float16, mx.float32)
            and mx.metal.is_available()
            and mx.default_device() == mx.gpu
            and all(
                supports_paired_mxfp8_qmv(layer)
                for layer in (self.up_proj, self.gate_proj, self.down_proj)
            )
        ):
            batch, length, hidden = x.shape
            order = mx.argsort(indices.reshape(-1))
            inverse = mx.argsort(order)
            selected = indices.reshape(-1)[order]
            values = x.reshape(-1, hidden)[order // indices.shape[-1]]
            up = paired_masked_mxfp8_qmv(values, self.up_proj, selected)
            gate = paired_masked_mxfp8_qmv(values, self.gate_proj, selected)
            output = paired_masked_mxfp8_qmv(
                self.activation(up, gate), self.down_proj, selected
            )
            return output[inverse].reshape(batch, length, indices.shape[-1], hidden)

        if (
            not self.training
            and x.ndim == 3
            and 1 <= x.shape[0] <= 4
            and 1 < x.shape[1] <= DECODE_BLOCK_SIZE
            and x.shape[0] * indices.shape[-1] < 64
            and getattr(self.down_proj, "mode", None) == "nvfp4"
            and x.dtype in (mx.bfloat16, mx.float16, mx.float32)
            and mx.metal.is_available()
            and mx.default_device() == mx.gpu
            and all(
                supports_masked_qmv(layer) for layer in (self.up_proj, self.gate_proj)
            )
        ):
            batch, length, hidden = x.shape
            routes = indices.shape[-1]
            if batch in (2, 4) and routes == 8:
                # Speculative positions often select the same experts. Pair
                # sorted routes to reuse packed weights without changing dots.
                order = mx.argsort(indices.reshape(-1))
                inverse = mx.argsort(order)
                selected = indices.reshape(-1)[order]
                values = x.reshape(-1, hidden)[order // routes]
                up = paired_masked_qmv(values, self.up_proj, selected)
                gate = paired_masked_qmv(values, self.gate_proj, selected)
                activated = self.activation(up, gate)
                if batch == 4 and supports_masked_qmv(
                    self.down_proj, input_alignment=256
                ):
                    output = grouped_masked_qmv(activated, self.down_proj, selected)
                else:
                    output = self.down_proj(
                        activated[:, None, None],
                        mx.maximum(selected, 0)[:, None],
                        sorted_indices=False,
                    ).reshape(selected.size, hidden)
                output = output[inverse].reshape(batch, length, routes, hidden)
                return mx.where((indices >= 0)[..., None], output, 0)
            selected = indices.reshape(-1, 1, routes)
            values = x.reshape(-1, 1, hidden)
            up = masked_qmv(values, self.up_proj, selected)
            gate = masked_qmv(values, self.gate_proj, selected)
            activated = self.activation(up, gate).reshape(batch, length, routes, -1)
            output = self.down_proj(
                mx.expand_dims(mx.contiguous(activated), -2),
                mx.maximum(indices, 0),
                sorted_indices=False,
            ).squeeze(-2)
        else:
            output = self(x, mx.maximum(indices, 0))
        return mx.where((indices >= 0)[..., None], output, 0)
