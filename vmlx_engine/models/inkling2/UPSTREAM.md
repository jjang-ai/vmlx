# Inkling 2 Mini upstream sources

The model, processor, and native packed weight support originate from
[MLX-VLM 111ac668](https://github.com/Neywa-Labs/mlx-vlm/tree/111ac668078e017c9db1eb3936f4d7a6e0a88f89).
The MLX-VLM license and original source notices apply. The language model uses
the vMLX-pinned MLX-LM cache ABI and switch layers. The bundled Inkling
short-block and cache kernels are scoped to this family.
The BF16 DSpark drafter and its configuration come from the same revision.
