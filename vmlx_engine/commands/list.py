# SPDX-License-Identifier: Apache-2.0
"""Model list command for vmlx-engine.

Lists models found in a directory.
"""

import sys
from pathlib import Path


def list_command(args) -> None:
    """List models in a directory."""
    from ..utils.model_inspector import inspect_model

    search_dir = Path(args.directory)
    if not search_dir.is_dir():
        print(f"Error: Not a directory: {search_dir}")
        sys.exit(1)

    models = []
    for child in sorted(search_dir.iterdir()):
        if child.is_dir() and (child / "config.json").exists():
            try:
                info = inspect_model(str(child))
                models.append(info)
            except Exception:
                pass

    if not models:
        print(f"No models found in {search_dir}")
        return

    print(f"Found {len(models)} model(s) in {search_dir}:\n")
    for info in models:
        name = Path(info.model_path).name
        quant = f" ({info.quant_bits}-bit)" if info.is_quantized else ""
        size = f" [{info.total_weight_size_gb:.1f} GB]" if info.total_weight_size_gb else ""
        print(f"  {name} — {info.model_type}{quant}{size}")
