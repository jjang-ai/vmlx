# SPDX-License-Identifier: Apache-2.0
"""Model info command for vmlx-engine.

Displays model metadata from config.json without loading weights.
"""

import sys


def info_command(args) -> None:
    """Display model metadata."""
    from ..utils.model_inspector import (
        format_model_info,
        inspect_model,
        resolve_model_path,
    )

    try:
        model_path = resolve_model_path(args.model)
    except FileNotFoundError as e:
        print(f"Error: {e}")
        sys.exit(1)

    try:
        info = inspect_model(model_path)
    except FileNotFoundError as e:
        print(f"Error: {e}")
        sys.exit(1)

    print(format_model_info(info))
