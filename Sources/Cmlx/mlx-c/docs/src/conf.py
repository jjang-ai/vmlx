# Copyright © 2023 Apple Inc.

# -*- coding: utf-8 -*-

import os
import subprocess

# -- Project information -----------------------------------------------------

project = "MLX C"
copyright = "2023-2026, MLX Contributors"
author = "MLX Contributors"
version = "0.6.0"
release = version

# -- General configuration ---------------------------------------------------

extensions = ["breathe"]
breathe_projects = {"mlxc" : "../build/xml"}
breathe_default_project = "mlxc"
templates_path = ["_templates"]
html_static_path = ["_static"]
source_suffix = ".rst"
master_doc = "index"
highlight_language = "c"
pygments_style = "sphinx"

# -- Options for HTML output -------------------------------------------------

html_theme = "sphinx_book_theme"

html_theme_options = {
    "show_toc_level": 2,
    "repository_url": "https://github.com/ml-explore/mlx-c",
    "use_repository_button": True,
    "navigation_with_keys": False,
    "logo": {
        "image_light": "_static/mlx_logo.png",
        "image_dark": "_static/mlx_logo_dark.png",
    },
}


# -- Options for HTMLHelp output ---------------------------------------------

htmlhelp_basename = "mlxc_doc"
