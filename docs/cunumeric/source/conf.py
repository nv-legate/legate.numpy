# Copyright 2024 NVIDIA Corporation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#

from os import getenv

from cunumeric import __version__

SWITCHER_PROD = "https://docs.nvidia.com/cunumeric/switcher.json"
SWITCHER_DEV = "http://localhost:8000/switcher.json"
JSON_URL = SWITCHER_DEV if getenv("SWITCHER_DEV") == "1" else SWITCHER_PROD

# -- Project information -----------------------------------------------------

project = "NVIDIA cuNumeric"
copyright = "2024, NVIDIA"
author = "NVIDIA Corporation"

# -- General configuration ---------------------------------------------------

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.intersphinx",
    "sphinx.ext.mathjax",
    "sphinx.ext.napoleon",
    "sphinx_copybutton",
    "IPython.sphinxext.ipython_console_highlighting",
    "myst_parser",
    "nbsphinx",
    "legate._sphinxext.settings",
    "cunumeric._sphinxext.comparison_table",
    "cunumeric._sphinxext.implemented_index",
    "cunumeric._sphinxext.missing_refs",
    "cunumeric._sphinxext.ufunc_formatter",
]

source_suffix = {".rst": "restructuredtext", ".md": "markdown"}

# -- Options for HTML output -------------------------------------------------

html_context = {
    # "default_mode": "light",
    "AUTHOR": author,
    "DESCRIPTION": "cuNumeric documentation site.",
}

html_static_path = ["_static"]

# This is pretty kludgy but the nv theme is not publicly available to
# install on CI, etc. We will use the pydata theme in those situations
if getenv("NV_THEME") == "1":
    html_theme = "nvidia_sphinx_theme"

    html_theme_options = {
        "switcher": {
            "json_url": JSON_URL,
            "navbar_start": ["navbar-logo", "version-switcher"],
            "version_match": ".".join(__version__.split(".", 2)[:2]),
        }
    }

else:
    html_theme = "pydata_sphinx_theme"

    html_theme_options = {
        "footer_start": ["copyright"],
        "github_url": "https://github.com/nv-legate/cunumeric",
        # https://github.com/pydata/pydata-sphinx-theme/issues/1220
        "icon_links": [],
        "logo": {
            "text": project,
            "link": "https://nv-legate.github.io/cunumeric",
        },
        "navbar_align": "left",
        "navbar_end": ["navbar-icon-links", "theme-switcher"],
        "primary_sidebar_end": ["indices.html"],
        "secondary_sidebar_items": ["page-toc"],
        "show_nav_level": 2,
        "show_toc_level": 2,
    }

templates_path = ["_templates"]

# -- Options for extensions --------------------------------------------------

autosummary_generate = True

copybutton_prompt_text = ">>> "

intersphinx_mapping = {
    "python": ("https://docs.python.org/3/", None),
    "numpy": ("https://numpy.org/doc/stable/", None),
}

mathjax_path = "https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-mml-chtml.js"

napoleon_custom_sections = [("Availability", "returns_style")]

nbsphinx_execute = "never"

pygments_style = "sphinx"


def setup(app):
    app.add_css_file("params.css")
