# Sphinx configuration for eo-processor documentation.
# Focus: Python API docs + high-level Rust UDF architecture notes.
# This file is designed to work both locally and on Read the Docs (RTD).
#
# Key points:
# - We mock the compiled PyO3 extension module (`eo_processor._core`) so that
#   autodoc can import the Python package even when Rust wheels are not built.
# - Autosummary + autodoc generate a structured API reference.
# - Napoleon parses NumPy / Google style docstrings.
# - MyST enables Markdown (README, QUICKSTART) inclusion.
# - Intersphinx gives cross-links to Python and NumPy docs.
#
# If you add new public functions:
#   1. Export in python/eo_processor/__init__.py
#   2. Ensure docstring quality (parameters, returns, notes)
#   3. Re-run `make docs` locally or trigger RTD build.

import sys
import importlib
from datetime import datetime
from pathlib import Path

# ---------------------------------------------------------------------------
# Path setup
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[2]
PYTHON_SRC = PROJECT_ROOT / "python"
if str(PYTHON_SRC) not in sys.path:
    sys.path.insert(0, str(PYTHON_SRC))

# ---------------------------------------------------------------------------
# Project metadata
# ---------------------------------------------------------------------------
project = "eo-processor"
author = "Benjamin Smith"
copyright = f"{datetime.now():%Y}, {author}"

# Attempt to get version from package; fall back gracefully.
def _read_version():
    try:
        pkg = importlib.import_module("eo_processor")
        return getattr(pkg, "__version__", "0.0.0")
    except Exception:
        # On RTD before build of extension module this can fail.
        return "0.0.0"

version = _read_version()
release = version

# ---------------------------------------------------------------------------
# General configuration
# ---------------------------------------------------------------------------
extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.intersphinx",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
    "sphinx.ext.githubpages",
    "sphinx.ext.doctest",
    "sphinx.ext.ifconfig",
    "sphinx.ext.todo",
    "sphinx.ext.mathjax",
    "sphinx.ext.autodoc.typehints",
    "myst_parser",
]

# Enable Markdown parsing (MyST)
myst_enable_extensions = [
    "deflist",
    "colon_fence",
    "linkify",
]

# Autosummary / autodoc
# NOTE: autosummary generation disabled to prevent regenerating eo_processor.*.rst pages.
autosummary_generate = False
autosummary_imported_members = False
autodoc_member_order = "groupwise"
autodoc_typehints = "description"
autodoc_inherit_docstrings = True
add_module_names = False

# Show TODOs only in non-production builds
todo_include_todos = True

# Napoleon (Google/NumPy style support)
napoleon_google_docstring = True
napoleon_numpy_docstring = True
napoleon_use_param = True
napoleon_use_rtype = True
napoleon_preprocess_types = True

# When building on RTD the Rust extension may not be compiled yet.
# Mock the C-extension to avoid ImportErrors during autodoc.
autodoc_mock_imports = ["eo_processor._core"]

# Prevent warnings from failed type imports
nitpicky = False

# ---------------------------------------------------------------------------
# Intersphinx configuration
# ---------------------------------------------------------------------------
intersphinx_mapping = {
    "python": ("https://docs.python.org/3", {}),
    "numpy": ("https://numpy.org/doc/stable", {}),
}

# ---------------------------------------------------------------------------
# Templates / static assets
# ---------------------------------------------------------------------------
templates_path = ["_templates"]
html_static_path = ["_static"]
html_css_files = [
    "custom.css",
]

# ---------------------------------------------------------------------------
# Source parsing / suffixes
# ---------------------------------------------------------------------------
source_suffix = {
    ".rst": "restructuredtext",
    ".md": "markdown",
}

master_doc = "index"
language = "en"
exclude_patterns = [
    "_build",
    "Thumbs.db",
    ".DS_Store",
]

# ---------------------------------------------------------------------------
# HTML theme
# ---------------------------------------------------------------------------
html_theme = "furo"
html_title = f"{project} {release}"
html_show_sourcelink = True
html_show_search_summary = True

# ---------------------------------------------------------------------------
# Extension specific options
# ---------------------------------------------------------------------------
# Provide nicer default role fallback to code literal
default_role = "code"

pygments_style = "sphinx"

# ---------------------------------------------------------------------------
# RST / Markdown substitutions
# ---------------------------------------------------------------------------
rst_epilog = """
.. |Project| replace:: eo-processor
.. |Rust| replace:: **Rust**
.. |NumPy| replace:: **NumPy**
"""

# ---------------------------------------------------------------------------
# Custom hook: ensure README & QUICKSTART included without duplication
# ---------------------------------------------------------------------------
def _link_external_markdown():
    """
    Copy or symlink top-level Markdown docs into Sphinx source directory.
    We keep them at build time to avoid manual duplication.
    """
    src_docs = ["README.md", "QUICKSTART.md"]
    dest_dir = Path(__file__).parent
    for fname in src_docs:
        src_path = PROJECT_ROOT / fname
        dest_path = dest_dir / fname
        if src_path.exists():
            # Minimal copy (not symlink to be safe on RTD)
            try:
                text = src_path.read_text(encoding="utf-8")
                # Avoid writing if unchanged
                if not dest_path.exists() or dest_path.read_text(encoding="utf-8") != text:
                    dest_path.write_text(text, encoding="utf-8")
            except Exception as e:
                print(f"[conf.py] Warning: could not process {fname}: {e}")

_link_external_markdown()

# ---------------------------------------------------------------------------
# High-level Rust UDF architecture note (injected into a page)
# ---------------------------------------------------------------------------
RUST_ARCH_NOTE = """
The core performance advantage of eo-processor stems from its Rust user-defined
functions (UDFs) exported via PyO3:

- GIL Release: Each kernel executes in native code without holding Python's GIL, enabling
  true multi-core parallelism for operations that justify parallel iteration (e.g., temporal
  aggregations over large stacks, pairwise distance matrices).
- Memory Safety: All numerical loops and slice operations leverage Rust's borrow checker
  with zero 'unsafe' blocks, eliminating common segmentation faults encountered in manual
  Cython extensions.
- Predictable Types: Inputs (any numeric dtype) are coerced to float64 internally ensuring
  stable arithmetic and reducing branching for integer / float cases.
- Dimensional Dispatch: Functions like spectral indices support 1D–4D inputs via runtime
  shape inspection, avoiding the overhead of repeated Python-level conditional logic.
- Parallel Strategy: Rayon parallel iterators activated only beyond size thresholds to
  prevent micro-task overhead on small arrays.

Compared with pure NumPy:
- Multi-step kernels (e.g., SAVI median compositing with NaN filtering) avoid creation of
  multiple temporary arrays thanks to in-place or single-pass designs.
- Change detection formulas run directly on raw reflectance arrays without Python loops.
- Pairwise distance calculations selectively parallelize outer loops once (N*M > threshold)
  instead of relying on broadcasting expansions that increase peak memory footprints.

For integration in XArray/Dask workflows, these functions behave as standard ufunc-like
callables. They can be wrapped via xarray.apply_ufunc with 'dask=parallelized' to combine
distributed chunking with Rust's local parallel execution.
"""

def setup(app):
    # Make the note available as a config value
    app.add_config_value("rust_arch_note", RUST_ARCH_NOTE, "env")
    return {"version": release, "parallel_read_safe": True}
