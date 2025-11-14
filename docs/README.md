# eo-processor Documentation Workspace

This `docs/` directory contains the Sphinx configuration and source files used to build the public API and architecture documentation for `eo-processor` (published on Read the Docs or via local builds).

---

## 1. Directory Layout

```
docs/
  requirements.txt        # RTD build environment (Python-only, mocks Rust extension)
  source/
    conf.py               # Sphinx configuration (themes, extensions, mocking)
    index.rst             # Root table of contents
    README.md             # Copied from project root (auto-managed)
    QUICKSTART.md         # Copied from project root (auto-managed)
    api/
      architecture.rst    # High-level Rust UDF architecture & performance model
      functions.rst       # Autosummary + inline API reference
    _static/
      custom.css          # Light styling overrides for theme
```

The root project’s `README.md` and `QUICKSTART.md` are copied into `source/` at build time by a helper in `conf.py` (`_link_external_markdown()`); do not manually edit the copies—edit the originals in the project root instead.

---

## 2. Building the Docs Locally

Prerequisites:
- Python (matching project version constraints)
- (Optional) Rust toolchain if you want real signatures from the compiled extension; otherwise mocking is fine.

Steps:

```bash
# Ensure dev dependencies are installed
uv sync

# Build wheel in development (optional; enables real imports)
maturin develop --release

# Build Sphinx HTML
make docs

# Open output
open docs/build/html/index.html
```

If you only need the Python wrappers and mocked imports, you can skip `maturin develop`. Sphinx will mock `eo_processor._core` (the PyO3 extension) using `autodoc_mock_imports`.

---

## 3. Publishing on Read the Docs (RTD)

RTD uses `docs/requirements.txt` to create an isolated environment. Because a Rust toolchain is not guaranteed on RTD, the compiled extension is *mocked*:

- All public Python names remain visible to autodoc.
- Docstrings from `python/eo_processor/__init__.py` are rendered.
- If future RTD builds support Rust, you can uncomment `maturin` and add `-e ..` to build the real extension for more precise signature introspection.

---

## 4. Adding a New Public Function (Checklist)

When you introduce a new Rust-backed function:

1. Implement in `src/` (e.g., `indices.rs`, `temporal.rs`, etc.) without `unsafe`.
2. Register with `wrap_pyfunction!` inside `src/lib.rs`.
3. Export via a Python wrapper in `python/eo_processor/__init__.py`.
4. Add a type stub entry to `python/eo_processor/__init__.pyi`.
5. Write or extend tests under `tests/test_<feature>.py`.
6. Update project root `README.md` (API summary, formula, example).
7. Update `source/api/functions.rst` autosummary list if needed.
8. Ensure docstring follows NumPy or Google style (Napoleon will parse).
9. Run `make docs` to verify the page appears and links resolve.
10. Bump version (minor) if public API expanded.

---

## 5. Updating Architecture Documentation

Edit `source/api/architecture.rst` for:
- Parallelism strategy changes
- New modules or performance thresholds
- Additional numerical stability techniques
- Planned optimization disclaimers

Keep the tone concise and technical—avoid marketing hype.

---

## 6. Markdown Integration (MyST)

The Sphinx configuration (`conf.py`) enables MyST Markdown:
- Use fenced code blocks ```python
- Use MyST directives or standard reStructuredText inside `.rst` pages
- Avoid duplicating content between `.rst` and Markdown; prefer linking
- To add a *new* Markdown guide: place `<name>.md` in `source/`, then reference it in a `toctree` within `index.rst`.

---

## 7. Autodoc & Autosummary Behavior

- `autosummary_generate = True` produces stub pages in `api/functions/`.
- `autodoc_mock_imports = ["eo_processor._core"]` prevents import errors without a compiled extension.
- If signatures look generic after compiling the real extension (e.g., missing parameter names), confirm wrappers in `__init__.py` have explicit signatures (avoid `*args, **kwargs` unless necessary).

---

## 8. Styling

Light CSS overrides live in `_static/custom.css`. Keep changes minimal:
- Maintain readability in both light/dark theme variants (Furo).
- Prefer high-contrast colors for admonition borders and code backgrounds.
- Test with local dark mode toggle.

---

## 9. Version Synchronization

Documentation pages reference `__version__` via import in `conf.py`. If:
- Python `__version__` differs from `pyproject.toml` version → fix immediately.
- After a version bump, rebuild docs to update HTML title and any displayed version badges.

---

## 10. Troubleshooting

Issue | Cause | Resolution
----- | ----- | ----------
Missing function page | Not in autosummary list | Add name to `index.rst` or `functions.rst` autosummary blocks
ImportError during build | Extension not compiled & not mocked | Ensure `autodoc_mock_imports` includes `_core`
Outdated README content in docs | Copied cache not refreshed | Delete `source/README.md` & rebuild, or run `make docs` again
Broken internal link | Path change or rename | Search for references in `source/` and update `:doc:` targets
Weird signature formatting | Wrapper uses dynamic args | Add explicit parameters to the Python wrapper function

---

## 11. CI / Automation (Future Option)

If adding a docs build job in CI:
- Run `make docs`
- Optionally publish `docs/build/html` as an artifact
- Validate that autosummary pages exist (e.g., grep for `normalized_difference.html`)

---

## 12. Conventions Recap

Do:
- Keep docstrings descriptive (Parameters / Returns / Notes)
- Include formulas for spectral indices
- State unit or scale assumptions (e.g., reflectance 0–1)
- Use `np.allclose` tolerance notes where relevant

Don’t:
- Introduce network or file I/O in examples that might hang builds
- Claim unverified performance gains
- Duplicate large code snippets verbatim across multiple pages

---

## 13. Quick Local Verification Script

After modifying docs, you can run:

```bash
make docs && python -c "import webbrowser; webbrowser.open('docs/build/html/index.html')"
```

---

## 14. Contact / Contribution

See project root `CONTRIBUTING.md` for full guidelines. Open issues for documentation improvement proposals, especially for:
- Clarity on array dimensionality expectations
- Additional change detection indices
- More elaborate benchmarking methodology

---

End of docs README.
