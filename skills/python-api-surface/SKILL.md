---
name: python-api-surface
description: Maintain the public Python API for eo-processor, including exports, __all__, docstrings, and type stubs. Use when adding/renaming functions, changing signatures/behavior, or ensuring Python typing and user-facing docs stay consistent with the Rust/PyO3 core.
license: Proprietary. See repository LICENSE
compatibility: Intended for this repository’s Python package (python/eo_processor) paired with a Rust PyO3 extension; assumes numpy/xarray usage patterns and standard Python tooling (pytest, ruff, mypy, tox/uv).
metadata:
  author: eo-processor maintainers
  version: "1.0"
---

# Python API surface & typing skill (eo-processor)

Use this skill when you need to change anything users import from `eo_processor`, or when you change behavior/signatures in the Rust core and must keep **Python exports + typing stubs + docs + tests** aligned.

This skill is intentionally opinionated: **public API stability and consistency beat cleverness**.

---

## What “Python API surface” means in this repo

The user-visible API typically includes:

- `python/eo_processor/__init__.py`
  - imports from the compiled extension module
  - `__all__`
  - module docstring / high-level documentation
  - occasionally `__version__` or metadata (if present in this repo)
- `python/eo_processor/__init__.pyi`
  - the canonical typing surface for users and CI type checks
- Any Python helpers/wrappers that sit on top of the Rust core (if present)
- Documentation that enumerates functions and guides usage:
  - `README.md` (always)
  - `QUICKSTART.md` (only when it materially improves onboarding)
  - `docs/` (if the repo generates longer-form docs)

**Key invariant:** if it’s public, it must be importable, typed, tested, and documented (at least minimally).

---

## Activation checklist (before you edit)

1. Identify whether the change is:
   - **new public function**
   - **renamed function**
   - **signature change**
   - **behavior change**
   - **docs-only change**
2. Identify the **single source of truth** for the behavior (usually Rust core), and confirm:
   - input shapes (1D/2D), dtype(s), and expected ranges
   - NaN/Inf behavior
   - divide-by-zero / epsilon strategy (if applicable)
3. Decide if the function should be **public**:
   - If it’s stable, broadly useful, and not experimental → public.
   - If it’s internal glue or unstable → keep internal, don’t export.

If any of those are unclear and you cannot infer them from existing functions, stop and ask the user for clarification (max 3 targeted questions).

---

## Step-by-step workflow

### Step 1: Wire exports correctly

When a function is intended to be public:

1. Import it into `python/eo_processor/__init__.py` from the extension module (or its wrapper module).
2. Add it to `__all__`.
3. Ensure naming matches docs and tests exactly.

**Rules:**
- Re-export under a single canonical name. Avoid multiple aliases unless there’s a compatibility story.
- Keep ordering/grouping consistent (e.g., indices together, masking together, temporal stats together).

### Step 2: Define the typing contract in `__init__.pyi`

The `.pyi` signature should be:
- accurate to runtime behavior (including accepted dtypes/shapes)
- helpful to users (clear parameter names, optional defaults)
- consistent across the library (similar functions should have similar typing style)

**Typing guidance:**
- Prefer `numpy.typing.NDArray` for NumPy arrays.
- If the project supports `xarray.DataArray` directly at runtime, either:
  - provide overloads, or
  - document that `xarray` usage is via `xr.apply_ufunc` and keep types NumPy-focused.
- Don’t promise more than the function actually supports (e.g., don’t type as accepting `ArrayLike` if it fails on lists).

**Return type:**
- If the function returns a new array: `NDArray[np.floating[Any]]` or a specific `np.float64` / `np.float32` depending on behavior.
- If dtype is guaranteed (common in Rust kernels), reflect that explicitly.

### Step 3: Update docstrings and high-level docs

For user-facing completeness:
- Ensure `README.md` mentions the function in the right section (API summary / feature list).
- Provide a minimal example snippet in docs if this is a new capability.
- If it’s an EO index, include:
  - formula
  - band meanings
  - expected input scaling (0–1 reflectance vs scaled integers)
  - output range and interpretation notes

Keep docs concise; move deep explanations to `docs/` if the repo uses it.

### Step 4: Add/adjust tests

Any public API change requires tests. At minimum:
- a correctness test on a small array
- shape mismatch behavior (if relevant)
- stability (near-zero denominator) tests for ratio/normalized difference functions

When behavior changes:
- update or add a regression test that captures the intended new behavior
- avoid “brittle” tests that encode implementation quirks rather than contract

---

## Compatibility and deprecation rules

### Renames
If renaming a public function:
- Prefer a deprecation window:
  - keep old name available and forward it to the new name
  - add a deprecation note in docs
  - (optionally) emit a `DeprecationWarning` from the old alias
- Update:
  - docs, tests, `__all__`, stubs
  - any examples referencing the old name

### Signature changes
Avoid breaking signature changes. If necessary:
- make new args optional with safe defaults
- document default behavior carefully
- update stubs and tests

If a breaking change is unavoidable, it must be explicit and coordinated with versioning policy.

---

## Quality bar (must pass before “done”)

- [ ] Public function is importable from `eo_processor`
- [ ] `__all__` includes it (if this repo uses `__all__` as canonical)
- [ ] `.pyi` typing matches runtime behavior
- [ ] Tests cover correctness + key edge cases
- [ ] Docs mention the function and provide guidance
- [ ] No redundant exports or conflicting names
- [ ] No undocumented behavior change

---

## Common pitfalls (avoid)

- Adding a Rust function but forgetting:
  - `python/eo_processor/__init__.py` export
  - `python/eo_processor/__init__.pyi` stub
  - docs update
  - tests
- Typing promises that runtime can’t fulfill (e.g., `ArrayLike` when only `ndarray` works)
- Silent behavioral changes without a corresponding doc/test update
- Inconsistent naming (e.g., `normalizedDifference` vs `normalized_difference`)
- Returning different dtype than typed/documented
- “Convenience” broadcasting rules in one function that don’t exist elsewhere

---

## Suggested “API change summary” format (use in your response/PR)

When you use this skill, summarize the API outcome:

1. **What changed**
2. **How to use it** (minimal snippet)
3. **Typing contract** (inputs/outputs)
4. **Edge cases** (NaN, divide-by-zero, shape mismatch)
5. **Tests added/updated**

---

## Local references (repo)

- Engineering and checklists: `AGENTS.md`
- User docs: `README.md`, `QUICKSTART.md`
- Workflows/examples: `WORKFLOWS.md`, `examples/`
- Python surface: `python/eo_processor/__init__.py`, `python/eo_processor/__init__.pyi`
- Tests: `tests/`
