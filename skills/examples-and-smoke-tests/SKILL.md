---
name: examples-and-smoke-tests
description: Create and maintain runnable examples under examples/ for onboarding and smoke testing. Use when adding/changing public APIs, when you need a quick “does it work end-to-end?” script, or when updating examples/README.md to match the current eo-processor API.
license: Proprietary. See repository LICENSE
compatibility: Intended for this repository’s examples/ scripts. Assumes a local Python environment with the built eo-processor extension available (e.g., via maturin/uv/tox) and optional xarray/dask for advanced examples.
metadata:
  author: eo-processor maintainers
  version: "1.0"
---

# Examples & smoke tests skill (eo-processor)

Use this skill to add or update **examples** that serve two purposes:

1. **User onboarding**: small, readable scripts that demonstrate “how do I use this?”
2. **Smoke testing**: quick, repeatable end-to-end checks that catch integration breakage (Rust ↔ PyO3 ↔ Python API).

Examples are not a substitute for unit tests. They’re a *second line of defense* and a *first line of comprehension*.

---

## Where this skill applies

Primary directory:
- `examples/`

Common example files in this repo:
- `examples/basic_usage.py`
- `examples/xarray_dask_usage.py`
- `examples/spectral_indices_extended.py`
- `examples/temporal_operations.py`
- `examples/zonal_stats_example.py`
- `examples/README.md`

---

## When to activate

Activate this skill when you:
- Add a new public function or change an existing one
- Modify behavior that examples rely on (dtype/shape rules, NaN handling, epsilon rules)
- Need a quick confirmation that the compiled extension, Python exports, and runtime behavior line up
- Want to provide a minimal “copy/paste runnable” onboarding path
- Need to demonstrate a workflow from `WORKFLOWS.md` in executable form

Do not activate when:
- You’re only refactoring internals with no user-visible API impact (unless it might break runtime wiring)

---

## Design goals (what “good” looks like)

A good example is:
- **Runnable**: `python examples/<file>.py` works in the repo’s standard dev environment
- **Small and focused**: demonstrates one concept (or a tightly-related set), not the entire library
- **Deterministic enough**: uses fixed seeds for random data and avoids flaky assertions
- **Honest**: no claims about features that don’t exist; no placeholder TODOs
- **Aligned with public API**: imports from `eo_processor` as users would
- **Light on dependencies**: uses only core deps unless the example explicitly targets an optional stack (xarray/dask)

A bad example is:
- Too big to read
- Depends on local data files without telling the user
- Requires niche libraries without documenting them
- Prints pages of output without meaning
- Quietly passes while doing the wrong thing

---

## Example types to create (pick the smallest one that fits)

### Type A: “Hello world” onboarding script
Use when introducing a new user-facing function.

Characteristics:
- Small arrays (1D or 2D)
- Clear printed output (min/max/mean, a small preview)
- A short explanation in comments
- No complex dependencies

Example topics:
- Spectral indices (NDVI/NDWI/…)
- Masking utilities
- Simple temporal aggregations

### Type B: Smoke test script for integration
Use when you need a quick end-to-end check that users can run.

Characteristics:
- Imports the public API exactly as users do: `from eo_processor import ...`
- Exercises the main code path (compiled extension)
- Validates invariants:
  - shape preserved
  - dtype as expected (or at least float)
  - finite results (or NaN behavior is explicitly expected)
- Exits non-zero on failure (use `assert` or explicit checks)

Important:
- Keep it fast (seconds, not minutes).
- Avoid huge arrays unless specifically testing performance.

### Type C: “Workflow” example script
Use when demonstrating multi-step pipelines (e.g., temporal compositing + index + morphology).

Characteristics:
- A few logical steps, each with a comment header
- Prints intermediate summaries (shape/dtype/min/max/percentiles)
- Avoids large IO; uses synthetic data unless the repo explicitly includes sample data

### Type D: Optional-stack example (xarray/dask)
Use only when:
- the feature is about `xr.apply_ufunc`, dask parallelism, or chunk behavior
- the dependency is documented clearly at the top of the example

Characteristics:
- A small chunked array to keep runtime short
- Emphasizes correct usage patterns (`xr.apply_ufunc`, `dask="parallelized"`, etc.)
- Avoids requiring a distributed cluster (local threads is enough)

---

## Step-by-step workflow

### Step 1: Decide the example’s purpose and audience
Write down, in one sentence:
- “This example is for ____ and demonstrates ____.”

Also decide:
- Does it target beginners (Type A) or is it a smoke/integration check (Type B)?

### Step 2: Choose file naming and placement
Rules:
- Put examples in `examples/`.
- Prefer descriptive names (e.g., `ndmi_example.py`, `masking_example.py`).
- If updating, prefer editing an existing related example rather than creating duplicates.

### Step 3: Write the script with a consistent structure

Recommended structure:

1. Header comment:
   - what it demonstrates
   - how to run it
   - optional dependencies (if any)

2. Imports:
   - `numpy` first (almost always)
   - `eo_processor` imports next
   - optional packages last

3. Deterministic input:
   - use `np.random.default_rng(seed)` if randomness is needed
   - keep sizes modest (e.g., 256x256 or similar)

4. Compute:
   - call the function(s) under test

5. Validate invariants:
   - `assert out.shape == in.shape`
   - `assert np.isfinite(out).all()` where appropriate
   - if NaNs are expected: assert their presence and explain why

6. Print a human-readable summary:
   - dtype, shape
   - min/max/mean
   - small preview slice if helpful (`out[:3, :3]`)

7. Exit cleanly:
   - if using asserts, Python will raise on failure

### Step 4: Keep outputs stable
Guidelines:
- Prefer summaries over full arrays.
- If printing numeric summaries, consider rounding for readability.
- Seed randomness so results don’t change unexpectedly.

### Step 5: Update `examples/README.md` if appropriate
If you add a new example or make a meaningful change:
- Add a short bullet:
  - what it demonstrates
  - how to run it
  - notes about optional deps

---

## Assertions and correctness guidance

Examples can include lightweight checks, but avoid heavy logic.

Recommended checks:
- Shape preservation
- Basic finiteness (when contract expects it)
- Known-value sanity for tiny hand-constructed arrays (best for onboarding)

Avoid in examples:
- Tight tolerance comparisons against complex baselines (put those in `tests/`)
- Large exhaustive test matrices (again, `tests/`)

If you need stronger correctness validation:
- Add/extend a `tests/test_*.py` unit test, and keep the example focused on usability.

---

## Dependency discipline

### Default dependency set for most examples
- `numpy`
- `eo_processor`

### Optional dependencies (only when needed)
- `xarray`, `dask[array]` for xarray/dask examples
- Any ML libs only for ML-specific demos (keep optional and documented)

Rules:
- If an example needs optional deps, say so at the top with install hints.
- Don’t introduce new dependencies solely for an example unless the repo already supports them.

---

## Common pitfalls (avoid)

- Importing private/internal modules instead of the public API
- Using huge arrays that make examples slow or memory-heavy
- Relying on local files without documenting where to get them
- Silent failures (no asserts, no meaningful output)
- Flaky results due to unseeded randomness
- Examples drifting out of sync with `README.md` and stubs

---

## “Definition of done” for a new/updated example

You’re done when:
- [ ] The example runs from the repo root with the standard dev environment
- [ ] It imports from `eo_processor` (public API) rather than internals
- [ ] It includes small but meaningful invariant checks (shape/dtype/finite as appropriate)
- [ ] Output is readable and stable (seeded randomness if used)
- [ ] `examples/README.md` is updated if a new example was added or behavior changed
- [ ] If the example reflects a new/changed public API, docs and stubs are also aligned (see `docs-and-release` and `python-api-surface` skills)

---

## Suggested cookbook patterns (choose one)

### Pattern 1: Minimal onboarding
- Tiny inputs (hand-coded arrays)
- Known outputs or easy-to-interpret summaries

### Pattern 2: Small realistic raster
- Synthetic 2D arrays (e.g., 256x256 reflectance in [0, 1])
- Print min/max/mean and a tiny window

### Pattern 3: Smoke test mode
- Minimal runtime
- Assertions + short summary
- Clear failure signal

---

## Local references (repo)
- Examples directory: `examples/`
- Example index: `examples/README.md`
- User docs: `README.md`, `QUICKSTART.md`, `WORKFLOWS.md`
- Engineering rules: `AGENTS.md`
- Public API surface: `python/eo_processor/__init__.py`, `python/eo_processor/__init__.pyi`
- Tests: `tests/`
