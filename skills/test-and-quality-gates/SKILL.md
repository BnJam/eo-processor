---
name: test-and-quality-gates
description: Run and interpret eo-processor quality gates (Rust+Python) and add the right tests for changes. Use when verifying work, fixing CI failures, or before committing PR-ready changes (fmt/clippy, ruff/mypy, pytest/tox, coverage, and minimal regression tests).
license: Proprietary. See repository LICENSE
compatibility: Assumes local Rust toolchain and Python tooling (tox/uv/pytest/ruff/mypy) are available; intended for this repository’s hybrid Rust+PyO3+Python package.
metadata:
  author: eo-processor maintainers
  version: "1.0"
---

# Test & quality gates skill (eo-processor)

Use this skill to keep changes safe and shippable by running the correct checks, adding the **right** tests, and diagnosing failures efficiently across **Rust**, **Python**, and **packaging** boundaries.

This repo is a hybrid:
- Rust core (PyO3 extension)
- Python package surface
- Tests primarily in Python
- Multiple quality gates (format, lint, type, unit, coverage, docs)

## When to activate
Activate this skill when you are:
- about to make a commit / open a PR
- fixing CI failures
- adding or changing a public API
- modifying numerical kernels (risk of subtle regressions)
- touching build/packaging (maturin, pyproject, Cargo)

---

## Gate 0: Decide what changed (scope-driven checklist)

First, classify your change. This determines which gates must run.

### A) Docs only
- Run: markdown sanity (optional), and ensure examples still look correct
- You can usually skip compilation checks unless docs reference generated API content

### B) Python-only logic (no Rust changes)
- Run: ruff + mypy (if configured) + pytest (or tox)
- Coverage if tests changed or coverage is tracked for this change

### C) Rust core change (PyO3 extension)
- Run: cargo fmt + clippy + (test/check)
- Rebuild extension and run Python tests that exercise the path
- If any Python API was touched, also run Python gates

### D) Public API change (new function / signature / behavior)
- Run: **everything relevant**:
  - Rust gates (if Rust involved)
  - Python gates (exports, typing, tests)
  - Docs updates and (if present) docs build
  - Coverage gates if the change impacts coverage tracking

**Rule:** the broader the public impact, the more gates you run.

---

## Gate 1: Test design (what tests to add)

### Always add at least one regression test for:
- a bug fix (reproduces old failure, asserts new behavior)
- a behavior change (encodes the new contract)
- a new public function (correctness + edge case)

### Recommended test coverage for numerical kernels (EO indices / ratios)
Add tests that cover:

1) **Correctness on tiny arrays**
- Hand-computable examples (2x2, 1D length 4)
- Assert shape and values within tolerance

2) **Stability edge cases**
- denominators near zero
- extremely small/large floats if relevant
- confirm behavior for divide-by-zero matches contract (finite vs inf vs NaN)

3) **Shape behavior**
- mismatched shapes should raise a predictable, Python-friendly error

4) **NaN / nodata behavior**
- if inputs can contain NaN: ensure behavior is consistent with the repo contract
- only sanitize NaNs if that is explicitly part of the function contract

5) **Dtype behavior**
- if float32 and float64 are supported, include both (or at least confirm output dtype is stable)

### Tolerances
- Prefer `np.testing.assert_allclose` (or equivalent)
- Use tighter tolerances for float64 than float32
- Avoid exact float equality unless the operation is guaranteed exact

### Avoid brittle tests
- Don’t test implementation details (loop order, internal eps value) unless the contract depends on it.
- Test behavior and contract, not incidental formatting.

---

## Gate 2: Rust quality gates (when Rust is touched)

### Required
1. Format:
- Run Rust formatter (repo standard)

2. Lint:
- Run clippy with warnings treated as errors (repo standard)

3. Compile/test:
- Run Rust tests if they exist
- If no Rust tests apply, at least ensure the crate checks/compiles

### Failure triage (Rust)
When a Rust gate fails:
- **fmt**: apply format; do not hand-fight formatting
- **clippy**:
  - treat warnings as real issues
  - prefer clear, idiomatic fixes
  - don’t suppress warnings unless there’s a documented reason
- **borrow checker / lifetime**:
  - simplify ownership at API boundaries (convert views once, avoid unnecessary clones)
- **performance concerns**:
  - avoid introducing extra allocations
  - watch for accidental temporaries in ndarray expressions

---

## Gate 3: Python quality gates (when Python/stubs/tests are touched)

### Required (as applicable)
1. Lint:
- run the repo’s linter (ruff is common here)

2. Types:
- run mypy (if configured) focusing on `python/eo_processor` and stubs

3. Unit tests:
- run pytest for relevant test modules
- run the canonical tox environment if that’s the repo standard

### Failure triage (Python)
- **ruff**: fix high-signal issues (unused imports, incorrect comparisons, unsafe patterns)
- **mypy**:
  - update stubs to match runtime behavior
  - prefer precise types over `Any` unless you truly can’t express it
- **pytest failures**:
  - confirm it’s not a stale compiled artifact (rebuild extension if Rust changed)
  - isolate failing test with a targeted run
  - inspect dtype/shape differences first for numerical failures

---

## Gate 4: Integration gate (Rust ↔ Python)

Use this gate whenever Rust core changed or new functions were added.

Checklist:
- Can you import `eo_processor` successfully?
- Can you call the changed/new function from Python?
- Do the Python exports and stubs match the compiled module?
- Do tests exercise the actual compiled path (not a fallback)?

Common integration failures:
- forgot to register the function in the PyO3 `#[pymodule]`
- forgot to re-export in `python/eo_processor/__init__.py`
- stubs (`__init__.pyi`) are out of sync
- stale build wheel/extension loaded in the environment

---

## Gate 5: Coverage gate (when tests changed)

Run coverage gates if:
- you added or removed tests
- you changed Python logic
- the repo’s CI enforces coverage thresholds
- coverage badges/artifacts are tracked

If coverage is updated and the repo expects it:
- run the coverage job
- regenerate any badge or report artifacts as per repo conventions

Do not “game” coverage with meaningless tests. Add tests that encode real behavior.

---

## Gate 6: Docs gate (when public API changes)

At minimum, ensure:
- `README.md` is updated if the public API surface changed
- quickstart/examples remain consistent with the API

If the repo has a docs build:
- run it when:
  - new functions are added to API listings
  - cross-references might break
  - docs tooling or dependencies changed

---

## Minimal “pre-commit” checklist for this repo

If you are about to commit, you should be able to say “yes” to all applicable items:

- [ ] Rust formatted (if Rust touched)
- [ ] Rust clippy passes with warnings as errors (if Rust touched)
- [ ] Rust compiles / tests pass (if Rust touched)
- [ ] Python lint passes (if Python touched)
- [ ] Type check passes (if Python/stubs touched)
- [ ] Unit tests pass (always run relevant subset; full tox if risky change)
- [ ] Public API exports/stubs/docs/tests consistent (if public API touched)
- [ ] Coverage updated if required by repo policy (if test/coverage impacted)
- [ ] No unrelated changes staged
- [ ] No secrets, no large binary artifacts

---

## Debugging playbook (fast triage)

### 1) Reproduce quickly
- run the smallest failing command (single test module, single lint tool)
- avoid running the full suite until you have a hypothesis

### 2) Investigate likely boundaries first
For EO numeric code, the usual suspects are:
- dtype conversion (float32 vs float64)
- integer-scaled inputs vs reflectance (0–1) assumptions
- divide-by-zero / epsilon differences
- shape mismatch or unexpected broadcasting

### 3) Reduce the case
- create a tiny reproducer array in a test
- assert intermediate invariants (shape, dtype, finite)

### 4) Fix root cause, then strengthen the test
- if you fixed a bug, add a test that would have failed before

---

## Expected output when this skill is used

When you activate this skill, produce a short actionable plan like:

1. **Gates to run** (based on change type)
2. **Tests to add/update** (what and why)
3. **Failure triage steps** (if something fails)
4. **Definition of done** (what must be green)

---

## Local references (repo)
- Core engineering rules & mandatory checklist: `AGENTS.md`
- User docs: `README.md`, `QUICKSTART.md`
- Workflows: `WORKFLOWS.md`
- Rust core: `src/`, `Cargo.toml`
- Python package surface: `python/eo_processor/`
- Tests: `tests/`
- Tool configuration: `pyproject.toml`, `tox.ini`, `pytest.ini`
