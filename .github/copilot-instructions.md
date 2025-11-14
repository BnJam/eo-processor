# GitHub Copilot / AI Agent Instructions

This document defines how AI code assistants (e.g., GitHub Copilot, autonomous agents) should behave when generating, modifying, or proposing code in the `eo-processor` repository. It complements `AGENTS.md` and focuses on inline assistance, code completion, and automated changes.

---

## 1. Scope of Assistance

AI assistance MAY:
- Suggest Rust functions implementing EO spectral indices and numeric transforms
- Provide Python integration layers (imports, type stubs, tests)
- Improve performance in pure Rust (keeping safety)
- Add unit tests and benchmarks (lightweight)
- Refactor for clarity without breaking public APIs
- Maintain or regenerate `coverage-badge.svg` via existing script logic
- Propose documentation updates (README, QUICKSTART)
- Add missing type hints, docstrings, or comments

AI assistance MUST NOT:
- Introduce external services, network calls, or file I/O into the core computational path
- Add unchecked dependencies without justification
- Use `unsafe` Rust blocks (unless explicitly approved and documented)
- Commit partial implementations (functions without tests or exports)
- Generate large boilerplate unrelated to repository goals
- Fabricate data, benchmarks, or security claims

---

## 2. Repository Structure Awareness

Key areas for code suggestions:
- `src/` – Rust source (PyO3 module functions)
- `python/eo_processor/` – Python package exports and type stubs
- `tests/` – Python tests (add test_*.py files here)
- `scripts/` – Utility scripts (badge generation, etc.)
- `pyproject.toml` – Metadata, versioning, dependencies
- `tox.ini` – Test environments, coverage configuration

Do not invent directories. Use existing patterns (e.g., 1D, 2D variants where appropriate).

---

## 3. Coding Conventions

Rust:
- Format with `cargo fmt`
- Zero clippy warnings (`cargo clippy -- -D warnings`)
- Prefer slice/array operations over explicit loops where clear
- Avoid `unwrap()`—use explicit error handling or safe assumptions
- Enforce shape matching before arithmetic when combining arrays

Python:
- Export new functions in `__init__.py`
- Add type stubs in `__init__.pyi`
- Keep tests deterministic (no random seeds unless specified)
- Use NumPy array assertions (`np.allclose`, `np.isfinite`)
- Prefer descriptive test names (`test_<function>_<case>`)

Documentation:
- Add formulas for new spectral indices
- Include minimal runnable examples
- Maintain consistency with existing README function lists

---

## 4. Performance Guidance

When suggesting optimizations:
- Preserve numerical correctness (tolerance ≤ 1e-9)
- Prefer vectorized ndarray operations in Rust
- Avoid micro-optimizations that reduce readability unless >20% speed gain
- Suggest benchmarking snippets only if relevant

Benchmarks must be realistic (large arrays, e.g., 5000 x 5000).

---

## 5. Safe Rust Patterns

DO:
- Use `ndarray` arithmetic operations
- Validate shapes early
- Use small constants like `1e-10` for numeric stability

DON'T:
- Introduce `unsafe`
- Allocate excessively in tight loops
- Return inconsistent array dimensions
- Silently swallow errors

---

## 6. Adding New Functionality (Checklist)

For each new function:
1. Implement in Rust (`src/lib.rs`)
2. Register with `#[pyfunction]` and add to `#[pymodule]`
3. Export in `python/eo_processor/__init__.py`
4. Add type stub in `python/eo_processor/__init__.pyi`
5. Create test file in `tests/`
6. Update README function list and examples
7. Run full pre-commit checklist (Section 8)
8. Consider version bump (patch/minor/major)

---

## 7. Versioning Rules (SemVer)

- Patch: Bug fixes / internal refactors
- Minor: New functions (backward-compatible)
- Major: Breaking changes (signature changes, removals)

AI agents should tag suggestions that require version bumps.

---

## 8. Pre-Commit / Pre-Push Checklist (MANDATORY)

Before generating or validating a commit, ensure:
- [ ] Rust code formatted (`cargo fmt`)
- [ ] No clippy warnings
- [ ] `cargo check --lib` succeeds (or `cargo test` if tests exist)
- [ ] Python style/lint: ruff passes
- [ ] `mypy python/eo_processor` has no new errors
- [ ] Tests pass (`tox` or individual env)
- [ ] Coverage updated if logic changed (`tox -e coverage`)
- [ ] Badge regenerated if coverage changed:
      `python scripts/generate_coverage_badge.py coverage.xml coverage-badge.svg`
- [ ] README / docs updated for public API changes
- [ ] Version updated if needed
- [ ] No secrets or unintended large binaries staged
- [ ] Commit message follows convention (Section 9)
- [ ] Performance impact evaluated (if relevant)

If any step fails, DO NOT propose commit.

---

## 9. Commit Message Pattern

Format:
```
<type>(scope): concise summary

Optional body explaining rationale, benchmarks, or docs
Refs: issue numbers, benchmarks, links
```

Types:
- feat
- fix
- perf
- docs
- test
- chore
- build
- ci

Example:
```
feat(indices): add Enhanced Vegetation Index (EVI)

Implements EVI with 1D/2D variants. Adds tests and docs.
Benchmarked at 1.28x speed vs NumPy baseline.
Refs: #45
```

---

## 10. When NOT to Suggest Code

AI agents should refrain if:
- Required context (file content) is missing
- Task involves strategic redesign (defer to human proposal)
- Operation would introduce external network access
- Performance trade-offs cannot be reasonably estimated
- Security posture could be affected (e.g., unsafe / dynamic execution)

Instead, propose clarification questions or planning steps.

---

## 11. Testing Guidance

Minimal test template:
```python
def test_new_index_basic():
    import numpy as np
    from eo_processor import new_index
    a = np.array([0.6, 0.7, 0.8])
    b = np.array([0.2, 0.3, 0.4])
    out = new_index(a, b)
    assert out.shape == a.shape
    assert np.isfinite(out).all()
```

Edge case tests:
- Zeros / near-zero denominators
- Matching shape enforcement
- Large arrays performance (sanity only; avoid huge runtime)

---

## 12. Coverage Badge Regeneration

Trigger conditions:
- Python code modified
- Tests added/removed
- Coverage threshold changes

Sequence:
1. `tox -e coverage`
2. Confirm `coverage.xml` exists
3. Run script
4. Verify SVG text alignment locally
5. Stage updated badge

---

## 13. Documentation Updates

For ANY new feature, public API change, performance model change, or behavior adjustment you MUST keep ALL documentation layers in sync before proposing a commit or PR.

Minimum required updates (if applicable):
- README:
  - Add / update function entry in the API summary table
  - Add or adjust usage snippet
  - Include the mathematical formula for new spectral / temporal indices
  - Note expected input range (e.g. reflectance 0–1) and typical output interpretation
- QUICKSTART:
  - Only update if the change materially helps onboarding (avoid bloating the quick start)
- Sphinx Docs (docs/):
  - Add/remove function in `docs/source/api/functions.rst` autosummary list
  - Update `docs/source/api/architecture.rst` if:
    - Parallelism strategy, thresholds, or internal design changed
    - Numerical stability approach (EPSILON handling, dtype coercion) changed
    - New module or category was introduced
  - Rebuild locally: `make docs` (or `tox -e docs`) and open `docs/build/html/index.html`
  - Ensure new function page is generated (autosummary) and cross-links resolve
- Python Wrapper:
  - Add/expand docstring in `python/eo_processor/__init__.py`
  - Keep docstring style NumPy/Google compatible (Napoleon parses it)
- Type Stubs:
  - Update `python/eo_processor/__init__.pyi` with precise signature (no broad *args unless required)
- Rust Source:
  - Add `///` doc comments (formula, parameter notes, numeric stability, shape expectations)
- Version Synchronization:
  - Ensure `__version__` (Python), `pyproject.toml`, and any badges or docs display agree
  - Bump version (SemVer) if a public API was added (minor) or changed incompatibly (major)
- CHANGELOG (if maintained): succinct entry describing addition / change

Performance / architecture refactor:
- Add a short subsection under "Architecture & Performance Model" summarizing:
  - What changed & why (e.g. fused loops, lowered parallel threshold)
  - Benchmark (array shape, old vs new timings, speedup)
  - Numerical equivalence tolerance
- Remove or adjust obsolete claims in existing text

Read the Docs considerations:
- If new doc dependencies added: update `[project.optional-dependencies].docs` and `docs/requirements.txt`
- If real extension build is newly required on RTD: adjust `.readthedocs.yaml`
- Validate that mocked import strategy (`autodoc_mock_imports`) still works when extension absent

Validation checklist (run locally before commit):
1. `make docs` succeeds without new warnings (or `tox -e docs`)
2. New/changed function appears in:
   - README API summary
   - Sphinx functions autosummary
3. Architecture page still accurate (no stale performance statements)
4. All internal links resolve (scan Sphinx build output)
5. No orphaned removed functions remain in autosummary or README
6. Version numbers consistent everywhere
7. If coverage changed due to added tests, badge regenerated

Prohibited:
- Merging feature code without doc updates
- Leaving placeholder “TODO” lines in published docs
- Claiming performance gains without benchmark snippet

Failing to meet documentation requirements = BLOCKED COMMIT (fix before proceeding).

---

## 14. Performance Claim Template

If suggesting optimization:
```
Benchmark:
Array size: 5000 x 5000
Old: 1.42s
New: 1.05s
Speedup: 1.35x

Methodology:
Single run, warm cache discarded
Python timing via time.time()
```

---

## 15. Do / Don’t Summary

DO:
- Maintain parity between Rust and Python exports
- Use explicit naming (`index_1d`, `index_2d`)
- Keep code minimal and safe
- Suggest incremental changes

DON’T:
- Combine unrelated changes in one diff
- Introduce unstable dependencies
- Modify unrelated files casually
- Reduce test coverage

---

## 16. Rollback Strategy (If Committed Mistake Detected)

1. Reset local changes: `git reset --hard HEAD~1` (if safe)
2. Revert remote commit: create a `revert:` commit
3. Re-run full checklist
4. Open issue documenting root cause if systemic

---

## 17. Example Rust Function Pattern (Reusable)

```rust
#[pyfunction]
fn example_index<'py>(
    py: Python<'py>,
    a: PyReadonlyArray2<f64>,
    b: PyReadonlyArray2<f64>,
) -> PyResult<&'py PyArray2<f64>> {
    let a_arr = a.as_array();
    let b_arr = b.as_array();
    assert_eq!(a_arr.shape(), b_arr.shape(), "Input arrays must have matching shapes");
    let numerator = &a_arr - &b_arr;
    let denominator = &a_arr + &b_arr + 1e-10;
    let out = numerator / denominator;
    Ok(out.into_pyarray(py))
}
```

---

## 18. Red Flags (Require Human Review)

- Proposal to add GPU or distributed code paths
- Changes to module initialization signature
- Introduction of concurrency primitives (threads / async)
- Large memory allocations (> several GB)
- Breaking API removals

---

## 19. Optional Extensions (Future Suggestions)

Agents may propose (but not implement without approval):
- Additional indices: EVI, SAVI, NBR, GCI
- Sliding window operations
- Batch spectral composite builder
- Multi-threaded internal tiling (must justify)

---

## 20. Final Rule

If uncertain about spec or impact → STOP and generate a structured clarification prompt rather than guessing.

Quality > speed. Safety > novelty.

---

End of GitHub Copilot / AI Agent Instructions.
