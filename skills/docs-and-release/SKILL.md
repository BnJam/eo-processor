---
name: docs-and-release
description: Keep documentation, versioning, and release artifacts consistent for eo-processor. Use when changing public APIs, adding features, updating performance claims, or preparing a release (align README/QUICKSTART/docs, stubs, changelog, and pyproject/Cargo versions).
license: Proprietary. See repository LICENSE
compatibility: Designed for this repository’s hybrid Rust+PyO3+Python package. Assumes ability to build/test locally and update markdown/Sphinx docs when present.
metadata:
  author: eo-processor maintainers
  version: "1.0"
---

# Docs & release consistency skill (eo-processor)

Use this skill to prevent the most common “shipped-but-confusing” failures:
- docs don’t match behavior
- types don’t match runtime
- versioning doesn’t reflect the change
- examples don’t run
- performance claims are unverified or outdated

This repo is hybrid (Rust + PyO3 + Python packaging), so **documentation is part of the API**. Treat it with the same rigor as code.

---

## When to activate

Activate this skill when you:
- add or change a public function
- modify function semantics (NaN behavior, epsilon rules, dtype handling, shape rules)
- update performance characteristics or parallelism behavior
- modify packaging/build metadata (pyproject, Cargo, maturin config)
- prepare a release (or anything that looks like one: version bump, changelog update, docs refresh)

Do *not* activate for purely internal refactors that produce no user-visible change—unless those refactors affect performance or behavior indirectly.

---

## Core invariants (must always hold)

1. **Public API is coherent across layers**
   - Rust export (PyO3) ↔ Python re-export ↔ typing stubs ↔ docs ↔ tests

2. **Examples are truthful**
   - Code snippets in `README.md` / `QUICKSTART.md` should be runnable with minimal edits.
   - Avoid pseudo-code that suggests features that do not exist.

3. **Versioning matches semantics**
   - Patch: bug fix, no API expansion
   - Minor: new backward-compatible features
   - Major: breaking changes (signature or contract changes)

4. **Performance claims are substantiated**
   - No “faster” claims without before/after numbers and workload context.

---

## Step 1: Classify your change (to decide what to update)

Choose the most specific category:

### A) Public API addition (new function)
You must update:
- Python exports (`__init__.py`) + `__all__` (if used)
- typing stubs (`__init__.pyi`)
- `README.md` API summary (and/or index listing sections)
- tests (at least correctness + one edge case)
- docs (if repo has `docs/` API listings)

Often update:
- `QUICKSTART.md` if it improves onboarding meaningfully (avoid bloat)
- `CHANGELOG` entry (if repo uses it for releases)

### B) Public API behavior change (contract change)
You must update:
- docs describing behavior (formula, ranges, NaN/Inf handling, epsilon)
- tests to encode the new contract (and prevent regression)
- typing stubs if signature/dtype changes

You must consider:
- semver implications (minor vs major depending on breakage)
- compatibility strategies (deprecation path, alias, warnings)

### C) Bug fix
You must update:
- tests (regression test that fails before fix)
- docs only if user-facing behavior is clarified or corrected

Usually:
- patch version bump (unless fix introduces new feature)
- changelog entry if repo practice expects it

### D) Docs-only update
You must ensure:
- examples remain correct
- no stale references to removed/renamed functions
- claims are accurate (especially performance and supported environments)

---

## Step 2: Documentation update checklist (minimum set)

### README alignment
Update `README.md` when:
- a public function is added/removed/renamed
- formulas or semantics change (e.g., epsilon, masking rules)
- performance story changes

Ensure:
- function name matches runtime import path (`from eo_processor import ...`)
- example inputs match expected dtype/range conventions
- outputs and interpretation notes are consistent (especially for indices)

### Quickstart discipline
Update `QUICKSTART.md` only if:
- it materially reduces time-to-first-success for a new user
- it adds a simple, representative example (not a full tutorial)

Avoid:
- duplicating large sections from README
- adding niche workflows that belong in `WORKFLOWS.md` or `docs/`

### Workflows and examples
Update `WORKFLOWS.md` / `examples/` when:
- a new feature unlocks a compelling end-to-end workflow
- behavior changes would break published workflows

Rule:
- workflows can be more narrative; README should stay concise.

### Longer-form docs (`docs/`)
If this repo has generated docs (e.g., Sphinx):
- update API listings (autosummary pages, function lists)
- ensure cross-references resolve
- keep architecture/performance pages consistent with current implementation

If docs build is part of CI, treat docs build failures as release blockers.

---

## Step 3: Release readiness checklist (versioning + changelog + artifacts)

### Version bump rules
Decide the version bump based on user-visible impact:
- Patch:
  - bug fixes
  - doc corrections
  - internal refactor with no semantic change
- Minor:
  - new function(s)
  - optional parameters added with backward-compatible defaults
  - performance improvements that do not change semantics
- Major:
  - function removed
  - signature changes that break callers
  - semantic/contract changes that break expectations (dtype/range/NaN behavior)

### Keep version sources consistent
In a hybrid Rust+Python repo, version may exist in multiple places (commonly):
- `pyproject.toml` (Python package version)
- Rust crate metadata (`Cargo.toml`)
- optionally a Python `__version__` constant (if present)

The invariant is: **the version users see matches the release tag and changelog**.

If the repo publishes to PyPI, ensure the Python package version is authoritative.

### Changelog discipline
If the repo maintains a `CHANGELOG`:
- add an entry for user-visible changes
- write in terms of user impact (what changed, how to use it, any migration notes)
- include performance notes only when benchmarked and reproducible

---

## Step 4: Performance documentation rules

Any time you modify a hot path or claim performance changes:
- include benchmark context:
  - function(s)
  - array shapes
  - dtype(s)
  - build mode (release)
  - thread settings (if applicable)
  - before/after numbers (median of multiple trials)

Do not:
- claim “X% faster” based on a single run
- compare debug vs release
- compare different thread counts or different BLAS settings without acknowledging it

If benchmarks are stored in repo artifacts (e.g., `dist_bench.md`, `benchmark-*.json`), update them only if you actually re-ran them.

---

## Step 5: Consistency sweep (the “find all references” mental pass)

Before considering the work done:
- search for old function names and update them in:
  - README
  - quickstart
  - docs
  - examples
  - tests
  - type stubs
- ensure any added function is included in:
  - Python exports
  - typing stubs
  - docs API list

Also verify:
- no documentation references non-existent CLI flags or modules
- formulas match the implementation (including any epsilon or masking)
- input scaling guidance is explicit for EO data (reflectance vs scaled integers)

---

## Output format: “Docs & release summary”

When you use this skill, provide a short summary:

1. **User-visible changes**
2. **Docs updated** (list files)
3. **Versioning decision** (patch/minor/major and why)
4. **Changelog entry** (what you added)
5. **Examples/workflows impacted**
6. **Verification plan** (what you will run)

Example:

- User-visible changes: Added `ndmi(a, b)` index and documented expected scaling.
- Docs updated: `README.md`, `docs/...` (API list), `QUICKSTART.md` (optional).
- Version: Minor (new backward-compatible function).
- Changelog: Added “New: NDMI index”.
- Examples: Updated `examples/...` to include NDMI usage.
- Verification: build extension, run pytest for new tests, run lint/type checks, optionally run docs build.

---

## Definition of done

You’re done when:
- [ ] Public API exports, typing stubs, docs, and tests all agree
- [ ] Version bump (if any) matches semver implications
- [ ] Changelog reflects user-visible change (if repo uses it)
- [ ] Published examples are correct and consistent
- [ ] No unverified performance claims remain in docs
- [ ] No references to removed/renamed functions remain

---

## Local references (repo)

- Engineering policies and mandatory checklist: `AGENTS.md`
- User docs: `README.md`, `QUICKSTART.md`
- Workflows: `WORKFLOWS.md`
- Docs (if present): `docs/`
- Packaging: `pyproject.toml`, `Cargo.toml`
- Python API surface and typing: `python/eo_processor/`
- Tests: `tests/`
