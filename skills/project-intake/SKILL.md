---
name: project-intake
description: Build a fast, accurate mental model of the eo-processor repo and the user’s task. Use when starting work, when requirements are unclear, or before making multi-file changes (Rust+PyO3+Python packaging/tests/docs).
license: Proprietary. See repository LICENSE
compatibility: Designed for agentic engineering in this repository; assumes local filesystem access and ability to run Rust/Python tooling.
metadata:
  author: eo-processor maintainers
  version: "1.0"
---

# Project intake & repo map (eo-processor)

Use this skill to (1) clarify what you’re building, (2) map the repository, and (3) choose the smallest safe change set.

## Goals

1. Produce a **repo map**: key directories, entrypoints, and “where to change what”.
2. Translate the user request into **concrete acceptance criteria**.
3. Identify **risk areas** (performance, numerical stability, API surface, safety) and a **verification plan**.

Non-goals:
- Don’t start coding until you can point to the exact files you’ll touch and why.
- Don’t broaden scope. If the request implies refactors, propose them separately.

---

## 1) Clarify the task (requirements triage)

Write down:
- **User intent**: What outcome do they want? (feature, bug fix, perf, docs)
- **Inputs/outputs**: data shapes (1D/2D), dtypes, expected ranges, NaN handling.
- **API surface**: new public function? change existing behavior? CLI change?
- **Constraints**: performance target, memory limits, backwards compatibility.
- **Examples**: at least one concrete example call and expected result properties.

### Minimal clarifying questions (ask only if blocked)
Ask *at most 3* questions, only when needed to proceed safely:
1. “Is this intended to be a new public API or internal refactor?”
2. “What shapes/dtypes should be supported (1D/2D, float32/float64)?”
3. “Any expected behavior for nodata/NaNs or divide-by-zero?”

If you can infer defaults from existing patterns in the repo, proceed without asking.

---

## 2) Repository map (what’s where)

Build a short map like this (fill with repo-specific paths):

### Core implementation
- `src/` — Rust crate implementing fast EO ops via PyO3 (native extension).
- `Cargo.toml` — Rust crate config and dependencies.
- `python/eo_processor/` — Python package wrapper: exports, docstrings, typing stubs.
- `pyproject.toml` / `uv.lock` — Python packaging + dependency lock.

### Tests & quality
- `tests/` — Python tests (pytest).
- `tox.ini` / `pytest.ini` — test environments and config.
- Lint/type tooling likely defined in `pyproject.toml` (ruff/mypy) and Rust (`clippy`, `fmt`).

### Docs & user guides
- `README.md` — primary API & examples.
- `QUICKSTART.md` — onboarding quickstart.
- `WORKFLOWS.md` — complex usage examples / pipelines.
- `docs/` — longer-form docs (if present; check for Sphinx/MD pipeline).

### Scripts / maintenance
- `scripts/` — maintenance scripts (e.g., coverage badge generation).

---

## 3) Change classification (choose the correct workflow)

Pick exactly one dominant category and follow it:

### A) New function (most common)
- Implement in Rust, expose through PyO3, export in Python, add typing stub, add tests, update docs.

### B) Bug fix
- Reproduce -> minimal fix -> regression test -> confirm no API break.

### C) Performance improvement
- Benchmark before/after, verify correctness, avoid unsafe, document speedup context.

### D) Docs-only / packaging
- Keep changes isolated; ensure examples remain runnable.

---

## 4) Safety/performance checklist (pre-implementation)

Before you change code, write:
- **Numerical stability plan**: EPS usage, division-by-zero behavior, NaNs, clipping rules.
- **Shape validation**: ensure consistent behavior for mismatched shapes.
- **Threading / parallelism**: confirm any parallel approach matches project conventions.
- **No side effects**: core ops should not do file I/O or network I/O.

Common pitfalls in EO numeric code:
- silent `NaN` propagation vs deliberate masking
- dtype promotion surprises (float32 vs float64)
- divide-by-zero in normalized differences
- avoid unnecessary allocations for large rasters

---

## 5) Verification plan (what you will run)

Define a verification plan proportional to risk:

### Always
- Rust formatting and linting
- Python linting and typing (when Python layer touched)
- Unit tests relevant to changed behavior

### When touching Rust core
- `cargo fmt`
- `cargo clippy` (treat warnings as errors)
- `cargo test` or at least `cargo check`

### When touching Python API/tests
- run targeted `pytest` for affected test module(s)
- run `tox` env used by the project (pick the most current default)

### When coverage impacted
- run coverage job and regenerate badge if this repo mandates it

---

## 6) Output format: the “Intake Summary” you should produce

When this skill is used, produce a short “Intake Summary” before coding:

1. **Problem statement**
2. **Proposed approach**
3. **Files likely to change** (list exact paths)
4. **Acceptance criteria** (bullet list)
5. **Test/verification plan**
6. **Risks & mitigations**

Example template:

- Problem: …
- Approach: …
- Files:
  - `src/...`
  - `python/eo_processor/...`
  - `tests/...`
  - `README.md`
- Acceptance criteria:
  - …
- Verification:
  - …
- Risks:
  - …

---

## 7) Heuristics for good agent behavior in this repo

- Prefer small, reviewable commits.
- Preserve public API stability.
- Keep Rust+Python exports/stubs/docs synchronized.
- Don’t claim performance improvements without benchmarks.
- Don’t introduce new dependencies unless clearly justified.

---

## References (local)

- Repo operations guide: `AGENTS.md`
- User-facing docs: `README.md`, `QUICKSTART.md`
- Workflows: `WORKFLOWS.md`
