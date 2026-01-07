# Agent Skills Index (eo-processor)

This directory contains **agentic skills** following the Agent Skills open standard (https://agentskills.io/specification).

Each skill lives in its own folder and includes a required `SKILL.md` file with:
- **Metadata** (`name`, `description`, etc.) used for discovery
- **Instructions** loaded when the skill is activated
- Optional supporting files (e.g., `references/`, `scripts/`, `assets/`)

## Skills in this workspace

### `project-intake/`
**Use when:** starting a new task, requirements are unclear, or before multi-file Rust+Python changes.  
**Focus:** repo map, acceptance criteria, risks, and a minimal verification plan.

### `rust-pyo3-function/`
**Use when:** implementing or modifying Rust compute kernels exposed to Python via PyO3.  
**Focus:** correctness, shape validation, numerical stability, avoiding unnecessary allocations, safe module registration.

### `python-api-surface/`
**Use when:** changing anything users import from `eo_processor`.  
**Focus:** exports (`__init__.py`), `__all__`, docstrings, and keeping `__init__.pyi` type stubs aligned with runtime behavior and docs.

### `test-and-quality-gates/`
**Use when:** verifying changes, fixing CI failures, or before committing.  
**Focus:** scope-driven gates across Rust (`fmt`, `clippy`, build), Python (lint, types, pytest/tox), integration, and coverage/docs checks.

### `benchmark-and-perf/`
**Use when:** making performance changes or claims.  
**Focus:** repeatable benchmark protocol (pinned settings, warmups, trials), correctness checks against a baseline, and result reporting.

### `docs-and-release/`
**Use when:** changing public APIs, behavior/semantics, packaging, or preparing a release.  
**Focus:** docs/typing/tests alignment, semver version decisions, changelog discipline, and truthful performance documentation.

### `examples-and-smoke-tests/`
**Use when:** creating or updating examples under `examples/` for onboarding and smoke testing.  
**Focus:** runnable scripts, minimal dependencies, predictable outputs, and keeping examples aligned with the public API.

## Conventions & tips

- Keep `SKILL.md` bodies concise (progressive disclosure); push deep reference material into `references/` if needed.
- When you add a new public function, assume you’ll update:
  - Rust export + module registration
  - Python exports and `__init__.pyi`
  - tests
  - README/docs
  - (optionally) examples under `examples/`

## Adding a new skill

Create a directory named after the skill:
- lowercase letters, numbers, hyphens
- must match the `name:` field in `SKILL.md`

Then add `SKILL.md` with frontmatter:
- `name` and `description` are required
- optional: `license`, `compatibility`, `metadata`, `allowed-tools`
