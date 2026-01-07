---
name: benchmark-and-perf
description: Measure, compare, and improve eo-processor performance safely. Use when adding/changing Rust compute kernels, refactoring hot paths, or making performance claims. Provides a repeatable benchmark protocol with correctness checks, profiling hints, and reporting guidelines.
license: Proprietary. See repository LICENSE
compatibility: Assumes a local Rust toolchain and Python environment capable of building the PyO3 extension (maturin). Benchmarking may use repo scripts/artifacts under benchmarking/ and dist_bench*. Network access not required.
metadata:
  author: eo-processor maintainers
  version: "1.0"
---

# Benchmarking & performance skill (eo-processor)

Use this skill to make performance work **repeatable**, **measurable**, and **safe** in a hybrid Rust+PyO3+Python Earth Observation library.

The core objective is to answer, defensibly:
1) Did performance change (time, memory, allocations)?
2) Did correctness change (numerics, dtype/shape contract)?
3) Is the change worth the complexity?

---

## When to activate

Activate this skill when you:
- add a new Rust kernel or change an existing one
- refactor any hot loop / ndarray expression in Rust
- adjust parallelism, chunking, or algorithmic complexity
- change dtype handling (float32/float64) or NaN/Inf behavior (performance often couples to semantics)
- plan to claim speedups in docs/CHANGELOG/PR

Do **not** activate for trivial doc updates or purely cosmetic refactors.

---

## Principles (performance without breaking trust)

1. **Correctness is a gate**: benchmarking without validating outputs is not acceptable.
2. **Measure the right thing**: avoid benchmarking builds that differ in optimization flags; compare like-for-like.
3. **Warm up**: compilation and caching effects can dominate first-run timings.
4. **Use representative shapes**: EO workloads are usually large (e.g., 1k²–10k² rasters), not just toy arrays.
5. **Avoid accidental regressions**: track both speed and memory; faster-but-alloc-heavy can be worse in real pipelines.
6. **No unverified claims**: do not claim performance improvements without before/after numbers and parameters.

---

## Step 1: Define the benchmark question

Write down:
- What operation is being measured? (function name and version)
- What data sizes/shapes? (e.g., 1000x1000, 5000x5000)
- What data distribution? (random uniform, realistic reflectance range, with/without NaNs)
- What environment? (CPU model optional, OS, Python version, Rust release build, number of threads)
- What metric? (wall time, throughput in pixels/s, peak RSS if available)

### Acceptance criteria examples
- “New implementation must be ≥ 1.2× faster than baseline for 5000x5000 float64 arrays”
- “No more than +5% memory overhead compared to baseline”
- “Numerical difference within tolerance (float64: 1e-12; float32: 1e-5)”

---

## Step 2: Always pin build mode & runtime settings

### Build mode
- Benchmark only **release** builds for the Rust extension.
- Ensure you’re not comparing a debug build vs release build.
- If you compare against NumPy, ensure BLAS settings are stable.

### Threading
Performance changes can be dominated by threading differences:
- Pin thread counts consistently across runs.
- If the repo uses Rayon or similar internally, control its thread count via environment (document what you used).
- For NumPy baselines, ensure you understand whether BLAS threads are involved.

### Stability
- Close other heavy processes.
- Prefer running multiple trials and report median (and optionally p95).

---

## Step 3: Correctness check BEFORE timing

Before timing, validate:
- output shape matches contract
- dtype matches contract
- outputs are finite/NaN per contract
- values match baseline within tolerance

### Baseline options
Pick the best available baseline:
- an existing eo-processor implementation (before refactor)
- a pure NumPy reference implementation of the formula
- a known-correct small example with asserted values

If you can’t produce a correctness check, stop and add one first (usually a unit test in `tests/`).

---

## Step 4: Benchmark protocol (repeatable)

Use this protocol for each benchmark you report:

1) **Generate inputs**
   - Use seeded random generation for reproducibility.
   - Use realistic ranges (e.g., reflectance in [0, 1]) unless the function expects something else.

2) **Warm-up**
   - Call the function at least once to warm caches and ensure the extension is loaded.

3) **Time multiple trials**
   - Run N trials (e.g., 5–20).
   - Record the median and a dispersion metric (min/max or p95).

4) **Report**
   - Provide: shape, dtype, trials, median time, throughput, build mode, thread count.

### Example Python timing skeleton (adapt to repo norms)
- Use `time.perf_counter()` not `time.time()`
- Avoid counting array creation time in the timed region
- Ensure outputs are consumed to avoid lazy evaluation traps (especially if using Dask wrappers)

---

## Step 5: Evaluate results (interpretation)

### Speedup thresholds
- < 1.05×: likely noise or not worth complexity unless it also reduces memory or fixes bugs.
- 1.05×–1.2×: consider whether it’s worth it; check memory/allocations and sustained performance.
- ≥ 1.2×: usually meaningful for EO rasters; still verify no semantic drift.

### Check for regressions
- Small arrays sometimes get slower while large arrays get faster. That’s acceptable if the library targets large rasters—just document it.
- Watch for “fast median, slow tail”: if p95 worsened, investigate allocation spikes or scheduling.

### Memory and allocations
If possible, assess:
- intermediate allocations (ndarray expression temporaries are common)
- peak memory usage (large rasters can OOM quickly)

If you can’t measure memory precisely, at least reason about allocations:
- Did you add temporaries?
- Did you switch from in-place fill to multiple intermediate arrays?

---

## Step 6: Profiling & root-cause techniques (use selectively)

Use profiling only when the benchmark indicates a meaningful issue.

### Common performance traps in this repo’s domain
- Extra temporaries from chained ndarray arithmetic
- Unintended dtype conversions / casts
- Poor cache locality due to iteration order
- Branch-heavy inner loops (NaN handling, conditional masking)
- Parallel overhead dominating small inputs

### Suggested methods (choose what applies)
- Add lightweight internal instrumentation (counts, timing spans) temporarily; remove before commit.
- Compare “fused loop” vs “expression-based” implementations.
- Ensure the hottest path is in Rust (not Python glue).

If you can’t profile locally, you can still:
- reduce the workload to isolate which step dominates
- compare variants with minimal changes to infer the cause

---

## Step 7: How to make performance changes safely

Preferred sequence:
1. Implement change with correctness tests.
2. Benchmark before/after with pinned settings.
3. If speedup is real, clean up code and document results.
4. If speedup is not real, revert or open a performance issue with data.

### Safe optimizations (common wins)
- Fuse computations into a single pass over the array data.
- Precompute invariants outside inner loops.
- Reduce allocations: write into a preallocated output array.
- Avoid repeated bounds checks where safe and idiomatic (without `unsafe` unless justified).

### Risky optimizations (require stronger evidence)
- Introducing `unsafe`
- Changing NaN/Inf behavior for speed
- Changing dtype semantics (float32 vs float64)
- Altering parallel thresholds or scheduling without benchmarks on multiple shapes

---

## Reporting template (use in your response/PR)

When you use this skill, report results in this format:

### Benchmark setup
- Function(s): `...`
- Build: release (how built)
- Hardware/OS: (if known)
- Threads: (Rayon/BLAS/Python settings)
- Input: shape(s), dtype(s), distribution (seeded)

### Correctness
- Baseline: (existing impl / NumPy reference)
- Tolerance: (e.g., rtol/atol)
- Result: pass/fail (and any notes about NaN behavior)

### Performance results
For each shape/dtype:
- Before: median X ms (N trials)
- After: median Y ms (N trials)
- Speedup: X/Y (or %)
- Throughput: pixels/s (optional)
- Notes: memory/allocations qualitative notes

### Decision
- Keep / revise / revert
- Follow-ups (tests, docs, perf issue)

---

## Definition of done (for perf work)

You’re done when:
- [ ] Correctness is validated against a baseline
- [ ] Benchmarks are repeatable and documented (inputs + settings)
- [ ] Any performance claim has before/after numbers
- [ ] No unacceptable memory regression is introduced (or it’s explicitly justified)
- [ ] Tests cover at least one representative correctness case and key edge cases
- [ ] The change doesn’t silently alter public semantics (or docs/versioning reflect it)

---

## Local references (repo)

- Engineering rules & quality gates: `AGENTS.md`
- User-facing docs and performance notes: `README.md`, `QUICKSTART.md`
- Complex workflows: `WORKFLOWS.md`
- Benchmark artifacts and harnesses: `benchmarking/`, `dist_bench.json`, `dist_bench.md`, `benchmark-*.json`
- Scripts and maintenance tooling: `scripts/`
- Tests: `tests/`
