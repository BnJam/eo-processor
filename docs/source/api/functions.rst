Functions
=========

This section documents the public EO Processor function API. Pages are manually maintained
(with autosummary generation disabled) to ensure clean sidebar names without the
`eo_processor.` prefix. When adding a new public function:

1. Create `docs/source/api/functions/<name>.rst`
2. Add the page under the appropriate category below
3. Keep docstring + type stubs in sync
4. Update README, version (minor bump for additions), and tests

Dimensionality Conventions
--------------------------
Unless otherwise noted:
- Spectral indices document 1D/2D usage explicitly; higher ranks may work internally but are not
  yet part of the formal contract.
- Temporal reducers (`temporal_mean`, `temporal_std`, `median`, `composite`) support 1D–4D with
  time-first layout.
- Advanced temporal processes (`moving_average_temporal`, `moving_average_temporal_stride`) and
  `pixelwise_transform` operate on 1D–4D time-first arrays.
- Masking utilities accept 1D–4D numeric arrays.
- Distance metrics operate on 2D point matrices `(N, D)` and `(M, D)`.

Categories
----------

Spectral Indices
----------------
.. toctree::
   :maxdepth: 1
   :titlesonly:

   functions/ndvi
   functions/ndwi
   functions/ndmi
   functions/nbr
   functions/nbr2
   functions/savi
   functions/enhanced_vegetation_index
   functions/evi
   functions/gci

Change Detection
----------------
.. toctree::
   :maxdepth: 1
   :titlesonly:

   functions/delta_ndvi
   functions/delta_nbr

Temporal Reducers & Compositing
-------------------------------
.. toctree::
   :maxdepth: 1
   :titlesonly:

   functions/temporal_mean
   functions/temporal_std
   functions/median
   functions/composite
   functions/temporal_sum

Advanced Temporal & Pixelwise Processes
---------------------------------------
.. toctree::
   :maxdepth: 1
   :titlesonly:

   functions/moving_average_temporal
   functions/moving_average_temporal_stride
   functions/pixelwise_transform
   functions/temporal_composite

Masking & Cleanup Utilities
---------------------------
.. toctree::
   :maxdepth: 1
   :titlesonly:

   functions/mask_vals
   functions/mask_invalid
   functions/mask_in_range
   functions/mask_out_range
   functions/mask_scl
   functions/replace_nans

Distance Metrics
----------------
.. toctree::
   :maxdepth: 1
   :titlesonly:

   functions/euclidean_distance
   functions/manhattan_distance
   functions/chebyshev_distance
   functions/minkowski_distance

Core Primitive
--------------
.. toctree::
   :maxdepth: 1
   :titlesonly:

   functions/normalized_difference

Maintenance Notes
-----------------
- All pages should start with the lowercase function name as the top-level heading.
- Avoid embedding `eo_processor.` in headings—use plain names.
- Include: formula (if index), usage example, supported shapes, dtype behavior, numerical stability,
  error handling, related functions.
- For performance claims, follow the project benchmark template (array size, old vs new timing, methodology).
- Keep cross-references (`See Also`) consistent and minimal.

Versioning
----------
Adding a new function requires a minor version bump. Non-breaking internal improvements are patch
bumps. Changing a documented signature or behavior incompatibly is a major bump (requires explicit
approval per repository governance).

Performance & Benchmark Automation
----------------------------------
You can generate reproducible benchmark tables for inclusion in docs using the benchmark script:

.. code-block:: bash

   python scripts/benchmark.py --group all --compare-numpy --loops 5 --warmups 1 \
       --height 2000 --width 2000 --time 24 --points-a 3000 --points-b 3000 --point-dim 64 \
       --md-out benchmarks.md --rst-out docs/source/api/benchmarks.rst --json-out benchmarks.json

Key flags:
- --group / --functions: choose predefined groups or explicit function list.
- --compare-numpy: include baseline timings (speedup > 1 means Rust faster).
- --loops / --warmups: control timing stability.
- --md-out / --rst-out / --json-out: emit Markdown, reStructuredText (Sphinx), and JSON artifacts.
- --height/--width/--time: spatial + temporal dimensions for temporal + spectral + process functions.
- --points-a/--points-b/--point-dim: shapes for distance metrics.
- --minkowski-p: order p for minkowski_distance.
- --ma-window / --ma-stride: window & stride parameters for moving average process benchmarks.

Sphinx Integration:
- Use --rst-out to create a reST grid table you can include with ``.. include:: api/benchmarks.rst``.
- Keep large raw benchmark tables out of individual function pages; instead summarize per-function highlights (already added) and link to the central benchmark report.

Updating Performance Claims (Checklist):
1. Run the script with representative large shapes (≥ millions of elements).
2. Validate correctness via np.allclose (the script does this internally).
3. Copy per-function speedup lines into each function's Performance section only if they are stable.
4. If a function regresses (speedup < 1), document it transparently with guidance.
5. For new optimized functions, add a claim template block following repository guidelines.

Interpreting Results:
- Speedups < 1 indicate NumPy faster for given shape; consider adjusting parallel thresholds or array sizes.
- Fused arithmetic functions (indices) typically show ≥1.3x gains on large tiles.
- Temporal reducers may need tuning; median often benefits most from native parallelism.
- Moving average processes can show >5x vs naive window scans due to O(T) prefix sums.

Re-running benchmarks on different hardware (template):
.. code-block:: bash

   python scripts/benchmark.py --group spectral --compare-numpy --loops 7 --warmups 2 \
       --height 5000 --width 5000 --md-out spectral_bench.md

End of functions index.
