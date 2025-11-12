#!/usr/bin/env python3
"""
Examples of using xarray's map_blocks to apply eo-processor UDFs block-wise.

This script compares three approaches for computing NDVI on large arrays:

  - xarray.apply_ufunc (dask parallelized)
  - xarray.map_blocks (block-wise, using xarray's map_blocks)
  - dask.array.map_blocks (block-wise on raw dask arrays)

Each approach is timed for a single compute() call so you can compare wall-clock
times on your machine / cluster.

Notes:
  - Tune `size` and `chunks` to match your environment.
  - This is an example file; it does not run any distributed cluster by itself.
"""

import time
import numpy as np

try:
    import xarray as xr
    import dask.array as da
    from eo_processor import ndvi
except Exception as exc:  # pragma: no cover - example script
    print("Missing dependency or import error:", exc)
    print("Please install the optional dependencies for examples:")
    print("  pip install eo-processor[dask] xarray dask")
    raise

# Helper to report timing and basic stats
def report(name: str, result_array):
    """Print name, time and some basic stats of the computed result."""
    if isinstance(result_array, xr.DataArray):
        arr = result_array
        data = arr.values if not isinstance(arr.data, da.Array) else arr.compute().values
    elif isinstance(result_array, da.Array):
        data = result_array.compute()
    else:
        # assume numpy array
        data = np.asarray(result_array)

    print(f"{name}")
    print("-" * max(20, len(name)))
    print(f"shape: {data.shape}")
    print(f"mean:  {np.nanmean(data):.6f}")
    print(f"min:   {np.nanmin(data):.6f}")
    print(f"max:   {np.nanmax(data):.6f}")
    print()


def example_map_blocks_vs_apply_ufunc():
    """
    Create large (size x size) arrays and compare three compute strategies.
    """
    print("Map Blocks vs apply_ufunc vs dask.map_blocks")
    print("=" * 60)

    # Problem size - tune as needed for your machine
    size = 1000
    # Choose chunk sizes so blocks are reasonably sized for a worker.
    # For 1000x1000 with (250, 250) you'll have 4x4=16 blocks.
    chunk_y = 250
    chunk_x = 250

    print(f"Data size: {size}x{size}, chunks: ({chunk_y}, {chunk_x})")
    print()

    # 1) In-memory numpy arrays (for baseline timing; uses Python/NumPy formula)
    print("Preparing in-memory NumPy arrays...")
    nir_np = np.random.rand(size, size) * 0.8 + 0.2
    red_np = np.random.rand(size, size) * 0.4

    # Baseline: pure NumPy formula (not using Rust ndvi)
    print("Timing: pure NumPy formula")
    start = time.time()
    ndvi_numpy = (nir_np - red_np) / (nir_np + red_np)
    t_numpy = time.time() - start
    print(f"NumPy compute time: {t_numpy:.3f} s")
    report("NumPy baseline", ndvi_numpy)

    # 2) xarray + dask arrays, then xr.apply_ufunc (dask='parallelized')
    print("Preparing xarray + dask arrays (for apply_ufunc)...")
    nir_dask = da.from_array(nir_np)
    red_dask = da.from_array(red_np)

    nir_xr = xr.DataArray(nir_dask, dims=("y", "x"), coords={"y": np.arange(size), "x": np.arange(size)})
    red_xr = xr.DataArray(red_dask, dims=("y", "x"), coords={"y": np.arange(size), "x": np.arange(size)})

    # Use apply_ufunc which can run the underlying Rust UDF in parallel.
    print("Timing: xarray.apply_ufunc (dask='parallelized') ...")
    start = time.time()
    ndvi_xr_ufunc = xr.apply_ufunc(
        ndvi,
        nir_xr,
        red_xr,
        input_core_dims=[["y", "x"], ["y", "x"]],
        output_core_dims=[["y", "x"]],
        dask="parallelized",
        vectorize=False,
        output_dtypes=[float],
        dask_gufunc_kwargs={'allow_rechunk': True}
    )
    # Trigger computation and measure
    ndvi_xr_ufunc_computed = ndvi_xr_ufunc.compute()
    t_apply = time.time() - start
    print(f"apply_ufunc compute time: {t_apply:.3f} s")
    report("xarray.apply_ufunc result", ndvi_xr_ufunc_computed)

    # 3) xarray.map_blocks
    # xarray.map_blocks applies a function block-by-block. The function should accept
    # numpy arrays (blocks) and return a numpy array or xarray DataArray for that block.
    print("Timing: xarray.map_blocks ...")
    # define block function that uses the Rust ndvi on numpy blocks
    def block_ndvi(darr_chunk: xr.DataArray):
        # ds will have 'nir' and 'red' DataArrays for the current block
        # Extract numpy arrays
        nir_block = darr_chunk.data[0]
        red_block = darr_chunk.data[1]
        # Call the Rust-accelerated ndvi function
        res_arr = ndvi(nir_block, red_block)
        # Wrap back to XArray
        return xr.DataArray(res_arr, dims=("y", "x"))

    # Provide a template so xarray knows the shape/dtype of output blocks.
    # template should represent a single-block (chunk) result, with dims y,x.
    template_block = xr.DataArray(
        da.empty((nir_xr.shape[0], nir_xr.shape[1]), dtype=np.float64),
        dims=("y", "x"),
    ).chunk({"y": chunk_y, "x": chunk_x})

    print("Preparing stacked xarray for map_blocks...")


    # Build xarray.DataArray inputs (already defined as nir_xr/red_xr)
    start = time.time()
    stacked_xr = xr.concat(
        [nir_xr, red_xr],
        dim="band",
    )
    stacked_xr = stacked_xr.assign_coords(band=["nir", "red"])
    stacked_xr = stacked_xr.transpose("band", "y", "x")
    # Chunk along band and spatial dimensions
    stacked_xr = stacked_xr.chunk({"band": 2, "y": chunk_y, "x": chunk_x})

    # Print out the stacked_xr structure for clarity
    print(f"Stacked xarray structure:\n{stacked_xr}\n")

    ndvi_xr_mapblocks = stacked_xr.map_blocks(
        block_ndvi,
        template=template_block,
    )
    ndvi_xr_mapblocks_computed = ndvi_xr_mapblocks.compute()
    t_mapblocks = time.time() - start
    print(f"xarray.map_blocks compute time: {t_mapblocks:.3f} s")
    report("xarray.map_blocks result", ndvi_xr_mapblocks_computed)

    # 4) dask.array.map_blocks on the underlying dask arrays
    print("Timing: dask.array.map_blocks ...")
    def dask_block_ndvi(nir_block, red_block):
        # this will be called with numpy arrays per block
        res_arr = ndvi(nir_block, red_block)
        # Wrap back to XArray if needed (here we just return numpy array)
        return xr.DataArray(res_arr, dims=("y", "x"))

    start = time.time()
    ndvi_dask_mapblocks = da.map_blocks(
        dask_block_ndvi,
        nir_dask,
        red_dask,
        dtype=np.float64,
        chunks=(chunk_y, chunk_x),
    )
    ndvi_dask_mapblocks_computed = ndvi_dask_mapblocks.compute()
    t_dask_map = time.time() - start
    print(f"dask.array.map_blocks compute time: {t_dask_map:.3f} s")
    report("dask.array.map_blocks result", ndvi_dask_mapblocks_computed)

    # Quick consistency checks (allow tiny floating differences)
    print("Sanity checks (all_close to NumPy baseline):")
    print("apply_ufunc ~ baseline:", np.allclose(ndvi_xr_ufunc_computed, ndvi_numpy, equal_nan=True, atol=1e-10))
    print("xarray.map_blocks ~ baseline:", np.allclose(ndvi_xr_mapblocks_computed, ndvi_numpy, equal_nan=True, atol=1e-10))
    print("dask.map_blocks ~ baseline:", np.allclose(ndvi_dask_mapblocks_computed, ndvi_numpy, equal_nan=True, atol=1e-10))
    print()

    # Summary timings
    print("Summary timings (seconds)")
    print("-" * 30)
    print(f"NumPy baseline        : {t_numpy:.3f} s")
    print(f"xarray.apply_ufunc    : {t_apply:.3f} s")
    print(f"xarray.map_blocks     : {t_mapblocks:.3f} s")
    print(f"dask.array.map_blocks : {t_dask_map:.3f} s")
    print()

    # Helpful note to user
    print("Notes:")
    print(" - apply_ufunc will try to call the kernel on each chunk in parallel.")
    print(" - map_blocks gives explicit block-by-block control; useful when you have")
    print("   custom per-block logic or want to avoid automatic rechunking.")
    print(" - dask.array.map_blocks works on raw dask arrays and can be simpler/faster")
    print("   when you don't need the xarray metadata machinery.")
    print()


if __name__ == "__main__":
    example_map_blocks_vs_apply_ufunc()
