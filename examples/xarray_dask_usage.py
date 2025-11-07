"""
XArray and Dask integration examples for eo-processor.

This script demonstrates how to use the Rust-accelerated UDFs with
XArray DataArrays and Dask for distributed/parallel computation.

Requirements:
    pip install eo-processor[dask]
"""

import numpy as np

try:
    import xarray as xr
    import dask.array as da
    from eo_processor import ndvi
except ImportError as e:
    print(f"Error: {e}")
    print("\nPlease install the required packages:")
    print("  pip install eo-processor[dask]")
    exit(1)


def example_xarray_basic():
    """Basic XArray usage with eo-processor."""
    print("Example 1: Basic XArray usage")
    print("-" * 40)

    # Create sample XArray DataArrays
    nir = xr.DataArray(
        np.random.rand(10, 10) * 0.8 + 0.2,
        dims=["y", "x"],
        coords={"y": np.arange(10), "x": np.arange(10)},
        attrs={"band": "NIR", "units": "reflectance"}
    )

    red = xr.DataArray(
        np.random.rand(10, 10) * 0.4,
        dims=["y", "x"],
        coords={"y": np.arange(10), "x": np.arange(10)},
        attrs={"band": "Red", "units": "reflectance"}
    )

    # Compute NDVI
    ndvi_result = xr.apply_ufunc(
        ndvi,
        nir,
        red,
        dask="parallelized",
        output_dtypes=[float],
    )

    print(f"NIR shape: {nir.shape}")
    print(f"Red shape: {red.shape}")
    print(f"NDVI shape: {ndvi_result.shape}")
    print(f"NDVI mean: {ndvi_result.mean().values:.4f}")
    print()


def example_dask_chunks():
    """Using Dask arrays with chunking for large datasets."""
    print("Example 2: Dask arrays with chunking")
    print("-" * 40)

    # Create large Dask arrays (chunked for parallel processing)
    nir_dask = da.random.random((1000, 1000), chunks=(100, 100)) * 0.8 + 0.2
    red_dask = da.random.random((1000, 1000), chunks=(100, 100)) * 0.4

    # print(f"NIR chunks: {nir_dask.chunk}")
    print(f"Chunk size: {nir_dask.chunksize}")

    # Wrap in XArray for better usability
    nir_xr = xr.DataArray(nir_dask, dims=["y", "x"])
    red_xr = xr.DataArray(red_dask, dims=["y", "x"])

    # Compute NDVI using apply_ufunc (leverages Rust UDF, bypasses GIL)
    ndvi_result = xr.apply_ufunc(
        ndvi,
        nir_xr,
        red_xr,
        dask="parallelized",
        output_dtypes=[float],
    )

    print(f"NDVI is lazy: {isinstance(ndvi_result.data, da.Array)}")
    # print(f"NDVI chunks: {ndvi_result.data.nchunks}")

    # Compute (triggers actual computation)
    ndvi_computed = ndvi_result.compute()
    print(f"NDVI computed shape: {ndvi_computed.shape}")
    print(f"NDVI mean: {ndvi_computed.mean().values:.4f}")
    print()


def example_map_blocks():
    """Using map_blocks for custom processing with Dask."""
    print("Example 3: map_blocks usage")
    print("-" * 40)

    # Create Dask arrays
    nir_dask = da.random.random((500, 500), chunks=(100, 100))
    red_dask = da.random.random((500, 500), chunks=(100, 100))

    # Define a function that uses eo-processor
    def compute_ndvi_block(nir_block, red_block):
        return ndvi(nir_block, red_block)

    # Apply function to blocks (each block processed independently, bypassing GIL)
    ndvi_result = da.map_blocks(
        compute_ndvi_block,
        nir_dask,
        red_dask,
        dtype=np.float64,
    )

    print(f"Result shape: {ndvi_result.shape}")
    # print(f"Result chunks: {ndvi_result.nchunks}")

    # Compute result
    ndvi_computed = ndvi_result.compute()
    print(f"NDVI mean: {ndvi_computed.mean():.4f}")
    print()


def example_multidimensional():
    """Working with multidimensional time-series data."""
    print("Example 4: Time-series processing")
    print("-" * 40)

    # Simulate a time series of satellite imagery
    times = 12  # 12 months
    height, width = 100, 100

    # Create 3D arrays (time, y, x)
    nir_ts = xr.DataArray(
        da.random.random((times, height, width), chunks=(1, 50, 50)) * 0.8 + 0.2,
        dims=["time", "y", "x"],
        coords={"time": np.arange(times), "y": np.arange(height), "x": np.arange(width)},
    )

    red_ts = xr.DataArray(
        da.random.random((times, height, width), chunks=(1, 50, 50)) * 0.4,
        dims=["time", "y", "x"],
        coords={"time": np.arange(times), "y": np.arange(height), "x": np.arange(width)},
    )

    print(f"Time series shape: {nir_ts.shape}")
    print(f"Chunks: {nir_ts.data.chunksize}")

    # Compute NDVI for each time step
    # apply_ufunc automatically handles the time dimension
    ndvi_ts = xr.apply_ufunc(
        ndvi,
        nir_ts,
        red_ts,
        input_core_dims=[["y", "x"], ["y", "x"]],
        output_core_dims=[["y", "x"]],
        vectorize=True,
        dask="parallelized",
        output_dtypes=[float],
        dask_gufunc_kwargs={'allow_rechunk': True},
    )

    print(f"NDVI time series shape: {ndvi_ts.shape}")

    # Compute mean NDVI over space for each time step
    mean_ndvi_over_time = ndvi_ts.mean(dim=["y", "x"]).compute()
    print(f"Mean NDVI over time: {mean_ndvi_over_time.values}")
    print()


if __name__ == "__main__":
    print("=" * 50)
    print("XArray and Dask Integration Examples")
    print("=" * 50)
    print()

    example_xarray_basic()
    example_dask_chunks()
    example_map_blocks()
    example_multidimensional()

    print("All examples completed successfully!")
