"""
XArray & Dask integration examples for eo-processor (extended indices & temporal composites).

Demonstrates:
  - Applying multiple spectral indices (NDVI, NDWI, EVI, SAVI, NBR, NDMI, NBR2, GCI)
  - Passing variable L to SAVI
  - Temporal statistics (mean, std, median composite)
  - Chunked Dask workflows with xr.apply_ufunc and da.map_blocks
  - Multi-band DataSet to compute indices in one pass

Requirements:
    pip install eo-processor[dask]
"""

import numpy as np

try:
    import xarray as xr
    import dask.array as da
    from eo_processor import (
        ndvi,
        ndwi,
        evi,
        savi,
        nbr,
        ndmi,
        nbr2,
        gci,
        temporal_mean,
        temporal_std,
        median,
    )
except ImportError as e:
    print(f"Error: {e}")
    print("\nPlease install the required packages:")
    print("  pip install eo-processor[dask]")
    raise SystemExit(1)


# ---------------------------------------------------------------------------
# Helper utilities
# ---------------------------------------------------------------------------
def _stats(name: str, arr) -> None:
    arr = xr.DataArray(arr) if not isinstance(arr, xr.DataArray) else arr
    finite = arr.where(np.isfinite(arr), drop=True)
    print(
        f"{name:<6} shape={arr.shape} "
        f"min={float(finite.min()):.4f} max={float(finite.max()):.4f} "
        f"mean={float(finite.mean()):.4f}"
    )


# ---------------------------------------------------------------------------
# 1. Basic XArray usage with multiple indices
# ---------------------------------------------------------------------------
def example_xarray_basic():
    print("Example 1: Basic spectral indices on small arrays")
    print("-" * 70)

    h, w = 12, 12
    rng = np.random.default_rng(0)
    nir = xr.DataArray(rng.uniform(0.3, 0.9, size=(h, w)), dims=["y", "x"], name="NIR")
    red = xr.DataArray(rng.uniform(0.05, 0.4, size=(h, w)), dims=["y", "x"], name="Red")
    green = xr.DataArray(rng.uniform(0.1, 0.6, size=(h, w)), dims=["y", "x"], name="Green")
    blue = xr.DataArray(rng.uniform(0.02, 0.25, size=(h, w)), dims=["y", "x"], name="Blue")
    swir1 = xr.DataArray(rng.uniform(0.2, 0.5, size=(h, w)), dims=["y", "x"], name="SWIR1")
    swir2 = xr.DataArray(rng.uniform(0.15, 0.4, size=(h, w)), dims=["y", "x"], name="SWIR2")

    # Compute indices (eager, small arrays)
    indices = {
        "NDVI": ndvi(nir.data, red.data),
        "NDWI": ndwi(green.data, nir.data),
        "EVI": evi(nir.data, red.data, blue.data),
        "SAVI": savi(nir.data, red.data, L=0.5),
        "NBR": nbr(nir.data, swir2.data),
        "NDMI": ndmi(nir.data, swir1.data),
        "NBR2": nbr2(swir1.data, swir2.data),
        "GCI": gci(nir.data, green.data),
    }

    for name, arr in indices.items():
        _stats(name, arr)

    print("Basic indices computed ✔\n")


# ---------------------------------------------------------------------------
# 2. Dask chunked arrays & parallel apply_ufunc
# ---------------------------------------------------------------------------
def example_dask_chunks():
    print("Example 2: Chunked NDVI & SAVI (apply_ufunc parallelized)")
    print("-" * 70)

    rng = np.random.default_rng(1)
    nir = xr.DataArray(
        da.from_array(rng.uniform(0.3, 0.9, size=(2000, 2000)), chunks=(250, 250)),
        dims=["y", "x"],
        name="NIR",
    )
    red = xr.DataArray(
        da.from_array(rng.uniform(0.05, 0.4, size=(2000, 2000)), chunks=(250, 250)),
        dims=["y", "x"],
        name="Red",
    )

    # NDVI via apply_ufunc
    ndvi_da = xr.apply_ufunc(
        ndvi,
        nir,
        red,
        dask="parallelized",
        output_dtypes=[float],
    )

    # SAVI with variable L (use L=0.25)
    savi_da = xr.apply_ufunc(
        savi,
        nir,
        red,
        kwargs={"L": 0.25},
        dask="parallelized",
        output_dtypes=[float],
    )

    print("Lazy NDVI dask graph:", isinstance(ndvi_da.data, da.Array))
    print("Lazy SAVI dask graph:", isinstance(savi_da.data, da.Array))

    ndvi_comp = ndvi_da.compute()
    savi_comp = savi_da.compute()

    _stats("NDVI", ndvi_comp)
    _stats("SAVI", savi_comp)

    print("Chunked NDVI & SAVI computed ✔\n")


# ---------------------------------------------------------------------------
# 3. map_blocks custom function example
# ---------------------------------------------------------------------------
def example_map_blocks():
    print("Example 3: map_blocks for multi-index computation")
    print("-" * 70)

    rng = np.random.default_rng(2)
    nir = da.from_array(rng.uniform(0.3, 0.9, size=(800, 800)), chunks=(200, 200))
    red = da.from_array(rng.uniform(0.05, 0.4, size=(800, 800)), chunks=(200, 200))
    green = da.from_array(rng.uniform(0.1, 0.6, size=(800, 800)), chunks=(200, 200))

    def block_indices(nir_blk, red_blk, green_blk):
        # Return stacked indices per block (channels: NDVI, SAVI)
        ndvi_blk = ndvi(nir_blk, red_blk)
        savi_blk = savi(nir_blk, red_blk, L=0.5)
        gci_blk = gci(nir_blk, green_blk)
        return np.stack([ndvi_blk, savi_blk, gci_blk], axis=0)

    stacked = da.map_blocks(
        block_indices,
        nir,
        red,
        green,
        dtype=np.float64,
        chunks=(3, 200, 200),
    )

    print("Result stacked shape (channels, y, x):", stacked.shape)
    comp = stacked.compute()
    print("Channels:", comp.shape[0], "NDVI mean:", comp[0].mean(), "SAVI mean:", comp[1].mean(), "GCI mean:", comp[2].mean())
    print("map_blocks multi-index computed ✔\n")


# ---------------------------------------------------------------------------
# 4. Time-series (temporal composites)
# ---------------------------------------------------------------------------
def example_temporal_composites():
    print("Example 4: Temporal mean/std/median composite")
    print("-" * 70)

    T, H, W = 10, 256, 256
    rng = np.random.default_rng(3)
    nir_ts = xr.DataArray(
        da.from_array(rng.uniform(0.3, 0.9, size=(T, H, W)), chunks=(-1, 128, 128)),
        dims=["time", "y", "x"],
        name="NIR",
    )
    red_ts = xr.DataArray(
        da.from_array(rng.uniform(0.05, 0.4, size=(T, H, W)), chunks=(-1, 128, 128)),
        dims=["time", "y", "x"],
        name="Red",
    )

    # Compute NDVI time series
    ndvi_ts = xr.apply_ufunc(
        ndvi,
        nir_ts,
        red_ts,
        input_core_dims=[["time"], ["time"]],
        output_core_dims=[["time"]],
        vectorize=True,
        dask="parallelized",
        output_dtypes=[float],
    )

    # Temporal composites
    ndvi_mean = ndvi_ts.mean(dim="time").compute()
    ndvi_std = ndvi_ts.std(dim="time").compute()
    ndvi_median = ndvi_ts.median(dim="time").compute()

    _stats("NDVI_mean", ndvi_mean)
    _stats("NDVI_std", ndvi_std)
    _stats("NDVI_med", ndvi_median)

    print("Temporal composites computed ✔\n")


# ---------------------------------------------------------------------------
# 5. Multi-band DataSet workflow
# ---------------------------------------------------------------------------
def example_multiband_dataset():
    print("Example 5: Multi-band DataSet processing")
    print("-" * 70)

    rng = np.random.default_rng(4)
    H, W = 128, 128
    ds = xr.Dataset(
        {
            "NIR": (("y", "x"), rng.uniform(0.3, 0.9, size=(H, W))),
            "Red": (("y", "x"), rng.uniform(0.05, 0.4, size=(H, W))),
            "Green": (("y", "x"), rng.uniform(0.1, 0.6, size=(H, W))),
            "Blue": (("y", "x"), rng.uniform(0.02, 0.25, size=(H, W))),
            "SWIR1": (("y", "x"), rng.uniform(0.2, 0.5, size=(H, W))),
            "SWIR2": (("y", "x"), rng.uniform(0.15, 0.4, size=(H, W))),
        }
    )

    # Compute indices, add as new DataArrays
    ds["NDVI"] = xr.DataArray(ndvi(ds["NIR"].data, ds["Red"].data), dims=["y", "x"])
    ds["SAVI"] = xr.DataArray(savi(ds["NIR"].data, ds["Red"].data, L=0.5), dims=["y", "x"])
    ds["NBR"] = xr.DataArray(nbr(ds["NIR"].data, ds["SWIR2"].data), dims=["y", "x"])
    ds["NDMI"] = xr.DataArray(ndmi(ds["NIR"].data, ds["SWIR1"].data), dims=["y", "x"])
    ds["GCI"] = xr.DataArray(gci(ds["NIR"].data, ds["Green"].data), dims=["y", "x"])

    print(ds[["NDVI", "SAVI", "NBR", "NDMI", "GCI"]])
    print("Multi-band dataset indices computed ✔\n")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    print("=" * 80)
    print("Extended XArray & Dask Integration Examples")
    print("=" * 80)
    print()

    example_xarray_basic()
    example_dask_chunks()
    example_map_blocks()
    example_temporal_composites()
    example_multiband_dataset()

    print("All extended examples completed successfully!")
