#!/usr/bin/env python3
"""
eo_cli.py - Command-line interface for computing Earth Observation spectral indices
using the eo-processor Rust-accelerated library.

Features:
  - Supports single or multiple index computations in one invocation.
  - Reads input bands from .npy files (NumPy arrays).
  - Automatically coerces input dtypes to float64 internally (as per library behavior).
  - Can output results as .npy (default) or optionally save a simple 8-bit PNG preview.
  - Supports cloud / invalid data masking (applies NaNs prior to computation).
  - Provides delta (change detection) indices: delta_ndvi, delta_nbr.

Supported Indices & Required Inputs:
  normalized_difference: a, b
  ndvi: nir, red
  ndwi: green, nir
  evi: nir, red, blue
  savi: nir, red (optionally --savi-l)
  nbr: nir, swir2
  ndmi: nir, swir1
  nbr2: swir1, swir2
  gci: nir, green
  delta_ndvi: pre_nir, pre_red, post_nir, post_red
  delta_nbr: pre_nir, pre_swir2, post_nir, post_swir2

Example:
  python scripts/eo_cli.py \
    --index ndvi \
    --nir data/nir.npy \
    --red data/red.npy \
    --out outputs/ndvi.npy

Multiple indices:
  python scripts/eo_cli.py \
    --index ndvi ndwi savi \
    --nir data/nir.npy \
    --red data/red.npy \
    --green data/green.npy \
    --out-dir outputs/

Delta (change detection):
  python scripts/eo_cli.py \
    --index delta_ndvi \
    --pre-nir pre/nir.npy --pre-red pre/red.npy \
    --post-nir post/nir.npy --post-red post/red.npy \
    --out outputs/delta_ndvi.npy

Cloud mask (0 = masked, 1 = valid):
  python scripts/eo_cli.py \
    --index ndvi \
    --nir data/nir.npy --red data/red.npy \
    --mask data/cloudmask.npy \
    --out outputs/ndvi_masked.npy

PNG preview (scaled 0–255):
  python scripts/eo_cli.py \
    --index ndvi \
    --nir data/nir.npy --red data/red.npy \
    --out outputs/ndvi.npy \
    --png-preview outputs/ndvi.png

Exit Codes:
  0 success
  1 argument / I/O error
  2 computation error

"""

from __future__ import annotations

import argparse
import sys
import os
import numpy as np
from typing import Dict, Callable, List, Optional

try:
    from eo_processor import (
        normalized_difference,
        ndvi,
        ndwi,
        evi,
        savi,
        nbr,
        ndmi,
        nbr2,
        gci,
        delta_ndvi,
        delta_nbr,
    )
except ImportError as exc:
    print(f"[ERROR] Failed to import eo_processor: {exc}", file=sys.stderr)
    sys.exit(1)


# ---------------------------------------------------------------------------
# Index Specification
# ---------------------------------------------------------------------------

IndexFunc = Callable[..., np.ndarray]

INDEX_SPECS: Dict[str, Dict[str, object]] = {
    "normalized_difference": {
        "func": normalized_difference,
        "bands": ["a", "b"],
        "description": "(a - b)/(a + b)",
    },
    "ndvi": {
        "func": ndvi,
        "bands": ["nir", "red"],
        "description": "(NIR - Red)/(NIR + Red)",
    },
    "ndwi": {
        "func": ndwi,
        "bands": ["green", "nir"],
        "description": "(Green - NIR)/(Green + NIR)",
    },
    "evi": {
        "func": evi,
        "bands": ["nir", "red", "blue"],
        "description": "G*(NIR-Red)/(NIR+C1*Red-C2*Blue+L)",
    },
    "savi": {
        "func": savi,
        "bands": ["nir", "red"],
        "description": "(NIR - Red)/(NIR + Red + L) * (1+L)",
    },
    "nbr": {
        "func": nbr,
        "bands": ["nir", "swir2"],
        "description": "(NIR - SWIR2)/(NIR + SWIR2)",
    },
    "ndmi": {
        "func": ndmi,
        "bands": ["nir", "swir1"],
        "description": "(NIR - SWIR1)/(NIR + SWIR1)",
    },
    "nbr2": {
        "func": nbr2,
        "bands": ["swir1", "swir2"],
        "description": "(SWIR1 - SWIR2)/(SWIR1 + SWIR2)",
    },
    "gci": {
        "func": gci,
        "bands": ["nir", "green"],
        "description": "(NIR/Green) - 1",
    },
    "delta_ndvi": {
        "func": delta_ndvi,
        "bands": ["pre_nir", "pre_red", "post_nir", "post_red"],
        "description": "NDVI(pre) - NDVI(post)",
    },
    "delta_nbr": {
        "func": delta_nbr,
        "bands": ["pre_nir", "pre_swir2", "post_nir", "post_swir2"],
        "description": "NBR(pre) - NBR(post)",
    },
}


# ---------------------------------------------------------------------------
# Argument Parsing
# ---------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="eo_cli.py",
        description="Compute EO spectral indices from .npy band files."
    )
    p.add_argument(
        "--index",
        nargs="+",
        required=True,
        help="One or more indices to compute. See script header for supported names."
    )

    # Common band arguments
    p.add_argument("--nir")
    p.add_argument("--red")
    p.add_argument("--green")
    p.add_argument("--blue")
    p.add_argument("--swir1")
    p.add_argument("--swir2")
    p.add_argument("--a")
    p.add_argument("--b")

    # Delta / change detection band arguments
    p.add_argument("--pre-nir")
    p.add_argument("--pre-red")
    p.add_argument("--post-nir")
    p.add_argument("--post-red")
    p.add_argument("--pre-swir2")
    p.add_argument("--post-swir2")

    # SAVI parameter
    p.add_argument("--savi-l", type=float, default=0.5, help="Soil brightness factor L for SAVI (default 0.5).")

    # Mask
    p.add_argument(
        "--mask",
        help="Optional .npy mask file (same shape). Values of 0 become NaN in inputs before computation."
    )

    # Output control
    p.add_argument("--out", help="Output file path if computing a single index.")
    p.add_argument("--out-dir", help="Directory for multiple index outputs (auto-named <index>.npy).")
    p.add_argument("--png-preview", help="Optional PNG preview path (only valid for single index).")
    p.add_argument(
        "--dtype",
        default="float32",
        choices=["float32", "float64"],
        help="Output dtype for saved .npy files (default float32)."
    )
    p.add_argument(
        "--clamp",
        nargs=2,
        type=float,
        metavar=("MIN", "MAX"),
        help="Clamp output before saving (applied prior to dtype conversion)."
    )
    p.add_argument(
        "--allow-missing",
        action="store_true",
        help="Skip indices missing required bands instead of failing."
    )
    p.add_argument(
        "--list",
        action="store_true",
        help="List supported indices and exit."
    )
    return p


# ---------------------------------------------------------------------------
# I/O Helpers
# ---------------------------------------------------------------------------

def load_npy(path: str) -> np.ndarray:
    if not path:
        raise ValueError("Empty path provided.")
    if not os.path.isfile(path):
        raise FileNotFoundError(path)
    arr = np.load(path)
    if arr.ndim not in (1, 2):
        raise ValueError(f"Only 1D or 2D arrays supported. Got shape {arr.shape} for {path}")
    return arr


def apply_mask(arr: np.ndarray, mask: np.ndarray) -> np.ndarray:
    if arr.shape != mask.shape:
        raise ValueError(f"Mask shape {mask.shape} does not match array shape {arr.shape}")
    return np.where(mask == 0, np.nan, arr)


def save_npy(path: str, arr: np.ndarray, dtype: str, clamp: Optional[List[float]]):
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    out = arr
    if clamp:
        lo, hi = clamp
        out = np.clip(out, lo, hi)
    out = out.astype(dtype)
    np.save(path, out)


def save_png(path: str, arr: np.ndarray, clamp: Optional[List[float]] = None):
    """
    Save a quicklook PNG (requires pillow). Scales finite data to 0–255.
    """
    try:
        from PIL import Image
    except ImportError:
        print("[WARN] Pillow not installed; skipping PNG preview.", file=sys.stderr)
        return
    data = np.asarray(arr)
    finite = data[np.isfinite(data)]
    if finite.size == 0:
        print("[WARN] All values are NaN; PNG skipped.", file=sys.stderr)
        return
    if clamp:
        lo, hi = clamp
        data = np.clip(data, lo, hi)
        finite = data[np.isfinite(data)]
    mn, mx = float(finite.min()), float(finite.max())
    if mx == mn:
        scaled = np.zeros_like(data, dtype=np.uint8)
    else:
        scaled = (255 * (data - mn) / (mx - mn)).astype(np.uint8)
    img = Image.fromarray(scaled)
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    img.save(path)


# ---------------------------------------------------------------------------
# Index Computation Dispatcher
# ---------------------------------------------------------------------------

def compute_index(name: str, bands: Dict[str, np.ndarray], savi_l: float) -> np.ndarray:
    spec = INDEX_SPECS[name]
    func = spec["func"]  # type: ignore

    # Build arguments based on index
    if name == "normalized_difference":
        return func(bands["a"], bands["b"])
    if name == "ndvi":
        return func(bands["nir"], bands["red"])
    if name == "ndwi":
        return func(bands["green"], bands["nir"])
    if name == "evi":
        return func(bands["nir"], bands["red"], bands["blue"])
    if name == "savi":
        return func(bands["nir"], bands["red"], L=savi_l)
    if name == "nbr":
        return func(bands["nir"], bands["swir2"])
    if name == "ndmi":
        return func(bands["nir"], bands["swir1"])
    if name == "nbr2":
        return func(bands["swir1"], bands["swir2"])
    if name == "gci":
        return func(bands["nir"], bands["green"])
    if name == "delta_ndvi":
        return func(
            bands["pre_nir"], bands["pre_red"],
            bands["post_nir"], bands["post_red"]
        )
    if name == "delta_nbr":
        return func(
            bands["pre_nir"], bands["pre_swir2"],
            bands["post_nir"], bands["post_swir2"]
        )
    raise ValueError(f"Unhandled index: {name}")


# ---------------------------------------------------------------------------
# Main Workflow
# ---------------------------------------------------------------------------

def main(argv: Optional[List[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.list:
        print("Supported indices:")
        for k, v in INDEX_SPECS.items():
            print(f"  {k:15} {v['description']}")
        return 0

    indices = args.index
    multi = len(indices) > 1

    if not multi and not args.out:
        parser.error("--out required when computing a single index")
    if multi and not args.out_dir:
        parser.error("--out-dir required when computing multiple indices")

    # Load mask if provided
    mask_arr = load_npy(args.mask) if args.mask else None

    # Load all band paths supplied
    band_paths = {
        "nir": args.nir,
        "red": args.red,
        "green": args.green,
        "blue": args.blue,
        "swir1": args.swir1,
        "swir2": args.swir2,
        "a": args.a,
        "b": args.b,
        "pre_nir": args.pre_nir,
        "pre_red": args.pre_red,
        "post_nir": args.post_nir,
        "post_red": args.post_red,
        "pre_swir2": args.pre_swir2,
        "post_swir2": args.post_swir2,
    }

    loaded_bands: Dict[str, np.ndarray] = {}

    # Lazy load only needed bands
    required_all = set()
    for idx in indices:
        spec = INDEX_SPECS.get(idx)
        if not spec:
            print(f"[ERROR] Unsupported index: {idx}", file=sys.stderr)
            return 1
        required_all.update(spec["bands"])  # type: ignore

    for band in required_all:
        path = band_paths.get(band)
        if not path:
            msg = f"Missing required band '{band}' for indices {indices}"
            if args.allow_missing:
                print(f"[WARN] {msg} (skipping affected indices)")
            else:
                print(f"[ERROR] {msg}", file=sys.stderr)
                return 1
        else:
            try:
                arr = load_npy(path)
                if mask_arr is not None:
                    arr = apply_mask(arr, mask_arr)
                loaded_bands[band] = arr
            except Exception as exc:
                print(f"[ERROR] Failed loading band '{band}' from {path}: {exc}", file=sys.stderr)
                return 1

    results: Dict[str, np.ndarray] = {}
    for idx in indices:
        spec = INDEX_SPECS[idx]
        needed = spec["bands"]  # type: ignore
        if any(b not in loaded_bands for b in needed):
            if args.allow_missing:
                print(f"[INFO] Skipping {idx} (missing bands)")
                continue
            else:
                print(f"[ERROR] Missing bands for {idx}: {needed}", file=sys.stderr)
                return 1
        try:
            res = compute_index(idx, loaded_bands, args.savi_l)
            results[idx] = res
            print(f"[OK] Computed {idx} shape={res.shape}")
        except Exception as exc:
            print(f"[ERROR] Failed computing {idx}: {exc}", file=sys.stderr)
            return 2

    # Save outputs
    if not results:
        print("[WARN] No results produced.", file=sys.stderr)
        return 0

    if multi:
        out_dir = args.out_dir
        assert out_dir
        os.makedirs(out_dir, exist_ok=True)
        for k, arr in results.items():
            out_path = os.path.join(out_dir, f"{k}.npy")
            save_npy(out_path, arr, args.dtype, args.clamp)
    else:
        # Single index
        arr = next(iter(results.values()))
        save_npy(args.out, arr, args.dtype, args.clamp)
        if args.png_preview:
            save_png(args.png_preview, arr, args.clamp)

    print("[DONE] All requested indices processed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
