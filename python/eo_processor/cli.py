#!/usr/bin/env python3
"""
Command-line interface for computing Earth Observation spectral indices
using eo-processor (Rust-accelerated core).

This mirrors the functionality of the external scripts/eo_cli.py helper but
is packaged so you can invoke via:

    python -m eo_processor.cli --index ndvi --nir nir.npy --red red.npy --out ndvi.npy

or (if console_scripts entry point is later added):

    eo-processor --index ndvi ...

Features:
  - Single or multiple index computation in one invocation
  - Batch delta (change detection) indices (delta_ndvi, delta_nbr)
  - Optional cloud / validity mask (0 = masked, non-zero = keep)
  - Optional range clamping and PNG quicklook generation
  - Supports any numeric NumPy dtype as input (auto-coerced to float64 internally)
  - Safe skipping of missing indices with --allow-missing

Exit Codes:
  0 success
  1 argument / I/O error
  2 computation error

NOTE: This file is intentionally self-contained (no import of scripts/eo_cli.py)
to avoid relying on non-package paths when installed from a wheel.
"""

from __future__ import annotations

import argparse
import os
import sys
from dataclasses import dataclass
from typing import Dict, Iterable, List, Mapping, Optional

import numpy as np

# Import public API (ensures delta indices are available)
from . import (
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


from numpy.typing import NDArray
from typing import Protocol, Any

# Public numeric array type (inputs may be any numeric dtype; outputs coerced to float64)
NumericArray = NDArray[np.float64]


class IndexFunctionProtocol(Protocol):
    def __call__(self, *args: Any, **kwargs: Any) -> NumericArray: ...


IndexFunction = IndexFunctionProtocol


@dataclass(frozen=True)
class IndexSpec:
    name: str
    func: IndexFunction
    required_bands: List[str]
    description: str

    def missing_bands(self, provided: Mapping[str, NumericArray]) -> List[str]:
        return [b for b in self.required_bands if b not in provided]

    # (Method moved into class definition above with annotations)


# Registry of supported indices
INDEX_SPECS: Dict[str, IndexSpec] = {
    "normalized_difference": IndexSpec(
        "normalized_difference",
        normalized_difference,
        ["a", "b"],
        "(a - b) / (a + b)",
    ),
    "ndvi": IndexSpec("ndvi", ndvi, ["nir", "red"], "(NIR - Red)/(NIR + Red)"),
    "ndwi": IndexSpec("ndwi", ndwi, ["green", "nir"], "(Green - NIR)/(Green + NIR)"),
    "evi": IndexSpec(
        "evi",
        evi,
        ["nir", "red", "blue"],
        "2.5*(NIR-Red)/(NIR+6*Red-7.5*Blue+1)",
    ),
    "savi": IndexSpec(
        "savi",
        savi,
        ["nir", "red"],
        "(NIR-Red)/(NIR+Red+L)*(1+L)",
    ),
    "nbr": IndexSpec("nbr", nbr, ["nir", "swir2"], "(NIR - SWIR2)/(NIR + SWIR2)"),
    "ndmi": IndexSpec("ndmi", ndmi, ["nir", "swir1"], "(NIR - SWIR1)/(NIR + SWIR1)"),
    "nbr2": IndexSpec(
        "nbr2", nbr2, ["swir1", "swir2"], "(SWIR1 - SWIR2)/(SWIR1 + SWIR2)"
    ),
    "gci": IndexSpec("gci", gci, ["nir", "green"], "(NIR/Green) - 1"),
    "delta_ndvi": IndexSpec(
        "delta_ndvi",
        delta_ndvi,
        ["pre_nir", "pre_red", "post_nir", "post_red"],
        "NDVI(pre) - NDVI(post)",
    ),
    "delta_nbr": IndexSpec(
        "delta_nbr",
        delta_nbr,
        ["pre_nir", "pre_swir2", "post_nir", "post_swir2"],
        "NBR(pre) - NBR(post)",
    ),
}


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="eo_processor.cli",
        description="Compute Earth Observation spectral indices from .npy band files.",
    )
    p.add_argument(
        "--index",
        nargs="+",
        help="One or more index names. Use --list to see supported indices. (Either --index or --list is required.)",
    )
    # Standard bands
    p.add_argument("--nir")
    p.add_argument("--red")
    p.add_argument("--green")
    p.add_argument("--blue")
    p.add_argument("--swir1")
    p.add_argument("--swir2")
    p.add_argument("--a")
    p.add_argument("--b")
    # Delta bands
    p.add_argument("--pre-nir")
    p.add_argument("--pre-red")
    p.add_argument("--post-nir")
    p.add_argument("--post-red")
    p.add_argument("--pre-swir2")
    p.add_argument("--post-swir2")
    # SAVI
    p.add_argument(
        "--savi-l",
        type=float,
        default=0.5,
        help="Soil brightness factor L for SAVI (default 0.5).",
    )
    # Mask
    p.add_argument(
        "--mask",
        help="Optional .npy mask (0 = masked). Shape must match band arrays.",
    )
    # Output
    p.add_argument(
        "--out",
        help="Output file path when computing a single index.",
    )
    p.add_argument(
        "--out-dir",
        help="Directory to write multiple index outputs (<index>.npy).",
    )
    p.add_argument(
        "--png-preview",
        help="Optional PNG preview path (only valid for single index).",
    )
    p.add_argument(
        "--dtype",
        default="float64",
        choices=["float32", "float64"],
        help="Output dtype for saved arrays (default float64).",
    )
    p.add_argument(
        "--clamp",
        nargs=2,
        type=float,
        metavar=("MIN", "MAX"),
        help="Clamp output range prior to dtype conversion.",
    )
    p.add_argument(
        "--allow-missing",
        action="store_true",
        help="Skip indices missing required bands instead of failing.",
    )
    p.add_argument(
        "--list",
        action="store_true",
        help="List supported indices and exit.",
    )
    return p


def list_indices() -> None:
    print("Supported indices:")
    for spec in INDEX_SPECS.values():
        bands = ",".join(spec.required_bands)
        print(f"  {spec.name:15} {spec.description:35} bands=[{bands}]")


def load_npy(path: str) -> NumericArray:
    if not path:
        raise ValueError("Empty path.")
    if not os.path.isfile(path):
        raise FileNotFoundError(path)
    arr = np.load(path)
    if arr.ndim not in (1, 2):
        raise ValueError(f"Only 1D or 2D arrays supported. Got {arr.shape} for {path}")
    return arr


def apply_mask(arr: NumericArray, mask: NumericArray) -> NumericArray:
    if arr.shape != mask.shape:
        raise ValueError(f"Mask shape {mask.shape} != array shape {arr.shape}")
    return np.where(mask == 0, np.nan, arr)


def save_npy(
    path: str, arr: NumericArray, dtype: str, clamp: Optional[List[float]]
) -> None:
    out_arr = arr
    if clamp:
        lo, hi = clamp
        out_arr = np.clip(out_arr, lo, hi)
    out_arr = out_arr.astype(dtype)
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    np.save(path, out_arr)


def save_png(path: str, arr: NumericArray, clamp: Optional[List[float]]) -> None:
    try:
        from PIL import Image
    except Exception:
        print("[WARN] Pillow not installed; PNG preview skipped.", file=sys.stderr)
        return
    data = np.asarray(arr)
    finite = data[np.isfinite(data)]
    if finite.size == 0:
        print("[WARN] All values NaN; PNG skipped.", file=sys.stderr)
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


def compute(
    spec: IndexSpec, bands: Mapping[str, NumericArray], savi_l: float
) -> NumericArray:
    # Dispatch based on spec.name (explicit to keep signature clarity)
    name = spec.name
    f = spec.func
    if name == "normalized_difference":
        return f(bands["a"], bands["b"])
    if name == "ndvi":
        return f(bands["nir"], bands["red"])
    if name == "ndwi":
        return f(bands["green"], bands["nir"])
    if name == "evi":
        return f(bands["nir"], bands["red"], bands["blue"])
    if name == "savi":
        return f(bands["nir"], bands["red"], L=savi_l)
    if name == "nbr":
        return f(bands["nir"], bands["swir2"])
    if name == "ndmi":
        return f(bands["nir"], bands["swir1"])
    if name == "nbr2":
        return f(bands["swir1"], bands["swir2"])
    if name == "gci":
        return f(bands["nir"], bands["green"])
    if name == "delta_ndvi":
        return f(
            bands["pre_nir"],
            bands["pre_red"],
            bands["post_nir"],
            bands["post_red"],
        )
    if name == "delta_nbr":
        return f(
            bands["pre_nir"],
            bands["pre_swir2"],
            bands["post_nir"],
            bands["post_swir2"],
        )
    raise ValueError(f"Unhandled index: {name}")


def _gather_required_bands(indices: Iterable[str]) -> List[str]:
    needed: List[str] = []
    for idx in indices:
        if idx not in INDEX_SPECS:
            raise KeyError(idx)
        for b in INDEX_SPECS[idx].required_bands:
            if b not in needed:
                needed.append(b)
    return needed


def cli(argv: Optional[List[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.list:
        list_indices()
        return 0

    if not args.index:
        parser.error("either --index or --list is required")

    indices: List[str] = args.index
    multi = len(indices) > 1

    if multi and not args.out_dir:
        parser.error("--out-dir required when computing multiple indices")
    if not multi and not args.out:
        parser.error("--out required when computing a single index")
    if args.png_preview and multi:
        parser.error("--png-preview only valid for a single index")

    # Load mask (optional)
    mask_arr: Optional[NumericArray] = None
    if args.mask:
        try:
            mask_arr = load_npy(args.mask)
        except Exception as exc:
            print(f"[ERROR] Failed loading mask: {exc}", file=sys.stderr)
            return 1

    # Map of band name -> numpy array
    band_path_map = {
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

    try:
        required = _gather_required_bands(indices)
    except KeyError as e:
        print(f"[ERROR] Unsupported index '{e.args[0]}'", file=sys.stderr)
        return 1
    loaded: Dict[str, NumericArray] = {}

    for band in required:
        path = band_path_map.get(band)
        if not path:
            if args.allow_missing:
                print(f"[WARN] Missing band '{band}' (some indices may be skipped)")
                continue
            else:
                print(f"[ERROR] Missing required band '{band}'", file=sys.stderr)
                return 1
        try:
            arr = load_npy(path)
            if mask_arr is not None:
                arr = apply_mask(arr, mask_arr)
            loaded[band] = arr
        except Exception as exc:
            print(
                f"[ERROR] Failed loading '{band}' from {path}: {exc}", file=sys.stderr
            )
            return 1

    results: Dict[str, NumericArray] = {}
    for idx in indices:
        spec = INDEX_SPECS.get(idx)
        if not spec:
            print(f"[ERROR] Unsupported index '{idx}'", file=sys.stderr)
            return 1
        if any(b not in loaded for b in spec.required_bands):
            if args.allow_missing:
                print(f"[INFO] Skipping {idx} (missing bands)")
                continue
            else:
                missing = [b for b in spec.required_bands if b not in loaded]
                print(f"[ERROR] Missing bands for {idx}: {missing}", file=sys.stderr)
                return 1
        try:
            res = compute(spec, loaded, args.savi_l)
            results[idx] = res
            print(f"[OK] Computed {idx} shape={res.shape}")
        except Exception as exc:
            print(f"[ERROR] Failed computing {idx}: {exc}", file=sys.stderr)
            return 2

    if not results:
        print("[WARN] No results produced.", file=sys.stderr)
        return 0

    if multi:
        out_dir = args.out_dir
        assert out_dir
        os.makedirs(out_dir, exist_ok=True)
        for name, arr in results.items():
            out_path = os.path.join(out_dir, f"{name}.npy")
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
    sys.exit(cli())
