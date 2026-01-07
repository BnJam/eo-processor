"""
BFAST Monitor example (synthetic time series) + optional PNG outputs.

This script demonstrates the `eo_processor.bfast_monitor` workflow using a
synthetic seasonal signal with (a) a known breakpoint and (b) a stable series.

It is designed for:
- onboarding: show how to format `stack` and `dates` inputs
- smoke testing: exercise the Rust/PyO3 path end-to-end

Run:
  python examples/bfast_monitor_example.py

Optional PNG outputs (written under examples/_out_bfast_monitor/):
  python examples/bfast_monitor_example.py --png

Optional parameters:
  --height 128 --width 128 --seed 42 --order 1 --h 0.35 --alpha 0.001

Notes:
- `dates` are passed as integers in YYYYMMDD format (e.g., 20150101).
- `stack` is a NumPy array shaped (time, y, x) and dtype float.
- Output is a (2, y, x) float64 array:
    channel 0: breakpoint date in fractional years (0.0 if none)
    channel 1: magnitude at detected breakpoint (0.0 if none)
- PNG writing is optional and uses matplotlib. Images are not committed into
  any distribution build artifacts because they are generated at runtime.

Outputs (`--png`):
- `bfast_detected_mask.png`: boolean mask (white=detected)
- `bfast_confidence.png`: a magnitude-based "confidence-like" visualization for demos

Goal:
- With the default parameters, the stable scenario should typically yield <5%
  detections while the break scenario detects widespread change.
- The break scenario should show *spatial variation* in both magnitude and (usually)
  break timing. If timing is uniform, increase delay range or tune `h`/`alpha`.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd

from eo_processor import bfast_monitor


def _date_range_16d_int(start_yyyymmdd: int, end_yyyymmdd: int) -> np.ndarray:
    """
    Create a 16-day cadence list of dates as integers (YYYYMMDD).

    Uses pandas for correctness and simplicity.
    """
    start_year = start_yyyymmdd // 10000
    start_month = (start_yyyymmdd // 100) % 100
    start_day = start_yyyymmdd % 100

    end_year = end_yyyymmdd // 10000
    end_month = (end_yyyymmdd // 100) % 100
    end_day = end_yyyymmdd % 100

    start = f"{start_year:04d}-{start_month:02d}-{start_day:02d}"
    end = f"{end_year:04d}-{end_month:02d}-{end_day:02d}"

    dates = pd.to_datetime(pd.date_range(start=start, end=end, freq="16D"))
    return (dates.year * 10000 + dates.month * 100 + dates.day).to_numpy(dtype=np.int64)


def _frac_years_from_yyyymmdd(dates_int: np.ndarray) -> np.ndarray:
    """
    Convert YYYYMMDD int dates to fractional years (year + day_of_year/365.25).

    Uses pandas for correctness (avoids invalid date parsing edge cases).
    """
    years = dates_int // 10000
    months = (dates_int // 100) % 100
    days = dates_int % 100

    dates = pd.to_datetime(
        {
            "year": years.astype(int),
            "month": months.astype(int),
            "day": days.astype(int),
        }
    )
    # pandas returns a Series here; day-of-year is accessed via the `.dt` accessor.
    doy = dates.dt.dayofyear.to_numpy(dtype=np.float64)

    return years.astype(np.float64) + doy / 365.25


def _make_synthetic_stack(
    *,
    height: int,
    width: int,
    dates_int: np.ndarray,
    history_start_date: int,
    monitor_start_date: int,
    seed: int,
    break_drop: float,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Build two (time, y, x) stacks:
      - stack_break: monitoring period includes a spatially-varying break (break scenario)
      - stack_stable: stable signal designed to yield <5% detections with defaults

    Goals:
    - Make the spatial fields (magnitude + break timing) visually interesting (not uniform).
    - Keep the *stable* scenario clean enough under default parameters.
    - Avoid external data dependencies; everything is synthetic.
    """
    rng = np.random.default_rng(seed)
    t_frac = _frac_years_from_yyyymmdd(dates_int)

    # Spatial coordinate grids in [-1, 1] for smooth fields.
    yy, xx = np.mgrid[0:height, 0:width]
    yy = (yy / max(1, height - 1)) * 2.0 - 1.0
    xx = (xx / max(1, width - 1)) * 2.0 - 1.0

    # Stable base signal (seasonal) with low noise.
    # Add a *spatially varying* offset and phase so pixels are not perfectly correlated.
    stable_noise = rng.normal(0.0, 0.005, size=t_frac.shape[0])
    spatial_offset = 0.03 * (0.6 * xx + 0.4 * yy)  # gentle gradient
    spatial_phase = 0.40 * np.sin(2.0 * np.pi * (0.35 * xx + 0.20 * yy))  # radians-ish
    spatial_amp = 0.90 + 0.15 * (np.sin(2.0 * np.pi * xx) * np.cos(2.0 * np.pi * yy))

    # Build time × space stable stack with per-pixel phase shifts.
    # This keeps the seasonal shape but makes maps less uniform.
    base_cos = np.cos(2.0 * np.pi * t_frac)[:, None, None]
    base_sin = np.sin(4.0 * np.pi * t_frac)[:, None, None]
    phase_term = np.cos(spatial_phase)[None, :, :]  # couples spatial phase into seasonal term

    stack_stable = (
        (0.55 + stable_noise)[:, None, None]
        + (0.18 * base_cos) * phase_term
        + (0.10 * base_sin)
    )
    stack_stable = (stack_stable * spatial_amp[None, :, :] + spatial_offset[None, :, :]).astype(np.float64)

    # --- Break scenario: spatially varying break timing + magnitude ---
    stack_break = stack_stable.copy()

    # Spatially varying drop magnitude (kept near break_drop, but not uniform).
    drop_field = break_drop * (0.75 + 0.35 * (np.sin(2.0 * np.pi * (0.22 * xx + 0.31 * yy)) + 1.0) / 2.0)
    drop_field = drop_field.astype(np.float64)

    # Spatially varying delay (in number of *samples*) after monitor_start.
    # This is intended to spread detected break dates across the map.
    #
    # NOTE: Depending on `h`/`alpha` and the MOSUM windowing, detections may cluster at a
    # similar date even when the onset varies. Increasing delay range generally helps.
    #
    # Range: [0, max_delay] samples.
    max_delay = 40
    delay_field = (
        max_delay * (0.5 + 0.5 * np.sin(2.0 * np.pi * (0.18 * xx - 0.27 * yy)))
    ).round().astype(int)

    # Find the monitoring start index in the time axis.
    monitor_start_idx = int(np.argmax(dates_int >= np.int64(monitor_start_date)))

    # Apply the break with per-pixel onset and magnitude.
    for y in range(height):
        for x in range(width):
            onset = monitor_start_idx + int(delay_field[y, x])
            if onset < stack_break.shape[0]:
                stack_break[onset:, y, x] = stack_break[onset:, y, x] - drop_field[y, x]

    # Add modest heteroscedastic noise to the break stack (varies over time) for texture.
    break_noise_t = rng.normal(0.0, 0.010, size=stack_break.shape[0]).astype(np.float64)
    stack_break = stack_break + break_noise_t[:, None, None]

    # Keep within a plausible range for demo stability.
    stack_stable = np.clip(stack_stable, -0.2, 1.2)
    stack_break = np.clip(stack_break, -0.2, 1.2)

    return stack_break, stack_stable


def _summarize_result(name: str, result: np.ndarray) -> None:
    break_dates = result[0]
    magnitudes = result[1]

    detected = break_dates > 0.0
    n = detected.size
    n_det = int(detected.sum())

    print(f"\n{name}:")
    print(f"  result shape: {result.shape} (channels, y, x)")
    print(f"  detected pixels: {n_det}/{n} ({(n_det / n * 100.0):.1f}%)")

    if n_det > 0:
        bd = break_dates[detected]
        mag = magnitudes[detected]
        print(f"  break_date_frac: min={bd.min():.3f}, p50={np.median(bd):.3f}, max={bd.max():.3f}")
        print(f"  magnitude:       min={mag.min():.3f}, p50={np.median(mag):.3f}, max={mag.max():.3f}")
    else:
        print("  break_date_frac: (none detected)")
        print("  magnitude:       (none detected)")


def _write_pngs(out_dir: Path, *, stack_break: np.ndarray, dates_int: np.ndarray, result_break: np.ndarray) -> None:
    """
    Write a few diagnostic PNGs for demonstration; requires matplotlib.

    Includes debug prints so you can confirm the numeric detection stats match
    what is being written to the PNGs.
    """
    try:
        import matplotlib.pyplot as plt  # type: ignore
    except Exception as e:  # pragma: no cover
        raise SystemExit(
            "PNG output requires matplotlib. Install it and re-run with --png.\n"
            f"Import error: {e}"
        ) from e

    out_dir.mkdir(parents=True, exist_ok=True)

    break_dates = result_break[0]
    magnitudes = result_break[1]
    detected = break_dates > 0.0

    # Debug: confirm the exact arrays used to generate PNGs match the printed summary.
    # This helps catch cases where you are viewing stale PNGs or results are different
    # from what you think is being plotted.
    det_ratio = float(detected.mean())
    bd_min = float(break_dates.min())
    bd_max = float(break_dates.max())
    mag_min = float(magnitudes.min())
    mag_max = float(magnitudes.max())
    print(
        "\nPNG debug (from result_break used for images):\n"
        f"  detected_ratio={det_ratio:.6f} (detected pixels={int(detected.sum())}/{detected.size})\n"
        f"  break_date_frac min/max={bd_min:.6f}/{bd_max:.6f}\n"
        f"  magnitude      min/max={mag_min:.6f}/{mag_max:.6f}\n"
        f"  output_dir={out_dir}"
    )

    # 1) Detected mask
    #
    # NOTE: Some viewers (and some matplotlib backends) can make a 0/1 uint8 image
    # appear “all black” due to autoscaling / interpolation / display heuristics.
    # Scale to 0/255 and force vmin/vmax so white pixels are unmistakably white.
    mask_img = (detected.astype(np.uint8) * 255)
    plt.figure(figsize=(6, 5))
    plt.imshow(mask_img, cmap="gray", vmin=0, vmax=255, interpolation="nearest")
    plt.title("BFAST Monitor: detected break mask (white=detected)")
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(out_dir / "bfast_detected_mask.png", dpi=150)
    plt.close()

    # 1b) Magnitude-based "confidence-like" visualization
    #
    # This is intended purely for demonstrations: higher magnitudes are rendered as
    # higher confidence of change. It does not represent a calibrated probability.
    #
    # We normalize magnitudes robustly using percentiles among detected pixels so
    # outliers don't dominate the colormap.
    conf = np.zeros_like(magnitudes, dtype=np.float64)
    if detected.any():
        mags = magnitudes[detected]
        lo = float(np.percentile(mags, 5))
        hi = float(np.percentile(mags, 95))
        if hi <= lo:
            hi = lo + 1e-12
        conf_detected = (mags - lo) / (hi - lo)
        conf_detected = np.clip(conf_detected, 0.0, 1.0)
        conf[detected] = conf_detected

    plt.figure(figsize=(6, 5))
    plt.imshow(conf, cmap="inferno", vmin=0.0, vmax=1.0, interpolation="nearest")
    plt.title("BFAST Monitor: magnitude-based confidence (normalized)")
    plt.colorbar(fraction=0.046, pad=0.04)
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(out_dir / "bfast_confidence.png", dpi=150)
    plt.close()

    # 2) Magnitude heatmap
    plt.figure(figsize=(6, 5))
    plt.imshow(magnitudes, cmap="magma")
    plt.title("BFAST Monitor: break magnitude")
    plt.colorbar(fraction=0.046, pad=0.04)
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(out_dir / "bfast_magnitude.png", dpi=150)
    plt.close()

    # 3) Break date map (fractional years; zeros where none)
    plt.figure(figsize=(6, 5))
    plt.imshow(break_dates, cmap="viridis")
    plt.title("BFAST Monitor: break date (fractional years; 0=none)")
    plt.colorbar(fraction=0.046, pad=0.04)
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(out_dir / "bfast_break_date_frac.png", dpi=150)
    plt.close()

    # 4) A couple of pixel time series plots (one expected break, one stable-ish)
    # Pick a pixel near center and a pixel with a strong amplitude modulation.
    h, w = break_dates.shape
    pixels = [(h // 2, w // 2), (h // 4, 3 * w // 4)]
    t_frac = _frac_years_from_yyyymmdd(dates_int)

    for (yy, xx) in pixels:
        ts = stack_break[:, yy, xx]
        bd = break_dates[yy, xx]
        mag = magnitudes[yy, xx]

        plt.figure(figsize=(9, 3))
        plt.plot(t_frac, ts, linewidth=1.5)
        plt.axvline(_frac_years_from_yyyymmdd(np.array([dates_int[0]], dtype=np.int64))[0], color="none")  # noop (keeps style)
        if bd > 0.0:
            plt.axvline(bd, color="red", linestyle="--", linewidth=1.25, label=f"break ~ {bd:.3f}")
        plt.title(f"Pixel ({yy}, {xx}) time series | mag={mag:.3f}")
        plt.xlabel("Year (fractional)")
        plt.ylabel("Signal")
        plt.grid(True, alpha=0.25)
        if bd > 0.0:
            plt.legend(loc="best")
        plt.tight_layout()
        plt.savefig(out_dir / f"bfast_timeseries_y{yy}_x{xx}.png", dpi=150)
        plt.close()

    print(f"\nWrote PNG outputs to: {out_dir}")


def main() -> int:
    parser = argparse.ArgumentParser(description="Run a synthetic BFAST Monitor demo.")
    parser.add_argument("--height", type=int, default=128, help="Raster height (y).")
    parser.add_argument("--width", type=int, default=128, help="Raster width (x).")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility.")
    parser.add_argument("--order", type=int, default=1, help="Harmonic order for BFAST Monitor.")
    parser.add_argument("--h", type=float, default=0.35, help="MOSUM window fraction (h).")
    parser.add_argument("--alpha", type=float, default=0.001, help="Significance level (alpha).")
    parser.add_argument(
        "--break-drop",
        type=float,
        default=0.4,
        help="Drop applied to monitoring-period values for the break scenario.",
    )
    parser.add_argument(
        "--png",
        action="store_true",
        help="Write demonstration PNG(s) under examples/_out_bfast_monitor/ (requires matplotlib).",
    )
    args = parser.parse_args()

    # Date configuration mirrors the unit test pattern: history then monitoring.
    history_start_date = 20100101
    monitor_start_date = 20150101
    end_date = 20171231

    # 16-day cadence date list across full period
    dates_int = _date_range_16d_int(history_start_date, end_date)

    print("BFAST Monitor example")
    print("--------------------")
    print(f"dates: {int(dates_int[0])} .. {int(dates_int[-1])} (n={dates_int.size})")
    print(f"stack shape: (time={dates_int.size}, y={args.height}, x={args.width})")

    # Build two synthetic stacks: with a break and without.
    stack_break, stack_stable = _make_synthetic_stack(
        height=args.height,
        width=args.width,
        dates_int=dates_int,
        history_start_date=history_start_date,
        monitor_start_date=monitor_start_date,
        seed=args.seed,
        break_drop=args.break_drop,
    )

    # Run BFAST Monitor (Rust-backed workflow)
    result_break = bfast_monitor(
        stack_break,
        dates_int.tolist(),
        history_start_date=history_start_date,
        monitor_start_date=monitor_start_date,
        order=args.order,
        h=args.h,
        alpha=args.alpha,
    )

    result_stable = bfast_monitor(
        stack_stable,
        dates_int.tolist(),
        history_start_date=history_start_date,
        monitor_start_date=monitor_start_date,
        order=args.order,
        h=args.h,
        alpha=args.alpha,
    )

    # Basic invariants (smoke-test style)
    assert isinstance(result_break, np.ndarray), "Expected numpy ndarray output"
    assert result_break.shape == (2, args.height, args.width)
    assert result_stable.shape == (2, args.height, args.width)

    _summarize_result("Break scenario", result_break)
    _summarize_result("Stable scenario", result_stable)

    # We want a demonstration that is both:
    # - a smoke run (exercises the compiled workflow end-to-end)
    # - a reasonably interpretable demo (stable scenario should be mostly "no break")
    break_detected_ratio = float((result_break[0] > 0.0).mean())
    stable_detected_ratio = float((result_stable[0] > 0.0).mean())
    print("\nSanity checks:")
    print(f"  break detected ratio:  {break_detected_ratio:.3f}")
    print(f"  stable detected ratio: {stable_detected_ratio:.3f}")

    # The break scenario should generally detect widespread change for this synthetic drop.
    # If it doesn't, something is likely wrong with the install / wiring / parameters.
    assert break_detected_ratio > 0.05, "Expected some break detections in the break scenario"

    # Stable scenario target: <5% detections with default parameters.
    # This is still a demo (not a unit test), but we keep this threshold to catch regressions
    # where the algorithm becomes overly sensitive for a clean seasonal signal.
    assert stable_detected_ratio < 0.05, (
        "Expected <5% detections in the stable scenario. If this fails, consider tuning "
        "`alpha` (smaller is stricter), `h`, or reducing synthetic noise/complexity."
    )

    if args.png:
        out_dir = Path("examples") / "_out_bfast_monitor"
        _write_pngs(out_dir, stack_break=stack_break, dates_int=dates_int, result_break=result_break)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())