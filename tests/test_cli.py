import numpy as np
import pytest

from eo_processor.cli import cli


def make_band(tmp_path, name, data):
    path = tmp_path / f"{name}.npy"
    np.save(path, np.asarray(data))
    return str(path)


def load_npy(path):
    return np.load(path)


def test_single_ndvi_run(tmp_path):
    nir = make_band(tmp_path, "nir", [0.8, 0.7, 0.6])
    red = make_band(tmp_path, "red", [0.2, 0.1, 0.3])
    out_path = tmp_path / "ndvi_out.npy"
    code = cli(
        [
            "--index",
            "ndvi",
            "--nir",
            nir,
            "--red",
            red,
            "--out",
            str(out_path),
        ]
    )
    assert code == 0
    assert out_path.exists()
    arr = load_npy(out_path)
    expected = (np.load(nir) - np.load(red)) / (np.load(nir) + np.load(red))
    assert arr.shape == expected.shape
    assert np.allclose(arr, expected, rtol=1e-12)


def test_multi_indices_with_allow_missing(tmp_path):
    # Provide bands only for ndvi (nir, red); request ndvi + ndwi; ndwi should be skipped.
    nir = make_band(tmp_path, "nir", [0.5, 0.6])
    red = make_band(tmp_path, "red", [0.1, 0.2])
    out_dir = tmp_path / "out_multi"
    code = cli(
        [
            "--index",
            "ndvi",
            "ndwi",
            "--nir",
            nir,
            "--red",
            red,
            "--out-dir",
            str(out_dir),
            "--allow-missing",
        ]
    )
    assert code == 0
    assert (out_dir / "ndvi.npy").exists()
    # ndwi missing green → skipped
    assert not (out_dir / "ndwi.npy").exists()
    ndvi_vals = load_npy(out_dir / "ndvi.npy")
    expected = (load_npy(nir) - load_npy(red)) / (load_npy(nir) + load_npy(red))
    assert np.allclose(ndvi_vals, expected)


def test_missing_required_band_without_allow_missing(tmp_path):
    # Request ndvi but omit red band
    nir = make_band(tmp_path, "nir", [0.3, 0.4])
    out_path = tmp_path / "ndvi_fail.npy"
    code = cli(
        [
            "--index",
            "ndvi",
            "--nir",
            nir,
            "--out",
            str(out_path),
        ]
    )
    # Should fail with code 1 (argument / I/O error)
    assert code == 1
    assert not out_path.exists()


def test_delta_nbr_run(tmp_path):
    pre_nir = make_band(tmp_path, "pre_nir", [0.8, 0.6])
    pre_swir2 = make_band(tmp_path, "pre_swir2", [0.2, 0.25])
    post_nir = make_band(tmp_path, "post_nir", [0.7, 0.55])
    post_swir2 = make_band(tmp_path, "post_swir2", [0.25, 0.3])
    out_path = tmp_path / "delta_nbr.npy"
    code = cli(
        [
            "--index",
            "delta_nbr",
            "--pre-nir",
            pre_nir,
            "--pre-swir2",
            pre_swir2,
            "--post-nir",
            post_nir,
            "--post-swir2",
            post_swir2,
            "--out",
            str(out_path),
        ]
    )
    assert code == 0
    assert out_path.exists()
    arr = load_npy(out_path)
    pre_nbr = (load_npy(pre_nir) - load_npy(pre_swir2)) / (
        load_npy(pre_nir) + load_npy(pre_swir2)
    )
    post_nbr = (load_npy(post_nir) - load_npy(post_swir2)) / (
        load_npy(post_nir) + load_npy(post_swir2)
    )
    expected = pre_nbr - post_nbr
    assert np.allclose(arr, expected, rtol=1e-12)


def test_clamp_applied(tmp_path):
    nir = make_band(tmp_path, "nir", [0.9, 0.05, 0.6])
    red = make_band(tmp_path, "red", [0.1, 0.02, 0.3])
    out_path = tmp_path / "ndvi_clamped.npy"
    code = cli(
        [
            "--index",
            "ndvi",
            "--nir",
            nir,
            "--red",
            red,
            "--out",
            str(out_path),
            "--clamp",
            "-0.05",
            "0.05",
        ]
    )
    assert code == 0
    vals = load_npy(out_path)
    assert np.all(vals >= -0.0500001)
    assert np.all(vals <= 0.0500001)


def test_savi_custom_L(tmp_path):
    nir = make_band(tmp_path, "nir", [0.7, 0.6])
    red = make_band(tmp_path, "red", [0.2, 0.3])
    out_path = tmp_path / "savi_L.npy"
    L = 0.25
    code = cli(
        [
            "--index",
            "savi",
            "--nir",
            nir,
            "--red",
            red,
            "--savi-l",
            str(L),
            "--out",
            str(out_path),
        ]
    )
    assert code == 0
    out = load_npy(out_path)
    nir_arr = load_npy(nir)
    red_arr = load_npy(red)
    denom = nir_arr + red_arr + L
    expected = (nir_arr - red_arr) / denom * (1.0 + L)
    mask = np.isclose(denom, 0.0, atol=1e-10)
    expected[mask] = 0.0
    assert np.allclose(out, expected, rtol=1e-12)


def test_png_preview_single_index(tmp_path):
    nir = make_band(tmp_path, "nir", [0.6, 0.7, 0.8])
    red = make_band(tmp_path, "red", [0.2, 0.3, 0.4])
    out_path = tmp_path / "ndvi.npy"
    png_path = tmp_path / "ndvi.png"
    code = cli(
        [
            "--index",
            "ndvi",
            "--nir",
            nir,
            "--red",
            red,
            "--out",
            str(out_path),
            "--png-preview",
            str(png_path),
        ]
    )
    assert code == 0
    assert out_path.exists()
    # PNG may or may not exist depending on Pillow presence; ensure code path executed without error.
    # If Pillow is installed, file should exist.
    if png_path.exists():
        assert png_path.stat().st_size > 0


def test_png_preview_invalid_with_multi_indices(tmp_path):
    nir = make_band(tmp_path, "nir", [0.6, 0.7])
    red = make_band(tmp_path, "red", [0.2, 0.3])
    out_dir = tmp_path / "out_dir"
    # Multi indices + png preview should trigger parser.error -> SystemExit
    with pytest.raises(SystemExit):
        cli(
            [
                "--index",
                "ndvi",
                "savi",
                "--nir",
                nir,
                "--red",
                red,
                "--out-dir",
                str(out_dir),
                "--png-preview",
                str(out_dir / "preview.png"),
            ]
        )


def test_mask_application(tmp_path):
    nir = make_band(tmp_path, "nir", [0.8, 0.7, 0.6])
    red = make_band(tmp_path, "red", [0.2, 0.1, 0.3])
    mask = make_band(tmp_path, "mask", [1, 0, 1])  # middle pixel masked
    out_path = tmp_path / "ndvi_masked.npy"
    code = cli(
        [
            "--index",
            "ndvi",
            "--nir",
            nir,
            "--red",
            red,
            "--mask",
            mask,
            "--out",
            str(out_path),
        ]
    )
    assert code == 0
    out = load_npy(out_path)
    assert np.isnan(
        out[1]
    )  # masked pixel becomes NaN after computation (inputs were NaN)


def test_allow_missing_skips_all(tmp_path):
    # Request indices requiring bands not provided so all skipped
    out_dir = tmp_path / "empty_out"
    code = cli(
        [
            "--index",
            "ndwi",
            "gci",
            "--out-dir",
            str(out_dir),
            "--allow-missing",
        ]
    )
    # Code returns 0 (no results produced but not considered error)
    assert code == 0
    # Directory may or may not be created; if created should be empty
    if out_dir.exists():
        assert not list(out_dir.iterdir())


def test_cli_list(tmp_path):
    # Just ensure --list exits cleanly with code 0
    code = cli(["--list"])
    assert code == 0


def test_cli_missing_out_dir_multi(tmp_path):
    # Multi-index without --out-dir should trigger parser.error -> SystemExit
    nir = tmp_path / "nir.npy"
    red = tmp_path / "red.npy"
    np.save(nir, np.array([0.6, 0.7]))
    np.save(red, np.array([0.2, 0.3]))
    with pytest.raises(SystemExit):
        cli(
            [
                "--index",
                "ndvi",
                "savi",
                "--nir",
                str(nir),
                "--red",
                str(red),
                # --out-dir intentionally omitted
            ]
        )


def test_cli_missing_out_single(tmp_path):
    # Single index without --out should raise SystemExit
    nir = tmp_path / "nir.npy"
    red = tmp_path / "red.npy"
    np.save(nir, np.array([0.6, 0.7]))
    np.save(red, np.array([0.2, 0.3]))
    with pytest.raises(SystemExit):
        cli(
            [
                "--index",
                "ndvi",
                "--nir",
                str(nir),
                "--red",
                str(red),
                # --out omitted
            ]
        )


def test_cli_invalid_index(tmp_path):
    # Unsupported index name should return exit code 1
    nir = tmp_path / "nir.npy"
    red = tmp_path / "red.npy"
    np.save(nir, np.array([0.6, 0.7]))
    np.save(red, np.array([0.2, 0.3]))
    out_path = tmp_path / "out.npy"
    code = cli(
        [
            "--index",
            "not_a_real_index",
            "--nir",
            str(nir),
            "--red",
            str(red),
            "--out",
            str(out_path),
        ]
    )
    assert code == 1
    assert not out_path.exists()


def test_cli_invalid_dimension_band(tmp_path):
    # Provide 3D array to NDVI; loader should reject and return error code 1
    nir = tmp_path / "nir.npy"
    red = tmp_path / "red.npy"
    np.save(nir, np.random.rand(2, 2, 2))
    np.save(red, np.random.rand(2, 2, 2))
    out_path = tmp_path / "ndvi.npy"
    code = cli(
        [
            "--index",
            "ndvi",
            "--nir",
            str(nir),
            "--red",
            str(red),
            "--out",
            str(out_path),
        ]
    )
    assert code == 1
    assert not out_path.exists()


def test_png_preview_all_nan(tmp_path):
    # All values become NaN (mask zeros out every pixel); PNG should be skipped (file absent)
    nir = make_band(tmp_path, "nir", [0.6, 0.7, 0.8])
    red = make_band(tmp_path, "red", [0.2, 0.3, 0.4])
    mask = make_band(tmp_path, "mask", [0, 0, 0])  # all masked -> all NaN
    out_path = tmp_path / "ndvi_all_nan.npy"
    png_path = tmp_path / "ndvi_all_nan.png"
    code = cli(
        [
            "--index",
            "ndvi",
            "--nir",
            nir,
            "--red",
            red,
            "--mask",
            mask,
            "--out",
            str(out_path),
            "--png-preview",
            str(png_path),
        ]
    )
    assert code == 0
    assert out_path.exists()
    # PNG should not be created because data are all NaN
    assert not png_path.exists()


def test_missing_file_handling(tmp_path):
    # Provide a non-existent band path; expect exit code 1 and no output file
    nir = make_band(tmp_path, "nir", [0.5, 0.6])
    missing_red = str(tmp_path / "does_not_exist.npy")
    out_path = tmp_path / "ndvi_missing.npy"
    code = cli(
        [
            "--index",
            "ndvi",
            "--nir",
            nir,
            "--red",
            missing_red,
            "--out",
            str(out_path),
        ]
    )
    assert code == 1
    assert not out_path.exists()


if __name__ == "__main__":  # pragma: no cover
    pytest.main([__file__])
