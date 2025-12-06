mask_with_scl
=============

.. currentmodule:: eo_processor
.. autofunction:: mask_with_scl

Overview
--------
``mask_with_scl`` applies SCL-based masking directly to a data array (e.g., spectral bands).
Unlike ``mask_scl`` which only filters the SCL array itself, this function uses the SCL
as a classification layer to mask corresponding pixels in the actual data.

This is particularly useful when:

- Processing multi-band imagery where clouds/shadows need to be removed
- Working with time-series data where the SCL mask should be applied consistently
- Building composites that require pre-masking of invalid pixels

Supported Array Shapes
----------------------
The function supports the following combinations:

+----------------------------+------------------------+
| Data Shape                 | SCL Shape              |
+============================+========================+
| 2D: (y, x)                 | 2D: (y, x)             |
+----------------------------+------------------------+
| 3D: (time, y, x)           | 3D: (time, y, x)       |
+----------------------------+------------------------+
| 4D: (time, band, y, x)     | 3D: (time, y, x)       |
+----------------------------+------------------------+

For 4D data with 3D SCL, the mask is broadcast across all bands, ensuring that if a pixel
is cloudy at time *t*, it is masked in all spectral bands at that time step.

Default Mask Codes
------------------
If ``mask_codes`` is not provided, the function masks the following SCL classes:

- 0: No Data
- 1: Saturated / Defective
- 2: Dark Area Pixels
- 3: Cloud Shadows
- 8: Cloud (Medium Probability)
- 9: Cloud (High Probability)
- 10: Thin Cirrus

This leaves valid surface pixels (vegetation, bare soil, water, snow/ice) intact.

Typical Sentinel-2 SCL Codes
----------------------------
For reference:

+------+-------------------------------------------+--------+
| Code | Meaning                                   | Action |
+------+-------------------------------------------+--------+
| 0    | No Data                                   | Mask   |
| 1    | Saturated / Defective                     | Mask   |
| 2    | Dark Area Pixels                          | Mask   |
| 3    | Cloud Shadows                             | Mask   |
| 4    | Vegetation                                | Keep   |
| 5    | Not Vegetated                             | Keep   |
| 6    | Water                                     | Keep   |
| 7    | Unclassified                              | Keep   |
| 8    | Cloud (Medium Probability)                | Mask   |
| 9    | Cloud (High Probability)                  | Mask   |
| 10   | Thin Cirrus                               | Mask   |
| 11   | Snow / Ice                                | Keep   |
+------+-------------------------------------------+--------+

Parameters
----------
- ``data`` (``numpy.ndarray``): The data array to mask (2D, 3D, or 4D).
- ``scl`` (``numpy.ndarray``): The SCL classification layer (2D or 3D).
- ``mask_codes`` (``Sequence[float] | None``): SCL codes to mask. Defaults to
  ``[0, 1, 2, 3, 8, 9, 10]``.
- ``fill_value`` (``float | None``): Value for masked pixels. Defaults to ``NaN``.

Returns
-------
``numpy.ndarray`` (``float64``): Data array with masked pixels replaced.

Examples
--------

Basic 3D Usage
~~~~~~~~~~~~~~
.. code-block:: python

    import numpy as np
    from eo_processor import mask_with_scl

    # Sentinel-2 bands: shape (time=10, y=100, x=100)
    red_band = np.random.rand(10, 100, 100)
    scl = np.random.choice([4, 5, 6, 8, 9], size=(10, 100, 100))

    # Mask cloud pixels (8, 9) in the red band
    masked_red = mask_with_scl(red_band, scl)

4D Multi-Band Usage
~~~~~~~~~~~~~~~~~~~
.. code-block:: python

    import numpy as np
    from eo_processor import mask_with_scl

    # Multi-band data: shape (time=10, band=4, y=100, x=100)
    # Bands: B02 (Blue), B03 (Green), B04 (Red), B08 (NIR)
    data = np.random.rand(10, 4, 100, 100)
    scl = np.random.choice([4, 5, 6, 8, 9], size=(10, 100, 100))

    # Mask clouds across all bands
    masked_data = mask_with_scl(data, scl)

Custom Mask Codes
~~~~~~~~~~~~~~~~~
.. code-block:: python

    # Only mask high-probability clouds and shadows
    masked = mask_with_scl(data, scl, mask_codes=[3, 9])

Custom Fill Value
~~~~~~~~~~~~~~~~~
.. code-block:: python

    # Use -9999 as nodata instead of NaN
    masked = mask_with_scl(data, scl, fill_value=-9999.0)

Integration with XArray/Dask
~~~~~~~~~~~~~~~~~~~~~~~~~~~~
.. code-block:: python

    import xarray as xr
    from eo_processor import mask_with_scl

    # Apply via xr.apply_ufunc for chunked processing
    masked_data = xr.apply_ufunc(
        mask_with_scl,
        data_array,    # 4D: (time, band, y, x)
        scl_array,     # 3D: (time, y, x)
        dask="parallelized",
        output_dtypes=[float],
    )

See Also
--------
- :func:`mask_scl` - Filter the SCL array itself
- :func:`mask_vals` - Mask specific values in any array
- :func:`replace_nans` - Replace NaN values after masking
