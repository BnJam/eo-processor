"""Type stubs for eo_processor"""

import numpy as np
from numpy.typing import NDArray

__version__: str

def normalized_difference(
    a: NDArray[np.float64], b: NDArray[np.float64]
) -> NDArray[np.float64]: ...
def ndvi(nir: NDArray[np.float64], red: NDArray[np.float64]) -> NDArray[np.float64]: ...
def ndwi(
    green: NDArray[np.float64], nir: NDArray[np.float64]
) -> NDArray[np.float64]: ...
def enhanced_vegetation_index(
    nir: NDArray[np.float64], red: NDArray[np.float64], blue: NDArray[np.float64]
) -> NDArray[np.float64]: ...

evi = enhanced_vegetation_index
