# ADR-001: Hybrid Rust/Python Architecture with PyO3 and Maturin

**Status**: Accepted

## Context

The `eo-processor` library is designed to provide high-performance functions for common Earth Observation (EO) and geospatial computations. These computations often involve large, multi-dimensional arrays of numerical data, and performance is a critical requirement. While Python is the de-facto language for scientific computing and data analysis, its performance for CPU-bound tasks is limited by the Global Interpreter Lock (GIL). Pure Python or NumPy-based solutions can be too slow for the desired scale of processing.

## Decision

The `eo-processor` library will be implemented as a hybrid Rust/Python package.

1.  **Core Logic in Rust**: The computationally intensive core logic, such as spectral index calculations, temporal statistics, and masking operations, will be written in Rust. Rust provides high performance, memory safety, and fine-grained control over system resources.
2.  **Python API with PyO3**: The Rust core will be exposed to Python as a native extension module using the [PyO3](https://pyo3.rs/) framework. PyO3 provides seamless interoperability between Rust and Python, allowing Rust functions to be called from Python with minimal overhead.
3.  **Building and Packaging with Maturin**: The project will be built and packaged using [Maturin](https://www.maturin.rs/). Maturin simplifies the process of building and distributing binary extensions for Python, and it integrates well with the standard Python packaging tools.

## Consequences

### Advantages

*   **High Performance**: By implementing the core logic in Rust, we can achieve performance that is comparable to C or C++, without sacrificing memory safety.
*   **GIL Release**: The Rust functions can release the Python GIL, allowing them to run in parallel on multiple CPU cores. This is particularly beneficial when integrating with Dask for distributed computing.
*   **Pythonic API**: The use of PyO3 allows us to create a Python API that is easy to use and feels natural to Python developers.
*   **Memory Safety**: The Rust compiler's strict memory safety guarantees help to prevent common bugs such as null pointer dereferences and buffer overflows.

### Disadvantages

*   **Increased Complexity**: The hybrid architecture introduces additional complexity into the development workflow. Developers need to be familiar with both Rust and Python, and the build process is more complex than that of a pure Python package.
*   **Compilation Overhead**: The Rust code needs to be compiled before it can be used from Python, which can slow down the development and testing cycle.
*   **Binary Distribution**: The library needs to be distributed as a binary wheel for different platforms and Python versions, which adds complexity to the release process.
