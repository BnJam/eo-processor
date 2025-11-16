# ADR-005: GPU Support

**Status**: Proposed

## Context

The `eo-processor` library is designed for high-performance computations on large, multi-dimensional arrays. While the current Rust-based implementation is highly optimized for CPUs, certain types of computations, such as pixel-wise operations and convolutions, can be significantly accelerated by using a Graphics Processing Unit (GPU).

## Decision

We will explore the possibility of adding GPU support to the `eo-processor` library.

1.  **CUDA and/or OpenCL**: We will investigate the use of [CUDA](https://developer.nvidia.com/cuda-toolkit) and/or [OpenCL](https://www.khronos.org/opencl/) to offload computations to the GPU. CUDA is a proprietary technology that is specific to NVIDIA GPUs, while OpenCL is an open standard that is supported by a wider range of hardware.
2.  **Rust GPU Libraries**: We will evaluate the existing ecosystem of Rust libraries for GPU computing, such as [accel](https://github.com/accel-rs/accel) and [ocl](https://github.com/cogciprocate/ocl).
3.  **Optional Feature**: GPU support will be an optional feature of the library. Users who do not have a compatible GPU or who do not wish to install the required dependencies will still be able to use the CPU-based implementation.

## Consequences

### Advantages

*   **Massive Performance Gains**: For certain types of computations, GPU acceleration can provide orders of magnitude of performance improvement over a CPU-based implementation.
*   **Increased Scalability**: GPU support will allow the library to scale to even larger datasets and more complex workflows.

### Disadvantages

*   **Increased Complexity**: GPU programming is notoriously complex, and the integration of GPU support will add a significant amount of complexity to the library.
*   **Hardware Dependencies**: GPU support will introduce a dependency on specific hardware and drivers.
*   **Limited Applicability**: Not all of the computations in the `eo-processor` library are well-suited for GPU acceleration. We will need to carefully identify the parts of the library that will benefit from GPU support.
