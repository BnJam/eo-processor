# ADR-004: Enhanced Dask Integration

**Status**: Proposed

## Context

The `eo-processor` library is designed to work with Dask, a flexible library for parallel computing in Python. The current integration is based on `dask.array.apply_gufunc`, which allows the Rust functions to be applied to Dask arrays. While this approach works, it does not take full advantage of the capabilities of Dask or the GIL-releasing nature of the Rust functions.

## Decision

The Dask integration will be enhanced to provide a more seamless and efficient user experience.

1.  **Custom Dask Collections**: We will develop custom Dask collections that are specifically designed for working with geospatial raster data. These collections will provide a high-level API for common EO workflows, and they will be backed by the Rust-based processing functions.
2.  **GIL-Aware Schedulers**: We will investigate the use of custom Dask schedulers that are aware of the GIL-releasing nature of the Rust functions. This will allow Dask to more efficiently schedule and execute tasks that call into the Rust core.
3.  **Graph Optimizations**: We will explore the use of custom Dask graph optimizations to fuse multiple `eo-processor` operations into a single, more efficient task.

## Consequences

### Advantages

*   **Improved Performance**: The enhanced Dask integration will lead to significant performance improvements, particularly for large-scale, distributed workflows.
*   **Improved User Experience**: The custom Dask collections will provide a more user-friendly and intuitive API for working with geospatial data.
*   **Tighter Integration**: The deeper integration with Dask will allow us to take full advantage of the capabilities of both libraries.

### Disadvantages

*   **Increased Complexity**: The development of custom Dask collections, schedulers, and graph optimizations will add significant complexity to the library.
*   **Maintenance Overhead**: The custom Dask integration will require ongoing maintenance to keep up with changes in the Dask library.
