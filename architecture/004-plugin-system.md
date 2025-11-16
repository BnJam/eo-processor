# ADR-003: Plugin System for Spectral Indices

**Status**: Proposed

## Context

The `eo-processor` library currently provides a fixed set of spectral indices that are implemented in Rust for high performance. However, researchers and practitioners in the field of Earth Observation often develop their own custom spectral indices for specific applications. Adding a new index to the library currently requires modifying the Rust source code, which is a significant barrier to entry for many users.

## Decision

A plugin system will be implemented to allow users to define their own custom spectral indices in Python and have them executed at near-native speed.

1.  **Python-based Definition**: Users will define their custom spectral indices as Python functions. These functions will operate on NumPy arrays, and they will be decorated with a special decorator that marks them for JIT compilation.
2.  **JIT Compilation with Numba**: The [Numba](https://numba.pydata.org/) library will be used to just-in-time (JIT) compile the user-defined Python functions to highly optimized machine code. Numba is a mature and well-supported library that is widely used in the scientific Python ecosystem.
3.  **Dynamic Loading**: The JIT-compiled functions will be dynamically loaded into the `eo-processor`'s execution engine. The library will provide a registration mechanism that allows users to register their custom indices and make them available to the rest of the library.

## Consequences

### Advantages

*   **Extensibility**: Users will be able to easily add their own custom spectral indices without modifying the core library.
*   **Performance**: The use of Numba will ensure that the custom indices run at near-native speed, with performance that is comparable to the built-in Rust-based indices.
*   **Ease of Use**: Users will be able to define their indices in Python, a language they are already familiar with.

### Disadvantages

*   **Additional Dependency**: The new plugin system will introduce an additional dependency on the Numba library.
*   **Increased Complexity**: The integration of Numba will add complexity to the library's architecture.
*   **Security Considerations**: While Numba is a mature library, the execution of user-defined code always introduces some level of security risk. We will need to carefully consider the security implications of this feature.
