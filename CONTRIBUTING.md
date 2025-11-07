# Contributing to eo-processor

Thank you for your interest in contributing to eo-processor! This document provides guidelines and instructions for contributors.

## Development Setup

### Prerequisites

- Python 3.8 or higher
- Rust toolchain (install from [rustup.rs](https://rustup.rs/))
- Git

### Setting Up Development Environment

1. Clone the repository:
   ```bash
   git clone https://github.com/BnJam/eo-processor.git
   cd eo-processor
   ```

2. Create and activate a virtual environment:
   ```bash
   python -m venv .venv
   source .venv/bin/activate  # On Windows: .venv\Scripts\activate
   ```

3. Install maturin and development dependencies:
   ```bash
   pip install maturin pytest numpy
   ```

4. Build the package in development mode:
   ```bash
   maturin develop
   ```

## Development Workflow

### Building

```bash
# Development build (fast, unoptimized)
maturin develop

# Release build (optimized)
maturin develop --release
```

### Testing

```bash
# Run Python tests
pytest tests/ -v

# Run Rust library checks
cargo check --lib

# Build release version
cargo build --release --lib
```

### Code Style

- **Rust**: Follow standard Rust formatting
  ```bash
  cargo fmt
  cargo clippy
  ```

- **Python**: Follow PEP 8
  ```bash
  # Install development tools
  pip install black isort flake8
  
  # Format code
  black python/ tests/ examples/
  isort python/ tests/ examples/
  
  # Check style
  flake8 python/ tests/ examples/
  ```

## Adding New Functions

### 1. Implement in Rust (src/lib.rs)

```rust
#[pyfunction]
fn my_new_function<'py>(
    py: Python<'py>,
    input: PyReadonlyArray1<f64>,
) -> PyResult<&'py PyArray1<f64>> {
    let input_array = input.as_array();
    // Your implementation here
    Ok(result.into_pyarray(py))
}

// Add to module
#[pymodule]
fn _core(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(my_new_function, m)?)?;
    // ... other functions
    Ok(())
}
```

### 2. Add Python Wrapper (python/eo_processor/__init__.py)

```python
from ._core import my_new_function

__all__ = [
    # ... existing exports
    "my_new_function",
]
```

### 3. Add Type Stubs (python/eo_processor/__init__.pyi)

```python
def my_new_function(input: NDArray[np.float64]) -> NDArray[np.float64]: ...
```

### 4. Add Tests (tests/test_*.py)

```python
def test_my_new_function():
    from eo_processor import my_new_function
    import numpy as np
    
    input_data = np.array([1.0, 2.0, 3.0])
    result = my_new_function(input_data)
    
    # Your assertions here
    assert result.shape == input_data.shape
```

### 5. Update Documentation

- Add usage examples to README.md
- Update the function list in README.md
- Add docstrings in both Rust and Python code

## Pull Request Process

1. **Create a Branch**: Use descriptive branch names
   ```bash
   git checkout -b feature/my-new-feature
   ```

2. **Make Changes**: Follow the coding standards

3. **Test**: Ensure all tests pass
   ```bash
   pytest tests/ -v
   cargo check --lib
   ```

4. **Commit**: Write clear commit messages
   ```bash
   git commit -m "Add feature: description of changes"
   ```

5. **Push and Create PR**:
   ```bash
   git push origin feature/my-new-feature
   ```

6. **PR Description**: Include:
   - What changes were made
   - Why the changes were necessary
   - Any relevant issue numbers
   - Test results

## Code Review Checklist

Before submitting a PR, ensure:

- [ ] Code follows project style guidelines
- [ ] All tests pass
- [ ] New tests added for new functionality
- [ ] Documentation updated
- [ ] No unsafe Rust code (unless absolutely necessary and well-justified)
- [ ] No compiler warnings
- [ ] Performance impact considered
- [ ] Security implications reviewed

## Common Tasks

### Adding a New Spectral Index

1. Implement the calculation in Rust
2. Add both 1D and 2D versions
3. Create a convenience wrapper that auto-detects dimensions
4. Add comprehensive tests
5. Update documentation with formula and use cases

### Improving Performance

1. Profile the code to identify bottlenecks
2. Consider using SIMD operations (ndarray supports this)
3. Benchmark against NumPy implementation
4. Document performance improvements in PR

### Adding Documentation

- Use clear, concise language
- Include code examples
- Explain the "why" not just the "what"
- Keep README.md updated

## Getting Help

- **Issues**: Open an issue on GitHub for bugs or feature requests
- **Discussions**: Use GitHub Discussions for questions
- **Documentation**: Check README.md and code comments

## License

By contributing, you agree that your contributions will be licensed under the MIT License.
