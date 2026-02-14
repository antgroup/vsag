# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Build System

VSAG uses CMake (3.18+) with a convenience Makefile wrapper. Build output goes to `build/` for debug and `build-release/` for release.

### Common Build Commands

```bash
# Development build with debug symbols
make debug

# Run all tests (unit tests, functional tests, mockimpl tests)
make test                                    # Runs: ./build/tests/unittests, ./build/tests/functests, ./build/mockimpl/tests_mockimpl

# Run a single test by name
make test CASE="[my_tag]"                    # Filter tests using Catch2 tags
make test CASE="TestNamePattern"             # Filter by test name pattern

# Build with sanitizers
make asan                                    # AddressSanitizer build
make test_asan                               # Run tests with ASan
make tsan                                    # ThreadSanitizer build
make test_tsan                               # Run tests with TSan

# Code coverage (requires >= 90% coverage)
make cov                                     # Build with coverage
make test_cov                                # Run tests and generate coverage report

# Code formatting and linting
make fmt                                     # Format code with clang-format
make lint                                    # Run clang-tidy

# Release builds
make release                                 # Optimized release build
make pyvsag                                  # Build Python wheel
make install                                 # Install release version
```

### Key Build Options (Environment Variables)

See `Makefile` for full list. Common options:
- `COMPILE_JOBS=8` - Parallel compilation jobs (default: 6)
- `VSAG_ENABLE_TESTS=ON/OFF` - Enable/disable tests
- `VSAG_ENABLE_PYBINDS=ON/OFF` - Enable Python bindings
- `VSAG_ENABLE_INTEL_MKL=ON/OFF` - Enable Intel MKL

## Testing

Tests use **Catch2** framework. Three separate test executables:

- `./build/tests/unittests` - Unit tests (alongside source in `src/`)
- `./build/tests/functests` - Functional tests (in `tests/`)
- `./build/mockimpl/tests_mockimpl` - Mock implementation tests

### Test Filtering Examples

```bash
# Run tests matching a tag
./build/tests/unittests "[hnsw]"

# Run specific test by name
./build/tests/unittests "Test Factory"

# List all tests
./build/tests/unittests --list-tests

# Run with detailed output
./build/tests/unittests -d yes
```

### Test Structure

- Unit tests are co-located with source files (e.g., `dataset_impl_test.cpp` alongside `dataset_impl.cpp`)
- Functional tests are in `tests/` directory
- Test fixtures and utilities are in `tests/fixtures/`
- Use tags like `[ft][factory]` for categorization

## Code Style

- **Language**: C++17 (C++20 for Clang 17+)
- **Indentation**: 4 spaces
- **Line length**: 100 characters
- **File extension**: `.cpp` (not `.cc`)
- **Style guide**: Google C++ Style Guide with modifications (see `.clang-format`)

### Required Copyright Header

All source files must include this header:

```cpp
// Copyright 2024-present the vsag project
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
```

## Project Architecture

VSAG is a vector indexing library for approximate nearest neighbor (ANN) search. It provides multiple indexing algorithms optimized for different use cases and hardware.

### Core Concepts

**Index Lifecycle:**
1. Create index via `vsag::Factory::CreateIndex(name, json_parameters)`
2. Build index with `index->Build(dataset)` or add vectors incrementally with `index->Add(dataset)`
3. Search with `index->KnnSearch(query, topk, search_params)` or `index->RangeSearch(...)`
4. Serialize/deserialize for persistence

**Key Types:**
- `vsag::Index` - Main index interface (see `include/vsag/index.h`)
- `vsag::Dataset` - Vector data container with fluent builder pattern
- `vsag::Factory` - Creates indices and file readers
- `tl::expected<T, Error>` - Result type for error handling

### Directory Structure

```
include/vsag/          # Public API headers
src/
  algorithm/           # Index algorithm implementations
    hgraph.cpp/h       # HGraph (enhanced HNSW)
    hnswlib/           # HNSW algorithm
    ivf.cpp/h          # IVF clustering-based index
    pyramid.cpp/h      # Pyramid multi-resolution index
    sindi/             # SINDI sparse vector index
    brute_force.cpp/h  # Exact search baseline
    inner_index_interface.h  # Base class for all indices
  datacell/            # Data storage abstractions
  quantization/        # Quantization methods (PQ, RaBitQ, etc.)
  simd/                # SIMD optimizations (AVX2, AVX512, NEON)
  io/                  # I/O abstractions
  storage/             # Persistent storage
  impl/                # Implementation details
  factory/             # Index creation factory
  utils/               # Utilities
tests/                 # Functional tests
  fixtures/            # Test utilities and fixtures
examples/              # Example code
  cpp/                 # C++ examples (101_index_hnsw.cpp, etc.)
  python/              # Python examples
extern/                # Third-party dependencies
python_bindings/       # pybind11 Python bindings
mockimpl/              # Mock implementations for testing
```

### Algorithm Architecture

All index algorithms inherit from `InnerIndexInterface` (in `src/algorithm/inner_index_interface.h`). Key methods to implement:

- `Add()` - Insert vectors into index
- `Build()` - Build index from dataset (batch construction)
- `KnnSearch()` - K-nearest neighbor search
- `RangeSearch()` - Radius-based search
- `Serialize()/Deserialize()` - Persistence

The factory pattern registers algorithms by name (e.g., "hnsw", "hgraph", "sindi").

### Dataset Builder Pattern

```cpp
auto dataset = vsag::Dataset::Make();
dataset->Dim(dim)
       ->NumElements(count)
       ->Ids(ids)
       ->Float32Vectors(vectors)
       ->Owner(true);  // VSAG owns the memory
```

### Key Algorithms

- **HNSW** (`src/algorithm/hnswlib/`): Hierarchical Navigable Small World graph
- **HGraph** (`src/algorithm/hgraph.cpp`): Enhanced HNSW with optimizations
- **SINDI** (`src/algorithm/sindi/`): Sparse vector indexing
- **Pyramid** (`src/algorithm/pyramid.cpp`): Multi-resolution indexing for large-scale data
- **IVF** (`src/algorithm/ivf.cpp`): Inverted file index with clustering
- **DiskANN** (extern integration): Disk-based ANN search

### SIMD Optimizations

The library uses runtime CPU detection and dispatches to optimized implementations:
- `src/simd/` contains SIMD-optimized distance computations
- Supports SSE, AVX2, AVX512, NEON (ARM), SVE
- Distance functions are selected at runtime based on CPU capabilities

### Quantization

Located in `src/quantization/`:
- Product Quantization (PQ)
- RaBitQ (from SIGMOD 2024)
- Scalar quantization variants

## DCO Sign-off

All commits must include a Developer Certificate of Origin sign-off:

```bash
git commit -s -m "message"
```

This adds: `Signed-off-by: Your Name <email@example.com>`

## CI and Quality Gates

- PRs require >= 90% code coverage
- All tests must pass with ASan and TSan
- Code must be formatted with `make fmt`
- No compiler warnings with `-Werror`

## Working with Tests

When adding new features:
- Add unit tests in `src/` alongside implementation
- Add functional tests in `tests/` for integration scenarios
- Use test fixtures from `tests/fixtures/` for common utilities
- Mock implementations go in `mockimpl/`

## Python Bindings

Python bindings are in `python_bindings/binding.cpp` using pybind11. Built with:
```bash
make pyvsag  # Build wheel for specific Python version
make pyvsag-all  # Build wheels for all supported versions
```

Package is published as `pyvsag` on PyPI.

## Resources

- GitHub Issues: https://github.com/antgroup/vsag/issues
- Discord: https://discord.com/invite/JyDmUzuhrp
- Key documentation in `docs/` directory
- Examples in `examples/cpp/` show API usage patterns
