# GitHub Copilot Instructions for VSAG

## Project Overview

VSAG is a high-performance vector indexing library for similarity search, written in C++ with Python bindings (pyvsag). The library provides graph-based approximate nearest neighbor (ANN) search algorithms optimized for various hardware platforms.

## Core Technologies

- **Language**: C++11 or later
- **Build System**: CMake 3.18+, Unix Makefiles for convenience
- **Testing Framework**: Catch2
- **Python Bindings**: pybind11 (pyvsag package)
- **Dependencies**: OpenMP, libaio, Intel MKL (optional), gfortran, curl

## Coding Standards

Follow the [Google C++ Style Guide](https://google.github.io/styleguide/cppguide.html) with these modifications:

- **Indentation**: 4 spaces (not 2)
- **File Extension**: Use `.cpp` (not `.cc`)
- **Line Length**: 100 characters maximum
- **Formatting**: Use `clang-format` (run `make fmt` before committing)

### File Headers

All source files must include this copyright header:

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

### Code Organization

- **Headers**: Place public APIs in `include/vsag/`
- **Implementation**: Place source files in `src/`
- **Tests**: Unit tests alongside implementation in `src/`, functional tests in `tests/`
- **Examples**: C++ examples in `examples/cpp/`, Python examples in `examples/python/`

## Building and Testing

### Development Commands

```bash
make debug           # Build with debug options
make test            # Build and run unit tests
make fmt             # Format code with clang-format
make test_cov        # Run tests with coverage (must be >= 90%)
make asan            # Build with AddressSanitizer
make tsan            # Build with ThreadSanitizer
```

### Production Commands

```bash
make release         # Build release version
make distribution    # Build distribution package
make pyvsag          # Build Python wheel
make install         # Install release version
```

### Docker Development

```bash
docker pull vsaglib/vsag:ubuntu
```

## Testing Requirements

- **Coverage**: All contributions must maintain >= 90% code coverage
- **Test Types**:
  - Unit tests (Catch2): Place in `src/` alongside implementation
  - Functional tests (Catch2): Place in `tests/`
- **Test Naming**: Use descriptive test names with tags, e.g., `TEST_CASE("Test Factory", "[ft][factory]")`
- **Required**: New features must include tests; bug fixes must include regression tests

## API Design Patterns

### Fluent Interface / Builder Pattern

The codebase extensively uses builder patterns with method chaining:

```cpp
auto dataset = vsag::Dataset::Make();
dataset->Dim(dim)
       ->NumElements(count)
       ->Ids(ids)
       ->Float32Vectors(vectors);
```

### Result Types

Use `tl::expected` or similar result types for error handling instead of exceptions where appropriate.

### Shared Pointers

APIs commonly return `std::shared_ptr<T>` (e.g., `DatasetPtr`, `IndexPtr`).

## Performance Considerations

- **SIMD Optimization**: Be aware of AVX2, AVX512, NEON optimizations
- **Memory Efficiency**: Consider memory layout for cache efficiency
- **Quantization**: The library uses various quantization techniques (RaBitQ, product quantization)
- **Threading**: Use OpenMP for parallelization where appropriate

## Key Algorithms

Familiarize yourself with these core algorithms when contributing:
- **HNSW**: Hierarchical Navigable Small World graphs
- **SINDI**: Sparse vector indexing
- **Pyramid**: Multi-resolution indexing
- **HGraph**: Enhanced graph structures

## Documentation Standards

- **Public APIs**: Must have Doxygen comments with `@brief`, `@param`, `@return`, etc.
- **README Updates**: Update examples if adding new features
- **DEVELOPMENT.md**: Update build instructions if changing build process

## Contributing Workflow

1. Fork the repository
2. Create a feature branch from up-to-date master
3. Make changes following coding standards
4. Run `make fmt` to format code
5. Run `make test_cov` to verify >= 90% coverage
6. Commit with DCO sign-off: `git commit -s -m "message"`
7. Submit pull request

### Developer Certificate of Origin (DCO)

All commits must include a sign-off:

```
Signed-off-by: Your Name <your.email@example.com>
```

Use `git commit -s` to automatically add this.

## Common Patterns to Follow

### Index Creation

```cpp
auto index_params = R"({"dim": 128, "dtype": "float32", ...})";
auto index = vsag::Factory::CreateIndex("hnsw", index_params);
if (index.has_value()) {
    auto idx = index.value();
    // Use index...
}
```

### Dataset Construction

```cpp
auto dataset = vsag::Dataset::Make();
dataset->Dim(dim)
       ->NumElements(num)
       ->Ids(ids)
       ->Float32Vectors(data);
```

### Namespace

All code should be in the `vsag` namespace:

```cpp
namespace vsag {
// Your code here
}  // namespace vsag
```

## Project Structure

- `cmake/`: CMake utility functions
- `docker/`: Dockerfiles for development and CI
- `docs/`: Design documents
- `examples/`: C++ and Python example code
- `extern/`: Third-party libraries
- `include/`: Public header files
- `mockimpl/`: Mock implementations for testing
- `python/`: pyvsag package
- `python_bindings/`: Python binding code
- `scripts/`: Utility scripts
- `src/`: Source code and unit tests
- `tests/`: Functional tests
- `tools/`: Development tools

## Important Notes

- **Breaking Changes**: Maintain API compatibility; reviewers will flag breaking changes
- **Performance**: Be mindful of performance implications; include benchmarks for performance-critical changes
- **Cross-Platform**: Support Ubuntu 20.04+, CentOS 7+, and macOS
- **Compiler Support**: GCC 9.4.0+, Clang 13.0.0+

## Resources

- [README.md](../README.md): Project overview and quick start
- [CONTRIBUTING.md](../CONTRIBUTING.md): Detailed contribution guidelines
- [DEVELOPMENT.md](../DEVELOPMENT.md): Development environment setup
- [CODE_OF_CONDUCT.md](../CODE_OF_CONDUCT.md): Community guidelines
- [GitHub Issues](https://github.com/antgroup/vsag/issues): Bug reports and feature requests
- [Discord Community](https://discord.com/invite/JyDmUzuhrp): Community discussions

## When Making Changes

1. **Understand the Algorithm**: For index algorithms, understand the underlying principles
2. **Test Thoroughly**: Include edge cases, concurrent access, and large-scale scenarios
3. **Benchmark**: For performance-critical code, include before/after benchmarks
4. **Document**: Update relevant documentation and examples
5. **Review Coverage**: Ensure test coverage remains >= 90%

## Common Gotchas

- Don't use `.cc` extension; use `.cpp`
- Don't use 2-space indentation; use 4 spaces
- Don't forget copyright headers
- Don't forget DCO sign-off on commits
- Don't skip running `make fmt` before committing
- Don't reduce test coverage below 90%
