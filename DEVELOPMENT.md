# VSAG Developer Guide

Welcome to the developer guide for VSAG! This guide is designed to provide both new and experienced contributors with a comprehensive resource for understanding the project's codebase, development processes, and best practices.

Whether you're an open-source enthusiast looking to make your first contribution or a seasoned developer seeking insights into the project's architecture, the guide aims to streamline your onboarding process and empower you to contribute effectively.

Let's dive in and explore how you can become an integral part of our vibrant open-source community!

## Development Environment

There are two ways to build and develop the VSAG project now.

### Use Docker(recommended)

![Docker Pulls](https://img.shields.io/docker/pulls/vsaglib/vsag)
![Docker Image Size](https://img.shields.io/docker/image-size/vsaglib/vsag)

```bash
docker pull vsaglib/vsag:ubuntu
```

### or Install Development Requirements

- Operating System:
  - Ubuntu 20.04 or later
  - or CentOS 7 or later
- Compiler:
  - GCC version 9.4.0 or later
  - or Clang version 13.0.0 or later
- Build Tools: 
  - CMake version 3.18.0 or later
  - clang-format version 15 EXACTLY (not higher, not lower - required for consistent formatting)
- Additional Dependencies:
  - gfortran
  - python 3.6+
  - omp
  - aio
  - curl

```bash
# for Debian/Ubuntu
$ ./scripts/deps/install_deps_ubuntu.sh

# for CentOS/AliOS
$ ./scripts/deps/install_deps_centos.sh
```

## VSAG Build Tool
VSAG project use the Unix Makefiles to compile, package and install the library. Here is the commands below:
```bash
Usage: make <target>

Targets:
help:                    ## Show the help.
##
## ================ development ================
debug:                   ## Build vsag with debug options.
test:                    ## Build and run unit tests.
asan:                    ## Build with AddressSanitizer option.
test_asan: asan          ## Run unit tests with AddressSanitizer option.
tsan:                    ## Build with ThreadSanitizer option.
test_tsan: tsan          ## Run unit tests with ThreadSanitizer option.
clean:                   ## Clear build/ directory.
##
## ================ integration ================
fmt:                     ## Format codes.
cov:                     ## Build unit tests with code coverage enabled.
test_parallel: debug     ## Run all tests parallel (used in CI).
test_asan_parallel: asan ## Run unit tests parallel with AddressSanitizer option.
test_tsan_parallel: tsan ## Run unit tests parallel with ThreadSanitizer option.
##
## ================ distribution ================
release:                 ## Build vsag with release options.
distribution:            ## Build vsag with distribution options.
libcxx:                  ## Build vsag using libc++.
pyvsag:                  ## Build pyvsag wheel.
clean-release:           ## Clear build-release/ directory.
install:                 ## Build and install the release version of vsag.
```

## CMake Build Options

VSAG provides several CMake options to customize the build:

### BLAS Library Options

- **`ENABLE_INTEL_MKL`** (default: `ON` on x86_64, `OFF` otherwise)
  - Enable Intel MKL as the BLAS backend (x86_64 platforms only)
  - When disabled, OpenBLAS is used instead

- **`USE_SYSTEM_OPENBLAS`** (default: `OFF`)
  - Use system-installed OpenBLAS instead of building from source
  - Requires `libopenblas-dev` and `liblapacke-dev` to be installed
  - Falls back to building from source if system OpenBLAS is not found
  - Example:
    ```bash
    # Install OpenBLAS on Ubuntu/Debian
    sudo apt-get install libopenblas-dev liblapacke-dev
    
    # Build with system OpenBLAS
    cmake -DUSE_SYSTEM_OPENBLAS=ON -DENABLE_INTEL_MKL=OFF -B build
    cmake --build build
    ```

### Other Build Options

For a complete list of build options, see the `option()` directives in `CMakeLists.txt`.

## Project Structure
- `cmake/`: cmake util functions
- `docker/`: the dockerfile to build develop and ci image
- `docs/`: the design documents
- `examples/`: cpp and python example codes
- `extern/`: third-party libraries
- `include/`: export header files
- `mockimpl/`: the mock implementation that can be used in interface test
- `python/`: the pyvsag package and setup tools
- `python_bindings/`: the python bindings
- `scripts/`: useful scripts
- `src/`: the source codes and unit tests
- `tests/`: the functional tests
- `tools/`: the tools
