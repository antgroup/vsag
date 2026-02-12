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
- Build Tools: CMake version 3.18.0 or later
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

### Build Performance Tuning

The VSAG project can be resource-intensive during compilation. You can control the number of parallel build jobs using the `COMPILE_JOBS` environment variable:

```bash
# Default: uses 4 parallel jobs
make debug

# Customize based on your system resources
# For systems with more CPU cores
make COMPILE_JOBS=8 debug

# For systems with limited memory, reduce parallelism
make COMPILE_JOBS=2 debug

# Disable specific features to speed up development builds
make VSAG_ENABLE_TESTS=OFF VSAG_ENABLE_EXAMPLES=OFF debug
```

**Resource Requirements:**
- Minimum: 4 CPU cores, 8GB RAM
- Recommended: 8+ CPU cores, 16GB RAM
- For full parallel build with all features: 16GB+ RAM recommended

**Tips for Resource-Constrained Environments:**
- Reduce `COMPILE_JOBS` to 1-2 if experiencing out-of-memory errors
- Disable unnecessary components during development (tests, examples, tools)
- Use `make clean` before rebuilding to free up disk space

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
