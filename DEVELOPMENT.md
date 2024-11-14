# VSAG Developer Guide

Welcome to the developer guide for VSAG! This guide is designed to provide both new and experienced contributors with a comprehensive resource for understanding the project's codebase, development processes, and best practices.

Whether you're an open-source enthusiast looking to make your first contribution or a seasoned developer seeking insights into the project's architecture, the guide aims to streamline your onboarding process and empower you to contribute effectively.

Let's dive in and explore how you can become an integral part of our vibrant open-source community!

## Development Environment

There are two ways to build and develop the VSAG project now.

### Use Docker(recommended)
```bash
docker pull vsaglib/vsag:ubuntu
```

### or Install Dependencies
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
debug:                   ## Build vsag with debug options.
release:                 ## Build vsag with release options.
distribution:            ## Build vsag with distribution options.
libcxx:                  ## Build vsag using libc++.
fmt:                     ## Format codes.
test:                    ## Build and run unit tests.
asan:                    ## Build with AddressSanitizer option.
test_asan_parallel: asan ## Run unit tests parallel with AddressSanitizer option.
test_asan: asan          ## Run unit tests with AddressSanitizer option.
tsan:                    ## Build with ThreadSanitizer option.
test_tsan: tsan          ## Run unit tests with ThreadSanitizer option.
test_cov: cov            ## Build and run unit tests with code coverage enabled.
clean:                   ## Clear build/ directory.
install:                 ## Build and install the release version of vsag.
```

## Project Structure
- `benchs/`: benchmark script in Python
- `cmake/`: cmake util functions
- `docker/`: the dockerfile to build develop and ci image
- `docs/`: the design documents
- `examples/`: cpp and python example codes
- `extern/`: third-party libraries
- `include/`: export header files
- `mockimpl/`: the mock implementation that can be used in interface test
- `python_bindings/`: the python bindings
- `scripts/`: useful scripts
- `src/`: the source codes and unit tests
- `tests/`: the functional tests
