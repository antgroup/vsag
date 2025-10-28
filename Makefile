
CMAKE_GENERATOR ?= "Unix Makefiles"
CMAKE_INSTALL_PREFIX ?= "/usr/local/"
COMPILE_JOBS ?= 6
DEBUG_BUILD_DIR ?= "./build/"
RELEASE_BUILD_DIR ?= "./build-release/"

VSAG_CMAKE_ARGS := -DCMAKE_EXPORT_COMPILE_COMMANDS=1 -DCMAKE_COLOR_DIAGNOSTICS=ON
VSAG_CMAKE_ARGS := ${VSAG_CMAKE_ARGS} -DCMAKE_INSTALL_PREFIX=${CMAKE_INSTALL_PREFIX} -DNUM_BUILDING_JOBS=${COMPILE_JOBS}
VSAG_CMAKE_ARGS := ${VSAG_CMAKE_ARGS} -DENABLE_TESTS=ON -DENABLE_PYBINDS=ON -DENABLE_TOOLS=ON
ifdef EXTRA_DEFINED
  VSAG_CMAKE_ARGS := ${VSAG_CMAKE_ARGS} ${EXTRA_DEFINED}
endif
VSAG_CMAKE_ARGS := ${VSAG_CMAKE_ARGS} -G ${CMAKE_GENERATOR} -S.

UT_FILTER = ""
ifdef CASE
  UT_FILTER = $(CASE)
endif
UT_SHARD = ""
ifdef SHARD
  UT_SHARD = $(SHARD)
endif


.PHONY: help
help:                    ## Show the help.
	@echo "Usage: make <target>"
	@echo ""
	@echo "Targets:"
	@fgrep "##" Makefile | fgrep -v fgrep

##
## ================ development ================
.PHONY: debug
debug:                   ## Build vsag with debug options.
	cmake ${VSAG_CMAKE_ARGS} -B${DEBUG_BUILD_DIR} -DCMAKE_BUILD_TYPE=Debug -DENABLE_ASAN=OFF -DENABLE_CCACHE=ON
	cmake --build ${DEBUG_BUILD_DIR} --parallel ${COMPILE_JOBS}

.PHONY: test
test:                    ## Build and run unit tests.
	cmake ${VSAG_CMAKE_ARGS} -B${DEBUG_BUILD_DIR} -DCMAKE_BUILD_TYPE=Debug -DENABLE_CCACHE=ON
	cmake --build ${DEBUG_BUILD_DIR} --parallel ${COMPILE_JOBS}
	./build/tests/unittests -d yes ${UT_FILTER} --allow-running-no-tests ${UT_SHARD}
	./build/tests/functests -d yes ${UT_FILTER} --allow-running-no-tests ${UT_SHARD}
	./build/mockimpl/tests_mockimpl -d yes ${UT_FILTER} --allow-running-no-tests ${UT_SHARD}

.PHONY: asan
asan:                    ## Build with AddressSanitizer option.
	cmake ${VSAG_CMAKE_ARGS} -B${DEBUG_BUILD_DIR} -DCMAKE_BUILD_TYPE=Sanitize -DENABLE_ASAN=ON -DENABLE_TSAN=OFF -DENABLE_CCACHE=ON
	cmake --build ${DEBUG_BUILD_DIR} --parallel ${COMPILE_JOBS}

.PHONY: test_asan
test_asan: asan          ## Run unit tests with AddressSanitizer option.
	./build/tests/unittests -d yes ${UT_FILTER} --allow-running-no-tests ${UT_SHARD}
	./build/tests/functests -d yes ${UT_FILTER} --allow-running-no-tests ${UT_SHARD}
	./build/mockimpl/tests_mockimpl -d yes ${UT_FILTER} --allow-running-no-tests ${UT_SHARD}

.PHONY: tsan
tsan:                    ## Build with ThreadSanitizer option.
	cmake ${VSAG_CMAKE_ARGS} -B${DEBUG_BUILD_DIR} -DCMAKE_BUILD_TYPE=Sanitize -DENABLE_ASAN=OFF -DENABLE_TSAN=ON -DENABLE_CCACHE=ON
	cmake --build ${DEBUG_BUILD_DIR} --parallel ${COMPILE_JOBS}

.PHONY: test_tsan
test_tsan: tsan          ## Run unit tests with ThreadSanitizer option.
	./build/tests/unittests -d yes ${UT_FILTER} --allow-running-no-tests ${UT_SHARD}
	./build/tests/functests -d yes ${UT_FILTER} --allow-running-no-tests ${UT_SHARD}
	./build/mockimpl/tests_mockimpl -d yes ${UT_FILTER} --allow-running-no-tests ${UT_SHARD}

.PHONY: clean
clean:                   ## Clear build/ directory.
	rm -rf ${DEBUG_BUILD_DIR}/*

##
## ================ integration ================
.PHONY: fmt
fmt:                     ## Format codes.
	@./scripts/format-cpp.sh

.PHONY: cov
cov:                     ## Build unit tests with code coverage enabled.
	cmake ${VSAG_CMAKE_ARGS} -B${DEBUG_BUILD_DIR} -DCMAKE_BUILD_TYPE=Debug -DENABLE_COVERAGE=ON -DENABLE_CCACHE=ON -DENABLE_ASAN=OFF
	cmake --build ${DEBUG_BUILD_DIR} --parallel ${COMPILE_JOBS}

.PHONEY: lint
lint:                    ## Check coding styles defined in `.clang-tidy`.
	@./scripts/linters/run-clang-tidy.py -p build-release/ -use-color -source-filter '^.*vsag\/src.*(?<!_test)\.cpp$$' -j ${COMPILE_JOBS}

.PHONEY: fix-lint
fix-lint:                ## Fix coding style issues in-place via clang-apply-replacements, use it be careful!!!
	@./scripts/linters/run-clang-tidy.py -p build-release/ -use-color -source-filter '^.*vsag\/src.*(?<!_test)\.cpp$$' -j ${COMPILE_JOBS} -fix

.PHONY: test_parallel
test_parallel:           ## Run all tests parallel (used in CI).
	cmake ${VSAG_CMAKE_ARGS} -B${DEBUG_BUILD_DIR} -DCMAKE_BUILD_TYPE=Sanitize -DENABLE_ASAN=OFF -DENABLE_CCACHE=OFF
	cmake --build ${DEBUG_BUILD_DIR} --parallel ${COMPILE_JOBS}
	@./scripts/test_parallel_bg.sh
	./build/mockimpl/tests_mockimpl -d yes ${UT_FILTER} --allow-running-no-tests ${UT_SHARD}

.PHONY: test_asan_parallel
test_asan_parallel: asan ## Run unit tests parallel with AddressSanitizer option.
	@./scripts/test_parallel_bg.sh
	./build/mockimpl/tests_mockimpl -d yes ${UT_FILTER} --allow-running-no-tests ${UT_SHARD}

.PHONY: test_tsan_parallel
test_tsan_parallel: tsan ## Run unit tests parallel with ThreadSanitizer option.
	@./scripts/test_parallel_bg.sh
	./build/mockimpl/tests_mockimpl -d yes ${UT_FILTER} --allow-running-no-tests ${UT_SHARD}

##
## ================ distribution ================
.PHONY: release
release:                 ## Build vsag with release options.
	cmake ${VSAG_CMAKE_ARGS} -B${RELEASE_BUILD_DIR} -DCMAKE_BUILD_TYPE=Release
	cmake --build ${RELEASE_BUILD_DIR} --parallel ${COMPILE_JOBS}

.PHONY: run-dist-tests
run-dist-tests:          ## Run distribution tests.
	@echo "running tests..."
	@${RELEASE_BUILD_DIR}/tests/unittests -d yes "~[daily]"
	@${RELEASE_BUILD_DIR}/tests/functests -d yes "~[daily]"
	@${RELEASE_BUILD_DIR}/mockimpl/tests_mockimpl -d yes "~[daily]"

.PHONY: dist-old-abi
dist-pre-cxx11-abi:      ## Build vsag with distribution options.
	echo "building dist-pre-cxx11-abi..."
	cmake ${VSAG_CMAKE_ARGS} -B${RELEASE_BUILD_DIR} -DCMAKE_BUILD_TYPE=Release -DENABLE_INTEL_MKL=off -DENABLE_CXX11_ABI=off -DENABLE_LIBCXX=off
	cmake --build ${RELEASE_BUILD_DIR} --parallel ${COMPILE_JOBS}
	$(MAKE) run-dist-tests

.PHONY: dist-cxx11-abi
dist-cxx11-abi:          ## Build vsag with distribution options.
	echo "building dist-cxx11-abi..."
	cmake ${VSAG_CMAKE_ARGS} -B${RELEASE_BUILD_DIR} -DCMAKE_BUILD_TYPE=Release -DENABLE_INTEL_MKL=off -DENABLE_CXX11_ABI=on -DENABLE_LIBCXX=off
	cmake --build ${RELEASE_BUILD_DIR} --parallel ${COMPILE_JOBS}
	$(MAKE) run-dist-tests

.PHONY: dist-libcxx
dist-libcxx:             ## Build vsag using libc++.
	cmake ${VSAG_CMAKE_ARGS} -B${RELEASE_BUILD_DIR} -DCMAKE_BUILD_TYPE=Release -DENABLE_LIBCXX=on
	cmake --build ${RELEASE_BUILD_DIR} --parallel ${COMPILE_JOBS}

PY_VERSION ?= 3.10

.PHONY: pyvsag pyvsag-all

pyvsag:                  ## Build a specific Python version wheel. Usage: make pyvsag PY_VERSION=3.10
	@echo "Building wheel for Python $(PY_VERSION)..."
	bash ./scripts/local_build_wheel.sh $(PY_VERSION)

pyvsag-all:              ## Build wheels for all supported versions. Usage: make pyvsag-all
	@echo "Building wheels for all supported versions..."
	bash ./scripts/local_build_wheel.sh

.PHONY: clean-release
clean-release:           ## Clear build-release/ directory.
	rm -rf ${RELEASE_BUILD_DIR}/*

.PHONY: install
install:                 ## Build and install the release version of vsag.
	cmake --install ${RELEASE_BUILD_DIR}/
