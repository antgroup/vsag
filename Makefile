
CMAKE_GENERATOR ?= "Unix Makefiles"
CMAKE_INSTALL_PREFIX ?= "/usr/local/"
COMPILE_JOBS ?= 6
DEBUG_BUILD_DIR ?= "./build/"
RELEASE_BUILD_DIR ?= "./build-release/"

VSAG_CMAKE_ARGS := -DCMAKE_EXPORT_COMPILE_COMMANDS=1
VSAG_CMAKE_ARGS := ${VSAG_CMAKE_ARGS} -DCMAKE_INSTALL_PREFIX=${CMAKE_INSTALL_PREFIX} -DNUM_BUILDING_JOBS=${COMPILE_JOBS}
VSAG_CMAKE_ARGS := ${VSAG_CMAKE_ARGS} -DENABLE_TESTS=1 -DENABLE_PYBINDS=1 -G ${CMAKE_GENERATOR} -S.

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
	cmake ${VSAG_CMAKE_ARGS} -B${DEBUG_BUILD_DIR} -DCMAKE_BUILD_TYPE=Debug -DENABLE_ASAN=ON -DENABLE_TSAN=OFF -DENABLE_CCACHE=ON
	cmake --build ${DEBUG_BUILD_DIR} --parallel ${COMPILE_JOBS}

.PHONY: test_asan
test_asan: asan          ## Run unit tests with AddressSanitizer option.
	./build/tests/unittests -d yes ${UT_FILTER} --allow-running-no-tests ${UT_SHARD}
	./build/tests/functests -d yes ${UT_FILTER} --allow-running-no-tests ${UT_SHARD}
	./build/mockimpl/tests_mockimpl -d yes ${UT_FILTER} --allow-running-no-tests ${UT_SHARD}

.PHONY: tsan
tsan:                    ## Build with ThreadSanitizer option.
	cmake ${VSAG_CMAKE_ARGS} -B${DEBUG_BUILD_DIR} -DCMAKE_BUILD_TYPE=Debug -DENABLE_ASAN=OFF -DENABLE_TSAN=ON -DENABLE_CCACHE=ON
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
lint:                    ## Run lint.
	@./scripts/linters/run-clang-tidy.py -p build/ -use-color -source-filter '^.*vsag\/src.*(?<!_test)\.cpp$$'

.PHONY: test_parallel
test_parallel: debug     ## Run all tests parallel (used in CI).
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

.PHONY: distribution
distribution:            ## Build vsag with distribution options.
	cmake ${VSAG_CMAKE_ARGS} -B${RELEASE_BUILD_DIR} -DCMAKE_BUILD_TYPE=Release -DENABLE_CXX11_ABI=off -DENABLE_LIBCXX=off
	cmake --build ${RELEASE_BUILD_DIR} --parallel ${COMPILE_JOBS}

.PHONY: libcxx
libcxx:                  ## Build vsag using libc++.
	cmake ${VSAG_CMAKE_ARGS} -B${RELEASE_BUILD_DIR} -DCMAKE_BUILD_TYPE=Release -DENABLE_LIBCXX=on
	cmake --build ${RELEASE_BUILD_DIR} --parallel ${COMPILE_JOBS}

PARAM1 := "-DNUM_BUILDING_JOBS=${COMPILE_JOBS} -DENABLE_PYBINDS=1 -S. -B${RELEASE_BUILD_DIR} -DCMAKE_BUILD_TYPE=Release"
PARAM2 := "--build ${RELEASE_BUILD_DIR} --parallel ${COMPILE_JOBS}"
PARAM3 := "${RELEASE_BUILD_DIR}"

.PHONY: pyvsag
pyvsag:                  ## Build pyvsag wheel.
	bash ./scripts/build_pyvsag_multiple_version.sh $(PARAM1) $(PARAM2) $(PARAM3)

.PHONY: clean-release
clean-release:           ## Clear build-release/ directory.
	rm -rf ${RELEASE_BUILD_DIR}/*

.PHONY: install
install:                 ## Build and install the release version of vsag.
	cmake --install ${RELEASE_BUILD_DIR}/
