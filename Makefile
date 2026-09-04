
# Personal Configurations, Disable Some Options via Env Vars to Accelerate Building
CMAKE_GENERATOR ?= "Unix Makefiles"
CMAKE_INSTALL_PREFIX ?= "/usr/local/"
COMPILE_JOBS ?= 6
CMAKE_BUILD_ARGS ?=
DEBUG_BUILD_DIR ?= "./build/"
RELEASE_BUILD_DIR ?= "./build-release/"
LITE_BUILD_DIR ?= "./build-lite/"
LITE_RELEASE_BUILD_DIR ?= "./build-lite-release/"
PERF_RELEASE_BUILD_DIR ?= "./build-release-perf/"
VSAG_ENABLE_TESTS ?= OFF
VSAG_ENABLE_LITE ?= OFF
VSAG_ENABLE_PYBINDS ?= OFF
VSAG_ENABLE_TOOLS ?= OFF
VSAG_ENABLE_EXAMPLES ?= OFF
VSAG_ENABLE_INTEL_MKL ?= OFF
VSAG_ENABLE_LIBAIO ?= ON
VSAG_USE_SYSTEM_DEPS ?= AUTO

VSAG_CMAKE_ARGS := -DCMAKE_EXPORT_COMPILE_COMMANDS=1 -DCMAKE_COLOR_DIAGNOSTICS=ON -DENABLE_INTEL_MKL=${VSAG_ENABLE_INTEL_MKL}
VSAG_CMAKE_ARGS := ${VSAG_CMAKE_ARGS} -DENABLE_LIBAIO=${VSAG_ENABLE_LIBAIO}
VSAG_CMAKE_ARGS := ${VSAG_CMAKE_ARGS} -DVSAG_USE_SYSTEM_DEPS=${VSAG_USE_SYSTEM_DEPS}
VSAG_CMAKE_ARGS := ${VSAG_CMAKE_ARGS} -DCMAKE_INSTALL_PREFIX=${CMAKE_INSTALL_PREFIX} -DNUM_BUILDING_JOBS=${COMPILE_JOBS}
VSAG_CMAKE_ARGS := ${VSAG_CMAKE_ARGS} -DENABLE_TESTS=${VSAG_ENABLE_TESTS} -DENABLE_PYBINDS=${VSAG_ENABLE_PYBINDS}
VSAG_CMAKE_ARGS := ${VSAG_CMAKE_ARGS} -DENABLE_LITE=${VSAG_ENABLE_LITE}
VSAG_CMAKE_ARGS := ${VSAG_CMAKE_ARGS} -DENABLE_TOOLS=${VSAG_ENABLE_TOOLS} -DENABLE_EXAMPLES=${VSAG_ENABLE_EXAMPLES}
# CI checks out these sources before configuration. Explicit source overrides
# prevent FetchContent from contacting the archive URLs declared under extern/.
ifneq ($(strip ${VSAG_FETCHCONTENT_BASE_DIR}),)
  VSAG_CMAKE_ARGS := ${VSAG_CMAKE_ARGS} \
    -DFETCHCONTENT_BASE_DIR=${VSAG_FETCHCONTENT_BASE_DIR}
  VSAG_CMAKE_ARGS := ${VSAG_CMAKE_ARGS} \
    -DFETCHCONTENT_SOURCE_DIR_NLOHMANN_JSON=${VSAG_FETCHCONTENT_BASE_DIR}/nlohmann_json-src
  VSAG_CMAKE_ARGS := ${VSAG_CMAKE_ARGS} \
    -DFETCHCONTENT_SOURCE_DIR_CATCH2=${VSAG_FETCHCONTENT_BASE_DIR}/catch2-src
  VSAG_CMAKE_ARGS := ${VSAG_CMAKE_ARGS} \
    -DFETCHCONTENT_SOURCE_DIR_CPUINFO=${VSAG_FETCHCONTENT_BASE_DIR}/cpuinfo-src
  VSAG_CMAKE_ARGS := ${VSAG_CMAKE_ARGS} \
    -DFETCHCONTENT_SOURCE_DIR_FMT=${VSAG_FETCHCONTENT_BASE_DIR}/fmt-src
  VSAG_CMAKE_ARGS := ${VSAG_CMAKE_ARGS} \
    -DFETCHCONTENT_SOURCE_DIR_THREAD_POOL=${VSAG_FETCHCONTENT_BASE_DIR}/thread_pool-src
  VSAG_CMAKE_ARGS := ${VSAG_CMAKE_ARGS} \
    -DFETCHCONTENT_SOURCE_DIR_TSL=${VSAG_FETCHCONTENT_BASE_DIR}/tsl-src
  VSAG_CMAKE_ARGS := ${VSAG_CMAKE_ARGS} \
    -DFETCHCONTENT_SOURCE_DIR_ROARINGBITMAP=${VSAG_FETCHCONTENT_BASE_DIR}/roaringbitmap-src
  VSAG_CMAKE_ARGS := ${VSAG_CMAKE_ARGS} \
    -DFETCHCONTENT_SOURCE_DIR_ARGPARSE=${VSAG_FETCHCONTENT_BASE_DIR}/argparse-src
  VSAG_CMAKE_ARGS := ${VSAG_CMAKE_ARGS} \
    -DFETCHCONTENT_SOURCE_DIR_YAML-CPP=${VSAG_FETCHCONTENT_BASE_DIR}/yaml-cpp-src
  VSAG_CMAKE_ARGS := ${VSAG_CMAKE_ARGS} \
    -DFETCHCONTENT_SOURCE_DIR_TABULATE=${VSAG_FETCHCONTENT_BASE_DIR}/tabulate-src
  VSAG_CMAKE_ARGS := ${VSAG_CMAKE_ARGS} \
    -DFETCHCONTENT_SOURCE_DIR_HTTPLIB=${VSAG_FETCHCONTENT_BASE_DIR}/httplib-src
endif
ifneq ($(strip ${VSAG_THIRDPARTY_DOWNLOAD_DIR}),)
  VSAG_CMAKE_ARGS := ${VSAG_CMAKE_ARGS} \
    -DDOWNLOAD_DIR=${VSAG_THIRDPARTY_DOWNLOAD_DIR}
endif
ifdef EXTRA_DEFINED
  VSAG_CMAKE_ARGS := ${VSAG_CMAKE_ARGS} ${EXTRA_DEFINED}
endif
VSAG_CMAKE_ARGS := ${VSAG_CMAKE_ARGS} -G ${CMAKE_GENERATOR} -S.

UNAME_S := $(shell uname -s)
ifeq ($(UNAME_S),Darwin)
  VSAG_CMAKE_ARGS := ${VSAG_CMAKE_ARGS} -DENABLE_LIBAIO=OFF -DVSAG_USE_SYSTEM_OPENBLAS=AUTO
  VSAG_CMAKE_ARGS := ${VSAG_CMAKE_ARGS} -DENABLE_LIBCXX=ON -DENABLE_WERROR=OFF -DCMAKE_C_COMPILER=clang -DCMAKE_CXX_COMPILER=clang++
endif

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

.PHONY: dev
dev:                     ## Build full developer configuration.
	cmake ${VSAG_CMAKE_ARGS} -B${DEBUG_BUILD_DIR} -DCMAKE_BUILD_TYPE=Debug -DENABLE_ASAN=OFF -DENABLE_CCACHE=ON -DENABLE_TESTS=ON -DENABLE_PYBINDS=ON -DENABLE_TOOLS=ON -DENABLE_EXAMPLES=ON
	cmake --build ${DEBUG_BUILD_DIR} --parallel ${COMPILE_JOBS}

.PHONY: test
test:                    ## Build and run unit tests.
	cmake ${VSAG_CMAKE_ARGS} -B${DEBUG_BUILD_DIR} -DCMAKE_BUILD_TYPE=Debug -DENABLE_ASAN=OFF -DENABLE_CCACHE=ON -DENABLE_TESTS=ON
	cmake --build ${DEBUG_BUILD_DIR} --parallel ${COMPILE_JOBS}
	./build/tests/unittests -d yes ${UT_FILTER} --allow-running-no-tests ${UT_SHARD}
	./build/tests/functests -d yes ${UT_FILTER} --allow-running-no-tests ${UT_SHARD}
	./build/tests/eval_monitor_test -d yes ${UT_FILTER} --allow-running-no-tests ${UT_SHARD}

.PHONY: lite
lite:                    ## Build the reduced VSAG Lite library.
	cmake ${VSAG_CMAKE_ARGS} -B${LITE_BUILD_DIR} -DCMAKE_BUILD_TYPE=Release -DENABLE_LITE=ON -DENABLE_CCACHE=ON
	cmake --build ${LITE_BUILD_DIR} --parallel ${COMPILE_JOBS}

.PHONY: test-lite
test-lite:               ## Build and run the VSAG Lite smoke tests.
	cmake ${VSAG_CMAKE_ARGS} -B${LITE_BUILD_DIR} -DCMAKE_BUILD_TYPE=Debug -DENABLE_LITE=ON -DENABLE_TESTS=ON -DENABLE_CCACHE=ON
	cmake --build ${LITE_BUILD_DIR} --parallel ${COMPILE_JOBS}
	${LITE_BUILD_DIR}/tests/unittests "[lite]" -d yes --allow-running-no-tests ${UT_SHARD}
	${LITE_BUILD_DIR}/tests/functests "[lite]" -d yes --allow-running-no-tests ${UT_SHARD}
	${LITE_BUILD_DIR}/tests/eval_monitor_test -d yes --allow-running-no-tests ${UT_SHARD}

.PHONY: test-cmake
test-cmake:              ## Run focused CMake helper tests.
	cmake -DVSAG_SOURCE_DIR=${CURDIR} -P tests/cmake/thirdparty_override_test.cmake
	cmake -DVSAG_SOURCE_DIR=${CURDIR} -P tests/cmake/release_build_modes_test.cmake
	cmake -DVSAG_SOURCE_DIR=${CURDIR} -P tests/cmake/object_library_test.cmake
	cmake -DVSAG_SOURCE_DIR=${CURDIR} -P tests/cmake/compile_flag_scope_test.cmake

.PHONY: asan configure-asan build-asan
asan:                    ## Build with AddressSanitizer option.
	$(MAKE) configure-asan
	$(MAKE) build-asan

configure-asan:
	cmake ${VSAG_CMAKE_ARGS} -B${DEBUG_BUILD_DIR} -DCMAKE_BUILD_TYPE=Sanitize -DENABLE_ASAN=ON -DENABLE_TSAN=OFF -DENABLE_CCACHE=ON -DENABLE_TESTS=ON

build-asan:
	cmake --build ${DEBUG_BUILD_DIR} --parallel ${COMPILE_JOBS} -- ${CMAKE_BUILD_ARGS}

.PHONY: test_asan
test_asan: asan          ## Run unit tests with AddressSanitizer option.
	./build/tests/unittests -d yes ${UT_FILTER} --allow-running-no-tests ${UT_SHARD}
	./build/tests/functests -d yes ${UT_FILTER} --allow-running-no-tests ${UT_SHARD}

.PHONY: tsan
tsan:                    ## Build with ThreadSanitizer option.
	cmake ${VSAG_CMAKE_ARGS} -B${DEBUG_BUILD_DIR} -DCMAKE_BUILD_TYPE=Sanitize -DENABLE_ASAN=OFF -DENABLE_TSAN=ON -DENABLE_CCACHE=ON -DENABLE_TESTS=ON
	cmake --build ${DEBUG_BUILD_DIR} --parallel ${COMPILE_JOBS}

.PHONY: test_tsan
test_tsan: tsan          ## Run unit tests with ThreadSanitizer option.
	./build/tests/unittests -d yes ${UT_FILTER} --allow-running-no-tests ${UT_SHARD}
	./build/tests/functests -d yes ${UT_FILTER} --allow-running-no-tests ${UT_SHARD}

.PHONY: clean
clean:                   ## Clear build/ directory.
	rm -rf ${DEBUG_BUILD_DIR} && mkdir -p ${DEBUG_BUILD_DIR}

##
## ================ integration ================
.PHONY: fmt
fmt:                     ## Format codes.
	@./scripts/format/format-cpp.sh

.PHONY: cov
cov:                     ## Build unit tests with code coverage enabled.
	cmake ${VSAG_CMAKE_ARGS} -B${DEBUG_BUILD_DIR} -DCMAKE_BUILD_TYPE=Debug -DENABLE_COVERAGE=ON -DENABLE_CCACHE=ON -DENABLE_ASAN=OFF -DENABLE_TESTS=ON
	cmake --build ${DEBUG_BUILD_DIR} --parallel ${COMPILE_JOBS}

.PHONEY: lint
lint:                    ## Check coding styles defined in `.clang-tidy`.
	@./scripts/linters/check-struct-names.py
	@./scripts/linters/run-clang-tidy-15.sh -p build-release/ -use-color -source-filter '^.*vsag\/src.*(?<!_test)\.cpp$$' -j ${COMPILE_JOBS}

.PHONEY: fix-lint
fix-lint:                ## Fix coding style issues in-place via clang-apply-replacements, use it be careful!!!
	@./scripts/linters/run-clang-tidy-15.sh -p build-release/ -use-color -source-filter '^.*vsag\/src.*(?<!_test)\.cpp$$' -j ${COMPILE_JOBS} -fix

.PHONY: test_parallel
test_parallel:           ## Run all tests parallel (used in CI).
	cmake ${VSAG_CMAKE_ARGS} -B${DEBUG_BUILD_DIR} -DCMAKE_BUILD_TYPE=Sanitize -DENABLE_ASAN=OFF -DENABLE_CCACHE=OFF -DENABLE_TESTS=ON
	cmake --build ${DEBUG_BUILD_DIR} --parallel ${COMPILE_JOBS}
	@./scripts/testing/test_parallel_bg.sh

.PHONY: test_asan_parallel
test_asan_parallel: asan ## Run unit tests parallel with AddressSanitizer option.
	@./scripts/testing/test_parallel_bg.sh

.PHONY: test_tsan_parallel
test_tsan_parallel: tsan ## Run unit tests parallel with ThreadSanitizer option.
	@./scripts/testing/test_parallel_bg.sh

##
## ================ distribution ================
release dist-pre-cxx11-abi dist-cxx11-abi dist-libcxx: VSAG_ENABLE_CCACHE ?= OFF

.PHONY: release
release:                 ## Build reproducible release/package output (ccache off by default).
	cmake ${VSAG_CMAKE_ARGS} -B${RELEASE_BUILD_DIR} -DCMAKE_BUILD_TYPE=Release -DENABLE_CCACHE=${VSAG_ENABLE_CCACHE}
	cmake --build ${RELEASE_BUILD_DIR} --parallel ${COMPILE_JOBS}

.PHONY: release-perf
release-perf: VSAG_ENABLE_CCACHE ?= ON
release-perf:            ## Build optimized output for iteration/benchmarks (ccache on by default).
	cmake ${VSAG_CMAKE_ARGS} -B${PERF_RELEASE_BUILD_DIR} -DCMAKE_BUILD_TYPE=Release -DENABLE_CCACHE=${VSAG_ENABLE_CCACHE}
	cmake --build ${PERF_RELEASE_BUILD_DIR} --parallel ${COMPILE_JOBS}

.PHONY: run-dist-tests
run-dist-tests:          ## Run distribution tests.
	@echo "running tests..."
	@${RELEASE_BUILD_DIR}/tests/unittests -d yes "~[daily]"
	@${RELEASE_BUILD_DIR}/tests/functests -d yes "~[daily]"

.PHONY: dist-pre-cxx11-abi
dist-pre-cxx11-abi:      ## Build vsag with distribution options.
	echo "building dist-pre-cxx11-abi..."
	cmake ${VSAG_CMAKE_ARGS} -B${RELEASE_BUILD_DIR} -DCMAKE_BUILD_TYPE=Release -DENABLE_CCACHE=${VSAG_ENABLE_CCACHE} -DENABLE_INTEL_MKL=off -DENABLE_CXX11_ABI=off -DENABLE_LIBCXX=off -DENABLE_TESTS=ON
	cmake --build ${RELEASE_BUILD_DIR} --parallel ${COMPILE_JOBS}
	$(MAKE) run-dist-tests

.PHONY: dist-cxx11-abi
dist-cxx11-abi:          ## Build vsag with distribution options.
	echo "building dist-cxx11-abi..."
	cmake ${VSAG_CMAKE_ARGS} -B${RELEASE_BUILD_DIR} -DCMAKE_BUILD_TYPE=Release -DENABLE_CCACHE=${VSAG_ENABLE_CCACHE} -DENABLE_INTEL_MKL=off -DENABLE_CXX11_ABI=on -DENABLE_LIBCXX=off -DENABLE_TESTS=ON
	cmake --build ${RELEASE_BUILD_DIR} --parallel ${COMPILE_JOBS}
	$(MAKE) run-dist-tests

.PHONY: dist-libcxx
dist-libcxx:             ## Build vsag using libc++.
	cmake ${VSAG_CMAKE_ARGS} -B${RELEASE_BUILD_DIR} -DCMAKE_BUILD_TYPE=Release -DENABLE_CCACHE=${VSAG_ENABLE_CCACHE} -DENABLE_LIBCXX=on
	cmake --build ${RELEASE_BUILD_DIR} --parallel ${COMPILE_JOBS}

PY_VERSION ?= 3.10

.PHONY: pyvsag pyvsag-all

pyvsag:                  ## Build a specific Python version wheel. Usage: make pyvsag PY_VERSION=3.10
	@echo "Building wheel for Python $(PY_VERSION)..."
	bash ./scripts/python/local_build_wheel.sh $(PY_VERSION)

pyvsag-all:              ## Build wheels for all supported versions. Usage: make pyvsag-all
	@echo "Building wheels for all supported versions..."
	bash ./scripts/python/local_build_wheel.sh

.PHONY: clean-release
clean-release:           ## Clear build-release/ directory.
	rm -rf ${RELEASE_BUILD_DIR} && mkdir -p ${RELEASE_BUILD_DIR}

.PHONY: clean-release-perf
clean-release-perf:      ## Clear build-release-perf/ directory.
	rm -rf ${PERF_RELEASE_BUILD_DIR} && mkdir -p ${PERF_RELEASE_BUILD_DIR}

.PHONY: install
install:                 ## Build and install the release version of vsag.
	cmake --install ${RELEASE_BUILD_DIR}/
