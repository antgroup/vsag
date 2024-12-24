
CMAKE_GENERATOR ?= "Unix Makefiles"
CMAKE_INSTALL_PREFIX ?= "/usr/local/"
COMPILE_JOBS ?= 6
DEBUG_BUILD_DIR ?= "./build/"
RELEASE_BUILD_DIR ?= "./build-release/"
VSAG_CMAKE_ARGS = -DCMAKE_EXPORT_COMPILE_COMMANDS=1 -DCMAKE_INSTALL_PREFIX=${CMAKE_INSTALL_PREFIX} -DNUM_BUILDING_JOBS=${COMPILE_JOBS} -DENABLE_TESTS=1 -DENABLE_PYBINDS=1 -G ${CMAKE_GENERATOR} -S.
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

# ================= development part =================
.PHONY: debug
debug:                   ## Build vsag with debug options.
	cmake ${VSAG_CMAKE_ARGS} -B${DEBUG_BUILD_DIR} -DCMAKE_BUILD_TYPE=Debug -DENABLE_CCACHE=ON -DENABLE_ASAN=OFF
	cmake --build ${DEBUG_BUILD_DIR} --parallel ${COMPILE_JOBS}

.PHONY: fmt
fmt:                     ## Format codes.
	find include/ -iname "*.h" -o -iname "*.cpp" | xargs clang-format -i
	find src/ -iname "*.h" -o -iname "*.cpp" | xargs clang-format -i
	find python_bindings/ -iname "*.h" -o -iname "*.cpp" | xargs clang-format -i
	find examples/cpp/ -iname "*.h" -o -iname "*.cpp" | xargs clang-format -i
	find mockimpl/ -iname "*.h" -o -iname "*.cpp" | xargs clang-format -i
	find tests/ -iname "*.h" -o -iname "*.cpp" | xargs clang-format -i
	find tools/ -iname "*.h" -o -iname "*.cpp" | xargs clang-format -i

.PHONY: test
test:                    ## Build and run unit tests.
	cmake ${VSAG_CMAKE_ARGS} -B${DEBUG_BUILD_DIR} -DCMAKE_BUILD_TYPE=Debug -DENABLE_CCACHE=ON
	cmake --build ${DEBUG_BUILD_DIR} --parallel ${COMPILE_JOBS}
	./build/tests/unittests -d yes ${UT_FILTER} --allow-running-no-tests ${UT_SHARD}
	./build/tests/functests -d yes ${UT_FILTER} --allow-running-no-tests ${UT_SHARD}
	./build/mockimpl/tests_mockimpl -d yes ${UT_FILTER} --allow-running-no-tests ${UT_SHARD}

.PHONY: test_parallel
test_parallel: debug
	@./scripts/test_parallel_bg.sh
	./build/mockimpl/tests_mockimpl -d yes ${UT_FILTER} --allow-running-no-tests ${UT_SHARD}

.PHONY: asan
asan:                    ## Build with AddressSanitizer option.
	cmake ${VSAG_CMAKE_ARGS} -B${DEBUG_BUILD_DIR} -DCMAKE_BUILD_TYPE=Debug -DENABLE_ASAN=ON -DENABLE_CCACHE=ON
	cmake --build ${DEBUG_BUILD_DIR} --parallel ${COMPILE_JOBS}

.PHONY: test_asan_parallel
test_asan_parallel: asan ## Run unit tests parallel with AddressSanitizer option.
	@./scripts/test_parallel_bg.sh
	./build/mockimpl/tests_mockimpl -d yes ${UT_FILTER} --allow-running-no-tests ${UT_SHARD}

.PHONY: test_asan
test_asan: asan          ## Run unit tests with AddressSanitizer option.
	./build/tests/unittests -d yes ${UT_FILTER} --allow-running-no-tests ${UT_SHARD}
	./build/tests/functests -d yes ${UT_FILTER} --allow-running-no-tests ${UT_SHARD}
	./build/mockimpl/tests_mockimpl -d yes ${UT_FILTER} --allow-running-no-tests ${UT_SHARD}

.PHONY: tsan
tsan:                    ## Build with ThreadSanitizer option.
	cmake ${VSAG_CMAKE_ARGS} -B${DEBUG_BUILD_DIR} -DCMAKE_BUILD_TYPE=Debug -DENABLE_TSAN=ON -DENABLE_CCACHE=ON
	cmake --build ${DEBUG_BUILD_DIR} --parallel ${COMPILE_JOBS}

.PHONY: test_tsan
test_tsan: tsan          ## Run unit tests with ThreadSanitizer option.
	./build/tests/unittests -d yes ${UT_FILTER} --allow-running-no-tests ${UT_SHARD}
	./build/tests/functests -d yes ${UT_FILTER} --allow-running-no-tests ${UT_SHARD}
	./build/mockimpl/tests_mockimpl -d yes ${UT_FILTER} --allow-running-no-tests ${UT_SHARD}

.PHONY: cov     # Build unit tests with code coverage enabled.
cov:
	cmake ${VSAG_CMAKE_ARGS} -B${DEBUG_BUILD_DIR} -DCMAKE_BUILD_TYPE=Debug -DENABLE_COVERAGE=ON -DENABLE_CCACHE=ON
	cmake --build ${DEBUG_BUILD_DIR} --parallel ${COMPILE_JOBS}

.PHONY: test_cov
test_cov: cov            ## Build and run unit tests with code coverage enabled.
	./build/tests/unittests -d yes ${UT_FILTER} --allow-running-no-tests ${UT_SHARD}
	./build/tests/functests -d yes ${UT_FILTER} --allow-running-no-tests ${UT_SHARD}
	./build/mockimpl/tests_mockimpl -d yes ${UT_FILTER} --allow-running-no-tests ${UT_SHARD}
	bash scripts/aci/collect_cpp_coverage.sh
	genhtml --output-directory testresult/coverage/html testresult/coverage/coverage.info --ignore-errors inconsistent,inconsistent

.PHONY: clean
clean:                   ## Clear build/ directory.
	rm -rf ${DEBUG_BUILD_DIR}/*

# ================= distribution part =================
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

.PHONY: install
install:                 ## Build and install the release version of vsag.
	cmake --install ${RELEASE_BUILD_DIR}/


PARAM1 := "-DNUM_BUILDING_JOBS=${COMPILE_JOBS} -DENABLE_PYBINDS=1 -S. -B${RELEASE_BUILD_DIR} -DCMAKE_BUILD_TYPE=Release"
PARAM2 := "--build ${RELEASE_BUILD_DIR} --parallel ${COMPILE_JOBS}"
PARAM3 := "${RELEASE_BUILD_DIR}"

.PHONY: pyvsag           ## Build pyvsag wheel
pyvsag:
	bash ./scripts/build_pyvsag_multiple_version.sh $(PARAM1) $(PARAM2) $(PARAM3)
