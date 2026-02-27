
# Copyright 2024-present the vsag project
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


#  unittests
add_executable (unittests
        test_main.cpp
        fixtures/fixtures.cpp
        fixtures/test_logger.cpp
)

if (APPLE)
        target_link_libraries (unittests
                        PRIVATE
                        Catch2::Catch2 
                        simd_test vsag_test algorithm_test factory_test attr_test
                        datacell_test quantizer_test storage_test io_test utils_test impl_test
                        vsag
        )
        # Ensure static test libraries are not dead-stripped on macOS.
        target_link_options (unittests PRIVATE
        "-Wl,-force_load,$<TARGET_FILE:simd_test>"
        "-Wl,-force_load,$<TARGET_FILE:vsag_test>"
        "-Wl,-force_load,$<TARGET_FILE:algorithm_test>"
        "-Wl,-force_load,$<TARGET_FILE:factory_test>"
        "-Wl,-force_load,$<TARGET_FILE:attr_test>"
        "-Wl,-force_load,$<TARGET_FILE:datacell_test>"
        "-Wl,-force_load,$<TARGET_FILE:quantizer_test>"
        "-Wl,-force_load,$<TARGET_FILE:storage_test>"
        "-Wl,-force_load,$<TARGET_FILE:io_test>"
        "-Wl,-force_load,$<TARGET_FILE:utils_test>"
        "-Wl,-force_load,$<TARGET_FILE:impl_test>"
        )
else()
        target_link_libraries (unittests
                        PRIVATE
                        Catch2::Catch2 
                        "-Wl,--whole-archive"
                        simd_test vsag_test algorithm_test factory_test attr_test
                        datacell_test quantizer_test storage_test io_test utils_test impl_test
                        "-Wl,--no-whole-archive"
                        vsag
        )
endif()

add_dependencies (unittests spdlog Catch2 vsag)

# function tests
file (GLOB FUNCTION_TESTS "test_*.cpp")
add_executable (functests
        ${FUNCTION_TESTS}
        fixtures/fixtures.cpp
        fixtures/test_dataset.cpp
        fixtures/test_dataset_pool.cpp
        fixtures/test_logger.cpp
)
target_include_directories (functests PRIVATE
        ${CMAKE_CURRENT_BINARY_DIR}/spdlog/install/include
        ${HDF5_INCLUDE_DIRS})
target_link_libraries (functests PRIVATE Catch2::Catch2 vsag simd)
add_dependencies (functests spdlog Catch2)

# --- Module-level unit test executables (directory-level split) ---
if (ENABLE_TESTS)
    set(UNIT_TEST_MODULES
        simd
        vsag
        algorithm
        factory
        attr
        datacell
        quantizer
        storage
        io
        utils
        impl
    )

    foreach(mod ${UNIT_TEST_MODULES})
        # Gather test sources from the module's src directory
        file(GLOB_RECURSE MODULE_TEST_SOURCES RELATIVE ${CMAKE_SOURCE_DIR}
            "src/${mod}/*_test.cpp"
            "src/${mod}/*/*_test.cpp")

        if (MODULE_TEST_SOURCES)
            set(target "${mod}_unittests")
            # create per-module executable compiling only that module's tests
            add_executable(${target}
                ${MODULE_TEST_SOURCES}
                test_main.cpp
                fixtures/fixtures.cpp
                fixtures/test_logger.cpp
            )

            target_include_directories(${target} PRIVATE ${CMAKE_SOURCE_DIR}/include ${CMAKE_SOURCE_DIR}/src)
            target_link_libraries(${target} PRIVATE Catch2::Catch2 vsag)
            if (TARGET ${mod})
                target_link_libraries(${target} PRIVATE ${mod})
            endif()

            # If a static test library for the module exists (e.g. simd_test),
            # link it into the per-module executable and ensure its objects
            # are not dead-stripped. On macOS use -force_load, on other platforms
            # use --whole-archive/--no-whole-archive around the test archive.
            if (TARGET ${mod}_test)
                if (APPLE)
                    # Ensure static test libraries are not dead-stripped on macOS
                    target_link_options(${target} PRIVATE
                        "-Wl,-force_load,$<TARGET_FILE:${mod}_test>"
                    )
                    # Also add the test target as a normal library so CMake knows
                    # the dependency and the linker sees the file.
                    target_link_libraries(${target} PRIVATE ${mod}_test)
                else()
                    target_link_libraries(${target} PRIVATE
                        "-Wl,--whole-archive" ${mod}_test "-Wl,--no-whole-archive"
                    )
                endif()
                add_dependencies(${target} ${mod}_test)
            endif()

            add_dependencies(${target} spdlog Catch2 vsag)

            # Register with CTest so ctest can drive per-module runs
            include(CTest)
            add_test(NAME ${target} COMMAND ${target} -d yes)
            set_tests_properties(${target} PROPERTIES TIMEOUT 600)
        endif()
    endforeach()
endif()
