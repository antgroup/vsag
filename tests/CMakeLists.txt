
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

# fixtures is already added by top-level CMakeLists.txt before src


# Unit tests are compiled into module-owned static libraries under src/. Keep the aggregate
# executable for CI and full validation, and expose opt-in module executables for local iteration.
set (VSAG_UNITTEST_FIXTURES
        test_main.cpp
        $<TARGET_OBJECTS:fixtures_allocator>
        $<TARGET_OBJECTS:fixtures_core>
        $<TARGET_OBJECTS:fixtures_data>
        $<TARGET_OBJECTS:fixtures_framework>
)

set (VSAG_UNITTEST_MODULES
        simd
        common
        algorithm
        factory
        attr
        datacell
        layout
        quantization
        storage
        io
        utils
        impl
)

set (VSAG_UNITTEST_LIBRARIES
        simd_test
        vsag_test
        algorithm_test
        factory_test
        attr_test
        datacell_test
        layout_test
        quantizer_test
        storage_test
        io_test
        utils_test
        impl_test
)

function (vsag_add_unittest_executable target)
    set (options EXCLUDE_FROM_ALL DEPEND_ON_SHARED_VSAG)
    set (multi_value_args TEST_LIBRARIES)
    cmake_parse_arguments (ARG "${options}" "" "${multi_value_args}" ${ARGN})

    set (_exclude_from_all)
    if (ARG_EXCLUDE_FROM_ALL)
        set (_exclude_from_all EXCLUDE_FROM_ALL)
    endif ()

    add_executable (${target}
            ${_exclude_from_all}
            ${VSAG_UNITTEST_FIXTURES}
    )

    target_link_libraries (${target}
            PRIVATE
            Catch2::Catch2
    )

    if (APPLE)
        target_link_libraries (${target} PRIVATE ${ARG_TEST_LIBRARIES} vsag_static)

        # Ensure static test libraries are not dead-stripped on macOS.
        foreach (_test_library IN LISTS ARG_TEST_LIBRARIES)
            target_link_options (${target} PRIVATE
                    "-Wl,-force_load,$<TARGET_FILE:${_test_library}>"
            )
        endforeach ()
    else ()
        target_link_libraries (${target}
                PRIVATE
                "-Wl,--whole-archive"
                ${ARG_TEST_LIBRARIES}
                "-Wl,--no-whole-archive"
                vsag_static
        )
    endif ()

    add_dependencies (${target} Catch2)
    if (ARG_DEPEND_ON_SHARED_VSAG)
        add_dependencies (${target} vsag)
    endif ()
endfunction ()

list (LENGTH VSAG_UNITTEST_MODULES _module_count)
list (LENGTH VSAG_UNITTEST_LIBRARIES _library_count)
if (NOT _module_count EQUAL _library_count)
    message (FATAL_ERROR "Unit-test module and library lists must have the same length")
endif ()

foreach (_module _test_library IN ZIP_LISTS VSAG_UNITTEST_MODULES VSAG_UNITTEST_LIBRARIES)
    vsag_add_unittest_executable (unittests_${_module}
            EXCLUDE_FROM_ALL
            TEST_LIBRARIES ${_test_library}
    )
endforeach ()

vsag_add_unittest_executable (unittests
        DEPEND_ON_SHARED_VSAG
        TEST_LIBRARIES ${VSAG_UNITTEST_LIBRARIES}
)

add_executable (eval_monitor_test
        ${PROJECT_SOURCE_DIR}/tools/eval/monitor/latency_monitor_test.cpp
        ${PROJECT_SOURCE_DIR}/tools/eval/monitor/latency_monitor.cpp
        ${PROJECT_SOURCE_DIR}/tools/eval/monitor/monitor.cpp
)

target_include_directories (eval_monitor_test PRIVATE
        ${PROJECT_SOURCE_DIR}/tools/eval
)

target_link_libraries (eval_monitor_test PRIVATE
        Catch2::Catch2WithMain
        nlohmann_json::nlohmann_json
)

add_dependencies (eval_monitor_test Catch2)

# function tests
file (GLOB FUNCTION_TESTS "test_*.cpp")
file (GLOB FUNCTION_TEST_INDEX_SOURCES "test_index/*.cpp")
add_executable (functests
        ${FUNCTION_TESTS}
        ${FUNCTION_TEST_INDEX_SOURCES}
        $<TARGET_OBJECTS:fixtures_allocator>
        $<TARGET_OBJECTS:fixtures_core>
        $<TARGET_OBJECTS:fixtures_data>
        $<TARGET_OBJECTS:fixtures_framework>
)

target_include_directories (functests PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}
        ${CMAKE_CURRENT_SOURCE_DIR}/test_index
)

target_link_libraries (functests PRIVATE
        Catch2::Catch2 
        vsag 
        simd
        fmt::fmt
)
add_dependencies (functests Catch2)
