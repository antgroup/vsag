# Copyright 2026-present the vsag project
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

if (NOT DEFINED VSAG_SOURCE_DIR)
    message (FATAL_ERROR "VSAG_SOURCE_DIR is required")
endif ()

file (READ "${VSAG_SOURCE_DIR}/tests/CMakeLists.txt" test_cmake)

function (require_text content text description)
    string (FIND "${content}" "${text}" position)
    if (position EQUAL -1)
        message (FATAL_ERROR "Missing ${description}")
    endif ()
endfunction ()

set (expected_modules simd common algorithm factory attr datacell layout quantization storage io
                      utils impl)
set (expected_libraries simd_test vsag_test algorithm_test factory_test attr_test datacell_test
                        layout_test quantizer_test storage_test io_test utils_test impl_test)

foreach (module IN LISTS expected_modules)
    require_text ("${test_cmake}" "        ${module}\n" "${module} unit-test module")
endforeach ()
foreach (library IN LISTS expected_libraries)
    require_text ("${test_cmake}" "        ${library}\n" "${library} aggregate input")
endforeach ()

require_text ("${test_cmake}"
              [=[foreach (_module _test_library IN ZIP_LISTS VSAG_UNITTEST_MODULES VSAG_UNITTEST_LIBRARIES)]=]
              "module-to-library mapping")
require_text ("${test_cmake}" "list (LENGTH VSAG_UNITTEST_MODULES _module_count)"
              "module mapping length check")
require_text ("${test_cmake}" "if (NOT _module_count EQUAL _library_count)"
              "module mapping mismatch failure")
require_text ("${test_cmake}" [=[vsag_add_unittest_executable (unittests_${_module}]=]
              "module executable creation")
require_text ("${test_cmake}" "            EXCLUDE_FROM_ALL\n"
              "opt-in module executables")
require_text ("${test_cmake}"
              "vsag_add_unittest_executable (unittests\n        DEPEND_ON_SHARED_VSAG\n"
              "aggregate compatibility executable")
require_text ("${test_cmake}" "    if (ARG_DEPEND_ON_SHARED_VSAG)\n"
              "selective shared-library dependency")
require_text ("${test_cmake}" [=["-Wl,-force_load,$<TARGET_FILE:${_test_library}>"]=]
              "Apple registration preservation")
require_text ("${test_cmake}" [=["-Wl,--whole-archive"]=]
              "GNU registration preservation")

foreach (module IN LISTS expected_modules)
    execute_process (
        COMMAND make --no-print-directory --dry-run test-module MODULE=${module}
                DEBUG_BUILD_DIR=./module-test-build
        WORKING_DIRECTORY "${VSAG_SOURCE_DIR}"
        RESULT_VARIABLE make_result
        OUTPUT_VARIABLE make_output
        ERROR_VARIABLE make_error)
    if (NOT make_result EQUAL 0)
        message (FATAL_ERROR "make test-module MODULE=${module} failed:\n${make_error}")
    endif ()
    require_text ("${make_output}" "--target unittests_${module}"
                  "${module} module build command")
    require_text ("${make_output}" "./module-test-build/tests/unittests_${module}"
                  "${module} module execution command")
endforeach ()

message (STATUS "Module unit-test target topology checks passed")
