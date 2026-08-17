if (NOT DEFINED VSAG_SOURCE_DIR)
    message (FATAL_ERROR "VSAG_SOURCE_DIR is required")
endif ()

include (${VSAG_SOURCE_DIR}/cmake/VSAGThirdPartyOverride.cmake)

function (assert_equal actual expected description)
    if (NOT "${actual}" STREQUAL "${expected}")
        message (FATAL_ERROR
                 "${description}: expected '${expected}', got '${actual}'")
    endif ()
endfunction ()

function (assert_matches actual pattern description)
    if (NOT "${actual}" MATCHES "${pattern}")
        message (FATAL_ERROR "${description}: '${actual}' does not match '${pattern}'")
    endif ()
endfunction ()

function (assert_not_matches actual pattern description)
    if ("${actual}" MATCHES "${pattern}")
        message (FATAL_ERROR "${description}: '${actual}' unexpectedly matches '${pattern}'")
    endif ()
endfunction ()

file (GLOB_RECURSE thirdparty_cmake_files "${VSAG_SOURCE_DIR}/extern/*.cmake")
set (override_inventory_count 0)
foreach (thirdparty_cmake_file IN LISTS thirdparty_cmake_files)
    file (READ "${thirdparty_cmake_file}" thirdparty_cmake_content)
    string (TOLOWER "${thirdparty_cmake_content}" thirdparty_cmake_content_lower)
    if (thirdparty_cmake_content_lower MATCHES
            "aliyuncs\\.com|vsagcache\\.oss|maintained by the vsag project")
        message (FATAL_ERROR
                 "Alibaba Cloud OSS cache remains in ${thirdparty_cmake_file}")
    endif ()
    if (thirdparty_cmake_content MATCHES "vsag_resolve_thirdparty_override")
        string (REGEX MATCHALL "vsag_resolve_thirdparty_override"
                override_calls "${thirdparty_cmake_content}")
        list (LENGTH override_calls override_call_count)
        math (EXPR override_inventory_count
              "${override_inventory_count} + ${override_call_count}")
    endif ()
    if (thirdparty_cmake_content MATCHES "ENV\\{VSAG_THIRDPARTY_")
        message (FATAL_ERROR
                 "Direct third-party environment lookup remains in ${thirdparty_cmake_file}")
    endif ()
endforeach ()
assert_equal ("${override_inventory_count}" "16" "1.0 override inventory size")

set (expected_paths
     antlr4/antlr4.cmake argparse/argparse.cmake boost/boost.cmake catch2/catch2.cmake
     cpuinfo/cpuinfo.cmake fmt/fmt.cmake hdf5/hdf5.cmake httplib/httplib.cmake
     json/json.cmake openblas/openblas.cmake pybind11/pybind11.cmake
     roaringbitmap/roaringbitmap.cmake tabulate/tabulate.cmake
     thread_pool/thread_pool.cmake tsl/tsl.cmake yaml-cpp/yaml-cpp.cmake)
set (expected_dependencies
     ANTLR4 ARGPARSE BOOST CATCH2 CPUINFO FMT HDF5 HTTPLIB JSON OPENBLAS PYBIND11
     ROARINGBITMAP TABULATE THREAD_POOL TSL YAML_CPP)
set (expected_pins
     4.13.2 v3.1 1.67.0 v3.7.1 ca678952a9a8eaa6de112d154e8e104b22f9ab3f
     10.2.1 hdf5_1.14.4 v0.35.0 v3.11.3 v0.3.24 v2.11.1 v3.0.1
     3a58301067bbc03da89ae5a51b3e05b7da719d38
     3507796e172d36555b47d6191f170823d9f6b12c v1.4.0 yaml-cpp-0.9.0)
set (expected_variables
     VSAG_THIRDPARTY_ANTLR4_4_13_2 VSAG_THIRDPARTY_ARGPARSE_3_1
     VSAG_THIRDPARTY_BOOST_1_67_0 VSAG_THIRDPARTY_CATCH2_3_7_1
     VSAG_THIRDPARTY_CPUINFO_COMMIT_CA678952A9A8 VSAG_THIRDPARTY_FMT_10_2_1
     VSAG_THIRDPARTY_HDF5_1_14_4 VSAG_THIRDPARTY_HTTPLIB_0_35_0
     VSAG_THIRDPARTY_JSON_3_11_3 VSAG_THIRDPARTY_OPENBLAS_0_3_24
     VSAG_THIRDPARTY_PYBIND11_2_11_1 VSAG_THIRDPARTY_ROARINGBITMAP_3_0_1
     VSAG_THIRDPARTY_TABULATE_COMMIT_3A58301067BB
     VSAG_THIRDPARTY_THREAD_POOL_COMMIT_3507796E172D VSAG_THIRDPARTY_TSL_1_4_0
     VSAG_THIRDPARTY_YAML_CPP_0_9_0)
set (expected_upstreams
     github.com/antlr/antlr4 github.com/p-ranav/argparse archives.boost.io/release/1.67.0
     github.com/catchorg/Catch2 github.com/pytorch/cpuinfo github.com/fmtlib/fmt
     github.com/HDFGroup/hdf5 github.com/yhirose/cpp-httplib github.com/nlohmann/json
     github.com/OpenMathLib/OpenBLAS github.com/pybind/pybind11
     github.com/RoaringBitmap/CRoaring github.com/p-ranav/tabulate
     github.com/log4cplus/ThreadPool github.com/Tessil/robin-map github.com/jbeder/yaml-cpp)
list (LENGTH expected_paths expected_override_count)
assert_equal ("${expected_override_count}" "16" "expected 1.0 override inventory size")

foreach (override_index RANGE 0 15)
    list (GET expected_paths ${override_index} relative_path)
    list (GET expected_dependencies ${override_index} dependency)
    list (GET expected_pins ${override_index} pin)
    list (GET expected_variables ${override_index} expected_variable)
    list (GET expected_upstreams ${override_index} upstream_host_path)
    file (READ "${VSAG_SOURCE_DIR}/extern/${relative_path}" dependency_content)
    string (REGEX REPLACE "[ \t\r\n]+" " " dependency_content "${dependency_content}")
    string (REPLACE "( " "(" dependency_content "${dependency_content}")
    set (expected_call "vsag_resolve_thirdparty_override (${dependency} ${pin} ")
    string (FIND "${dependency_content}" "${expected_call}" expected_call_position)
    if (expected_call_position EQUAL -1)
        message (FATAL_ERROR
                 "Missing 1.0 override mapping '${dependency} ${pin}' in ${relative_path}")
    endif ()
    string (FIND "${dependency_content}" "${upstream_host_path}" upstream_position)
    if (upstream_position EQUAL -1)
        message (FATAL_ERROR "Authoritative upstream URL is missing in ${relative_path}")
    endif ()
    if (NOT dependency_content MATCHES "URL_HASH")
        message (FATAL_ERROR "URL hash check is missing in ${relative_path}")
    endif ()
    vsag_thirdparty_pinned_variable ("${dependency}" "${pin}" actual_variable)
    assert_equal ("${actual_variable}" "${expected_variable}"
                  "1.0 pinned variable for ${dependency}")
endforeach ()

function (run_fixture pinned legacy output_variable)
    set (pinned_variable VSAG_THIRDPARTY_FMT_10_2_1)
    set (legacy_variable VSAG_THIRDPARTY_FMT)
    unset (ENV{${pinned_variable}})
    unset (ENV{${legacy_variable}})
    if (NOT "${pinned}" STREQUAL "")
        set (ENV{${pinned_variable}} "${pinned}")
    endif ()
    if (NOT "${legacy}" STREQUAL "")
        set (ENV{${legacy_variable}} "${legacy}")
    endif ()

    execute_process (
        COMMAND ${CMAKE_COMMAND} -DVSAG_SOURCE_DIR=${VSAG_SOURCE_DIR}
                -P ${VSAG_SOURCE_DIR}/tests/cmake/thirdparty_override_fixture.cmake
        RESULT_VARIABLE result
        OUTPUT_VARIABLE stdout
        ERROR_VARIABLE stderr)
    if (NOT result EQUAL 0)
        message (FATAL_ERROR "Fixture failed (${result}):\n${stdout}\n${stderr}")
    endif ()
    set (${output_variable} "${stdout}\n${stderr}" PARENT_SCOPE)
endfunction ()

vsag_thirdparty_pinned_variable (FMT 10.2.1 actual)
assert_equal ("${actual}" "VSAG_THIRDPARTY_FMT_10_2_1" "semantic version")
vsag_thirdparty_pinned_variable (FMT v10.2.1 actual)
assert_equal ("${actual}" "VSAG_THIRDPARTY_FMT_10_2_1" "leading v")
vsag_thirdparty_pinned_variable (HDF5 hdf5_1.14.4 actual)
assert_equal ("${actual}" "VSAG_THIRDPARTY_HDF5_1_14_4" "decorated version")
vsag_thirdparty_pinned_variable (YAML_CPP yaml-cpp-0.9.0 actual)
assert_equal ("${actual}" "VSAG_THIRDPARTY_YAML_CPP_0_9_0" "separator normalization")

vsag_thirdparty_pinned_variable (EXAMPLE release/foo actual)
assert_equal (
    "${actual}" "VSAG_THIRDPARTY_EXAMPLE_TAG_RELEASE_FOO_HD19D5EEB0FE0" "tag pin")
vsag_thirdparty_pinned_variable (EXAMPLE Release/Foo actual)
assert_matches (
    "${actual}" "^VSAG_THIRDPARTY_EXAMPLE_TAG_RELEASE_FOO_H[0-9A-F]+$"
    "case-sensitive tag digest")
string (LENGTH "${actual}" actual_length)
assert_equal ("${actual_length}" "53" "12-character tag digest")
assert_not_matches ("${actual}" "HD19D5EEB0FE0$" "case-sensitive tag distinction")

vsag_thirdparty_pinned_variable (
    CPUINFO ca678952a9a8eaa6de112d154e8e104b22f9ab3f actual)
assert_equal (
    "${actual}" "VSAG_THIRDPARTY_CPUINFO_COMMIT_CA678952A9A8" "commit pin")
vsag_thirdparty_pinned_variable (
    EXAMPLE 1111111111111aaaaaaaaaaaaaaaaaaaaaaaaaaa actual
    COLLISION_PINS 1111111111112bbbbbbbbbbbbbbbbbbbbbbbbbbb)
assert_equal (
    "${actual}" "VSAG_THIRDPARTY_EXAMPLE_COMMIT_1111111111111" "collision prefix extension")

set (secret "https://user:password@example.invalid/pinned.tar.gz")
run_fixture ("${secret}" "https://legacy.invalid/archive.tar.gz" output)
assert_matches ("${output}" "source=pinned" "pinned precedence source")
assert_matches ("${output}" "variable=VSAG_THIRDPARTY_FMT_10_2_1" "pinned variable")
assert_matches ("${output}" "legacy fallback VSAG_THIRDPARTY_FMT is ignored" "legacy ignored")
assert_matches ("${output}" "TEST_SELECTION=pinned" "pinned URL precedence")
assert_not_matches ("${output}" "password" "credential-safe diagnostics")

run_fixture ("" "legacy-archive" output)
assert_matches ("${output}" "source=legacy" "legacy fallback source")
assert_matches ("${output}" "use pinned variable" "deprecation guidance")
assert_matches ("${output}" "VSAG_THIRDPARTY_FMT_10_2_1" "expected pinned variable")
assert_matches ("${output}" "TEST_SELECTION=legacy" "legacy URL precedence")

set (ENV{VSAG_THIRDPARTY_FMT_9_0_0} "mismatched-archive")
run_fixture ("" "" output)
assert_matches ("${output}" "source=default" "default source")
assert_matches ("${output}" "TEST_SELECTION=default" "mismatched pin ignored")
assert_not_matches ("${output}" "mismatched-archive" "ignored variable value")

unset (ENV{VSAG_THIRDPARTY_OPENBLAS_0_3_24})
unset (ENV{VSAG_THIRDPARTY_OPENBLAS})
set (ENV{VSAG_THIRDPARTY_OPENBLAS_0_3_23} "obsolete-openblas-archive")
set (openblas_urls default-openblas)
vsag_resolve_thirdparty_override (OPENBLAS v0.3.24 openblas_urls)
list (GET openblas_urls 0 selected_openblas_url)
assert_equal ("${selected_openblas_url}" "default-openblas" "obsolete OpenBLAS pin ignored")

set (ENV{VSAG_THIRDPARTY_OPENBLAS} "legacy-openblas-archive")
set (openblas_urls default-openblas)
vsag_resolve_thirdparty_override (OPENBLAS v0.3.24 openblas_urls)
list (GET openblas_urls 0 selected_openblas_url)
assert_equal ("${selected_openblas_url}" "legacy-openblas-archive"
              "OpenBLAS legacy fallback")

set (ENV{VSAG_THIRDPARTY_OPENBLAS_0_3_24} "pinned-openblas-archive")
set (openblas_urls default-openblas)
vsag_resolve_thirdparty_override (OPENBLAS v0.3.24 openblas_urls)
list (GET openblas_urls 0 selected_openblas_url)
assert_equal ("${selected_openblas_url}" "pinned-openblas-archive"
              "OpenBLAS pinned precedence")
unset (ENV{VSAG_THIRDPARTY_OPENBLAS_0_3_24})
unset (ENV{VSAG_THIRDPARTY_OPENBLAS_0_3_23})
unset (ENV{VSAG_THIRDPARTY_OPENBLAS})

set (ENV{VSAG_THIRDPARTY_FMT} "https://user:legacy-secret@example.invalid/fmt.tar.gz")
execute_process (
    COMMAND ${CMAKE_COMMAND} -DVSAG_SOURCE_DIR=${VSAG_SOURCE_DIR}
            -P ${VSAG_SOURCE_DIR}/tests/cmake/thirdparty_override_fixture.cmake
    RESULT_VARIABLE result
    OUTPUT_VARIABLE stdout
    ERROR_VARIABLE stderr)
if (NOT result EQUAL 0)
    message (FATAL_ERROR "Credential-safe legacy fixture failed: ${result}")
endif ()
set (output "${stdout}\n${stderr}")
assert_not_matches ("${output}" "legacy-secret" "legacy credential-safe diagnostics")

execute_process (
    COMMAND ${CMAKE_COMMAND} -DVSAG_SOURCE_DIR=${VSAG_SOURCE_DIR} -DTEST_UNDEFINED_URLS=ON
            -P ${VSAG_SOURCE_DIR}/tests/cmake/thirdparty_override_fixture.cmake
    RESULT_VARIABLE result
    OUTPUT_VARIABLE stdout
    ERROR_VARIABLE stderr)
if (result EQUAL 0)
    message (FATAL_ERROR "An undefined third-party URL variable was accepted")
endif ()
set (output "${stdout}\n${stderr}")
assert_matches ("${output}" "URL variable 'missing_urls' is not defined"
                "undefined URL variable rejection")

set (hash_fixture "${CMAKE_CURRENT_BINARY_DIR}/thirdparty-override-hash-fixture")
set (download_fixture "${CMAKE_CURRENT_BINARY_DIR}/thirdparty-override-download")
file (WRITE "${hash_fixture}" "wrong archive content")
set (ENV{VSAG_THIRDPARTY_FMT_10_2_1} "file://${hash_fixture}")
execute_process (
    COMMAND ${CMAKE_COMMAND} -DVSAG_SOURCE_DIR=${VSAG_SOURCE_DIR}
            -DTEST_EXPECTED_HASH=MD5=00000000000000000000000000000000
            -DTEST_DOWNLOAD_PATH=${download_fixture}
            -P ${VSAG_SOURCE_DIR}/tests/cmake/thirdparty_override_fixture.cmake
    RESULT_VARIABLE result
    OUTPUT_VARIABLE stdout
    ERROR_VARIABLE stderr)
if (result EQUAL 0)
    message (FATAL_ERROR "A pinned variable with mismatched content passed hash verification")
endif ()
set (output "${stdout}\n${stderr}")
assert_matches ("${output}" "HASH mismatch" "mismatched archive rejection")
assert_not_matches ("${output}" "wrong archive content" "hash diagnostic content safety")
file (REMOVE "${hash_fixture}" "${download_fixture}")

message (STATUS "Third-party pinned-variable tests passed")
