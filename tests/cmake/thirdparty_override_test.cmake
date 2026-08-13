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
        math (EXPR override_inventory_count "${override_inventory_count} + 1")
        if (NOT thirdparty_cmake_content MATCHES "https://github\\.com/")
            message (FATAL_ERROR
                     "Third-party override has no authoritative upstream URL: "
                     "${thirdparty_cmake_file}")
        endif ()
    endif ()
endforeach ()
if (override_inventory_count EQUAL 0)
    message (FATAL_ERROR "No third-party override inventories were checked")
endif ()

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
