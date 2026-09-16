if (NOT DEFINED VSAG_SOURCE_DIR)
    message (FATAL_ERROR "VSAG_SOURCE_DIR is required")
endif ()
if (NOT DEFINED VSAG_BUILD_DIR)
    message (FATAL_ERROR "VSAG_BUILD_DIR is required")
endif ()

set (ninja_file "${VSAG_BUILD_DIR}/build.ninja")
set (rules_file "${VSAG_BUILD_DIR}/CMakeFiles/rules.ninja")
if (NOT EXISTS "${ninja_file}" OR NOT EXISTS "${rules_file}")
    message (FATAL_ERROR "VSAG_BUILD_DIR must contain a Ninja build")
endif ()

file (READ "${ninja_file}" ninja_content)
file (READ "${rules_file}" rules_content)

function (assert_object_pool source target expect_pool)
    set (object_path "src/datacell/CMakeFiles/${target}.dir/${source}.o")
    string (FIND "${ninja_content}" "build ${object_path}:" block_start)
    if (block_start EQUAL -1)
        message (FATAL_ERROR "Ninja object rule not found: ${object_path}")
    endif ()

    string (SUBSTRING "${ninja_content}" ${block_start} -1 remaining_content)
    string (FIND "${remaining_content}" "\n\n" block_length)
    if (block_length EQUAL -1)
        message (FATAL_ERROR "Ninja object rule is not terminated: ${object_path}")
    endif ()
    string (SUBSTRING "${remaining_content}" 0 ${block_length} object_rule)
    string (FIND "${object_rule}" "pool = rabitq_split_factory_pool" pool_position)

    if (expect_pool AND pool_position EQUAL -1)
        message (FATAL_ERROR "Serial pool is missing from ${object_path}")
    elseif (NOT expect_pool AND NOT pool_position EQUAL -1)
        message (FATAL_ERROR "Ordinary datacell object uses the serial pool: ${object_path}")
    endif ()
endfunction ()

set (rabitq_split_factory_sources
     rabitq_split_datacell_factory_l2.cpp
     rabitq_split_datacell_factory_ip.cpp
     rabitq_split_datacell_factory_cosine.cpp)
foreach (source IN LISTS rabitq_split_factory_sources)
    assert_object_pool ("${source}" rabitq_split_datacell_factory TRUE)
endforeach ()

file (GLOB datacell_sources RELATIVE "${VSAG_SOURCE_DIR}/src/datacell"
      "${VSAG_SOURCE_DIR}/src/datacell/*.cpp")
list (FILTER datacell_sources EXCLUDE REGEX "_test\\.cpp$")
list (REMOVE_ITEM datacell_sources ${rabitq_split_factory_sources})
foreach (source IN LISTS datacell_sources)
    assert_object_pool ("${source}" datacell FALSE)
endforeach ()

if (NOT rules_content MATCHES "pool rabitq_split_factory_pool\n  depth = 1")
    message (FATAL_ERROR "The RaBitQ split factory pool must have capacity one")
endif ()

message (STATUS "Datacell Ninja job-pool assignments passed")
