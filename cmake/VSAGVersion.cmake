include_guard (GLOBAL)

find_package (Git)

add_custom_target (version
    ${CMAKE_COMMAND}
    -D SRC=${CMAKE_CURRENT_SOURCE_DIR}/src/version.h.in
    -D DST=${CMAKE_CURRENT_SOURCE_DIR}/src/version.h
    -D GIT_EXECUTABLE=${GIT_EXECUTABLE}
    -P ${CMAKE_CURRENT_SOURCE_DIR}/cmake/GenerateVersionHeader.cmake)

if (TARGET vsag)
    add_dependencies (vsag version)
endif ()
if (TARGET vsag_static)
    add_dependencies (vsag_static version)
endif ()
