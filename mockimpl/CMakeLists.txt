
set (MOCK_SRCS
  vsag/simpleflat.cpp
  vsag/factory.cpp
  ../src/impl/bitset/bitset.cpp
  ../src/impl/bitset/computable_bitset.cpp
  ../src/impl/bitset/sparse_bitset.cpp
  ../src/impl/bitset/fast_bitset.cpp
  ../src/constants.cpp
  ../src/dataset_impl.cpp
)

add_library (vsag_mockimpl SHARED ${MOCK_SRCS})
add_library (vsag_mockimpl_static STATIC ${MOCK_SRCS})

target_link_libraries (vsag_mockimpl roaring fmt::fmt-header-only simd)
target_link_libraries (vsag_mockimpl_static roaring fmt::fmt-header-only simd)
add_dependencies (vsag_mockimpl version_mockimpl roaring)
add_dependencies (vsag_mockimpl_static version_mockimpl roaring)

set_target_properties(vsag_mockimpl_static PROPERTIES OUTPUT_NAME "vsag_mockimpl")
install(TARGETS vsag_mockimpl vsag_mockimpl_static
	LIBRARY DESTINATION "${CMAKE_INSTALL_LIBDIR}")

find_package (Git)
add_custom_target (version_mockimpl
  ${CMAKE_COMMAND} -D SRC=${CMAKE_CURRENT_SOURCE_DIR}/vsag/version.h.in
                   -D DST=${CMAKE_CURRENT_SOURCE_DIR}/vsag/version.h
                   -D GIT_EXECUTABLE=${GIT_EXECUTABLE}
                   -P ${CMAKE_CURRENT_SOURCE_DIR}/../cmake/GenerateVersionHeader.cmake
  )
add_dependencies (vsag_mockimpl version_mockimpl)
add_dependencies (vsag_mockimpl_static version_mockimpl)

if (ENABLE_TESTS)
    add_executable (tests_mockimpl
            ../tests/test_factory.cpp
    )
    target_include_directories(tests_mockimpl PRIVATE ${CMAKE_CURRENT_BINARY_DIR}/spdlog/install/include)
    target_link_libraries (tests_mockimpl PRIVATE Catch2::Catch2WithMain vsag_mockimpl)
    add_dependencies(tests_mockimpl spdlog Catch2)
endif ()
