
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


set (IO_SRC
        io_parameter.cpp
        memory_io.cpp
        memory_io_parameter.cpp
        memory_block_io_parameter.cpp
        buffer_io_parameter.cpp
        buffer_io.cpp
        async_io_parameter.cpp
        async_io.cpp
        uring_io_parameter.cpp
        uring_io.cpp
        mmap_io_parameter.cpp
        mmap_io.cpp
        memory_block_io.cpp
        reader_io.cpp
        reader_io_parameter.cpp
)

add_library (io OBJECT ${IO_SRC})
set (IO_LINK_LIBRARIES coverage_config vsag_src_common)
if (HAVE_LIBAIO)
    list (APPEND IO_LINK_LIBRARIES aio)
    message (STATUS "io module: linking with libaio")
else ()
    message (STATUS "io module: libaio not available, async I/O disabled")
endif ()
if (HAVE_LIBURING)
    list (APPEND IO_LINK_LIBRARIES ${URING_LIBRARY})
    message (STATUS "io module: linking with liburing")
else ()
    message (STATUS "io module: liburing not available, io_uring disabled")
endif ()
target_link_libraries (io PRIVATE ${IO_LINK_LIBRARIES})

if (ENABLE_TESTS)
    file (GLOB_RECURSE IO_TESTS "*_test.cpp")
    add_library (io_test STATIC ${IO_TESTS})
    target_link_libraries (io_test PRIVATE Catch2::Catch2 vsag_static unittest_deps)
    add_dependencies (io_test Catch2)
endif ()
