
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
        mmap_io_parameter.cpp
        mmap_io.cpp
        memory_block_io.cpp
        reader_io.cpp
        reader_io_parameter.cpp
)

add_library (io STATIC ${IO_SRC})
target_link_libraries (io PUBLIC fmt::fmt aio coverage_config)
maybe_add_dependencies (io spdlog)

if (ENABLE_TESTS)
    file (GLOB_RECURSE IO_TESTS "*_test.cpp")
    add_library (io_test STATIC ${IO_TESTS})
    target_link_libraries (io_test PRIVATE Catch2::Catch2 vsag)
    add_dependencies (io_test spdlog Catch2)
endif ()
