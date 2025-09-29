
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


set (UTILS_SRC
        util_functions.cpp
        linear_congruential_generator.cpp
        visited_list.cpp
        slow_task_timer.cpp
        timer.cpp
        window_result_queue.cpp
        sparse_vector_transform.cpp
        prefetch.cpp
        lock_strategy.cpp
)

add_library (utils OBJECT ${UTILS_SRC})
target_link_libraries (utils PUBLIC coverage_config)
add_dependencies (utils spdlog fmt::fmt)

if (ENABLE_TESTS)
    file (GLOB_RECURSE UTILS_TESTS "*_test.cpp")
    add_library (utils_test STATIC ${UTILS_TESTS})
    target_link_libraries (utils_test PRIVATE Catch2::Catch2 vsag)
    add_dependencies (utils_test spdlog Catch2)
endif ()
