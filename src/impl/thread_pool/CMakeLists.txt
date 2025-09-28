
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


set (THREAD_POOL_SRC
    default_thread_pool.cpp
    default_thread_pool.h
    safe_thread_pool.h
)

add_library (thread_pool OBJECT ${THREAD_POOL_SRC})
target_link_libraries (thread_pool PRIVATE fmt::fmt coverage_config)
maybe_add_dependencies (thread_pool spdlog)
