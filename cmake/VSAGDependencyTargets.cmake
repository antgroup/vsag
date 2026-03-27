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

include_guard (GLOBAL)

add_library (vsag_src_common INTERFACE)
target_link_libraries (vsag_src_common INTERFACE
    fmt::fmt
    nlohmann_json::nlohmann_json
    tsl::robin_map)

if (TARGET vsag_cpuinfo_headers)
    target_link_libraries (vsag_src_common INTERFACE vsag_cpuinfo_headers)
endif ()
if (TARGET vsag_diskann_headers)
    target_link_libraries (vsag_src_common INTERFACE vsag_diskann_headers)
endif ()
if (TARGET vsag_thread_pool_headers)
    target_link_libraries (vsag_src_common INTERFACE vsag_thread_pool_headers)
endif ()
if (TARGET vsag_antlr4_runtime_headers)
    target_link_libraries (vsag_src_common INTERFACE vsag_antlr4_runtime_headers)
endif ()
if (TARGET vsag_antlr4_autogen_headers)
    target_link_libraries (vsag_src_common INTERFACE vsag_antlr4_autogen_headers)
endif ()
if (TARGET vsag_roaring_headers)
    target_link_libraries (vsag_src_common INTERFACE vsag_roaring_headers)
endif ()
