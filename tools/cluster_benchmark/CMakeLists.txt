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

add_executable (cluster_benchmark cluster_benchmark.cpp)
target_link_libraries (cluster_benchmark PRIVATE
    argparse::argparse
    fmt::fmt
    vsag
    simd
    vsag_hdf5_headers
    ${HDF5_CPP_STATIC_LIBRARY}
    ${HDF5_C_STATIC_LIBRARY}
    z dl)
add_dependencies (cluster_benchmark hdf5)
