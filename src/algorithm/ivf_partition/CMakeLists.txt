
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


set (IVF_PARTITION_SRCS
        gno_imi_parameter.cpp
        ivf_nearest_partition.cpp
        gno_imi_partition.cpp
        ivf_partition_strategy_parameter.cpp
        ivf_partition_strategy.cpp
)

add_library (ivf_partition OBJECT ${IVF_PARTITION_SRCS})
target_link_libraries (ivf_partition PUBLIC coverage_config)
maybe_add_dependencies (ivf_partition spdlog fmt::fmt mkl openblas)
