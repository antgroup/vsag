
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


set (eval_srcs
        case/eval_case.cpp
        case/search_eval_case.cpp
        case/build_eval_case.cpp

        exporter/exporter.cpp
        exporter/formatter.cpp

        monitor/monitor.cpp
        monitor/latency_monitor.cpp
        monitor/recall_monitor.cpp
        monitor/memory_peak_monitor.cpp
        monitor/duration_monitor.cpp

        eval_config.cpp
        eval_dataset.cpp
        eval_job.cpp
        )
add_library (eval_obj OBJECT ${eval_srcs})
target_compile_options (eval_obj PRIVATE -fopenmp)
target_link_libraries (eval_obj PRIVATE cpr)
add_dependencies (eval_obj hdf5 spdlog yaml-cpp tabulate cpr)

add_executable (eval_performance main.cpp)
target_compile_options (eval_performance PRIVATE -fopenmp)
target_link_libraries (eval_performance
  PRIVATE
  vsag
  eval_obj
  yaml-cpp
  argparse::argparse
  tabulate
  simd
  libhdf5_cpp.a
  libhdf5.a
  z
)
