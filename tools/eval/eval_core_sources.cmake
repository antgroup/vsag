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

# Sources of the dependency-free evaluation core: building/searching an in-memory index and
# measuring it. They must not depend on HDF5, yaml-cpp, argparse, httplib or tabulate so the same
# list can be compiled into both the eval tools and libvsag_c (used by pyvsag.eval).
set (VSAG_EVAL_CORE_DIR ${CMAKE_CURRENT_LIST_DIR})
set (VSAG_EVAL_CORE_SRCS
    ${VSAG_EVAL_CORE_DIR}/case/eval_case.cpp
    ${VSAG_EVAL_CORE_DIR}/case/search_eval_case.cpp
    ${VSAG_EVAL_CORE_DIR}/case/build_eval_case.cpp
    ${VSAG_EVAL_CORE_DIR}/monitor/monitor.cpp
    ${VSAG_EVAL_CORE_DIR}/monitor/latency_monitor.cpp
    ${VSAG_EVAL_CORE_DIR}/monitor/recall_monitor.cpp
    ${VSAG_EVAL_CORE_DIR}/monitor/memory_peak_monitor.cpp
    ${VSAG_EVAL_CORE_DIR}/monitor/duration_monitor.cpp
    ${VSAG_EVAL_CORE_DIR}/eval_config.cpp
    ${VSAG_EVAL_CORE_DIR}/eval_dataset.cpp
    ${VSAG_EVAL_CORE_DIR}/evaluator.cpp)
