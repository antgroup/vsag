// Copyright 2024-present the vsag project
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#pragma once

#include <pybind11/numpy.h>

#include "vsag/index.h"

namespace vsag::python {

pybind11::dict
autotune_search(const IndexPtr& index,
                const pybind11::array& queries,
                const pybind11::array& ground_truth,
                uint64_t top_k,
                const pybind11::dict& parameter_space,
                const pybind11::dict& constraints,
                const std::string& objective,
                uint64_t concurrency,
                uint64_t max_trials,
                bool include_raw_evaluation);

}  // namespace vsag::python
