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

#include <string>
#include <vector>

#include "json_types.h"
#include "metric_type.h"
#include "vsag/index.h"

namespace vsag {

// Trusted C++-only entry point for IndexAnalyzer. Serialized internal parameters must not be
// accepted through Factory or the public bindings.
#if defined(__GNUC__) || defined(__clang__)
__attribute__((visibility("hidden")))
#endif
IndexPtr
CreateBruteForceIndexForAnalyzer(JsonType build_param,
                                 std::vector<std::string>* ignored_param_keys = nullptr);

// Read the quantizer metric after the trusted analyzer index has been deserialized.
#if defined(__GNUC__) || defined(__clang__)
__attribute__((visibility("hidden")))
#endif
MetricType
GetBruteForceMetricForAnalyzer(const IndexPtr& index);

}  // namespace vsag
