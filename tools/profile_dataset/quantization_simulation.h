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

#include <cstdint>
#include <vector>

#include "profile_types.h"

namespace vsag::dataset_profile {

/// Per-dimension range used by VSAG's ScalarQuantizationTrainer::trunc_bound_train: the
/// ignore_count-th smallest / largest value with ignore_count = max(1, floor((n - 1) * rate)).
void
truncated_bounds(const std::vector<float>& data,
                 uint64_t rows,
                 uint64_t dim,
                 double rate,
                 std::vector<float>& lower,
                 std::vector<float>& upper);

/// Simulates VSAG's fp16, bf16, sq8, sq4, sq8_uniform and sq4_uniform codes on a sampled base
/// and measures recall@k of the decoded distances against exact distances, plus the mean
/// relative reconstruction error. Ranges follow ScalarQuantizationTrainer (per-dimension
/// truncated bounds for sq8/sq4, global min/max for sq8_uniform, the narrowest truncated
/// bounds for sq4_uniform) and codes use the same floor-based encoding. Queries stay fp32
/// for fp16/bf16/sq8/sq4 and are encoded for the uniform quantizers, as in VSAG.
/// This is a model of the quantizers, not a call into them; validate with eval_performance.
JsonType
simulate_quantization(const std::vector<float>& base,
                      uint64_t base_rows,
                      const std::vector<float>& queries,
                      uint64_t query_rows,
                      uint64_t dim,
                      Metric metric,
                      uint64_t k);

}  // namespace vsag::dataset_profile
