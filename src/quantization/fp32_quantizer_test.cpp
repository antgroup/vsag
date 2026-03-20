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

#include "fp32_quantizer.h"

#include "quantization/quantizer_test.h"

using namespace vsag;

namespace {
const auto dims = std::vector<int>{64, 128};
const auto counts = std::vector<int>{10, 101};
constexpr float error = 2e-5f;
}  // namespace

DEFINE_QUANTIZER_ENCODE_DECODE_TESTS_SIMPLE("FP32Quantizer", FP32Quantizer, dims, counts, error)
DEFINE_QUANTIZER_COMPUTE_TESTS_WITH_SAME("FP32Quantizer", FP32Quantizer, dims, counts, error, 65536)
DEFINE_QUANTIZER_SERIALIZE_TESTS("FP32Quantizer", FP32Quantizer, dims, counts, error)