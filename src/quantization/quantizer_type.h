
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

namespace vsag {
enum class QuantizerType {
    QUANTIZER_TYPE_SQ8 = 0,
    QUANTIZER_TYPE_SQ8_UNIFORM = 1,
    QUANTIZER_TYPE_SQ4 = 2,
    QUANTIZER_TYPE_SQ4_UNIFORM = 3,
    QUANTIZER_TYPE_FP32 = 4,
    QUANTIZER_TYPE_FP16 = 5,
    QUANTIZER_TYPE_BF16 = 6,
    QUANTIZER_TYPE_INT8 = 7,
    QUANTIZER_TYPE_PQ = 8,
    QUANTIZER_TYPE_PQFS = 9,
    QUANTIZER_TYPE_RABITQ = 10,
    QUANTIZER_TYPE_SPARSE = 11,
    QUANTIZER_TYPE_TQ = 12,
};
}  // namespace vsag
