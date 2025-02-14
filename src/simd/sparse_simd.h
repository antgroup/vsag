
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
#include <string>

namespace vsag {

namespace generic {
float
SparseComputeIP(int32_t nnz1, const uint32_t* ids1, const float* vals1,
                int32_t nnz2, const uint32_t* ids2, const float* vals2);
}  // namespace generic

using SparseComputeType = float (*)(int32_t nnz1, const uint32_t* ids1, const float* vals1,
                      int32_t nnz2, const uint32_t* ids2, const float* vals2);
extern SparseComputeType SparseComputeIP;
}  // namespace vsag
