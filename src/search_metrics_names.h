// Copyright 2024-present the vsag project
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
// http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#pragma once

#include "vsag/search_metrics.h"

namespace vsag {

inline const char*
DistanceEvaluationPhaseName(DistanceEvaluationPhase phase) {
    switch (phase) {
        case DistanceEvaluationPhase::ROUTING:
            return "routing";
        case DistanceEvaluationPhase::APPROXIMATE:
            return "approximate";
        case DistanceEvaluationPhase::RERANK:
            return "rerank";
        default:
            return "unknown";
    }
}

inline const char*
DistanceEvaluationBackendName(DistanceEvaluationBackend backend) {
    static constexpr const char* names[] = {"fp32",
                                            "fp16",
                                            "bf16",
                                            "int8",
                                            "sq8",
                                            "sq4",
                                            "sq8_uniform",
                                            "sq4_uniform",
                                            "pq",
                                            "pq_fastscan",
                                            "rabitq",
                                            "binary",
                                            "sparse_fp32",
                                            "sparse_fp16",
                                            "sparse_sq8",
                                            "unknown"};
    // The name table mirrors the contiguous DistanceEvaluationBackend enumeration exactly;
    // a compile-time audit ensures additions or reordering cannot silently mislabel backends.
    static_assert(sizeof(names) / sizeof(names[0]) ==
                      static_cast<uint8_t>(DistanceEvaluationBackend::UNKNOWN) + 1,
                  "DistanceEvaluationBackend name table must cover every backend up to UNKNOWN");
    const auto index = static_cast<uint8_t>(backend);
    return index < sizeof(names) / sizeof(names[0]) ? names[index] : "unknown";
}

}  // namespace vsag
