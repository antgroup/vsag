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

#include "search_reasoning.h"

#include <cmath>

#include "simd/bf16_simd.h"
#include "simd/fp16_simd.h"

namespace vsag {

ReasoningContext::ReasoningContext(Allocator* allocator)
    : allocator_(allocator),
      expected_inner_ids_(AllocatorWrapper<InnerIdType>(allocator)),
      expected_traces_(
          AllocatorWrapper<std::pair<const InnerIdType, ExpectedTargetTrace>>(allocator)),
      reorder_changes_(AllocatorWrapper<ReorderRecord>(allocator)) {
}

ReasoningContext::~ReasoningContext() = default;

void
ReasoningContext::InitializeExpectedTargets(
    const Vector<int64_t>& labels,
    const UnorderedMap<int64_t, InnerIdType>& label_to_inner_id,
    const float* query,
    const void* precise_vectors,
    DataTypes data_type,
    uint64_t dim) {
    expected_inner_ids_.clear();
    expected_traces_.clear();

    float* casted_vec = nullptr;
    if (data_type != DataTypes::DATA_TYPE_FLOAT) {
        casted_vec = new float[dim];
    }

    for (const auto& label : labels) {
        auto it = label_to_inner_id.find(label);
        if (it != label_to_inner_id.end()) {
            InnerIdType inner_id = it->second;
            expected_inner_ids_.insert(inner_id);

            ExpectedTargetTrace trace;
            trace.label = label;
            trace.inner_id = inner_id;

            const float* vec = nullptr;

            if (data_type == DataTypes::DATA_TYPE_FLOAT) {
                vec = static_cast<const float*>(precise_vectors) +
                      static_cast<uint64_t>(inner_id) * dim;
            } else if (data_type == DataTypes::DATA_TYPE_INT8) {
                const int8_t* int8_vec = static_cast<const int8_t*>(precise_vectors) +
                                         static_cast<uint64_t>(inner_id) * dim;
                for (uint64_t i = 0; i < dim; ++i) {
                    casted_vec[i] = static_cast<float>(int8_vec[i]);
                }
                vec = casted_vec;
            } else if (data_type == DataTypes::DATA_TYPE_FP16) {
                const uint16_t* fp16_vec = static_cast<const uint16_t*>(precise_vectors) +
                                           static_cast<uint64_t>(inner_id) * dim;
                for (uint64_t i = 0; i < dim; ++i) {
                    casted_vec[i] = generic::FP16ToFloat(fp16_vec[i]);
                }
                vec = casted_vec;
            } else if (data_type == DataTypes::DATA_TYPE_BF16) {
                const uint16_t* bf16_vec = static_cast<const uint16_t*>(precise_vectors) +
                                           static_cast<uint64_t>(inner_id) * dim;
                for (uint64_t i = 0; i < dim; ++i) {
                    casted_vec[i] = generic::BF16ToFloat(bf16_vec[i]);
                }
                vec = casted_vec;
            }

            if (vec != nullptr) {
                float true_dist = 0.0F;
                for (uint64_t i = 0; i < dim; ++i) {
                    float diff = query[i] - vec[i];
                    true_dist += diff * diff;
                }
                trace.true_distance = std::sqrt(true_dist);
            }

            expected_traces_.insert(std::make_pair(inner_id, trace));
        }
    }

    delete[] casted_vec;
}

void
ReasoningContext::RecordVisit(InnerIdType id, float dist, uint32_t hop) {
    auto it = expected_traces_.find(id);
    if (it != expected_traces_.end()) {
        it.value().was_visited = true;
        it.value().visited_at_hop = static_cast<int32_t>(hop);
        if (it.value().quantized_distance == 0.0F) {
            it.value().quantized_distance = dist;
        }
    }
}

void
ReasoningContext::RecordEviction(InnerIdType id, uint32_t hop) {
    auto it = expected_traces_.find(id);
    if (it != expected_traces_.end()) {
        it.value().was_evicted = true;
        if (!it.value().was_visited) {
            it.value().was_visited = true;
            it.value().visited_at_hop = static_cast<int32_t>(hop);
        }
    }
}

void
ReasoningContext::RecordFilterReject(InnerIdType id) {
    auto it = expected_traces_.find(id);
    if (it != expected_traces_.end()) {
        it.value().filter_rejected = true;
        it.value().was_visited = true;
    }
}

void
ReasoningContext::RecordReorder(InnerIdType id, float dist_before, float dist_after) {
    auto it = expected_traces_.find(id);
    if (it != expected_traces_.end()) {
        it.value().reorder_evicted = true;
        it.value().quantized_distance = dist_before;
        it.value().true_distance = dist_after;
    }

    ReorderRecord record;
    record.id = id;
    record.dist_before = dist_before;
    record.dist_after = dist_after;
    reorder_changes_.push_back(record);
}

void
ReasoningContext::SetTermination(const std::string& reason) {
    termination_reason_ = reason;
}

void
ReasoningContext::MarkResult(const Vector<InnerIdType>& result_ids) {
    for (const auto& id : result_ids) {
        auto it = expected_traces_.find(id);
        if (it != expected_traces_.end()) {
            it.value().was_in_result_set = true;
        }
    }
}

void
ReasoningContext::DiagnoseExpectedTargets() {
    for (auto it = expected_traces_.begin(); it != expected_traces_.end(); ++it) {
        it.value().diagnosis = DiagnoseTarget(it.value());
    }
}

std::string
ReasoningContext::DiagnoseTarget(const ExpectedTargetTrace& trace) {
    if (!trace.was_visited) {
        return "not_reachable";
    }

    if (trace.filter_rejected) {
        return "filter_rejected";
    }

    if (trace.quantized_distance > trace.true_distance * 1.5F && trace.true_distance > 0.0F) {
        return "quantization_error";
    }

    if (trace.was_evicted && !trace.was_in_result_set) {
        return "ef_too_small";
    }

    if (trace.reorder_evicted && !trace.was_in_result_set) {
        return "reorder_evicted";
    }

    if (!trace.was_in_result_set) {
        return "unknown";
    }

    return "success";
}

std::string
ReasoningContext::GenerateReport() const {
    JsonType report;

    int found_count = 0;
    int missed_count = 0;

    for (const auto& pair : expected_traces_) {
        const auto& trace = pair.second;
        if (trace.was_in_result_set) {
            found_count++;
        } else {
            missed_count++;
        }
    }

    std::string summary = std::to_string(found_count) + "/" +
                          std::to_string(expected_traces_.size()) + " expected labels found, " +
                          std::to_string(missed_count) + " missed";

    report["expected_analysis"]["summary"].SetString(summary);

    return report.Dump();
}

void
ReasoningContext::SetSearchParams(int64_t topk,
                                  const std::string& index_type,
                                  bool use_reorder,
                                  bool filter_active) {
    topk_ = topk;
    index_type_ = index_type;
    use_reorder_ = use_reorder;
    filter_active_ = filter_active;
}

void
ReasoningContext::AddSearchHop() {
    total_hops_++;
}

void
ReasoningContext::AddDistanceComputation(uint32_t count) {
    total_dist_computations_ += count;
}

}  // namespace vsag
