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

namespace vsag {

ReasoningContext::ReasoningContext(Allocator* allocator)
    : allocator_(allocator),
      expected_inner_ids_(AllocatorWrapper<InnerIdType>(allocator)),
      expected_traces_(
          AllocatorWrapper<std::pair<const InnerIdType, ExpectedTargetTrace>>(allocator)),
      reorder_changes_(AllocatorWrapper<ReorderRecord>(allocator)),
      selected_buckets_(AllocatorWrapper<BucketIdType>(allocator)) {
}

ReasoningContext::~ReasoningContext() = default;

void
ReasoningContext::InitializeExpectedTargets(
    const Vector<int64_t>& labels, const UnorderedMap<int64_t, InnerIdType>& label_to_inner_id) {
    std::lock_guard<std::mutex> guard(mutex_);
    expected_inner_ids_.clear();
    expected_traces_.clear();

    for (const auto& label : labels) {
        auto it = label_to_inner_id.find(label);
        if (it != label_to_inner_id.end()) {
            InnerIdType inner_id = it->second;
            expected_inner_ids_.insert(inner_id);

            ExpectedTargetTrace trace;
            trace.label = label;
            trace.inner_id = inner_id;

            expected_traces_.insert(std::make_pair(inner_id, trace));
        }
    }
}

void
ReasoningContext::SetTrueDistance(InnerIdType id, float dist) {
    std::lock_guard<std::mutex> guard(mutex_);
    auto it = expected_traces_.find(id);
    if (it != expected_traces_.end()) {
        it.value().true_distance = dist;
    }
}

void
ReasoningContext::RecordVisit(InnerIdType id, float dist, uint32_t hop) {
    std::lock_guard<std::mutex> guard(mutex_);
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
    std::lock_guard<std::mutex> guard(mutex_);
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
    std::lock_guard<std::mutex> guard(mutex_);
    auto it = expected_traces_.find(id);
    if (it != expected_traces_.end()) {
        it.value().filter_rejected = true;
        it.value().was_visited = true;
    }
}

void
ReasoningContext::RecordReorder(InnerIdType id, float dist_before, float dist_after) {
    std::lock_guard<std::mutex> guard(mutex_);
    auto it = expected_traces_.find(id);
    if (it == expected_traces_.end()) {
        return;
    }

    it.value().quantized_distance = dist_before;
    it.value().true_distance = dist_after;

    ReorderRecord record;
    record.id = id;
    record.dist_before = dist_before;
    record.dist_after = dist_after;
    reorder_changes_.push_back(record);
}

void
ReasoningContext::RecordReorderEviction(InnerIdType id, uint32_t hop) {
    std::lock_guard<std::mutex> guard(mutex_);
    auto it = expected_traces_.find(id);
    if (it != expected_traces_.end()) {
        it.value().reorder_evicted = true;
        if (!it.value().was_visited) {
            it.value().was_visited = true;
            it.value().visited_at_hop = static_cast<int32_t>(hop);
        }
    }
}

void
ReasoningContext::SetTermination(const std::string& reason) {
    std::lock_guard<std::mutex> guard(mutex_);
    termination_reason_ = reason;
}

void
ReasoningContext::RecordBucketSelection(const Vector<BucketIdType>& buckets) {
    std::lock_guard<std::mutex> guard(mutex_);
    selected_buckets_ = buckets;
}

void
ReasoningContext::MarkResult(const Vector<InnerIdType>& result_ids) {
    std::lock_guard<std::mutex> guard(mutex_);
    for (const auto& id : result_ids) {
        auto it = expected_traces_.find(id);
        if (it != expected_traces_.end()) {
            it.value().was_in_result_set = true;
        }
    }
}

void
ReasoningContext::DiagnoseExpectedTargets() {
    std::lock_guard<std::mutex> guard(mutex_);
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
    std::lock_guard<std::mutex> guard(mutex_);
    JsonType report;
    JsonType missed_targets = JsonType::Parse("[]");

    int found_count = 0;
    int missed_count = 0;

    for (const auto& pair : expected_traces_) {
        const auto& trace = pair.second;
        if (trace.was_in_result_set) {
            found_count++;
        } else {
            missed_count++;

            JsonType detail;
            detail["label"].SetJson(JsonType::Parse(std::to_string(trace.label)));
            detail["inner_id"].SetJson(JsonType::Parse(std::to_string(trace.inner_id)));
            detail["diagnosis"].SetString(trace.diagnosis);
            detail["true_distance"].SetFloat(trace.true_distance);
            detail["quantized_distance"].SetFloat(trace.quantized_distance);
            detail["was_visited"].SetBool(trace.was_visited);
            detail["visited_at_hop"].SetJson(JsonType::Parse(std::to_string(trace.visited_at_hop)));
            detail["was_evicted"].SetBool(trace.was_evicted);
            detail["filter_rejected"].SetBool(trace.filter_rejected);
            detail["reorder_evicted"].SetBool(trace.reorder_evicted);
            missed_targets.AppendJson(detail);
        }
    }

    std::string summary = std::to_string(found_count) + "/" +
                          std::to_string(expected_traces_.size()) + " expected labels found, " +
                          std::to_string(missed_count) + " missed";

    report["expected_analysis"]["summary"].SetString(summary);
    report["expected_analysis"]["missed_targets"].SetJson(missed_targets);

    if (not selected_buckets_.empty()) {
        JsonType bucket_array = JsonType::Parse("[]");
        for (const auto bucket_id : selected_buckets_) {
            JsonType bucket_json;
            bucket_json.SetInt(static_cast<int64_t>(bucket_id));
            bucket_array.AppendJson(bucket_json);
        }
        report["bucket_selection"]["selected_bucket_count"].SetInt(
            static_cast<int64_t>(selected_buckets_.size()));
        report["bucket_selection"]["selected_buckets"].SetJson(bucket_array);
    }

    return report.Dump();
}

void
ReasoningContext::SetSearchParams(int64_t topk,
                                  const std::string& index_type,
                                  bool use_reorder,
                                  bool filter_active) {
    std::lock_guard<std::mutex> guard(mutex_);
    topk_ = topk;
    index_type_ = index_type;
    use_reorder_ = use_reorder;
    filter_active_ = filter_active;
}

void
ReasoningContext::AddSearchHop() {
    std::lock_guard<std::mutex> guard(mutex_);
    total_hops_++;
}

void
ReasoningContext::AddDistanceComputation(uint32_t count) {
    std::lock_guard<std::mutex> guard(mutex_);
    total_dist_computations_ += count;
}

}  // namespace vsag
