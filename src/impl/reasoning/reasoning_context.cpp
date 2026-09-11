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

#include "reasoning_context.h"

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
ReasoningContext::RecordBucketSelection(const Vector<BucketIdType>& buckets) {
    std::lock_guard<std::mutex> guard(mutex_);
    selected_buckets_ = buckets;
}

void
ReasoningContext::SetTermination(ReasoningTermination termination) {
    std::lock_guard<std::mutex> guard(mutex_);
    termination_ = termination;
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

ReasoningDiagnosis
ReasoningContext::DiagnoseTarget(const ExpectedTargetTrace& trace) {
    if (!trace.was_visited) {
        return ReasoningDiagnosis::kNotReachable;
    }

    if (trace.filter_rejected) {
        return ReasoningDiagnosis::kFilterRejected;
    }

    // Legacy diagnostic heuristic, not a metric-independent quantization bound.
    constexpr float quantization_error_ratio = 1.5F;
    if (trace.quantized_distance > trace.true_distance * quantization_error_ratio &&
        trace.true_distance > 0.0F) {
        return ReasoningDiagnosis::kQuantizationError;
    }

    if (trace.was_evicted && !trace.was_in_result_set) {
        return ReasoningDiagnosis::kEfTooSmall;
    }

    if (trace.reorder_evicted && !trace.was_in_result_set) {
        return ReasoningDiagnosis::kReorderEvicted;
    }

    if (!trace.was_in_result_set) {
        return ReasoningDiagnosis::kUnknown;
    }

    return ReasoningDiagnosis::kSuccess;
}

std::string
ReasoningContext::GenerateReport() const {
    std::lock_guard<std::mutex> guard(mutex_);
    JsonType report;

    // --- expected analysis --------------------------------------------------
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
            detail["label"].SetInt64(trace.label);
            detail["inner_id"].SetInt64(static_cast<int64_t>(trace.inner_id));
            detail["diagnosis"].SetString(ToString(trace.diagnosis));
            detail["true_distance"].SetFloat(trace.true_distance);
            detail["quantized_distance"].SetFloat(trace.quantized_distance);
            detail["was_visited"].SetBool(trace.was_visited);
            detail["visited_at_hop"].SetInt(static_cast<int64_t>(trace.visited_at_hop));
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

    // --- meta section -------------------------------------------------------
    report["meta"]["schema_version"].SetInt(1);
    report["meta"]["status"].SetString(ToString(ReasoningReportStatus::kOk));
    report["meta"]["index_type"].SetString(index_type_);
    report["meta"]["search_mode"].SetString(is_range_ ? "range" : "knn");
    report["meta"]["topk"].SetInt(topk_);
    report["meta"]["use_reorder"].SetBool(use_reorder_);
    report["meta"]["filter_active"].SetBool(filter_active_);
    report["meta"]["termination_reason"].SetString(ToString(termination_));
    report["meta"]["total_hops"].SetInt(static_cast<int64_t>(total_hops_));
    report["meta"]["total_distance_computations"].SetInt(
        static_cast<int64_t>(total_dist_computations_));

    const auto* capability = GetReasoningCapability(index_type_);

    JsonType diagnoses_json = JsonType::Parse("[]");
    constexpr ReasoningDiagnosis all_diagnoses[] = {ReasoningDiagnosis::kSuccess,
                                                    ReasoningDiagnosis::kNotReachable,
                                                    ReasoningDiagnosis::kFilterRejected,
                                                    ReasoningDiagnosis::kQuantizationError,
                                                    ReasoningDiagnosis::kEfTooSmall,
                                                    ReasoningDiagnosis::kReorderEvicted,
                                                    ReasoningDiagnosis::kUnknown};
    for (auto d : all_diagnoses) {
        {
            JsonType item;
            item.SetString(ToString(d));
            diagnoses_json.AppendJson(item);
        }
    }
    report["meta"]["available_diagnoses"].SetJson(diagnoses_json);

    JsonType events_json = JsonType::Parse("[]");
    for (uint8_t e = 0; e <= static_cast<uint8_t>(ReasoningEvent::kBucketSelection); ++e) {
        auto event = static_cast<ReasoningEvent>(e);
        if (capability != nullptr && HasReasoningEvent(capability->event_mask, event)) {
            {
                JsonType item;
                item.SetString(ToString(event));
                events_json.AppendJson(item);
            }
        }
    }
    report["meta"]["available_events"].SetJson(events_json);
    report["meta"]["supports_range"].SetBool(capability != nullptr && capability->supports_range);

    // --- bucket selection (optional) ----------------------------------------
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

std::string
ReasoningContext::MakeStatusReport(ReasoningReportStatus status, std::string_view index_type) {
    JsonType report;
    report["meta"]["schema_version"].SetInt(1);
    report["meta"]["status"].SetString(ToString(status));
    report["meta"]["index_type"].SetString(std::string(index_type));
    return report.Dump();
}

void
ReasoningContext::SetSearchParams(int64_t topk,
                                  const std::string& index_type,
                                  bool use_reorder,
                                  bool filter_active,
                                  bool is_range) {
    std::lock_guard<std::mutex> guard(mutex_);
    topk_ = topk;
    index_type_ = index_type;
    use_reorder_ = use_reorder;
    filter_active_ = filter_active;
    is_range_ = is_range;
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
