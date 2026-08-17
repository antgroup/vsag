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

#include "sindi_host_filter.h"

#include <fmt/format.h>

#include <algorithm>
#include <cmath>
#include <limits>
#include <numeric>

#include "impl/heap/standard_heap.h"
#include "utils/util_functions.h"
#include "vsag_exception.h"

namespace vsag {
namespace {

constexpr uint32_t SINDI_MAX_HOST_COUNT = 50'000'000;

class InnerIdHostFilter : public Filter {
public:
    InnerIdHostFilter(const Vector<SindiHostRange>* ranges,
                      uint32_t range_begin,
                      uint32_t range_end,
                      FilterPtr filter)
        : ranges_(ranges),
          range_begin_(range_begin),
          range_end_(range_end),
          filter_(std::move(filter)) {
    }

    [[nodiscard]] bool
    CheckValid(int64_t id) const override {
        if (id < 0) {
            return false;
        }
        const auto inner_id = static_cast<uint64_t>(id);
        const auto range_begin = ranges_->begin() + range_begin_;
        const auto range_end = ranges_->begin() + range_end_;
        const auto range = std::upper_bound(
            range_begin, range_end, inner_id, [](uint64_t value, const SindiHostRange& candidate) {
                return value < candidate.begin;
            });
        if (range == range_begin or inner_id >= std::prev(range)->end) {
            return false;
        }
        return filter_ == nullptr or filter_->CheckValid(id);
    }

private:
    const Vector<SindiHostRange>* ranges_;
    uint32_t range_begin_;
    uint32_t range_end_;
    FilterPtr filter_;
};

DatasetPtr
collect_results(const DistHeapPtr& results, Allocator* allocator) {
    auto [result, distances, ids] =
        create_fast_dataset(static_cast<int64_t>(results->Size()), allocator);
    if (results->Empty()) {
        result->Dim(0)->NumElements(1);
        return result;
    }
    for (auto i = static_cast<int64_t>(results->Size()) - 1; i >= 0; --i) {
        distances[i] = results->Top().first;
        ids[i] = results->Top().second;
        results->Pop();
    }
    return result;
}

}  // namespace

uint32_t
ParseSindiHostFilterThreshold(const JsonType& json) {
    if (not json.Contains(SPARSE_HOST_FILTER_THRESHOLD)) {
        return DEFAULT_SPARSE_HOST_FILTER_THRESHOLD;
    }
    const auto value = json[SPARSE_HOST_FILTER_THRESHOLD];
    CHECK_ARGUMENT(value.IsNumberInteger(), "host_filter_threshold must be a non-negative integer");
    uint64_t threshold = 0;
    if (value.IsNumberUnsigned()) {
        threshold = value.GetUint64();
    } else {
        const auto signed_threshold = value.GetInt();
        CHECK_ARGUMENT(
            signed_threshold >= 0,
            fmt::format("host_filter_threshold must be non-negative, got {}", signed_threshold));
        threshold = static_cast<uint64_t>(signed_threshold);
    }
    CHECK_ARGUMENT(threshold <= std::numeric_limits<uint32_t>::max(),
                   fmt::format("host_filter_threshold must be in [0, {}], got {}",
                               std::numeric_limits<uint32_t>::max(),
                               threshold));
    return static_cast<uint32_t>(threshold);
}

SindiHostBuildPlan::SindiHostBuildPlan(Allocator* allocator)
    : order_(allocator),
      host_ids_(allocator),
      input_offsets_(allocator),
      successful_counts_(allocator) {
}

void
SindiHostBuildPlan::RecordSuccess(uint32_t ordered_position) {
    if (not enabled_) {
        return;
    }
    while (successful_host_cursor_ + 1 < input_offsets_.size() and
           ordered_position >= input_offsets_[successful_host_cursor_ + 1]) {
        ++successful_host_cursor_;
    }
    ++successful_counts_[successful_host_cursor_];
}

SindiHostFilter::SindiHostFilter(uint32_t direct_search_threshold, Allocator* allocator)
    : direct_search_threshold_(direct_search_threshold),
      host_ids_(allocator),
      host_range_offsets_(allocator),
      host_ranges_(allocator) {
}

SindiHostBuildPlan
SindiHostFilter::PrepareBuild(const DatasetPtr& base, uint64_t current_element_count) const {
    SindiHostBuildPlan plan(host_ids_.get_allocator().allocator_);
    const auto* source_host_ids = base->GetUInt32Metadata(SINDI_HOST_ID_METADATA_NAME);
    if (source_host_ids == nullptr) {
        CHECK_ARGUMENT(not this->HasMetadata(), "SINDI host-aware Add requires host_id metadata");
        return plan;
    }

    if (current_element_count != 0) {
        CHECK_ARGUMENT(this->HasMetadata(),
                       "SINDI cannot add host metadata after host-unaware documents");
    }
    const auto data_num = base->GetNumElements();
    CHECK_ARGUMENT(current_element_count + static_cast<uint64_t>(data_num) <=
                       std::numeric_limits<uint32_t>::max(),
                   "SINDI host-filtered build exceeds uint32_t document capacity");

    plan.enabled_ = true;
    plan.order_.resize(static_cast<uint64_t>(data_num));
    std::iota(plan.order_.begin(), plan.order_.end(), 0);
    for (int64_t i = 0; i < data_num; ++i) {
        CHECK_ARGUMENT(source_host_ids[i] != 0, "SINDI host_id must be greater than zero");
    }
    std::sort(
        plan.order_.begin(), plan.order_.end(), [source_host_ids](uint32_t lhs, uint32_t rhs) {
            if (source_host_ids[lhs] != source_host_ids[rhs]) {
                return source_host_ids[lhs] < source_host_ids[rhs];
            }
            return lhs < rhs;
        });

    for (uint32_t position = 0; position < plan.order_.size(); ++position) {
        const auto host_id = source_host_ids[plan.order_[position]];
        if (plan.host_ids_.empty() or plan.host_ids_.back() != host_id) {
            plan.host_ids_.push_back(host_id);
            plan.input_offsets_.push_back(position);
        }
    }
    uint64_t combined_host_count = host_ids_.size();
    for (const auto host_id : plan.host_ids_) {
        if (not std::binary_search(host_ids_.begin(), host_ids_.end(), host_id)) {
            ++combined_host_count;
        }
    }
    CHECK_ARGUMENT(combined_host_count <= SINDI_MAX_HOST_COUNT,
                   fmt::format("SINDI unique host count must not exceed {}", SINDI_MAX_HOST_COUNT));
    plan.input_offsets_.push_back(static_cast<uint32_t>(data_num));
    plan.successful_counts_.resize(plan.host_ids_.size(), 0);
    return plan;
}

void
SindiHostFilter::CommitBuild(SindiHostBuildPlan&& plan,
                             uint32_t first_inner_id,
                             uint32_t end_inner_id) {
    if (not plan.Enabled()) {
        if (first_inner_id == 0) {
            this->Clear();
        }
        return;
    }

    uint32_t next_inner_id = first_inner_id;
    for (uint32_t i = 0; i < plan.successful_counts_.size(); ++i) {
        plan.input_offsets_[i] = next_inner_id;
        next_inner_id += plan.successful_counts_[i];
    }
    CHECK_ARGUMENT(next_inner_id == end_inner_id,
                   "SINDI host metadata count does not match inserted documents");

    Vector<uint32_t> merged_host_ids(host_ids_.get_allocator().allocator_);
    Vector<uint32_t> merged_range_offsets(host_ids_.get_allocator().allocator_);
    Vector<SindiHostRange> merged_ranges(host_ids_.get_allocator().allocator_);
    merged_host_ids.reserve(host_ids_.size() + plan.host_ids_.size());
    merged_range_offsets.reserve(host_ids_.size() + plan.host_ids_.size() + 1);
    merged_ranges.reserve(host_ranges_.size() + plan.host_ids_.size());
    merged_range_offsets.push_back(0);

    uint32_t existing = 0;
    uint32_t added = 0;
    while (existing < host_ids_.size() or added < plan.host_ids_.size()) {
        const bool take_existing =
            added == plan.host_ids_.size() or
            (existing < host_ids_.size() && host_ids_[existing] < plan.host_ids_[added]);
        const bool take_added =
            existing == host_ids_.size() or
            (added < plan.host_ids_.size() && plan.host_ids_[added] < host_ids_[existing]);
        const auto host_id = take_existing ? host_ids_[existing] : plan.host_ids_[added];
        const bool has_existing = not take_added;
        const bool has_added = not take_existing;
        const auto added_count = has_added ? plan.successful_counts_[added] : 0;

        if (has_existing or added_count != 0) {
            merged_host_ids.push_back(host_id);
            if (has_existing) {
                const auto range_begin = host_range_offsets_[existing];
                const auto range_end = host_range_offsets_[existing + 1];
                merged_ranges.insert(merged_ranges.end(),
                                     host_ranges_.begin() + range_begin,
                                     host_ranges_.begin() + range_end);
            }
            if (added_count != 0) {
                const auto begin = plan.input_offsets_[added];
                if (has_existing && merged_ranges.back().end == begin) {
                    merged_ranges.back().end += added_count;
                } else {
                    merged_ranges.push_back({begin, begin + added_count});
                }
            }
            merged_range_offsets.push_back(static_cast<uint32_t>(merged_ranges.size()));
        }
        if (has_existing) {
            ++existing;
        }
        if (has_added) {
            ++added;
        }
    }

    host_ids_ = std::move(merged_host_ids);
    host_range_offsets_ = std::move(merged_range_offsets);
    host_ranges_ = std::move(merged_ranges);
}

void
SindiHostFilter::Clear() {
    host_ids_.clear();
    host_range_offsets_.clear();
    host_ranges_.clear();
}

SindiHostSearchRoute
SindiHostFilter::Classify(const DatasetPtr& query, bool direct_search_available) const {
    const auto* query_host_id = query->GetUInt32Metadata(SINDI_HOST_ID_METADATA_NAME);
    if (host_ids_.empty() or query_host_id == nullptr) {
        return {};
    }
    const auto host = std::lower_bound(host_ids_.begin(), host_ids_.end(), query_host_id[0]);
    if (host == host_ids_.end() or *host != query_host_id[0]) {
        return {SindiHostRouteKind::EMPTY, 0, 0};
    }
    const auto host_index = static_cast<uint32_t>(host - host_ids_.begin());
    const auto range_begin = host_range_offsets_[host_index];
    const auto range_end = host_range_offsets_[host_index + 1];
    uint64_t document_count = 0;
    for (auto i = range_begin; i < range_end; ++i) {
        document_count += host_ranges_[i].end - host_ranges_[i].begin;
    }
    const auto kind = direct_search_available && document_count <= direct_search_threshold_
                          ? SindiHostRouteKind::DIRECT
                          : SindiHostRouteKind::WINDOW;
    return {kind, host_ranges_[range_begin].begin, host_ranges_[range_end - 1].end, host_index};
}

void
SindiHostFilter::ApplyFilter(const SindiHostSearchRoute& route, FilterPtr& filter) const {
    if (route.kind != SindiHostRouteKind::DIRECT && route.kind != SindiHostRouteKind::WINDOW) {
        return;
    }
    filter = std::make_shared<InnerIdHostFilter>(&host_ranges_,
                                                 host_range_offsets_[route.host_index],
                                                 host_range_offsets_[route.host_index + 1],
                                                 std::move(filter));
}

DatasetPtr
SindiHostFilter::SearchDirect(const SindiHostSearchRoute& route,
                              const SparseVector& query,
                              int64_t k,
                              const FilterPtr& filter,
                              const std::optional<float>& distance_threshold,
                              const FlattenInterfacePtr& rerank_flat,
                              const LabelTablePtr& label_table,
                              Allocator* allocator,
                              SearchStatistics* statistics) {
    CHECK_ARGUMENT(route.kind == SindiHostRouteKind::DIRECT,
                   "direct host search requires a direct route");
    CHECK_ARGUMENT(rerank_flat != nullptr, "direct host search requires a rerank data cell");
    Vector<InnerIdType> inner_ids(allocator);
    inner_ids.reserve(route.end - route.begin);
    for (uint32_t inner_id = route.begin; inner_id < route.end; ++inner_id) {
        if (filter == nullptr or filter->CheckValid(inner_id)) {
            inner_ids.push_back(inner_id);
        }
    }
    if (inner_ids.empty()) {
        return collect_results(std::make_shared<StandardHeap<true, false>>(allocator, -1),
                               allocator);
    }

    Vector<float> distances(inner_ids.size(), allocator);
    const auto computer = rerank_flat->FactoryComputer(&query);
    QueryContext context{
        .alloc = allocator, .stats = statistics, .distance_phase = DistanceEvaluationPhase::RERANK};
    rerank_flat->Query(distances.data(),
                       computer,
                       inner_ids.data(),
                       static_cast<InnerIdType>(inner_ids.size()),
                       &context);

    auto results = std::make_shared<StandardHeap<true, false>>(allocator, -1);
    for (uint64_t i = 0; i < inner_ids.size(); ++i) {
        const auto distance = distances[i];
        if (not std::isfinite(distance) or
            (distance_threshold.has_value() and distance > distance_threshold.value())) {
            continue;
        }
        results->Push(distance, label_table->GetLabelById(inner_ids[i]));
        if (results->Size() > static_cast<uint64_t>(k)) {
            results->Pop();
        }
    }
    return collect_results(results, allocator);
}

void
SindiHostFilter::ApplyWindowRoute(const SindiHostSearchRoute& route,
                                  uint32_t window_size,
                                  int64_t& min_window_id,
                                  int64_t& max_window_id) {
    if (route.kind != SindiHostRouteKind::WINDOW) {
        return;
    }
    min_window_id = std::max<int64_t>(min_window_id, route.begin / window_size);
    max_window_id = std::min<int64_t>(max_window_id, (route.end - 1) / window_size);
}

bool
SindiHostFilter::RequiresFullTermScan(const SindiHostSearchRoute& route,
                                      uint32_t window_id,
                                      uint32_t window_size) const {
    if (route.kind != SindiHostRouteKind::WINDOW) {
        return false;
    }
    const auto window_begin = static_cast<uint64_t>(window_id) * window_size;
    const auto window_end = window_begin + window_size;
    const auto range_begin = host_range_offsets_[route.host_index];
    const auto range_end = host_range_offsets_[route.host_index + 1];
    const auto range = std::upper_bound(
        host_ranges_.begin() + range_begin,
        host_ranges_.begin() + range_end,
        window_begin,
        [](uint64_t value, const SindiHostRange& candidate) { return value < candidate.begin; });
    if (range == host_ranges_.begin() + range_begin) {
        return true;
    }
    const auto& candidate = *std::prev(range);
    return candidate.begin > window_begin or candidate.end < window_end;
}

}  // namespace vsag
