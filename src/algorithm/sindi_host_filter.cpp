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
#include <limits>
#include <numeric>

#include "common.h"
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

}  // namespace

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

SindiHostFilter::SindiHostFilter(Allocator* allocator)
    : host_ids_(allocator), host_range_offsets_(allocator), host_ranges_(allocator) {
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
SindiHostFilter::Classify(const DatasetPtr& query) const {
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
    return {SindiHostRouteKind::WINDOW,
            host_ranges_[range_begin].begin,
            host_ranges_[range_end - 1].end,
            host_index};
}

void
SindiHostFilter::ApplyFilter(const SindiHostSearchRoute& route, FilterPtr& filter) const {
    if (route.kind != SindiHostRouteKind::WINDOW) {
        return;
    }
    filter = std::make_shared<InnerIdHostFilter>(&host_ranges_,
                                                 host_range_offsets_[route.host_index],
                                                 host_range_offsets_[route.host_index + 1],
                                                 std::move(filter));
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

void
SindiHostFilter::Serialize(StreamWriter& writer) const {
    StreamWriter::WriteVector(writer, host_ids_);
    StreamWriter::WriteVector(writer, host_range_offsets_);
    const uint64_t range_count = host_ranges_.size();
    StreamWriter::WriteObj(writer, range_count);
    for (const auto& range : host_ranges_) {
        StreamWriter::WriteObj(writer, range.begin);
        StreamWriter::WriteObj(writer, range.end);
    }
}

void
SindiHostFilter::Deserialize(StreamReader& reader, uint64_t element_count) {
    CHECK_ARGUMENT(  // NOLINT(readability-simplify-boolean-expr)
        element_count > 0 && element_count <= std::numeric_limits<uint32_t>::max(),
        "serialized SINDI host metadata element count is invalid");

    auto* allocator = host_ids_.get_allocator().allocator_;
    Vector<uint32_t> host_ids(allocator);
    Vector<uint32_t> range_offsets(allocator);
    Vector<SindiHostRange> ranges(allocator);

    uint64_t host_count = 0;
    StreamReader::ReadObj(reader, host_count);
    CHECK_ARGUMENT(  // NOLINT(readability-simplify-boolean-expr)
        host_count > 0 && host_count <= SINDI_MAX_HOST_COUNT && host_count <= element_count,
        "serialized SINDI host count is invalid");
    host_ids.resize(host_count);
    reader.Read(reinterpret_cast<char*>(host_ids.data()), host_count * sizeof(uint32_t));
    CHECK_ARGUMENT(
        std::adjacent_find(host_ids.begin(),
                           host_ids.end(),
                           [](uint32_t lhs, uint32_t rhs) { return lhs >= rhs; }) == host_ids.end(),
        "serialized SINDI host IDs must be strictly ordered");

    uint64_t offset_count = 0;
    StreamReader::ReadObj(reader, offset_count);
    CHECK_ARGUMENT(offset_count == host_count + 1,
                   "serialized SINDI host range offset count is invalid");
    range_offsets.resize(offset_count);
    reader.Read(reinterpret_cast<char*>(range_offsets.data()), offset_count * sizeof(uint32_t));

    uint64_t range_count = 0;
    StreamReader::ReadObj(reader, range_count);
    CHECK_ARGUMENT(  // NOLINT(readability-simplify-boolean-expr)
        range_count >= host_count && range_count <= element_count,
        "serialized SINDI host range count is invalid");
    CHECK_ARGUMENT(  // NOLINT(readability-simplify-boolean-expr)
        range_offsets.front() == 0 && range_offsets.back() == range_count,
        "serialized SINDI host range offsets are invalid");
    CHECK_ARGUMENT(std::adjacent_find(range_offsets.begin(),
                                      range_offsets.end(),
                                      [](uint32_t lhs, uint32_t rhs) { return lhs >= rhs; }) ==
                       range_offsets.end(),
                   "serialized SINDI host range offsets must be strictly ordered");

    ranges.resize(range_count);
    for (auto& range : ranges) {
        StreamReader::ReadObj(reader, range.begin);
        StreamReader::ReadObj(reader, range.end);
        CHECK_ARGUMENT(  // NOLINT(readability-simplify-boolean-expr)
            range.begin < range.end && range.end <= element_count,
            "serialized SINDI host range is invalid");
    }
    for (uint64_t host = 0; host < host_count; ++host) {
        const auto begin = range_offsets[host];
        const auto end = range_offsets[host + 1];
        for (uint32_t range = begin + 1; range < end; ++range) {
            CHECK_ARGUMENT(ranges[range - 1].end < ranges[range].begin,
                           "serialized SINDI ranges for one host must be ordered and disjoint");
        }
    }

    Vector<SindiHostRange> ranges_by_inner_id(ranges, allocator);
    std::sort(
        ranges_by_inner_id.begin(),
        ranges_by_inner_id.end(),
        [](const SindiHostRange& lhs, const SindiHostRange& rhs) { return lhs.begin < rhs.begin; });
    uint32_t next_inner_id = 0;
    for (const auto& range : ranges_by_inner_id) {
        CHECK_ARGUMENT(range.begin == next_inner_id,
                       "serialized SINDI host ranges must cover every document exactly once");
        next_inner_id = range.end;
    }
    CHECK_ARGUMENT(next_inner_id == element_count,
                   "serialized SINDI host ranges must cover every document exactly once");

    host_ids_ = std::move(host_ids);
    host_range_offsets_ = std::move(range_offsets);
    host_ranges_ = std::move(ranges);
}

}  // namespace vsag
